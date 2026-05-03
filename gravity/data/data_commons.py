"""
Google Data Commons loader.

Fetches aggregated demographic, housing, economic, and health indicators
from the Data Commons API. Supplements Census ACS with cross-source data.

API documentation:
    https://docs.datacommons.org/api/rest/v2/

Endpoint:
    GET https://api.datacommons.org/v2/observation
        ?key={api_key}
        &entity.dcids={dcid}
        &variable.dcids={stat_var}
        &date=LATEST
        &select=entity&select=variable&select=value&select=date
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUEST_TIMEOUT = 30  # seconds
_RATE_LIMIT = 0.5  # minimum seconds between requests

# Statistical variable DCIDs mapped to friendly names
_STAT_VARS = {
    "median_home_value": "Median_Value_OccupiedHousingUnit",
    "median_rent": "Median_Rent_OccupiedHousingUnit",
    "unemployment_rate": "UnemploymentRate_Person",
    "bachelors_or_higher_pct": "Percent_Person_25OrMoreYears_BachelorsDegreeOrHigher",
    "health_insurance_pct": "Percent_Person_WithHealthInsurance",
    "commute_time_minutes": "Median_Duration_Commuting_Person",
    "housing_units": "Count_HousingUnit",
    "population_density": "Count_Person_PerArea",
    "gini_index": "GiniIndex_EconomicActivity",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataCommonsLoader:
    """Load demographic and economic indicators from Google Data Commons.

    Fetches statistical variables for geographic entities identified by
    Data Commons IDs (DCIDs). Designed to supplement Census ACS data with
    additional cross-source indicators.

    Parameters
    ----------
    api_key : str or None
        Data Commons API key. If None, requests may be rate-limited or
        rejected by the API.
    timeout : int
        HTTP request timeout in seconds.
    rate_limit : float
        Minimum seconds between consecutive API requests.

    Examples
    --------
    >>> loader = DataCommonsLoader(api_key="your_key_here")
    >>> stats = loader.get_county_context("48", "453")
    >>> print(stats["median_home_value"])
    """

    BASE_URL = "https://api.datacommons.org"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = _REQUEST_TIMEOUT,
        rate_limit: float = _RATE_LIMIT,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._rate_limit = rate_limit
        self._session = requests.Session()
        self._last_request_time: float = 0.0
        # In-memory cache: keyed by (dcid, tuple_of_stat_vars)
        self._cache: dict[tuple[str, tuple[str, ...]], dict] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep if necessary to respect the rate limit."""
        if self._rate_limit <= 0:
            return
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

    def _safe_float(self, val, default: float = 0.0) -> float:
        """Safely convert a value to float."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _build_dcid(self, state_fips: str, county_fips: str) -> str:
        """Build a Data Commons place DCID from FIPS codes.

        Parameters
        ----------
        state_fips : str
            2-digit state FIPS (e.g. "48").
        county_fips : str
            3-digit county FIPS (e.g. "453").

        Returns
        -------
        str
            DCID like "geoId/48453".
        """
        return f"geoId/{state_fips.zfill(2)}{county_fips.zfill(3)}"

    def _fetch_observation(self, dcid: str, stat_var: str) -> Optional[dict]:
        """Fetch a single statistical variable for a place.

        Uses the V2 observation endpoint with the LATEST date selector.

        Returns
        -------
        dict or None
            Dict with ``value`` (float) and ``date`` (str), or None on failure.
        """
        url = f"{self.BASE_URL}/v2/observation"
        params = {
            "entity.dcids": dcid,
            "variable.dcids": stat_var,
            "date": "LATEST",
            "select": ["entity", "variable", "value", "date"],
        }
        if self.api_key:
            params["key"] = self.api_key

        self._throttle()
        self._last_request_time = time.monotonic()

        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Data Commons request failed for %s/%s: %s",
                dcid, stat_var, exc,
            )
            return None

        try:
            data = resp.json()
        except ValueError as exc:
            logger.warning(
                "Failed to parse Data Commons JSON for %s/%s: %s",
                dcid, stat_var, exc,
            )
            return None

        # V2 response structure:
        # { "byVariable": { "<stat_var>": { "byEntity": { "<dcid>": {
        #       "orderedFacets": [ { "observations": [ {"date": ..., "value": ...} ] } ]
        # } } } } }
        try:
            by_var = data.get("byVariable", {})
            by_entity = by_var.get(stat_var, {}).get("byEntity", {})
            entity_data = by_entity.get(dcid, {})
            facets = entity_data.get("orderedFacets", [])
            if not facets:
                return None
            observations = facets[0].get("observations", [])
            if not observations:
                return None
            obs = observations[0]
            value = self._safe_float(obs.get("value"), default=None)
            if value is None:
                return None
            return {
                "value": value,
                "date": obs.get("date", ""),
            }
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning(
                "Unexpected Data Commons response structure for %s/%s: %s",
                dcid, stat_var, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_place_stats(
        self,
        dcid: str,
        stat_vars: Optional[list[str]] = None,
    ) -> dict:
        """Fetch statistical variables for a Data Commons place.

        Parameters
        ----------
        dcid : str
            Data Commons place ID (e.g. ``"geoId/48453"`` for a county,
            ``"geoId/4845300"`` for a place/city).
        stat_vars : list[str] or None
            List of stat var DCIDs to fetch. If None, fetches all
            variables defined in ``_STAT_VARS``.

        Returns
        -------
        dict
            Keyed by friendly name. Each value is a float (the latest
            observation value). Returns empty dict on total failure.
        """
        if stat_vars is None:
            var_map = _STAT_VARS
        else:
            # Build a map from dcid to dcid (no friendly name remap)
            var_map = {sv: sv for sv in stat_vars}

        cache_key = (dcid, tuple(sorted(var_map.values())))
        if cache_key in self._cache:
            logger.debug("Data Commons cache hit for %s", dcid)
            return self._cache[cache_key]

        logger.info(
            "Fetching Data Commons stats for %s (%d variables)",
            dcid, len(var_map),
        )

        results: dict = {}
        for friendly_name, stat_var_dcid in var_map.items():
            obs = self._fetch_observation(dcid, stat_var_dcid)
            if obs is not None:
                results[friendly_name] = obs["value"]
            else:
                logger.warning(
                    "Data Commons: no data for %s at %s",
                    stat_var_dcid, dcid,
                )

        self._cache[cache_key] = results
        logger.info(
            "Data Commons loaded %d/%d variables for %s",
            len(results), len(var_map), dcid,
        )
        return results

    def get_county_context(
        self,
        state_fips: str,
        county_fips: str,
    ) -> dict:
        """Fetch all default statistical variables for a county.

        Converts state and county FIPS codes to a Data Commons DCID
        and fetches all indicators defined in ``_STAT_VARS``.

        Parameters
        ----------
        state_fips : str
            2-digit state FIPS code (e.g. ``"48"`` for Texas).
        county_fips : str
            3-digit county FIPS code (e.g. ``"453"`` for Travis County).

        Returns
        -------
        dict
            Friendly-named dict of indicator values (floats), or empty
            dict if the place is not found or all fetches fail.
        """
        dcid = self._build_dcid(state_fips, county_fips)
        return self.get_place_stats(dcid)

    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self._cache.clear()
        logger.debug("Data Commons cache cleared.")

    def __repr__(self) -> str:
        has_key = "set" if self.api_key else "unset"
        return (
            f"DataCommonsLoader(api_key={has_key}, "
            f"cached_places={len(self._cache)})"
        )
