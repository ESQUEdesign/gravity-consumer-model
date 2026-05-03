"""HealthData.gov / CDC Socrata data loader.

Fetches community health rankings, food environment data, and related
health metrics from CDC/HHS Socrata SODA API endpoints.  No API key
required, but an optional app_token increases rate limits.

API documentation:
    https://dev.socrata.com/foundry/data.cdc.gov/

Endpoint pattern:
    https://data.cdc.gov/resource/{dataset_id}.json?$where=...
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

_REQUEST_TIMEOUT = 30
_RATE_LIMIT = 0.25  # seconds between requests (unauthenticated: ~4/sec safe)

_DATASETS: dict[str, str] = {
    "community_health": "swc5-untb",   # County Health Rankings
    "food_environment": "k4e7-k3vi",   # CDC food environment
}

_SODA_BASE = "https://data.cdc.gov/resource"

# Mapping of Socrata field names to our normalized output keys.
# County Health Rankings dataset uses various column names across years;
# we try multiple candidates for each metric.
_HEALTH_FIELD_MAP: dict[str, list[str]] = {
    "obesity_pct": [
        "adult_obesity_raw_value",
        "v011_rawvalue",
        "adult_obesity",
        "obesity_percentage",
    ],
    "physical_inactivity_pct": [
        "physical_inactivity_raw_value",
        "v070_rawvalue",
        "physical_inactivity",
    ],
    "food_environment_index": [
        "food_environment_index_raw_value",
        "v133_rawvalue",
        "food_environment_index",
    ],
    "premature_death_rate": [
        "premature_death_raw_value",
        "v001_rawvalue",
        "premature_death",
        "years_of_potential_life_lost_rate",
    ],
    "uninsured_pct": [
        "uninsured_raw_value",
        "v085_rawvalue",
        "uninsured",
        "uninsured_adults_raw_value",
    ],
    "median_household_income": [
        "median_household_income_raw_value",
        "v063_rawvalue",
        "median_household_income",
    ],
}

# Food environment dataset field candidates
_FOOD_ENV_FIELD_MAP: dict[str, list[str]] = {
    "food_environment_index": [
        "food_environment_index",
        "foodenvironmentindex",
        "value",
    ],
    "low_access_pct": [
        "pct_low_access",
        "lalowi1share",
        "low_access",
    ],
    "snap_participation_rate": [
        "snap_participation_rate",
        "snappctpov",
        "snap_rate",
    ],
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HealthDataLoader:
    """Load county-level health metrics from CDC/HHS Socrata APIs.

    Provides community health rankings (obesity, physical inactivity,
    premature death, etc.) and food environment data at the county level.

    Parameters
    ----------
    app_token : str or None
        Optional Socrata app token.  Not required, but providing one
        increases the rate limit from ~1,000 to ~40,000 requests/hour.
    timeout : int
        HTTP request timeout in seconds.

    Examples
    --------
    >>> loader = HealthDataLoader()
    >>> health = loader.get_county_health_rankings("48", "453")
    >>> print(health.get("obesity_pct"))
    """

    def __init__(
        self,
        app_token: Optional[str] = None,
        timeout: int = _REQUEST_TIMEOUT,
    ) -> None:
        self.app_token = app_token
        self.timeout = timeout
        self._session = requests.Session()
        if app_token:
            self._session.headers["X-App-Token"] = app_token
        self._last_request_time: float = 0.0
        self._cache: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _RATE_LIMIT:
            time.sleep(_RATE_LIMIT - elapsed)

    def _build_fips(self, state_fips: str, county_fips: str) -> str:
        """Combine state and county FIPS into a 5-digit code."""
        return state_fips.zfill(2) + county_fips.zfill(3)

    def _query_dataset(
        self,
        dataset_id: str,
        state_fips: str,
        county_fips: str,
        limit: int = 5,
    ) -> list[dict]:
        """Query a Socrata dataset by FIPS code.

        Tries multiple common FIPS column names to accommodate schema
        differences across datasets.

        Returns a list of row dicts, or empty list on failure.
        """
        fips = self._build_fips(state_fips, county_fips)
        cache_key = f"{dataset_id}:{fips}"

        if cache_key in self._cache:
            logger.debug("HealthData cache hit: %s", cache_key)
            return self._cache[cache_key]

        # Different datasets use different FIPS column names; try several
        fips_queries = [
            f"fipscode='{fips}'",
            f"fips='{fips}'",
            f"countyfips='{fips}'",
            f"county_fips='{fips}'",
            f"fipscode={fips}",
        ]

        # Also try with separate state/county columns
        st = state_fips.zfill(2)
        ct = county_fips.zfill(3)
        separate_queries = [
            f"statecode='{st}' AND countycode='{ct}'",
            f"state_fips_code='{st}' AND county_fips_code='{ct}'",
            f"statefips='{st}' AND countyfips='{ct}'",
        ]

        all_queries = fips_queries + separate_queries

        for where_clause in all_queries:
            url = (
                f"{_SODA_BASE}/{dataset_id}.json"
                f"?$where={where_clause}"
                f"&$limit={limit}"
                f"&$order=:id DESC"
            )

            logger.debug("HealthData query: %s", url)
            self._throttle()
            self._last_request_time = time.monotonic()

            try:
                resp = self._session.get(url, timeout=self.timeout)
                if resp.status_code == 400:
                    # Bad query (column not found) -- try next variant
                    continue
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "HealthData request failed (%s, query=%s): %s",
                    dataset_id, where_clause, exc,
                )
                continue
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "HealthData JSON parse error (%s): %s", dataset_id, exc
                )
                continue

            if isinstance(data, list) and len(data) > 0:
                self._cache[cache_key] = data
                logger.info(
                    "HealthData loaded %d rows from %s for FIPS %s",
                    len(data), dataset_id, fips,
                )
                return data

        logger.warning(
            "HealthData: no results from dataset %s for state=%s county=%s",
            dataset_id, st, ct,
        )
        return []

    @staticmethod
    def _extract_field(
        row: dict,
        candidates: list[str],
        as_float: bool = True,
    ) -> Optional[float]:
        """Try multiple column names and return the first non-null value."""
        for field in candidates:
            val = row.get(field)
            if val is None:
                continue
            try:
                return float(val) if as_float else val
            except (TypeError, ValueError):
                continue
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_county_health_rankings(
        self,
        state_fips: str,
        county_fips: str,
    ) -> dict:
        """Get County Health Rankings metrics for a county.

        Parameters
        ----------
        state_fips : str
            2-digit state FIPS code (e.g. "48" for Texas).
        county_fips : str
            3-digit county FIPS code (e.g. "453" for Travis County).

        Returns
        -------
        dict
            Keys: obesity_pct, physical_inactivity_pct,
            food_environment_index, premature_death_rate,
            uninsured_pct, median_household_income.
            Values are floats or None.  Returns empty dict on failure.
        """
        dataset_id = _DATASETS.get("community_health", "")
        rows = self._query_dataset(dataset_id, state_fips, county_fips)
        if not rows:
            return {}

        # Use the most recent row (first, since we order DESC)
        row = rows[0]

        result: dict = {}
        for output_key, candidates in _HEALTH_FIELD_MAP.items():
            val = self._extract_field(row, candidates)
            result[output_key] = val

        # Log a summary of what we found
        found = sum(1 for v in result.values() if v is not None)
        logger.info(
            "County health rankings for %s%s: %d/%d metrics populated",
            state_fips.zfill(2), county_fips.zfill(3),
            found, len(result),
        )
        return result

    def get_food_environment(
        self,
        state_fips: str,
        county_fips: str,
    ) -> dict:
        """Get food environment metrics for a county.

        Returns
        -------
        dict
            Keys: food_environment_index, low_access_pct,
            snap_participation_rate.  Returns empty dict on failure.
        """
        dataset_id = _DATASETS.get("food_environment", "")
        rows = self._query_dataset(dataset_id, state_fips, county_fips)
        if not rows:
            return {}

        row = rows[0]
        result: dict = {}
        for output_key, candidates in _FOOD_ENV_FIELD_MAP.items():
            val = self._extract_field(row, candidates)
            result[output_key] = val

        found = sum(1 for v in result.values() if v is not None)
        logger.info(
            "Food environment data for %s%s: %d/%d metrics populated",
            state_fips.zfill(2), county_fips.zfill(3),
            found, len(result),
        )
        return result

    def get_health_context(
        self,
        state_fips: str,
        county_fips: str,
    ) -> dict:
        """Bundle all available health metrics for a county.

        Combines community health rankings and food environment data
        into a single dict.  Gracefully handles partial availability --
        if one dataset fails the other is still returned.

        Parameters
        ----------
        state_fips : str
            2-digit state FIPS code.
        county_fips : str
            3-digit county FIPS code.

        Returns
        -------
        dict
            Merged dict of all health metrics.  Returns empty dict only
            if both sources fail.
        """
        health = self.get_county_health_rankings(state_fips, county_fips)
        food = self.get_food_environment(state_fips, county_fips)

        if not health and not food:
            logger.warning(
                "HealthData: no data from any source for state=%s county=%s",
                state_fips.zfill(2), county_fips.zfill(3),
            )
            return {}

        combined: dict = {}
        combined.update(health)
        # Food environment may overlap on food_environment_index; prefer
        # the health rankings version if already set, otherwise fill in.
        for key, val in food.items():
            if key not in combined or combined[key] is None:
                combined[key] = val

        logger.info(
            "HealthData context for %s%s: %d total metrics",
            state_fips.zfill(2), county_fips.zfill(3), len(combined),
        )
        return combined

    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self._cache.clear()
        logger.debug("HealthData cache cleared.")

    def __repr__(self) -> str:
        return (
            f"HealthDataLoader(app_token={'set' if self.app_token else 'none'}, "
            f"cached_queries={len(self._cache)})"
        )
