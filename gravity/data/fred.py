"""
FRED (Federal Reserve Economic Data) loader.

Fetches macro economic indicators relevant to retail market analysis.
Requires a free API key from https://fred.stlouisfed.org/docs/api/api_key.html

API documentation:
    https://api.stlouisfed.org/fred/series/observations

Endpoint pattern:
    GET /fred/series/observations?series_id={id}&api_key={key}&file_type=json
        &sort_order=desc&limit={count}
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

# FRED series IDs mapped to friendly names
_SERIES = {
    "cpi": "CPIAUCSL",                    # Consumer Price Index
    "unemployment": "UNRATE",              # National unemployment rate
    "consumer_confidence": "UMCSENT",      # U of Michigan consumer sentiment
    "retail_sales": "RSAFS",               # Retail sales total
    "personal_income": "PI",               # Personal income
    "housing_starts": "HOUST",             # Housing starts
    "consumer_spending": "PCE",            # Personal consumption expenditures
    "food_cpi": "CPIUFDNS",              # CPI for food
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FREDLoader:
    """Load macroeconomic indicator data from the FRED API.

    Fetches time-series observations for key economic indicators and
    bundles them into retail-relevant context dictionaries.

    Parameters
    ----------
    api_key : str or None
        FRED API key. If None, all requests will return empty results.
    timeout : int
        HTTP request timeout in seconds.
    rate_limit : float
        Minimum seconds between consecutive API requests.

    Examples
    --------
    >>> loader = FREDLoader(api_key="your_key_here")
    >>> indicators = loader.get_macro_indicators()
    >>> print(indicators["cpi"])
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

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
        # In-memory cache: keyed by (series_id, count)
        self._cache: dict[tuple[str, int], list[dict]] = {}

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
            v = float(val)
            return v
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_series_latest(self, series_id: str, count: int = 1) -> list[dict]:
        """Fetch the latest N observations for a FRED series.

        Parameters
        ----------
        series_id : str
            FRED series identifier (e.g. "CPIAUCSL").
        count : int
            Number of most-recent observations to return.

        Returns
        -------
        list[dict]
            Each dict has keys ``date`` (str, YYYY-MM-DD) and
            ``value`` (float). Returns empty list on any failure.
        """
        if not self.api_key:
            logger.warning("FRED API key not set. Returning empty results.")
            return []

        cache_key = (series_id, count)
        if cache_key in self._cache:
            logger.debug("FRED cache hit for %s (count=%d)", series_id, count)
            return self._cache[cache_key]

        url = f"{self.BASE_URL}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": count,
        }

        logger.info("Fetching FRED series %s (latest %d)", series_id, count)
        self._throttle()
        self._last_request_time = time.monotonic()

        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "FRED API request failed for series %s: %s", series_id, exc
            )
            return []

        try:
            data = resp.json()
        except ValueError as exc:
            logger.warning("Failed to parse FRED JSON for %s: %s", series_id, exc)
            return []

        observations = data.get("observations", [])
        results = []
        for obs in observations:
            value = self._safe_float(obs.get("value", "."), default=None)
            if value is None:
                # FRED uses "." for missing values
                continue
            results.append({
                "date": obs.get("date", ""),
                "value": value,
            })

        self._cache[cache_key] = results
        logger.info(
            "FRED series %s loaded: %d observations", series_id, len(results)
        )
        return results

    def get_macro_indicators(self) -> dict:
        """Fetch the latest value for all key macro indicators.

        Returns
        -------
        dict
            Keyed by friendly name (e.g. ``cpi``, ``unemployment``).
            Each value is a dict with ``value`` (float) and ``date`` (str).
            Returns empty dict if the API key is missing or all fetches fail.
        """
        if not self.api_key:
            logger.warning("FRED API key not set. Returning empty indicators.")
            return {}

        indicators: dict = {}
        for name, series_id in _SERIES.items():
            obs = self.get_series_latest(series_id, count=1)
            if obs:
                indicators[name] = {
                    "value": obs[0]["value"],
                    "date": obs[0]["date"],
                }
            else:
                logger.warning(
                    "FRED: no data returned for %s (%s)", name, series_id
                )

        return indicators

    def get_retail_context(
        self,
        state_fips: Optional[str] = None,
        county_fips: Optional[str] = None,
    ) -> dict:
        """Bundle macro indicators relevant to retail site analysis.

        Fetches all macro indicators and computes derived signals:

        - ``cpi_trend``: ``"rising"`` / ``"stable"`` / ``"falling"`` based
          on a 3-month CPI comparison.
        - ``consumer_health``: ``"strong"`` / ``"moderate"`` / ``"weak"``
          based on unemployment, consumer confidence, and spending growth.

        Parameters
        ----------
        state_fips : str or None
            Not currently used for FRED queries (national data only).
            Reserved for future regional overlay.
        county_fips : str or None
            Not currently used. Reserved for future regional overlay.

        Returns
        -------
        dict
            Combined macro context dict, or empty dict on failure.
        """
        if not self.api_key:
            logger.warning("FRED API key not set. Returning empty context.")
            return {}

        context: dict = {}

        # Fetch current indicators
        indicators = self.get_macro_indicators()
        if not indicators:
            return {}

        context["indicators"] = indicators

        # ----- CPI trend (3-month comparison) -----
        cpi_obs = self.get_series_latest("CPIAUCSL", count=4)
        cpi_trend = "stable"
        if len(cpi_obs) >= 4:
            recent = cpi_obs[0]["value"]
            three_months_ago = cpi_obs[3]["value"]
            if three_months_ago > 0:
                pct_change = (recent - three_months_ago) / three_months_ago * 100
                if pct_change > 0.5:
                    cpi_trend = "rising"
                elif pct_change < -0.2:
                    cpi_trend = "falling"
        context["cpi_trend"] = cpi_trend

        # ----- Consumer health composite -----
        health_score = 0
        health_signals = 0

        # Unemployment: below 4% is strong, above 6% is weak
        unemp = indicators.get("unemployment", {}).get("value")
        if unemp is not None:
            health_signals += 1
            if unemp < 4.0:
                health_score += 2
            elif unemp < 6.0:
                health_score += 1

        # Consumer confidence: above 80 is strong, below 60 is weak
        conf = indicators.get("consumer_confidence", {}).get("value")
        if conf is not None:
            health_signals += 1
            if conf > 80:
                health_score += 2
            elif conf > 60:
                health_score += 1

        # Consumer spending growth (3-month)
        pce_obs = self.get_series_latest("PCE", count=4)
        if len(pce_obs) >= 4:
            health_signals += 1
            recent_pce = pce_obs[0]["value"]
            old_pce = pce_obs[3]["value"]
            if old_pce > 0:
                pce_growth = (recent_pce - old_pce) / old_pce * 100
                if pce_growth > 1.0:
                    health_score += 2
                elif pce_growth > 0.0:
                    health_score += 1

        if health_signals > 0:
            avg_score = health_score / health_signals
            if avg_score >= 1.5:
                consumer_health = "strong"
            elif avg_score >= 0.75:
                consumer_health = "moderate"
            else:
                consumer_health = "weak"
        else:
            consumer_health = "moderate"

        context["consumer_health"] = consumer_health

        logger.info(
            "FRED retail context: cpi_trend=%s, consumer_health=%s",
            cpi_trend, consumer_health,
        )
        return context

    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self._cache.clear()
        logger.debug("FRED data cache cleared.")

    def __repr__(self) -> str:
        has_key = "set" if self.api_key else "unset"
        return (
            f"FREDLoader(api_key={has_key}, "
            f"cached_series={len(self._cache)})"
        )
