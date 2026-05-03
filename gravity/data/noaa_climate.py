"""
NOAA Climate Data Online (CDO) loader.

Fetches county-level climate normals and recent weather summaries
from the NCEI Climate Data Online API v2.

Retail relevance:
  - Temperature/precipitation normals → seasonal traffic patterns
  - Extreme weather days → footfall disruption risk
  - Climate comfort index → outdoor retail viability

API docs: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
Token:    Free from https://www.ncei.noaa.gov/cdo-web/token
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

_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
_REQUEST_TIMEOUT = 30
_RATE_LIMIT = 0.25  # seconds between requests (CDO allows 5/sec)

# Dataset IDs
_NORMALS_MONTHLY = "NORMAL_MLY"  # 30-year climate normals, monthly
_GHCND = "GHCND"                 # Global Historical Climatology Network Daily

# Data type IDs for monthly normals
_NORMAL_DATATYPES = [
    "MLY-TAVG-NORMAL",   # average temperature (°F × 10)
    "MLY-TMAX-NORMAL",   # average daily max temperature
    "MLY-TMIN-NORMAL",   # average daily min temperature
    "MLY-PRCP-NORMAL",   # average precipitation (inches × 100)
    "MLY-SNOW-NORMAL",   # average snowfall
]

# Data type IDs for recent daily summaries
_DAILY_DATATYPES = [
    "TAVG",  # average temperature
    "TMAX",  # max temperature
    "TMIN",  # min temperature
    "PRCP",  # precipitation
    "SNOW",  # snowfall
]

# FIPS → NOAA location ID format
_FIPS_PREFIX = "FIPS:"


class NOAAClimateLoader:
    """Fetch climate data from NOAA Climate Data Online API."""

    def __init__(self, token: Optional[str] = None):
        self.token = token
        self._session = requests.Session()
        if token:
            self._session.headers["token"] = token
        self._cache: dict = {}
        self._last_request = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < _RATE_LIMIT:
            time.sleep(_RATE_LIMIT - elapsed)
        self._last_request = time.time()

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Issue GET request with token header, rate limiting, and caching."""
        if not self.token:
            return {}

        cache_key = f"{endpoint}:{params}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._throttle()
        try:
            url = f"{_BASE_URL}/{endpoint}"
            resp = self._session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning("NOAA CDO %s returned %s", endpoint, resp.status_code)
                return {}
            data = resp.json()
            self._cache[cache_key] = data
            return data
        except Exception as exc:
            logger.warning("NOAA CDO request failed: %s", exc)
            return {}

    @staticmethod
    def _fips_location(state_fips: str, county_fips: str) -> str:
        """Convert state+county FIPS to NOAA FIPS location ID."""
        return f"{_FIPS_PREFIX}{state_fips}{county_fips}"

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_climate_normals(self, state_fips: str, county_fips: str) -> dict:
        """Fetch 30-year monthly climate normals for a county.

        Returns dict with keys:
            avg_temp_f: annual average temperature (°F)
            avg_high_f: annual average daily high
            avg_low_f: annual average daily low
            annual_precip_in: total annual precipitation (inches)
            annual_snow_in: total annual snowfall (inches)
            monthly_temps: list of 12 monthly average temps
            monthly_precip: list of 12 monthly precipitation totals
            hot_months: count of months with avg high > 85°F
            cold_months: count of months with avg low < 32°F
        """
        if not self.token:
            return {}

        location_id = self._fips_location(state_fips, county_fips)

        params = {
            "datasetid": _NORMALS_MONTHLY,
            "locationid": location_id,
            "datatypeid": ",".join(_NORMAL_DATATYPES),
            "limit": 100,
            "units": "standard",
        }

        data = self._get("data", params)
        if not data or "results" not in data:
            return {}

        # Parse results into structured output
        monthly_tavg: dict[int, float] = {}
        monthly_tmax: dict[int, float] = {}
        monthly_tmin: dict[int, float] = {}
        monthly_prcp: dict[int, float] = {}
        monthly_snow: dict[int, float] = {}

        for rec in data.get("results", []):
            dt = rec.get("datatype", "")
            date_str = rec.get("date", "")
            value = rec.get("value")
            if value is None:
                continue

            # Extract month from date (format: YYYY-MM-DDT00:00:00)
            try:
                month = int(date_str[5:7])
            except (ValueError, IndexError):
                continue

            if dt == "MLY-TAVG-NORMAL":
                monthly_tavg[month] = value / 10.0  # tenths of °F
            elif dt == "MLY-TMAX-NORMAL":
                monthly_tmax[month] = value / 10.0
            elif dt == "MLY-TMIN-NORMAL":
                monthly_tmin[month] = value / 10.0
            elif dt == "MLY-PRCP-NORMAL":
                monthly_prcp[month] = value / 100.0  # hundredths of inch
            elif dt == "MLY-SNOW-NORMAL":
                monthly_snow[month] = value / 10.0  # tenths of inch

        if not monthly_tavg:
            return {}

        temps_list = [monthly_tavg.get(m, 0) for m in range(1, 13)]
        highs_list = [monthly_tmax.get(m, 0) for m in range(1, 13)]
        lows_list = [monthly_tmin.get(m, 0) for m in range(1, 13)]
        precip_list = [monthly_prcp.get(m, 0) for m in range(1, 13)]
        snow_list = [monthly_snow.get(m, 0) for m in range(1, 13)]

        result = {
            "avg_temp_f": round(sum(temps_list) / max(len(temps_list), 1), 1),
            "avg_high_f": round(sum(highs_list) / max(len(highs_list), 1), 1),
            "avg_low_f": round(sum(lows_list) / max(len(lows_list), 1), 1),
            "annual_precip_in": round(sum(precip_list), 1),
            "annual_snow_in": round(sum(snow_list), 1),
            "monthly_temps": [round(t, 1) for t in temps_list],
            "monthly_precip": [round(p, 2) for p in precip_list],
            "hot_months": sum(1 for h in highs_list if h > 85),
            "cold_months": sum(1 for lo in lows_list if lo < 32),
        }

        return result

    def get_climate_context(self, state_fips: str, county_fips: str) -> dict:
        """High-level climate context for retail analysis.

        Returns dict with:
            normals: full normals dict (see get_climate_normals)
            climate_type: "hot" / "cold" / "temperate" / "extreme"
            seasonal_risk: "low" / "moderate" / "high"
            outdoor_retail_viability: "high" / "moderate" / "low"
            summary: one-line narrative
        """
        normals = self.get_climate_normals(state_fips, county_fips)
        if not normals:
            return {}

        avg_high = normals.get("avg_high_f", 70)
        avg_low = normals.get("avg_low_f", 40)
        hot = normals.get("hot_months", 0)
        cold = normals.get("cold_months", 0)
        snow = normals.get("annual_snow_in", 0)
        precip = normals.get("annual_precip_in", 0)

        # Classify climate type
        if hot >= 5 and cold == 0:
            climate_type = "hot"
        elif cold >= 4 and hot == 0:
            climate_type = "cold"
        elif hot >= 3 and cold >= 2:
            climate_type = "extreme"
        else:
            climate_type = "temperate"

        # Seasonal disruption risk
        if snow > 40 or cold >= 5 or hot >= 7:
            seasonal_risk = "high"
        elif snow > 15 or cold >= 3 or hot >= 5:
            seasonal_risk = "moderate"
        else:
            seasonal_risk = "low"

        # Outdoor retail viability
        comfortable_months = 12 - hot - cold
        if comfortable_months >= 8:
            outdoor_viability = "high"
        elif comfortable_months >= 5:
            outdoor_viability = "moderate"
        else:
            outdoor_viability = "low"

        summary = (
            f"Avg high {avg_high:.0f}°F, low {avg_low:.0f}°F. "
            f"{precip:.0f}\" rain, {snow:.0f}\" snow/yr. "
            f"{hot} hot months, {cold} cold months. "
            f"Climate: {climate_type}."
        )

        return {
            "normals": normals,
            "climate_type": climate_type,
            "seasonal_risk": seasonal_risk,
            "outdoor_retail_viability": outdoor_viability,
            "summary": summary,
        }

    def clear_cache(self) -> None:
        self._cache.clear()
