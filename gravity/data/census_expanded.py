"""Census County Business Patterns (CBP) loader.

Fetches establishment counts, employment, and payroll by NAICS sector for a
county from the Census CBP API.  No API key required.

API documentation:
    https://www.census.gov/data/developers/data-sets/cbp-nonemp-zbp.html

Endpoint pattern:
    https://api.census.gov/data/{year}/cbp?get=ESTAB,EMP,PAYANN
        &for=county:{county_fips}&in=state:{state_fips}&NAICS2017={code}
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

_BASE_URL = "https://api.census.gov/data"
_DEFAULT_YEAR = 2021  # CBP latest available year
_REQUEST_TIMEOUT = 30
_RATE_LIMIT = 0.25  # seconds between requests

_VARIABLES = "ESTAB,EMP,PAYANN"

# Key NAICS 2-digit sectors to query for breakdown
_NAICS_SECTORS = {
    "00": "Total",
    "11": "Agriculture, Forestry, Fishing",
    "21": "Mining, Quarrying, Oil & Gas",
    "22": "Utilities",
    "23": "Construction",
    "31-33": "Manufacturing",
    "42": "Wholesale Trade",
    "44-45": "Retail Trade",
    "48-49": "Transportation & Warehousing",
    "51": "Information",
    "52": "Finance & Insurance",
    "53": "Real Estate",
    "54": "Professional & Technical Services",
    "55": "Management of Companies",
    "56": "Administrative & Waste Services",
    "61": "Educational Services",
    "62": "Health Care & Social Assistance",
    "71": "Arts, Entertainment & Recreation",
    "72": "Accommodation & Food Services",
    "81": "Other Services",
}

# Retail subsectors (4-digit NAICS under 44-45)
_RETAIL_SUBSECTORS = {
    "4411": "Automobile Dealers",
    "4413": "Auto Parts, Accessories & Tire Stores",
    "4421": "Furniture Stores",
    "4422": "Home Furnishings Stores",
    "4431": "Electronics & Appliance Stores",
    "4441": "Building Material & Supplies Dealers",
    "4442": "Lawn & Garden Equipment & Supplies Stores",
    "4451": "Grocery Stores",
    "4452": "Specialty Food Stores",
    "4453": "Beer, Wine & Liquor Stores",
    "4461": "Health & Personal Care Stores",
    "4471": "Gasoline Stations",
    "4481": "Clothing Stores",
    "4482": "Shoe Stores",
    "4483": "Jewelry, Luggage & Leather Goods Stores",
    "4511": "Sporting Goods & Hobby Stores",
    "4512": "Book & Music Stores",
    "4521": "Department Stores",
    "4529": "Other General Merchandise Stores",
    "4531": "Florists",
    "4532": "Office Supplies, Stationery & Gift Stores",
    "4533": "Used Merchandise Stores",
    "4539": "Other Miscellaneous Store Retailers",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CensusExpandedLoader:
    """Load county-level business pattern data from the Census CBP API.

    The CBP provides annual data on the number of establishments,
    employment, and payroll by industry sector at the county level.

    Parameters
    ----------
    api_key : str or None
        Optional Census API key.  CBP does not require one, but providing
        a key raises rate limits.
    year : int
        Default data year (2021 is the latest CBP release as of 2024).
    timeout : int
        HTTP request timeout in seconds.

    Examples
    --------
    >>> loader = CensusExpandedLoader()
    >>> data = loader.get_county_business_patterns("48", "453")
    >>> print(data["total_establishments"])
    """

    BASE_URL = _BASE_URL

    def __init__(
        self,
        api_key: Optional[str] = None,
        year: int = _DEFAULT_YEAR,
        timeout: int = _REQUEST_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.year = year
        self.timeout = timeout
        self._session = requests.Session()
        self._last_request_time: float = 0.0
        self._cache: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _RATE_LIMIT:
            time.sleep(_RATE_LIMIT - elapsed)

    def _fetch(
        self,
        state_fips: str,
        county_fips: str,
        naics_code: str,
        year: Optional[int] = None,
    ) -> list[list[str]]:
        """Fetch CBP data rows for a single NAICS code.

        Returns a list of rows (each row is a list of strings) including
        the header row, or an empty list on failure.
        """
        yr = year or self.year
        cache_key = f"{yr}:{state_fips}:{county_fips}:{naics_code}"
        if cache_key in self._cache:
            logger.debug("Census CBP cache hit: %s", cache_key)
            return self._cache[cache_key]

        url = (
            f"{self.BASE_URL}/{yr}/cbp"
            f"?get={_VARIABLES}"
            f"&for=county:{county_fips.zfill(3)}"
            f"&in=state:{state_fips.zfill(2)}"
            f"&NAICS2017={naics_code}"
        )
        if self.api_key:
            url += f"&key={self.api_key}"

        logger.info("Fetching Census CBP: %s", url)
        self._throttle()
        self._last_request_time = time.monotonic()

        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("Census CBP request failed (%s): %s", cache_key, exc)
            return []
        except (ValueError, TypeError) as exc:
            logger.warning("Census CBP JSON parse error (%s): %s", cache_key, exc)
            return []

        if not isinstance(data, list) or len(data) < 2:
            logger.warning("Census CBP returned no data rows for %s", cache_key)
            return []

        self._cache[cache_key] = data
        return data

    def _parse_row(self, header: list[str], row: list[str]) -> dict:
        """Convert a CBP response row into a dict of ints."""
        mapping = dict(zip(header, row))
        return {
            "establishments": self._safe_int(mapping.get("ESTAB")),
            "employment": self._safe_int(mapping.get("EMP")),
            "payroll": self._safe_int(mapping.get("PAYANN")),
        }

    @staticmethod
    def _safe_int(val, default: int = 0) -> int:
        try:
            v = int(float(val))
            return v if v >= 0 else default
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_county_business_patterns(
        self,
        state_fips: str,
        county_fips: str,
        year: Optional[int] = None,
    ) -> dict:
        """Get establishment, employment, and payroll data for a county.

        Fetches totals, retail (NAICS 44-45), food service (NAICS 72),
        and all 2-digit sectors for a full breakdown.

        Returns
        -------
        dict
            Keys: total_establishments, total_employment, total_payroll,
            retail_establishments, retail_employment,
            food_service_establishments, food_service_employment,
            establishment_breakdown (list of sector dicts).
            Returns empty dict on failure.
        """
        st = state_fips.zfill(2)
        ct = county_fips.zfill(3)

        result: dict = {
            "total_establishments": 0,
            "total_employment": 0,
            "total_payroll": 0,
            "retail_establishments": 0,
            "retail_employment": 0,
            "food_service_establishments": 0,
            "food_service_employment": 0,
            "establishment_breakdown": [],
        }

        # --- Fetch totals (NAICS2017=00) ---
        total_data = self._fetch(st, ct, "00", year)
        if len(total_data) >= 2:
            parsed = self._parse_row(total_data[0], total_data[1])
            result["total_establishments"] = parsed["establishments"]
            result["total_employment"] = parsed["employment"]
            result["total_payroll"] = parsed["payroll"]
        else:
            logger.warning(
                "Census CBP: no total data for state=%s county=%s", st, ct
            )
            return {}

        # --- Fetch retail (44-45) ---
        retail_data = self._fetch(st, ct, "44-45", year)
        if len(retail_data) >= 2:
            parsed = self._parse_row(retail_data[0], retail_data[1])
            result["retail_establishments"] = parsed["establishments"]
            result["retail_employment"] = parsed["employment"]

        # --- Fetch food service (72) ---
        food_data = self._fetch(st, ct, "72", year)
        if len(food_data) >= 2:
            parsed = self._parse_row(food_data[0], food_data[1])
            result["food_service_establishments"] = parsed["establishments"]
            result["food_service_employment"] = parsed["employment"]

        # --- Fetch all 2-digit sectors for breakdown ---
        breakdown = []
        for naics_code, name in _NAICS_SECTORS.items():
            if naics_code == "00":
                continue
            sector_data = self._fetch(st, ct, naics_code, year)
            if len(sector_data) >= 2:
                parsed = self._parse_row(sector_data[0], sector_data[1])
                breakdown.append({
                    "naics": naics_code,
                    "name": name,
                    "establishments": parsed["establishments"],
                    "employment": parsed["employment"],
                })

        breakdown.sort(key=lambda x: x["employment"], reverse=True)
        result["establishment_breakdown"] = breakdown

        logger.info(
            "Census CBP loaded for %s%s: %d establishments, %d employees",
            st, ct, result["total_establishments"], result["total_employment"],
        )
        return result

    def get_retail_subsectors(
        self,
        state_fips: str,
        county_fips: str,
        year: Optional[int] = None,
    ) -> list[dict]:
        """Get detailed retail subsector breakdown (4-digit NAICS under 44-45).

        Returns
        -------
        list[dict]
            Each dict has keys: naics, name, establishments, employment.
            Returns empty list on failure.
        """
        st = state_fips.zfill(2)
        ct = county_fips.zfill(3)

        subsectors = []
        for naics_code, name in _RETAIL_SUBSECTORS.items():
            data = self._fetch(st, ct, naics_code, year)
            if len(data) >= 2:
                parsed = self._parse_row(data[0], data[1])
                subsectors.append({
                    "naics": naics_code,
                    "name": name,
                    "establishments": parsed["establishments"],
                    "employment": parsed["employment"],
                })

        subsectors.sort(key=lambda x: x["establishments"], reverse=True)

        logger.info(
            "Census CBP retail subsectors for %s%s: %d subsectors found",
            st, ct, len(subsectors),
        )
        return subsectors

    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self._cache.clear()
        logger.debug("Census CBP cache cleared.")

    def __repr__(self) -> str:
        return (
            f"CensusExpandedLoader(year={self.year}, "
            f"cached_queries={len(self._cache)})"
        )
