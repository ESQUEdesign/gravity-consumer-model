"""
Bureau of Labor Statistics QCEW (Quarterly Census of Employment and Wages) loader.

Fetches county-level employment, wage, and establishment data from the
publicly available BLS QCEW API.  No API key is required.

API documentation:
    https://www.bls.gov/cew/downloadable-data-files.htm
    https://data.bls.gov/cew/doc/access/csv_data_slices.htm

Endpoint pattern:
    https://data.bls.gov/cew/data/api/{year}/{qtr}/area/{area_fips}.csv

Where:
    year      = 4-digit year (e.g. 2023)
    qtr       = 1, 2, 3, 4, or "a" for annual averages
    area_fips = 5-digit state FIPS + county FIPS (e.g. "48453" for Travis County, TX)
"""

from __future__ import annotations

import io
import logging
import time
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests

from gravity.data.schema import ConsumerOrigin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://data.bls.gov/cew/data/api/{year}/{qtr}/area/{area_fips}.csv"
_DEFAULT_YEAR = 2023
_DEFAULT_QTR = "a"  # annual averages
_REQUEST_TIMEOUT = 30  # seconds
_RATE_LIMIT = 1.0  # minimum seconds between requests

# NAICS ownership codes: 5 = private sector (most useful for retail analysis)
_PRIVATE_OWNERSHIP = "5"

# NAICS sector code for retail trade
_RETAIL_NAICS = "44-45"

# Top-level 2-digit NAICS sectors we summarize in the industry breakdown
_AGGREGATION_LEVEL_TOTAL = "70"  # county total, all industries
_AGGREGATION_LEVEL_SECTOR = "74"  # 2-digit NAICS sector within county

# NAICS 2-digit sector code → industry name (BLS QCEW API does NOT return titles)
_NAICS_SECTOR_NAMES = {
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
    "92": "Public Administration",
    "10": "Total, All Industries",
    "99": "Unclassified",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BLSLoader:
    """Load county-level employment and wage data from the BLS QCEW API.

    The QCEW data covers virtually all employers subject to state
    unemployment insurance laws, providing a near-census of employment
    and wages at the county level.

    Parameters
    ----------
    timeout : int
        HTTP request timeout in seconds.
    rate_limit : float
        Minimum seconds between consecutive API requests.

    Examples
    --------
    >>> loader = BLSLoader()
    >>> data = loader.get_county_employment("48", "453", year=2023)
    >>> print(data["total_employment"])
    """

    def __init__(
        self,
        timeout: int = _REQUEST_TIMEOUT,
        rate_limit: float = _RATE_LIMIT,
    ) -> None:
        self.timeout = timeout
        self._rate_limit = rate_limit
        self._session = requests.Session()
        self._last_request_time: float = 0.0
        # In-memory cache: keyed by (area_fips, year, qtr)
        self._cache: dict[tuple[str, int, str], pd.DataFrame] = {}

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

    def _build_area_fips(self, state_fips: str, county_fips: str) -> str:
        """Combine state and county FIPS into a 5-digit area FIPS code."""
        return state_fips.zfill(2) + county_fips.zfill(3)

    def _fetch_csv(
        self,
        area_fips: str,
        year: int = _DEFAULT_YEAR,
        qtr: str = _DEFAULT_QTR,
    ) -> pd.DataFrame:
        """Fetch and parse the QCEW CSV for a given area/year/quarter.

        Returns a pandas DataFrame with all rows from the CSV.  Results
        are cached in memory so repeated calls for the same parameters
        do not hit the API again.

        Returns an empty DataFrame on any network or parsing error.
        """
        cache_key = (area_fips, year, qtr)
        if cache_key in self._cache:
            logger.debug("BLS cache hit for %s/%d/%s", area_fips, year, qtr)
            return self._cache[cache_key]

        url = _BASE_URL.format(year=year, qtr=qtr, area_fips=area_fips)
        logger.info("Fetching BLS QCEW data: %s", url)

        self._throttle()
        self._last_request_time = time.monotonic()

        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "BLS QCEW API request failed for %s (year=%d, qtr=%s): %s",
                area_fips, year, qtr, exc,
            )
            return pd.DataFrame()

        try:
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception as exc:
            logger.warning("Failed to parse BLS CSV for %s: %s", area_fips, exc)
            return pd.DataFrame()

        # Normalize column names to lowercase with underscores
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        self._cache[cache_key] = df
        logger.info(
            "BLS QCEW data loaded for %s: %d rows, %d columns",
            area_fips, len(df), len(df.columns),
        )
        return df

    def _safe_int(self, val, default: int = 0) -> int:
        """Safely convert a value to int."""
        try:
            v = int(float(val))
            return v if v >= 0 else default
        except (TypeError, ValueError):
            return default

    def _safe_float(self, val, default: float = 0.0) -> float:
        """Safely convert a value to float."""
        try:
            v = float(val)
            return v if v >= 0 else default
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_county_employment(
        self,
        state_fips: str,
        county_fips: str,
        year: int = _DEFAULT_YEAR,
    ) -> dict:
        """Get aggregate employment and wage data for a county.

        Parameters
        ----------
        state_fips : str
            2-digit state FIPS code (e.g. "48" for Texas).
        county_fips : str
            3-digit county FIPS code (e.g. "453" for Travis County).
        year : int
            Data year (default 2023).

        Returns
        -------
        dict
            Keys:
            - ``total_employment`` (int): Total annual average employment.
            - ``avg_weekly_wage`` (float): Average weekly wage across all industries.
            - ``total_establishments`` (int): Total number of establishments.
            - ``annual_avg_pay`` (float): Annual average pay per employee.
            - ``industry_breakdown`` (list[dict]): Top 10 NAICS sectors by
              employment, each with keys ``industry_code``, ``industry_title``,
              ``employment``, ``avg_weekly_wage``, ``establishments``.

        Returns an empty dict if the BLS API is unreachable or data is
        unavailable for the requested area/year.
        """
        area_fips = self._build_area_fips(state_fips, county_fips)
        df = self._fetch_csv(area_fips, year=year)

        if df.empty:
            return {}

        # ---------------------------------------------------------------
        # Extract county-wide totals
        # ---------------------------------------------------------------
        # The QCEW CSV uses "agglvl_code" to denote aggregation level.
        # Code 70 = county total (all industries, all ownerships).
        # Code 74 = county-level, 2-digit NAICS sector.
        # Ownership code 0 = all ownerships combined.

        # Column names may vary slightly across years; handle common variants
        agglvl_col = None
        for candidate in ("agglvl_code", "agglvl_code "):
            if candidate in df.columns:
                agglvl_col = candidate
                break

        own_col = None
        for candidate in ("own_code", "own_code "):
            if candidate in df.columns:
                own_col = candidate
                break

        if agglvl_col is None or own_col is None:
            logger.warning(
                "BLS CSV missing expected columns (agglvl_code, own_code). "
                "Columns found: %s", list(df.columns),
            )
            return {}

        # County-wide totals (all ownerships)
        total_mask = df[agglvl_col].astype(str).str.strip() == _AGGREGATION_LEVEL_TOTAL
        total_rows = df[total_mask]

        result: dict = {
            "total_employment": 0,
            "avg_weekly_wage": 0.0,
            "total_establishments": 0,
            "annual_avg_pay": 0.0,
            "industry_breakdown": [],
        }

        if not total_rows.empty:
            row = total_rows.iloc[0]
            result["total_employment"] = self._safe_int(
                row.get("annual_avg_emplvl", row.get("annual_avg_emplvl ", 0))
            )
            result["avg_weekly_wage"] = self._safe_float(
                row.get("annual_avg_wkly_wage",
                         row.get("avg_wkly_wage",
                                  row.get("avg_wkly_wage ", 0.0)))
            )
            result["total_establishments"] = self._safe_int(
                row.get("annual_avg_estabs", row.get("annual_avg_estabs ", 0))
            )
            result["annual_avg_pay"] = self._safe_float(
                row.get("avg_annual_pay", row.get("avg_annual_pay ", 0.0))
            )

        # ---------------------------------------------------------------
        # Industry breakdown: top 10 NAICS sectors by employment
        # ---------------------------------------------------------------
        sector_mask = df[agglvl_col].astype(str).str.strip() == _AGGREGATION_LEVEL_SECTOR
        sector_df = df[sector_mask].copy()

        if not sector_df.empty:
            # Find the employment column
            empl_col = None
            for candidate in ("annual_avg_emplvl", "annual_avg_emplvl "):
                if candidate in sector_df.columns:
                    empl_col = candidate
                    break

            if empl_col is not None:
                sector_df["_empl"] = pd.to_numeric(
                    sector_df[empl_col], errors="coerce"
                ).fillna(0)
                sector_df = sector_df.sort_values("_empl", ascending=False)

                # Industry code/title columns
                code_col = next(
                    (c for c in ("industry_code", "industry_code ")
                     if c in sector_df.columns),
                    None,
                )
                title_col = next(
                    (c for c in ("industry_title", "industry_title ")
                     if c in sector_df.columns),
                    None,
                )
                wage_col = next(
                    (c for c in ("annual_avg_wkly_wage", "avg_wkly_wage",
                                 "avg_wkly_wage ")
                     if c in sector_df.columns),
                    None,
                )
                estab_col = next(
                    (c for c in ("annual_avg_estabs", "annual_avg_estabs ")
                     if c in sector_df.columns),
                    None,
                )

                breakdown = []
                for _, srow in sector_df.head(10).iterrows():
                    ind_code = str(srow[code_col]).strip() if code_col else ""
                    # Use title column if available, otherwise look up NAICS code
                    if title_col and pd.notna(srow.get(title_col)):
                        ind_title = str(srow[title_col]).strip()
                    else:
                        ind_title = _NAICS_SECTOR_NAMES.get(
                            ind_code, f"NAICS {ind_code}")
                    entry = {
                        "industry_code": ind_code,
                        "industry_title": ind_title,
                        "employment": self._safe_int(srow.get(empl_col, 0)),
                        "avg_weekly_wage": self._safe_float(
                            srow[wage_col] if wage_col else 0.0
                        ),
                        "establishments": self._safe_int(
                            srow[estab_col] if estab_col else 0
                        ),
                    }
                    breakdown.append(entry)

                result["industry_breakdown"] = breakdown

        return result

    def get_retail_employment(
        self,
        state_fips: str,
        county_fips: str,
        year: int = _DEFAULT_YEAR,
    ) -> dict:
        """Get retail-sector-specific employment data for a county.

        Filters to NAICS sector 44-45 (Retail Trade).

        Parameters
        ----------
        state_fips : str
            2-digit state FIPS code.
        county_fips : str
            3-digit county FIPS code.
        year : int
            Data year (default 2023).

        Returns
        -------
        dict
            Keys:
            - ``retail_employment`` (int): Retail sector employment count.
            - ``retail_establishments`` (int): Number of retail establishments.
            - ``retail_avg_wage`` (float): Average weekly wage in retail.

        Returns an empty dict if data is unavailable.
        """
        area_fips = self._build_area_fips(state_fips, county_fips)
        df = self._fetch_csv(area_fips, year=year)

        if df.empty:
            return {}

        # Find the industry_code column
        code_col = next(
            (c for c in ("industry_code", "industry_code ")
             if c in df.columns),
            None,
        )
        if code_col is None:
            logger.warning("BLS CSV missing industry_code column.")
            return {}

        # Filter to retail trade sector (NAICS 44-45)
        # The QCEW CSV represents this as "44-45" in the industry_code field
        retail_mask = df[code_col].astype(str).str.strip() == _RETAIL_NAICS
        retail_rows = df[retail_mask]

        if retail_rows.empty:
            # Try alternate representations
            for alt in ("44", "45", "44-45 "):
                alt_mask = df[code_col].astype(str).str.strip() == alt.strip()
                if alt_mask.any():
                    retail_rows = df[alt_mask]
                    break

        if retail_rows.empty:
            logger.info(
                "No retail trade (NAICS 44-45) data found for %s in %d.",
                area_fips, year,
            )
            return {}

        row = retail_rows.iloc[0]

        empl_col = next(
            (c for c in ("annual_avg_emplvl", "annual_avg_emplvl ")
             if c in df.columns),
            None,
        )
        wage_col = next(
            (c for c in ("annual_avg_wkly_wage", "avg_wkly_wage",
                         "avg_wkly_wage ")
             if c in df.columns),
            None,
        )
        estab_col = next(
            (c for c in ("annual_avg_estabs", "annual_avg_estabs ")
             if c in df.columns),
            None,
        )

        return {
            "retail_employment": self._safe_int(row[empl_col] if empl_col else 0),
            "retail_establishments": self._safe_int(row[estab_col] if estab_col else 0),
            "retail_avg_wage": self._safe_float(row[wage_col] if wage_col else 0.0),
        }

    def enrich_origins(
        self,
        origins: list[ConsumerOrigin],
        state_fips: str,
        county_fips: str,
        year: int = _DEFAULT_YEAR,
    ) -> list[ConsumerOrigin]:
        """Add BLS employment data to consumer origin demographics.

        Fetches county-level data once and writes it into each origin's
        ``demographics`` dict.  Fields added:

        - ``employment_rate``: ratio of total employment to county population
          (estimated from origin populations if Census total is unavailable).
        - ``avg_weekly_wage``: county-wide average weekly wage.
        - ``retail_establishments``: number of retail establishments in county.

        Parameters
        ----------
        origins : list[ConsumerOrigin]
            Origins to enrich (modified in place).
        state_fips : str
            2-digit state FIPS code.
        county_fips : str
            3-digit county FIPS code.
        year : int
            Data year (default 2023).

        Returns
        -------
        list[ConsumerOrigin]
            The same list, mutated in place with BLS data.
        """
        county_data = self.get_county_employment(state_fips, county_fips, year=year)
        retail_data = self.get_retail_employment(state_fips, county_fips, year=year)

        if not county_data:
            logger.warning(
                "No BLS data available for state=%s county=%s year=%d. "
                "Origins will not be enriched with employment data.",
                state_fips, county_fips, year,
            )
            return origins

        total_employment = county_data.get("total_employment", 0)
        avg_weekly_wage = county_data.get("avg_weekly_wage", 0.0)
        retail_establishments = retail_data.get("retail_establishments", 0)

        # Estimate employment rate using sum of origin populations as
        # a proxy for working-age population (rough approximation).
        total_population = sum(o.population for o in origins) if origins else 0
        employment_rate = (
            total_employment / total_population
            if total_population > 0 and total_employment > 0
            else 0.0
        )

        for origin in origins:
            origin.demographics["employment_rate"] = round(employment_rate, 4)
            origin.demographics["avg_weekly_wage"] = avg_weekly_wage
            origin.demographics["retail_establishments"] = retail_establishments
            # Also store the full county data for reference
            origin.demographics["bls_county_employment"] = total_employment
            origin.demographics["bls_annual_avg_pay"] = county_data.get(
                "annual_avg_pay", 0.0
            )

        logger.info(
            "Enriched %d origins with BLS data: employment_rate=%.4f, "
            "avg_weekly_wage=%.2f, retail_establishments=%d",
            len(origins), employment_rate, avg_weekly_wage, retail_establishments,
        )

        return origins

    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self._cache.clear()
        logger.debug("BLS data cache cleared.")

    def __repr__(self) -> str:
        return f"BLSLoader(timeout={self.timeout}, cached_areas={len(self._cache)})"
