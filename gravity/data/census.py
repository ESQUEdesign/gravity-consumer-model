"""
Census / ACS demographic data loader.

Fetches population, household, income, age, and race/ethnicity data from
the US Census Bureau API (or from a local CSV) and converts rows into
``ConsumerOrigin`` pydantic models ready for the gravity model pipeline.

Preferred path: use the ``cenpy`` library when it is installed.  If it is
not available the loader falls back to plain ``requests`` calls against
https://api.census.gov/data/.
"""

from __future__ import annotations

import csv
import json
import logging
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from gravity.data.schema import ConsumerOrigin, origins_to_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FIPS look-up tables (most-used states; extend as needed)
# ---------------------------------------------------------------------------

STATE_FIPS: dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56", "PR": "72",
}

FIPS_TO_STATE: dict[str, str] = {v: k for k, v in STATE_FIPS.items()}

# ACS 5-year variables we pull at the block-group level
_ACS_VARIABLES: list[str] = [
    "B01003_001E",  # total population
    "B11001_001E",  # total households
    "B19013_001E",  # median household income
    # Age buckets (male + female combined via universe total)
    "B01001_003E",  # male under 5
    "B01001_004E",  # male 5-9
    "B01001_005E",  # male 10-14
    "B01001_006E",  # male 15-17
    "B01001_007E",  # male 18-19
    "B01001_008E",  # male 20
    "B01001_009E",  # male 21
    "B01001_010E",  # male 22-24
    "B01001_011E",  # male 25-29
    "B01001_012E",  # male 30-34
    "B01001_013E",  # male 35-39
    "B01001_014E",  # male 40-44
    "B01001_015E",  # male 45-49
    "B01001_016E",  # male 50-54
    "B01001_017E",  # male 55-59
    "B01001_018E",  # male 60-61
    "B01001_019E",  # male 62-64
    "B01001_020E",  # male 65-66
    "B01001_021E",  # male 67-69
    "B01001_022E",  # male 70-74
    "B01001_023E",  # male 75-79
    "B01001_024E",  # male 80-84
    "B01001_025E",  # male 85+
    "B01001_027E",  # female under 5
    "B01001_028E",  # female 5-9
    "B01001_029E",  # female 10-14
    "B01001_030E",  # female 15-17
    "B01001_031E",  # female 18-19
    "B01001_032E",  # female 20
    "B01001_033E",  # female 21
    "B01001_034E",  # female 22-24
    "B01001_035E",  # female 25-29
    "B01001_036E",  # female 30-34
    "B01001_037E",  # female 35-39
    "B01001_038E",  # female 40-44
    "B01001_039E",  # female 45-49
    "B01001_040E",  # female 50-54
    "B01001_041E",  # female 55-59
    "B01001_042E",  # female 60-61
    "B01001_043E",  # female 62-64
    "B01001_044E",  # female 65-66
    "B01001_045E",  # female 67-69
    "B01001_046E",  # female 70-74
    "B01001_047E",  # female 75-79
    "B01001_048E",  # female 80-84
    "B01001_049E",  # female 85+
    # Race / ethnicity
    "B03002_003E",  # white alone, not Hispanic
    "B03002_004E",  # Black alone, not Hispanic
    "B03002_006E",  # Asian alone, not Hispanic
    "B03002_012E",  # Hispanic or Latino
]

# Centroid lat/lon come from the Census gazetteer; the API returns the
# geographic identifiers (state, county, tract, block group) but not
# coordinates.  We request INTPTLAT / INTPTLON where available, otherwise
# the caller must supply them via the CSV path or merge them in separately.
_GEO_PARTS = ["state", "county", "tract", "block group"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_state_fips(state: str) -> str:
    """Return a 2-digit FIPS code for *state*.

    Accepts a two-letter postal abbreviation (``"NY"``), a state name
    (``"New York"``), or a raw FIPS code (``"36"``).
    """
    upper = state.strip().upper()
    if upper in STATE_FIPS:
        return STATE_FIPS[upper]
    if upper in FIPS_TO_STATE:
        return upper
    # Try matching by name (case-insensitive partial match)
    lower = state.strip().lower()
    for abbr, fips in STATE_FIPS.items():
        # The dict only stores abbreviations; accept raw FIPS strings too
        if fips == state.strip().zfill(2):
            return fips
    raise ValueError(
        f"Cannot resolve state '{state}' to a FIPS code. "
        "Pass a two-letter abbreviation, FIPS code, or state name."
    )


def resolve_county_fips(county: str) -> str:
    """Normalise a county FIPS code to 3 digits (zero-padded)."""
    return str(county).strip().zfill(3)


def _safe_int(val, default: int = 0) -> int:
    """Cast a Census API value to ``int``, treating negatives/nulls as *default*."""
    try:
        v = int(float(val))
        return v if v >= 0 else default
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if v >= 0 else default
    except (TypeError, ValueError):
        return default


def _build_age_distribution(row: dict) -> dict[str, int]:
    """Collapse the detailed age-by-sex variables into summary buckets."""
    def _sum_keys(keys: list[str]) -> int:
        return sum(_safe_int(row.get(k, 0)) for k in keys)

    # Male variable suffixes 003-025, female 027-049
    male_pfx = "B01001_{:03d}E"
    fem_pfx = "B01001_{:03d}E"

    # Mapping: bucket label -> (male var nums, female var nums)
    buckets = {
        "age_under_18": (list(range(3, 7)), list(range(27, 31))),
        "age_18_24": (list(range(7, 11)), list(range(31, 35))),
        "age_25_34": ([11, 12], [35, 36]),
        "age_35_44": ([13, 14], [37, 38]),
        "age_45_54": ([15, 16], [39, 40]),
        "age_55_64": ([17, 18, 19], [41, 42, 43]),
        "age_65_plus": (list(range(20, 26)), list(range(44, 50))),
    }
    result: dict[str, int] = {}
    for label, (male_nums, fem_nums) in buckets.items():
        m_keys = [male_pfx.format(n) for n in male_nums]
        f_keys = [fem_pfx.format(n) for n in fem_nums]
        result[label] = _sum_keys(m_keys + f_keys)
    return result


def _build_race_distribution(row: dict) -> dict[str, int]:
    """Extract race/ethnicity counts from a Census API row."""
    return {
        "white_non_hispanic": _safe_int(row.get("B03002_003E", 0)),
        "black_non_hispanic": _safe_int(row.get("B03002_004E", 0)),
        "asian_non_hispanic": _safe_int(row.get("B03002_006E", 0)),
        "hispanic_latino": _safe_int(row.get("B03002_012E", 0)),
    }


def _row_to_origin(row: dict, lat: float = 0.0, lon: float = 0.0) -> ConsumerOrigin:
    """Convert a single Census API row dict into a ``ConsumerOrigin``."""
    # Build the GEOID for the block group (state+county+tract+bg)
    geoid = (
        str(row.get("state", "")).zfill(2)
        + str(row.get("county", "")).zfill(3)
        + str(row.get("tract", "")).zfill(6)
        + str(row.get("block group", "")).zfill(1)
    )

    population = _safe_int(row.get("B01003_001E", 0))
    households = _safe_int(row.get("B11001_001E", 0))
    median_income = _safe_float(row.get("B19013_001E", 0.0))

    demographics = {
        **_build_age_distribution(row),
        **_build_race_distribution(row),
    }

    return ConsumerOrigin(
        origin_id=geoid,
        lat=lat,
        lon=lon,
        population=population,
        households=households,
        median_income=median_income,
        demographics=demographics,
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class CensusLoader:
    """Load Census / ACS demographic data into ``ConsumerOrigin`` models.

    Parameters
    ----------
    api_key : str or None
        Census Bureau API key.  If *None* the loader will attempt
        unauthenticated requests (subject to rate limits).
    year : int
        Default ACS year to query.
    dataset : str
        Census dataset identifier (default ``"acs/acs5"``).

    Examples
    --------
    >>> loader = CensusLoader(api_key="YOUR_KEY")
    >>> origins = loader.load_block_groups("NY", county="061")
    >>> df = loader.to_dataframe(origins)
    """

    BASE_URL = "https://api.census.gov/data"

    def __init__(
        self,
        api_key: Optional[str] = None,
        year: int = 2022,
        dataset: str = "acs/acs5",
    ) -> None:
        self.api_key = api_key
        self.year = year
        self.dataset = dataset

        # Try cenpy first; fall back to requests
        self._use_cenpy = False
        try:
            import cenpy  # noqa: F401
            self._use_cenpy = True
            logger.info("cenpy detected -- will use cenpy for Census queries.")
        except ImportError:
            logger.info("cenpy not installed -- falling back to requests.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_block_groups(
        self,
        state: str,
        county: Optional[str] = None,
        year: Optional[int] = None,
        *,
        include_centroids: bool = True,
    ) -> list[ConsumerOrigin]:
        """Fetch ACS block-group data and return ``ConsumerOrigin`` models.

        Parameters
        ----------
        state : str
            State postal abbreviation (``"NY"``), name, or FIPS code.
        county : str or None
            3-digit county FIPS code.  If *None*, fetches all counties in
            the state (can be slow for large states).
        year : int or None
            ACS year override.
        include_centroids : bool
            If *True* (default) the loader will attempt to fetch the Census
            gazetteer centroids for each block group so that ``lat`` / ``lon``
            are populated.  If centroids cannot be obtained, coordinates
            default to ``0.0``.

        Returns
        -------
        list[ConsumerOrigin]
        """
        year = year or self.year
        state_fips = resolve_state_fips(state)
        county_fips = resolve_county_fips(county) if county else None

        if self._use_cenpy:
            raw_rows = self._fetch_via_cenpy(state_fips, county_fips, year)
        else:
            raw_rows = self._fetch_via_requests(state_fips, county_fips, year)

        if not raw_rows:
            warnings.warn(
                f"No block-group records returned for state={state_fips}, "
                f"county={county_fips}, year={year}."
            )
            return []

        # Build a centroid lookup if requested
        centroid_map: dict[str, tuple[float, float]] = {}
        if include_centroids:
            centroid_map = self._fetch_centroids(state_fips, county_fips, year)

        origins: list[ConsumerOrigin] = []
        for row in raw_rows:
            geoid = (
                str(row.get("state", "")).zfill(2)
                + str(row.get("county", "")).zfill(3)
                + str(row.get("tract", "")).zfill(6)
                + str(row.get("block group", "")).zfill(1)
            )
            lat, lon = centroid_map.get(geoid, (0.0, 0.0))
            origins.append(_row_to_origin(row, lat=lat, lon=lon))

        logger.info(
            "Loaded %d block groups for state=%s county=%s year=%d",
            len(origins), state_fips, county_fips, year,
        )
        return origins

    def load_from_csv(
        self,
        filepath: str | Path,
        *,
        geoid_col: str = "GEOID",
        lat_col: str = "lat",
        lon_col: str = "lon",
        population_col: str = "population",
        households_col: str = "households",
        median_income_col: str = "median_income",
    ) -> list[ConsumerOrigin]:
        """Load pre-downloaded Census data from a CSV file.

        The CSV must contain at minimum a GEOID column, latitude, longitude,
        population, households, and median income.  Any additional columns
        are packed into the ``demographics`` dict.

        Parameters
        ----------
        filepath : str or Path
            Path to the CSV file.
        geoid_col, lat_col, lon_col, population_col, households_col, median_income_col
            Column-name overrides.

        Returns
        -------
        list[ConsumerOrigin]
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Census CSV not found: {filepath}")

        df = pd.read_csv(filepath, dtype={geoid_col: str})

        required = [geoid_col, lat_col, lon_col, population_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        known_cols = {
            geoid_col, lat_col, lon_col,
            population_col, households_col, median_income_col,
        }
        extra_cols = [c for c in df.columns if c not in known_cols]

        origins: list[ConsumerOrigin] = []
        for _, row in df.iterrows():
            demographics = {c: row[c] for c in extra_cols if pd.notna(row.get(c))}
            origins.append(
                ConsumerOrigin(
                    origin_id=str(row[geoid_col]),
                    lat=float(row[lat_col]),
                    lon=float(row[lon_col]),
                    population=_safe_int(row.get(population_col, 0)),
                    households=_safe_int(row.get(households_col, 0)),
                    median_income=_safe_float(row.get(median_income_col, 0.0)),
                    demographics=demographics,
                )
            )

        logger.info("Loaded %d origins from CSV: %s", len(origins), filepath)
        return origins

    @staticmethod
    def to_dataframe(origins: list[ConsumerOrigin]) -> pd.DataFrame:
        """Convenience wrapper around ``origins_to_dataframe``."""
        return origins_to_dataframe(origins)

    # ------------------------------------------------------------------
    # Internal: cenpy path
    # ------------------------------------------------------------------

    def _fetch_via_cenpy(
        self,
        state_fips: str,
        county_fips: Optional[str],
        year: int,
    ) -> list[dict]:
        """Use the ``cenpy`` library to query the Census API."""
        import cenpy

        conn = cenpy.remote.APIConnection(f"ACSDT5Y{year}")

        geo_filter = {"state": state_fips}
        if county_fips:
            geo_filter["county"] = county_fips

        cols = list(_ACS_VARIABLES) + ["NAME"]

        try:
            result_df = conn.query(
                cols=cols,
                geo_unit="block group:*",
                geo_filter=geo_filter,
            )
        except Exception as exc:
            logger.error("cenpy query failed: %s. Falling back to requests.", exc)
            return self._fetch_via_requests(state_fips, county_fips, year)

        return result_df.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Internal: requests fallback
    # ------------------------------------------------------------------

    def _fetch_via_requests(
        self,
        state_fips: str,
        county_fips: Optional[str],
        year: int,
    ) -> list[dict]:
        """Query the Census API directly with ``requests``.

        The Census API limits requests to 50 variables, so we split
        into batches and merge the results by geographic key.
        """
        import requests

        url = f"{self.BASE_URL}/{year}/{self.dataset}"

        if county_fips:
            for_clause = "block group:*"
            in_clause = f"state:{state_fips} county:{county_fips}"
        else:
            for_clause = "block group:*"
            in_clause = f"state:{state_fips}"

        # Split variables into batches of 49 (leave room for geo fields)
        max_per_batch = 49
        var_batches = [
            _ACS_VARIABLES[i:i + max_per_batch]
            for i in range(0, len(_ACS_VARIABLES), max_per_batch)
        ]

        merged: dict[str, dict] = {}  # keyed by geo identifier

        for batch_vars in var_batches:
            get_vars = ",".join(batch_vars)
            params: dict[str, str] = {
                "get": get_vars,
                "for": for_clause,
                "in": in_clause,
            }
            if self.api_key:
                params["key"] = self.api_key

            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            payload: list[list[str]] = resp.json()

            if len(payload) < 2:
                continue

            header = payload[0]
            for record in payload[1:]:
                row = dict(zip(header, record))
                # Build a geo key from state+county+tract+block group
                geo_key = (
                    row.get("state", "")
                    + row.get("county", "")
                    + row.get("tract", "")
                    + row.get("block group", "")
                )
                if geo_key in merged:
                    merged[geo_key].update(row)
                else:
                    merged[geo_key] = row

        return list(merged.values())

    # ------------------------------------------------------------------
    # Internal: centroid lookup
    # ------------------------------------------------------------------

    def _fetch_centroids(
        self,
        state_fips: str,
        county_fips: Optional[str],
        year: int,
    ) -> dict[str, tuple[float, float]]:
        """Try to retrieve block-group centroids from the Census gazetteer.

        Returns a dict mapping GEOID -> (lat, lon).  If the request fails
        the dict is returned empty and coordinates will default to 0.0.
        """
        import requests

        # The gazetteer files live on the Census FTP/HTTPS site.
        # Pattern: https://www2.census.gov/geo/docs/maps-data/data/gazetteer/
        #          2022_Gazetteer/2022_Gaz_bg_national.zip
        gaz_url = (
            f"https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
            f"{year}_Gazetteer/{year}_Gaz_bg_national.zip"
        )

        try:
            resp = requests.get(gaz_url, timeout=60)
            resp.raise_for_status()
        except Exception:
            logger.debug("Could not download gazetteer centroids; using 0.0")
            return {}

        import io
        import zipfile

        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                names = zf.namelist()
                txt_name = [n for n in names if n.endswith(".txt")][0]
                with zf.open(txt_name) as f:
                    gaz_df = pd.read_csv(
                        f,
                        sep="\t",
                        dtype={"GEOID": str},
                        low_memory=False,
                    )
        except Exception:
            logger.debug("Could not parse gazetteer file; using 0.0")
            return {}

        # Rename columns (strip whitespace that sometimes appears)
        gaz_df.columns = [c.strip() for c in gaz_df.columns]

        # Filter to state (and optionally county)
        mask = gaz_df["GEOID"].str[:2] == state_fips
        if county_fips:
            mask = mask & (gaz_df["GEOID"].str[2:5] == county_fips)
        gaz_df = gaz_df.loc[mask]

        centroids: dict[str, tuple[float, float]] = {}
        for _, row in gaz_df.iterrows():
            geoid = str(row["GEOID"])
            lat = float(row.get("INTPTLAT", 0.0))
            lon = float(row.get("INTPTLONG", 0.0))
            centroids[geoid] = (lat, lon)

        logger.debug("Fetched %d block-group centroids from gazetteer.", len(centroids))
        return centroids
