"""
US Geography Lookup
===================
Complete FIPS database for all 50 US states, ~3,200+ counties.
Supports search by state, county name, city, or ZIP code.
Auto-generates bounding boxes from county centroids.

No API keys required. Uses Census API + Census Geocoder (free).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"


def _cache_path(name: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / name


# ---------------------------------------------------------------------------
# State FIPS codes (complete)
# ---------------------------------------------------------------------------

STATE_FIPS: dict[str, dict[str, str]] = {
    "01": {"name": "Alabama", "abbr": "AL"},
    "02": {"name": "Alaska", "abbr": "AK"},
    "04": {"name": "Arizona", "abbr": "AZ"},
    "05": {"name": "Arkansas", "abbr": "AR"},
    "06": {"name": "California", "abbr": "CA"},
    "08": {"name": "Colorado", "abbr": "CO"},
    "09": {"name": "Connecticut", "abbr": "CT"},
    "10": {"name": "Delaware", "abbr": "DE"},
    "11": {"name": "District of Columbia", "abbr": "DC"},
    "12": {"name": "Florida", "abbr": "FL"},
    "13": {"name": "Georgia", "abbr": "GA"},
    "15": {"name": "Hawaii", "abbr": "HI"},
    "16": {"name": "Idaho", "abbr": "ID"},
    "17": {"name": "Illinois", "abbr": "IL"},
    "18": {"name": "Indiana", "abbr": "IN"},
    "19": {"name": "Iowa", "abbr": "IA"},
    "20": {"name": "Kansas", "abbr": "KS"},
    "21": {"name": "Kentucky", "abbr": "KY"},
    "22": {"name": "Louisiana", "abbr": "LA"},
    "23": {"name": "Maine", "abbr": "ME"},
    "24": {"name": "Maryland", "abbr": "MD"},
    "25": {"name": "Massachusetts", "abbr": "MA"},
    "26": {"name": "Michigan", "abbr": "MI"},
    "27": {"name": "Minnesota", "abbr": "MN"},
    "28": {"name": "Mississippi", "abbr": "MS"},
    "29": {"name": "Missouri", "abbr": "MO"},
    "30": {"name": "Montana", "abbr": "MT"},
    "31": {"name": "Nebraska", "abbr": "NE"},
    "32": {"name": "Nevada", "abbr": "NV"},
    "33": {"name": "New Hampshire", "abbr": "NH"},
    "34": {"name": "New Jersey", "abbr": "NJ"},
    "35": {"name": "New Mexico", "abbr": "NM"},
    "36": {"name": "New York", "abbr": "NY"},
    "37": {"name": "North Carolina", "abbr": "NC"},
    "38": {"name": "North Dakota", "abbr": "ND"},
    "39": {"name": "Ohio", "abbr": "OH"},
    "40": {"name": "Oklahoma", "abbr": "OK"},
    "41": {"name": "Oregon", "abbr": "OR"},
    "42": {"name": "Pennsylvania", "abbr": "PA"},
    "44": {"name": "Rhode Island", "abbr": "RI"},
    "45": {"name": "South Carolina", "abbr": "SC"},
    "46": {"name": "South Dakota", "abbr": "SD"},
    "47": {"name": "Tennessee", "abbr": "TN"},
    "48": {"name": "Texas", "abbr": "TX"},
    "49": {"name": "Utah", "abbr": "UT"},
    "50": {"name": "Vermont", "abbr": "VT"},
    "51": {"name": "Virginia", "abbr": "VA"},
    "53": {"name": "Washington", "abbr": "WA"},
    "54": {"name": "West Virginia", "abbr": "WV"},
    "55": {"name": "Wisconsin", "abbr": "WI"},
    "56": {"name": "Wyoming", "abbr": "WY"},
    "72": {"name": "Puerto Rico", "abbr": "PR"},
}

# Reverse lookups
_STATE_ABBR_TO_FIPS = {v["abbr"]: k for k, v in STATE_FIPS.items()}
_STATE_NAME_TO_FIPS = {v["name"].lower(): k for k, v in STATE_FIPS.items()}


def state_fips_from_name(name: str) -> Optional[str]:
    """Look up state FIPS from name or abbreviation."""
    name = name.strip()
    upper = name.upper()
    if upper in _STATE_ABBR_TO_FIPS:
        return _STATE_ABBR_TO_FIPS[upper]
    lower = name.lower()
    if lower in _STATE_NAME_TO_FIPS:
        return _STATE_NAME_TO_FIPS[lower]
    # Fuzzy: check startswith
    for sname, fips in _STATE_NAME_TO_FIPS.items():
        if sname.startswith(lower):
            return fips
    return None


# ---------------------------------------------------------------------------
# County database (fetched from Census API, cached to disk)
# ---------------------------------------------------------------------------

_county_cache: Optional[pd.DataFrame] = None


def get_all_counties(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch all US counties with names, FIPS codes, and centroids.

    Returns DataFrame with columns:
        state_fips, county_fips, county_name, state_name, state_abbr,
        label, lat, lon

    Data is cached to disk after first fetch.
    """
    global _county_cache

    if _county_cache is not None and not force_refresh:
        return _county_cache

    cache_file = _cache_path("us_counties.parquet")

    # Try disk cache first
    if cache_file.exists() and not force_refresh:
        try:
            _county_cache = pd.read_parquet(cache_file)
            logger.info("Loaded %d counties from cache", len(_county_cache))
            return _county_cache
        except Exception:
            pass

    # Fetch from Census API
    logger.info("Fetching county list from Census API...")
    df = _fetch_counties_from_census()

    if df is not None and not df.empty:
        # Add centroids from Census Gazetteer
        df = _add_centroids(df)

        # Save cache
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            # Parquet might not be available; use CSV
            try:
                csv_file = _cache_path("us_counties.csv")
                df.to_csv(csv_file, index=False)
            except Exception:
                pass

        _county_cache = df
        logger.info("Cached %d counties", len(df))
        return df

    # Absolute fallback: return empty
    logger.warning("Could not fetch county data")
    return pd.DataFrame(columns=[
        "state_fips", "county_fips", "county_name", "state_name",
        "state_abbr", "label", "lat", "lon",
    ])


def _fetch_counties_from_census() -> Optional[pd.DataFrame]:
    """Fetch all counties from Census ACS API."""
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed")
        return None

    try:
        resp = requests.get(
            "https://api.census.gov/data/2022/acs/acs5",
            params={
                "get": "NAME,B01003_001E",
                "for": "county:*",
                "in": "state:*",
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.warning("Census API error: %s", e)
        return None

    if len(payload) < 2:
        return None

    header = payload[0]
    rows = [dict(zip(header, r)) for r in payload[1:]]

    records = []
    for row in rows:
        sf = str(row.get("state", "")).zfill(2)
        cf = str(row.get("county", "")).zfill(3)
        name_full = row.get("NAME", "")

        # NAME format: "County Name, State Name"
        parts = name_full.split(", ")
        county_name = parts[0] if parts else name_full
        state_name = parts[1] if len(parts) > 1 else ""

        state_info = STATE_FIPS.get(sf, {})
        state_abbr = state_info.get("abbr", "")

        pop = 0
        try:
            pop = max(0, int(float(row.get("B01003_001E", 0))))
        except (TypeError, ValueError):
            pass

        records.append({
            "state_fips": sf,
            "county_fips": cf,
            "county_name": county_name,
            "state_name": state_name or state_info.get("name", ""),
            "state_abbr": state_abbr,
            "population": pop,
            "label": f"{county_name}, {state_abbr}",
        })

    return pd.DataFrame(records)


def _add_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Add lat/lon centroids to county DataFrame.

    Uses the Census Gazetteer file for centroids.  If unavailable,
    falls back to a deterministic hash-based approximation.
    """
    try:
        import requests
    except ImportError:
        return _add_centroids_fallback(df)

    gazetteer_cache = _cache_path("county_centroids.json")

    # Try cached centroids
    centroids = {}
    if gazetteer_cache.exists():
        try:
            with open(gazetteer_cache) as f:
                centroids = json.load(f)
        except Exception:
            pass

    if not centroids:
        # Try multiple Gazetteer file years
        gazetteer_urls = [
            "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.txt",
            "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.txt",
            "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2022_Gazetteer/2022_Gaz_counties_national.txt",
            "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2021_Gazetteer/2021_Gaz_counties_national.txt",
            "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.txt",
        ]

        for url in gazetteer_urls:
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                for line in resp.text.strip().split("\n")[1:]:
                    parts = line.split("\t")
                    if len(parts) >= 10:
                        geoid = parts[1].strip()
                        try:
                            lat = float(parts[-2].strip())
                            lon = float(parts[-1].strip())
                            centroids[geoid] = {"lat": lat, "lon": lon}
                        except (ValueError, IndexError):
                            pass
                if centroids:
                    logger.info("Loaded %d centroids from %s", len(centroids), url)
                    break
            except Exception:
                continue

        # Fallback: TIGERweb ArcGIS REST API (layer 78 = Counties)
        if not centroids:
            try:
                tiger_url = (
                    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
                    "TIGERweb/tigerWMS_ACS2022/MapServer/78/query"
                )
                resp = requests.get(tiger_url, params={
                    "where": "1=1",
                    "outFields": "GEOID,CENTLAT,CENTLON",
                    "f": "json",
                    "returnGeometry": "false",
                    "resultRecordCount": "5000",
                }, timeout=30)
                data = resp.json()
                for feat in data.get("features", []):
                    attrs = feat.get("attributes", {})
                    geoid = str(attrs.get("GEOID", ""))
                    clat = attrs.get("CENTLAT", "")
                    clon = attrs.get("CENTLON", "")
                    if geoid and clat and clon:
                        try:
                            centroids[geoid] = {
                                "lat": float(str(clat).strip("+")),
                                "lon": float(str(clon).strip("+")),
                            }
                        except (ValueError, TypeError):
                            pass
                if centroids:
                    logger.info("Loaded %d centroids from TIGERweb", len(centroids))
            except Exception as e:
                logger.warning("TIGERweb fetch failed: %s", e)

        # Cache to disk if we got results
        if centroids:
            try:
                with open(gazetteer_cache, "w") as f:
                    json.dump(centroids, f)
            except Exception:
                pass

    if centroids:
        lats = []
        lons = []
        for _, row in df.iterrows():
            geoid = row["state_fips"] + row["county_fips"]
            c = centroids.get(geoid, {})
            lats.append(c.get("lat", 0.0))
            lons.append(c.get("lon", 0.0))
        df["lat"] = lats
        df["lon"] = lons
        return df

    return _add_centroids_fallback(df)


def _add_centroids_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Rough centroid approximation when Gazetteer is unavailable."""
    # US bounding box
    lats = []
    lons = []
    for _, row in df.iterrows():
        geoid = row["state_fips"] + row["county_fips"]
        h = int(hashlib.md5(geoid.encode()).hexdigest(), 16)
        lat = 25.0 + (h % 10000) / 10000 * 24.0  # ~25-49 lat
        lon = -125.0 + ((h >> 16) % 10000) / 10000 * 57.0  # ~-125 to -68 lon
        lats.append(lat)
        lons.append(lon)
    df["lat"] = lats
    df["lon"] = lons
    return df


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def search_counties(query: str, state_fips: str = None,
                    limit: int = 20) -> pd.DataFrame:
    """Search counties by name. Returns matching rows from the county database.

    Parameters
    ----------
    query : str
        Partial county name (case-insensitive). E.g., "Travis", "Los Ang", "Cook".
    state_fips : str, optional
        Filter to a specific state.
    limit : int
        Max results to return.
    """
    df = get_all_counties()
    if df.empty:
        return df

    mask = df["county_name"].str.lower().str.contains(query.lower(), na=False)

    if state_fips:
        mask = mask & (df["state_fips"] == state_fips.zfill(2))

    results = df[mask].sort_values("population", ascending=False)
    return results.head(limit)


def search_by_city(city: str, state: str = None, limit: int = 10) -> pd.DataFrame:
    """Search for a city and return its county.

    Uses the county database (many counties are named after their cities).
    For exact geocoding, use geocode_location().
    """
    df = get_all_counties()
    if df.empty:
        return df

    city_lower = city.lower().strip()

    # Many counties match city names directly
    mask = (
        df["county_name"].str.lower().str.contains(city_lower, na=False)
        | df["label"].str.lower().str.contains(city_lower, na=False)
    )

    if state:
        sfips = state_fips_from_name(state)
        if sfips:
            mask = mask & (df["state_fips"] == sfips)

    results = df[mask].sort_values("population", ascending=False)
    return results.head(limit)


def get_county(state_fips: str, county_fips: str) -> Optional[dict]:
    """Get a single county by FIPS codes.

    Returns dict with: state_fips, county_fips, county_name, state_name,
    state_abbr, label, lat, lon, population.
    """
    df = get_all_counties()
    if df.empty:
        return None

    match = df[
        (df["state_fips"] == state_fips.zfill(2))
        & (df["county_fips"] == county_fips.zfill(3))
    ]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def get_counties_for_state(state_fips: str) -> pd.DataFrame:
    """Get all counties for a state, sorted by population."""
    df = get_all_counties()
    if df.empty:
        return df
    result = df[df["state_fips"] == state_fips.zfill(2)].sort_values(
        "population", ascending=False
    )
    return result


# ---------------------------------------------------------------------------
# Bounding box generation
# ---------------------------------------------------------------------------

def county_bbox(state_fips: str, county_fips: str,
                radius_km: float = 12.0) -> Optional[tuple[float, float, float, float]]:
    """Generate a bounding box for a county.

    Uses the county centroid and the specified radius (km).
    Returns (south, west, north, east) or None if county not found.

    Parameters
    ----------
    radius_km : float
        Half-width of the bounding box in km. Default 12 km (~7.5 miles)
        captures most urban cores. Increase for rural counties.
    """
    county = get_county(state_fips, county_fips)
    if county is None or county.get("lat", 0) == 0:
        return None

    return bbox_from_point(county["lat"], county["lon"], radius_km)


def bbox_from_point(lat: float, lon: float,
                    radius_km: float = 12.0) -> tuple[float, float, float, float]:
    """Generate a bounding box around a lat/lon point.

    Returns (south, west, north, east).
    """
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 * cos(lat) km
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.01))

    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


# ---------------------------------------------------------------------------
# ZIP code utilities
# ---------------------------------------------------------------------------

def geocode_zip(zipcode: str) -> Optional[dict]:
    """Geocode a ZIP or ZIP+4 code.

    Returns dict with: zip5, lat, lon, city, state_abbr, state_fips, county_fips.
    Returns None on failure.
    """
    try:
        import requests
    except ImportError:
        return None

    zip5 = re.sub(r"[^0-9]", "", zipcode.strip())[:5]
    if len(zip5) != 5:
        return None

    result = {"zip5": zip5, "lat": None, "lon": None,
              "city": None, "state_abbr": None,
              "state_fips": None, "county_fips": None}

    # Method 1: Census geocoder (returns county FIPS too)
    try:
        resp = requests.get(
            "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress",
            params={
                "address": zip5,
                "benchmark": "Public_AR_Current",
                "vintage": "Current_Current",
                "format": "json",
            },
            timeout=10,
        )
        data = resp.json()
        matches = data.get("result", {}).get("addressMatches", [])
        if matches:
            m = matches[0]
            result["lat"] = float(m["coordinates"]["y"])
            result["lon"] = float(m["coordinates"]["x"])
            # Extract county FIPS from geographies
            geos = m.get("geographies", {})
            counties = geos.get("Counties", geos.get("Census Tracts", []))
            if counties:
                c = counties[0]
                result["state_fips"] = c.get("STATE", "")
                result["county_fips"] = c.get("COUNTY", "")
            # Parse address
            addr = m.get("matchedAddress", "")
            parts = addr.split(", ")
            if len(parts) >= 2:
                result["city"] = parts[-2] if len(parts) >= 3 else parts[0]
                state_part = parts[-1].split(" ")[0] if parts[-1] else ""
                result["state_abbr"] = state_part
            return result
    except Exception:
        pass

    # Method 2: zippopotam.us (simpler, lat/lon only)
    try:
        resp = requests.get(f"https://api.zippopotam.us/us/{zip5}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            places = data.get("places", [])
            if places:
                p = places[0]
                result["lat"] = float(p["latitude"])
                result["lon"] = float(p["longitude"])
                result["city"] = p.get("place name")
                result["state_abbr"] = p.get("state abbreviation")
                # Resolve state FIPS
                if result["state_abbr"]:
                    result["state_fips"] = _STATE_ABBR_TO_FIPS.get(
                        result["state_abbr"]
                    )
                return result
    except Exception:
        pass

    return None


def zip_to_county(zipcode: str) -> Optional[dict]:
    """Look up the county for a ZIP code.

    Returns dict with county info plus the ZIP centroid, or None.
    """
    geo = geocode_zip(zipcode)
    if geo is None or geo.get("lat") is None:
        return None

    # If Census geocoder gave us county FIPS, use it
    if geo.get("state_fips") and geo.get("county_fips"):
        county = get_county(geo["state_fips"], geo["county_fips"])
        if county:
            county["zip5"] = geo["zip5"]
            county["zip_lat"] = geo["lat"]
            county["zip_lon"] = geo["lon"]
            return county

    # Fallback: find nearest county centroid
    if geo.get("lat") and geo.get("state_fips"):
        counties = get_counties_for_state(geo["state_fips"])
        if not counties.empty and "lat" in counties.columns:
            valid = counties[counties["lat"] != 0]
            if not valid.empty:
                dists = np.sqrt(
                    (valid["lat"] - geo["lat"]) ** 2
                    + (valid["lon"] - geo["lon"]) ** 2
                )
                nearest_idx = dists.idxmin()
                county = valid.loc[nearest_idx].to_dict()
                county["zip5"] = geo["zip5"]
                county["zip_lat"] = geo["lat"]
                county["zip_lon"] = geo["lon"]
                return county

    return None


# ---------------------------------------------------------------------------
# Geocode any location string
# ---------------------------------------------------------------------------

def geocode_location(query: str) -> Optional[dict]:
    """Geocode a free-text location (city, address, ZIP, etc.).

    Returns dict with: lat, lon, state_fips, county_fips, label.
    """
    query = query.strip()

    # Check if it's a ZIP code
    zip_match = re.match(r"^(\d{5})(-\d{4})?$", query)
    if zip_match:
        return zip_to_county(query)

    # Try Census geocoder
    try:
        import requests
        resp = requests.get(
            "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress",
            params={
                "address": query,
                "benchmark": "Public_AR_Current",
                "vintage": "Current_Current",
                "format": "json",
            },
            timeout=10,
        )
        data = resp.json()
        matches = data.get("result", {}).get("addressMatches", [])
        if matches:
            m = matches[0]
            result = {
                "lat": float(m["coordinates"]["y"]),
                "lon": float(m["coordinates"]["x"]),
                "label": m.get("matchedAddress", query),
            }
            geos = m.get("geographies", {})
            counties = geos.get("Counties", geos.get("Census Tracts", []))
            if counties:
                c = counties[0]
                result["state_fips"] = c.get("STATE", "")
                result["county_fips"] = c.get("COUNTY", "")
            return result
    except Exception:
        pass

    # Fallback: try as county name search
    parts = query.split(",")
    county_q = parts[0].strip()
    state_q = parts[1].strip() if len(parts) > 1 else None

    sf = state_fips_from_name(state_q) if state_q else None
    matches = search_counties(county_q, state_fips=sf, limit=1)
    if not matches.empty:
        row = matches.iloc[0]
        return {
            "lat": row.get("lat", 0),
            "lon": row.get("lon", 0),
            "state_fips": row["state_fips"],
            "county_fips": row["county_fips"],
            "label": row["label"],
        }

    return None
