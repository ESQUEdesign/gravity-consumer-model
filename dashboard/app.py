"""
Gravity Consumer Model -- Interactive Dashboard
================================================
Analyze ANY US retail market using layered gravity models with live
Census + OpenStreetMap data. Demographic, competitive, and behavioral
overlays built in.

Run:
    python3 -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the gravity package is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gravity.data.schema import (
    Store, ConsumerOrigin,
    stores_to_dataframe, origins_to_dataframe, build_distance_matrix,
)
from gravity.core.huff import HuffModel
from gravity.data.census import CensusLoader
from gravity.data.geography import (
    STATE_FIPS, get_counties_for_state,
    county_bbox, geocode_zip as geo_geocode_zip,
)

# Lazy imports for optional modules
try:
    from gravity.core.mci import MCIModel
except Exception:
    MCIModel = None

try:
    from gravity.core.count_model import CountModel
except Exception:
    CountModel = None

try:
    from gravity.core.hierarchical_bayes import HierarchicalBayesMNL
except Exception:
    HierarchicalBayesMNL = None

try:
    from gravity.ensemble.conformal import ConformalPredictor
except Exception:
    ConformalPredictor = None

try:
    from gravity.core.competing_destinations import CompetingDestinationsModel
    _CD_OK = True
except Exception:
    _CD_OK = False

try:
    from gravity.spatial.trade_area import TradeAreaAnalyzer
    _TA_OK = True
except Exception:
    _TA_OK = False

try:
    from gravity.data.enrichment import (
        compute_composite_attractiveness,
        get_distance_matrix as pipeline_distance_matrix,
    )
    _ENRICHMENT_OK = True
except Exception:
    _ENRICHMENT_OK = False

try:
    from gravity.data.store_size import estimate_store_size
    _STORE_SIZE_OK = True
except Exception:
    _STORE_SIZE_OK = False

try:
    from gravity.data.bls import BLSLoader
    _BLS_OK = True
except Exception:
    _BLS_OK = False

try:
    from gravity.segmentation.census_psychographics import (
        CensusPsychographicClassifier, SEGMENT_PROFILES,
    )
    _PSYCHO_OK = True
except Exception:
    _PSYCHO_OK = False

try:
    from gravity.data.census_expanded import CensusExpandedLoader
    _CBP_OK = True
except Exception:
    _CBP_OK = False

try:
    from gravity.data.sec_edgar import SECEdgarLoader
    _SEC_OK = True
except Exception:
    _SEC_OK = False

try:
    from gravity.data.healthdata import HealthDataLoader
    _HEALTH_OK = True
except Exception:
    _HEALTH_OK = False

try:
    from gravity.data.fred import FREDLoader
    _FRED_OK = True
except Exception:
    _FRED_OK = False

try:
    from gravity.data.data_commons import DataCommonsLoader
    _DC_OK = True
except Exception:
    _DC_OK = False

try:
    from gravity.data.fmp import FinancialModelingPrepLoader
    _FMP_OK = True
except Exception:
    _FMP_OK = False

try:
    from gravity.data.ckan_datagov import CKANDataGovLoader
    _CKAN_OK = True
except Exception:
    _CKAN_OK = False

try:
    from gravity.data.noaa_climate import NOAAClimateLoader
    _NOAA_OK = True
except Exception:
    _NOAA_OK = False


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Gravity Consumer Model — Esque",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

_APP_VERSION = "2.1.0"  # expanded data sources

# ---------------------------------------------------------------------------
# Esque design system — Plotly global template
# ---------------------------------------------------------------------------
import plotly.graph_objects as go
import plotly.io as pio

_esque_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#0F0F0D",
        plot_bgcolor="#181815",
        font=dict(family="DM Sans, sans-serif", color="#EDE8E0", size=13),
        title=dict(font=dict(family="Cormorant Garamond, Georgia, serif", size=24, color="#F7F3EC")),
        colorway=["#C4A05A", "#7A8470", "#B85C3A", "#A89F92",
                   "#D9BC81", "#A3AE9C", "#8F7038", "#6B6A62"],
        xaxis=dict(
            gridcolor="rgba(54,54,48,0.6)", zerolinecolor="#363630",
            title=dict(font=dict(family="Raleway, sans-serif", size=11, color="#A89F92")),
            tickfont=dict(family="DM Sans, sans-serif", size=11, color="#6B6A62"),
        ),
        yaxis=dict(
            gridcolor="rgba(54,54,48,0.6)", zerolinecolor="#363630",
            title=dict(font=dict(family="Raleway, sans-serif", size=11, color="#A89F92")),
            tickfont=dict(family="DM Sans, sans-serif", size=11, color="#6B6A62"),
        ),
        legend=dict(font=dict(family="Raleway, sans-serif", size=11, color="#A89F92")),
    )
)
pio.templates["esque"] = _esque_template
pio.templates.default = "esque"

# Esque segment color overrides
_ESQUE_SEGMENT_COLORS = {
    "UP": "#C4A05A",
    "SF": "#7A8470",
    "AE": "#D9BC81",
    "CT": "#B85C3A",
    "RT": "#8F7038",
    "UE": "#A89F92",
    "SM": "#6B6A62",
    "MU": "#A3AE9C",
    "YM": "#C4A05A",
    "WE": "#363630",
    "GC": "#6B6A62",
    "RL": "#D9BC81",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CATEGORY_MAP = {
    "supermarket": "grocery", "convenience": "convenience",
    "department_store": "department", "clothes": "apparel",
    "shoes": "apparel", "electronics": "electronics",
    "furniture": "furniture", "hardware": "hardware",
    "bakery": "food_specialty", "butcher": "food_specialty",
    "greengrocer": "food_specialty", "mall": "mall",
    "general": "general", "variety_store": "variety",
    "alcohol": "liquor", "beverages": "beverages",
}

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
API_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def filter_by_zip(df, zip_lat, zip_lon, radius_km):
    dists = df.apply(lambda r: _haversine_km(zip_lat, zip_lon, r["lat"], r["lon"]), axis=1)
    return df[dists <= radius_km].copy()


# ---------------------------------------------------------------------------
# Consumer filter helpers
# ---------------------------------------------------------------------------

_AGE_BANDS = {
    "Under 18": ["age_under_18"],
    "18-24": ["age_18_24"],
    "25-34": ["age_25_34"],
    "35-44": ["age_35_44"],
    "45-54": ["age_45_54"],
    "55-64": ["age_55_64"],
    "65+": ["age_65_plus"],
}


def _dominant_age_band(demo: dict) -> str:
    """Return the age band label with the highest population."""
    best_label, best_val = "25-34", 0
    for label, keys in _AGE_BANDS.items():
        val = sum(demo.get(k, 0) for k in keys)
        if val > best_val:
            best_val = val
            best_label = label
    return best_label


def _housing_tenure_label(demo: dict) -> str:
    own = demo.get("pct_owner_occupied", 0.5)
    rent = demo.get("pct_renter_occupied", 0.5)
    if own >= 0.6:
        return "Owner-dominated"
    elif rent >= 0.6:
        return "Renter-dominated"
    return "Mixed"


def _get_all_retail_affinities() -> list[str]:
    """Collect unique retail affinities from all segment profiles."""
    if not _PSYCHO_OK:
        return []
    affinities = set()
    for prof in SEGMENT_PROFILES.values():
        for a in prof.get("consumer_behavior", {}).get("retail_affinity", []):
            affinities.add(a)
    return sorted(affinities)


def _segments_matching_behavior(field: str, values: list[str]) -> set[str]:
    """Return segment codes whose consumer_behavior[field] is in values."""
    if not _PSYCHO_OK or not values:
        return set()
    codes = set()
    for code, prof in SEGMENT_PROFILES.items():
        cb = prof.get("consumer_behavior", {})
        val = cb.get(field)
        if isinstance(val, list):
            if any(v in values for v in val):
                codes.add(code)
        elif val in values:
            codes.add(code)
    return codes


def apply_consumer_filters(origins_df, psycho_df, filters):
    """Return boolean mask on origins_df for active consumer filters."""
    mask = pd.Series(True, index=origins_df.index)

    # Income range
    inc_min = filters.get("income_min")
    inc_max = filters.get("income_max")
    if inc_min is not None and inc_max is not None:
        inc = origins_df["median_income"].fillna(0)
        mask &= (inc >= inc_min) & (inc <= inc_max)

    # Dominant age band
    age_bands = filters.get("age_bands", [])
    if age_bands:
        dom = origins_df["demographics"].apply(
            lambda d: _dominant_age_band(d) if isinstance(d, dict) else "25-34"
        )
        mask &= dom.isin(age_bands)

    # Housing tenure
    tenure = filters.get("tenure", [])
    if tenure:
        ten_labels = origins_df["demographics"].apply(
            lambda d: _housing_tenure_label(d) if isinstance(d, dict) else "Mixed"
        )
        mask &= ten_labels.isin(tenure)

    # Lifestyle segments
    segments = filters.get("segments", [])
    if segments and psycho_df is not None and not psycho_df.empty:
        seg_codes = psycho_df.reindex(origins_df.index)["segment_code"]
        seg_names = {code: SEGMENT_PROFILES[code]["name"]
                     for code in SEGMENT_PROFILES}
        name_to_code = {v: k for k, v in seg_names.items()}
        selected_codes = {name_to_code.get(s, s) for s in segments}
        mask &= seg_codes.isin(selected_codes)

    # Price sensitivity (behavioral → maps to segments)
    price_sens = filters.get("price_sensitivity", [])
    if price_sens:
        matching = _segments_matching_behavior("price_sensitivity", price_sens)
        if psycho_df is not None and not psycho_df.empty:
            seg_codes = psycho_df.reindex(origins_df.index)["segment_code"]
            mask &= seg_codes.isin(matching)

    # Channel preference
    channel = filters.get("channel_preference", [])
    if channel:
        matching = _segments_matching_behavior("channel_preference", channel)
        if psycho_df is not None and not psycho_df.empty:
            seg_codes = psycho_df.reindex(origins_df.index)["segment_code"]
            mask &= seg_codes.isin(matching)

    # Retail affinity
    affinity = filters.get("retail_affinity", [])
    if affinity:
        matching = _segments_matching_behavior("retail_affinity", affinity)
        if psycho_df is not None and not psycho_df.empty:
            seg_codes = psycho_df.reindex(origins_df.index)["segment_code"]
            mask &= seg_codes.isin(matching)

    return mask


def recalculate_from_filtered(prob_df, origins_df_filtered, stores_df):
    """Re-rank stores using only filtered origins' population."""
    prob_aligned = prob_df.reindex(index=origins_df_filtered.index).fillna(0)
    pop_w = origins_df_filtered["population"].reindex(prob_aligned.index).fillna(0).astype(float)
    demand = prob_aligned.mul(pop_w, axis=0).sum(axis=0)
    total = demand.sum()
    shares = (demand / total).sort_values(ascending=False) if total > 0 else demand
    store_results = pd.DataFrame({
        "name": stores_df["name"], "category": stores_df["category"],
        "lat": stores_df["lat"], "lon": stores_df["lon"],
        "demand": demand, "share": shares,
    }).sort_values("share", ascending=False)
    hhi = float((shares ** 2).sum() * 10_000)
    top5 = float(shares.head(5).sum())
    return store_results, hhi, top5


def count_active_filters(filters):
    """Count how many filter dimensions are active."""
    count = 0
    if filters.get("income_min") is not None:
        inc_min, inc_max = filters["income_min"], filters["income_max"]
        if inc_min > 0 or inc_max < 250_000:
            count += 1
    for key in ["age_bands", "tenure", "segments", "price_sensitivity",
                "channel_preference", "retail_affinity"]:
        if filters.get(key):
            count += 1
    return count


def _clean_str(val):
    """Return None if val is NaN/None, else str."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    return s if s else None


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        import math
        return v if not math.isnan(v) else default
    except (TypeError, ValueError):
        return default


def _sf(val, fmt=",", default=0.0):
    """Safe format: convert val to float then apply f-string format."""
    v = _safe_float(val, default)
    if fmt == ",":
        return f"{v:,}"
    elif fmt == ",.0f":
        return f"{v:,.0f}"
    elif fmt == ".0f":
        return f"{v:.0f}"
    elif fmt == ".1f":
        return f"{v:.1f}"
    elif fmt == ".2f":
        return f"{v:.2f}"
    elif fmt == ".4f":
        return f"{v:.4f}"
    elif fmt == ".4%":
        return f"{v:.4%}"
    elif fmt == ".2%":
        return f"{v:.2%}"
    elif fmt == "+.1f":
        return f"{v:+.1f}"
    else:
        return f"{v}"


def aggregate_demographics(origins_df):
    """Sum demographic counts across all origins."""
    age_keys = ["age_under_18", "age_18_24", "age_25_34", "age_35_44",
                "age_45_54", "age_55_64", "age_65_plus"]
    race_keys = ["white_non_hispanic", "black_non_hispanic",
                 "asian_non_hispanic", "hispanic_latino"]
    result = {}
    if "demographics" not in origins_df.columns:
        return result
    for key in age_keys + race_keys:
        result[key] = sum(
            d.get(key, 0) for d in origins_df["demographics"]
            if isinstance(d, dict)
        )
    return result


# ---------------------------------------------------------------------------
# Data fetching (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def geocode_zip_cached(zipcode):
    return geo_geocode_zip(zipcode)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_census(state_fips: str, county_fips: str, bbox: tuple) -> pd.DataFrame:
    """Fetch census block groups with real centroids and full demographics."""
    loader = CensusLoader()
    origins = loader.load_block_groups(state_fips, county=county_fips, include_centroids=True)
    if not origins:
        return pd.DataFrame()

    df = CensusLoader.to_dataframe(origins)

    # Filter to bbox and non-zero population
    south, west, north, east = bbox
    # Keep origins with real coordinates (non-zero)
    has_coords = (df["lat"] != 0.0) | (df["lon"] != 0.0)
    in_bbox = (
        (df["lat"] >= south) & (df["lat"] <= north) &
        (df["lon"] >= west) & (df["lon"] <= east)
    )
    has_pop = df["population"] > 0

    # If we have real coords, filter to bbox; otherwise keep all with pop > 0
    if has_coords.sum() > 0:
        df = df[has_coords & in_bbox & has_pop].copy()
    else:
        df = df[has_pop].copy()

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_osm_stores(bbox: tuple) -> pd.DataFrame:
    import requests
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:{API_TIMEOUT}][maxsize:5000000];
    (node["shop"]({south},{west},{north},{east}););
    out body 500;
    """
    resp = None
    for url in OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": query}, timeout=API_TIMEOUT)
            resp.raise_for_status()
            break
        except Exception:
            continue
    if resp is None:
        raise RuntimeError("All Overpass API servers timed out. Try a smaller county or increase radius.")
    resp.raise_for_status()
    stores = []
    for elem in resp.json().get("elements", []):
        tags = elem.get("tags", {})
        lat, lon = elem.get("lat"), elem.get("lon")
        if lat is None or lon is None:
            continue
        shop_type = tags.get("shop", tags.get("amenity", "other"))
        name = tags.get("name")
        brand = tags.get("brand")
        stores.append({
            "store_id": f"osm_{elem.get('type', 'node')}_{elem.get('id', 0)}",
            "name": name if name else f"({shop_type})",
            "lat": float(lat), "lon": float(lon),
            "category": _CATEGORY_MAP.get(shop_type, shop_type),
            "brand": brand,
            "square_footage": 0.0,
            "avg_rating": 0.0,
            "price_level": 2,
        })

    if not stores:
        return pd.DataFrame()

    df = pd.DataFrame(stores).set_index("store_id")

    # Estimate store sizes from brand/category lookup
    if _STORE_SIZE_OK:
        store_models = [
            Store(store_id=str(sid), name=row["name"], lat=row["lat"], lon=row["lon"],
                  category=row["category"], brand=_clean_str(row.get("brand")),
                  square_footage=0.0)
            for sid, row in df.iterrows()
        ]
        estimate_store_size(store_models)
        for sm in store_models:
            df.loc[sm.store_id, "square_footage"] = sm.square_footage
            df.loc[sm.store_id, "sqft_source"] = sm.attributes.get("sqft_source", "default")
    else:
        df["square_footage"] = 1000.0
        df["sqft_source"] = "default"

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bls(state_fips: str, county_fips: str) -> tuple:
    """Fetch BLS employment data. Returns (county_data, retail_data)."""
    if not _BLS_OK:
        return {}, {}
    try:
        loader = BLSLoader()
        county_data = loader.get_county_employment(state_fips, county_fips)
        retail_data = loader.get_retail_employment(state_fips, county_fips)
        return county_data, retail_data
    except Exception:
        return {}, {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cbp(state_fips: str, county_fips: str) -> dict:
    """Fetch Census County Business Patterns data."""
    if not _CBP_OK:
        return {}
    try:
        loader = CensusExpandedLoader()
        return loader.get_county_business_patterns(state_fips, county_fips)
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sec_benchmarks(brands: tuple) -> list:
    """Fetch SEC EDGAR financials for public retail brands."""
    if not _SEC_OK or not brands:
        return []
    try:
        loader = SECEdgarLoader()
        return loader.benchmark_competitors(list(brands))
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_health(state_fips: str, county_fips: str) -> dict:
    """Fetch CDC/HHS health context data."""
    if not _HEALTH_OK:
        return {}
    try:
        loader = HealthDataLoader()
        return loader.get_health_context(state_fips, county_fips)
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fred(api_key: str) -> dict:
    """Fetch FRED macro economic indicators."""
    if not _FRED_OK or not api_key:
        return {}
    try:
        loader = FREDLoader(api_key=api_key)
        return loader.get_retail_context()
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data_commons(state_fips: str, county_fips: str, api_key: str) -> dict:
    """Fetch Data Commons demographic/housing/economic indicators."""
    if not _DC_OK or not api_key:
        return {}
    try:
        loader = DataCommonsLoader(api_key=api_key)
        return loader.get_county_context(state_fips, county_fips)
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fmp_benchmarks(brands: tuple, api_key: str) -> list:
    """Fetch Financial Modeling Prep retailer benchmarks."""
    if not _FMP_OK or not api_key or not brands:
        return []
    try:
        loader = FinancialModelingPrepLoader(api_key=api_key)
        return loader.benchmark_retail(list(brands))
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ckan_datasets(state_name: str, county_name: str) -> dict:
    """Search data.gov for available government datasets."""
    if not _CKAN_OK:
        return {}
    try:
        loader = CKANDataGovLoader()
        return loader.summarize_available("", "", state_name, county_name)
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_noaa_climate(state_fips: str, county_fips: str, token: str) -> dict:
    """Fetch NOAA climate context for a county."""
    if not _NOAA_OK or not token:
        return {}
    try:
        loader = NOAAClimateLoader(token=token)
        return loader.get_climate_context(state_fips, county_fips)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_huff(origins_df, stores_df, alpha, lam, dist_matrix=None):
    if dist_matrix is None:
        dist_matrix = build_distance_matrix(origins_df, stores_df)
    model = HuffModel(alpha=alpha, lam=lam, attractiveness="square_footage",
                      distance_matrix=dist_matrix)
    prob_df = model.predict(origins_df, stores_df)

    pop_w = origins_df["population"].reindex(prob_df.index).fillna(0).astype(float)
    weighted_demand = prob_df.mul(pop_w, axis=0).sum(axis=0)
    total = weighted_demand.sum()
    shares = (weighted_demand / total).sort_values(ascending=False) if total > 0 else weighted_demand

    store_results = pd.DataFrame({
        "name": stores_df["name"], "category": stores_df["category"],
        "lat": stores_df["lat"], "lon": stores_df["lon"],
        "demand": weighted_demand, "share": shares,
    }).sort_values("share", ascending=False)

    hhi = float((shares ** 2).sum() * 10_000)
    top5 = float(shares.head(5).sum())
    avg_dist = float(dist_matrix.values.mean())

    return {
        "prob_df": prob_df, "store_results": store_results,
        "dist_matrix": dist_matrix, "hhi": hhi,
        "top5_concentration": top5, "avg_distance_km": avg_dist,
        "total_demand": total, "model_name": "Huff",
    }


def run_mci(origins_df, stores_df, dist_matrix):
    if MCIModel is None:
        return None
    try:
        model = MCIModel(variables=["square_footage"], include_distance=True)
        model.fit(origins_df, stores_df, dist_matrix)
        prob_df = model.predict(origins_df, stores_df, dist_matrix)
        return {"prob_df": prob_df, "model_name": "MCI",
                "coefficients": model.coefficients_summary,
                "r_squared": getattr(model, "_r_squared", None)}
    except Exception as e:
        return {"error": str(e), "model_name": "MCI"}


def run_count_model(origins_df, stores_df, dist_matrix):
    if CountModel is None:
        return None
    try:
        model = CountModel(model_type="negative_binomial", zero_inflated=False)
        model.fit(origins_df, stores_df, dist_matrix)
        prob_df = model.predict_proba(origins_df, stores_df, dist_matrix)
        return {"prob_df": prob_df, "model_name": "NegBin", "summary": model.summary}
    except Exception as e:
        return {"error": str(e), "model_name": "NegBin"}


def run_conformal(huff_prob):
    if ConformalPredictor is None:
        return None
    try:
        cp = ConformalPredictor(confidence=0.90, method="split")
        n = huff_prob.shape[0]
        cal_n = max(10, n // 5)
        cal_idx = huff_prob.index[:cal_n]
        noise = np.random.RandomState(42).normal(0, 0.01, huff_prob.loc[cal_idx].values.shape)
        pseudo_observed = np.clip(huff_prob.loc[cal_idx].values + noise, 0, 1)
        cp.calibrate(pseudo_observed.ravel(), huff_prob.loc[cal_idx].values.ravel())
        lower, upper = cp.predict_interval(huff_prob.values.ravel())
        lower_df = pd.DataFrame(lower.reshape(huff_prob.shape), index=huff_prob.index, columns=huff_prob.columns)
        upper_df = pd.DataFrame(upper.reshape(huff_prob.shape), index=huff_prob.index, columns=huff_prob.columns)
        return {"lower": lower_df, "upper": upper_df, "model_name": "Conformal"}
    except Exception as e:
        return {"error": str(e), "model_name": "Conformal"}


def run_competing_destinations(origins_df, stores_df, alpha, lam):
    if not _CD_OK:
        return None
    try:
        cd = CompetingDestinationsModel(alpha=alpha, lam=lam, delta=0.5, beta=1.0, radius_km=30.0)
        prob_df = cd.predict(origins_df, stores_df)
        cd_index = cd.compute_clustering_index(stores_df)
        return {"prob_df": prob_df, "cd_index": cd_index, "model_name": "CD"}
    except Exception as e:
        return {"error": str(e), "model_name": "CD"}


def compute_ensemble(model_results, origins_df, stores_df):
    prob_dfs, names = [], []
    for name, res in model_results.items():
        if res and "prob_df" in res and "error" not in res:
            prob_dfs.append(res["prob_df"])
            names.append(name)
    if not prob_dfs:
        return None

    common_idx = prob_dfs[0].index
    common_cols = prob_dfs[0].columns
    for pdf in prob_dfs[1:]:
        common_idx = common_idx.intersection(pdf.index)
        common_cols = common_cols.intersection(pdf.columns)

    aligned = [pdf.loc[common_idx, common_cols] for pdf in prob_dfs]
    ensemble_prob = sum(aligned) / len(aligned)

    pop_w = origins_df["population"].reindex(ensemble_prob.index).fillna(0).astype(float)
    demand = ensemble_prob.mul(pop_w, axis=0).sum(axis=0)
    total = demand.sum()
    shares = (demand / total).sort_values(ascending=False) if total > 0 else demand

    store_results = pd.DataFrame({
        "name": stores_df.loc[common_cols, "name"],
        "category": stores_df.loc[common_cols, "category"],
        "lat": stores_df.loc[common_cols, "lat"],
        "lon": stores_df.loc[common_cols, "lon"],
        "demand": demand, "share": shares,
    }).sort_values("share", ascending=False)

    hhi = float((shares ** 2).sum() * 10_000)
    top5 = float(shares.head(5).sum())

    return {
        "prob_df": ensemble_prob, "store_results": store_results,
        "hhi": hhi, "top5_concentration": top5,
        "models_used": names, "n_models": len(names),
    }


# ---------------------------------------------------------------------------
# Market intelligence engine
# ---------------------------------------------------------------------------

# National avg stores per 10K people by category (ICSC / Census CBP benchmarks)
_STORES_PER_10K = {
    "grocery": 2.5, "convenience": 5.0, "department": 0.4,
    "apparel": 3.0, "electronics": 0.8, "furniture": 1.2,
    "hardware": 1.5, "food_specialty": 2.0, "general": 2.0,
    "variety": 1.8, "liquor": 1.5, "beverages": 1.0, "mall": 0.15,
}

# Avg annual rent per sqft by category (CBRE / CoStar national avg)
_AVG_RENT_SQFT = {
    "grocery": 14, "convenience": 22, "department": 10,
    "apparel": 28, "electronics": 24, "furniture": 12,
    "hardware": 12, "food_specialty": 30, "general": 16,
    "variety": 14, "liquor": 20, "beverages": 25, "mall": 35,
}

# Marketing channel recommendations by segment behavior
_MARKETING_CHANNELS = {
    "in-store": ["in-store signage", "loyalty programs", "local newspaper", "direct mail"],
    "online": ["social media ads", "email marketing", "influencer partnerships", "SEO/SEM"],
    "omnichannel": ["social media", "email + in-store promos", "mobile app", "click-and-collect"],
}

_PRICE_MESSAGING = {
    "low": "Premium quality, curated experiences, exclusivity",
    "moderate": "Everyday value, trusted brands, family-friendly",
    "high": "Deep discounts, BOGO, clearance events, price-match guarantees",
    "very high": "Dollar deals, bulk savings, essential value packs",
}


def compute_site_viability(total_pop, avg_income, n_stores, category,
                           hhi, fit_score, pop_3mi):
    """Compute 0-100 site viability score with component breakdown."""
    scores = {}

    # 1. Population density (0-20): is there enough demand?
    if pop_3mi >= 50_000:
        scores["population"] = 20
    elif pop_3mi >= 20_000:
        scores["population"] = 15
    elif pop_3mi >= 10_000:
        scores["population"] = 10
    elif pop_3mi >= 5_000:
        scores["population"] = 6
    else:
        scores["population"] = 3

    # 2. Income fit (0-20): can they afford the category?
    cat_spend = _CATEGORY_SPEND.get(category, 2000)
    income_ratio = avg_income / _NATIONAL_MEDIAN_INCOME if avg_income > 0 else 0.5
    if income_ratio >= 1.2:
        scores["income"] = 20
    elif income_ratio >= 0.9:
        scores["income"] = 16
    elif income_ratio >= 0.7:
        scores["income"] = 10
    else:
        scores["income"] = 5

    # 3. Competition headroom (0-20): is the market oversaturated?
    expected = _STORES_PER_10K.get(category, 2.0) * total_pop / 10_000
    same_cat = n_stores  # count of same-category stores
    saturation = same_cat / expected if expected > 0 else 2.0
    if saturation <= 0.5:
        scores["competition"] = 20  # very underserved
    elif saturation <= 0.8:
        scores["competition"] = 16
    elif saturation <= 1.2:
        scores["competition"] = 10  # balanced
    elif saturation <= 1.5:
        scores["competition"] = 5
    else:
        scores["competition"] = 2   # oversaturated

    # 4. Category-audience fit (0-20)
    if fit_score is not None:
        scores["category_fit"] = min(20, int(fit_score / 5))
    else:
        scores["category_fit"] = 10  # neutral if unknown

    # 5. Market health (0-20): competitive market is better
    if hhi < 15:
        scores["market_health"] = 18  # healthy competition
    elif hhi < 25:
        scores["market_health"] = 12
    elif hhi < 40:
        scores["market_health"] = 8
    else:
        scores["market_health"] = 4  # monopoly risk

    total = sum(scores.values())

    if total >= 80:
        grade, verdict = "A", "Strong opportunity"
    elif total >= 65:
        grade, verdict = "B", "Good opportunity with manageable risks"
    elif total >= 50:
        grade, verdict = "C", "Moderate opportunity — proceed with caution"
    elif total >= 35:
        grade, verdict = "D", "Weak opportunity — significant headwinds"
    else:
        grade, verdict = "F", "Not recommended — critical issues"

    return {
        "total": total, "grade": grade, "verdict": verdict,
        "components": scores, "saturation": saturation,
    }


def compute_market_gaps(stores_df, total_pop):
    """Identify underserved and oversaturated categories."""
    gaps = []
    cat_counts = stores_df["category"].value_counts()
    for cat, benchmark in _STORES_PER_10K.items():
        expected = benchmark * total_pop / 10_000
        actual = cat_counts.get(cat, 0)
        ratio = actual / expected if expected > 0 else 0
        spend = _CATEGORY_SPEND.get(cat, 2000)
        tam = total_pop * spend
        gaps.append({
            "category": cat, "actual_stores": actual,
            "expected_stores": round(expected, 1),
            "saturation": round(ratio, 2),
            "tam": tam,
            "status": "Underserved" if ratio < 0.7 else "Oversaturated" if ratio > 1.3 else "Balanced",
        })
    return pd.DataFrame(gaps).sort_values("saturation")


def generate_executive_summary(market_label, total_pop, avg_income, n_stores,
                               hhi, dom_segment, dom_pct, gap_df,
                               viability):
    """Auto-generate executive summary for market analysis."""
    lines = []
    grade = viability["grade"]
    verdict = viability["verdict"]
    score = viability["total"]

    lines.append(f"**Market Grade: {grade} ({score}/100)** — {verdict}.")

    lines.append(
        f"{market_label} has a population of {total_pop:,} with a "
        f"median income of ${avg_income:,.0f} and {n_stores:,} retail locations."
    )

    if hhi < 15:
        lines.append("The market is highly competitive with no dominant players.")
    elif hhi < 25:
        lines.append("The market is moderately concentrated — a few chains hold significant share.")
    else:
        lines.append("The market is highly concentrated — dominated by a small number of retailers.")

    if dom_segment and dom_pct:
        lines.append(
            f"The primary consumer segment is **{dom_segment}** ({dom_pct:.0f}% of population)."
        )

    # Top opportunities
    if gap_df is not None and not gap_df.empty:
        underserved = gap_df[gap_df["status"] == "Underserved"].head(3)
        if not underserved.empty:
            cats = ", ".join(underserved["category"].tolist())
            lines.append(f"**Opportunity**: {cats} {'categories are' if len(underserved) > 1 else 'category is'} underserved relative to population.")
        oversat = gap_df[gap_df["status"] == "Oversaturated"].head(2)
        if not oversat.empty:
            cats = ", ".join(oversat["category"].tolist())
            lines.append(f"**Caution**: {cats} {'are' if len(oversat) > 1 else 'is'} oversaturated.")

    return " ".join(lines)


def build_marketing_playbook(segment_code):
    """Build actionable marketing recommendations for a segment."""
    if segment_code not in SEGMENT_PROFILES:
        return None
    prof = SEGMENT_PROFILES[segment_code]
    cb = prof.get("consumer_behavior", {})
    channel_pref = cb.get("channel_preference", "in-store")
    price_sens = cb.get("price_sensitivity", "moderate")
    channels = _MARKETING_CHANNELS.get(channel_pref, _MARKETING_CHANNELS["in-store"])
    messaging = _PRICE_MESSAGING.get(price_sens, _PRICE_MESSAGING["moderate"])
    freq = cb.get("shopping_frequency", "moderate")
    loyalty = cb.get("brand_loyalty", "moderate")

    promo_cadence = {
        "high": "Weekly promotions and flash sales",
        "moderate": "Bi-weekly or monthly campaigns",
        "low": "Seasonal events and milestone offers",
    }

    loyalty_strategy = {
        "very high": "Reinforce with loyalty rewards — they rarely switch. Focus on retention.",
        "high": "Loyalty programs with tiered rewards. They respond to exclusive member benefits.",
        "moderate": "Mix of loyalty perks and competitive pricing. They'll compare but prefer convenience.",
        "low": "Price and novelty drive decisions. Use limited-time offers and trend-driven assortment.",
    }

    return {
        "segment_name": prof["name"],
        "description": prof["description"],
        "who_they_are": prof["description"].split(".")[0] + ".",
        "channels": channels,
        "messaging": messaging,
        "promo_cadence": promo_cadence.get(freq, promo_cadence["moderate"]),
        "loyalty_strategy": loyalty_strategy.get(loyalty, loyalty_strategy["moderate"]),
        "retail_affinity": cb.get("retail_affinity", []),
        "price_sensitivity": price_sens,
        "shopping_frequency": freq,
        "channel_preference": channel_pref,
    }


def compute_supportable_rent(annual_revenue, sqft, category):
    """Calculate max supportable rent based on industry occupancy ratios."""
    if annual_revenue <= 0 or sqft <= 0:
        return None
    # Retail occupancy cost should be 5-10% of revenue (ICSC benchmark)
    rent_low = annual_revenue * 0.05 / sqft
    rent_mid = annual_revenue * 0.08 / sqft
    rent_high = annual_revenue * 0.10 / sqft
    market_avg = _AVG_RENT_SQFT.get(category, 16)
    return {
        "rent_low": rent_low, "rent_mid": rent_mid, "rent_high": rent_high,
        "market_avg": market_avg,
        "can_afford_market": rent_mid >= market_avg,
    }


# ---------------------------------------------------------------------------
# Insights report builder
# ---------------------------------------------------------------------------

def build_insights_report(market_label, total_pop, total_hh, avg_income,
                          stores_df, store_results, active_results,
                          gap_df, viability, exec_summary,
                          psycho_df, psycho_summary,
                          bls_county, bls_retail,
                          origins_df, bbox, rings):
    """Synthesise pipeline outputs into a structured insights report dict."""
    report = {}

    # ── Header ──────────────────────────────────────────────────────
    report["grade"] = viability["grade"]
    report["score"] = viability["total"]
    report["verdict"] = viability["verdict"]
    report["components"] = viability.get("components", {})

    # ── Opportunity gaps ────────────────────────────────────────────
    underserved = gap_df[gap_df["status"] == "Underserved"].head(5).to_dict("records") if gap_df is not None and not gap_df.empty else []
    oversaturated = gap_df[gap_df["status"] == "Oversaturated"].head(3).to_dict("records") if gap_df is not None and not gap_df.empty else []
    report["underserved"] = underserved
    report["oversaturated"] = oversaturated

    # ── Consumer DNA: top 3 segments ────────────────────────────────
    consumer_dna = []
    if _PSYCHO_OK and psycho_summary is not None and not psycho_summary.empty:
        ps = psycho_summary.copy()
        if "population" in ps.columns:
            ps = ps.sort_values("population", ascending=False)
        for _, row in ps.head(3).iterrows():
            code = row.get("segment_code", "")
            pct = row.get("pct", row.get("population", 0) / max(total_pop, 1) * 100)
            playbook = build_marketing_playbook(code) if code else None
            consumer_dna.append({
                "code": code,
                "name": row.get("segment_name", code),
                "population": int(row.get("population", 0)),
                "pct": float(pct),
                "playbook": playbook,
            })
    report["consumer_dna"] = consumer_dna

    # ── Revenue potential per category ──────────────────────────────
    rev_potential = []
    if gap_df is not None and not gap_df.empty:
        for _, row in gap_df.iterrows():
            cat = row["category"]
            tam = row.get("tam", 0)
            actual = row.get("actual_stores", 0)
            sat = row.get("saturation", 1.0)
            # Rough new-entrant revenue: TAM / (actual + 1)
            est_rev = tam / (actual + 1) if tam > 0 else 0
            rent = compute_supportable_rent(est_rev, 5000, cat)
            rev_potential.append({
                "category": cat, "tam": tam, "actual_stores": actual,
                "saturation": sat, "status": row.get("status", "Balanced"),
                "est_new_entrant_rev": est_rev,
                "supportable_rent_mid": rent["rent_mid"] if rent else 0,
                "market_avg_rent": rent["market_avg"] if rent else 0,
            })
    report["revenue_potential"] = sorted(rev_potential, key=lambda x: x["saturation"])

    # ── Competitive landscape ───────────────────────────────────────
    hhi = active_results.get("hhi", 0)
    if hhi < 15:
        comp_level = "Low"
        comp_desc = "Highly competitive market with no dominant players."
    elif hhi < 25:
        comp_level = "Moderate"
        comp_desc = "Moderately concentrated — a few chains hold significant share."
    else:
        comp_level = "High"
        comp_desc = "Highly concentrated — dominated by a small number of retailers."

    top5_stores = []
    if store_results is not None and not store_results.empty:
        for _, row in store_results.head(5).iterrows():
            top5_stores.append({
                "name": row.get("name", ""),
                "category": row.get("category", ""),
                "share": float(row.get("share", 0)),
            })

    cat_counts = stores_df["category"].value_counts().head(8).to_dict() if stores_df is not None else {}
    report["competition"] = {
        "hhi": hhi, "level": comp_level, "description": comp_desc,
        "top5_stores": top5_stores,
        "top5_share": float(active_results.get("top5_concentration", 0)),
        "category_mix": cat_counts,
    }

    # ── Actionable recommendations ──────────────────────────────────
    recs = []
    income_ratio = avg_income / 75_149 if avg_income > 0 else 0.5

    # 1. Underserved category opportunities
    for gap in underserved[:2]:
        cat = gap["category"]
        tam_m = gap.get("tam", 0) / 1e6
        recs.append({
            "title": f"Enter the {cat} market",
            "detail": (f"Only {gap.get('actual_stores', 0)} stores vs "
                       f"{gap.get('expected_stores', 0)} expected. "
                       f"${tam_m:,.1f}M addressable market is underserved "
                       f"(saturation: {gap.get('saturation', 0):.0%})."),
        })

    # 2. Dominant segment alignment
    if consumer_dna:
        top_seg = consumer_dna[0]
        pb = top_seg.get("playbook")
        if pb:
            affinities = ", ".join(pb.get("retail_affinity", [])[:3])
            recs.append({
                "title": f"Align store format with {top_seg['name']} segment",
                "detail": (f"{top_seg['name']} represents {top_seg['pct']:.0f}% of the "
                           f"population. They prefer {pb.get('channel_preference', 'in-store')} "
                           f"shopping and are drawn to {affinities}."),
            })

    # 3. Price positioning
    if income_ratio >= 1.2:
        recs.append({
            "title": "Position for premium pricing",
            "detail": (f"Area median income (${avg_income:,.0f}) is {income_ratio:.0%} of "
                       f"national median. Consumers can support premium/specialty formats."),
        })
    elif income_ratio <= 0.7:
        recs.append({
            "title": "Lead with value positioning",
            "detail": (f"Area median income (${avg_income:,.0f}) is {income_ratio:.0%} of "
                       f"national median. Dollar, discount, and value formats perform best."),
        })
    else:
        recs.append({
            "title": "Target moderate price points",
            "detail": (f"Area median income (${avg_income:,.0f}) tracks near national median. "
                       f"Mid-tier formats with selective premium offerings fit this market."),
        })

    # 4. Competition strategy
    if hhi >= 25:
        recs.append({
            "title": "Differentiate aggressively",
            "detail": ("Market is highly concentrated. New entrants must offer a clearly "
                       "distinct value proposition — niche category, better experience, "
                       "or underserved format."),
        })
    elif hhi < 15:
        recs.append({
            "title": "Compete on convenience and location",
            "detail": ("Low concentration means many small players. Winning depends on "
                       "site selection and convenience rather than brand power."),
        })

    # 5. BLS workforce insight
    if bls_county and bls_county.get("avg_weekly_wage", 0) > 0:
        wage = bls_county["avg_weekly_wage"]
        emp = bls_county.get("total_employment", 0)
        recs.append({
            "title": "Leverage local workforce",
            "detail": (f"County has {emp:,.0f} employed workers at ${wage:,.0f} avg weekly wage. "
                       f"{'Tight labor market may raise staffing costs.' if emp > 0 and wage > 900 else 'Affordable labor pool for retail staffing.'}"),
        })

    # 6. Distance ring location insight
    pop_3mi = rings.get("3mi", {}).get("population", 0) if rings else 0
    pop_5mi = rings.get("5mi", {}).get("population", 0) if rings else 0
    if pop_3mi > 0:
        if pop_3mi < 10_000:
            recs.append({
                "title": "Consider highway-corridor or pass-through locations",
                "detail": (f"Only {pop_3mi:,} people within 3 miles of county center. "
                           f"Rural density requires high-visibility roadside placement to "
                           f"capture drive-by traffic."),
            })
        elif pop_3mi >= 30_000:
            recs.append({
                "title": "Target walkable/urban infill locations",
                "detail": (f"{pop_3mi:,} people within 3 miles supports dense-format retail. "
                           f"Look for mixed-use or downtown locations with foot traffic."),
            })

    # 7. Oversaturated category warning
    for osat in oversaturated[:1]:
        recs.append({
            "title": f"Avoid the {osat['category']} category",
            "detail": (f"Market has {osat.get('actual_stores', 0)} {osat['category']} stores vs "
                       f"{osat.get('expected_stores', 0)} expected — oversaturated at "
                       f"{osat.get('saturation', 0):.0%}. New entrants face intense competition."),
        })

    report["recommendations"] = recs

    # ── Distance rings ──────────────────────────────────────────────
    report["distance_rings"] = rings

    # ── BLS summary ─────────────────────────────────────────────────
    report["bls"] = {
        "total_employment": bls_county.get("total_employment", 0) if bls_county else 0,
        "avg_weekly_wage": bls_county.get("avg_weekly_wage", 0) if bls_county else 0,
        "retail_employment": bls_retail.get("total_employment", 0) if bls_retail else 0,
        "retail_wage": bls_retail.get("avg_weekly_wage", 0) if bls_retail else 0,
        "sectors": bls_county.get("sectors", []) if bls_county else [],
    }

    # ── Income range ────────────────────────────────────────────────
    inc = origins_df["median_income"]
    inc_valid = inc[inc > 0]
    report["income_stats"] = {
        "min": float(inc_valid.min()) if not inc_valid.empty else 0,
        "median": float(inc_valid.median()) if not inc_valid.empty else 0,
        "max": float(inc_valid.max()) if not inc_valid.empty else 0,
        "mean": float(avg_income) if avg_income > 0 else 0,
    }

    # ── Development analysis ─────────────────────────────────────────
    report["development"] = _build_development_analysis(
        total_pop, total_hh, avg_income, income_ratio,
        origins_df, consumer_dna, underserved, oversaturated,
        gap_df, rings, bls_county, hhi,
    )

    return report


# ---------------------------------------------------------------------------
# Development analysis engine
# ---------------------------------------------------------------------------

# Segment → residential product affinity
_SEGMENT_HOUSING = {
    "UP": {"types": ["luxury apartments", "urban lofts", "mixed-use condos"],
           "price_tier": "premium", "density": "high",
           "amenities": ["rooftop lounge", "coworking space", "bike storage", "EV charging"]},
    "SF": {"types": ["single-family homes", "townhomes", "master-planned communities"],
           "price_tier": "moderate", "density": "low",
           "amenities": ["playground", "community pool", "walking trails", "sports courts"]},
    "AE": {"types": ["luxury single-family", "estate homes", "active-adult communities"],
           "price_tier": "premium", "density": "low",
           "amenities": ["golf course", "clubhouse", "wine cellar", "concierge"]},
    "CT": {"types": ["student housing", "micro-apartments", "co-living"],
           "price_tier": "budget", "density": "high",
           "amenities": ["study rooms", "high-speed wifi", "laundry", "package lockers"]},
    "RT": {"types": ["manufactured homes", "starter homes", "rural single-family"],
           "price_tier": "budget", "density": "very low",
           "amenities": ["large lots", "garages", "storage", "outdoor space"]},
    "UE": {"types": ["affordable apartments", "workforce housing", "LIHTC developments"],
           "price_tier": "budget", "density": "high",
           "amenities": ["laundry facilities", "community room", "transit access", "daycare"]},
    "SM": {"types": ["suburban townhomes", "garden-style apartments", "entry-level single-family"],
           "price_tier": "moderate", "density": "medium",
           "amenities": ["fitness center", "dog park", "community garden", "picnic area"]},
    "MU": {"types": ["mixed-income apartments", "live-work units", "adaptive reuse lofts"],
           "price_tier": "moderate", "density": "high",
           "amenities": ["cultural space", "multilingual signage", "community kitchen", "transit access"]},
    "YM": {"types": ["luxury apartments", "build-to-rent townhomes", "flex-living suites"],
           "price_tier": "moderate-premium", "density": "medium-high",
           "amenities": ["coworking", "fitness center", "package lockers", "pet spa"]},
    "WE": {"types": ["single-family homes", "large-lot homes", "hobby farm parcels"],
           "price_tier": "moderate", "density": "very low",
           "amenities": ["oversized garages", "workshops", "RV parking", "acreage"]},
    "GC": {"types": ["age-restricted communities", "patio homes", "ranch-style downsizers"],
           "price_tier": "moderate", "density": "low",
           "amenities": ["single-story", "low maintenance", "clubhouse", "walking paths"]},
    "RL": {"types": ["55+ active-adult", "assisted living", "independent living", "continuing care"],
           "price_tier": "moderate-premium", "density": "medium",
           "amenities": ["healthcare access", "dining hall", "shuttle service", "social programming"]},
}

# Income → housing price points (2024 national benchmarks)
_INCOME_TO_HOME_PRICE = {
    "very_low":   {"label": "< $35K",     "max_home": 125_000, "max_rent": 875,   "tier": "Affordable / workforce"},
    "low":        {"label": "$35K–$50K",   "max_home": 175_000, "max_rent": 1_250,  "tier": "Entry-level"},
    "moderate":   {"label": "$50K–$75K",   "max_home": 265_000, "max_rent": 1_875,  "tier": "Mid-market"},
    "upper_mid":  {"label": "$75K–$100K",  "max_home": 350_000, "max_rent": 2_500,  "tier": "Move-up"},
    "high":       {"label": "$100K–$150K", "max_home": 525_000, "max_rent": 3_750,  "tier": "Premium"},
    "very_high":  {"label": "$150K+",      "max_home": 750_000, "max_rent": 5_000,  "tier": "Luxury"},
}

# Segment → retail anchor recommendations
_SEGMENT_RETAIL_ANCHORS = {
    "UP": ["Whole Foods / Trader Joe's", "boutique fitness", "specialty coffee", "wine bar"],
    "SF": ["Kroger / Publix", "Target", "Chick-fil-A", "urgent care", "children's enrichment"],
    "AE": ["upscale dining", "specialty wine/spirits", "home furnishing", "professional services"],
    "CT": ["fast casual restaurants", "coffee shops", "thrift / resale", "phone repair"],
    "RT": ["Dollar General", "Tractor Supply", "AutoZone", "local diner", "hardware"],
    "UE": ["Dollar Tree", "check cashing / payday", "laundromat", "convenience / bodega"],
    "SM": ["grocery chain", "fast food corridor", "nail salon / barbershop", "mid-tier apparel"],
    "MU": ["ethnic grocery / market", "restaurant row", "wireless store", "remittance / money transfer"],
    "YM": ["fast casual", "boutique fitness", "pet store", "coworking / flex office"],
    "WE": ["Home Depot / Lowe's", "auto parts", "sporting goods", "BBQ / casual dining"],
    "GC": ["pharmacy (CVS/Walgreens)", "medical office", "casual dining", "grocery"],
    "RL": ["pharmacy", "medical/dental office", "casual dining", "grocery delivery hub"],
}


def _income_tier(avg_income):
    """Map average income to a tier key."""
    if avg_income < 35_000:
        return "very_low"
    elif avg_income < 50_000:
        return "low"
    elif avg_income < 75_000:
        return "moderate"
    elif avg_income < 100_000:
        return "upper_mid"
    elif avg_income < 150_000:
        return "high"
    return "very_high"


def _build_development_analysis(total_pop, total_hh, avg_income, income_ratio,
                                 origins_df, consumer_dna, underserved,
                                 oversaturated, gap_df, rings, bls_county, hhi):
    """Generate recommended residential, retail, and mixed-use development analysis."""
    dev = {}

    tier_key = _income_tier(avg_income)
    price_info = _INCOME_TO_HOME_PRICE[tier_key]

    # ── Housing demand indicators ────────────────────────────────
    # Owner vs renter split from demographics
    own_pcts, rent_pcts = [], []
    if "demographics" in origins_df.columns:
        for d in origins_df["demographics"]:
            if isinstance(d, dict):
                own_pcts.append(d.get("pct_owner_occupied", 0.5))
                rent_pcts.append(d.get("pct_renter_occupied", 0.5))
    avg_own = sum(own_pcts) / len(own_pcts) if own_pcts else 0.5
    avg_rent = sum(rent_pcts) / len(rent_pcts) if rent_pcts else 0.5

    # Age composition for housing demand
    age_young, age_mid, age_old = 0, 0, 0
    if "demographics" in origins_df.columns:
        for d in origins_df["demographics"]:
            if isinstance(d, dict):
                age_young += d.get("age_under_18", 0) + d.get("age_18_24", 0)
                age_mid += d.get("age_25_34", 0) + d.get("age_35_44", 0) + d.get("age_45_54", 0)
                age_old += d.get("age_55_64", 0) + d.get("age_65_plus", 0)
    total_age = age_young + age_mid + age_old
    pct_young = age_young / total_age if total_age > 0 else 0.33
    pct_mid = age_mid / total_age if total_age > 0 else 0.34
    pct_old = age_old / total_age if total_age > 0 else 0.33

    dev["tenure_split"] = {"owner_pct": avg_own, "renter_pct": avg_rent}
    dev["age_composition"] = {"young_pct": pct_young, "middle_pct": pct_mid, "older_pct": pct_old}

    # ── Price points ─────────────────────────────────────────────
    dev["price_tier"] = price_info["tier"]
    dev["max_home_price"] = price_info["max_home"]
    dev["max_monthly_rent"] = price_info["max_rent"]
    dev["income_tier"] = tier_key
    dev["income_label"] = price_info["label"]

    # Housing price range (using income distribution from origins)
    inc_valid = origins_df["median_income"][origins_df["median_income"] > 0]
    if not inc_valid.empty:
        p25 = float(inc_valid.quantile(0.25))
        p75 = float(inc_valid.quantile(0.75))
        dev["home_price_range"] = {
            "low": int(p25 * 3.5),   # 3.5x income = affordable threshold
            "mid": int(avg_income * 3.5),
            "high": int(p75 * 4.0),  # stretch for upper quartile
        }
        dev["rent_range"] = {
            "low": int(p25 * 0.30 / 12),   # 30% of income / 12 months
            "mid": int(avg_income * 0.30 / 12),
            "high": int(p75 * 0.30 / 12),
        }
    else:
        dev["home_price_range"] = {"low": 0, "mid": 0, "high": 0}
        dev["rent_range"] = {"low": 0, "mid": 0, "high": 0}

    # ── Residential recommendations by segment ───────────────────
    housing_recs = []
    for seg in consumer_dna:
        code = seg.get("code", "")
        sh = _SEGMENT_HOUSING.get(code)
        if sh:
            housing_recs.append({
                "segment": seg["name"],
                "pct": seg["pct"],
                "product_types": sh["types"],
                "price_tier": sh["price_tier"],
                "density": sh["density"],
                "amenities": sh["amenities"],
            })
    dev["housing_recs"] = housing_recs

    # ── Primary residential recommendation (narrative) ───────────
    primary_housing = []
    if avg_rent > 0.55:
        primary_housing.append("Multifamily rental dominates — prioritise apartments and build-to-rent.")
    elif avg_own > 0.70:
        primary_housing.append("Strong owner-occupied market — single-family and townhome developments have demand.")
    else:
        primary_housing.append("Mixed tenure market — consider a blend of rental and for-sale product.")

    if pct_young > 0.35:
        primary_housing.append("Young population skew supports student housing, starter homes, or affordable apartments.")
    if pct_old > 0.30:
        primary_housing.append("Aging population creates demand for age-restricted, patio homes, and assisted living.")
    if pct_mid > 0.45:
        primary_housing.append("Working-age majority supports family-oriented housing: townhomes, move-up homes, and garden apartments.")

    pop_3mi = rings.get("3mi", {}).get("population", 0) if rings else 0
    if pop_3mi < 10_000:
        primary_housing.append("Low density favors large-lot single-family, manufactured homes, or rural estate parcels.")
    elif pop_3mi >= 50_000:
        primary_housing.append("High density supports mid-rise to high-rise multifamily and mixed-use projects.")

    dev["primary_housing_narrative"] = primary_housing

    # ── Target audiences ─────────────────────────────────────────
    audiences = []
    for seg in consumer_dna:
        code = seg.get("code", "")
        pb = seg.get("playbook", {})
        if pb:
            audiences.append({
                "segment": seg["name"],
                "pct": seg["pct"],
                "description": pb.get("who_they_are", pb.get("description", "")),
                "price_sensitivity": pb.get("price_sensitivity", ""),
                "channel": pb.get("channel_preference", ""),
                "retail_draw": _SEGMENT_RETAIL_ANCHORS.get(code, []),
            })
    dev["target_audiences"] = audiences

    # ── Retail / commercial recommendations ──────────────────────
    retail_recs = []
    # From segment anchors
    seen_retail = set()
    for seg in consumer_dna[:2]:
        code = seg.get("code", "")
        anchors = _SEGMENT_RETAIL_ANCHORS.get(code, [])
        for a in anchors:
            if a not in seen_retail:
                retail_recs.append({
                    "store_type": a,
                    "driven_by": seg["name"],
                    "rationale": f"High affinity with {seg['name']} segment ({seg['pct']:.0f}% of pop)",
                })
                seen_retail.add(a)

    # From underserved categories
    for gap in underserved[:3]:
        cat = gap["category"]
        tam_m = gap.get("tam", 0) / 1e6
        if cat not in seen_retail:
            retail_recs.append({
                "store_type": f"{cat.title()} retailer",
                "driven_by": "Market gap",
                "rationale": f"Underserved at {gap.get('saturation', 0):.0%} saturation — ${tam_m:,.1f}M TAM",
            })
            seen_retail.add(cat)

    dev["retail_recs"] = retail_recs

    # ── Mixed-use development concept ────────────────────────────
    if housing_recs:
        top_housing = housing_recs[0]
        top_retail = retail_recs[:3] if retail_recs else []
        concept_parts = []
        concept_parts.append(f"Ground-floor commercial with {', '.join(r['store_type'] for r in top_retail)}" if top_retail else "Ground-floor flexible commercial")
        concept_parts.append(f"Upper floors: {', '.join(top_housing['product_types'][:2])}")
        if top_housing.get("amenities"):
            concept_parts.append(f"Amenities: {', '.join(top_housing['amenities'][:3])}")
        dev["mixed_use_concept"] = concept_parts
    else:
        dev["mixed_use_concept"] = []

    return dev


# ---------------------------------------------------------------------------
# Revenue prediction model
# ---------------------------------------------------------------------------

# Annual per-capita retail spending by category (BLS Consumer Expenditure Survey)
_CATEGORY_SPEND = {
    "grocery": 4942, "convenience": 1500, "department": 2500,
    "apparel": 1866, "electronics": 1200, "furniture": 2050,
    "hardware": 1600, "food_specialty": 800, "general": 2000,
    "variety": 1200, "liquor": 600, "beverages": 400, "mall": 3000,
    "new": 2000,
}
_NATIONAL_MEDIAN_INCOME = 75_149  # 2023 ACS national median household income
_INCOME_ELASTICITY = 0.5  # spending grows with income but sub-linearly

# Spending propensity by price sensitivity
_PRICE_SENS_MULT = {
    "low": 1.30, "moderate": 1.00, "high": 0.75, "very high": 0.60,
}
# Visit frequency multiplier
_FREQ_MULT = {"high": 1.20, "moderate": 1.00, "low": 0.80}

# Average basket size by category (BLS + industry benchmarks)
_AVG_BASKET = {
    "grocery": 42, "convenience": 12, "department": 65,
    "apparel": 58, "electronics": 95, "furniture": 180,
    "hardware": 48, "food_specialty": 25, "general": 35,
    "variety": 18, "liquor": 32, "beverages": 15, "mall": 72,
    "new": 35,
}

# Distance rings in miles
_DISTANCE_RINGS_MI = [1, 3, 5, 10]

# Category keywords for affinity matching
_CATEGORY_AFFINITY_MAP = {
    "grocery": ["grocery", "grocery chains", "organic", "ethnic grocery", "specialty food"],
    "convenience": ["convenience"],
    "department": ["department", "big-box", "Target"],
    "apparel": ["apparel", "fast fashion", "mid-tier apparel", "family apparel",
                 "athleisure", "discount apparel"],
    "electronics": ["electronics"],
    "furniture": ["furniture", "home furnishing"],
    "hardware": ["hardware", "home improvement", "home maintenance"],
    "food_specialty": ["specialty food", "organic", "ethnic grocery"],
    "general": ["general", "big-box", "Walmart", "Target"],
    "variety": ["variety", "dollar stores", "discount"],
    "liquor": ["liquor", "wine/spirits"],
    "mall": ["mall", "premium", "specialty"],
}


def _compute_distance_rings(lat, lon, origins_df):
    """Compute population within distance rings from a point."""
    km_per_mile = 1.60934
    dists = origins_df.apply(
        lambda r: _haversine_km(lat, lon, r["lat"], r["lon"]), axis=1
    )
    rings = {}
    for mi in _DISTANCE_RINGS_MI:
        mask = dists <= (mi * km_per_mile)
        rings[f"{mi}mi"] = {
            "population": int(origins_df.loc[mask, "population"].sum()),
            "households": int(origins_df.loc[mask, "households"].sum()) if "households" in origins_df.columns else 0,
            "block_groups": int(mask.sum()),
            "avg_income": float(origins_df.loc[mask & (origins_df["median_income"] > 0), "median_income"].mean()) if mask.any() else 0,
        }
    return rings


def _nearest_competitors(new_lat, new_lon, stores_df, store_results, n=10):
    """Find n nearest existing stores with distance, share, and attributes."""
    dists = stores_df.apply(
        lambda r: _haversine_km(new_lat, new_lon, r["lat"], r["lon"]), axis=1
    )
    nearest_idx = dists.nsmallest(n).index
    result = pd.DataFrame({
        "name": stores_df.loc[nearest_idx, "name"],
        "category": stores_df.loc[nearest_idx, "category"],
        "distance_km": dists.loc[nearest_idx],
        "distance_mi": dists.loc[nearest_idx] / 1.60934,
    })
    if "square_footage" in stores_df.columns:
        result["sqft"] = stores_df.loc[nearest_idx, "square_footage"]
    if "avg_rating" in stores_df.columns:
        result["rating"] = stores_df.loc[nearest_idx, "avg_rating"]
    if store_results is not None:
        result["share"] = store_results.loc[
            store_results.index.isin(nearest_idx), "share"
        ].reindex(nearest_idx)
    return result.sort_values("distance_km")


def _category_fit_score(category, psycho_df, origins_df, new_store_probs):
    """Score 0-100 how well the catchment psychographic profile matches
    the store category via retail_affinity overlap."""
    if not _PSYCHO_OK or psycho_df is None or psycho_df.empty:
        return None, None
    cat_keywords = _CATEGORY_AFFINITY_MAP.get(category, [])
    if not cat_keywords:
        return None, None
    probs = new_store_probs.reindex(origins_df.index).fillna(0)
    aligned = psycho_df.reindex(origins_df.index).dropna(subset=["segment_code"])
    if aligned.empty:
        return None, None
    weighted_score = 0.0
    total_weight = 0.0
    matching_segs = []
    for idx in aligned.index:
        w = probs.get(idx, 0)
        if w <= 0:
            continue
        code = aligned.loc[idx, "segment_code"]
        if code in SEGMENT_PROFILES:
            affinities = SEGMENT_PROFILES[code].get("consumer_behavior", {}).get("retail_affinity", [])
            overlap = len(set(a.lower() for a in affinities) & set(k.lower() for k in cat_keywords))
            seg_score = min(overlap / max(len(cat_keywords), 1), 1.0)
            weighted_score += seg_score * w
            total_weight += w
            if overlap > 0 and SEGMENT_PROFILES[code]["name"] not in matching_segs:
                matching_segs.append(SEGMENT_PROFILES[code]["name"])
    fit = (weighted_score / total_weight * 100) if total_weight > 0 else 0
    return round(fit, 1), matching_segs


def _generate_insight_narrative(category, new_name, rev, rings, fit_score,
                                fit_segments, dom_segment, dom_pct,
                                cannib_stores, avg_income, price_level):
    """Generate a 3-5 sentence consumer insight summary."""
    lines = []
    # Location population context
    pop_3mi = rings.get("3mi", {}).get("population", 0) if rings else 0
    pop_5mi = rings.get("5mi", {}).get("population", 0) if rings else 0
    lines.append(
        f"**{new_name}** serves a catchment of {pop_3mi:,} people within 3 miles "
        f"and {pop_5mi:,} within 5 miles."
    )
    # Dominant segment
    if dom_segment and dom_pct:
        lines.append(
            f"The primary consumer segment is **{dom_segment}** ({dom_pct:.0f}% of catchment)."
        )
    # Category fit
    if fit_score is not None:
        if fit_score >= 70:
            lines.append(f"Category-audience fit is **strong** ({fit_score:.0f}/100) — "
                         f"catchment segments align well with {category} retail.")
        elif fit_score >= 40:
            lines.append(f"Category-audience fit is **moderate** ({fit_score:.0f}/100) — "
                         f"partial alignment with {category} retail preferences.")
        else:
            lines.append(f"Category-audience fit is **weak** ({fit_score:.0f}/100) — "
                         f"consider whether {category} matches this audience.")
    # Income vs price level
    if avg_income > 0:
        price_desc = {"budget": "budget-conscious", "moderate": "moderate-income",
                      "premium": "higher-income", "luxury": "affluent"}
        ideal = price_desc.get(price_level, "moderate-income")
        if price_level in ("premium", "luxury") and avg_income < 60_000:
            lines.append(f"Average catchment income (${avg_income:,.0f}) may be low for "
                         f"a {price_level} positioning — consider adjusting price strategy.")
        elif price_level == "budget" and avg_income > 100_000:
            lines.append(f"Catchment income (${avg_income:,.0f}) is high — there may be "
                         f"room for a premium positioning instead of budget.")
        else:
            lines.append(f"Average catchment income (${avg_income:,.0f}) supports "
                         f"the {price_level} price positioning.")
    # Cannibalization
    if cannib_stores and len(cannib_stores) > 0:
        names = ", ".join(cannib_stores[:3])
        lines.append(f"**Cannibalization risk**: {len(cannib_stores)} same-category "
                     f"competitor(s) within 2 km ({names}).")
    return " ".join(lines)


def predict_revenue(origins_df, new_store_probs, category, psycho_df=None):
    """Predict annual revenue for a new store based on gravity model
    probabilities, demographics, and psychographic segments.

    Returns dict with:
      annual_revenue, monthly_revenue, revenue_per_sqft (if sqft known),
      revenue_by_segment (DataFrame), confidence_low, confidence_high,
      per_bg (DataFrame with per-block-group contribution)
    """
    base_spend = _CATEGORY_SPEND.get(category, 2000)
    probs = new_store_probs.reindex(origins_df.index).fillna(0)

    # Per-block-group revenue contribution
    pop = origins_df["population"].fillna(0).astype(float)
    income = origins_df["median_income"].fillna(_NATIONAL_MEDIAN_INCOME).astype(float)
    income = income.clip(lower=10_000)  # floor for suppressed data

    # Income multiplier: (local_income / national)^elasticity
    income_mult = (income / _NATIONAL_MEDIAN_INCOME) ** _INCOME_ELASTICITY

    # Base revenue per block group (before psychographic adjustment)
    bg_revenue = probs * pop * base_spend * income_mult

    # Psychographic spending propensity adjustment
    seg_mult = pd.Series(1.0, index=origins_df.index)
    if _PSYCHO_OK and psycho_df is not None and not psycho_df.empty:
        aligned_psycho = psycho_df.reindex(origins_df.index)
        for idx in origins_df.index:
            seg_code = aligned_psycho.loc[idx, "segment_code"] if idx in aligned_psycho.index and pd.notna(aligned_psycho.loc[idx, "segment_code"]) else None
            if seg_code and seg_code in SEGMENT_PROFILES:
                cb = SEGMENT_PROFILES[seg_code].get("consumer_behavior", {})
                ps = _PRICE_SENS_MULT.get(cb.get("price_sensitivity", "moderate"), 1.0)
                fq = _FREQ_MULT.get(cb.get("shopping_frequency", "moderate"), 1.0)
                seg_mult.loc[idx] = ps * fq

    bg_revenue = bg_revenue * seg_mult

    annual = float(bg_revenue.sum())

    # Revenue by segment breakdown
    rev_by_seg = None
    if _PSYCHO_OK and psycho_df is not None and not psycho_df.empty:
        aligned_psycho = psycho_df.reindex(origins_df.index)
        bg_df = pd.DataFrame({
            "revenue": bg_revenue,
            "segment_name": aligned_psycho["segment_name"] if "segment_name" in aligned_psycho.columns else None,
        })
        bg_df = bg_df.dropna(subset=["segment_name"])
        if not bg_df.empty:
            rev_by_seg = bg_df.groupby("segment_name")["revenue"].sum().sort_values(ascending=False)
            rev_by_seg = pd.DataFrame({
                "segment": rev_by_seg.index,
                "revenue": rev_by_seg.values,
                "pct": (rev_by_seg.values / annual * 100) if annual > 0 else 0,
            })

    return {
        "annual_revenue": annual,
        "monthly_revenue": annual / 12,
        "confidence_low": annual * 0.70,
        "confidence_high": annual * 1.35,
        "revenue_by_segment": rev_by_seg,
        "bg_revenue": bg_revenue,
    }


def simulate_new_store(origins_df, stores_df, new_lat, new_lon, new_sqft,
                       alpha, lam, name="NEW STORE", category="general",
                       rating=0.0, price_level="moderate"):
    # Price multiplier matching enrichment.py logic
    _price_mult = {"budget": 0.8, "moderate": 1.0, "premium": 1.15, "luxury": 1.3}
    price_m = _price_mult.get(price_level, 1.0)
    # Composite attractiveness: sqft × (rating/3) × price_mult (if rating provided)
    attr = float(new_sqft)
    if rating > 0:
        attr = attr * (rating / 3.0) * price_m

    new_store = pd.DataFrame([{
        "name": name, "lat": new_lat, "lon": new_lon,
        "category": category, "brand": None,
        "square_footage": new_sqft, "avg_rating": rating,
        "composite_attractiveness": attr,
    }], index=["new_store_scenario"])
    stores_with_new = pd.concat([stores_df, new_store])

    baseline = run_huff(origins_df, stores_df, alpha, lam)
    scenario = run_huff(origins_df, stores_with_new, alpha, lam)

    new_share = float(scenario["store_results"].loc["new_store_scenario", "share"])
    new_demand = float(scenario["store_results"].loc["new_store_scenario", "demand"])

    baseline_shares = baseline["store_results"]["share"]
    scenario_shares = scenario["store_results"]["share"].reindex(baseline_shares.index).fillna(0)
    change = scenario_shares - baseline_shares

    impact_df = pd.DataFrame({
        "name": stores_df["name"],
        "category": stores_df["category"],
        "baseline_share": baseline_shares,
        "scenario_share": scenario_shares,
        "change": change,
        "pct_change": (change / baseline_shares.replace(0, np.nan)) * 100,
    }).sort_values("change")

    # Catchment analysis: top origins by probability for the new store
    new_probs = scenario["prob_df"]["new_store_scenario"] if "new_store_scenario" in scenario["prob_df"].columns else None

    # Revenue prediction
    revenue = None
    if new_probs is not None:
        psycho_df = None
        try:
            import streamlit as _st
            psycho_df = _st.session_state.get("psycho_df")
        except Exception:
            pass
        revenue = predict_revenue(origins_df, new_probs, category, psycho_df)
        if revenue and new_sqft > 0:
            revenue["revenue_per_sqft"] = revenue["annual_revenue"] / new_sqft

    return {
        "new_store_share": new_share, "new_store_demand": new_demand,
        "impact_df": impact_df,
        "hhi_baseline": baseline["hhi"], "hhi_scenario": scenario["hhi"],
        "new_store_probs": new_probs, "revenue": revenue,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def render_store_map(store_results, origins_df, center_lat, center_lon):
    try:
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")
        max_s = store_results["share"].max()
        min_s = store_results["share"].min()
        rng = max_s - min_s if max_s > min_s else 1

        for sid, row in store_results.head(100).iterrows():
            intensity = (row["share"] - min_s) / rng
            r = int(196 + (184 - 196) * intensity)
            g = int(160 + (92 - 160) * intensity)
            b = int(90 + (58 - 90) * intensity)
            color = f"#{r:02x}{g:02x}{b:02x}"
            radius = max(3, min(12, 3 + intensity * 9))
            folium.CircleMarker(
                location=[row["lat"], row["lon"]], radius=radius,
                color=color, fill=True, fill_color=color, fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{row['name']}</b><br>Category: {row['category']}<br>"
                    f"Share: {_sf(row['share'], '.4%')}<br>Demand: {_sf(row['demand'], ',.0f')}",
                    max_width=250),
            ).add_to(m)
        st_folium(m, width=None, height=500, use_container_width=True)
    except ImportError:
        st.map(store_results[["lat", "lon"]].head(200), use_container_width=True)


def render_origin_heatmap(prob_df, origins_df, store_id, center_lat, center_lon):
    try:
        import folium
        from folium.plugins import HeatMap
        from streamlit_folium import st_folium

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")
        probs = prob_df[store_id]
        heat_data = [
            [origins_df.loc[oid, "lat"], origins_df.loc[oid, "lon"], float(p)]
            for oid, p in probs.items()
            if oid in origins_df.index and p > 0.001
        ]
        if heat_data:
            HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
        folium.Marker(
            location=[center_lat, center_lon], popup=f"Store: {store_id}",
            icon=folium.Icon(color="darkred", icon="star"),
        ).add_to(m)
        st_folium(m, width=None, height=500, use_container_width=True)
    except ImportError:
        st.info("Install folium and streamlit-folium for maps.")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.markdown("""
    <style>
    /* ── Esque Design System ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400;1,500&family=Raleway:wght@200;300;400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --obsidian: #0F0F0D;
        --obsidian-soft: #181815;
        --obsidian-mid: #252520;
        --obsidian-light: #363630;
        --stone: #6B6A62;
        --sand: #A89F92;
        --ivory: #EDE8E0;
        --ivory-warm: #F7F3EC;
        --gold: #C4A05A;
        --gold-light: #D9BC81;
        --gold-dark: #8F7038;
        --sage: #7A8470;
        --sage-light: #A3AE9C;
        --ember: #B85C3A;
        --font-display: 'Cormorant Garamond', Georgia, serif;
        --font-ui: 'Raleway', sans-serif;
        --font-body: 'DM Sans', sans-serif;
        --divider: 1px solid rgba(168,159,146,0.18);
        --radius: 3px;
        --radius-lg: 6px;
    }

    /* ── Reset & Base ── */
    .block-container { padding-top: 1.5rem; }

    html, body, [class*="css"] {
        font-family: var(--font-body);
        font-size: 15px;
        line-height: 1.7;
        color: var(--ivory);
        -webkit-font-smoothing: antialiased;
    }

    p, li, span, label {
        font-family: var(--font-body);
        color: var(--ivory);
    }

    /* ── Headings: Cormorant Garamond ── */
    h1 {
        font-family: var(--font-display) !important;
        font-weight: 300 !important;
        line-height: 1.05 !important;
        letter-spacing: -0.01em !important;
        color: var(--ivory-warm) !important;
        font-size: clamp(36px, 4vw, 56px) !important;
    }
    h2 {
        font-family: var(--font-display) !important;
        font-weight: 400 !important;
        line-height: 1.1 !important;
        color: var(--ivory-warm) !important;
        font-size: clamp(28px, 3vw, 42px) !important;
    }
    h3 {
        font-family: var(--font-display) !important;
        font-weight: 500 !important;
        line-height: 1.2 !important;
        color: var(--ivory) !important;
        font-size: clamp(22px, 2.4vw, 32px) !important;
    }
    h4, h5, h6 {
        font-family: var(--font-ui) !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em !important;
        color: var(--sand) !important;
    }

    /* ── Backgrounds ── */
    .stApp, [data-testid="stAppViewContainer"] { background-color: var(--obsidian); }
    [data-testid="stSidebar"] { background-color: var(--obsidian-soft); }
    [data-testid="stHeader"] { background-color: var(--obsidian); }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] label {
        font-family: var(--font-ui);
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--sand);
    }
    [data-testid="stSidebar"] .stMarkdown p {
        font-family: var(--font-body);
        font-size: 13px;
        color: var(--sand);
    }
    [data-testid="stSidebar"] h1 {
        font-family: var(--font-ui) !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        letter-spacing: 0.28em !important;
        text-transform: uppercase !important;
        color: var(--gold) !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: var(--font-ui) !important;
        font-size: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.22em !important;
        text-transform: uppercase !important;
        color: var(--gold) !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaption"] {
        font-family: var(--font-body);
        font-size: 12px;
        color: var(--stone);
        letter-spacing: 0.04em;
    }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {
        font-family: var(--font-display) !important;
        font-size: 2rem !important;
        font-weight: 300 !important;
        color: var(--ivory-warm) !important;
        line-height: 1.1 !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: var(--font-ui) !important;
        font-size: 10px !important;
        font-weight: 500 !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        color: var(--stone) !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: var(--font-body) !important;
        font-size: 12px !important;
    }
    [data-testid="stMetricDelta"] svg { fill: var(--sage); }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: var(--divider);
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-ui);
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--stone);
        background-color: transparent;
        border-bottom: 2px solid transparent;
        padding: 0.6rem 1.2rem;
        transition: color 0.25s, border-color 0.25s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--sand);
    }
    .stTabs [aria-selected="true"] {
        color: var(--gold) !important;
        border-bottom: 2px solid var(--gold) !important;
        background-color: transparent !important;
        font-weight: 600 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        font-family: var(--font-ui);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        background-color: var(--gold);
        color: var(--obsidian);
        border: none;
        border-radius: var(--radius);
        padding: 0.55rem 1.4rem;
        transition: background-color 0.25s, transform 0.15s;
    }
    .stButton > button:hover {
        background-color: var(--gold-light);
        color: var(--obsidian);
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Inputs & Selects ── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSlider [data-baseweb="slider"] {
        font-family: var(--font-body);
        font-size: 13px;
        background-color: var(--obsidian-mid);
        border-color: var(--obsidian-light);
        color: var(--ivory);
        border-radius: var(--radius);
    }
    [data-baseweb="select"] {
        font-family: var(--font-body);
        background-color: var(--obsidian-mid);
    }
    [data-baseweb="select"] * { color: var(--ivory) !important; }
    [data-baseweb="input"] {
        font-family: var(--font-body);
        background-color: var(--obsidian-mid) !important;
    }

    /* ── Tables ── */
    [data-testid="stTable"] table,
    .stDataFrame table {
        font-family: var(--font-body);
        font-size: 13px;
        background-color: var(--obsidian-soft);
        color: var(--ivory);
    }
    [data-testid="stTable"] th,
    .stDataFrame th {
        font-family: var(--font-ui) !important;
        font-size: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        background-color: var(--obsidian-mid);
        color: var(--gold) !important;
        border-bottom: 1px solid var(--obsidian-light);
        padding: 12px 16px !important;
    }
    [data-testid="stTable"] td,
    .stDataFrame td {
        border-bottom: 1px solid rgba(168,159,146,0.10);
        padding: 10px 16px !important;
        color: var(--sand);
    }
    [data-testid="stTable"] tr:nth-child(even) td,
    .stDataFrame tr:nth-child(even) td {
        background-color: rgba(37,37,32,0.5);
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background-color: var(--obsidian-soft);
        border: 1px solid rgba(168,159,146,0.10);
        border-radius: var(--radius-lg);
    }
    [data-testid="stExpander"] summary span {
        font-family: var(--font-ui) !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase;
        color: var(--gold) !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        border-top: var(--divider);
    }

    /* ── Dividers ── */
    hr, [data-testid="stSeparator"] {
        border-color: rgba(168,159,146,0.18) !important;
    }

    /* ── Progress Bars ── */
    [data-testid="stProgress"] > div > div {
        background-color: var(--gold);
    }

    /* ── Alerts / Info Boxes ── */
    [data-testid="stAlert"] {
        font-family: var(--font-body);
        border-radius: var(--radius-lg);
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--obsidian); }
    ::-webkit-scrollbar-thumb { background: var(--obsidian-light); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--stone); }

    /* ── Captions & Small Text ── */
    .stCaption, small, [data-testid="stCaption"] {
        font-family: var(--font-body) !important;
        font-size: 11px !important;
        color: var(--stone) !important;
        letter-spacing: 0.04em;
    }

    /* ── Plotly Charts ── */
    .js-plotly-plot .plotly .gtitle {
        font-family: var(--font-display) !important;
    }

    /* ── Markdown inside main area ── */
    .stMarkdown p {
        font-family: var(--font-body);
        font-size: 14px;
        line-height: 1.75;
        color: var(--sand);
    }
    .stMarkdown strong { color: var(--ivory); }
    .stMarkdown a {
        color: var(--gold);
        text-decoration: none;
        border-bottom: 1px solid rgba(196,160,90,0.3);
        transition: border-color 0.25s;
    }
    .stMarkdown a:hover {
        border-bottom-color: var(--gold);
    }

    /* ── Subheader styling (used heavily in tabs) ── */
    .stMarkdown h3 {
        font-family: var(--font-display) !important;
        font-weight: 500 !important;
        color: var(--ivory) !important;
        padding-bottom: 8px;
        border-bottom: var(--divider);
        margin-bottom: 16px !important;
    }

    /* ── Multiselect tags ── */
    [data-baseweb="tag"] {
        font-family: var(--font-ui);
        font-size: 10px;
        letter-spacing: 0.08em;
        background-color: var(--obsidian-light) !important;
        border-radius: var(--radius) !important;
    }

    /* ── Slider ── */
    [data-baseweb="slider"] [role="slider"] {
        background-color: var(--gold) !important;
    }
    [data-baseweb="slider"] [data-testid="stThumbValue"] {
        font-family: var(--font-body);
        font-size: 12px;
        color: var(--ivory);
    }

    /* ── Toast / Notifications ── */
    [data-testid="stToast"] {
        font-family: var(--font-body);
        background-color: var(--obsidian-soft) !important;
        border: 1px solid var(--obsidian-light) !important;
        border-radius: var(--radius-lg) !important;
    }

    /* ── Column gaps ── */
    [data-testid="column"] {
        padding: 0 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Gravity Model")
        st.caption("Retail market analysis for any US market")

        st.subheader("Market")

        # State selector
        state_options = sorted(STATE_FIPS.items(), key=lambda x: x[1]["name"])
        state_labels = [f"{v['name']} ({v['abbr']})" for _, v in state_options]
        state_fips_list = [k for k, _ in state_options]
        default_state_idx = next((i for i, f in enumerate(state_fips_list) if f == "48"), 0)

        selected_state_idx = st.selectbox(
            "State", range(len(state_labels)),
            format_func=lambda i: state_labels[i],
            index=default_state_idx,
        )
        selected_state_fips = state_fips_list[selected_state_idx]

        # County selector
        counties_df = get_counties_for_state(selected_state_fips)
        if counties_df.empty:
            county_labels, county_fips_list = ["(loading)"], ["000"]
        else:
            county_labels = [
                f"{row['county_name']} (pop: {_sf(row.get('population', 0), ',.0f')})"
                for _, row in counties_df.iterrows()
            ]
            county_fips_list = counties_df["county_fips"].tolist()

        default_county_idx = 0
        if selected_state_fips == "48":
            try:
                default_county_idx = county_fips_list.index("453")
            except ValueError:
                pass

        selected_county_idx = st.selectbox(
            "County", range(len(county_labels)),
            format_func=lambda i: county_labels[i],
            index=default_county_idx,
        )
        selected_county_fips = county_fips_list[selected_county_idx]

        # ZIP filter (inline)
        zip_input = st.text_input("ZIP code (optional)", placeholder="e.g. 78701")

        # ── Auto-trigger: detect market change ──────────────────────
        _current_market_key = f"{selected_state_fips}_{selected_county_fips}"
        _market_changed = False
        if "last_market_key" not in st.session_state:
            # First load — just remember the key, don't auto-trigger
            st.session_state["last_market_key"] = _current_market_key
        elif st.session_state["last_market_key"] != _current_market_key:
            # User changed state/county — auto-trigger
            _market_changed = True
            st.session_state["last_market_key"] = _current_market_key
            # Clear stale results so pipeline re-runs for new market
            for _k in ["results", "model_results", "ensemble", "origins_df",
                        "stores_df", "market_label", "bbox", "data_sources",
                        "demo_summary", "bls_county", "bls_retail",
                        "dist_matrix", "psycho_df", "psycho_summary",
                        "cbp_data", "sec_benchmarks", "health_data",
                        "fred_data", "dc_data", "fmp_benchmarks", "ckan_data",
                        "noaa_data"]:
                st.session_state.pop(_k, None)

        st.divider()
        run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

        # ── Consumer Filters (shown after analysis) ──────────────────
        cf_income_min, cf_income_max = 0, 250_000
        cf_age_bands, cf_tenure, cf_segments = [], [], []
        cf_price_sens, cf_channel, cf_affinity = [], [], []

        if "results" in st.session_state:
            st.divider()
            st.subheader("Consumer Filters")

            with st.expander("Demographics"):
                cf_income_min, cf_income_max = st.slider(
                    "Income range ($K)",
                    min_value=0, max_value=250_000, value=(0, 250_000),
                    step=5_000, format="$%d",
                )
                cf_age_bands = st.multiselect(
                    "Dominant age group",
                    list(_AGE_BANDS.keys()),
                )
                cf_tenure = st.multiselect(
                    "Housing tenure",
                    ["Owner-dominated", "Renter-dominated", "Mixed"],
                )

            if _PSYCHO_OK:
                seg_names = sorted(
                    [p["name"] for p in SEGMENT_PROFILES.values()]
                )
                with st.expander("Lifestyle Segments"):
                    cf_segments = st.multiselect(
                        "Select segments", seg_names,
                    )

                with st.expander("Consumer Behavior"):
                    cf_price_sens = st.multiselect(
                        "Price sensitivity",
                        ["low", "moderate", "high"],
                    )
                    cf_channel = st.multiselect(
                        "Channel preference",
                        ["in-store", "online", "omnichannel"],
                    )
                    cf_affinity = st.multiselect(
                        "Retail affinity",
                        _get_all_retail_affinities(),
                    )

            # Clear all
            _consumer_filters = {
                "income_min": cf_income_min, "income_max": cf_income_max,
                "age_bands": cf_age_bands, "tenure": cf_tenure,
                "segments": cf_segments, "price_sensitivity": cf_price_sens,
                "channel_preference": cf_channel, "retail_affinity": cf_affinity,
            }
            n_active = count_active_filters(_consumer_filters)
            if n_active > 0:
                if st.button("Clear All Filters", use_container_width=True):
                    st.rerun()

        # Advanced settings (collapsed)
        with st.expander("Advanced Settings"):
            bbox_radius = st.slider("Analysis radius (km)", 5.0, 50.0, 12.0, 1.0)
            alpha = st.slider("Alpha (attractiveness)", 0.1, 3.0, 1.0, 0.1)
            lam = st.slider("Lambda (distance decay)", 0.5, 5.0, 2.0, 0.1)
            zip_radius = st.slider("ZIP filter radius (km)", 0.5, 25.0, 5.0, 0.5)

        # API keys (collapsed)
        with st.expander("API Keys (optional)"):
            google_api_key = st.text_input("Google Places", type="password", placeholder="Adds ratings & reviews")
            yelp_api_key = st.text_input("Yelp Fusion", type="password", placeholder="Free: 5000/day")
            fred_api_key = st.text_input("FRED (Federal Reserve)", type="password", placeholder="Free from fred.stlouisfed.org")
            dc_api_key = st.text_input("Data Commons", type="password", placeholder="Free from datacommons.org")
            fmp_api_key = st.text_input("Financial Modeling Prep", type="password", placeholder="Free: 250/day")
            noaa_token = st.text_input("NOAA Climate (NCEI)", type="password", placeholder="Free from ncei.noaa.gov/cdo-web/token")

        st.divider()
        st.caption(f"Census ACS + OSM + BLS + CBP + SEC + CDC + NOAA | v{_APP_VERSION}")

    # ── Resolve geography ────────────────────────────────────────────────
    if "bbox_radius" not in dir():
        bbox_radius = 12.0
    if "alpha" not in dir():
        alpha, lam = 1.0, 2.0
    if "zip_radius" not in dir():
        zip_radius = 5.0

    state_fips = selected_state_fips
    county_fips = selected_county_fips
    bbox = county_bbox(state_fips, county_fips, radius_km=bbox_radius)

    if bbox is None:
        st.warning("Could not determine county bounds.")
        bbox = (30.22, -97.82, 30.35, -97.68)

    state_info = STATE_FIPS.get(state_fips, {})
    county_name = county_labels[selected_county_idx].split(" (pop")[0]
    market_label = f"{county_name}, {state_info.get('abbr', state_fips)}"
    st.session_state["_state_name"] = state_info.get("name", "")
    st.session_state["_county_name"] = county_name

    # ── Run analysis ─────────────────────────────────────────────────────
    _should_run = run_btn or _market_changed
    if _should_run or "results" in st.session_state:
        if _should_run and "results" not in st.session_state:
            status_container = st.container()
            with status_container:
                with st.status(f"Analyzing {market_label}...", expanded=True) as status:
                    data_sources = ["Census ACS"]

                    # Step 1: Census
                    st.write("Fetching Census demographics...")
                    try:
                        origins_df = fetch_census(state_fips, county_fips, bbox)
                        if origins_df.empty:
                            st.error("Census API returned no data. Try a different county.")
                            return
                        st.write(f"{len(origins_df)} block groups loaded")
                    except Exception as e:
                        st.error(f"Census API error: {e}")
                        return

                    # Step 2: Stores
                    st.write("Fetching stores from OpenStreetMap...")
                    try:
                        stores_df = fetch_osm_stores(bbox)
                        data_sources.append("OpenStreetMap")
                        if stores_df.empty:
                            st.error("No stores found. Try increasing the analysis radius.")
                            return
                        st.write(f"{len(stores_df)} stores found")
                    except Exception as e:
                        st.error(f"OSM error: {e}")
                        return

                    # Step 3: Store enrichment
                    if _STORE_SIZE_OK:
                        brand_n = (stores_df.get("sqft_source") == "brand").sum() if "sqft_source" in stores_df.columns else 0
                        st.write(f"Store sizes estimated ({brand_n} brand-matched)")
                        data_sources.append("Store Size DB")

                    # Step 4: Optional API enrichment
                    if google_api_key and google_api_key.strip():
                        st.write("Enriching with Google Places...")
                        try:
                            from gravity.data.google_places import GooglePlacesLoader
                            gp = GooglePlacesLoader(api_key=google_api_key.strip())
                            sm_list = [
                                Store(store_id=str(sid), name=row["name"], lat=row["lat"], lon=row["lon"],
                                      category=row.get("category"), brand=_clean_str(row.get("brand")))
                                for sid, row in stores_df.iterrows()
                            ]
                            gp.enrich_stores(sm_list, search_radius_m=150)
                            for sm in sm_list:
                                if sm.avg_rating > 0:
                                    stores_df.loc[sm.store_id, "avg_rating"] = sm.avg_rating
                                    stores_df.loc[sm.store_id, "price_level"] = sm.price_level
                            data_sources.append("Google Places")
                        except Exception as e:
                            st.write(f"Google Places: {e}")

                    if yelp_api_key and yelp_api_key.strip():
                        st.write("Enriching with Yelp Fusion...")
                        try:
                            from gravity.data.yelp import YelpLoader
                            yl = YelpLoader(api_key=yelp_api_key.strip())
                            sm_list = [
                                Store(store_id=str(sid), name=row["name"], lat=row["lat"], lon=row["lon"],
                                      category=row.get("category"), brand=_clean_str(row.get("brand")))
                                for sid, row in stores_df.iterrows()
                            ]
                            yl.enrich_stores(sm_list, search_radius_m=200)
                            for sm in sm_list:
                                if sm.avg_rating > 0 and stores_df.loc[sm.store_id, "avg_rating"] == 0:
                                    stores_df.loc[sm.store_id, "avg_rating"] = sm.avg_rating
                            data_sources.append("Yelp Fusion")
                        except Exception as e:
                            st.write(f"Yelp: {e}")

                    # Step 5: Composite attractiveness
                    if _ENRICHMENT_OK:
                        sm_list = [
                            Store(store_id=str(sid), name=row["name"], lat=row["lat"], lon=row["lon"],
                                  category=row.get("category"), brand=_clean_str(row.get("brand")),
                                  square_footage=row.get("square_footage", 0.0),
                                  avg_rating=row.get("avg_rating", 0.0),
                                  price_level=int(row.get("price_level", 2)))
                            for sid, row in stores_df.iterrows()
                        ]
                        compute_composite_attractiveness(sm_list)
                        for sm in sm_list:
                            stores_df.loc[sm.store_id, "square_footage"] = sm.square_footage

                    # Step 6: Distance matrix
                    st.write("Computing distances...")
                    if _ENRICHMENT_OK:
                        try:
                            dist_matrix = pipeline_distance_matrix(origins_df, stores_df, use_osrm=True)
                            data_sources.append("OSRM")
                        except Exception:
                            dist_matrix = build_distance_matrix(origins_df, stores_df)
                    else:
                        dist_matrix = build_distance_matrix(origins_df, stores_df)

                    # Step 7: Run models (all automatic)
                    st.write("Running gravity models...")
                    huff_results = run_huff(origins_df, stores_df, alpha, lam, dist_matrix)
                    model_results = {"Huff": huff_results}

                    mci_res = run_mci(origins_df, stores_df, dist_matrix)
                    if mci_res:
                        model_results["MCI"] = mci_res

                    nb_res = run_count_model(origins_df, stores_df, dist_matrix)
                    if nb_res:
                        model_results["NegBin"] = nb_res

                    conf_res = run_conformal(huff_results["prob_df"])
                    if conf_res:
                        model_results["Conformal"] = conf_res

                    cd_res = run_competing_destinations(origins_df, stores_df, alpha, lam)
                    if cd_res:
                        model_results["CD"] = cd_res

                    # Step 8: Ensemble
                    ensemble = compute_ensemble(model_results, origins_df, stores_df)

                    # Step 9: BLS
                    bls_county, bls_retail = fetch_bls(state_fips, county_fips)
                    if bls_county:
                        data_sources.append("BLS QCEW")

                    # Step 10: Demographics
                    demo_summary = aggregate_demographics(origins_df)

                    # Step 11: Psychographic segments
                    psycho_df = None
                    psycho_summary = None
                    if _PSYCHO_OK:
                        try:
                            st.write("Classifying lifestyle segments...")
                            classifier = CensusPsychographicClassifier()
                            psycho_df = classifier.classify(origins_df)
                            psycho_summary = classifier.segment_summary(psycho_df)
                            data_sources.append("Census Psychographics")
                            n_segs = psycho_summary["segment_code"].nunique() if psycho_summary is not None else 0
                            st.write(f"{n_segs} lifestyle segments identified")
                        except Exception as e:
                            st.write(f"Psychographics: {e}")

                    # Step 12: Census County Business Patterns
                    cbp_data = fetch_cbp(state_fips, county_fips)
                    if cbp_data:
                        data_sources.append("Census CBP")

                    # Step 13: Health context (CDC/HHS)
                    health_data = fetch_health(state_fips, county_fips)
                    if health_data:
                        data_sources.append("HealthData.gov")

                    # Step 14: SEC EDGAR benchmarks
                    sec_benchmarks = []
                    _brands = stores_df["brand"].dropna().unique().tolist() if "brand" in stores_df.columns else []
                    if _brands:
                        sec_benchmarks = fetch_sec_benchmarks(tuple(sorted(set(_brands))))
                        if sec_benchmarks:
                            data_sources.append("SEC EDGAR")

                    # Step 15: FRED macro indicators (needs API key)
                    fred_data = {}
                    if fred_api_key and fred_api_key.strip():
                        st.write("Fetching FRED macro indicators...")
                        fred_data = fetch_fred(fred_api_key.strip())
                        if fred_data:
                            data_sources.append("FRED")

                    # Step 16: Data Commons (needs API key)
                    dc_data = {}
                    if dc_api_key and dc_api_key.strip():
                        st.write("Fetching Data Commons indicators...")
                        dc_data = fetch_data_commons(state_fips, county_fips, dc_api_key.strip())
                        if dc_data:
                            data_sources.append("Data Commons")

                    # Step 17: FMP benchmarks (needs API key)
                    fmp_benchmarks = []
                    if fmp_api_key and fmp_api_key.strip() and _brands:
                        fmp_benchmarks = fetch_fmp_benchmarks(tuple(sorted(set(_brands))), fmp_api_key.strip())
                        if fmp_benchmarks:
                            data_sources.append("FMP")

                    # Step 18: CKAN/data.gov dataset discovery
                    ckan_data = fetch_ckan_datasets(
                        st.session_state.get("_state_name", ""),
                        st.session_state.get("_county_name", ""),
                    )

                    # Step 19: NOAA climate context (needs token)
                    noaa_data = {}
                    if noaa_token and noaa_token.strip():
                        st.write("Fetching NOAA climate data...")
                        noaa_data = fetch_noaa_climate(state_fips, county_fips, noaa_token.strip())
                        if noaa_data:
                            data_sources.append("NOAA NCEI")

                    n_models = sum(1 for r in model_results.values() if r and "error" not in r)
                    status.update(label=f"Done — {n_models} models, {len(data_sources)} data sources", state="complete")

                st.session_state.update({
                    "results": huff_results, "model_results": model_results,
                    "ensemble": ensemble, "origins_df": origins_df,
                    "stores_df": stores_df, "market_label": market_label,
                    "bbox": bbox, "alpha": alpha, "lam": lam,
                    "data_sources": data_sources, "demo_summary": demo_summary,
                    "bls_county": bls_county, "bls_retail": bls_retail,
                    "dist_matrix": dist_matrix,
                    "psycho_df": psycho_df, "psycho_summary": psycho_summary,
                    "cbp_data": cbp_data, "sec_benchmarks": sec_benchmarks,
                    "health_data": health_data, "fred_data": fred_data,
                    "dc_data": dc_data, "fmp_benchmarks": fmp_benchmarks,
                    "ckan_data": ckan_data, "noaa_data": noaa_data,
                })

        # Retrieve from session state
        results = st.session_state["results"]
        model_results = st.session_state.get("model_results", {"Huff": results})
        ensemble = st.session_state.get("ensemble")
        origins_df = st.session_state["origins_df"]
        stores_df = st.session_state["stores_df"]
        market_label = st.session_state["market_label"]
        bbox = st.session_state["bbox"]
        dist_matrix = st.session_state.get("dist_matrix", results.get("dist_matrix"))
        demo_summary = st.session_state.get("demo_summary", {})
        bls_county = st.session_state.get("bls_county", {})
        bls_retail = st.session_state.get("bls_retail", {})

        active_results = ensemble if ensemble else results
        store_results = active_results["store_results"]

        # ZIP filter
        zip_active = False
        if zip_input and zip_input.strip():
            zinfo = geocode_zip_cached(zip_input)
            if zinfo and zinfo.get("lat"):
                zip_active = True
                zip_lat, zip_lon = zinfo["lat"], zinfo["lon"]
                origins_df = filter_by_zip(origins_df, zip_lat, zip_lon, zip_radius)
                stores_df = filter_by_zip(stores_df, zip_lat, zip_lon, zip_radius)
                if origins_df.empty or stores_df.empty:
                    st.error(f"No data within {zip_radius} km of ZIP {zip_input}.")
                    return
                store_results = store_results.loc[store_results.index.isin(stores_df.index)].copy()
                if store_results.empty:
                    st.error("No store results in filtered area.")
                    return

        # ── Consumer filters ─────────────────────────────────────────────
        consumer_filter_active = False
        total_pop_unfiltered = int(origins_df["population"].sum())
        if "results" in st.session_state:
            _cf = {
                "income_min": cf_income_min, "income_max": cf_income_max,
                "age_bands": cf_age_bands, "tenure": cf_tenure,
                "segments": cf_segments, "price_sensitivity": cf_price_sens,
                "channel_preference": cf_channel, "retail_affinity": cf_affinity,
            }
            n_cf = count_active_filters(_cf)
            if n_cf > 0:
                psycho_df = st.session_state.get("psycho_df")
                prob_df = active_results.get("prob_df")
                cf_mask = apply_consumer_filters(origins_df, psycho_df, _cf)
                origins_df = origins_df[cf_mask].copy()
                if origins_df.empty:
                    st.error("No block groups match the selected filters.")
                    return
                if prob_df is not None:
                    store_results, cf_hhi, cf_top5 = recalculate_from_filtered(
                        prob_df, origins_df, stores_df
                    )
                    active_results = dict(active_results)
                    active_results["hhi"] = cf_hhi
                    active_results["top5_concentration"] = cf_top5
                    active_results["store_results"] = store_results
                consumer_filter_active = True

        # ── Header ───────────────────────────────────────────────────────
        zip_badge = f"  |  ZIP {zip_input.strip()}" if zip_active else ""
        filter_badge = f"  |  {n_cf} filter{'s' if n_cf != 1 else ''}" if consumer_filter_active else ""
        n_models = ensemble.get("n_models", 1) if ensemble else 1
        st.title(f"{market_label}{zip_badge}{filter_badge}")

        ds_list = st.session_state.get("data_sources", [])
        st.caption(f"{n_models} models | {' + '.join(ds_list)}")

        total_pop = int(origins_df["population"].sum())
        avg_income = origins_df.loc[origins_df["median_income"] > 0, "median_income"].mean()
        total_hh = int(origins_df["households"].sum()) if "households" in origins_df.columns else 0
        top_store = store_results.iloc[0]

        c1, c2, c3, c4, c5 = st.columns(5)
        pop_label = f"{_safe_float(total_pop):,.0f}"
        if consumer_filter_active:
            pct_match = total_pop / total_pop_unfiltered * 100 if total_pop_unfiltered > 0 else 0
            pop_label += f" ({pct_match:.0f}%)"
        c1.metric("Population", pop_label)
        c2.metric("Households", f"{_safe_float(total_hh):,.0f}")
        c3.metric("Stores", f"{len(stores_df):,}")
        c4.metric("Avg Income", f"${_sf(avg_income, ',.0f')}" if _safe_float(avg_income) > 0 else "N/A")
        c5.metric("HHI", _sf(active_results['hhi'], '.0f'))

        # ── Pre-compute market intelligence ─────────────────────────────
        # Dominant psychographic segment
        _ov_dom_seg, _ov_dom_pct = None, None
        psycho_df_ov = st.session_state.get("psycho_df")
        if _PSYCHO_OK and psycho_df_ov is not None and "segment_name" in psycho_df_ov.columns:
            vp = psycho_df_ov[psycho_df_ov["segment_name"].notna()]
            if not vp.empty:
                pop_by_s = vp.groupby("segment_name")["population"].sum()
                if pop_by_s.sum() > 0:
                    _ov_dom_seg = pop_by_s.idxmax()
                    _ov_dom_pct = pop_by_s.max() / pop_by_s.sum() * 100

        # Market gaps
        gap_df = compute_market_gaps(stores_df, total_pop)

        # Site viability (use county center for 3mi pop estimate)
        center_lat_v = (bbox[0] + bbox[2]) / 2
        center_lon_v = (bbox[1] + bbox[3]) / 2
        _rings_ov = _compute_distance_rings(center_lat_v, center_lon_v, origins_df)
        pop_3mi = _rings_ov.get("3mi", {}).get("population", total_pop)
        n_same_cat_all = len(stores_df)  # full market store count for overview

        viability = compute_site_viability(
            total_pop, _safe_float(avg_income), n_same_cat_all,
            "general", active_results["hhi"], None, pop_3mi,
        )

        # Executive summary
        exec_summary = generate_executive_summary(
            market_label, total_pop, _safe_float(avg_income), len(stores_df),
            active_results["hhi"], _ov_dom_seg, _ov_dom_pct, gap_df, viability,
        )

        # Build insights report
        psycho_summary_ov = st.session_state.get("psycho_summary")
        bls_county_ov = st.session_state.get("bls_county", {})
        bls_retail_ov = st.session_state.get("bls_retail", {})
        insights_report = build_insights_report(
            market_label, total_pop, total_hh, _safe_float(avg_income),
            stores_df, store_results, active_results,
            gap_df, viability, exec_summary,
            psycho_df_ov, psycho_summary_ov,
            bls_county_ov, bls_retail_ov,
            origins_df, bbox, _rings_ov,
        )

        # ── Tabs ─────────────────────────────────────────────────────────
        tab_insights, tab_overview, tab_demo, tab_psycho, tab_competition, tab_scenario, tab_sources = st.tabs([
            "Insights", "Market Overview", "Demographics", "Psychographics", "Competition", "Scenario", "Data Sources",
        ])

        # ── Tab 0: Insights ───────────────────────────────────────────────
        with tab_insights:
            import plotly.express as px

            # --- A. Market Grade Header ---
            _grade_emoji = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}
            _g = insights_report["grade"]
            st.markdown(
                f"## {_grade_emoji.get(_g, '')} Market Grade: {_g} &nbsp;&mdash;&nbsp; "
                f"{insights_report['score']}/100",
            )
            st.markdown(f"**{insights_report['verdict']}**")
            st.markdown(
                f"{market_label} &nbsp;|&nbsp; Pop {total_pop:,} &nbsp;|&nbsp; "
                f"{len(stores_df):,} stores &nbsp;|&nbsp; "
                f"Median Income ${_safe_float(avg_income):,.0f}"
            )

            # Score breakdown
            comps = insights_report.get("components", {})
            if comps:
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Population", f"{comps.get('population', 0)}/20")
                sc2.metric("Income Fit", f"{comps.get('income', 0)}/20")
                sc3.metric("Competition", f"{comps.get('competition', 0)}/20")
                sc4.metric("Category Fit", f"{comps.get('category_fit', 0)}/20")
                sc5.metric("Market Health", f"{comps.get('market_health', 0)}/20")

            st.divider()

            # --- B. Executive Summary ---
            st.subheader("Executive Summary")
            st.info(exec_summary)

            st.divider()

            # --- C. Opportunity Scorecard ---
            st.subheader("Opportunity Scorecard")
            _underserved = insights_report.get("underserved", [])
            if _underserved:
                _us_cols = st.columns(min(len(_underserved), 3))
                for i, gap in enumerate(_underserved[:3]):
                    with _us_cols[i]:
                        st.metric(gap["category"].title(), f"{gap.get('saturation', 0):.0%} saturated")
                        st.caption(
                            f"**{gap.get('actual_stores', 0)}** stores vs "
                            f"**{gap.get('expected_stores', 0)}** expected\n\n"
                            f"TAM: **${gap.get('tam', 0) / 1e6:,.1f}M**"
                        )
            else:
                st.caption("No significantly underserved categories detected.")

            _oversat = insights_report.get("oversaturated", [])
            if _oversat:
                cats = ", ".join(o["category"] for o in _oversat)
                st.warning(f"Oversaturated categories: **{cats}** — avoid or differentiate heavily.")

            st.divider()

            # --- D. Consumer DNA ---
            st.subheader("Consumer DNA")
            _dna = insights_report.get("consumer_dna", [])
            if _dna:
                _dna_cols = st.columns(min(len(_dna), 3))
                for i, seg in enumerate(_dna[:3]):
                    with _dna_cols[i]:
                        st.markdown(f"### {seg['name']}")
                        st.markdown(f"**{seg['pct']:.0f}%** of population ({seg['population']:,})")
                        pb = seg.get("playbook")
                        if pb:
                            st.markdown(f"_{pb.get('description', '')}_")
                            st.markdown(f"**Price sensitivity**: {pb.get('price_sensitivity', 'N/A')}")
                            st.markdown(f"**Channel**: {pb.get('channel_preference', 'N/A')}")
                            st.markdown(f"**Frequency**: {pb.get('shopping_frequency', 'N/A')}")
                            affs = pb.get("retail_affinity", [])
                            if affs:
                                st.markdown(f"**Retail affinity**: {', '.join(affs)}")
                            st.markdown(f"**Messaging**: {pb.get('messaging', '')}")
                            st.markdown(f"**Promo cadence**: {pb.get('promo_cadence', '')}")
                            st.markdown(f"**Loyalty**: {pb.get('loyalty_strategy', '')}")

                # Segment share chart
                if len(_dna) > 0 and psycho_summary_ov is not None and not psycho_summary_ov.empty:
                    ps_chart = psycho_summary_ov.copy()
                    if "segment_name" in ps_chart.columns and "population" in ps_chart.columns:
                        ps_chart = ps_chart.sort_values("population", ascending=True)
                        fig_seg = px.bar(ps_chart, x="population", y="segment_name",
                                         orientation="h", title="Segment Distribution by Population")
                        fig_seg.update_layout(height=300, yaxis_title=None,
                                              xaxis_title="Population", margin=dict(l=0, r=20, t=40, b=20))
                        st.plotly_chart(fig_seg, use_container_width=True)
            else:
                st.caption("Psychographic data not available.")

            st.divider()

            # --- E. Revenue Potential ---
            st.subheader("Revenue Potential by Category")
            _rev_pot = insights_report.get("revenue_potential", [])
            if _rev_pot:
                rev_df = pd.DataFrame(_rev_pot)
                rev_display = rev_df[["category", "status", "actual_stores", "saturation",
                                       "tam", "est_new_entrant_rev", "supportable_rent_mid",
                                       "market_avg_rent"]].copy()
                rev_display.columns = ["Category", "Status", "Stores", "Saturation",
                                        "TAM ($)", "Est. New Entrant Rev ($)",
                                        "Supportable Rent ($/sqft)", "Market Avg Rent ($/sqft)"]
                rev_display["TAM ($)"] = rev_display["TAM ($)"].apply(lambda x: f"${x:,.0f}")
                rev_display["Est. New Entrant Rev ($)"] = rev_display["Est. New Entrant Rev ($)"].apply(lambda x: f"${x:,.0f}")
                rev_display["Supportable Rent ($/sqft)"] = rev_display["Supportable Rent ($/sqft)"].apply(lambda x: f"${x:,.2f}")
                rev_display["Market Avg Rent ($/sqft)"] = rev_display["Market Avg Rent ($/sqft)"].apply(lambda x: f"${x:,.2f}")
                rev_display["Saturation"] = rev_display["Saturation"].apply(lambda x: f"{x:.0%}" if isinstance(x, (int, float)) else x)
                st.dataframe(rev_display, use_container_width=True, hide_index=True)

            st.divider()

            # --- F. Competitive Landscape ---
            st.subheader("Competitive Landscape")
            _comp = insights_report.get("competition", {})
            cl1, cl2, cl3 = st.columns(3)
            cl1.metric("Concentration (HHI)", f"{_comp.get('hhi', 0):.0f}")
            cl2.metric("Competition Level", _comp.get("level", "N/A"))
            cl3.metric("Top 5 Share", f"{_comp.get('top5_share', 0):.1%}")
            st.caption(_comp.get("description", ""))

            _top5 = _comp.get("top5_stores", [])
            if _top5:
                t5_col1, t5_col2 = st.columns(2)
                with t5_col1:
                    st.markdown("**Top 5 Stores by Market Share**")
                    for i, s in enumerate(_top5, 1):
                        st.markdown(f"{i}. **{s['name']}** ({s['category']}) — {s['share']:.2%}")

                with t5_col2:
                    _cat_mix = _comp.get("category_mix", {})
                    if _cat_mix:
                        st.markdown("**Category Mix**")
                        for cat, cnt in _cat_mix.items():
                            st.markdown(f"- {cat}: {cnt} stores")

            st.divider()

            # --- G. Actionable Recommendations ---
            st.subheader("Actionable Recommendations")
            _recs = insights_report.get("recommendations", [])
            for i, rec in enumerate(_recs, 1):
                st.markdown(f"**{i}. {rec['title']}**")
                st.markdown(f"  {rec['detail']}")
                st.markdown("")

            st.divider()

            # --- G2. Development Analysis ---
            st.subheader("Development Analysis")
            _dev = insights_report.get("development", {})

            if _dev:
                # -- Price points & housing demand --
                st.markdown("#### Housing Market Indicators")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Market Tier", _dev.get("price_tier", "N/A"))
                _tenure = _dev.get("tenure_split", {})
                d2.metric("Owner-Occupied", f"{_tenure.get('owner_pct', 0):.0%}")
                d3.metric("Renter-Occupied", f"{_tenure.get('renter_pct', 0):.0%}")
                _age_comp = _dev.get("age_composition", {})
                _dominant_age = "Young" if _age_comp.get("young_pct", 0) > 0.35 else "Aging" if _age_comp.get("older_pct", 0) > 0.30 else "Working-age"
                d4.metric("Age Profile", _dominant_age)

                # Price range cards
                _hpr = _dev.get("home_price_range", {})
                _rr = _dev.get("rent_range", {})
                if _hpr.get("mid", 0) > 0:
                    st.markdown("#### Supportable Price Points")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Entry-Level Home", f"${_hpr.get('low', 0):,}")
                    p2.metric("Market Mid-Point", f"${_hpr.get('mid', 0):,}")
                    p3.metric("Premium Home", f"${_hpr.get('high', 0):,}")

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Affordable Rent", f"${_rr.get('low', 0):,}/mo")
                    r2.metric("Market Rent", f"${_rr.get('mid', 0):,}/mo")
                    r3.metric("Premium Rent", f"${_rr.get('high', 0):,}/mo")

                    st.caption("Home prices based on 3.5x income; rents on 30% of income rule. "
                               f"Max supportable home: ${_dev.get('max_home_price', 0):,} | "
                               f"Max rent: ${_dev.get('max_monthly_rent', 0):,}/mo")

                # Primary housing narrative
                _phn = _dev.get("primary_housing_narrative", [])
                if _phn:
                    st.markdown("#### Market Signals")
                    for line in _phn:
                        st.markdown(f"- {line}")

                st.markdown("")

                # -- Residential product recommendations by segment --
                _hr = _dev.get("housing_recs", [])
                if _hr:
                    st.markdown("#### Recommended Residential Products")
                    _hr_cols = st.columns(min(len(_hr), 3))
                    for i, rec in enumerate(_hr[:3]):
                        with _hr_cols[i]:
                            st.markdown(f"**{rec['segment']}** ({rec['pct']:.0f}%)")
                            st.markdown(f"Density: {rec['density']} &nbsp;|&nbsp; Tier: {rec['price_tier']}")
                            st.markdown("**Product types:**")
                            for pt in rec["product_types"]:
                                st.markdown(f"- {pt}")
                            st.markdown("**Amenities:**")
                            for am in rec["amenities"]:
                                st.markdown(f"- {am}")

                st.markdown("")

                # -- Target audiences --
                _ta = _dev.get("target_audiences", [])
                if _ta:
                    st.markdown("#### Target Audiences")
                    for aud in _ta:
                        with st.expander(f"{aud['segment']} ({aud['pct']:.0f}% of population)"):
                            st.markdown(f"_{aud.get('description', '')}_")
                            st.markdown(f"**Price sensitivity**: {aud.get('price_sensitivity', 'N/A')}")
                            st.markdown(f"**Shopping channel**: {aud.get('channel', 'N/A')}")
                            rd = aud.get("retail_draw", [])
                            if rd:
                                st.markdown(f"**Retail draws**: {', '.join(rd)}")

                st.markdown("")

                # -- Retail / commercial recommendations --
                _rr_list = _dev.get("retail_recs", [])
                if _rr_list:
                    st.markdown("#### Recommended Retail & Commercial Tenants")
                    rr_df = pd.DataFrame(_rr_list)
                    rr_df.columns = ["Store / Tenant Type", "Driven By", "Rationale"]
                    st.dataframe(rr_df, use_container_width=True, hide_index=True)

                # -- Mixed-use concept --
                _muc = _dev.get("mixed_use_concept", [])
                if _muc:
                    st.markdown("#### Mixed-Use Development Concept")
                    for line in _muc:
                        st.markdown(f"- {line}")

            # --- H. Market at a Glance (collapsible) ---
            with st.expander("Market at a Glance"):
                _dr = insights_report.get("distance_rings", {})
                if _dr:
                    st.markdown("**Population by Distance from County Center**")
                    dr_data = []
                    for ring_key in ["1mi", "3mi", "5mi", "10mi"]:
                        r = _dr.get(ring_key, {})
                        dr_data.append({
                            "Ring": ring_key,
                            "Population": f"{r.get('population', 0):,}",
                            "Households": f"{r.get('households', 0):,}",
                            "Avg Income": f"${r.get('avg_income', 0):,.0f}" if r.get("avg_income", 0) > 0 else "N/A",
                        })
                    st.dataframe(pd.DataFrame(dr_data), use_container_width=True, hide_index=True)

                _bls = insights_report.get("bls", {})
                if _bls.get("total_employment", 0) > 0:
                    st.markdown("**BLS Employment**")
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Total Employment", f"{_bls['total_employment']:,}")
                    b2.metric("Avg Weekly Wage", f"${_bls['avg_weekly_wage']:,.0f}")
                    b3.metric("Retail Employment", f"{_bls.get('retail_employment', 0):,}")
                    b4.metric("Retail Wage", f"${_bls.get('retail_wage', 0):,.0f}")

                _inc = insights_report.get("income_stats", {})
                if _inc.get("median", 0) > 0:
                    st.markdown("**Income Distribution**")
                    i1, i2, i3, i4 = st.columns(4)
                    i1.metric("Min", f"${_inc['min']:,.0f}")
                    i2.metric("Median", f"${_inc['median']:,.0f}")
                    i3.metric("Mean", f"${_inc['mean']:,.0f}")
                    i4.metric("Max", f"${_inc['max']:,.0f}")

        # ── Tab 1: Market Overview ───────────────────────────────────────
        with tab_overview:
            import plotly.express as px

            # Executive summary
            st.info(exec_summary)

            # Market vitals row
            v1, v2, v3, v4, v5, v6 = st.columns(6)
            grade_colors = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}
            v1.metric("Market Grade", f"{grade_colors.get(viability['grade'], '')} {viability['grade']}")
            v2.metric("Score", f"{viability['total']}/100")

            # Total addressable market
            total_tam = sum(total_pop * _CATEGORY_SPEND.get(c, 2000)
                            for c in stores_df["category"].unique())
            v3.metric("Total TAM", f"${total_tam / 1e6:,.0f}M")
            v4.metric("Stores / 10K pop",
                       f"{len(stores_df) / (total_pop / 10_000):.1f}" if total_pop > 0 else "N/A")
            v5.metric("Avg Spend Power",
                       f"${_safe_float(avg_income) * total_hh / 1e6:,.0f}M" if total_hh > 0 else "N/A")
            v6.metric("Competition Level",
                       "Low" if active_results["hhi"] < 15 else "Moderate" if active_results["hhi"] < 25 else "High")

            st.divider()

            # Market opportunity gaps
            st.subheader("Market Opportunity Gaps")
            st.caption("Categories relative to national benchmarks (stores per 10K population)")
            col_gap, col_tam = st.columns(2)
            with col_gap:
                gap_chart = gap_df.copy()
                gap_chart["color"] = gap_chart["status"].map(
                    {"Underserved": "#7A8470", "Balanced": "#C4A05A", "Oversaturated": "#B85C3A"})
                fig_gap = px.bar(gap_chart, x="category", y="saturation",
                                 color="status",
                                 color_discrete_map={"Underserved": "#7A8470",
                                                     "Balanced": "#C4A05A",
                                                     "Oversaturated": "#B85C3A"},
                                 title="Category Saturation Index")
                fig_gap.add_hline(y=1.0, line_dash="dash", line_color="#6B6A62",
                                  annotation_text="Market equilibrium")
                fig_gap.update_layout(height=350, xaxis_tickangle=-35,
                                      yaxis_title="Saturation (1.0 = balanced)",
                                      margin=dict(l=0, r=20, t=40, b=80))
                st.plotly_chart(fig_gap, use_container_width=True)

            with col_tam:
                # TAM by category
                tam_df = gap_df[gap_df["tam"] > 0].sort_values("tam", ascending=True)
                fig_tam = px.bar(tam_df, x="tam", y="category", orientation="h",
                                  title="Total Addressable Market by Category",
                                  color="status",
                                  color_discrete_map={"Underserved": "#7A8470",
                                                      "Balanced": "#C4A05A",
                                                      "Oversaturated": "#B85C3A"})
                fig_tam.update_layout(height=350, xaxis_tickprefix="$",
                                      xaxis_tickformat=",.0s", yaxis_title=None,
                                      margin=dict(l=0, r=20, t=40, b=20))
                st.plotly_chart(fig_tam, use_container_width=True)

            # Underserved callout
            underserved = gap_df[gap_df["status"] == "Underserved"]
            if not underserved.empty:
                st.success(
                    f"**Opportunities**: "
                    + ", ".join(f"{r['category']} ({r['actual_stores']} stores vs {r['expected_stores']:.0f} expected)"
                               for _, r in underserved.iterrows())
                )

            st.divider()

            # Top 25 stores + concentration (existing)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Top 25 Stores")
                top25 = store_results.head(25).copy()
                top25["rank"] = range(1, len(top25) + 1)
                top25["share_pct"] = top25["share"].apply(lambda x: _sf(x, '.4%'))
                top25["demand_fmt"] = top25["demand"].apply(lambda x: _sf(x, ',.0f'))

                for col_name in ["avg_rating", "square_footage", "sqft_source"]:
                    if col_name in stores_df.columns and top25.index.isin(stores_df.index).any():
                        top25[col_name] = stores_df.loc[top25.index, col_name]

                disp = ["rank", "name", "category", "share_pct", "demand_fmt"]
                rn = {"rank": "#", "name": "Store", "category": "Category",
                      "share_pct": "Share", "demand_fmt": "Demand"}

                if "square_footage" in top25.columns:
                    top25["sqft_fmt"] = top25["square_footage"].apply(lambda x: _sf(x, ',.0f') if _safe_float(x) > 0 else "-")
                    disp.append("sqft_fmt")
                    rn["sqft_fmt"] = "Sq Ft"
                if "avg_rating" in top25.columns and (top25["avg_rating"] > 0).any():
                    top25["rating_fmt"] = top25["avg_rating"].apply(lambda x: _sf(x, '.1f') if _safe_float(x) > 0 else "-")
                    disp.append("rating_fmt")
                    rn["rating_fmt"] = "Rating"

                st.dataframe(top25[disp].rename(columns=rn),
                             use_container_width=True, hide_index=True, height=550)

            with col2:
                st.subheader("Concentration")
                st.metric("Top Store", top_store["name"][:30])
                st.metric("Top Share", _sf(top_store['share'], '.4%'))
                st.metric("Top 5", _sf(active_results['top5_concentration'], '.2%'))

                hhi_val = active_results["hhi"]
                if hhi_val < 15:
                    st.success("Highly competitive")
                elif hhi_val < 25:
                    st.warning("Moderately concentrated")
                else:
                    st.error("Highly concentrated")

                st.divider()
                st.subheader("Categories")
                for cat, count in stores_df["category"].value_counts().head(8).items():
                    st.text(f"{cat:<20s} {count:>4}")

            c1, c2 = st.columns(2)
            with c1:
                top20 = store_results.head(20).copy()
                top20["label"] = top20["name"].str[:25]
                top20 = top20.iloc[::-1]
                fig = px.bar(top20, x="share", y="label", orientation="h",
                             color="share", color_continuous_scale=[[0,"#C4A05A"],[0.5,"#8F7038"],[1,"#B85C3A"]],
                             title="Top 20 by Market Share")
                fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False,
                                  xaxis_tickformat=".2%", yaxis_title=None,
                                  margin=dict(l=0, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                cat_counts = store_results["category"].value_counts().head(10)
                fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                             title="Store Categories", hole=0.4)
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            # Store map
            st.subheader("Store Map")
            center_lat = (bbox[0] + bbox[2]) / 2
            center_lon = (bbox[1] + bbox[3]) / 2
            if zip_active:
                center_lat, center_lon = zip_lat, zip_lon
            render_store_map(store_results, origins_df, center_lat, center_lon)

        # ── Tab 2: Demographics ──────────────────────────────────────────
        with tab_demo:
            import plotly.express as px
            import plotly.graph_objects as go

            st.subheader("Population Demographics")

            if demo_summary:
                # Age distribution
                c1, c2 = st.columns(2)
                with c1:
                    age_labels = ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
                    age_keys = ["age_under_18", "age_18_24", "age_25_34", "age_35_44",
                                "age_45_54", "age_55_64", "age_65_plus"]
                    age_vals = [demo_summary.get(k, 0) for k in age_keys]
                    total_age = sum(age_vals)
                    age_pcts = [v / total_age * 100 if total_age > 0 else 0 for v in age_vals]

                    fig = px.bar(x=age_labels, y=age_pcts,
                                 labels={"x": "Age Group", "y": "% of Population"},
                                 title="Age Distribution", color=age_pcts,
                                 color_continuous_scale=[[0,"#181815"],[0.5,"#8F7038"],[1,"#C4A05A"]])
                    fig.update_layout(height=350, showlegend=False, coloraxis_showscale=False,
                                      margin=dict(l=0, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    race_labels = ["White", "Black", "Asian", "Hispanic/Latino"]
                    race_keys = ["white_non_hispanic", "black_non_hispanic",
                                 "asian_non_hispanic", "hispanic_latino"]
                    race_vals = [demo_summary.get(k, 0) for k in race_keys]
                    total_race = sum(race_vals)

                    if total_race > 0:
                        fig = px.pie(values=race_vals, names=race_labels,
                                     title="Race / Ethnicity", hole=0.4,
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                        fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Race/ethnicity data not available.")

                # Key demographic metrics
                median_age_pop = sum(age_vals[i] * [9, 21, 30, 40, 50, 60, 72][i]
                                     for i in range(len(age_vals)))
                est_median_age = median_age_pop / total_age if total_age > 0 else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Population", f"{total_pop:,}")
                c2.metric("Total Households", f"{total_hh:,}")
                c3.metric("Avg Household Size", _sf(total_pop / total_hh, '.1f') if total_hh > 0 else "N/A")
                c4.metric("Est. Median Age", _sf(est_median_age, '.0f'))

            else:
                st.info("Demographic data not available. Census API may not have returned full demographics.")

            st.divider()

            # Income distribution
            st.subheader("Income Distribution")
            valid_income = origins_df[origins_df["median_income"] > 0]["median_income"]
            if not valid_income.empty:
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig = px.histogram(valid_income, nbins=30, marginal="box",
                                       labels={"value": "Median Household Income ($)"},
                                       title="Block Group Income Distribution",
                                       color_discrete_sequence=["#C4A05A"])
                    fig.update_layout(height=350, showlegend=False,
                                      margin=dict(l=0, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    pop_weighted_income = (
                        origins_df.loc[origins_df["median_income"] > 0, "median_income"]
                        * origins_df.loc[origins_df["median_income"] > 0, "population"]
                    ).sum() / origins_df.loc[origins_df["median_income"] > 0, "population"].sum()

                    st.metric("Pop-Weighted Avg Income", f"${_sf(pop_weighted_income, ',.0f')}")
                    st.metric("Median Block Group", f"${_sf(valid_income.median(), ',.0f')}")
                    st.metric("Min", f"${_sf(valid_income.min(), ',.0f')}")
                    st.metric("Max", f"${_sf(valid_income.max(), ',.0f')}")
                    st.metric("Income Range", f"${_sf(valid_income.max() - valid_income.min(), ',.0f')}")
            else:
                st.info("Income data not available.")

            st.divider()

            # BLS Employment
            st.subheader("Employment & Economy (BLS)")
            if bls_county:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Employment", _sf(bls_county.get('total_employment', 0), ',.0f'))
                c2.metric("Avg Weekly Wage", f"${_sf(bls_county.get('avg_weekly_wage', 0), ',.0f')}")
                c3.metric("Establishments", _sf(bls_county.get('total_establishments', 0), ',.0f'))
                c4.metric("Retail Estabs.", _sf(bls_retail.get('retail_establishments', 0), ',.0f'))

                breakdown = bls_county.get("industry_breakdown", [])
                if breakdown:
                    st.subheader("Top Industries")
                    bd_df = pd.DataFrame(breakdown)
                    display_cols = [c for c in ["industry_title", "employment", "avg_weekly_wage", "establishments"]
                                    if c in bd_df.columns]
                    if display_cols:
                        st.dataframe(bd_df[display_cols].rename(columns={
                            "industry_title": "Industry", "employment": "Employment",
                            "avg_weekly_wage": "Avg Weekly Wage", "establishments": "Establishments",
                        }), use_container_width=True, hide_index=True)
            else:
                st.info("BLS employment data not available for this county.")

        # ── Tab 3: Psychographics ─────────────────────────────────────────
        with tab_psycho:
            import plotly.express as px

            st.subheader("Consumer Lifestyle Segments")
            st.caption(
                "Census-derived psychographic classification using education, housing, "
                "occupation, commute, and housing structure data as proxies for "
                "PRIZM / Tapestry-style lifestyle segments."
            )

            psycho_df = st.session_state.get("psycho_df")
            psycho_summary = st.session_state.get("psycho_summary")

            if psycho_df is not None and "segment_name" in psycho_df.columns:
                valid_psycho = psycho_df[psycho_df["segment_name"].notna()]

                if not valid_psycho.empty:
                    seg_counts = valid_psycho["segment_name"].value_counts()
                    dominant = seg_counts.index[0]
                    dominant_pct = seg_counts.iloc[0] / len(valid_psycho) * 100

                    # Population-weighted dominant
                    pop_by_seg = valid_psycho.groupby("segment_name")["population"].sum()
                    pop_dominant = pop_by_seg.idxmax()
                    pop_dominant_pct = pop_by_seg.max() / pop_by_seg.sum() * 100

                    st.info(
                        f"Dominant segment: **{pop_dominant}** "
                        f"({pop_dominant_pct:.0f}% of population, "
                        f"{seg_counts.get(pop_dominant, 0)} block groups)"
                    )

                    # Row 1: Distribution chart + summary metrics
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        if psycho_summary is not None and not psycho_summary.empty:
                            colors = [_ESQUE_SEGMENT_COLORS.get(c, "#C4A05A") for c in psycho_summary["segment_code"].tolist()] if "segment_code" in psycho_summary.columns else psycho_summary["color"].tolist()
                            fig = px.bar(
                                psycho_summary,
                                x="segment_name", y="total_population",
                                color="segment_name",
                                color_discrete_sequence=colors,
                                labels={"segment_name": "Segment", "total_population": "Population"},
                                title="Segment Distribution (by population)",
                            )
                            fig.update_layout(
                                height=400, showlegend=False,
                                margin=dict(l=0, r=20, t=40, b=80),
                                xaxis_tickangle=-35,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with c2:
                        st.metric("Segments Found", len(seg_counts))
                        st.metric("Block Groups Classified", len(valid_psycho))
                        st.metric("Total Population", f"{int(valid_psycho['population'].sum()):,}")
                        if psycho_summary is not None and not psycho_summary.empty:
                            avg_inc = psycho_summary.loc[
                                psycho_summary["segment_name"] == pop_dominant, "avg_income"
                            ]
                            if not avg_inc.empty:
                                st.metric(
                                    f"Avg Income ({pop_dominant[:15]})",
                                    f"${_sf(avg_inc.iloc[0], ',.0f')}"
                                )

                    # Row 2: Segment profile cards
                    st.divider()
                    st.subheader("Segment Profiles")

                    selected_seg = st.selectbox(
                        "Select a segment to view profile",
                        seg_counts.index.tolist(),
                    )

                    # Find the segment code for the selected name
                    seg_code = None
                    if _PSYCHO_OK:
                        for code, profile in SEGMENT_PROFILES.items():
                            if profile["name"] == selected_seg:
                                seg_code = code
                                break

                    if seg_code and seg_code in SEGMENT_PROFILES:
                        playbook = build_marketing_playbook(seg_code)
                        if playbook:
                            pc1, pc2 = st.columns([1, 1])
                            with pc1:
                                st.markdown(f"### {playbook['segment_name']}")
                                st.markdown(playbook["description"])
                                st.markdown("**Shopping Profile**")
                                st.markdown(f"- Price sensitivity: **{playbook['price_sensitivity']}**")
                                st.markdown(f"- Shopping frequency: **{playbook['shopping_frequency']}**")
                                st.markdown(f"- Channel preference: **{playbook['channel_preference']}**")
                                st.markdown(f"- Retail affinity: {', '.join(playbook['retail_affinity'])}")
                            with pc2:
                                st.markdown("### Marketing Playbook")
                                st.markdown("**How to reach them:**")
                                for ch in playbook["channels"]:
                                    st.markdown(f"- {ch}")
                                st.markdown(f"**Messaging strategy:** {playbook['messaging']}")
                                st.markdown(f"**Promotion cadence:** {playbook['promo_cadence']}")
                                st.markdown(f"**Loyalty approach:** {playbook['loyalty_strategy']}")

                    # Row 3: Summary table
                    st.divider()
                    st.subheader("All Segments Summary")
                    if psycho_summary is not None and not psycho_summary.empty:
                        display_summary = psycho_summary[[
                            "segment_name", "block_groups", "total_population",
                            "avg_income", "pct_of_population",
                        ]].copy()
                        display_summary["total_population"] = display_summary["total_population"].apply(
                            lambda x: _sf(x, ',.0f'))
                        display_summary["avg_income"] = display_summary["avg_income"].apply(
                            lambda x: f"${_sf(x, ',.0f')}")
                        display_summary["pct_of_population"] = display_summary["pct_of_population"].apply(
                            lambda x: _sf(x, '.1%') if isinstance(x, (int, float)) else x)
                        st.dataframe(
                            display_summary.rename(columns={
                                "segment_name": "Segment",
                                "block_groups": "Block Groups",
                                "total_population": "Population",
                                "avg_income": "Avg Income",
                                "pct_of_population": "% of Pop",
                            }),
                            use_container_width=True, hide_index=True,
                        )

                    # Row 4: Scatter — income vs education by segment
                    st.divider()
                    st.subheader("Segment Characteristics")
                    scatter_data = []
                    for idx, row in valid_psycho.iterrows():
                        demo = row.get("demographics", {})
                        if isinstance(demo, dict):
                            scatter_data.append({
                                "median_income": _safe_float(row.get("median_income", 0)),
                                "pct_bachelors": _safe_float(demo.get("pct_bachelors_plus", 0)) * 100,
                                "pct_owner": _safe_float(demo.get("pct_owner_occupied", 0)) * 100,
                                "segment": row.get("segment_name", "Unknown"),
                                "population": int(row.get("population", 0)),
                            })
                    if scatter_data:
                        scatter_df = pd.DataFrame(scatter_data)
                        scatter_df = scatter_df[scatter_df["median_income"] > 0]
                        if not scatter_df.empty:
                            fig = px.scatter(
                                scatter_df, x="median_income", y="pct_bachelors",
                                color="segment", size="population",
                                labels={
                                    "median_income": "Median Income ($)",
                                    "pct_bachelors": "% Bachelor's Degree+",
                                    "segment": "Segment",
                                },
                                title="Income vs Education by Segment",
                                opacity=0.7,
                            )
                            fig.update_layout(
                                height=450, margin=dict(l=0, r=20, t=40, b=20),
                                xaxis_tickformat="$,.0f",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No block groups could be classified. Census data may be too sparse.")
            else:
                st.info("Psychographic data not available. Run an analysis to generate segments.")

        # ── Tab 4: Competition ───────────────────────────────────────────
        with tab_competition:
            import plotly.express as px

            st.subheader("Competitive Landscape")

            # Fair share analysis
            n_stores = len(stores_df)
            fair_share = 1.0 / n_stores if n_stores > 0 else 0
            fsi_df = store_results.head(20).copy()
            fsi_df["fair_share"] = fair_share
            fsi_df["fsi"] = fsi_df["share"] / fair_share if fair_share > 0 else 0
            fsi_df["performance"] = fsi_df["fsi"].apply(
                lambda x: "Over-performs" if x > 1.1 else ("Under-performs" if x < 0.9 else "Fair share"))

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Fair Share Index (top 20)")
                fsi_display = fsi_df[["name", "share", "fsi", "performance"]].copy()
                fsi_display["share"] = fsi_display["share"].apply(lambda x: _sf(x, '.4%'))
                fsi_display["fsi"] = fsi_display["fsi"].apply(lambda x: _sf(x, '.2f'))
                st.dataframe(fsi_display.rename(columns={
                    "name": "Store", "share": "Share", "fsi": "FSI", "performance": "Status"
                }), use_container_width=True, hide_index=True, height=400)

            with c2:
                # Distance decay
                avg_dist = dist_matrix.mean(axis=0)
                plot_df = pd.DataFrame({
                    "name": store_results["name"],
                    "avg_distance_km": avg_dist.reindex(store_results.index),
                    "share": store_results["share"],
                    "category": store_results["category"],
                }).dropna()
                fig = px.scatter(plot_df.head(200), x="avg_distance_km", y="share",
                                 color="category", hover_name="name",
                                 labels={"avg_distance_km": "Avg Distance (km)", "share": "Market Share"},
                                 title="Distance Decay", opacity=0.7)
                fig.update_layout(height=400, yaxis_tickformat=".3%",
                                  margin=dict(l=0, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            # Competing Destinations
            cd_res = model_results.get("CD")
            if cd_res and "cd_index" in cd_res and cd_res["cd_index"] is not None and "error" not in cd_res:
                st.divider()
                st.subheader("Competing Destinations (Agglomeration Index)")
                st.caption("Stores near many other stores have higher CD index. "
                           "Positive delta = clusters attract more consumers.")
                cd_idx = cd_res["cd_index"]
                if isinstance(cd_idx, pd.Series):
                    cd_plot = pd.DataFrame({
                        "store": stores_df.loc[cd_idx.index, "name"].str[:25],
                        "cd_index": cd_idx,
                    }).sort_values("cd_index", ascending=False).head(20)
                    cd_plot = cd_plot.iloc[::-1]
                    fig = px.bar(cd_plot, x="cd_index", y="store", orientation="h",
                                 color="cd_index", color_continuous_scale=[[0,"#252520"],[0.5,"#7A8470"],[1,"#C4A05A"]],
                                 title="Store Clustering Index (top 20)")
                    fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False,
                                      yaxis_title=None, margin=dict(l=0, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            # Model comparison
            st.divider()
            st.subheader("Model Results")
            for name, res in model_results.items():
                if res is None:
                    continue
                elif "error" in res:
                    st.warning(f"**{name}**: {res['error']}")
                else:
                    st.success(f"**{name}**: OK")

            # Demand heatmap
            st.divider()
            st.subheader("Demand Heatmap")
            opts = store_results.head(15).apply(
                lambda r: f"{r['name'][:30]} ({_sf(r['share'], '.4%')})", axis=1)
            sel = st.selectbox("Select store", opts.values)
            idx = opts[opts == sel].index[0]
            row = store_results.loc[idx]
            render_origin_heatmap(results["prob_df"], origins_df, idx, row["lat"], row["lon"])

        # ── Tab 4: Scenario ──────────────────────────────────────────────
        with tab_scenario:
            st.subheader("New Store Scenario")
            st.caption("Place a hypothetical new store and see market impact.")

            center_lat = (bbox[0] + bbox[2]) / 2
            center_lon = (bbox[1] + bbox[3]) / 2
            if zip_active:
                center_lat, center_lon = zip_lat, zip_lon

            # -- Store identity --
            sc1, sc2 = st.columns(2)
            with sc1:
                new_name = st.text_input("Store Name", value="New Store")
            with sc2:
                cat_options = sorted(set(
                    list(_CATEGORY_MAP.values()) + stores_df["category"].dropna().unique().tolist()
                ))
                new_category = st.selectbox("Category", cat_options,
                                            index=cat_options.index("grocery") if "grocery" in cat_options else 0)

            # -- Location & size --
            c1, c2, c3 = st.columns(3)
            with c1:
                new_lat = st.number_input("Latitude", value=center_lat, format="%.6f")
            with c2:
                new_lon = st.number_input("Longitude", value=center_lon, format="%.6f")
            with c3:
                new_sqft = st.number_input("Square Footage", value=5000, step=500, min_value=500)

            # -- Store attributes --
            a1, a2 = st.columns(2)
            with a1:
                new_rating = st.slider("Expected Rating", 0.0, 5.0, 4.0, 0.1,
                                       help="0 = ignore rating in attractiveness")
            with a2:
                new_price = st.selectbox("Price Level",
                                         ["budget", "moderate", "premium", "luxury"],
                                         index=1)

            # -- Target audience (optional) --
            target_segments = []
            if _PSYCHO_OK:
                seg_names = sorted([p["name"] for p in SEGMENT_PROFILES.values()])
                target_segments = st.multiselect(
                    "Target audience segments (optional)",
                    seg_names,
                    help="Highlights demand from these segments in results",
                )

            if st.button("Simulate", type="primary"):
                with st.spinner("Running scenario..."):
                    scenario = simulate_new_store(
                        origins_df, stores_df, new_lat, new_lon, float(new_sqft),
                        st.session_state.get("alpha", 1.0),
                        st.session_state.get("lam", 2.0),
                        name=new_name, category=new_category,
                        rating=new_rating, price_level=new_price,
                    )

                import plotly.express as px
                import plotly.graph_objects as go

                new_probs = scenario.get("new_store_probs")
                rev = scenario.get("revenue")
                psycho_df_sc = st.session_state.get("psycho_df")

                # ── 1. CONSUMER INSIGHT NARRATIVE ─────────────────────
                # Pre-compute values used by narrative and later sections
                rings = _compute_distance_rings(new_lat, new_lon, origins_df)

                # Dominant segment
                dom_segment, dom_pct = None, None
                if _PSYCHO_OK and psycho_df_sc is not None and new_probs is not None:
                    prob_al = new_probs.reindex(origins_df.index).fillna(0)
                    al_ps = psycho_df_sc.reindex(origins_df.index).dropna(subset=["segment_name"])
                    if not al_ps.empty:
                        al_ps["w"] = prob_al.reindex(al_ps.index).fillna(0)
                        sw = al_ps.groupby("segment_name")["w"].sum()
                        if sw.sum() > 0:
                            dom_segment = sw.idxmax()
                            dom_pct = sw.max() / sw.sum() * 100

                # Category fit
                fit_score, fit_segments = None, None
                if new_probs is not None:
                    fit_score, fit_segments = _category_fit_score(
                        new_category, psycho_df_sc, origins_df, new_probs)

                # Cannibalization: same-category within 2km
                cannib_stores = []
                same_cat = stores_df[stores_df["category"] == new_category]
                if not same_cat.empty:
                    dists_c = same_cat.apply(
                        lambda r: _haversine_km(new_lat, new_lon, r["lat"], r["lon"]), axis=1)
                    near = dists_c[dists_c <= 2.0]
                    cannib_stores = same_cat.loc[near.index, "name"].tolist()

                # Weighted income for narrative
                w_income = 0.0
                if new_probs is not None:
                    prob_al2 = new_probs.reindex(origins_df.index).fillna(0)
                    tc = prob_al2[prob_al2 > 0]
                    if tc.sum() > 0:
                        w_income = float(
                            (origins_df.loc[tc.index, "median_income"] * tc).sum() / tc.sum()
                        )

                narrative = _generate_insight_narrative(
                    new_category, new_name, rev, rings, fit_score,
                    fit_segments, dom_segment, dom_pct,
                    cannib_stores, w_income, new_price,
                )
                st.info(narrative)

                # ── 2. KEY METRICS ROW ────────────────────────────────
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("New Store Share", _sf(scenario['new_store_share'], '.4%'))
                c2.metric("New Store Demand", _sf(scenario['new_store_demand'], ',.0f'))
                c3.metric("HHI Change", _sf(scenario['hhi_scenario'], '.1f'),
                           delta=_sf(_safe_float(scenario['hhi_scenario']) - _safe_float(scenario['hhi_baseline']), '+.1f'))
                c4.metric("Baseline HHI", _sf(scenario['hhi_baseline'], '.1f'))

                # ── 3. CUSTOMER VOLUME + REVENUE FORECAST ─────────────
                st.divider()
                if rev:
                    basket = _AVG_BASKET.get(new_category, 35)
                    annual_rev = rev["annual_revenue"]
                    annual_txns = annual_rev / basket if basket > 0 else 0
                    daily_customers = annual_txns / 365
                    weekly_customers = annual_txns / 52

                    st.subheader("Revenue & Customer Volume Forecast")
                    r1, r2, r3, r4, r5 = st.columns(5)
                    r1.metric("Annual Revenue", f"${annual_rev:,.0f}")
                    r2.metric("Monthly Revenue", f"${rev['monthly_revenue']:,.0f}")
                    r3.metric("Revenue / Sq Ft",
                              f"${rev.get('revenue_per_sqft', 0):,.0f}" if rev.get("revenue_per_sqft") else "N/A")
                    r4.metric("Daily Customers", f"{daily_customers:,.0f}")
                    r5.metric("Weekly Customers", f"{weekly_customers:,.0f}")

                    v1, v2, v3 = st.columns(3)
                    v1.metric("Avg Basket Size", f"${basket:,.0f}")
                    v2.metric("Annual Transactions", f"{annual_txns:,.0f}")
                    v3.metric("Confidence Range",
                              f"${rev['confidence_low']:,.0f} – ${rev['confidence_high']:,.0f}")

                    # Revenue by segment
                    rev_seg = rev.get("revenue_by_segment")
                    if rev_seg is not None and not rev_seg.empty:
                        col_rev, col_pie = st.columns(2)
                        with col_rev:
                            rss = rev_seg.sort_values("revenue", ascending=True)
                            fig_rev = px.bar(rss, x="revenue", y="segment",
                                             orientation="h", color="revenue",
                                             color_continuous_scale=[[0,"#252520"],[0.5,"#7A8470"],[1,"#A3AE9C"]],
                                             title="Revenue by Consumer Segment")
                            fig_rev.update_layout(height=350, coloraxis_showscale=False,
                                                  xaxis_tickprefix="$", xaxis_tickformat=",.0f",
                                                  yaxis_title=None,
                                                  margin=dict(l=0, r=20, t=40, b=20))
                            st.plotly_chart(fig_rev, use_container_width=True)
                        with col_pie:
                            fig_pie = px.pie(rev_seg, values="revenue", names="segment",
                                             title="Revenue Share by Segment", hole=0.4)
                            fig_pie.update_layout(height=350,
                                                  margin=dict(l=20, r=20, t=40, b=20))
                            st.plotly_chart(fig_pie, use_container_width=True)

                # ── 4. CATEGORY FIT + COMPARABLE BENCHMARKING ─────────
                st.divider()
                col_fit, col_bench = st.columns(2)

                with col_fit:
                    st.subheader("Category-Audience Fit")
                    if fit_score is not None:
                        # Gauge-style display
                        color = "#7A8470" if fit_score >= 70 else "#C4A05A" if fit_score >= 40 else "#B85C3A"
                        st.metric("Fit Score", f"{fit_score}/100")
                        st.progress(min(fit_score / 100, 1.0))
                        if fit_segments:
                            st.caption(f"Matching segments: {', '.join(fit_segments[:5])}")
                    else:
                        st.caption("Fit score unavailable (psychographic data required)")

                with col_bench:
                    st.subheader("Comparable Store Benchmark")
                    same_cat_stores = stores_df[stores_df["category"] == new_category]
                    if not same_cat_stores.empty and store_results is not None:
                        same_cat_results = store_results.loc[
                            store_results.index.isin(same_cat_stores.index)]
                        if not same_cat_results.empty:
                            avg_share = same_cat_results["share"].mean()
                            avg_demand = same_cat_results["demand"].mean()
                            avg_sqft = same_cat_stores["square_footage"].mean() if "square_footage" in same_cat_stores.columns else 0
                            n_comp = len(same_cat_results)
                            my_share = scenario["new_store_share"]
                            share_vs = ((my_share - avg_share) / avg_share * 100) if avg_share > 0 else 0

                            b1, b2 = st.columns(2)
                            b1.metric(f"Avg {new_category} Share", _sf(avg_share, '.4%'),
                                      delta=f"{share_vs:+.1f}% vs avg")
                            b2.metric(f"Avg {new_category} Sq Ft",
                                      f"{avg_sqft:,.0f}" if avg_sqft > 0 else "N/A")
                            st.caption(f"Based on {n_comp} {new_category} stores in market")
                            if rev and avg_sqft > 0:
                                mkt_rev_sqft = avg_demand * _CATEGORY_SPEND.get(new_category, 2000) / avg_sqft if avg_sqft > 0 else 0
                                my_rev_sqft = rev.get("revenue_per_sqft", 0)
                                st.metric("Your $/sqft vs market avg",
                                          f"${my_rev_sqft:,.0f}",
                                          delta=f"${my_rev_sqft - mkt_rev_sqft:+,.0f}")
                        else:
                            st.caption(f"No {new_category} stores in current results")
                    else:
                        st.caption(f"No {new_category} stores in market for comparison")

                # ── 5. CANNIBALIZATION WARNING ─────────────────────────
                if cannib_stores:
                    st.divider()
                    st.subheader("Cannibalization Risk")
                    idf = scenario["impact_df"]
                    cannib_detail = idf[idf["name"].isin(cannib_stores)].copy()
                    if not cannib_detail.empty:
                        cannib_detail["distance_km"] = cannib_detail.apply(
                            lambda r: _haversine_km(new_lat, new_lon,
                                                    stores_df.loc[r.name, "lat"],
                                                    stores_df.loc[r.name, "lon"])
                            if r.name in stores_df.index else 0, axis=1)
                        disp_cols = ["name", "distance_km", "baseline_share",
                                     "scenario_share", "change"]
                        cd_disp = cannib_detail[disp_cols].copy()
                        cd_disp["distance_km"] = cd_disp["distance_km"].apply(lambda x: f"{x:.1f}")
                        cd_disp["baseline_share"] = cd_disp["baseline_share"].apply(lambda x: _sf(x, '.4%'))
                        cd_disp["scenario_share"] = cd_disp["scenario_share"].apply(lambda x: _sf(x, '.4%'))
                        cd_disp["change"] = cd_disp["change"].apply(lambda x: _sf(x, '.4%'))
                        cd_disp = cd_disp.rename(columns={
                            "name": "Store", "distance_km": "Dist (km)",
                            "baseline_share": "Before", "scenario_share": "After",
                            "change": "Impact",
                        })
                        st.dataframe(cd_disp, use_container_width=True, hide_index=True)
                    else:
                        for s in cannib_stores[:5]:
                            st.warning(f"{s} — within 2 km, same category")

                # ── 6. DISTANCE RING ANALYSIS ─────────────────────────
                st.divider()
                st.subheader("Distance Ring Analysis")
                ring_cols = st.columns(len(_DISTANCE_RINGS_MI))
                for i, mi in enumerate(_DISTANCE_RINGS_MI):
                    key = f"{mi}mi"
                    rd = rings.get(key, {})
                    with ring_cols[i]:
                        st.metric(f"{mi}-Mile Ring", f"{rd.get('population', 0):,}")
                        st.caption(f"{rd.get('households', 0):,} HH")
                        inc = rd.get("avg_income", 0)
                        if inc > 0:
                            st.caption(f"${inc:,.0f} avg income")

                # Ring chart
                ring_pops = [rings.get(f"{mi}mi", {}).get("population", 0) for mi in _DISTANCE_RINGS_MI]
                incremental = [ring_pops[0]]
                for j in range(1, len(ring_pops)):
                    incremental.append(max(ring_pops[j] - ring_pops[j - 1], 0))
                ring_chart = pd.DataFrame({
                    "ring": [f"{mi} mi" for mi in _DISTANCE_RINGS_MI],
                    "cumulative": ring_pops,
                    "incremental": incremental,
                })
                fig_ring = px.bar(ring_chart, x="ring", y=["incremental"],
                                  title="Population by Distance Ring",
                                  labels={"value": "Population", "ring": "Distance"})
                fig_ring.update_layout(height=300, showlegend=False,
                                       margin=dict(l=0, r=20, t=40, b=20))
                st.plotly_chart(fig_ring, use_container_width=True)

                # ── 7. NEAREST COMPETITORS TABLE ──────────────────────
                st.divider()
                st.subheader("Nearest Competitors")
                comp_df = _nearest_competitors(new_lat, new_lon, stores_df, store_results, n=10)
                comp_disp = comp_df.copy()
                comp_disp["distance_mi"] = comp_disp["distance_mi"].apply(lambda x: f"{x:.1f}")
                comp_disp["distance_km"] = comp_disp["distance_km"].apply(lambda x: f"{x:.1f}")
                if "share" in comp_disp.columns:
                    comp_disp["share"] = comp_disp["share"].apply(lambda x: _sf(x, '.4%') if pd.notna(x) else "-")
                if "sqft" in comp_disp.columns:
                    comp_disp["sqft"] = comp_disp["sqft"].apply(lambda x: f"{x:,.0f}" if _safe_float(x) > 0 else "-")
                if "rating" in comp_disp.columns:
                    comp_disp["rating"] = comp_disp["rating"].apply(lambda x: f"{x:.1f}" if _safe_float(x) > 0 else "-")
                rename_map = {"name": "Store", "category": "Category",
                              "distance_mi": "Dist (mi)", "distance_km": "Dist (km)",
                              "sqft": "Sq Ft", "rating": "Rating", "share": "Share"}
                comp_disp = comp_disp.rename(columns=rename_map)
                st.dataframe(comp_disp, use_container_width=True, hide_index=True, height=390)

                # ── 8. SCENARIO MAP ───────────────────────────────────
                st.divider()
                st.subheader("Scenario Map")
                try:
                    import folium
                    from streamlit_folium import st_folium

                    sc_map = folium.Map(location=[new_lat, new_lon], zoom_start=13,
                                        tiles="CartoDB dark_matter")

                    # New store pin (star)
                    folium.Marker(
                        location=[new_lat, new_lon],
                        popup=f"<b>{new_name}</b><br>{new_category}<br>{new_sqft:,} sqft",
                        icon=folium.Icon(color="darkgreen", icon="star", prefix="fa"),
                    ).add_to(sc_map)

                    # Existing stores (small circles)
                    for sid, row in stores_df.iterrows():
                        d = _haversine_km(new_lat, new_lon, row["lat"], row["lon"])
                        if d <= 15:
                            clr = "red" if row["category"] == new_category else "blue"
                            folium.CircleMarker(
                                location=[row["lat"], row["lon"]], radius=4,
                                color=clr, fill=True, fill_color=clr, fill_opacity=0.6,
                                popup=f"<b>{row['name']}</b><br>{row['category']}<br>{d:.1f} km",
                            ).add_to(sc_map)

                    # Catchment heatmap from probability
                    if new_probs is not None:
                        heat_data = []
                        probs_al = new_probs.reindex(origins_df.index).fillna(0)
                        for idx in probs_al[probs_al > 0.005].index:
                            if idx in origins_df.index:
                                heat_data.append([
                                    origins_df.loc[idx, "lat"],
                                    origins_df.loc[idx, "lon"],
                                    float(probs_al.loc[idx]),
                                ])
                        if heat_data:
                            from folium.plugins import HeatMap
                            HeatMap(heat_data, radius=18, blur=15,
                                    max_zoom=13, min_opacity=0.3).add_to(sc_map)

                    # Distance rings (circles)
                    for mi in [1, 3, 5]:
                        folium.Circle(
                            location=[new_lat, new_lon],
                            radius=mi * 1609.34,
                            color="gray", weight=1, fill=False,
                            dash_array="5",
                            popup=f"{mi}-mile ring",
                        ).add_to(sc_map)

                    st_folium(sc_map, width=None, height=500, returned_objects=[])
                except Exception:
                    st.caption("Map unavailable (folium/streamlit-folium required)")

                # ── 9. COMPETITIVE IMPACT CHARTS ──────────────────────
                st.divider()
                st.subheader("Competitive Impact")
                col_impact, col_cat = st.columns(2)

                with col_impact:
                    impact = scenario["impact_df"].head(15).copy()
                    impact["label"] = impact["name"].str[:25]
                    impact = impact.iloc[::-1]
                    fig = px.bar(impact, x="change", y="label", orientation="h",
                                 color="change", color_continuous_scale=[[0,"#B85C3A"],[0.5,"#A89F92"],[1,"#7A8470"]],
                                 title="Most Impacted Stores")
                    fig.update_layout(height=450, coloraxis_showscale=False,
                                      xaxis_tickformat=".4%", yaxis_title=None,
                                      margin=dict(l=0, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                with col_cat:
                    idf = scenario["impact_df"]
                    cat_impact = idf.groupby("category")["change"].mean().sort_values()
                    cat_df = pd.DataFrame({"category": cat_impact.index,
                                           "avg_share_change": cat_impact.values})
                    fig2 = px.bar(cat_df, x="avg_share_change", y="category",
                                  orientation="h", color="avg_share_change",
                                  color_continuous_scale=[[0,"#B85C3A"],[0.5,"#A89F92"],[1,"#7A8470"]],
                                  title="Impact by Category")
                    fig2.update_layout(height=450, coloraxis_showscale=False,
                                       xaxis_tickformat=".4%", yaxis_title=None,
                                       margin=dict(l=0, r=20, t=40, b=20))
                    st.plotly_chart(fig2, use_container_width=True)

                # ── 10. CATCHMENT PROFILE + INCOME DISTRIBUTION ───────
                if new_probs is not None and not new_probs.empty:
                    st.divider()
                    st.subheader("Catchment Profile")

                    prob_aligned = new_probs.reindex(origins_df.index).fillna(0)
                    top_catchment = prob_aligned[prob_aligned > 0].sort_values(ascending=False)
                    n_bg = (top_catchment > 0.01).sum()
                    catch_pop = origins_df.loc[top_catchment.index, "population"]
                    weighted_pop = (catch_pop * top_catchment).sum()

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Catchment Block Groups", f"{n_bg:,}")
                    m2.metric("Weighted Population", f"{weighted_pop:,.0f}")
                    m3.metric("Avg Income (weighted)", f"${w_income:,.0f}")

                    # Income distribution histogram
                    col_hist, col_seg = st.columns(2)
                    with col_hist:
                        catch_incomes = origins_df.loc[
                            top_catchment.index, "median_income"
                        ].dropna()
                        catch_incomes = catch_incomes[catch_incomes > 0]
                        if not catch_incomes.empty:
                            fig_hist = px.histogram(
                                catch_incomes, nbins=20,
                                title="Catchment Income Distribution",
                                labels={"value": "Median Income", "count": "Block Groups"},
                            )
                            # Price level overlay line
                            price_lines = {"budget": 35000, "moderate": 60000,
                                           "premium": 90000, "luxury": 130000}
                            pl_val = price_lines.get(new_price, 60000)
                            fig_hist.add_vline(x=pl_val, line_dash="dash",
                                               line_color="#B85C3A",
                                               annotation_text=f"{new_price} target")
                            fig_hist.update_layout(
                                height=350, showlegend=False,
                                xaxis_tickprefix="$", xaxis_tickformat=",.0f",
                                margin=dict(l=0, r=20, t=40, b=20))
                            st.plotly_chart(fig_hist, use_container_width=True)

                    # Psychographic catchment breakdown
                    with col_seg:
                        if _PSYCHO_OK and psycho_df_sc is not None and not psycho_df_sc.empty:
                            catch_psycho = psycho_df_sc.reindex(top_catchment.index).dropna(subset=["segment_name"])
                            if not catch_psycho.empty:
                                catch_psycho["weight"] = top_catchment.reindex(catch_psycho.index).fillna(0)
                                sw = catch_psycho.groupby("segment_name")["weight"].sum()
                                sp = (sw / sw.sum() * 100).sort_values(ascending=False)
                                seg_chart = pd.DataFrame({"segment": sp.index, "pct": sp.values})
                                fig3 = px.bar(seg_chart, x="pct", y="segment",
                                              orientation="h", title="Catchment Lifestyle Segments",
                                              color="pct", color_continuous_scale=[[0,"#252520"],[0.5,"#7A8470"],[1,"#C4A05A"]])
                                fig3.update_layout(height=350, coloraxis_showscale=False,
                                                   xaxis_title="% of catchment", yaxis_title=None,
                                                   margin=dict(l=0, r=20, t=40, b=20))
                                st.plotly_chart(fig3, use_container_width=True)

                    # Target audience match
                    if target_segments and _PSYCHO_OK and psycho_df_sc is not None:
                        st.divider()
                        st.subheader("Target Audience Analysis")
                        catch_psycho = psycho_df_sc.reindex(top_catchment.index).dropna(subset=["segment_name"])
                        if not catch_psycho.empty:
                            catch_psycho["weight"] = top_catchment.reindex(catch_psycho.index).fillna(0)
                            sw = catch_psycho.groupby("segment_name")["weight"].sum()
                            sp = (sw / sw.sum() * 100)
                            target_pct = sum(sp.get(s, 0) for s in target_segments)

                            t1, t2 = st.columns(2)
                            t1.metric("Target Audience Match", f"{target_pct:.1f}%")
                            name_to_code = {p["name"]: c for c, p in SEGMENT_PROFILES.items()}
                            target_demand = 0
                            for seg_name in target_segments:
                                code = name_to_code.get(seg_name)
                                if code:
                                    seg_mask = catch_psycho["segment_code"] == code
                                    seg_bg = catch_psycho[seg_mask].index
                                    seg_pop = origins_df.loc[seg_bg, "population"]
                                    seg_prob = top_catchment.reindex(seg_bg).fillna(0)
                                    target_demand += (seg_pop * seg_prob).sum()
                            t2.metric("Target Segment Demand", f"{target_demand:,.0f}")

                            for seg_name in target_segments:
                                code = name_to_code.get(seg_name)
                                if code and code in SEGMENT_PROFILES:
                                    cb = SEGMENT_PROFILES[code].get("consumer_behavior", {})
                                    st.markdown(f"**{seg_name}**")
                                    st.text(f"  Retail affinity: {', '.join(cb.get('retail_affinity', []))}")
                                    st.text(f"  Price sensitivity: {cb.get('price_sensitivity', 'N/A')}")
                                    st.text(f"  Channel: {cb.get('channel_preference', 'N/A')}")

                # ── Supportable rent & trade area penetration ────────
                if rev and rev.get("annual_revenue", 0) > 0:
                    st.divider()
                    st.subheader("Lease Economics")
                    rent_data = compute_supportable_rent(
                        rev["annual_revenue"], float(new_sqft), new_category)
                    if rent_data:
                        le1, le2, le3, le4 = st.columns(4)
                        le1.metric("Max Rent (conservative)",
                                   f"${rent_data['rent_low']:,.0f}/sqft/yr")
                        le2.metric("Target Rent (8% ratio)",
                                   f"${rent_data['rent_mid']:,.0f}/sqft/yr")
                        le3.metric("Market Avg Rent",
                                   f"${rent_data['market_avg']:,.0f}/sqft/yr")
                        if rent_data["can_afford_market"]:
                            le4.metric("Rent Verdict", "Feasible")
                            st.success(
                                f"Projected revenue supports market-rate rent. "
                                f"Occupancy cost at ${rent_data['rent_mid']:,.0f}/sqft = "
                                f"{rent_data['rent_mid'] * float(new_sqft) / rev['annual_revenue'] * 100:.1f}% of revenue."
                            )
                        else:
                            le4.metric("Rent Verdict", "Challenging")
                            st.warning(
                                f"Market rent (${rent_data['market_avg']}/sqft) exceeds "
                                f"recommended occupancy ratio. Negotiate below "
                                f"${rent_data['rent_mid']:,.0f}/sqft or increase store volume."
                            )

                    # Trade area penetration
                    sc_rings = _compute_distance_rings(new_lat, new_lon, origins_df)
                    pop_3 = sc_rings.get("3mi", {}).get("population", 0)
                    if pop_3 > 0 and rev.get("annual_revenue", 0) > 0:
                        basket = _AVG_BASKET.get(new_category, 35)
                        annual_txns = rev["annual_revenue"] / basket if basket > 0 else 0
                        penetration = annual_txns / (pop_3 * 52) * 100 if pop_3 > 0 else 0
                        st.metric("Trade Area Penetration (3mi)",
                                  f"{penetration:.1f}% of population/week",
                                  help="% of 3-mile population visiting weekly")

                # ── Revenue model details (collapsed) ─────────────────
                if rev:
                    with st.expander("Revenue Model Details"):
                        st.markdown(f"""
**Model inputs:**
- **Category baseline**: ${_CATEGORY_SPEND.get(new_category, 2000):,}/year per capita ({new_category})
- **Avg basket size**: ${_AVG_BASKET.get(new_category, 35)} ({new_category})
- **Income adjustment**: local median income vs. national (${_NATIONAL_MEDIAN_INCOME:,}), elasticity {_INCOME_ELASTICITY}
- **Psychographic multiplier**: spending propensity x visit frequency by lifestyle segment
- **Gravity probability**: Huff model P(choose this store) per block group

**Confidence range**: 70%-135% of point estimate to account for model uncertainty,
local competitive dynamics, and seasonal variation.

**Customer volume**: Annual revenue / avg basket size = annual transactions.
Divided by 365 (daily) or 52 (weekly).
                        """)

        # ── Tab 6: Data Sources ──────────────────────────────────────────
        with tab_sources:
            st.header("Data Sources & APIs")
            st.markdown("All data sources connected to this analysis, with links to documentation.")

            _active_sources = st.session_state.get("data_sources", [])

            # --- All data sources registry ---
            _ALL_SOURCES = [
                {"name": "Census ACS", "url": "https://www.census.gov/", "status": "core",
                 "desc": "95 demographic variables at block-group level. Population, income, age, race, education, housing, commute.",
                 "key": "Optional"},
                {"name": "OpenStreetMap", "url": "https://www.openstreetmap.org/", "status": "core",
                 "desc": "Retail store locations, categories, and amenities via Overpass API.",
                 "key": "None"},
                {"name": "BLS QCEW", "url": "https://www.bls.gov/cew/", "status": "core",
                 "desc": "County employment, wages, and establishment counts by industry sector.",
                 "key": "None"},
                {"name": "OSRM", "url": "https://project-osrm.org/", "status": "core",
                 "desc": "Real driving-time distance matrices between origins and stores.",
                 "key": "None"},
                {"name": "Census CBP", "url": "https://www.census.gov/programs-surveys/cbp.html", "status": "active",
                 "desc": "County Business Patterns — establishment counts by NAICS sector.",
                 "key": "None"},
                {"name": "SEC EDGAR", "url": "https://www.sec.gov/search-filings", "status": "active",
                 "desc": "Public retailer financials — revenue, margins, store counts for 40+ chains.",
                 "key": "None"},
                {"name": "HealthData.gov", "url": "https://healthdata.gov/", "status": "active",
                 "desc": "CDC county health rankings, food environment data, healthcare facility counts.",
                 "key": "None"},
                {"name": "CKAN / data.gov", "url": "https://github.com/ckan/ckan", "status": "active",
                 "desc": "Government open data portal — searches for business licenses, permits, property records.",
                 "key": "None"},
                {"name": "Data Commons", "url": "https://www.datacommons.org/", "status": "api_key",
                 "desc": "Google's unified API — housing values, education, unemployment, health insurance, commute times.",
                 "key": "Free key"},
                {"name": "FRED", "url": "https://www.nber.org/", "status": "api_key",
                 "desc": "Federal Reserve Economic Data — CPI, unemployment, consumer confidence, retail sales indices.",
                 "key": "Free key"},
                {"name": "Financial Modeling Prep", "url": "https://site.financialmodelingprep.com/developer/docs", "status": "api_key",
                 "desc": "Pre-parsed retailer financial ratios, sector screening, company profiles.",
                 "key": "Free (250/day)"},
                {"name": "Zillow", "url": "https://www.zillow.com/", "status": "reference",
                 "desc": "Real estate listings and home values. Free API deprecated 2021; ZTRAX shut down 2023. No free programmatic access.",
                 "key": "N/A"},
                {"name": "ZoomInfo", "url": "https://www.zoominfo.com/", "status": "reference",
                 "desc": "Enterprise firmographic data — company headcount, revenue, locations. Starts at ~$50K/year.",
                 "key": "Enterprise"},
                {"name": "RudderStack", "url": "https://www.rudderstack.com/", "status": "reference",
                 "desc": "Customer data platform / event streaming pipeline. Not a data source — a data routing tool.",
                 "key": "N/A"},
                {"name": "Awesome Public Datasets", "url": "https://github.com/awesomedata/awesome-public-datasets", "status": "reference",
                 "desc": "Curated GitHub list of 500+ public datasets across economics, real estate, and more.",
                 "key": "N/A"},
                {"name": "Azure Open Datasets", "url": "https://azure.microsoft.com/en-us/products/open-datasets", "status": "reference",
                 "desc": "Microsoft-hosted open datasets (Census, CPI, weather). Same data available via free source APIs.",
                 "key": "Azure account"},
                {"name": "AWS Databases", "url": "https://aws.amazon.com/products/databases/", "status": "reference",
                 "desc": "Cloud database infrastructure and open data registry. Infrastructure service, not a direct data source.",
                 "key": "AWS account"},
                {"name": "Alpaca Markets", "url": "https://alpaca.markets/", "status": "reference",
                 "desc": "Stock trading and market data API. Retail stock prices — niche fit for retail site analysis.",
                 "key": "Free (limited)"},
                {"name": "NOAA NCEI", "url": "https://www.ncei.noaa.gov/", "status": "api_key",
                 "desc": "Climate normals, temperature, precipitation, snowfall. Seasonal traffic patterns and outdoor retail viability.",
                 "key": "Free token"},
            ]

            # Active sources section
            st.subheader("Active Sources in This Analysis")
            if _active_sources:
                _src_cols = st.columns(min(len(_active_sources), 4))
                for i, src_name in enumerate(_active_sources):
                    with _src_cols[i % min(len(_active_sources), 4)]:
                        st.metric(src_name, "Active")
            else:
                st.caption("Run an analysis to see active data sources.")

            st.divider()

            # Full registry
            st.subheader("All Connected Sources")
            for _src in _ALL_SOURCES:
                _status = _src["status"]
                if _status == "core":
                    _badge = "CORE"
                elif _status == "active":
                    _badge = "ACTIVE"
                elif _status == "api_key":
                    _badge = "AVAILABLE (needs key)"
                else:
                    _badge = "REFERENCE"

                _in_use = _src["name"] in " ".join(_active_sources) if _active_sources else False
                _icon = ">" if _in_use else "-"

                st.markdown(f"**{_icon} [{_src['name']}]({_src['url']})** — {_badge} | Key: {_src['key']}")
                st.caption(_src["desc"])

            # Show CKAN discovered datasets if available
            _ckan_results = st.session_state.get("ckan_data", {})
            if _ckan_results and _ckan_results.get("datasets"):
                st.divider()
                st.subheader("Discovered Government Datasets (data.gov)")
                st.caption(f"{_ckan_results.get('datasets_found', 0)} datasets found for this market area.")
                for _ds in _ckan_results["datasets"][:10]:
                    st.markdown(f"- [{_ds.get('title', 'Untitled')}]({_ds.get('url', '#')}) ({_ds.get('format', 'N/A')})")
                    if _ds.get("description"):
                        st.caption(_ds["description"][:200])

            # Show expanded data sections
            st.divider()
            st.subheader("Expanded Data Feeds")

            _exp_cols = st.columns(4)

            with _exp_cols[0]:
                st.markdown("**County Business Patterns**")
                _cbp = st.session_state.get("cbp_data", {})
                if _cbp:
                    st.metric("Total Establishments", f"{_cbp.get('total_establishments', 0):,}")
                    st.metric("Retail Establishments", f"{_cbp.get('retail_establishments', 0):,}")
                    st.metric("Food Service", f"{_cbp.get('food_service_establishments', 0):,}")
                else:
                    st.caption("Run analysis to populate.")

            with _exp_cols[1]:
                st.markdown("**Health & Wellness**")
                _hd = st.session_state.get("health_data", {})
                if _hd:
                    if "obesity_pct" in _hd:
                        st.metric("Obesity Rate", f"{_hd['obesity_pct']:.1f}%")
                    if "physical_inactivity_pct" in _hd:
                        st.metric("Physical Inactivity", f"{_hd['physical_inactivity_pct']:.1f}%")
                    if "food_environment_index" in _hd:
                        st.metric("Food Environment Index", f"{_hd['food_environment_index']:.1f}")
                    if "uninsured_pct" in _hd:
                        st.metric("Uninsured Rate", f"{_hd['uninsured_pct']:.1f}%")
                else:
                    st.caption("Run analysis to populate.")

            with _exp_cols[2]:
                st.markdown("**Macro Economic Context (FRED)**")
                _fd = st.session_state.get("fred_data", {})
                if _fd:
                    for _fk, _fv in _fd.items():
                        if isinstance(_fv, dict) and "value" in _fv:
                            st.metric(_fk.replace("_", " ").title(), f"{_fv['value']}")
                        elif isinstance(_fv, str):
                            st.metric(_fk.replace("_", " ").title(), _fv)
                else:
                    st.caption("Add a FRED API key to enable.")

            with _exp_cols[3]:
                st.markdown("**Climate (NOAA NCEI)**")
                _nd = st.session_state.get("noaa_data", {})
                if _nd:
                    _normals = _nd.get("normals", {})
                    if _normals.get("avg_temp_f"):
                        st.metric("Avg Temperature", f"{_normals['avg_temp_f']:.0f}°F")
                    if _normals.get("annual_precip_in"):
                        st.metric("Annual Precipitation", f"{_normals['annual_precip_in']:.1f}\"")
                    if _normals.get("annual_snow_in"):
                        st.metric("Annual Snowfall", f"{_normals['annual_snow_in']:.1f}\"")
                    if _nd.get("climate_type"):
                        st.metric("Climate Type", _nd["climate_type"].title())
                    if _nd.get("seasonal_risk"):
                        st.metric("Seasonal Risk", _nd["seasonal_risk"].title())
                    if _nd.get("outdoor_retail_viability"):
                        st.metric("Outdoor Retail", _nd["outdoor_retail_viability"].title())
                else:
                    st.caption("Add a NOAA token to enable.")

            # SEC/FMP benchmarks
            _sec = st.session_state.get("sec_benchmarks", [])
            _fmp = st.session_state.get("fmp_benchmarks", [])
            _benchmarks = _sec + _fmp
            if _benchmarks:
                st.divider()
                st.subheader("Public Retailer Benchmarks")
                _bm_df = pd.DataFrame(_benchmarks)
                _display_cols = [c for c in ["name", "revenue", "net_income", "gross_margin",
                                              "operating_margin", "total_stores", "year",
                                              "companyName", "mktCap", "sector"] if c in _bm_df.columns]
                if _display_cols:
                    st.dataframe(_bm_df[_display_cols], use_container_width=True, hide_index=True)

            # Data Commons
            _dc = st.session_state.get("dc_data", {})
            if _dc:
                st.divider()
                st.subheader("Data Commons Indicators")
                _dc_cols = st.columns(min(len(_dc), 4))
                for i, (_dk, _dv) in enumerate(_dc.items()):
                    with _dc_cols[i % min(len(_dc), 4)]:
                        _label = _dk.replace("_", " ").title()
                        if isinstance(_dv, float):
                            st.metric(_label, f"{_dv:,.1f}")
                        else:
                            st.metric(_label, str(_dv))

    else:
        # Landing page
        st.title("Gravity Consumer Model")
        st.markdown("""
        ### Retail Market Analysis Dashboard

        Analyze **any US county** with layered gravity models and live data
        from 7+ sources.

        **What you get:**
        - **Market Overview** — top stores, market share, concentration metrics
        - **Demographics** — age, race/ethnicity, income, households, BLS employment
        - **Psychographics** — 12 PRIZM-style lifestyle segments derived from Census data (education, housing, occupation, commute, vehicles)
        - **Competition** — fair share index, agglomeration analysis, distance decay, demand heatmaps
        - **Scenario** — hypothetical new store impact simulation

        **Data sources (all free, no keys required):**
        Census ACS (95 variables), Census CBP, OpenStreetMap, OSRM, BLS QCEW,
        SEC EDGAR, HealthData.gov, data.gov, 418-brand store size database

        **Optional (free keys):** Google Places, Yelp Fusion, FRED, Data Commons, FMP

        **Coverage:** All 50 states, 3,200+ counties, any ZIP code.

        Select a state and county in the sidebar — analysis runs automatically.
        """)


if __name__ == "__main__":
    main()
