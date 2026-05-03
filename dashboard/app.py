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


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Gravity Consumer Model",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

_APP_VERSION = "1.4.0"  # actionable insights

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

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
        max_s = store_results["share"].max()
        min_s = store_results["share"].min()
        rng = max_s - min_s if max_s > min_s else 1

        for sid, row in store_results.head(100).iterrows():
            intensity = (row["share"] - min_s) / rng
            r = int(255 * intensity)
            g = int(100 * (1 - intensity))
            color = f"#{r:02x}{g:02x}32"
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

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
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
            icon=folium.Icon(color="red", icon="star"),
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
    .block-container { padding-top: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; }
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

        st.divider()
        st.caption(f"Census ACS + OSM + OSRM + BLS + Store DB | v{_APP_VERSION}")

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

    # ── Run analysis ─────────────────────────────────────────────────────
    if run_btn or "results" in st.session_state:
        if run_btn:
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

        # ── Tabs ─────────────────────────────────────────────────────────
        tab_overview, tab_demo, tab_psycho, tab_competition, tab_scenario = st.tabs([
            "Market Overview", "Demographics", "Psychographics", "Competition", "Scenario",
        ])

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
                    {"Underserved": "#4CAF50", "Balanced": "#FF9800", "Oversaturated": "#F44336"})
                fig_gap = px.bar(gap_chart, x="category", y="saturation",
                                 color="status",
                                 color_discrete_map={"Underserved": "#4CAF50",
                                                     "Balanced": "#FF9800",
                                                     "Oversaturated": "#F44336"},
                                 title="Category Saturation Index")
                fig_gap.add_hline(y=1.0, line_dash="dash", line_color="gray",
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
                                  color_discrete_map={"Underserved": "#4CAF50",
                                                      "Balanced": "#FF9800",
                                                      "Oversaturated": "#F44336"})
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
                             color="share", color_continuous_scale="RdYlGn_r",
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
                                 color_continuous_scale="Blues")
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
                                       color_discrete_sequence=["#636EFA"])
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
                            colors = psycho_summary["color"].tolist()
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
                                 color="cd_index", color_continuous_scale="Viridis",
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
                                             color_continuous_scale="Greens",
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
                        color = "#4CAF50" if fit_score >= 70 else "#FF9800" if fit_score >= 40 else "#F44336"
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
                                        tiles="CartoDB positron")

                    # New store pin (star)
                    folium.Marker(
                        location=[new_lat, new_lon],
                        popup=f"<b>{new_name}</b><br>{new_category}<br>{new_sqft:,} sqft",
                        icon=folium.Icon(color="green", icon="star", prefix="fa"),
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
                                 color="change", color_continuous_scale="RdBu",
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
                                  color_continuous_scale="RdBu",
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
                                               line_color="red",
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
                                              color="pct", color_continuous_scale="Viridis")
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
        Census ACS (95 variables), OpenStreetMap, OSRM, BLS QCEW, 418-brand store size database

        **Optional:** Google Places API, Yelp Fusion API (keys in sidebar)

        **Coverage:** All 50 states, 3,200+ counties, any ZIP code.

        Select a state and county in the sidebar, then click **Run Analysis**.
        """)


if __name__ == "__main__":
    main()
