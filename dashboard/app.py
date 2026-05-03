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

_APP_VERSION = "1.3.0"  # consumer filters

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


def simulate_new_store(origins_df, stores_df, new_lat, new_lon, new_sqft, alpha, lam):
    new_store = pd.DataFrame([{
        "name": "NEW STORE", "lat": new_lat, "lon": new_lon,
        "category": "new", "brand": None, "square_footage": new_sqft,
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
        "baseline_share": baseline_shares,
        "scenario_share": scenario_shares,
        "change": change,
        "pct_change": (change / baseline_shares.replace(0, np.nan)) * 100,
    }).sort_values("change")

    return {
        "new_store_share": new_share, "new_store_demand": new_demand,
        "impact_df": impact_df,
        "hhi_baseline": baseline["hhi"], "hhi_scenario": scenario["hhi"],
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

        # ── Tabs ─────────────────────────────────────────────────────────
        tab_overview, tab_demo, tab_psycho, tab_competition, tab_scenario = st.tabs([
            "Market Overview", "Demographics", "Psychographics", "Competition", "Scenario",
        ])

        # ── Tab 1: Market Overview ───────────────────────────────────────
        with tab_overview:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Top 25 Stores")
                top25 = store_results.head(25).copy()
                top25["rank"] = range(1, len(top25) + 1)
                top25["share_pct"] = top25["share"].apply(lambda x: _sf(x, '.4%'))
                top25["demand_fmt"] = top25["demand"].apply(lambda x: _sf(x, ',.0f'))

                # Add enrichment columns
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

            # Charts row
            import plotly.express as px

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
                        profile = SEGMENT_PROFILES[seg_code]
                        pc1, pc2 = st.columns([1, 1])
                        with pc1:
                            st.markdown(f"**{profile['name']}** (`{seg_code}`)")
                            st.markdown(profile["description"])
                        with pc2:
                            behavior = profile.get("consumer_behavior", {})
                            st.markdown("**Consumer Behavior**")
                            st.markdown(f"- **Retail Affinity:** {', '.join(behavior.get('retail_affinity', []))}")
                            st.markdown(f"- **Price Sensitivity:** {behavior.get('price_sensitivity', 'N/A')}")
                            st.markdown(f"- **Brand Loyalty:** {behavior.get('brand_loyalty', 'N/A')}")
                            st.markdown(f"- **Shopping Frequency:** {behavior.get('shopping_frequency', 'N/A')}")
                            st.markdown(f"- **Channel Preference:** {behavior.get('channel_preference', 'N/A')}")

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

            c1, c2, c3 = st.columns(3)
            with c1:
                new_lat = st.number_input("Latitude", value=center_lat, format="%.6f")
            with c2:
                new_lon = st.number_input("Longitude", value=center_lon, format="%.6f")
            with c3:
                new_sqft = st.number_input("Square Footage", value=5000, step=500, min_value=500)

            if st.button("Simulate", type="primary"):
                with st.spinner("Running scenario..."):
                    scenario = simulate_new_store(
                        origins_df, stores_df, new_lat, new_lon, float(new_sqft),
                        st.session_state.get("alpha", 1.0),
                        st.session_state.get("lam", 2.0),
                    )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("New Store Share", _sf(scenario['new_store_share'], '.4%'))
                c2.metric("New Store Demand", _sf(scenario['new_store_demand'], ',.0f'))
                c3.metric("HHI Change", _sf(scenario['hhi_scenario'], '.1f'),
                           delta=_sf(_safe_float(scenario['hhi_scenario']) - _safe_float(scenario['hhi_baseline']), '+.1f'))
                c4.metric("Baseline HHI", _sf(scenario['hhi_baseline'], '.1f'))

                st.divider()
                import plotly.express as px
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
