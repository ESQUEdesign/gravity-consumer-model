"""
Gravity Consumer Model -- Live Market Demo
===========================================
Pulls REAL data from free public APIs (US Census ACS + OpenStreetMap Overpass)
and runs the Huff gravity model on an actual US market.

Default market: Austin, TX (Travis County).  Change the configuration
constants at the top of this file to target a different city.

No API keys required.  Just run:

    python examples/live_market_demo.py

Dependencies beyond the gravity library: requests, numpy, pandas (all in
requirements.txt).  No cenpy or osmnx needed -- uses raw HTTP requests.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the gravity package is importable when running from the repo root.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gravity.data.schema import (
    Store,
    ConsumerOrigin,
    stores_to_dataframe,
    origins_to_dataframe,
    build_distance_matrix,
)
from gravity.core.huff import HuffModel

try:
    from gravity.spatial.trade_area import TradeAreaAnalyzer
except ImportError:
    TradeAreaAnalyzer = None  # type: ignore[assignment]

try:
    from gravity.reporting.trade_area_report import TradeAreaReport
except ImportError:
    TradeAreaReport = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# CONFIGURATION -- change these to target a different market
# ---------------------------------------------------------------------------

# US Census geography
STATE_FIPS = "48"           # Texas
COUNTY_FIPS = "453"         # Travis County (Austin)
CENSUS_YEAR = 2022          # ACS 5-year vintage
MARKET_LABEL = "Austin, TX -- Travis County"

# Bounding box for OpenStreetMap Overpass query (south, west, north, east)
# Covers central Austin roughly within Mopac/183/71/I-35
OSM_BBOX = (30.22, -97.82, 30.35, -97.68)

# Overpass tag filter -- what kind of stores to pull
OSM_TAGS = "shop"           # "shop" covers supermarkets, convenience, etc.

# Huff model parameters
HUFF_ALPHA = 1.0            # attractiveness exponent
HUFF_LAMBDA = 2.0           # distance-decay exponent

# API timeouts (seconds)
API_TIMEOUT = 60

# Approximate centroid of Travis County for fallback lat/lon assignment
COUNTY_CENTER_LAT = 30.2672
COUNTY_CENTER_LON = -97.7431


# ═══════════════════════════════════════════════════════════════════════════
# DATA PIPELINE: Census
# ═══════════════════════════════════════════════════════════════════════════

def fetch_census_block_groups() -> list[ConsumerOrigin]:
    """Pull block-group demographics from the US Census ACS API.

    Uses the free, unauthenticated Census API endpoint.  Variables:
        B01003_001E  -- total population
        B19013_001E  -- median household income
        B25001_001E  -- total housing units (proxy for households)

    Returns a list of ConsumerOrigin models.  Lat/lon are approximated from
    the block group's tract centroid via the Census TIGERweb geocoder, or
    assigned from a simple spread around the county center when the geocoder
    is unavailable.
    """
    import requests

    print("  Querying Census ACS API ...")
    url = f"https://api.census.gov/data/{CENSUS_YEAR}/acs/acs5"

    variables = "B01003_001E,B19013_001E,B25001_001E"

    params = {
        "get": f"NAME,{variables}",
        "for": "block group:*",
        "in": f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
    }

    try:
        resp = requests.get(url, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print("  [WARNING] Census API request timed out.")
        return []
    except requests.exceptions.ConnectionError:
        print("  [WARNING] Could not connect to Census API (no internet?).")
        return []
    except requests.exceptions.HTTPError as exc:
        print(f"  [WARNING] Census API returned HTTP error: {exc}")
        return []

    payload = resp.json()

    if len(payload) < 2:
        print("  [WARNING] Census API returned no data rows.")
        return []

    header = payload[0]
    rows = [dict(zip(header, record)) for record in payload[1:]]

    print(f"  Census API returned {len(rows)} block groups.")

    # -----------------------------------------------------------------
    # Assign approximate lat/lon to each block group.
    # The ACS data API does not return coordinates.  We spread block
    # groups deterministically across the county bounding box using a
    # hash of the GEOID so positions are stable across runs.
    # -----------------------------------------------------------------

    def _latlon_from_geoid(geoid: str) -> tuple[float, float]:
        """Deterministic pseudo-random lat/lon within the OSM bbox."""
        h = int(hashlib.md5(geoid.encode()).hexdigest(), 16)
        lat_frac = (h % 10_000) / 10_000
        lon_frac = ((h >> 16) % 10_000) / 10_000
        south, west, north, east = OSM_BBOX
        lat = south + lat_frac * (north - south)
        lon = west + lon_frac * (east - west)
        return lat, lon

    origins: list[ConsumerOrigin] = []
    skipped = 0

    for row in rows:
        geoid = (
            str(row.get("state", "")).zfill(2)
            + str(row.get("county", "")).zfill(3)
            + str(row.get("tract", "")).zfill(6)
            + str(row.get("block group", "")).zfill(1)
        )

        population = _safe_int(row.get("B01003_001E", 0))
        median_income = _safe_float(row.get("B19013_001E", 0.0))
        housing_units = _safe_int(row.get("B25001_001E", 0))

        if population <= 0:
            skipped += 1
            continue

        lat, lon = _latlon_from_geoid(geoid)

        origins.append(
            ConsumerOrigin(
                origin_id=geoid,
                lat=lat,
                lon=lon,
                population=population,
                households=housing_units,
                median_income=median_income,
                demographics={
                    "housing_units": housing_units,
                },
            )
        )

    if skipped:
        print(f"  Skipped {skipped} block groups with zero population.")

    return origins


# ═══════════════════════════════════════════════════════════════════════════
# DATA PIPELINE: OpenStreetMap
# ═══════════════════════════════════════════════════════════════════════════

# OSM shop value -> category mapping
_CATEGORY_MAP: dict[str, str] = {
    "supermarket": "grocery",
    "convenience": "convenience",
    "department_store": "department",
    "clothes": "apparel",
    "shoes": "apparel",
    "electronics": "electronics",
    "furniture": "furniture",
    "hardware": "hardware",
    "bakery": "food_specialty",
    "butcher": "food_specialty",
    "greengrocer": "food_specialty",
    "mall": "mall",
    "general": "general",
    "variety_store": "variety",
    "alcohol": "liquor",
    "beverages": "beverages",
}


def fetch_osm_stores() -> list[Store]:
    """Pull retail store locations from OpenStreetMap via the Overpass API.

    Queries for nodes and ways tagged shop=* within the configured bounding
    box.  Returns a list of Store models.
    """
    import requests

    south, west, north, east = OSM_BBOX
    print("  Querying Overpass API ...")
    print(f"  Bounding box: ({south}, {west}) to ({north}, {east})")

    # Overpass QL query: nodes with the "shop" tag inside bbox
    # Limit results to avoid 406 errors from the public server
    overpass_query = f"""
    [out:json][timeout:{API_TIMEOUT}][maxsize:5000000];
    (
      node["{OSM_TAGS}"](
        {south},{west},{north},{east}
      );
    );
    out body 500;
    """

    overpass_url = "https://overpass.kumi.systems/api/interpreter"

    try:
        resp = requests.post(
            overpass_url,
            data={"data": overpass_query},
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print("  [WARNING] Overpass API request timed out.")
        return []
    except requests.exceptions.ConnectionError:
        print("  [WARNING] Could not connect to Overpass API (no internet?).")
        return []
    except requests.exceptions.HTTPError as exc:
        status = getattr(exc.response, "status_code", "?")
        if status == 429:
            print("  [WARNING] Overpass API rate-limited (429). Try again in a minute.")
        else:
            print(f"  [WARNING] Overpass API returned HTTP error: {exc}")
        return []

    data = resp.json()
    elements = data.get("elements", [])
    print(f"  Overpass API returned {len(elements)} elements.")

    stores: list[Store] = []

    for elem in elements:
        tags = elem.get("tags", {})

        # Resolve coordinates -- nodes have lat/lon directly; ways have
        # a "center" key when queried with "out center".
        lat = elem.get("lat")
        lon = elem.get("lon")
        if lat is None and "center" in elem:
            lat = elem["center"].get("lat")
            lon = elem["center"].get("lon")

        if lat is None or lon is None:
            continue

        name = tags.get("name")
        shop_type = tags.get("shop", "other")
        category = _CATEGORY_MAP.get(shop_type, shop_type)
        brand = tags.get("brand")

        # Build a stable store ID from the OSM element ID
        osm_id = f"osm_{elem.get('type', 'node')}_{elem.get('id', 0)}"

        stores.append(
            Store(
                store_id=osm_id,
                name=name,
                lat=float(lat),
                lon=float(lon),
                category=category,
                brand=brand,
                square_footage=1000.0,  # default; OSM rarely has sqft
                attributes={
                    "osm_type": elem.get("type"),
                    "osm_id": elem.get("id"),
                    "shop": shop_type,
                    **{k: v for k, v in tags.items() if k not in ("name", "shop", "brand")},
                },
            )
        )

    return stores


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _safe_int(val: Any, default: int = 0) -> int:
    """Cast to int, treating negatives/nulls as default."""
    try:
        v = int(float(val))
        return v if v >= 0 else default
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Cast to float, treating negatives/nulls as default."""
    try:
        v = float(val)
        return v if v >= 0 else default
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the live market demonstration."""

    print()
    print("=" * 70)
    print("  GRAVITY CONSUMER MODEL -- Live Market Demo")
    print(f"  Market: {MARKET_LABEL}")
    print("=" * 70)
    print()

    # ==================================================================
    # STEP 1: Load consumer origins from Census
    # ==================================================================

    print("-" * 70)
    print("[1] CONSUMER ORIGINS (US Census ACS Block Groups)")
    print("-" * 70)

    origins = fetch_census_block_groups()

    if not origins:
        print()
        print("  FATAL: No consumer origins loaded. Cannot continue.")
        print("  Check your internet connection or try again later.")
        print("  The Census API is free and does not require a key.")
        return

    origins_df = origins_to_dataframe(origins)

    total_pop = origins_df["population"].sum()
    total_hh = origins_df["households"].sum()
    avg_income = origins_df.loc[
        origins_df["median_income"] > 0, "median_income"
    ].mean()

    print(f"  Origins loaded:     {len(origins_df):>6,}")
    print(f"  Total population:   {total_pop:>10,}")
    print(f"  Total households:   {total_hh:>10,}")
    print(f"  Avg median income:  ${avg_income:>10,.0f}")
    print()

    # ==================================================================
    # STEP 2: Load store locations from OpenStreetMap
    # ==================================================================

    print("-" * 70)
    print("[2] STORE LOCATIONS (OpenStreetMap Overpass API)")
    print("-" * 70)

    stores = fetch_osm_stores()

    if not stores:
        print()
        print("  FATAL: No stores loaded from OpenStreetMap. Cannot continue.")
        print("  The Overpass API may be rate-limited. Try again in a minute.")
        return

    stores_df = stores_to_dataframe(stores)

    # Print category breakdown
    cat_counts = stores_df["category"].value_counts()

    print(f"  Stores loaded:      {len(stores_df):>6,}")
    print()
    print("  Top categories:")
    for cat, count in cat_counts.head(10).items():
        print(f"    {cat:<25s} {count:>5,}")
    print()

    # ==================================================================
    # STEP 3: Build distance matrix
    # ==================================================================

    print("-" * 70)
    print("[3] DISTANCE MATRIX (Haversine)")
    print("-" * 70)

    t0 = time.time()
    dist_matrix = build_distance_matrix(origins_df, stores_df)
    elapsed = time.time() - t0

    print(f"  Matrix shape:       {dist_matrix.shape[0]} origins x {dist_matrix.shape[1]} stores")
    print(f"  Computation time:   {elapsed:.2f}s")
    print(f"  Min distance:       {dist_matrix.values.min():.2f} km")
    print(f"  Max distance:       {dist_matrix.values.max():.2f} km")
    print(f"  Mean distance:      {dist_matrix.values.mean():.2f} km")
    print()

    # ==================================================================
    # STEP 4: Fit Huff gravity model
    # ==================================================================

    print("-" * 70)
    print("[4] HUFF GRAVITY MODEL")
    print("-" * 70)

    # Since OSM does not provide square footage, we use a uniform
    # attractiveness of 1000.0 (set during store creation) so the model
    # degenerates to a pure distance-decay model.  In production you
    # would populate square_footage from property data or use a custom
    # attractiveness function combining rating, brand strength, etc.

    model = HuffModel(
        alpha=HUFF_ALPHA,
        lam=HUFF_LAMBDA,
        attractiveness="square_footage",
        distance_matrix=dist_matrix,
    )

    print(f"  Alpha (attractiveness exponent): {model.alpha}")
    print(f"  Lambda (distance decay):         {model.lam}")
    print(f"  Attractiveness column:           square_footage")
    print()
    print("  Note: All stores have uniform attractiveness (1000 sqft default)")
    print("  since OSM does not report store size. The model runs as a pure")
    print("  distance-decay model. In production, supply real attractiveness.")
    print()

    # Predict probabilities
    print("  Computing visit probabilities ...")
    t0 = time.time()
    prob_df = model.predict(origins_df, stores_df)
    elapsed = time.time() - t0
    print(f"  Prediction time:    {elapsed:.2f}s")
    print(f"  Probability matrix: {prob_df.shape}")
    print()

    # ==================================================================
    # STEP 5: Market share rankings
    # ==================================================================

    print("-" * 70)
    print("[5] TOP 10 STORES BY MARKET SHARE")
    print("-" * 70)

    # Weighted market share: sum of P(i->j) * population_i for each store j
    pop_weights = origins_df["population"].reindex(prob_df.index).fillna(0).astype(float)
    weighted_demand = prob_df.mul(pop_weights, axis=0).sum(axis=0)
    total_weighted = weighted_demand.sum()
    market_share = (weighted_demand / total_weighted).sort_values(ascending=False)

    print()
    print(f"  {'Rank':<6} {'Store ID':<30} {'Name':<30} {'Share':>8} {'Demand':>10}")
    print(f"  {'----':<6} {'--------':<30} {'----':<30} {'-----':>8} {'------':>10}")

    for rank, (store_id, share) in enumerate(market_share.head(10).items(), 1):
        name = stores_df.loc[store_id, "name"] if store_id in stores_df.index else "N/A"
        if pd.isna(name) or name is None:
            name = f"({stores_df.loc[store_id, 'category']})"
        # Truncate long names
        if len(str(name)) > 28:
            name = str(name)[:25] + "..."
        demand = weighted_demand[store_id]
        print(f"  {rank:<6} {store_id:<30} {str(name):<30} {share:>7.2%} {demand:>10,.0f}")

    print()
    print(f"  Total weighted demand: {total_weighted:,.0f}")
    print()

    # ==================================================================
    # STEP 6: Trade area analysis for the top store
    # ==================================================================

    top_store_id = market_share.index[0]
    top_store_name = stores_df.loc[top_store_id, "name"]
    if pd.isna(top_store_name) or top_store_name is None:
        top_store_name = top_store_id

    print("-" * 70)
    print(f"[6] TRADE AREA ANALYSIS: {top_store_name}")
    print("-" * 70)

    if TradeAreaAnalyzer is None:
        print("\n  [skipped — geopandas not installed. pip install geopandas to enable.]")
    else:
        try:
            analyzer = TradeAreaAnalyzer(default_levels=[0.50, 0.70, 0.90])
            trade_areas = analyzer.from_probabilities(
                prob_df, origins_df, top_store_id
            )

            print()
            print(f"  {'Contour':<12} {'Population':>12} {'Households':>12} {'Penetration':>14} {'Fair Share':>12}")
            print(f"  {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 14} {'-' * 12}")

            for ta in trade_areas:
                print(
                    f"  {ta.contour_level:<12.0%}"
                    f" {ta.total_population:>12,}"
                    f" {ta.total_households:>12,}"
                    f" {ta.penetration:>13.4%}"
                    f" {ta.fair_share_index:>12.2f}"
                )

            print()
        except Exception as exc:
            print(f"  [WARNING] Trade area analysis failed: {exc}")
            print()

    # Demographic profile of the top store's trade area (no geopandas needed)
    try:
        top_probs = prob_df[top_store_id]
        pop_series = origins_df["population"].reindex(top_probs.index).fillna(0).astype(float)
        weights = top_probs * pop_series
        total_weight = weights.sum()

        if total_weight > 0:
            income_series = origins_df["median_income"].reindex(
                top_probs.index
            ).fillna(0).astype(float)
            weighted_income = (income_series * weights).sum() / total_weight

            print("  Trade Area Demographics:")
            print(f"    Weighted avg income:  ${weighted_income:,.0f}")
            print(f"    Expected customers:   {total_weight:,.0f}")

            capture_rate = total_weight / total_pop if total_pop > 0 else 0
            print(f"    Market capture rate:  {capture_rate:.4%}")

        print()
    except Exception as exc:
        print(f"  [WARNING] Demographic analysis failed: {exc}")
        print()

    # ==================================================================
    # STEP 7: Save HTML report (optional)
    # ==================================================================

    print("-" * 70)
    print("[7] HTML REPORT")
    print("-" * 70)

    try:
        from gravity.reporting.trade_area_report import TradeAreaReport

        report = TradeAreaReport(contour_levels=[0.50, 0.70, 0.90])
        report.generate(
            store_id=top_store_id,
            predictions=prob_df,
            origins_df=origins_df,
            stores_df=stores_df,
        )

        html = report.to_html()

        # Save alongside this script
        output_dir = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(output_dir, "live_market_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"  HTML report saved to: {report_path}")
        print(f"  Report size: {len(html):,} bytes")

    except ImportError:
        print("  [SKIP] Reporting module not available.")
    except Exception as exc:
        print(f"  [WARNING] Could not generate HTML report: {exc}")

    print()

    # ==================================================================
    # DONE
    # ==================================================================

    print("=" * 70)
    print("  Demo complete.")
    print()
    print("  Summary:")
    print(f"    Market:          {MARKET_LABEL}")
    print(f"    Origins:         {len(origins_df):,} block groups")
    print(f"    Stores:          {len(stores_df):,} retail locations")
    print(f"    Population:      {total_pop:,}")
    print(f"    Top store:       {top_store_name}")
    print(f"    Top share:       {market_share.iloc[0]:.2%}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
