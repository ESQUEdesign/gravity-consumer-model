"""
Gravity Consumer Model -- Multi-Market Batch Runner
====================================================
Runs the full Census + OSM + Huff pipeline across multiple US markets
and produces a consolidated cross-market comparison report.

Usage:
    python examples/batch_runner.py                    # run default markets
    python examples/batch_runner.py --markets all50    # all 50 state capitals
    python examples/batch_runner.py --json markets.json # custom market list

The output is:
    1. Per-market console summary
    2. A consolidated CSV with market-level metrics
    3. An HTML cross-market comparison report

No API keys required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the gravity package is importable
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

# ---------------------------------------------------------------------------
# Market definitions
# ---------------------------------------------------------------------------

@dataclass
class Market:
    """A US market defined by state/county FIPS and a bounding box."""
    label: str
    state_fips: str
    county_fips: str
    bbox: tuple[float, float, float, float]  # south, west, north, east

    def __str__(self) -> str:
        return self.label


@dataclass
class MarketResult:
    """Results from running the pipeline on a single market."""
    label: str
    state_fips: str
    county_fips: str
    n_origins: int = 0
    n_stores: int = 0
    total_population: int = 0
    total_households: int = 0
    avg_median_income: float = 0.0
    top_store_id: str = ""
    top_store_name: str = ""
    top_store_share: float = 0.0
    top_store_demand: float = 0.0
    hhi: float = 0.0  # Herfindahl-Hirschman Index
    top_5_concentration: float = 0.0
    avg_distance_km: float = 0.0
    top_categories: dict = field(default_factory=dict)
    status: str = "pending"
    error: str = ""
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Default market list: 25 diverse US markets
# ---------------------------------------------------------------------------

DEFAULT_MARKETS: list[Market] = [
    Market("New York, NY (Manhattan)", "36", "061", (40.70, -74.02, 40.82, -73.93)),
    Market("Los Angeles, CA", "06", "037", (33.95, -118.35, 34.10, -118.15)),
    Market("Chicago, IL (Cook)", "17", "031", (41.82, -87.72, 41.92, -87.60)),
    Market("Houston, TX (Harris)", "48", "201", (29.70, -95.45, 29.82, -95.30)),
    Market("Phoenix, AZ (Maricopa)", "04", "013", (33.40, -112.10, 33.52, -111.95)),
    Market("Philadelphia, PA", "42", "101", (39.92, -75.20, 40.02, -75.10)),
    Market("San Antonio, TX (Bexar)", "48", "029", (29.38, -98.55, 29.50, -98.40)),
    Market("San Diego, CA", "06", "073", (32.68, -117.20, 32.78, -117.08)),
    Market("Dallas, TX", "48", "113", (32.73, -96.85, 32.83, -96.73)),
    Market("Austin, TX (Travis)", "48", "453", (30.22, -97.82, 30.35, -97.68)),
    Market("Denver, CO", "08", "031", (39.68, -105.00, 39.78, -104.88)),
    Market("Nashville, TN (Davidson)", "47", "037", (36.10, -86.85, 36.20, -86.72)),
    Market("Portland, OR (Multnomah)", "41", "051", (45.48, -122.72, 45.56, -122.60)),
    Market("Miami, FL (Dade)", "12", "086", (25.72, -80.25, 25.82, -80.15)),
    Market("Atlanta, GA (Fulton)", "13", "121", (33.72, -84.42, 33.82, -84.33)),
    Market("Seattle, WA (King)", "53", "033", (47.58, -122.38, 47.66, -122.28)),
    Market("Boston, MA (Suffolk)", "25", "025", (42.33, -71.10, 42.38, -71.03)),
    Market("Minneapolis, MN (Hennepin)", "27", "053", (44.93, -93.32, 45.02, -93.22)),
    Market("Detroit, MI (Wayne)", "26", "163", (42.30, -83.12, 42.40, -83.00)),
    Market("Charlotte, NC (Mecklenburg)", "37", "119", (35.18, -80.88, 35.28, -80.78)),
    Market("St. Louis, MO (City)", "29", "510", (38.60, -90.28, 38.68, -90.18)),
    Market("Pittsburgh, PA (Allegheny)", "42", "003", (40.42, -80.02, 40.47, -79.93)),
    Market("Salt Lake City, UT", "49", "035", (40.72, -111.92, 40.79, -111.85)),
    Market("New Orleans, LA (Orleans)", "22", "071", (29.93, -90.10, 30.00, -90.00)),
    Market("Honolulu, HI", "15", "003", (21.28, -157.87, 21.34, -157.80)),
]

# All 50 state capitals
STATE_CAPITALS: list[Market] = [
    Market("Montgomery, AL", "01", "101", (32.34, -86.35, 32.42, -86.25)),
    Market("Juneau, AK", "02", "110", (58.28, -134.48, 58.32, -134.38)),
    Market("Phoenix, AZ", "04", "013", (33.40, -112.10, 33.52, -111.95)),
    Market("Little Rock, AR", "05", "119", (34.70, -92.32, 34.78, -92.22)),
    Market("Sacramento, CA", "06", "067", (38.54, -121.52, 38.60, -121.44)),
    Market("Denver, CO", "08", "031", (39.68, -105.00, 39.78, -104.88)),
    Market("Hartford, CT", "09", "003", (41.74, -72.72, 41.80, -72.64)),
    Market("Dover, DE", "10", "001", (39.14, -75.56, 39.18, -75.50)),
    Market("Tallahassee, FL", "12", "073", (30.42, -84.32, 30.48, -84.24)),
    Market("Atlanta, GA", "13", "121", (33.72, -84.42, 33.82, -84.33)),
    Market("Honolulu, HI", "15", "003", (21.28, -157.87, 21.34, -157.80)),
    Market("Boise, ID", "16", "001", (43.58, -116.24, 43.64, -116.16)),
    Market("Springfield, IL", "17", "167", (39.76, -89.68, 39.82, -89.60)),
    Market("Indianapolis, IN", "18", "097", (39.73, -86.20, 39.82, -86.10)),
    Market("Des Moines, IA", "19", "153", (41.55, -93.65, 41.63, -93.55)),
    Market("Topeka, KS", "20", "177", (39.02, -95.72, 39.08, -95.64)),
    Market("Frankfort, KY", "21", "073", (38.18, -84.90, 38.22, -84.84)),
    Market("Baton Rouge, LA", "22", "033", (30.40, -91.20, 30.48, -91.10)),
    Market("Augusta, ME", "23", "011", (44.30, -69.80, 44.34, -69.74)),
    Market("Annapolis, MD", "24", "003", (38.96, -76.52, 39.00, -76.46)),
    Market("Boston, MA", "25", "025", (42.33, -71.10, 42.38, -71.03)),
    Market("Lansing, MI", "26", "065", (42.70, -84.58, 42.76, -84.50)),
    Market("St. Paul, MN", "27", "123", (44.93, -93.12, 44.98, -93.04)),
    Market("Jackson, MS", "28", "049", (32.28, -90.24, 32.34, -90.14)),
    Market("Jefferson City, MO", "29", "051", (38.56, -92.20, 38.60, -92.14)),
    Market("Helena, MT", "30", "049", (46.58, -112.06, 46.62, -112.00)),
    Market("Lincoln, NE", "31", "109", (40.78, -96.72, 40.84, -96.64)),
    Market("Carson City, NV", "32", "510", (39.14, -119.80, 39.20, -119.72)),
    Market("Concord, NH", "33", "013", (43.18, -71.58, 43.24, -71.50)),
    Market("Trenton, NJ", "34", "021", (40.20, -74.80, 40.24, -74.74)),
    Market("Santa Fe, NM", "35", "049", (35.66, -105.98, 35.70, -105.92)),
    Market("Albany, NY", "36", "001", (42.62, -73.80, 42.68, -73.72)),
    Market("Raleigh, NC", "37", "183", (35.74, -78.68, 35.82, -78.58)),
    Market("Bismarck, ND", "38", "015", (46.78, -100.82, 46.84, -100.74)),
    Market("Columbus, OH", "39", "049", (39.93, -83.04, 40.02, -82.93)),
    Market("Oklahoma City, OK", "40", "109", (35.42, -97.56, 35.52, -97.46)),
    Market("Salem, OR", "41", "047", (44.90, -123.08, 44.96, -123.00)),
    Market("Harrisburg, PA", "42", "043", (40.24, -76.92, 40.30, -76.84)),
    Market("Providence, RI", "44", "007", (41.80, -71.44, 41.85, -71.38)),
    Market("Columbia, SC", "45", "079", (33.98, -81.06, 34.04, -80.98)),
    Market("Pierre, SD", "46", "065", (44.36, -100.38, 44.40, -100.32)),
    Market("Nashville, TN", "47", "037", (36.10, -86.85, 36.20, -86.72)),
    Market("Austin, TX", "48", "453", (30.22, -97.82, 30.35, -97.68)),
    Market("Salt Lake City, UT", "49", "035", (40.72, -111.92, 40.79, -111.85)),
    Market("Montpelier, VT", "50", "023", (44.24, -72.60, 44.28, -72.54)),
    Market("Richmond, VA", "51", "760", (37.52, -77.48, 37.58, -77.42)),
    Market("Olympia, WA", "53", "067", (47.02, -122.94, 47.08, -122.86)),
    Market("Charleston, WV", "54", "039", (38.33, -81.66, 38.37, -81.60)),
    Market("Madison, WI", "55", "025", (43.04, -89.44, 43.10, -89.34)),
    Market("Cheyenne, WY", "56", "021", (41.12, -104.84, 41.18, -104.78)),
]


# ---------------------------------------------------------------------------
# Category map (same as live demo)
# ---------------------------------------------------------------------------

_CATEGORY_MAP: dict[str, str] = {
    "supermarket": "grocery", "convenience": "convenience",
    "department_store": "department", "clothes": "apparel",
    "shoes": "apparel", "electronics": "electronics",
    "furniture": "furniture", "hardware": "hardware",
    "bakery": "food_specialty", "butcher": "food_specialty",
    "greengrocer": "food_specialty", "mall": "mall",
    "general": "general", "variety_store": "variety",
    "alcohol": "liquor", "beverages": "beverages",
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
CENSUS_BASE = "https://api.census.gov/data/2022/acs/acs5"
API_TIMEOUT = 60
OVERPASS_DELAY = 2.0  # seconds between Overpass requests (rate limit)


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        v = int(float(val))
        return v if v >= 0 else default
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if v >= 0 else default
    except (TypeError, ValueError):
        return default


def fetch_census(market: Market) -> list[ConsumerOrigin]:
    """Pull block-group demographics for a market from the Census ACS API."""
    import requests

    params = {
        "get": "NAME,B01003_001E,B19013_001E,B25001_001E",
        "for": "block group:*",
        "in": f"state:{market.state_fips} county:{market.county_fips}",
    }

    try:
        resp = requests.get(CENSUS_BASE, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return []

    payload = resp.json()
    if len(payload) < 2:
        return []

    header = payload[0]
    rows = [dict(zip(header, r)) for r in payload[1:]]

    south, west, north, east = market.bbox

    def _latlon(geoid: str) -> tuple[float, float]:
        h = int(hashlib.md5(geoid.encode()).hexdigest(), 16)
        return (
            south + (h % 10_000) / 10_000 * (north - south),
            west + ((h >> 16) % 10_000) / 10_000 * (east - west),
        )

    origins = []
    for row in rows:
        geoid = (
            str(row.get("state", "")).zfill(2)
            + str(row.get("county", "")).zfill(3)
            + str(row.get("tract", "")).zfill(6)
            + str(row.get("block group", "")).zfill(1)
        )
        pop = _safe_int(row.get("B01003_001E"))
        if pop <= 0:
            continue
        lat, lon = _latlon(geoid)
        origins.append(ConsumerOrigin(
            origin_id=geoid, lat=lat, lon=lon,
            population=pop,
            households=_safe_int(row.get("B25001_001E")),
            median_income=_safe_float(row.get("B19013_001E")),
        ))
    return origins


def fetch_osm(market: Market) -> list[Store]:
    """Pull retail stores from OpenStreetMap Overpass API."""
    import requests

    south, west, north, east = market.bbox
    query = f"""
    [out:json][timeout:{API_TIMEOUT}][maxsize:5000000];
    (node["shop"]({south},{west},{north},{east}););
    out body 500;
    """

    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=API_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return []

    stores = []
    for elem in resp.json().get("elements", []):
        tags = elem.get("tags", {})
        lat, lon = elem.get("lat"), elem.get("lon")
        if lat is None or lon is None:
            continue
        shop_type = tags.get("shop", "other")
        stores.append(Store(
            store_id=f"osm_{elem.get('type','node')}_{elem.get('id',0)}",
            name=tags.get("name"),
            lat=float(lat), lon=float(lon),
            category=_CATEGORY_MAP.get(shop_type, shop_type),
            brand=tags.get("brand"),
            square_footage=1000.0,
        ))
    return stores


# ---------------------------------------------------------------------------
# Pipeline: run one market
# ---------------------------------------------------------------------------

def run_market(market: Market, verbose: bool = True) -> MarketResult:
    """Run the full pipeline for a single market and return results."""
    result = MarketResult(
        label=market.label,
        state_fips=market.state_fips,
        county_fips=market.county_fips,
    )
    t0 = time.time()

    # --- Census ---
    if verbose:
        print(f"  [{market.label}] Fetching Census data ... ", end="", flush=True)
    origins = fetch_census(market)
    if not origins:
        result.status = "failed"
        result.error = "Census API returned no data"
        result.elapsed_s = time.time() - t0
        if verbose:
            print("FAILED")
        return result
    if verbose:
        print(f"{len(origins)} block groups")

    origins_df = origins_to_dataframe(origins)
    result.n_origins = len(origins_df)
    result.total_population = int(origins_df["population"].sum())
    result.total_households = int(origins_df["households"].sum())
    valid_income = origins_df.loc[origins_df["median_income"] > 0, "median_income"]
    result.avg_median_income = float(valid_income.mean()) if len(valid_income) > 0 else 0.0

    # --- Overpass (with rate-limit pause) ---
    if verbose:
        print(f"  [{market.label}] Fetching OSM stores ... ", end="", flush=True)
    time.sleep(OVERPASS_DELAY)
    stores = fetch_osm(market)
    if not stores:
        result.status = "no_stores"
        result.error = "Overpass returned no stores"
        result.elapsed_s = time.time() - t0
        if verbose:
            print("FAILED (0 stores)")
        return result
    if verbose:
        print(f"{len(stores)} stores")

    stores_df = stores_to_dataframe(stores)
    result.n_stores = len(stores_df)

    # Category breakdown
    cat_counts = stores_df["category"].value_counts()
    result.top_categories = cat_counts.head(5).to_dict()

    # --- Distance matrix ---
    dist_matrix = build_distance_matrix(origins_df, stores_df)
    result.avg_distance_km = float(dist_matrix.values.mean())

    # --- Huff model ---
    model = HuffModel(alpha=1.0, lam=2.0, attractiveness="square_footage",
                      distance_matrix=dist_matrix)
    prob_df = model.predict(origins_df, stores_df)

    # --- Market share ---
    pop_w = origins_df["population"].reindex(prob_df.index).fillna(0).astype(float)
    weighted_demand = prob_df.mul(pop_w, axis=0).sum(axis=0)
    total = weighted_demand.sum()
    shares = (weighted_demand / total).sort_values(ascending=False) if total > 0 else weighted_demand

    top_id = shares.index[0]
    top_name = stores_df.loc[top_id, "name"] if top_id in stores_df.index else top_id
    if pd.isna(top_name):
        top_name = top_id

    result.top_store_id = str(top_id)
    result.top_store_name = str(top_name)
    result.top_store_share = float(shares.iloc[0])
    result.top_store_demand = float(weighted_demand[top_id])
    result.top_5_concentration = float(shares.head(5).sum())

    # HHI (Herfindahl-Hirschman Index) — measure of market concentration
    result.hhi = float((shares ** 2).sum() * 10_000)

    result.status = "success"
    result.elapsed_s = time.time() - t0

    if verbose:
        print(f"  [{market.label}] Done in {result.elapsed_s:.1f}s "
              f"| pop={result.total_population:,} | stores={result.n_stores} "
              f"| top={result.top_store_share:.2%} | HHI={result.hhi:.0f}")

    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    markets: list[Market],
    output_dir: str,
    verbose: bool = True,
) -> list[MarketResult]:
    """Run the pipeline on all markets sequentially."""
    os.makedirs(output_dir, exist_ok=True)
    results: list[MarketResult] = []
    total = len(markets)

    print()
    print("=" * 78)
    print(f"  GRAVITY CONSUMER MODEL -- Multi-Market Batch Runner")
    print(f"  Markets: {total}")
    print(f"  Output:  {output_dir}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)
    print()

    for i, market in enumerate(markets, 1):
        print(f"  --- Market {i}/{total} ---")
        try:
            result = run_market(market, verbose=verbose)
        except Exception as exc:
            result = MarketResult(
                label=market.label,
                state_fips=market.state_fips,
                county_fips=market.county_fips,
                status="error",
                error=str(exc),
            )
            if verbose:
                print(f"  [{market.label}] ERROR: {exc}")
        results.append(result)
        print()

    # --- Save CSV ---
    csv_path = os.path.join(output_dir, "market_comparison.csv")
    _save_csv(results, csv_path)

    # --- Save HTML ---
    html_path = os.path.join(output_dir, "market_comparison.html")
    _save_html(results, html_path)

    # --- Print summary ---
    _print_summary(results)

    print(f"\n  CSV saved:  {csv_path}")
    print(f"  HTML saved: {html_path}")
    print()

    return results


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------

def _save_csv(results: list[MarketResult], path: str) -> None:
    cols = [
        "label", "state_fips", "county_fips", "status",
        "n_origins", "n_stores", "total_population", "total_households",
        "avg_median_income", "top_store_name", "top_store_share",
        "top_store_demand", "top_5_concentration", "hhi",
        "avg_distance_km", "elapsed_s", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            d = asdict(r)
            d["avg_median_income"] = round(d["avg_median_income"], 0)
            d["top_store_share"] = round(d["top_store_share"], 6)
            d["top_store_demand"] = round(d["top_store_demand"], 0)
            d["top_5_concentration"] = round(d["top_5_concentration"], 6)
            d["hhi"] = round(d["hhi"], 1)
            d["avg_distance_km"] = round(d["avg_distance_km"], 2)
            d["elapsed_s"] = round(d["elapsed_s"], 1)
            writer.writerow(d)


# ---------------------------------------------------------------------------
# Output: HTML comparison report
# ---------------------------------------------------------------------------

def _save_html(results: list[MarketResult], path: str) -> None:
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status != "success"]

    # Sort by population descending
    successful.sort(key=lambda r: r.total_population, reverse=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_pop = sum(r.total_population for r in successful)
    total_stores = sum(r.n_stores for r in successful)

    rows_html = ""
    for i, r in enumerate(successful, 1):
        hhi_class = ""
        if r.hhi > 2500:
            hhi_class = ' class="high"'
        elif r.hhi > 1500:
            hhi_class = ' class="med"'

        rows_html += f"""
        <tr>
            <td>{i}</td>
            <td><strong>{r.label}</strong></td>
            <td>{r.total_population:,}</td>
            <td>{r.n_stores}</td>
            <td>${r.avg_median_income:,.0f}</td>
            <td>{r.top_store_name}</td>
            <td>{r.top_store_share:.2%}</td>
            <td>{r.top_5_concentration:.1%}</td>
            <td{hhi_class}>{r.hhi:.0f}</td>
            <td>{r.avg_distance_km:.1f}</td>
        </tr>"""

    failed_html = ""
    if failed:
        failed_html = "<h2>Failed Markets</h2><ul>"
        for r in failed:
            failed_html += f"<li><strong>{r.label}</strong>: {r.status} — {r.error}</li>"
        failed_html += "</ul>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gravity Model — Cross-Market Comparison</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f5f5; color: #333; padding: 24px; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 20px; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat {{ background: #fff; border-radius: 8px; padding: 16px 20px;
           box-shadow: 0 1px 3px rgba(0,0,0,.1); min-width: 160px; }}
  .stat .value {{ font-size: 28px; font-weight: 700; color: #111; }}
  .stat .label {{ font-size: 13px; color: #888; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  th {{ background: #1a1a2e; color: #fff; padding: 10px 12px; text-align: left;
       font-size: 12px; text-transform: uppercase; letter-spacing: .5px; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 14px; }}
  tr:hover {{ background: #f9f9fb; }}
  td.high {{ background: #fee; color: #c00; font-weight: 600; }}
  td.med {{ background: #fff8e0; color: #a67c00; font-weight: 600; }}
  h2 {{ margin-top: 24px; margin-bottom: 12px; font-size: 18px; }}
  ul {{ margin-left: 20px; }}
  li {{ margin-bottom: 4px; }}
  .footer {{ margin-top: 24px; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Gravity Consumer Model — Cross-Market Comparison</h1>
  <p class="subtitle">Generated {now} | Huff model (alpha=1.0, lambda=2.0, uniform attractiveness)</p>

  <div class="stats">
    <div class="stat"><div class="value">{len(successful)}</div><div class="label">Markets Analyzed</div></div>
    <div class="stat"><div class="value">{total_pop:,}</div><div class="label">Total Population</div></div>
    <div class="stat"><div class="value">{total_stores:,}</div><div class="label">Total Stores</div></div>
    <div class="stat"><div class="value">{len(failed)}</div><div class="label">Failed</div></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>#</th><th>Market</th><th>Population</th><th>Stores</th>
        <th>Avg Income</th><th>Top Store</th><th>Top Share</th>
        <th>Top-5 Conc.</th><th>HHI</th><th>Avg Dist (km)</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  {failed_html}

  <p class="footer">
    HHI (Herfindahl-Hirschman Index): &lt;1500 = competitive, 1500-2500 = moderate, &gt;2500 = concentrated.<br>
    Top-5 Concentration: share of total demand captured by the 5 largest stores.<br>
    All models run with uniform store attractiveness (OSM lacks size data). Real attractiveness data would change rankings.
  </p>
</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(results: list[MarketResult]) -> None:
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status != "success"]

    print()
    print("=" * 78)
    print("  CROSS-MARKET SUMMARY")
    print("=" * 78)
    print()

    if not successful:
        print("  No markets completed successfully.")
        return

    print(f"  {'Market':<35} {'Pop':>10} {'Stores':>7} {'Income':>10} {'Top %':>7} {'HHI':>6}")
    print(f"  {'-'*35} {'-'*10} {'-'*7} {'-'*10} {'-'*7} {'-'*6}")

    for r in sorted(successful, key=lambda x: x.total_population, reverse=True):
        label = r.label[:33] + ".." if len(r.label) > 35 else r.label
        print(
            f"  {label:<35}"
            f" {r.total_population:>10,}"
            f" {r.n_stores:>7,}"
            f" ${r.avg_median_income:>8,.0f}"
            f" {r.top_store_share:>6.2%}"
            f" {r.hhi:>6.0f}"
        )

    print()
    total_pop = sum(r.total_population for r in successful)
    avg_hhi = np.mean([r.hhi for r in successful])
    print(f"  Total population:  {total_pop:>12,}")
    print(f"  Markets succeeded: {len(successful):>12,}")
    print(f"  Markets failed:    {len(failed):>12,}")
    print(f"  Avg HHI:           {avg_hhi:>12,.0f}")

    if failed:
        print()
        print("  Failed markets:")
        for r in failed:
            print(f"    - {r.label}: {r.error}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_markets_from_json(path: str) -> list[Market]:
    """Load markets from a JSON file.

    Expected format:
    [
        {"label": "Austin, TX", "state_fips": "48", "county_fips": "453",
         "bbox": [30.22, -97.82, 30.35, -97.68]},
        ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Market(
        label=m["label"],
        state_fips=str(m["state_fips"]).zfill(2),
        county_fips=str(m["county_fips"]).zfill(3),
        bbox=tuple(m["bbox"]),
    ) for m in data]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Huff gravity model across multiple US markets."
    )
    parser.add_argument(
        "--markets", default="default",
        help="Market set: 'default' (25 cities), 'all50' (state capitals), "
             "or path to a JSON file.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory. Default: ~/Library/CloudStorage/Dropbox/WORK/Claude Code/projects/gravity-consumer-model/output/",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N markets (for testing).",
    )
    args = parser.parse_args()

    # Resolve market list
    if args.markets == "default":
        markets = DEFAULT_MARKETS
    elif args.markets == "all50":
        markets = STATE_CAPITALS
    elif os.path.isfile(args.markets):
        markets = load_markets_from_json(args.markets)
    else:
        # Treat as JSON file path
        parser.error(f"Unknown market set or file not found: {args.markets}")
        return

    if args.limit:
        markets = markets[:args.limit]

    # Resolve output dir
    output_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "output"
    )

    run_batch(markets, output_dir)


if __name__ == "__main__":
    main()
