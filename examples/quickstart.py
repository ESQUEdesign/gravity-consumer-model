"""
Gravity Consumer Model -- Quickstart Example
=============================================
A minimal working example that demonstrates the library with synthetic
data.  Generates fake stores, consumer origins, and transaction records,
then fits a GravityModel with a small layer stack and prints predictions,
a trade area report, and a simple scenario analysis.

Run from the project root:
    python examples/quickstart.py
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the gravity package is importable when running from the repo root.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gravity import GravityModel
from gravity.data.schema import (
    Store,
    ConsumerOrigin,
    Transaction,
    stores_to_dataframe,
    origins_to_dataframe,
    transactions_to_dataframe,
)


def main() -> None:
    """Run the quickstart demonstration."""

    np.random.seed(42)
    print("=" * 65)
    print("  Gravity Consumer Model -- Quickstart")
    print("=" * 65)
    print()

    # ------------------------------------------------------------------
    # 1. Generate synthetic stores
    # ------------------------------------------------------------------
    # 10 stores scattered in a bounding box roughly covering midtown
    # Manhattan (lat 40.74-40.78, lon -74.00 to -73.96).

    print("[1] Generating 10 synthetic stores ...")

    store_lats = np.random.uniform(40.74, 40.78, 10)
    store_lons = np.random.uniform(-74.00, -73.96, 10)
    store_sqft = np.random.uniform(2000, 25000, 10).astype(int)
    store_ratings = np.round(np.random.uniform(2.5, 5.0, 10), 1)
    store_prices = np.random.choice([1, 2, 3, 4], 10)

    stores = [
        Store(
            store_id=f"store_{i:03d}",
            name=f"Shop {chr(65 + i)}",
            lat=float(store_lats[i]),
            lon=float(store_lons[i]),
            square_footage=float(store_sqft[i]),
            avg_rating=float(store_ratings[i]),
            price_level=int(store_prices[i]),
            parking_spaces=int(np.random.randint(0, 50)),
            product_count=int(np.random.randint(100, 2000)),
            category="grocery",
        )
        for i in range(10)
    ]

    stores_df = stores_to_dataframe(stores)
    print(f"    Stores created: {len(stores_df)}")
    print(f"    Avg square footage: {stores_df['square_footage'].mean():,.0f}")
    print()

    # ------------------------------------------------------------------
    # 2. Generate synthetic consumer origins
    # ------------------------------------------------------------------
    # 50 block groups surrounding the store cluster.

    print("[2] Generating 50 synthetic consumer origins ...")

    origin_lats = np.random.uniform(40.72, 40.80, 50)
    origin_lons = np.random.uniform(-74.02, -73.94, 50)
    populations = np.random.randint(500, 5000, 50)
    households = (populations * np.random.uniform(0.3, 0.5, 50)).astype(int)
    incomes = np.random.uniform(30000, 150000, 50).astype(int)

    origins = [
        ConsumerOrigin(
            origin_id=f"bg_{i:03d}",
            lat=float(origin_lats[i]),
            lon=float(origin_lons[i]),
            population=int(populations[i]),
            households=int(households[i]),
            median_income=float(incomes[i]),
        )
        for i in range(50)
    ]

    origins_df = origins_to_dataframe(origins)
    print(f"    Origins created: {len(origins_df)}")
    print(f"    Total population: {origins_df['population'].sum():,}")
    print(f"    Median income range: ${origins_df['median_income'].min():,.0f}"
          f" -- ${origins_df['median_income'].max():,.0f}")
    print()

    # ------------------------------------------------------------------
    # 3. Generate synthetic transactions
    # ------------------------------------------------------------------
    # 5,000 transactions spread over the past 6 months, with each
    # transaction tied to a random consumer (origin) and store.

    print("[3] Generating 5,000 synthetic transactions ...")

    n_txn = 5_000
    base_date = datetime(2026, 1, 1)
    consumer_ids = [f"bg_{np.random.randint(0, 50):03d}" for _ in range(n_txn)]
    store_ids = [f"store_{np.random.randint(0, 10):03d}" for _ in range(n_txn)]
    timestamps = [
        base_date + timedelta(days=int(np.random.randint(0, 180)),
                              hours=int(np.random.randint(8, 22)))
        for _ in range(n_txn)
    ]
    amounts = np.round(np.random.exponential(35, n_txn), 2)

    transactions = [
        Transaction(
            transaction_id=f"txn_{i:05d}",
            consumer_id=consumer_ids[i],
            store_id=store_ids[i],
            timestamp=timestamps[i],
            amount=float(amounts[i]),
            items=int(np.random.randint(1, 15)),
        )
        for i in range(n_txn)
    ]

    transactions_df = transactions_to_dataframe(transactions)
    print(f"    Transactions: {len(transactions_df):,}")
    print(f"    Date range: {transactions_df['timestamp'].min().date()}"
          f" to {transactions_df['timestamp'].max().date()}")
    print(f"    Avg basket: ${transactions_df['amount'].mean():.2f}")
    print()

    # ------------------------------------------------------------------
    # 4. Fit the GravityModel
    # ------------------------------------------------------------------
    # Use three layers:
    #   - huff: base spatial interaction
    #   - competing_destinations: agglomeration/competition correction
    #   - latent_class: discovers behavioural consumer segments

    print("[4] Fitting GravityModel (layers: huff, competing_destinations, latent_class) ...")

    model = GravityModel(
        layers=["huff", "competing_destinations", "latent_class"],
    )

    model.fit(
        stores=stores_df,
        origins=origins_df,
        transactions=transactions_df,
    )

    print("    Model fitted successfully.")
    print()

    # Print model summary.
    print("[4a] Model summary:")
    print("-" * 65)
    print(model.summary())
    print()

    # ------------------------------------------------------------------
    # 5. Predictions
    # ------------------------------------------------------------------

    print("[5] Generating predictions ...")

    predictions = model.predict()

    print(f"    Prediction rows: {len(predictions):,}")
    print(f"    Columns: {list(predictions.columns)}")
    print()

    # Show top 10 highest-probability origin-store pairs.
    top10 = predictions.nlargest(10, "probability")[
        ["origin_id", "store_id", "probability", "segment"]
    ]
    print("    Top 10 origin-store pairs by probability:")
    print(top10.to_string(index=False))
    print()

    # Market share summary (average probability per store).
    share_summary = (
        predictions.groupby("store_id")["probability"]
        .mean()
        .sort_values(ascending=False)
    )
    print("    Average probability (market share proxy) per store:")
    for sid, share in share_summary.items():
        print(f"      {sid}: {share:.4f}")
    print()

    # ------------------------------------------------------------------
    # 6. Trade area report
    # ------------------------------------------------------------------
    # Generate a report for the store with the highest market share.

    top_store = share_summary.index[0]
    print(f"[6] Trade area report for top store: {top_store}")
    print("-" * 65)

    report = model.trade_area_report(top_store)
    report_dict = report.to_dict()

    ta_summary = report_dict.get("trade_area_summary", {})
    for level_info in ta_summary.get("contours", []):
        lv = level_info.get("contour_level", "?")
        pop = level_info.get("total_population", 0)
        pen = level_info.get("penetration", 0)
        fsi = level_info.get("fair_share_index", 0)
        print(f"    Contour {lv:.0%}: pop={pop:,}, "
              f"penetration={pen:.4f}, fair-share={fsi:.2f}")

    comp = report_dict.get("competitive_analysis", {})
    if comp:
        print()
        print("    Competitive analysis:")
        for sid, ms in list(comp.get("market_shares", {}).items())[:5]:
            print(f"      {sid}: {ms:.4f}")
    print()

    # ------------------------------------------------------------------
    # 7. Simple scenario: new competitor opens
    # ------------------------------------------------------------------

    print("[7] Scenario: new competitor opens near the top store ...")

    # Place the new store 0.005 degrees away from the top store.
    top_store_data = stores_df.loc[top_store]
    new_lat = top_store_data["lat"] + 0.005
    new_lon = top_store_data["lon"] + 0.003

    scenario_result = model.scenario(
        action="new_store",
        location=(new_lat, new_lon),
        attractiveness=15000,
        attributes={
            "name": "Competitor X",
            "avg_rating": 4.5,
            "price_level": 2,
        },
    )

    print(f"    Scenario: {scenario_result.name}")
    print()

    # Market share changes.
    print("    Market share changes (top 5 most affected):")
    changes = scenario_result.share_change.sort_values()
    for sid, delta in list(changes.head(5).items()):
        base = scenario_result.baseline_shares.get(sid, 0)
        new = scenario_result.scenario_shares.get(sid, 0)
        print(f"      {sid}: {base:.4f} -> {new:.4f} ({delta:+.4f})")

    # The new store's captured share.
    new_store_ids = [
        s for s in scenario_result.scenario_shares.index
        if s not in scenario_result.baseline_shares.index
    ]
    if new_store_ids:
        ns = new_store_ids[0]
        print(f"\n    New store ({ns}) captured share: "
              f"{scenario_result.scenario_shares[ns]:.4f}")
    print()

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    print("=" * 65)
    print("  Quickstart complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
