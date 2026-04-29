"""
Gravity Consumer Model -- Full Pipeline Example
================================================
Comprehensive demonstration of the entire gravity consumer model stack,
from data generation through all model layers to reporting and scenario
analysis.

Layers exercised:
    - huff: base Huff gravity spatial interaction
    - competing_destinations: Fotheringham agglomeration correction
    - mixed_logit: heterogeneous discrete-choice model
    - latent_class: EM latent class segmentation
    - bayesian_update: real-time Dirichlet-Multinomial posterior updating
    - residual_boost: XGBoost residual learning on structural errors

Demonstrations:
    1. Synthetic data generation (stores, origins, transactions, choice data)
    2. Model fitting with the full layer stack
    3. Predictions with segment labels, HMM states, CLV, confidence intervals
    4. Segment profiles and CLV analysis
    5. HMM consumer state distribution
    6. Scenario comparison (new store vs. store closure)
    7. Real-time Bayesian updating

Run from the project root:
    python examples/full_pipeline.py
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
    build_distance_matrix,
)


# ===========================================================================
# Synthetic data generators
# ===========================================================================

def _generate_stores(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Generate a rich set of synthetic stores.

    Creates *n* stores in a bounding box roughly covering the Chicago
    Loop / Near North Side area, with varied attributes including
    square footage, ratings, price levels, parking, and product counts.

    Parameters
    ----------
    n : int
        Number of stores.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Stores DataFrame indexed by store_id.
    """
    rng = np.random.default_rng(seed)

    brands = ["FreshMart", "CityGrocers", "ValueStop", "PrimePantry",
              "QuickBasket", "GreenLeaf", "MetroMeals", None]
    categories = ["grocery", "convenience", "supermarket", "specialty"]

    stores = []
    for i in range(n):
        stores.append(Store(
            store_id=f"S{i:03d}",
            name=f"Store {i:03d}",
            lat=float(rng.uniform(41.87, 41.92)),
            lon=float(rng.uniform(-87.66, -87.61)),
            square_footage=float(rng.integers(1500, 40000)),
            avg_rating=float(round(rng.uniform(2.0, 5.0), 1)),
            price_level=int(rng.choice([1, 2, 3, 4])),
            parking_spaces=int(rng.integers(0, 80)),
            product_count=int(rng.integers(200, 5000)),
            brand=str(rng.choice(brands)) if rng.random() > 0.3 else None,
            category=str(rng.choice(categories)),
        ))
    return stores_to_dataframe(stores)


def _generate_origins(n: int = 80, seed: int = 123) -> pd.DataFrame:
    """Generate synthetic consumer origin points (block groups).

    Parameters
    ----------
    n : int
        Number of origins.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Origins DataFrame indexed by origin_id.
    """
    rng = np.random.default_rng(seed)

    segment_codes = ["A01", "B02", "C03", "D04", "E05"]

    origins = []
    for i in range(n):
        pop = int(rng.integers(300, 8000))
        origins.append(ConsumerOrigin(
            origin_id=f"BG{i:04d}",
            lat=float(rng.uniform(41.85, 41.94)),
            lon=float(rng.uniform(-87.68, -87.59)),
            population=pop,
            households=int(pop * rng.uniform(0.3, 0.5)),
            median_income=float(rng.integers(25000, 200000)),
            segment_code=str(rng.choice(segment_codes)),
            demographics={
                "pct_college": float(round(rng.uniform(0.1, 0.8), 2)),
                "median_age": float(round(rng.uniform(25, 55), 1)),
                "pct_homeowner": float(round(rng.uniform(0.1, 0.7), 2)),
            },
        ))
    return origins_to_dataframe(origins)


def _generate_transactions(
    stores_df: pd.DataFrame,
    origins_df: pd.DataFrame,
    n: int = 15_000,
    seed: int = 456,
) -> pd.DataFrame:
    """Generate synthetic transaction data with realistic distance-decay.

    Closer origins have higher transaction probabilities with nearby
    stores, mimicking real-world shopping behaviour.

    Parameters
    ----------
    stores_df : pd.DataFrame
        Stores.
    origins_df : pd.DataFrame
        Origins.
    n : int
        Number of transactions.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Transactions DataFrame.
    """
    rng = np.random.default_rng(seed)

    dist_matrix = build_distance_matrix(origins_df, stores_df)

    # Build a probability matrix based on inverse-distance weighting.
    inv_dist = 1.0 / (dist_matrix.values + 0.01) ** 2
    prob_matrix = inv_dist / inv_dist.sum(axis=1, keepdims=True)

    origin_ids = list(origins_df.index)
    store_ids = list(stores_df.index)
    base_date = datetime(2025, 7, 1)

    txns = []
    for i in range(n):
        # Pick a random origin (weighted by population).
        pop_weights = origins_df["population"].values.astype(float)
        pop_weights /= pop_weights.sum()
        oi = rng.choice(len(origin_ids), p=pop_weights)
        origin_id = origin_ids[oi]

        # Pick a store based on the distance-weighted probability.
        si = rng.choice(len(store_ids), p=prob_matrix[oi])
        store_id = store_ids[si]

        ts = base_date + timedelta(
            days=int(rng.integers(0, 300)),
            hours=int(rng.integers(7, 22)),
            minutes=int(rng.integers(0, 60)),
        )

        txns.append(Transaction(
            transaction_id=f"T{i:06d}",
            consumer_id=origin_id,
            store_id=store_id,
            timestamp=ts,
            amount=float(round(rng.exponential(40) + 5, 2)),
            items=int(rng.integers(1, 20)),
        ))

    return transactions_to_dataframe(txns)


def _generate_observed_shares(
    stores_df: pd.DataFrame,
    origins_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute observed visit shares from transactions.

    Each row (origin) sums to 1 across stores.

    Parameters
    ----------
    stores_df : pd.DataFrame
    origins_df : pd.DataFrame
    transactions_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Origin x store observed share matrix.
    """
    counts = (
        transactions_df.groupby(["consumer_id", "store_id"])
        .size()
        .reset_index(name="visits")
    )
    pivot = counts.pivot_table(
        index="consumer_id", columns="store_id", values="visits", fill_value=0
    )
    # Ensure all stores and origins are present.
    pivot = pivot.reindex(
        index=origins_df.index, columns=stores_df.index, fill_value=0
    )
    # Normalise rows to sum to 1.
    row_sums = pivot.sum(axis=1)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    shares = pivot.div(row_sums, axis=0)
    return shares


def _generate_choice_data(
    stores_df: pd.DataFrame,
    origins_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a discrete-choice dataset for the Mixed Logit layer.

    Each row represents one choice occasion with alternative-specific
    attributes.

    Parameters
    ----------
    stores_df : pd.DataFrame
    origins_df : pd.DataFrame
    transactions_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Choice data with columns: choice_id, consumer_id, store_id,
        chosen, distance, square_footage, avg_rating, price_level.
    """
    dist_matrix = build_distance_matrix(origins_df, stores_df)

    # Sample up to 2000 transactions for choice-data construction.
    sample = transactions_df.head(2000)
    rows = []
    choice_id = 0

    for _, txn in sample.iterrows():
        cid = txn["consumer_id"]
        chosen_store = txn["store_id"]

        if cid not in dist_matrix.index:
            continue

        for sid in stores_df.index:
            rows.append({
                "choice_id": choice_id,
                "consumer_id": cid,
                "store_id": sid,
                "chosen": 1 if sid == chosen_store else 0,
                "distance": float(dist_matrix.loc[cid, sid]),
                "square_footage": float(stores_df.loc[sid, "square_footage"]),
                "avg_rating": float(stores_df.loc[sid, "avg_rating"]),
                "price_level": int(stores_df.loc[sid, "price_level"]),
            })
        choice_id += 1

    return pd.DataFrame(rows)


# ===========================================================================
# Main pipeline
# ===========================================================================

def main() -> None:
    """Run the full pipeline demonstration."""

    print("=" * 70)
    print("  Gravity Consumer Model -- Full Pipeline Demonstration")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 1. Data generation
    # ------------------------------------------------------------------

    print("[1] GENERATING SYNTHETIC DATA")
    print("-" * 70)

    stores_df = _generate_stores(n=20)
    print(f"    Stores: {len(stores_df)}")

    origins_df = _generate_origins(n=80)
    print(f"    Origins: {len(origins_df)} "
          f"(total pop: {origins_df['population'].sum():,})")

    transactions_df = _generate_transactions(stores_df, origins_df, n=15_000)
    print(f"    Transactions: {len(transactions_df):,}")

    observed_shares = _generate_observed_shares(stores_df, origins_df, transactions_df)
    print(f"    Observed shares matrix: {observed_shares.shape}")

    choice_data = _generate_choice_data(stores_df, origins_df, transactions_df)
    print(f"    Choice data rows: {len(choice_data):,}")
    print()

    # ------------------------------------------------------------------
    # 2. Model fitting
    # ------------------------------------------------------------------

    print("[2] FITTING GRAVITY MODEL (ALL LAYERS)")
    print("-" * 70)

    # The full layer stack: huff -> competing_destinations -> mixed_logit
    # -> latent_class -> bayesian_update -> residual_boost.
    model = GravityModel(
        layers=[
            "huff",
            "competing_destinations",
            "mixed_logit",
            "latent_class",
            "bayesian_update",
            "residual_boost",
        ],
    )

    model.fit(
        stores=stores_df,
        origins=origins_df,
        transactions=transactions_df,
        observed_shares=observed_shares,
        choice_data=choice_data,
    )

    print()
    print("    MODEL SUMMARY:")
    print("    " + "-" * 62)
    for line in model.summary().split("\n"):
        print(f"    {line}")
    print()

    # ------------------------------------------------------------------
    # 3. Predictions
    # ------------------------------------------------------------------

    print("[3] PREDICTIONS")
    print("-" * 70)

    predictions = model.predict()

    print(f"    Total prediction rows: {len(predictions):,}")
    print(f"    Columns: {list(predictions.columns)}")
    print()

    # Show probability distribution statistics.
    print("    Probability distribution:")
    prob_stats = predictions["probability"].describe()
    for stat_name in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        print(f"      {stat_name:6s}: {prob_stats[stat_name]:.6f}")
    print()

    # Segment distribution.
    if predictions["segment"].notna().any():
        seg_counts = predictions["segment"].value_counts()
        print("    Segment distribution (prediction rows):")
        for seg, cnt in seg_counts.items():
            pct = cnt / len(predictions) * 100
            print(f"      Segment {seg}: {cnt:,} rows ({pct:.1f}%)")
        print()

    # Consumer state distribution.
    if predictions["consumer_state"].notna().any():
        state_counts = predictions["consumer_state"].value_counts()
        print("    Consumer state distribution:")
        for state, cnt in state_counts.items():
            pct = cnt / len(predictions) * 100
            print(f"      {state:12s}: {cnt:,} ({pct:.1f}%)")
        print()

    # CLV statistics.
    if predictions["clv"].notna().any():
        clv_stats = predictions["clv"].dropna().describe()
        print("    CLV statistics:")
        for stat_name in ["mean", "std", "min", "50%", "max"]:
            print(f"      {stat_name:6s}: ${clv_stats[stat_name]:,.2f}")
        print()

    # Confidence interval coverage.
    if predictions["confidence_lower"].notna().any():
        coverage = (
            (predictions["probability"] >= predictions["confidence_lower"])
            & (predictions["probability"] <= predictions["confidence_upper"])
        ).mean()
        print(f"    95% CI coverage: {coverage:.1%}")
        avg_width = (
            predictions["confidence_upper"] - predictions["confidence_lower"]
        ).mean()
        print(f"    Avg CI width: {avg_width:.6f}")
        print()

    # Market share per store.
    mkt_shares = (
        predictions.groupby("store_id")["probability"]
        .mean()
        .sort_values(ascending=False)
    )
    print("    Market shares (avg probability per store):")
    for sid, share in mkt_shares.head(10).items():
        print(f"      {sid}: {share:.4f}")
    if len(mkt_shares) > 10:
        print(f"      ... ({len(mkt_shares) - 10} more stores)")
    print()

    # ------------------------------------------------------------------
    # 4. Segment profiles
    # ------------------------------------------------------------------

    print("[4] SEGMENT PROFILES")
    print("-" * 70)

    try:
        seg_report = model.segment_report()
        seg_dict = seg_report.to_dict()

        profiles = seg_dict.get("profiles", [])
        for profile in profiles:
            seg_id = profile.get("segment", "?")
            size = profile.get("size", 0)
            share = profile.get("share_pct", 0)
            avg_clv = profile.get("avg_clv", None)
            clv_str = f"${avg_clv:,.2f}" if avg_clv is not None else "N/A"
            print(f"    Segment {seg_id}:")
            print(f"      Size: {size:,} consumers ({share:.1f}%)")
            print(f"      Avg CLV: {clv_str}")

            # Loyalty state breakdown.
            states = profile.get("loyalty_states", {})
            if states:
                print("      Loyalty states:", end="")
                for state, pct in states.items():
                    print(f"  {state}={pct:.0%}", end="")
                print()

            # RFM summary.
            rfm = profile.get("rfm_summary", {})
            if rfm:
                r_avg = rfm.get("avg_recency", "?")
                f_avg = rfm.get("avg_frequency", "?")
                m_avg = rfm.get("avg_monetary", "?")
                print(f"      RFM: R={r_avg}, F={f_avg}, M={m_avg}")
            print()
    except Exception as e:
        print(f"    Segment report generation skipped: {e}")
        print()

    # ------------------------------------------------------------------
    # 5. CLV analysis
    # ------------------------------------------------------------------

    print("[5] CLV ANALYSIS")
    print("-" * 70)

    clv_layer = model.layer_objects.get("clv")
    if clv_layer is not None and hasattr(clv_layer, "_fitted") and clv_layer._fitted:
        cust_data = clv_layer.customer_data_
        if cust_data is not None and len(cust_data) > 0:
            print(f"    Customers with CLV data: {len(cust_data):,}")
            if "clv" in cust_data.columns:
                total_clv = cust_data["clv"].sum()
                avg_clv = cust_data["clv"].mean()
                median_clv = cust_data["clv"].median()
                print(f"    Total portfolio CLV: ${total_clv:,.2f}")
                print(f"    Average CLV: ${avg_clv:,.2f}")
                print(f"    Median CLV: ${median_clv:,.2f}")

                # Top 10 customers.
                top10 = cust_data.nlargest(10, "clv")
                print("\n    Top 10 customers by CLV:")
                for _, row in top10.iterrows():
                    cid = row.name if hasattr(row, "name") else "?"
                    clv_val = row.get("clv", 0)
                    freq = row.get("frequency", "?")
                    alive_p = row.get("p_alive", "?")
                    alive_str = f"{alive_p:.2f}" if isinstance(alive_p, float) else str(alive_p)
                    print(f"      {cid}: CLV=${clv_val:,.2f}, "
                          f"freq={freq}, P(alive)={alive_str}")

                # CLV tiers.
                if hasattr(clv_layer, "segment_by_clv"):
                    try:
                        tiers = clv_layer.segment_by_clv(n_tiers=5)
                        print("\n    CLV tiers:")
                        for tier_name, tier_df in tiers.items():
                            tier_size = len(tier_df)
                            tier_avg = tier_df["clv"].mean() if "clv" in tier_df.columns else 0
                            print(f"      {tier_name}: {tier_size} customers, "
                                  f"avg CLV=${tier_avg:,.2f}")
                    except Exception:
                        pass
        else:
            print("    No customer-level CLV data available.")
    else:
        print("    CLV layer not fitted (transactions may be needed).")
    print()

    # ------------------------------------------------------------------
    # 6. HMM consumer states
    # ------------------------------------------------------------------

    print("[6] HMM CONSUMER STATES")
    print("-" * 70)

    hmm_layer = model.layer_objects.get("hmm")
    if hmm_layer is not None and hasattr(hmm_layer, "_fitted") and hmm_layer._fitted:
        print(f"    HMM states: {hmm_layer.states}")
        print(f"    Observations: {hmm_layer.observations}")

        # Transition matrix.
        if hmm_layer.A is not None:
            print("\n    Transition matrix (A):")
            header = "             " + "  ".join(f"{s:>10s}" for s in hmm_layer.states)
            print(f"    {header}")
            for i, from_state in enumerate(hmm_layer.states):
                row_vals = "  ".join(f"{hmm_layer.A[i, j]:>10.4f}"
                                     for j in range(len(hmm_layer.states)))
                print(f"    {from_state:>12s}  {row_vals}")

        # Stationary distribution.
        if hmm_layer.A is not None:
            try:
                # Compute stationary distribution from transition matrix.
                eigvals, eigvecs = np.linalg.eig(hmm_layer.A.T)
                # Find the eigenvector corresponding to eigenvalue ~1.
                idx = np.argmin(np.abs(eigvals - 1.0))
                stationary = np.real(eigvecs[:, idx])
                stationary = stationary / stationary.sum()
                print("\n    Stationary (long-run) state distribution:")
                for state, prob in zip(hmm_layer.states, stationary):
                    print(f"      {state:12s}: {prob:.4f}")
            except Exception:
                pass
    else:
        print("    HMM layer not fitted (temporal data may be needed).")
    print()

    # ------------------------------------------------------------------
    # 7. Scenario comparisons
    # ------------------------------------------------------------------

    print("[7] SCENARIO COMPARISONS")
    print("-" * 70)

    # Scenario A: New store opens in the centre of the study area.
    print("\n    Scenario A: New store opens at (41.895, -87.635)")

    scenario_a = model.scenario(
        action="new_store",
        location=(41.895, -87.635),
        attractiveness=20000,
        attributes={
            "name": "MegaMart Downtown",
            "avg_rating": 4.2,
            "price_level": 2,
            "parking_spaces": 40,
        },
    )

    print(f"      Name: {scenario_a.name}")
    changes_a = scenario_a.share_change.sort_values()
    print("      Most impacted stores:")
    for sid, delta in list(changes_a.head(5).items()):
        base = scenario_a.baseline_shares.get(sid, 0)
        print(f"        {sid}: {base:.4f} -> {base + delta:.4f} ({delta:+.4f})")

    # New store's share.
    new_ids_a = [
        s for s in scenario_a.scenario_shares.index
        if s not in scenario_a.baseline_shares.index
    ]
    if new_ids_a:
        print(f"      New store share: {scenario_a.scenario_shares[new_ids_a[0]]:.4f}")

    # Scenario B: Close the weakest store.
    weakest_store = mkt_shares.index[-1]
    print(f"\n    Scenario B: Close weakest store ({weakest_store})")

    scenario_b = model.scenario(
        action="close_store",
        store_id=weakest_store,
    )

    print(f"      Name: {scenario_b.name}")
    gains = scenario_b.share_change.sort_values(ascending=False)
    print("      Biggest beneficiaries:")
    for sid, delta in list(gains.head(5).items()):
        if sid == weakest_store:
            continue
        base = scenario_b.baseline_shares.get(sid, 0)
        print(f"        {sid}: {base:.4f} -> {base + delta:.4f} ({delta:+.4f})")
    closed_share = scenario_a.baseline_shares.get(weakest_store, 0)
    print(f"      Released demand from {weakest_store}: {closed_share:.4f}")

    # Side-by-side comparison.
    print("\n    Side-by-side summary:")
    print(f"      {'':20s} {'Scenario A':>14s} {'Scenario B':>14s}")
    print(f"      {'':20s} {'(new store)':>14s} {'(close store)':>14s}")
    print("      " + "-" * 50)

    total_realloc_a = scenario_a.share_change.abs().sum() / 2
    total_realloc_b = scenario_b.share_change.abs().sum() / 2
    print(f"      {'Total share moved':20s} {total_realloc_a:>14.4f} {total_realloc_b:>14.4f}")

    max_loss_a = changes_a.min()
    max_loss_b = scenario_b.share_change.min()
    print(f"      {'Max store loss':20s} {max_loss_a:>14.4f} {max_loss_b:>14.4f}")

    max_gain_a = scenario_a.share_change.max()
    max_gain_b = gains.max()
    print(f"      {'Max store gain':20s} {max_gain_a:>14.4f} {max_gain_b:>14.4f}")

    # Revenue impact.
    if hasattr(scenario_a, "revenue_impact") and scenario_a.revenue_impact is not None:
        rev_a = scenario_a.revenue_impact.sum()
        rev_b = scenario_b.revenue_impact.sum()
        print(f"      {'Net demand change':20s} {rev_a:>+14.1f} {rev_b:>+14.1f}")
    print()

    # ------------------------------------------------------------------
    # 8. Real-time Bayesian updating
    # ------------------------------------------------------------------

    print("[8] REAL-TIME BAYESIAN UPDATING")
    print("-" * 70)

    # Pick a consumer and show their probability vector evolving.
    test_consumer = origins_df.index[0]
    top_3_stores = mkt_shares.head(3).index.tolist()

    updater = model.layer_objects.get("bayesian_update")
    if updater is not None and hasattr(updater, "get_posterior"):
        # Show the prior (structural prediction).
        try:
            prior = updater.get_posterior(test_consumer)
            if prior is not None:
                print(f"    Consumer: {test_consumer}")
                print(f"    Prior probabilities (top 3 stores):")
                for sid in top_3_stores:
                    idx = updater._store_idx.get(sid, None)
                    p = prior[idx] if idx is not None else 0.0
                    print(f"      {sid}: {p:.4f}")
                print()

                # Simulate 5 visits, each to the first top store.
                visit_store = top_3_stores[0]
                print(f"    Simulating 5 visits to {visit_store}:")
                for visit_num in range(1, 6):
                    event = {
                        "consumer_id": test_consumer,
                        "store_id": visit_store,
                        "timestamp": datetime.now().isoformat(),
                    }
                    posterior = model.update(event)

                    if posterior is not None:
                        probs_str = ", ".join(
                            f"{sid}={posterior[updater._store_idx[sid]]:.4f}"
                            for sid in top_3_stores
                            if sid in updater._store_idx
                        )
                        print(f"      After visit {visit_num}: {probs_str}")

                # Now visit a different store.
                alt_store = top_3_stores[1]
                print(f"\n    Now 3 visits to {alt_store}:")
                for visit_num in range(1, 4):
                    event = {
                        "consumer_id": test_consumer,
                        "store_id": alt_store,
                        "timestamp": datetime.now().isoformat(),
                    }
                    posterior = model.update(event)

                    if posterior is not None:
                        probs_str = ", ".join(
                            f"{sid}={posterior[updater._store_idx[sid]]:.4f}"
                            for sid in top_3_stores
                            if sid in updater._store_idx
                        )
                        print(f"      After visit {visit_num}: {probs_str}")
            else:
                print(f"    Consumer {test_consumer} not registered in updater.")
        except Exception as e:
            print(f"    Bayesian updating demo skipped: {e}")
    else:
        print("    Bayesian updater not active.")
    print()

    # ------------------------------------------------------------------
    # 9. Trade area report for the top store
    # ------------------------------------------------------------------

    print("[9] TRADE AREA REPORT")
    print("-" * 70)

    top_store = mkt_shares.index[0]
    report = model.trade_area_report(top_store)
    report_dict = report.to_dict()

    print(f"    Store: {top_store}")
    ta_summary = report_dict.get("trade_area_summary", {})
    for contour in ta_summary.get("contours", []):
        lv = contour.get("contour_level", "?")
        pop = contour.get("total_population", 0)
        hh = contour.get("total_households", 0)
        pen = contour.get("penetration", 0)
        fsi = contour.get("fair_share_index", 0)
        print(f"    Contour {lv:.0%}:")
        print(f"      Population: {pop:,}")
        print(f"      Households: {hh:,}")
        print(f"      Penetration: {pen:.4f}")
        print(f"      Fair-share index: {fsi:.2f}")

    demo = report_dict.get("demographic_profile", {})
    if demo:
        print("\n    Demographic profile:")
        for key, val in demo.items():
            if isinstance(val, float):
                print(f"      {key}: {val:,.2f}")
            else:
                print(f"      {key}: {val}")

    comp = report_dict.get("competitive_analysis", {})
    if comp:
        shares = comp.get("market_shares", {})
        if shares:
            print("\n    Competitive market shares:")
            for sid, ms in sorted(shares.items(), key=lambda x: -x[1])[:5]:
                print(f"      {sid}: {ms:.4f}")
    print()

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    print("=" * 70)
    print("  Full pipeline demonstration complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
