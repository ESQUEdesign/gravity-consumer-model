"""
Synthetic test data with KNOWN ground-truth parameters.

Every generator in this module is fully deterministic (seeded RNG) and produces
data whose underlying parameters are stored in the ``GROUND_TRUTH`` dict.  This
lets downstream tests verify that each model layer recovers the correct values
to within a statistical tolerance.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from gravity.data.schema import (
    ConsumerOrigin,
    Store,
    build_distance_matrix,
    haversine_distance,
    origins_to_dataframe,
    stores_to_dataframe,
)

# ---------------------------------------------------------------------------
# Ground truth parameters — the single source of truth for all tests
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    # Huff model
    "huff_alpha": 1.0,
    "huff_lambda": 2.0,
    # Store universe
    "n_stores": 10,
    "store_center_lat": 40.75,
    "store_center_lon": -73.98,
    "square_footages": [500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000],
    "ratings": [2.5, 3.0, 3.5, 4.0, 4.5, 4.0, 3.5, 4.5, 3.0, 5.0],
    # Origin universe
    "n_origins": 50,
    "population_range": (500, 5000),
    "income_range": (30_000, 120_000),
    # Observed-visits sampling
    "visit_n_total": 50_000,
    # Transaction generation
    "transaction_n": 10_000,
    "transaction_amount_shape": 3.0,   # Gamma shape (k)
    "transaction_amount_scales": {     # Gamma scale (theta) per store tier
        "small": 15.0,   # sq_ft < 3000
        "medium": 25.0,  # 3000 <= sq_ft < 10000
        "large": 40.0,   # sq_ft >= 10000
    },
    "transaction_date_start": "2025-01-01",
    "transaction_date_end": "2025-12-31",
    # Choice data
    "choice_n": 5_000,
    # Random seeds
    "seed_stores": 42,
    "seed_origins": 42,
    "seed_visits": 42,
    "seed_transactions": 42,
    "seed_choices": 42,
}


# ---------------------------------------------------------------------------
# 1. Store generation
# ---------------------------------------------------------------------------

def generate_stores(n: int = 10, seed: int = 42) -> list[Store]:
    """
    Create *n* stores in a grid pattern around Manhattan (40.75, -73.98).

    Each store gets a deterministic id ("S001" .. "S010"), known square footage,
    and known average rating drawn from the GROUND_TRUTH vectors.
    """
    rng = np.random.default_rng(seed)
    center_lat = GROUND_TRUTH["store_center_lat"]
    center_lon = GROUND_TRUTH["store_center_lon"]
    sq_footages = GROUND_TRUTH["square_footages"][:n]
    ratings = GROUND_TRUTH["ratings"][:n]

    # Lay stores out in a roughly 2-row grid, spaced ~0.01 degrees apart
    # (~1.1 km lat, ~0.85 km lon at this latitude).
    cols = int(np.ceil(n / 2))
    lat_offsets = np.tile([-0.005, 0.005], cols)[:n]
    lon_offsets = np.repeat(np.linspace(-0.01 * (cols - 1) / 2,
                                        0.01 * (cols - 1) / 2, cols), 2)[:n]

    stores: list[Store] = []
    for i in range(n):
        stores.append(
            Store(
                store_id=f"S{i + 1:03d}",
                name=f"Store {i + 1}",
                lat=round(center_lat + lat_offsets[i], 6),
                lon=round(center_lon + lon_offsets[i], 6),
                square_footage=float(sq_footages[i]),
                avg_rating=float(ratings[i]),
                price_level=int(rng.integers(1, 5)),
                parking_spaces=int(rng.integers(0, 50)),
                product_count=int(rng.integers(100, 5000)),
                category="retail",
            )
        )
    return stores


# ---------------------------------------------------------------------------
# 2. Consumer-origin generation
# ---------------------------------------------------------------------------

def generate_origins(n: int = 50, seed: int = 42) -> list[ConsumerOrigin]:
    """
    Create *n* consumer origins in a grid surrounding the store cluster.

    Population ranges from 500 to 5000; median income from 30k to 120k.
    All values are deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    center_lat = GROUND_TRUTH["store_center_lat"]
    center_lon = GROUND_TRUTH["store_center_lon"]
    pop_lo, pop_hi = GROUND_TRUTH["population_range"]
    inc_lo, inc_hi = GROUND_TRUTH["income_range"]

    # Build a grid that wraps around the store cluster.
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    lat_vals = np.linspace(center_lat - 0.04, center_lat + 0.04, rows)
    lon_vals = np.linspace(center_lon - 0.04, center_lon + 0.04, cols)
    grid_lats, grid_lons = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    grid_lats = grid_lats.ravel()[:n]
    grid_lons = grid_lons.ravel()[:n]

    populations = np.linspace(pop_lo, pop_hi, n).astype(int)
    incomes = np.linspace(inc_lo, inc_hi, n).astype(float)

    # Shuffle populations and incomes so they are not spatially correlated
    # in a trivial way (but still deterministic).
    rng.shuffle(populations)
    rng.shuffle(incomes)

    origins: list[ConsumerOrigin] = []
    for i in range(n):
        origins.append(
            ConsumerOrigin(
                origin_id=f"O{i + 1:03d}",
                lat=round(float(grid_lats[i]), 6),
                lon=round(float(grid_lons[i]), 6),
                population=int(populations[i]),
                households=int(populations[i] // rng.integers(2, 5)),
                median_income=round(float(incomes[i]), 2),
            )
        )
    return origins


# ---------------------------------------------------------------------------
# 3. Huff probability ground truth
# ---------------------------------------------------------------------------

def generate_huff_ground_truth(
    stores: list[Store],
    origins: list[ConsumerOrigin],
    alpha: float = 1.0,
    lam: float = 2.0,
) -> pd.DataFrame:
    """
    Compute the TRUE Huff probability matrix.

    P(origin i -> store j) =  A_j^alpha / d_ij^lambda
                              --------------------------
                              sum_k A_k^alpha / d_ik^lambda

    where A_j = square_footage of store j and d_ij = haversine distance (km).

    Returns a DataFrame indexed by origin_id with columns = store_id, values
    are probabilities that sum to 1.0 across each row.
    """
    stores_df = stores_to_dataframe(stores)
    origins_df = origins_to_dataframe(origins)
    dist_km = build_distance_matrix(origins_df, stores_df)

    # Replace zero distances with a small epsilon to avoid division by zero
    dist_km = dist_km.clip(lower=1e-6)

    attractiveness = stores_df["square_footage"].values ** alpha  # (n_stores,)
    utility = attractiveness[np.newaxis, :] / (dist_km.values ** lam)  # (n_origins, n_stores)
    row_sums = utility.sum(axis=1, keepdims=True)
    prob = utility / row_sums

    return pd.DataFrame(prob, index=origins_df.index, columns=stores_df.index)


# ---------------------------------------------------------------------------
# 4. Observed visit counts (multinomial sampling)
# ---------------------------------------------------------------------------

def generate_observed_visits(
    prob_matrix: pd.DataFrame,
    n_total: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample visit counts from the true probability matrix.

    For each origin, the number of total visits is proportional to that
    origin's population (encoded in the row index).  Visits are drawn from a
    multinomial with the ground-truth Huff probabilities.

    Returns a DataFrame with columns:
        origin_id, store_id, visit_count, visit_share
    """
    rng = np.random.default_rng(seed)
    n_origins = len(prob_matrix)
    visits_per_origin = n_total // n_origins  # uniform for simplicity

    records: list[dict] = []
    for origin_id, row in prob_matrix.iterrows():
        probs = row.values
        counts = rng.multinomial(visits_per_origin, probs)
        total = counts.sum()
        for store_id, count in zip(prob_matrix.columns, counts):
            records.append(
                {
                    "origin_id": origin_id,
                    "store_id": store_id,
                    "visit_count": int(count),
                    "visit_share": float(count / total) if total > 0 else 0.0,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Transaction data
# ---------------------------------------------------------------------------

def _store_size_tier(sq_ft: float) -> str:
    if sq_ft < 3000:
        return "small"
    elif sq_ft < 10_000:
        return "medium"
    return "large"


def generate_transactions(
    stores: list[Store],
    origins: list[ConsumerOrigin],
    prob_matrix: pd.DataFrame,
    n_transactions: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic transaction records.

    Each transaction has:
        transaction_id, consumer_id, store_id, origin_id, timestamp, amount, items

    * Consumer store choices follow the true Huff probability distribution.
    * Timestamps are uniformly distributed over 12 months.
    * Amounts are Gamma-distributed with known shape/scale per store tier.
    """
    rng = np.random.default_rng(seed)
    gt = GROUND_TRUTH
    shape_k = gt["transaction_amount_shape"]
    scales = gt["transaction_amount_scales"]
    date_start = datetime.strptime(gt["transaction_date_start"], "%Y-%m-%d")
    date_end = datetime.strptime(gt["transaction_date_end"], "%Y-%m-%d")
    span_seconds = int((date_end - date_start).total_seconds())

    # Pre-compute store tier lookup
    store_tiers = {s.store_id: _store_size_tier(s.square_footage) for s in stores}
    store_ids = [s.store_id for s in stores]
    origin_ids = [o.origin_id for o in origins]

    # Build cumulative probability matrix for fast sampling
    cum_probs = prob_matrix.values.cumsum(axis=1)

    records: list[dict] = []
    for txn_idx in range(n_transactions):
        # Pick an origin uniformly
        o_idx = int(rng.integers(0, len(origin_ids)))
        origin_id = origin_ids[o_idx]

        # Choose a store according to Huff probabilities for this origin
        u = rng.random()
        s_idx = int(np.searchsorted(cum_probs[o_idx], u))
        s_idx = min(s_idx, len(store_ids) - 1)
        store_id = store_ids[s_idx]

        # Timestamp
        offset = int(rng.integers(0, span_seconds + 1))
        ts = date_start + timedelta(seconds=offset)

        # Amount (Gamma with store-tier-specific scale)
        tier = store_tiers[store_id]
        amount = float(rng.gamma(shape_k, scales[tier]))

        # Number of items
        items = int(rng.integers(1, 12))

        consumer_id = f"C{txn_idx + 1:06d}"

        records.append(
            {
                "transaction_id": f"T{txn_idx + 1:06d}",
                "consumer_id": consumer_id,
                "store_id": store_id,
                "origin_id": origin_id,
                "timestamp": ts,
                "amount": round(amount, 2),
                "items": items,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. Discrete choice data
# ---------------------------------------------------------------------------

def generate_choice_data(
    stores: list[Store],
    origins: list[ConsumerOrigin],
    prob_matrix: pd.DataFrame,
    n_choices: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate discrete choice observations.

    Each row represents one origin evaluating one store, with a binary
    ``chosen`` indicator (0/1).  Observations are grouped into choice sets
    (one set per decision event): each set contains all stores, with exactly
    one ``chosen == 1``.

    Columns:
        choice_id  – groups a single decision event
        origin_id  – the consumer origin making the choice
        store_id   – the store being evaluated
        chosen     – 1 if this store was selected, else 0
        distance_km       – haversine distance origin->store
        square_footage    – store attractiveness proxy
        avg_rating        – store quality signal
        population        – origin demand signal
        median_income     – origin income signal
    """
    rng = np.random.default_rng(seed)
    stores_df = stores_to_dataframe(stores)
    origins_df = origins_to_dataframe(origins)
    dist_matrix = build_distance_matrix(origins_df, stores_df)

    origin_ids = list(prob_matrix.index)
    store_ids = list(prob_matrix.columns)
    n_stores = len(store_ids)
    cum_probs = prob_matrix.values.cumsum(axis=1)

    records: list[dict] = []
    for choice_idx in range(n_choices):
        # Pick an origin
        o_idx = int(rng.integers(0, len(origin_ids)))
        origin_id = origin_ids[o_idx]

        # Choose a store using the true Huff probabilities
        u = rng.random()
        chosen_idx = int(np.searchsorted(cum_probs[o_idx], u))
        chosen_idx = min(chosen_idx, n_stores - 1)

        for s_idx, sid in enumerate(store_ids):
            records.append(
                {
                    "choice_id": choice_idx,
                    "origin_id": origin_id,
                    "store_id": sid,
                    "chosen": 1 if s_idx == chosen_idx else 0,
                    "distance_km": round(float(dist_matrix.loc[origin_id, sid]), 4),
                    "square_footage": float(stores_df.loc[sid, "square_footage"]),
                    "avg_rating": float(stores_df.loc[sid, "avg_rating"]),
                    "population": int(origins_df.loc[origin_id, "population"]),
                    "median_income": float(origins_df.loc[origin_id, "median_income"]),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Convenience: generate everything at once
# ---------------------------------------------------------------------------

def generate_all(
    seed: int = 42,
) -> dict:
    """
    Generate every synthetic dataset and return them in a single dict.

    Keys:
        stores, origins, prob_matrix, observed_visits,
        transactions, choice_data, ground_truth
    """
    stores = generate_stores(seed=seed)
    origins = generate_origins(seed=seed)
    prob_matrix = generate_huff_ground_truth(
        stores, origins,
        alpha=GROUND_TRUTH["huff_alpha"],
        lam=GROUND_TRUTH["huff_lambda"],
    )
    observed_visits = generate_observed_visits(prob_matrix, seed=seed)
    transactions = generate_transactions(stores, origins, prob_matrix, seed=seed)
    choice_data = generate_choice_data(stores, origins, prob_matrix, seed=seed)

    return {
        "stores": stores,
        "origins": origins,
        "prob_matrix": prob_matrix,
        "observed_visits": observed_visits,
        "transactions": transactions,
        "choice_data": choice_data,
        "ground_truth": GROUND_TRUTH,
    }
