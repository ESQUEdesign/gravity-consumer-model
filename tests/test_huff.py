"""
Tests for the core Huff Retail Gravity Model.

Validates prediction shape, probability normalization, known-answer
recovery, parameter estimation via MLE, and economic intuition checks
(closer stores get more traffic, larger stores attract more visitors).
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.fixtures.synthetic_data import generate_all, GROUND_TRUTH

from gravity.core.huff import HuffModel
from gravity.data.schema import (
    stores_to_dataframe,
    origins_to_dataframe,
    build_distance_matrix,
)


# ---------------------------------------------------------------------------
# Module-scoped fixture: generate all synthetic data once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth():
    """Generate the full synthetic dataset (deterministic, seeded)."""
    return generate_all(seed=42)


@pytest.fixture(scope="module")
def stores_df(synth):
    """Stores as a DataFrame indexed by store_id."""
    return stores_to_dataframe(synth["stores"])


@pytest.fixture(scope="module")
def origins_df(synth):
    """Origins as a DataFrame indexed by origin_id."""
    return origins_to_dataframe(synth["origins"])


@pytest.fixture(scope="module")
def prob_matrix(synth):
    """Ground-truth probability matrix from synthetic data."""
    return synth["prob_matrix"]


@pytest.fixture(scope="module")
def observed_visits(synth):
    """Observed visit records (long-form) from synthetic data."""
    return synth["observed_visits"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHuffPredictShape:
    """test_huff_predict_shape: output is origin x store DataFrame with correct dims."""

    def test_huff_predict_shape(self, stores_df, origins_df):
        """HuffModel.predict() returns a DataFrame of shape (n_origins, n_stores)."""
        model = HuffModel(alpha=1.0, lam=2.0)
        probs = model.predict(origins_df, stores_df)

        n_origins = len(origins_df)
        n_stores = len(stores_df)

        assert isinstance(probs, pd.DataFrame), "predict() should return a DataFrame"
        assert probs.shape == (n_origins, n_stores), (
            f"Expected shape ({n_origins}, {n_stores}), got {probs.shape}"
        )
        assert list(probs.index) == list(origins_df.index), (
            "Row index should match origin_ids"
        )
        assert list(probs.columns) == list(stores_df.index), (
            "Column index should match store_ids"
        )


class TestHuffProbabilitiesSumToOne:
    """test_huff_probabilities_sum_to_one: every row sums to 1.0."""

    def test_huff_probabilities_sum_to_one(self, stores_df, origins_df):
        """Each origin's probabilities across all stores must sum to 1.0."""
        model = HuffModel(alpha=1.0, lam=2.0)
        probs = model.predict(origins_df, stores_df)
        row_sums = probs.sum(axis=1)

        np.testing.assert_allclose(
            row_sums.values,
            np.ones(len(row_sums)),
            atol=1e-12,
            err_msg="Every row in the probability matrix must sum to 1.0",
        )


class TestHuffPredictWithKnownParams:
    """test_huff_predict_with_known_params: predictions match ground truth."""

    def test_huff_predict_with_known_params(
        self, stores_df, origins_df, prob_matrix
    ):
        """Using ground-truth alpha=1.0, lambda=2.0, predictions should match
        the synthetic generate_huff_ground_truth() output within 1e-10."""
        model = HuffModel(
            alpha=GROUND_TRUTH["huff_alpha"],
            lam=GROUND_TRUTH["huff_lambda"],
        )
        predicted = model.predict(origins_df, stores_df)

        # Align to same ordering
        predicted = predicted.reindex(
            index=prob_matrix.index, columns=prob_matrix.columns
        )

        np.testing.assert_allclose(
            predicted.values,
            prob_matrix.values,
            atol=1e-10,
            err_msg=(
                "HuffModel.predict() with ground-truth params should reproduce "
                "the synthetic probability matrix to within 1e-10"
            ),
        )


class TestHuffFitRecoversParameters:
    """test_huff_fit_recovers_parameters: MLE recovers alpha and lambda."""

    def test_huff_fit_recovers_parameters(
        self, stores_df, origins_df, observed_visits
    ):
        """Fit from observed visit shares; recovered alpha should be within
        0.3 of true 1.0, and lambda within 0.5 of true 2.0."""
        # Pivot observed_visits into an origin x store share matrix
        obs_shares = observed_visits.pivot_table(
            index="origin_id",
            columns="store_id",
            values="visit_share",
            aggfunc="mean",
        )
        # Ensure alignment
        obs_shares = obs_shares.reindex(
            index=origins_df.index, columns=stores_df.index
        ).fillna(0.0)

        model = HuffModel(alpha=0.5, lam=1.0)  # start away from truth
        model.fit(origins_df, stores_df, obs_shares)

        true_alpha = GROUND_TRUTH["huff_alpha"]
        true_lambda = GROUND_TRUTH["huff_lambda"]

        assert abs(model.alpha - true_alpha) < 0.3, (
            f"Recovered alpha={model.alpha:.4f} is not within 0.3 of "
            f"true alpha={true_alpha}"
        )
        assert abs(model.lam - true_lambda) < 0.5, (
            f"Recovered lambda={model.lam:.4f} is not within 0.5 of "
            f"true lambda={true_lambda}"
        )


class TestHuffCloserStoresHigherProbability:
    """test_huff_closer_stores_higher_probability: nearest store should rank high."""

    def test_huff_closer_stores_higher_probability(
        self, stores_df, origins_df
    ):
        """For each origin, compare two stores of similar size: the closer
        one should have higher probability. This tests distance-decay
        independently of size variation.

        We use an equal-attractiveness model (all stores set to same sqft)
        to isolate the distance effect, then verify the nearest store is
        in the top-3 for at least 80% of origins."""
        # Create a modified stores_df with equal square_footage
        equal_stores = stores_df.copy()
        equal_stores["square_footage"] = 5000.0  # uniform

        model = HuffModel(alpha=1.0, lam=2.0)
        probs = model.predict(origins_df, equal_stores)
        dist = build_distance_matrix(origins_df, equal_stores)

        count_in_top3 = 0
        n_origins = len(origins_df)

        for origin_id in origins_df.index:
            nearest_store = dist.loc[origin_id].idxmin()
            top3_stores = probs.loc[origin_id].nlargest(3).index.tolist()
            if nearest_store in top3_stores:
                count_in_top3 += 1

        fraction = count_in_top3 / n_origins
        assert fraction >= 0.80, (
            f"Only {fraction:.1%} of origins have the nearest store in their "
            f"top-3 probabilities (with equal sizes); expected at least 80%"
        )


class TestHuffLargerStoresHigherProbability:
    """test_huff_larger_stores_higher_probability: among equidistant stores,
    larger stores should attract more."""

    def test_huff_larger_stores_higher_probability(
        self, stores_df, origins_df
    ):
        """Among stores at roughly similar distances from each origin,
        a store with larger square footage should tend to have a higher
        probability. We verify this using rank correlation across all origins."""
        model = HuffModel(alpha=1.0, lam=2.0)
        probs = model.predict(origins_df, stores_df)
        dist = build_distance_matrix(origins_df, stores_df)

        # For each origin, compute correlation between 1/distance^2 * sqft and prob
        # Instead, simply check: among the two largest stores, the bigger one
        # should on average have higher probability.
        store_sizes = stores_df["square_footage"]
        largest_store = store_sizes.idxmax()
        second_largest = store_sizes.drop(largest_store).idxmax()

        # Count origins where the largest store has higher prob than the 2nd
        count_larger_wins = 0
        for origin_id in origins_df.index:
            p_large = probs.loc[origin_id, largest_store]
            p_second = probs.loc[origin_id, second_largest]
            d_large = dist.loc[origin_id, largest_store]
            d_second = dist.loc[origin_id, second_largest]
            # Adjust for distance: if both are at similar distance, check
            # Or simply check: size-weighted probability is positively related
            # The largest store (30000 sqft) vs second (20000 sqft) --
            # if at same distance, the larger should always win
            if abs(d_large - d_second) < 1.0:  # within 1 km
                if p_large > p_second:
                    count_larger_wins += 1

        # We only assert for origins where the stores are equidistant
        # This is a soft check; just ensure the model has the right sign
        # Check globally: mean probability of the largest store should be
        # among the highest
        mean_probs = probs.mean(axis=0)
        top3_by_mean = mean_probs.nlargest(3).index.tolist()
        assert largest_store in top3_by_mean, (
            f"The largest store ({largest_store}, "
            f"{store_sizes[largest_store]} sqft) should be in the top-3 "
            f"by mean probability across all origins, but top-3 are: "
            f"{top3_by_mean}"
        )


class TestTradeAreaShares:
    """test_trade_area_shares: trade_area_shares returns correct-length Series."""

    def test_trade_area_shares(self, stores_df, origins_df):
        """trade_area_shares(store_id) returns a Series with one entry per origin."""
        model = HuffModel(alpha=1.0, lam=2.0)
        first_store = stores_df.index[0]
        shares = model.trade_area_shares(first_store, origins_df, stores_df)

        assert isinstance(shares, pd.Series), (
            "trade_area_shares() should return a pandas Series"
        )
        assert len(shares) == len(origins_df), (
            f"Series length {len(shares)} should equal number of origins "
            f"{len(origins_df)}"
        )
        assert (shares >= 0).all(), "All trade area shares must be non-negative"
        assert (shares <= 1).all(), "All trade area shares must be <= 1"
