"""
Tests for consumer segmentation modules: RFM, CLV, and Latent Class.

Validates scoring ranges, segment assignment, CLV model fitting and
prediction, and latent class discovery from synthetic visit data.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.fixtures.synthetic_data import generate_all, GROUND_TRUTH

from gravity.segmentation.rfm import RFMScorer
from gravity.segmentation.clv import CLVEstimator
from gravity.segmentation.latent_class import LatentClassModel
from gravity.data.schema import stores_to_dataframe, origins_to_dataframe


# ---------------------------------------------------------------------------
# Module-scoped fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth():
    """Generate the full synthetic dataset (deterministic, seeded)."""
    return generate_all(seed=42)


@pytest.fixture(scope="module")
def transactions(synth):
    """Transaction DataFrame from synthetic data."""
    return synth["transactions"]


@pytest.fixture(scope="module")
def stores_df(synth):
    return stores_to_dataframe(synth["stores"])


@pytest.fixture(scope="module")
def origins_df(synth):
    return origins_to_dataframe(synth["origins"])


@pytest.fixture(scope="module")
def observed_visits(synth):
    return synth["observed_visits"]


# ---------------------------------------------------------------------------
# RFM Tests
# ---------------------------------------------------------------------------


class TestRFMScoring:
    """test_rfm_scoring: RFMScorer.fit() produces valid R, F, M scores."""

    def test_rfm_scoring(self, transactions):
        """After fitting, R, F, M scores should all be integers in [1, 5]."""
        scorer = RFMScorer(n_bins=5)
        scorer.fit(transactions)

        rfm = scorer.rfm_table_
        assert rfm is not None, "rfm_table_ should be populated after fit()"

        for col in ["r_score", "f_score", "m_score"]:
            assert col in rfm.columns, f"Missing column: {col}"
            values = rfm[col].values
            assert np.all(values >= 1), (
                f"{col} contains values < 1: min={values.min()}"
            )
            assert np.all(values <= 5), (
                f"{col} contains values > 5: max={values.max()}"
            )


class TestRFMSegmentLabels:
    """test_rfm_segment_labels: every consumer gets a valid segment label."""

    def test_rfm_segment_labels(self, transactions):
        """All consumers should have a non-empty string segment label."""
        scorer = RFMScorer(n_bins=5)
        scorer.fit(transactions)

        rfm = scorer.rfm_table_
        assert "rfm_segment" in rfm.columns, "Missing 'rfm_segment' column"

        # No nulls
        assert not rfm["rfm_segment"].isna().any(), (
            "Some consumers have null segment labels"
        )

        # All labels are non-empty strings
        for label in rfm["rfm_segment"].unique():
            assert isinstance(label, str) and len(label) > 0, (
                f"Invalid segment label: {label!r}"
            )


class TestRFMChampionsHighScores:
    """test_rfm_champions_have_high_scores: Champions should have high F and M."""

    def test_rfm_champions_have_high_scores(self, transactions):
        """Consumers labelled 'Champions' should have F >= 4 and M >= 4."""
        scorer = RFMScorer(n_bins=5)
        scorer.fit(transactions)

        rfm = scorer.rfm_table_
        champions = rfm[rfm["rfm_segment"] == "Champions"]

        if len(champions) == 0:
            pytest.skip("No Champions segment found in this synthetic dataset")

        assert (champions["f_score"] >= 4).all(), (
            "Champions should all have f_score >= 4, "
            f"but min is {champions['f_score'].min()}"
        )
        assert (champions["m_score"] >= 4).all(), (
            "Champions should all have m_score >= 4, "
            f"but min is {champions['m_score'].min()}"
        )


# ---------------------------------------------------------------------------
# CLV Tests
# ---------------------------------------------------------------------------


class TestCLVFit:
    """test_clv_fit: CLVEstimator.fit() completes and produces bgnbd_params_."""

    def test_clv_fit(self, transactions):
        """fit() should complete without error and populate bgnbd_params_."""
        estimator = CLVEstimator(penalizer_coef=0.01)
        estimator.fit(transactions)

        assert estimator.bgnbd_params_ is not None, (
            "bgnbd_params_ should not be None after fit()"
        )
        required_keys = {"r", "alpha", "a", "b"}
        assert required_keys <= set(estimator.bgnbd_params_.keys()), (
            f"bgnbd_params_ is missing keys: "
            f"{required_keys - set(estimator.bgnbd_params_.keys())}"
        )
        # All params should be positive
        for key, val in estimator.bgnbd_params_.items():
            assert val > 0, f"BG/NBD param '{key}' should be positive, got {val}"


class TestCLVPredict:
    """test_clv_predict: predict_clv() returns expected_purchases > 0 and p_alive in [0,1]."""

    def test_clv_predict(self, transactions):
        """predict_clv() should produce reasonable expected_purchases and p_alive."""
        estimator = CLVEstimator(penalizer_coef=0.01)
        estimator.fit(transactions)
        clv_df = estimator.predict_clv(time_horizon_days=365)

        assert "expected_purchases" in clv_df.columns, (
            "Missing 'expected_purchases' column"
        )
        assert "p_alive" in clv_df.columns, "Missing 'p_alive' column"

        # expected_purchases should be non-negative
        assert (clv_df["expected_purchases"] >= 0).all(), (
            "expected_purchases should be >= 0"
        )

        # p_alive should be in [0, 1]
        assert (clv_df["p_alive"] >= 0).all(), "p_alive should be >= 0"
        assert (clv_df["p_alive"] <= 1).all(), "p_alive should be <= 1"

        # At least some consumers should have expected_purchases > 0
        assert (clv_df["expected_purchases"] > 0).any(), (
            "At least some consumers should have expected_purchases > 0"
        )


class TestCLVTiers:
    """test_clv_tiers: segment_by_clv(n_tiers=5) assigns all consumers to a tier."""

    def test_clv_tiers(self, transactions):
        """Every consumer should get a CLV tier assignment."""
        estimator = CLVEstimator(penalizer_coef=0.01)
        estimator.fit(transactions)
        tiers_df = estimator.segment_by_clv(n_tiers=5)

        assert "clv_tier" in tiers_df.columns, "Missing 'clv_tier' column"
        assert not tiers_df["clv_tier"].isna().any(), (
            "Some consumers have null CLV tier"
        )

        unique_tiers = tiers_df["clv_tier"].nunique()
        assert unique_tiers >= 2, (
            f"Expected at least 2 distinct tiers, got {unique_tiers}"
        )


# ---------------------------------------------------------------------------
# Latent Class Tests
# ---------------------------------------------------------------------------


class TestLatentClassDiscoversSegments:
    """test_latent_class_discovers_segments: LatentClassModel with n_classes=3."""

    def test_latent_class_discovers_segments(
        self, observed_visits, origins_df, stores_df
    ):
        """LatentClassModel with n_classes=3 should discover exactly 3 segments."""
        # Prepare visit data in the format expected by the model
        visit_data = observed_visits.rename(
            columns={"visit_count": "visits"}
        )[["origin_id", "store_id", "visits"]]

        model = LatentClassModel(
            n_classes=3,
            n_restarts=2,
            max_iter=50,
            random_state=42,
        )
        model.fit(visit_data, origins_df, stores_df)

        assert model.n_classes_ == 3, (
            f"Expected 3 classes, got {model.n_classes_}"
        )
        assert model.class_params_ is not None, (
            "class_params_ should not be None after fit()"
        )
        assert len(model.class_params_) == 3, (
            f"Expected 3 sets of class params, got {len(model.class_params_)}"
        )


class TestLatentClassMembershipSumsToOne:
    """test_latent_class_membership_sums_to_one: posteriors sum to ~1.0."""

    def test_latent_class_membership_sums_to_one(
        self, observed_visits, origins_df, stores_df
    ):
        """Posterior membership probabilities per consumer should sum to ~1.0."""
        visit_data = observed_visits.rename(
            columns={"visit_count": "visits"}
        )[["origin_id", "store_id", "visits"]]

        model = LatentClassModel(
            n_classes=3,
            n_restarts=2,
            max_iter=50,
            random_state=42,
        )
        model.fit(visit_data, origins_df, stores_df)

        posterior = model.posterior_
        assert posterior is not None, "posterior_ should be populated after fit()"

        row_sums = posterior.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(len(row_sums)),
            atol=1e-6,
            err_msg="Posterior membership probabilities per consumer should sum to 1.0",
        )
