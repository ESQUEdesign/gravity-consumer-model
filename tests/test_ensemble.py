"""
Tests for ensemble model averaging and end-to-end pipeline.

Validates simple averaging, weight fitting, prediction intervals,
GravityModel integration, and scenario simulation.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.fixtures.synthetic_data import generate_all, GROUND_TRUTH

from gravity.ensemble.model_averaging import EnsembleAverager
from gravity.core.huff import HuffModel
from gravity.model import GravityModel
from gravity.reporting.scenario import ScenarioSimulator
from gravity.data.schema import stores_to_dataframe, origins_to_dataframe


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth():
    """Generate the full synthetic dataset."""
    return generate_all(seed=42)


@pytest.fixture(scope="module")
def stores_df(synth):
    return stores_to_dataframe(synth["stores"])


@pytest.fixture(scope="module")
def origins_df(synth):
    return origins_to_dataframe(synth["origins"])


@pytest.fixture(scope="module")
def prob_matrix(synth):
    return synth["prob_matrix"]


@pytest.fixture(scope="module")
def observed_visits(synth):
    return synth["observed_visits"]


@pytest.fixture(scope="module")
def two_model_predictions(prob_matrix):
    """Create two slightly different prediction matrices for ensemble testing."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, size=prob_matrix.shape)

    # Model A: ground truth + small noise
    model_a = prob_matrix + noise
    model_a = model_a.clip(lower=1e-10)
    model_a = model_a.div(model_a.sum(axis=1), axis=0)

    # Model B: ground truth + different noise
    noise_b = rng.normal(0, 0.02, size=prob_matrix.shape)
    model_b = prob_matrix + noise_b
    model_b = model_b.clip(lower=1e-10)
    model_b = model_b.div(model_b.sum(axis=1), axis=0)

    return model_a, model_b


# ---------------------------------------------------------------------------
# Ensemble Tests
# ---------------------------------------------------------------------------


class TestEnsembleSimpleAverage:
    """test_ensemble_simple_average: with 2 models and simple_average, result is the mean."""

    def test_ensemble_simple_average(self, prob_matrix, two_model_predictions):
        """With simple_average weighting, the blended prediction should be
        the arithmetic mean of the two input models."""
        model_a, model_b = two_model_predictions

        ens = EnsembleAverager(normalize=False)
        ens.add_model("model_a", model_a)
        ens.add_model("model_b", model_b)
        ens.fit_weights(prob_matrix, method="simple_average")

        blended = ens.predict()
        expected = (model_a + model_b) / 2.0

        np.testing.assert_allclose(
            blended.values,
            expected.values,
            atol=1e-12,
            err_msg="Simple average should produce the exact mean of inputs",
        )


class TestEnsembleWeightsSumToOne:
    """test_ensemble_weights_sum_to_one: fitted weights sum to 1.0."""

    def test_ensemble_weights_sum_to_one(self, prob_matrix, two_model_predictions):
        """After fit_weights, model weights should sum to 1.0."""
        model_a, model_b = two_model_predictions

        ens = EnsembleAverager()
        ens.add_model("model_a", model_a)
        ens.add_model("model_b", model_b)
        ens.fit_weights(prob_matrix, method="simple_average")

        weights = ens.weights
        total = sum(weights.values())

        np.testing.assert_allclose(
            total,
            1.0,
            atol=1e-10,
            err_msg=f"Ensemble weights should sum to 1.0, got {total}",
        )


class TestEnsemblePredictionIntervals:
    """test_ensemble_prediction_intervals: lower < mean < upper."""

    def test_ensemble_prediction_intervals(
        self, prob_matrix, two_model_predictions
    ):
        """Prediction intervals should satisfy lower <= mean <= upper."""
        model_a, model_b = two_model_predictions

        ens = EnsembleAverager()
        ens.add_model("model_a", model_a)
        ens.add_model("model_b", model_b)
        ens.fit_weights(prob_matrix, method="simple_average")

        mean = ens.predict()
        lower, upper = ens.prediction_intervals(confidence=0.95)

        # lower <= mean
        assert (lower.values <= mean.values + 1e-12).all(), (
            "Lower bound should not exceed the mean prediction"
        )
        # mean <= upper
        assert (mean.values <= upper.values + 1e-12).all(), (
            "Mean prediction should not exceed the upper bound"
        )
        # lower >= 0
        assert (lower.values >= -1e-12).all(), (
            "Lower bound should be non-negative"
        )
        # upper <= 1
        assert (upper.values <= 1.0 + 1e-12).all(), (
            "Upper bound should not exceed 1.0"
        )


# ---------------------------------------------------------------------------
# GravityModel end-to-end Tests
# ---------------------------------------------------------------------------


class TestGravityModelEndToEnd:
    """test_gravity_model_end_to_end: GravityModel with huff layer fits and predicts."""

    def test_gravity_model_end_to_end(self, stores_df, origins_df, observed_visits):
        """GravityModel(layers=['huff']) should fit and predict without error.

        GravityModel.predict() returns a long-form DataFrame with columns
        including origin_id, store_id, probability (one row per origin-store pair).
        """
        # Build observed shares matrix
        obs_shares = observed_visits.pivot_table(
            index="origin_id",
            columns="store_id",
            values="visit_share",
            aggfunc="mean",
        ).reindex(index=origins_df.index, columns=stores_df.index).fillna(0.0)

        model = GravityModel(layers=["huff"])
        model.fit(
            stores=stores_df,
            origins=origins_df,
            observed_shares=obs_shares,
        )

        predictions = model.predict()

        assert isinstance(predictions, pd.DataFrame), (
            "predict() should return a DataFrame"
        )

        n_origins = len(origins_df)
        n_stores = len(stores_df)
        expected_rows = n_origins * n_stores

        assert predictions.shape[0] == expected_rows, (
            f"Expected {expected_rows} rows (origins x stores), "
            f"got {predictions.shape[0]}"
        )

        # Should contain probability column
        assert "probability" in predictions.columns, (
            "predictions should contain a 'probability' column"
        )

        # Probabilities should be in [0, 1]
        probs = predictions["probability"]
        assert (probs >= 0).all(), "All probabilities should be >= 0"
        assert (probs <= 1).all(), "All probabilities should be <= 1"

        # Per-origin probabilities should sum to approximately 1.0
        if "origin_id" in predictions.columns:
            origin_sums = predictions.groupby("origin_id")["probability"].sum()
            np.testing.assert_allclose(
                origin_sums.values,
                np.ones(len(origin_sums)),
                atol=0.01,
                err_msg="Per-origin probabilities should approximately sum to 1.0",
            )


class TestGravityModelWithSegmentation:
    """test_gravity_model_with_segmentation: GravityModel with huff + latent_class."""

    def test_gravity_model_with_segmentation(
        self, stores_df, origins_df, observed_visits
    ):
        """GravityModel with ['huff', 'latent_class'] should fit without error."""
        obs_shares = observed_visits.pivot_table(
            index="origin_id",
            columns="store_id",
            values="visit_share",
            aggfunc="mean",
        ).reindex(index=origins_df.index, columns=stores_df.index).fillna(0.0)

        model = GravityModel(layers=["huff", "latent_class"])
        model.fit(
            stores=stores_df,
            origins=origins_df,
            observed_shares=obs_shares,
        )

        predictions = model.predict()

        assert isinstance(predictions, pd.DataFrame), (
            "predict() should return a DataFrame"
        )
        assert predictions.shape[0] > 0, "Predictions should have rows"
        assert predictions.shape[1] > 0, "Predictions should have columns"
        assert "probability" in predictions.columns, (
            "predictions should contain a 'probability' column"
        )


# ---------------------------------------------------------------------------
# Scenario Simulator Tests
# ---------------------------------------------------------------------------


class TestScenarioNewStore:
    """test_scenario_new_store: ScenarioSimulator.new_store() returns redistribution."""

    def test_scenario_new_store(self, stores_df, origins_df):
        """Opening a new store should redistribute market share; existing
        stores should generally lose share."""
        huff = HuffModel(alpha=1.0, lam=2.0)

        sim = ScenarioSimulator(
            predict_fn=huff.predict,
            origins_df=origins_df,
            stores_df=stores_df,
        )

        result = sim.new_store(
            location=(40.75, -73.98),
            attractiveness=20000.0,
            store_id="new_store",
        )

        # The new store should have positive market share
        assert result.scenario_shares["new_store"] > 0, (
            "New store should capture some market share"
        )

        # Existing stores should generally lose share (some share_change < 0)
        existing_stores = stores_df.index.tolist()
        existing_change = result.share_change.reindex(existing_stores)
        assert (existing_change < 0).any(), (
            "At least some existing stores should lose share after a new store opens"
        )

        # Total scenario shares should sum to ~1.0
        total = result.scenario_shares.sum()
        np.testing.assert_allclose(
            total, 1.0, atol=0.01,
            err_msg=f"Total scenario shares should sum to ~1.0, got {total}",
        )


class TestScenarioCloseStore:
    """test_scenario_close_store: closing a store redistributes demand."""

    def test_scenario_close_store(self, stores_df, origins_df):
        """Closing a store should redistribute its demand to remaining stores."""
        huff = HuffModel(alpha=1.0, lam=2.0)

        sim = ScenarioSimulator(
            predict_fn=huff.predict,
            origins_df=origins_df,
            stores_df=stores_df,
        )

        store_to_close = stores_df.index[0]
        result = sim.close_store(store_to_close)

        # Closed store should not appear in scenario predictions
        # (or have zero share)
        remaining_stores = stores_df.index.drop(store_to_close)

        # Remaining stores should see their shares increase on average
        remaining_changes = result.share_change.reindex(remaining_stores)
        assert (remaining_changes > 0).any(), (
            "At least some remaining stores should gain share when a competitor closes"
        )

        # Total scenario shares (excluding closed store) should sum to ~1.0
        remaining_shares = result.scenario_shares.reindex(remaining_stores)
        total = remaining_shares.sum()
        np.testing.assert_allclose(
            total, 1.0, atol=0.01,
            err_msg=(
                f"Remaining stores' shares should sum to ~1.0, got {total}"
            ),
        )
