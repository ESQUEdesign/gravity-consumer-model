"""
Tests for temporal dynamics modules: HMM, Hawkes Process, Bayesian Updater.

Validates model fitting, state decoding, transition matrix stochasticity,
Hawkes process stability, simulation, and Bayesian posterior normalization
and learning behavior.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.fixtures.synthetic_data import generate_all, GROUND_TRUTH

from gravity.temporal.hmm import ConsumerHMM
from gravity.temporal.hawkes import HawkesProcess
from gravity.temporal.bayesian_update import BayesianUpdater


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth():
    """Generate the full synthetic dataset."""
    return generate_all(seed=42)


@pytest.fixture(scope="module")
def synthetic_hmm_sequences():
    """Generate synthetic observation sequences for HMM testing.

    Simulates 50 consumers, each with a 20-period visit-frequency history.
    Observation symbols are drawn from a simple Markov-like process so the
    HMM has structure to discover.
    """
    rng = np.random.default_rng(42)
    obs_labels = ["high", "medium", "low", "none"]

    sequences = []
    for _ in range(50):
        # Start in a random state-proxy
        state = rng.integers(0, 3)
        seq = []
        for _ in range(20):
            # Biased emission: state 0 -> high/medium, 1 -> medium/low, 2 -> low/none
            if state == 0:
                obs_idx = rng.choice([0, 1], p=[0.7, 0.3])
            elif state == 1:
                obs_idx = rng.choice([1, 2], p=[0.5, 0.5])
            else:
                obs_idx = rng.choice([2, 3], p=[0.4, 0.6])
            seq.append(obs_labels[obs_idx])

            # State transition
            if rng.random() < 0.15:
                state = rng.integers(0, 3)
        sequences.append(seq)

    return sequences


@pytest.fixture(scope="module")
def synthetic_hawkes_events():
    """Generate synthetic Hawkes event times for testing.

    Uses a simple self-exciting process with known parameters:
    mu=0.5, alpha=0.3, beta=1.0 (branching ratio = 0.3 < 1).
    """
    rng = np.random.default_rng(42)
    mu, alpha, beta = 0.5, 0.3, 1.0
    T_max = 100.0

    events = []
    t = 0.0
    while t < T_max:
        # Current intensity
        lam = mu
        for ti in events:
            lam += alpha * np.exp(-beta * (t - ti))
        # Upper bound for thinning
        lam_bar = lam + alpha
        dt = rng.exponential(1.0 / lam_bar)
        t += dt
        if t >= T_max:
            break
        # Accept/reject
        lam_t = mu
        for ti in events:
            lam_t += alpha * np.exp(-beta * (t - ti))
        if rng.random() <= lam_t / lam_bar:
            events.append(t)

    return np.array(events)


# ---------------------------------------------------------------------------
# HMM Tests
# ---------------------------------------------------------------------------


class TestHMMFit:
    """test_hmm_fit: ConsumerHMM fit() completes without error."""

    def test_hmm_fit(self, synthetic_hmm_sequences):
        """fit() should complete and set _fitted to True."""
        hmm = ConsumerHMM(random_state=42)
        hmm.fit(synthetic_hmm_sequences, max_iter=30, n_init=2)

        assert hmm._fitted, "HMM should be marked as fitted after fit()"
        assert hmm.A is not None, "Transition matrix A should not be None"
        assert hmm.B is not None, "Emission matrix B should not be None"
        assert hmm.pi is not None, "Initial state distribution pi should not be None"


class TestHMMDecodeReturnsValidStates:
    """test_hmm_decode_returns_valid_states: decode() returns states from the configured list."""

    def test_hmm_decode_returns_valid_states(self, synthetic_hmm_sequences):
        """decode() should return only state labels defined in the model."""
        hmm = ConsumerHMM(random_state=42)
        hmm.fit(synthetic_hmm_sequences, max_iter=30, n_init=2)

        test_seq = synthetic_hmm_sequences[0]
        decoded = hmm.decode(test_seq)

        assert len(decoded) == len(test_seq), (
            f"Decoded path length {len(decoded)} should match input length {len(test_seq)}"
        )

        valid_states = set(hmm.states)
        for state in decoded:
            assert state in valid_states, (
                f"Decoded state '{state}' is not in the valid state set: {valid_states}"
            )


class TestHMMTransitionMatrixStochastic:
    """test_hmm_transition_matrix_is_stochastic: each row sums to 1.0."""

    def test_hmm_transition_matrix_is_stochastic(self, synthetic_hmm_sequences):
        """Every row of the transition matrix should sum to 1.0."""
        hmm = ConsumerHMM(random_state=42)
        hmm.fit(synthetic_hmm_sequences, max_iter=30, n_init=2)

        A = hmm.transition_matrix()
        row_sums = A.sum(axis=1)

        np.testing.assert_allclose(
            row_sums,
            np.ones(hmm.n_states),
            atol=1e-10,
            err_msg="Each row of the transition matrix should sum to 1.0",
        )

        # All entries should be non-negative
        assert (A >= 0).all(), "Transition matrix entries must be non-negative"


class TestHMMForecastStates:
    """test_hmm_forecast_states: forecast_states(5) returns 6 rows, each summing to 1.0."""

    def test_hmm_forecast_states(self, synthetic_hmm_sequences):
        """forecast_states(5) should return a (6, n_states) array where each
        row is a valid probability distribution."""
        hmm = ConsumerHMM(random_state=42)
        hmm.fit(synthetic_hmm_sequences, max_iter=30, n_init=2)

        trajectory = hmm.forecast_states(5)

        assert trajectory.shape == (6, hmm.n_states), (
            f"Expected shape (6, {hmm.n_states}), got {trajectory.shape}"
        )

        row_sums = trajectory.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(6),
            atol=1e-10,
            err_msg="Each row of the forecast trajectory should sum to 1.0",
        )


# ---------------------------------------------------------------------------
# Hawkes Process Tests
# ---------------------------------------------------------------------------


class TestHawkesFit:
    """test_hawkes_fit: HawkesProcess fit() recovers parameters in reasonable range."""

    def test_hawkes_fit(self, synthetic_hawkes_events):
        """Fitted parameters should be in a reasonable positive range."""
        hp = HawkesProcess(mu=0.1, alpha=0.1, beta=0.5)
        hp.fit(synthetic_hawkes_events, T=100.0)

        assert hp.fitted_, "HawkesProcess should be marked fitted after fit()"
        assert hp.mu > 0, f"mu should be positive, got {hp.mu}"
        assert hp.alpha > 0, f"alpha should be positive, got {hp.alpha}"
        assert hp.beta > 0, f"beta should be positive, got {hp.beta}"

        # Parameters should be in a reasonable order of magnitude
        assert hp.mu < 10.0, f"mu={hp.mu} seems unreasonably large"
        assert hp.alpha < 10.0, f"alpha={hp.alpha} seems unreasonably large"
        assert hp.beta < 50.0, f"beta={hp.beta} seems unreasonably large"


class TestHawkesStability:
    """test_hawkes_stability: fitted branching_ratio < 1.0."""

    def test_hawkes_stability(self, synthetic_hawkes_events):
        """The branching ratio alpha/beta should be < 1 for stationarity."""
        hp = HawkesProcess()
        hp.fit(synthetic_hawkes_events, T=100.0)

        br = hp.branching_ratio
        assert br < 1.0, (
            f"Branching ratio should be < 1.0 for stability, got {br:.4f}"
        )


class TestHawkesSimulate:
    """test_hawkes_simulate: simulate() generates events within the time window."""

    def test_hawkes_simulate(self):
        """simulate() should produce events within [0, duration]."""
        hp = HawkesProcess(mu=0.5, alpha=0.3, beta=1.0)
        duration = 50.0
        events = hp.simulate(duration=duration, seed=42)

        assert len(events) > 0, "simulate() should generate at least some events"
        assert events[0] >= 0, "First event time should be >= 0"
        assert events[-1] <= duration, (
            f"Last event time {events[-1]:.2f} should be <= duration {duration}"
        )
        # Events should be sorted
        assert np.all(np.diff(events) >= 0), "Event times should be sorted"


# ---------------------------------------------------------------------------
# Bayesian Update Tests
# ---------------------------------------------------------------------------


class TestBayesianUpdatePreservesNormalization:
    """test_bayesian_update_preserves_normalization: posteriors sum to 1.0."""

    def test_bayesian_update_preserves_normalization(self, synth):
        """After multiple updates, posterior probabilities should sum to 1.0."""
        store_ids = [s.store_id for s in synth["stores"]]
        updater = BayesianUpdater(store_ids=store_ids, prior_weight=10.0)

        rng = np.random.default_rng(42)
        consumer_id = "test_consumer_001"

        # Perform 20 random updates
        for _ in range(20):
            store = rng.choice(store_ids)
            updater.update(consumer_id, store)

        posterior = updater.get_posterior_array(consumer_id)

        np.testing.assert_allclose(
            posterior.sum(),
            1.0,
            atol=1e-12,
            err_msg="Posterior probabilities should sum to 1.0 after updates",
        )
        assert (posterior >= 0).all(), "All posterior probabilities should be non-negative"


class TestBayesianUpdateShiftsTowardObserved:
    """test_bayesian_update_shifts_toward_observed: repeated visits increase P(store)."""

    def test_bayesian_update_shifts_toward_observed(self, synth):
        """After 10 visits to store A, P(store A) should increase
        relative to the uniform prior."""
        store_ids = [s.store_id for s in synth["stores"]]
        updater = BayesianUpdater(store_ids=store_ids, prior_weight=10.0)

        consumer_id = "test_consumer_002"
        target_store = store_ids[0]

        # Get prior P(target) before any updates
        updater.register_consumer(consumer_id)
        prior_prob = updater.get_posterior(consumer_id)[target_store]

        # Update 10 times with the same store
        for _ in range(10):
            updater.update(consumer_id, target_store)

        posterior_prob = updater.get_posterior(consumer_id)[target_store]

        assert posterior_prob > prior_prob, (
            f"After 10 visits to {target_store}, P({target_store}) should "
            f"increase. Prior={prior_prob:.4f}, Posterior={posterior_prob:.4f}"
        )


class TestBayesianEntropyDecreasesWithEvidence:
    """test_bayesian_entropy_decreases_with_evidence: entropy drops as we observe one store."""

    def test_bayesian_entropy_decreases_with_evidence(self, synth):
        """Shannon entropy of the posterior should decrease as more evidence
        for a single store accumulates."""
        store_ids = [s.store_id for s in synth["stores"]]
        updater = BayesianUpdater(store_ids=store_ids, prior_weight=10.0)

        consumer_id = "test_consumer_003"
        target_store = store_ids[0]
        updater.register_consumer(consumer_id)

        # Measure initial entropy
        initial_entropy = updater.posterior_entropy(consumer_id)

        # Add 20 observations for the same store
        for _ in range(20):
            updater.update(consumer_id, target_store)

        final_entropy = updater.posterior_entropy(consumer_id)

        assert final_entropy < initial_entropy, (
            f"Entropy should decrease with accumulating evidence. "
            f"Initial={initial_entropy:.4f}, Final={final_entropy:.4f}"
        )
