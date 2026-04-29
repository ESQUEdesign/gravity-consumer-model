"""
Bayesian real-time updating of store-choice probabilities.

Combines structural gravity-model priors (e.g. Huff or Mixed Logit
origin x store probabilities) with observed consumer behaviour via
conjugate Dirichlet-Multinomial updating.

Prior:  Dir(alpha_1, ..., alpha_n)  where alpha_j = prior_weight * P(store_j)
Observation:  consumer chose store k
Posterior:  Dir(alpha_1, ..., alpha_k + 1, ..., alpha_n)

Supports streaming updates, temporal decay, batch processing, and
per-consumer state management.
"""

from typing import Dict, List, Optional, Tuple, Union
import time

import numpy as np
import pandas as pd


class BayesianUpdater:
    """Dirichlet-Multinomial Bayesian updater for store-choice probabilities.

    Parameters
    ----------
    store_ids : list of str
        Identifiers for the stores (columns of the probability matrix).
    prior_weight : float
        Concentration scaling applied to the structural prior.
        Higher values make the prior harder to move.
        A value of 10 means the structural model counts as 10
        pseudo-observations.
    default_prior : np.ndarray or dict, optional
        Default prior probability vector across stores.  If None,
        a uniform prior is used.  Can be overridden per consumer.

    Attributes
    ----------
    consumers_ : dict
        Mapping consumer_id -> dict with keys:
            'alpha'      : np.ndarray of Dirichlet parameters
            'n_updates'  : int, total observations incorporated
            'last_update': float, unix timestamp of last update
            'base_alpha' : np.ndarray, the structural prior alphas
                           (used for resets)
    """

    def __init__(
        self,
        store_ids: List[str],
        prior_weight: float = 10.0,
        default_prior: Optional[Union[np.ndarray, Dict[str, float]]] = None,
    ):
        self.store_ids = list(store_ids)
        self.n_stores = len(self.store_ids)
        self._store_idx: Dict[str, int] = {s: i for i, s in enumerate(self.store_ids)}
        self.prior_weight = float(prior_weight)

        # Default prior
        if default_prior is None:
            self._default_prior = np.ones(self.n_stores) / self.n_stores
        elif isinstance(default_prior, dict):
            self._default_prior = np.array(
                [default_prior.get(s, 1e-6) for s in self.store_ids], dtype=float
            )
            self._default_prior /= self._default_prior.sum()
        else:
            self._default_prior = np.asarray(default_prior, dtype=float)
            self._default_prior /= self._default_prior.sum()

        # Per-consumer state
        self.consumers_: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Consumer initialisation
    # ------------------------------------------------------------------

    def _init_consumer(
        self,
        consumer_id: str,
        prior_probs: Optional[np.ndarray] = None,
    ) -> None:
        """Lazily initialise a consumer's Dirichlet parameters.

        Parameters
        ----------
        consumer_id : str
        prior_probs : np.ndarray, optional
            Consumer-specific structural prior.  Falls back to
            ``default_prior``.
        """
        if prior_probs is None:
            probs = self._default_prior.copy()
        else:
            probs = np.asarray(prior_probs, dtype=float)
            probs /= probs.sum()

        alpha = self.prior_weight * probs
        # Ensure no zero entries (Dirichlet concentration must be > 0)
        alpha = np.maximum(alpha, 1e-8)

        self.consumers_[consumer_id] = {
            "alpha": alpha.copy(),
            "n_updates": 0,
            "last_update": time.time(),
            "base_alpha": alpha.copy(),
        }

    # ------------------------------------------------------------------
    # Single update
    # ------------------------------------------------------------------

    def update(
        self,
        consumer_id: str,
        event: str,
        prior_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Update posterior after observing a single store visit.

        Parameters
        ----------
        consumer_id : str
            Consumer identifier.
        event : str
            The store_id that was visited.
        prior_probs : np.ndarray, optional
            If the consumer has not been seen before, use this as
            the structural prior.  Ignored if the consumer already
            has state.

        Returns
        -------
        np.ndarray
            Updated posterior mean probabilities (length n_stores).

        Raises
        ------
        ValueError
            If *event* is not a recognised store_id.
        """
        if event not in self._store_idx:
            raise ValueError(
                f"Unknown store '{event}'. Known stores: {self.store_ids}"
            )

        if consumer_id not in self.consumers_:
            self._init_consumer(consumer_id, prior_probs)

        state = self.consumers_[consumer_id]
        idx = self._store_idx[event]
        state["alpha"][idx] += 1.0
        state["n_updates"] += 1
        state["last_update"] = time.time()

        return self._posterior_mean(state["alpha"])

    # ------------------------------------------------------------------
    # Batch update
    # ------------------------------------------------------------------

    def batch_update(
        self,
        consumer_id: str,
        events: List[str],
        prior_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Incorporate multiple observations at once.

        Parameters
        ----------
        consumer_id : str
        events : list of str
            Sequence of store_ids visited (order does not matter
            for the conjugate update, but ``n_updates`` accumulates).
        prior_probs : np.ndarray, optional
            Structural prior (used only for first-time consumers).

        Returns
        -------
        np.ndarray
            Posterior mean probabilities after all events.
        """
        if consumer_id not in self.consumers_:
            self._init_consumer(consumer_id, prior_probs)

        state = self.consumers_[consumer_id]

        for event in events:
            if event not in self._store_idx:
                raise ValueError(
                    f"Unknown store '{event}'. Known stores: {self.store_ids}"
                )
            idx = self._store_idx[event]
            state["alpha"][idx] += 1.0
            state["n_updates"] += 1

        state["last_update"] = time.time()
        return self._posterior_mean(state["alpha"])

    # ------------------------------------------------------------------
    # Posterior retrieval
    # ------------------------------------------------------------------

    def get_posterior(self, consumer_id: str) -> Dict[str, float]:
        """Return current posterior probabilities for a consumer.

        Parameters
        ----------
        consumer_id : str

        Returns
        -------
        dict
            store_id -> posterior probability (Dirichlet mean).

        Raises
        ------
        KeyError
            If *consumer_id* has no state.
        """
        if consumer_id not in self.consumers_:
            raise KeyError(
                f"Consumer '{consumer_id}' has no state. "
                f"Call update() first or init with a prior."
            )
        alpha = self.consumers_[consumer_id]["alpha"]
        probs = self._posterior_mean(alpha)
        return {s: float(probs[i]) for i, s in enumerate(self.store_ids)}

    def get_posterior_array(self, consumer_id: str) -> np.ndarray:
        """Return posterior mean as a numpy array.

        Parameters
        ----------
        consumer_id : str

        Returns
        -------
        np.ndarray
            Shape (n_stores,).
        """
        if consumer_id not in self.consumers_:
            raise KeyError(f"Consumer '{consumer_id}' has no state.")
        return self._posterior_mean(self.consumers_[consumer_id]["alpha"])

    def get_alpha(self, consumer_id: str) -> np.ndarray:
        """Return raw Dirichlet concentration parameters.

        Parameters
        ----------
        consumer_id : str

        Returns
        -------
        np.ndarray
            Shape (n_stores,).
        """
        if consumer_id not in self.consumers_:
            raise KeyError(f"Consumer '{consumer_id}' has no state.")
        return self.consumers_[consumer_id]["alpha"].copy()

    # ------------------------------------------------------------------
    # Temporal decay
    # ------------------------------------------------------------------

    def decay_prior(
        self,
        half_life_days: float,
        reference_time: Optional[float] = None,
        consumer_ids: Optional[List[str]] = None,
    ) -> int:
        """Time-decay accumulated observations so recent behaviour
        weighs more.

        Applies exponential decay to the difference between current
        alphas and the structural base prior:

            alpha_new = base_alpha + (alpha - base_alpha) * 2^(-dt / half_life)

        where dt is days since last update.

        Parameters
        ----------
        half_life_days : float
            Half-life in days.  After this many days, the
            informational content of past observations is halved.
        reference_time : float, optional
            Unix timestamp to measure staleness from.
            Defaults to now.
        consumer_ids : list of str, optional
            Subset of consumers to decay.  Defaults to all.

        Returns
        -------
        int
            Number of consumers decayed.
        """
        if reference_time is None:
            reference_time = time.time()

        half_life_seconds = half_life_days * 86400.0
        targets = consumer_ids if consumer_ids is not None else list(self.consumers_.keys())
        count = 0

        for cid in targets:
            if cid not in self.consumers_:
                continue
            state = self.consumers_[cid]
            dt_seconds = reference_time - state["last_update"]
            if dt_seconds <= 0:
                continue

            decay_factor = 2.0 ** (-dt_seconds / half_life_seconds)
            base = state["base_alpha"]
            state["alpha"] = base + (state["alpha"] - base) * decay_factor
            # Ensure positivity
            state["alpha"] = np.maximum(state["alpha"], 1e-8)
            count += 1

        return count

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, consumer_id: str) -> None:
        """Reset a consumer back to the structural prior.

        Parameters
        ----------
        consumer_id : str
            Consumer to reset.  No-op if consumer has no state.
        """
        if consumer_id not in self.consumers_:
            return
        state = self.consumers_[consumer_id]
        state["alpha"] = state["base_alpha"].copy()
        state["n_updates"] = 0
        state["last_update"] = time.time()

    def reset_all(self) -> None:
        """Reset every consumer back to structural prior."""
        for cid in list(self.consumers_.keys()):
            self.reset(cid)

    # ------------------------------------------------------------------
    # Bulk operations & reporting
    # ------------------------------------------------------------------

    def register_consumer(
        self,
        consumer_id: str,
        prior_probs: Optional[np.ndarray] = None,
    ) -> None:
        """Pre-register a consumer with a structural prior.

        Parameters
        ----------
        consumer_id : str
        prior_probs : np.ndarray, optional
            Store-choice probabilities from the gravity model.
        """
        self._init_consumer(consumer_id, prior_probs)

    def register_consumers_from_matrix(
        self,
        consumer_ids: List[str],
        prob_matrix: np.ndarray,
    ) -> None:
        """Bulk-register consumers from a probability matrix.

        Parameters
        ----------
        consumer_ids : list of str
            Length-m list of consumer identifiers.
        prob_matrix : np.ndarray
            Shape (m, n_stores) matrix of structural prior
            probabilities (e.g. from a Huff model).
        """
        prob_matrix = np.asarray(prob_matrix, dtype=float)
        if prob_matrix.shape != (len(consumer_ids), self.n_stores):
            raise ValueError(
                f"prob_matrix shape {prob_matrix.shape} does not match "
                f"({len(consumer_ids)}, {self.n_stores})."
            )
        for i, cid in enumerate(consumer_ids):
            self._init_consumer(cid, prob_matrix[i])

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising all tracked consumers.

        Returns
        -------
        pd.DataFrame
            Columns: consumer_id, n_updates, last_update,
            plus one column per store with posterior probabilities.
        """
        rows = []
        for cid, state in self.consumers_.items():
            probs = self._posterior_mean(state["alpha"])
            row = {
                "consumer_id": cid,
                "n_updates": state["n_updates"],
                "last_update": pd.Timestamp(state["last_update"], unit="s"),
            }
            for i, sid in enumerate(self.store_ids):
                row[sid] = probs[i]
            rows.append(row)

        if not rows:
            return pd.DataFrame(
                columns=["consumer_id", "n_updates", "last_update"] + self.store_ids
            )
        return pd.DataFrame(rows)

    def posterior_entropy(self, consumer_id: str) -> float:
        """Shannon entropy of the posterior (bits).

        Higher entropy means more uncertainty about which store
        the consumer will choose.

        Parameters
        ----------
        consumer_id : str

        Returns
        -------
        float
            Entropy in bits.
        """
        probs = self.get_posterior_array(consumer_id)
        # Filter zeros then renormalize so entropy is mathematically correct
        probs = probs[probs > 0]
        probs = probs / probs.sum()
        return float(-np.sum(probs * np.log2(probs)))

    def kl_from_prior(self, consumer_id: str) -> float:
        """KL divergence from structural prior to current posterior.

        D_KL(posterior || prior).  Measures how far observed
        behaviour has shifted the distribution away from the
        structural model.

        Parameters
        ----------
        consumer_id : str

        Returns
        -------
        float
            KL divergence in nats.
        """
        if consumer_id not in self.consumers_:
            raise KeyError(f"Consumer '{consumer_id}' has no state.")

        state = self.consumers_[consumer_id]
        posterior = self._posterior_mean(state["alpha"])
        prior = self._posterior_mean(state["base_alpha"])

        # Clip to avoid log(0) then renormalize to maintain valid distributions
        posterior = np.clip(posterior, 1e-15, 1.0)
        posterior = posterior / posterior.sum()
        prior = np.clip(prior, 1e-15, 1.0)
        prior = prior / prior.sum()

        return float(np.sum(posterior * np.log(posterior / prior)))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _posterior_mean(alpha: np.ndarray) -> np.ndarray:
        """Dirichlet mean: alpha_j / sum(alpha)."""
        total = alpha.sum()
        if total == 0:
            return np.ones_like(alpha) / len(alpha)
        return alpha / total

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_consumers(self) -> int:
        """Number of consumers currently tracked."""
        return len(self.consumers_)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BayesianUpdater(stores={self.n_stores}, "
            f"consumers={self.n_consumers}, "
            f"prior_weight={self.prior_weight})"
        )
