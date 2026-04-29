"""
Latent Class Gravity Model
===========================
Expectation-Maximization Latent Class Analysis for discovering behavioral
consumer segments from visit/purchase data.

Each latent class *k* receives its own Huff model parameters (alpha_k,
lambda_k), allowing different consumer segments to exhibit different
sensitivities to store attractiveness and travel distance.

The model maximises the mixture log-likelihood:

    LL = SUM_i log( SUM_k  pi_k * L_i(theta_k) )

where pi_k are mixing proportions and L_i(theta_k) is the multinomial
likelihood of consumer i's visit pattern under segment k's parameters.

Model selection uses BIC or AIC when ``max_classes='auto'``.

References
----------
Kamakura, W. A. & Russell, G. J. (1989). "A Probabilistic Choice Model
    for Market Segmentation and Elasticity Structure." *Journal of
    Marketing Research*, 26(4), 379-390.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gravity.data.schema import build_distance_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_DISTANCE_KM: float = 0.01
_CONVERGENCE_TOL: float = 1e-6
_MAX_EM_ITERATIONS: int = 200
_MIN_MIXING_PROPORTION: float = 1e-8


class LatentClassModel:
    """Expectation-Maximization Latent Class Analysis for consumer segments.

    Discovers K behavioral segments from consumer visit/purchase data.
    Each segment receives its own Huff gravity model parameters (alpha_k,
    lambda_k), capturing heterogeneous distance-decay and attractiveness
    sensitivity across the consumer population.

    Parameters
    ----------
    n_classes : int or str, default "auto"
        Number of latent classes.  If ``"auto"``, the model fits K = 2
        through ``max_classes`` and selects the best K by BIC.
    max_classes : int, default 8
        Maximum number of classes to evaluate when ``n_classes="auto"``.
    tol : float, default 1e-6
        Convergence tolerance on the log-likelihood change between EM
        iterations.
    max_iter : int, default 200
        Maximum number of EM iterations per model fit.
    n_restarts : int, default 5
        Number of random restarts to avoid local optima.
    random_state : int or None, default None
        Seed for reproducibility of parameter initialisation.

    Attributes
    ----------
    n_classes_ : int
        Number of classes selected after fitting.
    mixing_proportions_ : np.ndarray
        Array of shape ``(K,)`` with class mixing proportions.
    class_params_ : list[dict]
        Per-class parameters: ``[{"alpha": float, "lambda": float}, ...]``.
    posterior_ : np.ndarray
        Array of shape ``(N, K)`` with posterior membership probabilities
        from the last E-step.
    log_likelihood_ : float
        Final log-likelihood of the fitted model.
    bic_ : float
        Bayesian Information Criterion of the fitted model.
    aic_ : float
        Akaike Information Criterion of the fitted model.

    Examples
    --------
    >>> model = LatentClassModel(n_classes=3)
    >>> model.fit(visit_data, origins_df, stores_df, attractiveness_col="square_footage")
    >>> segments = model.predict_segments(visit_data)
    >>> params = model.get_segment_params()
    """

    def __init__(
        self,
        n_classes: Union[int, str] = "auto",
        max_classes: int = 8,
        tol: float = _CONVERGENCE_TOL,
        max_iter: int = _MAX_EM_ITERATIONS,
        n_restarts: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_classes = n_classes
        self.max_classes = max_classes
        self.tol = tol
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.random_state = random_state

        # Populated after fit()
        self.n_classes_: Optional[int] = None
        self.mixing_proportions_: Optional[np.ndarray] = None
        self.class_params_: Optional[list[dict]] = None
        self.posterior_: Optional[np.ndarray] = None
        self.log_likelihood_: Optional[float] = None
        self.bic_: Optional[float] = None
        self.aic_: Optional[float] = None
        self._fitted: bool = False
        self._origin_ids: Optional[np.ndarray] = None
        self._model_selection_results: Optional[list[dict]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        origins: pd.DataFrame,
        stores: pd.DataFrame,
        attractiveness_col: str = "square_footage",
        distance_matrix: Optional[pd.DataFrame] = None,
    ) -> "LatentClassModel":
        """Fit the latent class model to observed visit pattern data.

        Parameters
        ----------
        data : pd.DataFrame
            Consumer visit patterns with columns ``origin_id``, ``store_id``,
            and either ``visits`` (counts) or ``probability`` (shares).
            If ``visits`` is present, row-level shares are computed by
            normalising within each ``origin_id``.
        origins : pd.DataFrame
            Origins DataFrame indexed by ``origin_id`` with ``lat``, ``lon``.
        stores : pd.DataFrame
            Stores DataFrame indexed by ``store_id`` with ``lat``, ``lon``,
            and the attractiveness column.
        attractiveness_col : str, default "square_footage"
            Column name in ``stores`` to use as store attractiveness.
        distance_matrix : pd.DataFrame or None, default None
            Pre-computed origin x store distance matrix (km).  Computed
            from lat/lon if not supplied.

        Returns
        -------
        LatentClassModel
            Fitted model (``self``).

        Raises
        ------
        ValueError
            If required columns are missing from ``data``.
        """
        # --- Validate input ---
        required_cols = {"origin_id", "store_id"}
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"data is missing required columns: {missing}")
        if "visits" not in data.columns and "probability" not in data.columns:
            raise ValueError(
                "data must contain either a 'visits' or 'probability' column."
            )

        # --- Build observation matrix (origin x store shares) ---
        obs_matrix, origin_ids, store_ids = self._build_observation_matrix(data)
        n_obs = len(origin_ids)

        # --- Build distance matrix ---
        if distance_matrix is not None:
            dist = distance_matrix.reindex(index=origin_ids, columns=store_ids).values
        else:
            origins_sub = origins.reindex(origin_ids)
            stores_sub = stores.reindex(store_ids)
            dist = build_distance_matrix(origins_sub, stores_sub).values
        dist = np.maximum(dist, _MIN_DISTANCE_KM)

        # --- Attractiveness vector ---
        attract = stores.reindex(store_ids)[attractiveness_col].values.astype(
            np.float64
        )
        attract = np.clip(attract, 1e-9, None)

        # --- Store references for prediction ---
        self._origin_ids = origin_ids
        self._obs_matrix = obs_matrix
        self._dist = dist
        self._attract = attract

        # --- Determine K ---
        if self.n_classes == "auto":
            self._fit_auto(obs_matrix, dist, attract, n_obs)
        else:
            K = int(self.n_classes)
            result = self._fit_k(K, obs_matrix, dist, attract, n_obs)
            self._set_results(result, n_obs)

        self._fitted = True
        logger.info(
            "Fitted LatentClassModel with K=%d classes, LL=%.4f, BIC=%.4f",
            self.n_classes_,
            self.log_likelihood_,
            self.bic_,
        )
        return self

    def predict_segments(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Assign consumers to their most probable segment.

        Parameters
        ----------
        data : pd.DataFrame or None
            If None, uses the data from the last ``fit()`` call.  If
            provided, must have the same ``origin_id`` values as the
            training data (re-scoring with new data requires ``fit()``).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - ``origin_id``
            - ``segment_id`` (int, 0-indexed)
            - ``membership_probability`` (posterior probability of the
              assigned segment)
        """
        self._check_fitted()

        posterior = self.posterior_
        segment_ids = np.argmax(posterior, axis=1)
        membership_probs = posterior[np.arange(len(segment_ids)), segment_ids]

        result = pd.DataFrame(
            {
                "origin_id": self._origin_ids,
                "segment_id": segment_ids,
                "membership_probability": membership_probs,
            }
        )
        return result

    def get_segment_params(self) -> dict[int, dict]:
        """Return per-segment parameters.

        Returns
        -------
        dict
            Mapping of ``{segment_id: {"alpha": float, "lambda": float,
            "mixing_proportion": float}}``.
        """
        self._check_fitted()

        params = {}
        for k in range(self.n_classes_):
            params[k] = {
                "alpha": self.class_params_[k]["alpha"],
                "lambda": self.class_params_[k]["lambda"],
                "mixing_proportion": float(self.mixing_proportions_[k]),
            }
        return params

    def get_model_selection_results(self) -> Optional[pd.DataFrame]:
        """Return BIC/AIC comparison across K values (auto mode only).

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns ``K``, ``log_likelihood``, ``bic``,
            ``aic``, ``n_params``.  None if model was not fit with
            ``n_classes="auto"``.
        """
        if self._model_selection_results is None:
            return None
        return pd.DataFrame(self._model_selection_results)

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def _fit_auto(
        self,
        obs: np.ndarray,
        dist: np.ndarray,
        attract: np.ndarray,
        n_obs: int,
    ) -> None:
        """Fit models for K = 2..max_classes and select the best by BIC."""
        results = []
        best_bic = np.inf
        best_result = None

        for K in range(2, self.max_classes + 1):
            result = self._fit_k(K, obs, dist, attract, n_obs)
            n_params = self._count_params(K)
            bic = self._compute_bic(result["log_likelihood"], n_params, n_obs)
            aic = self._compute_aic(result["log_likelihood"], n_params)

            results.append(
                {
                    "K": K,
                    "log_likelihood": result["log_likelihood"],
                    "bic": bic,
                    "aic": aic,
                    "n_params": n_params,
                }
            )

            logger.info("K=%d: LL=%.4f, BIC=%.4f, AIC=%.4f", K, result["log_likelihood"], bic, aic)

            if bic < best_bic:
                best_bic = bic
                best_result = result

        self._model_selection_results = results
        self._set_results(best_result, n_obs)

    # ------------------------------------------------------------------
    # Core EM
    # ------------------------------------------------------------------

    def _fit_k(
        self,
        K: int,
        obs: np.ndarray,
        dist: np.ndarray,
        attract: np.ndarray,
        n_obs: int,
    ) -> dict:
        """Fit a single K-class model with multiple random restarts.

        Returns the best result (highest log-likelihood) across restarts.
        """
        rng = np.random.RandomState(self.random_state)
        best_ll = -np.inf
        best_result = None

        for restart in range(self.n_restarts):
            seed = rng.randint(0, 2**31)
            result = self._em(K, obs, dist, attract, n_obs, seed)
            if result["log_likelihood"] > best_ll:
                best_ll = result["log_likelihood"]
                best_result = result

        return best_result

    def _em(
        self,
        K: int,
        obs: np.ndarray,
        dist: np.ndarray,
        attract: np.ndarray,
        n_obs: int,
        seed: int,
    ) -> dict:
        """Run a single EM pass for K classes.

        Parameters
        ----------
        K : int
            Number of latent classes.
        obs : np.ndarray
            Observation matrix (n_origins, n_stores) of visit shares.
        dist : np.ndarray
            Distance matrix (n_origins, n_stores) in km.
        attract : np.ndarray
            Attractiveness vector (n_stores,).
        n_obs : int
            Number of observations (origins).
        seed : int
            Random seed for initialisation.

        Returns
        -------
        dict
            Keys: ``K``, ``mixing``, ``params``, ``posterior``,
            ``log_likelihood``.
        """
        rng = np.random.RandomState(seed)
        n_stores = obs.shape[1]

        # --- Initialise parameters ---
        mixing = np.full(K, 1.0 / K)
        params = []
        for k in range(K):
            params.append(
                {
                    "alpha": rng.uniform(0.5, 2.0),
                    "lambda": rng.uniform(0.5, 3.0),
                }
            )

        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            # --- E-step ---
            posterior, ll = self._e_step(obs, dist, attract, mixing, params, K, n_obs)

            # --- Check convergence ---
            if abs(ll - prev_ll) < self.tol and iteration > 0:
                logger.debug(
                    "EM converged at iteration %d (LL=%.6f)", iteration, ll
                )
                break
            prev_ll = ll

            # --- M-step ---
            mixing, params = self._m_step(
                obs, dist, attract, posterior, K, n_obs, params
            )

        return {
            "K": K,
            "mixing": mixing,
            "params": params,
            "posterior": posterior,
            "log_likelihood": ll,
        }

    def _e_step(
        self,
        obs: np.ndarray,
        dist: np.ndarray,
        attract: np.ndarray,
        mixing: np.ndarray,
        params: list[dict],
        K: int,
        n_obs: int,
    ) -> tuple[np.ndarray, float]:
        """E-step: compute posterior segment membership probabilities.

        For each consumer i and class k, compute:
            gamma_ik = pi_k * L_i(theta_k) / SUM_k' pi_k' * L_i(theta_k')

        where L_i(theta_k) is the multinomial likelihood of consumer i's
        visit pattern under class k's Huff parameters.

        Returns
        -------
        tuple
            (posterior, log_likelihood) where posterior is (n_obs, K) and
            log_likelihood is the scalar mixture log-likelihood.
        """
        # Compute log-likelihoods per class: (n_obs, K)
        log_lik = np.zeros((n_obs, K))
        for k in range(K):
            pred_k = self._huff_probabilities(
                params[k]["alpha"], params[k]["lambda"], attract, dist
            )
            # Multinomial log-likelihood per origin (up to constant)
            log_lik[:, k] = np.sum(obs * np.log(np.clip(pred_k, 1e-15, 1.0)), axis=1)

        # Add log mixing proportions
        log_joint = log_lik + np.log(np.clip(mixing, _MIN_MIXING_PROPORTION, 1.0))

        # Log-sum-exp for numerical stability
        max_log_joint = np.max(log_joint, axis=1, keepdims=True)
        log_marginal = max_log_joint + np.log(
            np.sum(np.exp(log_joint - max_log_joint), axis=1, keepdims=True)
        )

        # Posterior
        log_posterior = log_joint - log_marginal
        posterior = np.exp(log_posterior)

        # Normalise rows to ensure they sum to 1 (guards against numerical drift)
        row_sums = posterior.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-30)
        posterior = posterior / row_sums

        # Log-likelihood
        ll = float(np.sum(log_marginal))

        return posterior, ll

    def _m_step(
        self,
        obs: np.ndarray,
        dist: np.ndarray,
        attract: np.ndarray,
        posterior: np.ndarray,
        K: int,
        n_obs: int,
        current_params: list[dict],
    ) -> tuple[np.ndarray, list[dict]]:
        """M-step: update mixing proportions and per-class Huff parameters.

        Mixing proportions are updated in closed form.  Alpha_k and
        lambda_k are updated via weighted MLE (scipy.optimize.minimize)
        with the posterior probabilities as weights.

        Returns
        -------
        tuple
            (mixing, params) where mixing is (K,) and params is a list
            of dicts with keys ``"alpha"`` and ``"lambda"``.
        """
        # --- Update mixing proportions ---
        mixing = posterior.sum(axis=0) / n_obs
        mixing = np.clip(mixing, _MIN_MIXING_PROPORTION, 1.0)
        mixing /= mixing.sum()

        # --- Update per-class parameters ---
        params = []
        for k in range(K):
            weights_k = posterior[:, k]

            def neg_weighted_ll(theta: np.ndarray, w=weights_k) -> float:
                a, l = theta
                pred = self._huff_probabilities(a, l, attract, dist)
                ll_per_origin = np.sum(
                    obs * np.log(np.clip(pred, 1e-15, 1.0)), axis=1
                )
                return -np.sum(w * ll_per_origin)

            x0 = [current_params[k]["alpha"], current_params[k]["lambda"]]
            result = minimize(
                neg_weighted_ll,
                x0=x0,
                method="L-BFGS-B",
                bounds=[(0.01, 10.0), (0.01, 10.0)],
                options={"maxiter": 100, "disp": False},
            )

            params.append(
                {
                    "alpha": float(result.x[0]),
                    "lambda": float(result.x[1]),
                }
            )

        return mixing, params

    # ------------------------------------------------------------------
    # Huff probability computation
    # ------------------------------------------------------------------

    @staticmethod
    def _huff_probabilities(
        alpha: float,
        lam: float,
        attractiveness: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        """Compute Huff model probabilities for given parameters.

        Parameters
        ----------
        alpha : float
            Attractiveness exponent.
        lam : float
            Distance-decay exponent.
        attractiveness : np.ndarray
            1-D array of store attractiveness values (n_stores,).
        distances : np.ndarray
            2-D array of distances (n_origins, n_stores) in km.

        Returns
        -------
        np.ndarray
            Probability matrix (n_origins, n_stores) where rows sum to 1.
        """
        safe_dist = np.maximum(distances, _MIN_DISTANCE_KM)
        utility = (attractiveness[np.newaxis, :] ** alpha) * (safe_dist ** (-lam))
        row_sums = utility.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-30)
        return utility / row_sums

    # ------------------------------------------------------------------
    # Information criteria
    # ------------------------------------------------------------------

    @staticmethod
    def _count_params(K: int) -> int:
        """Count the number of free parameters for a K-class model.

        Each class has 2 parameters (alpha_k, lambda_k), plus (K-1)
        free mixing proportions.
        """
        return 2 * K + (K - 1)

    @staticmethod
    def _compute_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
        """Bayesian Information Criterion: BIC = -2*LL + p*ln(N)."""
        return -2.0 * log_likelihood + n_params * np.log(n_obs)

    @staticmethod
    def _compute_aic(log_likelihood: float, n_params: int) -> float:
        """Akaike Information Criterion: AIC = -2*LL + 2*p."""
        return -2.0 * log_likelihood + 2.0 * n_params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_results(self, result: dict, n_obs: int) -> None:
        """Store EM results on the model instance."""
        self.n_classes_ = result["K"]
        self.mixing_proportions_ = result["mixing"]
        self.class_params_ = result["params"]
        self.posterior_ = result["posterior"]
        self.log_likelihood_ = result["log_likelihood"]

        n_params = self._count_params(result["K"])
        self.bic_ = self._compute_bic(result["log_likelihood"], n_params, n_obs)
        self.aic_ = self._compute_aic(result["log_likelihood"], n_params)

    def _build_observation_matrix(
        self, data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert long-form visit data to an (n_origins, n_stores) matrix.

        Returns
        -------
        tuple
            (obs_matrix, origin_ids, store_ids)
        """
        if "probability" in data.columns:
            value_col = "probability"
        else:
            value_col = "visits"

        pivot = data.pivot_table(
            index="origin_id",
            columns="store_id",
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )

        # Normalise rows to shares
        row_sums = pivot.sum(axis=1)
        row_sums = row_sums.replace(0, 1)  # avoid division by zero
        obs_matrix = pivot.div(row_sums, axis=0).values.astype(np.float64)

        origin_ids = pivot.index.values
        store_ids = pivot.columns.values

        return obs_matrix, origin_ids, store_ids

    def _check_fitted(self) -> None:
        """Raise if model has not been fitted."""
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before using this method."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model."""
        if not self._fitted:
            return "LatentClassModel (not fitted)"

        lines = [
            "LatentClassModel",
            f"  classes (K)      : {self.n_classes_}",
            f"  log-likelihood   : {self.log_likelihood_:.4f}",
            f"  BIC              : {self.bic_:.4f}",
            f"  AIC              : {self.aic_:.4f}",
            "",
            "  Segment parameters:",
        ]
        for k in range(self.n_classes_):
            p = self.class_params_[k]
            lines.append(
                f"    Segment {k}: alpha={p['alpha']:.4f}, "
                f"lambda={p['lambda']:.4f}, "
                f"mixing={self.mixing_proportions_[k]:.4f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        K = self.n_classes_ if self._fitted else self.n_classes
        return f"LatentClassModel(n_classes={K}, status={status})"
