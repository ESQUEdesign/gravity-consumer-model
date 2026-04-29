"""
Geographically Weighted Regression for Spatially Varying Huff Parameters
=========================================================================
Extends the Huff gravity model by allowing the distance-decay exponent
(lambda) and attractiveness exponent (alpha) to vary continuously across
the study area.

The key insight: consumer spatial behaviour is not uniform. Urban shoppers
tolerate longer travel distances (lower lambda) than rural shoppers, and
the relative importance of store attractiveness (alpha) shifts with local
competitive density. A single global (alpha, lambda) pair therefore
misrepresents both environments.

At each origin *i*, GWR fits a local Huff model using all observations
weighted by a spatial kernel centred on origin *i*:

    w_ij = K(d(origin_i, origin_j) / bandwidth)

where K is a kernel function (Gaussian or bisquare) and *d* is the
great-circle distance between the two origins.  The weighted local
log-likelihood is:

    LL_i(alpha_i, lambda_i) = SUM_j w_ij * SUM_k obs_jk * log(P_jk)

where P_jk is the Huff probability at origin j for store k, computed
with parameters (alpha_i, lambda_i).

Bandwidth selection
-------------------
The bandwidth controls the bias-variance trade-off.  A small bandwidth
gives highly localised (low-bias, high-variance) estimates; a large
bandwidth smooths toward the global Huff solution.

Bandwidth can be:
    * **Fixed** -- a constant distance in kilometres.
    * **Adaptive** -- the distance to the N-th nearest neighbour, so
      each local model uses exactly N neighbours.

An optimal bandwidth (fixed or adaptive) can be selected via
leave-one-out cross-validation, minimising the total prediction error
across all origins.

References
----------
Fotheringham, A.S., Brunsdon, C. & Charlton, M. (2002).
    *Geographically Weighted Regression: The Analysis of Spatially
    Varying Relationships*. Wiley.

Huff, D.L. (1963). "A Probabilistic Analysis of Shopping Center Trade
    Areas." *Land Economics*, 39(1), 81-90.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gravity.data.schema import build_distance_matrix, haversine_distance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ALPHA: float = 1.0
_DEFAULT_LAMBDA: float = 2.0
_MIN_DISTANCE_KM: float = 0.01


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

def _gaussian_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian kernel: exp(-0.5 * (d / h)^2).

    Parameters
    ----------
    distances : np.ndarray
        Array of distances (any shape).
    bandwidth : float
        Kernel bandwidth (same units as distances).

    Returns
    -------
    np.ndarray
        Weights in (0, 1], same shape as ``distances``.
    """
    return np.exp(-0.5 * (distances / bandwidth) ** 2)


def _bisquare_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Bisquare (bi-weight) kernel: (1 - (d/h)^2)^2 for d < h, else 0.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances (any shape).
    bandwidth : float
        Kernel bandwidth (same units as distances).

    Returns
    -------
    np.ndarray
        Weights in [0, 1], same shape as ``distances``.
    """
    u = distances / bandwidth
    weights = (1.0 - u ** 2) ** 2
    weights[u >= 1.0] = 0.0
    return weights


_KERNELS = {
    "gaussian": _gaussian_kernel,
    "bisquare": _bisquare_kernel,
}


# ---------------------------------------------------------------------------
# GWRModel
# ---------------------------------------------------------------------------

class GWRModel:
    """Geographically Weighted Regression for spatially varying Huff parameters.

    For each consumer origin, the model estimates a local (alpha_i, lambda_i)
    pair by maximising a kernel-weighted log-likelihood over all observed
    origin-store visit shares.  The kernel centres on origin *i* and weights
    other origins by their proximity, so nearby origins have more influence
    on the local parameter estimate than distant ones.

    Parameters
    ----------
    kernel : str, default "bisquare"
        Spatial kernel function.  One of ``"gaussian"`` or ``"bisquare"``.
    bandwidth : float or None, default None
        Fixed kernel bandwidth in kilometres.  Mutually exclusive with
        ``n_neighbors``.  If both ``bandwidth`` and ``n_neighbors`` are
        ``None``, you must call ``select_bandwidth()`` before ``fit()``.
    n_neighbors : int or None, default None
        Number of nearest-neighbour origins to include in each local
        regression (adaptive bandwidth).  Mutually exclusive with
        ``bandwidth``.
    attractiveness : str or callable, default "square_footage"
        Column name or callable that extracts a non-negative attractiveness
        scalar for each store (same semantics as ``HuffModel``).
    alpha_bounds : tuple, default (0.01, 10.0)
        Bounds for the local alpha parameter during optimisation.
    lam_bounds : tuple, default (0.01, 10.0)
        Bounds for the local lambda parameter during optimisation.
    min_distance : float, default 0.01
        Floor applied to origin-store distances to prevent division by zero.
    global_alpha : float, default 1.0
        Starting value for alpha in each local optimisation.
    global_lam : float, default 2.0
        Starting value for lambda in each local optimisation.

    Examples
    --------
    >>> gwr = GWRModel(kernel="bisquare", n_neighbors=30)
    >>> gwr.fit(origins_df, stores_df, observed_shares)
    >>> local_params = gwr.get_local_params()
    >>> probs = gwr.predict(origins_df, stores_df)
    """

    def __init__(
        self,
        kernel: str = "bisquare",
        bandwidth: Optional[float] = None,
        n_neighbors: Optional[int] = None,
        attractiveness: Union[str, Callable[[pd.Series], float]] = "square_footage",
        alpha_bounds: tuple[float, float] = (0.01, 10.0),
        lam_bounds: tuple[float, float] = (0.01, 10.0),
        min_distance: float = _MIN_DISTANCE_KM,
        global_alpha: float = _DEFAULT_ALPHA,
        global_lam: float = _DEFAULT_LAMBDA,
    ) -> None:
        if kernel not in _KERNELS:
            raise ValueError(
                f"Unknown kernel '{kernel}'.  "
                f"Supported kernels: {list(_KERNELS.keys())}"
            )
        if bandwidth is not None and n_neighbors is not None:
            raise ValueError(
                "Specify either 'bandwidth' (fixed) or 'n_neighbors' "
                "(adaptive), not both."
            )
        if n_neighbors is not None and n_neighbors < 3:
            raise ValueError(
                f"n_neighbors must be >= 3, got {n_neighbors}."
            )

        self.kernel = kernel
        self.bandwidth = bandwidth
        self.n_neighbors = n_neighbors
        self.attractiveness = attractiveness
        self.alpha_bounds = alpha_bounds
        self.lam_bounds = lam_bounds
        self.min_distance = min_distance
        self.global_alpha = global_alpha
        self.global_lam = global_lam

        # Populated after fit()
        self._fitted: bool = False
        self._local_alphas: Optional[np.ndarray] = None
        self._local_lams: Optional[np.ndarray] = None
        self._origin_index: Optional[pd.Index] = None
        self._cv_score: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        origins: pd.DataFrame,
        stores: pd.DataFrame,
        observed_shares: pd.DataFrame,
        *,
        method: str = "L-BFGS-B",
        verbose: bool = False,
    ) -> "GWRModel":
        """Fit locally varying (alpha_i, lambda_i) for every origin.

        For each origin *i*, a weighted log-likelihood is maximised where
        the weights come from a spatial kernel centred on origin *i*.

        Parameters
        ----------
        origins : pd.DataFrame
            Origins indexed by ``origin_id`` with at least ``lat`` and ``lon``.
        stores : pd.DataFrame
            Stores indexed by ``store_id`` with at least ``lat``, ``lon``,
            and the attractiveness column.
        observed_shares : pd.DataFrame
            Origin x store matrix of observed visit shares.  Each row must
            sum to 1.0 (or close).  Index and columns must align with
            ``origins`` and ``stores``.
        method : str, default "L-BFGS-B"
            Optimiser method forwarded to ``scipy.optimize.minimize``.
        verbose : bool, default False
            If True, log per-origin optimisation details.

        Returns
        -------
        GWRModel
            ``self``, with locally estimated parameters stored internally.

        Raises
        ------
        ValueError
            If neither ``bandwidth`` nor ``n_neighbors`` is set.
        """
        if self.bandwidth is None and self.n_neighbors is None:
            raise ValueError(
                "No bandwidth specified.  Set 'bandwidth' (fixed) or "
                "'n_neighbors' (adaptive) before calling fit(), or use "
                "select_bandwidth() to choose one via cross-validation."
            )

        # Build matrices.
        dist_os = build_distance_matrix(origins, stores)
        attract = self._compute_attractiveness(stores)
        origin_dist = self._build_origin_distance_matrix(origins)

        # Align observed shares.
        obs = observed_shares.reindex(index=dist_os.index, columns=dist_os.columns)
        if obs.isna().any().any():
            raise ValueError(
                "observed_shares must cover all origin-store pairs present "
                "in the distance matrix.  Found NaN after alignment."
            )

        obs_np = obs.values.astype(np.float64)
        dist_np = dist_os.values.astype(np.float64)
        attract_np = attract.values.astype(np.float64)
        origin_dist_np = origin_dist.values.astype(np.float64)

        n_origins = len(origins)
        local_alphas = np.empty(n_origins, dtype=np.float64)
        local_lams = np.empty(n_origins, dtype=np.float64)

        for i in range(n_origins):
            bw_i = self._get_bandwidth(i, origin_dist_np)
            kernel_fn = _KERNELS[self.kernel]
            weights = kernel_fn(origin_dist_np[i], bw_i)

            alpha_i, lam_i = self._fit_local(
                weights, attract_np, dist_np, obs_np,
                method=method,
            )
            local_alphas[i] = alpha_i
            local_lams[i] = lam_i

            if verbose:
                oid = origins.index[i]
                logger.info(
                    "Origin %s: alpha=%.4f, lambda=%.4f (bw=%.2f km)",
                    oid, alpha_i, lam_i, bw_i,
                )

        self._local_alphas = local_alphas
        self._local_lams = local_lams
        self._origin_index = origins.index.copy()
        self._fitted = True

        logger.info(
            "GWR fit complete: %d origins, alpha range [%.4f, %.4f], "
            "lambda range [%.4f, %.4f]",
            n_origins,
            local_alphas.min(), local_alphas.max(),
            local_lams.min(), local_lams.max(),
        )
        return self

    def predict(
        self,
        origins: pd.DataFrame,
        stores: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute origin x store visit probabilities using local parameters.

        Each origin uses its own (alpha_i, lambda_i) to compute a Huff
        probability row.  Origins not seen during ``fit()`` use the mean
        of the fitted local parameters.

        Parameters
        ----------
        origins : pd.DataFrame
            Origins indexed by ``origin_id`` with at least ``lat`` and ``lon``.
        stores : pd.DataFrame
            Stores indexed by ``store_id`` with at least ``lat``, ``lon``,
            and the attractiveness column.

        Returns
        -------
        pd.DataFrame
            DataFrame of shape ``(n_origins, n_stores)`` where cell (i, j)
            is P(origin_i -> store_j).  Each row sums to 1.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted.  Call fit() first."
            )

        dist = build_distance_matrix(origins, stores)
        attract = self._compute_attractiveness(stores)
        dist_np = dist.values.astype(np.float64)
        attract_np = attract.values.astype(np.float64)

        n_origins = len(origins)
        n_stores = len(stores)
        probs = np.empty((n_origins, n_stores), dtype=np.float64)

        # Build a lookup from origin_id -> fitted parameter index.
        fitted_lookup = {
            oid: idx for idx, oid in enumerate(self._origin_index)
        }
        fallback_alpha = float(self._local_alphas.mean())
        fallback_lam = float(self._local_lams.mean())

        for i, oid in enumerate(origins.index):
            if oid in fitted_lookup:
                idx = fitted_lookup[oid]
                a_i = self._local_alphas[idx]
                l_i = self._local_lams[idx]
            else:
                a_i = fallback_alpha
                l_i = fallback_lam

            probs[i, :] = self._huff_row(a_i, l_i, attract_np, dist_np[i])

        return pd.DataFrame(probs, index=dist.index, columns=dist.columns)

    def get_local_params(self) -> pd.DataFrame:
        """Return a DataFrame of spatially varying parameters by origin.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by ``origin_id`` with columns ``alpha`` and
            ``lam`` (one row per origin from the training set).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted.  Call fit() first."
            )
        return pd.DataFrame(
            {"alpha": self._local_alphas, "lam": self._local_lams},
            index=self._origin_index,
        )

    def select_bandwidth(
        self,
        origins: pd.DataFrame,
        stores: pd.DataFrame,
        observed_shares: pd.DataFrame,
        *,
        mode: str = "adaptive",
        candidates: Optional[list] = None,
        method: str = "L-BFGS-B",
        verbose: bool = False,
    ) -> "GWRModel":
        """Select optimal bandwidth via leave-one-out cross-validation.

        For each candidate bandwidth, the model performs LOO-CV: for every
        origin *i*, it fits local parameters excluding origin *i* from its
        own kernel weight (setting w_ii = 0), then evaluates the prediction
        error at origin *i*.  The candidate that minimises total negative
        log-likelihood across all origins is selected.

        Parameters
        ----------
        origins : pd.DataFrame
            Origins indexed by ``origin_id``.
        stores : pd.DataFrame
            Stores indexed by ``store_id``.
        observed_shares : pd.DataFrame
            Origin x store observed visit-share matrix.
        mode : str, default "adaptive"
            ``"adaptive"`` searches over number of neighbours;
            ``"fixed"`` searches over distances in kilometres.
        candidates : list or None
            Candidate bandwidths to evaluate.

            * For ``"adaptive"`` mode: list of integers (neighbour counts).
              Defaults to ``[10, 15, 20, 30, 50, 75, 100]`` (capped at
              ``n_origins - 1``).
            * For ``"fixed"`` mode: list of floats (km).  Defaults to
              ``[5, 10, 20, 30, 50, 75, 100]``.
        method : str, default "L-BFGS-B"
            Optimiser method for each local fit.
        verbose : bool, default False
            If True, log CV scores for each candidate.

        Returns
        -------
        GWRModel
            ``self``, with ``bandwidth`` or ``n_neighbors`` set to the
            best candidate.  The attribute ``_cv_score`` stores the
            winning score.

        Raises
        ------
        ValueError
            If ``mode`` is not ``"adaptive"`` or ``"fixed"``.
        """
        if mode not in ("adaptive", "fixed"):
            raise ValueError(
                f"mode must be 'adaptive' or 'fixed', got '{mode}'."
            )

        n_origins = len(origins)

        if candidates is None:
            if mode == "adaptive":
                candidates = [n for n in [10, 15, 20, 30, 50, 75, 100]
                              if n < n_origins]
                if not candidates:
                    candidates = [max(3, n_origins - 1)]
            else:
                candidates = [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]

        # Pre-compute everything once.
        dist_os = build_distance_matrix(origins, stores)
        attract = self._compute_attractiveness(stores)
        origin_dist = self._build_origin_distance_matrix(origins)

        obs = observed_shares.reindex(index=dist_os.index, columns=dist_os.columns)
        if obs.isna().any().any():
            raise ValueError(
                "observed_shares must cover all origin-store pairs.  "
                "Found NaN after alignment."
            )

        obs_np = obs.values.astype(np.float64)
        dist_np = dist_os.values.astype(np.float64)
        attract_np = attract.values.astype(np.float64)
        origin_dist_np = origin_dist.values.astype(np.float64)

        best_score = np.inf
        best_candidate = candidates[0]

        for cand in candidates:
            score = self._loo_cv_score(
                cand, mode, origin_dist_np, attract_np, dist_np, obs_np,
                method=method,
            )
            if verbose:
                label = f"n_neighbors={cand}" if mode == "adaptive" else f"bw={cand:.1f} km"
                logger.info("CV %s: score=%.4f", label, score)

            if score < best_score:
                best_score = score
                best_candidate = cand

        # Apply the best candidate.
        if mode == "adaptive":
            self.n_neighbors = int(best_candidate)
            self.bandwidth = None
        else:
            self.bandwidth = float(best_candidate)
            self.n_neighbors = None

        self._cv_score = best_score

        label = (f"n_neighbors={self.n_neighbors}" if mode == "adaptive"
                 else f"bandwidth={self.bandwidth:.1f} km")
        logger.info(
            "Bandwidth selection complete: %s (CV score=%.4f)",
            label, best_score,
        )
        return self

    # ------------------------------------------------------------------
    # Diagnostics / introspection
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit()`` has been called successfully."""
        return self._fitted

    @property
    def params(self) -> dict:
        """Current model configuration and summary statistics."""
        info = {
            "kernel": self.kernel,
            "bandwidth": self.bandwidth,
            "n_neighbors": self.n_neighbors,
            "alpha_bounds": self.alpha_bounds,
            "lam_bounds": self.lam_bounds,
            "global_alpha": self.global_alpha,
            "global_lam": self.global_lam,
        }
        if self._fitted:
            info["alpha_mean"] = float(self._local_alphas.mean())
            info["alpha_std"] = float(self._local_alphas.std())
            info["lam_mean"] = float(self._local_lams.mean())
            info["lam_std"] = float(self._local_lams.std())
        if self._cv_score is not None:
            info["cv_score"] = self._cv_score
        return info

    def summary(self) -> str:
        """Human-readable summary of the model state."""
        status = "fitted" if self._fitted else "not fitted"
        bw_type = "adaptive" if self.n_neighbors is not None else "fixed"
        bw_value = (self.n_neighbors if self.n_neighbors is not None
                    else self.bandwidth)
        lines = [
            "GWRModel (Geographically Weighted Huff)",
            f"  status          : {status}",
            f"  kernel          : {self.kernel}",
            f"  bandwidth type  : {bw_type}",
            f"  bandwidth value : {bw_value}",
            f"  attractiveness  : {self.attractiveness}",
            f"  min_distance_km : {self.min_distance}",
        ]
        if self._fitted:
            lines.extend([
                f"  n_origins       : {len(self._origin_index)}",
                f"  alpha (mean)    : {self._local_alphas.mean():.4f}",
                f"  alpha (std)     : {self._local_alphas.std():.4f}",
                f"  alpha (range)   : [{self._local_alphas.min():.4f}, "
                f"{self._local_alphas.max():.4f}]",
                f"  lambda (mean)   : {self._local_lams.mean():.4f}",
                f"  lambda (std)    : {self._local_lams.std():.4f}",
                f"  lambda (range)  : [{self._local_lams.min():.4f}, "
                f"{self._local_lams.max():.4f}]",
            ])
        if self._cv_score is not None:
            lines.append(f"  cv_score        : {self._cv_score:.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        bw_str = (f"n_neighbors={self.n_neighbors}" if self.n_neighbors is not None
                  else f"bandwidth={self.bandwidth}")
        return (
            f"GWRModel(kernel={self.kernel!r}, {bw_str}, "
            f"fitted={self._fitted})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_origin_distance_matrix(
        self, origins: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build an origin x origin distance matrix (km).

        Uses ``build_distance_matrix`` with origins in both positions.

        Parameters
        ----------
        origins : pd.DataFrame
            Origins indexed by ``origin_id`` with ``lat`` and ``lon``.

        Returns
        -------
        pd.DataFrame
            Symmetric ``(n_origins, n_origins)`` distance matrix in km.
        """
        return build_distance_matrix(origins, origins)

    def _compute_attractiveness(self, stores: pd.DataFrame) -> pd.Series:
        """Derive a 1-D attractiveness vector from the stores DataFrame.

        Parameters
        ----------
        stores : pd.DataFrame
            Stores DataFrame indexed by ``store_id``.

        Returns
        -------
        pd.Series
            Non-negative attractiveness per store.

        Raises
        ------
        ValueError
            If any attractiveness value is negative.
        KeyError
            If the specified column does not exist.
        """
        if callable(self.attractiveness):
            attract = stores.apply(self.attractiveness, axis=1)
        elif isinstance(self.attractiveness, str):
            if self.attractiveness not in stores.columns:
                raise KeyError(
                    f"Attractiveness column '{self.attractiveness}' not found "
                    f"in stores DataFrame.  Available columns: "
                    f"{list(stores.columns)}"
                )
            attract = stores[self.attractiveness].copy()
        else:
            raise TypeError(
                f"attractiveness must be a str or callable, "
                f"got {type(self.attractiveness).__name__}"
            )

        attract = attract.astype(np.float64)
        if (attract < 0).any():
            raise ValueError(
                "Attractiveness values must be non-negative.  "
                "Found negative values for stores: "
                f"{list(attract[attract < 0].index)}"
            )
        attract = attract.clip(lower=1e-9)
        return attract

    def _get_bandwidth(
        self, i: int, origin_dist: np.ndarray,
    ) -> float:
        """Determine the effective bandwidth for origin *i*.

        For fixed bandwidth, returns ``self.bandwidth`` directly.  For
        adaptive bandwidth, returns the distance to the N-th nearest
        neighbour of origin *i*.

        Parameters
        ----------
        i : int
            Row index of the focal origin in ``origin_dist``.
        origin_dist : np.ndarray
            Square ``(n_origins, n_origins)`` distance matrix.

        Returns
        -------
        float
            Bandwidth in kilometres.
        """
        if self.bandwidth is not None:
            return self.bandwidth

        # Adaptive: distance to the N-th nearest neighbour.
        dists_i = origin_dist[i].copy()
        # Sort; index 0 is the origin itself (distance 0).
        sorted_dists = np.sort(dists_i)
        # The N-th nearest neighbour is at index n_neighbors (0-indexed:
        # index 0 = self, index 1 = nearest, ..., index N = N-th nearest).
        n = min(self.n_neighbors, len(sorted_dists) - 1)
        bw = sorted_dists[n]
        # Floor bandwidth to avoid degenerate kernels.
        return max(bw, _MIN_DISTANCE_KM)

    def _fit_local(
        self,
        weights: np.ndarray,
        attractiveness: np.ndarray,
        distances: np.ndarray,
        observed: np.ndarray,
        *,
        method: str = "L-BFGS-B",
    ) -> tuple[float, float]:
        """Fit a local (alpha, lambda) pair using weighted log-likelihood.

        Parameters
        ----------
        weights : np.ndarray
            1-D array of length ``n_origins`` -- spatial kernel weights.
        attractiveness : np.ndarray
            1-D array of length ``n_stores``.
        distances : np.ndarray
            2-D array ``(n_origins, n_stores)`` in km.
        observed : np.ndarray
            2-D array ``(n_origins, n_stores)`` of observed shares.
        method : str
            Optimiser method.

        Returns
        -------
        tuple[float, float]
            ``(alpha_i, lambda_i)`` for the focal origin.
        """
        def neg_wll(params: np.ndarray) -> float:
            a, l = params
            return self._weighted_neg_ll(
                a, l, weights, attractiveness, distances, observed,
            )

        result = minimize(
            neg_wll,
            x0=(self.global_alpha, self.global_lam),
            method=method,
            bounds=[self.alpha_bounds, self.lam_bounds],
            options={"maxiter": 300},
        )

        return float(result.x[0]), float(result.x[1])

    @staticmethod
    def _huff_row(
        alpha: float,
        lam: float,
        attractiveness: np.ndarray,
        distances_row: np.ndarray,
    ) -> np.ndarray:
        """Compute Huff probabilities for a single origin row.

        Parameters
        ----------
        alpha : float
            Attractiveness exponent.
        lam : float
            Distance-decay exponent.
        attractiveness : np.ndarray
            1-D array of length ``n_stores``.
        distances_row : np.ndarray
            1-D array of length ``n_stores`` -- distances from this origin.

        Returns
        -------
        np.ndarray
            1-D probability array summing to 1.
        """
        safe_dist = np.maximum(distances_row, _MIN_DISTANCE_KM)
        utility = (attractiveness ** alpha) * (safe_dist ** (-lam))
        total = utility.sum()
        if total < 1e-30:
            # Uniform fallback if all utilities are effectively zero.
            return np.full_like(utility, 1.0 / len(utility))
        return utility / total

    @staticmethod
    def _huff_probabilities(
        alpha: float,
        lam: float,
        attractiveness: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        """Core Huff calculation returning an (n_origins, n_stores) array.

        Parameters
        ----------
        alpha : float
            Attractiveness exponent.
        lam : float
            Distance-decay exponent.
        attractiveness : np.ndarray
            1-D array of length ``n_stores``.
        distances : np.ndarray
            2-D array ``(n_origins, n_stores)`` in km.

        Returns
        -------
        np.ndarray
            Probability matrix where each row sums to 1.
        """
        safe_dist = np.maximum(distances, _MIN_DISTANCE_KM)
        utility = (attractiveness[np.newaxis, :] ** alpha) * (safe_dist ** (-lam))
        row_sums = utility.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-30)
        return utility / row_sums

    @staticmethod
    def _weighted_neg_ll(
        alpha: float,
        lam: float,
        weights: np.ndarray,
        attractiveness: np.ndarray,
        distances: np.ndarray,
        observed: np.ndarray,
    ) -> float:
        """Weighted negative log-likelihood for a local Huff model.

        Parameters
        ----------
        alpha : float
        lam : float
        weights : np.ndarray
            1-D array of length ``n_origins`` -- spatial kernel weights.
        attractiveness : np.ndarray
            1-D array of length ``n_stores``.
        distances : np.ndarray
            2-D array ``(n_origins, n_stores)``.
        observed : np.ndarray
            2-D array ``(n_origins, n_stores)`` -- rows sum to 1.

        Returns
        -------
        float
            Scalar weighted negative log-likelihood:
            -SUM_j w_j * SUM_k obs_jk * log(pred_jk)
        """
        pred = GWRModel._huff_probabilities(alpha, lam, attractiveness, distances)
        pred = np.clip(pred, 1e-15, 1.0)

        # Per-origin log-likelihood: sum_k obs_jk * log(pred_jk)
        ll_per_origin = (observed * np.log(pred)).sum(axis=1)

        # Weight by kernel and sum.
        return -np.dot(weights, ll_per_origin)

    def _loo_cv_score(
        self,
        candidate,
        mode: str,
        origin_dist: np.ndarray,
        attractiveness: np.ndarray,
        distances: np.ndarray,
        observed: np.ndarray,
        *,
        method: str = "L-BFGS-B",
    ) -> float:
        """Compute leave-one-out cross-validation score for a candidate bandwidth.

        For each origin *i*, the kernel weight w_ii is set to zero (the
        origin is "left out" of its own local regression), a local model
        is fitted on the remaining weighted observations, and the
        prediction error at origin *i* is recorded.

        Parameters
        ----------
        candidate : int or float
            Neighbour count (adaptive) or distance in km (fixed).
        mode : str
            ``"adaptive"`` or ``"fixed"``.
        origin_dist : np.ndarray
            Square ``(n, n)`` origin-origin distance matrix.
        attractiveness : np.ndarray
            1-D array of length ``n_stores``.
        distances : np.ndarray
            ``(n_origins, n_stores)`` origin-store distance matrix.
        observed : np.ndarray
            ``(n_origins, n_stores)`` observed shares.
        method : str
            Optimiser method for each local fit.

        Returns
        -------
        float
            Total negative log-likelihood summed over all left-out origins.
        """
        n_origins = origin_dist.shape[0]
        kernel_fn = _KERNELS[self.kernel]
        total_nll = 0.0

        # Temporarily override bandwidth settings to use the candidate.
        saved_bw = self.bandwidth
        saved_nn = self.n_neighbors
        if mode == "adaptive":
            self.n_neighbors = int(candidate)
            self.bandwidth = None
        else:
            self.bandwidth = float(candidate)
            self.n_neighbors = None

        for i in range(n_origins):
            bw_i = self._get_bandwidth(i, origin_dist)
            weights = kernel_fn(origin_dist[i], bw_i)
            # Leave-one-out: zero the focal origin's weight.
            weights[i] = 0.0

            # Skip if all weights are zero (degenerate case).
            if weights.sum() < 1e-30:
                continue

            alpha_i, lam_i = self._fit_local(
                weights, attractiveness, distances, observed,
                method=method,
            )

            # Evaluate prediction at the left-out origin.
            pred_row = self._huff_row(
                alpha_i, lam_i, attractiveness, distances[i],
            )
            pred_row = np.clip(pred_row, 1e-15, 1.0)
            total_nll -= np.sum(observed[i] * np.log(pred_row))

        # Restore bandwidth settings.
        self.bandwidth = saved_bw
        self.n_neighbors = saved_nn

        return total_nll
