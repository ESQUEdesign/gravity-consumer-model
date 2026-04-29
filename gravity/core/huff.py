"""
Huff Retail Gravity Model
=========================
Foundational spatial interaction model for consumer store-choice probability.

The Huff model estimates the probability that a consumer at origin *i* will
choose store *j* as:

    P(C_i -> S_j) = (A_j^alpha * D_ij^(-lambda))
                     / SUM_k (A_k^alpha * D_ik^(-lambda))

where:
    A_j     = attractiveness of store j
    D_ij    = distance from origin i to store j
    alpha   = attractiveness exponent (sensitivity to store size/quality)
    lambda  = distance-decay exponent (sensitivity to travel cost)

Attractiveness can be a single numeric column (e.g. square_footage) or a
custom callable that computes a scalar from each row of the stores DataFrame.

Calibration
-----------
When observed visit-share data is available, ``fit()`` estimates alpha and
lambda via maximum-likelihood estimation (minimising KL-divergence between
predicted and observed shares).

References
----------
Huff, D.L. (1963). "A Probabilistic Analysis of Shopping Center Trade Areas."
    *Land Economics*, 39(1), 81-90.
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
_MIN_DISTANCE_KM: float = 0.01  # floor to avoid division by zero


# ---------------------------------------------------------------------------
# HuffModel
# ---------------------------------------------------------------------------

class HuffModel:
    """Huff Retail Gravity Model for store-choice probability estimation.

    Parameters
    ----------
    alpha : float, default 1.0
        Attractiveness exponent.  Higher values amplify differences in
        store attractiveness.
    lam : float, default 2.0
        Distance-decay exponent.  Higher values penalise travel distance
        more heavily.
    attractiveness : str or callable, default "square_footage"
        How to compute store attractiveness.

        * If a *string*, it is treated as a column name in the stores
          DataFrame (e.g. ``"square_footage"``).
        * If a *callable*, it receives a single-row ``pd.Series``
          (one store) and must return a non-negative float.
    distance_matrix : pd.DataFrame or None, default None
        Pre-computed origin x store distance matrix (km).  If ``None``,
        distances are computed from lat/lon columns using the haversine
        formula when ``predict()`` or ``fit()`` is called.
    min_distance : float, default 0.01
        Floor applied to distances (km) to prevent division-by-zero.

    Examples
    --------
    >>> model = HuffModel(alpha=1.0, lam=2.0, attractiveness="square_footage")
    >>> probs = model.predict(origins_df, stores_df)
    >>> model.fit(origins_df, stores_df, observed_shares)
    >>> calibrated_probs = model.predict(origins_df, stores_df)
    """

    def __init__(
        self,
        alpha: float = _DEFAULT_ALPHA,
        lam: float = _DEFAULT_LAMBDA,
        attractiveness: Union[str, Callable[[pd.Series], float]] = "square_footage",
        distance_matrix: Optional[pd.DataFrame] = None,
        min_distance: float = _MIN_DISTANCE_KM,
    ) -> None:
        self.alpha = alpha
        self.lam = lam
        self.attractiveness = attractiveness
        self.distance_matrix = distance_matrix
        self.min_distance = min_distance

        # Populated after fit()
        self._fitted: bool = False
        self._fit_result: Optional[object] = None

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
        alpha_bounds: tuple[float, float] = (0.01, 10.0),
        lam_bounds: tuple[float, float] = (0.01, 10.0),
        x0: Optional[tuple[float, float]] = None,
        verbose: bool = False,
    ) -> "HuffModel":
        """Calibrate alpha and lambda to observed visit shares via MLE.

        Parameters
        ----------
        origins : pd.DataFrame
            Origins indexed by ``origin_id`` with at least ``lat`` and ``lon``.
        stores : pd.DataFrame
            Stores indexed by ``store_id`` with at least ``lat``, ``lon``,
            and the attractiveness column.
        observed_shares : pd.DataFrame
            Origin x store matrix of *observed* visit shares.  Each row must
            sum to 1.0 (or close to it).  Index and columns must align with
            ``origins`` and ``stores``.
        method : str, default "L-BFGS-B"
            Optimiser method passed to ``scipy.optimize.minimize``.
        alpha_bounds : tuple, default (0.01, 10.0)
            Bounds for the alpha parameter.
        lam_bounds : tuple, default (0.01, 10.0)
            Bounds for the lambda parameter.
        x0 : tuple or None
            Starting point ``(alpha, lam)``.  Defaults to current values.
        verbose : bool, default False
            If True, log optimisation progress.

        Returns
        -------
        HuffModel
            ``self``, with updated ``alpha`` and ``lam``.
        """
        dist = self._resolve_distances(origins, stores)
        attract = self._compute_attractiveness(stores)

        # Align observed shares to the same index/columns as the distance matrix.
        obs = observed_shares.reindex(index=dist.index, columns=dist.columns)
        if obs.isna().any().any():
            raise ValueError(
                "observed_shares must cover all origin-store pairs present in "
                "the distance matrix.  Found NaN after alignment."
            )

        # Convert to numpy for speed.
        obs_np = obs.values.astype(np.float64)
        dist_np = dist.values.astype(np.float64)
        attract_np = attract.values.astype(np.float64)

        start = x0 if x0 is not None else (self.alpha, self.lam)

        def neg_log_likelihood(params: np.ndarray) -> float:
            a, l = params
            return self._neg_ll(a, l, attract_np, dist_np, obs_np)

        result = minimize(
            neg_log_likelihood,
            x0=start,
            method=method,
            bounds=[alpha_bounds, lam_bounds],
            options={"disp": verbose, "maxiter": 500},
        )

        if not result.success:
            logger.warning("Optimisation did not converge: %s", result.message)

        self.alpha, self.lam = float(result.x[0]), float(result.x[1])
        self._fitted = True
        self._fit_result = result

        logger.info(
            "Calibrated parameters: alpha=%.4f, lambda=%.4f (LL=%.4f)",
            self.alpha,
            self.lam,
            -result.fun,
        )
        return self

    def predict(
        self,
        origins: pd.DataFrame,
        stores: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute origin x store visit-probability matrix.

        Parameters
        ----------
        origins : pd.DataFrame
            Origins indexed by ``origin_id``.
        stores : pd.DataFrame
            Stores indexed by ``store_id``.

        Returns
        -------
        pd.DataFrame
            DataFrame of shape ``(n_origins, n_stores)`` where each cell is
            P(origin_i chooses store_j).  Rows sum to 1.
        """
        dist = self._resolve_distances(origins, stores)
        attract = self._compute_attractiveness(stores)

        probs = self._huff_probabilities(
            self.alpha, self.lam, attract.values, dist.values,
        )
        return pd.DataFrame(probs, index=dist.index, columns=dist.columns)

    def trade_area_shares(
        self,
        store_id: str,
        origins: pd.DataFrame,
        stores: pd.DataFrame,
    ) -> pd.Series:
        """Return per-origin probabilities for a single store.

        Parameters
        ----------
        store_id : str
            The target store whose trade-area shares to extract.
        origins : pd.DataFrame
            Origins indexed by ``origin_id``.
        stores : pd.DataFrame
            Stores indexed by ``store_id``.

        Returns
        -------
        pd.Series
            Series indexed by ``origin_id`` with P(origin_i -> store_id).

        Raises
        ------
        KeyError
            If ``store_id`` is not present in the stores DataFrame.
        """
        if store_id not in stores.index:
            raise KeyError(
                f"store_id '{store_id}' not found in stores DataFrame index."
            )
        probs = self.predict(origins, stores)
        return probs[store_id].rename(f"P(origin -> {store_id})")

    # ------------------------------------------------------------------
    # Diagnostics / introspection
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit()`` has been called successfully."""
        return self._fitted

    @property
    def params(self) -> dict[str, float]:
        """Current model parameters."""
        return {"alpha": self.alpha, "lam": self.lam}

    def summary(self) -> str:
        """Human-readable summary of the model state."""
        status = "fitted" if self._fitted else "not fitted"
        lines = [
            "HuffModel",
            f"  status          : {status}",
            f"  alpha           : {self.alpha:.4f}",
            f"  lambda          : {self.lam:.4f}",
            f"  attractiveness  : {self.attractiveness}",
            f"  min_distance_km : {self.min_distance}",
        ]
        if self._fitted and self._fit_result is not None:
            lines.append(f"  log-likelihood  : {-self._fit_result.fun:.4f}")
            lines.append(f"  converged       : {self._fit_result.success}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"HuffModel(alpha={self.alpha:.4f}, lam={self.lam:.4f}, "
            f"attractiveness={self.attractiveness!r}, fitted={self._fitted})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_distances(
        self, origins: pd.DataFrame, stores: pd.DataFrame
    ) -> pd.DataFrame:
        """Return an origin x store distance matrix, computing if necessary."""
        if self.distance_matrix is not None:
            return self.distance_matrix
        return build_distance_matrix(origins, stores)

    def _compute_attractiveness(self, stores: pd.DataFrame) -> pd.Series:
        """Derive a 1-D attractiveness vector from the stores DataFrame.

        Parameters
        ----------
        stores : pd.DataFrame
            Stores DataFrame indexed by ``store_id``.

        Returns
        -------
        pd.Series
            Non-negative attractiveness value per store, indexed by
            ``store_id``.

        Raises
        ------
        ValueError
            If any attractiveness value is negative.
        KeyError
            If the column specified by ``self.attractiveness`` does not exist.
        """
        if callable(self.attractiveness):
            attract = stores.apply(self.attractiveness, axis=1)
        elif isinstance(self.attractiveness, str):
            if self.attractiveness not in stores.columns:
                raise KeyError(
                    f"Attractiveness column '{self.attractiveness}' not found "
                    f"in stores DataFrame. Available columns: "
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
        # Replace exact zeros with a tiny value to avoid 0^alpha issues.
        attract = attract.clip(lower=1e-9)
        return attract

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
            2-D array of shape ``(n_origins, n_stores)`` in km.

        Returns
        -------
        np.ndarray
            Probability matrix where each row sums to 1.
        """
        # Floor distances to avoid div-by-zero.
        safe_dist = np.maximum(distances, _MIN_DISTANCE_KM)

        # Numerator: A_j^alpha * D_ij^(-lambda) for each (i, j).
        utility = (attractiveness[np.newaxis, :] ** alpha) * (safe_dist ** (-lam))

        # Denominator: sum across stores for each origin.
        row_sums = utility.sum(axis=1, keepdims=True)

        # Guard against rows where all utilities are zero (shouldn't happen
        # with the floor, but be defensive).
        row_sums = np.maximum(row_sums, 1e-30)

        return utility / row_sums

    @staticmethod
    def _neg_ll(
        alpha: float,
        lam: float,
        attractiveness: np.ndarray,
        distances: np.ndarray,
        observed: np.ndarray,
    ) -> float:
        """Negative log-likelihood of observed shares given parameters.

        Uses the multinomial/KL-divergence form:
            -LL = - SUM_i SUM_j  obs_ij * log(pred_ij)

        Parameters
        ----------
        alpha : float
        lam : float
        attractiveness : np.ndarray  (n_stores,)
        distances : np.ndarray       (n_origins, n_stores)
        observed : np.ndarray        (n_origins, n_stores) — rows sum to 1

        Returns
        -------
        float
            Scalar negative log-likelihood.
        """
        pred = HuffModel._huff_probabilities(alpha, lam, attractiveness, distances)
        # Clip predictions away from zero to keep log() finite.
        pred = np.clip(pred, 1e-15, 1.0)
        return -np.sum(observed * np.log(pred))
