"""
Fotheringham's Competing Destinations Model
============================================
An extension of the Huff gravity model that accounts for spatial clustering
(agglomeration) among competing stores.

The standard Huff model treats each store independently; it ignores the
fact that a dense cluster of stores may collectively draw more (or fewer)
consumers than the same stores would if they were spread apart.

Fotheringham's insight is to augment the Huff utility with a *competing
destinations* index (CD_j) that measures how many other stores are
accessible from store j's location:

    P(C_i -> S_j) = (A_j^alpha  *  D_ij^(-lambda)  *  CD_j^delta)
                     / SUM_k (A_k^alpha  *  D_ik^(-lambda)  *  CD_k^delta)

where:
    CD_j  = SUM_{k != j} (A_k / D_jk^beta)   for all stores k within a
            specified radius of store j

Parameters:
    alpha   -- attractiveness exponent (from Huff)
    lambda  -- distance-decay exponent (from Huff)
    delta   -- agglomeration/competition exponent
               > 0 : agglomeration benefit (clusters attract more)
               < 0 : competition penalty (clusters cannibalise)
               = 0 : reduces to the standard Huff model
    beta    -- inter-store distance-decay for the clustering index

References
----------
Fotheringham, A.S. (1983). "A New Set of Spatial-Interaction Models:
    The Theory of Competing Destinations."
    *Environment and Planning A*, 15(1), 15-36.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gravity.data.schema import build_distance_matrix
from gravity.core.huff import HuffModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DELTA: float = 0.0
_DEFAULT_BETA: float = 1.0
_DEFAULT_RADIUS_KM: float = 30.0
_MIN_DISTANCE_KM: float = 0.01


# ---------------------------------------------------------------------------
# CompetingDestinationsModel
# ---------------------------------------------------------------------------

class CompetingDestinationsModel:
    """Fotheringham's Competing Destinations extension of the Huff model.

    Wraps HuffModel behaviour and adds a per-store clustering index (CD)
    that modulates each store's utility based on how many competing stores
    are nearby.

    Parameters
    ----------
    alpha : float, default 1.0
        Attractiveness exponent (identical to HuffModel).
    lam : float, default 2.0
        Distance-decay exponent for origin-to-store distances.
    delta : float, default 0.0
        Agglomeration/competition exponent applied to the clustering
        index.  Positive values reward stores in dense clusters; negative
        values penalise them.  When delta is exactly 0 the model reduces
        to the standard Huff model.
    beta : float, default 1.0
        Distance-decay exponent used when computing inter-store
        accessibility in the clustering index.
    radius_km : float, default 30.0
        Maximum inter-store distance (km) considered when computing the
        clustering index.  Stores farther apart than this from store j
        contribute nothing to CD_j.
    attractiveness : str or callable, default "square_footage"
        Column name or callable that extracts a non-negative attractiveness
        scalar for each store (passed through to internal HuffModel logic).
    distance_matrix : pd.DataFrame or None, default None
        Pre-computed origin x store distance matrix (km).  If ``None``,
        distances are computed on the fly via haversine.
    store_distance_matrix : pd.DataFrame or None, default None
        Pre-computed store x store distance matrix (km).  If ``None``,
        it is computed from store lat/lon when the clustering index is
        first needed.
    min_distance : float, default 0.01
        Floor applied to all distances to prevent division by zero.

    Examples
    --------
    >>> model = CompetingDestinationsModel(alpha=1.0, lam=2.0, delta=0.5)
    >>> probs = model.predict(origins_df, stores_df)
    >>> cd_index = model.compute_clustering_index(stores_df)
    >>> model.fit(origins_df, stores_df, observed_shares)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        lam: float = 2.0,
        delta: float = _DEFAULT_DELTA,
        beta: float = _DEFAULT_BETA,
        radius_km: float = _DEFAULT_RADIUS_KM,
        attractiveness: Union[str, Callable[[pd.Series], float]] = "square_footage",
        distance_matrix: Optional[pd.DataFrame] = None,
        store_distance_matrix: Optional[pd.DataFrame] = None,
        min_distance: float = _MIN_DISTANCE_KM,
    ) -> None:
        self.alpha = alpha
        self.lam = lam
        self.delta = delta
        self.beta = beta
        self.radius_km = radius_km
        self.attractiveness = attractiveness
        self.distance_matrix = distance_matrix
        self.store_distance_matrix = store_distance_matrix
        self.min_distance = min_distance

        # Populated after fit()
        self._fitted: bool = False
        self._fit_result: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        origins: pd.DataFrame,
        stores: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute origin x store visit-probability matrix.

        Parameters
        ----------
        origins : pd.DataFrame
            Origins indexed by ``origin_id`` with at least ``lat`` and
            ``lon`` columns.
        stores : pd.DataFrame
            Stores indexed by ``store_id`` with at least ``lat``, ``lon``,
            and the attractiveness column.

        Returns
        -------
        pd.DataFrame
            DataFrame of shape ``(n_origins, n_stores)`` where cell (i, j)
            is the probability that a consumer at origin i visits store j.
            Each row sums to 1.
        """
        dist = self._resolve_origin_store_distances(origins, stores)
        attract = self._compute_attractiveness(stores)
        cd = self._compute_cd_array(stores, attract)

        probs = self._cd_probabilities(
            self.alpha, self.lam, self.delta,
            attract.values, dist.values, cd,
        )
        return pd.DataFrame(probs, index=dist.index, columns=dist.columns)

    def fit(
        self,
        origins: pd.DataFrame,
        stores: pd.DataFrame,
        observed_shares: pd.DataFrame,
        *,
        method: str = "L-BFGS-B",
        alpha_bounds: tuple[float, float] = (0.01, 10.0),
        lam_bounds: tuple[float, float] = (0.01, 10.0),
        delta_bounds: tuple[float, float] = (-5.0, 5.0),
        x0: Optional[tuple[float, float, float]] = None,
        verbose: bool = False,
    ) -> "CompetingDestinationsModel":
        """Calibrate alpha, lambda, and delta via maximum-likelihood estimation.

        The optimiser minimises the negative log-likelihood (equivalent to
        KL-divergence) between predicted and observed origin-store visit
        shares.

        Parameters
        ----------
        origins : pd.DataFrame
            Origins indexed by ``origin_id`` with at least ``lat`` and
            ``lon``.
        stores : pd.DataFrame
            Stores indexed by ``store_id`` with at least ``lat``, ``lon``,
            and the attractiveness column.
        observed_shares : pd.DataFrame
            Origin x store matrix of observed visit shares.  Each row must
            sum to 1.0 (or close).  Index and columns must align with
            ``origins`` and ``stores``.
        method : str, default "L-BFGS-B"
            Optimiser method forwarded to ``scipy.optimize.minimize``.
        alpha_bounds : tuple, default (0.01, 10.0)
            Bounds for alpha.
        lam_bounds : tuple, default (0.01, 10.0)
            Bounds for lambda.
        delta_bounds : tuple, default (-5.0, 5.0)
            Bounds for delta.  Negative values are permitted (competition
            penalty).
        x0 : tuple or None
            Starting point ``(alpha, lam, delta)``.  Defaults to the
            current parameter values.
        verbose : bool, default False
            If True, display optimiser progress.

        Returns
        -------
        CompetingDestinationsModel
            ``self``, with updated ``alpha``, ``lam``, and ``delta``.
        """
        dist = self._resolve_origin_store_distances(origins, stores)
        attract = self._compute_attractiveness(stores)
        cd = self._compute_cd_array(stores, attract)

        # Align observed shares to the distance matrix layout.
        obs = observed_shares.reindex(index=dist.index, columns=dist.columns)
        if obs.isna().any().any():
            raise ValueError(
                "observed_shares must cover all origin-store pairs present "
                "in the distance matrix.  Found NaN after alignment."
            )

        # Convert to numpy for speed.
        obs_np = obs.values.astype(np.float64)
        dist_np = dist.values.astype(np.float64)
        attract_np = attract.values.astype(np.float64)

        start = x0 if x0 is not None else (self.alpha, self.lam, self.delta)

        def neg_log_likelihood(params: np.ndarray) -> float:
            a, l, d = params
            return self._neg_ll(a, l, d, attract_np, dist_np, cd, obs_np)

        result = minimize(
            neg_log_likelihood,
            x0=start,
            method=method,
            bounds=[alpha_bounds, lam_bounds, delta_bounds],
            options={"disp": verbose, "maxiter": 500},
        )

        if not result.success:
            logger.warning("Optimisation did not converge: %s", result.message)

        self.alpha = float(result.x[0])
        self.lam = float(result.x[1])
        self.delta = float(result.x[2])
        self._fitted = True
        self._fit_result = result

        logger.info(
            "Calibrated parameters: alpha=%.4f, lambda=%.4f, delta=%.4f "
            "(LL=%.4f)",
            self.alpha, self.lam, self.delta, -result.fun,
        )
        return self

    def compute_clustering_index(
        self,
        stores: pd.DataFrame,
    ) -> pd.Series:
        """Compute the competing-destinations index for every store.

        For each store j the index is:

            CD_j = SUM_{k != j, D_jk <= radius} (A_k / D_jk^beta)

        Higher values indicate that store j is surrounded by many large,
        nearby competitors.

        Parameters
        ----------
        stores : pd.DataFrame
            Stores indexed by ``store_id`` with at least ``lat``, ``lon``,
            and the attractiveness column.

        Returns
        -------
        pd.Series
            Series indexed by ``store_id`` containing the CD value for
            each store.
        """
        attract = self._compute_attractiveness(stores)
        cd = self._compute_cd_array(stores, attract)
        return pd.Series(cd, index=stores.index, name="clustering_index")

    # ------------------------------------------------------------------
    # Convenience: delegate to a HuffModel when delta == 0
    # ------------------------------------------------------------------

    def as_huff(self) -> HuffModel:
        """Return an equivalent HuffModel (ignoring the clustering term).

        Useful for comparing the competing-destinations results against
        the baseline Huff probabilities.

        Returns
        -------
        HuffModel
            A new HuffModel instance initialised with this model's alpha,
            lambda, attractiveness, and distance matrix settings.
        """
        return HuffModel(
            alpha=self.alpha,
            lam=self.lam,
            attractiveness=self.attractiveness,
            distance_matrix=self.distance_matrix,
            min_distance=self.min_distance,
        )

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
        return {
            "alpha": self.alpha,
            "lam": self.lam,
            "delta": self.delta,
            "beta": self.beta,
            "radius_km": self.radius_km,
        }

    def summary(self) -> str:
        """Human-readable summary of the model state."""
        status = "fitted" if self._fitted else "not fitted"
        lines = [
            "CompetingDestinationsModel",
            f"  status          : {status}",
            f"  alpha           : {self.alpha:.4f}",
            f"  lambda          : {self.lam:.4f}",
            f"  delta           : {self.delta:.4f}",
            f"  beta            : {self.beta:.4f}",
            f"  radius_km       : {self.radius_km:.1f}",
            f"  attractiveness  : {self.attractiveness}",
            f"  min_distance_km : {self.min_distance}",
        ]
        if self._fitted and self._fit_result is not None:
            lines.append(f"  log-likelihood  : {-self._fit_result.fun:.4f}")
            lines.append(f"  converged       : {self._fit_result.success}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CompetingDestinationsModel(alpha={self.alpha:.4f}, "
            f"lam={self.lam:.4f}, delta={self.delta:.4f}, "
            f"beta={self.beta:.4f}, radius_km={self.radius_km:.1f}, "
            f"fitted={self._fitted})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_origin_store_distances(
        self, origins: pd.DataFrame, stores: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return an origin x store distance matrix, computing if needed."""
        if self.distance_matrix is not None:
            return self.distance_matrix
        return build_distance_matrix(origins, stores)

    def _resolve_store_store_distances(
        self, stores: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a store x store distance matrix, computing if needed.

        Uses ``build_distance_matrix`` with stores in both the origin and
        destination positions.
        """
        if self.store_distance_matrix is not None:
            return self.store_distance_matrix
        return build_distance_matrix(stores, stores)

    def _compute_attractiveness(self, stores: pd.DataFrame) -> pd.Series:
        """Derive a 1-D attractiveness vector from the stores DataFrame.

        Delegates to the same logic used by HuffModel: if
        ``self.attractiveness`` is a string it is used as a column name;
        if callable it is applied row-wise.

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

    def _compute_cd_array(
        self,
        stores: pd.DataFrame,
        attract: pd.Series,
    ) -> np.ndarray:
        """Compute the competing-destinations index as a numpy array.

        Parameters
        ----------
        stores : pd.DataFrame
            Stores DataFrame (needs lat/lon for distance computation).
        attract : pd.Series
            Pre-computed attractiveness vector aligned to ``stores.index``.

        Returns
        -------
        np.ndarray
            1-D array of length ``n_stores`` with CD_j values.
        """
        store_dist = self._resolve_store_store_distances(stores)
        store_dist_np = store_dist.values.astype(np.float64)
        attract_np = attract.values.astype(np.float64)

        n = len(attract_np)

        # Floor inter-store distances (the diagonal is zero -- we handle
        # the self-exclusion via the mask below).
        safe_dist = np.maximum(store_dist_np, self.min_distance)

        # Accessibility of each store k from location j: A_k / D_jk^beta
        accessibility = attract_np[np.newaxis, :] / (safe_dist ** self.beta)

        # Zero out the diagonal (store j does not compete with itself).
        np.fill_diagonal(accessibility, 0.0)

        # Apply the radius mask: set accessibility to 0 for stores beyond
        # the configured radius.
        if self.radius_km is not None and np.isfinite(self.radius_km):
            beyond_radius = store_dist_np > self.radius_km
            accessibility[beyond_radius] = 0.0

        # CD_j = sum across k for each j (row-wise sum).
        cd = accessibility.sum(axis=1)

        # Floor to a tiny positive value so that CD^delta is always finite.
        cd = np.maximum(cd, 1e-9)

        return cd

    @staticmethod
    def _cd_probabilities(
        alpha: float,
        lam: float,
        delta: float,
        attractiveness: np.ndarray,
        distances: np.ndarray,
        cd: np.ndarray,
    ) -> np.ndarray:
        """Core competing-destinations probability calculation.

        Parameters
        ----------
        alpha : float
            Attractiveness exponent.
        lam : float
            Distance-decay exponent for origin-to-store distances.
        delta : float
            Agglomeration/competition exponent on the clustering index.
        attractiveness : np.ndarray
            1-D array of length ``n_stores``.
        distances : np.ndarray
            2-D array of shape ``(n_origins, n_stores)`` in km.
        cd : np.ndarray
            1-D array of length ``n_stores`` -- the clustering index.

        Returns
        -------
        np.ndarray
            Probability matrix of shape ``(n_origins, n_stores)`` where
            each row sums to 1.
        """
        safe_dist = np.maximum(distances, _MIN_DISTANCE_KM)

        # Utility: A_j^alpha * D_ij^(-lambda) * CD_j^delta
        utility = (
            (attractiveness[np.newaxis, :] ** alpha)
            * (safe_dist ** (-lam))
            * (cd[np.newaxis, :] ** delta)
        )

        row_sums = utility.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-30)

        return utility / row_sums

    @staticmethod
    def _neg_ll(
        alpha: float,
        lam: float,
        delta: float,
        attractiveness: np.ndarray,
        distances: np.ndarray,
        cd: np.ndarray,
        observed: np.ndarray,
    ) -> float:
        """Negative log-likelihood of observed shares given parameters.

        Uses the multinomial form:
            -LL = - SUM_i SUM_j  obs_ij * log(pred_ij)

        Parameters
        ----------
        alpha : float
        lam : float
        delta : float
        attractiveness : np.ndarray  (n_stores,)
        distances : np.ndarray       (n_origins, n_stores)
        cd : np.ndarray              (n_stores,)
        observed : np.ndarray        (n_origins, n_stores) -- rows sum to 1

        Returns
        -------
        float
            Scalar negative log-likelihood.
        """
        pred = CompetingDestinationsModel._cd_probabilities(
            alpha, lam, delta, attractiveness, distances, cd,
        )
        pred = np.clip(pred, 1e-15, 1.0)
        return -np.sum(observed * np.log(pred))
