"""
Multiplicative Competitive Interaction (MCI) Model
===================================================
A log-linear extension of the gravity model that estimates the contribution
of multiple store attributes to consumer spatial choice via OLS regression
on log-centered data.

The MCI model (Nakanishi & Cooper, 1974) generalises the Huff model by
allowing *any number* of explanatory variables -- each with its own
empirically estimated exponent -- rather than fixing the functional form
to a single attractiveness measure and a distance-decay parameter.

Algorithm
---------
1. For each origin *i*, compute the geometric mean of each variable *k*
   across all stores *j*.
2. Log-center each observation:  ``lc(x_ijk) = log(x_ijk) - log(GM_ik)``
   where ``GM_ik`` is the geometric mean of variable *k* for origin *i*.
3. Apply the same transformation to the dependent variable (observed
   market shares or a uniform prior).
4. Estimate coefficients via OLS on the stacked, log-centered data.
5. Predict shares:  ``S_ij = exp(X_ij @ beta) / SUM_k exp(X_ik @ beta)``
   which satisfies the adding-up constraint (shares sum to 1 per origin).

The log-centering transformation converts the multiplicative share
equation into a linear regression that can be estimated by ordinary
least squares, while automatically satisfying the constraint that
predicted shares sum to 1 within each origin.

References
----------
Nakanishi, M. & Cooper, L.G. (1974). "Parameter Estimation for a
    Multiplicative Competitive Interaction Model -- Least Squares Approach."
    *Journal of Marketing Research*, 11(3), 303-311.

Cooper, L.G. & Nakanishi, M. (1988). *Market-Share Analysis*.
    Kluwer Academic Publishers.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import lstsq

from gravity.data.schema import build_distance_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-10  # floor for zeros before taking log
_MIN_DISTANCE_KM: float = 0.01  # floor to avoid log(0) for distances


# ---------------------------------------------------------------------------
# MCIModel
# ---------------------------------------------------------------------------

class MCIModel:
    """Multiplicative Competitive Interaction model for store-choice shares.

    The MCI model estimates market-share probabilities as a function of
    multiple store and distance attributes, with exponents (elasticities)
    determined empirically via OLS on log-centered data.

    Parameters
    ----------
    variables : list[str] or None
        Store attribute columns to include as explanatory variables.
        If ``None``, defaults to ``["square_footage"]``.
    include_distance : bool, default True
        Whether to include the origin-store distance as an additional
        explanatory variable.  When included, the fitted coefficient is
        expected to be negative (distance decay).

    Examples
    --------
    >>> model = MCIModel(variables=["square_footage", "avg_rating"])
    >>> model.fit(origins_df, stores_df, distance_matrix)
    >>> probs = model.predict(origins_df, stores_df, distance_matrix)
    >>> model.variable_importance()
    """

    def __init__(
        self,
        variables: Optional[list[str]] = None,
        include_distance: bool = True,
    ) -> None:
        self.variables = list(variables) if variables is not None else ["square_footage"]
        self.include_distance = include_distance

        # All variable names in canonical order (store attrs first, then distance).
        self._all_variables: list[str] = list(self.variables)
        if self.include_distance:
            self._all_variables.append("distance")

        # Populated after fit().
        self._fitted: bool = False
        self._coefficients: Optional[np.ndarray] = None
        self._std_errors: Optional[np.ndarray] = None
        self._t_stats: Optional[np.ndarray] = None
        self._r_squared: float = 0.0
        self._adj_r_squared: float = 0.0
        self._f_statistic: float = 0.0
        self._n_obs: int = 0
        self._n_params: int = 0
        self._distance_matrix: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        distance_matrix: pd.DataFrame,
        observed_shares: Optional[pd.DataFrame] = None,
    ) -> "MCIModel":
        """Fit the MCI model via OLS on log-centered data.

        Parameters
        ----------
        origins_df : pd.DataFrame
            Origins indexed by ``origin_id`` with demographic columns.
        stores_df : pd.DataFrame
            Stores indexed by ``store_id`` with attribute columns.
        distance_matrix : pd.DataFrame
            Origin x store distance matrix (km).  Index must align with
            ``origins_df``, columns with ``stores_df``.
        observed_shares : pd.DataFrame or None
            Origin x store matrix of observed visit shares (rows sum to 1).
            If ``None``, uniform shares ``1/n_stores`` are used as the
            dependent variable.  This still allows estimation of relative
            variable importance from the attribute variation alone.

        Returns
        -------
        MCIModel
            ``self``, with fitted coefficients accessible via
            ``coefficients_summary`` and ``variable_importance()``.

        Raises
        ------
        ValueError
            If required columns are missing from ``stores_df`` or if
            data shapes are incompatible.
        """
        self._validate_inputs(stores_df)
        self._distance_matrix = distance_matrix

        n_origins = len(origins_df)
        n_stores = len(stores_df)

        if n_origins == 0 or n_stores == 0:
            raise ValueError(
                f"Cannot fit with empty data: {n_origins} origins, "
                f"{n_stores} stores."
            )

        # ----- Build the raw variable matrices (n_origins x n_stores) -----
        raw_vars = self._build_raw_variables(stores_df, distance_matrix)
        # raw_vars: dict[str, np.ndarray] each of shape (n_origins, n_stores)

        # ----- Dependent variable -----
        if observed_shares is not None:
            shares = observed_shares.reindex(
                index=distance_matrix.index, columns=distance_matrix.columns
            ).values.astype(np.float64)
            if np.isnan(shares).any():
                raise ValueError(
                    "observed_shares contains NaN after alignment with the "
                    "distance matrix index/columns."
                )
        else:
            # Uniform shares as default.
            shares = np.full((n_origins, n_stores), 1.0 / n_stores)

        # Floor shares away from zero for log.
        shares = np.maximum(shares, _EPSILON)

        # ----- Log-centering transformation -----
        # For each origin row, compute geometric mean across stores, then
        # transform: lc(x) = log(x) - mean(log(x)) across stores.
        lc_y = self._log_center(shares)  # (n_origins, n_stores)

        lc_X_components = []
        for var_name in self._all_variables:
            lc_var = self._log_center(raw_vars[var_name])  # (n_origins, n_stores)
            lc_X_components.append(lc_var)

        # ----- Stack into long format for OLS -----
        # Each origin-store pair becomes one observation.
        n_obs = n_origins * n_stores
        y = lc_y.ravel()  # (n_obs,)
        X = np.column_stack(
            [comp.ravel() for comp in lc_X_components]
        )  # (n_obs, n_vars)

        # ----- OLS via scipy.linalg.lstsq -----
        n_params = X.shape[1]
        result = lstsq(X, y)
        beta = result[0]  # coefficients

        # ----- Compute diagnostics -----
        y_hat = X @ beta
        residuals = y - y_hat
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 1.0 - ss_res / max(ss_tot, _EPSILON)
        dof_res = max(n_obs - n_params, 1)
        dof_tot = max(n_obs - 1, 1)
        adj_r_squared = 1.0 - (ss_res / dof_res) / max(ss_tot / dof_tot, _EPSILON)

        # F-statistic: (SS_reg / k) / (SS_res / (n - k))
        ss_reg = ss_tot - ss_res
        f_stat = (ss_reg / max(n_params, 1)) / max(ss_res / dof_res, _EPSILON)

        # Standard errors of coefficients.
        mse = ss_res / dof_res
        # Variance-covariance: mse * (X'X)^{-1}
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            var_beta = mse * np.diag(XtX_inv)
            std_errors = np.sqrt(np.maximum(var_beta, 0.0))
        except np.linalg.LinAlgError:
            logger.warning(
                "X'X is singular; standard errors cannot be computed. "
                "This may indicate collinear variables."
            )
            std_errors = np.full(n_params, np.nan)

        t_stats = np.where(
            std_errors > _EPSILON,
            beta / std_errors,
            np.full(n_params, np.nan),
        )

        # ----- Store results -----
        self._coefficients = beta
        self._std_errors = std_errors
        self._t_stats = t_stats
        self._r_squared = float(r_squared)
        self._adj_r_squared = float(adj_r_squared)
        self._f_statistic = float(f_stat)
        self._n_obs = n_obs
        self._n_params = n_params
        self._fitted = True

        logger.info(
            "MCI model fitted: R2=%.4f, Adj-R2=%.4f, F=%.2f, %d observations, "
            "%d parameters.",
            self._r_squared,
            self._adj_r_squared,
            self._f_statistic,
            self._n_obs,
            self._n_params,
        )
        for i, name in enumerate(self._all_variables):
            logger.info(
                "  %s: beta=%.4f, SE=%.4f, t=%.2f",
                name,
                beta[i],
                std_errors[i],
                t_stats[i],
            )

        return self

    def predict(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        distance_matrix: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Predict origin x store visit probabilities.

        Computes shares from the fitted coefficients using the
        multiplicative model:

            S_ij = exp(X_ij @ beta) / SUM_k exp(X_ik @ beta)

        Parameters
        ----------
        origins_df : pd.DataFrame
            Origins indexed by ``origin_id``.
        stores_df : pd.DataFrame
            Stores indexed by ``store_id``.
        distance_matrix : pd.DataFrame or None
            Origin x store distance matrix (km).  If ``None``, uses the
            distance matrix from ``fit()`` or computes one from lat/lon.

        Returns
        -------
        pd.DataFrame
            Probability matrix of shape ``(n_origins, n_stores)`` where
            each row sums to 1.  Index and columns match the distance
            matrix.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted or self._coefficients is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict()."
            )

        dist = self._resolve_distance_matrix(
            origins_df, stores_df, distance_matrix
        )
        raw_vars = self._build_raw_variables(stores_df, dist)

        # Compute log of each variable (no centering needed for prediction;
        # the exp/sum-exp formulation automatically normalises).
        n_origins, n_stores = dist.shape
        log_utility = np.zeros((n_origins, n_stores))

        for idx, var_name in enumerate(self._all_variables):
            vals = raw_vars[var_name]
            safe_vals = np.maximum(vals, _EPSILON)
            log_utility += self._coefficients[idx] * np.log(safe_vals)

        # Convert to probabilities via softmax per origin row.
        # Shift for numerical stability.
        log_utility -= log_utility.max(axis=1, keepdims=True)
        exp_utility = np.exp(log_utility)
        row_sums = exp_utility.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, _EPSILON)
        probs = exp_utility / row_sums

        return pd.DataFrame(probs, index=dist.index, columns=dist.columns)

    @property
    def coefficients_summary(self) -> dict:
        """Return fitted coefficients with names, values, standard errors, and t-statistics.

        Returns
        -------
        dict
            Keys: ``"variables"``, ``"coefficients"``, ``"std_errors"``,
            ``"t_stats"``, ``"r_squared"``, ``"adj_r_squared"``,
            ``"f_statistic"``, ``"n_obs"``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before accessing "
                "coefficients_summary."
            )

        return {
            "variables": list(self._all_variables),
            "coefficients": self._coefficients.tolist(),
            "std_errors": self._std_errors.tolist(),
            "t_stats": self._t_stats.tolist(),
            "r_squared": self._r_squared,
            "adj_r_squared": self._adj_r_squared,
            "f_statistic": self._f_statistic,
            "n_obs": self._n_obs,
        }

    def variable_importance(self) -> pd.DataFrame:
        """Rank variables by the absolute value of their t-statistics.

        The t-statistic captures both the magnitude and precision of
        each coefficient, making it a more informative measure of
        variable importance than the raw coefficient alone.

        Returns
        -------
        pd.DataFrame
            Columns: ``variable``, ``coefficient``, ``std_error``,
            ``t_statistic``, ``abs_t_statistic``.
            Sorted by ``abs_t_statistic`` descending.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before "
                "variable_importance()."
            )

        rows = []
        for i, var_name in enumerate(self._all_variables):
            rows.append({
                "variable": var_name,
                "coefficient": self._coefficients[i],
                "std_error": self._std_errors[i],
                "t_statistic": self._t_stats[i],
                "abs_t_statistic": abs(self._t_stats[i]),
            })

        df = pd.DataFrame(rows).sort_values(
            "abs_t_statistic", ascending=False
        ).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Diagnostics / introspection
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit()`` has been called successfully."""
        return self._fitted

    def summary(self) -> str:
        """Human-readable summary of the model state."""
        status = "fitted" if self._fitted else "not fitted"
        lines = [
            "MCIModel (Nakanishi & Cooper, 1974)",
            f"  status         : {status}",
            f"  variables      : {self.variables}",
            f"  include_distance: {self.include_distance}",
        ]
        if self._fitted:
            lines.append(f"  R-squared      : {self._r_squared:.4f}")
            lines.append(f"  Adj R-squared  : {self._adj_r_squared:.4f}")
            lines.append(f"  F-statistic    : {self._f_statistic:.2f}")
            lines.append(f"  n_obs          : {self._n_obs}")
            lines.append("")
            lines.append("  Coefficients:")
            for i, name in enumerate(self._all_variables):
                lines.append(
                    f"    {name:20s}  beta={self._coefficients[i]:+.4f}  "
                    f"SE={self._std_errors[i]:.4f}  "
                    f"t={self._t_stats[i]:+.2f}"
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MCIModel(variables={self.variables}, "
            f"include_distance={self.include_distance}, "
            f"fitted={self._fitted})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, stores_df: pd.DataFrame) -> None:
        """Check that all required variable columns exist in the stores DataFrame.

        Parameters
        ----------
        stores_df : pd.DataFrame
            Stores DataFrame to validate.

        Raises
        ------
        ValueError
            If any variable column is missing from ``stores_df``.
        """
        missing = [v for v in self.variables if v not in stores_df.columns]
        if missing:
            raise ValueError(
                f"Store attribute columns not found in stores_df: {missing}. "
                f"Available columns: {list(stores_df.columns)}"
            )

    def _build_raw_variables(
        self,
        stores_df: pd.DataFrame,
        distance_matrix: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Build the raw variable matrices (n_origins x n_stores).

        Each store attribute is broadcast to all origins (same value per
        column), while distance varies by origin-store pair.

        Parameters
        ----------
        stores_df : pd.DataFrame
            Stores DataFrame with attribute columns.
        distance_matrix : pd.DataFrame
            Origin x store distance matrix (km).

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from variable name to (n_origins, n_stores) array.
        """
        n_origins = distance_matrix.shape[0]
        raw_vars: dict[str, np.ndarray] = {}

        for var_name in self.variables:
            # Store attribute: same value for all origins, varies by store.
            store_vals = stores_df.reindex(distance_matrix.columns)[var_name].values.astype(np.float64)
            # Broadcast to (n_origins, n_stores).
            raw_vars[var_name] = np.tile(store_vals, (n_origins, 1))

        if self.include_distance:
            dist_vals = distance_matrix.values.astype(np.float64)
            # Floor distances to avoid log(0).
            dist_vals = np.maximum(dist_vals, _MIN_DISTANCE_KM)
            raw_vars["distance"] = dist_vals

        return raw_vars

    @staticmethod
    def _log_center(matrix: np.ndarray) -> np.ndarray:
        """Apply the log-centering transformation row-wise.

        For each row *i*, computes:
            ``lc(x_ij) = log(x_ij) - mean_j(log(x_ij))``

        This is equivalent to dividing by the geometric mean and taking
        the log:  ``log(x_ij / GM_i)``.

        Parameters
        ----------
        matrix : np.ndarray
            2-D array of shape ``(n_origins, n_stores)`` with positive
            values (zeros should already be floored to epsilon).

        Returns
        -------
        np.ndarray
            Log-centered matrix of the same shape.
        """
        safe = np.maximum(matrix, _EPSILON)
        log_vals = np.log(safe)
        # Geometric-mean centering: subtract row-wise mean of log values.
        row_means = log_vals.mean(axis=1, keepdims=True)
        return log_vals - row_means

    def _resolve_distance_matrix(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        distance_matrix: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Return a distance matrix, using stored, provided, or computed.

        Parameters
        ----------
        origins_df : pd.DataFrame
        stores_df : pd.DataFrame
        distance_matrix : pd.DataFrame or None

        Returns
        -------
        pd.DataFrame
            Origin x store distance matrix.
        """
        if distance_matrix is not None:
            return distance_matrix
        if self._distance_matrix is not None:
            return self._distance_matrix
        logger.info("No distance matrix provided; computing from lat/lon.")
        return build_distance_matrix(origins_df, stores_df)
