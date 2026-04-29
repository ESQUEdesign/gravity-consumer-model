"""
Spatial Econometrics Models
===========================
Spatial lag and spatial error regression models for accounting for spatial
autocorrelation in consumer behaviour data (e.g. market shares, visit
rates, spending patterns).

Two classical specifications are supported:

    **Spatial Lag (SAR)**
        y = rho * W @ y + X @ beta + epsilon

        Neighbouring outcomes directly affect the local outcome.  In a
        retail context this captures demand spillovers: a block group's
        market share for a store is partly explained by the shares of
        adjacent block groups (word of mouth, social influence, shared
        amenity access, etc.).

    **Spatial Error (SEM)**
        y = X @ beta + u
        u = lambda_ * W @ u + epsilon

        Unobserved spatial confounders are correlated across neighbours.
        The disturbance is autoregressive, so while the response itself
        does not directly depend on neighbouring outcomes, the errors do.
        This corrects for omitted spatially-varying factors (walkability,
        parking quality, local competition density) without introducing a
        spatial lag in the outcome.

The module tries to import ``libpysal`` and ``spreg`` for production-grade
estimation.  When those packages are not available, a pure
numpy/scipy fallback is used that relies on generalised method of moments
(GMM) for the lag model and iteratively re-weighted least squares for the
error model.

References
----------
Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
LeSage, J. & Pace, R.K. (2009). *Introduction to Spatial Econometrics*.
    CRC Press.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar, minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import libpysal
    from libpysal.weights import KNN as _KNN, DistanceBand as _DistanceBand
    _HAS_PYSAL = True
except ImportError:
    _HAS_PYSAL = False

try:
    import spreg
    _HAS_SPREG = True
except ImportError:
    _HAS_SPREG = False


# ---------------------------------------------------------------------------
# Helpers: spatial weights
# ---------------------------------------------------------------------------

def _knn_weights_matrix(coords: np.ndarray, k: int = 5) -> np.ndarray:
    """Build a row-standardised KNN spatial weights matrix.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n, 2) with [lat, lon] rows.
    k : int
        Number of nearest neighbours.

    Returns
    -------
    np.ndarray
        Dense (n, n) row-standardised weights matrix.
    """
    tree = cKDTree(coords)
    n = len(coords)
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        _, idx = tree.query(coords[i], k=k + 1)
        neighbours = [j for j in idx if j != i][:k]
        for j in neighbours:
            W[i, j] = 1.0
    # Row-standardise.
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums
    return W


def _distance_band_weights_matrix(
    coords: np.ndarray, threshold: float
) -> np.ndarray:
    """Build a row-standardised distance-band weights matrix.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n, 2) with [lat, lon] rows.
    threshold : float
        Distance threshold (in the same units as coords -- degrees or km).

    Returns
    -------
    np.ndarray
        Dense (n, n) row-standardised weights matrix.
    """
    tree = cKDTree(coords)
    n = len(coords)
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        neighbours = tree.query_ball_point(coords[i], r=threshold)
        for j in neighbours:
            if j != i:
                W[i, j] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums
    return W


def _build_weights(
    coords: np.ndarray,
    method: str = "knn",
    k: int = 5,
    threshold: float = 0.05,
) -> np.ndarray:
    """Dispatcher for spatial weight construction.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array (n, 2).
    method : str
        ``"knn"`` or ``"distance_band"``.
    k : int
        Number of nearest neighbours (for KNN).
    threshold : float
        Distance threshold (for distance band).

    Returns
    -------
    np.ndarray
        Row-standardised (n, n) weights matrix.
    """
    if method == "knn":
        return _knn_weights_matrix(coords, k=k)
    elif method == "distance_band":
        return _distance_band_weights_matrix(coords, threshold=threshold)
    else:
        raise ValueError(f"Unknown weights method: {method!r}. Use 'knn' or 'distance_band'.")


# ---------------------------------------------------------------------------
# SpatialEconModel
# ---------------------------------------------------------------------------

class SpatialEconModel:
    """Spatial lag and spatial error regression models.

    Accounts for spatial autocorrelation in consumer-behaviour outcome
    variables (market shares, visit rates, spending) by explicitly
    modelling the dependence structure among geographically proximate
    observations.

    Parameters
    ----------
    weights_method : str, default ``"knn"``
        How to construct the spatial weights matrix from coordinates.
        ``"knn"`` uses *k*-nearest neighbours; ``"distance_band"``
        connects all pairs within a distance threshold.
    k : int, default 5
        Number of nearest neighbours when ``weights_method="knn"``.
    distance_threshold : float, default 0.05
        Threshold when ``weights_method="distance_band"`` (units match
        coordinate units -- degrees for lat/lon, km if projected).
    add_intercept : bool, default True
        Whether to prepend a column of ones to *X* before estimation.

    Attributes
    ----------
    beta_ : np.ndarray or None
        Estimated regression coefficients (populated after ``fit``).
    rho_ : float or None
        Spatial lag autoregressive parameter (spatial lag model only).
    lambda_ : float or None
        Spatial error autoregressive parameter (spatial error model only).
    W_ : np.ndarray or None
        Row-standardised spatial weights matrix.
    residuals_ : np.ndarray or None
        Residuals from the fitted model.
    model_type_ : str or None
        ``"lag"`` or ``"error"`` -- the specification used in the last
        ``fit()`` call.
    sigma2_ : float or None
        Estimated error variance.
    log_likelihood_ : float or None
        Log-likelihood at the fitted parameter values.
    n_ : int or None
        Number of observations.
    k_ : int or None
        Number of regressors (including intercept if added).

    Examples
    --------
    >>> model = SpatialEconModel(weights_method="knn", k=6)
    >>> model.fit(y, X, coords, model_type="lag")
    >>> print(model.summary())
    >>> y_hat = model.predict(X_new, coords_new)
    """

    def __init__(
        self,
        weights_method: str = "knn",
        k: int = 5,
        distance_threshold: float = 0.05,
        add_intercept: bool = True,
    ) -> None:
        self.weights_method = weights_method
        self.k = k
        self.distance_threshold = distance_threshold
        self.add_intercept = add_intercept

        # Populated after fit()
        self.beta_: Optional[np.ndarray] = None
        self.rho_: Optional[float] = None
        self.lambda_: Optional[float] = None
        self.W_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.model_type_: Optional[str] = None
        self.sigma2_: Optional[float] = None
        self.log_likelihood_: Optional[float] = None
        self.n_: Optional[int] = None
        self.k_: Optional[int] = None

        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._coords_train: Optional[np.ndarray] = None
        self._feature_names: Optional[list[str]] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Union[np.ndarray, pd.DataFrame],
        coords: np.ndarray,
        model_type: str = "lag",
        *,
        weights_matrix: Optional[np.ndarray] = None,
    ) -> "SpatialEconModel":
        """Fit a spatial lag or spatial error model.

        Parameters
        ----------
        y : array-like of shape (n,)
            Dependent variable (e.g. market shares, visit counts).
        X : array-like of shape (n, p)
            Feature matrix (regressors).  If a DataFrame, column names
            are stored for pretty-printing in ``summary()``.
        coords : np.ndarray of shape (n, 2)
            Spatial coordinates ``[lat, lon]`` used to build the spatial
            weights matrix (unless *weights_matrix* is given directly).
        model_type : str, default ``"lag"``
            ``"lag"`` for the spatial autoregressive (SAR) model:
                ``y = rho * W @ y + X @ beta + epsilon``
            ``"error"`` for the spatial error model (SEM):
                ``y = X @ beta + u;  u = lambda * W @ u + epsilon``
        weights_matrix : np.ndarray or None
            Pre-computed row-standardised spatial weights.  When ``None``,
            the matrix is built from *coords* using the configured method.

        Returns
        -------
        SpatialEconModel
            ``self``, fitted.

        Raises
        ------
        ValueError
            If *model_type* is not ``"lag"`` or ``"error"``.
        """
        if model_type not in ("lag", "error"):
            raise ValueError(
                f"model_type must be 'lag' or 'error', got {model_type!r}"
            )

        # Convert inputs.
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self._feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        coords_arr = np.asarray(coords, dtype=np.float64)

        n = len(y_arr)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        # Add intercept.
        if self.add_intercept:
            X_arr = np.column_stack([np.ones(n), X_arr])
            self._feature_names = ["intercept"] + self._feature_names

        # Build spatial weights.
        if weights_matrix is not None:
            W = np.asarray(weights_matrix, dtype=np.float64)
        else:
            W = _build_weights(
                coords_arr,
                method=self.weights_method,
                k=self.k,
                threshold=self.distance_threshold,
            )

        self.W_ = W
        self.n_ = n
        self.k_ = X_arr.shape[1]
        self.model_type_ = model_type
        self._X_train = X_arr
        self._y_train = y_arr
        self._coords_train = coords_arr

        # Try spreg first, then fall back to pure-numpy.
        if _HAS_SPREG and _HAS_PYSAL:
            self._fit_spreg(y_arr, X_arr, W, model_type)
        else:
            if model_type == "lag":
                self._fit_lag_numpy(y_arr, X_arr, W)
            else:
                self._fit_error_numpy(y_arr, X_arr, W)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Public: predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X_new: Union[np.ndarray, pd.DataFrame],
        coords_new: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict the dependent variable for new observations.

        For the **spatial lag** model the reduced-form prediction is:
            y_hat = (I - rho * W_new)^{-1} @ X_new @ beta

        When *coords_new* is ``None`` or building a cross-weight matrix
        is infeasible, the naive prediction is used:
            y_hat = X_new @ beta

        For the **spatial error** model the mean prediction is simply:
            y_hat = X_new @ beta

        (because the spatial-error structure only affects residuals,
        not the conditional mean).

        Parameters
        ----------
        X_new : array-like of shape (m, p)
            New feature matrix (same columns as training *X*).
        coords_new : np.ndarray of shape (m, 2) or None
            Coordinates for the new observations.  Required for
            spatial-lag predictions that account for the W structure.

        Returns
        -------
        np.ndarray of shape (m,)
            Predicted values.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.  Call fit() first.")

        if isinstance(X_new, pd.DataFrame):
            X_arr = X_new.values.astype(np.float64)
        else:
            X_arr = np.asarray(X_new, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if self.add_intercept:
            X_arr = np.column_stack([np.ones(len(X_arr)), X_arr])

        Xb = X_arr @ self.beta_

        if self.model_type_ == "error":
            # Spatial error model: E[y|X] = X @ beta.
            return Xb

        # Spatial lag model: need W_new for reduced-form prediction.
        if coords_new is not None and self.rho_ is not None:
            coords_new = np.asarray(coords_new, dtype=np.float64)
            m = len(coords_new)
            W_new = _build_weights(
                coords_new,
                method=self.weights_method,
                k=min(self.k, m - 1),
                threshold=self.distance_threshold,
            )
            # y = (I - rho * W)^{-1} X beta
            A = np.eye(m) - self.rho_ * W_new
            try:
                y_hat = np.linalg.solve(A, Xb)
            except np.linalg.LinAlgError:
                logger.warning(
                    "Singular matrix in spatial lag prediction; "
                    "returning naive Xb prediction."
                )
                y_hat = Xb
            return y_hat

        # Fallback: naive prediction without spatial structure.
        return Xb

    # ------------------------------------------------------------------
    # Public: Moran's I
    # ------------------------------------------------------------------

    @staticmethod
    def moran_i(
        residuals: np.ndarray,
        W: np.ndarray,
    ) -> dict:
        """Compute Moran's I statistic for spatial autocorrelation.

        Parameters
        ----------
        residuals : np.ndarray of shape (n,)
            Residual vector (or any spatially-referenced variable).
        W : np.ndarray of shape (n, n)
            Row-standardised spatial weights matrix.

        Returns
        -------
        dict
            ``"I"`` -- Moran's I statistic.
            ``"expected"`` -- expected value under the null (-1/(n-1)).
            ``"variance"`` -- variance under normality assumption.
            ``"z_score"`` -- standardised z-score.
            ``"p_value"`` -- two-tailed p-value from the normal
            approximation.
        """
        from scipy.stats import norm

        e = np.asarray(residuals, dtype=np.float64).ravel()
        n = len(e)
        e_demean = e - e.mean()

        S0 = W.sum()
        numerator = n * float(e_demean @ W @ e_demean)
        denominator = S0 * float(e_demean @ e_demean)

        I_val = numerator / denominator if denominator != 0 else 0.0
        expected = -1.0 / (n - 1)

        # Variance under normality (Cliff & Ord, 1981).
        S1 = 0.5 * np.sum((W + W.T) ** 2)
        S2 = np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2)
        D = float(np.sum(e_demean ** 4)) / (float(np.sum(e_demean ** 2)) ** 2) * n

        A_term = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
        B_term = D * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
        C_term = (n - 1) * (n - 2) * (n - 3) * S0 ** 2

        variance = (A_term - B_term) / C_term - expected ** 2
        variance = max(variance, 1e-15)

        z = (I_val - expected) / np.sqrt(variance)
        p_value = 2.0 * (1.0 - norm.cdf(abs(z)))

        return {
            "I": float(I_val),
            "expected": float(expected),
            "variance": float(variance),
            "z_score": float(z),
            "p_value": float(p_value),
        }

    # ------------------------------------------------------------------
    # Public: Lagrange Multiplier tests
    # ------------------------------------------------------------------

    def lagrange_multiplier_tests(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Union[np.ndarray, pd.DataFrame],
        W: Optional[np.ndarray] = None,
    ) -> dict:
        """Lagrange Multiplier tests for spatial lag vs. error specification.

        Runs OLS first, then computes the LM-lag, LM-error, and robust
        variants to guide model selection.

        Parameters
        ----------
        y : array-like of shape (n,)
            Dependent variable.
        X : array-like of shape (n, p)
            Feature matrix.
        W : np.ndarray or None
            Spatial weights matrix.  If ``None``, uses the matrix from
            the last ``fit()`` call.

        Returns
        -------
        dict
            ``"LM_lag"`` -- LM statistic for spatial lag.
            ``"LM_lag_pvalue"`` -- p-value.
            ``"LM_error"`` -- LM statistic for spatial error.
            ``"LM_error_pvalue"`` -- p-value.
            ``"robust_LM_lag"`` -- Robust LM for lag (controlling for error).
            ``"robust_LM_lag_pvalue"`` -- p-value.
            ``"robust_LM_error"`` -- Robust LM for error (controlling for lag).
            ``"robust_LM_error_pvalue"`` -- p-value.
        """
        from scipy.stats import chi2

        y_arr = np.asarray(y, dtype=np.float64).ravel()
        X_arr = np.asarray(X if not isinstance(X, pd.DataFrame) else X.values,
                           dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if self.add_intercept:
            X_arr = np.column_stack([np.ones(len(y_arr)), X_arr])

        if W is None:
            W = self.W_
        if W is None:
            raise ValueError("No spatial weights matrix available. Provide W or call fit() first.")

        n = len(y_arr)

        # OLS estimates.
        beta_ols = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
        e = y_arr - X_arr @ beta_ols
        sigma2 = float(e @ e) / n

        We = W @ e
        Wy = W @ y_arr

        # Trace terms.
        WpW = W.T + W
        WW = W @ W
        T = np.trace(WpW @ W)

        # LM-error (Anselin 1988): (e'We / sigma2)^2 / tr(W'W + W^2)
        eWe = float(e @ We)
        T_err = np.trace(W.T @ W + WW)
        LM_error = (eWe / sigma2) ** 2 / T_err
        LM_error_p = 1.0 - chi2.cdf(float(LM_error), df=1)

        # LM-lag (Anselin 1988): (e'Wy / sigma2)^2 / (nJ)
        # where J = [1/(n*sigma2)] * [(WXb)'M(WXb) + T * sigma2]
        Xb = X_arr @ beta_ols
        WXb = W @ Xb
        M = np.eye(n) - X_arr @ np.linalg.lstsq(X_arr, np.eye(n), rcond=None)[0]
        MWXb = M @ WXb
        J = (float(MWXb @ MWXb) + T * sigma2) / (n * sigma2)

        eWy = float(e @ Wy)
        LM_lag = (eWy / sigma2) ** 2 / (n * J) if J != 0 else 0.0
        LM_lag_p = 1.0 - chi2.cdf(float(LM_lag), df=1)

        # Robust LM tests (Anselin et al. 1996).
        # Robust LM-error = [(e'We/sigma2) - T_err^{-1} * T * (e'Wy/sigma2)]^2
        #                    / [T_err - T^2 / (n*J)]
        numer_robust_err = (eWe / sigma2) - (T / (n * J) if J != 0 else 0) * (eWy / sigma2)
        denom_robust_err = T_err - T ** 2 / (n * J) if J != 0 else T_err
        robust_LM_error = numer_robust_err ** 2 / denom_robust_err if denom_robust_err != 0 else 0.0
        robust_LM_error_p = 1.0 - chi2.cdf(float(robust_LM_error), df=1)

        # Robust LM-lag = [(e'Wy/sigma2) - (e'We/sigma2)]^2 / [n*J - T_err]
        numer_robust_lag = (eWy / sigma2) - (eWe / sigma2)
        denom_robust_lag = n * J - T_err
        robust_LM_lag = numer_robust_lag ** 2 / denom_robust_lag if denom_robust_lag != 0 else 0.0
        robust_LM_lag_p = 1.0 - chi2.cdf(float(robust_LM_lag), df=1)

        return {
            "LM_lag": float(LM_lag),
            "LM_lag_pvalue": float(LM_lag_p),
            "LM_error": float(LM_error),
            "LM_error_pvalue": float(LM_error_p),
            "robust_LM_lag": float(robust_LM_lag),
            "robust_LM_lag_pvalue": float(robust_LM_lag_p),
            "robust_LM_error": float(robust_LM_error),
            "robust_LM_error_pvalue": float(robust_LM_error_p),
        }

    # ------------------------------------------------------------------
    # Public: summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary of the fitted model.

        Returns
        -------
        str
            Multi-line string with model specification, coefficient
            estimates, spatial parameters, diagnostics (Moran's I on
            residuals), and fit statistics.
        """
        if not self._fitted:
            return "SpatialEconModel: not fitted."

        lines = [
            "Spatial Econometrics Model Summary",
            "=" * 60,
            f"  Specification  : spatial {'lag (SAR)' if self.model_type_ == 'lag' else 'error (SEM)'}",
            f"  Observations   : {self.n_}",
            f"  Regressors     : {self.k_}",
            f"  Weights method : {self.weights_method}",
        ]

        if self.model_type_ == "lag":
            lines.append(f"  rho            : {self.rho_:.6f}")
        else:
            lines.append(f"  lambda         : {self.lambda_:.6f}")

        lines.append(f"  sigma^2        : {self.sigma2_:.6f}")

        if self.log_likelihood_ is not None:
            lines.append(f"  Log-likelihood : {self.log_likelihood_:.4f}")

        # AIC / BIC.
        if self.log_likelihood_ is not None:
            n_params = self.k_ + 1 + 1  # beta + spatial param + sigma2
            aic = -2 * self.log_likelihood_ + 2 * n_params
            bic = -2 * self.log_likelihood_ + n_params * np.log(self.n_)
            lines.append(f"  AIC            : {aic:.4f}")
            lines.append(f"  BIC            : {bic:.4f}")

        # Pseudo R-squared.
        if self._y_train is not None:
            ss_res = float(np.sum(self.residuals_ ** 2))
            ss_tot = float(np.sum((self._y_train - self._y_train.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            lines.append(f"  Pseudo R^2     : {r2:.6f}")

        # Coefficient table.
        lines.append("")
        lines.append("Coefficients:")
        lines.append("-" * 60)
        header = f"  {'Variable':20s} {'Coefficient':>14s} {'Std. Error':>12s} {'z':>8s}"
        lines.append(header)
        lines.append("  " + "-" * 56)

        se = self._compute_standard_errors()
        for i, name in enumerate(self._feature_names):
            b = self.beta_[i]
            s = se[i] if se is not None and i < len(se) else float("nan")
            z = b / s if s > 0 else float("nan")
            lines.append(f"  {name:20s} {b:>14.6f} {s:>12.6f} {z:>8.3f}")

        # Moran's I on residuals.
        if self.residuals_ is not None and self.W_ is not None:
            lines.append("")
            lines.append("Spatial diagnostics (residuals):")
            lines.append("-" * 60)
            mi = self.moran_i(self.residuals_, self.W_)
            lines.append(f"  Moran's I      : {mi['I']:.6f}")
            lines.append(f"  Expected       : {mi['expected']:.6f}")
            lines.append(f"  z-score        : {mi['z_score']:.4f}")
            lines.append(f"  p-value        : {mi['p_value']:.6f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit()`` has been called."""
        return self._fitted

    # ------------------------------------------------------------------
    # Internal: spreg backend
    # ------------------------------------------------------------------

    def _fit_spreg(
        self,
        y: np.ndarray,
        X: np.ndarray,
        W_arr: np.ndarray,
        model_type: str,
    ) -> None:
        """Fit using the PySAL spreg backend.

        Parameters
        ----------
        y : np.ndarray
        X : np.ndarray
            Includes intercept column if add_intercept is True.
        W_arr : np.ndarray
            Dense weights matrix.
        model_type : str
            ``"lag"`` or ``"error"``.
        """
        # spreg expects a libpysal.weights.W object.
        # Build from the dense array.
        neighbours = {}
        weights_dict = {}
        n = len(y)
        for i in range(n):
            nb = []
            wt = []
            for j in range(n):
                if W_arr[i, j] > 0 and i != j:
                    nb.append(j)
                    wt.append(float(W_arr[i, j]))
            neighbours[i] = nb
            weights_dict[i] = wt

        w_pysal = libpysal.weights.W(neighbours, weights_dict)
        w_pysal.transform = "r"

        # Remove intercept column for spreg (it adds its own).
        if self.add_intercept:
            X_no_const = X[:, 1:]
        else:
            X_no_const = X

        try:
            if model_type == "lag":
                result = spreg.ML_Lag(y.reshape(-1, 1), X_no_const, w=w_pysal)
                self.rho_ = float(result.rho)
                self.lambda_ = None
            else:
                result = spreg.ML_Error(y.reshape(-1, 1), X_no_const, w=w_pysal)
                self.lambda_ = float(result.lam)
                self.rho_ = None

            # spreg includes constant as the first coefficient.
            self.beta_ = result.betas.ravel()
            if self.add_intercept:
                # beta_ layout: [const, x1, x2, ...]
                pass
            self.residuals_ = y - result.predy.ravel()
            self.sigma2_ = float(result.sig2)
            self.log_likelihood_ = float(result.logll) if hasattr(result, "logll") else None

            logger.info(
                "Fitted spatial %s model via spreg (LL=%.4f).",
                model_type,
                self.log_likelihood_ or 0.0,
            )
        except Exception as exc:
            logger.warning(
                "spreg fitting failed (%s); falling back to numpy.",
                exc,
            )
            if model_type == "lag":
                self._fit_lag_numpy(y, X, W_arr)
            else:
                self._fit_error_numpy(y, X, W_arr)

    # ------------------------------------------------------------------
    # Internal: numpy spatial lag (S2SLS / concentrated ML)
    # ------------------------------------------------------------------

    def _fit_lag_numpy(
        self, y: np.ndarray, X: np.ndarray, W: np.ndarray
    ) -> None:
        """Fit a spatial lag model via concentrated log-likelihood.

        Uses a 1-D search over rho, with OLS for beta conditional on rho.

        y = rho * W @ y + X @ beta + eps
        => (y - rho * Wy) = X @ beta + eps
        """
        n = len(y)
        Wy = W @ y

        # Eigenvalues of W for the log-determinant term.
        eigvals = np.linalg.eigvals(W).real
        min_rho = 1.0 / min(eigvals.min(), -1e-6) + 0.01 if eigvals.min() < 0 else -0.99
        max_rho = 1.0 / max(eigvals.max(), 1e-6) - 0.01 if eigvals.max() > 0 else 0.99
        min_rho = max(min_rho, -0.99)
        max_rho = min(max_rho, 0.99)

        def _neg_concentrated_ll(rho_val: float) -> float:
            """Negative concentrated log-likelihood for a given rho."""
            y_star = y - rho_val * Wy
            beta_hat = np.linalg.lstsq(X, y_star, rcond=None)[0]
            resid = y_star - X @ beta_hat
            s2 = float(resid @ resid) / n

            # log|I - rho W|
            log_det = np.sum(np.log(np.abs(1.0 - rho_val * eigvals) + 1e-30))
            ll = -0.5 * n * np.log(2 * np.pi * s2) - 0.5 * n + log_det
            return -ll

        result = minimize_scalar(
            _neg_concentrated_ll,
            bounds=(min_rho, max_rho),
            method="bounded",
        )

        self.rho_ = float(result.x)
        self.lambda_ = None

        y_star = y - self.rho_ * Wy
        self.beta_ = np.linalg.lstsq(X, y_star, rcond=None)[0]
        self.residuals_ = y_star - X @ self.beta_
        self.sigma2_ = float(self.residuals_ @ self.residuals_) / n
        self.log_likelihood_ = -result.fun

        logger.info(
            "Fitted spatial lag (numpy): rho=%.6f, sigma2=%.6f, LL=%.4f",
            self.rho_, self.sigma2_, self.log_likelihood_,
        )

    # ------------------------------------------------------------------
    # Internal: numpy spatial error (concentrated ML)
    # ------------------------------------------------------------------

    def _fit_error_numpy(
        self, y: np.ndarray, X: np.ndarray, W: np.ndarray
    ) -> None:
        """Fit a spatial error model via concentrated log-likelihood.

        y = X @ beta + u,  u = lambda * W @ u + eps
        => (I - lambda * W) @ y = (I - lambda * W) @ X @ beta + eps
        """
        n = len(y)
        eigvals = np.linalg.eigvals(W).real
        min_lam = 1.0 / min(eigvals.min(), -1e-6) + 0.01 if eigvals.min() < 0 else -0.99
        max_lam = 1.0 / max(eigvals.max(), 1e-6) - 0.01 if eigvals.max() > 0 else 0.99
        min_lam = max(min_lam, -0.99)
        max_lam = min(max_lam, 0.99)

        def _neg_concentrated_ll(lam_val: float) -> float:
            """Negative concentrated LL for a given lambda."""
            I_lW = np.eye(n) - lam_val * W
            y_star = I_lW @ y
            X_star = I_lW @ X
            beta_hat = np.linalg.lstsq(X_star, y_star, rcond=None)[0]
            resid = y_star - X_star @ beta_hat
            s2 = float(resid @ resid) / n

            log_det = np.sum(np.log(np.abs(1.0 - lam_val * eigvals) + 1e-30))
            ll = -0.5 * n * np.log(2 * np.pi * s2) - 0.5 * n + log_det
            return -ll

        result = minimize_scalar(
            _neg_concentrated_ll,
            bounds=(min_lam, max_lam),
            method="bounded",
        )

        self.lambda_ = float(result.x)
        self.rho_ = None

        I_lW = np.eye(n) - self.lambda_ * W
        y_star = I_lW @ y
        X_star = I_lW @ X
        self.beta_ = np.linalg.lstsq(X_star, y_star, rcond=None)[0]
        resid_transformed = y_star - X_star @ self.beta_
        self.residuals_ = y - X @ self.beta_
        self.sigma2_ = float(resid_transformed @ resid_transformed) / n
        self.log_likelihood_ = -result.fun

        logger.info(
            "Fitted spatial error (numpy): lambda=%.6f, sigma2=%.6f, LL=%.4f",
            self.lambda_, self.sigma2_, self.log_likelihood_,
        )

    # ------------------------------------------------------------------
    # Internal: standard errors
    # ------------------------------------------------------------------

    def _compute_standard_errors(self) -> Optional[np.ndarray]:
        """Approximate standard errors for beta from the Hessian / OLS formula.

        Returns
        -------
        np.ndarray or None
            Standard errors of the same length as beta_.
        """
        if self._X_train is None or self.sigma2_ is None:
            return None

        X = self._X_train
        try:
            if self.model_type_ == "error" and self.lambda_ is not None:
                I_lW = np.eye(self.n_) - self.lambda_ * self.W_
                X = I_lW @ X
            XtX_inv = np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(self.sigma2_ * XtX_inv))
            return se
        except np.linalg.LinAlgError:
            return None

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"SpatialEconModel(weights={self.weights_method!r}, "
            f"k={self.k}, type={self.model_type_!r}, {status})"
        )
