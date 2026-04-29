"""Demand forecasting module with SARIMA, Prophet-style, and ETS support.

Provides a unified ``DemandForecaster`` that can fit multiple time-series
methods (SARIMA, a Prophet-style Fourier decomposition, and Holt-Winters
ETS), select the best by AIC, and produce forecasts with prediction
intervals.  Also includes a PELT changepoint detector.

Optional accelerators (statsmodels, prophet) are imported lazily.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import gammaln

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _ensure_float_series(series: pd.Series) -> pd.Series:
    """Return a copy with float64 values and sorted DatetimeIndex."""
    s = series.copy()
    s.index = pd.DatetimeIndex(s.index)
    s = s.sort_index().astype(np.float64)
    return s


def _detect_freq(series: pd.Series) -> str:
    """Infer the most likely frequency string."""
    freq = pd.infer_freq(series.index)
    if freq is not None:
        # Normalise aliases
        if freq.startswith("W"):
            return "W"
        if freq.startswith("Q"):
            return "Q"
        if freq.startswith("M") or freq.startswith("ME") or freq.startswith("MS"):
            return "M"
        return freq
    # Fallback: median gap
    deltas = np.diff(series.index.values).astype("timedelta64[D]").astype(int)
    med = np.median(deltas)
    if med <= 2:
        return "D"
    if med <= 10:
        return "W"
    if med <= 45:
        return "M"
    return "Q"


def _seasonal_period(freq: str) -> int:
    """Return the number of observations per season."""
    return {"D": 7, "W": 52, "M": 12, "Q": 4}.get(freq, 1)


# ---------------------------------------------------------------------------
# ADF stationarity test (manual implementation)
# ---------------------------------------------------------------------------


def _adf_pvalue_approx(stat: float, n: int) -> float:
    """Very rough p-value approximation for the Dickey-Fuller t-statistic.

    Uses the Mackinnon (1994) critical-value surface for the case with
    constant, no trend.  Returns an approximate p-value.
    """
    # Critical values at 1%, 5%, 10% (constant, no trend, asymptotic)
    crit = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}
    if stat < crit[0.01]:
        return 0.005
    if stat < crit[0.05]:
        return 0.03
    if stat < crit[0.10]:
        return 0.07
    return 0.50


def _adf_test(y: np.ndarray, max_lag: int = 5) -> float:
    """Return approximate p-value of augmented Dickey-Fuller test."""
    n = len(y)
    if n < 10:
        return 1.0
    dy = np.diff(y)
    y_lag = y[:-1]
    max_lag = min(max_lag, n // 3)
    # Build regression matrix: dy_t = alpha + gamma*y_{t-1} + sum(phi_k*dy_{t-k})
    T = len(dy) - max_lag
    if T < 5:
        return 1.0
    Y = dy[max_lag:]
    X = np.ones((T, 1 + 1 + max_lag))
    X[:, 1] = y_lag[max_lag:]
    for k in range(1, max_lag + 1):
        X[:, 1 + k] = dy[max_lag - k: -k] if k < len(dy) else 0
    try:
        beta, res, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        if len(res) == 0:
            sse = np.sum((Y - X @ beta) ** 2)
        else:
            sse = res[0]
        se = np.sqrt(sse / max(T - X.shape[1], 1))
        XtX_inv = np.linalg.inv(X.T @ X)
        se_gamma = se * np.sqrt(XtX_inv[1, 1])
        t_stat = beta[1] / se_gamma if se_gamma > 0 else 0
    except np.linalg.LinAlgError:
        return 1.0
    return _adf_pvalue_approx(t_stat, n)


# ---------------------------------------------------------------------------
# SARIMA (manual implementation with scipy.optimize fallback)
# ---------------------------------------------------------------------------


class _SARIMAManual:
    """Minimal SARIMA(p,d,q)(P,D,Q,s) estimated via conditional MLE."""

    def __init__(
        self,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
    ) -> None:
        self.p, self.d, self.q = order
        self.P, self.D, self.Q, self.s = seasonal_order
        self._params: np.ndarray | None = None
        self._resid_std: float = 1.0
        self._y: np.ndarray | None = None
        self._aic: float = np.inf

    @property
    def n_params(self) -> int:
        return self.p + self.q + self.P + self.Q + 1  # +1 for constant

    def _difference(self, y: np.ndarray) -> np.ndarray:
        z = y.copy()
        for _ in range(self.D):
            if self.s > 0 and len(z) > self.s:
                z = z[self.s:] - z[:-self.s]
        for _ in range(self.d):
            z = np.diff(z)
        return z

    def _neg_log_lik(self, params: np.ndarray, z: np.ndarray) -> float:
        """Conditional negative log-likelihood (Gaussian errors)."""
        n = len(z)
        ar_p = params[: self.p]
        ma_q = params[self.p: self.p + self.q]
        sar_P = params[self.p + self.q: self.p + self.q + self.P]
        sma_Q = params[self.p + self.q + self.P: self.p + self.q + self.P + self.Q]
        const = params[-1]

        max_start = max(self.p, self.P * self.s if self.s > 0 else 0, 1)
        if max_start >= n:
            return 1e12

        errors = np.zeros(n)
        for t in range(max_start, n):
            pred = const
            for i in range(self.p):
                pred += ar_p[i] * z[t - 1 - i]
            for j in range(self.q):
                pred += ma_q[j] * errors[t - 1 - j]
            if self.s > 0:
                for i in range(self.P):
                    idx = t - self.s * (1 + i)
                    if idx >= 0:
                        pred += sar_P[i] * z[idx]
                for j in range(self.Q):
                    idx = t - self.s * (1 + j)
                    if idx >= 0:
                        pred += sma_Q[j] * errors[idx]
            errors[t] = z[t] - pred

        useful = errors[max_start:]
        T = len(useful)
        if T < 2:
            return 1e12
        ss = np.sum(useful ** 2)
        sigma2 = ss / T
        if sigma2 <= 0:
            return 1e12
        nll = 0.5 * T * np.log(2 * np.pi * sigma2) + ss / (2 * sigma2)
        return float(nll)

    def fit(self, y: np.ndarray) -> "_SARIMAManual":
        self._y = y.copy()
        z = self._difference(y)
        if len(z) < 5:
            self._aic = np.inf
            self._params = np.zeros(self.n_params)
            return self

        x0 = np.zeros(self.n_params)
        try:
            result = optimize.minimize(
                self._neg_log_lik,
                x0,
                args=(z,),
                method="L-BFGS-B",
                options={"maxiter": 300, "ftol": 1e-6},
            )
            self._params = result.x
            nll = result.fun
        except Exception:
            self._params = x0
            nll = self._neg_log_lik(x0, z)

        k = self.n_params
        self._aic = 2 * nll + 2 * k

        # Residual std for prediction intervals
        max_start = max(self.p, self.P * self.s if self.s > 0 else 0, 1)
        errors = self._compute_errors(z, self._params)
        useful = errors[max_start:]
        self._resid_std = float(np.std(useful)) if len(useful) > 0 else 1.0
        return self

    def _compute_errors(self, z: np.ndarray, params: np.ndarray) -> np.ndarray:
        n = len(z)
        ar_p = params[: self.p]
        ma_q = params[self.p: self.p + self.q]
        sar_P = params[self.p + self.q: self.p + self.q + self.P]
        sma_Q = params[self.p + self.q + self.P: self.p + self.q + self.P + self.Q]
        const = params[-1]
        max_start = max(self.p, self.P * self.s if self.s > 0 else 0, 1)
        errors = np.zeros(n)
        for t in range(max_start, n):
            pred = const
            for i in range(self.p):
                pred += ar_p[i] * z[t - 1 - i]
            for j in range(self.q):
                pred += ma_q[j] * errors[t - 1 - j]
            if self.s > 0:
                for i in range(self.P):
                    idx = t - self.s * (1 + i)
                    if idx >= 0:
                        pred += sar_P[i] * z[idx]
                for j in range(self.Q):
                    idx = t - self.s * (1 + j)
                    if idx >= 0:
                        pred += sma_Q[j] * errors[idx]
            errors[t] = z[t] - pred
        return errors

    def forecast(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (point_forecast, stderr_per_step)."""
        if self._params is None or self._y is None:
            return np.zeros(h), np.ones(h)

        z = self._difference(self._y)
        n = len(z)
        params = self._params
        ar_p = params[: self.p]
        ma_q = params[self.p: self.p + self.q]
        sar_P = params[self.p + self.q: self.p + self.q + self.P]
        sma_Q = params[self.p + self.q + self.P: self.p + self.q + self.P + self.Q]
        const = params[-1]

        # Extend z and errors arrays
        errors = self._compute_errors(z, params)
        z_ext = np.concatenate([z, np.zeros(h)])
        err_ext = np.concatenate([errors, np.zeros(h)])

        for t_offset in range(h):
            t = n + t_offset
            pred = const
            for i in range(self.p):
                idx = t - 1 - i
                if 0 <= idx < len(z_ext):
                    pred += ar_p[i] * z_ext[idx]
            # Future errors are zero (conditional mean forecast)
            for j in range(self.q):
                idx = t - 1 - j
                if 0 <= idx < len(err_ext):
                    pred += ma_q[j] * err_ext[idx]
            if self.s > 0:
                for i in range(self.P):
                    idx = t - self.s * (1 + i)
                    if 0 <= idx < len(z_ext):
                        pred += sar_P[i] * z_ext[idx]
                for j in range(self.Q):
                    idx = t - self.s * (1 + j)
                    if 0 <= idx < len(err_ext):
                        pred += sma_Q[j] * err_ext[idx]
            z_ext[t] = pred

        fc_diff = z_ext[n:]

        # Undo differencing (approximately, using last known values)
        fc = self._undo_difference(self._y, fc_diff, h)
        se = self._resid_std * np.sqrt(np.arange(1, h + 1))
        return fc, se

    def _undo_difference(
        self, y_orig: np.ndarray, fc_diff: np.ndarray, h: int
    ) -> np.ndarray:
        """Approximately invert the differencing to get level forecasts."""
        fc = fc_diff.copy()

        # Undo non-seasonal differencing
        for _ in range(self.d):
            # Cumsum starting from last observed differenced level
            last = y_orig[-1] if len(y_orig) > 0 else 0
            fc = np.cumsum(np.concatenate([[last], fc]))[1:]

        # Undo seasonal differencing
        if self.s > 0:
            for _ in range(self.D):
                # Reconstruct using last season's values
                tail = y_orig[-self.s:].copy() if len(y_orig) >= self.s else y_orig.copy()
                rebuilt = np.zeros(h)
                for t in range(h):
                    if t < self.s:
                        ref = tail[t % len(tail)] if len(tail) > 0 else 0
                    else:
                        ref = rebuilt[t - self.s]
                    rebuilt[t] = fc[t] + ref
                fc = rebuilt
        return fc


def _fit_sarima_auto(
    y: np.ndarray, s: int
) -> tuple[_SARIMAManual, float]:
    """Grid-search over ARIMA orders and return (best_model, best_aic)."""
    best_aic = np.inf
    best_model: _SARIMAManual | None = None

    # Test stationarity
    p_val = _adf_test(y)
    d_range = [0, 1] if p_val > 0.05 else [0]

    # Seasonal differencing test
    D_range = [0]
    if s > 1 and len(y) > 2 * s:
        D_range = [0, 1]

    for d in d_range:
        for q in range(3):
            for p in range(3):
                if s > 1:
                    seasonal_orders = [
                        (P, D, Q, s)
                        for P in range(2)
                        for D in D_range
                        for Q in range(2)
                    ]
                else:
                    seasonal_orders = [(0, 0, 0, 0)]

                for sorder in seasonal_orders:
                    try:
                        m = _SARIMAManual(order=(p, d, q), seasonal_order=sorder)
                        m.fit(y)
                        if m._aic < best_aic:
                            best_aic = m._aic
                            best_model = m
                    except Exception:
                        continue

    if best_model is None:
        best_model = _SARIMAManual(order=(0, 1, 0), seasonal_order=(0, 0, 0, 0))
        best_model.fit(y)
        best_aic = best_model._aic

    return best_model, best_aic


# ---------------------------------------------------------------------------
# Prophet-style decomposition (no fbprophet dependency)
# ---------------------------------------------------------------------------


class _ProphetStyle:
    """Trend + Fourier seasonality, fit with ridge regression."""

    def __init__(self, s: int, n_fourier: int = 5, ridge_alpha: float = 1.0) -> None:
        self.s = s
        self.n_fourier = min(n_fourier, s // 2) if s > 1 else 0
        self.ridge_alpha = ridge_alpha
        self._weights: np.ndarray | None = None
        self._changepoints: list[int] = []
        self._n: int = 0
        self._aic: float = np.inf
        self._resid_std: float = 1.0
        self._y: np.ndarray | None = None

    def _build_design(self, n: int, changepoints: list[int]) -> np.ndarray:
        """Design matrix: intercept, trend, changepoint deltas, Fourier terms."""
        cols: list[np.ndarray] = []
        t = np.arange(n, dtype=np.float64)

        # Intercept + trend
        cols.append(np.ones(n))
        cols.append(t / max(n, 1))

        # Piecewise linear trend (changepoint indicators)
        for cp in changepoints:
            delta = np.maximum(t - cp, 0) / max(n, 1)
            cols.append(delta)

        # Fourier seasonality
        if self.s > 1:
            for k in range(1, self.n_fourier + 1):
                cols.append(np.sin(2 * np.pi * k * t / self.s))
                cols.append(np.cos(2 * np.pi * k * t / self.s))

        return np.column_stack(cols)

    def _detect_changepoints(self, y: np.ndarray, n_max: int = 10) -> list[int]:
        """Simple changepoint candidates: largest curvature in trend."""
        n = len(y)
        if n < 20:
            return []
        # Smooth and find curvature
        win = max(3, n // 20)
        smoothed = np.convolve(y, np.ones(win) / win, mode="same")
        second_diff = np.abs(np.diff(smoothed, n=2))
        # Pick top n_max candidates (at least 10% from edges)
        margin = max(5, n // 10)
        second_diff[:margin] = 0
        second_diff[-margin:] = 0
        n_cp = min(n_max, max(1, n // 50))
        candidates = np.argsort(second_diff)[-n_cp:]
        return sorted(candidates.tolist())

    def fit(self, y: np.ndarray) -> "_ProphetStyle":
        self._y = y.copy()
        self._n = len(y)
        self._changepoints = self._detect_changepoints(y)

        X = self._build_design(self._n, self._changepoints)
        p = X.shape[1]

        # Ridge regression: (X'X + alpha*I)^{-1} X'y
        A = X.T @ X + self.ridge_alpha * np.eye(p)
        try:
            self._weights = np.linalg.solve(A, X.T @ y)
        except np.linalg.LinAlgError:
            self._weights = np.linalg.lstsq(X, y, rcond=None)[0]

        residuals = y - X @ self._weights
        ss = np.sum(residuals ** 2)
        self._resid_std = float(np.sqrt(ss / max(self._n - p, 1)))

        # AIC
        sigma2 = ss / max(self._n, 1)
        if sigma2 > 0:
            nll = 0.5 * self._n * (np.log(2 * np.pi * sigma2) + 1)
        else:
            nll = 0
        self._aic = 2 * nll + 2 * p
        return self

    def forecast(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        n_total = self._n + h
        X = self._build_design(n_total, self._changepoints)
        yhat_all = X @ self._weights
        fc = yhat_all[self._n:]
        se = self._resid_std * np.sqrt(np.arange(1, h + 1))
        return fc, se

    def decompose(self) -> dict:
        """Return trend, seasonal, residual arrays."""
        if self._weights is None or self._y is None:
            return {"trend": np.array([]), "seasonal": np.array([]), "residual": np.array([])}

        n = self._n
        t = np.arange(n, dtype=np.float64)

        # Trend = intercept + slope*t + changepoint deltas
        n_cp = len(self._changepoints)
        trend = self._weights[0] + self._weights[1] * t / max(n, 1)
        for i, cp in enumerate(self._changepoints):
            trend += self._weights[2 + i] * np.maximum(t - cp, 0) / max(n, 1)

        # Seasonal
        seasonal = np.zeros(n)
        base = 2 + n_cp
        if self.s > 1:
            for k in range(self.n_fourier):
                a = self._weights[base + 2 * k]
                b = self._weights[base + 2 * k + 1]
                seasonal += a * np.sin(2 * np.pi * (k + 1) * t / self.s)
                seasonal += b * np.cos(2 * np.pi * (k + 1) * t / self.s)

        residual = self._y - trend - seasonal
        return {"trend": trend, "seasonal": seasonal, "residual": residual}


# ---------------------------------------------------------------------------
# ETS (Holt-Winters)
# ---------------------------------------------------------------------------


class _ETSHoltWinters:
    """Holt-Winters exponential smoothing (additive trend & seasonality)."""

    def __init__(self, s: int) -> None:
        self.s = max(s, 1)
        self._params: np.ndarray | None = None
        self._level: np.ndarray | None = None
        self._trend: np.ndarray | None = None
        self._season: np.ndarray | None = None
        self._resid_std: float = 1.0
        self._aic: float = np.inf
        self._y: np.ndarray | None = None

    def _hw_filter(
        self, y: np.ndarray, alpha: float, beta: float, gamma: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Run the Holt-Winters recursion.  Returns (level, trend, season, mse)."""
        n = len(y)
        s = self.s

        level = np.zeros(n)
        trend = np.zeros(n)
        season = np.zeros(n + s)

        # Initialise
        if s > 1 and n >= 2 * s:
            level[0] = np.mean(y[:s])
            trend[0] = (np.mean(y[s: 2 * s]) - np.mean(y[:s])) / s
            for j in range(s):
                season[j] = y[j] - level[0]
        else:
            level[0] = y[0]
            trend[0] = y[1] - y[0] if n > 1 else 0

        sse = 0.0
        for t in range(1, n):
            s_idx = t % s if s > 1 else 0
            forecast_t = level[t - 1] + trend[t - 1] + (season[s_idx] if s > 1 else 0)
            error = y[t] - forecast_t
            sse += error ** 2

            level[t] = alpha * (y[t] - (season[s_idx] if s > 1 else 0)) + (1 - alpha) * (
                level[t - 1] + trend[t - 1]
            )
            trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
            if s > 1:
                season[t + s] = gamma * (y[t] - level[t]) + (1 - gamma) * season[s_idx]
            else:
                season[t] = 0

        mse = sse / max(n - 1, 1)
        return level, trend, season, mse

    def fit(self, y: np.ndarray) -> "_ETSHoltWinters":
        self._y = y.copy()
        n = len(y)

        def objective(params: np.ndarray) -> float:
            a, b, g = np.clip(params, 1e-4, 1 - 1e-4)
            _, _, _, mse = self._hw_filter(y, a, b, g)
            return mse

        x0 = np.array([0.3, 0.05, 0.1])
        bounds = [(1e-4, 0.9999)] * 3
        try:
            result = optimize.minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
            best_params = np.clip(result.x, 1e-4, 0.9999)
        except Exception:
            best_params = x0

        self._params = best_params
        a, b, g = best_params
        self._level, self._trend, self._season, mse = self._hw_filter(y, a, b, g)

        self._resid_std = float(np.sqrt(mse)) if mse > 0 else 1.0
        # AIC approximation
        k = 3  # alpha, beta, gamma
        if mse > 0:
            nll = 0.5 * n * np.log(2 * np.pi * mse) + 0.5 * n
        else:
            nll = 0
        self._aic = 2 * nll + 2 * k
        return self

    def forecast(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._y)
        s = self.s
        last_level = self._level[-1]
        last_trend = self._trend[-1]
        fc = np.zeros(h)
        for t in range(h):
            s_idx = (n + t) % s if s > 1 else 0
            s_val = self._season[s_idx] if s_idx < len(self._season) else 0
            fc[t] = last_level + (t + 1) * last_trend + s_val
        se = self._resid_std * np.sqrt(np.arange(1, h + 1))
        return fc, se


# ---------------------------------------------------------------------------
# PELT changepoint detection
# ---------------------------------------------------------------------------


def _pelt_changepoints(
    y: np.ndarray, penalty: str = "bic"
) -> list[int]:
    """PELT (Pruned Exact Linear Time) changepoint detection.

    Uses Gaussian mean-shift cost:  cost(y[a:b]) = (b-a)*log(var) + const.
    """
    n = len(y)
    if n < 4:
        return []

    # Penalty
    if penalty == "bic":
        pen = np.log(n)
    elif penalty == "aic":
        pen = 2.0
    else:
        pen = float(penalty)

    # Cost of segment y[a:b]  (Gaussian mean change)
    def cost(a: int, b: int) -> float:
        seg = y[a:b]
        m = len(seg)
        if m < 2:
            return 0.0
        var = np.var(seg)
        if var <= 0:
            return 0.0
        return m * np.log(var)

    # DP with PELT pruning
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    cp_list: list[list[int]] = [[] for _ in range(n + 1)]
    candidates: list[int] = [0]

    for t in range(1, n + 1):
        best_cost = np.inf
        best_s = 0
        for s in candidates:
            c = F[s] + cost(s, t) + pen
            if c < best_cost:
                best_cost = c
                best_s = s
        F[t] = best_cost
        cp_list[t] = cp_list[best_s] + [best_s]

        # Prune: keep only candidates where F[s] + cost(s,t) <= F[t]
        new_candidates = []
        for s in candidates:
            if F[s] + cost(s, t) <= F[t]:
                new_candidates.append(s)
        new_candidates.append(t)
        candidates = new_candidates

    # Return changepoint indices (drop the initial 0)
    cps = [c for c in cp_list[n] if c > 0]
    return cps


# ---------------------------------------------------------------------------
# DemandForecaster
# ---------------------------------------------------------------------------


class DemandForecaster:
    """Unified demand forecaster supporting SARIMA, Prophet-style, and ETS.

    Parameters
    ----------
    method : str
        ``"sarima"``, ``"prophet"``, ``"ets"``, or ``"auto"`` (tries all
        available methods and picks the one with best AIC).
    confidence : float
        Prediction interval coverage level (default 0.90).
    """

    def __init__(self, method: str = "auto", confidence: float = 0.90) -> None:
        if method not in ("sarima", "prophet", "ets", "auto"):
            raise ValueError(f"Unknown method: {method!r}")
        self.method = method
        self.confidence = confidence

        self._series: pd.Series | None = None
        self._freq: str | None = None
        self._seasonal_period: int = 1
        self._model: object | None = None
        self._model_name: str | None = None
        self._aic: float = np.inf
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, series: pd.Series, freq: str | None = None) -> "DemandForecaster":
        """Fit on a pandas Series with DatetimeIndex.

        Parameters
        ----------
        series : Series
            Time series values with a DatetimeIndex.
        freq : str, optional
            ``"D"``, ``"W"``, ``"M"``, ``"Q"``.  Auto-detected if *None*.

        Returns
        -------
        self
        """
        s = _ensure_float_series(series)
        if len(s) < 4:
            raise ValueError("Need at least 4 observations to fit.")

        self._series = s
        self._freq = freq or _detect_freq(s)
        self._seasonal_period = _seasonal_period(self._freq)
        y = s.values

        candidates: dict[str, tuple[object, float]] = {}

        if self.method in ("sarima", "auto"):
            candidates["sarima"] = self._fit_sarima(y)

        if self.method in ("prophet", "auto"):
            candidates["prophet"] = self._fit_prophet(y)

        if self.method in ("ets", "auto"):
            candidates["ets"] = self._fit_ets(y)

        # Select best
        best_name = min(candidates, key=lambda k: candidates[k][1])
        self._model, self._aic = candidates[best_name]
        self._model_name = best_name
        self._fitted = True
        logger.info(
            "Selected method=%s  AIC=%.2f  freq=%s  s=%d",
            best_name, self._aic, self._freq, self._seasonal_period,
        )
        return self

    # ------------------------------------------------------------------
    # forecast
    # ------------------------------------------------------------------

    def forecast(self, periods: int = 12) -> pd.DataFrame:
        """Return DataFrame with columns: date, forecast, lower, upper.

        Parameters
        ----------
        periods : int
            Number of future periods to forecast.

        Returns
        -------
        DataFrame
        """
        self._check_fitted()
        from scipy.stats import norm

        z = norm.ppf(1 - (1 - self.confidence) / 2)

        fc, se = self._forecast_raw(periods)
        lower = fc - z * se
        upper = fc + z * se

        last_date = self._series.index[-1]
        future_idx = pd.date_range(
            start=last_date, periods=periods + 1, freq=self._freq
        )[1:]

        return pd.DataFrame(
            {"date": future_idx, "forecast": fc, "lower": lower, "upper": upper}
        )

    # ------------------------------------------------------------------
    # decompose
    # ------------------------------------------------------------------

    def decompose(self) -> dict:
        """Seasonal decomposition: trend, seasonal, residual components.

        Returns
        -------
        dict with keys ``"trend"``, ``"seasonal"``, ``"residual"`` – each
        a numpy array aligned with the input series.
        """
        self._check_fitted()

        if isinstance(self._model, _ProphetStyle):
            return self._model.decompose()

        # For SARIMA / ETS fall back to a Prophet-style decomposition
        y = self._series.values
        sp = self._seasonal_period
        ps = _ProphetStyle(s=sp)
        ps.fit(y)
        return ps.decompose()

    # ------------------------------------------------------------------
    # from_market_results
    # ------------------------------------------------------------------

    @staticmethod
    def from_market_results(
        results_over_time: list[dict],
        metric: str = "total_population",
        dates: list | None = None,
    ) -> "DemandForecaster":
        """Convenience: build a forecaster from repeated batch-runner results.

        Parameters
        ----------
        results_over_time : list of dict
            Each dict is one snapshot containing at least the key *metric*.
        metric : str
            Which key to extract from each snapshot.
        dates : list, optional
            DatetimeIndex-compatible dates.  If *None*, sequential integers.

        Returns
        -------
        DemandForecaster (unfitted – caller should call ``fit``).
        """
        values = [r[metric] for r in results_over_time]
        if dates is None:
            idx = pd.date_range("2020-01-01", periods=len(values), freq="M")
        else:
            idx = pd.DatetimeIndex(dates)
        series = pd.Series(values, index=idx, dtype=np.float64)
        forecaster = DemandForecaster()
        forecaster.fit(series)
        return forecaster

    # ------------------------------------------------------------------
    # changepoints
    # ------------------------------------------------------------------

    def changepoints(self, penalty: str = "bic") -> list[int]:
        """Detect structural breaks / regime changes using PELT.

        Parameters
        ----------
        penalty : str
            ``"bic"`` or ``"aic"``.

        Returns
        -------
        list of int
            Changepoint indices into the original series.
        """
        if self._series is None:
            raise RuntimeError("No series available. Call fit() first.")
        return _pelt_changepoints(self._series.values, penalty=penalty)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def _fit_sarima(self, y: np.ndarray) -> tuple[object, float]:
        """Try statsmodels SARIMAX first, fall back to manual."""
        sp = self._seasonal_period
        try:
            import statsmodels.api as sm

            best_aic = np.inf
            best_model = None
            d_range = [0, 1]
            for p in range(3):
                for d in d_range:
                    for q in range(3):
                        try:
                            if sp > 1 and len(y) > 2 * sp:
                                mod = sm.tsa.SARIMAX(
                                    y,
                                    order=(p, d, q),
                                    seasonal_order=(1, 0, 1, sp),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )
                            else:
                                mod = sm.tsa.SARIMAX(
                                    y,
                                    order=(p, d, q),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                res = mod.fit(disp=False, maxiter=100)
                            if res.aic < best_aic:
                                best_aic = res.aic
                                best_model = res
                        except Exception:
                            continue
            if best_model is not None:
                logger.debug("statsmodels SARIMA AIC=%.2f", best_aic)
                return best_model, best_aic
        except ImportError:
            pass

        logger.debug("Falling back to manual SARIMA.")
        model, aic = _fit_sarima_auto(y, sp)
        return model, aic

    def _fit_prophet(self, y: np.ndarray) -> tuple[object, float]:
        """Try fbprophet / prophet first, fall back to Fourier decomposition."""
        try:
            from prophet import Prophet

            df = pd.DataFrame({
                "ds": self._series.index,
                "y": y,
            })
            m = Prophet(yearly_seasonality="auto", weekly_seasonality="auto")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(df)
            # Approximate AIC from in-sample residuals
            fitted = m.predict(df)
            resid = y - fitted["yhat"].values
            ss = np.sum(resid ** 2)
            n = len(y)
            k = 10  # rough param count
            sigma2 = ss / max(n, 1)
            nll = 0.5 * n * np.log(2 * np.pi * sigma2 + 1e-12) + 0.5 * n
            aic = 2 * nll + 2 * k
            logger.debug("prophet AIC=%.2f", aic)
            return m, aic
        except ImportError:
            pass

        logger.debug("Falling back to Prophet-style Fourier decomposition.")
        sp = self._seasonal_period
        model = _ProphetStyle(s=sp)
        model.fit(y)
        return model, model._aic

    def _fit_ets(self, y: np.ndarray) -> tuple[object, float]:
        sp = self._seasonal_period
        model = _ETSHoltWinters(s=sp)
        model.fit(y)
        return model, model._aic

    def _forecast_raw(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Dispatch forecasting to the underlying model."""
        model = self._model

        # statsmodels result
        try:
            import statsmodels.api as sm

            if hasattr(model, "get_forecast"):
                fcast = model.get_forecast(steps=h)
                fc = fcast.predicted_mean
                ci = fcast.conf_int(alpha=1 - self.confidence)
                # Convert CI to stderr estimate
                from scipy.stats import norm
                z = norm.ppf(1 - (1 - self.confidence) / 2)
                se = (ci.iloc[:, 1] - ci.iloc[:, 0]) / (2 * z)
                return fc.values, se.values
        except (ImportError, AttributeError):
            pass

        # prophet
        try:
            from prophet import Prophet

            if isinstance(model, Prophet):
                future = model.make_future_dataframe(periods=h, freq=self._freq)
                pred = model.predict(future)
                fc = pred["yhat"].values[-h:]
                lower = pred["yhat_lower"].values[-h:]
                upper = pred["yhat_upper"].values[-h:]
                se = (upper - lower) / (2 * 1.645)  # prophet uses ~90% by default
                return fc, np.maximum(se, 1e-6)
        except (ImportError, TypeError):
            pass

        # Manual models
        if isinstance(model, (_SARIMAManual, _ProphetStyle, _ETSHoltWinters)):
            return model.forecast(h)

        # Fallback: naive repeat
        y = self._series.values
        fc = np.full(h, y[-1])
        se = np.std(y) * np.sqrt(np.arange(1, h + 1))
        return fc, se
