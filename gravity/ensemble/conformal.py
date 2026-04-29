"""Conformal prediction for calibrated, distribution-free prediction intervals.

Provides split conformal and jackknife+ methods that wrap any point
predictor and produce intervals with finite-sample coverage guarantees.
Also supports matrix-level calibration for origin x store probability
grids and conditional (per-group) conformal intervals.

References
----------
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World*.
- Barber, Candes, Ramdas, Tibshirani (2021). "Predictive Inference with
  the Jackknife+." *Annals of Statistics*.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantile_with_coverage(scores: np.ndarray, alpha: float) -> float:
    """Compute the conformal quantile guaranteeing (1-alpha) coverage.

    For n calibration scores the quantile level is
    ceil((n+1)(1-alpha)) / n, clipped to the max score if needed.
    """
    n = len(scores)
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ConformalPredictor:
    """Distribution-free prediction intervals via conformal inference.

    Parameters
    ----------
    confidence : float
        Target coverage level (e.g. 0.90 for 90 % intervals).
    method : str
        ``"split"`` for split conformal or ``"jackknife_plus"`` for the
        jackknife+ method of Barber et al. (2021).
    """

    def __init__(self, confidence: float = 0.90, method: str = "split") -> None:
        if method not in ("split", "jackknife_plus"):
            raise ValueError(f"Unknown method: {method!r}")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1).")
        self.confidence = confidence
        self.method = method
        self._alpha = 1.0 - confidence
        self._scores: np.ndarray | None = None
        self._quantile: float | None = None
        self._calibrated: bool = False
        # For matrix calibration
        self._matrix_shape: tuple[int, int] | None = None
        self._matrix_index: pd.Index | None = None
        self._matrix_columns: pd.Index | None = None

    # ------------------------------------------------------------------
    # Scalar / vector calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
    ) -> "ConformalPredictor":
        """Compute nonconformity scores on calibration data.

        Parameters
        ----------
        y_true : array-like
            Actual observed values.
        y_pred : array-like
            Point predictions corresponding to *y_true*.

        Returns
        -------
        self
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        if len(y_true) == 0:
            raise ValueError("Calibration set must be non-empty.")

        if self.method == "split":
            self._scores = np.abs(y_true - y_pred)
            self._quantile = _quantile_with_coverage(self._scores, self._alpha)

        elif self.method == "jackknife_plus":
            # Accept pre-computed LOO residuals.  The caller is responsible
            # for supplying y_pred = \hat{y}_{-i} (the LOO prediction for
            # observation i).
            self._scores = np.abs(y_true - y_pred)
            self._quantile = _quantile_with_coverage(self._scores, self._alpha)

        self._calibrated = True
        logger.info(
            "Calibrated with %d scores.  quantile=%.6f  method=%s",
            len(self._scores), self._quantile, self.method,
        )
        return self

    def predict_interval(
        self, y_pred: np.ndarray | pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` arrays with guaranteed coverage.

        Parameters
        ----------
        y_pred : array-like
            New point predictions for which intervals are needed.

        Returns
        -------
        (lower, upper) : tuple of ndarray
        """
        self._check_calibrated()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        if self.method == "split":
            q = self._quantile
            return y_pred - q, y_pred + q

        elif self.method == "jackknife_plus":
            # Jackknife+ intervals:
            #   lower = quantile_{alpha/2}(y_pred - R_i)
            #   upper = quantile_{1-alpha/2}(y_pred + R_i)
            # where R_i are the LOO residuals.
            scores = self._scores
            lower_vals = np.quantile(
                y_pred[:, None] - scores[None, :], self._alpha / 2, axis=1
            ) if y_pred.ndim == 1 else y_pred - self._quantile
            upper_vals = np.quantile(
                y_pred[:, None] + scores[None, :], 1 - self._alpha / 2, axis=1
            ) if y_pred.ndim == 1 else y_pred + self._quantile

            # For large calibration sets the full outer product is expensive.
            # Fall back to scalar quantile if memory would exceed ~100 MB.
            n_pred = len(y_pred)
            n_cal = len(scores)
            if n_pred * n_cal * 8 > 100_000_000:
                logger.warning(
                    "Large calibration x prediction product (%d x %d). "
                    "Using scalar quantile approximation.",
                    n_pred, n_cal,
                )
                q = self._quantile
                return y_pred - q, y_pred + q

            return lower_vals, upper_vals

        # Should not reach here
        q = self._quantile
        return y_pred - q, y_pred + q

    # ------------------------------------------------------------------
    # Matrix calibration (origin x store)
    # ------------------------------------------------------------------

    def calibrate_matrix(
        self,
        observed_shares: pd.DataFrame,
        predicted_shares: pd.DataFrame,
    ) -> "ConformalPredictor":
        """Calibrate on full origin x store probability matrices.

        Flattens both matrices, computes per-cell nonconformity scores,
        then derives the conformal quantile.

        Parameters
        ----------
        observed_shares : DataFrame
            Actual visit probabilities (origin rows, store columns).
        predicted_shares : DataFrame
            Predicted probabilities, same shape.

        Returns
        -------
        self
        """
        if observed_shares.shape != predicted_shares.shape:
            raise ValueError("observed and predicted must have the same shape.")

        self._matrix_shape = observed_shares.shape
        self._matrix_index = observed_shares.index.copy()
        self._matrix_columns = observed_shares.columns.copy()

        y_true = observed_shares.values.ravel()
        y_pred = predicted_shares.values.ravel()
        return self.calibrate(y_true, y_pred)

    def predict_interval_matrix(
        self, predicted_shares: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(lower_df, upper_df)`` DataFrames with calibrated intervals.

        Parameters
        ----------
        predicted_shares : DataFrame
            New predicted probabilities (origin x store).

        Returns
        -------
        (lower_df, upper_df) : tuple of DataFrames
        """
        self._check_calibrated()
        flat = predicted_shares.values.ravel()
        lower, upper = self.predict_interval(flat)

        shape = predicted_shares.shape
        lower_df = pd.DataFrame(
            lower.reshape(shape),
            index=predicted_shares.index,
            columns=predicted_shares.columns,
        )
        upper_df = pd.DataFrame(
            upper.reshape(shape),
            index=predicted_shares.index,
            columns=predicted_shares.columns,
        )

        # Clip probabilities to [0, 1]
        lower_df = lower_df.clip(lower=0.0, upper=1.0)
        upper_df = upper_df.clip(lower=0.0, upper=1.0)

        return lower_df, upper_df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def coverage_diagnostic(
        self,
        y_true_test: np.ndarray | pd.Series,
        y_pred_test: np.ndarray | pd.Series,
    ) -> dict:
        """Check actual coverage on a held-out test set.

        Parameters
        ----------
        y_true_test : array-like
            True values on the test set.
        y_pred_test : array-like
            Point predictions on the test set.

        Returns
        -------
        dict with keys:
            ``empirical_coverage`` – fraction of test points inside the
                interval.
            ``avg_width`` – average interval width.
            ``efficiency`` – average width / mean(|y_true|) (lower is
                tighter).
            ``n_test`` – number of test observations.
        """
        self._check_calibrated()
        y_true_test = np.asarray(y_true_test, dtype=np.float64).ravel()
        y_pred_test = np.asarray(y_pred_test, dtype=np.float64).ravel()

        lower, upper = self.predict_interval(y_pred_test)
        covered = (y_true_test >= lower) & (y_true_test <= upper)
        widths = upper - lower

        mean_abs = np.mean(np.abs(y_true_test))
        efficiency = float(np.mean(widths) / mean_abs) if mean_abs > 0 else float("inf")

        result = {
            "empirical_coverage": float(np.mean(covered)),
            "avg_width": float(np.mean(widths)),
            "efficiency": efficiency,
            "n_test": len(y_true_test),
        }
        logger.info(
            "Coverage diagnostic: empirical=%.4f  target=%.4f  avg_width=%.6f",
            result["empirical_coverage"], self.confidence, result["avg_width"],
        )
        return result

    # ------------------------------------------------------------------
    # Conditional conformal
    # ------------------------------------------------------------------

    @staticmethod
    def conditional_conformal(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        groups: np.ndarray | pd.Series,
        confidence: float = 0.90,
    ) -> dict:
        """Per-group conformal intervals.

        Groups could be distance bands, consumer segments, income
        quintiles, etc.  A separate conformal quantile is computed for
        each group so that coverage is guaranteed conditionally.

        Parameters
        ----------
        y_true : array-like
            Actual values.
        y_pred : array-like
            Point predictions.
        groups : array-like
            Group labels (same length as *y_true*).
        confidence : float
            Target coverage (default 0.90).

        Returns
        -------
        dict mapping ``group_label -> {"quantile": float,
        "lower_fn": callable, "upper_fn": callable, "n_cal": int}``.

        The ``lower_fn`` and ``upper_fn`` callables accept a single
        numeric array and return the corresponding interval bound.
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        groups = np.asarray(groups).ravel()

        if not (len(y_true) == len(y_pred) == len(groups)):
            raise ValueError("y_true, y_pred, groups must have the same length.")

        alpha = 1.0 - confidence
        unique_groups = np.unique(groups)
        result: dict = {}

        for g in unique_groups:
            mask = groups == g
            scores_g = np.abs(y_true[mask] - y_pred[mask])
            q = _quantile_with_coverage(scores_g, alpha)
            result[g] = {
                "quantile": q,
                "lower_fn": lambda yhat, _q=q: np.asarray(yhat, dtype=np.float64) - _q,
                "upper_fn": lambda yhat, _q=q: np.asarray(yhat, dtype=np.float64) + _q,
                "n_cal": int(mask.sum()),
            }
            logger.debug("Group %s: n_cal=%d  quantile=%.6f", g, int(mask.sum()), q)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_calibrated(self) -> None:
        if not self._calibrated:
            raise RuntimeError(
                "Predictor has not been calibrated. Call calibrate() first."
            )
