"""
Ensemble Model Averaging
========================
Combines predictions from multiple model layers (Huff, GWR, Mixed Logit,
Latent Class, HMM-adjusted, XGBoost residual-corrected, etc.) into a
single consensus origin x store probability matrix.

Three combination strategies are provided:

    - **Bayesian Model Averaging (BMA)**: Weights proportional to
      exp(-BIC/2) or holdout log-likelihood, producing a principled
      posterior-weighted blend that penalises model complexity.
    - **Stacking (super-learner)**: A ridge-regression meta-learner
      trained on out-of-fold predictions, which can learn nonlinear
      complementarities between models.
    - **Simple Average**: Equal weights -- a surprisingly strong baseline
      in many forecasting contexts (Clemen 1989).

Usage
-----
>>> from gravity.ensemble.model_averaging import EnsembleAverager
>>> ens = EnsembleAverager()
>>> ens.add_model("huff", huff_probs)
>>> ens.add_model("mixed_logit", ml_probs)
>>> ens.add_model("xgboost", xgb_probs)
>>> ens.fit_weights(observed_shares, method="bayesian_averaging")
>>> blended = ens.predict()

References
----------
Hoeting, J. A. et al. (1999). "Bayesian Model Averaging: A Tutorial."
    *Statistical Science*, 14(4), 382-417.
Wolpert, D. H. (1992). "Stacked Generalization." *Neural Networks*,
    5(2), 241-259.
Clemen, R. T. (1989). "Combining forecasts: A review and annotated
    bibliography." *International Journal of Forecasting*, 5(4), 559-583.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EnsembleAverager
# ---------------------------------------------------------------------------

class EnsembleAverager:
    """Combine origin x store probability predictions from multiple models.

    Each registered model contributes a probability matrix of shape
    ``(n_origins, n_stores)``.  After fitting, ``predict()`` returns
    the weighted blend.

    Parameters
    ----------
    normalize : bool, default True
        If True, ensure that blended probabilities sum to 1.0 across
        stores for each origin (row-normalise after averaging).
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

        # Storage
        self._models: dict[str, dict] = {}  # name -> {predictions, weight, meta}
        self._weights: Optional[np.ndarray] = None
        self._model_names: list[str] = []
        self._fitted: bool = False
        self._fit_method: Optional[str] = None
        self._stacking_meta: Optional[Ridge] = None

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def add_model(
        self,
        name: str,
        predictions: pd.DataFrame,
        weight: Optional[float] = None,
        bic: Optional[float] = None,
        log_likelihood: Optional[float] = None,
        n_params: Optional[int] = None,
    ) -> "EnsembleAverager":
        """Register a model's origin x store probability predictions.

        Parameters
        ----------
        name : str
            Unique model identifier (e.g. ``"huff"``, ``"mixed_logit"``).
        predictions : pd.DataFrame
            Probability matrix, shape ``(n_origins, n_stores)``.
            Index = origin_id, columns = store_id.
        weight : float or None
            Manual weight override.  If supplied, it is used as a starting
            weight; ``fit_weights`` may replace it.
        bic : float or None
            Bayesian Information Criterion for this model (lower = better).
            Used by ``bayesian_averaging``.
        log_likelihood : float or None
            Holdout log-likelihood for this model (higher = better).
            Used by ``bayesian_averaging`` when BIC is not available.
        n_params : int or None
            Number of free parameters.  Used for BIC computation if BIC
            is not directly supplied.

        Returns
        -------
        EnsembleAverager
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If *name* is already registered.
        """
        if name in self._models:
            raise ValueError(
                f"Model '{name}' is already registered.  Remove it first "
                f"or use a different name."
            )

        self._models[name] = {
            "predictions": predictions.copy(),
            "weight": weight,
            "bic": bic,
            "log_likelihood": log_likelihood,
            "n_params": n_params,
        }
        self._model_names = list(self._models.keys())
        self._fitted = False
        logger.info("Registered model '%s' (%d origins x %d stores).",
                     name, predictions.shape[0], predictions.shape[1])
        return self

    def remove_model(self, name: str) -> "EnsembleAverager":
        """Remove a previously registered model.

        Parameters
        ----------
        name : str
            Model name to remove.

        Returns
        -------
        EnsembleAverager
            ``self``.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found.")
        del self._models[name]
        self._model_names = list(self._models.keys())
        self._fitted = False
        return self

    # ------------------------------------------------------------------
    # Weight fitting
    # ------------------------------------------------------------------

    def fit_weights(
        self,
        actual_shares: pd.DataFrame,
        method: str = "bayesian_averaging",
        *,
        stacking_alpha: float = 1.0,
        stacking_cv: int = 5,
    ) -> "EnsembleAverager":
        """Learn optimal model weights from holdout data.

        Parameters
        ----------
        actual_shares : pd.DataFrame
            Observed origin x store visit shares.  Index = origin_id,
            columns = store_id.  Rows should sum to 1.
        method : str, default "bayesian_averaging"
            Weight-learning strategy.  One of:

            - ``"bayesian_averaging"`` -- weight proportional to
              exp(-BIC/2) or exp(holdout_LL).  Falls back to holdout
              log-likelihood if BIC is not available.
            - ``"stacking"`` -- ridge-regression meta-learner on
              per-cell predictions.
            - ``"simple_average"`` -- equal weights.
        stacking_alpha : float, default 1.0
            Ridge regularisation strength (only used when
            method="stacking").
        stacking_cv : int, default 5
            Cross-validation folds for stacking meta-learner
            (only used when method="stacking").

        Returns
        -------
        EnsembleAverager
            ``self``, with fitted weights.

        Raises
        ------
        ValueError
            If fewer than 2 models are registered, or if *method* is
            unrecognised.
        """
        if len(self._models) < 2:
            raise ValueError(
                f"Need at least 2 models to ensemble; got {len(self._models)}."
            )

        valid_methods = {"bayesian_averaging", "stacking", "simple_average"}
        if method not in valid_methods:
            raise ValueError(
                f"Unknown method '{method}'.  Choose from {valid_methods}."
            )

        self._fit_method = method

        if method == "simple_average":
            self._weights = self._fit_simple_average()
        elif method == "bayesian_averaging":
            self._weights = self._fit_bayesian(actual_shares)
        elif method == "stacking":
            self._weights = self._fit_stacking(
                actual_shares,
                alpha=stacking_alpha,
                cv=stacking_cv,
            )

        self._fitted = True
        logger.info(
            "Fitted ensemble weights via '%s': %s",
            method,
            {n: round(float(w), 4)
             for n, w in zip(self._model_names, self._weights)},
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self) -> pd.DataFrame:
        """Return the weighted-average probability matrix.

        Returns
        -------
        pd.DataFrame
            Blended origin x store probabilities.  Rows sum to 1 if
            ``normalize=True``.

        Raises
        ------
        RuntimeError
            If weights have not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Weights not fitted.  Call fit_weights() first, or use "
                "simple_average which requires no observed data."
            )

        # Align all prediction matrices to common index/columns.
        aligned = self._align_predictions()

        # Weighted sum.
        blended = np.zeros_like(aligned[0].values, dtype=np.float64)
        for i, (name, arr) in enumerate(zip(self._model_names, aligned)):
            blended += self._weights[i] * arr.values

        result = pd.DataFrame(
            blended,
            index=aligned[0].index,
            columns=aligned[0].columns,
        )

        if self.normalize:
            result = self._row_normalize(result)

        return result

    def prediction_intervals(
        self, confidence: float = 0.95
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Uncertainty intervals from model disagreement.

        For each origin-store cell, the interval is derived from the
        spread of individual model predictions weighted by their
        ensemble weights.  Under a normal approximation:

            lower = mean - z * weighted_std
            upper = mean + z * weighted_std

        Parameters
        ----------
        confidence : float, default 0.95
            Confidence level in (0, 1).

        Returns
        -------
        lower : pd.DataFrame
            Lower bound of the prediction interval.
        upper : pd.DataFrame
            Upper bound of the prediction interval.

        Raises
        ------
        RuntimeError
            If weights have not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Weights not fitted.  Call fit_weights() first.")

        from scipy.stats import norm
        z = norm.ppf(0.5 + confidence / 2.0)

        aligned = self._align_predictions()
        stack = np.stack([a.values for a in aligned], axis=0)  # (K, O, S)

        # Weighted mean.
        w = self._weights[:, np.newaxis, np.newaxis]
        mean = (w * stack).sum(axis=0)

        # Weighted standard deviation.
        diff_sq = (stack - mean[np.newaxis, :, :]) ** 2
        var = (w * diff_sq).sum(axis=0)
        std = np.sqrt(var)

        lower_arr = np.clip(mean - z * std, 0.0, 1.0)
        upper_arr = np.clip(mean + z * std, 0.0, 1.0)

        idx = aligned[0].index
        cols = aligned[0].columns
        lower = pd.DataFrame(lower_arr, index=idx, columns=cols)
        upper = pd.DataFrame(upper_arr, index=idx, columns=cols)

        return lower, upper

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def model_comparison(
        self, actual_shares: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Compare each model's accuracy metrics.

        Parameters
        ----------
        actual_shares : pd.DataFrame or None
            Observed origin x store visit shares.  If None, only weight
            information is returned (no accuracy metrics).

        Returns
        -------
        pd.DataFrame
            One row per model with columns: ``weight``, ``rmse``,
            ``mae``, ``log_likelihood``, ``bic`` (where available).
        """
        records = []
        for name in self._model_names:
            info = self._models[name]
            weight = None
            if self._fitted:
                idx = self._model_names.index(name)
                weight = float(self._weights[idx])

            row = {
                "model": name,
                "weight": weight,
                "bic": info.get("bic"),
                "n_params": info.get("n_params"),
            }

            if actual_shares is not None:
                preds = info["predictions"]
                # Align to common index/columns.
                common_idx = preds.index.intersection(actual_shares.index)
                common_cols = preds.columns.intersection(actual_shares.columns)
                p = preds.loc[common_idx, common_cols].values.flatten()
                a = actual_shares.loc[common_idx, common_cols].values.flatten()

                row["rmse"] = float(np.sqrt(mean_squared_error(a, p)))
                row["mae"] = float(mean_absolute_error(a, p))

                # Log-likelihood (multinomial form).
                p_clipped = np.clip(p, 1e-15, 1.0)
                row["log_likelihood"] = float(np.sum(a * np.log(p_clipped)))

            records.append(row)

        df = pd.DataFrame(records).set_index("model")
        if "rmse" in df.columns:
            df = df.sort_values("rmse")
        return df

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> dict[str, float]:
        """Current model weights as a dict."""
        if self._weights is None:
            return {n: (self._models[n].get("weight") or 0.0)
                    for n in self._model_names}
        return {n: float(w)
                for n, w in zip(self._model_names, self._weights)}

    @property
    def is_fitted(self) -> bool:
        """Whether weights have been fitted."""
        return self._fitted

    @property
    def n_models(self) -> int:
        """Number of registered models."""
        return len(self._models)

    def __repr__(self) -> str:
        fitted_str = f"fitted={self._fitted}"
        if self._fitted:
            fitted_str += f", method='{self._fit_method}'"
        return (
            f"EnsembleAverager(n_models={self.n_models}, {fitted_str})"
        )

    # ------------------------------------------------------------------
    # Internal: weight-fitting methods
    # ------------------------------------------------------------------

    def _fit_simple_average(self) -> np.ndarray:
        """Equal weights for all models."""
        n = len(self._model_names)
        return np.full(n, 1.0 / n)

    def _fit_bayesian(self, actual_shares: pd.DataFrame) -> np.ndarray:
        """BMA weights proportional to exp(-BIC/2) or exp(holdout LL).

        If BIC values are provided on models, uses:
            w_k proportional to exp(-BIC_k / 2)

        Otherwise, computes holdout log-likelihood from actual_shares
        and uses:
            w_k proportional to exp(LL_k)

        Parameters
        ----------
        actual_shares : pd.DataFrame
            Observed shares for holdout log-likelihood computation.

        Returns
        -------
        np.ndarray
            Normalised weight vector.
        """
        log_weights = np.zeros(len(self._model_names))

        for i, name in enumerate(self._model_names):
            info = self._models[name]

            if info.get("bic") is not None:
                # BMA standard form: weight proportional to exp(-BIC/2).
                log_weights[i] = -info["bic"] / 2.0
            elif info.get("log_likelihood") is not None:
                log_weights[i] = info["log_likelihood"]
            else:
                # Compute holdout log-likelihood.
                preds = info["predictions"]
                common_idx = preds.index.intersection(actual_shares.index)
                common_cols = preds.columns.intersection(actual_shares.columns)
                p = preds.loc[common_idx, common_cols].values
                a = actual_shares.loc[common_idx, common_cols].values
                p_clipped = np.clip(p, 1e-15, 1.0)
                log_weights[i] = float(np.sum(a * np.log(p_clipped)))

        # Normalise via log-sum-exp for numerical stability.
        log_denom = logsumexp(log_weights)
        weights = np.exp(log_weights - log_denom)

        return weights

    def _fit_stacking(
        self,
        actual_shares: pd.DataFrame,
        alpha: float = 1.0,
        cv: int = 5,
    ) -> np.ndarray:
        """Stacking meta-learner (ridge regression) on model predictions.

        Each origin-store cell is treated as an observation.  The
        feature matrix has one column per model (the model's predicted
        probability for that cell).  The target is the actual share.

        Ridge regression is used as the meta-learner with a non-negativity
        post-processing step and normalisation so weights sum to 1.

        Parameters
        ----------
        actual_shares : pd.DataFrame
            Observed origin x store shares.
        alpha : float
            Ridge regularisation strength.
        cv : int
            Number of CV folds (not currently used for the final fit,
            but reserved for future OOF stacking).

        Returns
        -------
        np.ndarray
            Normalised weight vector.
        """
        aligned = self._align_predictions()

        # Build common index/columns.
        common_idx = aligned[0].index
        common_cols = aligned[0].columns
        for a in aligned[1:]:
            common_idx = common_idx.intersection(a.index)
            common_cols = common_cols.intersection(a.columns)
        common_idx = common_idx.intersection(actual_shares.index)
        common_cols = common_cols.intersection(actual_shares.columns)

        n_cells = len(common_idx) * len(common_cols)
        K = len(self._model_names)

        # Feature matrix: (n_cells, K).
        X = np.zeros((n_cells, K))
        for k, a in enumerate(aligned):
            X[:, k] = a.loc[common_idx, common_cols].values.flatten()

        y = actual_shares.loc[common_idx, common_cols].values.flatten()

        # Fit ridge regression.
        meta = Ridge(alpha=alpha, fit_intercept=False)
        meta.fit(X, y)
        self._stacking_meta = meta

        # Post-process: clip negative coefficients, normalise.
        raw_weights = meta.coef_.copy()
        raw_weights = np.maximum(raw_weights, 0.0)
        total = raw_weights.sum()
        if total > 0:
            weights = raw_weights / total
        else:
            # Fallback to equal weights.
            weights = np.full(K, 1.0 / K)

        return weights

    # ------------------------------------------------------------------
    # Internal: alignment and normalisation
    # ------------------------------------------------------------------

    def _align_predictions(self) -> list[pd.DataFrame]:
        """Align all model prediction matrices to a common index/columns.

        Missing entries are filled with zero.

        Returns
        -------
        list[pd.DataFrame]
            Aligned DataFrames in model-name order.
        """
        dfs = [self._models[n]["predictions"] for n in self._model_names]

        # Union of all indices and columns.
        all_idx = dfs[0].index
        all_cols = dfs[0].columns
        for df in dfs[1:]:
            all_idx = all_idx.union(df.index)
            all_cols = all_cols.union(df.columns)

        aligned = []
        for df in dfs:
            aligned.append(
                df.reindex(index=all_idx, columns=all_cols, fill_value=0.0)
            )
        return aligned

    @staticmethod
    def _row_normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise each row to sum to 1, handling zero rows gracefully.

        Parameters
        ----------
        df : pd.DataFrame
            Probability matrix.

        Returns
        -------
        pd.DataFrame
            Row-normalised matrix.
        """
        row_sums = df.sum(axis=1)
        row_sums = row_sums.replace(0.0, 1.0)  # Avoid division by zero.
        return df.div(row_sums, axis=0)
