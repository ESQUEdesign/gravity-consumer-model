"""
Residual Boosting Model
========================
Gradient boosting on structural model prediction errors to capture
nonlinear patterns that parametric gravity models (Huff, Mixed Logit) miss.

The core idea is simple: structural models impose functional-form
assumptions (distance decay, logit choice probabilities) that work well
on average but systematically mis-predict certain origin-store pairs.
These residuals -- actual_share minus predicted_share -- are not random;
they contain signal from omitted interactions, nonlinearities, and
context effects that a flexible tree ensemble can learn.

Pipeline
--------
1. A structural model produces predicted visit shares P_hat(i,j).
2. Residuals are computed: r(i,j) = P_actual(i,j) - P_hat(i,j).
3. An XGBoost or LightGBM model is trained on features X(i,j) to
   predict r(i,j).
4. At inference, the corrected probability is:
       P_corrected(i,j) = P_hat(i,j) + r_hat(i,j)
   followed by clipping to [0, 1] and row-normalisation so that
   probabilities sum to 1 across stores for each origin.

Spatial cross-validation is used during hyperparameter tuning to
prevent geographic leakage -- nearby origins share unobserved spatial
confounders, and naive k-fold CV would overstate predictive accuracy.

References
----------
Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
    Statistical Learning*, 2nd ed. Springer. Chapter 10 (Boosting).

Roberts, D.R. et al. (2017). "Cross-validation strategies for data with
    temporal, spatial, hierarchical, or phylogenetic structure."
    *Ecography*, 40(8), 913-929.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports -- fail gracefully
# ---------------------------------------------------------------------------

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_XGB_PARAMS: dict = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 500,
    "random_state": 42,
}

_DEFAULT_LGB_PARAMS: dict = {
    "objective": "regression",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 500,
    "random_state": 42,
    "verbose": -1,
}

_CLIP_EPS: float = 1e-8  # floor for re-normalised probabilities


# ---------------------------------------------------------------------------
# Spatial fold assignment
# ---------------------------------------------------------------------------

def _spatial_folds(
    coords: np.ndarray,
    n_folds: int,
    random_state: int = 42,
) -> np.ndarray:
    """Assign observations to spatially contiguous folds via k-means.

    Clusters the (lat, lon) coordinates into ``n_folds`` groups using
    k-means so that geographically nearby origins end up in the same
    fold.  This prevents spatial leakage during cross-validation.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape ``(n_origins, 2)`` with ``[lat, lon]`` per row.
    n_folds : int
        Number of spatial folds.
    random_state : int, default 42
        Seed for k-means initialisation.

    Returns
    -------
    np.ndarray
        Integer fold labels of shape ``(n_origins,)`` in [0, n_folds).
    """
    from sklearn.cluster import KMeans

    if len(coords) < n_folds:
        # Fewer points than folds -- fall back to identity assignment.
        return np.arange(len(coords)) % n_folds

    kmeans = KMeans(n_clusters=n_folds, random_state=random_state, n_init=10)
    return kmeans.fit_predict(coords)


# ---------------------------------------------------------------------------
# ResidualBoostModel
# ---------------------------------------------------------------------------

class ResidualBoostModel:
    """Gradient boosting on structural model residuals.

    Learns what the Huff / Mixed Logit model misses by training a tree
    ensemble to predict ``actual_share - structural_predicted_share``
    from a combined feature matrix of store attributes, origin
    demographics, and interaction terms.

    Parameters
    ----------
    model_type : {"xgboost", "lightgbm"}, default "xgboost"
        Which gradient boosting backend to use.
    params : dict or None
        Custom hyperparameters passed directly to the underlying
        estimator constructor.  If ``None``, sensible defaults are used.
    add_interaction_terms : bool, default True
        If True, automatically generate pairwise interaction features
        (products of all numeric columns) before fitting.  This gives
        the tree ensemble additional signal about cross-effects that
        the structural model may have missed.

    Attributes
    ----------
    model_ : estimator or None
        The fitted gradient boosting model.  ``None`` before ``fit()``.
    feature_names_ : list[str] or None
        Column names of the feature matrix used during fitting.
    cv_results_ : dict or None
        Spatial cross-validation results from the most recent call to
        ``spatial_cross_validate()`` or ``tune_hyperparameters()``.

    Examples
    --------
    >>> booster = ResidualBoostModel(model_type="xgboost")
    >>> booster.fit(actual_shares, predicted_shares, features_df)
    >>> corrected = booster.predict(predicted_shares, features_df)
    >>> importance = booster.feature_importance()
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        params: Optional[dict] = None,
        add_interaction_terms: bool = True,
    ) -> None:
        if model_type not in ("xgboost", "lightgbm"):
            raise ValueError(
                f"model_type must be 'xgboost' or 'lightgbm', got '{model_type}'"
            )
        if model_type == "xgboost" and not _HAS_XGB:
            raise ImportError(
                "xgboost is not installed.  Install it with: pip install xgboost"
            )
        if model_type == "lightgbm" and not _HAS_LGB:
            raise ImportError(
                "lightgbm is not installed.  Install it with: pip install lightgbm"
            )

        self.model_type = model_type
        self.user_params = dict(params) if params is not None else None
        self.add_interaction_terms = add_interaction_terms

        # Populated after fit().
        self.model_ = None
        self.feature_names_: Optional[list[str]] = None
        self.cv_results_: Optional[dict] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Optionally augment the feature matrix with interaction terms.

        Interaction terms are pairwise products of all numeric columns.
        This allows the tree ensemble to split more easily on cross-
        effects (e.g. high-income origins x large stores) without
        requiring very deep trees.

        Parameters
        ----------
        features_df : pd.DataFrame
            Raw feature matrix with one row per origin-store pair.

        Returns
        -------
        pd.DataFrame
            Augmented feature matrix (original columns + interactions).
        """
        df = features_df.copy()

        if not self.add_interaction_terms:
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return df

        # Cap at 20 base columns to avoid combinatorial explosion.
        if len(numeric_cols) > 20:
            logger.info(
                "Limiting interaction terms to the first 20 numeric columns "
                "(%d available).",
                len(numeric_cols),
            )
            numeric_cols = numeric_cols[:20]

        interaction_dfs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_a = numeric_cols[i]
                col_b = numeric_cols[j]
                name = f"{col_a}__x__{col_b}"
                interaction_dfs.append(
                    pd.Series(
                        df[col_a].values * df[col_b].values,
                        index=df.index,
                        name=name,
                    )
                )

        if interaction_dfs:
            df = pd.concat([df] + interaction_dfs, axis=1)

        return df

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _make_estimator(self, params: Optional[dict] = None):
        """Instantiate the gradient boosting estimator.

        Parameters
        ----------
        params : dict or None
            Hyperparameters.  If ``None``, use ``self.user_params`` or
            built-in defaults.

        Returns
        -------
        estimator
            An unfitted XGBRegressor or LGBMRegressor.
        """
        if params is not None:
            p = dict(params)
        elif self.user_params is not None:
            p = dict(self.user_params)
        elif self.model_type == "xgboost":
            p = dict(_DEFAULT_XGB_PARAMS)
        else:
            p = dict(_DEFAULT_LGB_PARAMS)

        if self.model_type == "xgboost":
            return xgb.XGBRegressor(**p)
        else:
            return lgb.LGBMRegressor(**p)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        actual_shares: np.ndarray,
        predicted_shares: np.ndarray,
        features_df: pd.DataFrame,
        model_type: Optional[str] = None,
        *,
        early_stopping_rounds: Optional[int] = 50,
        eval_fraction: float = 0.15,
        verbose: bool = False,
    ) -> "ResidualBoostModel":
        """Train the residual boosting model.

        Parameters
        ----------
        actual_shares : np.ndarray
            Observed visit shares for each origin-store pair.  1-D array
            of length ``n`` (flattened origin x store matrix).
        predicted_shares : np.ndarray
            Structural model predictions corresponding to the same
            origin-store pairs.  Same shape as ``actual_shares``.
        features_df : pd.DataFrame
            Feature matrix with ``n`` rows.  Should contain store
            attributes, origin demographics, and any hand-crafted
            interaction columns.
        model_type : str or None
            Override the model type set at construction.  Accepts
            ``"xgboost"`` or ``"lightgbm"``.
        early_stopping_rounds : int or None, default 50
            Stop training if validation loss does not improve for this
            many rounds.  Set to ``None`` to disable.
        eval_fraction : float, default 0.15
            Fraction of data reserved for early-stopping evaluation.
            Ignored when ``early_stopping_rounds`` is ``None``.
        verbose : bool, default False
            If True, print per-round training loss.

        Returns
        -------
        ResidualBoostModel
            ``self``, now fitted.

        Raises
        ------
        ValueError
            If array lengths do not match or inputs are invalid.
        """
        actual = np.asarray(actual_shares, dtype=np.float64).ravel()
        predicted = np.asarray(predicted_shares, dtype=np.float64).ravel()

        if len(actual) != len(predicted):
            raise ValueError(
                f"actual_shares ({len(actual)}) and predicted_shares "
                f"({len(predicted)}) must have the same length."
            )
        if len(actual) != len(features_df):
            raise ValueError(
                f"features_df has {len(features_df)} rows but shares arrays "
                f"have {len(actual)} elements."
            )

        if model_type is not None:
            self.model_type = model_type
            # Re-validate availability.
            if model_type == "xgboost" and not _HAS_XGB:
                raise ImportError("xgboost is not installed.")
            if model_type == "lightgbm" and not _HAS_LGB:
                raise ImportError("lightgbm is not installed.")

        # Compute residuals.
        residuals = actual - predicted

        # Build feature matrix.
        X = self._build_features(features_df)
        self.feature_names_ = list(X.columns)

        X_np = X.values.astype(np.float64)
        y_np = residuals.astype(np.float64)

        # Create estimator.
        self.model_ = self._make_estimator()

        # Fit with optional early stopping.
        fit_kwargs: dict = {}
        if early_stopping_rounds is not None and eval_fraction > 0:
            n_eval = max(1, int(len(X_np) * eval_fraction))
            # Use the last `n_eval` rows as a hold-out set (deterministic).
            indices = np.arange(len(X_np))
            rng = np.random.RandomState(42)
            rng.shuffle(indices)
            train_idx = indices[n_eval:]
            eval_idx = indices[:n_eval]

            X_train, y_train = X_np[train_idx], y_np[train_idx]
            X_eval, y_eval = X_np[eval_idx], y_np[eval_idx]

            if self.model_type == "xgboost":
                fit_kwargs["eval_set"] = [(X_eval, y_eval)]
                fit_kwargs["verbose"] = verbose
            else:
                fit_kwargs["eval_set"] = [(X_eval, y_eval)]
                if not verbose:
                    fit_kwargs["callbacks"] = [lgb.log_evaluation(period=-1)]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_.fit(X_train, y_train, **fit_kwargs)
        else:
            self.model_.fit(X_np, y_np)

        self._fitted = True

        # Training diagnostics.
        train_pred = self.model_.predict(X_np)
        rmse = np.sqrt(np.mean((y_np - train_pred) ** 2))
        mae = np.mean(np.abs(y_np - train_pred))
        logger.info(
            "ResidualBoostModel fitted (%s): train RMSE=%.6f, MAE=%.6f, "
            "%d features, %d observations.",
            self.model_type,
            rmse,
            mae,
            len(self.feature_names_),
            len(X_np),
        )

        return self

    def predict(
        self,
        predicted_shares: np.ndarray,
        features_df: pd.DataFrame,
        *,
        origin_groups: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Produce corrected probability estimates.

        The corrected probability for each origin-store pair is:

            P_corrected(i,j) = P_structural(i,j) + r_hat(i,j)

        followed by clipping to [0, 1] and row-normalisation within each
        origin so that probabilities sum to 1.

        Parameters
        ----------
        predicted_shares : np.ndarray
            Structural model predictions, 1-D array of length ``n``.
        features_df : pd.DataFrame
            Feature matrix, same format as used in ``fit()``.
        origin_groups : np.ndarray or None
            Integer or string labels identifying which origin each row
            belongs to, of length ``n``.  Used for row-normalisation.
            If ``None``, no normalisation is applied (the raw corrected
            probabilities are returned, clipped to [0, 1]).

        Returns
        -------
        np.ndarray
            Corrected probabilities, 1-D array of length ``n``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_fitted()

        predicted = np.asarray(predicted_shares, dtype=np.float64).ravel()
        X = self._build_features(features_df)

        # Align columns to the training feature set.
        missing_cols = set(self.feature_names_) - set(X.columns)
        extra_cols = set(X.columns) - set(self.feature_names_)
        if missing_cols:
            raise ValueError(
                f"features_df is missing columns that were present during "
                f"fit(): {sorted(missing_cols)}"
            )
        X = X[self.feature_names_]

        residual_pred = self.model_.predict(X.values.astype(np.float64))
        corrected = predicted + residual_pred

        # Clip to valid probability range.
        corrected = np.clip(corrected, _CLIP_EPS, 1.0)

        # Row-normalise within each origin group.
        if origin_groups is not None:
            groups = np.asarray(origin_groups)
            if len(groups) != len(corrected):
                raise ValueError(
                    f"origin_groups ({len(groups)}) must match the number of "
                    f"predictions ({len(corrected)})."
                )
            unique_groups = np.unique(groups)
            for g in unique_groups:
                mask = groups == g
                group_sum = corrected[mask].sum()
                if group_sum > 0:
                    corrected[mask] /= group_sum

        return corrected

    def spatial_cross_validate(
        self,
        actual_shares: np.ndarray,
        predicted_shares: np.ndarray,
        features_df: pd.DataFrame,
        coords: np.ndarray,
        n_folds: int = 5,
        *,
        random_state: int = 42,
    ) -> dict:
        """Evaluate residual boosting with spatial cross-validation.

        Origins are assigned to spatially contiguous folds via k-means
        clustering on coordinates.  Each fold is held out in turn, and
        the model is trained on the remaining folds.  This prevents
        geographic leakage from nearby origins sharing unobserved
        spatial confounders.

        Parameters
        ----------
        actual_shares : np.ndarray
            Observed visit shares, length ``n``.
        predicted_shares : np.ndarray
            Structural model predictions, length ``n``.
        features_df : pd.DataFrame
            Feature matrix, ``n`` rows.
        coords : np.ndarray
            Array of shape ``(n, 2)`` with ``[lat, lon]`` for each row.
            If origins repeat across multiple stores, provide the
            origin's coordinates for each row.
        n_folds : int, default 5
            Number of spatial folds.
        random_state : int, default 42
            Seed for k-means clustering.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``"fold_rmse"`` : list of per-fold RMSE values.
            - ``"fold_mae"``  : list of per-fold MAE values.
            - ``"mean_rmse"`` : mean RMSE across folds.
            - ``"mean_mae"``  : mean MAE across folds.
            - ``"std_rmse"``  : std of RMSE across folds.
            - ``"std_mae"``   : std of MAE across folds.
            - ``"fold_labels"``: fold assignment for each observation.
        """
        actual = np.asarray(actual_shares, dtype=np.float64).ravel()
        predicted = np.asarray(predicted_shares, dtype=np.float64).ravel()
        coords_np = np.asarray(coords, dtype=np.float64)
        residuals = actual - predicted

        if coords_np.shape[0] != len(actual):
            raise ValueError(
                f"coords has {coords_np.shape[0]} rows but shares arrays "
                f"have {len(actual)} elements."
            )
        if coords_np.ndim != 2 or coords_np.shape[1] != 2:
            raise ValueError(
                f"coords must have shape (n, 2), got {coords_np.shape}."
            )

        X = self._build_features(features_df)
        X_np = X.values.astype(np.float64)
        y_np = residuals.astype(np.float64)

        fold_labels = _spatial_folds(coords_np, n_folds, random_state)

        fold_rmse = []
        fold_mae = []

        for fold_id in range(n_folds):
            test_mask = fold_labels == fold_id
            train_mask = ~test_mask

            if test_mask.sum() == 0 or train_mask.sum() == 0:
                logger.warning(
                    "Spatial fold %d is empty; skipping.", fold_id
                )
                continue

            X_train, y_train = X_np[train_mask], y_np[train_mask]
            X_test, y_test = X_np[test_mask], y_np[test_mask]

            estimator = self._make_estimator()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_train, y_train)

            y_pred = estimator.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            mae = np.mean(np.abs(y_test - y_pred))
            fold_rmse.append(float(rmse))
            fold_mae.append(float(mae))

            logger.info(
                "Spatial CV fold %d/%d: RMSE=%.6f, MAE=%.6f "
                "(train=%d, test=%d)",
                fold_id + 1,
                n_folds,
                rmse,
                mae,
                train_mask.sum(),
                test_mask.sum(),
            )

        results = {
            "fold_rmse": fold_rmse,
            "fold_mae": fold_mae,
            "mean_rmse": float(np.mean(fold_rmse)) if fold_rmse else np.nan,
            "mean_mae": float(np.mean(fold_mae)) if fold_mae else np.nan,
            "std_rmse": float(np.std(fold_rmse)) if fold_rmse else np.nan,
            "std_mae": float(np.std(fold_mae)) if fold_mae else np.nan,
            "fold_labels": fold_labels,
        }

        self.cv_results_ = results

        logger.info(
            "Spatial CV complete: RMSE=%.6f +/- %.6f, MAE=%.6f +/- %.6f",
            results["mean_rmse"],
            results["std_rmse"],
            results["mean_mae"],
            results["std_mae"],
        )

        return results

    def tune_hyperparameters(
        self,
        actual_shares: np.ndarray,
        predicted_shares: np.ndarray,
        features_df: pd.DataFrame,
        coords: np.ndarray,
        n_folds: int = 5,
        *,
        param_grid: Optional[dict] = None,
        n_iter: int = 20,
        random_state: int = 42,
    ) -> dict:
        """Tune hyperparameters using spatial cross-validation.

        Performs randomised search over the parameter grid, evaluating
        each configuration with spatial CV.  The best configuration
        (lowest mean RMSE) is used to refit the model on all data.

        Parameters
        ----------
        actual_shares : np.ndarray
            Observed visit shares, length ``n``.
        predicted_shares : np.ndarray
            Structural model predictions, length ``n``.
        features_df : pd.DataFrame
            Feature matrix, ``n`` rows.
        coords : np.ndarray
            Array of shape ``(n, 2)`` with ``[lat, lon]``.
        n_folds : int, default 5
            Number of spatial folds.
        param_grid : dict or None
            Mapping of parameter names to lists of candidate values.
            If ``None``, a sensible default grid is used.
        n_iter : int, default 20
            Number of random configurations to evaluate.
        random_state : int, default 42
            Seed for random search sampling and spatial folds.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``"best_params"``   : dict of best hyperparameters.
            - ``"best_rmse"``     : mean spatial CV RMSE for best config.
            - ``"all_results"``   : list of (params, mean_rmse) tuples.
        """
        if param_grid is None:
            param_grid = {
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [3, 4, 5, 6, 8],
                "min_child_weight": [1, 3, 5, 10]
                    if self.model_type == "xgboost"
                    else [10, 20, 30, 50],
                "subsample": [0.6, 0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
                "reg_alpha": [0.0, 0.01, 0.1, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            }

        actual = np.asarray(actual_shares, dtype=np.float64).ravel()
        predicted = np.asarray(predicted_shares, dtype=np.float64).ravel()
        coords_np = np.asarray(coords, dtype=np.float64)
        residuals = actual - predicted

        X = self._build_features(features_df)
        X_np = X.values.astype(np.float64)
        y_np = residuals.astype(np.float64)

        fold_labels = _spatial_folds(coords_np, n_folds, random_state)

        rng = np.random.RandomState(random_state)

        # Sample random configurations.
        all_results = []
        best_rmse = np.inf
        best_params = None

        for iteration in range(n_iter):
            # Sample one value per parameter.
            config = {}
            for key, candidates in param_grid.items():
                config[key] = candidates[rng.randint(len(candidates))]

            # Build full parameter dict.
            if self.model_type == "xgboost":
                full_params = dict(_DEFAULT_XGB_PARAMS)
            else:
                full_params = dict(_DEFAULT_LGB_PARAMS)
            full_params.update(config)

            # Evaluate via spatial CV.
            fold_rmses = []
            for fold_id in range(n_folds):
                test_mask = fold_labels == fold_id
                train_mask = ~test_mask
                if test_mask.sum() == 0 or train_mask.sum() == 0:
                    continue

                estimator = self._make_estimator(full_params)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(X_np[train_mask], y_np[train_mask])

                y_pred = estimator.predict(X_np[test_mask])
                fold_rmses.append(
                    float(np.sqrt(np.mean((y_np[test_mask] - y_pred) ** 2)))
                )

            mean_rmse = float(np.mean(fold_rmses)) if fold_rmses else np.inf
            all_results.append((config, mean_rmse))

            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_params = config

            logger.info(
                "Tuning iter %d/%d: RMSE=%.6f %s",
                iteration + 1,
                n_iter,
                mean_rmse,
                config,
            )

        # Refit with best params on all data.
        if best_params is not None:
            if self.model_type == "xgboost":
                final_params = dict(_DEFAULT_XGB_PARAMS)
            else:
                final_params = dict(_DEFAULT_LGB_PARAMS)
            final_params.update(best_params)

            self.model_ = self._make_estimator(final_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_.fit(X_np, y_np)
            self.feature_names_ = list(X.columns)
            self._fitted = True

            logger.info(
                "Tuning complete. Best RMSE=%.6f, params=%s. "
                "Model refitted on all data.",
                best_rmse,
                best_params,
            )

        return {
            "best_params": best_params,
            "best_rmse": best_rmse,
            "all_results": sorted(all_results, key=lambda x: x[1]),
        }

    def feature_importance(
        self,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """Return feature importance ranking.

        Parameters
        ----------
        importance_type : {"gain", "weight", "cover"}, default "gain"
            Type of importance metric.  ``"gain"`` measures the average
            improvement in the loss function from splits on this
            feature.  ``"weight"`` counts the number of times a feature
            appears in a tree.  ``"cover"`` measures the average number
            of training samples affected by splits on this feature.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``feature``, ``importance``,
            ``importance_pct``, sorted by ``importance`` descending.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_fitted()

        if self.model_type == "xgboost":
            booster = self.model_.get_booster()
            scores = booster.get_score(importance_type=importance_type)
            # XGBoost names features f0, f1, ... by default; map back.
            importance = np.zeros(len(self.feature_names_))
            for key, val in scores.items():
                idx = int(key.replace("f", ""))
                if idx < len(importance):
                    importance[idx] = val
        else:
            importance = self.model_.feature_importances_

        total = importance.sum() if importance.sum() > 0 else 1.0
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance,
            "importance_pct": importance / total * 100,
        })

        return df.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)

    def shap_explanations(
        self,
        features_df: pd.DataFrame,
        *,
        max_samples: int = 500,
    ) -> Optional[object]:
        """Compute SHAP values for model interpretability.

        SHAP (SHapley Additive exPlanations) values decompose each
        prediction into per-feature contributions, enabling
        fine-grained understanding of why the model corrects the
        structural prediction up or down for a given origin-store pair.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix to explain.  If more than ``max_samples``
            rows, a random subsample is used for computational
            efficiency.
        max_samples : int, default 500
            Maximum number of rows to compute SHAP values for.

        Returns
        -------
        shap.Explanation or None
            SHAP Explanation object with ``.values``, ``.base_values``,
            and ``.data`` attributes.  Returns ``None`` if the ``shap``
            package is not installed.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_fitted()

        if not _HAS_SHAP:
            logger.warning(
                "The 'shap' package is not installed. "
                "Install with: pip install shap"
            )
            return None

        X = self._build_features(features_df)
        X = X[self.feature_names_]

        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)

        explainer = shap.TreeExplainer(self.model_)
        explanation = explainer(X)

        logger.info(
            "SHAP explanations computed for %d samples, %d features.",
            len(X),
            len(self.feature_names_),
        )

        return explanation

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit()`` has been called successfully."""
        return self._fitted

    def residual_diagnostics(
        self,
        actual_shares: np.ndarray,
        predicted_shares: np.ndarray,
        features_df: pd.DataFrame,
    ) -> dict:
        """Compute diagnostic statistics on the model's residual corrections.

        Parameters
        ----------
        actual_shares : np.ndarray
            Observed visit shares.
        predicted_shares : np.ndarray
            Structural model predictions.
        features_df : pd.DataFrame
            Feature matrix.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``"structural_rmse"``   : RMSE of the structural model alone.
            - ``"corrected_rmse"``    : RMSE after residual correction.
            - ``"structural_mae"``    : MAE of the structural model alone.
            - ``"corrected_mae"``     : MAE after residual correction.
            - ``"improvement_pct"``   : percentage RMSE improvement.
            - ``"residual_mean"``     : mean of the fitted residuals.
            - ``"residual_std"``      : std of the fitted residuals.
        """
        self._check_fitted()

        actual = np.asarray(actual_shares, dtype=np.float64).ravel()
        predicted = np.asarray(predicted_shares, dtype=np.float64).ravel()

        corrected = self.predict(predicted, features_df)

        structural_rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        corrected_rmse = float(np.sqrt(np.mean((actual - corrected) ** 2)))
        structural_mae = float(np.mean(np.abs(actual - predicted)))
        corrected_mae = float(np.mean(np.abs(actual - corrected)))

        improvement_pct = (
            (structural_rmse - corrected_rmse) / structural_rmse * 100
            if structural_rmse > 0
            else 0.0
        )

        raw_residuals = actual - predicted
        return {
            "structural_rmse": structural_rmse,
            "corrected_rmse": corrected_rmse,
            "structural_mae": structural_mae,
            "corrected_mae": corrected_mae,
            "improvement_pct": float(improvement_pct),
            "residual_mean": float(np.mean(raw_residuals)),
            "residual_std": float(np.std(raw_residuals)),
        }

    def summary(self) -> str:
        """Human-readable summary of the model state."""
        status = "fitted" if self._fitted else "not fitted"
        lines = [
            "ResidualBoostModel",
            f"  status         : {status}",
            f"  backend        : {self.model_type}",
            f"  interactions   : {self.add_interaction_terms}",
        ]
        if self._fitted:
            lines.append(f"  n_features     : {len(self.feature_names_)}")
        if self.cv_results_ is not None:
            lines.append(
                f"  spatial CV RMSE: {self.cv_results_['mean_rmse']:.6f} "
                f"+/- {self.cv_results_['std_rmse']:.6f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ResidualBoostModel(model_type={self.model_type!r}, "
            f"fitted={self._fitted})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self._fitted or self.model_ is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before using this method."
            )
