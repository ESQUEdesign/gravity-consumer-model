"""
Count Regression Models for Consumer Visit Prediction
=====================================================
Poisson, Negative Binomial, and Zero-Inflated count regression models
that predict *visit counts* rather than visit probabilities.

This is a key extension beyond the Huff gravity framework: instead of
modelling the *share* of consumers who visit each store, count models
estimate the *number* of visits, accounting for overdispersion (via the
Negative Binomial) and excess zeros (via zero-inflation).

Supported model types
---------------------
- **Poisson**: Var(Y) = mu.  Appropriate when the mean and variance of
  counts are approximately equal.
- **Negative Binomial (NB2)**: Var(Y) = mu + mu^2/r.  Handles
  overdispersion by introducing a dispersion parameter *r*.
- **Zero-Inflated Poisson (ZIP)**: Mixture of a point mass at zero and
  a Poisson distribution.  For data with more zeros than a Poisson
  would predict.
- **Zero-Inflated Negative Binomial (ZINB)**: Mixture of a point mass
  at zero and a NB2 distribution.  For overdispersed data with excess
  zeros.

All models are estimated via Maximum Likelihood using
``scipy.optimize.minimize`` -- no dependency on statsmodels.

Feature matrix
--------------
The default feature set for the linear predictor (log-link) is:

    log(mu_ij) = beta_0
                 + beta_1 * log(distance_ij)
                 + beta_2 * log(square_footage_j)
                 + beta_3 * log(population_i)
                 + beta_4 * median_income_i
                 + sum_c beta_c * I(category_j == c)
                 + log(exposure_i)          [offset, coefficient fixed at 1]

References
----------
Cameron, A.C. & Trivedi, P.K. (2013). *Regression Analysis of Count Data*,
    2nd ed. Cambridge University Press.

Lambert, D. (1992). "Zero-Inflated Poisson Regression, with an
    Application to Defects in Manufacturing." *Technometrics*, 34(1), 1-14.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from gravity.data.schema import build_distance_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-10
_MIN_DISTANCE_KM: float = 0.01
_MAX_MU: float = 1e8  # cap predicted mean to prevent overflow


# ---------------------------------------------------------------------------
# CountModel
# ---------------------------------------------------------------------------

class CountModel:
    """Count regression model for consumer visit prediction.

    Estimates the expected number of visits from each origin to each store,
    as opposed to visit probabilities.  Supports Poisson, Negative Binomial,
    and their Zero-Inflated variants.

    Parameters
    ----------
    model_type : str, default "negative_binomial"
        Base count distribution: ``"poisson"`` or ``"negative_binomial"``.
    zero_inflated : bool, default False
        If ``True``, estimates a zero-inflated version of the base
        distribution.  The zero-inflation probability ``pi`` is estimated
        as an additional parameter.

    Examples
    --------
    >>> model = CountModel(model_type="negative_binomial", zero_inflated=True)
    >>> model.fit(origins_df, stores_df, distance_matrix)
    >>> counts = model.predict(origins_df, stores_df, distance_matrix)
    >>> probs = model.predict_proba(origins_df, stores_df, distance_matrix)
    >>> lower, upper = model.predict_interval(origins_df, stores_df, distance_matrix)
    """

    _VALID_TYPES = {"poisson", "negative_binomial"}

    def __init__(
        self,
        model_type: str = "negative_binomial",
        zero_inflated: bool = False,
    ) -> None:
        if model_type not in self._VALID_TYPES:
            raise ValueError(
                f"model_type must be one of {sorted(self._VALID_TYPES)}, "
                f"got '{model_type}'."
            )
        self.model_type = model_type
        self.zero_inflated = zero_inflated

        # Populated after fit().
        self._fitted: bool = False
        self._beta: Optional[np.ndarray] = None  # regression coefficients
        self._dispersion: Optional[float] = None  # NB dispersion param r
        self._zi_prob: Optional[float] = None  # zero-inflation probability pi
        self._feature_names: list[str] = []
        self._n_features: int = 0
        self._aic: float = 0.0
        self._bic: float = 0.0
        self._n_obs: int = 0
        self._log_likelihood: float = 0.0
        self._distance_matrix: Optional[pd.DataFrame] = None
        self._category_levels: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        distance_matrix: pd.DataFrame,
        observed_counts: Optional[pd.DataFrame] = None,
        exposure: Optional[pd.Series] = None,
    ) -> "CountModel":
        """Fit the count model via Maximum Likelihood Estimation.

        Parameters
        ----------
        origins_df : pd.DataFrame
            Origins indexed by ``origin_id`` with columns including
            ``population`` and ``median_income``.
        stores_df : pd.DataFrame
            Stores indexed by ``store_id`` with columns including
            ``square_footage`` and ``category``.
        distance_matrix : pd.DataFrame
            Origin x store distance matrix (km).
        observed_counts : pd.DataFrame or None
            Origin x store matrix of actual visit counts.  If ``None``,
            synthetic counts are generated from ``population * probability``
            using a basic Huff-like probability and Poisson sampling.
        exposure : pd.Series or None
            Population vector used as an offset (``log(exposure)`` is added
            to the linear predictor with coefficient fixed at 1).
            If ``None``, defaults to ``origins_df["population"]``.

        Returns
        -------
        CountModel
            ``self``, with fitted parameters accessible via ``summary``.

        Raises
        ------
        ValueError
            If required columns are missing or data shapes are incompatible.
        """
        self._distance_matrix = distance_matrix
        self._validate_inputs(origins_df, stores_df)

        # ----- Build feature matrix and count vector -----
        X, y, offset, feature_names = self._prepare_data(
            origins_df, stores_df, distance_matrix,
            observed_counts, exposure,
        )

        self._feature_names = feature_names
        self._n_features = X.shape[1]
        self._n_obs = len(y)

        # ----- Determine number of parameters and pack initial guess -----
        n_beta = self._n_features
        # Parameters: [beta_0, beta_1, ..., beta_k, (log_r), (logit_pi)]
        n_params = n_beta
        if self.model_type == "negative_binomial":
            n_params += 1  # log(r) for dispersion
        if self.zero_inflated:
            n_params += 1  # logit(pi) for zero-inflation

        theta0 = np.zeros(n_params)
        # Start log(r) at a moderate value (r=1).
        if self.model_type == "negative_binomial":
            theta0[n_beta] = 0.0  # log(1) = 0
        # Start logit(pi) at a value giving pi ~= 0.1.
        if self.zero_inflated:
            zi_idx = n_beta + (1 if self.model_type == "negative_binomial" else 0)
            theta0[zi_idx] = -2.2  # logit(0.1) ~ -2.2

        # ----- MLE -----
        logger.info(
            "Fitting %s%s model: %d observations, %d features.",
            "zero-inflated " if self.zero_inflated else "",
            self.model_type,
            self._n_obs,
            self._n_features,
        )

        result = minimize(
            self._neg_log_likelihood,
            theta0,
            args=(X, y, offset),
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-10},
        )

        if not result.success:
            logger.warning(
                "MLE optimisation did not converge: %s", result.message
            )

        # ----- Unpack results -----
        theta = result.x
        self._beta = theta[:n_beta]

        idx = n_beta
        if self.model_type == "negative_binomial":
            self._dispersion = np.exp(theta[idx])  # r
            idx += 1
        else:
            self._dispersion = None

        if self.zero_inflated:
            self._zi_prob = 1.0 / (1.0 + np.exp(-theta[idx]))  # sigmoid
        else:
            self._zi_prob = None

        # ----- Goodness of fit -----
        self._log_likelihood = -result.fun
        n_total_params = n_params
        self._aic = 2.0 * n_total_params - 2.0 * self._log_likelihood
        self._bic = (
            np.log(self._n_obs) * n_total_params - 2.0 * self._log_likelihood
        )
        self._fitted = True

        logger.info(
            "Model fitted: LL=%.2f, AIC=%.2f, BIC=%.2f",
            self._log_likelihood,
            self._aic,
            self._bic,
        )
        if self._dispersion is not None:
            logger.info("  dispersion (r): %.4f", self._dispersion)
        if self._zi_prob is not None:
            logger.info("  zero-inflation (pi): %.4f", self._zi_prob)

        return self

    def predict(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        distance_matrix: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Predict expected visit counts for each origin-store pair.

        Parameters
        ----------
        origins_df : pd.DataFrame
            Origins indexed by ``origin_id``.
        stores_df : pd.DataFrame
            Stores indexed by ``store_id``.
        distance_matrix : pd.DataFrame or None
            Origin x store distance matrix (km).  If ``None``, uses the
            distance matrix from ``fit()`` or computes from lat/lon.

        Returns
        -------
        pd.DataFrame
            Origin x store DataFrame of predicted visit counts.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted or self._beta is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict()."
            )

        dist = self._resolve_distance_matrix(
            origins_df, stores_df, distance_matrix
        )
        X, offset = self._build_prediction_features(
            origins_df, stores_df, dist
        )

        mu = self._compute_mu(X, offset)

        # For zero-inflated models, expected count = (1 - pi) * mu.
        if self.zero_inflated and self._zi_prob is not None:
            expected = (1.0 - self._zi_prob) * mu
        else:
            expected = mu

        return pd.DataFrame(
            expected.reshape(len(origins_df), len(stores_df)),
            index=dist.index,
            columns=dist.columns,
        )

    def predict_proba(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        distance_matrix: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Predict visit probabilities (counts normalised per origin row).

        This provides output compatible with the Huff model's probability
        matrix by dividing each origin's predicted counts by the row total.

        Parameters
        ----------
        origins_df : pd.DataFrame
            Origins indexed by ``origin_id``.
        stores_df : pd.DataFrame
            Stores indexed by ``store_id``.
        distance_matrix : pd.DataFrame or None
            Origin x store distance matrix (km).

        Returns
        -------
        pd.DataFrame
            Origin x store probability matrix where each row sums to 1.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        counts = self.predict(origins_df, stores_df, distance_matrix)
        row_sums = counts.sum(axis=1)
        # Guard against zero row sums.
        row_sums = row_sums.replace(0.0, _EPSILON)
        probs = counts.div(row_sums, axis=0)
        return probs

    def predict_interval(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        distance_matrix: Optional[pd.DataFrame] = None,
        confidence: float = 0.90,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Predict confidence intervals for visit counts.

        Uses the appropriate distribution's quantile function:
        - Poisson: quantiles of Poisson(mu)
        - Negative Binomial: quantiles of NB(r, p) where p = r/(r+mu)
        - Zero-inflated: mixture quantiles computed from the base
          distribution with the zero-inflation mass.

        Parameters
        ----------
        origins_df : pd.DataFrame
            Origins indexed by ``origin_id``.
        stores_df : pd.DataFrame
            Stores indexed by ``store_id``.
        distance_matrix : pd.DataFrame or None
            Origin x store distance matrix (km).
        confidence : float, default 0.90
            Confidence level for the interval (e.g. 0.90 for 90%).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(lower, upper)`` DataFrames of count bounds.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If confidence is not in (0, 1).
        """
        if not self._fitted or self._beta is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict_interval()."
            )
        if not 0.0 < confidence < 1.0:
            raise ValueError(
                f"confidence must be in (0, 1), got {confidence}."
            )

        try:
            from scipy.stats import poisson as poisson_dist
            from scipy.stats import nbinom as nbinom_dist
        except ImportError as exc:
            raise ImportError(
                "scipy.stats is required for predict_interval."
            ) from exc

        dist = self._resolve_distance_matrix(
            origins_df, stores_df, distance_matrix
        )
        X, offset = self._build_prediction_features(
            origins_df, stores_df, dist
        )
        mu = self._compute_mu(X, offset)

        alpha_lower = (1.0 - confidence) / 2.0
        alpha_upper = 1.0 - alpha_lower

        n_origins = len(origins_df)
        n_stores = len(stores_df)

        if self.model_type == "poisson":
            lower_vals = poisson_dist.ppf(alpha_lower, mu)
            upper_vals = poisson_dist.ppf(alpha_upper, mu)
        else:
            # Negative Binomial parameterisation: NB(r, p) where p = r/(r+mu).
            r = self._dispersion if self._dispersion is not None else 1.0
            r = max(r, _EPSILON)
            p = r / (r + mu)
            p = np.clip(p, _EPSILON, 1.0 - _EPSILON)
            lower_vals = nbinom_dist.ppf(alpha_lower, r, p)
            upper_vals = nbinom_dist.ppf(alpha_upper, r, p)

        if self.zero_inflated and self._zi_prob is not None:
            # For zero-inflated models, adjust the quantiles.
            # P(Y <= k) = pi + (1 - pi) * P_base(Y <= k)
            # So the effective CDF at any k is inflated.
            # We solve for quantiles of the mixture distribution.
            pi = self._zi_prob

            # Adjusted lower quantile: find k such that
            # pi + (1-pi)*F_base(k) >= alpha_lower
            # => F_base(k) >= (alpha_lower - pi) / (1 - pi)
            adj_alpha_lower = max((alpha_lower - pi) / (1.0 - pi), 0.0)
            adj_alpha_upper = max((alpha_upper - pi) / (1.0 - pi), 0.0)

            if self.model_type == "poisson":
                lower_vals = poisson_dist.ppf(adj_alpha_lower, mu)
                upper_vals = poisson_dist.ppf(adj_alpha_upper, mu)
            else:
                r = self._dispersion if self._dispersion is not None else 1.0
                r = max(r, _EPSILON)
                p = r / (r + mu)
                p = np.clip(p, _EPSILON, 1.0 - _EPSILON)
                lower_vals = nbinom_dist.ppf(adj_alpha_lower, r, p)
                upper_vals = nbinom_dist.ppf(adj_alpha_upper, r, p)

        # Ensure non-negative integer counts.
        lower_vals = np.maximum(lower_vals, 0.0)
        upper_vals = np.maximum(upper_vals, 0.0)

        lower_df = pd.DataFrame(
            lower_vals.reshape(n_origins, n_stores),
            index=dist.index,
            columns=dist.columns,
        )
        upper_df = pd.DataFrame(
            upper_vals.reshape(n_origins, n_stores),
            index=dist.index,
            columns=dist.columns,
        )

        return lower_df, upper_df

    @property
    def summary(self) -> dict:
        """Model summary including coefficients, AIC, BIC, and distributional parameters.

        Returns
        -------
        dict
            Keys: ``"model_type"``, ``"zero_inflated"``, ``"feature_names"``,
            ``"coefficients"``, ``"aic"``, ``"bic"``, ``"log_likelihood"``,
            ``"n_obs"``, ``"dispersion"`` (NB only), ``"zero_inflation_prob"``
            (ZI only).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before accessing summary."
            )

        result = {
            "model_type": self.model_type,
            "zero_inflated": self.zero_inflated,
            "feature_names": list(self._feature_names),
            "coefficients": dict(zip(
                ["intercept"] + self._feature_names,
                self._beta.tolist(),
            )),
            "aic": self._aic,
            "bic": self._bic,
            "log_likelihood": self._log_likelihood,
            "n_obs": self._n_obs,
        }

        if self._dispersion is not None:
            result["dispersion"] = self._dispersion
        if self._zi_prob is not None:
            result["zero_inflation_prob"] = self._zi_prob

        return result

    # ------------------------------------------------------------------
    # Diagnostics / introspection
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit()`` has been called successfully."""
        return self._fitted

    def summary_text(self) -> str:
        """Human-readable summary of the model state."""
        zi_label = "Zero-Inflated " if self.zero_inflated else ""
        type_label = (
            "Negative Binomial" if self.model_type == "negative_binomial"
            else "Poisson"
        )
        status = "fitted" if self._fitted else "not fitted"

        lines = [
            f"CountModel: {zi_label}{type_label}",
            f"  status          : {status}",
        ]

        if self._fitted:
            lines.append(f"  log-likelihood  : {self._log_likelihood:.2f}")
            lines.append(f"  AIC             : {self._aic:.2f}")
            lines.append(f"  BIC             : {self._bic:.2f}")
            lines.append(f"  n_obs           : {self._n_obs}")
            if self._dispersion is not None:
                lines.append(f"  dispersion (r)  : {self._dispersion:.4f}")
            if self._zi_prob is not None:
                lines.append(f"  zero-inflation  : {self._zi_prob:.4f}")
            lines.append("")
            lines.append("  Coefficients:")
            names = ["intercept"] + self._feature_names
            for i, name in enumerate(names):
                lines.append(f"    {name:25s}  {self._beta[i]:+.6f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        zi_str = ", zero_inflated=True" if self.zero_inflated else ""
        return (
            f"CountModel(model_type='{self.model_type}'{zi_str}, "
            f"fitted={self._fitted})"
        )

    # ------------------------------------------------------------------
    # Internal: Data preparation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self, origins_df: pd.DataFrame, stores_df: pd.DataFrame
    ) -> None:
        """Check that required columns exist.

        Parameters
        ----------
        origins_df : pd.DataFrame
        stores_df : pd.DataFrame

        Raises
        ------
        ValueError
            If any required column is missing.
        """
        origin_required = {"population", "median_income"}
        store_required = {"square_footage"}
        missing_origin = origin_required - set(origins_df.columns)
        missing_store = store_required - set(stores_df.columns)
        if missing_origin:
            raise ValueError(
                f"origins_df is missing required columns: {sorted(missing_origin)}"
            )
        if missing_store:
            raise ValueError(
                f"stores_df is missing required columns: {sorted(missing_store)}"
            )

    def _prepare_data(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        distance_matrix: pd.DataFrame,
        observed_counts: Optional[pd.DataFrame],
        exposure: Optional[pd.Series],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Build the feature matrix, count vector, and offset for MLE.

        Features (log-link):
            intercept, log(distance), log(square_footage),
            log(population), median_income, [category dummies]

        Parameters
        ----------
        origins_df : pd.DataFrame
        stores_df : pd.DataFrame
        distance_matrix : pd.DataFrame
        observed_counts : pd.DataFrame or None
        exposure : pd.Series or None

        Returns
        -------
        X : np.ndarray
            Feature matrix ``(n_obs, n_features)`` including intercept.
        y : np.ndarray
            Count vector ``(n_obs,)``.
        offset : np.ndarray
            Log-exposure offset ``(n_obs,)``.
        feature_names : list[str]
            Names of features (excluding intercept which is implicit).
        """
        n_origins = len(origins_df)
        n_stores = len(stores_df)
        n_obs = n_origins * n_stores

        # ----- Distances -----
        dist_vals = distance_matrix.reindex(
            index=origins_df.index, columns=stores_df.index
        ).values.astype(np.float64)
        dist_vals = np.maximum(dist_vals, _MIN_DISTANCE_KM)
        log_dist = np.log(dist_vals).ravel()

        # ----- Store attributes -----
        sq_ft = stores_df.reindex(distance_matrix.columns)["square_footage"].values.astype(np.float64)
        sq_ft = np.maximum(sq_ft, _EPSILON)
        log_sq_ft = np.tile(np.log(sq_ft), n_origins)

        # ----- Origin attributes -----
        pop = origins_df["population"].values.astype(np.float64)
        pop = np.maximum(pop, 1.0)
        log_pop = np.repeat(np.log(pop), n_stores)

        med_income = origins_df["median_income"].values.astype(np.float64)
        # Standardise median_income to prevent numerical issues.
        income_mean = med_income.mean() if med_income.std() > 0 else 0.0
        income_std = med_income.std() if med_income.std() > 0 else 1.0
        med_income_std = (med_income - income_mean) / income_std
        med_income_long = np.repeat(med_income_std, n_stores)

        feature_names = ["log_distance", "log_square_footage", "log_population",
                         "median_income"]
        feature_cols = [log_dist, log_sq_ft, log_pop, med_income_long]

        # ----- Category dummies -----
        if "category" in stores_df.columns:
            categories = stores_df["category"].fillna("unknown").values
            unique_cats = sorted(set(categories))
            self._category_levels = unique_cats

            if len(unique_cats) > 1:
                # Drop the first category (reference level).
                for cat in unique_cats[1:]:
                    dummy = (categories == cat).astype(np.float64)
                    dummy_long = np.tile(dummy, n_origins)
                    feature_cols.append(dummy_long)
                    feature_names.append(f"category_{cat}")
        else:
            self._category_levels = None

        # ----- Assemble X with intercept -----
        intercept = np.ones(n_obs)
        X = np.column_stack([intercept] + feature_cols)

        # ----- Counts (dependent variable) -----
        if observed_counts is not None:
            y = observed_counts.reindex(
                index=distance_matrix.index, columns=distance_matrix.columns
            ).values.astype(np.float64).ravel()
            if np.isnan(y).any():
                raise ValueError(
                    "observed_counts contains NaN after alignment."
                )
        else:
            # Generate synthetic counts: population_i * P(i->j) via simple
            # inverse-distance weighting, then Poisson-sample.
            inv_dist = 1.0 / dist_vals
            row_sums = inv_dist.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, _EPSILON)
            probs = inv_dist / row_sums
            pop_col = origins_df["population"].values.astype(np.float64)
            expected = probs * pop_col[:, np.newaxis]
            rng = np.random.default_rng(42)
            y = rng.poisson(np.maximum(expected, _EPSILON)).astype(np.float64).ravel()
            logger.info(
                "No observed_counts provided; generated synthetic counts "
                "(mean=%.1f, max=%.0f).",
                y.mean(),
                y.max(),
            )

        # Ensure non-negative.
        y = np.maximum(y, 0.0)

        # ----- Exposure offset -----
        if exposure is not None:
            exp_vals = exposure.reindex(origins_df.index).values.astype(np.float64)
        else:
            exp_vals = origins_df["population"].values.astype(np.float64)
        exp_vals = np.maximum(exp_vals, 1.0)
        offset = np.repeat(np.log(exp_vals), n_stores)

        return X, y, offset, feature_names

    def _build_prediction_features(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        distance_matrix: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix and offset for prediction (no counts needed).

        Parameters
        ----------
        origins_df : pd.DataFrame
        stores_df : pd.DataFrame
        distance_matrix : pd.DataFrame

        Returns
        -------
        X : np.ndarray
            Feature matrix ``(n_obs, n_features)`` including intercept.
        offset : np.ndarray
            Log-exposure offset ``(n_obs,)``.
        """
        n_origins = len(origins_df)
        n_stores = len(stores_df)
        n_obs = n_origins * n_stores

        dist_vals = distance_matrix.reindex(
            index=origins_df.index, columns=stores_df.index
        ).values.astype(np.float64)
        dist_vals = np.maximum(dist_vals, _MIN_DISTANCE_KM)
        log_dist = np.log(dist_vals).ravel()

        sq_ft = stores_df.reindex(distance_matrix.columns)["square_footage"].values.astype(np.float64)
        sq_ft = np.maximum(sq_ft, _EPSILON)
        log_sq_ft = np.tile(np.log(sq_ft), n_origins)

        pop = origins_df["population"].values.astype(np.float64)
        pop = np.maximum(pop, 1.0)
        log_pop = np.repeat(np.log(pop), n_stores)

        med_income = origins_df["median_income"].values.astype(np.float64)
        income_mean = med_income.mean() if med_income.std() > 0 else 0.0
        income_std = med_income.std() if med_income.std() > 0 else 1.0
        med_income_std = (med_income - income_mean) / income_std
        med_income_long = np.repeat(med_income_std, n_stores)

        feature_cols = [log_dist, log_sq_ft, log_pop, med_income_long]

        # Category dummies -- must match the levels from fit().
        if self._category_levels is not None and len(self._category_levels) > 1:
            if "category" in stores_df.columns:
                categories = stores_df["category"].fillna("unknown").values
                for cat in self._category_levels[1:]:
                    dummy = (categories == cat).astype(np.float64)
                    dummy_long = np.tile(dummy, n_origins)
                    feature_cols.append(dummy_long)
            else:
                # If category column is absent at prediction time, fill with zeros.
                for _ in self._category_levels[1:]:
                    feature_cols.append(np.zeros(n_obs))

        intercept = np.ones(n_obs)
        X = np.column_stack([intercept] + feature_cols)

        # Offset.
        exp_vals = origins_df["population"].values.astype(np.float64)
        exp_vals = np.maximum(exp_vals, 1.0)
        offset = np.repeat(np.log(exp_vals), n_stores)

        return X, offset

    # ------------------------------------------------------------------
    # Internal: Log-likelihood functions
    # ------------------------------------------------------------------

    def _neg_log_likelihood(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        offset: np.ndarray,
    ) -> float:
        """Compute the negative log-likelihood for the current model type.

        Parameters
        ----------
        theta : np.ndarray
            Flat parameter vector:
            [beta_0, ..., beta_k, (log_r), (logit_pi)]
        X : np.ndarray
            Feature matrix ``(n_obs, n_features)``.
        y : np.ndarray
            Count vector ``(n_obs,)``.
        offset : np.ndarray
            Log-exposure offset ``(n_obs,)``.

        Returns
        -------
        float
            Negative log-likelihood (scalar to minimise).
        """
        n_beta = X.shape[1]
        beta = theta[:n_beta]

        idx = n_beta
        if self.model_type == "negative_binomial":
            log_r = theta[idx]
            r = np.exp(np.clip(log_r, -20, 20))  # clamp for numerical safety
            idx += 1
        else:
            r = None

        if self.zero_inflated:
            logit_pi = theta[idx]
            logit_pi = np.clip(logit_pi, -20, 20)
            pi = 1.0 / (1.0 + np.exp(-logit_pi))
        else:
            pi = 0.0

        # Linear predictor.
        eta = X @ beta + offset
        eta = np.clip(eta, -30, 30)  # prevent overflow in exp
        mu = np.exp(eta)
        mu = np.minimum(mu, _MAX_MU)

        if self.model_type == "poisson":
            ll = self._poisson_log_likelihood(y, mu, pi)
        else:
            ll = self._nb_log_likelihood(y, mu, r, pi)

        # Return negative for minimisation; guard against NaN.
        result = -ll
        if np.isnan(result) or np.isinf(result):
            return 1e15
        return result

    @staticmethod
    def _poisson_log_likelihood(
        y: np.ndarray, mu: np.ndarray, pi: float = 0.0
    ) -> float:
        """Poisson (or Zero-Inflated Poisson) log-likelihood.

        Poisson:
            log L = SUM [ y*log(mu) - mu - log(y!) ]

        Zero-Inflated Poisson:
            P(Y=0) = pi + (1-pi) * exp(-mu)
            P(Y=k) = (1-pi) * exp(-mu) * mu^k / k!   for k > 0

        Parameters
        ----------
        y : np.ndarray
            Observed counts.
        mu : np.ndarray
            Predicted means.
        pi : float
            Zero-inflation probability (0 for standard Poisson).

        Returns
        -------
        float
            Log-likelihood.
        """
        mu = np.maximum(mu, _EPSILON)

        if pi > _EPSILON:
            # Zero-inflated Poisson.
            ll = 0.0
            zero_mask = y == 0
            nonzero_mask = ~zero_mask

            if zero_mask.any():
                # P(Y=0) = pi + (1-pi)*exp(-mu)
                p_zero = pi + (1.0 - pi) * np.exp(-mu[zero_mask])
                p_zero = np.maximum(p_zero, _EPSILON)
                ll += np.sum(np.log(p_zero))

            if nonzero_mask.any():
                # P(Y=k) = (1-pi) * Poisson(k; mu)
                mu_nz = mu[nonzero_mask]
                y_nz = y[nonzero_mask]
                log_poisson = y_nz * np.log(mu_nz) - mu_nz - gammaln(y_nz + 1)
                ll += np.sum(np.log(1.0 - pi) + log_poisson)

            return ll
        else:
            # Standard Poisson.
            return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))

    @staticmethod
    def _nb_log_likelihood(
        y: np.ndarray, mu: np.ndarray, r: float, pi: float = 0.0
    ) -> float:
        """Negative Binomial (or Zero-Inflated NB) log-likelihood.

        NB2 parameterisation:
            P(Y=k) = Gamma(k+r)/(Gamma(r)*k!) * (r/(r+mu))^r * (mu/(r+mu))^k

        Log-likelihood:
            log L = SUM [ log Gamma(y+r) - log Gamma(r) - log(y!)
                         + r*log(r/(r+mu)) + y*log(mu/(r+mu)) ]

        Parameters
        ----------
        y : np.ndarray
            Observed counts.
        mu : np.ndarray
            Predicted means.
        r : float
            Dispersion parameter (size).  Larger r -> closer to Poisson.
        pi : float
            Zero-inflation probability (0 for standard NB).

        Returns
        -------
        float
            Log-likelihood.
        """
        mu = np.maximum(mu, _EPSILON)
        r = max(r, _EPSILON)

        p = r / (r + mu)  # success probability in scipy parameterisation
        p = np.clip(p, _EPSILON, 1.0 - _EPSILON)

        if pi > _EPSILON:
            # Zero-inflated Negative Binomial.
            ll = 0.0
            zero_mask = y == 0
            nonzero_mask = ~zero_mask

            if zero_mask.any():
                # P(Y=0|NB) = (r/(r+mu))^r
                p_zero_nb = np.exp(r * np.log(p[zero_mask]))
                p_zero = pi + (1.0 - pi) * p_zero_nb
                p_zero = np.maximum(p_zero, _EPSILON)
                ll += np.sum(np.log(p_zero))

            if nonzero_mask.any():
                mu_nz = mu[nonzero_mask]
                y_nz = y[nonzero_mask]
                p_nz = p[nonzero_mask]
                log_nb = (
                    gammaln(y_nz + r) - gammaln(r) - gammaln(y_nz + 1)
                    + r * np.log(p_nz)
                    + y_nz * np.log(1.0 - p_nz)
                )
                ll += np.sum(np.log(1.0 - pi) + log_nb)

            return ll
        else:
            # Standard Negative Binomial.
            log_nb = (
                gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                + r * np.log(p)
                + y * np.log(1.0 - p)
            )
            return float(np.sum(log_nb))

    # ------------------------------------------------------------------
    # Internal: Prediction helpers
    # ------------------------------------------------------------------

    def _compute_mu(self, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """Compute predicted means from the linear predictor.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix ``(n_obs, n_features)``.
        offset : np.ndarray
            Log-exposure offset ``(n_obs,)``.

        Returns
        -------
        np.ndarray
            Predicted means ``(n_obs,)``.
        """
        eta = X @ self._beta + offset
        eta = np.clip(eta, -30, 30)
        mu = np.exp(eta)
        return np.minimum(mu, _MAX_MU)

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
        """
        if distance_matrix is not None:
            return distance_matrix
        if self._distance_matrix is not None:
            return self._distance_matrix
        logger.info("No distance matrix provided; computing from lat/lon.")
        return build_distance_matrix(origins_df, stores_df)
