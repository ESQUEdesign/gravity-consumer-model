"""
Customer Lifetime Value Estimator
==================================
BG/NBD + Gamma-Gamma model for Customer Lifetime Value estimation.

The BG/NBD (Beta-Geometric / Negative Binomial Distribution) model
estimates the expected number of future purchases and the probability
that a customer is still "alive" (active).  The Gamma-Gamma model
estimates expected average monetary value conditional on being alive.

Together, they produce a discounted CLV forecast:

    CLV_i = E[M_i] * E[X_i(t)] * margin / discount_factor

References
----------
Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). "Counting Your
    Customers the Easy Way: An Alternative to the Pareto/NBD Model."
    *Marketing Science*, 24(2), 275-284.

Fader, P. S. & Hardie, B. G. S. (2013). "The Gamma-Gamma Model of
    Monetary Value." http://www.brucehardie.com/notes/025/
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, betaln

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PARAM_BOUNDS_BGNBD = [(1e-4, 1e4)] * 4  # r, alpha, a, b
_PARAM_BOUNDS_GG = [(1e-4, 1e4)] * 3     # p, q, v


class CLVEstimator:
    """BG/NBD + Gamma-Gamma Customer Lifetime Value estimator.

    Estimates purchase frequency, alive probability, expected monetary
    value, and discounted Customer Lifetime Value from transaction data.

    Parameters
    ----------
    penalizer_coef : float, default 0.0
        L2 regularisation penalty on model parameters during MLE.
        Helps with convergence on sparse data.

    Attributes
    ----------
    bgnbd_params_ : dict or None
        Fitted BG/NBD parameters: ``{"r", "alpha", "a", "b"}``.
    gg_params_ : dict or None
        Fitted Gamma-Gamma parameters: ``{"p", "q", "v"}``.
    customer_data_ : pd.DataFrame or None
        Processed customer-level summary (frequency, recency, T,
        monetary_value) from the last ``fit()`` call.

    Examples
    --------
    >>> estimator = CLVEstimator()
    >>> estimator.fit(transactions_df)
    >>> clv = estimator.predict_clv(time_horizon_days=365, discount_rate=0.1)
    >>> tiers = estimator.segment_by_clv(n_tiers=5)
    """

    def __init__(self, penalizer_coef: float = 0.0) -> None:
        self.penalizer_coef = penalizer_coef

        # Populated after fit()
        self.bgnbd_params_: Optional[dict] = None
        self.gg_params_: Optional[dict] = None
        self.customer_data_: Optional[pd.DataFrame] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        transactions_df: pd.DataFrame,
        consumer_id_col: str = "consumer_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
    ) -> "CLVEstimator":
        """Fit the BG/NBD and Gamma-Gamma models from transaction data.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Transaction-level data with at least ``consumer_id``,
            ``timestamp``, and ``amount`` columns.
        consumer_id_col : str, default "consumer_id"
            Column name for consumer identifiers.
        timestamp_col : str, default "timestamp"
            Column name for transaction timestamps.
        amount_col : str, default "amount"
            Column name for transaction amounts.

        Returns
        -------
        CLVEstimator
            Fitted estimator (``self``).

        Raises
        ------
        ValueError
            If required columns are missing or data is insufficient.
        """
        required = {consumer_id_col, timestamp_col, amount_col}
        missing = required - set(transactions_df.columns)
        if missing:
            raise ValueError(
                f"transactions_df is missing required columns: {missing}"
            )

        # Build customer summary
        customer_data = self._build_customer_summary(
            transactions_df, consumer_id_col, timestamp_col, amount_col
        )
        self.customer_data_ = customer_data

        # Fit BG/NBD
        self.bgnbd_params_ = self._fit_bgnbd(customer_data)
        logger.info(
            "BG/NBD fitted: r=%.4f, alpha=%.4f, a=%.4f, b=%.4f",
            self.bgnbd_params_["r"],
            self.bgnbd_params_["alpha"],
            self.bgnbd_params_["a"],
            self.bgnbd_params_["b"],
        )

        # Fit Gamma-Gamma (only on customers with frequency > 0)
        repeat_customers = customer_data[customer_data["frequency"] > 0]
        if len(repeat_customers) > 0:
            self.gg_params_ = self._fit_gamma_gamma(repeat_customers)
            logger.info(
                "Gamma-Gamma fitted: p=%.4f, q=%.4f, v=%.4f",
                self.gg_params_["p"],
                self.gg_params_["q"],
                self.gg_params_["v"],
            )
        else:
            logger.warning(
                "No repeat customers found; Gamma-Gamma model cannot be "
                "fitted. Monetary predictions will use observed averages."
            )
            self.gg_params_ = None

        self._fitted = True
        logger.info(
            "CLVEstimator fitted on %d customers (%d repeat purchasers)",
            len(customer_data),
            len(repeat_customers),
        )
        return self

    def predict_clv(
        self,
        time_horizon_days: int = 365,
        discount_rate: float = 0.1,
        margin: float = 1.0,
    ) -> pd.DataFrame:
        """Predict Customer Lifetime Value for each consumer.

        Parameters
        ----------
        time_horizon_days : int, default 365
            Prediction horizon in days.
        discount_rate : float, default 0.1
            Annual discount rate for NPV calculation.  Set to 0 for
            undiscounted CLV.
        margin : float, default 1.0
            Profit margin multiplier (e.g., 0.3 for 30% margin).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ``consumer_id``,
            ``expected_purchases``, ``p_alive``, ``predicted_clv``.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        self._check_fitted()

        cd = self.customer_data_.copy()
        params = self.bgnbd_params_

        # Time horizon in model units (same as T: days)
        t = float(time_horizon_days)

        # Expected purchases in [0, t]
        cd["expected_purchases"] = cd.apply(
            lambda row: self._conditional_expected_purchases(
                t,
                row["frequency"],
                row["recency"],
                row["T"],
                params,
            ),
            axis=1,
        )

        # P(alive)
        cd["p_alive"] = cd.apply(
            lambda row: self._p_alive(
                row["frequency"],
                row["recency"],
                row["T"],
                params,
            ),
            axis=1,
        )

        # Expected monetary value
        if self.gg_params_ is not None:
            cd["expected_monetary"] = cd.apply(
                lambda row: self._conditional_expected_monetary(
                    row["frequency"],
                    row["monetary_value"],
                    self.gg_params_,
                )
                if row["frequency"] > 0
                else row["monetary_value"],
                axis=1,
            )
        else:
            cd["expected_monetary"] = cd["monetary_value"]

        # Discount factor (continuous compounding approximation)
        if discount_rate > 0:
            daily_discount = (1 + discount_rate) ** (1 / 365) - 1
            # Approximate discount as average over the horizon
            avg_discount = (1 + daily_discount) ** (t / 2)
            discount_factor = 1.0 / avg_discount
        else:
            discount_factor = 1.0

        # CLV
        cd["predicted_clv"] = (
            cd["expected_monetary"]
            * cd["expected_purchases"]
            * margin
            * discount_factor
        )

        result = cd[
            ["consumer_id", "expected_purchases", "p_alive", "predicted_clv"]
        ].copy()
        result = result.round(
            {"expected_purchases": 4, "p_alive": 4, "predicted_clv": 2}
        )

        return result

    def segment_by_clv(
        self,
        n_tiers: int = 5,
        time_horizon_days: int = 365,
        discount_rate: float = 0.1,
        margin: float = 1.0,
        tier_labels: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Segment consumers into CLV tiers.

        Parameters
        ----------
        n_tiers : int, default 5
            Number of CLV tiers (quantile-based).
        time_horizon_days : int, default 365
            Prediction horizon for CLV calculation.
        discount_rate : float, default 0.1
            Annual discount rate.
        margin : float, default 1.0
            Profit margin multiplier.
        tier_labels : list of str or None
            Custom tier labels (highest CLV first).  If None, uses
            "Tier 1" through "Tier N" where Tier 1 is highest CLV.

        Returns
        -------
        pd.DataFrame
            CLV predictions with an additional ``clv_tier`` column.
        """
        self._check_fitted()

        clv_df = self.predict_clv(
            time_horizon_days=time_horizon_days,
            discount_rate=discount_rate,
            margin=margin,
        )

        if tier_labels is None:
            tier_labels = [f"Tier {i + 1}" for i in range(n_tiers)]
        else:
            if len(tier_labels) != n_tiers:
                raise ValueError(
                    f"tier_labels must have {n_tiers} elements, "
                    f"got {len(tier_labels)}"
                )

        # Assign tiers: highest CLV = Tier 1
        try:
            clv_df["clv_tier"] = pd.qcut(
                clv_df["predicted_clv"],
                q=n_tiers,
                labels=list(reversed(tier_labels)),
                duplicates="drop",
            )
        except ValueError:
            # Too few unique values for the requested number of bins
            logger.warning(
                "Could not create %d CLV tiers; falling back to rank-based assignment.",
                n_tiers,
            )
            ranks = clv_df["predicted_clv"].rank(method="first", ascending=False)
            bin_size = len(clv_df) / n_tiers
            clv_df["clv_tier"] = ranks.apply(
                lambda r: tier_labels[min(int((r - 1) / bin_size), n_tiers - 1)]
            )

        return clv_df

    def clv_summary(
        self,
        time_horizon_days: int = 365,
        discount_rate: float = 0.1,
        n_tiers: int = 5,
    ) -> pd.DataFrame:
        """Aggregate CLV statistics by tier.

        Returns
        -------
        pd.DataFrame
            One row per tier with ``clv_tier``, ``count``,
            ``pct_of_total``, ``avg_clv``, ``total_clv``,
            ``avg_expected_purchases``, ``avg_p_alive``.
        """
        self._check_fitted()

        tiers = self.segment_by_clv(
            n_tiers=n_tiers,
            time_horizon_days=time_horizon_days,
            discount_rate=discount_rate,
        )

        total = len(tiers)
        summary = (
            tiers.groupby("clv_tier", observed=False)
            .agg(
                count=("consumer_id", "size"),
                avg_clv=("predicted_clv", "mean"),
                total_clv=("predicted_clv", "sum"),
                avg_expected_purchases=("expected_purchases", "mean"),
                avg_p_alive=("p_alive", "mean"),
            )
            .reset_index()
        )
        summary["pct_of_total"] = (summary["count"] / total * 100).round(2)

        col_order = [
            "clv_tier", "count", "pct_of_total", "avg_clv",
            "total_clv", "avg_expected_purchases", "avg_p_alive",
        ]
        summary = summary[col_order].sort_values(
            "avg_clv", ascending=False
        ).reset_index(drop=True)

        return summary

    # ------------------------------------------------------------------
    # BG/NBD model
    # ------------------------------------------------------------------

    def _fit_bgnbd(self, customer_data: pd.DataFrame) -> dict:
        """Fit the BG/NBD model via maximum likelihood estimation.

        The BG/NBD likelihood for a customer with frequency x, recency
        t_x, and observation period T is:

            L(r, alpha, a, b | x, t_x, T) =
                B(a, b+x) / B(a, b)
                * Gamma(r+x) / Gamma(r)
                * (alpha / (alpha + T))^r
                * (1 / (alpha + T))^x
                + (a / (b + x - 1))
                * B(a, b+x-1) / B(a, b)
                * (delta(x>0))
                * Gamma(r+x) / Gamma(r)
                * (alpha / (alpha + t_x))^r
                * (1 / (alpha + t_x))^x

        Parameters
        ----------
        customer_data : pd.DataFrame
            Must contain ``frequency``, ``recency``, ``T``.

        Returns
        -------
        dict
            Fitted parameters ``{"r", "alpha", "a", "b"}``.
        """
        x = customer_data["frequency"].values.astype(np.float64)
        t_x = customer_data["recency"].values.astype(np.float64)
        T = customer_data["T"].values.astype(np.float64)

        def neg_ll(params: np.ndarray) -> float:
            r, alpha, a, b = params
            ll = self._bgnbd_log_likelihood(r, alpha, a, b, x, t_x, T)
            penalty = self.penalizer_coef * np.sum(params ** 2)
            return -ll + penalty

        # Try multiple starting points
        best_result = None
        best_nll = np.inf

        start_points = [
            [1.0, 1.0, 1.0, 1.0],
            [0.5, 10.0, 0.5, 0.5],
            [1.0, 5.0, 1.0, 3.0],
            [0.1, 1.0, 0.5, 1.0],
        ]

        for x0 in start_points:
            try:
                result = minimize(
                    neg_ll,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=_PARAM_BOUNDS_BGNBD,
                    options={"maxiter": 2000, "disp": False},
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except (RuntimeWarning, FloatingPointError):
                continue

        if best_result is None or not np.isfinite(best_nll):
            raise RuntimeError(
                "BG/NBD model fitting failed to converge. "
                "Consider increasing penalizer_coef or checking data quality."
            )

        r, alpha, a, b = best_result.x
        return {"r": float(r), "alpha": float(alpha), "a": float(a), "b": float(b)}

    @staticmethod
    def _bgnbd_log_likelihood(
        r: float,
        alpha: float,
        a: float,
        b: float,
        x: np.ndarray,
        t_x: np.ndarray,
        T: np.ndarray,
    ) -> float:
        """Compute the BG/NBD log-likelihood across all customers.

        Parameters
        ----------
        r, alpha, a, b : float
            BG/NBD parameters.
        x : np.ndarray
            Frequency (number of repeat purchases).
        t_x : np.ndarray
            Recency (time of last purchase in observation window).
        T : np.ndarray
            Total observation period length (time since first purchase
            to end of calibration).

        Returns
        -------
        float
            Total log-likelihood.
        """
        # Common terms
        ln_gamma_rx = gammaln(r + x) - gammaln(r)
        ln_beta_ratio = betaln(a, b + x) - betaln(a, b)

        # Term A1: customer is still alive at T
        A1 = (
            ln_gamma_rx
            + ln_beta_ratio
            + r * np.log(alpha)
            - (r + x) * np.log(alpha + T)
        )

        # Term A2: customer became inactive after purchase at t_x
        # Only applies when x > 0
        # Per Fader et al. (2005), the A2 term uses B(a+1, b+x-1)/B(a,b)
        # multiplied by a/(b+x-1), which simplifies in log-space to:
        #   log(a) - log(b+x-1) + lnB(a+1, b+x-1) - lnB(a,b)

        A2 = np.full_like(A1, -np.inf)
        mask = x > 0
        if mask.any():
            A2[mask] = (
                np.log(a) - np.log(b + x[mask] - 1)
                + ln_gamma_rx[mask]
                + betaln(a + 1, b + x[mask] - 1) - betaln(a, b)
                + r * np.log(alpha)
                - (r + x[mask]) * np.log(alpha + t_x[mask])
            )

        # Log-sum-exp for numerical stability
        max_ab = np.maximum(A1, A2)
        ll_individual = max_ab + np.log(
            np.exp(A1 - max_ab) + np.exp(A2 - max_ab)
        )

        return float(np.sum(ll_individual))

    def _conditional_expected_purchases(
        self,
        t: float,
        frequency: float,
        recency: float,
        T: float,
        params: dict,
    ) -> float:
        """Expected number of purchases in the next t time units.

        Uses the BG/NBD conditional expectation:

            E[X(t) | x, t_x, T, params] =
                (a + b + x - 1) / (a - 1)
                * [1 - ((alpha + T) / (alpha + T + t))^(r + x)
                   * hyp2f1(...)]
                * P(alive)  (approximately)

        For numerical stability, we use a simplified form.

        Parameters
        ----------
        t : float
            Future time period (days).
        frequency : float
            Number of repeat purchases.
        recency : float
            Time of last purchase.
        T : float
            Observation period length.
        params : dict
            BG/NBD parameters.

        Returns
        -------
        float
            Expected number of purchases in [0, t].
        """
        r = params["r"]
        alpha = params["alpha"]
        a = params["a"]
        b = params["b"]
        x = frequency

        p_alive = self._p_alive(frequency, recency, T, params)

        if p_alive < 1e-10:
            return 0.0

        # Expected purchase rate while alive
        # E[lambda | alive, x, T] approximation
        expected_rate = (r + x) / (alpha + T)

        # Expected purchases = rate * t * P(alive)
        # This is a first-order approximation; the full BG/NBD conditional
        # expectation involves a hypergeometric function.
        expected_purchases = expected_rate * t * p_alive

        return max(0.0, float(expected_purchases))

    @staticmethod
    def _p_alive(
        frequency: float,
        recency: float,
        T: float,
        params: dict,
    ) -> float:
        """Probability that a customer is still alive at time T.

        P(alive | x, t_x, T) = 1 / (1 + delta)

        where delta = (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x))^(r+x)

        for customers with x > 0.  For x = 0, uses a simpler form.

        Parameters
        ----------
        frequency : float
            Number of repeat purchases (x).
        recency : float
            Time of last purchase (t_x).
        T : float
            Observation period.
        params : dict
            BG/NBD parameters.

        Returns
        -------
        float
            Probability in [0, 1].
        """
        r = params["r"]
        alpha = params["alpha"]
        a = params["a"]
        b = params["b"]
        x = frequency

        if x == 0:
            # No repeat purchases: P(alive) based on zero-purchase likelihood
            delta = (a / (b - 1 + 1e-10)) * ((alpha + T) / alpha) ** r
            # Ensure this doesn't blow up
            delta = min(delta, 1e10)
        else:
            log_delta = (
                np.log(a) - np.log(b + x - 1)
                + (r + x) * np.log((alpha + T) / (alpha + recency))
            )
            # Clip to prevent overflow
            log_delta = min(log_delta, 500.0)
            delta = np.exp(log_delta)

        p = 1.0 / (1.0 + delta)
        return float(np.clip(p, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Gamma-Gamma model
    # ------------------------------------------------------------------

    def _fit_gamma_gamma(self, customer_data: pd.DataFrame) -> dict:
        """Fit the Gamma-Gamma model for monetary value.

        The Gamma-Gamma model assumes that the average transaction value
        for each customer follows a Gamma distribution, and that the
        shape parameter for this Gamma is itself Gamma-distributed across
        the population.

        Parameters
        ----------
        customer_data : pd.DataFrame
            Repeat customers only.  Must contain ``frequency`` and
            ``monetary_value``.

        Returns
        -------
        dict
            Fitted parameters ``{"p", "q", "v"}``.
        """
        x = customer_data["frequency"].values.astype(np.float64)
        m = customer_data["monetary_value"].values.astype(np.float64)

        # Filter out customers with zero or negative monetary value
        valid = (x > 0) & (m > 0)
        x = x[valid]
        m = m[valid]

        if len(x) == 0:
            raise ValueError(
                "No valid repeat customers with positive monetary values."
            )

        def neg_ll(params: np.ndarray) -> float:
            p, q, v = params
            ll = self._gamma_gamma_log_likelihood(p, q, v, x, m)
            penalty = self.penalizer_coef * np.sum(params ** 2)
            return -ll + penalty

        start_points = [
            [1.0, 1.0, 1.0],
            [0.5, 1.0, 10.0],
            [1.0, 5.0, 5.0],
        ]

        best_result = None
        best_nll = np.inf

        for x0 in start_points:
            try:
                result = minimize(
                    neg_ll,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=_PARAM_BOUNDS_GG,
                    options={"maxiter": 2000, "disp": False},
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except (RuntimeWarning, FloatingPointError):
                continue

        if best_result is None or not np.isfinite(best_nll):
            raise RuntimeError(
                "Gamma-Gamma model fitting failed to converge."
            )

        p, q, v = best_result.x
        return {"p": float(p), "q": float(q), "v": float(v)}

    @staticmethod
    def _gamma_gamma_log_likelihood(
        p: float,
        q: float,
        v: float,
        x: np.ndarray,
        m: np.ndarray,
    ) -> float:
        """Compute Gamma-Gamma log-likelihood.

        The individual log-likelihood for customer i with frequency x_i
        and average monetary value m_i is:

            log L_i = log Gamma(p*x_i + q)
                      - log Gamma(p*x_i)
                      - log Gamma(q)
                      + q * log(v)
                      + (p*x_i - 1) * log(m_i)
                      + (p*x_i) * log(x_i)
                      - (p*x_i + q) * log(x_i * m_i + v)

        Parameters
        ----------
        p, q, v : float
            Gamma-Gamma model parameters.
        x : np.ndarray
            Frequency (repeat purchases).
        m : np.ndarray
            Average monetary value per transaction.

        Returns
        -------
        float
            Total log-likelihood.
        """
        px = p * x

        ll = (
            gammaln(px + q)
            - gammaln(px)
            - gammaln(q)
            + q * np.log(v)
            + (px - 1) * np.log(m)
            + px * np.log(x)
            - (px + q) * np.log(x * m + v)
        )

        return float(np.sum(ll))

    @staticmethod
    def _conditional_expected_monetary(
        frequency: float,
        monetary_value: float,
        params: dict,
    ) -> float:
        """Expected average monetary value (Gamma-Gamma conditional mean).

        E[M | x, m_bar, p, q, v] = (q - 1) / (p*x + q - 1) * v / (p*x)
                                     + p*x / (p*x + q - 1) * m_bar

        Simplified form: a weighted average of the population mean (v/p)
        and the individual observed mean (m_bar).

        Parameters
        ----------
        frequency : float
            Number of repeat purchases (x).
        monetary_value : float
            Observed average monetary value (m_bar).
        params : dict
            Gamma-Gamma parameters.

        Returns
        -------
        float
            Expected average monetary value.
        """
        p = params["p"]
        q = params["q"]
        v = params["v"]
        x = frequency

        px = p * x
        denominator = px + q - 1
        if denominator <= 0:
            return monetary_value

        individual_weight = px / denominator
        population_weight = (q - 1) / denominator

        expected_m = individual_weight * monetary_value + population_weight * (v / p)
        return max(0.0, float(expected_m))

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _build_customer_summary(
        self,
        transactions_df: pd.DataFrame,
        consumer_id_col: str,
        timestamp_col: str,
        amount_col: str,
    ) -> pd.DataFrame:
        """Build RFM-style customer summary for BG/NBD and Gamma-Gamma.

        Computes per customer:
            - frequency: number of repeat purchases (total purchases - 1)
            - recency: time between first and last purchase (in days)
            - T: time between first purchase and end of observation (days)
            - monetary_value: average spend per transaction

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Raw transaction data.
        consumer_id_col, timestamp_col, amount_col : str
            Column names.

        Returns
        -------
        pd.DataFrame
            Customer-level summary.
        """
        df = transactions_df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        observation_end = df[timestamp_col].max()

        # Aggregate per customer
        grouped = df.groupby(consumer_id_col).agg(
            first_purchase=(timestamp_col, "min"),
            last_purchase=(timestamp_col, "max"),
            n_transactions=(amount_col, "count"),
            total_amount=(amount_col, "sum"),
        ).reset_index()

        grouped = grouped.rename(columns={consumer_id_col: "consumer_id"})

        # Frequency: number of REPEAT purchases (n - 1)
        grouped["frequency"] = (grouped["n_transactions"] - 1).clip(lower=0)

        # Recency: days from first purchase to last purchase
        grouped["recency"] = (
            grouped["last_purchase"] - grouped["first_purchase"]
        ).dt.days.astype(float)

        # T: days from first purchase to end of observation
        grouped["T"] = (
            observation_end - grouped["first_purchase"]
        ).dt.days.astype(float)

        # Monetary value: average spend per transaction (using all transactions)
        grouped["monetary_value"] = grouped["total_amount"] / grouped["n_transactions"]

        # Ensure T >= recency (should always hold, but be defensive)
        grouped["T"] = grouped[["T", "recency"]].max(axis=1)

        # Floor T and recency to avoid zero-division
        grouped["T"] = grouped["T"].clip(lower=1.0)

        customer_data = grouped[
            ["consumer_id", "frequency", "recency", "T", "monetary_value"]
        ].copy()

        return customer_data

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if the estimator has not been fitted."""
        if not self._fitted:
            raise RuntimeError(
                "Estimator has not been fitted. Call fit() first."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the fitted estimator."""
        if not self._fitted:
            return "CLVEstimator (not fitted)"

        cd = self.customer_data_
        lines = [
            "CLVEstimator",
            f"  customers        : {len(cd):,}",
            f"  repeat buyers    : {(cd['frequency'] > 0).sum():,}",
            "",
            "  BG/NBD parameters:",
            f"    r              : {self.bgnbd_params_['r']:.4f}",
            f"    alpha          : {self.bgnbd_params_['alpha']:.4f}",
            f"    a              : {self.bgnbd_params_['a']:.4f}",
            f"    b              : {self.bgnbd_params_['b']:.4f}",
        ]
        if self.gg_params_ is not None:
            lines.extend([
                "",
                "  Gamma-Gamma parameters:",
                f"    p              : {self.gg_params_['p']:.4f}",
                f"    q              : {self.gg_params_['q']:.4f}",
                f"    v              : {self.gg_params_['v']:.4f}",
            ])
        else:
            lines.append("\n  Gamma-Gamma: not fitted (no repeat purchasers)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        n = len(self.customer_data_) if self._fitted else 0
        return f"CLVEstimator(customers={n}, status={status})"
