"""
RFM Scorer
==========
Recency-Frequency-Monetary scoring from transaction data.

RFM analysis segments consumers based on three behavioral dimensions:

    - **Recency (R)**: How recently did the consumer purchase?
    - **Frequency (F)**: How often do they purchase?
    - **Monetary (M)**: How much do they spend?

Each dimension is scored into quantile-based bins (default: quintiles,
1-5 where 5 is best).  The combination of R, F, M scores produces
composite segment labels such as "Champions", "Loyal", "At Risk",
and "Lost".

References
----------
Bult, J. R. & Wansbeek, T. (1995). "Optimal Selection for Direct Mail."
    *Marketing Science*, 14(4), 378-394.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default RFM segment definitions
# ---------------------------------------------------------------------------

_DEFAULT_SEGMENT_MAP: dict[str, dict] = {
    "Champions": {
        "r_range": (4, 5),
        "f_range": (4, 5),
        "m_range": (4, 5),
        "description": "Bought recently, buy often, and spend the most.",
    },
    "Loyal": {
        "r_range": (3, 5),
        "f_range": (3, 5),
        "m_range": (3, 5),
        "description": "Spend good money regularly. Responsive to promotions.",
    },
    "Potential Loyalists": {
        "r_range": (3, 5),
        "f_range": (1, 3),
        "m_range": (1, 3),
        "description": "Recent customers with average frequency.",
    },
    "New Customers": {
        "r_range": (4, 5),
        "f_range": (1, 1),
        "m_range": (1, 5),
        "description": "Bought most recently, but not often.",
    },
    "Promising": {
        "r_range": (3, 4),
        "f_range": (1, 1),
        "m_range": (1, 5),
        "description": "Recent shoppers who have not bought much yet.",
    },
    "Needs Attention": {
        "r_range": (2, 3),
        "f_range": (2, 3),
        "m_range": (2, 3),
        "description": "Above average recency, frequency, and monetary values. May not have bought recently.",
    },
    "About to Sleep": {
        "r_range": (2, 3),
        "f_range": (1, 2),
        "m_range": (1, 2),
        "description": "Below average recency and frequency. Will lose them if not reactivated.",
    },
    "At Risk": {
        "r_range": (1, 2),
        "f_range": (3, 5),
        "m_range": (3, 5),
        "description": "Spent big money and purchased often but long time ago. Need to bring them back.",
    },
    "Cannot Lose": {
        "r_range": (1, 2),
        "f_range": (4, 5),
        "m_range": (4, 5),
        "description": "Made biggest purchases and most often, but have not returned for a long time.",
    },
    "Hibernating": {
        "r_range": (1, 2),
        "f_range": (1, 2),
        "m_range": (1, 2),
        "description": "Last purchase was long ago, low spenders, low number of orders.",
    },
    "Lost": {
        "r_range": (1, 1),
        "f_range": (1, 1),
        "m_range": (1, 5),
        "description": "Lowest recency, frequency scores.",
    },
}


class RFMScorer:
    """Recency-Frequency-Monetary scoring for consumer segmentation.

    Computes R, F, M values from transaction data, assigns quantile-based
    scores, and maps score combinations to named segment labels.

    Parameters
    ----------
    n_bins : int, default 5
        Number of quantile bins for each RFM dimension (e.g., 5 for
        quintiles, 4 for quartiles).
    segment_map : dict or None, default None
        Custom segment definitions mapping segment names to score-range
        rules.  If None, uses the built-in 11-segment scheme.  Each
        entry should be::

            {"r_range": (lo, hi), "f_range": (lo, hi), "m_range": (lo, hi)}

    Attributes
    ----------
    rfm_table_ : pd.DataFrame or None
        The computed RFM table from the last ``fit()`` or ``score()``
        call.
    reference_date_ : pd.Timestamp or None
        The reference date used for recency calculations.

    Examples
    --------
    >>> scorer = RFMScorer(n_bins=5)
    >>> scorer.fit(transactions_df)
    >>> result = scorer.score(transactions_df)
    >>> summary = scorer.segment_summary()
    """

    def __init__(
        self,
        n_bins: int = 5,
        segment_map: Optional[dict] = None,
    ) -> None:
        self.n_bins = n_bins
        self.segment_map = segment_map or _DEFAULT_SEGMENT_MAP
        self.rfm_table_: Optional[pd.DataFrame] = None
        self.reference_date_: Optional[pd.Timestamp] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        transactions_df: pd.DataFrame,
        reference_date: Optional[str] = None,
        consumer_id_col: str = "consumer_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
    ) -> "RFMScorer":
        """Compute RFM values from transaction data.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Transaction-level data.  Must contain at least
            ``consumer_id``, ``timestamp``, and ``amount`` columns.
        reference_date : str or None, default None
            Date string to compute recency against (e.g., "2024-01-01").
            If None, uses one day after the maximum transaction date.
        consumer_id_col : str, default "consumer_id"
            Column name for consumer identifiers.
        timestamp_col : str, default "timestamp"
            Column name for transaction timestamps.
        amount_col : str, default "amount"
            Column name for transaction amounts.

        Returns
        -------
        RFMScorer
            Fitted scorer (``self``).

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        self._validate_columns(
            transactions_df, consumer_id_col, timestamp_col, amount_col
        )

        df = transactions_df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Determine reference date
        if reference_date is not None:
            self.reference_date_ = pd.Timestamp(reference_date)
        else:
            self.reference_date_ = df[timestamp_col].max() + pd.Timedelta(days=1)

        # Compute raw RFM values per consumer
        rfm = self._compute_rfm(
            df, consumer_id_col, timestamp_col, amount_col
        )

        # Assign quantile scores
        rfm = self._assign_scores(rfm)

        # Assign segment labels
        rfm["rfm_segment"] = rfm.apply(self._classify_segment, axis=1)

        self.rfm_table_ = rfm
        self._fitted = True

        logger.info(
            "RFM scoring complete: %d consumers, %d segments identified",
            len(rfm),
            rfm["rfm_segment"].nunique(),
        )
        return self

    def score(
        self,
        transactions_df: pd.DataFrame,
        reference_date: Optional[str] = None,
        consumer_id_col: str = "consumer_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        """Compute and return the full RFM scoring table.

        Convenience method that calls ``fit()`` and returns the result
        table.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Transaction-level data.
        reference_date : str or None
            Reference date for recency calculation.
        consumer_id_col : str, default "consumer_id"
            Column name for consumer identifiers.
        timestamp_col : str, default "timestamp"
            Column name for transaction timestamps.
        amount_col : str, default "amount"
            Column name for transaction amounts.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ``consumer_id``, ``recency_days``,
            ``frequency``, ``monetary``, ``r_score``, ``f_score``,
            ``m_score``, ``rfm_segment``.
        """
        self.fit(
            transactions_df,
            reference_date=reference_date,
            consumer_id_col=consumer_id_col,
            timestamp_col=timestamp_col,
            amount_col=amount_col,
        )
        return self.rfm_table_.copy()

    def segment_summary(self) -> pd.DataFrame:
        """Aggregate statistics per RFM segment.

        Returns
        -------
        pd.DataFrame
            One row per segment with columns: ``rfm_segment``,
            ``count``, ``pct_of_total``, ``avg_recency_days``,
            ``avg_frequency``, ``avg_monetary``, ``total_monetary``,
            ``avg_r_score``, ``avg_f_score``, ``avg_m_score``.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        self._check_fitted()

        rfm = self.rfm_table_
        total = len(rfm)

        summary = (
            rfm.groupby("rfm_segment")
            .agg(
                count=("consumer_id", "size"),
                avg_recency_days=("recency_days", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
                total_monetary=("monetary", "sum"),
                avg_r_score=("r_score", "mean"),
                avg_f_score=("f_score", "mean"),
                avg_m_score=("m_score", "mean"),
            )
            .reset_index()
        )

        summary["pct_of_total"] = (summary["count"] / total * 100).round(2)

        # Round numeric columns for readability
        for col in ["avg_recency_days", "avg_frequency", "avg_monetary",
                     "total_monetary", "avg_r_score", "avg_f_score", "avg_m_score"]:
            summary[col] = summary[col].round(2)

        # Reorder columns
        col_order = [
            "rfm_segment", "count", "pct_of_total",
            "avg_recency_days", "avg_frequency", "avg_monetary",
            "total_monetary", "avg_r_score", "avg_f_score", "avg_m_score",
        ]
        summary = summary[col_order].sort_values(
            "count", ascending=False
        ).reset_index(drop=True)

        return summary

    def get_segment_consumers(self, segment_name: str) -> pd.DataFrame:
        """Return all consumers in a given RFM segment.

        Parameters
        ----------
        segment_name : str
            Name of the RFM segment (e.g., "Champions", "At Risk").

        Returns
        -------
        pd.DataFrame
            Subset of the RFM table for the requested segment.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        KeyError
            If ``segment_name`` is not found in the scored data.
        """
        self._check_fitted()

        mask = self.rfm_table_["rfm_segment"] == segment_name
        if not mask.any():
            available = sorted(self.rfm_table_["rfm_segment"].unique())
            raise KeyError(
                f"Segment '{segment_name}' not found. "
                f"Available segments: {available}"
            )
        return self.rfm_table_[mask].copy()

    # ------------------------------------------------------------------
    # Internal: RFM computation
    # ------------------------------------------------------------------

    def _compute_rfm(
        self,
        df: pd.DataFrame,
        consumer_id_col: str,
        timestamp_col: str,
        amount_col: str,
    ) -> pd.DataFrame:
        """Compute raw R, F, M values per consumer.

        Returns
        -------
        pd.DataFrame
            Columns: ``consumer_id``, ``recency_days``, ``frequency``,
            ``monetary``.
        """
        grouped = df.groupby(consumer_id_col).agg(
            last_purchase=(timestamp_col, "max"),
            frequency=(amount_col, "count"),
            monetary=(amount_col, "sum"),
        ).reset_index()

        grouped = grouped.rename(columns={consumer_id_col: "consumer_id"})

        # Recency: days since last purchase relative to reference date
        grouped["recency_days"] = (
            self.reference_date_ - grouped["last_purchase"]
        ).dt.days

        # Monetary: average spend per transaction
        grouped["monetary"] = grouped["monetary"] / grouped["frequency"]

        rfm = grouped[["consumer_id", "recency_days", "frequency", "monetary"]].copy()
        return rfm

    def _assign_scores(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Assign quantile-based R, F, M scores.

        Recency is scored inversely (lower recency = higher score, since
        recent purchases are better).  Frequency and monetary are scored
        directly (higher = better).

        Parameters
        ----------
        rfm : pd.DataFrame
            Raw RFM values with ``recency_days``, ``frequency``,
            ``monetary`` columns.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``r_score``, ``f_score``, ``m_score``
            columns appended.
        """
        # Recency: lower is better, so reverse the label order
        rfm["r_score"] = self._quantile_score(
            rfm["recency_days"], ascending=False
        )
        # Frequency: higher is better
        rfm["f_score"] = self._quantile_score(
            rfm["frequency"], ascending=True
        )
        # Monetary: higher is better
        rfm["m_score"] = self._quantile_score(
            rfm["monetary"], ascending=True
        )

        return rfm

    def _quantile_score(
        self, series: pd.Series, ascending: bool = True
    ) -> pd.Series:
        """Assign quantile-based integer scores to a Series.

        Parameters
        ----------
        series : pd.Series
            Numeric values to bin.
        ascending : bool, default True
            If True, higher values get higher scores.  If False,
            lower values get higher scores (used for recency).

        Returns
        -------
        pd.Series
            Integer scores from 1 to ``self.n_bins``.
        """
        try:
            if ascending:
                scores = pd.qcut(
                    series, q=self.n_bins, labels=False, duplicates="drop"
                ) + 1
            else:
                scores = pd.qcut(
                    series, q=self.n_bins, labels=False, duplicates="drop"
                )
                # Reverse: highest quantile gets score 1 (worst recency)
                scores = self.n_bins - scores
        except ValueError:
            # All values identical or too few unique values for n_bins
            scores = pd.Series(
                np.full(len(series), self.n_bins // 2 + 1),
                index=series.index,
            )
            logger.warning(
                "Could not create %d quantile bins for '%s'; "
                "assigned median score to all consumers.",
                self.n_bins,
                series.name,
            )

        # Fill any NaN from qcut duplicates="drop" and convert to int safely
        scores = scores.fillna(self.n_bins // 2 + 1)
        return scores.astype(int)

    def _classify_segment(self, row: pd.Series) -> str:
        """Map a consumer's R, F, M scores to a named segment.

        Segments are evaluated in definition order; the first matching
        rule wins.

        Parameters
        ----------
        row : pd.Series
            Must contain ``r_score``, ``f_score``, ``m_score``.

        Returns
        -------
        str
            Segment label.
        """
        r, f, m = int(row["r_score"]), int(row["f_score"]), int(row["m_score"])

        for name, rules in self.segment_map.items():
            r_lo, r_hi = rules["r_range"]
            f_lo, f_hi = rules["f_range"]
            m_lo, m_hi = rules["m_range"]

            if (r_lo <= r <= r_hi) and (f_lo <= f <= f_hi) and (m_lo <= m <= m_hi):
                return name

        return "Other"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_columns(
        self,
        df: pd.DataFrame,
        consumer_id_col: str,
        timestamp_col: str,
        amount_col: str,
    ) -> None:
        """Validate that required columns exist in the DataFrame."""
        required = {consumer_id_col, timestamp_col, amount_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"transactions_df is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def _check_fitted(self) -> None:
        """Raise if the scorer has not been fitted."""
        if not self._fitted:
            raise RuntimeError(
                "Scorer has not been fitted. Call fit() or score() first."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the scorer state."""
        if not self._fitted:
            return f"RFMScorer(n_bins={self.n_bins}, not fitted)"

        rfm = self.rfm_table_
        n_consumers = len(rfm)
        n_segments = rfm["rfm_segment"].nunique()

        lines = [
            "RFMScorer",
            f"  n_bins           : {self.n_bins}",
            f"  reference_date   : {self.reference_date_}",
            f"  consumers        : {n_consumers:,}",
            f"  segments found   : {n_segments}",
            "",
            "  Score distributions:",
            f"    R: mean={rfm['r_score'].mean():.2f}, std={rfm['r_score'].std():.2f}",
            f"    F: mean={rfm['f_score'].mean():.2f}, std={rfm['f_score'].std():.2f}",
            f"    M: mean={rfm['m_score'].mean():.2f}, std={rfm['m_score'].std():.2f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        n = len(self.rfm_table_) if self._fitted else 0
        return f"RFMScorer(n_bins={self.n_bins}, consumers={n}, status={status})"
