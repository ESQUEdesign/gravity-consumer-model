"""
POS / transaction data loader.

Reads transaction-level data from CSV, Parquet, or JSON files, validates
and cleans the records, and produces ``Transaction`` pydantic models plus
aggregation utilities for the gravity-model pipeline.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from gravity.data.schema import Transaction, transactions_to_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Canonical column names the loader expects internally.
_REQUIRED_COLUMNS = {"transaction_id", "consumer_id", "store_id", "timestamp", "amount"}
_OPTIONAL_COLUMNS = {"items", "category"}

# Default column mapping (identity -- expected names match actual names).
_DEFAULT_MAPPING: dict[str, str] = {col: col for col in _REQUIRED_COLUMNS | _OPTIONAL_COLUMNS}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_file(filepath: Path) -> pd.DataFrame:
    """Read CSV, Parquet, or JSON into a DataFrame."""
    suffix = filepath.suffix.lower()
    if suffix == ".parquet" or suffix == ".pq":
        return pd.read_parquet(filepath)
    elif suffix in (".csv", ".tsv", ".gz"):
        sep = "\t" if ".tsv" in filepath.name else ","
        return pd.read_csv(filepath, sep=sep, low_memory=False)
    elif suffix == ".json" or suffix == ".jsonl":
        # Support both a JSON array and line-delimited JSON
        lines = "jsonl" in filepath.name or suffix == ".jsonl"
        return pd.read_json(filepath, lines=lines)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            "Expected .csv, .tsv, .csv.gz, .parquet, .json, or .jsonl."
        )


def _apply_column_mapping(
    df: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    """Rename columns according to *mapping* (canonical -> actual).

    The mapping is ``{expected_name: actual_name}``.  After renaming, the
    DataFrame uses canonical column names.
    """
    # Invert: actual -> canonical
    rename_map = {actual: canonical for canonical, actual in mapping.items()
                  if actual in df.columns and actual != canonical}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _clean_amount(series: pd.Series) -> pd.Series:
    """Coerce an amount column to non-negative floats.

    Strips dollar signs, commas, and parentheses (accounting negative
    notation), then converts to float.  Negative values are clipped to 0.
    """
    if series.dtype == object:
        series = (
            series
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .str.replace(r"\((.+)\)", r"-\1", regex=True)
        )
    series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return series.clip(lower=0.0)


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse a timestamp column, handling common date/datetime formats."""
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class TransactionLoader:
    """Load, validate, and transform POS / transaction data.

    Parameters
    ----------
    column_mapping : dict or None
        Global column-name mapping applied on every ``load()`` call
        unless overridden.  Keys are canonical column names
        (``transaction_id``, ``consumer_id``, ``store_id``, ``timestamp``,
        ``amount``, ``items``, ``category``); values are the actual column
        names in the source file.

    Examples
    --------
    >>> loader = TransactionLoader(
    ...     column_mapping={"consumer_id": "cust_no", "timestamp": "txn_date"}
    ... )
    >>> txns = loader.load("sales_2024.csv")
    >>> df = loader.to_dataframe(txns)
    """

    def __init__(
        self,
        column_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        self.column_mapping = dict(_DEFAULT_MAPPING)
        if column_mapping:
            self.column_mapping.update(column_mapping)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        filepath: str | Path,
        column_mapping: Optional[dict[str, str]] = None,
    ) -> list[Transaction]:
        """Load transaction data from a file and return ``Transaction`` models.

        Parameters
        ----------
        filepath : str or Path
            Path to a CSV, Parquet, or JSON file.
        column_mapping : dict or None
            Per-call column-name overrides.  Merged on top of the
            instance-level mapping.

        Returns
        -------
        list[Transaction]

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        ValueError
            If required columns are missing after mapping.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Transaction file not found: {filepath}")

        df = _read_file(filepath)
        logger.info("Read %d raw transaction rows from %s", len(df), filepath)

        # Merge mappings
        effective_mapping = dict(self.column_mapping)
        if column_mapping:
            effective_mapping.update(column_mapping)

        df = _apply_column_mapping(df, effective_mapping)

        # --- Validate required columns ---
        # transaction_id is special: auto-generate if missing
        if "transaction_id" not in df.columns:
            df["transaction_id"] = [uuid.uuid4().hex[:16] for _ in range(len(df))]
            logger.info("Auto-generated transaction_id for %d rows.", len(df))

        required_present = {"consumer_id", "store_id", "timestamp", "amount"}
        missing = required_present - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns after mapping: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        # --- Clean & parse ---
        df["timestamp"] = _parse_timestamps(df["timestamp"])
        df["amount"] = _clean_amount(df["amount"])

        # Drop rows where timestamp could not be parsed
        null_ts = df["timestamp"].isna()
        if null_ts.any():
            n_bad = int(null_ts.sum())
            warnings.warn(
                f"Dropped {n_bad} rows with unparseable timestamps."
            )
            df = df.loc[~null_ts].copy()

        # Fill optional columns with defaults
        if "items" not in df.columns:
            df["items"] = 1
        else:
            df["items"] = pd.to_numeric(df["items"], errors="coerce").fillna(1).astype(int)
            df["items"] = df["items"].clip(lower=1)

        if "category" not in df.columns:
            df["category"] = None

        # --- Build models ---
        transactions: list[Transaction] = []
        for _, row in df.iterrows():
            cat = row.get("category")
            if isinstance(cat, float) and np.isnan(cat):
                cat = None

            transactions.append(
                Transaction(
                    transaction_id=str(row["transaction_id"]),
                    consumer_id=str(row["consumer_id"]),
                    store_id=str(row["store_id"]),
                    timestamp=row["timestamp"].to_pydatetime(),
                    amount=float(row["amount"]),
                    items=int(row["items"]),
                    category=str(cat) if cat is not None else None,
                )
            )

        logger.info(
            "Loaded %d valid transactions from %s", len(transactions), filepath,
        )
        return transactions

    @staticmethod
    def to_dataframe(txns: list[Transaction]) -> pd.DataFrame:
        """Convenience wrapper around ``transactions_to_dataframe``."""
        return transactions_to_dataframe(txns)

    @staticmethod
    def compute_visit_matrix(
        txns_df: pd.DataFrame,
        origin_mapping: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Build an origin x store visit-count matrix from transactions.

        Parameters
        ----------
        txns_df : pd.DataFrame
            Transaction DataFrame with at least ``consumer_id`` and
            ``store_id`` columns.
        origin_mapping : dict or None
            Optional mapping of ``consumer_id`` -> ``origin_id`` (e.g.
            linking a customer to a census block group).  If *None*,
            ``consumer_id`` itself is used as the origin dimension.

        Returns
        -------
        pd.DataFrame
            Origin (rows) x Store (columns) visit counts.
        """
        df = txns_df.copy()

        if origin_mapping is not None:
            df["origin_id"] = df["consumer_id"].map(origin_mapping)
            # Drop consumers that have no origin mapping
            unmapped = df["origin_id"].isna()
            if unmapped.any():
                n = int(unmapped.sum())
                warnings.warn(
                    f"{n} transactions have no origin mapping and will be "
                    "excluded from the visit matrix."
                )
                df = df.loc[~unmapped].copy()
        else:
            df["origin_id"] = df["consumer_id"]

        visit_counts = (
            df
            .groupby(["origin_id", "store_id"])
            .size()
            .reset_index(name="visits")
        )

        matrix = visit_counts.pivot(
            index="origin_id", columns="store_id", values="visits",
        ).fillna(0).astype(int)

        logger.info(
            "Built visit matrix: %d origins x %d stores",
            matrix.shape[0], matrix.shape[1],
        )
        return matrix

    @staticmethod
    def aggregate_by_period(
        txns_df: pd.DataFrame,
        period: str = "W",
    ) -> pd.DataFrame:
        """Aggregate transactions by time period.

        Groups by ``store_id`` and the specified period, computing
        total visits, total revenue, unique consumers, and average
        basket size.

        Parameters
        ----------
        txns_df : pd.DataFrame
            Transaction DataFrame (must have ``store_id``, ``timestamp``,
            ``amount``, ``consumer_id`` columns).
        period : str
            Pandas offset alias for the grouping period.
            ``"W"`` = weekly, ``"ME"`` = monthly, ``"D"`` = daily, etc.

        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame with a ``(store_id, period)`` MultiIndex.
        """
        df = txns_df.copy()

        if "timestamp" not in df.columns:
            raise ValueError("txns_df must contain a 'timestamp' column.")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["period"] = df["timestamp"].dt.to_period(period)

        agg = (
            df
            .groupby(["store_id", "period"])
            .agg(
                total_visits=("consumer_id", "count"),
                total_revenue=("amount", "sum"),
                unique_consumers=("consumer_id", "nunique"),
                avg_basket=("amount", "mean"),
            )
        )

        agg["avg_basket"] = agg["avg_basket"].round(2)
        agg["total_revenue"] = agg["total_revenue"].round(2)

        logger.info(
            "Aggregated transactions into %d store-period rows (period=%s).",
            len(agg), period,
        )
        return agg
