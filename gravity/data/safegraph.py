"""
SafeGraph foot-traffic / visit-pattern data loader.

Reads SafeGraph Weekly Patterns or Monthly Patterns files (CSV or
Parquet), parses the JSON-encoded ``visitor_home_cbgs`` column, and
produces normalised origin-x-store visit-share matrices suitable for
calibrating or validating a gravity model.
"""

from __future__ import annotations

import json
import logging
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from gravity.data.schema import Store, VisitEvent, stores_to_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_file(filepath: Path) -> pd.DataFrame:
    """Read a CSV or Parquet file, choosing parser by extension."""
    suffix = filepath.suffix.lower()
    if suffix == ".parquet" or suffix == ".pq":
        return pd.read_parquet(filepath)
    elif suffix in (".csv", ".tsv", ".gz"):
        sep = "\t" if ".tsv" in filepath.name else ","
        return pd.read_csv(filepath, sep=sep, low_memory=False)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            "Expected .csv, .tsv, .csv.gz, or .parquet."
        )


def _resolve_id_column(df: pd.DataFrame) -> str:
    """Determine which column holds the place identifier."""
    for candidate in ("placekey", "safegraph_place_id", "sg_place_id", "poi_id"):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "Cannot find a place-ID column. Expected one of: "
        "placekey, safegraph_place_id, sg_place_id, poi_id."
    )


def _parse_json_col(series: pd.Series) -> pd.Series:
    """Parse a column that contains JSON-encoded dicts (as strings).

    SafeGraph encodes ``visitor_home_cbgs``, ``visitor_home_aggregation``,
    ``popularity_by_hour``, etc. as JSON strings.  This helper converts
    them to native Python dicts / lists, replacing un-parseable values
    with empty dicts.
    """
    def _safe_parse(val):
        if isinstance(val, dict):
            return val
        if isinstance(val, list):
            return val
        if pd.isna(val):
            return {}
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {}

    return series.apply(_safe_parse)


def _filter_date_range(
    df: pd.DataFrame,
    date_range: tuple[str, str],
) -> pd.DataFrame:
    """Filter patterns DataFrame to rows within *date_range*.

    Expects columns ``date_range_start`` and ``date_range_end`` (standard
    SafeGraph column names).
    """
    start_str, end_str = date_range
    start = pd.to_datetime(start_str)
    end = pd.to_datetime(end_str)

    if "date_range_start" in df.columns:
        df["date_range_start"] = pd.to_datetime(df["date_range_start"])
        mask = df["date_range_start"] >= start
        if "date_range_end" in df.columns:
            df["date_range_end"] = pd.to_datetime(df["date_range_end"])
            mask = mask & (df["date_range_end"] <= end)
        df = df.loc[mask].copy()
    else:
        warnings.warn(
            "No date_range_start column found; date_range filter ignored."
        )

    return df


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class SafeGraphLoader:
    """Load and transform SafeGraph foot-traffic patterns.

    Examples
    --------
    >>> loader = SafeGraphLoader()
    >>> shares = loader.load_patterns("weekly_patterns.csv.gz")
    >>> print(shares.shape)   # (n_origins, n_stores)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_patterns(
        self,
        filepath: str | Path,
        date_range: Optional[tuple[str, str]] = None,
    ) -> pd.DataFrame:
        """Load SafeGraph patterns and return a normalised visit-share matrix.

        The returned DataFrame has census-block-group GEOIDs as the index
        (origins) and place-keys / POI IDs as columns (stores).  Each cell
        holds the share of that origin's total visits going to that store.

        Parameters
        ----------
        filepath : str or Path
            Path to a Weekly or Monthly Patterns file (CSV, gzipped CSV,
            or Parquet).
        date_range : tuple[str, str] or None
            Optional ``("YYYY-MM-DD", "YYYY-MM-DD")`` filter applied to
            ``date_range_start`` / ``date_range_end``.

        Returns
        -------
        pd.DataFrame
            Origin (rows) x Store (columns) visit-share matrix.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Patterns file not found: {filepath}")

        df = _read_file(filepath)
        logger.info("Read %d pattern rows from %s", len(df), filepath)

        if date_range is not None:
            df = _filter_date_range(df, date_range)
            logger.info("After date filter: %d rows", len(df))

        id_col = _resolve_id_column(df)

        # Parse the visitor_home_cbgs JSON column
        cbg_col = None
        for candidate in ("visitor_home_cbgs", "visitor_home_aggregation"):
            if candidate in df.columns:
                cbg_col = candidate
                break
        if cbg_col is None:
            raise KeyError(
                "Cannot find visitor_home_cbgs or visitor_home_aggregation "
                "column in the patterns file."
            )

        df[cbg_col] = _parse_json_col(df[cbg_col])

        # Explode CBG dicts into long-form (origin, store, visits)
        long_records: list[dict[str, object]] = []
        for _, row in df.iterrows():
            store_id = row[id_col]
            cbg_dict = row[cbg_col]
            if not isinstance(cbg_dict, dict):
                continue
            for cbg, count in cbg_dict.items():
                long_records.append({
                    "origin_id": str(cbg),
                    "store_id": str(store_id),
                    "visits": int(count),
                })

        if not long_records:
            warnings.warn("No origin-store visit records could be extracted.")
            return pd.DataFrame()

        visits_long = pd.DataFrame(long_records)

        # Aggregate in case multiple pattern rows map to the same pair
        visits_agg = (
            visits_long
            .groupby(["origin_id", "store_id"], as_index=False)["visits"]
            .sum()
        )

        # Pivot to wide matrix
        visits_wide = visits_agg.pivot(
            index="origin_id", columns="store_id", values="visits",
        ).fillna(0)

        # Normalise to shares
        shares = self.visits_to_shares(visits_wide)

        logger.info(
            "Built visit-share matrix: %d origins x %d stores",
            shares.shape[0], shares.shape[1],
        )
        return shares

    def load_places(self, filepath: str | Path) -> list[Store]:
        """Load the SafeGraph Core Places / POI file as ``Store`` models.

        Parameters
        ----------
        filepath : str or Path
            Path to the Core Places file (CSV or Parquet).

        Returns
        -------
        list[Store]
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Places file not found: {filepath}")

        df = _read_file(filepath)
        id_col = _resolve_id_column(df)

        stores: list[Store] = []
        for _, row in df.iterrows():
            store_id = str(row[id_col])

            lat = float(row.get("latitude", row.get("lat", 0.0)))
            lon = float(row.get("longitude", row.get("lon", 0.0)))
            if lat == 0.0 and lon == 0.0:
                continue

            name = row.get("location_name", row.get("name"))
            if isinstance(name, float) and np.isnan(name):
                name = None

            category = row.get("top_category", row.get("sub_category", row.get("naics_code")))
            if isinstance(category, float) and np.isnan(category):
                category = None

            brand = row.get("brands", row.get("brand"))
            if isinstance(brand, float) and np.isnan(brand):
                brand = None

            # Pack any extra useful columns into attributes
            attrs: dict = {}
            for col in ("street_address", "city", "region", "postal_code",
                        "phone_number", "open_hours", "naics_code"):
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    attrs[col] = val

            stores.append(
                Store(
                    store_id=store_id,
                    name=name,
                    lat=lat,
                    lon=lon,
                    category=str(category) if category is not None else None,
                    brand=str(brand) if brand is not None else None,
                    attributes=attrs,
                )
            )

        logger.info("Loaded %d stores from places file: %s", len(stores), filepath)
        return stores

    @staticmethod
    def visits_to_shares(visits_df: pd.DataFrame) -> pd.DataFrame:
        """Normalise a raw visit-count matrix to probability shares.

        Each row (origin) is divided by its row-sum so that values
        represent the proportion of that origin's total visits captured
        by each store.  Rows with zero total visits are left as zeros.

        Parameters
        ----------
        visits_df : pd.DataFrame
            Origin (rows) x Store (columns) integer visit counts.

        Returns
        -------
        pd.DataFrame
            Same shape, with row-normalised float values in [0, 1].
        """
        row_sums = visits_df.sum(axis=1)
        # Avoid division by zero
        row_sums = row_sums.replace(0, np.nan)
        shares = visits_df.div(row_sums, axis=0).fillna(0.0)
        return shares

    def to_visit_events(
        self,
        patterns_df: pd.DataFrame,
        *,
        default_timestamp: Optional[str] = None,
    ) -> list[VisitEvent]:
        """Convert a patterns DataFrame into ``VisitEvent`` models.

        Each row in *patterns_df* produces one ``VisitEvent`` per origin
        CBG that visited the store.  Because SafeGraph data is aggregated
        (not individual-level), ``consumer_id`` is set to the CBG GEOID
        and ``dwell_minutes`` is derived from ``median_dwell`` if present.

        Parameters
        ----------
        patterns_df : pd.DataFrame
            Raw patterns DataFrame (as read from CSV/Parquet, before
            pivoting).  Must contain the place-ID column and
            ``visitor_home_cbgs``.
        default_timestamp : str or None
            ISO-format timestamp to use when ``date_range_start`` is
            missing.  Defaults to ``"2023-01-01T00:00:00"``.

        Returns
        -------
        list[VisitEvent]
        """
        id_col = _resolve_id_column(patterns_df)

        cbg_col = None
        for candidate in ("visitor_home_cbgs", "visitor_home_aggregation"):
            if candidate in patterns_df.columns:
                cbg_col = candidate
                break
        if cbg_col is None:
            raise KeyError("Cannot find visitor_home_cbgs column.")

        patterns_df = patterns_df.copy()
        patterns_df[cbg_col] = _parse_json_col(patterns_df[cbg_col])

        default_ts = default_timestamp or "2023-01-01T00:00:00"

        events: list[VisitEvent] = []
        for _, row in patterns_df.iterrows():
            store_id = str(row[id_col])

            ts_raw = row.get("date_range_start", default_ts)
            try:
                ts = datetime.fromisoformat(str(ts_raw))
            except (ValueError, TypeError):
                ts = datetime.fromisoformat(default_ts)

            median_dwell = float(row.get("median_dwell", 0.0))
            if np.isnan(median_dwell):
                median_dwell = 0.0

            cbg_dict = row[cbg_col]
            if not isinstance(cbg_dict, dict):
                continue

            for cbg, count in cbg_dict.items():
                for _ in range(int(count)):
                    events.append(
                        VisitEvent(
                            event_id=uuid.uuid4().hex[:16],
                            consumer_id=str(cbg),
                            store_id=store_id,
                            timestamp=ts,
                            dwell_minutes=median_dwell,
                            source="safegraph",
                        )
                    )

        logger.info("Generated %d VisitEvent records from patterns.", len(events))
        return events

    @staticmethod
    def to_dataframe(stores: list[Store]) -> pd.DataFrame:
        """Convenience wrapper around ``stores_to_dataframe``."""
        return stores_to_dataframe(stores)
