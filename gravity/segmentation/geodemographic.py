"""
Geodemographic Mapper
=====================
Maps census geographies to external segmentation systems (e.g., PRIZM,
Tapestry, Mosaic, or custom cluster solutions).

Geodemographic segments enrich gravity model origins with lifestyle and
behavioral profiles, enabling segment-specific trade area analysis and
marketing strategy.

Typical workflow:
    1. Load a crosswalk that maps geo_id to segment codes.
    2. Map origin DataFrames to their segments.
    3. Query segment profiles and generate aggregate summaries.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GeodemographicMapper:
    """Maps census geographies to geodemographic segmentation systems.

    Supports standard commercial systems (PRIZM, Tapestry, Mosaic) as
    well as custom segmentation schemes defined as DataFrames.

    Parameters
    ----------
    system_name : str, default "custom"
        Name of the segmentation system (for labelling and logging).

    Attributes
    ----------
    crosswalk_ : pd.DataFrame or None
        The loaded crosswalk table mapping geo_id to segment codes.
    profiles_ : pd.DataFrame or None
        Segment-level profile data (demographics, behavior, etc.).
    system_name : str
        Name of the segmentation system.

    Examples
    --------
    >>> mapper = GeodemographicMapper(system_name="PRIZM")
    >>> mapper.load_crosswalk(crosswalk_df)
    >>> enriched = mapper.map_origins(origins_df)
    >>> profile = mapper.get_segment_profile("01A")
    >>> summary = mapper.segment_summary(origins_df)
    """

    def __init__(self, system_name: str = "custom") -> None:
        self.system_name = system_name
        self.crosswalk_: Optional[pd.DataFrame] = None
        self.profiles_: Optional[pd.DataFrame] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_crosswalk(
        self,
        crosswalk_df: pd.DataFrame,
        geo_id_col: str = "geo_id",
        segment_code_col: str = "segment_code",
        segment_name_col: Optional[str] = "segment_name",
    ) -> "GeodemographicMapper":
        """Load a geography-to-segment crosswalk table.

        Parameters
        ----------
        crosswalk_df : pd.DataFrame
            Crosswalk mapping with at least ``geo_id`` and
            ``segment_code`` columns.  May also include
            ``segment_name`` and any number of propensity score or
            demographic columns.
        geo_id_col : str, default "geo_id"
            Column name containing the geography identifier (e.g.,
            census tract FIPS, block group GEOID, ZIP code).
        segment_code_col : str, default "segment_code"
            Column name containing the segment code.
        segment_name_col : str or None, default "segment_name"
            Column name containing the human-readable segment name.
            Set to None if not available.

        Returns
        -------
        GeodemographicMapper
            ``self``, with the crosswalk loaded.

        Raises
        ------
        ValueError
            If required columns are missing from ``crosswalk_df``.
        """
        required = {geo_id_col, segment_code_col}
        missing = required - set(crosswalk_df.columns)
        if missing:
            raise ValueError(
                f"crosswalk_df is missing required columns: {missing}"
            )

        # Standardise column names internally
        rename_map = {
            geo_id_col: "geo_id",
            segment_code_col: "segment_code",
        }
        if segment_name_col and segment_name_col in crosswalk_df.columns:
            rename_map[segment_name_col] = "segment_name"

        cw = crosswalk_df.rename(columns=rename_map).copy()
        cw["geo_id"] = cw["geo_id"].astype(str)
        cw["segment_code"] = cw["segment_code"].astype(str)

        self.crosswalk_ = cw
        self._loaded = True

        # Build segment profiles from crosswalk propensity/demographic columns
        self._build_profiles(cw)

        n_geos = cw["geo_id"].nunique()
        n_segments = cw["segment_code"].nunique()
        logger.info(
            "Loaded %s crosswalk: %d geographies, %d segments",
            self.system_name,
            n_geos,
            n_segments,
        )
        return self

    def load_profiles(self, profiles_df: pd.DataFrame) -> "GeodemographicMapper":
        """Load standalone segment profile data.

        Use this when segment profiles come from a separate source
        (e.g., vendor documentation) rather than being embedded in
        the crosswalk.

        Parameters
        ----------
        profiles_df : pd.DataFrame
            Profile data indexed by or containing ``segment_code``.
            All columns beyond ``segment_code`` and ``segment_name``
            are treated as profile attributes.

        Returns
        -------
        GeodemographicMapper
            ``self``, with profiles loaded.
        """
        if "segment_code" in profiles_df.columns:
            profiles_df = profiles_df.set_index("segment_code")
        self.profiles_ = profiles_df.copy()
        logger.info(
            "Loaded %d segment profiles with %d attributes",
            len(profiles_df),
            len(profiles_df.columns),
        )
        return self

    def map_origins(
        self,
        origins_df: pd.DataFrame,
        geo_id_col: str = "origin_id",
        how: str = "left",
    ) -> pd.DataFrame:
        """Join origins to their geodemographic segment.

        Parameters
        ----------
        origins_df : pd.DataFrame
            Origins DataFrame.  Must contain a column matching
            ``geo_id_col`` that corresponds to the crosswalk's
            ``geo_id``.
        geo_id_col : str, default "origin_id"
            Column in ``origins_df`` to join on.  This is matched
            against the crosswalk's ``geo_id``.
        how : str, default "left"
            Join type.  ``"left"`` keeps all origins even if they
            have no matching segment.

        Returns
        -------
        pd.DataFrame
            A copy of ``origins_df`` with segment columns appended
            (``segment_code``, ``segment_name``, and any propensity
            scores from the crosswalk).

        Raises
        ------
        RuntimeError
            If no crosswalk has been loaded.
        """
        self._check_loaded()

        # If origins_df has the geo_id in the index, bring it into a column
        origins = origins_df.copy()
        if geo_id_col not in origins.columns and origins.index.name == geo_id_col:
            origins = origins.reset_index()

        origins[geo_id_col] = origins[geo_id_col].astype(str)

        merged = origins.merge(
            self.crosswalk_,
            left_on=geo_id_col,
            right_on="geo_id",
            how=how,
            suffixes=("", "_segment"),
        )

        # Drop the redundant geo_id column from crosswalk if it duplicates
        if "geo_id" in merged.columns and geo_id_col != "geo_id":
            merged = merged.drop(columns=["geo_id"])

        n_matched = merged["segment_code"].notna().sum()
        n_total = len(merged)
        match_rate = n_matched / n_total if n_total > 0 else 0.0
        logger.info(
            "Mapped %d / %d origins to segments (%.1f%% match rate)",
            n_matched,
            n_total,
            match_rate * 100,
        )

        return merged

    def get_segment_profile(
        self, segment_code: str
    ) -> dict:
        """Return the demographics/behavioral profile for a segment.

        Parameters
        ----------
        segment_code : str
            The segment code to look up.

        Returns
        -------
        dict
            Profile attributes for the segment.  Keys depend on the
            columns in the crosswalk or profiles data.

        Raises
        ------
        RuntimeError
            If no crosswalk or profiles have been loaded.
        KeyError
            If ``segment_code`` is not found.
        """
        self._check_loaded()

        segment_code = str(segment_code)

        if self.profiles_ is not None and segment_code in self.profiles_.index:
            profile = self.profiles_.loc[segment_code]
            result = profile.to_dict()
            result["segment_code"] = segment_code
            return result

        # Fall back to crosswalk aggregate
        cw = self.crosswalk_
        segment_rows = cw[cw["segment_code"] == segment_code]
        if segment_rows.empty:
            raise KeyError(
                f"Segment code '{segment_code}' not found in crosswalk or profiles."
            )

        result = {"segment_code": segment_code}
        if "segment_name" in segment_rows.columns:
            result["segment_name"] = segment_rows["segment_name"].iloc[0]

        # Aggregate numeric propensity/demographic columns
        numeric_cols = segment_rows.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            result[col] = float(segment_rows[col].mean())

        result["geo_count"] = len(segment_rows)
        return result

    def segment_summary(
        self,
        origins_df: pd.DataFrame,
        geo_id_col: str = "origin_id",
        agg_cols: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Aggregate statistics by geodemographic segment.

        Parameters
        ----------
        origins_df : pd.DataFrame
            Origins DataFrame with demographic columns (e.g.,
            ``population``, ``households``, ``median_income``).
        geo_id_col : str, default "origin_id"
            Column to join on (matched to crosswalk ``geo_id``).
        agg_cols : dict or None
            Custom aggregation specification mapping column names to
            aggregation functions.  Defaults to sensible aggregations
            for common demographic columns.

        Returns
        -------
        pd.DataFrame
            One row per segment with aggregate statistics.  Always
            includes ``segment_code``, ``segment_name`` (if available),
            and ``count`` (number of origins in the segment).
        """
        self._check_loaded()

        mapped = self.map_origins(origins_df, geo_id_col=geo_id_col)

        if agg_cols is None:
            agg_cols = self._default_agg_spec(mapped)

        if not agg_cols:
            # No numeric columns found; just count
            summary = (
                mapped.groupby("segment_code")
                .size()
                .reset_index(name="count")
            )
        else:
            summary = (
                mapped.groupby("segment_code")
                .agg(**{
                    col: pd.NamedAgg(column=col, aggfunc=func)
                    for col, func in agg_cols.items()
                    if col in mapped.columns
                })
                .reset_index()
            )
            summary["count"] = (
                mapped.groupby("segment_code").size().values
            )

        # Attach segment name if available
        if "segment_name" in self.crosswalk_.columns:
            name_map = (
                self.crosswalk_[["segment_code", "segment_name"]]
                .drop_duplicates("segment_code")
                .set_index("segment_code")["segment_name"]
            )
            summary["segment_name"] = (
                summary["segment_code"].map(name_map)
            )
            # Move segment_name next to segment_code
            cols = summary.columns.tolist()
            cols.remove("segment_name")
            idx = cols.index("segment_code") + 1
            cols.insert(idx, "segment_name")
            summary = summary[cols]

        summary = summary.sort_values("count", ascending=False).reset_index(
            drop=True
        )
        return summary

    def list_segments(self) -> pd.DataFrame:
        """Return a summary table of all segments in the crosswalk.

        Returns
        -------
        pd.DataFrame
            One row per segment with ``segment_code``, ``segment_name``
            (if available), and ``geo_count``.
        """
        self._check_loaded()

        cw = self.crosswalk_
        result = (
            cw.groupby("segment_code")
            .size()
            .reset_index(name="geo_count")
            .sort_values("geo_count", ascending=False)
            .reset_index(drop=True)
        )

        if "segment_name" in cw.columns:
            name_map = (
                cw[["segment_code", "segment_name"]]
                .drop_duplicates("segment_code")
                .set_index("segment_code")["segment_name"]
            )
            result["segment_name"] = result["segment_code"].map(name_map)
            result = result[["segment_code", "segment_name", "geo_count"]]

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_profiles(self, cw: pd.DataFrame) -> None:
        """Build segment profiles from crosswalk numeric columns."""
        numeric_cols = cw.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return

        profile_cols = ["segment_code"] + numeric_cols
        if "segment_name" in cw.columns:
            profile_cols.insert(1, "segment_name")

        agg_dict = {col: "mean" for col in numeric_cols}
        if "segment_name" in cw.columns:
            agg_dict["segment_name"] = "first"

        profiles = cw[profile_cols].groupby("segment_code").agg(agg_dict)
        self.profiles_ = profiles

    def _default_agg_spec(self, df: pd.DataFrame) -> dict[str, str]:
        """Build a default aggregation spec from common demographic columns."""
        spec = {}
        sum_cols = ["population", "households"]
        median_cols = ["median_income", "median_age", "avg_household_size"]
        mean_cols = []

        # Detect any propensity score columns (often prefixed with "prop_" or "index_")
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in sum_cols:
                spec[col] = "sum"
            elif col in median_cols:
                spec[col] = "median"
            elif col.startswith(("prop_", "index_", "propensity_")):
                spec[col] = "mean"
            elif col not in ["lat", "lon", "segment_code"]:
                mean_cols.append(col)

        for col in mean_cols:
            spec[col] = "mean"

        return spec

    def _check_loaded(self) -> None:
        """Raise if no crosswalk has been loaded."""
        if not self._loaded:
            raise RuntimeError(
                "No crosswalk loaded. Call load_crosswalk() first."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the mapper state."""
        if not self._loaded:
            return f"GeodemographicMapper(system={self.system_name}, not loaded)"

        n_geos = self.crosswalk_["geo_id"].nunique()
        n_segments = self.crosswalk_["segment_code"].nunique()
        lines = [
            f"GeodemographicMapper",
            f"  system           : {self.system_name}",
            f"  geographies      : {n_geos:,}",
            f"  segments         : {n_segments}",
        ]
        if self.profiles_ is not None:
            lines.append(
                f"  profile attrs    : {len(self.profiles_.columns)}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"GeodemographicMapper(system={self.system_name!r}, status={status})"
