"""
Trade Area Delineation
======================
Delineates trade areas from origin x store probability matrices produced by
any gravity model (Huff, competing-destinations, GWR, etc.).

A **trade area** is the geographic extent from which a store draws its
customers.  This module provides:

* Probability-contour definitions (e.g., the set of origins capturing 50 %,
  70 %, or 90 % of a store's expected demand).
* Market penetration (expected customers / total population).
* Fair-share index (actual probability share vs. 1/N equal share).
* GeoDataFrame export for choropleth mapping.
* Contour GeoJSON generation via convex hull or alpha shape.

Usage
-----
>>> from gravity.spatial.trade_area import TradeAreaAnalyzer
>>> analyzer = TradeAreaAnalyzer()
>>> ta = analyzer.from_probabilities(prob_df, origins_df, "store_001")
>>> gdf = analyzer.to_geodataframe(prob_df, origins_df, "store_001")
>>> geojson = analyzer.generate_contour_geojson(
...     prob_df, origins_df, "store_001", levels=[0.5, 0.7, 0.9]
... )
"""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, mapping

from gravity.data.schema import TradeArea

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    prob_df: pd.DataFrame,
    origins_df: pd.DataFrame,
    store_id: str,
) -> pd.Series:
    """Validate inputs and return the probability column for *store_id*.

    Parameters
    ----------
    prob_df : pd.DataFrame
        Origin x store probability matrix (index = origin_id, columns = store_id).
    origins_df : pd.DataFrame
        Origins table indexed by origin_id with at least ``lat``, ``lon``,
        ``population``, and ``households`` columns.
    store_id : str
        Target store identifier (must be a column in *prob_df*).

    Returns
    -------
    pd.Series
        Per-origin probabilities for *store_id*, aligned to *origins_df*.

    Raises
    ------
    KeyError
        If *store_id* is not in *prob_df* columns or required columns are
        missing from *origins_df*.
    """
    if store_id not in prob_df.columns:
        raise KeyError(
            f"store_id '{store_id}' not found in prob_df columns. "
            f"Available stores: {list(prob_df.columns[:20])}"
        )

    required_cols = {"lat", "lon", "population", "households"}
    missing = required_cols - set(origins_df.columns)
    if missing:
        raise KeyError(
            f"origins_df is missing required columns: {missing}"
        )

    probs = prob_df[store_id].reindex(origins_df.index)
    if probs.isna().all():
        raise ValueError(
            f"No overlapping origin_ids between prob_df and origins_df for "
            f"store '{store_id}'."
        )
    # Fill any unmatched origins with zero probability.
    probs = probs.fillna(0.0)
    return probs


def _alpha_shape(points: np.ndarray, alpha: float = 0.0) -> "shapely.geometry.base.BaseGeometry":
    """Compute an alpha shape (concave hull) for a set of 2-D points.

    When *alpha* is 0 the result is the convex hull.  Larger values of
    alpha produce tighter-fitting concave boundaries.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) with columns [lon, lat].
    alpha : float
        Alpha parameter controlling concavity.  0 = convex hull.

    Returns
    -------
    shapely Polygon or MultiPolygon
    """
    from scipy.spatial import Delaunay

    if len(points) < 3:
        return MultiPoint([Point(p) for p in points]).convex_hull

    if alpha <= 0:
        return MultiPoint([Point(p) for p in points]).convex_hull

    tri = Delaunay(points)
    edges = set()
    edge_points = []

    for simplex in tri.simplices:
        pa, pb, pc = points[simplex]
        # Lengths of triangle sides.
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        # Semi-perimeter and area.
        s = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 0:
            continue
        area = np.sqrt(area_sq)
        # Circumscribed circle radius.
        circum_r = (a * b * c) / (4.0 * area)
        if circum_r < 1.0 / alpha:
            for i, j in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((simplex[i], simplex[j])))
                if edge not in edges:
                    edges.add(edge)
                    edge_points.append(points[[simplex[i], simplex[j]]])

    from shapely.geometry import MultiLineString
    from shapely.ops import polygonize, unary_union

    lines = MultiLineString(edge_points)
    triangles = list(polygonize(lines))
    if not triangles:
        return MultiPoint([Point(p) for p in points]).convex_hull
    return unary_union(triangles)


# ---------------------------------------------------------------------------
# TradeAreaAnalyzer
# ---------------------------------------------------------------------------

class TradeAreaAnalyzer:
    """Delineate and analyse trade areas from gravity-model probabilities.

    Parameters
    ----------
    default_levels : list[float]
        Default contour levels used when *levels* is not supplied explicitly.
    alpha_shape_alpha : float
        Alpha parameter for concave-hull generation.  0 uses convex hull.
    """

    def __init__(
        self,
        default_levels: Optional[list[float]] = None,
        alpha_shape_alpha: float = 0.0,
    ) -> None:
        self.default_levels = default_levels or [0.50, 0.70, 0.90]
        self.alpha_shape_alpha = alpha_shape_alpha

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def from_probabilities(
        self,
        prob_df: pd.DataFrame,
        origins_df: pd.DataFrame,
        store_id: str,
        levels: Optional[list[float]] = None,
    ) -> list[TradeArea]:
        """Build :class:`TradeArea` objects for each contour level.

        For every level *L* in *levels* (e.g. 0.70), origins are sorted by
        descending probability-weighted demand (probability x population).
        Origins are accumulated until the cumulative weighted demand reaches
        *L* of the total weighted demand.  That subset of origins defines the
        contour.

        Parameters
        ----------
        prob_df : pd.DataFrame
            Origin x store probability matrix.
        origins_df : pd.DataFrame
            Origins with ``lat``, ``lon``, ``population``, ``households``.
        store_id : str
            Target store column in *prob_df*.
        levels : list[float] or None
            Contour thresholds (default ``[0.50, 0.70, 0.90]``).

        Returns
        -------
        list[TradeArea]
            One ``TradeArea`` pydantic model per contour level, sorted by
            ascending level.
        """
        levels = levels or self.default_levels
        probs = _validate_inputs(prob_df, origins_df, store_id)

        # Weighted demand: probability * population.
        pop = origins_df["population"].reindex(probs.index).fillna(0).astype(float)
        hh = origins_df["households"].reindex(probs.index).fillna(0).astype(float)
        weighted_demand = probs * pop
        total_demand = weighted_demand.sum()

        if total_demand <= 0:
            logger.warning(
                "Total weighted demand is zero for store '%s'. "
                "Returning empty trade areas.",
                store_id,
            )
            return [
                TradeArea(
                    store_id=store_id,
                    contour_level=lv,
                    total_population=0,
                    total_households=0,
                    penetration=0.0,
                    fair_share_index=0.0,
                    geojson=None,
                )
                for lv in sorted(levels)
            ]

        # Sort origins by descending weighted demand.
        sort_idx = weighted_demand.sort_values(ascending=False).index
        cum_demand = weighted_demand.reindex(sort_idx).cumsum()

        n_stores = prob_df.shape[1]
        total_pop = pop.sum()

        results: list[TradeArea] = []
        for lv in sorted(levels):
            # Origins within this contour: cumulative demand <= lv * total.
            mask = cum_demand <= lv * total_demand
            # Always include at least the top origin.
            if not mask.any():
                mask.iloc[0] = True
            contour_origins = mask[mask].index

            contour_pop = int(pop.reindex(contour_origins).sum())
            contour_hh = int(hh.reindex(contour_origins).sum())

            # Penetration within this contour.
            expected_customers = weighted_demand.reindex(contour_origins).sum()
            pen = float(expected_customers / contour_pop) if contour_pop > 0 else 0.0

            # Fair-share index for the contour origins.
            mean_prob = float(probs.reindex(contour_origins).mean()) if len(contour_origins) > 0 else 0.0
            expected_share = 1.0 / n_stores if n_stores > 0 else 0.0
            fsi = float(mean_prob / expected_share) if expected_share > 0 else 0.0

            # Build GeoJSON convex/alpha hull of contour origins.
            contour_latlons = origins_df.loc[
                origins_df.index.isin(contour_origins), ["lon", "lat"]
            ].values
            geojson_dict = None
            if len(contour_latlons) >= 1:
                shape = _alpha_shape(contour_latlons, alpha=self.alpha_shape_alpha)
                geojson_dict = mapping(shape)

            results.append(
                TradeArea(
                    store_id=store_id,
                    contour_level=lv,
                    total_population=contour_pop,
                    total_households=contour_hh,
                    penetration=round(pen, 6),
                    fair_share_index=round(fsi, 4),
                    geojson=geojson_dict,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Contour identification
    # ------------------------------------------------------------------

    def probability_contours(
        self,
        prob_df: pd.DataFrame,
        origins_df: pd.DataFrame,
        store_id: str,
        levels: Optional[list[float]] = None,
    ) -> dict[float, list[str]]:
        """Identify origin IDs that fall within each contour level.

        Origins are ranked by descending probability-weighted demand
        (probability x population).  For each level *L*, origins are
        accumulated until cumulative demand reaches *L* of total demand.

        Parameters
        ----------
        prob_df : pd.DataFrame
            Origin x store probability matrix.
        origins_df : pd.DataFrame
            Origins DataFrame.
        store_id : str
            Target store.
        levels : list[float] or None
            Contour thresholds.

        Returns
        -------
        dict[float, list[str]]
            Mapping from contour level to list of origin_ids within that
            contour.
        """
        levels = levels or self.default_levels
        probs = _validate_inputs(prob_df, origins_df, store_id)

        pop = origins_df["population"].reindex(probs.index).fillna(0).astype(float)
        weighted_demand = probs * pop
        total_demand = weighted_demand.sum()

        if total_demand <= 0:
            return {lv: [] for lv in sorted(levels)}

        sort_idx = weighted_demand.sort_values(ascending=False).index
        cum_demand = weighted_demand.reindex(sort_idx).cumsum()

        contours: dict[float, list[str]] = {}
        for lv in sorted(levels):
            mask = cum_demand <= lv * total_demand
            if not mask.any():
                mask.iloc[0] = True
            contours[lv] = list(mask[mask].index)

        return contours

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_penetration(
        self,
        store_id: str,
        prob_df: pd.DataFrame,
        origins_df: pd.DataFrame,
    ) -> float:
        """Market penetration = expected_customers / total_population.

        Expected customers for a store equals the sum of
        ``P(origin_i -> store) * population_i`` across all origins.

        Parameters
        ----------
        store_id : str
            Target store.
        prob_df : pd.DataFrame
            Origin x store probability matrix.
        origins_df : pd.DataFrame
            Origins with ``population`` column.

        Returns
        -------
        float
            Market penetration ratio in [0, 1].
        """
        probs = _validate_inputs(prob_df, origins_df, store_id)
        pop = origins_df["population"].reindex(probs.index).fillna(0).astype(float)

        total_pop = pop.sum()
        if total_pop <= 0:
            return 0.0

        expected_customers = (probs * pop).sum()
        return float(expected_customers / total_pop)

    def fair_share_index(
        self,
        store_id: str,
        prob_df: pd.DataFrame,
        n_stores: Optional[int] = None,
    ) -> float:
        """Fair-share index: actual share vs. expected equal share (1/N).

        An index > 1 means the store captures more than its equal share
        of demand; < 1 means underperforming.

        Parameters
        ----------
        store_id : str
            Target store.
        prob_df : pd.DataFrame
            Origin x store probability matrix.
        n_stores : int or None
            Number of competing stores.  Defaults to ``prob_df.shape[1]``.

        Returns
        -------
        float
            Fair-share index.
        """
        if store_id not in prob_df.columns:
            raise KeyError(
                f"store_id '{store_id}' not found in prob_df columns."
            )

        n = n_stores if n_stores is not None else prob_df.shape[1]
        if n <= 0:
            return 0.0

        expected_share = 1.0 / n
        actual_share = float(prob_df[store_id].mean())
        return actual_share / expected_share if expected_share > 0 else 0.0

    # ------------------------------------------------------------------
    # GeoDataFrame export
    # ------------------------------------------------------------------

    def to_geodataframe(
        self,
        prob_df: pd.DataFrame,
        origins_df: pd.DataFrame,
        store_id: str,
    ) -> gpd.GeoDataFrame:
        """Build a GeoDataFrame of origins with probabilities for one store.

        Suitable for choropleth mapping: each origin is a Point feature with
        columns ``probability``, ``population``, ``households``, and
        ``weighted_demand``.

        Parameters
        ----------
        prob_df : pd.DataFrame
            Origin x store probability matrix.
        origins_df : pd.DataFrame
            Origins with ``lat``, ``lon``, ``population``, ``households``.
        store_id : str
            Target store.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with Point geometries (EPSG:4326).
        """
        probs = _validate_inputs(prob_df, origins_df, store_id)

        gdf = origins_df[["lat", "lon", "population", "households"]].copy()
        gdf["probability"] = probs
        gdf["weighted_demand"] = probs * gdf["population"].astype(float)

        geometry = gpd.points_from_xy(gdf["lon"], gdf["lat"])
        gdf = gpd.GeoDataFrame(gdf, geometry=geometry, crs="EPSG:4326")
        return gdf

    # ------------------------------------------------------------------
    # Contour GeoJSON generation
    # ------------------------------------------------------------------

    def generate_contour_geojson(
        self,
        prob_df: pd.DataFrame,
        origins_df: pd.DataFrame,
        store_id: str,
        levels: Optional[list[float]] = None,
    ) -> dict:
        """Generate GeoJSON FeatureCollection of contour polygons.

        Each contour level produces a Polygon (convex hull or alpha shape)
        enclosing the origins within that contour.  The resulting GeoJSON
        contains one Feature per level, with properties ``contour_level``,
        ``n_origins``, ``total_population``, and ``total_households``.

        Parameters
        ----------
        prob_df : pd.DataFrame
            Origin x store probability matrix.
        origins_df : pd.DataFrame
            Origins DataFrame.
        store_id : str
            Target store.
        levels : list[float] or None
            Contour thresholds (default ``[0.50, 0.70, 0.90]``).

        Returns
        -------
        dict
            GeoJSON-compliant FeatureCollection dictionary.
        """
        levels = levels or self.default_levels
        contours = self.probability_contours(prob_df, origins_df, store_id, levels)

        pop = origins_df["population"].fillna(0).astype(float)
        hh = origins_df["households"].fillna(0).astype(float)

        features: list[dict] = []
        for lv in sorted(levels):
            origin_ids = contours.get(lv, [])
            if not origin_ids:
                continue

            pts = origins_df.loc[
                origins_df.index.isin(origin_ids), ["lon", "lat"]
            ].values

            if len(pts) == 0:
                continue

            shape = _alpha_shape(pts, alpha=self.alpha_shape_alpha)
            geom = mapping(shape)

            feature = {
                "type": "Feature",
                "properties": {
                    "store_id": store_id,
                    "contour_level": lv,
                    "n_origins": len(origin_ids),
                    "total_population": int(pop.reindex(origin_ids).sum()),
                    "total_households": int(hh.reindex(origin_ids).sum()),
                },
                "geometry": geom,
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TradeAreaAnalyzer(default_levels={self.default_levels}, "
            f"alpha_shape_alpha={self.alpha_shape_alpha})"
        )
