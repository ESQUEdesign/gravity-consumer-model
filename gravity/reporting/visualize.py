"""
Visualization Module
====================
Maps and charts for gravity model outputs.

Map methods return ``folium.Map`` objects for interactive exploration:

    - **probability_heatmap**: choropleth of origin-level visit probabilities
      for a single store.
    - **trade_area_map**: multi-store map with trade area contour overlays
      and store markers.

Chart methods return ``matplotlib.figure.Figure`` objects:

    - **segment_composition_chart**: pie and bar chart of segment
      distribution.
    - **temporal_trends**: line chart of HMM state distribution over time.
    - **scenario_comparison_chart**: grouped bar chart comparing market
      shares across what-if scenarios.

Usage
-----
>>> from gravity.reporting.visualize import Visualizer
>>> viz = Visualizer()
>>> m = viz.probability_heatmap(prob_df, origins_df, "store_001")
>>> m.save("heatmap.html")
>>> fig = viz.segment_composition_chart(segments_df)
>>> fig.savefig("segments.png", dpi=150)

Dependencies
------------
- folium (maps)
- matplotlib, seaborn (charts)
- numpy, pandas (data handling)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports -- fail gracefully with informative messages
# ---------------------------------------------------------------------------

try:
    import folium
    from folium.plugins import HeatMap
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server-side rendering.
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

_CONTOUR_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]
_SEGMENT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _require_folium() -> None:
    """Raise ImportError if folium is not installed."""
    if not _HAS_FOLIUM:
        raise ImportError(
            "folium is required for map visualizations.  "
            "Install it with: pip install folium"
        )


def _require_matplotlib() -> None:
    """Raise ImportError if matplotlib is not installed."""
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for chart visualizations.  "
            "Install it with: pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class Visualizer:
    """Maps and charts for gravity model outputs.

    Parameters
    ----------
    map_tiles : str, default "CartoDB positron"
        Tile layer for folium maps.
    map_zoom : int, default 11
        Default zoom level for maps.
    figsize : tuple[float, float], default (10, 6)
        Default figure size for matplotlib charts.
    style : str, default "seaborn-v0_8-whitegrid"
        Matplotlib style sheet.  Falls back to ``"ggplot"`` if
        seaborn styles are not available.
    """

    def __init__(
        self,
        map_tiles: str = "CartoDB positron",
        map_zoom: int = 11,
        figsize: tuple[float, float] = (10, 6),
        style: str = "seaborn-v0_8-whitegrid",
    ) -> None:
        self.map_tiles = map_tiles
        self.map_zoom = map_zoom
        self.figsize = figsize
        self.style = style

    # ------------------------------------------------------------------
    # Map: Probability Heatmap
    # ------------------------------------------------------------------

    def probability_heatmap(
        self,
        prob_df: pd.DataFrame,
        origins_df: pd.DataFrame,
        store_id: str,
        radius: int = 15,
        blur: int = 10,
        max_opacity: float = 0.8,
    ) -> "folium.Map":
        """Folium heatmap of visit probabilities for a single store.

        Each origin contributes a heat point weighted by its probability
        of visiting *store_id*.

        Parameters
        ----------
        prob_df : pd.DataFrame
            Origin x store probability matrix.
        origins_df : pd.DataFrame
            Origins with ``lat`` and ``lon`` columns.
        store_id : str
            Target store (column in *prob_df*).
        radius : int, default 15
            Heatmap point radius in pixels.
        blur : int, default 10
            Heatmap blur radius.
        max_opacity : float, default 0.8
            Maximum opacity for the heatmap layer.

        Returns
        -------
        folium.Map
            Interactive map with heatmap layer.

        Raises
        ------
        KeyError
            If *store_id* is not in *prob_df* columns.
        ImportError
            If folium is not installed.
        """
        _require_folium()

        if store_id not in prob_df.columns:
            raise KeyError(f"store_id '{store_id}' not in prob_df columns.")

        probs = prob_df[store_id].reindex(origins_df.index).fillna(0.0)

        # Filter to origins with non-negligible probability.
        mask = probs > 1e-6
        lats = origins_df.loc[mask, "lat"].values
        lons = origins_df.loc[mask, "lon"].values
        weights = probs[mask].values

        center_lat = float(lats.mean()) if len(lats) > 0 else 0.0
        center_lon = float(lons.mean()) if len(lons) > 0 else 0.0

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=self.map_zoom,
            tiles=self.map_tiles,
        )

        # Build heat data: [lat, lon, weight].
        heat_data = [
            [float(lat), float(lon), float(w)]
            for lat, lon, w in zip(lats, lons, weights)
        ]

        if heat_data:
            HeatMap(
                heat_data,
                radius=radius,
                blur=blur,
                max_val=float(weights.max()) if len(weights) > 0 else 1.0,
                max_zoom=18,
            ).add_to(m)

        # Add store marker.
        if store_id in origins_df.index:
            store_lat = float(origins_df.loc[store_id, "lat"])
            store_lon = float(origins_df.loc[store_id, "lon"])
        else:
            store_lat, store_lon = center_lat, center_lon

        folium.Marker(
            location=[center_lat, center_lon],
            popup=f"Store: {store_id}",
            icon=folium.Icon(color="red", icon="shopping-cart", prefix="fa"),
        ).add_to(m)

        return m

    # ------------------------------------------------------------------
    # Map: Trade Area Map
    # ------------------------------------------------------------------

    def trade_area_map(
        self,
        trade_areas: list[dict],
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        show_origin_markers: bool = False,
    ) -> "folium.Map":
        """Folium map with trade area contour overlays and store markers.

        Parameters
        ----------
        trade_areas : list[dict]
            List of trade area dicts, each with keys: ``store_id``,
            ``contour_level``, ``geojson`` (GeoJSON geometry dict).
            Typically produced by ``TradeAreaAnalyzer.from_probabilities``
            or similar.
        origins_df : pd.DataFrame
            Origins with ``lat``, ``lon``.
        stores_df : pd.DataFrame
            Stores with ``lat``, ``lon``, and optionally ``name``.
        show_origin_markers : bool, default False
            If True, add small circle markers for each origin.

        Returns
        -------
        folium.Map
            Interactive map.

        Raises
        ------
        ImportError
            If folium is not installed.
        """
        _require_folium()

        # Center map on the centroid of all stores.
        center_lat = float(stores_df["lat"].mean())
        center_lon = float(stores_df["lon"].mean())

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=self.map_zoom,
            tiles=self.map_tiles,
        )

        # --- Contour overlays ---
        store_ids_seen: set[str] = set()
        for i, ta in enumerate(trade_areas):
            geojson = ta.get("geojson")
            if geojson is None:
                continue

            sid = ta.get("store_id", f"store_{i}")
            level = ta.get("contour_level", 0.0)
            color = _CONTOUR_COLORS[i % len(_CONTOUR_COLORS)]

            feature = {
                "type": "Feature",
                "geometry": geojson,
                "properties": {
                    "store_id": sid,
                    "contour_level": level,
                },
            }

            folium.GeoJson(
                feature,
                style_function=lambda _, c=color, lv=level: {
                    "fillColor": c,
                    "color": c,
                    "weight": 2,
                    "fillOpacity": 0.15 + 0.15 * (1 - lv),
                },
                tooltip=f"{sid} ({level:.0%} contour)",
            ).add_to(m)

            store_ids_seen.add(sid)

        # --- Store markers ---
        for sid in stores_df.index:
            row = stores_df.loc[sid]
            name = row.get("name", sid)
            color = "red" if sid in store_ids_seen else "blue"

            folium.Marker(
                location=[float(row["lat"]), float(row["lon"])],
                popup=f"<b>{name}</b><br>ID: {sid}",
                icon=folium.Icon(color=color, icon="store", prefix="fa"),
            ).add_to(m)

        # --- Origin markers (optional) ---
        if show_origin_markers:
            for oid in origins_df.index:
                folium.CircleMarker(
                    location=[
                        float(origins_df.loc[oid, "lat"]),
                        float(origins_df.loc[oid, "lon"]),
                    ],
                    radius=3,
                    color="#555",
                    fill=True,
                    fill_opacity=0.4,
                    weight=0.5,
                ).add_to(m)

        return m

    # ------------------------------------------------------------------
    # Chart: Segment Composition
    # ------------------------------------------------------------------

    def segment_composition_chart(
        self,
        segments_df: pd.DataFrame,
        segment_col: str = "segment",
        population_col: Optional[str] = "population",
        chart_type: str = "both",
    ) -> "matplotlib.figure.Figure":
        """Matplotlib pie and/or bar chart of segment distribution.

        Parameters
        ----------
        segments_df : pd.DataFrame
            DataFrame with a segment column.
        segment_col : str, default "segment"
            Column containing segment labels.
        population_col : str or None, default "population"
            If present, weight counts by population.  If None or not
            in the DataFrame, use raw origin counts.
        chart_type : str, default "both"
            ``"pie"``, ``"bar"``, or ``"both"``.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        _require_matplotlib()

        if segment_col not in segments_df.columns:
            raise KeyError(f"Column '{segment_col}' not in DataFrame.")

        # Compute segment sizes.
        if population_col and population_col in segments_df.columns:
            sizes = segments_df.groupby(segment_col)[population_col].sum().sort_values(ascending=False)
            ylabel = "Population"
        else:
            sizes = segments_df[segment_col].value_counts().sort_values(ascending=False)
            ylabel = "Count"

        labels = sizes.index.tolist()
        values = sizes.values.astype(float)
        colors = _SEGMENT_PALETTE[:len(labels)]

        if chart_type == "both":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.3, self.figsize[1]))
        elif chart_type == "pie":
            fig, ax1 = plt.subplots(figsize=self.figsize)
            ax2 = None
        else:
            fig, ax2 = plt.subplots(figsize=self.figsize)
            ax1 = None

        # Pie chart.
        if ax1 is not None:
            wedges, texts, autotexts = ax1.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                pctdistance=0.8,
            )
            for txt in autotexts:
                txt.set_fontsize(8)
            ax1.set_title("Segment Composition", fontsize=12, fontweight="bold")

        # Bar chart.
        if ax2 is not None:
            bars = ax2.barh(labels[::-1], values[::-1], color=colors[::-1])
            ax2.set_xlabel(ylabel)
            ax2.set_title("Segment Size", fontsize=12, fontweight="bold")
            # Add value labels.
            for bar, val in zip(bars, values[::-1]):
                ax2.text(
                    bar.get_width() + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:,.0f}",
                    va="center",
                    fontsize=8,
                )

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Chart: Temporal Trends
    # ------------------------------------------------------------------

    def temporal_trends(
        self,
        states_over_time: pd.DataFrame,
        title: str = "Loyalty State Distribution Over Time",
    ) -> "matplotlib.figure.Figure":
        """Line chart of state distribution over time.

        Parameters
        ----------
        states_over_time : pd.DataFrame
            DataFrame where the index represents time periods (e.g.
            months) and columns are state labels (e.g. "Loyal",
            "Exploring", "Lapsing").  Values are proportions or counts
            at each time step.
        title : str
            Chart title.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        _require_matplotlib()

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = _SEGMENT_PALETTE[:len(states_over_time.columns)]
        for i, col in enumerate(states_over_time.columns):
            ax.plot(
                states_over_time.index,
                states_over_time[col],
                label=col,
                color=colors[i % len(colors)],
                linewidth=2,
                marker="o",
                markersize=4,
            )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Proportion" if states_over_time.max().max() <= 1.0 else "Count")
        ax.legend(loc="best", frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels if they are long.
        if len(states_over_time.index) > 10:
            plt.xticks(rotation=45, ha="right")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Chart: Scenario Comparison
    # ------------------------------------------------------------------

    def scenario_comparison_chart(
        self,
        scenarios: list,
        metric: str = "scenario_share",
        top_n: int = 15,
        title: Optional[str] = None,
    ) -> "matplotlib.figure.Figure":
        """Grouped bar chart comparing market shares across scenarios.

        Parameters
        ----------
        scenarios : list[ScenarioResult]
            List of ScenarioResult objects (from the scenario module).
        metric : str, default "scenario_share"
            Which metric to plot.  One of ``"scenario_share"``,
            ``"share_change"``, ``"revenue_impact"``.
        top_n : int, default 15
            Show only the top N stores by baseline share.
        title : str or None
            Chart title.  Auto-generated if None.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        _require_matplotlib()

        if not scenarios:
            raise ValueError("At least one scenario is required.")

        title = title or f"Scenario Comparison: {metric.replace('_', ' ').title()}"

        # Collect data.
        all_stores: set[str] = set()
        for sc in scenarios:
            all_stores.update(sc.baseline_shares.index)
        all_stores_sorted = sorted(all_stores)

        # Select top_n stores by baseline share.
        baseline = scenarios[0].baseline_shares.reindex(all_stores_sorted, fill_value=0.0)
        top_stores = baseline.nlargest(top_n).index.tolist()

        n_groups = len(top_stores)
        n_bars = len(scenarios) + 1  # +1 for baseline
        bar_width = 0.8 / n_bars
        x = np.arange(n_groups)

        fig, ax = plt.subplots(figsize=(max(self.figsize[0], n_groups * 0.8), self.figsize[1]))

        # Baseline bars.
        baseline_vals = baseline.reindex(top_stores).values
        ax.bar(
            x - (n_bars - 1) * bar_width / 2,
            baseline_vals if metric == "scenario_share" else np.zeros(n_groups),
            bar_width,
            label="Baseline",
            color="#cccccc",
            edgecolor="#999",
        )

        # Scenario bars.
        colors = _SEGMENT_PALETTE[:len(scenarios)]
        for i, sc in enumerate(scenarios):
            if metric == "scenario_share":
                vals = sc.scenario_shares.reindex(top_stores, fill_value=0.0).values
            elif metric == "share_change":
                vals = sc.share_change.reindex(top_stores, fill_value=0.0).values
            elif metric == "revenue_impact":
                vals = sc.revenue_impact.reindex(top_stores, fill_value=0.0).values
            else:
                raise ValueError(f"Unknown metric '{metric}'.")

            offset = (i + 1 - (n_bars - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                vals,
                bar_width,
                label=sc.name,
                color=colors[i % len(colors)],
                edgecolor="#444",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(top_stores, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="best", frameon=True, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        # Format y-axis as percentage for share metrics.
        if metric in ("scenario_share", "share_change"):
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Visualizer(tiles='{self.map_tiles}', "
            f"figsize={self.figsize})"
        )
