"""
Trade Area Report
=================
Generates comprehensive, self-contained trade area analysis reports for
a single store from gravity-model predictions and demographic data.

A report includes:

    - **Trade area summary**: contour populations, penetration, fair-share
      index, weighted-demand capture.
    - **Demographic profile**: income distribution, household size, age
      composition (when available in origins_df demographics).
    - **Competitive analysis**: market share by competitor, overlap
      origins, diversion ratios.
    - **Segment composition**: breakdown by geodemographic or latent-class
      segment (if segments are supplied).
    - **Market penetration**: per-origin probabilities and rank.

Output formats: ``dict``, ``json``, and a self-contained ``html`` file
with inline CSS (no external dependencies).

Usage
-----
>>> from gravity.reporting.trade_area_report import TradeAreaReport
>>> report = TradeAreaReport()
>>> report.generate(
...     store_id="store_001",
...     predictions=prob_df,
...     origins_df=origins_df,
...     stores_df=stores_df,
... )
>>> html = report.to_html()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TradeAreaReport
# ---------------------------------------------------------------------------

class TradeAreaReport:
    """Generate comprehensive trade area analysis for a single store.

    Parameters
    ----------
    contour_levels : list[float], optional
        Probability-contour thresholds used to delineate trade area
        rings.  Default ``[0.50, 0.70, 0.90]``.
    """

    def __init__(
        self,
        contour_levels: Optional[list[float]] = None,
    ) -> None:
        self.contour_levels = contour_levels or [0.50, 0.70, 0.90]
        self._report: Optional[dict] = None
        self._store_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        store_id: str,
        predictions: pd.DataFrame,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        segments: Optional[pd.Series] = None,
    ) -> "TradeAreaReport":
        """Produce a complete trade area report.

        Parameters
        ----------
        store_id : str
            Target store (must be a column in *predictions* and an
            index value in *stores_df*).
        predictions : pd.DataFrame
            Origin x store probability matrix.  Index = origin_id,
            columns = store_id.
        origins_df : pd.DataFrame
            Origins table indexed by ``origin_id`` with at least
            ``lat``, ``lon``, ``population``, ``households``.
            Optional: ``median_income``, ``demographics`` (dict column).
        stores_df : pd.DataFrame
            Stores table indexed by ``store_id`` with at least
            ``lat``, ``lon``, ``name`` (optional).
        segments : pd.Series or None
            Series indexed by ``origin_id`` mapping each origin to a
            segment label.  If supplied, segment composition is included
            in the report.

        Returns
        -------
        TradeAreaReport
            ``self``, with the report populated.

        Raises
        ------
        KeyError
            If *store_id* is not found in *predictions* or *stores_df*.
        """
        if store_id not in predictions.columns:
            raise KeyError(
                f"store_id '{store_id}' not in predictions columns."
            )
        if store_id not in stores_df.index:
            raise KeyError(
                f"store_id '{store_id}' not in stores_df index."
            )

        self._store_id = store_id
        probs = predictions[store_id].reindex(origins_df.index).fillna(0.0)

        store_row = stores_df.loc[store_id]

        self._report = {
            "metadata": self._build_metadata(store_id, store_row, origins_df),
            "trade_area_summary": self._build_trade_area_summary(
                store_id, probs, origins_df, predictions
            ),
            "demographic_profile": self._build_demographic_profile(
                probs, origins_df
            ),
            "competitive_analysis": self._build_competitive_analysis(
                store_id, predictions, origins_df
            ),
            "segment_composition": self._build_segment_composition(
                probs, origins_df, segments
            ),
            "market_penetration": self._build_penetration_detail(
                probs, origins_df
            ),
        }

        logger.info("Generated trade area report for store '%s'.", store_id)
        return self

    # ------------------------------------------------------------------
    # Output formats
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return the report as a nested dictionary.

        Returns
        -------
        dict
            Complete report structure.

        Raises
        ------
        RuntimeError
            If ``generate()`` has not been called.
        """
        if self._report is None:
            raise RuntimeError("No report generated.  Call generate() first.")
        return self._report

    def to_json(self, indent: int = 2) -> str:
        """Return the report as a JSON string.

        Parameters
        ----------
        indent : int, default 2
            JSON indentation level.

        Returns
        -------
        str
            JSON-encoded report.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_html(self) -> str:
        """Generate a self-contained HTML report with inline CSS.

        The HTML uses no external stylesheets, scripts, or images.
        Suitable for email attachments, local viewing, or embedding
        in dashboards.

        Returns
        -------
        str
            Complete HTML document.
        """
        report = self.to_dict()
        meta = report["metadata"]
        summary = report["trade_area_summary"]
        demo = report["demographic_profile"]
        comp = report["competitive_analysis"]
        seg = report["segment_composition"]
        pen = report["market_penetration"]

        store_name = meta.get("store_name") or meta["store_id"]

        # --- Build HTML sections ---
        summary_rows = ""
        for contour in summary.get("contours", []):
            summary_rows += (
                f"<tr>"
                f"<td>{contour['level']:.0%}</td>"
                f"<td>{contour['n_origins']:,}</td>"
                f"<td>{contour['population']:,}</td>"
                f"<td>{contour['households']:,}</td>"
                f"<td>{contour['penetration']:.2%}</td>"
                f"<td>{contour['fair_share_index']:.2f}</td>"
                f"</tr>"
            )

        demo_rows = ""
        for key, val in demo.items():
            if isinstance(val, float):
                demo_rows += f"<tr><td>{key}</td><td>{val:,.2f}</td></tr>"
            else:
                demo_rows += f"<tr><td>{key}</td><td>{val}</td></tr>"

        comp_rows = ""
        for comp_store in comp.get("competitors", []):
            comp_rows += (
                f"<tr>"
                f"<td>{comp_store['store_id']}</td>"
                f"<td>{comp_store['mean_probability']:.4f}</td>"
                f"<td>{comp_store['market_share']:.2%}</td>"
                f"<td>{comp_store['overlap_origins']}</td>"
                f"</tr>"
            )

        seg_rows = ""
        for s in seg.get("segments", []):
            seg_rows += (
                f"<tr>"
                f"<td>{s['segment']}</td>"
                f"<td>{s['n_origins']:,}</td>"
                f"<td>{s['pct_of_total']:.1%}</td>"
                f"<td>{s['mean_probability']:.4f}</td>"
                f"<td>{s['weighted_demand']:,.0f}</td>"
                f"</tr>"
            )

        pen_rows = ""
        for row in pen.get("origins", [])[:50]:
            pen_rows += (
                f"<tr>"
                f"<td>{row['origin_id']}</td>"
                f"<td>{row['probability']:.4f}</td>"
                f"<td>{row['population']:,}</td>"
                f"<td>{row['weighted_demand']:,.0f}</td>"
                f"<td>{row['rank']}</td>"
                f"</tr>"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Area Report: {store_name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         color: #1a1a2e; background: #f5f5f5; padding: 2rem; line-height: 1.6; }}
  .container {{ max-width: 960px; margin: 0 auto; background: #fff;
               border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
               padding: 2.5rem; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; color: #0f3460; }}
  .subtitle {{ color: #666; font-size: 0.9rem; margin-bottom: 2rem; }}
  h2 {{ font-size: 1.15rem; margin: 2rem 0 0.75rem; color: #16213e;
       border-bottom: 2px solid #e0e0e0; padding-bottom: 0.3rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; font-size: 0.88rem; }}
  th {{ background: #16213e; color: #fff; text-align: left; padding: 0.5rem 0.75rem;
       font-weight: 600; }}
  td {{ padding: 0.4rem 0.75rem; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                  gap: 1rem; margin-bottom: 1.5rem; }}
  .metric-card {{ background: #f0f4ff; border-radius: 6px; padding: 1rem; text-align: center; }}
  .metric-card .value {{ font-size: 1.4rem; font-weight: 700; color: #0f3460; }}
  .metric-card .label {{ font-size: 0.78rem; color: #666; margin-top: 0.25rem; }}
  .note {{ font-size: 0.8rem; color: #999; margin-top: 2rem; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <h1>Trade Area Report: {store_name}</h1>
  <p class="subtitle">Store ID: {meta['store_id']} | Generated: {meta['generated_at']}
     | Location: ({meta.get('lat', 'N/A')}, {meta.get('lon', 'N/A')})</p>

  <div class="metric-grid">
    <div class="metric-card">
      <div class="value">{summary.get('total_origins', 0):,}</div>
      <div class="label">Total Origins</div>
    </div>
    <div class="metric-card">
      <div class="value">{summary.get('total_population', 0):,}</div>
      <div class="label">Total Population</div>
    </div>
    <div class="metric-card">
      <div class="value">{summary.get('overall_penetration', 0):.2%}</div>
      <div class="label">Market Penetration</div>
    </div>
    <div class="metric-card">
      <div class="value">{summary.get('mean_probability', 0):.4f}</div>
      <div class="label">Mean Probability</div>
    </div>
  </div>

  <h2>Trade Area Contours</h2>
  <table>
    <tr><th>Contour</th><th>Origins</th><th>Population</th><th>Households</th>
        <th>Penetration</th><th>Fair Share Index</th></tr>
    {summary_rows}
  </table>

  <h2>Demographic Profile</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {demo_rows}
  </table>

  <h2>Competitive Analysis</h2>
  <table>
    <tr><th>Competitor</th><th>Mean Probability</th><th>Market Share</th><th>Overlap Origins</th></tr>
    {comp_rows}
  </table>

  {"<h2>Segment Composition</h2>" + '''
  <table>
    <tr><th>Segment</th><th>Origins</th><th>% of Total</th><th>Mean Prob</th><th>Weighted Demand</th></tr>
    ''' + seg_rows + "</table>" if seg_rows else ""}

  <h2>Market Penetration Detail (Top 50 Origins)</h2>
  <table>
    <tr><th>Origin</th><th>Probability</th><th>Population</th><th>Weighted Demand</th><th>Rank</th></tr>
    {pen_rows}
  </table>

  <p class="note">Generated by Gravity Consumer Model Library</p>
</div>
</body>
</html>"""

        return html

    # ------------------------------------------------------------------
    # Internal section builders
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        store_id: str,
        store_row: pd.Series,
        origins_df: pd.DataFrame,
    ) -> dict:
        """Build report metadata section."""
        return {
            "store_id": store_id,
            "store_name": store_row.get("name", None),
            "lat": float(store_row.get("lat", 0)),
            "lon": float(store_row.get("lon", 0)),
            "generated_at": datetime.utcnow().isoformat(),
            "n_origins": len(origins_df),
        }

    def _build_trade_area_summary(
        self,
        store_id: str,
        probs: pd.Series,
        origins_df: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> dict:
        """Build trade area summary with contour statistics."""
        pop = origins_df["population"].reindex(probs.index).fillna(0).astype(float)
        hh = origins_df["households"].reindex(probs.index).fillna(0).astype(float)
        weighted_demand = probs * pop
        total_demand = weighted_demand.sum()
        total_pop = pop.sum()

        # Overall metrics.
        n_stores = predictions.shape[1]
        overall_pen = float(total_demand / total_pop) if total_pop > 0 else 0.0
        mean_prob = float(probs.mean())

        # Build contours.
        sort_idx = weighted_demand.sort_values(ascending=False).index
        cum_demand = weighted_demand.reindex(sort_idx).cumsum()

        contours = []
        for level in sorted(self.contour_levels):
            if total_demand > 0:
                mask = cum_demand <= level * total_demand
                if not mask.any():
                    mask.iloc[0] = True
                contour_ids = mask[mask].index
            else:
                contour_ids = pd.Index([])

            c_pop = int(pop.reindex(contour_ids).sum()) if len(contour_ids) > 0 else 0
            c_hh = int(hh.reindex(contour_ids).sum()) if len(contour_ids) > 0 else 0
            c_demand = float(weighted_demand.reindex(contour_ids).sum()) if len(contour_ids) > 0 else 0.0
            c_pen = c_demand / c_pop if c_pop > 0 else 0.0
            c_mean = float(probs.reindex(contour_ids).mean()) if len(contour_ids) > 0 else 0.0
            fair_share = 1.0 / n_stores if n_stores > 0 else 0.0
            fsi = c_mean / fair_share if fair_share > 0 else 0.0

            contours.append({
                "level": level,
                "n_origins": len(contour_ids),
                "population": c_pop,
                "households": c_hh,
                "weighted_demand": round(c_demand, 2),
                "penetration": round(c_pen, 6),
                "fair_share_index": round(fsi, 4),
            })

        return {
            "total_origins": len(probs),
            "total_population": int(total_pop),
            "total_weighted_demand": round(float(total_demand), 2),
            "overall_penetration": round(overall_pen, 6),
            "mean_probability": round(mean_prob, 6),
            "n_competitors": n_stores - 1,
            "contours": contours,
        }

    def _build_demographic_profile(
        self,
        probs: pd.Series,
        origins_df: pd.DataFrame,
    ) -> dict:
        """Build weighted demographic profile of the trade area."""
        pop = origins_df["population"].reindex(probs.index).fillna(0).astype(float)
        weights = probs * pop
        total_weight = weights.sum()

        profile: dict = {}

        # Total population and households.
        profile["total_population"] = int(pop.sum())
        if "households" in origins_df.columns:
            profile["total_households"] = int(
                origins_df["households"].reindex(probs.index).fillna(0).sum()
            )

        # Weighted median income.
        if "median_income" in origins_df.columns:
            income = origins_df["median_income"].reindex(probs.index).fillna(0).astype(float)
            if total_weight > 0:
                profile["weighted_avg_income"] = round(
                    float((income * weights).sum() / total_weight), 2
                )
            else:
                profile["weighted_avg_income"] = 0.0

        # Extract keys from demographics dict column if present.
        if "demographics" in origins_df.columns:
            demo_keys: set[str] = set()
            for d in origins_df["demographics"].dropna():
                if isinstance(d, dict):
                    demo_keys.update(d.keys())

            for key in sorted(demo_keys):
                vals = origins_df["demographics"].apply(
                    lambda d, k=key: d.get(k, 0) if isinstance(d, dict) else 0
                ).reindex(probs.index).fillna(0).astype(float)
                if total_weight > 0:
                    profile[f"weighted_avg_{key}"] = round(
                        float((vals * weights).sum() / total_weight), 4
                    )

        return profile

    def _build_competitive_analysis(
        self,
        store_id: str,
        predictions: pd.DataFrame,
        origins_df: pd.DataFrame,
    ) -> dict:
        """Build competitive analysis: market shares, overlap, diversion."""
        pop = origins_df["population"].reindex(predictions.index).fillna(0).astype(float)
        total_demand = (predictions.mul(pop, axis=0)).sum()

        target_demand = total_demand.get(store_id, 0.0)
        grand_total = total_demand.sum()

        competitors = []
        for sid in predictions.columns:
            if sid == store_id:
                continue

            comp_probs = predictions[sid].reindex(predictions.index).fillna(0.0)
            target_probs = predictions[store_id].reindex(predictions.index).fillna(0.0)

            # Overlap: origins where both stores have > 5% probability.
            overlap_mask = (comp_probs > 0.05) & (target_probs > 0.05)
            overlap_count = int(overlap_mask.sum())

            mean_prob = float(comp_probs.mean())
            share = float(total_demand[sid] / grand_total) if grand_total > 0 else 0.0

            competitors.append({
                "store_id": sid,
                "mean_probability": round(mean_prob, 6),
                "market_share": round(share, 6),
                "weighted_demand": round(float(total_demand[sid]), 2),
                "overlap_origins": overlap_count,
            })

        # Sort by market share descending.
        competitors.sort(key=lambda x: x["market_share"], reverse=True)

        target_share = float(target_demand / grand_total) if grand_total > 0 else 0.0

        return {
            "target_store_share": round(target_share, 6),
            "target_weighted_demand": round(float(target_demand), 2),
            "n_competitors": len(competitors),
            "competitors": competitors,
        }

    def _build_segment_composition(
        self,
        probs: pd.Series,
        origins_df: pd.DataFrame,
        segments: Optional[pd.Series],
    ) -> dict:
        """Build segment composition of the trade area."""
        if segments is None:
            return {"segments": [], "note": "No segment data provided."}

        seg = segments.reindex(probs.index)
        pop = origins_df["population"].reindex(probs.index).fillna(0).astype(float)
        weighted_demand = probs * pop

        records = []
        total_origins = len(probs)
        for label in sorted(seg.dropna().unique()):
            mask = seg == label
            n = int(mask.sum())
            pct = n / total_origins if total_origins > 0 else 0.0
            mean_p = float(probs[mask].mean()) if n > 0 else 0.0
            wd = float(weighted_demand[mask].sum())

            records.append({
                "segment": str(label),
                "n_origins": n,
                "pct_of_total": round(pct, 4),
                "mean_probability": round(mean_p, 6),
                "weighted_demand": round(wd, 2),
            })

        records.sort(key=lambda x: x["weighted_demand"], reverse=True)
        return {"segments": records}

    def _build_penetration_detail(
        self,
        probs: pd.Series,
        origins_df: pd.DataFrame,
    ) -> dict:
        """Build per-origin penetration detail."""
        pop = origins_df["population"].reindex(probs.index).fillna(0).astype(float)
        weighted_demand = probs * pop

        df = pd.DataFrame({
            "probability": probs,
            "population": pop,
            "weighted_demand": weighted_demand,
        })
        df = df.sort_values("weighted_demand", ascending=False)
        df["rank"] = range(1, len(df) + 1)

        records = []
        for oid, row in df.iterrows():
            records.append({
                "origin_id": str(oid),
                "probability": round(float(row["probability"]), 6),
                "population": int(row["population"]),
                "weighted_demand": round(float(row["weighted_demand"]), 2),
                "rank": int(row["rank"]),
            })

        return {"origins": records}

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._report is None:
            return "TradeAreaReport(not generated)"
        return (
            f"TradeAreaReport(store_id='{self._store_id}', "
            f"contour_levels={self.contour_levels})"
        )
