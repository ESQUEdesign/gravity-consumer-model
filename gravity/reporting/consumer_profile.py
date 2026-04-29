"""
Consumer Profile Report
=======================
Segment-level consumer profiling that integrates data from multiple model
layers: geodemographic segments, RFM scores, CLV estimates, and HMM
loyalty states.

Each segment receives a rich profile containing:

    - **Size and share**: number of consumers and percentage of total.
    - **Average CLV**: expected customer lifetime value.
    - **Loyalty state distribution**: proportions in each HMM state
      (Loyal, Exploring, Lapsing, Churned, Won_back).
    - **RFM breakdown**: distribution of R, F, M quintile scores.
    - **Demographic summary**: weighted averages of demographic
      attributes.

The ``migration_analysis`` method compares segment membership at two
points in time to compute a transition matrix showing how consumers
moved between segments.

Usage
-----
>>> from gravity.reporting.consumer_profile import ConsumerProfileReport
>>> report = ConsumerProfileReport()
>>> report.generate(
...     segments_df=segments_df,
...     rfm_scores=rfm_df,
...     clv_data=clv_df,
...     hmm_states=hmm_df,
... )
>>> print(report.to_dict())
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
# ConsumerProfileReport
# ---------------------------------------------------------------------------

class ConsumerProfileReport:
    """Segment-level consumer profiling report.

    Aggregates consumer-level data (segments, RFM, CLV, HMM states)
    into per-segment profiles suitable for strategic analysis.
    """

    def __init__(self) -> None:
        self._report: Optional[dict] = None

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        segments_df: pd.DataFrame,
        rfm_scores: Optional[pd.DataFrame] = None,
        clv_data: Optional[pd.DataFrame] = None,
        hmm_states: Optional[pd.DataFrame] = None,
    ) -> "ConsumerProfileReport":
        """Compile segment profiles from consumer-level data.

        Parameters
        ----------
        segments_df : pd.DataFrame
            Consumer segment membership.  Must be indexed by
            ``consumer_id`` (or ``origin_id``) with at least a
            ``segment`` column.  Optional demographic columns
            (``median_income``, ``age``, etc.) are included in
            profile summaries when present.
        rfm_scores : pd.DataFrame or None
            RFM score table indexed by consumer_id with columns
            ``R``, ``F``, ``M`` (integer quintile scores 1-5) and
            optionally ``rfm_segment`` (label like "Champions").
        clv_data : pd.DataFrame or None
            CLV estimates indexed by consumer_id with at least a
            ``clv`` column.  Optional: ``alive_probability``,
            ``expected_purchases``.
        hmm_states : pd.DataFrame or None
            HMM state assignments indexed by consumer_id with at
            least a ``state`` column containing state labels
            (e.g. "Loyal", "Exploring", "Lapsing", "Churned",
            "Won_back").

        Returns
        -------
        ConsumerProfileReport
            ``self``, with the report populated.

        Raises
        ------
        KeyError
            If ``segments_df`` lacks a ``segment`` column.
        """
        if "segment" not in segments_df.columns:
            raise KeyError(
                "segments_df must contain a 'segment' column."
            )

        total_consumers = len(segments_df)
        segment_labels = sorted(segments_df["segment"].dropna().unique())

        profiles = []
        for label in segment_labels:
            mask = segments_df["segment"] == label
            seg_consumers = segments_df.index[mask]
            n = len(seg_consumers)

            profile: dict = {
                "segment": str(label),
                "size": n,
                "pct_of_total": round(n / total_consumers, 4) if total_consumers > 0 else 0.0,
            }

            # --- CLV ---
            profile["clv"] = self._aggregate_clv(seg_consumers, clv_data)

            # --- HMM loyalty states ---
            profile["loyalty_states"] = self._aggregate_hmm_states(
                seg_consumers, hmm_states
            )

            # --- RFM breakdown ---
            profile["rfm"] = self._aggregate_rfm(seg_consumers, rfm_scores)

            # --- Demographics ---
            profile["demographics"] = self._aggregate_demographics(
                seg_consumers, segments_df
            )

            profiles.append(profile)

        self._report = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_consumers": total_consumers,
                "n_segments": len(segment_labels),
            },
            "profiles": profiles,
        }

        logger.info(
            "Generated consumer profiles for %d segments (%d consumers).",
            len(segment_labels),
            total_consumers,
        )
        return self

    # ------------------------------------------------------------------
    # Migration analysis
    # ------------------------------------------------------------------

    def migration_analysis(
        self,
        segments_t1: pd.Series,
        segments_t2: pd.Series,
    ) -> dict:
        """Analyse how consumers moved between segments over time.

        Parameters
        ----------
        segments_t1 : pd.Series
            Segment assignments at time 1, indexed by consumer_id.
        segments_t2 : pd.Series
            Segment assignments at time 2, indexed by consumer_id.
            Must share consumer_ids with *segments_t1*.

        Returns
        -------
        dict
            Migration analysis with keys:

            - ``transition_matrix``: dict-of-dicts with
              transition_matrix[from_seg][to_seg] = count.
            - ``transition_rates``: same structure but normalised to
              row-proportions.
            - ``net_change``: dict mapping segment -> net gain/loss
              in consumer count.
            - ``retention_rates``: dict mapping segment -> proportion
              of consumers who stayed in the same segment.
            - ``n_matched``: number of consumers present in both
              time periods.
        """
        # Align to common consumers.
        common = segments_t1.index.intersection(segments_t2.index)
        t1 = segments_t1.reindex(common).dropna()
        t2 = segments_t2.reindex(common).dropna()
        common = t1.index.intersection(t2.index)
        t1 = t1.reindex(common)
        t2 = t2.reindex(common)

        all_segments = sorted(set(t1.unique()) | set(t2.unique()))

        # Build transition counts.
        transition_counts: dict[str, dict[str, int]] = {}
        for seg in all_segments:
            transition_counts[str(seg)] = {}
            for seg2 in all_segments:
                count = int(((t1 == seg) & (t2 == seg2)).sum())
                transition_counts[str(seg)][str(seg2)] = count

        # Normalise to rates.
        transition_rates: dict[str, dict[str, float]] = {}
        retention_rates: dict[str, float] = {}
        for seg in all_segments:
            row_total = sum(transition_counts[str(seg)].values())
            transition_rates[str(seg)] = {}
            for seg2 in all_segments:
                rate = (
                    transition_counts[str(seg)][str(seg2)] / row_total
                    if row_total > 0 else 0.0
                )
                transition_rates[str(seg)][str(seg2)] = round(rate, 4)
            retention_rates[str(seg)] = transition_rates[str(seg)].get(str(seg), 0.0)

        # Net change per segment.
        t1_counts = t1.value_counts()
        t2_counts = t2.value_counts()
        net_change: dict[str, int] = {}
        for seg in all_segments:
            before = int(t1_counts.get(seg, 0))
            after = int(t2_counts.get(seg, 0))
            net_change[str(seg)] = after - before

        result = {
            "transition_matrix": transition_counts,
            "transition_rates": transition_rates,
            "retention_rates": retention_rates,
            "net_change": net_change,
            "n_matched": len(common),
        }

        # Attach to report if available.
        if self._report is not None:
            self._report["migration"] = result

        return result

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

        Returns
        -------
        str
            Complete HTML document.
        """
        report = self.to_dict()
        meta = report["metadata"]
        profiles = report["profiles"]

        # --- Profile cards ---
        profile_cards = ""
        for p in profiles:
            # CLV summary.
            clv_info = p.get("clv", {})
            clv_str = (
                f"Mean: ${clv_info.get('mean_clv', 0):,.2f} | "
                f"Median: ${clv_info.get('median_clv', 0):,.2f} | "
                f"Total: ${clv_info.get('total_clv', 0):,.2f}"
                if clv_info.get("mean_clv") is not None
                else "No CLV data"
            )

            # Loyalty states.
            states = p.get("loyalty_states", {})
            states_rows = ""
            if states.get("distribution"):
                for state, pct in sorted(states["distribution"].items()):
                    bar_width = max(int(pct * 100), 1)
                    states_rows += (
                        f"<tr><td>{state}</td>"
                        f"<td><div style='background:#4a90d9;height:14px;"
                        f"width:{bar_width}%;border-radius:3px;'></div></td>"
                        f"<td>{pct:.1%}</td></tr>"
                    )

            # RFM.
            rfm = p.get("rfm", {})
            rfm_str = ""
            if rfm.get("mean_R") is not None:
                rfm_str = (
                    f"R: {rfm['mean_R']:.1f} | "
                    f"F: {rfm['mean_F']:.1f} | "
                    f"M: {rfm['mean_M']:.1f}"
                )
                if rfm.get("top_rfm_segments"):
                    top_segs = ", ".join(
                        f"{s['label']} ({s['pct']:.0%})"
                        for s in rfm["top_rfm_segments"][:3]
                    )
                    rfm_str += f"<br>Top segments: {top_segs}"
            else:
                rfm_str = "No RFM data"

            # Demographics.
            demo = p.get("demographics", {})
            demo_rows = ""
            for key, val in sorted(demo.items()):
                if isinstance(val, float):
                    demo_rows += f"<tr><td>{key}</td><td>{val:,.2f}</td></tr>"
                else:
                    demo_rows += f"<tr><td>{key}</td><td>{val}</td></tr>"

            profile_cards += f"""
  <div class="profile-card">
    <h3>{p['segment']}</h3>
    <div class="metric-row">
      <span class="metric"><strong>{p['size']:,}</strong> consumers</span>
      <span class="metric"><strong>{p['pct_of_total']:.1%}</strong> of total</span>
    </div>

    <h4>Customer Lifetime Value</h4>
    <p class="detail">{clv_str}</p>

    <h4>Loyalty State Distribution</h4>
    {"<table class='state-table'>" + states_rows + "</table>" if states_rows else "<p class='detail'>No HMM data</p>"}

    <h4>RFM Profile</h4>
    <p class="detail">{rfm_str}</p>

    {"<h4>Demographics</h4><table class='demo-table'>" + demo_rows + "</table>" if demo_rows else ""}
  </div>"""

        # --- Migration section (if available) ---
        migration_html = ""
        if "migration" in report:
            mig = report["migration"]
            matrix = mig["transition_rates"]
            segments = sorted(matrix.keys())

            header = "<th>From \\ To</th>" + "".join(
                f"<th>{s}</th>" for s in segments
            )
            rows = ""
            for from_seg in segments:
                cells = ""
                for to_seg in segments:
                    rate = matrix[from_seg].get(to_seg, 0.0)
                    bg = "background:#e8f5e9;" if from_seg == to_seg else ""
                    cells += f"<td style='{bg}'>{rate:.1%}</td>"
                rows += f"<tr><td><strong>{from_seg}</strong></td>{cells}</tr>"

            migration_html = f"""
  <h2>Segment Migration</h2>
  <p class="detail">Matched consumers: {mig['n_matched']:,}</p>
  <table>
    <tr>{header}</tr>
    {rows}
  </table>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Consumer Profile Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         color: #1a1a2e; background: #f5f5f5; padding: 2rem; line-height: 1.6; }}
  .container {{ max-width: 1000px; margin: 0 auto; background: #fff;
               border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
               padding: 2.5rem; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; color: #0f3460; }}
  h2 {{ font-size: 1.25rem; margin: 2rem 0 1rem; color: #16213e;
       border-bottom: 2px solid #e0e0e0; padding-bottom: 0.3rem; }}
  .subtitle {{ color: #666; font-size: 0.9rem; margin-bottom: 2rem; }}
  .profile-card {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 1.5rem;
                   margin-bottom: 1.5rem; }}
  .profile-card h3 {{ font-size: 1.1rem; color: #0f3460; margin-bottom: 0.5rem; }}
  .profile-card h4 {{ font-size: 0.9rem; color: #444; margin: 1rem 0 0.3rem;
                      border-bottom: 1px solid #eee; padding-bottom: 0.2rem; }}
  .metric-row {{ display: flex; gap: 2rem; margin-bottom: 0.5rem; }}
  .metric {{ font-size: 0.9rem; color: #555; }}
  .detail {{ font-size: 0.85rem; color: #555; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 1rem; font-size: 0.85rem; }}
  th {{ background: #16213e; color: #fff; text-align: left; padding: 0.4rem 0.6rem;
       font-weight: 600; }}
  td {{ padding: 0.35rem 0.6rem; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .state-table {{ width: auto; }}
  .state-table td {{ padding: 0.25rem 0.5rem; }}
  .demo-table {{ width: auto; max-width: 400px; }}
  .note {{ font-size: 0.8rem; color: #999; margin-top: 2rem; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <h1>Consumer Profile Report</h1>
  <p class="subtitle">Generated: {meta['generated_at']} |
     {meta['total_consumers']:,} consumers across {meta['n_segments']} segments</p>

  {profile_cards}

  {migration_html}

  <p class="note">Generated by Gravity Consumer Model Library</p>
</div>
</body>
</html>"""

        return html

    # ------------------------------------------------------------------
    # Internal aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_clv(
        self,
        consumer_ids: pd.Index,
        clv_data: Optional[pd.DataFrame],
    ) -> dict:
        """Aggregate CLV statistics for a set of consumers.

        Parameters
        ----------
        consumer_ids : pd.Index
            Consumer identifiers in this segment.
        clv_data : pd.DataFrame or None
            CLV table with ``clv`` column.

        Returns
        -------
        dict
            CLV summary statistics.
        """
        if clv_data is None or "clv" not in clv_data.columns:
            return {"mean_clv": None, "median_clv": None, "total_clv": None}

        matched = clv_data.reindex(consumer_ids).dropna(subset=["clv"])
        if matched.empty:
            return {"mean_clv": 0.0, "median_clv": 0.0, "total_clv": 0.0}

        clv_vals = matched["clv"]
        result = {
            "mean_clv": round(float(clv_vals.mean()), 2),
            "median_clv": round(float(clv_vals.median()), 2),
            "total_clv": round(float(clv_vals.sum()), 2),
            "std_clv": round(float(clv_vals.std()), 2) if len(clv_vals) > 1 else 0.0,
            "min_clv": round(float(clv_vals.min()), 2),
            "max_clv": round(float(clv_vals.max()), 2),
        }

        # Alive probability if available.
        if "alive_probability" in matched.columns:
            result["mean_alive_probability"] = round(
                float(matched["alive_probability"].mean()), 4
            )

        # Expected purchases if available.
        if "expected_purchases" in matched.columns:
            result["mean_expected_purchases"] = round(
                float(matched["expected_purchases"].mean()), 2
            )

        return result

    def _aggregate_hmm_states(
        self,
        consumer_ids: pd.Index,
        hmm_states: Optional[pd.DataFrame],
    ) -> dict:
        """Aggregate HMM loyalty state distribution for a segment.

        Parameters
        ----------
        consumer_ids : pd.Index
            Consumer identifiers in this segment.
        hmm_states : pd.DataFrame or None
            State table with ``state`` column.

        Returns
        -------
        dict
            State distribution and dominant state.
        """
        if hmm_states is None or "state" not in hmm_states.columns:
            return {"distribution": None, "dominant_state": None}

        matched = hmm_states.reindex(consumer_ids).dropna(subset=["state"])
        if matched.empty:
            return {"distribution": {}, "dominant_state": None}

        counts = matched["state"].value_counts()
        total = counts.sum()
        distribution = {
            str(state): round(float(count / total), 4)
            for state, count in counts.items()
        }

        return {
            "distribution": distribution,
            "dominant_state": str(counts.idxmax()),
            "n_matched": int(total),
        }

    def _aggregate_rfm(
        self,
        consumer_ids: pd.Index,
        rfm_scores: Optional[pd.DataFrame],
    ) -> dict:
        """Aggregate RFM score statistics for a segment.

        Parameters
        ----------
        consumer_ids : pd.Index
            Consumer identifiers.
        rfm_scores : pd.DataFrame or None
            RFM table with ``R``, ``F``, ``M`` columns.

        Returns
        -------
        dict
            Mean R/F/M scores and top RFM segment labels.
        """
        if rfm_scores is None:
            return {"mean_R": None, "mean_F": None, "mean_M": None}

        matched = rfm_scores.reindex(consumer_ids)

        result: dict = {}
        for col in ["R", "F", "M"]:
            if col in matched.columns:
                vals = matched[col].dropna()
                result[f"mean_{col}"] = round(float(vals.mean()), 2) if len(vals) > 0 else None
                result[f"median_{col}"] = round(float(vals.median()), 2) if len(vals) > 0 else None
            else:
                result[f"mean_{col}"] = None
                result[f"median_{col}"] = None

        # RFM segment distribution if available.
        if "rfm_segment" in matched.columns:
            seg_counts = matched["rfm_segment"].dropna().value_counts()
            total = seg_counts.sum()
            if total > 0:
                result["top_rfm_segments"] = [
                    {"label": str(label), "count": int(count),
                     "pct": round(float(count / total), 4)}
                    for label, count in seg_counts.head(5).items()
                ]

        return result

    def _aggregate_demographics(
        self,
        consumer_ids: pd.Index,
        segments_df: pd.DataFrame,
    ) -> dict:
        """Aggregate demographic columns for a segment.

        Parameters
        ----------
        consumer_ids : pd.Index
            Consumer identifiers.
        segments_df : pd.DataFrame
            Segments table (may contain demographic columns).

        Returns
        -------
        dict
            Averaged demographic metrics.
        """
        # Identify numeric columns that are not the segment column itself.
        matched = segments_df.loc[
            segments_df.index.isin(consumer_ids)
        ]
        if matched.empty:
            return {}

        demo_cols = [
            c for c in matched.columns
            if c != "segment" and pd.api.types.is_numeric_dtype(matched[c])
        ]

        result = {}
        for col in demo_cols:
            vals = matched[col].dropna()
            if len(vals) > 0:
                result[col] = round(float(vals.mean()), 4)

        return result

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._report is None:
            return "ConsumerProfileReport(not generated)"
        n = self._report["metadata"]["n_segments"]
        total = self._report["metadata"]["total_consumers"]
        return f"ConsumerProfileReport({n} segments, {total:,} consumers)"
