"""
Scenario Simulator
==================
What-if scenario modelling for retail network planning.

Given a fitted gravity model (or any callable that produces an origin x store
probability matrix), the simulator evaluates the impact of hypothetical
changes to the store network:

    - **New store**: open a store at a given location with specified
      attractiveness and attributes.
    - **Close store**: remove a store and redistribute demand.
    - **Modify store**: change a store's attractiveness, size, or other
      attributes.
    - **Competitor entry**: simulate a competitor opening nearby.

Each scenario returns:

    - Market share redistribution (before vs. after per store).
    - Cannibalization matrix (how much each existing store loses to the
      new/modified store).
    - Revenue impact estimates (change in weighted demand per store).

Multiple scenarios can be compared side-by-side.

Usage
-----
>>> from gravity.reporting.scenario import ScenarioSimulator
>>> sim = ScenarioSimulator(predict_fn=model.predict,
...                         origins_df=origins, stores_df=stores)
>>> result = sim.new_store(location=(40.7, -74.0),
...                        attractiveness=15000,
...                        attributes={"name": "New Downtown"})
>>> comp = sim.compare_scenarios([result, close_result])
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario result container
# ---------------------------------------------------------------------------

class ScenarioResult:
    """Container for a single scenario simulation result.

    Attributes
    ----------
    name : str
        Human-readable scenario name.
    scenario_type : str
        One of ``"new_store"``, ``"close_store"``, ``"modify_store"``,
        ``"competitor_entry"``.
    baseline_shares : pd.Series
        Market share per store before the scenario.
    scenario_shares : pd.Series
        Market share per store after the scenario.
    share_change : pd.Series
        Absolute change in market share per store.
    cannibalization : pd.DataFrame
        Cannibalization matrix: rows = existing stores, columns =
        affected store(s).  Each cell is the fraction of demand
        transferred.
    revenue_impact : pd.Series
        Change in weighted demand (probability x population) per store.
    baseline_probs : pd.DataFrame
        Full origin x store probability matrix before the scenario.
    scenario_probs : pd.DataFrame
        Full origin x store probability matrix after the scenario.
    """

    def __init__(
        self,
        name: str,
        scenario_type: str,
        baseline_shares: pd.Series,
        scenario_shares: pd.Series,
        share_change: pd.Series,
        cannibalization: pd.DataFrame,
        revenue_impact: pd.Series,
        baseline_probs: pd.DataFrame,
        scenario_probs: pd.DataFrame,
    ) -> None:
        self.name = name
        self.scenario_type = scenario_type
        self.baseline_shares = baseline_shares
        self.scenario_shares = scenario_shares
        self.share_change = share_change
        self.cannibalization = cannibalization
        self.revenue_impact = revenue_impact
        self.baseline_probs = baseline_probs
        self.scenario_probs = scenario_probs

    def summary(self) -> pd.DataFrame:
        """One-row-per-store summary DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``baseline_share``, ``scenario_share``,
            ``share_change``, ``revenue_impact``.
        """
        return pd.DataFrame({
            "baseline_share": self.baseline_shares,
            "scenario_share": self.scenario_shares,
            "share_change": self.share_change,
            "revenue_impact": self.revenue_impact,
        })

    def __repr__(self) -> str:
        return (
            f"ScenarioResult(name='{self.name}', type='{self.scenario_type}', "
            f"stores={len(self.scenario_shares)})"
        )


# ---------------------------------------------------------------------------
# ScenarioSimulator
# ---------------------------------------------------------------------------

class ScenarioSimulator:
    """What-if scenario modelling for store network changes.

    Parameters
    ----------
    predict_fn : callable
        A function with signature ``predict_fn(origins_df, stores_df) ->
        pd.DataFrame`` that returns an origin x store probability matrix.
        This is typically ``model.predict`` from a fitted gravity model.
    origins_df : pd.DataFrame
        Origins table indexed by ``origin_id`` with at least ``lat``,
        ``lon``, ``population``.
    stores_df : pd.DataFrame
        Stores table indexed by ``store_id`` with at least ``lat``,
        ``lon``, and the attractiveness column (e.g. ``square_footage``).
    attractiveness_col : str, default "square_footage"
        Column name used for store attractiveness.
    avg_revenue_per_visitor : float, default 50.0
        Average revenue per expected visitor, used to translate demand
        changes into dollar estimates.
    """

    def __init__(
        self,
        predict_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        attractiveness_col: str = "square_footage",
        avg_revenue_per_visitor: float = 50.0,
    ) -> None:
        self.predict_fn = predict_fn
        self.origins_df = origins_df.copy()
        self.stores_df = stores_df.copy()
        self.attractiveness_col = attractiveness_col
        self.avg_revenue_per_visitor = avg_revenue_per_visitor

        # Cache baseline predictions.
        self._baseline_probs: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    @property
    def baseline_probs(self) -> pd.DataFrame:
        """Baseline origin x store probability matrix (computed once).

        Returns
        -------
        pd.DataFrame
            Cached baseline predictions.
        """
        if self._baseline_probs is None:
            self._baseline_probs = self.predict_fn(
                self.origins_df, self.stores_df
            )
        return self._baseline_probs

    def reset_baseline(self) -> None:
        """Force recomputation of the baseline on next access."""
        self._baseline_probs = None

    # ------------------------------------------------------------------
    # Scenario methods
    # ------------------------------------------------------------------

    def new_store(
        self,
        location: tuple[float, float],
        attractiveness: float,
        attributes: Optional[dict[str, Any]] = None,
        store_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> ScenarioResult:
        """Simulate the impact of opening a new store.

        Parameters
        ----------
        location : tuple[float, float]
            ``(lat, lon)`` of the new store.
        attractiveness : float
            Value for the attractiveness column (e.g. square footage).
        attributes : dict or None
            Additional store attributes to set (column values).
        store_id : str or None
            Identifier for the new store.  Auto-generated if None.
        name : str or None
            Display name for the scenario.

        Returns
        -------
        ScenarioResult
            Simulation results including redistribution and cannibalization.
        """
        store_id = store_id or f"new_store_{len(self.stores_df) + 1}"
        name = name or f"Open {store_id}"

        # Build new store row.
        new_row = {
            "lat": location[0],
            "lon": location[1],
            self.attractiveness_col: attractiveness,
        }
        if attributes:
            new_row.update(attributes)

        new_store_df = pd.DataFrame([new_row], index=pd.Index([store_id]))
        # Ensure all columns from stores_df exist.
        for col in self.stores_df.columns:
            if col not in new_store_df.columns:
                new_store_df[col] = self.stores_df[col].iloc[0] if pd.api.types.is_numeric_dtype(self.stores_df[col]) else None
        new_store_df = new_store_df.reindex(columns=self.stores_df.columns, fill_value=0)

        scenario_stores = pd.concat([self.stores_df, new_store_df])
        scenario_probs = self.predict_fn(self.origins_df, scenario_stores)

        return self._build_result(
            name=name,
            scenario_type="new_store",
            scenario_probs=scenario_probs,
            target_store_id=store_id,
        )

    def close_store(
        self,
        store_id: str,
        name: Optional[str] = None,
    ) -> ScenarioResult:
        """Simulate the impact of closing an existing store.

        Parameters
        ----------
        store_id : str
            Identifier of the store to close.
        name : str or None
            Display name for the scenario.

        Returns
        -------
        ScenarioResult
            Simulation results showing demand redistribution.

        Raises
        ------
        KeyError
            If *store_id* is not in the stores DataFrame.
        """
        if store_id not in self.stores_df.index:
            raise KeyError(f"store_id '{store_id}' not in stores_df.")

        name = name or f"Close {store_id}"

        scenario_stores = self.stores_df.drop(index=store_id)
        scenario_probs = self.predict_fn(self.origins_df, scenario_stores)

        return self._build_result(
            name=name,
            scenario_type="close_store",
            scenario_probs=scenario_probs,
            target_store_id=store_id,
        )

    def modify_store(
        self,
        store_id: str,
        changes: dict[str, Any],
        name: Optional[str] = None,
    ) -> ScenarioResult:
        """Simulate the impact of changing an existing store's attributes.

        Parameters
        ----------
        store_id : str
            Identifier of the store to modify.
        changes : dict
            Column-value pairs to update.  For example,
            ``{"square_footage": 25000}`` to simulate a store expansion.
        name : str or None
            Display name for the scenario.

        Returns
        -------
        ScenarioResult
            Simulation results.

        Raises
        ------
        KeyError
            If *store_id* is not in the stores DataFrame.
        """
        if store_id not in self.stores_df.index:
            raise KeyError(f"store_id '{store_id}' not in stores_df.")

        name = name or f"Modify {store_id}"

        scenario_stores = self.stores_df.copy()
        for col, val in changes.items():
            if col in scenario_stores.columns:
                scenario_stores.at[store_id, col] = val
            else:
                logger.warning(
                    "Column '%s' not in stores_df; adding it.", col
                )
                scenario_stores[col] = None
                scenario_stores.at[store_id, col] = val

        scenario_probs = self.predict_fn(self.origins_df, scenario_stores)

        return self._build_result(
            name=name,
            scenario_type="modify_store",
            scenario_probs=scenario_probs,
            target_store_id=store_id,
        )

    def competitor_entry(
        self,
        location: tuple[float, float],
        attractiveness: float,
        competitor_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> ScenarioResult:
        """Simulate a competitor opening nearby.

        Functionally identical to ``new_store`` but labelled as a
        competitive threat.  The competitor is added to the probability
        matrix so that the Huff model redistributes demand to include
        the new competitor.

        Parameters
        ----------
        location : tuple[float, float]
            ``(lat, lon)`` of the competitor.
        attractiveness : float
            Attractiveness value for the competitor.
        competitor_id : str or None
            Identifier.  Auto-generated if None.
        attributes : dict or None
            Additional attributes.
        name : str or None
            Scenario display name.

        Returns
        -------
        ScenarioResult
            Simulation results.
        """
        competitor_id = competitor_id or f"competitor_{len(self.stores_df) + 1}"
        name = name or f"Competitor entry: {competitor_id}"

        # Build competitor row.
        new_row = {
            "lat": location[0],
            "lon": location[1],
            self.attractiveness_col: attractiveness,
        }
        if attributes:
            new_row.update(attributes)

        comp_df = pd.DataFrame([new_row], index=pd.Index([competitor_id]))
        comp_df = comp_df.reindex(columns=self.stores_df.columns, fill_value=0)

        scenario_stores = pd.concat([self.stores_df, comp_df])
        scenario_probs = self.predict_fn(self.origins_df, scenario_stores)

        return self._build_result(
            name=name,
            scenario_type="competitor_entry",
            scenario_probs=scenario_probs,
            target_store_id=competitor_id,
        )

    # ------------------------------------------------------------------
    # Scenario comparison
    # ------------------------------------------------------------------

    def compare_scenarios(
        self, scenarios: list[ScenarioResult]
    ) -> pd.DataFrame:
        """Side-by-side comparison of multiple scenarios.

        Parameters
        ----------
        scenarios : list[ScenarioResult]
            Results from ``new_store``, ``close_store``,
            ``modify_store``, or ``competitor_entry``.

        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame with columns grouped by scenario name.
            Each group has ``scenario_share``, ``share_change``, and
            ``revenue_impact``.
        """
        # Collect all store IDs across scenarios.
        all_stores: set[str] = set()
        for sc in scenarios:
            all_stores.update(sc.baseline_shares.index)
            all_stores.update(sc.scenario_shares.index)
        all_stores_sorted = sorted(all_stores)

        frames = {}
        for sc in scenarios:
            df = pd.DataFrame(index=all_stores_sorted)
            df["scenario_share"] = sc.scenario_shares.reindex(all_stores_sorted, fill_value=0.0)
            df["share_change"] = sc.share_change.reindex(all_stores_sorted, fill_value=0.0)
            df["revenue_impact"] = sc.revenue_impact.reindex(all_stores_sorted, fill_value=0.0)
            frames[sc.name] = df

        # Also include baseline.
        baseline_shares = scenarios[0].baseline_shares.reindex(
            all_stores_sorted, fill_value=0.0
        )
        baseline_df = pd.DataFrame(index=all_stores_sorted)
        baseline_df["share"] = baseline_shares
        frames = {"Baseline": baseline_df, **frames}

        result = pd.concat(frames, axis=1)
        result.index.name = "store_id"
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        name: str,
        scenario_type: str,
        scenario_probs: pd.DataFrame,
        target_store_id: str,
    ) -> ScenarioResult:
        """Build a ScenarioResult from baseline and scenario predictions.

        Parameters
        ----------
        name : str
            Scenario name.
        scenario_type : str
            Scenario type label.
        scenario_probs : pd.DataFrame
            Origin x store probabilities after the scenario.
        target_store_id : str
            The store that was added, removed, or modified.

        Returns
        -------
        ScenarioResult
        """
        baseline = self.baseline_probs
        pop = self.origins_df["population"].reindex(
            baseline.index
        ).fillna(0).astype(float)

        # --- Market shares ---
        baseline_demand = baseline.mul(pop, axis=0).sum()
        baseline_total = baseline_demand.sum()
        baseline_shares = baseline_demand / baseline_total if baseline_total > 0 else baseline_demand * 0.0

        # For scenario, align population to scenario's origin index.
        sc_pop = self.origins_df["population"].reindex(
            scenario_probs.index
        ).fillna(0).astype(float)
        scenario_demand = scenario_probs.mul(sc_pop, axis=0).sum()
        scenario_total = scenario_demand.sum()
        scenario_shares = scenario_demand / scenario_total if scenario_total > 0 else scenario_demand * 0.0

        # Align indices for comparison.
        all_stores = baseline_shares.index.union(scenario_shares.index)
        baseline_shares = baseline_shares.reindex(all_stores, fill_value=0.0)
        scenario_shares = scenario_shares.reindex(all_stores, fill_value=0.0)
        baseline_demand = baseline_demand.reindex(all_stores, fill_value=0.0)
        scenario_demand = scenario_demand.reindex(all_stores, fill_value=0.0)

        share_change = scenario_shares - baseline_shares
        revenue_impact = (scenario_demand - baseline_demand) * self.avg_revenue_per_visitor

        # --- Cannibalization matrix ---
        cannibalization = self._compute_cannibalization(
            baseline, scenario_probs, pop, target_store_id
        )

        return ScenarioResult(
            name=name,
            scenario_type=scenario_type,
            baseline_shares=baseline_shares,
            scenario_shares=scenario_shares,
            share_change=share_change,
            cannibalization=cannibalization,
            revenue_impact=revenue_impact,
            baseline_probs=baseline,
            scenario_probs=scenario_probs,
        )

    def _compute_cannibalization(
        self,
        baseline_probs: pd.DataFrame,
        scenario_probs: pd.DataFrame,
        pop: pd.Series,
        target_store_id: str,
    ) -> pd.DataFrame:
        """Compute the cannibalization matrix.

        For each existing store, computes the fraction of its demand
        that was diverted to the target store (new/modified) in the
        scenario.

        Parameters
        ----------
        baseline_probs : pd.DataFrame
            Baseline origin x store probabilities.
        scenario_probs : pd.DataFrame
            Scenario origin x store probabilities.
        pop : pd.Series
            Population per origin.
        target_store_id : str
            The store that was added/modified.

        Returns
        -------
        pd.DataFrame
            Cannibalization matrix.  Index = existing stores,
            column = ``target_store_id``.  Values are fractions
            of demand diverted (positive means demand lost to target).
        """
        # Common origins.
        common_origins = baseline_probs.index.intersection(scenario_probs.index)
        common_stores = baseline_probs.columns.intersection(scenario_probs.columns)

        base = baseline_probs.loc[common_origins, common_stores]
        scen = scenario_probs.loc[common_origins, common_stores]
        p = pop.reindex(common_origins).fillna(0).astype(float)

        # Demand change per store.
        base_demand = base.mul(p, axis=0).sum()
        scen_demand = scen.mul(p, axis=0).sum()
        demand_lost = base_demand - scen_demand

        # Express as fraction of baseline demand.
        cannib_frac = demand_lost / base_demand.replace(0, np.nan)
        cannib_frac = cannib_frac.fillna(0.0)

        # Package as DataFrame with target as column.
        result = pd.DataFrame(
            {target_store_id: cannib_frac},
            index=common_stores,
        )
        result.index.name = "store_id"
        return result

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ScenarioSimulator(n_origins={len(self.origins_df)}, "
            f"n_stores={len(self.stores_df)}, "
            f"attractiveness='{self.attractiveness_col}')"
        )
