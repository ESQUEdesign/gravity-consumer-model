"""
Gravity Consumer Model -- Main Orchestrator
============================================
User-facing API that ties all model layers together into a unified pipeline.

Usage
-----
>>> from gravity import GravityModel
>>> model = GravityModel(layers=["huff", "competing_destinations", "bayesian_update"])
>>> model.fit(stores=stores_df, origins=origins_df, observed_shares=shares)
>>> predictions = model.predict()
>>> report = model.trade_area_report("store_001")
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Subpackage imports
# ---------------------------------------------------------------------------

from gravity.core import HuffModel, CompetingDestinationsModel, GWRModel, MixedLogitModel
from gravity.segmentation import LatentClassModel, GeodemographicMapper, RFMScorer, CLVEstimator
from gravity.temporal import ConsumerHMM, HawkesProcess, BayesianUpdater
from gravity.ml import ResidualBoostModel, GraphConsumerModel
from gravity.ensemble import EnsembleAverager
from gravity.spatial import TradeAreaAnalyzer
from gravity.reporting import (
    TradeAreaReport,
    ConsumerProfileReport,
    ScenarioSimulator,
    Visualizer,
)
from gravity.data.schema import (
    Store,
    ConsumerOrigin,
    Transaction,
    VisitEvent,
    stores_to_dataframe,
    origins_to_dataframe,
    transactions_to_dataframe,
    build_distance_matrix,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate default config on disk
# ---------------------------------------------------------------------------

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "default_config.yaml")

# ---------------------------------------------------------------------------
# All available layer names (canonical order)
# ---------------------------------------------------------------------------

ALL_LAYERS: list[str] = [
    "huff",
    "competing_destinations",
    "gwr",
    "mixed_logit",
    "latent_class",
    "rfm",
    "clv",
    "hmm",
    "bayesian_update",
    "residual_boost",
    "graph_network",
    "ensemble",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(source: Union[str, dict, None]) -> dict:
    """Load and return a configuration dict.

    Parameters
    ----------
    source : str, dict, or None
        * ``str``: path to a YAML file.
        * ``dict``: used directly.
        * ``None``: loads ``config/default_config.yaml`` from the project root.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    if source is None:
        if os.path.isfile(_DEFAULT_CONFIG_PATH):
            with open(_DEFAULT_CONFIG_PATH, "r") as fh:
                return yaml.safe_load(fh) or {}
        logger.warning(
            "Default config not found at %s; using empty config.",
            _DEFAULT_CONFIG_PATH,
        )
        return {}
    if isinstance(source, dict):
        return deepcopy(source)
    if isinstance(source, str):
        with open(source, "r") as fh:
            return yaml.safe_load(fh) or {}
    raise TypeError(f"config must be a str path, dict, or None; got {type(source).__name__}")


def _to_dataframe(
    data: Union[pd.DataFrame, list, None],
    converter: Optional[Callable] = None,
    name: str = "data",
) -> Optional[pd.DataFrame]:
    """Normalise input data to a DataFrame (or None).

    Accepts a DataFrame directly, a list of Pydantic models (converted
    via *converter*), or None.
    """
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, list) and converter is not None:
        return converter(data)
    raise TypeError(
        f"{name} must be a DataFrame, list of models, or None; "
        f"got {type(data).__name__}"
    )


# ---------------------------------------------------------------------------
# GravityModel
# ---------------------------------------------------------------------------


class GravityModel:
    """Layered consumer-behaviour prediction engine.

    Orchestrates the full gravity-model stack from basic Huff spatial
    interaction through competing destinations, GWR, mixed logit, latent
    class segmentation, RFM/CLV, HMM temporal dynamics, Bayesian real-time
    updating, gradient-boosted residuals, graph networks, and Bayesian
    model averaging.

    Parameters
    ----------
    layers : list[str] or None
        Layer names to activate.  When ``None``, the list is read from the
        config file (key ``model.layers``).  Unrecognised names raise
        ``ValueError``.
    config : str, dict, or None
        YAML config path, pre-loaded dict, or ``None`` for project defaults.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        layers: Optional[list[str]] = None,
        config: Union[str, dict, None] = None,
    ) -> None:
        self.config = _load_config(config)

        # Resolve layer list.
        if layers is not None:
            self._layer_names = list(layers)
        else:
            self._layer_names = list(
                self.config.get("model", {}).get("layers", ["huff"])
            )

        # Validate.
        unknown = set(self._layer_names) - set(ALL_LAYERS)
        if unknown:
            raise ValueError(
                f"Unknown layer(s): {sorted(unknown)}.  "
                f"Available: {ALL_LAYERS}"
            )

        # ---- Instantiate requested layers --------------------------------
        self._layers: dict[str, Any] = {}
        self._instantiate_layers()

        # ---- Internal state -----------------------------------------------
        self._stores_df: Optional[pd.DataFrame] = None
        self._origins_df: Optional[pd.DataFrame] = None
        self._distance_matrix: Optional[pd.DataFrame] = None
        self._transactions_df: Optional[pd.DataFrame] = None
        self._observed_shares: Optional[pd.DataFrame] = None
        self._choice_data: Optional[pd.DataFrame] = None
        self._traffic: Optional[pd.DataFrame] = None

        # Per-layer prediction caches (origin x store probability matrices).
        self._predictions: dict[str, pd.DataFrame] = {}

        self._fitted: bool = False

        logger.info(
            "GravityModel initialised with layers: %s", self._layer_names
        )

    # ------------------------------------------------------------------ #
    # Layer factory
    # ------------------------------------------------------------------ #

    def _instantiate_layers(self) -> None:
        """Create model-layer objects for each requested layer."""
        cfg = self.config

        if "huff" in self._layer_names:
            huff_cfg = cfg.get("huff", {})
            self._layers["huff"] = HuffModel(
                alpha=huff_cfg.get("alpha", 1.0),
                lam=huff_cfg.get("lambda", 2.0),
                attractiveness=huff_cfg.get("attractiveness_field", "square_footage"),
            )

        if "competing_destinations" in self._layer_names:
            cd_cfg = cfg.get("competing_destinations", {})
            self._layers["competing_destinations"] = CompetingDestinationsModel(
                delta=cd_cfg.get("delta", 0.5),
                radius_km=cd_cfg.get("radius_km", 10.0),
            )

        if "gwr" in self._layer_names:
            gwr_cfg = cfg.get("gwr", {})
            bw_mode = gwr_cfg.get("bandwidth", "adaptive")
            gwr_kwargs: dict = {
                "kernel": gwr_cfg.get("kernel", "bisquare"),
            }
            if bw_mode == "adaptive":
                gwr_kwargs["n_neighbors"] = gwr_cfg.get("n_neighbors", 30)
            else:
                gwr_kwargs["bandwidth"] = gwr_cfg.get("fixed_bandwidth_km", 5.0)
            self._layers["gwr"] = GWRModel(**gwr_kwargs)

        if "mixed_logit" in self._layer_names:
            ml_cfg = cfg.get("mixed_logit", {})
            self._layers["mixed_logit"] = MixedLogitModel(
                attributes=ml_cfg.get("attributes"),
                random_coefficients=ml_cfg.get("random_coefficients"),
            )

        if "latent_class" in self._layer_names:
            seg_cfg = cfg.get("segmentation", {})
            n_classes = seg_cfg.get("latent_classes", "auto")
            self._layers["latent_class"] = LatentClassModel(
                n_classes=n_classes,
                max_classes=seg_cfg.get("max_classes", 8),
            )

        if "rfm" in self._layer_names:
            seg_cfg = cfg.get("segmentation", {})
            self._layers["rfm"] = RFMScorer(
                n_bins=seg_cfg.get("rfm_quintiles", 5),
            )

        if "clv" in self._layer_names:
            self._layers["clv"] = CLVEstimator()

        if "hmm" in self._layer_names:
            temp_cfg = cfg.get("temporal", {})
            self._layers["hmm"] = ConsumerHMM(
                states=temp_cfg.get("hmm_states"),
            )

        if "bayesian_update" in self._layer_names:
            temp_cfg = cfg.get("temporal", {})
            # BayesianUpdater needs store_ids; defer full init to fit().
            self._layers["bayesian_update"] = {
                "prior_weight": temp_cfg.get("bayesian_prior_weight", 0.7) * 10,
            }

        if "residual_boost" in self._layer_names:
            if ResidualBoostModel is None:
                logger.warning(
                    "Skipping residual_boost: xgboost/lightgbm not installed."
                )
            else:
                ml_cfg = cfg.get("ml", {})
                self._layers["residual_boost"] = ResidualBoostModel(
                    model_type=ml_cfg.get("residual_model", "xgboost"),
                    params={
                        "n_estimators": ml_cfg.get("n_estimators", 500),
                        "learning_rate": ml_cfg.get("learning_rate", 0.05),
                        "max_depth": ml_cfg.get("max_depth", 6),
                    },
                )

        if "graph_network" in self._layer_names:
            if GraphConsumerModel is None:
                logger.warning(
                    "Skipping graph_network: required ML libraries not installed."
                )
            else:
                self._layers["graph_network"] = GraphConsumerModel()

        if "ensemble" in self._layer_names:
            ens_cfg = cfg.get("ensemble", {})
            self._layers["ensemble"] = EnsembleAverager()
            self._ensemble_method = ens_cfg.get("method", "bayesian_averaging")

    # ------------------------------------------------------------------ #
    # fit()
    # ------------------------------------------------------------------ #

    def fit(
        self,
        stores: Union[list[Store], pd.DataFrame],
        origins: Union[list[ConsumerOrigin], pd.DataFrame],
        traffic: Optional[pd.DataFrame] = None,
        transactions: Union[list[Transaction], pd.DataFrame, None] = None,
        observed_shares: Optional[pd.DataFrame] = None,
        choice_data: Optional[pd.DataFrame] = None,
    ) -> "GravityModel":
        """Fit all active model layers in dependency order.

        Layers whose required data is not supplied are skipped gracefully
        with a warning.  The dependency order is:

        1. Huff base (stores, origins, optionally observed_shares)
        2. Competing destinations (stores, origins)
        3. GWR (origins, stores, observed_shares)
        4. Mixed Logit (choice_data)
        5. Latent Class (traffic or transactions)
        6. RFM + CLV (transactions)
        7. HMM (temporal visit sequences from traffic or transactions)
        8. Bayesian updater (structural predictions)
        9. Residual boost (observed_shares + structural predictions)
        10. Graph network (visit edges)
        11. Ensemble (all fitted layers)

        Parameters
        ----------
        stores : list[Store] or DataFrame
            Store locations and attributes.
        origins : list[ConsumerOrigin] or DataFrame
            Consumer demand origin points.
        traffic : DataFrame or None
            SafeGraph-style visit-pattern data.
        transactions : list[Transaction], DataFrame, or None
            Purchase transaction records.
        observed_shares : DataFrame or None
            Origin x store actual visit-share matrix (for calibration).
        choice_data : DataFrame or None
            Discrete-choice dataset (for mixed logit).

        Returns
        -------
        GravityModel
            ``self``, fitted.
        """
        # ---- Normalise inputs --------------------------------------------
        self._stores_df = _to_dataframe(stores, stores_to_dataframe, "stores")
        self._origins_df = _to_dataframe(origins, origins_to_dataframe, "origins")
        self._transactions_df = _to_dataframe(
            transactions, transactions_to_dataframe, "transactions"
        )
        self._observed_shares = observed_shares
        self._choice_data = choice_data
        self._traffic = traffic

        # Pre-compute the distance matrix (used by multiple layers).
        self._distance_matrix = build_distance_matrix(
            self._origins_df, self._stores_df
        )

        # ---- (a) Huff base -----------------------------------------------
        if "huff" in self._layers:
            logger.info("Fitting layer: huff")
            huff: HuffModel = self._layers["huff"]
            huff.distance_matrix = self._distance_matrix
            if self._observed_shares is not None:
                huff.fit(self._origins_df, self._stores_df, self._observed_shares)
            probs = huff.predict(self._origins_df, self._stores_df)
            self._predictions["huff"] = probs

        # ---- (b) Competing destinations -----------------------------------
        if "competing_destinations" in self._layers:
            logger.info("Fitting layer: competing_destinations")
            cd: CompetingDestinationsModel = self._layers["competing_destinations"]
            cd.distance_matrix = self._distance_matrix
            if self._observed_shares is not None:
                cd.fit(
                    self._origins_df, self._stores_df, self._observed_shares
                )
            probs = cd.predict(self._origins_df, self._stores_df)
            self._predictions["competing_destinations"] = probs

        # ---- (c) GWR -----------------------------------------------------
        if "gwr" in self._layers:
            gwr: GWRModel = self._layers["gwr"]
            if self._observed_shares is not None:
                logger.info("Fitting layer: gwr")
                gwr.fit(
                    self._origins_df,
                    self._stores_df,
                    self._observed_shares,
                )
                probs = gwr.predict(self._origins_df, self._stores_df)
                self._predictions["gwr"] = probs
            else:
                logger.warning(
                    "Skipping GWR: observed_shares required for calibration."
                )

        # ---- (d) Mixed Logit ---------------------------------------------
        if "mixed_logit" in self._layers:
            ml: MixedLogitModel = self._layers["mixed_logit"]
            if self._choice_data is not None:
                logger.info("Fitting layer: mixed_logit")
                ml_cfg = self.config.get("mixed_logit", {})
                n_draws = ml_cfg.get("n_draws", 500)
                ml.fit(self._choice_data, n_draws=n_draws)
            else:
                logger.warning(
                    "Skipping mixed_logit: choice_data not provided."
                )

        # ---- (e) Latent Class segmentation --------------------------------
        if "latent_class" in self._layers:
            lca: LatentClassModel = self._layers["latent_class"]
            visit_data = self._traffic if self._traffic is not None else None
            if visit_data is None and self._transactions_df is not None:
                # Build visit-count data from transactions.
                visit_data = (
                    self._transactions_df.groupby(["consumer_id", "store_id"])
                    .size()
                    .reset_index(name="visits")
                    .rename(columns={"consumer_id": "origin_id"})
                )
            if visit_data is not None:
                logger.info("Fitting layer: latent_class")
                lca.fit(
                    visit_data,
                    self._origins_df,
                    self._stores_df,
                )
            else:
                logger.warning(
                    "Skipping latent_class: no traffic or transaction data."
                )

        # ---- (f) RFM + CLV -----------------------------------------------
        if "rfm" in self._layers:
            rfm: RFMScorer = self._layers["rfm"]
            if self._transactions_df is not None:
                logger.info("Fitting layer: rfm")
                rfm.fit(self._transactions_df)
            else:
                logger.warning("Skipping rfm: transactions not provided.")

        if "clv" in self._layers:
            clv: CLVEstimator = self._layers["clv"]
            if self._transactions_df is not None:
                logger.info("Fitting layer: clv")
                clv.fit(self._transactions_df)
            else:
                logger.warning("Skipping clv: transactions not provided.")

        # ---- (g) HMM (temporal visit sequences) ---------------------------
        if "hmm" in self._layers:
            hmm: ConsumerHMM = self._layers["hmm"]
            sequences = self._build_hmm_sequences()
            if sequences:
                logger.info("Fitting layer: hmm")
                hmm.fit(sequences)
            else:
                logger.warning(
                    "Skipping hmm: no temporal visit data available."
                )

        # ---- (h) Bayesian updater -----------------------------------------
        if "bayesian_update" in self._layers:
            logger.info("Initialising layer: bayesian_update")
            bu_cfg = self._layers["bayesian_update"]
            store_ids = list(self._stores_df.index)
            # Use the best structural prediction as the default prior.
            prior_probs = self._best_structural_prediction()
            updater = BayesianUpdater(
                store_ids=store_ids,
                prior_weight=bu_cfg.get("prior_weight", 7.0),
                default_prior=(
                    prior_probs.values[0]
                    if prior_probs is not None and len(prior_probs) > 0
                    else None
                ),
            )
            # Register all origins with their structural priors.
            if prior_probs is not None:
                updater.register_consumers_from_matrix(
                    consumer_ids=list(prior_probs.index),
                    prob_matrix=prior_probs.values,
                )
            self._layers["bayesian_update"] = updater

        # ---- (i) Residual boost -------------------------------------------
        if "residual_boost" in self._layers:
            rb: ResidualBoostModel = self._layers["residual_boost"]
            structural = self._best_structural_prediction()
            if (
                self._observed_shares is not None
                and structural is not None
            ):
                logger.info("Fitting layer: residual_boost")
                features_df = self._build_residual_features()
                # Flatten matrices for the boosting model.
                common_idx = structural.index.intersection(
                    self._observed_shares.index
                )
                common_cols = structural.columns.intersection(
                    self._observed_shares.columns
                )
                actual_flat = self._observed_shares.loc[
                    common_idx, common_cols
                ].values.flatten()
                pred_flat = structural.loc[
                    common_idx, common_cols
                ].values.flatten()
                # Tile features for each (origin, store) pair.
                feat_flat = self._tile_features(features_df, common_idx, common_cols)
                rb.fit(actual_flat, pred_flat, feat_flat)
                self._predictions["residual_boost"] = self._unflatten_boost(
                    rb.predict(pred_flat, feat_flat),
                    common_idx,
                    common_cols,
                )
            else:
                logger.warning(
                    "Skipping residual_boost: need observed_shares and "
                    "at least one structural prediction."
                )

        # ---- (j) Graph network --------------------------------------------
        if "graph_network" in self._layers:
            gnn: GraphConsumerModel = self._layers["graph_network"]
            edges_df = self._build_graph_edges()
            if edges_df is not None and len(edges_df) > 0:
                logger.info("Fitting layer: graph_network")
                origin_feats = self._origins_df.select_dtypes(include=[np.number])
                store_feats = self._stores_df.select_dtypes(include=[np.number])
                gnn.fit(edges_df, origin_feats, store_feats)
            else:
                logger.warning(
                    "Skipping graph_network: no visit edge data available."
                )

        # ---- (k) Ensemble averaging ---------------------------------------
        if "ensemble" in self._layers and len(self._predictions) >= 2:
            logger.info("Fitting layer: ensemble")
            ens: EnsembleAverager = self._layers["ensemble"]
            for name, preds in self._predictions.items():
                ens.add_model(name, preds)
            ens.fit_weights(
                actual_shares=(
                    self._observed_shares
                    if self._observed_shares is not None
                    else list(self._predictions.values())[0]
                ),
                method=getattr(self, "_ensemble_method", "bayesian_averaging"),
            )

        self._fitted = True
        logger.info(
            "GravityModel fit complete. Layers fitted: %s",
            list(self._predictions.keys()),
        )
        return self

    # ------------------------------------------------------------------ #
    # predict()
    # ------------------------------------------------------------------ #

    def predict(
        self,
        origins: Optional[Union[list[ConsumerOrigin], pd.DataFrame]] = None,
        stores: Optional[Union[list[Store], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Produce origin x store predictions as a long-format DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``origin_id``, ``store_id``, ``probability``,
            ``segment``, ``consumer_state``, ``clv``,
            ``confidence_lower``, ``confidence_upper``.
        """
        self._check_fitted()

        o_df = (
            _to_dataframe(origins, origins_to_dataframe, "origins")
            if origins is not None
            else self._origins_df
        )
        s_df = (
            _to_dataframe(stores, stores_to_dataframe, "stores")
            if stores is not None
            else self._stores_df
        )

        # Get the probability matrix.
        prob_matrix = self._get_prediction_matrix(o_df, s_df)

        # Confidence intervals (from ensemble if available).
        lower_matrix = None
        upper_matrix = None
        if (
            "ensemble" in self._layers
            and isinstance(self._layers["ensemble"], EnsembleAverager)
            and self._layers["ensemble"].is_fitted
        ):
            lower_matrix, upper_matrix = self._layers[
                "ensemble"
            ].prediction_intervals(confidence=0.95)

        # Build long-format result.
        rows = []
        for oid in prob_matrix.index:
            for sid in prob_matrix.columns:
                row: dict[str, Any] = {
                    "origin_id": oid,
                    "store_id": sid,
                    "probability": float(prob_matrix.loc[oid, sid]),
                    "segment": None,
                    "consumer_state": None,
                    "clv": None,
                    "confidence_lower": None,
                    "confidence_upper": None,
                }

                # Segment (from latent class if fitted).
                if (
                    "latent_class" in self._layers
                    and isinstance(self._layers["latent_class"], LatentClassModel)
                    and self._layers["latent_class"]._fitted
                ):
                    try:
                        seg_df = self._layers["latent_class"].predict_segments()
                        match = seg_df.loc[
                            seg_df["origin_id"] == oid, "segment_id"
                        ]
                        if len(match) > 0:
                            row["segment"] = int(match.iloc[0])
                    except Exception:
                        pass

                # Consumer state (from HMM if fitted).
                if (
                    "hmm" in self._layers
                    and isinstance(self._layers["hmm"], ConsumerHMM)
                    and self._layers["hmm"]._fitted
                ):
                    row["consumer_state"] = self._get_hmm_state(oid)

                # CLV (from CLVEstimator if fitted).
                if (
                    "clv" in self._layers
                    and isinstance(self._layers["clv"], CLVEstimator)
                    and self._layers["clv"]._fitted
                ):
                    row["clv"] = self._get_consumer_clv(oid)

                # Confidence bounds.
                if lower_matrix is not None and upper_matrix is not None:
                    try:
                        row["confidence_lower"] = float(
                            lower_matrix.loc[oid, sid]
                        )
                        row["confidence_upper"] = float(
                            upper_matrix.loc[oid, sid]
                        )
                    except (KeyError, IndexError):
                        pass

                rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # update()
    # ------------------------------------------------------------------ #

    def update(self, event: dict) -> Optional[np.ndarray]:
        """Real-time Bayesian update from a single visit event.

        Parameters
        ----------
        event : dict
            Must contain ``consumer_id``, ``store_id``, ``timestamp``.
            Optionally ``amount``.

        Returns
        -------
        np.ndarray or None
            Updated posterior probabilities for the consumer, or None
            if the BayesianUpdater is not active.
        """
        updater = self._layers.get("bayesian_update")
        if updater is None or not isinstance(updater, BayesianUpdater):
            logger.warning(
                "update() called but bayesian_update layer is not active."
            )
            return None

        consumer_id = event["consumer_id"]
        store_id = event["store_id"]

        return updater.update(consumer_id, store_id)

    # ------------------------------------------------------------------ #
    # trade_area_report()
    # ------------------------------------------------------------------ #

    def trade_area_report(self, store_id: str) -> TradeAreaReport:
        """Generate a trade area report for a single store.

        Parameters
        ----------
        store_id : str
            Target store identifier.

        Returns
        -------
        TradeAreaReport
            Populated report object.
        """
        self._check_fitted()

        prob_matrix = self._get_prediction_matrix()
        contours = self.config.get("reporting", {}).get(
            "trade_area_contours", [0.5, 0.7, 0.9]
        )
        report = TradeAreaReport(contour_levels=contours)
        report.generate(
            store_id=store_id,
            predictions=prob_matrix,
            origins_df=self._origins_df,
            stores_df=self._stores_df,
        )
        return report

    # ------------------------------------------------------------------ #
    # scenario()
    # ------------------------------------------------------------------ #

    def scenario(self, action: str, **kwargs) -> Any:
        """Run a what-if scenario simulation.

        Parameters
        ----------
        action : str
            One of ``"new_store"``, ``"close_store"``, ``"modify_store"``,
            ``"competitor_entry"``.
        **kwargs
            Forwarded to the corresponding ``ScenarioSimulator`` method.

        Returns
        -------
        ScenarioResult
            Result container with baseline and scenario comparisons.
        """
        self._check_fitted()

        # Build a predict_fn that uses the currently fitted model.
        def predict_fn(origins_df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
            """Internal prediction function for the scenario simulator."""
            if "huff" in self._layers:
                huff: HuffModel = self._layers["huff"]
                return huff.predict(origins_df, stores_df)
            # Fallback: rebuild distance matrix and use best available layer.
            if self._predictions:
                return list(self._predictions.values())[-1]
            raise RuntimeError("No fitted layer available for scenario prediction.")

        sim = ScenarioSimulator(
            predict_fn=predict_fn,
            origins_df=self._origins_df,
            stores_df=self._stores_df,
        )

        action_map = {
            "new_store": sim.new_store,
            "close_store": sim.close_store,
            "modify_store": sim.modify_store,
            "competitor_entry": sim.competitor_entry,
        }

        if action not in action_map:
            raise ValueError(
                f"Unknown scenario action '{action}'.  "
                f"Available: {list(action_map.keys())}"
            )

        return action_map[action](**kwargs)

    # ------------------------------------------------------------------ #
    # segment_report()
    # ------------------------------------------------------------------ #

    def segment_report(self) -> ConsumerProfileReport:
        """Generate a consumer segmentation profile report.

        Aggregates segment membership, RFM scores, CLV estimates, and
        HMM states into a single report.

        Returns
        -------
        ConsumerProfileReport
            Populated report object.
        """
        self._check_fitted()

        report = ConsumerProfileReport()

        # Build segments DataFrame.
        segments_df = None
        if (
            "latent_class" in self._layers
            and isinstance(self._layers["latent_class"], LatentClassModel)
            and self._layers["latent_class"]._fitted
        ):
            seg_result = self._layers["latent_class"].predict_segments()
            segments_df = seg_result.rename(
                columns={"segment_id": "segment"}
            ).set_index("origin_id")

        if segments_df is None:
            # Fallback: single-segment DataFrame from origins.
            segments_df = pd.DataFrame(
                {"segment": "default"},
                index=self._origins_df.index,
            )
            segments_df.index.name = "origin_id"

        # RFM scores.
        rfm_scores = None
        if (
            "rfm" in self._layers
            and isinstance(self._layers["rfm"], RFMScorer)
            and self._layers["rfm"]._fitted
        ):
            rfm_scores = self._layers["rfm"].rfm_table_

        # CLV data.
        clv_data = None
        if (
            "clv" in self._layers
            and isinstance(self._layers["clv"], CLVEstimator)
            and self._layers["clv"]._fitted
        ):
            clv_data = self._layers["clv"].customer_data_

        # HMM states.
        hmm_states = None
        if (
            "hmm" in self._layers
            and isinstance(self._layers["hmm"], ConsumerHMM)
            and self._layers["hmm"]._fitted
        ):
            hmm_states = self._hmm_state_summary()

        report.generate(
            segments_df=segments_df,
            rfm_scores=rfm_scores,
            clv_data=clv_data,
            hmm_states=hmm_states,
        )
        return report

    # ------------------------------------------------------------------ #
    # summary()
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Return a high-level model summary.

        Returns
        -------
        str
            Multi-line string with active layers, parameters, and fit
            statistics.
        """
        lines = [
            "GravityModel Summary",
            "=" * 60,
            f"  Status       : {'fitted' if self._fitted else 'not fitted'}",
            f"  Active layers: {self._layer_names}",
        ]

        if self._stores_df is not None:
            lines.append(f"  Stores       : {len(self._stores_df)}")
        if self._origins_df is not None:
            lines.append(f"  Origins      : {len(self._origins_df)}")

        lines.append("")
        lines.append("Layer details:")
        lines.append("-" * 60)

        for name in self._layer_names:
            layer = self._layers.get(name)
            if layer is None:
                lines.append(f"  {name}: not initialised")
                continue

            # Layers with a summary() method.
            if hasattr(layer, "summary") and callable(layer.summary):
                try:
                    layer_summary = layer.summary()
                    if isinstance(layer_summary, str):
                        indented = "\n".join(
                            f"    {line}" for line in layer_summary.split("\n")
                        )
                        lines.append(f"  [{name}]")
                        lines.append(indented)
                    else:
                        lines.append(f"  [{name}]: fitted")
                except Exception:
                    lines.append(f"  [{name}]: summary unavailable")
            elif hasattr(layer, "is_fitted"):
                lines.append(
                    f"  [{name}]: "
                    f"{'fitted' if layer.is_fitted else 'not fitted'}"
                )
            elif isinstance(layer, dict):
                lines.append(f"  [{name}]: config = {layer}")
            else:
                lines.append(f"  [{name}]: present")

        # Ensemble weights.
        if (
            "ensemble" in self._layers
            and isinstance(self._layers["ensemble"], EnsembleAverager)
            and self._layers["ensemble"].is_fitted
        ):
            lines.append("")
            lines.append("Ensemble weights:")
            for model_name, weight in self._layers["ensemble"].weights.items():
                lines.append(f"    {model_name:30s} {weight:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit()`` has been called."""
        return self._fitted

    @property
    def active_layers(self) -> list[str]:
        """Names of active layers."""
        return list(self._layer_names)

    @property
    def layer_objects(self) -> dict[str, Any]:
        """Direct access to instantiated layer objects."""
        return dict(self._layers)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted.  Call fit() first."
            )

    def _best_structural_prediction(self) -> Optional[pd.DataFrame]:
        """Return the best available structural prediction matrix.

        Preference order: competing_destinations > huff > gwr > first available.
        """
        for key in [
            "competing_destinations",
            "huff",
            "gwr",
            "residual_boost",
        ]:
            if key in self._predictions:
                return self._predictions[key]
        if self._predictions:
            return next(iter(self._predictions.values()))
        return None

    def _get_prediction_matrix(
        self,
        origins_df: Optional[pd.DataFrame] = None,
        stores_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return the best prediction matrix (ensemble or single layer).

        If new origins/stores are supplied and differ from training data,
        re-predicts with the Huff layer (or best available).
        """
        # If new data is supplied, re-predict with the base layer.
        if origins_df is not None and stores_df is not None:
            same_origins = (
                self._origins_df is not None
                and origins_df.index.equals(self._origins_df.index)
            )
            same_stores = (
                self._stores_df is not None
                and stores_df.index.equals(self._stores_df.index)
            )
            if not (same_origins and same_stores):
                if "huff" in self._layers:
                    return self._layers["huff"].predict(origins_df, stores_df)

        # Use ensemble if available and fitted.
        if (
            "ensemble" in self._layers
            and isinstance(self._layers["ensemble"], EnsembleAverager)
            and self._layers["ensemble"].is_fitted
        ):
            return self._layers["ensemble"].predict()

        # Otherwise use the best single-layer prediction.
        best = self._best_structural_prediction()
        if best is not None:
            return best

        raise RuntimeError(
            "No predictions available.  Ensure at least one structural "
            "layer is fitted."
        )

    def _build_hmm_sequences(self) -> list[list[str]]:
        """Build observation sequences for the HMM from visit data.

        Converts visit counts per time-period into discretised frequency
        bins: 'high', 'medium', 'low', 'none'.
        """
        sequences: list[list[str]] = []
        self._hmm_consumer_sequences: dict[str, list[str]] = {}

        data = self._traffic if self._traffic is not None else self._transactions_df
        if data is None:
            return sequences

        # Identify the time and consumer columns.
        if "consumer_id" in data.columns:
            cid_col = "consumer_id"
        elif "origin_id" in data.columns:
            cid_col = "origin_id"
        else:
            return sequences

        if "timestamp" in data.columns:
            time_col = "timestamp"
        elif "date_range_start" in data.columns:
            time_col = "date_range_start"
        else:
            return sequences

        df = data.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])

        if df.empty:
            return sequences

        # Create a period column (monthly).
        df["_period"] = df[time_col].dt.to_period("M")

        # Count visits per consumer per period.
        counts = (
            df.groupby([cid_col, "_period"])
            .size()
            .reset_index(name="_visit_count")
        )

        # Discretise into frequency bins.
        def _discretise(count: int) -> str:
            if count >= 4:
                return "high"
            elif count >= 2:
                return "medium"
            elif count >= 1:
                return "low"
            return "none"

        counts["_obs"] = counts["_visit_count"].apply(_discretise)

        # Build per-consumer sequences sorted by period.
        for cid, grp in counts.groupby(cid_col):
            grp_sorted = grp.sort_values("_period")
            seq = grp_sorted["_obs"].tolist()
            if len(seq) >= 2:
                sequences.append(seq)
                self._hmm_consumer_sequences[str(cid)] = seq

        return sequences

    def _build_residual_features(self) -> pd.DataFrame:
        """Build a feature DataFrame for the residual boost model.

        Combines store attributes with origin demographics for each
        (origin, store) pair.
        """
        origin_cols = self._origins_df.select_dtypes(include=[np.number]).columns
        store_cols = self._stores_df.select_dtypes(include=[np.number]).columns

        rows = []
        for oid in self._origins_df.index:
            for sid in self._stores_df.index:
                row = {}
                for c in origin_cols:
                    row[f"origin_{c}"] = self._origins_df.loc[oid, c]
                for c in store_cols:
                    row[f"store_{c}"] = self._stores_df.loc[sid, c]
                row["distance_km"] = self._distance_matrix.loc[oid, sid]
                rows.append(row)

        return pd.DataFrame(rows)

    def _tile_features(
        self,
        features_df: pd.DataFrame,
        origin_idx: pd.Index,
        store_idx: pd.Index,
    ) -> pd.DataFrame:
        """Tile feature rows to match a flattened origin x store vector.

        The feature DataFrame from ``_build_residual_features`` has one
        row per (origin, store) pair in the full matrix.  This filters
        and reorders to match the common index/columns.
        """
        n_origins = len(origin_idx)
        n_stores = len(store_idx)
        total = n_origins * n_stores

        # If the pre-built features match dimensions, return directly.
        if len(features_df) == total:
            return features_df.reset_index(drop=True)

        # Otherwise rebuild for the specific subset.
        origin_cols = self._origins_df.select_dtypes(include=[np.number]).columns
        store_cols = self._stores_df.select_dtypes(include=[np.number]).columns

        rows = []
        for oid in origin_idx:
            for sid in store_idx:
                row = {}
                for c in origin_cols:
                    row[f"origin_{c}"] = self._origins_df.loc[oid, c]
                for c in store_cols:
                    row[f"store_{c}"] = self._stores_df.loc[sid, c]
                row["distance_km"] = self._distance_matrix.loc[oid, sid]
                rows.append(row)

        return pd.DataFrame(rows)

    def _unflatten_boost(
        self,
        flat_predictions: np.ndarray,
        origin_idx: pd.Index,
        store_idx: pd.Index,
    ) -> pd.DataFrame:
        """Reshape flat residual-boost predictions to an origin x store matrix."""
        matrix = flat_predictions.reshape(len(origin_idx), len(store_idx))
        return pd.DataFrame(matrix, index=origin_idx, columns=store_idx)

    def _build_graph_edges(self) -> Optional[pd.DataFrame]:
        """Build an edge DataFrame for the graph network from visit data."""
        data = self._traffic if self._traffic is not None else self._transactions_df
        if data is None:
            return None

        # Identify columns.
        if "consumer_id" in data.columns:
            cid_col = "consumer_id"
        elif "origin_id" in data.columns:
            cid_col = "origin_id"
        else:
            return None

        if "store_id" not in data.columns:
            return None

        df = data.copy()

        # Aggregate to get visits, recency, monetary per edge.
        agg = {"store_id": "count"}
        rename = {"store_id": "visits"}

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            ref_date = df["timestamp"].max()
            df["_recency_days"] = (ref_date - df["timestamp"]).dt.days
            agg["_recency_days"] = "min"
            rename["_recency_days"] = "recency"

        if "amount" in df.columns:
            agg["amount"] = "sum"
            rename["amount"] = "monetary"

        edges = df.groupby([cid_col, "store_id"]).agg(agg).reset_index()
        edges = edges.rename(columns={cid_col: "origin_id", **rename})

        # Fill missing columns with defaults.
        if "visits" not in edges.columns:
            edges["visits"] = 1
        if "recency" not in edges.columns:
            edges["recency"] = 0
        if "monetary" not in edges.columns:
            edges["monetary"] = 0.0

        return edges

    def _get_hmm_state(self, origin_id: str) -> Optional[str]:
        """Get the most recent HMM-decoded state for a consumer/origin.

        Uses Viterbi decoding on the consumer's stored observation
        sequence and returns the final (most recent) state label.
        """
        hmm: ConsumerHMM = self._layers.get("hmm")
        if hmm is None or not hmm._fitted:
            return None

        # Look up stored sequence for this consumer/origin
        sequences = getattr(self, "_hmm_consumer_sequences", {})
        seq = sequences.get(origin_id)
        if not seq or len(seq) < 1:
            return None

        try:
            state_path = hmm.decode(seq)
            return state_path[-1] if state_path else None
        except Exception:
            return None

    def _get_consumer_clv(self, origin_id: str) -> Optional[float]:
        """Get CLV estimate for a consumer/origin."""
        clv: CLVEstimator = self._layers.get("clv")
        if clv is None or not clv._fitted or clv.customer_data_ is None:
            return None
        if "predicted_clv" in clv.customer_data_.columns:
            if origin_id in clv.customer_data_.index:
                return float(clv.customer_data_.loc[origin_id, "predicted_clv"])
        return None

    def _hmm_state_summary(self) -> Optional[pd.DataFrame]:
        """Build a summary DataFrame of HMM states via Viterbi decoding."""
        hmm: ConsumerHMM = self._layers.get("hmm")
        if hmm is None or not hmm._fitted:
            return None

        sequences = getattr(self, "_hmm_consumer_sequences", {})
        if not sequences:
            return None

        rows = []
        for cid, seq in sequences.items():
            try:
                state_path = hmm.decode(seq)
                if state_path:
                    rows.append({
                        "consumer_id": cid,
                        "current_state": state_path[-1],
                        "n_periods": len(seq),
                    })
            except Exception:
                continue

        if not rows:
            return None

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"GravityModel(layers={self._layer_names}, "
            f"fitted={self._fitted})"
        )
