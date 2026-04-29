"""
Graph Consumer Model
=====================
Graph-based model for consumer-store interaction patterns, treating the
relationship network as a bipartite graph.

This module provides two implementations behind a single interface:

1. **GNN mode** (requires ``torch`` and ``torch_geometric``):
   A Graph Neural Network that learns node embeddings through message
   passing on the bipartite consumer-store graph.  Origin (consumer)
   nodes carry demographic features; store nodes carry attribute
   features; edges carry visit frequency, recency, and monetary value.
   Link prediction is performed by computing a score from the learned
   embeddings of an origin-store pair.

2. **Fallback mode** (pure sklearn, no PyTorch dependency):
   Extracts hand-crafted graph features -- node degree, common
   neighbours, Jaccard coefficient, Adamic-Adar index -- and trains a
   gradient-boosted classifier to predict links.  This runs on any
   Python installation with numpy, pandas, and scikit-learn.

The class automatically selects the GNN path when torch_geometric is
available, and falls back to the sklearn path otherwise.  The user can
also force a mode via the ``backend`` parameter.

References
----------
Kipf, T.N. & Welling, M. (2017). "Semi-Supervised Classification with
    Graph Convolutional Networks." *ICLR 2017*.

Hamilton, W., Ying, Z., & Leskovec, J. (2017). "Inductive Representation
    Learning on Large Graphs." *NeurIPS 2017*.

Liben-Nowell, D. & Kleinberg, J. (2007). "The Link-Prediction Problem
    for Social Networks." *JASIST*, 58(7), 1019-1031.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports -- fail gracefully
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import SAGEConv, to_hetero
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HIDDEN_DIM: int = 64
_DEFAULT_EMBEDDING_DIM: int = 32
_DEFAULT_N_LAYERS: int = 2
_DEFAULT_EPOCHS: int = 100
_DEFAULT_LR: float = 0.01

_DEFAULT_GBC_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 10,
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# GNN components (only instantiated when torch_geometric is available)
# ---------------------------------------------------------------------------

if _HAS_TORCH and _HAS_PYG:

    class _BipartiteGNNEncoder(nn.Module):
        """Bipartite graph encoder using GraphSAGE-style message passing.

        Learns separate embeddings for origin and store nodes through
        heterogeneous message passing on the bipartite graph.

        Parameters
        ----------
        origin_in_dim : int
            Dimensionality of origin (consumer) input features.
        store_in_dim : int
            Dimensionality of store input features.
        hidden_dim : int
            Hidden layer width.
        embedding_dim : int
            Output embedding dimensionality.
        n_layers : int
            Number of message-passing layers.
        """

        def __init__(
            self,
            origin_in_dim: int,
            store_in_dim: int,
            hidden_dim: int = _DEFAULT_HIDDEN_DIM,
            embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
            n_layers: int = _DEFAULT_N_LAYERS,
        ) -> None:
            super().__init__()
            self.origin_in_dim = origin_in_dim
            self.store_in_dim = store_in_dim
            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim
            self.n_layers = n_layers

            # Project heterogeneous input features to a common hidden dim.
            self.origin_proj = nn.Linear(origin_in_dim, hidden_dim)
            self.store_proj = nn.Linear(store_in_dim, hidden_dim)

            # SAGEConv layers (will be converted to heterogeneous).
            self.convs = nn.ModuleList()
            for i in range(n_layers):
                in_dim = hidden_dim if i == 0 else hidden_dim
                out_dim = embedding_dim if i == n_layers - 1 else hidden_dim
                self.convs.append(SAGEConv(in_dim, out_dim))

        def forward(self, x_dict, edge_index_dict):
            """Forward pass through the GNN encoder.

            Parameters
            ----------
            x_dict : dict[str, Tensor]
                Node feature tensors keyed by node type.
            edge_index_dict : dict[tuple, Tensor]
                Edge index tensors keyed by edge type tuples.

            Returns
            -------
            dict[str, Tensor]
                Node embeddings keyed by node type.
            """
            # Project to common dimension.
            x_dict["origin"] = F.relu(self.origin_proj(x_dict["origin"]))
            x_dict["store"] = F.relu(self.store_proj(x_dict["store"]))

            # Message passing.
            for i, conv in enumerate(self.convs):
                x_dict_new = {}
                for node_type in x_dict:
                    x_dict_new[node_type] = x_dict[node_type]

                # Apply conv to each edge type.
                for edge_type, edge_index in edge_index_dict.items():
                    src_type, _, dst_type = edge_type
                    src_x = x_dict[src_type]
                    dst_x = x_dict[dst_type]

                    # SAGEConv on the bipartite subgraph.
                    out = conv((src_x, dst_x), edge_index)
                    if i < self.n_layers - 1:
                        out = F.relu(out)
                        out = F.dropout(out, p=0.2, training=self.training)
                    x_dict_new[dst_type] = out

                x_dict = x_dict_new

            return x_dict

    class _LinkPredictor(nn.Module):
        """MLP-based link predictor from node embeddings.

        Takes concatenated origin and store embeddings and predicts
        the probability of a visit link.

        Parameters
        ----------
        embedding_dim : int
            Dimensionality of each node embedding.
        """

        def __init__(self, embedding_dim: int) -> None:
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embedding_dim, 1),
            )

        def forward(self, z_origin: torch.Tensor, z_store: torch.Tensor) -> torch.Tensor:
            """Predict link probability.

            Parameters
            ----------
            z_origin : Tensor
                Origin embeddings, shape ``(n_pairs, embedding_dim)``.
            z_store : Tensor
                Store embeddings, shape ``(n_pairs, embedding_dim)``.

            Returns
            -------
            Tensor
                Link probabilities, shape ``(n_pairs,)``.
            """
            h = torch.cat([z_origin, z_store], dim=-1)
            return self.mlp(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Graph feature extraction (fallback mode)
# ---------------------------------------------------------------------------

def _extract_graph_features(
    edges_df: pd.DataFrame,
    origin_ids: np.ndarray,
    store_ids: np.ndarray,
    origin_features: Optional[pd.DataFrame] = None,
    store_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract hand-crafted graph features for link prediction.

    Computes topological features from the bipartite graph structure:
    node degrees, common neighbours, Jaccard coefficient, and
    Adamic-Adar index.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Observed edges with columns ``origin_id``, ``store_id``,
        ``visits``, ``recency``, ``monetary``.
    origin_ids : np.ndarray
        Origin IDs for the pairs to featurise.
    store_ids : np.ndarray
        Store IDs for the pairs to featurise.
    origin_features : pd.DataFrame or None
        Origin-level features to include (indexed by origin_id).
    store_features : pd.DataFrame or None
        Store-level features to include (indexed by store_id).

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per (origin_id, store_id) pair.
    """
    # Build adjacency structures.
    origin_neighbours: dict[str, set] = {}
    store_neighbours: dict[str, set] = {}

    for _, row in edges_df.iterrows():
        oid = row["origin_id"]
        sid = row["store_id"]
        origin_neighbours.setdefault(oid, set()).add(sid)
        store_neighbours.setdefault(sid, set()).add(oid)

    # Precompute degrees.
    origin_degree = {k: len(v) for k, v in origin_neighbours.items()}
    store_degree = {k: len(v) for k, v in store_neighbours.items()}

    # Edge-level aggregates (visit stats per origin, per store).
    origin_visit_stats = edges_df.groupby("origin_id").agg(
        total_visits=("visits", "sum"),
        avg_recency=("recency", "mean"),
        avg_monetary=("monetary", "mean"),
        n_stores_visited=("store_id", "nunique"),
    )

    store_visit_stats = edges_df.groupby("store_id").agg(
        total_visits_received=("visits", "sum"),
        avg_recency_received=("recency", "mean"),
        avg_monetary_received=("monetary", "mean"),
        n_origins_served=("origin_id", "nunique"),
    )

    features = []
    for oid, sid in zip(origin_ids, store_ids):
        row_feats = {}

        # Degree features.
        row_feats["origin_degree"] = origin_degree.get(oid, 0)
        row_feats["store_degree"] = store_degree.get(sid, 0)

        # Common neighbours (origins that visit the same stores as this
        # origin, and then also visit this store -- projected graph).
        o_stores = origin_neighbours.get(oid, set())
        s_origins = store_neighbours.get(sid, set())

        # Neighbours in the projected origin graph via shared stores.
        common_stores = set()
        for other_origin in s_origins:
            other_stores = origin_neighbours.get(other_origin, set())
            common_stores |= (o_stores & other_stores)
        row_feats["common_neighbours"] = len(common_stores)

        # Jaccard coefficient (on projected graph).
        union_stores = set()
        for other_origin in s_origins:
            union_stores |= origin_neighbours.get(other_origin, set())
        all_stores = o_stores | union_stores
        row_feats["jaccard"] = (
            len(common_stores) / len(all_stores)
            if len(all_stores) > 0
            else 0.0
        )

        # Adamic-Adar index (sum of 1/log(degree) for common neighbours).
        adamic_adar = 0.0
        for cs in common_stores:
            deg = store_degree.get(cs, 1)
            if deg > 1:
                adamic_adar += 1.0 / np.log(deg)
        row_feats["adamic_adar"] = adamic_adar

        # Preferential attachment.
        row_feats["pref_attachment"] = (
            row_feats["origin_degree"] * row_feats["store_degree"]
        )

        # Visit stats for this origin.
        if oid in origin_visit_stats.index:
            ostats = origin_visit_stats.loc[oid]
            row_feats["origin_total_visits"] = ostats["total_visits"]
            row_feats["origin_avg_recency"] = ostats["avg_recency"]
            row_feats["origin_avg_monetary"] = ostats["avg_monetary"]
            row_feats["origin_n_stores"] = ostats["n_stores_visited"]
        else:
            row_feats["origin_total_visits"] = 0
            row_feats["origin_avg_recency"] = 0.0
            row_feats["origin_avg_monetary"] = 0.0
            row_feats["origin_n_stores"] = 0

        # Visit stats for this store.
        if sid in store_visit_stats.index:
            sstats = store_visit_stats.loc[sid]
            row_feats["store_total_visits"] = sstats["total_visits_received"]
            row_feats["store_avg_recency"] = sstats["avg_recency_received"]
            row_feats["store_avg_monetary"] = sstats["avg_monetary_received"]
            row_feats["store_n_origins"] = sstats["n_origins_served"]
        else:
            row_feats["store_total_visits"] = 0
            row_feats["store_avg_recency"] = 0.0
            row_feats["store_avg_monetary"] = 0.0
            row_feats["store_n_origins"] = 0

        features.append(row_feats)

    feature_df = pd.DataFrame(features)

    # Append origin- and store-level attribute features.
    if origin_features is not None:
        origin_feat_aligned = origin_features.reindex(origin_ids).reset_index(
            drop=True
        )
        origin_feat_aligned.columns = [
            f"origin_{c}" for c in origin_feat_aligned.columns
        ]
        feature_df = pd.concat([feature_df, origin_feat_aligned], axis=1)

    if store_features is not None:
        store_feat_aligned = store_features.reindex(store_ids).reset_index(
            drop=True
        )
        store_feat_aligned.columns = [
            f"store_{c}" for c in store_feat_aligned.columns
        ]
        feature_df = pd.concat([feature_df, store_feat_aligned], axis=1)

    # Fill any NaN from missing reindex matches.
    feature_df = feature_df.fillna(0.0)

    return feature_df


# ---------------------------------------------------------------------------
# GraphConsumerModel
# ---------------------------------------------------------------------------

class GraphConsumerModel:
    """Graph-based model for consumer-store interaction patterns.

    Models the consumer-store relationship network as a bipartite graph
    and learns to predict which consumers will visit which stores.

    When ``torch_geometric`` is available, uses a Graph Neural Network
    with SAGEConv message passing.  Otherwise, falls back to a
    gradient-boosted classifier trained on hand-crafted graph features
    (degree, common neighbours, Jaccard coefficient, Adamic-Adar index).

    Parameters
    ----------
    backend : {"auto", "gnn", "sklearn"}, default "auto"
        Which implementation to use.  ``"auto"`` selects GNN if
        torch_geometric is installed, otherwise sklearn.  ``"gnn"``
        forces the GNN path (raises ImportError if unavailable).
        ``"sklearn"`` forces the fallback path.
    hidden_dim : int, default 64
        Hidden layer width for the GNN encoder.
    embedding_dim : int, default 32
        Output embedding dimensionality for the GNN.
    n_layers : int, default 2
        Number of GNN message-passing layers.
    epochs : int, default 100
        Training epochs for the GNN.
    lr : float, default 0.01
        Learning rate for the GNN optimiser.
    negative_ratio : float, default 1.0
        Ratio of negative (non-existing) edges to positive edges
        during training.  1.0 means equal numbers.
    gbc_params : dict or None
        Parameters for the sklearn GradientBoostingClassifier.
        If ``None``, sensible defaults are used.

    Attributes
    ----------
    backend_ : str
        The actual backend being used (``"gnn"`` or ``"sklearn"``).
    model_ : object or None
        The fitted model.  A torch Module for GNN, or a
        GradientBoostingClassifier for sklearn.
    origin_ids_ : np.ndarray or None
        Unique origin IDs from the training data.
    store_ids_ : np.ndarray or None
        Unique store IDs from the training data.

    Examples
    --------
    >>> model = GraphConsumerModel(backend="auto")
    >>> model.fit(edges_df, origin_features, store_features)
    >>> predictions = model.predict_links(origin_features, store_features)
    >>> recs = model.recommend_stores(origin_id="O42", top_k=5)
    """

    def __init__(
        self,
        backend: str = "auto",
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
        n_layers: int = _DEFAULT_N_LAYERS,
        epochs: int = _DEFAULT_EPOCHS,
        lr: float = _DEFAULT_LR,
        negative_ratio: float = 1.0,
        gbc_params: Optional[dict] = None,
    ) -> None:
        if backend not in ("auto", "gnn", "sklearn"):
            raise ValueError(
                f"backend must be 'auto', 'gnn', or 'sklearn', got '{backend}'"
            )
        if backend == "gnn" and not (_HAS_TORCH and _HAS_PYG):
            raise ImportError(
                "GNN backend requires torch and torch_geometric.  "
                "Install with: pip install torch torch_geometric"
            )

        if backend == "auto":
            self.backend_ = "gnn" if (_HAS_TORCH and _HAS_PYG) else "sklearn"
        else:
            self.backend_ = backend

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.negative_ratio = negative_ratio
        self.gbc_params = dict(gbc_params) if gbc_params is not None else None

        # Populated after fit().
        self.model_ = None
        self.origin_ids_: Optional[np.ndarray] = None
        self.store_ids_: Optional[np.ndarray] = None
        self._edges_df: Optional[pd.DataFrame] = None
        self._origin_features: Optional[pd.DataFrame] = None
        self._store_features: Optional[pd.DataFrame] = None
        self._scaler: Optional[StandardScaler] = None
        self._fitted: bool = False

        # GNN-specific state.
        self._origin_embeddings: Optional[np.ndarray] = None
        self._store_embeddings: Optional[np.ndarray] = None
        self._link_predictor = None
        self._origin_id_to_idx: Optional[dict] = None
        self._store_id_to_idx: Optional[dict] = None
        self._training_losses: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        edges_df: pd.DataFrame,
        origin_features: pd.DataFrame,
        store_features: pd.DataFrame,
        *,
        verbose: bool = False,
    ) -> "GraphConsumerModel":
        """Train the graph consumer model.

        Parameters
        ----------
        edges_df : pd.DataFrame
            Edge table with columns: ``origin_id``, ``store_id``,
            ``visits`` (visit frequency), ``recency`` (days since last
            visit), ``monetary`` (total/average spend).
        origin_features : pd.DataFrame
            Origin (consumer) demographics, indexed by ``origin_id``.
            All columns should be numeric.
        store_features : pd.DataFrame
            Store attributes, indexed by ``store_id``.
            All columns should be numeric.
        verbose : bool, default False
            If True, print training progress.

        Returns
        -------
        GraphConsumerModel
            ``self``, now fitted.

        Raises
        ------
        ValueError
            If ``edges_df`` is missing required columns.
        """
        required_cols = {"origin_id", "store_id", "visits", "recency", "monetary"}
        missing = required_cols - set(edges_df.columns)
        if missing:
            raise ValueError(
                f"edges_df is missing required columns: {sorted(missing)}"
            )

        self._edges_df = edges_df.copy()
        self._origin_features = origin_features.copy()
        self._store_features = store_features.copy()
        self.origin_ids_ = np.array(sorted(origin_features.index.unique()))
        self.store_ids_ = np.array(sorted(store_features.index.unique()))

        if self.backend_ == "gnn":
            self._fit_gnn(edges_df, origin_features, store_features, verbose)
        else:
            self._fit_sklearn(edges_df, origin_features, store_features, verbose)

        self._fitted = True

        logger.info(
            "GraphConsumerModel fitted (backend=%s): %d origins, %d stores, "
            "%d edges.",
            self.backend_,
            len(self.origin_ids_),
            len(self.store_ids_),
            len(edges_df),
        )

        return self

    def predict_links(
        self,
        origin_features: pd.DataFrame,
        store_features: pd.DataFrame,
        *,
        pairs: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Predict visit probability for origin-store pairs.

        Parameters
        ----------
        origin_features : pd.DataFrame
            Origin features, indexed by ``origin_id``.
        store_features : pd.DataFrame
            Store features, indexed by ``store_id``.
        pairs : pd.DataFrame or None
            Specific pairs to predict, with columns ``origin_id`` and
            ``store_id``.  If ``None``, predicts all possible pairs
            (full cross-product).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``origin_id``, ``store_id``, and
            ``link_probability``, sorted by probability descending.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_fitted()

        if pairs is None:
            origin_ids = np.array(origin_features.index)
            store_ids = np.array(store_features.index)
            # Full cross-product.
            pairs_list = [
                (oid, sid) for oid in origin_ids for sid in store_ids
            ]
            pairs = pd.DataFrame(
                pairs_list, columns=["origin_id", "store_id"]
            )

        if self.backend_ == "gnn":
            probs = self._predict_gnn(pairs)
        else:
            probs = self._predict_sklearn(
                pairs, origin_features, store_features
            )

        result = pairs.copy()
        result["link_probability"] = probs
        return result.sort_values(
            "link_probability", ascending=False
        ).reset_index(drop=True)

    def recommend_stores(
        self,
        origin_id: Union[str, int],
        top_k: int = 5,
        *,
        exclude_visited: bool = True,
    ) -> pd.DataFrame:
        """Recommend top stores for a given origin.

        Parameters
        ----------
        origin_id : str or int
            The origin to generate recommendations for.
        top_k : int, default 5
            Number of top store recommendations to return.
        exclude_visited : bool, default True
            If True, exclude stores that the origin has already visited
            in the training data.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``store_id``, ``link_probability``,
            ``rank``, sorted by probability descending.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        KeyError
            If ``origin_id`` is not in the training data.
        """
        self._check_fitted()

        if origin_id not in self._origin_features.index:
            raise KeyError(
                f"origin_id '{origin_id}' not found in the training data."
            )

        # Get all store candidates.
        candidate_stores = self.store_ids_.copy()

        if exclude_visited and self._edges_df is not None:
            visited = set(
                self._edges_df.loc[
                    self._edges_df["origin_id"] == origin_id, "store_id"
                ]
            )
            candidate_stores = np.array(
                [s for s in candidate_stores if s not in visited]
            )

        if len(candidate_stores) == 0:
            logger.warning(
                "No candidate stores for origin '%s' (all already visited).",
                origin_id,
            )
            return pd.DataFrame(
                columns=["store_id", "link_probability", "rank"]
            )

        pairs = pd.DataFrame({
            "origin_id": [origin_id] * len(candidate_stores),
            "store_id": candidate_stores,
        })

        predictions = self.predict_links(
            self._origin_features,
            self._store_features,
            pairs=pairs,
        )

        top = predictions.head(top_k).copy()
        top["rank"] = range(1, len(top) + 1)
        return top[["store_id", "link_probability", "rank"]].reset_index(
            drop=True
        )

    # ------------------------------------------------------------------
    # sklearn fallback implementation
    # ------------------------------------------------------------------

    def _fit_sklearn(
        self,
        edges_df: pd.DataFrame,
        origin_features: pd.DataFrame,
        store_features: pd.DataFrame,
        verbose: bool = False,
    ) -> None:
        """Fit the gradient-boosted classifier on graph features.

        Generates positive examples from observed edges and negative
        examples from unobserved origin-store pairs (sampled at the
        configured ``negative_ratio``).

        Parameters
        ----------
        edges_df : pd.DataFrame
            Observed edges.
        origin_features : pd.DataFrame
            Origin demographics.
        store_features : pd.DataFrame
            Store attributes.
        verbose : bool
            Print progress if True.
        """
        # Positive examples: observed edges.
        pos_origins = edges_df["origin_id"].values
        pos_stores = edges_df["store_id"].values
        pos_labels = np.ones(len(edges_df))

        # Negative examples: unobserved edges (sampled).
        existing_edges = set(zip(pos_origins, pos_stores))
        all_origins = self.origin_ids_
        all_stores = self.store_ids_

        n_neg = int(len(edges_df) * self.negative_ratio)
        rng = np.random.RandomState(42)
        neg_origins = []
        neg_stores = []
        attempts = 0
        max_attempts = n_neg * 20

        while len(neg_origins) < n_neg and attempts < max_attempts:
            o = all_origins[rng.randint(len(all_origins))]
            s = all_stores[rng.randint(len(all_stores))]
            if (o, s) not in existing_edges:
                neg_origins.append(o)
                neg_stores.append(s)
                existing_edges.add((o, s))
            attempts += 1

        neg_labels = np.zeros(len(neg_origins))

        # Combine positive and negative.
        all_origin_ids = np.concatenate([pos_origins, np.array(neg_origins)])
        all_store_ids = np.concatenate([pos_stores, np.array(neg_stores)])
        all_labels = np.concatenate([pos_labels, neg_labels])

        # Extract features.
        feature_matrix = _extract_graph_features(
            edges_df,
            all_origin_ids,
            all_store_ids,
            origin_features,
            store_features,
        )

        # Scale features.
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(
            feature_matrix.values.astype(np.float64)
        )
        y = all_labels.astype(np.int32)

        # Train classifier.
        params = self.gbc_params if self.gbc_params is not None else dict(_DEFAULT_GBC_PARAMS)
        self.model_ = GradientBoostingClassifier(**params)
        self.model_.fit(X, y)

        # Store feature names for later use.
        self._graph_feature_names = list(feature_matrix.columns)

        if verbose:
            train_acc = self.model_.score(X, y)
            logger.info(
                "sklearn fallback fitted: %d pos, %d neg, train acc=%.4f, "
                "%d features.",
                int(pos_labels.sum()),
                int(neg_labels.sum()),
                train_acc,
                X.shape[1],
            )

    def _predict_sklearn(
        self,
        pairs: pd.DataFrame,
        origin_features: pd.DataFrame,
        store_features: pd.DataFrame,
    ) -> np.ndarray:
        """Predict link probabilities using the sklearn fallback.

        Parameters
        ----------
        pairs : pd.DataFrame
            Pairs to predict (``origin_id``, ``store_id``).
        origin_features : pd.DataFrame
            Origin features.
        store_features : pd.DataFrame
            Store features.

        Returns
        -------
        np.ndarray
            Predicted probabilities, shape ``(n_pairs,)``.
        """
        feature_matrix = _extract_graph_features(
            self._edges_df,
            pairs["origin_id"].values,
            pairs["store_id"].values,
            origin_features,
            store_features,
        )

        # Align columns to training feature set.
        for col in self._graph_feature_names:
            if col not in feature_matrix.columns:
                feature_matrix[col] = 0.0
        feature_matrix = feature_matrix[self._graph_feature_names]

        X = self._scaler.transform(
            feature_matrix.values.astype(np.float64)
        )
        return self.model_.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # GNN implementation
    # ------------------------------------------------------------------

    def _fit_gnn(
        self,
        edges_df: pd.DataFrame,
        origin_features: pd.DataFrame,
        store_features: pd.DataFrame,
        verbose: bool = False,
    ) -> None:
        """Fit the GNN on the bipartite graph.

        Builds a heterogeneous graph, trains the GNN encoder and link
        predictor end-to-end with binary cross-entropy loss.

        Parameters
        ----------
        edges_df : pd.DataFrame
            Observed edges.
        origin_features : pd.DataFrame
            Origin demographics.
        store_features : pd.DataFrame
            Store attributes.
        verbose : bool
            Print per-epoch loss if True.
        """
        # Build ID maps.
        self._origin_id_to_idx = {
            oid: i for i, oid in enumerate(self.origin_ids_)
        }
        self._store_id_to_idx = {
            sid: i for i, sid in enumerate(self.store_ids_)
        }

        # Prepare node features.
        origin_x = torch.tensor(
            origin_features.reindex(self.origin_ids_).fillna(0).values,
            dtype=torch.float32,
        )
        store_x = torch.tensor(
            store_features.reindex(self.store_ids_).fillna(0).values,
            dtype=torch.float32,
        )

        # Build edge index (positive edges).
        src_idx = [
            self._origin_id_to_idx[oid]
            for oid in edges_df["origin_id"]
            if oid in self._origin_id_to_idx
        ]
        dst_idx = [
            self._store_id_to_idx[sid]
            for sid in edges_df["store_id"]
            if sid in self._store_id_to_idx
        ]
        n_valid = min(len(src_idx), len(dst_idx))
        src_idx = src_idx[:n_valid]
        dst_idx = dst_idx[:n_valid]

        edge_index = torch.tensor(
            [src_idx, dst_idx], dtype=torch.long
        )

        # Reverse edges for message passing in both directions.
        rev_edge_index = torch.tensor(
            [dst_idx, src_idx], dtype=torch.long
        )

        # Build model.
        encoder = _BipartiteGNNEncoder(
            origin_in_dim=origin_x.shape[1],
            store_in_dim=store_x.shape[1],
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers,
        )
        link_pred = _LinkPredictor(self.embedding_dim)

        optimiser = torch.optim.Adam(
            list(encoder.parameters()) + list(link_pred.parameters()),
            lr=self.lr,
        )

        # Sample negative edges.
        existing_edges = set(zip(src_idx, dst_idx))
        n_neg = int(len(src_idx) * self.negative_ratio)

        rng = np.random.RandomState(42)

        # Training loop.
        self._training_losses = []
        encoder.train()
        link_pred.train()

        for epoch in range(self.epochs):
            optimiser.zero_grad()

            # Forward pass through encoder.
            x_dict = {"origin": origin_x, "store": store_x}
            edge_index_dict = {
                ("origin", "visits", "store"): edge_index,
                ("store", "visited_by", "origin"): rev_edge_index,
            }
            z_dict = encoder(x_dict, edge_index_dict)

            # Positive edge predictions.
            z_src_pos = z_dict["origin"][edge_index[0]]
            z_dst_pos = z_dict["store"][edge_index[1]]
            pos_scores = link_pred(z_src_pos, z_dst_pos)

            # Sample negative edges for this epoch.
            neg_src = []
            neg_dst = []
            attempts = 0
            while len(neg_src) < n_neg and attempts < n_neg * 10:
                s = rng.randint(len(self.origin_ids_))
                d = rng.randint(len(self.store_ids_))
                if (s, d) not in existing_edges:
                    neg_src.append(s)
                    neg_dst.append(d)
                attempts += 1

            neg_edge = torch.tensor([neg_src, neg_dst], dtype=torch.long)
            z_src_neg = z_dict["origin"][neg_edge[0]]
            z_dst_neg = z_dict["store"][neg_edge[1]]
            neg_scores = link_pred(z_src_neg, z_dst_neg)

            # Binary cross-entropy loss.
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            loss = pos_loss + neg_loss

            loss.backward()
            optimiser.step()

            self._training_losses.append(float(loss.item()))

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    "GNN epoch %d/%d: loss=%.4f (pos=%.4f, neg=%.4f)",
                    epoch + 1,
                    self.epochs,
                    loss.item(),
                    pos_loss.item(),
                    neg_loss.item(),
                )

        # Store trained components and compute final embeddings.
        encoder.eval()
        link_pred.eval()

        with torch.no_grad():
            x_dict = {"origin": origin_x, "store": store_x}
            edge_index_dict = {
                ("origin", "visits", "store"): edge_index,
                ("store", "visited_by", "origin"): rev_edge_index,
            }
            z_dict = encoder(x_dict, edge_index_dict)
            self._origin_embeddings = z_dict["origin"].numpy()
            self._store_embeddings = z_dict["store"].numpy()

        self._link_predictor = link_pred
        self.model_ = encoder

    def _predict_gnn(self, pairs: pd.DataFrame) -> np.ndarray:
        """Predict link probabilities using the GNN.

        Parameters
        ----------
        pairs : pd.DataFrame
            Pairs to predict (``origin_id``, ``store_id``).

        Returns
        -------
        np.ndarray
            Predicted probabilities, shape ``(n_pairs,)``.
        """
        origin_idx = [
            self._origin_id_to_idx.get(oid, -1)
            for oid in pairs["origin_id"]
        ]
        store_idx = [
            self._store_id_to_idx.get(sid, -1)
            for sid in pairs["store_id"]
        ]

        probs = np.zeros(len(pairs))

        for i, (oi, si) in enumerate(zip(origin_idx, store_idx)):
            if oi < 0 or si < 0:
                probs[i] = 0.0
                continue

            z_o = torch.tensor(
                self._origin_embeddings[oi:oi+1], dtype=torch.float32
            )
            z_s = torch.tensor(
                self._store_embeddings[si:si+1], dtype=torch.float32
            )

            with torch.no_grad():
                score = self._link_predictor(z_o, z_s)
                probs[i] = float(torch.sigmoid(score).item())

        return probs

    # ------------------------------------------------------------------
    # Graph analytics
    # ------------------------------------------------------------------

    def node_embeddings(self) -> Optional[dict]:
        """Return learned node embeddings (GNN mode only).

        Returns
        -------
        dict or None
            Dictionary with keys ``"origin_embeddings"`` (DataFrame
            indexed by origin_id) and ``"store_embeddings"`` (DataFrame
            indexed by store_id).  Returns ``None`` if using the sklearn
            fallback.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_fitted()

        if self.backend_ != "gnn" or self._origin_embeddings is None:
            logger.info(
                "Node embeddings are only available in GNN mode."
            )
            return None

        origin_emb_df = pd.DataFrame(
            self._origin_embeddings,
            index=self.origin_ids_,
            columns=[f"emb_{i}" for i in range(self._origin_embeddings.shape[1])],
        )
        store_emb_df = pd.DataFrame(
            self._store_embeddings,
            index=self.store_ids_,
            columns=[f"emb_{i}" for i in range(self._store_embeddings.shape[1])],
        )

        return {
            "origin_embeddings": origin_emb_df,
            "store_embeddings": store_emb_df,
        }

    def graph_feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importance from the sklearn fallback model.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns ``feature`` and ``importance``,
            sorted descending.  Returns ``None`` if using GNN mode.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_fitted()

        if self.backend_ != "sklearn":
            logger.info(
                "graph_feature_importance() is only available in sklearn mode."
            )
            return None

        importance = self.model_.feature_importances_
        df = pd.DataFrame({
            "feature": self._graph_feature_names,
            "importance": importance,
        })
        return df.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit()`` has been called successfully."""
        return self._fitted

    def summary(self) -> str:
        """Human-readable summary of the model state."""
        status = "fitted" if self._fitted else "not fitted"
        lines = [
            "GraphConsumerModel",
            f"  status      : {status}",
            f"  backend     : {self.backend_}",
        ]
        if self._fitted:
            lines.append(f"  origins     : {len(self.origin_ids_)}")
            lines.append(f"  stores      : {len(self.store_ids_)}")
            lines.append(f"  edges       : {len(self._edges_df)}")
            if self.backend_ == "gnn":
                lines.append(f"  hidden_dim  : {self.hidden_dim}")
                lines.append(f"  embed_dim   : {self.embedding_dim}")
                lines.append(f"  n_layers    : {self.n_layers}")
                lines.append(f"  epochs      : {self.epochs}")
                if self._training_losses:
                    lines.append(
                        f"  final_loss  : {self._training_losses[-1]:.4f}"
                    )
            else:
                lines.append(
                    f"  n_features  : {len(self._graph_feature_names)}"
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GraphConsumerModel(backend={self.backend_!r}, "
            f"fitted={self._fitted})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before using this method."
            )
