# Lazy imports — ML dependencies (xgboost, lightgbm, torch) are optional
try:
    from gravity.ml.residual_boost import ResidualBoostModel
except Exception:
    ResidualBoostModel = None  # type: ignore[assignment,misc]

try:
    from gravity.ml.graph_network import GraphConsumerModel
except Exception:
    GraphConsumerModel = None  # type: ignore[assignment,misc]

__all__ = ["ResidualBoostModel", "GraphConsumerModel"]
