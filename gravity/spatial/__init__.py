# Lazy imports — geopandas, libpysal, spreg are optional
try:
    from gravity.spatial.trade_area import TradeAreaAnalyzer
except ImportError:
    TradeAreaAnalyzer = None  # type: ignore[assignment,misc]

try:
    from gravity.spatial.spatial_econometrics import SpatialEconModel
except ImportError:
    SpatialEconModel = None  # type: ignore[assignment,misc]

try:
    from gravity.spatial.isochrone import IsochroneGenerator
except ImportError:
    IsochroneGenerator = None  # type: ignore[assignment,misc]

__all__ = ["TradeAreaAnalyzer", "SpatialEconModel", "IsochroneGenerator"]
