from gravity.data.schema import Store, ConsumerOrigin, Transaction, VisitEvent
from gravity.data.census import CensusLoader
from gravity.data.osm import OSMLoader
from gravity.data.safegraph import SafeGraphLoader
from gravity.data.transactions import TransactionLoader

# Optional loaders (require extra dependencies)
try:
    from gravity.data.google_places import GooglePlacesLoader
except ImportError:
    GooglePlacesLoader = None  # type: ignore[assignment,misc]

try:
    from gravity.data.osrm import OSRMDistanceProvider
except ImportError:
    OSRMDistanceProvider = None  # type: ignore[assignment,misc]

__all__ = [
    "Store", "ConsumerOrigin", "Transaction", "VisitEvent",
    "CensusLoader", "GooglePlacesLoader", "OSMLoader",
    "OSRMDistanceProvider", "SafeGraphLoader", "TransactionLoader",
]
