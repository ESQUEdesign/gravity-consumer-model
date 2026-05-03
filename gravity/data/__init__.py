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

try:
    from gravity.data.census_expanded import CensusExpandedLoader
except ImportError:
    CensusExpandedLoader = None  # type: ignore[assignment,misc]

try:
    from gravity.data.sec_edgar import SECEdgarLoader
except ImportError:
    SECEdgarLoader = None  # type: ignore[assignment,misc]

try:
    from gravity.data.healthdata import HealthDataLoader
except ImportError:
    HealthDataLoader = None  # type: ignore[assignment,misc]

try:
    from gravity.data.fred import FREDLoader
except ImportError:
    FREDLoader = None  # type: ignore[assignment,misc]

try:
    from gravity.data.data_commons import DataCommonsLoader
except ImportError:
    DataCommonsLoader = None  # type: ignore[assignment,misc]

try:
    from gravity.data.fmp import FinancialModelingPrepLoader
except ImportError:
    FinancialModelingPrepLoader = None  # type: ignore[assignment,misc]

try:
    from gravity.data.ckan_datagov import CKANDataGovLoader
except ImportError:
    CKANDataGovLoader = None  # type: ignore[assignment,misc]

try:
    from gravity.data.noaa_climate import NOAAClimateLoader
except ImportError:
    NOAAClimateLoader = None  # type: ignore[assignment,misc]

__all__ = [
    "Store", "ConsumerOrigin", "Transaction", "VisitEvent",
    "CensusLoader", "GooglePlacesLoader", "OSMLoader",
    "OSRMDistanceProvider", "SafeGraphLoader", "TransactionLoader",
    "CensusExpandedLoader", "SECEdgarLoader", "HealthDataLoader",
    "FREDLoader", "DataCommonsLoader", "FinancialModelingPrepLoader",
    "CKANDataGovLoader", "NOAAClimateLoader",
]
