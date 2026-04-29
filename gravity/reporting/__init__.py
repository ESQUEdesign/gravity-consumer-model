# Lazy imports — reporting modules may depend on folium/matplotlib
try:
    from gravity.reporting.trade_area_report import TradeAreaReport
except ImportError:
    TradeAreaReport = None  # type: ignore[assignment,misc]

try:
    from gravity.reporting.consumer_profile import ConsumerProfileReport
except ImportError:
    ConsumerProfileReport = None  # type: ignore[assignment,misc]

try:
    from gravity.reporting.scenario import ScenarioSimulator
except ImportError:
    ScenarioSimulator = None  # type: ignore[assignment,misc]

try:
    from gravity.reporting.visualize import Visualizer
except ImportError:
    Visualizer = None  # type: ignore[assignment,misc]

__all__ = ["TradeAreaReport", "ConsumerProfileReport", "ScenarioSimulator", "Visualizer"]
