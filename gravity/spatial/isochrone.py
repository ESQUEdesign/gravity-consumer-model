"""
Isochrone Generator
===================
Generate drive-time, walk-time, and bike-time isochrone polygons around
retail locations.

An **isochrone** is the geographic area reachable from a point within a
given travel time.  Isochrones are used in trade area analysis to define
realistic catchment zones that account for road networks and travel
modes -- a significant improvement over simple distance buffers.

The generator uses the **OSRM (Open Source Routing Machine)** public
demo API or a locally-hosted OSRM instance to query travel times.  For
each origin, rays are cast at regular bearing intervals, OSRM is
queried to determine how far one can travel along the road network, and
the resulting points are assembled into a convex hull or concave
boundary polygon.

When OSRM is unavailable (no network, rate-limited, etc.) a graceful
fallback generates a circular approximation using average travel speeds.

Usage
-----
>>> from gravity.spatial.isochrone import IsochroneGenerator
>>> gen = IsochroneGenerator()
>>> iso = gen.generate(lat=40.748, lon=-73.986, minutes=10, mode="drive")
>>> gen.contains_point(iso, 40.745, -73.990)
True

References
----------
OSRM project: http://project-osrm.org/
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from shapely.geometry import Point, Polygon, MultiPoint, mapping, shape
    from shapely.prepared import prep
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

try:
    import geopandas as gpd
    _HAS_GEOPANDAS = True
except ImportError:
    _HAS_GEOPANDAS = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Average travel speeds (km/h) for fallback circular approximation.
_FALLBACK_SPEEDS_KMH = {
    "drive": 40.0,
    "walk": 5.0,
    "bike": 15.0,
}

# OSRM profile mapping.
_OSRM_PROFILES = {
    "drive": "car",
    "walk": "foot",
    "bike": "bicycle",
}

_EARTH_RADIUS_KM = 6371.0

# Number of bearing rays to cast for isochrone generation.
_DEFAULT_N_BEARINGS = 36


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """Compute the destination point given a start, bearing, and distance.

    Uses the Vincenty direct formula on a spherical Earth.

    Parameters
    ----------
    lat, lon : float
        Starting point in decimal degrees.
    bearing_deg : float
        Bearing in degrees clockwise from north.
    distance_km : float
        Distance to travel in kilometres.

    Returns
    -------
    tuple[float, float]
        (lat, lon) of the destination point in decimal degrees.
    """
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    bearing_r = math.radians(bearing_deg)
    d_over_R = distance_km / _EARTH_RADIUS_KM

    lat2 = math.asin(
        math.sin(lat_r) * math.cos(d_over_R)
        + math.cos(lat_r) * math.sin(d_over_R) * math.cos(bearing_r)
    )
    lon2 = lon_r + math.atan2(
        math.sin(bearing_r) * math.sin(d_over_R) * math.cos(lat_r),
        math.cos(d_over_R) - math.sin(lat_r) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometres between two points."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return _EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# IsochroneGenerator
# ---------------------------------------------------------------------------

class IsochroneGenerator:
    """Generate drive-time, walk-time, and bike-time isochrone polygons.

    Parameters
    ----------
    osrm_base_url : str, default ``"http://router.project-osrm.org"``
        Base URL of the OSRM routing service.  Set to a local OSRM
        instance URL for higher throughput and no rate limits.
    n_bearings : int, default 36
        Number of evenly-spaced compass bearings to sample.  More
        bearings produce smoother polygons at the cost of additional
        OSRM queries.
    timeout : float, default 5.0
        HTTP request timeout in seconds per OSRM query.
    fallback_on_error : bool, default True
        If True, fall back to a circular approximation when OSRM
        queries fail.  If False, raise the exception.

    Examples
    --------
    >>> gen = IsochroneGenerator()
    >>> iso = gen.generate(40.748, -73.986, 10, "drive")
    >>> iso["type"]
    'Polygon'
    >>> gen.contains_point(iso, 40.750, -73.980)
    True
    """

    def __init__(
        self,
        osrm_base_url: str = "http://router.project-osrm.org",
        n_bearings: int = _DEFAULT_N_BEARINGS,
        timeout: float = 5.0,
        fallback_on_error: bool = True,
    ) -> None:
        self.osrm_base_url = osrm_base_url.rstrip("/")
        self.n_bearings = n_bearings
        self.timeout = timeout
        self.fallback_on_error = fallback_on_error

    # ------------------------------------------------------------------
    # Core: generate single isochrone
    # ------------------------------------------------------------------

    def generate(
        self,
        lat: float,
        lon: float,
        minutes: float,
        mode: str = "drive",
        *,
        n_bearings: Optional[int] = None,
    ) -> dict:
        """Generate an isochrone polygon as a GeoJSON geometry dict.

        Casts rays outward from ``(lat, lon)`` at evenly-spaced bearings,
        queries OSRM for the farthest point reachable within *minutes*
        along each ray, and wraps the result set in a convex hull
        polygon.

        Parameters
        ----------
        lat : float
            Origin latitude (decimal degrees).
        lon : float
            Origin longitude (decimal degrees).
        minutes : float
            Travel time budget in minutes.
        mode : str, default ``"drive"``
            Travel mode: ``"drive"``, ``"walk"``, or ``"bike"``.
        n_bearings : int or None
            Override the instance-level bearing count for this call.

        Returns
        -------
        dict
            GeoJSON Polygon geometry (``{"type": "Polygon", "coordinates": [...]}``).
            If only one or two reachable points are found, a Point or
            LineString may be returned.

        Raises
        ------
        ValueError
            If *mode* is not one of ``"drive"``, ``"walk"``, ``"bike"``.
        """
        if mode not in _OSRM_PROFILES:
            raise ValueError(
                f"mode must be one of {list(_OSRM_PROFILES.keys())}, got {mode!r}"
            )

        n = n_bearings or self.n_bearings
        bearings = np.linspace(0, 360, n, endpoint=False)

        # Estimate a generous search radius (straight-line km).
        speed_kmh = _FALLBACK_SPEEDS_KMH[mode]
        max_radius_km = speed_kmh * (minutes / 60.0) * 1.5

        # Generate candidate destination points along each bearing.
        candidates = []
        for bearing in bearings:
            dest_lat, dest_lon = _destination_point(lat, lon, bearing, max_radius_km)
            candidates.append((bearing, dest_lat, dest_lon))

        # Try OSRM routing.
        reachable_points = []
        used_osrm = False

        if _HAS_REQUESTS:
            profile = _OSRM_PROFILES[mode]
            for bearing, dest_lat, dest_lon in candidates:
                pt = self._osrm_reachable_point(
                    lat, lon, dest_lat, dest_lon, minutes, profile
                )
                if pt is not None:
                    reachable_points.append(pt)
                    used_osrm = True

        # If OSRM produced nothing (or requests not available), fall back.
        if not reachable_points:
            if not self.fallback_on_error and _HAS_REQUESTS:
                raise RuntimeError(
                    "OSRM returned no reachable points and "
                    "fallback_on_error is False."
                )
            logger.info(
                "Using circular fallback for isochrone "
                "(%.4f, %.4f, %s, %.0f min).",
                lat, lon, mode, minutes,
            )
            reachable_points = self._circular_fallback(lat, lon, minutes, mode, n)

        # Build the polygon.
        return self._points_to_geojson(reachable_points, lat, lon)

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def batch_generate(
        self,
        locations: list[tuple[float, float]],
        minutes: float,
        mode: str = "drive",
    ) -> list[dict]:
        """Generate isochrones for multiple locations.

        Parameters
        ----------
        locations : list of (lat, lon) tuples
            Origin points.
        minutes : float
            Travel time budget in minutes.
        mode : str, default ``"drive"``
            Travel mode.

        Returns
        -------
        list[dict]
            List of GeoJSON Polygon geometries, one per location.
        """
        results = []
        for i, (lat, lon) in enumerate(locations):
            logger.debug("Generating isochrone %d/%d...", i + 1, len(locations))
            iso = self.generate(lat, lon, minutes, mode)
            results.append(iso)
        return results

    # ------------------------------------------------------------------
    # Point-in-polygon test
    # ------------------------------------------------------------------

    @staticmethod
    def contains_point(
        isochrone_geojson: dict,
        lat: float,
        lon: float,
    ) -> bool:
        """Check whether a point falls within an isochrone polygon.

        Parameters
        ----------
        isochrone_geojson : dict
            GeoJSON geometry dict (Polygon, MultiPolygon, Point, etc.).
        lat : float
            Query latitude.
        lon : float
            Query longitude.

        Returns
        -------
        bool
            True if the point is inside (or on the boundary of) the
            isochrone polygon.
        """
        if not _HAS_SHAPELY:
            raise ImportError(
                "shapely is required for contains_point().  "
                "Install with: pip install shapely"
            )

        geom = shape(isochrone_geojson)
        point = Point(lon, lat)
        return geom.contains(point) or geom.boundary.distance(point) < 1e-10

    # ------------------------------------------------------------------
    # GeoDataFrame export
    # ------------------------------------------------------------------

    @staticmethod
    def to_geodataframe(
        isochrones: list[dict],
        labels: Optional[list[str]] = None,
    ) -> "gpd.GeoDataFrame":
        """Convert a list of GeoJSON isochrone geometries to a GeoDataFrame.

        Parameters
        ----------
        isochrones : list[dict]
            List of GeoJSON geometry dicts (as returned by ``generate``
            or ``batch_generate``).
        labels : list[str] or None
            Optional labels for each isochrone (used as the index).

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with a ``geometry`` column (CRS EPSG:4326).

        Raises
        ------
        ImportError
            If geopandas or shapely is not installed.
        """
        if not _HAS_GEOPANDAS:
            raise ImportError(
                "geopandas is required for to_geodataframe().  "
                "Install with: pip install geopandas"
            )
        if not _HAS_SHAPELY:
            raise ImportError(
                "shapely is required for to_geodataframe().  "
                "Install with: pip install shapely"
            )

        geometries = [shape(g) for g in isochrones]
        index = labels if labels else list(range(len(isochrones)))
        gdf = gpd.GeoDataFrame(
            {"label": index},
            geometry=geometries,
            crs="EPSG:4326",
        )
        return gdf

    # ------------------------------------------------------------------
    # OSRM helpers
    # ------------------------------------------------------------------

    def _osrm_reachable_point(
        self,
        orig_lat: float,
        orig_lon: float,
        dest_lat: float,
        dest_lon: float,
        minutes: float,
        profile: str,
    ) -> Optional[tuple[float, float]]:
        """Query OSRM route and find the farthest reachable point within time.

        Uses a binary search along the route geometry to find the point
        that is just within the time budget.

        Parameters
        ----------
        orig_lat, orig_lon : float
            Origin coordinates.
        dest_lat, dest_lon : float
            Far-away destination to define the ray direction.
        minutes : float
            Time budget.
        profile : str
            OSRM profile (``"car"``, ``"foot"``, ``"bicycle"``).

        Returns
        -------
        tuple[float, float] or None
            ``(lat, lon)`` of the farthest reachable point, or None
            on failure.
        """
        url = (
            f"{self.osrm_base_url}/route/v1/{profile}/"
            f"{orig_lon},{orig_lat};{dest_lon},{dest_lat}"
            f"?overview=full&geometries=geojson"
        )

        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:
            return None

        if data.get("code") != "Ok" or not data.get("routes"):
            return None

        route = data["routes"][0]
        total_duration_s = route.get("duration", 0)
        geometry = route.get("geometry", {})
        coords = geometry.get("coordinates", [])

        if not coords:
            return None

        time_budget_s = minutes * 60.0

        if total_duration_s <= time_budget_s:
            # The entire route is within budget.
            last = coords[-1]
            return (last[1], last[0])  # (lat, lon)

        # Interpolate along the route to find the point at the time budget.
        # Approximate: assume uniform speed along the route.
        fraction = time_budget_s / total_duration_s if total_duration_s > 0 else 0.0

        # Compute cumulative distances along the route.
        dists = [0.0]
        for i in range(1, len(coords)):
            d = _haversine_km(coords[i - 1][1], coords[i - 1][0],
                              coords[i][1], coords[i][0])
            dists.append(dists[-1] + d)

        total_dist = dists[-1]
        target_dist = fraction * total_dist

        # Find the segment containing the target distance.
        for i in range(1, len(dists)):
            if dists[i] >= target_dist:
                seg_start = dists[i - 1]
                seg_end = dists[i]
                seg_fraction = (
                    (target_dist - seg_start) / (seg_end - seg_start)
                    if seg_end > seg_start
                    else 0.0
                )
                interp_lon = coords[i - 1][0] + seg_fraction * (coords[i][0] - coords[i - 1][0])
                interp_lat = coords[i - 1][1] + seg_fraction * (coords[i][1] - coords[i - 1][1])
                return (interp_lat, interp_lon)

        # Fallback: return the last coordinate.
        last = coords[-1]
        return (last[1], last[0])

    # ------------------------------------------------------------------
    # Circular fallback
    # ------------------------------------------------------------------

    def _circular_fallback(
        self,
        lat: float,
        lon: float,
        minutes: float,
        mode: str,
        n_bearings: int,
    ) -> list[tuple[float, float]]:
        """Generate a circular set of points as a fallback when OSRM is unavailable.

        Parameters
        ----------
        lat, lon : float
            Origin coordinates.
        minutes : float
            Travel time.
        mode : str
            Travel mode (determines speed).
        n_bearings : int
            Number of points on the circle.

        Returns
        -------
        list[tuple[float, float]]
            List of (lat, lon) tuples forming the circle boundary.
        """
        speed_kmh = _FALLBACK_SPEEDS_KMH.get(mode, 30.0)
        radius_km = speed_kmh * (minutes / 60.0)

        points = []
        for bearing in np.linspace(0, 360, n_bearings, endpoint=False):
            pt = _destination_point(lat, lon, bearing, radius_km)
            points.append(pt)
        return points

    # ------------------------------------------------------------------
    # GeoJSON construction
    # ------------------------------------------------------------------

    def _points_to_geojson(
        self,
        points: list[tuple[float, float]],
        center_lat: float,
        center_lon: float,
    ) -> dict:
        """Convert a list of (lat, lon) boundary points to a GeoJSON polygon.

        Parameters
        ----------
        points : list[tuple[float, float]]
            Boundary points as (lat, lon) tuples.
        center_lat, center_lon : float
            The isochrone center point (included in the hull).

        Returns
        -------
        dict
            GeoJSON geometry dictionary.
        """
        if _HAS_SHAPELY:
            # Use shapely for a proper convex hull.
            shapely_points = [Point(lon, lat) for lat, lon in points]
            shapely_points.append(Point(center_lon, center_lat))
            mp = MultiPoint(shapely_points)
            hull = mp.convex_hull
            return mapping(hull)
        else:
            # Manual convex hull via Graham scan would be complex.
            # Use a simple polygon from sorted angles.
            all_pts = list(points) + [(center_lat, center_lon)]
            # Convert to (lon, lat) for GeoJSON.
            lonlat = [(lon, lat) for lat, lon in all_pts]
            # Sort by angle from centroid.
            cx = np.mean([p[0] for p in lonlat])
            cy = np.mean([p[1] for p in lonlat])
            sorted_pts = sorted(
                lonlat, key=lambda p: math.atan2(p[1] - cy, p[0] - cx)
            )
            # Close the ring.
            sorted_pts.append(sorted_pts[0])
            return {
                "type": "Polygon",
                "coordinates": [sorted_pts],
            }

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"IsochroneGenerator(osrm={self.osrm_base_url!r}, "
            f"n_bearings={self.n_bearings}, "
            f"fallback={self.fallback_on_error})"
        )
