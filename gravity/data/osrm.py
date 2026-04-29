"""
OSRM (Open Source Routing Machine) distance and routing provider.

Replaces haversine straight-line distances with real driving, cycling, or
walking distances and travel times via the OSRM Table and Route APIs.

Works with the public OSRM demo server at ``https://router.project-osrm.org``
or any self-hosted OSRM instance.  The public demo server enforces a limit
of roughly 100 coordinates per Table request and is rate-limited, so this
module automatically batches large requests and throttles to one request per
second when using the default server.

If OSRM is unreachable the module falls back to haversine distances with a
logged warning.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from itertools import islice
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
import requests

from gravity.data.schema import build_distance_matrix, haversine_distance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PUBLIC_SERVER = "https://router.project-osrm.org"
_VALID_PROFILES = {"car", "bike", "foot"}
_VALID_METRICS = {"duration", "distance"}
# The public demo server supports roughly 100 coordinate pairs per request.
_DEFAULT_BATCH_SIZE = 100
_DEFAULT_TIMEOUT = 30  # seconds
_PUBLIC_RATE_LIMIT = 1.0  # minimum seconds between requests for the demo server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coords_string(lats: Sequence[float], lons: Sequence[float]) -> str:
    """Build an OSRM-formatted coordinate string: ``lon,lat;lon,lat;...``."""
    return ";".join(f"{lon},{lat}" for lat, lon in zip(lats, lons))


def _batched(iterable, n: int):
    """Yield successive *n*-length chunks from *iterable*."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OSRMDistanceProvider:
    """Provide real-world driving/cycling/walking distances via OSRM.

    Parameters
    ----------
    server_url : str
        Base URL of the OSRM server (no trailing slash).
    profile : str
        Routing profile: ``"car"``, ``"bike"``, or ``"foot"``.
    timeout : int
        Request timeout in seconds.
    batch_size : int
        Maximum number of total coordinates per Table API request.
        The public demo server supports roughly 100.
    rate_limit : float or None
        Minimum seconds between consecutive requests.  Defaults to 1.0 s
        when using the public server, ``None`` (no throttle) otherwise.

    Examples
    --------
    >>> osrm = OSRMDistanceProvider(profile="car")
    >>> matrix = osrm.distance_matrix(origins_df, stores_df, metric="duration")
    """

    def __init__(
        self,
        server_url: str = _PUBLIC_SERVER,
        profile: str = "car",
        timeout: int = _DEFAULT_TIMEOUT,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        rate_limit: Optional[float] = None,
    ) -> None:
        server_url = server_url.rstrip("/")
        if profile not in _VALID_PROFILES:
            raise ValueError(
                f"Invalid profile {profile!r}. Must be one of {_VALID_PROFILES}."
            )

        self.server_url = server_url
        self.profile = profile
        self.timeout = timeout
        self.batch_size = batch_size
        self._session = requests.Session()

        # Default to 1 req/s for the public demo server.
        if rate_limit is not None:
            self._rate_limit = rate_limit
        elif _PUBLIC_SERVER in server_url:
            self._rate_limit = _PUBLIC_RATE_LIMIT
        else:
            self._rate_limit = 0.0

        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep if necessary to respect the rate limit."""
        if self._rate_limit <= 0:
            return
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """Issue a throttled GET request and return the parsed JSON.

        Raises
        ------
        requests.HTTPError
            On non-200 responses.
        requests.ConnectionError / requests.Timeout
            On network failures.
        """
        self._throttle()
        self._last_request_time = time.monotonic()

        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "Ok":
            msg = data.get("message", data.get("code", "Unknown OSRM error"))
            raise RuntimeError(f"OSRM API error: {msg}")

        return data

    # ------------------------------------------------------------------
    # Table API
    # ------------------------------------------------------------------

    def _table_request(
        self,
        origin_lats: np.ndarray,
        origin_lons: np.ndarray,
        store_lats: np.ndarray,
        store_lons: np.ndarray,
        metric: str,
    ) -> np.ndarray:
        """Single Table API call for one batch of origins and stores.

        Returns an ``(n_origins, n_stores)`` numpy array of the requested
        metric (seconds for duration, metres for distance).
        """
        n_origins = len(origin_lats)
        n_stores = len(store_lats)

        # Build coordinate string: origins first, then stores
        all_lats = np.concatenate([origin_lats, store_lats])
        all_lons = np.concatenate([origin_lons, store_lons])
        coords = _coords_string(all_lats, all_lons)

        url = f"{self.server_url}/table/v1/{self.profile}/{coords}"

        # Tell OSRM which indices are sources vs destinations
        sources = ";".join(str(i) for i in range(n_origins))
        destinations = ";".join(str(i) for i in range(n_origins, n_origins + n_stores))

        annotations = metric  # "duration" or "distance"

        params = {
            "sources": sources,
            "destinations": destinations,
            "annotations": annotations,
        }

        data = self._get(url, params=params)

        key = f"{metric}s"  # "durations" or "distances"
        matrix = np.array(data[key], dtype=float)

        # OSRM returns ``null`` for unreachable pairs; replace with NaN.
        return np.where(matrix is None, np.nan, matrix)

    def distance_matrix(
        self,
        origins_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        metric: Literal["duration", "distance"] = "duration",
    ) -> pd.DataFrame:
        """Build an origin x store matrix of driving distances or durations.

        Parameters
        ----------
        origins_df : pd.DataFrame
            Must contain ``lat`` and ``lon`` columns.  Index is used as row
            labels in the returned matrix.
        stores_df : pd.DataFrame
            Must contain ``lat`` and ``lon`` columns.  Index is used as
            column labels in the returned matrix.
        metric : str
            ``"duration"`` returns seconds; ``"distance"`` returns metres.

        Returns
        -------
        pd.DataFrame
            Shape ``(len(origins_df), len(stores_df))`` with the same index
            and columns as ``schema.build_distance_matrix`` would produce.

        Notes
        -----
        Large requests are automatically split into batches that respect the
        OSRM coordinate limit.  The public server supports approximately 100
        coordinates (origins + destinations combined) per request.
        """
        if metric not in _VALID_METRICS:
            raise ValueError(
                f"Invalid metric {metric!r}. Must be one of {_VALID_METRICS}."
            )

        n_origins = len(origins_df)
        n_stores = len(stores_df)

        if n_origins == 0 or n_stores == 0:
            return pd.DataFrame(
                index=origins_df.index, columns=stores_df.index, dtype=float,
            )

        origin_lats = origins_df["lat"].values
        origin_lons = origins_df["lon"].values
        store_lats = stores_df["lat"].values
        store_lons = stores_df["lon"].values

        try:
            result = self._batched_table(
                origin_lats, origin_lons,
                store_lats, store_lons,
                metric,
            )
        except (requests.RequestException, RuntimeError, OSError) as exc:
            warnings.warn(
                f"OSRM request failed ({exc}). Falling back to haversine distances.",
                stacklevel=2,
            )
            logger.warning("OSRM unreachable (%s). Using haversine fallback.", exc)
            return build_distance_matrix(origins_df, stores_df)

        return pd.DataFrame(
            result,
            index=origins_df.index,
            columns=stores_df.index,
        )

    def _batched_table(
        self,
        origin_lats: np.ndarray,
        origin_lons: np.ndarray,
        store_lats: np.ndarray,
        store_lons: np.ndarray,
        metric: str,
    ) -> np.ndarray:
        """Execute the Table API in batches to stay within coordinate limits.

        The OSRM Table endpoint accepts at most ``batch_size`` total
        coordinates (sources + destinations).  This method partitions origins
        and stores into chunks so that each individual request fits, then
        assembles the full matrix from the partial results.
        """
        n_origins = len(origin_lats)
        n_stores = len(store_lats)

        # Determine how many origins and stores can fit in a single request.
        # We need at least 1 origin + 1 store, so each side gets at most
        # batch_size - 1 slots.
        max_stores_per_batch = min(n_stores, max(1, self.batch_size - 1))
        max_origins_per_batch = min(n_origins, max(1, self.batch_size - max_stores_per_batch))

        # If everything fits in one call, take the fast path.
        if n_origins + n_stores <= self.batch_size:
            return self._table_request(
                origin_lats, origin_lons,
                store_lats, store_lons,
                metric,
            )

        # Otherwise, iterate over origin-chunks x store-chunks.
        result = np.full((n_origins, n_stores), np.nan)

        origin_indices = list(range(n_origins))
        store_indices = list(range(n_stores))

        for o_batch in _batched(origin_indices, max_origins_per_batch):
            o_idx = np.array(o_batch)
            for s_batch in _batched(store_indices, max_stores_per_batch):
                s_idx = np.array(s_batch)
                sub = self._table_request(
                    origin_lats[o_idx], origin_lons[o_idx],
                    store_lats[s_idx], store_lons[s_idx],
                    metric,
                )
                result[np.ix_(o_idx, s_idx)] = sub

        return result

    # ------------------------------------------------------------------
    # Route API
    # ------------------------------------------------------------------

    def route(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> dict:
        """Compute a single route between two points.

        Parameters
        ----------
        origin_lat, origin_lon : float
            Origin coordinates in decimal degrees.
        dest_lat, dest_lon : float
            Destination coordinates in decimal degrees.

        Returns
        -------
        dict
            ``{"distance_m": float, "duration_s": float, "geometry": str}``
            where *geometry* is an encoded polyline string.

        If OSRM is unreachable, falls back to haversine distance (converted
        to metres) and an estimated duration using a fixed speed assumption.
        """
        coords = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        url = f"{self.server_url}/route/v1/{self.profile}/{coords}"

        try:
            data = self._get(url, params={"overview": "full", "geometries": "polyline"})
        except (requests.RequestException, RuntimeError, OSError) as exc:
            warnings.warn(
                f"OSRM route request failed ({exc}). Falling back to haversine.",
                stacklevel=2,
            )
            logger.warning("OSRM route unreachable (%s). Using haversine fallback.", exc)
            return self._haversine_route_fallback(
                origin_lat, origin_lon, dest_lat, dest_lon,
            )

        leg = data["routes"][0]
        return {
            "distance_m": leg["distance"],
            "duration_s": leg["duration"],
            "geometry": leg["geometry"],
        }

    def _haversine_route_fallback(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> dict:
        """Produce a best-effort route dict using haversine distance.

        Assumes average speeds of 50 km/h (car), 15 km/h (bike), and
        5 km/h (foot) to estimate duration.
        """
        dist_km = haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)
        dist_m = dist_km * 1000.0

        speed_kmh = {"car": 50.0, "bike": 15.0, "foot": 5.0}
        duration_s = (dist_km / speed_kmh.get(self.profile, 50.0)) * 3600.0

        return {
            "distance_m": dist_m,
            "duration_s": duration_s,
            "geometry": "",  # no polyline available for haversine fallback
        }

    # ------------------------------------------------------------------
    # Batch routes
    # ------------------------------------------------------------------

    def batch_routes(
        self,
        pairs: Sequence[tuple[float, float, float, float]],
    ) -> list[dict]:
        """Compute routes for multiple origin-destination pairs.

        Parameters
        ----------
        pairs : sequence of (origin_lat, origin_lon, dest_lat, dest_lon)
            Each element is a 4-tuple of coordinates.

        Returns
        -------
        list[dict]
            One route dict per pair, in the same order as *pairs*.
            Each dict has keys ``distance_m``, ``duration_s``, and
            ``geometry`` (see :meth:`route`).
        """
        results: list[dict] = []
        for origin_lat, origin_lon, dest_lat, dest_lon in pairs:
            results.append(self.route(origin_lat, origin_lon, dest_lat, dest_lon))
        return results

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"OSRMDistanceProvider(server_url={self.server_url!r}, "
            f"profile={self.profile!r})"
        )
