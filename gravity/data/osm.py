"""
OpenStreetMap points-of-interest loader.

Uses ``osmnx`` to query the Overpass API for retail POIs (shops,
restaurants, cafes, banks, etc.) within a bounding box or geocodable place
name, and converts them into ``Store`` pydantic models.

If ``osmnx`` is not installed the module still imports cleanly; only
the ``load_from_geojson`` path will work.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from gravity.data.schema import Store, stores_to_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TAGS: dict[str, Any] = {
    "shop": True,
    "amenity": ["restaurant", "cafe", "bank"],
}

# OSM key -> Store.category mapping for the most common retail types
_CATEGORY_MAP: dict[str, str] = {
    "supermarket": "grocery",
    "convenience": "convenience",
    "department_store": "department",
    "clothes": "apparel",
    "shoes": "apparel",
    "electronics": "electronics",
    "furniture": "furniture",
    "hardware": "hardware",
    "bakery": "food_specialty",
    "butcher": "food_specialty",
    "greengrocer": "food_specialty",
    "deli": "food_specialty",
    "restaurant": "restaurant",
    "cafe": "cafe",
    "fast_food": "fast_food",
    "bar": "bar",
    "pub": "bar",
    "bank": "bank",
    "pharmacy": "pharmacy",
    "optician": "health",
    "hairdresser": "personal_care",
    "beauty": "personal_care",
    "car_repair": "auto_services",
    "car": "auto_dealership",
    "fuel": "gas_station",
}


def _osmnx_available() -> bool:
    """Return *True* if osmnx can be imported."""
    try:
        import osmnx  # noqa: F401
        return True
    except ImportError:
        return False


def _derive_category(tags: dict[str, Any]) -> str:
    """Best-effort category assignment from OSM tag values."""
    # Check "shop" value first, then "amenity"
    for key in ("shop", "amenity", "leisure", "tourism"):
        val = tags.get(key)
        if val and isinstance(val, str):
            if val in _CATEGORY_MAP:
                return _CATEGORY_MAP[val]
            return val  # use raw tag value as fallback
    return "other"


def _stable_id(name: Optional[str], lat: float, lon: float) -> str:
    """Generate a deterministic store id from name + coordinates."""
    raw = f"{name or 'unknown'}_{lat:.6f}_{lon:.6f}"
    return "osm_" + hashlib.md5(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# GeoDataFrame -> Store conversion
# ---------------------------------------------------------------------------

def _gdf_to_stores(gdf: "pd.DataFrame") -> list[Store]:
    """Convert an osmnx-style GeoDataFrame (or plain DataFrame) to Stores.

    Expects columns that include at least ``geometry`` (Point) or explicit
    ``lat`` / ``lon`` columns plus optional ``name``, ``shop``, ``amenity``,
    etc.
    """
    stores: list[Store] = []

    for idx, row in gdf.iterrows():
        # Resolve coordinates
        lat: float | None = None
        lon: float | None = None
        geom = row.get("geometry")
        if geom is not None and hasattr(geom, "y"):
            lat, lon = float(geom.y), float(geom.x)
        if lat is None:
            lat = float(row.get("lat", row.get("latitude", 0.0)))
            lon = float(row.get("lon", row.get("longitude", 0.0)))

        if lat == 0.0 and lon == 0.0:
            continue  # skip entries with no usable coordinates

        name = row.get("name") or None
        if isinstance(name, float) and np.isnan(name):
            name = None

        tags: dict[str, Any] = {}
        for col in ("shop", "amenity", "leisure", "tourism", "cuisine",
                     "brand", "opening_hours", "wheelchair", "phone",
                     "website", "addr:street", "addr:housenumber",
                     "addr:city", "addr:postcode"):
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                tags[col] = val

        category = _derive_category(tags)

        # Build the Store model
        store_id = str(row.get("osmid", idx))
        if store_id == str(idx) or not store_id:
            store_id = _stable_id(name, lat, lon)

        brand = tags.get("brand")
        if isinstance(brand, float) and np.isnan(brand):
            brand = None

        stores.append(
            Store(
                store_id=store_id,
                name=name,
                lat=lat,
                lon=lon,
                category=category,
                brand=brand if isinstance(brand, str) else None,
                attributes=tags,
            )
        )

    return stores


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class OSMLoader:
    """Load retail points of interest from OpenStreetMap.

    Parameters
    ----------
    network_type : str
        osmnx network type (only relevant if you later need the road
        network; not used for POI queries).

    Examples
    --------
    >>> loader = OSMLoader()
    >>> stores = loader.load_retail_pois(place="Manhattan, NY")
    >>> df = loader.to_dataframe(stores)
    """

    def __init__(self, network_type: str = "drive") -> None:
        self.network_type = network_type
        self._osmnx_ok = _osmnx_available()
        if not self._osmnx_ok:
            logger.info(
                "osmnx is not installed. Only load_from_geojson() is available. "
                "Install with: pip install osmnx"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_retail_pois(
        self,
        bbox: Optional[tuple[float, float, float, float]] = None,
        place: Optional[str] = None,
        tags: Optional[dict[str, Any]] = None,
    ) -> list[Store]:
        """Query OpenStreetMap for retail POIs and return ``Store`` models.

        You must supply exactly one of *bbox* or *place*.

        Parameters
        ----------
        bbox : tuple[float, float, float, float] or None
            ``(west, south, east, north)`` bounding box in decimal degrees.
        place : str or None
            A geocodable place name, e.g. ``"Manhattan, NY"``.
        tags : dict or None
            OSM tag filter passed to ``osmnx.features_from_*``.  Defaults
            to shops plus restaurants, cafes, and banks.

        Returns
        -------
        list[Store]

        Raises
        ------
        RuntimeError
            If ``osmnx`` is not installed.
        ValueError
            If neither *bbox* nor *place* is provided (or both are).
        """
        if not self._osmnx_ok:
            raise RuntimeError(
                "osmnx is required for live Overpass queries. "
                "Install with: pip install osmnx"
            )

        if (bbox is None) == (place is None):
            raise ValueError("Provide exactly one of `bbox` or `place`.")

        import osmnx as ox

        tags = tags or dict(DEFAULT_TAGS)

        try:
            if bbox is not None:
                west, south, east, north = bbox
                gdf = ox.features_from_bbox(
                    north=north, south=south, east=east, west=west, tags=tags,
                )
            else:
                gdf = ox.features_from_place(place, tags=tags)
        except Exception as exc:
            logger.error("osmnx query failed: %s", exc)
            raise

        # osmnx returns a GeoDataFrame; filter to Point geometries
        # (polygons represent building footprints — take their centroid)
        if hasattr(gdf, "geometry"):
            point_mask = gdf.geometry.geom_type == "Point"
            poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            # Convert polygon centroids to points
            centroids = gdf.loc[poly_mask].copy()
            if not centroids.empty:
                centroids["geometry"] = centroids.geometry.centroid
                gdf = pd.concat([gdf.loc[point_mask], centroids], ignore_index=True)
            else:
                gdf = gdf.loc[point_mask].reset_index(drop=True)

        stores = _gdf_to_stores(gdf)
        logger.info(
            "Loaded %d retail POIs from OSM (place=%s, bbox=%s)",
            len(stores), place, bbox,
        )
        return stores

    def load_from_geojson(self, filepath: str | Path) -> list[Store]:
        """Load POI data from a GeoJSON file.

        The GeoJSON should contain Point (or Polygon) features with
        ``name``, ``shop``, ``amenity``, and coordinate properties.

        Parameters
        ----------
        filepath : str or Path
            Path to the ``.geojson`` file.

        Returns
        -------
        list[Store]
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as fh:
            geojson = json.load(fh)

        features = geojson.get("features", [])
        if not features:
            warnings.warn(f"No features found in {filepath}")
            return []

        records: list[dict[str, Any]] = []
        for feat in features:
            props = dict(feat.get("properties", {}))
            geom = feat.get("geometry", {})

            # Extract coordinates
            coords = geom.get("coordinates")
            if geom.get("type") == "Point" and coords:
                props["lon"] = coords[0]
                props["lat"] = coords[1]
            elif geom.get("type") in ("Polygon", "MultiPolygon") and coords:
                # Use centroid of first ring for polygons
                try:
                    ring = coords[0] if geom["type"] == "Polygon" else coords[0][0]
                    lons = [c[0] for c in ring]
                    lats = [c[1] for c in ring]
                    props["lon"] = sum(lons) / len(lons)
                    props["lat"] = sum(lats) / len(lats)
                except (IndexError, TypeError):
                    continue
            else:
                continue

            # Carry over an ID if present
            if "id" in feat:
                props.setdefault("osmid", feat["id"])

            records.append(props)

        df = pd.DataFrame(records)
        stores = _gdf_to_stores(df)
        logger.info("Loaded %d stores from GeoJSON: %s", len(stores), filepath)
        return stores

    @staticmethod
    def to_dataframe(stores: list[Store]) -> pd.DataFrame:
        """Convenience wrapper around ``stores_to_dataframe``."""
        return stores_to_dataframe(stores)
