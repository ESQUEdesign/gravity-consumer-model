"""
Google Places API data loader.

Enriches ``Store`` models with quality and review data from the Google Places
API (New), and discovers new retail locations via Nearby Search.  This
complements the OSM loader, which provides location geometry but lacks
ratings, reviews, price levels, and operational details.

Endpoints used (Places API New):
    - Nearby Search:  POST https://places.googleapis.com/v1/places:searchNearby
    - Place Details:  GET  https://places.googleapis.com/v1/places/{place_id}

Authentication is via the ``X-Goog-Api-Key`` header; field selection is
controlled by ``X-Goog-FieldMask``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from difflib import SequenceMatcher
from typing import Any, Optional

import numpy as np

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

from gravity.data.schema import Store, haversine_distance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEARBY_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"
_PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"

# Default fields requested from Nearby Search
_NEARBY_FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.businessStatus",
    "places.types",
    "places.primaryType",
])

# Default fields requested from Place Details
_DETAILS_FIELD_MASK = ",".join([
    "id",
    "displayName",
    "formattedAddress",
    "location",
    "rating",
    "userRatingCount",
    "priceLevel",
    "businessStatus",
    "types",
    "primaryType",
    "regularOpeningHours",
    "currentOpeningHours",
    "reviews",
    "websiteUri",
    "nationalPhoneNumber",
    "internationalPhoneNumber",
    "photos",
    "editorialSummary",
])

# Google price-level string -> integer mapping
_PRICE_LEVEL_MAP: dict[str, int] = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}

# Default request delay (seconds) — keeps us well under 100 req/s
_DEFAULT_DELAY = 0.05

# Name similarity threshold for matching enrichment candidates
_NAME_SIMILARITY_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_id(name: Optional[str], lat: float, lon: float) -> str:
    """Generate a deterministic store id from name + coordinates."""
    raw = f"{name or 'unknown'}_{lat:.6f}_{lon:.6f}"
    return "gp_" + hashlib.md5(raw.encode()).hexdigest()[:12]


def _name_similarity(a: Optional[str], b: Optional[str]) -> float:
    """Return 0-1 similarity ratio between two store names.

    Returns 1.0 if both are None/empty (assumed match when names are
    unavailable).
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _parse_price_level(raw: Any) -> int:
    """Convert a Google price-level string or int to an integer 1-4.

    Falls back to 2 (moderate) when the value is missing or unrecognised.
    The Store schema expects a value in [1, 4], so we clamp accordingly.
    """
    if isinstance(raw, int):
        return max(1, min(raw, 4))
    if isinstance(raw, str):
        val = _PRICE_LEVEL_MAP.get(raw)
        if val is not None:
            return max(1, min(val, 4))
    return 2


def _place_to_store(place: dict[str, Any]) -> Store:
    """Convert a single Google Places API result dict to a ``Store``."""
    location = place.get("location", {})
    lat = float(location.get("latitude", 0.0))
    lon = float(location.get("longitude", 0.0))

    display_name = place.get("displayName", {})
    name = display_name.get("text") if isinstance(display_name, dict) else None

    place_id = place.get("id", "")
    store_id = place_id if place_id else _stable_id(name, lat, lon)

    rating = float(place.get("rating", 0.0))
    price_level = _parse_price_level(place.get("priceLevel"))
    user_ratings_total = int(place.get("userRatingCount", 0))
    business_status = place.get("businessStatus", "")
    types = place.get("types", [])
    primary_type = place.get("primaryType", "")

    # Best-effort category from primaryType or first type
    category = primary_type or (types[0] if types else "other")

    return Store(
        store_id=store_id,
        name=name,
        lat=lat,
        lon=lon,
        avg_rating=rating,
        price_level=price_level,
        category=category,
        attributes={
            "google_place_id": place_id,
            "user_ratings_total": user_ratings_total,
            "business_status": business_status,
            "types": types,
            "primary_type": primary_type,
            "formatted_address": place.get("formattedAddress", ""),
        },
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class GooglePlacesLoader:
    """Load and enrich retail data using the Google Places API (New).

    Parameters
    ----------
    api_key : str or None
        Google Places API key.  If *None*, reads from the
        ``GOOGLE_PLACES_API_KEY`` environment variable.
    request_delay : float
        Seconds to wait between successive API requests.  The default
        (0.05 s) keeps throughput well under Google's 100-req/s quota.

    Raises
    ------
    ValueError
        If no API key is found.

    Examples
    --------
    >>> loader = GooglePlacesLoader()
    >>> stores = loader.search_nearby(40.7580, -73.9855, radius_m=500)
    >>> enriched = loader.enrich_stores(stores, search_radius_m=100)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        request_delay: float = _DEFAULT_DELAY,
    ) -> None:
        if requests is None:
            raise ImportError(
                "The 'requests' package is required for GooglePlacesLoader. "
                "Install it with: pip install requests"
            )
        self.api_key = api_key or os.environ.get("GOOGLE_PLACES_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google Places API key is required.  Either pass api_key= "
                "or set the GOOGLE_PLACES_API_KEY environment variable."
            )
        self.request_delay = request_delay
        self._session = requests.Session()
        self._session.headers.update({"X-Goog-Api-Key": self.api_key})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep for the configured delay to respect rate limits."""
        if self.request_delay > 0:
            time.sleep(self.request_delay)

    def _post(self, url: str, payload: dict, field_mask: str) -> dict:
        """Execute a POST request with standard headers and error handling.

        Returns the parsed JSON response body or an empty dict on failure.
        """
        headers = {"X-Goog-FieldMask": field_mask}
        self._throttle()
        try:
            resp = self._session.post(url, json=payload, headers=headers, timeout=30)
        except requests.exceptions.RequestException as exc:
            logger.error("Network error during POST %s: %s", url, exc)
            return {}

        if resp.status_code == 429:
            logger.warning("Google Places API rate limit exceeded.  Backing off.")
            time.sleep(2.0)
            return {}
        if resp.status_code == 403:
            logger.error(
                "Google Places API returned 403 — check that your API key is "
                "valid and has the Places API (New) enabled."
            )
            return {}
        if resp.status_code != 200:
            logger.error(
                "Google Places API error %d: %s", resp.status_code, resp.text[:500]
            )
            return {}

        return resp.json()

    def _get(self, url: str, field_mask: str) -> dict:
        """Execute a GET request with standard headers and error handling.

        Returns the parsed JSON response body or an empty dict on failure.
        """
        headers = {"X-Goog-FieldMask": field_mask}
        self._throttle()
        try:
            resp = self._session.get(url, headers=headers, timeout=30)
        except requests.exceptions.RequestException as exc:
            logger.error("Network error during GET %s: %s", url, exc)
            return {}

        if resp.status_code == 429:
            logger.warning("Google Places API rate limit exceeded.  Backing off.")
            time.sleep(2.0)
            return {}
        if resp.status_code == 403:
            logger.error(
                "Google Places API returned 403 — check that your API key is "
                "valid and has the Places API (New) enabled."
            )
            return {}
        if resp.status_code != 200:
            logger.error(
                "Google Places API error %d: %s", resp.status_code, resp.text[:500]
            )
            return {}

        return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_nearby(
        self,
        lat: float,
        lon: float,
        radius_m: float = 1000,
        type: Optional[str] = None,
        keyword: Optional[str] = None,
        max_results: int = 20,
    ) -> list[Store]:
        """Discover retail locations via Google Places Nearby Search.

        Parameters
        ----------
        lat, lon : float
            Centre point of the search circle.
        radius_m : float
            Search radius in metres (max 50 000).
        type : str or None
            Restrict to a single Google place type, e.g. ``"restaurant"``.
        keyword : str or None
            Free-text keyword filter (passed as ``textQuery`` in
            ``routingParameters`` — note: the Nearby Search (New) uses
            ``includedTypes`` rather than a keyword field, so this is
            converted to a type filter when possible).
        max_results : int
            Maximum number of results to return.

        Returns
        -------
        list[Store]
            Discovered locations as ``Store`` models.
        """
        payload: dict[str, Any] = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lon},
                    "radius": min(float(radius_m), 50000.0),
                },
            },
            "maxResultCount": min(max_results, 20),
        }

        # The Nearby Search (New) endpoint uses includedTypes (list) and
        # excludedTypes rather than a single ``type`` string.
        included_types: list[str] = []
        if type:
            included_types.append(type)
        if keyword and not type:
            # Best-effort: treat keyword as a type filter
            included_types.append(keyword)
        if included_types:
            payload["includedTypes"] = included_types

        data = self._post(_NEARBY_SEARCH_URL, payload, _NEARBY_FIELD_MASK)
        places = data.get("places", [])

        stores = [_place_to_store(p) for p in places]
        logger.info(
            "Nearby search at (%.5f, %.5f) r=%dm returned %d results",
            lat, lon, radius_m, len(stores),
        )
        return stores

    def get_place_details(self, place_id: str) -> dict[str, Any]:
        """Fetch full details for a single place.

        Parameters
        ----------
        place_id : str
            The Google ``places/`` resource name, e.g.
            ``"places/ChIJN1t_tDeuEmsRUsoyG83frY4"``.  If the ``places/``
            prefix is missing it is prepended automatically.

        Returns
        -------
        dict
            Raw detail fields including ``reviews``, ``regularOpeningHours``,
            ``websiteUri``, ``nationalPhoneNumber``, and ``photos``.  Returns
            an empty dict on any error.
        """
        if not place_id.startswith("places/"):
            place_id = f"places/{place_id}"

        url = _PLACE_DETAILS_URL.format(place_id=place_id)
        return self._get(url, _DETAILS_FIELD_MASK)

    def enrich_stores(
        self,
        stores: list[Store],
        search_radius_m: float = 100,
    ) -> list[Store]:
        """Enrich existing stores with Google Places quality data.

        For each store, a Nearby Search is issued to find the best-matching
        Google Places result within *search_radius_m* metres.  Matching
        considers both geographic proximity and name similarity.

        Enrichment fields written to ``store.attributes``:
            - ``google_place_id``
            - ``user_ratings_total``
            - ``business_status``
            - ``types``

        Top-level ``Store`` fields updated:
            - ``avg_rating``
            - ``price_level``

        Parameters
        ----------
        stores : list[Store]
            Stores to enrich (typically from the OSM loader).
        search_radius_m : float
            Search radius in metres for finding the matching Google place.

        Returns
        -------
        list[Store]
            The same store list, mutated in place with enrichment data.
        """
        total = len(stores)
        matched = 0

        for i, store in enumerate(stores):
            logger.debug(
                "Enriching store %d/%d: %s (%.5f, %.5f)",
                i + 1, total, store.name, store.lat, store.lon,
            )

            # Search nearby for candidates
            candidates = self.search_nearby(
                lat=store.lat,
                lon=store.lon,
                radius_m=search_radius_m,
                max_results=5,
            )

            if not candidates:
                logger.debug("  No Google Places candidates found.")
                continue

            # Score candidates by combined proximity + name similarity
            best_candidate: Optional[Store] = None
            best_score: float = -1.0

            for cand in candidates:
                dist_km = haversine_distance(
                    store.lat, store.lon, cand.lat, cand.lon,
                )
                dist_m = dist_km * 1000.0

                # Normalise distance to a 0-1 score (1 = very close)
                dist_score = max(0.0, 1.0 - dist_m / search_radius_m)

                name_score = _name_similarity(store.name, cand.name)

                # Weighted combination: name is more important than distance
                combined = 0.4 * dist_score + 0.6 * name_score

                if combined > best_score:
                    best_score = combined
                    best_candidate = cand

            # Accept match only if score exceeds a minimum threshold
            if best_candidate is None or best_score < _NAME_SIMILARITY_THRESHOLD:
                logger.debug(
                    "  No sufficiently close match (best score=%.2f).", best_score,
                )
                continue

            # Apply enrichment
            store.avg_rating = best_candidate.avg_rating
            store.price_level = best_candidate.price_level

            store.attributes["google_place_id"] = best_candidate.attributes.get(
                "google_place_id", ""
            )
            store.attributes["user_ratings_total"] = best_candidate.attributes.get(
                "user_ratings_total", 0
            )
            store.attributes["business_status"] = best_candidate.attributes.get(
                "business_status", ""
            )
            store.attributes["types"] = best_candidate.attributes.get("types", [])

            matched += 1
            logger.debug(
                "  Matched -> %s (score=%.2f, rating=%.1f)",
                best_candidate.name, best_score, best_candidate.avg_rating,
            )

        logger.info(
            "Enrichment complete: %d/%d stores matched a Google Place.",
            matched, total,
        )
        return stores
