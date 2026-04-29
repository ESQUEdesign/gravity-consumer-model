"""
Yelp Fusion API v3 data loader.

Enriches ``Store`` models with ratings, reviews, price levels, and categories
from the Yelp Fusion API.  Also discovers new business locations via the
Business Search endpoint.

Endpoints used (Yelp Fusion API v3):
    - Business Search:  GET https://api.yelp.com/v3/businesses/search
    - Business Details: GET https://api.yelp.com/v3/businesses/{id}

Authentication is via the ``Authorization: Bearer {api_key}`` header.

Free tier: 5 000 API calls per day.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from difflib import SequenceMatcher
from typing import Any, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

from gravity.data.schema import Store, haversine_distance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUSINESS_SEARCH_URL = "https://api.yelp.com/v3/businesses/search"
_BUSINESS_DETAILS_URL = "https://api.yelp.com/v3/businesses/{id}"

# Default request delay (seconds) — 5 req/s to stay within free tier limits
_DEFAULT_DELAY = 0.2

# Name similarity threshold for matching enrichment candidates
_NAME_SIMILARITY_THRESHOLD = 0.5

# Yelp category alias -> canonical project category mapping
_YELP_CATEGORY_MAP: dict[str, str] = {
    # Grocery & food retail
    "grocery": "grocery",
    "organic_stores": "grocery",
    "healthmarkets": "grocery",
    "intlgrocery": "grocery",
    "gourmet": "grocery",
    "farmersmarket": "grocery",
    # Restaurants
    "restaurants": "restaurant",
    "newamerican": "restaurant",
    "tradamerican": "restaurant",
    "italian": "restaurant",
    "mexican": "restaurant",
    "chinese": "restaurant",
    "japanese": "restaurant",
    "thai": "restaurant",
    "indian": "restaurant",
    "korean": "restaurant",
    "vietnamese": "restaurant",
    "mediterranean": "restaurant",
    "french": "restaurant",
    "seafood": "restaurant",
    "steak": "restaurant",
    "sushi": "restaurant",
    "pizza": "restaurant",
    "burgers": "restaurant",
    "sandwiches": "restaurant",
    "delis": "restaurant",
    # Cafes & coffee
    "coffee": "cafe",
    "coffeeroasteries": "cafe",
    "coffeeshops": "cafe",
    "tea": "cafe",
    "bubbletea": "cafe",
    "bakeries": "cafe",
    "cafes": "cafe",
    "juicebars": "cafe",
    # Fast food & quick service
    "hotdog": "fast_food",
    "fast food": "fast_food",
    "foodtrucks": "fast_food",
    "chickenshop": "fast_food",
    # Bars & nightlife
    "bars": "bar",
    "sportsbars": "bar",
    "wine_bars": "bar",
    "cocktailbars": "bar",
    "pubs": "bar",
    "breweries": "bar",
    "beerbar": "bar",
    # Convenience
    "convenience": "convenience",
    "gas_stations": "convenience",
    "drugstores": "pharmacy",
    "pharmacy": "pharmacy",
    # Shopping
    "shopping": "retail",
    "departmentstores": "retail",
    "fashion": "retail",
    "electronics": "retail",
    "hardware": "retail",
    "bookstores": "retail",
    # Services
    "beautysvc": "service",
    "hairstylists": "service",
    "barbers": "service",
    "laundry_services": "service",
    "dryclean": "service",
    "autorepair": "service",
    # Fitness
    "gyms": "fitness",
    "yoga": "fitness",
    "pilates": "fitness",
    "martialarts": "fitness",
    # Entertainment
    "entertainment": "entertainment",
    "movietheaters": "entertainment",
    "museums": "entertainment",
    "arcades": "entertainment",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yelp_price_to_int(price_str: Optional[str]) -> int:
    """Convert a Yelp price string to an integer 1-4.

    Yelp uses "$", "$$", "$$$", or "$$$$".
    Returns 2 (moderate) when the value is missing or unrecognised.
    """
    if not price_str:
        return 2
    count = price_str.count("$")
    if 1 <= count <= 4:
        return count
    return 2


def _yelp_category_to_canonical(categories: list[dict[str, str]]) -> Optional[str]:
    """Map Yelp categories to a canonical project category.

    Parameters
    ----------
    categories : list[dict]
        Yelp category objects, each with ``alias`` and ``title`` keys.

    Returns
    -------
    str or None
        The first matching canonical category, or None if no match.
    """
    for cat in categories:
        alias = cat.get("alias", "").lower()
        if alias in _YELP_CATEGORY_MAP:
            return _YELP_CATEGORY_MAP[alias]
        # Try partial matching for parent categories
        for key, canonical in _YELP_CATEGORY_MAP.items():
            if key in alias or alias in key:
                return canonical
    return None


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


def _stable_id(name: Optional[str], lat: float, lon: float) -> str:
    """Generate a deterministic store id from name + coordinates."""
    raw = f"{name or 'unknown'}_{lat:.6f}_{lon:.6f}"
    return "yelp_" + hashlib.md5(raw.encode()).hexdigest()[:12]


def _business_to_store(biz: dict[str, Any]) -> Store:
    """Convert a single Yelp business API result dict to a ``Store``."""
    coords = biz.get("coordinates", {})
    lat = float(coords.get("latitude", 0.0))
    lon = float(coords.get("longitude", 0.0))

    name = biz.get("name")
    yelp_id = biz.get("id", "")
    store_id = yelp_id if yelp_id else _stable_id(name, lat, lon)

    rating = float(biz.get("rating", 0.0))
    price_level = _yelp_price_to_int(biz.get("price"))
    review_count = int(biz.get("review_count", 0))
    is_closed = biz.get("is_closed", False)

    categories = biz.get("categories", [])
    canonical_category = _yelp_category_to_canonical(categories)
    category = canonical_category or (
        categories[0].get("alias", "other") if categories else "other"
    )

    phone = biz.get("phone", "")
    url = biz.get("url", "")
    transactions = biz.get("transactions", [])

    return Store(
        store_id=store_id,
        name=name,
        lat=lat,
        lon=lon,
        avg_rating=rating,
        price_level=price_level,
        category=category,
        attributes={
            "yelp_id": yelp_id,
            "review_count": review_count,
            "is_closed": is_closed,
            "phone": phone,
            "url": url,
            "transactions": transactions,
            "yelp_categories": [
                {"alias": c.get("alias", ""), "title": c.get("title", "")}
                for c in categories
            ],
        },
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class YelpLoader:
    """Load and enrich retail data using the Yelp Fusion API v3.

    Parameters
    ----------
    api_key : str or None
        Yelp Fusion API key.  If *None*, reads from the ``YELP_API_KEY``
        environment variable.
    request_delay : float
        Seconds to wait between successive API requests.  The default
        (0.2 s) keeps throughput at ~5 req/s, well within the free-tier
        daily limit of 5 000 calls.

    Raises
    ------
    ValueError
        If no API key is found.

    Examples
    --------
    >>> loader = YelpLoader()
    >>> stores = loader.search_businesses(40.7580, -73.9855, radius_m=500)
    >>> enriched = loader.enrich_stores(stores, search_radius_m=200)
    """

    def __init__(
        self,
        api_key: str | None = None,
        request_delay: float = _DEFAULT_DELAY,
    ) -> None:
        if requests is None:
            raise ImportError(
                "The 'requests' package is required for YelpLoader. "
                "Install it with: pip install requests"
            )
        self.api_key = api_key or os.environ.get("YELP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Yelp Fusion API key is required.  Either pass api_key= "
                "or set the YELP_API_KEY environment variable."
            )
        self.request_delay = request_delay
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep for the configured delay to respect rate limits."""
        if self.request_delay > 0:
            time.sleep(self.request_delay)

    def _get(self, url: str, params: Optional[dict[str, Any]] = None) -> dict:
        """Execute a GET request with standard headers and error handling.

        Returns the parsed JSON response body or an empty dict on failure.
        """
        self._throttle()
        try:
            resp = self._session.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            logger.error("Network error during GET %s: %s", url, exc)
            return {}

        if resp.status_code == 401:
            logger.error(
                "Yelp API returned 401 Unauthorized -- check that your API "
                "key is valid and has not been revoked."
            )
            return {}
        if resp.status_code == 429:
            logger.warning(
                "Yelp API rate limit exceeded (429).  Backing off 2 seconds."
            )
            time.sleep(2.0)
            return {}
        if resp.status_code != 200:
            logger.error(
                "Yelp API error %d: %s", resp.status_code, resp.text[:500]
            )
            return {}

        return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_businesses(
        self,
        lat: float,
        lon: float,
        radius_m: int = 1000,
        categories: Optional[str] = None,
        limit: int = 50,
    ) -> list[Store]:
        """Discover businesses via Yelp Business Search.

        Parameters
        ----------
        lat, lon : float
            Centre point of the search circle.
        radius_m : int
            Search radius in metres (max 40 000).
        categories : str or None
            Comma-separated list of Yelp category aliases to filter by,
            e.g. ``"grocery,restaurants"``.
        limit : int
            Maximum number of results to return (max 50).

        Returns
        -------
        list[Store]
            Discovered businesses as ``Store`` models.
        """
        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "radius": min(int(radius_m), 40000),
            "limit": min(int(limit), 50),
            "sort_by": "distance",
        }
        if categories:
            params["categories"] = categories

        data = self._get(_BUSINESS_SEARCH_URL, params=params)
        businesses = data.get("businesses", [])

        stores = [_business_to_store(biz) for biz in businesses]
        logger.info(
            "Yelp search at (%.5f, %.5f) r=%dm returned %d results",
            lat, lon, radius_m, len(stores),
        )
        return stores

    def get_business_details(self, yelp_id: str) -> dict[str, Any]:
        """Fetch full details for a single business.

        Parameters
        ----------
        yelp_id : str
            The Yelp business ID, e.g. ``"north-india-restaurant-san-francisco"``.

        Returns
        -------
        dict
            Raw detail dict including ``hours``, ``photos``,
            ``special_hours``, ``messaging``, and more.  Returns an empty
            dict on any error.
        """
        url = _BUSINESS_DETAILS_URL.format(id=yelp_id)
        return self._get(url)

    def enrich_stores(
        self,
        stores: list[Store],
        search_radius_m: int = 200,
    ) -> list[Store]:
        """Enrich existing stores with Yelp quality and review data.

        For each store, a Business Search is issued to find the
        best-matching Yelp result within *search_radius_m* metres.
        Matching considers both geographic proximity (40 % weight) and
        name similarity (60 % weight), mirroring the logic used in the
        Google Places enricher.

        Enrichment fields written to ``store.attributes``:
            - ``yelp_id``
            - ``review_count``
            - ``yelp_url``
            - ``yelp_categories``
            - ``is_closed``

        Top-level ``Store`` fields updated:
            - ``avg_rating``
            - ``price_level``

        Parameters
        ----------
        stores : list[Store]
            Stores to enrich (typically from the OSM or census loader).
        search_radius_m : int
            Search radius in metres for finding the matching Yelp
            business.

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
            candidates = self.search_businesses(
                lat=store.lat,
                lon=store.lon,
                radius_m=search_radius_m,
                limit=5,
            )

            if not candidates:
                logger.debug("  No Yelp candidates found.")
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

                # Weighted combination: 60% name, 40% distance
                combined = 0.6 * name_score + 0.4 * dist_score

                if combined > best_score:
                    best_score = combined
                    best_candidate = cand

            # Accept match only if score exceeds a minimum threshold
            if best_candidate is None or best_score < _NAME_SIMILARITY_THRESHOLD:
                logger.debug(
                    "  No sufficiently close match (best score=%.2f).",
                    best_score,
                )
                continue

            # Apply enrichment -- top-level fields
            store.avg_rating = best_candidate.avg_rating
            store.price_level = best_candidate.price_level

            # Apply enrichment -- attributes
            store.attributes["yelp_id"] = best_candidate.attributes.get(
                "yelp_id", ""
            )
            store.attributes["review_count"] = best_candidate.attributes.get(
                "review_count", 0
            )
            store.attributes["yelp_url"] = best_candidate.attributes.get(
                "url", ""
            )
            store.attributes["yelp_categories"] = best_candidate.attributes.get(
                "yelp_categories", []
            )
            store.attributes["is_closed"] = best_candidate.attributes.get(
                "is_closed", False
            )

            matched += 1
            logger.debug(
                "  Matched -> %s (score=%.2f, rating=%.1f, price=%d)",
                best_candidate.name,
                best_score,
                best_candidate.avg_rating,
                best_candidate.price_level,
            )

        logger.info(
            "Yelp enrichment complete: %d/%d stores matched.",
            matched, total,
        )
        return stores
