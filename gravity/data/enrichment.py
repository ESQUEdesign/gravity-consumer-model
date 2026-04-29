"""
Data enrichment pipeline.

Orchestrates all available data sources (store size estimation, Google Places,
Yelp Fusion, OSRM routing, BLS employment) into composite attractiveness
scores that the Huff model uses for store selection probabilities.

The pipeline is designed to degrade gracefully: every external data source
is optional, and failure to reach any API is logged but never blocks the
model from running.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from gravity.data.schema import (
    Store,
    ConsumerOrigin,
    build_distance_matrix,
    haversine_distance,
    stores_to_dataframe,
    origins_to_dataframe,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price-level multiplier lookup
# ---------------------------------------------------------------------------

_PRICE_MULTIPLIER = {
    1: 0.8,   # budget -- lower perceived attractiveness
    2: 1.0,   # moderate -- baseline
    3: 1.2,   # upscale -- slight premium
    4: 1.1,   # luxury -- slightly less broad appeal than upscale
}


# ---------------------------------------------------------------------------
# Composite attractiveness
# ---------------------------------------------------------------------------

def compute_composite_attractiveness(stores: list[Store]) -> list[Store]:
    """Compute a composite attractiveness score for each store.

    The composite replaces the raw square footage in ``store.square_footage``
    (which is the field the Huff model reads as attractiveness) with a
    value that blends size, quality ratings, price positioning, and review
    volume.

    Formula::

        composite = base_sqft * rating_multiplier * price_multiplier * review_boost

    Where:
        - ``base_sqft`` = estimated or actual square footage (minimum 100)
        - ``rating_multiplier`` = (rating / 3.0) if rating > 0 else 1.0
          (normalized so a 3.0 rating is neutral)
        - ``price_multiplier`` = lookup from {1: 0.8, 2: 1.0, 3: 1.2, 4: 1.1}
        - ``review_boost`` = 1.0 + min(0.3, log10(1 + review_count) * 0.1)
          (more reviews = higher confidence, capped at +30%)

    The original square footage is preserved in ``store.attributes["raw_sqft"]``
    and the component multipliers are stored in
    ``store.attributes["attractiveness_components"]``.

    Parameters
    ----------
    stores : list[Store]
        Stores to score.  Modified in place.

    Returns
    -------
    list[Store]
        The same list, with ``square_footage`` replaced by the composite
        and component details written to ``attributes``.
    """
    for store in stores:
        base_sqft = max(store.square_footage, 100.0)

        # Rating multiplier
        rating = store.avg_rating
        if rating > 0:
            rating_multiplier = rating / 3.0
        else:
            rating_multiplier = 1.0

        # Price multiplier
        price_multiplier = _PRICE_MULTIPLIER.get(store.price_level, 1.0)

        # Review boost: pull review_count from attributes if available
        review_count = store.attributes.get("user_ratings_total", 0)
        if not isinstance(review_count, (int, float)):
            review_count = 0
        review_boost = 1.0 + min(0.3, math.log10(1 + review_count) * 0.1)

        composite = base_sqft * rating_multiplier * price_multiplier * review_boost

        # Store raw sqft and components for auditability
        store.attributes["raw_sqft"] = base_sqft
        store.attributes["attractiveness_components"] = {
            "base_sqft": base_sqft,
            "rating_multiplier": round(rating_multiplier, 4),
            "price_multiplier": price_multiplier,
            "review_boost": round(review_boost, 4),
            "composite": round(composite, 2),
        }

        store.square_footage = round(composite, 2)

        logger.debug(
            "Store %s: sqft=%.0f * rating=%.3f * price=%.2f * review=%.3f = %.2f",
            store.store_id, base_sqft, rating_multiplier,
            price_multiplier, review_boost, composite,
        )

    logger.info(
        "Computed composite attractiveness for %d stores. "
        "Range: %.0f -- %.0f",
        len(stores),
        min(s.square_footage for s in stores) if stores else 0,
        max(s.square_footage for s in stores) if stores else 0,
    )

    return stores


# ---------------------------------------------------------------------------
# Store enrichment pipeline
# ---------------------------------------------------------------------------

def enrich_stores(
    stores: list[Store],
    google_api_key: Optional[str] = None,
    yelp_api_key: Optional[str] = None,
    use_osrm: bool = True,
) -> list[Store]:
    """Run the full store enrichment pipeline.

    Steps executed in order:

    1. **Square footage estimation** -- uses brand/category lookup tables
       to fill in missing store sizes (requires ``gravity.data.store_size``).
    2. **Google Places enrichment** -- if ``google_api_key`` is provided,
       enriches stores with ratings, review counts, and price levels.
    3. **Yelp Fusion enrichment** -- if ``yelp_api_key`` is provided,
       enriches stores with Yelp ratings, review counts, and price levels.
    4. **Composite attractiveness** -- blends all available signals into
       a single attractiveness score written to ``store.square_footage``.

    Each step is independent and degrades gracefully if its data source
    is unavailable.

    Parameters
    ----------
    stores : list[Store]
        Stores to enrich.  Modified in place.
    google_api_key : str or None
        Google Places API key.  If None, the Google step is skipped.
    yelp_api_key : str or None
        Yelp Fusion API key.  If None, the Yelp step is skipped.
    use_osrm : bool
        Unused in store enrichment (kept for API symmetry with
        ``get_distance_matrix``).

    Returns
    -------
    list[Store]
        The enriched stores (same objects, mutated in place).
    """
    if not stores:
        logger.warning("enrich_stores called with empty store list.")
        return stores

    # ------------------------------------------------------------------
    # Step 1: Estimate square footage for stores that lack it
    # ------------------------------------------------------------------
    try:
        from gravity.data.store_size import estimate_store_size

        before = sum(1 for s in stores if s.square_footage <= 0)
        stores_needing_sqft = [s for s in stores if s.square_footage <= 0]
        if stores_needing_sqft:
            estimate_store_size(stores_needing_sqft)  # mutates in place

        after = sum(1 for s in stores if s.square_footage <= 0)
        logger.info(
            "Step 1 (store size): estimated sqft for %d stores (%d still missing).",
            before - after, after,
        )
    except ImportError:
        logger.info(
            "Step 1 (store size): gravity.data.store_size not available, skipping."
        )
    except Exception as exc:
        logger.warning("Step 1 (store size) failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 2: Google Places enrichment
    # ------------------------------------------------------------------
    if google_api_key:
        try:
            from gravity.data.google_places import GooglePlacesLoader

            gp = GooglePlacesLoader(api_key=google_api_key)
            stores = gp.enrich_stores(stores, search_radius_m=100)
            logger.info("Step 2 (Google Places): enrichment complete.")
        except ImportError:
            logger.info(
                "Step 2 (Google Places): gravity.data.google_places not available."
            )
        except Exception as exc:
            logger.warning("Step 2 (Google Places) failed: %s", exc)
    else:
        logger.info("Step 2 (Google Places): no API key provided, skipping.")

    # ------------------------------------------------------------------
    # Step 3: Yelp Fusion enrichment
    # ------------------------------------------------------------------
    if yelp_api_key:
        try:
            from gravity.data.yelp import YelpLoader

            yelp = YelpLoader(api_key=yelp_api_key)
            stores = yelp.enrich_stores(stores)
            logger.info("Step 3 (Yelp): enrichment complete.")
        except ImportError:
            logger.info(
                "Step 3 (Yelp): gravity.data.yelp not available."
            )
        except Exception as exc:
            logger.warning("Step 3 (Yelp) failed: %s", exc)
    else:
        logger.info("Step 3 (Yelp): no API key provided, skipping.")

    # ------------------------------------------------------------------
    # Step 4: Compute composite attractiveness
    # ------------------------------------------------------------------
    stores = compute_composite_attractiveness(stores)
    logger.info("Step 4 (composite attractiveness): complete.")

    return stores


# ---------------------------------------------------------------------------
# Distance matrix with OSRM fallback
# ---------------------------------------------------------------------------

def get_distance_matrix(
    origins_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    use_osrm: bool = True,
) -> pd.DataFrame:
    """Build an origin x store distance matrix, preferring real driving distances.

    Tries OSRM (Open Source Routing Machine) first for real-world driving
    times.  If OSRM is unavailable or ``use_osrm=False``, falls back to
    haversine great-circle distances.

    When using OSRM, the returned values are converted from travel duration
    (seconds) to equivalent kilometres using a 50 km/h average speed
    assumption, so that the matrix is interchangeable with the haversine
    fallback (both in km).

    Parameters
    ----------
    origins_df : pd.DataFrame
        Origin locations with ``lat`` and ``lon`` columns.
    stores_df : pd.DataFrame
        Store locations with ``lat`` and ``lon`` columns.
    use_osrm : bool
        If True (default), attempt OSRM routing first.

    Returns
    -------
    pd.DataFrame
        Shape ``(len(origins_df), len(stores_df))`` with distances in
        kilometres.  Index and columns match the input DataFrames.
    """
    if not use_osrm:
        logger.info("OSRM disabled. Using haversine distance matrix.")
        return build_distance_matrix(origins_df, stores_df)

    try:
        from gravity.data.osrm import OSRMDistanceProvider

        osrm = OSRMDistanceProvider(profile="car")
        logger.info("Requesting OSRM driving duration matrix...")

        # Get duration matrix in seconds
        duration_matrix = osrm.distance_matrix(
            origins_df, stores_df, metric="duration"
        )

        # Convert seconds to equivalent km at 50 km/h
        # 50 km/h = 50/3600 km/s ~= 0.01389 km/s
        speed_km_per_sec = 50.0 / 3600.0
        km_matrix = duration_matrix * speed_km_per_sec

        logger.info(
            "OSRM distance matrix built: %d x %d. "
            "Mean equivalent distance: %.2f km",
            km_matrix.shape[0], km_matrix.shape[1],
            km_matrix.values[np.isfinite(km_matrix.values)].mean()
            if np.any(np.isfinite(km_matrix.values)) else 0.0,
        )

        # Replace any NaN (unreachable) cells with haversine fallback values
        if km_matrix.isna().any().any():
            haversine_fallback = build_distance_matrix(origins_df, stores_df)
            nan_count = km_matrix.isna().sum().sum()
            km_matrix = km_matrix.fillna(haversine_fallback)
            logger.warning(
                "Filled %d unreachable OSRM cells with haversine distances.",
                nan_count,
            )

        return km_matrix

    except ImportError:
        logger.info(
            "gravity.data.osrm not available. Using haversine distance matrix."
        )
        return build_distance_matrix(origins_df, stores_df)

    except Exception as exc:
        logger.warning(
            "OSRM distance matrix failed (%s). Falling back to haversine.", exc
        )
        return build_distance_matrix(origins_df, stores_df)


# ---------------------------------------------------------------------------
# Origin enrichment with BLS data
# ---------------------------------------------------------------------------

def enrich_origins(
    origins: list[ConsumerOrigin],
    state_fips: str,
    county_fips: str,
    year: int = 2023,
) -> list[ConsumerOrigin]:
    """Enrich consumer origins with BLS employment and wage data.

    Attempts to add county-level employment rate, average weekly wage,
    and retail establishment count to each origin's demographics dict.
    If the BLS API is unreachable, logs a warning and returns the
    origins unchanged.

    Parameters
    ----------
    origins : list[ConsumerOrigin]
        Origins to enrich (modified in place).
    state_fips : str
        2-digit state FIPS code.
    county_fips : str
        3-digit county FIPS code.
    year : int
        Data year (default 2023).

    Returns
    -------
    list[ConsumerOrigin]
        The same list, potentially enriched with BLS fields.
    """
    try:
        from gravity.data.bls import BLSLoader

        loader = BLSLoader()
        origins = loader.enrich_origins(
            origins, state_fips, county_fips, year=year,
        )
        logger.info(
            "BLS enrichment complete for %d origins (state=%s, county=%s).",
            len(origins), state_fips, county_fips,
        )
    except ImportError:
        logger.info("gravity.data.bls not available. Skipping BLS enrichment.")
    except Exception as exc:
        logger.warning(
            "BLS origin enrichment failed: %s. Origins returned unchanged.", exc
        )

    return origins
