"""
Tests for spatial analysis modules: trade area delineation, penetration,
fair-share index, distance matrix properties, and haversine calculation.

Validates geometric properties of distance matrices, known-distance
calculations, and trade area analysis outputs.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.fixtures.synthetic_data import generate_all, GROUND_TRUTH

from gravity.spatial.trade_area import TradeAreaAnalyzer
from gravity.data.schema import (
    build_distance_matrix,
    haversine_distance,
    stores_to_dataframe,
    origins_to_dataframe,
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth():
    """Generate the full synthetic dataset."""
    return generate_all(seed=42)


@pytest.fixture(scope="module")
def stores_df(synth):
    return stores_to_dataframe(synth["stores"])


@pytest.fixture(scope="module")
def origins_df(synth):
    return origins_to_dataframe(synth["origins"])


@pytest.fixture(scope="module")
def prob_matrix(synth):
    return synth["prob_matrix"]


# ---------------------------------------------------------------------------
# Trade Area Tests
# ---------------------------------------------------------------------------


class TestTradeAreaContours:
    """test_trade_area_contours: from_probabilities returns correct number of contour levels."""

    def test_trade_area_contours(self, prob_matrix, origins_df, stores_df):
        """from_probabilities should return one TradeArea per contour level."""
        analyzer = TradeAreaAnalyzer()
        levels = [0.50, 0.70, 0.90]
        store_id = stores_df.index[0]

        trade_areas = analyzer.from_probabilities(
            prob_matrix, origins_df, store_id, levels=levels
        )

        assert len(trade_areas) == len(levels), (
            f"Expected {len(levels)} trade areas, got {len(trade_areas)}"
        )

        # Verify they are sorted by ascending contour level
        contour_levels = [ta.contour_level for ta in trade_areas]
        assert contour_levels == sorted(contour_levels), (
            f"Trade areas should be sorted by contour level: {contour_levels}"
        )

        # Each trade area should have non-negative population
        for ta in trade_areas:
            assert ta.total_population >= 0, (
                f"Trade area at level {ta.contour_level} has negative population"
            )

        # Higher contour level should include more (or equal) population
        for i in range(len(trade_areas) - 1):
            assert trade_areas[i].total_population <= trade_areas[i + 1].total_population, (
                f"Trade area at {trade_areas[i].contour_level} has more "
                f"population ({trade_areas[i].total_population}) than at "
                f"{trade_areas[i + 1].contour_level} ({trade_areas[i + 1].total_population})"
            )


class TestTradeAreaPenetration:
    """test_trade_area_penetration: compute_penetration returns value in [0, 1]."""

    def test_trade_area_penetration(self, prob_matrix, origins_df, stores_df):
        """Market penetration should be a ratio in [0, 1]."""
        analyzer = TradeAreaAnalyzer()
        store_id = stores_df.index[0]

        pen = analyzer.compute_penetration(store_id, prob_matrix, origins_df)

        assert isinstance(pen, float), (
            f"compute_penetration should return a float, got {type(pen)}"
        )
        assert 0.0 <= pen <= 1.0, (
            f"Penetration should be in [0, 1], got {pen:.6f}"
        )
        assert pen > 0.0, (
            f"Penetration for a store with observed visits should be > 0"
        )


class TestFairShareIndex:
    """test_fair_share_index: fair_share_index returns value > 0."""

    def test_fair_share_index(self, prob_matrix, stores_df):
        """Fair share index should be a positive number."""
        analyzer = TradeAreaAnalyzer()
        store_id = stores_df.index[0]

        fsi = analyzer.fair_share_index(store_id, prob_matrix)

        assert isinstance(fsi, float), (
            f"fair_share_index should return a float, got {type(fsi)}"
        )
        assert fsi > 0.0, (
            f"Fair share index should be positive, got {fsi:.4f}"
        )


# ---------------------------------------------------------------------------
# Distance Matrix Tests
# ---------------------------------------------------------------------------


class TestDistanceMatrixSymmetric:
    """test_distance_matrix_symmetric: distances are non-negative and roughly symmetric."""

    def test_distance_matrix_symmetric(self, origins_df, stores_df):
        """All distances should be non-negative. When origins and stores
        overlap, d(i,j) should approximately equal d(j,i)."""
        dist = build_distance_matrix(origins_df, stores_df)

        # All distances should be non-negative
        assert (dist.values >= 0).all(), (
            "Distance matrix contains negative values"
        )

        # Test symmetry by computing the reverse matrix
        # (stores as origins, origins as stores)
        dist_reverse = build_distance_matrix(stores_df, origins_df)

        # dist has shape (n_origins, n_stores), dist_reverse has (n_stores, n_origins)
        # Compare: dist[origin_i, store_j] vs dist_reverse[store_j, origin_i]
        for origin_id in origins_df.index[:5]:  # spot check first 5
            for store_id in stores_df.index[:5]:
                d_forward = dist.loc[origin_id, store_id]
                d_reverse = dist_reverse.loc[store_id, origin_id]
                np.testing.assert_allclose(
                    d_forward,
                    d_reverse,
                    rtol=1e-10,
                    err_msg=(
                        f"Distance from {origin_id} to {store_id} ({d_forward:.6f}) "
                        f"should equal reverse ({d_reverse:.6f})"
                    ),
                )


class TestDistanceMatrixDiagonalZero:
    """test_distance_matrix_diagonal_zero: distance from a point to itself is 0."""

    def test_distance_matrix_diagonal_zero(self):
        """When origins and stores are co-located, diagonal entries should be 0."""
        # Create a small set where origins == stores (same coordinates)
        data = pd.DataFrame({
            "lat": [40.75, 40.76, 40.77],
            "lon": [-73.98, -73.97, -73.96],
            "population": [1000, 2000, 3000],
            "households": [400, 800, 1200],
            "median_income": [50000, 60000, 70000],
            "square_footage": [5000, 10000, 15000],
        }, index=["P1", "P2", "P3"])

        # Use the same data as both origins and stores
        dist = build_distance_matrix(data, data)

        # Diagonal should be exactly 0
        for idx in data.index:
            assert dist.loc[idx, idx] == pytest.approx(0.0, abs=1e-10), (
                f"Distance from {idx} to itself should be 0, "
                f"got {dist.loc[idx, idx]}"
            )


class TestHaversineKnownDistance:
    """test_haversine_known_distance: NYC to LA is approximately 3944 km."""

    def test_haversine_known_distance(self):
        """Haversine distance from New York to Los Angeles should be
        approximately 3944 km (within 50 km tolerance)."""
        # New York City: 40.7128 N, 74.0060 W
        nyc_lat, nyc_lon = 40.7128, -74.0060
        # Los Angeles: 34.0522 N, 118.2437 W
        la_lat, la_lon = 34.0522, -118.2437

        distance_km = haversine_distance(nyc_lat, nyc_lon, la_lat, la_lon)

        # Known great-circle distance is approximately 3944 km
        expected_km = 3944.0
        tolerance_km = 50.0

        assert abs(distance_km - expected_km) < tolerance_km, (
            f"NYC to LA distance should be ~{expected_km} km, "
            f"got {distance_km:.1f} km (tolerance: {tolerance_km} km)"
        )

        # Also verify basic properties
        assert distance_km > 0, "Distance should be positive"

        # Symmetry check
        reverse_km = haversine_distance(la_lat, la_lon, nyc_lat, nyc_lon)
        np.testing.assert_allclose(
            distance_km,
            reverse_km,
            rtol=1e-10,
            err_msg="Haversine should be symmetric: d(A,B) == d(B,A)",
        )
