"""
Pydantic data models for all entities in the gravity consumer model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------

class Store(BaseModel):
    """A retail location (existing or hypothetical)."""

    store_id: str
    name: Optional[str] = None
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    square_footage: float = Field(0.0, ge=0)
    avg_rating: float = Field(0.0, ge=0, le=5)
    price_level: int = Field(2, ge=1, le=4)
    parking_spaces: int = Field(0, ge=0)
    product_count: int = Field(0, ge=0)
    brand: Optional[str] = None
    category: Optional[str] = None
    attributes: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class ConsumerOrigin(BaseModel):
    """A demand origin point (census block group, zip code, etc.)."""

    origin_id: str
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    population: int = Field(0, ge=0)
    households: int = Field(0, ge=0)
    median_income: float = Field(0.0, ge=0)
    segment_code: Optional[str] = None
    demographics: dict = Field(default_factory=dict)


class Transaction(BaseModel):
    """A single purchase/visit transaction."""

    transaction_id: str
    consumer_id: str
    store_id: str
    timestamp: datetime
    amount: float = Field(0.0, ge=0)
    items: int = Field(1, ge=1)
    category: Optional[str] = None


class VisitEvent(BaseModel):
    """A real-time location visit event (mobile ping or app event)."""

    event_id: Optional[str] = None
    consumer_id: str
    store_id: str
    timestamp: datetime
    dwell_minutes: float = Field(0.0, ge=0)
    source: str = "mobile"  # mobile | app | pos


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class Prediction(BaseModel):
    """A single origin→store visit probability prediction."""

    origin_id: str
    store_id: str
    probability: float = Field(..., ge=0, le=1)
    segment: Optional[str] = None
    consumer_state: Optional[str] = None
    clv: Optional[float] = None
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None


class TradeArea(BaseModel):
    """A store's trade area definition."""

    store_id: str
    contour_level: float = Field(..., ge=0, le=1)
    total_population: int = 0
    total_households: int = 0
    penetration: float = 0.0
    fair_share_index: float = 0.0
    geojson: Optional[dict] = None


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def stores_to_dataframe(stores: list[Store]) -> pd.DataFrame:
    """Convert a list of Store models to a DataFrame."""
    records = [s.model_dump() for s in stores]
    return pd.DataFrame(records).set_index("store_id")


def origins_to_dataframe(origins: list[ConsumerOrigin]) -> pd.DataFrame:
    """Convert a list of ConsumerOrigin models to a DataFrame."""
    records = [o.model_dump() for o in origins]
    return pd.DataFrame(records).set_index("origin_id")


def transactions_to_dataframe(txns: list[Transaction]) -> pd.DataFrame:
    """Convert a list of Transaction models to a DataFrame."""
    records = [t.model_dump() for t in txns]
    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two points."""
    R = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def build_distance_matrix(
    origins: pd.DataFrame, stores: pd.DataFrame
) -> pd.DataFrame:
    """Build an origin×store distance matrix (km) using haversine."""
    o_lats = origins["lat"].values[:, np.newaxis]
    o_lons = origins["lon"].values[:, np.newaxis]
    s_lats = stores["lat"].values[np.newaxis, :]
    s_lons = stores["lon"].values[np.newaxis, :]

    R = 6371.0
    dlat = np.radians(s_lats - o_lats)
    dlon = np.radians(s_lons - o_lons)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(o_lats)) * np.cos(np.radians(s_lats)) * np.sin(dlon / 2) ** 2
    )
    dist = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return pd.DataFrame(dist, index=origins.index, columns=stores.index)
