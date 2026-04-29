"""
Census-Derived Psychographic Lifestyle Segmentation
====================================================
Classifies Census block groups into PRIZM-style lifestyle segments
using ACS-derived indicators (education, housing, occupation, commute,
vehicles, structure type) as free proxies for proprietary psychographic
data (Claritas PRIZM, Esri Tapestry, Experian Mosaic).

Each block group is assigned to one of 12 lifestyle segments via
nearest-prototype matching on normalised feature vectors.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns extracted from ConsumerOrigin.demographics
# ---------------------------------------------------------------------------

FEATURE_KEYS: list[str] = [
    "pct_bachelors_plus",
    "pct_graduate_degree",
    "pct_owner_occupied",
    "pct_renter_occupied",
    "pct_white_collar",
    "pct_service_occ",
    "pct_blue_collar",
    "pct_no_vehicle",
    "pct_2plus_vehicles",
    "pct_public_transit",
    "pct_work_from_home",
    "pct_drove_alone",
    "pct_single_family",
    "pct_multi_unit_5plus",
    "pct_mobile_home",
]

# Income and home value are on different scales and need separate handling
INCOME_KEY = "median_income"
HOME_VALUE_KEY = "median_home_value"

# Age-derived features computed from demographics dict
AGE_YOUNG_KEYS = ["age_under_18", "age_18_24"]
AGE_MIDDLE_KEYS = ["age_25_34", "age_35_44", "age_45_54"]
AGE_OLDER_KEYS = ["age_55_64", "age_65_plus"]

# ---------------------------------------------------------------------------
# Segment definitions — 12 lifestyle segments
# ---------------------------------------------------------------------------

SEGMENT_PROFILES: dict[str, dict] = {
    "UP": {
        "name": "Urban Professionals",
        "description": (
            "Well-educated, higher-income individuals in dense urban areas. "
            "Renters in multi-unit buildings who commute via transit or work "
            "from home. Drawn to specialty retail, organic grocers, and "
            "experiential brands."
        ),
        "color": "#2196F3",
        "prototype": {
            "pct_bachelors_plus": 0.55, "pct_graduate_degree": 0.25,
            "pct_owner_occupied": 0.25, "pct_renter_occupied": 0.75,
            "pct_white_collar": 0.65, "pct_service_occ": 0.15,
            "pct_blue_collar": 0.05, "pct_no_vehicle": 0.25,
            "pct_2plus_vehicles": 0.10, "pct_public_transit": 0.25,
            "pct_work_from_home": 0.15, "pct_drove_alone": 0.35,
            "pct_single_family": 0.15, "pct_multi_unit_5plus": 0.60,
            "pct_mobile_home": 0.0, "income_norm": 0.70,
            "home_value_norm": 0.60, "pct_young": 0.15,
            "pct_middle": 0.55, "pct_older": 0.15,
        },
        "consumer_behavior": {
            "retail_affinity": ["specialty", "organic", "online", "fitness"],
            "price_sensitivity": "low",
            "brand_loyalty": "moderate",
            "shopping_frequency": "high",
            "channel_preference": "omnichannel",
        },
    },
    "SF": {
        "name": "Suburban Families",
        "description": (
            "Middle-income homeowners in single-family homes with children. "
            "Multiple vehicles, drive-to-shop orientation. Value-conscious "
            "but willing to spend on family needs."
        ),
        "color": "#4CAF50",
        "prototype": {
            "pct_bachelors_plus": 0.30, "pct_graduate_degree": 0.10,
            "pct_owner_occupied": 0.75, "pct_renter_occupied": 0.25,
            "pct_white_collar": 0.40, "pct_service_occ": 0.20,
            "pct_blue_collar": 0.15, "pct_no_vehicle": 0.03,
            "pct_2plus_vehicles": 0.60, "pct_public_transit": 0.02,
            "pct_work_from_home": 0.08, "pct_drove_alone": 0.80,
            "pct_single_family": 0.85, "pct_multi_unit_5plus": 0.03,
            "pct_mobile_home": 0.02, "income_norm": 0.55,
            "home_value_norm": 0.45, "pct_young": 0.35,
            "pct_middle": 0.45, "pct_older": 0.10,
        },
        "consumer_behavior": {
            "retail_affinity": ["big-box", "grocery", "family apparel", "home improvement"],
            "price_sensitivity": "moderate",
            "brand_loyalty": "high",
            "shopping_frequency": "high",
            "channel_preference": "in-store",
        },
    },
    "AE": {
        "name": "Affluent Empty Nesters",
        "description": (
            "High-income, well-educated homeowners in established neighborhoods. "
            "Older adults with high home values and discretionary income. "
            "Quality-focused shoppers drawn to premium brands."
        ),
        "color": "#9C27B0",
        "prototype": {
            "pct_bachelors_plus": 0.45, "pct_graduate_degree": 0.20,
            "pct_owner_occupied": 0.85, "pct_renter_occupied": 0.15,
            "pct_white_collar": 0.50, "pct_service_occ": 0.15,
            "pct_blue_collar": 0.10, "pct_no_vehicle": 0.03,
            "pct_2plus_vehicles": 0.55, "pct_public_transit": 0.02,
            "pct_work_from_home": 0.12, "pct_drove_alone": 0.75,
            "pct_single_family": 0.90, "pct_multi_unit_5plus": 0.02,
            "pct_mobile_home": 0.01, "income_norm": 0.80,
            "home_value_norm": 0.75, "pct_young": 0.08,
            "pct_middle": 0.30, "pct_older": 0.50,
        },
        "consumer_behavior": {
            "retail_affinity": ["premium", "home furnishing", "wine/spirits", "travel"],
            "price_sensitivity": "low",
            "brand_loyalty": "high",
            "shopping_frequency": "moderate",
            "channel_preference": "in-store",
        },
    },
    "CT": {
        "name": "College Towns",
        "description": (
            "Young, highly-educated renters with modest incomes — students "
            "and recent graduates. Dense housing near universities. "
            "Budget-conscious, tech-savvy, trend-driven shoppers."
        ),
        "color": "#FF9800",
        "prototype": {
            "pct_bachelors_plus": 0.40, "pct_graduate_degree": 0.15,
            "pct_owner_occupied": 0.20, "pct_renter_occupied": 0.80,
            "pct_white_collar": 0.35, "pct_service_occ": 0.35,
            "pct_blue_collar": 0.05, "pct_no_vehicle": 0.20,
            "pct_2plus_vehicles": 0.10, "pct_public_transit": 0.10,
            "pct_work_from_home": 0.05, "pct_drove_alone": 0.50,
            "pct_single_family": 0.20, "pct_multi_unit_5plus": 0.50,
            "pct_mobile_home": 0.01, "income_norm": 0.20,
            "home_value_norm": 0.25, "pct_young": 0.55,
            "pct_middle": 0.30, "pct_older": 0.05,
        },
        "consumer_behavior": {
            "retail_affinity": ["fast fashion", "fast food", "electronics", "streaming"],
            "price_sensitivity": "high",
            "brand_loyalty": "low",
            "shopping_frequency": "high",
            "channel_preference": "online",
        },
    },
    "RT": {
        "name": "Rural Traditional",
        "description": (
            "Homeowners in low-density areas with blue-collar employment. "
            "Single-family homes and mobile homes, lower education levels. "
            "Value-driven shoppers loyal to familiar brands."
        ),
        "color": "#795548",
        "prototype": {
            "pct_bachelors_plus": 0.10, "pct_graduate_degree": 0.03,
            "pct_owner_occupied": 0.75, "pct_renter_occupied": 0.25,
            "pct_white_collar": 0.15, "pct_service_occ": 0.20,
            "pct_blue_collar": 0.40, "pct_no_vehicle": 0.05,
            "pct_2plus_vehicles": 0.50, "pct_public_transit": 0.01,
            "pct_work_from_home": 0.05, "pct_drove_alone": 0.85,
            "pct_single_family": 0.65, "pct_multi_unit_5plus": 0.02,
            "pct_mobile_home": 0.20, "income_norm": 0.30,
            "home_value_norm": 0.20, "pct_young": 0.20,
            "pct_middle": 0.40, "pct_older": 0.25,
        },
        "consumer_behavior": {
            "retail_affinity": ["dollar stores", "farm supply", "auto parts", "Walmart"],
            "price_sensitivity": "high",
            "brand_loyalty": "high",
            "shopping_frequency": "moderate",
            "channel_preference": "in-store",
        },
    },
    "UE": {
        "name": "Urban Economy",
        "description": (
            "Lower-income renters in dense urban areas. Service-sector "
            "workers relying on public transit. Price-sensitive shoppers "
            "concentrated at discount and convenience retailers."
        ),
        "color": "#F44336",
        "prototype": {
            "pct_bachelors_plus": 0.12, "pct_graduate_degree": 0.03,
            "pct_owner_occupied": 0.20, "pct_renter_occupied": 0.80,
            "pct_white_collar": 0.20, "pct_service_occ": 0.35,
            "pct_blue_collar": 0.20, "pct_no_vehicle": 0.30,
            "pct_2plus_vehicles": 0.08, "pct_public_transit": 0.25,
            "pct_work_from_home": 0.03, "pct_drove_alone": 0.35,
            "pct_single_family": 0.15, "pct_multi_unit_5plus": 0.55,
            "pct_mobile_home": 0.02, "income_norm": 0.15,
            "home_value_norm": 0.15, "pct_young": 0.30,
            "pct_middle": 0.45, "pct_older": 0.15,
        },
        "consumer_behavior": {
            "retail_affinity": ["discount", "convenience", "dollar stores", "check cashing"],
            "price_sensitivity": "very high",
            "brand_loyalty": "low",
            "shopping_frequency": "high",
            "channel_preference": "in-store",
        },
    },
    "SM": {
        "name": "Suburban Middle",
        "description": (
            "Moderate-income homeowners in single-family suburbs. "
            "Sales and office workers who drive to work. Mainstream "
            "shoppers who value convenience and brand recognition."
        ),
        "color": "#607D8B",
        "prototype": {
            "pct_bachelors_plus": 0.20, "pct_graduate_degree": 0.06,
            "pct_owner_occupied": 0.65, "pct_renter_occupied": 0.35,
            "pct_white_collar": 0.30, "pct_service_occ": 0.25,
            "pct_blue_collar": 0.20, "pct_no_vehicle": 0.05,
            "pct_2plus_vehicles": 0.45, "pct_public_transit": 0.03,
            "pct_work_from_home": 0.06, "pct_drove_alone": 0.82,
            "pct_single_family": 0.70, "pct_multi_unit_5plus": 0.08,
            "pct_mobile_home": 0.05, "income_norm": 0.45,
            "home_value_norm": 0.35, "pct_young": 0.25,
            "pct_middle": 0.45, "pct_older": 0.18,
        },
        "consumer_behavior": {
            "retail_affinity": ["grocery chains", "Target", "mid-tier apparel", "restaurants"],
            "price_sensitivity": "moderate",
            "brand_loyalty": "moderate",
            "shopping_frequency": "high",
            "channel_preference": "in-store",
        },
    },
    "MU": {
        "name": "Multicultural Urban",
        "description": (
            "Ethnically diverse urban neighborhoods with mixed housing types. "
            "Transit-oriented, service and white-collar jobs. Attracted to "
            "ethnic grocers, specialty food, and community retail."
        ),
        "color": "#E91E63",
        "prototype": {
            "pct_bachelors_plus": 0.25, "pct_graduate_degree": 0.08,
            "pct_owner_occupied": 0.30, "pct_renter_occupied": 0.70,
            "pct_white_collar": 0.30, "pct_service_occ": 0.30,
            "pct_blue_collar": 0.15, "pct_no_vehicle": 0.18,
            "pct_2plus_vehicles": 0.15, "pct_public_transit": 0.20,
            "pct_work_from_home": 0.05, "pct_drove_alone": 0.45,
            "pct_single_family": 0.30, "pct_multi_unit_5plus": 0.40,
            "pct_mobile_home": 0.01, "income_norm": 0.35,
            "home_value_norm": 0.30, "pct_young": 0.30,
            "pct_middle": 0.45, "pct_older": 0.12,
        },
        "consumer_behavior": {
            "retail_affinity": ["ethnic grocery", "specialty food", "discount apparel", "wireless"],
            "price_sensitivity": "moderate",
            "brand_loyalty": "moderate",
            "shopping_frequency": "high",
            "channel_preference": "in-store",
        },
    },
    "YM": {
        "name": "Young & Mobile",
        "description": (
            "25-34 year-old renters in newer developments. Moderate incomes, "
            "career-focused. Heavy online shoppers drawn to trendy brands "
            "and subscription services."
        ),
        "color": "#00BCD4",
        "prototype": {
            "pct_bachelors_plus": 0.35, "pct_graduate_degree": 0.10,
            "pct_owner_occupied": 0.30, "pct_renter_occupied": 0.70,
            "pct_white_collar": 0.40, "pct_service_occ": 0.25,
            "pct_blue_collar": 0.10, "pct_no_vehicle": 0.10,
            "pct_2plus_vehicles": 0.25, "pct_public_transit": 0.08,
            "pct_work_from_home": 0.12, "pct_drove_alone": 0.65,
            "pct_single_family": 0.40, "pct_multi_unit_5plus": 0.35,
            "pct_mobile_home": 0.02, "income_norm": 0.45,
            "home_value_norm": 0.35, "pct_young": 0.15,
            "pct_middle": 0.60, "pct_older": 0.08,
        },
        "consumer_behavior": {
            "retail_affinity": ["fast casual", "athleisure", "subscription boxes", "coworking"],
            "price_sensitivity": "moderate",
            "brand_loyalty": "low",
            "shopping_frequency": "high",
            "channel_preference": "online",
        },
    },
    "WE": {
        "name": "Working Exurbs",
        "description": (
            "Blue-collar and skilled-trade workers in outer suburbs and "
            "exurban areas. Homeowners with multiple vehicles, moderate "
            "incomes. Practical shoppers focused on hardware, auto, and "
            "home improvement."
        ),
        "color": "#FF5722",
        "prototype": {
            "pct_bachelors_plus": 0.12, "pct_graduate_degree": 0.03,
            "pct_owner_occupied": 0.70, "pct_renter_occupied": 0.30,
            "pct_white_collar": 0.20, "pct_service_occ": 0.18,
            "pct_blue_collar": 0.35, "pct_no_vehicle": 0.03,
            "pct_2plus_vehicles": 0.55, "pct_public_transit": 0.01,
            "pct_work_from_home": 0.04, "pct_drove_alone": 0.85,
            "pct_single_family": 0.80, "pct_multi_unit_5plus": 0.02,
            "pct_mobile_home": 0.08, "income_norm": 0.40,
            "home_value_norm": 0.30, "pct_young": 0.25,
            "pct_middle": 0.45, "pct_older": 0.18,
        },
        "consumer_behavior": {
            "retail_affinity": ["home improvement", "auto parts", "sporting goods", "warehouse clubs"],
            "price_sensitivity": "moderate",
            "brand_loyalty": "high",
            "shopping_frequency": "moderate",
            "channel_preference": "in-store",
        },
    },
    "GC": {
        "name": "Graying Communities",
        "description": (
            "Pre-retirement and early-retirement homeowners (55-64) in "
            "established neighborhoods with older housing stock. Moderate "
            "incomes, loyal to familiar stores and brands."
        ),
        "color": "#9E9E9E",
        "prototype": {
            "pct_bachelors_plus": 0.18, "pct_graduate_degree": 0.06,
            "pct_owner_occupied": 0.75, "pct_renter_occupied": 0.25,
            "pct_white_collar": 0.28, "pct_service_occ": 0.22,
            "pct_blue_collar": 0.22, "pct_no_vehicle": 0.06,
            "pct_2plus_vehicles": 0.40, "pct_public_transit": 0.03,
            "pct_work_from_home": 0.05, "pct_drove_alone": 0.78,
            "pct_single_family": 0.75, "pct_multi_unit_5plus": 0.05,
            "pct_mobile_home": 0.08, "income_norm": 0.40,
            "home_value_norm": 0.35, "pct_young": 0.10,
            "pct_middle": 0.30, "pct_older": 0.45,
        },
        "consumer_behavior": {
            "retail_affinity": ["pharmacy", "grocery", "home maintenance", "healthcare"],
            "price_sensitivity": "moderate",
            "brand_loyalty": "very high",
            "shopping_frequency": "moderate",
            "channel_preference": "in-store",
        },
    },
    "RL": {
        "name": "Retirement & Leisure",
        "description": (
            "Retirees 65+ in owner-occupied homes with low workforce "
            "participation. Fixed incomes but often substantial home equity. "
            "Pharmacy, healthcare, and dining-oriented spending."
        ),
        "color": "#CDDC39",
        "prototype": {
            "pct_bachelors_plus": 0.20, "pct_graduate_degree": 0.08,
            "pct_owner_occupied": 0.80, "pct_renter_occupied": 0.20,
            "pct_white_collar": 0.25, "pct_service_occ": 0.20,
            "pct_blue_collar": 0.10, "pct_no_vehicle": 0.08,
            "pct_2plus_vehicles": 0.35, "pct_public_transit": 0.02,
            "pct_work_from_home": 0.05, "pct_drove_alone": 0.70,
            "pct_single_family": 0.75, "pct_multi_unit_5plus": 0.08,
            "pct_mobile_home": 0.10, "income_norm": 0.35,
            "home_value_norm": 0.40, "pct_young": 0.05,
            "pct_middle": 0.15, "pct_older": 0.65,
        },
        "consumer_behavior": {
            "retail_affinity": ["pharmacy", "healthcare", "casual dining", "grocery"],
            "price_sensitivity": "moderate",
            "brand_loyalty": "very high",
            "shopping_frequency": "moderate",
            "channel_preference": "in-store",
        },
    },
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class CensusPsychographicClassifier:
    """Classify Census block groups into lifestyle segments.

    Uses ACS-derived indicators (education, housing, occupation, commute,
    vehicles, structure type) as free proxies for PRIZM-style psychographic
    segmentation.

    Parameters
    ----------
    min_population : int
        Minimum block-group population to classify. Below this threshold
        the segment is set to ``None`` (data too sparse / suppressed).
    """

    def __init__(self, min_population: int = 50) -> None:
        self.min_population = min_population
        self._prototype_matrix: Optional[np.ndarray] = None
        self._segment_codes: list[str] = list(SEGMENT_PROFILES.keys())
        self._feature_names: list[str] = list(
            SEGMENT_PROFILES[self._segment_codes[0]]["prototype"].keys()
        )
        self._build_prototype_matrix()

    def _build_prototype_matrix(self) -> None:
        """Build a (n_segments, n_features) matrix from prototypes."""
        rows = []
        for code in self._segment_codes:
            proto = SEGMENT_PROFILES[code]["prototype"]
            rows.append([proto[f] for f in self._feature_names])
        self._prototype_matrix = np.array(rows, dtype=np.float64)

    def _extract_features(self, origins_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and normalise feature vectors from origins DataFrame."""
        records = []
        for idx, row in origins_df.iterrows():
            demo = row.get("demographics", {})
            if not isinstance(demo, dict):
                demo = {}

            feat: dict[str, float] = {}

            # Percentage features from demographics
            for key in FEATURE_KEYS:
                feat[key] = float(demo.get(key, 0.0))

            # Normalised income (0-1 scale, capped at $200k)
            income = float(row.get("median_income", 0.0))
            feat["income_norm"] = min(income / 200_000.0, 1.0)

            # Normalised home value (0-1 scale, capped at $750k)
            hv = float(demo.get("median_home_value", 0.0))
            feat["home_value_norm"] = min(hv / 750_000.0, 1.0)

            # Age distribution ratios
            pop = float(row.get("population", 0))
            if pop > 0:
                young = sum(demo.get(k, 0) for k in AGE_YOUNG_KEYS)
                middle = sum(demo.get(k, 0) for k in AGE_MIDDLE_KEYS)
                older = sum(demo.get(k, 0) for k in AGE_OLDER_KEYS)
                total_age = young + middle + older
                if total_age > 0:
                    feat["pct_young"] = young / total_age
                    feat["pct_middle"] = middle / total_age
                    feat["pct_older"] = older / total_age
                else:
                    feat["pct_young"] = 0.33
                    feat["pct_middle"] = 0.34
                    feat["pct_older"] = 0.33
            else:
                feat["pct_young"] = 0.33
                feat["pct_middle"] = 0.34
                feat["pct_older"] = 0.33

            records.append(feat)

        feat_df = pd.DataFrame(records, index=origins_df.index)
        # Ensure column order matches prototype features
        feat_df = feat_df.reindex(columns=self._feature_names, fill_value=0.0)
        return feat_df

    def classify(self, origins_df: pd.DataFrame) -> pd.DataFrame:
        """Assign each block group to its nearest lifestyle segment.

        Parameters
        ----------
        origins_df : pd.DataFrame
            Origins DataFrame with ``population``, ``median_income``,
            and ``demographics`` dict column containing lifestyle indicators.

        Returns
        -------
        pd.DataFrame
            Copy of origins_df with ``segment_code``, ``segment_name``,
            and ``segment_description`` columns added.
        """
        result = origins_df.copy()
        feat_df = self._extract_features(origins_df)
        feat_matrix = feat_df.values.astype(np.float64)

        # Compute Euclidean distance from each block group to each prototype
        # Shape: (n_origins, n_segments)
        dists = np.sqrt(
            ((feat_matrix[:, np.newaxis, :] - self._prototype_matrix[np.newaxis, :, :]) ** 2)
            .sum(axis=2)
        )
        assignments = np.argmin(dists, axis=1)

        codes = []
        names = []
        descriptions = []
        for i, row_idx in enumerate(origins_df.index):
            pop = origins_df.loc[row_idx, "population"] if "population" in origins_df.columns else 0
            if pop < self.min_population:
                codes.append(None)
                names.append(None)
                descriptions.append(None)
            else:
                seg_code = self._segment_codes[assignments[i]]
                profile = SEGMENT_PROFILES[seg_code]
                codes.append(seg_code)
                names.append(profile["name"])
                descriptions.append(profile["description"])

        result["segment_code"] = codes
        result["segment_name"] = names
        result["segment_description"] = descriptions

        n_classified = sum(1 for c in codes if c is not None)
        logger.info(
            "Classified %d / %d block groups into %d distinct segments",
            n_classified, len(origins_df),
            len(set(c for c in codes if c is not None)),
        )
        return result

    @staticmethod
    def get_segment_profile(segment_code: str) -> Optional[dict]:
        """Return the full profile for a segment code."""
        return SEGMENT_PROFILES.get(segment_code)

    @staticmethod
    def list_segments() -> pd.DataFrame:
        """Return a summary table of all available segments."""
        rows = []
        for code, profile in SEGMENT_PROFILES.items():
            rows.append({
                "code": code,
                "name": profile["name"],
                "description": profile["description"][:80] + "...",
                "color": profile["color"],
            })
        return pd.DataFrame(rows)

    def segment_summary(self, classified_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate statistics by segment from a classified DataFrame."""
        if "segment_code" not in classified_df.columns:
            return pd.DataFrame()

        valid = classified_df[classified_df["segment_code"].notna()].copy()
        if valid.empty:
            return pd.DataFrame()

        groups = valid.groupby("segment_code")
        summary = groups.agg(
            block_groups=("population", "count"),
            total_population=("population", "sum"),
            avg_income=("median_income", "mean"),
        ).reset_index()

        summary["pct_of_population"] = (
            summary["total_population"] / summary["total_population"].sum()
        )

        # Add segment names
        summary["segment_name"] = summary["segment_code"].map(
            {k: v["name"] for k, v in SEGMENT_PROFILES.items()}
        )
        summary["color"] = summary["segment_code"].map(
            {k: v["color"] for k, v in SEGMENT_PROFILES.items()}
        )

        return summary.sort_values("total_population", ascending=False).reset_index(drop=True)
