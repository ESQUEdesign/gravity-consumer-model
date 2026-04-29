"""
Gravity Consumer Model
======================
Layered consumer behavior prediction engine built on the Huff Retail Gravity Model.

Layers:
    0. Huff Gravity Model (base spatial interaction)
    1. Competing Destinations + GWR (spatial corrections)
    2. Mixed Logit (heterogeneous discrete choice)
    3. Latent Class + Geodemographic + RFM/CLV (segmentation)
    4. HMM + Hawkes + Bayesian Update (temporal dynamics)
    5. XGBoost residuals + GNN (machine learning)
    6. Bayesian Model Averaging (ensemble)
"""

__version__ = "0.1.0"

from gravity.model import GravityModel

__all__ = ["GravityModel"]
