# Gravity Consumer Model — Implementation Plan

## Overview
Python analytical library implementing a layered consumer behavior prediction engine built on the Huff Retail Gravity Model, enhanced with Mixed Logit discrete choice, Latent Class segmentation, Bayesian real-time updating, competing destinations correction, HMM state transitions, and ML residual learning.

## Project Structure

```
gravity-consumer-model/
├── setup.py
├── requirements.txt
├── README.md
├── config/
│   └── default_config.yaml          # Default parameters, decay rates, segments
├── gravity/
│   ├── __init__.py                   # Package exports
│   ├── core/
│   │   ├── __init__.py
│   │   ├── huff.py                   # Layer 0: Huff Gravity Model (base)
│   │   ├── competing_destinations.py # Fotheringham competing destinations correction
│   │   ├── gwr.py                    # Geographically Weighted Regression (spatial λ)
│   │   └── mixed_logit.py            # Mixed Logit discrete choice model
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── latent_class.py           # Latent Class Analysis (behavioral segments)
│   │   ├── geodemographic.py         # PRIZM/Mosaic-style segment integration
│   │   ├── rfm.py                    # Recency-Frequency-Monetary scoring
│   │   └── clv.py                    # Customer Lifetime Value estimation
│   ├── temporal/
│   │   ├── __init__.py
│   │   ├── hmm.py                    # Hidden Markov Model (consumer state transitions)
│   │   ├── hawkes.py                 # Hawkes process (visit momentum / self-excitation)
│   │   └── bayesian_update.py        # Real-time Bayesian posterior updating
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── residual_boost.py         # XGBoost/LightGBM on structural model residuals
│   │   └── graph_network.py          # GNN for consumer-store interaction patterns
│   ├── spatial/
│   │   ├── __init__.py
│   │   ├── trade_area.py             # Trade area delineation and mapping
│   │   ├── spatial_econometrics.py   # Spatial lag / spatial error models
│   │   └── isochrone.py              # Drive-time / walk-time isochrone generation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── census.py                 # Census/ACS data loader
│   │   ├── osm.py                    # OpenStreetMap POI loader
│   │   ├── safegraph.py              # SafeGraph/Placer.ai foot traffic loader
│   │   ├── transactions.py           # POS/transaction data loader
│   │   └── schema.py                 # Data validation schemas (pydantic)
│   ├── ensemble/
│   │   ├── __init__.py
│   │   └── model_averaging.py        # Bayesian Model Averaging / weighted ensemble
│   └── reporting/
│       ├── __init__.py
│       ├── trade_area_report.py      # Trade area analysis reports
│       ├── consumer_profile.py       # Consumer segment profiles
│       ├── scenario.py               # What-if scenario simulator
│       └── visualize.py              # Maps, charts, heatmaps (folium + matplotlib)
├── tests/
│   ├── test_huff.py
│   ├── test_mixed_logit.py
│   ├── test_segmentation.py
│   ├── test_temporal.py
│   ├── test_ensemble.py
│   └── fixtures/
│       └── sample_data.json          # Synthetic test data
└── examples/
    ├── quickstart.py                 # Minimal working example
    ├── full_pipeline.py              # End-to-end analysis pipeline
    └── scenario_analysis.py          # Competitive scenario modeling
```

## Implementation Phases

### Phase 1: Core Spatial Engine (Huff + Enhancements)
**Files:** `huff.py`, `competing_destinations.py`, `gwr.py`, `trade_area.py`, `schema.py`

1. **`schema.py`** — Pydantic models for Store, Consumer Origin, TradeArea, Segment
2. **`huff.py`** — Classic Huff model
   - Probability calculation: P(Cᵢ → Sⱼ) = (Aⱼ^α · Dᵢⱼ^-λ) / Σₖ(Aₖ^α · Dᵢₖ^-λ)
   - Configurable attractiveness function (size, reviews, brand, composite)
   - Distance calculation via haversine or network/driving distance
   - Calibration method: observed vs predicted visit shares (MLE)
3. **`competing_destinations.py`** — Fotheringham's agglomeration/competition correction
   - Adds clustering factor: how nearby competitors affect draw
4. **`gwr.py`** — Spatially varying parameters
   - Local calibration of λ (distance decay) and α (attractiveness weight)
   - Gaussian or bisquare spatial kernel weighting
5. **`trade_area.py`** — Trade area delineation
   - Probability contours (50%, 70%, 90%)
   - Drive-time isochrones via OSRM or Valhalla
   - GeoJSON / GeoDataFrame output

### Phase 2: Consumer Segmentation
**Files:** `latent_class.py`, `geodemographic.py`, `rfm.py`, `clv.py`

6. **`latent_class.py`** — Expectation-Maximization LCA
   - Discovers K behavioral segments from visit/purchase patterns
   - Each segment gets its own Huff parameter set (α_k, λ_k)
   - BIC/AIC for optimal K selection
7. **`geodemographic.py`** — External segment integration
   - Maps census tracts to PRIZM/Mosaic/Tapestry codes
   - Crosswalk to behavioral propensity scores
   - Accepts custom segment definitions
8. **`rfm.py`** — Transaction-based scoring
   - Recency, Frequency, Monetary value calculation from POS data
   - Quintile scoring + composite RFM segments
9. **`clv.py`** — BG/NBD + Gamma-Gamma CLV
   - Probabilistic customer lifetime value from transaction history
   - Identifies high-value vs at-risk customers per trade area

### Phase 3: Mixed Logit Choice Model
**Files:** `mixed_logit.py`

10. **`mixed_logit.py`** — Random-coefficients logit
    - Utility: Uᵢⱼ = β'xᵢⱼ + εᵢⱼ where β ~ f(θ)
    - Attributes: distance, size, price level, reviews, brand, parking, product mix
    - Simulated Maximum Likelihood estimation
    - Willingness-to-travel and attribute importance outputs
    - Per-segment coefficient distributions

### Phase 4: Temporal Dynamics
**Files:** `hmm.py`, `hawkes.py`, `bayesian_update.py`

11. **`hmm.py`** — Consumer state model
    - States: Loyal, Exploring, Lapsing, Churned, Won-back
    - Transition probabilities from longitudinal visit data
    - Viterbi decoding for individual consumer state paths
    - Segment-level state distribution forecasting
12. **`hawkes.py`** — Self-exciting visit process
    - Models how recent visits increase near-term revisit probability
    - Captures "momentum" and "cooling" effects
    - Parametric triggering kernel (exponential decay)
13. **`bayesian_update.py`** — Real-time posterior updating
    - Prior: structural model predictions (Huff/Mixed Logit)
    - Likelihood: new observation (mobile ping, transaction, app event)
    - Posterior: updated visit probability via conjugate or MCMC update
    - Streaming-compatible (processes events incrementally)

### Phase 5: ML Residual Layer
**Files:** `residual_boost.py`, `graph_network.py`

14. **`residual_boost.py`** — Gradient boosting on structural errors
    - Target: actual_share - structural_predicted_share
    - Features: all attributes + interactions the parametric models miss
    - XGBoost or LightGBM with spatial cross-validation
    - SHAP explanations for residual drivers
15. **`graph_network.py`** — Consumer-store graph learning
    - Bipartite graph: consumers ↔ stores
    - Node features: consumer demographics, store attributes
    - Edge features: visit frequency, recency, monetary value
    - PyTorch Geometric GNN for link prediction (who shops where next)

### Phase 6: Ensemble & Reporting
**Files:** `model_averaging.py`, `trade_area_report.py`, `consumer_profile.py`, `scenario.py`, `visualize.py`

16. **`model_averaging.py`** — Bayesian Model Averaging
    - Weights structural + ML predictions by held-out accuracy
    - Uncertainty quantification via prediction intervals
17. **`trade_area_report.py`** — Automated analysis output
    - Trade area size, penetration, fair-share index
    - Competitive vulnerability scores
    - Demographic/behavioral profile of trade area
18. **`consumer_profile.py`** — Segment reporting
    - Per-segment: size, value, loyalty state distribution, CLV
    - Migration analysis (segment flows over time)
19. **`scenario.py`** — What-if simulator
    - New store opening impact
    - Competitor entry/exit
    - Store renovation (attractiveness change)
    - Price/assortment changes
20. **`visualize.py`** — Map and chart generation
    - Folium interactive maps with trade area overlays
    - Probability heatmaps
    - Segment composition charts
    - Temporal trend plots

### Phase 7: Data Loaders & Configuration
**Files:** `census.py`, `osm.py`, `safegraph.py`, `transactions.py`, `default_config.yaml`

21. **Data loaders** — Standardized ingestion
    - Census: ACS 5-year via `cenpy` or direct API
    - OSM: `osmnx` for POI and network data
    - SafeGraph: patterns CSV/parquet loader with visit normalization
    - Transactions: flexible CSV/parquet with configurable column mapping
22. **Config** — YAML-driven parameterization
    - Distance decay defaults, segment count, kernel bandwidths
    - Data source paths, API keys
    - Model toggle flags (enable/disable layers)

## Key Dependencies

```
numpy>=1.24
scipy>=1.10
pandas>=2.0
geopandas>=0.13
scikit-learn>=1.3
statsmodels>=0.14
pydantic>=2.0
xgboost>=2.0
lightgbm>=4.0
hmmlearn>=0.3
folium>=0.14
matplotlib>=3.7
seaborn>=0.12
osmnx>=1.5
pysal>=23.1
torch>=2.0          # optional, for GNN
torch-geometric>=2.3 # optional, for GNN
cenpy>=1.0
pyyaml>=6.0
shapely>=2.0
```

## Usage Example (Target API)

```python
from gravity import GravityModel
from gravity.data import CensusLoader, OSMLoader, SafeGraphLoader, TransactionLoader

# Load data
stores = OSMLoader.load_retail_pois(bbox=(-74.1, 40.6, -73.7, 40.9))
origins = CensusLoader.load_block_groups(state="NY", county="061")
traffic = SafeGraphLoader.load_patterns("safegraph_patterns.csv")
txns = TransactionLoader.load("pos_data.parquet")

# Build model
model = GravityModel(
    layers=["huff", "competing_destinations", "mixed_logit",
            "latent_class", "bayesian_update", "residual_boost"],
    config="config/default_config.yaml"
)

# Fit
model.fit(stores=stores, origins=origins, traffic=traffic, transactions=txns)

# Predict
predictions = model.predict(origins=origins, stores=stores)
# → DataFrame: origin_id, store_id, probability, segment, state, clv

# Trade area report
report = model.trade_area_report(store_id="store_123")
report.save_html("trade_area_store_123.html")

# Scenario: what if competitor opens?
scenario = model.scenario(
    action="new_store",
    location=(-73.98, 40.75),
    attractiveness=8500,
)
scenario.impact_report()

# Real-time update
model.update(event={"consumer_id": "C456", "store_id": "store_123",
                     "timestamp": "2026-04-28T14:30:00", "amount": 47.50})
```

## Build Order (Sequential Dependencies)

```
schema.py → huff.py → competing_destinations.py → gwr.py → trade_area.py
                 ↓
         mixed_logit.py
                 ↓
    latent_class.py → geodemographic.py
    rfm.py → clv.py
                 ↓
    hmm.py → hawkes.py → bayesian_update.py
                 ↓
    residual_boost.py → graph_network.py
                 ↓
    model_averaging.py → reporting modules
                 ↓
    data loaders (parallel, any time)
    config + examples (last)
```
