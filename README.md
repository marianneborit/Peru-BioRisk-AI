# Peru BioRisk AI

> Open-source spatiotemporal machine learning framework for biological risk mapping in Peru.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Infra-Docker%20%2B%20K8s-2496ED.svg)](https://docker.com)
[![Data: CC-BY 4.0](https://img.shields.io/badge/Data-CC--BY%204.0-orange.svg)](https://creativecommons.org/licenses/by/4.0/)

**Peru BioRisk AI** integrates climatic, environmental, epidemiological, and land-use data to generate
weekly biological risk maps at district level (1,874 distritos) across Peru.
The system predicts outbreak probability for dengue, malaria, leptospirosis, leishmaniasis,
and bartonellosis at 4 / 8 / 12-week horizons.

---

## Table of contents

- [Why this project](#why-this-project)
- [Architecture overview](#architecture-overview)
- [Quickstart](#quickstart)
- [Project structure](#project-structure)
- [Data sources](#data-sources)
- [Models](#models)
- [API](#api)
- [Dashboard](#dashboard)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Publications](#publications)
- [License](#license)

---

## Why this project

Peru hosts three radically different ecosystems — arid coast, highland Andes, and Amazon rainforest —
each with distinct disease ecologies. Climate change, deforestation, and unplanned urbanisation are
reshuffling biological risk faster than traditional epidemiological surveillance can track.

Peru BioRisk AI provides:

- **Early warning** — district-level outbreak probability up to 12 weeks ahead
- **Causal insight** — SHAP-based attribution of risk drivers (deforestation, temperature anomalies, etc.)
- **Open science** — fully reproducible pipeline, versioned datasets, public API
- **Institutional bridge** — data contracts with MINSA, CDC Perú, MINAM, and SENAMHI

---

## Architecture overview

```
┌─────────────────────────────────────────────────────┐
│  Layer 1 · Data sources                             │
│  SENAMHI · ERA5 · CHIRPS · MODIS · CDC Perú · GBIF  │
│  GEOBOSQUES · MapBiomas · INEI · OpenStreetMap       │
└────────────────────┬────────────────────────────────┘
                     │ Airflow DAGs
┌────────────────────▼────────────────────────────────┐
│  Layer 2 · ETL Pipeline                             │
│  Extract → Reproject → Feature Eng. → Validate      │
│  PostGIS · TimescaleDB · MinIO · Feast Feature Store │
└────────────────────┬────────────────────────────────┘
                     │ MLflow runs
┌────────────────────▼────────────────────────────────┐
│  Layer 3 · ML Models                                │
│  XGBoost/LightGBM Ensemble · Bi-LSTM · TCN          │
│  BRT Species Distribution · GWR · Stacking          │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Layer 4 · Outputs                                  │
│  Geo Dashboard (React + Kepler.gl)                  │
│  REST / OGC API (FastAPI)                           │
│  Early-warning alerts (WebSocket + webhooks)        │
└─────────────────────────────────────────────────────┘
```

---

## Quickstart

### Prerequisites

- Docker ≥ 24 and Docker Compose v2
- Python 3.10+
- 16 GB RAM recommended (models + PostGIS)

### 1. Clone and configure

```bash
git clone https://github.com/peru-biorisk-ai/peru-biorisk-ai.git
cd peru-biorisk-ai
cp configs/config.example.yaml configs/config.yaml
# Edit configs/config.yaml with your API keys (SENAMHI, NASA Earthdata, CDS)
```

### 2. Spin up infrastructure

```bash
docker compose -f infrastructure/docker/docker-compose.yml up -d
# Services: PostGIS, TimescaleDB, MinIO, Airflow, MLflow, Grafana
```

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the ETL pipeline (demo mode — Lima + Loreto, dengue only)

```bash
python scripts/run_demo_etl.py --region lima,loreto --disease dengue --start 2023-01-01
```

### 5. Train baseline model

```bash
python src/models/train.py --config configs/models/xgboost_baseline.yaml
```

### 6. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

### 7. Start the dashboard

```bash
cd src/dashboard && npm install && npm run dev
# Open http://localhost:3000
```

---

## Project structure

```
peru-biorisk-ai/
│
├── configs/                    # YAML configs (model, ETL, API, infra)
│   ├── config.example.yaml
│   ├── models/
│   └── etl/
│
├── data/                       # Not committed — see .gitignore
│   ├── raw/                    # Downloaded originals
│   ├── processed/              # Reprojected, aligned rasters
│   └── features/               # Feature store outputs
│
├── src/
│   ├── ingestion/              # Data downloaders per source
│   │   ├── senamhi.py
│   │   ├── era5.py
│   │   ├── modis.py
│   │   ├── cdc_peru.py
│   │   └── geobosques.py
│   │
│   ├── etl/                    # Transform + validate
│   │   ├── reproject.py
│   │   ├── align.py
│   │   ├── impute.py
│   │   └── validate.py
│   │
│   ├── features/               # Feature engineering
│   │   ├── climate_features.py
│   │   ├── satellite_features.py
│   │   ├── epi_features.py
│   │   ├── landuse_features.py
│   │   ├── spatial_features.py
│   │   └── biorisk_index.py
│   │
│   ├── models/                 # Training + inference
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── ensemble.py
│   │   ├── lstm_model.py
│   │   ├── tcn_model.py
│   │   ├── brt_species.py
│   │   └── causal_analysis.py
│   │
│   ├── api/                    # FastAPI application
│   │   ├── main.py
│   │   ├── routers/
│   │   └── schemas/
│   │
│   └── dashboard/              # React + Kepler.gl frontend
│       ├── src/
│       └── public/
│
├── infrastructure/
│   ├── docker/
│   │   └── docker-compose.yml
│   ├── airflow/
│   │   └── dags/
│   └── k8s/
│
├── notebooks/                  # Exploratory analysis
├── tests/                      # pytest test suite
├── scripts/                    # Utility scripts
├── docs/                       # Extended documentation
├── .github/workflows/          # CI/CD
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Data sources

| Domain | Source | Frequency | Resolution |
|---|---|---|---|
| Temperature / humidity | SENAMHI, ERA5 | Daily | Station / 0.25° |
| Precipitation | CHIRPS v2, GPM IMERG | Daily | 0.05° |
| Land surface temp | MODIS MOD11A1 | Daily | 1 km |
| Vegetation (NDVI/EVI) | MODIS MOD13A2 | 16-day | 1 km |
| Deforestation alerts | GEOBOSQUES (MINAM) | Weekly | ~30 m |
| Land cover | MapBiomas Perú | Annual | 30 m |
| Disease notifications | CDC Perú / NOTI | Weekly | District |
| Vector occurrences | GBIF, VectorBase | Irregular | Point |
| Socioeconomic | INEI Censos 2017 | Annual | District |
| Infrastructure | OpenStreetMap | Continuous | Vector |

Full data dictionary: [`docs/data_dictionary.md`](docs/data_dictionary.md)

---

## Models

| Model | Task | Horizon | Target metric |
|---|---|---|---|
| XGBoost/LightGBM Ensemble | Binary outbreak + case count | 4 weeks | AUC-ROC ≥ 0.85 |
| Bidirectional LSTM | Case count forecast | 4/8/12 weeks | MAE, CRPS |
| TCN | Case count forecast | 4/8/12 weeks | MAE, CRPS |
| BRT (MaxEnt-style) | Vector habitat suitability | Seasonal | AUC, Boyce index |
| GWR | Spatial coefficient heterogeneity | Cross-sectional | Local R² |
| Stacking meta-model | Combined risk index | 4/8/12 weeks | Brier score |

All experiments tracked in MLflow at `http://localhost:5000`.

---

## API

Base URL: `https://api.perubiorisk.ai/v1` (production) or `http://localhost:8000` (local)

```
GET  /risk-map?week=2024-W10&disease=dengue&format=geojson
GET  /forecast/{ubigeo}?horizon_weeks=8
GET  /alerts/active?level=critical
POST /scenario   (counterfactual simulation)
GET  /features/{ubigeo}/{date}
WS   /ws/alerts  (real-time stream)
```

Full OpenAPI spec: `http://localhost:8000/docs`
Rate limit: 100 req / min (free, API key required).

---

## Contributing

We welcome contributions! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before submitting a PR.

Quick contribution guide:

1. Fork the repo and create a feature branch (`git checkout -b feature/my-improvement`)
2. Follow code style (`ruff`, `black`, type hints required)
3. Add tests for new functionality (`pytest tests/`)
4. Update relevant docs
5. Submit a pull request — a maintainer will review within 5 business days

See [`docs/development.md`](docs/development.md) for local dev setup details.

---

## Roadmap

| Phase | Period | Milestone |
|---|---|---|
| 0 — Foundation | M1–M2 | Infra, ETL skeleton, Lima + Loreto pilot |
| 1 — MVP | M3–M5 | 5 diseases, XGBoost baseline, public API v0.1 |
| 2 — Advanced models | M6–M9 | LSTM/TCN, BRT, meta-ensemble, full national coverage |
| 3 — Institutional | M10–M14 | MINSA MOU, automated ingestion, first paper |
| 4 — Climate scenarios | M15–M20 | CMIP6 projections 2030/2040/2050 |
| 5 — Regional expansion | M21–M30 | Bolivia, Ecuador, Colombia adaptation |

---

## Publications

Papers in preparation / submitted:

1. *Peru BioRisk AI: An open-source spatiotemporal ML framework for biological risk mapping* — Scientific Data
2. *Deforestation-driven amplification of vector-borne disease risk in the Peruvian Amazon* — PLOS NTDs
3. *Nonlinear climate thresholds for dengue outbreak prediction in Peru* — Environmental Health Perspectives
4. *Early warning system performance: prospective validation* — The Lancet Digital Health

---

## License

Code: [Apache 2.0 License](LICENSE)
Derived datasets: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Acknowledgements

Built with data from SENAMHI, MINSA/CDC Perú, MINAM/GEOBOSQUES, INEI, NASA Earthdata,
ECMWF/Copernicus, MapBiomas, GBIF, and the OpenStreetMap community. AI used by author for programming and writing.
