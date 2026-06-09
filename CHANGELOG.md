# Changelog

## [Unreleased]

### Added
- Initial project structure
- ETL pipeline skeleton (reproject, align, validate)
- Feature engineering module (climate, satellite, epi, landuse, spatial, BioRisk Index)
- XGBoost/LightGBM ensemble with Optuna + spatial block CV + MLflow tracking
- FastAPI application with risk-map, forecast, alerts, scenario, and WebSocket endpoints
- Apache Airflow weekly DAG
- Docker Compose stack (PostGIS/TimescaleDB, MinIO, MLflow, Airflow, Grafana, pg_tileserv)
- GitHub Actions CI pipeline
- Demo ETL script with synthetic data
- Comprehensive test suite (pytest)
- Data dictionary documentation
- CONTRIBUTING guide

## [0.1.0] — 2024-Q3 (planned)

### Planned
- SENAMHI, ERA5, MODIS, CDC Perú, GEOBOSQUES ingestion modules
- National coverage (1,874 districts)
- Public API v0.1
- Dashboard v0.1 (React + Kepler.gl)
