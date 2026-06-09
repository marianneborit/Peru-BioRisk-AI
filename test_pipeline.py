"""
Peru BioRisk AI — Test suite
Tests for ETL pipeline, feature engineering, and API endpoints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_panel() -> pd.DataFrame:
    """Minimal panel DataFrame for testing feature engineering."""
    rng = np.random.default_rng(0)
    ubigeos = ["150101", "160101"]
    weeks = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    rows = []
    for ubigeo in ubigeos:
        for week in weeks:
            rows.append({
                "ubigeo": ubigeo,
                "epi_week_start": week,
                "date": week,
                "temp_mean_c": rng.normal(25, 3),
                "precip_mm": max(0.0, rng.normal(50, 30)),
                "humidity_pct": rng.uniform(60, 95),
                "ndvi_mean": rng.uniform(0.3, 0.8),
                "lst_day_c": rng.normal(30, 5),
                "surface_water_km2": max(0.0, rng.normal(5, 2)),
                "deforestation_ha": max(0.0, rng.normal(10, 8)),
                "dengue_cases": max(0, int(rng.poisson(20))),
                "population": 50000,
                "forest_cover_pct": rng.uniform(10, 80),
                "agri_pct": rng.uniform(5, 40),
                "mining_pct": rng.uniform(0, 10),
                "poverty_pct": rng.uniform(10, 60),
                "water_access_hh_pct": rng.uniform(30, 95),
            })
    return pd.DataFrame(rows)


# ── ETL / validation tests ────────────────────────────────────────────────────

class TestValidation:
    def test_completeness_pass(self, sample_panel: pd.DataFrame) -> None:
        from src.etl.pipeline import validate_completeness
        nan_pcts = validate_completeness(sample_panel, max_nan_pct=0.05, raise_on_fail=True)
        assert all(v == 0.0 for v in nan_pcts.values())

    def test_completeness_fail(self, sample_panel: pd.DataFrame) -> None:
        from src.etl.pipeline import validate_completeness
        df = sample_panel.copy()
        df.loc[:20, "temp_mean_c"] = np.nan  # 20/104 rows ~ 19% NaN
        with pytest.raises(ValueError, match="NaN threshold"):
            validate_completeness(df, max_nan_pct=0.05, raise_on_fail=True)

    def test_range_validation_pass(self, sample_panel: pd.DataFrame) -> None:
        from src.etl.pipeline import validate_value_ranges, PHYSICAL_RANGES
        violations = validate_value_ranges(sample_panel, PHYSICAL_RANGES, raise_on_fail=False)
        assert violations == []

    def test_range_validation_fail(self, sample_panel: pd.DataFrame) -> None:
        from src.etl.pipeline import validate_value_ranges
        df = sample_panel.copy()
        df.loc[0, "temp_mean_c"] = 999.0  # impossible temperature
        violations = validate_value_ranges(
            df, {"temp_mean_c": (-10.0, 45.0)}, raise_on_fail=False
        )
        assert len(violations) > 0

    def test_epi_week_alignment(self, sample_panel: pd.DataFrame) -> None:
        from src.etl.pipeline import align_to_epi_week
        df = align_to_epi_week(sample_panel)
        assert "epi_year" in df.columns
        assert "epi_week" in df.columns
        assert df["epi_week"].between(1, 53).all()


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_climate_features_shape(self, sample_panel: pd.DataFrame) -> None:
        from src.features.feature_engineering import add_climate_features
        df = add_climate_features(sample_panel, lags_weeks=[1, 2, 4])
        assert "temp_mean_lag1w" in df.columns
        assert "temp_mean_lag4w" in df.columns
        assert "precip_cumul_30d" in df.columns
        assert "temp_anomaly_z" in df.columns
        assert len(df) == len(sample_panel)

    def test_climate_lags_are_shifted(self, sample_panel: pd.DataFrame) -> None:
        from src.features.feature_engineering import add_climate_features
        df = add_climate_features(sample_panel, lags_weeks=[1])
        grp = df.groupby("ubigeo")
        for _, group in grp:
            group = group.reset_index(drop=True)
            # lag1 at row i should equal temp at row i-1
            # First row should be NaN
            assert pd.isna(group["temp_mean_lag1w"].iloc[0])
            # Second row should match first row's temp
            assert pytest.approx(group["temp_mean_lag1w"].iloc[1]) == group["temp_mean_c"].iloc[0]

    def test_epi_features(self, sample_panel: pd.DataFrame) -> None:
        from src.features.feature_engineering import add_epi_features
        df = add_epi_features(
            sample_panel,
            case_cols={"dengue": "dengue_cases"},
            lags_weeks=[1, 2],
        )
        assert "dengue_cases_lag1w" in df.columns
        assert "dengue_incidence_100k" in df.columns
        assert (df["dengue_incidence_100k"] >= 0).all()

    def test_landuse_features(self, sample_panel: pd.DataFrame) -> None:
        from src.features.feature_engineering import add_landuse_features
        df = add_landuse_features(sample_panel)
        assert "eco_edge_index" in df.columns
        assert "anthropogenic_pressure" in df.columns
        assert df["eco_edge_index"].between(0, 100).all()

    def test_biorisk_index_range(self, sample_panel: pd.DataFrame) -> None:
        from src.features.feature_engineering import (
            add_climate_features, add_satellite_features,
            add_epi_features, add_landuse_features, compute_biorisk_index,
        )
        df = add_climate_features(sample_panel)
        df = add_satellite_features(df)
        df = add_epi_features(df, case_cols={"dengue": "dengue_cases"})
        df = add_landuse_features(df)
        df = compute_biorisk_index(df)
        valid = df["biorisk_index"].dropna()
        assert len(valid) > 0
        assert valid.between(0, 1).all(), "BioRisk Index must be in [0, 1]"

    def test_dry_spell_non_negative(self, sample_panel: pd.DataFrame) -> None:
        from src.features.feature_engineering import add_climate_features
        df = add_climate_features(sample_panel)
        assert (df["dry_spell_weeks"] >= 0).all()


# ── API tests ─────────────────────────────────────────────────────────────────

@pytest.fixture
def api_client() -> TestClient:
    from src.api.main import app
    return TestClient(app)


class TestAPI:
    def test_health(self, api_client: TestClient) -> None:
        r = api_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_risk_map_returns_geojson(self, api_client: TestClient) -> None:
        r = api_client.get("/api/v1/risk-map?week=2024-W10&disease=dengue")
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "FeatureCollection"
        assert "features" in body

    def test_forecast_valid_ubigeo(self, api_client: TestClient) -> None:
        r = api_client.get("/api/v1/forecast/150101?disease=dengue&horizon_weeks=4")
        assert r.status_code == 200
        body = r.json()
        assert body["ubigeo"] == "150101"
        assert len(body["forecast"]) == 4

    def test_forecast_invalid_ubigeo(self, api_client: TestClient) -> None:
        r = api_client.get("/api/v1/forecast/BADCODE")
        assert r.status_code == 422

    def test_active_alerts(self, api_client: TestClient) -> None:
        r = api_client.get("/api/v1/alerts/active")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_scenario_endpoint(self, api_client: TestClient) -> None:
        payload = {
            "ubigeo": "160101",
            "deforestation_delta_pct": -20.0,
            "temp_anomaly_delta_c": 0.5,
            "precip_delta_pct": 0.0,
            "horizon_weeks": 4,
        }
        r = api_client.post("/api/v1/scenario", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "baseline_risk" in body
        assert "scenario_risk" in body
        assert 0 <= body["scenario_risk"] <= 1

    def test_model_info(self, api_client: TestClient) -> None:
        r = api_client.get("/api/v1/model-info")
        assert r.status_code == 200
        assert "active_models" in r.json()
