"""
Peru BioRisk AI — REST API (FastAPI)
OGC API Features compatible endpoints for risk maps, forecasts, and alerts.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Peru BioRisk AI API",
    description=(
        "Spatiotemporal biological risk maps for Peru. "
        "OGC API Features compatible. Data licensed CC-BY 4.0."
    ),
    version="0.1.0",
    contact={"name": "Peru BioRisk AI", "url": "https://github.com/peru-biorisk-ai"},
    license_info={"name": "Apache 2.0 (code) / CC-BY 4.0 (data)", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://dashboard.perubiorisk.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Enums & schemas ───────────────────────────────────────────────────────────

class Disease(str, Enum):
    dengue = "dengue"
    malaria = "malaria"
    leptospirosis = "leptospirosis"
    leishmaniasis = "leishmaniasis"
    bartonellosis = "bartonellosis"


class AlertLevel(str, Enum):
    warning = "warning"
    alert = "alert"
    critical = "critical"


class Region(str, Enum):
    coast = "coast"
    sierra = "sierra"
    amazon = "amazon"
    all = "all"


class RiskMapParams(BaseModel):
    week: str = Field(..., example="2024-W10", description="ISO 8601 week (YYYY-Www)")
    disease: Disease = Disease.dengue
    format: str = Field("geojson", pattern="^(geojson|pmtiles|cog)$")


class ForecastResponse(BaseModel):
    ubigeo: str
    district_name: str
    disease: str
    horizon_weeks: int
    forecast: list[dict[str, Any]]
    model_version: str
    generated_at: datetime


class AlertItem(BaseModel):
    ubigeo: str
    district_name: str
    region: str
    disease: str
    risk_score: float
    level: AlertLevel
    top_drivers: list[str]
    change_vs_prev_week: float
    generated_at: datetime


class ScenarioRequest(BaseModel):
    ubigeo: str = Field(..., description="District UBIGEO code (6 digits)")
    deforestation_delta_pct: float = Field(0.0, ge=-100, le=200)
    temp_anomaly_delta_c: float = Field(0.0, ge=-5, le=5)
    precip_delta_pct: float = Field(0.0, ge=-100, le=200)
    horizon_weeks: int = Field(4, ge=1, le=12)


class ScenarioResponse(BaseModel):
    ubigeo: str
    baseline_risk: float
    scenario_risk: float
    risk_delta: float
    counterfactual_explanation: list[str]


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


# ── Risk map endpoint ─────────────────────────────────────────────────────────

@app.get(
    "/api/v1/risk-map",
    tags=["risk"],
    summary="District-level biological risk map",
    response_description="GeoJSON FeatureCollection with BioRisk Index per district",
)
async def get_risk_map(
    week: str = Query(..., example="2024-W10"),
    disease: Disease = Query(Disease.dengue),
    format: str = Query("geojson", pattern="^(geojson|pmtiles|cog)$"),
    min_risk: float = Query(0.0, ge=0, le=1, description="Filter districts below this risk score"),
) -> JSONResponse:
    """
    Returns biological risk scores for all 1,874 Peruvian districts
    for the requested epidemiological week and disease.

    Risk levels: warning ≥ 0.40, alert ≥ 0.65, critical ≥ 0.80.
    """
    # In production, this queries PostGIS / Feature Store
    # Here we return a minimal example structure
    example_feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-77.04, -12.04]},
        "properties": {
            "ubigeo": "150101",
            "district_name": "Lima",
            "biorisk_index": 0.72,
            "alert_level": "alert",
            "disease": disease.value,
            "week": week,
            "top_drivers": ["temp_anomaly_z", "precip_cumul_30d", "dengue_cases_lag2w"],
        },
    }
    geojson = {
        "type": "FeatureCollection",
        "features": [example_feature],
        "metadata": {
            "week": week,
            "disease": disease.value,
            "generated_at": datetime.utcnow().isoformat(),
            "model_version": "v0.1.0",
            "license": "CC-BY 4.0",
            "source": "Peru BioRisk AI",
        },
    }
    return JSONResponse(content=geojson)


# ── Forecast endpoint ─────────────────────────────────────────────────────────

@app.get(
    "/api/v1/forecast/{ubigeo}",
    tags=["forecast"],
    response_model=ForecastResponse,
    summary="District outbreak probability forecast",
)
async def get_forecast(
    ubigeo: str,
    disease: Disease = Query(Disease.dengue),
    horizon_weeks: int = Query(4, ge=1, le=12),
) -> ForecastResponse:
    """
    Returns weekly outbreak probability forecast with 80% and 95% confidence intervals
    for the specified district (ubigeo) and horizon.
    """
    if len(ubigeo) != 6 or not ubigeo.isdigit():
        raise HTTPException(status_code=422, detail="ubigeo must be a 6-digit string")

    # Placeholder forecast data
    forecast_points = [
        {
            "week_offset": i + 1,
            "outbreak_probability": round(0.45 + 0.05 * i, 3),
            "ci_80_lower": round(0.35 + 0.05 * i, 3),
            "ci_80_upper": round(0.55 + 0.05 * i, 3),
            "ci_95_lower": round(0.30 + 0.05 * i, 3),
            "ci_95_upper": round(0.62 + 0.05 * i, 3),
            "expected_cases": max(0, round(12 + i * 3)),
        }
        for i in range(horizon_weeks)
    ]

    return ForecastResponse(
        ubigeo=ubigeo,
        district_name="Example District",
        disease=disease.value,
        horizon_weeks=horizon_weeks,
        forecast=forecast_points,
        model_version="v0.1.0",
        generated_at=datetime.utcnow(),
    )


# ── Active alerts endpoint ────────────────────────────────────────────────────

@app.get(
    "/api/v1/alerts/active",
    tags=["alerts"],
    response_model=list[AlertItem],
    summary="Active biological risk alerts",
)
async def get_active_alerts(
    level: AlertLevel | None = Query(None),
    region: Region = Query(Region.all),
    disease: Disease | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> list[AlertItem]:
    """
    Returns districts currently at elevated biological risk.
    Filter by alert level (warning/alert/critical), ecological region, or disease.
    """
    # Placeholder — production queries PostGIS materialized view
    return [
        AlertItem(
            ubigeo="160101",
            district_name="Iquitos",
            region="amazon",
            disease="dengue",
            risk_score=0.83,
            level=AlertLevel.critical,
            top_drivers=["temp_anomaly_z (+2.1σ)", "defor_cumul_24w (+340 ha)", "surface_water_km2 (+18%)"],
            change_vs_prev_week=0.12,
            generated_at=datetime.utcnow(),
        )
    ]


# ── Scenario / counterfactual endpoint ───────────────────────────────────────

@app.post(
    "/api/v1/scenario",
    tags=["analysis"],
    response_model=ScenarioResponse,
    summary="Counterfactual scenario simulation",
)
async def run_scenario(payload: ScenarioRequest) -> ScenarioResponse:
    """
    Simulates the effect of environmental interventions on biological risk.
    Example: what happens to dengue risk in Loreto if deforestation decreases 20%?

    Uses SHAP-based feature attribution and do-calculus approximation.
    """
    # Placeholder computation
    baseline_risk = 0.72
    delta = (
        -payload.deforestation_delta_pct * 0.001
        + payload.temp_anomaly_delta_c * 0.04
        + payload.precip_delta_pct * 0.0005
    )
    scenario_risk = float(min(1.0, max(0.0, baseline_risk + delta)))

    return ScenarioResponse(
        ubigeo=payload.ubigeo,
        baseline_risk=round(baseline_risk, 4),
        scenario_risk=round(scenario_risk, 4),
        risk_delta=round(scenario_risk - baseline_risk, 4),
        counterfactual_explanation=[
            f"Deforestation change ({payload.deforestation_delta_pct:+.1f}%) contributes "
            f"{-payload.deforestation_delta_pct * 0.001:.4f} to risk delta",
            f"Temperature anomaly change ({payload.temp_anomaly_delta_c:+.2f}°C) contributes "
            f"{payload.temp_anomaly_delta_c * 0.04:.4f} to risk delta",
        ],
    )


# ── Feature vector endpoint (audit / reproducibility) ────────────────────────

@app.get(
    "/api/v1/features/{ubigeo}/{reference_date}",
    tags=["data"],
    summary="Full feature vector for a district on a given date",
)
async def get_features(ubigeo: str, reference_date: date) -> dict[str, Any]:
    """
    Returns the complete feature vector used for a prediction,
    enabling full reproducibility and external model auditing.
    """
    return {
        "ubigeo": ubigeo,
        "reference_date": str(reference_date),
        "features": {
            "temp_mean_c": 28.4,
            "temp_anomaly_z": 1.8,
            "precip_cumul_30d": 215.3,
            "ndvi_mean": 0.72,
            "defor_cumul_24w": 340.0,
            "dengue_cases_lag2w": 45,
            "biorisk_index": 0.72,
        },
        "model_version": "v0.1.0",
        "feature_schema_version": "v1",
    }


# ── Model metadata ────────────────────────────────────────────────────────────

@app.get("/api/v1/model-info", tags=["meta"])
async def get_model_info() -> dict[str, Any]:
    return {
        "active_models": {
            "dengue": {"version": "v0.1.0", "trained_on": "2024-01-01", "oof_auc_roc": 0.87},
            "malaria": {"version": "v0.1.0", "trained_on": "2024-01-01", "oof_auc_roc": 0.84},
        },
        "mlflow_run_id": "abc123",
        "data_freshness": {"last_etl_run": datetime.utcnow().isoformat()},
    }


# ── WebSocket — real-time alert stream ───────────────────────────────────────

class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active_connections.remove(ws)

    async def broadcast(self, message: str) -> None:
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/v1/alerts")
async def websocket_alerts(websocket: WebSocket) -> None:
    """
    Real-time alert stream. Emits JSON alert objects whenever a new
    district crosses a risk threshold.
    """
    await manager.connect(websocket)
    try:
        while True:
            # In production: subscribe to a Postgres NOTIFY channel
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps({"ping": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
