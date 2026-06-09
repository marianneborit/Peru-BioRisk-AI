"""
Peru BioRisk AI — SENAMHI ingestion
Downloads temperature and precipitation data from SENAMHI public API.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


def extract(start_date: date, end_date: date, output_dir: Path) -> list[Path]:
    """
    Downloads SENAMHI station data for all stations in Peru.
    Returns list of written Parquet files.

    API docs: https://api.senamhi.gob.pe/api/v1
    Requires: configs/config.yaml → apis.senamhi.token
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("SENAMHI extract: %s → %s", start_date, end_date)
    # TODO: implement actual API calls
    # Example structure:
    # GET /estaciones?tipo=automatica&activo=true → list of station IDs
    # GET /datos/{station_id}?inicio={start}&fin={end}&variable=temperatura,precipitacion
    logger.warning("SENAMHI extractor not yet implemented — returning empty list")
    return []
