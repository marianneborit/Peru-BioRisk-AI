"""Peru BioRisk AI — era5 ingestion stub."""
from __future__ import annotations
from datetime import date
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def extract(start_date: date, end_date: date, output_dir: Path) -> list:
    """Download era5 data. TODO: implement."""
    logger.warning("era5 extractor not yet implemented")
    return []
