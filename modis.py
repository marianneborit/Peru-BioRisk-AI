"""Peru BioRisk AI — modis ingestion stub."""
from __future__ import annotations
from datetime import date
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def extract(start_date: date, end_date: date, output_dir: Path) -> list:
    """Download modis data. TODO: implement."""
    logger.warning("modis extractor not yet implemented")
    return []
