"""Peru BioRisk AI — cdc_peru ingestion stub."""
from __future__ import annotations
from datetime import date
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def extract(start_date: date, end_date: date, output_dir: Path) -> list:
    """Download cdc_peru data. TODO: implement."""
    logger.warning("cdc_peru extractor not yet implemented")
    return []
