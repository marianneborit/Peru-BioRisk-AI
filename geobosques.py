"""Peru BioRisk AI — geobosques ingestion stub."""
from __future__ import annotations
from datetime import date
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def extract(start_date: date, end_date: date, output_dir: Path) -> list:
    """Download geobosques data. TODO: implement."""
    logger.warning("geobosques extractor not yet implemented")
    return []
