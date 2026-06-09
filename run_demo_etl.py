#!/usr/bin/env python3
"""
Peru BioRisk AI — Demo ETL script
Quick-start: runs the pipeline for a subset of regions and one disease.

Usage:
    python scripts/run_demo_etl.py --region lima,loreto --disease dengue --start 2023-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("demo_etl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Peru BioRisk AI demo ETL")
    parser.add_argument("--region", default="lima,loreto", help="Comma-separated region names")
    parser.add_argument("--disease", default="dengue", help="Disease target")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=str(date.today()), help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regions = [r.strip() for r in args.region.split(",")]
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    base = Path(args.data_dir)

    logger.info("=== Peru BioRisk AI — Demo ETL ===")
    logger.info("Regions : %s", regions)
    logger.info("Disease : %s", args.disease)
    logger.info("Period  : %s → %s", start, end)
    logger.info("Data dir: %s", base.resolve())

    if args.dry_run:
        logger.info("[DRY RUN] No files will be written.")
        print_pipeline_plan(regions, args.disease, start, end)
        return

    # Create directory structure
    for domain in ("climatic", "satellite", "epidemiological", "landuse"):
        (base / "raw" / domain).mkdir(parents=True, exist_ok=True)
    (base / "processed").mkdir(parents=True, exist_ok=True)
    (base / "features").mkdir(parents=True, exist_ok=True)

    # Run ETL stages
    logger.info("Step 1/5: Downloading synthetic demo data (no API keys required)...")
    _generate_synthetic_data(base, regions, args.disease, start, end)

    logger.info("Step 2/5: Reprojecting rasters (skipped in demo — no real rasters)...")

    logger.info("Step 3/5: Building feature matrix...")
    _build_demo_features(base, regions, args.disease)

    logger.info("Step 4/5: Validating features...")
    _validate_demo_features(base)

    logger.info("Step 5/5: Summary")
    feature_file = base / "features" / "demo_features.parquet"
    if feature_file.exists():
        import pandas as pd
        df = pd.read_parquet(feature_file)
        logger.info(
            "Feature matrix: %d rows × %d cols | districts: %d",
            len(df), len(df.columns), df["ubigeo"].nunique(),
        )
    else:
        logger.warning("Feature file not found — something went wrong.")

    logger.info("Demo ETL complete. Next: python src/models/train.py --config configs/models/xgboost_baseline.yaml")


def print_pipeline_plan(regions: list[str], disease: str, start: date, end: date) -> None:
    weeks = (end - start).days // 7
    print(f"\nPipeline plan for {disease} | {start} → {end} ({weeks} weeks)")
    print(f"Regions: {', '.join(regions)}")
    print("\nSteps:")
    print("  1. Download SENAMHI climate data (temperature, precipitation)")
    print("  2. Download MODIS LST and NDVI (via NASA Earthdata)")
    print("  3. Download CDC Perú epidemiological notifications")
    print("  4. Download GEOBOSQUES deforestation alerts")
    print("  5. Reproject all rasters → EPSG:32718 @ 1 km")
    print("  6. Zonal statistics per district (1,874 distritos)")
    print("  7. Temporal alignment to ISO epidemiological weeks")
    print("  8. Feature engineering (lags, rolling windows, spatial lag)")
    print("  9. Great Expectations validation suite")
    print(" 10. Write feature Parquet to data/features/")


def _generate_synthetic_data(
    base: Path, regions: list[str], disease: str, start: date, end: date
) -> None:
    """Generates synthetic panel data for demo purposes (no API keys needed)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    # Fake UBIGEO codes for demo regions
    ubigeos = {"lima": ["150101", "150102", "150103"], "loreto": ["160101", "160102"]}
    selected = [u for r in regions for u in ubigeos.get(r.lower(), [])]
    if not selected:
        selected = ["150101"]

    weeks = pd.date_range(start, end, freq="W-MON")
    rows = []
    for ubigeo in selected:
        for week in weeks:
            rows.append({
                "ubigeo": ubigeo,
                "epi_week_start": week,
                "temp_mean_c": rng.normal(25, 3),
                "precip_mm": max(0, rng.normal(50, 30)),
                "humidity_pct": rng.uniform(60, 95),
                "ndvi_mean": rng.uniform(0.3, 0.8),
                "lst_day_c": rng.normal(30, 5),
                "surface_water_km2": max(0, rng.normal(5, 2)),
                "deforestation_ha": max(0, rng.normal(10, 8)),
                f"{disease}_cases": max(0, int(rng.poisson(20))),
                "population": 50000,
                "forest_cover_pct": rng.uniform(10, 80),
                "poverty_pct": rng.uniform(10, 60),
            })

    df = pd.DataFrame(rows)
    out = base / "raw" / "epidemiological" / "demo_panel.parquet"
    df.to_parquet(out, index=False)
    logger.info("Generated %d synthetic rows → %s", len(df), out)


def _build_demo_features(base: Path, regions: list[str], disease: str) -> None:
    import pandas as pd
    from src.features.feature_engineering import (
        add_climate_features,
        add_epi_features,
        add_landuse_features,
        add_satellite_features,
        compute_biorisk_index,
    )

    raw_file = base / "raw" / "epidemiological" / "demo_panel.parquet"
    if not raw_file.exists():
        logger.warning("Raw demo file not found.")
        return

    df = pd.read_parquet(raw_file)
    df = df.rename(columns={"epi_week_start": "epi_week_start"})
    df["date"] = df["epi_week_start"]

    df = add_climate_features(df)
    df = add_satellite_features(df)
    df = add_epi_features(df, case_cols={disease: f"{disease}_cases"})
    df = add_landuse_features(df)
    df = compute_biorisk_index(df)

    out = base / "features" / "demo_features.parquet"
    df.to_parquet(out, index=False)
    logger.info("Feature matrix written → %s (%d cols)", out, len(df.columns))


def _validate_demo_features(base: Path) -> None:
    import pandas as pd
    from src.etl.pipeline import validate_completeness, validate_value_ranges, PHYSICAL_RANGES

    feature_file = base / "features" / "demo_features.parquet"
    if not feature_file.exists():
        return
    df = pd.read_parquet(feature_file)
    nan_report = validate_completeness(df, max_nan_pct=0.20, raise_on_fail=False)
    violations = validate_value_ranges(df, PHYSICAL_RANGES, raise_on_fail=False)
    if violations:
        logger.warning("Validation warnings: %s", violations)
    else:
        logger.info("All validation checks passed.")


if __name__ == "__main__":
    main()
