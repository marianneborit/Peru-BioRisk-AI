"""
Peru BioRisk AI — ETL Pipeline
Orchestrates extraction, reprojection, validation, and loading
for all data domains.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import mapping

logger = logging.getLogger(__name__)

Disease = Literal["dengue", "malaria", "leptospirosis", "leishmaniasis", "bartonellosis"]
DataDomain = Literal["climatic", "satellite", "epidemiological", "landuse", "socioeconomic"]

TARGET_CRS = CRS.from_epsg(32718)   # WGS 84 / UTM zone 18S
TARGET_RES_M = 1000                  # 1 km grid
PERU_BBOX = (-81.5, -18.5, -68.5, 0.5)  # lon_min, lat_min, lon_max, lat_max


@dataclass
class ETLConfig:
    start_date: date
    end_date: date
    domains: list[DataDomain] = field(default_factory=lambda: [
        "climatic", "satellite", "epidemiological", "landuse", "socioeconomic"
    ])
    diseases: list[Disease] = field(default_factory=lambda: [
        "dengue", "malaria", "leptospirosis"
    ])
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    features_dir: Path = Path("data/features")
    n_workers: int = 4


# ── Reprojection ──────────────────────────────────────────────────────────────

def reproject_raster(
    src_path: Path,
    dst_path: Path,
    target_crs: CRS = TARGET_CRS,
    resolution_m: int = TARGET_RES_M,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """
    Reprojects a raster to the target CRS and resolution.
    Returns dst_path.
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height,
            *src.bounds,
            resolution=resolution_m,
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": -9999,
            "dtype": "float32",
        })

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )
    logger.info("Reprojected %s → %s", src_path.name, dst_path.name)
    return dst_path


def reproject_vector(
    src_path: Path,
    dst_path: Path,
    target_crs: CRS = TARGET_CRS,
) -> gpd.GeoDataFrame:
    """Reprojects a vector file to target CRS."""
    gdf = gpd.read_file(src_path)
    gdf_reprojected = gdf.to_crs(target_crs)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_reprojected.to_file(dst_path, driver="GeoJSON")
    logger.info("Reprojected vector %s → %s", src_path.name, dst_path.name)
    return gdf_reprojected


# ── Temporal alignment ────────────────────────────────────────────────────────

def align_to_epi_week(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Aligns a DataFrame to ISO 8601 epidemiological weeks.
    Adds columns: epi_year, epi_week, epi_week_start.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["epi_year"] = df[date_col].dt.isocalendar().year.astype(int)
    df["epi_week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["epi_week_start"] = df[date_col] - pd.to_timedelta(
        df[date_col].dt.dayofweek, unit="D"
    )
    return df


# ── Missing value imputation ──────────────────────────────────────────────────

def impute_time_series(
    df: pd.DataFrame,
    method: Literal["mice", "knn", "forward_fill"] = "forward_fill",
    max_gap_weeks: int = 4,
) -> pd.DataFrame:
    """
    Imputes missing values in epidemiological time series.
    Gaps larger than max_gap_weeks are left as NaN.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if method == "forward_fill":
        df[numeric_cols] = (
            df[numeric_cols]
            .ffill(limit=max_gap_weeks)
            .bfill(limit=1)
        )

    elif method == "knn":
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    elif method == "mice":
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(max_iter=10, random_state=42)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        except ImportError:
            logger.warning("MICE imputer unavailable, falling back to forward_fill")
            df[numeric_cols] = df[numeric_cols].ffill(limit=max_gap_weeks)

    nan_pct = df[numeric_cols].isna().mean().mean() * 100
    logger.info("Post-imputation NaN: %.2f%%", nan_pct)
    return df


# ── Raster → district aggregation ────────────────────────────────────────────

def raster_to_district_stats(
    raster_path: Path,
    districts_gdf: gpd.GeoDataFrame,
    ubigeo_col: str = "ubigeo",
    stats: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extracts zonal statistics from a raster for each district polygon.
    Returns a DataFrame with one row per district.
    """
    import rasterstats  # type: ignore

    stats = stats or ["mean", "std", "min", "max", "count"]

    zs = rasterstats.zonal_stats(
        districts_gdf.geometry,
        str(raster_path),
        stats=stats,
        nodata=-9999,
        geojson_out=False,
    )
    result = pd.DataFrame(zs)
    result.insert(0, ubigeo_col, districts_gdf[ubigeo_col].values)
    return result


# ── Validation helpers ────────────────────────────────────────────────────────

def validate_completeness(
    df: pd.DataFrame,
    max_nan_pct: float = 0.05,
    raise_on_fail: bool = True,
) -> dict[str, float]:
    """
    Checks that no column exceeds max_nan_pct missing values.
    Returns per-column NaN percentages.
    """
    nan_pcts = df.isna().mean()
    failures = nan_pcts[nan_pcts > max_nan_pct]

    if not failures.empty:
        msg = f"Columns exceed {max_nan_pct:.0%} NaN threshold: {failures.to_dict()}"
        if raise_on_fail:
            raise ValueError(msg)
        logger.warning(msg)

    return nan_pcts.to_dict()


def validate_value_ranges(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]],
    raise_on_fail: bool = True,
) -> list[str]:
    """
    Checks that numeric columns fall within expected physical ranges.
    ranges = {"temp_mean": (-10, 45), "precip_mm": (0, 1000), ...}
    """
    violations: list[str] = []
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            violations.append(f"{col}: {out_of_range} values outside [{lo}, {hi}]")

    if violations and raise_on_fail:
        raise ValueError("Range violations: " + "; ".join(violations))
    return violations


PHYSICAL_RANGES: dict[str, tuple[float, float]] = {
    "temp_mean_c": (-10.0, 45.0),
    "temp_max_c": (-5.0, 50.0),
    "temp_min_c": (-20.0, 40.0),
    "precip_mm": (0.0, 1500.0),
    "humidity_pct": (0.0, 100.0),
    "ndvi": (-0.2, 1.0),
    "lst_day_c": (-5.0, 65.0),
    "dengue_cases": (0.0, 50000.0),
    "malaria_cases": (0.0, 20000.0),
}


# ── Main ETL runner ───────────────────────────────────────────────────────────

def run_etl(config: ETLConfig) -> None:
    """
    Top-level ETL orchestrator. Called by Airflow DAG or CLI.
    """
    logger.info(
        "Starting ETL | %s → %s | domains: %s",
        config.start_date, config.end_date, config.domains,
    )

    for domain in config.domains:
        logger.info("Processing domain: %s", domain)
        # Domain-specific extractors are called here.
        # Each returns raw files to config.raw_dir/<domain>/
        _dispatch_extractor(domain, config)

    logger.info("Reprojecting rasters to EPSG:32718 @ 1 km")
    _reproject_all(config)

    logger.info("Aggregating rasters to district level")
    _aggregate_to_districts(config)

    logger.info("Aligning to epidemiological weeks")
    _temporal_align(config)

    logger.info("Running validation suite")
    _validate_all(config)

    logger.info("ETL complete.")


def _dispatch_extractor(domain: DataDomain, config: ETLConfig) -> None:
    """Lazy import and run the appropriate extractor module."""
    from src.ingestion import (  # type: ignore
        cdc_peru, era5, geobosques, modis, senamhi,
    )
    extractors = {
        "climatic": [senamhi.extract, era5.extract],
        "satellite": [modis.extract],
        "epidemiological": [cdc_peru.extract],
        "landuse": [geobosques.extract],
        "socioeconomic": [],  # static — loaded once at DB init
    }
    for fn in extractors.get(domain, []):
        fn(config.start_date, config.end_date, config.raw_dir / domain)


def _reproject_all(config: ETLConfig) -> None:
    for raster_path in (config.raw_dir).rglob("*.tif"):
        dst = config.processed_dir / raster_path.relative_to(config.raw_dir)
        dst = dst.with_suffix(".tif")
        if not dst.exists():
            reproject_raster(raster_path, dst)


def _aggregate_to_districts(config: ETLConfig) -> None:
    districts_path = config.raw_dir / "boundaries" / "peru_districts.gpkg"
    if not districts_path.exists():
        logger.warning("District boundaries not found at %s", districts_path)
        return
    districts = gpd.read_file(districts_path).to_crs(TARGET_CRS)
    for raster_path in (config.processed_dir).rglob("*.tif"):
        stats_df = raster_to_district_stats(raster_path, districts)
        out = config.features_dir / raster_path.stem / "district_stats.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_parquet(out, index=False)


def _temporal_align(config: ETLConfig) -> None:
    for parquet_path in (config.features_dir).rglob("*.parquet"):
        df = pd.read_parquet(parquet_path)
        if "date" in df.columns:
            df = align_to_epi_week(df)
            df.to_parquet(parquet_path, index=False)


def _validate_all(config: ETLConfig) -> None:
    for parquet_path in (config.features_dir).rglob("*.parquet"):
        df = pd.read_parquet(parquet_path)
        validate_completeness(df, raise_on_fail=False)
        validate_value_ranges(df, PHYSICAL_RANGES, raise_on_fail=False)
