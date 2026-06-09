"""
Peru BioRisk AI — Feature Engineering
Computes all model-ready features from processed district-level data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen  # type: ignore
from esda.moran import Moran_Local  # type: ignore


# ── Climate features ──────────────────────────────────────────────────────────

def add_climate_features(
    df: pd.DataFrame,
    temp_col: str = "temp_mean_c",
    precip_col: str = "precip_mm",
    humidity_col: str = "humidity_pct",
    lags_weeks: list[int] | None = None,
    precip_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Adds lagged climate variables and rolling window aggregates.
    Expects df sorted by (ubigeo, epi_week_start).
    """
    lags_weeks = lags_weeks or [1, 2, 4, 8]
    precip_windows = precip_windows or [15, 30, 60]

    df = df.copy().sort_values(["ubigeo", "epi_week_start"])
    grp = df.groupby("ubigeo")

    # Temperature lags
    for lag in lags_weeks:
        df[f"temp_mean_lag{lag}w"] = grp[temp_col].shift(lag)

    # Precipitation lags
    for lag in lags_weeks:
        df[f"precip_lag{lag}w"] = grp[precip_col].shift(lag)

    # Rolling precipitation sums (converted from weekly to daily approx)
    weeks_per_window = {d: max(1, d // 7) for d in precip_windows}
    for days, weeks in weeks_per_window.items():
        df[f"precip_cumul_{days}d"] = (
            grp[precip_col].transform(lambda x: x.rolling(weeks, min_periods=1).sum())
        )

    # Temperature anomaly (z-score vs historical mean per district-week)
    df["temp_anomaly_z"] = grp[temp_col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

    # Consecutive dry days (weekly proxy: weeks with precip < 5 mm)
    df["dry_week_flag"] = (df[precip_col] < 5).astype(int)
    df["dry_spell_weeks"] = grp["dry_week_flag"].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )

    # Vapour pressure deficit proxy (simplified Magnus formula)
    if humidity_col in df.columns:
        df["vpd_kpa"] = (
            0.6108 * np.exp(17.27 * df[temp_col] / (df[temp_col] + 237.3))
            * (1 - df[humidity_col] / 100)
        ).clip(lower=0)

    df = df.drop(columns=["dry_week_flag"])
    return df


# ── Satellite / environmental features ───────────────────────────────────────

def add_satellite_features(
    df: pd.DataFrame,
    ndvi_col: str = "ndvi_mean",
    lst_col: str = "lst_day_c",
    water_col: str = "surface_water_km2",
    defor_col: str = "deforestation_ha",
) -> pd.DataFrame:
    """
    Adds NDVI trends, surface water stats, deforestation accumulation.
    """
    df = df.copy().sort_values(["ubigeo", "epi_week_start"])
    grp = df.groupby("ubigeo")

    if ndvi_col in df.columns:
        # 4-week NDVI trend (linear slope via rolling diff)
        df["ndvi_trend_4w"] = grp[ndvi_col].transform(
            lambda x: x.rolling(4, min_periods=2).apply(
                lambda w: np.polyfit(range(len(w)), w, 1)[0] if len(w) > 1 else 0,
                raw=True,
            )
        )
        df["ndvi_anomaly_z"] = grp[ndvi_col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

    if lst_col in df.columns:
        # Urban heat island index: LST anomaly within urban vs rural proxy
        df["lst_anomaly_z"] = grp[lst_col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

    if water_col in df.columns:
        df["water_area_lag4w"] = grp[water_col].shift(4)
        df["water_area_change_pct"] = (
            grp[water_col].pct_change(periods=4).clip(-1, 10) * 100
        )

    if defor_col in df.columns:
        # Cumulative deforestation over 6 months
        df["defor_cumul_24w"] = grp[defor_col].transform(
            lambda x: x.rolling(24, min_periods=1).sum()
        )

    return df


# ── Epidemiological (auto-regressive) features ────────────────────────────────

def add_epi_features(
    df: pd.DataFrame,
    case_cols: dict[str, str] | None = None,
    lags_weeks: list[int] | None = None,
    incidence_pop_col: str = "population",
) -> pd.DataFrame:
    """
    Adds lagged case counts and incidence rates per 100k.
    case_cols = {"dengue": "dengue_cases", "malaria": "malaria_cases", ...}
    """
    case_cols = case_cols or {
        "dengue": "dengue_cases",
        "malaria": "malaria_cases",
        "lepto": "leptospirosis_cases",
    }
    lags_weeks = lags_weeks or [1, 2, 4, 8]

    df = df.copy().sort_values(["ubigeo", "epi_week_start"])
    grp = df.groupby("ubigeo")

    for disease, col in case_cols.items():
        if col not in df.columns:
            continue
        for lag in lags_weeks:
            df[f"{disease}_cases_lag{lag}w"] = grp[col].shift(lag)

        if incidence_pop_col in df.columns:
            df[f"{disease}_incidence_100k"] = (
                df[col] / df[incidence_pop_col] * 100_000
            ).clip(lower=0)
            df[f"{disease}_incidence_lag4w"] = grp[
                f"{disease}_incidence_100k"
            ].shift(4)

        # 4-week rolling mean (smoothed signal)
        df[f"{disease}_cases_ma4w"] = grp[col].transform(
            lambda x: x.rolling(4, min_periods=1).mean()
        )

    return df


# ── Land-use features ─────────────────────────────────────────────────────────

def add_landuse_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives ecological edge density, interface exposure, and
    healthcare access composite.
    """
    df = df.copy()

    required = ["forest_cover_pct", "agri_pct", "mining_pct"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # Forest-agriculture interface exposure (proxy for eco-edge risk)
    df["eco_edge_index"] = (
        df["forest_cover_pct"] * df["agri_pct"] / 100
    ).clip(0, 100)

    # Anthropogenic pressure index
    df["anthropogenic_pressure"] = (
        df.get("agri_pct", 0) * 0.4
        + df.get("mining_pct", 0) * 0.4
        + (100 - df.get("forest_cover_pct", 100)).clip(0, 100) * 0.2
    )

    # Social vulnerability composite (if columns present)
    soc_cols = {
        "poverty_pct": 0.35,
        "water_access_hh_pct": -0.30,  # negative: more access = less vulnerability
        "literacy_pct": -0.20,
        "dist_healthcare_km": 0.15,
    }
    vuln_score = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    for col, w in soc_cols.items():
        if col in df.columns:
            normalised = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
            vuln_score += w * normalised
            total_weight += abs(w)
    if total_weight > 0:
        df["social_vulnerability"] = (vuln_score / total_weight).clip(-3, 3)

    return df


# ── Spatial features ──────────────────────────────────────────────────────────

def add_spatial_features(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    case_col: str = "dengue_cases",
    ubigeo_col: str = "ubigeo",
) -> pd.DataFrame:
    """
    Computes spatial lag and Local Moran's I for a given case column.
    gdf must be aligned (same rows/order) with df.
    """
    df = df.copy()

    # Build Queen contiguity weights
    try:
        w = Queen.from_dataframe(gdf)
        w.transform = "r"  # row-standardise

        if case_col in df.columns:
            case_vals = df[case_col].fillna(0).values

            # Spatial lag (weighted mean of neighbours)
            from libpysal.weights import lag_spatial  # type: ignore
            df[f"{case_col}_spatial_lag"] = lag_spatial(w, case_vals)

            # Local Moran's I (hotspot/coldspot)
            moran_loc = Moran_Local(case_vals, w, permutations=99, seed=42)
            df["moran_local_I"] = moran_loc.Is
            df["moran_quadrant"] = moran_loc.q  # 1=HH,2=LH,3=LL,4=HL
            df["moran_significant"] = (moran_loc.p_sim < 0.05).astype(int)

    except Exception as exc:
        import warnings
        warnings.warn(f"Spatial features failed (likely non-contiguous geometry): {exc}")

    return df


# ── BioRisk composite index ───────────────────────────────────────────────────

def compute_biorisk_index(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Computes the composite BioRisk Index from normalised sub-scores.
    Default weights: climate 0.30, habitat 0.25, epi 0.25, social 0.20.
    """
    weights = weights or {
        "climate_score": 0.30,
        "habitat_score": 0.25,
        "epi_score": 0.25,
        "social_score": 0.20,
    }
    df = df.copy()

    def _normalise(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(0.02), s.quantile(0.98)
        return ((s - lo) / (hi - lo + 1e-9)).clip(0, 1)

    # Climate score: anomalous heat + cumulative precipitation
    climate_inputs = [c for c in ["temp_anomaly_z", "precip_cumul_30d", "vpd_kpa"] if c in df.columns]
    if climate_inputs:
        df["climate_score"] = _normalise(df[climate_inputs].mean(axis=1))

    # Habitat score: standing water + deforestation + NDVI anomaly
    habitat_inputs = [c for c in ["water_area_lag4w", "defor_cumul_24w", "eco_edge_index"] if c in df.columns]
    if habitat_inputs:
        df["habitat_score"] = _normalise(df[habitat_inputs].mean(axis=1))

    # Epi score: recent case burden
    epi_inputs = [c for c in df.columns if c.endswith("_incidence_100k") or c.endswith("_ma4w")]
    if epi_inputs:
        df["epi_score"] = _normalise(df[epi_inputs].mean(axis=1))

    # Social score: vulnerability
    if "social_vulnerability" in df.columns:
        df["social_score"] = _normalise(df["social_vulnerability"])

    available_weights = {k: v for k, v in weights.items() if k in df.columns}
    total = sum(available_weights.values())
    if total > 0:
        df["biorisk_index"] = sum(
            (w / total) * df[k] for k, w in available_weights.items()
        )
    else:
        df["biorisk_index"] = np.nan

    return df


# ── Master feature builder ─────────────────────────────────────────────────────

def build_feature_matrix(
    district_ts: pd.DataFrame,
    districts_gdf: gpd.GeoDataFrame | None = None,
    lags_weeks: list[int] | None = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    district_ts: panel data (ubigeo × epi_week), wide format.
    Returns enriched DataFrame ready for ML training.
    """
    df = district_ts.copy()

    df = add_climate_features(df, lags_weeks=lags_weeks)
    df = add_satellite_features(df)
    df = add_epi_features(df, lags_weeks=lags_weeks)
    df = add_landuse_features(df)

    if districts_gdf is not None:
        # Merge geometry for spatial feature computation
        df_with_geom = districts_gdf[["ubigeo", "geometry"]].merge(df, on="ubigeo")
        df = add_spatial_features(df, df_with_geom)

    df = compute_biorisk_index(df)

    # Drop rows where all target lags are NaN (beginning of series)
    min_lag = min(lags_weeks or [1])
    df = df.iloc[min_lag:].reset_index(drop=True)

    return df
