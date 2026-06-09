# Data Dictionary — Peru BioRisk AI

All processed features are projected to **EPSG:32718** (WGS 84 / UTM zone 18S)
at **1 km spatial resolution**, aligned to **ISO 8601 epidemiological weeks**.

---

## Primary keys

| Column | Type | Description |
|---|---|---|
| `ubigeo` | STRING(6) | INEI district code — uniquely identifies 1 of 1,874 distritos |
| `epi_year` | INT | ISO calendar year |
| `epi_week` | INT | ISO week number (1–53) |
| `epi_week_start` | DATE | Monday of the epidemiological week |

---

## Climatic features

| Feature | Unit | Source | Lag options |
|---|---|---|---|
| `temp_mean_c` | °C | SENAMHI / ERA5 | — |
| `temp_max_c` | °C | SENAMHI / ERA5 | — |
| `temp_min_c` | °C | SENAMHI / ERA5 | — |
| `temp_mean_lag{N}w` | °C | Derived | N ∈ {1,2,4,8} |
| `temp_anomaly_z` | σ | Derived | — |
| `precip_mm` | mm/week | CHIRPS / GPM | — |
| `precip_lag{N}w` | mm | Derived | N ∈ {1,2,4,8} |
| `precip_cumul_{D}d` | mm | Derived | D ∈ {15,30,60} |
| `humidity_pct` | % | ERA5 | — |
| `vpd_kpa` | kPa | Derived (Magnus) | — |
| `dry_spell_weeks` | weeks | Derived | — |

---

## Satellite / environmental features

| Feature | Unit | Source | Notes |
|---|---|---|---|
| `ndvi_mean` | — | MODIS MOD13A2 | District mean, 16-day composite |
| `ndvi_trend_4w` | Δ/week | Derived | Linear slope over 4-week window |
| `ndvi_anomaly_z` | σ | Derived | vs. district historical mean |
| `lst_day_c` | °C | MODIS MOD11A1 | Daytime land surface temperature |
| `lst_anomaly_z` | σ | Derived | — |
| `surface_water_km2` | km² | JRC/GEE | Monthly permanent + seasonal |
| `water_area_lag4w` | km² | Derived | 4-week lag |
| `water_area_change_pct` | % | Derived | Change vs 4 weeks prior |
| `deforestation_ha` | ha/week | GEOBOSQUES | MINAM alert-based |
| `defor_cumul_24w` | ha | Derived | 6-month rolling sum |

---

## Epidemiological features

| Feature | Unit | Source | Notes |
|---|---|---|---|
| `dengue_cases` | count | CDC Perú / NOTI | Confirmed + probable |
| `malaria_cases` | count | CDC Perú / NOTI | P. vivax + P. falciparum |
| `leptospirosis_cases` | count | CDC Perú / NOTI | — |
| `leishmaniasis_cases` | count | CDC Perú / NOTI | Cutaneous + mucocutaneous |
| `bartonellosis_cases` | count | CDC Perú / NOTI | — |
| `{disease}_cases_lag{N}w` | count | Derived | N ∈ {1,2,4,8} |
| `{disease}_incidence_100k` | per 100k | Derived | cases / population × 100,000 |
| `{disease}_cases_ma4w` | count | Derived | 4-week moving average |

---

## Land-use features

| Feature | Unit | Source | Notes |
|---|---|---|---|
| `forest_cover_pct` | % | MapBiomas Perú | Annual, interpolated weekly |
| `agri_pct` | % | MapBiomas Perú | Agriculture fraction |
| `mining_pct` | % | MapBiomas Perú | Mining / bare soil fraction |
| `urban_pct` | % | MapBiomas Perú | Urban cover fraction |
| `eco_edge_index` | 0–100 | Derived | forest_pct × agri_pct / 100 |
| `anthropogenic_pressure` | 0–100 | Derived | Weighted composite |
| `dist_river_km` | km | DEM + OSM | Distance to nearest water body |
| `dist_healthcare_km` | km | OSM / MINSA | Distance to nearest health post |
| `elevation_m` | m | SRTM 30 m | District median elevation |
| `slope_deg` | ° | SRTM derived | District median slope |

---

## Socioeconomic features

| Feature | Unit | Source | Notes |
|---|---|---|---|
| `population` | persons | INEI 2017 | Interpolated with growth rate |
| `poverty_pct` | % | INEI ENAHO | Monetary poverty rate |
| `water_access_hh_pct` | % | INEI 2017 | Households with piped water |
| `literacy_pct` | % | INEI 2017 | Adult literacy rate |
| `social_vulnerability` | σ | Derived | Normalised composite |

---

## Spatial features

| Feature | Description |
|---|---|
| `{disease}_spatial_lag` | Queen-weighted mean of case counts in contiguous districts |
| `moran_local_I` | Local Moran's I statistic |
| `moran_quadrant` | 1=HH, 2=LH, 3=LL, 4=HL (hotspot classification) |
| `moran_significant` | 1 if p < 0.05 (99 permutations) |

---

## Composite indices

| Index | Formula | Range |
|---|---|---|
| `biorisk_index` | 0.30·climate + 0.25·habitat + 0.25·epi + 0.20·social | 0–1 |
| `eco_edge_index` | forest_cover_pct × agri_pct / 100 | 0–100 |
| `anthropogenic_pressure` | 0.4·agri + 0.4·mining + 0.2·(100−forest) | 0–100 |
| `social_vulnerability` | Weighted PCA-normalised composite | σ units |

---

## Target variables

| Variable | Type | Description |
|---|---|---|
| `outbreak_label` | BOOL | 1 if cases ≥ district 90th percentile in horizon week |
| `cases_t{H}` | INT | Case count at horizon H weeks ahead |
| `incidence_t{H}` | FLOAT | Incidence rate per 100k at horizon H weeks ahead |

---

## Missing data policy

| Scenario | Strategy |
|---|---|
| Climate: gap ≤ 4 weeks | Forward-fill then MICE imputation |
| Climate: gap > 4 weeks | Set to NaN; flag `{col}_imputed = 1` |
| Epi: gap ≤ 2 weeks | Forward-fill (reporting delay) |
| Epi: gap > 2 weeks | Set to NaN; exclude from training |
| Satellite: cloud cover > 50% | Use 16-day composite fallback |
| Socioeconomic | Annual interpolation from census years |
