# Data dictionary

## features_train4yr.csv / features_test1yr_2025-2026.csv  (755 points × 34 features + ptid, cls, lon, lat, blk, fold)

Row-aligned: the same 755 reference points in both files (`ptid` 0–754). `features_train4yr.csv` holds the
2021–2024 features (training window), `features_test1yr_2025-2026.csv` the 2025-06 to 2026-05 features
(independent year). Keep the column order: the Random Forest draws its feature subsets by column index,
so reordering the columns changes the fitted model.

- `cls`: 1 = broadleaf, 2 = conifer, 3 = bamboo (Fourth National Forest Resource Inventory forest-type map).
- `lon`, `lat`: WGS84 coordinates of the point.
- `blk`: spatial block = graticule cell of 0.1° longitude × 0.09° latitude (`floor(lon/0.1)_floor(lat/0.09)`), 204 cells.
- `fold`: fold of the deterministic `GroupKFold(5)` on `blk` (also in `folds.csv`).
- Optical phenology (Sentinel-2, 12): NDVI, NDMI, NBR and reNDVI, each `_mean` (collection mean), `_amp` and `_pha`
  (first-harmonic amplitude and phase, t in years since 2000-01-01).
- SAR (Sentinel-1 IW ascending, 9): `VV_mean`, `VH_mean`, `RATIO_mean` (VH − VV, dB), `RATIO_std`, `VH_amp`, `VH_pha`,
  `VV_wetdry`, `VH_wetdry`, `RATIO_wetdry` (wet season May–October minus dry season November–April).
- Climate (ERA5-Land monthly, 10): `temp_mean`, `dewp_mean`, `dewdep`, `soilm_mean`, `soilm_wet`, `soilm_dry`,
  `soilm_wetdry`, `precip_mm`, `precip_wet`, `precip_dry`.
- Terrain (SRTM, 3): `elev`, `slope`, `aspect_cos` = cos(aspect in radians).

## folds.csv  ptid, cls, lon, lat, blk, fold, the fold assignment used in every result.
## train_points.csv  lon, lat (WGS84), cls of the 780 points drawn; 755 were retained after removing points with missing features.
## cloud_monthly_stats_5yr.csv  monthly Sentinel-2 cloud statistics: mean_cloud, n_clear20, n_scenes, month, year.

## analyses/  inputs of the further analyses (`src/analyses/`)
- `features_train4yr_v2.csv`, `features_test1yr_2025-2026_v2.csv`: the same 34-feature tables in the layout read by the analysis scripts.
- `texture_lband.csv`: the eleven multi-scale texture features (GLCM entropy, homogeneity, contrast, variance; focal standard deviations of NDVI and NDRE) per point (`ptid`).
- `data.npz`: `Xc` (755 × 12 × 16) monthly sequences of the 2021–2024 window, `Xi` the 2025-06 to 2026-05 window (position k = month (5 + k) mod 12), `channels` (ten Sentinel-2 bands, VV, VH, temperature, precipitation, elevation, slope), `Ec`/`Ei` the frozen 128-dimensional Presto embeddings, `y`, `blk`.
- `presto_Ec.npy`, `presto_Ei.npy`: the Presto embeddings as separate arrays; `dino_emb.npy`: DINOv2 embeddings of the 10 m chips.
- `s2_monthly_multiindex.csv`, `s2_biennial_features.csv`: monthly Sentinel-2 red-edge indices 2018–2025 and the multi-year (biennial) descriptors.
- `landsat_annual_NDVI.csv`: annual Landsat NDVI 2005–2020 per point (periodicity test); `landsat_eraA_2008_2015.csv`, `landsat_eraB_2021_2025.csv`: the eleven Landsat features of the two eras.
- `visual_check_50pts.csv`: per-point records of the recent-image check: stratum, image date, agreement at the pixel scale (`px_agree` Y/N/U), confidence, current cover, stand edge within 30 m.
- `local_selection.json`: the 14-feature locally selected set.

## archive_v1_35features/  the earlier 35-feature tables of v1.0.0 (incl. `RVI_mean`; `aspect_cos` as archived), kept for the record.
