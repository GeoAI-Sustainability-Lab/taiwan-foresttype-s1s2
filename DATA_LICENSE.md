# Data licensing and attribution

The tables in `data/` are **derived products**. They are not original data. Each was computed from
openly licensed sources, listed below with the attribution their licence requires.

We do not, and cannot, place a new licence over the underlying data. Where a source licence carries
obligations, those obligations follow the derived tables as well. Anyone reusing `data/` should
carry the same attributions forward.

## Sources

### Forest-type reference labels

Applies to the `cls` column of `features_train4yr.csv`, `features_test1yr_2025-2026.csv`, `folds.csv`
and the tables under `data/analyses/` and `data/archive_v1_35features/`, and to `train_points.csv`
(point coordinates and class).

**第四次森林資源調查全島森林林型分布圖** (Fourth National Forest Resource Inventory, island-wide
forest-type distribution map), 農業部林業及自然保育署 / Forestry and Nature Conservation Agency,
Ministry of Agriculture, Taiwan.

- Published on the Taiwan open-data portal: https://data.gov.tw/dataset/57873
- Licence: **政府資料開放授權條款－第 1 版** (Open Government Data License, Taiwan, version 1.0),
  https://data.gov.tw/licenses

OGDL v1 permits reproduction, distribution, public transmission and adaptation, including for
commercial purposes, provided the source is acknowledged. The class labels here were obtained by
drawing random points inside the pure broadleaf, conifer and bamboo polygons of that map, so they
are an adaptation of it and the attribution above must be retained.

### Optical and radar features (`NDVI_*`, `NDMI_*`, `NBR_*`, `reNDVI_*`, `VV_*`, `VH_*`, `RATIO_*`, `RVI_*`)

**Copernicus Sentinel-2** (optical) and **Copernicus Sentinel-1** (SAR), European Space Agency.

- Licence: Legal Notice on the use of Copernicus Sentinel Data and Service Information
- Required acknowledgement: *"Contains modified Copernicus Sentinel data 2021–2026"*

### Climate features (`temp_*`, `dewp_*`, `dewdep`, `precip_*`, `soilm_*`)

**ERA5-Land**, produced by ECMWF and distributed through the Copernicus Climate Change Service (C3S).

- Neither the European Commission nor ECMWF is responsible for any use of this Copernicus
  information or of the data it contains.

### Terrain features (`elev`, `slope`, `aspect_cos`)

**SRTM Digital Elevation Data v3**, NASA / USGS. Public domain.

### Cloud statistics (`cloud_monthly_stats_5yr.csv`)

Computed from Sentinel-2 scene metadata (cloud fraction). Same Copernicus terms as above.

### Our own outputs (`feature_importance_5fold.csv`, `misclassified_points.csv`, `models/`)

Produced by the code in this repository and released under **CC BY 4.0**, subject to the
attributions above for the inputs they were computed from.

## Code

Code outside `data/` and `models/` is released under the MIT License, see `LICENSE`.

## When reusing this repository

1. Cite the Fourth National Forest Resource Inventory forest-type map (FANCA, Taiwan) as the source
   of the class labels.
2. State *"Contains modified Copernicus Sentinel data"* with the years used.
3. Acknowledge ERA5-Land / Copernicus Climate Change Service and include the ECMWF disclaimer.
4. SRTM requires no attribution, though crediting NASA/USGS is customary.
