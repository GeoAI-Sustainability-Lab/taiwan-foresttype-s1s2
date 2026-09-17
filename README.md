# Broadleaf, conifer and bamboo forest-type classification across Taiwan (Sentinel-1/2)

Open dataset, trained model and reproducible pipeline for the manuscript
*Classification of Broadleaf, Conifer and Bamboo Forests Across Taiwan by Multi-Source Remote Sensing
and Interpretable Machine Learning* (IEEE JSTARS, manuscript JSTARS-2026-04232, revised version).

Three forest types (broadleaf, conifer, bamboo) are classified across Taiwan from Sentinel-2 optical
phenology, Sentinel-1 SAR, ERA5-Land climate and SRTM terrain — **34 engineered features**, a Random
Forest under spatial-block cross-validation on 0.1° × 0.09° graticule cells, Shapley-value attribution
and accumulated local effects, and evaluation on a spatio-temporally independent year.

**Version 1.1.0 (revision R1)** — all results rest on 34 features (the radar vegetation index of the
submitted version is withdrawn; the aspect cosine is cos(aspect in radians)). The tables and scripts of
the submitted version (35 features) are kept under `data/archive_v1_submission/` and git tag `v1.0.0`.
See `CHANGELOG.md`.

## Reproduce

```bash
pip install -r requirements.txt        # locked versions: scikit-learn 1.7.2, numpy 2.2.6, ...
python src/train_eval.py               # Table 1: spatial-block CV + independent year + source ablation
python src/explain_shap_ale.py         # global SHAP, ALE, prediction entropy
python src/make_figures.py             # confusion / ablation / cloud-climatology figures
```

`src/train_eval.py` reproduces the following exactly (deterministic split and seed; the feature
**column order** of `data/features_train4yr.csv` must be kept, because the Random Forest draws its
feature subsets by column index).

| Feature set (dimension) | Same window OA / κ | Next year OA / κ | Next-year macro-F1 | Recall Bl / Co / Ba |
|---|---|---|---|---|
| Full (34) — reference model | 0.889 / 0.833 | **0.868 / 0.801** | 0.866 | 0.770 / 0.958 / 0.869 |
| Optical + environment (25) | 0.890 / 0.835 | 0.870 / 0.805 | 0.869 | 0.774 / 0.962 / 0.869 |
| SAR + environment (22), no optical input | 0.877 / 0.815 | 0.823 / 0.734 | 0.820 | 0.716 / 0.954 / 0.790 |
| Environment (13) | 0.869 / 0.803 | 0.816 / 0.723 | 0.813 | 0.650 / 0.954 / 0.833 |
| Optical only (12) | 0.811 / 0.716 | 0.784 / 0.676 | 0.782 | 0.728 / 0.873 / 0.746 |
| Optical + SAR (21) | 0.809 / 0.714 | 0.796 / 0.694 | 0.794 | 0.728 / 0.892 / 0.762 |
| SAR only (9) | 0.567 / 0.350 | 0.462 / 0.195 | 0.459 | 0.527 / 0.523 / 0.337 |

Full model, next year: 95 % block-bootstrap interval of OA 0.829–0.901; confusion matrix
(rows = reference broadleaf, conifer, bamboo) `[[187, 4, 52], [0, 249, 11], [26, 7, 219]]`.
Adding the eleven multi-scale texture features gives 0.883 (exploratory result, see the manuscript).

**Protocol.** Every accuracy is measured on data held out from training. Spatial hold-out is
`GroupKFold(5)` on the graticule cell `blk` (204 cells, about 10.2 km × 9.95 km), so neighbouring
polygons never straddle the train/test boundary. *Same window*: fit on the training cells of
2021–2024, predict the held-out cells of the same window. *Next year*: fit on the 2021–2024 features
of the training cells, predict the held-out cells with the 2025-06 to 2026-05 features — held out in
space and in time. Classifier: Random Forest, 300 trees, `min_samples_leaf=3`, balanced class
weights, seed 0. The fold of every point is stored in the `fold` column and in `data/folds.csv`.

## Revision analyses (`src/revision_R1/`, inputs in `data/R1/`, outputs in `results/R1/`)

| Script | Manuscript item | Output |
|---|---|---|
| `canonical_v3.py` | Table 1, texture gain with exact McNemar test and paired block-bootstrap interval, uncertainty / selective accuracy, two-class model of Table 3, XGBoost row of Table 2 | `canonical_v3.json`, `preds_v3.npz` |
| `canonical_v3_B.py` | label-efficiency curve under the block protocol (Fig. 5b), sample-size curve, label-noise sensitivity over ten seeds (Section III-F) | `canonical_v3_B.json` |
| `canonical_v3_C.py` | feature selection nested within the folds (Table S5), SHAP rankings | `canonical_v3_C.json` |
| `canonical_v3_D.py` | broadleaf–bamboo enhancement strategies (Table S3), Presto heads, multi-year biennial descriptors | `canonical_v3_D.json`, `preds_v3_D.npz` |
| `rerun_deep_aligned.py`, `canonical_v2_D1.py`, `canonical_v2_D2.py` | TempCNN, BiLSTM, Transformer and LTAE with calendar-aligned months, fold-internal standardisation, three seeds (Table 2, Fig. 5a); optical-month masking and sequence length (Fig. 5c, Section III-C); Transformer label efficiency (Fig. 5b) — require PyTorch | `rev_deep_aligned*.json`, `canonical_v2_D1.json`, `canonical_v2_D2.json` |
| `canonical_v2_E_periodicity.py` | period-two test of the 2005–2020 annual NDVI series (Section IV-B) | `canonical_v2_E_periodicity.json` |
| `visual_check_estimate.py` | recent-image check of 50 reference points (Section III-F, Table S4) | `visual_check_estimates.json` |

The JSON files in `results/R1/` are the outputs as run for the manuscript; running the scripts
regenerates them (the deep-model scripts take hours on CPU). `res_landsat_era.json` (two-era Landsat
comparison of Section III-F) and `rev_texture_class_medians.json` (Section IV-B) are included as run.

## Repository layout

```
data/       features_train4yr.csv, features_test1yr_2025-2026.csv (34 features + cls, lon, lat, blk, fold),
            folds.csv, train_points.csv, cloud_monthly_stats_5yr.csv — see data/DATA_DICTIONARY.md
data/R1/    inputs of the revision analyses (texture features, monthly sequences, embeddings,
            Landsat annual series, multi-year Sentinel-2 indices, recent-image check records)
data/archive_v1_submission/   tables of the submitted version (35 features), kept for the record
src/        train_eval.py, predict.py, explain_shap_ale.py, make_figures.py, boundary_dissolve.py
src/revision_R1/   scripts of the revision analyses (table above)
results/    table1.json (from train_eval.py) and results/R1/ (revision analyses as run)
gee/        Google Earth Engine scripts: 05_feature_exporter_34.js (feature construction and export),
            04_island_prior_weighted_20m.js (island-wide 20 m map of Fig. 6), 06 two-era Landsat features,
            07 multi-year Sentinel-2 indices, 08 recent-image check viewer, and the original scripts 01–03
models/     trained Random Forest rf_full.joblib (34 features) + rf_full_meta.json + model_card.md
figures/    figures of the revised manuscript produced by the pipeline
```

## Nationwide map (Google Earth Engine)

`gee/04_island_prior_weighted_20m.js` produces the island-wide 20 m map shown in Fig. 6 of the manuscript
(`taiwan_foresttype_20m_cal`). It is self-contained, including the training points inline, and re-weights
the class probabilities by their natural area priors before taking the arg-max. The export carries no
forest mask of its own: apply the Fourth National Forest Inventory forest mask when reading it, otherwise
every non-forest pixel defaults to broadleaf. Because the shares of a prior-weighted map are not comparable
with the balanced-class point accuracies, the map is a spatially explicit product rather than an area
estimate, and the mapped bamboo area is a lower bound.

## Citing

Please cite the manuscript and, for the code and data, the Zenodo record of this repository
(see `CITATION.cff`; a DOI is minted for every GitHub release).

## Data sources and licensing

All input data are openly licensed. See `DATA_LICENSE.md` for the required attribution of each
source and the terms that apply to the derived tables in `data/`. Code is MIT (`LICENSE`).

## Contact

Yu-Chun Hsu — Department of Forestry, National Chung Hsing University, Taichung, Taiwan
bigq@nchu.edu.tw · ORCID [0000-0002-6616-6906](https://orcid.org/0000-0002-6616-6906)
