# Broadleaf, conifer and bamboo forest-type classification across Taiwan (Sentinel-1/2)

Open data, trained model and a reproducible pipeline for three-class forest-type classification
(broadleaf, conifer, bamboo) across Taiwan from Sentinel-2 optical phenology, Sentinel-1 SAR,
ERA5-Land climate and SRTM terrain. 34 engineered features, a Random Forest under spatial-block
cross-validation on 0.1° × 0.09° graticule cells, Shapley-value attribution and accumulated local
effects, and evaluation on a spatio-temporally independent year. Every number below can be
regenerated from the files in this repository with the commands given.

## Quick start

```bash
pip install -r requirements.txt        # pinned versions: scikit-learn 1.7.2, numpy 2.2.6, ...
python src/train_eval.py               # accuracy of every feature set: spatial-block CV and independent year
python src/explain_shap_ale.py         # global SHAP, ALE, prediction entropy
python src/make_figures.py             # confusion / ablation / cloud-climatology figures
```

`src/train_eval.py` prints the following table and writes `results/accuracy_by_feature_set.json`.
The split and the seed are deterministic, so the values are reproduced exactly. Keep the feature
**column order** of `data/features_train4yr.csv`, because the Random Forest draws its feature subsets
by column index.

| Feature set (dimension) | Same window OA / κ | Next year OA / κ | Next-year macro-F1 | Recall Bl / Co / Ba |
|---|---|---|---|---|
| Full (34), reference model | 0.889 / 0.833 | **0.868 / 0.801** | 0.866 | 0.770 / 0.958 / 0.869 |
| Optical + environment (25) | 0.890 / 0.835 | 0.870 / 0.805 | 0.869 | 0.774 / 0.962 / 0.869 |
| SAR + environment (22), no optical input | 0.877 / 0.815 | 0.823 / 0.734 | 0.820 | 0.716 / 0.954 / 0.790 |
| Environment (13) | 0.869 / 0.803 | 0.816 / 0.723 | 0.813 | 0.650 / 0.954 / 0.833 |
| Optical only (12) | 0.811 / 0.716 | 0.784 / 0.676 | 0.782 | 0.728 / 0.873 / 0.746 |
| Optical + SAR (21) | 0.809 / 0.714 | 0.796 / 0.694 | 0.794 | 0.728 / 0.892 / 0.762 |
| SAR only (9) | 0.567 / 0.350 | 0.462 / 0.195 | 0.459 | 0.527 / 0.523 / 0.337 |

Full model, next year: 95 % block-bootstrap interval of OA 0.829 to 0.901, confusion matrix
(rows = reference broadleaf, conifer, bamboo) `[[187, 4, 52], [0, 249, 11], [26, 7, 219]]`.
Adding the eleven multi-scale texture features gives 0.883 (`src/analyses/canonical_v3.py`).

## Protocol

Every accuracy is measured on data held out from training. Spatial hold-out is `GroupKFold(5)` on
the graticule cell `blk` (204 cells, about 10.2 km × 9.95 km), so neighbouring polygons never
straddle the train/test boundary. *Same window* fits on the training cells of 2021–2024 and predicts
the held-out cells of the same window. *Next year* fits on the 2021–2024 features of the training
cells and predicts the held-out cells with the 2025-06 to 2026-05 features, held out in space and in
time. Classifier: Random Forest, 300 trees, `min_samples_leaf=3`, balanced class weights, seed 0.
The fold of every point is stored in the `fold` column and in `data/folds.csv`. Every accuracy is
agreement with the 2008–2014 inventory labels at the reference points.

## Further analyses (`src/analyses/`, inputs in `data/analyses/`, outputs in `results/analyses/`)

| Script | What it computes | Output |
|---|---|---|
| `canonical_v3.py` | accuracy of every feature set with block-bootstrap intervals, texture gain with exact McNemar test and paired interval, entropy-based uncertainty and selective accuracy, two-class (broadleaf/conifer) model, XGBoost | `canonical_v3.json`, `preds_v3.npz` |
| `canonical_v3_B.py` | label-efficiency curve under the block protocol, sample-size curve, label-noise sensitivity over ten seeds | `canonical_v3_B.json` |
| `canonical_v3_C.py` | feature selection nested within the folds, SHAP rankings with and without texture | `canonical_v3_C.json` |
| `canonical_v3_D.py` | broadleaf–bamboo enhancement strategies (texture, spring red-edge, hierarchical model, oracle gating, DINOv2 embeddings, multi-year biennial descriptors, fusion network), Presto heads | `canonical_v3_D.json`, `preds_v3_D.npz` |
| `rerun_deep_aligned.py`, `canonical_v2_D1.py`, `canonical_v2_D2.py` | TempCNN, BiLSTM, Transformer and LTAE on calendar-aligned monthly sequences with fold-internal standardisation and three seeds, optical-month masking, sequence length, Transformer label efficiency (require PyTorch) | `rev_deep_aligned*.json`, `canonical_v2_D1.json`, `canonical_v2_D2.json` |
| `canonical_v2_E_periodicity.py` | period-two test of the 2005–2020 annual Landsat NDVI series against an AR(1) null | `canonical_v2_E_periodicity.json` |
| `visual_check_estimate.py` | agreement of 50 existing reference points with recent high-resolution imagery, with Wilson intervals and a design-weighted estimate | `visual_check_estimates.json` |

The JSON files in `results/analyses/` are the outputs as run. Running the scripts regenerates them
(the deep-model scripts take hours on CPU). `res_landsat_era.json` (two-era Landsat comparison) and
`rev_texture_class_medians.json` (class medians of the texture features) are included as run.

## Repository layout

```
data/       features_train4yr.csv, features_test1yr_2025-2026.csv (34 features + cls, lon, lat, blk, fold),
            folds.csv, train_points.csv, cloud_monthly_stats_5yr.csv — see data/DATA_DICTIONARY.md
data/analyses/   inputs of the further analyses (texture features, monthly sequences, embeddings,
            Landsat annual series, multi-year Sentinel-2 indices, recent-image check records)
data/archive_v1_35features/   the earlier 35-feature tables (tag v1.0.0), kept for the record
src/        train_eval.py, predict.py, explain_shap_ale.py, make_figures.py, boundary_dissolve.py
src/analyses/   scripts of the further analyses (table above)
results/    accuracy_by_feature_set.json (from train_eval.py) and results/analyses/ (as run)
gee/        Google Earth Engine scripts: 05_feature_exporter_34.js (feature construction and export),
            04_island_prior_weighted_20m.js (prior-weighted island-wide 20 m map), 06 two-era Landsat features,
            07 multi-year Sentinel-2 indices, 08 recent-image check viewer, and the original scripts 01 to 03
models/     trained Random Forest rf_full.joblib (34 features) + rf_full_meta.json + model_card.md
figures/    figures produced by the pipeline
```

## Nationwide map (Google Earth Engine)

`gee/04_island_prior_weighted_20m.js` produces the prior-weighted island-wide 20 m map
(`taiwan_foresttype_20m_cal`). It is self-contained, including the training points inline, and re-weights
the class probabilities by their natural area priors before taking the arg-max. The export carries no
forest mask of its own, so apply the Fourth National Forest Inventory forest mask when reading it,
otherwise every non-forest pixel defaults to broadleaf. The shares of a prior-weighted map are not
comparable with balanced-class point accuracies, so the map is a spatially explicit product rather than
an area estimate, and the mapped bamboo area is a lower bound.

## Versions

`v1.1.1` (documentation and metadata revision of `v1.1.0`) is the 34-feature release described here. `v1.0.0` is the earlier 35-feature release; its
tables are kept under `data/archive_v1_35features/` and section 0 of `src/analyses/canonical_v3.py`
reproduces its numbers. See `CHANGELOG.md`.

## Citing

Please cite the Zenodo record of this repository (`CITATION.cff`; a DOI is minted for every release).

## Data sources and licensing

All input data are openly licensed. See `DATA_LICENSE.md` for the required attribution of each
source and the terms that apply to the derived tables in `data/`. Code is MIT (`LICENSE`).

## Contact

Yu-Chun Hsu, Department of Forestry, National Chung Hsing University, Taichung, Taiwan
bigq@nchu.edu.tw · ORCID [0000-0002-6616-6906](https://orcid.org/0000-0002-6616-6906)
