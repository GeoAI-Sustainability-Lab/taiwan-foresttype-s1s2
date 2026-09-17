# Changelog

## Unreleased

- `figures/fig1_study_area.png` and `figures/fig9_canopy_chips.png` redrawn with layout changes only
  (legend removed from the map and its frame widened; small gaps between the canopy chips). No change to data, code or results.

## v1.1.1 — 2026-09-18

- Documentation and metadata revised; analysis folders renamed to `src/analyses/`, `results/analyses/`,
  `data/analyses/` and `data/archive_v1_35features/`; `results/accuracy_by_feature_set.json` replaces `results/table1.json`.
  No change to the data, the scripts' logic or the results.

## v1.1.0 — 2026-09-18

- Feature set: 34 features. The radar vegetation index of v1.0.0 is withdrawn (its dual-polarisation
  form is defined on linear-power backscatter, whereas the archived features are in decibels, and its
  information is carried by the VV, VH and ratio features). The aspect cosine is cos(aspect in radians).
  `data/features_train4yr.csv` and `data/features_test1yr_2025-2026.csv` are the 34-feature tables and
  now carry `ptid`, `lon`, `lat`, `blk` (0.1° × 0.09° graticule cell) and `fold`.
- `src/train_eval.py` prints the accuracy of every feature set (overall accuracy, κ, macro-F1,
  per-class recall and F1, block-bootstrap intervals) and writes `results/accuracy_by_feature_set.json`.
- Headline numbers: same-window OA 0.889 (κ 0.833), next-year OA 0.868 (κ 0.801, macro-F1 0.866),
  previously 0.885 / 0.869 with 35 features.
- `models/rf_full.joblib` retrained on the 34 features (all 755 points, 2021–2024 features).
- New `src/analyses/` with the further analyses (texture gain with McNemar test and paired intervals,
  block-protocol label efficiency, label-noise sensitivity, nested feature selection, calendar-aligned
  deep models with three seeds, optical-month masking, periodicity test of the annual series,
  recent-image check estimator) and `results/analyses/` with their outputs as run.
- New `gee/05_feature_exporter_34.js` (rebuilt feature exporter, checked against the archived table),
  `gee/06_landsat_two_era.js`, `gee/07_s2_multiyear_biennial.js`, `gee/08_visual_check_chips.js`.
- Figures redrawn from the 34-feature results, texture figure added.
- `requirements.txt` pinned. `CITATION.cff` and `.zenodo.json` added.
- The 35-feature tables of v1.0.0 are kept under `data/archive_v1_35features/`.

## v1.0.0 — 2026-07-20

- Initial public release: 35 features, Random Forest, 10 km spatial-block cross-validation,
  spatio-temporally independent year, SHAP / ALE, island-wide 20 m map.
