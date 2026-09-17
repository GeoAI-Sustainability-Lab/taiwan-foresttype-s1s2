# Changelog

## v1.1.0 — 2026-09-18 (revision R1 of manuscript JSTARS-2026-04232)

- Feature set: 34 features. The radar vegetation index of the submitted version is withdrawn (its
  dual-polarisation form is defined on linear-power backscatter, whereas the archived features are in
  decibels; its information is carried by the VV, VH and ratio features). The aspect cosine is
  cos(aspect in radians). `data/features_train4yr.csv` and `data/features_test1yr_2025-2026.csv` are the
  34-feature tables; they now carry `ptid`, `lon`, `lat`, `blk` (0.1° × 0.09° graticule cell) and `fold`.
- `src/train_eval.py` reproduces Table 1 of the revised manuscript (overall accuracy, κ, macro-F1,
  per-class recall and F1, block-bootstrap intervals) and writes `results/table1.json`.
- Headline numbers: same-window OA 0.889 (κ 0.833); next-year OA 0.868 (κ 0.801, macro-F1 0.866),
  previously 0.885 / 0.869 with 35 features (within one percentage point; paired interval includes zero).
- `models/rf_full.joblib` retrained on the 34 features (all 755 points, 2021–2024 features).
- New `src/revision_R1/` with the scripts of the revision analyses (texture McNemar test and paired
  intervals, block-protocol label efficiency, label-noise sensitivity, nested feature selection,
  calendar-aligned deep models with three seeds, optical-month masking, periodicity test of the annual
  series, recent-image check estimator) and `results/R1/` with their outputs as run.
- New `gee/05_feature_exporter_34.js` (rebuilt feature exporter, checked against the archived table),
  `gee/06_landsat_two_era.js`, `gee/07_s2_multiyear_biennial.js`, `gee/08_visual_check_chips.js`.
- Figures 4, 5, 7, 8 and S1 redrawn from the 34-feature results; Fig. S2 (texture) added.
- `requirements.txt` pinned to the versions used for the revision.
- The tables of the submitted version are kept under `data/archive_v1_submission/`
  (git tag `v1.0.0` carries the full submitted-version repository).

## v1.0.0 — 2026-07-20 (submitted version)

- Initial public release: 35 features, Random Forest, 10 km spatial-block cross-validation,
  spatio-temporally independent year, SHAP / ALE, island-wide 20 m map.
