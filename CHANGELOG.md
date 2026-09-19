# Changelog

## v1.1.4 — 2026-09-19

- Fig. 3 of the manuscript redrawn with the first-order harmonic fit that produces the features. The submitted version of panels (a)
  and (b) had been drawn with a second-order fit (annual plus semi-annual terms), which the caption did not state. New
  `src/analyses/fig3_phenology_fit.py`, `data/analyses/fig3_ts_points.csv` (Sentinel-2 NDMI and Sentinel-1 ascending VH observations
  of one representative stand per class, 2023–2024, exported from GEE) and `results/analyses/fig3_fit.json` (coefficients, amplitude,
  phase, peak day of year and R² of the fits of both orders); `figures/fig3_phenology.png` regenerated from them, with the chips
  (c) to (e) unchanged (`figures/fig3_chips.png`). No change to the data tables, the models or the accuracies.

## v1.1.3 — 2026-09-19

- New `src/analyses/table4_niche.py` and `results/analyses/table4_niche.json`: class means of the environmental niche variables
  at the 755 reference points (2021–2024 features), with the NDMI first-harmonic phase as a circular mean (the phase wraps at ±π).
- Figures refreshed: `fig5_model_benchmark.png` (intervals of the sequence models now belong to the three-seed mean, as the bars do),
  `fig8_confusion.png` (panel c shows the mean independent-year NDMI curves of the 52 broadleaf points misclassified as bamboo and of the
  26 bamboo points misclassified as broadleaf, instead of two selected points), `fig2_cloud_availability.png` (panel b labelled as nominal
  acquisition frequency). No change to the data, the models or the accuracies.

## v1.1.2 — 2026-09-18

- Collinearity screen corrected. `canonical_v3_C.py` used the diagonal of the pseudo-inverse of the correlation matrix as the
  VIF and ran the screen once on all points, so two exactly dependent SAR triples (RATIO_mean = VH_mean − VV_mean,
  RATIO_wetdry = VH_wetdry − VV_wetdry) survived it. New `src/analyses/canonical_v3_C_vif.py` removes the exact linear
  combinations first, computes VIF = 1/(1 − R²) and runs the screen on the training blocks of each fold; 22 features are kept
  in every fold, out-of-fold OA 0.872, next-year OA 0.857 (`results/analyses/canonical_v3_C_vif.json`). The VIF entry of
  `canonical_v3_C.json` is superseded. No other result changes.
- New `src/analyses/models_perclass_f1.py` and `results/analyses/canonical_v3_models_f1.json`: per-class F1 of every
  configuration of the model comparison, from the frozen predictions; `preds_v3_D.npz` and `preds_v2_deep.npz` added to
  `results/analyses/`.
- New `src/analyses/prior_sensitivity.py` and `results/analyses/prior_sensitivity.json`: the island-map class priors applied
  to the frozen out-of-fold probabilities of the reference RF (bamboo recall at the reference points falls to 0.008, precision 1.000).
- `figures/fig1_study_area.png` and `figures/fig9_canopy_chips.png` redrawn with layout changes only
  (legend removed from the map and its frame widened; small gaps between the canopy chips).

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
