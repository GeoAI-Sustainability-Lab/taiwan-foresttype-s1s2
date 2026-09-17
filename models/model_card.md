# Model card, rf_full.joblib (34 features)

**Task** Per-pixel three-class forest-type classification for Taiwan
(1 = Broadleaf, 2 = Conifer, 3 = Bamboo).

**Model** scikit-learn 1.7.2 `RandomForestClassifier`
(`n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=0`),
trained on all 755 reference points with the 34 features of the 2021–2024 window
(Sentinel-2 optical phenology, Sentinel-1 SAR, ERA5-Land climate, SRTM terrain).
Exact feature order and class map are in `rf_full_meta.json`.

**Reported accuracy** (`src/train_eval.py`; GroupKFold(5) on 0.1° × 0.09° graticule cells):
same-window OA 0.889 / κ 0.833; spatio-temporally independent next year (train 2021–2024,
predict 2025-06 to 2026-05) OA 0.868 / κ 0.801 / macro-F1 0.866, per-class recall
0.770 / 0.958 / 0.869. Every accuracy is agreement with the 2008–2014 inventory labels at the
reference points.

**Intended use** Research reproduction and applying the trained model to new 34-feature
samples exported by `gee/05_feature_exporter_34.js`. Use `src/predict.py`.

**Limitations** Trained on the pure-forest reference of Taiwan's main island; the broadleaf–bamboo
pair overlaps spectrally and is confused in every configuration tested. Not
validated outside Taiwan or for mixed stands; the fitted weights and the attributions are tied to
local phenology and environment. Probabilities of an island-wide map should be read together with
the prediction-entropy layer; the mapped bamboo area is a lower bound.
