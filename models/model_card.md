# Model card — rf_full.joblib

**Task** Per-pixel three-class forest-type classification for Taiwan
(1 = Broadleaf, 2 = Conifer, 3 = Bamboo).

**Model** scikit-learn `RandomForestClassifier`
(`n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=0`),
trained on all labelled samples using the 35 multi-year features
(Sentinel-2 optical phenology, Sentinel-1 SAR, ERA5-Land climate, SRTM terrain).
Exact feature order and class map are in `rf_full_meta.json`.

**Reported accuracy** (10-km spatial-block GroupKFold; see `src/train_eval.py`):
same-window OA ≈ 0.875 / kappa ≈ 0.81; spatio-temporal independent (train 2021-2024,
predict the 2025-2026 next year) OA ≈ 0.869 / kappa ≈ 0.80. These reproduce the
manuscript values (0.879 and 0.873) to within cross-validation tolerance.

**Intended use** Research reproduction and applying the trained model to new
35-feature samples exported from Google Earth Engine (`gee/`). Use `src/predict.py`.

**Limitations** Trained on Taiwan pure-forest reference; the broadleaf–bamboo pair
overlaps spectrally (see manuscript). Not validated outside Taiwan or for mixed stands.
Probabilities should be calibrated to a natural prior before area estimation.
