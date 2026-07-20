# Broadleaf, conifer and bamboo forest-type classification across Taiwan (Sentinel-1/2)

Open dataset, trained model and reproducible pipeline for three-class forest-type classification
(broadleaf, conifer, bamboo) across Taiwan, from Sentinel-2 optical phenology, Sentinel-1 SAR,
ERA5-Land climate and SRTM terrain — 35 harmonic features, Random Forest, 10 km spatial-block
cross-validation, SHAP attribution, and evaluation on a spatio-temporally independent year.

## Reproduce

```bash
pip install -r requirements.txt
python src/train_eval.py          # spatial-block CV + independent-year eval + source ablation
python src/explain_shap_ale.py    # global SHAP, ALE, prediction entropy
python src/make_figures.py        # figures -> figures/
```

`src/train_eval.py` reproduces the following table exactly.

| Setting | OA | kappa |
|---|---|---|
| All 35 features, independent year | **0.869** | **0.803** |
| All 35 features, same-window spatial-block CV | 0.885 | 0.827 |
| Optical + environment (O + E), independent year | 0.875 | 0.813 |
| SAR + environment (S + E), no optical input, independent year | 0.817 | 0.725 |
| All 35 features + multi-scale canopy texture, independent year | 0.881 | 0.821 |

**Protocol.** Every accuracy above is measured on data held out from training in both space and
time. Spatial hold-out is GroupKFold(5) on 10 km blocks, so neighbouring polygons never straddle the
train/test boundary. The independent year fits on 2021–2024 features of the held-out folds, then
predicts the same folds using 2025-06 to 2026-05 features. Classifier: Random Forest, 300 trees,
`min_samples_leaf=3`, balanced class weights, seed 0. No in-sample accuracy is reported.

`PROVENANCE.md` records which experiment run each number comes from.

## Repository layout

```
data/     feature tables (train / test windows), sample points, cloud statistics,
          misclassified points, feature importance — see data/DATA_DICTIONARY.md
src/      train_eval.py, predict.py, explain_shap_ale.py, make_figures.py, boundary_dissolve.py
gee/      Google Earth Engine scripts: feature construction, sampling/export, national 20 m map
models/   trained Random Forest (rf_full.joblib) + rf_full_meta.json + model_card.md
figures/  figures produced by the pipeline
```

## Nationwide map (Google Earth Engine)

`gee/04_island_prior_weighted_20m.js` produces the island-wide 20 m map
(`taiwan_foresttype_20m_cal`). It is self-contained, including the training points inline. It uses
`smileRandomForest(150)` with `MULTIPROBABILITY` output and re-weights the class probabilities by
their natural area priors (broadleaf 0.800, conifer 0.153, bamboo 0.047) before taking the arg-max.

**Usage note.** The export carries no forest mask of its own. Apply the Fourth National Forest
Inventory forest mask when reading class shares, otherwise every non-forest pixel defaults to
broadleaf. With the mask applied the national shares are broadleaf 79.6%, conifer 19.6%,
bamboo 0.8%.

## Data sources and licensing

All input data are openly licensed. See `DATA_LICENSE.md` for the required attribution of each
source and the terms that apply to the derived tables in `data/`. Code is MIT (`LICENSE`).

## Contact

Yu-Chun Hsu — Department of Forestry, National Chung Hsing University, Taichung, Taiwan
bigq@nchu.edu.tw · ORCID [0000-0002-6616-6906](https://orcid.org/0000-0002-6616-6906)
