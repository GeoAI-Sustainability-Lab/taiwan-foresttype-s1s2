# Broadleaf, conifer and bamboo forest-type classification across Taiwan (Sentinel-1/2)

Reproducible **feature engineering and end-to-end pipeline**, data and trained model for the manuscript

> *Classification of Broadleaf, Conifer and Bamboo Forests Across Taiwan by Multi-Source Remote Sensing
> and Interpretable Machine Learning* — Yu-Chun Hsu, National Chung Hsing University.
> Submitted to *IEEE JSTARS*.

Three pure forest types (broadleaf, conifer, bamboo) are classified across Taiwan from multi-year
Sentinel-2 optical phenology, Sentinel-1 SAR, ERA5-Land climate and SRTM terrain (**35 harmonic
features**), with a Random Forest under 10 km spatial-block cross-validation, SHAP attribution, and a
spatio-temporally independent evaluation on a held-out following year.

## Headline numbers

| Setting | OA | kappa |
|---|---|---|
| Independent year (2025-06 to 2026-05), all 35 features | **0.869** | **0.803** |
| Same-window spatial-block CV | 0.885 | 0.827 |
| Optical + environment (O + E), independent year | 0.875 | 0.813 |
| SAR + environment (S + E), no optical input, independent year | 0.817 | 0.725 |
| All 35 features + multi-scale canopy texture, independent year | 0.881 | 0.821 |

`src/train_eval.py` reproduces Table 1 of the manuscript exactly.

Every accuracy above is measured on data held out from training in **both space and time**. Fitting
uses 2021–2024; inference is on the unseen following year. No in-sample accuracy is reported.

## Repository layout

```
gee/      Google Earth Engine scripts: build features, sample/export, national 20 m classification
data/     sample tables (features_5yr.csv, train/test windows, train_points.csv), cloud stats,
          misclassified points, feature importance
src/      train_eval.py, predict.py, explain_shap_ale.py, make_figures.py, boundary_dissolve.py
models/   trained Random Forest (rf_full.joblib) + rf_full_meta.json + model_card.md
figures/  publication figures, named by their number in the manuscript
```

## Reproduce (local analysis)

```bash
pip install -r requirements.txt
python src/train_eval.py          # spatial-block CV + independent eval + Table-1 ablation; saves the model
python src/explain_shap_ale.py    # global SHAP, ALE, prediction entropy
python src/make_figures.py        # classification + cloud-climatology figures -> figures/
```

## Nationwide map (Google Earth Engine)

`gee/04_island_prior_weighted_20m.js` is the script that produced the island-wide 20 m map
(`taiwan_foresttype_20m_cal`). It is self-contained, including the 755 training points inline.
It uses `smileRandomForest(150)` with `MULTIPROBABILITY` output and re-weights the class
probabilities by their natural area priors (broadleaf 0.800, conifer 0.153, bamboo 0.047)
before taking the arg-max.

**Note.** The export carries no forest mask of its own. Apply the Fourth National Forest Inventory
forest mask when reading class shares, otherwise every non-forest pixel defaults to broadleaf.
With the mask applied the national shares are broadleaf 79.6%, conifer 19.6%, bamboo 0.8%.

## Protocol (what the numbers mean)

- **Spatial hold-out** — GroupKFold(5) on 10 km blocks, so neighbouring polygons never straddle
  the train/test boundary.
- **Independent year** — fit on 2021–2024 features of the held-out folds, then predict the same
  folds using 2025-06 to 2026-05 features. Train and test overlap in neither space nor time.
- **Classifier** — Random Forest, 300 trees, `min_samples_leaf=3`, balanced class weights, seed 0.

`PROVENANCE.md` records which experiment run each published number comes from.

## Reading the bamboo area

Bamboo covers about 0.8% of the mapped national area against about 7% in the inventory. Its very
low natural prior compounds the spectral overlap of broadleaf and bamboo, so marginal bamboo pixels
are mostly absorbed into broadleaf. **The mapped bamboo area should be read as a lower bound**, and
the broadleaf area is overstated in turn. Per-point bamboo recall on the unseen year is 0.87.

## Licence

Code under MIT (`LICENSE`). Data under `DATA_LICENSE.md`. The forest-type reference derives from the
Fourth National Forest Resource Inventory of the Forestry and Nature Conservation Agency, Taiwan.

## Citation

```bibtex
@article{hsu2026taiwanforesttype,
  author  = {Hsu, Yu-Chun},
  title   = {Classification of Broadleaf, Conifer and Bamboo Forests Across Taiwan by
             Multi-Source Remote Sensing and Interpretable Machine Learning},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

Yu-Chun Hsu — Department of Forestry, National Chung Hsing University, Taichung, Taiwan
bigq@nchu.edu.tw · ORCID [0000-0002-6616-6906](https://orcid.org/0000-0002-6616-6906)
