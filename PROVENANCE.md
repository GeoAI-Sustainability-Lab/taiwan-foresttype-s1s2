# Number provenance — single source of truth (settled 2026-07-14)

## The problem found
Two different runs of the same experiment existed in the folder, and the manuscript mixed them.

| | same-window CV OA | independent OA | errors | Bl→Ba / Ba→Bl |
|---|---|---|---|---|
| **Run A** `_indep_summary.json` (superseded) | 0.879 | 0.873 | 96 | 52 / 22 |
| **Run B** `preds_sklearn.npz`, `res_*.json` (**CANONICAL**) | 0.885 | 0.869 | 99 | 51 / 26 |

Table 1, Table 3, Table S5 and the uncertainty results used Run B.
Table S4 and the "52 / 22" confusion counts used Run A. That mismatch produced every
numeric contradiction found in editorial review (52+22=74 vs baseline 77, etc.).

## Decision
**Run B is canonical.** It is the run that (a) follows the protocol stated in Section II-D/II-G,
(b) reproduces Table 1 exactly, and (c) is what `train_eval.py` in the public repository regenerates.
`_indep_summary.json` is retained for audit only and MUST NOT be cited.

## Canonical protocol
GroupKFold(5) on 10-km block `blk`; RandomForest(n_estimators=300, min_samples_leaf=3,
class_weight='balanced', random_state=0). same-window: fit train4yr[a] -> predict train4yr[b].
independent: fit train4yr[a] -> predict test1yr_2025-2026[b]. Deterministic; no tolerance.

## Canonical confusion matrix (independent year, RF, 35 features)
```
            pred Bl  pred Co  pred Ba   recall
true Bl        188        4       51     0.774
true Co          1      249       10     0.958
true Ba         26        7      219     0.869
OA 0.8689   kappa 0.8031   misclassified 99   Bl<->Ba 51+26 = 77
```

## Regenerated under the canonical protocol
- `canonical_tables.py`  -> Table 1 (reproduces all 7 rows exactly)
- `canon_S3a.json`       -> Table S3a sample-size curve (repeated draws at small n)
- `canon_S3b.json`       -> Table S3b hyperparameters; trees300/leaf3 = 0.885 = Table 1
- `canon_S4.json`        -> Table S4 feature selection
- `canon_vif.json`       -> VIF-decorrelated set: 23 features, all VIF < 10
- `canon_table2_shap.json` -> Table 2 dominant-SHAP row

## Claims corrected because the canonical run contradicted them
1. "top 20 features reach 0.877, above the full set, so selection improves extrapolation"
   -> canonical 0.866, BELOW the full set 0.869. Selection trims dimension at no cost but does
      not improve cross-year extrapolation.
2. "the VIF set retains elevation" -> elevation is DROPPED (duplicates dew point / temperature, r ~ 0.94).
3. "accuracy plateaus at 480 samples" -> still rising: 480 -> 755 adds ~1 pp (0.873->0.885 CV, 0.861->0.869 ind).
4. "52 and 22" confusions -> 51 and 26 (sum 77, ties to Table S5).
5. per-class error rates -> Bl 22.6/12.6, Co 4.2/4.2, Ba 13.1/21.8.
6. Fig. 6 caption "soil moisture dominates globally" -> elevation, temperature, dew point.
   "NBR amplitude ranks first for broadleaf" -> mean NDVI does.
7. "17 channels" -> 16 (exp_robust.py sets C=16; 10 S2 + 2 S1 + 2 ERA5 + 2 SRTM).
8. "only texture reduced total confusion" -> the fusion network reached 67 (< 68) but by trading
   bamboo recall (0.78) for broadleaf (0.88); texture is the only one that does NOT trade recall.
9. optical-dropout baseline 0.883 is the Transformer on raw monthly sequences, not the RF (0.869).
