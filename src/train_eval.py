"""
train_eval.py — reproduce Table 1 and the headline accuracies of the manuscript
"Classification of Broadleaf, Conifer and Bamboo Forests Across Taiwan by
 Multi-Source Remote Sensing and Interpretable Machine Learning" (revised version, 34 features).

Protocol (Section II-D of the manuscript)
  * 755 reference points (243 broadleaf, 260 conifer, 252 bamboo), 34 features
    (12 optical phenology, 9 SAR, 13 climate + terrain), see data/DATA_DICTIONARY.md.
  * Spatial blocks: graticule cells of 0.1 deg longitude x 0.09 deg latitude
    (column `blk`, 204 cells); sklearn GroupKFold(5) on `blk` (deterministic).
  * Random Forest: 300 trees, min_samples_leaf=3, class_weight='balanced', random_state=0.
  * Same window   : fit on the training blocks of the 2021-2024 window, predict the
                    held-out blocks of the same window (features_train4yr.csv).
  * Independent   : fit on the 2021-2024 features of the training blocks, predict the
                    held-out blocks with the 2025-06 to 2026-05 features
                    (features_test1yr_2025-2026.csv) — held out in space and in time.
  * 95% intervals : block bootstrap (spatial blocks resampled with replacement, 20,000 draws).

Running as a script prints Table 1 (overall accuracy, kappa, macro-F1, per-class recall
and F1 for every feature set), writes results/table1.json, and trains the final model on
all 755 points (34 features) -> models/rf_full.joblib + models/rf_full_meta.json.

Run:  python src/train_eval.py
Locked versions are listed in requirements.txt (scikit-learn 1.7.2, numpy 2.2.6).
"""
import os, json, numpy as np, pandas as pd, joblib, sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

HERE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(HERE, "data"); MODELS = os.path.join(HERE, "models"); RESULTS = os.path.join(HERE, "results")
NAMES  = {1: "Broadleaf", 2: "Conifer", 3: "Bamboo"}
RF     = dict(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1)

# ---------------- data (module level, so other scripts can import) ----------
_tr = pd.read_csv(os.path.join(DATA, "features_train4yr.csv"))
_te = pd.read_csv(os.path.join(DATA, "features_test1yr_2025-2026.csv"))
META_COLS = ("ptid", "cls", "lon", "lat", "blk", "fold")
FEATS  = [c for c in _tr.columns if c not in META_COLS]
assert len(FEATS) == 34 and list(_te[FEATS].columns) == FEATS
X      = _tr[FEATS]                       # 2021-2024 features (same-window evaluation)
X_next = _te[FEATS]                       # 2025-06 to 2026-05 features (independent year)
y      = _tr["cls"].values.astype(int)
blk    = _tr["blk"].astype(str).values    # 0.1 deg x 0.09 deg graticule cell of each point
groups = {
 "optical": [c for c in FEATS if c.split("_")[0] in ("NDVI", "NDMI", "NBR", "reNDVI")],
 "sar":     [c for c in FEATS if c.split("_")[0] in ("VV", "VH", "RATIO")],
 "env":     [c for c in FEATS if c in ("temp_mean","dewp_mean","dewdep","soilm_mean","soilm_wet",
            "soilm_dry","soilm_wetdry","precip_mm","precip_wet","precip_dry","elev","slope","aspect_cos")],
}
assert len(groups["optical"]) == 12 and len(groups["sar"]) == 9 and len(groups["env"]) == 13
FOLDS = list(GroupKFold(5).split(X, y, blk))
fold_id = np.zeros(len(y), int)
for k, (a, b) in enumerate(FOLDS): fold_id[b] = k
assert np.array_equal(fold_id, _tr["fold"].values), "fold column must equal the deterministic GroupKFold assignment"

def samewindow_cv(cols):
    """Same-window spatial hold-out within 2021-2024. Returns out-of-fold predictions."""
    pred = np.zeros(len(y), int)
    for a, b in FOLDS:
        pred[b] = RandomForestClassifier(**RF).fit(X[cols].iloc[a], y[a]).predict(X[cols].iloc[b])
    return pred

def independent_cv(cols, proba=False):
    """Train on 2021-2024 features of the training blocks, predict the held-out blocks on the next year."""
    pred = np.zeros(len(y), int); pr = np.zeros((len(y), 3))
    for a, b in FOLDS:
        m = RandomForestClassifier(**RF).fit(X[cols].iloc[a], y[a])
        q = m.predict_proba(X_next[cols].iloc[b]); pr[b] = q; pred[b] = m.classes_[q.argmax(1)]
    return (pred, pr) if proba else pred

spatial_cv = samewindow_cv   # backward-compatible name

def metrics(yt, yp):
    cm = confusion_matrix(yt, yp, labels=[1, 2, 3])
    P, R, F, _ = precision_recall_fscore_support(yt, yp, labels=[1, 2, 3], zero_division=0)
    return dict(OA=round(accuracy_score(yt, yp), 4), kappa=round(cohen_kappa_score(yt, yp), 4),
                macroF1=round(f1_score(yt, yp, average="macro"), 4),
                recall=[round(x, 3) for x in R], precision=[round(x, 3) for x in P], F1=[round(x, 3) for x in F],
                confusion_matrix=cm.tolist(), errors=int((yt != yp).sum()),
                broadleaf_bamboo_confusions=int(cm[0, 2] + cm[2, 0]))

_ublk, _binv = np.unique(blk, return_inverse=True); _NB = len(_ublk)
def block_bootstrap_ci(vals, n=20000, seed=0):
    """95% interval of the mean of a per-point quantity, resampling the spatial blocks with replacement."""
    rng = np.random.default_rng(seed)
    cnt = np.bincount(_binv, minlength=_NB).astype(float); sums = np.bincount(_binv, weights=vals, minlength=_NB)
    W = rng.multinomial(_NB, np.ones(_NB) / _NB, size=n).astype(float)
    stat = (W @ sums) / (W @ cnt)
    return [round(float(np.percentile(stat, 2.5)), 4), round(float(np.percentile(stat, 97.5)), 4)]

SETS = [("Full (34)", FEATS),
        ("Optical + environment (25)", groups["optical"] + groups["env"]),
        ("SAR + environment (22)", groups["sar"] + groups["env"]),
        ("Environment (13)", groups["env"]),
        ("Optical only (12)", groups["optical"]),
        ("Optical + SAR (21)", groups["optical"] + groups["sar"]),
        ("SAR only (9)", groups["sar"])]

if __name__ == "__main__":
    os.makedirs(MODELS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    print(f"scikit-learn {sklearn.__version__}, numpy {np.__version__}; n={len(y)}, features={len(FEATS)}, blocks={_NB}\n")
    print("Table 1 - classification performance of each feature set (34-feature revision)")
    print("  Same window: out-of-fold within 2021-2024; Next year: same held-out blocks on 2025-06..2026-05")
    hdr = (f"  {'Feature set (dim.)':<28}{'SW OA':>7}{'SW k':>7}{'NY OA':>8}{'NY k':>7}{'mF1':>7}"
           f"{'R Bl':>6}{'R Co':>6}{'R Ba':>6}{'F1 Bl':>7}{'F1 Co':>7}{'F1 Ba':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    out = {"protocol": "GroupKFold(5) on 0.1x0.09 deg graticule blocks; RF 300 trees, min_samples_leaf 3, balanced, seed 0",
           "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}, "table1": {}}
    c_full = None
    for tag, cols in SETS:
        p1 = samewindow_cv(cols); p2 = independent_cv(cols)
        m1 = metrics(y, p1); m2 = metrics(y, p2)
        m2["CI_block_OA"] = block_bootstrap_ci((p2 == y).astype(float))
        if tag == "Full (34)": c_full = (p2 == y)
        else: m2["delta_OA_vs_full_CI_block"] = block_bootstrap_ci((p2 == y).astype(float) - c_full.astype(float))
        out["table1"][tag] = dict(same_window=m1, next_year=m2)
        print(f"  {tag:<28}{m1['OA']:7.3f}{m1['kappa']:7.3f}{m2['OA']:8.3f}{m2['kappa']:7.3f}{m2['macroF1']:7.3f}"
              f"{m2['recall'][0]:6.2f}{m2['recall'][1]:6.2f}{m2['recall'][2]:6.2f}"
              f"{m2['F1'][0]:7.3f}{m2['F1'][1]:7.3f}{m2['F1'][2]:7.3f}")
    full = out["table1"]["Full (34)"]["next_year"]
    print(f"\n  Full model, next year: OA {full['OA']} (95% block interval {full['CI_block_OA']}), "
          f"kappa {full['kappa']}, macro-F1 {full['macroF1']}, confusion matrix {full['confusion_matrix']}")
    json.dump(out, open(os.path.join(RESULTS, "table1.json"), "w"), indent=1)
    print("  -> results/table1.json")

    final = RandomForestClassifier(**RF).fit(X, y)
    joblib.dump(final, os.path.join(MODELS, "rf_full.joblib"))
    json.dump({"features": FEATS, "classes": {"1": "Broadleaf", "2": "Conifer", "3": "Bamboo"},
               "rf_params": {k: v for k, v in RF.items() if k != "n_jobs"},
               "trained_on": f"all {len(X)} reference points, 2021-2024 features (34 features)",
               "sklearn": sklearn.__version__},
              open(os.path.join(MODELS, "rf_full_meta.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved final model -> models/rf_full.joblib  ({len(FEATS)} features, all {len(X)} samples)")
