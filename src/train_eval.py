"""
train_eval.py — reproduce the headline results of the manuscript
"Explainable classification of broadleaf, conifer and bamboo forests across Taiwan
 from multi-year Sentinel-1/2, with an all-weather SAR fallback."

All evaluations use 10-km spatial-block cross-validation (sklearn GroupKFold;
deterministic — no random seed in the split) with a Random Forest
(300 trees, min_samples_leaf=3, class_weight='balanced', random_state=0):

  (1) Same-window spatial hold-out : 5-year mean features (features_5yr.csv).
  (2) Spatio-temporal independent   : train on 2021-2024 features
      (features_train4yr.csv), predict the held-out spatial fold on the
      2025-2026 next year (features_test1yr_2025-2026.csv) — overlapping in
      neither space nor time. Includes the Table-1 feature-group ablation.

Running as a script also trains the FINAL model on all samples and saves it to
models/rf_full.joblib (+ rf_full_meta.json) for reuse / island-wide inference.

Run:  python src/train_eval.py
"""
import os, json, numpy as np, pandas as pd, joblib, pyproj
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

HERE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(HERE, "data"); MODELS = os.path.join(HERE, "models")
NAMES  = {1: "Broadleaf", 2: "Conifer", 3: "Bamboo"}
RF     = dict(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
              random_state=0, n_jobs=-1)

def _g10(x, y):
    return (np.floor(x/10000).astype(int).astype(str) + "_" +
            np.floor(y/10000).astype(int).astype(str))

# ---------------- data (module level, so figures/other scripts can import) ----
_d5 = pd.read_csv(os.path.join(DATA, "features_5yr.csv"))
_co = pd.read_csv(os.path.join(DATA, "train_points.csv"))
_n  = min(len(_d5), len(_co)); _d5 = _d5.iloc[:_n].copy(); _co = _co.iloc[:_n].copy()
_tx, _ty = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True).transform(
            _co.lon.values, _co.lat.values)
_d5["__g"] = _g10(_tx, _ty); _d5 = _d5.dropna().reset_index(drop=True)

FEATS  = [c for c in _d5.columns if c not in ("cls", "__g")]
X      = _d5[FEATS]                      # same-window features
y      = _d5["cls"].values.astype(int)
_BLK5  = _d5["__g"].values               # 10-km block per sample
groups = {
 "optical": [c for c in FEATS if c.split("_")[0] in ("NDVI", "NDMI", "NBR", "reNDVI")],
 "sar":     [c for c in FEATS if c.split("_")[0] in ("VV", "VH", "RATIO", "RVI")],
 "env":     [c for c in FEATS if c in ("temp_mean","dewp_mean","dewdep","soilm_mean","soilm_wet",
            "soilm_dry","soilm_wetdry","precip_mm","precip_wet","precip_dry","elev","slope","aspect_cos")],
}

_tr = pd.read_csv(os.path.join(DATA, "features_train4yr.csv"))
_te = pd.read_csv(os.path.join(DATA, "features_test1yr_2025-2026.csv"))
_ytr = _tr["cls"].values.astype(int); _blk = _tr["blk"].astype(str).values

def spatial_cv(Xs, k=5):
    """Same-window 10-km spatial-block CV. Returns (pred, OA, kappa)."""
    gkf = GroupKFold(n_splits=k); pred = np.zeros(len(y), int)
    for a, b in gkf.split(Xs, y, _BLK5):
        pred[b] = RandomForestClassifier(**RF).fit(Xs.iloc[a], y[a]).predict(Xs.iloc[b])
    return pred, accuracy_score(y, pred), cohen_kappa_score(y, pred)

def independent_cv(cols, k=5):
    """Train on 2021-2024, predict 2025-2026 held-out fold. Returns (pred, OA, kappa)."""
    gkf = GroupKFold(n_splits=k); pred = np.zeros(len(_ytr), int)
    for a, b in gkf.split(_tr[cols], _ytr, _blk):
        pred[b] = RandomForestClassifier(**RF).fit(_tr[cols].iloc[a], _ytr[a]).predict(_te[cols].iloc[b])
    return pred, accuracy_score(_ytr, pred), cohen_kappa_score(_ytr, pred)

def samewindow_cv(cols, k=5):
    """Same-window 10-km spatial-block CV *within the 2021-2024 window* (Table 1, cols 1-2)."""
    gkf = GroupKFold(n_splits=k); pred = np.zeros(len(_ytr), int)
    for a, b in gkf.split(_tr[cols], _ytr, _blk):
        pred[b] = RandomForestClassifier(**RF).fit(_tr[cols].iloc[a], _ytr[a]).predict(_tr[cols].iloc[b])
    return pred, accuracy_score(_ytr, pred), cohen_kappa_score(_ytr, pred)

def _line(ytrue, pred, tag):
    cm = confusion_matrix(ytrue, pred, labels=[1,2,3]); rec = cm.diagonal()/cm.sum(1)
    print(f"  {tag:<30} OA={accuracy_score(ytrue,pred):.3f}  kappa={cohen_kappa_score(ytrue,pred):.3f}"
          f"  recall[Bl/Co/Ba]={rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f}")

if __name__ == "__main__":
    os.makedirs(MODELS, exist_ok=True)
    print(f"n={len(_tr)}  |features|={len(FEATS)}  "
          f"(optical {len(groups['optical'])}, SAR {len(groups['sar'])}, environment {len(groups['env'])})\n")
    print("Table 1 - classification performance of each feature set")
    print("  10-km spatial-block GroupKFold (deterministic); RF 300 trees / leaf 3 / balanced / seed 0")
    print("  Spatial-CV  : train and predict within the 2021-2024 window (held-out spatial fold)")
    print("  Independent : train 2021-2024, predict the held-out fold on the 2025-2026 next year\n")
    hdr = f"  {'#':<2} {'Feature set (dim.)':<28}{'CV OA':>7}{'CV k':>7}{'Ind OA':>8}{'Ind k':>7}{'Bl':>6}{'Co':>6}{'Ba':>6}"
    print(hdr); print("  " + "-"*(len(hdr)-2))
    _sets = [("Full (35)", FEATS),
             ("Optical + environment (25)", groups['optical']+groups['env']),
             ("SAR + environment (23)", groups['sar']+groups['env']),
             ("Environment (13)", groups['env']),
             ("Optical only (12)", groups['optical']),
             ("Optical + SAR (22)", groups['optical']+groups['sar']),
             ("SAR only (10)", groups['sar'])]
    for _i, (_tag, _cols) in enumerate(_sets, 1):
        _p1, _o1, _k1 = samewindow_cv(_cols)
        _p2, _o2, _k2 = independent_cv(_cols)
        _cm = confusion_matrix(_ytr, _p2, labels=[1,2,3]); _rec = _cm.diagonal()/_cm.sum(1)
        print(f"  {_i:<2} {_tag:<28}{_o1:7.3f}{_k1:7.3f}{_o2:8.3f}{_k2:7.3f}{_rec[0]:6.2f}{_rec[1]:6.2f}{_rec[2]:6.2f}")
    print()

    final = RandomForestClassifier(**RF).fit(X, y)
    joblib.dump(final, os.path.join(MODELS, "rf_full.joblib"))
    json.dump({"features": FEATS, "classes": {"1":"Broadleaf","2":"Conifer","3":"Bamboo"},
               "rf_params": {k:v for k,v in RF.items() if k != "n_jobs"},
               "trained_on": f"all {len(X)} samples (5-year features)"},
              open(os.path.join(MODELS, "rf_full_meta.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved final model -> models/rf_full.joblib  ({len(FEATS)} features, all {len(X)} samples)")
