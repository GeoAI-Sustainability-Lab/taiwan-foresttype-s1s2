"""Apply the saved Random Forest model to new samples.

Usage:  python src/predict.py INPUT.csv [OUTPUT.csv]
INPUT.csv must contain the 35 feature columns listed in models/rf_full_meta.json.
Writes the predicted class (1=Broadleaf, 2=Conifer, 3=Bamboo), the class name,
and per-class probabilities. With no argument it demos on data/features_5yr.csv.
"""
import os, sys, json, joblib, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); M = os.path.join(HERE, "models")
meta = json.load(open(os.path.join(M, "rf_full_meta.json")))
feat = meta["features"]; rf = joblib.load(os.path.join(M, "rf_full.joblib"))
inp  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "features_5yr.csv")
out  = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
df = pd.read_csv(inp)
missing = [c for c in feat if c not in df.columns]
if missing: sys.exit(f"ERROR: input is missing feature columns: {missing}")
proba = rf.predict_proba(df[feat]); pred = rf.predict(df[feat])
res = df.copy(); res["pred_cls"] = pred
res["pred_name"] = [meta["classes"][str(p)] for p in pred]
for i, c in enumerate(rf.classes_):
    res[f"p_{meta['classes'][str(c)]}"] = proba[:, i].round(4)
res.to_csv(out, index=False)
print(f"wrote {out}  ({len(res)} rows)")
print(res["pred_name"].value_counts())
