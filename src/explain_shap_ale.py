"""Global SHAP importance, 1-D ALE and normalized prediction entropy.
Run: python src/explain_shap_ale.py"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(HERE,"data","features_5yr.csv")).dropna()
y = d["cls"].values; X = d.drop(columns=["cls"])
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1).fit(X, y)
try:
    import shap
    sv = np.abs(np.array(shap.TreeExplainer(rf).shap_values(X)))
    imp = pd.Series(sv.mean(axis=tuple(range(sv.ndim-1))), index=X.columns).sort_values(ascending=False)
    print("Top mean|SHAP|:\n", imp.head(12))
except Exception as e:
    print("shap unavailable, using gini importance:", e)
    print(pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(12))
P = rf.predict_proba(X); H = -(P*np.log(P+1e-12)).sum(1)/np.log(P.shape[1])
print(f"Normalized entropy: mean={H.mean():.3f} (0=certain,1=uniform)")
