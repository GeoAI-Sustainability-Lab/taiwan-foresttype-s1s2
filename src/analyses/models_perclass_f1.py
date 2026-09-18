# -*- coding: utf-8 -*-
"""Per-class F1 of every configuration of Table 2 on the independent year, from the frozen per-point predictions
(preds_v3.npz: RF and XGBoost on the 34 harmonic features; preds_v3_D.npz: Presto heads; preds_v2_deep.npz: the four
calendar-aligned sequence models, three seeds each). Sequence models: the three seeds are pooled (predictions of the three
seeds concatenated against three copies of the labels), which is how the OA, macro-F1 and recall of Table 2 were computed.
Output: results/analyses/canonical_v3_models_f1.json."""
import numpy as np, json, os
from sklearn.metrics import f1_score, recall_score, accuracy_score
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE)); OUT=os.path.join(REPO,"results","analyses")
P3=np.load(os.path.join(OUT,"preds_v3.npz"),allow_pickle=True); PD=np.load(os.path.join(OUT,"preds_v3_D.npz"),allow_pickle=True); PV=np.load(os.path.join(OUT,"preds_v2_deep.npz"),allow_pickle=True)
y=P3["y"]; assert (PV["y"]==y).all()
def st(p): return dict(OA=round(float(accuracy_score(y,p)),3),macroF1=round(float(f1_score(y,p,average="macro")),3),
                       recall=[round(float(x),3) for x in recall_score(y,p,average=None)],F1=[round(float(x),3) for x in f1_score(y,p,average=None)])
out={"protocol":"independent year 2025-06 to 2026-05, spatially held-out blocks; per-class order broadleaf, conifer, bamboo; sequence models averaged over three seeds"}
out["RF (34)"]=st(P3["p2_34"]); out["XGBoost (34)"]=st(P3["set|xgb34"])
out["Presto+Linear"]=st(PD["presto_lin_inde"]); out["Presto+RF"]=st(PD["presto_rf_inde"])
yy=np.concatenate([y]*3)
for m in ["TempCNN","LSTM","Transformer","LTAE"]:
    pp=np.concatenate([PV["%s|P2|s%d"%(m,s)] for s in range(3)])
    out[m]=dict(OA=round(float(accuracy_score(yy,pp)),3),macroF1=round(float(f1_score(yy,pp,average="macro")),3),
                recall=[round(float(x),3) for x in recall_score(yy,pp,average=None)],F1=[round(float(x),3) for x in f1_score(yy,pp,average=None)],
                per_seed=[st(PV["%s|P2|s%d"%(m,s)]) for s in range(3)])
for k,v in out.items():
    if isinstance(v,dict): print("%-16s OA %.3f mF1 %.3f recall %s F1 %s"%(k,v["OA"],v["macroF1"],v["recall"],v["F1"]))
json.dump(out,open(os.path.join(OUT,"canonical_v3_models_f1.json"),"w"),indent=1)
