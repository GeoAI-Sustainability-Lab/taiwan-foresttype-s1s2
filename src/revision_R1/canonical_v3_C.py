# -*- coding: utf-8 -*-
"""canonical_v3_C.py — feature selection nested within the spatial folds (Table S5) and SHAP global / per-class rankings (34 and 34 + 11 texture).
Nested selection: in each outer fold the RF is fitted, permutation importance computed and the top-k chosen on the training blocks only,
then the held-out blocks are scored (no selection leakage). Output: results/R1/canonical_v3_C.json."""
import numpy as np, pandas as pd, json, os, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, cohen_kappa_score
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
DATA=os.path.join(REPO,"data","R1"); OUT=os.path.join(REPO,"results","R1"); ARCH=os.path.join(REPO,"data","archive_v1_submission"); os.makedirs(OUT,exist_ok=True)
tr=pd.read_csv(os.path.join(DATA,"features_train4yr_v2.csv")); te=pd.read_csv(os.path.join(DATA,"features_test1yr_2025-2026_v2.csv"))
F34=[c for c in tr.columns if c not in ("cls","lon","lat","blk","RVI_mean")]
y=tr.cls.values.astype(int); blk=tr.blk.astype(str).values
RF=dict(n_estimators=300,min_samples_leaf=3,class_weight="balanced",random_state=0,n_jobs=-1)
FOLDS=list(GroupKFold(5).split(tr[F34],y,blk))
out={}
# ---- nested selection
sel_log={20:[],15:[]}; P={20:(np.zeros(len(y),int),np.zeros(len(y),int)),15:(np.zeros(len(y),int),np.zeros(len(y),int))}
for k,(a,b) in enumerate(FOLDS):
    m=RandomForestClassifier(**RF).fit(tr[F34].iloc[a],y[a])
    pi=permutation_importance(m,tr[F34].iloc[a],y[a],n_repeats=10,random_state=0,n_jobs=-1)
    order=[F34[i] for i in np.argsort(-pi.importances_mean)]
    for K in (20,15):
        cols=order[:K]; sel_log[K].append(cols)
        mm=RandomForestClassifier(**RF).fit(tr[cols].iloc[a],y[a])
        P[K][0][b]=mm.predict(tr[cols].iloc[b]); P[K][1][b]=mm.predict(te[cols].iloc[b])
    print("  fold",k,"top5:",order[:5],flush=True)
S4={}
for K in (20,15):
    p1,p2=P[K]; freq=pd.Series([c for L in sel_log[K] for c in L]).value_counts()
    S4["nested top %d"%K]=dict(cvOA=round(accuracy_score(y,p1),3),inOA=round(accuracy_score(y,p2),3),inK=round(cohen_kappa_score(y,p2),3),
        selected_in_all_folds=[c for c,v in freq.items() if v==5],n_stable=int((freq==5).sum()))
    print("  nested top%d  cv %.3f  ind %.3f  stable %d"%(K,S4["nested top %d"%K]["cvOA"],S4["nested top %d"%K]["inOA"],S4["nested top %d"%K]["n_stable"]),flush=True)
# VIF (unsupervised) with 34
def vifs(X):
    Xs=(X-X.mean(0))/(X.std(0)+1e-9); C=np.corrcoef(Xs,rowvar=False); return np.diag(np.linalg.pinv(C))
cols=list(F34); dropped=[]
while True:
    v=vifs(tr[cols].values)
    if v.max()<10 or len(cols)<=5: break
    j=int(np.argmax(v)); dropped.append((cols[j],round(float(v[j]),1))); cols.pop(j)
LOCAL=[c for c in json.load(open(os.path.join(DATA,"local_selection.json")))["local"] if c in F34]
def run(cs):
    p1=np.zeros(len(y),int); p2=np.zeros(len(y),int)
    for a,b in FOLDS:
        m=RandomForestClassifier(**RF).fit(tr[cs].iloc[a],y[a]); p1[b]=m.predict(tr[cs].iloc[b]); p2[b]=m.predict(te[cs].iloc[b])
    return round(accuracy_score(y,p1),3),round(accuracy_score(y,p2),3)
for tag,cs in [("Full (34)",F34),("VIF-decorrelated (%d)"%len(cols),cols),("Locally selected (%d)"%len(LOCAL),LOCAL)]:
    c1,i1=run(cs); S4[tag]=dict(cvOA=c1,inOA=i1,n=len(cs)); print("  %-26s cv %.3f ind %.3f"%(tag,c1,i1),flush=True)
out["S4_nested_34"]=S4; out["vif_kept_34"]=cols; out["vif_dropped_34"]=dropped
# ---- SHAP (34) on the all-sample fit (explanation of the fitted classifier, as in the manuscript)
import shap
X=tr[F34].values
rf=RandomForestClassifier(**{**RF,"n_jobs":2}).fit(X,y)
sv=shap.TreeExplainer(rf).shap_values(X)
sv=[sv[...,k] for k in range(sv.shape[2])] if isinstance(sv,np.ndarray) and sv.ndim==3 else sv
imp=np.mean([np.abs(s).mean(0) for s in sv],axis=0); gorder=np.argsort(-imp)
own=np.zeros_like(sv[0])
for i in range(len(y)): own[i]=sv[y[i]-1][i]
def class_rank(k):
    o=np.argsort(-np.abs(sv[k]).mean(0)); return [F34[i] for i in o[:5]]
def direction(k,f):
    j=F34.index(f); r=np.corrcoef(X[:,j],sv[k][:,j])[0,1]; return "+" if r>0 else "-"
SH={"global_top14":[(F34[i],round(float(imp[i]),4)) for i in gorder[:14]],
    "class_top5":{nm:class_rank(k) for k,nm in enumerate(["Broadleaf","Conifer","Bamboo"])},
    "class_top3_direction":{nm:[(f,direction(k,f)) for f in class_rank(k)[:3]] for k,nm in enumerate(["Broadleaf","Conifer","Bamboo"])}}
# Bl-vs-Ba attribution: among Bl and Ba samples, |SHAP_Bl - SHAP_Ba| per feature
mBB=(y==1)|(y==3); d=np.abs(sv[0][mBB]-sv[2][mBB]).mean(0); o=np.argsort(-d)
SH["BlBa_top10"]=[(F34[i],round(float(d[i]),4)) for i in o[:10]]
out["shap_34"]=SH; print("  SHAP global top:",SH["global_top14"][:6]); print("  class top5:",SH["class_top5"],flush=True)
# ---- SHAP with texture (34+11)
tx=pd.read_csv(os.path.join(DATA,"texture_lband.csv"))
TEX=['NDRE_fstd1','NDRE_fstd2','NDRE_fstd4','NDVI_fstd2','NDVI_fstd4','glcm2_contrast','glcm2_ent','glcm2_idm','glcm2_var','glcm4_contrast','glcm4_ent']
T=tx[TEX].fillna(tx[TEX].median()).values; X45=np.hstack([X,T]); F45=F34+TEX
rf45=RandomForestClassifier(**{**RF,"n_jobs":2}).fit(X45,y); sv45=shap.TreeExplainer(rf45).shap_values(X45)
sv45=[sv45[...,k] for k in range(sv45.shape[2])] if isinstance(sv45,np.ndarray) and sv45.ndim==3 else sv45
imp45=np.mean([np.abs(s).mean(0) for s in sv45],axis=0); o45=np.argsort(-imp45)
d45=np.abs(sv45[0][mBB]-sv45[2][mBB]).mean(0); ob=np.argsort(-d45)
tex_share_global=float(imp45[34:].sum()/imp45.sum()); tex_share_blba=float(d45[34:].sum()/d45.sum())
out["shap_45"]=dict(global_top10=[(F45[i],round(float(imp45[i]),4)) for i in o45[:10]],
    texture_ranks_global={f:int(np.where(o45==F45.index(f))[0][0]+1) for f in TEX},
    BlBa_top15=[(F45[i],round(float(d45[i]),4)) for i in ob[:15]],
    texture_ranks_BlBa={f:int(np.where(ob==F45.index(f))[0][0]+1) for f in TEX},
    texture_attribution_share_global=round(tex_share_global,3),texture_attribution_share_BlBa=round(tex_share_blba,3))
print("  texture share: global %.3f  Bl-vs-Ba %.3f ; best texture rank BlBa %d"%(tex_share_global,tex_share_blba,min(out["shap_45"]["texture_ranks_BlBa"].values())),flush=True)
json.dump(out,open(os.path.join(OUT,"canonical_v3_C.json"),"w"),indent=1,ensure_ascii=False); print("SAVED canonical_v3_C.json")
