# -*- coding: utf-8 -*-
"""Collinearity screen nested within the spatial folds (replaces the VIF row of the earlier canonical_v3_C.py).

Earlier version: the diagonal of the pseudo-inverse of the correlation matrix was used as the VIF and the screen was run once
on all points. With exact linear dependencies among the 34 features the pseudo-inverse diagonal is finite, so two exactly
dependent triples (RATIO_mean = VH_mean - VV_mean, RATIO_wetdry = VH_wetdry - VV_wetdry) survived that screen.

This version, on the training blocks of each outer fold only:
  1. removes every derived quantity that is an exact linear combination of the other retained features
     (R^2 of a least-squares fit on the others >= 1 - 1e-9), taking the derived quantities first
     (RATIO_mean, RATIO_wetdry, dewdep, precip_mm, soilm_mean, soilm_wetdry);
  2. computes the variance inflation factor as 1 / (1 - R^2) from a least-squares fit of each feature on the others and
     removes the feature of highest VIF in turn until every VIF is below 10;
  3. fits the RF on the training blocks with the fold's subset and scores the held-out blocks in the same window and on the
     following year.
Output: results/analyses/canonical_v3_C_vif.json: per-fold subsets, the features kept in all five folds, out-of-fold OA and next-year OA / kappa.
"""
import numpy as np, pandas as pd, json, os, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
DATA=os.path.join(REPO,"data","analyses"); OUT=os.path.join(REPO,"results","analyses")
tr=pd.read_csv(os.path.join(DATA,"features_train4yr_v2.csv")); te=pd.read_csv(os.path.join(DATA,"features_test1yr_2025-2026_v2.csv"))
F34=[c for c in tr.columns if c not in ("cls","lon","lat","blk","RVI_mean")]
assert len(F34)==34
y=tr.cls.values.astype(int); blk=tr.blk.astype(str).values
RF=dict(n_estimators=300,min_samples_leaf=3,class_weight="balanced",random_state=0,n_jobs=-1)
FOLDS=list(GroupKFold(5).split(tr[F34],y,blk))
DERIVED=["RATIO_mean","RATIO_wetdry","dewdep","precip_mm","soilm_mean","soilm_wetdry"]   # exact combinations of retained features
def r2_on_others(X,j):
    others=[k for k in range(X.shape[1]) if k!=j]; A=np.c_[X[:,others],np.ones(len(X))]; b=X[:,j]
    coef=np.linalg.lstsq(A,b,rcond=None)[0]; res=b-A@coef; ss=((b-b.mean())**2).sum()
    return 1.0-(res**2).sum()/ss if ss>0 else 1.0
def screen(Xdf):
    cols=list(Xdf.columns); log=[]
    # 1. exact linear combinations, derived quantities first
    for d in DERIVED:
        if d in cols:
            X=Xdf[cols].values.astype(float); r2=r2_on_others(X,cols.index(d))
            if r2>=1-1e-9: cols.remove(d); log.append((d,"exact"))
    # any remaining exact dependency (none expected)
    while True:
        X=Xdf[cols].values.astype(float); r2=[r2_on_others(X,j) for j in range(len(cols))]
        j=int(np.argmax(r2))
        if r2[j]<1-1e-9: break
        log.append((cols[j],"exact")); cols.pop(j)
    # 2. iterative VIF
    while True:
        X=Xdf[cols].values.astype(float); vif=[1.0/max(1e-12,1.0-r2_on_others(X,j)) for j in range(len(cols))]
        j=int(np.argmax(vif))
        if vif[j]<10: break
        log.append((cols[j],round(float(vif[j]),1))); cols.pop(j)
    return cols,log
out={"protocol":"nested collinearity screen: exact linear combinations removed (derived quantities first), then iterative VIF = 1/(1-R^2) until all below 10, computed on the training blocks of each outer GroupKFold(5) fold; RF(300 trees, min_samples_leaf 3, balanced, seed 0) fitted on the training blocks with the fold's subset; held-out blocks scored in the same window (out-of-fold) and on 2025-06 to 2026-05"}
p1=np.zeros(len(y),int); p2=np.zeros(len(y),int); folds=[]
for k,(a,b) in enumerate(FOLDS):
    cols,log=screen(tr[F34].iloc[a])
    m=RandomForestClassifier(**RF).fit(tr[cols].iloc[a],y[a])
    p1[b]=m.predict(tr[cols].iloc[b]); p2[b]=m.predict(te[cols].iloc[b])
    folds.append(dict(fold=k,n_kept=len(cols),kept=cols,removed=log)); print("fold",k,"kept",len(cols),"removed",[x[0] for x in log],flush=True)
freq=pd.Series([c for f in folds for c in f["kept"]]).value_counts()
allf,alllog=screen(tr[F34])
out["folds"]=folds
out["kept_in_all_folds"]=[c for c in F34 if freq.get(c,0)==5]
out["n_kept_range"]=[min(f["n_kept"] for f in folds),max(f["n_kept"] for f in folds)]
out["all_points_reference"]=dict(n_kept=len(allf),kept=allf,removed=alllog)
out["result"]=dict(cvOA=round(accuracy_score(y,p1),3),inOA=round(accuracy_score(y,p2),3),inK=round(cohen_kappa_score(y,p2),3),
                   inMacroF1=round(f1_score(y,p2,average="macro"),3),inRecall=[round(float(x),3) for x in pd.crosstab(y,p2).values.diagonal()/np.bincount(y)[1:]])
print(json.dumps(out["result"]),"kept in all folds:",len(out["kept_in_all_folds"]),"range",out["n_kept_range"])
json.dump(out,open(os.path.join(OUT,"canonical_v3_C_vif.json"),"w"),indent=1)
