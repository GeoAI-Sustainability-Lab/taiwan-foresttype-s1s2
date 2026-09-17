# -*- coding: utf-8 -*-
"""canonical_v3.py — canonical recomputation with the 34-feature set (revision R1)
Protocol: GroupKFold(5) on blk (0.1 deg x 0.09 deg graticule cells, 204 cells), RF(300, leaf 3, balanced, seed 0)
  same-window : fit train4yr[a] -> predict train4yr[b]
  independent : fit train4yr[a] -> predict test1yr[b]
Outputs results/R1/canonical_v3.json (all numbers) and preds_v3.npz (per-point predictions and probabilities for the later statistics).
Section 0 reproduces the submitted-version numbers from data/archive_v1_submission (35 features) for the record.
"""
import numpy as np, pandas as pd, json, os, sys, warnings, hashlib, platform, sklearn, scipy
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from scipy.stats import binomtest, chi2 as chi2d

HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
DATA=os.path.join(REPO,"data","R1"); OUT=os.path.join(REPO,"results","R1"); ARCH=os.path.join(REPO,"data","archive_v1_submission"); os.makedirs(OUT,exist_ok=True)
tr0=pd.read_csv(os.path.join(ARCH,"features_train4yr.csv")); te0=pd.read_csv(os.path.join(ARCH,"features_test1yr_2025-2026.csv"))
tr=pd.read_csv(os.path.join(DATA,"features_train4yr_v2.csv")); te=pd.read_csv(os.path.join(DATA,"features_test1yr_2025-2026_v2.csv"))
F35=[c for c in tr0.columns if c not in ("cls","lon","lat","blk")]
DROP=["RVI_mean"]   # the revised tables carry 34 features (radar vegetation index withdrawn; aspect_cos = cos(aspect in radians))
F34=[c for c in tr.columns if c not in ("cls","lon","lat","blk")+tuple(DROP)]
y=tr.cls.values.astype(int); blk=tr.blk.astype(str).values
RF=dict(n_estimators=300,min_samples_leaf=3,class_weight="balanced",random_state=0,n_jobs=-1)
G={"optical":[c for c in F34 if c.split("_")[0] in ("NDVI","NDMI","NBR","reNDVI")],
   "sar":[c for c in F34 if c.split("_")[0] in ("VV","VH","RATIO","RVI")],
   "env":[c for c in F34 if c in ("temp_mean","dewp_mean","dewdep","soilm_mean","soilm_wet","soilm_dry",
          "soilm_wetdry","precip_mm","precip_wet","precip_dry","elev","slope","aspect_cos")]}
assert len(F34)==34 and len(G["optical"])==12 and len(G["sar"])==9 and len(G["env"])==13
FOLDS=list(GroupKFold(5).split(tr[F34],y,blk))
fold_id=np.zeros(len(y),int)
for k,(a,b) in enumerate(FOLDS): fold_id[b]=k

def run(cols,n_est=300,leaf=3,proba=False):
    p1=np.zeros(len(y),int); p2=np.zeros(len(y),int); pr2=np.zeros((len(y),3))
    for a,b in FOLDS:
        m=RandomForestClassifier(**{**RF,"n_estimators":n_est,"min_samples_leaf":leaf}).fit(tr[cols].iloc[a],y[a])
        p1[b]=m.predict(tr[cols].iloc[b]); q=m.predict_proba(te[cols].iloc[b]); pr2[b]=q; p2[b]=m.classes_[q.argmax(1)]
    return (p1,p2,pr2) if proba else (p1,p2)

def metr(yt,yp):
    cm=confusion_matrix(yt,yp,labels=[1,2,3]); P,R,Fs,_=precision_recall_fscore_support(yt,yp,labels=[1,2,3],zero_division=0)
    return dict(OA=round(accuracy_score(yt,yp),4),kappa=round(cohen_kappa_score(yt,yp),4),macroF1=round(f1_score(yt,yp,average="macro"),4),
                recall=[round(x,3) for x in R],precision=[round(x,3) for x in P],F1=[round(x,3) for x in Fs],
                cm=cm.tolist(),errors=int((yt!=yp).sum()),blba=int(cm[0,2]+cm[2,0]))

# ---------- bootstrap helpers (fixed predictions) ----------
def boot_point(correct,n=2000,seed=0):
    rng=np.random.default_rng(seed); N=len(correct); idx=rng.integers(0,N,size=(n,N))
    oas=correct[idx].mean(1); return [round(float(np.percentile(oas,2.5)),4),round(float(np.percentile(oas,97.5)),4)]
ublk,binv=np.unique(blk,return_inverse=True); NB=len(ublk)
def boot_block_stat(vals,n=20000,seed=0):
    """vals: per-point quantity (e.g. correct indicator or correct_A - correct_B). Resample blocks with replacement."""
    rng=np.random.default_rng(seed); cnt=np.bincount(binv,minlength=NB).astype(float)
    sums=np.bincount(binv,weights=vals,minlength=NB)
    W=rng.multinomial(NB,np.ones(NB)/NB,size=n).astype(float)   # block multiplicities
    stat=(W@sums)/(W@cnt); return [round(float(np.percentile(stat,2.5)),4),round(float(np.percentile(stat,97.5)),4)]
def mcnemar(cA,cB):
    n01=int(np.sum(cA&~cB)); n10=int(np.sum(~cA&cB)); n=n01+n10
    pex=binomtest(min(n01,n10),n,0.5).pvalue if n>0 else 1.0
    chi=(abs(n01-n10)-1)**2/max(n,1); pcc=float(1-chi2d.cdf(chi,1))
    return dict(A_only=n01,B_only=n10,p_exact=round(float(pex),4),p_cc=round(pcc,4))
def paired(cA,cB):
    d=mcnemar(cA,cB); d["delta_OA"]=round(float(cA.mean()-cB.mean()),4); d["delta_block_CI"]=boot_block_stat(cA.astype(float)-cB.astype(float)); return d

out={"meta":dict(features=F34,dropped=DROP,n=int(len(y)),class_counts=[int((y==k).sum()) for k in (1,2,3)],
     blocks=int(NB),note="revision R1: 34 features (radar vegetation index withdrawn; aspect_cos = cos of aspect in radians, sampled in GEE at 10 m)",
     block_def="blk = floor(lon/0.1)_floor(lat/0.09); 0.1 deg x 0.09 deg graticule cells (~10.2 km x 9.95 km at 23.5N)",
     folds="GroupKFold(5) deterministic; fold_id saved in preds_v3.npz",
     versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__),
     rf=RF,
     sha256=dict(train=hashlib.sha256(open(os.path.join(ARCH,"features_train4yr.csv"),"rb").read()).hexdigest(),
                 test=hashlib.sha256(open(os.path.join(ARCH,"features_test1yr_2025-2026.csv"),"rb").read()).hexdigest()))}

# ---------- 0. reproduce 35 (archived) ----------
def run35():
    p1=np.zeros(len(y),int); p2=np.zeros(len(y),int)
    for a,b in FOLDS:
        m=RandomForestClassifier(**RF).fit(tr0[F35].iloc[a],y[a]); p1[b]=m.predict(tr0[F35].iloc[b]); p2[b]=m.predict(te0[F35].iloc[b])
    return p1,p2
p1_35,p2_35=run35()
out["archived35_repro"]=dict(same=metr(y,p1_35),inde=metr(y,p2_35))
print("35 repro  same %.4f  inde %.4f"%(out["archived35_repro"]["same"]["OA"],out["archived35_repro"]["inde"]["OA"]),flush=True)

# ---------- 1. main 34 ----------
p1,p2,pr2=run(F34,proba=True)
c34=(p2==y); c35=(p2_35==y)
main=dict(same=metr(y,p1),inde=metr(y,p2))
main["inde"]["CI_point"]=boot_point(c34.astype(float)); main["inde"]["CI_block"]=boot_block_stat(c34.astype(float))
main["same"]["CI_point"]=boot_point((p1==y).astype(float)); main["same"]["CI_block"]=boot_block_stat((p1==y).astype(float))
main["vs35_inde"]=paired(c34,c35)
main["per_fold_inde_OA"]=[round(float(c34[fold_id==k].mean()),3) for k in range(5)]
main["per_fold_n"]=[int((fold_id==k).sum()) for k in range(5)]
out["main34"]=main
print("34 main   same %.4f  inde %.4f  k %.3f  mF1 %.3f  blba %d  CIpt %s CIblk %s"%(main["same"]["OA"],main["inde"]["OA"],main["inde"]["kappa"],main["inde"]["macroF1"],main["inde"]["blba"],main["inde"]["CI_point"],main["inde"]["CI_block"]),flush=True)

# ---------- 2. Table 1 (34) ----------
sets=[("Full (34)",F34),("Optical + environment (25)",G["optical"]+G["env"]),("SAR + environment (22)",G["sar"]+G["env"]),
      ("Environment (13)",G["env"]),("Optical only (12)",G["optical"]),("Optical + SAR (21)",G["optical"]+G["sar"]),("SAR only (9)",G["sar"])]
T1={}; preds_sets={}
for tag,cols in sets:
    q1,q2=run(cols); r=dict(n=len(cols),cvOA=round(accuracy_score(y,q1),4),cvK=round(cohen_kappa_score(y,q1),4),
        inOA=round(accuracy_score(y,q2),4),inK=round(cohen_kappa_score(y,q2),4),in_macroF1=round(f1_score(y,q2,average="macro"),4),
        in_recall=[round(x,3) for x in confusion_matrix(y,q2,labels=[1,2,3]).diagonal()/np.bincount(y)[1:]],
        in_F1=[round(x,3) for x in f1_score(y,q2,average=None,labels=[1,2,3])],
        blba=int(confusion_matrix(y,q2,labels=[1,2,3])[0,2]+confusion_matrix(y,q2,labels=[1,2,3])[2,0]),
        CI_block_inde=boot_block_stat((q2==y).astype(float)))
    if tag!="Full (34)": r["vs_full_inde"]=paired(c34,(q2==y))
    T1[tag]=r; preds_sets[tag]=q2
    print("  %-28s cv %.3f/%.3f  ind %.3f/%.3f  mF1 %.3f rec %s blba %d"%(tag,r["cvOA"],r["cvK"],r["inOA"],r["inK"],r["in_macroF1"],r["in_recall"],r["blba"]),flush=True)
T1["paired_SE_vs_E"]=paired(preds_sets["SAR + environment (22)"]==y,preds_sets["Environment (13)"]==y)
T1["paired_OE_vs_full"]=paired(preds_sets["Optical + environment (25)"]==y,c34)
out["table1_34"]=T1
print("  S+E vs E:",T1["paired_SE_vs_E"]); print("  O+E vs Full:",T1["paired_OE_vs_full"],flush=True)

# ---------- 3. Table S3b hyperparameters (34) ----------
S3b={}
for nt in [100,300,500]:
    S3b[nt]={}
    for lf in [1,3,5]:
        q1,q2=run(F34,n_est=nt,leaf=lf); S3b[nt][lf]=dict(cvOA=round(accuracy_score(y,q1),3),inOA=round(accuracy_score(y,q2),3))
    print("  trees",nt,S3b[nt],flush=True)
out["S3b_34"]=S3b

# ---------- 4. texture (34 + 11) ----------
tx=pd.read_csv(os.path.join(DATA,"texture_lband.csv")) if os.path.exists(os.path.join(DATA,"texture_lband.csv")) else None
if tx is None:
    for cand in ["_revision_exp/texture_lband.csv","texture_features.csv"]:
        pth=os.path.join(DATA,cand)
        if os.path.exists(pth): tx=pd.read_csv(pth); break
TEX=['NDRE_fstd1','NDRE_fstd2','NDRE_fstd4','NDVI_fstd2','NDVI_fstd4','glcm2_contrast','glcm2_ent','glcm2_idm','glcm2_var','glcm4_contrast','glcm4_ent']
if tx is not None and all(c in tx.columns for c in TEX):
    T=tx[TEX].copy(); T=T.fillna(T.median())
    assert len(T)==len(y)
    trT=pd.concat([tr[F34].reset_index(drop=True),T.reset_index(drop=True)],axis=1); teT=pd.concat([te[F34].reset_index(drop=True),T.reset_index(drop=True)],axis=1)
    q1=np.zeros(len(y),int); q2=np.zeros(len(y),int)
    for a,b in FOLDS:
        m=RandomForestClassifier(**RF).fit(trT.iloc[a],y[a]); q1[b]=m.predict(trT.iloc[b]); q2[b]=m.predict(teT.iloc[b])
    cT=(q2==y)
    tex=dict(n_features=45,same=metr(y,q1),inde=metr(y,q2),vs_34=paired(cT,c34),fixed=int(np.sum(cT&~c34)),broke=int(np.sum(~cT&c34)),
             CI_block_inde=boot_block_stat(cT.astype(float)))
    out["texture_34p11"]=tex; preds_sets["texture45"]=q2
    print("  texture 45: inde OA %.4f mF1 %.3f blba %d fixed %d broke %d p_exact %s dCI %s"%(tex["inde"]["OA"],tex["inde"]["macroF1"],tex["inde"]["blba"],tex["fixed"],tex["broke"],tex["vs_34"]["p_exact"],tex["vs_34"]["delta_block_CI"]),flush=True)
else:
    print("  texture file not found; skipped",flush=True)

# (sections 5 and 6 of the working script, a within-island transfer sketch and a prior-weighting sensitivity at points, are not part of the manuscript and are omitted here)
# ---------- 7. uncertainty / selective accuracy (34) ----------
conf=pr2.max(1); ent=-(pr2*np.log(pr2+1e-9)).sum(1)/np.log(3); correct=c34.astype(int); err=1-correct
bins=np.linspace(0,1,11); ece=0
for i in range(10):
    m_=(conf>bins[i])&(conf<=bins[i+1])
    if m_.sum()>0: ece+=m_.mean()*abs(correct[m_].mean()-conf[m_].mean())
order=np.argsort(ent); acc_ret=np.cumsum(correct[order])/np.arange(1,len(y)+1)
out["uncertainty_34"]=dict(ECE=round(float(ece),3),AUROC_entropy_error=round(float(roc_auc_score(err,ent)),3),
    acc_full=round(float(correct.mean()),3),acc_cov90=round(float(acc_ret[int(0.9*len(y))-1]),3),
    acc_cov80=round(float(acc_ret[int(0.8*len(y))-1]),3),acc_cov70=round(float(acc_ret[int(0.7*len(y))-1]),3))
print("  uncertainty:",out["uncertainty_34"],flush=True)

# ---------- 8. two-class local model (503 Bl/Co points), 34 features ----------
m2=y!=3; y2=y[m2]; b2=blk[m2]; A=tr[F34][m2].reset_index(drop=True); E=te[F34][m2].reset_index(drop=True)
q1=np.zeros(len(y2),int); q2=np.zeros(len(y2),int)
for a,b in GroupKFold(5).split(A,y2,b2):
    m=RandomForestClassifier(**RF).fit(A.iloc[a],y2[a]); q1[b]=m.predict(A.iloc[b]); q2[b]=m.predict(E.iloc[b])
def two(yt,yp): return dict(OA=round(accuracy_score(yt,yp),4),rec_Bl=round(float(((yp==1)&(yt==1)).sum()/(yt==1).sum()),4),rec_Co=round(float(((yp==2)&(yt==2)).sum()/(yt==2).sum()),4))
out["twoclass_503_34"]=dict(n=int(m2.sum()),same=two(y2,q1),inde=two(y2,q2)); print("  two-class:",out["twoclass_503_34"],flush=True)

# ---------- 9. XGBoost (34) for model table ----------
try:
    from xgboost import XGBClassifier
    q1=np.zeros(len(y),int); q2=np.zeros(len(y),int)
    for a,b in FOLDS:
        m=XGBClassifier(n_estimators=400,max_depth=5,learning_rate=0.05,subsample=0.9,colsample_bytree=0.8,objective="multi:softmax",
                        num_class=3,random_state=0,n_jobs=-1,eval_metric="mlogloss").fit(tr[F34].iloc[a],y[a]-1)
        q1[b]=m.predict(tr[F34].iloc[b])+1; q2[b]=m.predict(te[F34].iloc[b])+1
    out["xgb_34"]=dict(same=metr(y,q1),inde=metr(y,q2),vs_RF34=paired((q2==y),c34)); preds_sets["xgb34"]=q2
    print("  XGB34: same %.3f inde %.3f mF1 %.3f"%(out["xgb_34"]["same"]["OA"],out["xgb_34"]["inde"]["OA"],out["xgb_34"]["inde"]["macroF1"]),flush=True)
except Exception as e: print("  xgboost skipped:",e)

json.dump(out,open(os.path.join(OUT,"canonical_v3.json"),"w"),indent=1,ensure_ascii=False)
np.savez_compressed(os.path.join(OUT,"preds_v3.npz"),y=y,blk=blk,fold_id=fold_id,p1_34=p1,p2_34=p2,proba2_34=pr2,p2_35=p2_35,
                    **{("set|"+k):v for k,v in preds_sets.items()})
print("SAVED canonical_v3.json, preds_v3.npz")
