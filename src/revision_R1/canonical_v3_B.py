# -*- coding: utf-8 -*-
"""canonical_v3_B.py — label-efficiency curve under the block protocol (RF on 34 features and Presto + linear head), sample-size curve,
and label-noise sensitivity over ten seeds (34 features). Block few-shot: within each outer GroupKFold fold, n points per class are drawn
from the training blocks only and the held-out blocks are scored with next-year features; repeated draws are averaged.
Output: results/R1/canonical_v3_B.json."""
import numpy as np, pandas as pd, json, os, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
DATA=os.path.join(REPO,"data","R1"); OUT=os.path.join(REPO,"results","R1"); ARCH=os.path.join(REPO,"data","archive_v1_submission"); os.makedirs(OUT,exist_ok=True)
tr=pd.read_csv(os.path.join(DATA,"features_train4yr_v2.csv")); te=pd.read_csv(os.path.join(DATA,"features_test1yr_2025-2026_v2.csv"))
F34=[c for c in tr.columns if c not in ("cls","lon","lat","blk","RVI_mean")]
y=tr.cls.values.astype(int); blk=tr.blk.astype(str).values
D=np.load(os.path.join(DATA,"data.npz"),allow_pickle=True); Ec=D["Ec"]; Ei=D["Ei"]; assert np.array_equal(D["y"].astype(int),y)
RF=dict(n_estimators=300,min_samples_leaf=3,class_weight="balanced",random_state=0,n_jobs=-1)
FOLDS=list(GroupKFold(5).split(tr[F34],y,blk))
Xtr=tr[F34].values; Xte=te[F34].values
def rec(yt,yp): cm=confusion_matrix(yt,yp,labels=[1,2,3]); return [round(float(x),3) for x in cm.diagonal()/cm.sum(1)]
def fewshot(model,per,seed):
    rng=np.random.default_rng(seed); p=np.zeros(len(y),int)
    for a,b in FOLDS:
        sel=np.concatenate([rng.permutation(a[y[a]==c])[:per] for c in (1,2,3)])
        if model=="RF":
            p[b]=RandomForestClassifier(**RF).fit(Xtr[sel],y[sel]).predict(Xte[b])
        else:
            sc=StandardScaler().fit(Ec[sel]); clf=LogisticRegression(max_iter=3000,class_weight="balanced",C=1.0).fit(sc.transform(Ec[sel]),y[sel])
            p[b]=clf.predict(sc.transform(Ei[b]))
    return p
out={"protocol":"block few-shot: per outer GroupKFold(5) fold, sample n per class from the TRAINING blocks only; predict held-out blocks with next-year features; OOF pooled; repeats = independent random draws",
     "few_shot_block":{}}
REPS={5:10,10:10,20:8,40:6,80:4,160:3}
for model in ["RF","Presto+Linear"]:
    out["few_shot_block"][model]={}
    for per,R in REPS.items():
        oas=[];mf=[];rc=[]
        for r in range(R):
            p=fewshot(model,per,r); oas.append(accuracy_score(y,p)); mf.append(f1_score(y,p,average="macro")); rc.append(rec(y,p))
        rc=np.array(rc)
        out["few_shot_block"][model][per]=dict(OA_mean=round(float(np.mean(oas)),3),OA_min=round(float(np.min(oas)),3),OA_max=round(float(np.max(oas)),3),
            macroF1_mean=round(float(np.mean(mf)),3),recall_mean=[round(float(x),3) for x in rc.mean(0)],reps=R)
        print("  %-14s n/class %3d  OA %.3f [%.3f-%.3f]  recall %s"%(model,per,np.mean(oas),np.min(oas),np.max(oas),rc.mean(0).round(3)),flush=True)
# full-sample references
pf=np.zeros(len(y),int)
for a,b in FOLDS: pf[b]=RandomForestClassifier(**RF).fit(Xtr[a],y[a]).predict(Xte[b])
out["few_shot_block"]["RF"]["all"]=dict(OA_mean=round(accuracy_score(y,pf),3),recall_mean=rec(y,pf),reps=1)
pp=np.zeros(len(y),int)
for a,b in FOLDS:
    sc=StandardScaler().fit(Ec[a]); clf=LogisticRegression(max_iter=3000,class_weight="balanced",C=1.0).fit(sc.transform(Ec[a]),y[a]); pp[b]=clf.predict(sc.transform(Ei[b]))
out["few_shot_block"]["Presto+Linear"]["all"]=dict(OA_mean=round(accuracy_score(y,pp),3),recall_mean=rec(y,pp),reps=1)
print("  all: RF %.3f  Presto+Linear %.3f"%(accuracy_score(y,pf),accuracy_score(y,pp)),flush=True)

# ---- S3a: sample-size curve (subsample all, then block CV) with 34 features
S3a={}
for per,R in [(5,10),(10,10),(20,8),(40,6),(80,4),(160,3),(None,1)]:
    cvs=[];ins=[]
    for r in range(R):
        if per is None: idx=np.arange(len(y))
        else:
            rng=np.random.default_rng(r); idx=np.concatenate([rng.permutation(np.where(y==c)[0])[:per] for c in (1,2,3)])
        Y=y[idx]; B=blk[idx]; p1=np.zeros(len(Y),int); p2=np.zeros(len(Y),int)
        for a,b in GroupKFold(5).split(Xtr[idx],Y,B):
            m=RandomForestClassifier(**RF).fit(Xtr[idx][a],Y[a]); p1[b]=m.predict(Xtr[idx][b]); p2[b]=m.predict(Xte[idx][b])
        cvs.append(accuracy_score(Y,p1)); ins.append(accuracy_score(Y,p2))
    tag="all (755)" if per is None else "%d / %d"%(per,per*3)
    S3a[tag]=dict(cvOA=round(float(np.mean(cvs)),3),inOA=round(float(np.mean(ins)),3),reps=R)
    print("  S3a %-10s cv %.3f  ind %.3f"%(tag,np.mean(cvs),np.mean(ins)),flush=True)
out["S3a_34"]=S3a

# ---- label noise x10 seeds (34)
NZ={}
for frac in [0.05,0.10,0.20,0.30]:
    oas=[]
    for s in range(10):
        rng=np.random.default_rng(s); yn=y.copy(); k=int(round(frac*len(y))); idx=rng.choice(len(y),k,replace=False)
        for i in idx: yn[i]=rng.choice([c for c in (1,2,3) if c!=y[i]])
        p=np.zeros(len(y),int)
        for a,b in FOLDS: p[b]=RandomForestClassifier(**RF).fit(Xtr[a],yn[a]).predict(Xte[b])
        oas.append(accuracy_score(y,p))
    NZ[str(frac)]=dict(mean=round(float(np.mean(oas)),3),p2_5=round(float(np.percentile(oas,2.5)),3),p97_5=round(float(np.percentile(oas,97.5)),3),min=round(float(np.min(oas)),3),max=round(float(np.max(oas)),3))
    print("  noise %.2f  OA %.3f [%.3f-%.3f]"%(frac,np.mean(oas),np.min(oas),np.max(oas)),flush=True)
out["label_noise_x10_34"]=dict(protocol="fraction of ALL labels flipped to a random other class (seeds 0-9); training uses noisy labels; held-out blocks scored against original labels with next-year features",results=NZ)
json.dump(out,open(os.path.join(OUT,"canonical_v3_B.json"),"w"),indent=1,ensure_ascii=False); print("SAVED canonical_v3_B.json")
