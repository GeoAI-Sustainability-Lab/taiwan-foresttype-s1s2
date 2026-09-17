# -*- coding: utf-8 -*-
"""canonical_v3_D.py — broadleaf–bamboo enhancement strategies evaluated on the 34-feature baseline,
plus the Presto heads under the same folds (same-window and independent year) and the multi-year (biennial) descriptors.
Protocol identical to canonical_v3.py: GroupKFold(5) on blk (0.1°×0.09° graticule, 204 cells), RF(300, leaf 3, balanced, seed 0);
  same-window : fit train4yr[a] -> predict train4yr[b];  independent : fit train4yr[a] -> predict test1yr[b].
Outputs canonical_v3_D.json and preds_v3_D.npz.
"""
import numpy as np, pandas as pd, json, os, sys, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support
from scipy.stats import binomtest, chi2 as chi2d

HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
DATA=os.path.join(REPO,"data","analyses"); OUT=os.path.join(REPO,"results","analyses"); ARCH=os.path.join(REPO,"data","archive_v1_35features"); os.makedirs(OUT,exist_ok=True)
tr=pd.read_csv(os.path.join(DATA,"features_train4yr_v2.csv")); te=pd.read_csv(os.path.join(DATA,"features_test1yr_2025-2026_v2.csv"))
DROP=["RVI_mean"]
F34=[c for c in tr.columns if c not in ("cls","lon","lat","blk")+tuple(DROP)]
assert len(F34)==34
y=tr.cls.values.astype(int); blk=tr.blk.astype(str).values
RF=dict(n_estimators=300,min_samples_leaf=3,class_weight="balanced",random_state=0,n_jobs=-1)
FOLDS=list(GroupKFold(5).split(tr[F34],y,blk))
ref=np.load(os.path.join(OUT,"preds_v3.npz"),allow_pickle=True)
if "fold_id" in ref.files:
    fid=np.zeros(len(y),int)
    for k,(a,b) in enumerate(FOLDS): fid[b]=k
    assert (fid==ref["fold_id"]).all(), "fold definition differs from canonical_v3"

def metr(yt,yp):
    cm=confusion_matrix(yt,yp,labels=[1,2,3]); P,R,Fs,_=precision_recall_fscore_support(yt,yp,labels=[1,2,3],zero_division=0)
    return dict(OA=round(accuracy_score(yt,yp),4),kappa=round(cohen_kappa_score(yt,yp),4),macroF1=round(f1_score(yt,yp,average="macro"),4),
                recall=[round(x,3) for x in R],precision=[round(x,3) for x in P],F1=[round(x,3) for x in Fs],
                cm=cm.tolist(),errors=int((yt!=yp).sum()),blba=int(cm[0,2]+cm[2,0]))
ublk,binv=np.unique(blk,return_inverse=True); NB=len(ublk)
def boot_block_stat(vals,n=20000,seed=0):
    rng=np.random.default_rng(seed); cnt=np.bincount(binv,minlength=NB).astype(float)
    sums=np.bincount(binv,weights=vals,minlength=NB)
    W=rng.multinomial(NB,np.ones(NB)/NB,size=n).astype(float)
    stat=(W@sums)/(W@cnt); return [round(float(np.percentile(stat,2.5)),4),round(float(np.percentile(stat,97.5)),4)]
def mcnemar(cA,cB):
    n01=int(np.sum(cA&~cB)); n10=int(np.sum(~cA&cB)); n=n01+n10
    pex=binomtest(min(n01,n10),n,0.5).pvalue if n>0 else 1.0
    chi=(abs(n01-n10)-1)**2/max(n,1); pcc=float(1-chi2d.cdf(chi,1))
    return dict(A_only=n01,B_only=n10,p_exact=round(float(pex),4),p_cc=round(pcc,4))
def paired(cA,cB):
    d=mcnemar(cA,cB); d["delta_OA"]=round(float(cA.mean()-cB.mean()),4); d["delta_block_CI"]=boot_block_stat(cA.astype(float)-cB.astype(float)); return d

def run_arrays(Atr,Ate):
    p1=np.zeros(len(y),int); p2=np.zeros(len(y),int)
    for a,b in FOLDS:
        m=RandomForestClassifier(**RF).fit(Atr[a],y[a]); p1[b]=m.predict(Atr[b]); p2[b]=m.predict(Ate[b])
    return p1,p2

out={"protocol":"GroupKFold(5) on 0.1x0.09 deg graticule blocks (204), RF(300, min_samples_leaf 3, balanced, seed 0); same folds as canonical_v3 (preds_v3.npz fold_id)",
     "baseline":"34 features (RVI removed, aspect_cos corrected)"}
preds={}
t0=time.time()
# ---------- 0. baseline 34 ----------
X34tr=tr[F34].values; X34te=te[F34].values
b1,b2=run_arrays(X34tr,X34te); c34=(b2==y)
out["S3_34"]={"Baseline 34":{"n_features":34,"same":metr(y,b1),"inde":metr(y,b2)}}
assert out["S3_34"]["Baseline 34"]["inde"]["OA"]==0.8675, out["S3_34"]["Baseline 34"]["inde"]["OA"]
preds["base_inde"]=b2
print("baseline ok",out["S3_34"]["Baseline 34"]["inde"]["OA"],flush=True)

# ---------- 1. + multi-scale texture (11) ----------
TX=pd.read_csv(os.path.join(DATA,"texture_lband.csv")).sort_values("ptid").reset_index(drop=True)
TEX=['NDRE_fstd1','NDRE_fstd2','NDRE_fstd4','NDVI_fstd2','NDVI_fstd4','glcm2_contrast','glcm2_ent','glcm2_idm','glcm2_var','glcm4_contrast','glcm4_ent']
for c in TEX: TX[c]=TX[c].fillna(TX[c].median())
Txt=TX[TEX].values
q1,q2=run_arrays(np.hstack([X34tr,Txt]),np.hstack([X34te,Txt]))
out["S3_34"]["+ Multi-scale texture (GLCM / focal SD), 45"]={"n_features":45,"same":metr(y,q1),"inde":metr(y,q2),"vs_baseline":paired(q2==y,c34)}
preds["texture_inde"]=q2
print("texture",out["S3_34"]["+ Multi-scale texture (GLCM / focal SD), 45"]["inde"]["OA"],flush=True)

# ---------- 2. + spring red-edge / senescence (8) ----------
df=pd.read_csv(os.path.join(DATA,"s2_monthly_multiindex.csv"))
def seasonal(years):
    sub=df[df.year.isin(years)]; rows={}
    for pid,g in sub.groupby("ptid"):
        r={}
        for idx in ['NDRE','CIre','LSWI','PSRI']:
            v=g[idx].values.astype(float); mo=g['month'].values; ok=np.isfinite(v); v,mo=v[ok],mo[ok]
            if len(v)<3: continue
            spr=v[np.isin(mo,[3,4,5])]; win=v[np.isin(mo,[12,1,2])]
            r[idx+'_spr']=spr.mean() if len(spr) else np.nan; r[idx+'_mean']=v.mean()
            r[idx+'_mar']=v[mo==3].mean() if (mo==3).any() else np.nan
            r[idx+'_spr_win']=(spr.mean() if len(spr) else np.nan)-(win.mean() if len(win) else np.nan)
            r[idx+'_apr_dip']=(v[mo==4].mean() if (mo==4).any() else np.nan)-v.mean()
        rows[int(pid)]=r
    return pd.DataFrame(rows).T.reindex(range(755))
COLS=["NDRE_spr","CIre_spr","NDRE_mean","NDRE_mar","LSWI_spr","NDRE_spr_win","PSRI_spr","PSRI_apr_dip"]
Str=seasonal([2021,2022,2023,2024]); Ste=seasonal([2025]); med=Str[COLS].median()
Atr=np.hstack([X34tr,Str[COLS].fillna(med).values]); Ate=np.hstack([X34te,Ste[COLS].fillna(med).values])
s1,s2=run_arrays(Atr,Ate)
out["S3_34"]["+ Spring red-edge / senescence, 42"]={"n_features":42,"same":metr(y,s1),"inde":metr(y,s2),"vs_baseline":paired(s2==y,c34),
    "features":COLS,"note":"seasonal means computed per window (2021-2024 for training, 2025 for the independent year)"}
preds["spring_inde"]=s2
print("spring",out["S3_34"]["+ Spring red-edge / senescence, 42"]["inde"]["OA"],flush=True)

# ---------- 3. hierarchical dedicated sub-model (34) and oracle gating (34) ----------
ph=np.zeros(len(y),int); po=b2.copy(); yc=(y==2).astype(int)
for a,b in FOLDS:
    s_1=RandomForestClassifier(**RF).fit(X34tr[a],yc[a]); pred_con=s_1.predict(X34te[b])
    m=np.isin(y[a],[1,3]); s_2=RandomForestClassifier(**RF).fit(X34tr[a][m],y[a][m]); pred_blba=s_2.predict(X34te[b])
    ph[b]=np.where(pred_con==1,2,pred_blba)
    te_m=np.isin(y[b],[1,3]); idxb=b[te_m]; po[idxb]=s_2.predict(X34te[idxb])
out["S3_34"]["Hierarchical dedicated sub-model, 34"]={"n_features":34,"inde":metr(y,ph),"vs_baseline":paired(ph==y,c34),
    "note":"stage 1 conifer-versus-rest on all training points; stage 2 broadleaf-versus-bamboo specialist trained on broadleaf and bamboo training points"}
out["S3_34"]["Oracle gating (diagnostic, not deployable), 34"]={"n_features":34,"inde":metr(y,po),"vs_baseline":paired(po==y,c34),
    "note":"specialist applied to the TRUE broadleaf/bamboo points of the held-out fold; uses the label at inference"}
preds["hier_inde"]=ph; preds["oracle_inde"]=po
print("hier",out["S3_34"]["Hierarchical dedicated sub-model, 34"]["inde"]["OA"],"oracle",out["S3_34"]["Oracle gating (diagnostic, not deployable), 34"]["inde"]["OA"],flush=True)

# ---------- 4. + DINOv2 embeddings (384) ----------
E=np.load(os.path.join(DATA,"dino_emb.npy"))
d1,d2=run_arrays(np.hstack([X34tr,E]),np.hstack([X34te,E]))
out["S3_34"]["+ DINOv2 ViT-S/14 embeddings (384), 418"]={"n_features":418,"same":metr(y,d1),"inde":metr(y,d2),"vs_baseline":paired(d2==y,c34),
    "note":"frozen DINOv2 ViT-S/14 embedding of the true-colour Sentinel-2 patch at each point (same value in training and testing)"}
preds["dino_inde"]=d2
print("dino",out["S3_34"]["+ DINOv2 ViT-S/14 embeddings (384), 418"]["inde"]["OA"],flush=True)

# ---------- 5. + multi-year (biennial) descriptors (30) ----------
FE=pd.read_csv(os.path.join(DATA,"s2_biennial_features.csv"),index_col=0).reindex(range(755))
for c in FE.columns: FE[c]=FE[c].fillna(FE[c].median())
m1,m2=run_arrays(np.hstack([X34tr,FE.values]),np.hstack([X34te,FE.values]))
out["S3_34"]["+ Multi-year biennial descriptors (30), 64"]={"n_features":64,"same":metr(y,m1),"inde":metr(y,m2),"vs_baseline":paired(m2==y,c34),
    "note":"inter-annual variance, 24-month harmonic amplitude, biennial/annual amplitude ratio, spring consecutive-year difference and odd/even alternation of six indices from 2017-2025 monthly Sentinel-2; multi-year descriptors take the same value in training and testing"}
preds["biennial_inde"]=m2
print("biennial",out["S3_34"]["+ Multi-year biennial descriptors (30), 64"]["inde"]["OA"],flush=True)

# ---------- 6. Presto heads under the same folds ----------
try:
    Ec=np.load(os.path.join(DATA,"presto_Ec.npy")); Ei=np.load(os.path.join(DATA,"presto_Ei.npy"))
    assert Ec.shape[0]==755 and Ei.shape[0]==755
    pl1=np.zeros(len(y),int); pl2=np.zeros(len(y),int); pr1=np.zeros(len(y),int); pr2=np.zeros(len(y),int)
    for a,b in FOLDS:
        sc=StandardScaler().fit(Ec[a])
        lin=LogisticRegression(max_iter=5000,class_weight="balanced",C=1.0,random_state=0).fit(sc.transform(Ec[a]),y[a])
        pl1[b]=lin.predict(sc.transform(Ec[b])); pl2[b]=lin.predict(sc.transform(Ei[b]))
        rf=RandomForestClassifier(**RF).fit(Ec[a],y[a]); pr1[b]=rf.predict(Ec[b]); pr2[b]=rf.predict(Ei[b])
    out["presto_heads"]={"Presto+Linear":{"same":metr(y,pl1),"inde":metr(y,pl2),"vs_RF34_inde":paired(pl2==y,c34)},
                         "Presto+RF":{"same":metr(y,pr1),"inde":metr(y,pr2),"vs_RF34_inde":paired(pr2==y,c34)},
                         "note":"frozen 128-d Presto embeddings (_presto_Ec.npy train window, _presto_Ei.npy independent year); linear head = standardised multinomial logistic regression (balanced), RF head = same RF as the main model"}
    preds["presto_lin_inde"]=pl2; preds["presto_rf_inde"]=pr2
    print("presto lin",out["presto_heads"]["Presto+Linear"]["same"]["OA"],out["presto_heads"]["Presto+Linear"]["inde"]["OA"],"rf",out["presto_heads"]["Presto+RF"]["same"]["OA"],out["presto_heads"]["Presto+RF"]["inde"]["OA"],flush=True)
except Exception as e:
    out["presto_heads"]={"error":str(e)}; print("presto skipped",e,flush=True)

json.dump(out,open(os.path.join(OUT,"canonical_v3_D.json"),"w"),indent=1)
np.savez(os.path.join(OUT,"preds_v3_D.npz"),**preds)
print("PART1 DONE %.0fs"%(time.time()-t0),flush=True)

# ---------- 7. fusion network with multi-objective loss (34 + texture), torch ----------
try:
    pass
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
    Atr=np.hstack([X34tr,Txt]); Ate=np.hstack([X34te,Txt]); NP=X34tr.shape[1]; NT=Txt.shape[1]; yt=y-1
    cw=torch.tensor([(len(y)/(3*np.sum(y==k))) for k in [1,2,3]],dtype=torch.float32)
    class Fusion(nn.Module):
        def __init__(s):
            super().__init__()
            s.ph=nn.Sequential(nn.Linear(NP,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,32),nn.ReLU())
            s.tx=nn.Sequential(nn.Linear(NT,32),nn.ReLU(),nn.Dropout(.3),nn.Linear(32,16),nn.ReLU())
            s.cls=nn.Linear(48,3); s.aux=nn.Linear(48,1)
        def forward(s,x):
            e=torch.cat([s.ph(x[:,:NP]),s.tx(x[:,NP:])],1); return s.cls(e),s.aux(e),F.normalize(e,dim=1)
    def supcon(z,lab,t=0.1):
        sim=torch.mm(z,z.t())/t; n=z.shape[0]
        m=(lab.view(-1,1)==lab.view(1,-1)).float(); m.fill_diagonal_(0)
        logits=sim-sim.max(1,keepdim=True).values.detach()
        exp=torch.exp(logits)*(1-torch.eye(n))
        logp=logits-torch.log(exp.sum(1,keepdim=True)+1e-9)
        mp=(m*logp).sum(1)/(m.sum(1)+1e-9)
        return -mp.mean()
    def train_pred(Xtr,ytr,Xte):
        sc=StandardScaler().fit(Xtr); Xtr=torch.tensor(sc.transform(Xtr),dtype=torch.float32); Xte=torch.tensor(sc.transform(Xte),dtype=torch.float32)
        yy=torch.tensor(ytr,dtype=torch.long); ybin=(torch.tensor(ytr)==2).float()
        net=Fusion(); opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-4)
        for ep in range(250):
            net.train(); opt.zero_grad(); lo,au,z=net(Xtr)
            loss=F.cross_entropy(lo,yy,weight=cw)+0.3*F.binary_cross_entropy_with_logits(au.squeeze(1),ybin)+0.3*supcon(z,yy)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): lo,_,_=net(Xte); return lo.argmax(1).numpy()+1
    pf=np.zeros(len(y),int)
    for a,b in FOLDS: pf[b]=train_pred(Atr[a],yt[a],Ate[b])
    out["S3_34"]["Fusion network + multi-objective loss (34 + texture)"]={"n_features":45,"inde":metr(y,pf),"vs_baseline":paired(pf==y,c34),
        "note":"two-branch MLP (34 harmonic features; 11 texture features) trained 250 full-batch epochs, Adam 2e-3, wd 1e-4, seed 0; loss = weighted cross-entropy + 0.3 x bamboo-versus-rest auxiliary + 0.3 x supervised contrastive term"}
    preds["fusion_inde"]=pf
    print("fusion",out["S3_34"]["Fusion network + multi-objective loss (34 + texture)"]["inde"]["OA"],flush=True)
except Exception as e:
    out["S3_34"]["Fusion network + multi-objective loss (34 + texture)"]={"error":str(e)}; print("fusion failed",e,flush=True)

json.dump(out,open(os.path.join(OUT,"canonical_v3_D.json"),"w"),indent=1)
np.savez(os.path.join(OUT,"preds_v3_D.npz"),**preds)
print("ALL DONE %.0fs"%(time.time()-t0),flush=True)
