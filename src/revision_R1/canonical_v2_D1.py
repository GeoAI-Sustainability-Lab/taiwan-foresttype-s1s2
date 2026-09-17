# -*- coding: utf-8 -*-
"""canonical_v2_D1.py — deep temporal models (calendar-aligned months, fold-internal standardisation and class weights), three seeds;
optical-month masking sensitivity; sequence-length ablation. Input: data/R1/data.npz with Xc (755,12,16) = 2021-2024 monthly means (Jan-Dec)
and Xi = 2025-06 .. 2026-05 (position k holds month (5+k) mod 12); alignment Xi_cal = np.roll(Xi,5,axis=1) so that position m is month m+1.
Outputs results/R1/canonical_v2_D1.json and preds_v2_deep.npz. Requires torch."""
import numpy as np, os, json, warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
torch.set_num_threads(os.cpu_count())
HERE=os.path.dirname(os.path.abspath(__file__))
D=np.load(os.path.join(DATA,"data.npz"),allow_pickle=True)
Xc=D["Xc"].astype(np.float32); Xi=np.roll(D["Xi"].astype(np.float32),5,axis=1); y=D["y"].astype(int); blk=D["blk"].astype(str)
C=Xc.shape[2]; T=12; NCLS=3; OPT=list(range(2,12))   # channels: VV,VH,B2..B12(10),temp,precip,elev,slope
FOLDS=list(GroupKFold(5).split(Xc,y,blk))
class TempCNN(nn.Module):
    def __init__(s):
        super().__init__(); h=64
        s.net=nn.Sequential(nn.Conv1d(C,h,3,padding=1),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(0.2),
            nn.Conv1d(h,h,3,padding=1),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(0.2),nn.Conv1d(h,h,3,padding=1),nn.BatchNorm1d(h),nn.ReLU()); s.fc=nn.Linear(h,NCLS)
    def forward(s,x): return s.fc(s.net(x.transpose(1,2)).mean(-1))
class LSTMm(nn.Module):
    def __init__(s): super().__init__(); s.l=nn.LSTM(C,64,1,batch_first=True,bidirectional=True); s.fc=nn.Linear(128,NCLS)
    def forward(s,x): o,_=s.l(x); return s.fc(o.mean(1))
class Trans(nn.Module):
    def __init__(s):
        super().__init__(); s.p=nn.Linear(C,64); s.pos=nn.Parameter(torch.randn(1,T,64)*0.02)
        el=nn.TransformerEncoderLayer(64,4,128,0.1,batch_first=True); s.enc=nn.TransformerEncoder(el,2); s.fc=nn.Linear(64,NCLS)
    def forward(s,x): return s.fc(s.enc(s.p(x)+s.pos[:,:x.size(1)]).mean(1))
class LTAE(nn.Module):
    def __init__(s):
        super().__init__(); s.p=nn.Linear(C,64); s.pos=nn.Parameter(torch.randn(1,T,64)*0.02)
        s.q=nn.Parameter(torch.randn(4,16)); s.k=nn.Linear(64,64); s.v=nn.Linear(64,64); s.fc=nn.Linear(64,NCLS)
    def forward(s,x):
        z=s.p(x)+s.pos[:,:x.size(1)]; B,L=z.size(0),z.size(1); K=s.k(z).view(B,L,4,16); V=s.v(z).view(B,L,4,16)
        att=torch.softmax((K*s.q).sum(-1)/4.0,dim=1); return s.fc((att.unsqueeze(-1)*V).sum(1).reshape(B,64))
MODELS={"TempCNN":TempCNN,"LSTM":LSTMm,"Transformer":Trans,"LTAE":LTAE}
def fit(Ztr,ytr,Make,cw,seed,epochs=100):
    torch.manual_seed(seed); np.random.seed(seed); m=Make(); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4); lossf=nn.CrossEntropyLoss(weight=cw)
    xt=torch.tensor(Ztr); yt=torch.tensor(ytr-1); m.train()
    for e in range(epochs): opt.zero_grad(); loss=lossf(m(xt),yt); loss.backward(); opt.step()
    return m.eval()
def predict(m,Z):
    with torch.no_grad(): return m(torch.tensor(Z)).argmax(1).numpy()+1
def prep(a):
    mu=np.nanmean(Xc[a],axis=(0,1),keepdims=True); sd=np.nanstd(Xc[a],axis=(0,1),keepdims=True)+1e-6
    cw=torch.tensor([len(a)/(3*np.sum(y[a]==k)) for k in (1,2,3)],dtype=torch.float32)
    z=lambda X: np.nan_to_num((X-mu)/sd,nan=0.0).astype(np.float32); return z,cw
def rec(yt,yp): cm=confusion_matrix(yt,yp,labels=[1,2,3]); return [round(float(x),3) for x in cm.diagonal()/cm.sum(1)]
import sys
STAGE=sys.argv[1]
JS=os.path.join(OUT,"canonical_v2_D1.json"); out=json.load(open(JS)) if os.path.exists(JS) else {"protocol":"GroupKFold(5) on blk; per-fold standardisation (train blocks only) and class weights; Xi rolled to calendar order; 100 full-batch epochs, Adam 1e-3, wd 1e-4; seeds 0,1,2"}
PZ=os.path.join(OUT,"preds_v2_deep.npz"); preds=dict(np.load(PZ,allow_pickle=True)) if os.path.exists(PZ) else {}
SEEDS=[0,1,2]
if STAGE in MODELS:
    nm=STAGE; Mk=MODELS[nm]; res=[]
    for s in SEEDS:
        p1=np.zeros(len(y),int); p2=np.zeros(len(y),int)
        for k,(a,b) in enumerate(FOLDS):
            z,cw=prep(a); m=fit(z(Xc[a]),y[a],Mk,cw,s); p1[b]=predict(m,z(Xc[b])); p2[b]=predict(m,z(Xi[b]))
        res.append(dict(seed=s,same_OA=round(accuracy_score(y,p1),4),inde_OA=round(accuracy_score(y,p2),4),inde_kappa=round(cohen_kappa_score(y,p2),4),
                        inde_macroF1=round(f1_score(y,p2,average="macro"),4),inde_recall=rec(y,p2)))
        preds[nm+"|P1|s%d"%s]=p1; preds[nm+"|P2|s%d"%s]=p2
        print("  %-11s seed %d  same %.3f  inde %.3f  mF1 %.3f  rec %s"%(nm,s,res[-1]["same_OA"],res[-1]["inde_OA"],res[-1]["inde_macroF1"],res[-1]["inde_recall"]),flush=True)
    io=[r["inde_OA"] for r in res]; so=[r["same_OA"] for r in res]; mf=[r["inde_macroF1"] for r in res]
    out[nm]=dict(per_seed=res,same_OA_mean=round(float(np.mean(so)),3),inde_OA_mean=round(float(np.mean(io)),3),inde_OA_range=[round(min(io),3),round(max(io),3)],
                 inde_macroF1_mean=round(float(np.mean(mf)),3),inde_recall_mean=[round(float(v),3) for v in np.mean([r["inde_recall"] for r in res],0)])
elif STAGE=="mask":
    MASK={}
    for nm in ["Transformer","LTAE"]:
        MASK[nm]={}; keep={}
        for k,(a,b) in enumerate(FOLDS):
            z,cw=prep(a); keep[k]=(fit(z(Xc[a]),y[a],MODELS[nm],cw,0),z)
        for frac in [0.0,0.25,0.5,0.75,1.0]:
            p2=np.zeros(len(y),int); rng=np.random.default_rng(0)
            for k,(a,b) in enumerate(FOLDS):
                m,z=keep[k]; Z=z(Xi[b]).copy(); nm_=int(round(frac*T))
                for i in range(len(b)):
                    months=rng.choice(T,nm_,replace=False) if nm_>0 else []
                    for mo in months: Z[i,mo,OPT]=0.0
                p2[b]=predict(m,Z)
            MASK[nm][str(frac)]=dict(OA=round(accuracy_score(y,p2),3),recall=rec(y,p2))
        print("  mask",nm,{k:v["OA"] for k,v in MASK[nm].items()},flush=True)
    out["optical_month_masking"]=dict(note="masked optical channels set to 0 = training-fold mean after standardisation; random months per sample, seed 0; SAME fold models as the seed-0 benchmark",results=MASK)
elif STAGE=="temporal":
    TL={}
    for L in [3,6,9,12]:
        p1=np.zeros(len(y),int)
        for k,(a,b) in enumerate(FOLDS):
            z,cw=prep(a); m=fit(z(Xc[a])[:,:L],y[a],Trans,cw,0); p1[b]=predict(m,z(Xc[b])[:,:L])
        TL[L]=round(accuracy_score(y,p1),3)
    print("  temporal length:",TL,flush=True); out["temporal_length_sameWindow_Transformer"]=TL
json.dump(out,open(JS,"w"),indent=1); np.savez_compressed(PZ,y=y,blk=blk,**{k:v for k,v in preds.items() if k not in ("y","blk")})
print("SAVED stage",STAGE)
