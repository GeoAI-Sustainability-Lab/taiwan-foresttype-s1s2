import numpy as np, os, json, warnings, sys; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
torch.set_num_threads(os.cpu_count())
import os
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
DATA=os.path.join(REPO,"data","R1"); OUT=os.path.join(REPO,"results","R1"); ARCH=os.path.join(REPO,"data","archive_v1_submission"); os.makedirs(OUT,exist_ok=True)
D=np.load(os.path.join(DATA,"data.npz"),allow_pickle=True)
Xc0=D["Xc"].astype(np.float32); Xi0=D["Xi"].astype(np.float32); y=D["y"].astype(int); blk=D["blk"].astype(str)
C=Xc0.shape[2]; T=12; NCLS=3
class TempCNN(nn.Module):
    def __init__(s):
        super().__init__(); h=64
        s.net=nn.Sequential(nn.Conv1d(C,h,3,padding=1),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(0.2),
            nn.Conv1d(h,h,3,padding=1),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(0.2),
            nn.Conv1d(h,h,3,padding=1),nn.BatchNorm1d(h),nn.ReLU()); s.fc=nn.Linear(h,NCLS)
    def forward(s,x): return s.fc(s.net(x.transpose(1,2)).mean(-1))
class LSTMm(nn.Module):
    def __init__(s): super().__init__(); s.l=nn.LSTM(C,64,1,batch_first=True,bidirectional=True); s.fc=nn.Linear(128,NCLS)
    def forward(s,x): o,_=s.l(x); return s.fc(o.mean(1))
class Trans(nn.Module):
    def __init__(s):
        super().__init__(); s.p=nn.Linear(C,64); s.pos=nn.Parameter(torch.randn(1,T,64)*0.02)
        el=nn.TransformerEncoderLayer(64,4,128,0.1,batch_first=True); s.enc=nn.TransformerEncoder(el,2); s.fc=nn.Linear(64,NCLS)
    def forward(s,x): return s.fc(s.enc(s.p(x)+s.pos).mean(1))
class LTAE(nn.Module):
    def __init__(s):
        super().__init__(); s.p=nn.Linear(C,64); s.pos=nn.Parameter(torch.randn(1,T,64)*0.02)
        s.q=nn.Parameter(torch.randn(4,16)); s.k=nn.Linear(64,64); s.v=nn.Linear(64,64); s.fc=nn.Linear(64,NCLS)
    def forward(s,x):
        z=s.p(x)+s.pos; B=z.size(0); K=s.k(z).view(B,T,4,16); V=s.v(z).view(B,T,4,16)
        att=torch.softmax((K*s.q).sum(-1)/4.0,dim=1); return s.fc((att.unsqueeze(-1)*V).sum(1).reshape(B,64))
MODELS={"TempCNN":TempCNN,"LSTM":LSTMm,"Transformer":Trans,"LTAE":LTAE}
def train_pred(Xtr,ytr,Xte_list,Make,cw,epochs=100,seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m=Make(); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4); lossf=nn.CrossEntropyLoss(weight=cw)
    xt=torch.tensor(Xtr); yt=torch.tensor(ytr-1); m.train()
    for e in range(epochs): opt.zero_grad(); loss=lossf(m(xt),yt); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): return [m(torch.tensor(X)).argmax(1).numpy()+1 for X in Xte_list]
def evaluate(mode):
    # mode: "archived" = global standardization + unaligned Xi ; "fixed" = per-fold standardization + aligned Xi
    Xi = Xi0 if mode=="archived" else np.roll(Xi0,5,axis=1)
    res={}
    for nm,Mk in MODELS.items():
        p1=np.zeros(len(y),int); p2=np.zeros(len(y),int)
        for a,b in GroupKFold(5).split(Xc0,y,blk):
            if mode=="archived":
                mu=np.nanmean(Xc0,axis=(0,1),keepdims=True); sd=np.nanstd(Xc0,axis=(0,1),keepdims=True)+1e-6
                cw=torch.tensor([len(y)/(3*np.sum(y==k)) for k in [1,2,3]],dtype=torch.float32)
            else:
                mu=np.nanmean(Xc0[a],axis=(0,1),keepdims=True); sd=np.nanstd(Xc0[a],axis=(0,1),keepdims=True)+1e-6
                cw=torch.tensor([len(a)/(3*np.sum(y[a]==k)) for k in [1,2,3]],dtype=torch.float32)
            Ztr=np.nan_to_num((Xc0[a]-mu)/sd,nan=0.0); Zc=np.nan_to_num((Xc0[b]-mu)/sd,nan=0.0); Zi=np.nan_to_num((Xi[b]-mu)/sd,nan=0.0)
            q1,q2=train_pred(Ztr,y[a],[Zc,Zi],Mk,cw)
            p1[b]=q1; p2[b]=q2
        rec=recall_score(y,p2,labels=[1,2,3],average=None)
        res[nm]=dict(same_OA=round(accuracy_score(y,p1),3), inde_OA=round(accuracy_score(y,p2),3),
                     inde_mF1=round(f1_score(y,p2,average="macro"),3), inde_recall=dict(Bl=round(rec[0],3),Co=round(rec[1],3),Ba=round(rec[2],3)))
        print(mode, nm, res[nm], flush=True)
    return res
out={"archived_repro":evaluate("archived"), "fixed_aligned_perfold":evaluate("fixed")}
json.dump(out,open(os.path.join(OUT,"rev_deep_aligned.json"),"w"),indent=1); print("saved")
