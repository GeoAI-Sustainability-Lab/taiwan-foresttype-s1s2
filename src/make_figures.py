"""Regenerate the classification figure (confusion + ablation + per-class PA/UA)
and the monthly cloud climatology from data/. Run: python src/make_figures.py
(Full figure set, incl. SHAP/ALE/t-SNE/maps, uses train_eval.py + explain_shap_ale.py
 + boundary_dissolve.py; see README.)"""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train_eval import X, y, spatial_cv, groups   # noqa
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(HERE,"figures"); os.makedirs(OUT,exist_ok=True)
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
pred = spatial_cv(list(X.columns)); oa = accuracy_score(y, pred); k = cohen_kappa_score(y, pred)
M = confusion_matrix(y, pred, labels=[1,2,3])
fig, ax = plt.subplots(1,3, figsize=(11,3.4))
im=ax[0].imshow(M, cmap="Blues")
for i in range(3):
    for j in range(3): ax[0].text(j,i,f"{M[i,j]}",ha="center",va="center")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["Broad","Conif","Bam"]); ax[0].set_yticks(range(3)); ax[0].set_yticklabels(["Broad","Conif","Bam"])
ax[0].set_title(f"Confusion (OA={oa:.3f}, kappa={k:.3f})"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Reference")
abl={"Optical":groups["optical"],"SAR":groups["sar"],"Env":groups["env"],"Opt+SAR":groups["optical"]+groups["sar"]}
vals=[accuracy_score(y, spatial_cv(c)) for c in abl.values()]+[oa]
ax[1].bar(list(abl.keys())+["Full"], vals); ax[1].set_ylim(0,1); ax[1].set_title("Feature-group ablation"); ax[1].set_ylabel("OA")
PA=[M[i,i]/M[i].sum() for i in range(3)]; UA=[M[i,i]/M[:,i].sum() for i in range(3)]; xx=np.arange(3)
ax[2].bar(xx-0.2,PA,0.4,label="Producer's"); ax[2].bar(xx+0.2,UA,0.4,label="User's")
ax[2].set_xticks(xx); ax[2].set_xticklabels(["Broad","Conif","Bam"]); ax[2].set_ylim(0,1.05); ax[2].legend(); ax[2].set_title("Per-class PA/UA")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"classification.png"), dpi=200); print("wrote classification.png")
cl=pd.read_csv(os.path.join(HERE,"data","cloud_monthly_stats_5yr.csv")); cl["prop"]=cl.n_clear20/cl.n_scenes*100
mg=cl.groupby("month").agg(mc=("mean_cloud","mean"),pc=("prop","mean")).reindex(range(1,13))
fig,ax=plt.subplots(1,2,figsize=(9,3.2))
ax[0].bar(range(1,13),mg.mc); ax[0].set_title("Monthly mean cloud cover (%)"); ax[0].set_xlabel("Month")
ax[1].bar(range(1,13),mg.pc,color="#31a354"); ax[1].set_title("Usable scenes (<20% cloud) %"); ax[1].set_xlabel("Month")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"cloud_climatology.png"), dpi=200); print("wrote cloud_climatology.png")
