# -*- coding: utf-8 -*-
"""canonical_v2_D2.py — label-efficiency curve under the block protocol for the Transformer (calendar-aligned, fold-internal preprocessing).
Usage: python canonical_v2_D2.py 5,10,20,40,80,160   (writes results/R1/canonical_v2_D2.json). Requires torch."""
import sys, numpy as np, os, json
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"canonical_v2_D1.py"),encoding="utf-8").read().split("import sys")[0])
sizes=[int(s) for s in sys.argv[1].split(",")]
JS=os.path.join(OUT,"canonical_v2_D2.json"); out=json.load(open(JS)) if os.path.exists(JS) else {"protocol":"block few-shot (per outer fold sample n/class from training blocks; predict held-out blocks with calendar-aligned next-year sequences); Transformer, per-fold standardisation, seeds = draw index"}
REPS={5:3,10:3,20:3,40:2,80:2,160:2}
for per in sizes:
    oas=[];rcs=[]
    for r in range(REPS[per]):
        rng=np.random.default_rng(r); p=np.zeros(len(y),int)
        for a,b in FOLDS:
            sel=np.concatenate([rng.permutation(a[y[a]==c])[:per] for c in (1,2,3)])
            z,cw=prep(sel); m=fit(z(Xc[sel]),y[sel],Trans,cw,r); p[b]=predict(m,z(Xi[b]))
        oas.append(accuracy_score(y,p)); rcs.append(rec(y,p))
    out[str(per)]=dict(OA_mean=round(float(np.mean(oas)),3),OA_min=round(float(min(oas)),3),OA_max=round(float(max(oas)),3),recall_mean=[round(float(v),3) for v in np.mean(rcs,0)],reps=REPS[per])
    print("  Transformer n/class %3d  OA %.3f [%.3f-%.3f] recall %s"%(per,np.mean(oas),min(oas),max(oas),np.mean(rcs,0).round(3)),flush=True)
json.dump(out,open(JS,"w"),indent=1); print("SAVED")
