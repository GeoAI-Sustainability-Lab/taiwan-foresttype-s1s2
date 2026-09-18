# -*- coding: utf-8 -*-
"""Sensitivity of the point-level predictions of the reference RF to the class priors used for the island map.
The reference RF is trained with balanced class weights on the 755 reference points. The island map (figures/fig6_island_map.png) multiplies
the class probabilities by the mapped area priors (broadleaf 0.800, conifer 0.153, bamboo 0.047) before taking the most
likely class. Here the same priors are applied to the frozen out-of-fold next-year probabilities of the point-level RF
(preds_v3.npz, proba2_34) and the held-out reference points are rescored. This describes what the prior does to the
predictions at the balanced reference sample; it is not the accuracy of the island map, which comes from a separate
model (2021 to 2025 window, 150 trees) over all forest pixels, whose class mix is not the balanced sample.
Output: results/analyses/prior_sensitivity.json."""
import numpy as np, json, os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE)); OUT=os.path.join(REPO,"results","analyses")
P=np.load(os.path.join(OUT,"preds_v3.npz"),allow_pickle=True); y=P["y"]; pr=P["proba2_34"]; p_bal=P["p2_34"]
assert ((pr.argmax(1)+1)==p_bal).all()
PRIOR=[0.800,0.153,0.047]
def st(p): return dict(OA=round(float(accuracy_score(y,p)),3),recall=[round(float(x),3) for x in recall_score(y,p,average=None)],
                       precision=[round(float(x),3) for x in precision_score(y,p,average=None,zero_division=0)],
                       F1=[round(float(x),3) for x in f1_score(y,p,average=None)],cm=confusion_matrix(y,p).tolist())
p_pri=(pr*np.array(PRIOR)).argmax(1)+1
out={"protocol":"independent year 2025-06 to 2026-05, out-of-fold over the five spatial folds; 34-feature RF (300 trees, min_samples_leaf 3, balanced class weights); priors applied to the frozen probabilities before the argmax",
     "priors":PRIOR,"balanced_as_trained":st(p_bal),"with_map_priors":st(p_pri),"n_per_class":np.bincount(y)[1:].tolist()}
print(json.dumps({k:v for k,v in out.items() if k!="protocol"},indent=None))
json.dump(out,open(os.path.join(OUT,"prior_sensitivity.json"),"w"),indent=1)
