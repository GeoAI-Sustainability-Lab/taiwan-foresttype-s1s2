# -*- coding: utf-8 -*-
"""Class means of the environmental niche variables of the 755 reference points (2021 to 2024 features, 34-feature table)
for the environmental-niche table: elevation, mean annual temperature, slope, and the NDMI first-harmonic phase as a circular mean
(the phase wraps at +/- pi, so its arithmetic mean is not meaningful). The dominant SHAP features and their directions per
class are read from canonical_v3_C.json (class_top3_direction). Output: results/analyses/table4_niche.json."""
import numpy as np, pandas as pd, json, os
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
DATA=os.path.join(REPO,"data","analyses"); OUT=os.path.join(REPO,"results","analyses")
tr=pd.read_csv(os.path.join(DATA,"features_train4yr_v2.csv"))
C=json.load(open(os.path.join(OUT,"canonical_v3_C.json")))["shap_34"]["class_top3_direction"]
def circmean(a): return float(np.arctan2(np.sin(a).mean(),np.cos(a).mean()))
out={"n":int(len(tr)),"source":"features_train4yr_v2.csv (755 reference points, 2021-2024 window)","classes":{}}
for c,name in ((1,"Broadleaf"),(2,"Conifer"),(3,"Bamboo")):
    d=tr[tr.cls==c]
    out["classes"][name]=dict(n=int(len(d)),elev_mean=round(float(d.elev.mean()),0),temp_mean=round(float(d.temp_mean.mean()),1),slope_mean=round(float(d.slope.mean()),1),
                              NDMI_pha_circmean=round(circmean(d.NDMI_pha.values),2),NDMI_pha_arithmetic_mean=round(float(d.NDMI_pha.mean()),2),
                              shap_top3_direction=C[name])
print(json.dumps(out["classes"],indent=None))
json.dump(out,open(os.path.join(OUT,"table4_niche.json"),"w"),indent=1)
