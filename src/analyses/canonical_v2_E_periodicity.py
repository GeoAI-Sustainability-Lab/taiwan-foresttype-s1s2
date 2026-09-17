# -*- coding: utf-8 -*-
"""canonical_v2_E_periodicity.py — period-two test of the annual Landsat NDVI series (2005-2020) at the reference points.
Per point: linearly detrended residuals r_t; statistic S2 = share of periodogram power at period two (Nyquist) = (sum r_t (-1)^t)^2 / (n sum r_t^2);
null: an AR(1) fitted to the residuals of that point (first-order autocorrelation preserved), 999 simulated series of the same length,
p = P(S2_null >= S2_obs). Class-mean series tested the same way. Output: results/analyses/canonical_v2_E_periodicity.json."""
import numpy as np, pandas as pd, json, os
HERE=os.path.dirname(os.path.abspath(__file__))
df=pd.read_csv(os.path.join(DATA,"landsat_annual_NDVI.csv"))
df=df[(df.year>=2005)&(df.year<=2020)].dropna(subset=["NDVI"])
rng=np.random.default_rng(20260908)
def detrend(v): t=np.arange(len(v)); b=np.polyfit(t,v,1); return v-np.polyval(b,t)
def S2(r): n=len(r); return float((np.sum(r*((-1.0)**np.arange(n)))**2)/(n*np.sum(r**2)+1e-12))
def ac(r,k): r=r-r.mean(); return float(np.sum(r[:-k]*r[k:])/(np.sum(r*r)+1e-12))
def ar1_null(r,nsim=999):
    phi=np.clip(ac(r,1),-0.95,0.95); n=len(r); sig=np.std(r)*np.sqrt(max(1-phi**2,1e-3))
    e=rng.normal(0,sig,size=(nsim,n+20)); x=np.zeros((nsim,n+20))
    for t in range(1,n+20): x[:,t]=phi*x[:,t-1]+e[:,t]
    x=x[:,20:]; x=x-x.mean(1,keepdims=True)
    # detrend each simulated series the same way
    tt=np.arange(n); A=np.vstack([tt,np.ones(n)]).T; beta=np.linalg.lstsq(A,x.T,rcond=None)[0]; xr=x-(A@beta).T
    return np.array([S2(row) for row in xr])
rows=[]
for (pt,cls),g in df.groupby(["ptid","cls"]):
    g=g.sort_values("year")
    if len(g)<14 or (np.diff(g.year.values)!=1).any(): continue
    v=g.NDVI.values.astype(float); r=detrend(v); s=S2(r); null=ar1_null(r)
    rows.append(dict(ptid=int(pt),cls=int(cls),n=len(v),lag1_raw=ac(v,1),lag1_detr=ac(r,1),lag2_detr=ac(r,2),S2=s,p_ar1=float((np.sum(null>=s)+1)/(len(null)+1))))
R=pd.DataFrame(rows); out={"n_points":int(len(R)),"years":"2005-2020 (consecutive, >=14 years)","per_class":{}}
NAME={1:"Broadleaf",2:"Conifer",3:"Bamboo"}
for c in (1,2,3):
    g=R[R.cls==c]
    out["per_class"][NAME[c]]=dict(n=int(len(g)),median_lag1_raw=round(float(g.lag1_raw.median()),3),median_lag1_detrended=round(float(g.lag1_detr.median()),3),
        median_lag2_detrended=round(float(g.lag2_detr.median()),3),median_S2=round(float(g.S2.median()),3),
        frac_S2_significant_005=round(float((g.p_ar1<0.05).mean()),3),expected_under_null=0.05,
        frac_lag2_negative=round(float((g.lag2_detr<0).mean()),3))
    print(NAME[c],out["per_class"][NAME[c]],flush=True)
# class-level (spatially averaged) series: synchrony test
wide=df.pivot_table(index="year",columns="ptid",values="NDVI")
out["class_mean_series"]={}
for c in (1,2,3):
    ids=R[R.cls==c].ptid.values; m=wide[ids].mean(1).dropna(); v=m.values; r=detrend(v); s=S2(r); null=ar1_null(r,4999)
    out["class_mean_series"][NAME[c]]=dict(S2=round(s,3),p_ar1=round(float((np.sum(null>=s)+1)/5000),3),lag1_raw=round(ac(v,1),3),lag2_detr=round(ac(r,2),3),
        years=[int(t) for t in m.index],mean_series=[round(float(x),4) for x in v],detrended=[round(float(x),4) for x in r])
    print("class-mean",NAME[c],{k:out["class_mean_series"][NAME[c]][k] for k in ("S2","p_ar1","lag1_raw","lag2_detr")},flush=True)
# year-wise sign synchrony among bamboo points: share of points with positive detrended residual each year
sync={}
for c in (1,2,3):
    ids=R[R.cls==c].ptid.values; W=wide[ids].dropna(); res=np.apply_along_axis(detrend,0,W.values); share=(res>0).mean(1)
    sync[NAME[c]]=dict(years=[int(t) for t in W.index],share_positive=[round(float(x),3) for x in share],
                       max_abs_dev_from_half=round(float(np.max(np.abs(share-0.5))),3),
                       alternation_index=round(float(np.mean(np.sign(share[1:]-0.5)!=np.sign(share[:-1]-0.5))),3))
out["yearwise_sign_share"]=sync
print("bamboo year-wise share positive:",sync["Bamboo"]["share_positive"],flush=True)
R.to_csv(os.path.join(DATA,"rev_periodicity_points.csv"),index=False)
json.dump(out,open(os.path.join(OUT,"canonical_v2_E_periodicity.json"),"w"),indent=1); print("SAVED")
