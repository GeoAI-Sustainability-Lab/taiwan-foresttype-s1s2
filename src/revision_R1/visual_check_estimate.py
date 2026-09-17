# -*- coding: utf-8 -*-
"""Estimator for the recent-image check of 50 existing reference points (Section III-F, Table S4 of the manuscript).
Input : data/R1/visual_check_50pts.csv  (per point: stratum, image date, agreement at the pixel scale of the classifier's
        sampling unit (3 x 3 Sentinel-2 pixels; Y / N / U = undeterminable), confidence, current cover, stand edge within 30 m)
Strata: flagged = census of the 20 points whose 2023-2025 Sentinel-2 signatures depart most from their class centroid;
        random  = 30 class-stratified random points. Agreement rates exclude U. Wilson 95% intervals.
Design-weighted agreement over the 755 points: flagged stratum weight 20/755 (census); random strata weighted by
class size minus the flagged points of that class, with finite-population correction.
Output: results/R1/visual_check_estimates.json"""
import csv,collections,math,json,os
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
rows=list(csv.DictReader(open(os.path.join(REPO,'data','R1','visual_check_50pts.csv'),encoding='utf8')))
N={'Broadleaf':243,'Conifer':260,'Bamboo':252}; NT=755; key='px_agree'
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return [round(c-h,3),round(c+h,3)]
def rate(rs):
    d=[r for r in rs if r[key] in ('Y','N')]; k=sum(r[key]=='Y' for r in d)
    return dict(Y=k,N=len(d)-k,U=len(rs)-len(d),rate=round(k/len(d),3) if d else None,wilson=wilson(k,len(d)) if d else None)
o={}
for st in ['random','flagged']:
    rs=[r for r in rows if r['stratum']==st]; o[st]=rate(rs)
    for cl in N: o[st+'_'+cl]=rate([r for r in rs if r['cls_name']==cl])
flag_by=collections.Counter(r['cls_name'] for r in rows if r['stratum']=='flagged')
fl=[r for r in rows if r['stratum']=='flagged' and r[key] in ('Y','N')]; pf=sum(r[key]=='Y' for r in fl)/len(fl)
tot=pf*20; var=0.0
for cl in N:
    rc=[r for r in rows if r['stratum']=='random' and r['cls_name']==cl and r[key] in ('Y','N')]
    n=len(rc); k=sum(r[key]=='Y' for r in rc); p=k/n; Nh=N[cl]-flag_by[cl]
    tot+=p*Nh; var+=(Nh/NT)**2*p*(1-p)/n*(1-n/Nh)
o['design_weighted']=dict(rate=round(tot/NT,3),SE=round(math.sqrt(var),3))
ed=[r for r in rows if r['edge'] in ('Y','N')]
o['edge_within_30m']=dict(recorded=len(ed),Y=sum(r['edge']=='Y' for r in ed))
o['imagery_2024_or_later']=sum(1 for r in rows if r['image_date'][:4]>='2024')
strict=[r for r in rows if r['stratum']=='random' and r['image_date'][:4]>='2024' and r[key] in ('Y','N')]
o['random_strict_2024plus']=dict(Y=sum(r[key]=='Y' for r in strict),n=len(strict))
os.makedirs(os.path.join(REPO,'results','R1'),exist_ok=True)
json.dump(o,open(os.path.join(REPO,'results','R1','visual_check_estimates.json'),'w'),indent=1)
print(json.dumps(o,indent=1))
