# -*- coding: utf-8 -*-
"""Fig. 3 (a, b) of the manuscript: seasonal phenology of one representative stand per class, observations of 2023 to 2024 and the
FIRST-ORDER harmonic fit y(t) = c0 + a cos(2 pi t) + b sin(2 pi t), t in decimal years since 2000-01-01, the same form and time origin
as the 34-feature exporter (gee/05_feature_exporter_34.js). The submitted version of the figure had been drawn with a second-order fit
(annual plus semi-annual terms); the revised figure uses the first-order fit that produces the features.
Data: data/analyses/fig3_ts_points.csv, the Sentinel-2 NDMI and Sentinel-1 ascending VH observations of the three stands exported from GEE.
Outputs: figures/fig3_phenology_panels.png/.pdf (panels a and b), figures/fig3_phenology.png (composite with the unchanged chips
figures/fig3_chips.png, if present) and results/analyses/fig3_fit.json (coefficients, amplitude, phase, peak DOY and R^2, both orders)."""
import numpy as np, pandas as pd, json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))
DATA=os.path.join(REPO,'data','analyses'); RES=os.path.join(REPO,'results','analyses'); FIG=os.path.join(REPO,'figures')
d=pd.read_csv(os.path.join(DATA,'fig3_ts_points.csv')); d['date']=pd.to_datetime(d['date']); d['doy']=d['date'].dt.dayofyear
d['t']=(d['date']-pd.Timestamp('2000-01-01')).dt.days/365.25          # decimal years since 2000-01-01, as in the exporter
CLS=['Broadleaf','Conifer','Bamboo']; COL={'Broadleaf':'#1b9e77','Conifer':'#7570b3','Bamboo':'#d95f02'}
CHIPS=[('(c)','2024-04-04'),('(d)','2024-08-02'),('(e)','2024-11-05')]
def X(t,order):
    cols=[np.ones_like(t)]
    for k in range(1,order+1): cols+=[np.cos(2*np.pi*k*t),np.sin(2*np.pi*k*t)]
    return np.vstack(cols).T
def fit(e,order):
    beta,*_=np.linalg.lstsq(X(e['t'].values,order),e['value'].values,rcond=None)
    res=e['value'].values-X(e['t'].values,order)@beta; r2=1-(res**2).sum()/((e['value'].values-e['value'].mean())**2).sum()
    return beta,float(r2)
# evaluation grid: DOY 1..365 of 2024 expressed in decimal years since 2000-01-01
DOY=np.arange(1,366); T=((pd.Timestamp('2024-01-01')-pd.Timestamp('2000-01-01')).days+DOY-1)/365.25
rec={'protocol':'first-order harmonic fit y(t)=c0+a cos(2 pi t)+b sin(2 pi t), t = decimal years since 2000-01-01, least squares per stand; one representative stand per class, observations 2023-01 to 2024-12; second-order fit given for comparison','stands':{}}
fig=plt.figure(figsize=(7.0,2.62),dpi=300)
gs=fig.add_gridspec(1,2,left=0.082,right=0.918,top=0.80,bottom=0.215,wspace=0.06)
axA=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1])
for ax,sensor,ttl in ((axA,'S2_NDMI','(a) Optical NDMI phenology'),(axB,'S1_VH','(b) Sentinel-1 ascending VH phenology')):
    for c in CLS:
        e=d[(d.sensor==sensor)&(d.cls==c)]
        b1,r1=fit(e,1); b2,r2=fit(e,2)
        y=X(T,1)@b1
        ax.scatter(e['doy'],e['value'],s=5,color=COL[c],alpha=0.45,linewidths=0,zorder=2)
        ax.plot(DOY,y,color=COL[c],lw=1.6,zorder=3)
        A=float(np.hypot(b1[1],b1[2])); ph=float(np.arctan2(b1[2],b1[1]))
        rec['stands'][sensor+'|'+c]=dict(n=int(len(e)),order1=dict(c0=round(float(b1[0]),4),a=round(float(b1[1]),4),b=round(float(b1[2]),4),amplitude=round(A,4),phase_rad=round(ph,3),peak_doy=int(DOY[y.argmax()]),trough_doy=int(DOY[y.argmin()]),range=round(float(y.max()-y.min()),4),R2=round(r1,3)),
                                          order2=dict(amplitude_annual=round(float(np.hypot(b2[1],b2[2])),4),amplitude_semiannual=round(float(np.hypot(b2[3],b2[4])),4),R2=round(r2,3)))
    # broadleaf observations nearest to the chip dates
    e=d[(d.sensor==sensor)&(d.cls=='Broadleaf')].groupby('date',as_index=False).agg(value=('value','mean'),doy=('doy','first'))   # two overlapping S2 tiles share a date: mean
    for lab,ds in CHIPS:
        i=(e['date']-pd.Timestamp(ds)).abs().idxmin(); x0,y0=e.loc[i,'doy'],e.loc[i,'value']
        ax.scatter([x0],[y0],s=60,marker='s',facecolors='none',edgecolors='black',linewidths=1.1,zorder=4)
        dx,dy=(6,0.010) if sensor=='S2_NDMI' else (6,0.45)
        if lab=='(c)': dy=-dy*3.2 if sensor=='S2_NDMI' else -dy*3.0
        ax.annotate(lab,(x0,y0),xytext=(x0+dx,y0+dy),fontsize=7,fontweight='bold')
        rec['stands'].setdefault('chip_marks',{})[sensor+'|'+lab]=dict(date=str(e.loc[i,'date'].date()),doy=int(x0),value=round(float(y0),4))
    ax.set_xlim(1,365); ax.set_xticks([1,60,120,180,240,300,365]); ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3,lw=0.5); ax.set_title(ttl,fontsize=8.5,loc='left'); ax.set_xlabel('Day of Year (DOY)',fontsize=7.5)
axA.set_ylabel('NDMI',fontsize=8)
axB.yaxis.tick_right(); axB.yaxis.set_label_position('right'); axB.set_ylabel('VH (dB)',fontsize=8)
h=[Line2D([0],[0],color=COL[c],lw=2,label=c) for c in CLS]+[Line2D([0],[0],marker='s',markerfacecolor='none',markeredgecolor='black',markersize=7,lw=0,label='Broadleaf observation at chip dates (c–e)')]
fig.legend(handles=h,loc='upper center',ncol=4,frameon=False,fontsize=7.2,bbox_to_anchor=(0.5,0.995),handlelength=2.2,columnspacing=1.6)
fig.savefig(os.path.join(FIG,'fig3_phenology_panels.png'),dpi=300); fig.savefig(os.path.join(FIG,'fig3_phenology_panels.pdf'))
# composite with the unchanged chips (c) to (e) of the submitted figure, if present
top=Image.open(os.path.join(FIG,'fig3_phenology_panels.png')).convert('RGB'); comp=top
chips_f=os.path.join(FIG,'fig3_chips.png')
if os.path.exists(chips_f):
    chips=Image.open(chips_f).convert('RGB'); W=top.size[0]; assert chips.size[0]==W,(top.size,chips.size)
    comp=Image.new('RGB',(W,top.size[1]+chips.size[1]),'white'); comp.paste(top,(0,0)); comp.paste(chips,(0,top.size[1]))
    comp.save(os.path.join(FIG,'fig3_phenology.png'),dpi=(300,300))
rec['composite_px']=list(comp.size); rec['panels_px']=list(top.size)
json.dump(rec,open(os.path.join(RES,'fig3_fit.json'),'w'),indent=1)
for k,v in rec['stands'].items():
    if k!='chip_marks': print('%-20s n=%3d  order1: A=%.3f phase=%+.2f peak DOY %3d range %.3f R2=%.3f | order2: A1=%.3f A2=%.3f R2=%.3f'%(k,v['n'],v['order1']['amplitude'],v['order1']['phase_rad'],v['order1']['peak_doy'],v['order1']['range'],v['order1']['R2'],v['order2']['amplitude_annual'],v['order2']['amplitude_semiannual'],v['order2']['R2']))
print(rec['stands']['chip_marks']); print('composite',comp.size)
