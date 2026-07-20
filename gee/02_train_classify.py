"""
02_train_classify.py
====================
三類林型分類:spatial-block 交叉驗證 + Random Forest,並做 S1 / S2 / fusion 消融。

重點(對應審稿會質疑的點):
  - 用 spatial block CV(以 TWD97 網格切 block)而非隨機 CV,避免同一林分鄰近像元
    同時落在 train/test 造成精度灌水 → 這是能不能投 Q1 的關鍵。
  - 報告 overall / per-class PA(producer)/UA(user)/F1 + 混淆矩陣。
  - 消融:只用 S2、只用 S1、S1+S2、再加地形,量化 SAR 對「常綠弱物候」的補償。

用法:
  真實資料: python 02_train_classify.py --csv forest3_training_features.csv
  自我測試: python 02_train_classify.py --selftest      (用模擬物候特徵驗證流程)
"""
import argparse, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

CLASS_NAMES = {0: 'Broadleaf', 1: 'Conifer', 2: 'Bamboo'}

# ----------------------------------------------------------------------------
def spatial_block_folds(x, y, block_km=10, k=5, seed=0):
    """以 block_km 網格指派 block,再把 block 隨機分到 k 個 fold。回傳 fold 標籤。"""
    bx = np.floor(x / (block_km * 1000)).astype(int)
    by = np.floor(y / (block_km * 1000)).astype(int)
    blocks = (bx.astype(str) + '_' + by.astype(str))
    uniq = pd.unique(blocks)
    rng = np.random.default_rng(seed); rng.shuffle(uniq)
    fold_of = {b: i % k for i, b in enumerate(uniq)}
    return blocks.map(fold_of).values if hasattr(blocks, 'map') else np.array([fold_of[b] for b in blocks])

def evaluate(df, feat_cols, label_col='cls', xcol='x', ycol='y', block_km=10, k=5, tag=''):
    X = df[feat_cols].values; y = df[label_col].values.astype(int)
    folds = spatial_block_folds(df[xcol].values, df[ycol].values, block_km, k)
    yt, yp = [], []
    for f in range(k):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                     n_jobs=-1, class_weight='balanced', random_state=0)
        clf.fit(X[tr], y[tr])
        yp.append(clf.predict(X[te])); yt.append(y[te])
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    oa = accuracy_score(yt, yp); f1 = f1_score(yt, yp, average='macro')
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    pa = cm.diagonal() / cm.sum(1).clip(1)      # producer acc (recall)
    ua = cm.diagonal() / cm.sum(0).clip(1)      # user acc (precision)
    print(f"\n=== {tag} === (spatial-block CV, block={block_km}km, k={k})")
    print(f"Overall Acc: {oa:.3f} | macro-F1: {f1:.3f}")
    print("            " + "  ".join(f"{CLASS_NAMES[i]:>10}" for i in range(3)))
    for i in range(3):
        print(f"true {CLASS_NAMES[i]:<9}" + "  ".join(f"{cm[i,j]:10d}" for j in range(3)))
    for i in range(3):
        print(f"  {CLASS_NAMES[i]:<10} PA={pa[i]:.3f}  UA={ua[i]:.3f}")
    return {'tag': tag, 'OA': oa, 'macroF1': f1,
            **{f'PA_{CLASS_NAMES[i]}': pa[i] for i in range(3)}}

def feature_groups(cols):
    s2 = [c for c in cols if any(c.startswith(p) for p in
          ['NDVI', 'EVI', 'NDMI', 'reNDVI', 'CIre', 'NBR'])]
    s1 = [c for c in cols if any(c.startswith(p) for p in
          ['VV', 'VH', 'RATIO', 'RVI'])]
    terr = [c for c in cols if c in ('elev', 'slope', 'aspect_cos')]
    return s2, s1, terr

def run(df):
    drop = {'cls', 'x', 'y', 'system:index', '.geo', 'ok', 'r'}
    feat_cols = [c for c in df.columns if c not in drop and df[c].dtype != object]
    df = df.dropna(subset=feat_cols + ['cls', 'x', 'y']).reset_index(drop=True)
    s2, s1, terr = feature_groups(feat_cols)
    print(f"n={len(df)}  |S2 feats|={len(s2)} |S1|={len(s1)} |terrain|={len(terr)}")
    res = []
    res.append(evaluate(df, s2,            tag='S2 only (optical phenology)'))
    res.append(evaluate(df, s1,            tag='S1 only (SAR)'))
    res.append(evaluate(df, s2 + s1,       tag='S1 + S2 fusion'))
    res.append(evaluate(df, s2 + s1 + terr, tag='S1 + S2 + terrain'))
    out = pd.DataFrame(res)
    print("\n================ ABLATION SUMMARY ================")
    print(out.round(3).to_string(index=False))
    return out

# ----------------------------------------------------------------------------
def make_synthetic(n=4500, seed=1):
    """模擬三類的諧波/物候特徵以驗證流程(非真實資料)。
    設計成:常綠闊葉 vs 針葉 在『季節振幅』上重疊(弱物候),但在 SAR 比值/紅邊/高程
    上可分 → 模擬真實難點,檢查 fusion 是否如預期優於 single-sensor。"""
    rng = np.random.default_rng(seed); rows = []
    # 在台灣範圍內灑點(TWD97 m)
    for c in range(3):
        for _ in range(n // 3):
            x = rng.uniform(160000, 340000); y = rng.uniform(2430000, 2790000)
            if c == 0:      # 闊葉:低-中海拔、弱季節振幅、SAR 高 volume scattering
                elev = rng.normal(900, 500); ndvi_amp = abs(rng.normal(0.05, 0.02))
                reNDVI_mean = rng.normal(0.55, 0.05); ratio_mean = rng.normal(-7, 1.2)
            elif c == 1:    # 針葉:高海拔、季節振幅也小但紅邊/含水不同、SAR 結構不同
                elev = rng.normal(2200, 600); ndvi_amp = abs(rng.normal(0.04, 0.02))
                reNDVI_mean = rng.normal(0.45, 0.05); ratio_mean = rng.normal(-9, 1.2)
            else:           # 竹:中低海拔、季節振幅較大、SAR 比值偏高
                elev = rng.normal(700, 400); ndvi_amp = abs(rng.normal(0.12, 0.03))
                reNDVI_mean = rng.normal(0.50, 0.06); ratio_mean = rng.normal(-5.5, 1.0)
            rows.append(dict(cls=c, x=x, y=y, elev=elev, slope=rng.uniform(5, 40),
                aspect_cos=rng.uniform(-1, 1),
                NDVI_mean=rng.normal(0.8, 0.05), NDVI_amp1=ndvi_amp,
                NDVI_phase1=rng.uniform(-3.14, 3.14), NDVI_std=ndvi_amp*1.3,
                EVI_mean=rng.normal(0.45, 0.05), EVI_amp1=ndvi_amp*0.8,
                NDMI_mean=rng.normal(0.3, 0.06), NDMI_amp1=abs(rng.normal(0.05, 0.02)),
                reNDVI_mean=reNDVI_mean, reNDVI_amp1=abs(rng.normal(0.03, 0.01)),
                CIre_mean=rng.normal(2.0, 0.4), NBR_mean=rng.normal(0.6, 0.05),
                VV_mean=rng.normal(-8, 1.5), VV_amp1=abs(rng.normal(0.5, 0.2)),
                VH_mean=rng.normal(-14, 1.5), VH_amp1=abs(rng.normal(0.6, 0.2)),
                RATIO_mean=ratio_mean, RATIO_amp1=abs(rng.normal(0.4, 0.15)),
                RVI_mean=rng.normal(0.6, 0.1)))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest or not a.csv:
        print(">> SELF-TEST on synthetic phenology features (pipeline validation only)\n")
        df = make_synthetic()
    else:
        df = pd.read_csv(a.csv)
    run(df)
