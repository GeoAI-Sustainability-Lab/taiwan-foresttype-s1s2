"""
03a_gee_export_timeseries.py
============================
為深度時序模型(TempCNN/GRU/LTAE)匯出「規則月合成時序堆疊」T×C。

與 01 的差別:01 匯出每 polygon 的諧波『摘要』特徵(給 RF);
本檔匯出每 polygon 每月的 S2+S1 通道值(原始 SITS),交給 03_deep_temporal.py。

輸出: 每 polygon 一列,欄位 = {band}_m00..m23 (24 個月) + 靜態 elev/slope/aspect + cls + x,y
後處理 (reshape 成 npz) 範例見檔尾。

執行前: pip install earthengine-api ; earthengine authenticate
"""
import ee
ee.Initialize()

ASSET = 'users/YOUR_NAME/stock1061'
START = ee.Date('2023-01-01'); N_MONTHS = 24
SCALE = 10; MIN_AREA_HA = 0.5; NEG_BUFFER_M = -20; PER_CLASS = 6000
EXPORT_FOLDER = 'forest_lulc'
BANDS = ['NDVI', 'reNDVI', 'NDMI', 'VV', 'VH', 'RATIO']
PURE = {'闊葉樹林型': 0, '針葉樹林型': 1, '竹林': 2}

aoi = ee.FeatureCollection(ASSET)

def mask_s2(img):
    scl = img.select('SCL')
    good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    b = img.updateMask(good).divide(10000)
    ndvi = b.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndmi = b.normalizedDifference(['B8', 'B11']).rename('NDMI')
    ren = b.normalizedDifference(['B8', 'B5']).rename('reNDVI')
    return ndvi.addBands([ndmi, ren]).copyProperties(img, ['system:time_start'])

def prep_s1(img):
    vv = img.select('VV'); vh = img.select('VH')
    rat = vh.subtract(vv).rename('RATIO')
    return vv.addBands([vh, rat]).copyProperties(img, ['system:time_start'])

s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(aoi)
      .filterDate(START, START.advance(N_MONTHS, 'month'))
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(mask_s2))
s1 = (ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi)
      .filterDate(START, START.advance(N_MONTHS, 'month'))
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
      .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).map(prep_s1))

def monthly_stack(coll, bands):
    def one(m):
        m = ee.Number(m); t0 = START.advance(m, 'month'); t1 = t0.advance(1, 'month')
        comp = coll.filterDate(t0, t1).median()
        # 缺值用全期中位數回填,確保每月每通道都有值(深度模型需規則長度)
        comp = comp.unmask(coll.median())
        suff = ee.String('_m').cat(m.format('%02d'))
        return comp.select(bands).rename([ee.String(b).cat(suff) for b in bands])
    imgs = ee.List.sequence(0, N_MONTHS-1).map(one)
    return ee.ImageCollection(imgs).toBands()

stack = ee.Image.cat([monthly_stack(s2, ['NDVI', 'reNDVI', 'NDMI']),
                      monthly_stack(s1, ['VV', 'VH', 'RATIO'])])
dem = ee.Image('USGS/SRTMGL1_003')
static = ee.Image.cat([dem.rename('elev'), ee.Terrain.slope(dem).rename('slope'),
                       ee.Terrain.aspect(dem).cos().rename('aspect_cos')])
allbands = stack.addBands(static).toFloat()

def label_clean(fc):
    fc = fc.filter(ee.Filter.inList('TypeName', list(PURE.keys()))).filter(ee.Filter.gte('Area_Ha', MIN_AREA_HA))
    def f(ft):
        g = ft.geometry().buffer(NEG_BUFFER_M)
        c = g.centroid(10).coordinates()
        return ee.Feature(g, {'cls': ee.Dictionary(PURE).get(ft.get('TypeName')),
                              'x': c.get(0), 'y': c.get(1)}).set('ok', g.area(1).gt(0))
    return fc.map(f).filter(ee.Filter.eq('ok', 1))

poly = label_clean(aoi)
def take(c): return poly.filter(ee.Filter.eq('cls', c)).randomColumn('r', 42).sort('r').limit(PER_CLASS)
bal = take(0).merge(take(1)).merge(take(2))

table = allbands.reduceRegions(bal, ee.Reducer.mean(), SCALE, tileScale=4)
ee.batch.Export.table.toDrive(collection=table, description='forest3_sits',
                              folder=EXPORT_FOLDER, fileFormat='CSV').start()
print('Export started -> Drive/%s/forest3_sits.csv' % EXPORT_FOLDER)

# ---------------------------------------------------------------------------
# 下載 CSV 後,用以下程式 reshape 成 03_deep_temporal.py 需要的 npz:
#
# import pandas as pd, numpy as np
# df = pd.read_csv('forest3_sits.csv')
# bands=['NDVI','reNDVI','NDMI','VV','VH','RATIO']; T=24
# seq=np.stack([np.stack([df[f'{b}_m{m:02d}'] for m in range(T)],1) for b in bands],2)  # N,T,C
# static=df[['elev','slope','aspect_cos']].values
# np.savez('sits.npz', X_seq=seq, X_static=static, y=df['cls'].values.astype(int),
#          xy=df[['x','y']].values, ch_names=bands, st_names=['elev','slope','aspect_cos'])
# 然後: python 03_deep_temporal.py --npz sits.npz
