"""
01_gee_build_training.py
=========================
台灣三類林型(闊葉/針葉/竹)時序遙測分類 — Google Earth Engine 特徵與訓練樣本建立。

輸入: 已上傳為 GEE asset 的「第四次森林資源調查林地蓄積圖1061」(EPSG:3826)。
輸出: 每個樣本 polygon 一列、含 Sentinel-2 + Sentinel-1 諧波/物候特徵 + 地形特徵
      的訓練表 (Drive CSV)，供 02_train_classify.py 做 spatial-block CV 與 RF 分類。

關鍵設計(對應研究 gap):
  - 標籤只取「純林型」polygon (TypeName ∈ 闊葉樹林型/針葉樹林型/竹林)，混淆林另存做為
    soft-label / 不確定性分析,避免硬標籤雜訊。
  - 內縮負緩衝 (-20 m) 去除邊界混合像元。
  - S2 弱物候 → 同時抽「絕對反射率/指標統計」與「諧波相位/振幅」,讓模型能在常綠林
    用結構/紅邊/SWIR 補足季節振幅不足的部分。
  - S1 諧波 + VH/VV(對含水較不敏感、對結構較敏感) 做為 SAR 物候/結構特徵。
  - 地形(高程/坡度) 為強 covariate(台灣林型隨海拔分層)。

執行前: pip install earthengine-api ;  earthengine authenticate

重要說明(關於本 repo 所附之樣點):
  data/train_points.csv 為本研究實際使用之參考樣點,由較早一次的取樣流程產生,其確切之
  過濾設定已無法完整重建。經事後查核,相當比例之闊葉樣點落於較小(< 0.5 ha)或近邊界之
  圖斑,亦即下方的最小面積與內縮緩衝並未完全反映在所附樣點上(此空間支持限制已載明於
  手稿)。因此:
    - 若要重現本文結果,請直接使用所附之 data/train_points.csv 與特徵表;
    - 若要重新抽樣,本腳本即為建議之取樣設計(面積欄名已自動偵測,並會檢查過濾確實生效)。

NOTE (on the sample shipped with this repo):
  data/train_points.csv is the reference sample actually used; it was produced by an earlier
  sampling run whose exact filter settings could not be fully reconstructed. A post-hoc audit
  shows a substantial fraction of broadleaf points fall in small (< 0.5 ha) or edge-adjacent
  polygons, i.e. the minimum-area and negative-buffer steps below are not fully reflected in the
  shipped points (a spatial-support limitation of the reference sample). Therefore:
    - to reproduce the published results, use the provided data/train_points.csv and feature tables;
    - to draw a NEW sample, this script is the recommended design (the area field is auto-detected
      and the filter is checked to actually take effect).
"""
import ee
ee.Initialize()

# ----------------------------------------------------------------------------
# 0. 參數
# ----------------------------------------------------------------------------
ASSET = 'users/YOUR_NAME/stock1061'        # <-- 改成你上傳的 asset id
YEAR_START = '2023-01-01'
YEAR_END   = '2024-12-31'                    # 2 年時序,增加諧波穩定度
EXPORT_FOLDER = 'forest_lulc'
SAMPLES_PER_CLASS = 6000                     # 每類抽樣上限(平衡)
MIN_AREA_HA = 0.5                            # polygon 最小面積(≈50 個 S2 像元)
NEG_BUFFER_M = -20                           # 內縮負緩衝,去邊界混合像元
SCALE = 10

aoi = ee.FeatureCollection(ASSET)

# ----------------------------------------------------------------------------
# 1. 標籤:純林型 → 三類 (闊=0, 針=1, 竹=2)。樹種碼首字 A針 / B闊 / C竹。
# ----------------------------------------------------------------------------
PURE = {'闊葉樹林型': 0, '針葉樹林型': 1, '竹林': 2}

def _area_field(fc):
    # the uploaded asset may use 'Area_Ha' or 'Area_ha'; filtering on a missing field silently
    # does nothing, which is exactly how a minimum-area filter can fail unnoticed.
    names = ee.Feature(fc.first()).propertyNames().getInfo()
    for c in ('Area_Ha', 'Area_ha', 'AREA_HA', 'area_ha'):
        if c in names:
            print('[area field]', c); return c
    raise ValueError('no area field found; asset properties = %s' % names)

def label_and_clean(fc):
    afield = _area_field(fc)
    fc = fc.filter(ee.Filter.inList('TypeName', list(PURE.keys())))
    n1 = fc.size().getInfo()
    fc = fc.filter(ee.Filter.gte(afield, MIN_AREA_HA))
    n2 = fc.size().getInfo()
    print('[min-area filter] %d -> %d polygons (>= %.1f ha)' % (n1, n2, MIN_AREA_HA))
    assert n2 < n1, 'area filter removed nothing - check the field name!'
    def f(ft):
        cls = ee.Dictionary(PURE).get(ft.get('TypeName'))
        geom = ft.geometry().buffer(NEG_BUFFER_M)
        return ee.Feature(geom, {'cls': cls}).set('ok', geom.area(1).gt(0))
    fc = fc.map(f).filter(ee.Filter.eq('ok', 1))
    return fc

samples_poly = label_and_clean(aoi)

# ----------------------------------------------------------------------------
# 2. Sentinel-2 SR 去雲 + 指標
# ----------------------------------------------------------------------------
def mask_s2(img):
    scl = img.select('SCL')
    good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return (img.updateMask(good)
            .divide(10000)
            .copyProperties(img, ['system:time_start']))

def add_indices_s2(img):
    b = img
    ndvi = b.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndmi = b.normalizedDifference(['B8', 'B11']).rename('NDMI')        # 冠層含水
    nbr  = b.normalizedDifference(['B8', 'B12']).rename('NBR')
    reNDVI = b.normalizedDifference(['B8', 'B5']).rename('reNDVI')     # 紅邊
    cire = b.select('B7').divide(b.select('B5')).subtract(1).rename('CIre')
    evi = b.expression('2.5*(N-R)/(N+6*R-7.5*B+1)',
                       {'N': b.select('B8'), 'R': b.select('B4'), 'B': b.select('B2')}).rename('EVI')
    return img.addBands([ndvi, ndmi, nbr, reNDVI, cire, evi])

s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(aoi).filterDate(YEAR_START, YEAR_END)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
      .map(mask_s2).map(add_indices_s2))

# ----------------------------------------------------------------------------
# 3. 諧波回歸:對每個指標解 constant + 年/半年週期的 cos/sin。
#    係數 → mean / amplitude / phase = 你說的「圓形/波形」週期特徵的量化。
# ----------------------------------------------------------------------------
HARM_BANDS = ['NDVI', 'EVI', 'NDMI', 'reNDVI', 'CIre', 'NBR']

def add_time_bands(img):
    t = ee.Number(img.date().difference('2000-01-01', 'year'))
    tb = ee.Image.constant(t).float().rename('t')
    return img.addBands(tb).addBands(
        ee.Image([tb.multiply(2*3.141592653589793).cos().rename('cos1'),
                  tb.multiply(2*3.141592653589793).sin().rename('sin1'),
                  tb.multiply(4*3.141592653589793).cos().rename('cos2'),
                  tb.multiply(4*3.141592653589793).sin().rename('sin2'),
                  ee.Image.constant(1).rename('const')]))

def harmonic_features(coll, band):
    c = coll.map(add_time_bands)
    indep = ['const', 'cos1', 'sin1', 'cos2', 'sin2']
    fit = (c.select(indep + [band])
           .reduce(ee.Reducer.linearRegression(numX=len(indep), numY=1)))
    coef = fit.select('coefficients').arrayProject([0]).arrayFlatten([indep])
    a1 = coef.select('cos1'); b1 = coef.select('sin1')
    amp1 = a1.hypot(b1).rename(band + '_amp1')                    # 年週期振幅
    pha1 = b1.atan2(a1).rename(band + '_phase1')                  # 年週期相位(peak 時間)
    a2 = coef.select('cos2'); b2 = coef.select('sin2')
    amp2 = a2.hypot(b2).rename(band + '_amp2')
    mean = coll.select(band).mean().rename(band + '_mean')
    std  = coll.select(band).reduce(ee.Reducer.stdDev()).rename(band + '_std')
    p = coll.select(band).reduce(ee.Reducer.percentile([10, 90])).rename([band+'_p10', band+'_p90'])
    return ee.Image.cat([mean, std, amp1, pha1, amp2, p])

s2_feat = ee.Image.cat([harmonic_features(s2, b) for b in HARM_BANDS])

# ----------------------------------------------------------------------------
# 4. Sentinel-1:VV/VH/比值 的諧波 + 統計(SAR 結構/含水物候)
# ----------------------------------------------------------------------------
s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(aoi).filterDate(YEAR_START, YEAR_END)
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
      .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')))

def s1_prep(img):
    vv = img.select('VV'); vh = img.select('VH')
    ratio = vh.subtract(vv).rename('RATIO')          # dB 差 = VH/VV
    rvi = vh.multiply(4).divide(vv.add(vh)).rename('RVI')
    return img.addBands([ratio, rvi]).copyProperties(img, ['system:time_start'])

s1 = s1.map(s1_prep)
s1_feat = ee.Image.cat([harmonic_features(s1, b) for b in ['VV', 'VH', 'RATIO', 'RVI']])

# ----------------------------------------------------------------------------
# 5. 地形(強 covariate)
# ----------------------------------------------------------------------------
dem = ee.Image('USGS/SRTMGL1_003')   # 若有台灣 20m DEM 可換成自有 asset
terrain = ee.Image.cat([
    dem.rename('elev'),
    ee.Terrain.slope(dem).rename('slope'),
    ee.Terrain.aspect(dem).cos().rename('aspect_cos'),
])

# ----------------------------------------------------------------------------
# 6. 疊合所有特徵,逐 polygon 取均值,抽樣輸出
# ----------------------------------------------------------------------------
features = ee.Image.cat([s2_feat, s1_feat, terrain]).toFloat()

# 平衡抽樣:每類各取上限數量的 polygon
def take(fc, cls, n):
    return fc.filter(ee.Filter.eq('cls', cls)).randomColumn('r', 42)\
             .sort('r').limit(n)
balanced = take(samples_poly, 0, SAMPLES_PER_CLASS)\
    .merge(take(samples_poly, 1, SAMPLES_PER_CLASS))\
    .merge(take(samples_poly, 2, SAMPLES_PER_CLASS))

# polygon 中心做 spatial block 用的座標(TWD97)
def add_xy(ft):
    c = ft.geometry().centroid(10).coordinates()
    return ft.set({'x': c.get(0), 'y': c.get(1)})
balanced = balanced.map(add_xy)

table = features.reduceRegions(
    collection=balanced,
    reducer=ee.Reducer.mean(),
    scale=SCALE,
    tileScale=4)

ee.batch.Export.table.toDrive(
    collection=table,
    description='forest3_training_features',
    folder=EXPORT_FOLDER,
    fileFormat='CSV').start()
print('Export started -> Drive/%s/forest3_training_features.csv' % EXPORT_FOLDER)
print('完成後執行 02_train_classify.py')
