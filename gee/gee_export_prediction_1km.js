/**** gee_export_prediction_1km.js *********************************************
 * 在 GEE Code Editor 貼上執行:用全部純林型樣本訓練 RF,對全島特徵影像逐像元
 * 分類,聚合成 1 km 多數類,匯出 GeoTIFF 到 Google Drive(EPSG:3826)。
 *
 * 你需要做的:
 *  1) 把「第四次森林資源調查林地蓄積圖1061」(或品質處理林型分布圖)上傳為 GEE asset,
 *     把下面 ASSET 改成你的 asset id。  (Assets → New → Shapefile;上傳 .shp/.shx/.dbf/.prj)
 *  2) 按 Run → 到 Tasks 分頁按 Run 匯出 → 完成後從 Drive 下載 taiwan_foresttype_1km.tif。
 *  3) 本機用 compare_survey_vs_prediction.py 與調查參考圖比對。
 *
 * 特徵與 01_gee_build_training.py 一致(S2 諧波物候 + S1 + 地形)。
 ******************************************************************************/

var ASSET = 'users/YOUR_NAME/stock1061';   // <-- 改成你的 asset id
var YEAR_START = '2023-01-01', YEAR_END = '2024-12-31';
var aoi = ee.FeatureCollection(ASSET);
var region = aoi.geometry().bounds();

// ---- 標籤:純林型 闊=1 針=2 竹=3(對齊本地參考圖類碼)----
var PURE = ee.Dictionary({'闊葉樹林型':1, '針葉樹林型':2, '竹林':3});
var pure = aoi.filter(ee.Filter.inList('TypeName', PURE.keys()))
              .filter(ee.Filter.gte('Area_Ha', 0.5))
              .map(function(ft){
                 var geom = ft.geometry().buffer(-20);
                 return ee.Feature(geom, {cls: PURE.get(ft.get('TypeName'))})
                          .set('ok', geom.area(1).gt(0));
              }).filter(ee.Filter.eq('ok', 1));

// ---- Sentinel-2 去雲 + 指標 ----
function maskS2(img){
  var scl = img.select('SCL');
  var good = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return img.updateMask(good).divide(10000).copyProperties(img, ['system:time_start']);
}
function addIdxS2(b){
  return b.addBands([
    b.normalizedDifference(['B8','B4']).rename('NDVI'),
    b.normalizedDifference(['B8','B11']).rename('NDMI'),
    b.normalizedDifference(['B8','B12']).rename('NBR'),
    b.normalizedDifference(['B8','B5']).rename('reNDVI'),
    b.select('B7').divide(b.select('B5')).subtract(1).rename('CIre'),
    b.expression('2.5*(N-R)/(N+6*R-7.5*B+1)',
      {N:b.select('B8'),R:b.select('B4'),B:b.select('B2')}).rename('EVI')
  ]);
}
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi).filterDate(YEAR_START,YEAR_END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',60))
  .map(maskS2).map(addIdxS2);

// ---- 諧波回歸 → mean/std/amp/phase/percentile ----
var PI = Math.PI;
function addT(img){
  var t = ee.Number(img.date().difference('2000-01-01','year'));
  var tb = ee.Image.constant(t).float().rename('t');
  return img.addBands(tb).addBands(ee.Image([
    tb.multiply(2*PI).cos().rename('cos1'), tb.multiply(2*PI).sin().rename('sin1'),
    tb.multiply(4*PI).cos().rename('cos2'), tb.multiply(4*PI).sin().rename('sin2'),
    ee.Image.constant(1).rename('const')]));
}
function harm(coll, band){
  var indep = ['const','cos1','sin1','cos2','sin2'];
  var fit = coll.map(addT).select(indep.concat([band]))
                .reduce(ee.Reducer.linearRegression(indep.length,1));
  var coef = fit.select('coefficients').arrayProject([0]).arrayFlatten([indep]);
  var amp1 = coef.select('cos1').hypot(coef.select('sin1')).rename(band+'_amp1');
  var pha1 = coef.select('sin1').atan2(coef.select('cos1')).rename(band+'_phase1');
  var amp2 = coef.select('cos2').hypot(coef.select('sin2')).rename(band+'_amp2');
  var mean = coll.select(band).mean().rename(band+'_mean');
  var sd   = coll.select(band).reduce(ee.Reducer.stdDev()).rename(band+'_std');
  var p    = coll.select(band).reduce(ee.Reducer.percentile([10,90])).rename([band+'_p10',band+'_p90']);
  return ee.Image.cat([mean,sd,amp1,pha1,amp2,p]);
}
var S2B = ['NDVI','EVI','NDMI','reNDVI','CIre','NBR'];
var s2feat = ee.Image.cat(S2B.map(function(b){return harm(s2,b);}));

// ---- Sentinel-1 ----
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi).filterDate(YEAR_START,YEAR_END)
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))
  .filter(ee.Filter.eq('orbitProperties_pass','ASCENDING'))
  .map(function(img){
     var vv=img.select('VV'), vh=img.select('VH');
     return img.addBands([vh.subtract(vv).rename('RATIO'),
                          vh.multiply(4).divide(vv.add(vh)).rename('RVI')])
               .copyProperties(img,['system:time_start']);
  });
var s1feat = ee.Image.cat(['VV','VH','RATIO','RVI'].map(function(b){return harm(s1,b);}));

// ---- 地形 ----
var dem = ee.Image('USGS/SRTMGL1_003');
var terrain = ee.Image.cat([dem.rename('elev'),
  ee.Terrain.slope(dem).rename('slope'),
  ee.Terrain.aspect(dem).cos().rename('aspect_cos')]);

var features = ee.Image.cat([s2feat,s1feat,terrain]).toFloat();
var bands = features.bandNames();

// ---- 訓練樣本(polygon 取均值)----
var training = features.sampleRegions({collection: pure, properties:['cls'], scale:10, tileScale:4, geometries:false});
var rf = ee.Classifier.smileRandomForest(300).train(training, 'cls', bands);

// ---- 全島逐像元分類 → 1 km 多數類 ----
var classified = features.classify(rf).rename('foresttype');
var pred1km = classified.reduceResolution({reducer: ee.Reducer.mode(), maxPixels: 1024})
                        .reproject({crs:'EPSG:3826', scale:1000});

// ---- 預覽 ----
Map.centerObject(aoi, 7);
Map.addLayer(pred1km, {min:1,max:3,palette:['1b9e77','7570b3','d95f02']},
             '1km 推論 (1闊2針3竹)');

// ---- 匯出 1 km(小檔,給比對用)----
Export.image.toDrive({
  image: pred1km.toByte(),
  description: 'taiwan_foresttype_1km',
  folder: 'forest_lulc',
  region: region, scale: 1000, crs: 'EPSG:3826', maxPixels: 1e10
});

// ---- (選用)匯出 10 m 精細圖,檔較大 ----
// Export.image.toDrive({image: classified.toByte(), description:'taiwan_foresttype_10m',
//   folder:'forest_lulc', region: region, scale:10, crs:'EPSG:3826', maxPixels:1e13});
