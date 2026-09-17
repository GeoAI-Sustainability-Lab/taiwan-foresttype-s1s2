// ===== 34-feature exporter (rebuilt for revision R1) and next-year classification =====
// Feature definitions (Section II-B/II-C of the manuscript):
//   Sentinel-2 SR Harmonized, CLOUDY_PIXEL_PERCENTAGE < 60 plus SCL mask (classes 3, 8, 9, 10, 11 removed);
//   first-order harmonic fit of NDVI, NDMI, NBR and reNDVI with t = years since 2000-01-01 -> _mean (collection mean), _amp, _pha;
//   Sentinel-1 GRD IW ascending VV + VH in dB, RATIO = VH - VV; wet season May-October, dry season November-April;
//   ERA5-Land monthly aggregates (temperature, dew point, soil moisture, precipitation); SRTM elevation, slope and cos(aspect in radians);
//   reduceRegions mean at scale 10 m. Column order of the exported table is the alphabetical order used in data/features_train4yr.csv.
// Rebuilt exporter checked against the archived feature table at a 95-point subsample: climate (13) r = 1.000, Sentinel-1 (9) r = 1.000,
//   Sentinel-2 means and amplitudes r >= 0.999, phases r 0.93-1.00 (angle wrap-around), slope 1.000.
// The map export at the end (all 755 points, smileRandomForest 300 trees, minLeaf 3, seed 0, 2025-06 to 2026-05 composite, no prior weighting)
//   is provided for completeness; the island-wide map shown in the manuscript (Fig. 6) is produced by 04_island_prior_weighted_20m.js.
// Replace the asset path below with your own copy of the reference points (data/features_train4yr.csv carries lon, lat and cls).
var P = 'projects/remotesensingpractice-490015/assets/';
var pts = ee.FeatureCollection(P + 'points_755');
var taiwan = ee.Geometry.Rectangle([120.0, 21.85, 122.05, 25.35]);
var WET = ee.Filter.calendarRange(5, 10, 'month');
var DRY = ee.Filter.or(ee.Filter.calendarRange(11, 12, 'month'), ee.Filter.calendarRange(1, 4, 'month'));
var PI = Math.PI;
function maskS2(img) {
  var scl = img.select('SCL');
  var good = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  var b = img.updateMask(good).divide(10000);
  return b.normalizedDifference(['B8', 'B4']).rename('NDVI')
    .addBands([b.normalizedDifference(['B8', 'B11']).rename('NDMI'), b.normalizedDifference(['B8', 'B12']).rename('NBR'), b.normalizedDifference(['B8', 'B5']).rename('reNDVI')])
    .copyProperties(img, ['system:time_start']);
}
function addT(i) {
  var t = ee.Number(i.date().difference('2000-01-01', 'year')); var ti = ee.Image.constant(t).float();
  return i.addBands(ee.Image([ee.Image.constant(1).rename('const'), ti.multiply(2 * PI).cos().rename('c1'), ti.multiply(2 * PI).sin().rename('s1')]));
}
function harm(coll, band) {
  var ind = ['const', 'c1', 's1'];
  var fit = coll.map(addT).select(ind.concat([band])).reduce(ee.Reducer.linearRegression(3, 1));
  var co = fit.select('coefficients').arrayProject([0]).arrayFlatten([ind]);
  return ee.Image.cat([coll.select(band).mean().rename(band + '_mean'), co.select('c1').hypot(co.select('s1')).rename(band + '_amp'), co.select('s1').atan2(co.select('c1')).rename(band + '_pha')]);
}
function features(S, E) {
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(taiwan).filterDate(S, E).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(maskS2);
  var s2f = ee.Image.cat([harm(s2, 'NDVI'), harm(s2, 'NDMI'), harm(s2, 'NBR'), harm(s2, 'reNDVI')]);
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(taiwan).filterDate(S, E).filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    .map(function (i) { return i.select(['VV', 'VH']).addBands(i.select('VH').subtract(i.select('VV')).rename('RATIO')).copyProperties(i, ['system:time_start']); });
  var w = s1.filter(WET), d = s1.filter(DRY); var vh = harm(s1, 'VH');
  var s1f = ee.Image.cat([s1.select('VV').mean().rename('VV_mean'), s1.select('VH').mean().rename('VH_mean'), s1.select('RATIO').mean().rename('RATIO_mean'),
    s1.select('RATIO').reduce(ee.Reducer.stdDev()).rename('RATIO_std'), vh.select('VH_amp'), vh.select('VH_pha'),
    w.select('VV').mean().subtract(d.select('VV').mean()).rename('VV_wetdry'), w.select('VH').mean().subtract(d.select('VH').mean()).rename('VH_wetdry'),
    w.select('RATIO').mean().subtract(d.select('RATIO').mean()).rename('RATIO_wetdry')]);
  var era = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').filterDate(S, E);
  var temp = era.select('temperature_2m').mean().subtract(273.15).rename('temp_mean'); var dewp = era.select('dewpoint_temperature_2m').mean().subtract(273.15).rename('dewp_mean');
  var sm = era.select('volumetric_soil_water_layer_1'); var pr = era.select('total_precipitation_sum');
  var clim = ee.Image.cat([temp, dewp, temp.subtract(dewp).rename('dewdep'), sm.mean().rename('soilm_mean'), sm.filter(WET).mean().rename('soilm_wet'), sm.filter(DRY).mean().rename('soilm_dry'),
    sm.filter(WET).mean().subtract(sm.filter(DRY).mean()).rename('soilm_wetdry'), pr.mean().multiply(1000).rename('precip_mm'), pr.filter(WET).mean().multiply(1000).rename('precip_wet'), pr.filter(DRY).mean().multiply(1000).rename('precip_dry')]);
  var dem = ee.Image('USGS/SRTMGL1_003');
  var terr = ee.Image.cat([dem.rename('elev'), ee.Terrain.slope(dem).rename('slope'), ee.Terrain.aspect(dem).multiply(PI / 180).cos().rename('aspect_cos')]);
  return ee.Image.cat([s2f, s1f, clim, terr]).toFloat();
}
var F34 = ['NDVI_mean','NDVI_amp','NDVI_pha','NDMI_mean','NDMI_amp','NDMI_pha','NBR_mean','NBR_amp','NBR_pha','reNDVI_mean','reNDVI_amp','reNDVI_pha',
  'VV_mean','VH_mean','RATIO_mean','RATIO_std','VH_amp','VH_pha','VV_wetdry','VH_wetdry','RATIO_wetdry',
  'temp_mean','dewp_mean','dewdep','soilm_mean','soilm_wet','soilm_dry','soilm_wetdry','precip_mm','precip_wet','precip_dry','elev','slope','aspect_cos'];
var imgA = features('2021-01-01', '2025-01-01').select(F34);   // training window
var imgB = features('2025-06-01', '2026-06-01').select(F34);   // next-year window (mapping)
var trainA = imgA.reduceRegions({collection: pts, reducer: ee.Reducer.mean(), scale: 10, tileScale: 4}).filter(ee.Filter.notNull(F34));
var rf = ee.Classifier.smileRandomForest({numberOfTrees: 300, minLeafPopulation: 3, bagFraction: 1.0, seed: 0}).train(trainA, 'cls', F34);
var cls = imgB.classify(rf).rename('foresttype').toByte();
Map.centerObject(taiwan, 8); Map.addLayer(cls, {min: 1, max: 3, palette: ['1b9e77', '7570b3', 'd95f02']}, 'forest type v2 2025-26');
Export.image.toDrive({image: cls, description: 'taiwan_foresttype_20m_v2_2025_2026', folder: 'GEE_chips', region: taiwan, scale: 20, crs: 'EPSG:3826', maxPixels: 1e10});
Export.table.toDrive({collection: trainA.map(function (f) { return f.setGeometry(null); }), description: 'v2_features_A_2021_2024_755', folder: 'GEE_chips', fileFormat: 'CSV'});
