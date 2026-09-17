// ============================================================
// Recent-image check of 50 existing reference points (Section III-F / Table S4 of the manuscript)
// Usage: paste into the Earth Engine Code Editor and run. Step through the points with the buttons on the left,
// compare the 2025 Sentinel-2 composite with the high-resolution basemap, and record whether the point still carries
// its labelled class in data/R1/visual_check_50pts.csv (px_agree = Y / N / U, U = undeterminable).
// The interpretation reported in the manuscript was made at the scale of the 3 x 3 Sentinel-2 pixels centred on the point.
// The 50 points are embedded below (no asset upload needed).

// ============================================================

var PTS = [
  [562, 121.36803, 25.13667, 3, 'Y'],
  [624, 120.63372, 23.63831, 3, 'Y'],
  [530, 121.74523, 25.02726, 3, 'Y'],
  [358, 121.07331, 23.46724, 2, 'Y'],
  [456, 120.95955, 23.65079, 2, 'Y'],
  [348, 121.15158, 23.54251, 2, 'Y'],
  [633, 121.14496, 24.89342, 3, 'Y'],
  [512, 121.40280, 25.13758, 3, 'Y'],
  [558, 120.95264, 24.38009, 3, 'Y'],
  [516, 120.55551, 24.04576, 3, 'Y'],
  [621, 121.61941, 25.17825, 3, 'Y'],
  [706, 120.41218, 23.12132, 3, 'Y'],
  [555, 120.66337, 23.70635, 3, 'Y'],
  [373, 121.42916, 24.31307, 2, 'Y'],
  [376, 121.24288, 23.59214, 2, 'Y'],
  [363, 120.95808, 23.48339, 2, 'Y'],
  [600, 121.13555, 24.84520, 3, 'Y'],
  [225, 121.14668, 22.76810, 1, 'Y'],
  [663, 120.87505, 24.64770, 3, 'Y'],
  [725, 120.93578, 24.47517, 3, 'Y'],
  [529, 121.38982, 24.97980, 3, ''],
  [509, 120.91597, 23.88390, 3, ''],
  [603, 121.00551, 23.86139, 3, ''],
  [619, 120.94834, 24.70226, 3, ''],
  [614, 121.57125, 24.66899, 3, ''],
  [686, 121.37879, 24.96395, 3, ''],
  [715, 121.36839, 24.86480, 3, ''],
  [595, 121.74493, 25.02882, 3, ''],
  [513, 121.01387, 24.71283, 3, ''],
  [655, 121.68960, 25.11218, 3, ''],
  [551, 120.78592, 23.91026, 3, ''],
  [720, 120.93250, 24.37741, 3, ''],
  [132, 120.50227, 23.22985, 1, ''],
  [30, 121.23425, 24.08396, 1, ''],
  [93, 120.79190, 22.21845, 1, ''],
  [234, 120.77448, 21.97977, 1, ''],
  [55, 121.03318, 22.71779, 1, ''],
  [68, 120.39446, 22.84078, 1, ''],
  [188, 120.34289, 22.81720, 1, ''],
  [240, 120.76944, 22.77005, 1, ''],
  [70, 120.93962, 22.46081, 1, ''],
  [58, 120.43369, 23.17767, 1, ''],
  [366, 121.31137, 24.25579, 2, ''],
  [437, 120.94573, 23.59767, 2, ''],
  [282, 121.19341, 23.56040, 2, ''],
  [372, 121.06834, 23.55595, 2, ''],
  [446, 121.05106, 24.31155, 2, ''],
  [310, 120.95765, 24.13732, 2, ''],
  [308, 121.48446, 24.12473, 2, ''],
  [451, 120.88134, 23.14499, 2, '']
];

var CLSNAME = {1:'Broadleaf', 2:'Conifer', 3:'Bamboo'};
var CLSCOL  = {1:'#009E73', 2:'#0072B2', 3:'#E69F00'};

// low-cloud 2025 Sentinel-2 true-colour composite
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate('2025-01-01','2025-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
  .median();
var vis = {bands:['B4','B3','B2'], min:0, max:2200};
Map.addLayer(s2, vis, 'S2 2025 median', true);

var idx = 0;
var panel = ui.Panel({style:{width:'320px'}});
var label = ui.Label('', {fontWeight:'bold', fontSize:'16px'});
var sub   = ui.Label('');
var marker;

function show(i){
  var p = PTS[i];
  var pt = ee.Geometry.Point([p[1], p[2]]);
  if (marker) Map.layers().remove(marker);
  marker = ui.Map.Layer(pt.buffer(105), {color: CLSCOL[p[3]]}, 'point '+p[0]);
  Map.layers().add(marker);
  Map.centerObject(pt, 17);
  label.setValue((i+1)+' / '+PTS.length+'  ptid '+p[0]+'  label: '+CLSNAME[p[3]]);
  sub.setValue(p[4]==='Y' ? 'flagged stratum (largest 2023-2025 signature departure)' : 'random stratum');
}
var prev = ui.Button('< previous', function(){ idx=Math.max(0,idx-1); show(idx); });
var next = ui.Button('next >', function(){ idx=Math.min(PTS.length-1,idx+1); show(idx); });
panel.add(ui.Label('Recent-image check', {fontWeight:'bold', fontSize:'18px'}));
panel.add(label); panel.add(sub);
panel.add(ui.Panel([prev,next], ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Label('The circle marks 210 m around the point; the manuscript interpretation uses the central 3 x 3 Sentinel-2 pixels.'));
panel.add(ui.Label('Switch to the Satellite basemap (top right) for the high-resolution view.'));
ui.root.insert(0, panel);
show(0);
