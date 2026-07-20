"""Dissolve the NLSC village-boundary shapefile into a single Taiwan land outline.
Download VILLAGE_NLSC_*.shp (TWD97/EPSG:3826) from NLSC, then:
  python src/boundary_dissolve.py /path/to/VILLAGE_NLSC.shp out/tw_boundary.gpkg"""
import sys, geopandas as gpd
from shapely.geometry import MultiPolygon
shp, out = sys.argv[1], sys.argv[2]
g = gpd.read_file(shp).to_crs(3826)
uni = g.geometry.union_all()
polys = [p for p in (uni.geoms if uni.geom_type=="MultiPolygon" else [uni]) if p.area > 1e6]
gpd.GeoSeries([MultiPolygon(polys)], crs=3826).simplify(120).to_file(out, driver="GPKG")
print("wrote", out, "polygons:", len(polys))
