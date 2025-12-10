from pathlib import Path
from typing import Tuple

import geopandas as gpd
from shapely.geometry import mapping


def load_aoi(path: str) -> gpd.GeoDataFrame:
    aoi_path = Path(path)
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")
    gdf = gpd.read_file(aoi_path)
    if gdf.empty:
        raise ValueError(f"AOI file is empty: {aoi_path}")
    if gdf.crs is None:
        raise ValueError(f"AOI CRS not defined: {aoi_path}")
    return gdf


def get_aoi_geometries(gdf: gpd.GeoDataFrame, target_crs: str) -> Tuple[dict, object]:
    """Return AOI geometries in WGS84 (for GEE) and target CRS (for raster ops)."""
    aoi_wgs84 = gdf.to_crs("EPSG:4326")
    aoi_target = gdf.to_crs(target_crs)
    geom_wgs84 = mapping(aoi_wgs84.unary_union)
    geom_target = aoi_target.unary_union
    return geom_wgs84, geom_target
