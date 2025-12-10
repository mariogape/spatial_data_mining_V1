from pathlib import Path
from typing import Any

import rioxarray
from shapely.geometry import mapping


def process_raster_to_target(
    src_path: Path,
    target_crs: str,
    resolution_m: float,
    aoi_geom_target: Any,
) -> Path:
    """
    Reproject, resample, and clip to the target AOI/CRS.
    Returns a path to a temporary GeoTIFF in the target CRS.
    """
    src_path = Path(src_path)
    processed_path = src_path.with_name(f"{src_path.stem}_processed.tif")

    data = rioxarray.open_rasterio(src_path, masked=True)
    data = data.squeeze()
    data = data.rio.reproject(target_crs, resolution=resolution_m)
    data = data.rio.clip([mapping(aoi_geom_target)], target_crs, drop=True)
    data.rio.to_raster(processed_path, compress="deflate")

    return processed_path
