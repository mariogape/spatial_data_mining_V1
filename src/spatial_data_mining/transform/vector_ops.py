import logging
from pathlib import Path
from typing import Any

import geopandas as gpd

logger = logging.getLogger(__name__)

_VECTOR_FORMATS = {
    "geojson": {"ext": "geojson", "driver": "GeoJSON"},
    "json": {"ext": "geojson", "driver": "GeoJSON"},
    "gpkg": {"ext": "gpkg", "driver": "GPKG"},
    "geopackage": {"ext": "gpkg", "driver": "GPKG"},
}


def normalize_vector_format(fmt: str | None) -> str:
    if not fmt:
        return "geojson"
    key = str(fmt).strip().lower()
    if key in _VECTOR_FORMATS:
        return _VECTOR_FORMATS[key]["ext"]
    raise ValueError(f"Unsupported vector format: {fmt!r} (use 'geojson' or 'gpkg').")


def _driver_for_format(fmt: str) -> str:
    key = str(fmt).strip().lower()
    if key in _VECTOR_FORMATS:
        return _VECTOR_FORMATS[key]["driver"]
    if key in {"geojson", "gpkg"}:
        return _VECTOR_FORMATS[key]["driver"]
    raise ValueError(f"Unsupported vector format: {fmt!r} (use 'geojson' or 'gpkg').")


def process_vector_to_target(
    src_path: str | Path,
    target_crs: str,
    resolution_m: float | None,
    aoi_geom_target: Any,
    output_format: str | None = None,
) -> Path:
    """
    Reproject a vector dataset to the target CRS and optionally clip to the AOI.
    Returns the processed dataset path in the requested output format.
    """
    src_path = Path(src_path)
    fmt = normalize_vector_format(output_format)
    driver = _driver_for_format(fmt)

    gdf = gpd.read_file(src_path)
    if gdf.empty:
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs or "EPSG:4326")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    try:
        gdf = gdf.to_crs(target_crs)
    except Exception as exc:
        logger.warning("Vector reprojection failed; writing in source CRS (%s).", exc)

    if aoi_geom_target is not None and not gdf.empty:
        try:
            gdf = gdf[gdf.geometry.intersects(aoi_geom_target)]
        except Exception as exc:
            logger.warning("Vector AOI clip failed; continuing without clip (%s).", exc)

    out_path = src_path.with_name(f"{src_path.stem}_projected.{fmt}")
    gdf.to_file(out_path, driver=driver)
    return out_path
