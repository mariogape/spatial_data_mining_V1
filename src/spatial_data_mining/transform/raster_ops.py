import logging
from pathlib import Path
from typing import Any

import numpy as np
import rioxarray
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform
from shapely.geometry import mapping, box
from shapely.ops import transform as shp_transform

logger = logging.getLogger(__name__)

# Standard nodata value used across all outputs.
NODATA_VALUE = -9999


def _standardize_nodata(data, nodata_value: float | int = NODATA_VALUE):
    """
    Ensure nodata is consistent across all outputs:
    - Set raster nodata metadata to `nodata_value`
    - Replace masked/NaN pixels with `nodata_value`
    """
    try:
        data = data.rio.write_nodata(nodata_value, inplace=False)
    except Exception as exc:
        logger.warning("Could not set nodata=%s (%s)", nodata_value, exc)
    try:
        data = data.fillna(nodata_value)
    except Exception as exc:
        logger.warning("Could not fill nodata pixels with %s (%s)", nodata_value, exc)
    return data


def _normalize_spatial_dims(data):
    """
    Ensure rioxarray data uses x/y spatial dims even if named differently.
    """
    # Only squeeze non-spatial singleton dims (e.g., band), keep spatial dims even if size 1.
    if "band" in data.dims and data.sizes.get("band", 0) == 1:
        data = data.squeeze("band", drop=True)
    if "variable" in data.dims and data.sizes.get("variable", 0) == 1:
        data = data.squeeze("variable", drop=True)

    if "x" not in data.dims or "y" not in data.dims:
        spatial_dims = [d for d in data.dims if d not in ("band", "variable")]
        if len(spatial_dims) >= 2:
            y_dim, x_dim = spatial_dims[-2:]
        elif len(data.dims) >= 2:
            y_dim, x_dim = data.dims[-2:]
        else:
            raise ValueError("Could not infer spatial dimensions for raster.")

        rename_map = {}
        if x_dim != "x":
            rename_map[x_dim] = "x"
        if y_dim != "y":
            rename_map[y_dim] = "y"
        if rename_map:
            data = data.rename(rename_map)
    return data.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)


def _reproject_raster(data, target_crs: str, resolution_m: float | None, resampling: Resampling):
    reproject_kwargs = {
        "dst_crs": target_crs,
        "resampling": resampling,
    }
    if resolution_m is not None:
        reproject_kwargs["resolution"] = resolution_m
    else:
        # Preserve native pixel grid: compute destination transform/shape matching source pixel counts.
        try:
            with data.rio.env():
                transform, width, height = calculate_default_transform(
                    data.rio.crs,
                    target_crs,
                    data.rio.width,
                    data.rio.height,
                    *data.rio.bounds(),
                    dst_width=data.rio.width,
                    dst_height=data.rio.height,
                )
            reproject_kwargs["transform"] = transform
            reproject_kwargs["shape"] = (height, width)
        except Exception as exc:
            logger.warning("Falling back to default transform when preserving native grid: %s", exc)

    return data.rio.reproject(**reproject_kwargs)


def _clip_to_aoi(data, target_crs: str, aoi_geom_target: Any):
    aoi_geom = aoi_geom_target
    try:
        return data.rio.clip([mapping(aoi_geom)], target_crs, drop=True)
    except Exception as exc:
        logger.warning("Clip failed (%s); retrying with all_touched=True", exc)
        try:
            return data.rio.clip([mapping(aoi_geom)], target_crs, drop=True, all_touched=True)
        except Exception as exc2:
            raster_box = box(*data.rio.bounds())
            if not raster_box.intersects(aoi_geom):
                raise
            logger.warning("Clip failed again (%s); writing un-clipped raster as fallback", exc2)
            return data


def _clip_to_source_aoi(data, target_crs: str, aoi_geom_target: Any):
    """
    Clip in the source CRS first to avoid reprojecting the full raster for large inputs.
    Falls back silently if any step fails.
    """
    try:
        transformer = Transformer.from_crs(target_crs, data.rio.crs, always_xy=True)
        aoi_geom_src = shp_transform(transformer.transform, aoi_geom_target)
        return _clip_to_aoi(data, data.rio.crs, aoi_geom_src)
    except Exception as exc:
        logger.warning("Pre-clip in source CRS failed; continuing without it (%s)", exc)
        return data


def _read_raster_clipped(src_path: Path, target_crs: str, aoi_geom_target: Any):
    """
    Read only the AOI footprint from the source raster to avoid loading huge scenes.
    Falls back to a full read if anything goes wrong.
    """
    try:
        with rasterio.open(src_path) as src:
            src_crs = src.crs
            if src_crs is None:
                raise ValueError("Source raster has no CRS")
            transformer = Transformer.from_crs(target_crs, src_crs, always_xy=True)
            aoi_geom_src = shp_transform(transformer.transform, aoi_geom_target)
            if not box(*src.bounds).intersects(aoi_geom_src):
                raise ValueError("AOI does not intersect source raster")

            data, transform = mask(
                src,
                [mapping(aoi_geom_src)],
                crop=True,
                filled=True,
                nodata=src.nodata,
            )
            profile = src.profile
            profile.update(
                height=data.shape[1],
                width=data.shape[2],
                transform=transform,
                driver="GTiff",
            )

        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(data)
            return rioxarray.open_rasterio(memfile, masked=True).load()
    except Exception as exc:
        logger.warning("Read+crop in source CRS failed for %s; falling back to full read (%s)", src_path, exc)
        return rioxarray.open_rasterio(src_path, masked=True)


def process_raster_to_target(
    src_path: Path,
    target_crs: str,
    resolution_m: float | None,
    aoi_geom_target: Any,
) -> Path:
    """
    Reproject, resample, and clip to the target AOI/CRS.
    Returns a path to a temporary GeoTIFF in the target CRS.
    """
    src_path = Path(src_path)
    processed_path = src_path.with_name(f"{src_path.stem}_processed.tif")

    data = _read_raster_clipped(src_path, target_crs, aoi_geom_target)
    data = _normalize_spatial_dims(data)
    data = _standardize_nodata(data)
    data = _clip_to_source_aoi(data, target_crs, aoi_geom_target)
    data = _reproject_raster(data, target_crs, resolution_m, Resampling.bilinear)
    data = _clip_to_aoi(data, target_crs, aoi_geom_target)
    data = _standardize_nodata(data)

    data.rio.to_raster(processed_path, compress="deflate")

    return processed_path


def process_fvc_to_target(
    src_path: Path,
    target_crs: str,
    resolution_m: float | None,
    aoi_geom_target: Any,
) -> Path:
    """
    Compute FVC from an NDVI composite, then reproject/resample/clip to the target AOI/CRS.
    FVC = (NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil), where NDVI_soil/veg are 5th/95th
    percentiles of NDVI within the AOI.
    """
    src_path = Path(src_path)
    processed_path = src_path.with_name(f"{src_path.stem}_fvc_processed.tif")

    data = _read_raster_clipped(src_path, target_crs, aoi_geom_target)
    data = _normalize_spatial_dims(data)
    data = _standardize_nodata(data)
    data = _clip_to_source_aoi(data, target_crs, aoi_geom_target)

    nodata = data.rio.nodata
    if nodata is None:
        nodata = NODATA_VALUE

    values = np.ma.array(data.values, dtype="float64")
    values = np.ma.masked_invalid(values)
    values = np.ma.masked_where(values == nodata, values)
    valid = values.compressed()
    if valid.size == 0:
        raise ValueError("FVC requires valid NDVI pixels in the AOI to compute percentiles.")

    ndvi_soil, ndvi_veg = np.percentile(valid, [5, 95])
    denom = ndvi_veg - ndvi_soil
    if denom == 0:
        raise ValueError("FVC cannot be computed because NDVI_veg equals NDVI_soil.")

    fvc = (data - ndvi_soil) / denom
    fvc = fvc.clip(min=0.0, max=1.0)
    fvc = fvc.where(data != nodata, other=NODATA_VALUE)
    try:
        fvc = fvc.astype("float32")
    except Exception:
        pass

    fvc = _reproject_raster(fvc, target_crs, resolution_m, Resampling.bilinear)
    fvc = _clip_to_aoi(fvc, target_crs, aoi_geom_target)
    fvc = _standardize_nodata(fvc)

    fvc.rio.to_raster(processed_path, compress="deflate")
    return processed_path


def process_clcplus_to_target(
    src_path: Path,
    target_crs: str,
    resolution_m: float | None,
    aoi_geom_target: Any,
) -> Path:
    """
    Reproject CLCplus rasters with nearest-neighbor resampling, clip to AOI,
    and recode 0 / nodata pixels to -9999 before writing a GeoTIFF.
    """
    src_path = Path(src_path)
    processed_path = src_path.with_name(f"{src_path.stem}_processed.tif")

    data = _read_raster_clipped(src_path, target_crs, aoi_geom_target)
    data = _normalize_spatial_dims(data)
    data = _standardize_nodata(data)
    data = _clip_to_source_aoi(data, target_crs, aoi_geom_target)
    data = _reproject_raster(data, target_crs, resolution_m, Resampling.nearest)
    data = _clip_to_aoi(data, target_crs, aoi_geom_target)

    # Recode nodata and zero values to NODATA_VALUE and preserve integer semantics.
    data = data.fillna(NODATA_VALUE)
    data = data.where(data != 0, other=NODATA_VALUE)
    try:
        data = data.astype("int32")
        data.rio.write_nodata(NODATA_VALUE, inplace=True)
    except Exception as exc:  # best-effort typing/nodata; continue even if write_nodata fails
        logger.warning("Could not enforce nodata/%s typing on CLCplus raster: %s", NODATA_VALUE, exc)

    data.rio.to_raster(processed_path, compress="deflate")

    return processed_path
