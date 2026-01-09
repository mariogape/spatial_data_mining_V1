import logging
from pathlib import Path
from typing import Any

import numpy as np
import rioxarray
import rasterio
import xarray as xr
from rasterio.io import MemoryFile
from rasterio.mask import mask
from pyproj import Transformer
from rasterio.enums import Resampling, ColorInterp
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


def _scale_reflectance(values: np.ndarray, src_path: Path, valid_mask: np.ndarray) -> np.ndarray:
    values = values.astype("float32", copy=False)
    scales = None
    offsets = None
    try:
        with rasterio.open(src_path) as src:
            scales = src.scales
            offsets = src.offsets
    except Exception:
        scales = None
        offsets = None

    applied = False
    if scales or offsets:
        for idx in range(min(values.shape[0], len(scales or []))):
            scale = scales[idx] if scales and scales[idx] is not None else 1.0
            offset = offsets[idx] if offsets and offsets[idx] is not None else 0.0
            if scale != 1.0 or offset != 0.0:
                values[idx] = values[idx] * scale + offset
                applied = True

    if not applied:
        valid_values = values[:, valid_mask] if valid_mask.any() else values.reshape(-1)
        if valid_values.size and np.nanmax(valid_values) > 1.5:
            values = values / 10000.0

    return values


def _sentinel_hub_true_color(b04: np.ndarray, b03: np.ndarray, b02: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_r = 3.0
    mid_r = 0.13
    sat = 1.3
    gamma = 2.3
    ray_r = 0.013
    ray_g = 0.024
    ray_b = 0.041

    g_off = 0.01
    g_off_pow = g_off**gamma
    g_off_range = (1 + g_off) ** gamma - g_off_pow

    def adj(a, tx, ty, max_c):
        ar = np.clip(a / max_c, 0.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return ar * (ar * (tx / max_c + ty - 1) - ty) / (ar * (2 * tx / max_c - 1) - tx / max_c)

    def adj_gamma(b):
        return (np.power((b + g_off), gamma) - g_off_pow) / g_off_range

    def s_adj(a):
        return adj_gamma(adj(a, mid_r, 1, max_r))

    def sat_enh(r, g, b):
        avg_s = (r + g + b) / 3.0 * (1 - sat)
        r_out = np.clip(avg_s + r * sat, 0.0, 1.0)
        g_out = np.clip(avg_s + g * sat, 0.0, 1.0)
        b_out = np.clip(avg_s + b * sat, 0.0, 1.0)
        return r_out, g_out, b_out

    def s_rgb(c):
        return np.where(
            c <= 0.0031308,
            12.92 * c,
            1.055 * np.power(c, 0.41666666666) - 0.055,
        )

    r_lin, g_lin, b_lin = sat_enh(
        s_adj(b04 - ray_r),
        s_adj(b03 - ray_g),
        s_adj(b02 - ray_b),
    )
    return s_rgb(r_lin), s_rgb(g_lin), s_rgb(b_lin)


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


def process_rgb_true_color(
    src_path: Path,
    target_crs: str,
    resolution_m: float | None,
    aoi_geom_target: Any,
) -> Path:
    """
    Apply Sentinel Hub L1C true color optimized visualization and output RGBA.
    """
    src_path = Path(src_path)
    processed_path = src_path.with_name(f"{src_path.stem}_truecolor.tif")

    data = _read_raster_clipped(src_path, target_crs, aoi_geom_target)
    data = _normalize_spatial_dims(data)
    if "band" in data.dims:
        data = data.transpose("band", "y", "x")

    values = data.values
    if np.ma.isMaskedArray(values):
        mask = np.ma.getmaskarray(values)
        values = values.filled(np.nan)
    else:
        mask = np.zeros_like(values, dtype=bool)

    nodata = data.rio.nodata
    if nodata is not None and nodata != 0:
        mask |= values == nodata
    mask |= ~np.isfinite(values)
    valid_mask = ~np.any(mask, axis=0)
    values = np.where(mask, 0.0, values)

    if values.shape[0] < 3:
        raise ValueError("RGB true color requires at least three bands (B04/B03/B02).")

    values = _scale_reflectance(values, src_path, valid_mask)

    r, g, b = _sentinel_hub_true_color(values[0], values[1], values[2])
    r = np.where(valid_mask, r, 0.0)
    g = np.where(valid_mask, g, 0.0)
    b = np.where(valid_mask, b, 0.0)
    alpha = np.where(valid_mask, 1.0, 0.0)

    rgba = np.stack([r, g, b, alpha], axis=0).astype("float32")
    rgba_da = xr.DataArray(
        rgba,
        dims=("band", "y", "x"),
        coords={"band": [1, 2, 3, 4], "y": data["y"], "x": data["x"]},
    )
    rgba_da = rgba_da.rio.write_crs(data.rio.crs, inplace=False)
    rgba_da = rgba_da.rio.write_transform(data.rio.transform(), inplace=False)

    rgba_da = _clip_to_source_aoi(rgba_da, target_crs, aoi_geom_target)
    rgba_da = _reproject_raster(rgba_da, target_crs, resolution_m, Resampling.bilinear)
    rgba_da = _clip_to_aoi(rgba_da, target_crs, aoi_geom_target)

    rgba_da = rgba_da.clip(min=0.0, max=1.0)
    rgba_da = (rgba_da * 255.0).round().astype("uint8")
    try:
        rgba_da = rgba_da.rio.write_nodata(None, inplace=False)
    except Exception:
        rgba_da = rgba_da.copy()
        rgba_da.attrs.pop("_FillValue", None)

    rgba_da.rio.to_raster(processed_path, compress="deflate")

    try:
        with rasterio.open(processed_path, "r+") as dst:
            dst.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )
            try:
                dst.descriptions = ("red", "green", "blue", "alpha")
            except Exception:
                pass
    except Exception:
        pass

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
