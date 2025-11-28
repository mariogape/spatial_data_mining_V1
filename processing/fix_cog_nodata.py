#!/usr/bin/env python3
"""
Rewrite Cloud Optimized GeoTIFFs so that 0 values and existing nodata values
become a new nodata value, while keeping the outputs as COGs.

Run from an environment where GDAL's Python bindings are available
(e.g., OSGeo4W Shell on Windows):

    python processing/fix_cog_nodata.py

Update the USER SETTINGS section below before running.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from osgeo import gdal, gdal_array

###############################################################################
# USER SETTINGS - EDIT THESE
###############################################################################

# Folder containing the source COGs
INPUT_FOLDER = Path(
    r"D:\OpenPas Spatial Data\Remote sensing indexes\GEE_OpenLandMap-20251128T071600Z-1-001\GEE_OpenLandMap\cog"
)

# Folder where fixed COGs will be written (files keep the same names)
OUTPUT_FOLDER = Path(
    r"D:\OpenPas Spatial Data\Remote sensing indexes\GEE_OpenLandMap-20251128T071600Z-1-001\GEE_OpenLandMap\cog_neg9999"
)

# New nodata value to apply and write
NEW_NODATA = -9999

# Overwrite existing files in OUTPUT_FOLDER when rerunning
OVERWRITE = True

# Search subfolders inside INPUT_FOLDER for rasters
RECURSIVE = False

# Force output data type (e.g., "Float32", "Int16"). Leave empty for automatic.
TARGET_DTYPE = ""

# Creation options passed to the COG driver
COG_CREATION_OPTIONS: Sequence[str] = (
    "COMPRESS=DEFLATE",
    "BIGTIFF=IF_SAFER",
    "RESAMPLING=NEAREST",
    "NUM_THREADS=ALL_CPUS",
    "OVERVIEWS=IGNORE_EXISTING",
    "BLOCKSIZE=512",
)

# Tolerance used when comparing to zero for floating-point rasters
ZERO_TOLERANCE = 1e-8

###############################################################################
# INTERNALS
###############################################################################


def list_rasters(folder: Path, recursive: bool, exclude: Optional[Path] = None) -> List[Path]:
    exts = {".tif", ".tiff"}
    if recursive:
        candidates: Iterable[Path] = folder.rglob("*")
    else:
        candidates = folder.iterdir()
    rasters: List[Path] = []
    for p in candidates:
        if p.is_dir():
            continue
        if p.suffix.lower() not in exts:
            continue
        if exclude and exclude in p.resolve().parents:
            continue
        rasters.append(p)
    return sorted(rasters)


def parse_dtype(name: str) -> Optional[int]:
    if not name:
        return None
    dtype = gdal.GetDataTypeByName(name)
    return None if dtype == gdal.GDT_Unknown else dtype


def nodata_fits_dtype(dtype: int, nodata_value: float) -> bool:
    """Check whether nodata_value can be represented in dtype without overflow."""
    if gdal.GetDataTypeName(dtype).startswith("Float"):
        return True
    np_dtype = np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(dtype))
    if not np.issubdtype(np_dtype, np.integer):
        return False
    info = np.iinfo(np_dtype)
    return info.min <= nodata_value <= info.max


def choose_dtype(src_dtype: int, new_nodata: float, forced_dtype: Optional[int]) -> int:
    if forced_dtype is not None:
        return forced_dtype
    if nodata_fits_dtype(src_dtype, new_nodata):
        return src_dtype
    # Promote unsigned integer sources so the negative nodata can be stored safely.
    if src_dtype == gdal.GDT_Byte:
        return gdal.GDT_Int16
    if src_dtype in (gdal.GDT_UInt16,):
        return gdal.GDT_Int32
    if src_dtype in (gdal.GDT_UInt32, gdal.GDT_UInt64):
        return gdal.GDT_Float32
    return gdal.GDT_Float32


def create_empty_copy(src_ds: gdal.Dataset, dtype: int, band_count: int) -> gdal.Dataset:
    mem = gdal.GetDriverByName("MEM").Create(
        "", src_ds.RasterXSize, src_ds.RasterYSize, band_count, dtype
    )
    mem.SetGeoTransform(src_ds.GetGeoTransform())
    mem.SetProjection(src_ds.GetProjection())
    mem.SetMetadata(src_ds.GetMetadata())
    if src_ds.GetGCPs():
        mem.SetGCPs(src_ds.GetGCPs(), src_ds.GetGCPProjection())
    return mem


def process_band(
    src_band: gdal.Band,
    dst_band: gdal.Band,
    new_nodata: float,
    target_np_dtype: np.dtype,
) -> None:
    src_nodata = src_band.GetNoDataValue()
    block_x, block_y = src_band.GetBlockSize()
    block_x = block_x or src_band.XSize
    block_y = block_y or src_band.YSize

    ysize, xsize = src_band.YSize, src_band.XSize
    for y_off in range(0, ysize, block_y):
        rows = min(block_y, ysize - y_off)
        for x_off in range(0, xsize, block_x):
            cols = min(block_x, xsize - x_off)
            data = src_band.ReadAsArray(x_off, y_off, cols, rows)
            if data is None:
                raise RuntimeError("Failed to read raster block.")
            mask_zero = np.isclose(data, 0.0, atol=ZERO_TOLERANCE)
            mask_nodata = np.zeros_like(mask_zero, dtype=bool)
            if src_nodata is not None:
                if np.isnan(src_nodata):
                    mask_nodata = np.isnan(data)
                else:
                    mask_nodata = np.isclose(data, src_nodata, atol=ZERO_TOLERANCE)
            mask_nan = np.isnan(data)
            mask = mask_zero | mask_nodata | mask_nan
            if mask.any():
                data = data.copy()
                data[mask] = new_nodata
            if data.dtype != target_np_dtype:
                data = data.astype(target_np_dtype, copy=False)
            dst_band.WriteArray(data, x_off, y_off)

    dst_band.SetNoDataValue(new_nodata)
    dst_band.SetDescription(src_band.GetDescription())
    dst_band.SetMetadata(src_band.GetMetadata())
    if src_band.GetScale() is not None:
        dst_band.SetScale(src_band.GetScale())
    if src_band.GetOffset() is not None:
        dst_band.SetOffset(src_band.GetOffset())
    color_table = src_band.GetColorTable()
    if color_table:
        dst_band.SetColorTable(color_table)


def convert_file(
    src_path: Path,
    dst_path: Path,
    new_nodata: float,
    creation_options: Sequence[str],
    forced_dtype: Optional[int],
) -> None:
    src_ds = gdal.Open(str(src_path))
    if src_ds is None:
        raise RuntimeError(f"Could not open source raster: {src_path}")

    band_count = src_ds.RasterCount
    if band_count < 1:
        raise RuntimeError(f"No raster bands found in {src_path}")

    dtype = choose_dtype(src_ds.GetRasterBand(1).DataType, new_nodata, forced_dtype)
    target_np_dtype = np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(dtype))

    temp_ds = create_empty_copy(src_ds, dtype, band_count)
    for idx in range(1, band_count + 1):
        process_band(src_ds.GetRasterBand(idx), temp_ds.GetRasterBand(idx), new_nodata, target_np_dtype)

    translate_opts = gdal.TranslateOptions(
        format="COG",
        creationOptions=list(creation_options),
        outputType=dtype,
        noData=new_nodata,
    )
    result = gdal.Translate(destName=str(dst_path), srcDS=temp_ds, options=translate_opts)
    if result is None:
        raise RuntimeError(f"Failed to write COG: {dst_path}")


def main() -> int:
    gdal.UseExceptions()

    input_dir = INPUT_FOLDER
    output_dir = OUTPUT_FOLDER

    if input_dir.resolve() == output_dir.resolve():
        print("[ERROR] INPUT_FOLDER and OUTPUT_FOLDER must be different.")
        return 1
    if not input_dir.is_dir():
        print(f"[ERROR] Input folder not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    exclude = output_dir.resolve() if output_dir in input_dir.resolve().parents else None
    rasters = list_rasters(input_dir, RECURSIVE, exclude=exclude)
    if not rasters:
        print(f"[INFO] No .tif/.tiff files found in {input_dir}")
        return 0

    forced_dtype = parse_dtype(TARGET_DTYPE)
    if TARGET_DTYPE and forced_dtype is None:
        print(f"[ERROR] Unsupported TARGET_DTYPE: {TARGET_DTYPE}")
        return 1

    print(f"--- Fixing nodata ---\n[IN ] {input_dir}\n[OUT] {output_dir}\n")
    for src in rasters:
        dst = output_dir / src.name
        if dst.exists() and not OVERWRITE:
            print(f"[SKIP] {dst.name} already exists (set OVERWRITE=True to rebuild).")
            continue
        print(f"[INFO] {src.name} -> {dst}")
        try:
            convert_file(src, dst, NEW_NODATA, COG_CREATION_OPTIONS, forced_dtype)
        except Exception as exc:
            print(f"[ERROR] {src.name}: {exc}")
            return 1

    print("\n[DONE] All rasters processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
