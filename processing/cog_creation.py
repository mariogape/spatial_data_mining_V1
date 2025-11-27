#!/usr/bin/env python3
"""
Batch-create Cloud Optimized GeoTIFFs (COGs) from one or more input folders.

Run this script from the OSGeo4W Shell (or any environment where GDAL CLI tools
are on PATH):

    python processing/cog_creation.py

Update the USER SETTINGS section below before running.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

###############################################################################
# USER SETTINGS - EDIT THESE
###############################################################################

# Every folder listed here will be processed. Output goes to
# <input_folder>/<OUTPUT_SUBDIR>.
INPUT_FOLDERS = [
    # r"D:\OpenPas Spatial Data\Land Cover\2023",
    r"D:\OpenPas Spatial Data\Precipitation annual sum",
    # r"D:\OpenPas Spatial Data\Snow\masked_scd",
    # r"D:\OpenPas Spatial Data\Temperature annual mean"
    # Add more input folders as needed, e.g.:
    # r"D:\OpenPas Spatial Data\Forest Canopy Height\GEE_canopy_height\merged",
]

# Name of the child folder where COGs will be written (inside each input folder)
OUTPUT_SUBDIR = "cog"

# Allowed: float32, int16, int8  (or leave empty "")
USER_DTYPE = ""

# Example: EPSG:4326, EPSG:25830  (or leave empty "")
TARGET_CRS = "EPSG:3035"

# Only used if TARGET_CRS is set
RESAMPLING = "near"

OUTPUT_SUFFIX = "_cog"

# Overwrite existing outputs when rerunning. If False, existing files are skipped.
OVERWRITE = True

# Command names (or full paths) for GDAL utilities
GDAL_TRANSLATE = "gdal_translate"
GDAL_WARP = "gdalwarp"

###############################################################################
# INTERNALS
###############################################################################


def map_dtype(user_dtype: str) -> Optional[str]:
    if not user_dtype:
        return None
    mapping = {
        "float32": "Float32",
        "int16": "Int16",
        "int8": "Byte",
        "byte": "Byte",
    }
    return mapping.get(user_dtype.lower())


def iter_rasters(folder: Path, exclude: Optional[Path] = None) -> List[Path]:
    exts = {".tif", ".tiff"}
    rasters: List[Path] = []
    for p in folder.rglob("*"):
        if p.suffix.lower() not in exts:
            continue
        if exclude and exclude in p.parents:
            continue
        rasters.append(p)
    return sorted(rasters)


def build_base_args(target_crs: str, resampling: str, overwrite: bool) -> Tuple[str, List[str]]:
    if target_crs:
        tool = GDAL_WARP
        args = [
            "-of",
            "COG",
            "-t_srs",
            target_crs,
            "-r",
            resampling,
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "BIGTIFF=IF_SAFER",
        ]
    else:
        tool = GDAL_TRANSLATE
        args = [
            "-of",
            "COG",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "BIGTIFF=IF_SAFER",
        ]
    if overwrite:
        args.insert(0, "-overwrite")
    return tool, args


def run_gdal_command(tool: str, base_args: List[str], infile: Path, outfile: Path) -> int:
    cmd = [tool, *base_args, str(infile), str(outfile)]
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def process_folder(
    tool: str,
    base_args: List[str],
    input_dir: Path,
    output_dir: Path,
    output_suffix: str,
    overwrite: bool,
) -> int:
    if not input_dir.is_dir():
        print(f"[ERROR] Input folder not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    rasters = iter_rasters(input_dir, exclude=output_dir)
    if not rasters:
        print(f"[INFO] No .tif/.tiff files found in {input_dir}")
        return 0

    print(f"--- Processing folder ---\n[IN ] {input_dir}\n[OUT] {output_dir}\n")
    for src in rasters:
        out_name = f"{src.stem}{output_suffix}.tif"
        dst = output_dir / out_name
        if dst.exists() and not overwrite:
            print(f"[SKIP] {dst.name} already exists (set OVERWRITE=True to rebuild).")
            continue
        print(f"[INFO] {src.name} -> {dst.name}")
        rc = run_gdal_command(tool, base_args, src, dst)
        if rc != 0:
            print(f"[ERROR] Failed to create {dst} (exit code {rc})")
            return rc
    return 0


def main() -> int:
    gdal_dtype = map_dtype(USER_DTYPE)
    if USER_DTYPE and gdal_dtype is None:
        print("[ERROR] Unsupported data type:", USER_DTYPE)
        print("        Use one of: float32, int16, int8 (or leave empty).")
        return 1

    gdal_tool, gdal_args = build_base_args(TARGET_CRS, RESAMPLING, OVERWRITE)
    if gdal_dtype:
        gdal_args.extend(["-ot", gdal_dtype])

    if shutil.which(gdal_tool) is None:
        print(f"[ERROR] {gdal_tool} not found on PATH. Run inside OSGeo4W Shell.")
        return 1

    if not INPUT_FOLDERS:
        print("[ERROR] No folders configured in INPUT_FOLDERS.")
        return 1

    for raw_input in INPUT_FOLDERS:
        in_dir = Path(raw_input)
        out_dir = in_dir / OUTPUT_SUBDIR
        rc = process_folder(
            gdal_tool,
            gdal_args,
            in_dir,
            out_dir,
            OUTPUT_SUFFIX,
            OVERWRITE,
        )
        if rc != 0:
            return rc

    print("\n[DONE] Batch COG creation finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
