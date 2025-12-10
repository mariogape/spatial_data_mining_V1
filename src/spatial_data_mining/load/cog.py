from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy


def write_cog(src_path: Path, dst_path: Path) -> None:
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cog_profile = dict(
        driver="COG",
        compress="deflate",
        blocksize=512,
        overview_resampling=Resampling.average,
        bigtiff="IF_SAFER",
    )
    with rasterio.Env():
        rio_copy(str(src_path), str(dst_path), **cog_profile)
