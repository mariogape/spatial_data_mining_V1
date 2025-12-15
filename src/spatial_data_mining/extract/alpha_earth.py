import logging
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import ee
import requests
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.io import MemoryFile
from shapely.geometry import box, shape, mapping
from spatial_data_mining.variables.metadata import get_variable_metadata
from affine import Affine
import numpy as np

# Starting tile size (degrees) for AlphaEarth tiling and the smallest fallback size.
TILE_DEG = 0.05
MIN_TILE_DEG = 0.005


class AlphaEarthExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.collection_id = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

    @staticmethod
    def _notify(cb: Optional[Callable[[str], None]], message: str) -> None:
        if cb:
            cb(message)

    def _initialize(self) -> None:
        try:
            ee.Initialize()
        except Exception:
            self.logger.info("Authenticating with Google Earth Engine...")
            ee.Authenticate()
            ee.Initialize()

    def _download_image(
        self,
        image: ee.Image,
        region_geojson: dict,
        scale: float,
        name: str,
        tmp_dir: Path,
    ) -> Path:
        url = image.getDownloadURL(
            {
                "region": region_geojson,
                "scale": scale,
                "fileFormat": "GeoTIFF",
                "filePerBand": False,
            }
        )
        resp = requests.get(url, timeout=600)
        if resp.status_code >= 400:
            # Include response text to aid detection of size-limit errors.
            raise RuntimeError(f"Download failed ({resp.status_code}): {resp.text}")

        tif_path = tmp_dir / f"{name}.tif"
        content = resp.content
        bio = BytesIO(content)
        if zipfile.is_zipfile(bio):
            bio.seek(0)
            with zipfile.ZipFile(bio) as zf:
                tif_members = [m for m in zf.namelist() if m.lower().endswith((".tif", ".tiff"))]
                if not tif_members:
                    raise ValueError("Downloaded archive does not contain a .tif")
                with zf.open(tif_members[0]) as src, tif_path.open("wb") as dst:
                    dst.write(src.read())
        else:
            if content[:4] not in (b"II*\x00", b"MM\x00*"):
                raise ValueError(
                    f"Download did not return a valid TIFF. Content-type: "
                    f"{resp.headers.get('content-type')} ; first bytes: {content[:20]!r}"
                )
            tif_path.write_bytes(content)
        return tif_path

    def _tile_aoi(self, aoi_geojson: dict, tile_deg: float) -> List[dict]:
        geom = shape(aoi_geojson)
        minx, miny, maxx, maxy = geom.bounds
        tiles = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                tile = box(x, y, min(x + tile_deg, maxx), min(y + tile_deg, maxy))
                inter = geom.intersection(tile)
                if not inter.is_empty:
                    tiles.append(mapping(inter))
                y += tile_deg
            x += tile_deg
        return tiles

    def _merge_tiles(self, tile_paths: List[Path], name: str, tmp_dir: Path) -> Path:
        out_path = tmp_dir / f"{name}_merged.tif"
        merge_dtype = "float32"  # force float32 to reduce memory footprint during merge

        # Prep tiles (flip south-up) while staying on the caller-provided disk.
        prepared_paths: List[Path] = []
        for p in tile_paths:
            with rasterio.open(p) as src:
                transform = src.transform
                if transform.e > 0:
                    # Flip south-up rasters to north-up (negative pixel height) for merging
                    data = src.read()
                    profile = src.profile
                    new_transform = transform * Affine.translation(0, src.height) * Affine.scale(1, -1)
                    data_flipped = data[:, ::-1, :]
                    profile.update(transform=new_transform)
                    fixed_path = tmp_dir / f"{p.stem}_upright.tif"
                    with rasterio.open(fixed_path, "w", **profile) as dst:
                        dst.write(data_flipped)
                    prepared_paths.append(fixed_path)
                else:
                    prepared_paths.append(p)

        srcs = [rasterio.open(p) for p in prepared_paths]
        def merge_single_band(band_index: int):
            """
            Merge one band at a time using temporary single-band datasets.
            """
            memfiles: List[MemoryFile] = []
            band_datasets = []
            try:
                for src in srcs:
                    band_data = src.read(band_index, out_dtype=merge_dtype)
                    profile = src.profile.copy()
                    profile.update(count=1, dtype=merge_dtype)
                    mf = MemoryFile()
                    ds = mf.open(**profile)
                    ds.write(band_data, 1)
                    memfiles.append(mf)
                    band_datasets.append(ds)
                mosaic, transform = rio_merge(band_datasets, dtype=merge_dtype)
                return mosaic, transform
            finally:
                for ds in band_datasets:
                    ds.close()
                for mf in memfiles:
                    mf.close()

        try:
            first_mosaic, transform = merge_single_band(1)
            meta = srcs[0].meta.copy()
            meta.update(
                {
                    "height": first_mosaic.shape[1],
                    "width": first_mosaic.shape[2],
                    "transform": transform,
                    "count": srcs[0].count,
                    "dtype": merge_dtype,
                }
            )

            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(np.squeeze(first_mosaic, axis=0), 1)
                for band in range(2, srcs[0].count + 1):
                    mosaic, t = merge_single_band(band)
                    if t != transform:
                        raise ValueError("Band transforms differ during merge; aborting.")
                    dst.write(np.squeeze(mosaic, axis=0), band)
        finally:
            for s in srcs:
                s.close()
        return out_path

    def extract(
        self,
        aoi_geojson: dict,
        year: int,
        season: str,  # ignored for annual data
        resolution_m: float | None,
        temp_dir: str | Path | None = None,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Path, float | None]:
        """
        Download annual AlphaEarth embeddings for the given year.
        Returns (path, effective resolution).
        """
        self._initialize()
        # Keep intermediates on the caller-specified disk to avoid filling /tmp.
        tmp_dir = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)
        region = ee.Geometry(aoi_geojson)

        start = f"{year}-01-01"
        end = f"{year + 1}-01-01"
        collection = (
            ee.ImageCollection(self.collection_id)
            .filterDate(start, end)
            .filterBounds(region)
        )
        image = collection.first()
        try:
            if image is None or image.getInfo() is None:
                raise ValueError
        except Exception:
            raise ValueError(f"No Alpha Earth embedding found for year {year}")

        meta = get_variable_metadata("alpha_earth")
        native_res = meta.get("native_resolution_m")
        scale = resolution_m if resolution_m is not None else native_res
        if scale is None:
            raise ValueError("No resolution provided and no native resolution defined for Alpha Earth.")

        name = f"alphaearth_{year}"
        try:
            path = self._download_image(image, aoi_geojson, scale, name, tmp_dir)
            self._notify(progress_cb, f"{name}: downloaded full AOI without tiling")
            return path, scale
        except Exception as exc:
            if "total request size" not in str(exc).lower():
                raise
            self.logger.info("Falling back to tiling due to request size limit.")

        tile_deg = TILE_DEG
        last_exc: Exception | None = None
        while tile_deg >= MIN_TILE_DEG:
            tile_regions = self._tile_aoi(aoi_geojson, tile_deg)
            if not tile_regions:
                raise ValueError("AOI produced no tiles.")

            total_tiles = len(tile_regions)
            self._notify(
                progress_cb,
                f"{name}: tiling AOI into {total_tiles} tile(s) at {tile_deg:.3f}°",
            )

            tile_paths: List[Path] = []
            for idx, tile_region in enumerate(tile_regions):
                tile_name = f"{name}_tile{idx}"
                try:
                    tile_path = self._download_image(image, tile_region, scale, tile_name, tmp_dir)
                except Exception as exc:
                    if "total request size" in str(exc).lower():
                        last_exc = exc
                        self.logger.info(
                            "Tile %s too large at %.3f°, will retry with smaller tiles.",
                            tile_name,
                            tile_deg,
                        )
                        break
                    raise
                tile_paths.append(tile_path)
                self._notify(progress_cb, f"{name}: downloaded tile {idx + 1}/{total_tiles}")

            if len(tile_paths) == total_tiles:
                merged = self._merge_tiles(tile_paths, name, tmp_dir)
                self._notify(progress_cb, f"{name}: merged {total_tiles} tiles")
                return merged, scale

            tile_deg /= 2
            self._notify(progress_cb, f"{name}: retrying with smaller tiles ({tile_deg:.3f}°)")

        raise RuntimeError(
            f"Could not download {name}: request still too large even after tiling"
        ) from last_exc
