import logging
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Tuple, List

import ee
import requests
import rasterio
from rasterio.merge import merge as rio_merge
from shapely.geometry import box, shape, mapping
from spatial_data_mining.variables.metadata import get_variable_metadata
from affine import Affine

TILE_DEG = 0.05  # fallback tile size in degrees when request is too large for AlphaEarth (dense, many bands)


class AlphaEarthExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.collection_id = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

    def _initialize(self) -> None:
        try:
            ee.Initialize()
        except Exception:
            self.logger.info("Authenticating with Google Earth Engine...")
            ee.Authenticate()
            ee.Initialize()

    def _download_image(self, image: ee.Image, region_geojson: dict, scale: float, name: str) -> Path:
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

        tmp_dir = Path(tempfile.gettempdir())
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

    def _tile_aoi(self, aoi_geojson: dict) -> List[dict]:
        geom = shape(aoi_geojson)
        minx, miny, maxx, maxy = geom.bounds
        tiles = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                tile = box(x, y, min(x + TILE_DEG, maxx), min(y + TILE_DEG, maxy))
                inter = geom.intersection(tile)
                if not inter.is_empty:
                    tiles.append(mapping(inter))
                y += TILE_DEG
            x += TILE_DEG
        return tiles

    def _merge_tiles(self, tile_paths: List[Path], name: str) -> Path:
        tmp_dir = Path(tempfile.gettempdir())
        out_path = tmp_dir / f"{name}_merged.tif"
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
        try:
            mosaic, transform = rio_merge(srcs)
            meta = srcs[0].meta.copy()
            meta.update(
                {
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": transform,
                    "count": mosaic.shape[0],
                }
            )
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(mosaic)
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
    ) -> Tuple[Path, float | None]:
        """
        Download annual AlphaEarth embeddings for the given year.
        Returns (path, effective resolution).
        """
        self._initialize()
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
            path = self._download_image(image, aoi_geojson, scale, name)
            return path, scale
        except Exception as exc:
            if "total request size" not in str(exc).lower():
                raise
            self.logger.info("Falling back to tiling due to request size limit.")

        tile_regions = self._tile_aoi(aoi_geojson)
        if not tile_regions:
            raise ValueError("AOI produced no tiles.")

        tile_paths: List[Path] = []
        for idx, tile_region in enumerate(tile_regions):
            tile_name = f"{name}_tile{idx}"
            tile_path = self._download_image(image, tile_region, scale, tile_name)
            tile_paths.append(tile_path)

        merged = self._merge_tiles(tile_paths, name)
        return merged, scale
