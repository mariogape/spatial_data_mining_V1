import logging
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import ee
import requests
import rasterio
from rasterio.merge import merge as rio_merge
from shapely.geometry import box, shape, mapping

SEASON_RANGES = {
    "spring": ("03-01", "05-31"),
    "summer": ("06-01", "08-31"),
    "autumn": ("09-01", "11-30"),
    "fall": ("09-01", "11-30"),
    "winter": ("12-01", "02-28"),
}

TILE_DEG = 0.2  # fallback tile size in degrees when request is too large

def season_date_range(year: int, season: str) -> Tuple[str, str]:
    season_l = season.lower()
    if season_l == "winter":
        return (f"{year}-12-01", f"{year + 1}-02-28")
    if season_l in SEASON_RANGES:
        start, end = SEASON_RANGES[season_l]
        return (f"{year}-{start}", f"{year}-{end}")
    return (f"{year}-01-01", f"{year}-12-31")


class GEEExtractor:
    def __init__(self, index: str):
        self.index = index.upper()
        self.logger = logging.getLogger(__name__)

    def _initialize(self) -> None:
        try:
            ee.Initialize()
        except Exception:
            self.logger.info("Authenticating with Google Earth Engine...")
            ee.Authenticate()
            ee.Initialize()

    def _apply_index(self, image: ee.Image) -> ee.Image:
        if self.index == "NDVI":
            return image.normalizedDifference(["B8", "B4"]).rename("ndvi")
        if self.index == "NDMI":
            return image.normalizedDifference(["B8", "B11"]).rename("ndmi")
        if self.index == "MSI":
            return image.select("B11").divide(image.select("B8")).rename("msi")
        raise ValueError(f"Unsupported index for GEEExtractor: {self.index}")

    def _download_image(self, image: ee.Image, region_geojson: dict, resolution_m: float, name: str) -> Path:
        url = image.getDownloadURL(
            {
                "scale": resolution_m,
                "crs": "EPSG:4326",
                "region": region_geojson,
                "fileFormat": "GeoTIFF",
                "filePerBand": False,
            }
        )
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()

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
        srcs = [rasterio.open(p) for p in tile_paths]
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
        season: str,
        resolution_m: float,
    ) -> Path:
        """
        Download a seasonal median Sentinel-2 index image clipped to AOI.
        Returns a local temporary GeoTIFF path (EPSG:4326).
        """
        self._initialize()
        region = ee.Geometry(aoi_geojson)
        start_date, end_date = season_date_range(year, season)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        )
        image = collection.median()
        image = self._apply_index(image)

        name = f"{self.index.lower()}_{year}_{season}"

        try:
            return self._download_image(image, aoi_geojson, resolution_m, name)
        except Exception as exc:
            if "Total request size" not in str(exc):
                raise
            self.logger.info("Falling back to tiling due to request size limit.")

        # Tile the AOI in WGS84 and merge tiles locally
        tile_regions = self._tile_aoi(aoi_geojson)
        if not tile_regions:
            raise ValueError("AOI produced no tiles.")

        tile_paths: List[Path] = []
        for idx, tile_region in enumerate(tile_regions):
            tile_name = f"{name}_tile{idx}"
            tile_path = self._download_image(image, tile_region, resolution_m, tile_name)
            tile_paths.append(tile_path)

        return self._merge_tiles(tile_paths, name)
