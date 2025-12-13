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
from shapely.geometry import box, shape, mapping
from spatial_data_mining.variables.metadata import get_variable_metadata

SEASON_RANGES = {
    "spring": ("03-01", "05-31"),
    "summer": ("06-01", "08-31"),
    "autumn": ("09-01", "11-30"),
    "fall": ("09-01", "11-30"),
    "winter": ("12-01", "02-28"),
    "annual": ("01-01", "12-31"),
    "year": ("01-01", "12-31"),
}

# Initial tile size (degrees) and the smallest size we'll try when falling back.
TILE_DEG = 0.05
MIN_TILE_DEG = 0.01

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

    def _apply_index(self, image: ee.Image) -> ee.Image:
        if self.index == "NDVI":
            return image.normalizedDifference(["B8", "B4"]).rename("ndvi")
        if self.index == "NDMI":
            return image.normalizedDifference(["B8", "B11"]).rename("ndmi")
        if self.index == "MSI":
            return image.select("B11").divide(image.select("B8")).rename("msi")
        raise ValueError(f"Unsupported index for GEEExtractor: {self.index}")

    def _download_image(
        self,
        image: ee.Image,
        region_geojson: dict,
        resolution_m: float | None,
        name: str,
        tmp_dir: Path,
    ) -> Path:
        params = {
            "region": region_geojson,  # region in WGS84
            "fileFormat": "GeoTIFF",
            "filePerBand": False,
        }
        if resolution_m is not None:
            params["scale"] = resolution_m  # units of the image's native projection
        url = image.getDownloadURL(params)
        resp = requests.get(url, timeout=600)
        if resp.status_code >= 400:
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
        resolution_m: float | None,
        temp_dir: str | Path | None = None,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Path, float | None]:
        """
        Download a seasonal median Sentinel-2 index image clipped to AOI.
        Returns (local temporary GeoTIFF path, effective resolution in meters).
        """
        self._initialize()
        # Keep intermediates on the caller-specified disk to avoid filling /tmp.
        tmp_dir = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)
        region = ee.Geometry(aoi_geojson)
        start_date, end_date = season_date_range(year, season)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        )
        base_image = collection.median()

        # Apply index; projection will follow the image defaults.
        image = self._apply_index(base_image)

        name = f"{self.index.lower()}_{year}_{season}"

        # If resolution not provided, use the native scale from metadata.
        meta = get_variable_metadata(self.index.lower())
        native_scale = meta.get("native_resolution_m")
        effective_res = resolution_m if resolution_m is not None else native_scale

        try:
            path = self._download_image(image, aoi_geojson, effective_res, name, tmp_dir)
            self._notify(progress_cb, f"{name}: downloaded full AOI without tiling")
            return path, effective_res
        except Exception as exc:
            if "total request size" not in str(exc).lower():
                raise
            self.logger.info("Falling back to tiling due to request size limit.")

        # Tile the AOI in WGS84 and merge tiles locally. If a tile still exceeds
        # request limits, progressively shrink tiles until we succeed or hit
        # MIN_TILE_DEG.
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
                    tile_path = self._download_image(
                        image,
                        tile_region,
                        effective_res,
                        tile_name,
                        tmp_dir,
                    )
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
                self._notify(
                    progress_cb,
                    f"{name}: downloaded tile {idx + 1}/{total_tiles}",
                )

            if len(tile_paths) == total_tiles:
                merged = self._merge_tiles(tile_paths, name, tmp_dir)
                self._notify(progress_cb, f"{name}: merged {total_tiles} tiles")
                return merged, effective_res

            tile_deg /= 2
            self._notify(progress_cb, f"{name}: retrying with smaller tiles ({tile_deg:.3f}°)")

        raise RuntimeError(
            f"Could not download {name}: request still too large even after tiling"
        ) from last_exc
