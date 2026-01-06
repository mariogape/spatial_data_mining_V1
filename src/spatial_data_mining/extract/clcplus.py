import logging
import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import rasterio
from rasterio.coords import BoundingBox
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from spatial_data_mining.utils.cancellation import check_cancelled


class CLCPlusExtractor:
    """
    Extractor for user-provided CLCplus rasters (one per country).
    Selects the raster that overlaps the AOI the most and returns its path.
    """

    def __init__(self, input_dir: str | Path | None):
        self.logger = logging.getLogger(__name__)
        self.input_dir = Path(input_dir) if input_dir else None

    @staticmethod
    def _notify(cb: Optional[Callable[[str], None]], message: str) -> None:
        if cb:
            cb(message)

    def _list_rasters(self) -> List[Path]:
        if self.input_dir is None:
            raise ValueError("CLCplus input_dir not provided.")
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise FileNotFoundError(f"CLCplus input_dir not found or not a directory: {self.input_dir}")
        patterns = ("*.tif", "*.tiff", "*.vrt")
        rasters: List[Path] = []
        for pat in patterns:
            rasters.extend(self.input_dir.glob(pat))
        rasters = sorted(set(rasters))
        if not rasters:
            raise FileNotFoundError(f"No GeoTIFF/VRT files found under {self.input_dir}")
        return rasters

    @staticmethod
    def _calc_native_resolution(src: rasterio.DatasetReader) -> float | None:
        try:
            res = src.res
            if res and res[0] and res[1]:
                return float(max(abs(res[0]), abs(res[1])))
        except Exception:
            return None
        return None

    @staticmethod
    def _project_geom(geom_wgs84, dst_crs) -> object:
        if dst_crs is None:
            return geom_wgs84
        transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        return shp_transform(transformer.transform, geom_wgs84)

    def _select_best_raster(
        self,
        aoi_geojson: dict,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Path, float | None]:
        aoi_geom_wgs84 = shape(aoi_geojson)
        candidates = self._list_rasters()

        best_path: Path | None = None
        best_native_res: float | None = None
        best_overlap: float = 0.0
        multiple_hits: List[Path] = []

        for path in candidates:
            check_cancelled(should_stop)
            with rasterio.open(path) as src:
                aoi_in_src = self._project_geom(aoi_geom_wgs84, src.crs)
                raster_geom = box(*BoundingBox(*src.bounds))
                if not raster_geom.intersects(aoi_in_src):
                    continue
                overlap_area = raster_geom.intersection(aoi_in_src).area
                if overlap_area > 0:
                    multiple_hits.append(path)
                if overlap_area > best_overlap:
                    best_overlap = overlap_area
                    best_path = path
                    best_native_res = self._calc_native_resolution(src)

        if best_path is None:
            raise RuntimeError(
                f"No CLCplus raster in {self.input_dir} intersects the AOI. "
                "Verify you selected the folder with the correct country files."
            )

        if len(multiple_hits) > 1:
            self.logger.info(
                "AOI intersects multiple CLCplus rasters; selected %s (largest overlap). Candidates: %s",
                best_path.name,
                ", ".join(p.name for p in multiple_hits),
            )

        return best_path, best_native_res

    @staticmethod
    def _stage_source(src_path: Path, temp_dir: Path) -> Path:
        """
        Stage the source raster into a temp directory so transforms do not write
        alongside user-provided inputs. Prefer hard links when possible.
        """
        temp_dir.mkdir(parents=True, exist_ok=True)
        staged_path = temp_dir / src_path.name
        if staged_path.exists():
            return staged_path
        try:
            os.link(src_path, staged_path)
        except Exception:
            shutil.copy2(src_path, staged_path)
        return staged_path

    def extract(
        self,
        aoi_geojson: dict,
        year: int,
        season: str,
        resolution_m: float | None,
        temp_dir: str | Path | None = None,
        progress_cb: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Path, float | None]:
        """
        Return the path to the CLCplus raster covering the AOI.
        No download occurs; we simply pick the best-overlapping local file.
        """
        check_cancelled(should_stop)
        path, native_res = self._select_best_raster(aoi_geojson, should_stop=should_stop)
        if temp_dir is not None:
            staged_path = self._stage_source(path, Path(temp_dir))
            if staged_path != path:
                self._notify(progress_cb, f"CLCplus: staged source raster in {staged_path.parent}")
            path = staged_path
        effective_res = resolution_m if resolution_m is not None else native_res
        self._notify(progress_cb, f"CLCplus: selected source raster {path.name}")
        return path, effective_res
