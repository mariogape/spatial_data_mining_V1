import json
import logging
import re
import struct
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

import requests
import rasterio
from rasterio.merge import merge as rio_merge
from shapely.geometry import shape, box
from shapely.ops import transform as shp_transform
from shapely.prepared import prep as shp_prep
from pyproj import CRS, Transformer

from spatial_data_mining.utils.cancellation import check_cancelled


class SoilGridsExtractor:
    """
    Extractor for SoilGrids 250m tiles via ISRIC file service.

    The extractor downloads only the tiles intersecting the AOI, mosaics them,
    and lets the standard transform pipeline handle reprojection/clipping.
    """

    DEFAULT_BASE_URL = "https://files.isric.org/soilgrids/latest/data"
    DEFAULT_DEPTH = "15-30cm"
    DEFAULT_STAT = "mean"
    TILE_INDEX_VERSION = 3
    TILE_INDEX_FILENAME = "soilgrids_tile_index.json"
    LEGACY_TILE_INDEX_FILENAMES = (
        "soilgrids_tile_index_v3.json",
        "soilgrids_tile_index_v2.json",
        "soilgrids_tile_index_v1.json",
    )

    _TILE_DIR_RE = re.compile(r'href="(?:\./)?(tileSG-\d{3}-\d{3})/"')
    _TILE_FILE_RE = re.compile(r'href="(?:\./)?(tileSG-\d{3}-\d{3}_[0-9-]+\.tif)"')

    _TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 12: 8}

    _index_cache: Dict[str, dict] = {}

    def __init__(
        self,
        variable: str,
        depth: Optional[str] = None,
        stat: Optional[str] = None,
        base_url: Optional[str] = None,
        tile_index_path: Optional[str] = None,
    ):
        self.variable = str(variable).lower()
        self.depth = self._normalize_depth(depth or self.DEFAULT_DEPTH)
        self.stat = self._normalize_stat(stat or self.DEFAULT_STAT)
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.tile_index_path = tile_index_path
        self.logger = logging.getLogger(__name__)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "spatial-data-mining/soilgrids"})

    @staticmethod
    def _notify(cb: Optional[Callable[[str], None]], message: str) -> None:
        if cb:
            cb(message)

    @staticmethod
    def _normalize_depth(depth: str) -> str:
        depth = str(depth).strip().lower().replace(" ", "")
        if not depth.endswith("cm"):
            depth = f"{depth}cm"
        return depth

    @staticmethod
    def _normalize_stat(stat: str) -> str:
        stat = str(stat).strip()
        if not stat:
            return SoilGridsExtractor.DEFAULT_STAT
        stat_lower = stat.lower()
        if stat_lower.startswith("q"):
            return f"Q{stat_lower[1:]}"
        return stat_lower

    def _variable_dir(self) -> str:
        return f"{self.base_url}/{self.variable}/{self.variable}_{self.depth}_{self.stat}/"

    def _resolve_tile_index_path(self) -> Path:
        if self.tile_index_path:
            return Path(self.tile_index_path)
        project_root = Path(__file__).resolve().parents[3]
        return project_root / "data" / "cache" / self.TILE_INDEX_FILENAME

    def _fetch_text(self, url: str) -> str:
        resp = self._session.get(url, timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} while fetching {url}")
        return resp.text

    def _list_tile_dirs(self, url: str) -> List[str]:
        html = self._fetch_text(url)
        dirs = sorted(set(self._TILE_DIR_RE.findall(html)))
        if not dirs:
            raise RuntimeError(f"No tile directories found at {url}")
        return dirs

    def _list_tile_files(self, url: str) -> List[str]:
        html = self._fetch_text(url)
        return sorted(set(self._TILE_FILE_RE.findall(html)))

    def _fetch_range(self, url: str, end: int) -> bytes:
        headers = {"Range": f"bytes=0-{end}"}
        resp = self._session.get(url, headers=headers, timeout=60)
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {resp.status_code} while fetching {url}")
        return resp.content

    @classmethod
    def _parse_ifd(cls, data: bytes, endian_fmt: str, ifd_offset: int) -> Tuple[dict, int]:
        if ifd_offset + 2 > len(data):
            return {}, ifd_offset + 2
        num_entries = struct.unpack(endian_fmt + "H", data[ifd_offset : ifd_offset + 2])[0]
        needed = ifd_offset + 2 + num_entries * 12
        if needed > len(data):
            return {}, needed
        entries = {}
        pos = ifd_offset + 2
        for _ in range(num_entries):
            tag, typ, count, value_offset = struct.unpack(
                endian_fmt + "HHII", data[pos : pos + 12]
            )
            pos += 12
            size = cls._TYPE_SIZES.get(typ, 1) * count
            entries[tag] = (typ, count, value_offset, size)
        return entries, needed

    @classmethod
    def _read_tag(cls, data: bytes, endian_fmt: str, entry: Tuple[int, int, int, int]):
        typ, count, value_offset, size = entry
        if size <= 4:
            raw = struct.pack(endian_fmt + "I", value_offset)[:size]
        else:
            raw = data[value_offset : value_offset + size]
        if typ == 3:
            return list(struct.unpack(endian_fmt + "H" * count, raw))
        if typ == 4:
            return list(struct.unpack(endian_fmt + "I" * count, raw))
        if typ == 5:
            vals = []
            for idx in range(count):
                num, den = struct.unpack(endian_fmt + "II", raw[idx * 8 : idx * 8 + 8])
                vals.append(num / den if den else None)
            return vals
        if typ == 12:
            return list(struct.unpack(endian_fmt + "d" * count, raw))
        if typ == 2:
            return raw.rstrip(b"\x00").decode("ascii", errors="ignore")
        if typ == 1:
            return list(raw)
        return raw

    @classmethod
    def _extract_wkt(cls, geo_ascii: str) -> Optional[str]:
        if not geo_ascii:
            return None
        if "ESRI PE String" in geo_ascii:
            tail = geo_ascii.split("ESRI PE String", 1)[1]
            if "=" in tail:
                tail = tail.split("=", 1)[1]
            start = tail.find("PROJCS[")
            if start == -1:
                start = tail.find("GEOGCS[")
            if start != -1:
                return tail[start:].split("|", 1)[0].strip()
        if geo_ascii.startswith(("PROJCS[", "GEOGCS[")):
            return geo_ascii.split("|", 1)[0].strip()
        return None

    def _read_tile_metadata(self, url: str) -> Tuple[Tuple[float, float, float, float], Optional[str]]:
        data = self._fetch_range(url, 4096 - 1)
        if len(data) < 8:
            raise ValueError(f"Could not read TIFF header from {url}")
        endian_tag = data[:2]
        endian_fmt = "<" if endian_tag == b"II" else ">"
        ifd_offset = struct.unpack(endian_fmt + "I", data[4:8])[0]

        entries, needed = self._parse_ifd(data, endian_fmt, ifd_offset)
        if not entries and needed > len(data):
            data = self._fetch_range(url, needed - 1)
            entries, needed = self._parse_ifd(data, endian_fmt, ifd_offset)

        if not entries:
            raise ValueError(f"Could not parse TIFF IFD in {url}")

        required_tags = [256, 257, 33550, 33922, 34737]
        max_needed = 0
        for tag_id in required_tags:
            entry = entries.get(tag_id)
            if entry and entry[3] > 4:
                max_needed = max(max_needed, entry[2] + entry[3])
        if max_needed > len(data):
            data = self._fetch_range(url, max_needed - 1)
            entries, _ = self._parse_ifd(data, endian_fmt, ifd_offset)

        width = self._read_tag(data, endian_fmt, entries[256])[0]
        height = self._read_tag(data, endian_fmt, entries[257])[0]
        scale = self._read_tag(data, endian_fmt, entries[33550])
        tie = self._read_tag(data, endian_fmt, entries[33922])
        geo_ascii = None
        if 34737 in entries:
            geo_ascii = self._read_tag(data, endian_fmt, entries[34737])

        if not scale or not tie:
            raise ValueError(f"Missing GeoTIFF tags in {url}")

        x_min = float(tie[3])
        y_max = float(tie[4])
        x_max = x_min + float(width) * float(scale[0])
        y_min = y_max - float(height) * float(scale[1])
        bbox = (x_min, y_min, x_max, y_max)

        wkt = self._extract_wkt(geo_ascii) if geo_ascii else None
        return bbox, wkt

    def _build_tile_index(
        self,
        index_url: str,
        progress_cb: Optional[Callable[[str], None]],
        should_stop: Optional[Callable[[], bool]],
    ) -> dict:
        self._notify(progress_cb, "building SoilGrids tile index (one-time)")
        tile_dirs = self._list_tile_dirs(index_url)
        total_dirs = len(tile_dirs)
        tiles: list[dict] = []
        crs_wkt = None
        to_wgs84 = None
        widths: list[float] = []
        heights: list[float] = []

        for idx, tile_dir in enumerate(tile_dirs, start=1):
            check_cancelled(should_stop)
            tile_url = f"{index_url}{tile_dir}/"
            tile_files = self._list_tile_files(tile_url)
            if not tile_files:
                continue
            for tile_file in tile_files:
                check_cancelled(should_stop)
                full_url = f"{tile_url}{tile_file}"
                try:
                    bbox, wkt = self._read_tile_metadata(full_url)
                except Exception as exc:
                    self.logger.warning("Skipping tile %s due to metadata error: %s", full_url, exc)
                    continue
                if crs_wkt is None and wkt:
                    crs_wkt = wkt
                if to_wgs84 is None and wkt:
                    try:
                        tile_crs = CRS.from_wkt(wkt)
                        to_wgs84 = Transformer.from_crs(tile_crs, "EPSG:4326", always_xy=True)
                    except Exception as exc:
                        self.logger.warning("Could not build SoilGrids CRS transformer: %s", exc)
                        to_wgs84 = None
                bbox_wgs84 = None
                if to_wgs84 is not None:
                    try:
                        xs = [bbox[0], bbox[2], bbox[2], bbox[0]]
                        ys = [bbox[1], bbox[1], bbox[3], bbox[3]]
                        lons, lats = to_wgs84.transform(xs, ys)
                        if all(v is not None for v in lons + lats):
                            bbox_wgs84 = (min(lons), min(lats), max(lons), max(lats))
                    except Exception:
                        bbox_wgs84 = None
                widths.append(abs(bbox[2] - bbox[0]))
                heights.append(abs(bbox[3] - bbox[1]))
                tiles.append(
                    {
                        "relpath": f"{tile_dir}/{tile_file}",
                        "bbox": bbox,
                        "bbox_wgs84": bbox_wgs84,
                    }
                )
            if idx % 50 == 0 or idx == total_dirs:
                self._notify(
                    progress_cb,
                    f"indexed {idx}/{total_dirs} tile folders",
                )

        if not tiles:
            raise RuntimeError("SoilGrids tile index is empty.")
        if not crs_wkt:
            raise RuntimeError("Could not extract SoilGrids CRS from tile headers.")

        tile_stats = {}
        if widths and heights:
            widths.sort()
            heights.sort()
            tile_stats = {
                "median_width": widths[len(widths) // 2],
                "median_height": heights[len(heights) // 2],
            }

        return {
            "version": self.TILE_INDEX_VERSION,
            "source_url": index_url,
            "crs_wkt": crs_wkt,
            "tile_stats": tile_stats,
            "tiles": tiles,
        }

    def _load_tile_index(
        self,
        index_url: str,
        progress_cb: Optional[Callable[[str], None]],
        should_stop: Optional[Callable[[], bool]],
    ) -> dict:
        index_path = self._resolve_tile_index_path()
        cache_key = f"{index_url}|{index_path}"
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]

        cache_paths = [index_path]
        for legacy_name in self.LEGACY_TILE_INDEX_FILENAMES:
            legacy_path = index_path.with_name(legacy_name)
            if legacy_path not in cache_paths:
                cache_paths.append(legacy_path)

        for cache_path in cache_paths:
            if not cache_path.exists():
                continue
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if data.get("version") == self.TILE_INDEX_VERSION and data.get("tiles"):
                    if cache_path != index_path:
                        index_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            index_path.write_text(json.dumps(data), encoding="utf-8")
                        except Exception as exc:
                            self.logger.warning(
                                "Could not write tile index cache %s: %s", index_path, exc
                            )
                    self._index_cache[cache_key] = data
                    return data
            except Exception as exc:
                self.logger.warning("Failed to read tile index cache %s: %s", cache_path, exc)

        data = self._build_tile_index(index_url, progress_cb, should_stop)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            index_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            self.logger.warning("Could not write tile index cache %s: %s", index_path, exc)

        self._index_cache[cache_key] = data
        return data

    def _download_tile(self, url: str, dst_path: Path) -> Path:
        resp = self._session.get(url, stream=True, timeout=600)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} while downloading {url}")
        with dst_path.open("wb") as dst:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    dst.write(chunk)
        return dst_path

    def _merge_tiles(
        self,
        tile_paths: List[Path],
        name: str,
        tmp_dir: Path,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Path:
        out_path = tmp_dir / f"{name}_merged.tif"
        srcs = [rasterio.open(p) for p in tile_paths]
        success = False
        try:
            check_cancelled(should_stop)
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
                check_cancelled(should_stop)
                dst.write(mosaic)
            success = True
        finally:
            for src in srcs:
                src.close()
            if success:
                for p in set(tile_paths):
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
                    except Exception as exc:
                        self.logger.warning("Could not delete tile %s: %s", p, exc)
        return out_path

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
        Download SoilGrids tiles intersecting the AOI and mosaic them.
        Returns (local GeoTIFF path, effective resolution in meters).
        """
        check_cancelled(should_stop)
        tmp_dir = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)

        index_url = self._variable_dir()
        tile_index = self._load_tile_index(index_url, progress_cb, should_stop)
        crs_wkt = tile_index.get("crs_wkt")
        if not crs_wkt:
            raise RuntimeError("SoilGrids tile index missing CRS WKT.")

        tile_crs = CRS.from_wkt(crs_wkt)
        geom_wgs84 = shape(aoi_geojson)
        transformer = Transformer.from_crs("EPSG:4326", tile_crs, always_xy=True)
        geom_tile = shp_transform(transformer.transform, geom_wgs84)
        if geom_tile.is_empty:
            raise ValueError("AOI geometry is empty after reprojecting to SoilGrids CRS.")
        if not geom_tile.is_valid:
            try:
                geom_tile = geom_tile.buffer(0)
            except Exception:
                pass
        prepared = shp_prep(geom_tile)
        aoi_bounds = geom_tile.bounds
        aoi_bounds_wgs84 = geom_wgs84.bounds

        def _bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
            return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

        tile_stats = tile_index.get("tile_stats") or {}
        buffer_x = float(tile_stats.get("median_width", 0.0)) * 0.5
        buffer_y = float(tile_stats.get("median_height", 0.0)) * 0.5
        buffered_bounds = (
            aoi_bounds[0] - buffer_x,
            aoi_bounds[1] - buffer_y,
            aoi_bounds[2] + buffer_x,
            aoi_bounds[3] + buffer_y,
        )

        hits_geom: list[dict] = []
        hits_bbox: list[dict] = []
        hits_bbox_wgs84: list[dict] = []
        for tile in tile_index["tiles"]:
            bbox = tile.get("bbox")
            if not bbox:
                continue
            if prepared.intersects(box(*bbox)):
                hits_geom.append(tile)
            if _bbox_intersects(bbox, buffered_bounds):
                hits_bbox.append(tile)
            bbox_wgs84 = tile.get("bbox_wgs84")
            if bbox_wgs84 and _bbox_intersects(bbox_wgs84, aoi_bounds_wgs84):
                hits_bbox_wgs84.append(tile)

        # Prefer tile-CRS bbox selection; fall back to WGS84 only if needed.
        hits = []
        seen = set()
        groups = (hits_bbox, hits_geom) if hits_bbox else (hits_bbox_wgs84, hits_geom)
        for group in groups:
            for tile in group:
                relpath = tile.get("relpath")
                if not relpath or relpath in seen:
                    continue
                seen.add(relpath)
                hits.append(tile)

        if not hits:
            raise ValueError("AOI does not intersect any SoilGrids tiles.")

        name = f"soilgrids_{self.variable}_{self.depth}_{self.stat}"
        self._notify(progress_cb, f"{name}: downloading {len(hits)} tile(s)")

        tile_paths: list[Path] = []
        for idx, tile in enumerate(hits, start=1):
            check_cancelled(should_stop)
            relpath = tile["relpath"]
            url = f"{index_url}{relpath}"
            dst_path = tmp_dir / Path(relpath).name
            try:
                self._download_tile(url, dst_path)
            except Exception as exc:
                self.logger.warning("Skipping tile %s due to download error: %s", url, exc)
                continue
            tile_paths.append(dst_path)
            if idx % 10 == 0 or idx == len(hits):
                self._notify(progress_cb, f"{name}: downloaded {idx}/{len(hits)} tiles")

        if not tile_paths:
            raise RuntimeError("No SoilGrids tiles were downloaded successfully.")

        merged = self._merge_tiles(tile_paths, name, tmp_dir, should_stop=should_stop)
        self._notify(progress_cb, f"{name}: merged {len(tile_paths)} tiles")

        effective_res = resolution_m if resolution_m is not None else None
        return merged, effective_res
