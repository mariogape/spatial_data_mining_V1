import logging
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import requests
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shp_transform, unary_union

from spatial_data_mining.extract.openeo_indices import season_date_range
from spatial_data_mining.utils.cancellation import check_cancelled
from spatial_data_mining.variables.metadata import get_variable_metadata


class OpenEORGBExtractor:
    """
    Extractor for Sentinel-2 RGB imagery via openEO using a target date.
    """

    def __init__(
        self,
        *,
        rgb_date: str | None = None,
        search_days: int | None = None,
        collection_id: str | None = None,
        bands: str | None = None,
        cloud_cover_max: float | None = None,
        cloud_cover_property: str | None = None,
        cloud_mask: bool | None = None,
        cloud_mask_band: str | None = None,
        cloud_mask_classes: str | None = None,
        oidc_provider_id: str | None = None,
        stac_url: str | None = None,
        stac_collection_id: str | None = None,
        prefilter: bool | None = None,
        allow_mosaic: bool | None = None,
        mosaic_max_dates: int | None = None,
        min_coverage: float | None = None,
        backend_url: str | None = None,
        verify_ssl: bool | None = None,
        variable_name: str | None = None,
    ):
        self.variable_name = str(variable_name or "rgb").lower()
        self.rgb_date = self._normalize_date(rgb_date or os.getenv("OPENEO_RGB_DATE"))
        raw_search_days = (
            search_days
            if search_days is not None
            else os.getenv("OPENEO_RGB_SEARCH_DAYS")
        )
        if raw_search_days is None or str(raw_search_days).strip() == "":
            raw_search_days = 30
        self.search_days = self._normalize_search_days(raw_search_days)
        self.collection_id = (
            collection_id
            or os.getenv("OPENEO_RGB_COLLECTION_ID")
            or os.getenv("OPENEO_S2_COLLECTION_ID")
            or "SENTINEL2_L2A"
        )
        self.bands = self._parse_bands(bands or os.getenv("OPENEO_RGB_BANDS") or "B04,B03,B02")
        raw_cloud_cover = (
            cloud_cover_max
            if cloud_cover_max is not None
            else os.getenv("OPENEO_RGB_MAX_CLOUD_COVER")
            or os.getenv("OPENEO_MAX_CLOUD_COVER")
        )
        if raw_cloud_cover is None or str(raw_cloud_cover).strip() == "":
            raw_cloud_cover = 40
        self.cloud_cover_max = self._normalize_cloud_cover(raw_cloud_cover)
        self.cloud_cover_property = (
            cloud_cover_property
            or os.getenv("OPENEO_RGB_CLOUD_COVER_PROPERTY")
            or os.getenv("OPENEO_CLOUD_COVER_PROPERTY")
            or "eo:cloud_cover"
        )
        if cloud_mask is None:
            self.cloud_mask = self._env_truthy("OPENEO_RGB_CLOUD_MASK", default=True)
        else:
            self.cloud_mask = bool(cloud_mask)
        self.cloud_mask_band = (
            cloud_mask_band
            or os.getenv("OPENEO_RGB_CLOUD_MASK_BAND")
            or "SCL"
        )
        self.cloud_mask_classes = self._parse_cloud_mask_classes(
            cloud_mask_classes
            or os.getenv("OPENEO_RGB_CLOUD_MASK_CLASSES")
            or "0,1,3,8,9,10,11"
        )
        self.oidc_provider_id = (
            oidc_provider_id
            or os.getenv("OPENEO_RGB_OIDC_PROVIDER_ID")
            or os.getenv("OPENEO_OIDC_PROVIDER_ID")
        )
        self.stac_url = (
            stac_url
            or os.getenv("OPENEO_RGB_STAC_URL")
            or "https://catalogue.dataspace.copernicus.eu/stac"
        )
        self.stac_collection_id = (
            stac_collection_id
            or os.getenv("OPENEO_RGB_STAC_COLLECTION_ID")
            or self._default_stac_collection(self.collection_id)
        )
        if prefilter is None:
            self.prefilter = self._env_truthy("OPENEO_RGB_PREFILTER", default=True)
        else:
            self.prefilter = bool(prefilter)
        if allow_mosaic is None:
            self.allow_mosaic = self._env_truthy("OPENEO_RGB_ALLOW_MOSAIC", default=True)
        else:
            self.allow_mosaic = bool(allow_mosaic)
        raw_mosaic_max = (
            mosaic_max_dates
            if mosaic_max_dates is not None
            else os.getenv("OPENEO_RGB_MOSAIC_MAX_DATES")
        )
        if raw_mosaic_max is None or str(raw_mosaic_max).strip() == "":
            raw_mosaic_max = 5
        try:
            self.mosaic_max_dates = max(1, int(raw_mosaic_max))
        except Exception as exc:
            raise ValueError(
                f"RGB mosaic max dates must be an integer (got {raw_mosaic_max!r})."
            ) from exc
        raw_min_cov = (
            min_coverage
            if min_coverage is not None
            else os.getenv("OPENEO_RGB_MIN_COVERAGE")
        )
        if raw_min_cov is None or str(raw_min_cov).strip() == "":
            raw_min_cov = 0.999
        try:
            min_cov_val = float(raw_min_cov)
        except Exception as exc:
            raise ValueError(
                f"RGB min coverage must be a float between 0 and 1 (got {raw_min_cov!r})."
            ) from exc
        if not (0.0 < min_cov_val <= 1.0):
            raise ValueError(
                f"RGB min coverage must be in (0, 1] (got {min_cov_val})."
            )
        self.min_coverage = min_cov_val
        self.backend_url = (
            backend_url
            or os.getenv("OPENEO_RGB_BACKEND_URL")
            or os.getenv("OPENEO_BACKEND_URL", "https://openeo.dataspace.copernicus.eu")
        )
        self.verify_ssl = verify_ssl if verify_ssl is not None else self._env_truthy(
            "OPENEO_VERIFY_SSL", default=True
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _notify(cb: Optional[Callable[[str], None]], message: str) -> None:
        if cb:
            cb(message)

    @staticmethod
    def _env_truthy(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        val = str(raw).strip().lower()
        if val in {"1", "true", "yes", "y", "on"}:
            return True
        if val in {"0", "false", "no", "n", "off"}:
            return False
        return default

    @staticmethod
    def _default_stac_collection(collection_id: str | None) -> str | None:
        if not collection_id:
            return None
        mapping = {
            "SENTINEL2_L2A": "sentinel-2-l2a",
            "SENTINEL2_L2A_SENTINELHUB": "sentinel-2-l2a",
        }
        if collection_id in mapping:
            return mapping[collection_id]
        upper = str(collection_id).upper()
        if "SENTINEL2" in upper and "L2A" in upper:
            return "sentinel-2-l2a"
        if "SENTINEL2" in upper and "L1C" in upper:
            return "sentinel-2-l1c"
        return None

    @staticmethod
    def _normalize_date(value: str | None) -> str | None:
        if value is None:
            return None
        raw = value
        if hasattr(raw, "isoformat"):
            raw = raw.isoformat()
        text = str(raw).strip()
        if not text:
            return None
        try:
            datetime.fromisoformat(text).date()
        except ValueError as exc:
            raise ValueError(
                f"RGB date must be ISO format YYYY-MM-DD (got {text!r})."
            ) from exc
        return text

    @staticmethod
    def _normalize_search_days(value) -> int:
        try:
            val = int(value)
        except Exception as exc:
            raise ValueError(f"RGB search days must be an integer (got {value!r}).") from exc
        return max(0, val)

    @staticmethod
    def _normalize_cloud_cover(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _parse_bands(value: str | None) -> list[str]:
        if value is None:
            return []
        parts = [p.strip() for p in str(value).split(",")]
        return [p for p in parts if p]

    @staticmethod
    def _parse_cloud_mask_classes(value: str | None) -> list[int]:
        if value is None:
            return []
        parts = [p.strip() for p in str(value).split(",")]
        classes: list[int] = []
        for part in parts:
            if not part:
                continue
            try:
                classes.append(int(part))
            except Exception:
                continue
        return classes

    @staticmethod
    def _default_date_for_season(season: str, year: int) -> date | None:
        if str(season).lower() == "static":
            return None
        try:
            start_str, end_str = season_date_range(int(year), season)
            start_dt = datetime.fromisoformat(start_str).date()
            end_dt = datetime.fromisoformat(end_str).date()
            return start_dt + (end_dt - start_dt) // 2
        except Exception:
            return None

    @staticmethod
    def _candidate_dates(target: date, search_days: int):
        yield target
        for offset in range(1, search_days + 1):
            yield target - timedelta(days=offset)
            yield target + timedelta(days=offset)

    def _stac_search_items(
        self,
        bbox: list[float],
        start_date: date,
        end_date: date,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> list[dict]:
        if not self.stac_collection_id:
            return []
        base_url = self.stac_url.rstrip("/")
        search_url = f"{base_url}/search"
        payload = {
            "collections": [self.stac_collection_id],
            "bbox": bbox,
            "datetime": f"{start_date.isoformat()}T00:00:00Z/{end_date.isoformat()}T23:59:59Z",
            "limit": 200,
        }
        items: list[dict] = []
        next_url = search_url
        method = "POST"
        body = payload
        while next_url:
            if method.upper() == "POST":
                resp = requests.post(next_url, json=body, timeout=60)
            else:
                resp = requests.get(next_url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            items.extend(data.get("features") or [])
            if len(items) >= 2000:
                break
            next_link = None
            for link in data.get("links", []):
                if link.get("rel") == "next":
                    next_link = link
                    break
            if not next_link:
                break
            next_url = next_link.get("href")
            method = (next_link.get("method") or "GET").upper()
            body = next_link.get("body")
        if progress_cb:
            self._notify(progress_cb, f"RGB: STAC search found {len(items)} items")
        return items

    def _stac_candidate_dates(
        self,
        aoi_geojson: dict,
        target_date: date,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> list[str]:
        if not self.prefilter or not self.stac_collection_id:
            return []
        aoi_geom = shape(aoi_geojson)
        if hasattr(aoi_geom, "is_valid") and not aoi_geom.is_valid:
            try:
                aoi_geom = aoi_geom.buffer(0)
            except Exception:
                pass
        if getattr(aoi_geom, "is_empty", False):
            return []
        aoi_area = getattr(aoi_geom, "area", 0.0) or 0.0
        minx, miny, maxx, maxy = aoi_geom.bounds
        search_days = self.search_days
        start_date = target_date - timedelta(days=search_days)
        end_date = target_date + timedelta(days=search_days)
        try:
            items = self._stac_search_items(
                [minx, miny, maxx, maxy],
                start_date,
                end_date,
                progress_cb=progress_cb,
            )
        except Exception as exc:
            self.logger.warning("RGB STAC prefilter failed; falling back to scan (%s)", exc)
            return []

        per_date: dict[date, dict[str, object]] = {}
        for item in items:
            geom = item.get("geometry")
            if not geom:
                bbox = item.get("bbox")
                if bbox and len(bbox) >= 4:
                    geom = box(*bbox[:4])
            if not geom:
                continue
            try:
                item_geom = shape(geom)
            except Exception:
                continue
            if hasattr(item_geom, "is_valid") and not item_geom.is_valid:
                try:
                    item_geom = item_geom.buffer(0)
                except Exception:
                    continue
            if getattr(item_geom, "is_empty", False):
                continue
            if not item_geom.intersects(aoi_geom):
                continue
            props = item.get("properties", {}) or {}
            dt_raw = props.get("datetime") or props.get("start_datetime") or ""
            dt_str = str(dt_raw)[:10]
            try:
                dt = datetime.fromisoformat(dt_str).date()
            except Exception:
                continue
            if dt > date.today():
                continue
            cloud = props.get("eo:cloud_cover")
            if cloud is None:
                cloud = props.get("cloudCover")
            if cloud is None:
                cloud = props.get("s2:cloud_coverage")
            if self.cloud_cover_max is not None and cloud is not None:
                try:
                    if float(cloud) > float(self.cloud_cover_max):
                        continue
                except Exception:
                    pass

            entry = per_date.setdefault(dt, {"geoms": [], "clouds": []})
            entry["geoms"].append(item_geom)
            if cloud is not None:
                try:
                    entry["clouds"].append(float(cloud))
                except Exception:
                    pass

        candidates: list[tuple[int, float, date]] = []
        for dt, info in per_date.items():
            geoms = info.get("geoms", [])
            if not geoms:
                continue
            try:
                union_geom = unary_union(geoms)
            except Exception:
                try:
                    union_geom = unary_union([box(*g.bounds) for g in geoms])
                except Exception:
                    continue
            if hasattr(union_geom, "is_valid") and not union_geom.is_valid:
                try:
                    union_geom = union_geom.buffer(0)
                except Exception:
                    pass
            if getattr(union_geom, "is_empty", False):
                continue
            covered = False
            try:
                covered = union_geom.covers(aoi_geom)
            except Exception:
                covered = False
            if not covered:
                if aoi_area > 0:
                    try:
                        coverage = union_geom.intersection(aoi_geom).area / aoi_area
                    except Exception:
                        coverage = 0.0
                    if coverage < 0.999:
                        continue
                else:
                    continue
            clouds = info.get("clouds", [])
            avg_cloud = float(sum(clouds) / len(clouds)) if clouds else 1e9
            delta = abs((dt - target_date).days)
            candidates.append((delta, avg_cloud, dt))

        if not candidates:
            return []
        candidates.sort()
        return [dt.isoformat() for _, __, dt in candidates]

    @staticmethod
    def _project_geom(geom_wgs84, dst_crs) -> object:
        if dst_crs is None:
            return geom_wgs84
        transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        return shp_transform(transformer.transform, geom_wgs84)

    @staticmethod
    def _has_full_coverage(path: Path, aoi_geojson: dict) -> bool:
        geom_wgs84 = shape(aoi_geojson)
        with rasterio.open(path) as src:
            if src.crs is None:
                return False
            geom = OpenEORGBExtractor._project_geom(geom_wgs84, src.crs)
            inside_mask = geometry_mask(
                [mapping(geom)],
                transform=src.transform,
                invert=True,
                out_shape=(src.height, src.width),
            )
            if not inside_mask.any():
                return False
            data = src.read(masked=True)
            mask = np.ma.getmaskarray(data)
            if mask is not np.ma.nomask:
                if np.any(mask & inside_mask):
                    return False
            if np.issubdtype(data.dtype, np.floating):
                if np.any(np.isnan(data.data) & inside_mask):
                    return False
        return True

    @staticmethod
    def _coverage_masks(path: Path, aoi_geojson: dict) -> tuple[np.ndarray, np.ndarray, dict, float | int | None]:
        geom_wgs84 = shape(aoi_geojson)
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError("RGB raster has no CRS")
            geom = OpenEORGBExtractor._project_geom(geom_wgs84, src.crs)
            inside_mask = geometry_mask(
                [mapping(geom)],
                transform=src.transform,
                invert=True,
                out_shape=(src.height, src.width),
            )
            data = src.read(masked=True)
            mask = np.ma.getmaskarray(data)
            if mask is np.ma.nomask:
                valid = np.ones((src.height, src.width), dtype=bool)
            else:
                valid = ~np.any(mask, axis=0)
            if np.issubdtype(data.dtype, np.floating):
                valid &= ~np.any(np.isnan(data.data), axis=0)
            profile = src.profile.copy()
            nodata = src.nodata
        return valid, inside_mask, profile, nodata

    @staticmethod
    def _coverage_ratio(valid_mask: np.ndarray, inside_mask: np.ndarray) -> float:
        denom = float(inside_mask.sum())
        if denom <= 0:
            return 0.0
        return float(valid_mask[inside_mask].sum()) / denom

    def _apply_cloud_mask(self, cube, progress_cb: Optional[Callable[[str], None]] = None):
        if not self.cloud_mask:
            return cube
        if not self.cloud_mask_band or not self.cloud_mask_classes:
            return cube
        try:
            if hasattr(cube, "band"):
                scl = cube.band(self.cloud_mask_band)
            elif hasattr(cube, "filter_bands"):
                scl = cube.filter_bands([self.cloud_mask_band])
            else:
                return cube
        except Exception as exc:
            self.logger.warning(
                "openEO cloud mask failed to access band %s (%s); continuing without mask.",
                self.cloud_mask_band,
                exc,
            )
            return cube

        mask = None
        for cls in self.cloud_mask_classes:
            try:
                cond = scl == int(cls)
            except Exception:
                continue
            mask = cond if mask is None else (mask | cond)

        if mask is None:
            return cube

        self._notify(
            progress_cb,
            f"openEO: applying cloud mask via {self.cloud_mask_band} classes {self.cloud_mask_classes}",
        )
        try:
            cube = cube.mask(mask)
        except Exception as exc:
            self.logger.warning(
                "openEO cloud mask application failed (%s); continuing without mask.",
                exc,
            )
            return cube

        if hasattr(cube, "filter_bands"):
            try:
                cube = cube.filter_bands(self.bands)
            except Exception:
                pass
        return cube

    def _build_mosaic(
        self,
        paths: list[Path],
        aoi_geojson: dict,
        tmp_dir: Path,
        target_label: str,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Path | None:
        if not paths:
            return None
        base_path = paths[0]
        with rasterio.open(base_path) as src:
            base_profile = src.profile.copy()
            base_transform = src.transform
            base_crs = src.crs
            base_data = src.read(masked=True)
        valid_base, inside_mask, _, nodata = self._coverage_masks(base_path, aoi_geojson)
        if not inside_mask.any():
            return None
        fill_value: float | int
        if nodata is not None:
            fill_value = nodata
        elif np.issubdtype(base_data.dtype, np.integer):
            fill_value = 0
        else:
            fill_value = np.nan
        data_out = np.array(base_data.filled(fill_value), copy=True)
        valid_out = valid_base.copy()

        for path in paths[1:]:
            with rasterio.open(path) as src:
                if src.crs != base_crs or src.transform != base_transform:
                    self.logger.warning(
                        "RGB mosaic skipped %s due to differing grid/CRS.", path
                    )
                    continue
                data_in = src.read(masked=True)
            valid_in, inside_mask_in, _, _ = self._coverage_masks(path, aoi_geojson)
            if inside_mask_in.shape != inside_mask.shape:
                self.logger.warning(
                    "RGB mosaic skipped %s due to differing raster shape.", path
                )
                continue
            to_fill = (~valid_out) & valid_in
            if np.any(to_fill):
                data_out[:, to_fill] = np.array(data_in.filled(fill_value))[:, to_fill]
                valid_out[to_fill] = True
            coverage = float(valid_out[inside_mask].sum()) / float(inside_mask.sum())
            self._notify(
                progress_cb,
                f"RGB: mosaic coverage {coverage:.2%} after {path.stem}",
            )
            if coverage >= 0.999:
                break

        if np.any(inside_mask & ~valid_out):
            return None

        out_path = tmp_dir / f"rgb_{target_label}_mosaic.tif"
        out_profile = base_profile.copy()
        out_profile.update(driver="GTiff")
        if nodata is not None:
            out_profile["nodata"] = nodata
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(data_out)
            mask = np.where(valid_out, 255, 0).astype("uint8")
            dst.write_mask(mask)
        return out_path

    def _connect_and_authenticate(self, progress_cb: Optional[Callable[[str], None]] = None):
        try:
            import openeo  # type: ignore
        except Exception as exc:  # pragma: no cover (env-dependent)
            raise ImportError(
                "Missing dependency 'openeo'. Install it with: pip install openeo"
            ) from exc

        session: requests.Session | None = None
        if not self.verify_ssl:
            session = requests.Session()
            session.verify = False
            self.logger.warning(
                "OPENEO_VERIFY_SSL=false: disabling TLS certificate verification for %s",
                self.backend_url,
            )
            self._notify(
                progress_cb,
                "WARNING: OPENEO_VERIFY_SSL=false: TLS verification is disabled (insecure).",
            )
            try:
                import urllib3

                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass

        try:
            conn = openeo.connect(self.backend_url, session=session)
        except requests.exceptions.SSLError as exc:
            raise RuntimeError(
                f"TLS verification failed when connecting to {self.backend_url}: {exc}. "
                "If the backend has an expired/broken certificate and you still want to test, "
                "set OPENEO_VERIFY_SSL=0 (insecure)."
            ) from exc

        try:
            conn.describe_account()
            return conn
        except Exception:
            pass

        self._notify(progress_cb, "Authenticating to openEO backend...")

        auth_method = os.getenv("OPENEO_AUTH_METHOD", "").strip().lower()
        oidc_provider_id = self.oidc_provider_id or None

        client_id = os.getenv("OPENEO_CLIENT_ID") or None
        client_secret = os.getenv("OPENEO_CLIENT_SECRET") or None
        username = os.getenv("OPENEO_USERNAME") or None
        password = os.getenv("OPENEO_PASSWORD") or None

        def _provider_error(exc: Exception) -> bool:
            msg = str(exc).lower()
            return "oidc provider" in msg and "not available" in msg

        def _auth_oidc(meth, *args):
            if oidc_provider_id:
                try:
                    return meth(*args, provider_id=oidc_provider_id)
                except TypeError:
                    return meth(*args, oidc_provider_id)
                except Exception as exc:
                    if _provider_error(exc):
                        self.logger.warning(
                            "Requested OIDC provider %s not available; retrying without provider_id.",
                            oidc_provider_id,
                        )
                        return meth(*args)
                    raise
            return meth(*args)

        if auth_method in {"client_credentials", "oidc_client_credentials"} and client_id and client_secret:
            meth = getattr(conn, "authenticate_oidc_client_credentials", None)
            if not meth:
                raise RuntimeError(
                    "OPENEO_AUTH_METHOD=client_credentials requires openeo client support for "
                    "authenticate_oidc_client_credentials()."
                )
            _auth_oidc(meth, client_id, client_secret)
            return conn

        if auth_method in {"basic"} and username and password:
            meth = getattr(conn, "authenticate_basic", None)
            if not meth:
                raise RuntimeError(
                    "OPENEO_AUTH_METHOD=basic requires openeo client support for authenticate_basic()."
                )
            meth(username, password)
            return conn

        if client_id and client_secret and hasattr(conn, "authenticate_oidc_client_credentials"):
            _auth_oidc(conn.authenticate_oidc_client_credentials, client_id, client_secret)
            return conn

        if username and password and hasattr(conn, "authenticate_basic"):
            conn.authenticate_basic(username, password)
            return conn

        if hasattr(conn, "authenticate_oidc"):
            _auth_oidc(conn.authenticate_oidc)
            return conn

        raise RuntimeError(
            "Could not authenticate to openEO. Set OPENEO_AUTH_METHOD and credentials env vars, "
            "or ensure your openeo client supports authenticate_oidc()."
        )

    def _reduce_to_single_time(self, cube):
        if hasattr(cube, "first"):
            try:
                return cube.first()
            except Exception:
                pass
        if hasattr(cube, "reduce_dimension"):
            for dimension in ("t", "time"):
                try:
                    return cube.reduce_dimension(dimension=dimension, reducer="first")
                except Exception:
                    continue
        return cube

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
        Download an RGB image for the closest available date near rgb_date.
        Returns (local temporary GeoTIFF path, effective resolution in meters).
        """
        check_cancelled(should_stop)

        tmp_dir = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)

        explicit_date = self.rgb_date
        if explicit_date:
            target_date = datetime.fromisoformat(explicit_date).date()
            target_label = explicit_date
        else:
            default_date = self._default_date_for_season(season, year)
            if default_date is None:
                raise ValueError(
                    "RGB requires an exact date (rgb_date / OPENEO_RGB_DATE) when season is static."
                )
            target_date = default_date
            target_label = target_date.isoformat()
            self._notify(
                progress_cb,
                f"RGB: using default mid-season date {target_label} for {season} {year}.",
            )
        today = datetime.now(timezone.utc).date()
        if target_date > today:
            raise ValueError(f"RGB date is in the future ({target_label}).")

        geom = shape(aoi_geojson)
        minx, miny, maxx, maxy = geom.bounds
        spatial_extent = {"west": minx, "south": miny, "east": maxx, "north": maxy}

        meta = get_variable_metadata(self.variable_name)
        native_scale = meta.get("native_resolution_m")
        effective_res = resolution_m if resolution_m is not None else native_scale

        conn = self._connect_and_authenticate(progress_cb=progress_cb)
        check_cancelled(should_stop)

        attempted: list[str] = []
        partial_paths: list[Path] = []
        combined_valid: np.ndarray | None = None
        combined_inside: np.ndarray | None = None
        self._notify(
            progress_cb,
            f"openEO: searching RGB near {target_label} (±{self.search_days} days)",
        )

        candidate_dates = self._stac_candidate_dates(
            aoi_geojson,
            target_date,
            progress_cb=progress_cb,
        )
        if candidate_dates:
            self._notify(
                progress_cb,
                f"RGB: STAC prefilter produced {len(candidate_dates)} candidate date(s).",
            )
        else:
            self._notify(
                progress_cb,
                "RGB: STAC prefilter found no full-coverage candidates; falling back to day-by-day search.",
            )
        if not candidate_dates:
            candidate_dates = [d.isoformat() for d in self._candidate_dates(target_date, self.search_days)]

        for date_str in candidate_dates:
            check_cancelled(should_stop)
            try:
                candidate = datetime.fromisoformat(date_str).date()
            except Exception:
                continue
            if candidate > today:
                continue
            attempted.append(date_str)
            self._notify(progress_cb, f"RGB: trying {date_str}...")

            load_bands = list(self.bands)
            if self.cloud_mask and self.cloud_mask_band:
                if self.cloud_mask_band not in load_bands:
                    load_bands.append(self.cloud_mask_band)

            load_kwargs: dict[str, object] = {
                "spatial_extent": spatial_extent,
                "temporal_extent": [date_str, date_str],
                "bands": load_bands,
            }

            cloud_filter_desc = "no cloud filter"
            if self.cloud_cover_max is not None:
                if not self.cloud_cover_property or self.cloud_cover_property == "eo:cloud_cover":
                    load_kwargs["max_cloud_cover"] = self.cloud_cover_max
                    cloud_filter_desc = f"max_cloud_cover<{self.cloud_cover_max}"
                else:
                    try:
                        import openeo  # type: ignore

                        load_kwargs["properties"] = [
                            openeo.collection_property(self.cloud_cover_property) < self.cloud_cover_max
                        ]
                        cloud_filter_desc = f"{self.cloud_cover_property}<{self.cloud_cover_max}"
                    except Exception as exc:
                        self.logger.warning(
                            "Could not build cloud-cover property filter for %s (%s); continuing without it.",
                            self.cloud_cover_property,
                            exc,
                        )

            self._notify(
                progress_cb,
                f"openEO: loading {self.collection_id} ({date_str}) with {cloud_filter_desc}",
            )
            try:
                cube = conn.load_collection(self.collection_id, **load_kwargs)
            except Exception as exc:
                if "max_cloud_cover" in load_kwargs or "properties" in load_kwargs:
                    self.logger.warning(
                        "openEO load_collection failed with cloud filter (%s); retrying without it.",
                        exc,
                    )
                    self._notify(
                        progress_cb,
                        "openEO: warning: cloud-cover filter failed; retrying without it.",
                    )
                    load_kwargs.pop("max_cloud_cover", None)
                    load_kwargs.pop("properties", None)
                    cube = conn.load_collection(self.collection_id, **load_kwargs)
                else:
                    raise

            try:
                cube = cube.mask_polygon(aoi_geojson)
            except Exception as exc:
                self.logger.warning("openEO mask_polygon failed; continuing without it (%s)", exc)

            cube = self._apply_cloud_mask(cube, progress_cb=progress_cb)

            cube = self._reduce_to_single_time(cube)

            name = f"rgb_{date_str}_openeo"
            out_path = tmp_dir / f"{name}.tif"

            self._notify(progress_cb, f"{name}: requesting GeoTIFF from openEO backend...")

            last_exc: Exception | None = None
            try:
                cube.download(str(out_path), format="GTiff")
            except Exception as exc:
                last_exc = exc

            if not out_path.exists() and hasattr(cube, "execute_batch"):
                try:
                    self._notify(
                        progress_cb,
                        f"{name}: synchronous download failed; falling back to openEO batch job...",
                    )
                    job = cube.execute_batch(out_format="GTiff", title=name)
                    job_id = getattr(job, "job_id", None)
                    if job_id:
                        self._notify(progress_cb, f"{name}: batch job id: {job_id}")

                    if hasattr(job, "start_and_wait"):
                        job.start_and_wait(
                            print=lambda m: self._notify(progress_cb, m),
                            show_error_logs=False,
                        )

                    if hasattr(job, "download_results"):
                        job.download_results(str(tmp_dir))
                    elif hasattr(job, "get_results"):
                        results = job.get_results()
                        if hasattr(results, "download_files"):
                            results.download_files(str(tmp_dir))
                        elif hasattr(results, "download_file"):
                            results.download_file(str(out_path))

                    if out_path.exists():
                        last_exc = None
                except Exception as exc:
                    last_exc = exc

            if not out_path.exists():
                self._notify(
                    progress_cb,
                    f"RGB: no output for {date_str} ({last_exc}); trying next date...",
                )
                continue

            try:
                valid_mask, inside_mask, _, _ = self._coverage_masks(out_path, aoi_geojson)
            except Exception as exc:
                self._notify(progress_cb, f"RGB: failed to assess coverage for {date_str} ({exc}); trying next date...")
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except Exception:
                        pass
                continue
            coverage = self._coverage_ratio(valid_mask, inside_mask)
            if coverage >= self.min_coverage:
                self._notify(
                    progress_cb,
                    f"RGB: selected {date_str} (coverage {coverage:.2%})",
                )
                return out_path, effective_res
            self._notify(progress_cb, f"RGB: {date_str} has nodata in AOI; trying next date...")
            if self.allow_mosaic:
                if combined_valid is None:
                    combined_valid = valid_mask.copy()
                    combined_inside = inside_mask
                else:
                    if combined_inside is not None and inside_mask.shape == combined_inside.shape:
                        combined_valid |= valid_mask
                    else:
                        self._notify(
                            progress_cb,
                            f"RGB: skipping mosaic coverage update for {date_str} due to grid mismatch.",
                        )
                        try:
                            out_path.unlink()
                        except Exception:
                            pass
                        continue
                partial_paths.append(out_path)
                if combined_inside is not None:
                    coverage = self._coverage_ratio(combined_valid, combined_inside)
                    self._notify(
                        progress_cb,
                        f"RGB: combined coverage {coverage:.2%} after {date_str}",
                    )
                    if coverage >= self.min_coverage:
                        break
                if len(partial_paths) >= self.mosaic_max_dates:
                    self._notify(
                        progress_cb,
                        f"RGB: reached mosaic max dates ({self.mosaic_max_dates}); stopping search.",
                    )
                    break
            else:
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except Exception:
                        pass

        if self.allow_mosaic and partial_paths and combined_inside is not None:
            mosaic_path = self._build_mosaic(
                partial_paths,
                aoi_geojson,
                tmp_dir,
                target_label,
                progress_cb=progress_cb,
            )
            if mosaic_path is not None:
                self._notify(progress_cb, f"RGB: mosaic created from {len(partial_paths)} date(s)")
                return mosaic_path, effective_res

        attempted_str = ", ".join(attempted[:10]) + ("..." if len(attempted) > 10 else "")
        raise RuntimeError(
            "RGB download failed: no date within the search window fully covered the AOI. "
            f"Target={target_label}, search_days={self.search_days}, attempted={attempted_str}"
        )
