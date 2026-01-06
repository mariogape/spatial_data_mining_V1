import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

import requests
from shapely.geometry import shape

from spatial_data_mining.utils.cancellation import check_cancelled
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


def season_date_range(year: int, season: str) -> Tuple[str, str]:
    season_l = str(season).lower()
    if season_l == "winter":
        return (f"{year}-12-01", f"{year + 1}-02-28")
    if season_l in SEASON_RANGES:
        start, end = SEASON_RANGES[season_l]
        return (f"{year}-{start}", f"{year}-{end}")
    return (f"{year}-01-01", f"{year}-12-31")


class OpenEOIndexExtractor:
    """
    Index extractor backed by Copernicus Data Space openEO.

    This is intended to replace the GEE download+tiling path for Sentinel-2 indices
    while keeping the rest of the ETL (transform + COG write) unchanged.
    """

    def __init__(
        self,
        index: str,
        *,
        backend_url: str | None = None,
        collection_id: str | None = None,
        cloud_cover_max: float = 40.0,
        cloud_cover_property: str | None = None,
        verify_ssl: bool | None = None,
    ):
        self.index = index.upper()
        self.backend_url = backend_url or os.getenv(
            "OPENEO_BACKEND_URL", "https://openeo.dataspace.copernicus.eu"
        )
        self.collection_id = collection_id or os.getenv("OPENEO_S2_COLLECTION_ID", "SENTINEL2_L2A")
        self.cloud_cover_max = float(os.getenv("OPENEO_MAX_CLOUD_COVER", str(cloud_cover_max)))
        self.cloud_cover_property = cloud_cover_property or os.getenv(
            "OPENEO_CLOUD_COVER_PROPERTY", "eo:cloud_cover"
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

    def _required_bands(self) -> list[str]:
        # Band naming follows Sentinel-2 convention used by most openEO backends.
        if self.index == "NDVI":
            return ["B08", "B04"]
        if self.index == "NDMI":
            return ["B08", "B11"]
        if self.index == "MSI":
            return ["B11", "B08"]
        if self.index == "BSI":
            return ["B11", "B04", "B08", "B02"]
        raise ValueError(f"Unsupported index for OpenEOIndexExtractor: {self.index}")

    @staticmethod
    def _band(cube, band_name: str):
        if hasattr(cube, "band"):
            return cube.band(band_name)
        if hasattr(cube, "filter_bands"):
            return cube.filter_bands([band_name])
        raise AttributeError("OpenEO DataCube object has no 'band' or 'filter_bands' selector.")

    def _apply_index(self, cube):
        if self.index == "NDVI":
            nir = self._band(cube, "B08")
            red = self._band(cube, "B04")
            return (nir - red) / (nir + red)
        if self.index == "NDMI":
            # NDMI = (NIR - SWIR1) / (NIR + SWIR1)
            # Resample 10m NIR to 20m SWIR grid for a stable 20m output (matches GEE native grid intent).
            nir = self._band(cube, "B08")
            swir = self._band(cube, "B11")
            if hasattr(nir, "resample_cube_spatial"):
                nir = nir.resample_cube_spatial(swir, method="near")
            return (nir - swir) / (nir + swir)
        if self.index == "MSI":
            # MSI = SWIR1 / NIR (typically reported at 20m due to SWIR1)
            swir = self._band(cube, "B11")
            nir = self._band(cube, "B08")
            if hasattr(nir, "resample_cube_spatial"):
                nir = nir.resample_cube_spatial(swir, method="near")
            return swir / nir
        if self.index == "BSI":
            # Bare Soil Index (BSI): ((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))
            # Keep output on the 20m SWIR1 grid by resampling 10m bands to SWIR1.
            swir = self._band(cube, "B11")
            red = self._band(cube, "B04")
            nir = self._band(cube, "B08")
            blue = self._band(cube, "B02")
            if hasattr(red, "resample_cube_spatial"):
                red = red.resample_cube_spatial(swir, method="near")
                nir = nir.resample_cube_spatial(swir, method="near")
                blue = blue.resample_cube_spatial(swir, method="near")
            num = (swir + red) - (nir + blue)
            den = (swir + red) + (nir + blue)
            return num / den
        raise ValueError(f"Unsupported index for OpenEOIndexExtractor: {self.index}")

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

        # If a previous session exists, account calls will succeed; otherwise try to authenticate.
        try:
            conn.describe_account()
            return conn
        except Exception:
            pass

        self._notify(progress_cb, "Authenticating to openEO backend...")

        auth_method = os.getenv("OPENEO_AUTH_METHOD", "").strip().lower()
        oidc_provider_id = os.getenv("OPENEO_OIDC_PROVIDER_ID") or None

        client_id = os.getenv("OPENEO_CLIENT_ID") or None
        client_secret = os.getenv("OPENEO_CLIENT_SECRET") or None
        username = os.getenv("OPENEO_USERNAME") or None
        password = os.getenv("OPENEO_PASSWORD") or None

        # Prefer explicit method selection when provided.
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

        # Auto mode: try client credentials, then basic, then interactive OIDC.
        if client_id and client_secret and hasattr(conn, "authenticate_oidc_client_credentials"):
            _auth_oidc(conn.authenticate_oidc_client_credentials, client_id, client_secret)
            return conn

        if username and password and hasattr(conn, "authenticate_basic"):
            conn.authenticate_basic(username, password)
            return conn

        # Notebook-friendly (device code / browser) auth.
        if hasattr(conn, "authenticate_oidc"):
            _auth_oidc(conn.authenticate_oidc)
            return conn

        raise RuntimeError(
            "Could not authenticate to openEO. Set OPENEO_AUTH_METHOD and credentials env vars, "
            "or ensure your openeo client supports authenticate_oidc()."
        )

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
        Download a seasonal composite index image for the AOI via openEO.
        Returns (local temporary GeoTIFF path, effective resolution in meters).
        """
        check_cancelled(should_stop)

        tmp_dir = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)

        start_date, end_date = season_date_range(year, season)
        today = datetime.now(timezone.utc).date()
        try:
            start_dt = datetime.fromisoformat(start_date).date()
        except ValueError:
            start_dt = None
        if start_dt and start_dt > today:
            raise ValueError(
                f"Requested season starts in the future ({start_date}..{end_date}). "
                "This pipeline defines winter as Dec of the given year through Feb of the next year "
                "(e.g., winter 2025 = 2025-12-01..2026-02-28)."
            )
        geom = shape(aoi_geojson)
        minx, miny, maxx, maxy = geom.bounds
        spatial_extent = {"west": minx, "south": miny, "east": maxx, "north": maxy}

        meta = get_variable_metadata(self.index.lower())
        native_scale = meta.get("native_resolution_m")
        effective_res = resolution_m if resolution_m is not None else native_scale

        conn = self._connect_and_authenticate(progress_cb=progress_cb)
        check_cancelled(should_stop)

        load_kwargs = {
            "spatial_extent": spatial_extent,
            "temporal_extent": [start_date, end_date],
            "bands": self._required_bands(),
        }

        # Match GEE logic: filter scenes by overall cloud cover percentage.
        # Prefer the dedicated max_cloud_cover argument when possible, otherwise fall back to
        # a CollectionProperty filter (openEO Python client does not accept {"lt": ...} dicts).
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
            f"openEO: loading {self.collection_id} ({start_date}..{end_date}) with {cloud_filter_desc}",
        )
        try:
            cube = conn.load_collection(self.collection_id, **load_kwargs)
        except Exception as exc:
            # If the backend/client rejects the cloud filter, retry once without it.
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
                cube = conn.load_collection(
                    self.collection_id,
                    **load_kwargs,
                )
            else:
                raise

        # Match existing GEE logic: median of reflectances, then compute the index.
        # Clip/mask to AOI polygon early to reduce payload.
        try:
            cube = cube.mask_polygon(aoi_geojson)
        except Exception as exc:
            self.logger.warning("openEO mask_polygon failed; continuing without it (%s)", exc)

        cube_median = cube.median_time()
        index_cube = self._apply_index(cube_median)

        name = f"{self.index.lower()}_{year}_{season}_openeo"
        out_path = tmp_dir / f"{name}.tif"

        self._notify(progress_cb, f"{name}: requesting GeoTIFF from openEO backend...")
        check_cancelled(should_stop)

        # Prefer synchronous download for small AOIs; fall back to batch when supported.
        last_exc: Exception | None = None
        try:
            index_cube.download(str(out_path), format="GTiff")
            return out_path, effective_res
        except Exception as exc:
            last_exc = exc

        if hasattr(index_cube, "execute_batch"):
            try:
                self._notify(
                    progress_cb,
                    f"{name}: synchronous download failed; falling back to openEO batch job...",
                )
                job = index_cube.execute_batch(out_format="GTiff", title=name)
                job_id = getattr(job, "job_id", None)
                if job_id:
                    self._notify(progress_cb, f"{name}: batch job id: {job_id}")

                if hasattr(job, "start_and_wait"):
                    try:
                        job.start_and_wait(
                            print=lambda m: self._notify(progress_cb, m),
                            show_error_logs=False,
                        )
                    except Exception as exc:
                        logs_text = None
                        try:
                            logs = job.logs(level="error")
                            if not logs:
                                logs = job.logs(level="warning")
                            if logs:
                                lines = []
                                for entry in list(logs)[:25]:
                                    ts = entry.get("time") or ""
                                    level = str(entry.get("level") or "").upper()
                                    code = entry.get("code") or ""
                                    message = (entry.get("message") or "").replace("\n", " ").strip()
                                    prefix = " ".join(p for p in (ts, level, code) if p).strip()
                                    lines.append(f"{prefix}: {message}" if prefix else message)
                                if len(logs) > 25:
                                    lines.append(f"... ({len(logs) - 25} more log entries)")
                                logs_text = "\n".join(lines)
                        except Exception:
                            logs_text = None

                        base = str(exc)
                        details = f"openEO batch job failed ({job_id}): {base}"
                        if logs_text:
                            details = f"{details}\nError logs:\n{logs_text}"
                        details = (
                            f"{details}\nFull logs: in Python `connection.job({job_id!r}).logs()` "
                            "or in an openEO web editor."
                        )
                        raise RuntimeError(details) from exc
                # Download results to temp dir; try to rename the first GeoTIFF to out_path.
                if hasattr(job, "download_results"):
                    job.download_results(str(tmp_dir))
                elif hasattr(job, "get_results"):
                    results = job.get_results()
                    if hasattr(results, "download_files"):
                        results.download_files(str(tmp_dir))
                    elif hasattr(results, "download_file"):
                        results.download_file(str(out_path))

                if out_path.exists():
                    return out_path, effective_res

                tif_candidates = sorted(tmp_dir.glob("*.tif")) + sorted(tmp_dir.glob("*.tiff"))
                if tif_candidates:
                    tif_candidates[0].replace(out_path)
                    return out_path, effective_res
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(f"openEO {self.index} download failed: {last_exc}") from last_exc


class OpenEOFVCExtractor(OpenEOIndexExtractor):
    """
    Extractor for Fractional Vegetation Cover (FVC) derived from NDVI via openEO.

    This downloads a seasonal NDVI median composite (median across NDVI images) and leaves
    the FVC computation to the local transform step.
    """

    def __init__(
        self,
        *,
        backend_url: str | None = None,
        collection_id: str | None = None,
        cloud_cover_max: float = 40.0,
        cloud_cover_property: str | None = None,
        verify_ssl: bool | None = None,
    ):
        super().__init__(
            "NDVI",
            backend_url=backend_url,
            collection_id=collection_id,
            cloud_cover_max=cloud_cover_max,
            cloud_cover_property=cloud_cover_property,
            verify_ssl=verify_ssl,
        )

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
        Download a seasonal NDVI median composite for the AOI via openEO.
        Returns (local temporary GeoTIFF path, effective resolution in meters).
        """
        check_cancelled(should_stop)

        tmp_dir = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)

        start_date, end_date = season_date_range(year, season)
        today = datetime.now(timezone.utc).date()
        try:
            start_dt = datetime.fromisoformat(start_date).date()
        except ValueError:
            start_dt = None
        if start_dt and start_dt > today:
            raise ValueError(
                f"Requested season starts in the future ({start_date}..{end_date}). "
                "This pipeline defines winter as Dec of the given year through Feb of the next year "
                "(e.g., winter 2025 = 2025-12-01..2026-02-28)."
            )
        geom = shape(aoi_geojson)
        minx, miny, maxx, maxy = geom.bounds
        spatial_extent = {"west": minx, "south": miny, "east": maxx, "north": maxy}

        meta = get_variable_metadata("fvc")
        native_scale = meta.get("native_resolution_m")
        effective_res = resolution_m if resolution_m is not None else native_scale

        conn = self._connect_and_authenticate(progress_cb=progress_cb)
        check_cancelled(should_stop)

        load_kwargs = {
            "spatial_extent": spatial_extent,
            "temporal_extent": [start_date, end_date],
            "bands": self._required_bands(),
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
            f"openEO: loading {self.collection_id} ({start_date}..{end_date}) with {cloud_filter_desc}",
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

        ndvi_cube = self._apply_index(cube)
        ndvi_median = ndvi_cube.median_time()

        name = f"fvc_{year}_{season}_ndvi_openeo"
        out_path = tmp_dir / f"{name}.tif"

        self._notify(progress_cb, f"{name}: requesting GeoTIFF from openEO backend...")
        check_cancelled(should_stop)

        last_exc: Exception | None = None
        try:
            ndvi_median.download(str(out_path), format="GTiff")
            return out_path, effective_res
        except Exception as exc:
            last_exc = exc

        if hasattr(ndvi_median, "execute_batch"):
            try:
                self._notify(
                    progress_cb,
                    f"{name}: synchronous download failed; falling back to openEO batch job...",
                )
                job = ndvi_median.execute_batch(out_format="GTiff", title=name)
                job_id = getattr(job, "job_id", None)
                if job_id:
                    self._notify(progress_cb, f"{name}: batch job id: {job_id}")

                if hasattr(job, "start_and_wait"):
                    try:
                        job.start_and_wait(
                            print=lambda m: self._notify(progress_cb, m),
                            show_error_logs=False,
                        )
                    except Exception as exc:
                        logs_text = None
                        try:
                            logs = job.logs(level="error")
                            if not logs:
                                logs = job.logs(level="warning")
                            if logs:
                                lines = []
                                for entry in list(logs)[:25]:
                                    ts = entry.get("time") or ""
                                    level = str(entry.get("level") or "").upper()
                                    code = entry.get("code") or ""
                                    message = (entry.get("message") or "").replace("\n", " ").strip()
                                    prefix = " ".join(p for p in (ts, level, code) if p).strip()
                                    lines.append(f"{prefix}: {message}" if prefix else message)
                                if len(logs) > 25:
                                    lines.append(f"... ({len(logs) - 25} more log entries)")
                                logs_text = "\n".join(lines)
                        except Exception:
                            logs_text = None

                        base = str(exc)
                        details = f"openEO batch job failed ({job_id}): {base}"
                        if logs_text:
                            details = f"{details}\nError logs:\n{logs_text}"
                        details = (
                            f"{details}\nFull logs: in Python `connection.job({job_id!r}).logs()` "
                            "or in an openEO web editor."
                        )
                        raise RuntimeError(details) from exc

                if hasattr(job, "download_results"):
                    job.download_results(str(tmp_dir))
                elif hasattr(job, "get_results"):
                    results = job.get_results()
                    if hasattr(results, "download_files"):
                        results.download_files(str(tmp_dir))
                    elif hasattr(results, "download_file"):
                        results.download_file(str(out_path))

                if out_path.exists():
                    return out_path, effective_res

                tif_candidates = sorted(tmp_dir.glob("*.tif")) + sorted(tmp_dir.glob("*.tiff"))
                if tif_candidates:
                    tif_candidates[0].replace(out_path)
                    return out_path, effective_res
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(f"openEO FVC download failed: {last_exc}") from last_exc


class OpenEOMultiIndexExtractor:
    """
    Compute multiple Sentinel-2 indices in a single openEO job and download as a multi-band GeoTIFF.

    This is primarily used to speed up runs where multiple indices are requested for the same AOI/year/season
    by reducing the number of openEO jobs (and doing only one reprojection/resample locally for the batch).
    """

    def __init__(
        self,
        indices: list[str],
        *,
        backend_url: str | None = None,
        collection_id: str | None = None,
        cloud_cover_max: float = 40.0,
        cloud_cover_property: str | None = None,
        verify_ssl: bool | None = None,
    ):
        if not indices:
            raise ValueError("OpenEOMultiIndexExtractor requires at least one index.")
        self.indices = [str(i).upper() for i in indices]
        self._base = OpenEOIndexExtractor(
            self.indices[0],
            backend_url=backend_url,
            collection_id=collection_id,
            cloud_cover_max=cloud_cover_max,
            cloud_cover_property=cloud_cover_property,
            verify_ssl=verify_ssl,
        )
        self.logger = logging.getLogger(__name__)

    def _required_bands(self) -> list[str]:
        required: set[str] = set()
        for idx in self.indices:
            required.update(OpenEOIndexExtractor(idx)._required_bands())
        return sorted(required)

    @staticmethod
    def _effective_resolution(indices: list[str], resolution_m: float | None) -> float | None:
        if resolution_m is not None:
            return resolution_m
        native_scales: list[float] = []
        for idx in indices:
            meta = get_variable_metadata(idx.lower())
            val = meta.get("native_resolution_m")
            if val:
                native_scales.append(float(val))
        return max(native_scales) if native_scales else None

    def _apply_index(
        self,
        cube,
        index: str,
        target_grid_cube=None,
        assume_common_grid: bool = False,
    ):
        idx = str(index).upper()
        band = OpenEOIndexExtractor._band
        if idx == "NDVI":
            nir = band(cube, "B08")
            red = band(cube, "B04")
            if (
                not assume_common_grid
                and target_grid_cube is not None
                and hasattr(nir, "resample_cube_spatial")
            ):
                nir = nir.resample_cube_spatial(target_grid_cube, method="near")
                red = red.resample_cube_spatial(target_grid_cube, method="near")
            return (nir - red) / (nir + red)
        if idx == "NDMI":
            nir = band(cube, "B08")
            swir = band(cube, "B11")
            if not assume_common_grid and hasattr(nir, "resample_cube_spatial"):
                nir = nir.resample_cube_spatial(swir, method="near")
            return (nir - swir) / (nir + swir)
        if idx == "MSI":
            swir = band(cube, "B11")
            nir = band(cube, "B08")
            if not assume_common_grid and hasattr(nir, "resample_cube_spatial"):
                nir = nir.resample_cube_spatial(swir, method="near")
            return swir / nir
        if idx == "BSI":
            swir = band(cube, "B11")
            red = band(cube, "B04")
            nir = band(cube, "B08")
            blue = band(cube, "B02")
            if not assume_common_grid and hasattr(red, "resample_cube_spatial"):
                red = red.resample_cube_spatial(swir, method="near")
                nir = nir.resample_cube_spatial(swir, method="near")
                blue = blue.resample_cube_spatial(swir, method="near")
            num = (swir + red) - (nir + blue)
            den = (swir + red) + (nir + blue)
            return num / den
        raise ValueError(f"Unsupported index for OpenEOMultiIndexExtractor: {idx}")

    def extract_multi(
        self,
        aoi_geojson: dict,
        year: int,
        season: str,
        resolution_m: float | None,
        temp_dir: str | Path | None = None,
        progress_cb: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Path, float | None, dict[str, int]]:
        """
        Download a seasonal composite multi-band image for the requested indices.
        Returns (local GeoTIFF path, effective resolution in meters, band_index_map).
        """
        check_cancelled(should_stop)

        tmp_dir = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)

        start_date, end_date = season_date_range(year, season)
        today = datetime.now(timezone.utc).date()
        try:
            start_dt = datetime.fromisoformat(start_date).date()
        except ValueError:
            start_dt = None
        if start_dt and start_dt > today:
            raise ValueError(
                f"Requested season starts in the future ({start_date}..{end_date}). "
                "This pipeline defines winter as Dec of the given year through Feb of the next year "
                "(e.g., winter 2025 = 2025-12-01..2026-02-28)."
            )

        geom = shape(aoi_geojson)
        minx, miny, maxx, maxy = geom.bounds
        spatial_extent = {"west": minx, "south": miny, "east": maxx, "north": maxy}

        effective_res = self._effective_resolution(self.indices, resolution_m)

        conn = self._base._connect_and_authenticate(progress_cb=progress_cb)
        check_cancelled(should_stop)

        load_kwargs = {
            "spatial_extent": spatial_extent,
            "temporal_extent": [start_date, end_date],
            "bands": self._required_bands(),
        }

        cloud_filter_desc = "no cloud filter"
        if self._base.cloud_cover_max is not None:
            if not self._base.cloud_cover_property or self._base.cloud_cover_property == "eo:cloud_cover":
                load_kwargs["max_cloud_cover"] = self._base.cloud_cover_max
                cloud_filter_desc = f"max_cloud_cover<{self._base.cloud_cover_max}"
            else:
                try:
                    import openeo  # type: ignore

                    load_kwargs["properties"] = [
                        openeo.collection_property(self._base.cloud_cover_property)
                        < self._base.cloud_cover_max
                    ]
                    cloud_filter_desc = f"{self._base.cloud_cover_property}<{self._base.cloud_cover_max}"
                except Exception as exc:
                    self.logger.warning(
                        "Could not build cloud-cover property filter for %s (%s); continuing without it.",
                        self._base.cloud_cover_property,
                        exc,
                    )

        self._base._notify(
            progress_cb,
            f"openEO: loading {self._base.collection_id} ({start_date}..{end_date}) with {cloud_filter_desc}",
        )
        try:
            cube = conn.load_collection(self._base.collection_id, **load_kwargs)
        except Exception as exc:
            if "max_cloud_cover" in load_kwargs or "properties" in load_kwargs:
                self.logger.warning(
                    "openEO load_collection failed with cloud filter (%s); retrying without it.",
                    exc,
                )
                self._base._notify(
                    progress_cb,
                    "openEO: warning: cloud-cover filter failed; retrying without it.",
                )
                load_kwargs.pop("max_cloud_cover", None)
                load_kwargs.pop("properties", None)
                cube = conn.load_collection(self._base.collection_id, **load_kwargs)
            else:
                raise

        try:
            cube = cube.mask_polygon(aoi_geojson)
        except Exception as exc:
            self.logger.warning("openEO mask_polygon failed; continuing without it (%s)", exc)

        cube_median = cube.median_time()
        target_grid = None
        cube_for_math = cube_median
        if "B11" in load_kwargs["bands"] and hasattr(cube_median, "resample_cube_spatial"):
            try:
                target_grid = OpenEOIndexExtractor._band(cube_median, "B11")
                cube_for_math = cube_median.resample_cube_spatial(target_grid, method="near")
                self._base._notify(
                    progress_cb,
                    "openEO: aligning bands to B11 grid for multi-index math",
                )
            except Exception as exc:
                self.logger.warning("openEO resample_cube_spatial failed; using native grids (%s)", exc)
                cube_for_math = cube_median

        merged = None
        band_map: dict[str, int] = {}
        for idx_pos, idx in enumerate(self.indices, start=1):
            idx_lower = idx.lower()
            band_map[idx_lower] = idx_pos
            idx_cube = self._apply_index(
                cube_for_math,
                idx,
                target_grid_cube=target_grid,
                assume_common_grid=cube_for_math is not cube_median,
            )
            # Ensure the result has a band dimension so multiple indices can be saved as multi-band GeoTIFF.
            idx_cube = idx_cube.add_dimension("bands", idx_lower, type="bands")
            merged = idx_cube if merged is None else merged.merge_cubes(idx_cube)

        name = f"indices_{'_'.join([i.lower() for i in self.indices])}_{year}_{season}_openeo"
        out_path = tmp_dir / f"{name}.tif"

        self._base._notify(progress_cb, f"{name}: requesting GeoTIFF from openEO backend...")
        check_cancelled(should_stop)

        last_exc: Exception | None = None
        try:
            merged.download(str(out_path), format="GTiff")
            return out_path, effective_res, band_map
        except Exception as exc:
            last_exc = exc

        if hasattr(merged, "execute_batch"):
            try:
                self._base._notify(
                    progress_cb,
                    f"{name}: synchronous download failed; falling back to openEO batch job...",
                )
                job = merged.execute_batch(out_format="GTiff", title=name)
                job_id = getattr(job, "job_id", None)
                if job_id:
                    self._base._notify(progress_cb, f"{name}: batch job id: {job_id}")
                if hasattr(job, "start_and_wait"):
                    job.start_and_wait(
                        print=lambda m: self._base._notify(progress_cb, m),
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
                    return out_path, effective_res, band_map

                tif_candidates = sorted(tmp_dir.glob("*.tif")) + sorted(tmp_dir.glob("*.tiff"))
                if tif_candidates:
                    tif_candidates[0].replace(out_path)
                    return out_path, effective_res, band_map
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(f"openEO multi-index download failed: {last_exc}") from last_exc
