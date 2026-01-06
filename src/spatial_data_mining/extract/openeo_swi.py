import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

import requests
from shapely.geometry import shape

from spatial_data_mining.extract.openeo_indices import season_date_range
from spatial_data_mining.utils.cancellation import check_cancelled
from spatial_data_mining.variables.metadata import get_variable_metadata


class OpenEOSoilWaterIndexExtractor:
    """
    Extractor for Copernicus Land Monitoring Service Soil Water Index products via openEO.
    """

    def __init__(
        self,
        *,
        collection_id: str | None = None,
        band: str | None = None,
        temporal_agg: str | None = None,
        swi_date: str | None = None,
        oidc_provider_id: str | None = None,
        backend_url: str | None = None,
        verify_ssl: bool | None = None,
    ):
        default_collection = "CGLS_SWI_V1_EUROPE"
        default_band = "SWI_100"
        default_backend = "https://openeo.vito.be"

        self.collection_id = (
            collection_id
            or os.getenv("OPENEO_SWI_COLLECTION_ID")
            or default_collection
        )
        self.band = band or os.getenv("OPENEO_SWI_BAND") or default_band
        self.temporal_agg = self._normalize_temporal_agg(
            temporal_agg or os.getenv("OPENEO_SWI_AGGREGATION", "none")
        )
        self.swi_date = self._normalize_date(
            swi_date or os.getenv("OPENEO_SWI_DATE")
        )
        self.backend_url = (
            backend_url
            or os.getenv("OPENEO_SWI_BACKEND_URL")
            or default_backend
        )
        self.oidc_provider_id = (
            oidc_provider_id
            or os.getenv("OPENEO_SWI_OIDC_PROVIDER_ID")
            or os.getenv("OPENEO_OIDC_PROVIDER_ID")
        )
        if not self.oidc_provider_id and "openeo.vito.be" in str(self.backend_url).lower():
            self.oidc_provider_id = "terrascope"
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
    def _normalize_temporal_agg(value: str | None) -> str | None:
        if value is None:
            return None
        val = str(value).strip().lower()
        if val in {"", "none", "raw", "skip"}:
            return None
        if val not in {"mean", "median"}:
            raise ValueError(f"Unsupported SWI temporal aggregation: {value!r}")
        return val

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
                f"SWI date must be ISO format YYYY-MM-DD (got {text!r})."
            ) from exc
        return text

    @staticmethod
    def _default_date_for_season(season: str, year: int) -> str | None:
        if str(season).lower() == "static":
            return None
        try:
            start_str, end_str = season_date_range(int(year), season)
            start_dt = datetime.fromisoformat(start_str).date()
            end_dt = datetime.fromisoformat(end_str).date()
            return (start_dt + (end_dt - start_dt) // 2).isoformat()
        except Exception:
            return None

    @staticmethod
    def _parse_bands(value: str | None) -> list[str] | None:
        if value is None:
            return None
        parts = [p.strip() for p in str(value).split(",")]
        bands = [p for p in parts if p]
        return bands or None

    @staticmethod
    def _select_bands(cube, bands: list[str] | None):
        if not bands:
            return cube
        if len(bands) == 1 and hasattr(cube, "band"):
            return cube.band(bands[0])
        if hasattr(cube, "filter_bands"):
            return cube.filter_bands(bands)
        return cube

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

    def _apply_temporal_agg(self, cube):
        if self.temporal_agg is None:
            return cube
        if self.temporal_agg == "mean" and hasattr(cube, "mean_time"):
            return cube.mean_time()
        if self.temporal_agg == "median" and hasattr(cube, "median_time"):
            return cube.median_time()
        if hasattr(cube, "reduce_dimension"):
            for dimension in ("t", "time"):
                try:
                    return cube.reduce_dimension(dimension=dimension, reducer=self.temporal_agg)
                except Exception:
                    continue
        raise ValueError(
            "openEO client does not support the requested temporal aggregation "
            f"({self.temporal_agg!r})."
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
        Download a seasonal composite Soil Water Index image for the AOI via openEO.
        Returns (local temporary GeoTIFF path, effective resolution in meters).
        """
        check_cancelled(should_stop)

        tmp_dir = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)

        season_l = str(season).lower()
        start_date = end_date = None
        if self.swi_date:
            if self.temporal_agg is not None:
                self.logger.info(
                    "SWI date provided; ignoring temporal aggregation (%s).", self.temporal_agg
                )
            start_date = end_date = self.swi_date
            if season_l != "static":
                season_start, season_end = season_date_range(year, season)
                date_dt = datetime.fromisoformat(self.swi_date).date()
                start_dt = datetime.fromisoformat(season_start).date()
                end_dt = datetime.fromisoformat(season_end).date()
                if not (start_dt <= date_dt <= end_dt):
                    raise ValueError(
                        f"SWI date {self.swi_date} is outside the {season} {year} range "
                        f"({season_start}..{season_end})."
                    )
        else:
            if self.temporal_agg is None:
                default_date = self._default_date_for_season(season, year)
                if default_date is None:
                    raise ValueError(
                        "SWI requires an exact date (swi_date / OPENEO_SWI_DATE) "
                        "when no temporal aggregation is configured."
                    )
                start_date = end_date = default_date
                self._notify(
                    progress_cb,
                    f"SWI: using default mid-season date {default_date} for {season} {year}.",
                )
            else:
                if season_l != "static":
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

        meta = get_variable_metadata("swi")
        native_scale = meta.get("native_resolution_m")
        effective_res = resolution_m if resolution_m is not None else native_scale

        conn = self._connect_and_authenticate(progress_cb=progress_cb)
        check_cancelled(should_stop)

        load_kwargs: dict[str, object] = {"spatial_extent": spatial_extent}
        if start_date and end_date:
            load_kwargs["temporal_extent"] = [start_date, end_date]
        bands = self._parse_bands(self.band)
        if bands:
            load_kwargs["bands"] = bands

        if start_date and end_date:
            temporal_desc = f"{start_date}..{end_date}"
        elif season_l == "static":
            temporal_desc = "static (no temporal filter)"
        else:
            temporal_desc = "unknown temporal window"
        band_desc = f" bands={bands}" if bands else ""
        self._notify(
            progress_cb,
            f"openEO: loading {self.collection_id} ({temporal_desc}){band_desc}",
        )
        try:
            cube = conn.load_collection(self.collection_id, **load_kwargs)
        except Exception as exc:
            if bands:
                self.logger.warning(
                    "openEO load_collection failed with band filter (%s); retrying without it.",
                    exc,
                )
                self._notify(
                    progress_cb,
                    "openEO: warning: band filter failed; retrying without it.",
                )
                load_kwargs.pop("bands", None)
                cube = conn.load_collection(self.collection_id, **load_kwargs)
            else:
                raise
        if bands:
            try:
                cube = self._select_bands(cube, bands)
            except Exception as exc:
                self.logger.warning(
                    "openEO band selection failed; continuing without it (%s)", exc
                )

        try:
            cube = cube.mask_polygon(aoi_geojson)
        except Exception as exc:
            self.logger.warning("openEO mask_polygon failed; continuing without it (%s)", exc)

        if start_date and end_date and self.temporal_agg is not None:
            cube = self._apply_temporal_agg(cube)

        name = f"swi_{year}_{season}_openeo"
        out_path = tmp_dir / f"{name}.tif"

        self._notify(progress_cb, f"{name}: requesting GeoTIFF from openEO backend...")
        check_cancelled(should_stop)

        last_exc: Exception | None = None
        try:
            cube.download(str(out_path), format="GTiff")
            return out_path, effective_res
        except Exception as exc:
            last_exc = exc

        if hasattr(cube, "execute_batch"):
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
                    return out_path, effective_res

                tif_candidates = sorted(tmp_dir.glob("*.tif")) + sorted(tmp_dir.glob("*.tiff"))
                if tif_candidates:
                    tif_candidates[0].replace(out_path)
                    return out_path, effective_res
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(f"openEO SWI download failed: {last_exc}") from last_exc
