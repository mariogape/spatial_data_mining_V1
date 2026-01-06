import logging
import os
import re
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

import rasterio

from spatial_data_mining.config import (
    load_job_config,
    load_job_config_from_dict,
)
from spatial_data_mining.extract.openeo_indices import OpenEOMultiIndexExtractor
from spatial_data_mining.load.cog import write_cog
from spatial_data_mining.load.gcs import upload_to_gcs
from spatial_data_mining.utils.aoi import load_aoi, get_aoi_geometries
from spatial_data_mining.utils.logging import setup_logging
from spatial_data_mining.variables.registry import get_variable
from spatial_data_mining.utils.cancellation import check_cancelled, PipelineCancelled

ProgressCB = Optional[Callable[[str], None]]


def _notify(cb: ProgressCB, message: str) -> None:
    if cb:
        try:
            cb(message)
        except Exception as exc:  # never fail the pipeline due to UI/logging issues
            logging.getLogger(__name__).warning("Progress callback failed: %s", exc)


def _slugify_name(name: str) -> str:
    """Lowercase slug for filenames; collapse non-alnum to underscores."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()
    return slug or "aoi"


def _build_gcs_prefix(
    base_prefix: Optional[str], var_slug: str, year: int, season_slug: str
) -> str:
    parts = [base_prefix, var_slug, str(year), season_slug]
    cleaned = [str(p).strip("/") for p in parts if p is not None and str(p).strip()]
    return "/".join(cleaned)


def _build_gcs_object_name(prefix: Optional[str], filename: str) -> str:
    prefix_clean = str(prefix).strip("/") if prefix else ""
    return f"{prefix_clean}/{filename}" if prefix_clean else filename


def _get_max_concurrent_tasks(progress_cb: ProgressCB = None) -> int:
    """
    Maximum number of concurrent variable runs.

    Copernicus Data Space openEO free tier limits to 2 concurrent synchronous requests and 2 concurrent batch jobs,
    so we clamp at 2 to avoid back-end throttling/errors.
    """
    raw = (
        os.getenv("SDM_MAX_CONCURRENT_TASKS")
        or os.getenv("SDM_MAX_WORKERS")
        or os.getenv("OPENEO_MAX_CONCURRENT_JOBS")
    )
    if raw is None or str(raw).strip() == "":
        return 1
    try:
        val = int(str(raw).strip())
    except ValueError:
        _notify(progress_cb, f"Invalid SDM_MAX_CONCURRENT_TASKS={raw!r}; falling back to 1.")
        return 1
    if val < 1:
        val = 1
    if val > 2:
        _notify(
            progress_cb,
            f"Requested {val} concurrent tasks, but clamping to 2 (openEO free tier limit).",
        )
        val = 2
    return val


def _run(
    job_cfg,
    logging_cfg,
    progress_cb: ProgressCB = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[Dict[str, Any]]:
    setup_logging(logging_cfg)
    logger = logging.getLogger("orchestrator")
    project_root = Path(__file__).resolve().parents[2]

    def _check_stop():
        check_cancelled(should_stop)

    logger.info("Loaded job: %s", job_cfg.name)
    _notify(progress_cb, f"Loaded job: {job_cfg.name}")
    _check_stop()

    output_dir = Path(job_cfg.storage.output_dir or (project_root / "data/outputs"))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_stop()

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    years_val = getattr(job_cfg, "years", None)
    seasons_val = getattr(job_cfg, "seasons", None)
    aois_val = getattr(job_cfg, "aoi_paths", None)

    years = years_val or ([] if getattr(job_cfg, "year", None) is None else [job_cfg.year])
    seasons = seasons_val or ([] if getattr(job_cfg, "season", None) is None else [job_cfg.season])
    aois = aois_val or ([] if getattr(job_cfg, "aoi_path", None) is None else [job_cfg.aoi_path])
    var_slug_map: Dict[str, str] = {}
    max_workers = _get_max_concurrent_tasks(progress_cb=progress_cb)
    if max_workers > 1:
        _notify(progress_cb, f"Parallel mode: running up to {max_workers} tasks concurrently.")

    try:
        for aoi_path in aois:
            _check_stop()
            try:
                aoi_gdf = load_aoi(aoi_path)
                geom_wgs84, geom_target = get_aoi_geometries(aoi_gdf, job_cfg.target_crs)
                aoi_slug = _slugify_name(Path(aoi_path).stem)
                _notify(progress_cb, f"AOI loaded and reprojected: {aoi_slug}")
            except PipelineCancelled:
                raise
            except Exception as exc:
                logger.exception("Skipping AOI %s due to error", aoi_path)
                _notify(progress_cb, f"Skipping AOI {aoi_path}: {exc}")
                errors.append({"aoi_path": str(aoi_path), "error": str(exc)})
                continue

            # Precompute slugs once (used for filenames).
            for var_name in job_cfg.variables:
                if var_name not in var_slug_map:
                    var_slug_map[var_name] = _slugify_name(var_name)

            OPENEO_B11_INDICES = {"ndmi", "msi", "bsi"}
            disable_index_batch = os.getenv("SDM_DISABLE_OPENEO_INDEX_BATCH", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }

            tasks: list[dict[str, Any]] = []
            for season in seasons:
                season_slug = _slugify_name(season)
                for year in years:
                    vars_for_run = list(job_cfg.variables)

                    # Speed-up: batch B11-driven indices (ndmi/msi/bsi) into one openEO job per AOI/year/season.
                    b11_vars = [v for v in vars_for_run if str(v).lower() in OPENEO_B11_INDICES]
                    other_vars = [v for v in vars_for_run if str(v).lower() not in OPENEO_B11_INDICES]

                    if not disable_index_batch and len(b11_vars) >= 2:
                        tasks.append(
                            {
                                "type": "openeo_multi_b11",
                                "var_names": b11_vars,
                                "year": year,
                                "season": season,
                                "season_slug": season_slug,
                            }
                        )
                        # Remove them from the singles list so we don't do extra openEO jobs.
                        other_vars = [v for v in other_vars if str(v).lower() not in OPENEO_B11_INDICES]
                    elif len(b11_vars) == 1:
                        other_vars = other_vars + b11_vars

                    for var_name in other_vars:
                        tasks.append(
                            {
                                "type": "single",
                                "var_name": var_name,
                                "year": year,
                                "season": season,
                                "season_slug": season_slug,
                            }
                        )

            def _build_task_error(task: dict[str, Any], exc: Exception) -> dict[str, Any]:
                year = task.get("year")
                season = task.get("season")
                season_slug = task.get("season_slug") or _slugify_name(season)
                label = task.get("var_name") or "/".join(map(str, task.get("var_names", [])))
                error_entry: dict[str, Any] = {
                    "aoi": aoi_slug,
                    "aoi_path": str(aoi_path),
                    "variable": label,
                    "year": year,
                    "season": season,
                    "error": str(exc),
                }
                if task.get("type") == "single" and task.get("var_name"):
                    var_name = task["var_name"]
                    var_slug = var_slug_map.get(var_name, _slugify_name(var_name))
                    filename = f"{var_slug}_{year}_{season_slug}_{aoi_slug}.tif"
                    error_entry["filename"] = filename
                    if job_cfg.storage.kind == "gcs_cog":
                        gcs_prefix = _build_gcs_prefix(
                            job_cfg.storage.prefix, var_slug, int(year), season_slug
                        )
                        object_name = _build_gcs_object_name(gcs_prefix, filename)
                        error_entry["gcs_uri"] = f"gs://{job_cfg.storage.bucket}/{object_name}"
                    else:
                        error_entry["local_path"] = str(output_dir / filename)
                return error_entry

            def _extract_single_band(src_path: Path, band_index: int, dst_path: Path) -> None:
                dst_path = Path(dst_path)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(src_path) as src:
                    profile = src.profile.copy()
                    profile.update(count=1)
                    with rasterio.open(dst_path, "w", **profile) as dst:
                        for _, window in src.block_windows(1):
                            data = src.read(band_index, window=window)
                            dst.write(data, 1, window=window)

            def _process_single(
                var_name: str, year: int, season: str, season_slug: str
            ) -> tuple[list[dict], list[dict]]:
                var_slug = var_slug_map.get(var_name, _slugify_name(var_name))

                def _progress(message: str) -> None:
                    _notify(progress_cb, f"{var_name} ({year}, {season}) {aoi_slug}: {message}")

                logger.info(
                    "Processing variable %s for year %s season %s (AOI %s)",
                    var_name,
                    year,
                    season,
                    aoi_slug,
                )
                _progress("starting...")
                _check_stop()

                var_def = get_variable(var_name, job_cfg=job_cfg)
                extractor = var_def["extractor"]
                transform_fn = var_def["transform"]

                # On Windows, cleanup can fail intermittently due to lingering file handles.
                # Cleanup errors should not cause a successful variable run to be reported as failed.
                with tempfile.TemporaryDirectory(dir=output_dir, ignore_cleanup_errors=True) as tmp_dir:
                    raw_result = extractor.extract(
                        aoi_geojson=geom_wgs84,
                        year=year,
                        season=season,
                        resolution_m=job_cfg.resolution_m,
                        temp_dir=tmp_dir,
                        progress_cb=_progress,
                        should_stop=should_stop,
                    )
                    # Allow extractors to optionally return (path, effective_resolution_m)
                    if isinstance(raw_result, tuple):
                        raw_path, effective_res = raw_result
                    else:
                        raw_path, effective_res = raw_result, job_cfg.resolution_m
                    _progress(f"downloaded raw image {raw_path}")
                    _check_stop()

                    processed_path = transform_fn(
                        src_path=raw_path,
                        target_crs=job_cfg.target_crs,
                        resolution_m=effective_res,
                        aoi_geom_target=geom_target,
                    )
                    _progress("transformed to target CRS/resolution")
                    _check_stop()

                    filename = f"{var_slug}_{year}_{season_slug}_{aoi_slug}.tif"
                    tmp_suffix = uuid.uuid4().hex
                    if job_cfg.storage.kind == "gcs_cog":
                        local_output = None
                        tmp_output = Path(tmp_dir) / f".{filename}.{tmp_suffix}.tmp.tif"
                    else:
                        local_output = output_dir / filename
                        tmp_output = output_dir / f".{filename}.{tmp_suffix}.tmp.tif"

                    gcs_uri = None
                    try:
                        write_cog(processed_path, tmp_output)
                        _check_stop()

                        if job_cfg.storage.kind == "gcs_cog":
                            _check_stop()
                            gcs_prefix = _build_gcs_prefix(
                                job_cfg.storage.prefix, var_slug, year, season_slug
                            )
                            gcs_uri = upload_to_gcs(
                                tmp_output,
                                job_cfg.storage.bucket,
                                gcs_prefix,
                                object_name=filename,
                            )
                            logger.info("Uploaded to GCS: %s", gcs_uri)
                            _progress(f"uploaded to {gcs_uri}")
                            _check_stop()

                        # Finalize the local output only after all required steps succeed
                        # (e.g., upload for gcs_cog), so incomplete runs don't leave outputs behind.
                        if local_output:
                            tmp_output.replace(local_output)
                    finally:
                        if tmp_output.exists():
                            try:
                                tmp_output.unlink()
                            except Exception:
                                pass

                    if local_output:
                        _progress(f"wrote COG {local_output}")
                    else:
                        _progress("completed without local output (GCS only)")
                    _check_stop()

                    result = {
                        "aoi": aoi_slug,
                        "aoi_path": str(Path(aoi_path).resolve()),
                        "variable": var_name,
                        "year": year,
                        "season": season,
                        "local_path": str(local_output) if local_output else None,
                        "gcs_uri": gcs_uri,
                    }
                    logger.info(
                        "Finished variable %s for year %s season %s (AOI %s)",
                        var_name,
                        year,
                        season,
                        aoi_slug,
                    )
                    _progress("finished")
                    return [result], []

            def _process_openeo_b11_group(
                var_names: list[str], year: int, season: str, season_slug: str
            ) -> tuple[list[dict], list[dict]]:
                # Keep output semantics: 1 COG per variable-year-season.
                # Optimize: one openEO job + one transform for all requested B11-based indices.
                unique_vars: list[str] = []
                seen: set[str] = set()
                for v in var_names:
                    key = str(v).lower()
                    if key not in seen:
                        unique_vars.append(v)
                        seen.add(key)

                var_label = "/".join(str(v).lower() for v in unique_vars)

                def _progress(message: str) -> None:
                    _notify(progress_cb, f"{var_label} ({year}, {season}) {aoi_slug}: {message}")

                logger.info(
                    "Processing openEO batch %s for year %s season %s (AOI %s)",
                    var_label,
                    year,
                    season,
                    aoi_slug,
                )
                _progress("starting (batched openEO extraction)...")
                _check_stop()

                indices = [str(v).upper() for v in unique_vars]
                extractor = OpenEOMultiIndexExtractor(indices)

                # Use the same transform chain as a normal variable run (indices use process_raster_to_target).
                var_def = get_variable(unique_vars[0], job_cfg=job_cfg)
                transform_fn = var_def["transform"]

                results_local: list[dict] = []
                errors_local: list[dict] = []

                with tempfile.TemporaryDirectory(dir=output_dir, ignore_cleanup_errors=True) as tmp_dir:
                    raw_path, effective_res, band_map = extractor.extract_multi(
                        aoi_geojson=geom_wgs84,
                        year=year,
                        season=season,
                        resolution_m=job_cfg.resolution_m,
                        temp_dir=tmp_dir,
                        progress_cb=_progress,
                        should_stop=should_stop,
                    )
                    _progress(f"downloaded raw multi-band image {raw_path}")
                    _check_stop()

                    processed_multi = transform_fn(
                        src_path=raw_path,
                        target_crs=job_cfg.target_crs,
                        resolution_m=effective_res,
                        aoi_geom_target=geom_target,
                    )
                    _progress("transformed multi-band raster to target CRS/resolution")
                    _check_stop()

                    for var_name in unique_vars:
                        key = str(var_name).lower()
                        var_slug = var_slug_map.get(var_name, _slugify_name(var_name))
                        filename = f"{var_slug}_{year}_{season_slug}_{aoi_slug}.tif"
                        band_index = band_map.get(key)
                        if not band_index:
                            error_entry = {
                                "aoi": aoi_slug,
                                "aoi_path": str(aoi_path),
                                "variable": var_name,
                                "year": year,
                                "season": season,
                                "filename": filename,
                                "error": f"Internal error: missing band for {key} in {band_map}",
                            }
                            if job_cfg.storage.kind == "gcs_cog":
                                gcs_prefix = _build_gcs_prefix(
                                    job_cfg.storage.prefix, var_slug, year, season_slug
                                )
                                object_name = _build_gcs_object_name(gcs_prefix, filename)
                                error_entry["gcs_uri"] = (
                                    f"gs://{job_cfg.storage.bucket}/{object_name}"
                                )
                            errors_local.append(error_entry)
                            continue

                        tmp_suffix = uuid.uuid4().hex
                        if job_cfg.storage.kind == "gcs_cog":
                            local_output = None
                            tmp_output = Path(tmp_dir) / f".{filename}.{tmp_suffix}.tmp.tif"
                        else:
                            local_output = output_dir / filename
                            tmp_output = output_dir / f".{filename}.{tmp_suffix}.tmp.tif"

                        band_tif = Path(tmp_dir) / f"{var_slug}_{year}_{season_slug}_{aoi_slug}_band.tif"
                        _extract_single_band(processed_multi, int(band_index), band_tif)
                        _check_stop()

                        gcs_uri = None
                        try:
                            write_cog(band_tif, tmp_output)
                            _check_stop()

                            if job_cfg.storage.kind == "gcs_cog":
                                _check_stop()
                                gcs_prefix = _build_gcs_prefix(
                                    job_cfg.storage.prefix, var_slug, year, season_slug
                                )
                                gcs_uri = upload_to_gcs(
                                    tmp_output,
                                    job_cfg.storage.bucket,
                                    gcs_prefix,
                                    object_name=filename,
                                )
                                logger.info("Uploaded to GCS: %s", gcs_uri)
                                _notify(
                                    progress_cb,
                                    f"{var_name} ({year}, {season}) {aoi_slug}: uploaded to {gcs_uri}",
                                )
                                _check_stop()

                            if local_output:
                                tmp_output.replace(local_output)
                        except Exception as exc:
                            logger.exception(
                                "Failed variable %s for year %s season %s (AOI %s)",
                                var_name,
                                year,
                                season,
                                aoi_slug,
                            )
                            _notify(
                                progress_cb,
                                f"Failed {var_name} ({year}, {season}) for {aoi_slug}: {exc}",
                            )
                            error_entry = {
                                "aoi": aoi_slug,
                                "aoi_path": str(aoi_path),
                                "variable": var_name,
                                "year": year,
                                "season": season,
                                "filename": filename,
                                "error": str(exc),
                            }
                            if job_cfg.storage.kind == "gcs_cog":
                                gcs_prefix = _build_gcs_prefix(
                                    job_cfg.storage.prefix, var_slug, year, season_slug
                                )
                                object_name = _build_gcs_object_name(gcs_prefix, filename)
                                error_entry["gcs_uri"] = (
                                    f"gs://{job_cfg.storage.bucket}/{object_name}"
                                )
                            else:
                                error_entry["local_path"] = str(local_output)
                            errors_local.append(error_entry)
                            continue
                        finally:
                            if tmp_output.exists():
                                try:
                                    tmp_output.unlink()
                                except Exception:
                                    pass

                        if local_output:
                            _notify(
                                progress_cb,
                                f"{var_name} ({year}, {season}) {aoi_slug}: wrote COG {local_output}",
                            )
                        else:
                            _notify(
                                progress_cb,
                                f"{var_name} ({year}, {season}) {aoi_slug}: completed without local output (GCS only)",
                            )

                        results_local.append(
                            {
                                "aoi": aoi_slug,
                                "aoi_path": str(Path(aoi_path).resolve()),
                                "variable": var_name,
                                "year": year,
                                "season": season,
                                "local_path": str(local_output) if local_output else None,
                                "gcs_uri": gcs_uri,
                            }
                        )

                _progress("finished")
                return results_local, errors_local

            def _process_task(task: dict[str, Any]) -> tuple[list[dict], list[dict]]:
                task_type = task.get("type")
                year = int(task["year"])
                season = str(task["season"])
                season_slug = str(task["season_slug"])

                if task_type == "openeo_multi_b11":
                    return _process_openeo_b11_group(task["var_names"], year, season, season_slug)

                if task_type == "single":
                    return _process_single(task["var_name"], year, season, season_slug)

                raise ValueError(f"Unknown task type: {task_type!r}")

            if max_workers <= 1 or len(tasks) <= 1:
                for task in tasks:
                    _check_stop()
                    try:
                        task_results, task_errors = _process_task(task)
                    except PipelineCancelled:
                        logger.info("Pipeline cancelled by user.")
                        _notify(progress_cb, "Pipeline stopped by user.")
                        raise
                    except Exception as exc:
                        year = task.get("year")
                        season = task.get("season")
                        label = task.get("var_name") or "/".join(map(str, task.get("var_names", [])))
                        logger.exception(
                            "Failed task %s for year %s season %s (AOI %s)",
                            label,
                            year,
                            season,
                            aoi_slug,
                        )
                        _notify(progress_cb, f"Failed {label} ({year}, {season}) for {aoi_slug}: {exc}")
                        task_results = []
                        task_errors = [_build_task_error(task, exc)]
                    results.extend(task_results)
                    errors.extend(task_errors)
            else:
                _notify(
                    progress_cb,
                    f"{aoi_slug}: scheduling {len(tasks)} task(s) with max_concurrency={max_workers}",
                )
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {executor.submit(_process_task, task): task for task in tasks}
                    for future in as_completed(future_map):
                        _check_stop()
                        task = future_map[future]
                        try:
                            task_results, task_errors = future.result()
                        except PipelineCancelled:
                            logger.info("Pipeline cancelled by user.")
                            _notify(progress_cb, "Pipeline stopped by user.")
                            raise
                        except Exception as exc:
                            year = task.get("year")
                            season = task.get("season")
                            label = task.get("var_name") or "/".join(map(str, task.get("var_names", [])))
                            logger.exception(
                                "Failed task %s for year %s season %s (AOI %s)",
                                label,
                                year,
                                season,
                                aoi_slug,
                            )
                            _notify(progress_cb, f"Failed {label} ({year}, {season}) for {aoi_slug}: {exc}")
                            task_results = []
                            task_errors = [_build_task_error(task, exc)]
                        results.extend(task_results)
                        errors.extend(task_errors)
    except PipelineCancelled:
        logger.info("Pipeline cancelled by user.")
        _notify(progress_cb, "Pipeline stopped by user.")
        raise

    if errors:
        logger.warning("Job %s completed with %d error(s).", job_cfg.name, len(errors))
        _notify(progress_cb, f"Job completed with {len(errors)} error(s); see logs for details.")
        _notify(progress_cb, "Failed outputs:")
        for err in errors:
            label_parts = []
            if err.get("filename"):
                label_parts.append(str(err["filename"]))
            if err.get("gcs_uri"):
                label_parts.append(str(err["gcs_uri"]))
            elif err.get("local_path"):
                label_parts.append(str(err["local_path"]))
            if not label_parts:
                fallback = err.get("aoi_path") or err.get("aoi") or err.get("variable") or "unknown output"
                label_parts.append(str(fallback))
            error_msg = err.get("error") or "unknown error"
            _notify(progress_cb, f"- {', '.join(label_parts)}: {error_msg}")
    else:
        logger.info("Job %s completed. Outputs: %s", job_cfg.name, results)
        _notify(progress_cb, "Job completed.")
    return results


def run_pipeline(
    config_path: str,
    progress_cb: ProgressCB = None,
    should_stop: Optional[Callable[[], bool]] = None,
    base_config_path: str = "config/base.yaml",
    **_: Any,
) -> List[Dict[str, Any]]:
    job_cfg, logging_cfg = load_job_config(config_path, base_config_path=base_config_path)
    return _run(job_cfg, logging_cfg, progress_cb=progress_cb, should_stop=should_stop)


def run_pipeline_from_dict(
    job_section: Dict[str, Any],
    progress_cb: ProgressCB = None,
    should_stop: Optional[Callable[[], bool]] = None,
    base_config_path: str = "config/base.yaml",
    **_: Any,
) -> List[Dict[str, Any]]:
    """
    Run pipeline directly from an in-memory job dict (no YAML needed).
    """
    job_cfg, logging_cfg = load_job_config_from_dict(
        job_section, base_config_path=base_config_path
    )
    return _run(job_cfg, logging_cfg, progress_cb=progress_cb, should_stop=should_stop)
