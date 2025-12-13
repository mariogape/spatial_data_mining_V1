import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

from spatial_data_mining.config import (
    load_job_config,
    load_job_config_from_dict,
)
from spatial_data_mining.load.cog import write_cog
from spatial_data_mining.load.gcs import upload_to_gcs
from spatial_data_mining.utils.aoi import load_aoi, get_aoi_geometries
from spatial_data_mining.utils.logging import setup_logging
from spatial_data_mining.variables.registry import get_variable

ProgressCB = Optional[Callable[[str], None]]


def _notify(cb: ProgressCB, message: str) -> None:
    if cb:
        cb(message)


def _run(job_cfg, logging_cfg, progress_cb: ProgressCB = None) -> List[Dict[str, Any]]:
    setup_logging(logging_cfg)
    logger = logging.getLogger("orchestrator")
    project_root = Path(__file__).resolve().parents[2]

    logger.info("Loaded job: %s", job_cfg.name)
    _notify(progress_cb, f"Loaded job: {job_cfg.name}")

    aoi_gdf = load_aoi(job_cfg.aoi_path)
    geom_wgs84, geom_target = get_aoi_geometries(aoi_gdf, job_cfg.target_crs)
    _notify(progress_cb, "AOI loaded and reprojected.")

    output_dir = Path(job_cfg.storage.output_dir or (project_root / "data/outputs"))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for var_name in job_cfg.variables:
        logger.info("Processing variable: %s", var_name)
        _notify(progress_cb, f"Processing {var_name}...")
        var_def = get_variable(var_name)
        extractor = var_def["extractor"]
        transform_fn = var_def["transform"]

        raw_result = extractor.extract(
            aoi_geojson=geom_wgs84,
            year=job_cfg.year,
            season=job_cfg.season,
            resolution_m=job_cfg.resolution_m,
            temp_dir=output_dir,
            progress_cb=progress_cb,
        )
        # Allow extractors to optionally return (path, effective_resolution_m)
        if isinstance(raw_result, tuple):
            raw_path, effective_res = raw_result
        else:
            raw_path, effective_res = raw_result, job_cfg.resolution_m
        _notify(progress_cb, f"{var_name}: downloaded raw image {raw_path}")

        processed_path = transform_fn(
            src_path=raw_path,
            target_crs=job_cfg.target_crs,
            resolution_m=effective_res,
            aoi_geom_target=geom_target,
        )
        _notify(progress_cb, f"{var_name}: transformed to target CRS/resolution")

        filename = (
            f"{job_cfg.name}_{var_name}_{job_cfg.year}_{job_cfg.season}_"
            f"{job_cfg.target_crs.replace(':', '')}.tif"
        )
        local_output = output_dir / filename
        write_cog(processed_path, local_output)
        _notify(progress_cb, f"{var_name}: wrote COG {local_output}")

        gcs_uri = None
        if job_cfg.storage.kind == "gcs_cog":
            gcs_uri = upload_to_gcs(local_output, job_cfg.storage.bucket, job_cfg.storage.prefix)
            logger.info("Uploaded to GCS: %s", gcs_uri)
            _notify(progress_cb, f"{var_name}: uploaded to {gcs_uri}")

        results.append(
            {
                "variable": var_name,
                "local_path": str(local_output),
                "gcs_uri": gcs_uri,
            }
        )
        logger.info("Finished variable: %s", var_name)
        _notify(progress_cb, f"Finished {var_name}")

    logger.info("Job %s completed. Outputs: %s", job_cfg.name, results)
    _notify(progress_cb, "Job completed.")
    return results


def run_pipeline(config_path: str, progress_cb: ProgressCB = None, **_: Any) -> List[Dict[str, Any]]:
    job_cfg, logging_cfg = load_job_config(config_path)
    return _run(job_cfg, logging_cfg, progress_cb=progress_cb)


def run_pipeline_from_dict(
    job_section: Dict[str, Any], progress_cb: ProgressCB = None, **_: Any
) -> List[Dict[str, Any]]:
    """
    Run pipeline directly from an in-memory job dict (no YAML needed).
    """
    job_cfg, logging_cfg = load_job_config_from_dict(job_section)
    return _run(job_cfg, logging_cfg, progress_cb=progress_cb)
