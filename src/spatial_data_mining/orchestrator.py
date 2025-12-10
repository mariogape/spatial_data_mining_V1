import logging
from pathlib import Path
from typing import List, Dict, Any

from spatial_data_mining.config import load_job_config
from spatial_data_mining.load.cog import write_cog
from spatial_data_mining.load.gcs import upload_to_gcs
from spatial_data_mining.utils.aoi import load_aoi, get_aoi_geometries
from spatial_data_mining.utils.logging import setup_logging
from spatial_data_mining.variables.registry import get_variable


def run_pipeline(config_path: str) -> List[Dict[str, Any]]:
    job_cfg, logging_cfg = load_job_config(config_path)
    setup_logging(logging_cfg)
    logger = logging.getLogger("orchestrator")

    logger.info("Loaded job: %s", job_cfg.name)
    aoi_gdf = load_aoi(job_cfg.aoi_path)
    geom_wgs84, geom_target = get_aoi_geometries(aoi_gdf, job_cfg.target_crs)

    output_dir = Path(job_cfg.storage.output_dir or "data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for var_name in job_cfg.variables:
        logger.info("Processing variable: %s", var_name)
        var_def = get_variable(var_name)
        extractor = var_def["extractor"]
        transform_fn = var_def["transform"]

        raw_path = extractor.extract(
            aoi_geojson=geom_wgs84,
            year=job_cfg.year,
            season=job_cfg.season,
            resolution_m=job_cfg.resolution_m,
        )

        processed_path = transform_fn(
            src_path=raw_path,
            target_crs=job_cfg.target_crs,
            resolution_m=job_cfg.resolution_m,
            aoi_geom_target=geom_target,
        )

        filename = (
            f"{job_cfg.name}_{var_name}_{job_cfg.year}_{job_cfg.season}_"
            f"{job_cfg.target_crs.replace(':', '')}.tif"
        )
        local_output = output_dir / filename
        write_cog(processed_path, local_output)

        gcs_uri = None
        if job_cfg.storage.kind == "gcs_cog":
            gcs_uri = upload_to_gcs(local_output, job_cfg.storage.bucket, job_cfg.storage.prefix)
            logger.info("Uploaded to GCS: %s", gcs_uri)

        results.append(
            {
                "variable": var_name,
                "local_path": str(local_output),
                "gcs_uri": gcs_uri,
            }
        )
        logger.info("Finished variable: %s", var_name)

    logger.info("Job %s completed. Outputs: %s", job_cfg.name, results)
    return results
