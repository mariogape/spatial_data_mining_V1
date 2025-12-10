from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from pydantic import BaseModel, Field, root_validator, validator


class StorageConfig(BaseModel):
    kind: str = "local_cog"  # "local_cog" or "gcs_cog"
    output_dir: str | None = None
    bucket: str | None = None
    prefix: str | None = None

    @root_validator
    def validate_storage(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        kind = values.get("kind")
        output_dir = values.get("output_dir")
        bucket = values.get("bucket")
        if kind == "local_cog" and not output_dir:
            raise ValueError("storage.output_dir is required for kind=local_cog")
        if kind == "gcs_cog" and not bucket:
            raise ValueError("storage.bucket is required for kind=gcs_cog")
        return values


class JobConfig(BaseModel):
    name: str
    aoi_path: str
    target_crs: str = Field(regex=r"EPSG:\d+")
    resolution_m: float
    year: int
    season: str
    variables: List[str]
    storage: StorageConfig

    @validator("variables")
    def variables_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("variables list cannot be empty")
        return v


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_job_config(
    job_config_path: str, base_config_path: str = "config/base.yaml"
) -> Tuple[JobConfig, Dict[str, Any]]:
    base_path = Path(base_config_path)
    job_path = Path(job_config_path)
    base_data = _load_yaml(base_path)
    job_data = _load_yaml(job_path)

    base_defaults = base_data.get("defaults", {})
    logging_cfg = base_data.get("logging", {})
    job_section = job_data.get("job", {})

    allowed_crs = base_defaults.get("allowed_crs", [])

    base_fields = {
        k: v
        for k, v in base_defaults.items()
        if k not in ("storage", "allowed_crs")
    }
    merged_storage = {**base_defaults.get("storage", {}), **job_section.get("storage", {})}
    merged_job = {**base_fields, **job_section, "storage": merged_storage}

    job_cfg = JobConfig(**merged_job)
    if allowed_crs and job_cfg.target_crs not in allowed_crs:
        raise ValueError(
            f"target_crs {job_cfg.target_crs} not in allowed list: {allowed_crs}"
        )

    return job_cfg, logging_cfg
