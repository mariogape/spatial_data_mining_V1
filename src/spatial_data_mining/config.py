from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class StorageConfig(BaseModel):
    kind: str = "local_cog"  # "local_cog" or "gcs_cog"
    output_dir: str | None = None
    bucket: str | None = None
    prefix: str | None = None

    @model_validator(mode="after")
    def validate_storage(cls, model: "StorageConfig") -> "StorageConfig":
        if model.kind == "local_cog" and not model.output_dir:
            raise ValueError("storage.output_dir is required for kind=local_cog")
        if model.kind == "gcs_cog" and not model.bucket:
            raise ValueError("storage.bucket is required for kind=gcs_cog")
        return model


class JobConfig(BaseModel):
    name: str
    aoi_path: str
    target_crs: str = Field(pattern=r"EPSG:\d+")
    resolution_m: float | None
    year: int | None = None
    years: List[int] | None = None
    season: str
    variables: List[str]
    storage: StorageConfig

    @field_validator("variables")
    @classmethod
    def variables_not_empty(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("variables list cannot be empty")
        return value

    @field_validator("resolution_m")
    @classmethod
    def resolution_positive_or_none(cls, value: float | None):
        if value is None:
            return None
        if value <= 0:
            raise ValueError("resolution_m must be positive when provided")
        return value

    @model_validator(mode="after")
    def normalize_years(cls, model: "JobConfig") -> "JobConfig":
        years_combined: List[int] = []
        if model.years:
            years_combined.extend(int(y) for y in model.years)
        if model.year is not None:
            years_combined.insert(0, int(model.year))

        if not years_combined:
            raise ValueError("Provide at least one year (year or years)")
        if any(y <= 0 for y in years_combined):
            raise ValueError("Years must be positive integers")

        unique_years: List[int] = []
        seen: set[int] = set()
        for y in years_combined:
            if y not in seen:
                unique_years.append(y)
                seen.add(y)

        model.year = unique_years[0]
        model.years = unique_years
        return model


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_job(
    base_defaults: Dict[str, Any], job_section: Dict[str, Any]
) -> Tuple[JobConfig, List[str]]:
    allowed_crs = base_defaults.get("allowed_crs", [])
    base_fields = {
        k: v for k, v in base_defaults.items() if k not in ("storage", "allowed_crs")
    }
    merged_storage = {**base_defaults.get("storage", {}), **job_section.get("storage", {})}
    merged_job = {**base_fields, **job_section, "storage": merged_storage}
    job_cfg = JobConfig(**merged_job)
    if allowed_crs and job_cfg.target_crs not in allowed_crs:
        raise ValueError(
            f"target_crs {job_cfg.target_crs} not in allowed list: {allowed_crs}"
        )
    return job_cfg, allowed_crs


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

    job_cfg, _ = _merge_job(base_defaults, job_section)
    return job_cfg, logging_cfg


def load_job_config_from_dict(
    job_section: Dict[str, Any], base_config_path: str = "config/base.yaml"
) -> Tuple[JobConfig, Dict[str, Any]]:
    """
    Same as load_job_config, but takes a dict instead of a YAML path.
    Useful for UI-driven runs where config is built on the fly.
    """
    base_path = Path(base_config_path)
    base_data = _load_yaml(base_path) if base_path.exists() else {}
    base_defaults = base_data.get("defaults", {})
    logging_cfg = base_data.get("logging", {})
    job_cfg, _ = _merge_job(base_defaults, job_section)
    return job_cfg, logging_cfg
