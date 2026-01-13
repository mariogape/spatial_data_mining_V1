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
    aoi_path: str | None = None
    aoi_paths: List[str] | None = None
    target_crs: str = Field(pattern=r"EPSG:\d+")
    resolution_m: float | None
    clcplus_input_dir: str | None = None
    soilgrids_depth: str | None = None
    soilgrids_stat: str | None = None
    soilgrids_base_url: str | None = None
    soilgrids_tile_index_path: str | None = None
    swi_collection_id: str | None = None
    swi_band: str | None = None
    swi_aggregation: str | None = None
    swi_backend_url: str | None = None
    swi_date: str | None = None
    swi_oidc_provider_id: str | None = None
    rgb_date: str | None = None
    rgb_search_days: int | None = None
    rgb_collection_id: str | None = None
    rgb_bands: str | None = None
    rgb_cloud_cover_max: float | None = None
    rgb_cloud_cover_property: str | None = None
    rgb_backend_url: str | None = None
    rgb_oidc_provider_id: str | None = None
    rgb_stac_url: str | None = None
    rgb_stac_collection_id: str | None = None
    rgb_prefilter: bool | None = None
    gbif_format: str | None = None
    gbif_max_records: int | None = None
    gbif_taxon_key: int | None = None
    gbif_dataset_key: str | None = None
    gbif_basis_of_record: str | None = None
    gbif_occurrence_status: str | None = None
    gbif_kingdom_keys: List[int] | None = None
    gbif_animal_class_keys: List[int] | None = None
    year: int | None = None
    years: List[int] | None = None
    season: str | None = None
    seasons: List[str] | None = None
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

    @field_validator("gbif_format")
    @classmethod
    def gbif_format_allowed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        val = str(value).strip().lower()
        if val in {"geojson", "json"}:
            return "geojson"
        if val in {"gpkg", "geopackage"}:
            return "gpkg"
        raise ValueError("gbif_format must be 'geojson' or 'gpkg'")

    @field_validator("gbif_max_records")
    @classmethod
    def gbif_max_records_positive(cls, value: int | None):
        if value is None:
            return None
        if value <= 0:
            raise ValueError("gbif_max_records must be positive when provided")
        return value

    @field_validator("gbif_taxon_key")
    @classmethod
    def gbif_taxon_key_positive(cls, value: int | None):
        if value is None:
            return None
        if value <= 0:
            raise ValueError("gbif_taxon_key must be positive when provided")
        return value

    @field_validator("gbif_kingdom_keys")
    @classmethod
    def gbif_kingdom_keys_positive(cls, value: List[int] | None):
        if value is None:
            return None
        if not value:
            raise ValueError("gbif_kingdom_keys cannot be empty when provided")
        cleaned: List[int] = []
        for key in value:
            try:
                key_int = int(key)
            except Exception as exc:
                raise ValueError(f"gbif_kingdom_keys must be integers, got {key!r}") from exc
            if key_int <= 0:
                raise ValueError(f"gbif_kingdom_keys must be positive, got {key_int}")
            if key_int not in cleaned:
                cleaned.append(key_int)
        return cleaned

    @field_validator("gbif_animal_class_keys")
    @classmethod
    def gbif_animal_class_keys_positive(cls, value: List[int] | None):
        if value is None:
            return None
        if not value:
            raise ValueError("gbif_animal_class_keys cannot be empty when provided")
        cleaned: List[int] = []
        for key in value:
            try:
                key_int = int(key)
            except Exception as exc:
                raise ValueError(f"gbif_animal_class_keys must be integers, got {key!r}") from exc
            if key_int <= 0:
                raise ValueError(f"gbif_animal_class_keys must be positive, got {key_int}")
            if key_int not in cleaned:
                cleaned.append(key_int)
        return cleaned

    @model_validator(mode="after")
    def normalize_aois(cls, model: "JobConfig") -> "JobConfig":
        paths: List[str] = []
        if model.aoi_paths:
            paths.extend(str(p) for p in model.aoi_paths)
        if model.aoi_path:
            paths.insert(0, str(model.aoi_path))

        if not paths:
            raise ValueError("Provide at least one AOI path (aoi_path or aoi_paths)")

        unique_paths: List[str] = []
        seen: set[str] = set()
        for p in paths:
            if p not in seen:
                unique_paths.append(p)
                seen.add(p)

        model.aoi_path = unique_paths[0]
        model.aoi_paths = unique_paths
        return model

    @model_validator(mode="after")
    def normalize_seasons(cls, model: "JobConfig") -> "JobConfig":
        seasons_combined: List[str] = []
        if model.seasons:
            seasons_combined.extend(str(s) for s in model.seasons)
        if model.season:
            seasons_combined.insert(0, str(model.season))

        if not seasons_combined:
            raise ValueError("Provide at least one season (season or seasons)")

        unique_seasons: List[str] = []
        seen: set[str] = set()
        for s in seasons_combined:
            if s not in seen:
                unique_seasons.append(s)
                seen.add(s)

        model.season = unique_seasons[0]
        model.seasons = unique_seasons
        return model

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

    @model_validator(mode="after")
    def validate_clcplus_inputs(cls, model: "JobConfig") -> "JobConfig":
        vars_lower = [v.lower() for v in model.variables]
        if "clcplus" in vars_lower:
            if not model.clcplus_input_dir:
                raise ValueError("clcplus_input_dir is required when requesting the clcplus variable")
            input_dir = Path(model.clcplus_input_dir)
            if not input_dir.exists() or not input_dir.is_dir():
                raise ValueError(f"clcplus_input_dir must be an existing directory: {input_dir}")
            model.clcplus_input_dir = str(input_dir.resolve())
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
