# Spatial Data Mining ETL

Config-driven geospatial ETL for extracting indices from Google Earth Engine (and future sources), transforming to analysis-ready rasters, and exporting as Cloud-Optimized GeoTIFFs (COGs) locally or to Google Cloud Storage (GCS).

## Quickstart
- Prereqs: Python 3.10+ (recommended), `earthengine-api`, and optional `gcloud` for GCS uploads.
- Setup: `./scripts/setup_env.sh`
- Authenticate GEE (first time): `earthengine authenticate`
- (Optional) GCS auth: `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`
- Configure a job: copy/edit `config/jobs/example.yaml`
- Run: `python scripts/run_pipeline.py --config config/jobs/example.yaml`

## What you configure
- AOI file: `job.aoi_path` (GeoJSON/Shapefile; CRS auto-detected)
- CRS: `job.target_crs` (supported: EPSG:3035, 4326, 3857, 25829, 25830, 25831)
- Resolution: `job.resolution_m`
- Time: `job.year`, `job.season`
- Variables: `job.variables` (start with `ndvi`, `ndmi`, `msi`)
- Storage:
  - `local_cog` → `job.storage.output_dir`
  - `gcs_cog` → `job.storage.bucket` (+ optional `prefix`)

## Repo structure (key folders)
- `scripts/` – setup and runner CLI.
- `config/` – defaults (`base.yaml`) and per-job configs (`jobs/*.yaml`).
- `src/spatial_data_mining/` – library code: orchestrator, extractors, transforms, loaders, utils, variable registry.
- `data/` – AOIs (`aoi/`) and outputs (`outputs/`).
- `legacy/` – existing notebooks/scripts parked here for reference.

## Orchestration flow
1) Load/validate config (merge `base.yaml` + job YAML).  
2) Load AOI, detect CRS, reproject to target CRS.  
3) For each variable: resolve extractor+transform chain → extract from source (GEE for now) → clip/reproject/resample.  
4) Write COG locally; if `storage.kind=gcs_cog`, upload to GCS.  
5) Log progress and summary.

## Adding a variable
1) Implement variable definition in `variables/registry.py` (map name → extractor args + transform chain).  
2) If new source needed, add an extractor module under `extract/` and reference it in the registry.  
3) Optionally add tests in `tests/`.

## Notes
- COG writing uses deflate compression and overviews.
- Keep AOIs small enough for GEE exports; consider tiling for very large areas (future enhancement).
- Seasonal filtering is placeholder; refine date filters as needed.
