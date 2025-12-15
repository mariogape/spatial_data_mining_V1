# Spatial Data Mining ETL

Config-driven geospatial ETL for extracting indices from Google Earth Engine (and future sources), transforming to analysis-ready rasters, and exporting as Cloud-Optimized GeoTIFFs (COGs) locally or to Google Cloud Storage (GCS).

## Environment setup
- Prereqs: Python 3.10+ recommended; `gcloud` only if you plan to upload to GCS.
- Create/activate venv and install deps:
  - Unix/macOS: `./scripts/setup_env.sh && source .venv/bin/activate`
  - Windows (PowerShell): `python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1; pip install -r requirements.txt`
- Authenticate Google Earth Engine (first time): `earthengine authenticate`
- Optional GCS auth: `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`

## Quickstart (Notebook UI)
- Prereqs: Python 3.10+, `earthengine-api`, `ipywidgets`; optional `gcloud` for GCS uploads.
- Setup: `./scripts/setup_env.sh` then activate the venv (`source .venv/bin/activate` on Unix/macOS, `.\\.venv\\Scripts\\Activate.ps1` on Windows).
- Authenticate GEE once: `earthengine authenticate`.
- Optional GCS auth: `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`.
- Prepare AOI: place your GeoJSON/Shapefile (with CRS) in `data/aoi/`.
- Launch Jupyter/Lab and open `notebooks/pipeline_ui.ipynb`.
- Cell 1: run to ensure paths are set (optionally run `!earthengine authenticate` inside).
- Cell 2: run to display the widget UI (AOI/CRS/resolution/year(s)/season/variables/storage). Click “Run pipeline” to execute.
- Outputs: COGs written to `data/outputs/` (and uploaded to GCS if selected).

## How to use the pipeline (step-by-step)
1) Environment: run `./scripts/setup_env.sh`, activate `.venv`.  
2) Auth: run `earthengine authenticate` (and, if using GCS, ensure ADC by `gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS`).  
3) AOI prep: copy your AOI file to `data/aoi/`.  
4) Launch notebook: `jupyter lab notebooks/pipeline_ui.ipynb` (or open via VS Code).  
5) Cell 1: executes path setup + optional authentication reminder.  
6) Cell 2 (UI):
   - Select AOI (dropdown lists files in `data/aoi/`, or enter a custom path).  
   - Choose CRS (EPSG:3035, 4326, 3857, 25829, 25830, 25831) and pixel resolution.  
   - Set year(s) + season (multi-select supported).  
   - Pick variables: ndvi, ndmi, msi (any combo).  
   - Storage: local COG directory or GCS bucket/prefix.  
   - Click “Run pipeline” to execute (logs/outputs shown below the button).  
7) Verify outputs:
   - Local: `data/outputs/<job>_<var>_<year>_<season>_<crs>.tif`.  
   - GCS: `gs://<bucket>/<prefix>/...` when using `gcs_cog`.  
8) Iterate: adjust UI selections and rerun as needed.

## What you configure (in the notebook UI)
- AOI file: pick from `data/aoi/` or enter a path (CRS auto-detected).
- CRS: EPSG:3035, 4326, 3857, 25829, 25830, 25831.
- Resolution: target pixel size (meters).
- Time: acquisition `year` (or multiple `years`) and descriptive `season`.
- Variables: ND-based indices (ndvi, ndmi, msi).
- Storage: local COG output directory or GCS bucket/prefix.

## Repo structure (key folders)
- `scripts/` – setup script (venv+deps).
- `notebooks/` – `pipeline_ui.ipynb` notebook UI for running jobs.
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
