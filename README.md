# Spatial Data Mining ETL

Config-driven geospatial ETL for extracting indices from Google Earth Engine (GEE) and Copernicus Data Space openEO, transforming to analysis-ready rasters, and exporting as Cloud-Optimized GeoTIFFs (COGs) locally or to Google Cloud Storage (GCS).

## What this repo provides
- Pipeline orchestrator for AOI-based, multi-year/season extraction.
- Extractors for Sentinel-2 indices (openEO) and AlphaEarth embeddings (GEE).
- Transform pipeline for reprojection, resampling, and AOI clipping.
- COG writer with optional GCS upload.
- Notebook UI for non-YAML usage.

## Requirements
- Python 3.10+ (recommended).
- JupyterLab/Notebook if using the notebook UI.
- `gcloud` only if uploading to GCS.
- Earth Engine auth (via `earthengine authenticate`) only if using `alpha_earth`.

## Setup
1) Create and activate a virtual environment.

Unix/macOS:
```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Optional: install JupyterLab if you plan to use the notebook UI.
```bash
pip install jupyterlab
```

3) Authenticate as needed:
- GEE (for `alpha_earth`):
```bash
earthengine authenticate
```
- GCS uploads (for `gcs_cog` storage):
```bash
gcloud auth application-default login
```
  Or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`.

## Quickstart (Notebook UI)
1) Place your AOI file (GeoJSON or Shapefile with CRS) in `data/aoi/`.
2) Launch JupyterLab and open `notebooks/pipeline_ui.ipynb`.
3) Run Cell 1 to set paths and review auth notes.
4) Run Cell 2 to open the UI, choose AOI/CRS/resolution/year(s)/season/variables/storage, and click Run.
5) Outputs are written to `data/outputs/` and optionally uploaded to GCS.

If the widgets do not render, install `ipywidgets` and restart Jupyter:
```bash
pip install ipywidgets ipyfilechooser
```

## Run from YAML (CLI)
A CLI helper is provided to run a job YAML directly:
```bash
python scripts/run_pipeline.py config/jobs/example.yaml
```

You can specify a different base config:
```bash
python scripts/run_pipeline.py config/jobs/example.yaml --base-config config/base.yaml
```

## Run from Python
```python
from spatial_data_mining.orchestrator import run_pipeline

results = run_pipeline("config/jobs/example.yaml")
for item in results:
    print(item)
```

## Configuration
The pipeline merges `config/base.yaml` defaults with a job YAML file under `config/jobs/`.

### Base config (`config/base.yaml`)
- `defaults.allowed_crs`: allowed EPSG codes for target CRS.
- `defaults.resolution_m`: default output resolution (meters).
- `defaults.storage`: default storage settings.
- `logging`: logging level and format.

### Job config example (`config/jobs/example.yaml`)
```yaml
job:
  name: nd_indices_demo
  aoi_path: data/aoi/sample.geojson
  target_crs: EPSG:4326
  resolution_m: 20
  year: 2023
  season: summer
  variables: [ndvi, fvc, ndmi, msi, bsi, swi, rgb, rgb_raw]
  storage:
    kind: local_cog          # or gcs_cog
    output_dir: data/outputs # used for local_cog
    bucket: openpas-hdh-dev-cache # used for gcs_cog
    prefix: tiles/active     # optional
```

### Job fields
- `aoi_path` or `aoi_paths`: one or more AOI files.
- `target_crs`: output CRS, must be in `defaults.allowed_crs` if defined.
- `resolution_m`: output pixel size in meters (or omit to use native).
- `year` or `years`: one or more years.
- `season` or `seasons`: `winter`, `spring`, `summer`, `autumn`, `annual` (or `static` for `alpha_earth`, `swi`, `rgb` when using static layers).
- `variables`: list of variables, e.g. `ndvi`, `fvc`, `ndmi`, `msi`, `bsi`, `swi`, `rgb`, `rgb_raw`, `alpha_earth`, `clcplus`.
- `storage`: `local_cog` (required `output_dir`) or `gcs_cog` (required `bucket`).
- `clcplus_input_dir`: required when requesting `clcplus`.
- `swi_collection_id`, `swi_band`, `swi_aggregation`, `swi_backend_url`, `swi_date`, `swi_oidc_provider_id`: optional overrides for the SWI extractor.
- `rgb_date`, `rgb_search_days`, `rgb_collection_id`, `rgb_bands`, `rgb_cloud_cover_max`, `rgb_cloud_cover_property`, `rgb_backend_url`, `rgb_oidc_provider_id`, `rgb_stac_url`, `rgb_stac_collection_id`, `rgb_prefilter`: optional overrides for the RGB extractor.
Note: if `rgb_date` is not provided, the pipeline uses the mid-season date for each year/season.
Note: for `gcs_cog`, `storage.prefix` is treated as a base prefix; the pipeline appends `<variable>/<year>/<season>/` automatically.

## Data sources and extraction logic
Each variable is extracted from the following source and processed as described:

- `ndvi`: Copernicus Data Space openEO (`SENTINEL2_L2A`). Seasonal median composite of required bands, then index computed. Cloud cover filter applied when available (`OPENEO_MAX_CLOUD_COVER`). Output resamples to target CRS/resolution and clips to AOI.
- `fvc`: Derived from a seasonal NDVI median composite (Sentinel-2 B08/B04). NDVI_soil and NDVI_veg are the 5th and 95th percentiles over the AOI; FVC is clipped to [0, 1]. Output resamples to target CRS/resolution and clips to AOI.
- `ndmi`: Same as `ndvi`, but uses B08/B11 and outputs on the 20m grid by default.
- `msi`: Same as `ndvi`, but uses B11/B08 and outputs on the 20m grid by default.
- `bsi`: Same as `ndvi`, uses B11/B04/B08/B02 and outputs on the 20m grid by default.
- `rgb`: Copernicus Data Space openEO (`SENTINEL2_L1C`). Applies the Sentinel Hub L1C true color optimized formula to B04/B03/B02 and outputs 8-bit RGBA for visualization. Uses the same date selection logic as `rgb_raw` (include both if you want raw + visualization outputs).
- `rgb_raw`: Copernicus Data Space openEO (`SENTINEL2_L2A`). Raw RGB reflectance (B04/B03/B02) for a single date; selects the closest available image to `rgb_date` within the search window. If no single date fully covers the AOI, the pipeline mosaics multiple dates to fill nodata (configurable). If no date is provided, uses the mid-season date.
- `swi`: Copernicus Land Monitoring Service Soil Water Index via openEO. Defaults to VITO `CGLS_SWI_V1_EUROPE` with band `SWI_100`. If no exact date is provided, the mid-season date is used; set `OPENEO_SWI_AGGREGATION` if you want aggregation. Prefer native resolution (omit `resolution_m`) to avoid upsampling.
- `alpha_earth`: Google Earth Engine (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`). Annual image for the given year, clipped to AOI; falls back to tiling if AOI is large.
- `clcplus`: Local, user-provided CLCplus raster(s). The pipeline selects the raster with the largest AOI overlap, then reprojects and clips; nodata and 0 values are recoded to -9999.

## Season date windows
Season names map to the following date ranges (used by openEO and GEE extractors). If a season name is not recognized, the pipeline defaults to the full year.

- `winter`: Dec 1 of `year` to Feb 28 of `year + 1`
- `spring`: Mar 1 to May 31 of `year`
- `summer`: Jun 1 to Aug 31 of `year`
- `autumn` / `fall`: Sep 1 to Nov 30 of `year`
- `annual` / `year`: Jan 1 to Dec 31 of `year`

## Authentication and environment variables
### Copernicus Data Space openEO
Sentinel-2 indices (`ndvi`, `fvc`, `ndmi`, `msi`, `bsi`) use openEO.
- Auth: The pipeline triggers OIDC device flow if no session exists.
- Optional env vars:
  - `OPENEO_BACKEND_URL` (default: `https://openeo.dataspace.copernicus.eu`)
  - `OPENEO_AUTH_METHOD`, `OPENEO_CLIENT_ID`, `OPENEO_CLIENT_SECRET`, `OPENEO_USERNAME`, `OPENEO_PASSWORD`
  - `OPENEO_S2_COLLECTION_ID` (default: `SENTINEL2_L2A`)
  - `OPENEO_MAX_CLOUD_COVER` (default: 40)
  - `OPENEO_CLOUD_COVER_PROPERTY` (default: `eo:cloud_cover`)
  - `OPENEO_RGB_DATE` (optional; `YYYY-MM-DD` target date)
  - `OPENEO_RGB_SEARCH_DAYS` (optional; days to search around target date, default 30)
  - `OPENEO_RGB_BANDS` (optional; default `B04,B03,B02`)
  - `OPENEO_RGB_COLLECTION_ID` (optional; default `SENTINEL2_L2A`)
  - `OPENEO_RGB_MAX_CLOUD_COVER` (optional; fallback to `OPENEO_MAX_CLOUD_COVER`)
  - `OPENEO_RGB_CLOUD_COVER_PROPERTY` (optional; fallback to `OPENEO_CLOUD_COVER_PROPERTY`)
  - `OPENEO_RGB_CLOUD_MASK` (optional; set `0` to disable SCL-based cloud masking, default on)
  - `OPENEO_RGB_CLOUD_MASK_BAND` (optional; default `SCL`)
  - `OPENEO_RGB_CLOUD_MASK_CLASSES` (optional; default `0,1,3,8,9,10,11`)
  - `OPENEO_RGB_BACKEND_URL` (optional; default `OPENEO_BACKEND_URL`)
  - `OPENEO_RGB_STAC_URL` (optional; default `https://catalogue.dataspace.copernicus.eu/stac`)
  - `OPENEO_RGB_STAC_COLLECTION_ID` (optional; default `sentinel-2-l2a`)
  - `OPENEO_RGB_PREFILTER` (optional; set `0` to disable STAC prefiltering)
  - `OPENEO_RGB_ALLOW_MOSAIC` (optional; set `0` to disable multi-date mosaicking when a single date does not fully cover the AOI)
  - `OPENEO_RGB_MOSAIC_MAX_DATES` (optional; maximum number of dates to mosaic, default 5)
  - `OPENEO_RGB_MIN_COVERAGE` (optional; minimum AOI coverage ratio to accept a single date or stop mosaicking, default 0.999)
  - `OPENEO_SWI_COLLECTION_ID` (SWI collection id in the openEO backend)
  - `OPENEO_SWI_BAND` (optional SWI band name; comma-separated for multiple)
  - `OPENEO_SWI_AGGREGATION` (optional; `mean`, `median`, or `none`)
  - `OPENEO_SWI_BACKEND_URL` (optional; default `https://openeo.vito.be`)
  - `OPENEO_SWI_DATE` (optional; `YYYY-MM-DD` exact date for SWI)
  - `OPENEO_SWI_OIDC_PROVIDER_ID` (optional; default `terrascope` on VITO)
  - `OPENEO_RGB_OIDC_PROVIDER_ID` (optional; e.g. `CDSE` for Copernicus Data Space)
- TLS: if the backend has a broken certificate and you still want to test, set `OPENEO_VERIFY_SSL=0` (insecure).

### Google Earth Engine
Used by `alpha_earth`.
- Auth once via `earthengine authenticate`.

### Parallelism
- `SDM_MAX_CONCURRENT_TASKS=2` caps concurrent runs to match openEO free-tier limits.
- `SDM_DISABLE_OPENEO_INDEX_BATCH=1` disables multi-index batching for B11-based indices.

## Outputs
COG files are written to `data/outputs/` by default.
Naming format:
```
<variable>_<year>_<season>_<aoi>.tif
```
If `gcs_cog` is selected, files are uploaded to:
```
gs://<bucket>/<prefix>/<variable>/<year>/<season>/<filename>.tif
```
With the default GCS settings, outputs land under:
```
gs://openpas-hdh-dev-cache/tiles/active/<variable>/<year>/<season>/<filename>.tif
```

## Repo structure
- `scripts/` - setup and CLI helpers.
- `notebooks/` - notebook UI for running jobs.
- `config/` - defaults (`base.yaml`) and job configs (`jobs/*.yaml`).
- `src/spatial_data_mining/` - core pipeline code.
- `data/` - AOIs (`aoi/`) and outputs (`outputs/`).
- `legacy/` - legacy notebooks/scripts.

## Troubleshooting
- Widgets do not render: `pip install ipywidgets ipyfilechooser` and restart Jupyter.
- openEO TLS errors: set `OPENEO_VERIFY_SSL=0` only for testing.
- AOI errors: ensure the AOI has a defined CRS and intersects the source imagery.
