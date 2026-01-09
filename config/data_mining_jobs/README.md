# Data Mining Job Configurations

This directory contains JSON configurations for spatial data mining jobs.

## JSON Schema

Each job configuration must follow this structure:

```json
{
  "job_name": "string - Unique identifier for this job",
  "description": "string - Human-readable description",

  "aoi_paths": [
    "string - Path or GCS URI to AOI GeoJSON file"
  ],

  "spatial_config": {
    "target_crs": "string - Target CRS (e.g., EPSG:4326, EPSG:3035)",
    "resolution_m": "number - Resolution in meters",
    "use_native_resolution": "boolean - Use native resolution if true"
  },

  "temporal_config": {
    "years": ["number - Year(s) to process"],
    "seasons": ["string - winter|spring|summer|autumn|annual"]
  },

  "variables": [
    "string - ndvi|ndmi|msi|bsi|swi|rgb|clcplus|alpha_earth|etc."
  ],

  "storage": {
    "kind": "string - local_cog|gcs_cog",
    "bucket": "string - GCS bucket name (required for gcs_cog)",
    "prefix": "string - GCS prefix/folder path"
  },

  "variable_specific": {
    "clcplus_input_dir": "string|null - Path to CLCplus data (if clcplus variable used)",
    "swi_date": "string|null - Date for SWI in YYYY-MM-DD format",
    "rgb_date": "string|null - Date for RGB in YYYY-MM-DD format"
  },

  "performance": {
    "max_concurrent_tasks": "number - Max parallel tasks (1-2 for openEO free tier)"
  }
}
```

## Example Usage

Create a new job config:
```bash
cp example_job.json my_job.json
# Edit my_job.json with your parameters
```

Deploy and run:
```bash
gcloud builds submit --config=cloudbuild-data-mining.yaml \
  --substitutions=_JOB_CONFIG_FILE=config/data_mining_jobs/my_job.json
```

## Variable Options

| Variable | Description | Requirements |
|----------|-------------|--------------|
| `ndvi` | Normalized Difference Vegetation Index | None |
| `ndmi` | Normalized Difference Moisture Index | None |
| `msi` | Moisture Stress Index | None |
| `bsi` | Bare Soil Index | None |
| `swi` | Soil Water Index | Requires `swi_date` |
| `rgb` | RGB Composite | Requires `rgb_date` |
| `clcplus` | Corine Land Cover Plus | Requires `clcplus_input_dir` |
| `alpha_earth` | Alpha Earth data | Requires GEE auth |

## CRS Options

Common CRS values:
- `EPSG:4326` - WGS84 (lat/lon)
- `EPSG:3035` - ETRS89-extended / LAEA Europe
- `EPSG:3857` - Web Mercator
- `EPSG:25829` - ETRS89 / UTM zone 29N
- `EPSG:25830` - ETRS89 / UTM zone 30N
- `EPSG:25831` - ETRS89 / UTM zone 31N

## Season Options

- `winter` - December to February
- `spring` - March to May
- `summer` - June to August
- `autumn` / `fall` - September to November
- `annual` / `year` - Full year

## Storage Modes

### local_cog
Saves files locally on the VM (usually for testing):
```json
"storage": {
  "kind": "local_cog",
  "bucket": null,
  "prefix": null
}
```

### gcs_cog (Recommended)
Uploads directly to Google Cloud Storage:
```json
"storage": {
  "kind": "gcs_cog",
  "bucket": "my-output-bucket",
  "prefix": "spatial_data/outputs"
}
```
Results will be saved to: `gs://my-output-bucket/spatial_data/outputs/{variable}/{year}/{season}/{filename}.tif`
