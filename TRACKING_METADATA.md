# Pipeline Tracking & Metadata System

## Overview

Every pipeline run automatically generates **tracking metadata files** that include:
- ✅ Job configuration details
- ✅ Git commit SHA (code version)
- ✅ Cloud Build ID (deployment tracking)
- ✅ VM instance name
- ✅ Execution timestamp
- ✅ List of all outputs with GCS URIs
- ✅ Success/failure status

**These files are automatically uploaded to your GCS bucket** for easy tracking and auditing.

---

## 📁 Generated Files

For each pipeline run, 3 files are generated:

### 1. **JSON Metadata** (`{job_name}_{timestamp}_metadata.json`)
Complete structured metadata including job config, execution details, and all outputs.

**Example:**
```json
{
  "job_info": {
    "job_name": "spain_ndvi_2023",
    "timestamp": "2024-01-15T10:30:45Z",
    "git_commit": "abc123def456",
    "build_id": "12345678-abcd-1234-5678-abc123def456",
    "vm_name": "data-mining-20240115-103045"
  },
  "configuration": {
    "variables": ["ndvi", "ndmi"],
    "years": [2023],
    "seasons": ["summer"],
    "num_aois": 2,
    "target_crs": "EPSG:4326",
    "resolution_m": 20
  },
  "storage": {
    "kind": "gcs_cog",
    "bucket": "my-output-bucket",
    "prefix": "outputs/spain"
  },
  "execution": {
    "total_outputs": 4,
    "successful_outputs": 4,
    "failed_outputs": 0
  },
  "outputs": [
    {
      "aoi": "region1",
      "variable": "ndvi",
      "year": 2023,
      "season": "summer",
      "gcs_uri": "gs://my-bucket/outputs/spain/ndvi/2023/summer/region1.tif"
    }
  ]
}
```

### 2. **CSV Outputs** (`{job_name}_{timestamp}_outputs.csv`)
Tabular format perfect for Excel, database imports, or analysis scripts.

**Columns:**
- job_name
- timestamp
- git_commit
- build_id
- vm_name
- target_crs
- resolution_m
- num_aois
- storage_kind
- bucket
- prefix
- aoi
- aoi_path
- variable
- year
- season
- local_path
- gcs_uri
- filename

**Example CSV:**
```csv
job_name,timestamp,git_commit,build_id,vm_name,variable,year,season,gcs_uri
spain_ndvi_2023,2024-01-15T10:30:45Z,abc123,12345678,data-mining-20240115,ndvi,2023,summer,gs://bucket/outputs/ndvi_2023.tif
spain_ndvi_2023,2024-01-15T10:30:45Z,abc123,12345678,data-mining-20240115,ndmi,2023,summer,gs://bucket/outputs/ndmi_2023.tif
```

### 3. **Text Summary** (`{job_name}_{timestamp}_summary.txt`)
Human-readable summary perfect for quick review or email reports.

**Example:**
```
================================================================================
SPATIAL DATA MINING PIPELINE - EXECUTION SUMMARY
================================================================================

JOB INFORMATION
--------------------------------------------------------------------------------
Job Name:       spain_ndvi_2023
Timestamp:      2024-01-15T10:30:45Z
Git Commit:     abc123def456
Build ID:       12345678-abcd-1234-5678-abc123def456
VM Instance:    data-mining-20240115-103045

CONFIGURATION
--------------------------------------------------------------------------------
Variables:      ndvi, ndmi
Years:          2023
Seasons:        summer
AOIs:           2 file(s)
Target CRS:     EPSG:4326
Resolution:     20 meters

STORAGE
--------------------------------------------------------------------------------
Storage Type:   gcs_cog
Bucket:         my-output-bucket
Prefix:         outputs/spain

EXECUTION RESULTS
--------------------------------------------------------------------------------
Total Outputs:      4
Successful:         4
Failed:             0

OUTPUTS
--------------------------------------------------------------------------------
1. region1 / ndvi / 2023 / summer
   GCS:   gs://my-bucket/outputs/spain/ndvi/2023/summer/region1.tif

2. region1 / ndmi / 2023 / summer
   GCS:   gs://my-bucket/outputs/spain/ndmi/2023/summer/region1.tif

...
```

---

## 📂 File Locations

### GCS Storage (Automatic)
When using `storage.kind: "gcs_cog"`, metadata files are automatically uploaded to:

```
gs://{your-bucket}/{prefix}/metadata/{job_name}_{timestamp}_metadata.json
gs://{your-bucket}/{prefix}/metadata/{job_name}_{timestamp}_outputs.csv
gs://{your-bucket}/{prefix}/metadata/{job_name}_{timestamp}_summary.txt
```

**Example:**
```
gs://my-bucket/outputs/spain/metadata/spain_ndvi_2023_20240115_103045_metadata.json
gs://my-bucket/outputs/spain/metadata/spain_ndvi_2023_20240115_103045_outputs.csv
gs://my-bucket/outputs/spain/metadata/spain_ndvi_2023_20240115_103045_summary.txt
```

### Local Storage
When using `storage.kind: "local_cog"`, files are saved to:
```
{output_dir}/metadata/{job_name}_{timestamp}_*.{json,csv,txt}
```

---

## 🔍 How to Use Tracking Files

### 1. **View Latest Run Summary**

```bash
# List recent metadata files
gsutil ls -l gs://my-bucket/outputs/spain/metadata/ | tail -10

# Download latest summary
gsutil cp gs://my-bucket/outputs/spain/metadata/*_summary.txt .
cat *_summary.txt
```

### 2. **Track All Runs in Spreadsheet**

```bash
# Download all CSV files
gsutil cp 'gs://my-bucket/outputs/spain/metadata/*_outputs.csv' ./tracking/

# Combine into master tracking sheet
cat tracking/*_outputs.csv | head -1 > master_tracking.csv  # Header
tail -n +2 -q tracking/*_outputs.csv >> master_tracking.csv   # All data rows

# Open in Excel/Google Sheets
```

### 3. **Find Outputs from Specific Git Commit**

```bash
# Search JSON metadata
gsutil cat 'gs://my-bucket/outputs/spain/metadata/*_metadata.json' | \
  jq 'select(.job_info.git_commit == "abc123def456")'

# Or using grep on CSV
gsutil cat 'gs://my-bucket/outputs/spain/metadata/*_outputs.csv' | \
  grep "abc123def456"
```

### 4. **Audit Pipeline Runs**

```bash
# Get all run timestamps and commits
gsutil cat 'gs://my-bucket/outputs/spain/metadata/*_metadata.json' | \
  jq -r '[.job_info.timestamp, .job_info.git_commit, .job_info.build_id] | @csv'

# Output:
# "2024-01-15T10:30:45Z","abc123","build-001"
# "2024-01-16T14:20:10Z","def456","build-002"
# "2024-01-17T09:15:30Z","ghi789","build-003"
```

### 5. **Check Failed Outputs**

```bash
# Find jobs with failures
gsutil cat 'gs://my-bucket/outputs/spain/metadata/*_metadata.json' | \
  jq 'select(.execution.failed_outputs > 0) | .job_info'
```

### 6. **Generate Report for Specific Time Period**

```bash
# Get all runs from January 2024
gsutil ls 'gs://my-bucket/outputs/spain/metadata/*202401*_summary.txt' | \
  while read file; do
    echo "===== $(basename $file) ====="
    gsutil cat "$file" | head -20
    echo ""
  done
```

---

## 🔗 Integration Examples

### Python: Load and Analyze Metadata

```python
import json
from google.cloud import storage

def load_metadata(bucket_name, prefix):
    """Load all metadata JSON files from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=f"{prefix}/metadata/")
    metadata_files = [b for b in blobs if b.name.endswith('_metadata.json')]

    all_metadata = []
    for blob in metadata_files:
        content = blob.download_as_text()
        metadata = json.loads(content)
        all_metadata.append(metadata)

    return all_metadata

# Usage
metadata = load_metadata('my-bucket', 'outputs/spain')

# Analyze
for run in metadata:
    job_name = run['job_info']['job_name']
    timestamp = run['job_info']['timestamp']
    total = run['execution']['total_outputs']
    successful = run['execution']['successful_outputs']

    print(f"{job_name} ({timestamp}): {successful}/{total} successful")
```

### SQL: Query CSV in BigQuery

```sql
-- Load CSV files into BigQuery
CREATE EXTERNAL TABLE `project.dataset.pipeline_outputs`
OPTIONS (
  format = 'CSV',
  uris = ['gs://my-bucket/outputs/spain/metadata/*_outputs.csv'],
  skip_leading_rows = 1
);

-- Query outputs by variable
SELECT
  job_name,
  timestamp,
  git_commit,
  variable,
  COUNT(*) as num_outputs
FROM `project.dataset.pipeline_outputs`
GROUP BY job_name, timestamp, git_commit, variable
ORDER BY timestamp DESC;

-- Find outputs from specific commit
SELECT
  aoi,
  variable,
  year,
  season,
  gcs_uri
FROM `project.dataset.pipeline_outputs`
WHERE git_commit = 'abc123def456';
```

### Bash: Automated Email Report

```bash
#!/bin/bash
# Send daily summary email

BUCKET="my-bucket"
PREFIX="outputs/spain/metadata"
TODAY=$(date +%Y%m%d)

# Get today's summary files
gsutil ls "gs://$BUCKET/$PREFIX/*${TODAY}*_summary.txt" | while read file; do
  gsutil cat "$file"
done | mail -s "Data Mining Pipeline Summary - $(date +%Y-%m-%d)" team@example.com
```

---

## 📊 Tracking Dashboard Ideas

### Option 1: Google Sheets Integration

1. Load CSV files into Google Sheets
2. Create pivot tables for:
   - Runs per day/week
   - Outputs by variable
   - Success rate over time

### Option 2: Custom Dashboard

Build a simple dashboard using:
- Python + Streamlit
- Load metadata JSONs from GCS
- Visualize with Plotly:
  - Timeline of runs
  - Output counts by variable
  - Git commit history

### Option 3: BigQuery + Data Studio

1. Load CSVs into BigQuery (automated with Cloud Function)
2. Create Data Studio dashboard
3. Real-time tracking and reporting

---

## 🛠️ Troubleshooting

### Metadata Files Not Generated

**Check:**
```bash
# Verify generate_job_metadata.py exists
ls scripts/generate_job_metadata.py

# Check Docker image includes scripts
docker run YOUR_IMAGE ls /app/scripts/
```

### Files Not Uploading to GCS

**Check logs:**
```bash
gcloud logging read "resource.type=gce_instance AND labels.type=data-mining" \
  --format="value(jsonPayload.message)" | grep -i metadata
```

**Common issues:**
- Service account lacks `storage.objects.create` permission
- Bucket name incorrect in job config
- Network connectivity issues (rare)

### Cannot Find Metadata Files

```bash
# List all metadata
gsutil ls -r 'gs://my-bucket/**/metadata/*.json'

# Search by job name
gsutil ls 'gs://my-bucket/**/metadata/*spain*'
```

---

## 🎯 Best Practices

1. **Keep CSVs for Historical Analysis**
   - Download and archive monthly
   - Import into database for long-term tracking

2. **Use Git Commits for Reproducibility**
   - Always commit before deploying
   - Use commit SHA to recreate exact pipeline version

3. **Monitor Failed Outputs**
   - Set up alerts for `failed_outputs > 0`
   - Review errors in Cloud Logging

4. **Version Your Job Configs**
   - Keep job JSONs in Git
   - Reference git commit in job config description

5. **Automate Reporting**
   - Schedule weekly summary emails
   - Dashboard for stakeholder visibility

---

## 📚 Related Documentation

- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **Job Configuration**: `config/data_mining_jobs/README.md`
- **Pipeline Code**: `scripts/run_from_json.py`
- **Metadata Generation**: `scripts/generate_job_metadata.py`

---

## ✨ Summary

**Every pipeline run now creates:**
- ✅ JSON metadata (machine-readable)
- ✅ CSV outputs (Excel/database friendly)
- ✅ Text summary (human-readable)

**Automatically tracked:**
- ✅ Git commit (code version)
- ✅ Build ID (deployment)
- ✅ VM instance
- ✅ All outputs with GCS URIs
- ✅ Success/failure status

**Easy to access:**
```bash
# View latest run
gsutil cat gs://my-bucket/outputs/*/metadata/*_summary.txt | tail -100

# Track all runs
gsutil cp 'gs://my-bucket/outputs/*/metadata/*_outputs.csv' ./
```

**Your pipeline runs are now fully auditable and trackable!** 🎉
