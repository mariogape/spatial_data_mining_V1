#!/bin/bash
# ========================================
# Startup Script: Spatial Data Mining Pipeline
# ========================================
#
# This script runs on VM startup to execute the data mining pipeline.
# Variables are replaced by Cloud Build before VM creation.
#
# ========================================

set -e  # Exit on any error

# ========================================
# Configuration (Replaced by Cloud Build)
# ========================================
REGION="__REGION__"
IMAGE_URI="__IMAGE_URI__"
JOB_CONFIG_FILE="__JOB_CONFIG_FILE__"
BASE_CONFIG_FILE="__BASE_CONFIG_FILE__"
VM_NAME="__VM_NAME__"
CLOUDBUILD_YAML="__CLOUDBUILD_YAML__"
BUILD_ID="__BUILD_ID__"
GIT_COMMIT="__GIT_COMMIT__"
PIPELINE_TITLE="__PIPELINE_TITLE__"
PROJECT_ID="__PROJECT_ID__"

# ========================================
# Logging Setup
# ========================================
LOG_FILE="/var/log/data-mining-pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "🚀 $PIPELINE_TITLE"
echo "=========================================="
echo "VM Name:      $VM_NAME"
echo "Build ID:     $BUILD_ID"
echo "Git Commit:   $GIT_COMMIT"
echo "Image:        $IMAGE_URI"
echo "Job Config:   $JOB_CONFIG_FILE"
echo "Started:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="
echo ""

# ========================================
# Function: Cleanup and shutdown
# ========================================
cleanup_and_shutdown() {
  EXIT_CODE=$?
  echo ""
  echo "=========================================="
  echo "🏁 Pipeline Execution Complete"
  echo "=========================================="
  echo "Exit Code: $EXIT_CODE"
  echo "Ended:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""

  if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Pipeline completed successfully"
  else
    echo "❌ Pipeline failed with exit code $EXIT_CODE"
    echo "📋 Check logs: $LOG_FILE"
  fi

  echo ""
  echo "💾 Uploading logs to GCS..."
  LOG_BUCKET="gs://${PROJECT_ID}-logs"
  LOG_PATH="${LOG_BUCKET}/data-mining/${BUILD_ID}/${VM_NAME}.log"

  # Try to upload logs (don't fail if bucket doesn't exist)
  if gsutil ls "$LOG_BUCKET" >/dev/null 2>&1; then
    gsutil cp "$LOG_FILE" "$LOG_PATH" || echo "⚠️  Failed to upload logs"
    echo "📁 Logs uploaded to: $LOG_PATH"
  else
    echo "⚠️  Log bucket $LOG_BUCKET not found, skipping log upload"
  fi

  echo ""
  echo "🔌 Shutting down VM in 30 seconds..."
  sleep 30

  # Shutdown the VM (works for both regular and preemptible instances)
  sudo poweroff
}

# Register cleanup function
trap cleanup_and_shutdown EXIT

# ========================================
# Step 1: Authenticate with Docker Registry
# ========================================
echo "=========================================="
echo "Step 1: Authenticating with Docker Registry"
echo "=========================================="

gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

echo "✅ Docker authentication complete"
echo ""

# ========================================
# Step 2: Pull Docker Image
# ========================================
echo "=========================================="
echo "Step 2: Pulling Docker Image"
echo "=========================================="
echo "Image: $IMAGE_URI"
echo ""

docker pull "$IMAGE_URI"

echo "✅ Image pulled successfully"
echo ""

# ========================================
# Step 3: Download Job Configuration from GCS
# ========================================
echo "=========================================="
echo "Step 3: Downloading Job Configuration"
echo "=========================================="

WORK_DIR="/workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone or copy the repository to get config files
# If job config is in GCS, download it
if [[ "$JOB_CONFIG_FILE" == gs://* ]]; then
  echo "Downloading job config from GCS..."
  gsutil cp "$JOB_CONFIG_FILE" "$WORK_DIR/job_config.json"
  JOB_CONFIG_PATH="$WORK_DIR/job_config.json"
else
  # Config is in repo, we need to clone it
  echo "Cloning repository to get config files..."
  REPO_URL="https://source.developers.google.com/p/${PROJECT_ID}/r/spatial_data_mining"

  # Try to clone (might not work if repo doesn't exist in Cloud Source Repositories)
  if gcloud source repos clone spatial_data_mining "$WORK_DIR/repo" --project="$PROJECT_ID" 2>/dev/null; then
    cd "$WORK_DIR/repo"
    git checkout "$GIT_COMMIT" 2>/dev/null || echo "⚠️  Could not checkout commit $GIT_COMMIT, using HEAD"
    JOB_CONFIG_PATH="$WORK_DIR/repo/$JOB_CONFIG_FILE"
  else
    echo "⚠️  Could not clone repository"
    echo "ℹ️  Job config must be provided as GCS path (gs://...)"
    exit 1
  fi
fi

echo "✅ Job configuration ready: $JOB_CONFIG_PATH"
echo ""

# ========================================
# Step 4: Run Data Mining Pipeline
# ========================================
echo "=========================================="
echo "Step 4: Running Data Mining Pipeline"
echo "=========================================="
echo "Config: $JOB_CONFIG_PATH"
echo ""

# Run the pipeline in Docker container with tracking metadata
docker run \
  --rm \
  -v "$WORK_DIR:/workspace" \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
  -e "GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json" \
  -e "SDM_MAX_CONCURRENT_TASKS=2" \
  "$IMAGE_URI" \
  python /app/scripts/run_from_json.py \
    --job-config="/workspace/$(basename "$JOB_CONFIG_PATH")" \
    --git-commit="$GIT_COMMIT" \
    --build-id="$BUILD_ID" \
    --vm-name="$VM_NAME"

echo ""
echo "✅ Pipeline execution finished"

# ========================================
# Cleanup will happen automatically via trap
# ========================================
