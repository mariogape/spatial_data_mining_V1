#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "If first time, run: earthengine authenticate"
echo "For GCS uploads, ensure ADC: gcloud auth application-default login OR set GOOGLE_APPLICATION_CREDENTIALS"
