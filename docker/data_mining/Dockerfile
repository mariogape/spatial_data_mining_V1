# ========================================
# Dockerfile: Spatial Data Mining Code Layer
# ========================================
#
# This Dockerfile contains the application code and changes frequently.
# Fast rebuilds (~20 seconds) since dependencies are in base image.
#
# Standard 2: Layered Docker builds (code layer)
# ========================================

# Use base image (built separately)
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Copy application code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY config/ /app/config/

# Make scripts executable
RUN chmod +x /app/scripts/*.py

# Set working directory
WORKDIR /app

# Default command runs the JSON-based pipeline
CMD ["python", "/app/scripts/run_from_json.py", "--help"]
