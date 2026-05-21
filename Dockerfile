# FLARE Production ML — Dockerfile
# ==================================
# Multi-stage approach not needed here since we're inference-only.
# Single stage: Python slim base + dependencies + code.
# ONNX models are NOT baked in — downloaded from HuggingFace at startup.
# This keeps the image small (~2GB vs ~4GB with models).
#
# Build:  docker build -t flare-api .
# Run:    docker run -p 8080:8080 flare-api

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# libgomp1: required by ONNX Runtime for OpenMP threading
# curl: used for health checks
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first
# Doing this before copying code means Docker caches this layer
# — only re-runs if requirements.txt changes, not on every code change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY src/ ./src/

# Create models directory
# ONNX models download here at runtime from HuggingFace
RUN mkdir -p models/onnx

# Expose port
EXPOSE 8080

# Health check
# Docker pings /health every 30s to verify container is alive
# --start-period 120s: gives models time to download on first startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the server
# workers=1: single worker since models are loaded once in memory
# Models download from HuggingFace on first startup via ensure_onnx_models()
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
