# ColdPath GPU Training Server
# For cool.grait.io (2× RTX 3090, CUDA 12.1+)
#
# Base: pytorch/pytorch with CUDA 12.1 (RTX 3090 compatible)
# Includes: All ColdPath ML modules, FastAPI server, training pipeline

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ColdPath source and dependency manifest
COPY pyproject.toml ./
COPY src/ src/

# Install all ML dependencies (torch already in base image)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[all]" \
    && pip install --no-cache-dir asyncpg psycopg2-binary

# Create persistent data directories
RUN mkdir -p /data/datasets /data/models /data/logs /data/mlflow

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Training-only mode: no HotPath, no live trading
ENV AUTOTRADER_ENABLED=false
ENV TRADING_MODE=paper
ENV STORAGE_BACKEND=sqlite
ENV DATABASE_PATH=/data/coldpath.db
ENV MODEL_ARTIFACT_DIR=/data/models

CMD ["uvicorn", "coldpath.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
