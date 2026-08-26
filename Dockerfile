# ==============================================================================
# Sentinel — Multimodal Workforce Risk Intelligence Platform
# Production Multi-Stage Dockerfile (FastAPI + React SPA Unified Deployment)
# ==============================================================================

# --- Stage 1: Build React Frontend ---
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# --- Stage 2: Production Python Application Runner ---
FROM python:3.12-slim AS runner
WORKDIR /app

# Configure execution environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    ENVIRONMENT=production \
    ARTIFACTS_DIR=/app/artifacts \
    HF_MODEL_REPO_ID=ParminderzHuggingFace/sentinel-workforce-risk-models \
    TORCH_NUM_THREADS=1 \
    TORCH_NUM_INTEROP_THREADS=1

# Install runtime curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata, configs, scripts, reports, and Python source tree
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY reports/ ./reports/

# Install package dependencies and register workforce-risk package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Download frozen model artifacts at build time from public Hugging Face repository
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('ParminderzHuggingFace/sentinel-workforce-risk-models', local_dir='/app/artifacts', allow_patterns=['structured_model/*', 'text_transformer/*', 'fusion/*', 'evaluation_summary.json'])"

# Copy compiled production frontend bundle from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Expose web server port (dynamically overridden by cloud platform $PORT)
EXPOSE 8000

# Container health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Execute production-aware server launcher
CMD ["python", "scripts/serve.py"]
