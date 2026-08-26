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
    ARTIFACTS_DIR=/app/artifacts

# Install runtime curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from pyproject.toml
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy application source modules and scripts
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Copy compiled production frontend bundle from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Copy staged runtime model artifacts
COPY deployment/artifacts/ /app/artifacts/

# Expose web server port (dynamically overridden by cloud platform $PORT)
EXPOSE 8000

# Container health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Execute production-aware server launcher
CMD ["python", "scripts/serve.py"]
