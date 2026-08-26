# Deployment Guide

This document describes the deployment configuration, containerization, local execution, and cloud deployment status of the Sentinel Workforce Risk Platform.

---

## 1. Deployment Overview & Current Status

- **Local Execution**: Fully verified and operational.
- **Docker Container Build**: Verified multi-stage Docker build that bakes frontend assets and Hugging Face model artifacts into a self-contained image.
- **Cloud Deployment Status**: **Prepared and configured**, but **not currently running in production on Render Free tier**.

---

## 2. Cloud Deployment Findings & Memory Constraints

### 2.1 Render Free Tier Evaluation
The application was packaged and tested for deployment on Render. The multi-stage Docker build, package registration, asset compilation, and Hugging Face artifact download all succeeded.

However, container execution on the **Render Free plan (512 MB RAM limit)** failed during runtime inference initialization due to Linux cgroup memory exhaustion (OOM).

### 2.2 Memory Breakdown vs. Artifact Size
While the model artifacts on disk total only **4.31 MB**, runtime RAM consumption is significantly higher due to framework and runtime requirements:

| Component / State | Disk Size | Measured RAM Usage | Notes |
| :--- | :---: | :---: | :--- |
| **Model Weights on Disk** | 4.31 MB | — | 8 serialized checkpoint / config / adapter files. |
| **Python & FastAPI Base Process** | — | ~75 MB | Uvicorn, FastAPI, Pydantic, Starlette runtime. |
| **PyTorch + Scikit-Learn + DistilBERT Loaded** | — | ~470–485 MB | In-memory model parameters, computation graph, vocabulary. |
| **Single Request Prediction Peak** | — | ~570–585 MB | Tensor allocations, intermediate activation buffers. |
| **50-Employee Batch Prediction Peak** | — | ~590 MB | Batch tensor transformation and inference. |
| **Render Free Tier Limit** | — | **512 MB** | Hard cgroup memory threshold. |

### 2.3 Oracle Cloud Infrastructure (OCI) Free Tier Evaluation
Deployment on OCI Always Free ARM64 compute (`VM.Standard.A1.Flex`) was also designed and tested. While VCN networking, routing, subnets, and security lists were successfully configured, instance provisioning encountered physical capacity limits (`500 Out of host capacity` in `ap-hyderabad-1`). The infrastructure changes were rolled back to preserve the existing tenant state.

---

## 3. Future Deployment Path

To run Sentinel reliably in a cloud environment:
- **Target Memory**: Minimum **1.0 GB RAM**, recommended **2.0 GB RAM** (e.g. Render Starter/Standard, AWS ECS, or a basic VM).
- **Target CPU**: 1 vCPU is sufficient for real-time single and small-batch inference.
- **Model Artifact Source**: The public Hugging Face Model Repository:
  `https://huggingface.co/ParminderzHuggingFace/sentinel-workforce-risk-models`

---

## 4. Local Execution

### 4.1 Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- Node.js 20+ and npm
- Git

### 4.2 Installation

```bash
# Clone the repository
git clone https://github.com/ParminderSinghGithub/Sentinel.git
cd Sentinel

# Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# or .\.venv\Scripts\Activate.ps1 on Windows

# Install Python dependencies and package in editable mode
pip install --upgrade pip
pip install -e .

# Install frontend dependencies and build the SPA
cd frontend
npm ci
npm run build
cd ..
```

### 4.3 Running the Unified Server Locally

```bash
python scripts/serve.py
```

The application will launch on `http://127.0.0.1:8000`:
- **Web UI**: Navigate to `http://127.0.0.1:8000/` in your browser.
- **API Documentation**: Open `http://127.0.0.1:8000/docs` for interactive Swagger UI.
- **Health Check**: `http://127.0.0.1:8000/health`.

---

## 5. Docker Deployment

### 5.1 Multi-Stage Dockerfile Architecture

The `Dockerfile` performs a clean two-stage build:
1. **Stage 1 (`frontend-builder`)**: Uses `node:20-slim` to run `npm ci` and `npm run build`, producing `frontend/dist/`.
2. **Stage 2 (`runner`)**: Uses `python:3.12-slim`, installs system `curl`, installs Python dependencies from `pyproject.toml`, downloads frozen model artifacts from Hugging Face into `/app/artifacts`, and copies the compiled frontend bundle.

### 5.2 Building and Running the Container

```bash
# Build the Docker image
docker build -t sentinel:latest .

# Run the container (binding to host port 8000)
docker run -d \
  --name sentinel-app \
  -p 8000:8000 \
  -e PORT=8000 \
  -e ENVIRONMENT=production \
  sentinel:latest

# Verify health status
curl http://localhost:8000/health
```

---

## 6. Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PORT` | `8000` | Port for the Uvicorn web server (automatically provided by cloud platforms like Render). |
| `HOST` | `0.0.0.0` in prod / `127.0.0.1` in dev | Network interface to bind. |
| `ENVIRONMENT` | `development` | Setting to `production` disables auto-reload and enables production logging. |
| `ARTIFACTS_DIR` | `artifacts` | Directory containing the 3 model subdirectories (`structured_model`, `text_transformer`, `fusion`). |
| `HF_MODEL_REPO_ID` | `ParminderzHuggingFace/sentinel-workforce-risk-models` | Hugging Face model repository ID used for downloading artifacts if missing locally. |
| `TORCH_NUM_THREADS` | `1` | Restricts PyTorch CPU intra-op thread pool to minimize virtual memory allocations in container environments. |
| `TORCH_NUM_INTEROP_THREADS` | `1` | Restricts PyTorch CPU inter-op thread pool. |
