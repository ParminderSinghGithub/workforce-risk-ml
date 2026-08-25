# Workforce Risk ML System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade multimodal machine learning system for enterprise employee attrition and workforce burnout risk prediction.

The system ingests high-dimensional workforce demographic, compensation, and performance metrics alongside unstructured qualitative feedback text, extracting complementary risk representations through specialized deep learning and NLP models before late-fusion inference.

---

## High-Level System Architecture

```
                                +---------------------------+
                                |  Raw Workforce Analytics  |
                                |  (850K Rows, 1.25 GB JSON)|
                                +-------------+-------------+
                                              |
                                              v
                                +---------------------------+
                                |    PySpark Engineering    |
                                | (ETL, Validation, Splits) |
                                +-------------+-------------+
                                              |
                      +-----------------------+-----------------------+
                      |                                               |
                      v                                               v
        +---------------------------+                   +---------------------------+
        |    Structured Features    |                   |  Employee Feedback Text   |
        |  (17 Numeric + 5 Cat)     |                   |     (recent_feedback)     |
        +-------------+-------------+                   +-------------+-------------+
                      |                                               |
                      v                                               v
        +---------------------------+                   +---------------------------+
        |        PyTorch MLP        |                   |   DistilBERT + PEFT/LoRA  |
        |  (Embedding + BatchNorm)  |                   |   (Sequence Classifier)   |
        +-------------+-------------+                   +-------------+-------------+
                      |                                               |
                      v                                               v
        +---------------------------+                   +---------------------------+
        |       p_structured        |                   |          p_text           |
        |  (Attrition Probability)  |                   |  (Burnout Risk Estimate)  |
        +-------------+-------------+                   +-------------+-------------+
                      |                                               |
                      +-----------------------+-----------------------+
                                              |
                                              v
                                +---------------------------+
                                |      Late Fusion MLP      |
                                |   [p_structured, p_text]  |
                                +-------------+-------------+
                                              |
                                              v
                                +---------------------------+
                                |  Final Attrition Risk (p) |
                                |   & Decision Risk Tier    |
                                +-------------+-------------+
                                              |
                                              v
                                +---------------------------+
                                |  FastAPI + Streamlit App  |
                                |   (Docker on OCI Free)    |
                                +---------------------------+
```

---

## Frozen System Specifications

| Component | Specification |
| :--- | :--- |
| **Dataset Candidate** | [`Umer112233/employee-burnout-turnover-prediction-800k`](https://huggingface.co/datasets/Umer112233/employee-burnout-turnover-prediction-800k) (849,999 records, 1.25 GB) |
| **Structured Target** | `left_company` (Binary classification: `71.47% False` / `28.53% True`) |
| **Text Feature Field**| `recent_feedback` (Qualitative employee commentary) |
| **Text Learning Task**| Binary High-Burnout Risk Classification ($P(\text{burnout\_risk} \ge 0.75 \mid \text{feedback})$) |
| **Structured Model** | PyTorch Multi-Layer Perceptron (Tabular Embedding + Dense + BatchNorm + Dropout) |
| **Text Model** | `distilbert-base-uncased` fine-tuned via **PEFT / LoRA** ($r=16, \alpha=32$) |
| **Multimodal Fusion**| Late Fusion MLP combining $[p_{\text{structured}}, p_{\text{text}}]$ into calibrated attrition risk |
| **Data Processing** | PySpark distributed pipeline for data ingestion, cleaning, and Parquet caching |
| **Serving & Backend**| FastAPI REST API with Pydantic schemas |
| **User Interface** | Streamlit interactive risk simulator and scenario analyzer |
| **Deployment** | Docker containerized deployment on Oracle Cloud Infrastructure (OCI) Always Free tier |
| **Model Registry** | Hugging Face Model Hub for adapter and checkpoint distribution |

---

## Leakage Prevention Protocol

The following attributes are strictly excluded from structured feature inputs:

- `employee_id` — Non-predictive identifier
- `turnover_reason` — Direct post-exit target leakage
- `turnover_probability_generated` — Synthetic generator target leakage
- `risk_factors_summary` — Rule-derived synthetic summary category
- `burnout_risk` — Reserved as intermediate text-signal training target

---

## Explicit Scope Boundaries

This repository is an engineering-first, production-oriented portfolio ML system. To ensure high reliability, maintainability, and clean architecture, the following are **explicitly out of scope**:

- ❌ RAG / Autonomous Agents (LangChain, AutoGen)
- ❌ Heavy streaming engines (Kafka)
- ❌ Heavy orchestrators (Airflow, Kubernetes)
- ❌ Heavy database clusters (PostgreSQL, MongoDB)
- ❌ Paid third-party APIs (OpenAI, AWS SageMaker)
- ❌ Microservice over-engineering

---

## Project Structure

```
workforce-risk-ml-system/
├── configs/
│   └── config.yaml             # Central frozen project configuration
├── data/
│   ├── raw/                    # Raw source dataset (git-ignored)
│   ├── processed/              # PySpark processed Parquet files (git-ignored)
│   └── splits/                 # Train / validation / test splits (git-ignored)
├── docs/
│   └── PROJECT_SPEC.md         # Detailed implementation specifications
├── src/
│   └── workforce_risk/
│       ├── __init__.py
│       ├── config.py           # Pydantic configuration loader & validator
│       └── utils/
│           ├── __init__.py
│           └── seed.py         # Global deterministic seed utility
├── tests/
│   └── test_config.py          # Configuration & utility unit tests
├── .dockerignore
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quickstart (Foundation Setup)

```bash
# 1. Clone repository
git clone https://github.com/ParminderSinghGithub/workforce-risk-ml.git
cd workforce-risk-ml

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install foundation dependencies
pip install -r requirements.txt

# 4. Run test suite
pytest

# 5. Run end-to-end multimodal inference smoke test
python scripts/predict_sample.py
```

---

## Programmatic Inference Quickstart

```python
from workforce_risk.inference import WorkforceRiskPredictor, EmployeeInput

# 1. Initialize predictor from saved disk artifacts (offline inference ready)
predictor = WorkforceRiskPredictor.from_artifacts()

# 2. Define employee profile
employee = EmployeeInput(
    employee_id="EMP-1001",
    department="Engineering",
    job_level="Senior",
    role="Senior Software Engineer",
    tenure_months=36.0,
    salary=135000.0,
    performance_score=0.88,
    satisfaction_score=0.85,
    workload_score=0.45,
    team_sentiment=0.82,
    stress_level=0.30,
    recent_feedback="Great quarter! Feeling very supported by management and love the project direction.",
)

# 3. Predict attrition risk
result = predictor.predict_single(employee)
print(result.to_dict())
```

---

## FastAPI Serving & HTTP Inference API

The system includes a production-grade FastAPI serving layer that loads the trained models once on application startup and performs offline multimodal inference.

### 1. Launch the Serving API

```bash
python scripts/serve.py --host 127.0.0.1 --port 8000
```
Interactive OpenAPI/Swagger documentation is automatically available at `http://127.0.0.1:8000/docs`.

### 2. Health & Model Readiness Check

```bash
curl -X GET http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "device": "cpu",
  "models_loaded": {
    "structured_mlp": true,
    "text_distilbert_lora": true,
    "multimodal_late_fusion": true
  },
  "decision_threshold": 0.2189,
  "offline_mode": true
}
```

### 3. Single-Employee Prediction Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "employee_id": "EMP-1001",
    "department": "Engineering",
    "job_level": "Senior",
    "role": "Senior Software Engineer",
    "tenure_months": 36.0,
    "salary": 135000.0,
    "performance_score": 0.88,
    "satisfaction_score": 0.85,
    "workload_score": 0.45,
    "team_sentiment": 0.82,
    "stress_level": 0.30,
    "recent_feedback": "Great quarter! Feeling very supported by management and love the project direction."
  }'
```

**Response:**
```json
{
  "employee_id": "EMP-1001",
  "fused_risk_probability": 0.2403,
  "structured_risk_probability": 0.4449,
  "text_risk_probability": 0.2778,
  "risk_prediction": 1,
  "risk_tier": "ELEVATED",
  "decision_threshold": 0.2189,
  "modality_breakdown": {
    "structured_weight": 0.181,
    "text_weight": 0.147,
    "intercept": -0.9703,
    "structured_logit": -0.2215,
    "text_logit": -0.9554,
    "structured_contribution": -0.0401,
    "text_contribution": -0.1404
  },
  "summary": "Employee EMP-1001 classified as ELEVATED RISK (Fused Risk Probability: 24.03%, Decision Threshold: 0.22). Structured Signal: 44.49%, Text Burnout Signal: 27.78%."
}
```


