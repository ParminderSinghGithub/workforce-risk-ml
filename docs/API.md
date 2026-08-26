# API Reference

Sentinel exposes a RESTful FastAPI interface for health checks, model metadata inspection, single-employee risk inference, and batch workforce scoring.

---

## 1. Endpoints Overview

| Method | Path | Summary | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health Check | Validates API status, execution device, and loaded model components. |
| `GET` | `/model-info` | Model Architecture | Exposes model topologies, fusion coefficients, and benchmark metrics. |
| `POST` | `/predict` | Single Prediction | Computes multimodal risk probability and tier for an individual employee. |
| `POST` | `/predict/batch` | Batch Prediction | Computes predictions for a cohort of employees in a single request. |
| `GET` | `/docs` | OpenAPI Swagger | Interactive browser API documentation. |

---

## 2. Endpoint Details

### 2.1 Health Check
- **Endpoint**: `GET /health` (also aliased at `/api/v1/health`)
- **Headers**: `Accept: application/json`

#### Response (`200 OK`):
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
  "decision_threshold": 0.2313,
  "offline_mode": true
}
```

---

### 2.2 Model Architecture & Metadata
- **Endpoint**: `GET /model-info` (also aliased at `/api/v1/model-info`)

#### Response (`200 OK`):
```json
{
  "platform": "Sentinel — Multimodal Workforce Risk Intelligence Platform",
  "version": "0.1.0",
  "architecture": {
    "structured_branch": "PyTorch StructuredMLP (Embedding + BatchNorm + Dense + Dropout)",
    "text_branch": "DistilBERT-base-uncased + PEFT/LoRA (Sequence Classifier, r=16, alpha=32)",
    "fusion_mechanism": "Calibrated Logistic Meta-Regression over unimodal log-odds"
  },
  "evaluation_benchmarks": { ... }
}
```

---

### 2.3 Single Employee Risk Prediction
- **Endpoint**: `POST /predict` (also aliased at `/api/v1/predict`)
- **Headers**: `Content-Type: application/json`

#### Request Body Schema (`EmployeeInput`):
```json
{
  "employee_id": "EMP-1001",
  "department": "Engineering",
  "job_level": "Senior",
  "role": "Senior Backend Engineer",
  "tenure_months": 28.0,
  "salary": 135000.0,
  "satisfaction_score": 0.78,
  "performance_score": 0.85,
  "overtime_hours_avg": 4.5,
  "promotion_last_2years": 1,
  "commute_distance_km": 12.0,
  "remote_ratio": 0.6,
  "recent_feedback": "Strong quarter overall. Team collaboration is productive and project milestones were met on schedule."
}
```

#### Response Body (`200 OK`):
```json
{
  "employee_id": "EMP-1001",
  "fused_risk_probability": 0.2099,
  "risk_tier": "LOW",
  "risk_prediction": 0,
  "decision_threshold": 0.2313,
  "structured_risk_probability": 0.2081,
  "text_risk_probability": 0.3245,
  "modality_breakdown": {
    "structured_weight": 1.0471,
    "text_weight": 0.0272,
    "intercept": 0.0094
  },
  "summary": "Employee EMP-1001 evaluated at LOW attrition risk (20.99% fused probability, below 23.13% threshold)."
}
```

---

### 2.4 Batch Workforce Risk Prediction
- **Endpoint**: `POST /predict/batch` (also aliased at `/api/v1/predict/batch`)
- **Headers**: `Content-Type: application/json`

#### Request Body Schema (`BatchPredictionRequest`):
```json
{
  "employees": [
    {
      "employee_id": "EMP-1001",
      "department": "Engineering",
      "job_level": "Senior",
      "role": "Senior Backend Engineer",
      "tenure_months": 28.0,
      "salary": 135000.0,
      "satisfaction_score": 0.78,
      "performance_score": 0.85,
      "recent_feedback": "Strong quarter overall."
    },
    {
      "employee_id": "EMP-1002",
      "department": "Sales",
      "job_level": "Associate",
      "role": "Account Executive",
      "tenure_months": 14.0,
      "salary": 65000.0,
      "satisfaction_score": 0.25,
      "performance_score": 0.45,
      "recent_feedback": "Exhausted from excessive travel and unattainable quarterly quota targets."
    }
  ]
}
```

#### Response Body (`200 OK`):
```json
{
  "total_records": 2,
  "predictions": [
    {
      "employee_id": "EMP-1001",
      "fused_risk_probability": 0.2099,
      "risk_tier": "LOW",
      "risk_prediction": 0,
      "decision_threshold": 0.2313,
      "structured_risk_probability": 0.2081,
      "text_risk_probability": 0.3245
    },
    {
      "employee_id": "EMP-1002",
      "fused_risk_probability": 0.7412,
      "risk_tier": "CRITICAL",
      "risk_prediction": 1,
      "decision_threshold": 0.2313,
      "structured_risk_probability": 0.7350,
      "text_risk_probability": 0.8812
    }
  ]
}
```
