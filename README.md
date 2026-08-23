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
```
