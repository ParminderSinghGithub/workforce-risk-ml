# Workforce Risk ML System — Frozen Technical Specification

## 1. Project Objective

Build an end-to-end, production-grade multimodal machine learning system for enterprise employee attrition and workforce burnout risk prediction. The system ingests high-dimensional workforce tabular data alongside qualitative employee feedback text, extracts calibrated risk signals through specialized models, fuses the multimodal representations, and serves real-time inference via a FastAPI backend and interactive Streamlit frontend deployed on OCI Always Free compute.

---

## 2. Dataset Specification

- **Hugging Face Repository**: `Umer112233/employee-burnout-turnover-prediction-800k`
- **Git Revision / Commit SHA**: `8d35f5bdcd0b7ff0ea04a5d5e93132eaae630e52`
- **File & Format**: `synthetic-employee-dataset.json` (1.25 GB, JSON array)
- **Total Records**: `849,999` rows, `31` columns
- **Modality**: Multimodal (Tabular metrics + Natural language feedback strings)

---

## 3. Target Formulations

### Primary Structured Target
- **Column**: `left_company`
- **Type**: Binary (`bool`)
- **Distribution**: `False` (71.47%) / `True` (28.53%)
- **Objective**: Predict whether an employee will voluntarily or involuntarily depart the organization.

### Text Risk Target (Intermediate Workforce Distress Modality)
- **Column**: `recent_feedback` $\to$ `high_burnout_risk`
- **Type**: Binary indicator derived from continuous `burnout_risk`
- **Threshold Rule**:
  $$\text{high\_burnout\_risk} = \begin{cases} 1 & \text{if } \text{burnout\_risk} \ge 0.75 \\ 0 & \text{if } \text{burnout\_risk} < 0.75 \end{cases}$$
- **Objective**: Extract a calibrated workforce distress probability $p_{\text{text}} \in [0, 1]$ from employee review commentary.

---

## 4. Mandatory Leakage Exclusions

The following 5 columns are strictly prohibited from structured model feature inputs:

1. `employee_id` — Non-predictive unique identifier.
2. `turnover_reason` — 100% deterministic post-exit attribute (`"Not Applicable"` when retained; reasons populated upon departure).
3. `turnover_probability_generated` — Synthetic generator internal probability.
4. `risk_factors_summary` — Rule-derived synthetic summary category.
5. `burnout_risk` — Reserved strictly as the training target for the intermediate text NLP model; excluded from primary tabular features to prevent synthetic collinearity.

---

## 5. Dataset Splitting & Sampling Strategy

### Structured Pipeline Split
- **Method**: Stratified Random Split on `left_company`
- **Proportions**: `80% Train` / `10% Validation` / `10% Test`
- **Training Sample Target**: `100,000` records for PyTorch tabular training; `849,999` records for PySpark ETL demonstration.

### Text Pipeline Split
- **Method**: Group Template Split on `recent_feedback` (1,000 unique review templates).
- **Leakage Prevention**: Zero template overlap between train, validation, and test partitions.
- **Training Sample Bounds**: `5,000 – 10,000` stratified records across the unique template clusters.

---

## 6. Model Architectures

```
+-------------------------------------------------------------------+
|                        Workforce Risk System                      |
+---------------------------------+---------------------------------+
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
+-----------------------+                   +-----------------------+
|  Structured Features  |                   |    recent_feedback    |
| (17 Numeric + 5 Cat)  |                   |        (Text)         |
+-----------+-----------+                   +-----------+-----------+
            |                                           |
            v                                           v
+-----------------------+                   +-----------------------+
|      PyTorch MLP      |                   | DistilBERT + PEFT/LoRA|
|  (Embedding + Dense)  |                   | (Sequence Classifier) |
+-----------+-----------+                   +-----------+-----------+
            |                                           |
            v                                           v
+-----------------------+                   +-----------------------+
|     p_structured      |                   |        p_text         |
| (Attrition Risk Prob) |                   |  (Burnout Risk Prob)  |
+-----------+-----------+                   +-----------+-----------+
            |                                           |
            +---------------------+---------------------+
                                  |
                                  v
                    +---------------------------+
                    |        Late Fusion        |
                    | [p_structured, p_text]    |
                    |     -> Small MLP Head     |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | Final Attrition Prob (p)  |
                    |      & Risk Tier          |
                    +---------------------------+
```

### 1. Structured Tabular Model
- **Framework**: PyTorch
- **Architecture**: Tabular Multi-Layer Perceptron (Categorical Entity Embeddings + Batch Normalization + Linear Layers + Dropout + ReLU).
- **Output**: Calibrated attrition probability $p_{\text{structured}} \in [0, 1]$.

### 2. Text NLP Model
- **Base Model**: `distilbert-base-uncased` (Hugging Face)
- **Adaptation**: Parameter-Efficient Fine-Tuning via LoRA ($r=16, \alpha=32$, target modules: `q_lin`, `v_lin`, dropout: `0.05`).
- **Output**: Calibrated burnout distress probability $p_{\text{text}} \in [0, 1]$.

### 3. Multimodal Late Fusion Model
- **Input**: Concatenation of probabilities $[p_{\text{structured}}, p_{\text{text}}]$ (2-dimensional vector).
- **Architecture**: Lightweight 2-layer MLP (`Linear(2, 16) -> ReLU -> Linear(16, 8) -> ReLU -> Linear(8, 1) -> Sigmoid`).
- **Output**: Final attrition risk probability $\hat{y} \in [0, 1]$.

---

## 7. Serving & Deployment Architecture

- **Inference API**: FastAPI service with Pydantic request/response validation schemas.
- **Frontend Dashboard**: Streamlit interactive UI featuring:
  - Single employee risk simulator (live sliders, dropdowns, feedback text box).
  - Risk categorization gauge (`Low`, `Medium`, `High`, `Critical`).
  - Multimodal risk decomposition showing structured vs text score contributions.
- **Containerization**: Docker multi-stage build.
- **Deployment Target**: Oracle Cloud Infrastructure (OCI) Always Free ARM / x86 compute instance.
- **Model Distribution**: Hugging Face Model Hub repository for pre-trained weights and LoRA adapters.

---

## 8. Explicit Scope Boundaries & Exclusions

To ensure strict engineering focus, high reliability, and portfolio defensibility, the following technologies and patterns are **explicitly excluded**:

- **NO** Retrieval-Augmented Generation (RAG)
- **NO** Autonomous Agent frameworks (LangChain, AutoGen, CrewAI)
- **NO** Heavy streaming engines (Kafka)
- **NO** Complex orchestrators (Airflow, Kubeflow, Kubernetes)
- **NO** Heavy database infrastructure (PostgreSQL, MongoDB)
- **NO** SHAP runtime computation (in favour of fast feature contribution breakdowns)
- **NO** ONNX runtime conversion overhead
- **NO** Paid cloud APIs (OpenAI, Anthropic, AWS SageMaker)
- **NO** Multi-repo / unnecessary microservice sprawl
