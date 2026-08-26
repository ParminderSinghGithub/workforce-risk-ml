# Model Specification & Evaluation Benchmarks

This document details the unimodal model branches, multimodal late-fusion mechanics, calibration, and holdout evaluation metrics for the Sentinel platform.

---

## 1. Modality Specifications

### 1.1 Structured Tabular Features
- **Total Encoded Dimension**: 380 features
- **Continuous Features (24)**: Age, tenure months, salary, salary hike percentage, distance from home, performance rating, job satisfaction score, work-life balance score, monthly overtime hours, training hours, previous companies count, promotions last 2 years, total working years, years with current manager, etc.
- **Categorical Features (8)**: Department, job role, job level, education field, marital status, gender, travel frequency, business unit.
- **Preprocessing Pipeline**: Imputation with column medians, Z-score standard scaling, and one-hot categorical encoding.

### 1.2 Unstructured Qualitative Feedback
- **Input Text**: Qualitative feedback comments, quarterly performance commentary, and open-ended satisfaction survey responses.
- **Tokenization**: Subword WordPiece tokenizer (`distilbert-base-uncased`), maximum sequence length 128 tokens.
- **Classification Target**: High burnout distress indicator (`high_burnout_risk`), derived from psychological sentiment and exhaustion markers.

---

## 2. Model Architectures

### 2.1 Tabular PyTorch MLP (`StructuredMLP`)
- **Topology**:
  - Linear Layer: `380 -> 128`, BatchNorm1d, ReLU, Dropout(0.20)
  - Linear Layer: `128 -> 64`, BatchNorm1d, ReLU, Dropout(0.20)
  - Linear Layer: `64 -> 32`, BatchNorm1d, ReLU, Dropout(0.20)
  - Linear Output: `32 -> 1`
- **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)
- **Optimizer**: AdamW (Learning Rate: `1e-3`, Weight Decay: `1e-4`)

### 2.2 NLP Sequence Classifier (`DistilBertForSequenceClassification` + LoRA)
- **Base Architecture**: 6-layer Transformer, 768 hidden dimension, 12 attention heads (66M parameters).
- **LoRA Configuration**:
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Target Modules: `q_lin`, `v_lin`
  - Trainable Parameters: 589,824 (0.88% of base model weights)
  - Memory Footprint on Disk: 3.55 MB (`adapter_model.safetensors`)

### 2.3 Multimodal Late Fusion (`MultimodalLateFusion`)
- **Meta-Classifier**: Calibrated Logistic Meta-Regression over unimodal log-odds:

$$\text{logit}(P_{\text{exit}}) = 0.0094 + 1.0471 \cdot \text{logit}(P_{\text{structured}}) + 0.0272 \cdot \text{logit}(P_{\text{burnout}})$$

- **Optimal Operating Decision Threshold**: $\tau^* = 0.2313$ (tuned on validation set to capture high-recall voluntary exits).

---

## 3. Holdout Evaluation Benchmarks

All models were evaluated on strict, held-out partitions disjoint from training data.

### 3.1 Tabular Structured Model Performance ($N = 85,096$)
- **Target**: `left_company` (Base Rate: 28.43%)
- **ROC-AUC**: **0.5755**
- **PR-AUC**: **0.3313**
- **Log Loss**: 0.5899
- **Brier Score**: 0.2008
- **Recall at $\tau = 0.2469$**: **84.70%** (20,493 / 24,195 true exits captured)
- **Precision at $\tau = 0.2469$**: 30.68%
- **F1 Score**: 0.4507

### 3.2 DistilBERT + LoRA NLP Model Performance ($N = 85,197$)
- **Target**: `high_burnout_risk` (Base Rate: 58.07%)
- **ROC-AUC**: **0.7363**
- **PR-AUC**: **0.7565**
- **Log Loss**: 0.6099
- **Brier Score**: 0.2079
- **Recall at $\tau = 0.3530$**: **86.46%** (42,777 / 49,477 true burnout cases captured)
- **Precision at $\tau = 0.3530$**: 67.26%
- **F1 Score**: 0.7565

### 3.3 Multimodal Late Fusion Model Performance ($N = 8,463$ Aligned Dual Holdout)
- **Target**: `left_company` (Base Rate: 28.83%)
- **ROC-AUC**: **0.5719**
- **PR-AUC**: **0.3387**
- **Log Loss**: 0.5942
- **Brier Score**: 0.2026
- **Recall at $\tau^* = 0.2313$**: **86.60%** (2,113 / 2,440 true exits captured)
- **Precision at $\tau^* = 0.2313$**: 30.29%
- **F1 Score**: 0.4488

---

## 4. Risk Stratification Scheme

| Risk Tier | Probability Range | Strategic Decision Guidance |
| :--- | :---: | :--- |
| **LOW** | $P < 0.2313$ | Retention healthy. Standard annual compensation and development reviews. |
| **ELEVATED** | $0.2313 \le P < 0.45$ | Early warning. Proactive manager 1-on-1 check-in recommended. |
| **HIGH** | $0.45 \le P < 0.70$ | High probability of exit. Review compensation, workload, and career growth. |
| **CRITICAL** | $P \ge 0.70$ | Severe burnout or imminent resignation. Execute immediate retention plan. |
