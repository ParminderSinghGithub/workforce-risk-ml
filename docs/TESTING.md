# Testing & Verification Guide

This document describes the test suite structure, test coverage categories, and local verification commands for the Sentinel codebase.

---

## 1. Test Suite Organization

The test suite is built using `pytest` and structured across 9 dedicated test modules in `tests/`:

| Test Module | Coverage Area | Key Verification Properties |
| :--- | :--- | :--- |
| `tests/test_config.py` | Configuration Validation | Validates YAML parsing, schema constraints, leakage exclusion lists, seed reproducibility. |
| `tests/test_data_pipeline.py` | Data Engineering | Validates raw schema parsing, synthetic cleaning, missing value handling, column types. |
| `tests/test_features.py` | Feature Engineering | Validates categorical indexing, numerical scaling, stratified and grouped splitting logic. |
| `tests/test_structured_model.py`| PyTorch Tabular Model | Validates `StructuredMLP` forward pass, gradient updates, preprocessing isolation, threshold sweeps. |
| `tests/test_text_baseline.py` | TF-IDF Baseline | Validates TF-IDF feature extraction, LogisticRegression baseline fitting, and serialization. |
| `tests/test_text_transformer.py`| Transformer NLP | Validates tokenization, LoRA adapter initialization, forward pass, trainable parameter counts. |
| `tests/test_fusion.py` | Late Fusion Model | Validates `safe_logit` stability, meta-regression fitting, probability bounds, serialization. |
| `tests/test_inference.py` | Inference Engine | Validates `WorkforceRiskPredictor`, single/batch predictions, Hugging Face download fallback. |
| `tests/test_serving.py` | FastAPI Endpoints | Validates `/health`, `/model-info`, `/predict`, `/predict/batch`, static SPA mounting, validation errors. |

---

## 2. Running Tests

### 2.1 Run the Full Test Suite
```bash
pytest tests/ -v
```

### 2.2 Run Serving and Inference Tests
```bash
pytest tests/test_inference.py tests/test_serving.py -v
```

### 2.3 Run Multimodal Fusion Tests
```bash
pytest tests/test_fusion.py -v
```

---

## 3. Verified Test Status

The full repository test suite was executed and verified:
- **Total Tests**: **57 passed**
- **Test Duration**: ~22.4 seconds
- **Pass Rate**: **100% (57/57)**
