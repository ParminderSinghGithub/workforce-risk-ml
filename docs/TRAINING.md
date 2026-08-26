# Training & Reproduction Guide

This guide details the training workflow, dataset preparation, GPU acceleration requirements, and model reproduction procedures for the Sentinel platform.

---

## 1. Training Environment & GPU Notebook

Training the full multimodal model suite on the 850,000-record dataset requires GPU acceleration for the DistilBERT Transformer fine-tuning stage.

- **Canonical GPU Training Environment**: Executed on Kaggle using an NVIDIA Tesla T4 GPU.
- **Kaggle GPU Training Notebook**:
  [Workforce Risk ML GPU Training Notebook](https://www.kaggle.com/code/parmindersingh2002/workforce-risk-ml-gpu-training)
- **Hugging Face Model Checkpoints**:
  [Sentinel Model Artifacts Repository](https://huggingface.co/ParminderzHuggingFace/sentinel-workforce-risk-models)

---

## 2. Dataset Pipeline & Splitting Methodology

The corpus comprises 849,999 records split with leakage protection:

1. **Tabular Structured Partitioning**:
   - Total rows: 849,999
   - Stratification: Stratified on target `left_company` (Train: 679,662 rows, Validation: 85,241 rows, Test: 85,096 rows).
2. **Text Feedback Partitioning**:
   - Total rows: 849,999
   - Grouping: Grouped disjoint on `template_id` to prevent syntactic leakage across splits (Train: 679,814 rows, Validation: 84,988 rows, Test: 85,197 rows).
3. **Aligned Dual Holdout**:
   - Validation set: 8,354 rows
   - Test set: 8,463 rows
   - Property: 100% employee-disjoint and template-disjoint across all training partitions.

---

## 3. Training Execution Stages

### Stage 1: Structured Model Training
```bash
python scripts/run_structured_training.py \
  --config configs/train_config.yaml \
  --device cuda  # or cpu for test runs
```
- Fits `TabularPreprocessor` on the training split.
- Trains `StructuredMLP` over 20 epochs with early stopping (patience: 3).
- Saves checkpoint to `artifacts/structured_model/best_checkpoint.pt`.

### Stage 2: DistilBERT + LoRA NLP Training
```bash
python scripts/run_text_training.py \
  --config configs/nlp_config.yaml \
  --device cuda
```
- Attaches LoRA adapter ($r=16, \alpha=32$) to `distilbert-base-uncased`.
- Trains for 3 epochs with AdamW and linear learning rate warmup.
- Saves adapter weights and tokenizer to `artifacts/text_transformer/best_model/`.

### Stage 3: Multimodal Late Fusion
```bash
python scripts/run_fusion.py \
  --config configs/fusion_config.yaml
```
- Generates out-of-fold unimodal log-odds for structured and text models on the aligned validation set.
- Fits `MultimodalLateFusion` logistic meta-regression model.
- Sweeps threshold $\tau \in [0.05, 0.95]$ to determine optimal decision boundary $\tau^* = 0.2313$.
- Evaluates on the aligned test split ($N = 8,463$) and saves `artifacts/fusion/fusion_model.joblib`.
