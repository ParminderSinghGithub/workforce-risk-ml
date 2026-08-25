# Sentinel: Final Machine Learning Evaluation & Model Selection Report

---

## 1. Executive Summary

**Sentinel** is a multimodal workforce risk intelligence platform designed to assess employee attrition risk while monitoring organizational distress and burnout.

This report documents the final, audited machine learning benchmarks for Sentinel across its tabular, NLP, and multimodal late-fusion layers. All reported metrics are traceable to underlying checkpoint artifacts, test predictions, and reproducible evaluation routines, with strict separation between distinct evaluation targets.

---

## 2. Prediction Targets & Data Partitioning

### 2.1 Target Definitions
The platform utilizes two distinct, complementary targets:
1. **Employee Attrition (`left_company`)**: Binary indicator predicting voluntary employee departure ($y \in \{0, 1\}$). Overall corpus base rate: **$28.43\%$**.
2. **Workplace Burnout Risk (`high_burnout_risk`)**: Binary indicator derived from unprompted text feedback reflecting acute workplace distress ($y \in \{0, 1\}$). Overall corpus base rate: **$58.07\%$**.

> **Target Correlation & Semantics**: The Pearson correlation between `high_burnout_risk` and `left_company` across the dataset is **$r = 0.0987$** ($R^2 \approx 1\%$). Subjective feedback sentiment serves as an early indicator of workplace friction rather than a direct surrogate for employee departure. Consequently, NLP performance on burnout ($0.7363$ ROC-AUC) is distinct from attrition prediction.

### 2.2 Dataset Partitions & Leakage Controls
The source dataset consists of **849,999 records** partitioned into non-overlapping splits:

| Modality Split | Partition Strategy | Training ($N$) | Validation ($N$) | Holdout Test ($N$) | Leakage Enforcement |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Structured Tabular** | Stratified random on `left_company` | 679,662 | 85,241 | 85,096 | Direct outcome proxies excluded (`turnover_reason`, `turnover_probability_generated`). Preprocessor fitted exclusively on train. |
| **Text Feedback** | Group-disjoint on `template_id` | 679,814 | 84,988 | 85,197 | Zero feedback template overlap between train and test. |
| **Aligned Dual Holdout** | Holdout intersection on `employee_id` | — | 8,354 | **8,463** | **Zero employee overlap** with structured train; **Zero template overlap** with text train. |

---

## 3. Tabular Model Selection & Architecture Experiments

To establish whether tabular attrition performance was constrained by neural network topology or by the intrinsic signal in the leakage-safe feature set, multiple architectures were evaluated on the full 679,662-row training partition:

| Model Architecture | Configuration | Val ROC-AUC | Optimal $\tau^*$ | Test ROC-AUC | Test PR-AUC | Test Log-Loss | Test Brier Score | Test Recall (@ $\tau^*$) | Test Precision (@ $\tau^*$) | Test F1 (@ $\tau^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (GBDT)** | 150 max iter, lr=0.08, 31 leaves | **0.5781** | 0.2334 | **0.5766** | 0.3319 | **0.5894** | **0.2005** | 83.36% | 30.95% | 0.4514 |
| **Structured MLP (Selected)** | `[128, 64, 32]`, drop=0.20, lr=1e-3 | 0.5777 | 0.2469 | **0.5755** | **0.3313** | 0.5899 | 0.2008 | **84.70%** | 30.68% | 0.4507 |
| **MLP Candidate A (Deeper)** | `[256, 128, 64, 32]`, drop=0.15 | 0.5776 | 0.2540 | 0.5753 | 0.3313 | 0.5902 | 0.2008 | 84.74% | 30.79% | **0.4516** |
| **MLP Candidate B (Compact)** | `[64, 32]`, drop=0.30, wd=1e-3 | 0.5779 | 0.2313 | 0.5757 | 0.3317 | 0.5895 | 0.2005 | 85.50% | 30.69% | **0.4517** |
| **MLP Candidate C (Wider)** | `[256, 128, 64]`, drop=0.20, lr=5e-4 | 0.5773 | 0.2280 | 0.5750 | 0.3309 | 0.5897 | 0.2006 | 87.71% | 30.31% | 0.4506 |

### Empirical Findings:
1. **Bounded Signal Ceiling**: The difference in holdout test ROC-AUC between the gradient boosted decision tree ($0.5766$) and the selected MLP ($0.5755$) is $\Delta = +0.0011$. Across all 4 neural network topologies and GBDT, performance remained in a narrow $[0.5750, 0.5766]$ window.
2. **Architecture Chasing Not Justified**: Because no candidate delivered a material improvement ($\ge 0.02$ ROC-AUC), the structured branch performance is characterized as signal-limited under strict leakage constraints rather than architecture-limited.
3. **Selected Tabular Model**: The PyTorch Structured MLP ([`artifacts/structured_model/best_checkpoint.pt`](file:///c:/Projects/Workforce-Risk-ML/artifacts/structured_model/best_checkpoint.pt)) is retained for its seamless PyTorch tensor batching and serving consistency.

---

## 4. NLP Modality Evaluation (DistilBERT + LoRA)

- **Architecture**: `distilbert-base-uncased` with PEFT/LoRA adapters ($r=16, \alpha=32, \text{dropout}=0.05$).
- **Native Task Benchmark ($N = 85,197$, `high_burnout_risk`)**:
  - **ROC-AUC**: **`0.7363`**
  - **PR-AUC**: **`0.7565`** (Base rate: $58.07\%$, Relative Lift: $+30.3\%$)
  - **Optimal Operating Point ($\tau^* = 0.3530$)**: Recall: **86.46%**, Precision: **67.26%**, F1: **0.7565**
  - **Log-Loss**: `0.6099` | **Brier Score**: `0.2079`
- **TF-IDF Baseline ($N = 85,197$)**: ROC-AUC: `0.6673`, PR-AUC: `0.7351`, F1: `0.7399`.
- **Decision on Additional GPU Training**:
  An additional 5-hour GPU run fine-tuning DistilBERT directly on `left_company` was evaluated and rejected. Given the $r = 0.0987$ correlation between feedback text and departure, direct fine-tuning on binary attrition without domain-specific exit interview text risks overfitting template artifacts rather than capturing genuine predictive signal. DistilBERT is correctly preserved for its validated role in **Burnout & Distress Risk Detection**.

---

## 5. Multimodal Late Fusion Evaluation

### 5.1 Late Fusion Meta-Model
A calibrated logistic meta-regressor combines logit-transformed unimodal probabilities:
$$\text{logit}(P(\text{left\_company})) = 0.0094 + 1.0471 \cdot \text{logit}(P_{\text{structured}}) + 0.0272 \cdot \text{logit}(P_{\text{burnout\_text}})$$

### 5.2 Aligned Dual-Holdout Benchmark Matrix ($N = 8,463$, Target: `left_company`)

| Model Branch | Modality | ROC-AUC | PR-AUC | Log-Loss | Brier Score | Recall (@ $\tau^*$) | Precision (@ $\tau^*$) | F1 (@ $\tau^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Structured Tabular MLP** | Tabular (380 dims) | 0.5712 | 0.3346 | 0.5943 | 0.2026 | 85.45% | 30.45% | 0.4490 |
| **TF-IDF Baseline (Out-of-Domain)** | Text NLP | 0.5199 | 0.3007 | 1.1337 | 0.3871 | 94.96% | 28.95% | 0.4437 |
| **DistilBERT + LoRA (Out-of-Domain)** | Text NLP | 0.5452 | 0.3189 | 1.0089 | 0.3554 | 100.0% | 28.83% | 0.4476 |
| **Sentinel Multimodal Late Fusion** | **Tabular + Text** | **`0.5719`** | **`0.3387`** | **`0.5942`** | **`0.2026`** | **`86.60%`** | **`30.29%`** | **`0.4488`** |

*Operating Point: Optimal threshold $\tau^* = 0.2313$ was selected strictly on validation log-odds and applied blindly to the holdout test partition, capturing **2,113 out of 2,440 actual employee exits**.*

---

## 6. Publication Figures

The following evaluation figures are generated in [`reports/figures/`](file:///c:/Projects/Workforce-Risk-ML/reports/figures):
1. [`reports/figures/model_selection_comparison.png`](file:///c:/Projects/Workforce-Risk-ML/reports/figures/model_selection_comparison.png): Tabular architecture and GBDT benchmark comparison.
2. [`reports/figures/roc_auc_comparison.png`](file:///c:/Projects/Workforce-Risk-ML/reports/figures/roc_auc_comparison.png): System-wide ROC-AUC benchmark distinguishing native vs. attrition targets.
3. [`reports/figures/pr_auc_comparison.png`](file:///c:/Projects/Workforce-Risk-ML/reports/figures/pr_auc_comparison.png): System-wide PR-AUC benchmark with baseline prevalences.
4. [`reports/figures/fusion_roc_pr_curves.png`](file:///c:/Projects/Workforce-Risk-ML/reports/figures/fusion_roc_pr_curves.png): Exact ROC and Precision-Recall curves on the aligned holdout partition ($N = 8,463$).

---

## 7. Methodological Limitations & Interview Context

1. **Feature Signal Bounds**: Synthetic workforce attributes exhibit high residual variance when direct outcome proxies (`turnover_reason`, `turnover_probability_generated`) are excluded. The resulting $\approx 0.576$ tabular ROC-AUC reflects genuine signal isolation.
2. **Target Asymmetry**: DistilBERT evaluates sentiment distress ($0.7363$ ROC-AUC), which does not directly determine turnover. The late-fusion meta-model applies logistic log-odds weighting ($w_{\text{struct}} = 1.0471, w_{\text{text}} = 0.0272$) to calibrate the contribution of text sentiment.
3. **Reproducibility Guarantee**: All test metrics were computed on untouched holdout partitions ($N=85,096$ structured, $N=85,197$ text, $N=8,463$ aligned) with zero data leakage.

---

## 8. Test Suite Status

The automated test suite passes with zero failures:
```
================= 56 passed, 19 warnings in 207.54s (0:03:27) =================
```
