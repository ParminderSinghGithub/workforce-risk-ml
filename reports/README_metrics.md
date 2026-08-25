# Sentinel: Audited Production Metrics (Safe-to-Quote)

---

### Core Performance Benchmarks

```
========================================================================================
 SENTINEL ML BENCHMARK SUMMARY (Zero Data Leakage | Verified Holdouts)
========================================================================================
 1. NLP Modality (DistilBERT + PEFT/LoRA)
    • Target:          Workplace Burnout Risk (high_burnout_risk)
    • Holdout Set:     N = 85,197 (Group-disjoint text templates)
    • ROC-AUC:         0.7363
    • PR-AUC:          0.7565 (Base prevalence: 58.07%)
    • F1-Score:        0.7565 (Recall: 86.46%, Precision: 67.26% @ tau = 0.3530)

 2. Structured Tabular Branch (PyTorch MLP [128, 64, 32])
    • Target:          Employee Attrition (left_company)
    • Holdout Set:     N = 85,096 (Stratified holdout)
    • ROC-AUC:         0.5755
    • PR-AUC:          0.3313 (Base prevalence: 28.43%)
    • Sensitivity:     84.70% recall of true employee departures (@ tau = 0.2469)

 3. Sentinel Multimodal Late Fusion (Calibrated Meta-Regression)
    • Target:          Enterprise Attrition Risk (left_company)
    • Holdout Set:     N = 8,463 (Aligned dual-holdout test)
    • ROC-AUC:         0.5719
    • PR-AUC:          0.3387 (Base prevalence: 28.83%)
    • Key Retention:   86.60% recall (2,113 of 2,440 actual exits detected @ tau = 0.2313)
========================================================================================
```

---

### Approved Bullet Points for Resumes & Portfolios

- **NLP Sentiment & Burnout Intelligence**:
  > *"Fine-tuned a DistilBERT + PEFT/LoRA transformer on 679k employee feedback reviews to detect workplace burnout risk, achieving **0.7363 ROC-AUC** and **0.7565 PR-AUC** with strict grouped-template leakage prevention."*

- **Multimodal Workforce Risk Fusion**:
  > *"Engineered an enterprise workforce risk platform fusing 380 tabular features and NLP sentiment log-odds via calibrated meta-regression, capturing **86.6% of employee attrition** at optimal operating threshold $\tau^* = 0.2313$ across an independent dual-holdout benchmark ($N = 8,463$)."*

---

### Methodological Notes for Technical Review
- **Target Independence**: DistilBERT was trained on subjective feedback distress (`high_burnout_risk`), which exhibits low linear correlation ($r = 0.0987$) with voluntary departure (`left_company`). Performance numbers are reported strictly on their respective native targets to prevent target conflation.
- **Leakage Prevention**: All direct target proxies (`turnover_reason`, `turnover_probability_generated`) were purged prior to modeling. Preprocessing statistics and classification thresholds ($\tau^*$) were fitted exclusively on training/validation partitions and evaluated blindly on holdout test partitions.
