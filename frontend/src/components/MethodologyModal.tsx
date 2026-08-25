import React from 'react';
import { X, Shield, Cpu, Database, FileText, Layers, Award, CheckCircle2, AlertCircle } from 'lucide-react';

interface MethodologyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const MethodologyModal: React.FC<MethodologyModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-container" style={{ maxWidth: '920px' }} onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
            <div style={{ padding: '0.4rem', borderRadius: 'var(--radius-md)', backgroundColor: 'var(--bg-surface-raised)', color: 'var(--brand-light)' }}>
              <Cpu size={20} />
            </div>
            <div>
              <div style={{ fontSize: '1.125rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                Sentinel Architecture & Methodology Audit
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Verified ML benchmarks, strict leakage prevention guarantees, and holdout evaluations
              </div>
            </div>
          </div>
          <button className="btn-ghost" style={{ padding: '0.35rem', borderRadius: 'var(--radius-sm)', border: 'none', cursor: 'pointer' }} onClick={onClose}>
            <X size={18} color="var(--text-muted)" />
          </button>
        </div>

        {/* Body */}
        <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', maxHeight: '75vh', overflowY: 'auto' }}>
          {/* Benchmark Table */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                <Award size={16} color="var(--risk-elevated)" />
                <span>Audited Holdout Test Benchmarks</span>
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Evaluated blindly on untouched partitions with zero training overlap
              </span>
            </div>

            <div className="data-table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model & Branch</th>
                    <th>Target</th>
                    <th>Holdout (N)</th>
                    <th>Operating Cutoff (&tau;*)</th>
                    <th>ROC-AUC</th>
                    <th>PR-AUC</th>
                    <th>Log-Loss</th>
                    <th>Brier Score</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>1. Structured Tabular MLP [128, 64, 32]</strong></td>
                    <td style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>left_company</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>85,096</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.2469</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>0.5755</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>0.3313</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.5899</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.2008</td>
                  </tr>
                  <tr>
                    <td>2. HistGradientBoosting (GBDT Baseline)</td>
                    <td style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>left_company</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>85,096</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.2334</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.5766</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.3319</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.5894</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.2005</td>
                  </tr>
                  <tr>
                    <td>3. DistilBERT + PEFT/LoRA (NLP Transformer)</td>
                    <td style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#fb923c' }}>high_burnout_risk</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>85,197</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.3530</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--brand-light)' }}>0.7363</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--brand-light)' }}>0.7565</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.6099</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.2079</td>
                  </tr>
                  <tr>
                    <td>4. TF-IDF Text Baseline</td>
                    <td style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#fb923c' }}>high_burnout_risk</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>85,197</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.2918</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.6673</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.7351</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.7041</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.2458</td>
                  </tr>
                  <tr style={{ backgroundColor: 'rgba(37, 99, 235, 0.12)', borderLeft: '3px solid var(--brand-primary)' }}>
                    <td><strong style={{ color: 'var(--brand-light)' }}>5. Sentinel Multimodal Late Fusion</strong></td>
                    <td style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>left_company</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>8,463</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>0.2313</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--brand-light)' }}>0.5719</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--brand-light)' }}>0.3387</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--risk-low)' }}>0.5942</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--risk-low)' }}>0.2026</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              <AlertCircle size={13} color="var(--brand-light)" />
              <span>
                <strong>Operating Point Sensitivity:</strong> At validation-fixed threshold &tau;* = 0.2313, Sentinel Late Fusion captures <strong>86.60% (2,113 of 2,440)</strong> of true voluntary employee exits on the aligned holdout test partition.
              </span>
            </div>
          </div>

          {/* Mathematical Modeling Specification */}
          <div className="grid-3" style={{ gap: '1rem' }}>
            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8125rem', fontWeight: 700, color: '#60a5fa', marginBottom: '0.35rem' }}>
                <Database size={14} />
                Tabular Embedding MLP
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                380 encoded features (24 continuous dimensions + 356 one-hot categorical dimensions) processed through [128, 64, 32] hidden layers with Dropout(0.20) and LayerNorm.
              </div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8125rem', fontWeight: 700, color: '#fb923c', marginBottom: '0.35rem' }}>
                <FileText size={14} />
                DistilBERT + LoRA (NLP)
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Fine-tuned transformer (r=16, &alpha;=32) trained on 679,814 reviews to detect subjective workplace distress and burnout risk (<span style={{ fontFamily: 'var(--font-mono)' }}>high_burnout_risk</span>).
              </div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8125rem', fontWeight: 700, color: 'var(--brand-light)', marginBottom: '0.35rem' }}>
                <Layers size={14} />
                Late Fusion Meta-Regression
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5, fontFamily: 'var(--font-mono)' }}>
                logit(p) = 0.0094 + 1.0471&middot;logit(p₁) + 0.0272&middot;logit(p₂)
              </div>
            </div>
          </div>

          {/* Leakage Prevention Guarantees */}
          <div style={{ backgroundColor: 'var(--bg-surface-raised)', padding: '1rem 1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8125rem', fontWeight: 700, color: 'var(--risk-low)', marginBottom: '0.5rem' }}>
              <Shield size={15} />
              Enterprise Data Leakage & Integrity Controls
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.35rem' }}>
                <CheckCircle2 size={13} color="var(--risk-low)" style={{ flexShrink: 0, marginTop: '0.15rem' }} />
                <span><strong>Target Proxy Exclusion:</strong> `turnover_reason` and `turnover_probability_generated` permanently removed from all feature pipelines.</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.35rem' }}>
                <CheckCircle2 size={13} color="var(--risk-low)" style={{ flexShrink: 0, marginTop: '0.15rem' }} />
                <span><strong>Grouped Template Splitting:</strong> 0% feedback text template memorization across train, validation, and test splits.</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.35rem' }}>
                <CheckCircle2 size={13} color="var(--risk-low)" style={{ flexShrink: 0, marginTop: '0.15rem' }} />
                <span><strong>Dual Holdout Disjointness:</strong> Aligned holdout (N = 8,463) has 0 employee overlap with structured train and 0 template overlap with text train.</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.35rem' }}>
                <CheckCircle2 size={13} color="var(--risk-low)" style={{ flexShrink: 0, marginTop: '0.15rem' }} />
                <span><strong>Target Independence:</strong> DistilBERT is optimized for burnout detection (0.7363 ROC-AUC), while Sentinel Fusion predicts employee attrition (0.5719 ROC-AUC).</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{ padding: '1rem 1.5rem', borderTop: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={onClose}>
            Close Audit
          </button>
        </div>
      </div>
    </div>
  );
};
