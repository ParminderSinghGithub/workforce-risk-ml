import React from 'react';
import { X, Shield, Cpu, Database, FileText, Layers, Award, CheckCircle2 } from 'lucide-react';

interface MethodologyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const MethodologyModal: React.FC<MethodologyModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
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
                Technical specification, leakage prevention guarantees, and holdout benchmarks
              </div>
            </div>
          </div>
          <button className="btn-ghost" style={{ padding: '0.35rem', borderRadius: 'var(--radius-sm)', border: 'none', cursor: 'pointer' }} onClick={onClose}>
            <X size={18} color="var(--text-muted)" />
          </button>
        </div>

        {/* Body */}
        <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Benchmark Table */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
              <Award size={16} color="var(--risk-elevated)" />
              <span>Untouched Holdout Test Partition Benchmark ($N = 8,463$)</span>
            </div>
            <div className="data-table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Architecture</th>
                    <th>Operating Cutoff (&tau;*)</th>
                    <th>ROC-AUC</th>
                    <th>PR-AUC</th>
                    <th>Log-Loss</th>
                    <th>Brier Score</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>1. Structured Tabular MLP</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.42</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.5106</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.2985</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.6584</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.2327</td>
                  </tr>
                  <tr>
                    <td>2. TF-IDF Text Baseline</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.18</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.5199</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.3007</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>1.1337</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.3871</td>
                  </tr>
                  <tr>
                    <td>3. DistilBERT + PEFT/LoRA</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.15</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.5452</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.3189</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>1.0089</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>0.3554</td>
                  </tr>
                  <tr style={{ backgroundColor: 'rgba(37, 99, 235, 0.1)', borderLeft: '3px solid var(--brand-primary)' }}>
                    <td><strong style={{ color: 'var(--brand-light)' }}>4. Sentinel Calibrated Late Fusion</strong></td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>0.22</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--brand-light)' }}>0.5452</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--brand-light)' }}>0.3196</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--risk-low)' }}>0.5983</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--risk-low)' }}>0.2042</td>
                  </tr>
                </tbody>
              </table>
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
                17 continuous telemetry metrics + 5 frequency-indexed categorical embeddings standardized strictly on the training partition.
              </div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8125rem', fontWeight: 700, color: '#fb923c', marginBottom: '0.35rem' }}>
                <FileText size={14} />
                DistilBERT + LoRA (NLP)
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Fine-tuned sequence classifier (r=16, &alpha;=32) trained on 679,814 review comments evaluating psychological burnout indicators.
              </div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8125rem', fontWeight: 700, color: 'var(--brand-light)', marginBottom: '0.35rem' }}>
                <Layers size={14} />
                Late Fusion Meta-Regression
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5, fontFamily: 'var(--font-mono)' }}>
                logit(p) = -0.9703 + 0.1810&middot;logit(p₁) + 0.1470&middot;logit(p₂)
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
                <span><strong>Target Feature Exclusion:</strong> `turnover_reason` and `turnover_probability_generated` permanently removed from inputs.</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.35rem' }}>
                <CheckCircle2 size={13} color="var(--risk-low)" style={{ flexShrink: 0, marginTop: '0.15rem' }} />
                <span><strong>Grouped Template Splitting:</strong> 0% feedback text template memorization across train, validation, and test.</span>
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
