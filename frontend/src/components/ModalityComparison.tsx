import React from 'react';
import { ModalityBreakdown } from '../types/api';
import { Database, FileText, Layers } from 'lucide-react';

interface ModalityComparisonProps {
  pStructured: number;
  pText: number;
  pFused: number;
  breakdown?: ModalityBreakdown;
}

export const ModalityComparison: React.FC<ModalityComparisonProps> = ({
  pStructured,
  pText,
  pFused,
  breakdown,
}) => {
  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">
            <Layers size={18} color="var(--brand-accent)" />
            Multimodal Modality Attribution
          </div>
          <div className="card-subtitle">
            Signal breakdown comparing structured tabular telemetry vs. qualitative feedback text
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginTop: '0.5rem' }}>
        {/* Structured Branch */}
        <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            <Database size={14} color="#3b82f6" />
            STRUCTURED TABULAR
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: '#60a5fa', margin: '0.35rem 0' }}>
            {(pStructured * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            PyTorch MLP (17 Numeric + 5 Categorical)
          </div>
          {breakdown && (
            <div style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
              Weight (w₁): <span style={{ color: 'var(--text-primary)' }}>+{breakdown.structured_weight}</span>
              <br />
              Logit Signal: <span style={{ color: 'var(--text-primary)' }}>{breakdown.structured_logit > 0 ? `+${breakdown.structured_logit}` : breakdown.structured_logit}</span>
            </div>
          )}
        </div>

        {/* Text Branch */}
        <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            <FileText size={14} color="#f97316" />
            TEXT BURNOUT SIGNAL
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: '#fb923c', margin: '0.35rem 0' }}>
            {(pText * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            DistilBERT + PEFT/LoRA (Qualitative NLP)
          </div>
          {breakdown && (
            <div style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
              Weight (w₂): <span style={{ color: 'var(--text-primary)' }}>+{breakdown.text_weight}</span>
              <br />
              Logit Signal: <span style={{ color: 'var(--text-primary)' }}>{breakdown.text_logit > 0 ? `+${breakdown.text_logit}` : breakdown.text_logit}</span>
            </div>
          )}
        </div>

        {/* Fused Sentinel Decision */}
        <div style={{ backgroundColor: 'rgba(37, 99, 235, 0.08)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', fontWeight: 700, color: 'var(--brand-accent)' }}>
            <Layers size={14} color="#3b82f6" />
            SENTINEL FUSED RISK
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', margin: '0.35rem 0' }}>
            {(pFused * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Calibrated Logistic Meta-Regression
          </div>
          {breakdown && (
            <div style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid rgba(59, 130, 246, 0.2)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
              Intercept (w₀): <span style={{ color: 'var(--text-primary)' }}>{breakdown.intercept}</span>
              <br />
              Net Contribution: <span style={{ color: '#60a5fa', fontWeight: 600 }}>{(breakdown.structured_contribution + breakdown.text_contribution).toFixed(4)}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
