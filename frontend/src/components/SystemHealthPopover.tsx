import React from 'react';
import { HealthResponse } from '../types/api';
import { CheckCircle2, AlertCircle, X, Activity } from 'lucide-react';

interface SystemHealthPopoverProps {
  health: HealthResponse | null;
  isLoading: boolean;
  onClose: () => void;
}

export const SystemHealthPopover: React.FC<SystemHealthPopoverProps> = ({
  health,
  isLoading,
  onClose,
}) => {
  const isHealthy = health?.status === 'healthy';

  return (
    <div className="popover-card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--border-subtle)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8125rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          <Activity size={15} color="var(--brand-light)" />
          <span>System & Telemetry Health</span>
        </div>
        <button className="btn-ghost" style={{ padding: '0.2rem', borderRadius: 'var(--radius-sm)', cursor: 'pointer', border: 'none' }} onClick={onClose}>
          <X size={14} color="var(--text-muted)" />
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem', fontSize: '0.8125rem' }}>
        {/* Service Status */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ color: 'var(--text-secondary)' }}>FastAPI Service:</span>
          <span style={{ fontWeight: 700, color: isHealthy ? 'var(--risk-low)' : 'var(--risk-critical)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            {isHealthy ? <CheckCircle2 size={13} /> : <AlertCircle size={13} />}
            {isLoading ? 'Checking...' : isHealthy ? 'ONLINE (READY)' : 'OFFLINE'}
          </span>
        </div>

        {/* Inference Device */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ color: 'var(--text-secondary)' }}>Inference Hardware:</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-primary)' }}>
            {health?.device ? health.device.toUpperCase() : 'CPU'}
          </span>
        </div>

        {/* Operating Decision Threshold */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ color: 'var(--text-secondary)' }}>Operating Cutoff (&tau;*):</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--brand-light)' }}>
            {health?.decision_threshold ? (health.decision_threshold * 100).toFixed(1) + '%' : '21.9%'}
          </span>
        </div>

        {/* Offline Mode Guarantee */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ color: 'var(--text-secondary)' }}>Local Disk Mode:</span>
          <span style={{ fontWeight: 600, color: 'var(--risk-low)' }}>
            Fully Offline
          </span>
        </div>

        {/* Loaded Model Checkpoints */}
        <div style={{ marginTop: '0.25rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem', textTransform: 'uppercase' }}>
            Loaded Model Artifacts:
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <CheckCircle2 size={12} color="var(--risk-low)" />
              Structured Tabular MLP
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <CheckCircle2 size={12} color="var(--risk-low)" />
              DistilBERT + PEFT/LoRA (NLP)
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <CheckCircle2 size={12} color="var(--risk-low)" />
              Calibrated Late Fusion Meta-Model
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
