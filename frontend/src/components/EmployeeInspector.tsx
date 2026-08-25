import React, { useState } from 'react';
import { X, Play, ChevronDown, ChevronUp, MessageSquare, AlertTriangle, Layers, Database, FileText } from 'lucide-react';
import { EmployeePredictionRequest, PredictionResponse } from '../types/api';
import { RiskScore } from './RiskScore';

interface EmployeeInspectorProps {
  employee: EmployeePredictionRequest | null;
  prediction: PredictionResponse | null;
  isOpen: boolean;
  onClose: () => void;
  onSendToSimulator: (employee: EmployeePredictionRequest) => void;
}

export const EmployeeInspector: React.FC<EmployeeInspectorProps> = ({
  employee,
  prediction,
  isOpen,
  onClose,
  onSendToSimulator,
}) => {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState<boolean>(false);

  if (!isOpen || !employee || !prediction) return null;

  // Extract human-readable risk drivers
  const riskDrivers: { label: string; detail: string; severity: 'high' | 'medium' | 'positive' }[] = [];

  if (employee.workload_score > 0.75) {
    riskDrivers.push({
      label: 'Extreme Workload Burden',
      detail: `Reported workload at ${(employee.workload_score * 100).toFixed(0)}% capacity`,
      severity: 'high',
    });
  }
  if (employee.overtime_hours >= 10) {
    riskDrivers.push({
      label: 'Excessive Overtime',
      detail: `${employee.overtime_hours} hours/week of overtime logged`,
      severity: 'high',
    });
  }
  if (employee.satisfaction_score < 0.45) {
    riskDrivers.push({
      label: 'Low Job Satisfaction',
      detail: `Satisfaction scored at ${(employee.satisfaction_score * 100).toFixed(0)}%`,
      severity: 'high',
    });
  }
  if (employee.stress_level > 0.7) {
    riskDrivers.push({
      label: 'Elevated Stress Index',
      detail: `Reported stress index at ${(employee.stress_level * 100).toFixed(0)}%`,
      severity: 'high',
    });
  }
  if (employee.team_sentiment < 0.5) {
    riskDrivers.push({
      label: 'Depressed Team Sentiment',
      detail: `Team sentiment indicator at ${(employee.team_sentiment * 100).toFixed(0)}%`,
      severity: 'medium',
    });
  }
  if (riskDrivers.length === 0) {
    riskDrivers.push({
      label: 'Healthy Organizational Telemetry',
      detail: 'Balanced workload, steady team sentiment, and sustainable overtime hours',
      severity: 'positive',
    });
  }

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <div className="drawer-panel" onClick={(e) => e.stopPropagation()}>
        {/* Drawer Header */}
        <div className="drawer-header">
          <div>
            <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              {employee.persona_name || employee.employee_id}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginTop: '0.2rem', fontSize: '0.8125rem' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-muted)' }}>
                {employee.employee_id}
              </span>
              <span style={{ color: 'var(--text-muted)' }}>&bull;</span>
              <span style={{ fontWeight: 600, color: 'var(--brand-light)' }}>
                {employee.role}
              </span>
              <span style={{ color: 'var(--text-muted)' }}>&bull;</span>
              <span style={{ color: 'var(--text-secondary)' }}>
                {employee.department}
              </span>
            </div>
          </div>
          <button className="btn-ghost" style={{ padding: '0.4rem', borderRadius: 'var(--radius-sm)', border: 'none', cursor: 'pointer' }} onClick={onClose}>
            <X size={18} color="var(--text-muted)" />
          </button>
        </div>

        {/* Drawer Body */}
        <div className="drawer-body">
          {/* Risk Score Summary Panel */}
          <div style={{ backgroundColor: 'var(--bg-surface-raised)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <RiskScore
              probability={prediction.fused_risk_probability}
              threshold={prediction.decision_threshold}
              tier={prediction.risk_tier}
              size="lg"
            />
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.625rem', display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)' }}>
              <span>Operating Decision Cutoff: {(prediction.decision_threshold * 100).toFixed(1)}%</span>
              <span style={{ color: prediction.fused_risk_probability >= prediction.decision_threshold ? 'var(--risk-critical)' : 'var(--risk-low)', fontWeight: 600 }}>
                {prediction.fused_risk_probability >= prediction.decision_threshold ? 'Action Recommended' : 'Low Attrition Likelihood'}
              </span>
            </div>
          </div>

          {/* Key Risk Drivers */}
          <div>
            <div style={{ fontSize: '0.8125rem', fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <AlertTriangle size={15} color="var(--risk-elevated)" />
              <span>Primary Risk Drivers</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {riskDrivers.map((driver, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: '0.75rem 1rem',
                    backgroundColor: 'var(--bg-input)',
                    borderRadius: 'var(--radius-sm)',
                    borderLeft: `3px solid ${
                      driver.severity === 'high' ? 'var(--risk-critical)' : driver.severity === 'medium' ? 'var(--risk-elevated)' : 'var(--risk-low)'
                    }`,
                  }}
                >
                  <div style={{ fontWeight: 700, fontSize: '0.8125rem', color: 'var(--text-primary)' }}>
                    {driver.label}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
                    {driver.detail}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Qualitative Feedback Stream */}
          <div>
            <div style={{ fontSize: '0.8125rem', fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <MessageSquare size={15} color="#fb923c" />
              <span>Recent Feedback Commentary</span>
            </div>
            <div style={{ padding: '1rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid #fb923c', fontStyle: 'italic', fontSize: '0.8125rem', color: 'var(--text-primary)', lineHeight: 1.55 }}>
              "{employee.recent_feedback}"
            </div>
          </div>

          {/* Key Telemetry Metrics */}
          <div>
            <div style={{ fontSize: '0.8125rem', fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '0.5rem' }}>
              Employee Context
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.625rem' }}>
              <div style={{ backgroundColor: 'var(--bg-input)', padding: '0.625rem', borderRadius: 'var(--radius-sm)' }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>COMPENSATION</div>
                <div style={{ fontSize: '0.9375rem', fontWeight: 700, fontFamily: 'var(--font-mono)', marginTop: '0.15rem' }}>
                  ${employee.salary.toLocaleString()}
                </div>
              </div>
              <div style={{ backgroundColor: 'var(--bg-input)', padding: '0.625rem', borderRadius: 'var(--radius-sm)' }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>TENURE</div>
                <div style={{ fontSize: '0.9375rem', fontWeight: 700, fontFamily: 'var(--font-mono)', marginTop: '0.15rem' }}>
                  {employee.tenure_months} mo
                </div>
              </div>
              <div style={{ backgroundColor: 'var(--bg-input)', padding: '0.625rem', borderRadius: 'var(--radius-sm)' }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>PERFORMANCE</div>
                <div style={{ fontSize: '0.9375rem', fontWeight: 700, fontFamily: 'var(--font-mono)', marginTop: '0.15rem' }}>
                  {(employee.performance_score * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          </div>

          {/* Progressive Disclosure: Technical Attribution Accordion */}
          <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '1rem' }}>
            <button
              onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                width: '100%',
                background: 'none',
                border: 'none',
                color: 'var(--text-muted)',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                padding: '0.25rem 0',
                textTransform: 'uppercase',
                letterSpacing: '0.04em',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Layers size={14} color="var(--brand-light)" />
                <span>Technical ML Attribution & Modality Weights</span>
              </div>
              {showTechnicalDetails ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>

            {showTechnicalDetails && (
              <div style={{ marginTop: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.625rem', backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: 'var(--radius-md)', fontSize: '0.75rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem' }}>
                  <div>
                    <div style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <Database size={12} color="#60a5fa" />
                      Structured Signal (p₁)
                    </div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.9375rem', color: '#60a5fa', marginTop: '0.15rem' }}>
                      {(prediction.structured_risk_probability * 100).toFixed(1)}%
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.7rem' }}>
                      w₁ = +{prediction.modality_breakdown.structured_weight} | Logit = {prediction.modality_breakdown.structured_logit}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <FileText size={12} color="#fb923c" />
                      Text Burnout Signal (p₂)
                    </div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.9375rem', color: '#fb923c', marginTop: '0.15rem' }}>
                      {(prediction.text_risk_probability * 100).toFixed(1)}%
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.7rem' }}>
                      w₂ = +{prediction.modality_breakdown.text_weight} | Logit = {prediction.modality_breakdown.text_logit}
                    </div>
                  </div>
                </div>

                <div style={{ paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.7rem' }}>
                  Intercept (w₀): {prediction.modality_breakdown.intercept} &bull; Net Contribution: {(prediction.modality_breakdown.structured_contribution + prediction.modality_breakdown.text_contribution).toFixed(4)}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Drawer Sticky Footer Action */}
        <div className="drawer-footer">
          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            onClick={() => {
              onSendToSimulator(employee);
              onClose();
            }}
          >
            <Play size={15} />
            Explore Retention Scenario in Simulator
          </button>
        </div>
      </div>
    </div>
  );
};
