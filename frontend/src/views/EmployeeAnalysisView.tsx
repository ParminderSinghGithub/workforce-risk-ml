import React, { useState, useEffect } from 'react';
import { UserCheck, MessageSquare, Briefcase, RefreshCw } from 'lucide-react';
import { SAMPLE_EMPLOYEES } from '../constants/sampleData';
import { predictEmployee } from '../services/api';
import { EmployeePredictionRequest, PredictionResponse } from '../types/api';
import { RiskGauge } from '../components/RiskGauge';
import { ModalityComparison } from '../components/ModalityComparison';

interface EmployeeAnalysisViewProps {
  selectedEmployeeId?: string;
  onNavigateToSimulator: (employee: EmployeePredictionRequest) => void;
}

export const EmployeeAnalysisView: React.FC<EmployeeAnalysisViewProps> = ({
  selectedEmployeeId = 'EMP-1001',
  onNavigateToSimulator,
}) => {
  const [currentEmpId, setCurrentEmpId] = useState<string>(selectedEmployeeId);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const selectedProfile = SAMPLE_EMPLOYEES.find(e => e.employee_id === currentEmpId) || SAMPLE_EMPLOYEES[0];

  const runPrediction = async (profile: EmployeePredictionRequest) => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await predictEmployee(profile);
      setPrediction(res);
    } catch (err: any) {
      setError(err.message || 'Failed to generate prediction');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    runPrediction(selectedProfile);
  }, [currentEmpId]);

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="page-title">
            <UserCheck size={26} color="var(--brand-accent)" />
            <span>Employee Risk Analysis & Inspector</span>
          </div>
          <div className="page-description">
            In-depth evaluation of individual employee attrition risk with full modality attribution breakdown.
          </div>
        </div>

        {/* Profile Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600 }}>SELECT PROFILE:</span>
          <select
            className="form-select"
            style={{ minWidth: '240px' }}
            value={currentEmpId}
            onChange={(e) => setCurrentEmpId(e.target.value)}
          >
            {SAMPLE_EMPLOYEES.map((emp) => (
              <option key={emp.employee_id} value={emp.employee_id}>
                {emp.employee_id} — {emp.role} ({emp.department})
              </option>
            ))}
          </select>
          <button
            className="btn btn-secondary"
            onClick={() => onNavigateToSimulator(selectedProfile)}
          >
            Open in Simulator
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: '1rem', backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', borderRadius: 'var(--radius-md)', color: '#fca5a5', marginBottom: '1.5rem' }}>
          {error}
        </div>
      )}

      {/* Main Analysis Panel */}
      <div className="grid-3" style={{ marginBottom: '2rem' }}>
        {/* Risk Gauge Card */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          {isLoading ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
              <RefreshCw size={24} className="animate-spin" style={{ margin: '0 auto 0.5rem' }} />
              Calculating multimodal forward passes...
            </div>
          ) : prediction ? (
            <RiskGauge
              probability={prediction.fused_risk_probability}
              threshold={prediction.decision_threshold}
              tier={prediction.risk_tier}
            />
          ) : null}
        </div>

        {/* Profile Overview */}
        <div className="card" style={{ gridColumn: 'span 2' }}>
          <div className="card-header">
            <div>
              <div className="card-title">
                <Briefcase size={18} color="var(--brand-accent)" />
                {selectedProfile.employee_id}: {selectedProfile.role}
              </div>
              <div className="card-subtitle">
                {selectedProfile.department} Department • {selectedProfile.job_level} Level • Tenure: {selectedProfile.tenure_months} months
              </div>
            </div>
          </div>

          <div className="grid-3" style={{ gap: '1rem', marginTop: '1rem' }}>
            <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ANNUAL SALARY</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                ${selectedProfile.salary.toLocaleString()}
              </div>
            </div>
            <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>SATISFACTION SCORE</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: selectedProfile.satisfaction_score < 0.5 ? 'var(--risk-critical)' : 'var(--risk-low)' }}>
                {(selectedProfile.satisfaction_score * 100).toFixed(0)}%
              </div>
            </div>
            <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>WORKLOAD BURDEN</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: selectedProfile.workload_score > 0.7 ? 'var(--risk-critical)' : 'var(--text-primary)' }}>
                {(selectedProfile.workload_score * 100).toFixed(0)}%
              </div>
            </div>
            <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>OVERTIME BURDEN</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                {selectedProfile.overtime_hours} hrs/week
              </div>
            </div>
            <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>STRESS INDEX</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: selectedProfile.stress_level > 0.7 ? 'var(--risk-critical)' : 'var(--text-primary)' }}>
                {(selectedProfile.stress_level * 100).toFixed(0)}%
              </div>
            </div>
            <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>TEAM SENTIMENT</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                {(selectedProfile.team_sentiment * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Qualitative Feedback Text & Modality Attribution */}
      <div className="grid-2" style={{ marginBottom: '2rem' }}>
        {/* Feedback Text Review */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <MessageSquare size={18} color="#f97316" />
                Qualitative Feedback Stream
              </div>
              <div className="card-subtitle">
                Raw employee commentary processed by fine-tuned DistilBERT + LoRA
              </div>
            </div>
          </div>

          <div style={{ padding: '1.25rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-md)', borderLeft: '4px solid #f97316', fontStyle: 'italic', color: 'var(--text-primary)', lineHeight: 1.6 }}>
            "{selectedProfile.recent_feedback}"
          </div>

          {prediction && (
            <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              <strong>Summary Assessment:</strong> {prediction.summary}
            </div>
          )}
        </div>

        {/* Modality Attribution Component */}
        {prediction && (
          <ModalityComparison
            pStructured={prediction.structured_risk_probability}
            pText={prediction.text_risk_probability}
            pFused={prediction.fused_risk_probability}
            breakdown={prediction.modality_breakdown}
          />
        )}
      </div>
    </div>
  );
};
