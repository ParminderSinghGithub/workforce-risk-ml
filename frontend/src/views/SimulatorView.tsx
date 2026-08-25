import React, { useState, useEffect } from 'react';
import { Sliders, Play, RotateCcw, TrendingDown, TrendingUp, Layers, ChevronDown, ChevronUp, Database, FileText } from 'lucide-react';
import { SAMPLE_EMPLOYEES } from '../constants/sampleData';
import { predictEmployee } from '../services/api';
import { EmployeePredictionRequest, PredictionResponse } from '../types/api';
import { RiskBadge } from '../components/RiskBadge';

interface SimulatorViewProps {
  initialEmployee?: EmployeePredictionRequest;
}

export const SimulatorView: React.FC<SimulatorViewProps> = ({ initialEmployee }) => {
  const [selectedId, setSelectedId] = useState<string>(
    initialEmployee?.employee_id || SAMPLE_EMPLOYEES[1].employee_id || 'EMP-2042'
  );

  const activeBase = initialEmployee?.employee_id === selectedId
    ? initialEmployee
    : SAMPLE_EMPLOYEES.find(e => e.employee_id === selectedId) || SAMPLE_EMPLOYEES[1];

  const [formData, setFormData] = useState<EmployeePredictionRequest>({ ...activeBase });
  const [baselinePred, setBaselinePred] = useState<PredictionResponse | null>(null);
  const [scenarioPred, setScenarioPred] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState<boolean>(false);

  // Sync when selected profile changes
  useEffect(() => {
    const base = SAMPLE_EMPLOYEES.find(e => e.employee_id === selectedId) || SAMPLE_EMPLOYEES[1];
    setFormData({ ...base });

    async function loadBaseline() {
      try {
        setIsLoading(true);
        setError(null);
        const res = await predictEmployee(base);
        setBaselinePred(res);
        setScenarioPred(res);
      } catch (err: any) {
        setError(err.message || 'Failed to initialize baseline');
      } finally {
        setIsLoading(false);
      }
    }
    loadBaseline();
  }, [selectedId]);

  const handleSimulate = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await predictEmployee(formData);
      setScenarioPred(res);
    } catch (err: any) {
      setError(err.message || 'Simulation execution failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({ ...activeBase });
    setScenarioPred(baselinePred);
  };

  const deltaPercentage = scenarioPred && baselinePred
    ? (scenarioPred.fused_risk_probability - baselinePred.fused_risk_probability) * 100
    : 0;

  const isImproved = deltaPercentage < 0;

  return (
    <div>
      {/* View Header */}
      <div className="view-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="view-title">
            <Sliders size={22} color="var(--brand-light)" />
            <span>Retention & Policy Simulator</span>
          </div>
          <div className="view-subtitle">
            Model organizational interventions (workload balancing, compensation adjustments, wellbeing shifts) to observe real-time retention trajectory impacts.
          </div>
        </div>

        {/* Profile Selector Toolbar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', fontWeight: 600 }}>BASELINE PROFILE:</span>
          <select
            className="form-select"
            style={{ minWidth: '220px' }}
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
          >
            {SAMPLE_EMPLOYEES.map((emp) => (
              <option key={emp.employee_id} value={emp.employee_id}>
                {emp.persona_name || emp.employee_id} ({emp.employee_id}) &bull; {emp.role}
              </option>
            ))}
          </select>
          <button className="btn btn-secondary" onClick={handleReset} title="Reset all sliders to original baseline">
            <RotateCcw size={14} />
            <span>Reset</span>
          </button>
          <button className="btn btn-primary" onClick={handleSimulate} disabled={isLoading}>
            <Play size={14} />
            <span>{isLoading ? 'Evaluating...' : 'Simulate Intervention'}</span>
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: '0.875rem 1.25rem', backgroundColor: 'var(--risk-critical-bg)', border: '1px solid var(--risk-critical-border)', borderRadius: 'var(--radius-md)', color: 'var(--risk-critical)', marginBottom: '1.5rem', fontSize: '0.875rem' }}>
          {error}
        </div>
      )}

      {/* Main Grid: Left Controls vs. Right Outcome */}
      <div className="grid-2" style={{ alignItems: 'start' }}>
        {/* Left Column: Grouped Policy & Attribute Levers */}
        <div className="panel">
          <div className="panel-header">
            <div>
              <div className="panel-title">
                <span>Intervention Levers</span>
              </div>
              <div className="panel-subtitle">
                Adjust organizational levers to simulate employee retention impact
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {/* Group 1: Workload & Overtime (Primary Stressors) */}
            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1.125rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#fb923c', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '1rem' }}>
                Workload & Operational Burden
              </div>

              {/* Workload Slider */}
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                  <label className="form-label" style={{ margin: 0 }}>Workload Burden Capacity</label>
                  <span className="slider-badge">{(formData.workload_score * 100).toFixed(0)}%</span>
                </div>
                <div className="slider-group">
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    className="slider-control"
                    value={formData.workload_score}
                    onChange={(e) => setFormData({ ...formData, workload_score: parseFloat(e.target.value) })}
                  />
                </div>
              </div>

              {/* Overtime Slider */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                  <label className="form-label" style={{ margin: 0 }}>Weekly Overtime Burden</label>
                  <span className="slider-badge">{formData.overtime_hours} hrs/wk</span>
                </div>
                <div className="slider-group">
                  <input
                    type="range"
                    min="0"
                    max="35"
                    step="1"
                    className="slider-control"
                    value={formData.overtime_hours}
                    onChange={(e) => setFormData({ ...formData, overtime_hours: parseFloat(e.target.value) })}
                  />
                </div>
              </div>
            </div>

            {/* Group 2: Compensation & Wellbeing */}
            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1.125rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--brand-light)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '1rem' }}>
                Compensation & Sentiment
              </div>

              {/* Salary Slider */}
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                  <label className="form-label" style={{ margin: 0 }}>Annual Base Compensation</label>
                  <span className="slider-badge">${formData.salary.toLocaleString()}</span>
                </div>
                <div className="slider-group">
                  <input
                    type="range"
                    min="45000"
                    max="220000"
                    step="5000"
                    className="slider-control"
                    value={formData.salary}
                    onChange={(e) => setFormData({ ...formData, salary: parseFloat(e.target.value) })}
                  />
                </div>
              </div>

              {/* Satisfaction Slider */}
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                  <label className="form-label" style={{ margin: 0 }}>Job Satisfaction Rating</label>
                  <span className="slider-badge">{(formData.satisfaction_score * 100).toFixed(0)}%</span>
                </div>
                <div className="slider-group">
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    className="slider-control"
                    value={formData.satisfaction_score}
                    onChange={(e) => setFormData({ ...formData, satisfaction_score: parseFloat(e.target.value) })}
                  />
                </div>
              </div>

              {/* Stress Level */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                  <label className="form-label" style={{ margin: 0 }}>Reported Stress Index</label>
                  <span className="slider-badge">{(formData.stress_level * 100).toFixed(0)}%</span>
                </div>
                <div className="slider-group">
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    className="slider-control"
                    value={formData.stress_level}
                    onChange={(e) => setFormData({ ...formData, stress_level: parseFloat(e.target.value) })}
                  />
                </div>
              </div>
            </div>

            {/* Group 3: Qualitative Feedback Text Stream */}
            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1.125rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '0.5rem' }}>
                Employee Qualitative Feedback (NLP Signal)
              </div>
              <textarea
                className="form-textarea"
                rows={3}
                value={formData.recent_feedback}
                onChange={(e) => setFormData({ ...formData, recent_feedback: e.target.value })}
                placeholder="Enter feedback sentiment commentary..."
              />
            </div>
          </div>
        </div>

        {/* Right Column: Simulated Outcome & Delta Shift */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Outcome Comparison Card */}
          <div className="panel">
            <div className="panel-header">
              <div>
                <div className="panel-title">
                  <span>Simulated Outcome & Trajectory</span>
                </div>
                <div className="panel-subtitle">
                  Live response from Sentinel Multimodal Late Fusion Engine
                </div>
              </div>
            </div>

            {scenarioPred && baselinePred && (
              <div>
                {/* Side-by-side Before & After */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
                  {/* Baseline State */}
                  <div style={{ backgroundColor: 'var(--bg-input)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                      Baseline State
                    </div>
                    <div style={{ fontSize: '2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', margin: '0.25rem 0' }}>
                      {(baselinePred.fused_risk_probability * 100).toFixed(1)}%
                    </div>
                    <RiskBadge tier={baselinePred.risk_tier} size="sm" />
                  </div>

                  {/* Simulated Scenario State */}
                  <div style={{ backgroundColor: 'var(--bg-surface-raised)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--brand-primary)' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--brand-light)', textTransform: 'uppercase' }}>
                      Simulated Scenario
                    </div>
                    <div style={{ fontSize: '2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', margin: '0.25rem 0' }}>
                      {(scenarioPred.fused_risk_probability * 100).toFixed(1)}%
                    </div>
                    <RiskBadge tier={scenarioPred.risk_tier} size="sm" />
                  </div>
                </div>

                {/* Trajectory Delta Banner */}
                <div
                  style={{
                    padding: '1rem 1.25rem',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: isImproved ? 'var(--risk-low-bg)' : deltaPercentage > 0 ? 'var(--risk-critical-bg)' : 'var(--bg-input)',
                    border: `1px solid ${isImproved ? 'var(--risk-low-border)' : deltaPercentage > 0 ? 'var(--risk-critical-border)' : 'var(--border-subtle)'}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
                    {isImproved ? (
                      <TrendingDown size={22} color="var(--risk-low)" />
                    ) : deltaPercentage > 0 ? (
                      <TrendingUp size={22} color="var(--risk-critical)" />
                    ) : null}
                    <div>
                      <div style={{ fontSize: '0.8125rem', fontWeight: 700, color: isImproved ? 'var(--risk-low)' : deltaPercentage > 0 ? 'var(--risk-critical)' : 'var(--text-primary)' }}>
                        {isImproved ? 'Retention Trajectory Improved' : deltaPercentage > 0 ? 'Risk Elevated' : 'No Net Trajectory Change'}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        Tier Transition: <strong>{baselinePred.risk_tier}</strong> &rarr; <strong>{scenarioPred.risk_tier}</strong>
                      </div>
                    </div>
                  </div>

                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, fontSize: '1.25rem', color: isImproved ? 'var(--risk-low)' : deltaPercentage > 0 ? 'var(--risk-critical)' : 'var(--text-primary)' }}>
                      {deltaPercentage > 0 ? `+${deltaPercentage.toFixed(1)} pp` : `${deltaPercentage.toFixed(1)} pp`}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Net Risk Delta</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Collapsible Technical Attribution */}
          {scenarioPred && (
            <div className="panel" style={{ padding: '1.25rem' }}>
              <button
                onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  width: '100%',
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-secondary)',
                  fontSize: '0.8125rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  padding: 0,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Layers size={15} color="var(--brand-light)" />
                  <span>Technical Modality Attribution (p₁, p₂, Weights)</span>
                </div>
                {showTechnicalDetails ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
              </button>

              {showTechnicalDetails && (
                <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.75rem' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem' }}>
                    <div style={{ backgroundColor: 'var(--bg-input)', padding: '0.75rem', borderRadius: 'var(--radius-sm)' }}>
                      <div style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        <Database size={12} color="#60a5fa" />
                        Tabular Signal (p₁)
                      </div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '1rem', color: '#60a5fa', marginTop: '0.2rem' }}>
                        {(scenarioPred.structured_risk_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div style={{ backgroundColor: 'var(--bg-input)', padding: '0.75rem', borderRadius: 'var(--radius-sm)' }}>
                      <div style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        <FileText size={12} color="#fb923c" />
                        NLP Burnout Signal (p₂)
                      </div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '1rem', color: '#fb923c', marginTop: '0.2rem' }}>
                        {(scenarioPred.text_risk_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.7rem', backgroundColor: 'var(--bg-input)', padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-sm)' }}>
                    logit(p) = -0.9703 + 0.1810&middot;({scenarioPred.modality_breakdown.structured_logit}) + 0.1470&middot;({scenarioPred.modality_breakdown.text_logit})
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
