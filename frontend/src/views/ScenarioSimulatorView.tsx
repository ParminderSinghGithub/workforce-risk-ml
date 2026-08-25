import React, { useState, useEffect } from 'react';
import { Sliders, Play, RotateCcw, ArrowRight, TrendingDown, TrendingUp } from 'lucide-react';
import { SAMPLE_EMPLOYEES } from '../constants/sampleData';
import { predictEmployee } from '../services/api';
import { EmployeePredictionRequest, PredictionResponse } from '../types/api';
import { RiskGauge } from '../components/RiskGauge';
import { ModalityComparison } from '../components/ModalityComparison';

interface ScenarioSimulatorViewProps {
  initialEmployee?: EmployeePredictionRequest;
}

export const ScenarioSimulatorView: React.FC<ScenarioSimulatorViewProps> = ({
  initialEmployee,
}) => {
  const baseProfile = initialEmployee || SAMPLE_EMPLOYEES[1]; // Default to high-risk profile for impactful what-if demos

  const [formData, setFormData] = useState<EmployeePredictionRequest>({ ...baseProfile });
  const [baselinePrediction, setBaselinePrediction] = useState<PredictionResponse | null>(null);
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Run baseline prediction on mount
  useEffect(() => {
    async function loadBaseline() {
      try {
        const res = await predictEmployee(baseProfile);
        setBaselinePrediction(res);
        setCurrentPrediction(res);
      } catch (err: any) {
        setError(err.message);
      }
    }
    loadBaseline();
  }, [baseProfile.employee_id]);

  const handleSimulate = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await predictEmployee(formData);
      setCurrentPrediction(res);
    } catch (err: any) {
      setError(err.message || 'Simulation request failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({ ...baseProfile });
    setCurrentPrediction(baselinePrediction);
  };

  const deltaProbability = currentPrediction && baselinePrediction
    ? (currentPrediction.fused_risk_probability - baselinePrediction.fused_risk_probability) * 100
    : 0;

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="page-title">
            <Sliders size={26} color="var(--brand-accent)" />
            <span>Interactive Scenario Simulator & What-If Engine</span>
          </div>
          <div className="page-description">
            Simulate organizational policy interventions, compensation adjustments, workload balancing, and feedback sentiment changes to observe calibrated risk trajectory impacts.
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button className="btn btn-secondary" onClick={handleReset}>
            <RotateCcw size={16} />
            Reset Baseline
          </button>
          <button className="btn btn-primary" onClick={handleSimulate} disabled={isLoading}>
            <Play size={16} />
            {isLoading ? 'Simulating...' : 'Run Simulation'}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: '1rem', backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', borderRadius: 'var(--radius-md)', color: '#fca5a5', marginBottom: '1.5rem' }}>
          {error}
        </div>
      )}

      {/* Grid Layout: Scenario Controls vs. Simulated Output */}
      <div className="grid-2" style={{ alignItems: 'start', marginBottom: '2rem' }}>
        {/* Left Column: Interactive Attribute Controls */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                Simulated Employee Attributes
              </div>
              <div className="card-subtitle">
                Modify parameters and execute live inference against the FastAPI serving engine
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {/* Department & Role */}
            <div className="grid-2" style={{ gap: '1rem' }}>
              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label">Department</label>
                <input
                  type="text"
                  className="form-input"
                  value={formData.department}
                  onChange={(e) => setFormData({ ...formData, department: e.target.value })}
                />
              </div>
              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label">Role Title</label>
                <input
                  type="text"
                  className="form-input"
                  value={formData.role}
                  onChange={(e) => setFormData({ ...formData, role: e.target.value })}
                />
              </div>
            </div>

            {/* Compensation Slider */}
            <div className="form-group" style={{ margin: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                <label className="form-label" style={{ margin: 0 }}>Annual Compensation ($)</label>
                <span className="slider-val">${formData.salary.toLocaleString()}</span>
              </div>
              <input
                type="range"
                min="40000"
                max="250000"
                step="5000"
                className="slider-input"
                value={formData.salary}
                onChange={(e) => setFormData({ ...formData, salary: parseFloat(e.target.value) })}
              />
            </div>

            {/* Workload Burden Slider */}
            <div className="form-group" style={{ margin: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                <label className="form-label" style={{ margin: 0 }}>Workload Burden Index</label>
                <span className="slider-val">{(formData.workload_score * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                className="slider-input"
                value={formData.workload_score}
                onChange={(e) => setFormData({ ...formData, workload_score: parseFloat(e.target.value) })}
              />
            </div>

            {/* Overtime Hours */}
            <div className="form-group" style={{ margin: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                <label className="form-label" style={{ margin: 0 }}>Weekly Overtime Burden (Hours)</label>
                <span className="slider-val">{formData.overtime_hours} hrs</span>
              </div>
              <input
                type="range"
                min="0"
                max="40"
                step="1"
                className="slider-input"
                value={formData.overtime_hours}
                onChange={(e) => setFormData({ ...formData, overtime_hours: parseFloat(e.target.value) })}
              />
            </div>

            {/* Job Satisfaction Score */}
            <div className="form-group" style={{ margin: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                <label className="form-label" style={{ margin: 0 }}>Job Satisfaction Score</label>
                <span className="slider-val">{(formData.satisfaction_score * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                className="slider-input"
                value={formData.satisfaction_score}
                onChange={(e) => setFormData({ ...formData, satisfaction_score: parseFloat(e.target.value) })}
              />
            </div>

            {/* Stress Level */}
            <div className="form-group" style={{ margin: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                <label className="form-label" style={{ margin: 0 }}>Reported Stress Index</label>
                <span className="slider-val">{(formData.stress_level * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                className="slider-input"
                value={formData.stress_level}
                onChange={(e) => setFormData({ ...formData, stress_level: parseFloat(e.target.value) })}
              />
            </div>

            {/* Qualitative Feedback Text Input */}
            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label">Employee Feedback Commentary (NLP Stream)</label>
              <textarea
                className="form-textarea"
                rows={3}
                value={formData.recent_feedback}
                onChange={(e) => setFormData({ ...formData, recent_feedback: e.target.value })}
                placeholder="Enter qualitative feedback comment..."
              />
            </div>
          </div>
        </div>

        {/* Right Column: Live Simulated Prediction Outcome */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Simulated Risk Meter */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">
                  Simulated Risk Outcome
                </div>
                <div className="card-subtitle">
                  Live response from Sentinel Multimodal Late Fusion API
                </div>
              </div>
            </div>

            {currentPrediction ? (
              <div>
                <RiskGauge
                  probability={currentPrediction.fused_risk_probability}
                  threshold={currentPrediction.decision_threshold}
                  tier={currentPrediction.risk_tier}
                />

                {/* Trajectory Comparison Delta */}
                {baselinePrediction && (
                  <div style={{ margin: '1rem 0 0', padding: '1rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>BASELINE PROBABILITY</div>
                      <div style={{ fontSize: '1.125rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                        {(baselinePrediction.fused_risk_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                    <ArrowRight size={18} color="var(--text-muted)" />
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>SIMULATED DELTA</div>
                      <div style={{ fontSize: '1.125rem', fontWeight: 700, fontFamily: 'var(--font-mono)', display: 'flex', alignItems: 'center', gap: '0.25rem', color: deltaProbability < 0 ? 'var(--risk-low)' : deltaProbability > 0 ? 'var(--risk-critical)' : 'var(--text-primary)' }}>
                        {deltaProbability < 0 ? <TrendingDown size={16} /> : deltaProbability > 0 ? <TrendingUp size={16} /> : null}
                        {deltaProbability > 0 ? `+${deltaProbability.toFixed(1)}%` : `${deltaProbability.toFixed(1)}%`}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : null}
          </div>

          {/* Modality Breakdown */}
          {currentPrediction && (
            <ModalityComparison
              pStructured={currentPrediction.structured_risk_probability}
              pText={currentPrediction.text_risk_probability}
              pFused={currentPrediction.fused_risk_probability}
              breakdown={currentPrediction.modality_breakdown}
            />
          )}
        </div>
      </div>
    </div>
  );
};
