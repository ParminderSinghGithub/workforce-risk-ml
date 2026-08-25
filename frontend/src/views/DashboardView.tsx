import React, { useEffect, useState } from 'react';
import { Users, AlertTriangle, TrendingUp, Layers, ArrowRight } from 'lucide-react';
import { StatCard } from '../components/StatCard';
import { RiskBadge } from '../components/RiskBadge';
import { SAMPLE_EMPLOYEES } from '../constants/sampleData';
import { predictBatch } from '../services/api';
import { PredictionResponse } from '../types/api';

interface DashboardViewProps {
  onNavigate: (view: 'analysis' | 'simulator' | 'batch') => void;
  onSelectEmployee: (employeeId: string) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({ onNavigate, onSelectEmployee }) => {
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadInitialBatch() {
      try {
        setIsLoading(true);
        const res = await predictBatch(SAMPLE_EMPLOYEES);
        setPredictions(res.predictions);
      } catch (err) {
        console.error('Failed to load dashboard batch prediction:', err);
      } finally {
        setIsLoading(false);
      }
    }
    loadInitialBatch();
  }, []);

  const totalAssessed = predictions.length || SAMPLE_EMPLOYEES.length;
  const elevatedOrHigher = predictions.filter(p => p.risk_tier !== 'LOW').length;
  const avgFusedProb = predictions.length
    ? (predictions.reduce((acc, p) => acc + p.fused_risk_probability, 0) / predictions.length) * 100
    : 28.5;

  return (
    <div>
      {/* Executive Header Banner */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="page-title">
            <span>Executive Workforce Risk Overview</span>
          </div>
          <div className="page-description">
            Sentinel continuously fuses high-dimensional demographic signals with qualitative feedback sentiment to detect early-stage enterprise turnover risk.
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button className="btn btn-secondary" onClick={() => onNavigate('simulator')}>
            Launch Scenario Simulator
          </button>
          <button className="btn btn-primary" onClick={() => onNavigate('batch')}>
            View Full Workforce
            <ArrowRight size={16} />
          </button>
        </div>
      </div>

      {/* Top Level KPIs */}
      <div className="grid-4" style={{ marginBottom: '2rem' }}>
        <StatCard
          title="Workforce Assessed"
          value={isLoading ? '...' : totalAssessed.toLocaleString()}
          subtitle="Enterprise sample monitored"
          icon={<Users size={20} />}
        />
        <StatCard
          title="Elevated Risk Population"
          value={isLoading ? '...' : `${elevatedOrHigher} (${((elevatedOrHigher / totalAssessed) * 100).toFixed(0)}%)`}
          subtitle="Above decision threshold (21.9%)"
          trendType="negative"
          icon={<AlertTriangle size={20} />}
        />
        <StatCard
          title="Average Exit Probability"
          value={isLoading ? '...' : `${avgFusedProb.toFixed(1)}%`}
          subtitle="Calibrated late-fusion mean"
          icon={<TrendingUp size={20} />}
        />
        <StatCard
          title="Active Risk Modalities"
          value="2 Streams"
          subtitle="Structured Telemetry + NLP"
          icon={<Layers size={20} />}
        />
      </div>

      {/* Main Analysis Grids */}
      <div className="grid-2" style={{ marginBottom: '2rem' }}>
        {/* Core Modality Synergy Overview */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <Layers size={18} color="var(--brand-accent)" />
                Multimodal Signal Architecture
              </div>
              <div className="card-subtitle">
                How Sentinel fuses complementary data streams into calibrated risk decisions
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ padding: '1rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-md)', borderLeft: '4px solid #3b82f6' }}>
              <div style={{ fontWeight: 700, fontSize: '0.9rem', color: '#60a5fa' }}>
                1. Structured Tabular Stream (PyTorch MLP)
              </div>
              <div style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                Ingests 17 continuous metrics (compensation, satisfaction, workload, overtime, tenure) and 5 categorical dimensions to quantify baseline organizational friction.
              </div>
            </div>

            <div style={{ padding: '1rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-md)', borderLeft: '4px solid #f97316' }}>
              <div style={{ fontWeight: 700, fontSize: '0.9rem', color: '#fb923c' }}>
                2. Qualitative Text Stream (DistilBERT + PEFT/LoRA)
              </div>
              <div style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                Fine-tuned sequence classifier evaluating qualitative review commentary to extract subtle psychological burnout and dissatisfaction cues invisible to numeric metrics.
              </div>
            </div>

            <div style={{ padding: '1rem', backgroundColor: 'rgba(37, 99, 235, 0.1)', borderRadius: 'var(--radius-md)', borderLeft: '4px solid #2563eb' }}>
              <div style={{ fontWeight: 700, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                3. Late Fusion Meta-Regressor (Log-Odds Mapping)
              </div>
              <div style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                Combines both probability streams via validation-fitted logistic regression, producing optimal probability calibration and a 0.5983 log-loss error floor.
              </div>
            </div>
          </div>
        </div>

        {/* Priority Attention List */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <AlertTriangle size={18} color="var(--risk-elevated)" />
                Immediate Action Queue
              </div>
              <div className="card-subtitle">
                High-priority employees exhibiting severe burnout or turnover signals
              </div>
            </div>
            <button className="btn btn-secondary" style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem' }} onClick={() => onNavigate('batch')}>
              View All
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {predictions.slice(0, 4).map((p) => {
              const emp = SAMPLE_EMPLOYEES.find(e => e.employee_id === p.employee_id);
              return (
                <div
                  key={p.employee_id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0.75rem 1rem',
                    backgroundColor: 'var(--bg-input)',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border-subtle)',
                    cursor: 'pointer',
                  }}
                  onClick={() => {
                    onSelectEmployee(p.employee_id);
                    onNavigate('analysis');
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                      {p.employee_id} — {emp?.role || 'Employee'}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      {emp?.department} • Score: {(p.fused_risk_probability * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <RiskBadge tier={p.risk_tier} size="sm" />
                    <ArrowRight size={14} color="var(--text-muted)" />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
