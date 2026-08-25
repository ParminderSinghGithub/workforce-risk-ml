import React from 'react';
import { AlertTriangle, ArrowRight, ShieldAlert } from 'lucide-react';
import { SAMPLE_EMPLOYEES } from '../constants/sampleData';
import { EmployeePredictionRequest, PredictionResponse } from '../types/api';
import { RiskBadge } from '../components/RiskBadge';

interface OverviewViewProps {
  predictions: PredictionResponse[];
  isLoading: boolean;
  onInspectEmployee: (employee: EmployeePredictionRequest, prediction: PredictionResponse) => void;
  onNavigateToWorkforce: () => void;
  onNavigateToSimulator: (employee: EmployeePredictionRequest) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({
  predictions,
  isLoading,
  onInspectEmployee,
  onNavigateToWorkforce,
}) => {
  const totalMonitored = SAMPLE_EMPLOYEES.length;
  const criticalHighCount = predictions.filter(p => p.risk_tier === 'CRITICAL' || p.risk_tier === 'HIGH').length;
  const elevatedCount = predictions.filter(p => p.risk_tier === 'ELEVATED').length;
  const lowCount = predictions.filter(p => p.risk_tier === 'LOW').length;

  const avgExitRisk = predictions.length
    ? (predictions.reduce((acc, p) => acc + p.fused_risk_probability, 0) / predictions.length) * 100
    : 28.5;

  // Department Breakdown
  const deptCounts: Record<string, { total: number; atRisk: number }> = {};
  SAMPLE_EMPLOYEES.forEach((emp) => {
    if (!deptCounts[emp.department]) {
      deptCounts[emp.department] = { total: 0, atRisk: 0 };
    }
    deptCounts[emp.department].total += 1;
    const pred = predictions.find(p => p.employee_id === emp.employee_id);
    if (pred && pred.risk_tier !== 'LOW') {
      deptCounts[emp.department].atRisk += 1;
    }
  });

  // Priority Queue: highest risk first
  const priorityQueue = [...predictions]
    .sort((a, b) => b.fused_risk_probability - a.fused_risk_probability)
    .slice(0, 5);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Executive Hero Banner & Context */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1.25rem' }}>
        <div>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', padding: '0.25rem 0.65rem', borderRadius: 'var(--radius-full)', backgroundColor: 'var(--risk-critical-bg)', border: '1px solid var(--risk-critical-border)', color: 'var(--risk-critical)', fontSize: '0.75rem', fontWeight: 700, marginBottom: '0.75rem' }}>
            <ShieldAlert size={13} />
            <span>{criticalHighCount} EMPLOYEES REQUIRE IMMEDIATE RETENTION ATTENTION</span>
          </div>
          <h1 style={{ fontSize: '1.875rem', fontWeight: 800, letterSpacing: '-0.025em', color: 'var(--text-primary)', margin: 0 }}>
            Workforce Retention Posture
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem', marginTop: '0.35rem', maxWidth: '680px', lineHeight: 1.5 }}>
            Sentinel continuously monitors employee demographic telemetry and qualitative review sentiment to forecast organizational attrition risk before formal resignations occur.
          </p>
        </div>

        <button className="btn btn-primary" onClick={onNavigateToWorkforce} style={{ padding: '0.625rem 1.25rem', fontSize: '0.875rem' }}>
          <span>View Full Roster</span>
          <ArrowRight size={15} />
        </button>
      </div>

      {/* Executive KPI Summary Strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.25rem' }}>
        <div className="panel" style={{ padding: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Workforce Monitored
          </div>
          <div style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.25rem', fontFamily: 'var(--font-sans)', letterSpacing: '-0.02em' }}>
            {isLoading ? '...' : totalMonitored}
          </div>
          <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
            Enterprise sample cohort
          </div>
        </div>

        <div className="panel" style={{ padding: '1.25rem', borderColor: criticalHighCount > 0 ? 'var(--risk-critical-border)' : 'var(--border-subtle)' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Immediate Action Required
          </div>
          <div style={{ fontSize: '1.875rem', fontWeight: 800, color: criticalHighCount > 0 ? 'var(--risk-critical)' : 'var(--text-primary)', marginTop: '0.25rem', fontFamily: 'var(--font-sans)', letterSpacing: '-0.02em' }}>
            {isLoading ? '...' : criticalHighCount}
          </div>
          <div style={{ fontSize: '0.8125rem', color: criticalHighCount > 0 ? 'var(--risk-critical)' : 'var(--text-muted)', marginTop: '0.35rem', fontWeight: 600 }}>
            {criticalHighCount > 0 ? 'Critical / High risk tier' : 'No critical alerts'}
          </div>
        </div>

        <div className="panel" style={{ padding: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Elevated Risk Count
          </div>
          <div style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--risk-elevated)', marginTop: '0.25rem', fontFamily: 'var(--font-sans)', letterSpacing: '-0.02em' }}>
            {isLoading ? '...' : elevatedCount}
          </div>
          <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
            Above 21.9% decision cutoff
          </div>
        </div>

        <div className="panel" style={{ padding: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Average Exit Probability
          </div>
          <div style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.25rem', fontFamily: 'var(--font-sans)', letterSpacing: '-0.02em' }}>
            {isLoading ? '...' : `${avgExitRisk.toFixed(1)}%`}
          </div>
          <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
            Calibrated late-fusion mean
          </div>
        </div>
      </div>

      {/* Risk Tier Distribution Bar */}
      <div className="panel" style={{ padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <div style={{ fontSize: '0.8125rem', fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
            Workforce Risk Tier Distribution
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            N = {totalMonitored} Employees Monitored
          </div>
        </div>

        {/* Stacked Proportional Bar */}
        <div style={{ display: 'flex', height: '10px', borderRadius: 'var(--radius-full)', overflow: 'hidden', backgroundColor: 'var(--bg-input)' }}>
          <div style={{ width: `${(lowCount / (totalMonitored || 1)) * 100}%`, backgroundColor: 'var(--risk-low)' }} title={`Low Risk: ${lowCount}`} />
          <div style={{ width: `${(elevatedCount / (totalMonitored || 1)) * 100}%`, backgroundColor: 'var(--risk-elevated)' }} title={`Elevated Risk: ${elevatedCount}`} />
          <div style={{ width: `${(criticalHighCount / (totalMonitored || 1)) * 100}%`, backgroundColor: 'var(--risk-critical)' }} title={`High/Critical Risk: ${criticalHighCount}`} />
        </div>

        {/* Semantic Legend */}
        <div style={{ display: 'flex', gap: '2rem', marginTop: '0.875rem', fontSize: '0.8125rem', color: 'var(--text-secondary)', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--risk-low)' }} />
            <span>Low Risk: <strong>{lowCount}</strong> ({((lowCount / (totalMonitored || 1)) * 100).toFixed(0)}%)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--risk-elevated)' }} />
            <span>Elevated Risk: <strong>{elevatedCount}</strong> ({((elevatedCount / (totalMonitored || 1)) * 100).toFixed(0)}%)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--risk-critical)' }} />
            <span>High / Critical: <strong>{criticalHighCount}</strong> ({((criticalHighCount / (totalMonitored || 1)) * 100).toFixed(0)}%)</span>
          </div>
        </div>
      </div>

      {/* Priority Action Queue (Left) & Department Risk Breakdown (Right) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
        {/* Priority Action List */}
        <div className="panel">
          <div className="panel-header">
            <div>
              <div className="panel-title">
                <ShieldAlert size={16} color="var(--risk-critical)" />
                <span>Priority Action Queue</span>
              </div>
              <div className="panel-subtitle">
                Employees exhibiting highest turnover risk signals
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
            {priorityQueue.map((pred) => {
              const emp = SAMPLE_EMPLOYEES.find(e => e.employee_id === pred.employee_id);
              if (!emp) return null;

              // Generate concise primary risk signal
              let primarySignal = 'Balanced Telemetry';
              if (emp.workload_score > 0.8 && emp.overtime_hours > 15) {
                primarySignal = `Extreme Workload (${(emp.workload_score * 100).toFixed(0)}%) & ${emp.overtime_hours}h Overtime`;
              } else if (emp.satisfaction_score < 0.4) {
                primarySignal = `Severe Job Dissatisfaction (${(emp.satisfaction_score * 100).toFixed(0)}%)`;
              } else if (emp.stress_level > 0.75) {
                primarySignal = `Elevated Stress (${(emp.stress_level * 100).toFixed(0)}%) & High Queue Volume`;
              } else if (emp.workload_score > 0.7) {
                primarySignal = `Heavy Workload Burden (${(emp.workload_score * 100).toFixed(0)}%)`;
              }

              return (
                <div
                  key={pred.employee_id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0.875rem 1rem',
                    backgroundColor: 'var(--bg-input)',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border-subtle)',
                    cursor: 'pointer',
                  }}
                  className="clickable-row"
                  onClick={() => onInspectEmployee(emp, pred)}
                >
                  <div style={{ flex: 1, marginRight: '1rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                        {emp.employee_id} &bull; {emp.role}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                      {emp.department} &bull; <span style={{ color: 'var(--risk-elevated)', fontWeight: 600 }}>{primarySignal}</span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, fontSize: '0.9375rem' }}>
                        {(pred.fused_risk_probability * 100).toFixed(1)}%
                      </div>
                      <RiskBadge tier={pred.risk_tier} size="sm" showIcon={false} />
                    </div>
                    <ArrowRight size={14} color="var(--text-muted)" />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Department Risk Concentration */}
        <div className="panel">
          <div className="panel-header">
            <div>
              <div className="panel-title">
                <AlertTriangle size={16} color="var(--risk-elevated)" />
                <span>Department Risk Concentration</span>
              </div>
              <div className="panel-subtitle">
                At-risk density across organizational business units
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {Object.entries(deptCounts).map(([dept, counts]) => {
              const atRiskPct = (counts.atRisk / counts.total) * 100;
              return (
                <div key={dept} style={{ backgroundColor: 'var(--bg-input)', padding: '0.875rem 1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', fontSize: '0.8125rem' }}>
                    <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{dept}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: atRiskPct > 50 ? 'var(--risk-critical)' : 'var(--text-secondary)' }}>
                      {counts.atRisk} of {counts.total} at risk ({atRiskPct.toFixed(0)}%)
                    </span>
                  </div>
                  <div style={{ height: '6px', borderRadius: 'var(--radius-full)', backgroundColor: 'var(--bg-surface-raised)', overflow: 'hidden' }}>
                    <div
                      style={{
                        height: '100%',
                        width: `${atRiskPct}%`,
                        backgroundColor: atRiskPct > 50 ? 'var(--risk-critical)' : atRiskPct > 0 ? 'var(--risk-elevated)' : 'var(--risk-low)',
                        borderRadius: 'var(--radius-full)',
                      }}
                    />
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
