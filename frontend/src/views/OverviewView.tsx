import React from 'react';
import { Users, AlertTriangle, TrendingUp, ShieldAlert, ArrowRight } from 'lucide-react';
import { StatCard } from '../components/StatCard';
import { RiskBadge } from '../components/RiskBadge';
import { SAMPLE_EMPLOYEES } from '../constants/sampleData';
import { EmployeePredictionRequest, PredictionResponse } from '../types/api';

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

  // Department Risk Concentration Breakdown
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

  // Priority Queue: Sort by highest risk first
  const priorityQueue = [...predictions]
    .sort((a, b) => b.fused_risk_probability - a.fused_risk_probability)
    .slice(0, 4);

  return (
    <div>
      {/* Executive Page Header */}
      <div className="view-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="view-title">
            <span>Executive Workforce Posture</span>
          </div>
          <div className="view-subtitle">
            Sentinel continuously fuses demographic telemetry and qualitative feedback to detect early-stage turnover risk before formal resignations occur.
          </div>
        </div>

        <button className="btn btn-primary" onClick={onNavigateToWorkforce}>
          <span>View All Employees</span>
          <ArrowRight size={15} />
        </button>
      </div>

      {/* KPI Strip */}
      <div className="grid-4" style={{ marginBottom: '1.75rem' }}>
        <StatCard
          label="Total Workforce Assessed"
          value={isLoading ? '...' : totalMonitored}
          sublabel="Enterprise sample cohort"
          icon={<Users size={18} />}
        />
        <StatCard
          label="Immediate Action Required"
          value={isLoading ? '...' : criticalHighCount}
          sublabel="High or Critical risk tier"
          trend={criticalHighCount > 0 ? "Requires Intervention" : "Healthy"}
          trendType={criticalHighCount > 0 ? "negative" : "positive"}
          icon={<ShieldAlert size={18} />}
        />
        <StatCard
          label="Elevated Risk Population"
          value={isLoading ? '...' : elevatedCount}
          sublabel="Above 21.9% decision cutoff"
          icon={<AlertTriangle size={18} />}
        />
        <StatCard
          label="Average Exit Probability"
          value={isLoading ? '...' : `${avgExitRisk.toFixed(1)}%`}
          sublabel="Calibrated multimodal mean"
          icon={<TrendingUp size={18} />}
        />
      </div>

      {/* Visual Risk Distribution Bar */}
      <div className="panel" style={{ marginBottom: '1.75rem', padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <div style={{ fontSize: '0.8125rem', fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
            Workforce Risk Tier Distribution
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Total Population: {totalMonitored}
          </div>
        </div>

        {/* Stacked Percentage Bar */}
        <div style={{ display: 'flex', height: '10px', borderRadius: 'var(--radius-full)', overflow: 'hidden', backgroundColor: 'var(--bg-input)' }}>
          <div style={{ width: `${(lowCount / (totalMonitored || 1)) * 100}%`, backgroundColor: 'var(--risk-low)' }} title={`Low Risk: ${lowCount}`} />
          <div style={{ width: `${(elevatedCount / (totalMonitored || 1)) * 100}%`, backgroundColor: 'var(--risk-elevated)' }} title={`Elevated Risk: ${elevatedCount}`} />
          <div style={{ width: `${(criticalHighCount / (totalMonitored || 1)) * 100}%`, backgroundColor: 'var(--risk-critical)' }} title={`High/Critical Risk: ${criticalHighCount}`} />
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.75rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--risk-low)' }} />
            <span>Low Risk: <strong>{lowCount}</strong> ({((lowCount / (totalMonitored || 1)) * 100).toFixed(0)}%)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--risk-elevated)' }} />
            <span>Elevated Risk: <strong>{elevatedCount}</strong> ({((elevatedCount / (totalMonitored || 1)) * 100).toFixed(0)}%)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--risk-critical)' }} />
            <span>High / Critical: <strong>{criticalHighCount}</strong> ({((criticalHighCount / (totalMonitored || 1)) * 100).toFixed(0)}%)</span>
          </div>
        </div>
      </div>

      {/* Main Grid: Priority Action Queue (Left) & Department Risk Concentration (Right) */}
      <div className="grid-2">
        {/* Priority Action Queue */}
        <div className="panel">
          <div className="panel-header">
            <div>
              <div className="panel-title">
                <ShieldAlert size={16} color="var(--risk-critical)" />
                <span>Priority Action Queue</span>
              </div>
              <div className="panel-subtitle">
                Employees exhibiting strongest multimodal attrition signals
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
            {priorityQueue.map((pred) => {
              const emp = SAMPLE_EMPLOYEES.find(e => e.employee_id === pred.employee_id);
              if (!emp) return null;

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
                    transition: 'border-color 0.15s ease',
                  }}
                  className="clickable-row"
                  onClick={() => onInspectEmployee(emp, pred)}
                >
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                      {emp.employee_id} &bull; {emp.role}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                      {emp.department} &bull; Workload: {(emp.workload_score * 100).toFixed(0)}% &bull; Overtime: {emp.overtime_hours}h/wk
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.875rem' }}>
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
                At-risk population density across organizational units
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
