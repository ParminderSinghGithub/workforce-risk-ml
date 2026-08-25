import React, { useState } from 'react';
import { Users, Filter, ArrowUpDown, RefreshCw, Eye, Search } from 'lucide-react';
import { SAMPLE_EMPLOYEES } from '../constants/sampleData';
import { EmployeePredictionRequest, PredictionResponse } from '../types/api';
import { RiskBadge } from '../components/RiskBadge';

interface WorkforceViewProps {
  predictions: PredictionResponse[];
  isLoading: boolean;
  onRefresh: () => void;
  onInspectEmployee: (employee: EmployeePredictionRequest, prediction: PredictionResponse) => void;
}

export const WorkforceView: React.FC<WorkforceViewProps> = ({
  predictions,
  isLoading,
  onRefresh,
  onInspectEmployee,
}) => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedDept, setSelectedDept] = useState<string>('ALL');
  const [selectedTier, setSelectedTier] = useState<string>('ALL');
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc');

  const departments = ['ALL', ...Array.from(new Set(SAMPLE_EMPLOYEES.map(e => e.department)))];

  // Filtering & Sorting
  const filteredEmployees = SAMPLE_EMPLOYEES.filter((emp) => {
    const pred = predictions.find(p => p.employee_id === emp.employee_id);
    if (!pred) return true;

    if (selectedDept !== 'ALL' && emp.department !== selectedDept) return false;
    if (selectedTier !== 'ALL' && pred.risk_tier !== selectedTier) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      const matchId = (emp.employee_id || '').toLowerCase().includes(q);
      const matchRole = emp.role.toLowerCase().includes(q);
      const matchDept = emp.department.toLowerCase().includes(q);
      if (!matchId && !matchRole && !matchDept) return false;
    }
    return true;
  }).sort((a, b) => {
    const predA = predictions.find(p => p.employee_id === a.employee_id);
    const predB = predictions.find(p => p.employee_id === b.employee_id);
    const valA = predA ? predA.fused_risk_probability : 0;
    const valB = predB ? predB.fused_risk_probability : 0;
    return sortOrder === 'desc' ? valB - valA : valA - valB;
  });

  return (
    <div>
      {/* View Header */}
      <div className="view-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="view-title">
            <Users size={22} color="var(--brand-light)" />
            <span>Workforce Risk Roster</span>
          </div>
          <div className="view-subtitle">
            Screen and prioritize employees across departments. Click any row to inspect drivers and test retention interventions.
          </div>
        </div>

        <button className="btn btn-secondary" onClick={onRefresh} disabled={isLoading}>
          <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
          <span>Refresh Cohort</span>
        </button>
      </div>

      {/* Filter Toolbar */}
      <div className="panel" style={{ marginBottom: '1.25rem', padding: '0.875rem 1.25rem' }}>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
          {/* Search Box */}
          <div style={{ flex: 1, minWidth: '220px', position: 'relative' }}>
            <Search size={14} color="var(--text-muted)" style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)' }} />
            <input
              type="text"
              className="form-input"
              style={{ paddingLeft: '2rem' }}
              placeholder="Search employee ID, role, or department..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          {/* Department Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Filter size={13} color="var(--text-muted)" />
            <select
              className="form-select"
              value={selectedDept}
              onChange={(e) => setSelectedDept(e.target.value)}
            >
              {departments.map((d) => (
                <option key={d} value={d}>Dept: {d}</option>
              ))}
            </select>
          </div>

          {/* Risk Tier Filter */}
          <div>
            <select
              className="form-select"
              value={selectedTier}
              onChange={(e) => setSelectedTier(e.target.value)}
            >
              <option value="ALL">All Risk Tiers</option>
              <option value="LOW">Low Risk</option>
              <option value="ELEVATED">Elevated Risk</option>
              <option value="HIGH">High Risk</option>
              <option value="CRITICAL">Critical Risk</option>
            </select>
          </div>

          {/* Sort Order Toggle */}
          <button
            className="btn btn-secondary"
            style={{ padding: '0.5625rem 0.8125rem' }}
            onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
            title="Toggle sort direction"
          >
            <ArrowUpDown size={13} />
            <span>{sortOrder === 'desc' ? 'Highest Risk First' : 'Lowest Risk First'}</span>
          </button>
        </div>
      </div>

      {/* Workforce Roster Table */}
      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Employee ID</th>
              <th>Role & Department</th>
              <th>Satisfaction</th>
              <th>Workload</th>
              <th>Overtime</th>
              <th>Exit Probability</th>
              <th>Risk Tier</th>
              <th style={{ textAlign: 'right' }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={8} style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                  Evaluating cohort predictions against Sentinel ML serving engine...
                </td>
              </tr>
            ) : filteredEmployees.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                  No employees found matching the filter criteria.
                </td>
              </tr>
            ) : (
              filteredEmployees.map((emp) => {
                const pred = predictions.find(p => p.employee_id === emp.employee_id);
                const exitProb = pred ? (pred.fused_risk_probability * 100).toFixed(1) : '—';
                const tier = pred ? pred.risk_tier : 'LOW';

                return (
                  <tr
                    key={emp.employee_id}
                    className="clickable-row"
                    onClick={() => pred && onInspectEmployee(emp, pred)}
                  >
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>
                      {emp.employee_id}
                    </td>
                    <td>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{emp.role}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {emp.department} &bull; {emp.job_level} Level
                      </div>
                    </td>
                    <td>
                      <span style={{ color: emp.satisfaction_score < 0.45 ? 'var(--risk-critical)' : 'var(--text-primary)' }}>
                        {(emp.satisfaction_score * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td>
                      <span style={{ color: emp.workload_score > 0.75 ? 'var(--risk-critical)' : 'var(--text-primary)' }}>
                        {(emp.workload_score * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td>
                      <span>{emp.overtime_hours} hrs/wk</span>
                    </td>
                    <td>
                      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, fontSize: '0.9375rem' }}>
                        {exitProb}%
                      </span>
                    </td>
                    <td>
                      <RiskBadge tier={tier} size="sm" />
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-ghost"
                        style={{ fontSize: '0.75rem', padding: '0.35rem 0.6rem' }}
                        onClick={(e) => {
                          e.stopPropagation();
                          pred && onInspectEmployee(emp, pred);
                        }}
                      >
                        <Eye size={13} />
                        <span>Inspect</span>
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
