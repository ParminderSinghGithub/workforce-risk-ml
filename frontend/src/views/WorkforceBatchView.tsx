import React, { useState, useEffect } from 'react';
import { Users, Filter, ArrowUpDown, RefreshCw, Eye } from 'lucide-react';
import { SAMPLE_EMPLOYEES } from '../constants/sampleData';
import { predictBatch } from '../services/api';
import { PredictionResponse } from '../types/api';
import { RiskBadge } from '../components/RiskBadge';

interface WorkforceBatchViewProps {
  onSelectEmployee: (empId: string) => void;
  onNavigateToAnalysis: () => void;
}

export const WorkforceBatchView: React.FC<WorkforceBatchViewProps> = ({
  onSelectEmployee,
  onNavigateToAnalysis,
}) => {
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters & Sorting
  const [selectedTier, setSelectedTier] = useState<string>('ALL');
  const [selectedDepartment, setSelectedDepartment] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [sortBy, setSortBy] = useState<'fused' | 'structured' | 'text'>('fused');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const loadBatch = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await predictBatch(SAMPLE_EMPLOYEES);
      setPredictions(res.predictions);
    } catch (err: any) {
      setError(err.message || 'Batch prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadBatch();
  }, []);

  const departments = ['ALL', ...Array.from(new Set(SAMPLE_EMPLOYEES.map(e => e.department)))];

  // Filter and sort items
  const filteredRecords = predictions.filter((p) => {
    const emp = SAMPLE_EMPLOYEES.find(e => e.employee_id === p.employee_id);
    if (!emp) return false;

    if (selectedTier !== 'ALL' && p.risk_tier !== selectedTier) return false;
    if (selectedDepartment !== 'ALL' && emp.department !== selectedDepartment) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      const matchId = p.employee_id.toLowerCase().includes(q);
      const matchRole = emp.role.toLowerCase().includes(q);
      const matchDept = emp.department.toLowerCase().includes(q);
      if (!matchId && !matchRole && !matchDept) return false;
    }
    return true;
  }).sort((a, b) => {
    let valA = a.fused_risk_probability;
    let valB = b.fused_risk_probability;
    if (sortBy === 'structured') {
      valA = a.structured_risk_probability;
      valB = b.structured_risk_probability;
    } else if (sortBy === 'text') {
      valA = a.text_risk_probability;
      valB = b.text_risk_probability;
    }
    return sortOrder === 'desc' ? valB - valA : valA - valB;
  });

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="page-title">
            <Users size={26} color="var(--brand-accent)" />
            <span>Workforce Cohort & Batch Risk View</span>
          </div>
          <div className="page-description">
            High-throughput enterprise population screening with calibrated multimodal risk ranking.
          </div>
        </div>

        <button className="btn btn-secondary" onClick={loadBatch} disabled={isLoading}>
          <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
          Refresh Batch
        </button>
      </div>

      {error && (
        <div style={{ padding: '1rem', backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', borderRadius: 'var(--radius-md)', color: '#fca5a5', marginBottom: '1.5rem' }}>
          {error}
        </div>
      )}

      {/* Filter Toolbar */}
      <div className="card" style={{ marginBottom: '1.5rem', padding: '1rem' }}>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
          {/* Search */}
          <div style={{ flex: 1, minWidth: '220px' }}>
            <input
              type="text"
              className="form-input"
              placeholder="Search by ID, role, or department..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          {/* Department Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Filter size={15} color="var(--text-muted)" />
            <select
              className="form-select"
              value={selectedDepartment}
              onChange={(e) => setSelectedDepartment(e.target.value)}
            >
              {departments.map((dept) => (
                <option key={dept} value={dept}>Dept: {dept}</option>
              ))}
            </select>
          </div>

          {/* Tier Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
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

          {/* Sort Control */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <ArrowUpDown size={15} color="var(--text-muted)" />
            <select
              className="form-select"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
            >
              <option value="fused">Sort: Fused Risk</option>
              <option value="text">Sort: Text Burnout</option>
              <option value="structured">Sort: Tabular Signal</option>
            </select>
            <button
              className="btn btn-secondary"
              style={{ padding: '0.625rem 0.75rem' }}
              onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
            >
              {sortOrder === 'desc' ? '↓ High-to-Low' : '↑ Low-to-High'}
            </button>
          </div>
        </div>
      </div>

      {/* Cohort Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Employee ID</th>
                <th>Role & Department</th>
                <th>Satisfaction</th>
                <th>Workload</th>
                <th>Structured Signal</th>
                <th>Text Burnout</th>
                <th>Fused Sentinel Risk</th>
                <th>Risk Tier</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                    Evaluating batch predictions against FastAPI serving engine...
                  </td>
                </tr>
              ) : filteredRecords.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                    No employee records match the active filter criteria.
                  </td>
                </tr>
              ) : (
                filteredRecords.map((p) => {
                  const emp = SAMPLE_EMPLOYEES.find(e => e.employee_id === p.employee_id);
                  if (!emp) return null;
                  return (
                    <tr key={p.employee_id}>
                      <td style={{ fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{p.employee_id}</td>
                      <td>
                        <div style={{ fontWeight: 600 }}>{emp.role}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{emp.department} • {emp.job_level}</div>
                      </td>
                      <td>{(emp.satisfaction_score * 100).toFixed(0)}%</td>
                      <td>{(emp.workload_score * 100).toFixed(0)}%</td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: '#60a5fa' }}>
                        {(p.structured_risk_probability * 100).toFixed(1)}%
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: '#fb923c' }}>
                        {(p.text_risk_probability * 100).toFixed(1)}%
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, fontSize: '1rem' }}>
                        {(p.fused_risk_probability * 100).toFixed(1)}%
                      </td>
                      <td>
                        <RiskBadge tier={p.risk_tier} size="sm" />
                      </td>
                      <td>
                        <button
                          className="btn btn-secondary"
                          style={{ fontSize: '0.75rem', padding: '0.35rem 0.65rem' }}
                          onClick={() => {
                            onSelectEmployee(p.employee_id);
                            onNavigateToAnalysis();
                          }}
                        >
                          <Eye size={13} />
                          Inspect
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
    </div>
  );
};
