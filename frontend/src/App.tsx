import React, { useState, useEffect } from 'react';
import { Header, ViewTab } from './components/Header';
import { DashboardView } from './views/DashboardView';
import { EmployeeAnalysisView } from './views/EmployeeAnalysisView';
import { ScenarioSimulatorView } from './views/ScenarioSimulatorView';
import { WorkforceBatchView } from './views/WorkforceBatchView';
import { MethodologyView } from './views/MethodologyView';
import { SystemStatusView } from './views/SystemStatusView';
import { fetchHealth } from './services/api';
import { EmployeePredictionRequest, HealthResponse } from './types/api';
import { SAMPLE_EMPLOYEES } from './constants/sampleData';

export const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<ViewTab>('dashboard');
  const [selectedEmployeeId, setSelectedEmployeeId] = useState<string>('EMP-1001');
  const [simulatorEmployee, setSimulatorEmployee] = useState<EmployeePredictionRequest>(SAMPLE_EMPLOYEES[1]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoadingHealth, setIsLoadingHealth] = useState<boolean>(true);

  // Poll health on mount and every 30 seconds
  useEffect(() => {
    async function checkHealth() {
      try {
        setIsLoadingHealth(true);
        const res = await fetchHealth();
        setHealth(res);
      } catch (err) {
        console.warn('FastAPI serving layer not reachable:', err);
        setHealth(null);
      } finally {
        setIsLoadingHealth(false);
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleNavigateToSimulator = (employee: EmployeePredictionRequest) => {
    setSimulatorEmployee(employee);
    setCurrentView('simulator');
  };

  const handleSelectEmployee = (empId: string) => {
    setSelectedEmployeeId(empId);
  };

  return (
    <div className="app-container">
      <Header
        currentView={currentView}
        onViewChange={setCurrentView}
        health={health}
        isLoadingHealth={isLoadingHealth}
      />

      <main className="main-content">
        {currentView === 'dashboard' && (
          <DashboardView
            onNavigate={(view) => setCurrentView(view as ViewTab)}
            onSelectEmployee={handleSelectEmployee}
          />
        )}

        {currentView === 'analysis' && (
          <EmployeeAnalysisView
            selectedEmployeeId={selectedEmployeeId}
            onNavigateToSimulator={handleNavigateToSimulator}
          />
        )}

        {currentView === 'simulator' && (
          <ScenarioSimulatorView
            initialEmployee={simulatorEmployee}
          />
        )}

        {currentView === 'batch' && (
          <WorkforceBatchView
            onSelectEmployee={handleSelectEmployee}
            onNavigateToAnalysis={() => setCurrentView('analysis')}
          />
        )}

        {currentView === 'methodology' && (
          <MethodologyView />
        )}

        {currentView === 'status' && (
          <SystemStatusView />
        )}
      </main>

      <footer style={{ borderTop: '1px solid var(--border-subtle)', padding: '1.5rem 2rem', textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
        <div style={{ maxWidth: '1440px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <strong>SENTINEL</strong> — Multimodal Workforce Risk Intelligence Platform &copy; 2026
          </div>
          <div style={{ fontFamily: 'var(--font-mono)' }}>
            PyTorch MLP &bull; DistilBERT + LoRA &bull; Calibrated Late Fusion &bull; FastAPI
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
