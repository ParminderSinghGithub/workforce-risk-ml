import React, { useState, useEffect } from 'react';
import { Header, PrimaryTab } from './components/Header';
import { OverviewView } from './views/OverviewView';
import { WorkforceView } from './views/WorkforceView';
import { SimulatorView } from './views/SimulatorView';
import { EmployeeInspector } from './components/EmployeeInspector';
import { MethodologyModal } from './components/MethodologyModal';
import { fetchHealth, predictBatch } from './services/api';
import { EmployeePredictionRequest, HealthResponse, PredictionResponse } from './types/api';
import { SAMPLE_EMPLOYEES } from './constants/sampleData';

export const App: React.FC = () => {
  // Navigation & Modal States
  const [currentTab, setCurrentTab] = useState<PrimaryTab>('overview');
  const [isMethodologyOpen, setIsMethodologyOpen] = useState<boolean>(false);

  // Inspector Drawer State
  const [inspectorEmployee, setInspectorEmployee] = useState<EmployeePredictionRequest | null>(null);
  const [inspectorPrediction, setInspectorPrediction] = useState<PredictionResponse | null>(null);
  const [isInspectorOpen, setIsInspectorOpen] = useState<boolean>(false);

  // Simulator Context State
  const [simulatorEmployee, setSimulatorEmployee] = useState<EmployeePredictionRequest>(SAMPLE_EMPLOYEES[1]);

  // Telemetry & Cohort Predictions State
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoadingHealth, setIsLoadingHealth] = useState<boolean>(true);
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [isLoadingBatch, setIsLoadingBatch] = useState<boolean>(true);

  // Poll Health
  const checkHealth = async () => {
    try {
      setIsLoadingHealth(true);
      const res = await fetchHealth();
      setHealth(res);
    } catch {
      setHealth(null);
    } finally {
      setIsLoadingHealth(false);
    }
  };

  // Load Cohort Predictions
  const loadBatch = async () => {
    try {
      setIsLoadingBatch(true);
      const res = await predictBatch(SAMPLE_EMPLOYEES);
      setPredictions(res.predictions);
    } catch (err) {
      console.error('Failed to load batch predictions:', err);
    } finally {
      setIsLoadingBatch(false);
    }
  };

  useEffect(() => {
    checkHealth();
    loadBatch();

    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handlers
  const handleInspectEmployee = (employee: EmployeePredictionRequest, prediction: PredictionResponse) => {
    setInspectorEmployee(employee);
    setInspectorPrediction(prediction);
    setIsInspectorOpen(true);
  };

  const handleSendToSimulator = (employee: EmployeePredictionRequest) => {
    setSimulatorEmployee(employee);
    setCurrentTab('simulator');
  };

  return (
    <div className="app-container">
      {/* Redesigned Header with 3 Core Tabs & Utilities */}
      <Header
        currentTab={currentTab}
        onTabChange={setCurrentTab}
        health={health}
        isLoadingHealth={isLoadingHealth}
        onOpenMethodology={() => setIsMethodologyOpen(true)}
      />

      {/* Main Operational Views */}
      <main className="main-content">
        {currentTab === 'overview' && (
          <OverviewView
            predictions={predictions}
            isLoading={isLoadingBatch}
            onInspectEmployee={handleInspectEmployee}
            onNavigateToWorkforce={() => setCurrentTab('workforce')}
            onNavigateToSimulator={handleSendToSimulator}
          />
        )}

        {currentTab === 'workforce' && (
          <WorkforceView
            predictions={predictions}
            isLoading={isLoadingBatch}
            onRefresh={loadBatch}
            onInspectEmployee={handleInspectEmployee}
          />
        )}

        {currentTab === 'simulator' && (
          <SimulatorView
            initialEmployee={simulatorEmployee}
          />
        )}
      </main>

      {/* Slide-out Contextual Employee Inspector Drawer */}
      <EmployeeInspector
        employee={inspectorEmployee}
        prediction={inspectorPrediction}
        isOpen={isInspectorOpen}
        onClose={() => setIsInspectorOpen(false)}
        onSendToSimulator={handleSendToSimulator}
      />

      {/* Secondary Methodology & Audit Modal */}
      <MethodologyModal
        isOpen={isMethodologyOpen}
        onClose={() => setIsMethodologyOpen(false)}
      />

      {/* Enterprise Footer */}
      <footer style={{ borderTop: '1px solid var(--border-subtle)', padding: '1.5rem 2rem', fontSize: '0.8125rem', color: 'var(--text-muted)', backgroundColor: 'var(--bg-surface)' }}>
        <div style={{ maxWidth: '1360px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <span style={{ fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>SENTINEL</span>
            <span style={{ margin: '0 0.5rem' }}>&bull;</span>
            <span>Built by <strong style={{ color: 'var(--text-primary)' }}>Parminder Singh</strong></span>
            <span style={{ margin: '0 0.5rem' }}>&bull;</span>
            <a
              href="https://github.com/ParminderSinghGithub/Sentinel"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: 'var(--brand-light)', textDecoration: 'none', fontWeight: 600 }}
            >
              GitHub: https://github.com/ParminderSinghGithub/Sentinel
            </a>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            PyTorch MLP &bull; DistilBERT + LoRA &bull; Calibrated Late Fusion &bull; FastAPI
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
