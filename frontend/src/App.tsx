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
          <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
            <span style={{ fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>&copy; 2026 Sentinel</span>
            <span>&bull;</span>
            <span>Built by <strong style={{ color: 'var(--text-primary)' }}>Parminder Singh</strong></span>
            <span>&bull;</span>
            <a
              href="https://github.com/ParminderSinghGithub/Sentinel"
              target="_blank"
              rel="noopener noreferrer"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', color: 'var(--brand-light)', textDecoration: 'none', fontWeight: 600 }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style={{ flexShrink: 0 }}>
                <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
              </svg>
              <span>GitHub</span>
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
