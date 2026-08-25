import React from 'react';
import { Shield, LayoutDashboard, UserCheck, Sliders, Users, Cpu, Activity } from 'lucide-react';
import { HealthResponse } from '../types/api';

export type ViewTab = 'dashboard' | 'analysis' | 'simulator' | 'batch' | 'methodology' | 'status';

interface HeaderProps {
  currentView: ViewTab;
  onViewChange: (view: ViewTab) => void;
  health: HealthResponse | null;
  isLoadingHealth: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  currentView,
  onViewChange,
  health,
  isLoadingHealth,
}) => {
  const isHealthy = health?.status === 'healthy';

  return (
    <header style={{ backgroundColor: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-subtle)', position: 'sticky', top: 0, zIndex: 50 }}>
      <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '0.75rem 2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }} onClick={() => onViewChange('dashboard')}>
          <div style={{ width: '36px', height: '36px', borderRadius: 'var(--radius-md)', backgroundColor: 'var(--brand-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: 'var(--shadow-glow)' }}>
            <Shield size={20} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.03em', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              SENTINEL
              <span style={{ fontSize: '0.65rem', fontWeight: 700, padding: '0.15rem 0.45rem', backgroundColor: 'rgba(59, 130, 246, 0.2)', color: '#60a5fa', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                ENTERPRISE
              </span>
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 500, letterSpacing: '0.02em' }}>
              Multimodal Workforce Risk Intelligence Platform
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{ display: 'flex', gap: '0.25rem', backgroundColor: 'var(--bg-primary)', padding: '0.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          {[
            { id: 'dashboard', label: 'Executive Dashboard', icon: LayoutDashboard },
            { id: 'analysis', label: 'Employee Analysis', icon: UserCheck },
            { id: 'simulator', label: 'Scenario Simulator', icon: Sliders },
            { id: 'batch', label: 'Workforce View', icon: Users },
            { id: 'methodology', label: 'Methodology', icon: Cpu },
            { id: 'status', label: 'System Status', icon: Activity },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = currentView === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onViewChange(tab.id as ViewTab)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  padding: '0.5rem 0.85rem',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.825rem',
                  fontWeight: 600,
                  border: 'none',
                  cursor: 'pointer',
                  backgroundColor: isActive ? 'var(--bg-card)' : 'transparent',
                  color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                  boxShadow: isActive ? 'var(--shadow-sm)' : 'none',
                  transition: 'all 0.15s ease',
                }}
              >
                <Icon size={15} color={isActive ? 'var(--brand-accent)' : 'currentColor'} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Backend Health Status Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          <div
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: isLoadingHealth ? '#f59e0b' : isHealthy ? '#10b981' : '#ef4444',
              boxShadow: isHealthy ? '0 0 8px #10b981' : 'none',
            }}
          />
          <span style={{ fontFamily: 'var(--font-mono)' }}>
            {isLoadingHealth ? 'CONNECTING...' : isHealthy ? 'API: READY (OFFLINE)' : 'API: DISCONNECTED'}
          </span>
        </div>
      </div>
    </header>
  );
};
