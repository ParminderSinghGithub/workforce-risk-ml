import React, { useState } from 'react';
import { Shield, LayoutDashboard, Users, Sliders, Cpu } from 'lucide-react';
import { HealthResponse } from '../types/api';
import { SystemHealthPopover } from './SystemHealthPopover';

export type PrimaryTab = 'overview' | 'workforce' | 'simulator';

interface HeaderProps {
  currentTab: PrimaryTab;
  onTabChange: (tab: PrimaryTab) => void;
  health: HealthResponse | null;
  isLoadingHealth: boolean;
  onOpenMethodology: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onTabChange,
  health,
  isLoadingHealth,
  onOpenMethodology,
}) => {
  const [showHealthPopover, setShowHealthPopover] = useState<boolean>(false);
  const isHealthy = health?.status === 'healthy';

  return (
    <header className="app-header">
      <div className="header-inner">
        {/* Product Brand Identity */}
        <div
          style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}
          onClick={() => onTabChange('overview')}
        >
          <div style={{ width: '32px', height: '32px', borderRadius: 'var(--radius-sm)', backgroundColor: 'var(--brand-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Shield size={18} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontSize: '1.0625rem', fontWeight: 800, letterSpacing: '-0.02em', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              SENTINEL
              <span style={{ fontSize: '0.625rem', fontWeight: 700, padding: '0.1rem 0.35rem', backgroundColor: 'rgba(59, 130, 246, 0.18)', color: 'var(--brand-light)', borderRadius: 'var(--radius-sm)' }}>
                INTELLIGENCE
              </span>
            </div>
            <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', fontWeight: 500 }}>
              Multimodal Workforce Risk Platform
            </div>
          </div>
        </div>

        {/* 3 Core Primary Navigation Tabs */}
        <nav className="nav-pill-group">
          <button
            className={`nav-pill ${currentTab === 'overview' ? 'active' : ''}`}
            onClick={() => onTabChange('overview')}
          >
            <LayoutDashboard size={14} />
            <span>Overview</span>
          </button>
          <button
            className={`nav-pill ${currentTab === 'workforce' ? 'active' : ''}`}
            onClick={() => onTabChange('workforce')}
          >
            <Users size={14} />
            <span>Workforce</span>
          </button>
          <button
            className={`nav-pill ${currentTab === 'simulator' ? 'active' : ''}`}
            onClick={() => onTabChange('simulator')}
          >
            <Sliders size={14} />
            <span>Simulator</span>
          </button>
        </nav>

        {/* Header Right Secondary Utilities */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', position: 'relative' }}>
          {/* Methodology Modal Trigger */}
          <button
            className="btn btn-secondary"
            style={{ fontSize: '0.75rem', padding: '0.4rem 0.75rem' }}
            onClick={onOpenMethodology}
          >
            <Cpu size={13} color="var(--brand-light)" />
            <span>Methodology & Audit</span>
          </button>

          {/* System Health Status Indicator & Popover Trigger */}
          <button
            className="btn-ghost"
            style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.4rem 0.625rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', cursor: 'pointer', fontSize: '0.75rem' }}
            onClick={() => setShowHealthPopover(!showHealthPopover)}
            title="Click to view API & model checkpoint telemetry"
          >
            <div
              style={{
                width: '7px',
                height: '7px',
                borderRadius: '50%',
                backgroundColor: isLoadingHealth ? 'var(--risk-elevated)' : isHealthy ? 'var(--risk-low)' : 'var(--risk-critical)',
              }}
            />
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
              {isLoadingHealth ? 'CONNECTING' : isHealthy ? 'API READY' : 'OFFLINE'}
            </span>
          </button>

          {/* Health Popover */}
          {showHealthPopover && (
            <SystemHealthPopover
              health={health}
              isLoading={isLoadingHealth}
              onClose={() => setShowHealthPopover(false)}
            />
          )}
        </div>
      </div>
    </header>
  );
};
