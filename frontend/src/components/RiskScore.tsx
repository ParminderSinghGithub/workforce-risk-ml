import React from 'react';
import { RiskTier } from '../types/api';
import { RiskBadge } from './RiskBadge';

interface RiskScoreProps {
  probability: number;
  threshold?: number;
  tier: RiskTier;
  size?: 'sm' | 'md' | 'lg';
  showBar?: boolean;
}

export const RiskScore: React.FC<RiskScoreProps> = ({
  probability,
  threshold = 0.2189,
  tier,
  size = 'md',
  showBar = true,
}) => {
  const percentage = Math.min(100, Math.max(0, probability * 100));
  const thresholdPct = Math.min(100, Math.max(0, threshold * 100));

  const getTierColor = () => {
    switch (tier) {
      case 'LOW': return 'var(--risk-low)';
      case 'ELEVATED': return 'var(--risk-elevated)';
      case 'HIGH': return 'var(--risk-high)';
      case 'CRITICAL': return 'var(--risk-critical)';
    }
  };

  if (size === 'sm') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.9375rem', color: getTierColor() }}>
          {percentage.toFixed(1)}%
        </span>
        <RiskBadge tier={tier} size="sm" showIcon={false} />
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
        <div>
          <span style={{ fontSize: size === 'lg' ? '2.5rem' : '1.75rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: getTierColor() }}>
            {percentage.toFixed(1)}%
          </span>
          <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginLeft: '0.4rem', fontWeight: 500 }}>
            Exit Risk
          </span>
        </div>
        <RiskBadge tier={tier} size={size === 'lg' ? 'md' : 'sm'} />
      </div>

      {showBar && (
        <div style={{ position: 'relative', width: '100%', height: '6px', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-full)', overflow: 'hidden' }}>
          <div
            style={{
              height: '100%',
              width: `${percentage}%`,
              backgroundColor: getTierColor(),
              borderRadius: 'var(--radius-full)',
              transition: 'width 0.3s ease',
            }}
          />
          {/* Validation Threshold Tick */}
          <div
            style={{
              position: 'absolute',
              left: `${thresholdPct}%`,
              top: 0,
              bottom: 0,
              width: '2px',
              backgroundColor: '#ffffff',
              opacity: 0.85,
            }}
            title={`Operating Decision Threshold: ${thresholdPct.toFixed(1)}%`}
          />
        </div>
      )}
    </div>
  );
};
