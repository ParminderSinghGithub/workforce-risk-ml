import React from 'react';
import { RiskTier } from '../types/api';
import { RiskBadge } from './RiskBadge';

interface RiskGaugeProps {
  probability: number;
  threshold: number;
  tier: RiskTier;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({
  probability,
  threshold,
  tier,
  label = "Fused Attrition Probability",
  size = 'lg',
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

  return (
    <div style={{ textAlign: 'center', padding: size === 'lg' ? '1.5rem' : '0.75rem' }}>
      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '0.5rem' }}>
        {label}
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: '0.25rem', margin: '0.5rem 0' }}>
        <span style={{ fontSize: size === 'lg' ? '3.25rem' : '2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: getTierColor() }}>
          {percentage.toFixed(1)}%
        </span>
      </div>

      <div style={{ margin: '0.75rem 0 1rem' }}>
        <RiskBadge tier={tier} size={size === 'lg' ? 'lg' : 'md'} />
      </div>

      {/* Probability Gauge Bar */}
      <div style={{ position: 'relative', width: '100%', height: '10px', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-full)', overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
        <div
          style={{
            height: '100%',
            width: `${percentage}%`,
            backgroundColor: getTierColor(),
            borderRadius: 'var(--radius-full)',
            transition: 'width 0.4s ease, background-color 0.4s ease',
          }}
        />
        {/* Decision Threshold Marker */}
        <div
          style={{
            position: 'absolute',
            left: `${thresholdPct}%`,
            top: 0,
            bottom: 0,
            width: '2px',
            backgroundColor: '#ffffff',
            boxShadow: '0 0 6px #ffffff',
            zIndex: 10,
          }}
          title={`Validation Operating Threshold: ${thresholdPct.toFixed(1)}%`}
        />
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.35rem', fontFamily: 'var(--font-mono)' }}>
        <span>0%</span>
        <span style={{ color: '#ffffff', fontWeight: 600 }}>Threshold: {thresholdPct.toFixed(1)}%</span>
        <span>100%</span>
      </div>
    </div>
  );
};
