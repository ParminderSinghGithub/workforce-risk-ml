import React from 'react';

interface StatCardProps {
  label: string;
  value: string | number;
  sublabel?: string;
  trend?: string;
  trendType?: 'positive' | 'negative' | 'neutral';
  icon?: React.ReactNode;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  sublabel,
  trend,
  trendType = 'neutral',
  icon,
}) => {
  const getTrendColor = () => {
    if (trendType === 'positive') return 'var(--risk-low)';
    if (trendType === 'negative') return 'var(--risk-critical)';
    return 'var(--text-muted)';
  };

  return (
    <div className="panel" style={{ padding: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            {label}
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.25rem', fontFamily: 'var(--font-sans)', letterSpacing: '-0.02em' }}>
            {value}
          </div>
        </div>
        {icon && (
          <div style={{ padding: '0.5rem', borderRadius: 'var(--radius-md)', backgroundColor: 'var(--bg-surface-raised)', color: 'var(--brand-light)' }}>
            {icon}
          </div>
        )}
      </div>

      {(sublabel || trend) && (
        <div style={{ marginTop: '0.5rem', fontSize: '0.8125rem', color: 'var(--text-muted)', display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
          {trend && <span style={{ fontWeight: 700, color: getTrendColor() }}>{trend}</span>}
          {sublabel && <span>{sublabel}</span>}
        </div>
      )}
    </div>
  );
};
