import React from 'react';
import { RiskTier } from '../types/api';
import { ShieldCheck, AlertTriangle, AlertOctagon, Flame } from 'lucide-react';

interface RiskBadgeProps {
  tier: RiskTier;
  size?: 'sm' | 'md' | 'lg';
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({ tier, size = 'md' }) => {
  const getIcon = () => {
    switch (tier) {
      case 'LOW':
        return <ShieldCheck size={size === 'sm' ? 12 : 14} />;
      case 'ELEVATED':
        return <AlertTriangle size={size === 'sm' ? 12 : 14} />;
      case 'HIGH':
        return <AlertOctagon size={size === 'sm' ? 12 : 14} />;
      case 'CRITICAL':
        return <Flame size={size === 'sm' ? 12 : 14} />;
    }
  };

  const badgeClass = `badge badge-${tier.toLowerCase()} ${size === 'sm' ? 'text-xs py-0.5 px-2' : size === 'lg' ? 'text-sm py-1.5 px-3' : ''}`;

  return (
    <span className={badgeClass}>
      {getIcon()}
      <span>{tier} RISK</span>
    </span>
  );
};
