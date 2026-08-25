import React from 'react';
import { RiskTier } from '../types/api';
import { ShieldCheck, AlertTriangle, AlertOctagon, Flame } from 'lucide-react';

interface RiskBadgeProps {
  tier: RiskTier;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({
  tier,
  size = 'md',
  showIcon = true,
}) => {
  const getIcon = () => {
    const iconSize = size === 'sm' ? 12 : size === 'lg' ? 15 : 13;
    switch (tier) {
      case 'LOW':
        return <ShieldCheck size={iconSize} />;
      case 'ELEVATED':
        return <AlertTriangle size={iconSize} />;
      case 'HIGH':
        return <AlertOctagon size={iconSize} />;
      case 'CRITICAL':
        return <Flame size={iconSize} />;
    }
  };

  const badgeClass = `risk-badge risk-badge-${tier.toLowerCase()} ${
    size === 'sm' ? 'text-xs py-0.5 px-2' : size === 'lg' ? 'text-sm py-1 px-3 font-bold' : ''
  }`;

  return (
    <span className={badgeClass}>
      {showIcon && getIcon()}
      <span>{tier} RISK</span>
    </span>
  );
};
