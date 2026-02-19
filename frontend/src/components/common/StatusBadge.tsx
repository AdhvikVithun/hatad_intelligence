import './StatusBadge.css';

interface Props {
  status: string;
  size?: 'sm' | 'md';
}

export function StatusBadge({ status, size = 'md' }: Props) {
  const s = status.toUpperCase();
  const variant = s === 'PASS' ? 'pass' : s === 'FAIL' ? 'fail' : s === 'WARNING' ? 'warn' : 'neutral';
  return (
    <span className={`status-badge status-badge--${variant} status-badge--${size}`}>
      {status}
    </span>
  );
}

interface SeverityProps {
  severity: string;
  size?: 'sm' | 'md';
}

export function SeverityBadge({ severity, size = 'md' }: SeverityProps) {
  const s = severity.toUpperCase();
  const variant = s === 'CRITICAL' ? 'critical' : s === 'HIGH' ? 'high' : s === 'MEDIUM' ? 'medium' : 'low';
  return (
    <span className={`severity-badge severity-badge--${variant} severity-badge--${size}`}>
      {severity}
    </span>
  );
}
