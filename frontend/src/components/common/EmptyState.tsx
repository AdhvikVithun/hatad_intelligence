import './EmptyState.css';

interface Props {
  title?: string;
  description?: string;
  icon?: string;
}

export function EmptyState({
  title = 'No data yet',
  description = 'Upload documents and run analysis to see results',
  icon = 'inbox',
}: Props) {
  return (
    <div className="empty-state">
      <span className="material-icons empty-state__icon">{icon}</span>
      <div className="empty-state__title">{title}</div>
      <div className="empty-state__desc">{description}</div>
    </div>
  );
}
