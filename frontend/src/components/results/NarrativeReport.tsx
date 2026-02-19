import './NarrativeReport.css';

interface Props {
  report: string | null;
}

export function NarrativeReport({ report }: Props) {
  if (!report) {
    return <div className="report-empty">No narrative report available</div>;
  }
  return (
    <div className="report-view">
      <div className="report-content">{report}</div>
    </div>
  );
}
