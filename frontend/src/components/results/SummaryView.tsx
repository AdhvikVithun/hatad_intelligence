import type { CheckResult } from '../../types';
import './SummaryView.css';

interface Props {
  verification: any;
  redFlags: string[];
  criticalChecks: CheckResult[];
  recommendations: string[];
  missingDocs: string[];
}

export function SummaryView({ verification, redFlags, criticalChecks, recommendations, missingDocs }: Props) {
  return (
    <div className="summary-view">
      {/* Executive Summary */}
      <div className="summary-card summary-card--info">
        <div className="summary-card__title">Executive Summary</div>
        <div className="summary-card__text">{verification?.executive_summary || 'No summary available'}</div>
      </div>

      {/* Red Flags */}
      {redFlags.length > 0 && (
        <div className="summary-section">
          <div className="summary-section__label summary-section__label--red">
            Red Flags ({redFlags.length})
          </div>
          {redFlags.map((flag, i) => (
            <div className="summary-card summary-card--critical" key={i}>
              <div className="summary-card__flag-icon">!</div>
              <div className="summary-card__text">{flag}</div>
            </div>
          ))}
        </div>
      )}

      {/* Critical Failures */}
      {criticalChecks.filter(c => c.status === 'FAIL').length > 0 && (
        <div className="summary-section">
          <div className="summary-section__label summary-section__label--red">Critical Failures</div>
          {criticalChecks.filter(c => c.status === 'FAIL').map((check, i) => (
            <div className="summary-card summary-card--critical" key={i}>
              <div className="summary-card__title">{check.rule_name}</div>
              <div className="summary-card__text">{check.explanation}</div>
              <div className="summary-card__rec">{check.recommendation}</div>
            </div>
          ))}
        </div>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="summary-section">
          <div className="summary-section__label summary-section__label--green">Recommendations</div>
          {recommendations.map((rec, i) => (
            <div className="summary-card summary-card--info" key={i}>
              <div className="summary-card__text">{i + 1}. {rec}</div>
            </div>
          ))}
        </div>
      )}

      {/* Missing Docs */}
      {missingDocs.length > 0 && (
        <div className="summary-section">
          <div className="summary-section__label summary-section__label--amber">Documents to Obtain</div>
          {missingDocs.map((doc, i) => (
            <div className="summary-card summary-card--warning" key={i}>
              <div className="summary-card__text">{doc}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
