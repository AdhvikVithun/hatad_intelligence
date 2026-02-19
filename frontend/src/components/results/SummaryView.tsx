import type { CheckResult } from '../../types';
import './SummaryView.css';

/* Convert DET_ codes to friendly text */
function friendlyFlag(raw: string): string {
  const map: Record<string, string> = {
    DET_SURVEY_MISMATCH: 'Survey number inconsistency across documents',
    DET_EC_INTERNAL_SURVEY_INCONSISTENCY: 'EC internal survey number inconsistency',
    DET_SURVEY_OCR_FUZZY: 'Survey number near-match (possible OCR error)',
    DET_SURVEY_SUBDIVISION: 'Survey number subdivision discrepancy',
    DET_SURVEY_TYPE_DIFF: 'Different survey type prefixes across documents',
    DET_AREA_MISMATCH: 'Property area mismatch across documents',
    DET_AREA_CONSISTENCY: 'Property area inconsistency across documents',
    DET_BUYER_PATTA_MISMATCH: 'Buyer and Patta owner identity mismatch',
    DET_CHAIN_NAME_GAP: 'Gap in chain of title names',
    DET_PLOT_IDENTITY_MISMATCH: 'Plot identity mismatch across documents',
    DET_FINANCIAL_SCALE_JUMP: 'Unusual financial scale jump between transactions',
    DET_MORTGAGE_EXCEEDS_SALE: 'Mortgage amount exceeds sale price',
    DET_ACTIVE_MORTGAGE_BURDEN: 'Active mortgage burden on property',
    DET_RAPID_FLIPPING: 'Rapid property flipping detected',
    DET_MULTIPLE_SALES: 'Multiple sales detected for same property',
    DET_MULTI_DISTRICT: 'Documents reference different districts',
    DET_MULTI_TALUK: 'Documents reference different taluks',
    DET_MULTI_VILLAGE: 'Documents reference multiple villages',
    DET_EC_PERIOD_INVALID: 'EC period is invalid',
    DET_EC_SHORT_PERIOD: 'EC period is too short',
    DET_EC_STALE: 'EC certificate is not recent',
    DET_LIMITATION_PERIOD: 'Transaction beyond limitation period',
    DET_REG_OUTSIDE_EC: 'Registration date outside EC period',
    DET_STAMP_DUTY_SHORT: 'Stamp duty shortfall detected',
    DET_UNDERVALUATION: 'Possible property undervaluation',
  };
  const trimmed = raw.trim();
  if (map[trimmed]) return map[trimmed];
  if (trimmed.startsWith('DET_')) {
    return trimmed.slice(4).toLowerCase().replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase());
  }
  return raw;
}

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
              <div className="summary-card__text">{friendlyFlag(flag)}</div>
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
