import type { PipelineState, CheckResult } from '../../types';
import { RiskGauge } from '../common/RiskGauge';
import './RiskPanel.css';

/* Map raw DET_ codes (sometimes emitted by LLM in red_flags) to friendly text */
const DET_FRIENDLY: Record<string, string> = {
  DET_EC_PERIOD_INVALID: 'EC period is invalid or missing',
  DET_EC_SHORT_PERIOD: 'EC period is too short',
  DET_EC_STALE: 'EC certificate is not recent',
  DET_EC_CHRONO_ORDER: 'Transaction date order inconsistency',
  DET_REG_OUTSIDE_EC: 'Registration date falls outside EC period',
  DET_LIMITATION_PERIOD: 'Transaction beyond limitation period',
  DET_STAMP_DUTY_SHORT: 'Stamp duty shortfall detected',
  DET_UNDERVALUATION: 'Possible property undervaluation',
  DET_IMPLAUSIBLE_LOW: 'Suspiciously low property value',
  DET_IMPLAUSIBLE_HIGH: 'Suspiciously high property value',
  DET_IMPLAUSIBLE_EXTENT: 'Suspicious property extent',
  DET_AREA_MISMATCH: 'Property area mismatch across documents',
  DET_AREA_CONSISTENCY: 'Property area inconsistency across documents',
  DET_BUYER_PATTA_MISMATCH: 'Buyer and Patta owner identity mismatch',
  DET_CHAIN_NAME_GAP: 'Gap in chain of title names',
  DET_SURVEY_TYPE_DIFF: 'Different survey type prefixes across documents',
  DET_SURVEY_SUBDIVISION: 'Survey number subdivision discrepancy',
  DET_SURVEY_OCR_FUZZY: 'Survey number near-match (possible OCR error)',
  DET_SURVEY_MISMATCH: 'Survey number inconsistency across documents',
  DET_EC_INTERNAL_SURVEY_INCONSISTENCY: 'EC internal survey number inconsistency',
  DET_RAPID_FLIPPING: 'Rapid property flipping detected',
  DET_MULTIPLE_SALES: 'Multiple sales detected for same property',
  DET_PLOT_IDENTITY_MISMATCH: 'Plot identity mismatch across documents',
  DET_FINANCIAL_SCALE_JUMP: 'Unusual financial scale jump between transactions',
  DET_MORTGAGE_EXCEEDS_SALE: 'Mortgage amount exceeds sale price',
  DET_ACTIVE_MORTGAGE_BURDEN: 'Active mortgage burden on property',
  DET_MULTI_DISTRICT: 'Documents reference different districts',
  DET_MULTI_TALUK: 'Documents reference different taluks',
  DET_MULTI_VILLAGE: 'Documents reference multiple villages',
  DET_AGE_IMPOSSIBLE: 'Impossible age detected for party',
  DET_MINOR_PARTY: 'Minor party involved in transaction',
  DET_INVALID_SURVEY_FORMAT: 'Invalid survey number format',
  DET_GARBLED_TAMIL: 'Garbled Tamil text in document',
  DET_ALL_AMOUNTS_ROUND: 'All amounts suspiciously round',
  DET_DUPLICATE_VALUES: 'Duplicate values detected',
  DET_EC_IDENTICAL_AMOUNTS: 'Identical amounts across EC entries',
  DET_ZERO_UNCERTAINTY: 'Zero uncertainty score (suspicious)',
  DET_REPEATED_PARTY: 'Repeated party name across transactions',
  DET_IMPLAUSIBLE_NAME: 'Suspicious party name',
  DET_UNPARSEABLE_DATE: 'Unparseable date in document',
  DET_BUYER_PATTA_MATCH: 'Buyer-Patta owner identity verified',
};

function friendlyFlag(raw: string): string {
  const trimmed = raw.trim();
  if (DET_FRIENDLY[trimmed]) return DET_FRIENDLY[trimmed];
  // Fallback: convert DET_FOO_BAR → "Foo bar"
  if (trimmed.startsWith('DET_')) {
    return trimmed.slice(4).toLowerCase().replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase());
  }
  return raw;
}

interface Props {
  processing: boolean;
  session: any;
  riskScore: number | null;
  riskBand: string | null;
  riskColor: string;
  elapsed: number;
  files: any[];
  checks: CheckResult[];
  passCount: number;
  failCount: number;
  warnCount: number;
  redFlags: string[];
  transactions: any[];
  pipeline: PipelineState;
  formatDuration: (s: number) => string;
}

export function RiskPanel({
  processing, session, riskScore, riskBand, riskColor,
  elapsed, files, checks, passCount, failCount, warnCount,
  redFlags, transactions, pipeline, formatDuration,
}: Props) {
  return (
    <aside className="risk-panel">
      <div className="risk-panel__header">
        <span className="material-icons risk-panel__header-icon">shield</span>
        <span className="risk-panel__title">Risk Assessment</span>
      </div>

      <RiskGauge
        score={riskScore}
        band={riskBand}
        color={riskColor}
        processing={processing}
      />

      <div className="risk-panel__metrics">
        {processing && (
          <>
            <Metric label="Elapsed" value={formatDuration(elapsed)} />
            <Metric label="Analysis Tasks" value={pipeline.llmCalls} />
            <Metric
              label="Stage"
              value={pipeline.currentStage.replace('_', ' ').toUpperCase()}
              valueColor="var(--primary)"
            />
          </>
        )}

        {session && !processing && (
          <>
            <Metric label="Documents" value={files.length} />
            <Metric label="Total Checks" value={checks.length} />
            <Metric label="Passed" value={passCount} valueColor="var(--green)" />
            <Metric label="Failed" value={failCount} valueColor="var(--red)" />
            <Metric label="Warnings" value={warnCount} valueColor="var(--amber)" />
            {transactions.length > 0 && <Metric label="EC Entries" value={transactions.length} />}
            {pipeline.mbFactCount > 0 && <Metric label="Facts Found" value={pipeline.mbFactCount} />}
            <Metric label="Analysis Tasks" value={pipeline.llmCalls} />
            <Metric label="Engine Time" value={`${pipeline.totalLlmTime.toFixed(0)}s`} />
          </>
        )}

        {!session && !processing && (
          <div className="risk-panel__empty">
            Upload documents and<br />run analysis to see<br />risk assessment
          </div>
        )}
      </div>

      {/* Red Flags */}
      {session && redFlags.length > 0 && (
        <div className="risk-panel__flags">
          <div className="risk-panel__flags-title">
            <span className="material-icons" style={{ fontSize: 14 }}>warning</span>
            Red Flags
          </div>
          {redFlags.slice(0, 5).map((flag, i) => (
            <div key={i} className="risk-panel__flag">
              <span className="risk-panel__flag-icon">!</span>
              <span>{friendlyFlag(flag)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Deterministic alerts (financial, geographical, property identity) */}
      {session && (() => {
        const alertCodes: Record<string, { icon: string; label: string }> = {
          DET_FINANCIAL_SCALE_JUMP: { icon: 'trending_up', label: 'Financial' },
          DET_MORTGAGE_EXCEEDS_SALE: { icon: 'account_balance', label: 'Financial' },
          DET_ACTIVE_MORTGAGE_BURDEN: { icon: 'account_balance', label: 'Financial' },
          DET_STAMP_DUTY_SHORT: { icon: 'receipt_long', label: 'Financial' },
          DET_UNDERVALUATION: { icon: 'price_check', label: 'Financial' },
          DET_IMPLAUSIBLE_LOW: { icon: 'price_check', label: 'Financial' },
          DET_IMPLAUSIBLE_HIGH: { icon: 'price_check', label: 'Financial' },
          DET_RAPID_FLIPPING: { icon: 'swap_horiz', label: 'Financial' },
          DET_MULTIPLE_SALES: { icon: 'content_copy', label: 'Financial' },
          DET_MULTI_VILLAGE: { icon: 'location_on', label: 'Geography' },
          DET_MULTI_TALUK: { icon: 'location_on', label: 'Geography' },
          DET_MULTI_DISTRICT: { icon: 'location_on', label: 'Geography' },
          DET_PLOT_IDENTITY_MISMATCH: { icon: 'home', label: 'Property' },
          DET_SURVEY_MISMATCH: { icon: 'map', label: 'Survey' },
          DET_EC_INTERNAL_SURVEY_INCONSISTENCY: { icon: 'map', label: 'Survey' },
          DET_SURVEY_OCR_FUZZY: { icon: 'map', label: 'Survey' },
          DET_SURVEY_TYPE_DIFF: { icon: 'map', label: 'Survey' },
          DET_AREA_MISMATCH: { icon: 'square_foot', label: 'Area' },
          DET_AREA_CONSISTENCY: { icon: 'square_foot', label: 'Area' },
          DET_BUYER_PATTA_MISMATCH: { icon: 'person_off', label: 'Identity' },
          DET_CHAIN_NAME_GAP: { icon: 'link_off', label: 'Chain' },
          DET_AGE_IMPOSSIBLE: { icon: 'person_alert', label: 'Identity' },
          DET_MINOR_PARTY: { icon: 'child_care', label: 'Identity' },
        };
        const alerts = checks.filter(c =>
          c.rule_code && alertCodes[c.rule_code] && (c.status === 'FAIL' || c.status === 'WARNING')
        );
        if (alerts.length === 0) return null;
        return (
          <div className="risk-panel__flags">
            <div className="risk-panel__flags-title">
              <span className="material-icons" style={{ fontSize: 14 }}>analytics</span>
              Analysis Alerts
            </div>
            {alerts.slice(0, 5).map((alert, i) => {
              const meta = alertCodes[alert.rule_code!];
              return (
                <div key={i} className="risk-panel__flag">
                  <span className="material-icons" style={{ fontSize: 12, marginRight: 4 }}>{meta.icon}</span>
                  <span><strong>{meta.label}:</strong> {alert.rule_name}</span>
                </div>
              );
            })}
          </div>
        );
      })()}
    </aside>
  );
}

function Metric({ label, value, valueColor }: { label: string; value: string | number; valueColor?: string }) {
  return (
    <div className="metric">
      <span className="metric__label">{label}</span>
      <span className="metric__value" style={valueColor ? { color: valueColor } : undefined}>
        {value}
      </span>
    </div>
  );
}
