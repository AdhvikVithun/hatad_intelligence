import type { PipelineState, CheckResult } from '../../types';
import { RiskGauge } from '../common/RiskGauge';
import './RiskPanel.css';

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
            <Metric label="LLM Calls" value={pipeline.llmCalls} />
            <Metric label="Tokens In" value={pipeline.totalTokensIn.toLocaleString()} />
            <Metric label="Tokens Out" value={pipeline.totalTokensOut.toLocaleString()} />
            <Metric label="Speed" value={`${pipeline.lastLlmSpeed.toFixed(1)} t/s`} />
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
            {transactions.length > 0 && <Metric label="EC Transactions" value={transactions.length} />}
            {pipeline.ragChunksIndexed > 0 && <Metric label="KB Chunks" value={pipeline.ragChunksIndexed} />}
            {pipeline.ragSearchCount > 0 && <Metric label="RAG Searches" value={pipeline.ragSearchCount} />}
            <Metric label="LLM Calls" value={pipeline.llmCalls} />
            <Metric label="LLM Time" value={`${pipeline.totalLlmTime.toFixed(0)}s`} />
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
              <span>{flag}</span>
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
          DET_MULTI_VILLAGE: { icon: 'location_on', label: 'Geography' },
          DET_MULTI_TALUK: { icon: 'location_on', label: 'Geography' },
          DET_MULTI_DISTRICT: { icon: 'location_on', label: 'Geography' },
          DET_PLOT_IDENTITY_MISMATCH: { icon: 'home', label: 'Property' },
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
