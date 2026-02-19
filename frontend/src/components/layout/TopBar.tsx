import type { PipelineState } from '../../types';
import './TopBar.css';

interface Props {
  processing: boolean;
  session: any;
  riskScore: number | null;
  riskBand: string | null;
  riskColor: string;
  elapsed: number;
  pipeline: PipelineState;
  llmOnline: boolean;
  clock: string;
  formatDuration: (s: number) => string;
}

export function TopBar({
  processing, session, riskScore, riskBand, riskColor,
  elapsed, pipeline, llmOnline, clock, formatDuration,
}: Props) {
  return (
    <header className="topbar">
      <div className="topbar__brand">
        <span className="topbar__logo">H</span>
        <div className="topbar__brand-text">
          <span className="topbar__name">HATAD</span>
          <span className="topbar__sub">Land Intelligence</span>
        </div>
      </div>

      <div className="topbar__center">
        {processing && (
          <div className="topbar__status topbar__status--active">
            <span className="topbar__pulse" />
            <span className="topbar__status-label">ANALYZING</span>
            <span className="topbar__status-timer">{formatDuration(elapsed)}</span>
            <span className="topbar__status-stage">
              {pipeline.currentStage.toUpperCase().replace('_', ' ')}
            </span>
          </div>
        )}
        {!processing && session && (
          <div className="topbar__status topbar__status--done">
            <span className="material-icons" style={{ fontSize: 14, color: riskColor }}>verified</span>
            <span className="topbar__status-label">COMPLETE</span>
            <span className="topbar__status-score" style={{ color: riskColor }}>
              RISK {riskScore}/100
            </span>
            <span className="topbar__status-band" style={{ color: riskColor }}>
              {riskBand}
            </span>
          </div>
        )}

        {processing && (
          <div className="topbar__metrics">
            <div className="topbar__metric">
              <span className="material-icons topbar__metric-icon">psychology</span>
              <span className="topbar__metric-value">{pipeline.llmCalls}</span>
              <span className="topbar__metric-label">calls</span>
            </div>
            <div className="topbar__metric">
              <span className="material-icons topbar__metric-icon">speed</span>
              <span className="topbar__metric-value">{pipeline.lastLlmSpeed > 0 ? pipeline.lastLlmSpeed.toFixed(0) : '—'}</span>
              <span className="topbar__metric-label">tok/s</span>
            </div>
            <div className="topbar__metric">
              <span className="material-icons topbar__metric-icon">timer</span>
              <span className="topbar__metric-value">{formatDuration(elapsed)}</span>
            </div>
          </div>
        )}
      </div>

      <div className="topbar__right">
        <div className="topbar__indicator">
          <span className={`topbar__dot ${llmOnline ? 'online' : 'offline'}`} />
          <span>LLM {llmOnline ? 'ONLINE' : 'OFFLINE'}</span>
        </div>
        <span className="topbar__clock">{clock}</span>
      </div>
    </header>
  );
}
