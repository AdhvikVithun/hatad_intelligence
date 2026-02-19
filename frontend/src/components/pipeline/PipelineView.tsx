import type { PipelineState, ProgressEntry } from '../../types';
import { StageTracker } from './StageTracker';
import { StageCards } from './StageCards';
import { VerifyMatrix } from './VerifyMatrix';
import { LLMMonitor } from './LLMMonitor';
import { KnowledgeMonitor } from './KnowledgeMonitor';
import { LiveFeed } from './LiveFeed';
import './PipelineView.css';

interface Props {
  pipeline: PipelineState;
  progress: ProgressEntry[];
  processing: boolean;
  session: any;
  riskScore: number | null;
  riskBand: string | null;
  riskColor: string;
  passCount: number;
  failCount: number;
  warnCount: number;
  elapsed: number;
  llmActive: boolean | "" | undefined;
  progressEndRef: React.RefObject<HTMLDivElement>;
  formatDuration: (s: number) => string;
}

export function PipelineView({
  pipeline, progress, processing, session,
  riskScore, riskBand, riskColor,
  passCount, failCount, warnCount,
  elapsed, llmActive, progressEndRef, formatDuration,
}: Props) {
  return (
    <div className="pipeline-view">
      {/* ── Stage progress rail ── */}
      <StageTracker pipeline={pipeline} />

      {/* ── HATAD engine module cards ── */}
      <StageCards pipeline={pipeline} progress={progress} processing={processing} />

      {/* ── Two-column detail area ── */}
      <div className="pipeline-columns">
        <div className="pipeline-col-left">
          <VerifyMatrix pipeline={pipeline} />
          <LLMMonitor pipeline={pipeline} processing={processing} llmActive={llmActive} />
          <KnowledgeMonitor pipeline={pipeline} processing={processing} />
        </div>
        <div className="pipeline-col-right">
          <LiveFeed progress={progress} processing={processing} progressEndRef={progressEndRef} />
        </div>
      </div>

      {/* ── Result Bar ── */}
      {!processing && session && riskScore !== null && (
        <div className="pipeline-result">
          <div className="pipeline-result__score" style={{ color: riskColor }}>
            <span className="pipeline-result__num">{riskScore}</span>
            <span className="pipeline-result__label">/100</span>
          </div>
          <div className="pipeline-result__band" style={{ background: riskColor }}>
            {riskBand}
          </div>
          <div className="pipeline-result__divider" />
          <div className="pipeline-result__stats">
            <span className="rs pass">{passCount} pass</span>
            <span className="rs fail">{failCount} fail</span>
            <span className="rs warn">{warnCount} warn</span>
          </div>
          <div className="pipeline-result__divider" />
          <div className="pipeline-result__stats">
            <span className="rs">{pipeline.llmCalls} tasks</span>
            <span className="rs">{formatDuration(elapsed)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
