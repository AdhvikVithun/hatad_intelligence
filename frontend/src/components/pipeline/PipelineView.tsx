import type { PipelineState, ProgressEntry } from '../../types';
import { StageTracker } from './StageTracker';
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
      <StageTracker pipeline={pipeline} />

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

      {/* Result Bar */}
      {!processing && session && riskScore !== null && (
        <div className="pipeline-result">
          <div className="pipeline-result__score" style={{ color: riskColor }}>
            <span className="pipeline-result__num">{riskScore}</span>
            <span className="pipeline-result__label">/100 RISK SCORE</span>
          </div>
          <div className="pipeline-result__band" style={{ background: riskColor, color: '#000' }}>
            {riskBand}
          </div>
          <div className="pipeline-result__stats">
            <span className="rs pass">{passCount} PASS</span>
            <span className="rs fail">{failCount} FAIL</span>
            <span className="rs warn">{warnCount} WARN</span>
          </div>
          <div className="pipeline-result__stats">
            <span className="rs">{pipeline.llmCalls} LLM calls</span>
            <span className="rs">{pipeline.totalLlmTime.toFixed(0)}s LLM time</span>
            <span className="rs">{formatDuration(elapsed)} total</span>
          </div>
        </div>
      )}
    </div>
  );
}
