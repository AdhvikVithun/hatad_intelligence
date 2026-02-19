import type { PipelineState } from '../../types';
import './LLMMonitor.css';

interface Props {
  pipeline: PipelineState;
  processing: boolean;
  llmActive: boolean | "" | undefined;
}

export function LLMMonitor({ pipeline, processing, llmActive }: Props) {
  /* Derive user-friendly task description */
  const friendlyTask = (raw: string): string => {
    if (!raw) return '';
    const lower = raw.toLowerCase();
    if (lower.includes('classif'))     return 'Identifying document type…';
    if (lower.includes('extract'))     return 'Extracting key fields…';
    if (lower.includes('summar'))      return 'Summarising documents…';
    if (lower.includes('verify'))      return 'Running verification checks…';
    if (lower.includes('narrative'))   return 'Writing due diligence report…';
    if (lower.includes('chain'))       return 'Analysing chain of title…';
    if (lower.includes('knowledge') || lower.includes('memory'))
      return 'Building knowledge base…';
    return 'Analysing…';
  };

  return (
    <div className="viz-card">
      <div className="viz-card__header">
        <span className="viz-card__title">HATAD ENGINE</span>
        <span className="viz-card__sub">
          {llmActive ? (
            <><span className="live-dot" /> PROCESSING</>
          ) : processing ? 'STANDBY' : 'COMPLETE'}
        </span>
      </div>
      <div className="llm-monitor">
        <div className="llm-grid">
          <HATADStat value={pipeline.llmCalls} label="Tasks Run" />
          <HATADStat value={`${pipeline.totalLlmTime.toFixed(0)}s`} label="Engine Time" />
          <HATADStat
            value={pipeline.toolCallCount > 0 ? pipeline.toolCallCount : '—'}
            label="Lookups"
          />
        </div>

        {llmActive && pipeline.lastLlmTask && (
          <div className="llm-task">
            <span className="llm-task__dot" />
            <span className="llm-task__text">{friendlyTask(pipeline.lastLlmTask)}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function HATADStat({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="llm-stat">
      <div className="llm-stat__value">{value}</div>
      <div className="llm-stat__label">{label}</div>
    </div>
  );
}
