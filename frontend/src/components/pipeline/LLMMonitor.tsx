import type { PipelineState } from '../../types';
import './LLMMonitor.css';

interface Props {
  pipeline: PipelineState;
  processing: boolean;
  llmActive: boolean | "" | undefined;
}

export function LLMMonitor({ pipeline, processing, llmActive }: Props) {
  return (
    <div className="viz-card">
      <div className="viz-card__header">
        <span className="viz-card__title">LLM MONITOR</span>
        <span className="viz-card__sub">
          {llmActive ? (
            <><span className="live-dot" /> ACTIVE</>
          ) : processing ? 'IDLE' : 'COMPLETE'}
        </span>
      </div>
      <div className="llm-monitor">
        <div className="llm-grid">
          <LLMStat value={pipeline.llmCalls} label="LLM Calls" />
          <LLMStat value={`${(pipeline.totalTokensIn / 1000).toFixed(1)}K`} label="Tokens In" />
          <LLMStat value={`${(pipeline.totalTokensOut / 1000).toFixed(1)}K`} label="Tokens Out" />
          <LLMStat value={`${pipeline.totalLlmTime.toFixed(0)}s`} label="LLM Time" />
          <LLMStat value={pipeline.lastLlmSpeed.toFixed(1)} label="tok/s" />
          <LLMStat
            value={pipeline.currentDataSize > 0 ? `${(pipeline.currentDataSize / 1000).toFixed(0)}K` : '—'}
            label="Input chars"
          />
        </div>

        {llmActive && pipeline.lastLlmTask && (
          <div className="llm-task">
            <span className="llm-task__dot" />
            <span className="llm-task__text">{pipeline.lastLlmTask}</span>
          </div>
        )}

        {(pipeline.schemaEnforcedCalls > 0 || pipeline.thinkingCalls > 0 || pipeline.toolCallCount > 0) && (
          <div className="llm-caps">
            {pipeline.schemaEnforcedCalls > 0 && (
              <span className="cap-badge cap-badge--schema">SCH x{pipeline.schemaEnforcedCalls}</span>
            )}
            {pipeline.thinkingCalls > 0 && (
              <span className="cap-badge cap-badge--thinking">CoT x{pipeline.thinkingCalls} ({(pipeline.totalThinkingChars/1000).toFixed(1)}K)</span>
            )}
            {pipeline.toolCallCount > 0 && (
              <span className="cap-badge cap-badge--tools">FN x{pipeline.toolCallCount}</span>
            )}
            {pipeline.ragSearchCount > 0 && (
              <span className="cap-badge cap-badge--rag">RAG x{pipeline.ragSearchCount}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function LLMStat({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="llm-stat">
      <div className="llm-stat__value">{value}</div>
      <div className="llm-stat__label">{label}</div>
    </div>
  );
}
