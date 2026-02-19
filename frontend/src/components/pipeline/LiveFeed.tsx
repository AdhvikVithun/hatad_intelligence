import type { ProgressEntry } from '../../types';
import './LiveFeed.css';

interface Props {
  progress: ProgressEntry[];
  processing: boolean;
  progressEndRef: React.RefObject<HTMLDivElement>;
}

export function LiveFeed({ progress, processing, progressEndRef }: Props) {
  return (
    <div className="viz-card viz-card--fill">
      <div className="viz-card__header">
        <span className="viz-card__title">LIVE FEED</span>
        <span className="viz-card__sub">{progress.length} events</span>
      </div>
      <div className="live-feed">
        {progress.length === 0 && processing && (
          <div className="feed-waiting">
            <span className="feed-spinner" />
            <span>Initializing pipeline...</span>
          </div>
        )}
        {progress.map((entry, i) => {
          const isLlm = entry.detail?.type?.startsWith('llm_');
          const llmType = entry.detail?.type || '';
          const isResponse = llmType === 'llm_response';
          const isDone = llmType === 'llm_done';
          const isError = llmType === 'llm_error' || llmType === 'llm_failed' || llmType === 'llm_parse_error';
          const isStart = llmType === 'llm_start';
          const isWaiting = llmType === 'llm_waiting';
          const isThinking = llmType === 'llm_thinking';
          const isToolCall = llmType === 'llm_tool_call';
          const isToolResult = llmType === 'llm_tool_result';

          let feedClass = 'feed-entry';
          if (entry.stage === 'error') feedClass += ' feed-entry--error';
          else if (entry.stage === 'complete') feedClass += ' feed-entry--complete';
          else if (entry.stage === 'knowledge') feedClass += ' feed-entry--rag';
          else if (isError) feedClass += ' feed-entry--error';
          else if (isResponse) feedClass += ' feed-entry--success';
          else if (isDone) feedClass += ' feed-entry--dim';
          else if (isStart) feedClass += ' feed-entry--start';
          else if (isWaiting) feedClass += ' feed-entry--waiting';
          else if (isThinking) feedClass += ' feed-entry--thinking';
          else if (isToolCall) feedClass += ' feed-entry--tool';
          else if (isToolResult) feedClass += ' feed-entry--tool-result';
          else if (isLlm) feedClass += ' feed-entry--llm';

          const stageIcon = isError ? '!' :
            isResponse ? '+' : isDone ? '·' : isStart ? '>' :
            isWaiting ? 'o' : isThinking ? '~' :
            isToolCall ? 'f' : isToolResult ? '<' :
            entry.stage === 'complete' ? '*' :
            entry.stage === 'verification' ? '#' :
            entry.stage === 'knowledge' ? 'K' : '-';

          return (
            <div className={feedClass} key={i}>
              <span className="feed-entry__time">{entry.timestamp.split(' ').pop()}</span>
              <span className="feed-entry__icon">{stageIcon}</span>
              <span className="feed-entry__msg">{entry.message}</span>
              {isLlm && entry.detail && (
                <span className="feed-entry__badges">
                  {isStart && entry.detail.schema_enforced && <span className="fb fb--schema">SCHEMA</span>}
                  {isStart && entry.detail.thinking_enabled && <span className="fb fb--thinking">CoT</span>}
                  {isStart && entry.detail.tools_enabled && <span className="fb fb--tools">TOOLS</span>}
                  {isStart && entry.detail.prompt_tokens_est != null && (
                    <span className="fb fb--tok-in">~{(entry.detail.prompt_tokens_est/1000).toFixed(1)}K in</span>
                  )}
                  {isResponse && entry.detail.response_tokens != null && (
                    <span className="fb fb--tok-out">{entry.detail.response_tokens} out</span>
                  )}
                  {isResponse && entry.detail.tokens_per_sec != null && (
                    <span className="fb fb--speed">{entry.detail.tokens_per_sec} t/s</span>
                  )}
                  {isResponse && entry.detail.elapsed_seconds != null && (
                    <span className="fb fb--time">{entry.detail.elapsed_seconds}s</span>
                  )}
                  {isThinking && entry.detail.thinking_chars != null && (
                    <span className="fb fb--thinking">{(entry.detail.thinking_chars/1000).toFixed(1)}K chars</span>
                  )}
                  {isToolCall && entry.detail.tool_name && (
                    <span className="fb fb--tools">{entry.detail.tool_name}()</span>
                  )}
                  {isToolCall && entry.detail.round != null && entry.detail.round > 1 && (
                    <span className="fb fb--round">round {entry.detail.round}</span>
                  )}
                  {isToolResult && entry.detail.tool_name && (
                    <span className="fb fb--tools">{entry.detail.tool_name}</span>
                  )}
                  {entry.detail.attempt != null && entry.detail.attempt > 1 && (
                    <span className="fb fb--retry">retry {entry.detail.attempt}</span>
                  )}
                </span>
              )}
            </div>
          );
        })}
        <div ref={progressEndRef as React.LegacyRef<HTMLDivElement>} />
      </div>
    </div>
  );
}
