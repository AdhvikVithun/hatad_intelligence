import type { ProgressEntry } from '../../types';
import './LogView.css';

interface Props {
  progress: ProgressEntry[];
}

export function LogView({ progress }: Props) {
  if (progress.length === 0) {
    return <div className="log-empty">No log entries</div>;
  }

  return (
    <div className="log-view">
      {progress.map((entry, i) => {
        const isLlm = entry.detail?.type?.startsWith('llm_');
        const llmType = entry.detail?.type || '';
        const stageClass = [
          entry.stage === 'error' ? 'log-entry--error' : '',
          entry.stage === 'complete' ? 'log-entry--complete' : '',
          isLlm ? `log-entry--llm log-entry--${llmType.replace('llm_', '')}` : '',
        ].filter(Boolean).join(' ');

        return (
          <div className={`log-entry ${stageClass}`} key={i}>
            <span className="log-entry__time">{entry.timestamp}</span>
            <span className={`log-entry__stage ${isLlm ? 'log-entry__stage--llm' : ''}`}>
              {isLlm ? llmType.replace('llm_', '').replace('_', ' ') : entry.stage}
            </span>
            <span className="log-entry__msg">{entry.message}</span>
            {isLlm && entry.detail && (
              <span className="log-entry__badges">
                {entry.detail.model && llmType === 'llm_start' && (
                  <span className={`lb ${entry.detail.model.includes('Vision') ? 'lb--vision' : 'lb--reasoning'}`}>
                    {entry.detail.model}
                  </span>
                )}
                {entry.detail.image_count != null && llmType === 'llm_start' && (
                  <span className="lb lb--pages">{entry.detail.image_count} page{entry.detail.image_count !== 1 ? 's' : ''} scanned</span>
                )}
                {entry.detail.prompt_tokens_est != null && llmType === 'llm_start' && (
                  <span className="lb lb--tokens">~{entry.detail.prompt_tokens_est.toLocaleString()} tok in</span>
                )}
                {entry.detail.response_tokens != null && (
                  <span className="lb lb--tokens">{entry.detail.response_tokens.toLocaleString()} tok out</span>
                )}
                {entry.detail.tokens_per_sec != null && (
                  <span className="lb lb--speed">{entry.detail.tokens_per_sec} tok/s</span>
                )}
                {entry.detail.elapsed_seconds != null && llmType === 'llm_response' && (
                  <span className="lb lb--time">{entry.detail.elapsed_seconds}s</span>
                )}
                {entry.detail.total_seconds != null && (
                  <span className="lb lb--time">{entry.detail.total_seconds}s total</span>
                )}
                {entry.detail.attempt != null && entry.detail.attempt > 1 && (
                  <span className="lb lb--retry">attempt {entry.detail.attempt}</span>
                )}
                {entry.detail.error && (
                  <span className="lb lb--error" title={entry.detail.error}>
                    {entry.detail.error.substring(0, 40)}
                  </span>
                )}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
