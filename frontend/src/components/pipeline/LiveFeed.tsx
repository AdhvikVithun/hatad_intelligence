import type { ProgressEntry } from '../../types';
import './LiveFeed.css';

interface Props {
  progress: ProgressEntry[];
  processing: boolean;
  progressEndRef: React.RefObject<HTMLDivElement>;
}

/* Map raw stage messages to user-friendly text */
function friendlyMessage(entry: ProgressEntry): string | null {
  const raw = entry.message;
  const llmType = entry.detail?.type || '';

  /* Hide noisy internal events from the user */
  if (llmType === 'llm_start' || llmType === 'llm_waiting' || llmType === 'llm_thinking')
    return null;
  if (llmType === 'llm_done')
    return null;
  if (llmType === 'llm_info' && !raw)
    return null;

  /* Clean up model/internal references */
  let msg = raw
    .replace(/gpt-oss[:\s]?\d*\w*/gi, 'HATAD Intelligence')
    .replace(/ollama/gi, 'HATAD Engine')
    .replace(/sarvam/gi, 'HATAD OCR')
    .replace(/Sarvam AI/gi, 'HATAD OCR')
    .replace(/nomic-embed-text/gi, 'HATAD Embeddings')
    .replace(/LLM/g, 'AI')
    .replace(/llm/g, 'AI')
    .replace(/qwen[\w-]*/gi, 'HATAD Vision');

  return msg;
}

/* Pick an icon for the entry */
function entryIcon(entry: ProgressEntry): string {
  const stage = entry.stage;
  const llmType = entry.detail?.type || '';

  if (stage === 'error' || llmType === 'llm_error' || llmType === 'llm_failed')
    return 'error';
  if (stage === 'complete') return 'check_circle';
  if (llmType === 'llm_response') return 'smart_toy';
  if (llmType === 'llm_tool_call') return 'search';
  if (llmType === 'llm_tool_result') return 'fact_check';
  if (stage === 'extraction') return 'document_scanner';
  if (stage === 'classification') return 'category';
  if (stage === 'data_extraction') return 'schema';
  if (stage === 'knowledge') return 'hub';
  if (stage === 'summarization') return 'compress';
  if (stage === 'verification') return 'verified_user';
  if (stage === 'report') return 'description';
  return 'circle';
}

export function LiveFeed({ progress, processing, progressEndRef }: Props) {
  return (
    <div className="viz-card viz-card--fill">
      <div className="viz-card__header">
        <span className="viz-card__title">ACTIVITY LOG</span>
        <span className="viz-card__sub">{progress.length} events</span>
      </div>
      <div className="live-feed">
        {progress.length === 0 && processing && (
          <div className="feed-waiting">
            <span className="feed-spinner" />
            <span>Starting HATAD analysis pipeline…</span>
          </div>
        )}
        {progress.map((entry, i) => {
          const msg = friendlyMessage(entry);
          if (!msg) return null;

          const isError = entry.stage === 'error' || entry.detail?.type === 'llm_error' || entry.detail?.type === 'llm_failed';
          const isComplete = entry.stage === 'complete';
          const isSuccess = entry.detail?.type === 'llm_response';

          let feedClass = 'feed-entry';
          if (isError) feedClass += ' feed-entry--error';
          else if (isComplete) feedClass += ' feed-entry--complete';
          else if (isSuccess) feedClass += ' feed-entry--success';
          else if (entry.stage === 'knowledge') feedClass += ' feed-entry--rag';

          return (
            <div className={feedClass} key={i}>
              <span className="feed-entry__time">{entry.timestamp.split(' ').pop()}</span>
              <span className="material-icons feed-entry__mi-icon">{entryIcon(entry)}</span>
              <span className="feed-entry__msg">{msg}</span>
            </div>
          );
        })}
        <div ref={progressEndRef as React.LegacyRef<HTMLDivElement>} />
      </div>
    </div>
  );
}
