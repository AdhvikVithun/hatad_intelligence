import type { ProgressEntry } from '../../types';
import './LogView.css';

interface Props {
  progress: ProgressEntry[];
}

/** Friendly stage label */
function stageLabel(entry: ProgressEntry): string {
  const stage = entry.stage;
  const map: Record<string, string> = {
    extraction: 'Scan',
    classification: 'Identify',
    data_extraction: 'Analyse',
    knowledge: 'Knowledge',
    summarization: 'Summarise',
    verification: 'Verify',
    report: 'Report',
    complete: 'Done',
    error: 'Error',
  };
  return map[stage] || stage;
}

/** Scrub model names from message text */
function scrub(msg: string): string {
  return msg
    .replace(/gpt-oss[:\s]?\d*\w*/gi, 'HATAD Intelligence')
    .replace(/ollama/gi, 'HATAD Engine')
    .replace(/sarvam/gi, 'HATAD OCR')
    .replace(/nomic-embed-text/gi, 'HATAD Embeddings')
    .replace(/LLM/g, 'AI')
    .replace(/llm/g, 'AI')
    .replace(/qwen[\w-]*/gi, 'HATAD Vision');
}

export function LogView({ progress }: Props) {
  if (progress.length === 0) {
    return <div className="log-empty">No log entries</div>;
  }

  return (
    <div className="log-view">
      {progress.map((entry, i) => {
        const isError = entry.stage === 'error';
        const isComplete = entry.stage === 'complete';
        const stageClass = [
          isError ? 'log-entry--error' : '',
          isComplete ? 'log-entry--complete' : '',
        ].filter(Boolean).join(' ');

        return (
          <div className={`log-entry ${stageClass}`} key={i}>
            <span className="log-entry__time">{entry.timestamp}</span>
            <span className="log-entry__stage">{stageLabel(entry)}</span>
            <span className="log-entry__msg">{scrub(entry.message)}</span>
          </div>
        );
      })}
    </div>
  );
}
