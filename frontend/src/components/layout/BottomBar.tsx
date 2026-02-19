import './BottomBar.css';

interface Props {
  sessionId: string | null;
  processing: boolean;
  session: any;
  llmOnline: boolean;
  llmActive: boolean | "" | undefined;
  lastLlmEvent: any;
}

export function BottomBar({ sessionId, processing, session, llmOnline, llmActive, lastLlmEvent }: Props) {
  const status = processing ? 'PROCESSING' : session ? 'COMPLETE' : 'READY';
  return (
    <footer className="bottombar">
      <span className="bottombar__item">HATAD v1.0</span>
      <span className="bottombar__sep" />
      <span className="bottombar__item">
        SESSION: <span className="bottombar__val">{sessionId ? sessionId.slice(0, 12) + '...' : 'NONE'}</span>
      </span>
      <span className="bottombar__sep" />
      <span className="bottombar__item">
        STATUS: <span className={`bottombar__val bottombar__val--${status.toLowerCase()}`}>{status}</span>
      </span>
      <span className="bottombar__sep" />
      <span className="bottombar__item">
        LLM: <span className={`bottombar__val ${llmOnline ? 'bottombar__val--online' : 'bottombar__val--offline'}`}>
          {llmOnline ? 'CONNECTED' : 'DISCONNECTED'}
        </span>
      </span>
      {llmActive && lastLlmEvent && (
        <>
          <span className="bottombar__sep" />
          <span className="bottombar__item bottombar__item--active">
            <span className="bottombar__activity-dot" />
            {lastLlmEvent.detail?.task || 'Processing'}
          </span>
        </>
      )}
      <div style={{ flex: 1 }} />
      <span className="bottombar__item bottombar__item--right">Tamil Nadu Land Intelligence</span>
    </footer>
  );
}
