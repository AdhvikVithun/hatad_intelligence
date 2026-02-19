import type { DocFile, SessionHistoryItem } from '../../types';
import './Sidebar.css';

interface Props {
  files: DocFile[];
  uploading: boolean;
  processing: boolean;
  session: any;
  sessionId: string | null;
  dragging: boolean;
  llmOnline: boolean;
  sessionHistory: SessionHistoryItem[];
  fileInputRef: React.RefObject<HTMLInputElement>;
  onUpload: (files: FileList | File[]) => void;
  onDrop: (e: React.DragEvent) => void;
  onStartAnalysis: () => void;
  onClear: () => void;
  onLoadHistory: () => void;
  onLoadSession: (sid: string) => void;
  setDragging: (v: boolean) => void;
  getReportPdfUrl: (sid: string) => string;
}

export function Sidebar({
  files, uploading, processing, session, sessionId,
  dragging, llmOnline, sessionHistory,
  fileInputRef, onUpload, onDrop, onStartAnalysis, onClear,
  onLoadHistory, onLoadSession, setDragging, getReportPdfUrl,
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar__header">
        <div className="sidebar__header-left">
          <span className="material-icons sidebar__header-icon">folder_open</span>
          <span className="sidebar__title">Documents</span>
        </div>
        {files.length > 0 && !processing && (
          <button className="sidebar__clear-btn" onClick={onClear}>
            <span className="material-icons" style={{ fontSize: 12 }}>close</span>
            Clear
          </button>
        )}
      </div>

      <div className="sidebar__content">
        {/* Drop Zone */}
        <div
          className={`dropzone ${dragging ? 'dropzone--active' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <span className="material-icons dropzone__icon">upload_file</span>
          <div className="dropzone__text">
            {uploading ? 'Uploading...' : 'Drop PDFs here'}
          </div>
          <div className="dropzone__hint">EC, Patta, Sale Deed, FMB</div>
          <input
            ref={fileInputRef as React.LegacyRef<HTMLInputElement>}
            type="file"
            accept=".pdf"
            multiple
            style={{ display: 'none' }}
            onChange={(e) => e.target.files && onUpload(e.target.files)}
          />
        </div>

        {/* File Cards */}
        {files.map((f, i) => (
          <div className="file-card" key={i}>
            <div className={`file-card__badge file-card__badge--${(f.document_type || 'other').toLowerCase()}`}>
              {(f.document_type || 'PDF').substring(0, 3)}
            </div>
            <div className="file-card__info">
              <div className="file-card__name">{f.original_name || f.filename}</div>
              <div className="file-card__meta">
                {f.pages ? `${f.pages}p · ` : ''}
                {(f.size / 1024).toFixed(0)} KB
              </div>
            </div>
            {f.document_type && (
              <span className="file-card__type">{f.document_type}</span>
            )}
          </div>
        ))}

        {/* Action Buttons */}
        {files.length > 0 && !processing && !session && (
          <button
            className="sidebar__action-btn sidebar__action-btn--primary"
            onClick={onStartAnalysis}
            disabled={!llmOnline}
          >
            <span className="material-icons" style={{ fontSize: 16, marginRight: 6 }}>play_arrow</span>
            {llmOnline ? `Analyze ${files.length} Document(s)` : 'LLM Offline'}
          </button>
        )}

        {session && (
          <button className="sidebar__action-btn" onClick={onClear}>
            <span className="material-icons" style={{ fontSize: 14, marginRight: 6 }}>refresh</span>
            New Analysis
          </button>
        )}

        {session && sessionId && (
          <a
            href={getReportPdfUrl(sessionId)}
            target="_blank"
            className="sidebar__action-btn sidebar__action-btn--primary"
            style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          >
            <span className="material-icons" style={{ fontSize: 16, marginRight: 6 }}>download</span>
            Download PDF Report
          </a>
        )}

        {/* Session History */}
        {!processing && (
          <div className="sidebar__history">
            <div className="sidebar__history-header">
              <span className="sidebar__history-title">History</span>
              <button className="sidebar__history-load" onClick={onLoadHistory}>
                <span className="material-icons" style={{ fontSize: 12 }}>refresh</span>
              </button>
            </div>
            {sessionHistory.map(s => (
              <div
                key={s.session_id}
                className={`history-card ${s.session_id === sessionId ? 'history-card--active' : ''} ${s.status !== 'completed' ? 'history-card--disabled' : ''}`}
                onClick={() => s.status === 'completed' && onLoadSession(s.session_id)}
              >
                <div className="history-card__top">
                  <span className="history-card__id">{s.session_id.slice(0, 8)}...</span>
                  <span className={`history-card__band history-card__band--${(s.risk_band || 'none').toLowerCase()}`}>
                    {s.risk_band || s.status}
                  </span>
                </div>
                <div className="history-card__docs">
                  {s.documents?.join(', ').slice(0, 40) || 'No docs'}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
