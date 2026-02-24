import { useRef, useEffect, useState, useCallback, type ReactNode } from 'react';
import type { ChatMessage } from '../../types';
import './ChatPanel.css';

const SUGGESTIONS = [
  'What are the main risks in this property?',
  'Is the chain of title complete?',
  'Summarise all red flags found.',
  'Who are the parties involved?',
  'Are there any survey number mismatches?',
];

interface Props {
  messages: ChatMessage[];
  streaming: boolean;
  thinking: string;
  error: string | null;
  onSend: (text: string) => void;
  onClear: () => void;
  onClose: () => void;
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

/** Safe Markdown-ish rendering: bold, inline code, line breaks — no dangerouslySetInnerHTML */
function renderContent(text: string): ReactNode[] | null {
  if (!text) return null;
  const paragraphs = text.split(/\n{2,}/);
  return paragraphs.map((para, pi) => {
    // Tokenise: split on **bold** and `code` markers, keeping delimiters
    const tokens = para.split(/(\*\*.+?\*\*|`.+?`|\n)/g);
    const children: ReactNode[] = tokens.map((tok, ti) => {
      if (tok === '\n') return <br key={ti} />;
      const boldMatch = tok.match(/^\*\*(.+)\*\*$/);
      if (boldMatch) return <strong key={ti}>{boldMatch[1]}</strong>;
      const codeMatch = tok.match(/^`(.+)`$/);
      if (codeMatch) return <code key={ti}>{codeMatch[1]}</code>;
      return tok;                // plain text — auto-escaped by React
    });
    return <p key={pi}>{children}</p>;
  });
}

export function ChatPanel({ messages, streaming, thinking, error, onSend, onClear, onClose }: Props) {
  const [input, setInput] = useState('');
  const messagesRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new content
  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages, streaming, thinking]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSend = useCallback(() => {
    if (!input.trim() || streaming) return;
    onSend(input.trim());
    setInput('');
  }, [input, streaming, onSend]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleSuggestion = useCallback((text: string) => {
    if (streaming) return;
    onSend(text);
  }, [streaming, onSend]);

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-panel">
      {/* Header */}
      <div className="chat-panel__header">
        <div className="chat-panel__header-left">
          <span className="material-icons chat-panel__header-icon">forum</span>
          <span className="chat-panel__header-title">Ask HATAD</span>
        </div>
        <div className="chat-panel__header-actions">
          {messages.length > 0 && (
            <button className="chat-panel__icon-btn" onClick={onClear} title="Clear chat">
              <span className="material-icons" style={{ fontSize: 16 }}>delete_outline</span>
            </button>
          )}
          <button className="chat-panel__icon-btn" onClick={onClose} title="Close chat">
            <span className="material-icons" style={{ fontSize: 16 }}>close</span>
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="chat-panel__messages" ref={messagesRef}>
        {isEmpty && !streaming && (
          <div className="chat-panel__empty">
            <span className="material-icons chat-panel__empty-icon">psychology</span>
            <div className="chat-panel__empty-text">
              Ask questions about your<br />analysis results
            </div>
            <div className="chat-panel__suggestions">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  className="chat-panel__suggestion"
                  onClick={() => handleSuggestion(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg chat-msg--${msg.role}`}>
            {/* Thinking indicator for streaming assistant */}
            {msg.role === 'assistant' && msg.streaming && thinking && !msg.content && (
              <div className="chat-msg__thinking">
                <span className="material-icons chat-msg__thinking-icon">autorenew</span>
                Thinking...
              </div>
            )}
            <div className="chat-msg__bubble">
              {msg.role === 'assistant' ? renderContent(msg.content) : msg.content}
              {msg.streaming && <span className="chat-msg__cursor" />}
            </div>
            {!msg.streaming && (
              <span className="chat-msg__time">{formatTime(msg.timestamp)}</span>
            )}
          </div>
        ))}
      </div>

      {/* Error */}
      {error && (
        <div className="chat-panel__error">
          <span className="material-icons" style={{ fontSize: 14 }}>error</span>
          {error}
        </div>
      )}

      {/* Input */}
      <div className="chat-panel__input-area">
        <textarea
          ref={inputRef}
          className="chat-panel__input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about this analysis..."
          rows={1}
          disabled={streaming}
        />
        <button
          className="chat-panel__send-btn"
          onClick={handleSend}
          disabled={!input.trim() || streaming}
          title="Send"
        >
          <span className="material-icons">send</span>
        </button>
      </div>
    </div>
  );
}
