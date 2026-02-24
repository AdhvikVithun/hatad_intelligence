import { useState, useCallback, useRef } from 'react';
import { streamChat } from '../api';
import type { ChatMessage } from '../types';

export interface UseChatReturn {
  messages: ChatMessage[];
  streaming: boolean;
  thinking: string;
  error: string | null;
  sendMessage: (text: string) => void;
  clearChat: () => void;
}

export function useChat(sessionId: string | null): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [thinking, setThinking] = useState('');
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback((text: string) => {
    if (!sessionId || !text.trim() || streaming) return;

    setError(null);

    // Add user message
    const userMsg: ChatMessage = {
      role: 'user',
      content: text.trim(),
      timestamp: new Date().toISOString(),
    };

    // Add placeholder assistant message
    const assistantMsg: ChatMessage = {
      role: 'assistant',
      content: '',
      thinking: '',
      timestamp: new Date().toISOString(),
      streaming: true,
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setStreaming(true);
    setThinking('');

    // Build history for the API (exclude the new messages)
    const history = messages.map(m => ({ role: m.role, content: m.content }));

    streamChat(
      sessionId,
      text.trim(),
      history,
      // onToken
      (token) => {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = { ...last, content: last.content + token };
          }
          return updated;
        });
      },
      // onThinking
      (chunk) => {
        setThinking(prev => prev + chunk);
      },
      // onDone
      (fullContent) => {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              content: fullContent,
              streaming: false,
              timestamp: new Date().toISOString(),
            };
          }
          return updated;
        });
        setStreaming(false);
        abortRef.current = null;
      },
      // onError
      (errMsg) => {
        setError(errMsg);
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === 'assistant' && last.streaming) {
            // Remove the empty streaming placeholder
            if (!last.content) {
              updated.pop();
            } else {
              updated[updated.length - 1] = { ...last, streaming: false };
            }
          }
          return updated;
        });
        setStreaming(false);
        abortRef.current = null;
      },
    ).then(ctrl => {
      abortRef.current = ctrl;
    });
  }, [sessionId, streaming, messages]);

  const clearChat = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setMessages([]);
    setStreaming(false);
    setThinking('');
    setError(null);
  }, []);

  return { messages, streaming, thinking, error, sendMessage, clearChat };
}
