/* API client for HATAD backend */

import type {
  UploadResponse,
  StartAnalysisResponse,
  SessionData,
  SessionsResponse,
  LLMHealthResponse,
  ChatMessage,
  ChatStreamEvent,
} from './types';

const BASE = '/api';

// ── Retry helper ────────────────────────────────────

async function fetchWithRetry(
  input: RequestInfo,
  init?: RequestInit,
  retries = 2,
): Promise<Response> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(input, init);
      // Retry on 5xx server errors (not on 4xx client errors)
      if (res.status >= 500 && attempt < retries) {
        await new Promise(r => setTimeout(r, 1000 * 2 ** attempt));
        continue;
      }
      return res;
    } catch (err) {
      lastError = err;
      if (attempt < retries) {
        await new Promise(r => setTimeout(r, 1000 * 2 ** attempt));
      }
    }
  }
  throw lastError;
}

// ── API functions ───────────────────────────────────

export async function uploadDocuments(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  const res = await fetchWithRetry(`${BASE}/documents/upload`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function startAnalysis(filenames: string[]): Promise<StartAnalysisResponse> {
  const res = await fetchWithRetry(`${BASE}/analyze/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filenames }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function streamAnalysis(
  sessionId: string,
  onMessage: (data: any) => void,
  onDone: () => void,
  onError: (err: any) => void,
): () => void {
  const es = new EventSource(`${BASE}/analyze/${sessionId}/stream`);
  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.error && data.stage !== 'final') {
        // Server-side timeout or error — not a normal completion
        onMessage({ stage: 'error', message: data.error, detail: { type: 'stream_error' } });
        es.close();
        onError(new Error(data.error));
        return;
      }
      onMessage(data);
      if (data.stage === 'final') {
        es.close();
        if (data.status === 'failed') {
          onError(new Error(data.error || 'Analysis failed'));
        } else {
          onDone();
        }
      }
    } catch { /* ignore parse errors */ }
  };
  es.onerror = (err) => {
    // EventSource auto-reconnects on error; only close on fatal
    if (es.readyState === EventSource.CLOSED) {
      onError(err);
    }
  };
  return () => es.close();
}

export async function getResults(sessionId: string): Promise<SessionData> {
  // Delay slightly before first attempt — gives Windows Defender / antivirus
  // time to release file locks on the freshly-written session JSON.
  await new Promise(r => setTimeout(r, 500));

  // More aggressive retries than the generic fetchWithRetry:
  // 5 total attempts with 1 s exponential backoff (≈ 15 s total).
  const maxRetries = 4;
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetch(`${BASE}/analyze/${sessionId}/results`);
      if (res.status >= 500 && attempt < maxRetries) {
        console.warn(`[HATAD] getResults attempt ${attempt + 1}/${maxRetries + 1} got ${res.status} — retrying`);
        await new Promise(r => setTimeout(r, 1000 * 2 ** attempt));
        continue;
      }
      if (res.status === 503 && attempt < maxRetries) {
        // Backend's retry-on-read says file is still locked — wait longer
        console.warn(`[HATAD] getResults attempt ${attempt + 1}/${maxRetries + 1} got 503 — retrying`);
        await new Promise(r => setTimeout(r, 1500 * 2 ** attempt));
        continue;
      }
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    } catch (err) {
      lastError = err;
      if (attempt < maxRetries) {
        console.warn(`[HATAD] getResults attempt ${attempt + 1}/${maxRetries + 1} failed:`, err);
        await new Promise(r => setTimeout(r, 1000 * 2 ** attempt));
      }
    }
  }
  throw lastError;
}

export async function getSessions(): Promise<SessionsResponse> {
  const res = await fetchWithRetry(`${BASE}/analyze/sessions`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function checkLLMHealth(): Promise<LLMHealthResponse> {
  try {
    const res = await fetchWithRetry(`${BASE}/analyze/health/llm`);
    return res.json();
  } catch {
    return { status: 'offline' };
  }
}

export function getReportPdfUrl(sessionId: string): string {
  return `${BASE}/analyze/${sessionId}/report/pdf`;
}

export function getReportHtmlUrl(sessionId: string): string {
  return `${BASE}/analyze/${sessionId}/report/html`;
}

// ── Chat streaming (POST-based SSE via fetch + ReadableStream) ──

export async function streamChat(
  sessionId: string,
  message: string,
  history: { role: string; content: string }[],
  onToken: (token: string) => void,
  onThinking: (chunk: string) => void,
  onDone: (fullContent: string) => void,
  onError: (error: string) => void,
): Promise<AbortController> {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/analyze/${sessionId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text();
        onError(text || `HTTP ${res.status}`);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        onError('No response body');
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE lines
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          const jsonStr = trimmed.slice(6);
          try {
            const evt: ChatStreamEvent = JSON.parse(jsonStr);
            if (evt.error) {
              onError(evt.error);
              return;
            }
            if (evt.thinking) {
              onThinking(evt.thinking);
            }
            if (evt.token) {
              onToken(evt.token);
            }
            if (evt.done && evt.content !== undefined) {
              onDone(evt.content);
              return;
            }
          } catch { /* skip malformed lines */ }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        onError(err.message || 'Chat stream failed');
      }
    }
  })();

  return controller;
}
