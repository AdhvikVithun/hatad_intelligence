/* API client for HATAD backend */

import type {
  UploadResponse,
  StartAnalysisResponse,
  SessionData,
  SessionsResponse,
  LLMHealthResponse,
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
  const res = await fetchWithRetry(`${BASE}/analyze/${sessionId}/results`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
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
