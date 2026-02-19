/* API client for HATAD backend */

const BASE = '/api';

export async function uploadDocuments(files: File[]): Promise<any> {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  const res = await fetch(`${BASE}/documents/upload`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function startAnalysis(filenames: string[]): Promise<any> {
  const res = await fetch(`${BASE}/analyze/start`, {
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

export async function getResults(sessionId: string): Promise<any> {
  const res = await fetch(`${BASE}/analyze/${sessionId}/results`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getSessions(): Promise<any> {
  const res = await fetch(`${BASE}/analyze/sessions`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function checkLLMHealth(): Promise<any> {
  try {
    const res = await fetch(`${BASE}/analyze/health/llm`);
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
