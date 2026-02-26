import type { JobStatus, QueryResponse } from './types';

// In development Vite proxies /api → http://localhost:8000
// In production Nginx handles the same proxy, so the path never changes.
const BASE = '/api/v1';

async function checkResponse(res: Response): Promise<Response> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  return res;
}

export async function ingestDocument(
  file: File,
): Promise<{ job_id: string; filename: string }> {
  const body = new FormData();
  body.append('file', file);
  const res = await checkResponse(await fetch(`${BASE}/ingest`, { method: 'POST', body }));
  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await checkResponse(await fetch(`${BASE}/jobs/${jobId}`));
  return res.json();
}

export async function queryDocuments(
  question: string,
  documentIds: string[],
  nResults = 5,
): Promise<QueryResponse> {
  const res = await checkResponse(
    await fetch(`${BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, document_ids: documentIds, n_results: nResults }),
    }),
  );
  return res.json();
}

export async function deleteDocument(documentId: string): Promise<void> {
  await checkResponse(await fetch(`${BASE}/documents/${documentId}`, { method: 'DELETE' }));
}

export type StreamEvent =
  | { type: 'delta'; text: string }
  | { type: 'done'; sources: import('./types').Source[] }
  | { type: 'error'; message: string };

export async function* queryStream(
  question: string,
  documentIds: string[],
  nResults = 5,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, document_ids: documentIds, n_results: nResults }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() ?? '';

    for (const part of parts) {
      const line = part.trim();
      if (line.startsWith('data: ')) {
        yield JSON.parse(line.slice(6)) as StreamEvent;
      }
    }
  }
}
