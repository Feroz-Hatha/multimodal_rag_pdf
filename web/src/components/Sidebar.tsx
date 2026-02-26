import { useState, useRef } from 'react';
import { Paperclip, Trash2, FileText } from 'lucide-react';
import type { SessionDoc } from '../types';
import { ingestDocument, getJobStatus, deleteDocument } from '../api';

interface Props {
  sessionDocs: SessionDoc[];
  selectedDocId: string | null;
  onDocumentAdded: (doc: SessionDoc) => void;
  onDocumentRemoved: (id: string) => void;
  onSelectDoc: (id: string | null) => void;
  onClearChat: () => void;
}

type UploadPhase = 'idle' | 'indexing' | 'success' | 'error';

interface UploadState {
  phase: UploadPhase;
  progress: number;
  stage: string;
  message?: string;
}

const IDLE: UploadState = { phase: 'idle', progress: 0, stage: '' };

export default function Sidebar({
  sessionDocs,
  selectedDocId,
  onDocumentAdded,
  onDocumentRemoved,
  onSelectDoc,
  onClearChat,
}: Props) {
  const [upload, setUpload] = useState<UploadState>(IDLE);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUpload({ phase: 'error', progress: 0, stage: '', message: 'Only PDF files are supported.' });
      return;
    }

    setUpload({ phase: 'indexing', progress: 0.05, stage: 'Uploading file…' });

    try {
      const { job_id } = await ingestDocument(file);

      while (true) {
        await new Promise(r => setTimeout(r, 1000));
        const status = await getJobStatus(job_id);
        setUpload({ phase: 'indexing', progress: status.progress, stage: status.stage });

        if (status.status === 'done') {
          onDocumentAdded({
            document_id: status.document_id!,
            filename: status.filename,
            total_chunks: status.total_chunks ?? 0,
            text_chunks: status.text_chunks ?? 0,
            table_chunks: status.table_chunks ?? 0,
            image_chunks: status.image_chunks ?? 0,
          });
          const msg = status.already_indexed
            ? 'Already indexed — added to your session.'
            : 'Indexed successfully.';
          setUpload({ phase: 'success', progress: 1, stage: '', message: msg });
          setTimeout(() => {
            setUpload(IDLE);
            if (fileInputRef.current) fileInputRef.current.value = '';
          }, 3000);
          break;
        }

        if (status.status === 'error') {
          setUpload({ phase: 'error', progress: 0, stage: '', message: status.error ?? 'Indexing failed.' });
          break;
        }
      }
    } catch (e) {
      setUpload({ phase: 'error', progress: 0, stage: '', message: String(e) });
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleDelete = async (doc: SessionDoc) => {
    try {
      await deleteDocument(doc.document_id);
      onDocumentRemoved(doc.document_id);
    } catch (e) {
      console.error('Delete failed:', e);
    }
  };

  return (
    <aside className="w-[19rem] flex-shrink-0 flex flex-col bg-sage-50 border-r border-sage-200 h-screen">
      {/* Header */}
      <div className="px-4 py-3.5 border-b border-sage-200">
        <p className="text-sm font-semibold text-sage-800 tracking-tight flex items-center gap-1.5">
          <FileText size={14} className="text-sage-600 flex-shrink-0" />
          PDF RAG Assistant
        </p>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-5">

        {/* ── Upload section ── */}
        <section>
          <p className="text-[10px] font-semibold text-sage-500 uppercase tracking-widest mb-2">
            Upload Document
          </p>

          {upload.phase === 'idle' && (
            <div
              className={`border-2 border-dashed rounded-lg p-3 text-center cursor-pointer transition-colors ${
                dragOver
                  ? 'border-sage-500 bg-sage-100'
                  : 'border-sage-300 hover:border-sage-400 hover:bg-sage-100/60'
              }`}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <p className="text-[11px] text-sage-400 mb-2 leading-snug">
                Drag and drop PDF here<br />(200 MB max.)
              </p>
              <button
                type="button"
                className="inline-flex items-center gap-1 text-[11px] font-medium text-sage-700 border border-sage-300 rounded px-2 py-0.5 bg-white hover:bg-sage-50 hover:border-sage-500 transition-colors"
                onClick={e => { e.stopPropagation(); fileInputRef.current?.click(); }}
              >
                <Paperclip size={11} />
                Browse files
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
              />
            </div>
          )}

          {upload.phase === 'indexing' && (
            <div className="space-y-1.5">
              <p className="text-[11px] text-sage-600 truncate">{upload.stage}</p>
              <div className="h-1.5 bg-sage-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-sage-500 rounded-full transition-all duration-500"
                  style={{ width: `${Math.round(upload.progress * 100)}%` }}
                />
              </div>
              <p className="text-[10px] text-sage-400 text-right">
                {Math.round(upload.progress * 100)}%
              </p>
            </div>
          )}

          {upload.phase === 'success' && (
            <p className="text-[11px] text-green-700 bg-green-50 border border-green-200 rounded-lg px-2.5 py-1.5">
              {upload.message}
            </p>
          )}

          {upload.phase === 'error' && (
            <div className="space-y-1.5">
              <p className="text-[11px] text-red-600 bg-red-50 border border-red-200 rounded-lg px-2.5 py-1.5">
                {upload.message}
              </p>
              <button
                className="text-[11px] text-sage-500 hover:text-sage-700 underline transition-colors"
                onClick={() => setUpload(IDLE)}
              >
                Try again
              </button>
            </div>
          )}
        </section>

        {/* ── My Documents ── */}
        <section>
          <p className="text-[10px] font-semibold text-sage-500 uppercase tracking-widest mb-2">
            My Documents
          </p>
          {sessionDocs.length === 0 ? (
            <p className="text-[11px] text-sage-400 italic">
              No documents yet. Upload a PDF above.
            </p>
          ) : (
            <ul className="space-y-1">
              {sessionDocs.map(doc => (
                <li key={doc.document_id} className="flex items-center gap-1 group py-0.5">
                  <span
                    className="flex-1 text-[11px] text-sage-700 truncate leading-snug"
                    title={doc.filename}
                  >
                    📄 {doc.filename}
                  </span>
                  <button
                    onClick={() => handleDelete(doc)}
                    title="Delete from system"
                    className="flex-shrink-0 opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-600 transition-all rounded p-0.5"
                  >
                    <Trash2 size={11} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* ── Query scope ── */}
        {sessionDocs.length > 0 && (
          <section>
            <p className="text-[10px] font-semibold text-sage-500 uppercase tracking-widest mb-2">
              Query Scope
            </p>
            <select
              className="w-full text-[11px] border border-sage-300 rounded-md px-2 py-1.5 bg-white text-sage-700 focus:outline-none focus:border-sage-500 focus:ring-1 focus:ring-sage-500/30 transition-colors"
              value={selectedDocId ?? ''}
              onChange={e => onSelectDoc(e.target.value || null)}
            >
              <option value="">All my documents</option>
              {sessionDocs.map(doc => (
                <option key={doc.document_id} value={doc.document_id}>
                  {doc.filename}
                </option>
              ))}
            </select>
          </section>
        )}
      </div>

      {/* Clear chat — pinned to bottom */}
      <div className="px-4 py-3 border-t border-sage-200 flex justify-center">
        <button
          onClick={onClearChat}
          className="flex items-center gap-1.5 text-[11px] text-red-400 hover:text-red-600 transition-colors"
        >
          <Trash2 size={11} />
          Clear chat
        </button>
      </div>
    </aside>
  );
}
