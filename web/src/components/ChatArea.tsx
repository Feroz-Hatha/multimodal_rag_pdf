import { useState, useRef, useEffect } from 'react';
import { Send, ChevronDown, ChevronUp } from 'lucide-react';
import type { Message, Source } from '../types';
import { queryStream } from '../api';

interface ChatAreaProps {
  messages: Message[];
  documentIds: string[];
  hasDocuments: boolean;
  onAddMessage: (msg: Message) => void;
}

export default function ChatArea({
  messages,
  documentIds,
  hasDocuments,
  onAddMessage,
}: ChatAreaProps) {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  const streamRef = useRef('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading, streamingContent]);

  const submit = async () => {
    const question = input.trim();
    if (!question || loading || !hasDocuments) return;

    setInput('');
    onAddMessage({ role: 'user', content: question });

    streamRef.current = '';
    setStreamingContent('');
    setLoading(true);

    try {
      let finalSources: Source[] = [];

      for await (const event of queryStream(question, documentIds)) {
        if (event.type === 'delta') {
          streamRef.current += event.text;
          setStreamingContent(streamRef.current);
        } else if (event.type === 'done') {
          finalSources = event.sources;
        } else if (event.type === 'error') {
          throw new Error(event.message);
        }
      }

      onAddMessage({ role: 'assistant', content: streamRef.current, sources: finalSources });
    } catch (e) {
      onAddMessage({ role: 'assistant', content: `Error: ${String(e)}` });
    } finally {
      setStreamingContent(null);
      streamRef.current = '';
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const autoResize = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  return (
    <main className="flex-1 flex flex-col h-screen overflow-hidden bg-white">

      {/* ── Messages ── */}
      <div className="flex-1 overflow-y-auto px-8 py-6">
        {messages.length === 0 ? (

          /* Welcome screen */
          <div className="h-full flex flex-col items-center justify-center text-center">
            <h1 className="text-4xl font-bold text-sage-900 tracking-tight mb-3 flex items-center gap-3">
              <img src="/logo.png" alt="Logo" className="w-10 h-10 object-contain" />
              PDF RAG Assistant
            </h1>
            <p className="text-sm text-sage-400 max-w-sm leading-relaxed">
              {hasDocuments
                ? 'Your documents are ready. Ask a question below.'
                : 'Upload a PDF in the sidebar, then ask questions about its content.'}
            </p>
          </div>

        ) : (
          <div className="space-y-5">
            {messages.map((msg, i) => (
              <ChatMessage key={i} message={msg} />
            ))}

            {/* Streaming assistant response */}
            {streamingContent !== null && (
              <ChatMessage
                message={{ role: 'assistant', content: streamingContent }}
                isStreaming
              />
            )}

            {/* Dots — shown only before first token arrives */}
            {loading && streamingContent === '' && (
              <div className="flex items-center gap-1 pl-0.5 py-1">
                {[0, 150, 300].map(delay => (
                  <span
                    key={delay}
                    className="w-1.5 h-1.5 rounded-full bg-sage-400 animate-bounce"
                    style={{ animationDelay: `${delay}ms` }}
                  />
                ))}
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* ── Input bar ── */}
      <div className="py-4 border-t border-sage-100 bg-white">
        <div className="w-[68%] mx-auto flex items-center gap-2 pl-4 pr-1.5 border border-sage-200 rounded-xl focus-within:border-sage-400 focus-within:ring-2 focus-within:ring-sage-400/20 transition-colors">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={autoResize}
            onKeyDown={handleKeyDown}
            placeholder={
              hasDocuments
                ? 'Ask a question about your documents…'
                : 'Upload a document first…'
            }
            disabled={!hasDocuments || loading}
            rows={1}
            className="flex-1 resize-none text-[13px] text-sage-800 placeholder-sage-300 py-2.5 focus:outline-none disabled:opacity-40 disabled:cursor-not-allowed overflow-y-auto leading-relaxed bg-transparent"
          />
          <button
            onClick={submit}
            disabled={!input.trim() || !hasDocuments || loading}
            className="flex-shrink-0 w-7 h-7 rounded-lg bg-sage-600 text-white flex items-center justify-center hover:bg-sage-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <Send size={13} />
          </button>
        </div>
        <p className="mt-1.5 text-center text-[10px] text-sage-300">
          Enter to send · Shift + Enter for a new line
        </p>
      </div>
    </main>
  );
}

/* ── Individual chat message ── */
function ChatMessage({
  message,
  isStreaming = false,
}: {
  message: Message;
  isStreaming?: boolean;
}) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';
  const hasSources = (message.sources?.length ?? 0) > 0;

  /* User message — right-aligned green bubble, no avatar */
  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="bg-sage-50 rounded-xl px-3.5 py-2.5 max-w-[70%] text-[13px] text-sage-800 leading-relaxed">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  /* Assistant message — left-aligned, no box, no avatar */
  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-[13px] text-sage-800 leading-relaxed">
        <p className="whitespace-pre-wrap">
          {message.content}
          {isStreaming && (
            <span className="inline-block w-0.5 h-3.5 bg-sage-600 ml-0.5 align-middle animate-pulse" />
          )}
        </p>
      </div>

      {hasSources && (
        <>
          <button
            onClick={() => setShowSources(v => !v)}
            className="self-start flex items-center gap-1 text-[11px] text-sage-400 hover:text-sage-600 transition-colors"
          >
            {showSources ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
            {message.sources!.length} source{message.sources!.length !== 1 ? 's' : ''}
          </button>

          {showSources && (
            <div className="space-y-1">
              {message.sources!.map((src, i) => (
                <SourceCard key={i} source={src} index={i + 1} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* ── Source card ── */
function SourceCard({ source, index }: { source: Source; index: number }) {
  const label =
    source.heading ||
    (source.section_hierarchy?.length
      ? source.section_hierarchy.join(' › ')
      : source.filename);
  const pages = source.page_numbers?.length
    ? `p. ${source.page_numbers.join(', ')}`
    : null;

  return (
    <div className="text-[11px] bg-sage-50 border border-sage-200 rounded-lg px-3 py-2 flex items-start justify-between gap-2">
      <span className="text-sage-700 leading-snug">
        <span className="font-semibold text-sage-500">[{index}]</span>{' '}
        <span className="font-medium">{source.filename}</span>
        {label !== source.filename && (
          <span className="text-sage-400"> — {label}</span>
        )}
        {pages && <span className="text-sage-400"> · {pages}</span>}
      </span>
      <span className="text-sage-300 flex-shrink-0 tabular-nums">
        {source.score.toFixed(3)}
      </span>
    </div>
  );
}
