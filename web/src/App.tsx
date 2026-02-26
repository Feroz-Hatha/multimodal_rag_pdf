import { useState, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import type { Message, SessionDoc } from './types';

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionDocs, setSessionDocs] = useState<SessionDoc[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  const addDoc = useCallback((doc: SessionDoc) => {
    setSessionDocs(prev =>
      prev.some(d => d.document_id === doc.document_id) ? prev : [...prev, doc],
    );
  }, []);

  const removeDoc = useCallback((documentId: string) => {
    setSessionDocs(prev => prev.filter(d => d.document_id !== documentId));
    setSelectedDocId(prev => (prev === documentId ? null : prev));
  }, []);

  const addMessage = useCallback((msg: Message) => {
    setMessages(prev => [...prev, msg]);
  }, []);

  const clearChat = useCallback(() => setMessages([]), []);

  const documentIds = selectedDocId
    ? [selectedDocId]
    : sessionDocs.map(d => d.document_id);

  return (
    <div className="flex h-screen overflow-hidden font-inter bg-stone-50">
      <Sidebar
        sessionDocs={sessionDocs}
        selectedDocId={selectedDocId}
        onDocumentAdded={addDoc}
        onDocumentRemoved={removeDoc}
        onSelectDoc={setSelectedDocId}
        onClearChat={clearChat}
      />
      <ChatArea
        messages={messages}
        documentIds={documentIds}
        hasDocuments={sessionDocs.length > 0}
        onAddMessage={addMessage}
      />
    </div>
  );
}
