'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { 
  ArrowLeftIcon,
  SparklesIcon,
  PaperAirplaneIcon,
  MicrophoneIcon,
  DocumentTextIcon,
  LightBulbIcon,
  ChatBubbleBottomCenterTextIcon,
  QuestionMarkCircleIcon,
  DocumentArrowUpIcon,
  PlusIcon,
  XMarkIcon,
  CheckCircleIcon,
  ClockIcon,
  TrashIcon,
  ChatBubbleLeftIcon
} from '@heroicons/react/24/outline';
import DocumentUpload from '@/components/DocumentUpload';
import { DocumentMetadata } from '@/lib/api-client';
import { useAppContext, ChatMessage, ChatSession } from '@/contexts/AppContext';

const SAMPLE_QUESTIONS = [
  "What are the key termination clauses in my contracts?",
  "Identify potential risks in liability sections",
  "Compare payment terms across documents",
  "Find missing confidentiality clauses",
  "Analyze force majeure provisions",
  "Review intellectual property clauses"
];

export default function AskAIPage() {
  const router = useRouter();
  const { 
    documents, 
    addDocument, 
    chatSessions, 
    currentSessionId, 
    createNewSession, 
    loadSession, 
    updateCurrentSession, 
    deleteSession, 
    getCurrentSession,
    initialChatContext,
    clearInitialChatContext
  } = useAppContext();

  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showContextPanel, setShowContextPanel] = useState(false);
  const [selectedContextDocs, setSelectedContextDocs] = useState<string[]>([]);
  const [localMessages, setLocalMessages] = useState<ChatMessage[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Initialize session if needed
  useEffect(() => {
    if (!currentSessionId) {
      createNewSession();
    }
  }, [currentSessionId, createNewSession]);

  // Sync local messages with current session
  useEffect(() => {
    const currentSession = getCurrentSession();
    if (currentSession) {
      setLocalMessages(currentSession.messages);
    }
  }, [currentSessionId, chatSessions, getCurrentSession]);

  // Handle initial chat context from navigation
  useEffect(() => {
    if (initialChatContext) {
      setSelectedContextDocs([initialChatContext]);
      clearInitialChatContext();
    }
  }, [initialChatContext, clearInitialChatContext]);

  const messages = localMessages;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    const newMessages = [...messages, userMessage];
    
    // Update local state immediately for responsive UI
    setLocalMessages(newMessages);
    // Also update context for persistence
    updateCurrentSession(newMessages);
    
    setInputValue('');
    setIsLoading(true);

    // Simulate AI thinking
    const thinkingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: 'Analyzing your documents and generating response...',
      timestamp: new Date(),
      thinking: true
    };

    const messagesWithThinking = [...newMessages, thinkingMessage];
    setLocalMessages(messagesWithThinking);
    updateCurrentSession(messagesWithThinking);

    // Simulate API response delay
    setTimeout(() => {
      const selectedDocs = getSelectedDocuments();
      const contextInfo = selectedDocs.length > 0 
        ? `\n\n🎯 **Context Documents**: ${selectedDocs.map(doc => doc.filename).join(', ')}\n`
        : '';
      
      const responseContent = documents.length > 0 
        ? `Based on my analysis of your uploaded documents${contextInfo ? ` (focusing on selected context)` : ''}, here's what I found regarding "${userMessage.content}":${contextInfo}

📋 **Document Analysis:**
- Analyzed ${selectedDocs.length > 0 ? selectedDocs.length : documents.length} document${(selectedDocs.length > 0 ? selectedDocs.length : documents.length) > 1 ? 's' : ''}: ${selectedDocs.length > 0 ? selectedDocs.map(doc => doc.filename).join(', ') : documents.map(doc => doc.filename).join(', ')}
- Found relevant clauses related to your query
- Identified potential compliance considerations

⚠️ **Key Findings:**
- Standard contract provisions detected
- Liability and risk clauses present
- Payment and termination terms analyzed

💡 **Recommendations:**
1. Review highlighted clauses for consistency
2. Consider legal counsel for complex provisions
3. Ensure compliance with applicable regulations

Would you like me to dive deeper into any specific document or clause type?`
        : `I'd be happy to help with your legal question: "${userMessage.content}"

📋 **General Legal Guidance:**
To provide the most accurate analysis, I recommend uploading your legal documents using the "Upload Docs" button above.

💡 **What I can analyze:**
- Contract clauses and terms
- Risk assessment and liability
- Compliance requirements
- Legal language interpretation

🔍 **Next Steps:**
1. Upload your legal documents
2. Select documents for context (if multiple uploaded)
3. Ask specific questions about clauses or terms
4. Get tailored analysis and recommendations

Would you like to upload documents now, or do you have a general legal question I can help with?`;

      const aiResponse: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: responseContent,
        timestamp: new Date()
      };

      const finalMessages = newMessages.concat([aiResponse]);
      setLocalMessages(finalMessages);
      updateCurrentSession(finalMessages);
      setIsLoading(false);
    }, 2000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleSampleQuestion = (question: string) => {
    setInputValue(question);
    inputRef.current?.focus();
  };

  const toggleVoiceInput = () => {
    setIsListening(!isListening);
  };

  const handleDocumentUpload = (docId: string, metadata: DocumentMetadata) => {
    addDocument(metadata);
    
    const uploadMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'system',
      content: `📄 Document "${metadata.filename}" uploaded successfully! I can now analyze this document for you. Ask me questions about its content, clauses, risks, or any legal concerns.`,
      timestamp: new Date()
    };
    
    const newMessages = [...messages, uploadMessage];
    setLocalMessages(newMessages);
    updateCurrentSession(newMessages);
    setShowUpload(false);
  };

  // Context selection functions
  const toggleContextDoc = (docId: string) => {
    setSelectedContextDocs(prev => 
      prev.includes(docId) 
        ? prev.filter(id => id !== docId)
        : [...prev, docId]
    );
  };

  const clearContextSelection = () => {
    setSelectedContextDocs([]);
  };

  const selectAllContextDocs = () => {
    setSelectedContextDocs(documents.map(doc => doc.id));
  };

  const getSelectedDocuments = () => {
    return documents.filter(doc => selectedContextDocs.includes(doc.id));
  };

  const handleUploadError = (error: string) => {
    const errorMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'system',
      content: `❌ Upload failed: ${error}. Please try again or contact support if the issue persists.`,
      timestamp: new Date()
    };
    
    const newMessages = [...messages, errorMessage];
    setLocalMessages(newMessages);
    updateCurrentSession(newMessages);
  };

  const handleNewChat = () => {
    createNewSession();
    setShowHistory(false);
  };

  const handleLoadSession = (sessionId: string) => {
    loadSession(sessionId);
    setShowHistory(false);
  };

  return (
    <div className="min-h-screen flex" style={{background: 'radial-gradient(circle at 30% 20%, rgba(88,101,242,0.15), transparent 60%), radial-gradient(circle at 70% 80%, rgba(213,72,165,0.15), transparent 55%)'}}>
      
      {/* Chat History Sidebar */}
      {showHistory && (
        <div className="w-80 gemini-card m-6 mr-0 p-4 overflow-y-auto">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-bold text-[var(--starlight-white)]">Chat History</h2>
            <button
              onClick={() => setShowHistory(false)}
              className="p-1 rounded-lg hover:bg-[rgba(240,242,252,0.1)] transition-colors"
            >
              <XMarkIcon className="h-4 w-4 text-[var(--starlight-white)]" />
            </button>
          </div>
          
          <button
            onClick={handleNewChat}
            className="w-full mb-4 flex items-center gap-2 p-3 rounded-lg bg-gradient-to-r from-[var(--electric-aqua)] to-[var(--vibrant-magenta)] text-black font-medium hover:scale-105 transition-transform"
          >
            <PlusIcon className="h-4 w-4" />
            New Chat
          </button>
          
          <div className="space-y-2">
            {chatSessions.map((session) => (
              <div
                key={session.id}
                className={`p-3 rounded-lg border transition-colors cursor-pointer ${
                  session.id === currentSessionId
                    ? 'bg-[rgba(240,242,252,0.1)] border-[var(--electric-aqua)]'
                    : 'bg-[rgba(240,242,252,0.05)] border-[rgba(138,145,180,0.2)] hover:bg-[rgba(240,242,252,0.08)]'
                }`}
                onClick={() => handleLoadSession(session.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-medium text-[var(--starlight-white)] truncate">
                      {session.title}
                    </h3>
                    <div className="flex items-center gap-1 mt-1">
                      <ClockIcon className="h-3 w-3 text-[var(--nebula-grey)]" />
                      <span className="text-xs text-[var(--nebula-grey)]">
                        {session.updatedAt.toLocaleDateString()}
                      </span>
                    </div>
                    <p className="text-xs text-[var(--nebula-grey)] mt-1">
                      {session.messages.length} messages
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteSession(session.id);
                    }}
                    className="p-1 rounded hover:bg-[rgba(240,242,252,0.1)] transition-colors"
                  >
                    <TrashIcon className="h-3 w-3 text-[var(--nebula-grey)] hover:text-red-400" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="gemini-card m-6 mb-0 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => router.back()}
                className="p-2 rounded-lg hover:bg-[rgba(240,242,252,0.1)] transition-colors"
              >
                <ArrowLeftIcon className="h-5 w-5 text-[var(--starlight-white)]" />
              </button>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-r from-[var(--electric-aqua)] to-[var(--vibrant-magenta)]">
                  <SparklesIcon className="h-6 w-6 text-black" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-[var(--starlight-white)]">AI Legal Assistant</h1>
                  <p className="text-xs text-[var(--nebula-grey)]">Powered by advanced legal analysis AI</p>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {/* Context Selection Button */}
              <button
                onClick={() => setShowContextPanel(!showContextPanel)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg border transition-colors ${
                  selectedContextDocs.length > 0 
                    ? 'bg-[rgba(36,216,218,0.15)] border-[rgba(36,216,218,0.5)] text-[var(--electric-aqua)]'
                    : 'bg-[rgba(240,242,252,0.1)] border-[rgba(138,145,180,0.3)] hover:bg-[rgba(240,242,252,0.15)] text-[var(--starlight-white)]'
                }`}
              >
                <DocumentTextIcon className="h-4 w-4" />
                <span className="text-sm">
                  Context {selectedContextDocs.length > 0 && `(${selectedContextDocs.length})`}
                </span>
              </button>

              {/* Chat History Button */}
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[rgba(240,242,252,0.1)] border border-[rgba(138,145,180,0.3)] hover:bg-[rgba(240,242,252,0.15)] transition-colors"
              >
                <ChatBubbleLeftIcon className="h-4 w-4 text-[var(--starlight-white)]" />
                <span className="text-sm text-[var(--starlight-white)]">History</span>
              </button>
              
              {/* Upload Documents Button */}
              <button
                onClick={() => setShowUpload(true)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-[var(--electric-aqua)] to-[var(--vibrant-magenta)] text-black font-medium hover:scale-105 transition-transform"
              >
                <DocumentArrowUpIcon className="h-4 w-4" />
                <span className="text-sm">Upload Docs</span>
              </button>
              
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1 px-3 py-1 rounded-full bg-[rgba(240,242,252,0.1)] border border-[rgba(138,145,180,0.3)]">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-xs text-[var(--starlight-white)]">Online</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col mx-6 mt-6">
          <div className="gemini-card flex-1 flex flex-col max-h-[calc(100vh-200px)]">
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-[80%] ${
                    message.type === 'user' 
                      ? 'bg-gradient-to-r from-[var(--electric-aqua)] to-[var(--vibrant-magenta)] text-black' 
                      : message.type === 'system'
                      ? 'bg-[rgba(240,242,252,0.1)] border border-[rgba(138,145,180,0.3)] text-[var(--starlight-white)]'
                      : 'bg-[rgba(240,242,252,0.05)] border border-[rgba(138,145,180,0.2)] text-[var(--starlight-white)]'
                  } rounded-2xl p-4`}>
                    {message.type !== 'user' && (
                      <div className="flex items-center gap-2 mb-2">
                        {message.type === 'system' ? (
                          <LightBulbIcon className="h-4 w-4 text-[var(--electric-aqua)]" />
                        ) : (
                          <SparklesIcon className="h-4 w-4 text-[var(--electric-aqua)]" />
                        )}
                        <span className="text-xs font-medium text-[var(--electric-aqua)]">
                          {message.type === 'system' ? 'System' : 'AI Assistant'}
                        </span>
                      </div>
                    )}
                    <div className={`text-sm ${message.thinking ? 'flex items-center gap-2' : ''}`}>
                      {message.thinking && (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-[var(--electric-aqua)]"></div>
                      )}
                      <span className="whitespace-pre-wrap">{message.content}</span>
                    </div>
                    <div className={`text-xs mt-2 ${
                      message.type === 'user' ? 'text-black/70' : 'text-[var(--nebula-grey)]'
                    }`}>
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            {/* Sample Questions */}
            {messages.length <= 1 && (
              <div className="p-6 pt-0">
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-[var(--starlight-white)] mb-3 flex items-center gap-2">
                    <QuestionMarkCircleIcon className="h-4 w-4" />
                    Try asking me about:
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {SAMPLE_QUESTIONS.map((question, index) => (
                      <button
                        key={index}
                        onClick={() => handleSampleQuestion(question)}
                        className="text-left p-3 rounded-lg bg-[rgba(240,242,252,0.05)] border border-[rgba(138,145,180,0.2)] hover:bg-[rgba(240,242,252,0.1)] transition-colors"
                      >
                        <span className="text-xs text-[var(--starlight-white)]">{question}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Input Area */}
            <div className="p-6 pt-0">
              {/* Context Indicator */}
              {selectedContextDocs.length > 0 && (
                <div className="mb-3 p-3 rounded-lg bg-[rgba(36,216,218,0.1)] border border-[rgba(36,216,218,0.3)]">
                  <div className="flex items-center gap-2 mb-2">
                    <DocumentTextIcon className="h-4 w-4 text-[var(--electric-aqua)]" />
                    <span className="text-sm font-medium text-[var(--electric-aqua)]">
                      Context Active ({selectedContextDocs.length} documents)
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {getSelectedDocuments().map((doc) => (
                      <span
                        key={doc.id}
                        className="text-xs px-2 py-1 rounded bg-[rgba(36,216,218,0.2)] text-[var(--electric-aqua)]"
                      >
                        {doc.filename}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex items-center gap-3">
                <div className="flex-1 relative">
                  <input
                    ref={inputRef}
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder={
                      selectedContextDocs.length > 0 
                        ? `Ask about your selected documents (${selectedContextDocs.length})...`
                        : "Ask me anything about your legal documents..."
                    }
                    className="w-full gemini-input py-3 px-4 pr-12 text-sm"
                    disabled={isLoading}
                  />
                  <button
                    onClick={toggleVoiceInput}
                    className={`absolute right-3 top-1/2 -translate-y-1/2 p-1.5 rounded-lg transition-colors ${
                      isListening 
                        ? 'bg-red-500 text-white' 
                        : 'hover:bg-[rgba(240,242,252,0.1)] text-[var(--nebula-grey)]'
                    }`}
                  >
                    <MicrophoneIcon className="h-4 w-4" />
                  </button>
                </div>
                <button
                  onClick={handleSendMessage}
                  disabled={!inputValue.trim() || isLoading}
                  className="btn-primary p-3 hover:scale-105 transition-transform disabled:opacity-50 disabled:scale-100"
                >
                  <PaperAirplaneIcon className="h-5 w-5" />
                </button>
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-[var(--nebula-grey)]">
                <span>Press Enter to send, Shift+Enter for new line</span>
                <div className="flex items-center gap-1">
                  <DocumentTextIcon className="h-3 w-3" />
                  <span>
                    {documents.length} documents available
                    {selectedContextDocs.length > 0 && ` • ${selectedContextDocs.length} in context`}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Modal */}
      {showUpload && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="w-full max-w-2xl">
            <div className="gemini-card p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-[var(--starlight-white)]">Upload Legal Documents</h2>
                <button
                  onClick={() => setShowUpload(false)}
                  className="p-2 rounded-lg hover:bg-[rgba(240,242,252,0.1)] transition-colors"
                >
                  <XMarkIcon className="h-5 w-5 text-[var(--starlight-white)]" />
                </button>
              </div>
              
              <DocumentUpload
                onUploadSuccess={handleDocumentUpload}
                onUploadError={handleUploadError}
                variant="dark"
                multiple={true}
                compact={false}
              />
              
              {documents.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-sm font-medium text-[var(--starlight-white)] mb-3">Recently Uploaded:</h3>
                  <div className="space-y-2">
                    {documents.slice(-3).map((doc) => (
                      <div key={doc.id} className="flex items-center gap-3 p-3 rounded-lg bg-[rgba(240,242,252,0.05)] border border-[rgba(138,145,180,0.2)]">
                        <DocumentTextIcon className="h-4 w-4 text-[var(--electric-aqua)]" />
                        <span className="text-sm text-[var(--starlight-white)] flex-1">{doc.filename}</span>
                        <div className="flex items-center gap-1">
                          <CheckCircleIcon className="h-4 w-4 text-green-400" />
                          <span className="text-xs text-green-400">Ready</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Context Selection Panel */}
      {showContextPanel && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="w-full max-w-2xl">
            <div className="gemini-card p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-[var(--starlight-white)]">Select Context Documents</h2>
                <button
                  onClick={() => setShowContextPanel(false)}
                  className="p-2 rounded-lg hover:bg-[rgba(240,242,252,0.1)] transition-colors"
                >
                  <XMarkIcon className="h-5 w-5 text-[var(--starlight-white)]" />
                </button>
              </div>
              
              {documents.length > 0 ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-[var(--nebula-grey)]">
                      Select documents to provide context for AI responses
                    </p>
                    <div className="flex gap-2">
                      <button
                        onClick={selectAllContextDocs}
                        className="text-xs px-3 py-1 rounded-lg bg-[rgba(36,216,218,0.1)] text-[var(--electric-aqua)] hover:bg-[rgba(36,216,218,0.2)] transition-colors"
                      >
                        Select All
                      </button>
                      <button
                        onClick={clearContextSelection}
                        className="text-xs px-3 py-1 rounded-lg bg-[rgba(240,242,252,0.1)] text-[var(--starlight-white)] hover:bg-[rgba(240,242,252,0.15)] transition-colors"
                      >
                        Clear
                      </button>
                    </div>
                  </div>

                  <div className="max-h-64 overflow-y-auto custom-scrollbar space-y-2">
                    {documents.map((doc) => (
                      <div
                        key={doc.id}
                        onClick={() => toggleContextDoc(doc.id)}
                        className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                          selectedContextDocs.includes(doc.id)
                            ? 'bg-[rgba(36,216,218,0.1)] border-[rgba(36,216,218,0.5)]'
                            : 'bg-[rgba(240,242,252,0.05)] border-[rgba(138,145,180,0.2)] hover:bg-[rgba(240,242,252,0.1)]'
                        }`}
                      >
                        <div className={`w-4 h-4 rounded border-2 flex items-center justify-center ${
                          selectedContextDocs.includes(doc.id)
                            ? 'bg-[var(--electric-aqua)] border-[var(--electric-aqua)]'
                            : 'border-[rgba(138,145,180,0.5)]'
                        }`}>
                          {selectedContextDocs.includes(doc.id) && (
                            <CheckCircleIcon className="h-3 w-3 text-black" />
                          )}
                        </div>
                        <DocumentTextIcon className={`h-4 w-4 ${
                          selectedContextDocs.includes(doc.id) 
                            ? 'text-[var(--electric-aqua)]' 
                            : 'text-[var(--nebula-grey)]'
                        }`} />
                        <div className="flex-1">
                          <span className={`text-sm font-medium ${
                            selectedContextDocs.includes(doc.id) 
                              ? 'text-[var(--electric-aqua)]' 
                              : 'text-[var(--starlight-white)]'
                          }`}>
                            {doc.filename}
                          </span>
                          <p className="text-xs text-[var(--nebula-grey)]">
                            {doc.total_pages || 0} pages • {doc.status}
                          </p>
                        </div>
                        <div className={`px-2 py-1 rounded text-xs ${
                          selectedContextDocs.includes(doc.id)
                            ? 'bg-[rgba(36,216,218,0.2)] text-[var(--electric-aqua)]'
                            : 'bg-[rgba(240,242,252,0.1)] text-[var(--nebula-grey)]'
                        }`}>
                          {selectedContextDocs.includes(doc.id) ? 'Selected' : 'Click to select'}
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-[rgba(138,145,180,0.2)]">
                    <p className="text-sm text-[var(--nebula-grey)]">
                      {selectedContextDocs.length} of {documents.length} documents selected
                    </p>
                    <button
                      onClick={() => setShowContextPanel(false)}
                      className="btn-primary px-4 py-2 text-sm hover:scale-105 transition-transform"
                    >
                      Apply Context
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <DocumentTextIcon className="h-12 w-12 text-[var(--nebula-grey)] mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-[var(--starlight-white)] mb-2">No Documents Available</h3>
                  <p className="text-sm text-[var(--nebula-grey)] mb-4">
                    Upload documents first to use them as context for AI responses
                  </p>
                  <button
                    onClick={() => {
                      setShowContextPanel(false);
                      setShowUpload(true);
                    }}
                    className="btn-primary px-4 py-2 text-sm hover:scale-105 transition-transform"
                  >
                    Upload Documents
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}