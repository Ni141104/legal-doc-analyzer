'use client';

import React, { useState, useRef, useEffect } from 'react';
import { 
  PaperAirplaneIcon, 
  SparklesIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  ChatBubbleLeftIcon,
  UserIcon,
  CpuChipIcon
} from '@heroicons/react/24/outline';
import { apiClient, QueryRequest, QueryResponse, formatConfidenceScore, getClauseTypeColor } from '@/lib/api-client';

interface QueryInterfaceProps {
  docId: string;
  documentName?: string;
  onError?: (error: string) => void;
}

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  query?: QueryRequest;
  response?: QueryResponse;
  isLoading?: boolean;
  error?: string;
}

const CLAUSE_TYPES = [
  { value: '', label: 'All Clause Types' },
  { value: 'general', label: 'General' },
  { value: 'payment', label: 'Payment' },
  { value: 'termination', label: 'Termination' },
  { value: 'liability', label: 'Liability' },
  { value: 'intellectual_property', label: 'Intellectual Property' },
  { value: 'confidentiality', label: 'Confidentiality' },
  { value: 'dispute_resolution', label: 'Dispute Resolution' },
  { value: 'force_majeure', label: 'Force Majeure' },
  { value: 'governing_law', label: 'Governing Law' },
  { value: 'amendment', label: 'Amendment' },
  { value: 'severability', label: 'Severability' },
  { value: 'entire_agreement', label: 'Entire Agreement' }
];

const SAMPLE_QUESTIONS = [
  "What are the payment terms in this contract?",
  "What are the termination clauses?",
  "Are there any liability limitations?",
  "What are the confidentiality requirements?",
  "Who is responsible for dispute resolution?"
];

export default function QueryInterface({ docId, documentName, onError }: QueryInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [selectedClauseType, setSelectedClauseType] = useState('');
  const [useHyDE, setUseHyDE] = useState(true);
  const [useCrossEncoder, setUseCrossEncoder] = useState(true);
  const [isQuerying, setIsQuerying] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (message: Omit<ChatMessage, 'id' | 'timestamp'>) => {
    const newMessage: ChatMessage = {
      ...message,
      id: Math.random().toString(36).substr(2, 9),
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newMessage]);
    return newMessage.id;
  };

  const updateMessage = (id: string, updates: Partial<ChatMessage>) => {
    setMessages(prev => prev.map(msg => 
      msg.id === id ? { ...msg, ...updates } : msg
    ));
  };

  const handleSubmitQuery = async (question?: string) => {
    const queryText = question || currentQuestion.trim();
    if (!queryText || isQuerying) return;

    setIsQuerying(true);
    setCurrentQuestion('');

    // Add user message
    const userMessageId = addMessage({
      type: 'user',
      content: queryText
    });

    // Add loading assistant message
    const assistantMessageId = addMessage({
      type: 'assistant',
      content: '',
      isLoading: true
    });

    try {
      const queryRequest: QueryRequest = {
        query: queryText
      };

      const response = await apiClient.queryDocument(docId, queryRequest);

      updateMessage(assistantMessageId, {
        content: response.answer,
        isLoading: false,
        query: queryRequest,
        response
      });

    } catch (error: any) {
      const errorMessage = error.message || 'Failed to query document';
      updateMessage(assistantMessageId, {
        content: errorMessage,
        isLoading: false,
        error: errorMessage
      });
      onError?.(errorMessage);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmitQuery();
    }
  };

  const renderMessage = (message: ChatMessage) => {
    if (message.type === 'user') {
      return (
        <div key={message.id} className="flex items-start space-x-3 mb-6">
          <div className="flex-shrink-0">
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
              <UserIcon className="w-4 h-4 text-white" />
            </div>
          </div>
          <div className="flex-1">
            <div className="bg-blue-50 rounded-lg p-4">
              <p className="text-gray-900">{message.content}</p>
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        </div>
      );
    }

    return (
      <div key={message.id} className="flex items-start space-x-3 mb-6">
        <div className="flex-shrink-0">
          <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
            {message.isLoading ? (
              <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
            ) : (
              <CpuChipIcon className="w-4 h-4 text-white" />
            )}
          </div>
        </div>
        <div className="flex-1">
          <div className={`rounded-lg p-4 ${message.error ? 'bg-red-50 border border-red-200' : 'bg-gray-50'}`}>
            {message.isLoading ? (
              <div className="flex items-center space-x-2 text-gray-600">
                <SparklesIcon className="w-4 h-4 animate-pulse" />
                <span>AI is analyzing the document...</span>
              </div>
            ) : message.error ? (
              <div className="flex items-center space-x-2 text-red-700">
                <ExclamationTriangleIcon className="w-4 h-4" />
                <span>{message.content}</span>
              </div>
            ) : (
              <div className="space-y-4">
                <p className="text-gray-900 whitespace-pre-wrap">{message.content}</p>
                
                {message.response && (
                  <div className="border-t pt-4 space-y-4">
                    {/* Confidence Score */}
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Confidence Score:</span>
                      <span className={`font-medium ${
                        message.response.confidence >= 0.8 ? 'text-green-600' :
                        message.response.confidence >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {formatConfidenceScore(message.response.confidence)}
                      </span>
                    </div>

                    {/* Sources */}
                    {message.response.sources && message.response.sources.length > 0 && (
                      <div className="mt-3">
                        <span className="text-sm font-medium text-gray-700">Sources:</span>
                        <ul className="mt-1 list-disc list-inside text-sm text-gray-600">
                          {message.response.sources.map((source, index) => (
                            <li key={index}>{source}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {message.timestamp.toLocaleTimeString()}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto">
      {/* Header */}
      <div className="border-b bg-white p-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Document Analysis</h2>
            {documentName && (
              <p className="text-sm text-gray-500 mt-1">{documentName}</p>
            )}
          </div>
          
          {/* Search Options */}
          <div className="flex items-center space-x-4">
            <select
              value={selectedClauseType}
              onChange={(e) => setSelectedClauseType(e.target.value)}
              className="text-sm border border-gray-300 rounded-md px-3 py-1 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {CLAUSE_TYPES.map(type => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
            
            <div className="flex items-center space-x-3 text-sm">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={useHyDE}
                  onChange={(e) => setUseHyDE(e.target.checked)}
                  className="mr-1"
                />
                HyDE
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={useCrossEncoder}
                  onChange={(e) => setUseCrossEncoder(e.target.checked)}
                  className="mr-1"
                />
                Cross-Encoder
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <ChatBubbleLeftIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Ask questions about your document</h3>
            <p className="text-gray-500 mb-6">Use advanced AI to analyze legal clauses and get instant answers</p>
            
            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-700">Try these sample questions:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {SAMPLE_QUESTIONS.map((question, index) => (
                  <button
                    key={index}
                    onClick={() => handleSubmitQuery(question)}
                    className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm hover:bg-blue-200 transition-colors"
                    disabled={isQuerying}
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          messages.map(renderMessage)
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t bg-white p-4">
        <div className="flex space-x-3">
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={currentQuestion}
              onChange={(e) => setCurrentQuestion(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about the document..."
              className="w-full resize-none border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={1}
              disabled={isQuerying}
            />
          </div>
          <button
            onClick={() => handleSubmitQuery()}
            disabled={!currentQuestion.trim() || isQuerying}
            className="px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isQuerying ? (
              <ClockIcon className="w-5 h-5 animate-pulse" />
            ) : (
              <PaperAirplaneIcon className="w-5 h-5" />
            )}
          </button>
        </div>
        
        {selectedClauseType && (
          <div className="mt-2 text-xs text-gray-500">
            Filtering by: <span className="font-medium">{CLAUSE_TYPES.find(t => t.value === selectedClauseType)?.label}</span>
          </div>
        )}
      </div>
    </div>
  );
}