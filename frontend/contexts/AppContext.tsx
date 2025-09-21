'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { DocumentMetadata } from '@/lib/api-client';

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  thinking?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

interface AppContextType {
  // Documents state
  documents: DocumentMetadata[];
  addDocument: (document: DocumentMetadata) => void;
  removeDocument: (documentId: string) => void;
  
  // Chat history state
  chatSessions: ChatSession[];
  currentSessionId: string | null;
  createNewSession: () => string;
  loadSession: (sessionId: string) => void;
  updateCurrentSession: (messages: ChatMessage[]) => void;
  deleteSession: (sessionId: string) => void;
  getCurrentSession: () => ChatSession | null;
  
  // Initial context for Ask AI
  initialChatContext: string | null;
  setInitialChatContext: (documentId: string | null) => void;
  clearInitialChatContext: () => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

const STORAGE_KEYS = {
  DOCUMENTS: 'legal-analyzer-documents',
  CHAT_SESSIONS: 'legal-analyzer-chat-sessions',
  CURRENT_SESSION: 'legal-analyzer-current-session'
};

export function AppProvider({ children }: { children: ReactNode }) {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [initialChatContext, setInitialChatContextState] = useState<string | null>(null);

  // Load data from localStorage on mount
  useEffect(() => {
    // Load documents
    const savedDocuments = localStorage.getItem(STORAGE_KEYS.DOCUMENTS);
    if (savedDocuments) {
      try {
        const parsedDocs = JSON.parse(savedDocuments);
        setDocuments(parsedDocs);
      } catch (error) {
        console.error('Error loading documents from localStorage:', error);
      }
    }

    // Load chat sessions
    const savedSessions = localStorage.getItem(STORAGE_KEYS.CHAT_SESSIONS);
    if (savedSessions) {
      try {
        const parsedSessions = JSON.parse(savedSessions).map((session: any) => ({
          ...session,
          createdAt: new Date(session.createdAt),
          updatedAt: new Date(session.updatedAt),
          messages: session.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }))
        }));
        setChatSessions(parsedSessions);
      } catch (error) {
        console.error('Error loading chat sessions from localStorage:', error);
      }
    }

    // Load current session
    const savedCurrentSession = localStorage.getItem(STORAGE_KEYS.CURRENT_SESSION);
    if (savedCurrentSession) {
      setCurrentSessionId(savedCurrentSession);
    }
  }, []);

  // Save documents to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.DOCUMENTS, JSON.stringify(documents));
  }, [documents]);

  // Save chat sessions to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.CHAT_SESSIONS, JSON.stringify(chatSessions));
  }, [chatSessions]);

  // Save current session ID whenever it changes
  useEffect(() => {
    if (currentSessionId) {
      localStorage.setItem(STORAGE_KEYS.CURRENT_SESSION, currentSessionId);
    }
  }, [currentSessionId]);

  const addDocument = (document: DocumentMetadata) => {
    setDocuments(prev => {
      const existing = prev.find(doc => doc.id === document.id);
      if (existing) {
        return prev.map(doc => doc.id === document.id ? document : doc);
      }
      return [...prev, document];
    });
  };

  const removeDocument = (documentId: string) => {
    setDocuments(prev => prev.filter(doc => doc.id !== documentId));
  };

  const createNewSession = (): string => {
    const sessionId = Date.now().toString();
    const newSession: ChatSession = {
      id: sessionId,
      title: `Chat ${new Date().toLocaleDateString()}`,
      messages: [{
        id: '1',
        type: 'system',
        content: "👋 Welcome to AI Legal Assistant! I can help you analyze your legal documents, identify risks, and answer complex legal questions. Upload documents or ask me anything!",
        timestamp: new Date()
      }],
      createdAt: new Date(),
      updatedAt: new Date()
    };

    setChatSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(sessionId);
    return sessionId;
  };

  const loadSession = (sessionId: string) => {
    setCurrentSessionId(sessionId);
  };

  const updateCurrentSession = (messages: ChatMessage[]) => {
    if (!currentSessionId) return;

    setChatSessions(prev => prev.map(session => {
      if (session.id === currentSessionId) {
        // Update title based on first user message
        const firstUserMessage = messages.find(msg => msg.type === 'user');
        const title = firstUserMessage 
          ? firstUserMessage.content.slice(0, 50) + (firstUserMessage.content.length > 50 ? '...' : '')
          : session.title;

        return {
          ...session,
          title,
          messages,
          updatedAt: new Date()
        };
      }
      return session;
    }));
  };

  const deleteSession = (sessionId: string) => {
    setChatSessions(prev => prev.filter(session => session.id !== sessionId));
    if (currentSessionId === sessionId) {
      setCurrentSessionId(null);
    }
  };

  const getCurrentSession = (): ChatSession | null => {
    if (!currentSessionId) return null;
    return chatSessions.find(session => session.id === currentSessionId) || null;
  };

  const setInitialChatContext = (documentId: string | null) => {
    setInitialChatContextState(documentId);
  };

  const clearInitialChatContext = () => {
    setInitialChatContextState(null);
  };

  const value: AppContextType = {
    documents,
    addDocument,
    removeDocument,
    chatSessions,
    currentSessionId,
    createNewSession,
    loadSession,
    updateCurrentSession,
    deleteSession,
    getCurrentSession,
    initialChatContext,
    setInitialChatContext,
    clearInitialChatContext
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
}