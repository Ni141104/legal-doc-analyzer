'use client'

import { useState, useRef, useEffect } from 'react'
import { 
  PaperAirplaneIcon,
  DocumentTextIcon,
  QuestionMarkCircleIcon,
  ClipboardDocumentIcon
} from '@heroicons/react/24/outline'

interface ChatMessage {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  sources?: Array<{
    docId: string
    docName: string
    clauseId?: string
    relevanceScore: number
  }>
}

interface ChatbotProps {
  documentIds: string[]
  focusDocId?: string | null
  onClose: () => void
}

export default function Chatbot({ documentIds, focusDocId, onClose }: ChatbotProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Initialize with welcome message
    const welcomeMessage: ChatMessage = {
      id: 'welcome',
      type: 'assistant',
      content: focusDocId 
        ? `Hi! I'm your AI legal assistant. I'm ready to answer questions about the document you selected. What would you like to know?`
        : `Hi! I'm your AI legal assistant. I have access to ${documentIds.length} document${documentIds.length !== 1 ? 's' : ''} in the current context. What legal questions can I help you with?`,
      timestamp: new Date()
    }
    setMessages([welcomeMessage])
  }, [documentIds, focusDocId])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: input.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      // Simulate API call to backend
      await new Promise(resolve => setTimeout(resolve, 1500))

      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        type: 'assistant',
        content: generateMockResponse(userMessage.content),
        timestamp: new Date(),
        sources: generateMockSources()
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Chat error:', error)
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        type: 'assistant',
        content: 'I apologize, but I encountered an error while processing your question. Please try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const generateMockResponse = (question: string) => {
    const responses = [
      "Based on the rental agreement, the early termination clause requires 60 days notice and payment of two months' rent as penalty. This is above the typical market standard of one month's rent.",
      "The liability clause in section 4.2 appears to be unusually broad, extending beyond normal wear and tear to include any damage regardless of fault. I recommend negotiating to limit liability to intentional damage or gross negligence.",
      "The automatic renewal clause lacks clear opt-out language. Standard practice requires at least 30 days written notice before auto-renewal. You should request this protection be added.",
      "The payment terms specify a 5-day grace period, which is shorter than the typical 10-15 days. The late fee of $75 is within reasonable bounds for this rent amount."
    ]
    
    return responses[Math.floor(Math.random() * responses.length)]
  }

  const generateMockSources = () => [
    {
      docId: 'doc-1',
      docName: 'Rental Agreement.pdf',
      clauseId: 'termination-clause',
      relevanceScore: 0.92
    },
    {
      docId: 'doc-1',
      docName: 'Rental Agreement.pdf',
      clauseId: 'liability-clause',
      relevanceScore: 0.87
    }
  ]

  const suggestedQuestions = [
    "What are the early termination penalties?",
    "Are there any unusual liability clauses?",
    "What are the payment terms and late fees?",
    "How does the automatic renewal work?",
    "What maintenance responsibilities do I have?"
  ]

  return (
    <div className="h-full flex flex-col glass-card">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white/50 backdrop-blur-sm rounded-t-lg">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
            <QuestionMarkCircleIcon className="h-5 w-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gradient">AI Legal Assistant</h3>
            <p className="text-sm text-gray-600">
              {focusDocId 
                ? "Focused on selected document" 
                : `Context: ${documentIds.length} document${documentIds.length !== 1 ? 's' : ''}`
              }
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 transition-colors p-1"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-3xl rounded-xl px-4 py-3 transition-all duration-300 ${
                message.type === 'user'
                  ? 'glass-card bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                  : 'glass-card bg-white/80 text-gray-900'
              }`}
            >
              <p className="text-sm leading-relaxed">{message.content}</p>
              
              {/* Sources */}
              {message.sources && message.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <p className="text-xs font-medium text-gray-600 mb-2">Sources:</p>
                  <div className="space-y-1">
                    {message.sources.map((source, idx) => (
                      <div key={idx} className="flex items-center justify-between text-xs">
                        <div className="flex items-center space-x-2">
                          <DocumentTextIcon className="h-3 w-3 text-gray-500" />
                          <span className="text-gray-700">{source.docName}</span>
                          {source.clauseId && (
                            <span className="text-gray-500">• {source.clauseId}</span>
                          )}
                        </div>
                        <span className="text-gray-500">{Math.round(source.relevanceScore * 100)}% match</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              <p className="text-xs text-gray-500 mt-2">
                {message.timestamp.toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg px-4 py-3 max-w-3xl">
              <div className="flex items-center space-x-2">
                <div className="loading-spinner"></div>
                <span className="text-sm text-gray-600">AI is analyzing...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Questions */}
      {messages.length === 1 && (
        <div className="px-4 py-2 border-t bg-gray-50">
          <p className="text-sm font-medium text-gray-700 mb-2">Suggested questions:</p>
          <div className="flex flex-wrap gap-2">
            {suggestedQuestions.slice(0, 3).map((question, idx) => (
              <button
                key={idx}
                onClick={() => setInput(question)}
                className="text-xs px-3 py-1 bg-white border border-gray-200 rounded-full hover:bg-gray-50 text-gray-700"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t">
        <div className="flex space-x-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your legal documents..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            <PaperAirplaneIcon className="h-4 w-4" />
          </button>
        </div>
        
        <div className="flex items-center justify-between mt-2">
          <p className="text-xs text-gray-500">
            Press Enter to send • Responses include clause-level citations
          </p>
          <button
            type="button"
            className="text-xs text-blue-600 hover:text-blue-800 flex items-center"
          >
            <ClipboardDocumentIcon className="h-3 w-3 mr-1" />
            Export Chat
          </button>
        </div>
      </form>
    </div>
  )
}