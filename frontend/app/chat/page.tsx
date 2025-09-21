'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { useDropzone } from 'react-dropzone'
import { 
  ChatBubbleLeftEllipsisIcon,
  DocumentTextIcon,
  ArrowLeftIcon,
  PaperAirplaneIcon,
  MagnifyingGlassIcon,
  LightBulbIcon,
  ExclamationTriangleIcon,
  ClipboardDocumentListIcon,
  SparklesIcon,
  UserIcon,
  CpuChipIcon,
  DocumentPlusIcon,
  XMarkIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline'
import { HeartIcon, BoltIcon, BookOpenIcon } from '@heroicons/react/24/solid'
import toast from 'react-hot-toast'

interface ChatMessage {
  id: string
  role: 'user' | 'ai'
  content: string
  timestamp: Date
  documentContext?: string[]
  suggestions?: string[]
}

interface DocumentContext {
  id: string
  name: string
  type: string
  summary: string
  riskScore: number
}

export default function ChatPage() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [documentContext, setDocumentContext] = useState<DocumentContext[]>([])
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isNavbarVisible, setIsNavbarVisible] = useState(true)
  const [lastScrollY, setLastScrollY] = useState(0)
  const [showUploadArea, setShowUploadArea] = useState(false)
  const [showContextOptions, setShowContextOptions] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Get document context from URL params
  useEffect(() => {
    const docId = searchParams.get('docId')
    const docName = searchParams.get('docName')
    const docType = searchParams.get('docType')
    const docSummary = searchParams.get('docSummary')
    const riskScore = searchParams.get('riskScore')

    if (docId && docName) {
      const contextDoc: DocumentContext = {
        id: docId,
        name: docName,
        type: docType || 'legal_document',
        summary: docSummary || 'Legal document for analysis',
        riskScore: parseInt(riskScore || '50')
      }
      setDocumentContext([contextDoc])
      setActiveDocumentId(docId)

      // Add welcome message
      const welcomeMessage: ChatMessage = {
        id: 'welcome-' + Date.now(),
        role: 'ai',
        content: `Hello! I'm your AI legal assistant. I'm ready to analyze and discuss "${docName}". Feel free to ask me anything about this document - risk assessment, clause analysis, recommendations, or general legal questions.`,
        timestamp: new Date(),
        documentContext: [docId],
        suggestions: [
          'What are the main risks in this document?',
          'Can you explain any unusual clauses?',
          'What should I negotiate?',
          'Summarize the key terms'
        ]
      }
      setMessages([welcomeMessage])
    } else {
      // Generic welcome for no specific document
      const genericWelcome: ChatMessage = {
        id: 'welcome-generic-' + Date.now(),
        role: 'ai',
        content: `Welcome to your AI Legal Assistant! I'm here to help you with legal document analysis, contract review, risk assessment, and general legal questions. How can I assist you today?`,
        timestamp: new Date(),
        suggestions: [
          'How do I review a contract?',
          'What are red flags in legal documents?',
          'Explain common contract terms',
          'Help me understand legal language'
        ]
      }
      setMessages([genericWelcome])
    }
  }, [searchParams])

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Focus input on load
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // Auto-scroll textarea to bottom when content changes
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.scrollTop = inputRef.current.scrollHeight
    }
  }, [inputMessage])

  // Scroll detection for navbar visibility
  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY
      
      // Show navbar when scrolling up or at top
      if (currentScrollY < lastScrollY || currentScrollY < 10) {
        setIsNavbarVisible(true)
      } 
      // Hide navbar when scrolling down (and not at the very top)
      else if (currentScrollY > lastScrollY && currentScrollY > 100) {
        setIsNavbarVisible(false)
      }
      
      setLastScrollY(currentScrollY)
    }

    // Throttle scroll events for performance
    let ticking = false
    const throttledHandleScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          handleScroll()
          ticking = false
        })
        ticking = true
      }
    }

    window.addEventListener('scroll', throttledHandleScroll, { passive: true })
    return () => window.removeEventListener('scroll', throttledHandleScroll)
  }, [lastScrollY])

  // Mock data generators for uploaded documents
  const generateMockSummary = (filename: string) => {
    const summaries = [
      "Rental agreement for a 2-bedroom apartment with standard terms",
      "Service contract with unusual liability clauses requiring attention", 
      "Employment agreement with competitive benefits and clear termination terms",
      "Loan agreement with variable interest rates and early payment penalties"
    ]
    return summaries[Math.floor(Math.random() * summaries.length)]
  }

  const generateMockAbnormalPoints = () => [
    "Unusually high early termination penalty (2x monthly rent)",
    "Automatic renewal clause without clear opt-out procedure",
    "Liability waiver extends beyond reasonable scope",
    "Payment terms favor one party disproportionately"
  ]

  const generateMockAISuggestions = () => [
    "Negotiate early termination penalty from 2 months to 1 month rent",
    "Add 30-day written notice requirement for automatic renewal", 
    "Limit liability waiver to exclude gross negligence",
    "Request payment terms adjustment with 15-day grace period"
  ]

  // Document upload functionality
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    setIsUploading(true)
    
    try {
      // Simulate API call - this would integrate with Firebase Storage and Google Cloud Document AI
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      const newDoc: DocumentContext = {
        id: `doc-${Date.now()}`,
        name: file.name,
        type: file.name.includes('rental') ? 'rental_agreement' : 
              file.name.includes('loan') ? 'loan_contract' :
              file.name.includes('employment') ? 'employment_contract' : 'legal_document',
        summary: generateMockSummary(file.name),
        riskScore: Math.floor(Math.random() * 100) + 1
      }
      
      // Add to document context
      setDocumentContext(prev => [newDoc, ...prev])
      setActiveDocumentId(newDoc.id)
      setShowUploadArea(false)
      
      // Add AI message about the new document
      const aiMessage: ChatMessage = {
        id: 'upload-' + Date.now(),
        role: 'ai',
        content: `Great! I've successfully analyzed "${file.name}". This document has a risk score of ${newDoc.riskScore}/100. I'm now ready to answer any questions about this document or help you understand its contents. What would you like to know?`,
        timestamp: new Date(),
        documentContext: [newDoc.id],
        suggestions: [
          'What are the main risks in this document?',
          'Explain the key terms and conditions',
          'What should I be careful about?',
          'How does this compare to standard contracts?'
        ]
      }
      
      setMessages(prev => [...prev, aiMessage])
      toast.success('Document uploaded and analyzed!')
      
    } catch (error) {
      console.error('Upload error:', error)
      toast.error('Failed to upload document')
    } finally {
      setIsUploading(false)
    }
  }, [])

  // Direct file upload handler
  const handleDirectFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      await onDrop([file])
      // Reset file input
      event.target.value = ''
    }
  }

  // Trigger file picker
  const triggerFileUpload = () => {
    fileInputRef.current?.click()
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'application/msword': ['.doc', '.docx']
    },
    maxFiles: 1,
    disabled: isUploading
  })

  const sendMessage = async () => {
    if (!inputMessage.trim()) return

    const userMessage: ChatMessage = {
      id: 'user-' + Date.now(),
      role: 'user',
      content: inputMessage,
      timestamp: new Date(),
      documentContext: activeDocumentId ? [activeDocumentId] : undefined
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsTyping(true)

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: ChatMessage = {
        id: 'ai-' + Date.now(),
        role: 'ai',
        content: generateAIResponse(inputMessage, documentContext[0]),
        timestamp: new Date(),
        documentContext: activeDocumentId ? [activeDocumentId] : undefined,
        suggestions: generateSuggestions(inputMessage)
      }
      setMessages(prev => [...prev, aiResponse])
      setIsTyping(false)
    }, 1500)
  }

  const generateAIResponse = (question: string, doc?: DocumentContext): string => {
    const responses = [
      `Based on my analysis of "${doc?.name || 'your document'}", I can provide detailed insights. ${question.toLowerCase().includes('risk') ? 'The risk assessment shows several key areas that need attention...' : 'Let me break down the important aspects...'}`,
      `Great question! Looking at the document context, I notice ${doc?.riskScore && doc.riskScore > 70 ? 'some high-risk elements' : 'moderate risk factors'} that we should discuss...`,
      `This is an excellent point to clarify. In legal documents like this, it's crucial to understand that...`,
      `I've reviewed the relevant sections, and here's what you need to know...`
    ]
    return responses[Math.floor(Math.random() * responses.length)]
  }

  const generateSuggestions = (lastMessage: string): string[] => {
    const suggestions = [
      ['Tell me more about this', 'What are the implications?', 'How can I address this?'],
      ['Show me examples', 'What should I watch out for?', 'Any red flags?'],
      ['Explain in simple terms', 'What does this mean legally?', 'How does this affect me?'],
      ['What are my options?', 'Can this be negotiated?', 'What would you recommend?']
    ]
    return suggestions[Math.floor(Math.random() * suggestions.length)]
  }

  const handleSuggestionClick = (suggestion: string) => {
    setInputMessage(suggestion)
    inputRef.current?.focus()
  }

  const switchActiveDocument = (docId: string) => {
    setActiveDocumentId(docId)
    const doc = documentContext.find(d => d.id === docId)
    if (doc) {
      const switchMessage: ChatMessage = {
        id: 'switch-' + Date.now(),
        role: 'ai',
        content: `I've switched to analyzing "${doc.name}". This document has a risk score of ${doc.riskScore}/100. What would you like to know about this document?`,
        timestamp: new Date(),
        documentContext: [docId],
        suggestions: [
          'Analyze the risks in this document',
          'Summarize the key points',
          'What should I negotiate?',
          'Compare with my other documents'
        ]
      }
      setMessages(prev => [...prev, switchMessage])
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="chat-layout" style={{backgroundColor: 'var(--cosmic-background)'}}>
      {/* Header */}
      <div 
        className={`gemini-card m-6 mb-0 transition-all duration-300 fixed top-0 left-0 right-0 z-50 ${
          isNavbarVisible ? 'translate-y-0' : '-translate-y-full'
        }`}
      >
        <div className="p-6 border-b" style={{borderColor: 'var(--nebula-grey)'}}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => router.back()}
                className="btn-ghost p-2 rounded-lg"
              >
                <ArrowLeftIcon className="h-5 w-5" />
              </button>
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-full flex items-center justify-center gemini-gradient">
                  <CpuChipIcon className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold" style={{color: 'var(--starlight-white)'}}>
                    AI Legal Assistant
                  </h1>
                  <p className="text-sm secondary-text">
                    {documentContext[0] ? `Analyzing: ${documentContext[0].name}` : 'Ready to help with legal questions'}
                  </p>
                </div>
              </div>
            </div>

            {/* Document Context Indicator */}
            {documentContext[0] && (
              <div className="flex items-center space-x-2 gemini-card p-3">
                <DocumentTextIcon className="h-5 w-5" style={{color: 'var(--electric-aqua)'}} />
                <div className="text-sm">
                  <div className="font-medium" style={{color: 'var(--starlight-white)'}}>
                    {documentContext.find(d => d.id === activeDocumentId)?.name || documentContext[0].name}
                  </div>
                  <div className="secondary-text">
                    Risk: {documentContext.find(d => d.id === activeDocumentId)?.riskScore || documentContext[0].riskScore}/100
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Spacer for fixed header */}
      <div className={`transition-all duration-300 ${isNavbarVisible ? 'h-32' : 'h-0'}`} />

      {/* Main Content Area */}
      <div className="chat-main mx-6">
        <div className="chat-content gemini-card mr-2 mb-6 transition-all duration-300">
          {/* Messages Area - Scrollable */}
          <div className="messages-area p-6 space-y-6 custom-scrollbar">
            {messages.map((message) => (
              <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-3xl ${message.role === 'user' ? 'order-2' : ''}`}>
                  {/* Avatar */}
                  <div className={`flex items-start space-x-3 ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      message.role === 'user' 
                        ? 'bg-gradient-to-r from-blue-500 to-purple-600' 
                        : 'gemini-gradient'
                    }`}>
                      {message.role === 'user' ? (
                        <UserIcon className="h-5 w-5 text-white" />
                      ) : (
                        <SparklesIcon className="h-5 w-5 text-white" />
                      )}
                    </div>

                    {/* Message Content */}
                    <div className={`flex-1 ${message.role === 'user' ? 'text-right' : ''}`}>
                      <div className={`rounded-2xl p-4 ${
                        message.role === 'user'
                          ? 'gemini-gradient text-white'
                          : 'border border-opacity-20'
                      }`} style={message.role === 'ai' ? {
                        backgroundColor: 'rgba(240, 242, 252, 0.03)',
                        borderColor: 'var(--nebula-grey)'
                      } : {}}>
                        <p className={`leading-relaxed ${
                          message.role === 'user' ? 'text-white' : 'text-primary-text'
                        }`}>
                          {message.content}
                        </p>
                      </div>

                      {/* Timestamp */}
                      <div className={`text-xs mt-2 secondary-text ${message.role === 'user' ? 'text-right' : ''}`}>
                        {message.timestamp.toLocaleTimeString()}
                      </div>

                      {/* AI Suggestions */}
                      {message.role === 'ai' && message.suggestions && (
                        <div className="mt-4 space-y-2">
                          <p className="text-sm secondary-text">Suggested questions:</p>
                          <div className="flex flex-wrap gap-2">
                            {message.suggestions.map((suggestion, idx) => (
                              <button
                                key={idx}
                                onClick={() => handleSuggestionClick(suggestion)}
                                className="text-xs px-3 py-1 rounded-full border border-opacity-30 hover:border-opacity-60 transition-all secondary-text hover:text-primary-text"
                                style={{borderColor: 'var(--electric-aqua)'}}
                              >
                                {suggestion}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* Typing Indicator */}
            {isTyping && (
              <div className="flex justify-start">
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 rounded-full gemini-gradient flex items-center justify-center">
                    <SparklesIcon className="h-5 w-5 text-white" />
                  </div>
                  <div className="rounded-2xl p-4 border border-opacity-20" style={{
                    backgroundColor: 'rgba(240, 242, 252, 0.03)',
                    borderColor: 'var(--nebula-grey)'
                  }}>
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 animate-bounce"></div>
                      <div className="w-2 h-2 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 animate-bounce" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 rounded-full bg-gradient-to-r from-pink-500 to-blue-500 animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area - Fixed at bottom */}
          <div className="chat-input-area p-6 shadow-lg">
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.doc,.docx,.txt"
              onChange={handleDirectFileUpload}
              style={{ display: 'none' }}
            />
            
            {/* Action Buttons */}
            <div className="flex space-x-3 mb-4">
              <button
                onClick={triggerFileUpload}
                className="btn-secondary flex items-center space-x-2"
                disabled={isUploading}
              >
                {isUploading ? (
                  <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <DocumentPlusIcon className="h-4 w-4" />
                )}
                <span>{isUploading ? 'Uploading...' : 'Add Document'}</span>
              </button>
              <button
                onClick={() => {
                  if (documentContext.length > 1) {
                    setShowContextOptions(!showContextOptions)
                  } else {
                    triggerFileUpload()
                  }
                }}
                className="btn-ghost flex items-center space-x-2"
                disabled={isUploading}
              >
                <ClipboardDocumentListIcon className="h-4 w-4" />
                <span>Add Context</span>
                {documentContext.length > 1 && (
                  <span className="text-xs bg-gradient-to-r from-blue-500 to-purple-600 text-white px-2 py-1 rounded-full">
                    {documentContext.length}
                  </span>
                )}
              </button>
            </div>
            
            {/* Context Options Dropdown */}
            {showContextOptions && documentContext.length > 1 && (
              <div className="mb-4 gemini-card p-4">
                <h4 className="text-sm font-medium mb-3" style={{color: 'var(--starlight-white)'}}>
                  Switch Document Context
                </h4>
                <div className="space-y-2">
                  {documentContext.map((doc) => (
                    <button
                      key={doc.id}
                      onClick={() => {
                        switchActiveDocument(doc.id)
                        setShowContextOptions(false)
                      }}
                      className={`w-full text-left p-2 rounded-lg border border-opacity-20 hover:border-opacity-40 transition-all flex items-center justify-between ${
                        activeDocumentId === doc.id ? 'border-opacity-60' : ''
                      }`}
                      style={{
                        borderColor: activeDocumentId === doc.id ? 'var(--electric-aqua)' : 'var(--nebula-grey)',
                        backgroundColor: activeDocumentId === doc.id ? 'rgba(36, 216, 218, 0.05)' : 'transparent'
                      }}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="font-medium truncate text-primary-text text-sm">
                          {doc.name}
                        </div>
                        <div className="text-xs secondary-text capitalize">
                          {doc.type.replace('_', ' ')}
                        </div>
                      </div>
                      <div className="flex items-center space-x-2 ml-2">
                        {activeDocumentId === doc.id && (
                          <CheckCircleIcon className="h-4 w-4" style={{color: 'var(--electric-aqua)'}} />
                        )}
                        <div className="text-xs font-medium" style={{
                          color: doc.riskScore > 70 ? 'var(--electric-rose)' : 
                                 doc.riskScore > 40 ? 'var(--vibrant-magenta)' : 'var(--electric-aqua)'
                        }}>
                          {doc.riskScore}/100
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
                <button
                  onClick={() => setShowContextOptions(false)}
                  className="mt-3 text-sm secondary-text hover:text-primary-text transition-colors"
                >
                  Close
                </button>
              </div>
            )}
            
            {/* Text Input Area */}
            <div className="relative">
              <textarea
                ref={inputRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about your legal document or general legal questions..."
                className="chat-input gemini-input w-full resize-none"
                disabled={isTyping}
                rows={3}
              />
              <button
                onClick={sendMessage}
                disabled={!inputMessage.trim() || isTyping}
                className="send-button btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <PaperAirplaneIcon className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className={`w-64 space-y-4 mb-6 transition-all duration-300 ${
          isNavbarVisible ? 'mt-0' : '-mt-16'
        }`}>
          {/* Document Upload Area */}
          {showUploadArea && (
            <div className="gemini-card p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold flex items-center" style={{color: 'var(--starlight-white)'}}>
                  <DocumentPlusIcon className="h-5 w-5 mr-2" style={{color: 'var(--electric-aqua)'}} />
                  Upload Document
                </h3>
                <button
                  onClick={() => setShowUploadArea(false)}
                  className="text-primary-text hover:text-white transition-colors"
                >
                  <XMarkIcon className="h-5 w-5" />
                </button>
              </div>
              
              <div {...getRootProps()} className={`gemini-dropzone p-6 cursor-pointer ${isDragActive ? 'drag-active' : ''} ${isUploading ? 'uploading opacity-50 cursor-not-allowed' : ''}`}>
                <input {...getInputProps()} />
                {isUploading ? (
                  <div className="space-y-2">
                    <div className="gemini-spinner mx-auto"></div>
                    <p className="text-sm secondary-text font-medium text-center">Analyzing...</p>
                  </div>
                ) : (
                  <div className="space-y-2 text-center">
                    <DocumentPlusIcon className="w-12 h-12 mx-auto mb-2" style={{color: 'var(--electric-aqua)'}} />
                    <p className="text-sm font-medium" style={{color: 'var(--starlight-white)'}}>
                      {isDragActive ? 'Drop here!' : 'Drop your Legal Document'}
                    </p>
                    <p className="text-xs secondary-text">
                      PDF, DOC, DOCX supported
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Document Context Manager */}
          {documentContext.length > 0 && (
            <div className="gemini-card p-4">
              <h3 className="font-semibold mb-3 flex items-center text-sm" style={{color: 'var(--starlight-white)'}}>
                <DocumentTextIcon className="h-4 w-4 mr-2" style={{color: 'var(--electric-aqua)'}} />
                Document Context ({documentContext.length})
              </h3>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {documentContext.map((doc) => (
                  <div 
                    key={doc.id}
                    className={`p-2 rounded-lg border border-opacity-20 hover:border-opacity-40 transition-all cursor-pointer ${
                      activeDocumentId === doc.id ? 'border-opacity-60' : ''
                    }`}
                    style={{
                      borderColor: activeDocumentId === doc.id ? 'var(--electric-aqua)' : 'var(--nebula-grey)',
                      backgroundColor: activeDocumentId === doc.id ? 'rgba(36, 216, 218, 0.05)' : 'transparent'
                    }}
                    onClick={() => switchActiveDocument(doc.id)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="font-medium truncate text-primary-text text-sm">
                          {doc.name}
                        </div>
                        <div className="text-xs secondary-text capitalize">
                          {doc.type.replace('_', ' ')}
                        </div>
                      </div>
                      <div className="flex items-center space-x-1 ml-1">
                        {activeDocumentId === doc.id && (
                          <CheckCircleIcon className="h-3 w-3" style={{color: 'var(--electric-aqua)'}} />
                        )}
                        <div className="text-xs text-right">
                          <div className="font-medium" style={{
                            color: doc.riskScore > 70 ? 'var(--electric-rose)' : 
                                   doc.riskScore > 40 ? 'var(--vibrant-magenta)' : 'var(--electric-aqua)'
                          }}>
                            {doc.riskScore}/100
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {/* Quick Actions */}
          <div className="gemini-card p-4">
            <h3 className="font-semibold mb-3 flex items-center text-sm" style={{color: 'var(--starlight-white)'}}>
              <LightBulbIcon className="h-4 w-4 mr-2" style={{color: 'var(--electric-aqua)'}} />
              Quick Actions
            </h3>
            <div className="space-y-2">
              <button 
                onClick={() => handleSuggestionClick("Analyze the risks in this document")}
                className="w-full text-left p-2 rounded-lg border border-opacity-20 hover:border-opacity-40 transition-all"
                style={{borderColor: 'var(--electric-rose)'}}
              >
                <div className="flex items-center space-x-2">
                  <ExclamationTriangleIcon className="h-4 w-4" style={{color: 'var(--electric-rose)'}} />
                  <span className="secondary-text text-sm">Risk Analysis</span>
                </div>
              </button>
              
              <button 
                onClick={() => handleSuggestionClick("Summarize the key terms and conditions")}
                className="w-full text-left p-2 rounded-lg border border-opacity-20 hover:border-opacity-40 transition-all"
                style={{borderColor: 'var(--electric-aqua)'}}
              >
                <div className="flex items-center space-x-2">
                  <ClipboardDocumentListIcon className="h-4 w-4" style={{color: 'var(--electric-aqua)'}} />
                  <span className="secondary-text text-sm">Key Terms</span>
                </div>
              </button>
              
              <button 
                onClick={() => handleSuggestionClick("What should I negotiate in this contract?")}
                className="w-full text-left p-2 rounded-lg border border-opacity-20 hover:border-opacity-40 transition-all"
                style={{borderColor: 'var(--vibrant-magenta)'}}
              >
                <div className="flex items-center space-x-2">
                  <BoltIcon className="h-4 w-4" style={{color: 'var(--vibrant-magenta)'}} />
                  <span className="secondary-text text-sm">Negotiation Tips</span>
                </div>
              </button>
            </div>
          </div>

          {/* Document Info */}
          {activeDocumentId && documentContext.find(d => d.id === activeDocumentId) && (
            <div className="gemini-card p-6">
              <h3 className="font-semibold mb-4 flex items-center" style={{color: 'var(--starlight-white)'}}>
                <DocumentTextIcon className="h-5 w-5 mr-2" style={{color: 'var(--electric-aqua)'}} />
                Active Document
              </h3>
              {(() => {
                const activeDoc = documentContext.find(d => d.id === activeDocumentId)!
                return (
                  <div className="space-y-3">
                    <div>
                      <label className="text-sm secondary-text">Document Name</label>
                      <p className="text-primary-text font-medium">{activeDoc.name}</p>
                    </div>
                    <div>
                      <label className="text-sm secondary-text">Type</label>
                      <p className="text-primary-text capitalize">{activeDoc.type.replace('_', ' ')}</p>
                    </div>
                    <div>
                      <label className="text-sm secondary-text">Risk Score</label>
                      <div className="flex items-center space-x-2">
                        <div className="flex-1 h-2 rounded-full" style={{backgroundColor: 'var(--nebula-grey)'}}>
                          <div 
                            className="h-2 rounded-full transition-all"
                            style={{
                              width: `${activeDoc.riskScore}%`,
                              background: activeDoc.riskScore > 70 
                                ? 'var(--electric-rose)' 
                                : activeDoc.riskScore > 40 
                                  ? 'var(--vibrant-magenta)' 
                                  : 'var(--electric-aqua)'
                            }}
                          ></div>
                        </div>
                        <span className="text-sm text-primary-text">{activeDoc.riskScore}/100</span>
                      </div>
                    </div>
                  </div>
                )
              })()}
            </div>
          )}

          {/* Legal Resources */}
          <div className="gemini-card p-4">
            <h3 className="font-semibold mb-3 flex items-center text-sm" style={{color: 'var(--starlight-white)'}}>
              <BookOpenIcon className="h-4 w-4 mr-2" style={{color: 'var(--electric-aqua)'}} />
              Legal Resources
            </h3>
            <div className="space-y-1 text-sm">
              <button className="w-full text-left secondary-text hover:text-primary-text transition-colors p-1">
                → Contract Law
              </button>
              <button className="w-full text-left secondary-text hover:text-primary-text transition-colors p-1">
                → Risk Assessment
              </button>
              <button className="w-full text-left secondary-text hover:text-primary-text transition-colors p-1">
                → Negotiation Tips
              </button>
              <button className="w-full text-left secondary-text hover:text-primary-text transition-colors p-1">
                → Legal Terms
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}