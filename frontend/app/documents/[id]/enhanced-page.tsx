'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import { 
  DocumentTextIcon, 
  ChatBubbleLeftEllipsisIcon,
  ExclamationTriangleIcon,
  ArrowLeftIcon,
  MagnifyingGlassIcon,
  ShareIcon,
  PrinterIcon
} from '@heroicons/react/24/outline'
import ClauseNavigator from '@/components/ClauseNavigator'
import Chatbot from '@/components/Chatbot'
import type { DocumentAnalysis } from '@/types'

export default function DocumentAnalysisPage() {
  const params = useParams()
  const docId = params.id as string
  
  const [analysis, setAnalysis] = useState<DocumentAnalysis | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedClauseId, setSelectedClauseId] = useState<string | null>(null)
  const [highlightedSpans, setHighlightedSpans] = useState<Array<{ 
    start: number
    end: number
    type: 'important' | 'risk'
    clauseId?: string
  }>>([])
  const [showChatbot, setShowChatbot] = useState(false)
  const [documentText] = useState(getMockDocumentText())
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedText, setSelectedText] = useState('')
  const [showQueryDialog, setShowQueryDialog] = useState(false)

  useEffect(() => {
    loadDocumentAnalysis()
  }, [docId])

  useEffect(() => {
    // Auto-highlight important and risky clauses
    if (analysis) {
      const spans = analysis.clause_cards.flatMap(clause => 
        clause.source_spans.map(span => ({
          ...span,
          type: clause.risk_level === 'HIGH' ? 'risk' as const : 'important' as const,
          clauseId: clause.clause_id
        }))
      )
      setHighlightedSpans(spans)
    }
  }, [analysis])

  const loadDocumentAnalysis = async () => {
    try {
      setLoading(true)
      // For demo purposes, use mock data
      setAnalysis(getMockAnalysis(docId))
    } catch (err) {
      console.error('Analysis loading error:', err)
      setError('Failed to load document analysis')
    } finally {
      setLoading(false)
    }
  }

  const handleClauseSelect = (clauseId: string) => {
    setSelectedClauseId(clauseId)
    
    // Scroll to clause in document
    const clause = analysis?.clause_cards.find(c => c.clause_id === clauseId)
    if (clause?.source_spans && clause.source_spans.length > 0) {
      const firstSpan = clause.source_spans[0]
      const element = document.querySelector(`[data-span-start="${firstSpan.start}"]`)
      element?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }

  const createHighlightedText = () => {
    if (!highlightedSpans.length) {
      // Simple search highlighting if no clause spans
      if (searchTerm) {
        const regex = new RegExp(`(${searchTerm})`, 'gi')
        return documentText.replace(regex, '<mark class="bg-blue-200">$1</mark>')
      }
      return documentText
    }

    // Sort spans by start position
    const sortedSpans = [...highlightedSpans].sort((a, b) => a.start - b.start)
    
    let result = ''
    let lastEnd = 0

    sortedSpans.forEach((span, idx) => {
      // Add text before this span
      const beforeText = documentText.slice(lastEnd, span.start)
      if (searchTerm) {
        const regex = new RegExp(`(${searchTerm})`, 'gi')
        result += beforeText.replace(regex, '<mark class="bg-blue-200">$1</mark>')
      } else {
        result += beforeText
      }
      
      // Add highlighted text
      const spanText = documentText.slice(span.start, span.end)
      const highlightClass = span.type === 'risk' 
        ? 'bg-red-200 border-2 border-red-400' 
        : 'bg-yellow-200 border border-yellow-400'
      
      result += `<mark class="${highlightClass} rounded px-1 cursor-pointer" data-span-start="${span.start}" data-clause-id="${span.clauseId || ''}" onclick="window.handleSpanClick('${span.clauseId || ''}')">${spanText}</mark>`
      
      lastEnd = span.end
    })

    // Add remaining text
    const remainingText = documentText.slice(lastEnd)
    if (searchTerm) {
      const regex = new RegExp(`(${searchTerm})`, 'gi')
      result += remainingText.replace(regex, '<mark class="bg-blue-200">$1</mark>')
    } else {
      result += remainingText
    }
    
    return result
  }

  const handleTextSelection = () => {
    const selection = window.getSelection()
    if (selection && selection.toString().trim()) {
      const selectedText = selection.toString()
      setSelectedText(selectedText)
      setShowQueryDialog(true)
    }
  }

  // Make span click handler available globally
  useEffect(() => {
    (window as any).handleSpanClick = (clauseId: string) => {
      if (clauseId) {
        setSelectedClauseId(clauseId)
      }
    }
    
    return () => {
      delete (window as any).handleSpanClick
    }
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="loading-spinner mx-auto mb-4"></div>
          <p className="text-gray-600">Analyzing document...</p>
        </div>
      </div>
    )
  }

  if (error || !analysis) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">Error Loading Analysis</h1>
          <p className="text-gray-600 mb-4">{error}</p>
          <button onClick={loadDocumentAnalysis} className="btn-primary">
            Try Again
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => window.history.back()}
                className="text-gray-500 hover:text-gray-700"
              >
                <ArrowLeftIcon className="h-6 w-6" />
              </button>
              
              <div>
                <h1 className="text-xl font-bold text-gray-900">{analysis.title}</h1>
                <div className="flex items-center space-x-4 mt-1 text-sm text-gray-600">
                  <span>{analysis.document_type.replace('_', ' ')}</span>
                  <span>•</span>
                  <span>{analysis.processing_metadata.pages_processed} pages</span>
                  <span>•</span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    analysis.overall_risk_score >= 7 ? 'bg-red-100 text-red-800' :
                    analysis.overall_risk_score >= 4 ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    Risk: {analysis.overall_risk_score}/10
                  </span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <button
                onClick={() => setShowChatbot(true)}
                className="flex items-center space-x-2 px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                <ChatBubbleLeftEllipsisIcon className="h-4 w-4" />
                <span>Ask Question</span>
              </button>
              
              <button className="p-2 text-gray-500 hover:text-gray-700">
                <ShareIcon className="h-5 w-5" />
              </button>
              
              <button className="p-2 text-gray-500 hover:text-gray-700">
                <PrinterIcon className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* High Risk Alert */}
        {analysis.clause_cards.some(c => c.risk_level === 'HIGH') && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <ExclamationTriangleIcon className="h-6 w-6 text-red-500 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-medium text-red-900 mb-1">
                  High-Risk Clauses Detected
                </h3>
                <p className="text-red-800 text-sm">
                  This document contains {analysis.clause_cards.filter(c => c.risk_level === 'HIGH').length} high-risk clause(s) highlighted in red. Review these carefully before signing.
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left: Clause Navigator */}
          <div className="lg:col-span-1">
            <div className="sticky top-24">
              <ClauseNavigator
                clauses={analysis.clause_cards.map(clause => ({
                  id: clause.clause_id,
                  original_text: clause.original_text,
                  simplified_text: clause.simplified_summary,
                  risk_level: clause.risk_level,
                  source_spans: clause.source_spans,
                  recommendations: clause.recommendations?.map(r => r.description) || [],
                  negotiation_points: clause.negotiation_points || []
                }))}
                selectedClauseId={selectedClauseId}
                onClauseSelect={handleClauseSelect}
                documentText=""
              />
            </div>
          </div>

          {/* Right: Document Viewer */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg border border-gray-200">
              {/* Document Header */}
              <div className="flex items-center justify-between p-4 border-b">
                <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                  <DocumentTextIcon className="h-5 w-5 mr-2" />
                  Document Preview
                </h3>
                
                <div className="flex items-center space-x-3">
                  <div className="relative">
                    <MagnifyingGlassIcon className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search in document..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-9 pr-3 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </div>

              {/* Legend */}
              <div className="px-4 py-3 bg-gray-50 border-b">
                <div className="flex items-center space-x-6 text-sm">
                  <div className="flex items-center space-x-2">
                    <span className="w-4 h-4 bg-yellow-200 border border-yellow-400 rounded"></span>
                    <span className="text-gray-700">Important clauses</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="w-4 h-4 bg-red-200 border-2 border-red-400 rounded"></span>
                    <span className="text-gray-700">High-risk clauses</span>
                  </div>
                  {searchTerm && (
                    <div className="flex items-center space-x-2">
                      <span className="w-4 h-4 bg-blue-200 rounded"></span>
                      <span className="text-gray-700">Search matches</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Document Content */}
              <div className="p-6 max-h-96 overflow-y-auto">
                <div 
                  className="prose prose-sm max-w-none leading-relaxed select-text whitespace-pre-wrap"
                  onMouseUp={handleTextSelection}
                  dangerouslySetInnerHTML={{
                    __html: createHighlightedText()
                  }}
                />
              </div>

              {/* Instructions */}
              <div className="px-4 py-3 bg-gray-50 border-t">
                <p className="text-sm text-gray-600 text-center">
                  💡 Click highlighted sections to view analysis • Select text to ask questions • {highlightedSpans.length} clause{highlightedSpans.length !== 1 ? 's' : ''} highlighted
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Chatbot Modal */}
      {showChatbot && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg w-full max-w-4xl h-96 m-4">
            <Chatbot
              documentIds={[docId]}
              focusDocId={docId}
              onClose={() => setShowChatbot(false)}
            />
          </div>
        </div>
      )}

      {/* Text Query Dialog */}
      {showQueryDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">
              Ask about selected text
            </h4>
            
            <div className="mb-4">
              <div className="text-sm text-gray-600 mb-2">Selected text:</div>
              <div className="bg-gray-100 p-3 rounded text-sm italic max-h-20 overflow-y-auto">
                "{selectedText}"
              </div>
            </div>

            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowQueryDialog(false)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowQueryDialog(false)
                  setShowChatbot(true)
                }}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Ask AI Assistant
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Mock data functions (same as before)
function getMockAnalysis(docId: string): DocumentAnalysis {
  return {
    doc_id: docId,
    title: "Sample Rental Agreement",
    document_type: "rental_agreement",
    processing_status: "completed",
    immediate_facts: [
      {
        fact_type: "amount",
        value: { amount: 2500, currency: "USD", period: "monthly" },
        confidence: 0.95,
        source_spans: [{ start: 450, end: 470 }],
        extraction_method: "deterministic"
      }
    ],
    clause_cards: [
      {
        clause_id: "early-termination",
        original_text: "Tenant may terminate this lease with 60 days written notice and payment of two months rent as penalty.",
        simplified_summary: "You can end the lease early but must give 60 days notice and pay 2 months rent as a penalty (total cost: $5,000).",
        risk_level: "HIGH",
        source_spans: [{ start: 1200, end: 1310 }],
        risk_flags: [],
        recommendations: [
          {
            type: "negotiate",
            priority: "HIGH",
            description: "Request reduction of early termination penalty from 2 months to 1 month rent",
            rationale: "Standard market rate is typically 1 month or less"
          }
        ],
        negotiation_points: [
          "Request lower termination penalty (1 month vs 2 months)",
          "Ask for graduated penalty based on lease duration remaining"
        ]
      }
    ],
    overall_risk_score: 7.2,
    processing_metadata: {
      pages_processed: 12,
      confidence_threshold: 0.85,
      verification_status: "verified",
      human_review_required: true
    }
  }
}

function getMockDocumentText(): string {
  return `RESIDENTIAL LEASE AGREEMENT

This Lease Agreement ("Agreement") is entered into on January 1, 2024 between Acme Properties LLC ("Landlord") and John Doe ("Tenant") for the rental of property located at 123 Main Street, Apartment 2B, Anytown, ST 12345.

TERMS AND CONDITIONS:

1. RENT: Tenant agrees to pay monthly rent of $2,500.00 due on the first day of each month. Late payments will incur a fee of $75.00 after a 5-day grace period.

2. SECURITY DEPOSIT: Tenant will pay a security deposit of $2,500.00 prior to occupancy. This deposit will be held to cover any damages beyond normal wear and tear.

3. LEASE TERM: This lease shall commence on January 1, 2024 and terminate on December 31, 2024, unless terminated earlier in accordance with the provisions herein.

4. EARLY TERMINATION: Tenant may terminate this lease with 60 days written notice and payment of two months rent as penalty. This penalty shall be in addition to any other amounts owed under this agreement.

5. AUTOMATIC RENEWAL: This lease will automatically renew for successive one-year periods unless either party provides written notice of non-renewal at least 30 days before the current term expires.

6. UTILITIES: Tenant is responsible for all utilities including electricity, gas, water, sewer, trash, and internet services. Estimated monthly cost is $200-300.

7. PETS: No pets are allowed on the premises without prior written consent from Landlord. If approved, a pet deposit of $500 per pet is required.

8. MAINTENANCE: Tenant is responsible for routine maintenance and minor repairs under $100. Landlord is responsible for major repairs and structural issues.

9. LIABILITY: Tenant agrees to indemnify and hold harmless Landlord from any claims, damages, or injuries occurring on the premises, regardless of fault or negligence.

10. GOVERNING LAW: This agreement shall be governed by the laws of the State of [STATE].

By signing below, both parties agree to the terms and conditions set forth in this Agreement.

Landlord: _________________ Date: _______
Tenant: _________________ Date: _______`
}