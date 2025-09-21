'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import ClauseNavigator from '@/components/ClauseNavigator'
import DocumentViewer from '@/components/DocumentViewer'
import type { DocumentAnalysis, ClauseCard } from '@/types'

export default function DocumentAnalysisPage() {
  const params = useParams()
  const docId = params.id as string
  
  const [analysis, setAnalysis] = useState<DocumentAnalysis | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedClauseId, setSelectedClauseId] = useState<string | null>(null)
  const [highlightedSpans, setHighlightedSpans] = useState<Array<{ start: number; end: number }>>([])

  useEffect(() => {
    loadDocumentAnalysis()
  }, [docId])

  const loadDocumentAnalysis = async () => {
    try {
      setLoading(true)
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/v1/docs/${docId}/analysis`, {
        headers: {
          'Authorization': 'Bearer demo-token'
        }
      })
      
      if (!response.ok) {
        throw new Error('Failed to load analysis')
      }
      
      const data = await response.json()
      setAnalysis(data)
    } catch (err) {
      console.error('Analysis loading error:', err)
      setError('Failed to load document analysis')
      
      // For demo purposes, use mock data
      setAnalysis(getMockAnalysis(docId))
    } finally {
      setLoading(false)
    }
  }

  const handleClauseSelect = (clauseId: string) => {
    setSelectedClauseId(clauseId)
    
    // Find the clause and highlight its spans
    const clause = analysis?.clause_cards.find(c => c.clause_id === clauseId)
    if (clause?.source_spans) {
      setHighlightedSpans(clause.source_spans)
    }
  }

  const handleTextQuery = async (selectedText: string, question: string) => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/v1/docs/${docId}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer demo-token'
        },
        body: JSON.stringify({
          query: question,
          context: selectedText
        })
      })
      
      if (response.ok) {
        const result = await response.json()
        console.log('Query result:', result)
        // Handle the query response - could show in a modal or sidebar
      }
    } catch (error) {
      console.error('Query error:', error)
    }
  }

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
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">{analysis.title}</h1>
          <div className="flex items-center space-x-4 mt-2 text-sm text-gray-600">
            <span>Document Type: {analysis.document_type}</span>
            <span>•</span>
            <span>Pages: {analysis.processing_metadata.pages_processed}</span>
            <span>•</span>
            <span className={`px-2 py-1 rounded text-xs font-medium ${
              analysis.overall_risk_score >= 7 ? 'bg-red-100 text-red-800' :
              analysis.overall_risk_score >= 4 ? 'bg-yellow-100 text-yellow-800' :
              'bg-green-100 text-green-800'
            }`}>
              Risk Score: {analysis.overall_risk_score}/10
            </span>
          </div>
        </div>
        
        <div className="flex space-x-3">
          <button className="btn-secondary">
            📧 Share Analysis
          </button>
          <button className="btn-primary">
            📥 Export Report
          </button>
        </div>
      </div>

      {/* Processing Status */}
      {analysis.processing_metadata.human_review_required && (
        <div className="card bg-orange-50 border-orange-200">
          <div className="flex items-start space-x-3">
            <div className="w-2 h-2 bg-orange-400 rounded-full mt-2"></div>
            <div>
              <h3 className="font-medium text-orange-900 mb-1">Human Review Required</h3>
              <p className="text-orange-800 text-sm">
                Some clauses need expert review due to complexity or ambiguity. 
                Confidence threshold: {analysis.processing_metadata.confidence_threshold}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Left: Clause Navigator */}
        <div>
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
            documentText="" // Not needed in this context
          />
        </div>

        {/* Right: Document Viewer */}
        <div>
          <DocumentViewer
            documentText={getMockDocumentText()}
            highlightedSpans={highlightedSpans}
            onTextSelect={(text, start, end) => {
              console.log('Text selected:', text, 'at', start, '-', end)
            }}
          />
        </div>
      </div>
    </div>
  )
}

// Mock data for demo purposes
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
      },
      {
        fact_type: "party",
        value: { name: "John Doe", role: "tenant" },
        confidence: 0.92,
        source_spans: [{ start: 120, end: 135 }],
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
        risk_flags: [
          {
            flag_type: "termination",
            severity: "HIGH",
            description: "High penalty for early termination equivalent to 2 months rent",
            source_spans: [{ start: 1200, end: 1310 }],
            mitigation_advice: "Negotiate for a lower penalty or graduated penalty structure"
          }
        ],
        recommendations: [
          {
            type: "negotiate",
            priority: "HIGH",
            description: "Request reduction of early termination penalty from 2 months to 1 month rent",
            rationale: "Standard market rate is typically 1 month or less",
            specific_language: "Tenant may terminate with 60 days notice and payment of one month's rent"
          }
        ],
        negotiation_points: [
          "Request lower termination penalty (1 month vs 2 months)",
          "Ask for graduated penalty based on lease duration remaining",
          "Negotiate exceptions for job relocation or medical emergencies"
        ],
        legal_precedents: ["Most jurisdictions consider 2+ month penalties excessive"]
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

This Lease Agreement ("Agreement") is entered into on [DATE] between [LANDLORD NAME] ("Landlord") and John Doe ("Tenant") for the rental of property located at [PROPERTY ADDRESS].

TERMS AND CONDITIONS:

1. RENT: Tenant agrees to pay monthly rent of $2,500.00 due on the first day of each month.

2. SECURITY DEPOSIT: Tenant will pay a security deposit of $2,500.00 prior to occupancy.

3. LEASE TERM: This lease shall commence on [START DATE] and terminate on [END DATE], unless terminated earlier in accordance with the provisions herein.

4. EARLY TERMINATION: Tenant may terminate this lease with 60 days written notice and payment of two months rent as penalty. This penalty shall be in addition to any other amounts owed under this agreement.

5. UTILITIES: Tenant is responsible for all utilities including electricity, gas, water, sewer, trash, and internet services.

6. PETS: No pets are allowed on the premises without prior written consent from Landlord.

7. MAINTENANCE: Tenant is responsible for routine maintenance and minor repairs under $100. Landlord is responsible for major repairs and structural issues.

8. GOVERNING LAW: This agreement shall be governed by the laws of the State of [STATE].

By signing below, both parties agree to the terms and conditions set forth in this Agreement.

Landlord: _________________ Date: _______
Tenant: _________________ Date: _______`
}