export interface SourceSpan {
  start: number
  end: number
  confidence?: number
}

export interface SimplifiedSentence {
  text: string
  source_spans: SourceSpan[]
  confidence: number
}

export interface RiskFlag {
  flag_type: 'liability' | 'termination' | 'payment' | 'privacy' | 'dispute' | 'modification'
  severity: 'HIGH' | 'MEDIUM' | 'LOW'
  description: string
  source_spans: SourceSpan[]
  mitigation_advice: string
}

export interface Recommendation {
  type: 'negotiate' | 'clarify' | 'accept' | 'reject'
  priority: 'HIGH' | 'MEDIUM' | 'LOW'
  description: string
  rationale: string
  specific_language?: string
}

export interface ClauseCard {
  clause_id: string
  original_text: string
  simplified_summary: string
  risk_level: 'HIGH' | 'MEDIUM' | 'LOW'
  source_spans: SourceSpan[]
  risk_flags: RiskFlag[]
  recommendations: Recommendation[]
  negotiation_points: string[]
  legal_precedents?: string[]
}

export interface ExtractedFact {
  fact_type: 'party' | 'amount' | 'date' | 'duration' | 'obligation'
  value: any
  confidence: number
  source_spans: SourceSpan[]
  extraction_method: 'deterministic' | 'ml' | 'hybrid'
}

export interface DocumentAnalysis {
  doc_id: string
  title: string
  document_type: string
  processing_status: 'processing' | 'completed' | 'failed'
  immediate_facts: ExtractedFact[]
  clause_cards: ClauseCard[]
  overall_risk_score: number
  processing_metadata: {
    pages_processed: number
    confidence_threshold: number
    verification_status: string
    human_review_required: boolean
  }
}

export interface QueryResponse {
  query_id: string
  answer: string
  confidence: number
  source_references: Array<{
    clause_id: string
    relevance_score: number
    evidence_spans: SourceSpan[]
  }>
  verification_status: 'verified' | 'needs_review' | 'contradicted'
  followup_questions?: string[]
}