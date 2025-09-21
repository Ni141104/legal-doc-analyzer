"""
Core data models for the Legal Document Analyzer.
Implements the schemas defined in the architecture specification.
Production-ready models for multi-agent Gen AI system.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import uuid


class RiskLevel(str, Enum):
    """Risk assessment levels for clause analysis."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ClauseType(str, Enum):
    """Standard clause types in legal documents."""
    PAYMENT = "Payment"
    DEPOSIT = "Deposit"
    TERMINATION = "Termination"
    LIABILITY = "Liability"
    CONFIDENTIALITY = "Confidentiality"
    INTELLECTUAL_PROPERTY = "IntellectualProperty"
    DISPUTE_RESOLUTION = "DisputeResolution"
    GOVERNING_LAW = "GoverningLaw"
    FORCE_MAJEURE = "ForceMajeure"
    AMENDMENT = "Amendment"
    OTHER = "Other"


class ReviewStatus(str, Enum):
    """Human review workflow statuses."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVISION = "requires_revision"


class ConfidenceLevel(str, Enum):
    """Confidence levels for generated content."""
    VERY_HIGH = "very_high"  # > 0.95
    HIGH = "high"            # 0.85 - 0.95
    MEDIUM = "medium"        # 0.70 - 0.85
    LOW = "low"              # 0.50 - 0.70


class ProcessingStatus(str, Enum):
    """Document processing pipeline statuses."""
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


# Core Entity Models

class SourceSpan(BaseModel):
    """Reference to specific location in source document."""
    span_id: str = Field(..., description="Unique identifier for the span")
    page: int = Field(..., description="Page number (1-indexed)")
    line_start: int = Field(..., description="Starting line number")
    line_end: int = Field(..., description="Ending line number")
    char_start: Optional[int] = Field(None, description="Starting character offset")
    char_end: Optional[int] = Field(None, description="Ending character offset")
    bbox: Optional[Dict[str, float]] = Field(None, description="Bounding box coordinates")


class SimplifiedSentence(BaseModel):
    """A simplified sentence with provenance and confidence."""
    text: str = Field(..., description="Simplified, human-readable sentence")
    source_spans: List[str] = Field(..., description="List of source span IDs supporting this sentence")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this sentence")
    rationale: str = Field(..., description="Explanation of why this simplification is accurate")
    verified: bool = Field(default=False, description="Whether human-verified")


class RiskFlag(BaseModel):
    """Identified risk factor in a clause."""
    type: str = Field(..., description="Type of risk (e.g., 'DepositHigh', 'TerminationUnfair')")
    level: RiskLevel = Field(..., description="Risk severity level")
    explanation: str = Field(..., description="Human-readable explanation of the risk")
    evidence_spans: List[str] = Field(..., description="Source spans supporting this risk assessment")
    mitigation: Optional[str] = Field(None, description="Suggested mitigation strategy")


class Recommendation(BaseModel):
    """Actionable recommendation for contract improvement."""
    text: str = Field(..., description="Recommendation description")
    actionable_redline: Optional[str] = Field(None, description="Specific text changes to make")
    priority: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Priority level")
    category: str = Field(..., description="Category of recommendation")


class NormalizedTerms(BaseModel):
    """Extracted and normalized key terms from clause."""
    amounts: Dict[str, float] = Field(default_factory=dict, description="Monetary amounts (currency normalized)")
    dates: Dict[str, str] = Field(default_factory=dict, description="Important dates (ISO format)")
    parties: Dict[str, str] = Field(default_factory=dict, description="Named parties and their roles")
    durations: Dict[str, int] = Field(default_factory=dict, description="Time periods in days")
    percentages: Dict[str, float] = Field(default_factory=dict, description="Percentage values")


class ClauseCard(BaseModel):
    """Main output schema for clause analysis - enforced JSON structure."""
    clause_id: str = Field(..., description="Unique clause identifier")
    simplified_sentences: List[SimplifiedSentence] = Field(..., description="Human-readable simplified sentences")
    normalized_terms: NormalizedTerms = Field(..., description="Extracted key terms and values")
    risk_flags: List[RiskFlag] = Field(default_factory=list, description="Identified risk factors")
    recommendations: List[Recommendation] = Field(default_factory=list, description="Actionable recommendations")
    clause_type: ClauseType = Field(..., description="Categorized clause type")
    confidence_overall: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in analysis")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    model_version: str = Field(..., description="Version of models used for generation")


# Document Models

class ExtractedFacts(BaseModel):
    """Deterministically extracted facts from document."""
    amounts: Dict[str, float] = Field(default_factory=dict, description="All monetary amounts found")
    dates: Dict[str, str] = Field(default_factory=dict, description="All dates found (ISO format)")
    parties: List[str] = Field(default_factory=list, description="All parties/entities mentioned")
    jurisdictions: List[str] = Field(default_factory=list, description="Legal jurisdictions mentioned")
    document_type: Optional[str] = Field(None, description="Detected document type")
    language: str = Field(default="en", description="Primary document language")
    page_count: int = Field(..., description="Total number of pages")


class ClauseMetadata(BaseModel):
    """Metadata for a single clause in the document."""
    clause_id: str = Field(..., description="Unique clause identifier")
    doc_id: str = Field(..., description="Parent document ID")
    section_title: Optional[str] = Field(None, description="Section or clause title")
    content_text: str = Field(..., description="Full clause text content")
    page: int = Field(..., description="Page number where clause appears")
    line_start: int = Field(..., description="Starting line number")
    line_end: int = Field(..., description="Ending line number")
    clause_type: ClauseType = Field(default=ClauseType.OTHER, description="Categorized clause type")
    template_fingerprint: Optional[str] = Field(None, description="Hash of similar template clauses")
    extracted_terms: NormalizedTerms = Field(default_factory=NormalizedTerms, description="Deterministically extracted terms")
    embedding_id: Optional[str] = Field(None, description="Vector embedding reference")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class DocumentMetadata(BaseModel):
    """Complete metadata for a processed document."""
    doc_id: str = Field(..., description="Unique document identifier")
    original_filename: str = Field(..., description="Original uploaded filename")
    storage_path: str = Field(..., description="Cloud Storage path")
    content_type: str = Field(..., description="MIME type")
    file_size: int = Field(..., description="File size in bytes")
    checksum: str = Field(..., description="File content hash")
    processing_status: ProcessingStatus = Field(..., description="Current processing status")
    extracted_facts: ExtractedFacts = Field(..., description="Deterministically extracted facts")
    total_clauses: int = Field(default=0, description="Total number of clauses identified")
    jurisdiction: Optional[str] = Field(None, description="Primary jurisdiction")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")
    processing_completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    user_id: str = Field(..., description="User who uploaded the document")
    tenant_id: Optional[str] = Field(None, description="Tenant isolation identifier")
    retention_policy: Optional[str] = Field(None, description="Data retention policy applied")


# API Request/Response Models

class DocumentUploadRequest(BaseModel):
    """Request for document upload."""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    ephemeral: bool = Field(default=False, description="Whether to process without persistent storage")
    retention_days: Optional[int] = Field(None, description="Custom retention period")


class DocumentUploadResponse(BaseModel):
    """Response for successful document upload."""
    doc_id: str = Field(..., description="Unique document identifier")
    upload_url: Optional[str] = Field(None, description="Signed URL for direct upload")
    immediate_facts: ExtractedFacts = Field(..., description="Quick-extracted facts")
    processing_status: ProcessingStatus = Field(..., description="Current processing status")
    estimated_completion: Optional[datetime] = Field(None, description="ETA for processing completion")


class ClauseQueryRequest(BaseModel):
    """Request for querying clauses in a document."""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    clause_types: Optional[List[ClauseType]] = Field(None, description="Filter by clause types")
    include_low_confidence: bool = Field(default=False, description="Include low-confidence results")


class ClauseQueryResponse(BaseModel):
    """Response for clause queries with evidence-anchored answers."""
    query: str = Field(..., description="Original query")
    clause_cards: List[ClauseCard] = Field(..., description="Generated clause cards with evidence")
    retrieval_metadata: Dict[str, Any] = Field(..., description="Information about retrieval process")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    model_versions: Dict[str, str] = Field(..., description="Versions of all models used")


class VerificationRequest(BaseModel):
    """Request to queue clause for human verification."""
    clause_id: str = Field(..., description="Clause to verify")
    priority: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Verification priority")
    reason: str = Field(..., description="Reason for human verification request")
    reviewer_notes: Optional[str] = Field(None, description="Additional context for reviewer")


class VerificationResponse(BaseModel):
    """Response for verification request."""
    review_id: str = Field(..., description="Unique review request identifier")
    status: ReviewStatus = Field(..., description="Current review status")
    estimated_completion: Optional[datetime] = Field(None, description="ETA for review completion")
    queue_position: Optional[int] = Field(None, description="Position in review queue")


class ReviewResult(BaseModel):
    """Result of human review."""
    review_id: str = Field(..., description="Review request identifier")
    clause_id: str = Field(..., description="Reviewed clause identifier")
    status: ReviewStatus = Field(..., description="Final review status")
    reviewer_id: str = Field(..., description="ID of reviewing attorney")
    original_card: ClauseCard = Field(..., description="Original generated clause card")
    revised_card: Optional[ClauseCard] = Field(None, description="Revised clause card if changes made")
    reviewer_notes: str = Field(..., description="Reviewer's comments and rationale")
    reviewed_at: datetime = Field(..., description="Review completion timestamp")
    approval_confidence: float = Field(..., ge=0.0, le=1.0, description="Reviewer's confidence in approval")


class ExportRequest(BaseModel):
    """Request for document export."""
    doc_id: str = Field(..., description="Document to export")
    format: str = Field(..., regex="^(json|pdf|docx)$", description="Export format")
    include_recommendations: bool = Field(default=True, description="Include recommendations in export")
    include_risk_analysis: bool = Field(default=True, description="Include risk analysis")
    redaction_level: str = Field(default="none", regex="^(none|partial|full)$", description="PII redaction level")


class ExportResponse(BaseModel):
    """Response for document export."""
    export_id: str = Field(..., description="Unique export identifier")
    download_url: str = Field(..., description="Signed URL for download")
    expires_at: datetime = Field(..., description="URL expiration time")
    file_size: int = Field(..., description="Export file size in bytes")
    export_format: str = Field(..., description="Actual export format")


# System Models

class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="API version")
    dependencies: Dict[str, str] = Field(..., description="Status of external dependencies")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: str = Field(..., description="Unique request identifier for tracking")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


# Internal Processing Models

class EmbeddingRequest(BaseModel):
    """Request for text embedding."""
    text: str = Field(..., description="Text to embed")
    embedding_type: str = Field(default="content", description="Type of embedding (content, label, alias)")


class RetrievalCandidate(BaseModel):
    """Candidate clause from retrieval system."""
    clause_id: str = Field(..., description="Clause identifier")
    score: float = Field(..., description="Composite relevance score")
    sparse_score: float = Field(..., description="BM25 sparse retrieval score")
    dense_score: float = Field(..., description="Dense vector similarity score")
    label_score: float = Field(..., description="Label-based similarity score")
    content_text: str = Field(..., description="Clause content")
    metadata: Dict[str, Any] = Field(..., description="Additional clause metadata")


class HyDERequest(BaseModel):
    """Request for HyDE pseudo-document generation."""
    query: str = Field(..., description="User query")
    context: Optional[str] = Field(None, description="Additional context")


class GenerationRequest(BaseModel):
    """Request for clause card generation."""
    clause_metadata: ClauseMetadata = Field(..., description="Source clause information")
    evidence_spans: List[Dict[str, Any]] = Field(..., description="Supporting evidence spans")
    query_context: Optional[str] = Field(None, description="Original user query for context")


class VerificationRequest(BaseModel):
    """Request for NLI verification."""
    generated_sentence: str = Field(..., description="Generated sentence to verify")
    evidence_spans: List[str] = Field(..., description="Supporting evidence span texts")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence threshold")


# Configuration Models

class ModelConfig(BaseModel):
    """Configuration for AI models."""
    embedding_model: str = Field(default="textembedding-gecko", description="Vertex AI embedding model")
    generation_model: str = Field(default="gemini-pro", description="Vertex AI generation model")
    verification_model: str = Field(default="textembedding-gecko", description="NLI verification model")
    temperature: float = Field(default=0.1, description="Generation temperature")
    max_tokens: int = Field(default=2048, description="Maximum generation tokens")


class RetrievalConfig(BaseModel):
    """Configuration for retrieval system."""
    sparse_weight: float = Field(default=0.35, description="Weight for sparse (BM25) retrieval")
    dense_weight: float = Field(default=0.40, description="Weight for dense vector retrieval")
    label_weight: float = Field(default=0.15, description="Weight for label-based retrieval")
    jurisdiction_weight: float = Field(default=0.05, description="Weight for jurisdiction matching")
    template_weight: float = Field(default=0.05, description="Weight for template similarity")
    top_k_candidates: int = Field(default=50, description="Initial candidate pool size")
    rerank_top_k: int = Field(default=20, description="Candidates to rerank")
    final_top_n: int = Field(default=8, description="Final results to return to generator")
    VERY_LOW = "very_low"    # < 0.50


class SourceSpan(BaseModel):
    """Reference to a specific location in the source document."""
    span_id: str = Field(..., description="Unique identifier for the span")
    doc_id: str = Field(..., description="Document identifier")
    page: int = Field(..., ge=1, description="Page number (1-indexed)")
    line_start: int = Field(..., ge=1, description="Starting line number")
    line_end: int = Field(..., ge=1, description="Ending line number")
    char_start: Optional[int] = Field(None, description="Character start position")
    char_end: Optional[int] = Field(None, description="Character end position")
    
    @validator('line_end')
    def line_end_must_be_gte_start(cls, v, values):
        if 'line_start' in values and v < values['line_start']:
            raise ValueError('line_end must be >= line_start')
        return v


class ExtractedFact(BaseModel):
    """Deterministically extracted fact from document."""
    fact_type: str = Field(..., description="Type of extracted fact")
    value: Union[str, int, float] = Field(..., description="Extracted value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    source_span: SourceSpan = Field(..., description="Source location")
    extraction_method: str = Field(..., description="Method used for extraction")
    normalized_value: Optional[Any] = Field(None, description="Normalized representation")


class ClauseRecord(BaseModel):
    """Database record for a parsed clause."""
    clause_id: str = Field(..., description="Unique clause identifier")
    doc_id: str = Field(..., description="Parent document identifier")
    section_title: Optional[str] = Field(None, description="Section heading")
    clause_type: ClauseType = Field(ClauseType.OTHER, description="Categorized clause type")
    content_text: str = Field(..., description="Full clause text")
    page: int = Field(..., ge=1, description="Page number")
    line_start: int = Field(..., ge=1, description="Starting line")
    line_end: int = Field(..., ge=1, description="Ending line")
    
    # Extracted structured data
    extracted_facts: List[ExtractedFact] = Field(default_factory=list)
    amount_deposit: Optional[float] = Field(None, description="Security deposit amount")
    amount_rent: Optional[float] = Field(None, description="Rent amount")
    currency: Optional[str] = Field(None, description="Currency code")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction")
    
    # Document metadata
    template_fingerprint: Optional[str] = Field(None, description="Template hash for similar docs")
    language: str = Field("en", description="Document language")
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Vector embeddings (stored separately but referenced here)
    content_vector_id: Optional[str] = Field(None)
    label_vector_id: Optional[str] = Field(None)
    alias_vector_ids: List[str] = Field(default_factory=list)


class SimplifiedSentence(BaseModel):
    """A simplified, human-readable sentence with provenance."""
    text: str = Field(..., description="Simplified sentence text")
    source_spans: List[str] = Field(..., description="Source span IDs supporting this sentence")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Generation confidence")
    confidence_level: ConfidenceLevel = Field(..., description="Categorized confidence")
    rationale: str = Field(..., description="Explanation of how sentence was derived")
    verified: bool = Field(False, description="Whether sentence passed NLI verification")
    verifier_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @validator('confidence_level', pre=True, always=True)
    def set_confidence_level(cls, v, values):
        if 'confidence' not in values:
            return v
        
        conf = values['confidence']
        if conf >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif conf >= 0.85:
            return ConfidenceLevel.HIGH
        elif conf >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif conf >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class RiskFlag(BaseModel):
    """Identified risk in a clause."""
    flag_type: str = Field(..., description="Type of risk identified")
    level: RiskLevel = Field(..., description="Severity level")
    explanation: str = Field(..., description="Human-readable explanation")
    evidence_spans: List[str] = Field(..., description="Supporting span IDs")
    actionable: bool = Field(True, description="Whether this risk can be addressed")
    statutory_reference: Optional[str] = Field(None, description="Relevant law/statute")


class Recommendation(BaseModel):
    """Actionable recommendation for document improvement."""
    text: str = Field(..., description="Recommendation description")
    priority: RiskLevel = Field(RiskLevel.MEDIUM, description="Implementation priority")
    actionable_redline: Optional[str] = Field(None, description="Specific text changes")
    estimated_impact: Optional[str] = Field(None, description="Expected outcome")
    legal_basis: Optional[str] = Field(None, description="Legal justification")


class ClauseCard(BaseModel):
    """Complete analysis card for a clause - main output format."""
    clause_id: str = Field(..., description="Source clause identifier")
    simplified_sentences: List[SimplifiedSentence] = Field(..., description="Human-readable explanations")
    normalized_terms: Dict[str, Union[str, int, float]] = Field(
        default_factory=dict, 
        description="Key-value pairs of normalized clause terms"
    )
    risk_flags: List[RiskFlag] = Field(default_factory=list, description="Identified risks")
    recommendations: List[Recommendation] = Field(default_factory=list, description="Improvement suggestions")
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = Field(..., description="Generator model version")
    retrieval_set_ids: List[str] = Field(..., description="Retrieved context span IDs")
    generator_prompt_hash: str = Field(..., description="Hash of generation prompt")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregate confidence score")
    
    @validator('overall_confidence', pre=True, always=True)
    def calculate_overall_confidence(cls, v, values):
        if 'simplified_sentences' not in values:
            return v or 0.0
        
        sentences = values['simplified_sentences']
        if not sentences:
            return 0.0
        
        # Weight by verification status and confidence
        total_weight = 0
        weighted_sum = 0
        
        for sentence in sentences:
            weight = 1.0 if sentence.verified else 0.5
            weighted_sum += sentence.confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class DocumentMetadata(BaseModel):
    """Metadata for uploaded documents."""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    mime_type: str = Field(..., description="MIME type")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_status: str = Field("uploaded", description="Processing pipeline status")
    
    # Document classification
    document_type: Optional[str] = Field(None, description="Detected document type")
    language: str = Field("en", description="Primary language")
    page_count: Optional[int] = Field(None, ge=1)
    
    # Security & privacy
    ephemeral: bool = Field(False, description="Whether to delete after processing")
    encrypted: bool = Field(True, description="Whether stored data is encrypted")
    retention_days: Optional[int] = Field(None, ge=1, description="Auto-deletion period")
    
    # Processing metadata
    ocr_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    clause_count: Optional[int] = Field(None, ge=0)
    extracted_facts_count: Optional[int] = Field(None, ge=0)


class QueryRequest(BaseModel):
    """User query against a document."""
    query: str = Field(..., min_length=1, description="Natural language query")
    doc_id: str = Field(..., description="Target document ID")
    top_k: int = Field(8, ge=1, le=20, description="Number of results to return")
    include_low_confidence: bool = Field(False, description="Include low-confidence results")
    jurisdiction_filter: Optional[str] = Field(None, description="Filter by jurisdiction")
    clause_type_filter: Optional[ClauseType] = Field(None, description="Filter by clause type")


class QueryResponse(BaseModel):
    """Response to user query."""
    query: str = Field(..., description="Original query")
    doc_id: str = Field(..., description="Document ID")
    clause_cards: List[ClauseCard] = Field(..., description="Matching clause analyses")
    processing_time_ms: float = Field(..., ge=0, description="Total processing time")
    
    # Retrieval metadata
    total_candidates: int = Field(..., ge=0, description="Total candidates considered")
    retrieval_method: str = Field(..., description="Primary retrieval method used")
    hybrid_scores: Optional[Dict[str, float]] = Field(None, description="Component scores breakdown")


class HumanReviewRequest(BaseModel):
    """Request for human attorney review."""
    clause_id: str = Field(..., description="Clause requiring review")
    review_type: str = Field(..., description="Type of review needed")
    priority: RiskLevel = Field(RiskLevel.MEDIUM, description="Review priority")
    context: Optional[str] = Field(None, description="Additional context for reviewer")
    requester_id: str = Field(..., description="ID of requesting user")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class HumanReviewResponse(BaseModel):
    """Attorney review result."""
    review_id: str = Field(..., description="Review request ID")
    reviewer_id: str = Field(..., description="Attorney ID")
    status: ReviewStatus = Field(..., description="Review outcome")
    reviewed_clause_card: Optional[ClauseCard] = Field(None, description="Revised analysis")
    reviewer_notes: Optional[str] = Field(None, description="Attorney comments")
    confidence_override: Optional[float] = Field(None, ge=0.0, le=1.0)
    approved_for_use: bool = Field(False, description="Safe for end-user consumption")
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)


class ExportRequest(BaseModel):
    """Request for document export/negotiation pack."""
    doc_id: str = Field(..., description="Document to export")
    export_format: str = Field("pdf", description="Output format")
    include_recommendations: bool = Field(True, description="Include actionable recommendations")
    include_redlines: bool = Field(True, description="Include suggested text changes")
    target_audience: str = Field("client", description="Audience for export")


class SystemMetrics(BaseModel):
    """System performance and quality metrics."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Performance metrics
    avg_processing_time_ms: float = Field(..., ge=0)
    p95_processing_time_ms: float = Field(..., ge=0)
    throughput_docs_per_hour: float = Field(..., ge=0)
    
    # Quality metrics
    hallucination_rate: float = Field(..., ge=0.0, le=1.0, description="% unsupported claims")
    extractor_accuracy: float = Field(..., ge=0.0, le=1.0, description="Numeric field accuracy")
    retrieval_precision_at_5: float = Field(..., ge=0.0, le=1.0)
    verifier_pass_rate: float = Field(..., ge=0.0, le=1.0, description="% sentences passing NLI")
    
    # Volume metrics
    documents_processed: int = Field(..., ge=0)
    queries_executed: int = Field(..., ge=0)
    human_reviews_requested: int = Field(..., ge=0)
    clauses_verified: int = Field(..., ge=0)


# Response models for API endpoints
class DocumentUploadResponse(BaseModel):
    doc_id: str
    immediate_facts: List[ExtractedFact]
    processing_status: str
    estimated_completion_time: Optional[str] = None


class DocumentClausesResponse(BaseModel):
    doc_id: str
    clauses: List[ClauseRecord]
    total_count: int
    page: int = 1
    page_size: int = 50


class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(default_factory=dict)  # service -> status