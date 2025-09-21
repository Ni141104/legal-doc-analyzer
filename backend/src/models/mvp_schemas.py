"""
MVP Data Models and Schemas for Legal Document Analyzer
Simplified schemas for hackathon prototype with no authentication.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ClauseType(str, Enum):
    """Legal clause types for classification."""
    GENERAL = "general"
    PAYMENT = "payment"
    TERMINATION = "termination"
    LIABILITY = "liability"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    CONFIDENTIALITY = "confidentiality"
    DISPUTE_RESOLUTION = "dispute_resolution"
    FORCE_MAJEURE = "force_majeure"
    GOVERNING_LAW = "governing_law"
    AMENDMENT = "amendment"
    SEVERABILITY = "severability"
    ENTIRE_AGREEMENT = "entire_agreement"


# Core Document Models

class DocumentMetadata(BaseModel):
    """Document metadata for MVP."""
    doc_id: str
    filename: str
    content_type: str
    file_size_bytes: int
    status: ProcessingStatus
    gcs_uri: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Processing statistics
    total_pages: Optional[int] = None
    total_clauses: Optional[int] = None
    clause_types: List[ClauseType] = Field(default_factory=list)


class ExtractedClause(BaseModel):
    """Extracted and classified legal clause."""
    clause_id: str
    doc_id: str
    text: str
    clause_type: ClauseType
    confidence_score: float = Field(ge=0.0, le=1.0)
    page_number: Optional[int] = None
    bounding_box: Optional[Dict[str, float]] = None
    created_at: datetime
    
    # Vector search metadata
    embedding_vector: Optional[List[float]] = None
    keywords: List[str] = Field(default_factory=list)


# Search and Retrieval Models

class VectorSearchRequest(BaseModel):
    """Vector search request."""
    query: str = Field(min_length=1, max_length=1000)
    doc_id: Optional[str] = None
    clause_type: Optional[ClauseType] = None
    top_k: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class VectorMetadata(BaseModel):
    """Vector search result metadata."""
    search_method: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    hyde_score: Optional[float] = None
    cross_encoder_score: Optional[float] = None


class VectorSearchResult(BaseModel):
    """Vector search result."""
    clause_id: str
    doc_id: str
    text: str
    clause_type: ClauseType
    similarity_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorSearchResponse(BaseModel):
    """Vector search response."""
    results: List[VectorSearchResult]
    query_time: float = Field(ge=0.0)


# RAG and Generation Models

class HyDEDocument(BaseModel):
    """Hypothetical Document Embeddings (HyDE) result."""
    original_query: str
    hypothetical_document: str
    embedding_vector: List[float]
    generated_at: datetime


class RAGAnswer(BaseModel):
    """RAG-generated answer with verification."""
    question: str
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    source_clause_ids: List[str]
    reasoning: Optional[str] = None
    verification_passed: bool = False
    generated_at: datetime


# API Request/Response Models

class DocumentUploadResponse(BaseModel):
    """Document upload response."""
    doc_id: str
    filename: str
    status: ProcessingStatus
    message: str
    created_at: datetime


class QueryRequest(BaseModel):
    """Document query request."""
    question: str = Field(min_length=1, max_length=1000)
    clause_type: Optional[ClauseType] = None
    max_results: Optional[int] = Field(default=8, ge=1, le=20)
    use_hyde: bool = Field(default=True)
    use_cross_encoder: bool = Field(default=True)


class QueryResponse(BaseModel):
    """Document query response."""
    doc_id: str
    question: str
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    retrieved_clauses: List[Dict[str, Any]]
    search_metadata: Dict[str, Any] = Field(default_factory=dict)


# Enhanced Search Models for Hybrid Search

class SparseVectorResult(BaseModel):
    """Sparse vector search result."""
    clause_id: str
    score: float
    term_frequencies: Dict[str, int]


class DenseVectorResult(BaseModel):
    """Dense vector search result."""
    clause_id: str
    score: float
    embedding_similarity: float


class CrossEncoderResult(BaseModel):
    """Cross-encoder reranking result."""
    clause_id: str
    score: float
    relevance_score: float


class HybridSearchResult(BaseModel):
    """Combined hybrid search result."""
    clause_id: str
    final_score: float
    component_scores: Dict[str, float]
    ranking_method: str


# Error Models

class ErrorResponse(BaseModel):
    """API error response."""
    error: str
    status_code: int
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


# Configuration Models

class SearchConfiguration(BaseModel):
    """Search configuration settings."""
    hybrid_search_enabled: bool = True
    hyde_enabled: bool = True
    cross_encoder_enabled: bool = True
    search_weights: Dict[str, float]
    models: Dict[str, str]