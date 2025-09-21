"""
Retrieval package for the Legal Document Analyzer.
"""

from .poly_vector import (
    PolyVectorRetriever,
    RetrievalCandidate,
    RetrievalResult,
    VectorIndex,
    PineconeIndex,
    ElasticsearchIndex,
    EmbeddingService,
    HyDEGenerator,
    CrossEncoderReranker,
    create_retriever
)

__all__ = [
    "PolyVectorRetriever",
    "RetrievalCandidate",
    "RetrievalResult", 
    "VectorIndex",
    "PineconeIndex",
    "ElasticsearchIndex",
    "EmbeddingService",
    "HyDEGenerator",
    "CrossEncoderReranker",
    "create_retriever"
]