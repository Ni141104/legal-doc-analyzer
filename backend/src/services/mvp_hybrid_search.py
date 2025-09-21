"""
Enhanced Hybrid Search Service for MVP
Combines dense vectors, sparse vectors, and cross-encoder reranking.
Uses Gemini 2.5 Pro and poly-vector indexing for superior retrieval.
"""

import asyncio
import logging
import uuid
import json
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Google Cloud / Vertex AI imports
try:
    from google.cloud import aiplatform
    from vertexai.language_models import TextEmbeddingModel
    from vertexai.generative_models import GenerativeModel
    import vertexai
except ImportError:
    aiplatform = None
    TextEmbeddingModel = None
    GenerativeModel = None
    vertexai = None

# Sentence transformers for cross-encoder reranking
try:
    from sentence_transformers import CrossEncoder
    import torch
except ImportError:
    CrossEncoder = None
    torch = None

from ..models.mvp_schemas import (
    VectorSearchResult, VectorSearchRequest, VectorSearchResponse, 
    VectorMetadata, ExtractedClause, ClauseType, HyDEDocument
)
from ..models.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SparseVector:
    """Sparse vector representation for BM25-style search."""
    indices: List[int]
    values: List[float]
    vocab_size: int


@dataclass
class PolyVectorResult:
    """Result from poly-vector search combining multiple retrieval methods."""
    clause_id: str
    dense_score: float
    sparse_score: float
    hyde_score: float
    cross_encoder_score: float
    final_score: float
    metadata: Dict[str, Any]


class HybridSearchService:
    """
    Advanced hybrid search combining multiple retrieval techniques:
    
    1. Dense Vector Search: Vertex AI embeddings + Matching Engine
    2. Sparse Vector Search: TF-IDF/BM25 style scoring
    3. HyDE Enhancement: Hypothetical Document Embeddings
    4. Cross-Encoder Reranking: Fine-tuned reranking model
    5. Poly-Vector Fusion: Weighted combination of all scores
    """
    
    def __init__(self):
        """Initialize hybrid search components."""
        self.embedding_model = None
        self.gemini_model = None
        self.cross_encoder = None
        self.matching_engine_client = None
        
        # Sparse search components
        self.vocabulary = {}  # Term to index mapping
        self.idf_scores = {}  # Inverse document frequency scores
        self.document_frequencies = {}  # Document frequency per term
        self.total_documents = 0
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all search components."""
        try:
            if vertexai:
                # Initialize Vertex AI
                vertexai.init(
                    project=settings.GOOGLE_CLOUD_PROJECT,
                    location=settings.VERTEX_AI_LOCATION
                )
                
                # Dense embeddings
                self.embedding_model = TextEmbeddingModel.from_pretrained(
                    settings.VERTEX_EMBEDDING_MODEL
                )
                
                # Gemini 2.5 Pro for HyDE and advanced reasoning
                self.gemini_model = GenerativeModel("gemini-2.5-pro")
                
                logger.info("Vertex AI models initialized (Gemini 2.5 Pro)")
            
            # Cross-encoder for reranking
            if CrossEncoder:
                # Use a legal domain cross-encoder if available, otherwise general
                model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self.cross_encoder = CrossEncoder(model_name)
                logger.info(f"Cross-encoder initialized: {model_name}")
            
            if aiplatform:
                # Matching Engine for dense vectors
                aiplatform.init(
                    project=settings.GOOGLE_CLOUD_PROJECT,
                    location=settings.VERTEX_AI_LOCATION
                )
                
                if settings.MATCHING_ENGINE_INDEX_ENDPOINT:
                    self.matching_engine_client = aiplatform.MatchingEngineIndexEndpoint(
                        index_endpoint_name=settings.MATCHING_ENGINE_INDEX_ENDPOINT
                    )
                    logger.info("Matching Engine client initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize hybrid search models: {str(e)}")
    
    async def generate_dense_embedding(self, text: str) -> List[float]:
        """Generate dense vector embedding using Vertex AI."""
        try:
            if not self.embedding_model:
                # Fallback random vector for development
                import random
                return [random.random() for _ in range(768)]
            
            clean_text = text.strip()[:8000]
            embeddings = self.embedding_model.get_embeddings([clean_text])
            return embeddings[0].values
            
        except Exception as e:
            logger.error(f"Dense embedding generation failed: {str(e)}")
            return [0.0] * 768
    
    def generate_sparse_vector(self, text: str) -> SparseVector:
        """Generate sparse vector using TF-IDF style approach."""
        try:
            # Tokenize and clean text
            tokens = self._tokenize(text.lower())
            
            # Calculate term frequencies
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            
            # Convert to sparse vector
            indices = []
            values = []
            
            for term, tf in term_freq.items():
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    
                    # TF-IDF calculation
                    tf_score = math.log(1 + tf)  # Log-normalized TF
                    idf_score = self.idf_scores.get(term, 1.0)
                    tfidf_score = tf_score * idf_score
                    
                    if tfidf_score > 0:
                        indices.append(term_idx)
                        values.append(tfidf_score)
            
            return SparseVector(
                indices=indices,
                values=values,
                vocab_size=len(self.vocabulary)
            )
            
        except Exception as e:
            logger.error(f"Sparse vector generation failed: {str(e)}")
            return SparseVector(indices=[], values=[], vocab_size=0)
    
    async def generate_hyde_embedding(self, query: str) -> Tuple[List[float], str]:
        """Generate HyDE (Hypothetical Document Embeddings) for query expansion."""
        try:
            if not self.gemini_model:
                # Fallback: use original query
                return await self.generate_dense_embedding(query), query
            
            # Generate hypothetical document with Gemini 2.5 Pro
            hyde_prompt = f"""
You are a legal expert. For this query about a legal document, generate a detailed hypothetical answer as if you were quoting directly from the relevant legal clauses.

Query: {query}

Generate a realistic legal text that would answer this query, including specific legal language, terms, conditions, and clauses that would typically appear in such documents.

Hypothetical Legal Text:"""
            
            response = self.gemini_model.generate_content(
                hyde_prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 500,
                    "top_p": 0.8
                }
            )
            
            hyde_text = response.text.strip()
            
            # Generate embedding for hypothetical document
            hyde_embedding = await self.generate_dense_embedding(hyde_text)
            
            logger.debug(f"Generated HyDE document for query: {query[:50]}...")
            return hyde_embedding, hyde_text
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {str(e)}")
            # Fallback to original query
            original_embedding = await self.generate_dense_embedding(query)
            return original_embedding, query
    
    async def hybrid_search(
        self,
        query: str,
        doc_id: Optional[str] = None,
        clause_type: Optional[ClauseType] = None,
        top_k: int = 20,
        use_hyde: bool = True,
        use_cross_encoder: bool = True
    ) -> VectorSearchResponse:
        """
        Perform hybrid search combining all retrieval methods.
        
        Pipeline:
        1. Dense vector search (original query + HyDE)
        2. Sparse vector search (BM25-style)
        3. Combine and rank candidates
        4. Cross-encoder reranking
        5. Final score fusion
        """
        try:
            start_time = datetime.utcnow()
            
            # Step 1: Dense vector search
            dense_results = await self._dense_vector_search(
                query, doc_id, clause_type, top_k * 2
            )
            
            # Step 2: HyDE-enhanced search
            hyde_results = []
            if use_hyde:
                hyde_embedding, hyde_text = await self.generate_hyde_embedding(query)
                hyde_results = await self._dense_vector_search_with_embedding(
                    hyde_embedding, doc_id, clause_type, top_k
                )
            
            # Step 3: Sparse vector search
            sparse_results = await self._sparse_vector_search(
                query, doc_id, clause_type, top_k
            )
            
            # Step 4: Combine all candidates
            all_candidates = self._combine_search_results(
                dense_results, hyde_results, sparse_results
            )
            
            # Step 5: Cross-encoder reranking
            if use_cross_encoder and self.cross_encoder:
                reranked_candidates = await self._cross_encoder_rerank(
                    query, all_candidates, top_k
                )
            else:
                # Simple score fusion without cross-encoder
                reranked_candidates = self._simple_score_fusion(all_candidates)[:top_k]
            
            # Step 6: Convert to final results
            final_results = self._convert_to_search_results(reranked_candidates)
            
            query_time = (datetime.utcnow() - start_time).total_seconds()
            
            response = VectorSearchResponse(
                results=final_results,
                query_time=query_time
            )
            
            logger.info(
                f"Hybrid search completed: {len(final_results)} results in {query_time:.3f}s "
                f"(dense: {len(dense_results)}, hyde: {len(hyde_results)}, "
                f"sparse: {len(sparse_results)})"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return VectorSearchResponse(results=[], query_time=0.0)
    
    async def _dense_vector_search(
        self,
        query: str,
        doc_id: Optional[str],
        clause_type: Optional[ClauseType],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform dense vector search using Matching Engine."""
        try:
            query_embedding = await self.generate_dense_embedding(query)
            return await self._dense_vector_search_with_embedding(
                query_embedding, doc_id, clause_type, top_k
            )
            
        except Exception as e:
            logger.error(f"Dense vector search failed: {str(e)}")
            return []
    
    async def _dense_vector_search_with_embedding(
        self,
        query_embedding: List[float],
        doc_id: Optional[str],
        clause_type: Optional[ClauseType],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform dense vector search with pre-computed embedding."""
        try:
            if not self.matching_engine_client:
                # Mock results for development
                return [
                    {
                        'clause_id': f'dense_clause_{i}',
                        'doc_id': doc_id or 'mock_doc',
                        'clause_type': clause_type.value if clause_type else 'general',
                        'text': f'Mock dense search result {i}',
                        'dense_score': 0.9 - (i * 0.1),
                        'metadata': {'search_type': 'dense'}
                    }
                    for i in range(min(top_k, 3))
                ]
            
            # In production: call Matching Engine API
            # response = self.matching_engine_client.find_neighbors(...)
            return []
            
        except Exception as e:
            logger.error(f"Dense vector search with embedding failed: {str(e)}")
            return []
    
    async def _sparse_vector_search(
        self,
        query: str,
        doc_id: Optional[str],
        clause_type: Optional[ClauseType],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform sparse vector search (BM25-style)."""
        try:
            # Generate sparse vector for query
            query_sparse = self.generate_sparse_vector(query)
            
            # Mock sparse search for development
            # In production: implement actual BM25/TF-IDF search
            return [
                {
                    'clause_id': f'sparse_clause_{i}',
                    'doc_id': doc_id or 'mock_doc',
                    'clause_type': clause_type.value if clause_type else 'general',
                    'text': f'Mock sparse search result {i}',
                    'sparse_score': 0.8 - (i * 0.15),
                    'metadata': {'search_type': 'sparse'}
                }
                for i in range(min(top_k, 2))
            ]
            
        except Exception as e:
            logger.error(f"Sparse vector search failed: {str(e)}")
            return []
    
    def _combine_search_results(
        self,
        dense_results: List[Dict[str, Any]],
        hyde_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]]
    ) -> List[PolyVectorResult]:
        """Combine results from different search methods."""
        try:
            # Collect all unique clauses
            clause_scores = {}
            
            # Process dense results
            for result in dense_results:
                clause_id = result['clause_id']
                if clause_id not in clause_scores:
                    clause_scores[clause_id] = PolyVectorResult(
                        clause_id=clause_id,
                        dense_score=0.0,
                        sparse_score=0.0,
                        hyde_score=0.0,
                        cross_encoder_score=0.0,
                        final_score=0.0,
                        metadata=result
                    )
                clause_scores[clause_id].dense_score = result.get('dense_score', 0.0)
            
            # Process HyDE results
            for result in hyde_results:
                clause_id = result['clause_id']
                if clause_id not in clause_scores:
                    clause_scores[clause_id] = PolyVectorResult(
                        clause_id=clause_id,
                        dense_score=0.0,
                        sparse_score=0.0,
                        hyde_score=0.0,
                        cross_encoder_score=0.0,
                        final_score=0.0,
                        metadata=result
                    )
                clause_scores[clause_id].hyde_score = result.get('dense_score', 0.0)
            
            # Process sparse results
            for result in sparse_results:
                clause_id = result['clause_id']
                if clause_id not in clause_scores:
                    clause_scores[clause_id] = PolyVectorResult(
                        clause_id=clause_id,
                        dense_score=0.0,
                        sparse_score=0.0,
                        hyde_score=0.0,
                        cross_encoder_score=0.0,
                        final_score=0.0,
                        metadata=result
                    )
                clause_scores[clause_id].sparse_score = result.get('sparse_score', 0.0)
            
            return list(clause_scores.values())
            
        except Exception as e:
            logger.error(f"Result combination failed: {str(e)}")
            return []
    
    async def _cross_encoder_rerank(
        self,
        query: str,
        candidates: List[PolyVectorResult],
        top_k: int
    ) -> List[PolyVectorResult]:
        """Rerank candidates using cross-encoder model."""
        try:
            if not self.cross_encoder or not candidates:
                return self._simple_score_fusion(candidates)[:top_k]
            
            # Prepare query-document pairs
            pairs = []
            for candidate in candidates:
                text = candidate.metadata.get('text', '')
                pairs.append([query, text])
            
            # Get cross-encoder scores
            if len(pairs) > 0:
                cross_scores = self.cross_encoder.predict(pairs)
                
                # Update candidates with cross-encoder scores
                for candidate, score in zip(candidates, cross_scores):
                    candidate.cross_encoder_score = float(score)
            
            # Final score fusion with cross-encoder
            for candidate in candidates:
                candidate.final_score = (
                    candidate.dense_score * 0.3 +
                    candidate.sparse_score * 0.2 +
                    candidate.hyde_score * 0.2 +
                    candidate.cross_encoder_score * 0.3
                )
            
            # Sort by final score
            reranked = sorted(candidates, key=lambda x: x.final_score, reverse=True)
            
            logger.debug(f"Cross-encoder reranked {len(candidates)} candidates")
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {str(e)}")
            return self._simple_score_fusion(candidates)[:top_k]
    
    def _simple_score_fusion(self, candidates: List[PolyVectorResult]) -> List[PolyVectorResult]:
        """Simple score fusion without cross-encoder."""
        try:
            for candidate in candidates:
                candidate.final_score = (
                    candidate.dense_score * 0.4 +
                    candidate.sparse_score * 0.3 +
                    candidate.hyde_score * 0.3
                )
            
            return sorted(candidates, key=lambda x: x.final_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Simple score fusion failed: {str(e)}")
            return candidates
    
    def _convert_to_search_results(
        self, 
        poly_results: List[PolyVectorResult]
    ) -> List[VectorSearchResult]:
        """Convert poly-vector results to standard search results."""
        try:
            search_results = []
            
            for poly_result in poly_results:
                metadata = poly_result.metadata
                
                search_result = VectorSearchResult(
                    clause_id=poly_result.clause_id,
                    doc_id=metadata.get('doc_id', 'unknown'),
                    text=metadata.get('text', ''),
                    clause_type=ClauseType(metadata.get('clause_type', 'general')),
                    similarity_score=poly_result.final_score,
                    metadata={
                        'dense_score': poly_result.dense_score,
                        'sparse_score': poly_result.sparse_score,
                        'hyde_score': poly_result.hyde_score,
                        'cross_encoder_score': poly_result.cross_encoder_score,
                        'search_method': 'hybrid_poly_vector'
                    }
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Result conversion failed: {str(e)}")
            return []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for sparse vectors."""
        import re
        # Remove special characters and split
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return tokens
    
    def update_vocabulary(self, documents: List[str]):
        """Update vocabulary and IDF scores from document corpus."""
        try:
            # Build vocabulary
            all_terms = set()
            doc_term_counts = []
            
            for doc in documents:
                terms = set(self._tokenize(doc.lower()))
                all_terms.update(terms)
                doc_term_counts.append(terms)
            
            # Create vocabulary mapping
            self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}
            
            # Calculate IDF scores
            self.total_documents = len(documents)
            self.document_frequencies = {}
            
            for terms in doc_term_counts:
                for term in terms:
                    self.document_frequencies[term] = self.document_frequencies.get(term, 0) + 1
            
            # Calculate IDF
            self.idf_scores = {}
            for term, df in self.document_frequencies.items():
                idf = math.log(self.total_documents / (1 + df))
                self.idf_scores[term] = idf
            
            logger.info(f"Updated vocabulary: {len(self.vocabulary)} terms, {self.total_documents} documents")
            
        except Exception as e:
            logger.error(f"Vocabulary update failed: {str(e)}")


# Global hybrid search service instance
hybrid_search_service = HybridSearchService()