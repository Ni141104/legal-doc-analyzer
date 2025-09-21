"""
Retrieval Service
Implements poly-vector hybrid retrieval with HyDE and cross-encoder reranking.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime

# Google Cloud imports (will be properly imported in production)
try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
except ImportError:
    # Fallback for development
    aiplatform = None

# Elasticsearch imports (will be properly imported in production)
try:
    from elasticsearch import AsyncElasticsearch
except ImportError:
    AsyncElasticsearch = None

from ..models.schemas import (
    ClauseMetadata, RetrievalCandidate, HyDERequest, 
    ClauseQueryRequest, ClauseQueryResponse
)
from ..models.config import settings
from .generation import generation_service

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for hybrid retrieval using dense vectors, sparse search, and HyDE."""
    
    def __init__(self):
        """Initialize the retrieval service."""
        self.embedding_client = None
        self.matching_engine_client = None
        self.elasticsearch_client = None
        self.cross_encoder = None
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all retrieval clients."""
        try:
            # Initialize Vertex AI for embeddings
            if aiplatform:
                aiplatform.init(
                    project=settings.GOOGLE_CLOUD_PROJECT,
                    location=settings.VERTEX_AI_LOCATION
                )
            
            # Initialize Elasticsearch
            if AsyncElasticsearch and settings.ELASTICSEARCH_HOST:
                es_config = {
                    'hosts': [f"{settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}"],
                    'use_ssl': settings.ELASTICSEARCH_USE_SSL,
                    'verify_certs': settings.ELASTICSEARCH_USE_SSL,
                }
                
                if settings.ELASTICSEARCH_USERNAME:
                    es_config['http_auth'] = (
                        settings.ELASTICSEARCH_USERNAME, 
                        settings.ELASTICSEARCH_PASSWORD
                    )
                
                self.elasticsearch_client = AsyncElasticsearch(**es_config)
            
            logger.info("Retrieval service clients initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize retrieval clients: {str(e)}")
    
    async def retrieve_clauses(
        self, 
        query: str, 
        doc_id: str,
        top_n: int = 8,
        clause_types: Optional[List[str]] = None,
        include_low_confidence: bool = False
    ) -> List[RetrievalCandidate]:
        """
        Retrieve relevant clauses using hybrid poly-vector approach.
        
        Args:
            query: Natural language query
            doc_id: Document ID to search within
            top_n: Number of results to return
            clause_types: Filter by clause types
            include_low_confidence: Include low-confidence results
            
        Returns:
            List of ranked retrieval candidates
        """
        try:
            start_time = datetime.utcnow()
            
            # Step 1: Generate HyDE pseudo-document if enabled
            hyde_query = None
            if settings.HYDE_ENABLED:
                hyde_query = await self._generate_hyde_query(query)
                logger.info(f"Generated HyDE query: {hyde_query[:100]}...")
            
            # Step 2: Parallel retrieval
            retrieval_tasks = []
            
            # Sparse retrieval (BM25)
            if self.elasticsearch_client:
                retrieval_tasks.append(
                    self._sparse_retrieval(query, doc_id, settings.RETRIEVAL_TOP_K_CANDIDATES)
                )
            
            # Dense retrieval (vector similarity)
            retrieval_tasks.append(
                self._dense_retrieval(query, doc_id, settings.RETRIEVAL_TOP_K_CANDIDATES)
            )
            
            # HyDE dense retrieval
            if hyde_query:
                retrieval_tasks.append(
                    self._dense_retrieval(hyde_query, doc_id, settings.RETRIEVAL_TOP_K_CANDIDATES)
                )
            
            # Label-based retrieval
            retrieval_tasks.append(
                self._label_retrieval(query, doc_id, settings.RETRIEVAL_TOP_K_CANDIDATES)
            )
            
            # Execute retrievals in parallel
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
            
            # Step 3: Merge and score candidates
            all_candidates = {}
            
            for i, result in enumerate(retrieval_results):
                if isinstance(result, Exception):
                    logger.error(f"Retrieval task {i} failed: {str(result)}")
                    continue
                
                for candidate in result:
                    clause_id = candidate.clause_id
                    if clause_id not in all_candidates:
                        all_candidates[clause_id] = candidate
                    else:
                        # Merge scores
                        existing = all_candidates[clause_id]
                        existing.sparse_score = max(existing.sparse_score, candidate.sparse_score)
                        existing.dense_score = max(existing.dense_score, candidate.dense_score)
                        existing.label_score = max(existing.label_score, candidate.label_score)
            
            # Step 4: Calculate composite scores
            scored_candidates = []
            for candidate in all_candidates.values():
                composite_score = self._calculate_composite_score(
                    candidate, query, doc_id
                )
                candidate.score = composite_score
                scored_candidates.append(candidate)
            
            # Step 5: Sort by composite score
            scored_candidates.sort(key=lambda x: x.score, reverse=True)
            
            # Step 6: Cross-encoder reranking
            if settings.RERANKER_ENABLED and len(scored_candidates) > top_n:
                top_candidates = scored_candidates[:settings.RETRIEVAL_RERANK_TOP_K]
                reranked = await self._cross_encoder_rerank(query, top_candidates)
                scored_candidates = reranked + scored_candidates[settings.RETRIEVAL_RERANK_TOP_K:]
            
            # Step 7: Apply filters
            filtered_candidates = self._apply_filters(
                scored_candidates, clause_types, include_low_confidence
            )
            
            # Step 8: Return top N
            final_results = filtered_candidates[:top_n]
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Retrieved {len(final_results)} clauses in {duration:.2f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"Clause retrieval failed: {str(e)}")
            return []
    
    async def _generate_hyde_query(self, query: str) -> str:
        """Generate HyDE pseudo-document for improved retrieval."""
        try:
            hyde_prompt = f'''System: You are a retrieval-helper. Given the user query, produce a concise hypothetical document (1-3 sentences) that best answers the query. This text will be embedded for retrieval only. Do NOT invent statutes or cases.

User: "{query}"

Hypothetical document:'''
            
            # Use generation service to create HyDE query
            hyde_response = await generation_service.generate_text(
                prompt=hyde_prompt,
                temperature=settings.HYDE_TEMPERATURE,
                max_tokens=200
            )
            
            return hyde_response.strip()
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {str(e)}")
            return query  # Fallback to original query
    
    async def _sparse_retrieval(self, query: str, doc_id: str, top_k: int) -> List[RetrievalCandidate]:
        """Perform BM25 sparse retrieval using Elasticsearch."""
        try:
            if not self.elasticsearch_client:
                return []
            
            # Construct Elasticsearch query
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content_text^2", "section_title^1.5"],
                                    "type": "best_fields",
                                    "operator": "or"
                                }
                            }
                        ],
                        "filter": [
                            {"term": {"doc_id": doc_id}}
                        ]
                    }
                },
                "size": top_k,
                "_source": ["clause_id", "content_text", "section_title", "clause_type", "page"]
            }
            
            # Execute search
            response = await self.elasticsearch_client.search(
                index=settings.ELASTICSEARCH_INDEX,
                body=es_query
            )
            
            # Convert to candidates
            candidates = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                candidate = RetrievalCandidate(
                    clause_id=source["clause_id"],
                    score=0.0,  # Will be calculated later
                    sparse_score=hit["_score"],
                    dense_score=0.0,
                    label_score=0.0,
                    content_text=source["content_text"],
                    metadata={
                        "section_title": source.get("section_title"),
                        "clause_type": source.get("clause_type"),
                        "page": source.get("page")
                    }
                )
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {str(e)}")
            return []
    
    async def _dense_retrieval(self, query: str, doc_id: str, top_k: int) -> List[RetrievalCandidate]:
        """Perform dense vector retrieval using Vertex AI Matching Engine."""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            if not query_embedding:
                return []
            
            # For now, return mock results since we don't have the actual vector index
            # In production, this would query Vertex AI Matching Engine
            mock_candidates = []
            
            # TODO: Replace with actual Matching Engine query
            # endpoint = MatchingEngineIndexEndpoint(settings.MATCHING_ENGINE_INDEX_ENDPOINT)
            # results = endpoint.find_neighbors(
            #     deployed_index_id=settings.MATCHING_ENGINE_DEPLOYED_INDEX_ID,
            #     queries=[query_embedding],
            #     num_neighbors=top_k
            # )
            
            return mock_candidates
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {str(e)}")
            return []
    
    async def _label_retrieval(self, query: str, doc_id: str, top_k: int) -> List[RetrievalCandidate]:
        """Perform label-aware retrieval for specific clause references."""
        try:
            # Extract potential clause labels from query
            import re
            label_patterns = [
                r'(?:clause|section|article)\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:clause|section|article)',
                r'paragraph\s*(\d+)',
            ]
            
            found_labels = []
            for pattern in label_patterns:
                matches = re.finditer(pattern, query.lower())
                for match in matches:
                    found_labels.append(match.group(1))
            
            if not found_labels:
                return []
            
            # Search for clauses with matching labels
            candidates = []
            
            # Mock implementation - in production, would search database/index
            for label in found_labels:
                # This would be a database query in production
                mock_candidate = RetrievalCandidate(
                    clause_id=f"{doc_id}_label_{label}",
                    score=0.0,
                    sparse_score=0.0,
                    dense_score=0.0,
                    label_score=1.0,  # High label score
                    content_text=f"Mock clause content for label {label}",
                    metadata={"label": label, "source": "label_retrieval"}
                )
                candidates.append(mock_candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Label retrieval failed: {str(e)}")
            return []
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Vertex AI."""
        try:
            # Mock implementation - replace with actual Vertex AI call
            # In production:
            # from vertexai.language_models import TextEmbeddingModel
            # model = TextEmbeddingModel.from_pretrained(settings.VERTEX_EMBEDDING_MODEL)
            # embeddings = model.get_embeddings([text])
            # return embeddings[0].values
            
            # Return mock embedding for now
            return [0.1] * 768  # Mock 768-dimensional embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return None
    
    def _calculate_composite_score(
        self, 
        candidate: RetrievalCandidate, 
        query: str, 
        doc_id: str
    ) -> float:
        """Calculate composite score using configured weights."""
        try:
            weights = settings.retrieval_weights
            
            # Normalize scores (simple min-max normalization)
            norm_sparse = min(candidate.sparse_score / 10.0, 1.0)  # Assuming max BM25 score ~10
            norm_dense = candidate.dense_score  # Already normalized 0-1
            norm_label = candidate.label_score  # Already 0-1
            
            # Additional scoring factors
            jurisdiction_score = self._calculate_jurisdiction_score(candidate, query)
            template_score = self._calculate_template_score(candidate, doc_id)
            
            # Composite score
            composite = (
                weights["sparse"] * norm_sparse +
                weights["dense"] * norm_dense +
                weights["label"] * norm_label +
                weights["jurisdiction"] * jurisdiction_score +
                weights["template"] * template_score
            )
            
            return composite
            
        except Exception as e:
            logger.error(f"Score calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_jurisdiction_score(self, candidate: RetrievalCandidate, query: str) -> float:
        """Calculate jurisdiction relevance score."""
        try:
            # Simple keyword matching for jurisdiction
            query_lower = query.lower()
            content_lower = candidate.content_text.lower()
            
            jurisdiction_keywords = [
                'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata',
                'maharashtra', 'karnataka', 'tamil nadu', 'west bengal',
                'indian law', 'indian courts'
            ]
            
            score = 0.0
            for keyword in jurisdiction_keywords:
                if keyword in query_lower and keyword in content_lower:
                    score += 0.2
            
            return min(score, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_template_score(self, candidate: RetrievalCandidate, doc_id: str) -> float:
        """Calculate template similarity score."""
        try:
            # Mock implementation - in production, would use template fingerprints
            return 0.1  # Small bonus for being from same document
            
        except Exception:
            return 0.0
    
    async def _cross_encoder_rerank(
        self, 
        query: str, 
        candidates: List[RetrievalCandidate]
    ) -> List[RetrievalCandidate]:
        """Rerank candidates using cross-encoder model."""
        try:
            # Mock implementation - in production, would use actual cross-encoder
            # from sentence_transformers import CrossEncoder
            # model = CrossEncoder(settings.RERANKER_MODEL)
            # pairs = [(query, candidate.content_text) for candidate in candidates]
            # scores = model.predict(pairs)
            
            # For now, just return candidates with slight score adjustments
            for i, candidate in enumerate(candidates):
                # Mock reranking score
                rerank_boost = 0.1 - (i * 0.01)  # Slight preference for initial ranking
                candidate.score += rerank_boost
            
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {str(e)}")
            return candidates
    
    def _apply_filters(
        self, 
        candidates: List[RetrievalCandidate],
        clause_types: Optional[List[str]] = None,
        include_low_confidence: bool = False
    ) -> List[RetrievalCandidate]:
        """Apply filters to candidates."""
        try:
            filtered = candidates
            
            # Filter by clause types
            if clause_types:
                filtered = [
                    c for c in filtered 
                    if c.metadata.get("clause_type") in clause_types
                ]
            
            # Filter by confidence
            if not include_low_confidence:
                confidence_threshold = 0.3
                filtered = [
                    c for c in filtered 
                    if c.score >= confidence_threshold
                ]
            
            return filtered
            
        except Exception as e:
            logger.error(f"Filtering failed: {str(e)}")
            return candidates
    
    async def index_clauses(self, clauses: List[ClauseMetadata]) -> bool:
        """Index clauses for retrieval."""
        try:
            if not clauses:
                return True
            
            # Index in Elasticsearch
            if self.elasticsearch_client:
                await self._index_clauses_elasticsearch(clauses)
            
            # Index in vector database
            await self._index_clauses_vector(clauses)
            
            logger.info(f"Indexed {len(clauses)} clauses")
            return True
            
        except Exception as e:
            logger.error(f"Clause indexing failed: {str(e)}")
            return False
    
    async def _index_clauses_elasticsearch(self, clauses: List[ClauseMetadata]) -> None:
        """Index clauses in Elasticsearch."""
        try:
            if not self.elasticsearch_client:
                return
            
            # Prepare bulk indexing
            actions = []
            for clause in clauses:
                action = {
                    "_index": settings.ELASTICSEARCH_INDEX,
                    "_id": clause.clause_id,
                    "_source": {
                        "clause_id": clause.clause_id,
                        "doc_id": clause.doc_id,
                        "content_text": clause.content_text,
                        "section_title": clause.section_title,
                        "clause_type": clause.clause_type.value,
                        "page": clause.page,
                        "created_at": clause.created_at.isoformat()
                    }
                }
                actions.append(action)
            
            # Execute bulk indexing
            from elasticsearch.helpers import async_bulk
            await async_bulk(self.elasticsearch_client, actions)
            
        except Exception as e:
            logger.error(f"Elasticsearch indexing failed: {str(e)}")
    
    async def _index_clauses_vector(self, clauses: List[ClauseMetadata]) -> None:
        """Index clauses in vector database."""
        try:
            for clause in clauses:
                # Generate embeddings
                content_embedding = await self._generate_embedding(clause.content_text)
                
                if clause.section_title:
                    label_embedding = await self._generate_embedding(clause.section_title)
                else:
                    label_embedding = content_embedding
                
                # Store embeddings (mock implementation)
                # In production, would store in Vertex AI Matching Engine
                logger.debug(f"Generated embeddings for clause {clause.clause_id}")
            
        except Exception as e:
            logger.error(f"Vector indexing failed: {str(e)}")


# Global service instance
retrieval_service = RetrievalService()