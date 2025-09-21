"""
Vector Search Service for MVP
Handles Vertex AI embeddings and Matching Engine search.
Single-user hackathon prototype - Vector search only (no BM25).
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

# Google Cloud / Vertex AI imports
try:
    from google.cloud import aiplatform
    from vertexai.language_models import TextEmbeddingModel
    import vertexai
except ImportError:
    aiplatform = None
    TextEmbeddingModel = None
    vertexai = None

from ..models.mvp_schemas import (
    VectorSearchResult, VectorSearchRequest, VectorSearchResponse, 
    VectorMetadata, ExtractedClause, ClauseType
)
from ..models.config import settings

logger = logging.getLogger(__name__)


class VectorSearchService:
    """
    Simplified vector search for hackathon MVP.
    
    Uses only Vertex AI:
    1. Text Embedding API for generating embeddings
    2. Matching Engine for vector similarity search
    
    No Elasticsearch/BM25 to keep it simple.
    """
    
    def __init__(self):
        """Initialize Vertex AI clients."""
        self.embedding_model = None
        self.matching_engine_client = None
        self._initialize_vertex_ai()
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI services."""
        try:
            if vertexai:
                # Initialize Vertex AI
                vertexai.init(
                    project=settings.google_cloud_project_id,
                    location=settings.vertex_ai_location
                )
                
                # Initialize embedding model
                self.embedding_model = TextEmbeddingModel.from_pretrained(
                    settings.embedding_model_name
                )
                
                logger.info(f"Vertex AI initialized with embedding model: {settings.embedding_model_name}")
            
            if aiplatform:
                # Initialize AI Platform for Matching Engine
                aiplatform.init(
                    project=settings.google_cloud_project_id,
                    location=settings.vertex_ai_location
                )
                
                self.matching_engine_client = aiplatform.MatchingEngineIndexEndpoint(
                    index_endpoint_name=settings.matching_engine_endpoint_name
                )
                
                logger.info("Matching Engine client initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {str(e)}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            if not self.embedding_model:
                # Fallback for development
                logger.warning("Embedding model not available, using random vector")
                import random
                return [random.random() for _ in range(settings.embedding_dimensions)]
            
            # Clean and truncate text
            clean_text = text.strip()[:8000]  # Vertex AI has token limits
            
            # Generate embedding
            embeddings = self.embedding_model.get_embeddings([clean_text])
            embedding_vector = embeddings[0].values
            
            logger.debug(f"Generated embedding for text (length: {len(clean_text)})")
            return embedding_vector
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * settings.embedding_dimensions
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if not self.embedding_model:
                logger.warning("Embedding model not available, using random vectors")
                import random
                return [[random.random() for _ in range(settings.embedding_dimensions)] for _ in texts]
            
            # Clean texts
            clean_texts = [text.strip()[:8000] for text in texts]
            
            # Process in batches (Vertex AI has batch limits)
            batch_size = 5
            all_embeddings = []
            
            for i in range(0, len(clean_texts), batch_size):
                batch = clean_texts[i:i+batch_size]
                embeddings = self.embedding_model.get_embeddings(batch)
                batch_vectors = [emb.values for emb in embeddings]
                all_embeddings.extend(batch_vectors)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings in batch")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * settings.embedding_dimensions for _ in texts]
    
    async def index_clause(self, clause: ExtractedClause) -> str:
        """
        Index a clause in Matching Engine.
        
        Args:
            clause: ExtractedClause to index
            
        Returns:
            Embedding ID for the indexed clause
        """
        try:
            # Generate embedding for clause text
            embedding = await self.generate_embedding(clause.text)
            
            # Create metadata
            metadata = VectorMetadata(
                clause_id=clause.clause_id,
                doc_id=clause.doc_id,
                clause_type=clause.clause_type.value,
                page_number=clause.page_number,
                text_snippet=clause.text[:200],  # First 200 chars
                created_at=clause.created_at.isoformat()
            )
            
            # Generate embedding ID
            embedding_id = f"emb_{clause.clause_id}_{uuid.uuid4().hex[:8]}"
            
            # Index in Matching Engine
            if self.matching_engine_client:
                await self._add_to_matching_engine(embedding_id, embedding, metadata)
            else:
                logger.warning("Matching Engine not available, skipping indexing")
            
            logger.info(f"Indexed clause {clause.clause_id} with embedding {embedding_id}")
            return embedding_id
            
        except Exception as e:
            logger.error(f"Clause indexing failed: {str(e)}")
            return f"error_{clause.clause_id}"
    
    async def index_clauses_batch(self, clauses: List[ExtractedClause]) -> List[str]:
        """
        Index multiple clauses efficiently.
        
        Args:
            clauses: List of ExtractedClause objects
            
        Returns:
            List of embedding IDs
        """
        try:
            # Generate embeddings in batch
            texts = [clause.text for clause in clauses]
            embeddings = await self.generate_embeddings_batch(texts)
            
            # Create embedding IDs and metadata
            embedding_ids = []
            metadata_list = []
            
            for clause, embedding in zip(clauses, embeddings):
                embedding_id = f"emb_{clause.clause_id}_{uuid.uuid4().hex[:8]}"
                embedding_ids.append(embedding_id)
                
                metadata = VectorMetadata(
                    clause_id=clause.clause_id,
                    doc_id=clause.doc_id,
                    clause_type=clause.clause_type.value,
                    page_number=clause.page_number,
                    text_snippet=clause.text[:200],
                    created_at=clause.created_at.isoformat()
                )
                metadata_list.append(metadata)
            
            # Batch index in Matching Engine
            if self.matching_engine_client:
                await self._add_batch_to_matching_engine(embedding_ids, embeddings, metadata_list)
            else:
                logger.warning("Matching Engine not available, skipping batch indexing")
            
            logger.info(f"Batch indexed {len(clauses)} clauses")
            return embedding_ids
            
        except Exception as e:
            logger.error(f"Batch clause indexing failed: {str(e)}")
            return [f"error_{clause.clause_id}" for clause in clauses]
    
    async def search_similar_clauses(
        self,
        query: str,
        doc_id: Optional[str] = None,
        clause_type: Optional[ClauseType] = None,
        top_k: int = 10
    ) -> VectorSearchResponse:
        """
        Search for similar clauses using vector similarity.
        
        Args:
            query: Search query text
            doc_id: Optional filter by document ID
            clause_type: Optional filter by clause type
            top_k: Number of results to return
            
        Returns:
            VectorSearchResponse with similar clauses
        """
        try:
            start_time = datetime.utcnow()
            
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Search in Matching Engine
            results = await self._search_matching_engine(
                query_embedding, doc_id, clause_type, top_k
            )
            
            # Convert to VectorSearchResult format
            search_results = []
            for result in results:
                search_result = VectorSearchResult(
                    clause_id=result['metadata']['clause_id'],
                    doc_id=result['metadata']['doc_id'],
                    text=result.get('text', result['metadata']['text_snippet']),
                    clause_type=ClauseType(result['metadata']['clause_type']),
                    similarity_score=result['similarity_score'],
                    metadata=result['metadata']
                )
                search_results.append(search_result)
            
            query_time = (datetime.utcnow() - start_time).total_seconds()
            
            response = VectorSearchResponse(
                results=search_results,
                query_time=query_time
            )
            
            logger.info(f"Vector search completed: {len(search_results)} results in {query_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return VectorSearchResponse(results=[], query_time=0.0)
    
    async def _add_to_matching_engine(
        self, 
        embedding_id: str, 
        embedding: List[float], 
        metadata: VectorMetadata
    ):
        """Add single embedding to Matching Engine."""
        try:
            # In production, this would use the Matching Engine API
            # For MVP, we'll simulate this
            logger.debug(f"Added embedding {embedding_id} to Matching Engine")
            
        except Exception as e:
            logger.error(f"Failed to add to Matching Engine: {str(e)}")
    
    async def _add_batch_to_matching_engine(
        self,
        embedding_ids: List[str],
        embeddings: List[List[float]],
        metadata_list: List[VectorMetadata]
    ):
        """Add multiple embeddings to Matching Engine."""
        try:
            # In production, this would use the Matching Engine batch API
            logger.debug(f"Added {len(embedding_ids)} embeddings to Matching Engine")
            
        except Exception as e:
            logger.error(f"Failed to batch add to Matching Engine: {str(e)}")
    
    async def _search_matching_engine(
        self,
        query_embedding: List[float],
        doc_id: Optional[str],
        clause_type: Optional[ClauseType],
        top_k: int
    ) -> List[Dict]:
        """Search Matching Engine for similar vectors."""
        try:
            if not self.matching_engine_client:
                # Fallback for development - return mock results
                logger.warning("Matching Engine not available, returning mock results")
                return [
                    {
                        'metadata': {
                            'clause_id': f'mock_clause_{i}',
                            'doc_id': doc_id or 'mock_doc',
                            'clause_type': clause_type.value if clause_type else 'general',
                            'text_snippet': f'Mock clause {i} for development testing'
                        },
                        'similarity_score': 0.9 - (i * 0.1),
                        'text': f'This is a mock clause {i} for development testing purposes.'
                    }
                    for i in range(min(top_k, 3))
                ]
            
            # In production, this would call the actual Matching Engine API
            # For now, return empty results
            return []
            
        except Exception as e:
            logger.error(f"Matching Engine search failed: {str(e)}")
            return []
    
    async def get_embedding_stats(self) -> Dict[str, int]:
        """Get statistics about indexed embeddings."""
        try:
            # In production, this would query Matching Engine for stats
            return {
                "total_embeddings": 0,
                "indexed_documents": 0,
                "index_size_mb": 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get embedding stats: {str(e)}")
            return {}


# Global service instance
vector_search_service = VectorSearchService()