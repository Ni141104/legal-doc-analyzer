"""
Poly-vector retrieval system for the Legal Document Analyzer.
Implements hybrid dense + sparse + label-aware retrieval with HyDE pseudo-doc generation.
"""

import asyncio
import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from ..models.schemas import ClauseRecord, SourceSpan
from ..models.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """A candidate clause with retrieval scores."""
    clause_id: str
    clause_record: ClauseRecord
    scores: Dict[str, float]  # component scores
    combined_score: float
    spans: List[SourceSpan]


@dataclass
class RetrievalResult:
    """Result of retrieval operation."""
    candidates: List[RetrievalCandidate]
    query: str
    retrieval_method: str
    processing_time_ms: float
    total_candidates: int
    metadata: Dict[str, Any]


class VectorIndex(ABC):
    """Abstract base class for vector indexes."""
    
    @abstractmethod
    async def add_vectors(self, vectors: List[Tuple[str, np.ndarray, Dict[str, Any]]]) -> bool:
        """Add vectors to the index."""
        pass
    
    @abstractmethod
    async def search(self, query_vector: np.ndarray, top_k: int, filters: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from index."""
        pass


class PineconeIndex(VectorIndex):
    """Pinecone vector index implementation."""
    
    def __init__(self, index_name: str, dimension: int = 768):
        import pinecone
        from ..models.config import vector_config
        
        config = vector_config.pinecone_config
        pinecone.init(api_key=config["api_key"], environment=config["environment"])
        
        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                metadata_config={"indexed": ["doc_id", "clause_type", "jurisdiction"]}
            )
        
        self.index = pinecone.Index(index_name)
        self.dimension = dimension
    
    async def add_vectors(self, vectors: List[Tuple[str, np.ndarray, Dict[str, Any]]]) -> bool:
        """Add vectors to Pinecone index."""
        try:
            upsert_data = []
            for vec_id, vector, metadata in vectors:
                upsert_data.append((vec_id, vector.tolist(), metadata))
            
            self.index.upsert(vectors=upsert_data)
            return True
        
        except Exception as e:
            logger.error(f"Failed to add vectors to Pinecone: {e}")
            return False
    
    async def search(self, query_vector: np.ndarray, top_k: int, filters: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """Search Pinecone index."""
        try:
            query_filter = {}
            if filters:
                for key, value in filters.items():
                    if value is not None:
                        query_filter[key] = {"$eq": value}
            
            results = self.index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                filter=query_filter if query_filter else None,
                include_metadata=False
            )
            
            return [(match.id, match.score) for match in results.matches]
        
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Pinecone."""
        try:
            self.index.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {e}")
            return False


class ElasticsearchIndex:
    """Elasticsearch sparse search index."""
    
    def __init__(self, index_name: str = "legal_clauses"):
        from elasticsearch import Elasticsearch
        from ..models.config import db_config
        
        config = db_config.elasticsearch_config
        self.es = Elasticsearch(**config)
        self.index_name = index_name
        
        # Create index if it doesn't exist
        if not self.es.indices.exists(index=index_name):
            self._create_index()
    
    def _create_index(self):
        """Create Elasticsearch index with proper mappings."""
        mapping = {
            "mappings": {
                "properties": {
                    "clause_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "content_text": {
                        "type": "text",
                        "analyzer": "legal_analyzer"
                    },
                    "section_title": {
                        "type": "text",
                        "analyzer": "legal_analyzer"
                    },
                    "clause_type": {"type": "keyword"},
                    "jurisdiction": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "extracted_amounts": {"type": "float"},
                    "extracted_parties": {"type": "text"},
                    "template_fingerprint": {"type": "keyword"},
                    "ingested_at": {"type": "date"}
                }
            },
            "settings": {
                "analysis": {
                    "analyzer": {
                        "legal_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "legal_synonyms",
                                "stop"
                            ]
                        }
                    },
                    "filter": {
                        "legal_synonyms": {
                            "type": "synonym",
                            "synonyms": [
                                "landlord,lessor,owner",
                                "tenant,lessee,renter", 
                                "agreement,contract,document",
                                "deposit,advance,security",
                                "rent,rental,lease amount"
                            ]
                        }
                    }
                }
            }
        }
        
        self.es.indices.create(index=self.index_name, body=mapping)
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to Elasticsearch."""
        try:
            from elasticsearch.helpers import bulk
            
            actions = []
            for doc in documents:
                action = {
                    "_index": self.index_name,
                    "_id": doc["clause_id"],
                    "_source": doc
                }
                actions.append(action)
            
            bulk(self.es, actions)
            return True
        
        except Exception as e:
            logger.error(f"Failed to add documents to Elasticsearch: {e}")
            return False
    
    async def search(self, query: str, top_k: int, filters: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """Search Elasticsearch index."""
        try:
            # Build query
            query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content_text^2", "section_title^1.5"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ]
                    }
                },
                "size": top_k
            }
            
            # Add filters
            if filters:
                filter_clauses = []
                for key, value in filters.items():
                    if value is not None:
                        filter_clauses.append({"term": {key: value}})
                
                if filter_clauses:
                    query_body["query"]["bool"]["filter"] = filter_clauses
            
            response = self.es.search(index=self.index_name, body=query_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append((hit["_id"], hit["_score"]))
            
            return results
        
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []


class EmbeddingService:
    """Service for generating embeddings using Vertex AI."""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Vertex AI client."""
        if self._client is None:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
            
            vertexai.init(
                project=settings.GCP_PROJECT_ID,
                location=settings.VERTEX_AI_LOCATION
            )
            self._client = TextEmbeddingModel.from_pretrained(self.model_name)
        
        return self._client
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        try:
            embeddings = self.client.get_embeddings([text])
            return np.array(embeddings[0].values)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.zeros(768)  # Return zero vector as fallback
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        try:
            embeddings = self.client.get_embeddings(texts)
            return [np.array(emb.values) for emb in embeddings]
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [np.zeros(768) for _ in texts]


class HyDEGenerator:
    """HyDE (Hypothetical Document Embeddings) generator using Gemini."""
    
    def __init__(self):
        self._client = None
    
    @property 
    def client(self):
        """Lazy initialization of Vertex AI Gemini client."""
        if self._client is None:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            vertexai.init(
                project=settings.GCP_PROJECT_ID,
                location=settings.VERTEX_AI_LOCATION
            )
            self._client = GenerativeModel(settings.GEMINI_MODEL)
        
        return self._client
    
    async def generate_pseudo_document(self, query: str) -> str:
        """Generate hypothetical document for HyDE retrieval."""
        prompt = f"""System: You are a retrieval-helper. Given the user query, produce a concise hypothetical document (1-3 sentences) that best answers the query. This text will be embedded for retrieval only. Do NOT invent statutes or cases.

User: "{query}"

Generate a hypothetical legal document snippet that would contain the answer:"""
        
        try:
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 150,
                    "top_p": 0.8
                }
            )
            
            return response.text.strip()
        
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return query  # Fallback to original query


class CrossEncoderReranker:
    """Cross-encoder for reranking retrieval results."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not available, skipping cross-encoder reranking")
                self._model = None
        return self._model
    
    async def rerank(self, query: str, candidates: List[RetrievalCandidate], top_k: int) -> List[RetrievalCandidate]:
        """Rerank candidates using cross-encoder."""
        if not self.model or not candidates:
            return candidates[:top_k]
        
        try:
            # Prepare query-document pairs
            pairs = []
            for candidate in candidates:
                pairs.append([query, candidate.clause_record.content_text])
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Update candidate scores
            for i, candidate in enumerate(candidates):
                candidate.scores["cross_encoder"] = float(scores[i])
                # Update combined score (weighted)
                candidate.combined_score = (
                    0.7 * candidate.combined_score + 
                    0.3 * candidate.scores["cross_encoder"]
                )
            
            # Sort by new combined score and return top-k
            candidates.sort(key=lambda x: x.combined_score, reverse=True)
            return candidates[:top_k]
        
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return candidates[:top_k]


class PolyVectorRetriever:
    """Main poly-vector retrieval system."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.hyde_generator = HyDEGenerator()
        self.cross_encoder = CrossEncoderReranker()
        
        # Vector indexes
        self.content_index = PineconeIndex("content-vectors")
        self.label_index = PineconeIndex("label-vectors") 
        self.alias_index = PineconeIndex("alias-vectors")
        
        # Sparse index
        self.sparse_index = ElasticsearchIndex()
        
        # In-memory clause storage (in production, use database)
        self.clause_store: Dict[str, ClauseRecord] = {}
    
    async def index_clause(self, clause: ClauseRecord) -> bool:
        """Index a clause with all vector types."""
        try:
            # Generate embeddings
            content_embedding = await self.embedding_service.embed_text(clause.content_text)
            
            # Create label text
            label_text = f"{clause.section_title or ''} {clause.clause_type.value}"
            label_embedding = await self.embedding_service.embed_text(label_text)
            
            # Generate alias embeddings (synonyms, translations)
            aliases = self._generate_aliases(clause)
            alias_embeddings = await self.embedding_service.embed_batch(aliases)
            
            # Add to vector indexes
            metadata = {
                "doc_id": clause.doc_id,
                "clause_type": clause.clause_type.value,
                "jurisdiction": clause.jurisdiction,
                "page": clause.page
            }
            
            await self.content_index.add_vectors([
                (f"content_{clause.clause_id}", content_embedding, metadata)
            ])
            
            await self.label_index.add_vectors([
                (f"label_{clause.clause_id}", label_embedding, metadata)
            ])
            
            alias_vectors = [
                (f"alias_{clause.clause_id}_{i}", emb, metadata)
                for i, emb in enumerate(alias_embeddings)
            ]
            await self.alias_index.add_vectors(alias_vectors)
            
            # Add to sparse index
            doc = {
                "clause_id": clause.clause_id,
                "doc_id": clause.doc_id,
                "content_text": clause.content_text,
                "section_title": clause.section_title,
                "clause_type": clause.clause_type.value,
                "jurisdiction": clause.jurisdiction,
                "page": clause.page,
                "extracted_amounts": [fact.value for fact in clause.extracted_facts if fact.fact_type == "currency"],
                "extracted_parties": [fact.value for fact in clause.extracted_facts if fact.fact_type == "party"],
                "template_fingerprint": clause.template_fingerprint,
                "ingested_at": clause.ingested_at.isoformat()
            }
            
            await self.sparse_index.add_documents([doc])
            
            # Store clause record
            self.clause_store[clause.clause_id] = clause
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to index clause {clause.clause_id}: {e}")
            return False
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 8,
        doc_id: Optional[str] = None,
        clause_type: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        use_hyde: bool = True
    ) -> RetrievalResult:
        """Retrieve relevant clauses using hybrid poly-vector approach."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Parse query for special tokens
            label_tokens = self._extract_label_tokens(query)
            
            # Generate query embeddings
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Generate HyDE pseudo-document if enabled
            hyde_embedding = None
            if use_hyde:
                pseudo_doc = await self.hyde_generator.generate_pseudo_document(query)
                hyde_embedding = await self.embedding_service.embed_text(pseudo_doc)
            
            # Build filters
            filters = {}
            if doc_id:
                filters["doc_id"] = doc_id
            if clause_type:
                filters["clause_type"] = clause_type
            if jurisdiction:
                filters["jurisdiction"] = jurisdiction
            
            # Parallel retrieval
            tasks = []
            
            # Sparse search
            tasks.append(self.sparse_index.search(query, settings.MAX_SPARSE_RESULTS, filters))
            
            # Dense content search
            tasks.append(self.content_index.search(query_embedding, settings.MAX_DENSE_RESULTS, filters))
            
            # HyDE search
            if hyde_embedding is not None:
                tasks.append(self.content_index.search(hyde_embedding, settings.MAX_DENSE_RESULTS, filters))
            else:
                tasks.append(asyncio.sleep(0, result=[]))  # No-op
            
            # Label search if label tokens found
            if label_tokens:
                label_query_embedding = await self.embedding_service.embed_text(" ".join(label_tokens))
                tasks.append(self.label_index.search(label_query_embedding, 20, filters))
            else:
                tasks.append(asyncio.sleep(0, result=[]))  # No-op
            
            # Execute all searches in parallel
            sparse_results, dense_results, hyde_results, label_results = await asyncio.gather(*tasks)
            
            # Merge and score candidates
            candidates = self._merge_and_score_candidates(
                query, sparse_results, dense_results, hyde_results, label_results
            )
            
            # Cross-encoder reranking
            if len(candidates) > top_k:
                candidates = await self.cross_encoder.rerank(query, candidates, top_k)
            else:
                candidates = candidates[:top_k]
            
            end_time = asyncio.get_event_loop().time()
            processing_time = (end_time - start_time) * 1000
            
            return RetrievalResult(
                candidates=candidates,
                query=query,
                retrieval_method="poly_vector_hybrid",
                processing_time_ms=processing_time,
                total_candidates=len(sparse_results) + len(dense_results) + len(hyde_results) + len(label_results),
                metadata={
                    "sparse_count": len(sparse_results),
                    "dense_count": len(dense_results),
                    "hyde_count": len(hyde_results),
                    "label_count": len(label_results),
                    "use_hyde": use_hyde,
                    "filters": filters
                }
            )
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return RetrievalResult(
                candidates=[],
                query=query,
                retrieval_method="failed",
                processing_time_ms=0,
                total_candidates=0,
                metadata={"error": str(e)}
            )
    
    def _generate_aliases(self, clause: ClauseRecord) -> List[str]:
        """Generate alias texts for a clause."""
        aliases = []
        
        # Add section title variations
        if clause.section_title:
            aliases.append(clause.section_title.lower())
        
        # Add clause type variations
        type_aliases = {
            "PAYMENT": ["payment terms", "payment clause", "amount due"],
            "DEPOSIT": ["security deposit", "advance payment", "refundable deposit"],
            "TERMINATION": ["termination clause", "ending agreement", "contract termination"],
            "LIABILITY": ["liability clause", "responsibility", "damages"],
        }
        
        clause_type_aliases = type_aliases.get(clause.clause_type.value, [])
        aliases.extend(clause_type_aliases)
        
        # Add extracted fact variations
        for fact in clause.extracted_facts:
            if fact.fact_type == "currency":
                aliases.append(f"amount {fact.value}")
            elif fact.fact_type == "party":
                aliases.append(str(fact.value))
        
        return aliases[:5]  # Limit aliases
    
    def _extract_label_tokens(self, query: str) -> List[str]:
        """Extract label/reference tokens from query."""
        import re
        
        label_patterns = [
            r'clause\s+(\d+(?:\.\d+)?)',
            r'section\s+(\d+(?:\.\d+)?)', 
            r'article\s+(\d+(?:\.\d+)?)',
            r'paragraph\s+(\d+(?:\.\d+)?)'
        ]
        
        tokens = []
        for pattern in label_patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                tokens.append(f"clause {match.group(1)}")
        
        return tokens
    
    def _merge_and_score_candidates(
        self,
        query: str,
        sparse_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        hyde_results: List[Tuple[str, float]],
        label_results: List[Tuple[str, float]]
    ) -> List[RetrievalCandidate]:
        """Merge retrieval results and compute combined scores."""
        
        # Collect all candidate IDs
        all_candidates = {}
        
        # Process sparse results
        sparse_scores = self._normalize_scores([score for _, score in sparse_results])
        for i, (clause_id, _) in enumerate(sparse_results):
            # Extract clause_id from vector ID
            actual_clause_id = clause_id.replace("content_", "").replace("label_", "").replace("alias_", "")
            if actual_clause_id in self.clause_store:
                if actual_clause_id not in all_candidates:
                    all_candidates[actual_clause_id] = {"sparse": 0, "dense": 0, "hyde": 0, "label": 0}
                all_candidates[actual_clause_id]["sparse"] = sparse_scores[i]
        
        # Process dense results  
        dense_scores = self._normalize_scores([score for _, score in dense_results])
        for i, (clause_id, _) in enumerate(dense_results):
            actual_clause_id = clause_id.replace("content_", "").replace("label_", "").replace("alias_", "")
            if actual_clause_id in self.clause_store:
                if actual_clause_id not in all_candidates:
                    all_candidates[actual_clause_id] = {"sparse": 0, "dense": 0, "hyde": 0, "label": 0}
                all_candidates[actual_clause_id]["dense"] = dense_scores[i]
        
        # Process HyDE results
        if hyde_results:
            hyde_scores = self._normalize_scores([score for _, score in hyde_results])
            for i, (clause_id, _) in enumerate(hyde_results):
                actual_clause_id = clause_id.replace("content_", "").replace("label_", "").replace("alias_", "")
                if actual_clause_id in self.clause_store:
                    if actual_clause_id not in all_candidates:
                        all_candidates[actual_clause_id] = {"sparse": 0, "dense": 0, "hyde": 0, "label": 0}
                    all_candidates[actual_clause_id]["hyde"] = hyde_scores[i]
        
        # Process label results
        if label_results:
            label_scores = self._normalize_scores([score for _, score in label_results])
            for i, (clause_id, _) in enumerate(label_results):
                actual_clause_id = clause_id.replace("content_", "").replace("label_", "").replace("alias_", "")
                if actual_clause_id in self.clause_store:
                    if actual_clause_id not in all_candidates:
                        all_candidates[actual_clause_id] = {"sparse": 0, "dense": 0, "hyde": 0, "label": 0}
                    all_candidates[actual_clause_id]["label"] = label_scores[i]
        
        # Compute combined scores
        weights = settings.retrieval_weights
        candidates = []
        
        for clause_id, scores in all_candidates.items():
            clause_record = self.clause_store[clause_id]
            
            # Calculate combined score
            combined_score = (
                weights["sparse"] * scores["sparse"] +
                weights["dense"] * scores["dense"] + 
                weights["label"] * scores["label"] +
                0.1 * scores["hyde"]  # HyDE bonus
            )
            
            # Create source spans
            spans = [SourceSpan(
                span_id=f"{clause_record.clause_id}:full",
                doc_id=clause_record.doc_id,
                page=clause_record.page,
                line_start=clause_record.line_start,
                line_end=clause_record.line_end
            )]
            
            candidate = RetrievalCandidate(
                clause_id=clause_id,
                clause_record=clause_record,
                scores=scores,
                combined_score=combined_score,
                spans=spans
            )
            
            candidates.append(candidate)
        
        # Sort by combined score
        candidates.sort(key=lambda x: x.combined_score, reverse=True)
        return candidates
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]


# Factory function
def create_retriever() -> PolyVectorRetriever:
    """Create a new poly-vector retriever instance."""
    return PolyVectorRetriever()