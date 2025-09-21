"""
RAG Generation Service for MVP
Handles Gemini-based RAG with HyDE for hackathon prototype.
Single-user system - focused on core AI pipeline.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Google Cloud / Vertex AI imports
try:
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    import vertexai
except ImportError:
    GenerativeModel = None
    GenerationConfig = None
    vertexai = None

from ..models.mvp_schemas import (
    RAGAnswer, HyDEDocument, VectorSearchResult, ExtractedClause, ClauseType
)
from ..models.config import settings

logger = logging.getLogger(__name__)


class RAGGenerationService:
    """
    Simplified RAG generation for hackathon MVP.
    
    Pipeline:
    1. HyDE: Generate hypothetical documents from query
    2. Vector Search: Find relevant clauses using query + HyDE
    3. RAG: Generate answer using Gemini with retrieved context
    4. Simple verification: Check answer consistency
    """
    
    def __init__(self):
        """Initialize Gemini model."""
        self.gemini_model = None
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Vertex AI Gemini model."""
        try:
            if vertexai:
                vertexai.init(
                    project=settings.google_cloud_project_id,
                    location=settings.vertex_ai_location
                )
                
                self.gemini_model = GenerativeModel(settings.llm_model_name)
                logger.info(f"Gemini model initialized: {settings.llm_model_name}")
            else:
                logger.warning("Vertex AI not available, using mock responses")
                
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
    
    async def generate_hyde_document(self, query: str) -> HyDEDocument:
        """
        Generate hypothetical document using HyDE technique.
        
        Args:
            query: User's question
            
        Returns:
            HyDEDocument with hypothetical answer
        """
        try:
            if not settings.hyde_enabled:
                return HyDEDocument(
                    query=query,
                    hypothetical_text="",
                    generated_at=datetime.utcnow()
                )
            
            # Create HyDE prompt
            hyde_prompt = f"""
You are a legal expert. Generate a hypothetical answer to this question about a legal document:

Question: {query}

Generate a detailed, realistic answer as if you were directly quoting from a legal document. Include specific terms, conditions, and legal language that would typically appear in such clauses.

Hypothetical Answer:"""
            
            # Generate hypothetical document
            hypothetical_text = await self._call_gemini(
                hyde_prompt,
                temperature=settings.hyde_temperature,
                max_tokens=settings.hyde_max_tokens
            )
            
            hyde_doc = HyDEDocument(
                query=query,
                hypothetical_text=hypothetical_text,
                generated_at=datetime.utcnow()
            )
            
            logger.info(f"Generated HyDE document for query: {query[:50]}...")
            return hyde_doc
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {str(e)}")
            return HyDEDocument(
                query=query,
                hypothetical_text="",
                generated_at=datetime.utcnow()
            )
    
    async def generate_rag_answer(
        self,
        query: str,
        retrieved_clauses: List[VectorSearchResult],
        hyde_document: Optional[HyDEDocument] = None
    ) -> RAGAnswer:
        """
        Generate final answer using RAG with retrieved clauses.
        
        Args:
            query: User's question
            retrieved_clauses: Relevant clauses from vector search
            hyde_document: Optional HyDE document for context
            
        Returns:
            RAGAnswer with generated response and metadata
        """
        try:
            # Build context from retrieved clauses
            context_parts = []
            for i, clause in enumerate(retrieved_clauses, 1):
                context_parts.append(f"""
Document {clause.doc_id} - Clause {i} ({clause.clause_type.value}):
{clause.text}
[Relevance Score: {clause.similarity_score:.3f}]
""")
            
            context = "\n".join(context_parts)
            
            # Include HyDE if available
            hyde_context = ""
            if hyde_document and hyde_document.hypothetical_text:
                hyde_context = f"\nHypothetical Reference:\n{hyde_document.hypothetical_text}\n"
            
            # Create RAG prompt
            rag_prompt = f"""
You are a legal assistant helping users understand legal documents. Answer the user's question based ONLY on the provided legal clauses.

User Question: {query}

Legal Document Clauses:
{context}
{hyde_context}

Instructions:
1. Answer in simple, clear language that a non-lawyer can understand
2. Base your answer ONLY on the provided clauses
3. If the clauses don't contain enough information, say so clearly
4. Include specific references to relevant clauses
5. Highlight any important conditions, deadlines, or requirements
6. Keep the answer concise but complete

Answer:"""
            
            # Generate answer
            answer_text = await self._call_gemini(
                rag_prompt,
                temperature=settings.rag_temperature,
                max_tokens=settings.rag_max_output_tokens
            )
            
            # Calculate confidence based on clause relevance scores
            confidence = await self._calculate_answer_confidence(
                retrieved_clauses, answer_text
            )
            
            # Simple verification
            verification_score = await self._verify_answer_consistency(
                query, answer_text, retrieved_clauses
            )
            
            rag_answer = RAGAnswer(
                query=query,
                answer=answer_text,
                confidence=confidence,
                supporting_clauses=retrieved_clauses,
                hyde_used=hyde_document is not None and bool(hyde_document.hypothetical_text),
                verified=verification_score > 0.7,
                verification_score=verification_score,
                generated_at=datetime.utcnow()
            )
            
            logger.info(f"Generated RAG answer (confidence: {confidence:.3f}, verified: {rag_answer.verified})")
            return rag_answer
            
        except Exception as e:
            logger.error(f"RAG generation failed: {str(e)}")
            return RAGAnswer(
                query=query,
                answer=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                confidence=0.0,
                supporting_clauses=retrieved_clauses,
                hyde_used=False,
                verified=False,
                generated_at=datetime.utcnow()
            )
    
    async def _call_gemini(
        self, 
        prompt: str, 
        temperature: float = 0.3, 
        max_tokens: int = 1000
    ) -> str:
        """Call Gemini model with prompt."""
        try:
            if not self.gemini_model:
                # Fallback for development
                logger.warning("Gemini not available, using mock response")
                return f"Mock response for development: {prompt[:100]}..."
            
            # Configure generation
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.8,
                top_k=40
            )
            
            # Generate response
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract text from response
            generated_text = response.text.strip()
            
            logger.debug(f"Gemini response generated (length: {len(generated_text)})")
            return generated_text
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    async def _calculate_answer_confidence(
        self, 
        retrieved_clauses: List[VectorSearchResult], 
        answer_text: str
    ) -> float:
        """Calculate confidence score for the generated answer."""
        try:
            if not retrieved_clauses:
                return 0.1
            
            # Base confidence on clause relevance scores
            avg_relevance = sum(clause.similarity_score for clause in retrieved_clauses) / len(retrieved_clauses)
            
            # Adjust based on number of supporting clauses
            clause_count_factor = min(1.0, len(retrieved_clauses) / 3.0)
            
            # Adjust based on answer length (reasonable answers should have substance)
            length_factor = 0.5 if len(answer_text) < 100 else 1.0
            
            # Simple keyword overlap check
            answer_lower = answer_text.lower()
            keyword_matches = 0
            total_keywords = 0
            
            for clause in retrieved_clauses:
                clause_words = set(clause.text.lower().split())
                answer_words = set(answer_lower.split())
                matches = len(clause_words.intersection(answer_words))
                keyword_matches += matches
                total_keywords += len(clause_words)
            
            keyword_factor = min(1.0, keyword_matches / max(total_keywords, 1) * 10)
            
            # Combine factors
            confidence = (avg_relevance * 0.5 + 
                         clause_count_factor * 0.2 + 
                         length_factor * 0.1 + 
                         keyword_factor * 0.2)
            
            return min(0.95, max(0.05, confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.5
    
    async def _verify_answer_consistency(
        self,
        query: str,
        answer: str,
        retrieved_clauses: List[VectorSearchResult]
    ) -> float:
        """Simple verification of answer consistency with source clauses."""
        try:
            # Simple verification approach for MVP
            verification_prompt = f"""
Verify if this answer is consistent with the provided legal clauses. 

Question: {query}
Answer: {answer}

Source Clauses:
{chr(10).join([f"- {clause.text[:200]}..." for clause in retrieved_clauses[:3]])}

Is the answer factually consistent with the source clauses? Rate from 0.0 (completely inconsistent) to 1.0 (fully consistent).

Provide only the numeric score (e.g., 0.8):"""
            
            verification_response = await self._call_gemini(
                verification_prompt,
                temperature=0.1,
                max_tokens=10
            )
            
            # Extract numeric score
            try:
                score = float(verification_response.strip())
                return min(1.0, max(0.0, score))
            except ValueError:
                # Fallback: simple text analysis
                return self._simple_consistency_check(answer, retrieved_clauses)
                
        except Exception as e:
            logger.error(f"Answer verification failed: {str(e)}")
            return 0.5
    
    def _simple_consistency_check(
        self, 
        answer: str, 
        retrieved_clauses: List[VectorSearchResult]
    ) -> float:
        """Simple rule-based consistency check."""
        try:
            answer_lower = answer.lower()
            
            # Check for contradiction indicators
            contradiction_phrases = [
                "i don't have", "not mentioned", "cannot determine",
                "unclear", "insufficient information", "error"
            ]
            
            has_contradiction = any(phrase in answer_lower for phrase in contradiction_phrases)
            if has_contradiction:
                return 0.3
            
            # Check for positive indicators
            positive_phrases = [
                "according to", "the document states", "as specified",
                "the clause mentions", "it is stated that"
            ]
            
            has_positive = any(phrase in answer_lower for phrase in positive_phrases)
            if has_positive:
                return 0.8
            
            # Check word overlap with source clauses
            answer_words = set(answer_lower.split())
            clause_words = set()
            
            for clause in retrieved_clauses:
                clause_words.update(clause.text.lower().split())
            
            if clause_words:
                overlap_ratio = len(answer_words.intersection(clause_words)) / len(answer_words)
                return min(0.9, max(0.2, overlap_ratio))
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Simple consistency check failed: {str(e)}")
            return 0.5
    
    async def generate_clause_summary(self, clause: ExtractedClause) -> str:
        """Generate a simple summary of a legal clause."""
        try:
            summary_prompt = f"""
Summarize this legal clause in simple, clear language that anyone can understand:

Clause Type: {clause.clause_type.value}
Text: {clause.text}

Provide a 1-2 sentence summary explaining what this clause means in plain English:"""
            
            summary = await self._call_gemini(
                summary_prompt,
                temperature=0.2,
                max_tokens=200
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Clause summary generation failed: {str(e)}")
            return "Unable to generate summary for this clause."


# Global service instance
rag_generation_service = RAGGenerationService()