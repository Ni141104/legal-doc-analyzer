"""
Generation Service
Handles RAG-based clause card generation with Gemini and constrained JSON output.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

# Google Cloud imports (will be properly imported in production)
try:
    from google.cloud import aiplatform
    from vertexai.generative_models import GenerativeModel, GenerationConfig
except ImportError:
    aiplatform = None
    GenerativeModel = None
    GenerationConfig = None

from ..models.schemas import (
    ClauseCard, ClauseMetadata, SimplifiedSentence, RiskFlag, 
    Recommendation, NormalizedTerms, ClauseType, RiskLevel,
    GenerationRequest
)
from ..models.config import settings

logger = logging.getLogger(__name__)


class GenerationService:
    """Service for generating clause cards using RAG with Gemini."""
    
    def __init__(self):
        """Initialize the generation service."""
        self.model = None
        self.generation_config = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemini model and configuration."""
        try:
            if aiplatform:
                aiplatform.init(
                    project=settings.GOOGLE_CLOUD_PROJECT,
                    location=settings.VERTEX_AI_LOCATION
                )
                
                # Initialize generation config
                self.generation_config = GenerationConfig(
                    temperature=settings.GENERATION_TEMPERATURE,
                    max_output_tokens=settings.GENERATION_MAX_TOKENS,
                    top_p=settings.GENERATION_TOP_P
                )
                
                # Initialize model
                if GenerativeModel:
                    self.model = GenerativeModel(settings.VERTEX_GENERATION_MODEL)
            
            logger.info("Generation service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize generation service: {str(e)}")
    
    async def generate_clause_card(
        self, 
        clause_metadata: ClauseMetadata,
        evidence_spans: List[Dict[str, Any]],
        query_context: Optional[str] = None
    ) -> ClauseCard:
        """
        Generate a ClauseCard with simplified sentences and analysis.
        
        Args:
            clause_metadata: Source clause information
            evidence_spans: Supporting evidence spans
            query_context: Original user query for context
            
        Returns:
            Generated ClauseCard with all analysis
        """
        try:
            start_time = datetime.utcnow()
            
            # Prepare evidence spans for prompt
            evidence_text = self._format_evidence_spans(evidence_spans)
            
            # Generate the ClauseCard using constrained JSON generation
            clause_card_json = await self._generate_constrained_json(
                clause_metadata, evidence_text, query_context
            )
            
            # Parse and validate JSON response
            if isinstance(clause_card_json, str):
                try:
                    clause_data = json.loads(clause_card_json)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON response from model")
                    return self._create_fallback_clause_card(clause_metadata)
            else:
                clause_data = clause_card_json
            
            # Create ClauseCard object
            clause_card = self._parse_clause_card_response(clause_data, clause_metadata)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Generated clause card for {clause_metadata.clause_id} in {duration:.2f}s")
            return clause_card
            
        except Exception as e:
            logger.error(f"Clause card generation failed: {str(e)}")
            return self._create_fallback_clause_card(clause_metadata)
    
    async def _generate_constrained_json(
        self, 
        clause_metadata: ClauseMetadata,
        evidence_text: str,
        query_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate ClauseCard JSON using constrained generation."""
        try:
            # Construct the RAG prompt
            prompt = self._build_rag_prompt(clause_metadata, evidence_text, query_context)
            
            # Generate response
            if self.model:
                response = await self._call_gemini_model(prompt)
            else:
                # Fallback for development
                response = self._generate_mock_response(clause_metadata)
            
            return response
            
        except Exception as e:
            logger.error(f"Constrained JSON generation failed: {str(e)}")
            return self._generate_mock_response(clause_metadata)
    
    def _build_rag_prompt(
        self, 
        clause_metadata: ClauseMetadata,
        evidence_text: str,
        query_context: Optional[str] = None
    ) -> str:
        """Build the RAG prompt for ClauseCard generation."""
        
        # Base system prompt
        system_prompt = '''System: You are a legal summarizer. You will be provided with EvidenceSpans (id + text). Produce EXACTLY one JSON object following the ClauseCard schema. Each simplified sentence MUST include 'source_spans' (ids of EvidenceSpans). If you cannot support a claim with given spans, DO NOT produce it. Use simple language (≈9th grade).

REQUIRED JSON Schema:
{
    "clause_id": "string",
    "simplified_sentences": [
        {
            "text": "Human-readable simplified sentence",
            "source_spans": ["span_id_1", "span_id_2"],
            "confidence": 0.95,
            "rationale": "Explanation of why this simplification is accurate"
        }
    ],
    "normalized_terms": {
        "amounts": {"rent": 25000, "deposit": 100000},
        "dates": {"start_date": "2024-01-01"},
        "parties": {"tenant": "John Doe", "landlord": "Jane Smith"},
        "durations": {"lease_term": 365},
        "percentages": {"late_fee": 2.5}
    },
    "risk_flags": [
        {
            "type": "DepositHigh",
            "level": "Medium",
            "explanation": "Deposit is 4x monthly rent, above market norm",
            "evidence_spans": ["span_id_1"],
            "mitigation": "Request deposit cap at 2 months rent"
        }
    ],
    "recommendations": [
        {
            "text": "Negotiate deposit reduction to 2 months rent",
            "actionable_redline": "Replace '4 months' with '2 months' in deposit clause",
            "priority": "Medium",
            "category": "Financial Terms"
        }
    ],
    "clause_type": "Deposit",
    "confidence_overall": 0.92
}'''
        
        # Add query context if provided
        context_section = ""
        if query_context:
            context_section = f"\nUser Query Context: {query_context}\n"
        
        # Evidence spans section
        evidence_section = f"\nEvidenceSpans:\n{evidence_text}\n"
        
        # Task instruction
        task_section = f'''
Task: Generate ClauseCard JSON for clause "{clause_metadata.clause_id}" of type "{clause_metadata.clause_type.value}".

Requirements:
1. Every simplified sentence MUST cite source_spans
2. Use extracted amounts/dates from clause: {json.dumps(clause_metadata.extracted_terms.dict(), indent=2)}
3. Identify specific risks relevant to clause type
4. Provide actionable recommendations
5. Output ONLY valid JSON, no additional text

JSON Response:'''
        
        # Combine all sections
        full_prompt = system_prompt + context_section + evidence_section + task_section
        
        return full_prompt
    
    async def _call_gemini_model(self, prompt: str) -> Dict[str, Any]:
        """Call Gemini model with the generation prompt."""
        try:
            if not self.model:
                raise ValueError("Gemini model not initialized")
            
            # Generate content
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
            )
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Try to extract JSON if wrapped in markdown or extra text
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]
            
            # Parse JSON
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"Gemini model call failed: {str(e)}")
            raise
    
    def _generate_mock_response(self, clause_metadata: ClauseMetadata) -> Dict[str, Any]:
        """Generate mock response for development/testing."""
        return {
            "clause_id": clause_metadata.clause_id,
            "simplified_sentences": [
                {
                    "text": f"This {clause_metadata.clause_type.value.lower()} clause contains important terms.",
                    "source_spans": [f"{clause_metadata.clause_id}_span_1"],
                    "confidence": 0.85,
                    "rationale": "Direct extraction from clause text"
                }
            ],
            "normalized_terms": clause_metadata.extracted_terms.dict(),
            "risk_flags": [
                {
                    "type": "Review Required",
                    "level": "Medium",
                    "explanation": "This clause requires human review for accuracy",
                    "evidence_spans": [f"{clause_metadata.clause_id}_span_1"],
                    "mitigation": "Consult with legal expert"
                }
            ],
            "recommendations": [
                {
                    "text": "Review this clause with a legal professional",
                    "actionable_redline": "Consider adding clarifying language",
                    "priority": "Medium",
                    "category": "Legal Review"
                }
            ],
            "clause_type": clause_metadata.clause_type.value,
            "confidence_overall": 0.75
        }
    
    def _format_evidence_spans(self, evidence_spans: List[Dict[str, Any]]) -> str:
        """Format evidence spans for the prompt."""
        formatted_spans = []
        for i, span in enumerate(evidence_spans):
            span_id = span.get("span_id", f"span_{i+1}")
            span_text = span.get("text", "")
            formatted_spans.append(f'{{id:"{span_id}", text:"{span_text[:500]}..."}}')
        
        return "[\n  " + ",\n  ".join(formatted_spans) + "\n]"
    
    def _parse_clause_card_response(
        self, 
        response_data: Dict[str, Any], 
        clause_metadata: ClauseMetadata
    ) -> ClauseCard:
        """Parse and validate the generated ClauseCard response."""
        try:
            # Parse simplified sentences
            simplified_sentences = []
            for sent_data in response_data.get("simplified_sentences", []):
                sentence = SimplifiedSentence(
                    text=sent_data.get("text", ""),
                    source_spans=sent_data.get("source_spans", []),
                    confidence=float(sent_data.get("confidence", 0.5)),
                    rationale=sent_data.get("rationale", ""),
                    verified=False
                )
                simplified_sentences.append(sentence)
            
            # Parse normalized terms
            normalized_terms_data = response_data.get("normalized_terms", {})
            normalized_terms = NormalizedTerms(
                amounts=normalized_terms_data.get("amounts", {}),
                dates=normalized_terms_data.get("dates", {}),
                parties=normalized_terms_data.get("parties", {}),
                durations=normalized_terms_data.get("durations", {}),
                percentages=normalized_terms_data.get("percentages", {})
            )
            
            # Parse risk flags
            risk_flags = []
            for risk_data in response_data.get("risk_flags", []):
                risk_level = RiskLevel.MEDIUM  # Default
                try:
                    risk_level = RiskLevel(risk_data.get("level", "Medium"))
                except ValueError:
                    pass
                
                risk_flag = RiskFlag(
                    type=risk_data.get("type", "Unknown"),
                    level=risk_level,
                    explanation=risk_data.get("explanation", ""),
                    evidence_spans=risk_data.get("evidence_spans", []),
                    mitigation=risk_data.get("mitigation")
                )
                risk_flags.append(risk_flag)
            
            # Parse recommendations
            recommendations = []
            for rec_data in response_data.get("recommendations", []):
                priority = RiskLevel.MEDIUM  # Default
                try:
                    priority = RiskLevel(rec_data.get("priority", "Medium"))
                except ValueError:
                    pass
                
                recommendation = Recommendation(
                    text=rec_data.get("text", ""),
                    actionable_redline=rec_data.get("actionable_redline"),
                    priority=priority,
                    category=rec_data.get("category", "General")
                )
                recommendations.append(recommendation)
            
            # Parse clause type
            clause_type = clause_metadata.clause_type
            try:
                if "clause_type" in response_data:
                    clause_type = ClauseType(response_data["clause_type"])
            except ValueError:
                pass
            
            # Create ClauseCard
            clause_card = ClauseCard(
                clause_id=clause_metadata.clause_id,
                simplified_sentences=simplified_sentences,
                normalized_terms=normalized_terms,
                risk_flags=risk_flags,
                recommendations=recommendations,
                clause_type=clause_type,
                confidence_overall=float(response_data.get("confidence_overall", 0.7)),
                model_version=f"{settings.VERTEX_GENERATION_MODEL}_v1"
            )
            
            return clause_card
            
        except Exception as e:
            logger.error(f"Failed to parse clause card response: {str(e)}")
            return self._create_fallback_clause_card(clause_metadata)
    
    def _create_fallback_clause_card(self, clause_metadata: ClauseMetadata) -> ClauseCard:
        """Create a fallback ClauseCard when generation fails."""
        return ClauseCard(
            clause_id=clause_metadata.clause_id,
            simplified_sentences=[
                SimplifiedSentence(
                    text="This clause requires manual review due to processing limitations.",
                    source_spans=[clause_metadata.clause_id],
                    confidence=0.3,
                    rationale="Fallback response due to generation failure",
                    verified=False
                )
            ],
            normalized_terms=clause_metadata.extracted_terms,
            risk_flags=[
                RiskFlag(
                    type="Processing Error",
                    level=RiskLevel.HIGH,
                    explanation="Automated processing failed, manual review required",
                    evidence_spans=[clause_metadata.clause_id]
                )
            ],
            recommendations=[
                Recommendation(
                    text="Have this clause reviewed by a legal professional",
                    priority=RiskLevel.HIGH,
                    category="Manual Review Required"
                )
            ],
            clause_type=clause_metadata.clause_type,
            confidence_overall=0.3,
            model_version="fallback"
        )
    
    async def generate_text(
        self, 
        prompt: str, 
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate text response for general use (e.g., HyDE queries)."""
        try:
            if not self.model:
                return "Mock generated text response"
            
            # Use custom parameters or defaults
            config = GenerationConfig(
                temperature=temperature or settings.GENERATION_TEMPERATURE,
                max_output_tokens=max_tokens or 512,
                top_p=settings.GENERATION_TOP_P
            )
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(prompt, generation_config=config)
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            return "Error: Text generation failed"
    
    async def batch_generate_clause_cards(
        self, 
        requests: List[GenerationRequest]
    ) -> List[ClauseCard]:
        """Generate multiple clause cards in batch."""
        try:
            tasks = []
            for request in requests:
                task = self.generate_clause_card(
                    request.clause_metadata,
                    request.evidence_spans,
                    request.query_context
                )
                tasks.append(task)
            
            # Execute in parallel with concurrency limit
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent generations
            
            async def limited_generate(task):
                async with semaphore:
                    return await task
            
            results = await asyncio.gather(*[limited_generate(task) for task in tasks])
            
            logger.info(f"Generated {len(results)} clause cards in batch")
            return results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            return []


# Global service instance
generation_service = GenerationService()