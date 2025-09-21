"""
Constrained RAG engine with Gemini for generating ClauseCards.
Implements JSON-only output with mandatory source citations.
"""

import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio

from ..models.schemas import (
    ClauseCard, SimplifiedSentence, RiskFlag, Recommendation,
    RiskLevel, SourceSpan
)
from ..models.config import settings
from ..retrieval.poly_vector import RetrievalResult, RetrievalCandidate

logger = logging.getLogger(__name__)


class PromptTemplates:
    """Centralized prompt templates for different generation tasks."""
    
    HYDE_PROMPT = """System: You are a retrieval-helper. Given the user query, produce a concise hypothetical document (1-3 sentences) that best answers the query. This text will be embedded for retrieval only. Do NOT invent statutes or cases.

User: "{query}"

Generate a hypothetical legal document snippet that would contain the answer:"""
    
    CLAUSE_CARD_PROMPT = """System: You are a legal summarizer. You will be provided with EvidenceSpans (id + text). Produce EXACTLY one JSON object following the ClauseCard schema. Each simplified sentence MUST include 'source_spans' (ids of EvidenceSpans). If you cannot support a claim with given spans, DO NOT produce it. Use simple language (≈9th grade).

Important rules:
1. Output ONLY valid JSON - no other text
2. Every simplified_sentence must have source_spans from the provided evidence
3. Use simple, clear language that a non-lawyer can understand
4. Focus on practical implications and actionable information
5. Confidence scores should reflect how well the evidence supports each sentence
6. Risk flags should identify potential issues for the user
7. Recommendations should be specific and actionable

ClauseCard JSON Schema:
{{
  "clause_id": "string",
  "simplified_sentences": [
    {{
      "text": "string",
      "source_spans": ["span_id1", "span_id2"],
      "confidence": float (0.0-1.0),
      "confidence_level": "very_high|high|medium|low|very_low",
      "rationale": "string",
      "verified": false,
      "verifier_score": null
    }}
  ],
  "normalized_terms": {{
    "key": "value"
  }},
  "risk_flags": [
    {{
      "flag_type": "string",
      "level": "Low|Medium|High|Critical",
      "explanation": "string", 
      "evidence_spans": ["span_id"],
      "actionable": true|false,
      "statutory_reference": "string or null"
    }}
  ],
  "recommendations": [
    {{
      "text": "string",
      "priority": "Low|Medium|High|Critical",
      "actionable_redline": "string or null",
      "estimated_impact": "string or null",
      "legal_basis": "string or null"
    }}
  ],
  "generated_at": "{timestamp}",
  "model_version": "{model_version}",
  "retrieval_set_ids": {retrieval_ids},
  "generator_prompt_hash": "{prompt_hash}",
  "overall_confidence": float (0.0-1.0)
}}

EvidenceSpans:
{evidence_spans}

Task: Generate ClauseCard JSON for the clause analysis."""

    SELF_CHECK_PROMPT = """System: You are a verifier. Given ClauseCard (simplified_sentences + evidence spans), for each sentence return:
{{sentence_idx:int, supported:bool, counter_evidence: [span_ids if any]}}.
If unsure, supported=false.

ClauseCard to verify:
{clause_card_json}

Evidence spans:
{evidence_spans}

For each simplified sentence, determine if it is fully supported by the cited evidence spans. Return ONLY a JSON array of verification results:"""


class GeminiGenerator:
    """Gemini-based generator with constrained JSON output."""
    
    def __init__(self):
        self._client = None
        self.model_version = settings.GEMINI_MODEL
        self.generation_config = settings.gemini_config
    
    @property
    def client(self):
        """Lazy initialization of Vertex AI Gemini client."""
        if self._client is None:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel
                
                vertexai.init(
                    project=settings.GCP_PROJECT_ID,
                    location=settings.VERTEX_AI_LOCATION
                )
                self._client = GenerativeModel(self.model_version)
            except ImportError:
                logger.error("Vertex AI SDK not available")
                raise
        
        return self._client
    
    async def generate_clause_card(
        self, 
        retrieval_result: RetrievalResult,
        query: str
    ) -> Optional[ClauseCard]:
        """Generate a ClauseCard from retrieval results."""
        
        if not retrieval_result.candidates:
            logger.warning("No candidates for ClauseCard generation")
            return None
        
        try:
            # Prepare evidence spans
            evidence_spans = self._prepare_evidence_spans(retrieval_result.candidates)
            
            # Generate prompt
            prompt = self._build_clause_card_prompt(evidence_spans)
            
            # Calculate prompt hash for auditing
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            
            # Generate with Gemini
            response = await self._generate_with_retry(prompt)
            
            if not response:
                return None
            
            # Parse JSON response
            clause_card_data = self._parse_json_response(response)
            
            if not clause_card_data:
                return None
            
            # Enhance with metadata
            clause_card_data.update({
                "generated_at": datetime.utcnow().isoformat(),
                "model_version": self.model_version,
                "retrieval_set_ids": [c.clause_id for c in retrieval_result.candidates],
                "generator_prompt_hash": prompt_hash
            })
            
            # Validate and create ClauseCard
            try:
                clause_card = ClauseCard(**clause_card_data)
                logger.info(f"Generated ClauseCard for query: {query[:50]}...")
                return clause_card
            
            except Exception as e:
                logger.error(f"Failed to create ClauseCard from generated data: {e}")
                return None
        
        except Exception as e:
            logger.error(f"ClauseCard generation failed: {e}")
            return None
    
    async def generate_hyde_document(self, query: str) -> str:
        """Generate HyDE pseudo-document."""
        try:
            prompt = PromptTemplates.HYDE_PROMPT.format(query=query)
            response = await self._generate_with_retry(prompt, max_tokens=150)
            return response.strip() if response else query
        
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return query
    
    async def self_check_clause_card(
        self, 
        clause_card: ClauseCard, 
        evidence_spans: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Perform self-check verification of generated content."""
        try:
            clause_card_json = clause_card.model_dump_json()
            evidence_text = json.dumps(evidence_spans, indent=2)
            
            prompt = PromptTemplates.SELF_CHECK_PROMPT.format(
                clause_card_json=clause_card_json,
                evidence_spans=evidence_text
            )
            
            response = await self._generate_with_retry(prompt, max_tokens=500)
            
            if response:
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    logger.warning("Self-check returned invalid JSON")
                    return []
            
            return []
        
        except Exception as e:
            logger.error(f"Self-check failed: {e}")
            return []
    
    async def _generate_with_retry(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        max_retries: int = 3
    ) -> Optional[str]:
        """Generate with retry logic."""
        
        config = self.generation_config.copy()
        if max_tokens:
            config["max_output_tokens"] = max_tokens
        
        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config=config
                )
                
                if response and response.text:
                    return response.text.strip()
                else:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
            
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All generation attempts failed for prompt: {prompt[:100]}...")
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        return None
    
    def _prepare_evidence_spans(self, candidates: List[RetrievalCandidate]) -> Dict[str, str]:
        """Prepare evidence spans for prompt."""
        evidence = {}
        
        for candidate in candidates:
            clause = candidate.clause_record
            
            # Create span ID
            span_id = f"{clause.clause_id}:full"
            
            # Combine section title and content
            content = ""
            if clause.section_title:
                content += f"Section: {clause.section_title}\n"
            content += clause.content_text
            
            evidence[span_id] = content
        
        return evidence
    
    def _build_clause_card_prompt(self, evidence_spans: Dict[str, str]) -> str:
        """Build the ClauseCard generation prompt."""
        
        # Format evidence spans for prompt
        evidence_text = ""
        for span_id, content in evidence_spans.items():
            evidence_text += f"{{\"id\": \"{span_id}\", \"text\": \"{content[:1000]}...\"}}\n"
        
        # Get retrieval IDs and generate other metadata
        retrieval_ids = list(evidence_spans.keys())
        timestamp = datetime.utcnow().isoformat()
        
        # Calculate prompt hash (will be updated after full prompt is built)
        temp_prompt = PromptTemplates.CLAUSE_CARD_PROMPT.format(
            evidence_spans=evidence_text,
            timestamp=timestamp,
            model_version=self.model_version,
            retrieval_ids=json.dumps(retrieval_ids),
            prompt_hash="PLACEHOLDER"
        )
        
        # Calculate actual hash
        prompt_hash = hashlib.md5(temp_prompt.encode()).hexdigest()
        
        # Final prompt with correct hash
        return PromptTemplates.CLAUSE_CARD_PROMPT.format(
            evidence_spans=evidence_text,
            timestamp=timestamp,
            model_version=self.model_version,
            retrieval_ids=json.dumps(retrieval_ids),
            prompt_hash=prompt_hash
        )
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse and validate JSON response from Gemini."""
        try:
            # Clean response (remove any markdown formatting)
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Parse JSON
            parsed = json.loads(cleaned)
            
            # Basic validation
            required_fields = [
                "clause_id", "simplified_sentences", "normalized_terms",
                "risk_flags", "recommendations"
            ]
            
            for field in required_fields:
                if field not in parsed:
                    logger.error(f"Missing required field: {field}")
                    return None
            
            # Validate simplified sentences have source_spans
            for i, sentence in enumerate(parsed.get("simplified_sentences", [])):
                if not sentence.get("source_spans"):
                    logger.error(f"Sentence {i} missing source_spans")
                    return None
            
            return parsed
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            logger.debug(f"Response text: {response}")
            return None
        
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None


class RiskAnalyzer:
    """Analyze clauses for potential risks and generate flags."""
    
    RISK_PATTERNS = {
        "DepositHigh": {
            "patterns": [r"deposit.*(\d+)", r"advance.*(\d+)"],
            "threshold_check": lambda amount: amount > 100000,  # > 1 lakh
            "level": RiskLevel.MEDIUM,
            "explanation": "Security deposit amount is higher than typical market rates"
        },
        "InterestRateHigh": {
            "patterns": [r"interest.*(\d+(?:\.\d+)?)\s*%"],
            "threshold_check": lambda rate: rate > 18.0,  # > 18% annual
            "level": RiskLevel.HIGH,
            "explanation": "Interest rate exceeds typical lending rates"
        },
        "NoticePeriodShort": {
            "patterns": [r"notice.*(\d+)\s*days?"],
            "threshold_check": lambda days: days < 30,
            "level": RiskLevel.MEDIUM,
            "explanation": "Notice period is shorter than recommended minimum"
        },
        "TerminationAbusive": {
            "patterns": [r"terminate.*without.*notice", r"immediate.*termination"],
            "threshold_check": lambda _: True,
            "level": RiskLevel.HIGH,
            "explanation": "Termination clause may be unfavorable to tenant/borrower"
        }
    }
    
    @classmethod
    def analyze_risks(cls, clause_text: str, extracted_facts: List[Any]) -> List[RiskFlag]:
        """Analyze clause for risks."""
        risks = []
        
        for risk_type, config in cls.RISK_PATTERNS.items():
            risk = cls._check_risk_pattern(
                clause_text, extracted_facts, risk_type, config
            )
            if risk:
                risks.append(risk)
        
        return risks
    
    @classmethod
    def _check_risk_pattern(
        cls, 
        text: str, 
        facts: List[Any], 
        risk_type: str, 
        config: Dict[str, Any]
    ) -> Optional[RiskFlag]:
        """Check specific risk pattern."""
        import re
        
        try:
            for pattern in config["patterns"]:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Extract value if pattern has capture group
                    if match.groups():
                        value = float(match.group(1).replace(',', ''))
                        if config["threshold_check"](value):
                            return RiskFlag(
                                flag_type=risk_type,
                                level=config["level"],
                                explanation=config["explanation"],
                                evidence_spans=[f"pattern_match:{match.start()}-{match.end()}"],
                                actionable=True,
                                statutory_reference=None
                            )
                    else:
                        # Pattern with no capture groups
                        if config["threshold_check"](None):
                            return RiskFlag(
                                flag_type=risk_type,
                                level=config["level"],
                                explanation=config["explanation"],
                                evidence_spans=[f"pattern_match:{match.start()}-{match.end()}"],
                                actionable=True,
                                statutory_reference=None
                            )
        
        except Exception as e:
            logger.warning(f"Risk analysis failed for {risk_type}: {e}")
        
        return None


class RecommendationEngine:
    """Generate actionable recommendations based on clause analysis."""
    
    RECOMMENDATION_RULES = {
        "DepositHigh": {
            "text": "Request security deposit be reduced to 2 months rent (typical market rate)",
            "priority": RiskLevel.MEDIUM,
            "actionable_redline": "Replace 'security deposit of [amount]' with 'security deposit equivalent to two months rent'",
            "legal_basis": "Most rental laws recommend 2-3 months rent as maximum deposit"
        },
        "InterestRateHigh": {
            "text": "Negotiate for lower interest rate closer to market rates (12-15% for personal loans)",
            "priority": RiskLevel.HIGH,
            "actionable_redline": "Request amendment to reduce interest rate to market competitive levels",
            "legal_basis": "Interest rates should be reasonable and not excessive per banking regulations"
        },
        "NoticePeriodShort": {
            "text": "Request minimum 30 days notice period for termination",
            "priority": RiskLevel.MEDIUM,
            "actionable_redline": "Change notice period from [current] to 'minimum 30 days written notice'",
            "legal_basis": "30 days is standard practice for most rental agreements"
        },
        "TerminationAbusive": {
            "text": "Add mutual termination clauses with proper notice requirements",
            "priority": RiskLevel.HIGH,
            "actionable_redline": "Add: 'Either party may terminate with 30 days written notice'",
            "legal_basis": "Termination clauses should be mutual and fair to both parties"
        }
    }
    
    @classmethod
    def generate_recommendations(cls, risk_flags: List[RiskFlag]) -> List[Recommendation]:
        """Generate recommendations based on identified risks."""
        recommendations = []
        
        for risk in risk_flags:
            if risk.flag_type in cls.RECOMMENDATION_RULES:
                rule = cls.RECOMMENDATION_RULES[risk.flag_type]
                
                recommendation = Recommendation(
                    text=rule["text"],
                    priority=rule["priority"],
                    actionable_redline=rule["actionable_redline"],
                    estimated_impact="Reduced risk and better terms",
                    legal_basis=rule["legal_basis"]
                )
                
                recommendations.append(recommendation)
        
        return recommendations


class ConstrainedRAGEngine:
    """Main RAG engine coordinating generation, risk analysis, and recommendations."""
    
    def __init__(self):
        self.generator = GeminiGenerator()
        self.risk_analyzer = RiskAnalyzer()
        self.recommendation_engine = RecommendationEngine()
    
    async def generate_clause_card(
        self, 
        retrieval_result: RetrievalResult,
        query: str
    ) -> Optional[ClauseCard]:
        """Generate complete ClauseCard with analysis."""
        
        try:
            # Generate base ClauseCard
            clause_card = await self.generator.generate_clause_card(retrieval_result, query)
            
            if not clause_card:
                return None
            
            # Enhance with risk analysis and recommendations
            if retrieval_result.candidates:
                primary_candidate = retrieval_result.candidates[0]
                clause_text = primary_candidate.clause_record.content_text
                extracted_facts = primary_candidate.clause_record.extracted_facts
                
                # Analyze risks
                risks = self.risk_analyzer.analyze_risks(clause_text, extracted_facts)
                
                # Generate recommendations
                recommendations = self.recommendation_engine.generate_recommendations(risks)
                
                # Update clause card
                clause_card.risk_flags.extend(risks)
                clause_card.recommendations.extend(recommendations)
                
                # Recalculate overall confidence based on risks
                if risks:
                    high_risk_count = sum(1 for risk in risks if risk.level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
                    confidence_penalty = min(0.3, high_risk_count * 0.1)
                    clause_card.overall_confidence = max(0.1, clause_card.overall_confidence - confidence_penalty)
            
            return clause_card
        
        except Exception as e:
            logger.error(f"RAG engine generation failed: {e}")
            return None
    
    async def validate_clause_card(self, clause_card: ClauseCard, evidence_spans: Dict[str, str]) -> ClauseCard:
        """Validate ClauseCard using self-check."""
        try:
            # Perform self-check
            verification_results = await self.generator.self_check_clause_card(
                clause_card, evidence_spans
            )
            
            # Update sentences with verification results
            for result in verification_results:
                sentence_idx = result.get("sentence_idx", -1)
                if 0 <= sentence_idx < len(clause_card.simplified_sentences):
                    sentence = clause_card.simplified_sentences[sentence_idx]
                    sentence.verified = result.get("supported", False)
                    
                    # Reduce confidence for unverified sentences
                    if not sentence.verified:
                        sentence.confidence *= 0.5
            
            # Recalculate overall confidence
            if clause_card.simplified_sentences:
                verified_count = sum(1 for s in clause_card.simplified_sentences if s.verified)
                verification_rate = verified_count / len(clause_card.simplified_sentences)
                clause_card.overall_confidence *= verification_rate
            
            return clause_card
        
        except Exception as e:
            logger.error(f"ClauseCard validation failed: {e}")
            return clause_card


# Factory function
def create_rag_engine() -> ConstrainedRAGEngine:
    """Create a new constrained RAG engine instance."""
    return ConstrainedRAGEngine()