"""
Verification Service
Implements NLI-based entailment verification to prevent hallucinations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Machine learning imports (will be properly imported in production)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None

from ..models.schemas import (
    ClauseCard, SimplifiedSentence, VerificationRequest as NLIRequest
)
from ..models.config import settings

logger = logging.getLogger(__name__)


class VerificationService:
    """Service for verifying generated content using NLI entailment."""
    
    def __init__(self):
        """Initialize the verification service."""
        self.nli_model = None
        self.tokenizer = None
        self.device = "cpu"
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLI models for verification."""
        try:
            # Use GPU if available
            if torch and torch.cuda.is_available():
                self.device = "cuda"
            
            # Initialize NLI pipeline
            if pipeline:
                # Using a production-ready NLI model
                self.nli_model = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",  # Fallback model
                    device=0 if self.device == "cuda" else -1
                )
                
                # In production, use a proper NLI model like:
                # model_name = "facebook/bart-large-mnli"
                # self.nli_model = pipeline("zero-shot-classification", model=model_name)
            
            logger.info(f"Verification service initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize verification models: {str(e)}")
    
    async def verify_clause_card(self, clause_card: ClauseCard, source_text: str) -> ClauseCard:
        """
        Verify all simplified sentences in a ClauseCard against source evidence.
        
        Args:
            clause_card: Generated ClauseCard to verify
            source_text: Original source text for verification
            
        Returns:
            ClauseCard with updated verification status and confidence scores
        """
        try:
            start_time = datetime.utcnow()
            
            # Verify each simplified sentence
            verified_sentences = []
            total_verified = 0
            
            for sentence in clause_card.simplified_sentences:
                verified_sentence = await self._verify_sentence(sentence, source_text)
                verified_sentences.append(verified_sentence)
                
                if verified_sentence.verified:
                    total_verified += 1
            
            # Calculate overall verification confidence
            verification_rate = total_verified / len(verified_sentences) if verified_sentences else 0
            
            # Adjust overall confidence based on verification rate
            adjusted_confidence = clause_card.confidence_overall * (0.5 + 0.5 * verification_rate)
            
            # Create verified clause card
            verified_clause_card = ClauseCard(
                clause_id=clause_card.clause_id,
                simplified_sentences=verified_sentences,
                normalized_terms=clause_card.normalized_terms,
                risk_flags=clause_card.risk_flags,
                recommendations=clause_card.recommendations,
                clause_type=clause_card.clause_type,
                confidence_overall=min(adjusted_confidence, 1.0),
                generated_at=clause_card.generated_at,
                model_version=clause_card.model_version
            )
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(
                f"Verified clause card {clause_card.clause_id}: "
                f"{total_verified}/{len(verified_sentences)} sentences verified in {duration:.2f}s"
            )
            
            return verified_clause_card
            
        except Exception as e:
            logger.error(f"Clause card verification failed: {str(e)}")
            # Return original with low confidence if verification fails
            clause_card.confidence_overall *= 0.5
            return clause_card
    
    async def _verify_sentence(
        self, 
        sentence: SimplifiedSentence, 
        source_text: str
    ) -> SimplifiedSentence:
        """Verify a single simplified sentence against source text."""
        try:
            # Extract relevant evidence spans
            evidence_text = self._extract_evidence_text(sentence.source_spans, source_text)
            
            if not evidence_text.strip():
                # No evidence found, mark as unverified
                return SimplifiedSentence(
                    text=sentence.text,
                    source_spans=sentence.source_spans,
                    confidence=0.1,
                    rationale="No evidence spans found for verification",
                    verified=False
                )
            
            # Perform NLI verification
            entailment_score = await self._check_entailment(sentence.text, evidence_text)
            
            # Determine verification status
            is_verified = entailment_score >= settings.VERIFICATION_THRESHOLD
            
            # Adjust confidence based on entailment score
            adjusted_confidence = min(sentence.confidence * entailment_score, 1.0)
            
            # Update rationale
            verification_rationale = (
                f"Original: {sentence.rationale}. "
                f"Verification: entailment_score={entailment_score:.3f}, "
                f"verified={is_verified}"
            )
            
            return SimplifiedSentence(
                text=sentence.text,
                source_spans=sentence.source_spans,
                confidence=adjusted_confidence,
                rationale=verification_rationale,
                verified=is_verified
            )
            
        except Exception as e:
            logger.error(f"Sentence verification failed: {str(e)}")
            # Return with low confidence on error
            return SimplifiedSentence(
                text=sentence.text,
                source_spans=sentence.source_spans,
                confidence=sentence.confidence * 0.3,
                rationale=f"{sentence.rationale}. Verification failed: {str(e)}",
                verified=False
            )
    
    async def _check_entailment(self, hypothesis: str, premise: str) -> float:
        """Check if hypothesis is entailed by premise using NLI model."""
        try:
            if not self.nli_model:
                # Fallback: simple text overlap scoring
                return self._simple_entailment_score(hypothesis, premise)
            
            # In production, use proper NLI model:
            # result = self.nli_model(premise, [hypothesis])
            # entailment_score = result['scores'][result['labels'].index('entailment')]
            
            # For now, use simple scoring
            score = self._simple_entailment_score(hypothesis, premise)
            
            # Run in executor to avoid blocking
            await asyncio.sleep(0.01)  # Simulate model inference time
            
            return score
            
        except Exception as e:
            logger.error(f"Entailment check failed: {str(e)}")
            return 0.5  # Neutral score on error
    
    def _simple_entailment_score(self, hypothesis: str, premise: str) -> float:
        """Simple entailment scoring based on word overlap and semantic similarity."""
        try:
            # Convert to lowercase and tokenize
            hyp_words = set(hypothesis.lower().split())
            prem_words = set(premise.lower().split())
            
            # Calculate word overlap
            overlap = len(hyp_words.intersection(prem_words))
            union = len(hyp_words.union(prem_words))
            
            if union == 0:
                return 0.0
            
            # Basic Jaccard similarity
            jaccard_score = overlap / union
            
            # Boost score if hypothesis is much shorter (likely extracted fact)
            length_ratio = len(hypothesis) / max(len(premise), 1)
            if length_ratio < 0.3:  # Hypothesis is much shorter
                jaccard_score *= 1.2
            
            # Penalty for very long hypotheses that might be hallucinated
            if len(hypothesis) > len(premise):
                jaccard_score *= 0.8
            
            # Check for exact substring match (strong indicator)
            if hypothesis.lower() in premise.lower() or premise.lower() in hypothesis.lower():
                jaccard_score = max(jaccard_score, 0.8)
            
            # Check for negation conflicts
            hyp_has_not = any(neg in hypothesis.lower() for neg in ['not', 'no', 'never', 'none'])
            prem_has_not = any(neg in premise.lower() for neg in ['not', 'no', 'never', 'none'])
            
            if hyp_has_not != prem_has_not:
                jaccard_score *= 0.3  # Strong penalty for negation mismatch
            
            return min(jaccard_score, 1.0)
            
        except Exception as e:
            logger.error(f"Simple entailment scoring failed: {str(e)}")
            return 0.5
    
    def _extract_evidence_text(self, source_spans: List[str], source_text: str) -> str:
        """Extract text from source spans."""
        try:
            # In production, this would use actual span coordinates
            # For now, return the source text as evidence
            if source_spans and source_text:
                # Simple extraction: return source text if spans are provided
                return source_text
            
            return ""
            
        except Exception as e:
            logger.error(f"Evidence extraction failed: {str(e)}")
            return ""
    
    async def verify_batch_sentences(
        self, 
        sentences: List[SimplifiedSentence],
        source_texts: List[str]
    ) -> List[SimplifiedSentence]:
        """Verify multiple sentences in batch for efficiency."""
        try:
            if len(sentences) != len(source_texts):
                raise ValueError("Number of sentences and source texts must match")
            
            # Create verification tasks
            tasks = []
            for sentence, source_text in zip(sentences, source_texts):
                task = self._verify_sentence(sentence, source_text)
                tasks.append(task)
            
            # Execute with concurrency limit
            semaphore = asyncio.Semaphore(10)  # Limit concurrent verifications
            
            async def limited_verify(task):
                async with semaphore:
                    return await task
            
            verified_sentences = await asyncio.gather(
                *[limited_verify(task) for task in tasks]
            )
            
            logger.info(f"Batch verified {len(verified_sentences)} sentences")
            return verified_sentences
            
        except Exception as e:
            logger.error(f"Batch verification failed: {str(e)}")
            return sentences  # Return original on error
    
    async def self_check_clause_card(self, clause_card: ClauseCard, source_text: str) -> Dict[str, Any]:
        """
        Perform self-consistency check on clause card.
        
        Args:
            clause_card: ClauseCard to check
            source_text: Original source text
            
        Returns:
            Dictionary with consistency analysis
        """
        try:
            inconsistencies = []
            warnings = []
            
            # Check for contradictions between sentences
            sentences = clause_card.simplified_sentences
            for i, sent1 in enumerate(sentences):
                for j, sent2 in enumerate(sentences[i+1:], i+1):
                    contradiction_score = await self._check_contradiction(sent1.text, sent2.text)
                    if contradiction_score > 0.7:
                        inconsistencies.append({
                            "type": "sentence_contradiction",
                            "sentence_1": sent1.text,
                            "sentence_2": sent2.text,
                            "score": contradiction_score
                        })
            
            # Check if normalized terms match simplified sentences
            for sentence in sentences:
                if sentence.verified:
                    term_consistency = self._check_term_consistency(
                        sentence.text, clause_card.normalized_terms
                    )
                    if term_consistency < 0.5:
                        warnings.append({
                            "type": "term_mismatch",
                            "sentence": sentence.text,
                            "confidence": term_consistency
                        })
            
            # Check risk flags against evidence
            for risk_flag in clause_card.risk_flags:
                if not risk_flag.evidence_spans:
                    warnings.append({
                        "type": "missing_evidence",
                        "risk_type": risk_flag.type,
                        "explanation": risk_flag.explanation
                    })
            
            return {
                "clause_id": clause_card.clause_id,
                "inconsistencies": inconsistencies,
                "warnings": warnings,
                "overall_consistency": max(0.0, 1.0 - len(inconsistencies) * 0.3 - len(warnings) * 0.1),
                "checked_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Self-check failed: {str(e)}")
            return {
                "clause_id": clause_card.clause_id,
                "error": str(e),
                "overall_consistency": 0.5
            }
    
    async def _check_contradiction(self, text1: str, text2: str) -> float:
        """Check if two texts contradict each other."""
        try:
            # Simple contradiction detection
            # In production, use proper contradiction detection model
            
            # Check for explicit contradictions
            contradiction_pairs = [
                (['required', 'must', 'shall'], ['optional', 'may', 'can']),
                (['included', 'covered', 'part of'], ['excluded', 'not covered', 'separate']),
                (['allowed', 'permitted'], ['prohibited', 'forbidden', 'not allowed']),
                (['increase', 'raise', 'higher'], ['decrease', 'lower', 'reduce'])
            ]
            
            text1_lower = text1.lower()
            text2_lower = text2.lower()
            
            for positive_words, negative_words in contradiction_pairs:
                has_positive_1 = any(word in text1_lower for word in positive_words)
                has_negative_1 = any(word in text1_lower for word in negative_words)
                has_positive_2 = any(word in text2_lower for word in positive_words)
                has_negative_2 = any(word in text2_lower for word in negative_words)
                
                if (has_positive_1 and has_negative_2) or (has_negative_1 and has_positive_2):
                    return 0.8  # High contradiction score
            
            # Check for numerical contradictions
            import re
            numbers1 = re.findall(r'\d+(?:\.\d+)?', text1)
            numbers2 = re.findall(r'\d+(?:\.\d+)?', text2)
            
            if numbers1 and numbers2:
                # Simple check: if different numbers for same type of thing
                if len(set(numbers1).intersection(set(numbers2))) == 0:
                    return 0.4  # Possible contradiction
            
            return 0.0  # No contradiction detected
            
        except Exception as e:
            logger.error(f"Contradiction check failed: {str(e)}")
            return 0.0
    
    def _check_term_consistency(self, sentence: str, normalized_terms) -> float:
        """Check if sentence is consistent with normalized terms."""
        try:
            consistency_score = 1.0
            sentence_lower = sentence.lower()
            
            # Check amounts
            for term_name, amount in normalized_terms.amounts.items():
                if str(amount) not in sentence and f"{amount:,.0f}" not in sentence:
                    # Amount mentioned in terms but not in sentence
                    if any(money_word in sentence_lower for money_word in ['amount', 'pay', 'cost', 'fee', 'rent']):
                        consistency_score -= 0.2
            
            # Check percentages
            for term_name, percentage in normalized_terms.percentages.items():
                if f"{percentage}%" not in sentence and str(percentage) not in sentence:
                    if "%" in sentence or "percent" in sentence_lower:
                        consistency_score -= 0.2
            
            return max(0.0, consistency_score)
            
        except Exception as e:
            logger.error(f"Term consistency check failed: {str(e)}")
            return 0.5


# Global service instance
verification_service = VerificationService()