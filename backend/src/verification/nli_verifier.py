"""
NLI-based verification system for checking generated sentences against source spans.
Implements entailment checking to prevent hallucinations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio

from ..models.schemas import SimplifiedSentence, ClauseCard, SourceSpan
from ..models.config import settings

logger = logging.getLogger(__name__)


class EntailmentLabel(str, Enum):
    """NLI entailment labels."""
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"  
    NEUTRAL = "neutral"


@dataclass
class VerificationResult:
    """Result of sentence verification."""
    sentence_index: int
    sentence_text: str
    entailment_label: EntailmentLabel
    confidence_score: float
    supported: bool
    rationale: str
    counter_evidence: List[str] = None


@dataclass
class VerificationReport:
    """Complete verification report for a ClauseCard."""
    clause_card_id: str
    total_sentences: int
    verified_sentences: int
    failed_sentences: int
    overall_pass_rate: float
    sentence_results: List[VerificationResult]
    processing_time_ms: float


class NLIModel:
    """Abstract base class for NLI models."""
    
    async def predict_entailment(
        self, 
        premise: str, 
        hypothesis: str
    ) -> Tuple[EntailmentLabel, float]:
        """Predict entailment relationship."""
        raise NotImplementedError


class HuggingFaceNLIModel(NLIModel):
    """HuggingFace-based NLI model implementation."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                
                # Use GPU if available
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                
            except ImportError:
                logger.error("Transformers library not available")
                raise
        
        return self._model
    
    async def predict_entailment(
        self, 
        premise: str, 
        hypothesis: str
    ) -> Tuple[EntailmentLabel, float]:
        """Predict entailment using HuggingFace model."""
        try:
            import torch
            
            # Tokenize input
            inputs = self._tokenizer(
                premise, 
                hypothesis, 
                return_tensors="pt", 
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to GPU if model is on GPU
            if next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Get prediction
            predicted_label_id = torch.argmax(probs, dim=-1).item()
            confidence = float(probs[0][predicted_label_id])
            
            # Map to entailment labels (model-dependent mapping)
            label_mapping = {0: EntailmentLabel.CONTRADICTION, 1: EntailmentLabel.NEUTRAL, 2: EntailmentLabel.ENTAILMENT}
            predicted_label = label_mapping.get(predicted_label_id, EntailmentLabel.NEUTRAL)
            
            return predicted_label, confidence
        
        except Exception as e:
            logger.error(f"NLI prediction failed: {e}")
            return EntailmentLabel.NEUTRAL, 0.5


class VertexAINLIModel(NLIModel):
    """Vertex AI-based NLI using a fine-tuned model or Gemini."""
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Vertex AI client."""
        if self._client is None:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel
                
                vertexai.init(
                    project=settings.GCP_PROJECT_ID,
                    location=settings.VERTEX_AI_LOCATION
                )
                self._client = GenerativeModel(settings.GEMINI_MODEL)
            except ImportError:
                logger.error("Vertex AI SDK not available")
                raise
        
        return self._client
    
    async def predict_entailment(
        self, 
        premise: str, 
        hypothesis: str
    ) -> Tuple[EntailmentLabel, float]:
        """Predict entailment using Vertex AI."""
        try:
            prompt = f"""Analyze the relationship between these two statements:

Premise: "{premise}"
Hypothesis: "{hypothesis}"

Does the premise entail (logically support) the hypothesis?

Respond with EXACTLY one of:
- ENTAILMENT: The premise logically supports the hypothesis
- CONTRADICTION: The premise contradicts the hypothesis  
- NEUTRAL: The premise neither supports nor contradicts the hypothesis

Response:"""
            
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 50,
                    "top_p": 0.8
                }
            )
            
            if response and response.text:
                response_text = response.text.strip().upper()
                
                if "ENTAILMENT" in response_text:
                    return EntailmentLabel.ENTAILMENT, 0.85
                elif "CONTRADICTION" in response_text:
                    return EntailmentLabel.CONTRADICTION, 0.85
                else:
                    return EntailmentLabel.NEUTRAL, 0.80
            
            return EntailmentLabel.NEUTRAL, 0.5
        
        except Exception as e:
            logger.error(f"Vertex AI NLI prediction failed: {e}")
            return EntailmentLabel.NEUTRAL, 0.5


class RuleBasedNLIModel(NLIModel):
    """Simple rule-based NLI for fallback."""
    
    def __init__(self):
        self.contradiction_indicators = [
            "not", "no", "never", "none", "neither", "cannot", "won't", "shouldn't"
        ]
        
        self.entailment_indicators = [
            "must", "shall", "required", "mandatory", "obligated"
        ]
    
    async def predict_entailment(
        self, 
        premise: str, 
        hypothesis: str
    ) -> Tuple[EntailmentLabel, float]:
        """Simple rule-based entailment checking."""
        
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()
        
        # Check for explicit contradictions
        premise_negative = any(indicator in premise_lower for indicator in self.contradiction_indicators)
        hypothesis_negative = any(indicator in hypothesis_lower for indicator in self.contradiction_indicators)
        
        if premise_negative != hypothesis_negative:
            return EntailmentLabel.CONTRADICTION, 0.70
        
        # Check for strong entailment indicators
        premise_strong = any(indicator in premise_lower for indicator in self.entailment_indicators)
        hypothesis_strong = any(indicator in hypothesis_lower for indicator in self.entailment_indicators)
        
        if premise_strong and hypothesis_strong:
            # Simple keyword overlap check
            premise_words = set(premise_lower.split())
            hypothesis_words = set(hypothesis_lower.split())
            overlap = len(premise_words.intersection(hypothesis_words))
            
            if overlap >= 3:  # Arbitrary threshold
                return EntailmentLabel.ENTAILMENT, 0.65
        
        return EntailmentLabel.NEUTRAL, 0.50


class EntailmentVerifier:
    """Main entailment verification system."""
    
    def __init__(self, primary_model: str = "vertex_ai"):
        """Initialize verifier with specified primary model."""
        
        # Initialize models in order of preference
        self.models = []
        
        if primary_model == "vertex_ai":
            try:
                self.models.append(VertexAINLIModel())
            except Exception:
                logger.warning("Vertex AI NLI model initialization failed")
        
        try:
            self.models.append(HuggingFaceNLIModel("facebook/bart-large-mnli"))
        except Exception:
            logger.warning("HuggingFace NLI model initialization failed")
        
        # Always add rule-based fallback
        self.models.append(RuleBasedNLIModel())
        
        self.threshold = settings.NLI_THRESHOLD
    
    async def verify_sentence(
        self, 
        sentence: SimplifiedSentence,
        evidence_spans: Dict[str, str],
        sentence_index: int
    ) -> VerificationResult:
        """Verify a single sentence against its evidence."""
        
        if not sentence.source_spans:
            return VerificationResult(
                sentence_index=sentence_index,
                sentence_text=sentence.text,
                entailment_label=EntailmentLabel.NEUTRAL,
                confidence_score=0.0,
                supported=False,
                rationale="No source spans provided",
                counter_evidence=[]
            )
        
        # Combine evidence from all source spans
        combined_evidence = ""
        valid_spans = []
        
        for span_id in sentence.source_spans:
            if span_id in evidence_spans:
                combined_evidence += evidence_spans[span_id] + " "
                valid_spans.append(span_id)
        
        if not combined_evidence.strip():
            return VerificationResult(
                sentence_index=sentence_index,
                sentence_text=sentence.text,
                entailment_label=EntailmentLabel.NEUTRAL,
                confidence_score=0.0,
                supported=False,
                rationale="No valid evidence spans found",
                counter_evidence=[]
            )
        
        # Try models in order until one works
        for model in self.models:
            try:
                label, confidence = await model.predict_entailment(
                    combined_evidence.strip(),
                    sentence.text
                )
                
                # Determine if sentence is supported
                supported = (
                    label == EntailmentLabel.ENTAILMENT and 
                    confidence >= self.threshold
                )
                
                rationale = self._generate_rationale(label, confidence, supported)
                
                return VerificationResult(
                    sentence_index=sentence_index,
                    sentence_text=sentence.text,
                    entailment_label=label,
                    confidence_score=confidence,
                    supported=supported,
                    rationale=rationale,
                    counter_evidence=[] if supported else valid_spans
                )
            
            except Exception as e:
                logger.warning(f"Model verification failed: {e}")
                continue
        
        # If all models failed
        return VerificationResult(
            sentence_index=sentence_index,
            sentence_text=sentence.text,
            entailment_label=EntailmentLabel.NEUTRAL,
            confidence_score=0.0,
            supported=False,
            rationale="All verification models failed",
            counter_evidence=valid_spans
        )
    
    async def verify_clause_card(
        self, 
        clause_card: ClauseCard,
        evidence_spans: Dict[str, str]
    ) -> VerificationReport:
        """Verify all sentences in a ClauseCard."""
        
        start_time = asyncio.get_event_loop().time()
        
        # Verify all sentences in parallel
        verification_tasks = []
        for i, sentence in enumerate(clause_card.simplified_sentences):
            task = self.verify_sentence(sentence, evidence_spans, i)
            verification_tasks.append(task)
        
        results = await asyncio.gather(*verification_tasks)
        
        # Calculate statistics
        verified_count = sum(1 for result in results if result.supported)
        failed_count = len(results) - verified_count
        pass_rate = verified_count / len(results) if results else 0.0
        
        end_time = asyncio.get_event_loop().time()
        processing_time = (end_time - start_time) * 1000
        
        return VerificationReport(
            clause_card_id=clause_card.clause_id,
            total_sentences=len(results),
            verified_sentences=verified_count,
            failed_sentences=failed_count,
            overall_pass_rate=pass_rate,
            sentence_results=results,
            processing_time_ms=processing_time
        )
    
    async def apply_verification_results(
        self, 
        clause_card: ClauseCard,
        verification_report: VerificationReport
    ) -> ClauseCard:
        """Apply verification results to update ClauseCard."""
        
        # Update individual sentences
        for result in verification_report.sentence_results:
            if result.sentence_index < len(clause_card.simplified_sentences):
                sentence = clause_card.simplified_sentences[result.sentence_index]
                sentence.verified = result.supported
                sentence.verifier_score = result.confidence_score
                
                # Reduce confidence for unverified sentences
                if not result.supported:
                    sentence.confidence *= 0.5
                    sentence.rationale += f" [VERIFIER: {result.rationale}]"
        
        # Update overall confidence based on verification rate
        original_confidence = clause_card.overall_confidence
        verification_factor = verification_report.overall_pass_rate
        clause_card.overall_confidence = original_confidence * verification_factor
        
        logger.info(
            f"Verification complete: {verification_report.verified_sentences}/"
            f"{verification_report.total_sentences} sentences verified "
            f"({verification_report.overall_pass_rate:.2%})"
        )
        
        return clause_card
    
    def _generate_rationale(
        self, 
        label: EntailmentLabel, 
        confidence: float,
        supported: bool
    ) -> str:
        """Generate human-readable rationale for verification result."""
        
        if supported:
            return f"Sentence is well-supported by evidence (entailment confidence: {confidence:.2f})"
        
        elif label == EntailmentLabel.CONTRADICTION:
            return f"Sentence contradicts the provided evidence (confidence: {confidence:.2f})"
        
        elif label == EntailmentLabel.NEUTRAL:
            if confidence < self.threshold:
                return f"Insufficient evidence to support sentence (confidence: {confidence:.2f} < {self.threshold})"
            else:
                return f"Evidence is neutral toward sentence (confidence: {confidence:.2f})"
        
        else:
            return f"Verification inconclusive (label: {label.value}, confidence: {confidence:.2f})"


class VerificationPipeline:
    """Complete verification pipeline with multiple checks."""
    
    def __init__(self):
        self.entailment_verifier = EntailmentVerifier()
        self.enable_self_check = settings.ENABLE_SELF_CHECK
    
    async def verify_clause_card(
        self, 
        clause_card: ClauseCard,
        evidence_spans: Dict[str, str]
    ) -> Tuple[ClauseCard, VerificationReport]:
        """Run complete verification pipeline."""
        
        try:
            # Primary NLI-based verification
            verification_report = await self.entailment_verifier.verify_clause_card(
                clause_card, evidence_spans
            )
            
            # Apply verification results
            verified_clause_card = await self.entailment_verifier.apply_verification_results(
                clause_card, verification_report
            )
            
            # Additional self-check if enabled
            if self.enable_self_check:
                verified_clause_card = await self._run_self_check(
                    verified_clause_card, evidence_spans
                )
            
            return verified_clause_card, verification_report
        
        except Exception as e:
            logger.error(f"Verification pipeline failed: {e}")
            return clause_card, VerificationReport(
                clause_card_id=clause_card.clause_id,
                total_sentences=len(clause_card.simplified_sentences),
                verified_sentences=0,
                failed_sentences=len(clause_card.simplified_sentences),
                overall_pass_rate=0.0,
                sentence_results=[],
                processing_time_ms=0.0
            )
    
    async def _run_self_check(
        self, 
        clause_card: ClauseCard,
        evidence_spans: Dict[str, str]
    ) -> ClauseCard:
        """Run additional self-check verification."""
        
        try:
            from ..generation.rag_engine import GeminiGenerator
            
            generator = GeminiGenerator()
            self_check_results = await generator.self_check_clause_card(
                clause_card, evidence_spans
            )
            
            # Cross-validate with self-check results
            for result in self_check_results:
                sentence_idx = result.get("sentence_idx", -1)
                if 0 <= sentence_idx < len(clause_card.simplified_sentences):
                    sentence = clause_card.simplified_sentences[sentence_idx]
                    self_check_supported = result.get("supported", False)
                    
                    # If both verifiers disagree, mark as unverified
                    if sentence.verified and not self_check_supported:
                        sentence.verified = False
                        sentence.confidence *= 0.3  # Heavy penalty for disagreement
                        sentence.rationale += " [SELF-CHECK: Failed cross-validation]"
            
            return clause_card
        
        except Exception as e:
            logger.warning(f"Self-check verification failed: {e}")
            return clause_card


# Factory function
def create_verifier() -> VerificationPipeline:
    """Create a new verification pipeline instance."""
    return VerificationPipeline()