"""
Deterministic extractors for structured data from legal documents.
High-accuracy rule-based extraction for amounts, dates, parties, and other facts.
Designed to minimize hallucination risk for critical numeric and structured data.
"""

import re
import dateparser
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
import spacy
from spacy.matcher import Matcher
import logging

from ..models.schemas import ExtractedFact, SourceSpan


logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of an extraction operation."""
    value: Any
    confidence: float
    span_start: int
    span_end: int
    method: str
    context: str = ""


class CurrencyExtractor:
    """Extract currency amounts and denominations."""
    
    # Indian currency patterns
    INDIAN_CURRENCY_PATTERNS = [
        # Rupees with lakhs/crores
        r'(?:Rs\.?\s*|INR\s*|₹\s*)(\d+(?:,\d{2,3})*(?:\.\d{2})?)\s*(?:lakhs?|lacs?)',
        r'(?:Rs\.?\s*|INR\s*|₹\s*)(\d+(?:,\d{2,3})*(?:\.\d{2})?)\s*(?:crores?)',
        
        # Standard rupee amounts
        r'(?:Rs\.?\s*|INR\s*|₹\s*)(\d+(?:,\d{2,3})*(?:\.\d{2})?)',
        
        # Spelled out amounts
        r'(\d+)\s*(?:lakhs?|lacs?)\s*(?:rupees?|Rs\.?|INR|₹)',
        r'(\d+)\s*(?:crores?)\s*(?:rupees?|Rs\.?|INR|₹)',
        
        # Amount followed by currency
        r'(\d+(?:,\d{2,3})*(?:\.\d{2})?)\s*(?:rupees?|Rs\.?|INR|₹)',
    ]
    
    # Global currency patterns
    GLOBAL_CURRENCY_PATTERNS = [
        r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',  # USD
        r'USD\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'€(\d+(?:,\d{3})*(?:\.\d{2})?)',   # EUR
        r'EUR\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'£(\d+(?:,\d{3})*(?:\.\d{2})?)',   # GBP
        r'GBP\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
    ]
    
    def __init__(self):
        self.patterns = self.INDIAN_CURRENCY_PATTERNS + self.GLOBAL_CURRENCY_PATTERNS
        self.compiled_patterns = [(re.compile(p, re.IGNORECASE), p) for p in self.patterns]
    
    def extract(self, text: str) -> List[ExtractionResult]:
        """Extract currency amounts from text."""
        results = []
        
        for pattern, pattern_str in self.compiled_patterns:
            for match in pattern.finditer(text):
                try:
                    amount_str = match.group(1)
                    amount = self._parse_amount(amount_str, pattern_str)
                    
                    if amount is not None:
                        currency = self._detect_currency(match.group(0))
                        confidence = self._calculate_confidence(match.group(0), pattern_str)
                        
                        results.append(ExtractionResult(
                            value={"amount": amount, "currency": currency, "raw": match.group(0)},
                            confidence=confidence,
                            span_start=match.start(),
                            span_end=match.end(),
                            method=f"currency_pattern_{pattern_str[:30]}",
                            context=text[max(0, match.start()-50):match.end()+50]
                        ))
                
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse currency amount: {e}")
                    continue
        
        return self._deduplicate_results(results)
    
    def _parse_amount(self, amount_str: str, pattern: str) -> Optional[float]:
        """Parse amount string to float."""
        try:
            # Remove commas and parse
            cleaned = amount_str.replace(',', '')
            amount = float(cleaned)
            
            # Apply multipliers for lakhs/crores
            if 'lakh' in pattern.lower() or 'lac' in pattern.lower():
                amount *= 100000
            elif 'crore' in pattern.lower():
                amount *= 10000000
            
            return amount
        except ValueError:
            return None
    
    def _detect_currency(self, matched_text: str) -> str:
        """Detect currency code from matched text."""
        text_lower = matched_text.lower()
        
        if any(symbol in matched_text for symbol in ['₹', 'rs.', 'rs ', 'inr', 'rupee']):
            return 'INR'
        elif '$' in matched_text or 'usd' in text_lower:
            return 'USD'
        elif '€' in matched_text or 'eur' in text_lower:
            return 'EUR'
        elif '£' in matched_text or 'gbp' in text_lower:
            return 'GBP'
        else:
            return 'INR'  # Default for Indian legal documents
    
    def _calculate_confidence(self, matched_text: str, pattern: str) -> float:
        """Calculate extraction confidence based on pattern specificity."""
        base_confidence = 0.85
        
        # Boost confidence for explicit currency symbols
        if any(symbol in matched_text for symbol in ['₹', '$', '€', '£']):
            base_confidence += 0.10
        
        # Boost for explicit currency codes
        if any(code in matched_text.upper() for code in ['INR', 'USD', 'EUR', 'GBP']):
            base_confidence += 0.05
        
        # Reduce confidence for ambiguous patterns
        if re.search(r'^\d+$', matched_text.strip()):
            base_confidence -= 0.20
        
        return min(1.0, base_confidence)
    
    def _deduplicate_results(self, results: List[ExtractionResult]) -> List[ExtractionResult]:
        """Remove overlapping or duplicate extractions."""
        if not results:
            return results
        
        # Sort by span start position
        results.sort(key=lambda x: x.span_start)
        
        deduplicated = []
        for result in results:
            # Check for overlap with existing results
            overlaps = False
            for existing in deduplicated:
                if (result.span_start < existing.span_end and 
                    result.span_end > existing.span_start):
                    # Keep the one with higher confidence
                    if result.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(result)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(result)
        
        return deduplicated


class DateExtractor:
    """Extract dates and time periods from legal documents."""
    
    def __init__(self):
        self.date_patterns = [
            # Indian date formats
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',  # DD/MM/YYYY or DD-MM-YYYY
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
            r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b',
            
            # Period patterns
            r'(\d+)\s*(?:days?|months?|years?)',
            r'(?:within|after|before)\s+(\d+)\s*(?:days?|months?|years?)',
            r'(?:from|until|till)\s+([^,\.]+?)(?:,|\.|$)',
        ]
        
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.date_patterns]
    
    def extract(self, text: str) -> List[ExtractionResult]:
        """Extract dates and periods from text."""
        results = []
        
        # Extract explicit dates
        for i, pattern in enumerate(self.compiled_patterns[:3]):
            for match in pattern.finditer(text):
                parsed_date = self._parse_date(match.group(0))
                if parsed_date:
                    results.append(ExtractionResult(
                        value={"date": parsed_date, "raw": match.group(0)},
                        confidence=0.95,
                        span_start=match.start(),
                        span_end=match.end(),
                        method=f"date_pattern_{i}",
                        context=text[max(0, match.start()-30):match.end()+30]
                    ))
        
        # Extract periods
        for i, pattern in enumerate(self.compiled_patterns[3:], 3):
            for match in pattern.finditer(text):
                period = self._parse_period(match.group(0))
                if period:
                    results.append(ExtractionResult(
                        value=period,
                        confidence=0.90,
                        span_start=match.start(),
                        span_end=match.end(),
                        method=f"period_pattern_{i}",
                        context=text[max(0, match.start()-30):match.end()+30]
                    ))
        
        return self._deduplicate_results(results)
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string using dateparser."""
        try:
            return dateparser.parse(date_str, settings={'DATE_ORDER': 'DMY'})
        except:
            return None
    
    def _parse_period(self, period_str: str) -> Optional[Dict[str, Any]]:
        """Parse period/duration string."""
        try:
            # Extract numeric value and unit
            number_match = re.search(r'(\d+)', period_str)
            if not number_match:
                return None
            
            value = int(number_match.group(1))
            period_lower = period_str.lower()
            
            if 'day' in period_lower:
                unit = 'days'
            elif 'month' in period_lower:
                unit = 'months'
            elif 'year' in period_lower:
                unit = 'years'
            else:
                unit = 'unknown'
            
            return {
                "value": value,
                "unit": unit,
                "raw": period_str
            }
        
        except:
            return None
    
    def _deduplicate_results(self, results: List[ExtractionResult]) -> List[ExtractionResult]:
        """Remove overlapping extractions."""
        if not results:
            return results
        
        results.sort(key=lambda x: x.span_start)
        deduplicated = []
        
        for result in results:
            overlaps = False
            for existing in deduplicated:
                if (result.span_start < existing.span_end and 
                    result.span_end > existing.span_start):
                    if result.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(result)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(result)
        
        return deduplicated


class PartyExtractor:
    """Extract party names and roles from legal documents."""
    
    def __init__(self):
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Party extraction will be limited.")
            self.nlp = None
        
        self.party_patterns = [
            # Common party designations
            r'(?:landlord|lessor|owner)\s*[:\-]?\s*([A-Z][a-zA-Z\s\.]+?)(?:,|\.|$)',
            r'(?:tenant|lessee|renter)\s*[:\-]?\s*([A-Z][a-zA-Z\s\.]+?)(?:,|\.|$)',
            r'(?:borrower|debtor)\s*[:\-]?\s*([A-Z][a-zA-Z\s\.]+?)(?:,|\.|$)',
            r'(?:lender|creditor|bank)\s*[:\-]?\s*([A-Z][a-zA-Z\s\.]+?)(?:,|\.|$)',
            r'(?:employer|company)\s*[:\-]?\s*([A-Z][a-zA-Z\s\.]+?)(?:,|\.|$)',
            r'(?:employee|worker)\s*[:\-]?\s*([A-Z][a-zA-Z\s\.]+?)(?:,|\.|$)',
            
            # Agreement parties
            r'between\s+([A-Z][a-zA-Z\s\.]+?)\s+(?:and|&)',
            r'and\s+([A-Z][a-zA-Z\s\.]+?)(?:,|\.|$)',
        ]
        
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.party_patterns]
    
    def extract(self, text: str) -> List[ExtractionResult]:
        """Extract party information from text."""
        results = []
        
        # Pattern-based extraction
        for i, pattern in enumerate(self.compiled_patterns):
            for match in pattern.finditer(text):
                try:
                    party_name = match.group(1).strip()
                    if self._is_valid_party_name(party_name):
                        role = self._infer_party_role(match.group(0))
                        
                        results.append(ExtractionResult(
                            value={"name": party_name, "role": role, "raw": match.group(0)},
                            confidence=0.85,
                            span_start=match.start(1),
                            span_end=match.end(1),
                            method=f"party_pattern_{i}",
                            context=text[max(0, match.start()-50):match.end()+50]
                        ))
                
                except IndexError:
                    continue
        
        # NER-based extraction if available
        if self.nlp:
            ner_results = self._extract_with_ner(text)
            results.extend(ner_results)
        
        return self._deduplicate_results(results)
    
    def _is_valid_party_name(self, name: str) -> bool:
        """Validate if extracted string is a valid party name."""
        if len(name) < 2 or len(name) > 100:
            return False
        
        # Should contain mostly letters and common name characters
        if not re.match(r'^[A-Za-z\s\.\-&,()]+$', name):
            return False
        
        # Should not be common legal terms
        invalid_terms = {
            'agreement', 'contract', 'party', 'parties', 'document', 
            'section', 'clause', 'terms', 'conditions', 'hereby',
            'whereas', 'therefore'
        }
        
        if name.lower().strip() in invalid_terms:
            return False
        
        return True
    
    def _infer_party_role(self, context: str) -> str:
        """Infer party role from context."""
        context_lower = context.lower()
        
        if any(term in context_lower for term in ['landlord', 'lessor', 'owner']):
            return 'landlord'
        elif any(term in context_lower for term in ['tenant', 'lessee', 'renter']):
            return 'tenant'
        elif any(term in context_lower for term in ['lender', 'creditor', 'bank']):
            return 'lender'
        elif any(term in context_lower for term in ['borrower', 'debtor']):
            return 'borrower'
        elif any(term in context_lower for term in ['employer', 'company']):
            return 'employer'
        elif any(term in context_lower for term in ['employee', 'worker']):
            return 'employee'
        else:
            return 'party'
    
    def _extract_with_ner(self, text: str) -> List[ExtractionResult]:
        """Extract parties using Named Entity Recognition."""
        results = []
        
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG']:
                    if self._is_valid_party_name(ent.text):
                        results.append(ExtractionResult(
                            value={"name": ent.text, "role": "party", "entity_type": ent.label_},
                            confidence=0.80,
                            span_start=ent.start_char,
                            span_end=ent.end_char,
                            method=f"ner_{ent.label_}",
                            context=text[max(0, ent.start_char-30):ent.end_char+30]
                        ))
        
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
        
        return results
    
    def _deduplicate_results(self, results: List[ExtractionResult]) -> List[ExtractionResult]:
        """Remove duplicate party extractions."""
        if not results:
            return results
        
        # Group by normalized name
        name_groups = {}
        for result in results:
            normalized_name = result.value["name"].lower().strip()
            if normalized_name not in name_groups:
                name_groups[normalized_name] = []
            name_groups[normalized_name].append(result)
        
        # Keep the best result for each name
        deduplicated = []
        for name, group in name_groups.items():
            best = max(group, key=lambda x: x.confidence)
            deduplicated.append(best)
        
        return deduplicated


class DeterministicExtractor:
    """Main coordinator for all deterministic extraction operations."""
    
    def __init__(self):
        self.currency_extractor = CurrencyExtractor()
        self.date_extractor = DateExtractor()
        self.party_extractor = PartyExtractor()
    
    def extract_all(self, text: str, doc_id: str, page: int) -> List[ExtractedFact]:
        """Extract all structured facts from text."""
        extracted_facts = []
        
        # Extract currencies
        currency_results = self.currency_extractor.extract(text)
        for result in currency_results:
            fact = self._create_extracted_fact(
                result, "currency", doc_id, page, text
            )
            extracted_facts.append(fact)
        
        # Extract dates
        date_results = self.date_extractor.extract(text)
        for result in date_results:
            fact = self._create_extracted_fact(
                result, "date", doc_id, page, text
            )
            extracted_facts.append(fact)
        
        # Extract parties
        party_results = self.party_extractor.extract(text)
        for result in party_results:
            fact = self._create_extracted_fact(
                result, "party", doc_id, page, text
            )
            extracted_facts.append(fact)
        
        return extracted_facts
    
    def _create_extracted_fact(
        self, 
        result: ExtractionResult, 
        fact_type: str, 
        doc_id: str, 
        page: int,
        full_text: str
    ) -> ExtractedFact:
        """Convert extraction result to ExtractedFact."""
        
        # Calculate line numbers from character positions
        lines_before = full_text[:result.span_start].count('\n')
        lines_in_span = full_text[result.span_start:result.span_end].count('\n')
        
        source_span = SourceSpan(
            span_id=f"{doc_id}:p{page}:{result.span_start}-{result.span_end}",
            doc_id=doc_id,
            page=page,
            line_start=lines_before + 1,
            line_end=lines_before + lines_in_span + 1,
            char_start=result.span_start,
            char_end=result.span_end
        )
        
        return ExtractedFact(
            fact_type=fact_type,
            value=result.value,
            confidence=result.confidence,
            source_span=source_span,
            extraction_method=result.method,
            normalized_value=self._normalize_value(result.value, fact_type)
        )
    
    def _normalize_value(self, value: Any, fact_type: str) -> Any:
        """Normalize extracted values for consistent processing."""
        if fact_type == "currency" and isinstance(value, dict):
            return {
                "amount_inr": self._convert_to_inr(value.get("amount", 0), value.get("currency", "INR")),
                "original_amount": value.get("amount"),
                "original_currency": value.get("currency")
            }
        elif fact_type == "date" and isinstance(value, dict):
            date_obj = value.get("date")
            if date_obj:
                return {
                    "iso_date": date_obj.isoformat(),
                    "timestamp": date_obj.timestamp()
                }
        elif fact_type == "party" and isinstance(value, dict):
            return {
                "normalized_name": value.get("name", "").strip().title(),
                "role": value.get("role", "party")
            }
        
        return value
    
    def _convert_to_inr(self, amount: float, currency: str) -> float:
        """Convert currency amounts to INR for comparison."""
        # Simplified conversion - in production, use real-time rates
        conversion_rates = {
            "INR": 1.0,
            "USD": 83.0,  # Approximate rate
            "EUR": 90.0,
            "GBP": 105.0
        }
        
        rate = conversion_rates.get(currency, 1.0)
        return amount * rate


# Factory function for easy instantiation
def create_extractor() -> DeterministicExtractor:
    """Create a new deterministic extractor instance."""
    return DeterministicExtractor()