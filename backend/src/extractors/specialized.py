"""
Specialized extractors for different types of legal documents.
Document-specific extraction patterns and rules.
"""

import re
from typing import Dict, List, Optional, Any
from ..models.schemas import ExtractedFact, SourceSpan, ClauseType
from .deterministic import DeterministicExtractor


class RentalAgreementExtractor:
    """Specialized extractor for rental/lease agreements."""
    
    def __init__(self, base_extractor: DeterministicExtractor):
        self.base_extractor = base_extractor
        self.rental_patterns = {
            'security_deposit': [
                r'security deposit[:\s]*(?:of\s*)?(?:Rs\.?\s*|INR\s*|₹\s*)?(\d+(?:,\d{2,3})*)',
                r'advance[:\s]*(?:of\s*)?(?:Rs\.?\s*|INR\s*|₹\s*)?(\d+(?:,\d{2,3})*)',
                r'deposit[:\s]*(?:amount\s*)?(?:Rs\.?\s*|INR\s*|₹\s*)?(\d+(?:,\d{2,3})*)',
            ],
            'monthly_rent': [
                r'monthly rent[:\s]*(?:of\s*)?(?:Rs\.?\s*|INR\s*|₹\s*)?(\d+(?:,\d{2,3})*)',
                r'rent per month[:\s]*(?:Rs\.?\s*|INR\s*|₹\s*)?(\d+(?:,\d{2,3})*)',
                r'rental[:\s]*(?:amount\s*)?(?:Rs\.?\s*|INR\s*|₹\s*)?(\d+(?:,\d{2,3})*)\s*per month',
            ],
            'lease_period': [
                r'lease period[:\s]*(\d+)\s*(months?|years?)',
                r'tenancy period[:\s]*(\d+)\s*(months?|years?)',
                r'(?:for a period of|duration of)[:\s]*(\d+)\s*(months?|years?)',
            ],
            'notice_period': [
                r'notice period[:\s]*(\d+)\s*(days?|months?)',
                r'(?:giving|serve)\s*(\d+)\s*(days?|months?)\s*notice',
                r'(?:minimum|at least)\s*(\d+)\s*(days?|months?)\s*(?:notice|prior notice)',
            ]
        }
    
    def extract(self, text: str, doc_id: str, page: int) -> List[ExtractedFact]:
        """Extract rental-specific facts."""
        facts = []
        
        # Use base extractor first
        base_facts = self.base_extractor.extract_all(text, doc_id, page)
        facts.extend(base_facts)
        
        # Extract rental-specific patterns
        for fact_type, patterns in self.rental_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    fact = self._create_rental_fact(
                        match, fact_type, doc_id, page, text, pattern
                    )
                    if fact:
                        facts.append(fact)
        
        return facts
    
    def _create_rental_fact(
        self, 
        match: re.Match, 
        fact_type: str, 
        doc_id: str, 
        page: int,
        text: str,
        pattern: str
    ) -> Optional[ExtractedFact]:
        """Create rental-specific extracted fact."""
        try:
            if fact_type in ['security_deposit', 'monthly_rent']:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                value = {"amount": amount, "currency": "INR", "type": fact_type}
            
            elif fact_type in ['lease_period', 'notice_period']:
                duration = int(match.group(1))
                unit = match.group(2).lower()
                value = {"duration": duration, "unit": unit, "type": fact_type}
            
            else:
                return None
            
            # Calculate line numbers
            lines_before = text[:match.start()].count('\n')
            
            source_span = SourceSpan(
                span_id=f"{doc_id}:p{page}:{match.start()}-{match.end()}",
                doc_id=doc_id,
                page=page,
                line_start=lines_before + 1,
                line_end=lines_before + 1,
                char_start=match.start(),
                char_end=match.end()
            )
            
            return ExtractedFact(
                fact_type=fact_type,
                value=value,
                confidence=0.90,  # High confidence for pattern matches
                source_span=source_span,
                extraction_method=f"rental_pattern_{fact_type}",
                normalized_value=value
            )
        
        except (ValueError, IndexError):
            return None


class LoanContractExtractor:
    """Specialized extractor for loan/credit agreements."""
    
    def __init__(self, base_extractor: DeterministicExtractor):
        self.base_extractor = base_extractor
        self.loan_patterns = {
            'principal_amount': [
                r'principal amount[:\s]*(?:Rs\.?\s*|INR\s*|₹\s*)?(\d+(?:,\d{2,3})*)',
                r'loan amount[:\s]*(?:Rs\.?\s*|INR\s*|₹\s*)?(\d+(?:,\d{2,3})*)',
                r'sanctioned amount[:\s]*(?:Rs\.?\s*|INR\s*|₹\s*)?(\d+(?:,\d{2,3})*)',
            ],
            'interest_rate': [
                r'interest rate[:\s]*(\d+(?:\.\d+)?)\s*%?\s*(?:per annum|p\.a\.)?',
                r'rate of interest[:\s]*(\d+(?:\.\d+)?)\s*%?\s*(?:per annum|p\.a\.)?',
                r'(?:at|@)\s*(\d+(?:\.\d+)?)\s*%\s*(?:per annum|p\.a\.)',
            ],
            'loan_tenure': [
                r'loan tenure[:\s]*(\d+)\s*(months?|years?)',
                r'repayment period[:\s]*(\d+)\s*(months?|years?)',
                r'(?:for a period of|over)\s*(\d+)\s*(months?|years?)',
            ],
            'emi_amount': [
                r'EMI[:\s]*(?:Rs\.?\s*|INR\s*|₹\s*)?(\d+(?:,\d{2,3})*)',
                r'monthly installment[:\s]*(?:Rs\.?\s*|INR\s*|₹\s*)?(\d+(?:,\d{2,3})*)',
                r'equated monthly installment[:\s]*(?:Rs\.?\s*|INR\s*|₹\s*)?(\d+(?:,\d{2,3})*)',
            ]
        }
    
    def extract(self, text: str, doc_id: str, page: int) -> List[ExtractedFact]:
        """Extract loan-specific facts."""
        facts = []
        
        # Use base extractor
        base_facts = self.base_extractor.extract_all(text, doc_id, page)
        facts.extend(base_facts)
        
        # Extract loan-specific patterns
        for fact_type, patterns in self.loan_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    fact = self._create_loan_fact(
                        match, fact_type, doc_id, page, text
                    )
                    if fact:
                        facts.append(fact)
        
        return facts
    
    def _create_loan_fact(
        self, 
        match: re.Match, 
        fact_type: str, 
        doc_id: str, 
        page: int,
        text: str
    ) -> Optional[ExtractedFact]:
        """Create loan-specific extracted fact."""
        try:
            if fact_type in ['principal_amount', 'emi_amount']:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                value = {"amount": amount, "currency": "INR", "type": fact_type}
            
            elif fact_type == 'interest_rate':
                rate = float(match.group(1))
                value = {"rate": rate, "unit": "percent_per_annum", "type": fact_type}
            
            elif fact_type == 'loan_tenure':
                duration = int(match.group(1))
                unit = match.group(2).lower()
                value = {"duration": duration, "unit": unit, "type": fact_type}
            
            else:
                return None
            
            lines_before = text[:match.start()].count('\n')
            
            source_span = SourceSpan(
                span_id=f"{doc_id}:p{page}:{match.start()}-{match.end()}",
                doc_id=doc_id,
                page=page,
                line_start=lines_before + 1,
                line_end=lines_before + 1,
                char_start=match.start(),
                char_end=match.end()
            )
            
            return ExtractedFact(
                fact_type=fact_type,
                value=value,
                confidence=0.90,
                source_span=source_span,
                extraction_method=f"loan_pattern_{fact_type}",
                normalized_value=value
            )
        
        except (ValueError, IndexError):
            return None


class DocumentTypeClassifier:
    """Classify document type to select appropriate extractor."""
    
    DOCUMENT_PATTERNS = {
        'rental_agreement': [
            r'rental agreement', r'lease agreement', r'tenancy agreement',
            r'landlord', r'tenant', r'lessor', r'lessee', r'rent',
            r'security deposit', r'lease period'
        ],
        'loan_contract': [
            r'loan agreement', r'credit agreement', r'loan contract',
            r'borrower', r'lender', r'principal amount', r'interest rate',
            r'EMI', r'equated monthly installment', r'loan tenure'
        ],
        'employment_contract': [
            r'employment agreement', r'employment contract', r'offer letter',
            r'employer', r'employee', r'salary', r'compensation',
            r'notice period', r'probation'
        ],
        'terms_of_service': [
            r'terms of service', r'terms and conditions', r'user agreement',
            r'privacy policy', r'service provider', r'user', r'website'
        ]
    }
    
    @classmethod
    def classify(cls, text: str) -> str:
        """Classify document type based on content."""
        text_lower = text.lower()
        scores = {}
        
        for doc_type, patterns in cls.DOCUMENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            scores[doc_type] = score
        
        # Return the type with highest score
        if scores:
            return max(scores, key=scores.get)
        else:
            return 'unknown'


class AdaptiveExtractor:
    """Main extractor that adapts based on document type."""
    
    def __init__(self):
        self.base_extractor = DeterministicExtractor()
        self.rental_extractor = RentalAgreementExtractor(self.base_extractor)
        self.loan_extractor = LoanContractExtractor(self.base_extractor)
    
    def extract(self, text: str, doc_id: str, page: int) -> List[ExtractedFact]:
        """Extract facts using appropriate specialized extractor."""
        # Classify document type
        doc_type = DocumentTypeClassifier.classify(text)
        
        # Select appropriate extractor
        if doc_type == 'rental_agreement':
            return self.rental_extractor.extract(text, doc_id, page)
        elif doc_type == 'loan_contract':
            return self.loan_extractor.extract(text, doc_id, page)
        else:
            # Fallback to base extractor
            return self.base_extractor.extract_all(text, doc_id, page)
    
    def get_document_type(self, text: str) -> str:
        """Get classified document type."""
        return DocumentTypeClassifier.classify(text)