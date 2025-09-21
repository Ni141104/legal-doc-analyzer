"""
Document Ingestion Service
Handles document upload, OCR via Document AI, and deterministic extraction.
"""

import asyncio
import logging
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import mimetypes
import re

from google.cloud import documentai, storage
from google.cloud.exceptions import GoogleCloudError

from ..models.schemas import (
    DocumentMetadata, ExtractedFacts, ProcessingStatus, 
    ClauseMetadata, ClauseType, NormalizedTerms
)
from ..models.config import settings

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """Service for processing document uploads and extraction."""
    
    def __init__(self):
        """Initialize the document ingestion service."""
        self.doc_ai_client = documentai.DocumentProcessorServiceClient()
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(settings.GCS_BUCKET)
        
        # Configure deterministic extractors
        self.amount_patterns = [
            r'₹\s*([0-9,]+(?:\.[0-9]{2})?)',  # Indian Rupees
            r'\$\s*([0-9,]+(?:\.[0-9]{2})?)',  # US Dollars
            r'([0-9,]+(?:\.[0-9]{2})?)\s*(?:rupees?|dollars?|USD|INR)',  # Word form
        ]
        
        self.date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',  # DD/MM/YYYY or MM/DD/YYYY
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',  # YYYY/MM/DD
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',  # DD Month YYYY
        ]
        
        self.party_patterns = [
            r'(?:Tenant|Landlord|Lessor|Lessee|Party|Borrower|Lender):\s*([A-Za-z\s,\.]+?)(?:\n|$)',
            r'between\s+([A-Za-z\s,\.]+?)\s+and\s+([A-Za-z\s,\.]+?)(?:\s|$)',
            r'Name:\s*([A-Za-z\s,\.]+?)(?:\n|Address:)',
        ]
    
    async def upload_document(
        self, 
        file_content: bytes, 
        filename: str, 
        user_id: str,
        content_type: Optional[str] = None,
        ephemeral: bool = False,
        retention_days: Optional[int] = None
    ) -> DocumentMetadata:
        """
        Upload and process a document.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            user_id: ID of user uploading
            content_type: MIME type (auto-detected if None)
            ephemeral: Whether to process without persistent storage
            retention_days: Custom retention period
            
        Returns:
            DocumentMetadata with processing status and initial facts
        """
        try:
            # Generate document ID and validate
            doc_id = str(uuid.uuid4())
            content_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
            file_size = len(file_content)
            checksum = hashlib.sha256(file_content).hexdigest()
            
            # Validate file
            if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                raise ValueError(f"File size {file_size} exceeds maximum {settings.MAX_FILE_SIZE_MB}MB")
            
            if not self._is_supported_file_type(content_type):
                raise ValueError(f"File type {content_type} is not supported")
            
            # Upload to Cloud Storage
            storage_path = f"documents/{user_id}/{doc_id}/{filename}"
            if not ephemeral:
                await self._upload_to_storage(storage_path, file_content, content_type)
            
            # Process with Document AI
            doc_ai_result = await self._process_with_document_ai(file_content, content_type)
            
            # Extract deterministic facts
            extracted_facts = await self._extract_deterministic_facts(
                doc_ai_result.text if doc_ai_result else "",
                doc_ai_result.pages if doc_ai_result else []
            )
            
            # Create document metadata
            document_metadata = DocumentMetadata(
                doc_id=doc_id,
                original_filename=filename,
                storage_path=storage_path if not ephemeral else "",
                content_type=content_type,
                file_size=file_size,
                checksum=checksum,
                processing_status=ProcessingStatus.EXTRACTING,
                extracted_facts=extracted_facts,
                total_clauses=0,  # Will be updated after clause extraction
                jurisdiction=self._detect_jurisdiction(extracted_facts),
                upload_timestamp=datetime.utcnow(),
                user_id=user_id,
                retention_policy=f"{retention_days or settings.DEFAULT_RETENTION_DAYS}_days"
            )
            
            # Start background clause extraction
            asyncio.create_task(self._extract_clauses_background(doc_id, doc_ai_result))
            
            logger.info(f"Document uploaded successfully: {doc_id}")
            return document_metadata
            
        except Exception as e:
            logger.error(f"Document upload failed: {str(e)}")
            raise
    
    async def _upload_to_storage(self, storage_path: str, content: bytes, content_type: str) -> None:
        """Upload file to Google Cloud Storage."""
        try:
            blob = self.bucket.blob(storage_path)
            
            # Set metadata
            blob.metadata = {
                "uploaded_at": datetime.utcnow().isoformat(),
                "content_type": content_type
            }
            
            # Upload with encryption if enabled
            if settings.ENABLE_KMS_ENCRYPTION and settings.KMS_KEY_NAME:
                kms_key = f"projects/{settings.GOOGLE_CLOUD_PROJECT}/locations/{settings.KMS_LOCATION}/keyRings/{settings.KMS_KEY_RING}/cryptoKeys/{settings.KMS_KEY_NAME}"
                blob.kms_key_name = kms_key
            
            await asyncio.get_event_loop().run_in_executor(
                None, blob.upload_from_string, content, content_type
            )
            
        except GoogleCloudError as e:
            logger.error(f"Storage upload failed: {str(e)}")
            raise
    
    async def _process_with_document_ai(self, content: bytes, content_type: str) -> Optional[Any]:
        """Process document with Google Document AI."""
        try:
            # Configure processor
            processor_name = (
                f"projects/{settings.GOOGLE_CLOUD_PROJECT}/"
                f"locations/{settings.DOCUMENT_AI_LOCATION}/"
                f"processors/{settings.DOCUMENT_AI_PROCESSOR_ID}"
            )
            
            # Create request
            raw_document = documentai.RawDocument(
                content=content,
                mime_type=content_type
            )
            
            request = documentai.ProcessRequest(
                name=processor_name,
                raw_document=raw_document
            )
            
            # Process document
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.doc_ai_client.process_document, request
            )
            
            return result.document
            
        except GoogleCloudError as e:
            logger.error(f"Document AI processing failed: {str(e)}")
            # Return None to allow fallback processing
            return None
    
    async def _extract_deterministic_facts(self, text: str, pages: List[Any]) -> ExtractedFacts:
        """Extract deterministic facts using regex patterns."""
        try:
            # Extract amounts
            amounts = {}
            for pattern in self.amount_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for i, match in enumerate(matches):
                    amount_str = match.group(1).replace(",", "")
                    try:
                        amount = float(amount_str)
                        amounts[f"amount_{i+1}"] = amount
                    except ValueError:
                        continue
            
            # Extract dates
            dates = {}
            for pattern in self.date_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for i, match in enumerate(matches):
                    date_str = match.group(1)
                    # Basic date normalization (could be enhanced)
                    dates[f"date_{i+1}"] = date_str
            
            # Extract parties
            parties = []
            for pattern in self.party_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    for group in match.groups():
                        if group and group.strip():
                            party_name = group.strip().rstrip(',.')
                            if party_name not in parties and len(party_name) > 2:
                                parties.append(party_name)
            
            # Detect jurisdictions
            jurisdiction_patterns = [
                r'(?:governed by|jurisdiction of|laws of)\s+([A-Za-z\s]+?)(?:\s|$)',
                r'(?:State of|Province of)\s+([A-Za-z\s]+?)(?:\s|$)',
                r'(?:Mumbai|Delhi|Bangalore|Chennai|Kolkata|Pune|Hyderabad)',  # Indian cities
                r'(?:Maharashtra|Karnataka|Tamil Nadu|West Bengal|Delhi)',  # Indian states
            ]
            
            jurisdictions = []
            for pattern in jurisdiction_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    jurisdiction = match.group(0).strip()
                    if jurisdiction not in jurisdictions:
                        jurisdictions.append(jurisdiction)
            
            # Detect document type
            doc_type = self._detect_document_type(text)
            
            return ExtractedFacts(
                amounts=amounts,
                dates=dates,
                parties=parties,
                jurisdictions=jurisdictions,
                document_type=doc_type,
                language="en",  # Could be enhanced with language detection
                page_count=len(pages) if pages else 1
            )
            
        except Exception as e:
            logger.error(f"Deterministic extraction failed: {str(e)}")
            return ExtractedFacts(page_count=1)
    
    async def _extract_clauses_background(self, doc_id: str, document: Any) -> None:
        """Extract clauses from document in background task."""
        try:
            if not document:
                logger.warning(f"No document data for clause extraction: {doc_id}")
                return
            
            clauses = []
            
            # Extract paragraphs as potential clauses
            for page_idx, page in enumerate(document.pages):
                for para_idx, paragraph in enumerate(page.paragraphs):
                    # Get paragraph text
                    para_text = self._get_layout_text(paragraph.layout, document.text)
                    
                    if len(para_text.strip()) > 50:  # Filter out short paragraphs
                        clause_id = f"{doc_id}_p{page_idx+1}_{para_idx+1}"
                        
                        # Classify clause type
                        clause_type = self._classify_clause_type(para_text)
                        
                        # Extract terms from clause
                        extracted_terms = self._extract_clause_terms(para_text)
                        
                        clause = ClauseMetadata(
                            clause_id=clause_id,
                            doc_id=doc_id,
                            section_title=self._extract_section_title(para_text),
                            content_text=para_text,
                            page=page_idx + 1,
                            line_start=para_idx * 10,  # Approximate
                            line_end=(para_idx + 1) * 10,
                            clause_type=clause_type,
                            extracted_terms=extracted_terms
                        )
                        
                        clauses.append(clause)
            
            # TODO: Store clauses in database
            logger.info(f"Extracted {len(clauses)} clauses from document {doc_id}")
            
        except Exception as e:
            logger.error(f"Clause extraction failed for {doc_id}: {str(e)}")
    
    def _get_layout_text(self, layout: Any, document_text: str) -> str:
        """Extract text from layout using text segments."""
        try:
            text_segments = []
            for segment in layout.text_anchor.text_segments:
                start_index = int(segment.start_index) if segment.start_index else 0
                end_index = int(segment.end_index) if segment.end_index else len(document_text)
                text_segments.append(document_text[start_index:end_index])
            return "".join(text_segments)
        except Exception:
            return ""
    
    def _classify_clause_type(self, text: str) -> ClauseType:
        """Classify clause type based on content."""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['payment', 'pay', 'amount', 'fee', 'rent']):
            return ClauseType.PAYMENT
        elif any(keyword in text_lower for keyword in ['deposit', 'security']):
            return ClauseType.DEPOSIT
        elif any(keyword in text_lower for keyword in ['terminate', 'termination', 'end']):
            return ClauseType.TERMINATION
        elif any(keyword in text_lower for keyword in ['liable', 'liability', 'responsible']):
            return ClauseType.LIABILITY
        elif any(keyword in text_lower for keyword in ['confidential', 'non-disclosure', 'nda']):
            return ClauseType.CONFIDENTIALITY
        elif any(keyword in text_lower for keyword in ['intellectual property', 'copyright', 'patent']):
            return ClauseType.INTELLECTUAL_PROPERTY
        elif any(keyword in text_lower for keyword in ['dispute', 'arbitration', 'mediation']):
            return ClauseType.DISPUTE_RESOLUTION
        elif any(keyword in text_lower for keyword in ['governing law', 'jurisdiction']):
            return ClauseType.GOVERNING_LAW
        elif any(keyword in text_lower for keyword in ['force majeure', 'act of god']):
            return ClauseType.FORCE_MAJEURE
        elif any(keyword in text_lower for keyword in ['amendment', 'modify', 'change']):
            return ClauseType.AMENDMENT
        else:
            return ClauseType.OTHER
    
    def _extract_clause_terms(self, text: str) -> NormalizedTerms:
        """Extract normalized terms from clause text."""
        amounts = {}
        dates = {}
        parties = {}
        durations = {}
        percentages = {}
        
        # Extract amounts
        for pattern in self.amount_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for i, match in enumerate(matches):
                try:
                    amount = float(match.group(1).replace(",", ""))
                    amounts[f"amount_{i+1}"] = amount
                except ValueError:
                    continue
        
        # Extract percentages
        percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percent_matches = re.finditer(percent_pattern, text)
        for i, match in enumerate(percent_matches):
            try:
                percentage = float(match.group(1))
                percentages[f"percentage_{i+1}"] = percentage
            except ValueError:
                continue
        
        # Extract durations (days, months, years)
        duration_patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*months?',
            r'(\d+)\s*years?'
        ]
        
        for pattern in duration_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value = int(match.group(1))
                    if 'day' in match.group(0).lower():
                        durations[f"duration_days"] = value
                    elif 'month' in match.group(0).lower():
                        durations[f"duration_months"] = value * 30  # Convert to days
                    elif 'year' in match.group(0).lower():
                        durations[f"duration_years"] = value * 365  # Convert to days
                except ValueError:
                    continue
        
        return NormalizedTerms(
            amounts=amounts,
            dates=dates,
            parties=parties,
            durations=durations,
            percentages=percentages
        )
    
    def _extract_section_title(self, text: str) -> Optional[str]:
        """Extract section title from clause text."""
        lines = text.split('\n')
        first_line = lines[0].strip()
        
        # Check if first line looks like a title
        if (len(first_line) < 100 and 
            (first_line.isupper() or 
             any(char in first_line for char in [':', '.', ')']) or
             re.match(r'^\d+\.?\s+', first_line))):
            return first_line
        
        return None
    
    def _detect_document_type(self, text: str) -> Optional[str]:
        """Detect document type from content."""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['rental agreement', 'lease agreement', 'tenancy']):
            return "rental_agreement"
        elif any(keyword in text_lower for keyword in ['loan agreement', 'credit agreement', 'borrowing']):
            return "loan_agreement"
        elif any(keyword in text_lower for keyword in ['employment contract', 'employment agreement']):
            return "employment_contract"
        elif any(keyword in text_lower for keyword in ['non-disclosure', 'nda', 'confidentiality agreement']):
            return "nda"
        elif any(keyword in text_lower for keyword in ['service agreement', 'consulting agreement']):
            return "service_agreement"
        elif any(keyword in text_lower for keyword in ['terms of service', 'terms and conditions', 'tos']):
            return "terms_of_service"
        else:
            return "unknown"
    
    def _detect_jurisdiction(self, facts: ExtractedFacts) -> Optional[str]:
        """Detect primary jurisdiction from extracted facts."""
        if facts.jurisdictions:
            # Simple heuristic: return the first jurisdiction found
            return facts.jurisdictions[0]
        return None
    
    def _is_supported_file_type(self, content_type: str) -> bool:
        """Check if file type is supported."""
        supported_types = [
            'application/pdf',
            'image/jpeg',
            'image/png',
            'image/tiff',
            'image/gif',
            'image/bmp',
            'image/webp',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ]
        return content_type in supported_types


# Global service instance
document_ingestion_service = DocumentIngestionService()