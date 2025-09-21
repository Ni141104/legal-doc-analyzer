"""
Document Processing Service for MVP
Handles Document AI OCR, GCS storage, and Firestore metadata.
Single-user hackathon prototype - NO AUTHENTICATION.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import io

# Google Cloud imports
try:
    from google.cloud import documentai
    from google.cloud import storage
    from google.cloud import firestore
    from google.auth import exceptions as auth_exceptions
except ImportError:
    documentai = None
    storage = None
    firestore = None
    auth_exceptions = None

from ..models.mvp_schemas import (
    DocumentMetadata, ExtractedClause, ProcessingStatus, ClauseType
)
from ..models.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessingService:
    """
    Simplified document processing for hackathon MVP.
    
    Workflow:
    1. Upload PDF/image to GCS
    2. Process with Document AI for OCR
    3. Extract and classify clauses
    4. Store metadata in Firestore Native
    5. Prepare for vector indexing
    """
    
    def __init__(self):
        """Initialize Google Cloud clients."""
        self.document_ai_client = None
        self.storage_client = None
        self.firestore_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Google Cloud clients."""
        try:
            if documentai:
                self.document_ai_client = documentai.DocumentProcessorServiceClient()
                logger.info("Document AI client initialized")
            
            if storage:
                self.storage_client = storage.Client(project=settings.google_cloud_project_id)
                self.bucket = self.storage_client.bucket(settings.gcs_bucket_name)
                logger.info(f"GCS client initialized for bucket: {settings.gcs_bucket_name}")
            
            if firestore:
                self.firestore_client = firestore.Client(
                    project=settings.google_cloud_project_id,
                    database=settings.firestore_database_id
                )
                logger.info("Firestore client initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud clients: {str(e)}")
    
    async def upload_and_process_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: str
    ) -> DocumentMetadata:
        """
        Main entry point: Upload document and start processing.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            content_type: MIME type (application/pdf, image/jpeg, etc.)
            
        Returns:
            DocumentMetadata with processing status
        """
        try:
            # Generate unique document ID
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"
            
            logger.info(f"Starting document processing: {doc_id} ({filename})")
            
            # Validate file
            if len(file_content) > settings.max_file_size_mb * 1024 * 1024:
                raise ValueError(f"File too large. Max size: {settings.max_file_size_mb}MB")
            
            # Create initial metadata
            doc_metadata = DocumentMetadata(
                doc_id=doc_id,
                filename=filename,
                content_type=content_type,
                file_size=len(file_content),
                status=ProcessingStatus.UPLOADING,
                uploaded_at=datetime.utcnow(),
                firestore_path=settings.get_firestore_document_path(doc_id)
            )
            
            # Upload to GCS
            gcs_uri = await self._upload_to_gcs(doc_id, filename, file_content, content_type)
            doc_metadata.gcs_uri = gcs_uri
            doc_metadata.status = ProcessingStatus.PROCESSING
            
            # Save initial metadata to Firestore
            await self._save_document_metadata(doc_metadata)
            
            # Process with Document AI (async)
            extracted_text, page_count, confidence = await self._process_with_document_ai(
                file_content, content_type
            )
            
            # Update metadata with extraction results
            doc_metadata.extracted_text = extracted_text
            doc_metadata.page_count = page_count
            doc_metadata.confidence_score = confidence
            doc_metadata.processed_at = datetime.utcnow()
            doc_metadata.status = ProcessingStatus.COMPLETED
            
            # Save updated metadata
            await self._save_document_metadata(doc_metadata)
            
            logger.info(f"Document processing completed: {doc_id}")
            return doc_metadata
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            # Update status to failed if we have doc_metadata
            if 'doc_metadata' in locals():
                doc_metadata.status = ProcessingStatus.FAILED
                await self._save_document_metadata(doc_metadata)
            raise
    
    async def _upload_to_gcs(
        self, 
        doc_id: str, 
        filename: str, 
        file_content: bytes, 
        content_type: str
    ) -> str:
        """Upload document to Google Cloud Storage."""
        try:
            if not self.storage_client:
                raise Exception("GCS client not initialized")
            
            # Create GCS object path
            blob_name = f"{settings.gcs_document_prefix}{doc_id}/{filename}"
            blob = self.bucket.blob(blob_name)
            
            # Upload with metadata
            blob.metadata = {
                "doc_id": doc_id,
                "original_filename": filename,
                "uploaded_at": datetime.utcnow().isoformat()
            }
            
            # Upload the file
            blob.upload_from_string(
                file_content,
                content_type=content_type
            )
            
            gcs_uri = f"gs://{settings.gcs_bucket_name}/{blob_name}"
            logger.info(f"Uploaded to GCS: {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            logger.error(f"GCS upload failed: {str(e)}")
            raise
    
    async def _process_with_document_ai(
        self,
        file_content: bytes,
        content_type: str
    ) -> Tuple[str, int, float]:
        """
        Process document with Google Document AI.
        
        Returns:
            Tuple of (extracted_text, page_count, confidence_score)
        """
        try:
            if not self.document_ai_client:
                # Fallback for development
                logger.warning("Document AI not available, using fallback")
                return "Sample extracted text for development", 1, 0.8
            
            # Prepare Document AI request
            raw_document = documentai.RawDocument(
                content=file_content,
                mime_type=content_type
            )
            
            request = documentai.ProcessRequest(
                name=settings.document_ai_processor_name,
                raw_document=raw_document
            )
            
            # Process document
            logger.info("Sending document to Document AI...")
            result = self.document_ai_client.process_document(request=request)
            document = result.document
            
            # Extract text and metadata
            extracted_text = document.text
            page_count = len(document.pages)
            
            # Calculate average confidence
            confidence_scores = []
            for page in document.pages:
                for block in page.blocks:
                    if hasattr(block, 'confidence'):
                        confidence_scores.append(block.confidence)
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.8
            
            logger.info(f"Document AI processing completed: {page_count} pages, confidence: {avg_confidence:.3f}")
            
            return extracted_text, page_count, avg_confidence
            
        except Exception as e:
            logger.error(f"Document AI processing failed: {str(e)}")
            # Return fallback for development
            return f"Error extracting text: {str(e)}", 1, 0.1
    
    async def extract_and_classify_clauses(self, doc_id: str) -> List[ExtractedClause]:
        """
        Extract and classify legal clauses from processed document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of ExtractedClause objects
        """
        try:
            # Get document metadata
            doc_metadata = await self.get_document_metadata(doc_id)
            if not doc_metadata or not doc_metadata.extracted_text:
                raise ValueError(f"Document {doc_id} not found or not processed")
            
            # Simple clause extraction using text patterns
            clauses = await self._extract_clauses_from_text(
                doc_metadata.extracted_text, doc_id
            )
            
            # Save clauses to Firestore
            for clause in clauses:
                await self._save_clause(clause)
            
            logger.info(f"Extracted {len(clauses)} clauses from document {doc_id}")
            return clauses
            
        except Exception as e:
            logger.error(f"Clause extraction failed for {doc_id}: {str(e)}")
            return []
    
    async def _extract_clauses_from_text(self, text: str, doc_id: str) -> List[ExtractedClause]:
        """
        Simple rule-based clause extraction for MVP.
        In production, this would use advanced NLP.
        """
        try:
            clauses = []
            
            # Split text into sections (simple approach)
            sections = text.split('\n\n')
            
            # Keywords for clause type classification
            clause_keywords = {
                ClauseType.PAYMENT_TERMS: ['payment', 'pay', 'fee', 'cost', 'invoice', 'billing'],
                ClauseType.TERMINATION: ['terminate', 'end', 'expire', 'cancel', 'dissolution'],
                ClauseType.LIABILITY: ['liable', 'liability', 'damages', 'responsible', 'fault'],
                ClauseType.CONFIDENTIALITY: ['confidential', 'non-disclosure', 'proprietary', 'secret'],
                ClauseType.INTELLECTUAL_PROPERTY: ['intellectual property', 'copyright', 'patent', 'trademark'],
                ClauseType.GOVERNING_LAW: ['governing law', 'jurisdiction', 'court', 'legal'],
                ClauseType.DISPUTE_RESOLUTION: ['dispute', 'arbitration', 'mediation', 'resolution'],
                ClauseType.WARRANTIES: ['warranty', 'guarantee', 'represent', 'warrants'],
                ClauseType.INDEMNIFICATION: ['indemnify', 'indemnification', 'hold harmless']
            }
            
            for i, section in enumerate(sections):
                if len(section.strip()) < 50:  # Skip very short sections
                    continue
                
                # Classify clause type
                clause_type = ClauseType.GENERAL
                max_score = 0
                
                section_lower = section.lower()
                for c_type, keywords in clause_keywords.items():
                    score = sum(1 for keyword in keywords if keyword in section_lower)
                    if score > max_score:
                        max_score = score
                        clause_type = c_type
                
                # Create clause
                clause_id = f"clause_{doc_id}_{i:03d}"
                confidence = min(0.9, 0.5 + (max_score * 0.1))  # Simple confidence calculation
                
                clause = ExtractedClause(
                    clause_id=clause_id,
                    doc_id=doc_id,
                    text=section.strip(),
                    clause_type=clause_type,
                    confidence=confidence,
                    page_number=None,  # Would be extracted from Document AI layout
                    bbox=None,
                    key_points=await self._extract_key_points(section.strip()),
                    created_at=datetime.utcnow()
                )
                
                clauses.append(clause)
            
            return clauses
            
        except Exception as e:
            logger.error(f"Text clause extraction failed: {str(e)}")
            return []
    
    async def _extract_key_points(self, clause_text: str) -> List[str]:
        """Extract key points from clause text (simple version for MVP)."""
        try:
            # Simple approach: extract sentences with numbers, dates, or important keywords
            sentences = clause_text.split('. ')
            key_points = []
            
            important_patterns = [
                'shall', 'must', 'required', 'prohibited', 'within', 'before', 'after',
                '$', '%', 'days', 'months', 'years', 'date'
            ]
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(pattern in sentence.lower() for pattern in important_patterns):
                    key_points.append(sentence)
            
            return key_points[:3]  # Limit to top 3 key points
            
        except Exception as e:
            logger.error(f"Key point extraction failed: {str(e)}")
            return []
    
    async def _save_document_metadata(self, doc_metadata: DocumentMetadata):
        """Save document metadata to Firestore."""
        try:
            if not self.firestore_client:
                logger.warning("Firestore not available, skipping metadata save")
                return
            
            doc_ref = self.firestore_client.collection(settings.firestore_documents_collection).document(doc_metadata.doc_id)
            
            # Convert to dict for Firestore
            doc_data = doc_metadata.dict()
            doc_data['uploaded_at'] = doc_metadata.uploaded_at
            doc_data['processed_at'] = doc_metadata.processed_at
            
            doc_ref.set(doc_data)
            logger.info(f"Saved document metadata to Firestore: {doc_metadata.doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to save document metadata: {str(e)}")
    
    async def _save_clause(self, clause: ExtractedClause):
        """Save clause to Firestore."""
        try:
            if not self.firestore_client:
                logger.warning("Firestore not available, skipping clause save")
                return
            
            clause_ref = self.firestore_client.collection(settings.firestore_clauses_collection).document(clause.clause_id)
            
            clause_data = clause.dict()
            clause_data['created_at'] = clause.created_at
            
            clause_ref.set(clause_data)
            logger.info(f"Saved clause to Firestore: {clause.clause_id}")
            
        except Exception as e:
            logger.error(f"Failed to save clause: {str(e)}")
    
    async def get_document_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Get document metadata from Firestore."""
        try:
            if not self.firestore_client:
                logger.warning("Firestore not available")
                return None
            
            doc_ref = self.firestore_client.collection(settings.firestore_documents_collection).document(doc_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                return DocumentMetadata(**data)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get document metadata: {str(e)}")
            return None
    
    async def get_document_clauses(self, doc_id: str) -> List[ExtractedClause]:
        """Get all clauses for a document from Firestore."""
        try:
            if not self.firestore_client:
                logger.warning("Firestore not available")
                return []
            
            clauses_ref = self.firestore_client.collection(settings.firestore_clauses_collection)
            query = clauses_ref.where('doc_id', '==', doc_id)
            
            clauses = []
            for doc in query.stream():
                data = doc.to_dict()
                clause = ExtractedClause(**data)
                clauses.append(clause)
            
            # Sort by clause_id for consistent ordering
            clauses.sort(key=lambda x: x.clause_id)
            
            logger.info(f"Retrieved {len(clauses)} clauses for document {doc_id}")
            return clauses
            
        except Exception as e:
            logger.error(f"Failed to get document clauses: {str(e)}")
            return []
    
    async def update_document_status(
        self,
        doc_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """Update document processing status."""
        try:
            if not self.firestore_client:
                logger.warning("Firestore client not initialized - using mock update")
                return True
            
            doc_ref = self.firestore_client.collection(
                settings.FIRESTORE_COLLECTION_DOCUMENTS
            ).document(doc_id)
            
            update_data = {
                "status": status.value,
                "updated_at": datetime.utcnow()
            }
            
            if error_message:
                update_data["error_message"] = error_message
            
            if status == ProcessingStatus.COMPLETED:
                update_data["processed_at"] = datetime.utcnow()
            
            doc_ref.update(update_data)
            
            logger.info(f"Updated document {doc_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document status: {str(e)}")
            return False
    
    async def get_document_clauses(
        self,
        doc_id: str,
        clause_type: Optional[ClauseType] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ExtractedClause]:
        """Get document clauses with optional filtering and pagination."""
        try:
            if not self.firestore_client:
                # Mock clauses for development
                mock_clauses = [
                    ExtractedClause(
                        clause_id=f"clause_{i}",
                        doc_id=doc_id,
                        text=f"Mock legal clause {i} content for document {doc_id}.",
                        clause_type=clause_type or ClauseType.GENERAL,
                        confidence_score=0.9 - (i * 0.1),
                        created_at=datetime.utcnow()
                    )
                    for i in range(min(limit, 3))
                ]
                return mock_clauses
            
            collection = self.firestore_client.collection(settings.FIRESTORE_COLLECTION_CLAUSES)
            query = collection.where("doc_id", "==", doc_id)
            
            if clause_type:
                query = query.where("clause_type", "==", clause_type.value)
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            docs = query.stream()
            clauses = []
            
            for doc in docs:
                data = doc.to_dict()
                clause = ExtractedClause(
                    clause_id=data["clause_id"],
                    doc_id=data["doc_id"],
                    text=data["text"],
                    clause_type=ClauseType(data["clause_type"]),
                    confidence_score=data["confidence_score"],
                    page_number=data.get("page_number"),
                    bounding_box=data.get("bounding_box"),
                    created_at=data["created_at"],
                    keywords=data.get("keywords", [])
                )
                clauses.append(clause)
            
            logger.info(f"Retrieved {len(clauses)} clauses for document {doc_id}")
            return clauses
            
        except Exception as e:
            logger.error(f"Failed to get document clauses: {str(e)}")
            return []


# Global service instance
document_processing_service = DocumentProcessingService()