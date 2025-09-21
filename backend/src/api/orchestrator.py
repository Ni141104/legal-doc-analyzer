"""
Document processing orchestrator that coordinates all system components.
Handles document upload, processing, querying, and human review workflows.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import tempfile
import os

from ..models.schemas import (
    DocumentUploadResponse, DocumentClausesResponse, QueryRequest, QueryResponse,
    HumanReviewRequest, HumanReviewResponse, ExportRequest, 
    ClauseCard, ClauseRecord, DocumentMetadata, ExtractedFact, SystemMetrics
)
from ..models.config import settings
from ..extractors import AdaptiveExtractor
from ..retrieval import create_retriever, PolyVectorRetriever
from ..generation import create_rag_engine, ConstrainedRAGEngine
from ..verification import create_verifier, VerificationPipeline

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document parsing and OCR using Google Document AI."""
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Document AI client."""
        if self._client is None:
            try:
                from google.cloud import documentai
                self._client = documentai.DocumentProcessorServiceClient()
            except ImportError:
                logger.error("Google Cloud Document AI client not available")
                raise
        return self._client
    
    async def process_document(self, content: bytes, mime_type: str) -> Dict[str, Any]:
        """Process document using Document AI."""
        try:
            from google.cloud import documentai
            from ..models.config import gcp_config
            
            config = gcp_config.document_ai_config
            processor_name = f"projects/{config['project_id']}/locations/{config['location']}/processors/{config['processor_id']}"
            
            # Create request
            request = documentai.ProcessRequest(
                name=processor_name,
                raw_document=documentai.RawDocument(
                    content=content,
                    mime_type=mime_type
                )
            )
            
            # Process document
            result = self.client.process_document(request=request)
            document = result.document
            
            # Extract text and structure
            pages = []
            for i, page in enumerate(document.pages):
                page_text = self._extract_page_text(document.text, page)
                pages.append({
                    "page_number": i + 1,
                    "text": page_text,
                    "confidence": page.layout.confidence if page.layout else 0.9
                })
            
            return {
                "text": document.text,
                "pages": pages,
                "confidence": self._calculate_overall_confidence(pages)
            }
        
        except Exception as e:
            logger.error(f"Document AI processing failed: {e}")
            # Fallback to simple text extraction
            return await self._fallback_text_extraction(content, mime_type)
    
    def _extract_page_text(self, full_text: str, page) -> str:
        """Extract text for a specific page."""
        try:
            if page.layout and page.layout.text_anchor:
                segments = page.layout.text_anchor.text_segments
                page_text = ""
                for segment in segments:
                    start_index = segment.start_index if segment.start_index else 0
                    end_index = segment.end_index if segment.end_index else len(full_text)
                    page_text += full_text[start_index:end_index]
                return page_text
            return ""
        except Exception:
            return ""
    
    def _calculate_overall_confidence(self, pages: List[Dict]) -> float:
        """Calculate overall OCR confidence."""
        if not pages:
            return 0.0
        
        total_confidence = sum(page.get("confidence", 0.0) for page in pages)
        return total_confidence / len(pages)
    
    async def _fallback_text_extraction(self, content: bytes, mime_type: str) -> Dict[str, Any]:
        """Fallback text extraction for when Document AI is unavailable."""
        try:
            if mime_type == "text/plain":
                text = content.decode('utf-8')
                return {
                    "text": text,
                    "pages": [{"page_number": 1, "text": text, "confidence": 1.0}],
                    "confidence": 1.0
                }
            elif mime_type == "application/pdf":
                try:
                    import PyPDF2
                    import io
                    
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                    pages = []
                    full_text = ""
                    
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        pages.append({
                            "page_number": i + 1,
                            "text": page_text,
                            "confidence": 0.8  # Assume reasonable quality
                        })
                        full_text += page_text + "\n"
                    
                    return {
                        "text": full_text,
                        "pages": pages,
                        "confidence": 0.8
                    }
                
                except ImportError:
                    logger.warning("PyPDF2 not available for PDF processing")
                    raise ValueError("PDF processing not available")
            
            else:
                raise ValueError(f"Unsupported mime type: {mime_type}")
        
        except Exception as e:
            logger.error(f"Fallback text extraction failed: {e}")
            raise


class DatabaseManager:
    """Manages document and clause storage."""
    
    def __init__(self):
        self._firestore_client = None
        self._bigquery_client = None
        
        # In-memory storage for demo (replace with real DB in production)
        self.documents: Dict[str, DocumentMetadata] = {}
        self.clauses: Dict[str, ClauseRecord] = {}
        self.reviews: Dict[str, HumanReviewResponse] = {}
        self.exports: Dict[str, Dict[str, Any]] = {}
    
    async def store_document(self, document: DocumentMetadata) -> bool:
        """Store document metadata."""
        try:
            self.documents[document.doc_id] = document
            return True
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            return False
    
    async def store_clauses(self, clauses: List[ClauseRecord]) -> bool:
        """Store extracted clauses."""
        try:
            for clause in clauses:
                self.clauses[clause.clause_id] = clause
            return True
        except Exception as e:
            logger.error(f"Failed to store clauses: {e}")
            return False
    
    async def get_document(self, doc_id: str, user_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata."""
        return self.documents.get(doc_id)
    
    async def get_clauses(
        self, 
        doc_id: str, 
        page: int = 1, 
        page_size: int = 50,
        clause_type: Optional[str] = None
    ) -> List[ClauseRecord]:
        """Retrieve clauses for a document."""
        doc_clauses = [
            clause for clause in self.clauses.values() 
            if clause.doc_id == doc_id
        ]
        
        if clause_type:
            doc_clauses = [
                clause for clause in doc_clauses 
                if clause.clause_type.value == clause_type
            ]
        
        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        return doc_clauses[start_idx:end_idx]
    
    async def delete_document(self, doc_id: str, user_id: str) -> bool:
        """Delete document and all associated data."""
        try:
            # Remove document
            if doc_id in self.documents:
                del self.documents[doc_id]
            
            # Remove associated clauses
            clause_ids_to_remove = [
                clause_id for clause_id, clause in self.clauses.items()
                if clause.doc_id == doc_id
            ]
            
            for clause_id in clause_ids_to_remove:
                del self.clauses[clause_id]
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False


class DocumentOrchestrator:
    """Main orchestrator coordinating all document processing components."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.db_manager = DatabaseManager()
        self.extractor = AdaptiveExtractor()
        self.retriever: Optional[PolyVectorRetriever] = None
        self.rag_engine: Optional[ConstrainedRAGEngine] = None
        self.verifier: Optional[VerificationPipeline] = None
    
    async def initialize(self):
        """Initialize all components."""
        try:
            self.retriever = create_retriever()
            self.rag_engine = create_rag_engine()
            self.verifier = create_verifier()
            logger.info("Document orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up orchestrator resources")
        # Cleanup code here if needed
    
    async def health_check(self) -> Dict[str, str]:
        """Check health of all components."""
        health = {
            "document_processor": "healthy",
            "database": "healthy",
            "retriever": "healthy" if self.retriever else "unavailable",
            "rag_engine": "healthy" if self.rag_engine else "unavailable",
            "verifier": "healthy" if self.verifier else "unavailable"
        }
        return health
    
    async def upload_document(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        user_id: str,
        ephemeral: bool = False
    ) -> DocumentUploadResponse:
        """Upload and process a document."""
        try:
            # Create document metadata
            doc_id = str(uuid.uuid4())
            doc_metadata = DocumentMetadata(
                doc_id=doc_id,
                filename=filename,
                file_size=len(content),
                mime_type=content_type,
                ephemeral=ephemeral,
                processing_status="processing"
            )
            
            # Store document metadata
            await self.db_manager.store_document(doc_metadata)
            
            # Process document with Document AI
            processed_doc = await self.document_processor.process_document(content, content_type)
            
            # Extract facts deterministically
            immediate_facts = []
            for page_info in processed_doc["pages"]:
                page_facts = self.extractor.extract(
                    page_info["text"], 
                    doc_id, 
                    page_info["page_number"]
                )
                immediate_facts.extend(page_facts)
            
            # Update document status
            doc_metadata.processing_status = "completed"
            doc_metadata.ocr_confidence = processed_doc["confidence"]
            doc_metadata.page_count = len(processed_doc["pages"])
            doc_metadata.extracted_facts_count = len(immediate_facts)
            
            # Create and store clause records
            clauses = await self._create_clause_records(doc_id, processed_doc, immediate_facts)
            await self.db_manager.store_clauses(clauses)
            
            # Index clauses for retrieval
            if self.retriever:
                for clause in clauses:
                    await self.retriever.index_clause(clause)
            
            doc_metadata.clause_count = len(clauses)
            await self.db_manager.store_document(doc_metadata)
            
            return DocumentUploadResponse(
                doc_id=doc_id,
                immediate_facts=immediate_facts[:10],  # Return first 10 facts
                processing_status="completed"
            )
        
        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            raise
    
    async def get_document_clauses(
        self,
        doc_id: str,
        user_id: str,
        page: int = 1,
        page_size: int = 50,
        clause_type: Optional[str] = None
    ) -> DocumentClausesResponse:
        """Get clauses from a document."""
        try:
            # Verify document exists and user has access
            document = await self.db_manager.get_document(doc_id, user_id)
            if not document:
                raise ValueError(f"Document {doc_id} not found")
            
            # Get clauses
            clauses = await self.db_manager.get_clauses(doc_id, page, page_size, clause_type)
            
            # Get total count
            all_clauses = await self.db_manager.get_clauses(doc_id, 1, 10000, clause_type)
            total_count = len(all_clauses)
            
            return DocumentClausesResponse(
                doc_id=doc_id,
                clauses=clauses,
                total_count=total_count,
                page=page,
                page_size=page_size
            )
        
        except Exception as e:
            logger.error(f"Failed to get clauses: {e}")
            raise
    
    async def process_query(self, request: QueryRequest, user_id: str) -> QueryResponse:
        """Process a user query against a document."""
        try:
            # Verify document access
            document = await self.db_manager.get_document(request.doc_id, user_id)
            if not document:
                raise ValueError(f"Document {request.doc_id} not found")
            
            # Retrieve relevant clauses
            if not self.retriever:
                raise ValueError("Retrieval system not available")
            
            retrieval_result = await self.retriever.retrieve(
                query=request.query,
                top_k=request.top_k,
                doc_id=request.doc_id,
                clause_type=request.clause_type_filter.value if request.clause_type_filter else None,
                jurisdiction=request.jurisdiction_filter
            )
            
            # Generate ClauseCards
            clause_cards = []
            if self.rag_engine and retrieval_result.candidates:
                clause_card = await self.rag_engine.generate_clause_card(
                    retrieval_result, request.query
                )
                
                if clause_card:
                    # Verify the generated content
                    if self.verifier:
                        evidence_spans = {
                            candidate.clause_id: candidate.clause_record.content_text
                            for candidate in retrieval_result.candidates
                        }
                        
                        verified_card, verification_report = await self.verifier.verify_clause_card(
                            clause_card, evidence_spans
                        )
                        clause_cards.append(verified_card)
                    else:
                        clause_cards.append(clause_card)
            
            return QueryResponse(
                query=request.query,
                doc_id=request.doc_id,
                clause_cards=clause_cards,
                processing_time_ms=retrieval_result.processing_time_ms,
                total_candidates=retrieval_result.total_candidates,
                retrieval_method=retrieval_result.retrieval_method,
                hybrid_scores=None  # Can be populated from retrieval_result.metadata
            )
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise
    
    async def request_human_review(self, request: HumanReviewRequest) -> str:
        """Request human review for a clause."""
        try:
            review_id = str(uuid.uuid4())
            
            # Create review record
            review = HumanReviewResponse(
                review_id=review_id,
                reviewer_id="",  # Will be assigned when picked up
                status="pending",
                reviewed_clause_card=None,
                reviewer_notes=None,
                confidence_override=None,
                approved_for_use=False
            )
            
            self.db_manager.reviews[review_id] = review
            
            # In production, this would notify attorney reviewers
            logger.info(f"Human review requested: {review_id}")
            
            return review_id
        
        except Exception as e:
            logger.error(f"Human review request failed: {e}")
            raise
    
    async def get_review_status(self, review_id: str, user_id: str) -> HumanReviewResponse:
        """Get status of human review."""
        review = self.db_manager.reviews.get(review_id)
        if not review:
            raise ValueError(f"Review {review_id} not found")
        return review
    
    async def start_export(self, request: ExportRequest, user_id: str) -> str:
        """Start document export process."""
        try:
            export_id = str(uuid.uuid4())
            
            export_record = {
                "export_id": export_id,
                "doc_id": request.doc_id,
                "format": request.export_format,
                "status": "processing",
                "created_at": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
            
            self.db_manager.exports[export_id] = export_record
            
            return export_id
        
        except Exception as e:
            logger.error(f"Export start failed: {e}")
            raise
    
    async def get_export_status(self, export_id: str, user_id: str) -> Dict[str, Any]:
        """Get export status."""
        export_record = self.db_manager.exports.get(export_id)
        if not export_record:
            raise ValueError(f"Export {export_id} not found")
        return export_record
    
    async def delete_document(self, doc_id: str, user_id: str):
        """Delete a document."""
        success = await self.db_manager.delete_document(doc_id, user_id)
        if not success:
            raise ValueError(f"Failed to delete document {doc_id}")
    
    async def get_system_metrics(self) -> SystemMetrics:
        """Get system performance metrics."""
        # In production, these would come from monitoring systems
        return SystemMetrics(
            avg_processing_time_ms=1500.0,
            p95_processing_time_ms=3000.0,
            throughput_docs_per_hour=120.0,
            hallucination_rate=0.03,
            extractor_accuracy=0.98,
            retrieval_precision_at_5=0.87,
            verifier_pass_rate=0.92,
            documents_processed=len(self.db_manager.documents),
            queries_executed=0,  # Would track this
            human_reviews_requested=len(self.db_manager.reviews),
            clauses_verified=len(self.db_manager.clauses)
        )
    
    async def process_document_background(self, doc_id: str):
        """Background processing for heavy document operations."""
        try:
            logger.info(f"Starting background processing for {doc_id}")
            # Additional processing that can happen asynchronously
            # For example: detailed analysis, risk scoring, etc.
            
            await asyncio.sleep(2)  # Simulate processing
            logger.info(f"Background processing complete for {doc_id}")
        
        except Exception as e:
            logger.error(f"Background processing failed for {doc_id}: {e}")
    
    async def generate_export_background(self, export_id: str):
        """Background export generation."""
        try:
            logger.info(f"Starting export generation for {export_id}")
            
            # Simulate export generation
            await asyncio.sleep(5)
            
            # Update export status
            if export_id in self.db_manager.exports:
                self.db_manager.exports[export_id]["status"] = "completed"
                self.db_manager.exports[export_id]["download_url"] = f"/downloads/{export_id}.pdf"
            
            logger.info(f"Export generation complete for {export_id}")
        
        except Exception as e:
            logger.error(f"Export generation failed for {export_id}: {e}")
            if export_id in self.db_manager.exports:
                self.db_manager.exports[export_id]["status"] = "failed"
    
    async def _create_clause_records(
        self, 
        doc_id: str, 
        processed_doc: Dict[str, Any],
        extracted_facts: List[ExtractedFact]
    ) -> List[ClauseRecord]:
        """Create clause records from processed document."""
        clauses = []
        
        # Simple clause segmentation - in production, use more sophisticated methods
        full_text = processed_doc["text"]
        
        # Split by common section markers
        import re
        section_pattern = r'\n\s*(?:\d+\.|\([a-z]\)|\([0-9]+\)|[A-Z]+\.)\s*[A-Z]'
        sections = re.split(section_pattern, full_text)
        
        for i, section_text in enumerate(sections):
            if len(section_text.strip()) < 50:  # Skip very short sections
                continue
            
            clause_id = f"{doc_id}_clause_{i+1}"
            
            # Find relevant extracted facts for this section
            section_facts = [
                fact for fact in extracted_facts
                if self._fact_in_section(fact, section_text)
            ]
            
            clause = ClauseRecord(
                clause_id=clause_id,
                doc_id=doc_id,
                section_title=f"Section {i+1}",
                content_text=section_text.strip(),
                page=1,  # Simplified - would calculate actual page
                line_start=1,
                line_end=len(section_text.split('\n')),
                extracted_facts=section_facts
            )
            
            clauses.append(clause)
        
        return clauses
    
    def _fact_in_section(self, fact: ExtractedFact, section_text: str) -> bool:
        """Check if an extracted fact belongs to a section."""
        # Simple heuristic - in production, use more sophisticated matching
        if hasattr(fact.value, 'get') and isinstance(fact.value, dict):
            fact_text = str(fact.value.get('raw', ''))
            return fact_text.lower() in section_text.lower()
        return str(fact.value).lower() in section_text.lower()