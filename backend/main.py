"""
Legal Document Analyzer MVP - FastAPI Application
Hackathon prototype with no authentication, three core endpoints.
Features: Document upload, querying, and clause extraction with hybrid search.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.models.mvp_schemas import (
    DocumentMetadata, ExtractedClause, RAGAnswer, VectorSearchResponse,
    ProcessingStatus, ClauseType, DocumentUploadResponse, QueryRequest, QueryResponse
)
from src.models.config import settings
from src.services.mvp_document_processing import document_processing_service
from src.services.mvp_hybrid_search import hybrid_search_service
from src.services.mvp_rag_generation import rag_generation_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="MVP Legal Document Demystification Tool for Hackathon",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.ENVIRONMENT
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info("MVP mode: No authentication required")
    
    # Validate hybrid search weights
    if not settings.validate_hybrid_search_weights():
        logger.warning("Hybrid search weights do not sum to 1.0")


# Core MVP Endpoints

@app.post("/v1/docs/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> DocumentUploadResponse:
    """
    Upload and process a legal document.
    
    This endpoint:
    1. Validates the uploaded file
    2. Stores it in Google Cloud Storage
    3. Processes it with Document AI for OCR
    4. Extracts and classifies clauses
    5. Generates embeddings and stores in vector database
    6. Returns processing status and document metadata
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Reset file pointer
        await file.seek(0)
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        logger.info(f"Processing document upload: {file.filename} (ID: {doc_id})")
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            doc_id,
            file,
            content
        )
        
        # Return immediate response
        response = DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            status=ProcessingStatus.PROCESSING,
            message="Document uploaded successfully. Processing in background.",
            created_at=datetime.utcnow()
        )
        
        logger.info(f"Document upload initiated: {doc_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Document upload failed")


@app.post("/v1/docs/{doc_id}/query", response_model=QueryResponse)
async def query_document(
    doc_id: str,
    request: QueryRequest
) -> QueryResponse:
    """
    Query a document using hybrid search and RAG.
    
    This endpoint:
    1. Validates the document exists
    2. Performs hybrid search (dense + sparse + HyDE + cross-encoder)
    3. Generates RAG answer using Gemini 2.5 Pro
    4. Returns answer with retrieved context and confidence scores
    """
    try:
        logger.info(f"Querying document {doc_id}: {request.question[:50]}...")
        
        # Check if document exists and is processed
        doc_metadata = await get_document_metadata(doc_id)
        if not doc_metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if doc_metadata.status != ProcessingStatus.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=f"Document is still processing. Status: {doc_metadata.status}"
            )
        
        # Perform hybrid search
        search_response = await hybrid_search_service.hybrid_search(
            query=request.question,
            doc_id=doc_id,
            clause_type=request.clause_type,
            top_k=request.max_results or settings.RETRIEVAL_FINAL_TOP_N,
            use_hyde=settings.HYDE_ENABLED,
            use_cross_encoder=settings.CROSS_ENCODER_ENABLED
        )
        
        if not search_response.results:
            return QueryResponse(
                doc_id=doc_id,
                question=request.question,
                answer="I couldn't find relevant information in the document to answer your question.",
                confidence_score=0.0,
                retrieved_clauses=[],
                search_metadata={
                    "search_method": "hybrid",
                    "results_count": 0,
                    "query_time": search_response.query_time
                }
            )
        
        # Generate RAG answer
        rag_answer = await rag_generation_service.generate_rag_answer(
            question=request.question,
            retrieved_clauses=[result.text for result in search_response.results],
            context_metadata={
                "doc_id": doc_id,
                "clause_types": [result.clause_type.value for result in search_response.results]
            }
        )
        
        # Build response
        response = QueryResponse(
            doc_id=doc_id,
            question=request.question,
            answer=rag_answer.answer,
            confidence_score=rag_answer.confidence_score,
            retrieved_clauses=[
                {
                    "clause_id": result.clause_id,
                    "text": result.text,
                    "clause_type": result.clause_type.value,
                    "similarity_score": result.similarity_score,
                    "metadata": result.metadata
                }
                for result in search_response.results
            ],
            search_metadata={
                "search_method": "hybrid_poly_vector",
                "results_count": len(search_response.results),
                "query_time": search_response.query_time,
                "hyde_enabled": settings.HYDE_ENABLED,
                "cross_encoder_enabled": settings.CROSS_ENCODER_ENABLED,
                "search_weights": settings.hybrid_search_weights
            }
        )
        
        logger.info(f"Query completed for {doc_id}: confidence={rag_answer.confidence_score:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed for document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Query processing failed")


@app.get("/v1/docs/{doc_id}/clauses", response_model=List[ExtractedClause])
async def get_document_clauses(
    doc_id: str,
    clause_type: Optional[ClauseType] = None,
    limit: int = 50,
    offset: int = 0
) -> List[ExtractedClause]:
    """
    Get all extracted clauses from a document.
    
    This endpoint:
    1. Validates the document exists and is processed
    2. Retrieves all clauses from Firestore
    3. Optionally filters by clause type
    4. Returns paginated results
    """
    try:
        logger.info(f"Retrieving clauses for document {doc_id}")
        
        # Check if document exists and is processed
        doc_metadata = await get_document_metadata(doc_id)
        if not doc_metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if doc_metadata.status != ProcessingStatus.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=f"Document is still processing. Status: {doc_metadata.status}"
            )
        
        # Retrieve clauses from Firestore
        clauses = await document_processing_service.get_document_clauses(
            doc_id=doc_id,
            clause_type=clause_type,
            limit=limit,
            offset=offset
        )
        
        logger.info(f"Retrieved {len(clauses)} clauses for document {doc_id}")
        return clauses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve clauses for document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve clauses")


# Additional utility endpoints

@app.get("/v1/docs/{doc_id}/status", response_model=DocumentMetadata)
async def get_document_status(doc_id: str) -> DocumentMetadata:
    """Get document processing status and metadata."""
    try:
        doc_metadata = await get_document_metadata(doc_id)
        if not doc_metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return doc_metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get document status")


@app.get("/v1/search/test")
async def test_search_capabilities():
    """Test endpoint for search capabilities."""
    return {
        "hybrid_search_enabled": settings.HYBRID_SEARCH_ENABLED,
        "hyde_enabled": settings.HYDE_ENABLED,
        "cross_encoder_enabled": settings.CROSS_ENCODER_ENABLED,
        "search_weights": settings.hybrid_search_weights,
        "models": {
            "embedding_model": settings.VERTEX_EMBEDDING_MODEL,
            "generation_model": settings.VERTEX_GENERATION_MODEL,
            "cross_encoder_model": settings.CROSS_ENCODER_MODEL
        }
    }


# Helper functions

async def process_document_background(
    doc_id: str,
    file: UploadFile,
    content: bytes
):
    """Background task for document processing."""
    try:
        logger.info(f"Starting background processing for document {doc_id}")
        
        # Process document
        await document_processing_service.upload_and_process_document(
            doc_id=doc_id,
            file_content=content,
            filename=file.filename,
            content_type=file.content_type or "application/pdf"
        )
        
        logger.info(f"Document processing completed for {doc_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for document {doc_id}: {str(e)}")
        # Update document status to failed
        await document_processing_service.update_document_status(
            doc_id=doc_id,
            status=ProcessingStatus.FAILED,
            error_message=str(e)
        )


async def get_document_metadata(doc_id: str) -> Optional[DocumentMetadata]:
    """Get document metadata from Firestore."""
    try:
        return await document_processing_service.get_document_metadata(doc_id)
    except Exception as e:
        logger.error(f"Failed to get metadata for document {doc_id}: {str(e)}")
        return None


# Exception handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.is_development(),
        log_level=settings.LOG_LEVEL.lower()
    )