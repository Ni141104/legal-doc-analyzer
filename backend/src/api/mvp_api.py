"""
Legal Document Analyzer MVP - Simplified API without Authentication
For hackathon prototype and development purposes only.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import json
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..models.schemas import (
    DocumentUploadResponse,
    DocumentAnalysisResult,
    QueryRequest,
    QueryResponse,
    ClauseExtractionResult,
    HealthCheckResponse
)
from ..models.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Legal Document Analyzer MVP",
    description="Simplified API for document analysis and querying - NO AUTHENTICATION",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock storage for development
documents_store: Dict[str, Dict[str, Any]] = {}
clauses_store: Dict[str, List[Dict[str, Any]]] = {}


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint with basic health check."""
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        services={
            "document_ai": "available",
            "vertex_ai": "available", 
            "vector_store": "available",
            "database": "available"
        }
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Detailed health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        services={
            "document_ai": "available",
            "vertex_ai": "available", 
            "vector_store": "available",
            "database": "available"
        }
    )


@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload and process a legal document.
    No authentication required for MVP.
    """
    start_time = time.time()
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.filename.lower().endswith(('.pdf', '.doc', '.docx', '.txt')):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF, DOC, DOCX, and TXT files are supported"
        )
    
    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Store document info
    documents_store[document_id] = {
        "id": document_id,
        "filename": file.filename,
        "status": "processing",
        "uploaded_at": datetime.now().isoformat(),
        "content_length": file_size,
        "content_type": file.content_type
    }
    
    # Process document in background
    background_tasks.add_task(process_document, document_id, content, file.filename)
    
    processing_time = time.time() - start_time
    
    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="processing",
        message="Document uploaded successfully and is being processed",
        processing_time=processing_time
    )


async def process_document(document_id: str, content: bytes, filename: str):
    """Process uploaded document (mock implementation for MVP)."""
    try:
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Mock analysis results
        mock_clauses = [
            {
                "id": f"clause_{i}",
                "text": f"Sample clause {i} from {filename}",
                "type": ["liability", "payment", "termination"][i % 3],
                "confidence": 0.85 + (i % 3) * 0.05,
                "page": (i // 3) + 1,
                "position": {"start": i * 100, "end": (i + 1) * 100}
            }
            for i in range(5)
        ]
        
        # Store clauses
        clauses_store[document_id] = mock_clauses
        
        # Update document status
        documents_store[document_id].update({
            "status": "completed",
            "total_pages": 3,
            "total_clauses": len(mock_clauses),
            "processed_at": datetime.now().isoformat()
        })
        
        logger.info(f"Document {document_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        documents_store[document_id]["status"] = "failed"


@app.get("/api/v1/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """Get document processing status."""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return documents_store[document_id]


@app.post("/api/v1/documents/{document_id}/query", response_model=QueryResponse)
async def query_document(
    document_id: str,
    request: QueryRequest
):
    """
    Query a processed document using hybrid search and RAG.
    No authentication required for MVP.
    """
    # Check if document exists
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = documents_store[document_id]
    if document["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Document is not ready for querying. Status: {document['status']}"
        )
    
    # Mock query processing
    await asyncio.sleep(1)  # Simulate processing time
    
    # Mock response based on query
    query = request.query.lower()
    
    if "liability" in query:
        answer = "Based on the document analysis, the liability clauses indicate limited liability provisions with caps on damages. The organization limits its liability to the amount paid under the contract."
        confidence = 0.92
        sources = ["Section 8.3 - Limitation of Liability", "Clause 15 - Indemnification"]
    elif "payment" in query or "fee" in query:
        answer = "The payment terms specify monthly payments due within 30 days of invoice. Late fees of 1.5% per month apply to overdue amounts."
        confidence = 0.88
        sources = ["Section 4.1 - Payment Terms", "Clause 12 - Late Fees"]
    elif "termination" in query or "cancel" in query:
        answer = "Either party may terminate this agreement with 30 days written notice. Immediate termination is allowed for material breach or bankruptcy."
        confidence = 0.90
        sources = ["Section 10.1 - Termination for Convenience", "Section 10.2 - Termination for Cause"]
    else:
        answer = f"I found relevant information about '{query}' in the document. The analysis shows standard contractual provisions with typical legal language and requirements."
        confidence = 0.75
        sources = ["General Document Analysis", "AI-Generated Summary"]
    
    return QueryResponse(
        answer=answer,
        confidence=confidence,
        sources=sources,
        metadata={
            "query": request.query,
            "document_id": document_id,
            "search_method": "hybrid_search",
            "processing_time": 1.2
        }
    )


@app.get("/api/v1/documents/{document_id}/clauses", response_model=ClauseExtractionResult)
async def get_document_clauses(
    document_id: str,
    clause_type: Optional[str] = Query(None, description="Filter by clause type"),
    limit: int = Query(20, ge=1, le=100, description="Number of clauses to return"),
    offset: int = Query(0, ge=0, description="Number of clauses to skip")
):
    """
    Get extracted clauses from a document.
    No authentication required for MVP.
    """
    # Check if document exists
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document_id not in clauses_store:
        raise HTTPException(status_code=404, detail="No clauses found for this document")
    
    clauses = clauses_store[document_id]
    
    # Filter by type if specified
    if clause_type:
        clauses = [c for c in clauses if c.get("type") == clause_type]
    
    # Apply pagination
    total = len(clauses)
    paginated_clauses = clauses[offset:offset + limit]
    
    # Get clause type statistics
    all_clauses = clauses_store[document_id]
    type_stats = {}
    for clause in all_clauses:
        clause_type_name = clause.get("type", "unknown")
        type_stats[clause_type_name] = type_stats.get(clause_type_name, 0) + 1
    
    return ClauseExtractionResult(
        document_id=document_id,
        clauses=paginated_clauses,
        total_clauses=total,
        clause_types=list(type_stats.keys()),
        statistics=type_stats,
        pagination={
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_more": offset + limit < total
        }
    )


@app.get("/api/v1/documents")
async def list_documents(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List all documents. No authentication required for MVP."""
    docs = list(documents_store.values())
    
    # Filter by status if specified
    if status:
        docs = [d for d in docs if d.get("status") == status]
    
    # Apply pagination
    total = len(docs)
    paginated_docs = docs[offset:offset + limit]
    
    return {
        "documents": paginated_docs,
        "total": total,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_more": offset + limit < total
        }
    }


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document. No authentication required for MVP."""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from stores
    del documents_store[document_id]
    if document_id in clauses_store:
        del clauses_store[document_id]
    
    return {"message": f"Document {document_id} deleted successfully"}


# Additional utility endpoints for development
@app.get("/api/v1/stats")
async def get_stats():
    """Get system statistics for development."""
    total_docs = len(documents_store)
    completed_docs = len([d for d in documents_store.values() if d.get("status") == "completed"])
    total_clauses = sum(len(clauses) for clauses in clauses_store.values())
    
    return {
        "total_documents": total_docs,
        "completed_documents": completed_docs,
        "processing_documents": total_docs - completed_docs,
        "total_clauses": total_clauses,
        "uptime_seconds": time.time(),
        "version": settings.APP_VERSION
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mvp_api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )