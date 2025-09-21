"""
Simplified FastAPI application for the Legal Document Analyzer.
Development version without external dependencies.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    processing_time: float

class DocumentAnalysisResult(BaseModel):
    document_id: str
    text: str
    entities: List[Dict[str, Any]]
    clauses: List[Dict[str, Any]]
    risk_factors: List[Dict[str, Any]]
    risk_score: int
    summary: str
    confidence: float

class QueryRequest(BaseModel):
    query: str
    document_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    timestamp: datetime

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str

# Create FastAPI app
app = FastAPI(
    title="Legal Document Analyzer API",
    description="AI-powered legal document analysis and consultation platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for development
documents_db: Dict[str, Dict] = {}
analysis_db: Dict[str, DocumentAnalysisResult] = {}

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Legal Document Analyzer API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a legal document"""
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.doc', '.docx', '.txt')):
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload PDF, DOC, DOCX, or TXT files."
            )
        
        # Generate document ID
        document_id = f"doc_{int(time.time())}_{hash(file.filename) % 10000}"
        
        # Read file content
        content = await file.read()
        
        # Store document metadata
        documents_db[document_id] = {
            "id": document_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "uploaded_at": datetime.now().isoformat(),
            "content": content
        }
        
        # Process document in background
        background_tasks.add_task(process_document_background, document_id, content, file.filename)
        
        processing_time = time.time() - start_time
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="uploaded",
            message="Document uploaded successfully. Analysis in progress.",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Document upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_document_background(document_id: str, content: bytes, filename: str):
    """Background task to process document"""
    try:
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Mock analysis (replace with real AI processing)
        mock_text = f"Sample extracted text from {filename}. This is a mock analysis for development."
        
        analysis = DocumentAnalysisResult(
            document_id=document_id,
            text=mock_text,
            entities=[
                {"type": "DATE", "text": "2024-01-01", "confidence": 0.9},
                {"type": "MONEY", "text": "$50,000", "confidence": 0.85},
                {"type": "ORGANIZATION", "text": "Legal Corp", "confidence": 0.8}
            ],
            clauses=[
                {"type": "termination", "count": 2, "risk_level": "medium"},
                {"type": "liability", "count": 1, "risk_level": "high"},
                {"type": "payment", "count": 3, "risk_level": "low"}
            ],
            risk_factors=[
                {"keyword": "unlimited liability", "severity": 8, "context": "Section 5.2"},
                {"keyword": "automatic renewal", "severity": 6, "context": "Section 3.1"}
            ],
            risk_score=75,
            summary="This document contains standard legal terms with some high-risk clauses that require attention.",
            confidence=0.85
        )
        
        # Store analysis
        analysis_db[document_id] = analysis
        
        logger.info(f"Document {document_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Document processing error for {document_id}: {str(e)}")

@app.get("/api/documents/{document_id}/analysis", response_model=DocumentAnalysisResult)
async def get_document_analysis(document_id: str):
    """Get document analysis results"""
    if document_id not in analysis_db:
        if document_id not in documents_db:
            raise HTTPException(status_code=404, detail="Document not found")
        raise HTTPException(status_code=202, detail="Analysis in progress")
    
    return analysis_db[document_id]

@app.get("/api/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """Get document processing status"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_data = documents_db[document_id]
    analysis = analysis_db.get(document_id)
    
    return {
        "id": document_id,
        "filename": doc_data["filename"],
        "content_type": doc_data.get("content_type"),
        "content_length": doc_data.get("size"),
        "status": "completed" if analysis else "processing",
        "uploaded_at": doc_data["uploaded_at"],
        "processed_at": datetime.now().isoformat() if analysis else None,
        "total_pages": analysis.text.count('\n') + 1 if analysis else None,
        "total_clauses": len(analysis.clauses) if analysis else None
    }

@app.post("/api/chat/query", response_model=QueryResponse)
async def chat_query(request: QueryRequest):
    """Process a chat query about a document"""
    try:
        # Mock AI response (replace with real AI processing)
        mock_responses = [
            "Based on the document analysis, this appears to be a standard legal agreement with moderate risk factors.",
            "The key terms include payment obligations, termination clauses, and liability limitations.",
            "I recommend reviewing the automatic renewal clause and liability sections carefully.",
            "This contract contains some unusual provisions that may require legal consultation."
        ]
        
        # Simple response selection based on query keywords
        response = mock_responses[0]  # Default response
        
        if "risk" in request.query.lower():
            response = "The document has a risk score of 75/100. Main concerns include unlimited liability and automatic renewal clauses."
        elif "payment" in request.query.lower():
            response = "Payment terms require net 30 days with late fees of 1.5% per month."
        elif "termination" in request.query.lower():
            response = "The contract can be terminated with 30 days written notice by either party."
        elif "negotiate" in request.query.lower():
            response = "I recommend negotiating the liability cap, termination notice period, and automatic renewal terms."
        
        return QueryResponse(
            answer=response,
            confidence=0.8,
            sources=["Document Analysis", "Legal Knowledge Base"],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Chat query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/api/documents", response_model=List[Dict[str, Any]])
async def list_documents():
    """List all uploaded documents"""
    documents = []
    for doc_id, doc_data in documents_db.items():
        analysis = analysis_db.get(doc_id)
        documents.append({
            "id": doc_id,
            "filename": doc_data["filename"],
            "uploaded_at": doc_data["uploaded_at"],
            "size": doc_data["size"],
            "analysis_status": "completed" if analysis else "processing",
            "risk_score": analysis.risk_score if analysis else None
        })
    return documents

@app.get("/api/documents/{document_id}/download")
async def download_document(document_id: str):
    """Download original document file"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_data = documents_db[document_id]
    
    # For now, return the document content as text since we don't store files
    # In a real implementation, this would return the actual file
    analysis = analysis_db.get(document_id)
    if analysis:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content=analysis.text,
            headers={
                "Content-Disposition": f"attachment; filename={doc_data['filename']}.txt"
            }
        )
    else:
        raise HTTPException(status_code=404, detail="Document content not available")

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its analysis"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from storage
    del documents_db[document_id]
    if document_id in analysis_db:
        del analysis_db[document_id]
    
    return {"message": "Document deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)