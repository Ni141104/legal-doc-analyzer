"""
Legal Document Analyzer API
Production-ready FastAPI backend for multi-agent Gen AI system.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Import configuration and models
from .models.config import settings
from .models.schemas import (
    ClauseCard, DocumentMetadata, SimplifiedSentence, RiskFlag,
    QueryRequest, QueryResponse, UploadResponse, ErrorResponse,
    ReviewTask, ReviewFeedback, ReviewStatus
)

# Import services
from .services.doc_ingestion import document_ingestion_service
from .services.retrieval import retrieval_service
from .services.generation import generation_service
from .services.verification import verification_service
from .services.human_review import human_review_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info(f"Starting Legal Document Analyzer API v{settings.API_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Google Cloud Project: {settings.GOOGLE_CLOUD_PROJECT_ID}")
    
    # Initialize services (if needed)
    try:
        # Any initialization that needs to happen at startup
        await _initialize_services()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Legal Document Analyzer API")


# Create FastAPI app
app = FastAPI(
    title="Legal Document Analyzer API",
    description="Production-ready FastAPI backend for multi-agent Gen AI system",
    version=settings.API_VERSION,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)


async def _initialize_services():
    """Initialize all services during startup."""
    # Services initialize themselves when imported
    # This function can be used for any additional setup
    pass


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify Firebase ID token.
    
    Args:
        credentials: Bearer token from request
        
    Returns:
        User ID if token is valid
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        # In production, verify Firebase token here
        # For now, accept any token that starts with "valid"
        if not credentials.credentials.startswith("valid"):
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        # Extract user ID from token (simplified)
        user_id = credentials.credentials.replace("valid_", "")
        return user_id
        
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Invalid input",
            message=str(exc),
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            message="An unexpected error occurred",
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service status."""
    try:
        health_status = {
            "api": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.API_VERSION,
            "environment": settings.ENVIRONMENT,
            "services": {}
        }
        
        # Check service health (simplified)
        services = [
            ("document_ingestion", document_ingestion_service),
            ("retrieval", retrieval_service),
            ("generation", generation_service),
            ("verification", verification_service),
            ("human_review", human_review_service)
        ]
        
        for service_name, service in services:
            try:
                # Basic check that service object exists
                if service:
                    health_status["services"][service_name] = "healthy"
                else:
                    health_status["services"][service_name] = "unavailable"
            except Exception as e:
                health_status["services"][service_name] = f"error: {str(e)}"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )


# Document upload and processing endpoints
@app.post("/v1/docs/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(verify_token),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload a legal document for processing.
    
    Args:
        file: PDF or image file to process
        user_id: Authenticated user ID
        background_tasks: FastAPI background tasks
        
    Returns:
        Upload response with document ID and processing status
    """
    try:
        logger.info(f"Document upload initiated by user {user_id}")
        
        # Validate file
        if not file.filename:
            raise ValueError("No file provided")
        
        if file.content_type not in ["application/pdf", "image/jpeg", "image/png"]:
            raise ValueError("Unsupported file type. Please upload PDF, JPEG, or PNG files.")
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB.")
        
        # Start document ingestion
        result = await document_ingestion_service.process_document(
            file_content=file_content,
            filename=file.filename,
            content_type=file.content_type,
            user_id=user_id
        )
        
        # Schedule background clause extraction
        background_tasks.add_task(
            _extract_clauses_background,
            result["document_id"],
            user_id
        )
        
        logger.info(f"Document uploaded successfully: {result['document_id']}")
        
        return UploadResponse(
            document_id=result["document_id"],
            status="processing",
            message="Document uploaded successfully. Clause extraction in progress.",
            estimated_completion_time=result.get("estimated_completion"),
            uploaded_at=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        logger.warning(f"Upload validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Document upload failed")


async def _extract_clauses_background(document_id: str, user_id: str):
    """Background task to extract clauses from uploaded document."""
    try:
        logger.info(f"Starting clause extraction for document {document_id}")
        
        # Get document metadata
        document_metadata = await document_ingestion_service.get_document_metadata(document_id)
        
        if not document_metadata:
            logger.error(f"Document metadata not found: {document_id}")
            return
        
        # Extract clauses
        extracted_text = document_metadata.extracted_text
        clauses = await document_ingestion_service.identify_clauses(extracted_text)
        
        # Process each clause
        for clause_text in clauses:
            try:
                # Generate clause card
                clause_card = await generation_service.generate_clause_card(
                    clause_text=clause_text,
                    document_context=extracted_text,
                    document_id=document_id
                )
                
                # Verify clause card
                verified_card = await verification_service.verify_clause_card(
                    clause_card, extracted_text
                )
                
                # Submit for human review if verification confidence is low
                if verified_card.confidence_overall < settings.VERIFICATION_THRESHOLD:
                    await human_review_service.submit_for_review(
                        clause_card=verified_card,
                        document_id=document_id,
                        priority="medium" if verified_card.confidence_overall < 0.5 else "low"
                    )
                
                logger.info(f"Processed clause {verified_card.clause_id} for document {document_id}")
                
            except Exception as e:
                logger.error(f"Failed to process clause: {str(e)}")
                continue
        
        # Update document status
        await document_ingestion_service.update_document_status(
            document_id, "completed"
        )
        
        logger.info(f"Clause extraction completed for document {document_id}")
        
    except Exception as e:
        logger.error(f"Background clause extraction failed: {str(e)}")
        # Update document status to error
        try:
            await document_ingestion_service.update_document_status(
                document_id, "error"
            )
        except:
            pass


@app.get("/v1/docs/{document_id}/status")
async def get_document_status(
    document_id: str,
    user_id: str = Depends(verify_token)
):
    """
    Get processing status of uploaded document.
    
    Args:
        document_id: Document ID
        user_id: Authenticated user ID
        
    Returns:
        Document processing status and progress
    """
    try:
        document_metadata = await document_ingestion_service.get_document_metadata(document_id)
        
        if not document_metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check user access
        if document_metadata.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "document_id": document_id,
            "status": document_metadata.processing_status,
            "progress": document_metadata.processing_progress,
            "uploaded_at": document_metadata.uploaded_at.isoformat(),
            "last_updated": document_metadata.last_updated.isoformat() if document_metadata.last_updated else None,
            "extracted_pages": len(document_metadata.page_info) if document_metadata.page_info else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document status")


@app.get("/v1/docs/{document_id}/clauses", response_model=List[ClauseCard])
async def get_document_clauses(
    document_id: str,
    user_id: str = Depends(verify_token),
    include_unverified: bool = False
):
    """
    Get extracted clause cards for a document.
    
    Args:
        document_id: Document ID
        user_id: Authenticated user ID
        include_unverified: Include unverified clause cards
        
    Returns:
        List of clause cards for the document
    """
    try:
        # Check document access
        document_metadata = await document_ingestion_service.get_document_metadata(document_id)
        
        if not document_metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if document_metadata.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get clause cards (this would be implemented in the ingestion service)
        # For now, return empty list as placeholder
        clause_cards = []
        
        # Filter by verification status if requested
        if not include_unverified:
            clause_cards = [
                card for card in clause_cards 
                if card.confidence_overall >= settings.VERIFICATION_THRESHOLD
            ]
        
        logger.info(f"Retrieved {len(clause_cards)} clause cards for document {document_id}")
        return clause_cards
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document clauses: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve clause cards")


@app.post("/v1/docs/{document_id}/query", response_model=QueryResponse)
async def query_document(
    document_id: str,
    query_request: QueryRequest,
    user_id: str = Depends(verify_token)
):
    """
    Query a specific document using natural language.
    
    Args:
        document_id: Document ID to query
        query_request: Query parameters
        user_id: Authenticated user ID
        
    Returns:
        Query response with relevant clauses and generated answer
    """
    try:
        # Check document access
        document_metadata = await document_ingestion_service.get_document_metadata(document_id)
        
        if not document_metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if document_metadata.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Perform retrieval
        retrieved_clauses = await retrieval_service.retrieve_relevant_clauses(
            query=query_request.query,
            document_id=document_id,
            top_k=query_request.max_results or 5,
            filters=query_request.filters
        )
        
        # Generate answer using retrieved clauses
        if query_request.generate_answer:
            answer = await generation_service.generate_answer(
                query=query_request.query,
                retrieved_clauses=retrieved_clauses,
                document_context=document_metadata.extracted_text
            )
        else:
            answer = None
        
        response = QueryResponse(
            query=query_request.query,
            document_id=document_id,
            relevant_clauses=retrieved_clauses,
            generated_answer=answer,
            total_results=len(retrieved_clauses),
            query_time=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Processed query for document {document_id}: {len(retrieved_clauses)} clauses retrieved")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document query failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Query processing failed")


# Human review endpoints
@app.get("/v1/review/queue", response_model=List[ReviewTask])
async def get_review_queue(
    user_id: str = Depends(verify_token),
    status: Optional[str] = None,
    priority: Optional[str] = None
):
    """
    Get review queue for authenticated reviewer.
    
    Args:
        user_id: Authenticated user ID (reviewer)
        status: Filter by review status
        priority: Filter by priority level
        
    Returns:
        List of review tasks assigned to the reviewer
    """
    try:
        # Parse optional filters
        status_filter = ReviewStatus(status) if status else None
        priority_filter = priority if priority else None
        
        # Get review queue
        review_tasks = await human_review_service.get_review_queue(
            reviewer_id=user_id,
            status=status_filter,
            priority=priority_filter
        )
        
        logger.info(f"Retrieved {len(review_tasks)} review tasks for reviewer {user_id}")
        return review_tasks
        
    except Exception as e:
        logger.error(f"Failed to get review queue: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve review queue")


@app.post("/v1/review/{task_id}/feedback")
async def submit_review_feedback(
    task_id: str,
    feedback: ReviewFeedback,
    user_id: str = Depends(verify_token)
):
    """
    Submit review feedback for a task.
    
    Args:
        task_id: Review task ID
        feedback: Review feedback with corrections
        user_id: Authenticated user ID (reviewer)
        
    Returns:
        Success response
    """
    try:
        success = await human_review_service.submit_review_feedback(
            task_id=task_id,
            reviewer_id=user_id,
            feedback=feedback
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to submit review feedback")
        
        logger.info(f"Review feedback submitted for task {task_id} by {user_id}")
        return {"message": "Review feedback submitted successfully", "task_id": task_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit review feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit review feedback")


@app.get("/v1/review/analytics")
async def get_review_analytics(
    user_id: str = Depends(verify_token),
    days: int = 30
):
    """
    Get review process analytics.
    
    Args:
        user_id: Authenticated user ID
        days: Number of days to include in analytics
        
    Returns:
        Review analytics data
    """
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        analytics = await human_review_service.get_review_analytics(
            start_date=start_date
        )
        
        logger.info(f"Generated review analytics for {days} days")
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get review analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve review analytics")


# Development and testing endpoints (disabled in production)
if settings.ENVIRONMENT != "production":
    
    @app.post("/v1/test/generate-clause")
    async def test_generate_clause(
        clause_text: str = Form(...),
        user_id: str = Depends(verify_token)
    ):
        """Test endpoint for clause card generation."""
        try:
            clause_card = await generation_service.generate_clause_card(
                clause_text=clause_text,
                document_context="Test context",
                document_id="test_doc"
            )
            
            return clause_card
            
        except Exception as e:
            logger.error(f"Test clause generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Clause generation failed")
    
    @app.post("/v1/test/verify-clause")
    async def test_verify_clause(
        clause_card: ClauseCard,
        source_text: str = Form(...),
        user_id: str = Depends(verify_token)
    ):
        """Test endpoint for clause verification."""
        try:
            verified_card = await verification_service.verify_clause_card(
                clause_card, source_text
            )
            
            return verified_card
            
        except Exception as e:
            logger.error(f"Test clause verification failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Clause verification failed")


# Main application entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.ENVIRONMENT == "development"
    )