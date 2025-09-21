"""
Configuration management for the Legal Document Analyzer MVP.
Simplified configuration for hackathon prototype using Firestore Native (not Firebase).
"""

from pydantic_settings import BaseSettings
from typing import Dict, List, Optional
import os
from functools import lru_cache


class Settings(BaseSettings):
    """MVP application configuration settings - Firestore Native focused."""
    
    # Application
    APP_NAME: str = "Legal Document Analyzer MVP"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    API_WORKERS: int = 1
    
    # Request handling
    MAX_FILE_SIZE_MB: int = 50
    MAX_CONCURRENT_UPLOADS: int = 10
    REQUEST_TIMEOUT_SECONDS: int = 300
    
    # CORS Configuration - for frontend integration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",  # Alternative dev port
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    ALLOW_CREDENTIALS: bool = True
    
    # Google Cloud Platform
    GOOGLE_CLOUD_PROJECT: str = "legal-doc-analyzer-mvp"
    GOOGLE_CLOUD_REGION: str = "us-central1"
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    
    # Cloud Storage
    GCS_BUCKET: str = "legal-docs-mvp"
    GCS_REGION: str = "us-central1"
    
    # Document AI
    DOCUMENT_AI_PROCESSOR_ID: str = "your-processor-id"
    DOCUMENT_AI_LOCATION: str = "us"
    
    # Vertex AI - Enhanced with latest models
    VERTEX_AI_LOCATION: str = "us-central1"
    VERTEX_EMBEDDING_MODEL: str = "text-embedding-004"
    VERTEX_GENERATION_MODEL: str = "gemini-2.5-pro"
    GENERATION_TEMPERATURE: float = 0.1
    GENERATION_MAX_TOKENS: int = 2048
    GENERATION_TOP_P: float = 0.95
    
    # Vertex AI Matching Engine
    MATCHING_ENGINE_INDEX_ENDPOINT: str = ""
    MATCHING_ENGINE_DEPLOYED_INDEX_ID: str = ""
    
    # Firestore Native Configuration (NO Firebase Auth for MVP)
    FIRESTORE_DATABASE: str = "(default)"
    FIRESTORE_COLLECTION_DOCUMENTS: str = "documents"
    FIRESTORE_COLLECTION_CLAUSES: str = "clauses"
    
    # Hybrid Search Configuration
    HYBRID_SEARCH_ENABLED: bool = True
    HYDE_ENABLED: bool = True
    HYDE_TEMPERATURE: float = 0.3
    CROSS_ENCODER_ENABLED: bool = True
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Poly-Vector Search Weights
    DENSE_VECTOR_WEIGHT: float = 0.3
    SPARSE_VECTOR_WEIGHT: float = 0.2
    HYDE_VECTOR_WEIGHT: float = 0.2
    CROSS_ENCODER_WEIGHT: float = 0.3
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K_CANDIDATES: int = 20
    RETRIEVAL_FINAL_TOP_N: int = 8
    
    # Verification Configuration
    VERIFICATION_THRESHOLD: float = 0.8
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = True
    
    @property
    def hybrid_search_weights(self) -> Dict[str, float]:
        """Get hybrid search scoring weights as dict."""
        return {
            "dense": self.DENSE_VECTOR_WEIGHT,
            "sparse": self.SPARSE_VECTOR_WEIGHT,
            "hyde": self.HYDE_VECTOR_WEIGHT,
            "cross_encoder": self.CROSS_ENCODER_WEIGHT
        }
    
    @property
    def gemini_config(self) -> Dict[str, any]:
        """Get Gemini 2.5 Pro generation configuration."""
        return {
            "model": self.VERTEX_GENERATION_MODEL,
            "temperature": self.GENERATION_TEMPERATURE,
            "max_output_tokens": self.GENERATION_MAX_TOKENS,
            "top_p": self.GENERATION_TOP_P
        }
    
    @property
    def vertex_ai_config(self) -> Dict[str, str]:
        """Vertex AI configuration."""
        return {
            "project": self.GOOGLE_CLOUD_PROJECT,
            "location": self.VERTEX_AI_LOCATION,
            "embedding_model": self.VERTEX_EMBEDDING_MODEL,
            "generation_model": self.VERTEX_GENERATION_MODEL
        }
    
    @property
    def document_ai_config(self) -> Dict[str, str]:
        """Document AI configuration."""
        return {
            "project_id": self.GOOGLE_CLOUD_PROJECT,
            "location": self.DOCUMENT_AI_LOCATION,
            "processor_id": self.DOCUMENT_AI_PROCESSOR_ID
        }
    
    @property
    def cloud_storage_config(self) -> Dict[str, str]:
        """Cloud Storage configuration."""
        return {
            "bucket_name": self.GCS_BUCKET,
            "project": self.GOOGLE_CLOUD_PROJECT,
            "region": self.GCS_REGION
        }
    
    @property
    def firestore_config(self) -> Dict[str, str]:
        """Firestore Native configuration - NO Firebase Auth."""
        return {
            "project": self.GOOGLE_CLOUD_PROJECT,
            "database": self.FIRESTORE_DATABASE,
            "documents_collection": self.FIRESTORE_COLLECTION_DOCUMENTS,
            "clauses_collection": self.FIRESTORE_COLLECTION_CLAUSES
        }
    
    def validate_hybrid_search_weights(self) -> bool:
        """Validate that hybrid search weights sum to 1.0."""
        total_weight = sum(self.hybrid_search_weights.values())
        return abs(total_weight - 1.0) < 0.01
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance  
settings = get_settings()