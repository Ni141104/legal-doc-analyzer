# Legal Document Analyzer MVP - Environment Setup Guide

## 🎯 Overview
This document provides all the necessary environment variables and setup instructions for the Legal Document Analyzer MVP, configured for hackathon/prototype use without authentication.

## 📋 Required Google Cloud APIs & Services

### Core Services
1. **Document AI** - For PDF/document processing
2. **Vertex AI** - For Gemini 2.5 Pro and embeddings
3. **Cloud Storage** - For document storage
4. **Firestore Native** - For metadata and clauses storage

### API Enablement Commands
```bash
gcloud services enable documentai.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable firestore.googleapis.com
```

## 🔑 Environment Variables

### Backend (.env)
```bash
# =============================================================================
# APPLICATION SETTINGS
# =============================================================================
APP_NAME=Legal Document Analyzer MVP
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
API_WORKERS=1
MAX_FILE_SIZE_MB=50
MAX_CONCURRENT_UPLOADS=10
REQUEST_TIMEOUT_SECONDS=300

# CORS Configuration (for frontend integration)
ALLOWED_ORIGINS=["http://localhost:3000","http://localhost:3001","http://localhost:5173","http://127.0.0.1:3000","http://127.0.0.1:5173"]
ALLOW_CREDENTIALS=true

# =============================================================================
# GOOGLE CLOUD PLATFORM CONFIGURATION
# =============================================================================

# Core GCP Settings
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json

# Google Cloud Storage
GCS_BUCKET=your-bucket-name
GCS_REGION=us-central1

# Document AI (for PDF processing)
DOCUMENT_AI_PROCESSOR_ID=your-document-ai-processor-id
DOCUMENT_AI_LOCATION=us

# Vertex AI Configuration (Gemini 2.5 Pro)
VERTEX_AI_LOCATION=us-central1
VERTEX_EMBEDDING_MODEL=text-embedding-004
VERTEX_GENERATION_MODEL=gemini-2.5-pro
GENERATION_TEMPERATURE=0.1
GENERATION_MAX_TOKENS=2048
GENERATION_TOP_P=0.95

# Vertex AI Matching Engine (for vector search)
MATCHING_ENGINE_INDEX_ENDPOINT=your-matching-engine-endpoint
MATCHING_ENGINE_DEPLOYED_INDEX_ID=your-deployed-index-id

# =============================================================================
# FIRESTORE NATIVE CONFIGURATION (NO FIREBASE AUTH)
# =============================================================================
FIRESTORE_DATABASE=(default)
FIRESTORE_COLLECTION_DOCUMENTS=documents
FIRESTORE_COLLECTION_CLAUSES=clauses

# =============================================================================
# HYBRID SEARCH CONFIGURATION
# =============================================================================
HYBRID_SEARCH_ENABLED=true
HYDE_ENABLED=true
HYDE_TEMPERATURE=0.3
CROSS_ENCODER_ENABLED=true
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Search Weights (must sum to 1.0)
DENSE_VECTOR_WEIGHT=0.3
SPARSE_VECTOR_WEIGHT=0.2
HYDE_VECTOR_WEIGHT=0.2
CROSS_ENCODER_WEIGHT=0.3

# Retrieval Configuration
RETRIEVAL_TOP_K_CANDIDATES=20
RETRIEVAL_FINAL_TOP_N=8

# Verification & Quality Settings
VERIFICATION_THRESHOLD=0.8
```

### Frontend (.env.local)
```bash
# =============================================================================
# API CONNECTION
# =============================================================================
NEXT_PUBLIC_API_BASE_URL=http://localhost:8080
NEXT_PUBLIC_API_VERSION=v1

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================
NODE_ENV=development
NEXT_PUBLIC_ENV=development

# =============================================================================
# FEATURE FLAGS (for prototype)
# =============================================================================
NEXT_PUBLIC_ENABLE_AUTH=false
NEXT_PUBLIC_ENABLE_DEMO_MODE=true
NEXT_PUBLIC_ENABLE_ANALYTICS=false
```

## 🛠️ Complete Setup Instructions

### 1. Create Google Cloud Project
```bash
# Create a new project
gcloud projects create your-project-id --name="Legal Doc Analyzer MVP"

# Set as active project
gcloud config set project your-project-id
```

### 2. Enable Required APIs
```bash
gcloud services enable documentai.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable firestore.googleapis.com
```

### 3. Create Service Account
```bash
gcloud iam service-accounts create legal-doc-analyzer-sa \
  --display-name="Legal Doc Analyzer Service Account"
```

### 4. Grant Required Permissions
```bash
# Document AI permissions
gcloud projects add-iam-policy-binding your-project-id \
  --member="serviceAccount:legal-doc-analyzer-sa@your-project-id.iam.gserviceaccount.com" \
  --role="roles/documentai.apiUser"

# Vertex AI permissions
gcloud projects add-iam-policy-binding your-project-id \
  --member="serviceAccount:legal-doc-analyzer-sa@your-project-id.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# Cloud Storage permissions
gcloud projects add-iam-policy-binding your-project-id \
  --member="serviceAccount:legal-doc-analyzer-sa@your-project-id.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Firestore permissions
gcloud projects add-iam-policy-binding your-project-id \
  --member="serviceAccount:legal-doc-analyzer-sa@your-project-id.iam.gserviceaccount.com" \
  --role="roles/datastore.user"
```

### 5. Create and Download Service Account Key
```bash
gcloud iam service-accounts keys create service-account-key.json \
  --iam-account=legal-doc-analyzer-sa@your-project-id.iam.gserviceaccount.com

# Move the key to the backend directory
mv service-account-key.json backend/
```

### 6. Create Cloud Storage Bucket
```bash
gsutil mb -p your-project-id -c STANDARD -l us-central1 gs://your-bucket-name
```

### 7. Create Document AI Processor
```bash
# Go to Document AI console (https://console.cloud.google.com/ai/document-ai)
# Create a "Form Parser" processor
# Note the processor ID and update DOCUMENT_AI_PROCESSOR_ID in .env
```

### 8. Initialize Firestore
```bash
gcloud firestore databases create --location=us-central1
```

### 9. Set Up Development Environment

#### Backend
```bash
cd backend
cp .env.example .env
# Edit .env with your actual values
pip install -r requirements.txt
python start_mvp.py
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 🚀 Quick Start (Minimum Setup)

For immediate testing without full Google Cloud setup:

1. **Backend**: Use `simple_main.py` for mock functionality
```bash
cd backend
python simple_main.py
```

2. **Frontend**: Start with default settings
```bash
cd frontend
npm run dev
```

3. **Access**: Open http://localhost:3000

## 📊 API Endpoints

### Document Upload
- **POST** `/api/v1/documents/upload`
- **GET** `/api/v1/documents/{id}/status`

### Document Querying
- **POST** `/api/v1/documents/{id}/query`
- **GET** `/api/v1/documents/{id}/clauses`

### Health & Status
- **GET** `/health`
- **GET** `/api/v1/stats`

## 🔧 Development Tools

### Backend
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health
- **Stats**: http://localhost:8080/api/v1/stats

### Frontend
- **Development Server**: http://localhost:3000
- **Production Build**: `npm run build`

## ⚠️ Important Notes

1. **No Authentication**: This setup is for prototype/demo purposes only
2. **Security**: Never commit service account keys to version control
3. **Costs**: Monitor Google Cloud usage to avoid unexpected charges
4. **Testing**: Use small documents for initial testing
5. **Error Handling**: Check logs in both frontend and backend for debugging

## 🤝 Troubleshooting

### Common Issues
1. **CORS Errors**: Verify ALLOWED_ORIGINS in backend .env
2. **API Connection**: Ensure backend is running on port 8080
3. **Google Cloud**: Verify service account permissions
4. **File Upload**: Check MAX_FILE_SIZE_MB setting

### Debug Commands
```bash
# Check backend health
curl http://localhost:8080/health

# Check frontend API connection
curl http://localhost:3000/api/health

# View backend logs
tail -f backend/logs/app.log

# View frontend dev logs
npm run dev --verbose
```

## 📚 Additional Resources

- [Google Cloud Document AI](https://cloud.google.com/document-ai)
- [Vertex AI Gemini](https://cloud.google.com/vertex-ai)
- [Firestore Documentation](https://cloud.google.com/firestore)
- [Next.js Documentation](https://nextjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Ready to analyze legal documents with AI! 🚀📄**