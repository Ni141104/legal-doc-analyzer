# Legal Document Analyzer - Backend API

A production-ready FastAPI backend for a multi-agent Gen AI system that processes legal documents using Google Cloud Platform services.

## Features

- **Document Processing**: Upload and OCR legal documents using Google Document AI
- **Hybrid Retrieval**: Poly-vector search combining dense embeddings, sparse BM25, HyDE, and cross-encoder reranking
- **AI Generation**: RAG-based clause card generation using Vertex AI Gemini with constrained JSON output
- **Verification System**: NLI-based entailment verification to prevent hallucinations
- **Human Review**: Attorney workflow with intelligent assignment and feedback loop
- **Production Ready**: Authentication, rate limiting, monitoring, and Google Cloud Run deployment

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI        │    │  Google Cloud   │
│   (React/Next)  │◄───┤   Backend        │◄───┤   Services      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Service Architecture                       │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│ Document        │ Retrieval       │ Generation      │ Human     │
│ Ingestion       │ Service         │ Service         │ Review    │
│                 │                 │                 │           │
│ • Document AI   │ • Elasticsearch │ • Vertex AI     │ • Queue   │
│ • Cloud Storage │ • Vector Search │ • Gemini        │ • Feedback│
│ • Text Extract  │ • HyDE          │ • RAG           │ • Analytics│
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud Platform account
- Docker (for containerized deployment)
- gcloud CLI configured

### Local Development

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd legal-doc-analyzer/backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Required Environment Variables**:
   ```bash
   # Google Cloud Configuration
   GOOGLE_CLOUD_PROJECT_ID=your-project-id
   GOOGLE_CLOUD_REGION=us-central1
   
   # Document AI
   DOCUMENTAI_PROCESSOR_ID=your-processor-id
   
   # Vertex AI
   VERTEX_AI_ENDPOINT_ID=your-endpoint-id
   VERTEX_AI_MODEL_NAME=gemini-1.5-pro
   
   # Storage
   STORAGE_BUCKET_NAME=your-bucket-name
   KMS_KEY_NAME=your-kms-key
   
   # Search Services
   ELASTICSEARCH_URL=your-elasticsearch-url
   ELASTICSEARCH_USERNAME=your-username
   ELASTICSEARCH_PASSWORD=your-password
   
   # Cache
   REDIS_URL=redis://localhost:6379
   
   # Application
   ENVIRONMENT=development
   LOG_LEVEL=INFO
   MAX_FILE_SIZE_MB=50
   VERIFICATION_THRESHOLD=0.7
   ```

4. **Run locally**:
   ```bash
   python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
   ```

5. **Access API**:
   - API: http://localhost:8080
   - Docs: http://localhost:8080/docs
   - Health: http://localhost:8080/health

## Deployment

### Google Cloud Run (Recommended)

1. **Quick deployment**:
   ```bash
   ./deploy.sh -p YOUR_PROJECT_ID -r us-central1
   ```

2. **Manual deployment**:
   ```bash
   # Build and push image
   gcloud builds submit --tag gcr.io/PROJECT_ID/legal-doc-analyzer
   
   # Deploy to Cloud Run
   gcloud run deploy legal-doc-analyzer \
     --image gcr.io/PROJECT_ID/legal-doc-analyzer \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2 \
     --max-instances 100
   ```

### Docker

```bash
# Build image
docker build -t legal-doc-analyzer .

# Run container
docker run -p 8080:8080 \
  --env-file .env \
  legal-doc-analyzer
```

## API Endpoints

### Document Processing

- `POST /v1/docs/upload` - Upload legal document
- `GET /v1/docs/{document_id}/status` - Get processing status
- `GET /v1/docs/{document_id}/clauses` - Get extracted clauses
- `POST /v1/docs/{document_id}/query` - Query document

### Human Review

- `GET /v1/review/queue` - Get review queue
- `POST /v1/review/{task_id}/feedback` - Submit review feedback
- `GET /v1/review/analytics` - Get review analytics

### System

- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health check

## Service Architecture

### Document Ingestion Service

Handles document upload and processing:

```python
from src.services.doc_ingestion import document_ingestion_service

# Process document
result = await document_ingestion_service.process_document(
    file_content=file_bytes,
    filename="contract.pdf",
    content_type="application/pdf",
    user_id="user123"
)
```

**Features**:
- Google Document AI OCR
- Deterministic fact extraction (amounts, dates, parties)
- Cloud Storage integration with KMS encryption
- Background clause identification

### Retrieval Service

Hybrid poly-vector retrieval system:

```python
from src.services.retrieval import retrieval_service

# Retrieve relevant clauses
clauses = await retrieval_service.retrieve_relevant_clauses(
    query="What are the payment terms?",
    document_id="doc123",
    top_k=5
)
```

**Features**:
- **Sparse Retrieval**: BM25 with Elasticsearch
- **Dense Retrieval**: Vertex AI embeddings with Matching Engine
- **HyDE**: Hypothetical Document Embeddings for query expansion
- **Cross-encoder**: Reranking with sentence transformers
- **Label-based**: Clause type filtering

### Generation Service

RAG-based clause card generation:

```python
from src.services.generation import generation_service

# Generate clause card
clause_card = await generation_service.generate_clause_card(
    clause_text="Payment terms...",
    document_context="Full document text...",
    document_id="doc123"
)
```

**Features**:
- Vertex AI Gemini integration
- Constrained JSON output using Pydantic schemas
- Risk flag identification
- Recommendation generation
- Fallback handling

### Verification Service

NLI-based verification to prevent hallucinations:

```python
from src.services.verification import verification_service

# Verify clause card
verified_card = await verification_service.verify_clause_card(
    clause_card, source_text
)
```

**Features**:
- Natural Language Inference (NLI) entailment checking
- Confidence adjustment based on verification scores
- Self-consistency checks
- Contradiction detection

### Human Review Service

Attorney workflow management:

```python
from src.services.human_review import human_review_service

# Submit for review
review_task = await human_review_service.submit_for_review(
    clause_card=clause_card,
    document_id="doc123",
    priority="medium"
)
```

**Features**:
- Intelligent reviewer assignment based on specialization
- Priority-based queue management
- Feedback collection and analytics
- Workload balancing

## Configuration

### Google Cloud Services Setup

1. **Document AI**:
   ```bash
   # Create a Document AI processor
   gcloud ai document-ai processors create \
     --display-name="Legal Document Processor" \
     --type="FORM_PARSER_PROCESSOR" \
     --location=us
   ```

2. **Vertex AI**:
   ```bash
   # Enable Vertex AI API
   gcloud services enable aiplatform.googleapis.com
   
   # Create matching engine index
   gcloud ai indexes create \
     --display-name="Legal Clauses Index" \
     --description="Vector index for legal clause embeddings"
   ```

3. **Cloud Storage**:
   ```bash
   # Create storage bucket
   gsutil mb gs://your-legal-docs-bucket
   
   # Set up KMS encryption
   gcloud kms keyrings create legal-docs-keyring --location=global
   gcloud kms keys create legal-docs-key \
     --keyring=legal-docs-keyring \
     --location=global \
     --purpose=encryption
   ```

### Elasticsearch Setup

```bash
# Using Elastic Cloud
curl -X PUT "your-elasticsearch-url/legal_clauses" \
  -H "Content-Type: application/json" \
  -d '{
    "mappings": {
      "properties": {
        "text": {"type": "text", "analyzer": "standard"},
        "embedding": {"type": "dense_vector", "dims": 768},
        "clause_type": {"type": "keyword"},
        "document_id": {"type": "keyword"}
      }
    }
  }'
```

## Security

### Authentication

Uses Firebase Authentication with JWT tokens:

```python
# Protected endpoint example
@app.post("/v1/docs/upload")
async def upload_document(
    file: UploadFile,
    user_id: str = Depends(verify_token)
):
    # user_id extracted from verified JWT token
```

### Authorization

Role-based access control:
- **Users**: Upload documents, view own documents
- **Reviewers**: Access review queue, submit feedback
- **Admins**: Analytics, system management

### Data Protection

- **Encryption**: KMS encryption for data at rest
- **Network**: TLS 1.3 for data in transit
- **Access**: IAM policies and service accounts
- **Audit**: Cloud Logging for all operations

## Monitoring and Observability

### Health Checks

```bash
# Basic health check
curl https://your-service-url/health

# Detailed health check
curl https://your-service-url/health/detailed
```

### Logging

Structured logging with Google Cloud Logging:

```python
import logging
logger = logging.getLogger(__name__)

# Logs automatically sent to Cloud Logging
logger.info("Document processed", extra={
    "document_id": doc_id,
    "user_id": user_id,
    "processing_time": duration
})
```

### Metrics

Key metrics tracked:
- Request latency and throughput
- Document processing success rate
- Verification accuracy
- Review queue metrics
- Error rates by service

## Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_generation.py -v
```

### Integration Tests

```bash
# Test with real services (requires credentials)
pytest tests/integration/ -v

# Test API endpoints
pytest tests/test_api.py -v
```

### Load Testing

```bash
# Using locust for load testing
pip install locust
locust -f tests/load_test.py --host=https://your-service-url
```

## Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

1. **Authentication errors**:
   ```bash
   # Check service account
   gcloud iam service-accounts list
   
   # Verify permissions
   gcloud projects get-iam-policy YOUR_PROJECT_ID
   ```

2. **Memory errors**:
   ```bash
   # Increase Cloud Run memory
   gcloud run services update legal-doc-analyzer \
     --memory 8Gi --region us-central1
   ```

3. **Import errors**:
   ```bash
   # Check dependencies
   pip check
   
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

### Logs

```bash
# View Cloud Run logs
gcloud logs read --service=legal-doc-analyzer --limit=100

# Filter by severity
gcloud logs read --service=legal-doc-analyzer \
  --filter="severity>=ERROR" --limit=50

# Real-time logs
gcloud logs tail --service=legal-doc-analyzer
```

## Performance Optimization

### Caching

Redis-based caching for:
- Document metadata
- Retrieval results
- Generated clause cards

### Async Processing

Background tasks for:
- Document processing
- Clause extraction
- Verification
- Analytics updates

### Resource Limits

Recommended Cloud Run configuration:
- **CPU**: 2 vCPUs
- **Memory**: 4Gi
- **Concurrency**: 80 requests per instance
- **Timeout**: 300 seconds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a pull request

## License

[Your License Here]

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [docs-url]
- Email: support@yourcompany.com