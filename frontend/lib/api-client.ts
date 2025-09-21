/**
 * Legal Document Analyzer MVP - API Client
 * Frontend integration layer for connecting to FastAPI backend
 * NO AUTHENTICATION REQUIRED - Simplified for prototype
 */

export interface DocumentMetadata {
  id: string;
  filename: string;
  content_type?: string;
  content_length?: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  uploaded_at: string;
  processed_at?: string;
  total_pages?: number;
  total_clauses?: number;
}

export interface ExtractedClause {
  clause_id: string;
  doc_id: string;
  text: string;
  clause_type: string;
  confidence_score: number;
  page_number?: number;
  bounding_box?: Record<string, number>;
  created_at: string;
  keywords?: string[];
}

export interface QueryRequest {
  query: string;
}

export interface ExtractedClause {
  id: string;
  text: string;
  type: string;
  confidence: number;
  page?: number;
  position?: {
    start: number;
    end: number;
  };
}

export interface ClauseExtractionResult {
  document_id: string;
  clauses: ExtractedClause[];
  total_clauses: number;
  clause_types: string[];
  statistics: Record<string, number>;
  pagination: {
    limit: number;
    offset: number;
    total: number;
    has_more: boolean;
  };
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  status: string;
  message: string;
  processing_time: number;
}

export interface QueryResponse {
  answer: string;
  confidence: number;
  sources: string[];
  metadata?: Record<string, any>;
}

export interface HealthCheckResponse {
  status: string;
  version: string;
  timestamp: string;
  environment: string;
}

export interface SearchCapabilitiesResponse {
  hybrid_search_enabled: boolean;
  hyde_enabled: boolean;
  cross_encoder_enabled: boolean;
  search_weights: Record<string, number>;
  models: Record<string, string>;
}

class APIError extends Error {
  constructor(public status: number, message: string, public details?: any) {
    super(message);
    this.name = 'APIError';
  }
}

class LegalDocAnalyzerAPI {
  private baseURL: string;
  private defaultHeaders: Record<string, string>;

  constructor(baseURL: string = 'http://localhost:8080') {
    this.baseURL = baseURL;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const response = await fetch(url, {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    });

    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      let errorDetails;

      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorMessage;
        errorDetails = errorData;
      } catch (e) {
        // If we can't parse the error response, use the status text
      }

      throw new APIError(response.status, errorMessage, errorDetails);
    }

    return await response.json();
  }

  // Health and status endpoints
  async healthCheck(): Promise<HealthCheckResponse> {
    return this.request<HealthCheckResponse>('/health');
  }

  async getSearchCapabilities(): Promise<SearchCapabilitiesResponse> {
    return this.request<SearchCapabilitiesResponse>('/v1/search/test');
  }

  // Document management endpoints
  async uploadDocument(file: File, onProgress?: (progress: number) => void): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseURL}/api/documents/upload`, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header for FormData - browser will set it with boundary
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        response.status, 
        errorData.error || `Upload failed: ${response.statusText}`,
        errorData
      );
    }

    return await response.json();
  }

  async getDocumentStatus(docId: string): Promise<DocumentMetadata> {
    return this.request<DocumentMetadata>(`/api/documents/${docId}/status`);
  }

  async getDocumentClauses(
    docId: string,
    clauseType?: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<ExtractedClause[]> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
    });

    if (clauseType) {
      params.append('clause_type', clauseType);
    }

    const response = await this.request<ClauseExtractionResult>(
      `/api/documents/${docId}/clauses?${params.toString()}`
    );
    
    return response.clauses;
  }

  // Query endpoint
  async queryDocument(docId: string, query: QueryRequest): Promise<QueryResponse> {
    return this.request<QueryResponse>(`/api/documents/${docId}/query`, {
      method: 'POST',
      body: JSON.stringify(query),
    });
  }

  // Utility methods
  async waitForDocumentProcessing(
    docId: string,
    maxWaitTime: number = 300000, // 5 minutes
    pollInterval: number = 2000    // 2 seconds
  ): Promise<DocumentMetadata> {
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitTime) {
      const status = await this.getDocumentStatus(docId);
      
      if (status.status === 'completed') {
        return status;
      }
      
      if (status.status === 'failed') {
        throw new APIError(
          500, 
          `Document processing failed: Unknown error`
        );
      }

      // Wait before polling again
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new APIError(408, 'Document processing timeout');
  }

  // Batch operations
  async batchQueryDocument(
    docId: string, 
    queries: string[],
    options: Partial<QueryRequest> = {}
  ): Promise<QueryResponse[]> {
    const results = await Promise.allSettled(
      queries.map(question => 
        this.queryDocument(docId, { query: question, ...options })
      )
    );    return results.map((result, index) => {
      if (result.status === 'fulfilled') {
        return result.value;
      } else {
        throw new APIError(
          500,
          `Query ${index + 1} failed: ${result.reason.message}`,
          result.reason
        );
      }
    });
  }
}

// Export singleton instance
export const apiClient = new LegalDocAnalyzerAPI(
  process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080'
);

// Export class for custom instances
export { LegalDocAnalyzerAPI, APIError };

// Helper functions
export const formatFileSize = (bytes: number): string => {
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 Bytes';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
};

export const formatConfidenceScore = (score: number): string => {
  return `${Math.round(score * 100)}%`;
};

export const getClauseTypeColor = (clauseType: string): string => {
  const colors: Record<string, string> = {
    'general': 'bg-gray-100 text-gray-800',
    'payment': 'bg-green-100 text-green-800',
    'termination': 'bg-red-100 text-red-800',
    'liability': 'bg-yellow-100 text-yellow-800',
    'intellectual_property': 'bg-blue-100 text-blue-800',
    'confidentiality': 'bg-purple-100 text-purple-800',
    'dispute_resolution': 'bg-orange-100 text-orange-800',
    'force_majeure': 'bg-pink-100 text-pink-800',
    'governing_law': 'bg-indigo-100 text-indigo-800',
    'amendment': 'bg-teal-100 text-teal-800',
    'severability': 'bg-cyan-100 text-cyan-800',
    'entire_agreement': 'bg-emerald-100 text-emerald-800',
  };
  return colors[clauseType] || colors['general'];
};