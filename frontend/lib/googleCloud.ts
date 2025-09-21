/**
 * Google Cloud Service - Simplified for MVP
 * This file provides type definitions and utilities for our Legal Document Analyzer MVP.
 * Instead of direct Google Cloud API calls, we use our backend API.
 */

// Type definitions for document analysis
export interface DocumentAnalysisResult {
  text: string;
  entities: Array<{
    type: string;
    text: string;
    confidence: number;
  }>;
  keyValuePairs: Array<{
    key: string;
    value: string;
  }>;
  clauses: Array<{
    type: string;
    count: number;
    examples: string[];
  }>;
  riskFactors: Array<{
    keyword: string;
    severity: number;
    context: string;
  }>;
  summary: string;
  confidence: number;
  processingTime: number;
}

export interface ProcessDocumentResponse {
  document?: {
    text?: string;
    entities?: any[];
    pages?: any[];
  };
}

// Simplified service class that uses our backend API
export class GoogleCloudService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080';
  }

  /**
   * Upload and process a document through our backend API
   * @param file - The file to upload
   * @returns Promise with upload response
   */
  async uploadDocument(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/api/v1/documents/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Check document processing status
   * @param documentId - The document ID
   * @returns Promise with document status
   */
  async getDocumentStatus(documentId: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/v1/documents/${documentId}/status`);
    
    if (!response.ok) {
      throw new Error(`Failed to get document status: ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Query a processed document
   * @param documentId - The document ID
   * @param query - The query string
   * @returns Promise with query response
   */
  async queryDocument(documentId: string, query: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/v1/documents/${documentId}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      throw new Error(`Query failed: ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Get extracted clauses from a document
   * @param documentId - The document ID
   * @param clauseType - Optional clause type filter
   * @returns Promise with clauses data
   */
  async getDocumentClauses(documentId: string, clauseType?: string): Promise<any> {
    const params = new URLSearchParams();
    if (clauseType) {
      params.append('clause_type', clauseType);
    }

    const url = `${this.baseUrl}/api/v1/documents/${documentId}/clauses?${params.toString()}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to get clauses: ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Get system health status
   * @returns Promise with health status
   */
  async getHealthStatus(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/health`);
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return await response.json();
  }
}

// Export a singleton instance
export const googleCloudService = new GoogleCloudService();

// Utility functions for document processing
export const DocumentUtils = {
  /**
   * Check if a file type is supported
   * @param file - The file to check
   * @returns boolean indicating if file is supported
   */
  isSupportedFileType(file: File): boolean {
    const supportedTypes = [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/plain'
    ];
    
    const supportedExtensions = ['.pdf', '.doc', '.docx', '.txt'];
    
    return supportedTypes.includes(file.type) || 
           supportedExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
  },

  /**
   * Format file size for display
   * @param bytes - Size in bytes
   * @returns Formatted string
   */
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  /**
   * Extract file extension
   * @param filename - The filename
   * @returns File extension
   */
  getFileExtension(filename: string): string {
    return filename.substring(filename.lastIndexOf('.') + 1).toLowerCase();
  },

  /**
   * Validate file size
   * @param file - The file to validate
   * @param maxSizeMB - Maximum size in MB
   * @returns boolean indicating if file size is valid
   */
  isValidFileSize(file: File, maxSizeMB: number = 50): boolean {
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    return file.size <= maxSizeBytes;
  }
};

// Export default service instance
export default googleCloudService;