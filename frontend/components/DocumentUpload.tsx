'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  CloudArrowUpIcon, 
  DocumentIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { apiClient, formatFileSize, DocumentUploadResponse, DocumentMetadata } from '@/lib/api-client';

interface DocumentUploadProps {
  onUploadSuccess?: (docId: string, metadata: DocumentMetadata) => void;
  onUploadError?: (error: string) => void;
  maxFileSize?: number; // in MB
  acceptedFileTypes?: string[];
  variant?: 'light' | 'dark';
  compact?: boolean;
  multiple?: boolean; // New prop for multiple uploads
  autoResetDelay?: number; // Auto reset delay in milliseconds (default: 3000)
}

interface UploadState {
  status: 'idle' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  uploadResponse?: DocumentUploadResponse;
  documentMetadata?: DocumentMetadata;
  error?: string;
  fileName?: string;
}

interface MultipleUploadState {
  uploads: Map<string, UploadState>;
  overall: {
    totalFiles: number;
    completedFiles: number;
    failedFiles: number;
    isUploading: boolean;
  };
}

export default function DocumentUpload({
  onUploadSuccess,
  onUploadError,
  maxFileSize = 50,
  acceptedFileTypes = ['.pdf', '.doc', '.docx'],
  variant = 'light',
  compact = false,
  multiple = false,
  autoResetDelay = 3000
}: DocumentUploadProps) {
  const [multipleUploadState, setMultipleUploadState] = useState<MultipleUploadState>({
    uploads: new Map(),
    overall: {
      totalFiles: 0,
      completedFiles: 0,
      failedFiles: 0,
      isUploading: false
    }
  });

  // Keep legacy single upload state for backward compatibility
  const [uploadState, setUploadState] = useState<UploadState>({
    status: 'idle',
    progress: 0
  });

  const [resetCountdown, setResetCountdown] = useState<number | null>(null);

  // Auto-reset functionality
  useEffect(() => {
    let resetTimer: NodeJS.Timeout;
    let countdownTimer: NodeJS.Timeout;
    
    if (multiple) {
      // For multiple uploads, reset when all uploads are complete
      const { totalFiles, completedFiles, failedFiles, isUploading } = multipleUploadState.overall;
      if (!isUploading && totalFiles > 0 && (completedFiles + failedFiles) === totalFiles) {
        // Start countdown
        let countdown = Math.ceil(autoResetDelay / 1000);
        setResetCountdown(countdown);
        
        countdownTimer = setInterval(() => {
          countdown -= 1;
          setResetCountdown(countdown);
          if (countdown <= 0) {
            clearInterval(countdownTimer);
          }
        }, 1000);

        resetTimer = setTimeout(() => {
          setResetCountdown(null);
          setMultipleUploadState({
            uploads: new Map(),
            overall: {
              totalFiles: 0,
              completedFiles: 0,
              failedFiles: 0,
              isUploading: false
            }
          });
        }, autoResetDelay);
      }
    } else {
      // For single upload, reset when completed
      if (uploadState.status === 'completed') {
        // Start countdown
        let countdown = Math.ceil(autoResetDelay / 1000);
        setResetCountdown(countdown);
        
        countdownTimer = setInterval(() => {
          countdown -= 1;
          setResetCountdown(countdown);
          if (countdown <= 0) {
            clearInterval(countdownTimer);
          }
        }, 1000);

        resetTimer = setTimeout(() => {
          setResetCountdown(null);
          setUploadState({ status: 'idle', progress: 0 });
        }, autoResetDelay);
      }
    }

    return () => {
      if (resetTimer) {
        clearTimeout(resetTimer);
      }
      if (countdownTimer) {
        clearInterval(countdownTimer);
      }
    };
  }, [uploadState.status, multipleUploadState.overall, multiple, autoResetDelay]);

  const uploadSingleFile = async (file: File, fileId: string) => {
    // Validate file size
    if (file.size > maxFileSize * 1024 * 1024) {
      const error = `File ${file.name} exceeds ${maxFileSize}MB limit`;
      setMultipleUploadState(prev => ({
        ...prev,
        uploads: new Map(prev.uploads).set(fileId, {
          status: 'error',
          progress: 0,
          error,
          fileName: file.name
        }),
        overall: {
          ...prev.overall,
          failedFiles: prev.overall.failedFiles + 1
        }
      }));
      onUploadError?.(error);
      return;
    }

    try {
      // Start upload
      setMultipleUploadState(prev => ({
        ...prev,
        uploads: new Map(prev.uploads).set(fileId, {
          status: 'uploading',
          progress: 0,
          fileName: file.name
        })
      }));

      const uploadResponse = await apiClient.uploadDocument(file, (progress) => {
        setMultipleUploadState(prev => ({
          ...prev,
          uploads: new Map(prev.uploads).set(fileId, {
            ...prev.uploads.get(fileId)!,
            progress
          })
        }));
      });

      setMultipleUploadState(prev => ({
        ...prev,
        uploads: new Map(prev.uploads).set(fileId, {
          status: 'processing',
          progress: 100,
          uploadResponse,
          fileName: file.name
        })
      }));

      // Wait for processing to complete
      const metadata = await apiClient.waitForDocumentProcessing(
        uploadResponse.document_id,
        300000, // 5 minutes
        2000    // 2 seconds
      );

      setMultipleUploadState(prev => ({
        ...prev,
        uploads: new Map(prev.uploads).set(fileId, {
          status: 'completed',
          progress: 100,
          uploadResponse,
          documentMetadata: metadata,
          fileName: file.name
        }),
        overall: {
          ...prev.overall,
          completedFiles: prev.overall.completedFiles + 1
        }
      }));

      onUploadSuccess?.(uploadResponse.document_id, metadata);

    } catch (error: any) {
      const errorMessage = error.message || `Upload failed for ${file.name}`;
      setMultipleUploadState(prev => ({
        ...prev,
        uploads: new Map(prev.uploads).set(fileId, {
          status: 'error',
          progress: 0,
          error: errorMessage,
          fileName: file.name
        }),
        overall: {
          ...prev.overall,
          failedFiles: prev.overall.failedFiles + 1
        }
      }));
      onUploadError?.(errorMessage);
    }
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (!multiple) {
      // Legacy single file upload
      const file = acceptedFiles[0];
      if (!file) return;

      // Validate file size
      if (file.size > maxFileSize * 1024 * 1024) {
        const error = `File size exceeds ${maxFileSize}MB limit`;
        setUploadState({ status: 'error', progress: 0, error });
        onUploadError?.(error);
        return;
      }

      try {
        // Start upload
        setUploadState({ status: 'uploading', progress: 0 });

        const uploadResponse = await apiClient.uploadDocument(file, (progress) => {
          setUploadState(prev => ({ ...prev, progress }));
        });

        setUploadState({
          status: 'processing',
          progress: 100,
          uploadResponse
        });

        // Wait for processing to complete
        const metadata = await apiClient.waitForDocumentProcessing(
          uploadResponse.document_id,
          300000, // 5 minutes
          2000    // 2 seconds
        );

        setUploadState({
          status: 'completed',
          progress: 100,
          uploadResponse,
          documentMetadata: metadata
        });

        onUploadSuccess?.(uploadResponse.document_id, metadata);

      } catch (error: any) {
        const errorMessage = error.message || 'Upload failed';
        setUploadState({ status: 'error', progress: 0, error: errorMessage });
        onUploadError?.(errorMessage);
      }
    } else {
      // Multiple file upload
      if (acceptedFiles.length === 0) return;

      setMultipleUploadState(prev => ({
        uploads: new Map(),
        overall: {
          totalFiles: acceptedFiles.length,
          completedFiles: 0,
          failedFiles: 0,
          isUploading: true
        }
      }));

      // Upload all files concurrently
      const uploadPromises = acceptedFiles.map((file, index) => {
        const fileId = `${file.name}-${Date.now()}-${index}`;
        return uploadSingleFile(file, fileId);
      });

      await Promise.all(uploadPromises);

      setMultipleUploadState(prev => ({
        ...prev,
        overall: {
          ...prev.overall,
          isUploading: false
        }
      }));
    }
  }, [maxFileSize, onUploadSuccess, onUploadError, multiple]);

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    maxFiles: multiple ? undefined : 1,
    maxSize: maxFileSize * 1024 * 1024,
    multiple: multiple
  });

  const resetUpload = () => {
    setUploadState({ status: 'idle', progress: 0 });
    setResetCountdown(null);
    setMultipleUploadState({
      uploads: new Map(),
      overall: {
        totalFiles: 0,
        completedFiles: 0,
        failedFiles: 0,
        isUploading: false
      }
    });
  };

  const getStatusIcon = () => {
    switch (uploadState.status) {
      case 'uploading':
      case 'processing':
        return (
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        );
      case 'completed':
        return <CheckCircleIcon className="h-8 w-8 text-green-600" />;
      case 'error':
        return <ExclamationTriangleIcon className="h-8 w-8 text-red-600" />;
      default:
        return <CloudArrowUpIcon className="h-8 w-8 text-gray-400" />;
    }
  };

  const getStatusMessage = () => {
    switch (uploadState.status) {
      case 'uploading':
        return `Uploading... ${uploadState.progress}%`;
      case 'processing':
        return 'Processing document with AI...';
      case 'completed':
        return 'Document processed successfully! Ready for more uploads.';
      case 'error':
        return uploadState.error || 'Upload failed';
      default:
        return isDragActive 
          ? 'Drop the document here...' 
          : 'Drag & drop a legal document here, or click to select';
    }
  };

  if (uploadState.status === 'completed' && uploadState.documentMetadata) {
    return (
      <div className={`w-full ${compact ? '' : 'max-w-2xl mx-auto'}`}>
        <div className={`${variant === 'dark' ? 'gemini-card bg-green-500/5 border-green-400/30' : 'bg-green-50 border border-green-200'} rounded-lg p-6`}>          
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <CheckCircleIcon className={`h-6 w-6 mr-2 ${variant === 'dark' ? 'text-green-400' : 'text-green-600'}`} />
              <h3 className={`text-lg font-semibold ${variant === 'dark' ? 'text-green-300' : 'text-green-900'}`}>
                Document Ready
              </h3>
            </div>
            <button
              onClick={resetUpload}
              className={`${variant === 'dark' ? 'text-green-300 hover:text-green-200' : 'text-green-600 hover:text-green-800'}`}
            >
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>

          <div className="space-y-3">
            <div className={`flex items-center text-sm ${variant === 'dark' ? 'text-green-200' : 'text-green-800'}`}>
              <DocumentIcon className={`h-4 w-4 mr-2 ${variant === 'dark' ? 'text-green-300' : ''}`} />
              <span className="font-medium break-all">{uploadState.documentMetadata.filename}</span>
            </div>
            
            <div className={`grid grid-cols-2 gap-4 text-sm ${variant === 'dark' ? 'text-green-300/80' : 'text-green-700'}`}>
              <div>
                <span className="font-medium">Size:</span> {uploadState.documentMetadata.content_length ? formatFileSize(uploadState.documentMetadata.content_length) : 'N/A'}
              </div>
              <div>
                <span className="font-medium">Pages:</span> {uploadState.documentMetadata.total_pages || 'N/A'}
              </div>
              <div>
                <span className="font-medium">Clauses:</span> {uploadState.documentMetadata.total_clauses || 'N/A'}
              </div>
              <div>
                <span className="font-medium">Status:</span> {uploadState.documentMetadata.status}
              </div>
            </div>

            {uploadState.documentMetadata.status === 'completed' && (
              <div>
                <span className={`font-medium text-sm ${variant === 'dark' ? 'text-green-300' : 'text-green-800'}`}>Status:</span>
                <span className={`ml-2 px-2 py-1 text-xs rounded-full ${variant === 'dark' ? 'bg-green-400/10 text-green-300 border border-green-400/30' : 'bg-green-100 text-green-800'}`}>
                  Processing Complete
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  const renderMultipleUploadProgress = () => {
    if (multipleUploadState.uploads.size === 0) return null;

    return (
      <div className="mt-4 space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className={variant === 'dark' ? 'text-starlight-white' : 'text-gray-700'}>
            Upload Progress ({multipleUploadState.overall.completedFiles}/{multipleUploadState.overall.totalFiles})
          </span>
          {multipleUploadState.overall.failedFiles > 0 && (
            <span className="text-red-400">
              {multipleUploadState.overall.failedFiles} failed
            </span>
          )}
        </div>
        
        <div className="max-h-32 overflow-y-auto space-y-1">
          {Array.from(multipleUploadState.uploads.entries()).map(([fileId, upload]) => (
            <div key={fileId} className={`text-xs p-2 rounded ${variant === 'dark' ? 'bg-[rgba(240,242,252,0.05)]' : 'bg-gray-100'}`}>
              <div className="flex items-center justify-between mb-1">
                <span className={`truncate flex-1 ${variant === 'dark' ? 'text-starlight-white' : 'text-gray-700'}`}>
                  {upload.fileName}
                </span>
                <span className={`ml-2 ${
                  upload.status === 'completed' ? 'text-green-500' :
                  upload.status === 'error' ? 'text-red-500' :
                  variant === 'dark' ? 'text-[var(--electric-aqua)]' : 'text-blue-500'
                }`}>
                  {upload.status === 'completed' ? '✓' :
                   upload.status === 'error' ? '✗' :
                   upload.status === 'processing' ? '⚡' : '↗'}
                </span>
              </div>
              
              {upload.status !== 'error' && (
                <div className={`w-full bg-gray-200 rounded-full h-1 ${variant === 'dark' ? 'bg-[rgba(138,145,180,0.3)]' : ''}`}>
                  <div 
                    className={`h-1 rounded-full transition-all duration-300 ${
                      upload.status === 'completed' ? 'bg-green-500' : 
                      variant === 'dark' ? 'bg-gradient-to-r from-[var(--electric-aqua)] to-[var(--vibrant-magenta)]' : 'bg-blue-500'
                    }`}
                    style={{ width: `${upload.progress}%` }}
                  ></div>
                </div>
              )}
              
              {upload.error && (
                <p className="text-red-400 mt-1 text-[10px]">{upload.error}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className={`w-full ${compact ? '' : 'max-w-2xl mx-auto'}`}>
      <div
        {...getRootProps()}
        className={`
          rounded-2xl p-8 text-center cursor-pointer transition-all relative overflow-hidden group
          ${variant === 'dark' 
            ? `border-2 border-dashed ${isDragActive ? 'border-[var(--electric-aqua)] bg-[rgba(36,216,218,0.07)]' : 
                (multiple ? multipleUploadState.overall.isUploading : uploadState.status === 'error') ? 'border-red-400/60 bg-red-500/10' : 'border-[rgba(138,145,180,0.35)] bg-[rgba(240,242,252,0.04)] hover:bg-[rgba(240,242,252,0.07)] backdrop-blur-xl'}`
            : `border-2 border-dashed ${isDragActive ? 'border-blue-400 bg-blue-50' : 
                (multiple ? multipleUploadState.overall.isUploading : uploadState.status === 'error') ? 'border-red-300 bg-red-50' : 'border-gray-300 bg-gray-50 hover:bg-gray-100'}`
          }
          ${((multiple ? multipleUploadState.overall.isUploading : (uploadState.status === 'uploading' || uploadState.status === 'processing'))) ? 'pointer-events-none' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center space-y-4">
          {multiple ? (
            multipleUploadState.overall.isUploading ? (
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            ) : multipleUploadState.uploads.size > 0 ? (
              <CheckCircleIcon className="h-8 w-8 text-green-600" />
            ) : (
              <CloudArrowUpIcon className="h-8 w-8 text-gray-400" />
            )
          ) : (
            getStatusIcon()
          )}
          
          <div className="space-y-2">
            <p className={`text-lg font-medium ${variant === 'dark' ? 'text-starlight-white' : ''} ${
              (multiple ? multipleUploadState.overall.failedFiles > 0 : uploadState.status === 'error') ? (variant === 'dark' ? 'text-red-400' : 'text-red-900') : (variant === 'dark' ? 'text-[var(--starlight-white)]' : 'text-gray-900')
            }`}>
              {multiple ? (
                multipleUploadState.overall.isUploading ? 'Uploading files...' :
                multipleUploadState.uploads.size > 0 ? 
                  `${multipleUploadState.overall.completedFiles} of ${multipleUploadState.overall.totalFiles} uploaded successfully! Ready for more.` :
                  isDragActive ? 'Drop multiple documents here' : 'Drop multiple documents here, or click to browse'
              ) : (
                getStatusMessage()
              )}
            </p>
            
            {uploadState.status === 'idle' && (
              <p className={`text-sm ${variant === 'dark' ? 'text-[var(--nebula-grey)]' : 'text-gray-500'}`}>
                Supports: {acceptedFileTypes.join(', ')} • Max {maxFileSize}MB
              </p>
            )}

            {/* Countdown indicator */}
            {resetCountdown !== null && resetCountdown > 0 && (
              <p className={`text-xs ${variant === 'dark' ? 'text-[var(--electric-aqua)]' : 'text-blue-600'}`}>
                Ready for more uploads... (resetting in {resetCountdown}s)
              </p>
            )}

            {uploadState.status === 'processing' && uploadState.uploadResponse && (
              <div className={`text-sm ${variant === 'dark' ? 'text-[var(--electric-aqua)]' : 'text-blue-600'}`}>
                <p>Document ID: {uploadState.uploadResponse.document_id}</p>
                <p>Using advanced AI for clause extraction and classification...</p>
              </div>
            )}
          </div>

          {/* Progress bar for uploading/processing */}
          {(uploadState.status === 'uploading' || uploadState.status === 'processing') && (
            <div className="w-full max-w-xs">
              <div className={`rounded-full h-2 ${variant === 'dark' ? 'bg-[rgba(138,145,180,0.25)]' : 'bg-gray-200'}`}>                
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${variant === 'dark' ? 'bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500' : 'bg-blue-600'}`}
                  style={{ width: `${uploadState.progress}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Multiple upload progress */}
      {multiple && renderMultipleUploadProgress()}

      {/* File rejection errors */}
      {fileRejections.length > 0 && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <h4 className="text-sm font-medium text-red-900 mb-2">File Upload Errors:</h4>
          <ul className="text-sm text-red-700 space-y-1">
            {fileRejections.map(({ file, errors }, index) => (
              <li key={index}>
                <strong>{file.name}:</strong>
                <ul className="ml-4 list-disc">
                  {errors.map((error, errorIndex) => (
                    <li key={errorIndex}>{error.message}</li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Retry button for errors */}
      {(multiple ? multipleUploadState.overall.failedFiles > 0 : uploadState.status === 'error') && (
        <div className="mt-4 text-center">
          <button
            onClick={resetUpload}
            className={`px-4 py-2 rounded-lg transition-colors ${variant === 'dark' ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-500 hover:to-purple-500' : 'bg-blue-600 text-white hover:bg-blue-700'}`}
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
}