'use client';

import React, { useMemo, useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import DocumentUpload from '@/components/DocumentUpload';
import DocumentContextManager from '@/components/DocumentContextManager';
import { DocumentMetadata } from '@/lib/api-client';
import DocumentCard, { ExtendedDocument } from '@/components/DocumentCard';
import { SparklesIcon, PlusCircleIcon, MagnifyingGlassIcon, ChatBubbleLeftRightIcon, DocumentArrowUpIcon } from '@heroicons/react/24/outline';
import { useAppContext } from '@/contexts/AppContext';

export default function HomePage() {
  const router = useRouter();
  const { documents, addDocument } = useAppContext();
  const [uploadedDocuments, setUploadedDocuments] = useState<ExtendedDocument[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<ExtendedDocument | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showUploadZone, setShowUploadZone] = useState(true);
  
  // Modal states
  const [showContextManager, setShowContextManager] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);

  // Enhanced document selection with auto-scroll
  const handleDocumentSelect = (document: ExtendedDocument) => {
    setSelectedDocument(document);
    // Auto-scroll to document details section
    setTimeout(() => {
      const documentDetailsSection = globalThis.document?.querySelector('[data-document-details]');
      if (documentDetailsSection) {
        documentDetailsSection.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start',
          inline: 'nearest'
        });
      }
    }, 100);
  };

  // Sync documents from context to local state
  useEffect(() => {
    const extendedDocs: ExtendedDocument[] = documents.map(doc => ({
      ...doc,
      riskScore: Math.floor(Math.random() * 100),
      moreScore: Math.floor(Math.random() * 100),
      abnormalFindings: [
        { type: 'Missing Clause', description: 'No confidentiality agreement found', severity: 'high' as const },
        { type: 'Ambiguous Term', description: 'Indemnification scope unclear', severity: 'medium' as const },
      ]
    }));
    setUploadedDocuments(extendedDocs);
    
    // Auto-select first document if none selected
    if (extendedDocs.length > 0 && !selectedDocument) {
      handleDocumentSelect(extendedDocs[0]);
    }
  }, [documents, selectedDocument]);

  const handleDocumentUpload = (_docId: string, metadata: DocumentMetadata) => {
    // Add to shared context
    addDocument(metadata);
    
    const newDocument: ExtendedDocument = {
      ...metadata,
      riskScore: Math.floor(Math.random() * 100),
      moreScore: Math.floor(Math.random() * 100),
      abnormalFindings: [
        { type: 'Missing Clause', description: 'No confidentiality agreement found', severity: 'high' },
        { type: 'Ambiguous Term', description: 'Indemnification scope unclear', severity: 'medium' },
      ],
      aiSuggestions: ['Consider adding force majeure clause', 'Review termination conditions']
    };
    
    // Local state update is handled by useEffect when context changes
    // Auto-select the first document if none is selected
    if (!selectedDocument) {
      setSelectedDocument(newDocument);
    }
  };

  const filtered = useMemo(() => {
    if (!searchQuery) return uploadedDocuments;
    return uploadedDocuments.filter(d => d.filename.toLowerCase().includes(searchQuery.toLowerCase()));
  }, [searchQuery, uploadedDocuments]);

  // Convert ExtendedDocument to the format expected by DocumentContextManager
  const contextDocuments = uploadedDocuments.map(doc => ({
    id: doc.id,
    name: doc.filename,
    type: doc.content_type || 'document',
    summary: `Risk Score: ${doc.riskScore}%`,
    riskScore: doc.riskScore,
    status: doc.status,
    uploadedAt: doc.uploaded_at
  }));

  const handleContextButtonClick = () => {
    setShowContextManager(true);
  };

  const handleAskButtonClick = () => {
    router.push('/ask-ai');
  };

  const handleUploadButtonClick = () => {
    setShowUploadModal(true);
  };

  return (
    <div className="min-h-screen flex flex-col gap-6 p-6" style={{background: 'radial-gradient(circle at 20% 20%, rgba(88,101,242,0.15), transparent 60%), radial-gradient(circle at 80% 60%, rgba(213,72,165,0.15), transparent 55%)'}}>
      {/* Top Row: Context/Ask + Drop Zone */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="gemini-card p-5 flex flex-col justify-between order-1 lg:order-none">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold text-gradient">Legal Doc Analyzer</h2>
                <p className="text-xs secondary-text mt-1">AI-Powered • Gemini Theme</p>
              </div>
              <SparklesIcon className="h-6 w-6 text-[var(--electric-aqua)]" />
            </div>
            <p className="text-xs secondary-text mb-4 leading-relaxed">Build rich context sets and ask domain aware legal questions. Add documents to enrich retrieval.</p>
          </div>
          <div className="space-y-3 mt-auto">
            <button 
              onClick={handleUploadButtonClick}
              className="w-full btn-primary text-xs py-2.5 flex items-center justify-center gap-2 bg-gradient-to-r from-[var(--electric-aqua)] to-[var(--vibrant-magenta)] hover:opacity-90 transition-opacity"
            >
              <DocumentArrowUpIcon className="h-4 w-4" /> Upload Documents
            </button>
            <div className="grid grid-cols-2 gap-3">
              <button 
                onClick={handleContextButtonClick}
                className="btn-primary text-xs py-2 flex items-center justify-center gap-1"
              >
                <PlusCircleIcon className="h-4 w-4" /> Context
              </button>
              <button 
                onClick={handleAskButtonClick}
                className="btn-ghost text-xs py-2 flex items-center justify-center gap-1"
              >
                <ChatBubbleLeftRightIcon className="h-4 w-4" /> Ask
              </button>
            </div>
          </div>
        </div>
        <div className="gemini-card p-8 relative overflow-hidden lg:col-span-2">
          <div className="absolute inset-0 pointer-events-none opacity-30 bg-[radial-gradient(circle_at_70%_30%,rgba(36,216,218,0.25),transparent_60%)]" />
          <div className="relative">
            <h1 className="text-2xl md:text-3xl font-bold mb-2 text-gradient">Drop your Docs</h1>
            <p className="secondary-text mb-6 text-sm max-w-2xl">Upload legal documents for instant AI clause extraction, risk scoring and smart recommendations.</p>
            <DocumentUpload variant='dark' onUploadSuccess={handleDocumentUpload} compact multiple />
          </div>
        </div>
      </div>

      {/* Documents Section (Full Width Below) */}
      <div className="gemini-card p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-4">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold tracking-wide uppercase text-[var(--nebula-grey)]">My Documents</h3>
            {uploadedDocuments.length > 0 && (
              <span className="text-[10px] px-2 py-1 rounded-full bg-[rgba(240,242,252,0.08)] border border-[rgba(138,145,180,0.25)]">{uploadedDocuments.length}</span>
            )}
          </div>
          <div className="relative w-full md:w-80">
            <MagnifyingGlassIcon className="h-4 w-4 absolute left-3 top-1/2 -translate-y-1/2 text-[var(--nebula-grey)]" />
            <input
              type="text"
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 gemini-input py-2 text-sm"
            />
          </div>
        </div>

        {/* Main Document Layout */}
        {uploadedDocuments.length > 0 && selectedDocument ? (
          <div className="space-y-6" data-document-details>
            {/* Main Document Preview Layout */}
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Left: Document Preview */}
              <div className="lg:col-span-1">
                <div className="gemini-card h-80 p-4 overflow-hidden">
                  <div className="h-full flex flex-col">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm font-semibold text-[var(--starlight-white)]">Document Preview</h4>
                      <span className="text-xs text-[var(--nebula-grey)]">Page 1 of {selectedDocument.total_pages || 'N/A'}</span>
                    </div>
                    
                    {/* Document Content Preview */}
                    <div className="flex-1 bg-[rgba(240,242,252,0.02)] rounded-lg border border-[rgba(138,145,180,0.2)] p-3 overflow-y-auto custom-scrollbar">
                      <div className="text-xs leading-relaxed text-[var(--starlight-white)] space-y-2">
                        <p className="font-semibold text-[var(--electric-aqua)]">{selectedDocument.filename}</p>
                        <div className="text-[10px] text-[var(--nebula-grey)] space-y-1">
                          <p>This document contains {selectedDocument.total_clauses || 0} clauses and has been analyzed for legal compliance.</p>
                          {selectedDocument.content_length && (
                            <p>Document size: {(selectedDocument.content_length / 1024).toFixed(1)}KB</p>
                          )}
                          <p>Processing status: {selectedDocument.status}</p>
                          {selectedDocument.abnormalFindings && selectedDocument.abnormalFindings.length > 0 && (
                            <p className="text-amber-400">⚠️ {selectedDocument.abnormalFindings.length} potential issues detected</p>
                          )}
                        </div>
                        
                        {/* Sample document content */}
                        <div className="mt-3 pt-3 border-t border-[rgba(138,145,180,0.1)]">
                          <p className="text-[10px] text-[var(--nebula-grey)] mb-2">Document Extract:</p>
                          <div className="text-[9px] text-[var(--starlight-white)] leading-relaxed space-y-1">
                            <p>"This Agreement shall commence on the Effective Date and shall continue for a period of..."</p>
                            <p>"The parties agree to the following terms and conditions as set forth herein..."</p>
                            <p>"Confidential Information shall mean any and all non-public information..."</p>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Preview Actions */}
                    <div className="mt-3 flex gap-2">
                      <button className="flex-1 bg-[rgba(36,216,218,0.1)] hover:bg-[rgba(36,216,218,0.2)] border border-[rgba(36,216,218,0.3)] rounded-lg py-2 px-3 text-xs text-[var(--electric-aqua)] transition-colors">
                        Full View
                      </button>
                      <button className="flex-1 bg-[rgba(240,242,252,0.05)] hover:bg-[rgba(240,242,252,0.1)] border border-[rgba(138,145,180,0.2)] rounded-lg py-2 px-3 text-xs text-[var(--starlight-white)] transition-colors">
                        Download
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right: Document Details and Actions */}
              <div className="lg:col-span-2 space-y-4">
                {/* Action Buttons Row */}
                <div className="flex gap-3">
                  <button className="flex-1 gemini-card py-3 px-4 text-center hover:scale-105 transition-transform">
                    <div className="text-lg font-bold text-[var(--electric-aqua)]">Risk %</div>
                    <div className="text-2xl font-bold text-[var(--starlight-white)] mt-1">{selectedDocument.riskScore}</div>
                  </button>
                  <button className="flex-1 gemini-card py-3 px-4 text-center hover:scale-105 transition-transform">
                    <div className="text-lg font-bold text-[var(--vibrant-magenta)]">More Score</div>
                    <div className="text-2xl font-bold text-[var(--starlight-white)] mt-1">{selectedDocument.moreScore}</div>
                  </button>
                  <button className="flex-1 gemini-card py-3 px-4 text-center hover:scale-105 transition-transform bg-gradient-to-r from-[var(--electric-aqua)] to-[var(--vibrant-magenta)] text-black font-semibold">
                    <div className="text-lg font-bold">AI Fix</div>
                    <div className="text-sm mt-1">Generate</div>
                  </button>
                </div>

                {/* Abnormal Points Section */}
                <div className="gemini-card p-4">
                  <h4 className="text-sm font-semibold text-[var(--starlight-white)] mb-3">Abnormal points</h4>
                  <div className="space-y-2 max-h-32 overflow-y-auto custom-scrollbar">
                    {selectedDocument.abnormalFindings?.map((finding, index) => (
                      <div key={index} className="flex items-start gap-2 p-2 rounded-lg bg-[rgba(240,242,252,0.05)] border border-[rgba(138,145,180,0.2)]">
                        <div className={`w-2 h-2 rounded-full mt-1 flex-shrink-0 ${
                          finding.severity === 'high' ? 'bg-red-400' : 
                          finding.severity === 'medium' ? 'bg-yellow-400' : 'bg-blue-400'
                        }`}></div>
                        <div>
                          <p className="text-xs font-medium text-[var(--starlight-white)]">{finding.type}</p>
                          <p className="text-xs secondary-text">{finding.description}</p>
                        </div>
                      </div>
                    )) || (
                      <p className="text-xs secondary-text">No abnormal findings detected</p>
                    )}
                  </div>
                </div>

                {/* Ask a Question Section */}
                <div className="gemini-card p-4">
                  <h4 className="text-sm font-semibold text-[var(--starlight-white)] mb-3">Ask a Question</h4>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      placeholder="Type your question about this document..."
                      className="flex-1 gemini-input py-2 px-3 text-sm"
                    />
                    <button 
                      onClick={() => router.push('/ask-ai')}
                      className="btn-primary px-4 py-2 text-sm hover:scale-105 transition-transform"
                    >
                      Ask
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Additional Document Cards */}
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filtered.filter(doc => doc.id !== selectedDocument.id).map(doc => (
                <DocumentCard
                  key={doc.id}
                  doc={doc as ExtendedDocument}
                  onSelect={handleDocumentSelect}
                  onAsk={(d) => console.log('Ask about', d.id)}
                />
              ))}
            </div>
          </div>
        ) : (
          /* No Documents or No Selection State */
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {filtered.length === 0 && (
              <p className="text-xs secondary-text col-span-full text-center py-10">No documents yet. Upload one ↑</p>
            )}
            {filtered.map(doc => (
              <DocumentCard
                key={doc.id}
                doc={doc as ExtendedDocument}
                onSelect={handleDocumentSelect}
                onAsk={(d) => console.log('Ask about', d.id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Context Manager Modal */}
      {showContextManager && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="w-full max-w-4xl max-h-[90vh] overflow-hidden">
            <DocumentContextManager
              documents={contextDocuments}
              selectedDocs={selectedDocuments}
              onSelectionChange={setSelectedDocuments}
              onClose={() => setShowContextManager(false)}
            />
          </div>
        </div>
      )}

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="w-full max-w-2xl">
            <div className="gemini-card p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-[var(--starlight-white)]">Upload Documents</h3>
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="text-[var(--nebula-grey)] hover:text-[var(--starlight-white)] transition-colors"
                >
                  ✕
                </button>
              </div>
              <DocumentUpload 
                variant='dark' 
                onUploadSuccess={(docId, metadata) => {
                  handleDocumentUpload(docId, metadata);
                  // Keep modal open for multiple uploads
                }} 
                multiple 
              />
              <div className="mt-4 text-center">
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="btn-ghost text-sm px-6 py-2"
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
