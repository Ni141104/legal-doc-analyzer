'use client';

import React, { useState, useEffect } from 'react';
import { 
  FunnelIcon,
  ArrowDownIcon,
  ArrowUpIcon,
  DocumentTextIcon,
  TagIcon,
  ChartBarIcon,
  EyeIcon,
  ClipboardDocumentIcon
} from '@heroicons/react/24/outline';
import { apiClient, ExtractedClause, formatConfidenceScore, getClauseTypeColor } from '@/lib/api-client';

interface ClausesViewProps {
  docId: string;
  documentName?: string;
  onError?: (error: string) => void;
}

type SortField = 'confidence_score' | 'clause_type' | 'created_at' | 'page_number';
type SortDirection = 'asc' | 'desc';

interface ClauseFilters {
  clauseType: string;
  minConfidence: number;
  searchText: string;
}

const CLAUSE_TYPES = [
  { value: '', label: 'All Types' },
  { value: 'general', label: 'General' },
  { value: 'payment', label: 'Payment' },
  { value: 'termination', label: 'Termination' },
  { value: 'liability', label: 'Liability' },
  { value: 'intellectual_property', label: 'Intellectual Property' },
  { value: 'confidentiality', label: 'Confidentiality' },
  { value: 'dispute_resolution', label: 'Dispute Resolution' },
  { value: 'force_majeure', label: 'Force Majeure' },
  { value: 'governing_law', label: 'Governing Law' },
  { value: 'amendment', label: 'Amendment' },
  { value: 'severability', label: 'Severability' },
  { value: 'entire_agreement', label: 'Entire Agreement' }
];

export default function ClausesView({ docId, documentName, onError }: ClausesViewProps) {
  const [clauses, setClauses] = useState<ExtractedClause[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedClause, setSelectedClause] = useState<ExtractedClause | null>(null);
  
  // Filtering and sorting
  const [filters, setFilters] = useState<ClauseFilters>({
    clauseType: '',
    minConfidence: 0,
    searchText: ''
  });
  const [sortField, setSortField] = useState<SortField>('confidence_score');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  
  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(20);
  const [totalClauses, setTotalClauses] = useState(0);

  // Load clauses
  useEffect(() => {
    loadClauses();
  }, [docId, filters.clauseType, currentPage]);

  const loadClauses = async () => {
    try {
      setLoading(true);
      setError(null);

      const offset = (currentPage - 1) * pageSize;
      const clauseType = filters.clauseType || undefined;
      
      const fetchedClauses = await apiClient.getDocumentClauses(
        docId,
        clauseType,
        pageSize,
        offset
      );

      setClauses(fetchedClauses);
      setTotalClauses(fetchedClauses.length); // Note: This is a simplified pagination
      
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to load clauses';
      setError(errorMessage);
      onError?.(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Filter and sort clauses locally
  const getFilteredAndSortedClauses = () => {
    let filtered = clauses.filter(clause => {
      // Confidence filter
      if (clause.confidence_score < filters.minConfidence) return false;
      
      // Text search
      if (filters.searchText) {
        const searchLower = filters.searchText.toLowerCase();
        return clause.text.toLowerCase().includes(searchLower) ||
               clause.clause_type.toLowerCase().includes(searchLower);
      }
      
      return true;
    });

    // Sort
    filtered.sort((a, b) => {
      let aVal: any, bVal: any;
      
      switch (sortField) {
        case 'confidence_score':
          aVal = a.confidence_score;
          bVal = b.confidence_score;
          break;
        case 'clause_type':
          aVal = a.clause_type;
          bVal = b.clause_type;
          break;
        case 'created_at':
          aVal = new Date(a.created_at);
          bVal = new Date(b.created_at);
          break;
        case 'page_number':
          aVal = a.page_number || 0;
          bVal = b.page_number || 0;
          break;
        default:
          return 0;
      }

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    return filtered;
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) return null;
    return sortDirection === 'asc' ? 
      <ArrowUpIcon className="w-4 h-4" /> : 
      <ArrowDownIcon className="w-4 h-4" />;
  };

  const filteredClauses = getFilteredAndSortedClauses();

  // Get clause type statistics
  const getClauseStats = () => {
    const stats: Record<string, number> = {};
    clauses.forEach(clause => {
      stats[clause.clause_type] = (stats[clause.clause_type] || 0) + 1;
    });
    return stats;
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Loading clauses...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <DocumentTextIcon className="w-12 h-12 text-red-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-red-900 mb-2">Error Loading Clauses</h3>
        <p className="text-red-600 mb-4">{error}</p>
        <button
          onClick={loadClauses}
          className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
        >
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b bg-white p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Extracted Clauses</h2>
            {documentName && (
              <p className="text-sm text-gray-500 mt-1">{documentName}</p>
            )}
          </div>
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <ChartBarIcon className="w-4 h-4" />
            <span>{filteredClauses.length} of {clauses.length} clauses</span>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-4 items-center">
          <div className="flex items-center space-x-2">
            <FunnelIcon className="w-4 h-4 text-gray-500" />
            <select
              value={filters.clauseType}
              onChange={(e) => setFilters({ ...filters, clauseType: e.target.value })}
              className="text-sm border border-gray-300 rounded-md px-3 py-1 focus:ring-2 focus:ring-blue-500"
            >
              {CLAUSE_TYPES.map(type => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <label className="text-sm text-gray-600">Min Confidence:</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={filters.minConfidence}
              onChange={(e) => setFilters({ ...filters, minConfidence: parseFloat(e.target.value) })}
              className="w-20"
            />
            <span className="text-sm text-gray-600 w-8">
              {Math.round(filters.minConfidence * 100)}%
            </span>
          </div>

          <div className="flex-1 max-w-md">
            <input
              type="text"
              placeholder="Search clauses..."
              value={filters.searchText}
              onChange={(e) => setFilters({ ...filters, searchText: e.target.value })}
              className="w-full text-sm border border-gray-300 rounded-md px-3 py-1 focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Clause Type Statistics */}
        {clauses.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {Object.entries(getClauseStats()).map(([type, count]) => (
              <span
                key={type}
                className={`px-2 py-1 text-xs rounded-full ${getClauseTypeColor(type)}`}
              >
                {type.replace('_', ' ').toUpperCase()}: {count}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Clauses List */}
      <div className="flex-1 overflow-hidden">
        {filteredClauses.length === 0 ? (
          <div className="text-center py-12">
            <DocumentTextIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Clauses Found</h3>
            <p className="text-gray-500">
              {clauses.length === 0 
                ? 'No clauses have been extracted from this document yet.'
                : 'No clauses match your current filters.'
              }
            </p>
          </div>
        ) : (
          <div className="h-full overflow-y-auto">
            {/* Sort Headers */}
            <div className="sticky top-0 bg-gray-50 border-b px-4 py-2">
              <div className="grid grid-cols-12 gap-4 text-xs font-medium text-gray-500 uppercase tracking-wide">
                <div className="col-span-1">
                  <button
                    onClick={() => handleSort('page_number')}
                    className="flex items-center space-x-1 hover:text-gray-700"
                  >
                    <span>Page</span>
                    {getSortIcon('page_number')}
                  </button>
                </div>
                <div className="col-span-2">
                  <button
                    onClick={() => handleSort('clause_type')}
                    className="flex items-center space-x-1 hover:text-gray-700"
                  >
                    <span>Type</span>
                    {getSortIcon('clause_type')}
                  </button>
                </div>
                <div className="col-span-7">
                  <span>Clause Text</span>
                </div>
                <div className="col-span-1">
                  <button
                    onClick={() => handleSort('confidence_score')}
                    className="flex items-center space-x-1 hover:text-gray-700"
                  >
                    <span>Score</span>
                    {getSortIcon('confidence_score')}
                  </button>
                </div>
                <div className="col-span-1">
                  <span>Actions</span>
                </div>
              </div>
            </div>

            {/* Clauses */}
            <div className="divide-y">
              {filteredClauses.map((clause, index) => (
                <div
                  key={clause.clause_id}
                  className="p-4 hover:bg-gray-50 transition-colors"
                >
                  <div className="grid grid-cols-12 gap-4 items-start">
                    <div className="col-span-1 text-sm text-gray-500">
                      {clause.page_number || '-'}
                    </div>
                    
                    <div className="col-span-2">
                      <span className={`inline-block px-2 py-1 text-xs rounded-full ${getClauseTypeColor(clause.clause_type)}`}>
                        {clause.clause_type.replace('_', ' ').toUpperCase()}
                      </span>
                    </div>
                    
                    <div className="col-span-7">
                      <p className="text-sm text-gray-900 line-clamp-3">
                        {clause.text}
                      </p>
                      {clause.keywords && clause.keywords.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-2">
                          {clause.keywords.slice(0, 3).map((keyword, idx) => (
                            <span
                              key={idx}
                              className="px-1 py-0.5 bg-blue-100 text-blue-800 text-xs rounded"
                            >
                              {keyword}
                            </span>
                          ))}
                          {clause.keywords.length > 3 && (
                            <span className="text-xs text-gray-500">
                              +{clause.keywords.length - 3} more
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    
                    <div className="col-span-1">
                      <span className={`text-sm font-medium ${
                        clause.confidence_score >= 0.8 ? 'text-green-600' :
                        clause.confidence_score >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {formatConfidenceScore(clause.confidence_score)}
                      </span>
                    </div>
                    
                    <div className="col-span-1 flex space-x-1">
                      <button
                        onClick={() => setSelectedClause(clause)}
                        className="p-1 text-gray-400 hover:text-blue-600 transition-colors"
                        title="View details"
                      >
                        <EyeIcon className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => copyToClipboard(clause.text)}
                        className="p-1 text-gray-400 hover:text-green-600 transition-colors"
                        title="Copy text"
                      >
                        <ClipboardDocumentIcon className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Clause Detail Modal */}
      {selectedClause && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-96 overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Clause Details</h3>
                <button
                  onClick={() => setSelectedClause(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-center space-x-4">
                  <span className={`px-3 py-1 text-sm rounded-full ${getClauseTypeColor(selectedClause.clause_type)}`}>
                    {selectedClause.clause_type.replace('_', ' ').toUpperCase()}
                  </span>
                  <span className="text-sm text-gray-500">
                    Confidence: {formatConfidenceScore(selectedClause.confidence_score)}
                  </span>
                  {selectedClause.page_number && (
                    <span className="text-sm text-gray-500">
                      Page {selectedClause.page_number}
                    </span>
                  )}
                </div>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-gray-900 whitespace-pre-wrap">{selectedClause.text}</p>
                </div>
                
                {selectedClause.keywords && selectedClause.keywords.length > 0 && (
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-2">Keywords:</p>
                    <div className="flex flex-wrap gap-2">
                      {selectedClause.keywords.map((keyword, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-1 bg-blue-100 text-blue-800 text-sm rounded"
                        >
                          {keyword}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                
                <div className="flex justify-end space-x-2">
                  <button
                    onClick={() => copyToClipboard(selectedClause.text)}
                    className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    Copy Text
                  </button>
                  <button
                    onClick={() => setSelectedClause(null)}
                    className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}