'use client'

import { useState } from 'react'
import { 
  DocumentTextIcon,
  CheckCircleIcon,
  XMarkIcon,
  MagnifyingGlassIcon,
  FunnelIcon
} from '@heroicons/react/24/outline'

interface Document {
  id: string
  name: string
  type: string
  summary: string
  riskScore: number
  status: string
  uploadedAt: string
}

interface DocumentContextManagerProps {
  documents: Document[]
  selectedDocs: string[]
  onSelectionChange: (selectedDocs: string[]) => void
  onClose: () => void
}

export default function DocumentContextManager({
  documents,
  selectedDocs,
  onSelectionChange,
  onClose
}: DocumentContextManagerProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [filterType, setFilterType] = useState<string>('all')
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'risk'>('date')

  const documentTypes = Array.from(new Set(documents.map(doc => doc.type)))

  const filteredAndSortedDocs = documents
    .filter(doc => {
      const matchesSearch = doc.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           doc.summary.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesType = filterType === 'all' || doc.type === filterType
      return matchesSearch && matchesType
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name)
        case 'risk':
          return b.riskScore - a.riskScore
        case 'date':
        default:
          return new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime()
      }
    })

  const handleToggleAll = () => {
    if (selectedDocs.length === filteredAndSortedDocs.length) {
      onSelectionChange([])
    } else {
      onSelectionChange(filteredAndSortedDocs.map(doc => doc.id))
    }
  }

  const handleToggleDocument = (docId: string) => {
    if (selectedDocs.includes(docId)) {
      onSelectionChange(selectedDocs.filter(id => id !== docId))
    } else {
      onSelectionChange([...selectedDocs, docId])
    }
  }

  const getDocumentIcon = (type: string) => {
    switch (type) {
      case 'rental_agreement':
        return '🏠'
      case 'loan_contract':
        return '💰'
      case 'employment_contract':
        return '👥'
      case 'service_agreement':
        return '🔧'
      default:
        return '📄'
    }
  }

  const formatDocumentType = (type: string) => {
    return type.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ')
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-semibold text-gray-900">Manage Document Context</h3>
          <p className="text-sm text-gray-600 mt-1">
            Select documents to include in your AI assistant's context
          </p>
        </div>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 p-1"
        >
          <XMarkIcon className="h-6 w-6" />
        </button>
      </div>

      {/* Controls */}
      <div className="space-y-4 mb-6">
        {/* Search */}
        <div className="relative">
          <MagnifyingGlassIcon className="h-5 w-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search documents..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        {/* Filters and Sort */}
        <div className="flex items-center justify-between space-x-4">
          <div className="flex items-center space-x-3">
            <FunnelIcon className="h-5 w-5 text-gray-400" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="border border-gray-300 rounded px-3 py-1 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="all">All Types</option>
              {documentTypes.map(type => (
                <option key={type} value={type}>
                  {formatDocumentType(type)}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center space-x-3">
            <span className="text-sm text-gray-600">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'name' | 'date' | 'risk')}
              className="border border-gray-300 rounded px-3 py-1 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="date">Upload Date</option>
              <option value="name">Name</option>
              <option value="risk">Risk Score</option>
            </select>
          </div>
        </div>

        {/* Bulk Actions */}
        <div className="flex items-center justify-between">
          <button
            onClick={handleToggleAll}
            className="text-sm text-blue-600 hover:text-blue-800 font-medium"
          >
            {selectedDocs.length === filteredAndSortedDocs.length ? 'Deselect All' : 'Select All'}
          </button>
          <span className="text-sm text-gray-600">
            {selectedDocs.length} of {filteredAndSortedDocs.length} selected
          </span>
        </div>
      </div>

      {/* Documents List */}
      <div className="flex-1 overflow-y-auto space-y-2 border border-gray-200 rounded-lg p-2">
        {filteredAndSortedDocs.length === 0 ? (
          <div className="text-center py-8">
            <DocumentTextIcon className="h-12 w-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500">No documents found</p>
            <p className="text-sm text-gray-400 mt-1">Try adjusting your search or filters</p>
          </div>
        ) : (
          filteredAndSortedDocs.map((doc) => (
            <div
              key={doc.id}
              className={`
                p-3 rounded-lg border-2 transition-all cursor-pointer hover:shadow-sm
                ${selectedDocs.includes(doc.id) 
                  ? 'border-blue-300 bg-blue-50' 
                  : 'border-gray-200 bg-white hover:border-gray-300'
                }
              `}
              onClick={() => handleToggleDocument(doc.id)}
            >
              <div className="flex items-start space-x-3">
                {/* Checkbox */}
                <div className="flex-shrink-0 mt-1">
                  <div className={`
                    w-5 h-5 rounded border-2 flex items-center justify-center
                    ${selectedDocs.includes(doc.id) 
                      ? 'border-blue-500 bg-blue-500' 
                      : 'border-gray-300'
                    }
                  `}>
                    {selectedDocs.includes(doc.id) && (
                      <CheckCircleIcon className="h-3 w-3 text-white" />
                    )}
                  </div>
                </div>

                {/* Document Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2 mb-1">
                    <span className="text-lg">{getDocumentIcon(doc.type)}</span>
                    <h4 className="font-medium text-gray-900 truncate">{doc.name}</h4>
                  </div>
                  
                  <p className="text-sm text-gray-600 line-clamp-2 mb-2">{doc.summary}</p>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span className="text-xs text-gray-500">
                        {formatDocumentType(doc.type)}
                      </span>
                      <span className="text-xs text-gray-400">
                        {new Date(doc.uploadedAt).toLocaleDateString()}
                      </span>
                    </div>
                    
                    <div className={`
                      px-2 py-1 rounded-full text-xs font-medium
                      ${doc.riskScore >= 70 ? 'bg-red-100 text-red-800' :
                        doc.riskScore >= 40 ? 'bg-yellow-100 text-yellow-800' :
                        'bg-green-100 text-green-800'
                      }
                    `}>
                      {doc.riskScore}% risk
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="mt-6 flex justify-between items-center">
        <div className="text-sm text-gray-600">
          Selected documents will be available to the AI assistant for questions and analysis.
        </div>
        
        <div className="flex space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:text-gray-800 font-medium"
          >
            Cancel
          </button>
          <button
            onClick={onClose}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 font-medium flex items-center"
          >
            Apply Context ({selectedDocs.length} document{selectedDocs.length !== 1 ? 's' : ''})
          </button>
        </div>
      </div>
    </div>
  )
}