'use client'

import { useState } from 'react'

interface DocumentViewerProps {
  documentText: string
  highlightedSpans?: Array<{ start: number; end: number }>
  onTextSelect?: (selectedText: string, start: number, end: number) => void
}

export default function DocumentViewer({ 
  documentText, 
  highlightedSpans = [], 
  onTextSelect 
}: DocumentViewerProps) {
  const [selectedText, setSelectedText] = useState('')
  const [showQueryDialog, setShowQueryDialog] = useState(false)

  // Create highlighted text with evidence spans
  const createHighlightedText = () => {
    if (!highlightedSpans.length) {
      return documentText
    }

    // Sort spans by start position
    const sortedSpans = [...highlightedSpans].sort((a, b) => a.start - b.start)
    
    let result = ''
    let lastEnd = 0

    sortedSpans.forEach((span, idx) => {
      // Add text before this span
      result += documentText.slice(lastEnd, span.start)
      
      // Add highlighted text
      const spanText = documentText.slice(span.start, span.end)
      result += `<mark class="bg-yellow-200 border border-yellow-400 rounded px-1" data-span-id="${idx}">${spanText}</mark>`
      
      lastEnd = span.end
    })

    // Add remaining text
    result += documentText.slice(lastEnd)
    
    return result
  }

  const handleTextSelection = () => {
    const selection = window.getSelection()
    if (selection && selection.toString().trim()) {
      const selectedText = selection.toString()
      setSelectedText(selectedText)
      setShowQueryDialog(true)
      
      // Calculate rough position (simplified)
      const range = selection.getRangeAt(0)
      const startOffset = range.startOffset
      const endOffset = range.endOffset
      
      if (onTextSelect) {
        onTextSelect(selectedText, startOffset, endOffset)
      }
    }
  }

  const handleQuerySubmit = async (question: string) => {
    console.log('Submitting query about selected text:', selectedText, 'Question:', question)
    // This would make an API call to query the selected text
    setShowQueryDialog(false)
    setSelectedText('')
  }

  return (
    <div className="space-y-4">
      {/* Document Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Document Text</h3>
        <div className="flex space-x-2">
          <button className="btn-secondary text-sm">
            🔍 Search
          </button>
          <button className="btn-secondary text-sm">
            📥 Download
          </button>
        </div>
      </div>

      {/* Evidence Legend */}
      {highlightedSpans.length > 0 && (
        <div className="card bg-yellow-50 border-yellow-200">
          <div className="flex items-center space-x-2">
            <span className="w-4 h-4 bg-yellow-200 border border-yellow-400 rounded"></span>
            <span className="text-sm font-medium text-yellow-800">
              Highlighted: Evidence supporting the current clause
            </span>
            <span className="text-xs text-yellow-700">
              ({highlightedSpans.length} span{highlightedSpans.length !== 1 ? 's' : ''})
            </span>
          </div>
        </div>
      )}

      {/* Document Content */}
      <div className="card max-h-96 overflow-y-auto">
        <div 
          className="prose prose-sm max-w-none leading-relaxed select-text"
          onMouseUp={handleTextSelection}
          dangerouslySetInnerHTML={{
            __html: createHighlightedText()
          }}
        />
      </div>

      {/* Selection Query Dialog */}
      {showQueryDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">
              Ask about selected text
            </h4>
            
            <div className="mb-4">
              <div className="text-sm text-gray-600 mb-2">Selected text:</div>
              <div className="bg-gray-100 p-3 rounded text-sm italic">
                "{selectedText}"
              </div>
            </div>

            <QueryForm 
              onSubmit={handleQuerySubmit}
              onCancel={() => setShowQueryDialog(false)}
              placeholder="What would you like to know about this text?"
            />
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="text-sm text-gray-500 text-center">
        💡 Select any text to ask questions about it, or click highlighted sections to see evidence
      </div>
    </div>
  )
}

// Simple query form component
interface QueryFormProps {
  onSubmit: (query: string) => void
  onCancel: () => void
  placeholder?: string
}

function QueryForm({ onSubmit, onCancel, placeholder = "Ask a question..." }: QueryFormProps) {
  const [query, setQuery] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      onSubmit(query.trim())
      setQuery('')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={placeholder}
        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
        rows={3}
        autoFocus
      />
      
      <div className="flex justify-end space-x-3">
        <button
          type="button"
          onClick={onCancel}
          className="btn-secondary"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={!query.trim()}
          className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Ask Question
        </button>
      </div>
    </form>
  )
}