'use client'

import { useState } from 'react'
import ClauseCard from './ClauseCard'
import type { ClauseCard as ClauseCardType } from '@/types'

interface Clause {
  id: string
  original_text: string
  simplified_text: string
  risk_level: 'HIGH' | 'MEDIUM' | 'LOW'
  source_spans: Array<{ start: number; end: number }>
  recommendations: string[]
  negotiation_points: string[]
}

interface ClauseNavigatorProps {
  clauses: Clause[]
  selectedClauseId?: string
  onClauseSelect: (clauseId: string) => void
  documentText: string
}

export default function ClauseNavigator({ 
  clauses, 
  selectedClauseId, 
  onClauseSelect, 
  documentText 
}: ClauseNavigatorProps) {
  const [filterRisk, setFilterRisk] = useState<string>('ALL')
  const [searchTerm, setSearchTerm] = useState('')

  const filteredClauses = clauses.filter(clause => {
    const matchesRisk = filterRisk === 'ALL' || clause.risk_level === filterRisk
    const matchesSearch = !searchTerm || 
      clause.simplified_text.toLowerCase().includes(searchTerm.toLowerCase()) ||
      clause.original_text.toLowerCase().includes(searchTerm.toLowerCase())
    
    return matchesRisk && matchesSearch
  })

  const riskCounts = {
    HIGH: clauses.filter(c => c.risk_level === 'HIGH').length,
    MEDIUM: clauses.filter(c => c.risk_level === 'MEDIUM').length,
    LOW: clauses.filter(c => c.risk_level === 'LOW').length
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">
          Clause Analysis ({clauses.length})
        </h2>
        
        <div className="flex space-x-2">
          <span className="badge badge-red">{riskCounts.HIGH} High Risk</span>
          <span className="badge badge-yellow">{riskCounts.MEDIUM} Medium</span>
          <span className="badge badge-green">{riskCounts.LOW} Low</span>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1">
          <input
            type="text"
            placeholder="Search clauses..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>
        
        <select
          value={filterRisk}
          onChange={(e) => setFilterRisk(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        >
          <option value="ALL">All Risk Levels</option>
          <option value="HIGH">High Risk Only</option>
          <option value="MEDIUM">Medium Risk Only</option>
          <option value="LOW">Low Risk Only</option>
        </select>
      </div>

      {/* Risk Summary */}
      {riskCounts.HIGH > 0 && (
        <div className="card bg-red-50 border-red-200">
          <div className="flex items-start space-x-3">
            <div className="w-2 h-2 bg-red-400 rounded-full mt-2 flex-shrink-0"></div>
            <div>
              <h3 className="font-medium text-red-900 mb-1">
                ⚠️ {riskCounts.HIGH} High-Risk Clause{riskCounts.HIGH !== 1 ? 's' : ''} Found
              </h3>
              <p className="text-red-800 text-sm">
                These clauses may significantly impact your rights or obligations. Review carefully before signing.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Clause Cards */}
      <div className="space-y-4">
        {filteredClauses.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            No clauses found matching your filters.
          </div>
        ) : (
          filteredClauses.map((clause) => (
            <div
              key={clause.id}
              className={`transition-all duration-200 ${
                selectedClauseId === clause.id ? 'ring-2 ring-primary-500' : ''
              }`}
              onClick={() => onClauseSelect(clause.id)}
            >
              <ClauseCard
                clause={{
                  clause_id: clause.id,
                  original_text: clause.original_text,
                  simplified_summary: clause.simplified_text,
                  risk_level: clause.risk_level,
                  source_spans: clause.source_spans,
                  risk_flags: [],
                  recommendations: [],
                  negotiation_points: clause.negotiation_points,
                  legal_precedents: []
                } as ClauseCardType}
                isSelected={selectedClauseId === clause.id}
                onHighlight={(spans) => {
                  // This would trigger highlighting in the document viewer
                  console.log('Highlight spans:', spans)
                }}
              />
            </div>
          ))
        )}
      </div>

      {/* Export Actions */}
      {clauses.length > 0 && (
        <div className="card bg-gray-50">
          <h3 className="font-medium text-gray-900 mb-3">Export Analysis</h3>
          <div className="flex flex-wrap gap-3">
            <button className="btn-secondary text-sm">
              📄 Download PDF Report
            </button>
            <button className="btn-secondary text-sm">
              📋 Copy Negotiation Points
            </button>
            <button className="btn-secondary text-sm">
              📊 Export Risk Summary
            </button>
          </div>
        </div>
      )}
    </div>
  )
}