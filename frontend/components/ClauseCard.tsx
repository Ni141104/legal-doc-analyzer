'use client'

import type { ClauseCard as ClauseCardType, RiskFlag, Recommendation } from '@/types'

interface ClauseCardProps {
  clause: ClauseCardType
  isSelected?: boolean
  onHighlight?: (spans: Array<{ start: number; end: number }>) => void
}

export default function ClauseCard({ clause, isSelected = false, onHighlight }: ClauseCardProps) {
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'HIGH': return 'text-red-700 bg-red-100 border-red-300'
      case 'MEDIUM': return 'text-yellow-700 bg-yellow-100 border-yellow-300'
      case 'LOW': return 'text-green-700 bg-green-100 border-green-300'
      default: return 'text-gray-700 bg-gray-100 border-gray-300'
    }
  }

  const getRiskIcon = (level: string) => {
    switch (level) {
      case 'HIGH': return '🚨'
      case 'MEDIUM': return '⚠️'
      case 'LOW': return '✅'
      default: return '❓'
    }
  }

  const handleShowEvidence = () => {
    if (onHighlight && clause.source_spans) {
      onHighlight(clause.source_spans)
    }
  }

  return (
    <div className={`card transition-all duration-200 ${isSelected ? 'ring-2 ring-primary-500 shadow-lg' : ''}`}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getRiskColor(clause.risk_level)}`}>
            {getRiskIcon(clause.risk_level)} {clause.risk_level} RISK
          </span>
          <span className="text-sm text-gray-500">
            Clause #{clause.clause_id}
          </span>
        </div>
        
        <button
          onClick={handleShowEvidence}
          className="text-sm text-primary-600 hover:text-primary-800 font-medium"
        >
          📍 Show Evidence
        </button>
      </div>

      {/* Original Text Preview */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Original Text:</h4>
        <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded border-l-4 border-gray-300">
          {clause.original_text.length > 200 
            ? `${clause.original_text.substring(0, 200)}...` 
            : clause.original_text
          }
        </div>
      </div>

      {/* Simplified Summary */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-900 mb-2">Plain English Summary:</h4>
        <p className="text-gray-700 leading-relaxed">
          {clause.simplified_summary}
        </p>
      </div>

      {/* Risk Flags */}
      {clause.risk_flags && clause.risk_flags.length > 0 && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Risk Flags:</h4>
          <div className="space-y-2">
            {clause.risk_flags.map((flag: RiskFlag, idx: number) => (
              <div key={idx} className={`p-3 rounded-lg border ${getRiskColor(flag.severity)}`}>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium capitalize">{flag.flag_type.replace('_', ' ')}</span>
                  <span className="text-xs uppercase tracking-wide">{flag.severity}</span>
                </div>
                <p className="text-sm mb-2">{flag.description}</p>
                {flag.mitigation_advice && (
                  <p className="text-xs italic">💡 {flag.mitigation_advice}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {clause.recommendations && clause.recommendations.length > 0 && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Recommendations:</h4>
          <div className="space-y-2">
            {clause.recommendations.map((rec: Recommendation, idx: number) => (
              <div key={idx} className="border border-gray-200 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    rec.type === 'negotiate' ? 'bg-blue-100 text-blue-800' :
                    rec.type === 'clarify' ? 'bg-yellow-100 text-yellow-800' :
                    rec.type === 'reject' ? 'bg-red-100 text-red-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {rec.type.toUpperCase()}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded ${getRiskColor(rec.priority)}`}>
                    {rec.priority}
                  </span>
                </div>
                <p className="text-sm text-gray-700 mb-1">{rec.description}</p>
                <p className="text-xs text-gray-600 italic">{rec.rationale}</p>
                {rec.specific_language && (
                  <div className="mt-2 p-2 bg-blue-50 rounded text-xs">
                    <strong>Suggested language:</strong> "{rec.specific_language}"
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Negotiation Points */}
      {clause.negotiation_points && clause.negotiation_points.length > 0 && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Negotiation Points:</h4>
          <ul className="space-y-1">
            {clause.negotiation_points.map((point: string, idx: number) => (
              <li key={idx} className="text-sm text-gray-700 flex items-start">
                <span className="text-primary-600 mr-2">•</span>
                {point}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Legal Precedents */}
      {clause.legal_precedents && clause.legal_precedents.length > 0 && (
        <div className="border-t border-gray-200 pt-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Legal Context:</h4>
          <div className="space-y-1">
            {clause.legal_precedents.map((precedent: string, idx: number) => (
              <p key={idx} className="text-xs text-gray-600 italic">
                📚 {precedent}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* Footer Actions */}
      <div className="border-t border-gray-200 pt-4 mt-4">
        <div className="flex justify-between items-center">
          <div className="text-xs text-gray-500">
            {clause.source_spans?.length || 0} evidence span{(clause.source_spans?.length || 0) !== 1 ? 's' : ''}
          </div>
          <div className="flex space-x-2">
            <button className="text-xs text-primary-600 hover:text-primary-800">
              📋 Copy Analysis
            </button>
            <button className="text-xs text-primary-600 hover:text-primary-800">
              🔗 Share
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}