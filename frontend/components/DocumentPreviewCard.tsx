'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { 
  ExclamationTriangleIcon, 
  WrenchScrewdriverIcon,
  ChatBubbleLeftEllipsisIcon,
  ChartBarIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline'

interface DocumentPreviewCardProps {
  document: {
    id: string
    name: string
    summary: string
    riskScore: number
    abnormalPoints: string[]
    aiSuggestions: string[]
    type: string
    status: string
  }
  isSelected: boolean
  onSelect: (selected: boolean) => void
  onPreview: () => void
}

export default function DocumentPreviewCard({
  document,
  isSelected,
  onSelect,
  onPreview
}: DocumentPreviewCardProps) {
  const router = useRouter()
  const [showAbnormalModal, setShowAbnormalModal] = useState(false)
  const [showAIFixModal, setShowAIFixModal] = useState(false)
  const [showMoreScore, setShowMoreScore] = useState(false)

  const handleAskAI = () => {
    const params = new URLSearchParams({
      docId: document.id,
      docName: document.name,
      docType: document.type,
      docSummary: document.summary,
      riskScore: document.riskScore.toString()
    })
    router.push(`/chat?${params.toString()}`)
  }

  const getRiskColor = (score: number) => {
    if (score >= 70) return 'risk-high'
    if (score >= 40) return 'risk-medium'
    return 'risk-low'
  }

  const getRiskIcon = (score: number) => {
    if (score >= 70) return '🚨'
    if (score >= 40) return '⚠️'
    return '✅'
  }

  return (
    <>
      <div className="gemini-card transition-all duration-300 p-6 relative group">
        {/* Selection Checkbox */}
        <div className="absolute top-4 right-4">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={(e) => onSelect(e.target.checked)}
            className="h-4 w-4 rounded border-2 focus:ring-2"
            style={{borderColor: 'var(--nebula-grey)', accentColor: 'var(--core-spark-blue)'}}
          />
        </div>

        {/* Document Preview Section */}
        <div className="mb-4 pr-8">
          <div className="flex items-start space-x-3 mb-3">
            <DocumentTextIcon className="h-6 w-6 flex-shrink-0 mt-1" style={{color: 'var(--electric-aqua)'}} />
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold truncate mb-1" style={{color: 'var(--starlight-white)'}}>{document.name}</h3>
              <p className="text-sm secondary-text line-clamp-2 leading-relaxed">{document.summary}</p>
            </div>
          </div>
        </div>

        {/* Risk Score and More Score */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className={`risk-badge ${getRiskColor(document.riskScore)}`}>
              {getRiskIcon(document.riskScore)} Risk {document.riskScore}%
            </div>
            <button
              onClick={() => setShowMoreScore(!showMoreScore)}
              className="text-sm gemini-link font-medium transition-colors"
            >
              More Score
            </button>
          </div>
        </div>

        {/* More Score Details */}
        {showMoreScore && (
          <div className="mb-4 p-4 gemini-card">
            <h4 className="text-sm font-medium mb-3" style={{color: 'var(--starlight-white)'}}>Detailed Risk Breakdown</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="secondary-text">Liability Clauses:</span>
                <span className={`font-medium`} style={{color: document.riskScore >= 70 ? '#ef4444' : 'var(--vibrant-magenta)'}}>
                  {Math.floor(document.riskScore * 0.4)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="secondary-text">Payment Terms:</span>
                <span className="font-medium" style={{color: 'var(--electric-aqua)'}}>{Math.floor(document.riskScore * 0.2)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="secondary-text">Termination:</span>
                <span className={`font-medium`} style={{color: document.riskScore >= 60 ? 'var(--vibrant-magenta)' : 'var(--electric-aqua)'}}>
                  {Math.floor(document.riskScore * 0.3)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="secondary-text">Data Privacy:</span>
                <span className="font-medium" style={{color: 'var(--electric-aqua)'}}>{Math.floor(document.riskScore * 0.1)}%</span>
              </div>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-2 mb-4">
          <button
            onClick={() => setShowAbnormalModal(true)}
            className="risk-badge risk-high text-xs flex items-center"
          >
            <ExclamationTriangleIcon className="h-3 w-3 mr-1" />
            Abnormal Points ({document.abnormalPoints.length})
          </button>
          <button
            onClick={() => setShowAIFixModal(true)}
            className="risk-badge risk-low text-xs flex items-center"
          >
            <WrenchScrewdriverIcon className="h-3 w-3 mr-1" />
            AI Fix ({document.aiSuggestions.length})
          </button>
        </div>

        {/* Bottom Actions */}
        <div className="flex space-x-2">
          <button
            onClick={onPreview}
            className="flex-1 btn-secondary text-sm"
          >
            Preview Document
          </button>
          <button
            onClick={handleAskAI}
            className="btn-primary text-sm flex items-center"
          >
            <ChatBubbleLeftEllipsisIcon className="h-4 w-4 mr-1" />
            Ask AI
          </button>
        </div>
      </div>

      {/* Abnormal Points Modal */}
      {showAbnormalModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="gemini-card max-w-2xl w-full max-h-[80vh] m-4">
            <div className="flex items-center justify-between p-6 border-b" style={{ borderColor: 'var(--nebula-grey)' }}>
              <h3 className="text-lg font-semibold gemini-gradient-text flex items-center">
                <ExclamationTriangleIcon className="h-5 w-5 mr-2" style={{ color: 'var(--electric-rose)' }} />
                Abnormal Points Found
              </h3>
              <button
                onClick={() => setShowAbnormalModal(false)}
                className="text-primary-text hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-6 overflow-y-auto max-h-96">
              <div className="space-y-4">
                {document.abnormalPoints.map((point, idx) => (
                  <div key={idx} className="risk-high">
                    <div className="flex items-start space-x-3">
                      <div className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0" style={{ backgroundColor: 'var(--electric-rose)', color: 'var(--starlight-white)' }}>
                        <span className="text-sm font-bold">{idx + 1}</span>
                      </div>
                      <div className="flex-1">
                        <p className="text-primary-text text-sm leading-relaxed">{point}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="px-6 py-4 border-t" style={{ borderColor: 'var(--nebula-grey)', backgroundColor: 'rgba(240, 242, 252, 0.02)' }}>
              <div className="flex justify-end space-x-3">
                <button
                  onClick={() => setShowAbnormalModal(false)}
                  className="btn-secondary"
                >
                  Close
                </button>
                <button
                  onClick={() => {
                    setShowAbnormalModal(false)
                    setShowAIFixModal(true)
                  }}
                  className="btn-primary"
                >
                  View AI Fixes
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* AI Fix Modal */}
      {showAIFixModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="gemini-card w-full max-w-2xl m-4 max-h-[80vh]">
            <div className="flex items-center justify-between p-6 border-b" style={{ borderColor: 'var(--nebula-grey)' }}>
              <h3 className="text-lg font-semibold gemini-gradient-text flex items-center">
                <WrenchScrewdriverIcon className="h-5 w-5 mr-2" style={{ color: 'var(--electric-aqua)' }} />
                AI-Powered Negotiation Suggestions
              </h3>
              <button
                onClick={() => setShowAIFixModal(false)}
                className="text-primary-text hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-6 overflow-y-auto max-h-96">
              <div className="space-y-4">
                {document.aiSuggestions.map((suggestion, idx) => (
                  <div key={idx} className="risk-low">
                    <div className="flex items-start space-x-3">
                      <div className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0" style={{ backgroundColor: 'var(--electric-aqua)', color: 'var(--cosmic-background)' }}>
                        <span className="text-sm font-bold">{idx + 1}</span>
                      </div>
                      <div className="flex-1">
                        <p className="text-primary-text text-sm leading-relaxed">{suggestion}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="px-6 py-4 border-t" style={{ borderColor: 'var(--nebula-grey)', backgroundColor: 'rgba(240, 242, 252, 0.02)' }}>
              <div className="flex justify-end space-x-3">
                <button
                  onClick={() => setShowAIFixModal(false)}
                  className="btn-secondary"
                >
                  Close
                </button>
                <button className="btn-primary">
                  📋 Copy All Suggestions
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}