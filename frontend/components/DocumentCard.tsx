'use client';
import React from 'react';
import { DocumentMetadata } from '@/lib/api-client';
import { SparklesIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';

export interface ExtendedDocument extends DocumentMetadata {
  riskScore?: number;
  moreScore?: number;
  abnormalFindings?: Array<{
    type: string;
    description: string;
    severity: 'low' | 'medium' | 'high';
  }>;
  aiSuggestions?: string[];
}

interface DocumentCardProps {
  doc: ExtendedDocument;
  onSelect?: (d: ExtendedDocument) => void;
  onAsk?: (d: ExtendedDocument) => void;
}

const riskColor = (risk?: number) => {
  if (risk === undefined) return 'var(--nebula-grey)';
  if (risk > 70) return 'var(--electric-rose, #ef4444)';
  if (risk > 40) return 'var(--vibrant-magenta)';
  return 'var(--electric-aqua)';
};

export function DocumentCard({ doc, onSelect, onAsk }: DocumentCardProps) {
  return (
    <div
      className="group gemini-card p-4 cursor-pointer transition-all duration-300 hover:shadow-xl"
      onClick={() => onSelect?.(doc)}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-semibold truncate" style={{color: 'var(--starlight-white)'}}>{doc.filename}</h4>
          <p className="text-[10px] uppercase tracking-wide mt-1 secondary-text">Document Preview</p>
        </div>
        {doc.riskScore !== undefined && (
          <div className="flex items-center space-x-2">
            <span className="text-xs font-semibold px-2 py-1 rounded-full" style={{
              background: 'rgba(36,216,218,0.08)',
              border: '1px solid rgba(138,145,180,0.25)',
              color: riskColor(doc.riskScore)
            }}>Risk {doc.riskScore}%</span>
          </div>
        )}
      </div>

      {/* Metrics Row */}
      <div className="flex flex-wrap gap-2 mb-3">
        <MetricBadge label="More Score" value={doc.moreScore} />
        <MetricBadge label="AI Fix" icon={<SparklesIcon className="h-3 w-3" />} />
      </div>

      {/* Abnormal Points */}
      {doc.abnormalFindings && doc.abnormalFindings.length > 0 && (
        <div className="rounded-lg border border-yellow-400/30 bg-yellow-400/10 p-3 mb-3">
          <div className="flex items-center mb-1 space-x-1 text-xs font-medium text-yellow-300">
            <ExclamationTriangleIcon className="h-4 w-4" />
            <span>Abnormal Points</span>
          </div>
          <ul className="space-y-1 max-h-20 overflow-y-auto pr-1 custom-scrollbar">
            {doc.abnormalFindings.slice(0,3).map((f,i) => (
              <li key={i} className="text-[11px] leading-snug text-yellow-100/90">
                {f.description}
              </li>
            ))}
            {doc.abnormalFindings.length > 3 && (
              <li className="text-[10px] italic text-yellow-200/70">+ {doc.abnormalFindings.length - 3} more...</li>
            )}
          </ul>
        </div>
      )}

      <button
        onClick={(e) => { e.stopPropagation(); onAsk?.(doc); }}
        className="w-full mt-auto btn-ghost text-xs py-2 px-3 flex items-center justify-center gap-1 hover:text-white"
        style={{borderRadius: 10}}
      >
        Ask a Question
      </button>
    </div>
  );
}

function MetricBadge({ label, value, icon }: { label: string; value?: number; icon?: React.ReactNode }) {
  return (
    <span className="text-[10px] font-medium tracking-wide px-2 py-1 rounded-full flex items-center gap-1 border" style={{
      background: 'rgba(240,242,252,0.04)',
      borderColor: 'rgba(138,145,180,0.3)',
      color: 'var(--starlight-white)'
    }}>
      {icon}
      {label}{value !== undefined && <span className="opacity-70"> {value}</span>}
    </span>
  );
}

export default DocumentCard;
