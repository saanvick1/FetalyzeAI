import type { HistoryEntry } from '../App'
import './HistoryPanel.css'

interface Props {
  history: HistoryEntry[]
  onSelect: (entry: HistoryEntry) => void
}

const RISK_COLORS = {
  Normal:       { bg: '#f0fdf4', border: '#bbf7d0', text: '#15803d' },
  Suspect:      { bg: '#fffbeb', border: '#fde68a', text: '#b45309' },
  Pathological: { bg: '#fef2f2', border: '#fecaca', text: '#b91c1c' },
}

export function HistoryPanel({ history, onSelect }: Props) {
  if (history.length === 0) {
    return (
      <div className="history-empty">
        <svg viewBox="0 0 48 48" fill="none" width="60" height="60">
          <circle cx="24" cy="24" r="22" stroke="#e5e7eb" strokeWidth="2" fill="#f9fafb"/>
          <path d="M16 24h4l2-6 4 12 3-6h3" stroke="#d1d5db" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <h3>No analysis history yet</h3>
        <p>Run your first CTG analysis to see results here.</p>
      </div>
    )
  }

  return (
    <div className="history-panel">
      <div className="history-panel__header">
        <h2>Analysis History</h2>
        <span className="history-panel__count">{history.length} session{history.length !== 1 ? 's' : ''}</span>
      </div>
      <div className="history-list">
        {history.map((entry, i) => {
          const cfg = RISK_COLORS[entry.risk_label]
          return (
            <button
              key={entry.id ?? i}
              className="history-item"
              onClick={() => onSelect(entry)}
            >
              <div
                className="history-item__badge"
                style={{ background: cfg.bg, borderColor: cfg.border, color: cfg.text }}
              >
                {entry.risk_label}
              </div>
              <div className="history-item__body">
                <div className="history-item__row">
                  <span className="history-item__metric">
                    Confidence: <strong>{(entry.confidence * 100).toFixed(1)}%</strong>
                  </span>
                  <span className="history-item__metric">
                    FRS: <strong>{entry.fetal_reserve_score}/100</strong>
                  </span>
                </div>
                <div className="history-item__probs">
                  <MiniBar label="N" value={entry.prob_normal} color="#16a34a" />
                  <MiniBar label="S" value={entry.prob_suspect} color="#d97706" />
                  <MiniBar label="P" value={entry.prob_pathological} color="#dc2626" />
                </div>
              </div>
              <div className="history-item__time">
                {formatTime(entry.timestamp)}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function MiniBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="mini-bar">
      <span className="mini-bar__label" style={{ color }}>{label}</span>
      <div className="mini-bar__track">
        <div className="mini-bar__fill" style={{ width: `${value * 100}%`, background: color }} />
      </div>
      <span className="mini-bar__value">{(value * 100).toFixed(0)}%</span>
    </div>
  )
}

function formatTime(d: Date) {
  const now = new Date()
  const diff = now.getTime() - d.getTime()
  if (diff < 60000) return 'just now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}
