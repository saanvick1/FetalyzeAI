import { useMemo, useState } from 'react'
import { predictLocally } from '../lib/predictor'
import { DEFAULT_VALUES, FEATURE_GROUPS, FEATURES, PRESETS, type FeatureKey, type FeatureValues } from '../lib/features'
import type { PredictionResult } from '../lib/api'

export function ReserveNetPredictorPanel() {
  const [values, setValues] = useState<FeatureValues>(DEFAULT_VALUES)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeGroup, setActiveGroup] = useState<string>(FEATURE_GROUPS[0])

  const groupFeatures = useMemo(() => FEATURES.filter(f => f.group === activeGroup), [activeGroup])

  function handleChange(key: FeatureKey, raw: string) {
    const v = parseFloat(raw)
    setValues(prev => ({ ...prev, [key]: Number.isNaN(v) ? 0 : v }))
  }

  function applyPreset(preset: typeof PRESETS[0]) {
    setValues({ ...DEFAULT_VALUES, ...preset.values })
    setError(null)
  }

  function resetToDefaults() {
    setValues({ ...DEFAULT_VALUES })
    setResult(null)
    setError(null)
  }

  function runPrediction() {
    try {
      setLoading(true)
      setError(null)
      setResult(predictLocally(values))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app__layout">
      <div className="form-card">
        <div className="form-card__presets">
          <span className="form-card__presets-label">Quick presets:</span>
          {PRESETS.map(p => (
            <button
              key={p.label}
              className={`preset-btn preset-btn--${p.tag.toLowerCase()}`}
              onClick={() => applyPreset(p)}
              title={p.desc}
            >
              {p.label}
            </button>
          ))}
          <button className="preset-btn preset-btn--reset" onClick={resetToDefaults}>
            Reset
          </button>
        </div>

        <div className="form-card__groups">
          {FEATURE_GROUPS.map(g => (
            <button
              key={g}
              className={`group-tab ${activeGroup === g ? 'group-tab--active' : ''}`}
              onClick={() => setActiveGroup(g)}
            >
              {g}
            </button>
          ))}
        </div>

        <div className="form-card__fields">
          {groupFeatures.map(feat => (
            <FeatureInput
              key={feat.key}
              featureKey={feat.key}
              label={feat.label}
              unit={feat.unit}
              description={feat.description}
              min={feat.min}
              max={feat.max}
              step={feat.step}
              importance={feat.importance}
              value={values[feat.key]}
              onChange={handleChange}
            />
          ))}
        </div>

        <div className="form-card__footer">
          <div className="form-card__count">
            {FEATURES.length} features across {FEATURE_GROUPS.length} groups
          </div>
          <button className="submit-btn" onClick={runPrediction} disabled={loading}>
            {loading ? (
              <>
                <span className="submit-btn__spinner" />
                Analysing...
              </>
            ) : (
              <>
                <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
                  <path d="M10 2a8 8 0 100 16A8 8 0 0010 2zm3.707 6.707l-4 4a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L9 10.586l3.293-3.293a1 1 0 011.414 1.414z" />
                </svg>
                Run ReserveNet Predictor
              </>
            )}
          </button>
        </div>
      </div>

      <div className="result-panel">
        {error ? <ErrorState message={error} /> : result ? <ReserveNetResult result={result} /> : <EmptyState />}
      </div>
    </div>
  )
}

function FeatureInput({
  featureKey,
  label,
  unit,
  description,
  min,
  max,
  step,
  importance,
  value,
  onChange,
}: {
  featureKey: FeatureKey
  label: string
  unit: string
  description: string
  min: number
  max: number
  step: number
  importance: string
  value: number
  onChange: (key: FeatureKey, val: string) => void
}) {
  const displayVal = step < 0.01 ? value.toFixed(4) : step < 0.1 ? value.toFixed(3) : step < 1 ? value.toFixed(1) : String(Math.round(value))
  const pct = ((value - min) / (max - min)) * 100

  return (
    <div className="feature-input">
      <div className="feature-input__header">
        <div className="feature-input__title">
          <span className={`feature-input__importance feature-input__importance--${importance}`} />
          <label className="feature-input__label" htmlFor={featureKey}>{label}</label>
          {unit && <span className="feature-input__unit">{unit}</span>}
        </div>
        <span className="feature-input__value">{displayVal}</span>
      </div>
      <input
        id={featureKey}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(featureKey, e.target.value)}
        className="feature-input__slider"
        style={{ '--pct': `${Math.max(0, Math.min(100, pct))}%` } as React.CSSProperties}
      />
      <div className="feature-input__range">
        <span>{min}</span>
        <span className="feature-input__desc">{description}</span>
        <span>{max}</span>
      </div>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(featureKey, e.target.value)}
        className="feature-input__number"
      />
    </div>
  )
}

function ReserveNetResult({ result }: { result: PredictionResult }) {
  const probData = [
    { name: 'Normal', value: Math.round(result.prob_normal * 100), fill: '#16a34a' },
    { name: 'Suspect', value: Math.round(result.prob_suspect * 100), fill: '#d97706' },
    { name: 'Pathological', value: Math.round(result.prob_pathological * 100), fill: '#dc2626' },
  ]

  const sortedProb = [...probData].sort((a, b) => b.value - a.value)
  const topRisk = sortedProb[0]
  const runnerUp = sortedProb[1]
  const margin = Math.max(0, topRisk.value - runnerUp.value)
  const decisionLevel = result.risk_label === 'Pathological'
    ? 'Urgent obstetric review'
    : result.risk_label === 'Suspect'
      ? 'Close monitoring'
      : 'Routine observation'

  return (
    <>
      <div className="result-risk" style={{ background: result.risk_label === 'Pathological' ? '#fef2f2' : result.risk_label === 'Suspect' ? '#fffbeb' : '#f0fdf4', borderColor: result.risk_label === 'Pathological' ? '#fecaca' : result.risk_label === 'Suspect' ? '#fde68a' : '#bbf7d0' }}>
        <div className="result-risk__icon" style={{ background: result.risk_label === 'Pathological' ? '#dc2626' : result.risk_label === 'Suspect' ? '#d97706' : '#16a34a' }}>
          {result.risk_label === 'Pathological' ? '!!' : result.risk_label === 'Suspect' ? '!' : '✓'}
        </div>
        <div className="result-risk__body">
          <div className="result-risk__tag" style={{ color: result.risk_label === 'Pathological' ? '#dc2626' : result.risk_label === 'Suspect' ? '#d97706' : '#16a34a' }}>{decisionLevel}</div>
          <div className="result-risk__label" style={{ color: result.risk_label === 'Pathological' ? '#dc2626' : result.risk_label === 'Suspect' ? '#d97706' : '#16a34a' }}>{result.risk_label}</div>
          <div className="result-risk__confidence">
            Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
            <span className="result-risk__uncertainty">&nbsp;&bull;&nbsp;{result.uncertainty} confidence</span>
          </div>
        </div>
      </div>

      <div className="result-section">
        <h3 className="result-section__title">Risk Probability</h3>
        <div className="prob-bars">
          {probData.map(d => (
            <ProbBar key={d.name} label={d.name} value={d.value} color={d.fill} />
          ))}
        </div>
      </div>

      <div className="result-section">
        <h3 className="result-section__title">Decision Support Snapshot</h3>
        <div className="decision-grid">
          <div className="decision-card"><div className="decision-card__label">Primary recommendation</div><div className="decision-card__value">{decisionLevel}</div></div>
          <div className="decision-card"><div className="decision-card__label">Top probability</div><div className="decision-card__value">{topRisk.name} {topRisk.value}%</div></div>
          <div className="decision-card"><div className="decision-card__label">Probability margin</div><div className="decision-card__value">{margin}%</div></div>
        </div>
        <div className="prob-compare">
          <div className="prob-compare__bar"><div className="prob-compare__fill prob-compare__fill--top" style={{ width: `${topRisk.value}%`, background: topRisk.fill }} /></div>
          <div className="prob-compare__row"><span>{topRisk.name}</span><strong>{topRisk.value}%</strong></div>
          <div className="prob-compare__bar"><div className="prob-compare__fill prob-compare__fill--runner" style={{ width: `${runnerUp.value}%`, background: runnerUp.fill }} /></div>
          <div className="prob-compare__row"><span>{runnerUp.name}</span><strong>{runnerUp.value}%</strong></div>
        </div>
      </div>

      <div className="result-section">
        <h3 className="result-section__title">Fetal Reserve Score</h3>
        <div className="frs-row">
          <div className="frs-gauge">
            <div className="frs-gauge__value">{result.fetal_reserve_score}</div>
            <div className="frs-gauge__label">/ 100</div>
          </div>
          <div className="frs-info">
            <div className="frs-info__grade">{result.fetal_reserve_score >= 70 ? 'Good Reserve' : result.fetal_reserve_score >= 40 ? 'Reduced Reserve' : 'Low Reserve'}</div>
            <p className="frs-info__desc">{result.fetal_reserve_score >= 70 ? 'Reassuring pattern.' : result.fetal_reserve_score >= 40 ? 'Close monitoring recommended.' : 'Urgent clinical review advised.'}</p>
          </div>
        </div>
      </div>

      <div className="result-section">
        <h3 className="result-section__title">Key Findings</h3>
        <ul className="findings-list">
          {result.explanation.map((e, i) => (
            <li key={i} className="findings-list__item">
              <span className="findings-list__dot" style={{ background: result.risk_label === 'Pathological' ? '#dc2626' : result.risk_label === 'Suspect' ? '#d97706' : '#16a34a' }} />
              {e}
            </li>
          ))}
        </ul>
      </div>
    </>
  )
}

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="prob-bar">
      <div className="prob-bar__header">
        <span className="prob-bar__label">{label}</span>
        <span className="prob-bar__value" style={{ color }}>{value}%</span>
      </div>
      <div className="prob-bar__track">
        <div className="prob-bar__fill" style={{ width: `${value}%`, background: color }} />
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="result-empty">
      <div className="result-empty__icon">
        <svg viewBox="0 0 48 48" fill="none">
          <circle cx="24" cy="24" r="22" stroke="#e5e7eb" strokeWidth="2" fill="#f9fafb" />
          <path d="M14 24h5l3-8 4 16 3-8h5" stroke="#d1d5db" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      <h3 className="result-empty__title">No analysis yet</h3>
      <p className="result-empty__desc">Adjust the CTG parameters on the left and click <strong>Run ReserveNet Predictor</strong> to get a risk classification.</p>
      <p className="result-empty__hint">Use a preset to get started quickly.</p>
    </div>
  )
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="result-error">
      <svg viewBox="0 0 20 20" fill="currentColor" width="20" height="20">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z" clipRule="evenodd" />
      </svg>
      <div>
        <strong>Analysis failed</strong>
        <p>{message}</p>
      </div>
    </div>
  )
}
