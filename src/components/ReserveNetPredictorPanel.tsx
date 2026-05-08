import { useMemo, useState } from 'react'
import { predictLocally } from '../lib/predictor'
import { DEFAULT_VALUES, FEATURE_GROUPS, FEATURES, PRESETS, type FeatureKey, type FeatureValues } from '../lib/features'
import type { PredictionResult } from '../lib/api'
import './ResultPanel.css'

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

      <DoctorGraphs result={result} />
    </>
  )
}

/* ───────────────────────────────────────────────────────────────────────────
   Doctor-facing graphs for the ReserveNet Predictor view.
   - Radial probability donut (Normal / Suspect / Pathological)
   - Risk-spectrum gauge with needle
   - Confidence vs Reserve "twin-dial" panel
   - Probability waterfall against population baseline
   - Three vital-signs cards (FHR baseline / variability / decel pattern)
   ─────────────────────────────────────────────────────────────────────── */
function DoctorGraphs({ result }: { result: PredictionResult }) {
  const accent =
    result.risk_label === 'Pathological' ? '#dc2626'
    : result.risk_label === 'Suspect'    ? '#d97706'
    :                                       '#16a34a'

  const atRiskPct = Math.round(
    Math.min(100, Math.max(0, (result.prob_pathological + result.prob_suspect) * 100))
  )

  // CTU-CHB population priors (552 records, ~19% at-risk)
  const populationAtRisk = 19
  const delta = atRiskPct - populationAtRisk

  // Radial donut (SVG) — circumference based on three slices
  const slices = [
    { label: 'Normal',       pct: Math.round(result.prob_normal * 100),       color: '#16a34a' },
    { label: 'Suspect',      pct: Math.round(result.prob_suspect * 100),      color: '#d97706' },
    { label: 'Pathological', pct: Math.round(result.prob_pathological * 100), color: '#dc2626' },
  ]
  const r = 64
  const C = 2 * Math.PI * r
  let acc = 0
  const arcs = slices.map(s => {
    const len = (s.pct / 100) * C
    const offset = -acc
    acc += len
    return { ...s, len, offset }
  })

  return (
    <div className="result-section">
      <h3 className="result-section__title">Clinical Visual Snapshot</h3>

      {/* Radial probability donut + twin dials */}
      <div className="doc-radial-row">
        <div className="doc-donut">
          <svg viewBox="0 0 160 160" width="170" height="170">
            <circle cx="80" cy="80" r={r} fill="none" stroke="#f3f4f6" strokeWidth="20" />
            {arcs.map(a => (
              <circle
                key={a.label}
                cx="80" cy="80" r={r}
                fill="none" stroke={a.color} strokeWidth="20"
                strokeDasharray={`${a.len} ${C - a.len}`}
                strokeDashoffset={a.offset}
                transform="rotate(-90 80 80)"
                strokeLinecap="butt"
              />
            ))}
            <text x="80" y="76" textAnchor="middle" fontSize="22" fontWeight="700" fill={accent}>
              {atRiskPct}%
            </text>
            <text x="80" y="94" textAnchor="middle" fontSize="9" fill="#6b7280">
              AT-RISK PROB
            </text>
          </svg>
          <div className="doc-donut__legend">
            {slices.map(s => (
              <div key={s.label} className="doc-donut__lg-row">
                <span className="doc-donut__sw" style={{ background: s.color }} />
                <span className="doc-donut__lab">{s.label}</span>
                <span className="doc-donut__val">{s.pct}%</span>
              </div>
            ))}
          </div>
        </div>

        <div className="doc-twindial">
          <DialChart
            value={Math.round(result.confidence * 100)}
            label="Model Confidence"
            color={result.uncertainty === 'low' ? '#16a34a' : result.uncertainty === 'medium' ? '#d97706' : '#dc2626'}
            footer={`${result.uncertainty.toUpperCase()} uncertainty`}
          />
          <DialChart
            value={result.fetal_reserve_score}
            label="Fetal Reserve"
            color={result.fetal_reserve_score >= 70 ? '#16a34a' : result.fetal_reserve_score >= 40 ? '#d97706' : '#dc2626'}
            footer={result.fetal_reserve_score >= 70 ? 'Good reserve' : result.fetal_reserve_score >= 40 ? 'Reduced reserve' : 'Low reserve'}
          />
        </div>
      </div>

      {/* Risk-spectrum gauge */}
      <div className="doc-spectrum">
        <div className="doc-spectrum__label">Where this case sits on the at-risk spectrum</div>
        <div className="doc-spectrum__bar">
          <div className="doc-spectrum__seg doc-spectrum__seg--g" style={{ flex: 1 }} />
          <div className="doc-spectrum__seg doc-spectrum__seg--y" style={{ flex: 1 }} />
          <div className="doc-spectrum__seg doc-spectrum__seg--r" style={{ flex: 1 }} />
          <div className="doc-spectrum__needle" style={{ left: `${Math.min(99, Math.max(1, atRiskPct))}%` }}>
            <div className="doc-spectrum__needle-pin" style={{ background: accent }} />
            <div className="doc-spectrum__needle-lbl" style={{ color: accent }}>{atRiskPct}%</div>
          </div>
        </div>
        <div className="doc-spectrum__ticks">
          <span>0% Low</span><span>33%</span><span>66%</span><span>100% High</span>
        </div>
      </div>

      {/* Population-baseline waterfall */}
      <div className="doc-waterfall">
        <div className="doc-waterfall__title">Comparison with CTU-CHB population baseline</div>
        <div className="doc-waterfall__rows">
          <div className="doc-wf-row">
            <div className="doc-wf-row__label">Population at-risk</div>
            <div className="doc-wf-row__track">
              <div className="doc-wf-row__fill" style={{ width: `${populationAtRisk}%`, background: '#9ca3af' }} />
              <span className="doc-wf-row__pct">{populationAtRisk}%</span>
            </div>
          </div>
          <div className="doc-wf-row">
            <div className="doc-wf-row__label">This case at-risk</div>
            <div className="doc-wf-row__track">
              <div className="doc-wf-row__fill" style={{ width: `${atRiskPct}%`, background: accent }} />
              <span className="doc-wf-row__pct">{atRiskPct}%</span>
            </div>
          </div>
          <div className="doc-wf-row">
            <div className="doc-wf-row__label">Δ vs population</div>
            <div className="doc-wf-row__track">
              <div
                className="doc-wf-row__fill"
                style={{
                  width: `${Math.min(90, Math.abs(delta))}%`,
                  background: delta >= 0 ? '#dc2626' : '#16a34a',
                }}
              />
              <span className="doc-wf-row__pct">{delta >= 0 ? '+' : ''}{delta} pts</span>
            </div>
          </div>
        </div>
      </div>

      {/* Vital-signs cards */}
      <div className="doc-vitals">
        {[
          {
            label: 'At-Risk Probability',
            value: `${atRiskPct}`,
            units: '%',
            status: atRiskPct >= 50 ? 'high' : atRiskPct >= 25 ? 'watch' : 'normal',
            good: '< 20%',
          },
          {
            label: 'Confidence',
            value: `${(result.confidence * 100).toFixed(0)}`,
            units: '%',
            status: result.uncertainty === 'low' ? 'normal' : result.uncertainty === 'medium' ? 'watch' : 'high',
            good: '≥ 70%',
          },
          {
            label: 'Decision',
            value:
              result.risk_label === 'Pathological' ? 'Urgent'
              : result.risk_label === 'Suspect'    ? 'Monitor'
              :                                       'Routine',
            units: '',
            status:
              result.risk_label === 'Pathological' ? 'high'
              : result.risk_label === 'Suspect'    ? 'watch'
              :                                       'normal',
            good: 'Routine',
          },
        ].map(v => (
          <div key={v.label} className={`doc-vital doc-vital--${v.status}`}>
            <div className="doc-vital__lab">{v.label}</div>
            <div className="doc-vital__val">{v.value} <span>{v.units}</span></div>
            <div className="doc-vital__good">Reassuring: {v.good}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function DialChart({ value, label, color, footer }: { value: number; label: string; color: string; footer: string }) {
  // Half-circle gauge from 0 to 100
  const v = Math.max(0, Math.min(100, value))
  const angle = -90 + (v / 100) * 180  // -90 ... +90 degrees
  const r = 50
  const cx = 60, cy = 60
  const rad = (angle * Math.PI) / 180
  const nx = cx + r * Math.cos(rad)
  const ny = cy + r * Math.sin(rad)
  // Track arc as path
  const arc = (frac: number, stroke: string) => {
    const a = -180 + frac * 180
    const ax = cx + r * Math.cos((a * Math.PI) / 180)
    const ay = cy + r * Math.sin((a * Math.PI) / 180)
    return <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${ax} ${ay}`} stroke={stroke} strokeWidth="10" fill="none" strokeLinecap="round" />
  }
  return (
    <div className="doc-dial">
      <svg viewBox="0 0 120 75" width="135" height="84">
        {arc(1, '#f3f4f6')}
        {arc(v / 100, color)}
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={color} strokeWidth="2.5" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="4" fill={color} />
        <text x="60" y="58" textAnchor="middle" fontSize="16" fontWeight="700" fill={color}>{Math.round(v)}</text>
      </svg>
      <div className="doc-dial__lab">{label}</div>
      <div className="doc-dial__foot" style={{ color }}>{footer}</div>
    </div>
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
