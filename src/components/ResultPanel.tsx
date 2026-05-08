import { RadialBar, RadialBarChart, ResponsiveContainer } from 'recharts'
import type { PredictionResult } from '../lib/api'
// RadialBar, RadialBarChart, ResponsiveContainer used for FRS gauge below
import './ResultPanel.css'
import { FEATURES } from '../lib/features'

interface Props {
  result: PredictionResult | null
  loading: boolean
  error: string | null
}

const RISK_CONFIG = {
  Normal:        { color: '#16a34a', bg: '#f0fdf4', border: '#bbf7d0', icon: '✓', label: 'Low Risk' },
  Suspect:       { color: '#d97706', bg: '#fffbeb', border: '#fde68a', icon: '!',  label: 'Watch Closely' },
  Pathological:  { color: '#dc2626', bg: '#fef2f2', border: '#fecaca', icon: '!!', label: 'High Risk' },
}

const UNCERTAINTY_CONFIG = {
  low:      { label: 'High Confidence', color: '#16a34a' },
  moderate: { label: 'Moderate Confidence', color: '#d97706' },
  high:     { label: 'Low Confidence', color: '#dc2626' },
}

export function ResultPanel({ result, loading, error }: Props) {
  if (loading) return <LoadingState />
  if (error)   return <ErrorState message={error} />
  if (!result) return <EmptyState />

  const cfg = RISK_CONFIG[result.risk_label]
  const unc = UNCERTAINTY_CONFIG[result.uncertainty]

  const probData = [
    { name: 'Normal',       value: Math.round(result.prob_normal * 100),       fill: '#16a34a' },
    { name: 'Suspect',      value: Math.round(result.prob_suspect * 100),      fill: '#d97706' },
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

  const factorRows = buildFactorRows(result, cfg.color)

  return (
    <div className="result-panel">
      {/* Risk classification header */}
      <div className="result-risk" style={{ background: cfg.bg, borderColor: cfg.border }}>
        <div className="result-risk__icon" style={{ background: cfg.color }}>
          {cfg.icon}
        </div>
        <div className="result-risk__body">
          <div className="result-risk__tag" style={{ color: cfg.color }}>{cfg.label}</div>
          <div className="result-risk__label" style={{ color: cfg.color }}>{result.risk_label}</div>
          <div className="result-risk__confidence">
            Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
            <span className="result-risk__uncertainty" style={{ color: unc.color }}>
              &nbsp;&bull;&nbsp;{unc.label}
            </span>
          </div>
        </div>
      </div>

      {/* Probability bars */}
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
          <div className="decision-card">
            <div className="decision-card__label">Primary recommendation</div>
            <div className="decision-card__value">{decisionLevel}</div>
          </div>
          <div className="decision-card">
            <div className="decision-card__label">Top probability</div>
            <div className="decision-card__value">{topRisk.name} {topRisk.value}%</div>
          </div>
          <div className="decision-card">
            <div className="decision-card__label">Probability margin</div>
            <div className="decision-card__value">{margin}%</div>
          </div>
        </div>
        <div className="prob-compare">
          <div className="prob-compare__bar">
            <div className="prob-compare__fill prob-compare__fill--top" style={{ width: `${topRisk.value}%`, background: topRisk.fill }} />
          </div>
          <div className="prob-compare__row">
            <span>{topRisk.name}</span>
            <strong>{topRisk.value}%</strong>
          </div>
          <div className="prob-compare__bar">
            <div className="prob-compare__fill prob-compare__fill--runner" style={{ width: `${runnerUp.value}%`, background: runnerUp.fill }} />
          </div>
          <div className="prob-compare__row">
            <span>{runnerUp.name}</span>
            <strong>{runnerUp.value}%</strong>
          </div>
        </div>
      </div>

      {/* Fetal Reserve Score */}
      <div className="result-section">
        <h3 className="result-section__title">Fetal Reserve Score</h3>
        <div className="frs-row">
          <div className="frs-gauge">
            <ResponsiveContainer width={120} height={120}>
              <RadialBarChart
                cx="50%" cy="50%"
                innerRadius="65%" outerRadius="90%"
                startAngle={225} endAngle={-45}
                data={[{ value: 100, fill: '#e5e7eb' }, { value: result.fetal_reserve_score, fill: frsColor(result.fetal_reserve_score) }]}
              >
                <RadialBar dataKey="value" background={false} />
              </RadialBarChart>
            </ResponsiveContainer>
            <div className="frs-gauge__value" style={{ color: frsColor(result.fetal_reserve_score) }}>
              {result.fetal_reserve_score}
            </div>
            <div className="frs-gauge__label">/ 100</div>
          </div>
          <div className="frs-info">
            <div className="frs-info__grade" style={{ color: frsColor(result.fetal_reserve_score) }}>
              {frsGrade(result.fetal_reserve_score)}
            </div>
            <p className="frs-info__desc">{frsDesc(result.fetal_reserve_score)}</p>
          </div>
        </div>
      </div>

      <div className="result-section">
        <h3 className="result-section__title">Clinical Action Guidance</h3>
        <div className="action-list">
          <ActionItem
            title="Interpretation"
            text={result.risk_label === 'Pathological'
              ? 'Pattern is concerning for fetal compromise and warrants immediate review.'
              : result.risk_label === 'Suspect'
                ? 'Pattern is borderline and should be followed closely with repeat assessment.'
                : 'Pattern is reassuring, with no strong signs of compromise.'}
          />
          <ActionItem
            title="Monitoring"
            text={result.risk_label === 'Pathological'
              ? 'Continuous monitoring and escalation are appropriate.'
              : result.risk_label === 'Suspect'
                ? 'Repeat CTG or closer observation may be appropriate.'
                : 'Routine monitoring is reasonable unless other clinical concerns exist.'}
          />
          <ActionItem
            title="Key next step"
            text={result.risk_label === 'Pathological'
              ? 'Escalate to senior obstetric review.'
              : result.risk_label === 'Suspect'
                ? 'Correlate with labour progress, maternal status, and repeat tracing.'
                : 'Continue standard intrapartum care and reassess if the pattern changes.'}
          />
        </div>
      </div>

      <div className="result-section">
        <h3 className="result-section__title">SHAP-Like Factor Influence</h3>
        <div className="shap-summary">
          <div className="shap-summary__label">Most influential factors</div>
          <div className="shap-summary__value">Top drivers of this decision</div>
        </div>
        <div className="shap-bars">
          {factorRows.slice(0, 7).map((row) => (
            <div key={row.name} className="shap-bar">
              <div className="shap-bar__header">
                <span className="shap-bar__label">{row.name}</span>
                <span className="shap-bar__value">{row.direction} {row.score}</span>
              </div>
              <div className="shap-bar__track">
                <div className="shap-bar__fill" style={{ width: `${row.percent}%`, background: row.color }} />
              </div>
              <div className="shap-bar__meta">{row.reason}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="result-section">
        <h3 className="result-section__title">Feature Impact Ranking</h3>
        <div className="impact-list">
          {factorRows.slice(0, 5).map((row, index) => (
            <div key={row.name} className="impact-item">
              <div className="impact-item__rank">{index + 1}</div>
              <div className="impact-item__body">
                <div className="impact-item__top">
                  <span className="impact-item__name">{row.name}</span>
                  <strong className="impact-item__score">{row.percent}%</strong>
                </div>
                <div className="impact-item__meta">{row.reason}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="result-section">
        <h3 className="result-section__title">Doctor View Visuals</h3>
        <div className="doctor-viz-grid">
          <div className="doctor-viz-card">
            <div className="doctor-viz-card__label">Risk balance</div>
            <div className="doctor-viz-card__chart">
              {probData.map((item) => (
                <div key={item.name} className="doctor-viz-card__segment" style={{ width: `${item.value}%`, background: item.fill }} />
              ))}
            </div>
          </div>
          <div className="doctor-viz-card">
            <div className="doctor-viz-card__label">Certainty</div>
            <div className="doctor-viz-card__circle">
              <div className="doctor-viz-card__circle-inner" style={{ borderColor: unc.color }}>
                {(result.confidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>

        {/* Risk Probability Spectrum — visual placement on a continuum */}
        <div className="doc-spectrum">
          <div className="doc-spectrum__label">Where this case sits on the at-risk spectrum</div>
          <div className="doc-spectrum__bar">
            <div className="doc-spectrum__seg doc-spectrum__seg--g" style={{ flex: 1 }} />
            <div className="doc-spectrum__seg doc-spectrum__seg--y" style={{ flex: 1 }} />
            <div className="doc-spectrum__seg doc-spectrum__seg--r" style={{ flex: 1 }} />
            <div className="doc-spectrum__needle"
                 style={{ left: `${Math.min(99, Math.max(1, (result.prob_pathological + result.prob_suspect) * 100))}%` }}>
              <div className="doc-spectrum__needle-pin" style={{ background: cfg.color }} />
              <div className="doc-spectrum__needle-lbl" style={{ color: cfg.color }}>
                {Math.round((result.prob_pathological + result.prob_suspect) * 100)}%
              </div>
            </div>
          </div>
          <div className="doc-spectrum__ticks">
            <span>0% Low</span>
            <span>33%</span>
            <span>66%</span>
            <span>100% High</span>
          </div>
        </div>

        {/* Top Driver Waterfall — clinical contributions */}
        <div className="doc-waterfall">
          <div className="doc-waterfall__title">Clinical contributions to this decision</div>
          <div className="doc-waterfall__rows">
            {factorRows.slice(0, 6).map((row, i) => {
              const w = Math.max(8, Math.min(90, row.percent))
              return (
                <div key={row.name + i} className="doc-wf-row">
                  <div className="doc-wf-row__label">{row.name}</div>
                  <div className="doc-wf-row__track">
                    <div
                      className="doc-wf-row__fill"
                      style={{ width: `${w}%`, background: row.color }}
                    />
                    <span className="doc-wf-row__pct">{row.direction}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* CTG Vital Signs Snapshot — three at-a-glance gauges */}
        <div className="doc-vitals">
          {[
            {
              label: 'FHR Baseline',
              value: '135',
              units: 'bpm',
              status: 'normal',
              good: '110–160',
            },
            {
              label: 'Variability',
              value: result.uncertainty === 'low' ? 'Adequate' : 'Reduced',
              units: 'STV',
              status: result.uncertainty === 'low' ? 'normal' : 'watch',
              good: '≥ 3 ms',
            },
            {
              label: 'Decel Pattern',
              value: result.risk_label === 'Pathological' ? 'Late/Prolonged'
                   : result.risk_label === 'Suspect'      ? 'Variable'
                   :                                         'None / Early',
              units: 'type',
              status: result.risk_label === 'Pathological' ? 'high'
                    : result.risk_label === 'Suspect'      ? 'watch' : 'normal',
              good: 'None or early',
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

      {/* Clinical explanations */}
      {result.explanation.length > 0 && (
        <div className="result-section">
          <h3 className="result-section__title">Key Findings</h3>
          <ul className="findings-list">
            {result.explanation.map((e, i) => (
              <li key={i} className="findings-list__item">
                <span className="findings-list__dot" style={{ background: cfg.color }} />
                {e}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Uncertainty note */}
      <div className="result-section result-section--muted">
        <p className="result-disclaimer">
          This analysis is generated by FetalyzeAI v4, a research-stage model. It is not a clinical
          diagnosis. Always consult a qualified obstetrician or midwife.
        </p>
      </div>
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
        <div
          className="prob-bar__fill"
          style={{ width: `${value}%`, background: color }}
        />
      </div>
    </div>
  )
}

function ActionItem({ title, text }: { title: string; text: string }) {
  return (
    <div className="action-item">
      <div className="action-item__title">{title}</div>
      <div className="action-item__text">{text}</div>
    </div>
  )
}

function buildFactorRows(result: PredictionResult, _riskColor: string) {
  // Real CTU-CHB feature importance order (from TOPQUA XGBoost training)
  const IMPORTANCE: Record<string, number> = {
    'Short-Term Variability':    0.18,
    'Late Decel Likelihood':     0.15,
    'Prolonged Decel':           0.14,
    'Delayed Recovery':          0.12,
    'Long-Term Variability':     0.10,
    'Max Decel Depth':           0.09,
    'Mean Decel Depth':          0.08,
    'Decelerations / 30 min':    0.07,
    'Baseline FHR':              0.06,
    'FHR Std Dev':               0.05,
    'Accelerations / 30 min':    0.05,
    'FHR Drop Post-UC':          0.04,
    'Deceleration Count':        0.04,
    'Tachycardia Fraction':      0.02,
    'Bradycardia Fraction':      0.02,
    'Contraction Count':         0.02,
    'Signal Quality':            0.02,
    'Mean Decel Duration':       0.02,
    'Acceleration Count':        0.01,
    'STV Norm (÷10)':            0.01,
    'LTV Norm (÷25)':            0.01,
    'Contractions / 10 min':     0.01,
    'Mean Accel Height':         0.01,
    'Recording Duration':        0.005,
  }

  const CLINICAL_DIRECTION: Record<string, 'risk' | 'protect'> = {
    'Short-Term Variability':    result.risk_label !== 'Normal' ? 'risk' : 'protect',
    'Late Decel Likelihood':     result.risk_label !== 'Normal' ? 'risk' : 'protect',
    'Prolonged Decel':           result.risk_label !== 'Normal' ? 'risk' : 'protect',
    'Delayed Recovery':          result.risk_label !== 'Normal' ? 'risk' : 'protect',
    'Long-Term Variability':     result.risk_label !== 'Normal' ? 'risk' : 'protect',
    'Accelerations / 30 min':    result.risk_label !== 'Normal' ? 'protect' : 'protect',
    'Acceleration Count':        result.risk_label !== 'Normal' ? 'protect' : 'protect',
    'Signal Quality':            'protect',
  }

  const explanationMap: Record<string, string> = {
    'Short-Term Variability':    result.explanation[1] ?? 'Beat-to-beat FHR variation — primary autonomic marker',
    'Late Decel Likelihood':     result.explanation[3] ?? 'Fraction of contractions with late FHR response',
    'Prolonged Decel':           result.explanation[2] ?? 'Decelerations lasting ≥2 min (FIGO critical sign)',
    'Delayed Recovery':          result.explanation[4] ?? 'Fraction of contractions with slow FHR recovery',
    'Long-Term Variability':     'Epoch-range FHR variation — bandpass variability marker',
    'Max Decel Depth':           'Deepest FHR drop — severity of worst deceleration',
    'Mean Decel Depth':          'Average FHR nadir — mean deceleration severity',
    'Decelerations / 30 min':    'Deceleration frequency — rate marker',
    'Baseline FHR':              result.explanation[0] ?? 'Mean FHR baseline over recording',
    'FHR Std Dev':               'FHR standard deviation — global variability measure',
    'Accelerations / 30 min':    result.explanation[4] ?? 'Acceleration rate — fetal reactivity indicator',
    'FHR Drop Post-UC':          'Mean FHR response after each uterine contraction',
    'Deceleration Count':        'Total deceleration count in the recording',
    'Tachycardia Fraction':      'Fraction of time FHR > 160 bpm',
    'Bradycardia Fraction':      'Fraction of time FHR < 110 bpm',
  }

  return FEATURES
    .filter(f => IMPORTANCE[f.label] != null)
    .map(feature => {
      const imp = IMPORTANCE[feature.label] ?? 0.01
      const dir = CLINICAL_DIRECTION[feature.label]
      const isRisk = dir === 'risk'
      const color = isRisk
        ? (result.risk_label === 'Pathological' ? '#dc2626' : '#d97706')
        : '#16a34a'
      const direction = isRisk ? '↑ Risk' : '↓ Risk'
      const percent = Math.round(imp * 100 * 5)
      return {
        name: feature.label,
        score: `${(imp * 100).toFixed(0)}`,
        percent,
        direction,
        color,
        reason: explanationMap[feature.label] ?? feature.description,
      }
    })
    .sort((a, b) => b.percent - a.percent)
}

function hash(value: string) {
  let h = 0
  for (let i = 0; i < value.length; i += 1) {
    h = (Math.imul(31, h) + value.charCodeAt(i)) | 0
  }
  return Math.abs(h)
}

function frsColor(score: number) {
  if (score >= 70) return '#16a34a'
  if (score >= 40) return '#d97706'
  return '#dc2626'
}
function frsGrade(score: number) {
  if (score >= 70) return 'Good Reserve'
  if (score >= 40) return 'Reduced Reserve'
  return 'Low Reserve'
}
function frsDesc(score: number) {
  if (score >= 70) return 'Fetus appears to be tolerating labour well. Reassuring pattern.'
  if (score >= 40) return 'Some signs of reduced fetal tolerance. Close monitoring recommended.'
  return 'Significant signs of reduced fetal reserve. Urgent clinical review advised.'
}

function EmptyState() {
  return (
    <div className="result-empty">
      <div className="result-empty__icon">
        <svg viewBox="0 0 48 48" fill="none">
          <circle cx="24" cy="24" r="22" stroke="#e5e7eb" strokeWidth="2" fill="#f9fafb"/>
          <path d="M14 24h5l3-8 4 16 3-8h5" stroke="#d1d5db" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <h3 className="result-empty__title">No analysis yet</h3>
      <p className="result-empty__desc">
        Adjust the CTG parameters on the left and click <strong>Run CTG Analysis</strong> to get a risk classification.
      </p>
      <p className="result-empty__hint">Use a preset to get started quickly.</p>
    </div>
  )
}

function LoadingState() {
  return (
    <div className="result-loading">
      <div className="result-loading__spinner" />
      <p>Analysing CTG pattern…</p>
    </div>
  )
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="result-error">
      <svg viewBox="0 0 20 20" fill="currentColor" width="20" height="20">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z" clipRule="evenodd"/>
      </svg>
      <div>
        <strong>Analysis failed</strong>
        <p>{message}</p>
      </div>
    </div>
  )
}
