import { useState } from 'react'
import { FEATURES, FEATURE_GROUPS, DEFAULT_VALUES, PRESETS, type FeatureValues } from '../lib/features'
import './PredictionForm.css'

interface Props {
  values: FeatureValues
  onChange: (v: FeatureValues) => void
  onSubmit: () => void
  loading: boolean
}

export function PredictionForm({ values, onChange, onSubmit, loading }: Props) {
  const [activeGroup, setActiveGroup] = useState<string>(FEATURE_GROUPS[0])

  const groupFeatures = FEATURES.filter(f => f.group === activeGroup)

  function handleChange(key: string, raw: string) {
    const v = parseFloat(raw)
    onChange({ ...values, [key]: isNaN(v) ? 0 : v })
  }

  function applyPreset(preset: typeof PRESETS[0]) {
    onChange({ ...DEFAULT_VALUES, ...preset.values })
  }

  function resetToDefaults() {
    onChange({ ...DEFAULT_VALUES })
  }

  return (
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
            feature={feat}
            value={values[feat.key]}
            onChange={handleChange}
          />
        ))}
      </div>

      <div className="form-card__footer">
        <div className="form-card__count">
          {FEATURES.length} features across {FEATURE_GROUPS.length} groups
        </div>
        <button
          className="submit-btn"
          onClick={onSubmit}
          disabled={loading}
        >
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
              Run CTG Analysis
            </>
          )}
        </button>
      </div>
    </div>
  )
}

interface FeatureInputProps {
  feature: typeof FEATURES[0]
  value: number
  onChange: (key: string, val: string) => void
}

function FeatureInput({ feature, value, onChange }: FeatureInputProps) {
  const importanceClass = `feature-input__importance--${feature.importance}`
  const displayVal = feature.step < 0.01
    ? value.toFixed(4)
    : feature.step < 0.1
    ? value.toFixed(3)
    : feature.step < 1
    ? value.toFixed(1)
    : String(Math.round(value))

  const pct = ((value - feature.min) / (feature.max - feature.min)) * 100

  return (
    <div className="feature-input">
      <div className="feature-input__header">
        <div className="feature-input__title">
          <span className={`feature-input__importance ${importanceClass}`} title={`${feature.importance} importance`} />
          <label className="feature-input__label" htmlFor={feature.key}>
            {feature.label}
          </label>
          {feature.unit && (
            <span className="feature-input__unit">{feature.unit}</span>
          )}
        </div>
        <span className="feature-input__value">{displayVal}</span>
      </div>
      <input
        id={feature.key}
        type="range"
        min={feature.min}
        max={feature.max}
        step={feature.step}
        value={value}
        onChange={e => onChange(feature.key, e.target.value)}
        className="feature-input__slider"
        style={{ '--pct': `${Math.max(0, Math.min(100, pct))}%` } as React.CSSProperties}
      />
      <div className="feature-input__range">
        <span>{feature.min}</span>
        <span className="feature-input__desc">{feature.description}</span>
        <span>{feature.max}</span>
      </div>
      <input
        type="number"
        min={feature.min}
        max={feature.max}
        step={feature.step}
        value={value}
        onChange={e => onChange(feature.key, e.target.value)}
        className="feature-input__number"
      />
    </div>
  )
}
