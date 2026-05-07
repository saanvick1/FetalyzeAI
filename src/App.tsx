import { useState, useCallback, useEffect } from 'react'
import { Header } from './components/Header'
import { PredictionForm } from './components/PredictionForm'
import { ResultPanel } from './components/ResultPanel'
import { HistoryPanel } from './components/HistoryPanel'
import { DisclaimerBanner } from './components/DisclaimerBanner'
import { predict, type PredictionResult } from './lib/api'
import { DEFAULT_VALUES, type FeatureValues } from './lib/features'
import { supabase } from './lib/supabase'
import './App.css'

function getSessionId() {
  let id = sessionStorage.getItem('fetalyze_session')
  if (!id) {
    id = crypto.randomUUID()
    sessionStorage.setItem('fetalyze_session', id)
  }
  return id
}

export interface HistoryEntry extends PredictionResult {
  timestamp: Date
  features: FeatureValues
}

export default function App() {
  const [values, setValues] = useState<FeatureValues>(DEFAULT_VALUES)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [activeTab, setActiveTab] = useState<'predict' | 'history' | 'research'>('predict')
  const sessionId = getSessionId()

  const handlePredict = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await predict(values, sessionId)
      setResult(res)
      setHistory(prev => [{ ...res, timestamp: new Date(), features: { ...values } }, ...prev].slice(0, 20))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [values, sessionId])

  useEffect(() => {
    if (!supabase) return
    supabase
      .from('predictions')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(10)
      .then(({ data }) => {
        if (data && data.length > 0) {
          const entries: HistoryEntry[] = data.map((row) => ({
            id: row.id,
            risk_class: row.risk_class,
            risk_label: row.risk_label,
            confidence: row.confidence,
            prob_normal: row.prob_normal,
            prob_suspect: row.prob_suspect,
            prob_pathological: row.prob_pathological,
            fetal_reserve_score: row.fetal_reserve_score ?? 0,
            explanation: [],
            uncertainty: 'moderate' as const,
            timestamp: new Date(row.created_at),
            features: row.features as FeatureValues,
          }))
          setHistory(entries)
        }
      })
  }, [])

  return (
    <div className="app">
      <Header />
      <DisclaimerBanner />

      <div className="app__tabs">
        <button
          className={`app__tab ${activeTab === 'predict' ? 'app__tab--active' : ''}`}
          onClick={() => setActiveTab('predict')}
        >
          CTG Analysis
        </button>
        <button
          className={`app__tab ${activeTab === 'history' ? 'app__tab--active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          History
          {history.length > 0 && <span className="app__tab-badge">{history.length}</span>}
        </button>
        <button
          className={`app__tab ${activeTab === 'research' ? 'app__tab--active' : ''}`}
          onClick={() => setActiveTab('research')}
        >
          Model & Evidence
        </button>
      </div>

      <main className="app__main">
        {activeTab === 'predict' ? (
          <div className="app__layout">
            <PredictionForm
              values={values}
              onChange={setValues}
              onSubmit={handlePredict}
              loading={loading}
            />
            <ResultPanel result={result} loading={loading} error={error} />
          </div>
        ) : (
          activeTab === 'history' ? (
            <HistoryPanel
              history={history}
              onSelect={(entry) => {
                setValues(entry.features)
                setResult(entry)
                setActiveTab('predict')
              }}
            />
          ) : (
            <ResearchPanel />
          )
        )}
      </main>
    </div>
  )
}

function ResearchPanel() {
  return (
    <div className="research-panel">
      <section className="research-hero">
        <div>
          <h2>FetalyzeAI — uncertainty-aware CTG second reader</h2>
          <p>
            Research-stage decision support for fetal monitoring, combining risk stratification,
            uncertainty estimation, factor influence ranking, and dataset-aware evaluation.
          </p>
        </div>
        <div className="research-hero__card">
          <strong>Clinical safety notice</strong>
          <span>Not a diagnosis. Not a treatment recommendation. Requires clinician review.</span>
        </div>
      </section>

      <div className="research-grid">
        <section className="research-card">
          <h3>How it works</h3>
          <ol>
            <li>CTG feature input and signal quality check</li>
            <li>Risk prediction with uncertainty and reserve scoring</li>
            <li>SHAP-style factor ranking and explanation</li>
            <li>Clinician review with escalation guidance</li>
          </ol>
        </section>

        <section className="research-card">
          <h3>Core model outputs</h3>
          <ul>
            <li>Risk class: Normal / Suspect / Pathological</li>
            <li>Confidence and entropy-based uncertainty</li>
            <li>Fetal reserve score</li>
            <li>Top factor influence ranking</li>
            <li>Action guidance for bedside review</li>
          </ul>
        </section>

        <section className="research-card">
          <h3>Model evaluation focus</h3>
          <ul>
            <li>High-risk recall and false negative rate</li>
            <li>Macro-F1 and balanced accuracy</li>
            <li>AUROC and AUPRC</li>
            <li>Calibration, ECE, and Brier score</li>
            <li>Cross-dataset generalization</li>
            <li>Fairness across risk classes</li>
          </ul>
        </section>

        <section className="research-card">
          <h3>Datasets used</h3>
          <ul>
            <li>CTGDL waveform development data</li>
            <li>CTU-UHB / CTU-CHB clinical benchmark</li>
          </ul>
        </section>

        <section className="research-card research-card--wide">
          <h3>Evaluation metrics for fairness</h3>
          <div className="metric-grid">
            <Metric label="Sensitivity" value="High-risk cases detected" />
            <Metric label="Specificity" value="Low false alarms" />
            <Metric label="Macro-F1" value="Balanced across all classes" />
            <Metric label="Balanced Accuracy" value="Handles class imbalance" />
            <Metric label="Calibration" value="Probability reliability" />
            <Metric label="Subgroup parity" value="Consistent performance" />
          </div>
        </section>

        <section className="research-card research-card--wide">
          <h3>Clinical safety and roadmap</h3>
          <p>
            FetalyzeAI is designed to support review of borderline patterns, not to replace bedside
            judgement. The next step is clinician-annotated validation on external CTG cohorts and
            prospective silent testing.
          </p>
        </section>
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-pill">
      <strong>{label}</strong>
      <span>{value}</span>
    </div>
  )
}
