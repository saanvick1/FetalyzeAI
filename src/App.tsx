import { useState, useCallback, useEffect } from 'react'
import { Header } from './components/Header'
import { PredictionForm } from './components/PredictionForm'
import { ResultPanel } from './components/ResultPanel'
import { HistoryPanel } from './components/HistoryPanel'
import { DisclaimerBanner } from './components/DisclaimerBanner'
import { ResearchPanel } from './components/ResearchPanel'
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
