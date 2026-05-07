import { EDGE_URL, EDGE_HEADERS } from './supabase'
import { predictLocally } from './predictor'
import type { FeatureValues } from './features'

export interface PredictionResult {
  id: string | null
  risk_class: 0 | 1 | 2
  risk_label: 'Normal' | 'Suspect' | 'Pathological'
  confidence: number
  prob_normal: number
  prob_suspect: number
  prob_pathological: number
  fetal_reserve_score: number
  explanation: string[]
  uncertainty: 'low' | 'moderate' | 'high'
}

export async function predict(
  features: FeatureValues,
  sessionId: string
): Promise<PredictionResult> {
  if (!EDGE_URL || !EDGE_HEADERS) {
    return predictLocally(features)
  }

  const res = await fetch(EDGE_URL, {
    method: 'POST',
    headers: EDGE_HEADERS,
    body: JSON.stringify({ features, session_id: sessionId }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error ?? 'Prediction failed')
  }
  return res.json()
}
