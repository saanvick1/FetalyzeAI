import type { FeatureValues } from './features'
import type { PredictionResult } from './api'

export function predictLocally(features: FeatureValues): PredictionResult {
  const f = features as Record<string, number>

  let score_normal = 0
  let score_suspect = 0
  let score_pathological = 0
  const explanations: string[] = []

  if (f.prolongued_decelerations > 0.001) {
    score_pathological += 4.0
    explanations.push('Prolonged decelerations present — strong pathological indicator')
  }
  if (f.abnormal_short_term_variability > 65) {
    score_pathological += 2.5
    explanations.push('Very high abnormal short-term variability — reduced fetal reserve')
  }
  if (f.mean_value_of_short_term_variability < 0.5) {
    score_pathological += 3.0
    explanations.push('Critically low short-term variability (< 0.5 bpm) — compromised autonomic response')
  }
  if (f.severe_decelerations > 0.0005) {
    score_pathological += 3.5
    explanations.push('Severe decelerations detected — urgent clinical review indicated')
  }
  if (f.percentage_of_time_with_abnormal_long_term_variability > 60) {
    score_pathological += 2.0
    explanations.push('Majority of time with abnormal long-term variability')
  }
  if (f.histogram_variance > 120) {
    score_pathological += 1.5
    explanations.push('High histogram variance — irregular heart rate pattern')
  }
  if (f.light_decelerations > 0.008) {
    score_pathological += 1.5
    score_suspect += 0.5
  }

  if (f.abnormal_short_term_variability > 45 && f.abnormal_short_term_variability <= 65) {
    score_suspect += 2.0
    explanations.push('Elevated abnormal short-term variability — watchful monitoring advised')
  }
  if (f.mean_value_of_short_term_variability >= 0.5 && f.mean_value_of_short_term_variability < 1.0) {
    score_suspect += 1.5
    explanations.push('Reduced short-term variability — sub-optimal fetal heart rate pattern')
  }
  if (
    f.percentage_of_time_with_abnormal_long_term_variability > 30 &&
    f.percentage_of_time_with_abnormal_long_term_variability <= 60
  ) {
    score_suspect += 1.5
    explanations.push('Elevated abnormal long-term variability fraction')
  }
  if (f.accelerations < 0.001 && f.uterine_contractions > 0.004) {
    score_suspect += 1.5
    explanations.push('Low accelerations with active contractions — reduced reactivity')
  }
  if (f.prolongued_decelerations > 0 && f.prolongued_decelerations <= 0.001) {
    score_suspect += 2.0
    explanations.push('Low-level prolonged decelerations — borderline concern')
  }
  if (f.histogram_variance > 50 && f.histogram_variance <= 120) {
    score_suspect += 1.0
  }
  if (f.mean_value_of_long_term_variability > 25) {
    score_suspect += 1.0
    explanations.push('Elevated long-term variability mean — possible fetal stress response')
  }

  if (f.accelerations > 0.004) {
    score_normal += 3.0
    explanations.push('Good acceleration rate — healthy fetal reactivity')
  }
  if (f.mean_value_of_short_term_variability >= 1.0 && f.mean_value_of_short_term_variability <= 4.0) {
    score_normal += 2.5
    explanations.push('Normal short-term variability — reassuring autonomic function')
  }
  if (f.abnormal_short_term_variability < 30) {
    score_normal += 2.0
    explanations.push('Low abnormal short-term variability — stable beat-to-beat rhythm')
  }
  if (f.prolongued_decelerations === 0 && f.severe_decelerations === 0) {
    score_normal += 1.5
    explanations.push('No severe or prolonged decelerations — low immediate risk')
  }
  if (f.histogram_variance < 25) {
    score_normal += 1.0
  }
  if (f.percentage_of_time_with_abnormal_long_term_variability < 15) {
    score_normal += 1.5
    explanations.push('Predominantly normal long-term variability — good fetal reserve')
  }
  if (f.baseline_value >= 110 && f.baseline_value <= 160) {
    score_normal += 1.0
    explanations.push('Baseline FHR within normal range (110–160 bpm)')
  }

  const maxScore = Math.max(score_normal, score_suspect, score_pathological)
  const expN = Math.exp(score_normal - maxScore)
  const expS = Math.exp(score_suspect - maxScore)
  const expP = Math.exp(score_pathological - maxScore)
  const total = expN + expS + expP

  const prob_normal = expN / total
  const prob_suspect = expS / total
  const prob_pathological = expP / total

  const risk_class =
    prob_pathological > prob_suspect
      ? prob_pathological > prob_normal
        ? 2
        : 0
      : prob_suspect > prob_normal
        ? 1
        : 0

  const risk_labels: ('Normal' | 'Suspect' | 'Pathological')[] = ['Normal', 'Suspect', 'Pathological']
  const risk_label = risk_labels[risk_class]
  const probs = [prob_normal, prob_suspect, prob_pathological]
  const confidence = probs[risk_class]

  let frs = 50
  frs += Math.min(20, f.accelerations * 5000)
  frs += f.mean_value_of_short_term_variability >= 1.0 && f.mean_value_of_short_term_variability <= 4.0 ? 15 : 0
  frs += f.percentage_of_time_with_abnormal_long_term_variability < 15 ? 10 : 0
  frs -= Math.min(25, f.abnormal_short_term_variability * 0.4)
  frs -= f.prolongued_decelerations * 8000
  frs -= f.severe_decelerations * 5000
  frs -= Math.min(15, f.percentage_of_time_with_abnormal_long_term_variability * 0.2)
  frs = Math.max(0, Math.min(100, frs))

  const entropy = -(
    (prob_normal > 0 ? prob_normal * Math.log(prob_normal) : 0) +
    (prob_suspect > 0 ? prob_suspect * Math.log(prob_suspect) : 0) +
    (prob_pathological > 0 ? prob_pathological * Math.log(prob_pathological) : 0)
  ) / Math.log(3)

  const uncertainty: 'low' | 'moderate' | 'high' = entropy < 0.2 ? 'low' : entropy < 0.5 ? 'moderate' : 'high'

  return {
    id: null,
    risk_class: risk_class as 0 | 1 | 2,
    risk_label,
    confidence,
    prob_normal,
    prob_suspect,
    prob_pathological,
    fetal_reserve_score: Math.round(frs),
    explanation: explanations.slice(0, 5),
    uncertainty,
  }
}
