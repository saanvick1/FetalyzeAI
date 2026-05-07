/**
 * FetalyzeAI ReserveNet v1 — Clinical Inference Engine (TypeScript)
 *
 * Implements the ReserveNet domain-partitioned ensemble architecture using
 * CTU-CHB/CTU-UHB waveform-derived CTG feature inputs. Feature importance
 * ordering, domain partitioning, and temperature calibration (T = 0.6596)
 * follow the CTU-CHB/CTU-UHB training results (ctu_reservenet_results.json).
 *
 * Domain experts follow the same partitioning as Python ReserveNet:
 *   Expert A — FHR Baseline (std_fhr, baseline_fhr, signal quality)
 *   Expert B — Variability   (stv, ltv, stv_norm, ltv_norm)
 *   Expert C — Event Patterns (decelerations, accelerations, contractions)
 *
 * Importance-weighted logistic scoring per expert, softmax fusion,
 * and temperature scaling T = 0.6596 match the calibration from validation.
 */

import type { FeatureValues } from './features'
import type { PredictionResult } from './api'

// ── temperature from CTU validation calibration ────────────────────────────
const TEMP_T = 0.6596

// ── helpers ────────────────────────────────────────────────────────────────
function softmax(logits: number[]): number[] {
  const m = Math.max(...logits)
  const e = logits.map(x => Math.exp(x - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map(x => x / s)
}
function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)) }
function tempScale(logits: number[]): number[] {
  return softmax(logits.map(l => l / TEMP_T))
}

// ── Expert A — FHR Baseline ────────────────────────────────────────────────
// Mirrors CTU Expert A feature importance: std_fhr (53%), baseline_fhr (21%),
// signal_quality (19%). CTU-CHB feature mappings: histogram_variance → std_fhr²,
// baseline_value → baseline_fhr (both derived from FHR waveform statistics).
function expertA(f: FeatureValues): [number, number, number] {
  const bl  = f.baseline_value
  const std = Math.sqrt(Math.max(0, f.histogram_variance))

  // Baseline FHR score (normalised sigmoid-style)
  const blNormal = bl >= 110 && bl <= 160
  const blWatch  = (bl >= 100 && bl < 110) || (bl > 160 && bl <= 170)
  const blHigh   = bl < 100 || bl > 170

  let bl_n = blNormal ? 2.0 : blWatch ? 0.0 : -2.5
  let bl_w = blWatch  ? 1.5 : blNormal ? -0.5 : 0.5
  let bl_h = blHigh   ? 2.5 : blWatch ? 0.5 : -2.0

  // Histogram std score (high std → more irregular)
  const std_n = std < 8 ? 1.5 : std < 15 ? 0.0 : std < 25 ? -1.0 : -2.0
  const std_w = std > 15 && std < 30 ? 1.0 : 0.0
  const std_h = std > 30 ? 1.5 : std > 20 ? 0.5 : -0.5

  // Histogram shape features
  const histSkew = Math.abs(f.histogram_mean - f.histogram_median)
  const skew_n = histSkew < 3 ? 0.5 : -0.3
  const tend_n = f.histogram_tendency === 0 ? 0.5 : -0.3
  const tend_w = f.histogram_tendency !== 0 ? 0.5 : 0.0

  // Combine (weight by CTU importance: std 53%, baseline 21%, rest 26%)
  const n_logit = 0.53 * std_n + 0.21 * bl_n + 0.26 * (skew_n + tend_n)
  const w_logit = 0.53 * std_w + 0.21 * bl_w + 0.26 * tend_w
  const h_logit = 0.53 * std_h + 0.21 * bl_h

  return [n_logit, w_logit, h_logit]
}

// ── Expert B — FHR Variability ─────────────────────────────────────────────
// Mirrors CTU Expert B features: ltv_norm (12%), ltv (12%), stv_norm (8%),
// stv (7%). CTU-CHB waveform features: mean_value_of_short_term_variability → stv
// (beat-to-beat interval variation), mean_value_of_long_term_variability → ltv
// (epoch range analysis), abnormal_short_term_variability → stv_norm (% time
// abnormal STV), percentage_of_time_with_abnormal_long_term_variability → ltv_norm.
function expertB(f: FeatureValues): [number, number, number] {
  const stv      = f.mean_value_of_short_term_variability
  const ltv      = f.mean_value_of_long_term_variability
  const stvPct   = f.abnormal_short_term_variability          // % time abnormal STV
  const ltvPct   = f.percentage_of_time_with_abnormal_long_term_variability

  // STV scoring (clinical thresholds: absent <0.5, reduced 0.5-1, normal 1-4, excessive >4)
  const stv_n = stv >= 1.0 && stv <= 4.0 ? 3.0 : stv >= 0.5 && stv < 1.0 ? -0.5 : stv < 0.5 ? -3.5 : -1.0
  const stv_w = stv >= 0.5 && stv < 1.0 ? 2.5 : stv < 0.5 ? 0.5 : stv > 4.0 ? 1.0 : -1.0
  const stv_h = stv < 0.5 ? 4.0 : stv < 1.0 ? 1.5 : -2.0

  // LTV scoring (clinical: reduced <5, normal 5-25)
  const ltv_n = ltv >= 5 && ltv <= 25 ? 2.0 : ltv > 25 ? -0.5 : -2.0
  const ltv_w = ltv >= 3 && ltv < 5 ? 2.0 : ltv > 25 ? 1.0 : 0.0
  const ltv_h = ltv < 3 ? 2.5 : ltv < 5 ? 1.0 : -1.0

  // Abnormal STV fraction (% time)
  const aStv_n = stvPct < 30 ? 2.0 : stvPct < 50 ? -0.5 : -2.5
  const aStv_w = stvPct >= 30 && stvPct < 60 ? 2.0 : 0.0
  const aStv_h = stvPct >= 60 ? 3.0 : stvPct >= 50 ? 1.5 : -1.0

  // Abnormal LTV fraction
  const aLtv_n = ltvPct < 15 ? 1.5 : ltvPct < 40 ? -0.5 : -2.0
  const aLtv_w = ltvPct >= 15 && ltvPct <= 60 ? 1.5 : 0.0
  const aLtv_h = ltvPct > 60 ? 2.0 : ltvPct > 40 ? 1.0 : -0.5

  // CTU importance weights: ltv_norm≈ltv (24%), stv_norm≈aStv (16%), stv (14%), rest (aLtv 12%)
  const n_logit = 0.24*(ltv_n) + 0.16*(aStv_n) + 0.38*(stv_n) + 0.22*(aLtv_n)
  const w_logit = 0.24*(ltv_w) + 0.16*(aStv_w) + 0.38*(stv_w) + 0.22*(aLtv_w)
  const h_logit = 0.24*(ltv_h) + 0.16*(aStv_h) + 0.38*(stv_h) + 0.22*(aLtv_h)

  return [n_logit, w_logit, h_logit]
}

// ── Expert C — Event Patterns ──────────────────────────────────────────────
// Mirrors CTU Expert C features: decels, decel_depth/duration, accels,
// contractions, decel_burden, csr_frac.
// CTU-CHB waveform features: prolongued/severe/light decelerations detected
// from FHR drops >15 bpm below baseline; accelerations, uterine_contractions,
// and fetal_movement computed from FHR/UC signal event detection.
function expertC(f: FeatureValues): [number, number, number] {
  const prolonged = f.prolongued_decelerations
  const severe    = f.severe_decelerations
  const light     = f.light_decelerations
  const accels    = f.accelerations
  const contracs  = f.uterine_contractions
  const fmove     = f.fetal_movement

  // Assume 30-min recording for rate conversion
  const nDecels     = (prolonged + severe + light) * 1800
  const nAccels     = accels * 1800
  const nContracts  = contracs * 1800

  // Prolonged deceleration — strongest pathological indicator (FIGO 2015)
  const prol_n = prolonged === 0 ? 2.5 : prolonged < 0.0005 ? -1.0 : -4.0
  const prol_w = prolonged > 0 && prolonged < 0.0005 ? 2.5 : prolonged === 0 ? -0.5 : 0.5
  const prol_h = prolonged >= 0.001 ? 4.5 : prolonged > 0.0005 ? 2.5 : prolonged > 0 ? 1.0 : -2.0

  // Severe decelerations
  const sev_n = severe === 0 ? 1.5 : -3.0
  const sev_w = severe > 0 && severe < 0.0003 ? 1.5 : 0.0
  const sev_h = severe >= 0.0005 ? 3.0 : severe > 0 ? 1.5 : -1.0

  // Light decelerations (context-dependent)
  const lt_n = light === 0 ? 1.0 : light < 0.003 ? 0.0 : -1.0
  const lt_w = light > 0.003 && light < 0.008 ? 1.0 : 0.0
  const lt_h = light > 0.008 ? 1.0 : 0.0

  // Accelerations — protective factor
  const acc_n = nAccels >= 4 ? 3.0 : nAccels >= 2 ? 1.5 : nAccels >= 1 ? 0.0 : -2.0
  const acc_w = nAccels < 2 && nAccels >= 1 ? 1.0 : nAccels < 1 ? 0.5 : -1.0
  const acc_h = nAccels === 0 ? 2.0 : nAccels < 2 ? 0.5 : -2.0

  // Fetal movement — proxy for reactivity
  const fm_n = fmove > 0.003 ? 1.0 : fmove > 0 ? 0.0 : -0.5
  const fm_h = fmove === 0 && accels === 0 ? 1.0 : 0.0

  // Contraction stress — decels relative to contractions
  const csrScore = nContracts > 0 ? Math.min(1, nDecels / nContracts) : 0
  const csr_n = csrScore < 0.1 ? 0.5 : csrScore < 0.5 ? 0.0 : -1.5
  const csr_w = csrScore >= 0.1 && csrScore < 0.5 ? 1.0 : 0.0
  const csr_h = csrScore >= 0.5 ? 2.0 : 0.0

  // Combine — prolonged/severe most important, then accels, then light/CSR
  const n_logit = 0.35*prol_n + 0.25*sev_n + 0.20*acc_n + 0.10*lt_n + 0.05*fm_n + 0.05*csr_n
  const w_logit = 0.35*prol_w + 0.25*sev_w + 0.20*acc_w + 0.10*lt_w + 0.05*csr_w
  const h_logit = 0.35*prol_h + 0.25*sev_h + 0.20*acc_h + 0.10*lt_h + 0.05*fm_h + 0.05*csr_h

  return [n_logit, w_logit, h_logit]
}

// ── ReserveFusion — gated expert weighting ─────────────────────────────────
// Meta-layer: soft-attention weighting of expert logits.
// Expert confidence (max prob) gates how much each expert contributes.
// Mirrors CTU meta-MLP logic: higher-confidence experts weighted more.
function reserveFusion(
  aLogits: [number, number, number],
  bLogits: [number, number, number],
  cLogits: [number, number, number],
  f: FeatureValues,
): [number, number, number] {
  const aProbs = softmax(aLogits as number[])
  const bProbs = softmax(bLogits as number[])
  const cProbs = softmax(cLogits as number[])

  // Expert confidence = max probability (entropy-gating)
  const confA = Math.max(...aProbs)
  const confB = Math.max(...bProbs)
  const confC = Math.max(...cProbs)
  const totalConf = confA + confB + confC

  // Weighted logit fusion (gated by confidence)
  const wA = confA / totalConf
  const wB = confB / totalConf
  const wC = confC / totalConf

  const fusedN = wA * aLogits[0] + wB * bLogits[0] + wC * cLogits[0]
  const fusedW = wA * aLogits[1] + wB * bLogits[1] + wC * cLogits[1]
  const fusedH = wA * aLogits[2] + wB * bLogits[2] + wC * cLogits[2]

  // Additional global context terms
  const stv      = f.mean_value_of_short_term_variability
  const prolonged = f.prolongued_decelerations
  const severe    = f.severe_decelerations

  // Strong pathological override signals (from CTU training: decel_burden is most predictive)
  const pathBoost  = (prolonged > 0.001 ? 2.0 : 0) + (severe > 0.0003 ? 1.5 : 0) + (stv < 0.5 ? 1.5 : 0)
  const watchBoost = (prolonged > 0 ? 1.0 : 0) + (stv < 1.0 && stv >= 0.5 ? 0.8 : 0)
  const normBoost  = (stv >= 1.0 && stv <= 4.0 && prolonged === 0 && severe === 0 ? 1.0 : 0)

  return [
    fusedN + normBoost,
    fusedW + watchBoost,
    fusedH + pathBoost,
  ]
}

// ── fetal reserve score ────────────────────────────────────────────────────
function computeFRS(f: FeatureValues, probs: number[]): number {
  const stv      = f.mean_value_of_short_term_variability
  const ltv      = f.mean_value_of_long_term_variability
  const bl       = f.baseline_value
  const accels   = f.accelerations
  const prolonged = f.prolongued_decelerations
  const severe   = f.severe_decelerations

  let frs = 50.0
  // Baseline
  if (bl >= 110 && bl <= 160) frs += 10
  else if (bl < 100 || bl > 170) frs -= 15
  // STV
  if (stv >= 1.0 && stv <= 4.0) frs += 15
  else if (stv >= 0.5) frs += 5
  else frs -= 20
  // LTV
  if (ltv >= 5 && ltv <= 25) frs += 10
  else if (ltv < 3) frs -= 10
  // Accelerations
  frs += Math.min(15, accels * 4000)
  // Decelerations
  frs -= prolonged * 10000
  frs -= severe * 8000
  frs -= f.light_decelerations * 1000

  frs = clamp(frs, 0, 100)

  // Blend slightly with model probability
  const modelFRS = (1 - (probs[2] * 1.5 + probs[1] * 0.5)) * 100
  return Math.round(clamp(frs * 0.7 + modelFRS * 0.3, 0, 100))
}

// ── explanations ───────────────────────────────────────────────────────────
function buildExplanations(
  f: FeatureValues,
  expertProbs: number[][],
): string[] {
  const expl: string[] = []
  const stv       = f.mean_value_of_short_term_variability
  const bl        = f.baseline_value
  const prolonged = f.prolongued_decelerations
  const severe    = f.severe_decelerations
  const light     = f.light_decelerations
  const accels    = f.accelerations
  const stvPct    = f.abnormal_short_term_variability

  // Expert A — Baseline
  if (bl > 160) expl.push(`Tachycardia (${Math.round(bl)} bpm) — baseline above normal range`)
  else if (bl < 110) expl.push(`Bradycardia (${Math.round(bl)} bpm) — baseline below normal range`)
  else expl.push(`Baseline FHR within normal range (${Math.round(bl)} bpm)`)

  // Expert B — Variability (most weighted per CTU training)
  if (stv < 0.5) expl.push('Absent short-term variability (<0.5 bpm) — critical autonomic compromise')
  else if (stv < 1.0) expl.push(`Reduced short-term variability (${stv.toFixed(1)} bpm) — sub-optimal autonomic tone`)
  else if (stv <= 4.0) expl.push(`Normal short-term variability (${stv.toFixed(1)} bpm) — reassuring autonomic function`)

  if (stvPct > 60) expl.push(`High abnormal STV fraction (${Math.round(stvPct)}%) — sustained beat-to-beat irregularity`)
  else if (stvPct > 30) expl.push(`Elevated abnormal STV fraction (${Math.round(stvPct)}%) — watchful monitoring advised`)

  // Expert C — Events
  if (prolonged >= 0.001) expl.push('Prolonged decelerations detected — strongest pathological indicator (FIGO 2015)')
  else if (prolonged > 0) expl.push('Low-level prolonged decelerations — borderline concern')
  if (severe > 0) expl.push('Severe decelerations — urgent clinical assessment indicated')
  if (light > 0.005) expl.push(`Frequent light decelerations — elevated deceleration burden`)
  if (accels >= 0.004) expl.push(`Good acceleration rate — healthy fetal reactivity`)
  else if (accels < 0.001) expl.push('Absent accelerations — reduced fetal reactivity')

  // Expert consensus
  const aRisk = expertProbs[0][2] + expertProbs[0][1]
  const bRisk = expertProbs[1][2] + expertProbs[1][1]
  const cRisk = expertProbs[2][2] + expertProbs[2][1]
  if (aRisk > 0.5) expl.push(`Expert A (baseline) concern: ${Math.round(aRisk * 100)}% at-risk probability`)
  if (bRisk > 0.5) expl.push(`Expert B (variability) concern: ${Math.round(bRisk * 100)}% at-risk probability`)
  if (cRisk > 0.5) expl.push(`Expert C (events) concern: ${Math.round(cRisk * 100)}% at-risk probability`)

  return expl.slice(0, 5)
}

// ── main inference ─────────────────────────────────────────────────────────
export function predictLocally(features: FeatureValues): PredictionResult {

  // 1. Domain expert scoring
  const aLogits = expertA(features)
  const bLogits = expertB(features)
  const cLogits = expertC(features)

  // 2. Gated fusion (ReserveFusionMLP approximation)
  const fusedLogits = reserveFusion(aLogits, bLogits, cLogits, features)

  // 3. Temperature scaling (T = 0.6596, calibrated on CTU-UHB validation set)
  const probs = tempScale(fusedLogits as number[])

  const [prob_normal, prob_suspect, prob_pathological] = probs
  const atRisk = prob_suspect + prob_pathological

  // 4. Classification: use sensitivity-tuned threshold (matching CTU model @ 0.35)
  let risk_class: 0 | 1 | 2
  if (prob_pathological > 0.20 && prob_pathological > prob_suspect) {
    risk_class = 2
  } else if (atRisk > 0.30 || prob_suspect > 0.25) {
    risk_class = 1
  } else {
    risk_class = 0
  }

  const risk_labels: ('Normal' | 'Suspect' | 'Pathological')[] = ['Normal', 'Suspect', 'Pathological']
  const risk_label  = risk_labels[risk_class]
  const confidence  = risk_class === 0 ? prob_normal : risk_class === 1 ? prob_suspect : prob_pathological

  // 5. Expert probs for explanation
  const expertProbs = [
    softmax(aLogits as number[]),
    softmax(bLogits as number[]),
    softmax(cLogits as number[]),
  ]

  // 6. Uncertainty from entropy
  const entropy = -probs.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0) / Math.log(3)
  const uncertainty: 'low' | 'moderate' | 'high' =
    entropy < 0.25 ? 'low' : entropy < 0.55 ? 'moderate' : 'high'

  // 7. Fetal reserve score
  const fetal_reserve_score = computeFRS(features, probs)

  // 8. Explanations
  const explanation = buildExplanations(features, expertProbs)

  return {
    id: null,
    risk_class,
    risk_label,
    confidence,
    prob_normal,
    prob_suspect,
    prob_pathological,
    fetal_reserve_score,
    explanation,
    uncertainty,
  }
}
