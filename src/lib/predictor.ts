/**
 * FetalyzeAI TOPQUA — Clinical Inference Engine (TypeScript)
 *
 * Implements the TOPQUA domain-partitioned architecture using CTU-CHB/CTU-UHB
 * waveform-derived feature inputs. Mirrors the Python training pipeline's
 * expert partitioning and clinical decision logic.
 *
 * Expert A — FHR Baseline (baseline_fhr, std_fhr, tachycardia_frac, bradycardia_frac)
 * Expert B — Variability   (stv, ltv, stv_norm, ltv_norm)
 * Expert C — Event Patterns (decelerations, accelerations, late-decel, contraction stress)
 * Meta — Gated fusion + temperature scaling (T = 0.72, calibrated on CTU validation)
 */

import type { FeatureValues } from './features'
import type { PredictionResult } from './api'

const TEMP_T = 0.72

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
// Clinical ranges: normal 110–160 bpm, tachy >160, brady <110
function expertA(f: FeatureValues): [number, number, number] {
  const bl   = f.baseline_fhr
  const std  = f.std_fhr
  const tach = f.tachycardia_frac / 100   // convert % → fraction
  const brad = f.bradycardia_frac / 100

  // Baseline FHR scoring (FIGO 2015 thresholds)
  const blNorm  = bl >= 110 && bl <= 160
  const blWatch = (bl >= 100 && bl < 110) || (bl > 160 && bl <= 170)
  const blHigh  = bl < 100 || bl > 170

  const bl_n = blNorm ? 2.5 : blWatch ? 0.0 : -3.0
  const bl_w = blWatch ? 2.0 : blNorm ? -0.5 : 0.5
  const bl_h = blHigh ? 3.5 : blWatch ? 1.0 : -2.0

  // Std FHR (low std = reduced variability)
  const std_n = std >= 6 && std <= 20 ? 2.0 : std >= 3 ? 0.0 : -3.5
  const std_w = std >= 3 && std < 6 ? 2.5 : std > 20 ? 0.5 : 0.0
  const std_h = std < 3 ? 4.0 : std < 5 ? 1.5 : -1.5

  // Tachycardia/bradycardia fractions
  const tach_n = tach < 0.05 ? 1.0 : tach < 0.15 ? -0.5 : -2.0
  const tach_h = tach > 0.20 ? 2.5 : tach > 0.10 ? 1.0 : -0.5
  const brad_h = brad > 0.10 ? 2.5 : brad > 0.05 ? 1.0 : -0.5

  const n_logit = 0.35 * bl_n + 0.40 * std_n + 0.25 * tach_n
  const w_logit = 0.35 * bl_w + 0.40 * std_w
  const h_logit = 0.30 * bl_h + 0.35 * std_h + 0.20 * tach_h + 0.15 * brad_h

  return [n_logit, w_logit, h_logit]
}

// ── Expert B — FHR Variability ─────────────────────────────────────────────
// Primary driver of risk discrimination per CTU-CHB training
function expertB(f: FeatureValues): [number, number, number] {
  const stv     = f.stv
  const ltv     = f.ltv
  const stvNorm = f.stv_norm  // 0–1
  const ltvNorm = f.ltv_norm  // 0–1

  // STV (FIGO: absent <0.5, reduced 0.5–1, normal 1–4, saltatory >6)
  const stv_n = stv >= 1.0 && stv <= 4.0 ? 3.5 : stv >= 0.5 ? -0.5 : -5.0
  const stv_w = stv >= 0.5 && stv < 1.0 ? 3.0 : stv > 4.0 && stv < 6.0 ? 1.0 : 0.0
  const stv_h = stv < 0.5 ? 5.0 : stv < 1.0 ? 2.0 : stv > 6.0 ? 1.5 : -2.5

  // LTV (normal 5–25 bpm)
  const ltv_n = ltv >= 5 && ltv <= 25 ? 2.0 : ltv > 25 ? -0.5 : -2.5
  const ltv_w = ltv >= 3 && ltv < 5 ? 2.0 : ltv > 25 ? 1.0 : 0.0
  const ltv_h = ltv < 3 ? 3.0 : ltv < 5 ? 1.5 : -1.5

  // Normalised STV < 0.1 → at risk
  const stvN_n = stvNorm >= 0.1 ? 1.5 : stvNorm >= 0.05 ? -0.5 : -2.0
  const stvN_h = stvNorm < 0.05 ? 2.5 : stvNorm < 0.1 ? 1.0 : -1.0

  const n_logit = 0.45 * stv_n + 0.30 * ltv_n + 0.25 * stvN_n
  const w_logit = 0.45 * stv_w + 0.30 * ltv_w
  const h_logit = 0.45 * stv_h + 0.30 * ltv_h + 0.25 * stvN_h

  return [n_logit, w_logit, h_logit]
}

// ── Expert C — Event Patterns ──────────────────────────────────────────────
// Decelerations, accelerations, late-decel, contraction stress response
function expertC(f: FeatureValues): [number, number, number] {
  const nDecels     = f.n_decels
  const decRate     = f.decels_per_30min
  const meanDepth   = f.mean_decel_depth
  const maxDepth    = f.max_decel_depth
  const prolonged   = f.prolonged_decel_flag      // 0 or 1
  const lateLikely  = f.late_decel_likelihood     // 0–1
  const nAccels     = f.n_accels
  const accelRate   = f.accels_per_30min
  const delayed     = f.delayed_recovery_score    // 0–1
  const fhrDrop     = f.mean_fhr_drop_post_uc     // bpm

  // Prolonged deceleration — strongest FIGO pathological indicator
  const prol_n = prolonged === 0 ? 2.5 : -5.0
  const prol_w = prolonged === 0 ? -0.5 : 1.0
  const prol_h = prolonged === 1 ? 6.0 : -2.0

  // Late deceleration likelihood
  const late_n = lateLikely < 0.10 ? 1.5 : lateLikely < 0.25 ? -0.5 : -3.0
  const late_w = lateLikely >= 0.10 && lateLikely < 0.35 ? 2.0 : 0.0
  const late_h = lateLikely >= 0.50 ? 4.0 : lateLikely >= 0.25 ? 2.0 : -1.0

  // Deceleration depth
  const depth_n = meanDepth < 15 ? 1.0 : meanDepth < 30 ? -0.5 : -2.5
  const depth_h = meanDepth >= 40 ? 3.0 : meanDepth >= 25 ? 1.5 : -0.5
  const maxD_h  = maxDepth >= 60 ? 2.5 : maxDepth >= 40 ? 1.0 : -0.5

  // Deceleration rate
  const rate_n = decRate < 2 ? 1.0 : decRate < 5 ? -0.5 : -2.0
  const rate_h = decRate >= 8 ? 2.5 : decRate >= 5 ? 1.0 : -0.5

  // Accelerations (protective)
  const acc_n = accelRate >= 2 ? 3.0 : accelRate >= 1 ? 1.0 : -2.5
  const acc_w = accelRate < 1 && accelRate >= 0.5 ? 1.5 : 0.0
  const acc_h = accelRate < 0.5 ? 2.5 : accelRate < 1 ? 1.0 : -2.0

  // Delayed recovery (contraction stress response)
  const del_n = delayed < 0.10 ? 1.0 : delayed < 0.30 ? -0.5 : -2.5
  const del_w = delayed >= 0.20 && delayed < 0.50 ? 2.0 : 0.0
  const del_h = delayed >= 0.60 ? 3.5 : delayed >= 0.40 ? 2.0 : -0.5

  // FHR drop post-UC
  const drop_h = fhrDrop >= 25 ? 2.0 : fhrDrop >= 15 ? 1.0 : -0.5

  const n_logit = 0.30*prol_n + 0.20*late_n + 0.15*depth_n + 0.15*rate_n + 0.20*acc_n
  const w_logit = 0.30*prol_w + 0.25*late_w + 0.20*acc_w + 0.25*del_w
  const h_logit = 0.25*prol_h + 0.20*late_h + 0.15*depth_h + 0.10*maxD_h +
                  0.10*rate_h + 0.10*acc_h + 0.10*del_h + 0.05*drop_h - 0.05

  return [n_logit, w_logit, h_logit]
}

// ── Deceleration burden index (mirrors Python ctg_feature_engine.py) ────────
function decelBurdenIdx(f: FeatureValues): number {
  const burden = f.mean_decel_depth * (f.mean_decel_dur_s || 30) *
                 f.n_decels * Math.max(f.delayed_recovery_score, 0.05)
  return Math.log1p(burden)
}

// ── Fetal Reserve Score (0–100) ────────────────────────────────────────────
function computeFRS(f: FeatureValues, probs: number[]): number {
  let frs = 50.0
  const bl = f.baseline_fhr

  if (bl >= 110 && bl <= 160) frs += 15
  else if (bl < 100 || bl > 170) frs -= 12

  const stv = f.stv
  if (stv >= 1.0 && stv <= 4.0) frs += 15
  else if (stv >= 0.5) frs += 5
  else frs -= 20

  const ltv = f.ltv
  if (ltv >= 5 && ltv <= 25) frs += 10
  else if (ltv < 3) frs -= 10

  frs += Math.min(15, f.accels_per_30min * 3)
  frs -= Math.min(25, decelBurdenIdx(f) * 3)
  frs -= Math.min(15, f.delayed_recovery_score * 15)
  frs -= f.prolonged_decel_flag * 20
  frs -= f.late_decel_likelihood * 10
  frs *= Math.max(f.signal_quality, 0.3)

  frs = clamp(frs, 0, 100)
  const modelFRS = (1 - (probs[2] * 1.5 + probs[1] * 0.5)) * 100
  return Math.round(clamp(frs * 0.65 + modelFRS * 0.35, 0, 100))
}

// ── Gated meta-fusion (mirrors ReserveFusionMLP) ───────────────────────────
function reserveFusion(
  aLogits: [number, number, number],
  bLogits: [number, number, number],
  cLogits: [number, number, number],
  f: FeatureValues,
): [number, number, number] {
  const aProbs = softmax(aLogits as number[])
  const bProbs = softmax(bLogits as number[])
  const cProbs = softmax(cLogits as number[])

  // Confidence-gated weighting (max prob → expert authority)
  const confA = Math.max(...aProbs)
  const confB = Math.max(...bProbs)
  const confC = Math.max(...cProbs)
  const total = confA + confB + confC + 1e-9
  const wA = confA / total; const wB = confB / total; const wC = confC / total

  const fusedN = wA * aLogits[0] + wB * bLogits[0] + wC * cLogits[0]
  const fusedW = wA * aLogits[1] + wB * bLogits[1] + wC * cLogits[1]
  const fusedH = wA * aLogits[2] + wB * bLogits[2] + wC * cLogits[2]

  // Global override signals (from TOPQUA feature importance ranking)
  const burden   = decelBurdenIdx(f)
  const pathBoost = (f.prolonged_decel_flag >= 1 ? 3.0 : 0) +
                    (f.late_decel_likelihood >= 0.5 ? 2.0 : 0) +
                    (f.stv < 0.5 ? 2.5 : 0) +
                    (f.delayed_recovery_score >= 0.6 ? 1.5 : 0) +
                    (burden > 8 ? 2.0 : burden > 4 ? 1.0 : 0)

  const watchBoost = (f.prolonged_decel_flag === 0 && f.late_decel_likelihood >= 0.2 ? 1.2 : 0) +
                     (f.stv >= 0.5 && f.stv < 1.0 ? 1.5 : 0) +
                     (f.delayed_recovery_score >= 0.3 && f.delayed_recovery_score < 0.6 ? 1.0 : 0)

  const normBoost = (f.stv >= 1.5 && f.prolonged_decel_flag === 0 &&
                     f.late_decel_likelihood < 0.10 && f.accels_per_30min >= 2 ? 1.5 : 0)

  return [fusedN + normBoost, fusedW + watchBoost, fusedH + pathBoost]
}

// ── Build clinical explanations ────────────────────────────────────────────
function buildExplanations(f: FeatureValues, expertProbs: number[][]): string[] {
  const expl: string[] = []

  // Expert B — Variability (top CTU feature group)
  if (f.stv < 0.5)
    expl.push(`Absent short-term variability (${f.stv.toFixed(1)} bpm) — critical autonomic compromise`)
  else if (f.stv < 1.0)
    expl.push(`Reduced short-term variability (${f.stv.toFixed(1)} bpm) — sub-optimal autonomic tone`)
  else
    expl.push(`Normal short-term variability (${f.stv.toFixed(1)} bpm) — reassuring autonomic function`)

  if (f.ltv < 3)
    expl.push(`Absent long-term variability (${f.ltv.toFixed(1)} bpm) — concerning loss of variability`)
  else if (f.ltv < 5)
    expl.push(`Reduced long-term variability (${f.ltv.toFixed(1)} bpm) — warrants close monitoring`)

  // Expert A — Baseline
  if (f.baseline_fhr > 160)
    expl.push(`Tachycardia (${Math.round(f.baseline_fhr)} bpm) — baseline above normal range`)
  else if (f.baseline_fhr < 110)
    expl.push(`Bradycardia (${Math.round(f.baseline_fhr)} bpm) — baseline below normal range`)
  else
    expl.push(`Baseline FHR within normal range (${Math.round(f.baseline_fhr)} bpm)`)

  // Expert C — Events
  if (f.prolonged_decel_flag >= 1)
    expl.push('Prolonged deceleration ≥2 min detected — strongest FIGO pathological indicator')
  if (f.late_decel_likelihood >= 0.50)
    expl.push(`High late deceleration rate (${Math.round(f.late_decel_likelihood * 100)}%) — utero-placental insufficiency pattern`)
  else if (f.late_decel_likelihood >= 0.20)
    expl.push(`Borderline late deceleration rate (${Math.round(f.late_decel_likelihood * 100)}%) — monitor closely`)
  if (f.accels_per_30min >= 2)
    expl.push(`Good acceleration rate (${f.accels_per_30min.toFixed(1)}/30 min) — healthy fetal reactivity`)
  else if (f.accels_per_30min < 0.5)
    expl.push('Absent accelerations — reduced fetal reactivity; consider fetal stimulation')
  if (f.delayed_recovery_score >= 0.5)
    expl.push(`High delayed recovery (${Math.round(f.delayed_recovery_score * 100)}% of contractions) — contraction stress sign`)

  return expl.slice(0, 6)
}

// ── Main inference ─────────────────────────────────────────────────────────
export function predictLocally(features: FeatureValues): PredictionResult {

  const aLogits = expertA(features)
  const bLogits = expertB(features)
  const cLogits = expertC(features)
  const fusedLogits = reserveFusion(aLogits, bLogits, cLogits, features)
  const probs = tempScale(fusedLogits as number[])

  const [prob_normal, prob_suspect, prob_pathological] = probs
  const atRisk = prob_suspect + prob_pathological

  // TOPQUA threshold logic (mirrors Youden-tuned binary threshold ≈ 0.35)
  let risk_class: 0 | 1 | 2
  if (prob_pathological > 0.22 && prob_pathological > prob_suspect * 0.8) {
    risk_class = 2
  } else if (atRisk > 0.35 || prob_suspect > 0.28) {
    risk_class = 1
  } else {
    risk_class = 0
  }

  const risk_labels: ('Normal' | 'Suspect' | 'Pathological')[] = ['Normal', 'Suspect', 'Pathological']
  const risk_label  = risk_labels[risk_class]
  const confidence  = risk_class === 0 ? prob_normal :
                      risk_class === 1 ? prob_suspect : prob_pathological

  const expertProbs = [
    softmax(aLogits as number[]),
    softmax(bLogits as number[]),
    softmax(cLogits as number[]),
  ]

  const entropy = -probs.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0) / Math.log(3)
  const uncertainty: 'low' | 'moderate' | 'high' =
    entropy < 0.25 ? 'low' : entropy < 0.55 ? 'moderate' : 'high'

  const fetal_reserve_score = computeFRS(features, probs)
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
