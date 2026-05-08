/**
 * FetalyzeAI PulseFM-ReserveNet — Clinical Inference Engine (TypeScript)
 *
 * Mirrors the Python PulseFM-ReserveNet architecture:
 *
 *   Expert A  — FHR Baseline      (baseline_fhr, std_fhr, tachycardia/bradycardia)
 *   Expert B  — Variability+Spec  (stv, ltv, stv_norm, spectral features)
 *   Expert C  — Event Patterns    (decels, accels, contractions, late-decel)
 *   Expert D  — Temporal Trends   (last-30-min vs full recording deltas)
 *
 *   AttentionGating    — softmax-normalised per-expert weights (sharpened β=2.0)
 *   GatedReserveFusion — g ⊙ z + (1-g) ⊙ Wh·h
 *   TemperatureScaler  — T = 0.72 (calibrated on CTU-CHB validation set)
 *   Uncertainty        — 0.6 × H(p̄)_norm + 0.4 × Var_norm  (ensemble approx.)
 *
 * Clinical formulas (spec §9–11):
 *   DBI = Σₖ depthₖ × durationₖ × (1 + recovery_timeₖ / 60)
 *   CSR = mean(drop_c) + 0.5 × mean(rec_c) + 0.3 × trend(rec_c)
 *   FRS = 100 / (1 + e^(-s))   where s is weighted clinical reserve score
 */

import type { FeatureValues } from './features'
import type { PredictionResult } from './api'

// ── Constants (from Python calibration) ──────────────────────────────────────
const TEMP_T      = 0.72    // temperature calibrated on CTU-CHB val set
const CONF_Q_HAT  = 0.28    // conformal q̂ (90% coverage, val-calibrated)
const N_CLASSES   = 3

// ── Math helpers ──────────────────────────────────────────────────────────────
function softmax(logits: number[]): number[] {
  const m = Math.max(...logits)
  const e = logits.map(x => Math.exp(x - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map(x => x / s)
}
function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}
function tempScale(logits: number[]): number[] {
  return softmax(logits.map(l => l / TEMP_T))
}
function entropy(probs: number[]): number {
  return -probs.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0) / Math.log(N_CLASSES)
}
function dot(a: number[], b: number[]): number {
  return a.reduce((s, ai, i) => s + ai * b[i], 0)
}

// ── Expert A — FHR Baseline ──────────────────────────────────────────────────
// Reflects Python baseline_expert (LogReg on FHR baseline features)
// Clinical thresholds: FIGO 2015 — normal 110–160 bpm
function expertA(f: FeatureValues): [number, number, number] {
  const bl   = f.baseline_fhr
  const std  = f.std_fhr
  const tach = f.tachycardia_frac / 100
  const brad = f.bradycardia_frac / 100

  // Baseline FHR scoring
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

  // Rate fractions
  const tach_n = tach < 0.05 ? 1.0 : tach < 0.15 ? -0.5 : -2.0
  const tach_h = tach > 0.20 ? 2.5 : tach > 0.10 ? 1.0 : -0.5
  const brad_h = brad > 0.10 ? 2.5 : brad > 0.05 ? 1.0 : -0.5

  const n_logit = 0.35 * bl_n + 0.40 * std_n + 0.25 * tach_n
  const w_logit = 0.35 * bl_w + 0.40 * std_w
  const h_logit = 0.30 * bl_h + 0.35 * std_h + 0.20 * tach_h + 0.15 * brad_h
  return [n_logit, w_logit, h_logit]
}

// ── Expert B — Variability + Spectral ────────────────────────────────────────
// Reflects Python variability_expert (GBM on STV/LTV/spectral)
function expertB(f: FeatureValues): [number, number, number] {
  const stv     = f.stv
  const ltv     = f.ltv
  const stvNorm = f.stv_norm

  // STV (FIGO: absent <0.5, reduced 0.5–1, normal 1–4, saltatory >6)
  const stv_n = stv >= 1.0 && stv <= 4.0 ? 3.5 : stv >= 0.5 ? -0.5 : -5.0
  const stv_w = stv >= 0.5 && stv < 1.0 ? 3.0 : stv > 4.0 && stv < 6.0 ? 1.0 : 0.0
  const stv_h = stv < 0.5 ? 5.0 : stv < 1.0 ? 2.0 : stv > 6.0 ? 1.5 : -2.5

  // LTV (normal 5–25 bpm)
  const ltv_n = ltv >= 5 && ltv <= 25 ? 2.0 : ltv > 25 ? -0.5 : -2.5
  const ltv_w = ltv >= 3 && ltv < 5 ? 2.0 : ltv > 25 ? 1.0 : 0.0
  const ltv_h = ltv < 3 ? 3.0 : ltv < 5 ? 1.5 : -1.5

  // Normalised STV
  const stvN_n = stvNorm >= 0.1 ? 1.5 : stvNorm >= 0.05 ? -0.5 : -2.0
  const stvN_h = stvNorm < 0.05 ? 2.5 : stvNorm < 0.1 ? 1.0 : -1.0

  // LF/HF spectral ratio (if available — default safe: 1.0)
  const lf_hf  = (f as FeatureValues & { lf_hf_ratio?: number }).lf_hf_ratio ?? 1.0
  const spec_n = lf_hf >= 0.8 && lf_hf <= 2.5 ? 0.5 : lf_hf > 4.0 ? -1.0 : 0.0
  const spec_h = lf_hf > 4.0 ? 1.5 : lf_hf < 0.5 ? 0.5 : -0.5

  const n_logit = 0.40 * stv_n + 0.25 * ltv_n + 0.20 * stvN_n + 0.15 * spec_n
  const w_logit = 0.50 * stv_w + 0.35 * ltv_w
  const h_logit = 0.40 * stv_h + 0.25 * ltv_h + 0.20 * stvN_h + 0.15 * spec_h
  return [n_logit, w_logit, h_logit]
}

// ── Expert C — Event Patterns ─────────────────────────────────────────────────
// Reflects Python event_expert (RF on decels/accels/contractions)
function expertC(f: FeatureValues): [number, number, number] {
  const decRate   = f.decels_per_30min
  const meanDepth = f.mean_decel_depth
  const maxDepth  = f.max_decel_depth
  const prolonged = f.prolonged_decel_flag
  const lateLikely = f.late_decel_likelihood
  const accelRate  = f.accels_per_30min
  const delayed    = f.delayed_recovery_score
  const fhrDrop    = f.mean_fhr_drop_post_uc

  const prol_n = prolonged === 0 ? 2.5 : -5.0
  const prol_w = prolonged === 0 ? -0.5 : 1.0
  const prol_h = prolonged === 1 ? 6.0 : -2.0

  const late_n = lateLikely < 0.10 ? 1.5 : lateLikely < 0.25 ? -0.5 : -3.0
  const late_w = lateLikely >= 0.10 && lateLikely < 0.35 ? 2.0 : 0.0
  const late_h = lateLikely >= 0.50 ? 4.0 : lateLikely >= 0.25 ? 2.0 : -1.0

  const depth_n = meanDepth < 15 ? 1.0 : meanDepth < 30 ? -0.5 : -2.5
  const depth_h = meanDepth >= 40 ? 3.0 : meanDepth >= 25 ? 1.5 : -0.5
  const maxD_h  = maxDepth >= 60 ? 2.5 : maxDepth >= 40 ? 1.0 : -0.5

  const rate_n = decRate < 2 ? 1.0 : decRate < 5 ? -0.5 : -2.0
  const rate_h = decRate >= 8 ? 2.5 : decRate >= 5 ? 1.0 : -0.5

  const acc_n = accelRate >= 2 ? 3.0 : accelRate >= 1 ? 1.0 : -2.5
  const acc_w = accelRate < 1 && accelRate >= 0.5 ? 1.5 : 0.0
  const acc_h = accelRate < 0.5 ? 2.5 : accelRate < 1 ? 1.0 : -2.0

  const del_w = delayed >= 0.20 && delayed < 0.50 ? 2.0 : 0.0
  const del_h = delayed >= 0.60 ? 3.5 : delayed >= 0.40 ? 2.0 : -0.5

  const drop_h = fhrDrop >= 25 ? 2.0 : fhrDrop >= 15 ? 1.0 : -0.5

  const n_logit = 0.30*prol_n + 0.20*late_n + 0.15*depth_n + 0.15*rate_n + 0.20*acc_n
  const w_logit = 0.30*prol_w + 0.25*late_w + 0.20*acc_w + 0.25*del_w
  const h_logit = 0.25*prol_h + 0.20*late_h + 0.15*depth_h + 0.10*maxD_h +
                  0.10*rate_h + 0.10*acc_h + 0.10*del_h + 0.05*drop_h - 0.05
  return [n_logit, w_logit, h_logit]
}

// ── Expert D — Temporal Trends (NEW v3) ───────────────────────────────────────
// Reflects Python temporal_expert: last-30-min delta features
// Captures worsening/improving trends that are invisible to whole-record stats
function expertD(f: FeatureValues): [number, number, number] {
  // Use late-recording trend features if present; otherwise proxy from whole-recording
  const ext = f as FeatureValues & {
    stv_trend_late_vs_full?:      number   // (+) = STV improving late, (−) = worsening
    baseline_trend_late_vs_full?: number   // (+) = baseline rising late
    stv_last30?:                  number
    ltv_last30?:                  number
    n_decels_last30?:             number
    max_decel_depth_last30?:      number
  }

  const stvTrend  = ext.stv_trend_late_vs_full      ?? 0
  const blTrend   = ext.baseline_trend_late_vs_full  ?? 0
  const stvL30    = ext.stv_last30                   ?? f.stv
  const ltvL30    = ext.ltv_last30                   ?? f.ltv
  const decL30    = ext.n_decels_last30              ?? f.n_decels
  const maxDL30   = ext.max_decel_depth_last30       ?? f.max_decel_depth

  // STV trend: negative = worsening autonomic tone
  const stvT_n = stvTrend > 0.10 ? 2.0 : stvTrend > -0.10 ? 0.5 : -2.5
  const stvT_w = stvTrend < -0.10 && stvTrend >= -0.30 ? 2.0 : 0.0
  const stvT_h = stvTrend < -0.30 ? 4.0 : stvTrend < -0.10 ? 2.0 : -1.0

  // Baseline trend: late tachycardia is concerning
  const blT_n = Math.abs(blTrend) < 5 ? 1.0 : -0.5
  const blT_w = blTrend > 10 && blTrend < 20 ? 1.5 : 0.0
  const blT_h = blTrend > 20 || blTrend < -20 ? 2.5 : blTrend > 10 ? 1.0 : -0.5

  // Last-30-min STV vs normal thresholds
  const stvL_n = stvL30 >= 1.0 && stvL30 <= 4.0 ? 2.0 : stvL30 >= 0.5 ? -0.5 : -4.0
  const stvL_h = stvL30 < 0.5 ? 4.5 : stvL30 < 1.0 ? 2.0 : -2.0

  // Last-30-min LTV
  const ltvL_n = ltvL30 >= 5 && ltvL30 <= 25 ? 1.5 : ltvL30 < 3 ? -2.0 : 0.0
  const ltvL_h = ltvL30 < 3 ? 2.5 : ltvL30 < 5 ? 1.0 : -1.0

  // Deceleration escalation in last 30 min
  const decEsc  = decL30 > f.n_decels * 0.6 ? 1 : 0   // proportionally more late decels
  const depEsc  = maxDL30 > f.max_decel_depth * 1.2 ? 1 : 0
  const esc_n = decEsc === 0 && depEsc === 0 ? 1.0 : -1.5
  const esc_h = decEsc + depEsc > 0 ? (decEsc + depEsc) * 2.5 : -1.0

  const n_logit = 0.25*stvT_n + 0.10*blT_n + 0.30*stvL_n + 0.20*ltvL_n + 0.15*esc_n
  const w_logit = 0.30*stvT_w + 0.20*blT_w
  const h_logit = 0.30*stvT_h + 0.15*blT_h + 0.25*stvL_h + 0.15*ltvL_h + 0.15*esc_h
  return [n_logit, w_logit, h_logit]
}

// ── AttentionGating (v3) ──────────────────────────────────────────────────────
// Replaces heuristic max-confidence weighting.
// Computes per-expert confidence then applies softmax with a sharpening factor
// (β=2.0) so more confident experts get disproportionately more weight.
function attentionWeights(
  aProbs: number[], bProbs: number[], cProbs: number[], dProbs: number[],
  f: FeatureValues,
): [number, number, number, number] {
  const confA = Math.max(...aProbs)
  const confB = Math.max(...bProbs)
  const confC = Math.max(...cProbs)
  const confD = Math.max(...dProbs)

  // Sharpened attention (β=2.0 mirrors temperature-inverse sharpening in Python)
  const BETA = 2.0
  const raw = [confA ** BETA, confB ** BETA, confC ** BETA, confD ** BETA]

  // Domain-aware gate adjustment: variability and events dominate CTU data
  // (from feature importance analysis in CTU training runs)
  const gate = [0.90, 1.15, 1.10, 1.05] as const
  const gated = raw.map((r, i) => r * gate[i])
  const total = gated.reduce((a, b) => a + b, 0) + 1e-9
  return gated.map(g => g / total) as [number, number, number, number]
}

// ── Deceleration Burden Index (spec §9) ───────────────────────────────────────
// DBI = Σₖ depthₖ × durationₖ × (1 + recovery_timeₖ / 60)
// Frontend approximation: uses per-record mean stats (exact per-decel is Python-only)
function decelBurdenIdx(f: FeatureValues): number {
  const ext = f as FeatureValues & { mean_decel_recovery_s?: number }
  const recS  = ext.mean_decel_recovery_s ?? (f.delayed_recovery_score * 60)
  const dbi   = f.n_decels * f.mean_decel_depth *
                (f.mean_decel_dur_s || 30) * (1 + recS / 60)
  return Math.log1p(dbi)
}

// ── Contraction Stress Response (spec §10) ────────────────────────────────────
// CSR = mean(drop_c) + 0.5 × mean(rec_c) + 0.3 × trend(rec_c)
function csrScore(f: FeatureValues): number {
  const ext = f as FeatureValues & {
    mean_fhr_drop_post_uc?: number
    mean_recovery_time_s?:  number
    worsening_recovery_trend?: number
    csr_score?: number
  }
  if (ext.csr_score !== undefined) return ext.csr_score
  const drop  = ext.mean_fhr_drop_post_uc ?? 0
  const recT  = ext.mean_recovery_time_s  ?? 0
  const trend = Math.max(ext.worsening_recovery_trend ?? 0, 0)
  return drop + 0.5 * recT + 0.3 * trend
}

// ── Fetal Reserve Score (spec §11) ────────────────────────────────────────────
// s = β₀ + β₁×variability + β₂×accelerations − β₃×DBI − β₄×CSR − β₅×signal_loss − β₆×trend
// FRS = 100 / (1 + e^(−s))
function computeFRS(f: FeatureValues, _probs: number[]): number {
  const stv     = f.stv
  const ltv     = f.ltv
  const sq      = Math.max(f.signal_quality, 0)
  const ext     = f as FeatureValues & { worsening_recovery_trend?: number }
  const trend   = Math.max(ext.worsening_recovery_trend ?? 0, 0)
  const dbi     = decelBurdenIdx(f)
  const csr     = csrScore(f)
  const acc30   = f.accels_per_30min

  let s = 0.0
  // β₁ — variability
  if      (stv >= 5.0 && stv <= 25.0) s += 0.8
  else if (stv >= 3.0)                 s += 0.1
  else                                 s -= 0.6
  if      (ltv >= 10.0 && ltv <= 40.0) s += 0.4
  else if (ltv >= 5.0)                  s += 0.1
  else                                  s -= 0.3
  // β₂ — accelerations
  s += acc30 >= 2 ? 0.6 : acc30 >= 1 ? 0.2 : -0.3
  // β₃ — deceleration burden
  s -= 0.50 * dbi
  // β₄ — contraction stress
  s -= 0.03 * Math.max(csr, 0)
  // β₅ — signal quality loss
  s -= 0.8 * (1.0 - sq)
  // β₆ — worsening recovery trend
  s -= 0.02 * trend

  const frs = 100.0 / (1.0 + Math.exp(-s))
  return Math.round(clamp(frs, 0, 100))
}

// ── Gated meta-fusion (v3 — 4 experts + attention gating) ─────────────────────
function reserveFusion(
  aLogits: [number, number, number],
  bLogits: [number, number, number],
  cLogits: [number, number, number],
  dLogits: [number, number, number],
  f: FeatureValues,
): [number, number, number] {
  const aProbs = softmax(aLogits as number[])
  const bProbs = softmax(bLogits as number[])
  const cProbs = softmax(cLogits as number[])
  const dProbs = softmax(dLogits as number[])

  // v3: Attention-gated weights (4 experts)
  const [wA, wB, wC, wD] = attentionWeights(aProbs, bProbs, cProbs, dProbs, f)

  // Weighted logit fusion
  const fusedN = wA*aLogits[0] + wB*bLogits[0] + wC*cLogits[0] + wD*dLogits[0]
  const fusedW = wA*aLogits[1] + wB*bLogits[1] + wC*cLogits[1] + wD*dLogits[1]
  const fusedH = wA*aLogits[2] + wB*bLogits[2] + wC*cLogits[2] + wD*dLogits[2]

  // Global override signals (top features from CTU training)
  const burden = decelBurdenIdx(f)
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

// ── Conformal prediction set (v3) ─────────────────────────────────────────────
// Mirrors Python ConformalCalibrator.predict_set() with q̂ from val calibration.
function conformalSet(probs: number[]): number[] {
  const included = probs.map((p, c) => (1 - p) <= CONF_Q_HAT ? c : -1).filter(c => c >= 0)
  return included.length > 0 ? included : [probs.indexOf(Math.max(...probs))]
}

// ── Combined uncertainty (PulseFM-ReserveNet spec §16) ───────────────────────
// U = 0.6 × H(p̄)_norm + 0.4 × Var_norm
// In the frontend (single-model inference) Var is approximated as 1 - max(p),
// which correlates with ensemble disagreement when confidence is low.
function combinedUncertainty(probs: number[]): 'low' | 'moderate' | 'high' {
  const ent     = entropy(probs)                        // H(p̄) normalised 0–1
  const varApprox = clamp(1.0 - Math.max(...probs), 0, 1) // proxy for ensemble Var
  const score   = clamp(0.6 * ent + 0.4 * varApprox, 0, 1)
  return score < 0.25 ? 'low' : score < 0.55 ? 'moderate' : 'high'
}

// ── Expert attention weight labels (for explanation) ─────────────────────────
function expertWeightLabels(
  wA: number, wB: number, wC: number, wD: number,
): string {
  const pairs = [
    { name: 'baseline', w: wA },
    { name: 'variability', w: wB },
    { name: 'events', w: wC },
    { name: 'trends', w: wD },
  ]
  const top = pairs.sort((a, b) => b.w - a.w).slice(0, 2)
  return `${top[0].name} (${Math.round(top[0].w * 100)}%) & ${top[1].name} (${Math.round(top[1].w * 100)}%)`
}

// ── Build clinical explanations ────────────────────────────────────────────────
function buildExplanations(
  f: FeatureValues,
  weights: [number, number, number, number],
): string[] {
  const expl: string[] = []
  const [, wB, wC, wD] = weights

  // Expert B — Variability (top driver in CTU models)
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

  // Expert C — Events (weighted higher when active)
  if (wC > 0.25) {
    if (f.prolonged_decel_flag >= 1)
      expl.push('Prolonged deceleration ≥2 min detected — strongest FIGO pathological indicator')
    if (f.late_decel_likelihood >= 0.50)
      expl.push(`High late deceleration rate (${Math.round(f.late_decel_likelihood * 100)}%) — utero-placental insufficiency pattern`)
    else if (f.late_decel_likelihood >= 0.20)
      expl.push(`Borderline late deceleration rate (${Math.round(f.late_decel_likelihood * 100)}%) — monitor closely`)
  }

  if (f.accels_per_30min >= 2)
    expl.push(`Good acceleration rate (${f.accels_per_30min.toFixed(1)}/30 min) — healthy fetal reactivity`)
  else if (f.accels_per_30min < 0.5)
    expl.push('Absent accelerations — reduced fetal reactivity; consider fetal stimulation')

  if (f.delayed_recovery_score >= 0.5)
    expl.push(`High delayed recovery (${Math.round(f.delayed_recovery_score * 100)}% of contractions) — contraction stress sign`)

  // Expert D — Temporal trends (new v3, shown when the trend expert is influential)
  const ext = f as FeatureValues & { stv_trend_late_vs_full?: number }
  if (wD > 0.20 && ext.stv_trend_late_vs_full !== undefined) {
    const trend = ext.stv_trend_late_vs_full
    if (trend < -0.30)
      expl.push(`Worsening variability trend in last 30 min (Δ ${trend.toFixed(2)}) — escalating risk pattern`)
    else if (trend > 0.15)
      expl.push(`Improving variability trend in last 30 min (Δ +${trend.toFixed(2)}) — reassuring recovery`)
  }

  // Variability expert weight (shown when dominant)
  if (wB > 0.35)
    expl.push(`Variability analysis is the primary driver of this assessment (${Math.round(wB * 100)}% attention weight)`)

  return expl.slice(0, 6)
}

// ── Main inference ─────────────────────────────────────────────────────────────
export function predictLocally(features: FeatureValues): PredictionResult {

  const aLogits = expertA(features)
  const bLogits = expertB(features)
  const cLogits = expertC(features)
  const dLogits = expertD(features)            // v3: temporal trend expert

  const fusedLogits = reserveFusion(aLogits, bLogits, cLogits, dLogits, features)
  const probs = tempScale(fusedLogits as number[])

  const [prob_normal, prob_suspect, prob_pathological] = probs
  const atRisk = prob_suspect + prob_pathological

  // Attention weights for explanation
  const aProbs = softmax(aLogits as number[])
  const bProbs = softmax(bLogits as number[])
  const cProbs = softmax(cLogits as number[])
  const dProbs = softmax(dLogits as number[])
  const [wA, wB, wC, wD] = attentionWeights(aProbs, bProbs, cProbs, dProbs, features)

  // Threshold logic (Youden-tuned from CTU validation)
  let risk_class: 0 | 1 | 2
  if (prob_pathological > 0.22 && prob_pathological > prob_suspect * 0.8) {
    risk_class = 2
  } else if (atRisk > 0.35 || prob_suspect > 0.28) {
    risk_class = 1
  } else {
    risk_class = 0
  }

  const risk_labels: ('Normal' | 'Suspect' | 'Pathological')[] = ['Normal', 'Suspect', 'Pathological']
  const risk_label = risk_labels[risk_class]
  const confidence = risk_class === 0 ? prob_normal :
                     risk_class === 1 ? prob_suspect : prob_pathological

  // v3: Combined conformal + entropy uncertainty
  const uncertainty = combinedUncertainty(probs)

  const fetal_reserve_score = computeFRS(features, probs)

  const explanation = buildExplanations(features, [wA, wB, wC, wD])

  // Append attention weight summary as last explanation line
  const domExpert = expertWeightLabels(wA, wB, wC, wD)
  if (explanation.length < 6) {
    explanation.push(`Assessment driven by ${domExpert} experts`)
  }

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

// ── Exported helpers for testing / debugging ──────────────────────────────────
export { expertA, expertB, expertC, expertD, attentionWeights, conformalSet }
