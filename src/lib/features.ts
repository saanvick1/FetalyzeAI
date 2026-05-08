/**
 * CTG feature definitions — aligned with CTU-CHB/CTU-UHB waveform model (TOPQUA architecture).
 * These map directly to the signal features extracted by ctg_feature_engine.py.
 */

export type FeatureKey =
  | 'baseline_fhr'
  | 'std_fhr'
  | 'tachycardia_frac'
  | 'bradycardia_frac'
  | 'stv'
  | 'ltv'
  | 'stv_norm'
  | 'ltv_norm'
  | 'n_accels'
  | 'accels_per_30min'
  | 'mean_accel_height'
  | 'n_decels'
  | 'decels_per_30min'
  | 'mean_decel_depth'
  | 'max_decel_depth'
  | 'mean_decel_dur_s'
  | 'prolonged_decel_flag'
  | 'late_decel_likelihood'
  | 'n_contractions'
  | 'contractions_per_10min'
  | 'mean_fhr_drop_post_uc'
  | 'delayed_recovery_score'
  | 'signal_quality'
  | 'duration_min'

export type FeatureValues = Record<FeatureKey, number>

export interface FeatureMeta {
  key: FeatureKey
  label: string
  unit: string
  description: string
  min: number
  max: number
  step: number
  defaultValue: number
  group: string
  importance: 'critical' | 'high' | 'medium' | 'low'
}

export const FEATURE_GROUPS = ['Heart Rate', 'Decelerations', 'Variability', 'Contractions']

export const FEATURES: FeatureMeta[] = [
  // ── Heart Rate ───────────────────────────────────────────────────────────
  {
    key: 'baseline_fhr',
    label: 'Baseline FHR',
    unit: 'bpm',
    description: 'Mean fetal heart rate baseline over full recording',
    min: 100, max: 180, step: 1, defaultValue: 135,
    group: 'Heart Rate', importance: 'high',
  },
  {
    key: 'std_fhr',
    label: 'FHR Std Dev',
    unit: 'bpm',
    description: 'Standard deviation of the FHR signal — overall variability measure',
    min: 0, max: 30, step: 0.5, defaultValue: 8,
    group: 'Heart Rate', importance: 'high',
  },
  {
    key: 'tachycardia_frac',
    label: 'Tachycardia Fraction',
    unit: '%',
    description: 'Fraction of time FHR exceeds 160 bpm',
    min: 0, max: 100, step: 1, defaultValue: 2,
    group: 'Heart Rate', importance: 'medium',
  },
  {
    key: 'bradycardia_frac',
    label: 'Bradycardia Fraction',
    unit: '%',
    description: 'Fraction of time FHR is below 110 bpm',
    min: 0, max: 100, step: 1, defaultValue: 1,
    group: 'Heart Rate', importance: 'medium',
  },

  // ── Decelerations ────────────────────────────────────────────────────────
  {
    key: 'n_decels',
    label: 'Deceleration Count',
    unit: 'count',
    description: 'Total number of FHR decelerations detected in the recording',
    min: 0, max: 60, step: 1, defaultValue: 3,
    group: 'Decelerations', importance: 'critical',
  },
  {
    key: 'decels_per_30min',
    label: 'Decelerations / 30 min',
    unit: '/30 min',
    description: 'Rate of decelerations normalised to 30-minute window',
    min: 0, max: 30, step: 0.5, defaultValue: 1.5,
    group: 'Decelerations', importance: 'critical',
  },
  {
    key: 'mean_decel_depth',
    label: 'Mean Decel Depth',
    unit: 'bpm',
    description: 'Average FHR drop below baseline during decelerations',
    min: 0, max: 80, step: 1, defaultValue: 20,
    group: 'Decelerations', importance: 'critical',
  },
  {
    key: 'max_decel_depth',
    label: 'Max Decel Depth',
    unit: 'bpm',
    description: 'Maximum FHR drop seen in any single deceleration',
    min: 0, max: 120, step: 1, defaultValue: 30,
    group: 'Decelerations', importance: 'critical',
  },
  {
    key: 'mean_decel_dur_s',
    label: 'Mean Decel Duration',
    unit: 'sec',
    description: 'Average duration of decelerations in seconds',
    min: 0, max: 300, step: 5, defaultValue: 60,
    group: 'Decelerations', importance: 'high',
  },
  {
    key: 'prolonged_decel_flag',
    label: 'Prolonged Decel',
    unit: '0/1',
    description: 'Flag: 1 if any deceleration lasts ≥ 2 minutes — strongest pathological indicator (FIGO)',
    min: 0, max: 1, step: 1, defaultValue: 0,
    group: 'Decelerations', importance: 'critical',
  },
  {
    key: 'late_decel_likelihood',
    label: 'Late Decel Likelihood',
    unit: '0–1',
    description: 'Fraction of contractions followed by a late FHR deceleration (nadir >30s post-UC)',
    min: 0, max: 1, step: 0.05, defaultValue: 0.05,
    group: 'Decelerations', importance: 'critical',
  },
  {
    key: 'n_accels',
    label: 'Acceleration Count',
    unit: 'count',
    description: 'Number of FHR accelerations ≥15 bpm lasting ≥15 s — protective reactivity marker',
    min: 0, max: 50, step: 1, defaultValue: 8,
    group: 'Decelerations', importance: 'high',
  },
  {
    key: 'accels_per_30min',
    label: 'Accelerations / 30 min',
    unit: '/30 min',
    description: 'Acceleration rate normalised to 30-minute window (≥2 is reassuring)',
    min: 0, max: 20, step: 0.5, defaultValue: 4,
    group: 'Decelerations', importance: 'high',
  },
  {
    key: 'mean_accel_height',
    label: 'Mean Accel Height',
    unit: 'bpm',
    description: 'Average peak height of accelerations above baseline',
    min: 0, max: 60, step: 1, defaultValue: 20,
    group: 'Decelerations', importance: 'medium',
  },

  // ── Variability ──────────────────────────────────────────────────────────
  {
    key: 'stv',
    label: 'Short-Term Variability',
    unit: 'bpm',
    description: 'Mean beat-to-beat FHR variation (STV) — key autonomic marker. Normal: 1–4 bpm.',
    min: 0, max: 10, step: 0.1, defaultValue: 1.5,
    group: 'Variability', importance: 'critical',
  },
  {
    key: 'ltv',
    label: 'Long-Term Variability',
    unit: 'bpm',
    description: 'Epoch range analysis of FHR variation (LTV). Normal: 5–25 bpm.',
    min: 0, max: 55, step: 0.5, defaultValue: 12,
    group: 'Variability', importance: 'high',
  },
  {
    key: 'stv_norm',
    label: 'STV Norm (÷10)',
    unit: '0–1',
    description: 'Short-term variability normalised to 0–1 scale (STV / 10)',
    min: 0, max: 1, step: 0.01, defaultValue: 0.15,
    group: 'Variability', importance: 'high',
  },
  {
    key: 'ltv_norm',
    label: 'LTV Norm (÷25)',
    unit: '0–1',
    description: 'Long-term variability normalised to 0–1 scale (LTV / 25)',
    min: 0, max: 2, step: 0.05, defaultValue: 0.48,
    group: 'Variability', importance: 'medium',
  },

  // ── Contractions ─────────────────────────────────────────────────────────
  {
    key: 'n_contractions',
    label: 'Contraction Count',
    unit: 'count',
    description: 'Number of uterine contractions detected in the recording',
    min: 0, max: 80, step: 1, defaultValue: 12,
    group: 'Contractions', importance: 'medium',
  },
  {
    key: 'contractions_per_10min',
    label: 'Contractions / 10 min',
    unit: '/10 min',
    description: 'Contraction rate (>5/10 min = tachysystole, associated with fetal compromise)',
    min: 0, max: 10, step: 0.25, defaultValue: 2,
    group: 'Contractions', importance: 'medium',
  },
  {
    key: 'mean_fhr_drop_post_uc',
    label: 'FHR Drop Post-UC',
    unit: 'bpm',
    description: 'Mean FHR drop in the 60 seconds following each uterine contraction',
    min: 0, max: 50, step: 1, defaultValue: 8,
    group: 'Contractions', importance: 'high',
  },
  {
    key: 'delayed_recovery_score',
    label: 'Delayed Recovery',
    unit: '0–1',
    description: 'Fraction of contractions where FHR recovery takes >30 s — contraction stress indicator',
    min: 0, max: 1, step: 0.05, defaultValue: 0.1,
    group: 'Contractions', importance: 'critical',
  },
  {
    key: 'signal_quality',
    label: 'Signal Quality',
    unit: '0–1',
    description: 'CTG signal quality score (1 = perfect, 0 = unusable — accounts for missing FHR, flatlines, jumps)',
    min: 0, max: 1, step: 0.05, defaultValue: 0.85,
    group: 'Contractions', importance: 'medium',
  },
  {
    key: 'duration_min',
    label: 'Recording Duration',
    unit: 'min',
    description: 'Total intrapartum CTG recording length in minutes',
    min: 10, max: 180, step: 5, defaultValue: 60,
    group: 'Contractions', importance: 'low',
  },
]

export const DEFAULT_VALUES: FeatureValues = Object.fromEntries(
  FEATURES.map(f => [f.key, f.defaultValue])
) as FeatureValues

export const PRESETS: { label: string; desc: string; tag: string; values: Partial<FeatureValues> }[] = [
  {
    label: 'Normal CTG',
    desc: 'Healthy intrapartum pattern: normal baseline, good variability, reactive accelerations, no significant decelerations',
    tag: 'Normal',
    values: {
      baseline_fhr: 135, std_fhr: 9, stv: 2.1, ltv: 14,
      n_accels: 10, accels_per_30min: 5, mean_accel_height: 22,
      n_decels: 2, decels_per_30min: 1, mean_decel_depth: 15,
      max_decel_depth: 20, prolonged_decel_flag: 0, late_decel_likelihood: 0.0,
      delayed_recovery_score: 0.05, tachycardia_frac: 1, bradycardia_frac: 0,
      signal_quality: 0.92,
    },
  },
  {
    label: 'Suspect Pattern',
    desc: 'Borderline: reduced variability, absent accelerations, some late decelerations — close monitoring warranted',
    tag: 'Suspect',
    values: {
      baseline_fhr: 142, std_fhr: 5, stv: 0.9, ltv: 6,
      n_accels: 1, accels_per_30min: 0.5, mean_accel_height: 16,
      n_decels: 8, decels_per_30min: 4, mean_decel_depth: 28,
      max_decel_depth: 45, prolonged_decel_flag: 0, late_decel_likelihood: 0.35,
      delayed_recovery_score: 0.4, tachycardia_frac: 3, bradycardia_frac: 1,
      signal_quality: 0.80,
    },
  },
  {
    label: 'Pathological CTG',
    desc: 'Critical: prolonged deceleration, absent STV, high late-decel rate — immediate obstetric review required',
    tag: 'Pathological',
    values: {
      baseline_fhr: 148, std_fhr: 3, stv: 0.3, ltv: 3,
      n_accels: 0, accels_per_30min: 0, mean_accel_height: 0,
      n_decels: 18, decels_per_30min: 9, mean_decel_depth: 50,
      max_decel_depth: 80, prolonged_decel_flag: 1, late_decel_likelihood: 0.75,
      delayed_recovery_score: 0.8, tachycardia_frac: 8, bradycardia_frac: 5,
      signal_quality: 0.70,
    },
  },
]
