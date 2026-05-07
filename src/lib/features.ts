export type FeatureKey =
  | 'baseline_value'
  | 'accelerations'
  | 'fetal_movement'
  | 'uterine_contractions'
  | 'light_decelerations'
  | 'severe_decelerations'
  | 'prolongued_decelerations'
  | 'abnormal_short_term_variability'
  | 'mean_value_of_short_term_variability'
  | 'percentage_of_time_with_abnormal_long_term_variability'
  | 'mean_value_of_long_term_variability'
  | 'histogram_width'
  | 'histogram_min'
  | 'histogram_max'
  | 'histogram_number_of_peaks'
  | 'histogram_number_of_zeroes'
  | 'histogram_mode'
  | 'histogram_mean'
  | 'histogram_median'
  | 'histogram_variance'
  | 'histogram_tendency'

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

export const FEATURE_GROUPS = ['Heart Rate', 'Decelerations', 'Variability', 'Histogram']

export const FEATURES: FeatureMeta[] = [
  // Heart Rate
  {
    key: 'baseline_value',
    label: 'Baseline FHR',
    unit: 'bpm',
    description: 'Mean fetal heart rate baseline',
    min: 100, max: 180, step: 1, defaultValue: 133,
    group: 'Heart Rate', importance: 'high',
  },
  {
    key: 'accelerations',
    label: 'Accelerations',
    unit: '/sec',
    description: 'Number of FHR accelerations per second',
    min: 0, max: 0.02, step: 0.001, defaultValue: 0.003,
    group: 'Heart Rate', importance: 'high',
  },
  {
    key: 'fetal_movement',
    label: 'Fetal Movement',
    unit: '/sec',
    description: 'Number of fetal movements per second',
    min: 0, max: 0.5, step: 0.001, defaultValue: 0.009,
    group: 'Heart Rate', importance: 'medium',
  },
  {
    key: 'uterine_contractions',
    label: 'Uterine Contractions',
    unit: '/sec',
    description: 'Number of uterine contractions per second',
    min: 0, max: 0.015, step: 0.001, defaultValue: 0.004,
    group: 'Heart Rate', importance: 'medium',
  },

  // Decelerations
  {
    key: 'light_decelerations',
    label: 'Light Decelerations',
    unit: '/sec',
    description: 'Number of light FHR decelerations per second',
    min: 0, max: 0.015, step: 0.001, defaultValue: 0.001,
    group: 'Decelerations', importance: 'medium',
  },
  {
    key: 'severe_decelerations',
    label: 'Severe Decelerations',
    unit: '/sec',
    description: 'Number of severe FHR decelerations per second',
    min: 0, max: 0.001, step: 0.0001, defaultValue: 0,
    group: 'Decelerations', importance: 'critical',
  },
  {
    key: 'prolongued_decelerations',
    label: 'Prolonged Decelerations',
    unit: '/sec',
    description: 'Number of prolonged decelerations per second — strongest pathological predictor',
    min: 0, max: 0.005, step: 0.0001, defaultValue: 0,
    group: 'Decelerations', importance: 'critical',
  },

  // Variability
  {
    key: 'abnormal_short_term_variability',
    label: 'Abnormal STV',
    unit: '%',
    description: 'Percentage of time with abnormal short-term variability',
    min: 0, max: 100, step: 1, defaultValue: 47,
    group: 'Variability', importance: 'critical',
  },
  {
    key: 'mean_value_of_short_term_variability',
    label: 'Mean STV',
    unit: 'bpm',
    description: 'Mean value of short-term variability — key autonomic marker',
    min: 0, max: 8, step: 0.1, defaultValue: 1.3,
    group: 'Variability', importance: 'critical',
  },
  {
    key: 'percentage_of_time_with_abnormal_long_term_variability',
    label: 'Abnormal LTV %',
    unit: '%',
    description: 'Percentage of time with abnormal long-term variability',
    min: 0, max: 100, step: 1, defaultValue: 10,
    group: 'Variability', importance: 'high',
  },
  {
    key: 'mean_value_of_long_term_variability',
    label: 'Mean LTV',
    unit: 'bpm',
    description: 'Mean value of long-term variability',
    min: 0, max: 55, step: 0.5, defaultValue: 8.2,
    group: 'Variability', importance: 'high',
  },

  // Histogram
  {
    key: 'histogram_width',
    label: 'Histogram Width',
    unit: 'bpm',
    description: 'Width of the FHR histogram (max − min)',
    min: 0, max: 200, step: 1, defaultValue: 70,
    group: 'Histogram', importance: 'medium',
  },
  {
    key: 'histogram_min',
    label: 'Histogram Min',
    unit: 'bpm',
    description: 'Minimum value of the FHR histogram',
    min: 40, max: 170, step: 1, defaultValue: 93,
    group: 'Histogram', importance: 'low',
  },
  {
    key: 'histogram_max',
    label: 'Histogram Max',
    unit: 'bpm',
    description: 'Maximum value of the FHR histogram',
    min: 100, max: 250, step: 1, defaultValue: 164,
    group: 'Histogram', importance: 'low',
  },
  {
    key: 'histogram_number_of_peaks',
    label: 'Histogram Peaks',
    unit: 'count',
    description: 'Number of peaks in the FHR histogram',
    min: 0, max: 20, step: 1, defaultValue: 4,
    group: 'Histogram', importance: 'low',
  },
  {
    key: 'histogram_number_of_zeroes',
    label: 'Histogram Zeroes',
    unit: 'count',
    description: 'Number of zeroes in the FHR histogram',
    min: 0, max: 12, step: 1, defaultValue: 0,
    group: 'Histogram', importance: 'low',
  },
  {
    key: 'histogram_mode',
    label: 'Histogram Mode',
    unit: 'bpm',
    description: 'Mode of the FHR histogram',
    min: 50, max: 200, step: 1, defaultValue: 137,
    group: 'Histogram', importance: 'medium',
  },
  {
    key: 'histogram_mean',
    label: 'Histogram Mean',
    unit: 'bpm',
    description: 'Mean of the FHR histogram',
    min: 60, max: 200, step: 1, defaultValue: 134,
    group: 'Histogram', importance: 'medium',
  },
  {
    key: 'histogram_median',
    label: 'Histogram Median',
    unit: 'bpm',
    description: 'Median of the FHR histogram',
    min: 60, max: 200, step: 1, defaultValue: 138,
    group: 'Histogram', importance: 'medium',
  },
  {
    key: 'histogram_variance',
    label: 'Histogram Variance',
    unit: '',
    description: 'Variance of the FHR histogram distribution',
    min: 0, max: 300, step: 1, defaultValue: 19,
    group: 'Histogram', importance: 'medium',
  },
  {
    key: 'histogram_tendency',
    label: 'Histogram Tendency',
    unit: '',
    description: 'Tendency of the histogram (−1 = left-leaning, 0 = symmetric, 1 = right-leaning)',
    min: -1, max: 1, step: 1, defaultValue: 0,
    group: 'Histogram', importance: 'low',
  },
]

export const DEFAULT_VALUES: FeatureValues = Object.fromEntries(
  FEATURES.map(f => [f.key, f.defaultValue])
) as FeatureValues

// Sample presets for demo purposes
export const PRESETS: { label: string; desc: string; tag: string; values: Partial<FeatureValues> }[] = [
  {
    label: 'Normal CTG',
    desc: 'Healthy fetal heart rate pattern with good variability and accelerations',
    tag: 'Normal',
    values: {
      baseline_value: 135,
      accelerations: 0.006,
      mean_value_of_short_term_variability: 1.8,
      abnormal_short_term_variability: 22,
      percentage_of_time_with_abnormal_long_term_variability: 5,
      prolongued_decelerations: 0,
      severe_decelerations: 0,
    },
  },
  {
    label: 'Suspect Pattern',
    desc: 'Reduced variability and absent accelerations — close monitoring warranted',
    tag: 'Suspect',
    values: {
      baseline_value: 140,
      accelerations: 0.001,
      mean_value_of_short_term_variability: 0.8,
      abnormal_short_term_variability: 55,
      percentage_of_time_with_abnormal_long_term_variability: 40,
      prolongued_decelerations: 0.0005,
      severe_decelerations: 0,
    },
  },
  {
    label: 'Pathological CTG',
    desc: 'Critical pattern: prolonged decelerations, absent variability',
    tag: 'Pathological',
    values: {
      baseline_value: 145,
      accelerations: 0,
      mean_value_of_short_term_variability: 0.3,
      abnormal_short_term_variability: 75,
      percentage_of_time_with_abnormal_long_term_variability: 70,
      prolongued_decelerations: 0.002,
      severe_decelerations: 0.0005,
    },
  },
]
