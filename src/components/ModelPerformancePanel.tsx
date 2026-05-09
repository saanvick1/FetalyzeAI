import results from '../../results/ctu_reservenet_results.json'

const pct = (v: number) => `${(v * 100).toFixed(1)}%`
const fmt = (v: number, d = 3) => v.toFixed(d)

interface MetricCardProps {
  label: string
  value: string
  sub?: string
  color?: 'green' | 'blue' | 'amber' | 'red' | 'purple' | 'neutral'
  wide?: boolean
}

function MetricCard({ label, value, sub, color = 'neutral', wide }: MetricCardProps) {
  const colors: Record<string, string> = {
    green: '#16a34a', blue: '#2563eb', amber: '#d97706',
    red: '#dc2626', purple: '#7c3aed', neutral: '#374151',
  }
  return (
    <div style={{
      background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10,
      padding: '14px 18px', gridColumn: wide ? 'span 2' : undefined,
      borderTop: `3px solid ${colors[color]}`,
    }}>
      <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: colors[color] }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: '#9ca3af', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 style={{ fontSize: 15, fontWeight: 700, color: '#1e3a5f', margin: '28px 0 12px', paddingBottom: 6, borderBottom: '2px solid #dbeafe', display: 'flex', alignItems: 'center', gap: 8 }}>
      {children}
    </h2>
  )
}

function RocCurve() {
  const sweep = results.threshold_sweep
  const points: [number, number][] = [[1, 1]]
  for (const p of sweep) { points.push([p.specificity === undefined ? 0 : 1 - p.specificity, p.sensitivity]) }
  points.push([0, 0])
  const W = 260, H = 220, pad = 36
  const sx = (x: number) => pad + x * (W - pad * 2)
  const sy = (y: number) => H - pad - y * (H - pad * 2)
  const pts = points.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')
  const auroc = results.binary_headline.auroc

  return (
    <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, padding: 16 }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 8 }}>ROC Curve — AUC {fmt(auroc, 3)}</div>
      <svg width={W} height={H} style={{ display: 'block', margin: '0 auto' }}>
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="#e5e7eb" strokeWidth={1} strokeDasharray="4,3" />
        <polyline points={pts} fill="none" stroke="#2563eb" strokeWidth={2} strokeLinejoin="round" />
        {[0, 0.2, 0.4, 0.6, 0.8, 1].map(v => (
          <g key={v}>
            <line x1={sx(v)} y1={sy(0)} x2={sx(v)} y2={sy(1)} stroke="#f3f4f6" strokeWidth={1} />
            <text x={sx(v)} y={sy(0) + 14} fontSize={9} fill="#9ca3af" textAnchor="middle">{v.toFixed(1)}</text>
            <line x1={sx(0)} y1={sy(v)} x2={sx(1)} y2={sy(v)} stroke="#f3f4f6" strokeWidth={1} />
            <text x={sx(0) - 4} y={sy(v) + 3} fontSize={9} fill="#9ca3af" textAnchor="end">{v.toFixed(1)}</text>
          </g>
        ))}
        <text x={W / 2} y={H - 2} fontSize={10} fill="#6b7280" textAnchor="middle">1 − Specificity (FPR)</text>
        <text x={10} y={H / 2} fontSize={10} fill="#6b7280" textAnchor="middle" transform={`rotate(-90, 10, ${H / 2})`}>Sensitivity (TPR)</text>
      </svg>
      <div style={{ fontSize: 11, color: '#6b7280', textAlign: 'center' }}>5-fold OOF · 552 records · no leakage</div>
    </div>
  )
}

function CalibrationCurve() {
  const curve = results.calibration_curve
  const W = 260, H = 220, pad = 36
  const sx = (x: number) => pad + x * (W - pad * 2)
  const sy = (y: number) => H - pad - y * (H - pad * 2)
  const pts = curve.map(b => `${sx(b.predicted)},${sy(b.observed)}`).join(' ')
  const ece = results.calibration_metrics.ece_binary_oof
  const brier = results.calibration_metrics.brier_binary_oof

  return (
    <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, padding: 16 }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 8 }}>Calibration Plot · ECE {pct(ece)} · Brier {fmt(brier, 3)}</div>
      <svg width={W} height={H} style={{ display: 'block', margin: '0 auto' }}>
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="#e5e7eb" strokeWidth={1} strokeDasharray="4,3" />
        <polyline points={pts} fill="none" stroke="#7c3aed" strokeWidth={2} strokeLinejoin="round" />
        {curve.map((b, i) => (
          <circle key={i} cx={sx(b.predicted)} cy={sy(b.observed)} r={3} fill="#7c3aed" />
        ))}
        {[0, 0.2, 0.4, 0.6, 0.8, 1].map(v => (
          <g key={v}>
            <text x={sx(v)} y={sy(0) + 14} fontSize={9} fill="#9ca3af" textAnchor="middle">{v.toFixed(1)}</text>
            <text x={sx(0) - 4} y={sy(v) + 3} fontSize={9} fill="#9ca3af" textAnchor="end">{v.toFixed(1)}</text>
          </g>
        ))}
        <text x={W / 2} y={H - 2} fontSize={10} fill="#6b7280" textAnchor="middle">Mean Predicted Probability</text>
        <text x={10} y={H / 2} fontSize={10} fill="#6b7280" textAnchor="middle" transform={`rotate(-90, 10, ${H / 2})`}>Observed Fraction</text>
      </svg>
      <div style={{ fontSize: 11, color: '#6b7280', textAlign: 'center' }}>Perfect calibration = diagonal dashed line</div>
    </div>
  )
}

function ScoreHistogram() {
  const hist = results.score_histogram.slice(0, 16)
  const maxVal = Math.max(...hist.map(b => b.normal + b.at_risk))
  const W = 560, H = 180, pad = 36
  const barW = (W - pad * 2) / hist.length - 2
  const sy = (v: number) => H - pad - (v / maxVal) * (H - pad * 2)

  return (
    <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, padding: 16 }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 8 }}>Score Distribution — Predicted Risk Probability (552 records)</div>
      <svg width={W} height={H} style={{ display: 'block', margin: '0 auto', maxWidth: '100%' }}>
        {hist.map((b, i) => {
          const x = pad + i * ((W - pad * 2) / hist.length) + 1
          const hNormal = (b.normal / maxVal) * (H - pad * 2)
          const hRisk = (b.at_risk / maxVal) * (H - pad * 2)
          return (
            <g key={i}>
              <rect x={x} y={sy(b.normal)} width={barW} height={hNormal} fill="#93c5fd" opacity={0.8} />
              <rect x={x} y={sy(b.normal + b.at_risk)} width={barW} height={hRisk} fill="#f87171" opacity={0.8} />
              {i % 2 === 0 && <text x={x + barW / 2} y={H - 2} fontSize={8} fill="#9ca3af" textAnchor="middle">{b.bin_low.toFixed(2)}</text>}
            </g>
          )
        })}
        <text x={W / 2} y={H - 14} fontSize={10} fill="#6b7280" textAnchor="middle">Risk Score</text>
        <text x={pad + 4} y={pad - 4} fontSize={10} fill="#6b7280">n</text>
      </svg>
      <div style={{ display: 'flex', gap: 16, justifyContent: 'center', fontSize: 11, color: '#6b7280', marginTop: 4 }}>
        <span><span style={{ display: 'inline-block', width: 10, height: 10, background: '#93c5fd', borderRadius: 2, marginRight: 4 }} />Normal (447)</span>
        <span><span style={{ display: 'inline-block', width: 10, height: 10, background: '#f87171', borderRadius: 2, marginRight: 4 }} />At-Risk (105)</span>
      </div>
    </div>
  )
}

function ConfusionMatrix() {
  const cm = results.test_metrics.confusion_matrix
  const labels = ['Low Risk', 'Watch', 'High Risk']
  const colors = ['#bbf7d0', '#fef3c7', '#fee2e2']
  const total = cm.flat().reduce((a, b) => a + b, 0)

  return (
    <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, padding: 16 }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 12 }}>3-Class Confusion Matrix (n=83 holdout)</div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', fontSize: 12, margin: '0 auto' }}>
          <thead>
            <tr>
              <th style={{ padding: '6px 10px', color: '#6b7280', fontWeight: 600 }}></th>
              {labels.map(l => <th key={l} style={{ padding: '6px 10px', color: '#6b7280', fontWeight: 600, textAlign: 'center' }}>Pred {l}</th>)}
            </tr>
          </thead>
          <tbody>
            {cm.map((row, i) => (
              <tr key={i}>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: '#374151', whiteSpace: 'nowrap' }}>True {labels[i]}</td>
                {row.map((v, j) => (
                  <td key={j} style={{
                    padding: '10px 18px', textAlign: 'center', fontWeight: 700,
                    fontSize: 15, background: i === j ? colors[i] : '#f9fafb',
                    border: '1px solid #e5e7eb',
                    color: i === j ? '#1f2937' : '#6b7280'
                  }}>
                    {v}
                    <div style={{ fontSize: 9, fontWeight: 400, color: '#9ca3af' }}>{((v / total) * 100).toFixed(1)}%</div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ThresholdSweep() {
  const sweep = results.threshold_sweep
  const W = 560, H = 180, pad = 36
  const xs = sweep.map(p => p.threshold)
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const sx = (x: number) => pad + ((x - minX) / (maxX - minX)) * (W - pad * 2)
  const sy = (y: number) => H - pad - y * (H - pad * 2)
  const ptsSens = sweep.map(p => `${sx(p.threshold)},${sy(p.sensitivity)}`).join(' ')
  const ptsSpec = sweep.map(p => `${sx(p.threshold)},${sy(p.specificity)}`).join(' ')
  const ptsF1 = sweep.map(p => `${sx(p.threshold)},${sy(p.f1)}`).join(' ')
  const thr = results.test_metrics.threshold_used

  return (
    <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, padding: 16 }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 8 }}>Threshold Sweep — Sensitivity / Specificity / F1</div>
      <svg width={W} height={H} style={{ display: 'block', margin: '0 auto', maxWidth: '100%' }}>
        <line x1={sx(thr)} y1={sy(0)} x2={sx(thr)} y2={sy(1)} stroke="#d1d5db" strokeWidth={1} strokeDasharray="3,2" />
        <polyline points={ptsSens} fill="none" stroke="#2563eb" strokeWidth={2} />
        <polyline points={ptsSpec} fill="none" stroke="#16a34a" strokeWidth={2} />
        <polyline points={ptsF1} fill="none" stroke="#d97706" strokeWidth={2} />
        {[0, 0.2, 0.4, 0.6, 0.8, 1].map(v => (
          <g key={v}>
            <text x={sx(minX + v * (maxX - minX))} y={H - 2} fontSize={9} fill="#9ca3af" textAnchor="middle">{(minX + v * (maxX - minX)).toFixed(2)}</text>
            <text x={sx(minX) - 4} y={sy(v) + 3} fontSize={9} fill="#9ca3af" textAnchor="end">{v.toFixed(1)}</text>
          </g>
        ))}
        <text x={W / 2} y={H - 14} fontSize={10} fill="#6b7280" textAnchor="middle">Decision Threshold</text>
      </svg>
      <div style={{ display: 'flex', gap: 16, justifyContent: 'center', fontSize: 11, color: '#6b7280', marginTop: 4 }}>
        <span><span style={{ display: 'inline-block', width: 24, height: 2, background: '#2563eb', marginRight: 4, verticalAlign: 'middle' }} />Sensitivity</span>
        <span><span style={{ display: 'inline-block', width: 24, height: 2, background: '#16a34a', marginRight: 4, verticalAlign: 'middle' }} />Specificity</span>
        <span><span style={{ display: 'inline-block', width: 24, height: 2, background: '#d97706', marginRight: 4, verticalAlign: 'middle' }} />F1</span>
        <span style={{ color: '#9ca3af' }}>│ dashed = used threshold ({fmt(thr, 3)})</span>
      </div>
    </div>
  )
}

export function ModelPerformancePanel() {
  const m = results.test_metrics
  const hl = results.binary_headline
  const holdout = results.holdout_test
  const cal = results.calibration_metrics
  const unc = results.uncertainty_coverage
  const ood = results.ood_detection
  const adv = results.adversarial_summary
  const sq = results.signal_quality_subgroups

  const lrPlus = m.sensitivity / (1 - m.specificity)
  const lrMinus = (1 - m.sensitivity) / m.specificity

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: '24px 20px', fontFamily: 'system-ui, sans-serif' }}>

      <div style={{ background: 'linear-gradient(135deg,#1e3a5f,#2563eb)', borderRadius: 12, padding: '20px 24px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 20, fontWeight: 700 }}>Model Performance — FetalyzeAI</div>
        <div style={{ fontSize: 13, opacity: 0.85, marginTop: 4 }}>
          {results.dataset_name} · {results.n_records_loaded} real records · {results.model_version} · Trained {new Date(results.training_date).toLocaleDateString()}
        </div>
        <div style={{ display: 'flex', gap: 20, marginTop: 14, flexWrap: 'wrap' }}>
          {[
            { label: 'Records', value: `${results.n_records_loaded}` },
            { label: 'Features', value: `${results.n_features}` },
            { label: 'Protocol', value: '5-fold OOF + 83-record holdout' },
            { label: 'Synthetic data', value: 'None' },
          ].map(({ label, value }) => (
            <div key={label} style={{ background: 'rgba(255,255,255,0.15)', borderRadius: 8, padding: '6px 14px' }}>
              <div style={{ fontSize: 10, opacity: 0.75, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
              <div style={{ fontSize: 13, fontWeight: 700 }}>{value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 1. Core Classification Metrics */}
      <SectionTitle>1. Core Classification Metrics (OOF · 552 records)</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 10 }}>
        <MetricCard label="Accuracy" value={pct(m.accuracy)} sub="Overall correct predictions" color="blue" />
        <MetricCard label="Sensitivity / Recall" value={pct(m.sensitivity)} sub="True positive rate" color="green" />
        <MetricCard label="Specificity" value={pct(m.specificity)} sub="True negative rate" color="green" />
        <MetricCard label="Precision / PPV" value={pct(m.ppv)} sub="Positive predictive value" color="amber" />
        <MetricCard label="NPV" value={pct(m.npv)} sub="Negative predictive value" color="green" />
        <MetricCard label="F1 Score" value={fmt(m.f1_binary)} sub="Harmonic mean prec/recall" color="blue" />
        <MetricCard label="Macro F1 (3-class)" value={fmt(m.macro_f1)} sub="Balanced across 3 classes" color="blue" />
        <MetricCard label="MCC" value={fmt(m.mcc)} sub="Matthews correlation coeff" color="purple" />
        <MetricCard label="Balanced Accuracy" value={pct(m.balanced_accuracy)} sub="Mean sensitivity/specificity" color="blue" />
        <MetricCard label="False Positive Rate" value={pct(m.false_positive_rate)} sub="1 − specificity" color="red" />
        <MetricCard label="False Negative Rate" value={pct(m.false_negative_rate)} sub="1 − sensitivity" color="red" />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginTop: 10 }}>
        {[
          { label: 'True Positives', value: String(m.tp), color: 'green' as const },
          { label: 'True Negatives', value: String(m.tn), color: 'green' as const },
          { label: 'False Positives', value: String(m.fp), color: 'red' as const },
          { label: 'False Negatives', value: String(m.fn), color: 'red' as const },
        ].map(c => <MetricCard key={c.label} {...c} sub="OOF pool (binary)" />)}
      </div>

      {/* 2. Discrimination */}
      <SectionTitle>2. Discrimination Metrics</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 10, marginBottom: 16 }}>
        <MetricCard label="AUC-ROC (OOF)" value={fmt(hl.auroc)} sub="Ability to rank cases" color="blue" />
        <MetricCard label="AUC-PR (OOF)" value={fmt(hl.auprc)} sub="Precision-recall tradeoff" color="purple" />
        <MetricCard label="AUC-ROC (Holdout)" value={fmt(holdout.auroc)} sub="83-record test set" color="blue" />
        <MetricCard label="Holdout Sensitivity" value={pct(holdout.sensitivity)} sub="93.75% — critical safety" color="green" />
        <MetricCard label="Holdout Specificity" value={pct(holdout.specificity)} sub="83-record test set" color="amber" />
        <MetricCard label="Holdout F1" value={fmt(holdout.f1)} sub="Binary at-risk F1" color="blue" />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <RocCurve />
        <CalibrationCurve />
      </div>

      {/* 3. Calibration */}
      <SectionTitle>3. Calibration Metrics</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 10 }}>
        <MetricCard label="ECE (OOF binary)" value={pct(cal.ece_binary_oof)} sub="Expected calibration error" color="green" />
        <MetricCard label="Brier Score (OOF)" value={fmt(cal.brier_binary_oof)} sub="Mean sq prob error (0=best)" color="green" />
        <MetricCard label="Log Loss (OOF)" value={fmt(cal.log_loss_binary_oof)} sub="Cross-entropy loss" color="blue" />
        <MetricCard label="Temperature T" value={fmt(cal.temperature_T)} sub="Post-hoc calibration scaling" color="neutral" />
        <MetricCard label="ECE (3-class test)" value={pct(cal.ece_3class_test)} sub="3-class holdout" color="amber" />
        <MetricCard label="Brier (3-class test)" value={fmt(cal.brier_3class_test)} sub="3-class holdout" color="amber" />
      </div>

      {/* 4. Clinical Utility */}
      <SectionTitle>4. Clinical Utility Metrics</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 10, marginBottom: 16 }}>
        <MetricCard label="LR+ (Pos. Likelihood)" value={fmt(lrPlus, 2)} sub="How much risk ↑ if positive" color="amber" />
        <MetricCard label="LR− (Neg. Likelihood)" value={fmt(lrMinus, 2)} sub="How much risk ↓ if negative" color="green" />
        <MetricCard label="NPV" value={pct(m.npv)} sub="Safe to rule out at-risk" color="green" />
        <MetricCard label="PPV" value={pct(m.ppv)} sub="Confidence when flagging" color="amber" />
        <MetricCard label="Uncertain Rate" value={pct(unc.uncertain_rate)} sub="Grey-zone cases [0.35–0.65]" color="neutral" />
        <MetricCard label="Confident Accuracy" value={pct(unc.confident_accuracy)} sub="When model is confident" color="blue" />
        <MetricCard label="HR Recall (Confident)" value={pct(unc.high_risk_recall_confident)} sub="High-risk catch rate" color="green" />
        <MetricCard label="Decision Threshold" value={fmt(results.decision_threshold)} sub="Optimised on val F2-macro" color="neutral" />
      </div>

      {/* 5. Per-class recall */}
      <SectionTitle>5. Per-Class Performance (3-Class)</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 10, marginBottom: 16 }}>
        <MetricCard label="Low Risk Recall" value={pct(m.low_risk_recall)} sub="Correctly identified normals" color="green" />
        <MetricCard label="Watch Recall" value={pct(m.watch_recall)} sub="Borderline cases caught" color="amber" />
        <MetricCard label="High Risk Recall" value={pct(m.high_risk_recall)} sub="Critical cases caught" color="red" />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <ConfusionMatrix />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[
            { sq: 'good', label: 'Good Signal Quality', n: sq.good.n, acc: sq.good.accuracy, hr: sq.good.high_risk_recall, unc: sq.good.uncertainty_rate, color: '#16a34a' },
            { sq: 'acceptable', label: 'Acceptable Signal', n: sq.acceptable.n, acc: sq.acceptable.accuracy, hr: sq.acceptable.high_risk_recall, unc: sq.acceptable.uncertainty_rate, color: '#d97706' },
            { sq: 'poor', label: 'Poor Signal Quality', n: sq.poor.n, acc: sq.poor.accuracy, hr: sq.poor.high_risk_recall, unc: sq.poor.uncertainty_rate, color: '#dc2626' },
          ].map(g => (
            <div key={g.sq} style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, padding: '12px 16px', borderLeft: `4px solid ${g.color}` }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#374151' }}>{g.label} (n={g.n})</div>
              <div style={{ display: 'flex', gap: 16, marginTop: 6, fontSize: 12, color: '#6b7280' }}>
                <span>Accuracy <b style={{ color: '#1f2937' }}>{pct(g.acc)}</b></span>
                <span>HR Recall <b style={{ color: g.color }}>{pct(g.hr)}</b></span>
                <span>Uncertain <b style={{ color: '#1f2937' }}>{pct(g.unc)}</b></span>
              </div>
            </div>
          ))}
          <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, padding: '12px 16px' }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 6 }}>OOD Detection (IsolationForest)</div>
            <div style={{ fontSize: 12, color: '#6b7280' }}>
              OOD rate test <b style={{ color: '#1f2937' }}>{pct(ood.ood_rate_test)}</b> · High-risk OOD <b style={{ color: '#dc2626' }}>{pct(ood.ood_rate_high_risk_test)}</b>
            </div>
            <div style={{ fontSize: 11, color: '#9ca3af', marginTop: 4 }}>{ood.interpretation.substring(0, 90)}…</div>
          </div>
        </div>
      </div>

      {/* 6. Score Distribution */}
      <SectionTitle>6. Score Distribution (Probability Distribution Plot)</SectionTitle>
      <ScoreHistogram />

      {/* 7. Threshold sweep */}
      <SectionTitle>7. Threshold Effects — Sensitivity / Specificity / F1</SectionTitle>
      <ThresholdSweep />

      {/* 8. Adversarial Stress Tests */}
      <SectionTitle>8. Adversarial Stress Tests</SectionTitle>
      <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, overflow: 'hidden' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 10, padding: 12, background: '#f8fafc', borderBottom: '1px solid #e5e7eb' }}>
          <MetricCard label="3-Class Pass Rate" value={pct(adv.pass_rate_3class)} sub={`${adv.n_cases} canonical CTG profiles`} color="amber" />
          <MetricCard label="Binary Pass Rate" value={pct(adv.pass_rate_binary)} sub="At-risk vs normal discrimination" color="green" />
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f9fafb' }}>
              {['Case', 'Expected', 'Predicted', 'P(Normal)', 'P(Watch)', 'P(High Risk)', '3-class', 'Binary'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#6b7280', fontWeight: 600, borderBottom: '1px solid #e5e7eb', whiteSpace: 'nowrap' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {results.adversarial_stress_tests.map((t, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f3f4f6', background: i % 2 ? '#fafafa' : '#fff' }}>
                <td style={{ padding: '7px 10px', fontWeight: 500, color: '#374151' }}>{t.case}</td>
                <td style={{ padding: '7px 10px', color: '#6b7280' }}>{t.expected_label}</td>
                <td style={{ padding: '7px 10px', color: '#374151' }}>{t.predicted_label}</td>
                <td style={{ padding: '7px 10px', color: '#6b7280' }}>{pct(t.prob_normal)}</td>
                <td style={{ padding: '7px 10px', color: '#6b7280' }}>{pct(t.prob_watch)}</td>
                <td style={{ padding: '7px 10px', color: '#6b7280' }}>{pct(t.prob_high_risk)}</td>
                <td style={{ padding: '7px 10px' }}>
                  <span style={{ background: t.correct_3class ? '#dcfce7' : '#fee2e2', color: t.correct_3class ? '#16a34a' : '#dc2626', borderRadius: 4, padding: '2px 7px', fontWeight: 600 }}>
                    {t.correct_3class ? '✓' : '✗'}
                  </span>
                </td>
                <td style={{ padding: '7px 10px' }}>
                  <span style={{ background: t.correct_binary ? '#dcfce7' : '#fee2e2', color: t.correct_binary ? '#16a34a' : '#dc2626', borderRadius: 4, padding: '2px 7px', fontWeight: 600 }}>
                    {t.correct_binary ? '✓' : '✗'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 9. Reporting notes */}
      <SectionTitle>9. Reporting Notes</SectionTitle>
      <div style={{ background: '#fffbeb', border: '1px solid #fcd34d', borderRadius: 10, padding: '14px 18px', fontSize: 12, color: '#92400e', lineHeight: 1.6 }}>
        <b>Evaluation protocol:</b> {results.binary_headline.evaluation_protocol}<br />
        <b>Holdout note:</b> {results.holdout_test.note}<br />
        <b>Calibration note:</b> {results.calibration_metrics.note}<br />
        <b>OOD method:</b> {results.ood_detection.method}<br />
        <b>Threshold optimization:</b> {results.threshold_3class.optimization_metric} on validation fold (val F2 = {fmt(results.threshold_3class.val_f2_macro)})
      </div>

      <div style={{ marginTop: 24, fontSize: 11, color: '#9ca3af', textAlign: 'center' }}>
        For research use only · Not FDA-cleared · All metrics derived from real CTU-CHB/CTU-UHB intrapartum recordings
      </div>
    </div>
  )
}
