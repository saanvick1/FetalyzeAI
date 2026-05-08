import rawResults from '../../results/ctu_reservenet_results.json'

const R = rawResults as any

function pct(v: number | null | undefined, dp = 1) {
  if (v == null) return '—'
  return `${(v * 100).toFixed(dp)}%`
}
function fmt(v: number | null | undefined, dp = 3) {
  if (v == null) return '—'
  return Number(v).toFixed(dp)
}
function badge(v: number | null | undefined, threshold = 0.89, invert = false) {
  if (v == null) return 'rn-badge-val--na'
  const pass = invert ? v < threshold : v >= threshold
  return pass ? 'rn-badge-val--pass' : 'rn-badge-val--warn'
}

const LAYER_COLORS = ['#3b82f6', '#a855f7', '#f59e0b', '#22c55e', '#ec4899']
const EXPERT_COLORS: Record<string, string> = {
  baseline_expert:    '#3b82f6',
  variability_expert: '#a855f7',
  event_expert:       '#f59e0b',
}
const EXPERT_LABELS: Record<string, string> = {
  baseline_expert:    'Expert A — FHR Baseline',
  variability_expert: 'Expert B — Variability',
  event_expert:       'Expert C — Event Patterns',
}

export function ReserveNetPanel() {
  const arch   = R.architecture ?? {}
  const cv     = R.cv5 ?? {}
  const boot   = R.bootstrap_ci ?? {}
  const ens    = R.test_metrics ?? {}
  const xm     = R.xgb_test_metrics ?? {}
  const hl     = R.binary_headline ?? ens
  const imps   = R.xgb_feature_importance ?? []
  const etImps = R.et_feature_importance ?? []
  const expI   = (R.expert_importances ?? {}) as Record<string, { feature: string; importance: number }[]>
  const roc    = R.roc_curve ?? []
  const pr     = R.pr_curve ?? []
  const pcm    = R.per_class_metrics ?? []
  const dist   = R.label_distribution ?? {}

  const maxImpXgb = Math.max(...imps.map((i: any) => i.importance ?? 0), 0.001)
  const maxImpEt  = Math.max(...etImps.map((i: any) => i.importance ?? 0), 0.001)
  const cmRows    = ['Normal (0)', 'Watch (1)', 'High Risk (2)']

  // ROC path for SVG
  const W = 220; const H = 180; const PAD = 24
  function rocPath() {
    if (!roc.length) return ''
    const pts = roc.map((p: any) =>
      `${PAD + p.fpr * (W - PAD * 2)},${H - PAD - p.tpr * (H - PAD * 2)}`
    )
    return 'M ' + pts.join(' L ')
  }
  function prPath() {
    if (!pr.length) return ''
    const pts = pr.map((p: any) =>
      `${PAD + p.recall * (W - PAD * 2)},${H - PAD - p.precision * (H - PAD * 2)}`
    )
    return 'M ' + pts.join(' L ')
  }

  const AUROC = hl.auroc ?? hl.auroc_binary ?? ens.auroc_binary
  const SENS  = hl.sensitivity  ?? ens.sensitivity
  const SPEC  = hl.specificity  ?? ens.specificity
  const F1    = hl.f1 ?? hl.f1_binary ?? ens.f1_binary
  const PREC  = hl.precision ?? hl.precision_binary ?? ens.precision_binary
  const AUPRC = hl.auprc ?? hl.auprc_binary ?? ens.auprc_binary
  const tsweep = (R.threshold_sweep ?? []) as Array<{threshold:number;sensitivity:number;specificity:number;f1:number;precision:number}>
  const calib  = (R.calibration_curve ?? []) as Array<{predicted:number;observed:number;count:number}>
  const shist  = (R.score_histogram ?? []) as Array<{bin_low:number;bin_high:number;normal:number;at_risk:number}>
  const holdout = R.holdout_test ?? {}
  const nEval = hl.n_eval ?? hl.n_test ?? 552
  const nAtRisk = hl.n_atrisk_eval ?? hl.n_atrisk_test ?? 113
  const _passEs = (v:number|null|undefined, t=0.89) => v != null && v >= t; void _passEs;

  return (
    <div className="rn">

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="rn-hero">
        <div className="rn-hero__left">
          <div className="rn-hero__tag">TOPQUA Architecture v2.0</div>
          <h2 className="rn-hero__title">FetalyzeAI ReserveNet</h2>
          <p className="rn-hero__sub">
            SMOTE-balanced stacked ensemble with Optuna-tuned XGBoost, ExtraTrees, RF and logistic
            meta-learner, trained exclusively on{' '}
            <strong>552 real CTU-CHB/CTU-UHB intrapartum CTG recordings</strong>.
            Labels derived from cord blood pH only — no feature-derived labels, no synthetic data.
          </p>
          <div className="rn-badges">
            <span className="rn-badge rn-badge--blue">552 real CTG records</span>
            <span className="rn-badge rn-badge--purple">SMOTE minority balancing</span>
            <span className="rn-badge rn-badge--amber">Optuna-tuned XGB</span>
            <span className="rn-badge rn-badge--green">OOF stacked meta-learner</span>
            <span className="rn-badge rn-badge--pink">pH-only labels</span>
            <span className="rn-badge rn-badge--blue">500-iter bootstrap CIs</span>
          </div>
        </div>
        <div className="rn-hero__stats">
          <div className="rn-stat">
            <span className="rn-stat__val">{R.n_records_labeled ?? R.n_records_total ?? '—'}</span>
            <span className="rn-stat__lab">Real records</span>
          </div>
          <div className="rn-stat">
            <span className="rn-stat__val">{R.n_features ?? '—'}</span>
            <span className="rn-stat__lab">Features</span>
          </div>
          <div className="rn-stat">
            <span className="rn-stat__val">{R.split?.train ?? '—'}/{R.split?.val ?? '—'}/{R.split?.test ?? '—'}</span>
            <span className="rn-stat__lab">Train/Val/Test</span>
          </div>
          <div className="rn-stat">
            <span className="rn-stat__val">{R.training_time_s ?? '—'}s</span>
            <span className="rn-stat__lab">Train time</span>
          </div>
          <div className="rn-stat">
            <span className="rn-stat__val rn-stat__val--mono">{fmt(AUROC)}</span>
            <span className="rn-stat__lab">AUROC (binary)</span>
          </div>
          <div className="rn-stat">
            <span className="rn-stat__val rn-stat__val--mono">{pct(SENS)}</span>
            <span className="rn-stat__lab">Sensitivity</span>
          </div>
        </div>
      </section>

      {/* ── Clinical Priority Metrics ────────────────────────────────────── */}
      <section className="rn-card">
        <h3 className="rn-card__title">Clinical Priority Metrics — Key Numbers at a Glance</h3>
        <p className="rn-note" style={{ marginBottom: 16 }}>
          The metrics that matter most for a clinical CTG second-reader. Balanced accuracy and macro-F1
          correct for the 81/12/7% class imbalance. High-risk recall and FNR capture the most dangerous
          miss type. AUPRC beats AUROC for imbalanced high-risk detection. ECE and Brier score measure
          calibration — does 80% confidence actually mean 80% correct? Uncertainty coverage shows
          whether the uncertainty flag catches dangerous misses.
        </p>
        <div className="rn-hl-grid">
          {[
            { label: 'Balanced Accuracy', val: ens.balanced_accuracy,                                          ci: boot.balanced_accuracy, thr: 0.70, inv: false, desc: 'Average recall per class — corrects for 81/12/7% imbalance' },
            { label: 'Macro-F1',          val: ens.macro_f1,                                                   ci: boot.macro_f1,          thr: 0.55, inv: false, desc: 'Equal-weighted F1 across Low Risk / Watch / High Risk' },
            { label: 'High-Risk Recall',  val: ens.high_risk_recall,                                           ci: boot.high_risk_recall,  thr: 0.80, inv: false, desc: 'Fraction of true high-risk cases found — the most critical clinical metric' },
            { label: 'High-Risk FNR',     val: ens.high_risk_recall != null ? 1 - ens.high_risk_recall : null, ci: null,                   thr: 0.20, inv: true,  desc: 'False-negative rate for high-risk: cases missed entirely (lower = safer)' },
            { label: 'Watch Recall',      val: ens.watch_recall,                                               ci: null,                   thr: 0.65, inv: false, desc: 'Watch-closely cases correctly identified — borderline cases matter too' },
            { label: 'High-Risk AUPRC',   val: AUPRC,                                                          ci: boot.auprc_binary,      thr: 0.40, inv: false, desc: 'Precision-recall AUC — more informative than AUROC for imbalanced detection' },
            { label: 'ECE',               val: (R.ece ?? null) as number | null,                               ci: null,                   thr: 0.10, inv: true,  desc: 'Expected calibration error — lower means probabilities match observed frequencies' },
            { label: 'Brier Score',       val: (R.brier_score ?? null) as number | null,                       ci: null,                   thr: 0.15, inv: true,  desc: 'Mean squared probability error — punishes confident wrong predictions' },
            { label: 'Uncertainty Rate',  val: (R.uncertainty_coverage as any)?.uncertain_rate ?? null,         ci: null,                   thr: null,             desc: 'Fraction of cases the model flags as uncertain rather than committing' },
          ].map(({ label, val, ci, thr, inv, desc }) => (
            <div key={label} className="rn-hl-cell">
              <div className="rn-hl-cell__label">{label}</div>
              <div className={`rn-hl-cell__val ${thr != null ? badge(val as number, thr, inv ?? false) : ''}`}>
                {val != null ? pct(val as number) : '—'}
              </div>
              {ci && (ci as any).ci_lo != null && (
                <div className="rn-hl-cell__ci">
                  95% CI [{fmt((ci as any).ci_lo)}, {fmt((ci as any).ci_hi)}]
                </div>
              )}
              <div className="rn-hl-cell__desc">{desc}</div>
            </div>
          ))}
        </div>
        <div className="rn-note rn-note--clinical" style={{ marginTop: 14 }}>
          <strong>Missing values (—):</strong> ECE, Brier score, and uncertainty coverage are computed
          by the Python training pipeline. Run <code>python train_adaptive.py</code> to populate them.
          All other values come from the current <code>ctu_reservenet_results.json</code>.
        </div>
      </section>

      {/* ── Headline Binary Metrics ───────────────────────────────────────── */}
      <section className="rn-card">
        <h3 className="rn-card__title">
          Headline Performance — OOF Stacked Ensemble over all {nEval} CTU-CHB Records
        </h3>
        <p className="rn-note" style={{ marginBottom: 16 }}>
          Primary task: normal (pH ≥ 7.15) vs at-risk (watch + high-risk). Predictions are
          out-of-fold (5-fold cross-fit meta-learner) so every record is held-out — no leakage.
          This is the standard scientifically-valid way to report stacked-ensemble performance
          and pools all {nEval} records ({nAtRisk} at-risk) for full statistical power.
          Bootstrap CIs from 500 resamplings.
        </p>
        <div className="rn-hl-grid">
          {[
            { label: 'AUROC',        val: AUROC,  ci: boot.auroc_binary, hi: true,  invert: false, desc: 'Area under ROC — discrimination power' },
            { label: 'Sensitivity',  val: SENS,   ci: boot.sensitivity,  hi: true,  invert: false, desc: 'At-risk recall (true positive rate)' },
            { label: 'Specificity',  val: SPEC,   ci: boot.specificity,  hi: true,  invert: false, desc: 'Normal recall (true negative rate)' },
            { label: 'F1 (binary)',  val: F1,     ci: boot.f1_binary,    hi: true,  invert: false, desc: 'Harmonic mean of precision and recall' },
            { label: 'Precision',    val: PREC,   ci: null,              hi: true,  invert: false, desc: 'Positive predictive value' },
            { label: 'AUPRC',        val: AUPRC,  ci: boot.auprc_binary, hi: true,  invert: false, desc: 'Area under precision-recall curve' },
            { label: 'Bal. Accuracy',val: hl.balanced_accuracy ?? ens.balanced_accuracy, ci: null, hi: true, invert: false, desc: '(sensitivity + specificity) / 2' },
            { label: 'Threshold',    val: R.decision_threshold ?? hl.threshold, ci: null, hi: false, invert: false, desc: 'Youden decision threshold on validation' },
          ].map(({ label, val, ci, hi, invert, desc }) => (
            <div key={label} className="rn-hl-cell">
              <div className="rn-hl-cell__label">{label}</div>
              <div className={`rn-hl-cell__val ${badge(val as number, 0.89, invert)}`}>
                {val != null ? (hi && val < 2 ? pct(val as number) : fmt(val as number, 3)) : '—'}
              </div>
              {ci && (
                <div className="rn-hl-cell__ci">
                  95% CI [{fmt(ci.ci_lo)}, {fmt(ci.ci_hi)}]
                </div>
              )}
              <div className="rn-hl-cell__desc">{desc}</div>
            </div>
          ))}
        </div>
        <div className="rn-note rn-note--clinical" style={{ marginTop: 16 }}>
          <strong>Evaluation pool:</strong> {nEval} records ({nAtRisk} at-risk, {nEval - nAtRisk} normal).
          Each prediction comes from a meta-learner that did <em>not</em> see that record during fitting.
          {holdout.auroc != null && (
            <> &nbsp;Single-split hold-out (n={holdout.n_test}) reports AUROC&nbsp;
              <strong>{fmt(holdout.auroc)}</strong>, sens&nbsp;<strong>{pct(holdout.sensitivity)}</strong>,
              spec&nbsp;<strong>{pct(holdout.specificity)}</strong> for transparency — small-n variance is high.
            </>
          )}
        </div>
      </section>

      {/* ── Threshold Sweep ──────────────────────────────────────────────── */}
      {tsweep.length > 0 && (
        <section className="rn-card">
          <h3 className="rn-card__title">Decision-Threshold Sweep — Sensitivity / Specificity / F1</h3>
          <p className="rn-note" style={{ marginBottom: 14 }}>
            How the at-risk decision changes as the cut-off moves from 0.05 → 0.95.
            The Youden-optimal threshold (vertical line) maximises sens + spec − 1.
            Doctors who want higher recall (catch more at-risk cases) can pick a lower threshold;
            higher specificity (fewer false alarms) means a higher threshold.
          </p>
          {(() => {
            const SW = 560; const SH = 220; const SP = 30
            const xAt = (t:number) => SP + ((t - 0.05) / 0.9) * (SW - SP*2)
            const yAt = (v:number) => SH - SP - v * (SH - SP*2)
            const path = (k:'sensitivity'|'specificity'|'f1'|'precision') =>
              'M ' + tsweep.map(p => `${xAt(p.threshold)},${yAt((p as any)[k] ?? 0)}`).join(' L ')
            const youden = R.decision_threshold ?? hl.threshold ?? 0.5
            return (
              <svg viewBox={`0 0 ${SW} ${SH}`} style={{ width: '100%', height: 240 }}>
                {[0, 0.25, 0.5, 0.75, 1].map(v => (
                  <g key={v}>
                    <line x1={SP} y1={yAt(v)} x2={SW - SP} y2={yAt(v)} stroke="#e5e7eb" strokeWidth="0.7" strokeDasharray="3,2" />
                    <text x={SP - 4} y={yAt(v) + 3} fontSize="9" textAnchor="end" fill="#9ca3af">{v.toFixed(2)}</text>
                  </g>
                ))}
                {[0.1, 0.3, 0.5, 0.7, 0.9].map(t => (
                  <text key={t} x={xAt(t)} y={SH - SP + 12} fontSize="9" textAnchor="middle" fill="#9ca3af">{t.toFixed(1)}</text>
                ))}
                <line x1={xAt(youden)} y1={SP} x2={xAt(youden)} y2={SH - SP} stroke="#111827" strokeWidth="1" strokeDasharray="4,2" />
                <text x={xAt(youden) + 4} y={SP + 10} fontSize="10" fill="#111827">Youden = {fmt(youden, 2)}</text>
                <path d={path('sensitivity')} fill="none" stroke="#dc2626" strokeWidth="2" />
                <path d={path('specificity')} fill="none" stroke="#16a34a" strokeWidth="2" />
                <path d={path('f1')}          fill="none" stroke="#2563eb" strokeWidth="2" />
                <path d={path('precision')}   fill="none" stroke="#a855f7" strokeWidth="1.5" strokeDasharray="3,2" />
                <line x1={SP} y1={SP} x2={SP} y2={SH - SP} stroke="#6b7280" strokeWidth="1" />
                <line x1={SP} y1={SH - SP} x2={SW - SP} y2={SH - SP} stroke="#6b7280" strokeWidth="1" />
                <text x={SW / 2} y={SH - 4} fontSize="10" textAnchor="middle" fill="#374151">Decision threshold</text>
                <g transform={`translate(${SW - 170}, ${SP + 10})`}>
                  {[
                    { c: '#dc2626', l: 'Sensitivity (recall)' },
                    { c: '#16a34a', l: 'Specificity' },
                    { c: '#2563eb', l: 'F1' },
                    { c: '#a855f7', l: 'Precision' },
                  ].map((it, i) => (
                    <g key={it.l} transform={`translate(0, ${i * 14})`}>
                      <rect x="0" y="-7" width="10" height="3" fill={it.c} />
                      <text x="14" y="-3" fontSize="10" fill="#374151">{it.l}</text>
                    </g>
                  ))}
                </g>
              </svg>
            )
          })()}
        </section>
      )}

      {/* ── Calibration Curve ────────────────────────────────────────────── */}
      {calib.length > 0 && (
        <section className="rn-card">
          <h3 className="rn-card__title">Calibration / Reliability Curve</h3>
          <p className="rn-note" style={{ marginBottom: 14 }}>
            Predicted probability vs observed at-risk frequency, by 10% bin.
            A perfectly-calibrated model lies on the diagonal. Bubble size = number of records in that bin.
            Calibration matters clinically — a “60% at-risk” prediction should mean 60% of those cases
            actually were at risk.
          </p>
          {(() => {
            const CW = 360; const CH = 280; const CP = 36
            const xAt = (v:number) => CP + v * (CW - CP*2)
            const yAt = (v:number) => CH - CP - v * (CH - CP*2)
            const maxN = Math.max(...calib.map(c => c.count), 1)
            return (
              <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: '100%', maxWidth: 480, height: 320 }}>
                {[0, 0.25, 0.5, 0.75, 1].map(v => (
                  <g key={v}>
                    <line x1={CP} y1={yAt(v)} x2={CW-CP} y2={yAt(v)} stroke="#e5e7eb" strokeDasharray="3,2" strokeWidth="0.7" />
                    <text x={CP - 4} y={yAt(v) + 3} fontSize="9" textAnchor="end" fill="#9ca3af">{v.toFixed(2)}</text>
                    <text x={xAt(v)} y={CH - CP + 12} fontSize="9" textAnchor="middle" fill="#9ca3af">{v.toFixed(2)}</text>
                  </g>
                ))}
                <line x1={xAt(0)} y1={yAt(0)} x2={xAt(1)} y2={yAt(1)} stroke="#9ca3af" strokeDasharray="4,3" strokeWidth="1" />
                <path
                  d={'M ' + calib.map(c => `${xAt(c.predicted)},${yAt(c.observed)}`).join(' L ')}
                  fill="none" stroke="#2563eb" strokeWidth="2"
                />
                {calib.map((c, i) => (
                  <g key={i}>
                    <circle cx={xAt(c.predicted)} cy={yAt(c.observed)}
                      r={3 + 8 * (c.count / maxN)} fill="#2563eb" fillOpacity="0.45" stroke="#1e40af" strokeWidth="1" />
                  </g>
                ))}
                <line x1={CP} y1={CP} x2={CP} y2={CH-CP} stroke="#6b7280" strokeWidth="1" />
                <line x1={CP} y1={CH-CP} x2={CW-CP} y2={CH-CP} stroke="#6b7280" strokeWidth="1" />
                <text x={CW / 2} y={CH - 6} fontSize="10" textAnchor="middle" fill="#374151">Predicted at-risk probability</text>
                <text x={10} y={CH/2} fontSize="10" textAnchor="middle" fill="#374151" transform={`rotate(-90,10,${CH/2})`}>Observed at-risk fraction</text>
              </svg>
            )
          })()}
        </section>
      )}

      {/* ── Calibration Quality Metrics ──────────────────────────────────── */}
      <section className="rn-card">
        <h3 className="rn-card__title">Calibration &amp; Probability Quality</h3>
        <p className="rn-note" style={{ marginBottom: 16 }}>
          A clinical AI tool must not just classify correctly — its confidence scores must match
          observed event rates. When it says "80% at-risk", roughly 80% of those cases should truly
          be at risk. These three metrics each measure that from a different angle.
        </p>
        <div className="rn-hl-grid">
          {[
            { label: 'ECE',            val: (R.ece         ?? null) as number | null, desc: 'Expected calibration error — avg gap between stated and actual confidence. Lower is better. < 0.05 is excellent; < 0.10 is good.',  inv: true,  thr: 0.10 },
            { label: 'Brier Score',    val: (R.brier_score ?? null) as number | null, desc: 'Mean squared probability error — penalises confident wrong answers. Lower is better. Perfect = 0; baseline (always-majority) ≈ 0.15.', inv: true,  thr: 0.15 },
            { label: 'Log-Loss',       val: (R.log_loss    ?? null) as number | null, desc: 'Negative log-likelihood — harsh penalty for high-confidence misses. Lower is better. Perfect calibration ≈ 0.',                      inv: true,  thr: 0.50 },
            { label: 'Temperature T',  val: (R.temperature_T ?? null) as number | null, desc: 'Platt / temperature scaling factor fit on validation set. T > 1 softens overconfident probabilities.',                             inv: false, thr: null  },
          ].map(({ label, val, desc, inv, thr }) => (
            <div key={label} className="rn-hl-cell">
              <div className="rn-hl-cell__label">{label}</div>
              <div className={`rn-hl-cell__val ${thr != null ? badge(val as number, thr, inv) : ''}`}>
                {val != null ? fmt(val as number, 4) : '—'}
              </div>
              <div className="rn-hl-cell__desc">{desc}</div>
            </div>
          ))}
        </div>
        {(R.ece == null && R.brier_score == null) && (
          <div className="rn-note" style={{ marginTop: 10 }}>
            ECE, Brier score, and log-loss are computed by the Python training pipeline.
            Run <code>python train_adaptive.py</code> to populate these values.
          </div>
        )}
      </section>

      {/* ── Uncertainty Coverage ──────────────────────────────────────────── */}
      <section className="rn-card">
        <h3 className="rn-card__title">Uncertainty Coverage — Selective Prediction</h3>
        <p className="rn-note" style={{ marginBottom: 16 }}>
          FetalyzeAI flags cases where its confidence is low. These metrics measure whether that flag
          is clinically useful: does accuracy improve when the model only predicts on confident cases?
          Does the uncertainty flag catch dangerous misses that would otherwise slip through?
        </p>
        {(R.uncertainty_coverage as any) ? (
          <div className="rn-hl-grid">
            {[
              { label: 'Uncertain Rate',         val: (R.uncertainty_coverage as any).uncertain_rate,             desc: 'Fraction of all cases flagged as uncertain — not committed to a prediction' },
              { label: 'Accuracy (confident)',   val: (R.uncertainty_coverage as any).confident_accuracy,         desc: 'Overall accuracy restricted to cases the model is confident about' },
              { label: 'HR Recall (confident)',  val: (R.uncertainty_coverage as any).high_risk_recall_confident, desc: 'High-risk recall on the confident subset — safety metric for committed predictions' },
              { label: 'HR Cases → Uncertain',   val: (R.uncertainty_coverage as any).high_risk_flagged_uncertain, desc: 'Fraction of true high-risk cases caught by the uncertainty flag (a safety net)' },
            ].map(({ label, val, desc }) => (
              <div key={label} className="rn-hl-cell">
                <div className="rn-hl-cell__label">{label}</div>
                <div className="rn-hl-cell__val">{val != null ? pct(val as number) : '—'}</div>
                <div className="rn-hl-cell__desc">{desc}</div>
              </div>
            ))}
          </div>
        ) : (
          <>
            <div className="rn-note">
              Uncertainty coverage statistics will appear here after training.
              Run <code>python train_adaptive.py</code> — the pipeline computes uncertain_rate,
              confident_accuracy, and high_risk_flagged_uncertain automatically.
            </div>
            <div className="rn-hl-grid" style={{ marginTop: 12 }}>
              {['Uncertain Rate','Accuracy (confident)','HR Recall (confident)','HR Cases → Uncertain'].map(l => (
                <div key={l} className="rn-hl-cell">
                  <div className="rn-hl-cell__label">{l}</div>
                  <div className="rn-hl-cell__val">—</div>
                  <div className="rn-hl-cell__desc">Populated after training</div>
                </div>
              ))}
            </div>
          </>
        )}
      </section>

      {/* ── Score Distribution Histogram ─────────────────────────────────── */}
      {shist.length > 0 && (
        <section className="rn-card">
          <h3 className="rn-card__title">Predicted-Score Distribution by True Outcome</h3>
          <p className="rn-note" style={{ marginBottom: 14 }}>
            Histogram of OOF predicted at-risk probabilities, separated by true outcome.
            A well-discriminating model pushes Normal cases to the left (low scores) and At-Risk cases
            to the right. Overlap in the middle is the irreducible clinical grey-zone where additional
            judgment is required.
          </p>
          {(() => {
            const HW = 600; const HH = 220; const HP = 32
            const maxC = Math.max(...shist.flatMap(b => [b.normal, b.at_risk]), 1)
            const bw = (HW - HP*2) / shist.length
            const yAt = (c:number) => HH - HP - (c / maxC) * (HH - HP*2)
            const youden = R.decision_threshold ?? hl.threshold ?? 0.5
            return (
              <svg viewBox={`0 0 ${HW} ${HH}`} style={{ width: '100%', height: 240 }}>
                {[0, 0.25, 0.5, 0.75, 1].map(v => {
                  const c = Math.round(maxC * v)
                  return (
                    <g key={v}>
                      <line x1={HP} y1={yAt(c)} x2={HW-HP} y2={yAt(c)} stroke="#e5e7eb" strokeDasharray="3,2" strokeWidth="0.7" />
                      <text x={HP-4} y={yAt(c)+3} fontSize="9" textAnchor="end" fill="#9ca3af">{c}</text>
                    </g>
                  )
                })}
                {shist.map((b, i) => {
                  const x = HP + i * bw
                  const yN = yAt(b.normal); const yR = yAt(b.at_risk)
                  return (
                    <g key={i}>
                      <rect x={x + 1} y={yN} width={bw/2 - 1} height={(HH - HP) - yN}
                            fill="#16a34a" fillOpacity="0.75">
                        <title>{`${b.bin_low.toFixed(2)}–${b.bin_high.toFixed(2)}: Normal n=${b.normal}`}</title>
                      </rect>
                      <rect x={x + bw/2} y={yR} width={bw/2 - 1} height={(HH - HP) - yR}
                            fill="#dc2626" fillOpacity="0.75">
                        <title>{`${b.bin_low.toFixed(2)}–${b.bin_high.toFixed(2)}: At-risk n=${b.at_risk}`}</title>
                      </rect>
                    </g>
                  )
                })}
                <line x1={HP + ((youden-0)/1)*(HW-HP*2)} y1={HP} x2={HP + ((youden-0)/1)*(HW-HP*2)} y2={HH-HP}
                      stroke="#111827" strokeWidth="1" strokeDasharray="4,2" />
                <text x={HP + youden*(HW-HP*2) + 4} y={HP + 10} fontSize="10" fill="#111827">Decision = {fmt(youden,2)}</text>
                {[0,0.25,0.5,0.75,1].map(v => (
                  <text key={v} x={HP + v*(HW-HP*2)} y={HH - HP + 12} fontSize="9" textAnchor="middle" fill="#9ca3af">{v.toFixed(2)}</text>
                ))}
                <line x1={HP} y1={HH-HP} x2={HW-HP} y2={HH-HP} stroke="#6b7280" strokeWidth="1" />
                <line x1={HP} y1={HP} x2={HP} y2={HH-HP} stroke="#6b7280" strokeWidth="1" />
                <text x={HW/2} y={HH - 4} fontSize="10" textAnchor="middle" fill="#374151">Predicted at-risk probability</text>
                <g transform={`translate(${HW - 180}, ${HP + 4})`}>
                  <rect x="0" y="0" width="10" height="10" fill="#16a34a" /><text x="14" y="9" fontSize="10" fill="#374151">Normal (true)</text>
                  <rect x="100" y="0" width="10" height="10" fill="#dc2626" /><text x="114" y="9" fontSize="10" fill="#374151">At-risk (true)</text>
                </g>
              </svg>
            )
          })()}
          <div className="rn-note" style={{ marginTop: 10 }}>
            <strong>Read this chart:</strong> bars on the left = green dominates → model correctly
            sends Normals low. Bars on the right = red dominates → at-risk cases correctly flagged.
            The vertical line marks the decision threshold.
          </div>
        </section>
      )}

      {/* ── ROC + PR Curves ──────────────────────────────────────────────── */}
      {roc.length > 0 && (
        <section className="rn-card">
          <h3 className="rn-card__title">ROC Curve &amp; Precision-Recall Curve — Test Set</h3>
          <div className="rn-curves">
            <div className="rn-curve-block">
              <div className="rn-curve-block__title">ROC Curve (AUROC = {fmt(AUROC)})</div>
              <svg viewBox={`0 0 ${W} ${H}`} className="rn-curve-svg">
                {/* Grid lines */}
                {[0.25, 0.5, 0.75].map(v => (
                  <g key={v}>
                    <line x1={PAD} y1={H - PAD - v * (H - PAD * 2)} x2={W - PAD} y2={H - PAD - v * (H - PAD * 2)} stroke="#e5e7eb" strokeWidth="0.8" strokeDasharray="3,2" />
                    <line x1={PAD + v * (W - PAD * 2)} y1={PAD} x2={PAD + v * (W - PAD * 2)} y2={H - PAD} stroke="#e5e7eb" strokeWidth="0.8" strokeDasharray="3,2" />
                  </g>
                ))}
                {/* Diagonal baseline */}
                <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={PAD} stroke="#d1d5db" strokeWidth="1" strokeDasharray="4,3" />
                {/* ROC path */}
                <path d={rocPath()} fill="none" stroke="#3b82f6" strokeWidth="2.5" strokeLinejoin="round" />
                <path d={rocPath()} fill="#3b82f6" fillOpacity="0.08" stroke="none" />
                {/* Axes */}
                <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#6b7280" strokeWidth="1.2" />
                <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#6b7280" strokeWidth="1.2" />
                {/* Labels */}
                <text x={W / 2} y={H - 4} fontSize="8" textAnchor="middle" fill="#6b7280">False Positive Rate</text>
                <text x={10} y={H / 2} fontSize="8" textAnchor="middle" fill="#6b7280" transform={`rotate(-90,10,${H/2})`}>True Positive Rate</text>
                {['0', '0.5', '1'].map((t, i) => (
                  <text key={t} x={PAD + i * (W - PAD * 2) / 2} y={H - PAD + 9} fontSize="7" textAnchor="middle" fill="#9ca3af">{t}</text>
                ))}
              </svg>
              <div className="rn-curve-stats">
                <span>AUROC = <strong>{fmt(AUROC)}</strong></span>
                <span>CI [{fmt(boot.auroc_binary?.ci_lo)}, {fmt(boot.auroc_binary?.ci_hi)}]</span>
              </div>
            </div>

            {pr.length > 0 && (
              <div className="rn-curve-block">
                <div className="rn-curve-block__title">Precision-Recall Curve (AUPRC = {fmt(AUPRC)})</div>
                <svg viewBox={`0 0 ${W} ${H}`} className="rn-curve-svg">
                  {[0.25, 0.5, 0.75].map(v => (
                    <g key={v}>
                      <line x1={PAD} y1={H - PAD - v * (H - PAD * 2)} x2={W - PAD} y2={H - PAD - v * (H - PAD * 2)} stroke="#e5e7eb" strokeWidth="0.8" strokeDasharray="3,2" />
                      <line x1={PAD + v * (W - PAD * 2)} y1={PAD} x2={PAD + v * (W - PAD * 2)} y2={H - PAD} stroke="#e5e7eb" strokeWidth="0.8" strokeDasharray="3,2" />
                    </g>
                  ))}
                  <path d={prPath()} fill="none" stroke="#a855f7" strokeWidth="2.5" strokeLinejoin="round" />
                  <path d={prPath()} fill="#a855f7" fillOpacity="0.08" stroke="none" />
                  <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#6b7280" strokeWidth="1.2" />
                  <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#6b7280" strokeWidth="1.2" />
                  <text x={W / 2} y={H - 4} fontSize="8" textAnchor="middle" fill="#6b7280">Recall</text>
                  <text x={10} y={H / 2} fontSize="8" textAnchor="middle" fill="#6b7280" transform={`rotate(-90,10,${H/2})`}>Precision</text>
                  {['0', '0.5', '1'].map((t, i) => (
                    <text key={t} x={PAD + i * (W - PAD * 2) / 2} y={H - PAD + 9} fontSize="7" textAnchor="middle" fill="#9ca3af">{t}</text>
                  ))}
                </svg>
                <div className="rn-curve-stats">
                  <span>AUPRC = <strong>{fmt(AUPRC)}</strong></span>
                  <span>Baseline = {pct(hl.n_atrisk_test && hl.n_test ? hl.n_atrisk_test / hl.n_test : 0.19)}</span>
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {/* ── 5-Fold CV ─────────────────────────────────────────────────────── */}
      <section className="rn-card">
        <h3 className="rn-card__title">5-Fold Cross-Validation — All 552 Records (SMOTE per fold)</h3>
        <p className="rn-note" style={{ marginBottom: 16 }}>
          Each fold applies SMOTE independently to its training partition only — no data leakage.
          Inner validation split used for per-fold Youden threshold tuning. The CV mean gives
          the most reliable out-of-sample estimate at this dataset size.
        </p>
        {/* Priority CV summary — spec §22 */}
        <div style={{ background: '#f0f9ff', borderRadius: 8, padding: 14, marginBottom: 16, border: '1px solid #bae6fd' }}>
          <div style={{ fontWeight: 600, fontSize: 13, color: '#0369a1', marginBottom: 10 }}>Spec-Required CV Mean ± Std</div>
          <div className="rn-cv-metrics">
            {[
              { label: 'Balanced Accuracy', mean: cv.mean_balanced_accuracy, std: cv.std_balanced_accuracy },
              { label: 'Macro-F1',          mean: cv.mean_macro_f1,          std: cv.std_macro_f1          },
              { label: 'High-Risk Recall',  mean: cv.mean_high_risk_recall,  std: cv.std_high_risk_recall  },
              { label: 'ECE',               mean: cv.mean_ece,               std: cv.std_ece               },
            ].map(m => (
              <div key={m.label} className="rn-cv-metric">
                <span className="rn-cv-metric__val">{m.mean != null ? pct(m.mean) : '—'}</span>
                <span className="rn-cv-metric__sd">± {m.std != null ? pct(m.std) : '—'}</span>
                <span className="rn-cv-metric__lab">{m.label}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="rn-cv-metrics">
          {[
            { label: 'AUROC',       mean: cv.mean_auroc, std: cv.std_auroc },
            { label: 'Sensitivity', mean: cv.mean_sens,  std: cv.std_sens  },
            { label: 'Specificity', mean: cv.mean_spec,  std: cv.std_spec  },
            { label: 'F1 (binary)', mean: cv.mean_f1,    std: cv.std_f1    },
            { label: 'Precision',   mean: cv.mean_prec,  std: cv.std_prec  },
          ].map(m => (
            <div key={m.label} className="rn-cv-metric">
              <span className="rn-cv-metric__val">{fmt(m.mean)}</span>
              <span className="rn-cv-metric__sd">± {fmt(m.std)}</span>
              <span className="rn-cv-metric__lab">{m.label}</span>
            </div>
          ))}
        </div>
        <div className="rn-table-wrap">
          <table className="rn-table">
            <thead>
              <tr><th>Fold</th><th>AUROC</th><th>Bal. Acc</th><th>Macro-F1</th><th>HR Recall</th><th>Sensitivity</th><th>Specificity</th><th>F1</th></tr>
            </thead>
            <tbody>
              {(cv.fold_auroc ?? []).map((auc: number, i: number) => (
                <tr key={i}>
                  <td className="rn-table__fold">Fold {i + 1}</td>
                  <td>{fmt(auc)}</td>
                  <td>{cv.fold_balanced_accuracy?.[i] != null ? pct(cv.fold_balanced_accuracy[i]) : '—'}</td>
                  <td>{cv.fold_macro_f1?.[i] != null ? fmt(cv.fold_macro_f1[i]) : '—'}</td>
                  <td>{cv.fold_high_risk_recall?.[i] != null ? pct(cv.fold_high_risk_recall[i]) : '—'}</td>
                  <td>{pct(cv.fold_sens?.[i])}</td>
                  <td>{pct(cv.fold_spec?.[i])}</td>
                  <td>{fmt(cv.fold_f1?.[i])}</td>
                </tr>
              ))}
              <tr className="rn-table__mean">
                <td>Mean ± SD</td>
                <td>{fmt(cv.mean_auroc)} ± {fmt(cv.std_auroc)}</td>
                <td>{cv.mean_balanced_accuracy != null ? `${pct(cv.mean_balanced_accuracy)} ± ${pct(cv.std_balanced_accuracy)}` : '—'}</td>
                <td>{cv.mean_macro_f1 != null ? `${fmt(cv.mean_macro_f1)} ± ${fmt(cv.std_macro_f1)}` : '—'}</td>
                <td>{cv.mean_high_risk_recall != null ? `${pct(cv.mean_high_risk_recall)} ± ${pct(cv.std_high_risk_recall)}` : '—'}</td>
                <td>{pct(cv.mean_sens)} ± {pct(cv.std_sens)}</td>
                <td>{pct(cv.mean_spec)} ± {pct(cv.std_spec)}</td>
                <td>{fmt(cv.mean_f1)} ± {fmt(cv.std_f1)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Per-Class Metrics ─────────────────────────────────────────────── */}
      {pcm.length > 0 && (
        <section className="rn-card">
          <h3 className="rn-card__title">Per-Class Precision / Recall / F1 — 3-Class XGBoost (test set)</h3>
          <p className="rn-note" style={{ marginBottom: 14 }}>
            3-class secondary model (Normal / Watch / High Risk). Binary at-risk metrics above are
            the primary headline. With only ~6 high-risk and ~10 watch cases in the test set,
            per-class recall is highly variable — treat as indicative only.
          </p>
          <div className="rn-table-wrap">
            <table className="rn-table">
              <thead>
                <tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th></tr>
              </thead>
              <tbody>
                {pcm.map((row: any) => (
                  <tr key={row.class}>
                    <td><span className={`rn-class-pill rn-class-pill--${row.class.includes('Normal') ? 'normal' : row.class.includes('Watch') ? 'watch' : 'high'}`}>{row.class}</span></td>
                    <td>{row.support}</td>
                    <td className={row.precision >= 0.8 ? 'rn-td--good' : row.precision >= 0.5 ? 'rn-td--warn' : 'rn-td--bad'}>{pct(row.precision)}</td>
                    <td className={row.recall >= 0.8 ? 'rn-td--good' : row.recall >= 0.5 ? 'rn-td--warn' : 'rn-td--bad'}>{pct(row.recall)}</td>
                    <td>{pct(row.f1)}</td>
                    <td className="rn-td--mono">{row.tp}</td>
                    <td className="rn-td--mono">{row.fp}</td>
                    <td className="rn-td--mono">{row.fn}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* ── Confusion Matrix ─────────────────────────────────────────────── */}
      {ens.confusion_matrix && (
        <section className="rn-card">
          <h3 className="rn-card__title">Confusion Matrix — 3-Class XGBoost (test set)</h3>
          <div className="rn-cm-wrap">
            <div className="rn-cm">
              <div className="rn-cm__header">
                <div className="rn-cm__corner">Actual ↓ / Pred →</div>
                {cmRows.map(r => <div key={r} className="rn-cm__col-head">{r}</div>)}
              </div>
              {ens.confusion_matrix.map((row: number[], ri: number) => {
                const rowTotal = row.reduce((a: number, b: number) => a + b, 0) || 1
                return (
                  <div key={ri} className="rn-cm__row">
                    <div className="rn-cm__row-head">{cmRows[ri]}</div>
                    {row.map((v: number, ci: number) => (
                      <div key={ci}
                        className={`rn-cm__cell ${ri === ci ? 'rn-cm__cell--diag' : v > 0 ? 'rn-cm__cell--err' : ''}`}>
                        <div className="rn-cm__cell-count">{v}</div>
                        <div className="rn-cm__cell-pct">{((v / rowTotal) * 100).toFixed(0)}%</div>
                      </div>
                    ))}
                  </div>
                )
              })}
            </div>
            <div className="rn-cm-legend">
              <div className="rn-cm-legend__item"><span className="rn-cm-legend__dot rn-cm-legend__dot--diag" />Diagonal = correct predictions</div>
              <div className="rn-cm-legend__item"><span className="rn-cm-legend__dot rn-cm-legend__dot--err" />Off-diagonal = misclassifications</div>
              <div className="rn-cm-legend__item rn-note">Rows = actual, columns = predicted. Counts and row-% shown.</div>
            </div>
          </div>
        </section>
      )}

      {/* ── XGB vs Ensemble Comparison ────────────────────────────────────── */}
      <section className="rn-card">
        <h3 className="rn-card__title">XGB Baseline vs Ensemble — Metric Comparison</h3>
        <div className="rn-metrics-cmp">
          {[
            { key: 'auroc_binary', label: 'AUROC', note: 'At-risk discrimination', hi: true },
            { key: 'sensitivity',  label: 'Sensitivity', note: 'At-risk recall', hi: true },
            { key: 'specificity',  label: 'Specificity', note: 'Normal recall', hi: true },
            { key: 'f1_binary',    label: 'F1 (binary)', note: 'Precision-recall balance', hi: true },
            { key: 'macro_f1',     label: 'Macro F1', note: '3-class balance', hi: true },
          ].map(({ key, label, note, hi }) => {
            const xv = xm[key] as number | null
            const ev = ens[key] as number | null
            const better = ev != null && xv != null ? (hi ? ev > xv + 0.002 : ev < xv - 0.002) : false
            return (
              <div key={key} className="rn-mc__col">
                <div className="rn-mc__label">{label}</div>
                <div className="rn-mc__note">{note}</div>
                <div className="rn-mc__row">
                  <div className="rn-mc__block rn-mc__block--xgb">
                    <div className="rn-mc__tag">XGB</div>
                    <div className="rn-mc__val">{xv != null ? pct(xv < 2 ? xv : xv) : '—'}</div>
                  </div>
                  <div className="rn-mc__arrow">{better ? '▲' : '→'}</div>
                  <div className="rn-mc__block rn-mc__block--ens">
                    <div className="rn-mc__tag">Ensemble</div>
                    <div className="rn-mc__val">{ev != null ? pct(ev < 2 ? ev : ev) : '—'}</div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </section>

      {/* ── CTG-Specific Validation ───────────────────────────────────────── */}
      <section className="rn-card">
        <h3 className="rn-card__title">CTG-Specific Clinical Validation</h3>
        <p className="rn-note" style={{ marginBottom: 16 }}>
          These sections validate the three custom FetalyzeAI clinical features: Fetal Reserve Score
          (FRS), Deceleration Burden Index (DBI), and Contraction Stress Response (CSR). For a sound
          clinical AI, these features should correlate with pH and discriminate risk classes.
          Results are populated when training writes <code>ctg_specific</code> into the results JSON.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>

          {/* FRS vs pH */}
          <div style={{ background: '#f8faff', borderRadius: 8, padding: 16, border: '1px solid #e0e7ff' }}>
            <div style={{ fontWeight: 700, color: '#3b82f6', marginBottom: 8, fontSize: 14 }}>Fetal Reserve Score vs pH</div>
            <p className="rn-note" style={{ marginBottom: 12 }}>FRS should correlate negatively with pH — lower reserve → worse outcome.</p>
            {(R.ctg_specific as any)?.frs_vs_ph ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {[
                  { label: 'Spearman r (FRS–pH)', val: (R.ctg_specific as any).frs_vs_ph.spearman_r },
                  { label: 'Pearson r (FRS–pH)',  val: (R.ctg_specific as any).frs_vs_ph.pearson_r  },
                  { label: 'AUC (low FRS → HR)',  val: (R.ctg_specific as any).frs_vs_ph.auc        },
                ].map(({ label, val }) => (
                  <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                    <span style={{ color: '#374151' }}>{label}</span>
                    <strong style={{ color: '#1e40af' }}>{val != null ? fmt(val as number, 3) : '—'}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rn-note">Run training to populate FRS vs pH correlations.</div>
            )}
          </div>

          {/* DBI by class */}
          <div style={{ background: '#fff8f0', borderRadius: 8, padding: 16, border: '1px solid #fde68a' }}>
            <div style={{ fontWeight: 700, color: '#d97706', marginBottom: 8, fontSize: 14 }}>Deceleration Burden Index by Class</div>
            <p className="rn-note" style={{ marginBottom: 12 }}>Higher DBI should correlate with worse risk class and lower pH.</p>
            {(R.ctg_specific as any)?.dbi_by_class ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {[
                  { label: 'Low Risk — mean DBI',   val: (R.ctg_specific as any).dbi_by_class.low_risk,   color: '#16a34a' },
                  { label: 'Watch — mean DBI',       val: (R.ctg_specific as any).dbi_by_class.watch,      color: '#d97706' },
                  { label: 'High Risk — mean DBI',   val: (R.ctg_specific as any).dbi_by_class.high_risk,  color: '#dc2626' },
                  { label: 'AUC (DBI → high-risk)',  val: (R.ctg_specific as any).dbi_by_class.auc,        color: '#374151' },
                  { label: 'Corr. with pH',          val: (R.ctg_specific as any).dbi_by_class.corr_ph,    color: '#374151' },
                ].map(({ label, val, color }) => (
                  <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                    <span style={{ color: '#374151' }}>{label}</span>
                    <strong style={{ color }}>{val != null ? fmt(val as number, 3) : '—'}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rn-note">Run training to populate DBI-by-class validation.</div>
            )}
          </div>

          {/* CSR by class */}
          <div style={{ background: '#f0fff4', borderRadius: 8, padding: 16, border: '1px solid #bbf7d0' }}>
            <div style={{ fontWeight: 700, color: '#16a34a', marginBottom: 8, fontSize: 14 }}>Contraction Stress Response by Class</div>
            <p className="rn-note" style={{ marginBottom: 12 }}>Higher CSR (delayed recovery) should track with worse outcomes.</p>
            {(R.ctg_specific as any)?.csr_by_class ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {[
                  { label: 'Low Risk — mean CSR',      val: (R.ctg_specific as any).csr_by_class.low_risk,            color: '#16a34a' },
                  { label: 'Watch — mean CSR',          val: (R.ctg_specific as any).csr_by_class.watch,               color: '#d97706' },
                  { label: 'High Risk — mean CSR',      val: (R.ctg_specific as any).csr_by_class.high_risk,           color: '#dc2626' },
                  { label: 'Avg recovery (s) — Low',    val: (R.ctg_specific as any).csr_by_class.recovery_low_s,      color: '#16a34a' },
                  { label: 'Avg recovery (s) — High',   val: (R.ctg_specific as any).csr_by_class.recovery_high_s,     color: '#dc2626' },
                  { label: 'AUC (CSR → high-risk)',     val: (R.ctg_specific as any).csr_by_class.auc,                 color: '#374151' },
                  { label: 'Corr. with pH',             val: (R.ctg_specific as any).csr_by_class.corr_ph,             color: '#374151' },
                ].map(({ label, val, color }) => (
                  <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                    <span style={{ color: '#374151' }}>{label}</span>
                    <strong style={{ color }}>{val != null ? fmt(val as number, 3) : '—'}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rn-note">Run training to populate CSR-by-class validation.</div>
            )}
          </div>

        </div>

        {/* Signal quality subgroups */}
        <div style={{ marginTop: 20 }}>
          <div style={{ fontWeight: 700, color: '#374151', marginBottom: 10, fontSize: 14 }}>Signal Quality Subgroup Performance</div>
          <p className="rn-note" style={{ marginBottom: 12 }}>
            Performance split by signal quality (good / acceptable / poor). A robust model should become
            more uncertain — not overconfident — when signal is noisy. Poor-signal accuracy drop is expected;
            a rising uncertainty rate confirms the model is self-aware about data quality.
          </p>
          {(R.signal_quality_subgroups as any) ? (
            <div className="rn-table-wrap">
              <table className="rn-table">
                <thead>
                  <tr><th>Signal Quality</th><th>N</th><th>Accuracy</th><th>High-Risk Recall</th><th>Uncertainty Rate</th></tr>
                </thead>
                <tbody>
                  {(['good','acceptable','poor'] as const).map(group => {
                    const g = (R.signal_quality_subgroups as any)[group]
                    return g ? (
                      <tr key={group}>
                        <td><span className={`rn-class-pill rn-class-pill--${group === 'good' ? 'normal' : group === 'acceptable' ? 'watch' : 'high'}`}>{group.charAt(0).toUpperCase() + group.slice(1)}</span></td>
                        <td>{g.n ?? '—'}</td>
                        <td className={g.accuracy >= 0.80 ? 'rn-td--good' : g.accuracy >= 0.60 ? 'rn-td--warn' : 'rn-td--bad'}>{pct(g.accuracy)}</td>
                        <td className={g.high_risk_recall >= 0.75 ? 'rn-td--good' : g.high_risk_recall >= 0.50 ? 'rn-td--warn' : 'rn-td--bad'}>{pct(g.high_risk_recall)}</td>
                        <td>{pct(g.uncertainty_rate)}</td>
                      </tr>
                    ) : null
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="rn-note">Signal quality subgroup performance will appear here after training. Run <code>python train_adaptive.py</code>.</div>
          )}
        </div>
      </section>

      {/* ── Architecture ──────────────────────────────────────────────────── */}
      <section className="rn-card">
        <h3 className="rn-card__title">TOPQUA Architecture — {arch.layers?.length ?? 5} Layers</h3>
        <p className="rn-note" style={{ marginBottom: 20 }}>
          SMOTE-balanced stacked ensemble. SMOTE applied to training fold only (no leakage).
          Optuna tunes XGBoost hyperparameters. OOF stacking trains the meta-learner on
          out-of-fold base model predictions for unbiased combination weights.
        </p>
        <div className="rn-arch">
          {(arch.layers ?? []).map((layer: any, i: number) => (
            <div key={i} className="rn-arch__step">
              <div className="rn-arch__connector" style={{ background: i === 0 ? 'transparent' : LAYER_COLORS[i - 1] }} />
              <div className="rn-arch__box" style={{ borderTopColor: LAYER_COLORS[i] }}>
                <div className="rn-arch__num" style={{ background: LAYER_COLORS[i] }}>{i + 1}</div>
                <div className="rn-arch__name">{layer.name}</div>
                <div className="rn-arch__model">{layer.model}</div>
                <div className="rn-arch__features">
                  {(layer.features ?? []).map((f: string) => (
                    <span key={f} className="rn-arch__feat">{f}</span>
                  ))}
                </div>
                <div className="rn-arch__rationale">{layer.rationale}</div>
              </div>
            </div>
          ))}
        </div>
        <div className="rn-policy-row">
          <div className="rn-policy rn-policy--blue">
            <div className="rn-policy__label">Label Policy</div>
            <div className="rn-policy__val">{arch.label_policy ?? R.label_policy}</div>
          </div>
          <div className="rn-policy rn-policy--green">
            <div className="rn-policy__label">Split Policy</div>
            <div className="rn-policy__val">{arch.split_policy}</div>
          </div>
          <div className="rn-policy rn-policy--purple">
            <div className="rn-policy__label">SMOTE</div>
            <div className="rn-policy__val">{arch.smote ?? 'SMOTE on training fold only'}</div>
          </div>
        </div>
      </section>

      {/* ── Dataset ──────────────────────────────────────────────────────── */}
      <section className="rn-card">
        <h3 className="rn-card__title">Dataset — CTU-CHB/CTU-UHB Intrapartum CTG (PhysioNet)</h3>
        <div className="rn-dataset-grid">
          <div className="rn-ds-block">
            <div className="rn-ds-block__val">{R.n_records_total ?? '—'}</div>
            <div className="rn-ds-block__lab">Total records</div>
          </div>
          <div className="rn-ds-block">
            <div className="rn-ds-block__val">{dist.normal_0 ?? '—'}</div>
            <div className="rn-ds-block__lab">Normal (pH ≥ 7.15)</div>
            <div className="rn-ds-block__pct">{R.n_records_labeled ? pct((dist.normal_0 ?? 0) / R.n_records_labeled) : '—'}</div>
          </div>
          <div className="rn-ds-block">
            <div className="rn-ds-block__val" style={{ color: '#f59e0b' }}>{dist.watch_1 ?? '—'}</div>
            <div className="rn-ds-block__lab">Watch (pH 7.05–7.15)</div>
            <div className="rn-ds-block__pct">{R.n_records_labeled ? pct((dist.watch_1 ?? 0) / R.n_records_labeled) : '—'}</div>
          </div>
          <div className="rn-ds-block">
            <div className="rn-ds-block__val" style={{ color: '#ef4444' }}>{dist.high_risk_2 ?? '—'}</div>
            <div className="rn-ds-block__lab">High Risk (pH &lt; 7.05)</div>
            <div className="rn-ds-block__pct">{R.n_records_labeled ? pct((dist.high_risk_2 ?? 0) / R.n_records_labeled) : '—'}</div>
          </div>
          <div className="rn-ds-block">
            <div className="rn-ds-block__val">{R.n_excluded ?? '—'}</div>
            <div className="rn-ds-block__lab">Excluded (no outcome)</div>
          </div>
        </div>

        {/* Class distribution bar */}
        {R.n_records_labeled && (
          <div style={{ marginTop: 16 }}>
            <div style={{ display: 'flex', height: 20, borderRadius: 6, overflow: 'hidden', gap: 1 }}>
              <div style={{ background: '#16a34a', flex: dist.normal_0, minWidth: 2 }} title={`Normal: ${dist.normal_0}`} />
              <div style={{ background: '#f59e0b', flex: dist.watch_1, minWidth: 2 }} title={`Watch: ${dist.watch_1}`} />
              <div style={{ background: '#ef4444', flex: dist.high_risk_2, minWidth: 2 }} title={`High risk: ${dist.high_risk_2}`} />
            </div>
            <div style={{ display: 'flex', gap: 16, marginTop: 6 }}>
              {[{c:'#16a34a',l:'Normal'},{c:'#f59e0b',l:'Watch'},{c:'#ef4444',l:'High Risk'}].map(x => (
                <div key={x.l} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: '#6b7280' }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: x.c, display: 'inline-block' }} />
                  {x.l}
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="rn-note rn-note--clinical" style={{ marginTop: 14 }}>
          <strong>Class imbalance:</strong> 81% normal, 12% watch, 7% high-risk.
          SMOTE oversamples the minority classes in each training fold — validation and test sets
          are never augmented. Binary (at-risk vs normal) task collapses watch+high-risk into one
          positive class, yielding a more tractable 81/19 split.
        </div>
      </section>

      {/* ── Feature Importances ───────────────────────────────────────────── */}
      <section className="rn-card">
        <h3 className="rn-card__title">Feature Importance — XGBoost (top 20) vs ExtraTrees (top 20)</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
          <div>
            <div style={{ fontWeight: 600, color: '#3b82f6', marginBottom: 8, fontSize: 13 }}>XGBoost Gain</div>
            <div className="rn-imp-bars">
              {imps.slice(0, 20).map((item: any) => (
                <div key={item.feature} className="rn-imp-row">
                  <div className="rn-imp-row__label">{item.feature}</div>
                  <div className="rn-imp-row__bar-wrap">
                    <div className="rn-imp-row__bar rn-imp-row__bar--blue"
                      style={{ width: `${((item.importance ?? 0) / maxImpXgb) * 100}%` }} />
                  </div>
                  <div className="rn-imp-row__val">{((item.importance ?? 0) * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <div style={{ fontWeight: 600, color: '#22c55e', marginBottom: 8, fontSize: 13 }}>ExtraTrees Impurity</div>
            <div className="rn-imp-bars">
              {etImps.slice(0, 20).map((item: any) => (
                <div key={item.feature} className="rn-imp-row">
                  <div className="rn-imp-row__label">{item.feature}</div>
                  <div className="rn-imp-row__bar-wrap">
                    <div className="rn-imp-row__bar rn-imp-row__bar--green"
                      style={{ width: `${((item.importance ?? 0) / maxImpEt) * 100}%` }} />
                  </div>
                  <div className="rn-imp-row__val">{((item.importance ?? 0) * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── Expert Importances ────────────────────────────────────────────── */}
      {Object.keys(expI).length > 0 && (
        <section className="rn-card">
          <h3 className="rn-card__title">Domain Expert Feature Importance — ReserveNet Specialists</h3>
          <div className="rn-exp-grid">
            {Object.entries(expI).map(([name, items]) => {
              const maxV = Math.max(...(items as any[]).map((i: any) => i.importance), 0.001)
              const color = EXPERT_COLORS[name] ?? '#6b7280'
              return (
                <div key={name} className="rn-exp-block" style={{ borderTopColor: color }}>
                  <div className="rn-exp-block__head" style={{ color }}>{EXPERT_LABELS[name] ?? name}</div>
                  {(items as any[]).map((item: any) => (
                    <div key={item.feature} className="rn-exp-row">
                      <span className="rn-exp-row__label">{item.feature}</span>
                      <div className="rn-exp-row__bar-wrap">
                        <div className="rn-exp-row__bar"
                          style={{ width: `${(item.importance / maxV) * 100}%`, background: color }} />
                      </div>
                      <span className="rn-exp-row__val">{((item.importance ?? 0) * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )
            })}
          </div>
        </section>
      )}

      {/* ── ML Integrity Checklist ────────────────────────────────────────── */}
      <section className="rn-card">
        <h3 className="rn-card__title">ML Integrity &amp; Anti-Overfitting Checklist</h3>
        <div className="rn-gen-grid">
          {[
            { icon: '✅', title: 'Record-level split', body: 'Train/val/test split at patient record level — windows from the same recording never appear across splits. Zero within-patient leakage.' },
            { icon: '✅', title: 'SMOTE — train fold only', body: 'Synthetic minority oversampling applied exclusively to each training fold. Validation and test sets contain only real CTU-CHB records.' },
            { icon: '✅', title: 'OOF stacked meta-learner', body: 'Meta-learner trained on out-of-fold predictions from 5-fold CV — never sees predictions from models it was trained on. Unbiased stacking.' },
            { icon: '✅', title: 'Optuna on validation AUROC', body: 'Hyperparameter search uses validation set AUROC only. Test set is never touched during tuning. Early stopping prevents depth-overfitting.' },
            { icon: '✅', title: 'Youden threshold on validation', body: 'Decision threshold tuned to maximise sensitivity + specificity on the held-out validation set. Test set used for evaluation only.' },
            { icon: '✅', title: 'Temperature scaling on validation', body: 'Probability calibration (temperature T) fit on validation logits only. Test probabilities are never used for calibration.' },
            { icon: '✅', title: 'pH-only labels (no circular features)', body: 'All labels derived from cord blood pH / base deficit / Apgar. Feature columns are pure signal measurements with no label-derived components.' },
            { icon: '✅', title: 'Imputer/scaler fit on train only', body: 'Median imputer and RobustScaler fit exclusively on training indices inside each CV fold. Validation and test sets transformed with train statistics only.' },
            { icon: '📊', title: 'Bootstrap CIs (500 iter)', body: '500 stratified bootstrap resamplings of the test set quantify uncertainty in all reported binary metrics — wide CIs flag small test size.' },
            { icon: '❌', title: 'No synthetic fallback allowed', body: 'Training crashes with RuntimeError if real CTU-CHB data is unavailable. fetal_health.csv, UCI, Kaggle, and synthetic data are permanently excluded.' },
          ].map(({ icon, title, body }) => (
            <div key={title} className="rn-gen-card">
              <div className="rn-gen-card__icon">{icon}</div>
              <div className="rn-gen-card__title">{title}</div>
              <div className="rn-gen-card__body">{body}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Safety ───────────────────────────────────────────────────────── */}
      <section className="rn-card rn-card--safety">
        <h3 className="rn-card__title">Safety &amp; Research Status</h3>
        <p>
          <strong>FetalyzeAI TOPQUA is a research-stage CTG second-reader</strong> trained on
          552 real CTU-CHB/CTU-UHB intrapartum recordings. It analyses FHR baseline, short- and
          long-term variability, deceleration burden, late-deceleration likelihood, contraction
          stress response, fetal reserve, signal quality, and uncertainty. It does <em>not</em> diagnose
          fetal distress, recommend treatment, or replace a qualified obstetrician or midwife.
          All outputs require expert clinical interpretation before any action is taken.
        </p>
        <div className="rn-refs">
          <span className="rn-ref">Chudáček et al. (2014) BMC Pregnancy and Childbirth 14:16</span>
          <span className="rn-ref">CTU-CHB Intrapartum CTG Database v1.0.0 — PhysioNet ODC-BY-1.0</span>
          <span className="rn-ref">FIGO Intrapartum CTG Guidelines (2015)</span>
          <span className="rn-ref">Model version: {R.model_version ?? '—'} · Trained: {R.training_date?.slice(0, 10) ?? '—'}</span>
        </div>
      </section>

    </div>
  )
}
