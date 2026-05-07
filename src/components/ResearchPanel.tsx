import results from '../../ctu_model_results.json'

const FEATURE_LABELS: Record<string, string> = {
  std_fhr: 'FHR Variability (SD)',
  accels_per_30min: 'Accelerations / 30 min',
  stv: 'Short-term Variability (STV)',
  bradycardia_frac: 'Bradycardia Fraction',
  baseline_fhr: 'Baseline FHR (bpm)',
  ltv: 'Long-term Variability (LTV)',
  max_decel_depth: 'Max Deceleration Depth (bpm)',
  n_accels: 'Number of Accelerations',
  n_contractions: 'Number of Contractions',
  mean_decel_dur_s: 'Mean Deceleration Duration (s)',
  n_decels: 'Number of Decelerations',
  decels_per_30min: 'Decelerations / 30 min',
  mean_decel_depth: 'Mean Deceleration Depth (bpm)',
  mean_fhr: 'Mean FHR (bpm)',
  tachycardia_frac: 'Tachycardia Fraction',
  contractions_per_10min: 'Contractions / 10 min',
}

function pct(v: number) { return `${(v * 100).toFixed(1)}%` }
function fmt(v: number, dp = 4) { return v.toFixed(dp) }

export function ResearchPanel() {
  const ds   = results.dataset
  const ho   = results.holdout
  const cv   = results.cv5
  const fi   = results.feature_importance
  const abl  = results.ablation
  const lc   = results.learning_curve
  const cal  = results.calibration_curve
  const mc   = results.model_comparison
  const mv2  = results.model_v2
  const phH  = results.ph_histogram.filter(b => b.count > 0)
  const stvBph = results.stv_by_ph
  const maxPh  = Math.max(...phH.map(b => b.count))
  const maxFI  = Math.max(...fi.map(f => f.importance))
  const fullAUC = abl[0]?.auc_mean ?? 0
  const maxLC  = Math.max(...lc.map(r => r.train_auc))

  return (
    <div className="rp">

      {/* ── Hero ── */}
      <section className="rp-hero">
        <div>
          <h2>FetalyzeAI v2 — Model Evidence &amp; Anti-Overfitting Report</h2>
          <p>
            Evaluated on all <strong>552 real CTU-CHB intrapartum records</strong> with real
            cord-blood pH outcomes. Model v2 removes hand-crafted composite features, uses a
            soft-vote ensemble with strong regularisation, and reports a <strong>learning
            curve</strong> to diagnose overfitting honestly.
          </p>
          <div className="rp-sources">
            <span className="rp-badge rp-badge--blue">CTU-CHB · Chudáček et al. 2014</span>
            <span className="rp-badge rp-badge--teal">PhysioNet WFDB · Zenodo 19510407</span>
            <span className="rp-badge rp-badge--green">552 records · 552/552 pH labelled</span>
            <span className="rp-badge rp-badge--amber">Model v2: no composite leakage</span>
          </div>
        </div>
        <div className="rp-hero__warn">
          <strong>Clinical safety notice</strong>
          <span>Research prototype only. Not a diagnosis. Not a treatment recommendation. Requires clinician review at all times.</span>
        </div>
      </section>

      {/* ── What changed in v2 ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">What changed in Model v2 — Anti-Overfitting Design</h3>
        <div className="rp-v2-grid">
          <div className="rp-v2-block rp-v2-block--bad">
            <strong>Model v1 problems</strong>
            <ul>
              <li><span className="rp-tag rp-tag--red">Leakage</span> <code>decel_burden</code> = depth × count — a near-direct label proxy (AUROC 0.98 was inflated)</li>
              <li><span className="rp-tag rp-tag--red">Leakage</span> <code>fetal_reserve_score</code> hand-crafted from label-correlated signals</li>
              <li><span className="rp-tag rp-tag--red">Overfit</span> Single XGBoost, depth 3, 300 trees — train/val gap 0.24</li>
              <li><span className="rp-tag rp-tag--red">Overfit</span> No early stopping, gain-based importance unreliable</li>
            </ul>
          </div>
          <div className="rp-v2-block rp-v2-block--good">
            <strong>Model v2 fixes</strong>
            <ul>
              <li><span className="rp-tag rp-tag--green">Fixed</span> Composites removed — 16 atomic signal features only</li>
              <li><span className="rp-tag rp-tag--green">Fixed</span> Soft-vote ensemble: LR + XGBoost (tight) + Random Forest</li>
              <li><span className="rp-tag rp-tag--green">Fixed</span> Strong regularisation: min_child_weight=8, gamma=2, L2=8, C=0.1</li>
              <li><span className="rp-tag rp-tag--green">Fixed</span> Decision threshold 0.35 — optimised for clinical sensitivity</li>
              <li><span className="rp-tag rp-tag--green">Fixed</span> Train/val gap reduced from 0.24 → 0.13 (learning curve below)</li>
            </ul>
          </div>
        </div>
        <p className="rp-note rp-note--clinical">
          <strong>Honest result:</strong> CTG signal features alone achieve AUROC ≈ 0.67–0.69 for predicting cord-blood acidosis (pH &lt; 7.15). This is consistent with clinical literature — FIGO 2015 notes CTG has moderate discriminative ability and requires clinician interpretation. The model is tuned for <strong>high sensitivity (88–93%)</strong> at the cost of specificity (23%).
        </p>
      </section>

      {/* ── Dataset overview ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Dataset — CTU-CHB Intrapartum CTG Database (552 records)</h3>
        <div className="rp-dataset-grid">
          <div className="rp-stat"><span className="rp-stat__val">552</span><span className="rp-stat__lbl">Total records (all used)</span></div>
          <div className="rp-stat"><span className="rp-stat__val">552 / 552</span><span className="rp-stat__lbl">Real cord-blood pH available</span></div>
          <div className="rp-stat rp-stat--red"><span className="rp-stat__val">{ds.acidosis}</span><span className="rp-stat__lbl">Acidosis (pH &lt; 7.05)</span></div>
          <div className="rp-stat rp-stat--amber"><span className="rp-stat__val">{ds.borderline}</span><span className="rp-stat__lbl">Borderline (pH 7.05–7.15)</span></div>
          <div className="rp-stat rp-stat--green"><span className="rp-stat__val">{ds.normal_ph}</span><span className="rp-stat__lbl">Normal pH (≥ 7.15)</span></div>
          <div className="rp-stat"><span className="rp-stat__val">{ds.ph_mean} ± {ds.ph_std}</span><span className="rp-stat__lbl">Mean cord-blood pH ± SD</span></div>
        </div>
        <div className="rp-label-split">
          <div className="rp-label-bar">
            <div className="rp-label-fill rp-label-fill--safe" style={{ width: `${(ds.n_safe / 552) * 100}%` }}>
              Safe {ds.n_safe} ({((ds.n_safe / 552) * 100).toFixed(0)}%)
            </div>
            <div className="rp-label-fill rp-label-fill--risk" style={{ width: `${(ds.n_atrisk / 552) * 100}%` }}>
              At-risk {ds.n_atrisk} ({((ds.n_atrisk / 552) * 100).toFixed(0)}%)
            </div>
          </div>
          <p className="rp-note" style={{ marginTop: 6 }}>Binary label: At-risk = pH &lt; 7.15 OR Apgar-1 &lt; 7 (no hand-crafted signal composites in label)</p>
        </div>
      </section>

      {/* ── pH histogram ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Cord-blood pH Distribution (n = 552)</h3>
        <div className="rp-ph-zones">
          <span className="rp-zone rp-zone--red">Acidosis &lt; 7.05</span>
          <span className="rp-zone rp-zone--amber">Borderline 7.05–7.15</span>
          <span className="rp-zone rp-zone--green">Normal ≥ 7.15</span>
        </div>
        <div className="rp-histogram">
          {phH.map(b => {
            const zone = b.bin < 7.05 ? 'red' : b.bin < 7.15 ? 'amber' : 'green'
            return (
              <div key={b.bin} className="rp-hist-col">
                <span className="rp-hist-count">{b.count}</span>
                <div className={`rp-hist-bar rp-hist-bar--${zone}`}
                  style={{ height: `${Math.round((b.count / maxPh) * 120)}px` }} />
                <span className="rp-hist-label">{b.bin.toFixed(2)}</span>
              </div>
            )
          })}
        </div>
      </section>

      {/* ── Learning curve — overfitting diagnosis ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Learning Curve — Overfitting Diagnosis</h3>
        <p className="rp-note" style={{ marginBottom: 16 }}>
          Train vs. validation AUROC as training set size grows (5-fold CV mean). A large gap between curves indicates overfitting. Model v2 (LR+RF ensemble) reduces the gap from 0.24 (v1 XGBoost) to 0.13 through stronger regularisation.
        </p>
        <div className="rp-lc">
          <div className="rp-lc__legend">
            <span className="rp-lc__dot rp-lc__dot--train" /> Train AUROC (memorisation)
            <span className="rp-lc__dot rp-lc__dot--val" style={{ marginLeft: 16 }} /> Val AUROC (generalisation)
          </div>
          <div className="rp-lc__chart">
            {lc.map(row => (
              <div key={row.frac} className="rp-lc__col">
                <div className="rp-lc__bars">
                  <div className="rp-lc__bar-wrap">
                    <div className="rp-lc__bar rp-lc__bar--train"
                      style={{ height: `${(row.train_auc / maxLC) * 100}px` }}>
                      <span className="rp-lc__bval">{row.train_auc.toFixed(3)}</span>
                    </div>
                  </div>
                  <div className="rp-lc__bar-wrap">
                    <div className="rp-lc__bar rp-lc__bar--val"
                      style={{ height: `${(row.val_auc / maxLC) * 100}px` }}>
                      <span className="rp-lc__bval">{row.val_auc.toFixed(3)}</span>
                    </div>
                  </div>
                </div>
                <div className="rp-lc__gap">gap {row.gap.toFixed(3)}</div>
                <div className="rp-lc__xlabel">{Math.round(row.frac * 100)}%<br /><small>n≈{row.n}</small></div>
              </div>
            ))}
          </div>
          <p className="rp-note rp-note--clinical">
            <strong>Interpretation:</strong> The gap converges as training set grows but remains ~0.13 at 100%, confirming mild-to-moderate overfitting rather than severe memorisation. Adding more clinical records would likely improve val AUROC. The validation plateau at ~0.68 reflects the genuine discriminative ceiling of CTG signals for acidosis prediction.
          </p>
        </div>
      </section>

      {/* ── Model comparison ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Model Comparison — 5-fold CV (threshold 0.35)</h3>
        <p className="rp-note" style={{ marginBottom: 16 }}>
          All models trained on the same 16 atomic FHR/UC signal features with identical preprocessing. AUROC is threshold-independent; F1 and Sensitivity use threshold = 0.35 (calibrated for clinical sensitivity priority).
        </p>
        <div className="rp-mc">
          {mc.map((m, i) => {
            const isBest = i === mc.length - 1
            return (
              <div key={m.model} className={`rp-mc__row ${isBest ? 'rp-mc__row--best' : ''}`}>
                <div className="rp-mc__name">
                  {isBest && <span className="rp-tag rp-tag--blue">Selected</span>}
                  {m.model}
                </div>
                <div className="rp-mc__auc">
                  <div className="rp-mc__auc-val">{fmt(m.auroc, 4)}</div>
                  <div className="rp-mc__auc-sd">±{fmt(m.std_auc, 4)}</div>
                  <div className="rp-mc__bar-wrap">
                    <div className={`rp-mc__bar ${isBest ? 'rp-mc__bar--best' : ''}`}
                      style={{ width: `${(m.auroc / 0.75) * 100}%` }} />
                  </div>
                  <div className="rp-mc__lbl">AUROC</div>
                </div>
                <div className="rp-mc__cell">
                  <div className="rp-mc__big">{pct(m.f1)}</div>
                  <div className="rp-mc__lbl">F1</div>
                </div>
                <div className="rp-mc__cell">
                  <div className={`rp-mc__big ${m.sens >= 0.80 ? 'rp-mc__big--hi' : ''}`}>{pct(m.sens)}</div>
                  <div className="rp-mc__lbl">Sensitivity</div>
                </div>
              </div>
            )
          })}
        </div>
        <p className="rp-note">
          All single models achieve similar AUROC (≈0.67). The ensemble wins on <strong>sensitivity</strong> (88% vs 15–54%) by averaging probability outputs and applying a lower decision threshold — critical for a screening tool where missing an at-risk case is more dangerous than a false alarm.
        </p>
      </section>

      {/* ── Key metrics ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Ensemble Performance — Held-out Test Set (n = {ho.n_test})</h3>
        <p className="rp-note" style={{ marginBottom: 16 }}>
          80/20 stratified split. Decision threshold = 0.35 (lower than default 0.5 to prioritise sensitivity). Preprocessing fit only on training fold — leakage-free.
        </p>
        <div className="rp-perf-grid">
          {[
            { label: 'AUROC', val: fmt(ho.auroc, 4), note: 'Threshold-independent discrimination', hi: false },
            { label: 'AUPRC', val: fmt(ho.auprc, 4), note: 'Precision-recall area (class imbalanced)', hi: false },
            { label: 'Sensitivity', val: pct(ho.sensitivity), note: 'At-risk recall — clinical priority', hi: ho.sensitivity >= 0.85 },
            { label: 'Specificity', val: pct(ho.specificity), note: 'Safe recall (low = many false alarms)', hi: false },
            { label: 'F1 Score', val: fmt(ho.f1, 4), note: 'Harmonic mean precision/recall', hi: false },
            { label: 'Balanced Acc.', val: pct(ho.balanced_accuracy), note: 'Accounts for class imbalance', hi: false },
            { label: 'Brier Score', val: fmt(ho.brier, 4), note: 'Calibration quality (lower = better)', hi: false },
            { label: 'Precision', val: pct(ho.precision), note: 'PPV — positive predictive value', hi: false },
          ].map(m => (
            <div key={m.label} className={`rp-metric ${m.hi ? 'rp-metric--hi' : ''}`}>
              <span className="rp-metric__val">{m.val}</span>
              <span className="rp-metric__label">{m.label}</span>
              <span className="rp-metric__note">{m.note}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Confusion matrix ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Confusion Matrix — Held-out Test Set (threshold = 0.35)</h3>
        <div className="rp-cm-wrap">
          <div className="rp-cm">
            <div className="rp-cm__corner" />
            <div className="rp-cm__head">Predicted: Safe</div>
            <div className="rp-cm__head">Predicted: At-risk</div>
            <div className="rp-cm__row-head">Actual: Safe</div>
            <div className="rp-cm__cell rp-cm__cell--tn">
              <span className="rp-cm__big">{ho.tn}</span>
              <span className="rp-cm__tag">True Negative</span>
            </div>
            <div className="rp-cm__cell rp-cm__cell--fp">
              <span className="rp-cm__big">{ho.fp}</span>
              <span className="rp-cm__tag">False Positive (over-alert)</span>
            </div>
            <div className="rp-cm__row-head">Actual: At-risk</div>
            <div className="rp-cm__cell rp-cm__cell--fn">
              <span className="rp-cm__big">{ho.fn}</span>
              <span className="rp-cm__tag">False Negative ⚠</span>
            </div>
            <div className="rp-cm__cell rp-cm__cell--tp">
              <span className="rp-cm__big">{ho.tp}</span>
              <span className="rp-cm__tag">True Positive</span>
            </div>
          </div>
          <div className="rp-cm-notes">
            <div className="rp-cm-note">
              <strong>False negatives (missed) = {ho.fn}</strong>
              <p>Most dangerous errors. At threshold 0.35, missed-at-risk rate = {pct(1 - ho.sensitivity)}. The ensemble is tuned to minimise these.</p>
            </div>
            <div className="rp-cm-note">
              <strong>False positives (over-alert) = {ho.fp}</strong>
              <p>Safe cases flagged as at-risk — a deliberate trade-off. High false positive rate is expected when prioritising sensitivity. Clinician review is always required.</p>
            </div>
            <div className="rp-cm-note rp-cm-note--warn">
              <strong>Clinical interpretation</strong>
              <p>A negative result does not rule out fetal compromise. All model outputs require bedside clinician assessment.</p>
            </div>
          </div>
        </div>
      </section>

      {/* ── 5-Fold CV ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">5-Fold Stratified CV — All 552 Records (Ensemble)</h3>
        <div className="rp-table-wrap">
          <table className="rp-table">
            <thead>
              <tr>
                <th>Fold</th><th>AUROC</th><th>F1</th><th>Sensitivity</th><th>Specificity</th><th>Balanced Acc.</th>
              </tr>
            </thead>
            <tbody>
              {cv.fold_auc.map((auc, i) => (
                <tr key={i}>
                  <td className="rp-table__fold">Fold {i + 1}</td>
                  <td>{fmt(auc, 4)}</td>
                  <td>{fmt(cv.fold_f1[i], 4)}</td>
                  <td>{pct(cv.fold_sens[i])}</td>
                  <td>{pct(cv.fold_spec[i])}</td>
                  <td>{pct(cv.fold_bal[i])}</td>
                </tr>
              ))}
              <tr className="rp-table__mean">
                <td>Mean ± SD</td>
                <td>{fmt(cv.mean_auc, 4)} ± {fmt(cv.std_auc, 4)}</td>
                <td>{fmt(cv.mean_f1, 4)} ± {fmt(cv.std_f1, 4)}</td>
                <td>{pct(cv.mean_sens)} ± {pct(cv.std_sens)}</td>
                <td>{pct(cv.mean_spec)} ± {pct(cv.std_spec)}</td>
                <td>{pct(cv.mean_bal)} ± {pct(cv.std_bal)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Feature importance (LR coefficients) ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Feature Importance — Logistic Regression |Coefficient| (Interpretable)</h3>
        <p className="rp-note" style={{ marginBottom: 16 }}>
          Absolute logistic regression coefficients after robust scaling — directly interpretable as the weight each feature contributes to the log-odds of "at-risk". Unlike XGBoost gain, these are stable across runs.
        </p>
        <div className="rp-fi">
          {fi.map(f => (
            <div key={f.feature} className="rp-fi__row">
              <span className="rp-fi__label">{FEATURE_LABELS[f.feature] ?? f.feature}</span>
              <div className="rp-fi__bar-wrap">
                <div className="rp-fi__bar" style={{ width: `${(f.importance / maxFI) * 100}%` }} />
              </div>
              <span className="rp-fi__val">{f.importance.toFixed(3)}</span>
            </div>
          ))}
        </div>
        <p className="rp-note rp-note--clinical">
          <strong>Clinical interpretation:</strong> FHR variability (SD, STV, LTV) and accelerations top the list — consistent with FIGO guidance that reduced variability and loss of accelerations are primary markers of fetal compromise. Decelerations contribute but rank lower in isolation; their clinical impact is captured via the count and depth features.
        </p>
      </section>

      {/* ── Ablation studies ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Ablation Studies — Feature Group Contribution (5-fold CV AUROC)</h3>
        <p className="rp-note" style={{ marginBottom: 16 }}>
          Each row removes or isolates one feature group. Delta vs. full model ({fmt(fullAUC, 4)}). Note: ablation uses Logistic Regression for stable estimates.
        </p>
        <div className="rp-abl">
          {abl.map((a, i) => {
            const delta = a.auc_mean - fullAUC
            const isFull = i === 0
            const isWinner = !isFull && delta === Math.max(...abl.slice(1).map(x => x.auc_mean - fullAUC))
            const isWorst  = !isFull && delta === Math.min(...abl.slice(1).map(x => x.auc_mean - fullAUC))
            return (
              <div key={a.group} className={`rp-abl__row ${isFull ? 'rp-abl__row--full' : ''} ${isWorst ? 'rp-abl__row--worst' : ''}`}>
                <div className="rp-abl__name">
                  {isFull   && <span className="rp-abl__pill rp-abl__pill--blue">Baseline</span>}
                  {isWorst  && <span className="rp-abl__pill rp-abl__pill--red">Most critical</span>}
                  {isWinner && <span className="rp-abl__pill rp-abl__pill--green">Can remove safely</span>}
                  {a.group}
                </div>
                <div className="rp-abl__feats">{a.n_feats} feats</div>
                <div className="rp-abl__auc">{fmt(a.auc_mean, 4)} ± {fmt(a.auc_std, 4)}</div>
                <div className={`rp-abl__delta ${delta < -0.005 ? 'rp-abl__delta--neg' : delta > 0.005 ? 'rp-abl__delta--pos' : ''}`}>
                  {isFull ? '—' : `${delta >= 0 ? '+' : ''}${fmt(delta, 4)}`}
                </div>
                <div className="rp-abl__bar-wrap">
                  <div className={`rp-abl__bar ${isFull ? 'rp-abl__bar--full' : isWorst ? 'rp-abl__bar--worst' : ''}`}
                    style={{ width: `${(a.auc_mean / 0.75) * 100}%` }} />
                </div>
              </div>
            )
          })}
        </div>
        <p className="rp-note rp-note--clinical">
          <strong>Key finding:</strong> Removing accelerations causes the largest single-group drop (−{fmt(fullAUC - (abl.find(a => a.group.includes('acceleration'))?.auc_mean ?? fullAUC), 4)} AUROC), confirming that loss of accelerations is the most informative signal. Interestingly, removing decelerations slightly <em>improves</em> AUROC (+{fmt((abl.find(a => a.group.includes('deceleration'))?.auc_mean ?? fullAUC) - fullAUC, 4)}), suggesting raw deceleration counts add noise in isolation. This reverses the v1 finding where `decel_burden` dominated — that was a composite leakage artefact.
        </p>
      </section>

      {/* ── Calibration ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Probability Calibration — Reliability Diagram</h3>
        <p className="rp-note" style={{ marginBottom: 16 }}>
          A perfectly calibrated model's predicted probabilities match observed frequencies (diagonal line). Points above the diagonal = model under-predicts risk; points below = over-predicts. Brier score = {fmt(ho.brier, 4)} (0 = perfect, 0.25 = random).
        </p>
        <div className="rp-cal">
          <div className="rp-cal__chart">
            <div className="rp-cal__perfect" title="Perfect calibration (diagonal)" />
            {cal.map((pt, i) => (
              <div key={i} className="rp-cal__pt-wrap"
                style={{ left: `${pt.mean_pred * 100}%`, bottom: `${pt.frac_pos * 100}%` }}>
                <div className="rp-cal__pt" title={`Predicted: ${pt.mean_pred} → Actual: ${pt.frac_pos}`} />
                <div className="rp-cal__pt-label">
                  pred {pt.mean_pred}<br />act {pt.frac_pos}
                </div>
              </div>
            ))}
            <div className="rp-cal__x-axis"><span>0.0</span><span>0.5</span><span>1.0</span></div>
            <div className="rp-cal__y-axis"><span>1.0</span><span>0.5</span><span>0.0</span></div>
          </div>
          <div className="rp-cal__table">
            <table className="rp-table">
              <thead><tr><th>Predicted prob.</th><th>Actual freq.</th><th>Δ</th></tr></thead>
              <tbody>
                {cal.map((pt, i) => {
                  const delta = pt.frac_pos - pt.mean_pred
                  return (
                    <tr key={i}>
                      <td>{pt.mean_pred.toFixed(3)}</td>
                      <td>{pt.frac_pos.toFixed(3)}</td>
                      <td className={delta > 0.05 ? 'rp-cal__over' : delta < -0.05 ? 'rp-cal__under' : ''}>{delta >= 0 ? '+' : ''}{delta.toFixed(3)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <p className="rp-note" style={{ marginTop: 10 }}>
              Model over-predicts risk at low probabilities (0.28 predicted → 0.09 actual) — a known issue with imbalanced datasets. Use AUROC for ranking, not raw probabilities, in clinical practice.
            </p>
          </div>
        </div>
      </section>

      {/* ── STV by pH ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Short-term Variability (STV) by pH Risk Group</h3>
        <p className="rp-note" style={{ marginBottom: 16 }}>
          STV = mean absolute beat-to-beat FHR difference at 4 Hz. Higher STV generally indicates better autonomic tone.
        </p>
        <div className="rp-stv">
          {stvBph.map(g => {
            const label = g.ph_bucket.replace('\n', ' ')
            const zone = label.includes('Acidosis') ? 'red' : label.includes('Borderline') ? 'amber' : label.includes('High Normal') ? 'blue' : 'green'
            return (
              <div key={g.ph_bucket} className={`rp-stv__group rp-stv__group--${zone}`}>
                <div className="rp-stv__label">{label}</div>
                <div className="rp-stv__val">{g.stv_mean?.toFixed(3) ?? '—'}</div>
                <div className="rp-stv__sd">± {g.stv_std?.toFixed(3) ?? '—'}</div>
                <div className="rp-stv__n">n = {g.count}</div>
                <div className="rp-stv__bar-wrap">
                  <div className={`rp-stv__bar rp-stv__bar--${zone}`}
                    style={{ width: `${((g.stv_mean ?? 0) / 1.0) * 100}%` }} />
                </div>
              </div>
            )
          })}
        </div>
        <p className="rp-note rp-note--clinical">
          <strong>Clinical note:</strong> STV values are slightly <em>higher</em> in the acidotic group — likely reflecting compensatory sympathetic activation in early acidosis, or signal artefact during decelerations. STV alone is a weak discriminator; the model's top features are FHR variability (SD) and acceleration rate.
        </p>
      </section>

      {/* ── Architecture ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Model Architecture — v2 Ensemble</h3>
        <div className="rp-arch-grid">
          <div className="rp-arch-block">
            <strong>Logistic Regression</strong>
            <ul>
              <li>C = 0.1 (strong L2 regularisation)</li>
              <li>class_weight = 'balanced'</li>
              <li>max_iter = 2000</li>
              <li>Role: stable, interpretable base predictor</li>
            </ul>
          </div>
          <div className="rp-arch-block">
            <strong>XGBoost (tight)</strong>
            <ul>
              <li>max_depth = 2, n_estimators = 80</li>
              <li>gamma = 3.0, reg_lambda = 8.0</li>
              <li>min_child_weight = 8</li>
              <li>Role: captures non-linear interactions</li>
            </ul>
          </div>
          <div className="rp-arch-block">
            <strong>Random Forest</strong>
            <ul>
              <li>n_estimators = 200, max_depth = 4</li>
              <li>min_samples_leaf = 8, max_features = sqrt</li>
              <li>class_weight = 'balanced'</li>
              <li>Role: variance reduction via bagging</li>
            </ul>
          </div>
          <div className="rp-arch-block">
            <strong>Ensemble + Decision threshold</strong>
            <ul>
              <li>Soft-vote average of 3 model probabilities</li>
              <li>Decision threshold = 0.35 (vs default 0.5)</li>
              <li>At threshold 0.35: Sens=88–93%, Spec=23%</li>
              <li>Preprocessing: median imputer + robust scaler</li>
            </ul>
          </div>
        </div>
        <div className="rp-oc-list">
          <strong>Overfitting controls applied:</strong>
          {(mv2.overfitting_controls as string[]).map((c: string, i: number) => (
            <span key={i} className="rp-oc-tag">{c}</span>
          ))}
        </div>
      </section>

      {/* ── References ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Data Sources &amp; References</h3>
        <div className="rp-refs">
          <div className="rp-ref"><span className="rp-ref__id">[1]</span><div><strong>CTU-CHB Intrapartum CTG Database</strong><br />Chudáček V et al. (2014). BMC Pregnancy and Childbirth, 14:16.<br /><span className="rp-ref__link">PhysioNet · https://doi.org/10.13026/C22013</span></div></div>
          <div className="rp-ref"><span className="rp-ref__id">[2]</span><div><strong>CTU-CHB Annotation Dataset (CTGDL / FHRMA)</strong><br />Petránek V et al. (2020). Zenodo · https://doi.org/10.5281/zenodo.19510407</div></div>
          <div className="rp-ref"><span className="rp-ref__id">[3]</span><div><strong>FIGO Consensus Guidelines on Intrapartum Fetal Monitoring</strong><br />Ayres-de-Campos D et al. (2015). Int J Gynecol Obstet 131(1):13–24.</div></div>
          <div className="rp-ref"><span className="rp-ref__id">[4]</span><div><strong>XGBoost: A Scalable Tree Boosting System</strong><br />Chen T &amp; Guestrin C (2016). ACM SIGKDD.</div></div>
        </div>
      </section>

    </div>
  )
}
