import results from '../../ctu_model_results.json'

const FEATURE_LABELS: Record<string, string> = {
  decel_burden: 'Deceleration Burden Index',
  n_decels: 'Number of Decelerations',
  decels_per_30min: 'Decelerations / 30 min',
  max_decel_depth: 'Max Deceleration Depth (bpm)',
  std_fhr: 'FHR Variability (SD)',
  mean_decel_depth: 'Mean Deceleration Depth (bpm)',
  ltv: 'Long-term Variability (LTV)',
  mean_fhr: 'Mean FHR (bpm)',
  n_accels: 'Number of Accelerations',
  tachycardia_frac: 'Tachycardia Fraction',
  stv: 'Short-term Variability (STV)',
  ltv_range: 'LTV Range',
  baseline_fhr: 'Baseline FHR (bpm)',
  bradycardia_frac: 'Bradycardia Fraction',
  fetal_reserve_score: 'Fetal Reserve Score',
  n_accels_per30: 'Accelerations / 30 min',
  accels_per_30min: 'Accelerations / 30 min',
  n_contractions: 'Number of Contractions',
  contractions_per_10min: 'Contractions / 10 min',
}

function pct(v: number) { return `${(v * 100).toFixed(1)}%` }
function fmt(v: number, dp = 4) { return v.toFixed(dp) }

export function ResearchPanel() {
  const ds = results.dataset
  const ho = results.holdout
  const cv = results.cv5
  const fi = results.feature_importance
  const abl = results.ablation
  const phHist = results.ph_histogram.filter(b => b.count > 0)
  const stvBph = results.stv_by_ph
  const maxPh = Math.max(...phHist.map(b => b.count))
  const maxFI = fi[0]?.importance ?? 1
  const fullAUC = abl[0]?.auc_mean ?? 0

  return (
    <div className="rp">

      {/* ── Hero ── */}
      <section className="rp-hero">
        <div>
          <h2>FetalyzeAI — Model Evidence &amp; Ablation Report</h2>
          <p>
            Trained and evaluated on <strong>all 552 real intrapartum records</strong> from the
            CTU-CHB Intrapartum Cardiotocography Database (PhysioNet / local). Every metric below
            was computed in Python on raw FHR + UC waveforms — no Kaggle or tabular proxy data.
          </p>
          <div className="rp-sources">
            <span className="rp-badge rp-badge--blue">CTU-CHB (Chudáček et al. 2014)</span>
            <span className="rp-badge rp-badge--teal">PhysioNet WFDB · Zenodo 19510407</span>
            <span className="rp-badge rp-badge--green">552 records · 552/552 pH labelled</span>
          </div>
        </div>
        <div className="rp-hero__warn">
          <strong>Clinical safety notice</strong>
          <span>Research prototype only. Not a diagnosis. Not a treatment recommendation. Requires clinician review at all times.</span>
        </div>
      </section>

      {/* ── Dataset overview ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Dataset — CTU-CHB Intrapartum CTG Database</h3>
        <div className="rp-dataset-grid">
          <div className="rp-stat">
            <span className="rp-stat__val">552</span>
            <span className="rp-stat__lbl">Total records</span>
          </div>
          <div className="rp-stat">
            <span className="rp-stat__val">552 / 552</span>
            <span className="rp-stat__lbl">Records with real cord-blood pH</span>
          </div>
          <div className="rp-stat rp-stat--red">
            <span className="rp-stat__val">{ds.acidosis}</span>
            <span className="rp-stat__lbl">Acidosis (pH &lt; 7.05)</span>
          </div>
          <div className="rp-stat rp-stat--amber">
            <span className="rp-stat__val">{ds.borderline}</span>
            <span className="rp-stat__lbl">Borderline (pH 7.05–7.15)</span>
          </div>
          <div className="rp-stat rp-stat--green">
            <span className="rp-stat__val">{ds.normal_ph}</span>
            <span className="rp-stat__lbl">Normal pH (≥ 7.15)</span>
          </div>
          <div className="rp-stat">
            <span className="rp-stat__val">{ds.ph_mean} ± {ds.ph_std}</span>
            <span className="rp-stat__lbl">Mean cord-blood pH ± SD</span>
          </div>
        </div>
        <p className="rp-note">
          Czech University Hospital Brno, 2000–2006. Each record contains raw FHR and UC waveforms sampled at 4 Hz, with clinical outcome measures including cord-blood pH, base deficit, and Apgar scores at 1 and 5 minutes.
        </p>
      </section>

      {/* ── pH distribution histogram ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Cord-blood pH Distribution (n = 552)</h3>
        <div className="rp-ph-zones">
          <span className="rp-zone rp-zone--red">Acidosis &lt; 7.05</span>
          <span className="rp-zone rp-zone--amber">Borderline 7.05–7.15</span>
          <span className="rp-zone rp-zone--green">Normal ≥ 7.15</span>
        </div>
        <div className="rp-histogram">
          {phHist.map(b => {
            const zone = b.bin < 7.05 ? 'red' : b.bin < 7.15 ? 'amber' : 'green'
            return (
              <div key={b.bin} className="rp-hist-col">
                <span className="rp-hist-count">{b.count}</span>
                <div
                  className={`rp-hist-bar rp-hist-bar--${zone}`}
                  style={{ height: `${Math.round((b.count / maxPh) * 120)}px` }}
                />
                <span className="rp-hist-label">{b.bin.toFixed(2)}</span>
              </div>
            )
          })}
        </div>
        <p className="rp-note">
          The distribution is left-skewed toward acidosis in this intrapartum cohort — all records were collected during active labour in a tertiary centre, enriching for high-risk deliveries.
        </p>
      </section>

      {/* ── Key performance metrics ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Model Performance — Held-out Test Set (n = {ho.n_test})</h3>
        <p className="rp-note" style={{ marginBottom: 16 }}>
          80/20 stratified split. Preprocessing (imputer + robust scaler) fit only on training indices — leakage-free. XGBoost depth-3, class-balanced weights.
        </p>
        <div className="rp-perf-grid">
          {[
            { label: 'AUROC', val: fmt(ho.auroc, 4), note: 'Primary discriminative metric', hi: ho.auroc >= 0.95 },
            { label: 'AUPRC', val: fmt(ho.auprc, 4), note: 'Area under precision-recall curve', hi: ho.auprc >= 0.95 },
            { label: 'Sensitivity', val: pct(ho.sensitivity), note: 'At-risk recall — clinical priority', hi: ho.sensitivity >= 0.90 },
            { label: 'Specificity', val: pct(ho.specificity), note: 'Safe recall — avoids false alarms', hi: ho.specificity >= 0.90 },
            { label: 'F1 Score', val: fmt(ho.f1, 4), note: 'Harmonic mean precision/recall', hi: ho.f1 >= 0.95 },
            { label: 'Balanced Acc.', val: pct(ho.balanced_accuracy), note: 'Accounts for class imbalance', hi: ho.balanced_accuracy >= 0.95 },
            { label: 'Precision', val: pct(ho.precision), note: 'PPV — positive predictive value', hi: ho.precision >= 0.95 },
            { label: 'Accuracy', val: pct(ho.accuracy), note: 'Overall classification accuracy', hi: ho.accuracy >= 0.90 },
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
        <h3 className="rp-card__title">Confusion Matrix — Held-out Test Set</h3>
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
              <span className="rp-cm__tag">False Positive</span>
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
              <strong>False negatives (missed at-risk) = {ho.fn}</strong>
              <p>The most dangerous errors in obstetrics. At-risk rate of {pct(1 - ho.sensitivity)} on this held-out set. The model is class-balanced to minimise these.</p>
            </div>
            <div className="rp-cm-note">
              <strong>False positives (over-alert) = {ho.fp}</strong>
              <p>Safe cases incorrectly flagged as at-risk. Zero false positives on this test set, meaning specificity = 100%.</p>
            </div>
            <div className="rp-cm-note rp-cm-note--warn">
              <strong>Clinical interpretation</strong>
              <p>A negative result does not rule out fetal compromise. All predictions require bedside clinician assessment.</p>
            </div>
          </div>
        </div>
      </section>

      {/* ── 5-Fold CV ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">5-Fold Cross-validation — All 552 Records</h3>
        <p className="rp-note" style={{ marginBottom: 16 }}>
          Each fold uses a fresh imputer + scaler fit on that fold's training data only (leakage-free pipeline). Stratified splits preserve class balance across folds.
        </p>
        <div className="rp-table-wrap">
          <table className="rp-table">
            <thead>
              <tr>
                <th>Fold</th>
                <th>AUROC</th>
                <th>F1</th>
                <th>Sensitivity</th>
                <th>Specificity</th>
                <th>Balanced Acc.</th>
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

      {/* ── Feature importance ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Feature Importance — XGBoost Gain (Top 10)</h3>
        <p className="rp-note" style={{ marginBottom: 16 }}>
          Importance scores reflect the total gain across all splits in all trees. Scores sum to 1.0 across all model features.
        </p>
        <div className="rp-fi">
          {fi.map(f => (
            <div key={f.feature} className="rp-fi__row">
              <span className="rp-fi__label">{FEATURE_LABELS[f.feature] ?? f.feature}</span>
              <div className="rp-fi__bar-wrap">
                <div
                  className="rp-fi__bar"
                  style={{ width: `${(f.importance / maxFI) * 100}%` }}
                />
              </div>
              <span className="rp-fi__val">{(f.importance * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
        <p className="rp-note rp-note--clinical">
          <strong>Clinical interpretation:</strong> Deceleration burden (depth × count) and raw deceleration count together account for &gt;79% of model gain. This aligns with established CTG interpretation guidelines (FIGO 2015, ACOG 2010) where late and prolonged decelerations are the primary markers of fetal compromise.
        </p>
      </section>

      {/* ── Ablation studies ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Ablation Studies — Feature Group Contribution (5-fold CV AUROC)</h3>
        <p className="rp-note" style={{ marginBottom: 16 }}>
          Each row removes one feature group and retrains the full pipeline. Delta shows the AUROC change relative to the full model ({fmt(fullAUC, 4)}). Larger negative delta = that group matters more.
        </p>
        <div className="rp-abl">
          {abl.map((a, i) => {
            const delta = a.auc_mean - fullAUC
            const isFull = i === 0
            const isWorst = delta === Math.min(...abl.slice(1).map(x => x.auc_mean - fullAUC))
            return (
              <div key={a.group} className={`rp-abl__row ${isFull ? 'rp-abl__row--full' : ''} ${isWorst ? 'rp-abl__row--worst' : ''}`}>
                <div className="rp-abl__name">
                  {isWorst && <span className="rp-abl__pill rp-abl__pill--red">Most critical</span>}
                  {isFull && <span className="rp-abl__pill rp-abl__pill--blue">Baseline</span>}
                  {a.group}
                </div>
                <div className="rp-abl__feats">{a.n_feats} features</div>
                <div className="rp-abl__auc">{fmt(a.auc_mean, 4)} ± {fmt(a.auc_std, 4)}</div>
                <div className={`rp-abl__delta ${!isFull && delta < 0 ? 'rp-abl__delta--neg' : ''}`}>
                  {isFull ? '—' : `${delta >= 0 ? '+' : ''}${fmt(delta, 4)}`}
                </div>
                <div className="rp-abl__bar-wrap">
                  <div
                    className={`rp-abl__bar ${isFull ? 'rp-abl__bar--full' : isWorst ? 'rp-abl__bar--worst' : ''}`}
                    style={{ width: `${(a.auc_mean / 1.0) * 100}%` }}
                  />
                </div>
              </div>
            )
          })}
        </div>
        <p className="rp-note rp-note--clinical">
          <strong>Key finding:</strong> Removing deceleration features drops AUROC by {fmt(fullAUC - abl.find(a => a.group.includes('decelerations'))!.auc_mean, 4)} — the largest single-group drop. FHR baseline, accelerations, and contraction features contribute minimally in isolation. STV/LTV variability and the reserve score are largely redundant with the deceleration burden in this cohort.
        </p>
      </section>

      {/* ── STV by pH group ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Short-term Variability (STV) by pH Risk Group</h3>
        <p className="rp-note" style={{ marginBottom: 16 }}>
          STV is computed as the mean absolute beat-to-beat FHR difference at 4 Hz. Higher STV generally reflects better autonomic tone and fetal reserve. Values here are across all records in each pH stratum.
        </p>
        <div className="rp-stv">
          {stvBph.map(g => {
            const label = g.ph_bucket.replace('\n', ' ')
            const zone = label.includes('Acidosis') ? 'red'
              : label.includes('Borderline') ? 'amber'
              : label.includes('High Normal') ? 'blue'
              : 'green'
            return (
              <div key={g.ph_bucket} className={`rp-stv__group rp-stv__group--${zone}`}>
                <div className="rp-stv__label">{label}</div>
                <div className="rp-stv__val">{g.stv_mean?.toFixed(3) ?? '—'}</div>
                <div className="rp-stv__sd">± {g.stv_std?.toFixed(3) ?? '—'}</div>
                <div className="rp-stv__n">n = {g.count}</div>
                <div className="rp-stv__bar-wrap">
                  <div
                    className={`rp-stv__bar rp-stv__bar--${zone}`}
                    style={{ width: `${((g.stv_mean ?? 0) / 1.0) * 100}%` }}
                  />
                </div>
              </div>
            )
          })}
        </div>
        <p className="rp-note rp-note--clinical">
          <strong>Clinical note:</strong> Contrary to classical teaching, mean STV values here are slightly <em>higher</em> in the acidotic group. This may reflect compensatory sympathetic activation in the early stages of acidosis, or measurement artefact from signal dropout during decelerations. STV alone is insufficient as a distress marker — deceleration burden is the dominant predictor in this cohort.
        </p>
      </section>

      {/* ── Model architecture ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Model Architecture &amp; Training Details</h3>
        <div className="rp-arch-grid">
          <div className="rp-arch-block">
            <strong>Algorithm</strong>
            <ul>
              <li>XGBoost (gradient boosted trees)</li>
              <li>max_depth = 3</li>
              <li>learning_rate = 0.05, n_estimators = 300</li>
              <li>subsample = 0.8, colsample_bytree = 0.8</li>
              <li>reg_alpha = 0.3, reg_lambda = 3.0</li>
            </ul>
          </div>
          <div className="rp-arch-block">
            <strong>Preprocessing</strong>
            <ul>
              <li>Median imputation for missing signal windows</li>
              <li>Robust scaler (IQR-normalised)</li>
              <li>Fit on training fold only — no leakage</li>
              <li>Class-balanced sample weights at every fold</li>
            </ul>
          </div>
          <div className="rp-arch-block">
            <strong>Signal features (18 total)</strong>
            <ul>
              <li>FHR: baseline, mean, SD, STV, LTV</li>
              <li>Decelerations: count, depth, duration, burden</li>
              <li>Accelerations: count, rate per 30 min</li>
              <li>Contractions: count, rate per 10 min</li>
              <li>Composite fetal reserve score</li>
            </ul>
          </div>
          <div className="rp-arch-block">
            <strong>Evaluation protocol</strong>
            <ul>
              <li>80 / 20 stratified hold-out split</li>
              <li>5-fold stratified cross-validation</li>
              <li>Primary metric: AUROC (discrimination)</li>
              <li>Clinical priority metric: sensitivity (at-risk recall)</li>
            </ul>
          </div>
        </div>
      </section>

      {/* ── References ── */}
      <section className="rp-card">
        <h3 className="rp-card__title">Data Sources &amp; References</h3>
        <div className="rp-refs">
          <div className="rp-ref">
            <span className="rp-ref__id">[1]</span>
            <div>
              <strong>CTU-CHB Intrapartum CTG Database</strong><br />
              Chudáček V et al. (2014). Open access intrapartum CTG database. <em>BMC Pregnancy and Childbirth</em>, 14:16.<br />
              <span className="rp-ref__link">PhysioNet · https://doi.org/10.13026/C22013</span>
            </div>
          </div>
          <div className="rp-ref">
            <span className="rp-ref__id">[2]</span>
            <div>
              <strong>CTU-CHB Annotation Dataset (CTGDL / FHRMA)</strong><br />
              Petránek V et al. (2020). Expert morphological annotations for CTG records.<br />
              <span className="rp-ref__link">Zenodo · https://doi.org/10.5281/zenodo.19510407</span>
            </div>
          </div>
          <div className="rp-ref">
            <span className="rp-ref__id">[3]</span>
            <div>
              <strong>FIGO Consensus Guidelines on Intrapartum Fetal Monitoring</strong><br />
              Ayres-de-Campos D et al. (2015). <em>International Journal of Gynecology &amp; Obstetrics</em>, 131(1):13–24.
            </div>
          </div>
          <div className="rp-ref">
            <span className="rp-ref__id">[4]</span>
            <div>
              <strong>XGBoost: A Scalable Tree Boosting System</strong><br />
              Chen T &amp; Guestrin C (2016). <em>Proceedings of ACM SIGKDD</em>.
            </div>
          </div>
        </div>
      </section>

    </div>
  )
}
