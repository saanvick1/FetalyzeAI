import rawResults from '../../results/ctu_reservenet_results.json'

const R = rawResults

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
const LAYER_COLORS = ['#3b82f6', '#a855f7', '#f59e0b', '#22c55e', '#ec4899']

function pct(v: number | null, dp = 1) {
  if (v === null || v === undefined) return '—'
  return `${(v * 100).toFixed(dp)}%`
}
function fmt(v: number | null, dp = 3) {
  if (v === null || v === undefined) return '—'
  return v.toFixed(dp)
}

export function ReserveNetPanel() {
  const arch  = R.architecture
  const cv    = R.cv5
  const boot  = R.bootstrap_ci
  const xm    = R.xgb_test_metrics
  const ens   = R.test_metrics
  const imps  = R.xgb_feature_importance
  const expI  = R.expert_importances as Record<string, { feature: string; importance: number }[]>

  const maxImp = Math.max(...imps.map(i => i.importance ?? 0), 0.001)
  const cmRows = ['Normal (0)', 'Watch (1)', 'High Risk (2)']

  return (
    <div className="rn">

      {/* ── Hero ── */}
      <section className="rn-hero">
        <div className="rn-hero__left">
          <div className="rn-hero__tag">New Architecture</div>
          <h2 className="rn-hero__title">FetalyzeAI ReserveNet v1</h2>
          <p className="rn-hero__sub">
            Domain-Partitioned Stacked Ensemble with Gated Clinical Fusion, trained
            exclusively on <strong>552 real CTU-CHB/CTU-UHB intrapartum recordings</strong>.
            Labels derived solely from cord blood pH — no feature-derived labels.
          </p>
          <div className="rn-badges">
            <span className="rn-badge rn-badge--blue">552 real CTG records</span>
            <span className="rn-badge rn-badge--purple">Record-level 70/15/15 split</span>
            <span className="rn-badge rn-badge--amber">pH-only labels</span>
            <span className="rn-badge rn-badge--green">No synthetic data</span>
            <span className="rn-badge rn-badge--pink">Temperature calibration on val</span>
          </div>
        </div>
        <div className="rn-hero__stats">
          <div className="rn-stat">
            <span className="rn-stat__val">{R.n_records_labeled}</span>
            <span className="rn-stat__lab">Real records</span>
          </div>
          <div className="rn-stat">
            <span className="rn-stat__val">{R.n_features}</span>
            <span className="rn-stat__lab">Features</span>
          </div>
          <div className="rn-stat">
            <span className="rn-stat__val">{R.split.train}/{R.split.val}/{R.split.test}</span>
            <span className="rn-stat__lab">Train/Val/Test</span>
          </div>
          <div className="rn-stat">
            <span className="rn-stat__val">{R.training_time_s}s</span>
            <span className="rn-stat__lab">Train time</span>
          </div>
        </div>
      </section>

      {/* ── Architecture Diagram ── */}
      <section className="rn-card">
        <h3 className="rn-card__title">ReserveNet Architecture — 5 Layers</h3>
        <p className="rn-note" style={{ marginBottom: 20 }}>
          Three clinical-domain specialists each trained on a physiologically coherent feature subset.
          Their outputs feed a neural meta-learner (ReserveFusionMLP) which learns to weight each
          expert per case. Temperature scaling on the validation set calibrates final probabilities.
        </p>
        <div className="rn-arch">
          {arch.layers.map((layer, i) => (
            <div key={i} className="rn-arch__step">
              <div className="rn-arch__connector" style={{ background: i === 0 ? 'transparent' : LAYER_COLORS[i-1] }} />
              <div className="rn-arch__box" style={{ borderTopColor: LAYER_COLORS[i] }}>
                <div className="rn-arch__num" style={{ background: LAYER_COLORS[i] }}>{i + 1}</div>
                <div className="rn-arch__name">{layer.name}</div>
                <div className="rn-arch__model">{layer.model}</div>
                <div className="rn-arch__features">
                  {layer.features.map(f => (
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
            <div className="rn-policy__val">{arch.label_policy}</div>
          </div>
          <div className="rn-policy rn-policy--green">
            <div className="rn-policy__label">Split Policy</div>
            <div className="rn-policy__val">{arch.split_policy}</div>
          </div>
          <div className="rn-policy rn-policy--purple">
            <div className="rn-policy__label">Calibration Set</div>
            <div className="rn-policy__val">{arch.calibration_set}</div>
          </div>
        </div>
      </section>

      {/* ── Dataset ── */}
      <section className="rn-card">
        <h3 className="rn-card__title">Dataset — CTU-CHB/CTU-UHB (PhysioNet)</h3>
        <div className="rn-dataset-grid">
          <div className="rn-ds-block">
            <div className="rn-ds-block__val">{R.n_records_total}</div>
            <div className="rn-ds-block__lab">Total records loaded</div>
          </div>
          <div className="rn-ds-block">
            <div className="rn-ds-block__val">{R.label_distribution.normal_0}</div>
            <div className="rn-ds-block__lab">Normal (pH ≥ 7.15)</div>
            <div className="rn-ds-block__pct">{pct(R.label_distribution.normal_0 / R.n_records_labeled)}</div>
          </div>
          <div className="rn-ds-block">
            <div className="rn-ds-block__val" style={{ color: '#f59e0b' }}>{R.label_distribution.watch_1}</div>
            <div className="rn-ds-block__lab">Watch (pH 7.05–7.15)</div>
            <div className="rn-ds-block__pct">{pct(R.label_distribution.watch_1 / R.n_records_labeled)}</div>
          </div>
          <div className="rn-ds-block">
            <div className="rn-ds-block__val" style={{ color: '#ef4444' }}>{R.label_distribution.high_risk_2}</div>
            <div className="rn-ds-block__lab">High Risk (pH &lt; 7.05)</div>
            <div className="rn-ds-block__pct">{pct(R.label_distribution.high_risk_2 / R.n_records_labeled)}</div>
          </div>
          <div className="rn-ds-block">
            <div className="rn-ds-block__val">{R.n_excluded}</div>
            <div className="rn-ds-block__lab">Excluded (unknown label)</div>
          </div>
        </div>
        <div className="rn-note rn-note--clinical" style={{ marginTop: 16 }}>
          <strong>Class imbalance context:</strong> Only 7.2% of records are high-risk (pH &lt; 7.05)
          and 11.8% watch. The test set contains ~6 high-risk and ~10 watch cases — this makes
          per-class recall highly variable. Binary AUROC and 5-fold CV sensitivity are the more
          reliable estimates at this dataset size.
        </div>
      </section>

      {/* ── Key Test Metrics ── */}
      <section className="rn-card">
        <h3 className="rn-card__title">Held-Out Test Set — XGB Baseline vs Ensemble (n={R.split.test})</h3>
        <div className="rn-metrics-cmp">
          {[
            { key: 'auroc_binary',     label: 'AUROC (binary)',    note: 'At-risk vs safe', hi: true },
            { key: 'sensitivity',      label: 'Sensitivity',       note: 'At-risk recall',  hi: true },
            { key: 'specificity',      label: 'Specificity',       note: 'Safe recall',     hi: true },
            { key: 'macro_f1',         label: 'Macro F1',          note: '3-class balance', hi: true },
            { key: 'ece',              label: 'ECE',               note: 'Calibration ↓',   hi: false },
            { key: 'auprc_binary',     label: 'AUPRC',            note: 'Precision-recall', hi: true },
          ].map(({ key, label, note, hi }) => {
            const xv = (xm as unknown as Record<string, number>)[key] ?? null
            const ev = (ens as unknown as Record<string, number>)[key] ?? null
            const better = ev !== null && xv !== null
              ? (hi ? ev > xv + 0.002 : ev < xv - 0.002)
              : false
            return (
              <div key={key} className="rn-mc__col">
                <div className="rn-mc__label">{label}</div>
                <div className="rn-mc__note">{note}</div>
                <div className="rn-mc__row">
                  <div className="rn-mc__block rn-mc__block--xgb">
                    <div className="rn-mc__tag">XGB</div>
                    <div className="rn-mc__val">{xv !== null ? (xv < 1 ? pct(xv) : fmt(xv)) : '—'}</div>
                  </div>
                  <div className="rn-mc__arrow">{better ? '▲' : '→'}</div>
                  <div className="rn-mc__block rn-mc__block--ens">
                    <div className="rn-mc__tag">Ensemble</div>
                    <div className="rn-mc__val">{ev !== null ? (ev < 1 ? pct(ev) : fmt(ev)) : '—'}</div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
        <div className="rn-boot">
          <div className="rn-boot__item">
            <span className="rn-boot__lab">AUROC 95% CI:</span>
            <span className="rn-boot__val">[{fmt(boot.auroc_binary.ci_lo)}, {fmt(boot.auroc_binary.ci_hi)}]</span>
          </div>
          <div className="rn-boot__item">
            <span className="rn-boot__lab">Sensitivity 95% CI:</span>
            <span className="rn-boot__val">[{fmt(boot.sensitivity.ci_lo)}, {fmt(boot.sensitivity.ci_hi)}]</span>
          </div>
          <div className="rn-boot__item">
            <span className="rn-boot__lab">Temperature T:</span>
            <span className="rn-boot__val">{fmt(R.temperature_T, 3)}</span>
          </div>
          <div className="rn-boot__item">
            <span className="rn-boot__lab">Bootstrap n:</span>
            <span className="rn-boot__val">200 iterations</span>
          </div>
        </div>
        <div className="rn-note rn-note--warn" style={{ marginTop: 16 }}>
          <strong>Note on small test set:</strong> The test set has 83 records — roughly 6 high-risk
          and 10 watch cases. A single misclassification shifts sensitivity by ~17pp. The 5-fold CV
          results below, computed over all 552 records, give a more stable sensitivity estimate (53% ± 7%).
        </div>
      </section>

      {/* ── 5-Fold CV ── */}
      <section className="rn-card">
        <h3 className="rn-card__title">5-Fold Cross-Validation — ReserveNet (all 552 records)</h3>
        <div className="rn-cv-metrics">
          {[
            { label: 'AUROC',       mean: cv.mean_auroc, std: cv.std_auroc },
            { label: 'Sensitivity', mean: cv.mean_sens,  std: cv.std_sens  },
            { label: 'Specificity', mean: cv.mean_spec,  std: cv.std_spec  },
            { label: 'Macro F1',    mean: cv.mean_f1,    std: cv.std_f1    },
          ].map(m => (
            <div key={m.label} className={`rn-cv-metric ${m.label === 'Sensitivity' ? 'rn-cv-metric--hi' : ''}`}>
              <span className="rn-cv-metric__val">{fmt(m.mean)}</span>
              <span className="rn-cv-metric__sd">± {fmt(m.std)}</span>
              <span className="rn-cv-metric__lab">{m.label}</span>
            </div>
          ))}
        </div>
        <div className="rn-table-wrap">
          <table className="rn-table">
            <thead>
              <tr><th>Fold</th><th>AUROC</th><th>Sensitivity</th><th>Specificity</th><th>Macro F1</th></tr>
            </thead>
            <tbody>
              {cv.fold_auroc.map((auc, i) => (
                <tr key={i}>
                  <td className="rn-table__fold">Fold {i + 1}</td>
                  <td>{fmt(auc)}</td>
                  <td>{pct(cv.fold_sens[i])}</td>
                  <td>{pct(cv.fold_spec[i])}</td>
                  <td>{fmt(cv.fold_f1[i])}</td>
                </tr>
              ))}
              <tr className="rn-table__mean">
                <td>Mean ± SD</td>
                <td>{fmt(cv.mean_auroc)} ± {fmt(cv.std_auroc)}</td>
                <td>{pct(cv.mean_sens)} ± {pct(cv.std_sens)}</td>
                <td>{pct(cv.mean_spec)} ± {pct(cv.std_spec)}</td>
                <td>{fmt(cv.mean_f1)} ± {fmt(cv.std_f1)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Feature Importances ── */}
      <section className="rn-card">
        <h3 className="rn-card__title">Feature Importance — XGBoost Baseline (top 12)</h3>
        <div className="rn-imp-bars">
          {imps.map((item) => (
            <div key={item.feature} className="rn-imp-row">
              <div className="rn-imp-row__label">{item.feature}</div>
              <div className="rn-imp-row__bar-wrap">
                <div
                  className="rn-imp-row__bar"
                  style={{ width: `${((item.importance ?? 0) / maxImp) * 100}%` }}
                />
              </div>
              <div className="rn-imp-row__val">{((item.importance ?? 0) * 100).toFixed(1)}%</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Expert Importances ── */}
      <section className="rn-card">
        <h3 className="rn-card__title">Expert Domain Importances — Top Features per Specialist</h3>
        <div className="rn-exp-grid">
          {Object.entries(expI).map(([name, items]) => {
            const max = Math.max(...items.map(i => i.importance), 0.001)
            return (
              <div key={name} className="rn-exp-block" style={{ borderTopColor: EXPERT_COLORS[name] }}>
                <div className="rn-exp-block__head" style={{ color: EXPERT_COLORS[name] }}>
                  {EXPERT_LABELS[name] ?? name}
                </div>
                {items.map(item => (
                  <div key={item.feature} className="rn-exp-row">
                    <span className="rn-exp-row__label">{item.feature}</span>
                    <div className="rn-exp-row__bar-wrap">
                      <div
                        className="rn-exp-row__bar"
                        style={{
                          width: `${(item.importance / max) * 100}%`,
                          background: EXPERT_COLORS[name],
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )
          })}
        </div>
      </section>

      {/* ── Why ReserveNet Generalizes ── */}
      <section className="rn-card">
        <h3 className="rn-card__title">Why ReserveNet Generalizes to Other CTG Datasets</h3>
        <div className="rn-gen-grid">
          {[
            {
              icon: '🧬',
              title: 'Clinically-partitioned features',
              body: 'Each expert uses features from a single physiological domain (baseline, variability, event patterns). Domain-partitioned learners transfer better because each domain is independently meaningful across CTG devices and populations.',
            },
            {
              icon: '📊',
              title: 'pH-only labels',
              body: 'Labels come from cord blood pH — the gold-standard clinical outcome — not from derived features. This means the model learns real clinical signal rather than fitting to circular feature-label dependencies.',
            },
            {
              icon: '🔒',
              title: 'Record-level split',
              body: 'Zero patient-level data leakage. Windows from the same recording never appear in both train and test. This is the correct evaluation protocol for any multi-window CTG dataset.',
            },
            {
              icon: '🌡️',
              title: 'Temperature calibration on validation',
              body: 'Probability calibration is fit on the validation set (T=0.66), never on test data. Calibrated probabilities transfer better to external datasets because they reflect true uncertainty rather than overconfident predictions.',
            },
            {
              icon: '⚖️',
              title: 'Class-balanced training',
              body: 'All three experts use class-balanced weighting. Without this, extreme imbalance (81% normal) would collapse sensitivity to near zero. Balanced training preserves the rare-class signal critical for clinical screening.',
            },
            {
              icon: '🔄',
              title: 'Atomic features only',
              body: 'No composite leakage. All 24 features are atomic signal measurements (FHR baseline, STV, LTV, decel counts, depths, durations, CSR fraction). These are directly computable from any standard FHR/UC waveform.',
            },
          ].map(({ icon, title, body }) => (
            <div key={title} className="rn-gen-card">
              <div className="rn-gen-card__icon">{icon}</div>
              <div className="rn-gen-card__title">{title}</div>
              <div className="rn-gen-card__body">{body}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Confusion Matrix ── */}
      <section className="rn-card">
        <h3 className="rn-card__title">Confusion Matrix — Ensemble Test Set</h3>
        <p className="rn-note" style={{ marginBottom: 16 }}>
          Rows = actual class, columns = predicted class. With only 6 high-risk and 10 watch
          cases in the test set (83 records total), per-class recall is highly variable.
          The model's strength at this dataset size is AUROC discrimination (0.68), not
          per-class accuracy.
        </p>
        <div className="rn-cm">
          <div className="rn-cm__header">
            <div className="rn-cm__corner">Actual ↓ / Pred →</div>
            {cmRows.map(r => <div key={r} className="rn-cm__col-head">{r}</div>)}
          </div>
          {ens.confusion_matrix.map((row: number[], ri: number) => (
            <div key={ri} className="rn-cm__row">
              <div className="rn-cm__row-head">{cmRows[ri]}</div>
              {row.map((v: number, ci: number) => (
                <div key={ci}
                  className={`rn-cm__cell ${ri === ci ? 'rn-cm__cell--diag' : v > 0 ? 'rn-cm__cell--err' : ''}`}>
                  {v}
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>

      {/* ── Safety ── */}
      <section className="rn-card rn-card--safety">
        <h3 className="rn-card__title">Safety &amp; Research Status</h3>
        <p>
          <strong>FetalyzeAI ReserveNet is a research-stage CTG second-reader</strong> trained on
          CTU-CHB/CTU-UHB waveform data. It analyzes fetal heart rate, uterine contractions,
          deceleration burden, contraction stress response, fetal reserve, signal quality, and
          uncertainty to support clinician review. It does not diagnose fetal distress, recommend
          treatment, or replace clinicians. All outputs require expert clinical interpretation
          before any action is taken.
        </p>
        <div className="rn-refs">
          <span className="rn-ref">Chudáček et al. (2014) BMC Pregnancy and Childbirth 14:16</span>
          <span className="rn-ref">CTU-CHB/CTU-UHB Database v1.0.0 — PhysioNet ODC-BY-1.0</span>
          <span className="rn-ref">FIGO Guidelines for Intrapartum CTG Interpretation (2015)</span>
        </div>
      </section>

    </div>
  )
}
