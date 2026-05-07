import { useState } from 'react'

// Loaded dynamically so it always reflects the latest tuning_results.json on disk
// We import it at module level — Vite will bundle it at build time
import rawResults from '../../tuning_results.json'

type TuningResults = typeof rawResults
const results = rawResults as TuningResults

const MODEL_COLORS: Record<string, string> = {
  XGBoost:           '#3b82f6',
  RandomForest:      '#22c55e',
  LogisticRegression:'#a855f7',
}
const MODEL_SHORT: Record<string, string> = {
  XGBoost:           'XGBoost',
  RandomForest:      'Random Forest',
  LogisticRegression:'Logistic Reg.',
}

function pct(v: number, dp = 1) { return `${(v * 100).toFixed(dp)}%` }
function fmt(v: number, dp = 4) { return v.toFixed(dp) }
function delta(after: number, before: number) {
  const d = after - before
  return `${d >= 0 ? '+' : ''}${(d * 100).toFixed(1)}pp`
}
function deltaClass(after: number, before: number, higherIsBetter = true) {
  const d = (after - before) * (higherIsBetter ? 1 : -1)
  return d > 0.002 ? 'tp-delta--pos' : d < -0.002 ? 'tp-delta--neg' : 'tp-delta--flat'
}

export function TuningPanel() {
  const [filterModel, setFilterModel] = useState<string>('all')

  const obj    = results.tuning_objective
  const trials = results.trials
  const bests  = results.best_scores
  const bp     = results.best_params
  const cv     = results.tuned_cv5
  const cmp    = results.comparison
  const ho     = results.holdout_tuned

  const models = ['XGBoost', 'RandomForest', 'LogisticRegression']
  const visibleTrials = filterModel === 'all' ? trials : trials.filter(t => t.model === filterModel)

  // Running best per model for convergence chart
  const convergenceByModel: Record<string, { x: number; best: number }[]> = {}
  for (const m of models) {
    let runBest = -Infinity
    const mTrials = trials.filter(t => t.model === m)
    convergenceByModel[m] = mTrials.map((t, i) => {
      runBest = Math.max(runBest, t.score)
      return { x: i + 1, best: runBest }
    })
  }

  const maxScore = Math.max(...trials.map(t => t.score))
  const minScore = Math.min(...trials.map(t => t.score))
  const scoreRange = maxScore - minScore || 0.01

  return (
    <div className="tp">

      {/* ── Hero ── */}
      <section className="tp-hero">
        <div>
          <h2>Auto-Tune — Medical-Objective Hyperparameter Search</h2>
          <p>
            Ran <strong>{results.n_trials} randomised trials</strong> across XGBoost, Random Forest, and
            Logistic Regression in <strong>{results.tuning_time_s}s</strong>. Each trial was scored
            with a medical composite function that heavily rewards sensitivity — because missing an
            at-risk fetus is the critical failure mode.
          </p>
          <div className="tp-badges">
            <span className="tp-badge tp-badge--blue">{results.n_trials} trials</span>
            <span className="tp-badge tp-badge--green">3-fold CV per trial</span>
            <span className="tp-badge tp-badge--purple">Threshold = 0.35</span>
            <span className="tp-badge tp-badge--amber">No composites · 16 atomic features</span>
          </div>
        </div>
        <div className="tp-objective">
          <div className="tp-objective__title">Tuning Objective</div>
          <div className="tp-objective__formula">{obj.formula}</div>
          <div className="tp-objective__penalty">Penalty: {obj.penalty}</div>
          <div className="tp-objective__constraints">
            {obj.hard_constraints.map(c => (
              <span key={c} className="tp-constraint">{c}</span>
            ))}
          </div>
        </div>
      </section>

      {/* ── Before / After comparison ── */}
      <section className="tp-card">
        <h3 className="tp-card__title">Before vs After Tuning — 5-fold CV Metrics</h3>
        <div className="tp-cmp">
          {[
            { key: 'auc',   label: 'AUROC',       hi: true,  note: 'Discrimination' },
            { key: 'sens',  label: 'Sensitivity',  hi: true,  note: 'At-risk recall ★' },
            { key: 'spec',  label: 'Specificity',  hi: true,  note: 'Safe recall' },
            { key: 'f1',    label: 'F1 Score',     hi: true,  note: 'Precision-recall balance' },
            { key: 'brier', label: 'Brier Score',  hi: false, note: 'Calibration (↓ better)' },
            { key: 'score', label: 'Medical Score', hi: true, note: 'Tuning objective' },
          ].map(({ key, label, hi, note }) => {
            const bVal = (cmp.before as unknown as Record<string, number>)[key]
            const aVal = (cmp.after  as unknown as Record<string, number>)[key]
            const improved = hi ? aVal > bVal + 0.002 : aVal < bVal - 0.002
            return (
              <div key={key} className={`tp-cmp__col ${improved ? 'tp-cmp__col--better' : ''}`}>
                <div className="tp-cmp__label">{label}</div>
                <div className="tp-cmp__note">{note}</div>
                <div className="tp-cmp__row">
                  <div className="tp-cmp__block tp-cmp__block--before">
                    <div className="tp-cmp__tag">Before</div>
                    <div className="tp-cmp__val">{key === 'score' ? fmt(bVal, 3) : key === 'brier' ? fmt(bVal, 4) : pct(bVal)}</div>
                  </div>
                  <div className={`tp-cmp__delta ${deltaClass(aVal, bVal, hi)}`}>
                    {improved ? '▲' : Math.abs(aVal - bVal) < 0.002 ? '—' : '▼'}
                  </div>
                  <div className="tp-cmp__block tp-cmp__block--after">
                    <div className="tp-cmp__tag">After</div>
                    <div className="tp-cmp__val">{key === 'score' ? fmt(aVal, 3) : key === 'brier' ? fmt(aVal, 4) : pct(aVal)}</div>
                  </div>
                </div>
                <div className={`tp-cmp__pp ${deltaClass(aVal, bVal, hi)}`}>
                  {key === 'score'
                    ? `${(aVal - bVal >= 0 ? '+' : '')}${fmt(aVal - bVal, 3)}`
                    : delta(aVal, bVal)}
                </div>
              </div>
            )
          })}
        </div>
      </section>

      {/* ── Best params found ── */}
      <section className="tp-card">
        <h3 className="tp-card__title">Best Hyperparameters Found per Model</h3>
        <div className="tp-params-grid">
          {models.map(m => (
            <div key={m} className="tp-params-block" style={{ borderTopColor: MODEL_COLORS[m] }}>
              <div className="tp-params-block__head">
                <span className="tp-params-block__model" style={{ color: MODEL_COLORS[m] }}>
                  {MODEL_SHORT[m]}
                </span>
                <span className="tp-params-block__score">
                  Best score: <strong>{fmt(bests[m as keyof typeof bests], 3)}</strong>
                </span>
              </div>
              <div className="tp-params-block__pairs">
                {Object.entries(bp[m as keyof typeof bp] || {}).map(([k, v]) => (
                  <div key={k} className="tp-param-pair">
                    <span className="tp-param-pair__key">{k}</span>
                    <span className="tp-param-pair__val">{String(v)}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Trial scatter ── */}
      <section className="tp-card">
        <h3 className="tp-card__title">All Trials — AUROC vs Sensitivity (coloured by model)</h3>
        <p className="tp-note" style={{ marginBottom: 16 }}>
          Each point is one trial (random hyperparameter combination evaluated on 3-fold CV). The ideal region is top-right: high AUROC + high sensitivity. Trials that violate the sensitivity ≥ 0.75 constraint are penalised heavily and appear in the lower region.
        </p>
        <div className="tp-filter">
          {['all', ...models].map(m => (
            <button key={m} className={`tp-filter__btn ${filterModel === m ? 'tp-filter__btn--active' : ''}`}
              style={filterModel === m && m !== 'all' ? { borderColor: MODEL_COLORS[m], color: MODEL_COLORS[m] } : {}}
              onClick={() => setFilterModel(m)}>
              {m === 'all' ? 'All models' : MODEL_SHORT[m]}
            </button>
          ))}
        </div>
        <div className="tp-scatter">
          <div className="tp-scatter__y-label">Sensitivity →</div>
          <div className="tp-scatter__plot">
            {/* Target zone overlay */}
            <div className="tp-scatter__zone" title="Target zone: AUROC ≥ 0.60, Sens ≥ 0.75" />
            {visibleTrials.map((t, i) => {
              const cx = ((t.auc - 0.55) / (0.80 - 0.55)) * 100
              const cy = (t.sens / 1.0) * 100
              const col = MODEL_COLORS[t.model] ?? '#94a3b8'
              const isBest = t.score === Math.max(...trials.filter(x => x.model === t.model).map(x => x.score))
              return (
                <div key={i}
                  className={`tp-dot ${isBest ? 'tp-dot--best' : ''}`}
                  style={{
                    left: `${Math.max(0, Math.min(100, cx))}%`,
                    bottom: `${Math.max(0, Math.min(100, cy))}%`,
                    background: col,
                    boxShadow: isBest ? `0 0 0 3px ${col}40` : undefined,
                  }}
                  title={`${MODEL_SHORT[t.model]}: AUROC=${t.auc} Sens=${t.sens} Score=${t.score}`}
                />
              )
            })}
            {/* Axis labels */}
            <div className="tp-scatter__x-ticks">
              <span>0.55</span><span>0.60</span><span>0.65</span><span>0.70</span><span>0.75</span><span>0.80</span>
            </div>
            <div className="tp-scatter__y-ticks">
              <span>100%</span><span>75%</span><span>50%</span><span>25%</span><span>0%</span>
            </div>
          </div>
          <div className="tp-scatter__x-label">AUROC →</div>
          <div className="tp-scatter__legend">
            {models.map(m => (
              <span key={m} className="tp-scatter__leg-item">
                <span className="tp-scatter__leg-dot" style={{ background: MODEL_COLORS[m] }} />
                {MODEL_SHORT[m]}
              </span>
            ))}
            <span className="tp-scatter__leg-item">
              <span className="tp-scatter__leg-dot tp-scatter__leg-dot--best" />
              Best per model
            </span>
          </div>
        </div>
        <p className="tp-note tp-note--clinical">
          <strong>Target zone</strong> (shaded): AUROC ≥ 0.60, Sensitivity ≥ 0.75. Trials in this region meet minimum medical screening requirements. The clustering near sensitivity = 80–95% reflects the penalty term actively steering the search away from low-sensitivity configurations.
        </p>
      </section>

      {/* ── Convergence chart ── */}
      <section className="tp-card">
        <h3 className="tp-card__title">Convergence — Best Medical Score Found Over Trials</h3>
        <p className="tp-note" style={{ marginBottom: 16 }}>
          Running best score after each trial. A flattening curve means the search has explored enough of the space; a still-rising curve suggests more trials would help. Trials are shown per model type.
        </p>
        <div className="tp-conv">
          {models.map(m => {
            const series = convergenceByModel[m]
            const mMax = Math.max(...series.map(s => s.best))
            return (
              <div key={m} className="tp-conv__model">
                <div className="tp-conv__head">
                  <span style={{ color: MODEL_COLORS[m] }}>{MODEL_SHORT[m]}</span>
                  <span className="tp-conv__best">Best: {fmt(mMax, 3)}</span>
                </div>
                <div className="tp-conv__chart">
                  {series.map((pt, i) => (
                    <div key={i} className="tp-conv__col">
                      <div className="tp-conv__bar"
                        style={{
                          height: `${((pt.best - minScore) / scoreRange) * 80 + 10}px`,
                          background: MODEL_COLORS[m],
                          opacity: i === series.length - 1 ? 1 : 0.6 + (i / series.length) * 0.4,
                        }}
                        title={`Trial ${pt.x}: best so far = ${pt.best.toFixed(3)}`}
                      />
                    </div>
                  ))}
                </div>
                <div className="tp-conv__x-labels">
                  <span>Trial 1</span>
                  <span>Trial {series.length}</span>
                </div>
              </div>
            )
          })}
        </div>
      </section>

      {/* ── Tuned 5-fold CV ── */}
      <section className="tp-card">
        <h3 className="tp-card__title">Tuned Ensemble — 5-fold CV Results (threshold = 0.35)</h3>
        <div className="tp-cv-grid">
          {[
            { label: 'AUROC',       val: fmt(cv.mean_auc, 4),  sd: fmt(cv.std_auc, 4),  note: 'Threshold-independent',    hi: false },
            { label: 'Sensitivity', val: pct(cv.mean_sens),     sd: pct(cv.std_sens),     note: 'At-risk recall',           hi: cv.mean_sens >= 0.80 },
            { label: 'Specificity', val: pct(cv.mean_spec),     sd: pct(cv.std_spec),     note: 'Safe recall',              hi: false },
            { label: 'F1 Score',    val: fmt(cv.mean_f1, 4),   sd: fmt(cv.std_f1, 4),   note: 'Precision-recall balance',  hi: false },
            { label: 'Balanced Acc', val: pct(cv.mean_bal),    sd: '—',                   note: 'Class-balanced accuracy',  hi: false },
            { label: 'Brier (ho)', val: fmt(ho.brier, 4),      sd: '—',                   note: 'Calibration quality',      hi: false },
          ].map(m => (
            <div key={m.label} className={`tp-metric ${m.hi ? 'tp-metric--hi' : ''}`}>
              <span className="tp-metric__val">{m.val}</span>
              <span className="tp-metric__sd">± {m.sd}</span>
              <span className="tp-metric__label">{m.label}</span>
              <span className="tp-metric__note">{m.note}</span>
            </div>
          ))}
        </div>
        <div className="tp-table-wrap">
          <table className="tp-table">
            <thead>
              <tr><th>Fold</th><th>AUROC</th><th>Sensitivity</th><th>Specificity</th><th>F1</th><th>Bal. Acc</th></tr>
            </thead>
            <tbody>
              {cv.fold_auc.map((auc, i) => (
                <tr key={i}>
                  <td className="tp-table__fold">Fold {i + 1}</td>
                  <td>{fmt(auc, 4)}</td>
                  <td>{pct(cv.fold_sens[i])}</td>
                  <td>{pct(cv.fold_spec[i])}</td>
                  <td>{fmt(cv.fold_f1[i], 4)}</td>
                  <td>{pct(cv.fold_bal[i])}</td>
                </tr>
              ))}
              <tr className="tp-table__mean">
                <td>Mean ± SD</td>
                <td>{fmt(cv.mean_auc,4)} ± {fmt(cv.std_auc,4)}</td>
                <td>{pct(cv.mean_sens)} ± {pct(cv.std_sens)}</td>
                <td>{pct(cv.mean_spec)} ± {pct(cv.std_spec)}</td>
                <td>{fmt(cv.mean_f1,4)} ± {fmt(cv.std_f1,4)}</td>
                <td>{pct(cv.mean_bal)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Search space ── */}
      <section className="tp-card">
        <h3 className="tp-card__title">Search Space — Parameter Ranges Explored</h3>
        <div className="tp-space-grid">
          {models.map(m => (
            <div key={m} className="tp-space-block">
              <div className="tp-space-block__head" style={{ color: MODEL_COLORS[m] }}>{MODEL_SHORT[m]}</div>
              {Object.entries(results.search_space[m as keyof typeof results.search_space] || {}).map(([k, vs]) => (
                <div key={k} className="tp-space-row">
                  <span className="tp-space-row__key">{k}</span>
                  <span className="tp-space-row__vals">{(vs as string[]).join(', ')}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
        <p className="tp-note" style={{ marginTop: 16 }}>
          Each trial samples one value uniformly at random from each parameter's list. The fixed constants (objective function, class weighting, random seed) are not searched — only the regularisation and architecture parameters that affect overfitting are tuned.
        </p>
      </section>

    </div>
  )
}
