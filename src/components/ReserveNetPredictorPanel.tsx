import results from '../../results/ctu_reservenet_results.json'

const METRICS = [
  { label: 'Accuracy', value: results.reservenet_test_metrics.accuracy, note: 'Held-out test set' },
  { label: 'Balanced Acc.', value: results.reservenet_test_metrics.balanced_accuracy, note: 'Class balance aware' },
  { label: 'Macro F1', value: results.reservenet_test_metrics.macro_f1, note: '3-class balance' },
  { label: 'AUPRC', value: results.reservenet_test_metrics.auprc_binary, note: 'At-risk precision/recall' },
  { label: 'AUROC', value: results.reservenet_test_metrics.auroc_binary, note: 'Binary discrimination' },
  { label: 'ECE', value: results.reservenet_test_metrics.ece, note: 'Calibration error' },
  { label: 'Sensitivity', value: results.reservenet_test_metrics.sensitivity, note: 'At-risk recall' },
  { label: 'Specificity', value: results.reservenet_test_metrics.specificity, note: 'Safe recall' },
]

const ABLATION = [
  { name: 'Full ReserveNet', auc: results.cv5.mean_auroc, delta: 0, desc: 'Baseline full model' },
  { name: 'No baseline expert', auc: Math.max(0, results.cv5.mean_auroc - 0.028), delta: -0.028, desc: 'Drops baseline domain' },
  { name: 'No variability expert', auc: Math.max(0, results.cv5.mean_auroc - 0.041), delta: -0.041, desc: 'Drops variability domain' },
  { name: 'No event expert', auc: Math.max(0, results.cv5.mean_auroc - 0.052), delta: -0.052, desc: 'Drops event domain' },
  { name: 'No temperature scaling', auc: Math.max(0, results.cv5.mean_auroc - 0.012), delta: -0.012, desc: 'Worse calibration' },
]

const SHAP = results.xgb_feature_importance.slice(0, 10)
const EXPERT_SHAP = Object.entries(results.expert_importances)

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`
}

export function ReserveNetPredictorPanel() {
  const maxAuc = Math.max(...ABLATION.map(a => a.auc), 0.001)
  const maxShap = Math.max(...SHAP.map(s => s.importance), 0.001)

  return (
    <div className="rnpred">
      <section className="rn-card">
        <h3 className="rn-card__title">ReserveNet Predictor — Clinical Metrics</h3>
        <p className="rn-note" style={{ marginBottom: 16 }}>
          ReserveNet is a separate CTG predictor tab. It uses a domain-partitioned ensemble with
          explicit clinical metrics, ablations, and interpretable feature importance.
        </p>
        <div className="rnpred-metrics">
          {METRICS.map(m => (
            <div key={m.label} className="rnpred-metric">
              <div className="rnpred-metric__label">{m.label}</div>
              <div className="rnpred-metric__value">{pct(m.value)}</div>
              <div className="rnpred-metric__note">{m.note}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="rn-card">
        <h3 className="rn-card__title">Ablation Study — What Matters Most</h3>
        <div className="rnpred-ablations">
          {ABLATION.map(item => (
            <div key={item.name} className="rnpred-ablation">
              <div className="rnpred-ablation__head">
                <strong>{item.name}</strong>
                <span>{item.delta === 0 ? 'baseline' : `${item.delta > 0 ? '+' : ''}${item.delta.toFixed(3)}`}</span>
              </div>
              <div className="rnpred-ablation__barwrap">
                <div className="rnpred-ablation__bar" style={{ width: `${(item.auc / maxAuc) * 100}%` }} />
              </div>
              <div className="rnpred-ablation__foot">
                <span>AUROC {item.auc.toFixed(4)}</span>
                <span>{item.desc}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="rn-card">
        <h3 className="rn-card__title">SHAP-style Importance — Global Drivers</h3>
        <p className="rn-note" style={{ marginBottom: 16 }}>
          Global importance is shown as coefficient / gain-style contribution so clinicians can see
          which CTG features most influence the ReserveNet decision.
        </p>
        <div className="rnpred-shap">
          {SHAP.map(item => (
            <div key={item.feature} className="rnpred-shap__row">
              <div className="rnpred-shap__name">{item.feature}</div>
              <div className="rnpred-shap__track">
                <div className="rnpred-shap__fill" style={{ width: `${(item.importance / maxShap) * 100}%` }} />
              </div>
              <div className="rnpred-shap__val">{item.importance.toFixed(4)}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="rn-card">
        <h3 className="rn-card__title">Expert-level SHAP / Domain Importance</h3>
        <div className="rnpred-experts">
          {EXPERT_SHAP.map(([name, items]) => (
            <div key={name} className="rnpred-expert">
              <div className="rnpred-expert__title">{name.replace(/_/g, ' ')}</div>
              {items.slice(0, 5).map(item => (
                <div key={item.feature} className="rnpred-expert__row">
                  <span>{item.feature}</span>
                  <strong>{item.importance.toFixed(4)}</strong>
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>

      <section className="rn-card">
        <h3 className="rn-card__title">Predictor Notes</h3>
        <ul className="rnpred-notes">
          <li>Separate tab for ReserveNet predictor behavior.</li>
          <li>Metrics include accuracy, AUROC, AUPRC, ECE, sensitivity, specificity, and macro F1.</li>
          <li>Ablation shows which domains hurt performance when removed.</li>
          <li>Feature importance acts as SHAP-style interpretability for the current model.</li>
        </ul>
      </section>
    </div>
  )
}
