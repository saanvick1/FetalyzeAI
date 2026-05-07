"""
FetalyzeAI v4.0 — Prediction Model Trainer (Tab 2)
====================================================
Leakage-free retraining of prediction_model.pkl for the interactive prediction tab.

Key fixes vs v3:
  - imputer/scaler fit ONLY on training indices
  - XGBoost depth 3, strong regularization (was depth 8)
  - Nested CV re-fits preprocessing inside each fold
  - Saves both scaler/imputer fitted on train split for correct single-sample inference
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

print("FetalyzeAI v4.0 — Retraining prediction_model.pkl (leakage-free)...")

df = pd.read_csv('fetal_health.csv')
feature_names = df.columns[:-1].tolist()
X_raw = df[feature_names].values
y_raw = (df['fetal_health'].values - 1).astype(int)

# ── Leakage-free split ────────────────────────────────────────────────────────
idx_tr, idx_te = train_test_split(
    np.arange(len(X_raw)), test_size=0.2, stratify=y_raw, random_state=42
)

imputer = SimpleImputer(strategy='median')
scaler  = RobustScaler()

X_tr_imp = imputer.fit_transform(X_raw[idx_tr])
X_te_imp = imputer.transform(X_raw[idx_te])
X_tr     = scaler.fit_transform(X_tr_imp)
X_te     = scaler.transform(X_te_imp)

y_tr = y_raw[idx_tr]
y_te = y_raw[idx_te]

cw  = compute_class_weight('balanced', classes=np.unique(y_raw), y=y_raw)
sw_tr = np.array([cw[yi] for yi in y_tr])

# ── Regularized XGBoost ───────────────────────────────────────────────────────
print("  Training regularized XGBoost (depth 3, strong L1/L2)...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=1.0,
    reg_alpha=0.5,
    reg_lambda=5.0,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
)
model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=0)

test_acc  = accuracy_score(y_te, model.predict(X_te))
test_f1   = f1_score(y_te, model.predict(X_te), average='macro', zero_division=0)
test_path = recall_score(y_te, model.predict(X_te), labels=[2], average='macro', zero_division=0)

print(f"  Held-out test: accuracy={test_acc*100:.2f}%  F1={test_f1:.4f}  PathRecall={test_path:.4f}")

# ── Leakage-free 5-fold CV ────────────────────────────────────────────────────
print("  Running leakage-free 5-fold CV (preprocessing re-fit per fold)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accs = []
cv_f1s  = []

for fold, (fold_tr, fold_val) in enumerate(skf.split(X_raw, y_raw)):
    imp_f = SimpleImputer(strategy='median')
    sc_f  = RobustScaler()
    X_ftr  = sc_f.fit_transform(imp_f.fit_transform(X_raw[fold_tr]))
    X_fval = sc_f.transform(imp_f.transform(X_raw[fold_val]))
    y_ftr  = y_raw[fold_tr]
    y_fval = y_raw[fold_val]
    cw_f   = compute_class_weight('balanced', classes=np.unique(y_ftr), y=y_ftr)
    sw_f   = np.array([cw_f[yi] for yi in y_ftr])

    m_f = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=1.0, reg_alpha=0.5, reg_lambda=5.0,
        objective='multi:softprob', eval_metric='mlogloss',
        random_state=42, n_jobs=-1, tree_method='hist',
    )
    m_f.fit(X_ftr, y_ftr, sample_weight=sw_f, verbose=0)
    preds_fval = m_f.predict(X_fval)
    cv_accs.append(accuracy_score(y_fval, preds_fval))
    cv_f1s.append(f1_score(y_fval, preds_fval, average='macro', zero_division=0))

cv_acc_mean = float(np.mean(cv_accs))
cv_f1_mean  = float(np.mean(cv_f1s))
print(f"  5-fold CV: accuracy={cv_acc_mean*100:.2f}% ± {np.std(cv_accs)*100:.2f}%  "
      f"F1={cv_f1_mean:.4f} ± {np.std(cv_f1s):.4f}")

# ── Permutation importance (on test set, not train) ───────────────────────────
print("  Computing permutation importance (on held-out test set)...")
pi = permutation_importance(
    model, X_te, y_te, n_repeats=15, random_state=42, n_jobs=-1
)
perm_imp = {
    feature_names[i]: max(0.0, float(pi.importances_mean[i]))
    for i in range(len(feature_names))
}

# ── Save ──────────────────────────────────────────────────────────────────────
payload = {
    'model':            model,          # XGBoost fit on train split only
    'scaler':           scaler,         # fit on train split only
    'imputer':          imputer,        # fit on train split only
    'features':         feature_names,
    'feature_means':    {f: float(df[f].median()) for f in feature_names},
    'cv_mean_accuracy': cv_acc_mean,
    'cv_std_accuracy':  float(np.std(cv_accs)),
    'cv_mean_f1':       cv_f1_mean,
    'cv_std_f1':        float(np.std(cv_f1s)),
    'test_accuracy':    test_acc,
    'test_f1':          test_f1,
    'test_pathological_recall': test_path,
    'permutation_importance': perm_imp,
    'trained_on':       'UCI CTG dataset (fetal_health.csv) — 2,126 samples, 21 features',
    'architecture':     'FetalyzeAI v4.0: Regularized XGBoost (depth 3) — leakage-free',
    'version':          '4.0',
    'metadata': {
        'n_samples':    len(df),
        'n_features':   len(feature_names),
        'class_weights': {str(i): float(cw[i]) for i in range(3)},
        'preprocessing': 'leakage-free (fit on train split only)',
        'primary_metrics': 'macro-F1 and pathological recall (not full-dataset accuracy)',
        'xgb_params': {
            'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.03,
            'reg_alpha': 0.5, 'reg_lambda': 5.0,
        },
    },
}

with open('prediction_model.pkl', 'wb') as f:
    pickle.dump(payload, f)

print(f"  ✓ prediction_model.pkl saved (v4.0 — leakage-free)")
print(f"  Held-out test accuracy  : {test_acc*100:.2f}%")
print(f"  5-fold CV macro-F1      : {cv_f1_mean:.4f} ± {np.std(cv_f1s):.4f}  ← primary")
print(f"  Pathological recall     : {test_path*100:.2f}%  ← clinical priority")
print("Done.")
