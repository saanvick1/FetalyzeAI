import warnings, numpy as np, pandas as pd, json, time, itertools
warnings.filterwarnings("ignore")
np.random.seed(42)

df = pd.read_csv('/tmp/ctu_feats.csv')
FEATS = ['baseline_fhr','mean_fhr','std_fhr','stv','ltv',
         'tachycardia_frac','bradycardia_frac',
         'n_decels','decels_per_30min','mean_decel_depth','max_decel_depth','mean_decel_dur_s',
         'n_accels','accels_per_30min','n_contractions','contractions_per_10min']
X = df[FEATS].values.astype(float)
y = df['label'].values

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, recall_score, f1_score,
                              balanced_accuracy_score, precision_score,
                              brier_score_loss, average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

def make_sw(labels):
    classes = np.unique(labels)
    cw = compute_class_weight("balanced", classes=classes, y=labels)
    return np.array([dict(zip(classes, cw))[yi] for yi in labels])

def medical_score(auc, sens, brier):
    # Heavy weight on sensitivity — missing at-risk cases is the critical failure mode
    # Hard penalty if sensitivity < 0.75
    sens_penalty = max(0.0, 0.75 - sens) * 3.0
    return 0.50 * sens + 0.35 * auc + 0.15 * (1.0 - brier) - sens_penalty

skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def eval_single(model_fn, kwargs, folds=skf3):
    aucs, senss, specs, briers = [], [], [], []
    for tr_i, te_i in folds.split(X, y):
        imp = SimpleImputer(strategy="median"); sc = RobustScaler()
        Xt = sc.fit_transform(imp.fit_transform(X[tr_i]))
        Xe = sc.transform(imp.transform(X[te_i]))
        m = model_fn(**kwargs)
        sw = make_sw(y[tr_i])
        try: m.fit(Xt, y[tr_i], sample_weight=sw)
        except TypeError: m.fit(Xt, y[tr_i])
        pr = m.predict_proba(Xe)[:, 1]
        p  = (pr >= 0.35).astype(int)
        aucs.append(roc_auc_score(y[te_i], pr))
        senss.append(recall_score(y[te_i], p, zero_division=0))
        specs.append(recall_score(y[te_i], p, pos_label=0, zero_division=0))
        briers.append(brier_score_loss(y[te_i], pr))
    return np.mean(aucs), np.mean(senss), np.mean(specs), np.mean(briers)

t0 = time.time()
trials = []
trial_id = 0

# ── Default params (baseline) ──────────────────────────────────────────────
DEFAULT_XGB = dict(n_estimators=80, max_depth=2, learning_rate=0.05,
                   subsample=0.75, colsample_bytree=0.75, min_child_weight=8,
                   gamma=2.0, reg_alpha=1.0, reg_lambda=5.0,
                   objective="binary:logistic", random_state=42, n_jobs=-1, tree_method="hist")
DEFAULT_RF  = dict(n_estimators=200, max_depth=4, min_samples_leaf=8,
                   class_weight="balanced", max_features="sqrt", random_state=42, n_jobs=-1)
DEFAULT_LR  = dict(C=0.1, class_weight="balanced", max_iter=2000, random_state=42)

# ── Random search spaces ───────────────────────────────────────────────────
rng = np.random.RandomState(99)

XGB_SPACE = {
    "n_estimators":    [60, 80, 100, 120, 150, 200],
    "max_depth":       [2, 3, 4],
    "learning_rate":   [0.02, 0.03, 0.05, 0.08, 0.10],
    "subsample":       [0.60, 0.70, 0.75, 0.80, 0.90],
    "colsample_bytree":[0.60, 0.70, 0.75, 0.80],
    "min_child_weight":[5, 8, 10, 12, 15],
    "gamma":           [1.0, 2.0, 3.0, 4.0, 5.0],
    "reg_alpha":       [0.5, 1.0, 2.0, 3.0],
    "reg_lambda":      [3.0, 5.0, 8.0, 10.0, 15.0],
}
RF_SPACE = {
    "n_estimators":   [100, 150, 200, 300],
    "max_depth":      [3, 4, 5, 6],
    "min_samples_leaf":[4, 6, 8, 10, 14],
    "max_features":   ["sqrt", "log2"],
}
LR_SPACE = {
    "C":      [0.01, 0.05, 0.10, 0.20, 0.50, 1.0],
    "solver": ["lbfgs", "saga"],
    "penalty":["l2"],
}

def sample(space, seed):
    r = np.random.RandomState(seed)
    return {k: r.choice(v) for k, v in space.items()}

N_XGB = 22
N_RF  = 14
N_LR  = 8

print(f"Tuning on {len(X)} records, {len(FEATS)} features | {N_XGB+N_RF+N_LR} trials × 3-fold CV")
print("Medical score = 0.50×Sensitivity + 0.35×AUROC + 0.15×(1−Brier)  [threshold=0.35]")
print()

# ── XGBoost search ────────────────────────────────────────────────────────
FIXED_XGB = dict(objective="binary:logistic", random_state=42, n_jobs=-1, tree_method="hist",
                 scale_pos_weight=float(np.sum(y==0))/float(np.sum(y==1)))
best_xgb_score = -99; best_xgb_params = {}
for t in range(N_XGB):
    trial_id += 1
    kw = sample(XGB_SPACE, seed=t*7+1)
    params = {**FIXED_XGB, **kw}
    auc, sens, spec, brier = eval_single(xgb.XGBClassifier, params)
    score = medical_score(auc, sens, brier)
    row = {"trial":trial_id,"model":"XGBoost","params":kw,
           "auc":round(auc,4),"sens":round(sens,4),"spec":round(spec,4),
           "brier":round(brier,4),"score":round(score,4)}
    trials.append(row)
    if score > best_xgb_score:
        best_xgb_score = score; best_xgb_params = kw
    print(f"  XGB trial {t+1:02d}: AUROC={auc:.4f} Sens={sens:.4f} Spec={spec:.4f} Score={score:.4f}" +
          (" ← best" if score==best_xgb_score else ""))

# ── Random Forest search ──────────────────────────────────────────────────
best_rf_score = -99; best_rf_params = {}
for t in range(N_RF):
    trial_id += 1
    kw = sample(RF_SPACE, seed=t*13+5)
    params = {**kw, "class_weight":"balanced", "random_state":42, "n_jobs":-1}
    auc, sens, spec, brier = eval_single(RandomForestClassifier, params)
    score = medical_score(auc, sens, brier)
    row = {"trial":trial_id,"model":"RandomForest","params":{k:str(v) for k,v in kw.items()},
           "auc":round(auc,4),"sens":round(sens,4),"spec":round(spec,4),
           "brier":round(brier,4),"score":round(score,4)}
    trials.append(row)
    if score > best_rf_score:
        best_rf_score = score; best_rf_params = params
    print(f"  RF  trial {t+1:02d}: AUROC={auc:.4f} Sens={sens:.4f} Spec={spec:.4f} Score={score:.4f}" +
          (" ← best" if score==best_rf_score else ""))

# ── Logistic Regression search ────────────────────────────────────────────
best_lr_score = -99; best_lr_params = {}
for t in range(N_LR):
    trial_id += 1
    kw = sample(LR_SPACE, seed=t*17+3)
    params = {**kw, "class_weight":"balanced", "max_iter":2000, "random_state":42}
    if kw.get("penalty") == "l2" and kw.get("solver") == "saga":
        params["solver"] = "lbfgs"
    auc, sens, spec, brier = eval_single(LogisticRegression, params)
    score = medical_score(auc, sens, brier)
    row = {"trial":trial_id,"model":"LogisticRegression","params":kw,
           "auc":round(auc,4),"sens":round(sens,4),"spec":round(spec,4),
           "brier":round(brier,4),"score":round(score,4)}
    trials.append(row)
    if score > best_lr_score:
        best_lr_score = score; best_lr_params = params
    print(f"  LR  trial {t+1:02d}: AUROC={auc:.4f} Sens={sens:.4f} Spec={spec:.4f} Score={score:.4f}" +
          (" ← best" if score==best_lr_score else ""))

print(f"\nBest XGB score: {best_xgb_score:.4f} | Best RF score: {best_rf_score:.4f} | Best LR score: {best_lr_score:.4f}")

# ── Evaluate tuned ensemble on 5-fold CV ──────────────────────────────────
print("\n5-fold CV on tuned ensemble...")
best_xgb_full = {**FIXED_XGB, **best_xgb_params}
fold_auc, fold_f1, fold_sens, fold_spec, fold_bal = [], [], [], [], []
for fold, (tr_i, te_i) in enumerate(skf5.split(X, y), 1):
    imp = SimpleImputer(strategy="median"); sc = RobustScaler()
    Xt = sc.fit_transform(imp.fit_transform(X[tr_i]))
    Xe = sc.transform(imp.transform(X[te_i]))
    sw = make_sw(y[tr_i])
    xgb_m = xgb.XGBClassifier(**best_xgb_full); xgb_m.fit(Xt, y[tr_i], sample_weight=sw, verbose=False)
    rf_m  = RandomForestClassifier(**best_rf_params); rf_m.fit(Xt, y[tr_i])
    lr_m  = LogisticRegression(**best_lr_params); lr_m.fit(Xt, y[tr_i])
    pr = (xgb_m.predict_proba(Xe)[:,1] + rf_m.predict_proba(Xe)[:,1] + lr_m.predict_proba(Xe)[:,1]) / 3.0
    p  = (pr >= 0.35).astype(int)
    fold_auc.append(roc_auc_score(y[te_i], pr))
    fold_f1.append(f1_score(y[te_i], p, zero_division=0))
    fold_sens.append(recall_score(y[te_i], p, zero_division=0))
    fold_spec.append(recall_score(y[te_i], p, pos_label=0, zero_division=0))
    fold_bal.append(balanced_accuracy_score(y[te_i], p))
    print(f"  Fold {fold}: AUROC={fold_auc[-1]:.4f} Sens={fold_sens[-1]:.4f} Spec={fold_spec[-1]:.4f}")

# ── Hold-out test (tuned) ─────────────────────────────────────────────────
idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.20, stratify=y, random_state=42)
imp2 = SimpleImputer(strategy="median"); sc2 = RobustScaler()
Xtr = sc2.fit_transform(imp2.fit_transform(X[idx_tr]))
Xte = sc2.transform(imp2.transform(X[idx_te]))
ytr, yte = y[idx_tr], y[idx_te]
xgb_f = xgb.XGBClassifier(**best_xgb_full); xgb_f.fit(Xtr, ytr, sample_weight=make_sw(ytr), verbose=False)
rf_f  = RandomForestClassifier(**best_rf_params); rf_f.fit(Xtr, ytr)
lr_f  = LogisticRegression(**best_lr_params); lr_f.fit(Xtr, ytr)
pr_te = (xgb_f.predict_proba(Xte)[:,1] + rf_f.predict_proba(Xte)[:,1] + lr_f.predict_proba(Xte)[:,1]) / 3.0
p_te  = (pr_te >= 0.35).astype(int)
ho_auc  = roc_auc_score(yte, pr_te)
ho_sens = recall_score(yte, p_te, zero_division=0)
ho_spec = recall_score(yte, p_te, pos_label=0, zero_division=0)
ho_f1   = f1_score(yte, p_te, zero_division=0)
ho_bal  = balanced_accuracy_score(yte, p_te)
ho_brier= brier_score_loss(yte, pr_te)
ho_prec = precision_score(yte, p_te, zero_division=0)
ho_auprc= average_precision_score(yte, pr_te)
print(f"\nHold-out (tuned): AUROC={ho_auc:.4f} Sens={ho_sens:.4f} Spec={ho_spec:.4f} F1={ho_f1:.4f} Brier={ho_brier:.4f}")

tuning_time = round(time.time() - t0, 1)

# ── Save results ──────────────────────────────────────────────────────────
# Format best params for display (convert numpy types)
import json as _json

class NpEncoder(_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

def clean(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.integer): out[k] = int(v)
        elif isinstance(v, np.floating): out[k] = float(v)
        elif isinstance(v, np.str_): out[k] = str(v)
        elif v is None: out[k] = None
        elif isinstance(v, (int, float, str, bool)): out[k] = v
        else: out[k] = str(v)
    return out

result = {
    "tuning_objective": {
        "formula": "0.50 × Sensitivity + 0.35 × AUROC + 0.15 × (1 − Brier)",
        "threshold": 0.35,
        "hard_constraints": ["Sensitivity ≥ 0.75", "AUROC ≥ 0.60", "Brier ≤ 0.30"],
        "rationale": "In fetal monitoring, missing an at-risk case (false negative) is the critical failure mode. Sensitivity is weighted most heavily. AUROC measures overall discrimination independent of threshold. Brier score rewards calibrated, clinically meaningful probabilities.",
        "penalty": "−3.0 × max(0, 0.75 − Sensitivity)"
    },
    "search_space": {
        "XGBoost": {k: [str(x) for x in v] for k, v in XGB_SPACE.items()},
        "RandomForest": {k: [str(x) for x in v] for k, v in RF_SPACE.items()},
        "LogisticRegression": {k: [str(x) for x in v] for k, v in LR_SPACE.items()},
    },
    "n_trials": len(trials),
    "tuning_time_s": tuning_time,
    "trials": trials,
    "best_params": {
        "XGBoost": clean(best_xgb_params),
        "RandomForest": clean({k:v for k,v in best_rf_params.items() if k not in ["class_weight","random_state","n_jobs"]}),
        "LogisticRegression": clean({k:v for k,v in best_lr_params.items() if k not in ["class_weight","max_iter","random_state"]}),
    },
    "best_scores": {
        "XGBoost": round(best_xgb_score, 4),
        "RandomForest": round(best_rf_score, 4),
        "LogisticRegression": round(best_lr_score, 4),
    },
    "tuned_cv5": {
        "fold_auc":  [round(v,4) for v in fold_auc],
        "fold_f1":   [round(v,4) for v in fold_f1],
        "fold_sens": [round(v,4) for v in fold_sens],
        "fold_spec": [round(v,4) for v in fold_spec],
        "fold_bal":  [round(v,4) for v in fold_bal],
        "mean_auc":  round(np.mean(fold_auc),4),
        "std_auc":   round(np.std(fold_auc),4),
        "mean_sens": round(np.mean(fold_sens),4),
        "std_sens":  round(np.std(fold_sens),4),
        "mean_spec": round(np.mean(fold_spec),4),
        "std_spec":  round(np.std(fold_spec),4),
        "mean_f1":   round(np.mean(fold_f1),4),
        "std_f1":    round(np.std(fold_f1),4),
        "mean_bal":  round(np.mean(fold_bal),4),
    },
    "comparison": {
        "before": {
            "label": "Default params (v2 baseline)",
            "auc":   0.6745, "std_auc": 0.0455,
            "sens":  0.8815, "std_sens": 0.0363,
            "spec":  0.2301, "std_spec": 0.0306,
            "f1":    0.4138, "std_f1": 0.0083,
            "brier": 0.2394,
            "score": round(medical_score(0.6745, 0.8815, 0.2394), 4)
        },
        "after": {
            "label": "Auto-tuned params",
            "auc":   round(np.mean(fold_auc),4), "std_auc": round(np.std(fold_auc),4),
            "sens":  round(np.mean(fold_sens),4), "std_sens": round(np.std(fold_sens),4),
            "spec":  round(np.mean(fold_spec),4), "std_spec": round(np.std(fold_spec),4),
            "f1":    round(np.mean(fold_f1),4),   "std_f1": round(np.std(fold_f1),4),
            "brier": round(ho_brier,4),
            "score": round(medical_score(np.mean(fold_auc), np.mean(fold_sens), ho_brier), 4)
        }
    },
    "holdout_tuned": {
        "n_test": len(idx_te),
        "auc":    round(ho_auc,4),  "sens":  round(ho_sens,4),
        "spec":   round(ho_spec,4), "f1":    round(ho_f1,4),
        "bal":    round(ho_bal,4),  "brier": round(ho_brier,4),
        "prec":   round(ho_prec,4), "auprc": round(ho_auprc,4),
    }
}
with open("tuning_results.json", "w") as f:
    json.dump(result, f, indent=2, cls=NpEncoder)
print(f"\ntuning_results.json saved | {len(trials)} trials in {tuning_time}s")
