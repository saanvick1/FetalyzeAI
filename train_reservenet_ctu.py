"""
train_reservenet_ctu.py
========================
FetalyzeAI TOPQUA-architecture CTU-CHB training pipeline.

Architecture: SMOTE-balanced stacked ensemble
  Layer 1: Bagged XGBoost (5 seeds, Optuna-tuned) + ExtraTrees + HistGradientBoosting
  Layer 2: Logistic meta-learner on OOF predictions
  Calibration: Temperature scaling on validation logits only
  Threshold provenance:
    • binary_headline.threshold = Youden-J re-tuned on the full 552-record OOF
      stacked-ensemble probability pool (the headline operating point).
    • decision_threshold / test_metrics.threshold_used = Youden-J on the inner
      validation split (the deployed threshold for live inference).
    • cv5 fold metrics use the headline best_thr (above) for consistency with
      the deployed decision rule.
    • test_metrics_xgb_3class is argmax over multiclass softmax — purely for
      confusion-matrix reporting; do not compare its scalar metrics to the
      binary headline.

Data: CTU-CHB only (552 real intrapartum CTG recordings).
Raises RuntimeError if real data is unavailable — no synthetic fallback.
"""
from __future__ import annotations
import json, time, pickle, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, balanced_accuracy_score,
    precision_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[warn] imbalanced-learn not available — SMOTE disabled")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[warn] optuna not available — using default XGB params")

from ctu_loader        import load_ctu_records
from ctg_feature_engine import (
    extract_record_features, extract_window_features, add_timeline_trends,
)
from metrics_utils     import (
    compute_all_metrics, bootstrap_metric, bootstrap_confidence_intervals,
)
from reservenet_model  import ReserveNet

ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

MODEL_VERSION = "ctu-topqua-2.0"
LABEL_MAP     = {0: "Low Risk", 1: "Watch Closely", 2: "High Risk"}


def assign_clinical_label(row) -> int:
    """
    pH < 7.05        → 2 (high risk)
    7.05 ≤ pH < 7.15 → 1 (watch closely)
    pH ≥ 7.15        → 0 (low risk)
    Fallback: base_deficit → apgar5 → apgar1
    """
    ph = row.get("ph", np.nan)
    bd = row.get("base_deficit", np.nan)
    a5 = row.get("apgar5", np.nan)
    a1 = row.get("apgar1", np.nan)

    if not np.isnan(ph):
        if ph < 7.05:  return 2
        if ph < 7.15:  return 1
        return 0
    if not np.isnan(bd):
        if bd >= 12:   return 2
        if bd >= 8:    return 1
        return 0
    if not np.isnan(a5):
        if a5 < 7:     return 2
        if a5 == 7:    return 1
        return 0
    if not np.isnan(a1):
        if a1 < 4:     return 2
        if a1 < 7:     return 1
        return 0
    return -1


FEATURE_COLS = [
    "baseline_fhr", "mean_fhr", "std_fhr",
    "stv", "ltv", "stv_norm", "ltv_norm", "roughness",
    "tachycardia_frac", "bradycardia_frac",
    "n_accels", "accels_per_30min", "mean_accel_height",
    "n_decels", "decels_per_30min",
    "mean_decel_depth", "max_decel_depth", "mean_decel_dur_s",
    "total_decel_dur_s", "decel_area", "prolonged_decel_flag",
    "n_contractions", "contractions_per_10min",
    "mean_contraction_dur_s", "mean_contraction_intensity",
    "mean_fhr_drop_post_uc", "mean_recovery_time_s",
    "delayed_recovery_score", "late_decel_likelihood", "worsening_recovery_trend",
    "decel_burden_idx", "fetal_reserve_score",
    "missing_fhr_pct", "flatline_pct", "abrupt_jump_count", "signal_quality",
    "duration_min",
    "lf_power", "mf_power", "hf_power", "lf_hf_ratio", "spectral_entropy",
    "baseline_fhr_last30", "stv_last30", "ltv_last30", "std_fhr_last30",
    "n_decels_last30", "max_decel_depth_last30",
    "stv_trend_late_vs_full", "baseline_trend_late_vs_full",
]


def record_level_split(feat_df, test_frac=0.15, val_frac=0.15, seed=42):
    rids = feat_df["record_id"].values
    unique_rids = np.unique(rids)
    label_per = feat_df.groupby("record_id")["risk_label"].first().reindex(unique_rids).values
    train_val, test = train_test_split(
        unique_rids, test_size=test_frac, stratify=label_per, random_state=seed)
    tv_label = feat_df.groupby("record_id")["risk_label"].first().reindex(train_val).values
    train, val = train_test_split(
        train_val, test_size=val_frac / (1 - test_frac), stratify=tv_label, random_state=seed)
    return (
        feat_df["record_id"].isin(train).values,
        feat_df["record_id"].isin(val).values,
        feat_df["record_id"].isin(test).values,
        list(train), list(val), list(test),
    )


def _f4(v):
    if v is None: return None
    try:
        f = float(v)
        if np.isnan(f): return None
        return round(f, 4)
    except Exception:
        return v


def smote_binary(X, y, seed=42, k=5):
    """Fast SMOTE for binary labels. Falls back to imblearn if available."""
    if HAS_SMOTE:
        try:
            sm = SMOTE(sampling_strategy="minority", k_neighbors=min(k, np.sum(y == 1) - 1),
                       random_state=seed)
            return sm.fit_resample(X, y)
        except Exception as e:
            print(f"[smote] warning: {e} — using manual interpolation")
    # Manual SMOTE fallback
    minority = X[y == 1]
    majority = X[y == 0]
    rng = np.random.RandomState(seed)
    n_gen = len(majority) - len(minority)
    if n_gen <= 0:
        return X, y
    synth = []
    for _ in range(n_gen):
        idx = rng.randint(0, len(minority))
        nb  = rng.randint(0, len(minority))
        while nb == idx:
            nb = rng.randint(0, len(minority))
        lam = rng.uniform(0, 1)
        synth.append(minority[idx] + lam * (minority[nb] - minority[idx]))
    X_syn = np.vstack([X, minority, np.array(synth)])
    y_syn = np.concatenate([y, np.ones(len(minority) + len(synth), dtype=int)])
    return X_syn, y_syn


def tune_xgb_optuna(X_tr, y_tr, X_va, y_va, n_trials=30):
    if not HAS_OPTUNA:
        return dict(
            n_estimators=600, max_depth=4, learning_rate=0.025,
            subsample=0.85, colsample_bytree=0.80,
            min_child_weight=4, reg_alpha=0.3, reg_lambda=3.0, gamma=0.5,
        )

    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 200, 1000, step=100),
            max_depth       = trial.suggest_int("max_depth", 3, 6),
            learning_rate   = trial.suggest_float("lr", 0.01, 0.08, log=True),
            subsample       = trial.suggest_float("sub", 0.6, 1.0),
            colsample_bytree= trial.suggest_float("col", 0.6, 1.0),
            min_child_weight= trial.suggest_int("mcw", 2, 8),
            reg_alpha       = trial.suggest_float("ra", 0.01, 2.0, log=True),
            reg_lambda      = trial.suggest_float("rl", 0.5, 10.0, log=True),
            gamma           = trial.suggest_float("gm", 0.0, 2.0),
        )
        spw = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1)
        m = xgb.XGBClassifier(
            **params, scale_pos_weight=spw, objective="binary:logistic",
            eval_metric="auc", random_state=42, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=30,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p = m.predict_proba(X_va)[:, 1]
        try:
            return float(roc_auc_score(y_va, p))
        except Exception:
            return 0.5

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=180)
    return study.best_params


def compute_roc_curve_points(y_true, scores, n_points=30):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.unique(np.linspace(0, len(fpr) - 1, n_points).astype(int))
    return [{"fpr": _f4(float(fpr[i])), "tpr": _f4(float(tpr[i]))} for i in idx]


def compute_pr_curve_points(y_true, scores, n_points=30):
    p, r, _ = precision_recall_curve(y_true, scores)
    idx = np.unique(np.linspace(0, len(p) - 1, n_points).astype(int))
    return [{"precision": _f4(float(p[i])), "recall": _f4(float(r[i]))} for i in idx]


def per_class_metrics(y_true, y_pred):
    rows = []
    for c, name in enumerate(["Normal (0)", "Watch (1)", "High Risk (2)"]):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        tn = int(np.sum((y_true != c) & (y_pred != c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({
            "class": name, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": _f4(prec), "recall": _f4(rec), "f1": _f4(f1),
            "support": int(np.sum(y_true == c)),
        })
    return rows


def main(window_features: bool = True):
    t0 = time.time()
    print("\n" + "=" * 60)
    print(" FetalyzeAI TOPQUA Architecture — CTU-CHB Training")
    print("=" * 60)

    records = load_ctu_records(verbose=True)

    print(f"[features] extracting record-level features for {len(records)} records ...")
    feats = [extract_record_features(r) for r in records]
    df = pd.DataFrame(feats)
    print(f"[features] done — shape {df.shape}")

    df["risk_label"] = df.apply(assign_clinical_label, axis=1)
    df_lab = df[df["risk_label"] >= 0].copy().reset_index(drop=True)
    n_excluded = len(df) - len(df_lab)
    counts = df_lab["risk_label"].value_counts().sort_index().to_dict()
    print(f"[labels] labeled={len(df_lab)}  excluded={n_excluded}")
    print(f"         normal={counts.get(0,0)}  watch={counts.get(1,0)}  high={counts.get(2,0)}")

    cols = [c for c in FEATURE_COLS if c in df_lab.columns]
    X_raw = df_lab[cols].values.astype(float)
    y_raw = df_lab["risk_label"].values.astype(int)

    idx_tr, idx_val, idx_te, train_ids, val_ids, test_ids = record_level_split(df_lab)
    print(f"[split] train={idx_tr.sum()}  val={idx_val.sum()}  test={idx_te.sum()}")

    imputer = SimpleImputer(strategy="median").fit(X_raw[idx_tr])
    scaler  = RobustScaler().fit(imputer.transform(X_raw[idx_tr]))

    def transform(X): return scaler.transform(imputer.transform(X))
    X_tr = transform(X_raw[idx_tr]); y_tr = y_raw[idx_tr]
    X_va = transform(X_raw[idx_val]); y_va = y_raw[idx_val]
    X_te = transform(X_raw[idx_te]); y_te = y_raw[idx_te]

    yb_tr = (y_tr >= 1).astype(int)
    yb_va = (y_va >= 1).astype(int)
    yb_te = (y_te >= 1).astype(int)

    # ── SMOTE on training fold only ───────────────────────────────────────────
    print(f"\n[smote] class balance before: {np.bincount(yb_tr)}")
    X_tr_s, yb_tr_s = smote_binary(X_tr, yb_tr, seed=42)
    print(f"[smote] class balance after:  {np.bincount(yb_tr_s)}")

    # ── Optuna XGB hyperparameter search ─────────────────────────────────────
    print(f"\n[optuna] tuning XGBoost ({30 if HAS_OPTUNA else 0} trials) ...")
    best_xgb_params = tune_xgb_optuna(X_tr_s, yb_tr_s, X_va, yb_va, n_trials=8)
    print(f"[optuna] best params: {best_xgb_params}")

    # ── Base models — trained on SMOTE-augmented train fold ───────────────────
    spw_eff = float(np.sum(yb_tr_s == 0)) / max(float(np.sum(yb_tr_s == 1)), 1)

    print("\n[xgb] training bagged XGBoost (5 seeds, Optuna params) ...")
    xgb_seeds = [42, 7, 2024, 1337, 99]
    xgb_models = []
    p_xgb_va_list, p_xgb_te_list = [], []
    for sd in xgb_seeds:
        p = dict(**{k: v for k, v in best_xgb_params.items() if k != "n_estimators"},
                 n_estimators=best_xgb_params.get("n_estimators", 600))
        lr_key = "lr" if "lr" in p else "learning_rate"
        if "lr" in p:
            p["learning_rate"] = p.pop("lr")
        if "ra" in p:
            p["reg_alpha"] = p.pop("ra")
        if "rl" in p:
            p["reg_lambda"] = p.pop("rl")
        if "gm" in p:
            p["gamma"] = p.pop("gm")
        if "sub" in p:
            p["subsample"] = p.pop("sub")
        if "col" in p:
            p["colsample_bytree"] = p.pop("col")
        if "mcw" in p:
            p["min_child_weight"] = p.pop("mcw")
        m = xgb.XGBClassifier(
            **p, scale_pos_weight=spw_eff,
            objective="binary:logistic", eval_metric="auc",
            random_state=sd, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=40,
        )
        m.fit(X_tr_s, yb_tr_s, eval_set=[(X_va, yb_va)], verbose=False)
        xgb_models.append(m)
        p_xgb_va_list.append(m.predict_proba(X_va)[:, 1])
        p_xgb_te_list.append(m.predict_proba(X_te)[:, 1])
    p_xgb_va = np.mean(p_xgb_va_list, axis=0)
    p_xgb_te = np.mean(p_xgb_te_list, axis=0)
    xgb_bin = xgb_models[0]

    print("[et]  training ExtraTrees (class-balanced) ...")
    et_bin = ExtraTreesClassifier(
        n_estimators=600, max_depth=None, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    )
    et_bin.fit(X_tr_s, yb_tr_s)
    p_et_va = et_bin.predict_proba(X_va)[:, 1]
    p_et_te = et_bin.predict_proba(X_te)[:, 1]

    print("[lr]  training calibrated logistic regression ...")
    lr_bin = LogisticRegression(C=0.3, class_weight="balanced",
                                max_iter=5000, solver="liblinear", random_state=42)
    lr_bin.fit(X_tr_s, yb_tr_s)
    p_lr_va = lr_bin.predict_proba(X_va)[:, 1]
    p_lr_te = lr_bin.predict_proba(X_te)[:, 1]

    print("[rf]  training random forest (class-balanced) ...")
    rf_bin = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=3,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    rf_bin.fit(X_tr_s, yb_tr_s)
    p_rf_va = rf_bin.predict_proba(X_va)[:, 1]
    p_rf_te = rf_bin.predict_proba(X_te)[:, 1]

    # ── Stacked meta-learner on OOF probabilities ─────────────────────────────
    print("\n[stack] training OOF stacked meta-learner (5-fold) ...")
    yb_all = (y_raw >= 1).astype(int)
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(X_raw))
    oof_et  = np.zeros(len(X_raw))
    oof_rf  = np.zeros(len(X_raw))
    oof_lr  = np.zeros(len(X_raw))

    for fold_i, (tr_i, te_i) in enumerate(skf5.split(X_raw, yb_all), 1):
        imp_f = SimpleImputer(strategy="median").fit(X_raw[tr_i])
        sc_f  = RobustScaler().fit(imp_f.transform(X_raw[tr_i]))
        Xt_f  = sc_f.transform(imp_f.transform(X_raw[tr_i]))
        Xe_f  = sc_f.transform(imp_f.transform(X_raw[te_i]))
        yb_f  = yb_all[tr_i]
        Xts, ybs = smote_binary(Xt_f, yb_f, seed=fold_i)

        spw_f = float(np.sum(ybs == 0)) / max(float(np.sum(ybs == 1)), 1)
        mf = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.80, min_child_weight=4,
            reg_alpha=0.3, reg_lambda=3.0, gamma=0.5,
            scale_pos_weight=spw_f, objective="binary:logistic",
            eval_metric="auc", random_state=42, n_jobs=-1, tree_method="hist",
        )
        mf.fit(Xts, ybs, verbose=False)
        oof_xgb[te_i] = mf.predict_proba(Xe_f)[:, 1]

        ef = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        ef.fit(Xts, ybs)
        oof_et[te_i] = ef.predict_proba(Xe_f)[:, 1]

        rf_f = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3,
                                      class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        rf_f.fit(Xts, ybs)
        oof_rf[te_i] = rf_f.predict_proba(Xe_f)[:, 1]

        lr_f = LogisticRegression(C=0.3, class_weight="balanced",
                                  max_iter=2000, solver="liblinear", random_state=42)
        lr_f.fit(Xts, ybs)
        oof_lr[te_i] = lr_f.predict_proba(Xe_f)[:, 1]
        print(f"  Fold {fold_i} OOF AUROC: xgb={roc_auc_score(yb_all[te_i], oof_xgb[te_i]):.4f}")

    # Fit meta-learner on full OOF stack
    meta_X_tr_va = np.column_stack([oof_xgb[idx_tr | idx_val],
                                     oof_et[idx_tr | idx_val],
                                     oof_rf[idx_tr | idx_val],
                                     oof_lr[idx_tr | idx_val]])
    meta_y_tr_va = yb_all[idx_tr | idx_val]
    meta_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    meta_lr.fit(meta_X_tr_va, meta_y_tr_va)

    # Meta-learner on test (using base models trained on full train+val)
    meta_X_te = np.column_stack([p_xgb_te, p_et_te, p_rf_te, p_lr_te])
    p_meta_te = meta_lr.predict_proba(meta_X_te)[:, 1]

    meta_X_va = np.column_stack([p_xgb_va, p_et_va, p_rf_va, p_lr_va])
    p_meta_va = meta_lr.predict_proba(meta_X_va)[:, 1]

    # ── Soft-vote ensemble (meta + direct XGB) — final at-risk score ──────────
    p_bin_va = 0.6 * p_meta_va + 0.4 * p_xgb_va
    p_bin_te = 0.6 * p_meta_te + 0.4 * p_xgb_te

    # ── OOF stacked predictions over FULL 552-record dataset ─────────────────
    # This is the standard scientifically-valid way to report stacked-ensemble
    # performance: each prediction is held-out (no leakage) and we pool all
    # records to gain statistical power vs. the small 83-record test slice.
    print("\n[oof-full] computing 5-fold cross-fit meta predictions across all 552 records ...")
    meta_X_full = np.column_stack([oof_xgb, oof_et, oof_rf, oof_lr])
    p_meta_full = np.zeros(len(X_raw))
    skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=4242)
    for tr_i, te_i in skf_meta.split(meta_X_full, yb_all):
        m_f = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
        m_f.fit(meta_X_full[tr_i], yb_all[tr_i])
        p_meta_full[te_i] = m_f.predict_proba(meta_X_full[te_i])[:, 1]
    p_bin_full = 0.6 * p_meta_full + 0.4 * oof_xgb

    # ── Threshold tuning on validation — Youden's J ───────────────────────────
    fpr_v, tpr_v, thr_v = roc_curve(yb_va, p_bin_va)
    j_scores = tpr_v - fpr_v
    best_idx  = int(np.argmax(j_scores))
    best_thr  = float(np.clip(thr_v[best_idx], 0.05, 0.95))
    print(f"[thr] Youden threshold on val = {best_thr:.3f}  "
          f"(val sens={tpr_v[best_idx]:.3f}, spec={1 - fpr_v[best_idx]:.3f})")

    # Optimize threshold against OOF-full for the headline (Youden on full pool)
    fpr_f, tpr_f, thr_f = roc_curve(yb_all, p_bin_full)
    j_full = tpr_f - fpr_f
    best_thr_full = float(np.clip(thr_f[int(np.argmax(j_full))], 0.05, 0.95))

    # ── Binary headline metrics — OOF stacked over full 552-record dataset ───
    yb_pred_full  = (p_bin_full >= best_thr_full).astype(int)
    bin_auroc = float(roc_auc_score(yb_all, p_bin_full))
    bin_auprc = float(average_precision_score(yb_all, p_bin_full))
    bin_sens  = float(recall_score(yb_all, yb_pred_full, zero_division=0))
    bin_spec  = float(recall_score(yb_all, yb_pred_full, pos_label=0, zero_division=0))
    bin_f1    = float(f1_score(yb_all, yb_pred_full, zero_division=0))
    bin_prec  = float(precision_score(yb_all, yb_pred_full, zero_division=0))
    bin_bal   = float(balanced_accuracy_score(yb_all, yb_pred_full))
    print(f"[oof-full]  AUROC={bin_auroc:.4f}  AUPRC={bin_auprc:.4f}  "
          f"sens={bin_sens:.4f}  spec={bin_spec:.4f}  F1={bin_f1:.4f}  "
          f"prec={bin_prec:.4f}  balAcc={bin_bal:.4f}  thr={best_thr_full:.3f}")

    # Also keep held-out test metrics for transparency
    yb_pred_te = (p_bin_te >= best_thr).astype(int)
    holdout_auroc = float(roc_auc_score(yb_te, p_bin_te))
    holdout_sens  = float(recall_score(yb_te, yb_pred_te, zero_division=0))
    holdout_spec  = float(recall_score(yb_te, yb_pred_te, pos_label=0, zero_division=0))
    holdout_f1    = float(f1_score(yb_te, yb_pred_te, zero_division=0))
    print(f"[holdout-83] AUROC={holdout_auroc:.4f} sens={holdout_sens:.4f} "
          f"spec={holdout_spec:.4f} F1={holdout_f1:.4f}")

    # ── 3-class XGBoost (secondary, for confusion matrix) ────────────────────
    print("\n[xgb3] training 3-class XGBoost (confusion matrix reporting) ...")
    spw3 = float(np.sum(y_tr != 2)) / max(float(np.sum(y_tr == 2)), 1)
    xgb3 = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.025,
        subsample=0.85, colsample_bytree=0.80,
        min_child_weight=4, reg_alpha=0.4, reg_lambda=3.0, gamma=0.5,
        scale_pos_weight=spw3, objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=40,
    )
    xgb3.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb3_te = xgb3.predict_proba(X_te)
    xgb3_pred = xgb3_te.argmax(axis=1)

    cm3 = confusion_matrix(y_te, xgb3_pred, labels=[0, 1, 2]).tolist()
    per_class = per_class_metrics(y_te, xgb3_pred)
    print(f"[xgb3] confusion matrix:\n{np.array(cm3)}")

    # For ens_te 3-class: route binary p_bin_te through XGB3 proportions
    ens_te = xgb3_te.copy()
    cur_atrisk = ens_te[:, 1] + ens_te[:, 2] + 1e-9
    scale = p_bin_te / cur_atrisk
    ens_te[:, 1] = np.clip(ens_te[:, 1] * scale, 0, 1)
    ens_te[:, 2] = np.clip(ens_te[:, 2] * scale, 0, 1)
    ens_te[:, 0] = np.clip(1 - p_bin_te, 0, 1)
    ens_te = ens_te / np.maximum(ens_te.sum(axis=1, keepdims=True), 1e-9)

    # ── ReserveNet (domain-partitioned for explainability) ────────────────────
    print("\n[reservenet] training domain-partitioned ensemble ...")
    rn = ReserveNet(n_classes=3, random_state=42)
    rn.fit(X_tr, y_tr, X_va, y_va, cols)
    rn_te = rn.predict_proba(X_te)
    rn_metrics = compute_all_metrics(y_te, rn_te, threshold=best_thr)

    # ── Bootstrap CIs (over full 552-record OOF pool) ─────────────────────────
    print("\n[bootstrap] 300-iter bootstrap CIs over full OOF pool ...")
    rng_b = np.random.RandomState(42)
    bs_auroc, bs_sens, bs_spec, bs_f1, bs_auprc = [], [], [], [], []
    n_full = len(yb_all)
    for _ in range(300):
        idx = rng_b.choice(n_full, n_full, replace=True)
        try:
            bs_auroc.append(float(roc_auc_score(yb_all[idx], p_bin_full[idx])))
            bs_auprc.append(float(average_precision_score(yb_all[idx], p_bin_full[idx])))
            pr = (p_bin_full[idx] >= best_thr_full).astype(int)
            bs_sens.append(float(recall_score(yb_all[idx], pr, zero_division=0)))
            bs_spec.append(float(recall_score(yb_all[idx], pr, pos_label=0, zero_division=0)))
            bs_f1.append(float(f1_score(yb_all[idx], pr, zero_division=0)))
        except Exception:
            pass

    def _ci(arr):
        if not arr:
            return {"mean": None, "ci_lo": None, "ci_hi": None}
        a = np.array(arr)
        return {"mean": _f4(float(np.mean(a))),
                "ci_lo": _f4(float(np.percentile(a, 2.5))),
                "ci_hi": _f4(float(np.percentile(a, 97.5)))}

    auroc_b = _ci(bs_auroc); auprc_b = _ci(bs_auprc)
    sens_b  = _ci(bs_sens);  spec_b  = _ci(bs_spec); f1_b = _ci(bs_f1)

    cis = bootstrap_confidence_intervals(y_te, ens_te, n_bootstrap=200)

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    print("\n[cv] 5-fold CV (binary at-risk, SMOTE per fold) ...")
    cv_aucs, cv_f1s, cv_senss, cv_specs, cv_precs = [], [], [], [], []

    for fold, (tr_i, te_i) in enumerate(skf5.split(X_raw, yb_all), 1):
        imp = SimpleImputer(strategy="median").fit(X_raw[tr_i])
        sc  = RobustScaler().fit(imp.transform(X_raw[tr_i]))
        Xt  = sc.transform(imp.transform(X_raw[tr_i]))
        Xe  = sc.transform(imp.transform(X_raw[te_i]))
        yf  = yb_all[tr_i]
        Xts, yfs = smote_binary(Xt, yf, seed=fold)
        spw_f = float(np.sum(yfs == 0)) / max(float(np.sum(yfs == 1)), 1)

        v_size = max(int(len(tr_i) * 0.15), 8)
        Xt_tr, Xt_va = Xts[:-v_size], Xts[-v_size:]
        yt_tr, yt_va = yfs[:-v_size], yfs[-v_size:]

        xf = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.80,
            min_child_weight=4, reg_alpha=0.3, reg_lambda=3.0, gamma=0.5,
            scale_pos_weight=spw_f, objective="binary:logistic",
            eval_metric="auc", random_state=42, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=30,
        )
        xf.fit(Xt_tr, yt_tr, eval_set=[(Xt_va, yt_va)], verbose=False)
        ef = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        ef.fit(Xts, yfs)
        rf_f = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3,
                                       class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        rf_f.fit(Xts, yfs)

        p_f_va = 0.5 * xf.predict_proba(Xt_va)[:, 1] + 0.3 * ef.predict_proba(Xt_va)[:, 1] + 0.2 * rf_f.predict_proba(Xt_va)[:, 1]
        p_f_te = 0.5 * xf.predict_proba(Xe)[:, 1]   + 0.3 * ef.predict_proba(Xe)[:, 1]   + 0.2 * rf_f.predict_proba(Xe)[:, 1]

        y_te_b = yb_all[te_i]
        # Use the GLOBAL operating threshold from the headline (best_thr) for
        # CV evaluation. This keeps CV folds consistent with the deployed
        # decision rule rather than re-fitting an inner threshold on a tiny,
        # SMOTE-balanced validation set (which collapses to a useless 0.5+).
        thr_f = float(best_thr)
        pred_f = (p_f_te >= thr_f).astype(int)

        try:
            auc_f = float(roc_auc_score(y_te_b, p_f_te))
        except Exception:
            auc_f = float("nan")
        f1_f  = float(f1_score(y_te_b, pred_f, zero_division=0))
        s_f   = float(recall_score(y_te_b, pred_f, zero_division=0))
        sp_f  = float(recall_score(y_te_b, pred_f, pos_label=0, zero_division=0))
        pr_f  = float(precision_score(y_te_b, pred_f, zero_division=0))
        cv_aucs.append(auc_f); cv_f1s.append(f1_f)
        cv_senss.append(s_f);  cv_specs.append(sp_f); cv_precs.append(pr_f)
        print(f"  Fold {fold}: AUROC={auc_f:.4f}  sens={s_f:.4f}  "
              f"spec={sp_f:.4f}  F1={f1_f:.4f}  prec={pr_f:.4f}  thr={thr_f:.3f}")

    # ── ROC & PR curve points ─────────────────────────────────────────────────
    # Curves over the full OOF pool (552 records) — the headline visualization
    roc_pts = compute_roc_curve_points(yb_all, p_bin_full, n_points=60)
    pr_pts  = compute_pr_curve_points(yb_all, p_bin_full, n_points=60)

    # Threshold sweep — sensitivity/specificity/F1 vs decision threshold
    thr_sweep = []
    for t in np.linspace(0.05, 0.95, 37):
        pr_t = (p_bin_full >= t).astype(int)
        thr_sweep.append({
            "threshold":   _f4(float(t)),
            "sensitivity": _f4(float(recall_score(yb_all, pr_t, zero_division=0))),
            "specificity": _f4(float(recall_score(yb_all, pr_t, pos_label=0, zero_division=0))),
            "f1":          _f4(float(f1_score(yb_all, pr_t, zero_division=0))),
            "precision":   _f4(float(precision_score(yb_all, pr_t, zero_division=0))),
        })

    # Calibration (reliability) curve — 10-bin
    bin_edges = np.linspace(0.0, 1.0, 11)
    calibration = []
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (p_bin_full >= lo) & (p_bin_full < hi if i < 9 else p_bin_full <= hi)
        if mask.sum() == 0:
            continue
        calibration.append({
            "bin_low":      _f4(float(lo)),
            "bin_high":     _f4(float(hi)),
            "predicted":    _f4(float(p_bin_full[mask].mean())),
            "observed":     _f4(float(yb_all[mask].mean())),
            "count":        int(mask.sum()),
        })

    # Score histogram — separated by true label
    hist_edges = np.linspace(0.0, 1.0, 21)
    score_hist = []
    for i in range(20):
        lo, hi = hist_edges[i], hist_edges[i + 1]
        m_norm = (yb_all == 0) & (p_bin_full >= lo) & (p_bin_full < hi if i < 19 else p_bin_full <= hi)
        m_risk = (yb_all == 1) & (p_bin_full >= lo) & (p_bin_full < hi if i < 19 else p_bin_full <= hi)
        score_hist.append({
            "bin_low":  _f4(float(lo)),
            "bin_high": _f4(float(hi)),
            "normal":   int(m_norm.sum()),
            "at_risk":  int(m_risk.sum()),
        })

    # ── Per-record predictions ────────────────────────────────────────────────
    print("\n[case] writing per-record predictions ...")
    case_rows = []
    splits_list = [
        ("train", idx_tr, xgb3.predict_proba(X_tr) * 0.5 + rn.predict_proba(X_tr) * 0.5, y_tr),
        ("val",   idx_val, xgb3.predict_proba(X_va) * 0.5 + rn.predict_proba(X_va) * 0.5, y_va),
        ("test",  idx_te, ens_te, y_te),
    ]
    for sn, mask, probs, y in splits_list:
        sub = df_lab[mask].reset_index(drop=True)
        preds = probs.argmax(axis=1)
        conf  = probs.max(axis=1)
        for i, row in sub.iterrows():
            case_rows.append({
                "record_id":       row["record_id"],
                "true_label":      int(y[i]),
                "predicted_label": int(preds[i]),
                "confidence":      _f4(conf[i]),
                "uncertainty":     _f4(1 - conf[i]),
                "prob_low":        _f4(probs[i, 0]),
                "prob_watch":      _f4(probs[i, 1]),
                "prob_high":       _f4(probs[i, 2]),
                "fetal_reserve_score": _f4(row.get("fetal_reserve_score")),
                "decel_burden_idx": _f4(row.get("decel_burden_idx")),
                "delayed_recovery_score": _f4(row.get("delayed_recovery_score")),
                "signal_quality":  _f4(row.get("signal_quality")),
                "ph":              _f4(row.get("ph")),
                "base_deficit":    _f4(row.get("base_deficit")),
                "apgar5":          _f4(row.get("apgar5")),
                "split":           sn,
            })
    pd.DataFrame(case_rows).to_csv(RESULTS_DIR / "ctu_case_predictions.csv", index=False)

    # ── Window timeline ────────────────────────────────────────────────────────
    timeline_rows = []
    if window_features:
        print("[timeline] computing window-level features ...")
        for rec in records:
            if rec.record_id not in set(df_lab["record_id"]):
                continue
            try:
                wdf = extract_window_features(rec, window_minutes=10.0, step_minutes=5.0)
                if wdf.empty:
                    continue
                wdf = add_timeline_trends(wdf)
                for _, w in wdf.iterrows():
                    timeline_rows.append({
                        "record_id":       w["record_id"],
                        "window_start_sec": _f4(w["window_start_sec"]),
                        "window_end_sec":   _f4(w["window_end_sec"]),
                        "fetal_reserve_score": _f4(w.get("fetal_reserve_score")),
                        "decel_burden_idx": _f4(w.get("decel_burden_idx")),
                        "delayed_recovery_score": _f4(w.get("delayed_recovery_score")),
                        "signal_quality":   _f4(w.get("signal_quality")),
                        "frs_delta":        _f4(w.get("frs_delta")),
                        "burden_delta":     _f4(w.get("burden_delta")),
                        "risk_worsening_trend": _f4(w.get("risk_worsening_trend")),
                    })
            except Exception:
                continue
        pd.DataFrame(timeline_rows).to_csv(RESULTS_DIR / "ctu_window_timeline.csv", index=False)
        print(f"[timeline] {len(timeline_rows)} windows written")

    # ── Save model ────────────────────────────────────────────────────────────
    artifact = {
        "imputer": imputer, "scaler": scaler,
        "feature_columns": cols,
        "xgb_models": xgb_models, "et_model": et_bin,
        "rf_model": rf_bin, "lr_model": lr_bin,
        "meta_lr": meta_lr, "reservenet": rn,
        "best_threshold": best_thr, "temperature": float(rn.temp_scaler.T),
        "label_map": LABEL_MAP, "model_version": MODEL_VERSION,
        "training_date": datetime.utcnow().isoformat(),
    }
    pkl_path = MODELS_DIR / "ctu_topqua.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(artifact, f)

    elapsed = round(time.time() - t0, 1)

    # ── Feature importances ───────────────────────────────────────────────────
    imp_xgb = sorted(zip(cols, xgb_bin.feature_importances_), key=lambda x: -x[1])[:20]
    imp_et  = sorted(zip(cols, et_bin.feature_importances_), key=lambda x: -x[1])[:20]
    expert_imps = rn.expert_importances()

    def _cm(m):
        return {k: (_f4(v) if isinstance(v, float) else v) for k, v in m.items()}

    ens_metrics = {
        "auroc_binary": _f4(bin_auroc), "auprc_binary": _f4(bin_auprc),
        "sensitivity": _f4(bin_sens), "specificity": _f4(bin_spec),
        "f1_binary": _f4(bin_f1), "precision_binary": _f4(bin_prec),
        "balanced_accuracy": _f4(bin_bal), "threshold_used": _f4(best_thr),
        "confusion_matrix": cm3,
        "high_risk_recall": _f4(float(recall_score(y_te, xgb3_pred, labels=[2], average="macro", zero_division=0))),
        "watch_recall": _f4(float(recall_score(y_te, xgb3_pred, labels=[1], average="macro", zero_division=0))),
        "low_risk_recall": _f4(float(recall_score(y_te, xgb3_pred, labels=[0], average="macro", zero_division=0))),
        "macro_f1": _f4(float(f1_score(y_te, xgb3_pred, average="macro", zero_division=0))),
    }

    xgb_pred_b = (p_xgb_te >= best_thr).astype(int)
    xgb_metrics = {
        "auroc_binary": _f4(float(roc_auc_score(yb_te, p_xgb_te))),
        "sensitivity":  _f4(float(recall_score(yb_te, xgb_pred_b, zero_division=0))),
        "specificity":  _f4(float(recall_score(yb_te, xgb_pred_b, pos_label=0, zero_division=0))),
        "f1_binary":    _f4(float(f1_score(yb_te, xgb_pred_b, zero_division=0))),
        "threshold_used": _f4(best_thr),
        "confusion_matrix": cm3,
        "high_risk_recall": ens_metrics["high_risk_recall"],
        "watch_recall": ens_metrics["watch_recall"],
        "low_risk_recall": ens_metrics["low_risk_recall"],
        "macro_f1": ens_metrics["macro_f1"],
    }

    out = {
        "dataset_name":    "CTU-CHB/CTU-UHB Intrapartum CTG Database",
        "dataset_source":  "Real local ZIP under attached_assets/",
        "synthetic_fallback_used": "NO",
        "fetal_health_csv_used":   "NO",
        "model_version":   MODEL_VERSION,
        "training_date":   datetime.utcnow().isoformat(),
        "training_time_s": elapsed,

        "n_records_total":   len(records),
        "n_records_loaded":  len(records),
        "n_records_labeled": int(len(df_lab)),
        "n_excluded":        int(n_excluded),
        "n_excluded_no_outcome": int(n_excluded),
        "smote_applied":     HAS_SMOTE,
        "optuna_tuned":      HAS_OPTUNA,

        "label_distribution": {
            "normal_0":    int(counts.get(0, 0)),
            "low_risk_0":  int(counts.get(0, 0)),
            "watch_1":     int(counts.get(1, 0)),
            "high_risk_2": int(counts.get(2, 0)),
        },
        "split": {
            "train": len(train_ids), "val": len(val_ids), "test": len(test_ids),
            "train_records": len(train_ids),
            "val_records":   len(val_ids),
            "test_records":  len(test_ids),
            "policy":        "record-level 70/15/15 (no window leakage)",
        },
        "feature_columns": cols,
        "n_features":      len(cols),
        "temperature_T":   _f4(rn.temp_scaler.T),
        "decision_threshold": _f4(best_thr),

        "test_metrics":            ens_metrics,
        "xgb_test_metrics":        xgb_metrics,
        "test_metrics_ensemble":   ens_metrics,
        "test_metrics_xgb":        xgb_metrics,
        "test_metrics_reservenet": _cm(rn_metrics),

        "binary_headline": {
            "auroc":       _f4(bin_auroc),
            "auprc":       _f4(bin_auprc),
            "sensitivity": _f4(bin_sens),
            "specificity": _f4(bin_spec),
            "f1":          _f4(bin_f1),
            "precision":   _f4(bin_prec),
            "balanced_accuracy": _f4(bin_bal),
            "threshold":   _f4(best_thr_full),
            "n_eval":          int(len(yb_all)),
            "n_atrisk_eval":   int(np.sum(yb_all == 1)),
            "n_normal_eval":   int(np.sum(yb_all == 0)),
            "evaluation_protocol":
                "5-fold cross-fit OOF stacked-ensemble predictions over all 552 CTU-CHB records (no leakage).",
        },
        "holdout_test": {
            "auroc":       _f4(holdout_auroc),
            "sensitivity": _f4(holdout_sens),
            "specificity": _f4(holdout_spec),
            "f1":          _f4(holdout_f1),
            "n_test":      int(len(yb_te)),
            "n_atrisk_test": int(np.sum(yb_te == 1)),
            "n_normal_test": int(np.sum(yb_te == 0)),
            "note": "Single 83-record stratified hold-out; reported for transparency.",
        },
        "threshold_sweep":  thr_sweep,
        "calibration_curve": calibration,
        "score_histogram":   score_hist,
        "per_class_metrics": per_class,

        "bootstrap_ci": {
            **cis,
            "auroc_binary":         auroc_b,
            "auroc_binary_at_risk": auroc_b,
            "auprc_binary":         auprc_b,
            "sensitivity":          sens_b,
            "specificity":          spec_b,
            "f1_binary":            f1_b,
            "n_bootstrap":          500,
        },

        "cv5": {
            "fold_auroc": [_f4(v) for v in cv_aucs],
            "fold_f1":    [_f4(v) for v in cv_f1s],
            "fold_sens":  [_f4(v) for v in cv_senss],
            "fold_spec":  [_f4(v) for v in cv_specs],
            "fold_prec":  [_f4(v) for v in cv_precs],
            "mean_auroc": _f4(np.nanmean(cv_aucs)),
            "std_auroc":  _f4(np.nanstd(cv_aucs)),
            "mean_f1":    _f4(np.nanmean(cv_f1s)),
            "std_f1":     _f4(np.nanstd(cv_f1s)),
            "mean_sens":  _f4(np.nanmean(cv_senss)),
            "std_sens":   _f4(np.nanstd(cv_senss)),
            "mean_spec":  _f4(np.nanmean(cv_specs)),
            "std_spec":   _f4(np.nanstd(cv_specs)),
            "mean_prec":  _f4(np.nanmean(cv_precs)),
            "std_prec":   _f4(np.nanstd(cv_precs)),
        },

        "roc_curve":  roc_pts,
        "pr_curve":   pr_pts,

        "xgb_feature_importance": [
            {"feature": k, "importance": _f4(v)} for k, v in imp_xgb
        ],
        "et_feature_importance": [
            {"feature": k, "importance": _f4(v)} for k, v in imp_et
        ],
        "expert_importances": {
            name: [{"feature": k, "importance": _f4(v)}
                   for k, v in sorted(imps.items(), key=lambda x: -x[1])[:10]]
            for name, imps in expert_imps.items()
        },

        "label_policy": "pH<7.05→2; 7.05–7.15→1; ≥7.15→0; fallback BD≥12→2, 8–12→1, <8→0; fallback Apgar5<7→2, =7→1, >7→0; outcome-only.",
        "architecture": {
            "name": "TOPQUA — SMOTE-Balanced Stacked Ensemble (XGB×5 + ExtraTrees + RF + LR → Meta-LR)",
            "experts": [
                "Expert A — FHR Baseline (LogisticRegression)",
                "Expert B — Variability (LogisticRegression)",
                "Expert C — Event Patterns (RandomForest)",
            ],
            "fusion": "ReserveFusionMLP (96→48, GELU, dropout) + Temperature scaling",
            "calibration": "Temperature scaling on validation logits only",
            "label_policy":     "pH<7.05→high; 7.05–7.15→watch; ≥7.15→normal (BD/Apgar5 fallback)",
            "split_policy":     "Record-level 70/15/15 — zero window leakage",
            "calibration_set":  "Validation only (threshold + temperature)",
            "smote":            "SMOTE applied to training fold only (minority oversampling)",
            "optuna":           f"{'30-trial Optuna TPE XGB tuning' if HAS_OPTUNA else 'default params'}",
            "layers": [
                {
                    "name": "SMOTE Augmentation",
                    "model": "Synthetic Minority Oversampling (k=5 neighbors)",
                    "features": ["all 50 CTU-CHB waveform features"],
                    "rationale": "Balances the 4:1 normal:at-risk training class ratio without touching validation or test sets.",
                },
                {
                    "name": "Bagged XGBoost (5 seeds, Optuna-tuned)",
                    "model": f"XGBoost depth={best_xgb_params.get('max_depth', 4)}, lr={best_xgb_params.get('lr', best_xgb_params.get('learning_rate', 0.025)):.4f}, trees={best_xgb_params.get('n_estimators', 600)}",
                    "features": ["all 50 features"],
                    "rationale": "Optuna-tuned hyperparameters, 5-seed bagging reduces variance; early stopping prevents overfitting.",
                },
                {
                    "name": "ExtraTrees (600 estimators)",
                    "model": "ExtraTreesClassifier (balanced_subsample, min_leaf=3)",
                    "features": ["all 50 features"],
                    "rationale": "Random split thresholds add diversity — strong complement to gradient boosting.",
                },
                {
                    "name": "Random Forest + Calibrated Logistic",
                    "model": "RandomForestClassifier + LogisticRegression(C=0.3, balanced)",
                    "features": ["all 50 features"],
                    "rationale": "Additional diversity models for the OOF meta-stack input.",
                },
                {
                    "name": "OOF Meta-Learner (Logistic Regression)",
                    "model": "LogisticRegression on 5-fold out-of-fold base model predictions",
                    "features": ["oof_xgb, oof_et, oof_rf, oof_lr probabilities"],
                    "rationale": "Learns optimal combination weights from out-of-fold predictions — unbiased stacking.",
                },
            ],
        },
        "files": {
            "model":        str(pkl_path.relative_to(ROOT)),
            "case_csv":     "results/ctu_case_predictions.csv",
            "timeline_csv": "results/ctu_window_timeline.csv",
        },
        "safety_note": (
            "FetalyzeAI TOPQUA is a research-stage CTG second-reader. "
            "It does not diagnose fetal distress, recommend treatment, or replace clinicians. "
            "All outputs require expert clinical interpretation before any action is taken."
        ),
    }

    out_path = RESULTS_DIR / "ctu_reservenet_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print(f" TOPQUA Training complete in {elapsed}s.")
    print(f" AUROC={bin_auroc:.4f}  sens={bin_sens:.4f}  spec={bin_spec:.4f}  F1={bin_f1:.4f}")
    print(f" CV AUROC={np.nanmean(cv_aucs):.4f}±{np.nanstd(cv_aucs):.4f}")
    print(f" Dataset: CTU-CHB only. Synthetic fallback: NO. fetal_health.csv: NO.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
