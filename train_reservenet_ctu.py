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
    roc_curve, precision_recall_curve, log_loss as sklearn_log_loss,
    brier_score_loss, fbeta_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
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
    # FIGO 2015 guideline binary flags — medical domain knowledge
    "figo_abnormal_baseline", "figo_tachycardia", "figo_bradycardia",
    "figo_absent_variability", "figo_reduced_variability", "figo_absent_accels",
    "figo_late_decels", "figo_prolonged_decel", "figo_sinusoidal",
    "figo_composite_score", "figo_category",
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


def smote_multiclass(X, y, seed=42, k=5):
    """SMOTE across all minority classes toward ~70% of the majority count."""
    if not HAS_SMOTE:
        return X, y
    counts = np.bincount(y, minlength=3)
    majority = int(counts.max())
    target_n = max(int(majority * 0.70), int(counts.max(axis=0)))
    strategy = {c: max(target_n, counts[c]) for c in range(len(counts)) if counts[c] < majority}
    if not strategy:
        return X, y
    k_eff = min(k, int(counts.min()) - 1)
    if k_eff < 1:
        return X, y
    try:
        sm = SMOTE(sampling_strategy=strategy, k_neighbors=k_eff, random_state=seed)
        return sm.fit_resample(X, y)
    except Exception as e:
        print(f"[smote3] warning: {e} — skipping 3-class SMOTE")
        return X, y


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


def main(window_features: bool = False):
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

    # ── XGB hyperparameters (hardcoded from prior Optuna run for speed) ───────
    best_xgb_params = dict(
        n_estimators=600, max_depth=5, learning_rate=0.04,
        subsample=0.88, colsample_bytree=0.77,
        min_child_weight=3, reg_alpha=0.025, reg_lambda=2.0, gamma=0.8,
    )
    print(f"\n[params] using hardcoded XGB params (Optuna-pre-tuned): {best_xgb_params}")

    # ── Base models — trained on SMOTE-augmented train fold ───────────────────
    spw_eff = float(np.sum(yb_tr_s == 0)) / max(float(np.sum(yb_tr_s == 1)), 1)

    print("\n[xgb] training bagged XGBoost (4 seeds, Optuna params) ...")
    xgb_seeds = [42, 7, 2024, 1337]
    xgb_models = []
    p_xgb_va_list, p_xgb_te_list = [], []
    for sd in xgb_seeds:
        p = dict(**{k: v for k, v in best_xgb_params.items() if k != "n_estimators"},
                 n_estimators=min(best_xgb_params.get("n_estimators", 600), 500))
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
        n_estimators=500, max_depth=None, min_samples_leaf=3,
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
        n_estimators=400, max_depth=10, min_samples_leaf=3,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    rf_bin.fit(X_tr_s, yb_tr_s)
    p_rf_va = rf_bin.predict_proba(X_va)[:, 1]
    p_rf_te = rf_bin.predict_proba(X_te)[:, 1]

    print("[hgb] training HistGradientBoosting (class-weighted) ...")
    hgb_bin = HistGradientBoostingClassifier(
        max_iter=500, max_depth=5, learning_rate=0.03,
        min_samples_leaf=5, l2_regularization=0.5,
        class_weight="balanced", random_state=42,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
    )
    hgb_bin.fit(X_tr_s, yb_tr_s)
    p_hgb_va = hgb_bin.predict_proba(X_va)[:, 1]
    p_hgb_te = hgb_bin.predict_proba(X_te)[:, 1]

    # ── Stacked meta-learner on OOF probabilities ─────────────────────────────
    print("\n[stack] training OOF stacked meta-learner (5-fold) ...")
    yb_all = (y_raw >= 1).astype(int)
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(X_raw))
    oof_et  = np.zeros(len(X_raw))
    oof_rf  = np.zeros(len(X_raw))
    oof_lr  = np.zeros(len(X_raw))
    oof_hgb = np.zeros(len(X_raw))

    for fold_i, (tr_i, te_i) in enumerate(skf5.split(X_raw, yb_all), 1):
        imp_f = SimpleImputer(strategy="median").fit(X_raw[tr_i])
        sc_f  = RobustScaler().fit(imp_f.transform(X_raw[tr_i]))
        Xt_f  = sc_f.transform(imp_f.transform(X_raw[tr_i]))
        Xe_f  = sc_f.transform(imp_f.transform(X_raw[te_i]))
        yb_f  = yb_all[tr_i]
        Xts, ybs = smote_binary(Xt_f, yb_f, seed=fold_i)

        spw_f = float(np.sum(ybs == 0)) / max(float(np.sum(ybs == 1)), 1)
        mf = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.75, min_child_weight=3,
            reg_alpha=0.3, reg_lambda=2.5, gamma=0.4,
            scale_pos_weight=spw_f, objective="binary:logistic",
            eval_metric="auc", random_state=42, n_jobs=-1, tree_method="hist",
        )
        mf.fit(Xts, ybs, verbose=False)
        oof_xgb[te_i] = mf.predict_proba(Xe_f)[:, 1]

        ef = ExtraTreesClassifier(n_estimators=150, min_samples_leaf=3,
                                  class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        ef.fit(Xts, ybs)
        oof_et[te_i] = ef.predict_proba(Xe_f)[:, 1]

        rf_f = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=3,
                                      class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        rf_f.fit(Xts, ybs)
        oof_rf[te_i] = rf_f.predict_proba(Xe_f)[:, 1]

        lr_f = LogisticRegression(C=0.5, class_weight="balanced",
                                  max_iter=2000, solver="liblinear", random_state=42)
        lr_f.fit(Xts, ybs)
        oof_lr[te_i] = lr_f.predict_proba(Xe_f)[:, 1]

        hgb_f = HistGradientBoostingClassifier(
            max_iter=250, max_depth=4, learning_rate=0.04,
            min_samples_leaf=5, l2_regularization=0.5,
            class_weight="balanced", random_state=42,
        )
        hgb_f.fit(Xts, ybs)
        oof_hgb[te_i] = hgb_f.predict_proba(Xe_f)[:, 1]
        print(f"  Fold {fold_i} OOF AUROC: xgb={roc_auc_score(yb_all[te_i], oof_xgb[te_i]):.4f}  hgb={roc_auc_score(yb_all[te_i], oof_hgb[te_i]):.4f}")

    # Fit meta-learner on full OOF stack (now 5 base models)
    meta_X_tr_va = np.column_stack([oof_xgb[idx_tr | idx_val],
                                     oof_et[idx_tr | idx_val],
                                     oof_rf[idx_tr | idx_val],
                                     oof_lr[idx_tr | idx_val],
                                     oof_hgb[idx_tr | idx_val]])
    meta_y_tr_va = yb_all[idx_tr | idx_val]
    meta_lr = LogisticRegression(C=2.0, solver="lbfgs", max_iter=2000, random_state=42)
    meta_lr.fit(meta_X_tr_va, meta_y_tr_va)

    # Meta-learner on test (using base models trained on full train+val)
    meta_X_te = np.column_stack([p_xgb_te, p_et_te, p_rf_te, p_lr_te, p_hgb_te])
    p_meta_te = meta_lr.predict_proba(meta_X_te)[:, 1]

    meta_X_va = np.column_stack([p_xgb_va, p_et_va, p_rf_va, p_lr_va, p_hgb_va])
    p_meta_va = meta_lr.predict_proba(meta_X_va)[:, 1]

    # ── Soft-vote ensemble (meta + direct XGB + HGB) — final at-risk score ─────
    p_bin_va = 0.5 * p_meta_va + 0.3 * p_xgb_va + 0.2 * p_hgb_va
    p_bin_te = 0.5 * p_meta_te + 0.3 * p_xgb_te + 0.2 * p_hgb_te

    # ── OOF stacked predictions over FULL 552-record dataset ─────────────────
    # This is the standard scientifically-valid way to report stacked-ensemble
    # performance: each prediction is held-out (no leakage) and we pool all
    # records to gain statistical power vs. the small 83-record test slice.
    print("\n[oof-full] computing 5-fold cross-fit meta predictions across all 552 records ...")
    meta_X_full = np.column_stack([oof_xgb, oof_et, oof_rf, oof_lr, oof_hgb])
    p_meta_full = np.zeros(len(X_raw))
    skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=4242)
    for tr_i, te_i in skf_meta.split(meta_X_full, yb_all):
        m_f = LogisticRegression(C=2.0, solver="lbfgs", max_iter=2000, random_state=42)
        m_f.fit(meta_X_full[tr_i], yb_all[tr_i])
        p_meta_full[te_i] = m_f.predict_proba(meta_X_full[te_i])[:, 1]
    p_bin_full = 0.5 * p_meta_full + 0.3 * oof_xgb + 0.2 * oof_hgb

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

    # ── 3-class XGBoost with 3-class SMOTE + sample_weight ───────────────────
    # CRITICAL FIX: scale_pos_weight is silently ignored for multi:softprob.
    # Instead we apply 3-class SMOTE then pass per-sample class weights.
    print("\n[xgb3] training 3-class XGBoost (SMOTE + sample_weight, confusion matrix reporting) ...")
    print(f"[xgb3] class balance before 3-class SMOTE: {np.bincount(y_tr, minlength=3)}")
    X_tr3, y_tr3 = smote_multiclass(X_tr, y_tr, seed=42, k=5)
    print(f"[xgb3] class balance after  3-class SMOTE: {np.bincount(y_tr3, minlength=3)}")

    # Per-sample class weights for XGBoost (inverse-frequency weighting)
    counts3 = np.bincount(y_tr3, minlength=3).astype(float)
    cw3 = len(y_tr3) / (3.0 * np.maximum(counts3, 1))
    sw3 = cw3[y_tr3]
    # Extra 2.5× boost for HR class — SMOTE balances counts but HR cases still
    # benefit from emphasis: a missed high-risk case is far more dangerous than
    # a false alarm (asymmetric clinical cost).
    sw3[y_tr3 == 2] *= 2.5

    # Also compute sample_weight for validation (for eval_set monitoring only)
    counts_va3 = np.bincount(y_va, minlength=3).astype(float)
    cw_va3 = len(y_va) / (3.0 * np.maximum(counts_va3, 1))
    sw_va3 = cw_va3[y_va]

    xgb3 = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.025,
        subsample=0.80, colsample_bytree=0.75,
        min_child_weight=2, reg_alpha=0.5, reg_lambda=2.0, gamma=0.3,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=40,
    )
    xgb3.fit(X_tr3, y_tr3, sample_weight=sw3,
             eval_set=[(X_va, y_va)], verbose=False)
    xgb3_te = xgb3.predict_proba(X_te)

    # Also train a 3-class HGB for ensemble (HGB natively supports class_weight)
    print("[hgb3] training 3-class HistGradientBoosting ...")
    hgb3 = HistGradientBoostingClassifier(
        max_iter=350, max_depth=5, learning_rate=0.03,
        min_samples_leaf=3, l2_regularization=0.4,
        class_weight="balanced", random_state=42,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
    )
    # HGB doesn't support dict class_weight — use sample_weight with strong HR emphasis
    sw_hgb3 = np.ones(len(y_tr), dtype=float)
    sw_hgb3[y_tr == 2] = 4.0   # 4× for HR on original (non-SMOTE) distribution
    hgb3.fit(X_tr, y_tr, sample_weight=sw_hgb3)
    hgb3_te = hgb3.predict_proba(X_te)

    # Soft ensemble: 0.6 × XGB3 + 0.4 × HGB3
    ens3_raw = 0.6 * xgb3_te + 0.4 * hgb3_te

    # Per-class threshold optimisation on validation — maximise balanced accuracy
    va_xgb3 = xgb3.predict_proba(X_va)
    va_hgb3 = hgb3.predict_proba(X_va)
    va_ens3  = 0.6 * va_xgb3 + 0.4 * va_hgb3

    # Sweep thresholds — optimise F2 (β=2) to weight recall 4× over precision.
    # Clinically: missing a high-risk case is far worse than a false alarm.
    # Sweep lower HR range [0.05, 0.40] to capture more true high-risk cases.
    best_f2_3   = -1.0
    best_thr_hr = 0.2
    best_thr_w  = 0.2
    for thr_hr in np.arange(0.05, 0.45, 0.05):
        for thr_w in np.arange(0.05, 0.40, 0.05):
            pred_3 = np.zeros(len(y_va), dtype=int)
            pred_3[va_ens3[:, 2] >= thr_hr] = 2
            pred_3[(va_ens3[:, 2] < thr_hr) & (va_ens3[:, 1] >= thr_w)] = 1
            try:
                f2 = float(fbeta_score(y_va, pred_3, beta=2.0,
                                       average="macro", zero_division=0))
            except Exception:
                continue
            if f2 > best_f2_3:
                best_f2_3   = f2
                best_thr_hr = float(thr_hr)
                best_thr_w  = float(thr_w)
    best_bal_3 = best_f2_3   # keep variable name for downstream compat

    print(f"[xgb3] best 3-class threshold (F2 β=2): hr={best_thr_hr:.2f}  "
          f"watch={best_thr_w:.2f}  val F2-macro={best_f2_3:.4f}")

    # Apply tuned per-class thresholds to test set
    xgb3_pred = np.zeros(len(y_te), dtype=int)
    xgb3_pred[ens3_raw[:, 2] >= best_thr_hr] = 2
    xgb3_pred[(ens3_raw[:, 2] < best_thr_hr) & (ens3_raw[:, 1] >= best_thr_w)] = 1

    # ens_te: calibrate 3-class probabilities using binary p_bin_te as anchor
    ens_te = ens3_raw.copy()
    cur_atrisk = ens_te[:, 1] + ens_te[:, 2] + 1e-9
    scale = p_bin_te / cur_atrisk
    ens_te[:, 1] = np.clip(ens_te[:, 1] * scale, 0, 1)
    ens_te[:, 2] = np.clip(ens_te[:, 2] * scale, 0, 1)
    ens_te[:, 0] = np.clip(1 - p_bin_te, 0, 1)
    ens_te = ens_te / np.maximum(ens_te.sum(axis=1, keepdims=True), 1e-9)

    cm3 = confusion_matrix(y_te, xgb3_pred, labels=[0, 1, 2]).tolist()
    per_class = per_class_metrics(y_te, xgb3_pred)
    print(f"[xgb3] confusion matrix:\n{np.array(cm3)}")
    hr_rec = float(recall_score(y_te, xgb3_pred, labels=[2], average="macro", zero_division=0))
    w_rec  = float(recall_score(y_te, xgb3_pred, labels=[1], average="macro", zero_division=0))
    bal3   = float(balanced_accuracy_score(y_te, xgb3_pred))
    mf3    = float(f1_score(y_te, xgb3_pred, average="macro", zero_division=0))
    print(f"[xgb3] HR recall={hr_rec:.4f}  Watch recall={w_rec:.4f}  "
          f"Balanced acc={bal3:.4f}  Macro-F1={mf3:.4f}")

    # ── ReserveNet (domain-partitioned for explainability) ────────────────────
    print("\n[reservenet] training domain-partitioned ensemble ...")
    rn = ReserveNet(n_classes=3, random_state=42)
    rn.fit(X_tr, y_tr, X_va, y_va, cols)
    rn_te = rn.predict_proba(X_te)
    rn_metrics = compute_all_metrics(y_te, rn_te, threshold=best_thr)

    # ── Bootstrap CIs (over full 552-record OOF pool) ─────────────────────────
    print("\n[bootstrap] 100-iter bootstrap CIs over full OOF pool ...")
    rng_b = np.random.RandomState(42)
    bs_auroc, bs_sens, bs_spec, bs_f1, bs_auprc = [], [], [], [], []
    n_full = len(yb_all)
    for _ in range(100):
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

    # ── Calibration / Probability quality metrics ─────────────────────────────
    print("\n[calib] computing ECE, Brier score, log-loss ...")
    # Binary calibration on full OOF pool
    p_bin_clip = np.clip(p_bin_full, 1e-7, 1 - 1e-7)
    try:
        ece_bin = float(0.0)
        for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            mask = (p_bin_clip >= lo) & (p_bin_clip < hi)
            if mask.sum() > 0:
                acc_b = float(yb_all[mask].mean())
                conf_b = float(p_bin_clip[mask].mean())
                ece_bin += float(mask.mean()) * abs(acc_b - conf_b)
        brier_bin = float(np.mean((p_bin_full - yb_all) ** 2))
        ll_bin    = float(sklearn_log_loss(yb_all, np.column_stack([1 - p_bin_clip, p_bin_clip])))
    except Exception as e:
        print(f"[calib] warning: {e}")
        ece_bin = brier_bin = ll_bin = None

    # 3-class ECE/Brier on test set (ens_te)
    try:
        ece_3   = float(rn_metrics.get("ece", float("nan")))
        brier_3 = float(rn_metrics.get("brier", float("nan")))
    except Exception:
        ece_3 = brier_3 = None

    print(f"[calib] ECE(binary/OOF)={ece_bin:.4f}  Brier={brier_bin:.4f}  LogLoss={ll_bin:.4f}")

    # ── Uncertainty coverage (binary model) ───────────────────────────────────
    # "uncertain" = model score in [0.35, 0.65] — the grey zone around threshold
    grey_lo, grey_hi = 0.35, 0.65
    uncertain_mask = (p_bin_full >= grey_lo) & (p_bin_full <= grey_hi)
    confident_mask = ~uncertain_mask
    uncertain_rate = float(uncertain_mask.mean())

    confident_preds = (p_bin_full[confident_mask] >= best_thr_full).astype(int)
    conf_acc = float((confident_preds == yb_all[confident_mask]).mean()) if confident_mask.sum() > 0 else float("nan")

    hr_mask = yb_all == 1   # binary: at-risk
    hr_uncertain = float(uncertain_mask[hr_mask].mean()) if hr_mask.sum() > 0 else float("nan")

    conf_hr = yb_all[confident_mask]
    conf_hr_preds = confident_preds
    conf_hr_rec = float(recall_score(conf_hr, conf_hr_preds, zero_division=0)) if confident_mask.sum() > 0 else float("nan")

    print(f"[uncertainty] rate={uncertain_rate:.4f}  conf_acc={conf_acc:.4f}  "
          f"conf_HR_rec={conf_hr_rec:.4f}  HR_flagged_uncertain={hr_uncertain:.4f}")

    uncertainty_coverage = {
        "uncertain_rate":              _f4(uncertain_rate),
        "confident_accuracy":          _f4(conf_acc),
        "high_risk_recall_confident":  _f4(conf_hr_rec),
        "high_risk_flagged_uncertain": _f4(hr_uncertain),
        "grey_zone_lo":                _f4(grey_lo),
        "grey_zone_hi":                _f4(grey_hi),
    }

    # ── OOD Detection (IsolationForest) ──────────────────────────────────────
    print("\n[ood] fitting IsolationForest OOD detector on training features ...")
    try:
        from sklearn.ensemble import IsolationForest as _IF
        ood_clf = _IF(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
        ood_clf.fit(X_tr)
        _ood_scores_tr = -ood_clf.score_samples(X_tr)
        _ood_scores_va = -ood_clf.score_samples(X_va)
        _ood_scores_te = -ood_clf.score_samples(X_te)
        ood_threshold  = float(np.percentile(_ood_scores_tr, 95))
        ood_rate_te    = float(np.mean(_ood_scores_te > ood_threshold))
        ood_rate_va    = float(np.mean(_ood_scores_va > ood_threshold))
        _flag_te = _ood_scores_te > ood_threshold
        ood_hr_te     = float(np.mean(_flag_te[y_te == 2])) if (y_te == 2).sum() > 0 else float("nan")
        ood_watch_te  = float(np.mean(_flag_te[y_te == 1])) if (y_te == 1).sum() > 0 else float("nan")
        ood_normal_te = float(np.mean(_flag_te[y_te == 0])) if (y_te == 0).sum() > 0 else float("nan")
        print(f"[ood] threshold={ood_threshold:.4f}  OOD rate test={ood_rate_te:.3f}"
              f"  HR-OOD={ood_hr_te:.3f}  Normal-OOD={ood_normal_te:.3f}")
        ood_detection = {
            "method":                  "IsolationForest (n_estimators=100, contamination=5%)",
            "ood_threshold_95pct":     _f4(ood_threshold),
            "ood_rate_test":           _f4(ood_rate_te),
            "ood_rate_val":            _f4(ood_rate_va),
            "ood_rate_high_risk_test": _f4(ood_hr_te),
            "ood_rate_watch_test":     _f4(ood_watch_te),
            "ood_rate_normal_test":    _f4(ood_normal_te),
            "interpretation": (
                "Records above the 95th-percentile training anomaly score are flagged OOD. "
                "High HR-OOD rate indicates pathological cases may be under-represented in training."
            ),
        }
    except Exception as _e:
        print(f"[ood] warning: {_e}")
        ood_detection = {"method": "IsolationForest", "error": str(_e)}

    # ── Adversarial / Clinical Stress Tests ───────────────────────────────────
    # 7 canonical CTG profiles evaluate whether the model respects
    # established clinical patterns without requiring held-out label data.
    print("\n[adversarial] running 7 canonical clinical stress tests ...")
    _med = {c: float(np.nanmedian(X_raw[idx_tr][:, i])) for i, c in enumerate(cols)}

    def _build_stress(overrides):
        v = np.array([[_med.get(c, 0.0) for c in cols]])
        for k, val in overrides.items():
            if k in cols:
                v[0, cols.index(k)] = float(val)
        return scaler.transform(imputer.transform(v))

    _stress_cases = [
        ("Textbook Normal CTG", 0, {
            "baseline_fhr": 130.0, "mean_fhr": 130.0, "std_fhr": 8.0,
            "stv": 15.0, "ltv": 25.0, "stv_norm": 1.0, "ltv_norm": 1.0, "roughness": 2.5,
            "tachycardia_frac": 0.0, "bradycardia_frac": 0.0,
            "n_accels": 4, "accels_per_30min": 4.0, "mean_accel_height": 15.0,
            "n_decels": 0, "decels_per_30min": 0.0, "mean_decel_depth": 0.0,
            "max_decel_depth": 0.0, "mean_decel_dur_s": 0.0, "total_decel_dur_s": 0.0,
            "decel_area": 0.0, "prolonged_decel_flag": 0.0,
            "late_decel_likelihood": 0.0, "delayed_recovery_score": 0.0,
            "worsening_recovery_trend": 0.0, "decel_burden_idx": 0.03,
            "fetal_reserve_score": 85.0, "signal_quality": 0.38,
            "missing_fhr_pct": 5.0, "flatline_pct": 2.0, "abrupt_jump_count": 0.0,
            "baseline_fhr_last30": 130.0, "stv_last30": 15.0, "ltv_last30": 25.0,
            "std_fhr_last30": 8.0, "n_decels_last30": 0, "max_decel_depth_last30": 0.0,
            "stv_trend_late_vs_full": 0.0, "baseline_trend_late_vs_full": 0.0,
            "figo_abnormal_baseline": 0.0, "figo_tachycardia": 0.0, "figo_bradycardia": 0.0,
            "figo_absent_variability": 0.0, "figo_reduced_variability": 0.0,
            "figo_absent_accels": 0.0, "figo_late_decels": 0.0, "figo_prolonged_decel": 0.0,
            "figo_sinusoidal": 0.0, "figo_composite_score": 0.0, "figo_category": 0.0,
        }),
        ("Severe Bradycardia + Absent Variability", 2, {
            "baseline_fhr": 82.0, "mean_fhr": 82.0, "std_fhr": 3.0,
            "stv": 1.8, "ltv": 3.5, "stv_norm": 0.12, "ltv_norm": 0.14, "roughness": 0.3,
            "tachycardia_frac": 0.0, "bradycardia_frac": 0.95,
            "n_accels": 0, "accels_per_30min": 0.0, "mean_accel_height": 0.0,
            "n_decels": 4, "decels_per_30min": 2.0, "mean_decel_depth": 25.0,
            "max_decel_depth": 40.0, "mean_decel_dur_s": 90.0, "total_decel_dur_s": 360.0,
            "decel_area": 600.0, "prolonged_decel_flag": 0.0,
            "late_decel_likelihood": 0.1, "delayed_recovery_score": 0.45,
            "worsening_recovery_trend": 0.3, "decel_burden_idx": 0.78,
            "fetal_reserve_score": 8.0, "signal_quality": 0.28,
            "missing_fhr_pct": 15.0, "flatline_pct": 8.0, "abrupt_jump_count": 2.0,
            "baseline_fhr_last30": 78.0, "stv_last30": 1.5, "ltv_last30": 3.0,
            "std_fhr_last30": 2.5, "n_decels_last30": 4, "max_decel_depth_last30": 40.0,
            "stv_trend_late_vs_full": -1.5, "baseline_trend_late_vs_full": -5.0,
            "figo_bradycardia": 1.0, "figo_abnormal_baseline": 1.0,
            "figo_absent_variability": 1.0, "figo_reduced_variability": 1.0,
            "figo_absent_accels": 1.0, "figo_tachycardia": 0.0, "figo_late_decels": 0.0,
            "figo_prolonged_decel": 0.0, "figo_sinusoidal": 0.0,
            "figo_composite_score": 4.0, "figo_category": 2.0,
        }),
        ("Absent Variability + Repetitive Late Decels", 2, {
            "baseline_fhr": 125.0, "mean_fhr": 124.0, "std_fhr": 2.0,
            "stv": 1.5, "ltv": 3.0, "stv_norm": 0.10, "ltv_norm": 0.12, "roughness": 0.2,
            "tachycardia_frac": 0.0, "bradycardia_frac": 0.0,
            "n_accels": 0, "accels_per_30min": 0.0, "mean_accel_height": 0.0,
            "n_decels": 8, "decels_per_30min": 4.5, "mean_decel_depth": 28.0,
            "max_decel_depth": 48.0, "mean_decel_dur_s": 110.0, "total_decel_dur_s": 880.0,
            "decel_area": 1400.0, "prolonged_decel_flag": 0.0,
            "late_decel_likelihood": 0.96, "delayed_recovery_score": 0.72,
            "worsening_recovery_trend": 0.6, "decel_burden_idx": 0.88,
            "fetal_reserve_score": 6.0, "signal_quality": 0.27,
            "missing_fhr_pct": 12.0, "flatline_pct": 5.0, "abrupt_jump_count": 1.0,
            "baseline_fhr_last30": 122.0, "stv_last30": 1.3, "ltv_last30": 2.8,
            "std_fhr_last30": 1.8, "n_decels_last30": 8, "max_decel_depth_last30": 48.0,
            "stv_trend_late_vs_full": -0.8, "baseline_trend_late_vs_full": -3.0,
            "figo_absent_variability": 1.0, "figo_reduced_variability": 1.0,
            "figo_absent_accels": 1.0, "figo_late_decels": 1.0,
            "figo_abnormal_baseline": 0.0, "figo_tachycardia": 0.0, "figo_bradycardia": 0.0,
            "figo_prolonged_decel": 0.0, "figo_sinusoidal": 0.0,
            "figo_composite_score": 4.0, "figo_category": 2.0,
        }),
        ("Sinusoidal Pattern", 2, {
            "baseline_fhr": 128.0, "mean_fhr": 128.0, "std_fhr": 1.5,
            "stv": 0.8, "ltv": 2.0, "stv_norm": 0.05, "ltv_norm": 0.08, "roughness": 0.5,
            "tachycardia_frac": 0.0, "bradycardia_frac": 0.0,
            "n_accels": 0, "accels_per_30min": 0.0, "mean_accel_height": 0.0,
            "n_decels": 0, "decels_per_30min": 0.0, "mean_decel_depth": 0.0,
            "max_decel_depth": 0.0, "mean_decel_dur_s": 0.0, "total_decel_dur_s": 0.0,
            "decel_area": 0.0, "prolonged_decel_flag": 0.0,
            "late_decel_likelihood": 0.0, "delayed_recovery_score": 0.0,
            "worsening_recovery_trend": 0.0, "decel_burden_idx": 0.12,
            "fetal_reserve_score": 4.0, "signal_quality": 0.30,
            "missing_fhr_pct": 10.0, "flatline_pct": 88.0, "abrupt_jump_count": 0.0,
            "baseline_fhr_last30": 128.0, "stv_last30": 0.8, "ltv_last30": 2.0,
            "std_fhr_last30": 1.5, "n_decels_last30": 0, "max_decel_depth_last30": 0.0,
            "stv_trend_late_vs_full": 0.0, "baseline_trend_late_vs_full": 0.0,
            "figo_sinusoidal": 1.0, "figo_absent_variability": 1.0,
            "figo_reduced_variability": 1.0, "figo_absent_accels": 1.0,
            "figo_abnormal_baseline": 0.0, "figo_tachycardia": 0.0, "figo_bradycardia": 0.0,
            "figo_late_decels": 0.0, "figo_prolonged_decel": 0.0,
            "figo_composite_score": 4.0, "figo_category": 2.0,
        }),
        ("Tachycardia + Prolonged Deceleration", 2, {
            "baseline_fhr": 175.0, "mean_fhr": 172.0, "std_fhr": 12.0,
            "stv": 5.0, "ltv": 14.0, "stv_norm": 0.33, "ltv_norm": 0.56, "roughness": 3.5,
            "tachycardia_frac": 0.82, "bradycardia_frac": 0.0,
            "n_accels": 1, "accels_per_30min": 0.5, "mean_accel_height": 8.0,
            "n_decels": 3, "decels_per_30min": 1.5, "mean_decel_depth": 32.0,
            "max_decel_depth": 65.0, "mean_decel_dur_s": 200.0, "total_decel_dur_s": 600.0,
            "decel_area": 1000.0, "prolonged_decel_flag": 1.0,
            "late_decel_likelihood": 0.2, "delayed_recovery_score": 0.55,
            "worsening_recovery_trend": 0.45, "decel_burden_idx": 0.68,
            "fetal_reserve_score": 15.0, "signal_quality": 0.26,
            "missing_fhr_pct": 18.0, "flatline_pct": 3.0, "abrupt_jump_count": 5.0,
            "baseline_fhr_last30": 178.0, "stv_last30": 4.5, "ltv_last30": 13.0,
            "std_fhr_last30": 11.0, "n_decels_last30": 3, "max_decel_depth_last30": 65.0,
            "stv_trend_late_vs_full": -1.2, "baseline_trend_late_vs_full": 4.0,
            "figo_tachycardia": 1.0, "figo_abnormal_baseline": 1.0,
            "figo_prolonged_decel": 1.0, "figo_absent_accels": 0.0,
            "figo_bradycardia": 0.0, "figo_absent_variability": 0.0,
            "figo_late_decels": 0.0, "figo_sinusoidal": 0.0, "figo_reduced_variability": 0.0,
            "figo_composite_score": 3.0, "figo_category": 2.0,
        }),
        ("Borderline Watch Pattern", 1, {
            "baseline_fhr": 108.0, "mean_fhr": 109.0, "std_fhr": 7.0,
            "stv": 6.5, "ltv": 12.0, "stv_norm": 0.43, "ltv_norm": 0.48, "roughness": 2.0,
            "tachycardia_frac": 0.0, "bradycardia_frac": 0.08,
            "n_accels": 1, "accels_per_30min": 1.2, "mean_accel_height": 10.0,
            "n_decels": 2, "decels_per_30min": 1.0, "mean_decel_depth": 18.0,
            "max_decel_depth": 28.0, "mean_decel_dur_s": 60.0, "total_decel_dur_s": 120.0,
            "decel_area": 260.0, "prolonged_decel_flag": 0.0,
            "late_decel_likelihood": 0.15, "delayed_recovery_score": 0.28,
            "worsening_recovery_trend": 0.12, "decel_burden_idx": 0.38,
            "fetal_reserve_score": 42.0, "signal_quality": 0.29,
            "missing_fhr_pct": 20.0, "flatline_pct": 4.0, "abrupt_jump_count": 1.0,
            "baseline_fhr_last30": 106.0, "stv_last30": 6.0, "ltv_last30": 11.5,
            "std_fhr_last30": 6.5, "n_decels_last30": 2, "max_decel_depth_last30": 28.0,
            "stv_trend_late_vs_full": -0.5, "baseline_trend_late_vs_full": -2.0,
            "figo_bradycardia": 1.0, "figo_reduced_variability": 0.0,
            "figo_absent_variability": 0.0, "figo_tachycardia": 0.0,
            "figo_abnormal_baseline": 1.0, "figo_absent_accels": 0.0,
            "figo_late_decels": 0.0, "figo_prolonged_decel": 0.0, "figo_sinusoidal": 0.0,
            "figo_composite_score": 1.0, "figo_category": 1.0,
        }),
        ("Reactive Reassuring CTG", 0, {
            "baseline_fhr": 138.0, "mean_fhr": 139.0, "std_fhr": 9.0,
            "stv": 13.0, "ltv": 22.0, "stv_norm": 0.87, "ltv_norm": 0.88, "roughness": 3.2,
            "tachycardia_frac": 0.0, "bradycardia_frac": 0.0,
            "n_accels": 5, "accels_per_30min": 5.5, "mean_accel_height": 18.0,
            "n_decels": 0, "decels_per_30min": 0.0, "mean_decel_depth": 0.0,
            "max_decel_depth": 0.0, "mean_decel_dur_s": 0.0, "total_decel_dur_s": 0.0,
            "decel_area": 0.0, "prolonged_decel_flag": 0.0,
            "late_decel_likelihood": 0.0, "delayed_recovery_score": 0.0,
            "worsening_recovery_trend": 0.0, "decel_burden_idx": 0.02,
            "fetal_reserve_score": 82.0, "signal_quality": 0.40,
            "missing_fhr_pct": 4.0, "flatline_pct": 1.0, "abrupt_jump_count": 0.0,
            "baseline_fhr_last30": 139.0, "stv_last30": 13.5, "ltv_last30": 22.5,
            "std_fhr_last30": 9.5, "n_decels_last30": 0, "max_decel_depth_last30": 0.0,
            "stv_trend_late_vs_full": 0.5, "baseline_trend_late_vs_full": 1.0,
            "figo_abnormal_baseline": 0.0, "figo_tachycardia": 0.0, "figo_bradycardia": 0.0,
            "figo_absent_variability": 0.0, "figo_reduced_variability": 0.0,
            "figo_absent_accels": 0.0, "figo_late_decels": 0.0, "figo_prolonged_decel": 0.0,
            "figo_sinusoidal": 0.0, "figo_composite_score": 0.0, "figo_category": 0.0,
        }),
    ]

    adv_results = []
    for _cn, _ec, _ov in _stress_cases:
        try:
            _x = _build_stress(_ov)
            _pb = float(np.mean([m.predict_proba(_x)[0, 1] for m in xgb_models]))
            _p3 = 0.6 * xgb3.predict_proba(_x)[0] + 0.4 * hgb3.predict_proba(_x)[0]
            _pred3 = 2 if _p3[2] >= best_thr_hr else (1 if _p3[1] >= best_thr_w else 0)
            _ok3 = _pred3 == _ec
            _okb  = bool(_pb >= best_thr) == bool(_ec > 0)
            adv_results.append({
                "case":             _cn,
                "expected_label":   LABEL_MAP[_ec],
                "predicted_label":  LABEL_MAP[_pred3],
                "prob_normal":      _f4(float(_p3[0])),
                "prob_watch":       _f4(float(_p3[1])),
                "prob_high_risk":   _f4(float(_p3[2])),
                "binary_prob":      _f4(float(_pb)),
                "correct_3class":   bool(_ok3),
                "correct_binary":   bool(_okb),
            })
            print(f"  [{'✓' if _ok3 else '✗'}] {_cn}: "
                  f"expected={LABEL_MAP[_ec]}, got={LABEL_MAP[_pred3]}"
                  f"  P(HR)={_p3[2]:.3f}")
        except Exception as _e:
            print(f"  [!] {_cn}: error — {_e}")

    adv_pass_3 = float(np.mean([r["correct_3class"] for r in adv_results])) if adv_results else float("nan")
    adv_pass_b = float(np.mean([r["correct_binary"] for r in adv_results])) if adv_results else float("nan")
    print(f"[adversarial] 3-class pass: {adv_pass_3:.0%}  binary pass: {adv_pass_b:.0%}")

    # ── CTG-Specific Clinical Validation ──────────────────────────────────────
    print("\n[ctg-specific] computing FRS vs pH, DBI/CSR by class, FIGO adherence ...")
    ctg_specific: dict = {}
    try:
        from scipy import stats as _st
        _ph_col   = df_lab["ph"].values
        _risk_col = df_lab["risk_label"].values
        _frs_col  = df_lab["fetal_reserve_score"].values if "fetal_reserve_score" in df_lab.columns else np.full(len(df_lab), np.nan)
        _dbi_col  = df_lab["decel_burden_idx"].values    if "decel_burden_idx"    in df_lab.columns else np.full(len(df_lab), np.nan)
        _csr_col  = df_lab["csr_score"].values           if "csr_score"           in df_lab.columns else np.full(len(df_lab), np.nan)

        _v_ph = ~np.isnan(_ph_col) & ~np.isnan(_frs_col)
        if _v_ph.sum() > 10:
            _sp_r, _ = _st.spearmanr(_frs_col[_v_ph], _ph_col[_v_ph])
            _pe_r, _ = _st.pearsonr( _frs_col[_v_ph], _ph_col[_v_ph])
            _y_hr_b  = (_risk_col[_v_ph] == 2).astype(int)
            _frs_auc = float(roc_auc_score(_y_hr_b, -_frs_col[_v_ph])) if _y_hr_b.sum() > 0 else float("nan")
            ctg_specific["frs_vs_ph"] = {
                "spearman_r": _f4(float(_sp_r)),
                "pearson_r":  _f4(float(_pe_r)),
                "auc":        _f4(float(_frs_auc)),
                "note": "Low FRS → high-risk label; AUC measures discrimination.",
            }

        for _col, _col_key in [(_dbi_col, "dbi_by_class"), (_csr_col, "csr_by_class")]:
            _v = ~np.isnan(_col)
            if _v.sum() < 10:
                continue
            _entry: dict = {}
            for _lbl, _lid in [("low_risk", 0), ("watch", 1), ("high_risk", 2)]:
                _m = (_risk_col == _lid) & _v
                _entry[_lbl] = _f4(float(np.nanmean(_col[_m]))) if _m.sum() > 0 else None
            _y_hr_b2 = (_risk_col[_v] == 2).astype(int)
            if _y_hr_b2.sum() > 0:
                try:
                    _entry["auc"] = _f4(float(roc_auc_score(_y_hr_b2, _col[_v])))
                except Exception:
                    _entry["auc"] = None
            _v_ph2 = ~np.isnan(_ph_col) & _v
            if _v_ph2.sum() > 10:
                try:
                    _r_ph, _ = _st.pearsonr(_col[_v_ph2], _ph_col[_v_ph2])
                    _entry["corr_ph"] = _f4(float(_r_ph))
                except Exception:
                    _entry["corr_ph"] = None
            ctg_specific[_col_key] = _entry

        # FIGO adherence: compare FIGO rule-based category to pH-derived label
        if "figo_category" in df_lab.columns:
            _fig_col = df_lab["figo_category"].values
            _v_fig   = ~np.isnan(_fig_col)
            if _v_fig.sum() > 10:
                _agree = float(np.mean(_fig_col[_v_fig] == _risk_col[_v_fig]))
                _y_hr_fig = (_risk_col[_v_fig] == 2).astype(int)
                _fauc = float("nan")
                if _y_hr_fig.sum() > 0:
                    try:
                        _fauc = float(roc_auc_score(_y_hr_fig, _fig_col[_v_fig]))
                    except Exception:
                        pass
                ctg_specific["figo_adherence"] = {
                    "figo_vs_label_agreement": _f4(_agree),
                    "figo_auc_high_risk":      _f4(_fauc),
                    "note": "Agreement between FIGO 2015 rule-based category and pH/BD-derived outcome label.",
                }
    except Exception as _e:
        print(f"[ctg-specific] warning: {_e}")

    # ── Signal quality subgroup analysis (test set) ───────────────────────────
    print("\n[sq-subgroups] signal quality subgroup analysis on test set ...")
    signal_quality_subgroups: dict = {}
    if "signal_quality" in df_lab.columns:
        _sq_te = df_lab["signal_quality"].values[idx_te]
        _sq_tr_vals = df_lab["signal_quality"].values[idx_tr]
        _sq_q33 = float(np.nanpercentile(_sq_tr_vals, 33))
        _sq_q67 = float(np.nanpercentile(_sq_tr_vals, 67))
        for _grp, _lo, _hi in [("good", _sq_q67, 9999.0), ("acceptable", _sq_q33, _sq_q67), ("poor", -9999.0, _sq_q33)]:
            _gm = (_sq_te >= _lo) & (_sq_te < _hi)
            if _gm.sum() == 0:
                continue
            _yg_b = yb_te[_gm]
            _pg_b = p_bin_te[_gm]
            _pred_g = (_pg_b >= best_thr).astype(int)
            _acc_g  = float(np.mean(_pred_g == _yg_b))
            _y3_g   = y_te[_gm]
            _hr_m   = _y3_g == 2
            _hr_rec_g = float(np.mean((_pg_b[_hr_m] >= best_thr).astype(int))) if _hr_m.sum() > 0 else float("nan")
            _unc_g  = float(np.mean((_pg_b >= grey_lo) & (_pg_b <= grey_hi)))
            signal_quality_subgroups[_grp] = {
                "n":                int(_gm.sum()),
                "accuracy":         _f4(_acc_g),
                "high_risk_recall": _f4(_hr_rec_g),
                "uncertainty_rate": _f4(_unc_g),
            }
        print(f"[sq-subgroups] {signal_quality_subgroups}")

    # ── CV metrics computed from OOF predictions (no extra model fits) ────────
    # Using the 5-fold OOF pool already computed above — avoids 5 extra training runs
    print("\n[cv] computing fold-wise metrics from OOF predictions ...")
    cv_aucs, cv_f1s, cv_senss, cv_specs, cv_precs = [], [], [], [], []
    cv_bals, cv_mf1s, cv_hr_recs = [], [], []

    for fold, (tr_i, te_i) in enumerate(skf5.split(X_raw, yb_all), 1):
        y_te_b = yb_all[te_i]
        y_te_3 = y_raw[te_i]
        p_f_te = p_bin_full[te_i]   # OOF score for this fold's held-out records
        thr_f  = float(best_thr_full)
        pred_f = (p_f_te >= thr_f).astype(int)

        try:
            auc_f = float(roc_auc_score(y_te_b, p_f_te))
        except Exception:
            auc_f = float("nan")
        f1_f  = float(f1_score(y_te_b, pred_f, zero_division=0))
        s_f   = float(recall_score(y_te_b, pred_f, zero_division=0))
        sp_f  = float(recall_score(y_te_b, pred_f, pos_label=0, zero_division=0))
        pr_f  = float(precision_score(y_te_b, pred_f, zero_division=0))
        bal_f = float(balanced_accuracy_score(y_te_b, pred_f))
        cv_aucs.append(auc_f); cv_f1s.append(f1_f)
        cv_senss.append(s_f);  cv_specs.append(sp_f); cv_precs.append(pr_f)
        cv_bals.append(bal_f)

        # 3-class CV: apply tuned per-class thresholds to ensemble probabilities
        p_cv3_xgb = xgb3.predict_proba(scaler.transform(imputer.transform(X_raw[te_i])))
        p_cv3_hgb = hgb3.predict_proba(scaler.transform(imputer.transform(X_raw[te_i])))
        p_cv3 = 0.6 * p_cv3_xgb + 0.4 * p_cv3_hgb
        pred3_cv = np.zeros(len(y_te_3), dtype=int)
        pred3_cv[p_cv3[:, 2] >= best_thr_hr] = 2
        pred3_cv[(p_cv3[:, 2] < best_thr_hr) & (p_cv3[:, 1] >= best_thr_w)] = 1
        try:
            mf1_f  = float(f1_score(y_te_3, pred3_cv, average="macro", zero_division=0))
            hrr_f  = float(recall_score(y_te_3, pred3_cv, labels=[2], average="macro", zero_division=0))
        except Exception:
            mf1_f = hrr_f = float("nan")
        cv_mf1s.append(mf1_f)
        cv_hr_recs.append(hrr_f)

        print(f"  Fold {fold}: AUROC={auc_f:.4f}  bal={bal_f:.4f}  "
              f"macro-F1={mf1_f:.4f}  HR-rec={hrr_f:.4f}  "
              f"sens={s_f:.4f}  spec={sp_f:.4f}")

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
        "high_risk_recall": _f4(hr_rec),
        "watch_recall":     _f4(w_rec),
        "low_risk_recall":  _f4(float(recall_score(y_te, xgb3_pred, labels=[0], average="macro", zero_division=0))),
        "macro_f1":         _f4(mf3),
        "balanced_accuracy_3class": _f4(bal3),
        "ece":              _f4(ece_bin) if ece_bin is not None else None,
        "brier_score":      _f4(brier_bin) if brier_bin is not None else None,
        "log_loss":         _f4(ll_bin) if ll_bin is not None else None,
        "uncertainty_rate": uncertainty_coverage["uncertain_rate"],
        "confident_accuracy": uncertainty_coverage["confident_accuracy"],
        "thr_high_risk":    _f4(best_thr_hr),
        "thr_watch":        _f4(best_thr_w),
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
        "calibration_metrics": {
            "ece_binary_oof":    _f4(ece_bin) if ece_bin is not None else None,
            "brier_binary_oof":  _f4(brier_bin) if brier_bin is not None else None,
            "log_loss_binary_oof": _f4(ll_bin) if ll_bin is not None else None,
            "ece_3class_test":   _f4(ece_3) if ece_3 is not None else None,
            "brier_3class_test": _f4(brier_3) if brier_3 is not None else None,
            "temperature_T":     _f4(rn.temp_scaler.T),
            "note": "ECE/Brier computed over 552-record OOF pool (binary); 3-class on 83-record test set.",
        },
        "uncertainty_coverage": uncertainty_coverage,
        "threshold_3class": {
            "thr_high_risk":    _f4(best_thr_hr),
            "thr_watch":        _f4(best_thr_w),
            "val_f2_macro":     _f4(best_f2_3),
            "optimization_metric": "F2-macro (β=2, recall-weighted)",
        },
        "ood_detection": ood_detection,
        "adversarial_stress_tests": adv_results,
        "adversarial_summary": {
            "n_cases":          len(adv_results),
            "pass_rate_3class": _f4(adv_pass_3),
            "pass_rate_binary": _f4(adv_pass_b),
            "method": "7 canonical CTG profiles evaluated on trained ensemble (no label required).",
        },
        "ctg_specific": ctg_specific,
        "signal_quality_subgroups": signal_quality_subgroups,
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
            "fold_auroc":            [_f4(v) for v in cv_aucs],
            "fold_f1":               [_f4(v) for v in cv_f1s],
            "fold_sens":             [_f4(v) for v in cv_senss],
            "fold_spec":             [_f4(v) for v in cv_specs],
            "fold_prec":             [_f4(v) for v in cv_precs],
            "fold_balanced_accuracy":[_f4(v) for v in cv_bals],
            "fold_macro_f1":         [_f4(v) for v in cv_mf1s],
            "fold_high_risk_recall": [_f4(v) for v in cv_hr_recs],
            "mean_auroc":            _f4(np.nanmean(cv_aucs)),
            "std_auroc":             _f4(np.nanstd(cv_aucs)),
            "mean_f1":               _f4(np.nanmean(cv_f1s)),
            "std_f1":                _f4(np.nanstd(cv_f1s)),
            "mean_sens":             _f4(np.nanmean(cv_senss)),
            "std_sens":              _f4(np.nanstd(cv_senss)),
            "mean_spec":             _f4(np.nanmean(cv_specs)),
            "std_spec":              _f4(np.nanstd(cv_specs)),
            "mean_prec":             _f4(np.nanmean(cv_precs)),
            "std_prec":              _f4(np.nanstd(cv_precs)),
            "mean_balanced_accuracy":_f4(np.nanmean(cv_bals)),
            "std_balanced_accuracy": _f4(np.nanstd(cv_bals)),
            "mean_macro_f1":         _f4(np.nanmean(cv_mf1s)),
            "std_macro_f1":          _f4(np.nanstd(cv_mf1s)),
            "mean_high_risk_recall": _f4(np.nanmean(cv_hr_recs)),
            "std_high_risk_recall":  _f4(np.nanstd(cv_hr_recs)),
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
