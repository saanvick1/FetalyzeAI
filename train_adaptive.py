"""
train_adaptive.py
=================
FetalyzeAI AdaptiveReserveNet v3.0 — Training Entry Point

Architecture: AdaptiveReserveNet v3.0
  Layer 0 : Preprocessing (impute + RobustScale, leakage-free per fold)
  Layer 1 : 4 domain-expert classifiers
              A — Baseline FHR        (Logistic Regression)
              B — Variability + Spec  (Gradient Boosting)
              C — Event patterns      (Random Forest)
              D — Temporal trends     (Logistic Regression)  ← NEW
  Layer 2 : AttentionGatingMLP        ← NEW (replaces heuristic conf weighting)
  Layer 3 : ReserveFusionMLP meta-learner
              Input: expert_probs(12) | attention_gates(4) | top_raw(10)
  Layer 4 : Temperature scaling (val-calibrated)
  Layer 5 : Conformal prediction sets ← NEW
  Layer 6 : IncrementalAdapter        ← NEW (partial_fit + EWC)

Outer loop: TOPQUA stacked ensemble (XGB bag + ET + RF + LR) feeds
            OOF probabilities for the headline binary AUROC/sens/spec.
            AdaptiveReserveNet is trained as the primary explainable model.

Data: CTU-CHB only (552 real intrapartum recordings). No synthetic fallback.

Usage:
    python train_adaptive.py                    # full training run
    python train_adaptive.py --update <pkl>     # incremental update with new data
"""

from __future__ import annotations

import argparse, json, time, pickle, warnings
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
    roc_curve, precision_recall_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from ctu_loader          import load_ctu_records
from ctg_feature_engine  import (
    extract_record_features, add_timeline_trends,
)
from metrics_utils       import (
    compute_all_metrics, bootstrap_confidence_intervals,
)
from adaptive_reservenet import AdaptiveReserveNet
from model_registry      import ModelRegistry

ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

MODEL_VERSION = "arnet-3.0"
LABEL_MAP     = {0: "Low Risk", 1: "Watch Closely", 2: "High Risk"}

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


# ─────────────────────────────────────────────────────────────────────────────
# Label assignment (identical to train_reservenet_ctu.py)
# ─────────────────────────────────────────────────────────────────────────────

def assign_clinical_label(row) -> int:
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


def record_level_split(feat_df, test_frac=0.15, val_frac=0.15, seed=42):
    rids      = feat_df["record_id"].values
    unique_r  = np.unique(rids)
    lbl_per   = feat_df.groupby("record_id")["risk_label"].first().reindex(unique_r).values
    train_val, test = train_test_split(
        unique_r, test_size=test_frac, stratify=lbl_per, random_state=seed)
    tv_lbl = feat_df.groupby("record_id")["risk_label"].first().reindex(train_val).values
    train, val = train_test_split(
        train_val, test_size=val_frac / (1 - test_frac), stratify=tv_lbl, random_state=seed)
    return (
        feat_df["record_id"].isin(train).values,
        feat_df["record_id"].isin(val).values,
        feat_df["record_id"].isin(test).values,
    )


def _f4(v):
    if v is None: return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f): return None
        return round(f, 4)
    except Exception:
        return v


def smote_binary(X, y, seed=42, k=5):
    if HAS_SMOTE:
        try:
            sm = SMOTE(sampling_strategy="minority",
                       k_neighbors=min(k, int(np.sum(y == 1)) - 1),
                       random_state=seed)
            return sm.fit_resample(X, y)
        except Exception:
            pass
    minority = X[y == 1]; majority = X[y == 0]
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
        synth.append(minority[idx] + rng.uniform(0, 1) * (minority[nb] - minority[idx]))
    X_syn = np.vstack([X, minority, np.array(synth)])
    y_syn = np.concatenate([y, np.ones(len(minority) + len(synth), dtype=int)])
    return X_syn, y_syn


def tune_xgb(X_tr, y_tr, X_va, y_va, n_trials=8):
    if not HAS_OPTUNA:
        return dict(n_estimators=600, max_depth=4, learning_rate=0.025,
                    subsample=0.85, colsample_bytree=0.80,
                    min_child_weight=4, reg_alpha=0.3, reg_lambda=3.0, gamma=0.5)

    def objective(trial):
        p = dict(
            n_estimators     = trial.suggest_int("ne", 200, 1000, step=100),
            max_depth        = trial.suggest_int("md", 3, 6),
            learning_rate    = trial.suggest_float("lr", 0.01, 0.08, log=True),
            subsample        = trial.suggest_float("sub", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("col", 0.6, 1.0),
            min_child_weight = trial.suggest_int("mcw", 2, 8),
            reg_alpha        = trial.suggest_float("ra", 0.01, 2.0, log=True),
            reg_lambda       = trial.suggest_float("rl", 0.5, 10.0, log=True),
            gamma            = trial.suggest_float("gm", 0.0, 2.0),
        )
        spw = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1)
        m = xgb.XGBClassifier(
            **p, scale_pos_weight=spw, objective="binary:logistic",
            eval_metric="auc", random_state=42, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=30,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        try:
            return float(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))
        except Exception:
            return 0.5

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=180)
    return study.best_params


def roc_pts(y, s, n=30):
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return [{"fpr": _f4(float(fpr[i])), "tpr": _f4(float(tpr[i]))} for i in idx]


def pr_pts(y, s, n=30):
    p, r, _ = precision_recall_curve(y, s)
    idx = np.unique(np.linspace(0, len(p) - 1, n).astype(int))
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
        rows.append({"class": name, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": _f4(prec), "recall": _f4(rec), "f1": _f4(f1),
                     "support": int(np.sum(y_true == c))})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("\n" + "=" * 65)
    print(" FetalyzeAI AdaptiveReserveNet v3.0 — CTU-CHB Training")
    print("=" * 65)

    records = load_ctu_records(verbose=True)

    print(f"[features] extracting for {len(records)} records ...")
    feats = [extract_record_features(r) for r in records]
    df = pd.DataFrame(feats)
    print(f"[features] shape {df.shape}")

    df["risk_label"] = df.apply(assign_clinical_label, axis=1)
    df_lab = df[df["risk_label"] >= 0].copy().reset_index(drop=True)
    counts = df_lab["risk_label"].value_counts().sort_index().to_dict()
    print(f"[labels] labeled={len(df_lab)}  "
          f"normal={counts.get(0,0)}  watch={counts.get(1,0)}  high={counts.get(2,0)}")

    cols    = [c for c in FEATURE_COLS if c in df_lab.columns]
    X_raw   = df_lab[cols].values.astype(float)
    y_raw   = df_lab["risk_label"].values.astype(int)
    rec_map = {r: i for i, r in enumerate(df_lab["record_id"].values)}

    idx_tr, idx_val, idx_te = record_level_split(df_lab)
    print(f"[split] train={idx_tr.sum()}  val={idx_val.sum()}  test={idx_te.sum()}")

    # Leakage-free preprocessing
    imputer = SimpleImputer(strategy="median").fit(X_raw[idx_tr])
    scaler  = RobustScaler().fit(imputer.transform(X_raw[idx_tr]))
    def transform(X): return scaler.transform(imputer.transform(X))

    X_tr = transform(X_raw[idx_tr]); y_tr = y_raw[idx_tr]
    X_va = transform(X_raw[idx_val]); y_va = y_raw[idx_val]
    X_te = transform(X_raw[idx_te]); y_te = y_raw[idx_te]

    yb_tr = (y_tr >= 1).astype(int)
    yb_va = (y_va >= 1).astype(int)
    yb_te = (y_te >= 1).astype(int)

    # SMOTE on training fold only
    print(f"\n[smote] before: {np.bincount(yb_tr)}")
    X_tr_s, yb_tr_s = smote_binary(X_tr, yb_tr, seed=42)
    print(f"[smote] after:  {np.bincount(yb_tr_s)}")

    # ── AdaptiveReserveNet — primary explainable model ────────────────────────
    print("\n[arnet] training AdaptiveReserveNet v3.0 ...")
    arnet = AdaptiveReserveNet(
        n_classes=3, random_state=42,
        use_pulse_enc=True,
        replay_capacity=200,
        ewc_lambda=0.5,
        conformal_alpha=0.10,
    )
    arnet.fit(X_tr, y_tr, X_va, y_va, cols)
    arnet_te   = arnet.predict_proba(X_te)
    arnet_unc  = arnet.predict_with_uncertainty(X_te)
    arnet_metr = compute_all_metrics(y_te, arnet_te, threshold=0.35)
    print(f"[arnet] test AUROC(macro)={arnet_metr['auroc_macro']:.4f}  "
          f"sens={arnet_metr['sensitivity']:.4f}  F1={arnet_metr['macro_f1']:.4f}")

    # Save arnet + preprocessors together
    arnet_bundle = {"model": arnet, "imputer": imputer, "scaler": scaler, "cols": cols}
    with open(MODELS_DIR / "adaptive_reservenet.pkl", "wb") as f:
        pickle.dump(arnet_bundle, f)

    # ── TOPQUA stacked ensemble — headline binary metrics (OOF) ───────────────
    print("\n[topqua] training TOPQUA stacked ensemble for headline metrics ...")
    spw = float(np.sum(yb_tr_s == 0)) / max(float(np.sum(yb_tr_s == 1)), 1)

    print("[xgb] Optuna-tuning XGBoost ...")
    best_p = tune_xgb(X_tr_s, yb_tr_s, X_va, yb_va, n_trials=8)
    # normalise Optuna short keys
    for old, new in [("lr","learning_rate"),("ra","reg_alpha"),("rl","reg_lambda"),
                     ("gm","gamma"),("sub","subsample"),("col","colsample_bytree"),
                     ("mcw","min_child_weight"),("ne","n_estimators"),("md","max_depth")]:
        if old in best_p: best_p[new] = best_p.pop(old)

    xgb_models, pv_xgb, pt_xgb = [], [], []
    for sd in [42, 7, 2024, 1337, 99]:
        m = xgb.XGBClassifier(
            **best_p, scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="auc",
            random_state=sd, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=40,
        )
        m.fit(X_tr_s, yb_tr_s, eval_set=[(X_va, yb_va)], verbose=False)
        xgb_models.append(m)
        pv_xgb.append(m.predict_proba(X_va)[:, 1])
        pt_xgb.append(m.predict_proba(X_te)[:, 1])
    p_xgb_va = np.mean(pv_xgb, axis=0)
    p_xgb_te = np.mean(pt_xgb, axis=0)

    et_bin = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=3,
                                  class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    et_bin.fit(X_tr_s, yb_tr_s)
    p_et_va = et_bin.predict_proba(X_va)[:, 1]
    p_et_te = et_bin.predict_proba(X_te)[:, 1]

    rf_bin = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=3,
                                    class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rf_bin.fit(X_tr_s, yb_tr_s)
    p_rf_va = rf_bin.predict_proba(X_va)[:, 1]
    p_rf_te = rf_bin.predict_proba(X_te)[:, 1]

    lr_bin = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000,
                                solver="liblinear", random_state=42)
    lr_bin.fit(X_tr_s, yb_tr_s)
    p_lr_va = lr_bin.predict_proba(X_va)[:, 1]
    p_lr_te = lr_bin.predict_proba(X_te)[:, 1]

    # 5-fold OOF stacked meta
    print("[stack] 5-fold OOF meta-learner ...")
    yb_all = (y_raw >= 1).astype(int)
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(X_raw)); oof_et = np.zeros(len(X_raw))
    oof_rf  = np.zeros(len(X_raw)); oof_lr = np.zeros(len(X_raw))

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
        print(f"  Fold {fold_i} XGB OOF AUROC: {roc_auc_score(yb_all[te_i], oof_xgb[te_i]):.4f}")

    meta_X_tv = np.column_stack([oof_xgb[idx_tr | idx_val], oof_et[idx_tr | idx_val],
                                  oof_rf[idx_tr | idx_val],  oof_lr[idx_tr | idx_val]])
    meta_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    meta_lr.fit(meta_X_tv, yb_all[idx_tr | idx_val])

    meta_X_te = np.column_stack([p_xgb_te, p_et_te, p_rf_te, p_lr_te])
    p_meta_te = meta_lr.predict_proba(meta_X_te)[:, 1]
    meta_X_va = np.column_stack([p_xgb_va, p_et_va, p_rf_va, p_lr_va])
    p_meta_va = meta_lr.predict_proba(meta_X_va)[:, 1]

    # Soft-vote final score
    p_bin_va = 0.6 * p_meta_va + 0.4 * p_xgb_va
    p_bin_te = 0.6 * p_meta_te + 0.4 * p_xgb_te

    # OOF full 552-record pool
    meta_X_full = np.column_stack([oof_xgb, oof_et, oof_rf, oof_lr])
    p_meta_full = np.zeros(len(X_raw))
    for tr_i, te_i in StratifiedKFold(n_splits=5, shuffle=True, random_state=4242).split(meta_X_full, yb_all):
        m_f = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
        m_f.fit(meta_X_full[tr_i], yb_all[tr_i])
        p_meta_full[te_i] = m_f.predict_proba(meta_X_full[te_i])[:, 1]
    p_bin_full = 0.6 * p_meta_full + 0.4 * oof_xgb

    # Youden threshold
    fpr_v, tpr_v, thr_v = roc_curve(yb_va, p_bin_va)
    best_thr = float(np.clip(thr_v[int(np.argmax(tpr_v - fpr_v))], 0.05, 0.95))
    fpr_f, tpr_f, thr_f = roc_curve(yb_all, p_bin_full)
    best_thr_full = float(np.clip(thr_f[int(np.argmax(tpr_f - fpr_f))], 0.05, 0.95))

    yb_pred_full  = (p_bin_full >= best_thr_full).astype(int)
    bin_auroc     = float(roc_auc_score(yb_all, p_bin_full))
    bin_auprc     = float(average_precision_score(yb_all, p_bin_full))
    bin_sens      = float(recall_score(yb_all, yb_pred_full, zero_division=0))
    bin_spec      = float(recall_score(yb_all, yb_pred_full, pos_label=0, zero_division=0))
    bin_f1        = float(f1_score(yb_all, yb_pred_full, zero_division=0))
    bin_prec      = float(precision_score(yb_all, yb_pred_full, zero_division=0))
    bin_bal       = float(balanced_accuracy_score(yb_all, yb_pred_full))
    print(f"\n[oof-full]  AUROC={bin_auroc:.4f}  AUPRC={bin_auprc:.4f}  "
          f"sens={bin_sens:.4f}  spec={bin_spec:.4f}  F1={bin_f1:.4f}")

    yb_pred_te    = (p_bin_te >= best_thr).astype(int)
    holdout_auroc = float(roc_auc_score(yb_te, p_bin_te))
    holdout_sens  = float(recall_score(yb_te, yb_pred_te, zero_division=0))
    holdout_spec  = float(recall_score(yb_te, yb_pred_te, pos_label=0, zero_division=0))
    holdout_f1    = float(f1_score(yb_te, yb_pred_te, zero_division=0))
    print(f"[holdout]   AUROC={holdout_auroc:.4f}  sens={holdout_sens:.4f}  "
          f"spec={holdout_spec:.4f}  F1={holdout_f1:.4f}")

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    print("\n[bootstrap] computing 300-iter CIs ...")
    rng_b = np.random.RandomState(42)
    bs_a, bs_s, bs_sp, bs_f = [], [], [], []
    for _ in range(300):
        idx = rng_b.choice(len(yb_all), len(yb_all), replace=True)
        try:
            bs_a.append(float(roc_auc_score(yb_all[idx], p_bin_full[idx])))
            yp = (p_bin_full[idx] >= best_thr_full).astype(int)
            bs_s.append(float(recall_score(yb_all[idx], yp, zero_division=0)))
            bs_sp.append(float(recall_score(yb_all[idx], yp, pos_label=0, zero_division=0)))
            bs_f.append(float(f1_score(yb_all[idx], yp, zero_division=0)))
        except Exception:
            pass

    def ci(arr):
        if not arr: return {"mean": None, "ci_lo": None, "ci_hi": None}
        return {"mean": _f4(np.mean(arr)),
                "ci_lo": _f4(np.percentile(arr, 2.5)),
                "ci_hi": _f4(np.percentile(arr, 97.5))}

    # ── Confusion matrix ───────────────────────────────────────────────────────
    xgb3 = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.025,
        subsample=0.85, colsample_bytree=0.80,
        min_child_weight=4, reg_alpha=0.4, reg_lambda=3.0, gamma=0.5,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=40,
    )
    spw3 = float(np.sum(y_tr != 2)) / max(float(np.sum(y_tr == 2)), 1)
    xgb3.set_params(scale_pos_weight=spw3)
    xgb3.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb3_pred = xgb3.predict_proba(X_te).argmax(axis=1)
    cm3 = confusion_matrix(y_te, xgb3_pred, labels=[0, 1, 2]).tolist()
    per_class = per_class_metrics(y_te, xgb3_pred)

    # ── Expert importances from arnet ─────────────────────────────────────────
    expert_imps = arnet.expert_importances()
    attn_sample = arnet.attention_weights_for(X_te[:10]).mean(axis=0).tolist()

    # ── Conformal uncertainty summary ─────────────────────────────────────────
    unc_dist = {
        "mean":   _f4(float(arnet_unc["uncertainty"].mean())),
        "low_pct":  _f4(float(np.mean(arnet_unc["uncertainty"] < 0.33) * 100)),
        "mid_pct":  _f4(float(np.mean((arnet_unc["uncertainty"] >= 0.33) & (arnet_unc["uncertainty"] < 0.67)) * 100)),
        "high_pct": _f4(float(np.mean(arnet_unc["uncertainty"] >= 0.67) * 100)),
        "conformal_q":   _f4(float(arnet.conformal.q_hat)),
        "conformal_alpha": arnet.conformal_alpha,
        "temperature_T":   _f4(float(arnet.temp_scaler.T)),
    }

    elapsed = round(time.time() - t0, 1)

    # ── Build output JSON ─────────────────────────────────────────────────────
    out = {
        "generated_at":      datetime.now().isoformat(),
        "model_version":     MODEL_VERSION,
        "architecture":      "AdaptiveReserveNet v3.0",
        "dataset":           "CTU-CHB Intrapartum CTG (real records only)",
        "n_records":         int(len(records)),
        "n_labeled":         int(len(df_lab)),
        "class_counts":      {str(k): int(v) for k, v in counts.items()},
        "n_features":        len(cols),
        "feature_cols":      cols,
        "split": {
            "train": int(idx_tr.sum()), "val": int(idx_val.sum()),
            "test":  int(idx_te.sum()),
        },
        "architecture_detail": {
            "n_experts":           4,
            "experts":             ["baseline_fhr", "variability_spectral", "event_patterns", "temporal_trends"],
            "attention_gating":    True,
            "pulse_encoder":       arnet.pulse_encoder is not None,
            "meta_learner":        "MLPClassifier(hidden=[128,64,32])",
            "temperature_scaling": True,
            "conformal_prediction": True,
            "incremental_learning": True,
            "ewc_regularisation":  True,
        },
        "binary_headline": {
            "auroc":       _f4(bin_auroc),
            "auprc":       _f4(bin_auprc),
            "sensitivity": _f4(bin_sens),
            "specificity": _f4(bin_spec),
            "f1":          _f4(bin_f1),
            "precision":   _f4(bin_prec),
            "balanced_acc":_f4(bin_bal),
            "threshold":   _f4(best_thr_full),
            "method":      "5-fold OOF stacked ensemble (552 records)",
        },
        "holdout_test": {
            "auroc":       _f4(holdout_auroc),
            "sensitivity": _f4(holdout_sens),
            "specificity": _f4(holdout_spec),
            "f1":          _f4(holdout_f1),
            "threshold":   _f4(best_thr),
            "n":           int(idx_te.sum()),
        },
        "arnet_test": {k: _f4(v) if isinstance(v, float) else v
                       for k, v in arnet_metr.items()
                       if not isinstance(v, (list, np.ndarray))},
        "bootstrap_cis": {
            "auroc":       ci(bs_a),
            "sensitivity": ci(bs_s),
            "specificity": ci(bs_sp),
            "f1":          ci(bs_f),
        },
        "roc_curve":    roc_pts(yb_all, p_bin_full),
        "pr_curve":     pr_pts(yb_all, p_bin_full),
        "confusion_matrix_3class": cm3,
        "per_class_metrics":       per_class,
        "uncertainty":             unc_dist,
        "attention_mean_weights":  {
            "baseline_fhr":          _f4(attn_sample[0]),
            "variability_spectral":  _f4(attn_sample[1]),
            "event_patterns":        _f4(attn_sample[2]),
            "temporal_trends":       _f4(attn_sample[3]),
        },
        "expert_top_features": {
            name: sorted(imp.items(), key=lambda x: -x[1])[:5]
            for name, imp in expert_imps.items() if imp
        },
        "training_time_s": elapsed,
    }

    # Write primary results file
    out_path = ROOT / "ctu_model_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[output] {out_path}")

    # Also write to results/ for registry
    with open(RESULTS_DIR / f"arnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(out, f, indent=2)

    # ── Model registry ────────────────────────────────────────────────────────
    registry = ModelRegistry(MODELS_DIR / "registry")
    registry.save(arnet, out["binary_headline"] | out.get("arnet_test", {}),
                  out["architecture_detail"], tag="ctu-chb")
    registry.export_summary(RESULTS_DIR / "model_registry_summary.json")
    registry.summary()

    print(f"\n{'='*65}")
    print(f" Training complete in {elapsed:.1f}s")
    print(f" Binary AUROC (OOF-full): {bin_auroc:.4f}")
    print(f" Sensitivity:             {bin_sens:.4f}")
    print(f" Specificity:             {bin_spec:.4f}")
    print(f" Holdout AUROC:           {holdout_auroc:.4f}")
    print(f"{'='*65}\n")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Incremental update mode
# ─────────────────────────────────────────────────────────────────────────────

def run_update(bundle_path: str) -> None:
    """
    Load an existing AdaptiveReserveNet bundle and run partial_fit
    on any new data found in attached_assets/.

    Usage: python train_adaptive.py --update models/adaptive_reservenet.pkl
    """
    print(f"\n[update] loading {bundle_path} ...")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    model: AdaptiveReserveNet = bundle["model"]
    imputer = bundle["imputer"]
    scaler  = bundle["scaler"]
    cols    = bundle["cols"]

    print(f"[update] model version {model.VERSION}, replay buffer size {len(model.replay_buffer)}")

    # Load new records (same strict loader — no synthetic fallback)
    new_records = load_ctu_records(verbose=True)
    feats = [extract_record_features(r) for r in new_records]
    df = pd.DataFrame(feats)
    df["risk_label"] = df.apply(assign_clinical_label, axis=1)
    df_new = df[df["risk_label"] >= 0].reset_index(drop=True)
    print(f"[update] {len(df_new)} labeled records available for update")

    X_new = scaler.transform(imputer.transform(df_new[[c for c in cols if c in df_new.columns]].values.astype(float)))
    y_new = df_new["risk_label"].values.astype(int)

    model.partial_fit(X_new, y_new)
    bundle["model"] = model
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[update] saved updated model → {bundle_path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=str, default=None,
                        help="Path to existing model bundle for incremental update")
    args = parser.parse_args()

    if args.update:
        run_update(args.update)
    else:
        main()
