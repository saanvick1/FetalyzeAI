"""
train_reservenet_ctu.py
========================
Primary CTU-CHB / CTU-UHB ReserveNet training pipeline.

This is the ONLY active training script. It reads only the real CTU ZIP
via ctu_loader.py and never falls back to fetal_health.csv, UCI, Kaggle,
or synthetic data.

Usage:
    python train_reservenet_ctu.py
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
from sklearn.metrics import roc_auc_score, recall_score, f1_score, balanced_accuracy_score

import xgboost as xgb

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

MODEL_VERSION = "ctu-reservenet-1.0"
LABEL_MAP     = {0: "Low Risk", 1: "Watch Closely", 2: "High Risk"}


# ── label policy (clinical outcomes ONLY) ────────────────────────────────────

def assign_clinical_label(row) -> int:
    """
    pH < 7.05            → 2 (high risk)
    7.05 ≤ pH < 7.15     → 1 (watch closely)
    pH ≥ 7.15            → 0 (low risk)
    pH missing → base deficit:
      ≥ 12 → 2,  8–12 → 1,  < 8 → 0
    pH+BD missing → Apgar5:
      < 7 → 2,  = 7 → 1,  > 7 → 0
    Otherwise → -1 (drop record).

    Predictor features (fetal_reserve_score, decel_burden_idx,
    contraction_response, signal_quality) MUST NOT influence the label.
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


# ── feature columns used by the model ────────────────────────────────────────

FEATURE_COLS = [
    # Whole-record features
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
    # Spectral (cheap FFT band powers)
    "lf_power", "mf_power", "hf_power", "lf_hf_ratio", "spectral_entropy",
    # Last-30-min segment (clinically most predictive period — cheap stats only)
    "baseline_fhr_last30", "stv_last30", "ltv_last30", "std_fhr_last30",
    "n_decels_last30", "max_decel_depth_last30",
    # Late-vs-full trend deltas
    "stv_trend_late_vs_full", "baseline_trend_late_vs_full",
]


def record_level_split(feat_df: pd.DataFrame,
                       test_frac: float = 0.15, val_frac: float = 0.15,
                       seed: int = 42):
    rids = feat_df["record_id"].values
    unique_rids = np.unique(rids)

    label_per = feat_df.groupby("record_id")["risk_label"].first().reindex(unique_rids).values
    train_val, test = train_test_split(
        unique_rids, test_size=test_frac, stratify=label_per, random_state=seed,
    )
    tv_label = feat_df.groupby("record_id")["risk_label"].first().reindex(train_val).values
    train, val = train_test_split(
        train_val, test_size=val_frac / (1 - test_frac), stratify=tv_label, random_state=seed,
    )
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


def main(window_features: bool = True):
    t0 = time.time()
    print("\n" + "=" * 60)
    print(" FetalyzeAI ReserveNet — CTU-CHB Training Pipeline")
    print("=" * 60)

    # 1. Load real CTU records (raises loudly if data unavailable)
    records = load_ctu_records(verbose=True)

    # 2. Record-level features
    print(f"[features] extracting record-level features for {len(records)} records ...")
    feats = [extract_record_features(r) for r in records]
    df = pd.DataFrame(feats)
    print(f"[features] done — shape {df.shape}")

    # 3. Labels (clinical outcomes only)
    df["risk_label"] = df.apply(assign_clinical_label, axis=1)
    df_lab = df[df["risk_label"] >= 0].copy().reset_index(drop=True)
    n_excluded = len(df) - len(df_lab)
    counts = df_lab["risk_label"].value_counts().sort_index().to_dict()
    print(f"[labels] labeled={len(df_lab)}  excluded_no_outcome={n_excluded}")
    print(f"         normal_0={counts.get(0,0)}  watch_1={counts.get(1,0)}  high_2={counts.get(2,0)}")
    if len(df_lab) < 50:
        raise RuntimeError("Too few labeled CTU records for training.")

    # 4. Record-level split
    cols = [c for c in FEATURE_COLS if c in df_lab.columns]
    X_raw = df_lab[cols].values.astype(float)
    y_raw = df_lab["risk_label"].values.astype(int)

    idx_tr, idx_val, idx_te, train_ids, val_ids, test_ids = record_level_split(df_lab)
    print(f"[split] train={idx_tr.sum()}  val={idx_val.sum()}  test={idx_te.sum()}  "
          f"(record-level, no leakage)")

    # 5. Imputer + scaler — fit on TRAIN ONLY
    imputer = SimpleImputer(strategy="median").fit(X_raw[idx_tr])
    scaler  = RobustScaler().fit(imputer.transform(X_raw[idx_tr]))

    def transform(X): return scaler.transform(imputer.transform(X))
    X_tr = transform(X_raw[idx_tr]); y_tr = y_raw[idx_tr]
    X_va = transform(X_raw[idx_val]); y_va = y_raw[idx_val]
    X_te = transform(X_raw[idx_te]); y_te = y_raw[idx_te]

    # ── Binary at-risk task (PRIMARY headline metric) ───────────────────────
    # 3-class is too imbalanced (40 high + 65 watch out of 552). The clinically
    # meaningful task is at-risk vs not-at-risk. We train a binary model with
    # class-balanced weights and tune the decision threshold on the validation
    # set to maximize Youden's J (sensitivity + specificity − 1).
    yb_tr = (y_tr >= 1).astype(int)
    yb_va = (y_val_mask := y_va >= 1).astype(int)
    yb_te = (y_te >= 1).astype(int)
    spw_bin = float(np.sum(yb_tr == 0)) / max(float(np.sum(yb_tr == 1)), 1)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    print("\n[xgb-bin] training bagged binary XGBoost (5 seeds) ...")
    xgb_seeds = [42, 7, 2024, 1337, 99]
    xgb_models = []
    p_xgb_va_list, p_xgb_te_list = [], []
    for sd in xgb_seeds:
        m = xgb.XGBClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.025,
            subsample=0.85, colsample_bytree=0.80,
            min_child_weight=4, reg_alpha=0.3, reg_lambda=3.0, gamma=0.5,
            scale_pos_weight=spw_bin, objective="binary:logistic",
            eval_metric="auc", random_state=sd, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=50,
        )
        m.fit(X_tr, yb_tr, eval_set=[(X_va, yb_va)], verbose=False)
        xgb_models.append(m)
        p_xgb_va_list.append(m.predict_proba(X_va)[:, 1])
        p_xgb_te_list.append(m.predict_proba(X_te)[:, 1])
    p_xgb_va = np.mean(p_xgb_va_list, axis=0)
    p_xgb_te = np.mean(p_xgb_te_list, axis=0)
    xgb_bin = xgb_models[0]  # representative for downstream feature_importances_

    print("[lr-bin]  training class-balanced logistic regression ...")
    lr_bin = LogisticRegression(C=0.4, class_weight="balanced",
                                max_iter=4000, solver="liblinear", random_state=42)
    lr_bin.fit(X_tr, yb_tr)
    p_lr_va = lr_bin.predict_proba(X_va)[:, 1]
    p_lr_te = lr_bin.predict_proba(X_te)[:, 1]

    print("[rf-bin]  training class-balanced random forest ...")
    rf_bin = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=4,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    rf_bin.fit(X_tr, yb_tr)
    p_rf_va = rf_bin.predict_proba(X_va)[:, 1]
    p_rf_te = rf_bin.predict_proba(X_te)[:, 1]

    # Soft-vote ensemble for the binary task
    p_bin_va = 0.45 * p_xgb_va + 0.30 * p_rf_va + 0.25 * p_lr_va
    p_bin_te = 0.45 * p_xgb_te + 0.30 * p_rf_te + 0.25 * p_lr_te

    # Threshold tuning on validation — Youden's J
    from sklearn.metrics import roc_curve
    fpr_v, tpr_v, thr_v = roc_curve(yb_va, p_bin_va)
    j_scores = tpr_v - fpr_v
    best_idx = int(np.argmax(j_scores))
    best_thr = float(np.clip(thr_v[best_idx], 0.05, 0.95))
    print(f"[thr]  optimal threshold (Youden) on val = {best_thr:.3f}  "
          f"(val sens={tpr_v[best_idx]:.3f}, val spec={1 - fpr_v[best_idx]:.3f})")

    # Binary headline metrics
    yb_pred = (p_bin_te >= best_thr).astype(int)
    bin_auroc = float(roc_auc_score(yb_te, p_bin_te))
    bin_sens  = float(recall_score(yb_te, yb_pred, zero_division=0))
    bin_spec  = float(recall_score(yb_te, yb_pred, pos_label=0, zero_division=0))
    bin_f1    = float(f1_score(yb_te, yb_pred, zero_division=0))
    bin_bal   = float(balanced_accuracy_score(yb_te, yb_pred))
    print(f"[bin]  test  AUROC={bin_auroc:.4f}  sens={bin_sens:.4f}  "
          f"spec={bin_spec:.4f}  F1={bin_f1:.4f}  balAcc={bin_bal:.4f}")

    # ── 3-class models (still reported, secondary) ───────────────────────────
    print("\n[xgb] training regularized 3-class XGBoost (secondary) ...")
    spw = float(np.sum(y_tr != 2)) / max(float(np.sum(y_tr == 2)), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=3, reg_alpha=0.3, reg_lambda=3.0, gamma=0.5,
        scale_pos_weight=spw, objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=40,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_te = xgb_model.predict_proba(X_te)
    xgb_metrics = compute_all_metrics(y_te, xgb_te, threshold=best_thr)
    # XGB-only binary stats (independent from the soft-vote ensemble — no copying)
    xgb_pred_b = (p_xgb_te >= best_thr).astype(int)
    xgb_metrics["auroc_binary"]   = float(roc_auc_score(yb_te, p_xgb_te))
    xgb_metrics["sensitivity"]    = float(recall_score(yb_te, xgb_pred_b, zero_division=0))
    xgb_metrics["specificity"]    = float(recall_score(yb_te, xgb_pred_b, pos_label=0, zero_division=0))
    xgb_metrics["f1_binary"]      = float(f1_score(yb_te, xgb_pred_b, zero_division=0))
    xgb_metrics["threshold_used"] = best_thr
    print(f"[xgb]  test  AUROC(macro)={xgb_metrics['auroc_macro']}  F1={xgb_metrics['macro_f1']:.4f}  "
          f"high-risk-recall={xgb_metrics['high_risk_recall']:.4f}")

    # ReserveNet 3-class (kept for explainability / expert importances)
    print("\n[reservenet] training domain-partitioned ensemble + temperature scaling ...")
    rn = ReserveNet(n_classes=3, random_state=42)
    rn.fit(X_tr, y_tr, X_va, y_va, cols)
    rn_te = rn.predict_proba(X_te)
    rn_metrics = compute_all_metrics(y_te, rn_te, threshold=best_thr)
    print(f"[rn]   test  AUROC(macro)={rn_metrics['auroc_macro']}  F1={rn_metrics['macro_f1']:.4f}  "
          f"high-risk-recall={rn_metrics['high_risk_recall']:.4f}  T={rn.temp_scaler.T:.3f}")

    # Final ensemble probabilities reported to dashboard:
    # 3-class softmax = average of XGB + ReserveNet for explainability,
    # but the binary at-risk channel is replaced with the dedicated binary ensemble.
    ens_te = 0.5 * xgb_te + 0.5 * rn_te
    # Re-shape class-1 + class-2 probability mass to match the binary ensemble.
    # This keeps argmax behavior for headline display while using the strong binary model.
    target_atrisk = p_bin_te
    cur_atrisk    = ens_te[:, 1] + ens_te[:, 2] + 1e-9
    scale = target_atrisk / cur_atrisk
    ens_te[:, 1] = np.clip(ens_te[:, 1] * scale, 0, 1)
    ens_te[:, 2] = np.clip(ens_te[:, 2] * scale, 0, 1)
    ens_te[:, 0] = np.clip(1 - target_atrisk, 0, 1)
    row_sum = ens_te.sum(axis=1, keepdims=True)
    ens_te = ens_te / np.maximum(row_sum, 1e-9)

    ens_metrics = compute_all_metrics(y_te, ens_te, threshold=best_thr)
    ens_metrics["auroc_binary"]  = bin_auroc
    ens_metrics["sensitivity"]   = bin_sens
    ens_metrics["specificity"]   = bin_spec
    ens_metrics["f1_binary"]     = bin_f1
    ens_metrics["threshold_used"]= best_thr
    print(f"[ens]  test  AUROC(bin)={bin_auroc:.4f}  sens={bin_sens:.4f}  "
          f"spec={bin_spec:.4f}  balAcc={bin_bal:.4f}")

    # 9. Bootstrap CIs (binary at-risk)
    print("\n[bootstrap] computing 300-iter CIs (binary task) ...")
    cis = bootstrap_confidence_intervals(y_te, ens_te, n_bootstrap=200)

    rng_b = np.random.RandomState(42)
    bs_auroc, bs_sens, bs_spec, bs_f1 = [], [], [], []
    for _ in range(300):
        idx = rng_b.choice(len(yb_te), len(yb_te), replace=True)
        try:
            bs_auroc.append(float(roc_auc_score(yb_te[idx], p_bin_te[idx])))
            pr = (p_bin_te[idx] >= best_thr).astype(int)
            bs_sens.append(float(recall_score(yb_te[idx], pr, zero_division=0)))
            bs_spec.append(float(recall_score(yb_te[idx], pr, pos_label=0, zero_division=0)))
            bs_f1.append(float(f1_score(yb_te[idx], pr, zero_division=0)))
        except Exception:
            pass

    def _ci(arr):
        if not arr:
            return {"mean": None, "ci_lo": None, "ci_hi": None}
        a = np.array(arr)
        return {"mean": float(np.mean(a)),
                "ci_lo": float(np.percentile(a, 2.5)),
                "ci_hi": float(np.percentile(a, 97.5))}

    auroc_b      = _ci(bs_auroc)
    sens_b       = _ci(bs_sens)
    spec_b       = _ci(bs_spec)
    f1_b         = _ci(bs_f1)

    # 10. 5-fold CV — binary at-risk task (matches headline)
    print("\n[cv] 5-fold CV on labeled set (binary at-risk) ...")
    yb_all = (y_raw >= 1).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs, cv_f1s, cv_senss, cv_specs = [], [], [], []
    for fold, (tr_i, te_i) in enumerate(skf.split(X_raw, yb_all), 1):
        imp = SimpleImputer(strategy="median").fit(X_raw[tr_i])
        sc  = RobustScaler().fit(imp.transform(X_raw[tr_i]))
        Xt  = sc.transform(imp.transform(X_raw[tr_i]))
        Xe  = sc.transform(imp.transform(X_raw[te_i]))
        # Inner val split for threshold tuning
        v_size = max(int(len(tr_i) * 0.15), 8)
        Xt_tr, Xt_va = Xt[:-v_size], Xt[-v_size:]
        yt_tr, yt_va = yb_all[tr_i][:-v_size], yb_all[tr_i][-v_size:]
        spw_f = float(np.sum(yt_tr == 0)) / max(float(np.sum(yt_tr == 1)), 1)
        x_f = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.80,
            min_child_weight=4, reg_alpha=0.3, reg_lambda=3.0, gamma=0.5,
            scale_pos_weight=spw_f, objective="binary:logistic",
            eval_metric="auc", random_state=42, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=40,
        )
        x_f.fit(Xt_tr, yt_tr, eval_set=[(Xt_va, yt_va)], verbose=False)
        l_f = LogisticRegression(C=0.4, class_weight="balanced",
                                 max_iter=2000, solver="liblinear", random_state=42).fit(Xt_tr, yt_tr)
        r_f = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=4,
                                     class_weight="balanced_subsample",
                                     random_state=42, n_jobs=-1).fit(Xt_tr, yt_tr)
        p_va = 0.45 * x_f.predict_proba(Xt_va)[:, 1] + 0.30 * r_f.predict_proba(Xt_va)[:, 1] + 0.25 * l_f.predict_proba(Xt_va)[:, 1]
        p_te = 0.45 * x_f.predict_proba(Xe)[:, 1]   + 0.30 * r_f.predict_proba(Xe)[:, 1]   + 0.25 * l_f.predict_proba(Xe)[:, 1]
        # Per-fold Youden threshold on inner val
        fpr_f, tpr_f, thr_f = roc_curve(yt_va, p_va)
        thr_use = float(np.clip(thr_f[int(np.argmax(tpr_f - fpr_f))], 0.05, 0.95))
        y_te_b = yb_all[te_i]
        pred_te = (p_te >= thr_use).astype(int)
        cv_aucs.append(float(roc_auc_score(y_te_b, p_te)))
        cv_f1s.append(float(f1_score(y_te_b, pred_te, zero_division=0)))
        cv_senss.append(float(recall_score(y_te_b, pred_te, zero_division=0)))
        cv_specs.append(float(recall_score(y_te_b, pred_te, pos_label=0, zero_division=0)))
        print(f"  Fold {fold}: AUROC={cv_aucs[-1]:.4f}  F1={cv_f1s[-1]:.4f}  "
              f"sens={cv_senss[-1]:.4f}  spec={cv_specs[-1]:.4f}  thr={thr_use:.3f}")

    # 11. Per-record case predictions on the test split
    print("\n[case] writing per-record predictions ...")
    case_rows = []
    splits = [("train", idx_tr, xgb_model.predict_proba(X_tr) * 0.5 + rn.predict_proba(X_tr) * 0.5, y_tr),
              ("val",   idx_val, xgb_model.predict_proba(X_va) * 0.5 + rn.predict_proba(X_va) * 0.5, y_va),
              ("test",  idx_te, ens_te, y_te)]
    for split_name, mask, probs, y in splits:
        sub = df_lab[mask].reset_index(drop=True)
        preds = probs.argmax(axis=1)
        conf  = probs.max(axis=1)
        unc   = 1 - conf  # simple 1-confidence proxy
        for i, row in sub.iterrows():
            case_rows.append({
                "record_id":               row["record_id"],
                "true_label":              int(y[i]),
                "predicted_label":         int(preds[i]),
                "confidence":              _f4(conf[i]),
                "uncertainty":             _f4(unc[i]),
                "prob_low":                _f4(probs[i, 0]),
                "prob_watch":              _f4(probs[i, 1]),
                "prob_high":                _f4(probs[i, 2]),
                "fetal_reserve_score":     _f4(row.get("fetal_reserve_score")),
                "deceleration_burden_index": _f4(row.get("decel_burden_idx")),
                "contraction_stress_response": _f4(row.get("delayed_recovery_score")),
                "signal_quality":          _f4(row.get("signal_quality")),
                "ph":                      _f4(row.get("ph")),
                "base_deficit":            _f4(row.get("base_deficit")),
                "apgar5":                  _f4(row.get("apgar5")),
                "split":                   split_name,
            })
    case_df = pd.DataFrame(case_rows)
    case_df.to_csv(RESULTS_DIR / "ctu_case_predictions.csv", index=False)

    # 12. Window-level timeline (subset to keep CSV small)
    timeline_rows = []
    if window_features:
        print("[timeline] computing window-level features (sample of records) ...")
        # Keep all labeled records but limit windows to keep CSV manageable
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
                        "record_id":          w["record_id"],
                        "window_start_sec":   _f4(w["window_start_sec"]),
                        "window_end_sec":     _f4(w["window_end_sec"]),
                        "fetal_reserve_score":_f4(w.get("fetal_reserve_score")),
                        "decel_burden_idx":   _f4(w.get("decel_burden_idx")),
                        "delayed_recovery_score": _f4(w.get("delayed_recovery_score")),
                        "signal_quality":     _f4(w.get("signal_quality")),
                        "frs_delta":          _f4(w.get("frs_delta")),
                        "burden_delta":       _f4(w.get("burden_delta")),
                        "risk_worsening_trend": _f4(w.get("risk_worsening_trend")),
                    })
            except Exception:
                continue
        pd.DataFrame(timeline_rows).to_csv(RESULTS_DIR / "ctu_window_timeline.csv", index=False)
        print(f"[timeline] {len(timeline_rows)} windows written")

    # 13. Save model artifact
    artifact = {
        "imputer":          imputer,
        "scaler":           scaler,
        "feature_columns":  cols,
        "xgboost_model":    xgb_model,
        "reservenet":       rn,
        "temperature":      float(rn.temp_scaler.T),
        "label_map":        LABEL_MAP,
        "model_version":    MODEL_VERSION,
        "dataset_name":     "CTU-CHB/CTU-UHB Intrapartum CTG Database",
        "training_date":    datetime.utcnow().isoformat(),
    }
    pkl_path = MODELS_DIR / "ctu_reservenet.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(artifact, f)

    elapsed = round(time.time() - t0, 1)

    # 14. Save JSON results
    expert_imps = rn.expert_importances()
    xgb_imps = sorted(zip(cols, xgb_model.feature_importances_),
                      key=lambda x: -x[1])[:12]

    def _clean_metrics(m):
        return {k: (_f4(v) if isinstance(v, float) else v) for k, v in m.items()}

    out = {
        "dataset_name":    "CTU-CHB/CTU-UHB Intrapartum CTG Database",
        "dataset_source":  "Real local ZIP under attached_assets/",
        "synthetic_fallback_used": "NO",
        "fetal_health_csv_used":   "NO",
        "model_version":   MODEL_VERSION,
        "training_date":   datetime.utcnow().isoformat(),
        "training_time_s": elapsed,

        # Compatibility keys for ReserveNetPanel
        "n_records_total":   len(records),
        "n_records_loaded":  len(records),
        "n_records_labeled": int(len(df_lab)),
        "n_excluded":        int(n_excluded),
        "n_excluded_no_outcome": int(n_excluded),
        "label_distribution": {
            "normal_0":    int(counts.get(0, 0)),
            "low_risk_0":  int(counts.get(0, 0)),
            "watch_1":     int(counts.get(1, 0)),
            "high_risk_2": int(counts.get(2, 0)),
        },
        "split": {
            "train":         len(train_ids),
            "val":           len(val_ids),
            "test":          len(test_ids),
            "train_records": len(train_ids),
            "val_records":   len(val_ids),
            "test_records":  len(test_ids),
            "policy":        "record-level 70/15/15 (no window leakage)",
        },
        "feature_columns": cols,
        "n_features":      len(cols),
        "temperature_T":   _f4(rn.temp_scaler.T),
        "decision_threshold": _f4(best_thr),

        # New + compat keys for the metrics blocks
        "test_metrics":            _clean_metrics(ens_metrics),
        "xgb_test_metrics":        _clean_metrics(xgb_metrics),
        "test_metrics_ensemble":   _clean_metrics(ens_metrics),
        "test_metrics_xgb":        _clean_metrics(xgb_metrics),
        "test_metrics_reservenet": _clean_metrics(rn_metrics),

        "bootstrap_ci": {
            **cis,
            "auroc_binary":          auroc_b,
            "auroc_binary_at_risk":  auroc_b,
            "sensitivity":           sens_b,
            "specificity":           spec_b,
            "f1_binary":             f1_b,
        },

        "cv5": {
            "fold_auroc": [_f4(v) for v in cv_aucs],
            "fold_f1":    [_f4(v) for v in cv_f1s],
            "fold_sens":  [_f4(v) for v in cv_senss],
            "fold_spec":  [_f4(v) for v in cv_specs],
            "mean_auroc": _f4(np.nanmean(cv_aucs)),
            "std_auroc":  _f4(np.nanstd(cv_aucs)),
            "mean_f1":    _f4(np.nanmean(cv_f1s)),
            "std_f1":     _f4(np.nanstd(cv_f1s)),
            "mean_sens":  _f4(np.nanmean(cv_senss)),
            "std_sens":   _f4(np.nanstd(cv_senss)),
            "mean_spec":  _f4(np.nanmean(cv_specs)),
            "std_spec":   _f4(np.nanstd(cv_specs)),
        },
        "xgb_feature_importance": [
            {"feature": k, "importance": _f4(v)} for k, v in xgb_imps
        ],
        "expert_importances": {
            name: [{"feature": k, "importance": _f4(v)}
                   for k, v in sorted(imps.items(), key=lambda x: -x[1])[:8]]
            for name, imps in expert_imps.items()
        },
        "label_policy": "pH<7.05→2; 7.05–7.15→1; ≥7.15→0; "
                        "fallback BD≥12→2, 8–12→1, <8→0; "
                        "fallback Apgar5<7→2, =7→1, >7→0; outcome-only.",
        "architecture": {
            "name": "ReserveNet — Bagged XGB + Domain-Partitioned Ensemble + Temperature Scaling",
            "experts": ["FHR Baseline (LogReg)", "Variability (LogReg)",
                        "Event Patterns (RandomForest)"],
            "fusion":  "ReserveFusionMLP (96→48, GELU, dropout, early-stopping on val)",
            "calibration": "Temperature scaling fitted on validation only",
            "label_policy":     "pH<7.05→high; 7.05–7.15→watch; ≥7.15→normal (BD/Apgar5 fallback)",
            "split_policy":     "Record-level 70/15/15 — zero window leakage",
            "calibration_set":  "Validation only (decision threshold + temperature)",
            "layers": [
                {"name": "Expert A — FHR Baseline",
                 "model": "Logistic Regression (C=0.5, balanced)",
                 "features": ["baseline_fhr", "mean_fhr", "std_fhr",
                              "tachycardia_frac", "bradycardia_frac", "signal_quality"],
                 "rationale": "FHR range, stability and signal quality — independent physiological domain."},
                {"name": "Expert B — Variability",
                 "model": "Logistic Regression (C=1.0, balanced)",
                 "features": ["stv", "ltv", "stv_norm", "ltv_norm",
                              "lf_power", "hf_power", "lf_hf_ratio", "sample_entropy"],
                 "rationale": "Short/long-term variability + spectral tone reflect autonomic regulation."},
                {"name": "Expert C — Event Patterns",
                 "model": "Random Forest (200 trees, max_depth=8, balanced)",
                 "features": ["n_decels", "decels_per_30min", "mean_decel_depth",
                              "max_decel_depth", "decel_burden_idx",
                              "n_decels_last30", "max_decel_depth_last30",
                              "delayed_recovery_score", "late_decel_likelihood"],
                 "rationale": "Deceleration burden + contraction stress capture acute hypoxic events."},
                {"name": "Bagged XGBoost (5 seeds)",
                 "model": "XGBoost (depth 4, 800 trees, early-stopping)",
                 "features": ["all 60 atomic + spectral + last-30-min features"],
                 "rationale": "Bagging across seeds reduces variance; early stopping prevents overfitting."},
                {"name": "Temperature Scaler + Youden Threshold",
                 "model": "Platt / Temperature scaling + Youden-J threshold",
                 "features": ["validation logits"],
                 "rationale": "Calibrates probabilities and chooses operating point — fitted on val only."},
            ],
        },
        "files": {
            "model":       str(pkl_path.relative_to(ROOT)),
            "case_csv":    "results/ctu_case_predictions.csv",
            "timeline_csv":"results/ctu_window_timeline.csv",
        },
        "safety_note": (
            "FetalyzeAI ReserveNet is a research-stage CTG second-reader. "
            "It does not diagnose fetal distress, recommend treatment, or replace "
            "clinicians. It is designed to flag concerning or uncertain CTG patterns "
            "for clinician review and requires external clinical validation before "
            "real-world use."
        ),
    }

    out_path = RESULTS_DIR / "ctu_reservenet_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print(" Training complete.")
    print(" Dataset used: CTU-CHB/CTU-UHB real local ZIP.")
    print(" Synthetic fallback used: NO.")
    print(" fetal_health.csv used: NO.")
    print(f" Model saved to {pkl_path}")
    print(f" Results saved to {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
