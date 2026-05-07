"""
FetalyzeAI ReserveNet — Primary Training Script
================================================
Trains on the real CTU-CHB/CTU-UHB dataset (552 recordings).

Pipeline:
  1. Extract uploaded CTU zip from attached_assets/
  2. Load FHR + UC signals via WFDB
  3. Extract record-level CTG features (no synthetic fallback)
  4. Assign labels ONLY from clinical outcomes (pH / base_deficit / Apgar)
  5. Record-level 70/15/15 split (no window leakage)
  6. Fit imputer / scaler on training records only
  7. Train ReserveNet (domain-partitioned stacked ensemble + temperature scaling)
  8. Calibrate on validation set (never on test)
  9. Evaluate ONCE on held-out test set
  10. Save results/ctu_reservenet_results.json

Run:
  python scripts/train_reservenet_ctu.py
"""

import sys, warnings, json, time
from pathlib import Path

warnings.filterwarnings("ignore")

# ── make scripts/ importable ────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score, balanced_accuracy_score

import xgboost as xgb

from metrics_utils import compute_all_metrics, bootstrap_metric
from reservenet_model import ReserveNet

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
ZIP_GLOB   = list(ROOT.glob("attached_assets/ctu-chb-intrapartum*.zip"))
ZIP_PATH   = ZIP_GLOB[0] if ZIP_GLOB else None
EXTRACT_DIR = ROOT / "data" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

FS = 4   # Hz

# ── STEP 1 — extract zip ─────────────────────────────────────────────────────

def ensure_extracted() -> Path:
    if EXTRACT_DIR.exists() and len(list(EXTRACT_DIR.glob("*.hea"))) > 100:
        print(f"[zip] already extracted → {EXTRACT_DIR}")
        return EXTRACT_DIR
    if ZIP_PATH is None or not ZIP_PATH.exists():
        raise RuntimeError(
            "CTU-CHB zip not found in attached_assets/. "
            "Upload ctu-chb-intrapartum-cardiotocography-database-1.0.0.zip first."
        )
    import zipfile
    print(f"[zip] extracting {ZIP_PATH.name} ...")
    EXTRACT_DIR.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(EXTRACT_DIR.parent)
    n = len(list(EXTRACT_DIR.glob("*.hea")))
    print(f"[zip] extracted {n} .hea files")
    return EXTRACT_DIR


# ── STEP 2 — load signals ─────────────────────────────────────────────────────

def load_signals(data_dir: Path, verbose: bool = True):
    import wfdb
    hea_files = sorted(data_dir.glob("*.hea"))
    print(f"[load] {len(hea_files)} records found")

    records = []
    skipped = 0
    for hf in hea_files:
        rid = hf.stem
        try:
            rec = wfdb.rdrecord(str(data_dir / rid))
            snames = [s.upper() for s in rec.sig_name]
            fi = next((i for i, s in enumerate(snames) if "FHR" in s), None)
            ui = next((i for i, s in enumerate(snames) if any(t in s for t in ("UC","TOCO","TOC"))), None)
            if fi is None:
                skipped += 1; continue

            fhr = rec.p_signal[:, fi].astype(float)
            uc  = rec.p_signal[:, ui].astype(float) if ui is not None else np.full_like(fhr, np.nan)
            fhr[(fhr <= 0) | (fhr > 300)] = np.nan
            uc[uc < 0] = np.nan

            meta = _parse_header(rec)
            ph = float(meta.get("ph", np.nan))
            records.append({
                "record_id":    rid,
                "fhr":          fhr,
                "uc":           uc,
                "ph":           ph,
                "base_deficit": float(meta.get("base_deficit", np.nan)),
                "apgar1":       float(meta.get("apgar1", np.nan)),
                "apgar5":       float(meta.get("apgar5", np.nan)),
                "gest_weeks":   float(meta.get("gestational_age", np.nan)),
                "birth_weight": float(meta.get("birth_weight", np.nan)),
                "delivery_type":str(meta.get("delivery_type", "unknown")),
                "duration_min": len(fhr) / (FS * 60),
                "signal_quality": float(np.mean(~np.isnan(fhr))),
            })
        except Exception:
            skipped += 1

    print(f"[load] loaded {len(records)}, skipped {skipped}")
    return records


def _parse_header(rec) -> dict:
    info = {}
    KEY_MAP = {
        "pH": "ph", "BDecf": "base_deficit", "BE": "base_deficit",
        "Apgar1": "apgar1", "Apgar5": "apgar5",
        "Gest.": "gestational_age", "Weight(g)": "birth_weight",
        "Deliv.": "delivery_type",
    }
    for comment in (rec.comments or []):
        parts = comment.split()
        for n_tok in (1, 2):
            key = " ".join(parts[:n_tok]).rstrip(":.,")
            if key in KEY_MAP and len(parts) > n_tok:
                try:
                    info[KEY_MAP[key]] = float(parts[n_tok])
                except ValueError:
                    info[KEY_MAP[key]] = parts[n_tok]
                break
    return info


# ── STEP 3 — extract features ─────────────────────────────────────────────────

def _valid(arr): return arr[~np.isnan(arr)]

def extract_features(rec: dict) -> dict:
    fhr = rec["fhr"]
    uc  = rec["uc"]
    v   = _valid(fhr)

    # Baseline
    baseline = float(np.nanmedian(v)) if len(v) >= 10 else np.nan
    std_fhr  = float(np.std(v)) if len(v) >= 10 else np.nan
    mean_fhr = float(np.mean(v)) if len(v) >= 4 else np.nan

    # STV / LTV
    stv = float(np.mean(np.abs(np.diff(v)))) if len(v) >= 2 else np.nan
    ltv = np.nan
    epoch = int(60 * FS)
    if len(v) >= epoch:
        ranges = [v[i:i+epoch].max() - v[i:i+epoch].min()
                  for i in range(0, len(v) - epoch, epoch)]
        ltv = float(np.mean(ranges)) if ranges else np.nan

    # Tachycardia / bradycardia fractions
    tach = float(np.mean(v > 160)) if len(v) else np.nan
    brad = float(np.mean(v < 110)) if len(v) else np.nan

    # Deceleration detection (simple threshold: >15 bpm below baseline for >15s)
    n_decels = 0
    decel_depths = []
    decel_durs   = []
    decel_burden = 0.0
    if not np.isnan(baseline) and len(v) >= 60:
        fhr_i = np.where(~np.isnan(fhr), fhr, baseline)
        below = (fhr_i < (baseline - 15)).astype(int)
        diffs = np.diff(below, prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends   = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            dur = (e - s) / FS
            if dur >= 15:
                depth = baseline - fhr_i[s:e].min()
                n_decels += 1
                decel_depths.append(depth)
                decel_durs.append(dur)
                decel_burden += depth * dur

    # Acceleration detection (>15 bpm above baseline for >15s)
    n_accels = 0
    if not np.isnan(baseline) and len(v) >= 60:
        fhr_i = np.where(~np.isnan(fhr), fhr, baseline)
        above = (fhr_i > (baseline + 15)).astype(int)
        diffs = np.diff(above, prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends   = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) / FS >= 15:
                n_accels += 1

    dur_min  = rec["duration_min"]
    dur_30   = max(dur_min / 30, 0.001)
    dur_10   = max(dur_min / 10, 0.001)

    # Uterine contractions (UC > 30 mmHg for >30s)
    n_contractions = 0
    csr_frac = np.nan
    uv = _valid(uc)
    if len(uv) >= 120:
        uc_thresh = 30
        uc_i = np.where(~np.isnan(uc), uc, 0)
        above_uc = (uc_i > uc_thresh).astype(int)
        diffs = np.diff(above_uc, prepend=0, append=0)
        uc_starts = np.where(diffs == 1)[0]
        uc_ends   = np.where(diffs == -1)[0]
        for s, e in zip(uc_starts, uc_ends):
            if (e - s) / FS >= 30:
                n_contractions += 1
        if n_contractions > 0:
            # fraction of contractions followed by a deceleration within 60s
            responded = 0
            if not np.isnan(baseline):
                fhr_i = np.where(~np.isnan(fhr), fhr, baseline)
                for cs, ce in zip(uc_starts, uc_ends):
                    if (ce - cs) / FS < 30:
                        continue
                    window_end = min(ce + 60 * FS, len(fhr_i))
                    segment = fhr_i[ce:int(window_end)]
                    if len(segment) > 0 and segment.min() < (baseline - 15):
                        responded += 1
            csr_frac = responded / n_contractions

    # Fetal reserve score (simplified inline version for self-containment)
    frs = 50.0
    if not np.isnan(baseline):
        frs += 20 if 110 <= baseline <= 160 else (10 if 100 <= baseline < 110 or 160 < baseline <= 170 else 0)
    if not np.isnan(stv):
        frs += 20 if 5 <= stv <= 25 else (10 if 3 <= stv < 5 else 0)
    if not np.isnan(ltv):
        frs += 15 if 10 <= ltv <= 40 else (7 if 5 <= ltv < 10 else 0)
    if not np.isnan(baseline) and dur_min > 0:
        acc_rate = n_accels / (dur_min / 30)
        frs += 15 if acc_rate >= 2 else (8 if acc_rate >= 1 else 0)
    decel_pen = min(20, decel_burden * 0.001 + n_decels * 2)
    frs = float(np.clip(frs - decel_pen - 50, 0, 100))

    # Normalised variability (for variability expert)
    stv_norm = stv / 10.0 if not np.isnan(stv) else np.nan
    ltv_norm = ltv / 25.0 if not np.isnan(ltv) else np.nan

    return {
        "record_id":           rec["record_id"],
        "baseline_fhr":        baseline,
        "mean_fhr":            mean_fhr,
        "std_fhr":             std_fhr,
        "stv":                 stv,
        "ltv":                 ltv,
        "stv_norm":            stv_norm,
        "ltv_norm":            ltv_norm,
        "tachycardia_frac":    tach,
        "bradycardia_frac":    brad,
        "n_decels":            float(n_decels),
        "decels_per_30min":    n_decels / dur_30,
        "mean_decel_depth":    float(np.mean(decel_depths)) if decel_depths else 0.0,
        "max_decel_depth":     float(max(decel_depths)) if decel_depths else 0.0,
        "mean_decel_dur_s":    float(np.mean(decel_durs)) if decel_durs else 0.0,
        "n_accels":            float(n_accels),
        "accels_per_30min":    n_accels / dur_30,
        "n_contractions":      float(n_contractions),
        "contractions_per_10min": n_contractions / dur_10,
        "decel_burden":        decel_burden,
        "csr_frac":            csr_frac,
        "fetal_reserve_score": frs,
        "duration_min":        dur_min,
        "signal_quality":      rec["signal_quality"],
        "missing_fhr":         float(np.mean(np.isnan(rec["fhr"]))),
        "ph":                  rec["ph"],
        "base_deficit":        rec["base_deficit"],
        "apgar1":              rec["apgar1"],
        "apgar5":              rec["apgar5"],
    }


# ── STEP 4 — label assignment (clinical outcomes ONLY) ───────────────────────

def assign_risk_label(row) -> int:
    """
    Returns: 0=normal, 1=watch, 2=high_risk, -1=unknown (exclude)
    Labels come ONLY from pH / base_deficit / Apgar — never from features.
    """
    ph  = row.get("ph", np.nan)
    bd  = row.get("base_deficit", np.nan)
    a1  = row.get("apgar1", np.nan)

    if not np.isnan(ph):
        if ph < 7.05:  return 2
        if ph < 7.15:  return 1
        return 0

    if not np.isnan(bd):
        if bd >= 12:   return 2
        if bd >= 8:    return 1
        return 0

    if not np.isnan(a1):
        if a1 < 4:     return 2
        if a1 < 7:     return 1
        return 0

    return -1   # unknown — excluded from training


# ── STEP 5 — record-level split ──────────────────────────────────────────────

def record_level_split(feat_df: pd.DataFrame, test_frac=0.15, val_frac=0.15, seed=42):
    rids = feat_df["record_id"].values
    unique_rids = np.unique(rids)

    labels_per_rid = feat_df.groupby("record_id")["risk_label"].first().reindex(unique_rids).values

    train_val, test = train_test_split(
        unique_rids, test_size=test_frac, stratify=labels_per_rid, random_state=seed
    )
    tv_labels = feat_df.groupby("record_id")["risk_label"].first().reindex(train_val).values
    train, val = train_test_split(
        train_val, test_size=val_frac / (1 - test_frac), stratify=tv_labels, random_state=seed
    )

    idx_train = feat_df["record_id"].isin(train)
    idx_val   = feat_df["record_id"].isin(val)
    idx_test  = feat_df["record_id"].isin(test)
    return idx_train, idx_val, idx_test


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("\n" + "="*60)
    print(" FetalyzeAI ReserveNet — Training Pipeline")
    print("="*60)

    # 1. Extract
    data_dir = ensure_extracted()

    # 2. Load signals
    records = load_signals(data_dir)
    if len(records) == 0:
        raise RuntimeError("No records loaded. Check the CTU zip.")

    # 3. Extract features
    print(f"\n[features] extracting from {len(records)} records...")
    feats = [extract_features(r) for r in records]
    df = pd.DataFrame(feats)
    print(f"[features] done — shape {df.shape}")

    # 4. Assign labels
    df["risk_label"] = df.apply(assign_risk_label, axis=1)
    df_labeled = df[df["risk_label"] >= 0].copy()
    n_excluded = len(df) - len(df_labeled)
    label_counts = df_labeled["risk_label"].value_counts().sort_index().to_dict()
    print(f"\n[labels] {len(df_labeled)} records with known outcomes ({n_excluded} excluded)")
    print(f"         Normal=0: {label_counts.get(0,0)}  Watch=1: {label_counts.get(1,0)}  High=2: {label_counts.get(2,0)}")

    if len(df_labeled) < 50:
        raise RuntimeError("Too few labeled records for training.")

    # 5. Feature matrix
    FEATURE_COLS = [
        "baseline_fhr", "mean_fhr", "std_fhr",
        "stv", "ltv", "stv_norm", "ltv_norm",
        "tachycardia_frac", "bradycardia_frac",
        "n_decels", "decels_per_30min",
        "mean_decel_depth", "max_decel_depth", "mean_decel_dur_s",
        "n_accels", "accels_per_30min",
        "n_contractions", "contractions_per_10min",
        "decel_burden", "csr_frac",
        "fetal_reserve_score", "duration_min", "signal_quality", "missing_fhr",
    ]
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df_labeled.columns]

    X_raw = df_labeled[FEATURE_COLS].values.astype(float)
    y_raw = df_labeled["risk_label"].values.astype(int)

    # 6. Record-level split
    idx_tr, idx_val, idx_te = record_level_split(df_labeled)
    X_tr_raw = X_raw[idx_tr]; y_tr = y_raw[idx_tr]
    X_va_raw = X_raw[idx_val]; y_va = y_raw[idx_val]
    X_te_raw = X_raw[idx_te]; y_te = y_raw[idx_te]

    # Impute / scale fit on train only
    imp = SimpleImputer(strategy="median")
    sc  = RobustScaler()
    X_tr = sc.fit_transform(imp.fit_transform(X_tr_raw))
    X_va = sc.transform(imp.transform(X_va_raw))
    X_te = sc.transform(imp.transform(X_te_raw))

    print(f"\n[split] train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")

    # 7a. Train XGBoost baseline
    print("\n[xgb] training XGBoost baseline ...")
    spw = float(np.sum(y_tr != 2)) / max(float(np.sum(y_tr == 2)), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=8, reg_alpha=1.0, reg_lambda=5.0,
        gamma=2.0, scale_pos_weight=spw,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, tree_method="hist",
    )
    xgb_model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  verbose=False)
    xgb_probs_va = xgb_model.predict_proba(X_va)
    xgb_probs_te = xgb_model.predict_proba(X_te)
    xgb_metrics  = compute_all_metrics(y_te, xgb_probs_te)
    print(f"[xgb] test AUROC={xgb_metrics.get('auroc_macro','n/a'):.4f}  "
          f"macro-F1={xgb_metrics['macro_f1']:.4f}  "
          f"high-risk-recall={xgb_metrics['high_risk_recall']:.4f}")

    # 7b. Train ReserveNet
    print("\n[reservenet] training domain-partitioned ensemble ...")
    model = ReserveNet(n_classes=3, random_state=42)
    model.fit(X_tr, y_tr, X_va, y_va, FEATURE_COLS)
    rn_probs_te = model.predict_proba(X_te)
    rn_metrics  = compute_all_metrics(y_te, rn_probs_te)
    print(f"[reservenet] test AUROC={rn_metrics.get('auroc_macro','n/a'):.4f}  "
          f"macro-F1={rn_metrics['macro_f1']:.4f}  "
          f"high-risk-recall={rn_metrics['high_risk_recall']:.4f}  "
          f"sensitivity={rn_metrics['sensitivity']:.4f}")
    print(f"[reservenet] Temperature T={model.temp_scaler.T:.3f}")

    # 7c. Stacked ensemble: average XGB + ReserveNet
    ens_probs_te = 0.5 * xgb_probs_te + 0.5 * rn_probs_te
    ens_metrics  = compute_all_metrics(y_te, ens_probs_te)
    print(f"\n[ensemble] test AUROC={ens_metrics.get('auroc_macro','n/a'):.4f}  "
          f"macro-F1={ens_metrics['macro_f1']:.4f}  "
          f"high-risk-recall={ens_metrics['high_risk_recall']:.4f}  "
          f"sensitivity={ens_metrics['sensitivity']:.4f}")

    # 8. Bootstrap CIs on ensemble test metrics
    print("\n[bootstrap] computing confidence intervals (200 iterations)...")
    auroc_boot = bootstrap_metric(
        y_te, ens_probs_te,
        lambda y, p: roc_auc_score(
            (y >= 1).astype(int), p[:, 1] + p[:, 2]
        ),
        n_boot=200,
    )
    sens_boot = bootstrap_metric(
        y_te, ens_probs_te,
        lambda y, p: recall_score(
            (y >= 1).astype(int), ((p[:, 1] + p[:, 2]) >= 0.35).astype(int),
            zero_division=0
        ),
        n_boot=200,
    )

    # 9. Feature importances
    expert_imps = model.expert_importances()
    xgb_imps_raw = dict(zip(FEATURE_COLS, xgb_model.feature_importances_))
    xgb_imps = sorted(xgb_imps_raw.items(), key=lambda x: -x[1])[:12]

    # 10. 5-fold CV on full labeled set (for Research tab)
    print("\n[cv] 5-fold CV on labeled data ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs, cv_f1s, cv_senss, cv_specs = [], [], [], []
    for fold, (tr_i, te_i) in enumerate(skf.split(X_raw, y_raw), 1):
        imp2 = SimpleImputer(strategy="median"); sc2 = RobustScaler()
        Xt = sc2.fit_transform(imp2.fit_transform(X_raw[tr_i]))
        Xe = sc2.transform(imp2.transform(X_raw[te_i]))
        m = ReserveNet(n_classes=3, random_state=42)
        val_size = max(int(len(tr_i) * 0.15), 5)
        Xv, yv = Xt[-val_size:], y_raw[tr_i][-val_size:]
        Xt2, yt2 = Xt[:-val_size], y_raw[tr_i][:-val_size]
        m.fit(Xt2, yt2, Xv, yv, FEATURE_COLS)
        probs = m.predict_proba(Xe)
        met = compute_all_metrics(y_raw[te_i], probs)
        cv_aucs.append(met.get("auroc_macro", np.nan))
        cv_f1s.append(met["macro_f1"])
        cv_senss.append(met["sensitivity"])
        cv_specs.append(met["specificity"])
        print(f"  Fold {fold}: AUROC={cv_aucs[-1]:.4f}  F1={cv_f1s[-1]:.4f}  Sens={cv_senss[-1]:.4f}")

    elapsed = round(time.time() - t0, 1)

    # ── Save results ────────────────────────────────────────────────────────

    def f4(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return round(float(v), 4)

    out = {
        "model_name":    "FetalyzeAI ReserveNet v1",
        "dataset":       "CTU-CHB/CTU-UHB Intrapartum CTG Database",
        "n_records_total":   len(records),
        "n_records_labeled": int(len(df_labeled)),
        "n_excluded":        int(n_excluded),
        "label_distribution": {
            "normal_0":   int(label_counts.get(0, 0)),
            "watch_1":    int(label_counts.get(1, 0)),
            "high_risk_2":int(label_counts.get(2, 0)),
        },
        "split": {
            "train":      int(np.sum(idx_tr)),
            "val":        int(np.sum(idx_val)),
            "test":       int(np.sum(idx_te)),
            "strategy":   "record-level (no window leakage)",
        },
        "training_time_s": elapsed,
        "temperature_T":   f4(model.temp_scaler.T),
        "n_features":      len(FEATURE_COLS),
        "feature_cols":    FEATURE_COLS,

        # Final ensemble test metrics
        "test_metrics": {k: f4(v) if isinstance(v, float) else v
                         for k, v in ens_metrics.items()},

        # Per-model test metrics for comparison
        "xgb_test_metrics": {k: f4(v) if isinstance(v, float) else v
                              for k, v in xgb_metrics.items()},
        "reservenet_test_metrics": {k: f4(v) if isinstance(v, float) else v
                                    for k, v in rn_metrics.items()},

        # Bootstrap CIs
        "bootstrap_ci": {
            "auroc_binary": auroc_boot,
            "sensitivity":  sens_boot,
        },

        # 5-fold CV
        "cv5": {
            "fold_auroc":  [f4(v) for v in cv_aucs],
            "fold_f1":     [f4(v) for v in cv_f1s],
            "fold_sens":   [f4(v) for v in cv_senss],
            "fold_spec":   [f4(v) for v in cv_specs],
            "mean_auroc":  f4(np.nanmean(cv_aucs)),
            "std_auroc":   f4(np.nanstd(cv_aucs)),
            "mean_f1":     f4(np.nanmean(cv_f1s)),
            "std_f1":      f4(np.nanstd(cv_f1s)),
            "mean_sens":   f4(np.nanmean(cv_senss)),
            "std_sens":    f4(np.nanstd(cv_senss)),
            "mean_spec":   f4(np.nanmean(cv_specs)),
            "std_spec":    f4(np.nanstd(cv_specs)),
        },

        # Feature importances
        "xgb_feature_importance": [
            {"feature": k, "importance": f4(v)} for k, v in xgb_imps
        ],
        "expert_importances": {
            name: [{"feature": k, "importance": f4(v)}
                   for k, v in sorted(imps.items(), key=lambda x: -x[1])[:8]]
            for name, imps in expert_imps.items()
        },

        # Architecture description
        "architecture": {
            "name": "ReserveNet — Domain-Partitioned Stacked Ensemble",
            "layers": [
                {
                    "name": "Expert A — FHR Baseline",
                    "model": "Logistic Regression (C=0.5, balanced)",
                    "features": ["baseline_fhr", "mean_fhr", "std_fhr",
                                 "tachycardia_frac", "bradycardia_frac",
                                 "missing_fhr", "signal_quality"],
                    "rationale": "Captures FHR range, stability, and signal quality — independent physiological domain.",
                },
                {
                    "name": "Expert B — Variability",
                    "model": "Logistic Regression (C=1.0, balanced)",
                    "features": ["stv", "ltv", "stv_norm", "ltv_norm"],
                    "rationale": "Short- and long-term variability reflect autonomic nervous system tone.",
                },
                {
                    "name": "Expert C — Event Patterns",
                    "model": "Random Forest (200 trees, max_depth=5, balanced)",
                    "features": ["n_decels", "decels_per_30min", "mean_decel_depth",
                                 "max_decel_depth", "mean_decel_dur_s", "n_accels",
                                 "accels_per_30min", "n_contractions",
                                 "contractions_per_10min", "decel_burden", "csr_frac"],
                    "rationale": "Deceleration burden, contraction stress response, and reactive patterns capture acute hypoxic stress.",
                },
                {
                    "name": "Meta-Learner — ReserveFusionMLP",
                    "model": "MLP (96→48, ReLU, L2=0.05, early stopping on val)",
                    "features": ["expert_A_probs", "expert_B_probs", "expert_C_probs", "all_raw_features"],
                    "rationale": "Learns which expert to trust per case — gated clinical fusion.",
                },
                {
                    "name": "Temperature Scaler",
                    "model": "Platt / Temperature Scaling (fit on validation only)",
                    "features": ["meta_logits"],
                    "rationale": "Calibrates probabilities for clinical use. T fitted on validation; never on test.",
                },
            ],
            "label_policy": "pH < 7.05 → high_risk; 7.05–7.15 → watch; ≥ 7.15 → normal; no feature-derived labels",
            "split_policy": "Record-level 70/15/15 — zero window leakage across patients",
            "calibration_set": "validation (not test)",
        },
    }

    out_path = RESULTS_DIR / "ctu_reservenet_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}")
    print(f" Results saved → {out_path}")
    print(f" Training time: {elapsed}s")
    print(f"\n FINAL TEST METRICS (ensemble):")
    print(f"   AUROC (binary):  {ens_metrics['auroc_binary']:.4f}  CI [{auroc_boot['ci_lo']:.3f}, {auroc_boot['ci_hi']:.3f}]")
    print(f"   Sensitivity:     {ens_metrics['sensitivity']:.4f}  CI [{sens_boot['ci_lo']:.3f}, {sens_boot['ci_hi']:.3f}]")
    print(f"   Specificity:     {ens_metrics['specificity']:.4f}")
    print(f"   Macro F1:        {ens_metrics['macro_f1']:.4f}")
    print(f"   High-risk recall:{ens_metrics['high_risk_recall']:.4f}")
    print(f"   ECE:             {ens_metrics['ece']:.4f}")
    print(f"   Temperature T:   {model.temp_scaler.T:.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
