"""
Export FetalyzeAI ReserveNet inference parameters for TypeScript.

Trains a fully-portable variant of ReserveNet where ALL experts are
LogisticRegression (no RandomForest), so the entire forward pass
can be serialised and run in TypeScript without loss of fidelity.

Pipeline (all sklearn, fully exportable):
  1. Expert A  — LR on baseline features (7 feats, C=0.5, balanced)
  2. Expert B  — LR on variability features (4 feats, C=1.0, balanced)
  3. Expert C  — LR on event features (12 feats, C=0.5, balanced)
  4. Meta-MLP  — trained on [9 LR expert probs + 24 raw scaled feats]
  5. Temperature scaling on validation set

Output: results/reservenet_inference.json  (~50-60 KB)
"""

import sys, warnings, json
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
ROOT = SCRIPTS_DIR.parent

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from reservenet_model import FEATURE_GROUPS, TemperatureScaler

# ── data pipeline (identical to train_reservenet_ctu.py) ─────────────────────
FS = 4

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


def _valid(arr): return arr[~np.isnan(arr)]


def _parse_header(rec) -> dict:
    info = {}
    KEY_MAP = {
        "pH": "ph", "BDecf": "base_deficit", "BE": "base_deficit",
        "Apgar1": "apgar1", "Apgar5": "apgar5",
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


def load_signals(data_dir: Path):
    import wfdb
    hea_files = sorted(data_dir.glob("*.hea"))
    print(f"[load] {len(hea_files)} records found")
    records = []; skipped = 0
    for hf in hea_files:
        rid = hf.stem
        try:
            rec = wfdb.rdrecord(str(data_dir / rid))
            snames = [s.upper() for s in rec.sig_name]
            fi = next((i for i, s in enumerate(snames) if "FHR" in s), None)
            ui = next((i for i, s in enumerate(snames) if any(t in s for t in ("UC","TOCO","TOC"))), None)
            if fi is None: skipped += 1; continue
            fhr = rec.p_signal[:, fi].astype(float)
            uc  = rec.p_signal[:, ui].astype(float) if ui is not None else np.full_like(fhr, np.nan)
            fhr[(fhr <= 0)|(fhr > 300)] = np.nan
            uc[uc < 0] = np.nan
            meta = _parse_header(rec)
            records.append({
                "record_id": rid, "fhr": fhr, "uc": uc,
                "ph": float(meta.get("ph", np.nan)),
                "base_deficit": float(meta.get("base_deficit", np.nan)),
                "apgar1": float(meta.get("apgar1", np.nan)),
                "apgar5": float(meta.get("apgar5", np.nan)),
                "duration_min": len(fhr) / (FS * 60),
                "signal_quality": float(np.mean(~np.isnan(fhr))),
            })
        except Exception: skipped += 1
    print(f"[load] loaded {len(records)}, skipped {skipped}")
    return records


def extract_features(rec: dict) -> dict:
    fhr = rec["fhr"]; uc = rec["uc"]; v = _valid(fhr)
    baseline = float(np.nanmedian(v)) if len(v) >= 10 else np.nan
    std_fhr  = float(np.std(v)) if len(v) >= 10 else np.nan
    mean_fhr = float(np.mean(v)) if len(v) >= 4 else np.nan
    stv = float(np.mean(np.abs(np.diff(v)))) if len(v) >= 2 else np.nan
    ltv = np.nan
    epoch = int(60 * FS)
    if len(v) >= epoch:
        ranges = [v[i:i+epoch].max()-v[i:i+epoch].min() for i in range(0,len(v)-epoch,epoch)]
        ltv = float(np.mean(ranges)) if ranges else np.nan
    tach = float(np.mean(v > 160)) if len(v) else np.nan
    brad = float(np.mean(v < 110)) if len(v) else np.nan
    n_decels=0; decel_depths=[]; decel_durs=[]; decel_burden=0.0
    if not np.isnan(baseline) and len(v) >= 60:
        fhr_i = np.where(~np.isnan(fhr), fhr, baseline)
        below = (fhr_i < (baseline-15)).astype(int)
        diffs = np.diff(below, prepend=0, append=0)
        starts = np.where(diffs==1)[0]; ends = np.where(diffs==-1)[0]
        for s,e in zip(starts,ends):
            dur = (e-s)/FS
            if dur >= 15:
                depth = baseline - fhr_i[s:e].min()
                n_decels += 1; decel_depths.append(depth); decel_durs.append(dur); decel_burden += depth*dur
    n_accels=0
    if not np.isnan(baseline) and len(v) >= 60:
        fhr_i = np.where(~np.isnan(fhr), fhr, baseline)
        above = (fhr_i > (baseline+15)).astype(int)
        diffs = np.diff(above, prepend=0, append=0)
        starts = np.where(diffs==1)[0]; ends = np.where(diffs==-1)[0]
        for s,e in zip(starts,ends):
            if (e-s)/FS >= 15: n_accels += 1
    dur_min = rec["duration_min"]; dur_30 = max(dur_min/30,0.001); dur_10 = max(dur_min/10,0.001)
    n_contractions=0; csr_frac=np.nan; uv = _valid(uc)
    if len(uv) >= 120:
        uc_i = np.where(~np.isnan(uc),uc,0); above_uc = (uc_i>30).astype(int)
        diffs = np.diff(above_uc,prepend=0,append=0)
        uc_starts=np.where(diffs==1)[0]; uc_ends=np.where(diffs==-1)[0]
        for s,e in zip(uc_starts,uc_ends):
            if (e-s)/FS >= 30: n_contractions += 1
        if n_contractions > 0:
            responded = 0
            if not np.isnan(baseline):
                fhr_i = np.where(~np.isnan(fhr),fhr,baseline)
                for cs,ce in zip(uc_starts,uc_ends):
                    if (ce-cs)/FS < 30: continue
                    seg = fhr_i[ce:min(ce+60*FS,len(fhr_i))]
                    if len(seg) > 0 and seg.min() < (baseline-15): responded += 1
            csr_frac = responded / n_contractions
    frs=50.0
    if not np.isnan(baseline): frs += 20 if 110<=baseline<=160 else (10 if (100<=baseline<110 or 160<baseline<=170) else 0)
    if not np.isnan(stv): frs += 20 if 5<=stv<=25 else (10 if 3<=stv<5 else 0)
    if not np.isnan(ltv): frs += 15 if 10<=ltv<=40 else (7 if 5<=ltv<10 else 0)
    if not np.isnan(baseline) and dur_min > 0:
        acc_rate = n_accels/(dur_min/30); frs += 15 if acc_rate>=2 else (8 if acc_rate>=1 else 0)
    frs = float(np.clip(frs - min(20, decel_burden*0.001+n_decels*2) - 50, 0, 100))
    stv_norm = stv/10.0 if not np.isnan(stv) else np.nan
    ltv_norm = ltv/25.0 if not np.isnan(ltv) else np.nan
    return {
        "record_id": rec["record_id"], "baseline_fhr": baseline, "mean_fhr": mean_fhr, "std_fhr": std_fhr,
        "stv": stv, "ltv": ltv, "stv_norm": stv_norm, "ltv_norm": ltv_norm,
        "tachycardia_frac": tach, "bradycardia_frac": brad,
        "n_decels": float(n_decels), "decels_per_30min": n_decels/dur_30,
        "mean_decel_depth": float(np.mean(decel_depths)) if decel_depths else 0.0,
        "max_decel_depth": float(max(decel_depths)) if decel_depths else 0.0,
        "mean_decel_dur_s": float(np.mean(decel_durs)) if decel_durs else 0.0,
        "n_accels": float(n_accels), "accels_per_30min": n_accels/dur_30,
        "n_contractions": float(n_contractions), "contractions_per_10min": n_contractions/dur_10,
        "decel_burden": decel_burden, "csr_frac": csr_frac,
        "fetal_reserve_score": frs, "duration_min": dur_min,
        "signal_quality": rec["signal_quality"], "missing_fhr": float(np.mean(np.isnan(rec["fhr"]))),
        "ph": rec["ph"], "base_deficit": rec["base_deficit"], "apgar1": rec["apgar1"], "apgar5": rec["apgar5"],
    }


def assign_risk_label(row) -> int:
    ph=row.get("ph",np.nan); bd=row.get("base_deficit",np.nan); a1=row.get("apgar1",np.nan)
    if not np.isnan(ph): return 2 if ph<7.05 else (1 if ph<7.15 else 0)
    if not np.isnan(bd): return 2 if bd>=12 else (1 if bd>=8 else 0)
    if not np.isnan(a1): return 2 if a1<4 else (1 if a1<7 else 0)
    return -1


def record_level_split(feat_df, test_frac=0.15, val_frac=0.15, seed=42):
    rids = feat_df["record_id"].values
    unique_rids = np.unique(rids)
    labels_per_rid = feat_df.groupby("record_id")["risk_label"].first().reindex(unique_rids).values
    train_val, test = train_test_split(unique_rids, test_size=test_frac, stratify=labels_per_rid, random_state=seed)
    tv_labels = feat_df.groupby("record_id")["risk_label"].first().reindex(train_val).values
    train, val = train_test_split(train_val, test_size=val_frac/(1-test_frac), stratify=tv_labels, random_state=seed)
    return feat_df["record_id"].isin(train), feat_df["record_id"].isin(val), feat_df["record_id"].isin(test)


# ── helpers ───────────────────────────────────────────────────────────────────
def arr_to_list(a): return [[round(float(x), 6) for x in row] for row in np.atleast_2d(a)]
def vec_to_list(v): return [round(float(x), 6) for x in v]


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== ReserveNet LR-Portable Export ===\n")

    data_dir = ROOT / "data" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
    if not data_dir.exists():
        raise RuntimeError("CTU data not extracted. Run train_reservenet_ctu.py first.")

    records = load_signals(data_dir)

    print(f"\n[features] extracting from {len(records)} records...")
    feats = [extract_features(r) for r in records]
    df = pd.DataFrame(feats)
    df["risk_label"] = df.apply(assign_risk_label, axis=1)
    df_labeled = df[df["risk_label"] >= 0].copy()
    print(f"[labels] {len(df_labeled)} labeled  0={sum(df_labeled.risk_label==0)} 1={sum(df_labeled.risk_label==1)} 2={sum(df_labeled.risk_label==2)}")

    cols = [c for c in FEATURE_COLS if c in df_labeled.columns]
    X_raw = df_labeled[cols].values.astype(float)
    y_raw = df_labeled["risk_label"].values.astype(int)

    idx_tr, idx_val, idx_te = record_level_split(df_labeled)
    X_tr_raw = X_raw[idx_tr]; y_tr = y_raw[idx_tr]
    X_va_raw = X_raw[idx_val]; y_va = y_raw[idx_val]

    imp = SimpleImputer(strategy="median")
    sc  = RobustScaler()
    X_tr = sc.fit_transform(imp.fit_transform(X_tr_raw))
    X_va = sc.transform(imp.transform(X_va_raw))

    print(f"[data] train={len(y_tr)} val={len(y_va)}")
    print(f"       train: 0={sum(y_tr==0)} 1={sum(y_tr==1)} 2={sum(y_tr==2)}")

    # ── build column index ────────────────────────────────────────────────────
    col_index = {c: i for i, c in enumerate(cols)}
    group_cols = {}
    for group, wanted in FEATURE_GROUPS.items():
        if wanted is None:
            group_cols[group] = list(range(len(cols)))
        else:
            group_cols[group] = [col_index[c] for c in wanted if c in col_index]

    expert_names = ["baseline_expert", "variability_expert", "event_expert"]

    # ── train all-LR experts ──────────────────────────────────────────────────
    C_map = {"baseline_expert": 0.5, "variability_expert": 1.0, "event_expert": 0.5}
    lr_experts = {}
    for name in expert_names:
        idx = group_cols[name]
        lr = LogisticRegression(C=C_map[name], class_weight="balanced", max_iter=3000, random_state=42)
        lr.fit(X_tr[:, idx], y_tr)
        lr_experts[name] = lr
        va_probs = lr.predict_proba(X_va[:, idx])
        # quick val stats
        preds = va_probs.argmax(axis=1)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_va, preds, average="macro", zero_division=0)
        print(f"[{name}] val macro-F1={f1:.3f}")

    # ── get LR expert probs on train + val ────────────────────────────────────
    def get_expert_probs(X):
        parts = []
        for name in expert_names:
            idx = group_cols[name]
            parts.append(lr_experts[name].predict_proba(X[:, idx]))
        return np.hstack(parts)

    ep_tr = get_expert_probs(X_tr)
    ep_va = get_expert_probs(X_va)

    meta_tr = np.hstack([ep_tr, X_tr])
    meta_va = np.hstack([ep_va, X_va])

    # ── train MLP meta-learner on LR expert probs ─────────────────────────────
    print("\n[MLP] training meta-learner on LR expert probs + raw features...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(96, 48),
        activation="relu", alpha=0.05,
        learning_rate="adaptive", max_iter=800,
        early_stopping=True, validation_fraction=0.15,
        random_state=42,
    )
    mlp.fit(meta_tr, y_tr)

    # Evaluate on val
    val_probs_raw = mlp.predict_proba(meta_va)
    val_preds = val_probs_raw.argmax(axis=1)
    from sklearn.metrics import recall_score
    sens = recall_score((y_va >= 1).astype(int), (val_probs_raw[:, 1]+val_probs_raw[:, 2] >= 0.35).astype(int), zero_division=0)
    print(f"[MLP] val sensitivity (thr=0.35): {sens:.3f}")

    # ── temperature scaling ───────────────────────────────────────────────────
    val_probs_c = np.clip(val_probs_raw, 1e-7, 1-1e-7)
    val_logits  = np.log(val_probs_c)
    ts = TemperatureScaler()
    ts.fit(val_logits, y_va)
    print(f"[temperature] T = {ts.T:.4f}")

    # ── verify on pathological-style val samples ──────────────────────────────
    high_mask = y_va == 2
    if high_mask.any():
        hp_raw = mlp.predict_proba(meta_va[high_mask])
        hp_cal = ts.scale(np.log(np.clip(hp_raw, 1e-7, 1-1e-7)))
        at_risk = hp_cal[:, 1] + hp_cal[:, 2]
        print(f"[verify] high-risk val cases (n={sum(high_mask)}): mean at-risk prob = {at_risk.mean():.3f}  (max={at_risk.max():.3f})")

    watch_mask = y_va == 1
    if watch_mask.any():
        wp_raw = mlp.predict_proba(meta_va[watch_mask])
        wp_cal = ts.scale(np.log(np.clip(wp_raw, 1e-7, 1-1e-7)))
        at_risk_w = wp_cal[:, 1] + wp_cal[:, 2]
        print(f"[verify] watch val cases (n={sum(watch_mask)}): mean at-risk prob = {at_risk_w.mean():.3f}")

    norm_mask = y_va == 0
    if norm_mask.any():
        np_raw = mlp.predict_proba(meta_va[norm_mask])
        np_cal = ts.scale(np.log(np.clip(np_raw, 1e-7, 1-1e-7)))
        at_risk_n = np_cal[:, 1] + np_cal[:, 2]
        print(f"[verify] normal val cases (n={sum(norm_mask)}): mean at-risk prob = {at_risk_n.mean():.3f}")

    # ── build export JSON ─────────────────────────────────────────────────────
    experts_out = {}
    for name in expert_names:
        idx = group_cols[name]
        fcols = [cols[i] for i in idx]
        lr = lr_experts[name]
        experts_out[name] = {
            "features":  fcols,
            "col_idx":   idx,
            "coef":      arr_to_list(lr.coef_),
            "intercept": vec_to_list(lr.intercept_),
            "classes":   [int(c) for c in lr.classes_],
        }

    mlp_layers = [
        {"W": arr_to_list(W.T), "b": vec_to_list(b)}
        for W, b in zip(mlp.coefs_, mlp.intercepts_)
    ]

    out = {
        "model":        "FetalyzeAI ReserveNet LR-Portable v1",
        "feature_cols": cols,
        "classes":      [0, 1, 2],
        "class_names":  ["Normal", "Watch", "High Risk"],
        "temperature_T": round(float(ts.T), 6),
        "scaler": {
            "center": vec_to_list(sc.center_),
            "scale":  vec_to_list(sc.scale_),
        },
        "imputer": {
            "medians": vec_to_list(imp.statistics_),
        },
        "experts": experts_out,
        "meta_mlp": {
            "n_expert_probs": 9,
            "n_raw_features": len(cols),
            "n_input":        9 + len(cols),
            "activation":     "relu",
            "layers":         mlp_layers,
            "out_classes":    [int(c) for c in mlp.classes_],
        },
    }

    out_path = ROOT / "results" / "reservenet_inference.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    size_kb = out_path.stat().st_size / 1024
    print(f"\n[export] → {out_path}  ({size_kb:.1f} KB)")
    print("[export] complete")


if __name__ == "__main__":
    main()
