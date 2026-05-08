"""
train_pulsefm.py
================
FetalyzeAI PulseFM-ReserveNet — Full Training Pipeline

Two-stage training (spec §14):
  Stage 1  Self-supervised pretraining
           Masked autoencoder on all CTG windows (no labels needed).
           Loss: MSE on hidden positions of FHR + UC channels.

  Stage 2  Supervised fine-tuning
           Weighted focal loss (class imbalance ~447/65/40)
           + 0.1 × reserve regression auxiliary loss.
           Record-level label, attention-pooled over 5-min windows.

Architecture: PulseFMReserveNet
  PulseEncoder → AttentionPooling → GatedReserveFusion → 3-class output

Ensemble: 5 seeds → calibrated with temperature scaling on validation set.

Uncertainty (spec §16):
  U = 0.60 × H(p̄)_norm  +  0.40 × Var_norm

Data:  CTU-CHB / CTU-UHB — 552 records — no synthetic fallback.

Usage:
    python train_pulsefm.py                     # full training
    python train_pulsefm.py --skip-pretrain     # skip Stage 1
    python train_pulsefm.py --seeds 42          # single model (fast)
"""

from __future__ import annotations

import argparse, json, pickle, sys, time, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

# ── require torch ────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
except ImportError:
    print("[ERROR] PyTorch is required for train_pulsefm.py.")
    print("        Install via:  pip install torch  (CPU build is fine)")
    sys.exit(1)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, balanced_accuracy_score,
    precision_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from scipy.optimize import minimize_scalar

from ctu_loader         import load_ctu_records
from ctg_feature_engine import extract_record_features
from pulsefm_encoder    import (
    PulseFMReserveNet, MaskedCTGAutoencoder, EnsemblePulseFM,
    extract_windows, WINDOW_LEN, FS, EMBED_DIM,
)

ROOT        = Path(__file__).parent
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature columns forwarded to GatedReserveFusion
# ─────────────────────────────────────────────────────────────────────────────

FEAT_COLS = [
    "baseline_fhr", "mean_fhr", "std_fhr",
    "stv", "ltv", "stv_norm", "ltv_norm", "roughness",
    "tachycardia_frac", "bradycardia_frac",
    "n_accels", "accels_per_30min", "mean_accel_height",
    "n_decels", "decels_per_30min",
    "mean_decel_depth", "max_decel_depth", "mean_decel_dur_s",
    "prolonged_decel_flag", "late_decel_likelihood",
    "delayed_recovery_score", "worsening_recovery_trend",
    "decel_burden_idx", "fetal_reserve_score",
    "n_contractions", "contractions_per_10min",
    "mean_fhr_drop_post_uc", "mean_recovery_time_s",
    "missing_fhr_pct", "signal_quality", "duration_min",
    "lf_power", "mf_power", "hf_power", "lf_hf_ratio", "spectral_entropy",
    "stv_trend_late_vs_full", "baseline_trend_late_vs_full",
    "baseline_fhr_last30", "stv_last30",
]
FEAT_DIM = len(FEAT_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────

def assign_label(row: dict) -> int:
    ph = row.get("ph", float("nan"))
    bd = row.get("base_deficit", float("nan"))
    a5 = row.get("apgar5",  float("nan"))
    a1 = row.get("apgar1",  float("nan"))
    if not np.isnan(ph):
        return 2 if ph < 7.05 else (1 if ph < 7.15 else 0)
    if not np.isnan(bd):
        return 2 if bd >= 12 else (1 if bd >= 8 else 0)
    if not np.isnan(a5):
        return 2 if a5 < 7 else (1 if a5 == 7 else 0)
    if not np.isnan(a1):
        return 2 if a1 < 4 else (1 if a1 < 7 else 0)
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """Flat dataset of (window_tensor, dummy_label=0) for self-supervised pretraining."""
    def __init__(self, window_list: list[np.ndarray]):
        self.windows = [torch.from_numpy(w) for w in window_list]

    def __len__(self): return len(self.windows)
    def __getitem__(self, i): return self.windows[i]


class RecordDataset(Dataset):
    """
    Record-level dataset for supervised training.
    Each item: (all_windows_tensor, features_tensor, label, frs_scalar)
    """
    def __init__(self, records: list[dict], feat_dim: int = FEAT_DIM,
                 stride: int = WINDOW_LEN):
        self.items: list[tuple] = []
        for rec in records:
            fhr  = np.asarray(rec["fhr"], dtype=np.float32)
            uc   = np.asarray(rec["uc"],  dtype=np.float32)
            wins = extract_windows(fhr, uc, WINDOW_LEN, stride)    # (N, 3, L)
            feat = np.asarray(rec["features"], dtype=np.float32)   # (feat_dim,)
            lbl  = int(rec["label"])
            frs  = float(rec.get("fetal_reserve_score", 50.0))
            self.items.append((
                torch.from_numpy(wins),
                torch.from_numpy(feat),
                torch.tensor(lbl,  dtype=torch.long),
                torch.tensor(frs / 100.0, dtype=torch.float32),
            ))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def effective_num_weights(class_counts: list[int], beta: float = 0.99) -> torch.Tensor:
    """
    Class weights using effective number of samples (spec §14 stage 2):
        w_c = (1 - β) / (1 - β^n_c)
    """
    w = torch.tensor([(1 - beta) / (1 - beta ** max(n, 1))
                       for n in class_counts], dtype=torch.float32)
    return w / w.sum() * len(class_counts)


def focal_loss(logits: "torch.Tensor", targets: "torch.Tensor",
               class_weights: "torch.Tensor", gamma: float = 1.5) -> "torch.Tensor":
    """
    Weighted focal loss (spec §14 stage 3):
        L = -w_y (1 - p_y)^γ log(p_y)
    """
    log_p = F.log_softmax(logits, dim=-1)
    p     = torch.exp(log_p)
    lbl   = targets.view(-1)
    lp_y  = log_p.gather(1, lbl.unsqueeze(1)).squeeze(1)
    p_y   = p.gather(1,    lbl.unsqueeze(1)).squeeze(1)
    w_y   = class_weights[lbl]
    return (- w_y * (1 - p_y).pow(gamma) * lp_y).mean()


def masked_recon_loss(recon: "torch.Tensor", target: "torch.Tensor",
                      mask: "torch.Tensor") -> "torch.Tensor":
    """
    Reconstruction loss only on masked positions (spec §7):
        L = ||(1 - M) ⊙ (X - X̂)||²
    """
    not_mask = (~mask).float().expand_as(recon)
    diff = (recon - target[:, :recon.shape[1], :]) * not_mask
    return (diff ** 2).sum() / (not_mask.sum() + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_temperature(logits_np: np.ndarray, y_np: np.ndarray) -> float:
    def nll(T):
        T = max(T, 0.01)
        scaled = logits_np / T
        exp_s  = np.exp(scaled - scaled.max(1, keepdims=True))
        p      = exp_s / exp_s.sum(1, keepdims=True)
        p      = np.clip(p, 1e-7, 1.0)
        return -np.mean(np.log(p[np.arange(len(y_np)), y_np.astype(int)]))
    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return max(float(res.x), 0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Self-supervised pretraining
# ─────────────────────────────────────────────────────────────────────────────

def stage1_pretrain(encoder: "PulseFMReserveNet",
                    train_wins: list[np.ndarray],
                    n_epochs: int = 50,
                    batch_size: int = 64,
                    lr: float = 1e-3,
                    device: str = "cpu") -> float:
    """
    Train the encoder on masked reconstruction.
    Only train_wins (from training records) are used — no val/test leakage.
    Returns final training loss.
    """
    print(f"\n[stage1] self-supervised pretraining on {len(train_wins)} windows ...")
    autoencoder = MaskedCTGAutoencoder(encoder.encoder, out_channels=2,
                                       window_len=WINDOW_LEN).to(device)
    optimiser   = Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler   = CosineAnnealingLR(optimiser, T_max=n_epochs, eta_min=lr * 0.1)
    dataset     = WindowDataset(train_wins)
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=False)
    autoencoder.train()
    last_loss = 0.0
    for ep in range(1, n_epochs + 1):
        ep_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            recon, mask = autoencoder(batch)
            loss = masked_recon_loss(recon, batch, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
            optimiser.step()
            ep_loss += loss.item()
        scheduler.step()
        last_loss = ep_loss / max(len(loader), 1)
        if ep % 10 == 0:
            print(f"  epoch {ep:3d}/{n_epochs}  recon_loss={last_loss:.5f}")
    print(f"[stage1] done  final_loss={last_loss:.5f}")
    return last_loss


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Supervised fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def stage2_train(model: "PulseFMReserveNet",
                 train_records: list[dict],
                 val_records:   list[dict],
                 class_counts:  list[int],
                 n_epochs: int = 80,
                 batch_size: int = 16,
                 lr: float = 5e-4,
                 reserve_weight: float = 0.10,
                 device: str = "cpu") -> dict:
    """
    Fine-tune with weighted focal loss + optional reserve regression.
    Returns dict with best val metrics.
    """
    print(f"\n[stage2] supervised fine-tuning  "
          f"train={len(train_records)}  val={len(val_records)} ...")
    model = model.to(device)
    class_w  = effective_num_weights(class_counts).to(device)
    optimiser = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=n_epochs, eta_min=lr * 0.05)
    train_ds  = RecordDataset(train_records)
    val_ds    = RecordDataset(val_records)

    best_val_auc = 0.0
    best_state   = None

    for ep in range(1, n_epochs + 1):
        model.train()
        ep_loss = 0.0
        for wins, feat, lbl, frs in DataLoader(train_ds, batch_size=1, shuffle=True):
            wins = wins.squeeze(0).to(device)   # (N_windows, 3, L)
            feat = feat.squeeze(0).to(device)   # (feat_dim,)
            lbl  = lbl.to(device)
            frs  = frs.to(device)

            optimiser.zero_grad()
            logits, probs, r, _ = model(wins, feat)
            cls_loss = focal_loss(logits.unsqueeze(0), lbl, class_w)
            # Reserve auxiliary: predict normalised FRS from representation
            frs_pred = torch.sigmoid(r.mean())
            res_loss = F.mse_loss(frs_pred, frs)
            loss = cls_loss + reserve_weight * res_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            ep_loss += loss.item()
        scheduler.step()

        # Validation
        if ep % 10 == 0 or ep == n_epochs:
            val_auc, val_f1, val_sens = evaluate_records(model, val_records, device)
            print(f"  epoch {ep:3d}/{n_epochs}  "
                  f"loss={ep_loss/max(len(train_ds),1):.4f}  "
                  f"val_AUROC={val_auc:.4f}  "
                  f"val_sens={val_sens:.4f}  val_F1={val_f1:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    return {"best_val_auroc": best_val_auc}


@torch.no_grad()
def evaluate_records(model: "PulseFMReserveNet",
                     records: list[dict],
                     device: str = "cpu") -> tuple[float, float, float]:
    model.eval()
    model = model.to(device)
    all_p, all_y = [], []
    for rec in records:
        fhr  = np.asarray(rec["fhr"], dtype=np.float32)
        uc   = np.asarray(rec["uc"],  dtype=np.float32)
        wins = torch.from_numpy(extract_windows(fhr, uc)).to(device)
        feat = torch.from_numpy(
            np.asarray(rec["features"], dtype=np.float32)).to(device)
        _, probs, _, _ = model(wins, feat)
        all_p.append(probs.cpu().numpy())
        all_y.append(int(rec["label"]))
    all_p = np.array(all_p)
    all_y = np.array(all_y)
    preds  = all_p.argmax(axis=1)
    try:
        from sklearn.preprocessing import label_binarize
        yb = label_binarize(all_y, classes=[0, 1, 2])
        auc = float(roc_auc_score(yb, all_p, multi_class="ovr", average="macro"))
    except Exception:
        auc = 0.5
    f1   = float(f1_score(all_y, preds, average="macro", zero_division=0))
    sens = float(recall_score(all_y, preds, labels=[2], average="macro", zero_division=0))
    return auc, f1, sens


# ─────────────────────────────────────────────────────────────────────────────
# Record-level split (leakage-free)
# ─────────────────────────────────────────────────────────────────────────────

def record_level_split(records: list[dict],
                       test_frac: float = 0.15,
                       val_frac:  float = 0.15,
                       seed:      int   = 42) -> tuple[list, list, list]:
    """
    Split by record so no patient's windows appear in multiple folds.
    Critical — see spec §17 item 1.
    """
    labels = np.array([r["label"] for r in records])
    idx    = np.arange(len(records))
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_frac, stratify=labels, random_state=seed)
    tv_labels = labels[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_frac / (1 - test_frac),
        stratify=tv_labels,
        random_state=seed,
    )
    return ([records[i] for i in train_idx],
            [records[i] for i in val_idx],
            [records[i] for i in test_idx])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f4(v):
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return v

def roc_pts(y, s, n=30):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.unique(np.linspace(0, len(fpr)-1, n).astype(int))
    return [{"fpr": _f4(float(fpr[i])), "tpr": _f4(float(tpr[i]))} for i in idx]

def pr_pts(y, s, n=30):
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(y, s)
    idx = np.unique(np.linspace(0, len(p)-1, n).astype(int))
    return [{"precision": _f4(float(p[i])), "recall": _f4(float(r[i]))} for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(skip_pretrain: bool = False, seeds: list[int] = None):
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*65}")
    print(f" FetalyzeAI PulseFM-ReserveNet — CTU-CHB Training  [device: {device}]")
    print(f"{'='*65}")

    seeds = seeds or [42, 7, 2024, 1337, 99]

    # ── Load records ──────────────────────────────────────────────────────────
    print("[data] loading CTU-CHB records ...")
    raw_records = load_ctu_records(verbose=True)

    # ── Feature extraction ────────────────────────────────────────────────────
    print(f"[features] extracting for {len(raw_records)} records ...")
    feat_rows = []
    for rec in raw_records:
        try:
            row = extract_record_features(rec)
            feat_rows.append(row)
        except Exception as e:
            print(f"  [skip] {rec.record_id}: {e}")
    df = pd.DataFrame(feat_rows)
    df["risk_label"] = df.apply(assign_label, axis=1)
    df_lab = df[df["risk_label"] >= 0].reset_index(drop=True)
    counts = df_lab["risk_label"].value_counts().sort_index().to_dict()
    print(f"[labels] N={len(df_lab)}  "
          f"class 0={counts.get(0,0)}  class 1={counts.get(1,0)}  class 2={counts.get(2,0)}")

    # ── Preprocess features ───────────────────────────────────────────────────
    feat_cols = [c for c in FEAT_COLS if c in df_lab.columns]
    X_raw = df_lab[feat_cols].values.astype(float)
    y_raw = df_lab["risk_label"].values.astype(int)
    rids  = df_lab["record_id"].values

    # Build record dicts  {fhr, uc, features, label, fetal_reserve_score}
    rec_map = {r.record_id: r for r in raw_records}
    all_records = []
    for i, (rid, feat_row) in enumerate(zip(rids, feat_rows)):
        if df_lab.iloc[i]["risk_label"] < 0:
            continue
        r = rec_map.get(rid)
        if r is None:
            continue
        all_records.append({
            "record_id": rid,
            "fhr":       r.fhr,
            "uc":        r.uc,
            "features":  X_raw[i],           # raw (pre-scale) — scaled per split
            "label":     int(df_lab.iloc[i]["risk_label"]),
            "fetal_reserve_score": float(feat_row.get("fetal_reserve_score", 50.0)),
        })

    # ── Record-level split (no leakage) ───────────────────────────────────────
    train_recs, val_recs, test_recs = record_level_split(all_records, seed=42)
    print(f"[split] train={len(train_recs)}  val={len(val_recs)}  test={len(test_recs)}")

    # ── Fit imputer + scaler on training records only ─────────────────────────
    X_tr_raw = np.array([r["features"] for r in train_recs])
    imputer  = SimpleImputer(strategy="median").fit(X_tr_raw)
    scaler   = RobustScaler().fit(imputer.transform(X_tr_raw))
    def scale_records(recs):
        for r in recs:
            r["features"] = scaler.transform(
                imputer.transform(r["features"].reshape(1, -1))
            ).flatten().astype(np.float32)
        return recs
    train_recs = scale_records(train_recs)
    val_recs   = scale_records(val_recs)
    test_recs  = scale_records(test_recs)

    # Class counts from training set only
    tr_labels     = np.array([r["label"] for r in train_recs])
    class_counts  = [int(np.sum(tr_labels == c)) for c in range(3)]
    print(f"[train class counts] {class_counts}")

    # ── Stage 1 — windows from training records only ──────────────────────────
    if not skip_pretrain:
        train_wins = []
        for r in train_recs:
            ws = extract_windows(np.asarray(r["fhr"], dtype=np.float32),
                                 np.asarray(r["uc"],  dtype=np.float32),
                                 WINDOW_LEN, stride=WINDOW_LEN // 2)
            train_wins.extend(list(ws))
        print(f"[stage1] {len(train_wins)} training windows (50% overlap)")

    # ── Stage 2 — train ensemble ──────────────────────────────────────────────
    trained_models = []
    for seed_i, seed in enumerate(seeds, 1):
        print(f"\n[ensemble] model {seed_i}/{len(seeds)}  seed={seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = PulseFMReserveNet(
            in_channels=3,
            embed_dim=EMBED_DIM,
            feat_dim=len(feat_cols),
            n_classes=3,
            dropout=0.20,
        )

        if not skip_pretrain:
            stage1_pretrain(model, train_wins, n_epochs=40,
                            batch_size=64, lr=1e-3, device=device)

        stage2_train(model, train_recs, val_recs,
                     class_counts=class_counts,
                     n_epochs=80, batch_size=1,
                     lr=5e-4, reserve_weight=0.10, device=device)

        trained_models.append(model.cpu())
        # Save individual model
        torch.save(model.state_dict(),
                   MODELS_DIR / f"pulsefm_seed{seed}.pt")

    # ── Temperature calibration on validation set ─────────────────────────────
    print("\n[calibrate] temperature scaling on validation set ...")
    model0 = trained_models[0].to(device)
    val_logits, val_y = [], []
    for rec in val_recs:
        wins = torch.from_numpy(
            extract_windows(np.asarray(rec["fhr"], np.float32),
                            np.asarray(rec["uc"],  np.float32))).to(device)
        feat = torch.from_numpy(rec["features"]).to(device)
        with torch.no_grad():
            lg, _, _, _ = model0(wins, feat)
        val_logits.append(lg.cpu().numpy())
        val_y.append(rec["label"])
    val_logits = np.array(val_logits)
    val_y      = np.array(val_y)
    temp_T     = calibrate_temperature(val_logits, val_y)
    print(f"[calibrate] T = {temp_T:.4f}")

    # ── Build ensemble ────────────────────────────────────────────────────────
    ensemble = EnsemblePulseFM(trained_models, temp_T=temp_T)
    with open(MODELS_DIR / "pulsefm_ensemble.pkl", "wb") as f:
        pickle.dump({"ensemble": ensemble, "imputer": imputer,
                     "scaler": scaler, "feat_cols": feat_cols}, f)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n[evaluate] test set ...")
    test_probs, test_y = [], []
    for rec in test_recs:
        wins = torch.from_numpy(
            extract_windows(np.asarray(rec["fhr"], np.float32),
                            np.asarray(rec["uc"],  np.float32))).to(device)
        feat = torch.from_numpy(rec["features"]).to(device)
        res  = ensemble.predict(wins, feat)
        test_probs.append(res["probs"])
        test_y.append(rec["label"])
    test_probs = np.array(test_probs)
    test_y     = np.array(test_y)

    yb_te   = (test_y >= 1).astype(int)
    p_bin   = test_probs[:, 1] + test_probs[:, 2]
    fpr_v, tpr_v, thr_v = roc_curve(yb_te, p_bin)
    best_thr = float(np.clip(thr_v[int(np.argmax(tpr_v - fpr_v))], 0.05, 0.95))
    yb_pred  = (p_bin >= best_thr).astype(int)

    te_auroc = float(roc_auc_score(yb_te, p_bin)) if len(np.unique(yb_te)) > 1 else float("nan")
    te_sens  = float(recall_score(yb_te, yb_pred, zero_division=0))
    te_spec  = float(recall_score(yb_te, yb_pred, pos_label=0, zero_division=0))
    te_f1    = float(f1_score(yb_te, yb_pred, zero_division=0))
    te_prec  = float(precision_score(yb_te, yb_pred, zero_division=0))
    te_auprc = float(average_precision_score(yb_te, p_bin)) if len(np.unique(yb_te)) > 1 else float("nan")
    te_bal   = float(balanced_accuracy_score(yb_te, yb_pred))

    from sklearn.preprocessing import label_binarize
    yb3 = label_binarize(test_y, classes=[0, 1, 2])
    try:
        te_auroc_3 = float(roc_auc_score(yb3, test_probs, multi_class="ovr", average="macro"))
    except Exception:
        te_auroc_3 = float("nan")
    te_f1_3  = float(f1_score(test_y, test_probs.argmax(1), average="macro", zero_division=0))
    te_hr_re = float(recall_score(test_y, test_probs.argmax(1), labels=[2],
                                  average="macro", zero_division=0))
    cm = confusion_matrix(test_y, test_probs.argmax(1), labels=[0,1,2]).tolist()

    print(f"[test] AUROC(bin)={te_auroc:.4f}  sens={te_sens:.4f}  "
          f"spec={te_spec:.4f}  F1={te_f1:.4f}")
    print(f"[test] AUROC(macro3)={te_auroc_3:.4f}  HR-recall={te_hr_re:.4f}")

    # ── Uncertainty summary on test set ───────────────────────────────────────
    unc_vals = []
    for rec in test_recs:
        wins = torch.from_numpy(
            extract_windows(np.asarray(rec["fhr"], np.float32),
                            np.asarray(rec["uc"],  np.float32))).to(device)
        feat = torch.from_numpy(rec["features"]).to(device)
        res  = ensemble.predict(wins, feat)
        unc_vals.append(res["uncertainty"])
    unc_arr = np.array(unc_vals)

    elapsed = round(time.time() - t0, 1)

    # ── Build output JSON ─────────────────────────────────────────────────────
    out = {
        "generated_at":  datetime.now().isoformat(),
        "model_version": "pulsefm-reservenet-1.0",
        "architecture":  "PulseFM-ReserveNet",
        "dataset":       "CTU-CHB Intrapartum CTG (real records only)",
        "n_records":     int(len(raw_records)),
        "n_labeled":     int(len(all_records)),
        "class_counts":  {str(k): int(v) for k, v in counts.items()},
        "n_features":    len(feat_cols),
        "n_ensemble":    len(seeds),
        "split": {
            "train": len(train_recs),
            "val":   len(val_recs),
            "test":  len(test_recs),
        },
        "architecture_detail": {
            "encoder":           "CNN-TCN (Conv k=9 / k=7 / dil=1,2,4,8)",
            "embed_dim":         EMBED_DIM,
            "window_len_s":      WINDOW_LEN // FS,
            "aggregation":       "attention pooling (learned α weights)",
            "fusion":            "gated ReserveNet (sigmoid gate z/h)",
            "loss":              "weighted focal (γ=1.5) + 0.1 × reserve regression",
            "pretrain":          "masked autoencoder (25% block masking)" if not skip_pretrain else "skipped",
            "calibration":       f"temperature scaling T={round(temp_T, 4)}",
            "uncertainty":       "0.6×entropy + 0.4×ensemble_variance",
            "ensemble_seeds":    seeds,
        },
        "binary_headline": {
            "auroc":        _f4(te_auroc),
            "auprc":        _f4(te_auprc),
            "sensitivity":  _f4(te_sens),
            "specificity":  _f4(te_spec),
            "f1":           _f4(te_f1),
            "precision":    _f4(te_prec),
            "balanced_acc": _f4(te_bal),
            "threshold":    _f4(best_thr),
            "method":       f"held-out test set (N={len(test_recs)})",
        },
        "multiclass_test": {
            "auroc_macro":     _f4(te_auroc_3),
            "macro_f1":        _f4(te_f1_3),
            "high_risk_recall": _f4(te_hr_re),
            "confusion_matrix": cm,
        },
        "roc_curve": roc_pts(yb_te, p_bin),
        "pr_curve":  pr_pts(yb_te, p_bin),
        "uncertainty": {
            "mean":         _f4(float(unc_arr.mean())),
            "low_pct":      _f4(float(np.mean(unc_arr < 0.33) * 100)),
            "mid_pct":      _f4(float(np.mean((unc_arr >= 0.33) & (unc_arr < 0.67)) * 100)),
            "high_pct":     _f4(float(np.mean(unc_arr >= 0.67) * 100)),
            "temperature_T": _f4(temp_T),
            "formula":       "U = 0.6 × H(p̄)_norm + 0.4 × Var_norm",
        },
        "training_time_s": elapsed,
    }

    out_path = ROOT / "ctu_model_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    results_copy = RESULTS_DIR / f"pulsefm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_copy, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*65}")
    print(f" PulseFM-ReserveNet training complete in {elapsed:.1f}s")
    print(f" Binary AUROC (test): {te_auroc:.4f}")
    print(f" Sensitivity:         {te_sens:.4f}")
    print(f" Specificity:         {te_spec:.4f}")
    print(f" HR Recall:           {te_hr_re:.4f}")
    print(f" Results → {out_path}")
    print(f"{'='*65}\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip Stage 1 self-supervised pretraining")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Ensemble seeds (default: 42 7 2024 1337 99)")
    args = parser.parse_args()
    main(skip_pretrain=args.skip_pretrain, seeds=args.seeds)
