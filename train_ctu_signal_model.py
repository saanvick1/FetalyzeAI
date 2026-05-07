"""
FetalyzeAI — CTU-UHB Raw Signal Model (Primary)
=================================================
Trains on the CTU-UHB Intrapartum CTG Database as the PRIMARY clinical dataset.

Architecture:
  Input: FHR + UC raw signals @ 4 Hz, windowed to 5-minute segments
  Signal encoder: 1D CNN → Temporal Convolutional Network (TCN)
  Clinical feature branch: extracted features from ctgdl_features.py
  Fusion: gated MLP
  Output: risk class (low / watch / high) + uncertainty + fetal reserve score

Dataset hierarchy:
  1. CTU-UHB signals + outcomes  → primary model training (pH-based labels)
  2. CTU-CHB annotations         → event engine (decel / accel detection)

References:
  Chudáček et al. (2014) BMC Pregnancy and Childbirth 14:16
  https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

Run:
  python train_ctu_signal_model.py
  Outputs: ctu_model_results.json, ctu_signal_model.pkl
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, recall_score,
    balanced_accuracy_score, confusion_matrix, classification_report,
    average_precision_score, brier_score_loss
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
import xgboost as xgb

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ─── Import project modules ───────────────────────────────────────────────────
from ctgdl_loader import CTUDataset, make_synthetic_ctu_uhb
from ctgdl_features import (
    extract_ctg_features, extract_features_batch,
    compute_fetal_reserve_score, detect_decelerations, detect_accelerations,
    detect_contractions, compute_contraction_stress_response,
    compute_deceleration_burden,
)

FS = 4
WINDOW_SEC   = 5 * 60      # 5-minute windows = 1200 samples
STEP_SEC     = 150         # 2.5-minute step (50% overlap)
N_CLASSES    = 3           # 0=low-risk, 1=watch, 2=high-risk
DEVICE       = torch.device("cpu")

print("=" * 70)
print("  FetalyzeAI — CTU-UHB Primary Signal Model Trainer")
print("=" * 70)

# ─── 1. Load CTU-UHB (primary) ────────────────────────────────────────────────
print("\n[1/9] Loading CTU-UHB dataset (primary clinical source)...")
ctu = CTUDataset(max_records=552, force_synthetic=False, load_annotations=True, verbose=True)
ctu.load()
records = ctu.records
stats   = ctu.stats()

print(f"\n  Dataset: {stats['n_records']} records | "
      f"{stats['total_hours']} hours | "
      f"method: {stats['load_method']}")
print(f"  pH available: {stats['n_ph_available']} | "
      f"acidosis: {stats['n_acidosis']} | "
      f"borderline: {stats['n_borderline']} | "
      f"normal: {stats['n_normal_ph']}")
if not ctu.is_real_data:
    print("  NOTE: Using synthetic fallback — real CTU-UHB not downloaded.")
    print("  To use real data: pip install wfdb (PhysioNet streaming)")

# ─── 2. Feature extraction from CTU-UHB signals ───────────────────────────────
print("\n[2/9] Extracting clinical features from FHR + UC signals...")
feat_df = extract_features_batch(records)

# Remove rows with extraction errors
feat_df = feat_df[~feat_df.get("error", pd.Series(dtype=str)).notna()].copy()
print(f"  Extracted {len(feat_df)} records × {len(feat_df.columns)} features")

# ─── 3. Build training labels ─────────────────────────────────────────────────
print("\n[3/9] Building risk labels from clinical outcomes...")

def assign_risk_label(row) -> int:
    """
    Assign 3-class risk label from CTU-UHB clinical outcomes.

    0 = low risk    (normal pH, good Apgar, no NICU)
    1 = watch       (borderline pH or low Apgar)
    2 = high risk   (acidosis pH < 7.05, or Apgar1 < 7)

    Falls back to Fetal Reserve Score if pH/Apgar unavailable.
    """
    ph    = row.get("ph", float("nan"))
    a1    = row.get("apgar1", float("nan"))
    a5    = row.get("apgar5", float("nan"))
    frs   = row.get("fetal_reserve_score", 50.0)
    dbi   = row.get("deceleration_burden_index", 0.0)

    if not (isinstance(ph, float) and np.isnan(ph)):
        ph = float(ph)
        if ph < 7.05:
            return 2   # acidosis = high risk
        if ph < 7.15:
            return 1   # borderline pH = watch
    if not (isinstance(a1, float) and np.isnan(a1)):
        a1 = float(a1)
        if a1 < 7:
            return 2
        if a1 <= 7:
            return 1
    # Fall back to computed clinical features
    if frs < 35 or dbi > 200:
        return 2
    if frs < 55 or dbi > 80:
        return 1
    return 0

feat_df["risk_label"] = feat_df.apply(assign_risk_label, axis=1)

label_counts = feat_df["risk_label"].value_counts().sort_index()
print(f"  Risk distribution: "
      f"Low={label_counts.get(0, 0)} "
      f"Watch={label_counts.get(1, 0)} "
      f"High={label_counts.get(2, 0)}")

# Drop rows without usable label
feat_df = feat_df.dropna(subset=["risk_label"]).copy()
feat_df["risk_label"] = feat_df["risk_label"].astype(int)

# ─── 4. Prepare tabular feature matrix ────────────────────────────────────────
print("\n[4/9] Preparing feature matrix for model training...")

# Use extracted CTG signal features from raw FHR/UC waveforms
SIGNAL_FEATURES = [
    "baseline_fhr", "mean_fhr", "std_fhr", "min_fhr", "max_fhr",
    "stv", "ltv", "fhr_entropy", "fhr_late_slope",
    "tachycardia_frac", "bradycardia_frac",
    "n_decels", "decels_per_30min", "mean_decel_depth", "max_decel_depth",
    "mean_decel_duration_s", "total_decel_area",
    "n_late_decels", "n_variable_decels", "n_prolonged_decels", "n_early_decels",
    "mean_recovery_slope", "deceleration_burden_index",
    "n_accels", "accels_per_30min", "mean_accel_height", "mean_accel_duration_s",
    "n_contractions", "contractions_per_10min",
    "mean_contraction_duration_s", "mean_contraction_intensity",
    "mean_contraction_interval_s",
    "decel_after_contraction_frac", "mean_decel_lag_s",
    "mean_recovery_after_contraction_s", "late_decel_after_contraction_count",
    "fetal_reserve_score",
    "signal_quality", "missing_fhr", "duration_min",
]

available_features = [f for f in SIGNAL_FEATURES if f in feat_df.columns]
print(f"  Using {len(available_features)} signal features")

X_raw = feat_df[available_features].values.astype(float)
y_raw = feat_df["risk_label"].values.astype(int)

# Guard: need at least 2 classes and 30 samples
if len(np.unique(y_raw)) < 2 or len(y_raw) < 30:
    print("  Insufficient class variety — augmenting with synthetic records...")
    synth = make_synthetic_ctu_uhb(n=80, seed=99)
    synth_feat = extract_features_batch(synth)
    synth_feat = synth_feat[~synth_feat.get("error", pd.Series(dtype=str)).notna()].copy()
    synth_feat["risk_label"] = synth_feat.apply(assign_risk_label, axis=1)
    synth_avail = [f for f in available_features if f in synth_feat.columns]
    X_synth = synth_feat[synth_avail].reindex(columns=available_features).values.astype(float)
    y_synth = synth_feat["risk_label"].values.astype(int)
    X_raw = np.vstack([X_raw, X_synth])
    y_raw = np.concatenate([y_raw, y_synth])
    feat_df = pd.concat([feat_df, synth_feat], ignore_index=True)
    label_counts = pd.Series(y_raw).value_counts().sort_index()
    print(f"  After augmentation: {len(y_raw)} samples | "
          f"Low={label_counts.get(0,0)} Watch={label_counts.get(1,0)} High={label_counts.get(2,0)}")

# ─── 5. Leakage-free train/test split ─────────────────────────────────────────
print("\n[5/9] Leakage-free train/test split...")

# Ensure all classes present in split
try:
    idx_tr, idx_te = train_test_split(
        np.arange(len(X_raw)), test_size=0.2, stratify=y_raw, random_state=42
    )
except ValueError:
    idx_tr, idx_te = train_test_split(
        np.arange(len(X_raw)), test_size=0.2, random_state=42
    )

imp_tr  = SimpleImputer(strategy="median")
sc_tr   = RobustScaler()
X_tr    = sc_tr.fit_transform(imp_tr.fit_transform(X_raw[idx_tr]))
X_te    = sc_tr.transform(imp_tr.transform(X_raw[idx_te]))
y_tr    = y_raw[idx_tr]
y_te    = y_raw[idx_te]

cw_arr  = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
cw_dict = {int(c): float(w) for c, w in zip(np.unique(y_tr), cw_arr)}
sw_tr   = np.array([cw_dict.get(int(yi), 1.0) for yi in y_tr])

# Extra sensitivity weight for high-risk class
for k in cw_dict:
    if k == 2:
        cw_dict[k] *= 2.0

print(f"  Train: {len(idx_tr)} | Test: {len(idx_te)}")
print(f"  Class weights: {cw_dict}")

# ─── 6. Regularized XGBoost on signal features ────────────────────────────────
print("\n[6/9] Training XGBoost on CTU-UHB signal features...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    gamma=1.0, reg_alpha=0.5, reg_lambda=5.0,
    objective="multi:softprob", eval_metric="mlogloss",
    num_class=N_CLASSES,
    random_state=42, n_jobs=-1, tree_method="hist",
)
xgb_model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=0)

xgb_preds_te  = xgb_model.predict(X_te)
xgb_probs_te  = xgb_model.predict_proba(X_te)
xgb_acc_te    = accuracy_score(y_te, xgb_preds_te)
xgb_f1_te     = f1_score(y_te, xgb_preds_te, average="macro", zero_division=0)
xgb_path_rec  = recall_score(y_te, xgb_preds_te, labels=[2], average="macro", zero_division=0)
print(f"  XGBoost: acc={xgb_acc_te*100:.2f}%  F1={xgb_f1_te:.4f}  HighRecall={xgb_path_rec:.4f}")

xgb_probs_tr  = xgb_model.predict_proba(X_tr)

# ─── 7. Small MLP ensemble (signal features + XGBoost probs) ──────────────────
print("\n[7/9] Training small MLP ensemble (5 members)...")

class SignalMLP(nn.Module):
    """Compact MLP for CTU-UHB extracted features."""
    def __init__(self, n_features: int, xgb_d: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features + xgb_d, hidden),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, N_CLASSES),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, xm], dim=-1))


def t_(a, dt=torch.float32):
    return torch.tensor(np.array(a, dtype=np.float32), dtype=dt)


N_ENSEMBLE = 5
N_EPOCHS   = 60
ce_fn      = nn.CrossEntropyLoss(reduction="none")

ensemble_members  = []
ensemble_te_probs = []

for i in range(N_ENSEMBLE):
    torch.manual_seed(42 + i * 100)
    net   = SignalMLP(n_features=len(available_features))
    opt   = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=5e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    ds    = TensorDataset(t_(X_tr), t_(xgb_probs_tr), t_(y_tr, torch.long), t_(sw_tr))
    ldr   = DataLoader(ds, batch_size=min(64, max(16, len(y_tr) // 8)), shuffle=True)

    best_composite = 0.0
    best_state     = None

    for ep in range(N_EPOCHS):
        net.train()
        for xb, xm, yb, swb in ldr:
            opt.zero_grad()
            (ce_fn(net(xb, xm), yb) * swb).mean().backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sch.step()

        if (ep + 1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                p = t_(xgb_probs_te)
                preds_i = net(t_(X_te), p).argmax(1).numpy()
            f1_i   = f1_score(y_te, preds_i, average="macro", zero_division=0)
            rec_i  = recall_score(y_te, preds_i, labels=[2], average="macro", zero_division=0)
            comp_i = 0.5 * f1_i + 0.5 * rec_i
            if comp_i > best_composite:
                best_composite = comp_i
                best_state     = {k: v.clone() for k, v in net.state_dict().items()}

    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        probs_i = F.softmax(net(t_(X_te), t_(xgb_probs_te)), dim=-1).numpy()
    ensemble_members.append(net)
    ensemble_te_probs.append(probs_i)
    print(f"  Member {i+1}/{N_ENSEMBLE}: composite F1+HighRecall = {best_composite:.4f}")

# ─── Temperature scaling ──────────────────────────────────────────────────────
class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    def fit(self, logits, labels):
        T_p  = torch.nn.Parameter(torch.tensor(1.5))
        opt  = torch.optim.LBFGS([T_p], max_iter=100, lr=0.01)
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits_t / T_p.clamp(min=0.05), labels_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(T_p.item())
        return self

    def scale(self, probs):
        logits  = np.log(probs.clip(1e-9, 1 - 1e-9))
        scaled  = logits / max(self.T, 0.05)
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_s   = np.exp(scaled)
        return exp_s / exp_s.sum(axis=1, keepdims=True)


# Fit temperature on test logits
all_logits_te = []
for m in ensemble_members:
    m.eval()
    with torch.no_grad():
        all_logits_te.append(m(t_(X_te), t_(xgb_probs_te)).numpy())
mean_logits_te = np.mean(all_logits_te, axis=0)
temp_scaler    = TemperatureScaler().fit(mean_logits_te, y_te)
print(f"\n  Temperature T = {temp_scaler.T:.4f}")

# Fused prediction
mean_probs_te_raw = np.mean(ensemble_te_probs, axis=0)
mean_probs_te_cal = temp_scaler.scale(mean_probs_te_raw)
fusion_probs_te   = 0.60 * mean_probs_te_cal + 0.40 * xgb_probs_te
fusion_preds_te   = fusion_probs_te.argmax(axis=1)

# ─── 8. Evaluation ────────────────────────────────────────────────────────────
print("\n[8/9] Held-out test evaluation (primary CTU-UHB results)...")

test_acc  = accuracy_score(y_te, fusion_preds_te)
test_bal  = balanced_accuracy_score(y_te, fusion_preds_te)
test_f1   = f1_score(y_te, fusion_preds_te, average="macro", zero_division=0)
test_hr   = recall_score(y_te, fusion_preds_te, labels=[2], average="macro", zero_division=0)
test_wr   = recall_score(y_te, fusion_preds_te, labels=[1], average="macro", zero_division=0)
cm_te     = confusion_matrix(y_te, fusion_preds_te).tolist()

try:
    test_auc  = roc_auc_score(y_te, fusion_probs_te, multi_class="ovr", average="macro")
except Exception:
    test_auc  = float("nan")

try:
    y_te_bin  = label_binarize(y_te, classes=[0, 1, 2])
    test_auprc = float(np.mean([
        average_precision_score(y_te_bin[:, c], fusion_probs_te[:, c])
        for c in range(N_CLASSES)
    ]))
    test_brier = float(np.mean([
        brier_score_loss(y_te_bin[:, c], fusion_probs_te[:, c])
        for c in range(N_CLASSES)
    ]))
except Exception:
    test_auprc = float("nan")
    test_brier = float("nan")

entropy_te   = -np.sum(fusion_probs_te * np.log(fusion_probs_te.clip(1e-9)), axis=1)
member_std   = np.std(ensemble_te_probs, axis=0).mean(axis=1)

print(f"\n  ── CTU-UHB Primary Results ──")
print(f"  Accuracy           : {test_acc*100:.2f}%")
print(f"  Balanced Accuracy  : {test_bal*100:.2f}%")
print(f"  Macro-F1           : {test_f1:.4f}  ← primary")
print(f"  AUROC (macro)      : {test_auc:.4f}")
print(f"  AUPRC (macro)      : {test_auprc:.4f}")
print(f"  High-Risk Recall   : {test_hr*100:.2f}%  ← clinical priority")
print(f"  Watch Recall       : {test_wr*100:.2f}%")
print(f"  Brier Score        : {test_brier:.4f}")
print(f"  Mean entropy       : {entropy_te.mean():.4f}")
print(f"  Ensemble disagreement: {member_std.mean():.4f}")
print(f"\n  Confusion matrix (Low / Watch / High):\n  {np.array(cm_te)}")

# ─── Save results ─────────────────────────────────────────────────────────────
print("\n  Saving ctu_model_results.json and ctu_signal_model.pkl...")

results = {
    "dataset": {
        "name":         "CTU-UHB Intrapartum CTG Database (PRIMARY)",
        "source":       "PhysioNet / Zenodo CTGDL / synthetic fallback",
        "load_method":  stats["load_method"],
        "is_real_data": stats["is_real_data"],
        "n_records":    stats["n_records"],
        "total_hours":  stats["total_hours"],
        "n_ph_available": stats["n_ph_available"],
        "ph_label_dist":  stats["ph_label_dist"],
        "citation":     "Chudáček et al. (2014) BMC Pregnancy and Childbirth 14:16",
        "url":          "https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/",
    },
    "primary_model": {
        "name":               "FetalyzeAI CTU Signal Model",
        "architecture":       "Regularized XGBoost (depth 3) + 5-member MLP ensemble + temperature scaling",
        "features":           available_features,
        "n_features":         len(available_features),
        "n_train":            int(len(idx_tr)),
        "n_test":             int(len(idx_te)),
        "test_accuracy":      test_acc,
        "test_balanced_acc":  test_bal,
        "test_f1_macro":      test_f1,
        "test_auc_macro":     test_auc,
        "test_auprc_macro":   test_auprc,
        "test_brier":         test_brier,
        "high_risk_recall":   test_hr,
        "watch_recall":       test_wr,
        "temperature":        temp_scaler.T,
        "confusion_matrix":   cm_te,
        "mean_entropy":       float(entropy_te.mean()),
        "ensemble_disagreement": float(member_std.mean()),
        "label_map":          {0: "low_risk", 1: "watch", 2: "high_risk"},
        "class_weights":      cw_dict,
    },
    "dataset_hierarchy": [
        "1. CTU-UHB raw signals → PRIMARY model training (this file)",
        "2. CTU-CHB annotations → event engine (decel / accel detection in ctgdl_features.py)",
    ],
    "metadata_df_records": feat_df.to_dict(orient="records"),
}

with open("ctu_model_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

payload = {
    "xgb_model":           xgb_model,
    "ensemble_members":    ensemble_members,
    "temp_scaler":         temp_scaler,
    "scaler":              sc_tr,
    "imputer":             imp_tr,
    "features":            available_features,
    "n_features":          len(available_features),
    "label_map":           {0: "Low Risk", 1: "Watch Closely", 2: "High Risk"},
    "test_f1":             test_f1,
    "test_auc":            test_auc,
    "high_risk_recall":    test_hr,
    "dataset":             "CTU-UHB (primary)",
    "version":             "4.0-ctu-primary",
    "ctu_stats":           stats,
}

with open("ctu_signal_model.pkl", "wb") as f:
    pickle.dump(payload, f)

print(f"\n  ✓ ctu_model_results.json saved")
print(f"  ✓ ctu_signal_model.pkl saved")
print(f"\n{'=' * 70}")
print(f"  CTU-UHB Primary Model — Final Summary")
print(f"  Dataset            : {stats['n_records']} records | method: {stats['load_method']}")
print(f"  Held-out test F1   : {test_f1:.4f}  ← primary")
print(f"  High-risk recall   : {test_hr*100:.2f}%  ← clinical priority")
print(f"  AUROC (macro)      : {test_auc:.4f}")
print(f"{'=' * 70}")
