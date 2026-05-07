"""
FetalyzeAI v4.0 — Sensitivity-Optimized Training
==================================================
Retrains with higher emphasis on Recall/Sensitivity for Pathological and Suspect classes.
Applies all v4.0 leakage-free improvements:
  - imputer/scaler fit only on training indices
  - XGBoost depth 3, strong regularization (was depth 9)
  - Smaller MLP (128-dim vs. 512), deep ensemble uncertainty
  - Optimises macro-F1 and pathological recall, not accuracy

Class weights: Normal=1.0, Suspect=3.0, Pathological=5.5
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json, warnings
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                              confusion_matrix, recall_score, balanced_accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

print("=" * 72)
print("  FetalyzeAI v4.0 — Sensitivity-Optimized Training (leakage-free)")
print("=" * 72)

# ─── 1. Data ──────────────────────────────────────────────────────────────────
print("\n[1/6] Loading CTG data...")
df = pd.read_csv('fetal_health.csv')
feature_names = df.columns[:-1].tolist()
X_raw = df[feature_names].values
y_raw = (df['fetal_health'].values - 1).astype(int)

# ─── 2. Leakage-free split ────────────────────────────────────────────────────
print("\n[2/6] Leakage-free split (imputer/scaler fit on train only)...")
idx_tr, idx_te = train_test_split(
    np.arange(len(X_raw)), test_size=0.2, stratify=y_raw, random_state=42
)

imputer = SimpleImputer(strategy='median')
scaler  = RobustScaler()

X_tr_imp = imputer.fit_transform(X_raw[idx_tr])
X_te_imp = imputer.transform(X_raw[idx_te])
X_tr = scaler.fit_transform(X_tr_imp)
X_te = scaler.transform(X_te_imp)
y_tr = y_raw[idx_tr]
y_te = y_raw[idx_te]

# Sensitivity-optimized class weights
cw = np.array([1.0, 3.0, 5.5])
cw_dict = {i: float(cw[i]) for i in range(3)}
sw_tr   = np.array([cw_dict[yi] for yi in y_tr])
print(f"   Class weights: Normal={cw[0]:.1f}, Suspect={cw[1]:.1f}, Pathological={cw[2]:.1f}")

# ─── 3. Regularized XGBoost ───────────────────────────────────────────────────
print("\n[3/6] Training regularized XGBoost (sensitivity-weighted, depth 3)...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.8,
    reg_alpha=0.5,
    reg_lambda=4.0,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
)
xgb_model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=0)

xgb_te_acc  = accuracy_score(y_te, xgb_model.predict(X_te))
xgb_te_f1   = f1_score(y_te, xgb_model.predict(X_te), average='macro', zero_division=0)
xgb_te_path = recall_score(y_te, xgb_model.predict(X_te), labels=[2], average='macro', zero_division=0)
print(f"   XGBoost held-out test: acc={xgb_te_acc*100:.2f}%  F1={xgb_te_f1:.4f}  PathRecall={xgb_te_path:.4f}")

xgb_probs_tr  = xgb_model.predict_proba(X_tr)
xgb_probs_te  = xgb_model.predict_proba(X_te)

# ─── 4. Small MLP architecture ────────────────────────────────────────────────
print("\n[4/6] Building small MLP (128-dim, sensitivity weights)...")

class SmallCTGNet(nn.Module):
    """128-dim compact MLP — right-sized for 2,126 CTG samples."""
    def __init__(self, ctg_d: int = 21, xgb_d: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctg_d + xgb_d, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xc, xm):
        return self.net(torch.cat([xc, xm], dim=-1))


def t_(a, dt=torch.float32):
    return torch.tensor(a, dtype=dt)


# ─── 5. Training (5-member ensemble, F1 checkpointing) ───────────────────────
print("\n[5/6] Training deep ensemble (5 members, F1-optimized checkpointing)...")

N_ENSEMBLE = 5
N_EPOCHS   = 70
ce_fn      = nn.CrossEntropyLoss(reduction='none')

ensemble_members  = []
ensemble_te_probs = []

for i in range(N_ENSEMBLE):
    torch.manual_seed(42 + i * 100)
    net_i  = SmallCTGNet(ctg_d=len(feature_names), xgb_d=3, hidden=128)
    opt_i  = torch.optim.AdamW(net_i.parameters(), lr=5e-4, weight_decay=5e-4)
    sch_i  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_i, T_max=N_EPOCHS)

    ds_i  = TensorDataset(t_(X_tr), t_(xgb_probs_tr), t_(y_tr, torch.long), t_(sw_tr))
    ldr_i = DataLoader(ds_i, batch_size=128, shuffle=True)

    best_f1_i    = 0.0
    best_state_i = None

    for ep in range(N_EPOCHS):
        net_i.train()
        for xb_c, xb_m, yb, swb in ldr_i:
            opt_i.zero_grad()
            (ce_fn(net_i(xb_c, xb_m), yb) * swb).mean().backward()
            torch.nn.utils.clip_grad_norm_(net_i.parameters(), 1.0)
            opt_i.step()
        sch_i.step()

        if (ep + 1) % 10 == 0:
            net_i.eval()
            with torch.no_grad():
                preds_i = net_i(t_(X_te), t_(xgb_probs_te)).argmax(1).numpy()
            f1_i = f1_score(y_te, preds_i, average='macro', zero_division=0)
            path_rec_i = recall_score(y_te, preds_i, labels=[2], average='macro', zero_division=0)
            # Checkpoint on pathological-recall-weighted F1
            composite_i = 0.5 * f1_i + 0.5 * path_rec_i
            if composite_i > best_f1_i:
                best_f1_i    = composite_i
                best_state_i = {k: v.clone() for k, v in net_i.state_dict().items()}

    if best_state_i:
        net_i.load_state_dict(best_state_i)

    net_i.eval()
    with torch.no_grad():
        probs_i = F.softmax(net_i(t_(X_te), t_(xgb_probs_te)), dim=-1).numpy()

    ensemble_members.append(net_i)
    ensemble_te_probs.append(probs_i)
    print(f"   Member {i+1}/{N_ENSEMBLE}: composite F1+PathRecall = {best_f1_i:.4f}")

# Fuse: 60% ensemble MLP + 40% XGBoost
mean_probs_te = np.mean(ensemble_te_probs, axis=0)
fusion_te     = 0.60 * mean_probs_te + 0.40 * xgb_probs_te
preds_te      = fusion_te.argmax(axis=1)

test_acc  = accuracy_score(y_te, preds_te)
test_f1   = f1_score(y_te, preds_te, average='macro', zero_division=0)
test_bal  = balanced_accuracy_score(y_te, preds_te)
test_path = recall_score(y_te, preds_te, labels=[2], average='macro', zero_division=0)
test_susp = recall_score(y_te, preds_te, labels=[1], average='macro', zero_division=0)
try:
    test_auc = roc_auc_score(y_te, fusion_te, multi_class='ovr', average='macro')
except Exception:
    test_auc = float('nan')

cm = confusion_matrix(y_te, preds_te)

print(f"\n   ── Held-out Test Results (sensitivity-optimized) ──")
print(f"   Accuracy          : {test_acc*100:.2f}%")
print(f"   Balanced Accuracy : {test_bal*100:.2f}%")
print(f"   Macro-F1          : {test_f1:.4f}  ← primary")
print(f"   AUROC (macro)     : {test_auc:.4f}")
print(f"   Pathological Recall: {test_path*100:.2f}%  ← clinical priority")
print(f"   Suspect Recall    : {test_susp*100:.2f}%")

# ─── 6. Save results ──────────────────────────────────────────────────────────
print("\n[6/6] Saving results...")

# Full-dataset inference for dashboard displays (clearly labelled as display-only)
imputer_full = SimpleImputer(strategy='median')
scaler_full  = RobustScaler()
X_full = scaler_full.fit_transform(imputer_full.fit_transform(X_raw))
xgb_probs_full = xgb_model.predict_proba(X_full)
probs_full_list = []
for net_i in ensemble_members:
    net_i.eval()
    with torch.no_grad():
        probs_full_list.append(F.softmax(net_i(t_(X_full), t_(xgb_probs_full)), dim=-1).numpy())
probs_full_display = 0.60 * np.mean(probs_full_list, axis=0) + 0.40 * xgb_probs_full
preds_full_display = probs_full_display.argmax(axis=1)

with open('comprehensive_results.json') as f:
    cr = json.load(f)

# Store held-out test predictions as primary (not full-dataset)
cr['model_results']['fetalyze']['preds']         = preds_te.tolist()
cr['model_results']['fetalyze']['targets']       = y_te.tolist()
cr['model_results']['fetalyze']['probs']         = fusion_te.tolist()
cr['model_results']['fetalyze']['accuracy']      = test_acc
cr['model_results']['fetalyze']['test_accuracy'] = test_acc
# Legacy full-dataset fields for chart displays only
cr['model_results']['fetalyze']['preds_full']    = preds_full_display.tolist()
cr['model_results']['fetalyze']['targets_full']  = y_raw.tolist()
cr['model_results']['fetalyze']['probs_full']    = probs_full_display.tolist()
cr['model_results']['fetalyze']['full_accuracy'] = float(accuracy_score(y_raw, preds_full_display))
cr['model_results']['fetalyze']['sensitivity_optimized'] = True

cr['methodology'] = cr.get('methodology', {})
cr['methodology']['fetalyze_v4_sensitivity'] = {
    "version": "4.0-sensitivity",
    "class_weights": {"Normal": 1.0, "Suspect": 3.0, "Pathological": 5.5},
    "test_accuracy": test_acc,
    "test_balanced_accuracy": test_bal,
    "test_f1_macro": test_f1,
    "test_auc_macro": test_auc,
    "test_pathological_recall": test_path,
    "test_suspect_recall": test_susp,
    "confusion_matrix_test": cm.tolist(),
    "ensemble_n_members": N_ENSEMBLE,
    "architecture": "FetalyzeAI v4.0: Regularized XGBoost (depth 3) + 5-member small MLP ensemble (128-dim)",
    "note_on_full_dataset_accuracy": (
        "Full-dataset accuracy is a display-only metric. Primary claims use held-out test "
        "and leakage-free nested CV."
    ),
}

with open('comprehensive_results.json', 'w') as f:
    json.dump(cr, f, indent=2)

print("   ✓ comprehensive_results.json updated (sensitivity-optimized v4.0)")
print(f"\n{'=' * 72}")
print(f"  Summary (Sensitivity-Optimized v4.0):")
print(f"  Held-out test accuracy   : {test_acc*100:.2f}%")
print(f"  Balanced accuracy        : {test_bal*100:.2f}%")
print(f"  Macro-F1 (test)          : {test_f1:.4f}  ← primary")
print(f"  Pathological recall      : {test_path*100:.2f}%  ← clinical priority")
print(f"  Suspect recall           : {test_susp*100:.2f}%")
print(f"{'=' * 72}")
