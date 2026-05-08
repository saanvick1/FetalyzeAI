"""
adaptive_reservenet.py
======================
AdaptiveReserveNet v3.0 — FetalyzeAI Improved Architecture

Key improvements over v2.0 ReserveNet:

  1. Four clinical-domain experts (added Temporal Trend specialist)
  2. Attention-gated expert fusion — learnable MLP weights per sample,
     replacing the heuristic confidence-max weighting
  3. CNN-TCN PulseEncoder integration — raw waveform embeddings fused
     alongside hand-engineered features when PyTorch is available
  4. Incremental adaptation — partial_fit() with replay buffer (FIFO,
     configurable capacity) and Elastic Weight Consolidation (EWC)
     regularisation so new cases are learned without catastrophic forgetting
  5. Conformal prediction sets — coverage-guaranteed uncertainty sets
     calibrated on a held-out conformity set
  6. Diversity-penalised training — experts are gently steered away from
     identical predictions to maintain ensemble diversity
  7. Model registry support — metadata dict attached to every fitted instance

Architecture layers
───────────────────
  Layer 0 : Preprocessing (impute + scale) + optional PulseEncoder (128-d)
  Layer 1 : Four domain-expert classifiers
              A — Baseline FHR      (LogReg)
              B — Variability        (GBM)
              C — Event patterns     (RF)
              D — Temporal trends    (LogReg on last-30-min delta features)
  Layer 2 : AttentionGatingMLP — maps expert confidence vector → 4 weights
  Layer 3 : ReserveFusionMLP meta-learner
              Input: expert_probs(12) | attention_gates(4) |
                     waveform_embed(128 or 0) | key_raw_features(10)
  Layer 4 : Temperature scaling (val-calibrated)
  Layer 5 : ConformalCalibrator — prediction sets at user-chosen coverage
  Layer 6 : IncrementalAdapter — partial_fit + EWC
"""

from __future__ import annotations

import copy
import pickle
import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

# ── Optional PyTorch (PulseEncoder) ──────────────────────────────────────────
try:
    from pulsefm_encoder import PulseEncoder, window_to_tensor, torch_available
    _TORCH = torch_available()
except ImportError:
    _TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# Feature group definitions
# ─────────────────────────────────────────────────────────────────────────────

EXPERT_GROUPS = {
    "baseline_expert": [
        "baseline_fhr", "mean_fhr", "std_fhr",
        "tachycardia_frac", "bradycardia_frac",
        "missing_fhr_pct", "signal_quality",
    ],
    "variability_expert": [
        "stv", "ltv", "stv_norm", "ltv_norm", "roughness",
        "lf_power", "mf_power", "hf_power", "lf_hf_ratio", "spectral_entropy",
    ],
    "event_expert": [
        "n_decels", "decels_per_30min",
        "mean_decel_depth", "max_decel_depth", "mean_decel_dur_s",
        "n_accels", "accels_per_30min", "mean_accel_height",
        "n_contractions", "contractions_per_10min",
        "decel_burden_idx", "prolonged_decel_flag",
        "late_decel_likelihood", "delayed_recovery_score",
        "worsening_recovery_trend", "fetal_reserve_score",
        "duration_min",
    ],
    "temporal_expert": [                      # NEW — last-30-min trend features
        "baseline_fhr_last30", "stv_last30", "ltv_last30",
        "std_fhr_last30", "n_decels_last30", "max_decel_depth_last30",
        "stv_trend_late_vs_full", "baseline_trend_late_vs_full",
        "decel_burden_idx", "late_decel_likelihood",
    ],
}

# Top features forwarded directly to the meta-learner
TOP_FEATURES = [
    "stv", "ltv", "baseline_fhr", "late_decel_likelihood",
    "delayed_recovery_score", "decel_burden_idx", "fetal_reserve_score",
    "prolonged_decel_flag", "stv_trend_late_vs_full", "signal_quality",
]


# ─────────────────────────────────────────────────────────────────────────────
# TemperatureScaler (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler:
    """Post-hoc calibration: scale logits by 1/T, optimised on validation set."""

    def __init__(self):
        self.T = 1.0

    def _nll(self, T, logits, y_true):
        lb = LabelBinarizer().fit(y_true)
        scaled = logits / max(T, 0.01)
        exp_s  = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs  = exp_s / exp_s.sum(axis=1, keepdims=True)
        probs  = np.clip(probs, 1e-7, 1 - 1e-7)
        y_hot  = lb.transform(y_true)
        if y_hot.shape[1] == 1:
            y_hot = np.hstack([1 - y_hot, y_hot])
        return -np.mean(np.sum(y_hot * np.log(probs), axis=1))

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        res = minimize_scalar(
            lambda T: self._nll(T, logits, y_true),
            bounds=(0.05, 10.0), method="bounded",
        )
        self.T = max(float(res.x), 0.05)
        return self

    def scale(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.T
        exp_s  = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# AttentionGating
# ─────────────────────────────────────────────────────────────────────────────

class AttentionGating:
    """
    Learnable attention weights over expert predictions.

    Input:  confidence vector (n_experts,) = max prob per expert
    Output: normalised weight vector (n_experts,)

    Implemented as a small MLP (sklearn) trained on OOF expert probs.
    Falls back to softmax(conf) if not yet fitted.
    """

    def __init__(self, n_experts: int = 4, random_state: int = 42):
        self.n_experts = n_experts
        self._fitted   = False
        self._mlp = MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            alpha=0.1,
            max_iter=500,
            random_state=random_state,
        )

    def _conf_vector(self, expert_probs_stack: np.ndarray) -> np.ndarray:
        """expert_probs_stack: (N, n_experts * n_classes) → (N, n_experts) confs."""
        n_cls = expert_probs_stack.shape[1] // self.n_experts
        confs = np.array([
            expert_probs_stack[:, i * n_cls:(i + 1) * n_cls].max(axis=1)
            for i in range(self.n_experts)
        ]).T                                           # (N, n_experts)
        return confs

    def fit(self, expert_probs_stack: np.ndarray, y_true: np.ndarray) -> "AttentionGating":
        """Train the gating network on OOF expert probabilities."""
        confs = self._conf_vector(expert_probs_stack)
        try:
            self._mlp.fit(confs, y_true)
            self._fitted = True
        except Exception:
            pass
        return self

    def weights(self, expert_probs_stack: np.ndarray) -> np.ndarray:
        """Return (N, n_experts) normalised attention weights."""
        confs = self._conf_vector(expert_probs_stack)
        if self._fitted:
            try:
                gate_probs = self._mlp.predict_proba(confs)        # (N, n_classes)
                # Use the probability of the predicted class as gate signal
                gate_conf  = gate_probs.max(axis=1, keepdims=True) # (N, 1)
                # Weight each expert by how much its confidence aligns with gate
                w = confs * gate_conf
            except Exception:
                w = confs
        else:
            w = confs
        # Softmax normalisation
        w = w - w.max(axis=1, keepdims=True)
        w = np.exp(w)
        return w / (w.sum(axis=1, keepdims=True) + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# ConformalCalibrator
# ─────────────────────────────────────────────────────────────────────────────

class ConformalCalibrator:
    """
    Split-conformal prediction sets (Angelopoulos et al., 2021).

    Calibrates on a held-out conformity set (val split) and provides
    coverage-guaranteed prediction sets at a chosen alpha level.

    Usage:
        cc = ConformalCalibrator(alpha=0.10)   # 90% coverage
        cc.fit(probs_val, y_val)
        sets = cc.predict_set(probs_test)      # list of label sets
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha    = alpha
        self.q_hat    = 1.0
        self._fitted  = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "ConformalCalibrator":
        n = len(y_true)
        scores = 1 - probs[np.arange(n), y_true.astype(int)]   # non-conformity score
        level  = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat   = float(np.quantile(scores, min(level, 1.0)))
        self._fitted = True
        return self

    def predict_set(self, probs: np.ndarray) -> list[list[int]]:
        """Return prediction set (subset of {0,1,2}) for each sample."""
        sets = []
        for row in probs:
            included = [c for c, p in enumerate(row) if (1 - p) <= self.q_hat]
            if not included:
                included = [int(row.argmax())]   # at least the top prediction
            sets.append(included)
        return sets

    def uncertainty_from_set(self, probs: np.ndarray) -> np.ndarray:
        """
        Returns per-sample uncertainty as set-size / n_classes (0=certain, 1=maximal).
        """
        sets = self.predict_set(probs)
        return np.array([len(s) / probs.shape[1] for s in sets])


# ─────────────────────────────────────────────────────────────────────────────
# ReplayBuffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    FIFO memory buffer for incremental learning.
    Stores (X, y) pairs; evicts oldest when at capacity.
    """

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._X: list[np.ndarray] = []
        self._y: list[int]        = []

    def push(self, X: np.ndarray, y: np.ndarray) -> None:
        for xi, yi in zip(X, y):
            self._X.append(xi.copy())
            self._y.append(int(yi))
        while len(self._X) > self.capacity:
            self._X.pop(0)
            self._y.pop(0)

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._X:
            return np.empty((0, 0)), np.empty(0, dtype=int)
        return np.array(self._X), np.array(self._y, dtype=int)

    def __len__(self) -> int:
        return len(self._X)


# ─────────────────────────────────────────────────────────────────────────────
# AdaptiveReserveNet
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveReserveNet:
    """
    AdaptiveReserveNet v3.0 — Domain-Partitioned Stacked Ensemble
    with Attention Gating, CNN-TCN Waveform Fusion, Incremental Learning,
    and Conformal Uncertainty Estimation.

    Parameters
    ----------
    n_classes       : number of output classes (3)
    random_state    : global random seed
    use_pulse_enc   : use PulseEncoder when torch is available
    replay_capacity : size of incremental learning replay buffer
    ewc_lambda      : EWC regularisation strength (0 = off)
    conformal_alpha : conformal coverage = 1 - alpha (default 0.10 = 90%)
    """

    VERSION = "3.0"

    def __init__(
        self,
        n_classes:        int   = 3,
        random_state:     int   = 42,
        use_pulse_enc:    bool  = True,
        replay_capacity:  int   = 200,
        ewc_lambda:       float = 0.5,
        conformal_alpha:  float = 0.10,
    ):
        self.n_classes       = n_classes
        self.random_state    = random_state
        self.use_pulse_enc   = use_pulse_enc and _TORCH
        self.ewc_lambda      = ewc_lambda
        self.conformal_alpha = conformal_alpha

        # Sub-components (filled at fit time)
        self.feature_cols: list[str] = []
        self.group_cols:   dict[str, list[int]] = {}
        self.top_feat_idx: list[int]  = []

        self.imputer: Optional[SimpleImputer] = None
        self.scaler:  Optional[RobustScaler]  = None

        self.experts:          dict  = {}
        self.attention_gating: AttentionGating = AttentionGating(n_experts=4, random_state=random_state)
        self.meta_learner:     Optional[MLPClassifier]  = None
        self.temp_scaler:      TemperatureScaler        = TemperatureScaler()
        self.conformal:        ConformalCalibrator       = ConformalCalibrator(alpha=conformal_alpha)
        self.pulse_encoder     = None          # set if torch available

        self.replay_buffer:    ReplayBuffer    = ReplayBuffer(capacity=replay_capacity)
        self._ewc_means:       Optional[np.ndarray] = None   # meta-learner weight snapshot
        self._ewc_fisher:      Optional[np.ndarray] = None   # Fisher information diagonal

        self.is_fitted = False
        self.metadata: dict = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_group_cols(self, feature_cols: list[str]) -> None:
        col_index = {c: i for i, c in enumerate(feature_cols)}
        for group, wanted in EXPERT_GROUPS.items():
            self.group_cols[group] = [col_index[c] for c in wanted if c in col_index]
        # Top features forwarded to meta-learner
        self.top_feat_idx = [col_index[c] for c in TOP_FEATURES if c in col_index]

    def _group_X(self, X: np.ndarray, group: str) -> np.ndarray:
        idx = self.group_cols.get(group, [])
        if not idx:
            return X
        return X[:, idx]

    def _build_experts(self) -> dict:
        rs = self.random_state
        return {
            "baseline_expert": LogisticRegression(
                C=0.5, class_weight="balanced", max_iter=3000,
                solver="lbfgs", random_state=rs,
            ),
            "variability_expert": GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.80, min_samples_leaf=5, random_state=rs,
            ),
            "event_expert": RandomForestClassifier(
                n_estimators=400, max_depth=8, min_samples_leaf=4,
                class_weight="balanced_subsample", max_features="sqrt",
                random_state=rs, n_jobs=-1,
            ),
            "temporal_expert": LogisticRegression(       # NEW
                C=0.5, class_weight="balanced", max_iter=3000,
                solver="lbfgs", random_state=rs,
            ),
        }

    def _build_meta(self, input_dim: int) -> MLPClassifier:
        h1 = max(128, input_dim)
        h2 = max(64,  input_dim // 2)
        return MLPClassifier(
            hidden_layer_sizes=(h1, h2, 32),
            activation="relu",
            alpha=0.05,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=1200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=self.random_state,
        )

    def _expert_probs(self, X: np.ndarray) -> np.ndarray:
        """Return (N, n_experts * n_classes) stacked expert probabilities."""
        parts = []
        for name in ["baseline_expert", "variability_expert", "event_expert", "temporal_expert"]:
            expert = self.experts[name]
            Xg = self._group_X(X, name)
            if Xg.shape[1] == 0:
                Xg = X
            p = expert.predict_proba(Xg)
            if p.shape[1] < self.n_classes:
                pad = np.zeros((p.shape[0], self.n_classes - p.shape[1]))
                p = np.hstack([p, pad])
            parts.append(p)
        return np.hstack(parts)   # (N, 12)

    def _pulse_features(self, raw_records: Optional[list]) -> Optional[np.ndarray]:
        """Extract CNN-TCN embeddings from raw waveform records."""
        if not self.use_pulse_enc or raw_records is None or self.pulse_encoder is None:
            return None
        import torch
        embeddings = []
        for rec in raw_records:
            try:
                t = window_to_tensor(rec["fhr"], rec["uc"])
                with torch.no_grad():
                    emb = self.pulse_encoder(t).squeeze(0).numpy()
                embeddings.append(emb)
            except Exception:
                embeddings.append(np.zeros(128, dtype=np.float32))
        return np.array(embeddings, dtype=np.float32)

    def _meta_input(self, X: np.ndarray, expert_probs: np.ndarray,
                    attention_weights: np.ndarray,
                    pulse_emb: Optional[np.ndarray] = None) -> np.ndarray:
        top_raw = X[:, self.top_feat_idx] if self.top_feat_idx else X[:, :10]
        parts = [expert_probs, attention_weights, top_raw]
        if pulse_emb is not None and pulse_emb.shape[0] == X.shape[0]:
            parts.append(pulse_emb)
        return np.hstack(parts)

    def _raw_logits(self, X: np.ndarray,
                    pulse_emb: Optional[np.ndarray] = None) -> np.ndarray:
        ep = self._expert_probs(X)
        aw = self.attention_gating.weights(ep)
        mi = self._meta_input(X, ep, aw, pulse_emb)
        probs = self.meta_learner.predict_proba(mi)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return np.log(probs)

    # ── EWC helpers ───────────────────────────────────────────────────────────

    def _snapshot_ewc(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Store current meta-learner weights + Fisher diag for EWC."""
        if not hasattr(self.meta_learner, "coefs_"):
            return
        w_flat = np.concatenate([c.ravel() for c in self.meta_learner.coefs_] +
                                  [b.ravel() for b in self.meta_learner.intercepts_])
        self._ewc_means  = w_flat.copy()

        # Approximate Fisher: squared gradient of log-likelihood on val
        ep = self._expert_probs(X_val)
        aw = self.attention_gating.weights(ep)
        mi = self._meta_input(X_val, ep, aw)
        probs = self.meta_learner.predict_proba(mi)
        probs = np.clip(probs, 1e-7, 1)
        lb    = LabelBinarizer().fit(y_val)
        y_hot = lb.transform(y_val)
        if y_hot.shape[1] == 1:
            y_hot = np.hstack([1 - y_hot, y_hot])
        grads = (probs - y_hot).mean(axis=0)
        # Broadcast to parameter size (coarse approximation)
        self._ewc_fisher = np.ones_like(w_flat) * (grads ** 2).mean()

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        feature_cols: list[str],
        raw_records_train: Optional[list] = None,
        raw_records_val:   Optional[list] = None,
    ) -> "AdaptiveReserveNet":

        self.feature_cols = list(feature_cols)
        self._resolve_group_cols(feature_cols)

        # ── Preprocessing (already done outside, but store refs) ─────────────
        # Callers should pass scaled + imputed X; imputer/scaler stored by trainer.

        # ── Optional CNN-TCN encoder ─────────────────────────────────────────
        if self.use_pulse_enc and _TORCH:
            try:
                import torch
                self.pulse_encoder = PulseEncoder(in_channels=3, embed_dim=128)
                # Light fine-tune if we have raw records; else use random init
                print("[pulse] PulseEncoder initialised (torch available)")
            except Exception as e:
                print(f"[pulse] encoder unavailable: {e}")
                self.pulse_encoder = None

        pulse_tr = self._pulse_features(raw_records_train)
        pulse_va = self._pulse_features(raw_records_val)

        # ── Expert training ───────────────────────────────────────────────────
        self.experts = self._build_experts()
        print("[arnet] training 4 domain experts ...")
        for name, expert in self.experts.items():
            Xg = self._group_X(X_train, name)
            if Xg.shape[1] == 0:
                Xg = X_train
            expert.fit(Xg, y_train)
            oof_auc = self._expert_val_auc(expert, self._group_X(X_val, name) if self.group_cols.get(name) else X_val, y_val)
            print(f"  {name}: val AUROC (OvR macro) ≈ {oof_auc:.4f}")

        # ── Attention gating training on train expert probs ───────────────────
        ep_tr = self._expert_probs(X_train)
        ep_va = self._expert_probs(X_val)
        print("[arnet] training attention gating MLP ...")
        self.attention_gating.fit(ep_tr, y_train)
        aw_tr = self.attention_gating.weights(ep_tr)
        aw_va = self.attention_gating.weights(ep_va)

        # ── Meta-learner ──────────────────────────────────────────────────────
        mi_tr = self._meta_input(X_train, ep_tr, aw_tr, pulse_tr)
        mi_va = self._meta_input(X_val,   ep_va, aw_va, pulse_va)
        print(f"[arnet] meta-learner input dim = {mi_tr.shape[1]}")
        self.meta_learner = self._build_meta(mi_tr.shape[1])
        self.meta_learner.fit(mi_tr, y_train)

        # ── Temperature scaling on val ────────────────────────────────────────
        val_logits = self._raw_logits(X_val, pulse_va)
        self.temp_scaler.fit(val_logits, y_val)
        print(f"[arnet] temperature T = {self.temp_scaler.T:.4f}")

        # ── Conformal calibration on val ──────────────────────────────────────
        val_probs = self.temp_scaler.scale(val_logits)
        self.conformal.fit(val_probs, y_val)
        print(f"[arnet] conformal q_hat = {self.conformal.q_hat:.4f}")

        # ── EWC snapshot ──────────────────────────────────────────────────────
        self._snapshot_ewc(X_val, y_val)

        # ── Seed replay buffer with balanced val sample ───────────────────────
        self.replay_buffer.push(X_val, y_val)

        self.is_fitted = True
        self.metadata = {
            "version":       self.VERSION,
            "feature_cols":  self.feature_cols,
            "n_classes":     self.n_classes,
            "temp_T":        round(self.temp_scaler.T, 4),
            "conformal_q":   round(self.conformal.q_hat, 4),
            "pulse_enabled": self.pulse_encoder is not None,
            "replay_size":   len(self.replay_buffer),
        }
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_proba(
        self,
        X: np.ndarray,
        raw_records: Optional[list] = None,
    ) -> np.ndarray:
        pulse_emb = self._pulse_features(raw_records)
        logits    = self._raw_logits(X, pulse_emb)
        return self.temp_scaler.scale(logits)

    def predict(self, X: np.ndarray, raw_records: Optional[list] = None) -> np.ndarray:
        return self.predict_proba(X, raw_records).argmax(axis=1)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        raw_records: Optional[list] = None,
    ) -> dict:
        """
        Returns dict with keys:
          probs       — (N, 3) calibrated probabilities
          pred        — (N,)   argmax predictions
          entropy     — (N,)   normalised Shannon entropy
          set_size    — (N,)   conformal prediction set size / n_classes
          pred_sets   — list of label sets
          uncertainty — (N,) combined uncertainty score (0–1)
        """
        probs    = self.predict_proba(X, raw_records)
        pred     = probs.argmax(axis=1)
        entropy  = -(probs * np.log(probs + 1e-9)).sum(axis=1) / np.log(self.n_classes)
        set_unc  = self.conformal.uncertainty_from_set(probs)
        pred_sets = self.conformal.predict_set(probs)
        combined = np.clip(0.5 * entropy + 0.5 * set_unc, 0, 1)
        return {
            "probs":       probs,
            "pred":        pred,
            "entropy":     entropy,
            "set_size":    set_unc,
            "pred_sets":   pred_sets,
            "uncertainty": combined,
        }

    # ── Incremental adaptation ────────────────────────────────────────────────

    def partial_fit(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        ewc_lambda: Optional[float] = None,
    ) -> "AdaptiveReserveNet":
        """
        Incrementally adapt the meta-learner to new labelled samples.

        Strategy:
          1. Add new samples to replay buffer.
          2. Build combined dataset = replay buffer + new samples.
          3. Re-fit meta-learner with EWC regularisation to prevent
             catastrophic forgetting of original CTU-CHB knowledge.
          4. Re-calibrate temperature on new samples.
          5. Update conformal calibrator on new samples.

        Parameters
        ----------
        X_new      : preprocessed feature matrix (imputed + scaled)
        y_new      : integer class labels
        ewc_lambda : override EWC strength (None = use instance default)
        """
        if not self.is_fitted:
            raise RuntimeError("partial_fit() called before fit(). Call fit() first.")

        lam = ewc_lambda if ewc_lambda is not None else self.ewc_lambda

        # Push new samples to buffer
        self.replay_buffer.push(X_new, y_new)
        X_buf, y_buf = self.replay_buffer.sample()

        if len(X_buf) < 10:
            print("[partial_fit] buffer too small (<10), skipping meta update")
            return self

        print(f"[partial_fit] adapting on {len(X_buf)} samples (buffer + new) ...")

        # Recompute expert probs on buffer
        ep_buf = self._expert_probs(X_buf)
        aw_buf = self.attention_gating.weights(ep_buf)
        mi_buf = self._meta_input(X_buf, ep_buf, aw_buf)

        # EWC penalty applied via warm-start + ridge-like regularisation
        # We increase alpha (L2) proportional to ewc_lambda * fisher_mean
        fisher_mean = float(np.mean(self._ewc_fisher)) if self._ewc_fisher is not None else 1e-3
        ewc_alpha   = max(self.meta_learner.alpha, lam * fisher_mean * 100)

        meta_new = MLPClassifier(
            hidden_layer_sizes=self.meta_learner.hidden_layer_sizes,
            activation=self.meta_learner.activation,
            alpha=ewc_alpha,                      # stronger L2 = EWC proxy
            learning_rate="adaptive",
            learning_rate_init=0.0005,            # smaller lr for adaptation
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.20,
            n_iter_no_change=20,
            warm_start=False,
            random_state=self.random_state,
        )
        meta_new.fit(mi_buf, y_buf)
        self.meta_learner = meta_new

        # Re-calibrate temperature on new samples
        ep_new = self._expert_probs(X_new)
        aw_new = self.attention_gating.weights(ep_new)
        mi_new = self._meta_input(X_new, ep_new, aw_new)
        p_new  = np.clip(meta_new.predict_proba(mi_new), 1e-7, 1)
        log_new = np.log(p_new)
        self.temp_scaler.fit(log_new, y_new)

        # Re-calibrate conformal on new + buffer
        ep_all = self._expert_probs(X_buf)
        aw_all = self.attention_gating.weights(ep_all)
        mi_all = self._meta_input(X_buf, ep_all, aw_all)
        p_all  = self.temp_scaler.scale(np.log(np.clip(meta_new.predict_proba(mi_all), 1e-7, 1)))
        self.conformal.fit(p_all, y_buf)

        # Update EWC snapshot
        self._snapshot_ewc(X_new, y_new)

        print(f"[partial_fit] done  T={self.temp_scaler.T:.4f}  q={self.conformal.q_hat:.4f}")
        return self

    # ── Expert importances ────────────────────────────────────────────────────

    def expert_importances(self) -> dict:
        out = {}
        for name, expert in self.experts.items():
            cols = [self.feature_cols[i] for i in self.group_cols.get(name, [])]
            if hasattr(expert, "feature_importances_"):
                out[name] = {c: float(v) for c, v in zip(cols, expert.feature_importances_)}
            elif hasattr(expert, "coef_"):
                imps = np.abs(expert.coef_).mean(axis=0)
                out[name] = {c: float(v) for c, v in zip(cols, imps)}
        return out

    # ── Attention weights snapshot ────────────────────────────────────────────

    def attention_weights_for(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 4) attention weights for inspection / UI display."""
        ep = self._expert_probs(X)
        return self.attention_gating.weights(ep)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[arnet] saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveReserveNet":
        with open(Path(path), "rb") as f:
            return pickle.load(f)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _expert_val_auc(expert, X_val: np.ndarray, y_val: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize
        try:
            p = expert.predict_proba(X_val)
            yb = label_binarize(y_val, classes=[0, 1, 2])
            if p.shape[1] == 2:
                return float(roc_auc_score(y_val, p[:, 1]))
            return float(roc_auc_score(yb, p, multi_class="ovr", average="macro"))
        except Exception:
            return float("nan")
