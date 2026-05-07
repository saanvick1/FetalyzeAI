"""
FetalyzeAI ReserveNet — Model Architecture
===========================================

Domain-Partitioned Stacked Ensemble with Gated Clinical Fusion.

Architecture overview:
  Layer 0 — Three clinical-domain specialists (each on a clinically coherent subset):
    Expert A — FHR Baseline      : baseline stability, tachycardia/bradycardia rates
    Expert B — Variability       : short- and long-term FHR variability
    Expert C — Event Patterns    : decelerations, accelerations, contractions, burden
  Layer 1 — ReserveFusionMLP    : meta-learner combining expert probabilities + top raw features
  Post-fit — Temperature scaling : calibrated probabilities on validation set

Clinical motivation:
  Different CTG modules capture independent physiological mechanisms.
  Partitioning prevents cross-contamination between feature groups.
  The meta-learner learns which expert to trust per case.
  Temperature scaling ensures realistic probability estimates for clinical use.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.calibration import CalibratedClassifierCV


FEATURE_GROUPS = {
    "baseline_expert": [
        "baseline_fhr", "mean_fhr", "std_fhr",
        "tachycardia_frac", "bradycardia_frac",
        "missing_fhr", "signal_quality",
    ],
    "variability_expert": [
        "stv", "ltv",
        "stv_norm", "ltv_norm",
    ],
    "event_expert": [
        "n_decels", "decels_per_30min",
        "mean_decel_depth", "max_decel_depth", "mean_decel_dur_s",
        "n_accels", "accels_per_30min",
        "n_contractions", "contractions_per_10min",
        "decel_burden", "csr_frac",
        "duration_min",
    ],
    "all_features": None,   # filled at runtime
}


class TemperatureScaler:
    """Post-hoc calibration: scale logits by 1/T, optimised on validation set."""

    def __init__(self):
        self.T = 1.0

    def _nll(self, T, logits, y_true):
        lb = LabelBinarizer().fit(y_true)
        scaled = logits / T
        exp_s  = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs  = exp_s / exp_s.sum(axis=1, keepdims=True)
        probs  = np.clip(probs, 1e-7, 1 - 1e-7)
        y_hot  = lb.transform(y_true)
        if y_hot.shape[1] == 1:
            y_hot = np.hstack([1 - y_hot, y_hot])
        return -np.mean(np.sum(y_hot * np.log(probs), axis=1))

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        res = minimize_scalar(
            lambda T: self._nll(max(T, 0.01), logits, y_true),
            bounds=(0.05, 10.0), method="bounded",
        )
        self.T = max(float(res.x), 0.05)
        return self

    def scale(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.T
        exp_s  = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)


def _cols_present(df_cols, wanted):
    return [c for c in wanted if c in df_cols]


class ReserveNet:
    """
    Domain-Partitioned Stacked Ensemble with Temperature Scaling.

    Stages:
      1. Three domain specialists produce class probabilities.
      2. ReserveFusionMLP meta-learner ingests those probs + top raw features.
      3. Temperature scaler calibrates on validation set.
    """

    def __init__(self, n_classes: int = 3, random_state: int = 42):
        self.n_classes     = n_classes
        self.random_state  = random_state
        self.feature_cols  = None
        self.group_cols    = {}
        self.experts       = {}
        self.meta_learner  = None
        self.temp_scaler   = TemperatureScaler()
        self.is_fitted      = False

    def _build_experts(self):
        rs = self.random_state
        return {
            "baseline_expert": LogisticRegression(
                C=0.5, class_weight="balanced",
                max_iter=2000, random_state=rs,
            ),
            "variability_expert": LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=2000, random_state=rs,
            ),
            "event_expert": RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=6,
                class_weight="balanced", max_features="sqrt",
                random_state=rs, n_jobs=-1,
            ),
        }

    def _build_meta(self):
        return MLPClassifier(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            alpha=0.05,
            learning_rate="adaptive",
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=self.random_state,
        )

    def _group_X(self, X: np.ndarray, group: str) -> np.ndarray:
        idx = self.group_cols.get(group, list(range(X.shape[1])))
        return X[:, idx] if idx else X

    def _expert_probs(self, X: np.ndarray) -> np.ndarray:
        parts = []
        for name, expert in self.experts.items():
            Xg = self._group_X(X, name)
            parts.append(expert.predict_proba(Xg))
        return np.hstack(parts)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray,
            feature_cols: list) -> "ReserveNet":

        self.feature_cols = feature_cols
        col_index = {c: i for i, c in enumerate(feature_cols)}

        for group, wanted in FEATURE_GROUPS.items():
            if wanted is None:
                self.group_cols[group] = list(range(len(feature_cols)))
            else:
                self.group_cols[group] = [col_index[c] for c in wanted if c in col_index]

        self.experts = self._build_experts()
        for name, expert in self.experts.items():
            Xg = self._group_X(X_train, name)
            if Xg.shape[1] == 0:
                Xg = X_train
            expert.fit(Xg, y_train)

        meta_train = np.hstack([self._expert_probs(X_train), X_train])
        self.meta_learner = self._build_meta()
        self.meta_learner.fit(meta_train, y_train)

        val_logits = self._raw_logits(X_val)
        self.temp_scaler.fit(val_logits, y_val)

        self.is_fitted = True
        return self

    def _raw_logits(self, X: np.ndarray) -> np.ndarray:
        meta_X = np.hstack([self._expert_probs(X), X])
        probs  = self.meta_learner.predict_proba(meta_X)
        probs  = np.clip(probs, 1e-7, 1 - 1e-7)
        return np.log(probs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._raw_logits(X)
        return self.temp_scaler.scale(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def expert_importances(self) -> dict:
        out = {}
        for name, expert in self.experts.items():
            if hasattr(expert, "feature_importances_"):
                cols = [self.feature_cols[i] for i in self.group_cols.get(name, [])]
                imps = expert.feature_importances_
                out[name] = {c: float(v) for c, v in zip(cols, imps)}
            elif hasattr(expert, "coef_"):
                cols = [self.feature_cols[i] for i in self.group_cols.get(name, [])]
                imps = np.abs(expert.coef_).mean(axis=0)
                out[name] = {c: float(v) for c, v in zip(cols, imps)}
        return out
