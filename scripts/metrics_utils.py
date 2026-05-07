"""
FetalyzeAI — Shared Metrics Utility
Used by both training scripts and (via JSON export) the React frontend.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    recall_score, precision_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss,
)
from sklearn.preprocessing import label_binarize


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    preds = probs.argmax(axis=1)
    conf  = probs.max(axis=1)
    correct = (preds == y_true).astype(float)
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() > 0:
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_binary_at_risk(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.35) -> dict:
    """Binary at-risk metrics: classes 1+2 = positive."""
    y_bin_true = (y_true >= 1).astype(int)
    prob_pos   = probs[:, 1] + probs[:, 2]
    preds_bin  = (prob_pos >= threshold).astype(int)
    return {
        "auroc_binary":   float(roc_auc_score(y_bin_true, prob_pos)),
        "auprc_binary":   float(average_precision_score(y_bin_true, prob_pos)),
        "sensitivity":    float(recall_score(y_bin_true, preds_bin, zero_division=0)),
        "specificity":    float(recall_score(y_bin_true, preds_bin, pos_label=0, zero_division=0)),
        "precision":      float(precision_score(y_bin_true, preds_bin, zero_division=0)),
        "f1_binary":      float(f1_score(y_bin_true, preds_bin, zero_division=0)),
        "threshold_used": threshold,
    }


def compute_all_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.35) -> dict:
    preds  = probs.argmax(axis=1)
    y_bin  = label_binarize(y_true, classes=[0, 1, 2])

    try:
        if len(np.unique(y_true)) == 3:
            auroc_macro = float(roc_auc_score(y_bin, probs, multi_class="ovr", average="macro"))
        else:
            auroc_macro = float("nan")
    except Exception:
        auroc_macro = float("nan")

    metrics = {
        "n_samples":          int(len(y_true)),
        "accuracy":           float(accuracy_score(y_true, preds)),
        "balanced_accuracy":  float(balanced_accuracy_score(y_true, preds)),
        "macro_f1":           float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1":        float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "high_risk_recall":   float(recall_score(y_true, preds, labels=[2], average="macro", zero_division=0)),
        "watch_recall":       float(recall_score(y_true, preds, labels=[1], average="macro", zero_division=0)),
        "normal_recall":      float(recall_score(y_true, preds, labels=[0], average="macro", zero_division=0)),
        "auroc_macro":        auroc_macro,
        "ece":                expected_calibration_error(y_true, probs),
        "confusion_matrix":   confusion_matrix(y_true, preds, labels=[0, 1, 2]).tolist(),
    }
    metrics.update(compute_binary_at_risk(y_true, probs, threshold))
    return metrics


def bootstrap_metric(y_true: np.ndarray, probs: np.ndarray,
                     metric_fn, n_boot: int = 200, seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(metric_fn(y_true[idx], probs[idx]))
        except Exception:
            pass
    if not scores:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    arr = np.array(scores)
    return {
        "mean":  float(np.mean(arr)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
    }
