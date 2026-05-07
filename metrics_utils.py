"""
metrics_utils.py
================
Honest evaluation metrics + bootstrap confidence intervals for ReserveNet.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    recall_score, precision_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss,
)
from sklearn.preprocessing import label_binarize


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    preds   = probs.argmax(axis=1)
    conf    = probs.max(axis=1)
    correct = (preds == y_true).astype(float)
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() > 0:
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_binary_at_risk(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.35) -> dict:
    y_bin_true = (y_true >= 1).astype(int)
    prob_pos   = probs[:, 1] + probs[:, 2]
    preds_bin  = (prob_pos >= threshold).astype(int)
    out = {
        "sensitivity":    float(recall_score(y_bin_true, preds_bin, zero_division=0)),
        "specificity":    float(recall_score(y_bin_true, preds_bin, pos_label=0, zero_division=0)),
        "precision":      float(precision_score(y_bin_true, preds_bin, zero_division=0)),
        "f1_binary":      float(f1_score(y_bin_true, preds_bin, zero_division=0)),
        "threshold_used": threshold,
    }
    try:
        out["auroc_binary"] = float(roc_auc_score(y_bin_true, prob_pos))
        out["auprc_binary"] = float(average_precision_score(y_bin_true, prob_pos))
    except Exception:
        out["auroc_binary"] = float("nan")
        out["auprc_binary"] = float("nan")
    return out


def compute_all_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.35) -> dict:
    y_true = np.asarray(y_true)
    preds  = probs.argmax(axis=1)
    y_bin  = label_binarize(y_true, classes=[0, 1, 2])

    try:
        if len(np.unique(y_true)) >= 2 and y_bin.shape[1] >= 2:
            auroc_macro = float(roc_auc_score(y_bin, probs, multi_class="ovr", average="macro"))
            auprc_macro = float(average_precision_score(y_bin, probs, average="macro"))
        else:
            auroc_macro = auprc_macro = float("nan")
    except Exception:
        auroc_macro = auprc_macro = float("nan")

    # Per-class probs for Brier
    try:
        brier = float(np.mean(np.sum((probs - y_bin) ** 2, axis=1)))
    except Exception:
        brier = float("nan")

    cm = confusion_matrix(y_true, preds, labels=[0, 1, 2])
    high_fn = int(cm[2, 0] + cm[2, 1]) if cm.shape == (3, 3) else 0
    high_total = int(cm[2].sum()) if cm.shape == (3, 3) else 0
    high_fnr = float(high_fn / high_total) if high_total else float("nan")

    metrics = {
        "n_samples":          int(len(y_true)),
        "accuracy":           float(accuracy_score(y_true, preds)),
        "balanced_accuracy":  float(balanced_accuracy_score(y_true, preds)),
        "macro_f1":           float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1":        float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "high_risk_recall":   float(recall_score(y_true, preds, labels=[2], average="macro", zero_division=0)),
        "watch_recall":       float(recall_score(y_true, preds, labels=[1], average="macro", zero_division=0)),
        "low_risk_recall":    float(recall_score(y_true, preds, labels=[0], average="macro", zero_division=0)),
        "high_risk_precision":float(precision_score(y_true, preds, labels=[2], average="macro", zero_division=0)),
        "high_risk_fn_count": high_fn,
        "high_risk_fn_rate":  high_fnr,
        "auroc_macro":        auroc_macro,
        "auprc_macro":        auprc_macro,
        "brier":              brier,
        "ece":                expected_calibration_error(y_true, probs),
        "confusion_matrix":   cm.tolist(),
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


def bootstrap_confidence_intervals(y_true: np.ndarray, probs: np.ndarray,
                                   n_bootstrap: int = 200, seed: int = 42) -> dict:
    """Bootstrap CIs for balanced accuracy, macro-F1, and high-risk recall."""
    return {
        "balanced_accuracy": bootstrap_metric(
            y_true, probs,
            lambda y, p: balanced_accuracy_score(y, p.argmax(1)),
            n_boot=n_bootstrap, seed=seed,
        ),
        "macro_f1": bootstrap_metric(
            y_true, probs,
            lambda y, p: f1_score(y, p.argmax(1), average="macro", zero_division=0),
            n_boot=n_bootstrap, seed=seed,
        ),
        "high_risk_recall": bootstrap_metric(
            y_true, probs,
            lambda y, p: recall_score(y, p.argmax(1), labels=[2],
                                      average="macro", zero_division=0),
            n_boot=n_bootstrap, seed=seed,
        ),
    }
