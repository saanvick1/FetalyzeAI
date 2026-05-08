"""
reservenet_model.py
===================
FetalyzeAI ReserveNet — backward-compatible re-export of AdaptiveReserveNet v3.0.

The v2.0 ReserveNet class is kept here for import compatibility with any existing
code that does `from reservenet_model import ReserveNet`.

AdaptiveReserveNet v3.0 improvements vs v2.0:
  - 4th expert: Temporal Trend specialist (last-30-min delta features)
  - Attention-gated fusion (learnable MLP, not heuristic confidence weighting)
  - CNN-TCN PulseEncoder integrated when PyTorch available
  - Incremental adaptation: partial_fit() + replay buffer + EWC
  - Conformal prediction sets for calibrated uncertainty
  - Model registry support
"""

from __future__ import annotations

# ── v3.0 — preferred ─────────────────────────────────────────────────────────
from adaptive_reservenet import (
    AdaptiveReserveNet,
    TemperatureScaler,
    AttentionGating,
    ConformalCalibrator,
    ReplayBuffer,
    EXPERT_GROUPS,
    TOP_FEATURES,
)

# Alias so all existing imports of ReserveNet get the upgraded class
ReserveNet = AdaptiveReserveNet

__all__ = [
    "ReserveNet",
    "AdaptiveReserveNet",
    "TemperatureScaler",
    "AttentionGating",
    "ConformalCalibrator",
    "ReplayBuffer",
    "EXPERT_GROUPS",
    "TOP_FEATURES",
]
