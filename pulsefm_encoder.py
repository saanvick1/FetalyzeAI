"""
pulsefm_encoder.py
==================
FetalyzeAI PulseFM — CNN-TCN waveform encoder + full PulseFM-ReserveNet model.

Architecture (spec §18):
  ┌─ PulseEncoder (per window) ────────────────────────────────────────────┐
  │  Conv1D(3→32, k=9) → BN → GELU → Dropout(0.15)                        │
  │  Conv1D(32→64, k=7) → BN → GELU                                        │
  │  TCN dilation 1 → 2 → 4 → 8  (depthwise residual, k=3 each)           │
  │  Global average pool → Dense(64→128)  =  z_ij ∈ R^128                 │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ AttentionPooling (windows → record) ───────────────────────────────────┐
  │  α_ij = softmax(aᵀ tanh(V z_ij))                                       │
  │  z_i  = Σⱼ α_ij z_ij          =  z_i ∈ R^128                          │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ GatedReserveFusion ────────────────────────────────────────────────────┐
  │  g_i  = sigmoid(Wg [z_i ; h_i])                                        │
  │  r_i  = g_i ⊙ z_i  +  (1 - g_i) ⊙ Wh h_i                             │
  │  p_i  = softmax(Wr r_i + br)   ∈  R^3                                  │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ EnsemblePulseFM (5 members + MC-Dropout) ──────────────────────────────┐
  │  p̄    = (1/K) Σ_k p_k(y|x)                                             │
  │  U    = α H(p̄)  +  β (1/K) Σ_k ||p_k - p̄||²                          │
  └─────────────────────────────────────────────────────────────────────────┘

All classes are wrapped so the module imports cleanly even when PyTorch is
absent.  Callers must check torch_available() before instantiating any class.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

FS          = 4          # CTU-CHB sampling rate (Hz)
WINDOW_LEN  = 1200       # 5 minutes × 60 s × 4 Hz
EMBED_DIM   = 128        # latent dimension


def torch_available() -> bool:
    return _TORCH_AVAILABLE


# ─────────────────────────────────────────────────────────────────────────────
# Torch-dependent classes
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:

    # ── Building block: dilated TCN residual block ────────────────────────────
    class _DilatedBlock(nn.Module):
        """
        Dilated causal-style residual block:
          Conv1D(ch→ch, k=3, dil=d, pad=d) → BN → GELU → Dropout → + input
        padding=dilation achieves same-length output for k=3.
        """
        def __init__(self, ch: int, dilation: int, dropout: float = 0.15):
            super().__init__()
            self.conv = nn.Conv1d(ch, ch, kernel_size=3,
                                  padding=dilation, dilation=dilation)
            self.bn   = nn.BatchNorm1d(ch)
            self.act  = nn.GELU()
            self.drop = nn.Dropout(dropout)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.drop(self.act(self.bn(self.conv(x)))) + x

    # ── PulseEncoder ──────────────────────────────────────────────────────────
    class PulseEncoder(nn.Module):
        """
        Per-window CNN-TCN encoder.

        Input:  (B, C, T)  C ∈ {2, 3, 4}  [FHR, UC, missingness_mask, ...]
        Output: (B, 128)   latent embedding z
        """
        def __init__(self, in_channels: int = 3, embed_dim: int = EMBED_DIM,
                     dropout: float = 0.15):
            super().__init__()
            self.embed_dim = embed_dim
            self.stem = nn.Sequential(
                # Layer 1: Conv1D(C→32, k=9) — captures ~2.25 s context at 4 Hz
                nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Dropout(dropout),
                # Layer 2: Conv1D(32→64, k=7)
                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.GELU(),
            )
            # TCN stack — receptive field doubles with each block
            self.tcn = nn.Sequential(
                _DilatedBlock(64, dilation=1, dropout=dropout),   # ±1 step
                _DilatedBlock(64, dilation=2, dropout=dropout),   # ±2 steps
                _DilatedBlock(64, dilation=4, dropout=dropout),   # ±4 steps
                _DilatedBlock(64, dilation=8, dropout=dropout),   # ±8 steps
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),   # (B, 64, 1)
                nn.Flatten(),              # (B, 64)
                nn.Linear(64, embed_dim), # (B, 128)
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.head(self.tcn(self.stem(x)))

    # ── MaskedCTGAutoencoder (Stage-1 self-supervised pretraining) ────────────
    class MaskedCTGAutoencoder(nn.Module):
        """
        Self-supervised pretraining: randomly mask 25% of the signal and
        train the encoder to reconstruct the missing regions.

        Loss (spec §7):
            L_recon = || (1 - M) ⊙ (X - X̂) ||²   (only hidden positions)

        Input:  (B, C, T)
        Output: recon (B, 2, T)   [FHR + UC only],  mask (B, 1, T)
        """
        def __init__(self, encoder: "PulseEncoder",
                     out_channels: int = 2,
                     window_len: int = WINDOW_LEN,
                     mask_ratio: float = 0.25):
            super().__init__()
            self.encoder    = encoder
            self.window_len = window_len
            self.mask_ratio = mask_ratio
            embed = encoder.embed_dim
            self.decoder = nn.Sequential(
                nn.Linear(embed, embed * 4),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(embed * 4, out_channels * window_len),
            )

        @staticmethod
        def _block_mask(x: "torch.Tensor", mask_ratio: float = 0.25,
                        block_len: int = 20) -> "torch.Tensor":
            """
            Block-wise random mask so contiguous segments are hidden
            (more realistic than i.i.d. masking for physiological signals).
            Returns boolean mask (B, 1, T) where True = hidden.
            """
            B, C, T = x.shape
            mask = torch.zeros(B, 1, T, dtype=torch.bool, device=x.device)
            n_blocks = max(1, int(T * mask_ratio / block_len))
            for b in range(B):
                for _ in range(n_blocks):
                    start = torch.randint(0, max(1, T - block_len), (1,)).item()
                    mask[b, 0, start: start + block_len] = True
            return mask

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            mask    = self._block_mask(x, self.mask_ratio)
            x_masked = x.clone()
            x_masked[mask.expand_as(x)] = 0.0
            z     = self.encoder(x_masked)
            recon = self.decoder(z).view(x.shape[0], -1, self.window_len)
            return recon, mask

    # ── AttentionPooling ──────────────────────────────────────────────────────
    class AttentionPooling(nn.Module):
        """
        Attention-based window-to-record aggregation (spec §12):
            α_ij = softmax(aᵀ tanh(V z_ij))
            z_i  = Σⱼ α_ij z_ij

        Input:  (N_windows, embed_dim)  or  (B, N_windows, embed_dim)
        Output: (embed_dim,)            or  (B, embed_dim)
        """
        def __init__(self, embed_dim: int = EMBED_DIM, hidden: int = 128):
            super().__init__()
            self.V = nn.Linear(embed_dim, hidden, bias=False)
            self.a = nn.Linear(hidden, 1,         bias=False)

        def forward(self, Z: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            # Z: (N, D) or (B, N, D)
            scores  = self.a(torch.tanh(self.V(Z)))   # (..., N, 1)
            weights = torch.softmax(scores, dim=-2)   # (..., N, 1)
            pooled  = (weights * Z).sum(dim=-2)        # (..., D)
            return pooled, weights.squeeze(-1)

    # ── GatedReserveFusion ────────────────────────────────────────────────────
    class GatedReserveFusion(nn.Module):
        """
        Gated fusion of waveform embedding z and clinical feature vector h
        (spec §13):
            g  = sigmoid(Wg [z ; h])
            r  = g ⊙ z  +  (1 - g) ⊙ (Wh h)
            p  = softmax(Wr r)

        Intuition: poor signal quality → gate trusts clinical features more.
        """
        def __init__(self, embed_dim: int = EMBED_DIM, feat_dim: int = 40,
                     n_classes: int = 3, dropout: float = 0.20):
            super().__init__()
            self.embed_dim = embed_dim
            self.feat_dim  = feat_dim
            # Gate: [z; h] → embed_dim-dimensional gate
            self.gate = nn.Sequential(
                nn.Linear(embed_dim + feat_dim, embed_dim),
                nn.Sigmoid(),
            )
            # Clinical feature projection: h → embed_dim
            self.Wh = nn.Sequential(
                nn.Linear(feat_dim,  embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            # Classifier: r → logits
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, n_classes),
            )

        def forward(self, z: "torch.Tensor", h: "torch.Tensor") \
                -> tuple["torch.Tensor", "torch.Tensor"]:
            g = self.gate(torch.cat([z, h], dim=-1))   # (B, embed_dim)
            r = g * z + (1 - g) * self.Wh(h)          # (B, embed_dim)
            logits = self.classifier(r)                 # (B, n_classes)
            return logits, r

    # ── PulseFMReserveNet ─────────────────────────────────────────────────────
    class PulseFMReserveNet(nn.Module):
        """
        Full end-to-end PulseFM-ReserveNet model for a single record.

        Forward signature:
            windows : (N_windows, C, L)   — windowed CTG tensor
            features: (feat_dim,)          — clinical feature vector

        Returns:
            logits  : (n_classes,)
            probs   : (n_classes,)
            r       : (embed_dim,)         — fused representation
            attn_w  : (N_windows,)         — attention weights over windows
        """
        def __init__(self, in_channels: int = 3, embed_dim: int = EMBED_DIM,
                     feat_dim: int = 40, n_classes: int = 3,
                     dropout: float = 0.20):
            super().__init__()
            self.encoder = PulseEncoder(in_channels, embed_dim, dropout=0.15)
            self.pooling  = AttentionPooling(embed_dim, hidden=embed_dim)
            self.fusion   = GatedReserveFusion(embed_dim, feat_dim, n_classes, dropout)

        def forward(self, windows: "torch.Tensor", features: "torch.Tensor") \
                -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            # Encode all windows
            B_w = windows.shape[0]
            z_wins = self.encoder(windows)         # (N_windows, embed_dim)
            # Attention pool to record embedding
            z_rec, attn_w = self.pooling(z_wins)   # (embed_dim,), (N_windows,)
            z_rec = z_rec.unsqueeze(0)             # (1, embed_dim)
            h = features.unsqueeze(0)              # (1, feat_dim)
            logits, r = self.fusion(z_rec, h)      # (1, n_classes)
            probs = torch.softmax(logits, dim=-1)
            return logits.squeeze(0), probs.squeeze(0), r.squeeze(0), attn_w

        def predict(self, windows: "torch.Tensor", features: "torch.Tensor",
                    mc_samples: int = 0) -> dict:
            """Convenience inference; mc_samples>0 enables MC-Dropout uncertainty."""
            if mc_samples > 0:
                self.train()  # keep dropout active
                all_probs = []
                with torch.no_grad():
                    for _ in range(mc_samples):
                        _, p, _, _ = self.forward(windows, features)
                        all_probs.append(p.unsqueeze(0))
                self.eval()
                stack = torch.cat(all_probs, dim=0)  # (mc_samples, n_classes)
                p_bar = stack.mean(dim=0)
                var   = ((stack - p_bar.unsqueeze(0)) ** 2).mean(dim=0).sum()
                return {"probs": p_bar, "variance": float(var)}
            else:
                self.eval()
                with torch.no_grad():
                    _, p, _, attn = self.forward(windows, features)
                return {"probs": p, "attn": attn}

    # ── EnsemblePulseFM ───────────────────────────────────────────────────────
    class EnsemblePulseFM:
        """
        5-member PulseFM-ReserveNet ensemble.

        Uncertainty (spec §16):
            p̄   = (1/K) Σ_k p_k
            H   = -Σ_c p̄_c log p̄_c   (entropy)
            Var = (1/K) Σ_k ||p_k - p̄||²
            U   = 0.6 × H_norm + 0.4 × Var_norm

        Parameters
        ----------
        models   : list of PulseFMReserveNet (len = K, default 5)
        temp_T   : temperature for scaling (optimised on validation)
        """

        def __init__(self, models: list, temp_T: float = 1.0):
            self.models = models
            self.temp_T = max(temp_T, 0.05)
            self._fitted = True

        def _scale(self, logits: "torch.Tensor") -> "torch.Tensor":
            return torch.softmax(logits / self.temp_T, dim=-1)

        def predict(self, windows: "torch.Tensor",
                    features: "torch.Tensor") -> dict:
            """
            Returns calibrated ensemble prediction with uncertainty.
            """
            all_probs = []
            for m in self.models:
                m.eval()
                with torch.no_grad():
                    logits, _, _, _ = m(windows, features)
                    all_probs.append(self._scale(logits).unsqueeze(0))

            stack = torch.cat(all_probs, dim=0)        # (K, n_classes)
            p_bar = stack.mean(dim=0)                  # (n_classes,)

            # Entropy
            safe_p = torch.clamp(p_bar, 1e-7, 1.0)
            entropy = float(-(safe_p * torch.log(safe_p)).sum())
            entropy_norm = entropy / np.log(p_bar.shape[0])   # 0–1

            # Ensemble variance (disagreement)
            var = float(((stack - p_bar.unsqueeze(0)) ** 2).mean(0).sum())
            var_norm = min(var * 4.0, 1.0)    # rough normalisation

            # Combined uncertainty
            uncertainty = min(0.60 * entropy_norm + 0.40 * var_norm, 1.0)

            pred = int(p_bar.argmax())
            return {
                "probs":       p_bar.tolist(),
                "pred":        pred,
                "confidence":  float(p_bar[pred]),
                "entropy":     round(entropy_norm, 4),
                "variance":    round(var_norm, 4),
                "uncertainty": round(uncertainty, 4),
            }

        def save(self, path: str) -> None:
            import pickle
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self, f)

        @classmethod
        def load(cls, path: str) -> "EnsemblePulseFM":
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)

else:
    # ── Graceful stubs when torch is not installed ────────────────────────────
    class PulseEncoder:
        def __init__(self, *a, **kw):
            raise ImportError("torch is not installed; PulseEncoder is unavailable.")

    class MaskedCTGAutoencoder:
        def __init__(self, *a, **kw):
            raise ImportError("torch is not installed; MaskedCTGAutoencoder is unavailable.")

    class AttentionPooling:
        def __init__(self, *a, **kw):
            raise ImportError("torch is not installed; AttentionPooling is unavailable.")

    class GatedReserveFusion:
        def __init__(self, *a, **kw):
            raise ImportError("torch is not installed; GatedReserveFusion is unavailable.")

    class PulseFMReserveNet:
        def __init__(self, *a, **kw):
            raise ImportError("torch is not installed; PulseFMReserveNet is unavailable.")

    class EnsemblePulseFM:
        def __init__(self, *a, **kw):
            raise ImportError("torch is not installed; EnsemblePulseFM is unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# Numpy helpers (no torch dependency)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_window(fhr: np.ndarray, uc: np.ndarray,
                      window_len: int = WINDOW_LEN,
                      fs: int = FS) -> np.ndarray:
    """
    Normalise a 5-minute CTG window into a (3, window_len) float32 array.
        channel 0: FHR / 200      (nan → 0)
        channel 1: UC  / 100      (nan → 0)
        channel 2: FHR missingness mask (1 = missing)
    """
    fhr = np.asarray(fhr, dtype=np.float32)[:window_len]
    uc  = np.asarray(uc,  dtype=np.float32)[:window_len]
    # Pad if shorter than window_len
    if len(fhr) < window_len:
        fhr = np.pad(fhr, (0, window_len - len(fhr)), constant_values=np.nan)
        uc  = np.pad(uc,  (0, window_len - len(uc)),  constant_values=np.nan)
    mask = np.isnan(fhr).astype(np.float32)
    fhr  = np.where(np.isnan(fhr), 0.0, fhr) / 200.0
    uc   = np.where(np.isnan(uc),  0.0, uc)  / 100.0
    return np.stack([fhr, uc, mask], axis=0)      # (3, window_len)


def extract_windows(fhr: np.ndarray, uc: np.ndarray,
                    window_len: int = WINDOW_LEN,
                    stride: int = WINDOW_LEN,
                    fs: int = FS) -> np.ndarray:
    """
    Slide a window over the full CTG record.

    Returns: (N_windows, 3, window_len) float32 array.
    stride = window_len for non-overlapping (supervised).
    stride = window_len // 2 for 50% overlap (pretraining, more windows).
    """
    n = min(len(fhr), len(uc))
    windows = []
    start = 0
    while start + window_len <= n:
        w = preprocess_window(fhr[start:start + window_len],
                               uc [start:start + window_len], window_len, fs)
        windows.append(w)
        start += stride
    if not windows:
        # Record shorter than one window — use whatever we have
        windows.append(preprocess_window(fhr[:window_len], uc[:window_len],
                                          window_len, fs))
    return np.stack(windows, axis=0).astype(np.float32)  # (N, 3, L)


def window_to_tensor(fhr: np.ndarray, uc: np.ndarray,
                     window_len: int = WINDOW_LEN):
    """Return a single window as a (1, 3, window_len) torch tensor."""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch not installed.")
    import torch
    arr = preprocess_window(fhr, uc, window_len)
    return torch.from_numpy(arr[None, ...])  # (1, 3, L)


def record_to_tensors(fhr: np.ndarray, uc: np.ndarray,
                      clinical_features: np.ndarray,
                      window_len: int = WINDOW_LEN,
                      stride: int = WINDOW_LEN):
    """
    Convert a full CTG record to (windows_tensor, features_tensor).

    Returns:
        windows_tensor  : torch.Tensor (N_windows, 3, window_len)
        features_tensor : torch.Tensor (feat_dim,)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch not installed.")
    import torch
    wins = extract_windows(fhr, uc, window_len, stride)
    return (torch.from_numpy(wins),
            torch.from_numpy(np.asarray(clinical_features, dtype=np.float32)))
