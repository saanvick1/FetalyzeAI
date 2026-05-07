"""
pulsefm_encoder.py
==================
Optional CNN-TCN encoder for raw CTG windows ("PulseFM").

This is intentionally small (~50k params) and remains optional —
the main ReserveNet pipeline trains and the dashboard runs even when
PyTorch is unavailable. If torch is missing, importing PulseEncoder
raises ImportError and callers should treat it as opt-in.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:                         # pragma: no cover
    _TORCH_AVAILABLE = False


def torch_available() -> bool:
    return _TORCH_AVAILABLE


if _TORCH_AVAILABLE:

    class _DilatedBlock(nn.Module):
        def __init__(self, ch: int, dilation: int, dropout: float = 0.15):
            super().__init__()
            pad = dilation
            self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=pad, dilation=dilation)
            self.bn   = nn.BatchNorm1d(ch)
            self.act  = nn.GELU()
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return self.drop(self.act(self.bn(self.conv(x)))) + x

    class PulseEncoder(nn.Module):
        """
        Input shape: (B, C, T) where C ∈ {2,3,4}:
          [FHR, UC, missingness_mask, signal_quality?]
        Output: (B, 128) latent embedding.
        """

        def __init__(self, in_channels: int = 3, embed_dim: int = 128):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.GELU(),
            )
            self.tcn = nn.Sequential(
                _DilatedBlock(64, 1),
                _DilatedBlock(64, 2),
                _DilatedBlock(64, 4),
                _DilatedBlock(64, 8),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, embed_dim),
            )

        def forward(self, x):
            return self.head(self.tcn(self.stem(x)))


    class MaskedCTGAutoencoder(nn.Module):
        """Optional masked-CTG self-supervised pretraining wrapper."""

        def __init__(self, encoder: "PulseEncoder", out_channels: int = 2,
                     window_len: int = 1200):
            super().__init__()
            self.encoder = encoder
            self.window_len = window_len
            self.decoder = nn.Sequential(
                nn.Linear(128, 128 * 4),
                nn.GELU(),
                nn.Linear(128 * 4, out_channels * window_len),
            )

        @staticmethod
        def random_mask(x: "torch.Tensor", mask_ratio: float = 0.25):
            B, C, T = x.shape
            mask = (torch.rand(B, 1, T, device=x.device) < mask_ratio)
            x_masked = x.masked_fill(mask, 0.0)
            return x_masked, mask

        def forward(self, x):
            x_masked, mask = self.random_mask(x)
            z = self.encoder(x_masked)
            recon = self.decoder(z).view(x.shape[0], -1, self.window_len)
            return recon, mask

else:
    class PulseEncoder:                    # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is not installed; PulseEncoder is unavailable.")

    class MaskedCTGAutoencoder:            # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is not installed; MaskedCTGAutoencoder is unavailable.")


def window_to_tensor(fhr: np.ndarray, uc: np.ndarray):
    """Stack [FHR, UC, missingness_mask] into a (1, 3, T) tensor."""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch not installed.")
    fhr = np.asarray(fhr, dtype=np.float32)
    uc  = np.asarray(uc,  dtype=np.float32)
    mask = np.isnan(fhr).astype(np.float32)
    fhr  = np.where(np.isnan(fhr), 0.0, fhr) / 200.0
    uc   = np.where(np.isnan(uc),  0.0, uc)  / 100.0
    arr  = np.stack([fhr, uc, mask], axis=0)[None, ...]
    return torch.from_numpy(arr)
