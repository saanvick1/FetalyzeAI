"""
model_registry.py
=================
FetalyzeAI Model Registry — version tracking, rollback, and A/B comparison.

Each registered version stores:
  - model artifact  (pickle)
  - evaluation metrics (JSON)
  - training config snapshot
  - timestamp + version ID

Usage
-----
    from model_registry import ModelRegistry
    reg = ModelRegistry("models/registry")
    vid = reg.save(model, metrics, config)
    m   = reg.load(vid)
    reg.compare("v1", "v2")
    reg.rollback("v1")
"""

from __future__ import annotations

import json
import pickle
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class ModelRegistry:
    """
    Filesystem-backed model registry.

    Directory layout
    ─────────────────
    <root>/
      registry.json          ← index of all versions
      <version_id>/
        model.pkl            ← serialised model
        metrics.json         ← evaluation metrics
        config.json          ← training configuration
    """

    INDEX_FILE = "registry.json"

    def __init__(self, root: str | Path = "models/registry"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict = self._load_index()

    # ── Index helpers ─────────────────────────────────────────────────────────

    def _index_path(self) -> Path:
        return self.root / self.INDEX_FILE

    def _load_index(self) -> dict:
        p = self._index_path()
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {"versions": [], "active": None}

    def _save_index(self) -> None:
        with open(self._index_path(), "w") as f:
            json.dump(self._index, f, indent=2)

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(
        self,
        model,
        metrics: dict,
        config:  dict,
        tag: str = "",
    ) -> str:
        """
        Persist a trained model and its metadata.

        Returns the version ID string (e.g. "v3_20260508_143201").
        """
        ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        vid = f"v{getattr(model, 'VERSION', '?')}_{ts}"
        if tag:
            vid = f"{vid}_{tag}"

        version_dir = self.root / vid
        version_dir.mkdir(parents=True, exist_ok=True)

        # Serialise model
        model_path = version_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metrics + config
        with open(version_dir / "metrics.json", "w") as f:
            json.dump(_json_safe(metrics), f, indent=2)
        with open(version_dir / "config.json", "w") as f:
            json.dump(_json_safe(config), f, indent=2)

        # Model fingerprint (SHA256 of pkl bytes)
        sha = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]

        entry = {
            "version_id":    vid,
            "timestamp":     ts,
            "model_version": getattr(model, "VERSION", "?"),
            "tag":           tag,
            "sha256":        sha,
            "key_metrics":   _extract_key_metrics(metrics),
            "model_path":    str(model_path.relative_to(self.root)),
            "is_active":     False,
        }
        self._index["versions"].append(entry)
        self._set_active(vid)
        self._save_index()
        print(f"[registry] saved {vid}  sha={sha}  active=True")
        return vid

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(self, version_id: Optional[str] = None):
        """
        Load model by version ID.
        If version_id is None, loads the active version.
        """
        vid = version_id or self._index.get("active")
        if not vid:
            raise ValueError("No version ID provided and no active version set.")
        model_path = self.root / vid / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"[registry] loaded {vid}")
        return model

    # ── Rollback ──────────────────────────────────────────────────────────────

    def rollback(self, version_id: str) -> None:
        """Set an older version as the active model."""
        ids = [v["version_id"] for v in self._index["versions"]]
        if version_id not in ids:
            raise ValueError(f"Version '{version_id}' not found. Available: {ids}")
        self._set_active(version_id)
        self._save_index()
        print(f"[registry] rolled back → {version_id}")

    # ── Compare ───────────────────────────────────────────────────────────────

    def compare(self, vid_a: str, vid_b: str) -> dict:
        """
        Compare key metrics between two versions.
        Returns a dict with per-metric delta (B − A).
        """
        def _get(vid):
            p = self.root / vid / "metrics.json"
            if not p.exists():
                return {}
            with open(p) as f:
                return json.load(f)

        ma, mb = _get(vid_a), _get(vid_b)
        common = set(ma) & set(mb)
        result = {
            "versions":  [vid_a, vid_b],
            "metrics_a": {k: ma[k] for k in common},
            "metrics_b": {k: mb[k] for k in common},
            "delta":     {},
        }
        for k in common:
            try:
                result["delta"][k] = round(float(mb[k]) - float(ma[k]), 6)
            except (TypeError, ValueError):
                pass
        return result

    # ── List ─────────────────────────────────────────────────────────────────

    def list_versions(self) -> list[dict]:
        """Return all registered versions, newest first."""
        return list(reversed(self._index["versions"]))

    def active_version(self) -> Optional[str]:
        return self._index.get("active")

    def summary(self) -> None:
        """Print a compact table of registered versions."""
        versions = self.list_versions()
        if not versions:
            print("[registry] No versions registered.")
            return
        print(f"\n{'Version':<30} {'Timestamp':<18} {'AUROC':>7} {'Sens':>7} {'Active':<8}")
        print("-" * 75)
        for v in versions:
            km  = v.get("key_metrics", {})
            aur = km.get("auroc", "—")
            sen = km.get("sensitivity", "—")
            act = "✓" if v["version_id"] == self._index.get("active") else ""
            aur_s = f"{aur:.4f}" if isinstance(aur, float) else str(aur)
            sen_s = f"{sen:.4f}" if isinstance(sen, float) else str(sen)
            print(f"{v['version_id']:<30} {v['timestamp']:<18} {aur_s:>7} {sen_s:>7} {act:<8}")

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, version_id: str, force: bool = False) -> None:
        """Delete a version. Active version cannot be deleted without force=True."""
        if version_id == self._index.get("active") and not force:
            raise ValueError(f"Cannot delete active version '{version_id}'. "
                             "Use force=True or rollback first.")
        version_dir = self.root / version_id
        if version_dir.exists():
            shutil.rmtree(version_dir)
        self._index["versions"] = [v for v in self._index["versions"]
                                   if v["version_id"] != version_id]
        if self._index.get("active") == version_id:
            self._index["active"] = None
        self._save_index()
        print(f"[registry] deleted {version_id}")

    # ── Private ───────────────────────────────────────────────────────────────

    def _set_active(self, vid: str) -> None:
        for v in self._index["versions"]:
            v["is_active"] = (v["version_id"] == vid)
        self._index["active"] = vid

    # ── Export registry as JSON summary ───────────────────────────────────────

    def export_summary(self, path: str | Path) -> None:
        """Write a clean JSON summary suitable for the frontend Research panel."""
        summary = {
            "active_version": self._index.get("active"),
            "n_versions":     len(self._index["versions"]),
            "versions":       [
                {
                    "id":      v["version_id"],
                    "ts":      v["timestamp"],
                    "metrics": v.get("key_metrics", {}),
                    "active":  v["is_active"],
                }
                for v in self.list_versions()
            ],
        }
        with open(Path(path), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[registry] summary exported → {path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_safe(obj):
    """Recursively convert numpy/float types to JSON-serialisable Python."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(float(obj), 6)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def _extract_key_metrics(metrics: dict) -> dict:
    """Pull the most important scalar metrics for the registry index."""
    keys = [
        "auroc", "auroc_binary", "sensitivity", "specificity",
        "f1_binary", "macro_f1", "balanced_accuracy", "auprc_binary",
        "brier", "ece",
    ]
    out = {}
    for k in keys:
        v = metrics.get(k)
        if v is not None:
            try:
                out[k] = round(float(v), 4)
            except (TypeError, ValueError):
                pass
    return out
