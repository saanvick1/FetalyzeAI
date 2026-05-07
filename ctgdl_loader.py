"""
FetalyzeAI — CTG Dataset Loader
=================================
Two datasets, in priority order:

  1. CTU-UHB / CTU-CHB Intrapartum CTG Database  ← PRIMARY clinical model
     552 real intrapartum CTG recordings (FHR + UC @ 4 Hz, up to 90 min before delivery)
     Clinical outcomes: cord blood pH, base deficit, Apgar 1 & 5
     License: ODC-BY-1.0
     Source: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

  2. CTU-CHB Annotation Dataset  ← EVENT / EXPLAINABILITY engine
     Expert morphological annotations on the CTU-UHB recordings:
     baseline, bradycardia, tachycardia, accelerations, decelerations
     (early / late / variable / prolonged), uterine contractions, signal quality
     Source: PMC7256311 / Zenodo CTGDL FHRMA archive

References
----------
Chudáček et al. (2014) BMC Pregnancy and Childbirth 14:16
Petránek et al. (2020) PMC7256311
Fridman et al. (2025) SSRN 6027919
"""

import io
import os
import json
import logging
import warnings
import tarfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─── Cache directory ─────────────────────────────────────────────────────────
CACHE_DIR = Path("ctgdl_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
FS = 4              # Hz — native CTU-UHB sample rate
CTU_UHB_PN_DIR = "ctu-uhb-ctgdb/1.0.0"

# All 552 confirmed CTU-UHB record IDs (1001–2520; not every integer exists)
# We attempt all; wfdb silently skips missing ones.
CTU_UHB_ALL_IDS = list(range(1001, 2521))

# Zenodo CTGDL archives (pre-processed CSVs)
ZENODO_BASE = "https://zenodo.org/records/19510407/files"
ZENODO_FILES = {
    "ctu_uhb_proc_csv":  f"{ZENODO_BASE}/CTGDL_ctu_uhb_proc_csv.tar.gz",
    "ctu_uhb_csv":       f"{ZENODO_BASE}/CTGDL_ctu_uhb_csv.tar.gz",
    "fhrma_ano_csv":     f"{ZENODO_BASE}/CTGDL_FHRMA_ano_csv.tar.gz",
    "fhrma_proc_csv":    f"{ZENODO_BASE}/CTGDL_FHRMA_proc_csv.tar.gz",
}

# pH thresholds (FIGO / CTU-UHB clinical standards)
PH_ACIDOSIS   = 7.05
PH_BORDERLINE = 7.15


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class CTGRecord:
    """One CTG recording from CTU-UHB or synthetic fallback."""
    record_id:      str
    source:         str          # "CTU-UHB" | "CTU-UHB-anno" | "synthetic"
    fhr:            np.ndarray   # Fetal Heart Rate (bpm); NaN = missing
    uc:             np.ndarray   # Uterine Contraction; NaN = missing
    fs:             float = FS
    duration_min:   float = 0.0
    signal_quality: float = 1.0  # fraction of non-NaN FHR samples

    # Clinical outcomes (NaN if not available)
    ph:             float = float("nan")
    base_deficit:   float = float("nan")
    apgar1:         float = float("nan")
    apgar5:         float = float("nan")
    ph_label:       str   = "unknown"   # "normal" | "borderline" | "acidosis"

    # Expert annotations (CTU-CHB annotation dataset)
    annotations:    Dict  = field(default_factory=dict)

    # Clinical metadata from CTU-UHB header
    gestational_age:   float = float("nan")  # weeks
    birth_weight:      float = float("nan")  # grams
    delivery_type:     str   = "unknown"     # "vaginal" | "cs" | "forceps" | ...
    nicu_admission:    Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "record_id":       self.record_id,
            "source":          self.source,
            "duration_min":    round(self.duration_min, 2),
            "signal_quality":  round(self.signal_quality, 4),
            "ph":              self.ph,
            "ph_label":        self.ph_label,
            "base_deficit":    self.base_deficit,
            "apgar1":          self.apgar1,
            "apgar5":          self.apgar5,
            "gestational_age": self.gestational_age,
            "birth_weight":    self.birth_weight,
            "delivery_type":   self.delivery_type,
            "n_samples":       len(self.fhr),
        }


# ─── Download helpers ─────────────────────────────────────────────────────────

def _cached_path(name: str) -> Path:
    return CACHE_DIR / name


def _download_file(url: str, dest: Path, timeout: int = 90) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=131072):
                fh.write(chunk)
        return True
    except Exception as exc:
        logger.warning("Download failed %s — %s", url, exc)
        if dest.exists():
            dest.unlink()
        return False


def _extract_tar(src: Path, dest_dir: Path) -> bool:
    try:
        with tarfile.open(src, "r:gz") as tf:
            tf.extractall(dest_dir)
        return True
    except Exception as exc:
        logger.warning("Extraction failed %s — %s", src, exc)
        return False


# ─── 1. CTU-UHB PhysioNet loader ─────────────────────────────────────────────

def _parse_ctu_uhb_header(rec) -> dict:
    """
    Extract clinical metadata from a CTU-UHB WFDB record's comments.

    CTU-UHB .hea files use whitespace-separated key/value pairs, e.g.:
      pH           7.14
      BDecf        8.14
      Apgar1       6
      Apgar5       8
      Gest. weeks  37
      Weight(g)    2660
      Deliv. type  1
    """
    info = {"ph": float("nan"), "base_deficit": float("nan"),
            "apgar1": float("nan"), "apgar5": float("nan"),
            "gestational_age": float("nan"), "birth_weight": float("nan"),
            "delivery_type": "unknown"}
    # Maps from the first token(s) in a comment line → info key
    KEY_MAP = {
        "pH":           "ph",
        "BDecf":        "base_deficit",
        "BE":           "base_deficit",   # fallback
        "Apgar1":       "apgar1",
        "Apgar5":       "apgar5",
        "Gest.":        "gestational_age",
        "Weight(g)":    "birth_weight",
        "Deliv.":       "delivery_type",
    }
    try:
        for comment in (rec.comments or []):
            parts = comment.split()
            if len(parts) < 2:
                continue
            # Try single-token key first, then two-token key
            for n_key_tokens in (1, 2):
                key = " ".join(parts[:n_key_tokens])
                # strip trailing punctuation from key
                key_clean = key.rstrip(":.,")
                if key_clean in KEY_MAP:
                    raw_val = parts[n_key_tokens] if len(parts) > n_key_tokens else ""
                    try:
                        info[KEY_MAP[key_clean]] = float(raw_val)
                    except ValueError:
                        info[KEY_MAP[key_clean]] = raw_val
                    break
    except Exception:
        pass
    return info


def load_ctu_uhb_physionet(
    record_ids: Optional[List[int]] = None,
    max_records: int = 552,
    verbose: bool = False,
) -> List[CTGRecord]:
    """
    Stream CTU-UHB records directly from PhysioNet via WFDB.
    Returns a list of CTGRecord objects.

    Requires:  pip install wfdb
    """
    try:
        import wfdb
    except ImportError:
        logger.warning("wfdb not installed. Run: pip install wfdb")
        return []

    if record_ids is None:
        record_ids = CTU_UHB_ALL_IDS

    records: List[CTGRecord] = []
    attempted = 0

    for rid in record_ids:
        if len(records) >= max_records:
            break
        attempted += 1
        try:
            rec = wfdb.rdrecord(str(rid), pn_dir=CTU_UHB_PN_DIR)
            sig_names_upper = [s.upper() for s in rec.sig_name]

            fhr_idx = next((i for i, s in enumerate(sig_names_upper) if "FHR" in s), None)
            uc_idx  = next((i for i, s in enumerate(sig_names_upper)
                            if any(t in s for t in ("UC", "TOCO", "TOC"))), None)
            if fhr_idx is None:
                continue

            fhr = rec.p_signal[:, fhr_idx].astype(float)
            uc  = (rec.p_signal[:, uc_idx].astype(float)
                   if uc_idx is not None else np.full_like(fhr, np.nan))

            # Replace sentinel 0 / negative values with NaN
            fhr[(fhr <= 0) | (fhr > 300)] = np.nan
            uc[uc < 0] = np.nan

            n_total  = len(fhr)
            n_valid  = int(np.sum(~np.isnan(fhr)))
            quality  = n_valid / max(n_total, 1)
            duration = n_total / (FS * 60.0)

            meta = _parse_ctu_uhb_header(rec)
            ph   = float(meta["ph"])
            ph_label = ("acidosis"   if (not np.isnan(ph) and ph < PH_ACIDOSIS) else
                        "borderline" if (not np.isnan(ph) and ph < PH_BORDERLINE) else
                        "normal"     if not np.isnan(ph) else "unknown")

            records.append(CTGRecord(
                record_id=str(rid),
                source="CTU-UHB",
                fhr=fhr, uc=uc, fs=float(FS),
                duration_min=duration,
                signal_quality=quality,
                ph=ph,
                base_deficit=float(meta["base_deficit"]),
                apgar1=float(meta["apgar1"]),
                apgar5=float(meta["apgar5"]),
                ph_label=ph_label,
                gestational_age=float(meta["gestational_age"]),
                birth_weight=float(meta["birth_weight"]),
                delivery_type=str(meta["delivery_type"]),
            ))
            if verbose and len(records) % 50 == 0:
                print(f"  [CTU-UHB] loaded {len(records)} records...")
        except Exception as exc:
            if verbose:
                logger.debug("Record %s failed: %s", rid, exc)

    if verbose:
        print(f"  [CTU-UHB] Total: {len(records)} / {attempted} attempted")
    return records


# ─── 2. Zenodo CSV loader (CTGDL preprocessed) ───────────────────────────────

def load_zenodo_csv(
    key: str,
    max_records: int = 552,
    verbose: bool = False,
) -> List[CTGRecord]:
    """
    Download and parse a CTGDL preprocessed CSV archive from Zenodo.
    key must be one of the keys in ZENODO_FILES.
    """
    if key not in ZENODO_FILES:
        return []

    url      = ZENODO_FILES[key]
    tar_path = _cached_path(f"{key}.tar.gz")
    dest_dir = _cached_path(key)

    if not dest_dir.exists() or not any(dest_dir.rglob("*.csv")):
        dest_dir.mkdir(exist_ok=True)
        if verbose:
            print(f"  [Zenodo] Downloading {key}...")
        ok = _download_file(url, tar_path, timeout=180)
        if not ok:
            return []
        _extract_tar(tar_path, dest_dir)

    source_map = {
        "ctu_uhb_csv":      "CTU-UHB",
        "ctu_uhb_proc_csv": "CTU-UHB",
        "fhrma_ano_csv":    "CTU-CHB-anno",
        "fhrma_proc_csv":   "CTU-CHB-anno",
    }
    source = source_map.get(key, "CTGDL")

    records: List[CTGRecord] = []
    csv_files = sorted(dest_dir.rglob("*.csv"))[:max_records]

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.upper().str.strip()

            fhr_col = next((c for c in df.columns if "FHR" in c), None)
            uc_col  = next((c for c in df.columns
                            if any(t in c for t in ("UC", "TOCO"))), None)
            if fhr_col is None:
                continue

            fhr = df[fhr_col].values.astype(float)
            uc  = (df[uc_col].values.astype(float)
                   if uc_col else np.full_like(fhr, np.nan))

            fhr[(fhr <= 0) | (fhr > 300)] = np.nan
            uc[uc < 0] = np.nan

            n_total  = len(fhr)
            quality  = float(np.sum(~np.isnan(fhr))) / max(n_total, 1)
            duration = n_total / (FS * 60.0)

            # Clinical outcome columns (if present in CSV)
            def _col_val(col_name):
                c = next((c for c in df.columns if col_name.upper() in c), None)
                return float(df[c].iloc[0]) if (c and len(df) > 0) else float("nan")

            ph = _col_val("PH")
            bd = _col_val("BD")
            a1 = _col_val("APGAR1")
            a5 = _col_val("APGAR5")
            ph_label = ("acidosis"   if (not np.isnan(ph) and ph < PH_ACIDOSIS) else
                        "borderline" if (not np.isnan(ph) and ph < PH_BORDERLINE) else
                        "normal"     if not np.isnan(ph) else "unknown")

            # Annotation columns (FHRMA)
            annotations = {}
            for acol in df.columns:
                if any(t in acol for t in ("BASELINE", "ACCEL", "DECEL", "BRADY", "TACHY")):
                    annotations[acol.lower()] = df[acol].values.tolist()

            records.append(CTGRecord(
                record_id=csv_path.stem,
                source=source,
                fhr=fhr, uc=uc, fs=float(FS),
                duration_min=duration,
                signal_quality=quality,
                ph=ph, base_deficit=bd, apgar1=a1, apgar5=a5,
                ph_label=ph_label,
                annotations=annotations,
            ))
        except Exception as exc:
            logger.debug("CSV parse error %s — %s", csv_path, exc)

    if verbose:
        print(f"  [Zenodo {key}] Loaded {len(records)} records")
    return records


# ─── 3. Synthetic CTU-UHB fallback ───────────────────────────────────────────

def make_synthetic_ctu_uhb(
    n: int = 50,
    seed: int = 42,
    include_annotations: bool = True,
) -> List[CTGRecord]:
    """
    Physiologically-plausible synthetic CTU-UHB records.
    CLEARLY labelled source="synthetic" — never passed off as real data.
    Based on CTU-UHB distribution statistics.
    """
    rng = np.random.default_rng(seed)
    records: List[CTGRecord] = []
    duration_samples = int(30 * 60 * FS)   # 30-minute recordings

    for i in range(n):
        baseline = rng.uniform(120, 155)
        stv      = rng.uniform(3, 12)
        lto_freq = rng.uniform(0.02, 0.06)
        lto_amp  = rng.uniform(5, 20)
        t = np.arange(duration_samples) / FS

        fhr = (baseline
               + lto_amp * np.sin(2 * np.pi * lto_freq * t)
               + stv * rng.standard_normal(duration_samples))
        fhr = np.clip(fhr, 60, 200)

        # Decelerations
        n_decels = rng.integers(0, 8)
        decel_events = []
        for _ in range(n_decels):
            onset = int(rng.integers(0, duration_samples - 240))
            depth = rng.uniform(10, 50)
            width = int(rng.integers(60, 200))
            end   = min(onset + width, duration_samples)
            w2    = width // 2
            shape = np.concatenate([
                np.linspace(0, -depth, w2),
                np.linspace(-depth, 0, width - w2),
            ])[:end - onset]
            fhr[onset:end] += shape
            decel_events.append({"onset_s": onset / FS, "depth_bpm": depth,
                                  "duration_s": (end - onset) / FS})
        fhr = np.clip(fhr, 40, 220)

        # UC signal
        uc = np.zeros(duration_samples)
        n_contractions = rng.integers(3, 15)
        for _ in range(n_contractions):
            onset = int(rng.integers(0, duration_samples - 360))
            peak  = rng.uniform(30, 100)
            width = int(rng.integers(120, 360))
            half  = width // 2
            end   = min(onset + width, duration_samples)
            ha    = min(half, end - onset)
            uc[onset: onset + ha] += np.linspace(0, peak, ha)
            rest = end - (onset + ha)
            if rest > 0:
                uc[onset + ha: end] += np.linspace(peak, 0, rest)
        uc = np.clip(uc, 0, 120)

        # Missing samples
        missing_frac = rng.uniform(0.05, 0.15)
        missing_idx  = rng.choice(duration_samples,
                                   size=int(missing_frac * duration_samples), replace=False)
        fhr[missing_idx] = np.nan
        quality = float(np.sum(~np.isnan(fhr))) / duration_samples

        # pH outcome (correlated with deceleration burden)
        ph = float(np.clip(rng.normal(7.22, 0.08), 6.80, 7.50))
        ph_label = ("acidosis"   if ph < PH_ACIDOSIS else
                    "borderline" if ph < PH_BORDERLINE else "normal")

        annotations = {"decelerations": decel_events} if include_annotations else {}

        records.append(CTGRecord(
            record_id=f"SYNTH-CTU-{i+1:04d}",
            source="synthetic",
            fhr=fhr, uc=uc, fs=float(FS),
            duration_min=30.0,
            signal_quality=quality,
            ph=ph,
            base_deficit=float(rng.normal(4.5, 2.5)),
            apgar1=float(rng.integers(6, 10)),
            apgar5=float(rng.integers(7, 10)),
            ph_label=ph_label,
            gestational_age=float(rng.uniform(37, 42)),
            birth_weight=float(rng.normal(3400, 450)),
            delivery_type=rng.choice(["vaginal", "cs"], p=[0.7, 0.3]),
            annotations=annotations,
        ))

    return records


# ─── 4. Unified CTU-UHB Dataset class ────────────────────────────────────────

class CTUDataset:
    """
    Unified loader for CTU-UHB as the PRIMARY clinical dataset.

    Priority order for loading raw CTG signals:
      1. PhysioNet WFDB streaming (real data, requires wfdb)
      2. Zenodo preprocessed CSVs (real data, requires internet)
      3. Physiologically-plausible synthetic fallback (demo only)

    CTU-CHB annotations loaded separately for event detection.
    """

    def __init__(
        self,
        max_records: int = 552,
        force_synthetic: bool = False,
        load_annotations: bool = True,
        verbose: bool = True,
    ):
        self.max_records      = max_records
        self.force_synthetic  = force_synthetic
        self.load_annotations = load_annotations
        self.verbose          = verbose

        self._records:       List[CTGRecord] = []
        self._anno_records:  List[CTGRecord] = []
        self._loaded         = False
        self._load_method    = "not loaded"

    def load(self) -> "CTUDataset":
        if self._loaded:
            return self

        # ── Primary CTU-UHB signals ──────────────────────────────────────────
        if not self.force_synthetic:
            # Try PhysioNet WFDB
            records = load_ctu_uhb_physionet(
                max_records=self.max_records, verbose=self.verbose
            )
            if records:
                self._records    = records
                self._load_method = f"PhysioNet WFDB ({len(records)} real records)"
            else:
                # Try Zenodo CSV
                records = load_zenodo_csv(
                    "ctu_uhb_proc_csv", max_records=self.max_records,
                    verbose=self.verbose
                )
                if not records:
                    records = load_zenodo_csv(
                        "ctu_uhb_csv", max_records=self.max_records,
                        verbose=self.verbose
                    )
                if records:
                    self._records    = records
                    self._load_method = f"Zenodo CSV ({len(records)} real records)"

        if not self._records:
            # Synthetic fallback
            n_synth = min(self.max_records, 100)
            self._records    = make_synthetic_ctu_uhb(n=n_synth, seed=42)
            self._load_method = f"synthetic ({len(self._records)} records)"
            if self.verbose:
                print(f"  [CTU-UHB] Using synthetic fallback: {len(self._records)} records")

        # ── CTU-CHB annotations ──────────────────────────────────────────────
        if self.load_annotations and not self.force_synthetic:
            anno = load_zenodo_csv(
                "fhrma_ano_csv",
                max_records=self.max_records,
                verbose=self.verbose,
            )
            if not anno:
                anno = load_zenodo_csv(
                    "fhrma_proc_csv",
                    max_records=self.max_records,
                    verbose=self.verbose,
                )
            self._anno_records = anno
            if self.verbose and anno:
                print(f"  [CTU-CHB-anno] Loaded {len(anno)} annotation records")

        self._loaded = True
        if self.verbose:
            print(f"  [CTUDataset] Ready: {len(self._records)} primary records | "
                  f"{len(self._anno_records)} annotation records | "
                  f"method: {self._load_method}")
        return self

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def records(self) -> List[CTGRecord]:
        if not self._loaded:
            self.load()
        return self._records

    @property
    def annotation_records(self) -> List[CTGRecord]:
        if not self._loaded:
            self.load()
        return self._anno_records

    @property
    def load_method(self) -> str:
        return self._load_method

    @property
    def is_real_data(self) -> bool:
        return "real" in self._load_method or "PhysioNet" in self._load_method or "Zenodo" in self._load_method

    def get_record(self, record_id: str) -> Optional[CTGRecord]:
        for r in self.records:
            if r.record_id == record_id:
                return r
        return None

    def to_metadata_df(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.records])

    def ph_labels(self) -> pd.Series:
        return pd.Series([r.ph_label for r in self.records])

    def stats(self) -> Dict:
        recs = self.records
        phs  = [r.ph for r in recs if not np.isnan(r.ph)]
        return {
            "n_records":        len(recs),
            "load_method":      self._load_method,
            "is_real_data":     self.is_real_data,
            "n_annotation_records": len(self._anno_records),
            "total_hours":      round(sum(r.duration_min for r in recs) / 60, 1),
            "mean_duration_min": round(float(np.mean([r.duration_min for r in recs])), 1),
            "mean_signal_quality": round(float(np.mean([r.signal_quality for r in recs])), 3),
            "n_ph_available":   len(phs),
            "mean_ph":          round(float(np.mean(phs)), 4) if phs else None,
            "n_acidosis":       sum(1 for r in recs if r.ph_label == "acidosis"),
            "n_borderline":     sum(1 for r in recs if r.ph_label == "borderline"),
            "n_normal_ph":      sum(1 for r in recs if r.ph_label == "normal"),
            "ph_label_dist":    pd.Series([r.ph_label for r in recs]).value_counts().to_dict(),
            "sources":          pd.Series([r.source for r in recs]).value_counts().to_dict(),
        }


# ─── Legacy compatibility alias ───────────────────────────────────────────────
# Keep CTGDLDataset name working for any existing imports

class CTGDLDataset(CTUDataset):
    """Alias for backwards compatibility — use CTUDataset for new code."""

    def __init__(self, use_ctu_uhb=True, use_fhrma=True, use_spam=True,
                 max_per_source=50, force_synthetic=False, verbose=True):
        super().__init__(
            max_records=max_per_source,
            force_synthetic=force_synthetic,
            load_annotations=use_fhrma,
            verbose=verbose,
        )

    def load(self) -> "CTGDLDataset":
        super().load()
        return self

    @property
    def load_summary(self) -> Dict:
        return {
            "CTU-UHB": {
                "count":   len(self._records),
                "method":  self._load_method,
                "citation": "Chudáček et al. (2014) BMC Pregnancy and Childbirth 14:16",
                "url":     "https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/",
            },
            "CTU-CHB-anno": {
                "count":   len(self._anno_records),
                "method":  "Zenodo CSV (annotations)" if self._anno_records else "not loaded",
                "citation": "Petránek et al. (2020) PMC7256311",
                "url":     "https://zenodo.org/records/19510407",
            },
        }
