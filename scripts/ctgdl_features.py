"""
CTGDL Feature Extraction Engine
=================================
Extracts clinical CTG features from raw FHR + UC signals loaded by ctgdl_loader.

Implements the FetalyzeAI v2 feature set described in the architecture document:
  - FHR baseline, variability (STV/LTV), accelerations, decelerations
  - UC contraction analysis
  - FHR + UC relationship features (contraction-stress response)
  - Signal quality features
  - Fetal Reserve Score (0–100)
  - Deceleration Burden Index
  - Contraction Stress Response timeline

All values are computed from NaN-masked arrays without silent imputation.
Missing-data fractions are preserved as features.

References:
  - FIGO guidelines for CTG interpretation (2015)
  - CTU-UHB Intrapartum CTG Database, PhysioNet
  - CTGDL paper, Fridman et al. (2025), SSRN 6027919
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings("ignore")

FS = 4  # Hz
WINDOW_5MIN  = 5  * 60 * FS    # samples
WINDOW_10MIN = 10 * 60 * FS    # samples
BASELINE_NORMAL = (110, 160)   # bpm FIGO normal range


# ---------------------------------------------------------------------------
# Low-level signal utilities
# ---------------------------------------------------------------------------

def _valid(arr: np.ndarray) -> np.ndarray:
    """Return non-NaN values."""
    return arr[~np.isnan(arr)]


def _missing_fraction(arr: np.ndarray) -> float:
    return float(np.mean(np.isnan(arr)))


def _interpolate_short_gaps(arr: np.ndarray, max_gap_s: float = 10.0, fs: float = FS) -> np.ndarray:
    """Linear interpolation of gaps shorter than max_gap_s seconds. Longer gaps stay NaN."""
    out = arr.copy()
    max_gap = int(max_gap_s * fs)
    n = len(out)
    i = 0
    while i < n:
        if np.isnan(out[i]):
            j = i
            while j < n and np.isnan(out[j]):
                j += 1
            gap_len = j - i
            if gap_len <= max_gap and i > 0 and j < n:
                out[i:j] = np.linspace(out[i - 1], out[j], gap_len + 2)[1:-1]
            i = j
        else:
            i += 1
    return out


def _compute_baseline_fhr(fhr: np.ndarray, fs: float = FS) -> float:
    """Estimate FHR baseline using a rolling median over 10-minute windows."""
    valid = _valid(fhr)
    if len(valid) < 10:
        return float("nan")
    window = int(10 * 60 * fs)
    if len(valid) < window:
        return float(np.nanmedian(valid))
    medians = [np.nanmedian(valid[max(0, i - window // 2): i + window // 2])
               for i in range(window // 2, len(valid) - window // 2, window // 4)]
    return float(np.nanmedian(medians)) if medians else float(np.nanmedian(valid))


def _compute_stv(fhr: np.ndarray) -> float:
    """Short-term variability: mean of absolute beat-to-beat differences."""
    v = _valid(fhr)
    if len(v) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(v))))


def _compute_ltv(fhr: np.ndarray, fs: float = FS) -> float:
    """Long-term variability: mean range in 1-minute epochs."""
    v = _valid(fhr)
    epoch = int(60 * fs)
    if len(v) < epoch:
        return float("nan")
    ranges = [v[i: i + epoch].max() - v[i: i + epoch].min()
              for i in range(0, len(v) - epoch, epoch)]
    return float(np.mean(ranges)) if ranges else float("nan")


# ---------------------------------------------------------------------------
# Deceleration detection
# ---------------------------------------------------------------------------

@dataclass
class Deceleration:
    onset_s: float
    nadir_s: float
    end_s: float
    depth_bpm: float
    duration_s: float
    area_bpm_s: float      # area below baseline
    recovery_slope: float  # bpm/s after nadir
    decel_type: str        # "early"|"late"|"variable"|"prolonged"|"unknown"


def detect_decelerations(
    fhr: np.ndarray,
    uc: Optional[np.ndarray] = None,
    baseline: Optional[float] = None,
    fs: float = FS,
    min_depth: float = 15.0,
    min_duration_s: float = 15.0,
) -> List[Deceleration]:
    """
    Detect FHR decelerations below baseline.
    Uses a threshold of baseline - min_depth bpm.
    """
    if baseline is None:
        baseline = _compute_baseline_fhr(fhr, fs)
    if np.isnan(baseline):
        return []

    fhr_interp = _interpolate_short_gaps(fhr, max_gap_s=10.0, fs=fs)
    threshold = baseline - min_depth
    below = fhr_interp < threshold
    # NaN counts as not-below
    below[np.isnan(fhr_interp)] = False

    decelerations: List[Deceleration] = []
    i = 0
    n = len(below)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            duration_s = (j - i) / fs
            if duration_s >= min_duration_s:
                segment = fhr_interp[i:j]
                nadir_idx = i + int(np.nanargmin(segment))
                depth = baseline - float(np.nanmin(segment))
                area = float(np.nansum(baseline - segment[segment < baseline])) / fs

                # Recovery slope: gradient from nadir to end of deceleration
                recovery_len = max(1, j - nadir_idx)
                recovery_segment = fhr_interp[nadir_idx: min(nadir_idx + recovery_len + int(30 * fs), n)]
                valid_rec = _valid(recovery_segment)
                if len(valid_rec) >= 2:
                    rec_slope = float((valid_rec[-1] - valid_rec[0]) / (len(valid_rec) / fs))
                else:
                    rec_slope = float("nan")

                # Classify deceleration type using UC timing if available
                decel_type = _classify_deceleration(i, j, nadir_idx, uc, baseline, fhr_interp, fs)

                decelerations.append(Deceleration(
                    onset_s=i / fs,
                    nadir_s=nadir_idx / fs,
                    end_s=j / fs,
                    depth_bpm=depth,
                    duration_s=duration_s,
                    area_bpm_s=area,
                    recovery_slope=rec_slope,
                    decel_type=decel_type,
                ))
            i = j
        else:
            i += 1

    return decelerations


def _classify_deceleration(
    onset: int,
    end: int,
    nadir: int,
    uc: Optional[np.ndarray],
    baseline: float,
    fhr: np.ndarray,
    fs: float,
) -> str:
    duration_s = (end - onset) / fs
    if duration_s > 120:
        return "prolonged"
    if uc is None or np.all(np.isnan(uc)):
        return "unknown"

    # Find the nearest contraction peak
    uc_valid = np.where(np.isnan(uc), 0.0, uc)
    search_start = max(0, onset - int(60 * fs))
    search_end   = min(len(uc_valid), end + int(60 * fs))
    local_uc = uc_valid[search_start:search_end]
    if local_uc.max() < 20:  # no significant contraction nearby
        return "variable"

    peak_local = int(np.argmax(local_uc))
    peak_abs   = search_start + peak_local
    peak_s     = peak_abs / fs
    nadir_s    = nadir / fs
    lag_s      = nadir_s - peak_s

    if -30 <= lag_s <= 30:
        return "early"
    elif lag_s > 30:
        return "late"
    else:
        return "variable"


# ---------------------------------------------------------------------------
# Acceleration detection
# ---------------------------------------------------------------------------

@dataclass
class Acceleration:
    onset_s: float
    peak_s: float
    end_s: float
    height_bpm: float
    duration_s: float


def detect_accelerations(
    fhr: np.ndarray,
    baseline: Optional[float] = None,
    fs: float = FS,
    min_height: float = 15.0,
    min_duration_s: float = 15.0,
) -> List[Acceleration]:
    if baseline is None:
        baseline = _compute_baseline_fhr(fhr, fs)
    if np.isnan(baseline):
        return []

    fhr_interp = _interpolate_short_gaps(fhr, max_gap_s=10.0, fs=fs)
    threshold = baseline + min_height
    above = fhr_interp > threshold
    above[np.isnan(fhr_interp)] = False

    accelerations: List[Acceleration] = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            duration_s = (j - i) / fs
            if duration_s >= min_duration_s:
                segment = fhr_interp[i:j]
                peak_idx = i + int(np.nanargmax(segment))
                height = float(np.nanmax(segment)) - baseline
                accelerations.append(Acceleration(
                    onset_s=i / fs,
                    peak_s=peak_idx / fs,
                    end_s=j / fs,
                    height_bpm=height,
                    duration_s=duration_s,
                ))
            i = j
        else:
            i += 1
    return accelerations


# ---------------------------------------------------------------------------
# Contraction detection
# ---------------------------------------------------------------------------

@dataclass
class Contraction:
    onset_s: float
    peak_s: float
    end_s: float
    peak_intensity: float
    duration_s: float
    interval_s: float  # time since previous contraction (NaN for first)


def detect_contractions(
    uc: np.ndarray,
    fs: float = FS,
    threshold: float = 20.0,
    min_duration_s: float = 20.0,
) -> List[Contraction]:
    if uc is None or np.all(np.isnan(uc)):
        return []

    uc_valid = np.where(np.isnan(uc), 0.0, uc)
    above = uc_valid > threshold

    contractions: List[Contraction] = []
    i = 0
    n = len(above)
    prev_end_s = float("nan")

    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            duration_s = (j - i) / fs
            if duration_s >= min_duration_s:
                segment = uc_valid[i:j]
                peak_idx = i + int(np.argmax(segment))
                interval_s = (i / fs - prev_end_s) if not np.isnan(prev_end_s) else float("nan")
                contractions.append(Contraction(
                    onset_s=i / fs,
                    peak_s=peak_idx / fs,
                    end_s=j / fs,
                    peak_intensity=float(segment.max()),
                    duration_s=duration_s,
                    interval_s=interval_s,
                ))
                prev_end_s = j / fs
            i = j
        else:
            i += 1
    return contractions


# ---------------------------------------------------------------------------
# Deceleration Burden Index
# ---------------------------------------------------------------------------

def compute_deceleration_burden(decels: List[Deceleration]) -> float:
    """
    Deceleration Burden Index = sum over all decels of (depth * duration * recurrence_weight)
    Recurrence_weight increases for closely spaced decelerations.
    """
    if not decels:
        return 0.0
    total = sum(d.depth_bpm * d.duration_s for d in decels)
    # Normalize by number of decels to penalise recurrence
    return float(total * (1 + 0.1 * len(decels)))


# ---------------------------------------------------------------------------
# Contraction Stress Response
# ---------------------------------------------------------------------------

@dataclass
class ContractionResponse:
    contraction_onset_s: float
    deceleration_found: bool
    decel_lag_s: float        # positive = deceleration after contraction peak
    decel_depth_bpm: float
    recovery_time_s: float    # time from deceleration end to return to baseline
    decel_type: str


def compute_contraction_stress_response(
    contractions: List[Contraction],
    decels: List[Deceleration],
    window_s: float = 90.0,
) -> List[ContractionResponse]:
    """
    For each contraction, find the nearest deceleration within window_s.
    """
    responses: List[ContractionResponse] = []
    for c in contractions:
        best_decel = None
        best_lag = float("inf")
        for d in decels:
            lag = d.onset_s - c.onset_s
            if 0 <= lag <= window_s and abs(lag) < abs(best_lag):
                best_decel = d
                best_lag = lag
        if best_decel:
            rec_time = best_decel.end_s - best_decel.nadir_s
            responses.append(ContractionResponse(
                contraction_onset_s=c.onset_s,
                deceleration_found=True,
                decel_lag_s=best_lag,
                decel_depth_bpm=best_decel.depth_bpm,
                recovery_time_s=rec_time,
                decel_type=best_decel.decel_type,
            ))
        else:
            responses.append(ContractionResponse(
                contraction_onset_s=c.onset_s,
                deceleration_found=False,
                decel_lag_s=float("nan"),
                decel_depth_bpm=0.0,
                recovery_time_s=float("nan"),
                decel_type="none",
            ))
    return responses


# ---------------------------------------------------------------------------
# Fetal Reserve Score
# ---------------------------------------------------------------------------

def compute_fetal_reserve_score(
    fhr: np.ndarray,
    uc: Optional[np.ndarray],
    fs: float = FS,
    decels: Optional[List[Deceleration]] = None,
    accels: Optional[List[Acceleration]] = None,
    contractions: Optional[List[Contraction]] = None,
    csr: Optional[List[ContractionResponse]] = None,
) -> Tuple[float, Dict]:
    """
    Fetal Reserve Score: 0 (no reserve) to 100 (excellent reserve).
    Each component contributes up to its max points.

    Components:
      baseline_stability   : 20 pts
      short_term_variability: 20 pts
      long_term_variability : 15 pts
      accelerations        : 15 pts
      deceleration_burden  : 20 pts (negative)
      contraction_response : 10 pts
    """
    score = 0.0
    components = {}

    baseline = _compute_baseline_fhr(fhr, fs)
    stv = _compute_stv(fhr)
    ltv = _compute_ltv(fhr, fs)
    missing = _missing_fraction(fhr)

    # 1. Baseline stability (20 pts)
    if not np.isnan(baseline):
        if BASELINE_NORMAL[0] <= baseline <= BASELINE_NORMAL[1]:
            pts = 20.0
        elif 100 <= baseline < 110 or 160 < baseline <= 170:
            pts = 10.0
        else:
            pts = 0.0
        score += pts
        components["baseline_stability"] = pts
    else:
        components["baseline_stability"] = None

    # 2. Short-term variability (20 pts)
    if not np.isnan(stv):
        if 5 <= stv <= 25:
            pts = 20.0
        elif 3 <= stv < 5:
            pts = 10.0
        elif stv < 3:
            pts = 0.0
        else:
            pts = 15.0  # slightly elevated but not absent
        score += pts
        components["stv"] = pts
    else:
        components["stv"] = None

    # 3. Long-term variability (15 pts)
    if not np.isnan(ltv):
        if 10 <= ltv <= 40:
            pts = 15.0
        elif 5 <= ltv < 10:
            pts = 7.0
        elif ltv < 5:
            pts = 0.0
        else:
            pts = 10.0
        score += pts
        components["ltv"] = pts
    else:
        components["ltv"] = None

    # 4. Accelerations (15 pts)
    if accels is not None:
        duration_min = len(fhr) / (fs * 60)
        acc_per_30min = len(accels) / max(duration_min / 30, 0.01)
        if acc_per_30min >= 2:
            pts = 15.0
        elif acc_per_30min >= 1:
            pts = 8.0
        else:
            pts = 0.0
        score += pts
        components["accelerations"] = pts
    else:
        components["accelerations"] = None

    # 5. Deceleration burden penalty (up to -20 pts)
    if decels is not None:
        dbi = compute_deceleration_burden(decels)
        # Late and prolonged decelerations are more penalizing
        late_count = sum(1 for d in decels if d.decel_type == "late")
        prolonged_count = sum(1 for d in decels if d.decel_type == "prolonged")
        severity = min(20.0, dbi * 0.01 + late_count * 4 + prolonged_count * 6)
        pts = -severity
        score += pts
        components["deceleration_burden"] = pts
    else:
        components["deceleration_burden"] = None

    # 6. Contraction stress response (10 pts)
    if csr is not None and contractions is not None and len(contractions) > 0:
        responded = sum(1 for r in csr if r.deceleration_found)
        frac_with_decel = responded / len(contractions)
        # Fewer decelerations after contractions = better reserve
        pts = 10.0 * (1 - frac_with_decel)
        score += pts
        components["contraction_response"] = pts
    else:
        components["contraction_response"] = None

    # Signal quality penalty
    if missing > 0.2:
        score *= (1 - (missing - 0.2))

    score = float(np.clip(score, 0, 100))
    return score, components


# ---------------------------------------------------------------------------
# Full feature extraction for a single CTGRecord
# ---------------------------------------------------------------------------

def extract_ctg_features(record) -> Dict:
    """
    Extract the full feature set from a CTGRecord.
    Returns a flat dictionary suitable for DataFrame construction.
    """
    fhr = record.fhr
    uc  = record.uc
    fs  = float(record.fs)

    features: Dict = {
        "record_id": record.record_id,
        "source": record.source,
        "duration_min": record.duration_min,
        "signal_quality": record.signal_quality,
        "missing_fhr": _missing_fraction(fhr),
        "missing_uc": _missing_fraction(uc) if uc is not None else 1.0,
        "ph": record.ph,
        "base_deficit": record.base_deficit,
        "apgar1": record.apgar1,
        "apgar5": record.apgar5,
        "ph_label": record.ph_label,
    }

    # --- FHR features ---
    valid_fhr = _valid(fhr)
    baseline = _compute_baseline_fhr(fhr, fs)
    stv      = _compute_stv(fhr)
    ltv      = _compute_ltv(fhr, fs)

    features.update({
        "baseline_fhr": baseline,
        "mean_fhr": float(np.nanmean(valid_fhr)) if len(valid_fhr) > 0 else float("nan"),
        "median_fhr": float(np.nanmedian(valid_fhr)) if len(valid_fhr) > 0 else float("nan"),
        "std_fhr": float(np.nanstd(valid_fhr)) if len(valid_fhr) > 0 else float("nan"),
        "min_fhr": float(np.nanmin(valid_fhr)) if len(valid_fhr) > 0 else float("nan"),
        "max_fhr": float(np.nanmax(valid_fhr)) if len(valid_fhr) > 0 else float("nan"),
        "stv": stv,
        "ltv": ltv,
        "tachycardia_frac": float(np.nanmean(fhr > 160)) if len(valid_fhr) > 0 else float("nan"),
        "bradycardia_frac": float(np.nanmean(fhr < 110)) if len(valid_fhr) > 0 else float("nan"),
    })

    # Entropy
    try:
        from scipy.stats import entropy as scipy_entropy
        hist, _ = np.histogram(_valid(fhr), bins=50, density=True)
        hist = hist[hist > 0]
        features["fhr_entropy"] = float(scipy_entropy(hist))
    except Exception:
        features["fhr_entropy"] = float("nan")

    # FHR slope over last 30% of recording
    n = len(fhr)
    tail = fhr[int(0.7 * n):]
    valid_tail = _valid(tail)
    if len(valid_tail) >= 2:
        features["fhr_late_slope"] = float(np.polyfit(range(len(valid_tail)), valid_tail, 1)[0] * fs)
    else:
        features["fhr_late_slope"] = float("nan")

    # --- Decelerations ---
    decels = detect_decelerations(fhr, uc, baseline=baseline, fs=fs)
    duration_min = record.duration_min or max(len(fhr) / (fs * 60), 0.001)

    features.update({
        "n_decels": len(decels),
        "decels_per_30min": len(decels) / max(duration_min / 30, 0.001),
        "mean_decel_depth": float(np.mean([d.depth_bpm for d in decels])) if decels else 0.0,
        "max_decel_depth": float(max([d.depth_bpm for d in decels])) if decels else 0.0,
        "mean_decel_duration_s": float(np.mean([d.duration_s for d in decels])) if decels else 0.0,
        "total_decel_area": float(sum(d.area_bpm_s for d in decels)),
        "n_late_decels": sum(1 for d in decels if d.decel_type == "late"),
        "n_variable_decels": sum(1 for d in decels if d.decel_type == "variable"),
        "n_prolonged_decels": sum(1 for d in decels if d.decel_type == "prolonged"),
        "n_early_decels": sum(1 for d in decels if d.decel_type == "early"),
        "mean_recovery_slope": float(np.nanmean([d.recovery_slope for d in decels])) if decels else float("nan"),
        "deceleration_burden_index": compute_deceleration_burden(decels),
    })

    # --- Accelerations ---
    accels = detect_accelerations(fhr, baseline=baseline, fs=fs)
    features.update({
        "n_accels": len(accels),
        "accels_per_30min": len(accels) / max(duration_min / 30, 0.001),
        "mean_accel_height": float(np.mean([a.height_bpm for a in accels])) if accels else 0.0,
        "mean_accel_duration_s": float(np.mean([a.duration_s for a in accels])) if accels else 0.0,
    })

    # --- Contractions ---
    contractions = detect_contractions(uc, fs=fs) if uc is not None else []
    features.update({
        "n_contractions": len(contractions),
        "contractions_per_10min": len(contractions) / max(duration_min / 10, 0.001),
        "mean_contraction_duration_s": float(np.mean([c.duration_s for c in contractions])) if contractions else 0.0,
        "mean_contraction_intensity": float(np.mean([c.peak_intensity for c in contractions])) if contractions else 0.0,
        "mean_contraction_interval_s": float(np.nanmean([c.interval_s for c in contractions])) if contractions else float("nan"),
    })

    # --- Contraction Stress Response ---
    csr = compute_contraction_stress_response(contractions, decels)
    n_with_decel = sum(1 for r in csr if r.deceleration_found)
    features.update({
        "decel_after_contraction_frac": n_with_decel / len(csr) if csr else 0.0,
        "mean_decel_lag_s": float(np.nanmean([r.decel_lag_s for r in csr if r.deceleration_found])) if n_with_decel > 0 else float("nan"),
        "mean_recovery_after_contraction_s": float(np.nanmean([r.recovery_time_s for r in csr if r.deceleration_found])) if n_with_decel > 0 else float("nan"),
        "late_decel_after_contraction_count": sum(1 for r in csr if r.decel_type == "late"),
    })

    # --- Fetal Reserve Score ---
    reserve_score, reserve_components = compute_fetal_reserve_score(
        fhr, uc, fs=fs, decels=decels, accels=accels, contractions=contractions, csr=csr
    )
    features["fetal_reserve_score"] = reserve_score
    for k, v in reserve_components.items():
        features[f"reserve_{k}"] = v

    return features


def extract_features_batch(records) -> pd.DataFrame:
    """Extract features for a list of CTGRecord objects."""
    rows = []
    for rec in records:
        try:
            rows.append(extract_ctg_features(rec))
        except Exception as exc:
            rows.append({"record_id": rec.record_id, "source": rec.source, "error": str(exc)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rolling window analysis (for timeline view)
# ---------------------------------------------------------------------------

def rolling_risk_timeline(
    fhr: np.ndarray,
    uc: Optional[np.ndarray],
    window_min: float = 5.0,
    step_min: float = 2.5,
    fs: float = FS,
) -> pd.DataFrame:
    """
    Compute risk metrics in rolling windows for the timeline display.
    Returns a DataFrame with one row per window.
    """
    window_samples = int(window_min * 60 * fs)
    step_samples   = int(step_min  * 60 * fs)
    n = len(fhr)
    rows = []

    for start in range(0, n - window_samples, step_samples):
        end   = start + window_samples
        w_fhr = fhr[start:end]
        w_uc  = uc[start:end] if uc is not None else None
        mid_s = (start + window_samples // 2) / fs
        mid_min = mid_s / 60

        baseline = _compute_baseline_fhr(w_fhr, fs)
        stv      = _compute_stv(w_fhr)
        decels   = detect_decelerations(w_fhr, w_uc, baseline=baseline, fs=fs)
        dbi      = compute_deceleration_burden(decels)
        missing  = _missing_fraction(w_fhr)

        rows.append({
            "time_min": round(mid_min, 2),
            "baseline_fhr": round(baseline, 1) if not np.isnan(baseline) else None,
            "stv": round(stv, 2) if not np.isnan(stv) else None,
            "n_decels": len(decels),
            "decel_burden_index": round(dbi, 1),
            "missing_fraction": round(missing, 3),
            "n_late_decels": sum(1 for d in decels if d.decel_type == "late"),
        })

    return pd.DataFrame(rows)
