"""
ctg_feature_engine.py
=====================
Extracts clinical CTG features from real FHR + UC waveforms.

Produces predictor features ONLY — never assigns labels.
Adds the unique FetalyzeAI metrics:
  - Fetal Reserve Score
  - Deceleration Burden Index
  - Contraction Stress Response
  - Signal Quality Score
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FS = 4  # Hz (CTU-CHB)


# ────────────────────────────────────────────────────────────────────────────
# 1. clean_signal
# ────────────────────────────────────────────────────────────────────────────

def clean_signal(fhr: np.ndarray, uc: np.ndarray):
    fhr = np.asarray(fhr, dtype=float).copy()
    uc  = np.asarray(uc,  dtype=float).copy()
    invalid_fhr = (fhr < 50) | (fhr > 220) | np.isnan(fhr)
    fhr[invalid_fhr] = np.nan
    uc[(uc < 0) | np.isnan(uc)] = np.nan
    missing_mask = np.isnan(fhr)
    return fhr, uc, missing_mask


# ────────────────────────────────────────────────────────────────────────────
# 2. interpolate_short_gaps
# ────────────────────────────────────────────────────────────────────────────

def interpolate_short_gaps(signal: np.ndarray,
                           max_gap_seconds: float = 5.0,
                           fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    sig = np.asarray(signal, dtype=float).copy()
    missing = np.isnan(sig)
    if not missing.any():
        return sig, missing

    max_gap = int(max_gap_seconds * fs)
    n = len(sig)
    i = 0
    while i < n:
        if missing[i]:
            j = i
            while j < n and missing[j]:
                j += 1
            gap_len = j - i
            if gap_len <= max_gap and i > 0 and j < n:
                left, right = sig[i - 1], sig[j]
                if not (np.isnan(left) or np.isnan(right)):
                    sig[i:j] = np.linspace(left, right, gap_len + 2)[1:-1]
            i = j
        else:
            i += 1
    return sig, missing


# ────────────────────────────────────────────────────────────────────────────
# 3. estimate_baseline_fhr
# ────────────────────────────────────────────────────────────────────────────

def estimate_baseline_fhr(fhr: np.ndarray, fs: int = FS) -> float:
    valid = fhr[~np.isnan(fhr)]
    if len(valid) < 10:
        return float("nan")
    # Robust 10-min rolling-median, then median of those baselines
    win = int(10 * 60 * fs)
    if len(valid) < win:
        return float(np.nanmedian(valid))
    rolled = pd.Series(valid).rolling(win, min_periods=win // 4, center=True).median()
    return float(np.nanmedian(rolled.dropna()))


# ────────────────────────────────────────────────────────────────────────────
# 4. variability features
# ────────────────────────────────────────────────────────────────────────────

def compute_variability_features(fhr: np.ndarray, fs: int = FS) -> dict:
    valid = fhr[~np.isnan(fhr)]
    if len(valid) < 4:
        return {"stv": np.nan, "ltv": np.nan, "std_fhr": np.nan, "roughness": np.nan}
    stv = float(np.mean(np.abs(np.diff(valid))))
    std_fhr = float(np.std(valid))
    epoch = int(60 * fs)
    if len(valid) >= epoch:
        ranges = [valid[i:i+epoch].max() - valid[i:i+epoch].min()
                  for i in range(0, len(valid) - epoch, epoch)]
        ltv = float(np.mean(ranges)) if ranges else np.nan
    else:
        ltv = np.nan
    diffs = np.diff(valid)
    rough = float(np.std(diffs) / (np.std(valid) + 1e-6))
    return {"stv": stv, "ltv": ltv, "std_fhr": std_fhr, "roughness": rough}


# ────────────────────────────────────────────────────────────────────────────
# 5/6. accelerations & decelerations
# ────────────────────────────────────────────────────────────────────────────

def _detect_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    diffs = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    return list(zip(starts, ends))


def detect_accelerations(fhr: np.ndarray, baseline: float, fs: int = FS,
                         delta: float = 15.0, min_dur_s: float = 15.0) -> dict:
    if np.isnan(baseline):
        return {"n_accels": 0, "mean_accel_height": 0.0, "mean_accel_dur_s": 0.0}
    fhr_i = np.where(np.isnan(fhr), baseline, fhr)
    above = fhr_i > (baseline + delta)
    runs  = _detect_runs(above)
    heights, durs = [], []
    for s, e in runs:
        d = (e - s) / fs
        if d >= min_dur_s:
            heights.append(fhr_i[s:e].max() - baseline)
            durs.append(d)
    return {
        "n_accels":          len(heights),
        "mean_accel_height": float(np.mean(heights)) if heights else 0.0,
        "mean_accel_dur_s":  float(np.mean(durs))    if durs    else 0.0,
    }


def detect_decelerations(fhr: np.ndarray, baseline: float, fs: int = FS,
                         delta: float = 15.0, min_dur_s: float = 15.0) -> dict:
    if np.isnan(baseline):
        return dict(n_decels=0, mean_decel_depth=0.0, mean_decel_dur_s=0.0,
                    total_decel_dur_s=0.0, decel_area=0.0, prolonged_flag=0,
                    max_decel_depth=0.0, decel_runs=[], decel_recoveries=[])
    fhr_i = np.where(np.isnan(fhr), baseline, fhr)
    below = fhr_i < (baseline - delta)
    runs  = _detect_runs(below)
    depths, durs, areas, recoveries = [], [], [], []
    prolonged  = 0
    decel_runs = []
    for s, e in runs:
        d = (e - s) / fs
        if d < min_dur_s:
            continue
        seg   = fhr_i[s:e]
        depth = float(baseline - seg.min())
        area  = float(np.sum(np.maximum(baseline - seg, 0)) / fs)
        depths.append(depth)
        durs.append(d)
        areas.append(area)
        if d >= 120:
            prolonged = 1
        # Per-decel recovery time: seconds until FHR returns within 5 bpm of baseline
        post_end = min(e + int(120 * fs), len(fhr_i))
        post     = fhr_i[e:post_end]
        rec_idx  = np.where(post >= (baseline - 5))[0]
        rec_s    = float(rec_idx[0] / fs) if len(rec_idx) else float(len(post) / fs)
        recoveries.append(rec_s)
        decel_runs.append((s, e, depth, d, rec_s))
    return dict(
        n_decels          = len(depths),
        mean_decel_depth  = float(np.mean(depths))      if depths     else 0.0,
        max_decel_depth   = float(np.max(depths))        if depths     else 0.0,
        mean_decel_dur_s  = float(np.mean(durs))         if durs       else 0.0,
        total_decel_dur_s = float(np.sum(durs))          if durs       else 0.0,
        decel_area        = float(np.sum(areas))         if areas      else 0.0,
        mean_decel_recovery_s = float(np.mean(recoveries)) if recoveries else 0.0,
        prolonged_flag    = int(prolonged),
        decel_runs        = decel_runs,
        decel_recoveries  = recoveries,
    )


# ────────────────────────────────────────────────────────────────────────────
# 7. contractions
# ────────────────────────────────────────────────────────────────────────────

def detect_contractions(uc: np.ndarray, fs: int = FS,
                        thresh: float = 30.0, min_dur_s: float = 30.0) -> dict:
    if uc is None or np.all(np.isnan(uc)):
        return {"n_contractions": 0, "mean_contraction_dur_s": 0.0,
                "mean_contraction_intensity": 0.0,
                "contractions_per_10min": 0.0,
                "uc_runs": []}
    uc_i = np.where(np.isnan(uc), 0.0, uc)
    runs = _detect_runs(uc_i > thresh)
    durs, peaks, kept = [], [], []
    for s, e in runs:
        d = (e - s) / fs
        if d >= min_dur_s:
            durs.append(d)
            peaks.append(float(uc_i[s:e].max()))
            kept.append((s, e))
    total_min = len(uc) / (fs * 60)
    return dict(
        n_contractions             = len(kept),
        mean_contraction_dur_s     = float(np.mean(durs))  if durs  else 0.0,
        mean_contraction_intensity = float(np.mean(peaks)) if peaks else 0.0,
        contractions_per_10min     = (len(kept) / total_min * 10) if total_min > 0 else 0.0,
        uc_runs                    = kept,
    )


# ────────────────────────────────────────────────────────────────────────────
# 8. contraction stress response (unique)
# ────────────────────────────────────────────────────────────────────────────

def compute_contraction_stress_response(fhr: np.ndarray, uc: np.ndarray,
                                        baseline: float, fs: int = FS,
                                        lam: float = 0.50,
                                        gamma: float = 0.30) -> dict:
    """
    Contraction Stress Response (spec §10):
        drop_c  = baseline - min(FHR[t_peak : t_peak + Δ])
        rec_c   = time for FHR to return near baseline after nadir
        CSR     = mean(drop_c) + λ × mean(rec_c) + γ × trend(rec_c)

    λ = 0.50 penalises slow recovery; γ = 0.30 penalises worsening over time.
    trend(rec_c) > 0 means recovery is getting slower → higher risk.
    """
    _empty = dict(mean_fhr_drop=0.0, mean_recovery_time_s=0.0,
                  delayed_recovery_score=0.0, late_decel_likelihood=0.0,
                  worsening_recovery_trend=0.0, csr_score=0.0)
    if np.isnan(baseline):
        return _empty
    contractions = detect_contractions(uc, fs)
    runs = contractions["uc_runs"]
    if not runs:
        return _empty

    fhr_i = np.where(np.isnan(fhr), baseline, fhr)
    drops, recoveries, late_decel = [], [], 0
    delta = int(90 * fs)   # 90-second post-contraction window (spec §10)
    for s, e in runs:
        post_end = min(e + delta, len(fhr_i))
        post = fhr_i[e:post_end]
        if len(post) < fs * 5:
            continue
        drop = float(baseline - post.min())
        drops.append(max(drop, 0.0))
        # Recovery: first time FHR returns within 5 bpm of baseline
        rec_idx = np.where(post >= (baseline - 5))[0]
        recoveries.append(float(rec_idx[0] / fs) if len(rec_idx) else float(len(post) / fs))
        # Late deceleration flag: nadir ≥ 30 s after contraction end
        if drop > 15:
            nadir_t = int(np.argmin(post))
            if nadir_t / fs >= 30:
                late_decel += 1

    if not drops:
        return _empty

    mean_drop = float(np.mean(drops))
    mean_rec  = float(np.mean(recoveries))

    # trend(rec_c) = slope of recovery times over the recording
    if len(recoveries) >= 3:
        t_idx  = np.arange(len(recoveries), dtype=float)
        try:
            trend_rec = float(np.polyfit(t_idx, recoveries, 1)[0])
        except Exception:
            trend_rec = 0.0
    else:
        half = len(recoveries) // 2 or 1
        trend_rec = float(np.mean(recoveries[half:])) - float(np.mean(recoveries[:half]))

    # CSR formula (spec §10)
    csr_score = mean_drop + lam * mean_rec + gamma * max(trend_rec, 0.0)

    return dict(
        mean_fhr_drop            = mean_drop,
        mean_recovery_time_s     = mean_rec,
        delayed_recovery_score   = float(np.mean([r > 30 for r in recoveries])),
        late_decel_likelihood    = float(late_decel / len(runs)),
        worsening_recovery_trend = trend_rec,
        csr_score                = float(csr_score),
    )


# ────────────────────────────────────────────────────────────────────────────
# 9. deceleration burden
# ────────────────────────────────────────────────────────────────────────────

def compute_deceleration_burden_index(decel_features: dict,
                                      delayed_recovery_score: float = 0.0) -> float:
    """
    Deceleration Burden Index (spec §9):
        DBI = Σₖ depthₖ × durationₖ × (1 + recovery_timeₖ / 60)

    Uses per-deceleration recovery times when available (decel_runs contains
    5-tuples: start, end, depth, duration_s, recovery_s).
    Falls back to mean-based approximation for backward compatibility.
    """
    runs = decel_features.get("decel_runs", [])
    dbi  = 0.0
    if runs:
        for entry in runs:
            if len(entry) >= 5:
                _s, _e, depth_k, dur_k, rec_k = entry[:5]
            else:
                _s, _e, depth_k, dur_k = entry[:4]
                rec_k = decel_features.get("mean_decel_recovery_s", 0.0)
            dbi += float(depth_k) * float(dur_k) * (1.0 + float(rec_k) / 60.0)
    else:
        # Fallback: approximate from aggregated stats
        depth   = decel_features.get("mean_decel_depth", 0.0)
        dur     = decel_features.get("mean_decel_dur_s", 0.0)
        n       = decel_features.get("n_decels", 0)
        rec_s   = decel_features.get("mean_decel_recovery_s",
                  max(delayed_recovery_score * 60.0, 0.0))
        dbi = float(depth) * float(dur) * (1.0 + rec_s / 60.0) * float(n)
    return float(np.log1p(dbi))


# ────────────────────────────────────────────────────────────────────────────
# 10. fetal reserve score (0–100)
# ────────────────────────────────────────────────────────────────────────────

def compute_fetal_reserve_score(features: dict) -> float:
    """
    Fetal Reserve Score (spec §11):
        s = β₀
            + β₁ × variability
            + β₂ × accelerations
            - β₃ × DBI
            - β₄ × CSR
            - β₅ × signal_loss
            - β₆ × worsening_trend
        FRS = 100 / (1 + e^(-s))

    Interpretation:
        FRS > 70  — strong reserve
        40–70     — borderline
        FRS < 40  — low reserve
    """
    stv     = features.get("stv",                    np.nan)
    ltv     = features.get("ltv",                    np.nan)
    n_acc   = features.get("n_accels",               0.0)
    burden  = features.get("decel_burden_idx",       0.0)
    csr     = features.get("csr_score",              0.0)
    sq      = features.get("signal_quality",         1.0)
    trend   = features.get("worsening_recovery_trend", 0.0)
    dur_min = max(features.get("duration_min",       1.0), 1.0)

    # β₀ — neutral baseline: maps to FRS ≈ 50 with no features
    s = 0.0

    # β₁ — variability (normal STV 5–25 bpm, normal LTV 10–40 bpm)
    if not np.isnan(stv):
        if   5.0 <= stv <= 25.0: s += 0.8
        elif 3.0 <= stv <  5.0:  s += 0.1
        else:                     s -= 0.6
    if not np.isnan(ltv):
        if   10.0 <= ltv <= 40.0: s += 0.4
        elif  5.0 <= ltv < 10.0:  s += 0.1
        else:                      s -= 0.3

    # β₂ — accelerations (≥2 per 30 min is reassuring)
    acc_per30 = float(n_acc) / (dur_min / 30.0)
    s += 0.6 if acc_per30 >= 2 else (0.2 if acc_per30 >= 1 else -0.3)

    # β₃ — deceleration burden (log-scaled, higher = more risk)
    s -= 0.50 * float(burden)

    # β₄ — contraction stress response (higher = more risk)
    s -= 0.03 * float(max(csr, 0.0))

    # β₅ — signal loss penalty
    s -= 0.8 * (1.0 - float(max(sq, 0.0)))

    # β₆ — worsening recovery trend (positive trend = getting worse)
    s -= 0.02 * float(max(trend, 0.0))

    # FRS = 100 × sigmoid(s)
    frs = 100.0 / (1.0 + np.exp(-s))
    return float(np.clip(frs, 0.0, 100.0))


# ────────────────────────────────────────────────────────────────────────────
# 11b. FIGO 2015 guideline binary flags (medical domain knowledge integration)
# Reference: Ayres-de-Campos et al. (2015) Int J Gynecol Obstet 131(1):13-24
# ────────────────────────────────────────────────────────────────────────────

def compute_figo_flags(features: dict) -> dict:
    """
    Encode FIGO 2015 intrapartum CTG classification criteria as binary flags.

    Normal:       Baseline 110-160, variability 5-25, no late/prolonged decels,
                  accelerations present.
    Suspicious:   Baseline 100-109 or 161-180, variability <5 for <90 min,
                  no accelerations ≥40 min, early/variable decels.
    Pathological: Baseline <100 or >180, variability <5 for ≥90 min, sinusoidal,
                  repetitive late decels, prolonged decel.

    These rules encode decades of expert obstetric knowledge directly into the
    feature space — addressing the 'lack of medical knowledge integration' gap
    in pure data-driven models.
    """
    baseline = features.get("baseline_fhr", float("nan"))
    stv      = features.get("stv",          float("nan"))
    flat_pct = features.get("flatline_pct", 0.0)
    n_acc    = features.get("n_accels",     0.0)
    dur_min  = max(features.get("duration_min", 1.0), 1.0)
    late_dl  = features.get("late_decel_likelihood", 0.0)
    prol_f   = features.get("prolonged_decel_flag",  0.0)

    figo_tachycardia        = int(not np.isnan(baseline) and baseline > 160)
    figo_bradycardia        = int(not np.isnan(baseline) and baseline < 110)
    figo_abnormal_baseline  = int(figo_tachycardia or figo_bradycardia)
    figo_absent_variability = int(not np.isnan(stv) and stv < 5.0)
    figo_reduced_variability= int(not np.isnan(stv) and 5.0 <= stv < 10.0)
    # Sinusoidal: persistent flat oscillation with very low variability
    figo_sinusoidal         = int(flat_pct > 80.0 and not np.isnan(stv) and stv < 3.0)
    # Absent accels: concerning only if recording is long enough (≥40 min)
    figo_absent_accels      = int(dur_min >= 40.0 and float(n_acc) == 0.0)
    figo_late_decels        = int(late_dl > 0.30)
    figo_prolonged_decel    = int(float(prol_f) > 0)

    # Pathological feature count
    path_count = (
        figo_abnormal_baseline      +
        figo_sinusoidal             +
        figo_absent_variability     +
        int(late_dl > 0.50)         +   # repetitive late decels
        figo_prolonged_decel
    )
    # Suspicious feature count
    susp_count = (
        figo_reduced_variability    +
        figo_absent_accels          +
        figo_late_decels
    )

    # Simplified 3-level FIGO category
    if path_count >= 1:
        figo_category = 2   # pathological
    elif susp_count >= 1 or figo_tachycardia or figo_bradycardia:
        figo_category = 1   # suspicious
    else:
        figo_category = 0   # normal

    return {
        "figo_abnormal_baseline":   float(figo_abnormal_baseline),
        "figo_tachycardia":         float(figo_tachycardia),
        "figo_bradycardia":         float(figo_bradycardia),
        "figo_absent_variability":  float(figo_absent_variability),
        "figo_reduced_variability": float(figo_reduced_variability),
        "figo_absent_accels":       float(figo_absent_accels),
        "figo_late_decels":         float(figo_late_decels),
        "figo_prolonged_decel":     float(figo_prolonged_decel),
        "figo_sinusoidal":          float(figo_sinusoidal),
        "figo_composite_score":     float(path_count + susp_count),
        "figo_category":            float(figo_category),
    }


# ────────────────────────────────────────────────────────────────────────────
# 11. signal quality
# ────────────────────────────────────────────────────────────────────────────

def compute_signal_quality(fhr: np.ndarray, uc: np.ndarray, fs: int = FS) -> dict:
    n = max(len(fhr), 1)
    missing_pct = float(np.mean(np.isnan(fhr)) * 100)

    valid = fhr[~np.isnan(fhr)]
    if len(valid) >= 10:
        flat_pct = float(np.mean(np.abs(np.diff(valid)) < 0.5) * 100)
        jumps    = int(np.sum(np.abs(np.diff(valid)) > 25))
    else:
        flat_pct = 100.0
        jumps    = 0

    quality = max(0.0, 1 - missing_pct / 100) * max(0.0, 1 - flat_pct / 100) * \
              max(0.3, 1 - jumps / max(n, 1) * 50)
    return dict(
        missing_pct       = missing_pct,
        flatline_pct      = flat_pct,
        abrupt_jump_count = jumps,
        signal_quality    = float(np.clip(quality, 0, 1)),
    )


# ────────────────────────────────────────────────────────────────────────────
# 12. record-level features
# ────────────────────────────────────────────────────────────────────────────

def extract_record_features(record, light: bool = False) -> dict:
    rd = record.as_dict() if hasattr(record, "as_dict") else dict(record)
    fs  = int(rd.get("fs", FS))
    fhr = np.asarray(rd["fhr"], dtype=float)
    uc  = np.asarray(rd["uc"],  dtype=float)

    fhr_c, uc_c, _missing = clean_signal(fhr, uc)
    fhr_i, _ = interpolate_short_gaps(fhr_c, max_gap_seconds=5.0, fs=fs)

    baseline = estimate_baseline_fhr(fhr_i, fs)
    var_f    = compute_variability_features(fhr_i, fs)
    acc_f    = detect_accelerations(fhr_i, baseline, fs)
    dec_f    = detect_decelerations(fhr_i, baseline, fs)
    con_f    = detect_contractions(uc_c, fs)
    csr_f    = compute_contraction_stress_response(fhr_i, uc_c, baseline, fs)
    sq_f     = compute_signal_quality(fhr_c, uc_c, fs)

    valid = fhr_i[~np.isnan(fhr_i)]
    mean_fhr = float(np.mean(valid)) if len(valid) else np.nan
    tach     = float(np.mean(valid > 160)) if len(valid) else np.nan
    brad     = float(np.mean(valid < 110)) if len(valid) else np.nan
    dur_min  = float(rd.get("duration_min", len(fhr) / (fs * 60)))
    dur_30   = max(dur_min / 30, 0.001)

    burden_idx = compute_deceleration_burden_index(dec_f, csr_f["delayed_recovery_score"])

    pre = dict(
        baseline_fhr               = baseline,
        mean_fhr                   = mean_fhr,
        std_fhr                    = var_f["std_fhr"],
        stv                        = var_f["stv"],
        ltv                        = var_f["ltv"],
        stv_norm                   = var_f["stv"] / 10.0 if var_f["stv"] is not None and not np.isnan(var_f["stv"]) else np.nan,
        ltv_norm                   = var_f["ltv"] / 25.0 if var_f["ltv"] is not None and not np.isnan(var_f["ltv"]) else np.nan,
        roughness                  = var_f["roughness"],
        tachycardia_frac           = tach,
        bradycardia_frac           = brad,
        n_accels                   = float(acc_f["n_accels"]),
        accels_per_30min           = acc_f["n_accels"] / dur_30,
        mean_accel_height          = acc_f["mean_accel_height"],
        n_decels                   = float(dec_f["n_decels"]),
        decels_per_30min           = dec_f["n_decels"] / dur_30,
        mean_decel_depth           = dec_f["mean_decel_depth"],
        max_decel_depth            = dec_f["max_decel_depth"],
        mean_decel_dur_s           = dec_f["mean_decel_dur_s"],
        total_decel_dur_s          = dec_f["total_decel_dur_s"],
        decel_area                 = dec_f["decel_area"],
        mean_decel_recovery_s      = dec_f.get("mean_decel_recovery_s", 0.0),
        prolonged_decel_flag       = float(dec_f["prolonged_flag"]),
        n_contractions             = float(con_f["n_contractions"]),
        mean_contraction_dur_s     = con_f["mean_contraction_dur_s"],
        mean_contraction_intensity = con_f["mean_contraction_intensity"],
        contractions_per_10min     = con_f["contractions_per_10min"],
        mean_fhr_drop_post_uc      = csr_f["mean_fhr_drop"],
        mean_recovery_time_s       = csr_f["mean_recovery_time_s"],
        delayed_recovery_score     = csr_f["delayed_recovery_score"],
        late_decel_likelihood      = csr_f["late_decel_likelihood"],
        worsening_recovery_trend   = csr_f["worsening_recovery_trend"],
        csr_score                  = csr_f.get("csr_score", 0.0),
        decel_burden_idx           = burden_idx,
        missing_fhr_pct            = sq_f["missing_pct"],
        flatline_pct               = sq_f["flatline_pct"],
        abrupt_jump_count          = float(sq_f["abrupt_jump_count"]),
        signal_quality             = sq_f["signal_quality"],
        duration_min               = dur_min,
    )
    pre["fetal_reserve_score"] = compute_fetal_reserve_score(pre)
    pre.update(compute_figo_flags(pre))

    if light:
        pre["record_id"]     = rd["record_id"]
        pre["ph"]            = rd.get("ph", np.nan)
        pre["base_deficit"]  = rd.get("base_deficit", np.nan)
        pre["apgar1"]        = rd.get("apgar1", np.nan)
        pre["apgar5"]        = rd.get("apgar5", np.nan)
        return pre

    # Spectral feature (cheap log-band powers via FFT)
    try:
        pre.update(compute_spectral_features(fhr_i, fs))
    except Exception:
        pre.update({"lf_power": np.nan, "mf_power": np.nan, "hf_power": np.nan,
                    "lf_hf_ratio": np.nan, "spectral_entropy": np.nan})

    # Cheap last-30-min stats (no spectral, no entropy)
    last_n = int(30 * 60 * fs)
    if len(fhr_i) > last_n + 60 * fs:
        fhr_l = fhr_i[-last_n:]
        uc_l  = uc_c[-last_n:] if len(uc_c) >= last_n else uc_c
        try:
            bl_l  = estimate_baseline_fhr(fhr_l, fs)
            var_l = compute_variability_features(fhr_l, fs)
            dec_l = detect_decelerations(fhr_l, bl_l, fs)
            valid_l = fhr_l[~np.isnan(fhr_l)]
            pre.update({
                "baseline_fhr_last30":         bl_l,
                "stv_last30":                  var_l["stv"],
                "ltv_last30":                  var_l["ltv"],
                "std_fhr_last30":              var_l["std_fhr"],
                "n_decels_last30":             float(dec_l["n_decels"]),
                "max_decel_depth_last30":      dec_l["max_decel_depth"],
            })
            pre["stv_trend_late_vs_full"] = (var_l["stv"] - pre["stv"]) if not np.isnan(var_l["stv"]) and not np.isnan(pre["stv"]) else 0.0
            pre["baseline_trend_late_vs_full"] = (bl_l - baseline) if not np.isnan(bl_l) and not np.isnan(baseline) else 0.0
        except Exception:
            for k in ("baseline_fhr_last30", "stv_last30", "ltv_last30",
                      "std_fhr_last30", "n_decels_last30", "max_decel_depth_last30"):
                pre[k] = pre.get(k.replace("_last30", ""), np.nan)
            pre["stv_trend_late_vs_full"] = 0.0
            pre["baseline_trend_late_vs_full"] = 0.0
    else:
        for k in ("baseline_fhr_last30", "stv_last30", "ltv_last30",
                  "std_fhr_last30", "n_decels_last30", "max_decel_depth_last30"):
            pre[k] = pre.get(k.replace("_last30", ""), np.nan)
        pre["stv_trend_late_vs_full"] = 0.0
        pre["baseline_trend_late_vs_full"] = 0.0

    pre["record_id"]     = rd["record_id"]
    pre["ph"]            = rd.get("ph", np.nan)
    pre["base_deficit"]  = rd.get("base_deficit", np.nan)
    pre["apgar1"]        = rd.get("apgar1", np.nan)
    pre["apgar5"]        = rd.get("apgar5", np.nan)
    return pre


# ────────────────────────────────────────────────────────────────────────────
# spectral & complexity features (added without breaking existing API)
# ────────────────────────────────────────────────────────────────────────────

def compute_spectral_features(fhr: np.ndarray, fs: int = FS) -> dict:
    """Power-spectral density features in physiologically meaningful FHR bands."""
    valid = fhr[~np.isnan(fhr)]
    if len(valid) < fs * 60:
        return {"lf_power": np.nan, "mf_power": np.nan, "hf_power": np.nan,
                "lf_hf_ratio": np.nan, "spectral_entropy": np.nan}
    sig = valid - np.mean(valid)
    n = len(sig)
    fft = np.fft.rfft(sig * np.hanning(n))
    psd = (np.abs(fft) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    lf = float(np.sum(psd[(freqs > 0.03) & (freqs <= 0.15)]))
    mf = float(np.sum(psd[(freqs > 0.15) & (freqs <= 0.50)]))
    hf = float(np.sum(psd[(freqs > 0.50) & (freqs <= 1.00)]))
    total = lf + mf + hf + 1e-12
    p = psd / (psd.sum() + 1e-12)
    p = p[p > 0]
    spectral_entropy = float(-np.sum(p * np.log(p)) / np.log(len(p))) if len(p) else np.nan
    return {
        "lf_power": float(np.log1p(lf)),
        "mf_power": float(np.log1p(mf)),
        "hf_power": float(np.log1p(hf)),
        "lf_hf_ratio": float(lf / (hf + 1e-6)),
        "spectral_entropy": spectral_entropy,
    }


def compute_complexity_features(fhr: np.ndarray) -> dict:
    """Approximate entropy proxy + linear trend slope."""
    valid = fhr[~np.isnan(fhr)]
    if len(valid) < 50:
        return {"sample_entropy": np.nan, "trend_slope": np.nan,
                "perm_entropy": np.nan}

    # Subsample aggressively for speed (ApEn is O(n²))
    if len(valid) > 600:
        idx = np.linspace(0, len(valid) - 1, 600).astype(int)
        s = valid[idx]
    else:
        s = valid

    def _approx_entropy(x, m=2, r_factor=0.2):
        x = np.asarray(x, dtype=float)
        N = len(x)
        if N < m + 2:
            return np.nan
        r = r_factor * np.std(x)
        if r <= 0:
            return np.nan
        def _phi(m):
            X = np.array([x[i:i+m] for i in range(N - m + 1)])
            C = np.array([
                np.sum(np.max(np.abs(X - X[i]), axis=1) <= r) / (N - m + 1.0)
                for i in range(N - m + 1)
            ])
            C = C[C > 0]
            return float(np.sum(np.log(C)) / (N - m + 1.0))
        try:
            return abs(_phi(m) - _phi(m + 1))
        except Exception:
            return np.nan

    apen = _approx_entropy(s)

    # Linear trend slope across full record
    t = np.arange(len(valid))
    try:
        slope = float(np.polyfit(t, valid, 1)[0])
    except Exception:
        slope = np.nan

    # Permutation entropy (simple order-3)
    try:
        from itertools import permutations
        m = 3
        perms = list(permutations(range(m)))
        counts = {p: 0 for p in perms}
        for i in range(len(s) - m + 1):
            order = tuple(np.argsort(s[i:i+m]))
            counts[order] = counts.get(order, 0) + 1
        total = sum(counts.values()) or 1
        ps = np.array([c / total for c in counts.values() if c > 0])
        pe = float(-np.sum(ps * np.log(ps)) / np.log(len(perms)))
    except Exception:
        pe = np.nan

    return {"sample_entropy": apen, "trend_slope": slope, "perm_entropy": pe}


# ────────────────────────────────────────────────────────────────────────────
# 13. window-level features
# ────────────────────────────────────────────────────────────────────────────

def extract_window_features(record, window_minutes: float = 5.0,
                            step_minutes: float = 2.5) -> pd.DataFrame:
    rd  = record.as_dict() if hasattr(record, "as_dict") else dict(record)
    fs  = int(rd.get("fs", FS))
    fhr = np.asarray(rd["fhr"], dtype=float)
    uc  = np.asarray(rd["uc"],  dtype=float)
    win  = int(window_minutes * 60 * fs)
    step = int(step_minutes   * 60 * fs)
    rows = []
    if win <= 0 or len(fhr) < win:
        return pd.DataFrame()
    for start in range(0, len(fhr) - win + 1, step):
        end = start + win
        sub = {
            "record_id":   rd["record_id"],
            "fhr":         fhr[start:end],
            "uc":          uc[start:end],
            "fs":          fs,
            "ph":          rd.get("ph", np.nan),
            "base_deficit":rd.get("base_deficit", np.nan),
            "apgar1":      rd.get("apgar1", np.nan),
            "apgar5":      rd.get("apgar5", np.nan),
            "duration_min":(end - start) / (fs * 60),
        }

        class _W:
            def __init__(self, d): self._d = d
            def as_dict(self): return self._d
        feat = extract_record_features(_W(sub), light=True)
        feat["window_start_sec"] = start / fs
        feat["window_end_sec"]   = end   / fs
        rows.append(feat)
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# 14. timeline trends
# ────────────────────────────────────────────────────────────────────────────

def add_timeline_trends(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return window_df
    out = window_df.sort_values(["record_id", "window_start_sec"]).copy()
    grp = out.groupby("record_id")
    for col, new in [
        ("fetal_reserve_score", "frs_delta"),
        ("decel_burden_idx",    "burden_delta"),
        ("stv",                 "stv_delta"),
        ("signal_quality",      "sq_delta"),
    ]:
        if col in out.columns:
            out[new] = grp[col].diff().fillna(0.0)
    out["risk_worsening_trend"] = (
        out.get("burden_delta", 0) - out.get("frs_delta", 0)
    )
    return out
