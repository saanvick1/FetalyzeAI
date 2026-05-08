"""
ctu_loader.py
=============
Strict loader for the real CTU-CHB / CTU-UHB intrapartum CTG dataset.

This is the ONLY loader allowed in the active training pipeline.

Hard rules:
  - Reads only from the uploaded CTU ZIP under attached_assets/
  - Never falls back to fetal_health.csv, UCI, Kaggle, or synthetic data
  - Crashes loudly with the exact error messages required by the spec
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

ROOT          = Path(__file__).parent
ATTACHED_DIR  = ROOT / "attached_assets"
EXTRACT_PARENT = ROOT / "data"
EXTRACT_DIR    = EXTRACT_PARENT / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"

ZIP_NAME_GLOB = "ctu-chb-intrapartum-cardiotocography-database-1.0.0*.zip"

FS = 4  # CTU-CHB sampling rate (Hz) for both FHR and UC channels

NOT_FOUND_MSG = (
    "Real CTU-CHB/CTU-UHB files were not found. "
    "Synthetic or fetal_health.csv fallback data is not allowed."
)
EXTRACT_FAIL_MSG = (
    "CTU-CHB/CTU-UHB extraction failed. No real .hea/.dat files were found."
)
NO_RECORDS_MSG = "No real CTU records were loaded. Training stopped."


# ────────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CTURecord:
    record_id:        str
    fhr:              np.ndarray
    uc:               np.ndarray
    fs:               int                  = FS
    duration_min:     float                = 0.0
    signal_quality:   float                = 0.0
    missingness_pct:  float                = 0.0
    ph:               float                = float("nan")
    base_deficit:     float                = float("nan")
    apgar1:           float                = float("nan")
    apgar5:           float                = float("nan")
    gestational_age:  float                = float("nan")
    birth_weight:     float                = float("nan")
    delivery_type:    str                  = "unknown"
    # Rich clinical context metadata from CTU-CHB header
    maternal_age:     float                = float("nan")
    parity:           float                = float("nan")
    gravidity:        float                = float("nan")
    diabetes:         float                = float("nan")
    hypertension:     float                = float("nan")
    preeclampsia:     float                = float("nan")
    liq_praecox:      float                = float("nan")
    pyrexia:          float                = float("nan")
    meconium:         float                = float("nan")
    presentation:     float                = float("nan")
    induced:          float                = float("nan")
    i_stage_min:      float                = float("nan")
    ii_stage_min:     float                = float("nan")
    no_progress:      float                = float("nan")
    ck_kp:            float                = float("nan")
    sig2birth_s:      float                = float("nan")
    sex:              float                = float("nan")
    rec_type:         float                = float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["fhr"] = self.fhr
        d["uc"]  = self.uc
        return d


# ────────────────────────────────────────────────────────────────────────────
# Extraction
# ────────────────────────────────────────────────────────────────────────────

def _find_zip() -> Optional[Path]:
    candidates = sorted(ATTACHED_DIR.glob(ZIP_NAME_GLOB))
    return candidates[0] if candidates else None


def _ensure_extracted(verbose: bool = True) -> Path:
    """Extract the CTU ZIP if not already extracted. Crashes loudly if missing."""
    hea_already = list(EXTRACT_DIR.glob("*.hea")) if EXTRACT_DIR.exists() else []
    if len(hea_already) > 100:
        if verbose:
            print(f"[ctu_loader] already extracted → {EXTRACT_DIR}  ({len(hea_already)} .hea)")
        return EXTRACT_DIR

    zip_path = _find_zip()
    if zip_path is None:
        raise RuntimeError(NOT_FOUND_MSG)

    if verbose:
        print(f"[ctu_loader] extracting {zip_path.name} ...")
    EXTRACT_PARENT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(EXTRACT_PARENT)

    hea = list(EXTRACT_DIR.glob("*.hea"))
    dat = list(EXTRACT_DIR.glob("*.dat"))
    if not hea or not dat:
        raise RuntimeError(EXTRACT_FAIL_MSG)
    if verbose:
        print(f"[ctu_loader] extracted {len(hea)} .hea / {len(dat)} .dat")
    return EXTRACT_DIR


# ────────────────────────────────────────────────────────────────────────────
# Header parsing
# ────────────────────────────────────────────────────────────────────────────

# Maps the leading prefix of each comment line in CTU .hea files to our field name.
_HEADER_KEYS = [
    ("pH",            "ph"),
    ("BDecf",         "base_deficit"),
    ("BE",            "_be_alt"),
    ("Apgar1",        "apgar1"),
    ("Apgar5",        "apgar5"),
    ("Gest. weeks",   "gestational_age"),
    ("Weight(g)",     "birth_weight"),
    ("Deliv. type",   "delivery_type"),
    # Rich clinical context fields from CTU-CHB .hea comments
    ("Age",           "maternal_age"),
    ("Parity",        "parity"),
    ("Gravidity",     "gravidity"),
    ("Diabetes",      "diabetes"),
    ("Hypertension",  "hypertension"),
    ("Preeclampsia",  "preeclampsia"),
    ("Liq. praecox",  "liq_praecox"),
    ("Pyrexia",       "pyrexia"),
    ("Meconium",      "meconium"),
    ("Presentation",  "presentation"),
    ("Induced",       "induced"),
    ("II.stage",      "ii_stage_min"),
    ("I.stage",       "i_stage_min"),
    ("NoProgress",    "no_progress"),
    ("CK/KP",         "ck_kp"),
    ("Sig2Birth",     "sig2birth_s"),
    ("Sex",           "sex"),
    ("Rec. type",     "rec_type"),
]


def _parse_comments(comments: list[str]) -> dict:
    out: dict = {}
    for raw in comments or []:
        line = raw.strip().lstrip("#").strip()
        if not line:
            continue
        for prefix, target in _HEADER_KEYS:
            if line.startswith(prefix):
                rest = line[len(prefix):].strip()
                tok  = rest.split()[0] if rest else ""
                if not tok:
                    continue
                if target == "delivery_type":
                    out["delivery_type"] = tok
                else:
                    try:
                        out[target] = float(tok)
                    except ValueError:
                        pass
                break
    if "ph" not in out and "_be_alt" in out:
        # BE is a base-excess proxy; keep it as base_deficit only if BDecf was missing.
        out.setdefault("base_deficit", -out["_be_alt"])
    out.pop("_be_alt", None)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def load_ctu_records(max_records: Optional[int] = None,
                     verbose: bool = True) -> list[CTURecord]:
    """Load real CTU-CHB records. Raises if data is unavailable. Never returns synthetic."""
    import wfdb

    data_dir = _ensure_extracted(verbose=verbose)
    hea_files = sorted(data_dir.glob("*.hea"))
    dat_files = sorted(data_dir.glob("*.dat"))

    fhr_channel_seen = False
    uc_channel_seen  = False
    total_signal_seconds = 0.0

    records: list[CTURecord] = []
    skipped = 0

    iter_files = hea_files if max_records is None else hea_files[:max_records]
    for hf in iter_files:
        rid = hf.stem
        try:
            rec = wfdb.rdrecord(str(data_dir / rid))
            snames = [s.upper() for s in (rec.sig_name or [])]
            fi = next((i for i, s in enumerate(snames) if "FHR" in s), None)
            ui = next((i for i, s in enumerate(snames)
                       if any(t in s for t in ("UC", "TOCO", "TOC"))), None)
            if fi is None:
                skipped += 1
                continue
            fhr_channel_seen = True
            if ui is not None:
                uc_channel_seen = True

            fhr = rec.p_signal[:, fi].astype(float)
            uc  = (rec.p_signal[:, ui].astype(float)
                   if ui is not None else np.full_like(fhr, np.nan))

            # Mark physiologically impossible values as missing — do not invent data.
            fhr_missing_raw = (fhr <= 0) | (fhr > 300)
            fhr[fhr_missing_raw] = np.nan
            uc[uc < 0] = np.nan

            meta = _parse_comments(rec.comments or [])
            duration_min   = len(fhr) / (FS * 60)
            missing_pct    = float(np.mean(np.isnan(fhr)) * 100)
            signal_quality = float(1.0 - missing_pct / 100)
            total_signal_seconds += len(fhr) / FS

            def _mf(key):
                v = meta.get(key, float("nan"))
                try:   return float(v)
                except Exception: return float("nan")

            records.append(CTURecord(
                record_id        = rid,
                fhr              = fhr,
                uc               = uc,
                fs               = FS,
                duration_min     = duration_min,
                signal_quality   = signal_quality,
                missingness_pct  = missing_pct,
                ph               = _mf("ph"),
                base_deficit     = _mf("base_deficit"),
                apgar1           = _mf("apgar1"),
                apgar5           = _mf("apgar5"),
                gestational_age  = _mf("gestational_age"),
                birth_weight     = _mf("birth_weight"),
                delivery_type    = str(meta.get("delivery_type", "unknown")),
                maternal_age     = _mf("maternal_age"),
                parity           = _mf("parity"),
                gravidity        = _mf("gravidity"),
                diabetes         = _mf("diabetes"),
                hypertension     = _mf("hypertension"),
                preeclampsia     = _mf("preeclampsia"),
                liq_praecox      = _mf("liq_praecox"),
                pyrexia          = _mf("pyrexia"),
                meconium         = _mf("meconium"),
                presentation     = _mf("presentation"),
                induced          = _mf("induced"),
                i_stage_min      = _mf("i_stage_min"),
                ii_stage_min     = _mf("ii_stage_min"),
                no_progress      = _mf("no_progress"),
                ck_kp            = _mf("ck_kp"),
                sig2birth_s      = _mf("sig2birth_s"),
                sex              = _mf("sex"),
                rec_type         = _mf("rec_type"),
            ))
        except Exception as e:
            skipped += 1
            if verbose and skipped < 5:
                print(f"[ctu_loader] skipped {rid}: {e}")

    # ── Audit ─────────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "─" * 60)
        print(" CTU-CHB / CTU-UHB Dataset Audit")
        print("─" * 60)
        print(f" Dataset source: CTU-CHB/CTU-UHB local ZIP")
        print(f" ZIP found: yes")
        print(f" Records loaded: {len(records)}")
        print(f" HEA files found: {len(hea_files)}")
        print(f" DAT files found: {len(dat_files)}")
        print(f" FHR channel found: {'yes' if fhr_channel_seen else 'no'}")
        print(f" UC channel found:  {'yes' if uc_channel_seen else 'no'}")
        print(f" Total signal hours: {total_signal_seconds / 3600:.2f}")
        print(f" Synthetic fallback used: NO")
        print(f" fetal_health.csv used: NO")
        print("─" * 60 + "\n")

    if len(records) == 0:
        raise RuntimeError(NO_RECORDS_MSG)

    return records


if __name__ == "__main__":
    recs = load_ctu_records(max_records=5)
    for r in recs:
        print(r.record_id, "pH=", r.ph, "Apgar5=", r.apgar5,
              "duration=", round(r.duration_min, 1), "min")
