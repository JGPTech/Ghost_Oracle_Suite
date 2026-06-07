#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
D_M PROBE 05 — DIMENSIONAL INVARIANCE / FORBIDDEN-CORRUPTION CONTROLS
==============================================================================

Drop this file in:

    ghost_oracle/D_M/probes/d_m_probe05_dimensional_invariance_controls.py

Purpose
-------
This probe tests the key D_M claim:

    Allowed channel transformations should preserve D_M.
    Forbidden transformations should collapse D_M.

It is designed as the follow-up to Probe 04W. Probe 04W showed that a
witness-label shuffle may *not* kill the signal. For D_M, that can be a feature,
not a bug: XY / YZ / ZY / YX are channel views of the same dimensional agreement
manifold, not arbitrary labels.

This probe therefore reports two scores:

    1. canonical_yzzy
       Treats YZ as primary and ZY as reciprocal/inverted.

    2. dimensional_invariant
       Searches equivalent reciprocal channel descriptions and keeps the best
       D_M phase/energy agreement. This is the score that should survive allowed
       channel rotations.

Controls
--------
Allowed / should survive:

    observed
        Unmodified record.

    equiv_pair_swap
        Swap the YZ/ZY reciprocal pair with the XY/YX pair.

    equiv_reciprocal_swap
        Swap primary/return inside each reciprocal pair.

    equiv_cyclic_rotation
        Rotate all four channel columns as an allowed channel-basis relabeling.

Forbidden / should collapse or weaken:

    reciprocal_break
        Keep labels and delay order, but break the YZ <-> ZY same-rung inverse
        relationship by permuting reciprocal channels across rungs.

    cross_rung_delay_scramble
        Keep witness values and labels, but scramble delay order x.

    same_label_wrong_delay
        Keep each label's value distribution, but independently assign each
        label to wrong rungs.

    non_equivalence_channel_corruption
        Per-rung random channel order plus random sign flips; violates stable
        reciprocal channel geometry.

    independent_bit_shuffle
        Preserve each tile's q0/q1 marginal bit distributions, but break
        same-shot pairing. Requires raw pair[tile, shot, 2].

Inputs
------
Reads canonical D_M qproj/gproj .npz files with arrays like:

    pair[tile, shot, 2]
    tile_rung_index[tile]
    tile_witness_index[tile]     0=XY, 1=YZ, 2=ZY, 3=YX
    tile_total_delay_dt[tile]

Run
---
From repo root:

    python ghost_oracle/D_M/probes/d_m_probe05_dimensional_invariance_controls.py --auto --window 4096

Or explicitly:

    python ghost_oracle/D_M/probes/d_m_probe05_dimensional_invariance_controls.py ^
      --path base_only=ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_XXXX.npz ^
      --path offset_on=ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_YYYY.npz ^
      --n-perm 1000

Outputs
-------
Creates:

    ghost_oracle/D_M/analysis/dm_probe05_dimensional_invariance_<timestamp>/
        probe05_summary.csv
        probe05_control_distribution.csv
        probe05_rung_values.csv
        probe_config.json

==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# PATHS
# =============================================================================

PROBE_DIR = Path(__file__).resolve().parent
D_M_DIR = PROBE_DIR.parent
DATA_DIR = D_M_DIR / "data"
ANALYSIS_DIR = PROBE_DIR / "analyze"

# Keep these aligned with the current D_M retrieval harness defaults.
DEFAULT_QPROJ_NULL = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fm4ihvjngc73aq3ccg.npz"
DEFAULT_QPROJ_BASE = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8flk2jo3njc73f0g560.npz"
DEFAULT_QPROJ_OFFSET = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fl82bo3njc73f0fgd0.npz"

DEFAULT_GPROJ_NULL = DATA_DIR / "dm_gpu_data_null_4096shots_seed9031229662612491082.npz"
DEFAULT_GPROJ_BASE = DATA_DIR / "dm_gpu_data_base_delay_4096shots_seed2669559634056472362.npz"
DEFAULT_GPROJ_OFFSET = DATA_DIR / "dm_gpu_data_offset_deformed_4096shots_seed6727069190982977623.npz"


# =============================================================================
# CONSTANTS
# =============================================================================

EPS = 1.0e-12
WITNESS_LABELS = ["XY", "YZ", "ZY", "YX"]
WITNESS_TO_INDEX = {name: i for i, name in enumerate(WITNESS_LABELS)}

# Canonical D_M interpretation.
IDX_XY = 0
IDX_YZ = 1
IDX_ZY = 2
IDX_YX = 3

# Equivalent channel descriptions that the dimensional score is allowed to use.
# Format: (primary_index, reciprocal_index, comparison_a, comparison_b, name)
RECIPROCAL_DESCRIPTIONS = [
    (IDX_YZ, IDX_ZY, IDX_XY, IDX_YX, "YZ<-ZY canonical"),
    (IDX_ZY, IDX_YZ, IDX_YX, IDX_XY, "ZY<-YZ reciprocal"),
    (IDX_XY, IDX_YX, IDX_YZ, IDX_ZY, "XY<-YX pair"),
    (IDX_YX, IDX_XY, IDX_ZY, IDX_YZ, "YX<-XY reciprocal"),
]

# Deterministic allowed channel-basis rotations.
ALLOWED_ROTATIONS: Dict[str, Tuple[int, int, int, int]] = {
    "equiv_identity": (0, 1, 2, 3),
    "equiv_pair_swap": (1, 0, 3, 2),          # [XY,YZ,ZY,YX] <- [YZ,XY,YX,ZY]
    "equiv_reciprocal_swap": (3, 2, 1, 0),   # swap within pairs and reverse pair order
    "equiv_cyclic_rotation": (1, 2, 3, 0),   # channel-basis rotation
}

SUMMARY_FIELDS = [
    "record",
    "substrate",
    "condition",
    "transform",
    "family",
    "expectation",
    "windows",
    "points_per_window",
    "canonical_pi_score",
    "canonical_energy_mean",
    "canonical_specificity_mean",
    "canonical_energy_track_r",
    "canonical_phase_vel_r",
    "canonical_phase_span_pi",
    "dim_pi_score",
    "dim_best_description",
    "dim_energy_mean",
    "dim_specificity_mean",
    "dim_energy_track_r",
    "dim_phase_vel_r",
    "dim_phase_span_pi",
    "dim_retention_vs_observed",
    "dim_p_ge_observed",
    "canonical_retention_vs_observed",
    "canonical_p_ge_observed",
    "n_perm",
    "source",
]

DIST_FIELDS = [
    "record",
    "condition",
    "control",
    "perm_index",
    "canonical_pi_score",
    "dim_pi_score",
    "dim_best_description",
    "dim_energy_mean",
    "dim_specificity_mean",
]

RUNG_FIELDS = [
    "record",
    "condition",
    "window",
    "rung",
    "x_total_delay",
    "XY",
    "YZ",
    "ZY",
    "YX",
    "canonical_energy",
    "canonical_specificity",
    "canonical_phase",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DMRecord:
    label: str
    substrate: str
    condition: str
    path: Path
    pair: np.ndarray
    tile_rung: np.ndarray
    tile_witness: np.ndarray
    tile_total_delay: np.ndarray
    meta: Dict[str, Any]


@dataclass
class WindowedWitness:
    W: np.ndarray          # shape: (windows, rungs, 4)
    x: np.ndarray          # shape: (windows, rungs)
    valid: np.ndarray      # shape: (windows, rungs), bool
    windows: int
    rungs: int


@dataclass
class Score:
    pi_score: float
    best_description: str
    energy_mean: float
    specificity_mean: float
    energy_track_r: float
    phase_vel_r: float
    phase_span_pi: float


# =============================================================================
# IO HELPERS
# =============================================================================

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def decode_str_array(arr: Any) -> List[str]:
    a = np.asarray(arr)
    out: List[str] = []
    for x in a.reshape(-1):
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


# =============================================================================
# LOAD RECORDS
# =============================================================================

def infer_condition_from_arrays(path: Path, z: Any, tile_total: np.ndarray, tile_base: Optional[np.ndarray], tile_offset: Optional[np.ndarray]) -> str:
    name = path.name.lower()
    if "null" in name:
        return "null"
    if "offset" in name or "deformed" in name:
        # QPU bell_listener names all contain cavity_offset, so do not rely only on this.
        pass
    if tile_total.size and np.all(tile_total == 0):
        return "null"
    if tile_offset is not None and tile_offset.size and np.all(tile_offset == 0):
        return "base_only"
    if "base_delay" in name or "base_only" in name:
        return "base_only"
    return "offset_on"


def infer_substrate(path: Path, z: Any) -> str:
    if "substrate" in z.files:
        try:
            v = np.asarray(z["substrate"]).item()
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            return str(v)
        except Exception:
            pass
    name = path.name.lower()
    if "gpu" in name or "gproj" in name:
        return "gproj"
    return "qproj"


def optional_int_array(z: Any, name: str, tiles: int, default: int = 0) -> np.ndarray:
    if name in z.files:
        arr = np.asarray(z[name], dtype=np.int32).reshape(-1)
        if arr.shape[0] == tiles:
            return np.ascontiguousarray(arr)
    return np.full((tiles,), int(default), dtype=np.int32)


def load_record(path: Path, label: Optional[str] = None, condition: Optional[str] = None) -> DMRecord:
    if not path.exists():
        raise FileNotFoundError(f"D_M record not found: {path}")

    z = np.load(path, allow_pickle=True)

    if "pair" in z.files:
        pair = np.asarray(z["pair"], dtype=np.uint8)
    else:
        pair_keys = sorted(
            [k for k in z.files if k.startswith("pair_tile")],
            key=lambda k: int(k.replace("pair_tile", "")),
        )
        if not pair_keys:
            raise KeyError(f"{path} has no pair or pair_tile* arrays")
        pair = np.stack([np.asarray(z[k], dtype=np.uint8) for k in pair_keys], axis=0)

    if pair.ndim != 3 or pair.shape[2] != 2:
        raise ValueError(f"{path} pair must have shape (tiles, shots, 2), got {pair.shape}")

    pair = np.ascontiguousarray(pair.astype(np.uint8, copy=False))
    tiles = int(pair.shape[0])

    if "tile_rung_index" in z.files:
        tile_rung = np.asarray(z["tile_rung_index"], dtype=np.int32).reshape(-1)
    else:
        tile_rung = (np.arange(tiles) // 4).astype(np.int32)

    if "tile_witness_index" in z.files:
        tile_witness = np.asarray(z["tile_witness_index"], dtype=np.int32).reshape(-1)
    elif "tile_witness_label" in z.files:
        labels = decode_str_array(z["tile_witness_label"])
        tile_witness = np.asarray([WITNESS_TO_INDEX.get(x, i % 4) for i, x in enumerate(labels[:tiles])], dtype=np.int32)
    else:
        tile_witness = (np.arange(tiles) % 4).astype(np.int32)

    if tile_rung.shape[0] != tiles or tile_witness.shape[0] != tiles:
        raise ValueError(f"metadata length mismatch in {path}: tiles={tiles}")

    tile_base = optional_int_array(z, "tile_base_delay_dt", tiles, 0)
    tile_offset = optional_int_array(z, "tile_offset_dt", tiles, 0)
    tile_total = optional_int_array(z, "tile_total_delay_dt", tiles, 0)

    substrate = infer_substrate(path, z)
    cond = condition or infer_condition_from_arrays(path, z, tile_total, tile_base, tile_offset)
    rec_label = label or f"{substrate}_{cond}"

    meta = {
        "files": list(z.files),
        "tiles": int(pair.shape[0]),
        "shots": int(pair.shape[1]),
        "rungs": int(np.max(tile_rung)) + 1 if tile_rung.size else 0,
        "path": str(path),
    }
    return DMRecord(
        label=rec_label,
        substrate=substrate,
        condition=cond,
        path=path,
        pair=pair,
        tile_rung=np.ascontiguousarray(tile_rung.astype(np.int32, copy=False)),
        tile_witness=np.ascontiguousarray(tile_witness.astype(np.int32, copy=False)),
        tile_total_delay=np.ascontiguousarray(tile_total.astype(np.int32, copy=False)),
        meta=meta,
    )


def parse_path_arg(item: str) -> Tuple[Optional[str], Optional[str], Path]:
    """
    Accepts:
        path/to/file.npz
        label=path/to/file.npz
        condition:label=path/to/file.npz
        condition=path/to/file.npz
    """
    raw = str(item).strip()
    condition = None
    label = None
    path_s = raw

    if "=" in raw:
        lhs, rhs = raw.split("=", 1)
        path_s = rhs.strip()
        lhs = lhs.strip()
        if ":" in lhs:
            condition, label = [x.strip() for x in lhs.split(":", 1)]
        elif lhs in {"null", "base_only", "offset_on"}:
            condition = lhs
            label = lhs
        else:
            label = lhs
    return label, condition, Path(path_s)


def discover_auto_records() -> List[Tuple[str, str, Path]]:
    candidates: List[Tuple[str, str, Path]] = []

    defaults = [
        ("qproj_null", "null", DEFAULT_QPROJ_NULL),
        ("qproj_base_only", "base_only", DEFAULT_QPROJ_BASE),
        ("qproj_offset_on", "offset_on", DEFAULT_QPROJ_OFFSET),
        ("gproj_null", "null", DEFAULT_GPROJ_NULL),
        ("gproj_base_only", "base_only", DEFAULT_GPROJ_BASE),
        ("gproj_offset_on", "offset_on", DEFAULT_GPROJ_OFFSET),
    ]
    for label, condition, path in defaults:
        if path.exists():
            candidates.append((label, condition, path))

    # Fallback: include newest plausible D_M npz files if defaults are absent.
    if not candidates and DATA_DIR.exists():
        globs = list(DATA_DIR.glob("dm_data_bell_listener*.npz")) + list(DATA_DIR.glob("dm_gpu_data*.npz"))
        globs = sorted(globs, key=lambda p: p.stat().st_mtime, reverse=True)
        for path in globs[:12]:
            label = path.stem
            candidates.append((label, "", path))

    return candidates


# =============================================================================
# NUMERICAL CORE
# =============================================================================

def spins_from_bits(bits: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * bits.astype(np.float64)


def connected_corr(pair: np.ndarray) -> np.ndarray:
    """Return connected correlator per tile for pair shape (tiles, shots, 2)."""
    s0 = spins_from_bits(pair[:, :, 0])
    s1 = spins_from_bits(pair[:, :, 1])
    corr = np.mean(s0 * s1, axis=1)
    m0 = np.mean(s0, axis=1)
    m1 = np.mean(s1, axis=1)
    return (corr - m0 * m1).astype(np.float64)


def make_windowed_witness(record: DMRecord, window: int) -> WindowedWitness:
    pair = record.pair
    tiles, shots, _ = pair.shape
    if window <= 0 or window > shots:
        window = shots
    windows = max(1, shots // window)
    rungs = int(np.max(record.tile_rung)) + 1 if record.tile_rung.size else 0

    W = np.zeros((windows, rungs, 4), dtype=np.float64)
    counts = np.zeros((windows, rungs, 4), dtype=np.int32)
    xsum = np.zeros((windows, rungs), dtype=np.float64)
    xcount = np.zeros((windows, rungs), dtype=np.int32)

    for wi in range(windows):
        lo = wi * window
        hi = min(shots, lo + window)
        tile_conn = connected_corr(pair[:, lo:hi, :])
        for t in range(tiles):
            r = int(record.tile_rung[t])
            widx = int(record.tile_witness[t])
            if 0 <= r < rungs and 0 <= widx < 4:
                W[wi, r, widx] += float(tile_conn[t])
                counts[wi, r, widx] += 1
                xsum[wi, r] += float(record.tile_total_delay[t])
                xcount[wi, r] += 1

    nonzero = counts > 0
    W[nonzero] = W[nonzero] / counts[nonzero]
    x = np.zeros((windows, rungs), dtype=np.float64)
    okx = xcount > 0
    x[okx] = xsum[okx] / xcount[okx]
    valid = np.all(counts > 0, axis=2) & okx

    return WindowedWitness(W=W, x=x, valid=valid, windows=windows, rungs=rungs)


def wrap_pi(x: np.ndarray) -> np.ndarray:
    return np.mod(x, math.pi)


def wrap_pi_delta(d: np.ndarray) -> np.ndarray:
    y = np.mod(d + 0.5 * math.pi, math.pi) - 0.5 * math.pi
    return y


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n < 3:
        return 0.0
    x = x[:n]
    y = y[:n]
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return 0.0
    x = x - np.mean(x)
    y = y - np.mean(y)
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= EPS or ny <= EPS:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def norm_x(x: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if mode == "log":
        arr = np.log1p(np.maximum(0.0, arr))
    mn = float(np.min(arr)) if arr.size else 0.0
    mx = float(np.max(arr)) if arr.size else 0.0
    if abs(mx - mn) <= EPS:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def pi_score(x: np.ndarray, phase: np.ndarray) -> Tuple[float, str]:
    if len(x) < 3:
        return 0.0, "none"
    best_score = 0.0
    best_mode = "none"
    for mode in ("linear", "log"):
        xn = norm_x(x, mode)
        c2 = np.cos(2.0 * phase)
        s2 = np.sin(2.0 * phase)
        rc = corr_safe(xn, c2)
        rs = corr_safe(xn, s2)
        score = float(min(1.0, math.sqrt(rc * rc + rs * rs)))
        if score > best_score:
            best_score = score
            best_mode = mode
    return best_score, best_mode


def phase_velocity_r(x: np.ndarray, phase: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3:
        return 0.0, 0.0
    order = np.argsort(x)
    xs = np.asarray(x, dtype=np.float64)[order]
    ph = np.asarray(phase, dtype=np.float64)[order]

    acc = [float(ph[0])]
    mids: List[float] = []
    vels: List[float] = []
    for i in range(1, len(xs)):
        dx = float(xs[i] - xs[i - 1])
        dp = float(wrap_pi_delta(np.asarray([ph[i] - ph[i - 1]]))[0])
        acc.append(acc[-1] + dp)
        if abs(dx) > EPS:
            mids.append(0.5 * float(xs[i] + xs[i - 1]))
            vels.append(dp / dx)

    span = (max(acc) - min(acc)) / math.pi if acc else 0.0
    if len(mids) < 3:
        return 0.0, float(abs(span))
    return corr_safe(norm_x(np.asarray(mids), "log"), np.asarray(vels)), float(abs(span))


def score_description(W: np.ndarray, x: np.ndarray, valid: np.ndarray, desc: Tuple[int, int, int, int, str]) -> Score:
    primary_idx, reciprocal_idx, comp_a_idx, comp_b_idx, name = desc

    xs: List[float] = []
    energy: List[float] = []
    spec: List[float] = []
    phase: List[float] = []

    windows, rungs, _ = W.shape
    for win in range(windows):
        mask = valid[win]
        if not np.any(mask):
            continue
        p = W[win, mask, primary_idx]
        reciprocal_return = -W[win, mask, reciprocal_idx]
        comp_a = W[win, mask, comp_a_idx]
        comp_b = W[win, mask, comp_b_idx]
        e = np.sqrt(p * p + reciprocal_return * reciprocal_return)
        c = np.sqrt(comp_a * comp_a + comp_b * comp_b)
        s = e - c
        ph = wrap_pi(np.arctan2(reciprocal_return, p))

        xs.extend(x[win, mask].tolist())
        energy.extend(e.tolist())
        spec.extend(s.tolist())
        phase.extend(ph.tolist())

    xa = np.asarray(xs, dtype=np.float64)
    ea = np.asarray(energy, dtype=np.float64)
    sa = np.asarray(spec, dtype=np.float64)
    pha = np.asarray(phase, dtype=np.float64)

    if xa.size < 3:
        return Score(0.0, name, 0.0, 0.0, 0.0, 0.0, 0.0)

    ps, mode = pi_score(xa, pha)
    er_lin = corr_safe(norm_x(xa, "linear"), ea)
    er_log = corr_safe(norm_x(xa, "log"), ea)
    er = er_log if abs(er_log) > abs(er_lin) else er_lin
    pvr, span = phase_velocity_r(xa, pha)

    return Score(
        pi_score=float(ps),
        best_description=f"{name};{mode}",
        energy_mean=float(np.mean(ea)),
        specificity_mean=float(np.mean(sa)),
        energy_track_r=float(er),
        phase_vel_r=float(pvr),
        phase_span_pi=float(span),
    )


def score_canonical(ww: WindowedWitness) -> Score:
    return score_description(ww.W, ww.x, ww.valid, RECIPROCAL_DESCRIPTIONS[0])


def score_dimensional(ww: WindowedWitness) -> Score:
    scores = [score_description(ww.W, ww.x, ww.valid, desc) for desc in RECIPROCAL_DESCRIPTIONS]
    # Main ranking: phase trajectory; tie-breaks: positive energy/specificity.
    return max(scores, key=lambda s: (s.pi_score, s.energy_mean, s.specificity_mean))


def score_both(ww: WindowedWitness) -> Tuple[Score, Score]:
    return score_canonical(ww), score_dimensional(ww)


# =============================================================================
# TRANSFORMS
# =============================================================================

def clone_ww(ww: WindowedWitness) -> WindowedWitness:
    return WindowedWitness(W=ww.W.copy(), x=ww.x.copy(), valid=ww.valid.copy(), windows=ww.windows, rungs=ww.rungs)


def apply_allowed_rotation(ww: WindowedWitness, perm: Tuple[int, int, int, int]) -> WindowedWitness:
    out = clone_ww(ww)
    out.W = out.W[:, :, list(perm)].copy()
    return out


def transform_reciprocal_break(ww: WindowedWitness, rng: np.random.Generator) -> WindowedWitness:
    """Break same-rung inverse relation by permuting reciprocal channels across rungs."""
    out = clone_ww(ww)
    for win in range(out.windows):
        perm = rng.permutation(out.rungs)
        out.W[win, :, IDX_ZY] = out.W[win, perm, IDX_ZY]
        perm2 = rng.permutation(out.rungs)
        out.W[win, :, IDX_YX] = out.W[win, perm2, IDX_YX]
    return out


def transform_cross_rung_delay_scramble(ww: WindowedWitness, rng: np.random.Generator) -> WindowedWitness:
    """Keep witness manifold intact, but scramble delay order x."""
    out = clone_ww(ww)
    for win in range(out.windows):
        out.x[win, :] = out.x[win, rng.permutation(out.rungs)]
    return out


def transform_same_label_wrong_delay(ww: WindowedWitness, rng: np.random.Generator) -> WindowedWitness:
    """Preserve each label distribution but assign each label to wrong rungs independently."""
    out = clone_ww(ww)
    for win in range(out.windows):
        for ch in range(4):
            out.W[win, :, ch] = out.W[win, rng.permutation(out.rungs), ch]
    return out


def transform_non_equivalence_corruption(ww: WindowedWitness, rng: np.random.Generator) -> WindowedWitness:
    """Per-rung random channel order plus random sign flips; violates stable channel geometry."""
    out = clone_ww(ww)
    for win in range(out.windows):
        for r in range(out.rungs):
            perm = rng.permutation(4)
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=4)
            out.W[win, r, :] = out.W[win, r, perm] * signs
    return out


def make_independent_bit_shuffle_record(record: DMRecord, rng: np.random.Generator) -> DMRecord:
    pair = record.pair.copy()
    tiles, shots, _ = pair.shape
    for t in range(tiles):
        # q0 and q1 marginal distributions survive; same-shot pairing does not.
        pair[t, :, 0] = pair[t, rng.permutation(shots), 0]
        pair[t, :, 1] = pair[t, rng.permutation(shots), 1]
    return DMRecord(
        label=record.label,
        substrate=record.substrate,
        condition=record.condition,
        path=record.path,
        pair=pair,
        tile_rung=record.tile_rung,
        tile_witness=record.tile_witness,
        tile_total_delay=record.tile_total_delay,
        meta=record.meta,
    )


CONTROL_FACTORIES: Dict[str, Callable[[WindowedWitness, np.random.Generator], WindowedWitness]] = {
    "reciprocal_break": transform_reciprocal_break,
    "cross_rung_delay_scramble": transform_cross_rung_delay_scramble,
    "same_label_wrong_delay": transform_same_label_wrong_delay,
    "non_equivalence_channel_corruption": transform_non_equivalence_corruption,
}


# =============================================================================
# REPORTING
# =============================================================================

def retention(value: float, baseline: float) -> float:
    if abs(baseline) <= EPS:
        return 0.0 if abs(value) <= EPS else float("inf")
    return float(value / baseline)


def p_ge_observed(obs: float, null_vals: Sequence[float]) -> float:
    if not null_vals:
        return 1.0
    vals = np.asarray(null_vals, dtype=np.float64)
    return float((np.count_nonzero(vals >= obs) + 1.0) / (vals.size + 1.0))


def row_from_scores(
    record: DMRecord,
    transform: str,
    family: str,
    expectation: str,
    ww: WindowedWitness,
    canonical: Score,
    dimensional: Score,
    obs_canonical: Optional[Score],
    obs_dimensional: Optional[Score],
    n_perm: int,
    canonical_null: Optional[Sequence[float]] = None,
    dim_null: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    return {
        "record": record.label,
        "substrate": record.substrate,
        "condition": record.condition,
        "transform": transform,
        "family": family,
        "expectation": expectation,
        "windows": int(ww.windows),
        "points_per_window": int(ww.rungs),
        "canonical_pi_score": canonical.pi_score,
        "canonical_energy_mean": canonical.energy_mean,
        "canonical_specificity_mean": canonical.specificity_mean,
        "canonical_energy_track_r": canonical.energy_track_r,
        "canonical_phase_vel_r": canonical.phase_vel_r,
        "canonical_phase_span_pi": canonical.phase_span_pi,
        "dim_pi_score": dimensional.pi_score,
        "dim_best_description": dimensional.best_description,
        "dim_energy_mean": dimensional.energy_mean,
        "dim_specificity_mean": dimensional.specificity_mean,
        "dim_energy_track_r": dimensional.energy_track_r,
        "dim_phase_vel_r": dimensional.phase_vel_r,
        "dim_phase_span_pi": dimensional.phase_span_pi,
        "dim_retention_vs_observed": retention(dimensional.pi_score, obs_dimensional.pi_score) if obs_dimensional else "",
        "dim_p_ge_observed": p_ge_observed(obs_dimensional.pi_score, dim_null) if (obs_dimensional and dim_null is not None) else "",
        "canonical_retention_vs_observed": retention(canonical.pi_score, obs_canonical.pi_score) if obs_canonical else "",
        "canonical_p_ge_observed": p_ge_observed(obs_canonical.pi_score, canonical_null) if (obs_canonical and canonical_null is not None) else "",
        "n_perm": int(n_perm),
        "source": str(record.path),
    }


def rung_rows(record: DMRecord, ww: WindowedWitness) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for win in range(ww.windows):
        for r in range(ww.rungs):
            if not bool(ww.valid[win, r]):
                continue
            yz = ww.W[win, r, IDX_YZ]
            zy_return = -ww.W[win, r, IDX_ZY]
            energy = math.sqrt(yz * yz + zy_return * zy_return)
            comp = math.sqrt(ww.W[win, r, IDX_XY] ** 2 + ww.W[win, r, IDX_YX] ** 2)
            rows.append({
                "record": record.label,
                "condition": record.condition,
                "window": win,
                "rung": r,
                "x_total_delay": float(ww.x[win, r]),
                "XY": float(ww.W[win, r, IDX_XY]),
                "YZ": float(ww.W[win, r, IDX_YZ]),
                "ZY": float(ww.W[win, r, IDX_ZY]),
                "YX": float(ww.W[win, r, IDX_YX]),
                "canonical_energy": float(energy),
                "canonical_specificity": float(energy - comp),
                "canonical_phase": float(wrap_pi(np.asarray([math.atan2(zy_return, yz)]))[0]),
            })
    return rows


def print_record_header(record: DMRecord, ww: WindowedWitness) -> None:
    print("-" * 100)
    print(f"  {record.label:<20} {record.condition:<12} substrate={record.substrate} path={record.path.name}")
    print(f"    tiles/shots : {record.pair.shape[0]}/{record.pair.shape[1]}  windows={ww.windows}  points/window={ww.rungs}")


def print_score_line(prefix: str, canonical: Score, dimensional: Score, obs_dim: Optional[Score] = None) -> None:
    ret = ""
    if obs_dim is not None:
        ret = f"  dim_ret={retention(dimensional.pi_score, obs_dim.pi_score):.3f}"
    print(
        f"    {prefix:<34} "
        f"canon_pi={canonical.pi_score:.4f} "
        f"dim_pi={dimensional.pi_score:.4f}{ret} "
        f"dim_energy={dimensional.energy_mean:.6f} "
        f"dim_spec={dimensional.specificity_mean:+.6f} "
        f"best={dimensional.best_description}"
    )


# =============================================================================
# MAIN PROBE
# =============================================================================

def run_probe_for_record(record: DMRecord, args: argparse.Namespace, out_dir: Path, rng: np.random.Generator) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    dist_rows: List[Dict[str, Any]] = []
    all_rung_rows: List[Dict[str, Any]] = []

    ww = make_windowed_witness(record, int(args.window))
    obs_can, obs_dim = score_both(ww)

    print_record_header(record, ww)
    print_score_line("observed", obs_can, obs_dim)

    summary_rows.append(row_from_scores(
        record=record,
        transform="observed",
        family="observed",
        expectation="baseline",
        ww=ww,
        canonical=obs_can,
        dimensional=obs_dim,
        obs_canonical=obs_can,
        obs_dimensional=obs_dim,
        n_perm=0,
    ))
    all_rung_rows.extend(rung_rows(record, ww))

    # Allowed deterministic channel transformations.
    for name, perm in ALLOWED_ROTATIONS.items():
        if name == "equiv_identity":
            continue
        tww = apply_allowed_rotation(ww, perm)
        can, dim = score_both(tww)
        print_score_line(name, can, dim, obs_dim)
        summary_rows.append(row_from_scores(
            record=record,
            transform=name,
            family="allowed_equivalence",
            expectation="should_survive",
            ww=tww,
            canonical=can,
            dimensional=dim,
            obs_canonical=obs_can,
            obs_dimensional=obs_dim,
            n_perm=0,
        ))

    # Forbidden stochastic controls on witness matrices.
    for control_name, factory in CONTROL_FACTORIES.items():
        can_vals: List[float] = []
        dim_vals: List[float] = []
        dim_energy_vals: List[float] = []
        dim_spec_vals: List[float] = []
        best_desc: List[str] = []

        for i in range(int(args.n_perm)):
            cww = factory(ww, rng)
            can, dim = score_both(cww)
            can_vals.append(can.pi_score)
            dim_vals.append(dim.pi_score)
            dim_energy_vals.append(dim.energy_mean)
            dim_spec_vals.append(dim.specificity_mean)
            best_desc.append(dim.best_description)
            if args.save_distribution:
                dist_rows.append({
                    "record": record.label,
                    "condition": record.condition,
                    "control": control_name,
                    "perm_index": i,
                    "canonical_pi_score": can.pi_score,
                    "dim_pi_score": dim.pi_score,
                    "dim_best_description": dim.best_description,
                    "dim_energy_mean": dim.energy_mean,
                    "dim_specificity_mean": dim.specificity_mean,
                })

        can_mean_score = Score(
            pi_score=float(np.mean(can_vals)) if can_vals else 0.0,
            best_description="mean_control",
            energy_mean=0.0,
            specificity_mean=0.0,
            energy_track_r=0.0,
            phase_vel_r=0.0,
            phase_span_pi=0.0,
        )
        dim_mean_score = Score(
            pi_score=float(np.mean(dim_vals)) if dim_vals else 0.0,
            best_description="mean_control",
            energy_mean=float(np.mean(dim_energy_vals)) if dim_energy_vals else 0.0,
            specificity_mean=float(np.mean(dim_spec_vals)) if dim_spec_vals else 0.0,
            energy_track_r=0.0,
            phase_vel_r=0.0,
            phase_span_pi=0.0,
        )

        p_dim = p_ge_observed(obs_dim.pi_score, dim_vals)
        p_can = p_ge_observed(obs_can.pi_score, can_vals)
        print(
            f"    {control_name:<34} "
            f"canon_null={np.mean(can_vals):.4f}±{np.std(can_vals):.4f} p={p_can:.4f}  "
            f"dim_null={np.mean(dim_vals):.4f}±{np.std(dim_vals):.4f} p={p_dim:.4f}"
        )

        summary_rows.append(row_from_scores(
            record=record,
            transform=control_name,
            family="forbidden_control",
            expectation="should_collapse_or_weaken",
            ww=ww,
            canonical=can_mean_score,
            dimensional=dim_mean_score,
            obs_canonical=obs_can,
            obs_dimensional=obs_dim,
            n_perm=int(args.n_perm),
            canonical_null=can_vals,
            dim_null=dim_vals,
        ))

    # Independent bit shuffle is expensive because it recomputes from raw pair.
    if not args.skip_bit_shuffle:
        can_vals = []
        dim_vals = []
        dim_energy_vals = []
        dim_spec_vals = []
        n_bit_perm = int(args.bit_shuffle_perm or args.n_perm)
        for i in range(n_bit_perm):
            shuffled_record = make_independent_bit_shuffle_record(record, rng)
            sww = make_windowed_witness(shuffled_record, int(args.window))
            can, dim = score_both(sww)
            can_vals.append(can.pi_score)
            dim_vals.append(dim.pi_score)
            dim_energy_vals.append(dim.energy_mean)
            dim_spec_vals.append(dim.specificity_mean)
            if args.save_distribution:
                dist_rows.append({
                    "record": record.label,
                    "condition": record.condition,
                    "control": "independent_bit_shuffle",
                    "perm_index": i,
                    "canonical_pi_score": can.pi_score,
                    "dim_pi_score": dim.pi_score,
                    "dim_best_description": dim.best_description,
                    "dim_energy_mean": dim.energy_mean,
                    "dim_specificity_mean": dim.specificity_mean,
                })

        can_mean_score = Score(float(np.mean(can_vals)), "mean_control", 0.0, 0.0, 0.0, 0.0, 0.0)
        dim_mean_score = Score(
            float(np.mean(dim_vals)),
            "mean_control",
            float(np.mean(dim_energy_vals)),
            float(np.mean(dim_spec_vals)),
            0.0,
            0.0,
            0.0,
        )
        p_dim = p_ge_observed(obs_dim.pi_score, dim_vals)
        p_can = p_ge_observed(obs_can.pi_score, can_vals)
        print(
            f"    {'independent_bit_shuffle':<34} "
            f"canon_null={np.mean(can_vals):.4f}±{np.std(can_vals):.4f} p={p_can:.4f}  "
            f"dim_null={np.mean(dim_vals):.4f}±{np.std(dim_vals):.4f} p={p_dim:.4f}"
        )
        summary_rows.append(row_from_scores(
            record=record,
            transform="independent_bit_shuffle",
            family="forbidden_control",
            expectation="should_collapse_or_weaken",
            ww=ww,
            canonical=can_mean_score,
            dimensional=dim_mean_score,
            obs_canonical=obs_can,
            obs_dimensional=obs_dim,
            n_perm=n_bit_perm,
            canonical_null=can_vals,
            dim_null=dim_vals,
        ))

    return summary_rows, dist_rows, all_rung_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D_M Probe 05 — dimensional invariance vs forbidden corruption controls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--auto", action="store_true", default=True, help="Use default/discovered D_M qproj/gproj files from ghost_oracle/D_M/data.")
    p.add_argument("--path", action="append", default=[], help="Record path, optionally label=path or condition:label=path. Repeatable.")
    p.add_argument("--window", type=int, default=4096, help="Shots per snapshot/window. If larger than shots, uses all shots.")
    p.add_argument("--n-perm", type=int, default=2000, help="Permutation count for forbidden controls.")
    p.add_argument("--bit-shuffle-perm", type=int, default=0, help="Permutation count for independent_bit_shuffle. 0 means use --n-perm.")
    p.add_argument("--skip-bit-shuffle", action="store_true", help="Skip raw pair independent bit shuffle control.")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--save-distribution", action="store_true", default=True, help="Save every permutation row. Otherwise only summaries are saved.")
    p.add_argument("--out-dir", default=None, help="Optional output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))

    specs: List[Tuple[Optional[str], Optional[str], Path]] = []
    for item in args.path:
        specs.append(parse_path_arg(item))

    if args.auto or not specs:
        for label, condition, path in discover_auto_records():
            specs.append((label, condition or None, path))

    # Deduplicate while preserving order.
    seen = set()
    deduped: List[Tuple[Optional[str], Optional[str], Path]] = []
    for label, condition, path in specs:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            deduped.append((label, condition, path))

    if not deduped:
        raise FileNotFoundError(
            "No D_M records found. Use --path condition:label=path or run with --auto from the repo tree."
        )

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"dm_probe_23_dimensional_invariance_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("  D_M PROBE 05 — DIMENSIONAL INVARIANCE / FORBIDDEN-CORRUPTION CONTROLS")
    print("=" * 100)
    print(f"  Out dir : {out_dir}")
    print(f"  window  : {args.window} shots/snapshot")
    print(f"  n_perm  : {args.n_perm}")
    print("  Rule    : allowed channel rotations should survive; forbidden corruptions should collapse/weaken")

    summary_rows: List[Dict[str, Any]] = []
    dist_rows: List[Dict[str, Any]] = []
    rung_value_rows: List[Dict[str, Any]] = []
    records_loaded: List[Dict[str, Any]] = []

    for label, condition, path in deduped:
        try:
            record = load_record(path, label=label, condition=condition)
        except Exception as e:
            print(f"  [skip] {path}: {e}")
            continue

        records_loaded.append({
            "label": record.label,
            "condition": record.condition,
            "substrate": record.substrate,
            "path": str(record.path),
            **record.meta,
        })
        srows, drows, rrows = run_probe_for_record(record, args, out_dir, rng)
        summary_rows.extend(srows)
        dist_rows.extend(drows)
        rung_value_rows.extend(rrows)

    if not summary_rows:
        raise RuntimeError("No records were successfully processed.")

    summary_path = out_dir / "probe05_summary.csv"
    dist_path = out_dir / "probe05_control_distribution.csv"
    rung_path = out_dir / "probe05_rung_values.csv"
    config_path = out_dir / "probe_config.json"

    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    write_csv(dist_path, dist_rows, DIST_FIELDS)
    write_csv(rung_path, rung_value_rows, RUNG_FIELDS)
    write_json(config_path, {
        "probe": "D_M PROBE 05 — DIMENSIONAL INVARIANCE / FORBIDDEN-CORRUPTION CONTROLS",
        "args": vars(args),
        "records": records_loaded,
        "witness_labels": WITNESS_LABELS,
        "reciprocal_descriptions": [list(x) for x in RECIPROCAL_DESCRIPTIONS],
        "allowed_rotations": {k: list(v) for k, v in ALLOWED_ROTATIONS.items()},
        "interpretation": {
            "canonical_yzzy": "YZ primary, ZY reciprocal/inverted.",
            "dimensional_invariant": "Best equivalent reciprocal channel description across allowed D_M views.",
            "expected": "Allowed rotations preserve dimensional score; forbidden controls weaken/collapse it.",
        },
    })

    print("-" * 100)
    print(f"  [SAVED] {out_dir}")
    print(f"    {summary_path.name}")
    print(f"    {dist_path.name}")
    print(f"    {rung_path.name}")
    print(f"    {config_path.name}")
    print("=" * 100)


if __name__ == "__main__":
    main()
