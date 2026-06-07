#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
D_M PROBE 05B — DIMENSIONAL ERROR-CORRECTION / CORRUPTION BOUNDARY
==============================================================================

Purpose
-------
Probe 05 showed that single forbidden controls often do NOT collapse D_M. That is
not necessarily a failure if D_M is a dimensional error-correcting operator.

This probe asks the sharper question:

    How many independent structural faults can the D_M manifold absorb before
    dimensional agreement becomes unrecoverable?

The probe builds a clean D_M manifold from qproj/gproj records, then applies
compound corruptions at increasing fault depth k:

    k=0  clean manifold
    k=1  one independent structural violation
    k=2  two independent structural violations
    ...

The expected pattern for dimensional error correction is:

    allowed channel re-descriptions survive
    single-fault corruption often survives / partially survives
    compound corruption weakens progressively
    high-depth corruption crosses a collapse boundary

Core idea
---------
D_M channels are not arbitrary labels. XY/YZ/ZY/YX are channel views of the same
dimensional agreement manifold. Therefore allowed channel rotations are scored
through a best-repair search over equivalent reciprocal channel descriptions.

A manifold is considered corrupted only when the remaining dimensions can no
longer repair/reconstruct the original agreement:

    same-shot pair structure
    reciprocal-return structure
    delay/rung ordering
    channel-basis agreement
    energy/specificity geometry

Inputs
------
Canonical D_M qproj/gproj .npz files with:

    pair[tile, shot, 2]
    tile_rung_index[tile]
    tile_witness_index[tile]      0=XY, 1=YZ, 2=ZY, 3=YX
    tile_total_delay_dt[tile]

Outputs
-------
Creates:

    ghost_oracle/D_M/analysis/dm_probe05b_corruption_boundary_<timestamp>/

with:

    probe05b_depth_summary.csv
    probe05b_trial_rows.csv
    probe05b_allowed_rows.csv
    probe05b_observed_rows.csv
    probe_config.json

Example
-------
From repo root:

    python ghost_oracle/D_M/probes/d_m_probe05b_corruption_boundary.py --auto ^
      --window 4096 --trials-per-depth 500 --save-trials

==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# PATHS / DEFAULTS
# =============================================================================

PROBE_DIR = Path(__file__).resolve().parent
D_M_DIR = PROBE_DIR.parent
DATA_DIR = D_M_DIR / "data"
ANALYSIS_DIR = PROBE_DIR / "analyze"

DEFAULT_QPROJ_NULL = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fm4ihvjngc73aq3ccg.npz"
DEFAULT_QPROJ_BASE = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8flk2jo3njc73f0g560.npz"
DEFAULT_QPROJ_OFFSET = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fl82bo3njc73f0fgd0.npz"

DEFAULT_GPROJ_NULL = DATA_DIR / "dm_gpu_data_null_4096shots_seed9031229662612491082.npz"
DEFAULT_GPROJ_BASE = DATA_DIR / "dm_gpu_data_base_delay_4096shots_seed2669559634056472362.npz"
DEFAULT_GPROJ_OFFSET = DATA_DIR / "dm_gpu_data_offset_deformed_4096shots_seed6727069190982977623.npz"

WITNESS_LABELS = ["XY", "YZ", "ZY", "YX"]
WITNESS_TO_INDEX = {x: i for i, x in enumerate(WITNESS_LABELS)}
EPS = 1.0e-12

# Equivalent D_M channel descriptions.
# name, primary_idx, return_source_idx, comparison_a_idx, comparison_b_idx
# The return axis is interpreted as -channel[return_source_idx].
ORIENTATIONS: List[Tuple[str, int, int, int, int]] = [
    ("YZ<-ZY canonical", 1, 2, 0, 3),
    ("ZY<-YZ reciprocal", 2, 1, 3, 0),
    ("XY<-YX pair", 0, 3, 1, 2),
    ("YX<-XY reciprocal", 3, 0, 2, 1),
]

FAULTS = [
    "reciprocal_break",
    "cross_rung_delay_scramble",
    "same_label_wrong_delay",
    "non_equivalence_channel_corruption",
    "independent_bit_shuffle",
]

ALLOWED_TRANSFORMS = [
    "identity",
    "equiv_pair_swap",
    "equiv_reciprocal_swap",
    "equiv_cyclic_rotation",
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
    tile_base_delay: np.ndarray
    tile_offset: np.ndarray


@dataclass
class Manifold:
    W: np.ndarray                    # shape (windows, rungs, 4)
    delays: np.ndarray               # shape (rungs,)
    windows: int
    rungs: int
    source_label: str
    substrate: str
    condition: str


@dataclass
class OrientationFeatures:
    name: str
    primary: np.ndarray
    ret: np.ndarray
    comparison_a: np.ndarray
    comparison_b: np.ndarray
    energy: np.ndarray
    comparison_energy: np.ndarray
    specificity: np.ndarray
    phase: np.ndarray
    cos2: np.ndarray
    sin2: np.ndarray
    delay: np.ndarray
    phase_score: float
    phase_mode: str
    energy_track_r: float
    specificity_track_r: float
    phase_velocity_r: float
    phase_span_pi: float
    mean_energy: float
    mean_specificity: float


@dataclass
class CleanTemplate:
    source_label: str
    substrate: str
    condition: str
    orientation: OrientationFeatures
    orientation_name: str
    internal_score: float


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


def optional_int_array(z: Any, name: str, tiles: int, default: int = 0) -> np.ndarray:
    if name in z.files:
        arr = np.asarray(z[name], dtype=np.int32)
        if arr.shape[0] == tiles:
            return np.ascontiguousarray(arr)
    return np.full((tiles,), int(default), dtype=np.int32)


# =============================================================================
# LOAD D_M RECORDS
# =============================================================================

def load_record(path: Path, label: str, substrate: str, condition: str) -> DMRecord:
    if not path.exists():
        raise FileNotFoundError(f"Missing D_M record: {path}")

    z = np.load(path, allow_pickle=True)

    if "pair" in z.files:
        pair = np.asarray(z["pair"], dtype=np.uint8)
    else:
        keys = sorted(
            [k for k in z.files if k.startswith("pair_tile")],
            key=lambda k: int(k.replace("pair_tile", "")),
        )
        if not keys:
            raise KeyError(f"{path} has no pair or pair_tile* arrays")
        pair = np.stack([np.asarray(z[k], dtype=np.uint8) for k in keys], axis=0)

    if pair.ndim != 3 or pair.shape[2] != 2:
        raise ValueError(f"{path} pair must have shape (tiles, shots, 2), got {pair.shape}")

    pair = np.ascontiguousarray(pair.astype(np.uint8, copy=False))
    tiles = int(pair.shape[0])

    if "tile_rung_index" in z.files:
        tile_rung = np.asarray(z["tile_rung_index"], dtype=np.int32)
    else:
        tile_rung = (np.arange(tiles) // 4).astype(np.int32)

    if "tile_witness_index" in z.files:
        tile_witness = np.asarray(z["tile_witness_index"], dtype=np.int32)
    elif "tile_witness_label" in z.files:
        labels = decode_str_array(z["tile_witness_label"])
        tile_witness = np.asarray(
            [WITNESS_TO_INDEX.get(labels[i], i % 4) for i in range(min(len(labels), tiles))],
            dtype=np.int32,
        )
    else:
        tile_witness = (np.arange(tiles) % 4).astype(np.int32)

    if tile_rung.shape[0] != tiles or tile_witness.shape[0] != tiles:
        raise ValueError(f"{path} tile metadata length mismatch for tiles={tiles}")

    tile_total = optional_int_array(z, "tile_total_delay_dt", tiles, 0)
    tile_base = optional_int_array(z, "tile_base_delay_dt", tiles, 0)
    tile_offset = optional_int_array(z, "tile_offset_dt", tiles, 0)

    return DMRecord(
        label=label,
        substrate=substrate,
        condition=condition,
        path=path,
        pair=pair,
        tile_rung=np.ascontiguousarray(tile_rung.astype(np.int32, copy=False)),
        tile_witness=np.ascontiguousarray(tile_witness.astype(np.int32, copy=False)),
        tile_total_delay=np.ascontiguousarray(tile_total.astype(np.int32, copy=False)),
        tile_base_delay=np.ascontiguousarray(tile_base.astype(np.int32, copy=False)),
        tile_offset=np.ascontiguousarray(tile_offset.astype(np.int32, copy=False)),
    )


def build_auto_records(args: argparse.Namespace) -> List[DMRecord]:
    candidates = [
        ("qproj_null", "qproj", "null", Path(args.qproj_null)),
        ("qproj_base_only", "qproj", "base_only", Path(args.qproj_base)),
        ("qproj_offset_on", "qproj", "offset_on", Path(args.qproj_offset)),
        ("gproj_null", "gproj", "null", Path(args.gproj_null)),
        ("gproj_base_only", "gproj", "base_only", Path(args.gproj_base)),
        ("gproj_offset_on", "gproj", "offset_on", Path(args.gproj_offset)),
    ]

    records: List[DMRecord] = []
    for label, substrate, condition, path in candidates:
        if path.exists():
            records.append(load_record(path, label, substrate, condition))
        else:
            print(f"  [skip] missing {label}: {path}")
    return records


def build_manual_records(args: argparse.Namespace) -> List[DMRecord]:
    records: List[DMRecord] = []
    for i, raw in enumerate(args.files or []):
        path = Path(raw)
        label = path.stem
        substrate = args.substrate or "manual"
        condition = args.condition or "manual"
        records.append(load_record(path, label, substrate, condition))
    return records


# =============================================================================
# NUMERIC HELPERS
# =============================================================================

def safe_mean(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    if a.size <= 0:
        return 0.0
    return float(np.mean(a))


def clip01(x: float) -> float:
    if not math.isfinite(float(x)):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n < 3:
        return 0.0
    a = a[:n]
    b = b[:n]
    da = a - np.mean(a)
    db = b - np.mean(b)
    va = float(np.dot(da, da))
    vb = float(np.dot(db, db))
    if va <= EPS or vb <= EPS:
        return 0.0
    return float(np.dot(da, db) / math.sqrt(va * vb))


def corr01(x: np.ndarray, y: np.ndarray) -> float:
    return clip01(0.5 * (pearson(x, y) + 1.0))


def cosine01(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n <= 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    return clip01(0.5 * (c + 1.0))


def rel_rmse_factor(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n <= 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    denom = float(np.linalg.norm(a) / math.sqrt(max(1, n))) + EPS
    rmse = float(np.linalg.norm(a - b) / math.sqrt(max(1, n)))
    return clip01(math.exp(-rmse / denom))


def strength_ratio_factor(candidate: float, reference: float) -> float:
    ref = abs(float(reference)) + EPS
    cand = abs(float(candidate))
    ratio = cand / ref
    # Symmetric around 1.0. Values too weak or too over-amplified both lose trust.
    return clip01(math.exp(-abs(math.log(max(ratio, EPS)))))


def phase_similarity(ref_phase: np.ndarray, cand_phase: np.ndarray) -> float:
    a = np.asarray(ref_phase, dtype=np.float64).reshape(-1)
    b = np.asarray(cand_phase, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n <= 0:
        return 0.0
    # Phase is pi-periodic, so compare 2*phase.
    sim = 0.5 * (np.cos(2.0 * (a[:n] - b[:n])) + 1.0)
    return clip01(float(np.mean(sim)))


def unwrap_pi_phase(phase: np.ndarray) -> np.ndarray:
    p = np.asarray(phase, dtype=np.float64).reshape(-1)
    if p.size <= 1:
        return p.copy()
    out = np.zeros_like(p)
    out[0] = p[0]
    acc = p[0]
    for i in range(1, p.size):
        d = p[i] - p[i - 1]
        d = ((d + 0.5 * math.pi) % math.pi) - 0.5 * math.pi
        acc += d
        out[i] = acc
    return out


def normalized_delay_modes(delay: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    x = np.asarray(delay, dtype=np.float64).reshape(-1)
    out: List[Tuple[str, np.ndarray]] = []
    for name, raw in [("linear", x), ("log", np.log1p(np.maximum(0.0, x)) )]:
        mn = float(np.min(raw)) if raw.size else 0.0
        mx = float(np.max(raw)) if raw.size else 0.0
        if abs(mx - mn) <= EPS:
            out.append((name, np.zeros_like(raw)))
        else:
            out.append((name, (raw - mn) / (mx - mn)))
    return out


def pi_periodic_score(delay: np.ndarray, phase: np.ndarray) -> Tuple[float, str]:
    d = np.asarray(delay, dtype=np.float64).reshape(-1)
    p = np.asarray(phase, dtype=np.float64).reshape(-1)
    n = min(d.size, p.size)
    if n < 3:
        return 0.0, "none"
    d = d[:n]
    p = p[:n]
    c2 = np.cos(2.0 * p)
    s2 = np.sin(2.0 * p)

    best = 0.0
    best_mode = "none"
    for mode, x in normalized_delay_modes(d):
        rc = pearson(x, c2)
        rs = pearson(x, s2)
        score = math.sqrt(rc * rc + rs * rs)
        if math.isfinite(score) and score > best:
            best = min(1.0, score)
            best_mode = mode
    return float(best), best_mode


def phase_velocity_r(delay: np.ndarray, phase: np.ndarray) -> Tuple[float, float]:
    d = np.asarray(delay, dtype=np.float64).reshape(-1)
    p = unwrap_pi_phase(phase)
    n = min(d.size, p.size)
    if n < 3:
        return 0.0, 0.0
    d = d[:n]
    p = p[:n]
    dd = np.diff(d)
    dp = np.diff(p)
    mask = np.abs(dd) > EPS
    if int(np.sum(mask)) < 3:
        span = float((np.max(p) - np.min(p)) / math.pi) if p.size else 0.0
        return 0.0, abs(span)
    vel = dp[mask] / dd[mask]
    mid = 0.5 * (d[1:][mask] + d[:-1][mask])
    return pearson(np.log1p(np.maximum(0.0, mid)), vel), abs(float((np.max(p) - np.min(p)) / math.pi))


# =============================================================================
# MANIFOLD CONSTRUCTION
# =============================================================================

def compute_manifold_from_pair(
    record: DMRecord,
    window: int,
    rng: Optional[np.random.Generator] = None,
    independent_bit_shuffle: bool = False,
) -> Manifold:
    pair = record.pair
    tiles, shots, _ = pair.shape
    if window <= 0 or window > shots:
        window = shots
    windows = max(1, shots // window)
    usable_shots = windows * window

    rungs = int(np.max(record.tile_rung)) + 1 if tiles else 0
    W = np.zeros((windows, rungs, 4), dtype=np.float64)
    counts = np.zeros((windows, rungs, 4), dtype=np.int32)

    for wi in range(windows):
        lo = wi * window
        hi = lo + window
        for t in range(tiles):
            rung = int(record.tile_rung[t])
            witness = int(record.tile_witness[t])
            if rung < 0 or rung >= rungs or witness < 0 or witness >= 4:
                continue

            b0 = pair[t, lo:hi, 0].astype(np.uint8, copy=True)
            b1 = pair[t, lo:hi, 1].astype(np.uint8, copy=True)

            if independent_bit_shuffle:
                if rng is None:
                    rng = np.random.default_rng(1234)
                b0 = b0[rng.permutation(b0.size)]
                b1 = b1[rng.permutation(b1.size)]

            s0 = 1.0 - 2.0 * b0.astype(np.float64)
            s1 = 1.0 - 2.0 * b1.astype(np.float64)
            corr = float(np.mean(s0 * s1))
            connected = corr - float(np.mean(s0)) * float(np.mean(s1))

            W[wi, rung, witness] += connected
            counts[wi, rung, witness] += 1

    mask = counts > 0
    W[mask] = W[mask] / counts[mask]

    delays = np.zeros((rungs,), dtype=np.float64)
    dcounts = np.zeros((rungs,), dtype=np.int32)
    for t in range(tiles):
        r = int(record.tile_rung[t])
        if 0 <= r < rungs:
            delays[r] += float(record.tile_total_delay[t])
            dcounts[r] += 1
    dmask = dcounts > 0
    delays[dmask] = delays[dmask] / dcounts[dmask]

    # If the null file has all-zero delays, use rung index as a harmless trajectory
    # coordinate for internal comparisons, but flat data will still score as null.
    if np.max(delays) - np.min(delays) <= EPS:
        delays = np.arange(rungs, dtype=np.float64)

    return Manifold(
        W=np.ascontiguousarray(W),
        delays=np.ascontiguousarray(delays),
        windows=windows,
        rungs=rungs,
        source_label=record.label,
        substrate=record.substrate,
        condition=record.condition,
    )


def delay_vector_for_manifold(m: Manifold) -> np.ndarray:
    return np.tile(m.delays.reshape(1, -1), (m.windows, 1)).reshape(-1)


def orientation_features(m: Manifold, orient: Tuple[str, int, int, int, int]) -> OrientationFeatures:
    name, p_idx, r_idx, ca_idx, cb_idx = orient
    W = m.W

    P = W[:, :, p_idx].reshape(-1).astype(np.float64)
    R = (-W[:, :, r_idx]).reshape(-1).astype(np.float64)
    CA = W[:, :, ca_idx].reshape(-1).astype(np.float64)
    CB = (-W[:, :, cb_idx]).reshape(-1).astype(np.float64)

    energy = np.sqrt(P * P + R * R)
    comp_energy = np.sqrt(CA * CA + CB * CB)
    specificity = energy - comp_energy
    phase = np.mod(np.arctan2(R, P), math.pi)
    c2 = np.cos(2.0 * phase)
    s2 = np.sin(2.0 * phase)
    delay = delay_vector_for_manifold(m)

    phase_score, phase_mode = pi_periodic_score(delay, phase)
    e_r = pearson(np.log1p(np.maximum(0.0, delay)), energy)
    s_r = pearson(np.log1p(np.maximum(0.0, delay)), specificity)
    pv_r, span = phase_velocity_r(delay, phase)

    return OrientationFeatures(
        name=name,
        primary=P,
        ret=R,
        comparison_a=CA,
        comparison_b=CB,
        energy=energy,
        comparison_energy=comp_energy,
        specificity=specificity,
        phase=phase,
        cos2=c2,
        sin2=s2,
        delay=delay,
        phase_score=float(phase_score),
        phase_mode=phase_mode,
        energy_track_r=float(e_r),
        specificity_track_r=float(s_r),
        phase_velocity_r=float(pv_r),
        phase_span_pi=float(span),
        mean_energy=safe_mean(energy),
        mean_specificity=safe_mean(specificity),
    )


def internal_orientation_score(f: OrientationFeatures) -> float:
    # Internal score chooses the most D_M-like orientation inside one record.
    # Keep this gentle: phase + energy + positive specificity + temporal tracking.
    e = max(0.0, f.mean_energy)
    spec = max(0.0, f.mean_specificity)
    track = 0.5 * (abs(f.energy_track_r) + abs(f.specificity_track_r))
    return float(
        0.35 * e
        + 0.25 * spec
        + 0.20 * f.phase_score
        + 0.10 * track
        + 0.10 * abs(f.phase_velocity_r)
    )


def choose_clean_template(m: Manifold) -> CleanTemplate:
    feats = [orientation_features(m, o) for o in ORIENTATIONS]
    best = max(feats, key=internal_orientation_score)
    return CleanTemplate(
        source_label=m.source_label,
        substrate=m.substrate,
        condition=m.condition,
        orientation=best,
        orientation_name=best.name,
        internal_score=internal_orientation_score(best),
    )


# =============================================================================
# SURVIVAL SCORING
# =============================================================================

def survival_components(ref: OrientationFeatures, cand: OrientationFeatures) -> Dict[str, float]:
    ref_pair = np.stack([ref.primary, ref.ret], axis=1).reshape(-1)
    cand_pair = np.stack([cand.primary, cand.ret], axis=1).reshape(-1)

    channel_shape = cosine01(ref_pair, cand_pair)
    channel_corr = corr01(ref_pair, cand_pair)
    energy_shape = rel_rmse_factor(ref.energy, cand.energy)
    energy_order = corr01(ref.energy, cand.energy)
    phase_match = phase_similarity(ref.phase, cand.phase)

    ref_unwrap = unwrap_pi_phase(ref.phase)
    cand_unwrap = unwrap_pi_phase(cand.phase)
    if ref_unwrap.size >= 3 and cand_unwrap.size >= 3:
        phase_order = corr01(np.diff(ref_unwrap), np.diff(cand_unwrap))
    else:
        phase_order = 0.0

    specificity_shape = rel_rmse_factor(ref.specificity, cand.specificity)
    specificity_order = corr01(ref.specificity, cand.specificity)
    energy_strength = strength_ratio_factor(cand.mean_energy, ref.mean_energy)

    # Positive specificity retention only rewards preserved D_M separation from comparison channels.
    if ref.mean_specificity > EPS:
        specificity_retention = clip01(cand.mean_specificity / (ref.mean_specificity + EPS))
    else:
        specificity_retention = 1.0 if abs(cand.mean_specificity) <= EPS else 0.5

    delay_tracking = 0.5 * (
        corr01(ref.energy, cand.energy)
        + corr01(unwrap_pi_phase(ref.phase), unwrap_pi_phase(cand.phase))
    )

    # Phase self-coherence is not enough by itself, but it helps distinguish a coherent repair
    # from a random high-energy survivor.
    phase_self = cand.phase_score

    return {
        "channel_shape": clip01(0.5 * channel_shape + 0.5 * channel_corr),
        "energy_shape": clip01(0.5 * energy_shape + 0.5 * energy_order),
        "phase_match": clip01(0.7 * phase_match + 0.3 * phase_order),
        "delay_integrity": clip01(delay_tracking),
        "specificity_shape": clip01(0.5 * specificity_shape + 0.5 * specificity_order),
        "specificity_retention": clip01(specificity_retention),
        "energy_strength": clip01(energy_strength),
        "phase_self": clip01(phase_self),
    }


def composite_survival(components: Dict[str, float]) -> float:
    # Geometric mean: a manifold must preserve multiple structural promises at once.
    weights = {
        "channel_shape": 1.25,
        "energy_shape": 1.00,
        "phase_match": 1.00,
        "delay_integrity": 1.25,
        "specificity_shape": 1.00,
        "specificity_retention": 1.00,
        "energy_strength": 0.75,
        "phase_self": 0.50,
    }
    total_w = sum(weights.values())
    acc = 0.0
    for k, w in weights.items():
        v = clip01(components.get(k, 0.0))
        acc += w * math.log(max(v, 1.0e-9))
    return clip01(math.exp(acc / total_w))


def best_repaired_survival(template: CleanTemplate, candidate: Manifold) -> Dict[str, Any]:
    ref = template.orientation
    rows: List[Dict[str, Any]] = []
    for o in ORIENTATIONS:
        cand = orientation_features(candidate, o)
        comps = survival_components(ref, cand)
        surv = composite_survival(comps)
        rows.append({
            "survival": surv,
            "candidate_orientation": cand.name,
            "candidate_phase_score": cand.phase_score,
            "candidate_phase_mode": cand.phase_mode,
            "candidate_energy": cand.mean_energy,
            "candidate_specificity": cand.mean_specificity,
            **comps,
        })
    return max(rows, key=lambda r: float(r["survival"]))


# =============================================================================
# TRANSFORMS / FAULTS
# =============================================================================

def copy_manifold(m: Manifold, W: Optional[np.ndarray] = None, suffix: str = "") -> Manifold:
    return Manifold(
        W=np.ascontiguousarray(np.array(m.W if W is None else W, dtype=np.float64, copy=True)),
        delays=np.ascontiguousarray(np.array(m.delays, dtype=np.float64, copy=True)),
        windows=m.windows,
        rungs=m.rungs,
        source_label=m.source_label + suffix,
        substrate=m.substrate,
        condition=m.condition,
    )


def apply_allowed_transform(m: Manifold, name: str) -> Manifold:
    W = np.array(m.W, copy=True)
    if name == "identity":
        pass
    elif name == "equiv_pair_swap":
        # Swap the two reciprocal pairs: XY/YX <-> YZ/ZY.
        W = W[:, :, [1, 0, 3, 2]]
    elif name == "equiv_reciprocal_swap":
        # Swap primary/return sides inside each reciprocal pair.
        W = W[:, :, [3, 2, 1, 0]]
    elif name == "equiv_cyclic_rotation":
        # Coordinate rotation over channel basis.
        W = np.roll(W, shift=1, axis=2)
    else:
        raise ValueError(f"unknown allowed transform: {name}")
    return copy_manifold(m, W, suffix=f"::{name}")


def fault_reciprocal_break(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(W, copy=True)
    # Damage return-side axes for both reciprocal pairs while preserving their marginal distributions.
    for ch in (2, 3):
        flat = out[:, :, ch].reshape(-1)
        out[:, :, ch] = flat[rng.permutation(flat.size)].reshape(out.shape[0], out.shape[1])
    return out


def fault_cross_rung_delay_scramble(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(W, copy=True)
    # Same permutation for all channels: tuple survives, but trajectory/delay order is wrong.
    for w in range(out.shape[0]):
        perm = rng.permutation(out.shape[1])
        out[w, :, :] = out[w, perm, :]
    return out


def fault_same_label_wrong_delay(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(W, copy=True)
    # Each channel keeps its label but gets its own wrong delay path.
    for w in range(out.shape[0]):
        for ch in range(out.shape[2]):
            perm = rng.permutation(out.shape[1])
            out[w, :, ch] = out[w, perm, ch]
    return out


def fault_non_equivalence_channel_corruption(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(W, copy=True)
    # Non-equivalence corruption: per-point channel mixing + sign flips.
    # This is deliberately not a valid D_M channel rotation.
    for w in range(out.shape[0]):
        for r in range(out.shape[1]):
            perm = rng.permutation(4)
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=4)
            out[w, r, :] = out[w, r, perm] * signs
    return out


def apply_w_fault(W: np.ndarray, fault: str, rng: np.random.Generator) -> np.ndarray:
    if fault == "reciprocal_break":
        return fault_reciprocal_break(W, rng)
    if fault == "cross_rung_delay_scramble":
        return fault_cross_rung_delay_scramble(W, rng)
    if fault == "same_label_wrong_delay":
        return fault_same_label_wrong_delay(W, rng)
    if fault == "non_equivalence_channel_corruption":
        return fault_non_equivalence_channel_corruption(W, rng)
    raise ValueError(f"unknown W fault: {fault}")


def apply_fault_combo(
    record: DMRecord,
    clean: Manifold,
    faults: Sequence[str],
    window: int,
    rng: np.random.Generator,
) -> Manifold:
    # Independent bit shuffle must be applied before connected-correlation reduction.
    if "independent_bit_shuffle" in faults:
        m = compute_manifold_from_pair(record, window, rng=rng, independent_bit_shuffle=True)
    else:
        m = copy_manifold(clean)

    W = np.array(m.W, copy=True)
    for f in faults:
        if f == "independent_bit_shuffle":
            continue
        W = apply_w_fault(W, f, rng)

    return copy_manifold(m, W, suffix="::" + "+".join(faults))


# =============================================================================
# SUMMARY HELPERS
# =============================================================================

def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def summarize_depth_rows(rows: Sequence[Dict[str, Any]], threshold: float, preserve_threshold: float) -> Dict[str, Any]:
    vals = [float(r["survival"]) for r in rows]
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    best_row = max(rows, key=lambda r: float(r["survival"]))
    worst_row = min(rows, key=lambda r: float(r["survival"]))
    return {
        "n_trials": len(vals),
        "mean_survival": float(np.mean(arr)),
        "std_survival": float(np.std(arr)),
        "min_survival": float(np.min(arr)),
        "p10_survival": quantile(vals, 0.10),
        "median_survival": quantile(vals, 0.50),
        "p90_survival": quantile(vals, 0.90),
        "max_survival": float(np.max(arr)),
        "frac_below_collapse_threshold": float(np.mean(arr < threshold)),
        "frac_above_preserve_threshold": float(np.mean(arr >= preserve_threshold)),
        "best_combo": best_row.get("fault_combo", ""),
        "worst_combo": worst_row.get("fault_combo", ""),
    }


def combo_name(combo: Sequence[str]) -> str:
    return "+".join(combo) if combo else "clean"


def all_combos_at_depth(depth: int) -> List[Tuple[str, ...]]:
    return list(itertools.combinations(FAULTS, depth))


def sample_combo(depth: int, rng: np.random.Generator) -> Tuple[str, ...]:
    return tuple(rng.choice(np.asarray(FAULTS), size=depth, replace=False).tolist())


# =============================================================================
# PROBE RUNNER
# =============================================================================

def run_one_record(
    record: DMRecord,
    null_manifold: Optional[Manifold],
    out_dir: Path,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    clean = compute_manifold_from_pair(record, int(args.window), rng=rng, independent_bit_shuffle=False)
    template = choose_clean_template(clean)
    obs = template.orientation

    observed_rows: List[Dict[str, Any]] = []
    allowed_rows: List[Dict[str, Any]] = []
    trial_rows: List[Dict[str, Any]] = []
    depth_summary_rows: List[Dict[str, Any]] = []

    null_survival = None
    if null_manifold is not None and record.condition != "null":
        null_survival = best_repaired_survival(template, null_manifold)

    observed_row = {
        "record": record.label,
        "substrate": record.substrate,
        "condition": record.condition,
        "path": str(record.path),
        "tiles": int(record.pair.shape[0]),
        "shots": int(record.pair.shape[1]),
        "windows": int(clean.windows),
        "rungs": int(clean.rungs),
        "best_orientation": template.orientation_name,
        "internal_score": template.internal_score,
        "phase_score": obs.phase_score,
        "phase_mode": obs.phase_mode,
        "mean_energy": obs.mean_energy,
        "mean_specificity": obs.mean_specificity,
        "energy_track_r": obs.energy_track_r,
        "specificity_track_r": obs.specificity_track_r,
        "phase_velocity_r": obs.phase_velocity_r,
        "phase_span_pi": obs.phase_span_pi,
        "null_survival": "" if null_survival is None else null_survival["survival"],
        "null_best_orientation": "" if null_survival is None else null_survival["candidate_orientation"],
    }
    observed_rows.append(observed_row)

    print(f"  {record.label:<20} {record.condition:<11} substrate={record.substrate} path={record.path.name}")
    print(f"    tiles/shots : {record.pair.shape[0]}/{record.pair.shape[1]}  windows={clean.windows}  rungs={clean.rungs}")
    print(
        f"    observed    : phase={obs.phase_score:.4f} mode={obs.phase_mode:<6} "
        f"energy={obs.mean_energy:.6f} spec={obs.mean_specificity:+.6f} "
        f"best={template.orientation_name}"
    )
    if null_survival is not None:
        print(
            f"    null floor  : survival={float(null_survival['survival']):.4f} "
            f"best={null_survival['candidate_orientation']}"
        )

    # Allowed channel re-descriptions.
    for name in ALLOWED_TRANSFORMS:
        transformed = apply_allowed_transform(clean, name)
        surv = best_repaired_survival(template, transformed)
        row = {
            "record": record.label,
            "substrate": record.substrate,
            "condition": record.condition,
            "transform": name,
            "survival": surv["survival"],
            "candidate_orientation": surv["candidate_orientation"],
            "candidate_phase_score": surv["candidate_phase_score"],
            "candidate_energy": surv["candidate_energy"],
            "candidate_specificity": surv["candidate_specificity"],
            **{f"component_{k}": v for k, v in surv.items() if k in (
                "channel_shape", "energy_shape", "phase_match", "delay_integrity",
                "specificity_shape", "specificity_retention", "energy_strength", "phase_self"
            )},
        }
        allowed_rows.append(row)

    print("    allowed     : " + "  ".join(
        f"{r['transform']}={float(r['survival']):.3f}" for r in allowed_rows[-len(ALLOWED_TRANSFORMS):]
    ))

    if record.condition == "null" and not args.test_null_depths:
        print("    depth curve : skipped for null condition")
        print("-" * 100)
        return observed_rows, allowed_rows, trial_rows, depth_summary_rows

    max_depth = min(int(args.max_depth), len(FAULTS))
    for depth in range(1, max_depth + 1):
        depth_rows: List[Dict[str, Any]] = []
        combos = all_combos_at_depth(depth)

        # Cover every combo at least once, then fill the rest with random samples.
        trial_combos: List[Tuple[str, ...]] = []
        if bool(args.exhaustive_combos) or len(combos) >= int(args.trials_per_depth):
            if bool(args.exhaustive_combos):
                trial_combos = combos
            else:
                idx = rng.choice(np.arange(len(combos)), size=int(args.trials_per_depth), replace=True)
                trial_combos = [combos[int(i)] for i in idx]
        else:
            trial_combos.extend(combos)
            while len(trial_combos) < int(args.trials_per_depth):
                trial_combos.append(sample_combo(depth, rng))

        for trial_index, combo in enumerate(trial_combos):
            trial_seed = int(rng.integers(0, 2**31 - 1))
            local_rng = np.random.default_rng(trial_seed)
            corrupted = apply_fault_combo(record, clean, combo, int(args.window), local_rng)
            surv = best_repaired_survival(template, corrupted)
            row = {
                "record": record.label,
                "substrate": record.substrate,
                "condition": record.condition,
                "depth": depth,
                "trial": trial_index,
                "trial_seed": trial_seed,
                "fault_combo": combo_name(combo),
                "survival": surv["survival"],
                "candidate_orientation": surv["candidate_orientation"],
                "candidate_phase_score": surv["candidate_phase_score"],
                "candidate_phase_mode": surv["candidate_phase_mode"],
                "candidate_energy": surv["candidate_energy"],
                "candidate_specificity": surv["candidate_specificity"],
                "channel_shape": surv["channel_shape"],
                "energy_shape": surv["energy_shape"],
                "phase_match": surv["phase_match"],
                "delay_integrity": surv["delay_integrity"],
                "specificity_shape": surv["specificity_shape"],
                "specificity_retention": surv["specificity_retention"],
                "energy_strength": surv["energy_strength"],
                "phase_self": surv["phase_self"],
            }
            depth_rows.append(row)
            if args.save_trials:
                trial_rows.append(row)

        summary = summarize_depth_rows(depth_rows, float(args.collapse_threshold), float(args.preserve_threshold))
        summary_row = {
            "record": record.label,
            "substrate": record.substrate,
            "condition": record.condition,
            "depth": depth,
            "collapse_threshold": float(args.collapse_threshold),
            "preserve_threshold": float(args.preserve_threshold),
            "null_survival": "" if null_survival is None else null_survival["survival"],
            **summary,
        }
        depth_summary_rows.append(summary_row)

    # Collapse border summaries.
    median_border = None
    p90_border = None
    null_border = None
    null_level = None if null_survival is None else float(null_survival["survival"])

    for row in depth_summary_rows[-max_depth:]:
        d = int(row["depth"])
        if median_border is None and float(row["median_survival"]) < float(args.collapse_threshold):
            median_border = d
        if p90_border is None and float(row["p90_survival"]) < float(args.preserve_threshold):
            p90_border = d
        if null_level is not None and null_border is None and float(row["median_survival"]) <= null_level:
            null_border = d

    print("    depth curve :")
    for row in depth_summary_rows[-max_depth:]:
        print(
            f"      k={int(row['depth'])}  mean={float(row['mean_survival']):.3f} "
            f"med={float(row['median_survival']):.3f} p10={float(row['p10_survival']):.3f} "
            f"p90={float(row['p90_survival']):.3f} "
            f"below={float(row['frac_below_collapse_threshold']):.2f} "
            f"worst={row['worst_combo']}"
        )
    print(
        f"    border      : median<thr @ {median_border if median_border is not None else 'not reached'}; "
        f"p90<preserve @ {p90_border if p90_border is not None else 'not reached'}; "
        f"median<=null @ {null_border if null_border is not None else 'not reached'}"
    )
    print("-" * 100)

    return observed_rows, allowed_rows, trial_rows, depth_summary_rows


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D_M Probe 05B — compound corruption boundary / dimensional error-correction threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--auto", action="store_true", default=True, help="Use default qproj/gproj null/base/offset files.")
    p.add_argument("--files", nargs="*", default=None, help="Manual .npz files to test.")
    p.add_argument("--substrate", default=None, help="Manual substrate label when using --files.")
    p.add_argument("--condition", default=None, help="Manual condition label when using --files.")

    p.add_argument("--qproj-null", default=str(DEFAULT_QPROJ_NULL))
    p.add_argument("--qproj-base", default=str(DEFAULT_QPROJ_BASE))
    p.add_argument("--qproj-offset", default=str(DEFAULT_QPROJ_OFFSET))
    p.add_argument("--gproj-null", default=str(DEFAULT_GPROJ_NULL))
    p.add_argument("--gproj-base", default=str(DEFAULT_GPROJ_BASE))
    p.add_argument("--gproj-offset", default=str(DEFAULT_GPROJ_OFFSET))

    p.add_argument("--window", type=int, default=4096, help="Shots per snapshot/window.")
    p.add_argument("--trials-per-depth", type=int, default=500, help="Random compound-corruption trials per depth.")
    p.add_argument("--max-depth", type=int, default=len(FAULTS), help="Maximum corruption depth to test.")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--collapse-threshold", type=float, default=0.50, help="Survival below this is counted as collapsed.")
    p.add_argument("--preserve-threshold", type=float, default=0.75, help="Survival above this is counted as preserved/repaired.")
    p.add_argument("--save-trials", action="store_true", default=True, help="Save full per-trial rows.")
    p.add_argument("--exhaustive-combos", action="store_true", default=True, help="Run each fault combination once per depth instead of random trial sampling.")
    p.add_argument("--test-null-depths", action="store_true", default=True, help="Also run corruption depth curves for null records.")
    p.add_argument("--out-dir", default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.auto and not args.files:
        args.auto = True

    rng = np.random.default_rng(int(args.seed))

    tag = now_tag()
    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"dm_probe_24_corruption_boundary_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("  D_M PROBE 05B — DIMENSIONAL ERROR-CORRECTION / CORRUPTION BOUNDARY")
    print("=" * 100)
    print(f"  Out dir            : {out_dir}")
    print(f"  window             : {args.window} shots/snapshot")
    print(f"  trials/depth       : {args.trials_per_depth}")
    print(f"  max depth          : {args.max_depth}")
    print(f"  collapse threshold : {args.collapse_threshold}")
    print(f"  preserve threshold : {args.preserve_threshold}")
    print("  Rule               : single faults may repair; compound faults should reveal the collapse border")
    print("-" * 100)

    records = build_auto_records(args) if args.auto else build_manual_records(args)
    if not records:
        raise RuntimeError("No D_M records loaded.")

    # Precompute null manifolds by substrate for null-floor comparison.
    null_manifolds: Dict[str, Manifold] = {}
    for rec in records:
        if rec.condition == "null":
            null_manifolds[rec.substrate] = compute_manifold_from_pair(rec, int(args.window), rng=rng)

    observed_rows: List[Dict[str, Any]] = []
    allowed_rows: List[Dict[str, Any]] = []
    trial_rows: List[Dict[str, Any]] = []
    depth_summary_rows: List[Dict[str, Any]] = []

    for rec in records:
        nman = null_manifolds.get(rec.substrate)
        obs, allowed, trials, summaries = run_one_record(rec, nman, out_dir, args, rng)
        observed_rows.extend(obs)
        allowed_rows.extend(allowed)
        trial_rows.extend(trials)
        depth_summary_rows.extend(summaries)

    observed_fields = [
        "record", "substrate", "condition", "path", "tiles", "shots", "windows", "rungs",
        "best_orientation", "internal_score", "phase_score", "phase_mode", "mean_energy",
        "mean_specificity", "energy_track_r", "specificity_track_r", "phase_velocity_r",
        "phase_span_pi", "null_survival", "null_best_orientation",
    ]
    allowed_fields = [
        "record", "substrate", "condition", "transform", "survival", "candidate_orientation",
        "candidate_phase_score", "candidate_energy", "candidate_specificity",
        "component_channel_shape", "component_energy_shape", "component_phase_match",
        "component_delay_integrity", "component_specificity_shape", "component_specificity_retention",
        "component_energy_strength", "component_phase_self",
    ]
    summary_fields = [
        "record", "substrate", "condition", "depth", "collapse_threshold", "preserve_threshold",
        "null_survival", "n_trials", "mean_survival", "std_survival", "min_survival",
        "p10_survival", "median_survival", "p90_survival", "max_survival",
        "frac_below_collapse_threshold", "frac_above_preserve_threshold",
        "best_combo", "worst_combo",
    ]
    trial_fields = [
        "record", "substrate", "condition", "depth", "trial", "trial_seed", "fault_combo",
        "survival", "candidate_orientation", "candidate_phase_score", "candidate_phase_mode",
        "candidate_energy", "candidate_specificity", "channel_shape", "energy_shape",
        "phase_match", "delay_integrity", "specificity_shape", "specificity_retention",
        "energy_strength", "phase_self",
    ]

    write_csv(out_dir / "probe05b_observed_rows.csv", observed_rows, observed_fields)
    write_csv(out_dir / "probe05b_allowed_rows.csv", allowed_rows, allowed_fields)
    write_csv(out_dir / "probe05b_depth_summary.csv", depth_summary_rows, summary_fields)
    if args.save_trials:
        write_csv(out_dir / "probe05b_trial_rows.csv", trial_rows, trial_fields)

    write_json(out_dir / "probe_config.json", {
        "script": "d_m_probe05b_corruption_boundary.py",
        "timestamp": tag,
        "window": int(args.window),
        "trials_per_depth": int(args.trials_per_depth),
        "max_depth": int(args.max_depth),
        "seed": int(args.seed),
        "collapse_threshold": float(args.collapse_threshold),
        "preserve_threshold": float(args.preserve_threshold),
        "faults": FAULTS,
        "allowed_transforms": ALLOWED_TRANSFORMS,
        "orientations": [o[0] for o in ORIENTATIONS],
        "records": [{
            "label": r.label,
            "substrate": r.substrate,
            "condition": r.condition,
            "path": str(r.path),
            "tiles": int(r.pair.shape[0]),
            "shots": int(r.pair.shape[1]),
        } for r in records],
    })

    print("=" * 100)
    print("  D_M PROBE 05B COMPLETE")
    print("=" * 100)
    print(f"  Output: {out_dir}")
    print("  Files : probe05b_observed_rows.csv")
    print("          probe05b_allowed_rows.csv")
    print("          probe05b_depth_summary.csv")
    if args.save_trials:
        print("          probe05b_trial_rows.csv")
    print("          probe_config.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
