#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
D_M HOLOGRAPHIC PROJECTION PROBE
==============================================================================

Drop this file in:

    ghost_oracle/D_M/probes/d_m_holographic_projection_probe.py

Purpose
-------
This probe tests the framing:

    GPT-2 produces a free-running pre-softmax QK product.
    D_M qproj/gproj/geo bases define constrained projection boundaries.
    The same input text data is mapped into both product domains.
    We then ask whether the free GPT-2 product and D_M-constrained products
    are geometrically coherent.

Important
---------
This probe does NOT use attention probabilities.

It stops at:

    L_ij = Q_i · K_j / sqrt(d)

No softmax.
No causal masking for the reciprocal raw product.
No claim that D_M is a transformer head.
No claim of 1-to-1 reproduction.

Bounded claim being probed
--------------------------
D_M may produce a compatible pre-softmax bilinear-product geometry when input
data is projected through qproj/gproj/geo boundary domains.

The D_M bases act as the holographic aperture / boundary geometry.

Substrates
----------
qproj:
    Loaded from real QPU D_M base files.

gproj:
    Loaded from GPU D_M base files.

geo:
    Analytic D_M manifold generated from the same D_M condition geometry.

GPT-2:
    Free-running reference product extracted from real Q/K projections.

Default base paths
------------------
These match the current user-provided D_M files:

    qproj-null
    qproj-base
    qproj-offset
    gproj-null
    gproj-base
    gproj-offset

Outputs
-------
Creates:

    ghost_oracle/D_M/analysis/d_m_holographic_probe_<timestamp>/

with:

    d_m_boundary_manifolds.csv
    gpt2_free_manifolds.csv
    holographic_projection_scores.csv
    probe_config.json

Usage
-----
From repo root:

    python ghost_oracle/D_M/probes/d_m_holographic_projection_probe.py

With your own text file:

    python ghost_oracle/D_M/probes/d_m_holographic_projection_probe.py ^
      --text-file data/my_probe_text.txt

Quick CPU-safe smoke test without transformers:

    python ghost_oracle/D_M/probes/d_m_holographic_projection_probe.py --no-gpt2

Run fewer heads:

    python ghost_oracle/D_M/probes/d_m_holographic_projection_probe.py ^
      --layers 0,1,2 ^
      --heads 0,1,2,3

==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# PATHS
# =============================================================================

PROBE_DIR = Path(__file__).resolve().parent
D_M_DIR = PROBE_DIR.parent
DATA_DIR = D_M_DIR / "data"
ANALYSIS_DIR = PROBE_DIR / "analyze"


# =============================================================================
# DEFAULT D_M BASE FILES
# =============================================================================

DEFAULT_QPROJ_NULL = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fm4ihvjngc73aq3ccg.npz"
DEFAULT_QPROJ_BASE = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8flk2jo3njc73f0g560.npz"
DEFAULT_QPROJ_OFFSET = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fl82bo3njc73f0fgd0.npz"

DEFAULT_GPROJ_NULL = DATA_DIR / "dm_gpu_data_null_4096shots_seed9031229662612491082.npz"
DEFAULT_GPROJ_BASE = DATA_DIR / "dm_gpu_data_base_delay_4096shots_seed2669559634056472362.npz"
DEFAULT_GPROJ_OFFSET = DATA_DIR / "dm_gpu_data_offset_deformed_4096shots_seed6727069190982977623.npz"


# =============================================================================
# D_M CONSTANTS
# =============================================================================

WITNESS_LABELS = ["XY", "YZ", "ZY", "YX"]
WITNESS_TO_INDEX = {x: i for i, x in enumerate(WITNESS_LABELS)}

DEFAULT_BASE_DELAYS = [0, 2, 4, 8, 16]
DEFAULT_NULL_DELAYS = [0, 0, 0, 0, 0]
DEFAULT_OFFSET_DT = 128

EPS = 1e-12


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DMBoundary:
    substrate: str
    condition: str
    source: str
    n_rungs: int
    witnesses: np.ndarray
    # witnesses shape: (n_rungs, 4), columns XY,YZ,ZY,YX
    energy: np.ndarray
    phase: np.ndarray
    specificity: np.ndarray
    meta: Dict[str, Any]


@dataclass
class GPT2HeadManifold:
    model_name: str
    layer: int
    head: int
    bucket_mode: str
    center_mode: str
    n_rungs: int
    witnesses: np.ndarray
    # witnesses shape: (n_rungs, 4), columns XY,YZ,ZY,YX
    energy: np.ndarray
    phase: np.ndarray
    specificity: np.ndarray
    meta: Dict[str, Any]


@dataclass
class ProjectionScore:
    model_name: str
    layer: int
    head: int
    bucket_mode: str
    center_mode: str
    substrate: str
    condition: str
    boundary_source: str
    boundary_alignment: float
    free_retention: float
    projected_energy_mean: float
    projected_specificity_mean: float
    projected_phase_span_pi: float
    boundary_phase_span_pi: float
    phase_mae_pi: float
    coherence_score: float


# =============================================================================
# BASIC HELPERS
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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n <= 0:
        return 0.0
    x = x[:n]
    y = y[:n]
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < EPS or ny < EPS:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def phase_distance_pi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Circular phase distance on [0, pi), returned in pi units.
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    d = np.abs(aa - bb)
    d = np.minimum(d, math.pi - d)
    return d / math.pi


def manifold_from_witnesses(witnesses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert witness channels XY,YZ,ZY,YX into D_M-style manifold coordinates.

    YZ is primary.
    ZY is reciprocal/inverted through R = -ZY.
    XY/YX are comparison channels.

    Returns:
        energy, phase, specificity
    """
    w = np.asarray(witnesses, dtype=np.float64)
    xy = w[:, 0]
    yz = w[:, 1]
    zy = w[:, 2]
    yx = w[:, 3]

    reciprocal = -zy
    energy = np.sqrt(yz * yz + reciprocal * reciprocal)
    comparison_energy = np.sqrt(xy * xy + yx * yx)
    specificity = energy - comparison_energy
    phase = np.mod(np.arctan2(reciprocal, yz), math.pi)

    return energy, phase, specificity


def normalize_boundary(witnesses: np.ndarray) -> np.ndarray:
    """
    Create a unit aperture from a D_M boundary.

    This is intentionally not a fitted transform. It is an elementwise aperture:
    the data can express itself only through the signed boundary channels.

    Shape is preserved:
        (rungs, witnesses)
    """
    w = np.asarray(witnesses, dtype=np.float64)
    rms = float(np.sqrt(np.mean(w * w))) if w.size else 0.0
    if rms < EPS:
        return np.zeros_like(w)
    return w / rms


def holographic_project(data_witnesses: np.ndarray, boundary_witnesses: np.ndarray) -> np.ndarray:
    """
    Project data through the raw bounded D_M base.

    Important:
        The D_M base is already the bounded aperture.
        Do NOT normalize it.
        Do NOT rescale null into an active-strength basis.

    Projection:

        projected[rung, witness] =
            data_product[rung, witness] * boundary[rung, witness]

    The delay/rung structure belongs to the D_M boundary, not to an additional
    GPT-2-side bucket transform.
    """
    data = np.asarray(data_witnesses, dtype=np.float64)
    boundary = np.asarray(boundary_witnesses, dtype=np.float64)

    r = min(data.shape[0], boundary.shape[0])
    c = min(data.shape[1], boundary.shape[1])

    out = np.zeros((r, c), dtype=np.float64)
    out[:, :] = data[:r, :c] * boundary[:r, :c]
    return out


# =============================================================================
# D_M BASE LOADING
# =============================================================================

def infer_condition_from_metadata(base: np.ndarray, off: np.ndarray, total: np.ndarray) -> str:
    if base.size == 0 or off.size == 0 or total.size == 0:
        return "unknown"
    if int(np.max(base)) == 0 and int(np.max(off)) == 0 and int(np.max(total)) == 0:
        return "null"
    if int(np.max(base)) > 0 and int(np.max(off)) == 0:
        return "base_only"
    if int(np.max(base)) > 0 and int(np.max(off)) > 0:
        return "offset_on"
    return "unknown"


def compute_connected_from_pair(pair_tile: np.ndarray) -> float:
    """
    Compute connected two-bit correlator from pair[shot, 2].

    Bits are mapped to spins in {-1, +1}, then:

        C = <s0 s1> - <s0><s1>

    This matches the D_M projector idea of subtracting single-channel marginals.
    """
    p = np.asarray(pair_tile, dtype=np.uint8)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"Expected pair tile shape (shots,2), got {p.shape}")

    s0 = (2.0 * p[:, 0].astype(np.float64)) - 1.0
    s1 = (2.0 * p[:, 1].astype(np.float64)) - 1.0

    return float(np.mean(s0 * s1) - np.mean(s0) * np.mean(s1))


def load_dm_boundary(path: Path, substrate: str, condition_hint: Optional[str] = None) -> DMBoundary:
    if not path.exists():
        raise FileNotFoundError(f"Missing D_M base file: {path}")

    z = np.load(path, allow_pickle=True)

    if "pair" in z.files:
        pair = np.asarray(z["pair"], dtype=np.uint8)
    else:
        pair_keys = sorted(
            [k for k in z.files if k.startswith("pair_tile")],
            key=lambda k: int(k.replace("pair_tile", "")),
        )
        if not pair_keys:
            raise KeyError(f"{path} has no pair or pair_tile* arrays.")
        pair = np.stack([np.asarray(z[k], dtype=np.uint8) for k in pair_keys], axis=0)

    if pair.ndim != 3 or pair.shape[2] != 2:
        raise ValueError(f"{path} pair must have shape (tiles, shots, 2), got {pair.shape}")

    tiles = int(pair.shape[0])
    shots = int(pair.shape[1])

    if "tile_rung_index" in z.files:
        tile_rung = np.asarray(z["tile_rung_index"], dtype=np.int32)
    else:
        tile_rung = (np.arange(tiles) // 4).astype(np.int32)

    if "tile_witness_index" in z.files:
        tile_witness = np.asarray(z["tile_witness_index"], dtype=np.int32)
    elif "tile_witness_label" in z.files:
        labels = decode_str_array(z["tile_witness_label"])
        tile_witness = np.asarray(
            [WITNESS_TO_INDEX.get(x, i % 4) for i, x in enumerate(labels[:tiles])],
            dtype=np.int32,
        )
    else:
        tile_witness = (np.arange(tiles) % 4).astype(np.int32)

    def optional_int_array(name: str, default: int = 0) -> np.ndarray:
        if name in z.files:
            arr = np.asarray(z[name], dtype=np.int32)
            if arr.shape[0] == tiles:
                return arr
        return np.full((tiles,), default, dtype=np.int32)

    tile_base = optional_int_array("tile_base_delay_dt", 0)
    tile_off = optional_int_array("tile_offset_dt", 0)
    tile_total = optional_int_array("tile_total_delay_dt", 0)

    condition = condition_hint or infer_condition_from_metadata(tile_base, tile_off, tile_total)

    n_rungs = int(np.max(tile_rung)) + 1 if tiles else 0
    witnesses = np.zeros((n_rungs, 4), dtype=np.float64)
    counts = np.zeros((n_rungs, 4), dtype=np.int32)

    for t in range(tiles):
        r = int(tile_rung[t])
        wi = int(tile_witness[t])
        if 0 <= r < n_rungs and 0 <= wi < 4:
            witnesses[r, wi] += compute_connected_from_pair(pair[t])
            counts[r, wi] += 1

    for r in range(n_rungs):
        for wi in range(4):
            if counts[r, wi] > 0:
                witnesses[r, wi] /= float(counts[r, wi])

    energy, phase, specificity = manifold_from_witnesses(witnesses)

    meta = {
        "path": str(path),
        "tiles": tiles,
        "shots": shots,
        "condition": condition,
        "tile_base_delay_dt": tile_base.tolist(),
        "tile_offset_dt": tile_off.tolist(),
        "tile_total_delay_dt": tile_total.tolist(),
        "counts": counts.tolist(),
    }

    return DMBoundary(
        substrate=substrate,
        condition=condition,
        source=str(path),
        n_rungs=n_rungs,
        witnesses=witnesses,
        energy=energy,
        phase=phase,
        specificity=specificity,
        meta=meta,
    )


# =============================================================================
# GEO ANALYTIC BOUNDARY
# =============================================================================

def normalize_log_delay(values: Sequence[int]) -> np.ndarray:
    arr = np.log1p(np.maximum(0.0, np.asarray(values, dtype=np.float64)))
    if arr.size == 0:
        return arr
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if abs(mx - mn) < EPS:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def make_geo_boundary(condition: str) -> DMBoundary:
    """
    Lightweight analytic D_M boundary.

    This mirrors the role of the GEO path: no records, closed-form manifold.

    It is intentionally simple and conservative. The production benchmark's
    CUDA GEO kernel is still the ground-truth fast path. This probe-level GEO
    exists so the holographic comparison has qproj/gproj/geo columns.
    """
    if condition == "null":
        base_delays = list(DEFAULT_NULL_DELAYS)
        offset_dt = 0
    elif condition == "base_only":
        base_delays = list(DEFAULT_BASE_DELAYS)
        offset_dt = 0
    elif condition == "offset_on":
        base_delays = list(DEFAULT_BASE_DELAYS)
        offset_dt = DEFAULT_OFFSET_DT
    else:
        raise ValueError(f"unknown geo condition: {condition}")

    n = len(base_delays)
    tile_offsets = np.asarray([r * 4 * offset_dt for r in range(n)], dtype=np.float64)
    total_delay = np.asarray(base_delays, dtype=np.float64) + tile_offsets
    x = normalize_log_delay(total_delay)

    witnesses = np.zeros((n, 4), dtype=np.float64)

    if condition == "null":
        # Small residual comparison-free boundary.
        for r in range(n):
            witnesses[r, 0] = 0.002 * math.sin(r + 0.3)
            witnesses[r, 1] = 0.006 * math.cos(r + 0.1)
            witnesses[r, 2] = -0.006 * math.sin(r + 0.4)
            witnesses[r, 3] = 0.002 * math.cos(r + 0.7)
    else:
        for r in range(n):
            envelope = 0.04 + 0.28 * (0.20 + 0.80 * (float(x[r]) ** 1.10))
            phase = math.pi * (0.05 + 0.37 * float(x[r]))

            if condition == "offset_on":
                phase += 0.18 * math.sin(2.0 * math.pi * float(x[r]) + 0.35)
                envelope *= 0.90 + 0.18 * math.cos(2.0 * math.pi * float(x[r]) + 0.17)

            yz = envelope * math.cos(phase)
            ret = envelope * math.sin(phase)
            zy = -ret

            witnesses[r, 0] = 0.010 * math.sin(2.1 * r + 0.4)
            witnesses[r, 1] = yz
            witnesses[r, 2] = zy
            witnesses[r, 3] = 0.010 * math.cos(1.7 * r + 0.2)

    energy, phase, specificity = manifold_from_witnesses(witnesses)

    return DMBoundary(
        substrate="geo",
        condition=condition,
        source=f"analytic_geo_{condition}",
        n_rungs=n,
        witnesses=witnesses,
        energy=energy,
        phase=phase,
        specificity=specificity,
        meta={
            "condition": condition,
            "base_delays": base_delays,
            "offset_dt": offset_dt,
            "note": "Probe-level analytic GEO. Production GEO kernel remains canonical.",
        },
    )


# =============================================================================
# TEXT INPUT
# =============================================================================

DEFAULT_TEXTS = [
    "Ghost Oracle projects data through fixed operator boundaries instead of letting it run unconstrained.",
    "The pre-softmax product is the bilinear object before attention becomes a probability distribution.",
    "D_M uses a YZ primary channel and a ZY reciprocal channel across an ordered delay ladder.",
    "The purpose of this probe is compatibility, not one to one reproduction.",
    "A physical base can act like a holographic aperture for a computational signal.",
    "The same data can produce different products depending on the geometry it is projected through.",
]


def load_texts(args: argparse.Namespace) -> List[str]:
    if args.text_file:
        p = Path(args.text_file)
        if not p.exists():
            raise FileNotFoundError(f"text file not found: {p}")
        raw = p.read_text(encoding="utf-8", errors="replace")
        if args.split_lines:
            texts = [line.strip() for line in raw.splitlines() if line.strip()]
        else:
            chunks = [x.strip() for x in raw.split("\n\n") if x.strip()]
            texts = chunks if chunks else [raw.strip()]
    elif args.text:
        texts = [args.text]
    else:
        texts = list(DEFAULT_TEXTS)

    if args.max_texts and args.max_texts > 0:
        texts = texts[: int(args.max_texts)]

    return texts


# =============================================================================
# OFFSET BUCKETS
# =============================================================================

def build_offset_buckets(seq_len: int, n_rungs: int, mode: str) -> List[List[int]]:
    """
    Build positive token-offset buckets.

    Offset delta means:
        i = query position
        j = key/reference position
        delta = i - j > 0

    GPT-2 path uses raw QK products for both L_ij and L_ji.
    """
    max_delta = max(1, seq_len - 1)
    offsets = list(range(1, max_delta + 1))

    if n_rungs <= 1:
        return [offsets]

    if mode == "log":
        # Mirrors the D_M ladder idea: local, short, mid, long, far.
        raw = [
            [1],
            [d for d in offsets if 2 <= d <= 3],
            [d for d in offsets if 4 <= d <= 7],
            [d for d in offsets if 8 <= d <= 15],
            [d for d in offsets if d >= 16],
        ]
        if n_rungs == 5:
            return [b for b in raw]
        # Generic fallback if a future base uses different rung count.
        return quantile_offset_buckets(seq_len, n_rungs)

    if mode == "linear":
        buckets: List[List[int]] = []
        edges = np.linspace(1, max_delta + 1, n_rungs + 1)
        for r in range(n_rungs):
            lo = int(math.floor(edges[r]))
            hi = int(math.floor(edges[r + 1]))
            b = [d for d in offsets if lo <= d < hi]
            buckets.append(b)
        return buckets

    if mode == "balanced":
        return quantile_offset_buckets(seq_len, n_rungs)

    raise ValueError(f"unknown bucket mode: {mode}")


def quantile_offset_buckets(seq_len: int, n_rungs: int) -> List[List[int]]:
    max_delta = max(1, seq_len - 1)
    offsets = list(range(1, max_delta + 1))

    # Pair count at delta is roughly seq_len - delta.
    expanded: List[int] = []
    for d in offsets:
        expanded.extend([d] * max(1, seq_len - d))

    cuts = np.quantile(expanded, np.linspace(0.0, 1.0, n_rungs + 1))
    buckets: List[List[int]] = []

    for r in range(n_rungs):
        lo = int(math.floor(cuts[r]))
        hi = int(math.floor(cuts[r + 1]))
        if r == n_rungs - 1:
            b = [d for d in offsets if lo <= d <= hi]
        else:
            b = [d for d in offsets if lo <= d < hi]
        if not b:
            b = [min(max_delta, max(1, lo))]
        buckets.append(sorted(set(b)))

    return buckets


# =============================================================================
# GPT-2 PRE-SOFTMAX QK EXTRACTION
# =============================================================================

def parse_int_list(s: Optional[str], max_value: Optional[int] = None) -> Optional[List[int]]:
    if s is None or str(s).strip() == "":
        return None
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if max_value is not None and not (0 <= value < max_value):
            raise ValueError(f"value {value} outside allowed range 0..{max_value - 1}")
        out.append(value)
    return sorted(set(out))


def center_logits(logits: np.ndarray, mode: str) -> np.ndarray:
    """
    Center raw QK logits without softmax.

    none:
        use raw logits.

    row:
        subtract row mean, analogous to removing query-position baseline.

    global:
        subtract global mean for the head.

    zscore:
        subtract global mean and divide by global std.
    """
    x = np.asarray(logits, dtype=np.float64)

    if mode == "none":
        return x

    if mode == "row":
        return x - np.mean(x, axis=-1, keepdims=True)

    if mode == "global":
        return x - float(np.mean(x))

    if mode == "zscore":
        mu = float(np.mean(x))
        sd = float(np.std(x))
        if sd < EPS:
            return x - mu
        return (x - mu) / sd

    raise ValueError(f"unknown center mode: {mode}")

def compute_base_native_witnesses_from_logits(
    logits: np.ndarray,
    center_mode: str,
    n_rungs: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert raw GPT-2 QK products into D_M witness channels without adding an
    extra token-delay ladder.

    The data produces one global pre-softmax product field.

    The D_M base supplies:
        - rung structure
        - delay structure
        - physical/operator boundary

    Witness mapping:
        YZ = forward raw product L_ij
        ZY = reciprocal raw product L_ji
        XY/YX = deterministic mismatched comparison channels

    No softmax.
    No GPT-2-side delay buckets.
    No normalization of the D_M base.
    """
    raw = np.asarray(logits, dtype=np.float64)
    if raw.ndim != 3:
        raise ValueError(f"logits must have shape (batch,seq,seq), got {raw.shape}")

    centered = np.stack(
        [center_logits(raw[b], center_mode) for b in range(raw.shape[0])],
        axis=0,
    )

    batch, seq_len, _ = centered.shape

    xy_vals: List[float] = []
    yz_vals: List[float] = []
    zy_vals: List[float] = []
    yx_vals: List[float] = []

    for b in range(batch):
        L = centered[b]

        for i in range(1, seq_len):
            for j in range(0, i):
                # Primary forward / reciprocal raw QK products.
                yz_vals.append(float(L[i, j]))
                zy_vals.append(float(L[j, i]))

                # Deterministic mismatched comparison channels.
                j_cmp = (j + 1) % seq_len
                if j_cmp == i:
                    j_cmp = (j + 2) % seq_len

                i_cmp = (i - 1) % seq_len
                if i_cmp == j:
                    i_cmp = (i - 2) % seq_len

                xy_vals.append(float(L[i, j_cmp]))
                yx_vals.append(float(L[j, i_cmp]))

    if not yz_vals:
        witnesses = np.zeros((n_rungs, 4), dtype=np.float64)
        counts = np.zeros((n_rungs,), dtype=np.int64)
    else:
        base_vector = np.asarray(
            [
                float(np.mean(xy_vals)),
                float(np.mean(yz_vals)),
                float(np.mean(zy_vals)),
                float(np.mean(yx_vals)),
            ],
            dtype=np.float64,
        )

        # Same data product is presented to every rung.
        # The base decides how that product expresses across delayed rungs.
        witnesses = np.tile(base_vector.reshape(1, 4), (n_rungs, 1))
        counts = np.full((n_rungs,), len(yz_vals), dtype=np.int64)

    meta = {
        "bucket_mode": "base",
        "center_mode": center_mode,
        "counts": counts.tolist(),
        "note": (
            "Base-native raw pre-softmax QK product. "
            "No GPT-2-side delay buckets. D_M base supplies rung/delay geometry."
        ),
    }

    return witnesses, meta

def compute_witnesses_from_logits(
    logits: np.ndarray,
    bucket_mode: str,
    center_mode: str,
    n_rungs: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert raw QK logits into a D_M-compatible witness manifold.

    logits shape:
        (batch, seq, seq)

    Witness mapping:
        YZ = forward raw product L_ij
        ZY = reciprocal raw product L_ji
        XY = forward comparison product using a neighboring/mismatched key
        YX = reciprocal comparison product using a neighboring/mismatched key

    No softmax is used.

    The comparison channels are deliberately simple local controls. They are not
    the claim; they exist so D_M specificity has a non-YZ/ZY reference.
    """
    if bucket_mode in ("base", "base_native", "no_delay"):
      return compute_base_native_witnesses_from_logits(
            logits=logits,
            center_mode=center_mode,
            n_rungs=n_rungs,
        )

    raw = np.asarray(logits, dtype=np.float64)
    if raw.ndim != 3:
        raise ValueError(f"logits must have shape (batch,seq,seq), got {raw.shape}")

    centered = np.stack([center_logits(raw[b], center_mode) for b in range(raw.shape[0])], axis=0)

    batch, seq_len, _ = centered.shape
    buckets = build_offset_buckets(seq_len, n_rungs, bucket_mode)

    witnesses = np.zeros((n_rungs, 4), dtype=np.float64)
    counts = np.zeros((n_rungs,), dtype=np.int64)

    for r, deltas in enumerate(buckets):
        xy_vals: List[float] = []
        yz_vals: List[float] = []
        zy_vals: List[float] = []
        yx_vals: List[float] = []

        for b in range(batch):
            L = centered[b]
            for delta in deltas:
                if delta <= 0 or delta >= seq_len:
                    continue

                for i in range(delta, seq_len):
                    j = i - delta

                    # Primary forward / reciprocal raw QK products.
                    yz_vals.append(float(L[i, j]))
                    zy_vals.append(float(L[j, i]))

                    # Comparison channels: same query rows, nearby mismatched keys.
                    # These are control channels, not the main claim.
                    j_cmp = max(0, j - 1)
                    i_cmp = min(seq_len - 1, i + 1)

                    xy_vals.append(float(L[i, j_cmp]))
                    yx_vals.append(float(L[j, i_cmp]))

        if yz_vals:
            witnesses[r, 0] = float(np.mean(xy_vals))
            witnesses[r, 1] = float(np.mean(yz_vals))
            witnesses[r, 2] = float(np.mean(zy_vals))
            witnesses[r, 3] = float(np.mean(yx_vals))
            counts[r] = len(yz_vals)

    meta = {
        "bucket_mode": bucket_mode,
        "center_mode": center_mode,
        "buckets": buckets,
        "counts": counts.tolist(),
        "note": "Raw pre-softmax QK product manifold. No softmax.",
    }

    return witnesses, meta


def load_gpt2_qk_manifolds(
    texts: Sequence[str],
    model_name: str,
    layers: Optional[List[int]],
    heads: Optional[List[int]],
    bucket_modes: Sequence[str],
    center_modes: Sequence[str],
    n_rungs: int,
    max_length: int,
    device: str,
    gpt2_batch_size: int = 16,
) -> List[GPT2HeadManifold]:
    """
    Extract raw GPT-2 QK products head-by-head in small text batches.

    This avoids freezing or huge memory spikes from processing hundreds of
    sequences in one GPT-2 forward pass.

    Still no softmax.

    Product extracted:

        L_ij = Q_i · K_j / sqrt(d)

    The reciprocal product L_ji is computed from the same raw QK matrix.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "GPT-2 extraction requires torch and transformers. "
            "Install them or rerun with --no-gpt2."
        ) from e

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gpt2_batch_size = max(1, int(gpt2_batch_size))

    print(f"  loading tokenizer/model: {model_name}")
    print(f"  device               : {device}")
    print(f"  gpt2 batch size      : {gpt2_batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    n_layer = len(model.transformer.h)
    n_head = int(model.config.n_head)
    n_embd = int(model.config.n_embd)
    head_dim = n_embd // n_head

    layer_ids = list(range(n_layer)) if layers is None else [x for x in layers if 0 <= x < n_layer]
    head_ids = list(range(n_head)) if heads is None else [x for x in heads if 0 <= x < n_head]

    # Accumulators:
    # key -> {"sum": witnesses_sum[rung,4], "count": counts[rung]}
    accum: Dict[Tuple[int, int, str, str], Dict[str, np.ndarray]] = {}

    for layer in layer_ids:
        for head in head_ids:
            for bucket_mode in bucket_modes:
                for center_mode in center_modes:
                    accum[(layer, head, bucket_mode, center_mode)] = {
                        "sum": np.zeros((n_rungs, 4), dtype=np.float64),
                        "count": np.zeros((n_rungs,), dtype=np.float64),
                    }

    total = len(texts)
    n_batches = int(math.ceil(total / gpt2_batch_size))

    for batch_index in range(n_batches):
        lo = batch_index * gpt2_batch_size
        hi = min(total, lo + gpt2_batch_size)
        batch_texts = list(texts[lo:hi])

        print(f"  GPT-2 batch {batch_index + 1}/{n_batches}: texts {lo}:{hi}")

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            out = model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = out.hidden_states

        for layer in layer_ids:
            block = model.transformer.h[layer]
            x = hidden_states[layer]

            with torch.no_grad():
                qkv = block.attn.c_attn(x)
                q, k, _v = qkv.split(n_embd, dim=2)

                batch, seq_len, _ = q.shape

                q = q.view(batch, seq_len, n_head, head_dim).permute(0, 2, 1, 3)
                k = k.view(batch, seq_len, n_head, head_dim).permute(0, 2, 1, 3)

                # Raw pre-softmax QK product.
                logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(head_dim))
                logits_np = logits.detach().float().cpu().numpy()

            for head in head_ids:
                head_logits = logits_np[:, head, :, :]

                for bucket_mode in bucket_modes:
                    for center_mode in center_modes:
                        witnesses, meta = compute_witnesses_from_logits(
                            logits=head_logits,
                            bucket_mode=bucket_mode,
                            center_mode=center_mode,
                            n_rungs=n_rungs,
                        )

                        counts = np.asarray(meta.get("counts", [0] * n_rungs), dtype=np.float64)
                        key = (layer, head, bucket_mode, center_mode)

                        for r in range(n_rungs):
                            c = float(counts[r])
                            if c > 0:
                                accum[key]["sum"][r, :] += witnesses[r, :] * c
                                accum[key]["count"][r] += c

        # Keep memory clean between batches.
        del out
        del hidden_states
        del input_ids
        del attention_mask

        if device == "cuda":
            torch.cuda.empty_cache()

    manifolds: List[GPT2HeadManifold] = []

    for layer in layer_ids:
        for head in head_ids:
            for bucket_mode in bucket_modes:
                for center_mode in center_modes:
                    key = (layer, head, bucket_mode, center_mode)
                    sums = accum[key]["sum"]
                    counts = accum[key]["count"]

                    witnesses = np.zeros((n_rungs, 4), dtype=np.float64)
                    for r in range(n_rungs):
                        if counts[r] > 0:
                            witnesses[r, :] = sums[r, :] / counts[r]

                    energy, phase, specificity = manifold_from_witnesses(witnesses)

                    manifolds.append(
                        GPT2HeadManifold(
                            model_name=model_name,
                            layer=int(layer),
                            head=int(head),
                            bucket_mode=str(bucket_mode),
                            center_mode=str(center_mode),
                            n_rungs=int(n_rungs),
                            witnesses=witnesses,
                            energy=energy,
                            phase=phase,
                            specificity=specificity,
                            meta={
                                "bucket_mode": bucket_mode,
                                "center_mode": center_mode,
                                "counts": counts.tolist(),
                                "note": "Batched raw pre-softmax QK product manifold. No softmax.",
                            },
                        )
                    )

    return manifolds


# =============================================================================
# FALLBACK DATA MANIFOLD
# =============================================================================

def deterministic_text_embedding_manifolds(
    texts: Sequence[str],
    bucket_modes: Sequence[str],
    center_modes: Sequence[str],
    n_rungs: int,
    max_length: int,
) -> List[GPT2HeadManifold]:
    """
    Fallback path for --no-gpt2.

    This does not claim to be GPT-2. It creates deterministic token-hash product
    matrices so the rest of the D_M projection path can be smoke-tested without
    torch/transformers.
    """
    import hashlib

    tokens: List[List[str]] = []
    for text in texts:
        parts = text.strip().split()
        if not parts:
            parts = ["empty"]
        tokens.append(parts[:max_length])

    seq_len = min(max_length, max(len(x) for x in tokens))
    batch = len(tokens)

    x = np.zeros((batch, seq_len, 64), dtype=np.float64)

    for b, words in enumerate(tokens):
        for i in range(seq_len):
            word = words[i % len(words)]
            h = hashlib.sha256(f"{b}:{i}:{word}".encode("utf-8")).digest()
            vals = np.frombuffer(h + h, dtype=np.uint8).astype(np.float64)[:64]
            vals = (vals - 127.5) / 127.5
            x[b, i, :] = vals

    q = x
    k = np.roll(x, shift=1, axis=2)
    logits = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(float(x.shape[-1]))

    manifolds: List[GPT2HeadManifold] = []
    for bucket_mode in bucket_modes:
        for center_mode in center_modes:
            witnesses, meta = compute_witnesses_from_logits(
                logits=logits,
                bucket_mode=bucket_mode,
                center_mode=center_mode,
                n_rungs=n_rungs,
            )
            energy, phase, specificity = manifold_from_witnesses(witnesses)
            manifolds.append(
                GPT2HeadManifold(
                    model_name="deterministic_text_hash_fallback",
                    layer=0,
                    head=0,
                    bucket_mode=bucket_mode,
                    center_mode=center_mode,
                    n_rungs=n_rungs,
                    witnesses=witnesses,
                    energy=energy,
                    phase=phase,
                    specificity=specificity,
                    meta=meta,
                )
            )

    return manifolds


# =============================================================================
# SCORING
# =============================================================================

def score_projection(head: GPT2HeadManifold, boundary: DMBoundary) -> ProjectionScore:
    projected = holographic_project(head.witnesses, boundary.witnesses)
    p_energy, p_phase, p_spec = manifold_from_witnesses(projected)

    r = min(head.witnesses.shape[0], boundary.witnesses.shape[0])
    free_w = head.witnesses[:r]
    bound_w = boundary.witnesses[:r]
    proj_w = projected[:r]

    boundary_alignment = cosine(proj_w, bound_w)
    free_retention = cosine(proj_w, free_w)

    phase_mae = float(np.mean(phase_distance_pi(p_phase[:r], boundary.phase[:r]))) if r > 0 else 1.0

    projected_energy_mean = float(np.mean(p_energy[:r])) if r > 0 else 0.0
    projected_spec_mean = float(np.mean(p_spec[:r])) if r > 0 else 0.0
    projected_phase_span_pi = float((np.max(p_phase[:r]) - np.min(p_phase[:r])) / math.pi) if r > 0 else 0.0
    boundary_phase_span_pi = float((np.max(boundary.phase[:r]) - np.min(boundary.phase[:r])) / math.pi) if r > 0 else 0.0

    # Transparent heuristic score for sorting only.
    # Raw metric columns are the real analysis.
    ba01 = 0.5 * (boundary_alignment + 1.0)
    fr01 = 0.5 * (free_retention + 1.0)
    phase01 = max(0.0, 1.0 - phase_mae)
    spec01 = 1.0 if projected_spec_mean > 0 else 0.0

    coherence_score = float(
        0.40 * ba01 +
        0.30 * fr01 +
        0.20 * phase01 +
        0.10 * spec01
    )

    return ProjectionScore(
        model_name=head.model_name,
        layer=head.layer,
        head=head.head,
        bucket_mode=head.bucket_mode,
        center_mode=head.center_mode,
        substrate=boundary.substrate,
        condition=boundary.condition,
        boundary_source=boundary.source,
        boundary_alignment=float(boundary_alignment),
        free_retention=float(free_retention),
        projected_energy_mean=projected_energy_mean,
        projected_specificity_mean=projected_spec_mean,
        projected_phase_span_pi=projected_phase_span_pi,
        boundary_phase_span_pi=boundary_phase_span_pi,
        phase_mae_pi=phase_mae,
        coherence_score=coherence_score,
    )


# =============================================================================
# ROW BUILDERS
# =============================================================================

def boundary_rows(boundaries: Sequence[DMBoundary]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for b in boundaries:
        for r in range(b.n_rungs):
            rows.append({
                "substrate": b.substrate,
                "condition": b.condition,
                "source": b.source,
                "rung": r,
                "XY": b.witnesses[r, 0],
                "YZ": b.witnesses[r, 1],
                "ZY": b.witnesses[r, 2],
                "YX": b.witnesses[r, 3],
                "energy": b.energy[r],
                "phase": b.phase[r],
                "phase_pi": b.phase[r] / math.pi,
                "specificity": b.specificity[r],
            })
    return rows


def gpt2_rows(heads: Sequence[GPT2HeadManifold]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for h in heads:
        for r in range(h.n_rungs):
            rows.append({
                "model_name": h.model_name,
                "layer": h.layer,
                "head": h.head,
                "bucket_mode": h.bucket_mode,
                "center_mode": h.center_mode,
                "rung": r,
                "XY": h.witnesses[r, 0],
                "YZ": h.witnesses[r, 1],
                "ZY": h.witnesses[r, 2],
                "YX": h.witnesses[r, 3],
                "energy": h.energy[r],
                "phase": h.phase[r],
                "phase_pi": h.phase[r] / math.pi,
                "specificity": h.specificity[r],
                "bucket_offsets": json.dumps(h.meta.get("buckets", [])),
                "bucket_counts": json.dumps(h.meta.get("counts", [])),
            })
    return rows


def score_rows(scores: Sequence[ProjectionScore]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in scores:
        rows.append({
            "model_name": s.model_name,
            "layer": s.layer,
            "head": s.head,
            "bucket_mode": s.bucket_mode,
            "center_mode": s.center_mode,
            "substrate": s.substrate,
            "condition": s.condition,
            "boundary_source": s.boundary_source,
            "boundary_alignment": s.boundary_alignment,
            "free_retention": s.free_retention,
            "projected_energy_mean": s.projected_energy_mean,
            "projected_specificity_mean": s.projected_specificity_mean,
            "projected_phase_span_pi": s.projected_phase_span_pi,
            "boundary_phase_span_pi": s.boundary_phase_span_pi,
            "phase_mae_pi": s.phase_mae_pi,
            "coherence_score": s.coherence_score,
        })
    return rows


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="D_M holographic projection probe: GPT-2 pre-softmax product through qproj/gproj/geo boundaries."
    )

    p.add_argument("--qproj-null", type=Path, default=DEFAULT_QPROJ_NULL)
    p.add_argument("--qproj-base", type=Path, default=DEFAULT_QPROJ_BASE)
    p.add_argument("--qproj-offset", type=Path, default=DEFAULT_QPROJ_OFFSET)

    p.add_argument("--gproj-null", type=Path, default=DEFAULT_GPROJ_NULL)
    p.add_argument("--gproj-base", type=Path, default=DEFAULT_GPROJ_BASE)
    p.add_argument("--gproj-offset", type=Path, default=DEFAULT_GPROJ_OFFSET)

    p.add_argument("--no-geo", action="store_true", help="Disable probe-level analytic GEO boundaries.")
    p.add_argument("--no-gpt2", action="store_true", help="Use deterministic text-hash fallback instead of GPT-2.")

    p.add_argument("--model", default="gpt2", help="Hugging Face GPT-2 style model name.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-texts", type=int, default=0)

    p.add_argument("--text", default="", help="Single text input.")
    p.add_argument("--text-file", default="", help="Text file input.")
    p.add_argument("--split-lines", action="store_true", help="Use each non-empty line as one sequence.")

    p.add_argument("--layers", default="", help="Comma-separated layer ids. Default: all.")
    p.add_argument("--heads", default="", help="Comma-separated head ids. Default: all.")

    p.add_argument(
        "--bucket-modes",
        default="base",
        help="Comma-separated product mapping modes: base,log,linear,balanced.",
    )
    p.add_argument(
        "--center-modes",
        default="row,global,zscore",
        help="Comma-separated centering modes: none,row,global,zscore.",
    )
    p.add_argument(
        "--gpt2-batch-size",
        type=int,
        default=16,
        help="Number of text sequences per GPT-2 forward pass.",
    )

    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--top", type=int, default=20, help="Rows to print from sorted scores.")

    return p


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = build_arg_parser().parse_args()

    tag = now_tag()
    out_dir = args.out_dir or (ANALYSIS_DIR / f"dm_probe_14_holographic_probe_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_modes = [x.strip() for x in args.bucket_modes.split(",") if x.strip()]
    center_modes = [x.strip() for x in args.center_modes.split(",") if x.strip()]

    texts = load_texts(args)

    print()
    print("=" * 112)
    print("  D_M HOLOGRAPHIC PROJECTION PROBE")
    print("=" * 112)
    print(f"  D_M dir       : {D_M_DIR}")
    print(f"  Output dir    : {out_dir}")
    print(f"  Text sequences: {len(texts)}")
    print(f"  GPT-2 path    : {'disabled/fallback' if args.no_gpt2 else args.model}")
    print(f"  Bucket modes  : {bucket_modes}")
    print(f"  Center modes  : {center_modes}")
    print("  Rule          : pre-softmax QK product only; no softmax")
    print()

    # -------------------------------------------------------------------------
    # Load D_M boundaries.
    # -------------------------------------------------------------------------
    boundaries: List[DMBoundary] = []

    base_specs = [
        ("qproj", "null", args.qproj_null),
        ("qproj", "base_only", args.qproj_base),
        ("qproj", "offset_on", args.qproj_offset),
        ("gproj", "null", args.gproj_null),
        ("gproj", "base_only", args.gproj_base),
        ("gproj", "offset_on", args.gproj_offset),
    ]

    print("[LOAD] D_M qproj/gproj boundaries")
    for substrate, condition, path in base_specs:
        b = load_dm_boundary(path=path, substrate=substrate, condition_hint=condition)
        boundaries.append(b)
        print(
            f"  {substrate:5s} {condition:9s} "
            f"rungs={b.n_rungs} "
            f"E_mean={np.mean(b.energy):+.6f} "
            f"S_mean={np.mean(b.specificity):+.6f} "
            f"{path}"
        )

    if not args.no_geo:
        print("[LOAD] D_M geo analytic boundaries")
        for condition in ["null", "base_only", "offset_on"]:
            b = make_geo_boundary(condition)
            boundaries.append(b)
            print(
                f"  {'geo':5s} {condition:9s} "
                f"rungs={b.n_rungs} "
                f"E_mean={np.mean(b.energy):+.6f} "
                f"S_mean={np.mean(b.specificity):+.6f} "
                f"{b.source}"
            )

    n_rungs = min(b.n_rungs for b in boundaries)
    if n_rungs <= 0:
        raise RuntimeError("No valid D_M rungs found.")

    # -------------------------------------------------------------------------
    # Build GPT-2/free product manifolds.
    # -------------------------------------------------------------------------
    layer_ids = parse_int_list(args.layers)
    head_ids = parse_int_list(args.heads)

    print()
    print("[BUILD] Free product manifolds")
    t0 = time.time()

    if args.no_gpt2:
        head_manifolds = deterministic_text_embedding_manifolds(
            texts=texts,
            bucket_modes=bucket_modes,
            center_modes=center_modes,
            n_rungs=n_rungs,
            max_length=args.max_length,
        )
    else:
        head_manifolds = load_gpt2_qk_manifolds(
            texts=texts,
            model_name=args.model,
            layers=layer_ids,
            heads=head_ids,
            bucket_modes=bucket_modes,
            center_modes=center_modes,
            n_rungs=n_rungs,
            max_length=args.max_length,
            device=args.device,
            gpt2_batch_size=args.gpt2_batch_size,
        )

    dt = time.time() - t0
    print(f"  built manifolds: {len(head_manifolds)} in {dt:.2f}s")

    # -------------------------------------------------------------------------
    # Score holographic projections.
    # -------------------------------------------------------------------------
    print()
    print("[SCORE] Project free product through D_M boundaries")
    scores: List[ProjectionScore] = []

    for h in head_manifolds:
        for b in boundaries:
            scores.append(score_projection(h, b))

    scores_sorted = sorted(scores, key=lambda s: s.coherence_score, reverse=True)

    # -------------------------------------------------------------------------
    # Write outputs.
    # -------------------------------------------------------------------------
    boundary_csv = out_dir / "d_m_boundary_manifolds.csv"
    gpt2_csv = out_dir / "gpt2_free_manifolds.csv"
    scores_csv = out_dir / "holographic_projection_scores.csv"
    config_json = out_dir / "probe_config.json"

    write_csv(
        boundary_csv,
        boundary_rows(boundaries),
        fields=[
            "substrate", "condition", "source", "rung",
            "XY", "YZ", "ZY", "YX",
            "energy", "phase", "phase_pi", "specificity",
        ],
    )

    write_csv(
        gpt2_csv,
        gpt2_rows(head_manifolds),
        fields=[
            "model_name", "layer", "head", "bucket_mode", "center_mode", "rung",
            "XY", "YZ", "ZY", "YX",
            "energy", "phase", "phase_pi", "specificity",
            "bucket_offsets", "bucket_counts",
        ],
    )

    write_csv(
        scores_csv,
        score_rows(scores_sorted),
        fields=[
            "model_name", "layer", "head", "bucket_mode", "center_mode",
            "substrate", "condition", "boundary_source",
            "boundary_alignment", "free_retention",
            "projected_energy_mean", "projected_specificity_mean",
            "projected_phase_span_pi", "boundary_phase_span_pi",
            "phase_mae_pi", "coherence_score",
        ],
    )

    write_json(
        config_json,
        {
            "created": tag,
            "probe": "d_m_holographic_projection_probe",
            "rule": "pre-softmax QK product only; no softmax",
            "out_dir": out_dir,
            "texts_count": len(texts),
            "model": "fallback" if args.no_gpt2 else args.model,
            "bucket_modes": bucket_modes,
            "center_modes": center_modes,
            "n_rungs": n_rungs,
            "base_paths": {
                "qproj_null": args.qproj_null,
                "qproj_base": args.qproj_base,
                "qproj_offset": args.qproj_offset,
                "gproj_null": args.gproj_null,
                "gproj_base": args.gproj_base,
                "gproj_offset": args.gproj_offset,
            },
            "outputs": {
                "boundary_csv": boundary_csv,
                "gpt2_csv": gpt2_csv,
                "scores_csv": scores_csv,
            },
        },
    )

    # -------------------------------------------------------------------------
    # Print top rows.
    # -------------------------------------------------------------------------
    print()
    print("=" * 112)
    print("  TOP HOLOGRAPHIC PROJECTION SCORES")
    print("=" * 112)
    print(
        "  layer head bucket   center   substrate condition   "
        "score   align   retain  phaseMAE  projE    projS"
    )
    print("  " + "-" * 108)

    for s in scores_sorted[: max(1, int(args.top))]:
        print(
            f"  {s.layer:5d} {s.head:4d} "
            f"{s.bucket_mode:8s} {s.center_mode:8s} "
            f"{s.substrate:8s} {s.condition:9s} "
            f"{s.coherence_score:6.3f} "
            f"{s.boundary_alignment:+7.3f} "
            f"{s.free_retention:+7.3f} "
            f"{s.phase_mae_pi:8.3f} "
            f"{s.projected_energy_mean:+8.5f} "
            f"{s.projected_specificity_mean:+8.5f}"
        )

    print()
    print("[DONE]")
    print(f"  boundary manifolds : {boundary_csv}")
    print(f"  free manifolds     : {gpt2_csv}")
    print(f"  projection scores  : {scores_csv}")
    print(f"  config             : {config_json}")
    print()


if __name__ == "__main__":
    main()