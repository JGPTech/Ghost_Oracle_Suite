#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GHOST ORACLE SUITE — D_M PROBE 06: FREQUENCY BASE-10 SEED PROJECTOR
================================================================================

Purpose
-------
This is the corrected D_M Probe 06.

D_M is tested as ONE projector:

    dm_frequency_base10_seed_projector

No D_M channels.
No D_M scheduler.
No D_M family-best.
No D_M variant search.
No D_M random-matrix generator.

Core operator idea
------------------
D_M is a dimensional projector built from:

    1. frozen QPU D_M scene
    2. base-10 dimensional state logic
    3. frequency / phase stepping
    4. calibrated seed folded into the dimensional state
    5. one projection run

The seed is not used to seed a random matrix.
The seed becomes an active per-sample dimensional channel.

Operator sketch
---------------
Given base data X:

    QPU scene -> global dimensional calibration
    X         -> base frequency calibration
    QPU + X   -> calibrated base-10 alpha ladder
    X         -> per-sample base-10 seed states
    X + seed  -> frequency/base-10 fold
    output    -> Z

Controls
--------
Controls are included only for comparison:

    pca_projection
    random_gaussian_single
    random_gaussian_mean/best/worst_N
    random_orthogonal_mean/best/worst_N
    random_sparse_achlioptas_mean/best/worst_N
    random_shuffle_crop
    random_sign_flip_crop

Important
---------
random_*_best_N is an oracle/search ceiling. It gets N attempts and keeps the
winner. It is not the fair single-shot baseline.

The fair comparison is:

    dm_frequency_base10_seed_projector
        vs
    random_*_mean_N
    random_gaussian_single
    pca_projection

Usage
-----
    python ghost_oracle/D_M/probes/dm_probe_06_frequency_base10_seed_projector.py ^
      --qpu-base ghost_oracle/D_M/data/dm_job_d8fb033o3njc73f01170.npz ^
      --random-trials 32

================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import secrets
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# =============================================================================
# PATHS / CONFIG
# =============================================================================

HERE = Path(__file__).resolve().parent
DM_DIR = HERE.parent
DATA_DIR = DM_DIR / "data"
ANALYSIS_DIR = HERE / "analyze"

DEFAULT_N = 5000
DEFAULT_INPUT_DIM = 64
DEFAULT_OUTPUT_DIM = 4
DEFAULT_K = 10
DEFAULT_RANDOM_TRIALS = 128

DATASETS = [
    "blobs",
    "rings",
    "swiss_roll_like",
    "s_curve_like",
    "sparse_binary",
]


# =============================================================================
# BASIC HELPERS
# =============================================================================

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_master_seed(seed_arg: int | None) -> int:
    return int(seed_arg) if seed_arg is not None else secrets.randbits(128)


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
    return x


def load_npz_scalar(x: Any) -> Any:
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    return x


def standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - np.mean(x, axis=0, keepdims=True)) / np.maximum(
        np.std(x, axis=0, keepdims=True),
        eps,
    )


def normalize_prob(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    s = float(np.sum(x))
    if s <= 1e-15:
        return np.ones_like(x, dtype=np.float64) / max(1, x.size)
    return x / s


def state_to_bit_weight(state_prob: np.ndarray) -> np.ndarray:
    p = normalize_prob(state_prob)
    out = np.zeros(4, dtype=np.float64)
    for state in range(16):
        for bit in range(4):
            if (state >> bit) & 1:
                out[bit] += p[state]
    return out


# =============================================================================
# METRICS
# =============================================================================

def pairwise_distances(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sq = np.sum(x * x, axis=1, keepdims=True)
    d = sq + sq.T - 2.0 * (x @ x.T)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)
    return d


def knn_indices(x: np.ndarray, k: int) -> np.ndarray:
    d = pairwise_distances(x)
    return np.argpartition(d, kth=k, axis=1)[:, :k]


def neighbor_overlap(ref_nn: np.ndarray, test_nn: np.ndarray) -> float:
    k = ref_nn.shape[1]
    return float(np.mean([
        len(set(ref_nn[i]).intersection(set(test_nn[i]))) / float(k)
        for i in range(ref_nn.shape[0])
    ]))


def label_purity(labels: np.ndarray, test_nn: np.ndarray) -> float:
    labels = np.asarray(labels)
    return float(np.mean([
        np.mean(labels[test_nn[i]] == labels[i])
        for i in range(test_nn.shape[0])
    ]))


def trustworthiness_lite(x_ref: np.ndarray, x_test: np.ndarray, k: int) -> float:
    n = x_ref.shape[0]
    d_ref = pairwise_distances(x_ref)
    d_test = pairwise_distances(x_test)

    ref_order = np.argsort(d_ref, axis=1)
    test_nn = np.argpartition(d_test, kth=k, axis=1)[:, :k]

    ranks = np.empty((n, n), dtype=np.int32)
    for i in range(n):
        ranks[i, ref_order[i]] = np.arange(1, n + 1)

    penalty = 0.0
    for i in range(n):
        ref_set = set(ref_order[i, :k])
        for j in test_nn[i]:
            if int(j) not in ref_set:
                penalty += ranks[i, j] - k

    denom = n * k * (2 * n - 3 * k - 1)
    if denom <= 0:
        return float("nan")

    return float(max(0.0, min(1.0, 1.0 - (2.0 / denom) * penalty)))


def spearman_distance_rank_sample(
    x_ref: np.ndarray,
    x_test: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int = 20000,
) -> float:
    n = x_ref.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)

    mask = i != j
    i = i[mask]
    j = j[mask]

    if i.size < 10:
        return float("nan")

    dr = np.sum((x_ref[i] - x_ref[j]) ** 2, axis=1)
    dt = np.sum((x_test[i] - x_test[j]) ** 2, axis=1)

    rr = np.empty_like(np.argsort(dr), dtype=np.float64)
    rt = np.empty_like(np.argsort(dt), dtype=np.float64)

    rr[np.argsort(dr)] = np.arange(dr.size, dtype=np.float64)
    rt[np.argsort(dt)] = np.arange(dt.size, dtype=np.float64)

    rr = (rr - np.mean(rr)) / max(np.std(rr), 1e-12)
    rt = (rt - np.mean(rt)) / max(np.std(rt), 1e-12)

    return float(np.mean(rr * rt))


def ranking_score(overlap: float, trust: float, purity: float, rankcorr: float) -> float:
    return (
        0.45 * overlap
        + 0.25 * trust
        + 0.20 * purity
        + 0.10 * max(-1.0, min(1.0, rankcorr))
    )


# =============================================================================
# SYNTHETIC DATASETS
# =============================================================================

def embed_low_to_high(
    z: np.ndarray,
    input_dim: int,
    rng: np.random.Generator,
    noise: float = 0.03,
) -> np.ndarray:
    z = standardize(z)
    proj = rng.normal(0.0, 1.0, size=(z.shape[1], input_dim))
    x = z @ proj
    x += noise * rng.normal(0.0, 1.0, size=x.shape)

    if input_dim >= 8:
        nonlinear = []
        for i in range(min(z.shape[1], 4)):
            nonlinear.append(np.sin(z[:, i:i + 1]))
            nonlinear.append(np.cos(z[:, i:i + 1]))
        nl = np.concatenate(nonlinear, axis=1)
        width = min(nl.shape[1], input_dim)
        x[:, :width] += 0.35 * nl[:, :width]

    return standardize(x)


def make_blobs(n: int, input_dim: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.array([
        [-2.5, -2.0],
        [-2.0, 2.0],
        [2.0, -2.0],
        [2.5, 2.0],
        [0.0, 0.0],
    ], dtype=np.float64)

    labels = rng.integers(0, len(centers), size=n)
    z = centers[labels] + 0.45 * rng.normal(size=(n, 2))
    return embed_low_to_high(z, input_dim, rng, noise=0.04), labels


def make_rings(n: int, input_dim: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, 3, size=n)
    radii = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    r = radii[labels] + 0.08 * rng.normal(size=n)
    z = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    return embed_low_to_high(z, input_dim, rng, noise=0.04), labels


def make_swiss_roll_like(n: int, input_dim: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n)
    h = rng.uniform(-1.0, 1.0, size=n)

    z = np.stack([
        t * np.cos(t),
        h * 6.0,
        t * np.sin(t),
    ], axis=1)

    labels = np.digitize(t, np.quantile(t, [0.2, 0.4, 0.6, 0.8]))
    return embed_low_to_high(z, input_dim, rng, noise=0.035), labels.astype(np.int32)


def make_s_curve_like(n: int, input_dim: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    t = rng.uniform(-1.5 * np.pi, 1.5 * np.pi, size=n)
    y = rng.uniform(-1.0, 1.0, size=n)

    z = np.stack([
        np.sin(t),
        y,
        np.sign(t) * (np.cos(t) - 1.0),
    ], axis=1)

    labels = np.digitize(t, np.quantile(t, [0.2, 0.4, 0.6, 0.8]))
    return embed_low_to_high(z, input_dim, rng, noise=0.035), labels.astype(np.int32)


def make_sparse_binary(n: int, input_dim: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n_groups = 6
    labels = rng.integers(0, n_groups, size=n)

    prototypes = np.zeros((n_groups, input_dim), dtype=np.float64)
    block = max(2, input_dim // n_groups)

    for g in range(n_groups):
        start = g * block
        end = min(input_dim, start + block)
        prototypes[g, start:end] = 1.0
        prototypes[g, g % input_dim] = 1.0
        prototypes[g, (g * 7 + 3) % input_dim] = 1.0

    x = prototypes[labels].copy()
    flip = rng.random(size=x.shape) < 0.06
    x = np.where(flip, 1.0 - x, x)
    x += 0.03 * rng.normal(size=x.shape)
    return standardize(x), labels


def make_dataset(name: str, n: int, input_dim: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if name == "blobs":
        return make_blobs(n, input_dim, rng)
    if name == "rings":
        return make_rings(n, input_dim, rng)
    if name == "swiss_roll_like":
        return make_swiss_roll_like(n, input_dim, rng)
    if name == "s_curve_like":
        return make_s_curve_like(n, input_dim, rng)
    if name == "sparse_binary":
        return make_sparse_binary(n, input_dim, rng)
    raise ValueError(f"unknown dataset: {name}")


# =============================================================================
# QPU D_M CALIBRATION
# =============================================================================

def find_latest_qpu_base() -> Path:
    ptr = DATA_DIR / "latest_dm_qpu_data.json"
    if ptr.exists():
        with open(ptr, "r", encoding="utf-8") as f:
            meta = json.load(f)
        p = Path(meta["path"])
        if p.exists():
            return p

    matches = sorted(DATA_DIR.glob("dm_job_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError("No D_M QPU base found. Pass --qpu-base explicitly.")
    return matches[0]


def get_tile_arrays(npz: Any, tile: int) -> Dict[str, np.ndarray]:
    def get(name: str) -> np.ndarray:
        key = f"{name}_tile{tile}"
        if key in npz.files:
            return np.asarray(npz[key])
        if name in npz.files:
            return np.asarray(npz[name][tile])
        raise KeyError(f"missing {name} for tile {tile}")

    out: Dict[str, np.ndarray] = {}
    has_ctrl = f"ctrl_tile{tile}" in npz.files or "ctrl" in npz.files
    if has_ctrl:
        out["ctrl"] = get("ctrl").astype(np.uint8)
        out["dim"] = get("dim").astype(np.uint8)
        out["edge"] = get("edge").astype(np.uint8)
        out["aux"] = get("aux").astype(np.uint8)
        out["meta"] = get("meta").astype(np.uint8)
    else:
        pair_key = f"pair_tile{tile}"
        pair = np.asarray(npz[pair_key] if pair_key in npz.files else npz["pair"][tile]).astype(np.uint8)
        q0, q1 = pair[:, 0], pair[:, 1]
        p00 = ((q0 == 0) & (q1 == 0)).astype(np.uint8)
        p01 = ((q0 == 0) & (q1 == 1)).astype(np.uint8)
        p10 = ((q0 == 1) & (q1 == 0)).astype(np.uint8)
        p11 = ((q0 == 1) & (q1 == 1)).astype(np.uint8)
        out["ctrl"] = q0
        out["dim"] = np.column_stack([p00, p01, p10, p11])
        out["edge"] = np.bitwise_xor(q0, q1)
        out["aux"] = q1
        out["meta"] = q0

    if f"dim_state_tile{tile}" in npz.files:
        out["dim_state"] = np.asarray(npz[f"dim_state_tile{tile}"]).astype(np.uint8)
    elif "dim_state" in npz.files:
        out["dim_state"] = np.asarray(npz["dim_state"][tile]).astype(np.uint8)
    else:
        weights = np.asarray([1, 2, 4, 8], dtype=np.uint8)
        out["dim_state"] = (out["dim"] * weights[None, :]).sum(axis=1).astype(np.uint8)

    out["popcount"] = out["dim"].sum(axis=1).astype(np.uint8)
    out["parity"] = np.bitwise_xor.reduce(out["dim"], axis=1).astype(np.uint8)
    out["boundary"] = np.bitwise_xor(out["dim"][:, 0], out["dim"][:, 3]).astype(np.uint8)
    out["interior"] = np.bitwise_xor(out["dim"][:, 1], out["dim"][:, 2]).astype(np.uint8)

    return out


@dataclass
class QPUCalibration:
    qpu_base_path: str
    job_id: str
    backend: str
    num_tiles: int
    shots: int

    global_state_prob: List[float]
    global_bit_weight: List[float]
    ctrl_mean: float
    edge_mean: float
    aux_mean: float
    meta_mean: float
    popcount_mean: float
    parity_mean: float
    boundary_mean: float
    interior_mean: float

    alpha_ladder: List[float]
    alpha_state_profile: List[float]
    alpha_dominant_profile: List[float]
    frequency_profile: List[float]


def base10_state_probs(alpha: float, num_qubits: int = 4) -> np.ndarray:
    """
    Base-10 dimensional state selector.

    This preserves the core old logic:
        sin(10 * 2^i * pi * x + pi/4)
        hard threshold
        CNOT-like bit chain
        dominant state with small phase skirt
    """
    x = float(alpha) / 10.0

    bits = []
    for i in range(num_qubits):
        s = math.sin(10.0 * (2 ** i) * math.pi * x + math.pi / 4.0)
        bits.append(1 if s >= 0 else 0)

    folded = bits[:]
    for i in range(num_qubits - 1):
        folded[i + 1] = folded[i + 1] ^ folded[i]

    dominant = 0
    for i, b in enumerate(folded):
        dominant |= int(b) << i

    n_states = 2 ** num_qubits
    probs = np.zeros(n_states, dtype=np.float64)
    probs[dominant] = 1.0

    for state in range(n_states):
        if state == dominant:
            continue

        hamming = 0
        for i in range(num_qubits):
            hamming += ((state >> i) & 1) ^ ((dominant >> i) & 1)

        skirt_phase = 0.5 + 0.5 * math.cos((math.pi / 10.0) * (state + 1) * (alpha + 1.0))
        probs[state] = 0.025 * skirt_phase / (1.0 + hamming)

    return normalize_prob(probs)


def build_qpu_calibration(qpu_base_path: Path) -> QPUCalibration:
    npz = np.load(qpu_base_path, allow_pickle=True)

    job_id = str(load_npz_scalar(npz["job_id"])) if "job_id" in npz.files else qpu_base_path.stem
    backend = str(load_npz_scalar(npz["backend"])) if "backend" in npz.files else "unknown"
    num_tiles = int(load_npz_scalar(npz["num_tiles"])) if "num_tiles" in npz.files else int(np.asarray(npz["dim"]).shape[0])
    shots = int(load_npz_scalar(npz["shots"])) if "shots" in npz.files else -1

    tile_state_probs = []
    ctrl_vals = []
    edge_vals = []
    aux_vals = []
    meta_vals = []
    pop_vals = []
    parity_vals = []
    boundary_vals = []
    interior_vals = []

    for tile in range(num_tiles):
        arr = get_tile_arrays(npz, tile)

        hist = np.bincount(arr["dim_state"].astype(np.int64), minlength=16).astype(np.float64)
        tile_state_probs.append(normalize_prob(hist))

        ctrl_vals.append(float(np.mean(arr["ctrl"])))
        edge_vals.append(float(np.mean(arr["edge"])))
        aux_vals.append(float(np.mean(arr["aux"])))
        meta_vals.append(float(np.mean(arr["meta"])))
        pop_vals.append(float(np.mean(arr["popcount"])))
        parity_vals.append(float(np.mean(arr["parity"])))
        boundary_vals.append(float(np.mean(arr["boundary"])))
        interior_vals.append(float(np.mean(arr["interior"])))

    global_state_prob = normalize_prob(np.mean(np.stack(tile_state_probs, axis=0), axis=0))
    global_bit_weight = state_to_bit_weight(global_state_prob)

    ctrl_mean = float(np.mean(ctrl_vals))
    edge_mean = float(np.mean(edge_vals))
    aux_mean = float(np.mean(aux_vals))
    meta_mean = float(np.mean(meta_vals))
    popcount_mean = float(np.mean(pop_vals))
    parity_mean = float(np.mean(parity_vals))
    boundary_mean = float(np.mean(boundary_vals))
    interior_mean = float(np.mean(interior_vals))

    base_alpha = np.linspace(0.0, 10.0, 10, endpoint=False, dtype=np.float64) + (np.pi / 40.0)

    # QPU-calibrated base-10 alpha ladder.
    state10 = normalize_prob(global_state_prob[:10])
    qpu_drive = np.array([
        ctrl_mean,
        edge_mean,
        aux_mean,
        meta_mean,
        popcount_mean / 4.0,
        parity_mean,
        boundary_mean,
        interior_mean,
        global_bit_weight[0],
        global_bit_weight[3],
    ], dtype=np.float64)

    qpu_drive = qpu_drive - float(np.mean(qpu_drive))
    qpu_drive = qpu_drive / max(float(np.max(np.abs(qpu_drive))), 1e-12)

    state_drive = state10 - float(np.mean(state10))
    state_drive = state_drive / max(float(np.max(np.abs(state_drive))), 1e-12)

    alpha_shift = 0.60 * state_drive + 0.40 * qpu_drive
    alpha_ladder = np.mod(base_alpha + 0.35 * alpha_shift, 10.0)

    alpha_probs = []
    dominant_hist = np.zeros(16, dtype=np.float64)

    for alpha in alpha_ladder:
        probs = base10_state_probs(float(alpha))
        alpha_probs.append(probs)
        dominant_hist[int(np.argmax(probs))] += 1.0

    alpha_state_profile = normalize_prob(np.mean(np.stack(alpha_probs, axis=0), axis=0))
    alpha_dominant_profile = normalize_prob(dominant_hist)

    # Frequency profile: QPU state distribution turned into frequency weights.
    # This is not random. This is the calibrated dimensional frequency scaffold.
    raw_freq = (
        0.45 * global_state_prob
        + 0.35 * alpha_state_profile
        + 0.20 * alpha_dominant_profile
    )
    frequency_profile = normalize_prob(raw_freq)

    return QPUCalibration(
        qpu_base_path=str(qpu_base_path),
        job_id=job_id,
        backend=backend,
        num_tiles=num_tiles,
        shots=shots,

        global_state_prob=global_state_prob.tolist(),
        global_bit_weight=global_bit_weight.tolist(),
        ctrl_mean=ctrl_mean,
        edge_mean=edge_mean,
        aux_mean=aux_mean,
        meta_mean=meta_mean,
        popcount_mean=popcount_mean,
        parity_mean=parity_mean,
        boundary_mean=boundary_mean,
        interior_mean=interior_mean,

        alpha_ladder=alpha_ladder.tolist(),
        alpha_state_profile=alpha_state_profile.tolist(),
        alpha_dominant_profile=alpha_dominant_profile.tolist(),
        frequency_profile=frequency_profile.tolist(),
    )


# =============================================================================
# ONE D_M PROJECTOR
# =============================================================================

def build_seed_channel(
    x: np.ndarray,
    calibration: QPUCalibration,
    output_dim: int,
) -> np.ndarray:
    """
    Build active per-sample seed channel.

    This is the key correction:
        seed is not RNG for W
        seed is a dimensional channel folded through frequency/base-10 logic

    Output:
        seed_channel shape = (n_samples, output_dim)
    """
    x = standardize(x)
    n, d = x.shape

    alpha_ladder = np.asarray(calibration.alpha_ladder, dtype=np.float64)
    freq_profile = normalize_prob(np.asarray(calibration.frequency_profile, dtype=np.float64))
    state_profile = normalize_prob(np.asarray(calibration.alpha_state_profile, dtype=np.float64))
    dominant_profile = normalize_prob(np.asarray(calibration.alpha_dominant_profile, dtype=np.float64))
    bit_weight = np.asarray(calibration.global_bit_weight, dtype=np.float64)

    # Base-data calibration run.
    sample_energy = np.mean(x * x, axis=1)
    sample_mean = np.mean(x, axis=1)
    sample_abs = np.mean(np.abs(x), axis=1)

    # Normalize to [0, 1]-ish phase coordinates.
    e = (sample_energy - np.min(sample_energy)) / max(float(np.ptp(sample_energy)), 1e-12)
    m = (sample_mean - np.min(sample_mean)) / max(float(np.ptp(sample_mean)), 1e-12)
    a = (sample_abs - np.min(sample_abs)) / max(float(np.ptp(sample_abs)), 1e-12)

    seed = np.zeros((n, output_dim), dtype=np.float64)

    for i in range(n):
        # Per-sample phase coordinate.
        phase_x = 0.55 * e[i] + 0.25 * a[i] + 0.20 * m[i]

        # Base-10 state accumulation.
        state_accum = np.zeros(16, dtype=np.float64)

        for j, alpha in enumerate(alpha_ladder):
            # Frequency-stepped alpha.
            freq = 1.0 + 10.0 * freq_profile[j % 16]
            alpha_eff = np.mod(alpha + freq * phase_x, 10.0)
            probs = base10_state_probs(float(alpha_eff))
            state_accum += probs

        state_accum = normalize_prob(state_accum)

        # Fold state distribution into output seed dimensions.
        for out_j in range(output_dim):
            bit_j = out_j % 4
            val = 0.0

            for state in range(16):
                bit = (state >> bit_j) & 1
                sign = 1.0 if bit else -1.0

                val += sign * state_accum[state]
                val += 0.35 * sign * state_profile[state]
                val += 0.25 * sign * dominant_profile[state]

            seed[i, out_j] = val * (0.5 + bit_weight[bit_j])

    return standardize(seed)


def dm_frequency_base10_seed_projector(
    x: np.ndarray,
    output_dim: int,
    calibration: QPUCalibration,
) -> np.ndarray:
    """
    The only D_M method in this probe.

    One calibrated dimensional/frequency/base-10 seed projector.

    Steps:
        1. standardize base data
        2. perform base-data frequency calibration
        3. build active per-sample seed channel
        4. fold original dimensions through frequency/base-10 routing
        5. inject seed channel as active projected dimension
        6. emit one Z
    """
    x = standardize(x)
    n, d = x.shape

    freq_profile = normalize_prob(np.asarray(calibration.frequency_profile, dtype=np.float64))
    state_profile = normalize_prob(np.asarray(calibration.alpha_state_profile, dtype=np.float64))
    dominant_profile = normalize_prob(np.asarray(calibration.alpha_dominant_profile, dtype=np.float64))
    qpu_state = normalize_prob(np.asarray(calibration.global_state_prob, dtype=np.float64))
    bit_weight = np.asarray(calibration.global_bit_weight, dtype=np.float64)

    seed_channel = build_seed_channel(x, calibration, output_dim)

    # Base-data frequency calibration.
    feature_energy = np.mean(x * x, axis=0)
    feature_abs = np.mean(np.abs(x), axis=0)
    feature_mean = np.mean(x, axis=0)

    feature_energy = feature_energy / max(float(np.mean(feature_energy)), 1e-12)
    feature_abs = feature_abs / max(float(np.mean(feature_abs)), 1e-12)

    # Single deterministic fold. No random matrix.
    z = np.zeros((n, output_dim), dtype=np.float64)

    for dim_i in range(d):
        # Dimensional state index from calibrated frequency and base-10 scaffold.
        state_i = (
            dim_i
            + int(round(10.0 * feature_energy[dim_i]))
            + int(round(10.0 * feature_abs[dim_i]))
            + int(np.argmax(qpu_state))
            + int(np.argmax(dominant_profile))
        ) % 16

        # Output route determined by base-10 state bits and feature frequency.
        route = (
            state_i
            + int(round(100.0 * freq_profile[state_i]))
            + int(round(10.0 * abs(feature_mean[dim_i])))
        ) % output_dim

        # Frequency/base-10 amplitude.
        amp = (
            0.35 * feature_energy[dim_i]
            + 0.25 * feature_abs[dim_i]
            + 0.18 * qpu_state[state_i] / max(float(np.mean(qpu_state)), 1e-12)
            + 0.14 * state_profile[state_i] / max(float(np.mean(state_profile)), 1e-12)
            + 0.08 * dominant_profile[state_i] / max(float(np.mean(dominant_profile)), 1e-12)
        )

        # Sign from dimensional bit parity.
        parity = bin(state_i).count("1") % 2
        sign = 1.0 if parity == 0 else -1.0

        z[:, route] += sign * amp * x[:, dim_i]

        # Controlled leakage to nearby output dimensions, frequency shaped.
        for out_j in range(output_dim):
            if out_j == route:
                continue

            bit_j = out_j % 4
            leak_phase = math.sin(
                2.0 * math.pi * (dim_i + 1) * (out_j + 1) * (1.0 + freq_profile[state_i]) / max(1, d)
            )
            leak = 0.035 * bit_weight[bit_j] * leak_phase
            z[:, out_j] += leak * x[:, dim_i]

    # Seed is folded in as active dimensional channel.
    # This is not afterthought noise; this is the calibrated seed dimension.
    seed_strength = (
        0.50
        + 0.25 * calibration.popcount_mean / 4.0
        + 0.15 * calibration.parity_mean
        + 0.10 * calibration.boundary_mean
    )

    z = z + seed_strength * seed_channel

    # Base-10 final phase fold.
    if output_dim >= 2:
        phase_strength = float(np.sum(freq_profile * state_profile))
        z[:, 0] += 0.05 * phase_strength * np.sin(seed_channel[:, 1])
        z[:, 1] += 0.05 * phase_strength * np.cos(seed_channel[:, 0])

    return standardize(z)


# =============================================================================
# CONTROL METHODS
# =============================================================================

def pca_project(x: np.ndarray, output_dim: int) -> np.ndarray:
    x0 = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    return standardize(x0 @ vt[:output_dim].T)


def random_gaussian_project(x: np.ndarray, output_dim: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.normal(0.0, 1.0, size=(x.shape[1], output_dim))
    w = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)
    return standardize(x @ w)


def random_orthogonal_project(x: np.ndarray, output_dim: int, rng: np.random.Generator) -> np.ndarray:
    d = x.shape[1]
    a = rng.normal(0.0, 1.0, size=(d, d))
    q, _ = np.linalg.qr(a)
    return standardize(x @ q[:, :output_dim])


def random_sparse_achlioptas_project(x: np.ndarray, output_dim: int, rng: np.random.Generator) -> np.ndarray:
    d = x.shape[1]
    u = rng.random(size=(d, output_dim))

    w = np.zeros((d, output_dim), dtype=np.float64)
    s3 = math.sqrt(3.0)

    w[u < (1.0 / 6.0)] = s3
    w[u > (5.0 / 6.0)] = -s3
    w = w / math.sqrt(max(1, output_dim))

    return standardize(x @ w)


def random_shuffle_crop(x: np.ndarray, output_dim: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.permutation(x.shape[1])[:output_dim]
    return standardize(x[:, idx])


def random_sign_flip_crop(x: np.ndarray, output_dim: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.permutation(x.shape[1])[:output_dim]
    signs = rng.choice([-1.0, 1.0], size=(output_dim,))
    return standardize(x[:, idx] * signs[None, :])


# =============================================================================
# EVAL STRUCTURES
# =============================================================================

@dataclass
class EvalRow:
    dataset: str
    method: str
    output_dim: int
    input_dim: int
    compression_ratio: float

    neighbor_overlap_at_k: float
    trustworthiness_lite: float
    label_purity_at_k: float
    spearman_distance_rank_sample: float
    ranking_score: float

    n: int
    k: int
    trials: int
    stat: str


@dataclass
class TrialRow:
    dataset: str
    family: str
    trial: int
    neighbor_overlap_at_k: float
    trustworthiness_lite: float
    label_purity_at_k: float
    spearman_distance_rank_sample: float
    ranking_score: float


def eval_embedding(
    dataset: str,
    method: str,
    x_ref: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    ref_nn: np.ndarray,
    k: int,
    rng: np.random.Generator,
    trials: int = 0,
    stat: str = "",
) -> EvalRow:
    test_nn = knn_indices(z, k)

    overlap = neighbor_overlap(ref_nn, test_nn)
    trust = trustworthiness_lite(x_ref, z, k)
    purity = label_purity(labels, test_nn)
    rankcorr = spearman_distance_rank_sample(x_ref, z, rng)
    score = ranking_score(overlap, trust, purity, rankcorr)

    return EvalRow(
        dataset=dataset,
        method=method,
        output_dim=int(z.shape[1]),
        input_dim=int(x_ref.shape[1]),
        compression_ratio=float(x_ref.shape[1] / max(1, z.shape[1])),

        neighbor_overlap_at_k=overlap,
        trustworthiness_lite=trust,
        label_purity_at_k=purity,
        spearman_distance_rank_sample=rankcorr,
        ranking_score=score,

        n=int(x_ref.shape[0]),
        k=int(k),
        trials=int(trials),
        stat=str(stat),
    )


def aggregate_random_rows(dataset: str, family: str, rows: List[EvalRow], stat: str) -> EvalRow:
    if stat == "mean":
        return EvalRow(
            dataset=dataset,
            method=f"{family}_mean_{len(rows)}",
            output_dim=rows[0].output_dim,
            input_dim=rows[0].input_dim,
            compression_ratio=rows[0].compression_ratio,
            neighbor_overlap_at_k=float(np.mean([r.neighbor_overlap_at_k for r in rows])),
            trustworthiness_lite=float(np.mean([r.trustworthiness_lite for r in rows])),
            label_purity_at_k=float(np.mean([r.label_purity_at_k for r in rows])),
            spearman_distance_rank_sample=float(np.mean([r.spearman_distance_rank_sample for r in rows])),
            ranking_score=float(np.mean([r.ranking_score for r in rows])),
            n=rows[0].n,
            k=rows[0].k,
            trials=len(rows),
            stat="mean",
        )

    ordered = sorted(rows, key=lambda r: r.ranking_score, reverse=True)

    if stat == "best":
        r = ordered[0]
        label = "best"
    elif stat == "worst":
        r = ordered[-1]
        label = "worst"
    else:
        raise ValueError(stat)

    return EvalRow(
        dataset=dataset,
        method=f"{family}_{label}_{len(rows)}",
        output_dim=r.output_dim,
        input_dim=r.input_dim,
        compression_ratio=r.compression_ratio,
        neighbor_overlap_at_k=r.neighbor_overlap_at_k,
        trustworthiness_lite=r.trustworthiness_lite,
        label_purity_at_k=r.label_purity_at_k,
        spearman_distance_rank_sample=r.spearman_distance_rank_sample,
        ranking_score=r.ranking_score,
        n=r.n,
        k=r.k,
        trials=len(rows),
        stat=label,
    )


def run_random_family(
    dataset: str,
    family: str,
    fn: Any,
    x: np.ndarray,
    labels: np.ndarray,
    ref_nn: np.ndarray,
    output_dim: int,
    k: int,
    rng: np.random.Generator,
    trials: int,
) -> Tuple[List[EvalRow], List[TrialRow]]:
    trial_eval_rows: List[EvalRow] = []
    trial_rows: List[TrialRow] = []

    for t in range(trials):
        z = fn(x, output_dim, rng)

        r = eval_embedding(
            dataset=dataset,
            method=f"{family}_trial_{t:03d}",
            x_ref=x,
            z=z,
            labels=labels,
            ref_nn=ref_nn,
            k=k,
            rng=rng,
            trials=trials,
            stat="trial",
        )

        trial_eval_rows.append(r)
        trial_rows.append(TrialRow(
            dataset=dataset,
            family=family,
            trial=t,
            neighbor_overlap_at_k=r.neighbor_overlap_at_k,
            trustworthiness_lite=r.trustworthiness_lite,
            label_purity_at_k=r.label_purity_at_k,
            spearman_distance_rank_sample=r.spearman_distance_rank_sample,
            ranking_score=r.ranking_score,
        ))

    return [
        aggregate_random_rows(dataset, family, trial_eval_rows, "mean"),
        aggregate_random_rows(dataset, family, trial_eval_rows, "best"),
        aggregate_random_rows(dataset, family, trial_eval_rows, "worst"),
    ], trial_rows


# =============================================================================
# OUTPUT
# =============================================================================

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def summarize_results(rows: List[EvalRow]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[EvalRow]] = {}

    for r in rows:
        if r.method == "identity_reference":
            continue
        if "_trial_" in r.method:
            continue
        by_method.setdefault(r.method, []).append(r)

    summary: List[Dict[str, Any]] = []

    for method, items in sorted(by_method.items()):
        summary.append({
            "method": method,
            "n_datasets": len(items),
            "neighbor_overlap_at_k_mean": float(np.mean([r.neighbor_overlap_at_k for r in items])),
            "trustworthiness_lite_mean": float(np.mean([r.trustworthiness_lite for r in items])),
            "label_purity_at_k_mean": float(np.mean([r.label_purity_at_k for r in items])),
            "spearman_distance_rank_sample_mean": float(np.mean([r.spearman_distance_rank_sample for r in items])),
            "ranking_score_mean": float(np.mean([r.ranking_score for r in items])),
            "compression_ratio_mean": float(np.mean([r.compression_ratio for r in items])),
        })

    summary.sort(
        key=lambda r: (
            r["neighbor_overlap_at_k_mean"],
            r["trustworthiness_lite_mean"],
            r["label_purity_at_k_mean"],
            r["ranking_score_mean"],
        ),
        reverse=True,
    )
    return summary


def write_report(
    path: Path,
    config: Dict[str, Any],
    rows: List[EvalRow],
    summary: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []

    lines.append("# D_M Probe 06 — Frequency Base-10 Seed Projector")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- One D_M projector only.")
    lines.append("- No D_M channel comparisons.")
    lines.append("- No D_M scheduler.")
    lines.append("- No D_M variant or family-best result.")
    lines.append("- Random best-of-N appears only as a ceiling/control.")
    lines.append("")
    lines.append("## D_M Method")
    lines.append("")
    lines.append("`dm_frequency_base10_seed_projector`")
    lines.append("")
    lines.append("## Operator")
    lines.append("")
    lines.append("Frozen QPU scene + frequency/base-10 calibration + active per-sample seed dimension → one projected representation.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- QPU base: `{config['qpu_base_path']}`")
    lines.append(f"- Job ID: `{config['calibration']['job_id']}`")
    lines.append(f"- Backend: `{config['calibration']['backend']}`")
    lines.append(f"- Random trials: `{config['random_trials']}`")
    lines.append(f"- Master seed: `{config['master_seed']}`")
    lines.append("")
    lines.append("## Overall Ranking")
    lines.append("")
    lines.append("| Rank | Method | Neighbor overlap@k | Trust-lite | Label purity@k | Distance-rank corr | Score |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")

    for i, r in enumerate(summary, start=1):
        lines.append(
            f"| {i} | `{r['method']}` | "
            f"{r['neighbor_overlap_at_k_mean']:.4f} | "
            f"{r['trustworthiness_lite_mean']:.4f} | "
            f"{r['label_purity_at_k_mean']:.4f} | "
            f"{r['spearman_distance_rank_sample_mean']:.4f} | "
            f"{r['ranking_score_mean']:.4f} |"
        )

    lines.append("")
    lines.append("## Dataset Results")
    lines.append("")
    lines.append("| Dataset | Method | Neighbor overlap@k | Trust-lite | Label purity@k | Distance-rank corr | Score |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for r in rows:
        if "_trial_" in r.method:
            continue

        lines.append(
            f"| `{r.dataset}` | `{r.method}` | "
            f"{r.neighbor_overlap_at_k:.4f} | "
            f"{r.trustworthiness_lite:.4f} | "
            f"{r.label_purity_at_k:.4f} | "
            f"{r.spearman_distance_rank_sample:.4f} | "
            f"{r.ranking_score:.4f} |"
        )

    lines.append("")
    lines.append("## Interpretation Rule")
    lines.append("")
    lines.append(
        "The only D_M row is `dm_frequency_base10_seed_projector`. "
        "`random_*_best_N` is not the fair baseline; it is a search ceiling. "
        "The fair controls are random single, random mean rows, and PCA."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_make_plots(out_dir: Path, summary: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable; skipping plots: {e}")
        return

    methods = [r["method"] for r in summary]
    overlap = [r["neighbor_overlap_at_k_mean"] for r in summary]

    plt.figure(figsize=(12, max(5, 0.35 * len(methods))))
    y = np.arange(len(methods))
    plt.barh(y, overlap)
    plt.yticks(y, methods, fontsize=8)
    plt.xlabel("Mean neighbor overlap@k")
    plt.title("D_M Probe 06 — Frequency Base-10 Seed Projector")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "dm_probe_06_overall_ranking.png", dpi=160)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — D_M Probe 06 frequency/base10 seed projector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qpu-base", default=None, help="Frozen dm_job_<JOB_ID>.npz. Defaults to latest.")
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--input-dim", type=int, default=DEFAULT_INPUT_DIM)
    p.add_argument("--output-dim", type=int, default=DEFAULT_OUTPUT_DIM)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--random-trials", type=int, default=DEFAULT_RANDOM_TRIALS)
    p.add_argument("--seed", type=int, default=None, help="Optional deterministic master seed.")
    p.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to probes/analysis/dm_probe_06_frequency_base10_seed_projector_<timestamp>/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    master_seed = make_master_seed(args.seed)
    seed_source = "user" if args.seed is not None else "secrets.randbits(128)"
    rng = np.random.default_rng(master_seed)

    qpu_base_path = Path(args.qpu_base) if args.qpu_base else find_latest_qpu_base()
    calibration = build_qpu_calibration(qpu_base_path)

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = ANALYSIS_DIR / f"dm_probe_06_frequency_base10_seed_projector_{now_tag()}"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 96}")
    print("  GHOST ORACLE SUITE — D_M PROBE 06: FREQUENCY BASE-10 SEED PROJECTOR")
    print(f"{'=' * 96}")
    print(f"  QPU base      : {qpu_base_path}")
    print(f"  Out dir       : {out_dir}")
    print(f"  Job ID        : {calibration.job_id}")
    print(f"  Backend       : {calibration.backend}")
    print(f"  Datasets      : {DATASETS}")
    print(f"  n             : {args.n}")
    print(f"  input_dim     : {args.input_dim}")
    print(f"  output_dim    : {args.output_dim}")
    print(f"  k             : {args.k}")
    print(f"  random_trials : {args.random_trials}")
    print(f"  master_seed   : {master_seed}")
    print(f"  seed_source   : {seed_source}")
    print("\n[SCOPE]")
    print("  ONE D_M projector only.")
    print("  Frequency/base-10 stepping enabled.")
    print("  Seed folded as active dimensional channel.")
    print("  No D_M channels.")
    print("  No D_M scheduler.")
    print("  No D_M family-best.")

    all_rows: List[EvalRow] = []
    all_trials: List[TrialRow] = []

    for dataset in DATASETS:
        print(f"\n[DATASET] {dataset}")

        x, labels = make_dataset(dataset, args.n, args.input_dim, rng)
        x = standardize(x)
        ref_nn = knn_indices(x, args.k)

        rows: List[EvalRow] = []

        rows.append(eval_embedding(
            dataset=dataset,
            method="identity_reference",
            x_ref=x,
            z=x,
            labels=labels,
            ref_nn=ref_nn,
            k=args.k,
            rng=rng,
        ))

        rows.append(eval_embedding(
            dataset=dataset,
            method="dm_frequency_base10_seed_projector",
            x_ref=x,
            z=dm_frequency_base10_seed_projector(x, args.output_dim, calibration),
            labels=labels,
            ref_nn=ref_nn,
            k=args.k,
            rng=rng,
            trials=1,
            stat="single",
        ))

        rows.append(eval_embedding(
            dataset=dataset,
            method="pca_projection",
            x_ref=x,
            z=pca_project(x, args.output_dim),
            labels=labels,
            ref_nn=ref_nn,
            k=args.k,
            rng=rng,
        ))

        rows.append(eval_embedding(
            dataset=dataset,
            method="random_gaussian_single",
            x_ref=x,
            z=random_gaussian_project(x, args.output_dim, rng),
            labels=labels,
            ref_nn=ref_nn,
            k=args.k,
            rng=rng,
            trials=1,
            stat="single",
        ))

        rows.append(eval_embedding(
            dataset=dataset,
            method="random_shuffle_crop",
            x_ref=x,
            z=random_shuffle_crop(x, args.output_dim, rng),
            labels=labels,
            ref_nn=ref_nn,
            k=args.k,
            rng=rng,
            trials=1,
            stat="single",
        ))

        rows.append(eval_embedding(
            dataset=dataset,
            method="random_sign_flip_crop",
            x_ref=x,
            z=random_sign_flip_crop(x, args.output_dim, rng),
            labels=labels,
            ref_nn=ref_nn,
            k=args.k,
            rng=rng,
            trials=1,
            stat="single",
        ))

        for family, fn in [
            ("random_gaussian", random_gaussian_project),
            ("random_orthogonal", random_orthogonal_project),
            ("random_sparse_achlioptas", random_sparse_achlioptas_project),
        ]:
            agg, trials = run_random_family(
                dataset=dataset,
                family=family,
                fn=fn,
                x=x,
                labels=labels,
                ref_nn=ref_nn,
                output_dim=args.output_dim,
                k=args.k,
                rng=rng,
                trials=args.random_trials,
            )
            rows.extend(agg)
            all_trials.extend(trials)

        all_rows.extend(rows)

        ranked = sorted(
            [r for r in rows if r.method != "identity_reference"],
            key=lambda r: (
                r.neighbor_overlap_at_k,
                r.trustworthiness_lite,
                r.label_purity_at_k,
                r.ranking_score,
            ),
            reverse=True,
        )

        for r in ranked:
            print(
                f"  {r.method:<42} "
                f"overlap={r.neighbor_overlap_at_k:7.4f} "
                f"trust={r.trustworthiness_lite:7.4f} "
                f"purity={r.label_purity_at_k:7.4f} "
                f"rankcorr={r.spearman_distance_rank_sample:7.4f} "
                f"score={r.ranking_score:7.4f}"
            )

    summary = summarize_results(all_rows)

    write_csv(out_dir / "dm_probe_06_results.csv", [asdict(r) for r in all_rows])
    write_csv(out_dir / "dm_probe_06_random_trials.csv", [asdict(r) for r in all_trials])
    write_csv(out_dir / "dm_probe_06_method_summary.csv", summary)

    config = {
        "schema": "ghost_oracle.dm.probe_06_frequency_base10_seed_projector.v1",
        "operator": "D_M",
        "probe": "dm_probe_06_frequency_base10_seed_projector",
        "qpu_base_path": str(qpu_base_path),
        "datasets": DATASETS,
        "n": int(args.n),
        "input_dim": int(args.input_dim),
        "output_dim": int(args.output_dim),
        "k": int(args.k),
        "random_trials": int(args.random_trials),
        "master_seed": int(master_seed),
        "seed_source": seed_source,
        "calibration": asdict(calibration),
        "scope": {
            "one_dm_projector_only": True,
            "frequency_base10_enabled": True,
            "seed_folded_as_active_dimension": True,
            "no_dm_channels": True,
            "no_dm_scheduler": True,
            "no_dm_variant_search": True,
            "random_best_is_ceiling_only": True,
            "synthetic_only": True,
            "final_benchmark": False,
        },
    }

    with open(out_dir / "dm_probe_06_config.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(config), f, indent=2)

    with open(out_dir / "dm_probe_06_calibration_profile.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(asdict(calibration)), f, indent=2)

    write_report(out_dir / "dm_probe_06_report.md", config, all_rows, summary)
    maybe_make_plots(out_dir, summary)

    print("\n[OVERALL RANKING]")
    for i, r in enumerate(summary, start=1):
        print(
            f"  {i:02d}. {r['method']:<42} "
            f"overlap={r['neighbor_overlap_at_k_mean']:7.4f} "
            f"trust={r['trustworthiness_lite_mean']:7.4f} "
            f"purity={r['label_purity_at_k_mean']:7.4f} "
            f"rankcorr={r['spearman_distance_rank_sample_mean']:7.4f} "
            f"score={r['ranking_score_mean']:7.4f}"
        )

    print(f"\n{'=' * 96}")
    print("  D_M PROBE 06 COMPLETE")
    print(f"{'=' * 96}")
    print(f"  Report      : {out_dir / 'dm_probe_06_report.md'}")
    print(f"  Results CSV : {out_dir / 'dm_probe_06_results.csv'}")
    print(f"  Summary CSV : {out_dir / 'dm_probe_06_method_summary.csv'}")
    print(f"  Trials CSV  : {out_dir / 'dm_probe_06_random_trials.csv'}")
    print(f"  Config JSON : {out_dir / 'dm_probe_06_config.json'}")
    print(f"  Calibration : {out_dir / 'dm_probe_06_calibration_profile.json'}")
    print(f"{'=' * 96}\n")


if __name__ == "__main__":
    main()