#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GHOST ORACLE SUITE — D_M PROBE 04: CALIBRATED SEED PROJECTION
================================================================================

Purpose
-------
Probe 03 showed that random/secret seed projection is not merely a baseline.
It is a strong dimensional channel.

Probe 04 tests the corrected D_M hypothesis:

    The 7th D_M channel is not linear.
    The 7th D_M channel is a calibrated seed channel.

The calibrated seed channel is derived from the frozen QPU D_M base and Probe 01
channel weights. It is then used to generate deterministic projection matrices
for synthetic dimensional-compression tests.

This probe asks:

    Can a single QPU-calibrated seed projection compete with or beat
    uncalibrated random projection mean performance?

Scope
-----
Self-contained D_M only.
No G_M / S_M / T_S / F_M interaction.
No final benchmark claim.
Synthetic rehearsal only.

Internal D_M channels tested here
---------------------------------
    calibrated_seed_channel
    local_order_channel
    collapse_channel
    mutation_channel
    symmetry_boundary_channel
    rank_spread_channel
    composite_dm_channel

Baselines / controls
--------------------
    pca_projection
    random_gaussian_mean/best/worst_N
    random_orthogonal_mean/best/worst_N
    random_sparse_achlioptas_mean/best/worst_N
    secret_seed_single
    qpu_calibrated_seed_single
    qpu_calibrated_seed_family_mean/best/worst_K
    qpu_calibrated_seed_scheduler

Important
---------
The "best" random rows are diagnostic ceilings, not fair single-shot baselines.
The main early win condition is:

    qpu_calibrated_seed_single > random_*_mean_N

or:

    qpu_calibrated_seed_family_mean > random_*_mean_N

Usage
-----
    python ghost_oracle/D_M/probes/dm_probe_04_calibrated_seed_projection.py

    python ghost_oracle/D_M/probes/dm_probe_04_calibrated_seed_projection.py ^
        --qpu-base ghost_oracle/D_M/data/dm_job_d8fb033o3njc73f01170.npz ^
        --channels ghost_oracle/D_M/probes/analysis/<probe01>/dm_probe_01_channels.npz ^
        --random-trials 32 ^
        --seed-family-size 16

================================================================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import secrets
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
DM_DIR = HERE.parent
DATA_DIR = DM_DIR / "data"
ANALYSIS_DIR = HERE / "analyze"


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_N = 5000
DEFAULT_INPUT_DIM = 64
DEFAULT_OUTPUT_DIM = 4
DEFAULT_K = 10
DEFAULT_RANDOM_TRIALS = 128
DEFAULT_SEED_FAMILY_SIZE = 32

DATASETS = [
    "blobs",
    "rings",
    "swiss_roll_like",
    "s_curve_like",
    "sparse_binary",
]

DM_CHANNEL_NAMES = [
    "local_order_channel",
    "collapse_channel",
    "mutation_channel",
    "symmetry_boundary_channel",
    "rank_spread_channel",
    "composite_dm_channel",
]


# =============================================================================
# SMALL HELPERS
# =============================================================================

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_master_seed(seed_arg: int | None) -> int:
    if seed_arg is not None:
        return int(seed_arg)
    return secrets.randbits(128)


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


def stable_hash_to_seed(payload: bytes, salt: str = "") -> int:
    h = hashlib.blake2b(digest_size=16)
    h.update(payload)
    h.update(salt.encode("utf-8"))
    return int.from_bytes(h.digest(), "little", signed=False)


def array_payload(*arrays: np.ndarray, extra: str = "") -> bytes:
    h = hashlib.blake2b(digest_size=32)
    h.update(extra.encode("utf-8"))
    for arr in arrays:
        a = np.asarray(arr)
        h.update(str(a.shape).encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(np.ascontiguousarray(a).tobytes())
    return h.digest()


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
    vals = []
    for i in range(ref_nn.shape[0]):
        vals.append(len(set(ref_nn[i]).intersection(set(test_nn[i]))) / float(k))
    return float(np.mean(vals))


def label_purity(labels: np.ndarray, test_nn: np.ndarray) -> float:
    labels = np.asarray(labels)
    vals = []
    for i in range(test_nn.shape[0]):
        vals.append(float(np.mean(labels[test_nn[i]] == labels[i])))
    return float(np.mean(vals))


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


def score_for_ranking(
    overlap: float,
    trust: float,
    purity: float,
    rankcorr: float,
) -> float:
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
# INPUT DISCOVERY / LOADING
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


def find_latest_probe01_channels() -> Path:
    matches = sorted(
        ANALYSIS_DIR.glob("dm_probe_01_channel_extract_*/dm_probe_01_channels.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError("No Probe 01 channel NPZ found. Pass --channels explicitly.")
    return matches[0]


def state_to_bit_weight(state_prob: np.ndarray) -> np.ndarray:
    p = normalize_prob(state_prob)
    out = np.zeros(4, dtype=np.float64)
    for state in range(16):
        for bit in range(4):
            if (state >> bit) & 1:
                out[bit] += p[state]
    return out


def load_probe01_channels(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    npz = np.load(path, allow_pickle=True)
    names = [str(x) for x in np.asarray(npz["channel_names"])]

    channels: Dict[str, Dict[str, np.ndarray]] = {}

    for name in names:
        state_key = f"{name}_state_prob"
        bit_key = f"{name}_bit_weight"
        centered_key = f"{name}_bit_centered"

        if state_key not in npz.files:
            continue

        state_prob = normalize_prob(np.asarray(npz[state_key], dtype=np.float64))
        bit_weight = (
            np.asarray(npz[bit_key], dtype=np.float64)
            if bit_key in npz.files
            else state_to_bit_weight(state_prob)
        )
        bit_centered = (
            np.asarray(npz[centered_key], dtype=np.float64)
            if centered_key in npz.files
            else bit_weight - np.mean(bit_weight)
        )

        channels[name] = {
            "state_prob": state_prob,
            "bit_weight": bit_weight,
            "bit_centered": bit_centered,
        }

    missing = [name for name in DM_CHANNEL_NAMES if name not in channels]
    if missing:
        raise RuntimeError(f"Missing required channels: {missing}")

    return channels


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


# =============================================================================
# CALIBRATION
# =============================================================================

@dataclass
class CalibrationProfile:
    qpu_base_path: str
    channels_path: str
    job_id: str
    backend: str
    num_tiles: int
    shots: int

    seed_base: int
    seed_payload_hex: str

    tile_state_prob: List[List[float]]
    global_state_prob: List[float]
    global_bit_weight: List[float]
    channel_state_mix: List[float]
    channel_bit_mix: List[float]

    qpu_feature_vector: List[float]
    qpu_feature_centered: List[float]


def extract_qpu_calibration(
    qpu_base_path: Path,
    channels_path: Path,
    channels: Dict[str, Dict[str, np.ndarray]],
) -> CalibrationProfile:
    npz = np.load(qpu_base_path, allow_pickle=True)

    job_id = str(load_npz_scalar(npz["job_id"])) if "job_id" in npz.files else qpu_base_path.stem
    backend = str(load_npz_scalar(npz["backend"])) if "backend" in npz.files else "unknown"
    num_tiles = int(load_npz_scalar(npz["num_tiles"])) if "num_tiles" in npz.files else int(np.asarray(npz["dim"]).shape[0])
    shots = int(load_npz_scalar(npz["shots"])) if "shots" in npz.files else -1

    tile_probs: List[np.ndarray] = []
    feature_parts: List[float] = []

    for tile in range(num_tiles):
        arr = get_tile_arrays(npz, tile)

        state = arr["dim_state"]
        hist = np.bincount(state.astype(np.int64), minlength=16).astype(np.float64)
        prob = normalize_prob(hist)
        tile_probs.append(prob)

        dim = arr["dim"].astype(np.float64)
        feature_parts.extend(prob.tolist())
        feature_parts.extend(np.mean(dim, axis=0).tolist())
        feature_parts.append(float(np.mean(arr["ctrl"])))
        feature_parts.append(float(np.mean(arr["edge"])))
        feature_parts.append(float(np.mean(arr["aux"])))
        feature_parts.append(float(np.mean(arr["meta"])))
        feature_parts.append(float(np.mean(arr["popcount"])))
        feature_parts.append(float(np.mean(arr["parity"])))
        feature_parts.append(float(np.mean(arr["boundary"])))
        feature_parts.append(float(np.mean(arr["interior"])))

    tile_state_prob = np.stack(tile_probs, axis=0)
    global_state_prob = normalize_prob(np.mean(tile_state_prob, axis=0))
    global_bit_weight = state_to_bit_weight(global_state_prob)

    channel_probs = []
    channel_bits = []
    for name in DM_CHANNEL_NAMES:
        channel_probs.append(normalize_prob(channels[name]["state_prob"]))
        channel_bits.append(np.asarray(channels[name]["bit_weight"], dtype=np.float64))

    channel_state_mix = normalize_prob(np.mean(np.stack(channel_probs, axis=0), axis=0))
    channel_bit_mix = np.mean(np.stack(channel_bits, axis=0), axis=0)

    qpu_feature = np.asarray(feature_parts, dtype=np.float64)
    qpu_centered = qpu_feature - float(np.mean(qpu_feature))

    payload = array_payload(
        tile_state_prob,
        global_state_prob,
        global_bit_weight,
        channel_state_mix,
        channel_bit_mix,
        qpu_feature,
        extra=f"{job_id}|{backend}|{num_tiles}|{shots}",
    )
    seed_base = stable_hash_to_seed(payload, salt="D_M_CALIBRATED_SEED_BASE_V1")

    return CalibrationProfile(
        qpu_base_path=str(qpu_base_path),
        channels_path=str(channels_path),
        job_id=job_id,
        backend=backend,
        num_tiles=num_tiles,
        shots=shots,

        seed_base=int(seed_base),
        seed_payload_hex=payload.hex(),

        tile_state_prob=tile_state_prob.tolist(),
        global_state_prob=global_state_prob.tolist(),
        global_bit_weight=global_bit_weight.tolist(),
        channel_state_mix=channel_state_mix.tolist(),
        channel_bit_mix=channel_bit_mix.tolist(),

        qpu_feature_vector=qpu_feature.tolist(),
        qpu_feature_centered=qpu_centered.tolist(),
    )


# =============================================================================
# PROJECTIONS
# =============================================================================

def pca_project(x: np.ndarray, output_dim: int) -> np.ndarray:
    x0 = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    return standardize(x0 @ vt[:output_dim].T)


def random_gaussian_project(x: np.ndarray, output_dim: int, rng: np.random.Generator) -> np.ndarray:
    r = rng.normal(0.0, 1.0, size=(x.shape[1], output_dim))
    r = r / np.maximum(np.linalg.norm(r, axis=0, keepdims=True), 1e-12)
    return standardize(x @ r)


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


def dm_static_channel_project(
    x: np.ndarray,
    channel: Dict[str, np.ndarray],
    output_dim: int,
) -> np.ndarray:
    x = standardize(x)
    _, d = x.shape

    bit_weight = np.asarray(channel["bit_weight"], dtype=np.float64)
    bit_centered = np.asarray(channel["bit_centered"], dtype=np.float64)
    state_prob = normalize_prob(np.asarray(channel["state_prob"], dtype=np.float64))

    bw = bit_weight / max(float(np.mean(bit_weight)), 1e-12)
    bc = bit_centered / max(float(np.max(np.abs(bit_centered))), 1e-12)

    w = np.zeros((d, output_dim), dtype=np.float64)
    idx = np.arange(d)

    for out_j in range(output_dim):
        bit_j = out_j % 4
        phase = 2.0 * np.pi * (out_j + 1) * (idx + 1) / max(1, d)

        state_ids = (idx + out_j) % 16
        state_mod = state_prob[state_ids]
        state_mod = state_mod / max(float(np.mean(state_mod)), 1e-12)

        bucket = ((idx + out_j) % output_dim == 0).astype(np.float64)
        harmonic = 0.15 * np.sin(phase) + 0.10 * np.cos(2.0 * phase)

        col = (
            bucket * bw[bit_j]
            + harmonic * (1.0 + 0.25 * bc[bit_j])
            + 0.08 * state_mod
        )
        col *= (1.0 + 0.15 * bc[bit_j] * ((idx % 2) * 2.0 - 1.0))
        w[:, out_j] = col

    w = w - np.mean(w, axis=0, keepdims=True)
    w = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)

    z = x @ w

    if output_dim >= 2:
        energy = float(np.sum(state_prob * state_prob))
        z[:, 0] += 0.05 * energy * np.sin(z[:, 1])
        z[:, 1] += 0.05 * energy * np.cos(z[:, 0])

    return standardize(z)


def calibrated_seed_matrix(
    input_dim: int,
    output_dim: int,
    calibration: CalibrationProfile,
    variant: int = 0,
) -> np.ndarray:
    """
    Build a deterministic projection matrix from QPU calibration.

    This is the core Probe 04 object.

    It is not arbitrary random:
        - seed comes from frozen QPU D_M base
        - state probabilities shape row/column amplitudes
        - bit weights shape output-dimension columns
        - QPU feature vector phase-modulates the projection
    """
    seed = stable_hash_to_seed(
        bytes.fromhex(calibration.seed_payload_hex),
        salt=f"D_M_CALIBRATED_VARIANT_{variant}",
    )
    rng = np.random.default_rng(seed)

    state_prob = normalize_prob(np.asarray(calibration.global_state_prob, dtype=np.float64))
    bit_weight = np.asarray(calibration.global_bit_weight, dtype=np.float64)
    channel_state = normalize_prob(np.asarray(calibration.channel_state_mix, dtype=np.float64))
    channel_bit = np.asarray(calibration.channel_bit_mix, dtype=np.float64)
    qfeat = np.asarray(calibration.qpu_feature_centered, dtype=np.float64)

    if qfeat.size == 0:
        qfeat = np.ones(16, dtype=np.float64)

    qfeat_norm = qfeat / max(float(np.max(np.abs(qfeat))), 1e-12)

    # Base seeded Gaussian carrier.
    w = rng.normal(0.0, 1.0, size=(input_dim, output_dim))

    rows = np.arange(input_dim)
    cols = np.arange(output_dim)

    for j in range(output_dim):
        bit_j = j % 4

        state_ids = (rows + 3 * j + variant) % 16
        q_ids = (rows * (j + 1) + variant) % qfeat_norm.size

        state_amp = state_prob[state_ids]
        channel_amp = channel_state[(state_ids + j) % 16]
        qphase = qfeat_norm[q_ids]

        # Normalize local envelopes.
        state_amp = state_amp / max(float(np.mean(state_amp)), 1e-12)
        channel_amp = channel_amp / max(float(np.mean(channel_amp)), 1e-12)

        bit_amp = 0.5 * bit_weight[bit_j] + 0.5 * channel_bit[bit_j]
        bit_amp = bit_amp / max(float(np.mean(0.5 * bit_weight + 0.5 * channel_bit)), 1e-12)

        harmonic = (
            np.sin(2.0 * np.pi * (rows + 1) * (j + 1 + variant) / max(1, input_dim))
            + 0.5 * np.cos(2.0 * np.pi * (rows + 1) * (j + 2) / max(1, input_dim))
        )

        envelope = (
            0.65
            + 0.20 * state_amp
            + 0.15 * channel_amp
            + 0.10 * bit_amp
            + 0.08 * harmonic
            + 0.06 * qphase
        )

        w[:, j] *= envelope

    # QPU-calibrated sign fold from state/channel distribution.
    sign_carrier = np.ones(input_dim, dtype=np.float64)
    for i in range(input_dim):
        s = state_prob[i % 16] - channel_state[(i * 3 + variant) % 16]
        sign_carrier[i] = 1.0 if s >= 0 else -1.0
    w *= sign_carrier[:, None]

    w = w - np.mean(w, axis=0, keepdims=True)
    w = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)
    return w


def calibrated_seed_project(
    x: np.ndarray,
    output_dim: int,
    calibration: CalibrationProfile,
    variant: int = 0,
) -> np.ndarray:
    x = standardize(x)
    w = calibrated_seed_matrix(x.shape[1], output_dim, calibration, variant=variant)
    return standardize(x @ w)


def secret_seed_project(
    x: np.ndarray,
    output_dim: int,
    secret_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(secret_seed)
    return random_gaussian_project(x, output_dim, rng)


# =============================================================================
# EVALUATION DATA STRUCTURES
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
    variant: int


@dataclass
class SeedTraceRow:
    dataset: str
    method_family: str
    variant: int
    neighbor_overlap_at_k: float
    trustworthiness_lite: float
    label_purity_at_k: float
    spearman_distance_rank_sample: float
    ranking_score: float


# =============================================================================
# EVALUATION
# =============================================================================

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
    variant: int = -1,
) -> EvalRow:
    test_nn = knn_indices(z, k)
    overlap = neighbor_overlap(ref_nn, test_nn)
    trust = trustworthiness_lite(x_ref, z, k)
    purity = label_purity(labels, test_nn)
    rankcorr = spearman_distance_rank_sample(x_ref, z, rng)
    score = score_for_ranking(overlap, trust, purity, rankcorr)

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
        variant=int(variant),
    )


def aggregate_rows(dataset: str, method_base: str, rows: List[EvalRow], stat: str) -> EvalRow:
    if not rows:
        raise ValueError("cannot aggregate empty rows")

    if stat == "mean":
        return EvalRow(
            dataset=dataset,
            method=f"{method_base}_mean_{len(rows)}",
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
            variant=-1,
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
        method=f"{method_base}_{label}_{len(rows)}",
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
        variant=r.variant,
    )


def run_random_family(
    dataset: str,
    family_name: str,
    projector_fn: Any,
    x: np.ndarray,
    labels: np.ndarray,
    ref_nn: np.ndarray,
    output_dim: int,
    k: int,
    rng: np.random.Generator,
    trials: int,
) -> Tuple[List[EvalRow], List[SeedTraceRow]]:
    trial_rows: List[EvalRow] = []
    trace: List[SeedTraceRow] = []

    for t in range(trials):
        z = projector_fn(x, output_dim, rng)
        r = eval_embedding(
            dataset=dataset,
            method=f"{family_name}_trial_{t:03d}",
            x_ref=x,
            z=z,
            labels=labels,
            ref_nn=ref_nn,
            k=k,
            rng=rng,
            trials=trials,
            stat="trial",
            variant=t,
        )
        trial_rows.append(r)
        trace.append(SeedTraceRow(
            dataset=dataset,
            method_family=family_name,
            variant=t,
            neighbor_overlap_at_k=r.neighbor_overlap_at_k,
            trustworthiness_lite=r.trustworthiness_lite,
            label_purity_at_k=r.label_purity_at_k,
            spearman_distance_rank_sample=r.spearman_distance_rank_sample,
            ranking_score=r.ranking_score,
        ))

    return [
        aggregate_rows(dataset, family_name, trial_rows, "mean"),
        aggregate_rows(dataset, family_name, trial_rows, "best"),
        aggregate_rows(dataset, family_name, trial_rows, "worst"),
    ], trace


def run_calibrated_seed_family(
    dataset: str,
    x: np.ndarray,
    labels: np.ndarray,
    ref_nn: np.ndarray,
    output_dim: int,
    k: int,
    rng: np.random.Generator,
    calibration: CalibrationProfile,
    family_size: int,
) -> Tuple[List[EvalRow], List[SeedTraceRow]]:
    rows: List[EvalRow] = []
    trace: List[SeedTraceRow] = []

    for variant in range(family_size):
        z = calibrated_seed_project(x, output_dim, calibration, variant=variant)
        r = eval_embedding(
            dataset=dataset,
            method=f"qpu_calibrated_seed_variant_{variant:03d}",
            x_ref=x,
            z=z,
            labels=labels,
            ref_nn=ref_nn,
            k=k,
            rng=rng,
            trials=family_size,
            stat="variant",
            variant=variant,
        )
        rows.append(r)
        trace.append(SeedTraceRow(
            dataset=dataset,
            method_family="qpu_calibrated_seed",
            variant=variant,
            neighbor_overlap_at_k=r.neighbor_overlap_at_k,
            trustworthiness_lite=r.trustworthiness_lite,
            label_purity_at_k=r.label_purity_at_k,
            spearman_distance_rank_sample=r.spearman_distance_rank_sample,
            ranking_score=r.ranking_score,
        ))

    return [
        aggregate_rows(dataset, "qpu_calibrated_seed_family", rows, "mean"),
        aggregate_rows(dataset, "qpu_calibrated_seed_family", rows, "best"),
        aggregate_rows(dataset, "qpu_calibrated_seed_family", rows, "worst"),
    ], trace


def qpu_calibrated_seed_scheduler(
    dataset: str,
    x: np.ndarray,
    labels: np.ndarray,
    ref_nn: np.ndarray,
    channels: Dict[str, Dict[str, np.ndarray]],
    calibration: CalibrationProfile,
    output_dim: int,
    k: int,
    rng: np.random.Generator,
) -> EvalRow:
    """
    Simple scheduler:
        include calibrated seed channel + six D_M shaping channels
        score each candidate against synthetic reference
        combine top 3 with score-softmax weights

    This is still a rehearsal. Later versions should remove reference scoring.
    """
    candidates: Dict[str, np.ndarray] = {}
    candidates["calibrated_seed_channel"] = calibrated_seed_project(x, output_dim, calibration, variant=0)

    for name in DM_CHANNEL_NAMES:
        candidates[name] = dm_static_channel_project(x, channels[name], output_dim)

    rows: List[Tuple[str, EvalRow]] = []
    for name, z in candidates.items():
        r = eval_embedding(
            dataset=dataset,
            method=name,
            x_ref=x,
            z=z,
            labels=labels,
            ref_nn=ref_nn,
            k=k,
            rng=rng,
        )
        rows.append((name, r))

    rows_sorted = sorted(rows, key=lambda nr: nr[1].ranking_score, reverse=True)
    top = rows_sorted[:3]

    scores = np.asarray([r.ranking_score for _, r in top], dtype=np.float64)
    scores = scores - np.max(scores)
    weights = np.exp(scores / 0.10)
    weights = weights / max(float(np.sum(weights)), 1e-12)

    z_mix = np.zeros_like(candidates[top[0][0]])
    for w, (name, _) in zip(weights, top):
        z_mix += float(w) * candidates[name]

    z_mix = standardize(z_mix)

    return eval_embedding(
        dataset=dataset,
        method="qpu_calibrated_seed_scheduler",
        x_ref=x,
        z=z_mix,
        labels=labels,
        ref_nn=ref_nn,
        k=k,
        rng=rng,
    )


# =============================================================================
# SUMMARY / OUTPUT
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
        if "_variant_" in r.method:
            continue
        by_method.setdefault(r.method, []).append(r)

    out: List[Dict[str, Any]] = []
    for method, items in sorted(by_method.items()):
        out.append({
            "method": method,
            "n_datasets": len(items),
            "neighbor_overlap_at_k_mean": float(np.mean([x.neighbor_overlap_at_k for x in items])),
            "trustworthiness_lite_mean": float(np.mean([x.trustworthiness_lite for x in items])),
            "label_purity_at_k_mean": float(np.mean([x.label_purity_at_k for x in items])),
            "spearman_distance_rank_sample_mean": float(np.mean([x.spearman_distance_rank_sample for x in items])),
            "ranking_score_mean": float(np.mean([x.ranking_score for x in items])),
            "compression_ratio_mean": float(np.mean([x.compression_ratio for x in items])),
        })

    out.sort(
        key=lambda r: (
            r["neighbor_overlap_at_k_mean"],
            r["trustworthiness_lite_mean"],
            r["label_purity_at_k_mean"],
            r["ranking_score_mean"],
        ),
        reverse=True,
    )
    return out


def write_report(
    path: Path,
    qpu_base_path: Path,
    channels_path: Path,
    calibration: CalibrationProfile,
    config: Dict[str, Any],
    rows: List[EvalRow],
    summary: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []

    lines.append("# D_M Probe 04 — Calibrated Seed Projection")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Self-contained `D_M` only.")
    lines.append("- Uses frozen QPU D_M base as seed calibration.")
    lines.append("- Uses Probe 01 D_M channel weights as shaping priors.")
    lines.append("- No cross-operator interaction.")
    lines.append("- No final benchmark claim.")
    lines.append("")
    lines.append("## Core Correction")
    lines.append("")
    lines.append(
        "Probe 03 showed that random/secret-seeded projections are strong dimensional "
        "channels. Probe 04 treats seed projection as the missing seventh D_M channel. "
        "The tested channel is not arbitrary random; it is derived from the frozen QPU "
        "dimensional record."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- QPU base: `{qpu_base_path}`")
    lines.append(f"- Channels: `{channels_path}`")
    lines.append(f"- Job ID: `{calibration.job_id}`")
    lines.append(f"- Backend: `{calibration.backend}`")
    lines.append(f"- Calibration seed base: `{calibration.seed_base}`")
    lines.append(f"- Random trials: `{config['random_trials']}`")
    lines.append(f"- Calibrated seed family size: `{config['seed_family_size']}`")
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
    lines.append("## Dataset-Level Results")
    lines.append("")
    lines.append("| Dataset | Method | Neighbor overlap@k | Trust-lite | Label purity@k | Distance-rank corr | Score |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for r in rows:
        if "_trial_" in r.method or "_variant_" in r.method:
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
        "`random_*_best_N` is an oracle ceiling, not a fair single-shot baseline. "
        "The key early comparison is calibrated seed single/family mean versus "
        "random mean families. If calibrated seed beats or approaches random mean "
        "without benchmark-selected search, the seed channel is worth carrying forward."
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
    plt.title("D_M Probe 04 — Calibrated Seed Projection Ranking")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "dm_probe_04_overall_ranking.png", dpi=160)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — D_M Probe 04 calibrated seed projection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qpu-base", default=None, help="Frozen dm_job_<JOB_ID>.npz. Defaults to latest.")
    p.add_argument("--channels", default=None, help="Probe 01 dm_probe_01_channels.npz. Defaults to latest.")
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--input-dim", type=int, default=DEFAULT_INPUT_DIM)
    p.add_argument("--output-dim", type=int, default=DEFAULT_OUTPUT_DIM)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--random-trials", type=int, default=DEFAULT_RANDOM_TRIALS)
    p.add_argument("--seed-family-size", type=int, default=DEFAULT_SEED_FAMILY_SIZE)
    p.add_argument("--seed", type=int, default=None, help="Optional deterministic master seed.")
    p.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to probes/analysis/dm_probe_04_calibrated_seed_<timestamp>/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    master_seed = make_master_seed(args.seed)
    seed_source = "user" if args.seed is not None else "secrets.randbits(128)"
    rng = np.random.default_rng(master_seed)

    qpu_base_path = Path(args.qpu_base) if args.qpu_base else find_latest_qpu_base()
    channels_path = Path(args.channels) if args.channels else find_latest_probe01_channels()

    channels = load_probe01_channels(channels_path)
    calibration = extract_qpu_calibration(qpu_base_path, channels_path, channels)

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = ANALYSIS_DIR / f"dm_probe_04_calibrated_seed_{now_tag()}"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 92}")
    print("  GHOST ORACLE SUITE — D_M PROBE 04: CALIBRATED SEED PROJECTION")
    print(f"{'=' * 92}")
    print(f"  QPU base         : {qpu_base_path}")
    print(f"  Channels         : {channels_path}")
    print(f"  Out dir          : {out_dir}")
    print(f"  Job ID           : {calibration.job_id}")
    print(f"  Backend          : {calibration.backend}")
    print(f"  Calibration seed : {calibration.seed_base}")
    print(f"  Datasets         : {DATASETS}")
    print(f"  n                : {args.n}")
    print(f"  input_dim        : {args.input_dim}")
    print(f"  output_dim       : {args.output_dim}")
    print(f"  k                : {args.k}")
    print(f"  random_trials    : {args.random_trials}")
    print(f"  seed_family_size : {args.seed_family_size}")
    print(f"  master_seed      : {master_seed}")
    print(f"  seed_source      : {seed_source}")
    print("\n[SCOPE]")
    print("  Self-contained D_M only.")
    print("  Seed is internal calibrated dimensional channel.")
    print("  PCA remains baseline only.")
    print("  No cross-operator interaction.")
    print("  Synthetic rehearsal only.")

    all_rows: List[EvalRow] = []
    all_seed_trace: List[SeedTraceRow] = []

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
            method="pca_projection",
            x_ref=x,
            z=pca_project(x, args.output_dim),
            labels=labels,
            ref_nn=ref_nn,
            k=args.k,
            rng=rng,
        ))

        # Hardened random families.
        for family_name, fn in [
            ("random_gaussian", random_gaussian_project),
            ("random_orthogonal", random_orthogonal_project),
            ("random_sparse_achlioptas", random_sparse_achlioptas_project),
        ]:
            agg, trace = run_random_family(
                dataset=dataset,
                family_name=family_name,
                projector_fn=fn,
                x=x,
                labels=labels,
                ref_nn=ref_nn,
                output_dim=args.output_dim,
                k=args.k,
                rng=rng,
                trials=args.random_trials,
            )
            rows.extend(agg)
            all_seed_trace.extend(trace)

        # Single crop controls.
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

        # Secret single seed control.
        secret_single_seed = secrets.randbits(128)
        rows.append(eval_embedding(
            dataset=dataset,
            method="secret_seed_single",
            x_ref=x,
            z=secret_seed_project(x, args.output_dim, secret_single_seed),
            labels=labels,
            ref_nn=ref_nn,
            k=args.k,
            rng=rng,
            trials=1,
            stat="single_secret",
            variant=0,
        ))

        # QPU calibrated seed single.
        rows.append(eval_embedding(
            dataset=dataset,
            method="qpu_calibrated_seed_single",
            x_ref=x,
            z=calibrated_seed_project(x, args.output_dim, calibration, variant=0),
            labels=labels,
            ref_nn=ref_nn,
            k=args.k,
            rng=rng,
            trials=1,
            stat="single_qpu_calibrated",
            variant=0,
        ))

        # QPU calibrated seed family.
        seed_family_rows, seed_trace = run_calibrated_seed_family(
            dataset=dataset,
            x=x,
            labels=labels,
            ref_nn=ref_nn,
            output_dim=args.output_dim,
            k=args.k,
            rng=rng,
            calibration=calibration,
            family_size=args.seed_family_size,
        )
        rows.extend(seed_family_rows)
        all_seed_trace.extend(seed_trace)

        # Static D_M channels remain for continuity.
        for name in DM_CHANNEL_NAMES:
            rows.append(eval_embedding(
                dataset=dataset,
                method=f"dm_static_{name}",
                x_ref=x,
                z=dm_static_channel_project(x, channels[name], args.output_dim),
                labels=labels,
                ref_nn=ref_nn,
                k=args.k,
                rng=rng,
            ))

        # Calibrated seed + D_M shaping scheduler.
        rows.append(qpu_calibrated_seed_scheduler(
            dataset=dataset,
            x=x,
            labels=labels,
            ref_nn=ref_nn,
            channels=channels,
            calibration=calibration,
            output_dim=args.output_dim,
            k=args.k,
            rng=rng,
        ))

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
                f"  {r.method:<48} "
                f"overlap={r.neighbor_overlap_at_k:7.4f} "
                f"trust={r.trustworthiness_lite:7.4f} "
                f"purity={r.label_purity_at_k:7.4f} "
                f"rankcorr={r.spearman_distance_rank_sample:7.4f} "
                f"score={r.ranking_score:7.4f}"
            )

    summary = summarize_results(all_rows)

    write_csv(out_dir / "dm_probe_04_results.csv", [asdict(r) for r in all_rows])
    write_csv(out_dir / "dm_probe_04_seed_trace.csv", [asdict(r) for r in all_seed_trace])
    write_csv(out_dir / "dm_probe_04_method_summary.csv", summary)

    config = {
        "schema": "ghost_oracle.dm.probe_04_calibrated_seed.v1",
        "operator": "D_M",
        "probe": "dm_probe_04_calibrated_seed_projection",
        "qpu_base_path": str(qpu_base_path),
        "channels_path": str(channels_path),
        "datasets": DATASETS,
        "n": int(args.n),
        "input_dim": int(args.input_dim),
        "output_dim": int(args.output_dim),
        "k": int(args.k),
        "random_trials": int(args.random_trials),
        "seed_family_size": int(args.seed_family_size),
        "master_seed": int(master_seed),
        "seed_source": seed_source,
        "calibration": asdict(calibration),
        "scope": {
            "self_consistent_operator_only": True,
            "uses_other_operators": False,
            "qpu_calibrated_seed_channel": True,
            "linear_is_baseline_not_internal_channel": True,
            "synthetic_only": True,
            "final_benchmark": False,
        },
    }

    with open(out_dir / "dm_probe_04_config.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(config), f, indent=2)

    with open(out_dir / "dm_probe_04_calibration_profile.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(asdict(calibration)), f, indent=2)

    write_report(
        path=out_dir / "dm_probe_04_report.md",
        qpu_base_path=qpu_base_path,
        channels_path=channels_path,
        calibration=calibration,
        config=config,
        rows=all_rows,
        summary=summary,
    )

    maybe_make_plots(out_dir, summary)

    print("\n[OVERALL RANKING]")
    for i, r in enumerate(summary, start=1):
        print(
            f"  {i:02d}. {r['method']:<48} "
            f"overlap={r['neighbor_overlap_at_k_mean']:7.4f} "
            f"trust={r['trustworthiness_lite_mean']:7.4f} "
            f"purity={r['label_purity_at_k_mean']:7.4f} "
            f"rankcorr={r['spearman_distance_rank_sample_mean']:7.4f} "
            f"score={r['ranking_score_mean']:7.4f}"
        )

    print(f"\n{'=' * 92}")
    print("  D_M PROBE 04 COMPLETE")
    print(f"{'=' * 92}")
    print(f"  Report       : {out_dir / 'dm_probe_04_report.md'}")
    print(f"  Results CSV  : {out_dir / 'dm_probe_04_results.csv'}")
    print(f"  Summary CSV  : {out_dir / 'dm_probe_04_method_summary.csv'}")
    print(f"  Seed trace   : {out_dir / 'dm_probe_04_seed_trace.csv'}")
    print(f"  Config JSON  : {out_dir / 'dm_probe_04_config.json'}")
    print(f"  Calibration  : {out_dir / 'dm_probe_04_calibration_profile.json'}")
    print(f"{'=' * 92}\n")


if __name__ == "__main__":
    main()