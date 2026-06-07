#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GHOST ORACLE SUITE — D_M PROBE 03: LINEAR-ACTIVATED CHANNEL SCHEDULER
================================================================================

Purpose
-------
Probe 02 tested D_M channels as static single-track projections and got beaten by
PCA/random projection. That failure exposed the missing obvious dimension:

    the linear direction.

Probe 03 treats linear structure as an internal D_M channel, not merely as an
external challenger.

D_M now has 7 internal channels:

    0. linear_channel
    1. local_order_channel
    2. collapse_channel
    3. mutation_channel
    4. symmetry_boundary_channel
    5. rank_spread_channel
    6. composite_dm_channel

This updated Probe 03 also fixes the weak random baseline problem.

Instead of one lucky Gaussian random projection, this probe evaluates:

    random_gaussian_mean_N
    random_gaussian_best_N
    random_gaussian_worst_N
    random_orthogonal_mean_N
    random_orthogonal_best_N
    random_orthogonal_worst_N
    random_sparse_achlioptas_mean_N
    random_sparse_achlioptas_best_N
    random_sparse_achlioptas_worst_N
    random_shuffle_crop
    random_sign_flip_crop

Randomness
----------
By default this probe uses a secrets-generated 128-bit master seed, then feeds
that seed into NumPy's Generator for reproducibility inside the run.

Use --seed if you want deterministic reproduction.

Scope
-----
Self-contained D_M only.
No cross-operator interaction.
No final benchmark claim.
Synthetic rehearsal only.

Usage
-----
    python ghost_oracle/D_M/probes/dm_probe_03_linear_scheduler.py

    python ghost_oracle/D_M/probes/dm_probe_03_linear_scheduler.py ^
        --channels ghost_oracle/D_M/probes/analysis/<probe01>/dm_probe_01_channels.npz

    python ghost_oracle/D_M/probes/dm_probe_03_linear_scheduler.py ^
        --seed 12345 --random-trials 64

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
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
DM_DIR = HERE.parent
ANALYSIS_DIR = HERE / "analyze"


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_N = 5000
DEFAULT_INPUT_DIM = 64
DEFAULT_OUTPUT_DIM = 4
DEFAULT_K = 10
DEFAULT_TIMESTEPS = 20
DEFAULT_RANDOM_TRIALS = 128

DATASETS = [
    "blobs",
    "rings",
    "swiss_roll_like",
    "s_curve_like",
    "sparse_binary",
]

DM_CHANNEL_ORDER = [
    "linear_channel",
    "local_order_channel",
    "collapse_channel",
    "mutation_channel",
    "symmetry_boundary_channel",
    "rank_spread_channel",
    "composite_dm_channel",
]


# =============================================================================
# HELPERS
# =============================================================================

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_master_seed(seed_arg: int | None) -> int:
    """
    Use user seed if provided. Otherwise generate a proper high-entropy seed.

    The generated seed is printed and saved to config so the run can be repeated.
    """
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


def standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - np.mean(x, axis=0, keepdims=True)) / np.maximum(
        np.std(x, axis=0, keepdims=True),
        eps,
    )


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    t = max(float(temperature), 1e-9)
    z = x / t
    z = z - np.max(z)
    e = np.exp(z)
    return e / max(float(np.sum(e)), 1e-12)


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
    scores = []
    for i in range(ref_nn.shape[0]):
        scores.append(len(set(ref_nn[i]).intersection(set(test_nn[i]))) / float(k))
    return float(np.mean(scores))


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


def pca_project(x: np.ndarray, output_dim: int) -> np.ndarray:
    x0 = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    return standardize(x0 @ vt[:output_dim].T)


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
# CHANNEL LOADING
# =============================================================================

def find_latest_probe01_channels() -> Path:
    matches = sorted(
        ANALYSIS_DIR.glob("dm_probe_01_channel_extract_*/dm_probe_01_channels.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            "No Probe 01 channel NPZ found. Pass --channels explicitly."
        )
    return matches[0]


def state_to_bit_weight(state_prob: np.ndarray) -> np.ndarray:
    p = np.asarray(state_prob, dtype=np.float64)
    p = p / max(float(np.sum(p)), 1e-12)

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

        state_prob = np.asarray(npz[state_key], dtype=np.float64)
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

    required = [
        "local_order_channel",
        "collapse_channel",
        "mutation_channel",
        "symmetry_boundary_channel",
        "rank_spread_channel",
        "composite_dm_channel",
    ]
    missing = [x for x in required if x not in channels]
    if missing:
        raise RuntimeError(f"missing required Probe 01 channels: {missing}")

    return channels


# =============================================================================
# RANDOM PROJECTION CONTROL SUITE
# =============================================================================

def random_gaussian_project(x: np.ndarray, output_dim: int, rng: np.random.Generator) -> np.ndarray:
    r = rng.normal(0.0, 1.0, size=(x.shape[1], output_dim))
    r = r / np.maximum(np.linalg.norm(r, axis=0, keepdims=True), 1e-12)
    return standardize(x @ r)


def random_orthogonal_project(x: np.ndarray, output_dim: int, rng: np.random.Generator) -> np.ndarray:
    d = x.shape[1]
    a = rng.normal(0.0, 1.0, size=(d, d))
    q, _ = np.linalg.qr(a)
    w = q[:, :output_dim]
    return standardize(x @ w)


def random_sparse_achlioptas_project(x: np.ndarray, output_dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sparse random projection using Achlioptas-style values:

        sqrt(3) with p=1/6
        0       with p=2/3
       -sqrt(3) with p=1/6
    """
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
# D_M CHANNEL PROJECTIONS
# =============================================================================

def dm_static_channel_project(
    x: np.ndarray,
    channel: Dict[str, np.ndarray],
    output_dim: int,
) -> np.ndarray:
    x = standardize(x)
    _, d = x.shape

    bit_weight = np.asarray(channel["bit_weight"], dtype=np.float64)
    bit_centered = np.asarray(channel["bit_centered"], dtype=np.float64)
    state_prob = np.asarray(channel["state_prob"], dtype=np.float64)

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


def build_all_phi(
    x: np.ndarray,
    channels: Dict[str, Dict[str, np.ndarray]],
    output_dim: int,
) -> Dict[str, np.ndarray]:
    """
    Build all seven internal D_M channel candidates.

    linear_channel is PCA. Here it is not merely a challenger; it is an internal
    D_M dimension that the scheduler may activate.
    """
    phis: Dict[str, np.ndarray] = {}

    phis["linear_channel"] = pca_project(x, output_dim)

    for name in [
        "local_order_channel",
        "collapse_channel",
        "mutation_channel",
        "symmetry_boundary_channel",
        "rank_spread_channel",
        "composite_dm_channel",
    ]:
        phis[name] = dm_static_channel_project(x, channels[name], output_dim)

    return phis


# =============================================================================
# EVALUATION STRUCTURES
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

    n: int
    k: int
    random_trials: int
    random_stat: str


@dataclass
class SchedulerStep:
    dataset: str
    timestep: int
    method: str
    neighbor_overlap_at_k: float
    trustworthiness_lite: float
    label_purity_at_k: float
    spearman_distance_rank_sample: float

    alpha_linear: float
    alpha_local_order: float
    alpha_collapse: float
    alpha_mutation: float
    alpha_symmetry_boundary: float
    alpha_rank_spread: float
    alpha_composite: float

    active_channel: str
    effective_alpha_dim: float
    z_variance_sum: float
    z_rank_proxy: int


def evaluate_embedding(
    x_ref: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    ref_nn: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    test_nn = knn_indices(z, k)
    overlap = neighbor_overlap(ref_nn, test_nn)
    trust = trustworthiness_lite(x_ref, z, k)
    purity = label_purity(labels, test_nn)
    rankcorr = spearman_distance_rank_sample(x_ref, z, rng)
    return overlap, trust, purity, rankcorr


def eval_row(
    dataset: str,
    method: str,
    x_ref: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    ref_nn: np.ndarray,
    k: int,
    rng: np.random.Generator,
    random_trials: int = 0,
    random_stat: str = "",
) -> EvalRow:
    overlap, trust, purity, rankcorr = evaluate_embedding(x_ref, z, labels, ref_nn, k, rng)

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
        n=int(x_ref.shape[0]),
        k=int(k),
        random_trials=int(random_trials),
        random_stat=str(random_stat),
    )


def score_eval_row_for_ranking(row: EvalRow) -> float:
    return (
        0.45 * row.neighbor_overlap_at_k
        + 0.25 * row.trustworthiness_lite
        + 0.20 * row.label_purity_at_k
        + 0.10 * max(-1.0, min(1.0, row.spearman_distance_rank_sample))
    )


def aggregate_random_trials(
    dataset: str,
    method_base: str,
    rows: List[EvalRow],
    stat: str,
) -> EvalRow:
    """
    Aggregate random trial rows into mean/best/worst pseudo-row.

    best/worst are selected by the same ranking score used elsewhere.
    """
    if not rows:
        raise ValueError("no rows to aggregate")

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
            n=rows[0].n,
            k=rows[0].k,
            random_trials=len(rows),
            random_stat="mean",
        )

    sorted_rows = sorted(rows, key=score_eval_row_for_ranking, reverse=True)

    if stat == "best":
        r = sorted_rows[0]
        label = "best"
    elif stat == "worst":
        r = sorted_rows[-1]
        label = "worst"
    else:
        raise ValueError(f"unknown random stat: {stat}")

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
        n=r.n,
        k=r.k,
        random_trials=len(rows),
        random_stat=label,
    )


def run_random_control_suite(
    dataset: str,
    x: np.ndarray,
    labels: np.ndarray,
    ref_nn: np.ndarray,
    output_dim: int,
    k: int,
    rng: np.random.Generator,
    random_trials: int,
) -> Tuple[List[EvalRow], List[EvalRow]]:
    """
    Return:
        aggregate_rows, per_trial_rows
    """
    aggregate_rows: List[EvalRow] = []
    per_trial_rows: List[EvalRow] = []

    random_families = [
        ("random_gaussian", random_gaussian_project),
        ("random_orthogonal", random_orthogonal_project),
        ("random_sparse_achlioptas", random_sparse_achlioptas_project),
    ]

    for family_name, fn in random_families:
        trial_rows: List[EvalRow] = []
        for trial in range(random_trials):
            z = fn(x, output_dim, rng)
            row = eval_row(
                dataset=dataset,
                method=f"{family_name}_trial_{trial:03d}",
                x_ref=x,
                z=z,
                labels=labels,
                ref_nn=ref_nn,
                k=k,
                rng=rng,
                random_trials=random_trials,
                random_stat="trial",
            )
            trial_rows.append(row)
            per_trial_rows.append(row)

        aggregate_rows.append(aggregate_random_trials(dataset, family_name, trial_rows, "mean"))
        aggregate_rows.append(aggregate_random_trials(dataset, family_name, trial_rows, "best"))
        aggregate_rows.append(aggregate_random_trials(dataset, family_name, trial_rows, "worst"))

    # Single-shot destructive-ish random controls.
    z_shuffle = random_shuffle_crop(x, output_dim, rng)
    aggregate_rows.append(eval_row(
        dataset=dataset,
        method="random_shuffle_crop",
        x_ref=x,
        z=z_shuffle,
        labels=labels,
        ref_nn=ref_nn,
        k=k,
        rng=rng,
        random_trials=1,
        random_stat="single",
    ))

    z_sign = random_sign_flip_crop(x, output_dim, rng)
    aggregate_rows.append(eval_row(
        dataset=dataset,
        method="random_sign_flip_crop",
        x_ref=x,
        z=z_sign,
        labels=labels,
        ref_nn=ref_nn,
        k=k,
        rng=rng,
        random_trials=1,
        random_stat="single",
    ))

    return aggregate_rows, per_trial_rows


# =============================================================================
# SCHEDULER
# =============================================================================

def alpha_effective_dim(alpha: np.ndarray) -> float:
    denom = float(np.sum(alpha * alpha))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 / denom)


def representation_rank_proxy(z: np.ndarray) -> int:
    _, s, _ = np.linalg.svd(z - np.mean(z, axis=0, keepdims=True), full_matrices=False)
    if s.size == 0:
        return 0
    threshold = 1e-6 * float(np.max(s))
    return int(np.sum(s > threshold))


def channel_scores_against_ref(
    x_ref: np.ndarray,
    phis: Dict[str, np.ndarray],
    labels: np.ndarray,
    ref_nn: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Synthetic rehearsal is allowed to use known reference neighborhoods.

    Later real-data benchmark can replace this with unsupervised diagnostics.
    """
    scores: Dict[str, float] = {}
    for name, z in phis.items():
        overlap, trust, purity, rankcorr = evaluate_embedding(x_ref, z, labels, ref_nn, k, rng)
        scores[name] = (
            0.45 * overlap
            + 0.25 * trust
            + 0.20 * purity
            + 0.10 * max(-1.0, min(1.0, rankcorr))
        )
    return scores


def scheduled_dm_project(
    dataset: str,
    x: np.ndarray,
    labels: np.ndarray,
    channels: Dict[str, Dict[str, np.ndarray]],
    output_dim: int,
    k: int,
    timesteps: int,
    rng: np.random.Generator,
    method: str,
) -> Tuple[np.ndarray, List[SchedulerStep]]:
    x = standardize(x)
    ref_nn = knn_indices(x, k)
    phis = build_all_phi(x, channels, output_dim)

    base_scores = channel_scores_against_ref(x, phis, labels, ref_nn, k, rng)
    score_vec = np.asarray([base_scores[name] for name in DM_CHANNEL_ORDER], dtype=np.float64)

    alpha = np.ones(len(DM_CHANNEL_ORDER), dtype=np.float64) / len(DM_CHANNEL_ORDER)
    alpha[DM_CHANNEL_ORDER.index("linear_channel")] += 0.35
    alpha = alpha / np.sum(alpha)

    z = np.zeros_like(phis["linear_channel"])
    trace: List[SchedulerStep] = []
    previous_overlap = 0.0

    for t in range(timesteps):
        if method == "dm7_soft_scheduler":
            temp = max(0.18, 0.75 - 0.06 * t)
            alpha = softmax(score_vec + 0.25 * np.log(alpha + 1e-12), temperature=temp)

        elif method == "dm7_hard_scheduler":
            adjusted = score_vec + 0.12 * np.log(alpha + 1e-12)
            active_idx = int(np.argmax(adjusted))
            alpha = np.zeros_like(alpha)
            alpha[active_idx] = 1.0

        elif method == "dm7_residual_scheduler":
            nonlin_names = [n for n in DM_CHANNEL_ORDER if n != "linear_channel"]
            nonlin_scores = np.asarray([base_scores[n] for n in nonlin_names], dtype=np.float64)
            nonlin_alpha = softmax(nonlin_scores, temperature=max(0.20, 0.65 - 0.05 * t))

            linear_share = max(0.35, 0.70 - 0.04 * t)
            alpha = np.zeros(len(DM_CHANNEL_ORDER), dtype=np.float64)
            alpha[DM_CHANNEL_ORDER.index("linear_channel")] = linear_share
            for a, name in zip(nonlin_alpha, nonlin_names):
                alpha[DM_CHANNEL_ORDER.index(name)] = (1.0 - linear_share) * a
            alpha = alpha / np.sum(alpha)

        else:
            raise ValueError(f"unknown scheduler method: {method}")

        z_new = np.zeros_like(z)
        for c_idx, name in enumerate(DM_CHANNEL_ORDER):
            z_new += alpha[c_idx] * phis[name]

        memory = 0.25 if t > 0 else 0.0
        z = standardize((1.0 - memory) * z_new + memory * z)

        overlap, trust, purity, rankcorr = evaluate_embedding(x, z, labels, ref_nn, k, rng)
        active_channel = DM_CHANNEL_ORDER[int(np.argmax(alpha))]

        improvement = overlap - previous_overlap
        if method != "dm7_hard_scheduler":
            score_vec = score_vec + 0.05 * improvement * alpha
        previous_overlap = overlap

        var_sum = float(np.sum(np.var(z, axis=0)))
        rank_proxy = representation_rank_proxy(z)

        trace.append(
            SchedulerStep(
                dataset=dataset,
                timestep=t,
                method=method,
                neighbor_overlap_at_k=overlap,
                trustworthiness_lite=trust,
                label_purity_at_k=purity,
                spearman_distance_rank_sample=rankcorr,

                alpha_linear=float(alpha[0]),
                alpha_local_order=float(alpha[1]),
                alpha_collapse=float(alpha[2]),
                alpha_mutation=float(alpha[3]),
                alpha_symmetry_boundary=float(alpha[4]),
                alpha_rank_spread=float(alpha[5]),
                alpha_composite=float(alpha[6]),

                active_channel=active_channel,
                effective_alpha_dim=alpha_effective_dim(alpha),
                z_variance_sum=var_sum,
                z_rank_proxy=rank_proxy,
            )
        )

    return z, trace


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

    out: List[Dict[str, Any]] = []
    for method, items in sorted(by_method.items()):
        out.append({
            "method": method,
            "n_datasets": len(items),
            "neighbor_overlap_at_k_mean": float(np.mean([x.neighbor_overlap_at_k for x in items])),
            "trustworthiness_lite_mean": float(np.mean([x.trustworthiness_lite for x in items])),
            "label_purity_at_k_mean": float(np.mean([x.label_purity_at_k for x in items])),
            "spearman_distance_rank_sample_mean": float(np.mean([x.spearman_distance_rank_sample for x in items])),
            "compression_ratio_mean": float(np.mean([x.compression_ratio for x in items])),
        })

    out.sort(
        key=lambda r: (
            r["neighbor_overlap_at_k_mean"],
            r["trustworthiness_lite_mean"],
            r["label_purity_at_k_mean"],
        ),
        reverse=True,
    )
    return out


def write_report(
    path: Path,
    channels_path: Path,
    config: Dict[str, Any],
    rows: List[EvalRow],
    summary: List[Dict[str, Any]],
    trace_rows: List[SchedulerStep],
) -> None:
    lines: List[str] = []

    lines.append("# D_M Probe 03 — Linear-Activated Channel Scheduler")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Self-contained `D_M` only.")
    lines.append("- Uses Probe 01 extracted D_M channels.")
    lines.append("- Adds `linear_channel` as internal D_M dimension.")
    lines.append("- Uses hardened random-control suite.")
    lines.append("- No cross-operator interaction.")
    lines.append("- No final benchmark claim.")
    lines.append("")
    lines.append("## Randomness")
    lines.append("")
    lines.append(f"- Master seed: `{config['master_seed']}`")
    lines.append(f"- Seed source: `{config['seed_source']}`")
    lines.append(f"- Random trials per random family: `{config['random_trials']}`")
    lines.append("")
    lines.append("## Core Correction")
    lines.append("")
    lines.append(
        "Probe 02 treated D_M channels as separate static tracks. Probe 03 treats "
        "them as internal dimensions of a timestep scheduler. The linear direction "
        "is included as one of the seven D_M channels rather than only as an external baseline."
    )
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Channels: `{channels_path}`")
    lines.append(f"- n: `{config['n']}`")
    lines.append(f"- input_dim: `{config['input_dim']}`")
    lines.append(f"- output_dim: `{config['output_dim']}`")
    lines.append(f"- k: `{config['k']}`")
    lines.append(f"- timesteps: `{config['timesteps']}`")
    lines.append("")
    lines.append("## Overall Ranking")
    lines.append("")
    lines.append("| Rank | Method | Neighbor overlap@k | Trust-lite | Label purity@k | Distance-rank corr |")
    lines.append("|---:|---|---:|---:|---:|---:|")

    for i, r in enumerate(summary, start=1):
        lines.append(
            f"| {i} | `{r['method']}` | "
            f"{r['neighbor_overlap_at_k_mean']:.4f} | "
            f"{r['trustworthiness_lite_mean']:.4f} | "
            f"{r['label_purity_at_k_mean']:.4f} | "
            f"{r['spearman_distance_rank_sample_mean']:.4f} |"
        )

    lines.append("")
    lines.append("## Dataset-Level Results")
    lines.append("")
    lines.append("| Dataset | Method | Neighbor overlap@k | Trust-lite | Label purity@k | Distance-rank corr |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in rows:
        if "_trial_" in r.method:
            continue
        lines.append(
            f"| `{r.dataset}` | `{r.method}` | "
            f"{r.neighbor_overlap_at_k:.4f} | "
            f"{r.trustworthiness_lite:.4f} | "
            f"{r.label_purity_at_k:.4f} | "
            f"{r.spearman_distance_rank_sample:.4f} |"
        )

    lines.append("")
    lines.append("## Scheduler Trace Summary")
    lines.append("")
    lines.append("| Dataset | Method | Final active channel | Final alpha dim | Final overlap@k |")
    lines.append("|---|---|---|---:|---:|")

    by_pair: Dict[Tuple[str, str], SchedulerStep] = {}
    for tr in trace_rows:
        by_pair[(tr.dataset, tr.method)] = tr

    for key in sorted(by_pair.keys()):
        tr = by_pair[key]
        lines.append(
            f"| `{tr.dataset}` | `{tr.method}` | `{tr.active_channel}` | "
            f"{tr.effective_alpha_dim:.3f} | {tr.neighbor_overlap_at_k:.4f} |"
        )

    lines.append("")
    lines.append("## Interpretation Rule")
    lines.append("")
    lines.append(
        "This probe tests whether D_M should be modeled as a meta-dimensional channel "
        "scheduler. The hardened random suite is included to prevent a single lucky "
        "random projection from being mistaken for a meaningful baseline."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_make_plots(
    out_dir: Path,
    summary: List[Dict[str, Any]],
    trace_rows: List[SchedulerStep],
) -> None:
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
    plt.title("D_M Probe 03 — Overall Ranking")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "dm_probe_03_overall_ranking.png", dpi=160)
    plt.close()

    scheduler = [tr for tr in trace_rows if tr.method == "dm7_soft_scheduler"]
    if scheduler:
        max_t = max(tr.timestep for tr in scheduler)
        alpha_names = [
            "alpha_linear",
            "alpha_local_order",
            "alpha_collapse",
            "alpha_mutation",
            "alpha_symmetry_boundary",
            "alpha_rank_spread",
            "alpha_composite",
        ]

        plt.figure(figsize=(12, 6))
        for name in alpha_names:
            vals = []
            for t in range(max_t + 1):
                step_vals = [getattr(tr, name) for tr in scheduler if tr.timestep == t]
                vals.append(float(np.mean(step_vals)) if step_vals else np.nan)
            plt.plot(range(max_t + 1), vals, marker="o", label=name.replace("alpha_", ""))
        plt.xlabel("timestep")
        plt.ylabel("mean alpha")
        plt.title("D_M Probe 03 — Soft Scheduler Mean Channel Weights")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "dm_probe_03_soft_scheduler_alpha_trace.png", dpi=160)
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — D_M Probe 03 linear scheduler with hardened random controls",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--channels",
        default=None,
        help="Probe 01 dm_probe_01_channels.npz. Defaults to latest in probes/analysis.",
    )
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--input-dim", type=int, default=DEFAULT_INPUT_DIM)
    p.add_argument("--output-dim", type=int, default=DEFAULT_OUTPUT_DIM)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument("--random-trials", type=int, default=DEFAULT_RANDOM_TRIALS)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed. If omitted, uses secrets.randbits(128).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output analysis directory. Defaults to probes/analysis/dm_probe_03_linear_scheduler_<timestamp>/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    master_seed = make_master_seed(args.seed)
    seed_source = "user" if args.seed is not None else "secrets.randbits(128)"
    rng = np.random.default_rng(master_seed)

    channels_path = Path(args.channels) if args.channels else find_latest_probe01_channels()
    channels = load_probe01_channels(channels_path)

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = ANALYSIS_DIR / f"dm_probe_03_linear_scheduler_{now_tag()}"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 88}")
    print("  GHOST ORACLE SUITE — D_M PROBE 03: LINEAR-ACTIVATED CHANNEL SCHEDULER")
    print(f"{'=' * 88}")
    print(f"  Channels      : {channels_path}")
    print(f"  Out dir       : {out_dir}")
    print(f"  Datasets      : {DATASETS}")
    print(f"  n             : {args.n}")
    print(f"  input_dim     : {args.input_dim}")
    print(f"  output_dim    : {args.output_dim}")
    print(f"  k             : {args.k}")
    print(f"  timesteps     : {args.timesteps}")
    print(f"  random_trials : {args.random_trials}")
    print(f"  master_seed   : {master_seed}")
    print(f"  seed_source   : {seed_source}")
    print("\n[SCOPE]")
    print("  Self-contained D_M only.")
    print("  Linear is internal channel 0.")
    print("  Hardened random-control suite enabled.")
    print("  No cross-operator interaction.")
    print("  Synthetic rehearsal only.")

    all_rows: List[EvalRow] = []
    all_random_trial_rows: List[EvalRow] = []
    all_trace: List[SchedulerStep] = []

    scheduler_methods = [
        "dm7_soft_scheduler",
        "dm7_hard_scheduler",
        "dm7_residual_scheduler",
    ]

    for dataset in DATASETS:
        print(f"\n[DATASET] {dataset}")

        x, labels = make_dataset(dataset, args.n, args.input_dim, rng)
        x = standardize(x)
        ref_nn = knn_indices(x, args.k)

        rows: List[EvalRow] = []

        z_identity = x
        z_pca = pca_project(x, args.output_dim)

        rows.append(eval_row(
            dataset,
            "identity_reference",
            x,
            z_identity,
            labels,
            ref_nn,
            args.k,
            rng,
        ))

        rows.append(eval_row(
            dataset,
            "pca_projection",
            x,
            z_pca,
            labels,
            ref_nn,
            args.k,
            rng,
        ))

        random_aggregate_rows, random_trial_rows = run_random_control_suite(
            dataset=dataset,
            x=x,
            labels=labels,
            ref_nn=ref_nn,
            output_dim=args.output_dim,
            k=args.k,
            rng=rng,
            random_trials=args.random_trials,
        )
        rows.extend(random_aggregate_rows)
        all_random_trial_rows.extend(random_trial_rows)

        for name in [
            "local_order_channel",
            "collapse_channel",
            "mutation_channel",
            "symmetry_boundary_channel",
            "rank_spread_channel",
            "composite_dm_channel",
        ]:
            z = dm_static_channel_project(x, channels[name], args.output_dim)
            rows.append(eval_row(
                dataset,
                f"dm_static_{name}",
                x,
                z,
                labels,
                ref_nn,
                args.k,
                rng,
            ))

        for method in scheduler_methods:
            z_sched, trace = scheduled_dm_project(
                dataset=dataset,
                x=x,
                labels=labels,
                channels=channels,
                output_dim=args.output_dim,
                k=args.k,
                timesteps=args.timesteps,
                rng=rng,
                method=method,
            )
            rows.append(eval_row(
                dataset,
                method,
                x,
                z_sched,
                labels,
                ref_nn,
                args.k,
                rng,
            ))
            all_trace.extend(trace)

        all_rows.extend(rows)

        ranked = sorted(
            [r for r in rows if r.method != "identity_reference"],
            key=lambda r: (
                r.neighbor_overlap_at_k,
                r.trustworthiness_lite,
                r.label_purity_at_k,
            ),
            reverse=True,
        )

        for r in ranked:
            print(
                f"  {r.method:<46} "
                f"overlap={r.neighbor_overlap_at_k:7.4f} "
                f"trust={r.trustworthiness_lite:7.4f} "
                f"purity={r.label_purity_at_k:7.4f} "
                f"rankcorr={r.spearman_distance_rank_sample:7.4f}"
            )

    summary = summarize_results(all_rows)

    write_csv(out_dir / "dm_probe_03_results.csv", [asdict(r) for r in all_rows])
    write_csv(out_dir / "dm_probe_03_random_trials.csv", [asdict(r) for r in all_random_trial_rows])
    write_csv(out_dir / "dm_probe_03_method_summary.csv", summary)
    write_csv(out_dir / "dm_probe_03_scheduler_trace.csv", [asdict(t) for t in all_trace])

    config = {
        "schema": "ghost_oracle.dm.probe_03_linear_scheduler.v2",
        "operator": "D_M",
        "probe": "dm_probe_03_linear_scheduler",
        "channels_path": str(channels_path),
        "datasets": DATASETS,
        "n": int(args.n),
        "input_dim": int(args.input_dim),
        "output_dim": int(args.output_dim),
        "k": int(args.k),
        "timesteps": int(args.timesteps),
        "random_trials": int(args.random_trials),
        "master_seed": int(master_seed),
        "seed_source": seed_source,
        "internal_channels": DM_CHANNEL_ORDER,
        "random_controls": [
            "random_gaussian_mean/best/worst",
            "random_orthogonal_mean/best/worst",
            "random_sparse_achlioptas_mean/best/worst",
            "random_shuffle_crop",
            "random_sign_flip_crop",
        ],
        "scope": {
            "self_consistent_operator_only": True,
            "uses_other_operators": False,
            "linear_channel_internal": True,
            "hardened_random_controls": True,
            "synthetic_only": True,
            "final_benchmark": False,
        },
    }

    with open(out_dir / "dm_probe_03_config.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(config), f, indent=2)

    write_report(
        path=out_dir / "dm_probe_03_report.md",
        channels_path=channels_path,
        config=config,
        rows=all_rows,
        summary=summary,
        trace_rows=all_trace,
    )

    maybe_make_plots(out_dir, summary, all_trace)

    print("\n[OVERALL RANKING]")
    for i, r in enumerate(summary, start=1):
        print(
            f"  {i:02d}. {r['method']:<46} "
            f"overlap={r['neighbor_overlap_at_k_mean']:7.4f} "
            f"trust={r['trustworthiness_lite_mean']:7.4f} "
            f"purity={r['label_purity_at_k_mean']:7.4f} "
            f"rankcorr={r['spearman_distance_rank_sample_mean']:7.4f}"
        )

    print(f"\n{'=' * 88}")
    print("  D_M PROBE 03 COMPLETE")
    print(f"{'=' * 88}")
    print(f"  Report        : {out_dir / 'dm_probe_03_report.md'}")
    print(f"  Results CSV   : {out_dir / 'dm_probe_03_results.csv'}")
    print(f"  Random trials : {out_dir / 'dm_probe_03_random_trials.csv'}")
    print(f"  Summary CSV   : {out_dir / 'dm_probe_03_method_summary.csv'}")
    print(f"  Trace CSV     : {out_dir / 'dm_probe_03_scheduler_trace.csv'}")
    print(f"  Config JSON   : {out_dir / 'dm_probe_03_config.json'}")
    print(f"{'=' * 88}\n")


if __name__ == "__main__":
    main()