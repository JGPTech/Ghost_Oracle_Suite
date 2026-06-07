#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GHOST ORACLE SUITE — D_M PROBE 02: SYNTHETIC MAPS
================================================================================

Purpose
-------
Probe 02 is the first self-contained D_M task rehearsal.

It tests whether the candidate D_M channels extracted in Probe 01 can guide
structure-preserving dimensional compression on synthetic datasets with known
neighborhood structure.

This probe does NOT make the final D_M benchmark claim.
This probe does NOT use G_M, S_M, T_S, F_M, or any other operator.
This probe does NOT compare against heavyweight best-practice systems yet.

It asks:

    Do any D_M channel projections preserve nearest-neighbor structure better
    than simple baselines on controlled synthetic maps?

Inputs
------
Probe 01 channel file:

    dm_probe_01_channels.npz

Expected arrays:

    channel_names
    <channel>_state_prob
    <channel>_bit_weight
    <channel>_bit_centered

Synthetic datasets
------------------
Generated internally:

    blobs
    rings
    swiss_roll_like
    s_curve_like
    sparse_binary

Methods
-------
Baselines:

    identity_reference
    random_projection
    pca_projection

D_M candidates:

    dm_<channel_name>

Metrics
-------
For each dataset/method:

    neighbor_overlap@k
    trustworthiness_lite
    label_purity@k
    spearman_distance_rank_sample
    output_dim
    compression_ratio

Usage
-----
    python ghost_oracle/D_M/probes/dm_probe_02_synthetic_maps.py

    python ghost_oracle/D_M/probes/dm_probe_02_synthetic_maps.py ^
        --channels ghost_oracle/D_M/probes/analysis/<probe01>/dm_probe_01_channels.npz

================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
DM_DIR = HERE.parent
DATA_DIR = DM_DIR / "data"
ANALYSIS_DIR = HERE / "analyze"


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

DEFAULT_SEED = 12345
DEFAULT_N = 5000
DEFAULT_INPUT_DIM = 64
DEFAULT_OUTPUT_DIM = 4
DEFAULT_K = 10

DATASETS = [
    "blobs",
    "rings",
    "swiss_roll_like",
    "s_curve_like",
    "sparse_binary",
]


# =============================================================================
# HELPERS
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
    return x


def load_npz_scalar(x: Any) -> Any:
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    return x


def find_latest_probe01_channels() -> Path:
    matches = sorted(
        ANALYSIS_DIR.glob("dm_probe_01_channel_extract_*/dm_probe_01_channels.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            "No Probe 01 channels file found.\n"
            "Pass --channels path/to/dm_probe_01_channels.npz explicitly."
        )
    return matches[0]


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - np.mean(x, axis=0, keepdims=True)) / np.maximum(np.std(x, axis=0, keepdims=True), eps)


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    """
    Squared Euclidean pairwise distances, stable enough for small synthetic probes.
    """
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
    if labels is None:
        return float("nan")
    labels = np.asarray(labels)
    vals = []
    for i in range(test_nn.shape[0]):
        vals.append(float(np.mean(labels[test_nn[i]] == labels[i])))
    return float(np.mean(vals))


def trustworthiness_lite(x_ref: np.ndarray, x_test: np.ndarray, k: int) -> float:
    """
    Lightweight trustworthiness approximation.

    Full trustworthiness penalizes unexpected neighbors by rank in original space.
    This implementation computes exact ranks from the original distance matrix
    for n around 1k, which is fine for this probe.
    """
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

    score = 1.0 - (2.0 / denom) * penalty
    return float(max(0.0, min(1.0, score)))


def spearman_distance_rank_sample(
    x_ref: np.ndarray,
    x_test: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int = 20000,
) -> float:
    """
    Spearman-like rank correlation over sampled pairwise distances.

    Implemented without scipy. Ties are ignored by using argsort ranks.
    """
    n = x_ref.shape[0]
    total_pairs = n * (n - 1) // 2
    n_pairs = int(min(n_pairs, total_pairs))

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
    comps = vt[:output_dim].T
    return x0 @ comps


def random_project(x: np.ndarray, output_dim: int, rng: np.random.Generator) -> np.ndarray:
    r = rng.normal(0.0, 1.0, size=(x.shape[1], output_dim))
    r /= np.maximum(np.linalg.norm(r, axis=0, keepdims=True), 1e-12)
    return x @ r


# =============================================================================
# SYNTHETIC DATASETS
# =============================================================================

def embed_low_to_high(
    z: np.ndarray,
    input_dim: int,
    rng: np.random.Generator,
    noise: float = 0.03,
) -> np.ndarray:
    """
    Embed low-dimensional synthetic coordinates into higher-dimensional space.
    """
    z = np.asarray(z, dtype=np.float64)
    z = standardize(z)

    proj = rng.normal(0.0, 1.0, size=(z.shape[1], input_dim))
    x = z @ proj
    x += noise * rng.normal(0.0, 1.0, size=x.shape)

    # Add mild nonlinear features to make PCA nontrivial.
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
    centers_2d = np.array([
        [-2.5, -2.0],
        [-2.0, 2.0],
        [2.0, -2.0],
        [2.5, 2.0],
        [0.0, 0.0],
    ], dtype=np.float64)

    labels = rng.integers(0, len(centers_2d), size=n)
    z = centers_2d[labels] + 0.45 * rng.normal(size=(n, 2))
    x = embed_low_to_high(z, input_dim, rng, noise=0.04)
    return x, labels


def make_rings(n: int, input_dim: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, 3, size=n)
    radii = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    theta = rng.uniform(0, 2 * np.pi, size=n)

    r = radii[labels] + 0.08 * rng.normal(size=n)
    z = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x = embed_low_to_high(z, input_dim, rng, noise=0.04)
    return x, labels


def make_swiss_roll_like(n: int, input_dim: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n)
    h = rng.uniform(-1.0, 1.0, size=n)

    z3 = np.stack([
        t * np.cos(t),
        h * 6.0,
        t * np.sin(t),
    ], axis=1)

    labels = np.digitize(t, np.quantile(t, [0.2, 0.4, 0.6, 0.8]))
    x = embed_low_to_high(z3, input_dim, rng, noise=0.035)
    return x, labels.astype(np.int32)


def make_s_curve_like(n: int, input_dim: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    t = rng.uniform(-1.5 * np.pi, 1.5 * np.pi, size=n)
    y = rng.uniform(-1.0, 1.0, size=n)

    z3 = np.stack([
        np.sin(t),
        y,
        np.sign(t) * (np.cos(t) - 1.0),
    ], axis=1)

    labels = np.digitize(t, np.quantile(t, [0.2, 0.4, 0.6, 0.8]))
    x = embed_low_to_high(z3, input_dim, rng, noise=0.035)
    return x, labels.astype(np.int32)


def make_sparse_binary(n: int, input_dim: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n_groups = 6
    labels = rng.integers(0, n_groups, size=n)

    prototypes = np.zeros((n_groups, input_dim), dtype=np.float64)
    block = max(2, input_dim // n_groups)

    for g in range(n_groups):
        start = g * block
        end = min(input_dim, start + block)
        prototypes[g, start:end] = 1.0

        # Add a shared overlapping signature.
        prototypes[g, g % input_dim] = 1.0
        prototypes[g, (g * 7 + 3) % input_dim] = 1.0

    x = prototypes[labels].copy()

    # Flip noise.
    flip = rng.random(size=x.shape) < 0.06
    x = np.where(flip, 1.0 - x, x)

    # Small continuous jitter.
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
# D_M CHANNEL PROJECTION
# =============================================================================

def load_channels(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    npz = np.load(path, allow_pickle=True)

    if "channel_names" not in npz.files:
        raise KeyError(f"{path} does not contain channel_names")

    names = [str(x) for x in np.asarray(npz["channel_names"])]

    channels: Dict[str, Dict[str, np.ndarray]] = {}
    for name in names:
        state_key = f"{name}_state_prob"
        bit_key = f"{name}_bit_weight"
        ctr_key = f"{name}_bit_centered"

        if state_key not in npz.files:
            continue

        state_prob = np.asarray(npz[state_key], dtype=np.float64)
        bit_weight = np.asarray(npz[bit_key], dtype=np.float64) if bit_key in npz.files else state_to_bit_weight(state_prob)
        bit_centered = np.asarray(npz[ctr_key], dtype=np.float64) if ctr_key in npz.files else bit_weight - np.mean(bit_weight)

        channels[name] = {
            "state_prob": state_prob,
            "bit_weight": bit_weight,
            "bit_centered": bit_centered,
        }

    if not channels:
        raise RuntimeError(f"No usable D_M channels found in {path}")

    return channels


def state_to_bit_weight(state_prob: np.ndarray) -> np.ndarray:
    p = np.asarray(state_prob, dtype=np.float64)
    p = p / max(float(np.sum(p)), 1e-12)

    out = np.zeros(4, dtype=np.float64)
    for state in range(16):
        for bit in range(4):
            if (state >> bit) & 1:
                out[bit] += p[state]
    return out


def dm_channel_project(
    x: np.ndarray,
    channel: Dict[str, np.ndarray],
    output_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Convert a Probe 01 D_M channel into a deterministic projection recipe.

    This is intentionally simple and transparent. It is not the final D_M method.

    Recipe:
        1. Split input dimensions into output_dim buckets.
        2. Use channel bit weights to shape bucket weighting.
        3. Use 16-state probabilities to add state-modulated harmonic mixing.
        4. Produce a low-dimensional representation.
    """
    x = standardize(x)

    n, d = x.shape
    output_dim = int(output_dim)

    bit_weight = np.asarray(channel["bit_weight"], dtype=np.float64)
    bit_centered = np.asarray(channel["bit_centered"], dtype=np.float64)
    state_prob = np.asarray(channel["state_prob"], dtype=np.float64)

    if bit_weight.size != 4:
        raise ValueError("D_M channel bit_weight must have size 4")
    if state_prob.size != 16:
        raise ValueError("D_M channel state_prob must have size 16")

    # Normalize bit weights into a stable positive scale.
    bw = bit_weight / max(float(np.mean(bit_weight)), 1e-12)
    bc = bit_centered / max(float(np.max(np.abs(bit_centered))), 1e-12)

    # Build deterministic channel projection matrix.
    w = np.zeros((d, output_dim), dtype=np.float64)

    idx = np.arange(d)
    for out_j in range(output_dim):
        bit_j = out_j % 4
        bucket_phase = 2.0 * np.pi * (out_j + 1) * (idx + 1) / max(1, d)

        # State modulation uses the 16-state profile repeated over input dims.
        state_ids = (idx + out_j) % 16
        state_mod = state_prob[state_ids]
        state_mod = state_mod / max(float(np.mean(state_mod)), 1e-12)

        # Local bucket assignment.
        bucket = ((idx + out_j) % output_dim == 0).astype(np.float64)

        # Smooth harmonic mixing so non-bucket dimensions still contribute weakly.
        harmonic = 0.15 * np.sin(bucket_phase) + 0.10 * np.cos(2.0 * bucket_phase)

        # Final column.
        col = (
            bucket * bw[bit_j]
            + harmonic * (1.0 + 0.25 * bc[bit_j])
            + 0.08 * state_mod
        )

        # Alternating contrast from centered bit channel.
        col *= (1.0 + 0.15 * bc[bit_j] * ((idx % 2) * 2.0 - 1.0))

        w[:, out_j] = col

    # Stabilize columns.
    w = w - np.mean(w, axis=0, keepdims=True)
    w = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)

    z = x @ w

    # Add channel-level nonlinear residual from state distribution.
    # Still deterministic, still self-contained.
    if output_dim >= 2:
        state_energy = float(np.sum(state_prob * state_prob))
        z[:, 0] += 0.05 * state_energy * np.sin(z[:, 1])
        z[:, 1] += 0.05 * state_energy * np.cos(z[:, 0])

    return standardize(z)


# =============================================================================
# EVALUATION
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

    x_ref_dim: int
    z_dim: int
    n: int
    k: int


def evaluate_method(
    dataset: str,
    method: str,
    x_ref: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    ref_nn: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> EvalRow:
    test_nn = knn_indices(z, k)

    overlap = neighbor_overlap(ref_nn, test_nn)
    trust = trustworthiness_lite(x_ref, z, k)
    purity = label_purity(labels, test_nn)
    spear = spearman_distance_rank_sample(x_ref, z, rng)

    return EvalRow(
        dataset=dataset,
        method=method,
        output_dim=int(z.shape[1]),
        input_dim=int(x_ref.shape[1]),
        compression_ratio=float(x_ref.shape[1] / max(1, z.shape[1])),

        neighbor_overlap_at_k=overlap,
        trustworthiness_lite=trust,
        label_purity_at_k=purity,
        spearman_distance_rank_sample=spear,

        x_ref_dim=int(x_ref.shape[1]),
        z_dim=int(z.shape[1]),
        n=int(x_ref.shape[0]),
        k=int(k),
    )


def run_dataset(
    name: str,
    channels: Dict[str, Dict[str, np.ndarray]],
    n: int,
    input_dim: int,
    output_dim: int,
    k: int,
    rng: np.random.Generator,
) -> Tuple[List[EvalRow], Dict[str, np.ndarray]]:
    x, labels = make_dataset(name, n, input_dim, rng)
    x = standardize(x)

    ref_nn = knn_indices(x, k)

    rows: List[EvalRow] = []
    embeddings_for_plot: Dict[str, np.ndarray] = {}

    # Identity reference is evaluated at original dimension. Its overlap is 1 by design,
    # but trustworthiness and rank are useful sanity checks.
    rows.append(evaluate_method(
        dataset=name,
        method="identity_reference",
        x_ref=x,
        z=x,
        labels=labels,
        ref_nn=ref_nn,
        k=k,
        rng=rng,
    ))

    z_rand = random_project(x, output_dim, rng)
    rows.append(evaluate_method(
        dataset=name,
        method="random_projection",
        x_ref=x,
        z=z_rand,
        labels=labels,
        ref_nn=ref_nn,
        k=k,
        rng=rng,
    ))
    embeddings_for_plot["random_projection"] = z_rand[:, :2] if z_rand.shape[1] >= 2 else z_rand

    z_pca = pca_project(x, output_dim)
    rows.append(evaluate_method(
        dataset=name,
        method="pca_projection",
        x_ref=x,
        z=z_pca,
        labels=labels,
        ref_nn=ref_nn,
        k=k,
        rng=rng,
    ))
    embeddings_for_plot["pca_projection"] = z_pca[:, :2] if z_pca.shape[1] >= 2 else z_pca

    for channel_name, channel in channels.items():
        z_dm = dm_channel_project(x, channel, output_dim, rng)
        method = f"dm_{channel_name}"
        rows.append(evaluate_method(
            dataset=name,
            method=method,
            x_ref=x,
            z=z_dm,
            labels=labels,
            ref_nn=ref_nn,
            k=k,
            rng=rng,
        ))
        embeddings_for_plot[method] = z_dm[:, :2] if z_dm.shape[1] >= 2 else z_dm

    return rows, embeddings_for_plot


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
    rows: List[EvalRow],
    summary: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# D_M Probe 02 — Synthetic Maps")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Self-contained `D_M` only.")
    lines.append("- Uses Probe 01 extracted D_M channels.")
    lines.append("- No cross-operator interaction.")
    lines.append("- No final benchmark claim.")
    lines.append("- Task rehearsal: structure-preserving dimensional compression for nearest-neighbor retrieval.")
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Channels: `{channels_path}`")
    lines.append(f"- Synthetic samples per dataset: `{config['n']}`")
    lines.append(f"- Input dim: `{config['input_dim']}`")
    lines.append(f"- Output dim: `{config['output_dim']}`")
    lines.append(f"- k: `{config['k']}`")
    lines.append("")
    lines.append("## Overall Method Ranking")
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
        lines.append(
            f"| `{r.dataset}` | `{r.method}` | "
            f"{r.neighbor_overlap_at_k:.4f} | "
            f"{r.trustworthiness_lite:.4f} | "
            f"{r.label_purity_at_k:.4f} | "
            f"{r.spearman_distance_rank_sample:.4f} |"
        )

    lines.append("")
    lines.append("## Interpretation Rule")
    lines.append("")
    lines.append(
        "This probe only checks whether a D_M channel is worth carrying into the next "
        "synthetic-control stage. A D_M channel beating random projection or PCA on "
        "one or more toy maps is not a final result; it is a direction signal."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_make_plots(out_dir: Path, rows: List[EvalRow], summary: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable; skipping plots: {e}")
        return

    # Overall ranking plot.
    methods = [r["method"] for r in summary]
    overlap = [r["neighbor_overlap_at_k_mean"] for r in summary]

    plt.figure(figsize=(12, max(5, 0.35 * len(methods))))
    y = np.arange(len(methods))
    plt.barh(y, overlap)
    plt.yticks(y, methods, fontsize=8)
    plt.xlabel("Mean neighbor overlap@k")
    plt.title("D_M Probe 02 — Overall Synthetic Map Ranking")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "dm_probe_02_overall_ranking.png", dpi=160)
    plt.close()

    # Dataset x method heatmap for neighbor overlap.
    datasets = sorted(set(r.dataset for r in rows))
    method_names = [m for m in methods if m != "identity_reference"]

    mat = np.zeros((len(datasets), len(method_names)), dtype=np.float64)
    lookup = {(r.dataset, r.method): r.neighbor_overlap_at_k for r in rows}
    for i, ds in enumerate(datasets):
        for j, method in enumerate(method_names):
            mat[i, j] = lookup.get((ds, method), np.nan)

    plt.figure(figsize=(max(10, 0.55 * len(method_names)), max(4, 0.55 * len(datasets))))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="neighbor overlap@k")
    plt.xticks(range(len(method_names)), method_names, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(len(datasets)), datasets, fontsize=9)
    plt.title("D_M Probe 02 — Dataset × Method Neighbor Preservation")
    plt.tight_layout()
    plt.savefig(out_dir / "dm_probe_02_dataset_method_heatmap.png", dpi=160)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — D_M Probe 02 synthetic maps",
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
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--out",
        default=None,
        help="Output analysis directory. Defaults to probes/analysis/dm_probe_02_synthetic_maps_<timestamp>/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    channels_path = Path(args.channels) if args.channels else find_latest_probe01_channels()
    if not channels_path.exists():
        raise FileNotFoundError(channels_path)

    rng = np.random.default_rng(args.seed)
    channels = load_channels(channels_path)

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = ANALYSIS_DIR / f"dm_probe_02_synthetic_maps_{now_tag()}"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 88}")
    print("  GHOST ORACLE SUITE — D_M PROBE 02: SYNTHETIC MAPS")
    print(f"{'=' * 88}")
    print(f"  Channels  : {channels_path}")
    print(f"  Out dir   : {out_dir}")
    print(f"  Datasets  : {DATASETS}")
    print(f"  n         : {args.n}")
    print(f"  input_dim : {args.input_dim}")
    print(f"  output_dim: {args.output_dim}")
    print(f"  k         : {args.k}")
    print(f"  seed      : {args.seed}")
    print("\n[SCOPE]")
    print("  Self-contained D_M only.")
    print("  Synthetic rehearsal only.")
    print("  No final benchmark claim.")

    all_rows: List[EvalRow] = []

    for dataset in DATASETS:
        print(f"\n[DATASET] {dataset}")
        rows, _ = run_dataset(
            name=dataset,
            channels=channels,
            n=args.n,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            k=args.k,
            rng=rng,
        )
        all_rows.extend(rows)

        # Print compact ranking for this dataset, excluding identity reference.
        ranked = sorted(
            [r for r in rows if r.method != "identity_reference"],
            key=lambda r: (r.neighbor_overlap_at_k, r.trustworthiness_lite, r.label_purity_at_k),
            reverse=True,
        )

        for r in ranked:
            print(
                f"  {r.method:<38} "
                f"overlap={r.neighbor_overlap_at_k:7.4f} "
                f"trust={r.trustworthiness_lite:7.4f} "
                f"purity={r.label_purity_at_k:7.4f} "
                f"rankcorr={r.spearman_distance_rank_sample:7.4f}"
            )

    summary = summarize_results(all_rows)

    row_dicts = [asdict(r) for r in all_rows]
    write_csv(out_dir / "dm_probe_02_results.csv", row_dicts)
    write_csv(out_dir / "dm_probe_02_method_summary.csv", summary)

    config = {
        "schema": "ghost_oracle.dm.probe_02_synthetic_maps.v1",
        "operator": "D_M",
        "probe": "dm_probe_02_synthetic_maps",
        "channels_path": str(channels_path),
        "datasets": DATASETS,
        "n": int(args.n),
        "input_dim": int(args.input_dim),
        "output_dim": int(args.output_dim),
        "k": int(args.k),
        "seed": int(args.seed),
        "scope": {
            "self_consistent_operator_only": True,
            "uses_other_operators": False,
            "synthetic_only": True,
            "final_benchmark": False,
        },
    }

    with open(out_dir / "dm_probe_02_config.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(config), f, indent=2)

    write_report(
        path=out_dir / "dm_probe_02_report.md",
        channels_path=channels_path,
        rows=all_rows,
        summary=summary,
        config=config,
    )

    maybe_make_plots(out_dir, all_rows, summary)

    print("\n[OVERALL RANKING]")
    for i, r in enumerate(summary, start=1):
        print(
            f"  {i:02d}. {r['method']:<38} "
            f"overlap={r['neighbor_overlap_at_k_mean']:7.4f} "
            f"trust={r['trustworthiness_lite_mean']:7.4f} "
            f"purity={r['label_purity_at_k_mean']:7.4f} "
            f"rankcorr={r['spearman_distance_rank_sample_mean']:7.4f}"
        )

    print(f"\n{'=' * 88}")
    print("  D_M PROBE 02 COMPLETE")
    print(f"{'=' * 88}")
    print(f"  Report      : {out_dir / 'dm_probe_02_report.md'}")
    print(f"  Results CSV : {out_dir / 'dm_probe_02_results.csv'}")
    print(f"  Summary CSV : {out_dir / 'dm_probe_02_method_summary.csv'}")
    print(f"  Config JSON : {out_dir / 'dm_probe_02_config.json'}")
    print(f"{'=' * 88}\n")


if __name__ == "__main__":
    main()