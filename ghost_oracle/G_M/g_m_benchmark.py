#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — G_M FINAL BENCHMARK (OPTIMIZED CAPSTONE)
==============================================================================
Combines the two capstone scripts into one optimized benchmark:

    final_benchmark_5way.py   -> the VERIFICATION story
                                 (CUBLAS / TIED / GEO / QPROJ / GPROJ on the
                                  gaussian-jitter same-dim attack, sig/spk
                                  Flash-Squelch certificate, agreement)
    auto_oracle.py            -> the SCALE + CONTROLS story
                                 (clustered semantic env up to M=1M, mixed
                                  softmax components, recall@K / MRR, and the
                                  load-bearing + separation negative controls)

ENGINE
  One compile of:
        kernels/ghost_kernel.cu   (Section 1 device helpers: clipped_log_pair,
                                    geometry_channel, projection_channel, EPS)
      + kernels/megakernels_2d.cu (the trig-hoisted 2D-tiled production kernels)

  Production path uses the 2D megakernels exclusively:
        geo_megakernel_2d   -> geometry channel        (probes / scale sweep)
        proj_megakernel_2d  -> projection channel       (calibration / probes)
        tied_megakernel_2d  -> geom + proj + agreement  (the 5-way verify)

  The inline FINAL_KERNELS block from final_benchmark_5way is gone. The 2D
  megakernels fold the phase-lift + cos INTO the global->shared transfer
  (lift_and_cos), so every kernel here consumes RAW embeddings. There is no
  host-side phase-lift anymore.

CALIBRATION (two objectives — the metric must match what the task does)
  The verify scores under a threshold; the sweep/probes are pure argmax
  retrieval that ignore the threshold. Calibrating both the same way is wrong,
  so there are two selectors:

  VERIFY  calibrate_candidates() -> Flash-Squelch certificate.
          Scores every (tile, mask, threshold) on GPU via proj_megakernel_2d:
              clean := median spk <= SPIKE_TOLERANCE
              rank  := clean first, then signal fraction descending
          The projection half of tied_megakernel_2d is bit-identical math to
          proj_megakernel_2d (both call projection_from_cos), so the calibrated
          threshold applies exactly to the verify. The metric used to calibrate
          is the metric the verify reports. Uses the single best component on
          the gaussian-jitter task -> one clean number vs cuBLAS.

  SWEEP / PROBES  calibrate_recall1() -> recall@1 (auto_oracle's selector).
          Ranks (tile, mask) by the SAME argmax retrieval the sweep runs, on a
          small clustered semantic env. A component is only selected if it
          retrieves, so the certificate's "attack-clean but scrambles semantic
          neighbourhood order" failure mode cannot pick a non-retriever. Sweep
          uses the top-K softmax mix; probes reuse the single best.

CONTROLS ARE FIRST CLASS
  --probe runs:
    A) load-bearing : real vs permuted vs uniformized shot counts on the
                      calibrated component. real >> perm, real >> uniform
                      means the physical bucket structure is doing the work
                      and projection is not silently collapsing to geometry.
    B) separation   : cosine vs geo vs proj across attack magnitudes; at
                      magnitude 0 cosine must stay competitive (the check
                      that geo/proj aren't winning only on the attack).

USAGE
    python G_M_final_benchmark.py                       # verify (default)
    python G_M_final_benchmark.py --N 4096
    python G_M_final_benchmark.py --skip-gpu            # QPU bases only
    python G_M_final_benchmark.py --sweep MEDIUM        # + semantic scale sweep
    python G_M_final_benchmark.py --probe               # + negative controls
    python G_M_final_benchmark.py --skip-verify --sweep ALL --probe
==============================================================================
"""

import argparse
import json
import math
import secrets
import sys
import time
import warnings
from pathlib import Path

import numpy as np

try:
    import cupy as cp
    _HAVE_CUPY = True
except Exception:
    cp = None
    _HAVE_CUPY = False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, *a, **k):
        return it

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG  (matches the tuned Probe 20 settings shared by both source scripts)
# =============================================================================
ANGLE_SCALE = 1.05
ALPHA_NORM = 0.9127

MATRIX_A_ORIG = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B_ORIG = np.array([1.00, 0.80, 0.40, 0.10])

# Verify (gaussian-jitter) defaults — mid-dimensional regime where the coherent
# same-dim spike engages the softmax/dot bottleneck. At high d / low magnitude
# the attack is weak relative to the embedding norm and cuBLAS catches up (top1
# ~100%, nothing separates). d=64 / magnitude=200 drives cuBLAS toward its ~50%
# single-dim-attack floor while bounded G_M geometry holds.
DEFAULT_D = 64
DEFAULT_N = 4096
DEFAULT_JITTER = 0.3
DEFAULT_ATTACK_FRACTION = 0.05
DEFAULT_MAGNITUDE = 200.0
DEFAULT_POWER = 256.0

# Semantic scale sweep defaults — auto_oracle regime.
DEFAULT_SWEEP_D = 1024
SEED = 42
MAX_VRAM_BYTES = 512 * 1024**2
TOP_K = [1, 5, 10]

# Kernel launch geometry (megakernels_2d.cu).
MK_TILE = 32

# Calibration (certificate) parameters.
CALIB_N = 256            # gaussian calibration env size (queries == keys)
CALIB_SEEDS = 3          # median over this many seeds
CALIB_THRESHOLDS = 50    # Flash-Squelch threshold sweep resolution
SPIKE_TOLERANCE = 0.05   # a candidate is "clean" if median spk <= this
CALIB_MIN_R1 = 0.50      # sweep/probes: a base's best component must clear this
                         # recall@1 on the calibration env to be "usable". Below
                         # it the base is excluded rather than handed over as a
                         # weight-1.00 component that retrieves nothing (e.g. a
                         # degenerate near-zero-angle tile).

# Semantic calibration env (used to pick sweep / probe components).
SEM_CAL_M = 4096
SEM_CAL_N = 128
SEM_CAL_CLUSTERS = 32
SEM_CAL_NOISE = 0.10
SEM_CAL_OUTLIER_FRAC = 0.05
SEM_CAL_OUTLIER_MAG = 50.0

# Probe B (separation spectrum) — two single-variable sweeps so magnitude and
# noise are never confounded.
#   B1 holds noise at the calibration value and walks magnitude  -> attack axis
#   B2 holds magnitude at a fixed attack level and walks noise    -> noise axis
PROBE_M = 250_000
PROBE_N = 1024
PROBE_OUTLIER_FRAC = 0.05
PROBE_FIXED_NOISE = SEM_CAL_NOISE          # B1: in-distribution for the component
PROBE_MAG_SWEEP = [0.0, 5.0, 20.0, 40.0, 60.0, 100.0]
PROBE_FIXED_MAG = 60.0                      # B2: a solid attack, present throughout
PROBE_NOISE_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

MASK_CONFIGS = [
    ("M1", "Baseline (all 9)",               []),
    ("M2", "CUDA accident (drop 4-8)",       [4, 5, 6, 7, 8]),
    ("M3", "Anti-pillars (0,2)(2,0)",        [2, 6]),
    ("M4", "Drop (0,1)(1,0)",                [1, 3]),
    ("M5", "Drop (1,2)(2,1)",                [5, 7]),
    ("M6", "Drop pillars (0,0)(2,2)",        [0, 8]),
    ("M7", "Pure core (0,0|0,1|1,0|1,1)",    [2, 5, 6, 7, 8]),
    ("M8", "Mirror core (1,1|1,2|2,1|2,2)",  [0, 1, 2, 3, 6]),
]

SWEEPS = {
    "SMALL":  {"M":    50_000, "N": 1024, "NOISE": 0.08, "OUTLIER_FRAC": 0.01, "OUTLIER_MAG":  40.0},
    "MEDIUM": {"M":   250_000, "N": 1024, "NOISE": 0.12, "OUTLIER_FRAC": 0.03, "OUTLIER_MAG":  60.0},
    "LARGE":  {"M": 1_000_000, "N": 1024, "NOISE": 0.18, "OUTLIER_FRAC": 0.05, "OUTLIER_MAG": 100.0},
}

HERE = Path(__file__).resolve().parent
DEFAULT_KERNEL = HERE / "kernels" / "ghost_kernel.cu"
DEFAULT_MK_KERNEL = HERE / "kernels" / "megakernels_2d.cu"
DEFAULT_DATA_DIR = HERE / "data"


def data_to_angles(d, scale=ANGLE_SCALE):
    return (d / np.max(np.abs(d))) * (math.pi / 2) * scale


ORIG_A = data_to_angles(MATRIX_A_ORIG)
ORIG_B = data_to_angles(MATRIX_B_ORIG)


# =============================================================================
# BASE / BUCKET HELPERS
# =============================================================================
def pairs_for_num_tiles(n, matrix_size=4):
    return [(r, c) for r in range(matrix_size) for c in range(matrix_size)][:n]


def get_bitmask(dropped):
    bits = 0b111111111
    for b in dropped:
        bits &= ~(1 << b)
    return bits


def zero_buckets_counts(counts18, dropped):
    arr = counts18.copy()
    for b in dropped:
        arr[b * 2] = 0
        arr[b * 2 + 1] = 0
    return arr


def load_base(path):
    d = np.load(path)
    n = int(d["num_tiles"])
    return ({t: d[f"ctrl_tile{t}"] for t in range(n)},
            {t: d[f"ghost_tile{t}"] for t in range(n)}, n)


def build_bucket_counts(ctrl_dict, ghost_dict, n):
    counts = np.zeros((n, 3, 3, 2), dtype=np.int32)
    for t in range(n):
        g = ghost_dict[t]
        a_sum = g[:, 0].astype(np.int32) + g[:, 1].astype(np.int32)
        b_sum = g[:, 2].astype(np.int32) + g[:, 3].astype(np.int32)
        c = ctrl_dict[t]
        for a_b in range(3):
            for b_b in range(3):
                m = (a_sum == a_b) & (b_sum == b_b)
                counts[t, a_b, b_b, 0] = int(((c == 0) & m).sum())
                counts[t, a_b, b_b, 1] = int(((c == 1) & m).sum())
    return counts


# =============================================================================
# EMBEDDING ENVIRONMENTS
# =============================================================================
def make_attacked_jittered_embeddings(N, M, d, jitter, frac, mag, seed):
    """Gaussian keys, query i = key i + jitter, frac of keys spiked same-dim."""
    rng = np.random.default_rng(seed)
    X_K = rng.normal(size=(M, d)).astype(np.float32)
    X_Q = X_K + (jitter * rng.normal(size=(N, d))).astype(np.float32)
    n_out = max(1, int(M * frac))
    out_idx = rng.choice(M, size=n_out, replace=False)
    k_bad = int(rng.integers(0, d))
    X_K[out_idx, k_bad] = mag
    spike_mask = np.zeros(M, dtype=np.int32)
    spike_mask[out_idx] = 1
    return X_Q, X_K, spike_mask, out_idx


def generate_semantic_environment(M, N, dim, num_clusters, noise,
                                  outlier_frac, outlier_mag, seed=42):
    """Clustered semantic embeddings with same-dim coherent outliers.
       Returns X_Q, X_K, query_indices (truth), outlier_idx."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(num_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    cluster_ids = rng.integers(0, num_clusters, size=M)
    X_K = centers[cluster_ids]
    X_K += 0.25 * rng.normal(size=(M, dim)).astype(np.float32)
    query_indices = rng.choice(M, size=N, replace=False)
    X_Q = X_K[query_indices].copy()
    X_Q += noise * rng.normal(size=(N, dim)).astype(np.float32)
    X_Q /= np.linalg.norm(X_Q, axis=1, keepdims=True)
    n_outliers = max(1, int(M * outlier_frac))
    outlier_idx = rng.choice(M, size=n_outliers, replace=False)
    bad_dims = rng.integers(0, dim, size=n_outliers)
    X_K /= np.linalg.norm(X_K, axis=1, keepdims=True)
    for i in range(n_outliers):
        X_K[outlier_idx[i], bad_dims[i]] = outlier_mag
    return (X_Q.astype(np.float32), X_K.astype(np.float32),
            query_indices.astype(np.int64), outlier_idx.astype(np.int64))


# =============================================================================
# KERNEL WRAPPERS (2D megakernels — RAW embeddings in)
# =============================================================================
def run_geo_mega(k_geo, dQ_flat, dK_flat, N, M, d):
    score = cp.empty((N, M), dtype=cp.float32)
    grid = ((M + MK_TILE - 1) // MK_TILE, (N + MK_TILE - 1) // MK_TILE, 1)
    block = (MK_TILE, MK_TILE, 1)
    k_geo(grid, block,
          (dQ_flat, dK_flat, score,
           np.int32(N), np.int32(M), np.int32(d), np.float32(ALPHA_NORM)))
    cp.cuda.Device().synchronize()
    return score


def run_proj_mega(k_proj, dQ_flat, dK_flat, d_counts, orig_a, orig_b,
                  N, M, d, mask_bits):
    score = cp.empty((N, M), dtype=cp.float32)
    grid = ((M + MK_TILE - 1) // MK_TILE, (N + MK_TILE - 1) // MK_TILE, 1)
    block = (MK_TILE, MK_TILE, 1)
    k_proj(grid, block,
           (d_counts, np.float32(orig_a), np.float32(orig_b),
            dQ_flat, dK_flat, score,
            np.int32(N), np.int32(M), np.int32(d),
            np.float32(ALPHA_NORM), np.int32(mask_bits)))
    cp.cuda.Device().synchronize()
    return score


def run_tied_mega(k_tied, dQ_flat, dK_flat, d_counts, orig_a, orig_b,
                  N, M, d, mask_bits):
    """tied_megakernel_2d: geom (N,M), proj (N,M), per-query agreement (N,).
       agree_partial is (N, num_m_blocks) and reduced host-side; it MUST be
       zeroed because the kernel atomicAdds into it."""
    geom = cp.empty((N, M), dtype=cp.float32)
    proj = cp.empty((N, M), dtype=cp.float32)
    num_m_blocks = (M + MK_TILE - 1) // MK_TILE
    agree_partial = cp.zeros((N, num_m_blocks), dtype=cp.float32)
    grid = (num_m_blocks, (N + MK_TILE - 1) // MK_TILE, 1)
    block = (MK_TILE, MK_TILE, 1)
    k_tied(grid, block,
           (d_counts, np.float32(orig_a), np.float32(orig_b),
            dQ_flat, dK_flat, geom, proj, agree_partial,
            np.int32(N), np.int32(M), np.int32(d),
            np.float32(ALPHA_NORM), np.int32(mask_bits)))
    cp.cuda.Device().synchronize()
    agreement = agree_partial.sum(axis=1) / float(M)
    return geom, proj, agreement


# =============================================================================
# SCORING (Flash-Squelch) — shared between calibration and verify
# =============================================================================
def score_path(scores_gpu, true_idx, spike_mask, threshold, power):
    """top1, signal fraction, spike fraction for an (N, M) score matrix."""
    argmax = cp.argmax(scores_gpu, axis=1).get()
    top1 = float(np.mean(argmax == true_idx))

    shifted = cp.maximum(scores_gpu - threshold, 0.0)
    row_max = shifted.max(axis=1, keepdims=True)
    row_max = cp.where(row_max == 0.0, 1.0, row_max)
    w = cp.power(shifted / row_max, power)
    denom = w.sum(axis=1, keepdims=True)
    denom_safe = cp.where(denom > 1e-25, denom, 1.0)
    w_norm = cp.where(denom > 1e-25, w / denom_safe, 0.0)
    N = scores_gpu.shape[0]
    sig = float(w_norm[cp.arange(N), cp.asarray(true_idx)].mean())
    spk_mask = cp.asarray(spike_mask).astype(cp.float32)
    spk = float((w_norm * spk_mask[None, :]).sum(axis=1).mean())
    return top1, sig, spk


def _sig_spk(S, true_idx_dev, spike_f_dev, thr, power):
    """Lean sig/spk for the calibration threshold sweep (device-resident
    truth/spike). Identical Flash-Squelch math to score_path, minus argmax."""
    shifted = cp.maximum(S - thr, 0.0)
    row_max = shifted.max(axis=1, keepdims=True)
    row_max = cp.where(row_max == 0.0, 1.0, row_max)
    w = cp.power(shifted / row_max, power)
    denom = w.sum(axis=1, keepdims=True)
    denom_safe = cp.where(denom > 1e-25, denom, 1.0)
    w_norm = cp.where(denom > 1e-25, w / denom_safe, 0.0)
    n = S.shape[0]
    sig = float(w_norm[cp.arange(n), true_idx_dev].mean())
    spk = float((w_norm * spike_f_dev[None, :]).sum(axis=1).mean())
    return sig, spk


# =============================================================================
# RETRIEVAL WRAPPERS (top-K, chunked) — for the scale sweep and probes
# =============================================================================
def cosine_retrieval(X_Q, X_K, topk, max_vram_bytes):
    N, d = X_Q.shape
    M = X_K.shape[0]
    XQ = X_Q / (np.linalg.norm(X_Q, axis=1, keepdims=True) + 1e-8)
    XK = X_K / (np.linalg.norm(X_K, axis=1, keepdims=True) + 1e-8)
    d_K = cp.asarray(XK)
    chunk = max(1, max_vram_bytes // (M * 4))
    out = cp.empty((N, topk), dtype=cp.int32)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        d_q = cp.asarray(XQ[i:end])
        s = d_q @ d_K.T
        idx = cp.argpartition(-s, kth=topk - 1, axis=1)[:, :topk]
        partial = cp.take_along_axis(s, idx, axis=1)
        order = cp.argsort(-partial, axis=1)
        out[i:end] = cp.take_along_axis(idx, order, axis=1).astype(cp.int32)
    cp.cuda.Device().synchronize()
    return out, (chunk * M * 4) / (1024**2)


def geo_retrieval_topk(k_geo, X_Q, X_K, topk, max_vram_bytes):
    N, d = X_Q.shape
    M = X_K.shape[0]
    d_K = cp.asarray(X_K.reshape(-1))
    chunk = max(1, max_vram_bytes // (M * 4))
    out = cp.empty((N, topk), dtype=cp.int32)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        cur = end - i
        d_Q = cp.asarray(X_Q[i:end].reshape(-1))
        scores = run_geo_mega(k_geo, d_Q, d_K, cur, M, d)
        idx = cp.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
        out[i:end] = cp.take_along_axis(
            idx, cp.argsort(-cp.take_along_axis(scores, idx, axis=1), axis=1),
            axis=1).astype(cp.int32)
    cp.cuda.Device().synchronize()
    return out, (chunk * M * 4) / (1024**2)


def proj_retrieval_topk_single(k_proj, X_Q, X_K, counts18, orig_a, orig_b,
                               mask_bits, topk, max_vram_bytes, silent=False):
    N, d = X_Q.shape
    M = X_K.shape[0]
    d_K = cp.asarray(X_K.reshape(-1))
    d_counts = cp.asarray(counts18)
    chunk = max(1, max_vram_bytes // (M * 4))
    out = cp.empty((N, topk), dtype=cp.int32)
    iterable = range(0, N, chunk) if silent else tqdm(range(0, N, chunk),
                                                      desc="    qproj", leave=False)
    for i in iterable:
        end = min(i + chunk, N)
        cur = end - i
        d_Q = cp.asarray(X_Q[i:end].reshape(-1))
        scores = run_proj_mega(k_proj, d_Q, d_K, d_counts, orig_a, orig_b,
                               cur, M, d, mask_bits)
        idx = cp.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
        out[i:end] = cp.take_along_axis(
            idx, cp.argsort(-cp.take_along_axis(scores, idx, axis=1), axis=1),
            axis=1).astype(cp.int32)
    cp.cuda.Device().synchronize()
    return out, (chunk * M * 4) / (1024**2)


def proj_retrieval_topk_mixed(k_proj, X_Q, X_K, components, topk, max_vram_bytes):
    N, d = X_Q.shape
    M = X_K.shape[0]
    d_K = cp.asarray(X_K.reshape(-1))
    comp_state = [{"d_counts": cp.asarray(c["counts18"]), "orig_a": c["orig_a"],
                   "orig_b": c["orig_b"], "mask_bits": c["mask_bits"],
                   "weight": c["weight"]} for c in components]
    chunk = max(1, max_vram_bytes // (M * 4 * 2))
    out = cp.empty((N, topk), dtype=cp.int32)
    for i in tqdm(range(0, N, chunk), desc="    qproj-mixed", leave=False):
        end = min(i + chunk, N)
        cur = end - i
        d_Q = cp.asarray(X_Q[i:end].reshape(-1))
        combined = cp.zeros((cur, M), dtype=cp.float32)
        for cs in comp_state:
            temp = run_proj_mega(k_proj, d_Q, d_K, cs["d_counts"],
                                 cs["orig_a"], cs["orig_b"], cur, M, d,
                                 cs["mask_bits"])
            combined += cs["weight"] * temp
        idx = cp.argpartition(-combined, kth=topk - 1, axis=1)[:, :topk]
        out[i:end] = cp.take_along_axis(
            idx, cp.argsort(-cp.take_along_axis(combined, idx, axis=1), axis=1),
            axis=1).astype(cp.int32)
    cp.cuda.Device().synchronize()
    return out, (chunk * M * 8) / (1024**2)


def recall_at_k(topk_indices, truth_indices, k):
    top = topk_indices[:, :k].get() if hasattr(topk_indices, "get") else topk_indices[:, :k]
    return float(np.mean([int(truth_indices[i] in top[i]) for i in range(top.shape[0])]))


def mean_reciprocal_rank(topk_indices, truth_indices):
    top = topk_indices.get() if hasattr(topk_indices, "get") else topk_indices
    rr = []
    for row, gt in zip(top, truth_indices):
        found = False
        for rank, idx in enumerate(row, start=1):
            if idx == gt:
                rr.append(1.0 / rank)
                found = True
                break
        if not found:
            rr.append(0.0)
    return float(np.mean(rr))


# =============================================================================
# MISC
# =============================================================================
def clear_gpu():
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def time_block(fn, warmup=2, reps=5):
    for _ in range(warmup):
        fn()
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(reps):
        out = fn()
    cp.cuda.Device().synchronize()
    return out, (time.perf_counter() - t0) / reps


def section(t, w=130):
    print()
    print("=" * w)
    print(f"  {t}")
    print("=" * w)


def scramble_counts18(counts18, mode, seed=7):
    """Negative controls preserving total shots but destroying structure:
        'perm'    redistribute shots uniformly at random across live buckets
        'uniform' every live bucket exactly equal (strongest control)
    Masked (zeroed) buckets stay zero so mask geometry is held fixed."""
    rng = np.random.default_rng(seed)
    arr = counts18.astype(np.int64).copy()
    live = np.where(arr > 0)[0]
    if live.size == 0:
        return counts18
    total = int(arr.sum())
    out = np.zeros_like(arr)
    if mode == "uniform":
        out[live] = max(total // live.size, 1)
    elif mode == "perm":
        out[live] = rng.multinomial(total, np.full(live.size, 1.0 / live.size))
    else:
        return counts18
    return out.astype(counts18.dtype)


# =============================================================================
# CALIBRATION (one certificate-scored routine; the env decides the task)
# =============================================================================
def calibrate_candidates(k_proj, counts_per_tile, num_tiles, d, env_builder,
                         n_seeds, master_seed, power,
                         n_thresholds=CALIB_THRESHOLDS,
                         spike_tol=SPIKE_TOLERANCE):
    """Score every (tile, mask, threshold) on GPU via proj_megakernel_2d and
    return candidate dicts sorted best-first by the Flash-Squelch certificate.

    env_builder(seed) -> (X_Q, X_K, true_idx[np], spike_f[np (M,)], N, M)
    Each candidate carries the masked counts18 it was scored with, so the
    verify / sweep / probes feed proj exactly what calibration measured.
    """
    pairs = pairs_for_num_tiles(num_tiles)
    thresholds = np.linspace(0.0, 0.90, n_thresholds)

    rng = np.random.default_rng(master_seed)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]
    envs = []
    for s in seeds:
        X_Q, X_K, true_idx, spike_f, N, M = env_builder(s)
        envs.append((cp.asarray(X_Q.reshape(-1)), cp.asarray(X_K.reshape(-1)),
                     cp.asarray(true_idx.astype(np.int64)),
                     cp.asarray(spike_f.astype(np.float32)), N, M))

    cands = []
    for t in range(num_tiles):
        r, c = pairs[t]
        oa, ob = float(ORIG_A[r]), float(ORIG_B[c])
        base18 = counts_per_tile[t].reshape(-1)
        for tag, name, drops in MASK_CONFIGS:
            mask_bits = get_bitmask(drops)
            c_mod = zero_buckets_counts(base18, drops)
            d_counts = cp.asarray(c_mod)

            sigs = np.zeros((n_seeds, n_thresholds))
            spks = np.zeros((n_seeds, n_thresholds))
            for si, (dQ, dK, dTrue, dSpk, N, M) in enumerate(envs):
                S = run_proj_mega(k_proj, dQ, dK, d_counts, oa, ob, N, M, d, mask_bits)
                for ti, thr in enumerate(thresholds):
                    sg, sp = _sig_spk(S, dTrue, dSpk, float(thr), power)
                    sigs[si, ti] = sg
                    spks[si, ti] = sp

            med_sig = np.median(sigs, axis=0)
            med_spk = np.median(spks, axis=0)
            clean = med_spk <= spike_tol
            if clean.any():
                idx = int(np.argmax(np.where(clean, med_sig, -1.0)))
            else:
                idx = int(np.argmax(med_sig))

            cands.append({
                "tile_idx": t, "tile_rc": [r, c],
                "orig_a": oa, "orig_b": ob,
                "mask_tag": tag, "mask_name": name,
                "mask_drops": list(drops), "mask_bits": mask_bits,
                "thr": float(thresholds[idx]),
                "sig": float(med_sig[idx]),
                "score": float(med_sig[idx]),
                "spk": float(med_spk[idx]),
                "clean": bool(med_spk[idx] <= spike_tol),
                "counts18": c_mod,
            })

    # clean candidates rank above dirty; within each, signal fraction descending.
    cands.sort(key=lambda x: (x["clean"], x["sig"]), reverse=True)
    return cands


def calibrate_recall1(k_proj, counts_per_tile, num_tiles, d, env_builder,
                      n_seeds, master_seed):
    """Rank (tile, mask) by recall@1 — the SAME argmax retrieval the sweep and
    probes actually run (auto_oracle's selector). A component is only selected
    if it retrieves, so the certificate's "attack-clean but scrambles semantic
    neighbourhood order" failure mode cannot pick a non-retriever.

    env_builder(seed) -> (X_Q, X_K, gt). Candidates sorted best-first by median
    recall@1; thr is unused by argmax retrieval and stored as 0.0.
    """
    pairs = pairs_for_num_tiles(num_tiles)
    rng = np.random.default_rng(master_seed)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]
    envs = []
    for s in seeds:
        X_Q, X_K, gt = env_builder(s)
        envs.append((cp.asarray(X_Q.reshape(-1)), cp.asarray(X_K.reshape(-1)),
                     gt.astype(np.int64), X_Q.shape[0], X_K.shape[0]))

    cands = []
    for t in range(num_tiles):
        r, c = pairs[t]
        oa, ob = float(ORIG_A[r]), float(ORIG_B[c])
        base18 = counts_per_tile[t].reshape(-1)
        for tag, name, drops in MASK_CONFIGS:
            mask_bits = get_bitmask(drops)
            c_mod = zero_buckets_counts(base18, drops)
            d_counts = cp.asarray(c_mod)
            r1s = []
            for dQ, dK, gt, N, M in envs:
                S = run_proj_mega(k_proj, dQ, dK, d_counts, oa, ob, N, M, d, mask_bits)
                pred = cp.argmax(S, axis=1).get()
                r1s.append(float(np.mean(pred == gt)))
            r1 = float(np.median(r1s))
            cands.append({
                "tile_idx": t, "tile_rc": [r, c],
                "orig_a": oa, "orig_b": ob,
                "mask_tag": tag, "mask_name": name,
                "mask_drops": list(drops), "mask_bits": mask_bits,
                "thr": 0.0, "r1": r1, "score": r1,
                "counts18": c_mod,
            })
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands


def cal_to_component(c):
    return {"tile_idx": c["tile_idx"], "mask_tag": c["mask_tag"],
            "mask_bits": c["mask_bits"], "orig_a": c["orig_a"],
            "orig_b": c["orig_b"], "counts18": c["counts18"], "weight": 1.0}


def build_mixed_components(cands, mixed_k, mixed_temp):
    ranked = cands[:max(1, mixed_k)]
    score = np.array([c["score"] for c in ranked])
    logits = mixed_temp * score
    w = np.exp(logits - logits.max())
    w = w / w.sum()
    comps = []
    for c, wi in zip(ranked, w):
        comp = cal_to_component(c)
        comp["weight"] = float(wi)
        comps.append(comp)
    return comps


def cal_json(c):
    """JSON-safe view of a candidate (drops the numpy counts18 array)."""
    return {k: v for k, v in c.items() if k != "counts18"}


# =============================================================================
# STAGE 1 — FIVE-WAY VERIFICATION (gaussian-jitter, single best component)
# =============================================================================
def run_verify(args, k_proj, k_tied, bases, out_path, master_seed):
    section("FINAL BENCHMARK — FIVE-WAY VERIFICATION (megakernel engine)")
    print(f"  N (queries=keys) : {args.N}")
    print(f"  d (per-dim)      : {args.d}")
    print(f"  attack           : jitter={args.jitter}  frac={args.attack_fraction}  magnitude={args.magnitude}")
    print(f"  power (squelch)  : {args.power}")
    print(f"  master seed      : {master_seed}")
    print(f"  base files       : {len(bases)}")

    # One shared embedding set so every row is comparable.
    X_Q, X_K, spike_mask, _ = make_attacked_jittered_embeddings(
        args.N, args.N, args.d, args.jitter, args.attack_fraction,
        args.magnitude, master_seed)
    true_idx = np.arange(args.N, dtype=np.int64)
    dQ = cp.asarray(X_Q.reshape(-1))
    dK = cp.asarray(X_K.reshape(-1))

    # cuBLAS baseline (raw embeddings, base-independent, runs once).
    section("CUBLAS BASELINE (base-independent)")
    d_Q = cp.asarray(X_Q)
    d_K = cp.asarray(X_K)

    def fn_cublas():
        s = (d_Q @ d_K.T) * (1.0 / math.sqrt(args.d))
        cp.cuda.Device().synchronize()
        return s

    scores_cublas, t_cublas = time_block(fn_cublas)
    cublas_top1, cublas_sig, cublas_spk = score_path(
        scores_cublas, true_idx, spike_mask, threshold=0.0, power=args.power)
    print(f"  cuBLAS dot-product:  top1={cublas_top1:.1%}  "
          f"sig={cublas_sig:.1%}  spk={cublas_spk:.6f}  "
          f"time={t_cublas * 1000:.2f} ms")

    # Gaussian calibration env (matches the verify task).
    def gaussian_env(seed):
        Xq, Xk, sm, _ = make_attacked_jittered_embeddings(
            CALIB_N, CALIB_N, args.d, args.jitter, args.attack_fraction,
            args.magnitude, seed)
        return Xq, Xk, np.arange(CALIB_N, dtype=np.int64), sm.astype(np.float32), CALIB_N, CALIB_N

    section("PER-BASE: TIED, GEO, QPROJ/GPROJ")
    hdr = (f"  {'#':>2}  {'type':>4}  {'file':<46}  {'path':<8}  "
           f"{'tile':>4}  {'mask':<4}  {'thr':>6}  "
           f"{'top1':>7}  {'sig':>7}  {'spk':>8}  {'time_ms':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    summary = {
        "config": {
            "N": args.N, "d": args.d, "jitter": args.jitter,
            "attack_fraction": args.attack_fraction,
            "magnitude": args.magnitude, "power": args.power,
            "master_seed": master_seed,
        },
        "cublas": {"top1": cublas_top1, "sig": cublas_sig, "spk": cublas_spk,
                   "time_ms": t_cublas * 1000},
        "entries": [],
    }

    for idx, (path_obj, type_label) in enumerate(bases):
        filename = path_obj.name
        ctrl, ghost, n = load_base(str(path_obj))
        counts = build_bucket_counts(ctrl, ghost, n)

        clear_gpu()
        cands = calibrate_candidates(k_proj, counts, n, args.d, gaussian_env,
                                     CALIB_SEEDS, master_seed, args.power)
        clear_gpu()
        cal = cands[0]
        d_counts = cp.asarray(cal["counts18"])
        mask_bits = int(cal["mask_bits"])
        thr = float(cal["thr"])

        # One tied launch -> geom + proj + agreement.
        def fn_tied():
            return run_tied_mega(k_tied, dQ, dK, d_counts, cal["orig_a"],
                                 cal["orig_b"], args.N, args.N, args.d, mask_bits)
        (geom_gpu, proj_gpu, agree_gpu), t_kernel = time_block(fn_tied)

        (geo_top1, geo_sig, geo_spk), t_score_geo = time_block(
            lambda: score_path(geom_gpu, true_idx, spike_mask, 0.0, args.power))
        (proj_top1, proj_sig, proj_spk), t_score_proj = time_block(
            lambda: score_path(proj_gpu, true_idx, spike_mask, thr, args.power))

        # TIED argmax == geometry argmax (same numbers); TIED also pays for the
        # projection work and reports the agreement certificate.
        tied_top1, tied_sig, tied_spk = geo_top1, geo_sig, geo_spk
        agreement = float(agree_gpu.mean().get())

        t_tied = (t_kernel + t_score_geo) * 1000
        t_geo = (t_kernel + t_score_geo) * 1000
        t_proj = (t_kernel + t_score_proj) * 1000
        path_label = "QPROJ" if type_label == "QPU" else "GPROJ"

        print(f"  {idx:>2}  {type_label:>4}  {filename[:46]:<46}  "
              f"{'TIED':<8}  {cal['tile_idx']:>4}  {cal['mask_tag']:<4}  "
              f"{thr:>6.3f}  {tied_top1:>6.1%}  {tied_sig:>6.1%}  "
              f"{tied_spk:>8.4f}  {t_tied:>8.2f}")
        print(f"  {'':>2}  {'':>4}  {'':<46}  "
              f"{'GEO':<8}  {cal['tile_idx']:>4}  {'--':<4}  "
              f"{'--':>6}  {geo_top1:>6.1%}  {geo_sig:>6.1%}  "
              f"{geo_spk:>8.4f}  {t_geo:>8.2f}")
        print(f"  {'':>2}  {'':>4}  {'':<46}  "
              f"{path_label:<8}  {cal['tile_idx']:>4}  {cal['mask_tag']:<4}  "
              f"{thr:>6.3f}  {proj_top1:>6.1%}  {proj_sig:>6.1%}  "
              f"{proj_spk:>8.4f}  {t_proj:>8.2f}")
        print(f"  {'':>2}  {'':>4}  {'':<46}  "
              f"{'agreement':<8} = {agreement:.4f}   (mean |geom - proj| per query)")
        print()

        summary["entries"].append({
            "file": filename, "type": type_label,
            "calibration": cal_json(cal),
            "agreement": agreement,
            "kernel_time_ms": t_kernel * 1000,
            "tied": {"top1": tied_top1, "sig": tied_sig, "spk": tied_spk, "time_ms": t_tied},
            "geo": {"top1": geo_top1, "sig": geo_sig, "spk": geo_spk, "time_ms": t_geo},
            "proj": {"top1": proj_top1, "sig": proj_sig, "spk": proj_spk,
                     "time_ms": t_proj, "path_label": path_label},
        })

    section("FIVE-WAY SUMMARY")
    print(f"  CUBLAS              top1={cublas_top1:.1%}  "
          f"sig={cublas_sig:.1%}  spk={cublas_spk:.6f}  t={t_cublas * 1000:.2f} ms")
    geo_top1s = [e["geo"]["top1"] for e in summary["entries"]]
    if geo_top1s:
        print(f"  GEO (mean across bases)   top1={np.mean(geo_top1s):.1%}")
    for tag, label in [("QPU", "QPROJ"), ("GPU", "GPROJ")]:
        sel = [e for e in summary["entries"] if e["type"] == tag]
        if sel:
            print(f"  {label} (mean across {tag} bases)  "
                  f"top1={np.mean([e['proj']['top1'] for e in sel]):.1%}  "
                  f"sig={np.mean([e['proj']['sig'] for e in sel]):.1%}  "
                  f"spk={np.mean([e['proj']['spk'] for e in sel]):.4f}")

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  full results saved to: {out_path}")
    except Exception as e:
        print(f"  WARNING: could not save summary: {e}")

    section("INTERPRETATION")
    print("""
  Five paths, one retrieval algorithm, different score backends, plus the
  classical control:

    CUBLAS  standard dot-product attention on raw embeddings. Under the
            coherent same-dim attack, expect attack-key concentration (spk).
    TIED    production dual-channel megakernel (geom + proj + agreement).
            Argmax = geometry; the agreement certificate is reported.
    GEO     geometry-only argmax. Same top-1 as TIED.
    QPROJ   projection on QPU bucket counts, calibrated (tile, mask, thr).
    GPROJ   projection on noiseless GPU bucket counts, calibrated the same way.

  Same physics, three substrates. QPROJ and GPROJ should both retrieve; the
  QPU's hardware noise should attenuate sig relative to the noiseless GPROJ.
""")


# =============================================================================
# STAGE 2 — SEMANTIC SCALE SWEEP (mixed softmax components)
# =============================================================================
def run_sweep(args, k_geo, k_proj, base_to_components, excluded):
    if excluded:
        names = ", ".join(Path(f).stem[:28] for f, _ in excluded)
        print(f"\n  [sweep] excluded (no usable component): {names}")
    sweep_keys = list(SWEEPS.keys()) if args.sweep == "ALL" else [args.sweep]
    for name in sweep_keys:
        cfg = SWEEPS[name]
        section(f"SCALE SWEEP: {name}  [strategy=mixed, d={args.sweep_d}]")
        X_Q, X_K, gt, _ = generate_semantic_environment(
            cfg["M"], cfg["N"], args.sweep_d, 64, cfg["NOISE"],
            cfg["OUTLIER_FRAC"], cfg["OUTLIER_MAG"], SEED)

        clear_gpu()
        cos_topk, _ = cosine_retrieval(X_Q, X_K, max(TOP_K), MAX_VRAM_BYTES)
        clear_gpu()
        geo_topk, _ = geo_retrieval_topk(k_geo, X_Q, X_K, max(TOP_K), MAX_VRAM_BYTES)

        proj_results = []
        for fname, comps in base_to_components.items():
            clear_gpu()
            t0 = time.perf_counter()
            p_topk, _ = proj_retrieval_topk_mixed(k_proj, X_Q, X_K, comps,
                                                  max(TOP_K), MAX_VRAM_BYTES)
            proj_results.append((fname, time.perf_counter() - t0, p_topk))

        def col(name_, r1, r5, r10, mrr):
            return name_, r1, r5, r10, mrr

        rows = [
            col("cosine", recall_at_k(cos_topk, gt, 1), recall_at_k(cos_topk, gt, 5),
                recall_at_k(cos_topk, gt, 10), mean_reciprocal_rank(cos_topk, gt)),
            col("geo", recall_at_k(geo_topk, gt, 1), recall_at_k(geo_topk, gt, 5),
                recall_at_k(geo_topk, gt, 10), mean_reciprocal_rank(geo_topk, gt)),
        ]
        for fname, _, p_topk in proj_results:
            stem = Path(fname).stem
            label = stem if len(stem) <= 18 else stem[:17] + "\u2026"
            rows.append(col(f"proj:{label}", recall_at_k(p_topk, gt, 1),
                            recall_at_k(p_topk, gt, 5), recall_at_k(p_topk, gt, 10),
                            mean_reciprocal_rank(p_topk, gt)))

        print(f"\n  {'backend':<28}{'R@1':>9}{'R@5':>9}{'R@10':>9}{'MRR':>9}")
        print("  " + "-" * 64)
        for nm, r1, r5, r10, mrr in rows:
            print(f"  {nm:<28}{r1:>8.2%}{r5:>8.2%}{r10:>8.2%}{mrr:>9.3f}")
        print()


# =============================================================================
# STAGE 3 — NEGATIVE CONTROL PROBES (load-bearing + separation)
# =============================================================================
def probe_base_load_bearing(k_proj, base_to_component, args):
    section("PROBE A: IS THE BASE LOAD-BEARING? (negative control)")
    if not base_to_component:
        print("  No usable bases (all excluded by the recall@1 floor); skipping.")
        return
    print("  Reusing the calibrated winning component per base; only shot counts vary.\n")
    X_Q, X_K, gt, _ = generate_semantic_environment(
        250_000, 1024, args.sweep_d, 64, 0.12, 0.03, 60.0, SEED)

    print(f"  {'base':>28} | {'tile/mask':>10} | {'real R@1':>9} | {'perm R@1':>9} | {'uniform R@1':>11}")
    print("  " + "-" * 78)
    for fname, comp in base_to_component.items():
        oa, ob, bits = comp["orig_a"], comp["orig_b"], comp["mask_bits"]
        c_real = comp["counts18"]
        tag = f"t{comp['tile_idx']}/{comp['mask_tag']}"

        def run(cnts):
            idx, _ = proj_retrieval_topk_single(
                k_proj, X_Q, X_K, cnts, oa, ob, bits, 1, MAX_VRAM_BYTES, silent=True)
            return recall_at_k(idx, gt, 1)

        r_real = run(c_real); clear_gpu()
        r_perm = run(scramble_counts18(c_real, "perm")); clear_gpu()
        r_unif = run(scramble_counts18(c_real, "uniform")); clear_gpu()
        print(f"  {fname[:28]:>28} | {tag:>10} | {r_real:>8.2%} | {r_perm:>8.2%} | {r_unif:>10.2%}")
    print("""
  Read: real >> perm and real >> uniform means the physical shot counts are
  doing the work — projection is not silently reducing to geometry. Every
  number is proj_megakernel_2d on the calibrated component; only counts18 changes.
""")


def _separation_row(k_proj, k_geo, comp, mag, noise, args):
    """One (cosine, geo, proj) recall@1 triple at a given (mag, noise) point."""
    oa, ob, bits, c_real = comp["orig_a"], comp["orig_b"], comp["mask_bits"], comp["counts18"]
    X_Q, X_K, gt, _ = generate_semantic_environment(
        PROBE_M, PROBE_N, args.sweep_d, 64, noise, PROBE_OUTLIER_FRAC, mag, SEED)
    cos_idx, _ = cosine_retrieval(X_Q, X_K, 1, MAX_VRAM_BYTES); clear_gpu()
    geo_idx, _ = geo_retrieval_topk(k_geo, X_Q, X_K, 1, MAX_VRAM_BYTES); clear_gpu()
    prj_idx, _ = proj_retrieval_topk_single(
        k_proj, X_Q, X_K, c_real, oa, ob, bits, 1, MAX_VRAM_BYTES, silent=True); clear_gpu()
    return (recall_at_k(cos_idx, gt, 1),
            recall_at_k(geo_idx, gt, 1),
            recall_at_k(prj_idx, gt, 1))


def probe_separation(k_proj, k_geo, base_to_component, args):
    section("PROBE B: SEPARATION SPECTRUM (is the task too easy?)")
    if not base_to_component:
        print("  No usable bases (all excluded by the recall@1 floor); skipping.")
        return
    fname = next(iter(base_to_component))
    comp = base_to_component[fname]
    print(f"  base {fname[:40]}  component t{comp['tile_idx']}/{comp['mask_tag']}  d={args.sweep_d}")

    # ---- B1: magnitude varies, noise fixed at the calibration value ----
    print(f"\n  B1  outlier_mag varies | noise fixed at {PROBE_FIXED_NOISE}")
    print(f"  {'outlier_mag':>12} | {'noise':>6} | {'cosine':>8} | {'geo':>8} | {'proj':>8}")
    print("  " + "-" * 56)
    for mag in PROBE_MAG_SWEEP:
        cos_r, geo_r, prj_r = _separation_row(k_proj, k_geo, comp, mag, PROBE_FIXED_NOISE, args)
        print(f"  {mag:>12.1f} | {PROBE_FIXED_NOISE:>6.2f} | "
              f"{cos_r:>7.2%} | {geo_r:>7.2%} | {prj_r:>7.2%}")
    print("""
  Read: at outlier_mag=0 all three should be competitive — the check that
  geo/proj aren't winning only on the attack. As magnitude rises cosine falls
  toward its single-shared-dim floor while geo/proj hold; that gap is the
  attack-robustness separation. Noise is held fixed so this column is the
  magnitude axis alone.
""")

    # ---- B2: noise varies, magnitude fixed (attack present throughout) ----
    print(f"  B2  noise varies | outlier_mag fixed at {PROBE_FIXED_MAG}")
    print(f"  {'noise':>12} | {'o_mag':>6} | {'cosine':>8} | {'geo':>8} | {'proj':>8}")
    print("  " + "-" * 56)
    for noise in PROBE_NOISE_SWEEP:
        cos_r, geo_r, prj_r = _separation_row(k_proj, k_geo, comp, PROBE_FIXED_MAG, noise, args)
        print(f"  {noise:>12.2f} | {PROBE_FIXED_MAG:>6.1f} | "
              f"{cos_r:>7.2%} | {geo_r:>7.2%} | {prj_r:>7.2%}")
    print(f"""
  Read: query noise rises with the attack held fixed. The component was
  recall@1-calibrated at noise={SEM_CAL_NOISE}; if proj degrades faster than
  cosine as noise climbs past that, it is the calibration-distribution mismatch
  showing, not a base failure. geo is the noise-robustness ceiling (it never
  touches the bucket counts).
""")


# =============================================================================
# SEMANTIC CALIBRATION (drives sweep mixed components + probe component)
# =============================================================================
def calibrate_semantic(args, k_proj, bases, master_seed):
    """Per base: recall@1-score on a small semantic env (matching the argmax
    retrieval the sweep/probes run). A base whose best component can't clear the
    recall@1 floor is excluded and announced, not slipped into the sweep as a
    weight-1.00 component that prints 0%. Returns (components, single-best,
    excluded)."""
    def semantic_env(seed):
        Xq, Xk, gt, _ = generate_semantic_environment(
            SEM_CAL_M, SEM_CAL_N, args.sweep_d, SEM_CAL_CLUSTERS, SEM_CAL_NOISE,
            SEM_CAL_OUTLIER_FRAC, SEM_CAL_OUTLIER_MAG, seed)
        return Xq, Xk, gt

    base_to_components = {}
    base_to_component = {}
    excluded = []
    section("SEMANTIC CALIBRATION (recall@1-scored, for sweep / probes)")
    for path_obj, _type in bases:
        fname = path_obj.name
        ctrl, ghost, n = load_base(str(path_obj))
        counts = build_bucket_counts(ctrl, ghost, n)
        clear_gpu()
        cands = calibrate_recall1(k_proj, counts, n, args.sweep_d, semantic_env,
                                  CALIB_SEEDS, master_seed)
        clear_gpu()
        best_r1 = cands[0]["r1"]
        if best_r1 < args.cal_min_r1:
            excluded.append((fname, best_r1))
            print(f"  {fname[:46]:<46}  ->  NO USABLE COMPONENT "
                  f"(best cal R@1={best_r1:.2%} < floor {args.cal_min_r1:.0%}; "
                  f"best was t{cands[0]['tile_idx']}/{cands[0]['mask_tag']})")
            continue
        comps = build_mixed_components(cands, args.mixed_k, args.mixed_temp)
        base_to_components[fname] = comps
        base_to_component[fname] = cal_to_component(cands[0])
        sel = "  ".join(f"t{c['tile_idx']}/{c['mask_tag']}={c['weight']:.2f}" for c in comps)
        print(f"  {fname[:46]:<46}  ->  {sel}   (cal R@1={best_r1:.2%})")
    if excluded:
        print(f"\n  {len(excluded)} base(s) excluded from sweep/probes — no component "
              f"cleared the {args.cal_min_r1:.0%} recall@1 floor.")
    return base_to_components, base_to_component, excluded


# =============================================================================
# KERNEL COMPILE + BASE DISCOVERY
# =============================================================================
def compile_kernels(kernel_path, mk_kernel_path):
    kp = Path(kernel_path)
    mkp = Path(mk_kernel_path)
    if not kp.exists():
        sys.exit(f"[FATAL] kernel not found: {kp}")
    if not mkp.exists():
        sys.exit(f"[FATAL] megakernel not found: {mkp}")
    src = kp.read_text() + "\n\n" + mkp.read_text()
    try:
        mod = cp.RawModule(code=src, options=("-use_fast_math", "-std=c++17"))
        return (mod.get_function("geo_megakernel_2d"),
                mod.get_function("proj_megakernel_2d"),
                mod.get_function("tied_megakernel_2d"))
    except Exception as e:
        sys.exit(f"[FATAL] kernel compile failed: {e}")


def discover_bases(data_dir, skip_qpu, skip_gpu):
    data_dir = Path(data_dir)
    qpu = [] if skip_qpu else sorted(data_dir.glob("job_*.npz"))
    gpu = [] if skip_gpu else (sorted(data_dir.glob("ghost_oracle_gpu_*.npz")) +
                               sorted(data_dir.glob("noiseless_base_*.npz")))
    return [(p, "QPU") for p in qpu] + [(p, "GPU") for p in gpu]


# =============================================================================
# CLI / MAIN
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="G_M final benchmark: 5-way verify + semantic sweep + controls")
    # verify (gaussian) regime
    p.add_argument("--N", type=int, default=DEFAULT_N, help="Verify queries == keys.")
    p.add_argument("--d", type=int, default=DEFAULT_D, help="Verify per-dim.")
    p.add_argument("--jitter", type=float, default=DEFAULT_JITTER)
    p.add_argument("--attack-fraction", type=float, default=DEFAULT_ATTACK_FRACTION)
    p.add_argument("--magnitude", type=float, default=DEFAULT_MAGNITUDE)
    p.add_argument("--power", type=float, default=DEFAULT_POWER)
    # semantic (sweep / probe) regime
    p.add_argument("--sweep", choices=["SMALL", "MEDIUM", "LARGE", "ALL"], default=None,
                   help="Run the clustered semantic scale sweep.")
    p.add_argument("--sweep-d", type=int, default=DEFAULT_SWEEP_D,
                   help="Per-dim for the semantic sweep / probes.")
    p.add_argument("--mixed-k", type=int, default=1, help="Top-K components to mix.")
    p.add_argument("--mixed-temp", type=float, default=10.0, help="Softmax temperature.")
    p.add_argument("--cal-min-r1", type=float, default=CALIB_MIN_R1,
                   help="Min calibration recall@1 for a base to be usable in sweep/probes.")
    p.add_argument("--probe", action="store_true",
                   help="Run the load-bearing + separation negative controls.")
    # shared
    p.add_argument("--master-seed", type=int, default=None)
    p.add_argument("--kernel", default=str(DEFAULT_KERNEL))
    p.add_argument("--mk-kernel", default=str(DEFAULT_MK_KERNEL))
    p.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out", default=None,
                   help="Verify JSON summary path. Default: <data>/G_M_final_benchmark.json")
    p.add_argument("--skip-gpu", action="store_true")
    p.add_argument("--skip-qpu", action="store_true")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the 5-way verification stage.")
    return p.parse_args()


def main():
    args = parse_args()
    if not _HAVE_CUPY:
        sys.exit("[FATAL] cupy not available.")

    if args.skip_verify and args.sweep is None and not args.probe:
        sys.exit("[FATAL] nothing to do: --skip-verify with no --sweep and no --probe.")

    k_geo, k_proj, k_tied = compile_kernels(args.kernel, args.mk_kernel)

    bases = discover_bases(args.data, args.skip_qpu, args.skip_gpu)
    if not bases:
        sys.exit(f"[FATAL] no base files found in {args.data}")

    master_seed = args.master_seed if args.master_seed is not None else secrets.randbits(63)
    out_path = Path(args.out) if args.out else Path(args.data) / "G_M_final_benchmark.json"

    if not args.skip_verify:
        run_verify(args, k_proj, k_tied, bases, out_path, master_seed)

    if args.sweep is not None or args.probe:
        base_to_components, base_to_component, excluded = calibrate_semantic(
            args, k_proj, bases, master_seed)

        if args.sweep is not None:
            run_sweep(args, k_geo, k_proj, base_to_components, excluded)

        if args.probe:
            probe_base_load_bearing(k_proj, base_to_component, args)
            probe_separation(k_proj, k_geo, base_to_component, args)

    section("DONE")


if __name__ == "__main__":
    main()