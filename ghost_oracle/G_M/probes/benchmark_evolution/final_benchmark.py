#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE — FINAL BENCHMARK
==============================================================================
Single-file benchmark that compares four backends, all running on the same
GPU, all computing some attention-relevant primitive:

  cublas_T1        : |cos(a-b)| via cublasGemmEx with tf32 tensor cores
  ghost_kernel_T1  : |cos(a-b)| via custom CUDA kernel, fp32
  gpu_projection   : G_M via importance-reweighting on noiseless GPU base
  qpu_projection   : G_M via importance-reweighting on QPU base

Default mode (no flags): run all four on the 4x4 case using whichever
ghost_oracle_qpu_*.npz and ghost_oracle_gpu_*.npz files are in cwd.

Sweep modes (--sweep small|attention|extreme): scale up N and d to attention-
relevant sizes. The projection backends use a single-tile estimator at scale
(any tile's 18 bucket counts produce a valid G_M estimate at arbitrary angles).

Modes:
  --mode operator   : just the similarity scores
  --mode attention  : full pipeline: similarity + softmax (cuBLAS only) + V mul

Autotune: first run on a new GPU sweeps block-size for the projection kernel
and caches to ~/.ghost_projection_autotune.json. RTX 3090 has baked values.

Usage:
    # Default smoke test (4x4 on .npz files in cwd)
    python final_benchmark.py

    # Explicit paths
    python final_benchmark.py --qpu QPU.npz --gpu GPU.npz

    # Scale sweep
    python final_benchmark.py --sweep attention

    # Full attention pipeline
    python final_benchmark.py --sweep attention --mode attention
==============================================================================
"""

import argparse
import glob
import json
import math
import os
import secrets
import sys
import time
import warnings

import numpy as np

try:
    import cupy as cp
    from cupy.cuda import cublas
    _HAVE_CUPY = True
except Exception:
    cp = None
    cublas = None
    _HAVE_CUPY = False

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
NUM_TILES   = 12
PAIRS       = [(r, c) for r in range(4) for c in range(4)][:NUM_TILES]
ANGLE_SCALE = 1.05
ALPHA_NORM  = 0.9127

MATRIX_A_ORIG = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B_ORIG = np.array([1.00, 0.80, 0.40, 0.10])

DEFAULT_PEAK_FP32_TFLOPS = 35.58   # RTX 3090
DEFAULT_PEAK_TF32_TFLOPS = 71.0    # RTX 3090 tensor cores

AUTOTUNE_CACHE = os.path.expanduser("~/.ghost_projection_autotune.json")
RTX3090_BLOCK = (32, 8)            # baked, known good on Ampere SM 8.6
BLOCK_CANDIDATES = [(16, 16), (32, 8), (32, 16), (64, 4), (16, 8), (8, 32)]


def data_to_angles(d):
    return (d / np.max(np.abs(d))) * (np.pi / 2) * ANGLE_SCALE


ORIG_A = data_to_angles(MATRIX_A_ORIG)
ORIG_B = data_to_angles(MATRIX_B_ORIG)


# =============================================================================
# EMBEDDED CUDA KERNELS
# =============================================================================
GHOST_KERNEL_T1_SRC = r"""
// Custom T1 kernel: |cos(a - b)| for batched (N, M) inputs.
// One thread per output entry.
extern "C" __global__ void ghost_t1_batch(
    const float* __restrict__ angles_a,    // (N,)
    const float* __restrict__ angles_b,    // (M,)
    float* __restrict__ out,                // (N, M)
    int N, int M)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= M) return;
    float a = angles_a[i];
    float b = angles_b[j];
    out[i * M + j] = fabsf(__cosf(a - b));
}
"""

# Projection kernel: single-tile estimator variant. Each thread handles one
# (i, j) output entry, using one fixed tile's 18 bucket counts to estimate
# G_M at angles (new_a[i], new_b[j]).
GHOST_PROJECTION_SRC = r"""
#define EPS 0.05f
#define CLIP_LOG_W 3.0f
#define NUM_BUCKETS 3

__device__ inline void clipped_log_pair(float theta, float* log_p, float* log_1mp) {
    float s = __sinf(0.5f * theta);
    float p = s * s;
    if (p < EPS) p = EPS;
    if (p > 1.0f - EPS) p = 1.0f - EPS;
    *log_p = __logf(p);
    *log_1mp = __logf(1.0f - p);
}

// Single-tile G_M projection at scale.
// Inputs:
//   counts18:    (3, 3, 2) int32 — bucket counts for one chosen tile
//   orig_a, orig_b: scalar prepared angles for that tile
//   new_a:       (N,) float32
//   new_b:       (M,) float32
//   out:         (N, M) float32
extern "C" __global__ void ghost_projection_single_tile(
    const int* __restrict__ counts18,
    float orig_a,
    float orig_b,
    const float* __restrict__ new_a,
    const float* __restrict__ new_b,
    float* __restrict__ out,
    int N, int M,
    float alpha_norm)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= M) return;

    float na = new_a[i];
    float nb = new_b[j];

    float log_pa_o, log_1mpa_o, log_pb_o, log_1mpb_o;
    float log_pa_n, log_1mpa_n, log_pb_n, log_1mpb_n;
    clipped_log_pair(orig_a, &log_pa_o, &log_1mpa_o);
    clipped_log_pair(orig_b, &log_pb_o, &log_1mpb_o);
    clipped_log_pair(na, &log_pa_n, &log_1mpa_n);
    clipped_log_pair(nb, &log_pb_n, &log_1mpb_n);

    float a_base = log_1mpa_n - log_1mpa_o;
    float a_slope = (log_pa_n - log_pa_o) - a_base;
    float b_base = log_1mpb_n - log_1mpb_o;
    float b_slope = (log_pb_n - log_pb_o) - b_base;

    float w_sum = 0.0f, w0_sum = 0.0f;

    #pragma unroll
    for (int a_b = 0; a_b < NUM_BUCKETS; a_b++) {
        float fa = 0.5f * (float)a_b;
        float lw_a = a_base + fa * a_slope;
        #pragma unroll
        for (int b_b = 0; b_b < NUM_BUCKETS; b_b++) {
            float fb = 0.5f * (float)b_b;
            float lw = lw_a + b_base + fb * b_slope;
            if (lw > CLIP_LOG_W) lw = CLIP_LOG_W;
            if (lw < -CLIP_LOG_W) lw = -CLIP_LOG_W;
            float w = __expf(lw);
            int off = (a_b * NUM_BUCKETS + b_b) * 2;
            int n_zero = counts18[off + 0];
            int n_one  = counts18[off + 1];
            int n_tot  = n_zero + n_one;
            w_sum  += w * (float)n_tot;
            w0_sum += w * (float)n_zero;
        }
    }

    float p0 = (w_sum > 1e-12f) ? (w0_sum / w_sum) : 0.5f;
    float raw = 2.0f * p0 - 1.0f;
    if (raw < 0.0f) raw = 0.0f;
    float val = sqrtf(raw) / alpha_norm;
    if (val > 1.0f) val = 1.0f;
    out[i * M + j] = val;
}

// Multi-tile (4x4 grid) projection — preserved for the default 4x4 mode.
// Identical to the existing ghost_projection.cu single-output-per-tile kernel.
extern "C" __global__ void ghost_projection_4x4(
    const int* __restrict__ counts,        // (T, 3, 3, 2)
    const float* __restrict__ tile_orig_a, // (T,)
    const float* __restrict__ tile_orig_b, // (T,)
    const int* __restrict__ tile_r,
    const int* __restrict__ tile_c,
    const float* __restrict__ new_a,       // (N, 4)
    const float* __restrict__ new_b,       // (N, 4)
    float* __restrict__ out,               // (N, 4, 4)
    int num_tiles, int n_matrices,
    int matrix_size, float alpha_norm)
{
    int tile_idx = blockIdx.x;
    int matrix_idx = blockIdx.y;
    if (matrix_idx >= n_matrices || tile_idx >= num_tiles) return;

    int r = tile_r[tile_idx];
    int c = tile_c[tile_idx];
    float oa = tile_orig_a[tile_idx];
    float ob = tile_orig_b[tile_idx];
    float na = new_a[matrix_idx * matrix_size + r];
    float nb = new_b[matrix_idx * matrix_size + c];

    float log_pa_o, log_1mpa_o, log_pb_o, log_1mpb_o;
    float log_pa_n, log_1mpa_n, log_pb_n, log_1mpb_n;
    clipped_log_pair(oa, &log_pa_o, &log_1mpa_o);
    clipped_log_pair(ob, &log_pb_o, &log_1mpb_o);
    clipped_log_pair(na, &log_pa_n, &log_1mpa_n);
    clipped_log_pair(nb, &log_pb_n, &log_1mpb_n);

    float a_base = log_1mpa_n - log_1mpa_o;
    float a_slope = (log_pa_n - log_pa_o) - a_base;
    float b_base = log_1mpb_n - log_1mpb_o;
    float b_slope = (log_pb_n - log_pb_o) - b_base;

    int counts_base = tile_idx * NUM_BUCKETS * NUM_BUCKETS * 2;
    float w_sum = 0.0f, w0_sum = 0.0f;

    #pragma unroll
    for (int a_b = 0; a_b < NUM_BUCKETS; a_b++) {
        float fa = 0.5f * (float)a_b;
        float lw_a = a_base + fa * a_slope;
        #pragma unroll
        for (int b_b = 0; b_b < NUM_BUCKETS; b_b++) {
            float fb = 0.5f * (float)b_b;
            float lw = lw_a + b_base + fb * b_slope;
            if (lw > CLIP_LOG_W) lw = CLIP_LOG_W;
            if (lw < -CLIP_LOG_W) lw = -CLIP_LOG_W;
            float w = __expf(lw);
            int bucket_off = counts_base + (a_b * NUM_BUCKETS + b_b) * 2;
            int n_zero = counts[bucket_off + 0];
            int n_one  = counts[bucket_off + 1];
            int n_tot  = n_zero + n_one;
            w_sum  += w * (float)n_tot;
            w0_sum += w * (float)n_zero;
        }
    }

    float p0 = (w_sum > 1e-12f) ? (w0_sum / w_sum) : 0.5f;
    float raw = 2.0f * p0 - 1.0f;
    if (raw < 0.0f) raw = 0.0f;
    float val = sqrtf(raw) / alpha_norm;
    if (val > 1.0f) val = 1.0f;
    out[matrix_idx * matrix_size * matrix_size + r * matrix_size + c] = val;
}
"""


def compile_kernels():
    if not _HAVE_CUPY:
        return None, None, None
    try:
        mod_t1 = cp.RawModule(code=GHOST_KERNEL_T1_SRC,
                              options=("-use_fast_math",))
        mod_proj = cp.RawModule(code=GHOST_PROJECTION_SRC,
                                options=("-use_fast_math",))
        return (mod_t1.get_function("ghost_t1_batch"),
                mod_proj.get_function("ghost_projection_single_tile"),
                mod_proj.get_function("ghost_projection_4x4"))
    except Exception as e:
        sys.stderr.write(f"[WARN] kernel compile failed: {e}\n")
        return None, None, None


# =============================================================================
# AUTOTUNE
# =============================================================================
def detect_gpu_name():
    if not _HAVE_CUPY:
        return "no-gpu"
    try:
        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        return "unknown-gpu"


def load_autotune_cache():
    if os.path.exists(AUTOTUNE_CACHE):
        try:
            with open(AUTOTUNE_CACHE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_autotune_cache(cache):
    try:
        with open(AUTOTUNE_CACHE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def autotune_projection_block(kernel_single, counts18, orig_a, orig_b,
                              N=2048, M=2048):
    """Sweep block sizes on a representative shape, return the fastest."""
    print(f"[AUTOTUNE] Sweeping block sizes at N={N}, M={M}...")
    d_counts = cp.asarray(counts18)
    d_a = cp.asarray(np.random.uniform(0.1, math.pi/2, N).astype(np.float32))
    d_b = cp.asarray(np.random.uniform(0.1, math.pi/2, M).astype(np.float32))
    d_out = cp.zeros((N, M), dtype=cp.float32)

    best = (float("inf"), None)
    for (bx, by) in BLOCK_CANDIDATES:
        if bx * by > 1024:
            continue
        grid = ((M + bx - 1) // bx, (N + by - 1) // by, 1)
        block = (bx, by, 1)
        # Warmup
        for _ in range(3):
            kernel_single(grid, block,
                          (d_counts, np.float32(orig_a), np.float32(orig_b),
                           d_a, d_b, d_out, np.int32(N), np.int32(M),
                           np.float32(ALPHA_NORM)))
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            kernel_single(grid, block,
                          (d_counts, np.float32(orig_a), np.float32(orig_b),
                           d_a, d_b, d_out, np.int32(N), np.int32(M),
                           np.float32(ALPHA_NORM)))
        cp.cuda.Device().synchronize()
        t = (time.perf_counter() - t0) / 5
        print(f"           block=({bx:>3},{by:>3})  {t*1000:.3f} ms")
        if t < best[0]:
            best = (t, (bx, by))
    print(f"[AUTOTUNE] best block = {best[1]}  ({best[0]*1000:.3f} ms)")
    return best[1]


def get_projection_block(kernel_single, counts18, orig_a, orig_b):
    """Return cached or autotuned block size."""
    gpu = detect_gpu_name()
    if "3090" in gpu:
        return RTX3090_BLOCK
    cache = load_autotune_cache()
    if gpu in cache and "projection_block" in cache[gpu]:
        bx, by = cache[gpu]["projection_block"]
        return (bx, by)
    bx, by = autotune_projection_block(kernel_single, counts18,
                                       orig_a, orig_b)
    cache.setdefault(gpu, {})["projection_block"] = [bx, by]
    save_autotune_cache(cache)
    return (bx, by)


# =============================================================================
# BASE LOADING
# =============================================================================
def auto_find_npz(prefix):
    pattern = f"{prefix}_*.npz"
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_base(path):
    d = np.load(path)
    ctrl = {t: d[f"ctrl_tile{t}"] for t in range(NUM_TILES)}
    ghost = {t: d[f"ghost_tile{t}"] for t in range(NUM_TILES)}
    label = str(d.get("job_id", os.path.basename(path)))
    return ctrl, ghost, label


def build_bucket_counts(ctrl_dict, ghost_dict):
    counts = np.zeros((NUM_TILES, 3, 3, 2), dtype=np.int32)
    for t in range(NUM_TILES):
        g = ghost_dict[t]
        a_sum = g[:, 0].astype(np.int32) + g[:, 1].astype(np.int32)
        b_sum = g[:, 2].astype(np.int32) + g[:, 3].astype(np.int32)
        ctrl = ctrl_dict[t]
        for a_b in range(3):
            for b_b in range(3):
                mask = (a_sum == a_b) & (b_sum == b_b)
                counts[t, a_b, b_b, 0] = int(((ctrl == 0) & mask).sum())
                counts[t, a_b, b_b, 1] = int(((ctrl == 1) & mask).sum())
    return counts


def representative_tile(counts):
    """Pick tile with most balanced ctrl distribution as the representative."""
    best_idx = 0
    best_imbalance = float("inf")
    for t in range(counts.shape[0]):
        n0 = counts[t, :, :, 0].sum()
        n1 = counts[t, :, :, 1].sum()
        imb = abs(n0 - n1)
        if imb < best_imbalance:
            best_imbalance = imb
            best_idx = t
    return best_idx


# =============================================================================
# BACKENDS — SCALE (N x M output)
# =============================================================================
def cublas_T1_tf32(angles_a, angles_b):
    """
    |cos(a-b)| via cublasGemmEx with tf32 tensor cores.
    Lifted: A=[cos a, sin a] (N,2), B=[cos b, sin b]^T (2,M). C = A @ B (N,M).
    """
    N = angles_a.shape[0]
    M = angles_b.shape[0]
    d_a = cp.asarray(angles_a)
    d_b = cp.asarray(angles_b)
    A = cp.empty((N, 2), dtype=cp.float32)
    B = cp.empty((2, M), dtype=cp.float32)
    A[:, 0] = cp.cos(d_a); A[:, 1] = cp.sin(d_a)
    B[0, :] = cp.cos(d_b); B[1, :] = cp.sin(d_b)
    C = cp.empty((N, M), dtype=cp.float32)

    handle = cp.cuda.device.get_cublas_handle()
    # Enable tensor cores
    try:
        cublas.setMathMode(handle, cublas.CUBLAS_TF32_TENSOR_OP_MATH)
    except Exception:
        pass
    alpha = np.float32(1.0); beta = np.float32(0.0)
    # cuBLAS is column-major: compute B^T @ A^T = (A @ B)^T in column-major
    # Or just use cupy matmul which calls gemmEx under the hood
    C = cp.matmul(A, B)
    cp.cuda.Device().synchronize()
    return cp.abs(C)


def ghost_kernel_T1(angles_a, angles_b, kernel_t1):
    N = angles_a.shape[0]
    M = angles_b.shape[0]
    d_a = cp.asarray(angles_a)
    d_b = cp.asarray(angles_b)
    d_out = cp.empty((N, M), dtype=cp.float32)
    bx, by = 32, 8
    grid = ((M + bx - 1) // bx, (N + by - 1) // by, 1)
    block = (bx, by, 1)
    kernel_t1(grid, block, (d_a, d_b, d_out, np.int32(N), np.int32(M)))
    cp.cuda.Device().synchronize()
    return d_out


def projection_single_tile(angles_a, angles_b, counts18, orig_a, orig_b,
                           kernel_single, block):
    N = angles_a.shape[0]
    M = angles_b.shape[0]
    d_a = cp.asarray(angles_a)
    d_b = cp.asarray(angles_b)
    d_counts = cp.asarray(counts18)
    d_out = cp.empty((N, M), dtype=cp.float32)
    bx, by = block
    grid = ((M + bx - 1) // bx, (N + by - 1) // by, 1)
    blk = (bx, by, 1)
    kernel_single(grid, blk,
                  (d_counts, np.float32(orig_a), np.float32(orig_b),
                   d_a, d_b, d_out, np.int32(N), np.int32(M),
                   np.float32(ALPHA_NORM)))
    cp.cuda.Device().synchronize()
    return d_out


def analytical_G_M(angles_a, angles_b):
    """Closed form on the same shapes for MAE checking."""
    a = angles_a[:, None]
    b = angles_b[None, :]
    co = np.cos(a) * np.cos(b)
    raw = np.sqrt(np.clip((1 + co) / 2, 0, None))
    return np.minimum(raw / ALPHA_NORM, 1.0).astype(np.float32)


def analytical_T1(angles_a, angles_b):
    return np.abs(np.cos(angles_a[:, None] - angles_b[None, :])).astype(np.float32)


# =============================================================================
# BACKENDS — 4x4 mode (legacy compatibility)
# =============================================================================
def build_angle_arrays(inputs_raw):
    n = len(inputs_raw)
    new_a = np.empty((n, 4), dtype=np.float32)
    new_b = np.empty((n, 4), dtype=np.float32)
    for i, (a, b) in enumerate(inputs_raw):
        new_a[i] = data_to_angles(a)
        new_b[i] = data_to_angles(b)
    return new_a, new_b


def random_input(rng):
    a = 0.1 + 0.9 * rng.random(4)
    b = 0.1 + 0.9 * rng.random(4)
    return a, b


def cublas_T1_4x4_batched(new_a_all, new_b_all):
    n = new_a_all.shape[0]
    d_a = cp.asarray(new_a_all)
    d_b = cp.asarray(new_b_all)
    A = cp.stack([cp.cos(d_a), cp.sin(d_a)], axis=-1)
    B = cp.stack([cp.cos(d_b), cp.sin(d_b)], axis=-2)
    handle = cp.cuda.device.get_cublas_handle()
    try:
        cublas.setMathMode(handle, cublas.CUBLAS_TF32_TENSOR_OP_MATH)
    except Exception:
        pass
    C = cp.matmul(A, B)
    cp.cuda.Device().synchronize()
    return cp.abs(C).get()


def ghost_kernel_T1_4x4_batched(new_a_all, new_b_all, kernel_t1):
    """Per-matrix kernel call; for 4x4 this is fine."""
    n = new_a_all.shape[0]
    out = np.empty((n, 4, 4), dtype=np.float32)
    d_out = cp.empty((4, 4), dtype=cp.float32)
    for i in range(n):
        d_a = cp.asarray(new_a_all[i])
        d_b = cp.asarray(new_b_all[i])
        kernel_t1((1, 1, 1), (4, 4, 1),
                  (d_a, d_b, d_out, np.int32(4), np.int32(4)))
        out[i] = d_out.get()
    return out


def projection_4x4_batched(new_a_all, new_b_all, counts, kernel_4x4):
    n = new_a_all.shape[0]
    M = 4
    d_a = cp.asarray(new_a_all)
    d_b = cp.asarray(new_b_all)
    d_counts = cp.asarray(counts)
    d_orig_a = cp.asarray(np.array([ORIG_A[r] for (r, c) in PAIRS], np.float32))
    d_orig_b = cp.asarray(np.array([ORIG_B[c] for (r, c) in PAIRS], np.float32))
    d_tile_r = cp.asarray(np.array([r for (r, c) in PAIRS], np.int32))
    d_tile_c = cp.asarray(np.array([c for (r, c) in PAIRS], np.int32))
    d_out = cp.zeros((n, M, M), dtype=cp.float32)

    MAX_GRID_Y = 65535
    for start in range(0, n, MAX_GRID_Y):
        stop = min(start + MAX_GRID_Y, n)
        chunk = stop - start
        grid = (NUM_TILES, chunk, 1)
        kernel_4x4(grid, (1, 1, 1),
                   (d_counts, d_orig_a, d_orig_b, d_tile_r, d_tile_c,
                    d_a[start:stop], d_b[start:stop], d_out[start:stop],
                    np.int32(NUM_TILES), np.int32(chunk), np.int32(M),
                    np.float32(ALPHA_NORM)))
    cp.cuda.Device().synchronize()
    out = d_out.get()
    covered = set(PAIRS)
    if len(covered) < M * M:
        for r in range(M):
            for c in range(M):
                if (r, c) not in covered:
                    out[:, r, c] = np.nan
    return out


def analytical_T1_4x4(new_a_all, new_b_all):
    a = new_a_all[:, :, None]
    b = new_b_all[:, None, :]
    return np.abs(np.cos(a - b)).astype(np.float64)


def analytical_T3_4x4(new_a_all, new_b_all):
    a = new_a_all[:, :, None]
    b = new_b_all[:, None, :]
    co = np.cos(a) * np.cos(b)
    raw = np.sqrt(np.clip((1 + co) / 2, 0, None))
    return np.minimum(raw / ALPHA_NORM, 1.0).astype(np.float64)


# =============================================================================
# TIMING
# =============================================================================
def time_call(fn, *args, warmup=3, reps=5):
    for _ in range(warmup):
        _ = fn(*args)
    if _HAVE_CUPY:
        cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
        if _HAVE_CUPY:
            cp.cuda.Device().synchronize()
    elapsed = (time.perf_counter() - t0) / reps
    return out, elapsed


def mae(A, B):
    if isinstance(A, cp.ndarray) if _HAVE_CUPY else False:
        A = A.get()
    if isinstance(B, cp.ndarray) if _HAVE_CUPY else False:
        B = B.get()
    mask = ~(np.isnan(A) | np.isnan(B))
    return float(np.mean(np.abs(A[mask] - B[mask])))


def hline(c="-", w=98): print(c * w)
def section(t, w=98):
    print(); hline("=", w); print(f"  {t}"); hline("=", w)


# =============================================================================
# DEFAULT 4x4 MODE
# =============================================================================
def run_4x4_default(qpu_path, gpu_path, kernel_t1, kernel_4x4,
                    n_matrices, peak_tflops):
    section("DEFAULT 4x4 BENCHMARK — all four backends, .npz files in cwd")
    print(f"  QPU: {qpu_path}")
    print(f"  GPU: {gpu_path}")
    print(f"  N matrices: {n_matrices}")
    print()

    qpu_ctrl, qpu_ghost, qpu_label = load_base(qpu_path)
    gpu_ctrl, gpu_ghost, gpu_label = load_base(gpu_path)
    print(f"  QPU job_id: {qpu_label}")
    print(f"  GPU job_id: {gpu_label}")
    print()

    qpu_counts = build_bucket_counts(qpu_ctrl, qpu_ghost)
    gpu_counts = build_bucket_counts(gpu_ctrl, gpu_ghost)

    rng = np.random.default_rng(secrets.randbits(63))
    inputs = [random_input(rng) for _ in range(n_matrices)]
    new_a, new_b = build_angle_arrays(inputs)

    # cuBLAS T1 (tf32)
    cublas_out, t_cublas = time_call(cublas_T1_4x4_batched, new_a, new_b,
                                      warmup=3, reps=5)

    # Ghost kernel T1
    if kernel_t1 is not None:
        ghost_out, t_ghost = time_call(ghost_kernel_T1_4x4_batched,
                                        new_a, new_b, kernel_t1,
                                        warmup=2, reps=3)
    else:
        ghost_out, t_ghost = None, float("nan")

    # GPU projection (12-tile, 4x4)
    gpu_proj, t_gpu = time_call(projection_4x4_batched, new_a, new_b,
                                 gpu_counts, kernel_4x4, warmup=3, reps=5)

    # QPU projection
    qpu_proj, t_qpu = time_call(projection_4x4_batched, new_a, new_b,
                                 qpu_counts, kernel_4x4, warmup=3, reps=5)

    # Analytical targets
    t1_ref = analytical_T1_4x4(new_a, new_b)
    t3_ref = analytical_T3_4x4(new_a, new_b)

    print(f"  {'backend':<22} {'target':<5} {'sec':>9} {'matrices/s':>13} {'GiB/s_out':>10}")
    hline()

    def report(name, target, secs, out_arr):
        rate = n_matrices / secs if secs > 0 else float("inf")
        bytes_per = 4 * 16  # 16 fp32 entries per matrix
        bw = (n_matrices * bytes_per) / secs / (1024**3) if secs > 0 else 0
        print(f"  {name:<22} {target:<5} {secs:>9.4f} {rate:>13,.0f} {bw:>10.2f}")

    report("cublas_T1_tf32", "T1", t_cublas, cublas_out)
    if ghost_out is not None:
        report("ghost_kernel_T1", "T1", t_ghost, ghost_out)
    report("gpu_projection", "G_M", t_gpu, gpu_proj)
    report("qpu_projection", "G_M", t_qpu, qpu_proj)
    print()

    print(f"  Pairwise MAE on this batch:")
    if ghost_out is not None:
        print(f"    cublas vs analytical T1   : {mae(cublas_out, t1_ref):.2e}")
        print(f"    ghost  vs analytical T1   : {mae(ghost_out, t1_ref):.2e}")
        print(f"    ghost  vs cublas          : {mae(ghost_out, cublas_out):.2e}")
    else:
        print(f"    cublas vs analytical T1   : {mae(cublas_out, t1_ref):.2e}")
    print(f"    gpu_proj  vs analytical T3 : {mae(gpu_proj, t3_ref):.2e}")
    print(f"    qpu_proj  vs analytical T3 : {mae(qpu_proj, t3_ref):.2e}")
    print(f"    gpu_proj  vs cublas T1     : {mae(gpu_proj, t1_ref):.2e}  (T1-T3 gap)")
    print()


# =============================================================================
# SCALE SWEEP
# =============================================================================
def run_sweep(qpu_path, gpu_path, sweep_kind, mode,
              kernel_t1, kernel_single, peak_tflops):
    section(f"SCALE SWEEP — {sweep_kind}, mode={mode}")
    sweep_table = {
        "small":     [(256, 256), (1024, 1024), (4096, 4096)],
        "attention": [(1024, 1024), (4096, 4096), (16384, 16384)],
        "extreme":   [(4096, 4096), (16384, 16384), (65536, 65536)],
    }
    shapes = sweep_table[sweep_kind]

    qpu_ctrl, qpu_ghost, _ = load_base(qpu_path)
    gpu_ctrl, gpu_ghost, _ = load_base(gpu_path)
    qpu_counts = build_bucket_counts(qpu_ctrl, qpu_ghost)
    gpu_counts = build_bucket_counts(gpu_ctrl, gpu_ghost)

    rep_idx_qpu = representative_tile(qpu_counts)
    rep_idx_gpu = representative_tile(gpu_counts)
    print(f"  QPU representative tile : {rep_idx_qpu} (r={PAIRS[rep_idx_qpu][0]}, c={PAIRS[rep_idx_qpu][1]})")
    print(f"  GPU representative tile : {rep_idx_gpu}")
    print()

    qpu_counts18 = qpu_counts[rep_idx_qpu]
    gpu_counts18 = gpu_counts[rep_idx_gpu]
    orig_a_qpu = float(ORIG_A[PAIRS[rep_idx_qpu][0]])
    orig_b_qpu = float(ORIG_B[PAIRS[rep_idx_qpu][1]])
    orig_a_gpu = float(ORIG_A[PAIRS[rep_idx_gpu][0]])
    orig_b_gpu = float(ORIG_B[PAIRS[rep_idx_gpu][1]])

    block = get_projection_block(kernel_single, qpu_counts18,
                                  orig_a_qpu, orig_b_qpu)
    print(f"  Projection kernel block: {block}")
    print()

    print(f"  {'shape':>13} {'backend':<22} {'sec':>9} {'entries/s':>14} "
          f"{'GFLOPS_eq':>11} {'MAE_ref':>10}")
    hline()

    for (N, M) in shapes:
        # Use uniform random angles in [0.1, pi/2]
        rng = np.random.default_rng(42)
        a = (0.1 + (math.pi/2 - 0.1) * rng.random(N)).astype(np.float32)
        b = (0.1 + (math.pi/2 - 0.1) * rng.random(M)).astype(np.float32)

        # cuBLAS T1 (tf32)
        cublas_out, t_cublas = time_call(cublas_T1_tf32, a, b,
                                          warmup=3, reps=5)
        # Ghost kernel T1
        if kernel_t1 is not None:
            ghost_out, t_ghost = time_call(ghost_kernel_T1, a, b, kernel_t1,
                                            warmup=3, reps=5)
        else:
            ghost_out, t_ghost = None, float("nan")
        # GPU projection
        gpu_out, t_gpu = time_call(projection_single_tile, a, b, gpu_counts18,
                                    orig_a_gpu, orig_b_gpu,
                                    kernel_single, block,
                                    warmup=3, reps=5)
        # QPU projection
        qpu_out, t_qpu = time_call(projection_single_tile, a, b, qpu_counts18,
                                    orig_a_qpu, orig_b_qpu,
                                    kernel_single, block,
                                    warmup=3, reps=5)

        # MAE checks
        t1_ref = analytical_T1(a, b)
        gm_ref = analytical_G_M(a, b)
        entries = N * M

        def show(name, secs, mae_val):
            rate = entries / secs if secs > 0 else 0
            # T1 ~ 4 ops/entry (cos, sin, cos, sin, mul, mul, add, abs) ~ 8 ops
            # G_M projection ~ 9*8 = ~72 ops/entry
            ops_per = 72.0 if "proj" in name else 8.0
            gflops = (entries * ops_per) / secs / 1e9 if secs > 0 else 0
            print(f"  {N}x{M:<7} {name:<22} {secs:>9.4f} "
                  f"{rate:>14,.0f} {gflops:>11.2f} {mae_val:>10.2e}")

        show("cublas_T1_tf32", t_cublas, mae(cublas_out, t1_ref))
        if ghost_out is not None:
            show("ghost_kernel_T1", t_ghost, mae(ghost_out, t1_ref))
        show("gpu_projection_G_M", t_gpu, mae(gpu_out, gm_ref))
        show("qpu_projection_G_M", t_qpu, mae(qpu_out, gm_ref))
        print()

    if mode == "attention":
        print()
        section("ATTENTION PIPELINE COMPARISON")
        print("  Full pipeline: similarity + (softmax for cuBLAS) + V multiply")
        print()
        d_v = 64
        for (N, M) in shapes:
            rng = np.random.default_rng(42)
            a = (0.1 + (math.pi/2 - 0.1) * rng.random(N)).astype(np.float32)
            b = (0.1 + (math.pi/2 - 0.1) * rng.random(M)).astype(np.float32)
            V = cp.asarray(rng.normal(size=(M, d_v)).astype(np.float32))

            def cublas_attn():
                S = cublas_T1_tf32(a, b)
                P = cp.exp(S - S.max(axis=1, keepdims=True))
                P /= P.sum(axis=1, keepdims=True)
                out = cp.matmul(P, V)
                cp.cuda.Device().synchronize()
                return out

            def projection_attn(counts18, oa, ob):
                G = projection_single_tile(a, b, counts18, oa, ob,
                                            kernel_single, block)
                # G is already bounded in [0,1]; normalize by row-sum
                row_sum = G.sum(axis=1, keepdims=True)
                P = G / cp.maximum(row_sum, 1e-9)
                out = cp.matmul(P, V)
                cp.cuda.Device().synchronize()
                return out

            _, t_cu = time_call(cublas_attn, warmup=3, reps=3)
            _, t_gpu = time_call(lambda: projection_attn(gpu_counts18,
                                                           orig_a_gpu, orig_b_gpu),
                                  warmup=3, reps=3)
            _, t_qpu = time_call(lambda: projection_attn(qpu_counts18,
                                                           orig_a_qpu, orig_b_qpu),
                                  warmup=3, reps=3)

            print(f"  {N}x{M} attention end-to-end:")
            print(f"    cublas_T1 + softmax + V        : {t_cu*1000:>8.2f} ms")
            print(f"    gpu_projection_G_M + V         : {t_gpu*1000:>8.2f} ms  "
                  f"({t_gpu/t_cu:.2f}x cuBLAS)")
            print(f"    qpu_projection_G_M + V         : {t_qpu*1000:>8.2f} ms  "
                  f"({t_qpu/t_cu:.2f}x cuBLAS)")
            print()


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpu", default=None, help="QPU base .npz (autodetect if missing)")
    ap.add_argument("--gpu", default=None, help="GPU base .npz (autodetect if missing)")
    ap.add_argument("--sweep", default=None,
                    choices=["small", "attention", "extreme"],
                    help="Run scale sweep instead of 4x4 default")
    ap.add_argument("--mode", default="operator",
                    choices=["operator", "attention"],
                    help="operator (similarity only) or attention (full pipeline)")
    ap.add_argument("--n-matrices", type=int, default=65536,
                    help="Number of 4x4 matrices in default mode")
    ap.add_argument("--peak-tflops", type=float, default=DEFAULT_PEAK_FP32_TFLOPS)
    args = ap.parse_args()

    if not _HAVE_CUPY:
        sys.stderr.write("[FATAL] cupy is required for this benchmark.\n")
        sys.exit(1)

    _data = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
    qpu_path = args.qpu or auto_find_npz(os.path.join(_data, "job"))
    gpu_path = args.gpu or auto_find_npz(os.path.join(_data, "ghost_oracle_gpu"))
    if qpu_path is None or gpu_path is None:
        sys.stderr.write("[FATAL] need both qpu and gpu .npz files.\n")
        sys.stderr.write("  searched: ghost_oracle_qpu_*.npz, ghost_oracle_gpu_*.npz\n")
        sys.exit(1)

    section("GHOST ORACLE — FINAL BENCHMARK")
    print(f"  GPU: {detect_gpu_name()}")
    print(f"  Peak fp32 ref: {args.peak_tflops:.2f} TFLOPS")
    print(f"  Mode: {'sweep ' + args.sweep if args.sweep else '4x4 default'}")
    if args.sweep and args.mode == "attention":
        print(f"  Pipeline: full attention (similarity + softmax + V@)")

    kernel_t1, kernel_single, kernel_4x4 = compile_kernels()
    if kernel_t1 is None or kernel_single is None or kernel_4x4 is None:
        sys.stderr.write("[FATAL] kernel compilation failed.\n")
        sys.exit(1)

    if args.sweep is None:
        run_4x4_default(qpu_path, gpu_path, kernel_t1, kernel_4x4,
                        args.n_matrices, args.peak_tflops)
    else:
        run_sweep(qpu_path, gpu_path, args.sweep, args.mode,
                  kernel_t1, kernel_single, args.peak_tflops)

    section("DONE")
    print(f"  Default 4x4 mode is the smoke test.")
    print(f"  --sweep attention is the headline benchmark.")
    print(f"  --sweep attention --mode attention runs the full pipeline.")
    print()


if __name__ == "__main__":
    main()
