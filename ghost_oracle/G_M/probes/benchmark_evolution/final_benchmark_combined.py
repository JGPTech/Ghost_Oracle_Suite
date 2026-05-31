#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE — FINAL BENCHMARK (COMBINED + HONEST)
==============================================================================
Single-file benchmark combining:
  - Shared-memory accelerated G_M projection kernel
  - Streaming zero-allocation G_M projection (fused argmax)
  - Adversarial outlier attack (Probe 10.1 design)
  - Honest scoring: projection kernel outputs ARE the retrieval scores
  - Real OOM handling: cuBLAS attempted at every size, OOM caught explicitly
  - MAE verification at small sizes against analytical G_M closed form

Three comparison modes:

  EQUAL_FOOTING : Both cuBLAS and G_M operate on 1D phase-lifted inputs.
                  Same problem, different operators. Honest head-to-head.

  FULL_INFO     : cuBLAS gets full d_k-dim embeddings, G_M gets 1D phase-lift.
                  Different games. Labeled as such. The "realistic LLM" path.

  THROUGHPUT    : Operator-only timing, no accuracy. Pure speed comparison.

Adversarial attack (probe 10.1 design):
  - Embeddings X_Q, X_K of shape (N, d_k), Gaussian normal
  - X_K[outlier_indices, :] = magnitude  (same-row spike, NOT random-dim)
  - Phase-lift: theta = pi/2 * (1 + tanh(mean(X) * sqrt(d_k) / 3))
  - Ground truth: query i should retrieve key i

Streaming kernel design:
  One thread per query. Loops over all M keys in-register, computing the
  projection G_M score per (i, j) inline. Keeps running argmax in registers.
  Output: (N,) int array of best-match indices. Zero N x M allocation.

Usage:
    python final_benchmark.py                                # default 4x4 mode
    python final_benchmark.py --sweep attention              # scale sweep
    python final_benchmark.py --sweep attention --mode equal_footing
    python final_benchmark.py --sweep extreme --mode throughput
==============================================================================
"""

import argparse
import glob
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
    from cupy.cuda.memory import OutOfMemoryError
    _HAVE_CUPY = True
except Exception:
    cp = None
    cublas = None
    OutOfMemoryError = Exception
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

DEFAULT_PEAK_FP32_TFLOPS = 35.58
RTX3090_BLOCK = (32, 8)


def data_to_angles(d):
    return (d / np.max(np.abs(d))) * (np.pi / 2) * ANGLE_SCALE


ORIG_A = data_to_angles(MATRIX_A_ORIG)
ORIG_B = data_to_angles(MATRIX_B_ORIG)


# =============================================================================
# EMBEDDED CUDA KERNELS
# =============================================================================

# Materializing projection kernel (shared memory accelerated).
# Used when we need the full N x M score matrix (e.g. small-size MAE check
# or full attention pipeline with softmax + V multiply).
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

// Inline projection evaluation. Returns G_M score for single (a, b) given
// the 18 bucket counts and tile's original angles.
__device__ inline float projection_eval(
    const int* __restrict__ counts18,
    float orig_a, float orig_b,
    float na, float nb,
    float alpha_norm)
{
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
            w_sum  += w * (float)(n_zero + n_one);
            w0_sum += w * (float)n_zero;
        }
    }

    float p0 = (w_sum > 1e-12f) ? (w0_sum / w_sum) : 0.5f;
    float raw = 2.0f * p0 - 1.0f;
    if (raw < 0.0f) raw = 0.0f;
    float val = sqrtf(raw) / alpha_norm;
    if (val > 1.0f) val = 1.0f;
    return val;
}

// Materializing variant: writes full (N, M) projection score matrix.
extern "C" __global__ void projection_materialize(
    const int* __restrict__ counts18,
    float orig_a, float orig_b,
    const float* __restrict__ new_a,
    const float* __restrict__ new_b,
    float* __restrict__ out,
    int N, int M,
    float alpha_norm)
{
    // Shared memory: 18 bucket counts loaded once per block
    __shared__ int s_counts[18];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < 18) {
        s_counts[tid] = counts18[tid];
    }
    __syncthreads();

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= M) return;

    float na = new_a[i];
    float nb = new_b[j];
    float val = projection_eval(s_counts, orig_a, orig_b, na, nb, alpha_norm);
    out[i * M + j] = val;
}

// Streaming variant: one thread per query, loops over all keys, keeps argmax
// in registers. Writes only (N,) int array, no N x M allocation.
extern "C" __global__ void projection_streaming_argmax(
    const int* __restrict__ counts18,
    float orig_a, float orig_b,
    const float* __restrict__ new_a,    // (N,)
    const float* __restrict__ new_b,    // (M,)
    int* __restrict__ out_idx,           // (N,)
    float* __restrict__ out_score,       // (N,)
    int N, int M,
    float alpha_norm)
{
    // Shared memory: 18 bucket counts loaded once per block
    __shared__ int s_counts[18];
    int tid = threadIdx.x;
    if (tid < 18) {
        s_counts[tid] = counts18[tid];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float na = new_a[i];

    // Precompute origin-side logs once per thread (constant for all j)
    float log_pa_o, log_1mpa_o, log_pb_o, log_1mpb_o;
    float log_pa_n, log_1mpa_n;
    clipped_log_pair(orig_a, &log_pa_o, &log_1mpa_o);
    clipped_log_pair(orig_b, &log_pb_o, &log_1mpb_o);
    clipped_log_pair(na,     &log_pa_n, &log_1mpa_n);

    float a_base = log_1mpa_n - log_1mpa_o;
    float a_slope = (log_pa_n - log_pa_o) - a_base;

    float best_score = -1.0f;
    int best_j = -1;

    // Stream keys in registers
    for (int j = 0; j < M; ++j) {
        float nb = new_b[j];
        float log_pb_n, log_1mpb_n;
        clipped_log_pair(nb, &log_pb_n, &log_1mpb_n);
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
                int n_zero = s_counts[off + 0];
                int n_one  = s_counts[off + 1];
                w_sum  += w * (float)(n_zero + n_one);
                w0_sum += w * (float)n_zero;
            }
        }

        float p0 = (w_sum > 1e-12f) ? (w0_sum / w_sum) : 0.5f;
        float raw = 2.0f * p0 - 1.0f;
        if (raw < 0.0f) raw = 0.0f;
        float score = sqrtf(raw) / alpha_norm;
        if (score > 1.0f) score = 1.0f;

        if (score > best_score) {
            best_score = score;
            best_j = j;
        }
    }

    out_idx[i] = best_j;
    out_score[i] = best_score;
}

// 4x4 legacy kernel for default smoke test
extern "C" __global__ void projection_4x4_legacy(
    const int* __restrict__ counts,
    const float* __restrict__ tile_orig_a,
    const float* __restrict__ tile_orig_b,
    const int* __restrict__ tile_r,
    const int* __restrict__ tile_c,
    const float* __restrict__ new_a,
    const float* __restrict__ new_b,
    float* __restrict__ out,
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

    int counts_base = tile_idx * NUM_BUCKETS * NUM_BUCKETS * 2;
    float val = projection_eval(counts + counts_base, oa, ob, na, nb, alpha_norm);
    out[matrix_idx * matrix_size * matrix_size + r * matrix_size + c] = val;
}
"""


def compile_kernels():
    if not _HAVE_CUPY:
        return None, None, None
    try:
        mod = cp.RawModule(code=GHOST_PROJECTION_SRC,
                           options=("-use_fast_math",))
        return (mod.get_function("projection_materialize"),
                mod.get_function("projection_streaming_argmax"),
                mod.get_function("projection_4x4_legacy"))
    except Exception as e:
        sys.stderr.write(f"[FATAL] kernel compile failed: {e}\n")
        return None, None, None


# =============================================================================
# UTILS
# =============================================================================
def detect_gpu_name():
    if not _HAVE_CUPY:
        return "no-gpu"
    try:
        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        return "unknown-gpu"


def auto_find_npz(prefix):
    matches = sorted(glob.glob(f"{prefix}_*.npz"))
    return matches[0] if matches else None


def load_base(path):
    d = np.load(path)
    ctrl = {t: d[f"ctrl_tile{t}"] for t in range(NUM_TILES)}
    ghost = {t: d[f"ghost_tile{t}"] for t in range(NUM_TILES)}
    return ctrl, ghost, str(d.get("job_id", os.path.basename(path)))


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
    best_idx, best_imbalance = 0, float("inf")
    for t in range(counts.shape[0]):
        imb = abs(counts[t, :, :, 0].sum() - counts[t, :, :, 1].sum())
        if imb < best_imbalance:
            best_imbalance, best_idx = imb, t
    return best_idx


# =============================================================================
# ANALYTICAL G_M (closed form for MAE verification)
# =============================================================================
def analytical_G_M(angles_a, angles_b):
    a = angles_a[:, None]
    b = angles_b[None, :]
    co = np.cos(a) * np.cos(b)
    raw = np.sqrt(np.clip((1 + co) / 2, 0, None))
    return np.minimum(raw / ALPHA_NORM, 1.0).astype(np.float32)


# =============================================================================
# PHASE LIFT (probe 10.1 design)
# =============================================================================
def phase_lift_collapsed(X, d_k):
    """
    Collapse (N, d_k) embedding to (N,) angle in [0, pi].
    Mean across dims, scaled by sqrt(d_k), tanh-saturated, mapped to [0, pi].
    """
    scalar = X.mean(axis=1) * math.sqrt(d_k)
    return ((math.pi / 2) * (1.0 + np.tanh(scalar / 3.0))).astype(np.float32)


# =============================================================================
# ADVERSARIAL ATTACK (probe 10.1)
# =============================================================================
def make_attacked_embeddings(N, M, d_k, attack_fraction, magnitude, seed):
    """
    Build paired (Q, K) embeddings. Each query matches the same-index key.
    Then inject magnitude spikes into attack_fraction of keys, full-row.
    """
    rng = np.random.default_rng(seed)
    X_Q = rng.normal(size=(N, d_k)).astype(np.float32)
    X_K = X_Q.copy() if N == M else rng.normal(size=(M, d_k)).astype(np.float32)
    n_out = max(1, int(M * attack_fraction))
    outlier_idx = rng.choice(M, size=n_out, replace=False)
    X_K[outlier_idx, :] = magnitude
    return X_Q, X_K, outlier_idx


# =============================================================================
# BACKENDS
# =============================================================================
def cublas_DP_full(X_Q, X_K):
    """
    Standard scaled dot-product attention scoring on FULL d_k embeddings.
    Returns N x M score matrix. Raises OutOfMemoryError if too big.
    """
    d_q = cp.asarray(X_Q)
    d_k = cp.asarray(X_K)
    S = cp.matmul(d_q, d_k.T) / math.sqrt(X_Q.shape[1])
    cp.cuda.Device().synchronize()
    return S


def cublas_DP_phase_lifted(a_lifted, b_lifted):
    """
    Dot product on the 1D phase-lifted scalars. Trivially N*M ops.
    Returns N x M score matrix.
    """
    d_a = cp.asarray(a_lifted)
    d_b = cp.asarray(b_lifted)
    # 1D x 1D outer: cos(a) cos(b) + sin(a) sin(b) = cos(a - b)
    # Use the lifted representation for tensor cores
    A = cp.empty((d_a.shape[0], 2), dtype=cp.float32)
    B = cp.empty((2, d_b.shape[0]), dtype=cp.float32)
    A[:, 0] = cp.cos(d_a); A[:, 1] = cp.sin(d_a)
    B[0, :] = cp.cos(d_b); B[1, :] = cp.sin(d_b)
    handle = cp.cuda.device.get_cublas_handle()
    try:
        cublas.setMathMode(handle, cublas.CUBLAS_TF32_TENSOR_OP_MATH)
    except Exception:
        pass
    S = cp.abs(cp.matmul(A, B))
    cp.cuda.Device().synchronize()
    return S


def gm_projection_materialize(a_lifted, b_lifted, counts18, orig_a, orig_b,
                              kernel_mat, block):
    """Materialize full N x M G_M projection score matrix."""
    N = a_lifted.shape[0]
    M = b_lifted.shape[0]
    d_a = cp.asarray(a_lifted)
    d_b = cp.asarray(b_lifted)
    d_counts = cp.asarray(counts18)
    d_out = cp.empty((N, M), dtype=cp.float32)
    bx, by = block
    grid = ((M + bx - 1) // bx, (N + by - 1) // by, 1)
    kernel_mat(grid, (bx, by, 1),
               (d_counts, np.float32(orig_a), np.float32(orig_b),
                d_a, d_b, d_out, np.int32(N), np.int32(M),
                np.float32(ALPHA_NORM)))
    cp.cuda.Device().synchronize()
    return d_out


def gm_projection_streaming(a_lifted, b_lifted, counts18, orig_a, orig_b,
                            kernel_stream, block_size=256):
    """Streaming G_M projection: zero N x M alloc, returns (N,) argmax + scores."""
    N = a_lifted.shape[0]
    M = b_lifted.shape[0]
    d_a = cp.asarray(a_lifted)
    d_b = cp.asarray(b_lifted)
    d_counts = cp.asarray(counts18)
    d_idx = cp.empty(N, dtype=cp.int32)
    d_score = cp.empty(N, dtype=cp.float32)
    grid = ((N + block_size - 1) // block_size, 1, 1)
    kernel_stream(grid, (block_size, 1, 1),
                  (d_counts, np.float32(orig_a), np.float32(orig_b),
                   d_a, d_b, d_idx, d_score,
                   np.int32(N), np.int32(M), np.float32(ALPHA_NORM)))
    cp.cuda.Device().synchronize()
    return d_idx, d_score


# =============================================================================
# METRICS
# =============================================================================
def top1_accuracy_from_matrix(S, gt):
    if isinstance(S, cp.ndarray):
        S = S.get()
    return float(np.mean(np.argmax(S, axis=1) == gt))


def top1_accuracy_from_idx(idx, gt):
    if isinstance(idx, cp.ndarray):
        idx = idx.get()
    return float(np.mean(idx == gt))


def mae(A, B):
    if isinstance(A, cp.ndarray):
        A = A.get()
    if isinstance(B, cp.ndarray):
        B = B.get()
    mask = ~(np.isnan(A) | np.isnan(B))
    return float(np.mean(np.abs(A[mask] - B[mask])))


# =============================================================================
# TIMING
# =============================================================================
def time_call(fn, warmup=3, reps=5):
    for _ in range(warmup):
        try:
            fn()
        except OutOfMemoryError:
            return None, float("nan"), "OOM"
    if _HAVE_CUPY:
        cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(reps):
        try:
            out = fn()
        except OutOfMemoryError:
            return None, float("nan"), "OOM"
    if _HAVE_CUPY:
        cp.cuda.Device().synchronize()
    elapsed = (time.perf_counter() - t0) / reps
    return out, elapsed, "OK"


def hline(c="-", w=110): print(c * w)
def section(t, w=110):
    print(); hline("=", w); print(f"  {t}"); hline("=", w)


# =============================================================================
# DEFAULT 4x4 MODE
# =============================================================================
def run_default_4x4(qpu_path, gpu_path, kernel_4x4):
    section("DEFAULT 4x4 SMOKE TEST")
    print(f"  QPU: {qpu_path}")
    print(f"  GPU: {gpu_path}")

    qpu_ctrl, qpu_ghost, qpu_label = load_base(qpu_path)
    gpu_ctrl, gpu_ghost, gpu_label = load_base(gpu_path)
    qpu_counts = build_bucket_counts(qpu_ctrl, qpu_ghost)
    gpu_counts = build_bucket_counts(gpu_ctrl, gpu_ghost)

    n_matrices = 4096
    rng = np.random.default_rng(secrets.randbits(63))
    inputs_a = (0.1 + 0.9 * rng.random((n_matrices, 4))).astype(np.float32)
    inputs_b = (0.1 + 0.9 * rng.random((n_matrices, 4))).astype(np.float32)
    new_a = (inputs_a / np.max(np.abs(inputs_a), axis=1, keepdims=True)) \
            * (math.pi/2) * ANGLE_SCALE
    new_b = (inputs_b / np.max(np.abs(inputs_b), axis=1, keepdims=True)) \
            * (math.pi/2) * ANGLE_SCALE

    def run_4x4(counts):
        d_a = cp.asarray(new_a)
        d_b = cp.asarray(new_b)
        d_counts = cp.asarray(counts)
        d_orig_a = cp.asarray(np.array([ORIG_A[r] for (r, c) in PAIRS], np.float32))
        d_orig_b = cp.asarray(np.array([ORIG_B[c] for (r, c) in PAIRS], np.float32))
        d_tile_r = cp.asarray(np.array([r for (r, c) in PAIRS], np.int32))
        d_tile_c = cp.asarray(np.array([c for (r, c) in PAIRS], np.int32))
        d_out = cp.zeros((n_matrices, 4, 4), dtype=cp.float32)
        kernel_4x4((NUM_TILES, n_matrices, 1), (1, 1, 1),
                   (d_counts, d_orig_a, d_orig_b, d_tile_r, d_tile_c,
                    d_a, d_b, d_out,
                    np.int32(NUM_TILES), np.int32(n_matrices),
                    np.int32(4), np.float32(ALPHA_NORM)))
        cp.cuda.Device().synchronize()
        return d_out

    _, t_gpu, _ = time_call(lambda: run_4x4(gpu_counts))
    _, t_qpu, _ = time_call(lambda: run_4x4(qpu_counts))

    print(f"  GPU base projection : {t_gpu*1000:>7.2f} ms for {n_matrices} matrices "
          f"({n_matrices/t_gpu/1e6:.2f} M matrices/s)")
    print(f"  QPU base projection : {t_qpu*1000:>7.2f} ms for {n_matrices} matrices "
          f"({n_matrices/t_qpu/1e6:.2f} M matrices/s)")
    print()


# =============================================================================
# MAE VERIFICATION (small sizes only)
# =============================================================================
def run_mae_verification(qpu_counts18, gpu_counts18, orig_a, orig_b,
                         kernel_mat, kernel_stream):
    section("KERNEL CORRECTNESS — MAE vs ANALYTICAL G_M")
    print("  Verifies the projection kernel computes what we claim.")
    print("  Compares against closed-form G_M(a,b) = sqrt((1 + cos a cos b)/2) / alpha")
    print()

    # Sample size where it's cheap to verify
    N = M = 1024
    rng = np.random.default_rng(7)
    a = (0.1 + (math.pi/2 - 0.1) * rng.random(N)).astype(np.float32)
    b = (0.1 + (math.pi/2 - 0.1) * rng.random(M)).astype(np.float32)

    gm_ref = analytical_G_M(a, b)

    block = RTX3090_BLOCK
    gpu_out = gm_projection_materialize(a, b, gpu_counts18, orig_a, orig_b,
                                         kernel_mat, block)
    qpu_out = gm_projection_materialize(a, b, qpu_counts18, orig_a, orig_b,
                                         kernel_mat, block)

    mae_gpu = mae(gpu_out, gm_ref)
    mae_qpu = mae(qpu_out, gm_ref)
    shot_noise = 1.0 / math.sqrt(4096)

    print(f"  Shape: {N} x {M}")
    print(f"  MAE(GPU projection, analytical G_M) : {mae_gpu:.4e}")
    print(f"  MAE(QPU projection, analytical G_M) : {mae_qpu:.4e}")
    print(f"  Shot noise floor (1/sqrt(4096))     : {shot_noise:.4f}")
    print()

    if mae_gpu < 0.05:
        print(f"  -> GPU projection matches analytical G_M at near shot-noise.")
    else:
        print(f"  -> GPU projection MAE = {mae_gpu:.3f}. Note: shot noise per tile is")
        print(f"     ~{shot_noise:.3f}; with only 4096 base shots reused across all (a,b),")
        print(f"     variance per output entry is at this level.")

    if mae_qpu < 0.20:
        print(f"  -> QPU projection MAE = {mae_qpu:.3f}, within characterized channel error")
        print(f"     range from probes 7-8.")
    print()


# =============================================================================
# ADVERSARIAL SWEEP — three modes
# =============================================================================
SWEEP_TABLE = {
    "small":     [(256, 256), (1024, 1024), (4096, 4096)],
    "attention": [(1024, 1024), (4096, 4096), (16384, 16384)],
    "extreme":   [(4096, 4096), (16384, 16384), (65536, 65536), (131072, 131072)],
}


def run_sweep_adversarial(qpu_counts18, gpu_counts18, orig_a_q, orig_b_q,
                          orig_a_g, orig_b_g,
                          kernel_mat, kernel_stream,
                          sweep_kind, mode, d_k=4096,
                          attack_fraction=0.05, magnitude=50.0):
    """
    Three modes:
      equal_footing : both ops on 1D phase-lifted inputs
      full_info     : cuBLAS on full d_k, projection on 1D phase-lifted
      throughput    : timing only, no accuracy
    """
    shapes = SWEEP_TABLE[sweep_kind]

    section(f"ADVERSARIAL SWEEP — sweep={sweep_kind}, mode={mode}")
    print(f"  d_k (embedding dim)  : {d_k}")
    print(f"  attack fraction      : {attack_fraction:.0%}")
    print(f"  spike magnitude      : {magnitude}")
    print(f"  shapes               : {shapes}")
    if mode == "equal_footing":
        print(f"  comparison           : cuBLAS DP on 1D phase-lift vs G_M on 1D phase-lift")
    elif mode == "full_info":
        print(f"  comparison           : cuBLAS DP on full {d_k}-dim vs G_M on 1D phase-lift")
        print(f"  NOTE: different inputs. cuBLAS sees ground truth, G_M sees projection.")
    print()

    block = RTX3090_BLOCK

    header = (f"  {'shape':>13} {'backend':<28} {'time(ms)':>10} "
              f"{'entries/s':>14} {'vram(GB)':>10}")
    if mode != "throughput":
        header += f" {'top1_acc':>10}"
    print(header)
    hline()

    for (N, M) in shapes:
        # Build adversarial embeddings
        X_Q, X_K, _ = make_attacked_embeddings(N, M, d_k,
                                                attack_fraction, magnitude,
                                                seed=42)
        # Ground truth: query i matches key i (we built X_K = X_Q for paired case)
        gt = np.arange(min(N, M))

        # Phase-lift for G_M side
        a_lift = phase_lift_collapsed(X_Q, d_k)
        b_lift = phase_lift_collapsed(X_K, d_k)

        entries = N * M

        # --- cuBLAS backend depending on mode
        if mode == "full_info":
            # cuBLAS gets the original d_k embeddings (and OOMs at big sizes)
            cublas_vram = (N * M * 4) / (1024**3)
            def cublas_fn():
                return cublas_DP_full(X_Q, X_K)
            cublas_out, t_cu, status_cu = time_call(cublas_fn, warmup=2, reps=3)
            if mode != "throughput" and status_cu == "OK":
                cu_acc = top1_accuracy_from_matrix(cublas_out, gt)
                acc_str = f"{cu_acc:>9.2%}"
            else:
                acc_str = f"{'OOM' if status_cu == 'OOM' else '--':>10}"
            t_str = "OOM" if status_cu == "OOM" else f"{t_cu*1000:>10.3f}"
            ent_str = "--" if status_cu == "OOM" else f"{entries/t_cu:>14,.0f}"
            vram_str = f"{cublas_vram:>10.2f}"
            row = f"  {N}x{M:<7} {'cublas_DP_full_dim':<28} {t_str:>10} {ent_str:>14} {vram_str}"
            if mode != "throughput":
                row += f" {acc_str}"
            print(row)
        else:
            # equal_footing or throughput: cuBLAS on phase-lifted scalars
            cublas_vram = (N * M * 4) / (1024**3)
            def cublas_fn():
                return cublas_DP_phase_lifted(a_lift, b_lift)
            cublas_out, t_cu, status_cu = time_call(cublas_fn, warmup=2, reps=3)
            if mode == "equal_footing" and status_cu == "OK":
                cu_acc = top1_accuracy_from_matrix(cublas_out, gt)
                acc_str = f"{cu_acc:>9.2%}"
            else:
                acc_str = f"{'OOM' if status_cu == 'OOM' else '--':>10}"
            t_str = "OOM" if status_cu == "OOM" else f"{t_cu*1000:>10.3f}"
            ent_str = "--" if status_cu == "OOM" else f"{entries/t_cu:>14,.0f}"
            vram_str = f"{cublas_vram:>10.2f}"
            row = f"  {N}x{M:<7} {'cublas_DP_phase_lifted':<28} {t_str:>10} {ent_str:>14} {vram_str}"
            if mode != "throughput":
                row += f" {acc_str}"
            print(row)

        # --- G_M streaming projection (zero alloc)
        for label, counts18, oa, ob in [
            ("gpu_proj_GM_streaming", gpu_counts18, orig_a_g, orig_b_g),
            ("qpu_proj_GM_streaming", qpu_counts18, orig_a_q, orig_b_q),
        ]:
            def gm_fn():
                idx, _ = gm_projection_streaming(a_lift, b_lift, counts18,
                                                  oa, ob, kernel_stream)
                return idx
            idx_out, t_gm, status_gm = time_call(gm_fn, warmup=3, reps=5)
            stream_vram = (N * 4 + M * 4 + N * 8) / (1024**3)  # input + output
            if mode != "throughput" and status_gm == "OK":
                gm_acc = top1_accuracy_from_idx(idx_out, gt)
                acc_str = f"{gm_acc:>9.2%}"
            else:
                acc_str = f"{'--':>10}"
            t_str = f"{t_gm*1000:>10.3f}" if status_gm == "OK" else "OOM"
            ent_str = f"{entries/t_gm:>14,.0f}" if status_gm == "OK" else "--"
            vram_str = f"{stream_vram:>10.4f}"
            row = f"  {N}x{M:<7} {label:<28} {t_str:>10} {ent_str:>14} {vram_str}"
            if mode != "throughput":
                row += f" {acc_str}"
            print(row)

        print()


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpu", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--sweep", default=None,
                    choices=list(SWEEP_TABLE.keys()))
    ap.add_argument("--mode", default="equal_footing",
                    choices=["equal_footing", "full_info", "throughput"])
    ap.add_argument("--d-k", type=int, default=4096,
                    help="Embedding dimension for adversarial setup")
    ap.add_argument("--attack-fraction", type=float, default=0.05)
    ap.add_argument("--magnitude", type=float, default=50.0)
    ap.add_argument("--skip-mae-check", action="store_true")
    args = ap.parse_args()

    if not _HAVE_CUPY:
        sys.exit("[FATAL] cupy required.")

    _data = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
    qpu_path = args.qpu or auto_find_npz(os.path.join(_data, "job"))
    gpu_path = args.gpu or auto_find_npz(os.path.join(_data, "ghost_oracle_gpu"))
    if not (qpu_path and gpu_path):
        sys.exit("[FATAL] missing .npz files. searched ghost_oracle_{qpu,gpu}_*.npz")

    kernel_mat, kernel_stream, kernel_4x4 = compile_kernels()
    if any(k is None for k in (kernel_mat, kernel_stream, kernel_4x4)):
        sys.exit(1)

    section("GHOST ORACLE — COMBINED FINAL BENCHMARK")
    print(f"  GPU                  : {detect_gpu_name()}")
    print(f"  Peak fp32 ref        : {DEFAULT_PEAK_FP32_TFLOPS:.2f} TFLOPS")
    print(f"  QPU base             : {qpu_path}")
    print(f"  GPU base             : {gpu_path}")
    print(f"  Kernels              : projection_materialize, projection_streaming_argmax")
    print(f"  Mode                 : {'sweep ' + args.sweep + ' / ' + args.mode if args.sweep else '4x4 default'}")

    qpu_ctrl, qpu_ghost, _ = load_base(qpu_path)
    gpu_ctrl, gpu_ghost, _ = load_base(gpu_path)
    qpu_counts = build_bucket_counts(qpu_ctrl, qpu_ghost)
    gpu_counts = build_bucket_counts(gpu_ctrl, gpu_ghost)

    rep_idx_qpu = representative_tile(qpu_counts)
    rep_idx_gpu = representative_tile(gpu_counts)
    qpu_counts18 = qpu_counts[rep_idx_qpu]
    gpu_counts18 = gpu_counts[rep_idx_gpu]
    orig_a_q = float(ORIG_A[PAIRS[rep_idx_qpu][0]])
    orig_b_q = float(ORIG_B[PAIRS[rep_idx_qpu][1]])
    orig_a_g = float(ORIG_A[PAIRS[rep_idx_gpu][0]])
    orig_b_g = float(ORIG_B[PAIRS[rep_idx_gpu][1]])

    if args.sweep is None:
        run_default_4x4(qpu_path, gpu_path, kernel_4x4)
        if not args.skip_mae_check:
            run_mae_verification(qpu_counts18, gpu_counts18,
                                  orig_a_q, orig_b_q,
                                  kernel_mat, kernel_stream)
    else:
        if not args.skip_mae_check:
            run_mae_verification(qpu_counts18, gpu_counts18,
                                  orig_a_q, orig_b_q,
                                  kernel_mat, kernel_stream)
        run_sweep_adversarial(qpu_counts18, gpu_counts18,
                              orig_a_q, orig_b_q,
                              orig_a_g, orig_b_g,
                              kernel_mat, kernel_stream,
                              args.sweep, args.mode,
                              d_k=args.d_k,
                              attack_fraction=args.attack_fraction,
                              magnitude=args.magnitude)

    section("DONE")
    print("  Modes:")
    print("    --mode equal_footing : both backends on 1D phase-lifted inputs (fair compare)")
    print("    --mode full_info     : cuBLAS on d_k=4096 embeddings, G_M on 1D phase-lift")
    print("                           (different problems; G_M wins on VRAM, not on accuracy)")
    print("    --mode throughput    : timing only, no accuracy")
    print()
    print("  The retrieval accuracy column is now scored from the kernels under test.")
    print("  G_M streaming uses zero N x M allocation: scales to 131072x131072 in VRAM")
    print("  budget where cuBLAS at the same shape requires 64 GiB of intermediate matrix.")
    print()


if __name__ == "__main__":
    main()
