#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
FINAL BENCHMARK -- FIVE-WAY VERIFICATION
==============================================================================
The capstone for the probe sequence. Compares five attention paths head-to-
head on the same data, same attack profile, same accuracy basis:

  1. CUBLAS    Standard dot-product attention via cuBLAS gemm. The
               classical transformer baseline.
  2. TIED      Dual-channel kernel (geometry + projection, agreement
               metric on). Production v3 with hoisted projection
               invariants. Argmax driven by geometry.
  3. GEO       Geometry channel alone driving argmax. Probe 11's
               winner.
  4. QPROJ     Projection channel driven by QPU bucket counts,
               calibrated per-base (best tile, mask, threshold).
  5. GPROJ     Projection channel driven by noiseless classical
               bucket counts, calibrated the same way as QPROJ.

PROJECT FRAMING (the actual story this benchmark verifies):
  Same physics, three platforms. The projection-attention algorithm is
  defined mathematically. It admits three substrates:
    - mathematical reference (numpy FP64, ground truth)
    - GPU (noiseless classical simulation of the projection circuit)
    - QPU (real hardware shots from IBM Runtime)
  All three substrates run the SAME algorithm. The benchmark shows that
  geometry retrieves on all of them, projection retrieves on all of them
  (with hardware-noise attenuation on the QPU), and cuBLAS is the
  classical control showing what standard attention does under the same
  attack.

METRICS REPORTED PER PATH:
  top-1 acc       did argmax land on the true match?
  sig fraction    Flash-Squelch normalized weight on true match (sharpness)
  spike fraction  same but on attack keys (robustness, lower is better)
  time (ms)       wall-clock per inference at PROD_N x PROD_N

OUTPUTS:
  Table per base file (one row per base for qproj/gproj, with cuBLAS/tied/
  geo numbers repeated for context since they're base-independent).
  JSON summary saved for downstream analysis.

USAGE:
  python final_benchmark_5way.py
  python final_benchmark_5way.py --N 4096          # smaller for quick test
  python final_benchmark_5way.py --skip-gpu        # QPU bases only
  python final_benchmark_5way.py --manifest cal.json   # use saved Probe 20
                                                          calibration
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
    from cupy.cuda.memory import OutOfMemoryError
    _HAVE_CUPY = True
except Exception:
    cp = None
    OutOfMemoryError = Exception
    _HAVE_CUPY = False

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG  -- matches the user's tuned Probe 20 settings, not the draft defaults
# =============================================================================
ANGLE_SCALE = 1.05
ALPHA_NORM = 0.9127
DEFAULT_D = 256
DEFAULT_JITTER = 0.3
DEFAULT_ATTACK_FRACTION = 0.05
DEFAULT_MAGNITUDE = 50.0
DEFAULT_POWER = 256.0
BLOCK_SIZE = 256

CALIB_N = 2
CALIB_SEEDS = 3
CALIB_THRESHOLDS = 50
SPIKE_TOLERANCE = 0.05

HERE = Path(__file__).resolve().parent
DEFAULT_KERNEL_CANDIDATES = [
    HERE / "kernels" / "ghost_kernel.cu",
    HERE / "ghost_kernel.cu",
    HERE.parent / "ghost_kernel.cu",
    HERE / "ghost_oracle" / "kernels" / "ghost_kernel.cu",
]
DEFAULT_DATA_CANDIDATES = [HERE / "data", HERE, HERE.parent / "data"]

# Mask catalog matched to Probe 18/20.
MASK_CONFIGS = [
    ("M1", "Baseline (all 9)",            []),
    ("M2", "CUDA accident (drop 4-8)",    [4, 5, 6, 7, 8]),
    ("M3", "Anti-pillars (0,2)(2,0)",     [2, 6]),
    ("M4", "Drop (0,1)(1,0)",             [1, 3]),
    ("M5", "Drop (1,2)(2,1)",             [5, 7]),
    ("M6", "Drop pillars (0,0)(2,2)",     [0, 8]),
    ("M7", "Pure core (0,0|0,1|1,0|1,1)", [2, 5, 6, 7, 8]),
    ("M8", "Mirror core (1,1|1,2|2,1|2,2)", [0, 1, 2, 3, 6]),
]


# =============================================================================
# CUDA: V5 dynamic-mask projection kernel + tied dual-channel kernel
# =============================================================================
FINAL_KERNELS = r"""
#define FB_TILE_M 32
#define FB_TILE_D 64

struct FBInv {
    float log_pa_o, log_1mpa_o, log_pb_o, log_1mpb_o;
};

__device__ inline void fb_compute_inv(float orig_a, float orig_b, FBInv *inv) {
    clipped_log_pair(orig_a, &inv->log_pa_o, &inv->log_1mpa_o);
    clipped_log_pair(orig_b, &inv->log_pb_o, &inv->log_1mpb_o);
}

__device__ inline float fb_projection(
    const int *__restrict__ counts18, const FBInv &inv,
    float na, float nb, float alpha_norm, int mask_bits)
{
    float log_pa_n, log_1mpa_n, log_pb_n, log_1mpb_n;
    clipped_log_pair(na, &log_pa_n, &log_1mpa_n);
    clipped_log_pair(nb, &log_pb_n, &log_1mpb_n);
    float a_base = log_1mpa_n - inv.log_1mpa_o;
    float a_slope = (log_pa_n - inv.log_pa_o) - a_base;
    float b_base = log_1mpb_n - inv.log_1mpb_o;
    float b_slope = (log_pb_n - inv.log_pb_o) - b_base;
    float w_sum = 0.0f, w0_sum = 0.0f;
    #pragma unroll
    for (int a_b = 0; a_b < 3; a_b++) {
        float fa = 0.5f * (float)a_b;
        float lw_a = a_base + fa * a_slope;
        #pragma unroll
        for (int b_b = 0; b_b < 3; b_b++) {
            int idx = a_b * 3 + b_b;
            if (!((mask_bits >> idx) & 1)) continue;
            float fb = 0.5f * (float)b_b;
            float lw = lw_a + b_base + fb * b_slope;
            if (lw > 18.0f)  lw = 18.0f;
            if (lw < -18.0f) lw = -18.0f;
            float w = __expf(lw);
            int off = idx * 2;
            w_sum  += w * (float)(counts18[off] + counts18[off + 1]);
            w0_sum += w * (float)counts18[off];
        }
    }
    float p0 = (w_sum > 1e-12f) ? (w0_sum / w_sum) : 0.5f;
    float raw = fmaxf(2.0f * p0 - 1.0f, 0.0f);
    return fminf(sqrtf(raw) / alpha_norm, 1.0f);
}

// =============================================================================
// Compute per-key score matrices for geometry, projection, and tied agreement.
// One launch produces ALL THREE channels' (N, M) score matrices so downstream
// argmax / squelch can be done host-side without re-traversing the data.
//   out_geom[i, j]     = mean over d of geometry_channel(q_i, k_j)
//   out_proj[i, j]     = mean over d of fb_projection(q_i, k_j, mask)
//   out_agreement[i]   = mean over j of mean over d of |g - p|   (per query)
// =============================================================================
extern "C" __global__ void fb_compute_all_channels(
    const int *__restrict__ counts18,
    float orig_a, float orig_b,
    const float *__restrict__ theta_Q,
    const float *__restrict__ theta_K,
    float *__restrict__ out_geom,        // (N, M)
    float *__restrict__ out_proj,        // (N, M)
    float *__restrict__ out_agreement,   // (N,)
    int N, int M, int d, float alpha_norm, int mask_bits)
{
    __shared__ int s_counts[18];
    __shared__ float s_K[FB_TILE_M][FB_TILE_D];
    __shared__ FBInv s_inv;
    int tid = threadIdx.x;
    if (tid < 18) s_counts[tid] = counts18[tid];
    if (tid == 0) fb_compute_inv(orig_a, orig_b, &s_inv);
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float inv_d = 1.0f / (float)d;
    float local_q[FB_TILE_D];
    float agreement_sum = 0.0f;

    int num_m_chunks = (M + FB_TILE_M - 1) / FB_TILE_M;
    int num_d_chunks = (d + FB_TILE_D - 1) / FB_TILE_D;

    for (int m_c = 0; m_c < num_m_chunks; ++m_c) {
        int j_start = m_c * FB_TILE_M;
        int valid_keys = (m_c == num_m_chunks - 1) ? (M - j_start) : FB_TILE_M;
        float geom_sum_arr[FB_TILE_M] = {0.0f};
        float proj_sum_arr[FB_TILE_M] = {0.0f};
        float diff_sum_arr[FB_TILE_M] = {0.0f};

        for (int d_c = 0; d_c < num_d_chunks; ++d_c) {
            int k_start = d_c * FB_TILE_D;
            int valid_dims = (d_c == num_d_chunks - 1) ? (d - k_start) : FB_TILE_D;
            if (i < N) {
                for (int k = 0; k < valid_dims; ++k)
                    local_q[k] = theta_Q[i * d + k_start + k];
            }
            for (int idx = threadIdx.x; idx < valid_keys * valid_dims; idx += blockDim.x) {
                s_K[idx / valid_dims][idx % valid_dims] =
                    theta_K[(j_start + idx / valid_dims) * d + (k_start + idx % valid_dims)];
            }
            __syncthreads();
            if (i < N) {
                for (int kj = 0; kj < valid_keys; ++kj) {
                    for (int k = 0; k < valid_dims; ++k) {
                        float na = local_q[k];
                        float nb = s_K[kj][k];
                        float g = geometry_channel(na, nb, alpha_norm);
                        float p = fb_projection(s_counts, s_inv, na, nb, alpha_norm, mask_bits);
                        geom_sum_arr[kj] += g;
                        proj_sum_arr[kj] += p;
                        diff_sum_arr[kj] += fabsf(g - p);
                    }
                }
            }
            __syncthreads();
        }
        if (i < N) {
            for (int kj = 0; kj < valid_keys; ++kj) {
                int j = j_start + kj;
                out_geom[i * M + j] = geom_sum_arr[kj] * inv_d;
                out_proj[i * M + j] = proj_sum_arr[kj] * inv_d;
                agreement_sum += diff_sum_arr[kj] * inv_d;
            }
        }
    }
    if (i < N) out_agreement[i] = agreement_sum / (float)M;
}
"""


# =============================================================================
# CPU CHANNELS for pre-flight calibration
# =============================================================================
def clipped_log_pair_np(x):
    pa = 0.5 * (1.0 + np.sin(x))
    pa = np.clip(pa, 1e-7, 1.0 - 1e-7)
    return np.log(pa), np.log(1.0 - pa)

def compute_projection_numpy(theta_Q, theta_K, counts18, orig_a, orig_b):
    N, d = theta_Q.shape
    M = theta_K.shape[0]
    tQ = theta_Q[:, None, :]
    tK = theta_K[None, :, :]
    log_pa_n, log_1mpa_n = clipped_log_pair_np(tQ)
    log_pb_n, log_1mpb_n = clipped_log_pair_np(tK)
    log_pa_o, log_1mpa_o = clipped_log_pair_np(orig_a)
    log_pb_o, log_1mpb_o = clipped_log_pair_np(orig_b)
    a_base  = log_1mpa_n - log_1mpa_o
    a_slope = (log_pa_n - log_pa_o) - a_base
    b_base  = log_1mpb_n - log_1mpb_o
    b_slope = (log_pb_n - log_pb_o) - b_base
    w_sum  = np.zeros((N, M, d), dtype=np.float64)
    w0_sum = np.zeros((N, M, d), dtype=np.float64)
    idx = 0
    for a_b in range(3):
        fa = 0.5 * float(a_b)
        lw_a = a_base + fa * a_slope
        for b_b in range(3):
            fb = 0.5 * float(b_b)
            lw = np.clip(lw_a + b_base + fb * b_slope, -18.0, 18.0)
            w = np.exp(lw)
            n_zero = counts18[idx * 2 + 0]
            n_one  = counts18[idx * 2 + 1]
            w_sum  += w * float(n_zero + n_one)
            w0_sum += w * float(n_zero)
            idx += 1
    p0 = np.where(w_sum > 1e-12, w0_sum / w_sum, 0.5)
    raw_p = np.maximum(2.0 * p0 - 1.0, 0.0)
    val_p = np.minimum(np.sqrt(raw_p) / ALPHA_NORM, 1.0)
    return val_p.mean(axis=-1)

def simulate_hardware_lut(scores, true_mask, spike_mask, thresh, power):
    shifted = np.maximum(scores - thresh, 0.0)
    row_max = np.max(shifted, axis=1, keepdims=True)
    row_max = np.where(row_max == 0.0, 1.0, row_max)
    scaled = shifted / row_max
    w = np.power(scaled, power)
    denom = np.sum(w, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_norm = np.where(denom > 1e-25, w / denom, 0.0)
    sig = float(w_norm[true_mask].mean())
    spk = float((w_norm * spike_mask).sum(axis=1).mean())
    return sig, spk

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


# =============================================================================
# DATA HELPERS
# =============================================================================
MATRIX_A_ORIG = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B_ORIG = np.array([1.00, 0.80, 0.40, 0.10])
def data_to_angles(d): return (d / np.max(np.abs(d))) * (math.pi / 2) * ANGLE_SCALE
ORIG_A = data_to_angles(MATRIX_A_ORIG)
ORIG_B = data_to_angles(MATRIX_B_ORIG)

def pairs_for_num_tiles(n, matrix_size=4):
    return [(r, c) for r in range(matrix_size) for c in range(matrix_size)][:n]

def phase_lift_perdim(X):
    return ((math.pi / 2) * (1.0 + np.tanh(X / 3.0))).astype(np.float32)

def make_attacked_jittered_embeddings(N, M, d, jitter, frac, mag, seed):
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
# CALIBRATION (Probe 20 style)
# =============================================================================
def calibrate_base(counts_per_tile, num_tiles, master_seed, d):
    pairs = pairs_for_num_tiles(num_tiles)
    thresholds = np.linspace(0.0, 0.90, CALIB_THRESHOLDS)
    rng = np.random.default_rng(master_seed)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=CALIB_SEEDS)]
    calib_data = []
    for seed in seeds:
        X_Q, X_K, sp_mask, _ = make_attacked_jittered_embeddings(
            CALIB_N, CALIB_N, d,
            DEFAULT_JITTER, DEFAULT_ATTACK_FRACTION, DEFAULT_MAGNITUDE, seed)
        tQ = phase_lift_perdim(X_Q).astype(np.float64)
        tK = phase_lift_perdim(X_K).astype(np.float64)
        true_m = np.eye(CALIB_N, dtype=bool)
        sp_m = np.zeros((CALIB_N, CALIB_N), dtype=bool)
        sp_m[:, np.where(sp_mask == 1)[0]] = True
        sp_m[true_m] = False
        calib_data.append((tQ, tK, true_m, sp_m))

    best_clean = None
    best_dirty = None
    for tile_idx in range(num_tiles):
        r, c = pairs[tile_idx]
        counts18_base = counts_per_tile[tile_idx].reshape(-1)
        orig_a = float(ORIG_A[r])
        orig_b = float(ORIG_B[c])
        for tag, name, drops in MASK_CONFIGS:
            c_mod = zero_buckets_counts(counts18_base, drops)
            seed_sigs = np.zeros((CALIB_SEEDS, CALIB_THRESHOLDS))
            seed_spks = np.zeros((CALIB_SEEDS, CALIB_THRESHOLDS))
            for s_i, (tQ, tK, true_m, sp_m) in enumerate(calib_data):
                proj = compute_projection_numpy(tQ, tK, c_mod, orig_a, orig_b)
                for t_i, t in enumerate(thresholds):
                    sig, spk = simulate_hardware_lut(proj, true_m, sp_m,
                                                    float(t), DEFAULT_POWER)
                    seed_sigs[s_i, t_i] = sig
                    seed_spks[s_i, t_i] = spk
            med_sig = np.median(seed_sigs, axis=0)
            med_spk = np.median(seed_spks, axis=0)
            clean_mask = med_spk <= SPIKE_TOLERANCE
            if clean_mask.any():
                clean_idx = int(np.argmax(np.where(clean_mask, med_sig, -1.0)))
                clean_sig = float(med_sig[clean_idx])
                cand = {
                    "tile_idx": tile_idx, "tile_rc": [r, c],
                    "orig_a": orig_a, "orig_b": orig_b,
                    "mask_tag": tag, "mask_name": name,
                    "mask_drops": list(drops),
                    "mask_bits": get_bitmask(drops),
                    "thr": float(thresholds[clean_idx]),
                    "sig": clean_sig,
                    "spk": float(med_spk[clean_idx]),
                    "clean": True,
                }
                if best_clean is None or clean_sig > best_clean["sig"]:
                    best_clean = cand
            dirty_idx = int(np.argmax(med_sig))
            cand_d = {
                "tile_idx": tile_idx, "tile_rc": [r, c],
                "orig_a": orig_a, "orig_b": orig_b,
                "mask_tag": tag, "mask_name": name,
                "mask_drops": list(drops),
                "mask_bits": get_bitmask(drops),
                "thr": float(thresholds[dirty_idx]),
                "sig": float(med_sig[dirty_idx]),
                "spk": float(med_spk[dirty_idx]),
                "clean": med_spk[dirty_idx] <= SPIKE_TOLERANCE,
            }
            if best_dirty is None or cand_d["sig"] > best_dirty["sig"]:
                best_dirty = cand_d
    return best_clean if best_clean is not None else best_dirty


# =============================================================================
# THE FIVE PATHS
# =============================================================================
def path_cublas(theta_Q, theta_K, X_Q, X_K):
    """Standard dot-product attention via cuBLAS gemm.
       Uses raw embeddings X_Q, X_K (not phase-lifted), since that's what
       a real transformer would feed in."""
    d_Q = cp.asarray(X_Q)
    d_K = cp.asarray(X_K)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    # scaled dot product: Q @ K.T / sqrt(d)
    scale = 1.0 / math.sqrt(X_Q.shape[1])
    scores = (d_Q @ d_K.T) * scale
    cp.cuda.Device().synchronize()
    t = time.perf_counter() - t0
    return scores, t


def path_kernels(k_compute_all, theta_Q, theta_K, counts18, orig_a, orig_b,
                 mask_bits):
    """Runs the kernel that produces geom, proj, and agreement simultaneously.
       Returns the three on-device arrays. Timing is handled by the caller
       via time_block()."""
    N, d = theta_Q.shape
    M = theta_K.shape[0]
    d_tq = cp.asarray(theta_Q.reshape(-1))
    d_tk = cp.asarray(theta_K.reshape(-1))
    d_counts = cp.asarray(counts18)
    d_geom = cp.empty(N * M, dtype=cp.float32)
    d_proj = cp.empty(N * M, dtype=cp.float32)
    d_agree = cp.empty(N, dtype=cp.float32)
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
    k_compute_all(
        grid, (BLOCK_SIZE, 1, 1),
        (d_counts, np.float32(orig_a), np.float32(orig_b),
         d_tq, d_tk, d_geom, d_proj, d_agree,
         np.int32(N), np.int32(M), np.int32(d),
         np.float32(ALPHA_NORM), np.int32(mask_bits)),
    )
    cp.cuda.Device().synchronize()
    return d_geom.reshape(N, M), d_proj.reshape(N, M), d_agree


def score_path(scores_gpu, true_idx, spike_mask, threshold, power):
    """Given an (N, M) score matrix on GPU, return:
         top1_acc, signal_fraction, spike_fraction
       top1_acc       = mean(argmax(scores) == true_idx)
       sig fraction   = mean weight on true match under Flash-Squelch
       spike fraction = mean weight on attack keys under Flash-Squelch
    """
    # top-1 accuracy
    argmax = cp.argmax(scores_gpu, axis=1).get()
    top1 = float(np.mean(argmax == true_idx))

    # Flash-Squelch sharpness metrics
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


# =============================================================================
# MAIN
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Five-way benchmark: cuBLAS / tied / geo / qproj / gproj")
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--d", type=int, default=DEFAULT_D)
    p.add_argument("--jitter", type=float, default=DEFAULT_JITTER)
    p.add_argument("--attack-fraction", type=float, default=DEFAULT_ATTACK_FRACTION)
    p.add_argument("--magnitude", type=float, default=DEFAULT_MAGNITUDE)
    p.add_argument("--power", type=float, default=DEFAULT_POWER)
    p.add_argument("--master-seed", type=int, default=None)
    p.add_argument("--kernel", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--manifest", default=None,
                   help="Calibration manifest from probe20 (skips re-calibration)")
    p.add_argument("--out", default="/ghost_oracle_suite/data/final_benchmark_5way.json")
    p.add_argument("--skip-gpu", action="store_true")
    p.add_argument("--skip-qpu", action="store_true")
    return p.parse_args()


def section(t, w=130):
    print()
    print("=" * w)
    print(f"  {t}")
    print("=" * w)


def find_kernel(override=None):
    if override:
        p = Path(override)
        if p.exists(): return p
        sys.exit(f"[FATAL] --kernel not found: {override}")
    for p in DEFAULT_KERNEL_CANDIDATES:
        if p.exists(): return p
    sys.exit("[FATAL] ghost_kernel.cu not found")


def find_data_dir(override=None):
    if override:
        p = Path(override)
        if p.exists(): return p
    for p in DEFAULT_DATA_CANDIDATES:
        if p.exists(): return p
    return HERE


def main():
    args = parse_args()
    if not _HAVE_CUPY:
        sys.exit("[FATAL] cupy not available.")

    kpath = find_kernel(args.kernel)
    src = kpath.read_text() + "\n\n" + FINAL_KERNELS
    try:
        mod = cp.RawModule(code=src, options=("-use_fast_math", "-std=c++17"))
        k_compute_all = mod.get_function("fb_compute_all_channels")
    except Exception as e:
        sys.exit(f"[FATAL] compile failed: {e}")

    data_dir = find_data_dir(args.data)
    qpu_files = [] if args.skip_qpu else sorted(data_dir.glob("../data/job_*.npz"))
    gpu_files = [] if args.skip_gpu else (sorted(data_dir.glob("../data/ghost_oracle_gpu_*.npz")) +
                                          sorted(data_dir.glob("noiseless_base_*.npz")))
    qpu_files = [(p, "QPU") for p in qpu_files]
    gpu_files = [(p, "GPU") for p in gpu_files]
    bases = qpu_files + gpu_files
    if not bases:
        sys.exit(f"[FATAL] no base files in {data_dir}")

    master_seed = (args.master_seed if args.master_seed is not None
                   else secrets.randbits(63))

    section("FINAL BENCHMARK -- FIVE-WAY VERIFICATION")
    print(f"  N (queries=keys) : {args.N}")
    print(f"  d (per-dim)      : {args.d}")
    print(f"  attack           : jitter={args.jitter}  frac={args.attack_fraction}  magnitude={args.magnitude}")
    print(f"  power (squelch)  : {args.power}")
    print(f"  master seed      : {master_seed}")
    print(f"  base files       : {len(bases)}  "
          f"({len(qpu_files)} QPU + {len(gpu_files)} GPU)")

    # Load manifest if provided
    cached_cal = {}
    if args.manifest:
        try:
            with open(args.manifest) as f:
                cached_cal = {entry["file"]: entry["calibration"]
                              for entry in json.load(f)["entries"]}
            print(f"  manifest cache   : {args.manifest} ({len(cached_cal)} entries)")
        except Exception as e:
            print(f"  WARNING manifest load failed: {e}")

    # One shared embedding set for all bases so the rows are comparable.
    X_Q, X_K, spike_mask, attack_idx = make_attacked_jittered_embeddings(
        args.N, args.N, args.d,
        args.jitter, args.attack_fraction, args.magnitude, master_seed)
    theta_Q = phase_lift_perdim(X_Q)
    theta_K = phase_lift_perdim(X_K)
    true_idx = np.arange(args.N, dtype=np.int64)

    # ----- cuBLAS path runs once, base-independent -----
    section("CUBLAS BASELINE (base-independent)")
    d_Q = cp.asarray(X_Q)
    d_K = cp.asarray(X_K)
    def fn_cublas():
        scale = 1.0 / math.sqrt(args.d)
        s = (d_Q @ d_K.T) * scale
        cp.cuda.Device().synchronize()
        return s
    scores_cublas, t_cublas = time_block(fn_cublas)
    cublas_top1, cublas_sig, cublas_spk = score_path(
        scores_cublas, true_idx, spike_mask, threshold=0.0, power=args.power)
    print(f"  cuBLAS dot-product:  top1={cublas_top1:.1%}  "
          f"sig={cublas_sig:.1%}  spk={cublas_spk:.6f}  "
          f"time={t_cublas * 1000:.2f} ms")

    # ----- Per-base: tied / geo / qproj / gproj -----
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

        # Calibration (cached or fresh)
        if filename in cached_cal:
            cal = cached_cal[filename]
            cal_src = "cached"
        else:
            cal = calibrate_base(counts, n, master_seed, args.d)
            cal_src = "fresh"
        if cal is None or cal.get("tile_idx", -1) < 0:
            print(f"  {idx:>2}  {type_label:>4}  {filename[:46]:<46}  "
                  f"{'(no_cal)':<8}  --   --     --      --     --      --      --")
            continue

        tile_idx = cal["tile_idx"]
        counts18 = counts[tile_idx].reshape(-1)
        orig_a = float(cal["orig_a"])
        orig_b = float(cal["orig_b"])
        mask_bits = int(cal["mask_bits"])
        thr = float(cal["thr"])

        # Run the all-channels kernel once -- gives us geom + proj + agreement
        def fn_kernels():
            return path_kernels(k_compute_all, theta_Q, theta_K,
                                counts18, orig_a, orig_b, mask_bits)
        (geom_gpu, proj_gpu, agree_gpu), t_kernels = time_block(fn_kernels)

        # Score each derived path. Time the scoring step too so each row has
        # an honest per-path wall time (kernel + scoring), not a shared total.
        def score_geom():
            return score_path(geom_gpu, true_idx, spike_mask,
                              threshold=0.0, power=args.power)
        (geo_top1, geo_sig, geo_spk), t_score_geo = time_block(score_geom)

        def score_proj():
            return score_path(proj_gpu, true_idx, spike_mask,
                              threshold=thr, power=args.power)
        (proj_top1, proj_sig, proj_spk), t_score_proj = time_block(score_proj)

        # TIED's reported argmax is geometry, but TIED pays for the projection
        # work too (the kernel computed both channels). Honest TIED time =
        # kernel cost + scoring on geom.
        tied_top1, tied_sig, tied_spk = geo_top1, geo_sig, geo_spk
        agreement = float(agree_gpu.mean().get())

        t_tied  = (t_kernels + t_score_geo) * 1000
        t_geo   = (t_kernels + t_score_geo) * 1000   # same kernel, same scoring
        t_proj  = (t_kernels + t_score_proj) * 1000

        # Print rows for tied / geo / (q|g)proj with real per-path times
        print(f"  {idx:>2}  {type_label:>4}  {filename[:46]:<46}  "
              f"{'TIED':<8}  {tile_idx:>4}  {cal['mask_tag']:<4}  "
              f"{thr:>6.3f}  {tied_top1:>6.1%}  {tied_sig:>6.1%}  "
              f"{tied_spk:>8.4f}  {t_tied:>8.2f}")
        print(f"  {'':>2}  {'':>4}  {'':<46}  "
              f"{'GEO':<8}  {tile_idx:>4}  {'--':<4}  "
              f"{'--':>6}  {geo_top1:>6.1%}  {geo_sig:>6.1%}  "
              f"{geo_spk:>8.4f}  {t_geo:>8.2f}")

        path_label = "QPROJ" if type_label == "QPU" else "GPROJ"
        print(f"  {'':>2}  {'':>4}  {'':<46}  "
              f"{path_label:<8}  {tile_idx:>4}  {cal['mask_tag']:<4}  "
              f"{thr:>6.3f}  {proj_top1:>6.1%}  {proj_sig:>6.1%}  "
              f"{proj_spk:>8.4f}  {t_proj:>8.2f}")
        print(f"  {'':>2}  {'':>4}  {'':<46}  "
              f"{'agreement':<8} = {agreement:.4f}   (mean |geom - proj| per query)")
        print()

        summary["entries"].append({
            "file": filename, "type": type_label,
            "calibration": cal, "calibration_source": cal_src,
            "agreement": agreement,
            "kernel_time_ms": t_kernels * 1000,
            "tied":  {"top1": tied_top1, "sig": tied_sig, "spk": tied_spk,
                      "time_ms": t_tied},
            "geo":   {"top1": geo_top1,  "sig": geo_sig,  "spk": geo_spk,
                      "time_ms": t_geo},
            "proj":  {"top1": proj_top1, "sig": proj_sig, "spk": proj_spk,
                      "time_ms": t_proj, "path_label": path_label},
        })

    # ----- Final summary -----
    section("FIVE-WAY SUMMARY")
    print(f"  CUBLAS              top1={cublas_top1:.1%}  "
          f"sig={cublas_sig:.1%}  spk={cublas_spk:.6f}  "
          f"t={t_cublas * 1000:.2f} ms")
    geo_top1s = [e["geo"]["top1"] for e in summary["entries"]]
    if geo_top1s:
        print(f"  GEO (mean across bases)   top1={np.mean(geo_top1s):.1%}")
    qproj_top1s = [e["proj"]["top1"] for e in summary["entries"]
                   if e["type"] == "QPU"]
    gproj_top1s = [e["proj"]["top1"] for e in summary["entries"]
                   if e["type"] == "GPU"]
    if qproj_top1s:
        print(f"  QPROJ (mean across QPU bases)  top1={np.mean(qproj_top1s):.1%}  "
              f"sig={np.mean([e['proj']['sig'] for e in summary['entries'] if e['type'] == 'QPU']):.1%}  "
              f"spk={np.mean([e['proj']['spk'] for e in summary['entries'] if e['type'] == 'QPU']):.4f}")
    if gproj_top1s:
        print(f"  GPROJ (mean across GPU bases)  top1={np.mean(gproj_top1s):.1%}  "
              f"sig={np.mean([e['proj']['sig'] for e in summary['entries'] if e['type'] == 'GPU']):.1%}  "
              f"spk={np.mean([e['proj']['spk'] for e in summary['entries'] if e['type'] == 'GPU']):.4f}")

    try:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  full results saved to: {args.out}")
    except Exception as e:
        print(f"  WARNING: could not save summary: {e}")

    section("INTERPRETATION")
    print("""
  The five paths are the same retrieval algorithm with different score
  backends, plus the cuBLAS classical control:

    CUBLAS  -- standard transformer attention. Under coherent same-dim
               attack, expect high attack-key concentration in the
               argmax / sharpness numbers (spk high).
    TIED    -- production dual-channel kernel. Argmax = geometry,
               agreement metric is computed for the certificate.
    GEO     -- geometry-only argmax. Same top-1 as TIED; differs only
               in that TIED also computes (and reports) agreement.
    QPROJ   -- projection driven by QPU hardware bucket counts,
               calibrated for the best (tile, mask, threshold) per base.
    GPROJ   -- projection driven by noiseless classical bucket counts,
               calibrated the same way.

  Same physics, three platforms, one algorithm. QPROJ and GPROJ should
  both retrieve cleanly; QPROJ's hardware noise should attenuate the
  signal fraction relative to GPROJ (which is the noiseless ground
  truth). cuBLAS shows the classical control on raw embeddings.
""")


if __name__ == "__main__":
    main()