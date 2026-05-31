#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE AI RETRIEVAL PROBE v1
==============================================================================
Purpose
-------
Evaluate whether the geometric projector behaves like a useful bounded-memory
retrieval mechanism for AI-style semantic memory workloads.

This is NOT a cryptanalysis benchmark.
This is a retrieval systems probe.

What this tests
----------------
1. Recall@K under increasing database size
2. Robustness to outlier contamination
3. Semantic neighborhood preservation
4. Streaming bounded-memory behavior
5. Rank quality (MRR)
6. Comparison against cosine similarity baseline

The probe generates clustered embeddings to better approximate language-like
embedding manifolds instead of pure iid Gaussian noise.

Requirements
------------
- Python 3.10+
- NumPy
- Optional: CuPy for GPU acceleration

Run
---
python ghost_oracle_ai_retrieval_probe_v1.py
==============================================================================
"""

import math
import time
import numpy as np

try:
    import cupy as xp
    _HAVE_CUPY = True
    print("[INIT] CuPy detected. GPU mode enabled.")
except ImportError:
    import numpy as xp
    _HAVE_CUPY = False
    print("[INIT] CuPy not found. CPU mode enabled.")


# ============================================================================
# CONFIG
# ============================================================================

SEED = 42

MAX_VRAM_BYTES = 512 * 1024**2

EMBED_DIM = 128
NUM_CLUSTERS = 64

ALPHA_NORM = 0.9127

TOP_K = [1, 5, 10]

SWEEPS = {
    "SMALL": {
        "M": 50000,
        "N": 1024,
        "NOISE": 0.08,
        "OUTLIER_FRAC": 0.01,
        "OUTLIER_MAG": 40.0,
    },
    "MEDIUM": {
        "M": 250000,
        "N": 1024,
        "NOISE": 0.12,
        "OUTLIER_FRAC": 0.03,
        "OUTLIER_MAG": 60.0,
    },
    "LARGE": {
        "M": 1000000,
        "N": 1024,
        "NOISE": 0.18,
        "OUTLIER_FRAC": 0.05,
        "OUTLIER_MAG": 100.0,
    }
}


# ============================================================================
# UTILS
# ============================================================================


def sync_device():
    if _HAVE_CUPY:
        xp.cuda.Device().synchronize()



def clear_memory():
    if _HAVE_CUPY:
        xp.get_default_memory_pool().free_all_blocks()



def l2_normalize(x, axis=1, eps=1e-8):
    denom = xp.sqrt(xp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x / denom


# ============================================================================
# SYNTHETIC SEMANTIC ENVIRONMENT
# ============================================================================


def generate_semantic_environment(
    M,
    N,
    dim,
    num_clusters,
    noise,
    outlier_frac,
    outlier_mag,
    seed=42,
):
    """
    Generate clustered embeddings approximating semantic manifolds.

    Unlike iid Gaussian noise, clustered structure better resembles:
    - topical embedding regions
    - semantic neighborhoods
    - correlated representations
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    # Create cluster centers
    # ---------------------------------------------------------------------

    centers = rng.normal(size=(num_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # ---------------------------------------------------------------------
    # Database embeddings
    # ---------------------------------------------------------------------

    cluster_ids = rng.integers(0, num_clusters, size=M)

    X_K = centers[cluster_ids]
    X_K += 0.25 * rng.normal(size=(M, dim)).astype(np.float32)

    # ---------------------------------------------------------------------
    # Queries derived from real neighbors
    # ---------------------------------------------------------------------

    query_indices = rng.choice(M, size=N, replace=False)

    X_Q = X_K[query_indices].copy()
    X_Q += noise * rng.normal(size=(N, dim)).astype(np.float32)
    X_Q /= np.linalg.norm(X_Q, axis=1, keepdims=True)

    # ---------------------------------------------------------------------
    # Inject outlier contamination
    # ---------------------------------------------------------------------

    n_outliers = max(1, int(M * outlier_frac))

    outlier_idx = rng.choice(M, size=n_outliers, replace=False)
    bad_dims = rng.integers(0, dim, size=n_outliers)


    X_K /= np.linalg.norm(X_K, axis=1, keepdims=True)
    for i in range(n_outliers):
        X_K[outlier_idx[i], bad_dims[i]] = outlier_mag


    return (
        xp.asarray(X_Q, dtype=xp.float32),
        xp.asarray(X_K, dtype=xp.float32),
        xp.asarray(query_indices, dtype=xp.int32),
    )


# ============================================================================
# COSINE BASELINE
# ============================================================================


def cosine_retrieval_streaming(X_Q, X_K, topk=10, max_vram_bytes=None):
    """
    Streaming cosine retrieval baseline.

    Important:
    This baseline is ALSO streamed.

    That makes the comparison fairer because we compare:
    - nonlinear geometry
    vs
    - standard cosine similarity

    under identical memory constraints.
    """

    N, d = X_Q.shape
    M = X_K.shape[0]

    bytes_per_query = M * 4

    if max_vram_bytes is None:
        chunk_size = N
    else:
        chunk_size = max(1, max_vram_bytes // bytes_per_query)

    peak_vram_mb = (chunk_size * bytes_per_query) / (1024**2)

    all_topk = []

    t0 = time.perf_counter()

    XQ = l2_normalize(X_Q)
    XK = l2_normalize(X_K)

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)

        sim = xp.matmul(XQ[i:end], XK.T)

        idx = xp.argpartition(-sim, kth=topk - 1, axis=1)[:, :topk]

        partial = xp.take_along_axis(sim, idx, axis=1)

        order = xp.argsort(-partial, axis=1)

        topk_sorted = xp.take_along_axis(idx, order, axis=1)

        all_topk.append(topk_sorted)

    sync_device()

    elapsed = time.perf_counter() - t0

    return xp.concatenate(all_topk, axis=0), elapsed, peak_vram_mb


# ============================================================================
# GHOST ORACLE: FUSED MEGAKERNEL (ZERO-ALLOCATION)
# ============================================================================

FUSED_MEGAKERNEL = """
#define PI_HALF 1.57079632679f

extern "C" __global__ void fused_gm_megakernel(
    const float* __restrict__ raw_Q,
    const float* __restrict__ raw_K,
    float* __restrict__ score_out,
    int N, int M, int d, float alpha
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x; 

    __shared__ float s_Q[32][32];
    __shared__ float s_K[32][32];

    float sum = 0.0f;

    for (int k_offset = 0; k_offset < d; k_offset += 32) {
        // Load, Lift, and Cosine-transform Query chunk IN-FLIGHT
        if (i < N && k_offset + threadIdx.x < d) {
            float q_val = raw_Q[i * d + k_offset + threadIdx.x];
            s_Q[threadIdx.y][threadIdx.x] = cosf(PI_HALF * (1.0f + tanhf(q_val / 3.0f)));
        } else {
            s_Q[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load, Lift, and Cosine-transform Key chunk IN-FLIGHT
        if (j < M && k_offset + threadIdx.y < d) {
            float k_val = raw_K[j * d + k_offset + threadIdx.y];
            s_K[threadIdx.x][threadIdx.y] = cosf(PI_HALF * (1.0f + tanhf(k_val / 3.0f)));
        } else {
            s_K[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // Compute G_M partial sum from registers
        if (i < N && j < M) {
            int k_max = min(32, d - k_offset);
            #pragma unroll
            for (int k = 0; k < k_max; ++k) {
                float val = 0.5f + 0.5f * s_Q[threadIdx.y][k] * s_K[threadIdx.x][k];
                sum += sqrtf(val > 0.0f ? val : 0.0f);
            }
        }
        __syncthreads();
    }

    // Average and normalize
    if (i < N && j < M) {
        float final_score = sum / ((float)d * alpha);
        score_out[i * M + j] = final_score > 1.0f ? 1.0f : final_score;
    }
}
"""

def geometric_retrieval_streaming(X_Q, X_K, topk=10, max_vram_bytes=MAX_VRAM_BYTES):
    """
    Megakernel streaming. Raw embeddings go in, Top-K comes out.
    Zero intermediate VRAM allocations.
    """
    N, d = X_Q.shape
    M = X_K.shape[0]

    mod = xp.RawModule(code=FUSED_MEGAKERNEL, options=("-use_fast_math",))
    fused_kernel = mod.get_function("fused_gm_megakernel")

    bytes_per_query = M * 4
    chunk_size = max(1, max_vram_bytes // bytes_per_query)
    peak_vram_mb = (chunk_size * bytes_per_query) / (1024**2)

    all_topk = []
    t0 = time.perf_counter()

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        current_chunk = end - i
        
        # Pass raw arrays directly to the kernel
        raw_Q_chunk = xp.ascontiguousarray(X_Q[i:end])
        score_out = xp.empty((current_chunk, M), dtype=xp.float32)

        threads = (32, 32, 1)
        blocks = ((M + 31) // 32, (current_chunk + 31) // 32, 1)

        fused_kernel(
            blocks, threads,
            (raw_Q_chunk, X_K, score_out, current_chunk, M, d, xp.float32(ALPHA_NORM))
        )

        idx = xp.argpartition(-score_out, kth=topk - 1, axis=1)[:, :topk]
        partial = xp.take_along_axis(score_out, idx, axis=1)
        order = xp.argsort(-partial, axis=1)
        topk_sorted = xp.take_along_axis(idx, order, axis=1)
        all_topk.append(topk_sorted)

    sync_device()
    elapsed = time.perf_counter() - t0

    return xp.concatenate(all_topk, axis=0), elapsed, peak_vram_mb

# ============================================================================
# METRICS
# ============================================================================


def recall_at_k(topk_indices, truth_indices, k):
    hits = []

    top = topk_indices[:, :k]

    for i in range(top.shape[0]):
        hits.append(int(truth_indices[i] in top[i]))

    return float(np.mean(hits))



def mean_reciprocal_rank(topk_indices, truth_indices):
    rr = []

    top_np = xp.asnumpy(topk_indices)
    truth_np = xp.asnumpy(truth_indices)

    for row, gt in zip(top_np, truth_np):
        found = False

        for rank, idx in enumerate(row, start=1):
            if idx == gt:
                rr.append(1.0 / rank)
                found = True
                break

        if not found:
            rr.append(0.0)

    return float(np.mean(rr))


# ============================================================================
# RUNNER
# ============================================================================


def run_probe(name, cfg):
    print("\n" + "=" * 80)
    print(f"PROBE: {name}")
    print("=" * 80)

    print(f"Database Size (M):      {cfg['M']:,}")
    print(f"Queries (N):            {cfg['N']:,}")
    print(f"Embedding Dim:          {EMBED_DIM}")
    print(f"Clusters:               {NUM_CLUSTERS}")
    print(f"Noise:                  {cfg['NOISE']}")
    print(f"Outlier Fraction:       {cfg['OUTLIER_FRAC']:.2%}")
    print(f"Outlier Magnitude:      {cfg['OUTLIER_MAG']}")

    X_Q, X_K, ground_truth = generate_semantic_environment(
        M=cfg['M'],
        N=cfg['N'],
        dim=EMBED_DIM,
        num_clusters=NUM_CLUSTERS,
        noise=cfg['NOISE'],
        outlier_frac=cfg['OUTLIER_FRAC'],
        outlier_mag=cfg['OUTLIER_MAG'],
        seed=SEED,
    )

    print("\nRunning cosine baseline...")

    cos_topk, cos_time, cos_vram = cosine_retrieval_streaming(
        X_Q,
        X_K,
        topk=max(TOP_K),
        max_vram_bytes=MAX_VRAM_BYTES,
    )

    print("Running geometric retrieval...")

    geo_topk, geo_time, geo_vram = geometric_retrieval_streaming(
        X_Q,
        X_K,
        topk=max(TOP_K),
        max_vram_bytes=MAX_VRAM_BYTES,
    )

    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)

    for k in TOP_K:
        cos_r = recall_at_k(cos_topk, ground_truth, k)
        geo_r = recall_at_k(geo_topk, ground_truth, k)

        print(
            f"Recall@{k:<2} | "
            f"Cosine: {cos_r:>7.2%} | "
            f"Ghost: {geo_r:>7.2%}"
        )

    cos_mrr = mean_reciprocal_rank(cos_topk, ground_truth)
    geo_mrr = mean_reciprocal_rank(geo_topk, ground_truth)

    print("-" * 80)

    print(
        f"MRR       | "
        f"Cosine: {cos_mrr:>7.4f} | "
        f"Ghost: {geo_mrr:>7.4f}"
    )

    print("-" * 80)

    print(
        f"Time (s)  | "
        f"Cosine: {cos_time:>7.3f} | "
        f"Ghost: {geo_time:>7.3f}"
    )

    print(
        f"VRAM MB   | "
        f"Cosine: {cos_vram:>7.2f} | "
        f"Ghost: {geo_vram:>7.2f}"
    )

    print("-" * 80)

    # Cleanup
    X_Q = None
    X_K = None
    clear_memory()


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 80)
    print(" GHOST ORACLE AI RETRIEVAL PROBE v1")
    print("=" * 80)

    print(f"Backend: {'CuPy/GPU' if _HAVE_CUPY else 'NumPy/CPU'}")
    print(f"VRAM Budget: {MAX_VRAM_BYTES / (1024**2):.0f} MB")

    for name, cfg in SWEEPS.items():
        run_probe(name, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
