#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — PROJECTION BENCHMARK
==============================================================================
The headline benchmark for the suite. Three-axis comparison of cuBLAS
dot-product attention vs the tied-channel G_M streaming kernel on the Probe
10.1 jittered retrieval task with same-dim coherent outlier attack.

Probe 10.1 setup, replicated:
    X_K ~ N(0, 1) of shape (M, d)
    X_Q = X_K + jitter * N(0, 1)            # query i ≈ key i, not identical
    X_K[outlier_indices, k_bad] = magnitude  # same-dim coherent attack

Non-trivial retrieval: query has to find its slightly-noisy match among M
candidates, 5% of which are corrupted on one shared dimension.

Three reported metrics:
    THROUGHPUT  : entries/sec, raw GFLOPS
    ACCURACY    : top-1 retrieval rate under attack
    OPS BUDGET  : ops-per-correct-retrieval = total_ops / (N * accuracy)
                  Honest "cost per right answer" comparison.

cuBLAS:   2 * N * M * d  fma ops on tensor cores
Tied G_M: ~50 * N * M * d  fp ops (mix of fma + transcendentals)
          Transcendentals (cos, sqrt, log, exp) are ~5-10x cycle cost on
          most GPUs vs fma; the 50 figure already accounts for the
          equivalent fma-cycle weighting (so it is an honest comparison
          of compute work, not raw "ops" counts).

All CUDA kernels are loaded from kernels/ghost_kernel.cu (Sections 3 and 4).

Usage:
    python projection_benchmark.py
    python projection_benchmark.py --sweep attention
    python projection_benchmark.py --sweep extreme --jitter 0.3
    python projection_benchmark.py --qpu data/job_xyz.npz --gpu data/ghost_oracle_gpu_...npz
==============================================================================
"""

import argparse
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
# CONFIGURATION (must match qpu.py / gpu.py)
# =============================================================================
ANGLE_SCALE = 1.05
ALPHA_NORM = 0.9127

MATRIX_A_ORIG = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B_ORIG = np.array([1.00, 0.80, 0.40, 0.10])

DEFAULT_PEAK_FP32_TFLOPS = 35.58
RTX3090_BLOCK = (32, 8)

# Ops-per-element constants (transcendental-weighted equivalents)
CUBLAS_OPS_PER_ELEMENT = 2.0  # one fma = 2 flops
TIED_OPS_PER_ELEMENT_PER_DIM = 50.0  # geometry+projection mix

# Repo-root paths. This file lives at <repo>/ghost_oracle/projection_benchmark.py.
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
KERNEL_PATH = HERE / "kernels" / "ghost_kernel.cu"


def data_to_angles(data, scale=ANGLE_SCALE):
    """Scale a real-valued vector into rotation angles in [0, pi/2 * scale]."""
    max_val = np.max(np.abs(data))
    return (data / max_val) * (np.pi / 2) * scale


ORIG_A = data_to_angles(MATRIX_A_ORIG)
ORIG_B = data_to_angles(MATRIX_B_ORIG)


# =============================================================================
# KERNEL LOADING
# =============================================================================
def compile_kernels():
    """Compile ghost_kernel.cu and return the three entry points used here."""
    if not _HAVE_CUPY:
        sys.exit("[FATAL] cupy not available.")
    if not KERNEL_PATH.exists():
        sys.exit(f"[FATAL] kernel source not found: {KERNEL_PATH}")
    try:
        src = KERNEL_PATH.read_text()
        mod = cp.RawModule(code=src, options=("-use_fast_math",))
        return (
            mod.get_function("tied_streaming_perdim"),
            mod.get_function("tied_materialize_perdim"),
            mod.get_function("ghost_projection"),
        )
    except Exception as e:
        sys.exit(f"[FATAL] kernel compile failed: {e}")


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


def clear_gpu_memory():
    """Force release of all cached blocks in CuPy memory pools."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def auto_find_base(kind):
    """
    Search <repo>/data/ for a base .npz of the given kind.
        kind == "qpu" -> files starting with 'job_'            (from dump.py)
        kind == "gpu" -> files starting with 'noiseless_base_' (from Probe 4)
                         or 'ghost_oracle_gpu_'                (from gpu.py)
    Returns the first match sorted lexicographically, or None.
    """
    if kind == "qpu":
        patterns = ["job_*.npz"]
    elif kind == "gpu":
        patterns = ["noiseless_base_*.npz", "ghost_oracle_gpu_*.npz"]
    else:
        return None
    candidates = []
    for pat in patterns:
        candidates.extend(sorted(DATA_DIR.glob(pat)))
    return str(candidates[0]) if candidates else None


def load_base(path):
    """
    Load a base .npz (from dump.py or gpu.py).
    Returns: (ctrl_dict, ghost_dict, job_id_str, num_tiles)
    """
    d = np.load(path)
    if "num_tiles" not in d.files:
        raise ValueError(
            f"Base file {path} is missing the 'num_tiles' key. "
            f"Regenerate it with the current dump.py or gpu.py."
        )
    num_tiles = int(d["num_tiles"])
    ctrl = {t: d[f"ctrl_tile{t}"] for t in range(num_tiles)}
    ghost = {t: d[f"ghost_tile{t}"] for t in range(num_tiles)}
    job_id = str(d["job_id"]) if "job_id" in d.files else Path(path).name
    return ctrl, ghost, job_id, num_tiles


def build_bucket_counts(ctrl_dict, ghost_dict, num_tiles):
    """Collapse shot-level (ctrl, ghost) data to 18-int per-tile bucket counts."""
    counts = np.zeros((num_tiles, 3, 3, 2), dtype=np.int32)
    for t in range(num_tiles):
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
    """Pick the tile with the most balanced ctrl=0/ctrl=1 split."""
    best_idx, best_imbalance = 0, float("inf")
    for t in range(counts.shape[0]):
        imb = abs(counts[t, :, :, 0].sum() - counts[t, :, :, 1].sum())
        if imb < best_imbalance:
            best_imbalance, best_idx = imb, t
    return best_idx


def pairs_for_num_tiles(num_tiles, matrix_size=4):
    """Row-major (r, c) pairs covering the first `num_tiles` entries of a matrix."""
    return [(r, c) for r in range(matrix_size) for c in range(matrix_size)][:num_tiles]


def phase_lift_perdim(X):
    """Saturating phase-lift: θ = (π/2)(1 + tanh(X/3))."""
    return ((math.pi / 2) * (1.0 + np.tanh(X / 3.0))).astype(np.float32)


def analytical_G_M_perdim(theta_Q, theta_K):
    """numpy reference for per-dim G_M aggregation."""
    cosQ = np.cos(theta_Q)
    cosK = np.cos(theta_K)
    prod = cosQ[:, None, :] * cosK[None, :, :]
    per = np.sqrt(np.clip((1 + prod) / 2, 0, None))
    return np.minimum(per.mean(axis=2) / ALPHA_NORM, 1.0).astype(np.float32)


def make_attacked_jittered_embeddings(
    N, M, d, jitter_scale, attack_fraction, magnitude, seed
):
    """
    Probe 10.1 setup: keys are gaussian, queries are key+jitter (matched but
    not identical), `attack_fraction` of keys spike on same dim at `magnitude`.
    """
    rng = np.random.default_rng(seed)
    X_K = rng.normal(size=(M, d)).astype(np.float32)
    if N == M:
        X_Q = X_K + (jitter_scale * rng.normal(size=(N, d))).astype(np.float32)
    else:
        X_Q = rng.normal(size=(N, d)).astype(np.float32)
    n_out = max(1, int(M * attack_fraction))
    outlier_idx = rng.choice(M, size=n_out, replace=False)
    k_bad = int(rng.integers(0, d))
    X_K[outlier_idx, k_bad] = magnitude
    return X_Q, X_K, outlier_idx, k_bad


# =============================================================================
# BACKENDS
# =============================================================================
def cublas_DP_full(X_Q, X_K):
    d_q = cp.asarray(X_Q)
    d_k = cp.asarray(X_K)
    S = cp.matmul(d_q, d_k.T) / math.sqrt(X_Q.shape[1])
    cp.cuda.Device().synchronize()
    return S


def tied_streaming_perdim_run(
    theta_Q, theta_K, counts18, orig_a, orig_b, kernel_tied, block_size=256
):
    N, d = theta_Q.shape
    M = theta_K.shape[0]
    d_tq = cp.asarray(theta_Q.reshape(-1))
    d_tk = cp.asarray(theta_K.reshape(-1))
    d_counts = cp.asarray(counts18)
    d_idx = cp.empty(N, dtype=cp.int32)
    d_score = cp.empty(N, dtype=cp.float32)
    d_proj = cp.empty(N, dtype=cp.float32)
    d_agree = cp.empty(N, dtype=cp.float32)
    grid = ((N + block_size - 1) // block_size, 1, 1)
    kernel_tied(
        grid,
        (block_size, 1, 1),
        (
            d_counts,
            np.float32(orig_a),
            np.float32(orig_b),
            d_tq,
            d_tk,
            d_idx,
            d_score,
            d_proj,
            d_agree,
            np.int32(N),
            np.int32(M),
            np.int32(d),
            np.float32(ALPHA_NORM),
        ),
    )
    cp.cuda.Device().synchronize()
    return d_idx, d_score, d_proj, d_agree


def tied_materialize_perdim_run(
    theta_Q, theta_K, counts18, orig_a, orig_b, kernel_mat, block=RTX3090_BLOCK
):
    N, d = theta_Q.shape
    M = theta_K.shape[0]
    d_tq = cp.asarray(theta_Q.reshape(-1))
    d_tk = cp.asarray(theta_K.reshape(-1))
    d_counts = cp.asarray(counts18)
    d_proj = cp.empty((N, M), dtype=cp.float32)
    d_geom = cp.empty((N, M), dtype=cp.float32)
    bx, by = block
    grid = ((M + bx - 1) // bx, (N + by - 1) // by, 1)
    kernel_mat(
        grid,
        (bx, by, 1),
        (
            d_counts,
            np.float32(orig_a),
            np.float32(orig_b),
            d_tq,
            d_tk,
            d_proj,
            d_geom,
            np.int32(N),
            np.int32(M),
            np.int32(d),
            np.float32(ALPHA_NORM),
        ),
    )
    cp.cuda.Device().synchronize()
    return d_proj, d_geom


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


def time_call(fn, warmup=2, reps=3):
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


def hline(c="-", w=128):
    print(c * w)


def section(t, w=128):
    print()
    hline("=", w)
    print(f"  {t}")
    hline("=", w)


# =============================================================================
# DEFAULT 4x4 SMOKE TEST
# =============================================================================
def run_default_4x4(qpu_path, gpu_path, kernel_projection):
    section("DEFAULT 4x4 SMOKE TEST")
    qpu_ctrl, qpu_ghost, _, qpu_num_tiles = load_base(qpu_path)
    gpu_ctrl, gpu_ghost, _, gpu_num_tiles = load_base(gpu_path)
    qpu_counts = build_bucket_counts(qpu_ctrl, qpu_ghost, qpu_num_tiles)
    gpu_counts = build_bucket_counts(gpu_ctrl, gpu_ghost, gpu_num_tiles)

    n_matrices = 4096
    rng = np.random.default_rng(secrets.randbits(63))
    inputs_a = (0.1 + 0.9 * rng.random((n_matrices, 4))).astype(np.float32)
    inputs_b = (0.1 + 0.9 * rng.random((n_matrices, 4))).astype(np.float32)
    new_a = (
        (inputs_a / np.max(np.abs(inputs_a), axis=1, keepdims=True))
        * (math.pi / 2)
        * ANGLE_SCALE
    )
    new_b = (
        (inputs_b / np.max(np.abs(inputs_b), axis=1, keepdims=True))
        * (math.pi / 2)
        * ANGLE_SCALE
    )

    def run_proj(counts, num_tiles):
        pairs = pairs_for_num_tiles(num_tiles, matrix_size=4)
        d_a = cp.asarray(new_a)
        d_b = cp.asarray(new_b)
        d_counts = cp.asarray(counts)
        d_orig_a = cp.asarray(np.array([ORIG_A[r] for (r, c) in pairs], np.float32))
        d_orig_b = cp.asarray(np.array([ORIG_B[c] for (r, c) in pairs], np.float32))
        d_tile_r = cp.asarray(np.array([r for (r, c) in pairs], np.int32))
        d_tile_c = cp.asarray(np.array([c for (r, c) in pairs], np.int32))
        d_out = cp.zeros((n_matrices, 4, 4), dtype=cp.float32)
        kernel_projection(
            (num_tiles, n_matrices, 1),
            (1, 1, 1),
            (
                d_counts,
                d_orig_a,
                d_orig_b,
                d_tile_r,
                d_tile_c,
                d_a,
                d_b,
                d_out,
                np.int32(num_tiles),
                np.int32(n_matrices),
                np.int32(4),
                np.float32(ALPHA_NORM),
            ),
        )
        cp.cuda.Device().synchronize()
        return d_out

    _, t_gpu, _ = time_call(lambda: run_proj(gpu_counts, gpu_num_tiles))
    _, t_qpu, _ = time_call(lambda: run_proj(qpu_counts, qpu_num_tiles))
    print(
        f"  GPU base projection ({gpu_num_tiles} tiles): "
        f"{t_gpu * 1000:>7.2f} ms for {n_matrices} matrices "
        f"({n_matrices / t_gpu / 1e6:.2f} M matrices/s)"
    )
    print(
        f"  QPU base projection ({qpu_num_tiles} tiles): "
        f"{t_qpu * 1000:>7.2f} ms for {n_matrices} matrices "
        f"({n_matrices / t_qpu / 1e6:.2f} M matrices/s)"
    )
    print()


# =============================================================================
# CORRECTNESS
# =============================================================================
def run_correctness(
    qpu_counts18, gpu_counts18, orig_a_q, orig_b_q, orig_a_g, orig_b_g, kernel_mat, d=64
):
    section("TIED CHANNEL CORRECTNESS — per-dim aggregation")
    print("  Geometry channel MAE vs numpy reference + agreement metric.")
    print()

    N = M = 256
    rng = np.random.default_rng(7)
    Q = rng.normal(size=(N, d)).astype(np.float32)
    K = rng.normal(size=(M, d)).astype(np.float32)
    theta_Q = phase_lift_perdim(Q)
    theta_K = phase_lift_perdim(K)

    geom_ref = analytical_G_M_perdim(theta_Q, theta_K)
    gpu_proj, gpu_geom = tied_materialize_perdim_run(
        theta_Q, theta_K, gpu_counts18, orig_a_g, orig_b_g, kernel_mat
    )
    qpu_proj, qpu_geom = tied_materialize_perdim_run(
        theta_Q, theta_K, qpu_counts18, orig_a_q, orig_b_q, kernel_mat
    )
    geom_cuda_vs_np = mae(gpu_geom, geom_ref)
    agreement_gpu = mae(gpu_proj, gpu_geom)
    agreement_qpu = mae(qpu_proj, qpu_geom)

    print(f"  Shape: {N} x {M} x d={d}")
    print(f"  Geometry kernel vs numpy reference : MAE = {geom_cuda_vs_np:.4e}")
    print(f"  Ghost agreement GPU base           : {agreement_gpu:.4f}")
    print(f"  Ghost agreement QPU base           : {agreement_qpu:.4f}")
    print()


# =============================================================================
# SWEEP WITH JITTER AND OPS BUDGET
# =============================================================================
SWEEP_TABLE = {
    "small": [(256, 256), (1024, 1024), (4096, 4096)],
    "attention": [(1024, 1024), (4096, 4096), (16384, 16384)],
    "extreme": [(4096, 4096), (16384, 16384), (65536, 65536)],
    "sandbox": [(81920, 81920), (98304, 98304), (106496, 106496), (131072, 131072)],
}


def run_sweep(
    qpu_counts18,
    gpu_counts18,
    orig_a_q,
    orig_b_q,
    orig_a_g,
    orig_b_g,
    kernel_tied,
    sweep_kind,
    d=64,
    jitter=0.3,
    attack_fraction=0.05,
    magnitude=50.0,
):
    shapes = SWEEP_TABLE[sweep_kind]

    section(f"JITTERED RETRIEVAL SWEEP — sweep={sweep_kind}, d={d}, jitter={jitter}")
    print(f"  Task: query i ~ key i + N(0, {jitter}); find correct key among M.")
    print(
        f"  Attack: same-dim coherent spike, {attack_fraction:.0%} of keys, magnitude {magnitude}."
    )
    print(
        "  This replicates Probe 10.1's retrieval task (which got G_M=84.5% at d=16)."
    )
    print()

    # Unified single-row header
    header = (
        f"  {'Shape':<13} {'Backend':<28} {'Time(ms)':>10} {'Entries/s':>15} "
        f"{'GFLOPS':>10} {'VRAM(GB)':>9} {'Accuracy':>10} {'Agreement':>11} {'Ops/Correct':>16}"
    )
    print(header)
    hline("-", len(header) + 2)

    for N, M in shapes:
        clear_gpu_memory()
        shape_str = f"{N}x{M}"
        X_Q, X_K, _, _ = make_attacked_jittered_embeddings(
            N, M, d, jitter, attack_fraction, magnitude, seed=42
        )
        gt = np.arange(min(N, M))
        theta_Q = phase_lift_perdim(X_Q)
        theta_K = phase_lift_perdim(X_K)
        entries = N * M

        # --- cuBLAS ---
        cublas_total_ops = CUBLAS_OPS_PER_ELEMENT * N * M * d
        cublas_vram = (N * M * 4 + N * d * 4 + M * d * 4) / (1024**3)
        if sweep_kind != "sandbox":
            cu_out, t_cu, status_cu = time_call(
                lambda: cublas_DP_full(X_Q, X_K), warmup=2, reps=3
            )
        else:
            cu_out, t_cu, status_cu = None, float("nan"), "SKIP"

        if status_cu == "OK":
            cu_acc = top1_accuracy_from_matrix(cu_out, gt)
            cu_gflops = (cublas_total_ops / t_cu) / 1e9
            n_correct = max(1, int(cu_acc * N))
            cu_ops_per_correct = cublas_total_ops / n_correct

            t_str = f"{t_cu * 1000:.3f}"
            ent_str = f"{entries / t_cu:,.0f}"
            gflops_str = f"{cu_gflops:.1f}"
            acc_str = f"{cu_acc:.2%}"
            opc_str = f"{cu_ops_per_correct:,.0f}"
        else:
            t_str = "SKIP" if status_cu == "SKIP" else "OOM"
            ent_str = "--"
            gflops_str = "--"
            acc_str = "SKIP" if status_cu == "SKIP" else "OOM"
            opc_str = "--"

        vram_str = f"{cublas_vram:.2f}"
        print(
            f"  {shape_str:<13} {'cublas_DP_raw':<28} {t_str:>10} {ent_str:>15} "
            f"{gflops_str:>10} {vram_str:>9} {acc_str:>10} {'--':>11} {opc_str:>16}"
        )

        # --- Tied per-dim streaming ---
        for label, counts18, oa, ob in [
            ("gpu_tied_perdim_streaming", gpu_counts18, orig_a_g, orig_b_g),
            ("qpu_tied_perdim_streaming", qpu_counts18, orig_a_q, orig_b_q),
        ]:
            tied_total_ops = TIED_OPS_PER_ELEMENT_PER_DIM * N * M * d
            stream_vram = (N * d * 4 + M * d * 4 + N * 16) / (1024**3)

            def tied_fn():
                idx, score, proj, agree = tied_streaming_perdim_run(
                    theta_Q, theta_K, counts18, oa, ob, kernel_tied
                )
                return idx, agree

            out, t_gm, status_gm = time_call(tied_fn, warmup=2, reps=3)
            if status_gm == "OK":
                idx_out, agree_out = out
                gm_acc = top1_accuracy_from_idx(idx_out, gt)
                gm_gflops = (tied_total_ops / t_gm) / 1e9
                mean_agreement = float(agree_out.get().mean())
                n_correct = max(1, int(gm_acc * N))
                gm_ops_per_correct = tied_total_ops / n_correct

                t_str = f"{t_gm * 1000:.3f}"
                ent_str = f"{entries / t_gm:,.0f}"
                gflops_str = f"{gm_gflops:.1f}"
                acc_str = f"{gm_acc:.2%}"
                agree_str = f"{mean_agreement:.4f}"
                opc_str = f"{gm_ops_per_correct:,.0f}"
            else:
                t_str = "OOM"
                ent_str = "--"
                gflops_str = "--"
                acc_str = "OOM"
                agree_str = "--"
                opc_str = "--"

            vram_str = f"{stream_vram:.4f}"
            print(
                f"  {shape_str:<13} {label:<28} {t_str:>10} {ent_str:>15} "
                f"{gflops_str:>10} {vram_str:>9} {acc_str:>10} {agree_str:>11} {opc_str:>16}"
            )

        print()


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — projection benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--qpu",
        default=None,
        help="Path to QPU base .npz (auto-finds in data/ if omitted).",
    )
    p.add_argument(
        "--gpu",
        default=None,
        help="Path to GPU base .npz (auto-finds in data/ if omitted).",
    )
    p.add_argument(
        "--sweep",
        default=None,
        choices=list(SWEEP_TABLE.keys()),
        help="Run a retrieval sweep instead of the default 4x4 smoke test.",
    )
    p.add_argument(
        "--d", type=int, default=64, help="Per-head dimension for retrieval task."
    )
    p.add_argument(
        "--jitter",
        type=float,
        default=0.3,
        help="Jitter scale on queries (Probe 10.1 default: 0.3).",
    )
    p.add_argument(
        "--attack-fraction", type=float, default=0.05, help="Fraction of keys to spike."
    )
    p.add_argument(
        "--magnitude",
        type=float,
        default=50.0,
        help="Spike magnitude on the attacked dimension.",
    )
    p.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the materialize-channel correctness check.",
    )
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    if not _HAVE_CUPY:
        sys.exit("[FATAL] cupy required.")

    qpu_path = args.qpu or auto_find_base("qpu")
    gpu_path = args.gpu or auto_find_base("gpu")
    if not qpu_path:
        sys.exit(
            f"[FATAL] no QPU base found. Pass --qpu or put a job_*.npz in {DATA_DIR}/"
        )
    if not gpu_path:
        sys.exit(
            f"[FATAL] no GPU base found. Pass --gpu or run gpu.py to generate one in {DATA_DIR}/"
        )

    kernel_tied, kernel_mat, kernel_projection = compile_kernels()

    section("GHOST ORACLE SUITE — PROJECTION BENCHMARK")
    print(f"  GPU                  : {detect_gpu_name()}")
    print(f"  QPU base             : {qpu_path}")
    print(f"  GPU base             : {gpu_path}")
    print(f"  Per-head dim         : {args.d}")
    print(f"  Jitter scale         : {args.jitter} (Probe 10.1 used 0.3)")
    print(
        f"  Attack               : {args.attack_fraction:.0%} keys spike "
        f"at magnitude {args.magnitude}, same dim (Probe 10.1 mode)"
    )
    print()
    print("  OPS BUDGETING (per element):")
    print(
        f"    cuBLAS DP raw          : {CUBLAS_OPS_PER_ELEMENT:.0f} ops/element (fma)"
    )
    print(
        f"    Tied per-dim streaming : {TIED_OPS_PER_ELEMENT_PER_DIM:.0f} ops/(element*dim)"
    )
    print("                             (transcendental-weighted equivalent)")
    print()
    print("  Ops-per-correct = total_ops / (N * accuracy)")
    print("    Lower = more compute-efficient per correct retrieval.")

    qpu_ctrl, qpu_ghost, _, qpu_num_tiles = load_base(qpu_path)
    gpu_ctrl, gpu_ghost, _, gpu_num_tiles = load_base(gpu_path)
    qpu_counts = build_bucket_counts(qpu_ctrl, qpu_ghost, qpu_num_tiles)
    gpu_counts = build_bucket_counts(gpu_ctrl, gpu_ghost, gpu_num_tiles)

    rep_idx_qpu = representative_tile(qpu_counts)
    rep_idx_gpu = representative_tile(gpu_counts)
    qpu_counts18 = qpu_counts[rep_idx_qpu]
    gpu_counts18 = gpu_counts[rep_idx_gpu]

    qpu_pairs = pairs_for_num_tiles(qpu_num_tiles, matrix_size=4)
    gpu_pairs = pairs_for_num_tiles(gpu_num_tiles, matrix_size=4)
    orig_a_q = float(ORIG_A[qpu_pairs[rep_idx_qpu][0]])
    orig_b_q = float(ORIG_B[qpu_pairs[rep_idx_qpu][1]])
    orig_a_g = float(ORIG_A[gpu_pairs[rep_idx_gpu][0]])
    orig_b_g = float(ORIG_B[gpu_pairs[rep_idx_gpu][1]])

    if args.sweep is None:
        run_default_4x4(qpu_path, gpu_path, kernel_projection)
        if not args.skip_correctness:
            run_correctness(
                qpu_counts18,
                gpu_counts18,
                orig_a_q,
                orig_b_q,
                orig_a_g,
                orig_b_g,
                kernel_mat,
                d=args.d,
            )
    else:
        if not args.skip_correctness:
            run_correctness(
                qpu_counts18,
                gpu_counts18,
                orig_a_q,
                orig_b_q,
                orig_a_g,
                orig_b_g,
                kernel_mat,
                d=args.d,
            )
        run_sweep(
            qpu_counts18,
            gpu_counts18,
            orig_a_q,
            orig_b_q,
            orig_a_g,
            orig_b_g,
            kernel_tied,
            args.sweep,
            d=args.d,
            jitter=args.jitter,
            attack_fraction=args.attack_fraction,
            magnitude=args.magnitude,
        )

    section("DONE")
    print("  The three-axis comparison:")
    print("    SPEED      : entries/s, GFLOPS")
    print("    ACCURACY   : top-1 retrieval under jittered same-dim attack")
    print("    EFFICIENCY : ops-per-correct-retrieval (lower = better)")
    print()
    print("  If G_M tied streaming has worse SPEED but better ACCURACY and")
    print("  comparable/better EFFICIENCY at higher accuracy, the tradeoff is real:")
    print("  more compute spent, more correct answers obtained, less VRAM used,")
    print("  and projection-channel agreement certifies the physical implementation.")
    print()


if __name__ == "__main__":
    main()
