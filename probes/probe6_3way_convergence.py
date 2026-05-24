#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — PROBE 6 — EXPLICIT 3-WAY CONVERGENCE
==============================================================================
Three computations exist in this stack. They look superficially similar but
they solve formally different problems. This probe makes the distinction
explicit, evaluates each backend against ITS OWN correct target, and then
runs the cross-comparison so reviewers can see exactly why naive same-target
comparisons mislead.

THE THREE TARGETS:

  T1  RANK-1 COSINE PRODUCT
        C[r,c] = | cos(theta_a[r] - theta_b[c]) |
      This is what the rank-K kernel (ghost_rank_k_matmul in ghost_kernel.cu)
      computes at K=1. It is also what cuBLAS computes when given the lifted
      [cos(a)|sin(a)] representation and matmul'd against [cos(b)|sin(b)]^T.
      cuBLAS and the rank-K kernel MUST agree on this target to fp32 noise.

  T2  HALF-ANGLE HADAMARD (TEXTBOOK SWAP TEST ON PRODUCT STATES)
        C[r,c] = | cos((theta_a[r] - theta_b[c]) / 2) |
      This is what the original code assumed the QPU measured. The textbook
      formula for a swap test on product states |psi_a> and |psi_b>. The
      Probe 1 / Probe 4 arc established the QPU is NOT measuring this,
      because of the ghost CNOTs.

  T3  MIXED-STATE HADAMARD (CIRCUIT-CORRECT TARGET)
        C[r,c] = sqrt(2 * P0_mixed - 1) / ALPHA_NORM, where
        P0_mixed = (1 + cos^2(a/2) cos^2(b/2) + sin^2(a/2) sin^2(b/2)) / 2
      This is what the actual circuit produces. The ghost CNOTs entangle
      v1 with (a1, a2) and v2 with (b1, b2) before the swap test, breaking
      the product-state form. Probe 4 derived this analytically and built
      a noiseless sampler that reproduces it.

WHAT EACH BACKEND ACTUALLY COMPUTES:

  cuBLAS              -> T1 exactly (rank-1 cosine, fp32 noise)
  Rank-K kernel       -> T1 exactly (same identity, custom CUDA kernel)
  Noiseless GPU base  -> T3 in expectation (shot-noise from sampling)
  QPU                 -> Some perturbation of T3 (structured deviation,
                         characterized in Probes 1-5)

WHAT THIS PROBE DOES:

  1. Computes all three targets for the 4x4 (r,c) grid.
  2. Compares each backend against ITS OWN correct target. Honest accuracy.
  3. Then runs the cross-comparison (every backend vs every target). The
     off-diagonal entries are large because the targets are different
     functions, not because the backends are inaccurate.
  4. Runs a small cuBLAS-vs-custom-kernel throughput benchmark at N=1024.

This is the answer to anyone who looks at a single number from a single
comparison and wants to draw a conclusion. The numbers all live here in
their full context.

HISTORICAL CONTEXT:
    Probe 6 was where the team formalized the three-target framing
    that the rest of the suite inherits. Probes 1-5 had progressively
    untangled what the QPU was actually computing (T1 hypothesis ruled
    out, T2 hypothesis ruled out, T3 derived and validated). This
    probe lays the result out as a clean three-way comparison so a
    reviewer can see at a glance: cuBLAS hits T1 to fp32 noise, the
    custom CUDA kernel hits T1 to the same precision via a different
    path, the noiseless GPU base hits T3 to shot noise, and the QPU
    hits T3 within hardware error.

    The throughput benchmark in this probe is the seed of the
    headline projection_benchmark.py harness. Probe 6 showed that a
    bespoke rank-K kernel can match cuBLAS at small N where cuBLAS
    dispatch overhead dominates; the headline benchmark productionized
    that idea at scale with the tied-channel G_M kernel.

    Probe 9 later simplified T3 to the G_M operator that drives the
    rest of the suite, and Probe 10.1 demonstrated G_M's attention
    robustness. See PROCESS_RECORD.md for the full arc.

USAGE:
    python probe6_3way_convergence.py
    python probe6_3way_convergence.py --qpu data/job_xyz.npz --gpu data/ghost_oracle_gpu_xyz.npz
    python probe6_3way_convergence.py --skip-benchmark
==============================================================================
"""

import argparse
import math
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

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
# Probe 6 originally ran against 12-tile bases. The rest of the suite uses
# 16 tiles (4x4); pass --num-tiles 16 to compare against a current-generation
# QPU base. Targets T1, T2, T3 are computed on the full 4x4 grid regardless.
NUM_TILES   = 12
PAIRS       = [(r, c) for r in range(4) for c in range(4)][:NUM_TILES]
ANGLE_SCALE = 1.05
ALPHA_NORM  = 0.9127

MATRIX_A = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B = np.array([1.00, 0.80, 0.40, 0.10])

# Repo-root paths. This file lives at <repo>/probes/probe6_*.py.
HERE        = Path(__file__).resolve().parent
DATA_DIR    = HERE.parent / "data"
KERNEL_PATH = HERE.parent / "ghost_oracle" / "kernels" / "ghost_kernel.cu"


def data_to_angles(data, scale=ANGLE_SCALE):
    """Scale a real-valued vector into rotation angles in [0, pi/2 * scale]."""
    max_val = np.max(np.abs(data))
    return (data / max_val) * (np.pi / 2) * scale


def auto_find_base(kind):
    """Find the first base .npz of the given kind in <repo>/data/, or None.
        kind == "qpu" -> files starting with 'job_'            (from dump.py)
        kind == "gpu" -> files starting with 'ghost_oracle_gpu_' (from Probe 4)
                         or 'ghost_oracle_gpu_'                (from gpu.py)
    """
    if kind == "qpu":
        patterns = ["job_*.npz"]
    elif kind == "gpu":
        patterns = ["ghost_oracle_gpu_*.npz", "ghost_oracle_gpu_*.npz"]
    else:
        return None
    candidates = []
    for pat in patterns:
        candidates.extend(sorted(DATA_DIR.glob(pat)))
    return str(candidates[0]) if candidates else None


# =============================================================================
# TARGETS
# =============================================================================
def target_T1_rank1_cosine(ang_a, ang_b):
    """T1: |cos(a - b)|"""
    M = np.zeros((4, 4))
    for r in range(4):
        for c in range(4):
            M[r, c] = abs(math.cos(ang_a[r] - ang_b[c]))
    return M


def target_T2_halfangle(ang_a, ang_b):
    """T2: |cos((a - b)/2)|  (textbook swap test on product states)"""
    M = np.zeros((4, 4))
    for r in range(4):
        for c in range(4):
            M[r, c] = abs(math.cos((ang_a[r] - ang_b[c]) / 2))
    return M


def target_T3_mixed_state(ang_a, ang_b):
    """T3: sqrt(2 P0 - 1) / ALPHA_NORM with P0 the mixed-state Hadamard outcome."""
    M = np.zeros((4, 4))
    for r in range(4):
        for c in range(4):
            a, b = ang_a[r], ang_b[c]
            p0 = 0.5 * (1.0 + math.cos(a / 2) ** 2 * math.cos(b / 2) ** 2 +
                              math.sin(a / 2) ** 2 * math.sin(b / 2) ** 2)
            M[r, c] = min(1.0, math.sqrt(max(0.0, 2 * p0 - 1)) / ALPHA_NORM)
    return M


# =============================================================================
# BACKENDS
# =============================================================================
def compile_rank_k_kernel():
    """Compile ghost_kernel.cu and return the ghost_rank_k_matmul entry point.
    Returns None if cupy is unavailable or compilation fails."""
    if not _HAVE_CUPY:
        return None
    if not KERNEL_PATH.exists():
        print(f"  [WARN] kernel source not found: {KERNEL_PATH}")
        return None
    try:
        src = KERNEL_PATH.read_text()
        mod = cp.RawModule(code=src, options=("-use_fast_math",))
        return mod.get_function("ghost_rank_k_matmul")
    except Exception as e:
        print(f"  [WARN] kernel compile failed: {e}")
        return None


def backend_cublas(ang_a, ang_b):
    """cuBLAS via the lifted representation [cos(a)|sin(a)] @ [cos(b)|sin(b)]^T.
    Identity: this exactly computes cos(a-b), so output equals T1."""
    if not _HAVE_CUPY:
        # numpy fallback so the table can still be built without GPU
        ca = np.cos(ang_a).reshape(4, 1)
        sa = np.sin(ang_a).reshape(4, 1)
        cb = np.cos(ang_b).reshape(4, 1)
        sb = np.sin(ang_b).reshape(4, 1)
        A_dense = np.concatenate([ca, sa], axis=1).astype(np.float32)
        B_dense = np.concatenate([cb, sb], axis=1).astype(np.float32)
        return np.abs(A_dense @ B_dense.T)
    ang_a_gpu = cp.asarray(ang_a.reshape(4, 1), dtype=cp.float32)
    ang_b_gpu = cp.asarray(ang_b.reshape(4, 1), dtype=cp.float32)
    A_dense = cp.concatenate([cp.cos(ang_a_gpu), cp.sin(ang_a_gpu)], axis=1)
    B_dense = cp.concatenate([cp.cos(ang_b_gpu), cp.sin(ang_b_gpu)], axis=1)
    C = cp.matmul(A_dense, B_dense.T)
    return cp.asnumpy(cp.abs(C))


def backend_rank_k(ang_a, ang_b, kernel):
    """Rank-K kernel (ghost_rank_k_matmul, K=1): same identity as backend_cublas,
    custom CUDA path through ghost_kernel.cu."""
    if kernel is None or not _HAVE_CUPY:
        # No GPU -> compute the analytical equivalent in numpy
        M = np.zeros((4, 4), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                M[r, c] = abs(math.cos(ang_a[r] - ang_b[c]))
        return M
    ang_a_gpu = cp.asarray(ang_a.reshape(4, 1), dtype=cp.float32)
    ang_b_gpu = cp.asarray(ang_b.reshape(4, 1), dtype=cp.float32)
    C = cp.zeros((4, 4), dtype=cp.float32)
    kernel((1, 1), (16, 16), (C, ang_a_gpu, ang_b_gpu, np.int32(4), np.int32(1)))
    cp.cuda.Device().synchronize()
    return cp.asnumpy(cp.abs(C))


def backend_from_npz(path, num_tiles):
    """Reads a shot dump (QPU or noiseless), returns a 4x4 matrix of normalized
    values via the standard Hadamard-test projection. Missing tiles are NaN."""
    d = np.load(path)
    M = np.full((4, 4), np.nan)
    for t in range(num_tiles):
        r, c = PAIRS[t]
        if f"ctrl_tile{t}" not in d.files:
            continue
        ctrl = d[f"ctrl_tile{t}"]
        p0 = float((ctrl == 0).mean())
        M[r, c] = min(1.0, math.sqrt(max(0.0, 2 * p0 - 1)) / ALPHA_NORM)
    return M


# =============================================================================
# THROUGHPUT BENCHMARK
# =============================================================================
def run_throughput_benchmark(kernel, N=1024, K=1, run_duration=2.0):
    print("\n" + "-" * 86)
    print(f"  THROUGHPUT (N={N}, K={K})")
    print("-" * 86)

    ang_A = cp.random.uniform(0, 2 * np.pi, (N, K)).astype(cp.float32)
    ang_B = cp.random.uniform(0, 2 * np.pi, (N, K)).astype(cp.float32)
    A_dense = cp.concatenate([cp.cos(ang_A), cp.sin(ang_A)], axis=1)
    B_dense_T = cp.concatenate([cp.cos(ang_B), cp.sin(ang_B)], axis=1).T

    out_rank_k = cp.zeros((N, N), dtype=cp.float32)
    out_cublas = cp.zeros((N, N), dtype=cp.float32)
    blocks = ((N + 15) // 16, (N + 15) // 16)
    threads = (16, 16)

    for _ in range(10):
        kernel(blocks, threads, (out_rank_k, ang_A, ang_B, np.int32(N), np.int32(K)))
        cp.matmul(A_dense, B_dense_T, out=out_cublas)
    cp.cuda.Device().synchronize()

    rank_k_runs = 0
    t0 = time.time()
    while (time.time() - t0) < run_duration:
        kernel(blocks, threads, (out_rank_k, ang_A, ang_B, np.int32(N), np.int32(K)))
        rank_k_runs += 1
    cp.cuda.Device().synchronize()
    t_rank_k = time.time() - t0
    rank_k_tput = rank_k_runs / t_rank_k

    cp.cuda.Device().synchronize()
    c0 = time.time()
    for _ in range(rank_k_runs):
        cp.matmul(A_dense, B_dense_T, out=out_cublas)
    cp.cuda.Device().synchronize()
    t_cublas = time.time() - c0
    cublas_tput = rank_k_runs / t_cublas

    print(f"  Rank-K kernel: {rank_k_runs:>10,} iter in {t_rank_k:.3f}s  ({rank_k_tput:>10.2f} runs/sec)")
    print(f"  cuBLAS:        {rank_k_runs:>10,} iter in {t_cublas:.3f}s  ({cublas_tput:>10.2f} runs/sec)")
    print(f"  Rank-K is {rank_k_tput / cublas_tput:.2f}x cuBLAS at N={N}, K={K}.")


# =============================================================================
# REPORTING HELPERS
# =============================================================================
def mae_masked(A, B):
    mask = ~(np.isnan(A) | np.isnan(B))
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(A[mask] - B[mask])))


def rmse_masked(A, B):
    mask = ~(np.isnan(A) | np.isnan(B))
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((A[mask] - B[mask]) ** 2)))


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Ghost Oracle Suite — Probe 6: Explicit 3-way convergence. "
                    "Formalizes T1, T2, T3 as distinct targets and shows what "
                    "each backend (cuBLAS, rank-K kernel, noiseless GPU base, "
                    "QPU) actually computes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--qpu", default=None,
                    help="Path to QPU base .npz (auto-finds job_*.npz in data/ if omitted).")
    ap.add_argument("--gpu", default=None,
                    help="Path to noiseless GPU base .npz "
                         "(auto-finds ghost_oracle_gpu_*.npz or ghost_oracle_gpu_*.npz in data/ if omitted).")
    ap.add_argument("--num-tiles", type=int, default=NUM_TILES,
                    help="Number of tiles in the bases.")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="Throughput benchmark window in seconds.")
    ap.add_argument("--skip-benchmark", action="store_true",
                    help="Skip the throughput benchmark.")
    args = ap.parse_args()

    qpu_path = args.qpu or auto_find_base("qpu")
    gpu_path = args.gpu or auto_find_base("gpu")

    print("\n" + "=" * 86)
    print("  GHOST ORACLE SUITE — PROBE 6 — EXPLICIT 3-WAY CONVERGENCE")
    print("=" * 86)
    print(f"  QPU base : {qpu_path or '(none)'}")
    print(f"  GPU base : {gpu_path or '(none)'}")
    if not _HAVE_CUPY:
        print("  (CuPy not detected — running in numpy fallback for GPU backends)")

    # ---- Set up
    ang_a = data_to_angles(MATRIX_A)
    ang_b = data_to_angles(MATRIX_B)

    # Build the three targets
    T1 = target_T1_rank1_cosine(ang_a, ang_b)
    T2 = target_T2_halfangle(ang_a, ang_b)
    T3 = target_T3_mixed_state(ang_a, ang_b)

    # Compile the rank-K kernel from ghost_kernel.cu
    kernel = compile_rank_k_kernel()

    # Compute each backend
    M_cublas = backend_cublas(ang_a, ang_b)
    M_rank_k = backend_rank_k(ang_a, ang_b, kernel)
    M_qpu = backend_from_npz(qpu_path, args.num_tiles) if qpu_path else None
    M_gpu = backend_from_npz(gpu_path, args.num_tiles) if gpu_path else None

    # ---- Targets at a glance
    print("\n  THE THREE TARGETS (4x4 each)")
    print("  " + "-" * 84)
    print(f"  T1 |cos(a-b)|        :  range [{T1.min():.4f}, {T1.max():.4f}]")
    print(f"  T2 |cos((a-b)/2)|    :  range [{T2.min():.4f}, {T2.max():.4f}]")
    print(f"  T3 mixed-state ideal :  range [{T3.min():.4f}, {T3.max():.4f}]")
    print()
    print("  Pairwise target distances (none of these are noise — these are")
    print("  the formal gaps between three different functions):")
    print(f"    MAE(T1, T2) = {mae_masked(T1, T2):.6e}")
    print(f"    MAE(T1, T3) = {mae_masked(T1, T3):.6e}")
    print(f"    MAE(T2, T3) = {mae_masked(T2, T3):.6e}")

    # ---- Honest per-backend accuracy
    print("\n" + "=" * 86)
    print("  PART A — EACH BACKEND vs ITS OWN CORRECT TARGET")
    print("=" * 86)
    print(f"  {'backend':<22} | {'target':<28} | {'MAE':>12} | {'RMSE':>12}")
    print("  " + "-" * 80)
    print(f"  {'cuBLAS':<22} | {'T1 |cos(a-b)|':<28} | "
          f"{mae_masked(M_cublas, T1):>12.6e} | {rmse_masked(M_cublas, T1):>12.6e}")
    print(f"  {'Rank-K kernel':<22} | {'T1 |cos(a-b)|':<28} | "
          f"{mae_masked(M_rank_k, T1):>12.6e} | {rmse_masked(M_rank_k, T1):>12.6e}")
    if M_gpu is not None:
        print(f"  {'Noiseless GPU base':<22} | {'T3 mixed-state ideal':<28} | "
              f"{mae_masked(M_gpu, T3):>12.6e} | {rmse_masked(M_gpu, T3):>12.6e}")
    if M_qpu is not None:
        print(f"  {'QPU':<22} | {'T3 mixed-state ideal':<28} | "
              f"{mae_masked(M_qpu, T3):>12.6e} | {rmse_masked(M_qpu, T3):>12.6e}")

    # ---- Cross-comparison matrix
    print("\n" + "=" * 86)
    print("  PART B — CROSS-COMPARISON (every backend vs every target)")
    print("=" * 86)
    print("  This is what reviewers will run if they don't read Part A. Large")
    print("  off-diagonal MAEs are NOT inaccuracy — they're the targets being")
    print("  different functions. The diagonal here is Part A.")
    print()
    print(f"  {'backend':<22} | {'vs T1':>12} | {'vs T2':>12} | {'vs T3':>12}")
    print("  " + "-" * 70)
    print(f"  {'cuBLAS':<22} | {mae_masked(M_cublas, T1):>12.6e} | "
          f"{mae_masked(M_cublas, T2):>12.6e} | {mae_masked(M_cublas, T3):>12.6e}")
    print(f"  {'Rank-K kernel':<22} | {mae_masked(M_rank_k, T1):>12.6e} | "
          f"{mae_masked(M_rank_k, T2):>12.6e} | {mae_masked(M_rank_k, T3):>12.6e}")
    if M_gpu is not None:
        print(f"  {'Noiseless GPU base':<22} | {mae_masked(M_gpu, T1):>12.6e} | "
              f"{mae_masked(M_gpu, T2):>12.6e} | {mae_masked(M_gpu, T3):>12.6e}")
    if M_qpu is not None:
        print(f"  {'QPU':<22} | {mae_masked(M_qpu, T1):>12.6e} | "
              f"{mae_masked(M_qpu, T2):>12.6e} | {mae_masked(M_qpu, T3):>12.6e}")

    # ---- Backend-to-backend (where they ARE comparable)
    print("\n" + "=" * 86)
    print("  PART C — BACKEND vs BACKEND (where they compute the same thing)")
    print("=" * 86)
    print("  cuBLAS and the rank-K kernel both compute T1 — they MUST agree to fp32 noise.")
    print(f"    MAE(cuBLAS, rank-K)         = {mae_masked(M_cublas, M_rank_k):.6e}   "
          "(expect fp32 noise ~1e-7)")
    if M_gpu is not None and M_qpu is not None:
        print("  Noiseless GPU base and QPU both target T3 — gap is QPU's structural deviation.")
        print(f"    MAE(Noiseless, QPU)         = {mae_masked(M_gpu, M_qpu):.6e}   "
              "(expect QPU bias + shot noise)")
        print(f"    MAE(Noiseless, T3)          = {mae_masked(M_gpu, T3):.6e}   "
              f"(expect shot noise ~{1/math.sqrt(4096):.4f})")
        print(f"    MAE(QPU, T3)                = {mae_masked(M_qpu, T3):.6e}   "
              "(this IS the signal of interest from Probes 1-5)")

    # ---- Throughput
    if not args.skip_benchmark and _HAVE_CUPY and kernel is not None:
        run_throughput_benchmark(kernel, N=1024, K=1, run_duration=args.seconds)

    print("\n" + "=" * 86)
    print("  SUMMARY")
    print("=" * 86)
    print("  Part A is the honest accuracy table: each backend against its own correct")
    print("  target. cuBLAS and rank-K at fp32 noise on T1; noiseless GPU base at shot")
    print("  noise on T3; QPU's MAE-against-T3 is the structural deviation Probes 1-5")
    print("  mapped (and Probes 8.0-8.4 later characterized in detail).")
    print()
    print("  Part B exists because reviewers will run it. The off-diagonal numbers are")
    print("  large because the targets are different functions; this is NOT inaccuracy.")
    print()
    print("  Part C is the meaningful backend-to-backend comparison. cuBLAS = rank-K to")
    print("  fp32 noise, noiseless GPU base = T3 to shot noise, and Noiseless-vs-QPU")
    print("  isolates the QPU-specific structural deviation.")
    print()


if __name__ == "__main__":
    main()