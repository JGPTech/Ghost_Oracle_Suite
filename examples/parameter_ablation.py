#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — PARAMETER ABLATION
==============================================================================
A sister script to projection_benchmark.py. Instead of sweeping matrix sizes,
this script locks a target shape and performs 1D sensitivity sweeps across:
    1. Embedding Dimension (d)
    2. Jitter Scale (noise)
    3. Attack Magnitude (coherent same-dim outlier strength)
    4. Attack Fraction (sparsity of the attack)

Usage:
    python parameter_ablation.py
    python parameter_ablation.py --N 2048 --M 2048
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
# CONFIGURATION & CONSTANTS
# =============================================================================
ANGLE_SCALE = 1.05
ALPHA_NORM = 0.9127

MATRIX_A_ORIG = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B_ORIG = np.array([1.00, 0.80, 0.40, 0.10])
RTX3090_BLOCK = (32, 8)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
KERNEL_PATH = HERE.parent / "ghost_oracle" / "kernels" / "ghost_kernel.cu"

# =============================================================================
# CORE UTILITIES (Shared with projection_benchmark.py)
# =============================================================================
def data_to_angles(data, scale=ANGLE_SCALE):
    max_val = np.max(np.abs(data))
    return (data / max_val) * (np.pi / 2) * scale

ORIG_A = data_to_angles(MATRIX_A_ORIG)
ORIG_B = data_to_angles(MATRIX_B_ORIG)

def compile_kernels():
    if not _HAVE_CUPY:
        sys.exit("[FATAL] cupy not available.")
    if not KERNEL_PATH.exists():
        sys.exit(f"[FATAL] kernel source not found: {KERNEL_PATH}")
    src = KERNEL_PATH.read_text()
    mod = cp.RawModule(code=src, options=("-use_fast_math",))
    return mod.get_function("tied_streaming_perdim")

def detect_gpu_name():
    return cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if _HAVE_CUPY else "no-gpu"

def clear_gpu_memory():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def auto_find_base(kind):
    patterns = ["job_*.npz"] if kind == "qpu" else ["ghost_oracle_gpu_*.npz"]
    candidates = []
    for pat in patterns:
        candidates.extend(sorted(DATA_DIR.glob(pat)))
    return str(candidates[0]) if candidates else None

def load_base(path):
    d = np.load(path)
    num_tiles = int(d["num_tiles"])
    ctrl = {t: d[f"ctrl_tile{t}"] for t in range(num_tiles)}
    ghost = {t: d[f"ghost_tile{t}"] for t in range(num_tiles)}
    return ctrl, ghost, num_tiles

def build_bucket_counts(ctrl_dict, ghost_dict, num_tiles):
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
    best_idx, best_imbalance = 0, float("inf")
    for t in range(counts.shape[0]):
        imb = abs(counts[t, :, :, 0].sum() - counts[t, :, :, 1].sum())
        if imb < best_imbalance:
            best_imbalance, best_idx = imb, t
    return best_idx

def pairs_for_num_tiles(num_tiles, matrix_size=4):
    return [(r, c) for r in range(matrix_size) for c in range(matrix_size)][:num_tiles]

def phase_lift_perdim(X):
    return ((math.pi / 2) * (1.0 + np.tanh(X / 3.0))).astype(np.float32)

def make_attacked_jittered_embeddings(N, M, d, jitter_scale, attack_fraction, magnitude, seed):
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
    return X_Q, X_K

def cublas_DP_full(X_Q, X_K):
    d_q = cp.asarray(X_Q)
    d_k = cp.asarray(X_K)
    S = cp.matmul(d_q, d_k.T) / math.sqrt(X_Q.shape[1])
    cp.cuda.Device().synchronize()
    return S

def tied_streaming_perdim_run(theta_Q, theta_K, counts18, orig_a, orig_b, kernel_tied, block_size=256):
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
        grid, (block_size, 1, 1),
        (d_counts, np.float32(orig_a), np.float32(orig_b), d_tq, d_tk, 
         d_idx, d_score, d_proj, d_agree, np.int32(N), np.int32(M), 
         np.int32(d), np.float32(ALPHA_NORM))
    )
    cp.cuda.Device().synchronize()
    return d_idx

def top1_accuracy_from_matrix(S, gt):
    if isinstance(S, cp.ndarray): S = S.get()
    return float(np.mean(np.argmax(S, axis=1) == gt))

def top1_accuracy_from_idx(idx, gt):
    if isinstance(idx, cp.ndarray): idx = idx.get()
    return float(np.mean(idx == gt))

def hline(c="-", w=100): print(c * w)
def section(t, w=100):
    print()
    hline("=", w)
    print(f"  {t}")
    hline("=", w)

# =============================================================================
# ABLATION RUNNER
# =============================================================================
def run_ablation(sweep_name, param_name, param_values, default_kwargs, kernel_tied, counts18, orig_a, orig_b):
    section(f"SWEEP: Varying {sweep_name}")
    header = f"  {param_name:<15} | {'cuBLAS Acc':>15} | {'G_M Tied Acc':>15} | {'Delta':>15}"
    print(header)
    hline("-", len(header))

    for val in param_values:
        kwargs = default_kwargs.copy()
        kwargs[param_name] = val
        
        N, M, d = kwargs['N'], kwargs['M'], kwargs['d']
        jitter = kwargs['jitter_scale']
        atk_frac = kwargs['attack_fraction']
        mag = kwargs['magnitude']

        clear_gpu_memory()
        X_Q, X_K = make_attacked_jittered_embeddings(N, M, d, jitter, atk_frac, mag, seed=42)
        gt = np.arange(min(N, M))
        
        # cuBLAS
        try:
            cu_out = cublas_DP_full(X_Q, X_K)
            cu_acc = top1_accuracy_from_matrix(cu_out, gt)
        except OutOfMemoryError:
            cu_acc = float('nan')

        # G_M Tied
        try:
            theta_Q = phase_lift_perdim(X_Q)
            theta_K = phase_lift_perdim(X_K)
            idx_out = tied_streaming_perdim_run(theta_Q, theta_K, counts18, orig_a, orig_b, kernel_tied)
            gm_acc = top1_accuracy_from_idx(idx_out, gt)
        except OutOfMemoryError:
            gm_acc = float('nan')

        cu_str = f"{cu_acc:.2%}" if not math.isnan(cu_acc) else "OOM"
        gm_str = f"{gm_acc:.2%}" if not math.isnan(gm_acc) else "OOM"
        delta = f"{(gm_acc - cu_acc):+.2%}" if not (math.isnan(cu_acc) or math.isnan(gm_acc)) else "--"

        # Formatting param display
        if isinstance(val, float) and val > 1: val_str = f"{val:.1f}"
        elif isinstance(val, float): val_str = f"{val:.3f}"
        else: val_str = str(val)

        print(f"  {val_str:<15} | {cu_str:>15} | {gm_str:>15} | {delta:>15}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Ghost Oracle Suite — parameter ablation")
    p.add_argument("--qpu", default=None, help="Path to QPU base .npz")
    p.add_argument("--gpu", default=None, help="Path to GPU base .npz")
    p.add_argument("--N", type=int, default=1024, help="Queries")
    p.add_argument("--M", type=int, default=1024, help="Keys")
    args = p.parse_args()

    if not _HAVE_CUPY: sys.exit("[FATAL] cupy required.")

    qpu_path = args.qpu or auto_find_base("qpu")
    if not qpu_path: sys.exit("[FATAL] no QPU base found.")
    
    qpu_ctrl, qpu_ghost, qpu_num_tiles = load_base(qpu_path)
    qpu_counts = build_bucket_counts(qpu_ctrl, qpu_ghost, qpu_num_tiles)
    rep_idx = representative_tile(qpu_counts)
    counts18 = qpu_counts[rep_idx]
    
    qpu_pairs = pairs_for_num_tiles(qpu_num_tiles, matrix_size=4)
    orig_a = float(ORIG_A[qpu_pairs[rep_idx][0]])
    orig_b = float(ORIG_B[qpu_pairs[rep_idx][1]])

    kernel_tied = compile_kernels()

    section(f"GHOST ORACLE SUITE — PARAMETER ABLATION (Shape: {args.N}x{args.M})")
    print(f"  GPU    : {detect_gpu_name()}")
    print(f"  Base   : {qpu_path}")

    # The baseline setup we perturb
    defaults = {
        'N': args.N, 'M': args.M, 'd': 64,
        'jitter_scale': 0.3,
        'attack_fraction': 0.05,
        'magnitude': 50.0
    }

    # Sweep 1: Dimension
    run_ablation("Dimension (d)", "d", [8, 16, 32, 64, 128, 256], defaults, kernel_tied, counts18, orig_a, orig_b)

    # Sweep 2: Jitter Scale
    run_ablation("Jitter Scale", "jitter_scale", [0.0, 0.1, 0.3, 0.5, 0.8, 1.0], defaults, kernel_tied, counts18, orig_a, orig_b)

    # Sweep 3: Attack Magnitude
    run_ablation("Attack Magnitude", "magnitude", [0.0, 5.0, 20.0, 50.0, 100.0, 500.0], defaults, kernel_tied, counts18, orig_a, orig_b)

    # Sweep 4: Attack Fraction (Sparsity)
    run_ablation("Attack Fraction", "attack_fraction", [0.01, 0.05, 0.10, 0.20, 0.50], defaults, kernel_tied, counts18, orig_a, orig_b)

    section("DONE")

if __name__ == "__main__":
    main()