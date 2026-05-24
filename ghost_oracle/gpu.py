#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — NOISELESS GPU BASE BUILDER
==============================================================================
Constructs a noiseless reference base on the GPU using the closed-form GHZ
sampler from ghost_kernel.cu (Section 5). Output is byte-compatible with the
QPU base dump from dump.py: identical key set, dtypes, and shapes. Any probe
or projection script consumes either file without modification.

Per-tile circuit (qubits: a1, v1, a2, ctrl, b1, v2, b2):
    1. Ry(theta_a) on v1; Ry(theta_b) on v2
    2. CNOT(v1 -> a1); CNOT(v1 -> a2); CNOT(v2 -> b1); CNOT(v2 -> b2)
    3. H on ctrl
    4. CSWAP(ctrl; v1, v2)
    5. H on ctrl
    6. Measure {ctrl, a1, a2, b1, b2}

After the ghost CNOTs, {v1, a1, a2} and {v2, b1, b2} are each in a GHZ
block, so a1 == a2 and b1 == b2 deterministically. The 32-bin joint
distribution collapses to 8 bins with independent factors:

    P(ctrl=1) = 0.5 * (1 - cos^2(a/2)cos^2(b/2) - sin^2(a/2)sin^2(b/2))
    P(a=1)    = sin^2(a/2)
    P(b=1)    = sin^2(b/2)

This is the T3 target derived in Section 3.3 of the paper and in Probe 4.
The CUDA kernel samples from this closed form directly — no statevector
simulation, three curand_uniform draws per shot.

OUTPUT (.npz keys, matching dump.py exactly):
    job_id        : str    (here: "ghost_oracle_gpu_seed{seed}")
    num_tiles     : int
    ctrl_tile{t}  : uint8, shape (n_shots,)
    ghost_tile{t} : uint8, shape (n_shots, 4)   columns = [a1, a2, b1, b2]

Usage:
    python gpu.py
    python gpu.py --shots 4096 --seed 42
    python gpu.py --verify                  # print full per-tile table
    python gpu.py --out custom.npz

Output:
    <repo>/data/ghost_oracle_gpu_<shots>shots_seed<seed>.npz   (default)
==============================================================================
"""

import argparse
import math
import secrets
import sys
from pathlib import Path

import numpy as np

try:
    import cupy as cp

    _HAVE_CUPY = True
except Exception:
    cp = None
    _HAVE_CUPY = False

# =============================================================================
# CONFIGURATION (must match qpu.py)
# =============================================================================
ANGLE_SCALE = 1.05

# Repo-root paths. This file lives at <repo>/ghost_oracle/gpu.py.
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
KERNEL_PATH = HERE / "kernels" / "ghost_kernel.cu"


def data_to_angles(data, scale=ANGLE_SCALE):
    """Scale a real-valued vector into rotation angles in [0, pi/2 * scale]."""
    max_val = np.max(np.abs(data))
    return (data / max_val) * (np.pi / 2) * scale


# =============================================================================
# KERNEL LOADING
# =============================================================================
def compile_kernel():
    """Compile ghost_kernel.cu and return the ghost_ghz_sample entry point."""
    if not _HAVE_CUPY:
        sys.exit("[FATAL] cupy not available — install cupy and a CUDA toolchain.")
    if not KERNEL_PATH.exists():
        sys.exit(f"[FATAL] kernel source not found: {KERNEL_PATH}")
    try:
        src = KERNEL_PATH.read_text()
        mod = cp.RawModule(code=src, options=("-use_fast_math",))
        return mod.get_function("ghost_ghz_sample")
    except Exception as e:
        sys.exit(f"[FATAL] kernel compile failed: {e}")


# =============================================================================
# BASE BUILDER (importable)
# =============================================================================
def make_pairs(matrix_size):
    """All (r, c) entries of an N x N output matrix, in row-major order."""
    return [(r, c) for r in range(matrix_size) for c in range(matrix_size)]


def make_matrix_family(matrix_size):
    """
    Generate the parameterized A, B input vectors for a given matrix size.
    Endpoints match the canonical 4-vector pipeline; intermediate values
    are linearly interpolated. For N=4 this returns exactly the qpu.py
    DEFAULT_MATRIX_A and DEFAULT_MATRIX_B values.
    """
    A = np.linspace(0.25, 1.00, matrix_size)
    B = np.linspace(1.00, 0.10, matrix_size)
    return A, B


def analytical_marginals(theta_a, theta_b):
    """Closed-form marginals for one tile (Probe 4 derivation)."""
    ca = math.cos(theta_a / 2) ** 2
    sa = math.sin(theta_a / 2) ** 2
    cb = math.cos(theta_b / 2) ** 2
    sb = math.sin(theta_b / 2) ** 2
    swap_exp = ca * cb + sa * sb
    expected_p0 = 0.5 * (1.0 + swap_exp)
    return {
        "expected_p0": expected_p0,
        "expected_a": sa,
        "expected_b": sb,
    }


def build_base(matrix_size=4, n_shots=4096, seed=None, verbose=False, kernel=None):
    """
    Construct a noiseless Ghost Oracle GPU base for an N x N output matrix.

    Parameters
    ----------
    matrix_size : int
        Output side length N. The base contains N**2 tiles covering all
        (r, c) entries.
    n_shots : int
        Shots sampled per tile.
    seed : int or None
        cuRAND seed. If None, a cryptographic random seed is drawn.
    verbose : bool
        If True, prints per-tile analytical-vs-empirical diagnostics.
    kernel : cupy RawKernel or None
        Pre-compiled ghost_ghz_sample kernel. If None, compiles it.

    Returns
    -------
    data : dict
        Schema (matches dump.py exactly):
            job_id        : str
            num_tiles     : int  (= matrix_size**2)
            ctrl_tile{t}  : uint8 (n_shots,)
            ghost_tile{t} : uint8 (n_shots, 4)   columns = [a1, a2, b1, b2]
        Plus diagnostic keys (stripped before on-disk write by CLI):
            max_p0_residual : float    sanity check, ~shot-noise
            max_anti        : float    ghost anti-correlation, should be 0 exactly
            seed            : int      cuRAND seed used
    """
    if kernel is None:
        kernel = compile_kernel()

    if seed is None:
        seed = secrets.randbits(63)

    pairs = make_pairs(matrix_size)
    num_tiles = len(pairs)
    A, B = make_matrix_family(matrix_size)
    angles_a = data_to_angles(A)
    angles_b = data_to_angles(B)

    # Per-tile angle arrays (one entry per tile, indexed by tile_idx).
    tile_angles_a = np.array([angles_a[r] for (r, c) in pairs], dtype=np.float32)
    tile_angles_b = np.array([angles_b[c] for (r, c) in pairs], dtype=np.float32)

    if verbose:
        print(f"\n  ANGLES_A: {angles_a}")
        print(f"  ANGLES_B: {angles_b}")
        print("\n" + "-" * 92)
        print(
            f"  {'tile':>4} {'(r,c)':>7} {'a':>7} {'b':>7} "
            f"{'P(ctrl=0)':>11} {'(target)':>10} "
            f"{'P(a=1)':>9} {'(target)':>9} "
            f"{'P(b=1)':>9} {'(target)':>9} "
            f"{'a_anti':>9} {'b_anti':>9}"
        )
        print("-" * 92)

    # ---------- GPU launch ----------
    d_angles_a = cp.asarray(tile_angles_a)
    d_angles_b = cp.asarray(tile_angles_b)
    d_ctrl = cp.empty((num_tiles, n_shots), dtype=cp.uint8)
    d_ghost = cp.empty((num_tiles, n_shots, 4), dtype=cp.uint8)

    block = (min(256, max(1, n_shots)), 1, 1)
    grid = (num_tiles, 1, 1)

    kernel(
        grid,
        block,
        (
            d_angles_a,
            d_angles_b,
            np.uint64(seed),
            np.int32(n_shots),
            np.int32(num_tiles),
            d_ctrl,
            d_ghost,
        ),
    )
    cp.cuda.Device().synchronize()

    ctrl_host = cp.asnumpy(d_ctrl)
    ghost_host = cp.asnumpy(d_ghost)

    # ---------- Pack into output dict (dump.py-compatible schema) ----------
    data = {
        "job_id": f"ghost_oracle_gpu_seed{seed}",
        "num_tiles": num_tiles,
    }

    max_anti = 0.0
    max_p0_residual = 0.0

    for t in range(num_tiles):
        r, c = pairs[t]
        ctrl = ctrl_host[t]
        ghost = ghost_host[t]

        data[f"ctrl_tile{t}"] = ctrl
        data[f"ghost_tile{t}"] = ghost

        # Per-tile sanity (empirical vs analytical)
        diag = analytical_marginals(angles_a[r], angles_b[c])
        emp_p0 = float((ctrl == 0).mean())
        emp_a1 = float(ghost[:, 0].mean())
        emp_a2 = float(ghost[:, 1].mean())
        emp_b1 = float(ghost[:, 2].mean())
        emp_b2 = float(ghost[:, 3].mean())
        a_anti = float((ghost[:, 0] != ghost[:, 1]).mean())
        b_anti = float((ghost[:, 2] != ghost[:, 3]).mean())

        max_anti = max(max_anti, a_anti, b_anti)
        max_p0_residual = max(max_p0_residual, abs(emp_p0 - diag["expected_p0"]))

        if verbose:
            print(
                f"  {t:>4} ({r},{c})  "
                f"{angles_a[r]:>6.4f} {angles_b[c]:>6.4f}  "
                f"{emp_p0:>9.6f} {diag['expected_p0']:>10.6f}  "
                f"{emp_a1:>7.4f}  {diag['expected_a']:>7.4f}  "
                f"{emp_b1:>7.4f}  {diag['expected_b']:>7.4f}  "
                f"{a_anti:>7.4f}  {b_anti:>7.4f}"
            )

    data["max_p0_residual"] = max_p0_residual
    data["max_anti"] = max_anti
    data["seed"] = seed
    return data


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — noiseless GPU base builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--shots", type=int, default=4096, help="Shots per tile (matches QPU)."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="cuRAND seed (defaults to cryptographic random).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .npz path (auto-named under data/ if omitted).",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Print full per-tile analytical-vs-empirical table.",
    )
    p.add_argument(
        "--matrix-size",
        type=int,
        default=4,
        help="Output matrix side length N. The base contains N^2 tiles.",
    )
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    if not _HAVE_CUPY:
        sys.exit("[FATAL] cupy required.")

    print(f"\n{'=' * 78}")
    print("  GHOST ORACLE SUITE — NOISELESS GPU BASE BUILDER")
    print(f"{'=' * 78}")
    print(f"  Matrix : {args.matrix_size} x {args.matrix_size}")
    print(f"  Tiles  : {args.matrix_size**2}")
    print(f"  Shots  : {args.shots} per tile")

    try:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        gpu_name = "unknown-gpu"
    print(f"  GPU    : {gpu_name}")

    print("\n[BUILD] Compiling ghost_kernel.cu (Section 5: ghost_ghz_sample)...")
    kernel = compile_kernel()

    print("[BUILD] Launching GHZ sampler...")
    data = build_base(
        matrix_size=args.matrix_size,
        n_shots=args.shots,
        seed=args.seed,
        verbose=args.verify,
        kernel=kernel,
    )
    seed = data["seed"]
    print(f"  Seed   : {seed}")

    # Strip diagnostic keys from the on-disk payload (dump.py-compatible only).
    max_p0_residual = data.pop("max_p0_residual")
    max_anti = data.pop("max_anti")
    data.pop("seed")

    if args.out:
        out_path = Path(args.out)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DATA_DIR / f"ghost_oracle_gpu_{args.shots}shots_seed{seed}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)

    # Shot-noise floor for n_shots Bernoulli samples is ~0.5/sqrt(n_shots).
    shot_floor = 0.5 / math.sqrt(args.shots)

    print("\n  Empirical sanity:")
    print(
        f"    Max |P(ctrl=0) - T3|        : {max_p0_residual:.2e}   "
        f"(shot-noise floor ~ {shot_floor:.2e})"
    )
    print(f"    Max ghost anti-correlation  : {max_anti:.2e}   (target: 0 exactly)")

    print(f"\n{'=' * 78}")
    print("  BASE COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Output    : {out_path}")
    print(f"  Tiles     : {args.matrix_size**2}")
    print(f"  Shots     : {args.shots}")
    print(f"  Seed      : {seed}")
    print("\n  Use --verify to see the full per-tile analytical table.")
    print("  Byte-compatible with QPU dumps from dump.py.")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
