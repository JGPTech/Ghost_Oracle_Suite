#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — PROBE 1 — IDENTITY BRIDGE
==============================================================================
Tests whether the QPU half-angle Hadamard projection is exactly the rank-1
case of the rank-K cosine kernel under the current normalization.

Three paths, same angles:
  (A) QPU:        norm_val[r,c] = sqrt(2*P(0)[r,c] - 1) / ALPHA_NORM
  (B) GPU K=1:    rank_k_kernel with half-angles, abs of output
  (C) Analytical: |cos((theta_a[r] - theta_b[c]) / 2)|  in float64

Pairwise comparisons:
  - QPU vs analytical  : tells us what the QPU is actually measuring
  - GPU vs analytical  : sanity check the kernel at K=1
  - QPU vs GPU         : the identity claim
  - QPU residual map   : per-tile residual, looking for spatial structure
                         that ALPHA_NORM is absorbing

HISTORICAL CONTEXT:
    This was the first probe in the trajectory. At the time of writing,
    the team believed the QPU was computing T2 = |cos((a-b)/2)| (the
    textbook half-angle Hadamard form). This probe tests that belief.

    The original 12-tile job that motivated the rest of the probe series
    reported QPU vs analytical MAE ≈ 0.19 with -0.18 bias and max diff
    0.50 — well beyond shot noise, with bias dominating spatial spread.
    Running this script against the sample QPU base shipped in data/
    will produce a different residual magnitude (typically smaller, as
    later jobs were better calibrated), but the qualitative finding is
    the same: the QPU does not match |cos((a-b)/2)| at shot-noise
    precision, and ALPHA_NORM behaves like a global gain rather than a
    geometry correction.

    That finding is what motivated Probes 2 through 7 — eventually
    leading to Probe 4's discovery that the actual target is T3 (the
    ghost-CNOT mixed-state formula), which Probe 9 simplified to the
    G_M operator that drives the rest of the suite. See
    PROCESS_RECORD.md for the full arc.

USAGE:
    python probe1_identity_bridge.py --base data/job_xyz.npz
==============================================================================
"""

import argparse
import math
import sys
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
# CONFIGURATION
# =============================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KERNEL_PATH = (
    Path(__file__).resolve().parent.parent / "kernels" / "ghost_kernel.cu"
)

# Probe 1 originally ran against a 12-tile job. The rest of the suite uses
# 16 tiles (4x4); pass --num-tiles 16 to compare against a current-generation
# QPU base. The findings reported in the docstring and PROCESS_RECORD are
# from the historical 12-tile run.
NUM_TILES = 12
PAIRS = [(r, c) for r in range(4) for c in range(4)][:NUM_TILES]

ANGLE_SCALE = 1.05
ALPHA_NORM = 0.9127

MATRIX_A = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B = np.array([1.00, 0.80, 0.40, 0.10])


def data_to_angles(data, scale=ANGLE_SCALE):
    """Scale a real-valued vector into rotation angles in [0, pi/2 * scale]."""
    max_val = np.max(np.abs(data))
    return (data / max_val) * (np.pi / 2) * scale


def auto_find_base(kind):
    """Find the first base .npz of the given kind in <repo>/data/, or None.
        kind == "qpu" -> files starting with 'job_'            (from dump.py)
        kind == "gpu" -> files starting with 'noiseless_base_' (from Probe 4)
                         or 'ghost_oracle_gpu_'                (from gpu.py)
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


def gpu_rank1_halfangle(angles_a_full, angles_b_full):
    """
    Evaluate the GPU rank-K kernel at K=1 on half-angles, returning
    |cos((a-b)/2)|. Uses ghost_rank_k_matmul from the consolidated kernel.
    """
    if not _HAVE_CUPY:
        # CPU fallback (just numpy — kernel parity isn't the point if no GPU)
        a = angles_a_full / 2.0
        b = angles_b_full / 2.0
        out = np.cos(a[:, None] - b[None, :])
        return np.abs(out)

    if not KERNEL_PATH.exists():
        raise FileNotFoundError(f"Kernel source not found: {KERNEL_PATH}")

    N = len(angles_a_full)
    K = 1
    ang_A = cp.asarray((angles_a_full / 2.0).reshape(N, K), dtype=cp.float32)
    ang_B = cp.asarray((angles_b_full / 2.0).reshape(N, K), dtype=cp.float32)
    out_C = cp.zeros((N, N), dtype=cp.float32)

    src = KERNEL_PATH.read_text()
    mod = cp.RawModule(code=src, options=("-use_fast_math",))
    kernel = mod.get_function("ghost_rank_k_matmul")
    threads = (16, 16)
    blocks = ((N + 15) // 16, (N + 15) // 16)
    kernel(blocks, threads, (out_C, ang_A, ang_B, np.int32(N), np.int32(K)))
    cp.cuda.Device().synchronize()
    return np.abs(cp.asnumpy(out_C))


def analytical_halfangle(angles_a_full, angles_b_full):
    """Float64 reference: |cos((a-b)/2)|"""
    a = angles_a_full.astype(np.float64)
    b = angles_b_full.astype(np.float64)
    return np.abs(np.cos((a[:, None] - b[None, :]) / 2.0))


# =============================================================================
# QPU READOUT
# =============================================================================
def load_qpu_manifold(npz_path, num_tiles, pairs):
    """
    Load QPU shot data from a dump.py-produced .npz and project each tile
    to its normalized control-measurement value. Returns a 4x4 matrix
    (NaN where no tile covers an entry) plus per-tile shot counts.
    """
    print(f"[LOAD] {npz_path}")
    data = np.load(npz_path)
    M = np.full((4, 4), np.nan)
    shots_per_tile = {}
    for t_idx in range(num_tiles):
        r, c = pairs[t_idx]
        ctrl = data[f"ctrl_tile{t_idx}"]
        n_shots = len(ctrl)
        p0 = float((ctrl == 0).mean())
        raw_val = math.sqrt(max(0.0, 2.0 * p0 - 1.0))
        norm_val = min(1.0, raw_val / ALPHA_NORM)
        M[r, c] = norm_val
        shots_per_tile[t_idx] = n_shots
    return M, shots_per_tile


# =============================================================================
# COMPARISON
# =============================================================================
def compare(name_a, A, name_b, B, mask):
    diff = (A - B)[mask]
    abs_diff = np.abs(diff)
    print(f"\n  [{name_a}  vs  {name_b}]")
    print(f"    MAE          : {np.mean(abs_diff):.6e}")
    print(f"    RMSE         : {np.sqrt(np.mean(diff**2)):.6e}")
    print(f"    Max |diff|   : {np.max(abs_diff):.6e}")
    print(f"    Signed mean  : {np.mean(diff):+.6e}   (bias)")
    return diff


def main():
    parser = argparse.ArgumentParser(
        description="Probe 1 — Identity Bridge: tests whether the QPU output "
        "matches |cos((a-b)/2)| (the T2 hypothesis).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Path to QPU base .npz from dump.py. Auto-finds in data/ if omitted.",
    )
    parser.add_argument(
        "--num-tiles",
        type=int,
        default=NUM_TILES,
        help="Number of tiles to compare (must match the base).",
    )
    args = parser.parse_args()

    base_path = args.base or auto_find_base("qpu")
    if not base_path:
        sys.exit(f"[FATAL] no QPU base found. Pass --base or put a job_*.npz in {DATA_DIR}/")

    angles_a = data_to_angles(MATRIX_A)
    angles_b = data_to_angles(MATRIX_B)

    print("\n" + "=" * 78)
    print("  PROBE 1 — IDENTITY BRIDGE")
    print("=" * 78)
    print(f"  Angles A: {angles_a}")
    print(f"  Angles B: {angles_b}")

    # ---- Path A: QPU
    M_qpu, shots = load_qpu_manifold(base_path, args.num_tiles, PAIRS)
    M_gpu = gpu_rank1_halfangle(angles_a, angles_b)
    M_ref = analytical_halfangle(angles_a, angles_b)

    # Mask: only compare entries the QPU actually measured
    mask = ~np.isnan(M_qpu)

    print("\n" + "-" * 78)
    print("  PER-TILE TABLE")
    print("-" * 78)
    print(f"  {'(r,c)':<8}{'QPU':>12}{'GPU K=1':>14}{'Analytical':>14}"
          f"{'QPU-Ref':>12}")
    print("-" * 78)
    for r, c in PAIRS:
        print(f"  ({r},{c})  "
              f"{M_qpu[r,c]:>10.6f}  "
              f"{M_gpu[r,c]:>12.6f}  "
              f"{M_ref[r,c]:>12.6f}  "
              f"{M_qpu[r,c]-M_ref[r,c]:>+10.6f}")

    print("\n" + "-" * 78)
    print("  PAIRWISE METRICS (over measured tiles only)")
    print("-" * 78)
    diff_qpu_ref = compare("QPU       ", M_qpu, "Analytical", M_ref, mask)
    compare("GPU K=1   ", M_gpu, "Analytical", M_ref, mask)
    compare("QPU       ", M_qpu, "GPU K=1   ", M_gpu, mask)

    # ---- ALPHA_NORM diagnostic
    # If ALPHA_NORM is a pure global gain, the QPU residual should look like a
    # constant bias. If it has spatial structure across (r,c), ALPHA_NORM is
    # absorbing geometry.
    print("\n" + "-" * 78)
    print("  ALPHA_NORM DIAGNOSTIC")
    print("-" * 78)
    # What ALPHA_NORM would have to be, per tile, for QPU to exactly match
    # analytical: norm_val = sqrt(2 p0 - 1) / alpha = ref
    # So alpha_implied = ALPHA_NORM * (M_qpu_raw / M_ref) where
    # M_qpu_raw = M_qpu * ALPHA_NORM.
    print("  Per-tile residual structure (QPU - Analytical):")
    resid = (M_qpu - M_ref)
    print(f"    Mean residual (bias)      : {np.nanmean(resid):+.6e}")
    print(f"    Std of residual           : {np.nanstd(resid):.6e}")
    print(f"    Range                     : [{np.nanmin(resid):+.4e}, "
          f"{np.nanmax(resid):+.4e}]")
    if np.nanstd(resid) > 3 * abs(np.nanmean(resid)):
        print("    -> Spatial structure dominates over bias.")
        print("    -> ALPHA_NORM is likely absorbing geometry, not just gain.")
    else:
        print("    -> Bias dominates over spatial spread.")
        print("    -> ALPHA_NORM looks consistent with a pure global gain.")

    print("\n" + "=" * 78)
    print("  VERDICT NOTES")
    print("=" * 78)
    print("  - If GPU K=1 vs Analytical is at fp32 noise (~1e-7), the kernel")
    print("    identity is confirmed.")
    print("  - If QPU vs GPU K=1 sits within ~1/sqrt(shots) of zero (shot")
    print(f"    noise floor ~ {1/math.sqrt(max(shots.values())):.4f}), the QPU is")
    print("    measuring the rank-1 form to within statistics.")
    print("  - Anything beyond shot noise in the residual is signal for the")
    print("    later probes to chase.")
    print()


if __name__ == "__main__":
    main()
