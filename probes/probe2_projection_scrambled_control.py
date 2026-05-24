#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — PROBE 2 — PROJECTION-SCRAMBLED CONTROL
==============================================================================
Tests whether the Benford/recursive structure in the QPU manifold stream is
COUPLED to the intended projection geometry or COEXISTENT with it.

Three projections, same shot stream:
  (P1) INTENDED   : half-angle Hadamard projection with the angles the
                    circuit was prepared with (the original "right" geometry)
  (P2) SCRAMBLED  : same projection FORM, but angles drawn from an
                    unrelated distribution (the "wrong" geometry)
  (P3) RANDOMIZED : ensemble of M scrambled projections, gives a
                    null distribution for the Benford score under
                    wrong-geometry projection

INTERPRETATION:
  - If P1 Benford >> P2 and >> P3 ensemble  -> structure is geometry-coupled.
    The holographic claim survives.
  - If P1 ~ P2 ~ P3                          -> structure is in the bitstream
    regardless of projection geometry. The claim retreats to a different layer.
  - If P2 collapses but P1 is also weak      -> intended geometry doesn't
    resolve much structure either; the original Benford signal may have been
    sampling-dependent.

The script has two modes:
  --mode smoke     : runs in ~30s on CPU. Smaller M, no GPU. For sanity.
  --mode stress    : full N counts. For the real result.

HISTORICAL CONTEXT:
    Probe 2 was the first control experiment in the trajectory. The early
    Ghost Oracle code (pre-suite) reported a "Benford holographic structure"
    in the QPU manifold stream that appeared to align with the intended
    projection geometry. This probe tests whether that alignment is real.

    The original 12-tile run reported intended geometry at z = -0.36
    (Benford) and z = -0.99 (recursive) against the scrambled null —
    no separation. Running this script against the sample QPU base
    shipped in data/ will produce different z-scores (depending on which
    job dump is auto-found), but the qualitative finding is the same:
    the intended geometry is statistically indistinguishable from
    scrambled geometry on these metrics, and so the "holographic
    geometry-coupling" claim collapses.

    That finding is what motivated the rest of the probe series —
    eventually leading to Probe 4's discovery of T3 and Probe 9's
    simplification to the G_M operator that drives the suite. See
    PROCESS_RECORD.md for the full arc.

USAGE:
    python probe2_projection_scrambled.py --mode smoke
    python probe2_projection_scrambled.py --mode stress
    python probe2_projection_scrambled.py --mode smoke --base data/job_xyz.npz
==============================================================================
"""

import argparse
import math
import secrets
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

try:
    import cupy as cp
    _HAVE_CUPY = True
except Exception:
    cp = None
    _HAVE_CUPY = False


# =============================================================================
# CONFIG
# =============================================================================
# Probe 2 originally ran against a 12-tile job. The rest of the suite uses
# 16 tiles (4x4); pass --num-tiles 16 to compare against a current-generation
# QPU base. The findings reported in the docstring and PROCESS_RECORD are
# from the historical 12-tile run.
NUM_TILES        = 12
PAIRS            = [(r, c) for r in range(4) for c in range(4)][:NUM_TILES]
ANGLE_SCALE      = 1.05
ALPHA_NORM       = 0.9127
RECURSIVE_MEMORY = 0.35     # probe-local hyperparameter; not suite-wide

MATRIX_A_ORIG = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B_ORIG = np.array([1.00, 0.80, 0.40, 0.10])

# Repo-root data/ directory: this file lives at <repo>/probes/probe2_*.py.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def data_to_angles(data, scale=ANGLE_SCALE):
    """Scale a real-valued vector into rotation angles in [0, pi/2 * scale]."""
    max_val = np.max(np.abs(data))
    return (data / max_val) * (np.pi / 2) * scale


ORIG_A = data_to_angles(MATRIX_A_ORIG)
ORIG_B = data_to_angles(MATRIX_B_ORIG)


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


# =============================================================================
# BENFORD / RECURSIVE TELEMETRY
# =============================================================================
BENFORD_LOG = np.array([math.log10(1 + 1 / d) for d in range(1, 10)])


def leading_digit(x):
    s = f"{abs(x):e}"
    return int(s[0])


def benford_distribution_score(stream, depth=3):
    stream = np.asarray(stream, dtype=np.float64)
    deltas = np.diff(stream)
    digits = [leading_digit(d) for d in deltas if d != 0 and not np.isnan(d)]
    valid = [d for d in digits if 1 <= d <= 9]
    if len(valid) < 5:
        return 0.0
    obs = np.array([valid.count(d) / len(valid) for d in range(1, 10)])
    score = math.exp(-np.sum(((obs - BENFORD_LOG) ** 2) / BENFORD_LOG) * 5.0)
    if depth <= 1:
        return score
    return (score + benford_distribution_score(deltas, depth - 1)) / 2.0


def recursive_manifold_analysis(stream, depth=4):
    """Depth >=2 so STABILITY actually discriminates (see Probe 1 notes)."""
    stream = np.asarray(stream, dtype=np.float64)
    if len(stream) < 8:
        return 0.0, 0.0
    scores = []
    memory = np.zeros_like(stream)
    current = stream.copy()
    for _ in range(depth):
        current = (1.0 - RECURSIVE_MEMORY) * current + RECURSIVE_MEMORY * memory[:len(current)]
        mu, sigma = np.mean(current), np.std(current)
        if sigma > 1e-12:
            current = (current - mu) / sigma
        scores.append(benford_distribution_score(current, depth=2))
        memory = 0.7 * memory[:len(current)] + 0.3 * current
        current = np.diff(current)
        if len(current) < 8:
            break
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(1.0 - np.std(scores))


# =============================================================================
# PROJECTION LOGIC
# =============================================================================
def project_shots(ctrl_t, ghost_t, orig_a, orig_b, new_a, new_b):
    """Same importance-reweighting projection as the original Ghost Oracle code."""
    a_fire = ghost_t[:, :2].mean(axis=1)
    b_fire = ghost_t[:, 2:].mean(axis=1)

    exp_a_orig = math.sin(orig_a / 2) ** 2
    exp_b_orig = math.sin(orig_b / 2) ** 2
    exp_a_new  = math.sin(new_a / 2) ** 2
    exp_b_new  = math.sin(new_b / 2) ** 2

    eps = 0.05

    def ll(obs, p):
        p = np.clip(p, eps, 1 - eps)
        return obs * np.log(p) + (1 - obs) * np.log(1 - p)

    log_w = np.clip(
        (ll(a_fire, exp_a_new) - ll(a_fire, exp_a_orig)) +
        (ll(b_fire, exp_b_new) - ll(b_fire, exp_b_orig)),
        -3, 3,
    )
    w = np.exp(log_w)
    w /= w.sum()
    return float(np.sum(w * (ctrl_t == 0)))


def manifold_stream(ctrl_data, ghost_data, test_matrices, orig_a, orig_b):
    """Per-shot-collapsed manifold stream, same evaluation as the original code."""
    stream = []
    for a_vec, b_vec in test_matrices:
        new_a = data_to_angles(a_vec)
        new_b = data_to_angles(b_vec)
        for t_idx, (r, c) in enumerate(PAIRS):
            p0 = project_shots(
                ctrl_data[t_idx], ghost_data[t_idx],
                orig_a[r], orig_b[c],
                new_a[r], new_b[c],
            )
            if 0.45 <= p0 < 0.5:
                p0 = 0.5
            raw = math.sqrt(max(0.0, 2 * p0 - 1.0))
            norm = min(1.0, raw / ALPHA_NORM)
            stream.append(norm)
    return np.array(stream)


# =============================================================================
# GEOMETRY GENERATORS
# =============================================================================
sysrand = secrets.SystemRandom()


def gen_test_matrices_intended(n, seed_offset=0):
    """Crypto-uniform draws in the intended-geometry distribution."""
    out = []
    for _ in range(n):
        a = np.array([0.1 + 0.9 * sysrand.random() for _ in range(4)])
        b = np.array([0.1 + 0.9 * sysrand.random() for _ in range(4)])
        out.append((a, b))
    return out


def gen_test_matrices_scrambled(n, mode="permuted"):
    """Wrong-geometry generators. Same statistical family, different structure.

    "permuted": draw a, b from same range but permute angle indices so the
                (r,c) -> angle assignment is shuffled. Same marginal distribution,
                no preserved geometric coupling to the prepared state.
    "shifted":  draw a, b but apply a pi/2 phase rotation. Valid angles, wrong axis.
    """
    out = []
    for _ in range(n):
        a = np.array([0.1 + 0.9 * sysrand.random() for _ in range(4)])
        b = np.array([0.1 + 0.9 * sysrand.random() for _ in range(4)])
        if mode == "permuted":
            perm_a = list(range(4)); sysrand.shuffle(perm_a)
            perm_b = list(range(4)); sysrand.shuffle(perm_b)
            a = a[perm_a]
            b = b[perm_b]
        elif mode == "shifted":
            a = a + math.pi / 2
            b = b + math.pi / 2
        out.append((a, b))
    return out


# =============================================================================
# CONTROLLED BIT-LEVEL SHUFFLES (kept for parity with the original probe)
# =============================================================================
def crypto_permutation(arr):
    a = list(arr)
    sysrand.shuffle(a)
    return np.array(a, dtype=arr.dtype)


def crypto_bitstream(length):
    return np.array([sysrand.randint(0, 1) for _ in range(length)], dtype=np.uint8)


# =============================================================================
# DATA LOAD
# =============================================================================
def load_qpu_base(path, num_tiles):
    """Load a QPU base .npz (dump.py schema)."""
    print(f"[LOAD] {path}")
    d = np.load(path)
    ctrl  = {t: d[f"ctrl_tile{t}"]  for t in range(num_tiles)}
    ghost = {t: d[f"ghost_tile{t}"] for t in range(num_tiles)}
    return ctrl, ghost


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate(name, ctrl, ghost, test_mats, orig_a, orig_b):
    stream = manifold_stream(ctrl, ghost, test_mats, orig_a, orig_b)
    bscore = benford_distribution_score(stream)
    rscore, rstab = recursive_manifold_analysis(stream)
    return {
        "name": name, "stream_len": len(stream),
        "mean": float(np.mean(stream)), "std": float(np.std(stream)),
        "benford": bscore, "recursive": rscore, "stability": rstab,
    }


def print_row(r):
    print(f"  {r['name']:<32} | "
          f"len={r['stream_len']:>6} | "
          f"mean={r['mean']:>6.4f} | "
          f"std={r['std']:>6.4f} | "
          f"benford={r['benford']:>8.6f} | "
          f"recursive={r['recursive']:>8.6f} | "
          f"stability={r['stability']:>8.6f}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Ghost Oracle Suite — Probe 2: Projection-Scrambled Control. "
                    "Tests whether Benford structure in the QPU manifold stream "
                    "is geometry-coupled or just sitting in the bitstream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mode", choices=["smoke", "stress"], default="smoke",
                    help="smoke = small N for sanity; stress = full N for real result.")
    ap.add_argument("--base", default=None,
                    help="Path to QPU base .npz from dump.py. "
                         "Auto-finds in data/ if omitted.")
    ap.add_argument("--num-tiles", type=int, default=NUM_TILES,
                    help="Number of tiles in the base (must match the .npz).")
    ap.add_argument("--n-intended", type=int, default=None,
                    help="Override # of intended-geometry trajectories.")
    ap.add_argument("--n-scrambled-per-mode", type=int, default=None,
                    help="Override # of scrambled trajectories per scramble mode.")
    ap.add_argument("--n-null-ensemble", type=int, default=None,
                    help="Override # of independent scrambled draws for null distribution.")
    args = ap.parse_args()

    base_path = args.base or auto_find_base("qpu")
    if not base_path:
        sys.exit(f"[FATAL] no QPU base found. Pass --base or put a "
                 f"job_*.npz in {DATA_DIR}/")

    if args.mode == "smoke":
        N_INT  = args.n_intended or 100
        N_SCR  = args.n_scrambled_per_mode or 100
        N_NULL = args.n_null_ensemble or 20
    else:
        N_INT  = args.n_intended or 1000
        N_SCR  = args.n_scrambled_per_mode or 1000
        N_NULL = args.n_null_ensemble or 100

    print("\n" + "=" * 110)
    print(f"  GHOST ORACLE SUITE — PROBE 2 — PROJECTION-SCRAMBLED CONTROL  (mode={args.mode})")
    print("=" * 110)
    print(f"  Intended trajectories      : {N_INT}")
    print(f"  Scrambled trajectories     : {N_SCR}  (per scramble mode)")
    print(f"  Null ensemble draws        : {N_NULL}")
    print(f"  CuPy available             : {_HAVE_CUPY}")

    ctrl, ghost = load_qpu_base(base_path, args.num_tiles)

    # Sanity match: also run shuffled/random/zero controls so we can confirm
    # the projection-scrambled column is interpretable against known anchors.
    shuffled_ctrl = {t: crypto_permutation(ctrl[t])    for t in range(args.num_tiles)}
    random_ctrl   = {t: crypto_bitstream(len(ctrl[t])) for t in range(args.num_tiles)}

    print("\n" + "-" * 110)
    print("  PHASE 1 — BASELINE PROJECTIONS (intended geometry, varied control streams)")
    print("-" * 110)
    test_mats_intended = gen_test_matrices_intended(N_INT)
    rows = []
    rows.append(evaluate("1a. REAL  + INTENDED geom",   ctrl,          ghost, test_mats_intended, ORIG_A, ORIG_B))
    rows.append(evaluate("1b. SHUF  + INTENDED geom",   shuffled_ctrl, ghost, test_mats_intended, ORIG_A, ORIG_B))
    rows.append(evaluate("1c. RAND  + INTENDED geom",   random_ctrl,   ghost, test_mats_intended, ORIG_A, ORIG_B))
    for r in rows: print_row(r)

    print("\n" + "-" * 110)
    print("  PHASE 2 — SCRAMBLED-GEOMETRY PROJECTIONS (real control stream, wrong geometry)")
    print("-" * 110)
    test_mats_perm  = gen_test_matrices_scrambled(N_SCR, mode="permuted")
    test_mats_shift = gen_test_matrices_scrambled(N_SCR, mode="shifted")
    rows2 = []
    rows2.append(evaluate("2a. REAL  + PERMUTED geom", ctrl, ghost, test_mats_perm,  ORIG_A, ORIG_B))
    rows2.append(evaluate("2b. REAL  + SHIFTED  geom", ctrl, ghost, test_mats_shift, ORIG_A, ORIG_B))
    for r in rows2: print_row(r)

    print("\n" + "-" * 110)
    print("  PHASE 3 — NULL ENSEMBLE (independent scrambled draws -> distribution under H0)")
    print("-" * 110)
    null_b = []
    null_r = []
    null_s = []
    for i in range(N_NULL):
        test_mats_null = gen_test_matrices_scrambled(max(50, N_SCR // 4), mode="permuted")
        m = evaluate(f"   draw {i+1}", ctrl, ghost, test_mats_null, ORIG_A, ORIG_B)
        null_b.append(m["benford"])
        null_r.append(m["recursive"])
        null_s.append(m["stability"])
        if (i + 1) % max(1, N_NULL // 10) == 0:
            print(f"    null draw {i+1}/{N_NULL} -> benford={m['benford']:.4f}  "
                  f"recursive={m['recursive']:.4f}")
    null_b = np.array(null_b); null_r = np.array(null_r); null_s = np.array(null_s)

    intended_b = rows[0]["benford"]
    intended_r = rows[0]["recursive"]

    p_b = float(np.mean(null_b >= intended_b))
    p_r = float(np.mean(null_r >= intended_r))

    print(f"\n  Null distribution (Benford):    "
          f"mean={null_b.mean():.4f}  std={null_b.std():.4f}  "
          f"min={null_b.min():.4f}  max={null_b.max():.4f}")
    print(f"  Null distribution (Recursive):  "
          f"mean={null_r.mean():.4f}  std={null_r.std():.4f}  "
          f"min={null_r.min():.4f}  max={null_r.max():.4f}")
    print(f"\n  Intended-geometry Benford   = {intended_b:.6f}")
    print(f"  Intended-geometry Recursive = {intended_r:.6f}")
    print(f"\n  Empirical p-value (Benford  >= intended): {p_b:.4f}")
    print(f"  Empirical p-value (Recursive >= intended): {p_r:.4f}")

    z_b = (intended_b - null_b.mean()) / max(null_b.std(), 1e-9)
    z_r = (intended_r - null_r.mean()) / max(null_r.std(), 1e-9)
    print(f"\n  Z-score (Benford):   {z_b:+.2f}")
    print(f"  Z-score (Recursive): {z_r:+.2f}")

    print("\n" + "=" * 110)
    print("  VERDICT")
    print("=" * 110)
    if z_b > 3 and z_r > 3:
        print("  STRONG separation. Intended geometry sits >3 sigma above the null.")
        print("  Structure is COUPLED to projection geometry. Holographic frame survives.")
    elif z_b > 2 or z_r > 2:
        print("  MODERATE separation. Suggestive but not conclusive.")
        print("  Worth running stress mode with larger N_NULL before committing.")
    elif abs(z_b) < 1 and abs(z_r) < 1:
        print("  NO separation. Intended geometry indistinguishable from scrambled.")
        print("  Structure is in the bitstream regardless of projection geometry.")
        print("  Holographic frame needs to be re-stated at a different layer.")
    else:
        print("  AMBIGUOUS. Mixed signals.")
        print("  Increase N_NULL or N_INT and re-run before drawing conclusions.")
    print()


if __name__ == "__main__":
    main()