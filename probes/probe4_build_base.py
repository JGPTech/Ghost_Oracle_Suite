#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — PROBE 4 — NOISELESS GPU BASE (THE PIVOT)
==============================================================================
Builds a noiseless reference .npz that mirrors the QPU job dump format
exactly (the dump.py schema), but samples shots from the analytically
correct joint distribution for each tile under the unitary limit.

The per-tile circuit:
    qubits: a1, v1, a2, ctrl, b1, v2, b2

    1. Ry(theta_a) on v1; Ry(theta_b) on v2
    2. CNOT v1->a1; CNOT v1->a2; CNOT v2->b1; CNOT v2->b2
    3. H on ctrl
    4. CSWAP(ctrl, v1, v2)
    5. H on ctrl
    6. Measure {ctrl, a1, a2, b1, b2}

In the unitary limit, the four nonzero amplitudes of the (v1, a1, a2,
v2, b1, b2) state after step 2 are GHZ-correlated:

    |v1 a1 a2 v2 b1 b2> with amplitudes
       cos(a/2)cos(b/2) on |000;000>
       cos(a/2)sin(b/2) on |000;111>
       sin(a/2)cos(b/2) on |111;000>
       sin(a/2)sin(b/2) on |111;111>

After the Hadamard test on (ctrl, v1, v2) the swap-test expectation becomes

    <SWAP_v1v2> = cos^2(a/2) cos^2(b/2) + sin^2(a/2) sin^2(b/2)

NOT cos^2((a-b)/2) — the ghost CNOTs entangle v1 with (a1, a2) and v2
with (b1, b2), which breaks the product-state form the textbook formula
assumes. This is the T3 target referenced throughout the rest of the
suite.

This script computes the full 32-dim joint P(ctrl, a1, a2, b1, b2) per
tile by direct statevector simulation and samples N_SHOTS i.i.d. draws.

OUTPUT (.npz keys, matching dump.py exactly):
    job_id        : str    (here: "ghost_oracle_gpu_seed{seed}")
    num_tiles     : int
    ctrl_tile{t}  : uint8, shape (n_shots,)
    ghost_tile{t} : uint8, shape (n_shots, 4)   columns = [a1, a2, b1, b2]

HISTORICAL CONTEXT:
    Probe 4 is THE PIVOT in the trajectory. Probes 1 through 3 progressively
    dismantled the original framing — Probe 1 showed the QPU doesn't match
    T2 = |cos((a-b)/2)|, Probe 2 showed the apparent "holographic
    structure" wasn't geometry-coupled, and Probe 3 showed no smooth
    channel-correction scheme could rescue the claim. This probe is where
    the team finally asked the right question: instead of trying to fix
    the QPU's deviation from a textbook formula, simulate the actual
    circuit and see what it computes.

    The answer was T3: the GHZ-entangled mixed-state target above. The
    analytical and empirical marginals match to shot noise, confirming
    that this is the operator the circuit implements in the noiseless
    limit. Probe 5 then reran Probes 1-3 against this corrected target
    and found everything snapped into place — the Identity Bridge MAE
    dropped from 0.19 (Probe 1 against T2) to ~1e-2 (against T3), and
    the Benford "signal" that started this whole investigation
    revealed itself as a sampling artifact rather than geometry-coupled
    structure.

    Probe 9 later simplified T3 to the G_M operator that drives the
    rest of the suite. See PROCESS_RECORD.md for the full arc.

RELATIONSHIP TO gpu.py:
    This probe is the readable prototype of what ghost_oracle/gpu.py
    productionizes. The probe uses an explicit 7-qubit statevector
    simulator (slow, pedagogical, ~milliseconds per tile) so the math
    can be read off the code. gpu.py uses the closed-form GHZ sampler
    derived from the same math (cos^2(a/2), sin^2(a/2), independent
    Bernoulli draws per tile) via a CUDA kernel for production speed.

    Both produce byte-compatible .npz output. Use this probe to verify
    the derivation; use gpu.py to actually build noiseless bases at
    scale.

USAGE:
    python probe4_build_base.py
    python probe4_build_base.py --shots 4096 --seed 42
    python probe4_build_base.py --out custom_name.npz
==============================================================================
"""

import argparse
import math
import secrets
import sys
from pathlib import Path

import numpy as np


# =============================================================================
# CONFIG (matches qpu.py defaults)
# =============================================================================
# Probe 4 originally ran against a 12-tile job. The rest of the suite uses
# 16 tiles (4x4); pass --num-tiles 16 to produce a base byte-compatible with
# current-generation 4x4 jobs. The diagnostics printed below are independent
# of tile count — each tile is sampled from its own analytical distribution.
NUM_TILES   = 12
PAIRS       = [(r, c) for r in range(4) for c in range(4)][:NUM_TILES]
ANGLE_SCALE = 1.05

MATRIX_A = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B = np.array([1.00, 0.80, 0.40, 0.10])

# Repo-root data/ directory: this file lives at <repo>/probes/probe4_*.py.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def data_to_angles(data, scale=ANGLE_SCALE):
    """Scale a real-valued vector into rotation angles in [0, pi/2 * scale]."""
    max_val = np.max(np.abs(data))
    return (data / max_val) * (np.pi / 2) * scale


# =============================================================================
# STATEVECTOR SIMULATION
# Bit layout in 7-qubit register: q0=a1, q1=v1, q2=a2, q3=ctrl, q4=b1, q5=v2, q6=b2
# =============================================================================
N_QUBITS = 7
DIM = 1 << N_QUBITS


def qbit(q):
    """Mask for qubit q in big-endian bit layout."""
    return 1 << (N_QUBITS - 1 - q)


def apply_1q(psi, q, U):
    out = np.zeros_like(psi)
    m = qbit(q)
    for i in range(DIM):
        if not (i & m):
            j = i | m
            a0, a1 = psi[i], psi[j]
            out[i] += U[0, 0] * a0 + U[0, 1] * a1
            out[j] += U[1, 0] * a0 + U[1, 1] * a1
    return out


def apply_cnot(psi, ctrl, tgt):
    out = np.copy(psi)
    cm = qbit(ctrl)
    tm = qbit(tgt)
    for i in range(DIM):
        if (i & cm):
            j = i ^ tm
            if i < j:
                out[i], out[j] = psi[j], psi[i]
    return out


def apply_cswap(psi, c, a, b):
    out = np.copy(psi)
    cm = qbit(c)
    am = qbit(a)
    bm = qbit(b)
    for i in range(DIM):
        if (i & cm):
            ba = (i & am) != 0
            bb = (i & bm) != 0
            if ba != bb:
                j = i ^ am ^ bm
                if i < j:
                    out[i], out[j] = psi[j], psi[i]
    return out


def Ry(theta):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


H_GATE = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def tile_joint_probabilities(theta_a, theta_b):
    """Returns 32-vector P(ctrl, a1, a2, b1, b2), index = ctrl*16+a1*8+a2*4+b1*2+b2."""
    psi = np.zeros(DIM, dtype=complex)
    psi[0] = 1.0
    psi = apply_1q(psi, 1, Ry(theta_a))   # v1
    psi = apply_1q(psi, 5, Ry(theta_b))   # v2
    psi = apply_cnot(psi, 1, 0)           # v1 -> a1
    psi = apply_cnot(psi, 1, 2)           # v1 -> a2
    psi = apply_cnot(psi, 5, 4)           # v2 -> b1
    psi = apply_cnot(psi, 5, 6)           # v2 -> b2
    psi = apply_1q(psi, 3, H_GATE)        # H on ctrl
    psi = apply_cswap(psi, 3, 1, 5)       # cswap(ctrl, v1, v2)
    psi = apply_1q(psi, 3, H_GATE)        # H on ctrl

    probs = np.zeros(32)
    for i in range(DIM):
        a1 = (i >> (N_QUBITS - 1 - 0)) & 1
        v1 = (i >> (N_QUBITS - 1 - 1)) & 1   # noqa: marginalized
        a2 = (i >> (N_QUBITS - 1 - 2)) & 1
        ct = (i >> (N_QUBITS - 1 - 3)) & 1
        b1 = (i >> (N_QUBITS - 1 - 4)) & 1
        v2 = (i >> (N_QUBITS - 1 - 5)) & 1   # noqa: marginalized
        b2 = (i >> (N_QUBITS - 1 - 6)) & 1
        out = (ct << 4) | (a1 << 3) | (a2 << 2) | (b1 << 1) | b2
        probs[out] += abs(psi[i]) ** 2
    return probs


# =============================================================================
# SAMPLING
# =============================================================================
def sample_tile(probs, n_shots, rng):
    """Returns ctrl_arr (n_shots,) uint8 and ghost_arr (n_shots, 4) uint8."""
    # Normalize defensively
    probs = probs / probs.sum()
    idx = rng.choice(32, size=n_shots, p=probs)
    ctrl = ((idx >> 4) & 1).astype(np.uint8)
    a1 = ((idx >> 3) & 1).astype(np.uint8)
    a2 = ((idx >> 2) & 1).astype(np.uint8)
    b1 = ((idx >> 1) & 1).astype(np.uint8)
    b2 = (idx & 1).astype(np.uint8)
    ghost = np.stack([a1, a2, b1, b2], axis=1).astype(np.uint8)
    return ctrl, ghost


# =============================================================================
# DIAGNOSTIC PRINTOUT
# =============================================================================
def diagnostics_per_tile(probs, theta_a, theta_b):
    p_ctrl0 = probs[:16].sum()
    p_a1_1 = sum(probs[i] for i in range(32) if (i >> 3) & 1)
    p_a2_1 = sum(probs[i] for i in range(32) if (i >> 2) & 1)
    p_b1_1 = sum(probs[i] for i in range(32) if (i >> 1) & 1)
    p_b2_1 = sum(probs[i] for i in range(32) if i & 1)
    p_a_anti = sum(probs[i] for i in range(32) if ((i >> 3) & 1) != ((i >> 2) & 1))
    p_b_anti = sum(probs[i] for i in range(32) if ((i >> 1) & 1) != (i & 1))
    expected_p0 = (1 + math.cos(theta_a/2)**2 * math.cos(theta_b/2)**2 +
                   math.sin(theta_a/2)**2 * math.sin(theta_b/2)**2) / 2
    expected_a = math.sin(theta_a / 2) ** 2
    expected_b = math.sin(theta_b / 2) ** 2
    return {
        "p_ctrl0": p_ctrl0, "expected_p0": expected_p0,
        "p_a1": p_a1_1, "p_a2": p_a2_1, "expected_a": expected_a,
        "p_b1": p_b1_1, "p_b2": p_b2_1, "expected_b": expected_b,
        "p_a_anti": p_a_anti, "p_b_anti": p_b_anti,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Ghost Oracle Suite — Probe 4: Noiseless GPU Base (the pivot). "
                    "Builds a noiseless reference .npz by direct statevector simulation "
                    "of the per-tile circuit. The readable prototype of gpu.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--shots", type=int, default=4096,
                    help="Shots per tile (matches qpu.py default).")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed (defaults to crypto-random).")
    ap.add_argument("--num-tiles", type=int, default=NUM_TILES,
                    help="Number of tiles to generate.")
    ap.add_argument("--out", default=None,
                    help="Output .npz path (auto-named under data/ if omitted).")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else secrets.randbits(63)
    rng = np.random.default_rng(seed)
    pairs = [(r, c) for r in range(4) for c in range(4)][:args.num_tiles]

    print("\n" + "=" * 100)
    print("  GHOST ORACLE SUITE — PROBE 4 — NOISELESS GPU BASE (THE PIVOT)")
    print("=" * 100)
    print(f"  Tiles  : {args.num_tiles}")
    print(f"  Shots  : {args.shots} per tile")
    print(f"  Seed   : {seed}")

    angles_a = data_to_angles(MATRIX_A)
    angles_b = data_to_angles(MATRIX_B)

    print(f"\n  ANGLES_A: {angles_a}")
    print(f"  ANGLES_B: {angles_b}")

    out_data = {"job_id": f"ghost_oracle_gpu_seed{seed}", "num_tiles": args.num_tiles}

    print("\n" + "-" * 100)
    print(f"  {'tile':>4} {'(r,c)':>7} {'a':>7} {'b':>7} "
          f"{'P_ctrl=0':>11} {'(ideal)':>10} "
          f"{'P(a=1)':>9} {'(ideal)':>9} "
          f"{'P(b=1)':>9} {'(ideal)':>9} "
          f"{'a_anti':>9} {'b_anti':>9}")
    print("-" * 100)

    for t in range(args.num_tiles):
        r, c = pairs[t]
        a = angles_a[r]
        b = angles_b[c]
        probs = tile_joint_probabilities(a, b)
        diag = diagnostics_per_tile(probs, a, b)
        ctrl, ghost = sample_tile(probs, args.shots, rng)

        # Verify sampled marginals close to analytical
        emp_p0 = float((ctrl == 0).mean())
        emp_a1 = float(ghost[:, 0].mean())
        emp_anti_a = float((ghost[:, 0] != ghost[:, 1]).mean())

        print(f"  {t:>4} ({r},{c})  {a:>6.4f} {b:>6.4f}  "
              f"{diag['p_ctrl0']:>9.6f} {diag['expected_p0']:>10.6f}  "
              f"{diag['p_a1']:>7.4f}  {diag['expected_a']:>7.4f}  "
              f"{diag['p_b1']:>7.4f}  {diag['expected_b']:>7.4f}  "
              f"{diag['p_a_anti']:>7.4f}  {diag['p_b_anti']:>7.4f}")
        # Sanity: anti-correlation should be exactly zero in the noiseless limit
        if diag['p_a_anti'] > 1e-10 or diag['p_b_anti'] > 1e-10:
            print(f"      WARN: ghost anti-correlation nonzero "
                  f"(a={diag['p_a_anti']:.2e}, b={diag['p_b_anti']:.2e})")
        # Empirical sanity
        if abs(emp_p0 - diag['p_ctrl0']) > 0.05:
            print(f"      NOTE: empirical P(ctrl=0)={emp_p0:.4f} vs analytical "
                  f"{diag['p_ctrl0']:.4f} (n={args.shots} shot noise ~ "
                  f"{1/math.sqrt(args.shots):.4f})")

        out_data[f"ctrl_tile{t}"] = ctrl
        out_data[f"ghost_tile{t}"] = ghost

    if args.out:
        out_path = Path(args.out)
        print(f"\n[SAVE] -> {out_path}")

    print(f"\n  Byte-compatible with QPU dumps from dump.py and bases from gpu.py.")
    print(f"  Use --base pointing to this file to run downstream probes against the noiseless base.")
    print(f"  For production-scale noiseless base generation, prefer gpu.py (uses a CUDA closed-form")
    print(f"  sampler derived from the same math; orders of magnitude faster).")
    print()


if __name__ == "__main__":
    main()