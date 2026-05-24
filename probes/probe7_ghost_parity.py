#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — PROBE 7 — GHOST ENTANGLEMENT PARITY TEST
==============================================================================
Direct physical confirmation of the GHZ correlation Probe 4 derived
analytically. Probes 1-5 inferred that the ghost CNOTs in the per-tile
circuit entangle v1 with (a1, a2) and v2 with (b1, b2); this probe
verifies it on the actual QPU hardware data by looking at the ancilla
bit parities directly.

THEORY:
    If the textbook product-state formula were correct, the ancillas a1
    and a2 (and likewise b1 and b2) would be statistically independent
    measurements of the same underlying angle. Under independence:
        P(a1 != a2) = P(a1)(1 - P(a2)) + (1 - P(a1))P(a2)
    which for marginals near 0.5 approaches 0.5.

    Under the actual circuit, after CNOT(v1 -> a1) and CNOT(v1 -> a2),
    the (v1, a1, a2) register is in a GHZ-correlated state:
        cos(a/2)|000> + sin(a/2)|111>
    so a1 == a2 deterministically. The same logic gives b1 == b2.

    Predictions:
        Noiseless GPU base (T3) -> P(a1 != a2) = 0 exactly
        Physical QPU            -> P(a1 != a2) significantly below the
                                   independence null, with the gap
                                   measuring hardware decoherence
        Pure product state      -> P(a1 != a2) ~= independence null

HISTORICAL CONTEXT:
    Probe 7 was the direct physical confirmation that closed the loop on
    the Probe 4 pivot. Probes 1-3 ruled out the textbook target by
    indirect statistical evidence; Probe 4 derived the correct GHZ-based
    target analytically; Probe 5 validated it against the noiseless GPU
    base; Probe 6 formalized the three-target framing. This probe
    points at the QPU data and asks: is the GHZ correlation physically
    there?

    The original 12-tile run answered yes. Mean P(a1 != a2) = 0.15 vs
    independence null of 0.30; mean P(b1 != b2) = 0.17 vs null of 0.36.
    Most tiles flagged strong entanglement evidence (observed
    anti-correlation below 75% of the independence null). The gap from
    the noiseless zero is hardware decoherence — the residual that
    Probes 8.0 through 8.4 went on to characterize in detail.

    Running this script against the sample bases shipped in data/ will
    produce different specific numbers (different jobs, different
    seeds), but the qualitative finding holds: physical QPU
    anti-correlation sits well below the independence null, and the
    noiseless GPU base sits at exactly zero. See PROCESS_RECORD.md for
    the full arc.

USAGE:
    python probe7_ghost_parity.py
    python probe7_ghost_parity.py --qpu data/job_xyz.npz --gpu data/ghost_oracle_gpu_xyz.npz
==============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# =============================================================================
# CONFIG
# =============================================================================
# Probe 7 originally ran against a 12-tile job. The rest of the suite uses
# 16 tiles (4x4); pass --num-tiles 16 to compare against a current-generation
# QPU base.
NUM_TILES = 12
PAIRS     = [(r, c) for r in range(4) for c in range(4)][:NUM_TILES]

# Repo-root data/ directory: this file lives at <repo>/probes/probe7_*.py.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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
# DATA LOAD
# =============================================================================
def load_ghost_data(path, num_tiles):
    """Loads the ghost ancilla bitstreams from a base .npz (dump.py schema)."""
    print(f"[LOAD] {path}")
    try:
        d = np.load(path)
        # ghost_tile{t} is shape (n_shots, 4), columns = [a1, a2, b1, b2]
        ghost = {t: d[f"ghost_tile{t}"] for t in range(num_tiles)}
        return ghost
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None


# =============================================================================
# PARITY ANALYSIS
# =============================================================================
def calculate_parity_stats(ghost_data, num_tiles):
    """Per-tile observed anti-correlation and the product-state independence null.

    If a1 and a2 were independent, P(a1 != a2) = P(a1)(1-P(a2)) + (1-P(a1))P(a2).
    The actual circuit produces GHZ-correlated (v1, a1, a2) so a1 == a2 in the
    noiseless limit. The gap (null - observed) is the physical evidence of the
    ghost-CNOT entanglement.
    """
    stats = {}
    for t in range(num_tiles):
        g = ghost_data[t]

        a1, a2 = g[:, 0], g[:, 1]
        b1, b2 = g[:, 2], g[:, 3]

        # Observed anti-correlation
        p_a_anti = float(np.mean(a1 != a2))
        p_b_anti = float(np.mean(b1 != b2))

        # Product-state (independence) null prediction
        pa1, pa2 = float(np.mean(a1)), float(np.mean(a2))
        pb1, pb2 = float(np.mean(b1)), float(np.mean(b2))

        null_prod_a = (pa1 * (1.0 - pa2)) + ((1.0 - pa1) * pa2)
        null_prod_b = (pb1 * (1.0 - pb2)) + ((1.0 - pb1) * pb2)

        stats[t] = {
            "obs_a_anti":  p_a_anti,
            "obs_b_anti":  p_b_anti,
            "null_prod_a": null_prod_a,
            "null_prod_b": null_prod_b,
        }

    return stats


# =============================================================================
# REPORTING
# =============================================================================
def print_parity_report(label, stats, num_tiles):
    """Formats and prints the parity statistics for a given dataset."""
    print(f"\n{'-' * 80}")
    print(f" PARITY REPORT: {label.upper()}")
    print(f"{'-' * 80}")
    print(f" {'Tile':>4} | {'(r,c)':>5} | {'P(a1 != a2)':>12} | {'Null (Indep)':>12} "
          f"| {'P(b1 != b2)':>12} | {'Null (Indep)':>12}")
    print("-" * 80)

    mean_a_anti = 0.0
    mean_b_anti = 0.0

    for t in range(num_tiles):
        r, c = PAIRS[t]
        s = stats[t]

        mean_a_anti += s["obs_a_anti"]
        mean_b_anti += s["obs_b_anti"]

        # Flag physical entanglement if observed anti-correlation is significantly
        # below the independence null (under 75% of null = strong evidence).
        flag_a = "*" if s["obs_a_anti"] < (s["null_prod_a"] * 0.75) else " "
        flag_b = "*" if s["obs_b_anti"] < (s["null_prod_b"] * 0.75) else " "

        print(f" {t:>4} | {r},{c}   | {s['obs_a_anti']:>10.4f} {flag_a}| "
              f"{s['null_prod_a']:>12.4f} | {s['obs_b_anti']:>10.4f} {flag_b}| "
              f"{s['null_prod_b']:>12.4f}")

    mean_a_anti /= num_tiles
    mean_b_anti /= num_tiles

    print("-" * 80)
    print(f" MEAN |       | {mean_a_anti:>10.4f}  |              "
          f"| {mean_b_anti:>10.4f}  |")
    print(f" * Indicates strong evidence of entanglement "
          f"(observed anti-correlation < 75% of product-state null).")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Ghost Oracle Suite — Probe 7: Ghost Entanglement Parity Test. "
                    "Direct physical confirmation of the GHZ correlation Probe 4 "
                    "derived analytically.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--qpu", default=None,
                    help="Path to QPU base .npz (auto-finds job_*.npz in data/ if omitted).")
    ap.add_argument("--gpu", default=None,
                    help="Path to noiseless GPU base .npz "
                         "(auto-finds ghost_oracle_gpu_*.npz or ghost_oracle_gpu_*.npz in data/ if omitted).")
    ap.add_argument("--num-tiles", type=int, default=NUM_TILES,
                    help="Number of tiles in the bases.")
    args = ap.parse_args()

    qpu_path = args.qpu or auto_find_base("qpu")
    gpu_path = args.gpu or auto_find_base("gpu")

    if not qpu_path and not gpu_path:
        sys.exit(f"[FATAL] no bases found. Pass --qpu and/or --gpu, or put "
                 f"job_*.npz / ghost_oracle_gpu_*.npz in {DATA_DIR}/")

    print("\n" + "=" * 80)
    print("  GHOST ORACLE SUITE — PROBE 7 — GHOST ENTANGLEMENT PARITY TEST")
    print("=" * 80)
    print(f"  QPU base : {qpu_path or '(none)'}")
    print(f"  GPU base : {gpu_path or '(none)'}")

    if qpu_path:
        qpu_ghost = load_ghost_data(qpu_path, args.num_tiles)
        if qpu_ghost is not None:
            qpu_stats = calculate_parity_stats(qpu_ghost, args.num_tiles)
            print_parity_report("Physical QPU base", qpu_stats, args.num_tiles)

    if gpu_path:
        gpu_ghost = load_ghost_data(gpu_path, args.num_tiles)
        if gpu_ghost is not None:
            gpu_stats = calculate_parity_stats(gpu_ghost, args.num_tiles)
            print_parity_report("Noiseless GPU base", gpu_stats, args.num_tiles)

    print()


if __name__ == "__main__":
    main()