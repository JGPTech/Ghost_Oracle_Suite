#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — JOB DUMP
==============================================================================
Pulls per-tile control bitstrings and ghost bitstrings from a completed IBM
Runtime job as integer arrays, plus metadata, and saves to a single .npz in
the repo's data/ directory. The output file is self-contained — no IBM
Runtime needed to re-analyze.

The .npz layout is the canonical QPU base consumed by the projection script
and all downstream probes.

Usage:
    python dump.py <JOB_ID>

Output:
    <repo>/data/job_<JOB_ID>.npz
==============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

# =============================================================================
# CONFIGURATION
# =============================================================================
# Matches qpu.py's 4x4 default (4 rows * 4 cols = 16 tiles).
NUM_TILES = 16

# Repo-root data/ directory: this file lives at <repo>/ghost_oracle/dump.py.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — extract a Runtime job to .npz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("job_id", help="IBM Runtime job ID.")
    p.add_argument(
        "--num-tiles",
        type=int,
        default=NUM_TILES,
        help="Number of tiles in the job. Must match the tile count submitted by qpu.py.",
    )
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    job_id = args.job_id
    num_tiles = args.num_tiles

    print(f"\n{'=' * 78}\n  GHOST ORACLE SUITE — JOB DUMP\n{'=' * 78}")
    print(f"  Job ID    : {job_id}")
    print(f"  Tiles     : {num_tiles}")
    print(f"  Data dir  : {DATA_DIR}")

    print("\n[LOAD] Connecting to IBM Runtime...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    job = service.job(job_id)
    result = job.result()[0]

    backend_name = job.backend().name if hasattr(job, "backend") else "unknown"
    print(f"        Backend: {backend_name}")

    data = {}
    data["job_id"] = job_id
    data["num_tiles"] = num_tiles

    print("\n[EXTRACT] Per-tile bitstrings...")
    for t in range(num_tiles):
        c_bits = getattr(result.data, f"matmul_tile{t}").get_bitstrings()
        g_bits = getattr(result.data, f"ghost_tile{t}").get_bitstrings()

        # Control: single bit per shot
        ctrl_arr = np.array([int(b) for b in c_bits], dtype=np.uint8)

        # Ghost: 4 bits per shot, reversed to [a1, a2, b1, b2] ordering
        ghost_arr = np.array(
            [[int(x) for x in bs[::-1]] for bs in g_bits],
            dtype=np.uint8,
        )

        data[f"ctrl_tile{t}"] = ctrl_arr
        data[f"ghost_tile{t}"] = ghost_arr

        print(
            f"  tile{t}: ctrl shape {ctrl_arr.shape}, "
            f"ghost shape {ghost_arr.shape}, "
            f"p(0)={(ctrl_arr == 0).mean():.4f}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"job_{job_id}.npz"
    np.savez_compressed(out_path, **data)

    print(f"\n{'=' * 78}")
    print("  DUMP COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Output    : {out_path}")
    print(f"  Backend   : {backend_name}")
    print(f"  Tiles     : {num_tiles}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
