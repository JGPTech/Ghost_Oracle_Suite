#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — JOB DUMP
==============================================================================
Pulls per-tile control bitstrings and ghost bitstrings from a completed IBM
Runtime job as integer arrays, plus metadata, and saves to a single .npz in
the repo's data/ directory.

Works after the repo split:

    ghost_oracle/G_M/dump.py
    data/job_<JOB_ID>.npz

Usage:
    python ghost_oracle/G_M/dump.py <JOB_ID>
    python ghost_oracle/G_M/dump.py <JOB_ID> --num-tiles 12
==============================================================================
"""

import argparse
import re
from pathlib import Path

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent          # repo/ghost_oracle/G_M
REPO_ROOT = HERE.parent.parent                  # repo/
DATA_DIR = REPO_ROOT / "data"


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
        default=None,
        help=(
            "Number of tiles in the job. If omitted, infer from DataBin "
            "register names like matmul_tile0, ghost_tile0, ..."
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional output .npz path. Default: <repo>/data/job_<JOB_ID>.npz",
    )
    return p.parse_args()


# =============================================================================
# HELPERS
# =============================================================================

def public_attrs(obj):
    return [a for a in dir(obj) if not a.startswith("_")]


def infer_tile_indices(databin):
    """
    Infer tile indices from DataBin attributes.

    Expected register names:
        matmul_tile{t}
        ghost_tile{t}
    """
    attrs = public_attrs(databin)

    matmul = set()
    ghost = set()

    for name in attrs:
        m = re.fullmatch(r"matmul_tile(\d+)", name)
        if m:
            matmul.add(int(m.group(1)))

        g = re.fullmatch(r"ghost_tile(\d+)", name)
        if g:
            ghost.add(int(g.group(1)))

    both = sorted(matmul & ghost)
    if not both:
        raise RuntimeError(
            "No matching matmul_tile*/ghost_tile* register pairs found.\n"
            f"Available DataBin attrs: {attrs}"
        )

    missing_matmul = sorted(ghost - matmul)
    missing_ghost = sorted(matmul - ghost)

    if missing_matmul or missing_ghost:
        print("[warn] unmatched tile registers detected:")
        if missing_matmul:
            print(f"       ghost exists but matmul missing for tiles: {missing_matmul}")
        if missing_ghost:
            print(f"       matmul exists but ghost missing for tiles: {missing_ghost}")

    return both


def extract_bitstrings(register_obj):
    """
    Extract bitstrings from a Qiskit Runtime register object.

    Current code path uses get_bitstrings(), matching your working extraction.
    """
    if hasattr(register_obj, "get_bitstrings"):
        return register_obj.get_bitstrings()

    raise RuntimeError(
        f"Register object does not expose get_bitstrings(). "
        f"type={type(register_obj)} attrs={public_attrs(register_obj)}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    job_id = args.job_id

    print(f"\n{'=' * 78}\n  GHOST ORACLE SUITE — JOB DUMP\n{'=' * 78}")
    print(f"  Job ID    : {job_id}")
    print(f"  Data dir  : {DATA_DIR}")

    print("\n[LOAD] Connecting to IBM Runtime...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    job = service.job(job_id)
    result = job.result()[0]
    databin = result.data

    backend_name = job.backend().name if hasattr(job, "backend") else "unknown"
    print(f"        Backend: {backend_name}")

    if args.num_tiles is None:
        tile_indices = infer_tile_indices(databin)
        print(f"  Tiles     : auto-detected {len(tile_indices)} -> {tile_indices}")
    else:
        tile_indices = list(range(args.num_tiles))
        print(f"  Tiles     : requested {args.num_tiles} -> {tile_indices}")

    data = {
        "job_id": job_id,
        "backend": backend_name,
        "num_tiles": len(tile_indices),
        "tile_indices": np.array(tile_indices, dtype=np.int32),
    }

    print("\n[EXTRACT] Per-tile bitstrings...")
    for out_t, t in enumerate(tile_indices):
        matmul_name = f"matmul_tile{t}"
        ghost_name = f"ghost_tile{t}"

        if not hasattr(databin, matmul_name):
            raise AttributeError(
                f"DataBin has no register '{matmul_name}'. "
                f"Available tile registers: {infer_tile_indices(databin)}"
            )
        if not hasattr(databin, ghost_name):
            raise AttributeError(
                f"DataBin has no register '{ghost_name}'. "
                f"Available tile registers: {infer_tile_indices(databin)}"
            )

        c_bits = extract_bitstrings(getattr(databin, matmul_name))
        g_bits = extract_bitstrings(getattr(databin, ghost_name))

        # Control: single bit per shot.
        ctrl_arr = np.array([int(b) for b in c_bits], dtype=np.uint8)

        # Ghost: 4 bits per shot, reversed to [a1, a2, b1, b2] ordering.
        ghost_arr = np.array(
            [[int(x) for x in bs[::-1]] for bs in g_bits],
            dtype=np.uint8,
        )

        # Store contiguously as ctrl_tile0..N-1 even if original tile IDs differ.
        data[f"ctrl_tile{out_t}"] = ctrl_arr
        data[f"ghost_tile{out_t}"] = ghost_arr

        print(
            f"  tile{t} -> stored tile{out_t}: "
            f"ctrl shape {ctrl_arr.shape}, "
            f"ghost shape {ghost_arr.shape}, "
            f"p(0)={(ctrl_arr == 0).mean():.4f}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else DATA_DIR / f"job_{job_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_path, **data)

    print(f"\n{'=' * 78}")
    print("  DUMP COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Output    : {out_path}")
    print(f"  Backend   : {backend_name}")
    print(f"  Tiles     : {len(tile_indices)}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()