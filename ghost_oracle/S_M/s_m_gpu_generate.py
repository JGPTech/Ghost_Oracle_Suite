#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
S_M — GPU / SYNTHETIC BASE GENERATOR
==============================================================================
Constructs a local GPU-generated S_M base with the same analysis-facing schema as
raw QPU dumps produced by sm_qpu.py dump.

This file is the S_M analogue of the G_M GPU base builder:

    G_M/gpu.py              -> noiseless / controlled projection base
    S_M/s_m_gpu_generate.py -> controlled syndrome-spacetime field base

Purpose
-------
S_M is not a scalar logical-error-rate object. It is a syndrome-spacetime field:

    final data bits     D[i]
    final edge parity   E[i] = D[i] XOR D[i+1]
    syndrome field      S[t, i]

The QPU path creates this object by submitting repeated syndrome-extraction
circuits to IBM Runtime and dumping the shot-order classical registers. This GPU
path creates a controlled local base with the same downstream `.npz` arrays:

    data_d{d}      uint8, shape (shots, d)
    synd_d{d}      uint8, shape (shots, rounds, d-1)
    flag_d{d}      optional uint8, shape (shots, rounds, n_flags)

The generated base is intended for:

    - development without waiting for QPU jobs,
    - control/fixture generation,
    - benchmark plumbing tests,
    - S_M documentation examples,
    - comparing real QPU fields against controlled synthetic fields.

It is NOT a QPU simulator claim. It is a controlled syndrome-field generator.

Model
-----
For each distance d and shot:

1. Prepare a logical product/cat record.
   - zero/one: starts mostly as all-0 or all-1.
   - plus/minus: Z-basis readout of a cat state is represented as a random
     logical branch, all-0 or all-1, before physical errors.

2. Evolve a hidden data state through repeated rounds.
   - data flips occur with probability `p_data` per qubit per round.
   - syndrome readout reports adjacent edge parity with measurement noise
     `p_syndrome`.
   - final data readout receives `p_readout` noise.

3. Optional flags are generated for f=1/f=2 layouts as a noisy local-defect
   indicator. They preserve the QPU dump shape but are diagnostic only.

The important object is the field relation between final edge parity and
syndrome spacetime, not majority-vote logical error.

Default output
--------------
If this file lives in:

    ghost_oracle/S_M/s_m_gpu_generate.py

then generated files are written under:

    ghost_oracle/S_M/data/

Example:

    ghost_oracle/S_M/data/sm_gpu_data_plus_4096shots_seed1234.npz

A metadata JSON and latest pointer are also written:

    ghost_oracle/S_M/data/sm_gpu_job_<TAG>.json
    ghost_oracle/S_M/data/latest_sm_gpu_data.json
    ghost_oracle/S_M/data/latest_sm_data.json

The final pointer is intentionally named `latest_sm_data.json` so analysis tools
can consume the GPU-generated base the same way they consume dumped QPU data.

Usage
-----
Default controlled plus-cat base:

    python ghost_oracle/S_M/s_m_gpu_generate.py

Verify full diagnostics:

    python ghost_oracle/S_M/s_m_gpu_generate.py --verify

Match a QPU run shape:

    python ghost_oracle/S_M/s_m_gpu_generate.py --shots 4096 --rounds 10 --distances 3 5 7 9

Generate flag-shaped data:

    python ghost_oracle/S_M/s_m_gpu_generate.py --flag 1

Force CPU fallback when CuPy is unavailable:

    python ghost_oracle/S_M/s_m_gpu_generate.py --allow-cpu

Notes
-----
- CuPy is used when available.
- CPU fallback is available only when explicitly allowed with --allow-cpu.
- The saved arrays are NumPy arrays regardless of generation backend.
- The schema is intentionally aligned with sm_qpu.py dump output.
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp
    _HAVE_CUPY = True
except Exception:
    cp = None
    _HAVE_CUPY = False


# =============================================================================
# DEFAULTS — match S_M QPU defaults unless intentionally overridden
# =============================================================================

DEFAULT_SHOTS = 4096
DEFAULT_ROUNDS = 10
DEFAULT_DISTANCES = [3, 5, 7, 9]
DEFAULT_FLAG_LEVEL = 0
DEFAULT_INIT_STATE = "plus"
DEFAULT_BASIS = "z"
DEFAULT_LOGICAL = 0

# Conservative field-noise defaults. These produce nontrivial syndrome fields
# without making the record completely random.
DEFAULT_P_DATA = 0.010
DEFAULT_P_SYNDROME = 0.030
DEFAULT_P_READOUT = 0.020
DEFAULT_P_FLAG = 0.005
DEFAULT_P_BURST = 0.000
DEFAULT_BURST_WIDTH = 3


# =============================================================================
# PATHS / IO
# =============================================================================

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def to_numpy(x: Any) -> np.ndarray:
    if _HAVE_CUPY and cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def gpu_name() -> str:
    if not _HAVE_CUPY:
        return "cupy-unavailable"
    try:
        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        return "unknown-cuda-gpu"


# =============================================================================
# SMALL BIT HELPERS
# =============================================================================


def bits(x: Any, xp) -> Any:
    return (x.astype(xp.uint8) & xp.uint8(1)).astype(xp.uint8)


def xor(a: Any, b: Any, xp) -> Any:
    return xp.bitwise_xor(a.astype(xp.uint8), b.astype(xp.uint8)).astype(xp.uint8)


def terminal_edge_parity(data: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(data[:, :-1], data[:, 1:]).astype(np.uint8)


def majority_vote(data: np.ndarray) -> np.ndarray:
    return (data.sum(axis=1) > (data.shape[1] / 2)).astype(np.uint8)


def last_syndrome_mismatch(data: np.ndarray, synd: np.ndarray) -> float:
    if data.shape[1] - 1 != synd.shape[2]:
        return float("nan")
    return float(np.bitwise_xor(terminal_edge_parity(data), synd[:, -1, :]).mean())


def syndrome_instability(synd: np.ndarray) -> float:
    if synd.shape[1] < 2:
        return 0.0
    return float(np.bitwise_xor(synd[:, 1:, :], synd[:, :-1, :]).mean())


# =============================================================================
# GENERATION MODEL
# =============================================================================


def flag_width(distance: int, flag_level: int) -> int:
    """Return total flag bits per round for a distance/flag-level layout."""
    if flag_level == 0:
        return 0
    if flag_level == 1:
        return 2 * (distance - 1)
    if flag_level == 2:
        return 4 * (distance - 1)
    raise ValueError(f"unsupported flag level: {flag_level}")


def init_logical_state(
    *,
    shots: int,
    distance: int,
    init_state: str,
    logical: int,
    rng: Any,
    xp: Any,
) -> Any:
    """
    Build the initial hidden data state.

    For plus/minus cat states, Z-basis readout is represented as a random logical
    branch. This is the correct analysis-facing abstraction for S_M because the
    downstream operator uses final edge parity and syndrome spacetime rather than
    majority vote against a fixed logical bit.
    """
    if init_state == "zero":
        branch = xp.zeros((shots, 1), dtype=xp.uint8)
    elif init_state == "one":
        branch = xp.ones((shots, 1), dtype=xp.uint8)
    elif init_state in ("plus", "minus"):
        branch = (rng.random((shots, 1)) < 0.5).astype(xp.uint8)
    else:
        # Keep old logical option available for compatibility.
        branch = xp.full((shots, 1), int(logical) & 1, dtype=xp.uint8)

    return xp.repeat(branch, distance, axis=1).astype(xp.uint8)


def apply_burst_errors(
    state: Any,
    *,
    p_burst: float,
    burst_width: int,
    rng: Any,
    xp: Any,
) -> Any:
    """
    Optional correlated local bursts.

    These are off by default. When enabled, they add short contiguous error
    patches so synthetic fields can test whether S_M diagnostics notice spatially
    structured disturbances rather than only independent Bernoulli noise.
    """
    if p_burst <= 0.0 or burst_width <= 0:
        return state

    shots, distance = state.shape
    if distance <= 1:
        return state

    starts = rng.integers(0, distance, size=(shots,)) if hasattr(rng, "integers") else rng.randint(0, distance, size=(shots,))
    active = (rng.random((shots,)) < float(p_burst)).astype(xp.uint8)

    mask = xp.zeros_like(state, dtype=xp.uint8)
    width = max(1, int(burst_width))
    cols = xp.arange(distance, dtype=xp.int32)[None, :]
    starts2 = starts.reshape(-1, 1).astype(xp.int32)
    # Circular patch distance along the code chain. Small and cheap.
    rel = (cols - starts2) % int(distance)
    patch = (rel < width).astype(xp.uint8)
    mask = (patch * active.reshape(-1, 1)).astype(xp.uint8)
    return xor(state, mask, xp)


def generate_distance_record(
    *,
    distance: int,
    shots: int,
    rounds: int,
    flag_level: int,
    init_state: str,
    logical: int,
    p_data: float,
    p_syndrome: float,
    p_readout: float,
    p_flag: float,
    p_burst: float,
    burst_width: int,
    rng: Any,
    xp: Any,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generate one distance block and return NumPy arrays.

    Returns
    -------
    data : uint8, shape (shots, distance)
        Final data bits.
    synd : uint8, shape (shots, rounds, distance-1)
        Syndrome spacetime field.
    flags : uint8 or None, shape (shots, rounds, n_flags)
        Optional diagnostic flag field for f=1/f=2.
    """
    if distance < 2:
        raise ValueError("distance must be >= 2")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    state = init_logical_state(
        shots=shots,
        distance=distance,
        init_state=init_state,
        logical=logical,
        rng=rng,
        xp=xp,
    )

    synd_rounds: List[Any] = []
    flag_rounds: List[Any] = []
    n_flags = flag_width(distance, flag_level)

    for _ in range(rounds):
        # Physical data flips before each syndrome measurement.
        if p_data > 0.0:
            flips = (rng.random((shots, distance)) < float(p_data)).astype(xp.uint8)
            state = xor(state, flips, xp)
        else:
            flips = xp.zeros((shots, distance), dtype=xp.uint8)

        state = apply_burst_errors(
            state,
            p_burst=p_burst,
            burst_width=burst_width,
            rng=rng,
            xp=xp,
        )

        edge = xor(state[:, :-1], state[:, 1:], xp)

        if p_syndrome > 0.0:
            meas_noise = (rng.random((shots, distance - 1)) < float(p_syndrome)).astype(xp.uint8)
            synd_r = xor(edge, meas_noise, xp)
        else:
            meas_noise = xp.zeros((shots, distance - 1), dtype=xp.uint8)
            synd_r = edge.astype(xp.uint8)

        synd_rounds.append(synd_r)

        if n_flags:
            # Local defect proxy: flags are more likely where a neighboring data
            # flip or syndrome readout disturbance happened, plus a false-positive
            # floor p_flag. This preserves shape and a plausible local relation.
            left_flip = flips[:, :-1]
            right_flip = flips[:, 1:]
            local_defect = xp.maximum(xp.maximum(left_flip, right_flip), meas_noise).astype(xp.uint8)
            repeats = 2 if flag_level == 1 else 4
            local_flag_prob = xp.repeat(local_defect, repeats, axis=1).astype(xp.float32)
            false_pos = rng.random((shots, n_flags)) < float(p_flag)
            defect_hit = rng.random((shots, n_flags)) < (0.50 * local_flag_prob)
            flag_r = xp.logical_or(false_pos, defect_hit).astype(xp.uint8)
            flag_rounds.append(flag_r)

    if p_readout > 0.0:
        readout_noise = (rng.random((shots, distance)) < float(p_readout)).astype(xp.uint8)
        data = xor(state, readout_noise, xp)
    else:
        data = state.astype(xp.uint8)

    synd = xp.stack(synd_rounds, axis=1).astype(xp.uint8)
    flags = xp.stack(flag_rounds, axis=1).astype(xp.uint8) if n_flags else None

    return to_numpy(data).astype(np.uint8), to_numpy(synd).astype(np.uint8), None if flags is None else to_numpy(flags).astype(np.uint8)


def build_sm_gpu_base(
    *,
    shots: int,
    rounds: int,
    distances: Sequence[int],
    flag_level: int,
    init_state: str,
    basis: str,
    logical: int,
    p_data: float,
    p_syndrome: float,
    p_readout: float,
    p_flag: float,
    p_burst: float,
    burst_width: int,
    seed: Optional[int],
    use_gpu: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the complete S_M GPU base and metadata."""
    if seed is None:
        seed = secrets.randbits(63)

    if use_gpu:
        xp = cp
        rng = cp.random.default_rng(seed)
        backend = "cupy_gpu"
        device = gpu_name()
    else:
        xp = np
        rng = np.random.default_rng(seed)
        backend = "numpy_cpu_fallback"
        device = "cpu"

    distances = [int(d) for d in distances]
    tag = f"sm_gpu_{init_state}_{shots}shots_seed{seed}"

    saved: Dict[str, Any] = {
        "schema": np.array("sm_data"),
        "job_id": np.array(tag),
        "backend": np.array(backend),
        "shots": np.array(int(shots), dtype=np.int64),
        "rounds": np.array(int(rounds), dtype=np.int64),
        "flag_level": np.array(int(flag_level), dtype=np.int64),
        "logical_init": np.array(int(logical), dtype=np.int64),
        "basis": np.array(str(basis)),
        "init_state": np.array(str(init_state)),
        "distances": np.asarray(distances, dtype=np.int64),
    }

    diagnostics: List[Dict[str, Any]] = []

    for d in distances:
        data, synd, flags = generate_distance_record(
            distance=d,
            shots=shots,
            rounds=rounds,
            flag_level=flag_level,
            init_state=init_state,
            logical=logical,
            p_data=p_data,
            p_syndrome=p_syndrome,
            p_readout=p_readout,
            p_flag=p_flag,
            p_burst=p_burst,
            burst_width=burst_width,
            rng=rng,
            xp=xp,
        )

        saved[f"data_d{d}"] = data
        saved[f"synd_d{d}"] = synd
        if flags is not None:
            saved[f"flag_d{d}"] = flags

        diagnostics.append({
            "distance": int(d),
            "shots": int(data.shape[0]),
            "bits": int(data.shape[1]),
            "data_mean": float(data.mean()),
            "synd_mean": float(synd.mean()),
            "flag_mean": None if flags is None else float(flags.mean()),
            "majority_ler_diagnostic": float(np.mean(majority_vote(data) != np.uint8(logical))),
            "last_syndrome_mismatch": last_syndrome_mismatch(data, synd),
            "syndrome_instability": syndrome_instability(synd),
        })

    meta = {
        "schema": "sm_gpu_synthetic_base",
        "job_id": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "s_m_gpu_generate.py",
        "generation_backend": backend,
        "device": device,
        "shots": int(shots),
        "rounds": int(rounds),
        "distances": distances,
        "flag_level": int(flag_level),
        "logical_init": int(logical),
        "basis": basis,
        "init_state": init_state,
        "state_family": "logical_cat_branch_model" if init_state in ("plus", "minus") else "logical_product_model",
        "noise_model": {
            "p_data": float(p_data),
            "p_syndrome": float(p_syndrome),
            "p_readout": float(p_readout),
            "p_flag": float(p_flag),
            "p_burst": float(p_burst),
            "burst_width": int(burst_width),
        },
        "seed": int(seed),
        "diagnostics": diagnostics,
        "notes": (
            "Controlled S_M syndrome-spacetime generator. This is not a QPU simulation claim. "
            "It writes the same analysis-facing sm_data arrays as sm_qpu.py dump."
        ),
    }

    return saved, meta


# =============================================================================
# REPORTING
# =============================================================================


def print_diagnostics(meta: Dict[str, Any]) -> None:
    print("\n" + "=" * 104)
    print("  LIGHT DIAGNOSTICS")
    print("=" * 104)
    print(
        f"  {'item':>10} | {'shots':>6} | {'bits':>5} | {'maj LER':>8} | "
        f"{'data 1s':>8} | {'synd 1s':>8} | {'flag 1s':>8} | "
        f"{'lastmis':>8} | {'instab':>8}"
    )
    print("  " + "-" * 102)
    for row in meta["diagnostics"]:
        flag_rate = "n/a" if row["flag_mean"] is None else f"{row['flag_mean']:.4f}"
        print(
            f"  {'d' + str(row['distance']):>10} | {row['shots']:>6} | {row['bits']:>5} | "
            f"{row['majority_ler_diagnostic']:>8.2%} | {row['data_mean']:>8.4f} | "
            f"{row['synd_mean']:>8.4f} | {flag_rate:>8} | "
            f"{row['last_syndrome_mismatch']:>8.4f} | {row['syndrome_instability']:>8.4f}"
        )
    if meta.get("init_state") in ("plus", "minus"):
        print("\n  [note] plus/minus cat branch model: majority LER is diagnostic only.")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S_M — GPU / synthetic syndrome-spacetime base generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help="Shots per distance block.")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="Syndrome rounds.")
    p.add_argument("--distances", type=int, nargs="+", default=DEFAULT_DISTANCES, help="Code distances to generate.")
    p.add_argument("--flag", type=int, choices=[0, 1, 2], default=DEFAULT_FLAG_LEVEL, help="Flag layout level.")
    p.add_argument("--basis", choices=["z", "x"], default=DEFAULT_BASIS, help="Recorded basis label. S_M cat runs are normally z.")
    p.add_argument("--init-state", choices=["zero", "one", "plus", "minus"], default=DEFAULT_INIT_STATE)
    p.add_argument("--logical", type=int, choices=[0, 1], default=DEFAULT_LOGICAL)

    p.add_argument("--p-data", type=float, default=DEFAULT_P_DATA, help="Per-qubit data-flip probability per round.")
    p.add_argument("--p-syndrome", type=float, default=DEFAULT_P_SYNDROME, help="Per-edge syndrome measurement noise probability.")
    p.add_argument("--p-readout", type=float, default=DEFAULT_P_READOUT, help="Final data readout noise probability.")
    p.add_argument("--p-flag", type=float, default=DEFAULT_P_FLAG, help="Flag false-positive floor for f=1/f=2.")
    p.add_argument("--p-burst", type=float, default=DEFAULT_P_BURST, help="Optional correlated local burst probability per shot/round.")
    p.add_argument("--burst-width", type=int, default=DEFAULT_BURST_WIDTH, help="Width of optional circular burst patches.")

    p.add_argument("--seed", type=int, default=None, help="RNG seed. Defaults to cryptographic random.")
    p.add_argument("--out", default=None, help="Output .npz path. Defaults to S_M/data/.")
    p.add_argument("--meta-out", default=None, help="Optional metadata JSON path. Defaults to S_M/data/.")
    p.add_argument("--allow-cpu", action="store_true", help="Allow NumPy CPU fallback if CuPy/CUDA is unavailable.")
    p.add_argument("--verify", action="store_true", help="Print light diagnostics after generation.")
    p.add_argument("--no-latest", action="store_true", help="Do not update latest_sm_gpu_data.json/latest_sm_data.json.")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()

    use_gpu = _HAVE_CUPY
    if not use_gpu and not args.allow_cpu:
        sys.exit("[FATAL] CuPy/CUDA unavailable. Install cupy-cuda12x or rerun with --allow-cpu.")

    print(f"\n{'=' * 96}")
    print("  S_M — GPU / SYNTHETIC BASE GENERATOR")
    print(f"{'=' * 96}")
    print(f"  Backend      : {'cupy_gpu' if use_gpu else 'numpy_cpu_fallback'}")
    print(f"  Device       : {gpu_name() if use_gpu else 'cpu'}")
    print(f"  Distances    : {args.distances}")
    print(f"  Flag level   : f={args.flag}")
    print(f"  Rounds       : {args.rounds}")
    print(f"  Shots        : {args.shots}")
    print(f"  Basis        : {args.basis.upper()}")
    print(f"  Init state   : {args.init_state}")
    print(f"  Data dir     : {DATA_DIR}")
    print("\n[MODEL]")
    print(f"  p_data       : {args.p_data}")
    print(f"  p_syndrome   : {args.p_syndrome}")
    print(f"  p_readout    : {args.p_readout}")
    print(f"  p_flag       : {args.p_flag}")
    print(f"  p_burst      : {args.p_burst}")

    saved, meta = build_sm_gpu_base(
        shots=int(args.shots),
        rounds=int(args.rounds),
        distances=args.distances,
        flag_level=int(args.flag),
        init_state=str(args.init_state),
        basis=str(args.basis),
        logical=int(args.logical),
        p_data=float(args.p_data),
        p_syndrome=float(args.p_syndrome),
        p_readout=float(args.p_readout),
        p_flag=float(args.p_flag),
        p_burst=float(args.p_burst),
        burst_width=int(args.burst_width),
        seed=args.seed,
        use_gpu=use_gpu,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = str(meta["job_id"])

    out_path = Path(args.out) if args.out else DATA_DIR / f"sm_gpu_data_{args.init_state}_{args.shots}shots_seed{meta['seed']}.npz"
    meta_path = Path(args.meta_out) if args.meta_out else DATA_DIR / f"sm_gpu_job_{tag}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_path, **saved)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(meta), f, indent=2)

    if not args.no_latest:
        latest_gpu = DATA_DIR / "latest_sm_gpu_data.json"
        latest_any = DATA_DIR / "latest_sm_data.json"
        latest_obj = {"job_id": tag, "npz": str(out_path), "meta": str(meta_path), "source": "gpu"}
        with open(latest_gpu, "w", encoding="utf-8") as f:
            json.dump(latest_obj, f, indent=2)
        with open(latest_any, "w", encoding="utf-8") as f:
            json.dump(latest_obj, f, indent=2)

    if args.verify:
        print_diagnostics(meta)

    print(f"\n{'=' * 96}")
    print("  S_M GPU BASE COMPLETE")
    print(f"{'=' * 96}")
    print(f"  Output    : {out_path}")
    print(f"  Metadata  : {meta_path}")
    print(f"  Latest    : {DATA_DIR / 'latest_sm_gpu_data.json'}")
    print(f"  Analysis  : {DATA_DIR / 'latest_sm_data.json'}")
    print(f"  Seed      : {meta['seed']}")
    print("\n  Next:")
    print("    python ghost_oracle/S_M/s_m_gpu_generate.py --verify")
    print("    python ghost_oracle/S_M/sm_analyze.py --npz <OUTPUT_PATH>")
    print(f"{'=' * 96}\n")


if __name__ == "__main__":
    main()
