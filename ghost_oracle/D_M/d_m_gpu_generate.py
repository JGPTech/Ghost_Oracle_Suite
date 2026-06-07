#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
D_M — GPU / SYNTHETIC DIMENSIONAL ENTANGLEMENT BASE GENERATOR
==============================================================================

Constructs a local GPU-generated D_M base with the same analysis-facing schema
as D_M QPU dumps produced by:

    ghost_oracle/D_M/d_m_qpu_generate.py dump <JOB_ID>

This follows the Ghost Oracle Suite generator convention used by the other
operators:

    qproj:
        real QPU listener records from IBM Runtime

    gproj:
        GPU-generated controlled listener records with the same .npz schema

    geo:
        later analytic metadata-to-manifold path, not implemented here

Purpose
-------
D_M is the Dimensional Entanglement Projection operator.

It does not reconstruct a density matrix and does not certify device-independent
Bell nonlocality. It projects a Bell-witness manifold from listener records:

    pair[tile, shot, 2]

where each tile measures one witness basis:

    XY, YZ, ZY, YX

Current discovered D_M orientation:

    YZ = primary witness dimension
    ZY = reciprocal / inverted witness dimension
    XY / YX = comparison dimensions

The generated base is intended for:

    - development without waiting for QPU jobs,
    - controlled gproj fixture generation,
    - benchmark plumbing tests,
    - CUDA kernel validation at larger scale,
    - qproj/gproj/geo final benchmark development.

It is NOT a QPU simulator claim. It is a controlled D_M witness-manifold
generator.

Model
-----
For each tile, the generator selects a target connected spin correlator:

    C = <P0 P1> - <P0><P1>

Then samples two-bit shot records with that target correlation and optional
single-qubit marginal bias.

Conditions
----------
The generator supports the three D_M benchmark conditions discovered from QPROJ:

1. null

    base_delays_dt = [0, 0, 0, 0, 0]
    offset_dt      = 0

    Weak residual witness only.

2. base_delay

    base_delays_dt = [0, 256, 1024, 4096, 16384]
    offset_dt      = 0

    Clean YZ-primary / ZY-reciprocal π-phase witness manifold.

3. offset_deformed

    base_delays_dt = [0, 256, 1024, 4096, 16384]
    offset_dt      = 128 by default

    Active YZ/ZY witness manifold with offset deformation / phase jitter.

Output schema
-------------
Saved .npz fields include:

    pair                      uint8, shape (tiles, shots, 2)
    basis                     int8,  shape (tiles, 2)

    tile_indices              int32
    tile_rung_index           int32
    tile_witness_index        int32
    tile_base_delay_dt        int32
    tile_offset_dt            int32
    tile_total_delay_dt       int32
    tile_basis_q0             int8
    tile_basis_q1             int8
    tile_witness_label        str

    base_delays_dt            int32
    offset_dt                 int32
    witness_pairs             int8, shape (4, 2)
    witness_labels            str

    tile_target_connected     float32
    tile_target_corr          float32
    tile_target_mean_q0       float32
    tile_target_mean_q1       float32
    tile_meta_json            str
    generator_meta_json       str

Default output
--------------
If this file lives in:

    ghost_oracle/D_M/d_m_gpu_generate.py

then generated files are written under:

    ghost_oracle/D_M/data/

Example:

    ghost_oracle/D_M/data/dm_gpu_data_base_delay_4096shots_seed1234.npz

A metadata JSON and latest pointer are also written:

    ghost_oracle/D_M/data/dm_gpu_job_<TAG>.json
    ghost_oracle/D_M/data/latest_dm_gpu_data.json
    ghost_oracle/D_M/data/latest_dm_data.json

Usage
-----
Generate a default offset-deformed base:

    python ghost_oracle/D_M/d_m_gpu_generate.py --verify

Generate all three benchmark conditions:

    python ghost_oracle/D_M/d_m_gpu_generate.py --condition null --verify
    python ghost_oracle/D_M/d_m_gpu_generate.py --condition base_delay --verify
    python ghost_oracle/D_M/d_m_gpu_generate.py --condition offset_deformed --verify

Match larger shot count for kernel benchmarking:

    python ghost_oracle/D_M/d_m_gpu_generate.py --shots 1000000 --condition offset_deformed

Force CPU fallback when CuPy is unavailable:

    python ghost_oracle/D_M/d_m_gpu_generate.py --allow-cpu

Notes
-----
- CuPy is used when available.
- CPU fallback is available only when explicitly allowed with --allow-cpu.
- Saved arrays are NumPy arrays regardless of generation backend.
- The schema is intentionally aligned with D_M qproj dump output and the
  D_M CUDA qproj harness.

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
# DEFAULTS
# =============================================================================

DEFAULT_SHOTS = 4096
DEFAULT_CONDITION = "offset_deformed"
DEFAULT_BASE_DELAYS_DT = [0, 256, 1024, 4096, 16384]
DEFAULT_NULL_BASE_DELAYS_DT = [0, 0, 0, 0, 0]
DEFAULT_OFFSET_DT = 128

QUBITS_PER_TILE = 2
WITNESS_PAIRS = [(0, 1), (1, 2), (2, 1), (1, 0)]  # XY, YZ, ZY, YX
WITNESS_LABELS = ["XY", "YZ", "ZY", "YX"]
BASIS_LABELS = ["X", "Y", "Z"]

# Generator defaults chosen to resemble the discovered QPROJ scale without
# trying to exactly reproduce IBM hardware.
DEFAULT_NULL_SCALE = 0.012
DEFAULT_BASE_ENERGY = 0.030
DEFAULT_ENERGY_GAIN = 0.285
DEFAULT_COMPARISON_SCALE = 0.010
DEFAULT_MARGIN_SCALE = 0.030
DEFAULT_NOISE_SCALE = 0.010
DEFAULT_PHASE_JITTER = 0.13
DEFAULT_OFFSET_DEFORM = 0.18


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
        return {str(k): json_safe(v) for k, v in x.items()}
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
# METADATA PLAN
# =============================================================================

def condition_defaults(condition: str) -> Tuple[List[int], int]:
    if condition == "null":
        return list(DEFAULT_NULL_BASE_DELAYS_DT), 0
    if condition == "base_delay":
        return list(DEFAULT_BASE_DELAYS_DT), 0
    if condition == "offset_deformed":
        return list(DEFAULT_BASE_DELAYS_DT), DEFAULT_OFFSET_DT
    raise ValueError(f"unknown condition: {condition}")


def normalize_delay_values(values: Sequence[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    mx = float(np.max(arr))
    mn = float(np.min(arr))
    if abs(mx - mn) <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mn) / (mx - mn)


def normalize_log_delay_values(values: Sequence[int]) -> np.ndarray:
    arr = np.log1p(np.maximum(0.0, np.asarray(values, dtype=np.float64)))
    if arr.size == 0:
        return arr
    mx = float(np.max(arr))
    mn = float(np.min(arr))
    if abs(mx - mn) <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mn) / (mx - mn)


def build_tile_plan(
    *,
    base_delays_dt: Sequence[int],
    offset_dt: int,
    max_tiles: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build deterministic D_M tile metadata.

    Each rung contains the full witness quartet:
        XY, YZ, ZY, YX

    Absolute tile delay:
        total_delay_dt = base_delay_dt + tile * offset_dt
    """
    plan: List[Dict[str, Any]] = []
    t = 0
    for rung, base in enumerate(base_delays_dt):
        for witness_index, (b0, b1) in enumerate(WITNESS_PAIRS):
            if max_tiles is not None and t >= int(max_tiles):
                return plan
            off = int(t * int(offset_dt))
            total = int(base) + off
            label = WITNESS_LABELS[witness_index]
            plan.append({
                "tile": int(t),
                "rung_index": int(rung),
                "witness_index": int(witness_index),
                "base_delay_dt": int(base),
                "offset_dt": int(off),
                "total_delay_dt": int(total),
                "basis_q0": int(b0),
                "basis_q1": int(b1),
                "witness_label": label,
                "physical_q0": -1,
                "physical_q1": -1,
                "role": f"rung{rung}_base{int(base)}_off{off}_{label}",
            })
            t += 1
    return plan


# =============================================================================
# TARGET MANIFOLD MODEL
# =============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def target_connected_for_tile(
    *,
    condition: str,
    witness_index: int,
    rung_index: int,
    n_rungs: int,
    base_delay_dt: int,
    total_delay_dt: int,
    total_delay_norm: float,
    log_delay_norm: float,
    rng_np: np.random.Generator,
    null_scale: float,
    base_energy: float,
    energy_gain: float,
    comparison_scale: float,
    noise_scale: float,
    phase_jitter: float,
    offset_deform: float,
) -> float:
    """
    Return target connected spin correlator for one D_M witness tile.

    The model is intentionally simple and operator-facing:

        null:
            weak residual correlations only.

        base_delay:
            clean YZ-primary / ZY-reciprocal π-phase manifold.

        offset_deformed:
            similar active YZ/ZY manifold, but phase/energy are deformed by
            offset/total-delay structure.

    This is not a QPU simulator.
    """
    # Small per-tile stochastic residue.
    eps = float(rng_np.normal(0.0, noise_scale))

    if condition == "null":
        return clamp(float(rng_np.normal(0.0, null_scale)), -0.08, 0.08)

    x = float(log_delay_norm)
    xl = float(total_delay_norm)

    # Energy envelope: grows with delay and saturates.
    envelope = base_energy + energy_gain * (0.20 + 0.80 * (x ** 1.10))
    envelope = clamp(envelope, 0.0, 0.55)

    # Keep YZ primary mostly positive by restricting phase range to a positive
    # cos branch. The reciprocal return is R=-ZY.
    base_phase = math.pi * (0.05 + 0.37 * x)

    if condition == "offset_deformed":
        # Offset deformation: bends phase and amplitude without changing the
        # core YZ-primary / ZY-return family.
        deform = offset_deform * math.sin(2.0 * math.pi * xl + 0.35)
        jitter = float(rng_np.normal(0.0, phase_jitter * 0.25))
        phase = base_phase + deform + jitter
        envelope *= 0.90 + 0.18 * math.cos(2.0 * math.pi * xl + 0.17)
    else:
        phase = base_phase + float(rng_np.normal(0.0, phase_jitter * 0.10))

    # Wrap phase into [0, π), then gently avoid an exactly zero YZ branch.
    phase = phase % math.pi

    yz = envelope * math.cos(phase)
    ret = envelope * math.sin(phase)  # R = -ZY
    zy = -ret

    # Current discovered QPROJ has YZ as primary positive dimension. If the
    # analytic toy crosses negative, fold it back into the primary orientation.
    if yz < 0:
        yz = -0.35 * yz

    xy = float(rng_np.normal(0.0, comparison_scale))
    yx = float(rng_np.normal(0.0, comparison_scale))

    if witness_index == 0:
        return clamp(xy + eps, -0.20, 0.20)
    if witness_index == 1:
        return clamp(yz + eps, -0.65, 0.65)
    if witness_index == 2:
        return clamp(zy + eps, -0.65, 0.65)
    if witness_index == 3:
        return clamp(yx + eps, -0.20, 0.20)

    return 0.0


def build_targets(
    *,
    condition: str,
    tile_plan: Sequence[Dict[str, Any]],
    seed: int,
    null_scale: float,
    base_energy: float,
    energy_gain: float,
    comparison_scale: float,
    margin_scale: float,
    noise_scale: float,
    phase_jitter: float,
    offset_deform: float,
) -> Dict[str, np.ndarray]:
    rng_np = np.random.default_rng(seed ^ 0xD00D_D00D)

    total_values = [int(m["total_delay_dt"]) for m in tile_plan]
    log_norm_all = normalize_log_delay_values(total_values)
    lin_norm_all = normalize_delay_values(total_values)
    n_rungs = max(int(m["rung_index"]) for m in tile_plan) + 1 if tile_plan else 0

    target_conn = np.zeros(len(tile_plan), dtype=np.float32)
    target_m0 = np.zeros(len(tile_plan), dtype=np.float32)
    target_m1 = np.zeros(len(tile_plan), dtype=np.float32)
    target_corr = np.zeros(len(tile_plan), dtype=np.float32)

    for i, m in enumerate(tile_plan):
        conn = target_connected_for_tile(
            condition=condition,
            witness_index=int(m["witness_index"]),
            rung_index=int(m["rung_index"]),
            n_rungs=n_rungs,
            base_delay_dt=int(m["base_delay_dt"]),
            total_delay_dt=int(m["total_delay_dt"]),
            total_delay_norm=float(lin_norm_all[i]),
            log_delay_norm=float(log_norm_all[i]),
            rng_np=rng_np,
            null_scale=null_scale,
            base_energy=base_energy,
            energy_gain=energy_gain,
            comparison_scale=comparison_scale,
            noise_scale=noise_scale,
            phase_jitter=phase_jitter,
            offset_deform=offset_deform,
        )

        # Mild marginal bias. The connected correlator is what D_M uses, so
        # marginals create realistic-looking raw readout bias without being the
        # operator target.
        m0 = float(rng_np.normal(0.0, margin_scale))
        m1 = float(rng_np.normal(0.0, margin_scale))

        # Desired <P0 P1> = connected + <P0><P1>.
        corr = conn + m0 * m1

        # Keep joint distribution valid by backing off if necessary.
        # For spin ±1 variables, probabilities:
        # p++ = (1+m0+m1+corr)/4, etc. Must be >= 0.
        for _ in range(20):
            probs = np.asarray([
                (1.0 + m0 + m1 + corr) / 4.0,
                (1.0 + m0 - m1 - corr) / 4.0,
                (1.0 - m0 + m1 - corr) / 4.0,
                (1.0 - m0 - m1 + corr) / 4.0,
            ], dtype=np.float64)
            if np.all(probs >= 1e-6):
                break
            corr *= 0.95
            conn = corr - m0 * m1

        target_conn[i] = np.float32(conn)
        target_m0[i] = np.float32(m0)
        target_m1[i] = np.float32(m1)
        target_corr[i] = np.float32(corr)

    return {
        "tile_target_connected": target_conn,
        "tile_target_mean_q0": target_m0,
        "tile_target_mean_q1": target_m1,
        "tile_target_corr": target_corr,
    }


# =============================================================================
# PAIR SAMPLING
# =============================================================================

def joint_probs_from_spin_moments(m0: float, m1: float, corr: float) -> np.ndarray:
    probs = np.asarray([
        (1.0 + m0 + m1 + corr) / 4.0,  # ++ -> bits 00
        (1.0 + m0 - m1 - corr) / 4.0,  # +- -> bits 01
        (1.0 - m0 + m1 - corr) / 4.0,  # -+ -> bits 10
        (1.0 - m0 - m1 + corr) / 4.0,  # -- -> bits 11
    ], dtype=np.float64)

    probs = np.maximum(probs, 1e-9)
    probs = probs / probs.sum()
    return probs


def sample_pair_tile(
    *,
    shots: int,
    m0: float,
    m1: float,
    corr: float,
    rng: Any,
    xp: Any,
) -> Any:
    """
    Sample pair[shot, 2] for one tile.

    State mapping:
        0: spin ++ -> bits 00
        1: spin +- -> bits 01
        2: spin -+ -> bits 10
        3: spin -- -> bits 11
    """
    probs_np = joint_probs_from_spin_moments(m0, m1, corr)
    cdf_np = np.cumsum(probs_np)

    r = rng.random((shots,))

    if xp is np:
        cdf = cdf_np
        states = np.searchsorted(cdf, r, side="right").astype(np.uint8)
        out = np.zeros((shots, 2), dtype=np.uint8)
        out[:, 0] = ((states == 2) | (states == 3)).astype(np.uint8)
        out[:, 1] = ((states == 1) | (states == 3)).astype(np.uint8)
        return out

    cdf = xp.asarray(cdf_np, dtype=xp.float32)
    states = xp.searchsorted(cdf, r, side="right").astype(xp.uint8)
    out = xp.zeros((shots, 2), dtype=xp.uint8)
    out[:, 0] = xp.logical_or(states == 2, states == 3).astype(xp.uint8)
    out[:, 1] = xp.logical_or(states == 1, states == 3).astype(xp.uint8)
    return out


def generate_pair_records(
    *,
    shots: int,
    targets: Dict[str, np.ndarray],
    seed: int,
    use_gpu: bool,
) -> np.ndarray:
    if use_gpu:
        xp = cp
        rng = cp.random.default_rng(seed)
    else:
        xp = np
        rng = np.random.default_rng(seed)

    n_tiles = int(targets["tile_target_connected"].shape[0])
    pieces = []

    for t in range(n_tiles):
        tile = sample_pair_tile(
            shots=shots,
            m0=float(targets["tile_target_mean_q0"][t]),
            m1=float(targets["tile_target_mean_q1"][t]),
            corr=float(targets["tile_target_corr"][t]),
            rng=rng,
            xp=xp,
        )
        pieces.append(tile)

    if use_gpu:
        pair = xp.stack(pieces, axis=0).astype(xp.uint8)
    else:
        pair = np.stack(pieces, axis=0).astype(np.uint8)

    return to_numpy(pair).astype(np.uint8)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def spin_stats(pair_tile: np.ndarray) -> Tuple[float, float, float, float]:
    b0 = pair_tile[:, 0].astype(np.float64)
    b1 = pair_tile[:, 1].astype(np.float64)
    s0 = 1.0 - 2.0 * b0
    s1 = 1.0 - 2.0 * b1
    m0 = float(s0.mean())
    m1 = float(s1.mean())
    corr = float((s0 * s1).mean())
    conn = corr - m0 * m1
    return m0, m1, corr, conn


def project_diagnostics(pair: np.ndarray, tile_plan: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    n_rungs = max(int(m["rung_index"]) for m in tile_plan) + 1 if tile_plan else 0

    for t, m in enumerate(tile_plan):
        m0, m1, corr, conn = spin_stats(pair[t])
        rows.append({
            "tile": t,
            "rung": int(m["rung_index"]),
            "witness": str(m["witness_label"]),
            "mean_q0": m0,
            "mean_q1": m1,
            "corr": corr,
            "connected": conn,
            "base_delay": int(m["base_delay_dt"]),
            "offset": int(m["offset_dt"]),
            "total_delay": int(m["total_delay_dt"]),
        })

    rung_rows = []
    for r in range(n_rungs):
        by_w = {w: 0.0 for w in WITNESS_LABELS}
        for row in rows:
            if int(row["rung"]) == r:
                by_w[str(row["witness"])] = float(row["connected"])

        yz = by_w["YZ"]
        zy = by_w["ZY"]
        xy = by_w["XY"]
        yx = by_w["YX"]
        ret = -zy
        energy = math.sqrt(yz * yz + ret * ret)
        comp = math.sqrt(xy * xy + yx * yx)
        spec = energy - comp
        phase = math.atan2(ret, yz) % math.pi

        rung_rows.append({
            "rung": r,
            "YZ": yz,
            "ZY": zy,
            "XY": xy,
            "YX": yx,
            "energy": energy,
            "specificity": spec,
            "phase_deg": phase * 180.0 / math.pi,
        })

    yz_vals = np.asarray([x["YZ"] for x in rung_rows], dtype=np.float64)
    energy_vals = np.asarray([x["energy"] for x in rung_rows], dtype=np.float64)
    spec_vals = np.asarray([x["specificity"] for x in rung_rows], dtype=np.float64)

    return {
        "tile_rows": rows,
        "rung_rows": rung_rows,
        "summary": {
            "yz_mean": float(yz_vals.mean()) if yz_vals.size else 0.0,
            "yz_positive_fraction": float(np.mean(yz_vals > 0.0)) if yz_vals.size else 0.0,
            "energy_mean": float(energy_vals.mean()) if energy_vals.size else 0.0,
            "energy_max": float(energy_vals.max()) if energy_vals.size else 0.0,
            "specificity_mean": float(spec_vals.mean()) if spec_vals.size else 0.0,
            "specificity_max": float(spec_vals.max()) if spec_vals.size else 0.0,
        },
    }


def print_diagnostics(diag: Dict[str, Any]) -> None:
    print("\n" + "=" * 104)
    print("  D_M GPROJ LIGHT DIAGNOSTICS")
    print("=" * 104)
    print("  PER-RUNG MANIFOLD")
    print("  " + "-" * 102)
    for r in diag["rung_rows"]:
        print(
            f"  rung {r['rung']:02d} "
            f"YZ={r['YZ']:+.5f} ZY={r['ZY']:+.5f} "
            f"E={r['energy']:.5f} spec={r['specificity']:+.5f} "
            f"phase={r['phase_deg']:7.2f}°"
        )

    s = diag["summary"]
    print("  " + "-" * 102)
    print(f"  YZ mean / positive fraction : {s['yz_mean']:+.6f} / {s['yz_positive_fraction']:.3f}")
    print(f"  energy mean / max           : {s['energy_mean']:.6f} / {s['energy_max']:.6f}")
    print(f"  specificity mean / max      : {s['specificity_mean']:+.6f} / {s['specificity_max']:+.6f}")


# =============================================================================
# BUILD BASE
# =============================================================================

def build_dm_gpu_base(
    *,
    condition: str,
    shots: int,
    base_delays_dt: Sequence[int],
    offset_dt: int,
    seed: Optional[int],
    use_gpu: bool,
    null_scale: float,
    base_energy: float,
    energy_gain: float,
    comparison_scale: float,
    margin_scale: float,
    noise_scale: float,
    phase_jitter: float,
    offset_deform: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if seed is None:
        seed = secrets.randbits(63)

    tile_plan = build_tile_plan(
        base_delays_dt=base_delays_dt,
        offset_dt=offset_dt,
        max_tiles=None,
    )

    targets = build_targets(
        condition=condition,
        tile_plan=tile_plan,
        seed=int(seed),
        null_scale=float(null_scale),
        base_energy=float(base_energy),
        energy_gain=float(energy_gain),
        comparison_scale=float(comparison_scale),
        margin_scale=float(margin_scale),
        noise_scale=float(noise_scale),
        phase_jitter=float(phase_jitter),
        offset_deform=float(offset_deform),
    )

    pair = generate_pair_records(
        shots=int(shots),
        targets=targets,
        seed=int(seed),
        use_gpu=use_gpu,
    )

    n_tiles = int(pair.shape[0])
    basis = np.asarray([[m["basis_q0"], m["basis_q1"]] for m in tile_plan], dtype=np.int8)

    tag = f"dm_gpu_{condition}_{shots}shots_seed{seed}"

    backend = "cupy_gpu" if use_gpu else "numpy_cpu_fallback"

    saved: Dict[str, Any] = {
        "schema": np.array("ghost_oracle.dm.gproj.v1"),
        "suite": np.array("Ghost Oracle Suite"),
        "operator": np.array("D_M"),
        "substrate": np.array("gproj"),
        "circuit_family": np.array("bell_listener_dimensional_entanglement_gproj_v1"),
        "job_id": np.array(tag),
        "backend": np.array(backend),
        "generator": np.array("d_m_gpu_generate.py"),
        "condition": np.array(str(condition)),
        "shots": np.array(int(shots), dtype=np.int64),
        "num_tiles": np.array(n_tiles, dtype=np.int32),
        "tile_indices": np.arange(n_tiles, dtype=np.int32),
        "qubits_per_tile": np.array(QUBITS_PER_TILE, dtype=np.int32),
        "pair": pair.astype(np.uint8),
        "basis": basis,
        "base_delays_dt": np.asarray(base_delays_dt, dtype=np.int32),
        "offset_dt": np.array(int(offset_dt), dtype=np.int32),
        "witness_pairs": np.asarray(WITNESS_PAIRS, dtype=np.int8),
        "witness_labels": np.asarray(WITNESS_LABELS),
        "basis_labels": np.asarray(BASIS_LABELS),
        "tile_rung_index": np.asarray([m["rung_index"] for m in tile_plan], dtype=np.int32),
        "tile_witness_index": np.asarray([m["witness_index"] for m in tile_plan], dtype=np.int32),
        "tile_base_delay_dt": np.asarray([m["base_delay_dt"] for m in tile_plan], dtype=np.int32),
        "tile_offset_dt": np.asarray([m["offset_dt"] for m in tile_plan], dtype=np.int32),
        "tile_total_delay_dt": np.asarray([m["total_delay_dt"] for m in tile_plan], dtype=np.int32),
        "tile_basis_q0": np.asarray([m["basis_q0"] for m in tile_plan], dtype=np.int8),
        "tile_basis_q1": np.asarray([m["basis_q1"] for m in tile_plan], dtype=np.int8),
        "tile_witness_label": np.asarray([m["witness_label"] for m in tile_plan]),
        "tile_role": np.asarray([m["role"] for m in tile_plan]),
        "tile_physical_q0": np.asarray([m["physical_q0"] for m in tile_plan], dtype=np.int32),
        "tile_physical_q1": np.asarray([m["physical_q1"] for m in tile_plan], dtype=np.int32),
        "tile_cluster": np.asarray([[m["physical_q0"], m["physical_q1"]] for m in tile_plan], dtype=np.int32),
        "tile_target_connected": targets["tile_target_connected"],
        "tile_target_corr": targets["tile_target_corr"],
        "tile_target_mean_q0": targets["tile_target_mean_q0"],
        "tile_target_mean_q1": targets["tile_target_mean_q1"],
        "tile_meta_json": np.array(json.dumps(json_safe(tile_plan))),
    }

    diag = project_diagnostics(pair, tile_plan)

    meta = {
        "schema": "ghost_oracle.dm.gpu_meta.v1",
        "job_id": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operator": "D_M",
        "substrate": "gproj",
        "condition": condition,
        "generator": "d_m_gpu_generate.py",
        "generation_backend": backend,
        "device": gpu_name() if use_gpu else "cpu",
        "shots": int(shots),
        "num_tiles": n_tiles,
        "rungs": len(base_delays_dt),
        "qubits_per_tile": QUBITS_PER_TILE,
        "base_delays_dt": [int(x) for x in base_delays_dt],
        "offset_dt": int(offset_dt),
        "witness_pairs": [list(x) for x in WITNESS_PAIRS],
        "witness_labels": list(WITNESS_LABELS),
        "basis_labels": list(BASIS_LABELS),
        "seed": int(seed),
        "model": {
            "null_scale": float(null_scale),
            "base_energy": float(base_energy),
            "energy_gain": float(energy_gain),
            "comparison_scale": float(comparison_scale),
            "margin_scale": float(margin_scale),
            "noise_scale": float(noise_scale),
            "phase_jitter": float(phase_jitter),
            "offset_deform": float(offset_deform),
        },
        "diagnostics": diag["summary"],
        "notes": (
            "Controlled D_M dimensional entanglement base generator. "
            "This is not a QPU simulator claim. It writes the same analysis-facing "
            "D_M base arrays as d_m_qpu_generate.py dump."
        ),
    }

    saved["generator_meta_json"] = np.array(json.dumps(json_safe(meta)))

    return saved, meta, diag


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D_M — GPU / synthetic dimensional entanglement base generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--condition", choices=["null", "base_delay", "offset_deformed"], default=DEFAULT_CONDITION)
    p.add_argument("--shots", type=int, default=DEFAULT_SHOTS)

    p.add_argument("--base-delays-dt", type=int, nargs="+", default=None,
                   help="Override base-delay ladder. Default depends on condition.")
    p.add_argument("--offset-dt", type=int, default=None,
                   help="Override offset step dt. Default depends on condition.")
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--null-scale", type=float, default=DEFAULT_NULL_SCALE)
    p.add_argument("--base-energy", type=float, default=DEFAULT_BASE_ENERGY)
    p.add_argument("--energy-gain", type=float, default=DEFAULT_ENERGY_GAIN)
    p.add_argument("--comparison-scale", type=float, default=DEFAULT_COMPARISON_SCALE)
    p.add_argument("--margin-scale", type=float, default=DEFAULT_MARGIN_SCALE)
    p.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE_SCALE)
    p.add_argument("--phase-jitter", type=float, default=DEFAULT_PHASE_JITTER)
    p.add_argument("--offset-deform", type=float, default=DEFAULT_OFFSET_DEFORM)

    p.add_argument("--out", default=None, help="Output .npz path. Defaults to D_M/data/.")
    p.add_argument("--meta-out", default=None, help="Optional metadata JSON path. Defaults to D_M/data/.")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--no-latest", action="store_true")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    use_gpu = _HAVE_CUPY
    if not use_gpu and not args.allow_cpu:
        sys.exit("[FATAL] CuPy/CUDA unavailable. Install cupy-cuda12x or rerun with --allow-cpu.")

    default_base_delays, default_offset = condition_defaults(args.condition)
    base_delays_dt = args.base_delays_dt if args.base_delays_dt is not None else default_base_delays
    offset_dt = int(args.offset_dt) if args.offset_dt is not None else int(default_offset)

    print(f"\n{'=' * 100}")
    print("  D_M — GPU / SYNTHETIC DIMENSIONAL ENTANGLEMENT BASE GENERATOR")
    print(f"{'=' * 100}")
    print(f"  Backend      : {'cupy_gpu' if use_gpu else 'numpy_cpu_fallback'}")
    print(f"  Device       : {gpu_name() if use_gpu else 'cpu'}")
    print(f"  Condition    : {args.condition}")
    print(f"  Shots        : {args.shots}")
    print(f"  Base delays  : {base_delays_dt}")
    print(f"  Offset dt    : {offset_dt}")
    print(f"  Witnesses    : {WITNESS_LABELS}")
    print(f"  Data dir     : {DATA_DIR}")
    print("\n[MODEL]")
    print(f"  null_scale       : {args.null_scale}")
    print(f"  base_energy      : {args.base_energy}")
    print(f"  energy_gain      : {args.energy_gain}")
    print(f"  comparison_scale : {args.comparison_scale}")
    print(f"  margin_scale     : {args.margin_scale}")
    print(f"  noise_scale      : {args.noise_scale}")
    print(f"  phase_jitter     : {args.phase_jitter}")
    print(f"  offset_deform    : {args.offset_deform}")

    saved, meta, diag = build_dm_gpu_base(
        condition=str(args.condition),
        shots=int(args.shots),
        base_delays_dt=base_delays_dt,
        offset_dt=int(offset_dt),
        seed=args.seed,
        use_gpu=use_gpu,
        null_scale=float(args.null_scale),
        base_energy=float(args.base_energy),
        energy_gain=float(args.energy_gain),
        comparison_scale=float(args.comparison_scale),
        margin_scale=float(args.margin_scale),
        noise_scale=float(args.noise_scale),
        phase_jitter=float(args.phase_jitter),
        offset_deform=float(args.offset_deform),
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = str(meta["job_id"])

    out_path = Path(args.out) if args.out else DATA_DIR / f"dm_gpu_data_{args.condition}_{args.shots}shots_seed{meta['seed']}.npz"
    meta_path = Path(args.meta_out) if args.meta_out else DATA_DIR / f"dm_gpu_job_{tag}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_path, **saved)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(meta), f, indent=2)

    if not args.no_latest:
        latest_gpu = DATA_DIR / "latest_dm_gpu_data.json"
        latest_any = DATA_DIR / "latest_dm_data.json"
        latest_obj = {
            "schema": "ghost_oracle.dm.latest_pointer.v1",
            "operator": "D_M",
            "substrate": "gproj",
            "source": "gpu",
            "condition": str(args.condition),
            "job_id": tag,
            "npz": str(out_path),
            "path": str(out_path),
            "meta": str(meta_path),
            "shots": int(args.shots),
            "num_tiles": int(saved["num_tiles"]),
        }
        with open(latest_gpu, "w", encoding="utf-8") as f:
            json.dump(json_safe(latest_obj), f, indent=2)
        with open(latest_any, "w", encoding="utf-8") as f:
            json.dump(json_safe(latest_obj), f, indent=2)

    if args.verify:
        print_diagnostics(diag)

    print(f"\n{'=' * 100}")
    print("  D_M GPU BASE COMPLETE")
    print(f"{'=' * 100}")
    print(f"  Output    : {out_path}")
    print(f"  Metadata  : {meta_path}")
    print(f"  Latest    : {DATA_DIR / 'latest_dm_gpu_data.json'}")
    print(f"  Analysis  : {DATA_DIR / 'latest_dm_data.json'}")
    print(f"  Seed      : {meta['seed']}")
    print("\n  Next:")
    print("    python ghost_oracle/D_M/probes/d_m_cuda_qproj_harness.py --qpu-base <OUTPUT_PATH>")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
