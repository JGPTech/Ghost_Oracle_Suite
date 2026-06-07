#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — D_M CUDA QPROJ HARNESS
==============================================================================

Purpose
-------
Validation / benchmark harness for:

    ghost_oracle/D_M/kernels/dm_projector_kernel.cu

This script loads a D_M qproj/gproj base:

    pair[tile, shot, 2]
    tile_rung_index[tile]
    tile_witness_index[tile]    0=XY, 1=YZ, 2=ZY, 3=YX
    tile_base_delay_dt[tile]
    tile_offset_dt[tile]
    tile_total_delay_dt[tile]

and runs the CUDA projector path:

    dm_tile_correlator_kernel_u8
        raw pair records -> per-tile connected correlators

    dm_rung_projection_kernel_f32
        per-tile correlators -> D_M rung manifold

    dm_projection_summary_kernel_f32
        rung manifold -> D_M projection vector

    dm_independent_bit_shuffle_tile_kernel_u8
        destructive q0/q1 pairing control

It also computes a CPU reference projection and compares CUDA vs CPU.

Current D_M interpretation
--------------------------
D_M is a dimensional entanglement projection operator.

Current discovered witness orientation:

    YZ = primary witness dimension
    ZY = reciprocal / inverted witness dimension
    XY / YX = comparison dimensions

Core rung coordinates:

    Y  = connected(YZ)
    Z  = connected(ZY)
    R  = -Z
    E  = sqrt(Y^2 + R^2)
    S  = E - sqrt(XY^2 + YX^2)
    φ  = atan2(R, Y) mod π

Usage
-----
From repo root:

    python ghost_oracle/D_M/probes/d_m_cuda_qproj_harness.py ^
        --qpu-base ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_<JOB_ID>.npz

With explicit kernel:

    python ghost_oracle/D_M/probes/d_m_cuda_qproj_harness.py ^
        --qpu-base ghost_oracle/D_M/data/file.npz ^
        --kernel ghost_oracle/D_M/kernels/dm_projector_kernel.cu

Run quick speed reps:

    python ghost_oracle/D_M/probes/d_m_cuda_qproj_harness.py ^
        --qpu-base ghost_oracle/D_M/data/file.npz ^
        --reps 200

Outputs
-------
    analysis/dm_probe_11_cuda_qproj_harness_<timestamp>/
        result.json
        cuda_tile_stats.csv
        cuda_rung_stats.csv
        cpu_rung_stats.csv
        cuda_summary.csv
        cpu_summary.csv
        control_summary.csv

Notes
-----
This is not the final D_M benchmark. It is the qproj optimization harness:
correctness first, then speed.

==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# =============================================================================
# OPTIONAL CUDA
# =============================================================================

try:
    import cupy as cp
    HAVE_CUPY = True
except Exception:
    cp = None
    HAVE_CUPY = False


# =============================================================================
# CONSTANTS MUST MATCH dm_projector_kernel.cu
# =============================================================================

TILE_METRICS = [
    "n_shots",
    "mean_q0",
    "mean_q1",
    "corr",
    "connected",
    "p00",
    "p01",
    "p10",
    "p11",
    "q0_one_rate",
    "q1_one_rate",
    "abs_connected",
]
N_TILE_METRICS = len(TILE_METRICS)

RUNG_METRICS = [
    "XY",
    "YZ",
    "ZY",
    "YX",
    "YZ_primary",
    "ZY_return",
    "YZ_ZY_energy",
    "comparison_energy",
    "directional_specificity",
    "directional_gap",
    "inversion",
    "pi_phase",
    "pi_cos2",
    "pi_sin2",
    "base_delay",
    "offset",
    "total_delay",
    "count_all",
    "count_yzzy",
]
N_RUNG_METRICS = len(RUNG_METRICS)

SUMMARY_METRICS = [
    "n_rungs",
    "yz_mean",
    "yz_pos_frac",
    "zy_mean",
    "zy_inverted_frac",
    "yzzy_energy_mean",
    "yzzy_energy_max",
    "specificity_mean",
    "specificity_max",
    "pi_periodic_score",
    "pi_periodic_mode",
    "energy_tracking_r",
    "specificity_tracking_r",
    "phase_velocity_r",
    "phase_span_pi_units",
    "projection_score",
]
N_SUMMARY_METRICS = len(SUMMARY_METRICS)

WITNESS_LABELS = ["XY", "YZ", "ZY", "YX"]
DEFAULT_BASE_DELAYS_DT = [0, 256, 1024, 4096, 16384]
DEFAULT_OFFSET_DT = 128


# =============================================================================
# PATHS / IO
# =============================================================================

HERE = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists() or (p / "requirements.txt").exists():
            return p
    return cur


REPO_ROOT = HERE.parent  # c:\D_M
ANALYSIS_DIR = HERE / "analyze"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def scalar_str(obj: Any) -> str:
    arr = np.asarray(obj)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(obj)


def decode_str_array(arr: Any) -> List[str]:
    a = np.asarray(arr)
    out = []
    for x in a.reshape(-1):
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


def parse_tile_meta_json(npz: Any) -> List[Dict[str, Any]]:
    if "tile_meta_json" not in npz.files:
        return []
    try:
        s = scalar_str(npz["tile_meta_json"])
        return json.loads(s)
    except Exception:
        return []


# =============================================================================
# DATA LOADING / METADATA REPAIR
# =============================================================================

def infer_condition_from_delays(base: np.ndarray, off: np.ndarray, total: np.ndarray) -> str:
    if np.max(base) == 0 and np.max(off) == 0 and np.max(total) == 0:
        return "null_no_delay_no_offset"
    if np.max(base) > 0 and np.max(off) == 0:
        return "base_delay_only"
    if np.max(base) > 0 and np.max(off) > 0:
        return "base_delay_plus_offset"
    return "unknown"


def repair_metadata(num_tiles: int, base_delays_dt=None, offset_dt: int = DEFAULT_OFFSET_DT) -> Dict[str, np.ndarray]:
    """
    Fallback metadata repair for legacy broken D_M bases.

    Assumes 4 witness tiles per rung ordered as:
        XY, YZ, ZY, YX
    """
    if base_delays_dt is None:
        base_delays_dt = DEFAULT_BASE_DELAYS_DT

    tile_rung = np.zeros(num_tiles, dtype=np.int32)
    tile_wi = np.zeros(num_tiles, dtype=np.int32)
    tile_base = np.zeros(num_tiles, dtype=np.int32)
    tile_off = np.zeros(num_tiles, dtype=np.int32)
    tile_total = np.zeros(num_tiles, dtype=np.int32)

    for t in range(num_tiles):
        r = t // 4
        wi = t % 4
        base = int(base_delays_dt[min(r, len(base_delays_dt) - 1)])
        off = int(t * offset_dt)
        tile_rung[t] = r
        tile_wi[t] = wi
        tile_base[t] = base
        tile_off[t] = off
        tile_total[t] = base + off

    return {
        "tile_rung_index": tile_rung,
        "tile_witness_index": tile_wi,
        "tile_base_delay_dt": tile_base,
        "tile_offset_dt": tile_off,
        "tile_total_delay_dt": tile_total,
    }


def load_dm_base(path: Path, repair: bool = False, repair_offset_dt: int = DEFAULT_OFFSET_DT) -> Dict[str, Any]:
    npz = np.load(path, allow_pickle=True)

    if "pair" in npz.files:
        pair = np.asarray(npz["pair"], dtype=np.uint8)
    else:
        # legacy fallback: pair_tile{t}
        keys = sorted([k for k in npz.files if k.startswith("pair_tile")],
                      key=lambda k: int(k.replace("pair_tile", "")))
        if not keys:
            raise KeyError(f"No 'pair' or pair_tile* arrays found in {path}")
        pair = np.stack([np.asarray(npz[k], dtype=np.uint8) for k in keys], axis=0)

    if pair.ndim != 3 or pair.shape[2] != 2:
        raise ValueError(f"pair must have shape (tiles, shots, 2), got {pair.shape}")

    tiles, shots, bits = pair.shape
    if bits != 2:
        raise ValueError("D_M pair records must have exactly 2 bits per tile")

    meta = parse_tile_meta_json(npz)

    def arr_or_none(name: str, dtype):
        if name in npz.files:
            a = np.asarray(npz[name], dtype=dtype)
            if a.shape[0] == tiles:
                return a
        return None

    tile_rung = arr_or_none("tile_rung_index", np.int32)
    tile_wi = arr_or_none("tile_witness_index", np.int32)
    tile_base = arr_or_none("tile_base_delay_dt", np.int32)
    tile_off = arr_or_none("tile_offset_dt", np.int32)
    tile_total = arr_or_none("tile_total_delay_dt", np.int32)

    # Older/fixed generator may store witness label but not witness index.
    if tile_wi is None and "tile_witness_label" in npz.files:
        labels = decode_str_array(npz["tile_witness_label"])
        lookup = {lab: i for i, lab in enumerate(WITNESS_LABELS)}
        tile_wi = np.asarray([lookup.get(x, -1) for x in labels[:tiles]], dtype=np.int32)

    # If metadata exists but is all -1, treat as missing.
    missing = (
        tile_rung is None or tile_wi is None or tile_base is None or tile_off is None or tile_total is None
        or np.any(tile_wi < 0) or np.any(tile_base < 0) or np.any(tile_total < 0)
    )

    repaired = False
    if missing:
        if not repair:
            raise KeyError(
                "D_M base is missing usable tile metadata. "
                "Use --repair-metadata for legacy broken bases, or regenerate with fixed d_m_qpu_generate.py."
            )
        fixed = repair_metadata(tiles, offset_dt=repair_offset_dt)
        tile_rung = fixed["tile_rung_index"]
        tile_wi = fixed["tile_witness_index"]
        tile_base = fixed["tile_base_delay_dt"]
        tile_off = fixed["tile_offset_dt"]
        tile_total = fixed["tile_total_delay_dt"]
        repaired = True

    # If rung index missing but witness/base metadata present, infer by tile block.
    if tile_rung is None or np.any(tile_rung < 0):
        tile_rung = (np.arange(tiles) // 4).astype(np.int32)

    n_rungs = int(np.max(tile_rung)) + 1 if tiles else 0

    backend = scalar_str(npz["backend"]) if "backend" in npz.files else ""
    job_id = scalar_str(npz["job_id"]) if "job_id" in npz.files else ""

    return {
        "path": str(path),
        "pair": pair,
        "tiles": int(tiles),
        "shots": int(shots),
        "backend": backend,
        "job_id": job_id,
        "tile_rung_index": np.asarray(tile_rung, dtype=np.int32),
        "tile_witness_index": np.asarray(tile_wi, dtype=np.int32),
        "tile_base_delay_dt": np.asarray(tile_base, dtype=np.int32),
        "tile_offset_dt": np.asarray(tile_off, dtype=np.int32),
        "tile_total_delay_dt": np.asarray(tile_total, dtype=np.int32),
        "n_rungs": n_rungs,
        "repaired": repaired,
        "condition": infer_condition_from_delays(
            np.asarray(tile_base, dtype=np.int32),
            np.asarray(tile_off, dtype=np.int32),
            np.asarray(tile_total, dtype=np.int32),
        ),
    }


# =============================================================================
# CPU REFERENCE
# =============================================================================

def wrap_pi(x: float) -> float:
    y = math.fmod(x, math.pi)
    if y < 0:
        y += math.pi
    return y


def wrap_pi_delta(d: float) -> float:
    y = math.fmod(d + 0.5 * math.pi, math.pi)
    if y < 0:
        y += math.pi
    return y - 0.5 * math.pi


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or y.size < 3:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def normalize_x(raw: np.ndarray, mode: int) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float64).copy()
    if mode == 1:
        x = np.log1p(np.maximum(0.0, x))
    span = float(np.max(x) - np.min(x)) if x.size else 0.0
    if abs(span) <= 1e-12:
        return np.zeros_like(x)
    return (x - np.min(x)) / span


def pi_periodic_score(x_raw: np.ndarray, phase: np.ndarray, mode: int) -> float:
    if len(x_raw) < 3:
        return 0.0
    x = normalize_x(x_raw, mode)
    c2 = np.cos(2.0 * phase)
    s2 = np.sin(2.0 * phase)
    rc = safe_corr(x, c2)
    rs = safe_corr(x, s2)
    return float(min(1.0, math.sqrt(rc * rc + rs * rs)))


def cpu_tile_stats(pair: np.ndarray) -> np.ndarray:
    tiles, shots, _ = pair.shape
    out = np.zeros((tiles, N_TILE_METRICS), dtype=np.float32)

    for t in range(tiles):
        b0 = pair[t, :, 0].astype(np.uint8)
        b1 = pair[t, :, 1].astype(np.uint8)
        s0 = 1.0 - 2.0 * b0.astype(np.float64)
        s1 = 1.0 - 2.0 * b1.astype(np.float64)
        prod = s0 * s1

        mean0 = float(s0.mean())
        mean1 = float(s1.mean())
        corr = float(prod.mean())
        conn = corr - mean0 * mean1

        out[t, 0] = shots
        out[t, 1] = mean0
        out[t, 2] = mean1
        out[t, 3] = corr
        out[t, 4] = conn
        out[t, 5] = float(np.mean((b0 == 0) & (b1 == 0)))
        out[t, 6] = float(np.mean((b0 == 0) & (b1 == 1)))
        out[t, 7] = float(np.mean((b0 == 1) & (b1 == 0)))
        out[t, 8] = float(np.mean((b0 == 1) & (b1 == 1)))
        out[t, 9] = float(np.mean(b0 == 1))
        out[t, 10] = float(np.mean(b1 == 1))
        out[t, 11] = abs(conn)

    return out


def cpu_rung_stats(base: Dict[str, Any], tile_stats: np.ndarray) -> np.ndarray:
    n_rungs = int(base["n_rungs"])
    out = np.zeros((n_rungs, N_RUNG_METRICS), dtype=np.float32)

    rung_idx = base["tile_rung_index"]
    wi_idx = base["tile_witness_index"]
    bdelay = base["tile_base_delay_dt"]
    odelay = base["tile_offset_dt"]
    tdelay = base["tile_total_delay_dt"]

    for r in range(n_rungs):
        witness = np.zeros(4, dtype=np.float64)
        seen = np.zeros(4, dtype=np.int32)
        rows = np.where(rung_idx == r)[0]

        for t in rows:
            wi = int(wi_idx[t])
            if 0 <= wi < 4:
                witness[wi] = float(tile_stats[t, 4])
                seen[wi] = 1

        xy, yz, zy, yx = witness.tolist()
        ret = -zy
        energy = math.sqrt(yz * yz + ret * ret)
        comp = math.sqrt(xy * xy + yx * yx)
        spec = energy - comp
        gap = yz - zy
        inversion = -yz * zy
        phase = wrap_pi(math.atan2(ret, yz))

        denom = max(1, len(rows))
        out[r, 0] = xy
        out[r, 1] = yz
        out[r, 2] = zy
        out[r, 3] = yx
        out[r, 4] = yz
        out[r, 5] = ret
        out[r, 6] = energy
        out[r, 7] = comp
        out[r, 8] = spec
        out[r, 9] = gap
        out[r, 10] = inversion
        out[r, 11] = phase
        out[r, 12] = math.cos(2.0 * phase)
        out[r, 13] = math.sin(2.0 * phase)
        out[r, 14] = float(np.mean(bdelay[rows])) if len(rows) else 0.0
        out[r, 15] = float(np.mean(odelay[rows])) if len(rows) else 0.0
        out[r, 16] = float(np.mean(tdelay[rows])) if len(rows) else 0.0
        out[r, 17] = len(rows)
        out[r, 18] = 1.0 if seen[1] and seen[2] else 0.0

    return out


def cpu_summary(rung_stats: np.ndarray) -> np.ndarray:
    valid = rung_stats[:, 18] > 0
    rs = rung_stats[valid]
    out = np.zeros(N_SUMMARY_METRICS, dtype=np.float32)

    if rs.size == 0:
        return out

    yz = rs[:, 4].astype(np.float64)
    zy = rs[:, 2].astype(np.float64)
    energy = rs[:, 6].astype(np.float64)
    spec = rs[:, 8].astype(np.float64)
    phase = rs[:, 11].astype(np.float64)
    x_total = rs[:, 16].astype(np.float64)
    n = len(rs)

    yz_mean = float(yz.mean())
    zy_mean = float(zy.mean())
    yz_pos_frac = float(np.mean(yz > 0.0))
    zy_inv_frac = float(np.mean(yz * zy < 0.0))
    energy_mean = float(energy.mean())
    energy_max = float(energy.max())
    spec_mean = float(spec.mean())
    spec_max = float(spec.max())

    x_lin = normalize_x(x_total, 0)
    x_log = normalize_x(x_total, 1)

    er_lin = safe_corr(x_lin, energy)
    er_log = safe_corr(x_log, energy)
    sr_lin = safe_corr(x_lin, spec)
    sr_log = safe_corr(x_log, spec)

    energy_r = er_log if abs(er_log) > abs(er_lin) else er_lin
    spec_r = sr_log if abs(sr_log) > abs(sr_lin) else sr_lin

    pi_lin = pi_periodic_score(x_total, phase, 0)
    pi_log = pi_periodic_score(x_total, phase, 1)
    pi_score = pi_log if pi_log > pi_lin else pi_lin
    pi_mode = 1.0 if pi_log > pi_lin else 0.0

    phase_velocity_r = 0.0
    phase_span = 0.0

    if n >= 3:
        unwrapped = [float(phase[0])]
        mids = []
        vel = []
        accum = float(phase[0])
        for i in range(1, n):
            dx = float(x_total[i] - x_total[i - 1])
            dph = wrap_pi_delta(float(phase[i] - phase[i - 1]))
            accum += dph
            unwrapped.append(accum)
            if abs(dx) > 1e-12:
                mids.append(0.5 * float(x_total[i] + x_total[i - 1]))
                vel.append(dph / dx)
        phase_span = (max(unwrapped) - min(unwrapped)) / math.pi if unwrapped else 0.0
        if len(vel) >= 3:
            phase_velocity_r = safe_corr(normalize_x(np.asarray(mids), 1), np.asarray(vel))

    energy_term = min(1.0, energy_mean / 0.35)
    spec_term = min(1.0, max(0.0, spec_mean) / 0.25)
    yz_term = yz_pos_frac
    pi_term = pi_score
    tracking_term = 0.5 * (abs(energy_r) + abs(spec_r))

    projection = (
        0.30 * energy_term +
        0.20 * spec_term +
        0.20 * yz_term +
        0.20 * pi_term +
        0.10 * tracking_term
    )

    vals = [
        n,
        yz_mean,
        yz_pos_frac,
        zy_mean,
        zy_inv_frac,
        energy_mean,
        energy_max,
        spec_mean,
        spec_max,
        pi_score,
        pi_mode,
        energy_r,
        spec_r,
        phase_velocity_r,
        phase_span,
        projection,
    ]
    out[:] = np.asarray(vals, dtype=np.float32)
    return out


# =============================================================================
# CUDA RUNNER
# =============================================================================

def compile_kernel(kernel_path: Path):
    if not HAVE_CUPY:
        raise RuntimeError("CuPy is not installed. Install cupy-cuda12x / matching CuPy first.")
    if not kernel_path.exists():
        raise FileNotFoundError(f"CUDA kernel not found: {kernel_path}")
    code = kernel_path.read_text(encoding="utf-8")
    return cp.RawModule(code=code, options=("--std=c++11",), name_expressions=[
        "dm_tile_correlator_kernel_u8",
        "dm_independent_bit_shuffle_tile_kernel_u8",
        "dm_rung_projection_kernel_f32",
        "dm_projection_summary_kernel_f32",
    ])


def run_cuda_projection(base: Dict[str, Any], kernel_path: Path, block_size: int, control_seed: int, reps: int) -> Dict[str, Any]:
    mod = compile_kernel(kernel_path)

    k_tile = mod.get_function("dm_tile_correlator_kernel_u8")
    k_shuffle = mod.get_function("dm_independent_bit_shuffle_tile_kernel_u8")
    k_rung = mod.get_function("dm_rung_projection_kernel_f32")
    k_summary = mod.get_function("dm_projection_summary_kernel_f32")

    pair = np.asarray(base["pair"], dtype=np.uint8)
    tiles = np.int32(base["tiles"])
    shots = np.int32(base["shots"])
    n_rungs = np.int32(base["n_rungs"])

    d_pair = cp.asarray(pair)
    d_tile_rung = cp.asarray(base["tile_rung_index"].astype(np.int32))
    d_tile_wi = cp.asarray(base["tile_witness_index"].astype(np.int32))
    d_base = cp.asarray(base["tile_base_delay_dt"].astype(np.int32))
    d_off = cp.asarray(base["tile_offset_dt"].astype(np.int32))
    d_total = cp.asarray(base["tile_total_delay_dt"].astype(np.int32))

    d_tile_stats = cp.zeros((int(tiles), N_TILE_METRICS), dtype=cp.float32)
    d_control_tile_stats = cp.zeros((int(tiles), N_TILE_METRICS), dtype=cp.float32)
    d_rung_stats = cp.zeros((int(n_rungs), N_RUNG_METRICS), dtype=cp.float32)
    d_control_rung_stats = cp.zeros((int(n_rungs), N_RUNG_METRICS), dtype=cp.float32)
    d_summary = cp.zeros((N_SUMMARY_METRICS,), dtype=cp.float32)
    d_control_summary = cp.zeros((N_SUMMARY_METRICS,), dtype=cp.float32)

    # Warmup.
    k_tile((int(tiles),), (block_size,), (d_pair, tiles, shots, d_tile_stats))
    k_rung((int(n_rungs),), (1,), (
        d_tile_stats, d_tile_rung, d_tile_wi, d_base, d_off, d_total, tiles, n_rungs, d_rung_stats
    ))
    k_summary((1,), (1,), (d_rung_stats, n_rungs, d_summary))
    cp.cuda.Stream.null.synchronize()

    # Timed projection path.
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(max(1, reps)):
        k_tile((int(tiles),), (block_size,), (d_pair, tiles, shots, d_tile_stats))
        k_rung((int(n_rungs),), (1,), (
            d_tile_stats, d_tile_rung, d_tile_wi, d_base, d_off, d_total, tiles, n_rungs, d_rung_stats
        ))
        k_summary((1,), (1,), (d_rung_stats, n_rungs, d_summary))
    end.record()
    end.synchronize()
    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    per_rep_ms = elapsed_ms / max(1, reps)

    # Control path timing once/reps.
    cstart = cp.cuda.Event()
    cend = cp.cuda.Event()
    cstart.record()
    for _ in range(max(1, reps)):
        k_shuffle((int(tiles),), (block_size,), (
            d_pair, tiles, shots, np.int32(control_seed), d_control_tile_stats
        ))
        k_rung((int(n_rungs),), (1,), (
            d_control_tile_stats, d_tile_rung, d_tile_wi, d_base, d_off, d_total,
            tiles, n_rungs, d_control_rung_stats
        ))
        k_summary((1,), (1,), (d_control_rung_stats, n_rungs, d_control_summary))
    cend.record()
    cend.synchronize()
    control_elapsed_ms = cp.cuda.get_elapsed_time(cstart, cend)
    control_per_rep_ms = control_elapsed_ms / max(1, reps)

    return {
        "cuda_tile_stats": cp.asnumpy(d_tile_stats),
        "cuda_rung_stats": cp.asnumpy(d_rung_stats),
        "cuda_summary": cp.asnumpy(d_summary),
        "control_tile_stats": cp.asnumpy(d_control_tile_stats),
        "control_rung_stats": cp.asnumpy(d_control_rung_stats),
        "control_summary": cp.asnumpy(d_control_summary),
        "timing": {
            "reps": int(reps),
            "projection_total_ms": float(elapsed_ms),
            "projection_per_rep_ms": float(per_rep_ms),
            "control_total_ms": float(control_elapsed_ms),
            "control_per_rep_ms": float(control_per_rep_ms),
            "records_per_rep": int(base["tiles"] * base["shots"] * 2),
            "records_per_sec_projection": float((base["tiles"] * base["shots"] * 2) / (per_rep_ms / 1000.0)) if per_rep_ms > 0 else None,
        },
    }


# =============================================================================
# REPORT HELPERS
# =============================================================================

def tile_rows(tile_stats: np.ndarray, base: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for t in range(base["tiles"]):
        wi = int(base["tile_witness_index"][t])
        rows.append({
            "tile": t,
            "rung": int(base["tile_rung_index"][t]),
            "witness_index": wi,
            "witness": WITNESS_LABELS[wi] if 0 <= wi < 4 else "??",
            "base_delay": int(base["tile_base_delay_dt"][t]),
            "offset": int(base["tile_offset_dt"][t]),
            "total_delay": int(base["tile_total_delay_dt"][t]),
            **{TILE_METRICS[i]: float(tile_stats[t, i]) for i in range(N_TILE_METRICS)},
        })
    return rows


def rung_rows(rung_stats: np.ndarray) -> List[Dict[str, Any]]:
    rows = []
    for r in range(rung_stats.shape[0]):
        rows.append({
            "rung": r,
            **{RUNG_METRICS[i]: float(rung_stats[r, i]) for i in range(N_RUNG_METRICS)},
            "pi_phase_deg": float(rung_stats[r, 11] * 180.0 / math.pi),
        })
    return rows


def summary_row(summary: np.ndarray, label: str) -> Dict[str, Any]:
    return {
        "label": label,
        **{SUMMARY_METRICS[i]: float(summary[i]) for i in range(N_SUMMARY_METRICS)},
        "pi_periodic_mode_label": "log1p" if int(round(float(summary[10]))) == 1 else "linear",
    }


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D_M CUDA qproj projection harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--qpu-base",
        "--base",
        dest="qpu_base",
        default=None,
        help="D_M qproj/gproj .npz file. Defaults to ghost_oracle/D_M/data/latest_dm_data.json if available.",
    )
    p.add_argument(
        "--kernel",
        default=None,
        help="Path to dm_projector_kernel.cu. Defaults to ghost_oracle/D_M/kernels/dm_projector_kernel.cu.",
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--reps", type=int, default=500)
    p.add_argument("--control-seed", type=int, default=12345)
    p.add_argument("--repair-metadata", action="store_true")
    p.add_argument("--repair-offset-dt", type=int, default=DEFAULT_OFFSET_DT)
    p.add_argument("--cpu-only", action="store_true", help="Run CPU reference only, useful before CUDA is available.")
    return p.parse_args()


def resolve_default_base() -> Path:
    latest = HERE.parent / "data" / "latest_dm_data.json"
    if latest.exists():
        obj = json.loads(latest.read_text(encoding="utf-8"))
        if "npz" in obj:
            return Path(obj["npz"])
        if "path" in obj:
            return Path(obj["path"])
    raise FileNotFoundError("No --qpu-base provided and latest_dm_data.json was not found/usable.")


def main() -> None:
    args = parse_args()

    base_path = Path(args.qpu_base) if args.qpu_base else resolve_default_base()
    if not base_path.is_absolute():
        base_path = (REPO_ROOT / base_path).resolve()

    kernel_path = Path(args.kernel) if args.kernel else HERE.parent / "kernels" / "dm_projector_kernel.cu"
    if not kernel_path.is_absolute():
        kernel_path = (REPO_ROOT / kernel_path).resolve()

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"dm_probe_11_cuda_qproj_harness_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_dm_base(base_path, repair=args.repair_metadata, repair_offset_dt=args.repair_offset_dt)

    print("=" * 108)
    print("  GHOST ORACLE SUITE — D_M PROBE 11 CUDA QPROJ HARNESS")
    print("=" * 108)
    print(f"  Input     : {base_path}")
    print(f"  Kernel    : {kernel_path}")
    print(f"  Backend   : {base['backend']}")
    print(f"  Job ID    : {base['job_id']}")
    print(f"  Condition : {base['condition']}")
    print(f"  Tiles     : {base['tiles']}  Shots: {base['shots']}  Rungs: {base['n_rungs']}")
    print(f"  Repaired  : {base['repaired']}")
    print(f"  Out dir   : {out_dir}")
    print("-" * 108)

    # CPU reference.
    t0 = time.perf_counter()
    cpu_t = cpu_tile_stats(base["pair"])
    cpu_r = cpu_rung_stats(base, cpu_t)
    cpu_s = cpu_summary(cpu_r)
    cpu_elapsed = time.perf_counter() - t0

    print("  CPU REFERENCE")
    print(f"    projection_score     : {cpu_s[15]:.6f}")
    print(f"    yz_mean              : {cpu_s[1]:+.6f}")
    print(f"    yz_pos_frac          : {cpu_s[2]:.3f}")
    print(f"    yzzy_energy_mean     : {cpu_s[5]:.6f}")
    print(f"    specificity_mean     : {cpu_s[7]:+.6f}")
    print(f"    pi_periodic_score    : {cpu_s[9]:.6f}")
    print(f"    elapsed              : {cpu_elapsed*1000.0:.3f} ms")
    print("-" * 108)

    result: Dict[str, Any] = {
        "operator": "D_M",
        "script": "d_m_cuda_qproj_harness.py",
        "input": str(base_path),
        "kernel": str(kernel_path),
        "base_meta": {
            k: base[k] for k in ["backend", "job_id", "condition", "tiles", "shots", "n_rungs", "repaired"]
        },
        "cpu_elapsed_sec": cpu_elapsed,
        "cpu_summary": summary_row(cpu_s, "cpu"),
    }

    cuda_ok = False
    if not args.cpu_only:
        if not HAVE_CUPY:
            print("  CUDA")
            print("    [SKIP] CuPy is not installed. Re-run after installing matching cupy-cuda package.")
        else:
            try:
                cuda = run_cuda_projection(
                    base=base,
                    kernel_path=kernel_path,
                    block_size=args.block_size,
                    control_seed=args.control_seed,
                    reps=args.reps,
                )
                cuda_ok = True

                cs = cuda["cuda_summary"]
                ctl = cuda["control_summary"]

                print("  CUDA PROJECTION")
                print(f"    projection_score     : {cs[15]:.6f}")
                print(f"    yz_mean              : {cs[1]:+.6f}")
                print(f"    yz_pos_frac          : {cs[2]:.3f}")
                print(f"    yzzy_energy_mean     : {cs[5]:.6f}")
                print(f"    specificity_mean     : {cs[7]:+.6f}")
                print(f"    pi_periodic_score    : {cs[9]:.6f}")
                print(f"    per_rep              : {cuda['timing']['projection_per_rep_ms']:.6f} ms")
                print(f"    records/sec          : {cuda['timing']['records_per_sec_projection']:,.0f}")
                print()
                print("  CUDA CONTROL — independent_bit_shuffle")
                print(f"    projection_score     : {ctl[15]:.6f}")
                print(f"    yzzy_energy_mean     : {ctl[5]:.6f}")
                print(f"    specificity_mean     : {ctl[7]:+.6f}")
                print(f"    per_rep              : {cuda['timing']['control_per_rep_ms']:.6f} ms")
                print()
                print("  CUDA VS CPU")
                print(f"    tile max abs diff    : {max_abs_diff(cuda['cuda_tile_stats'], cpu_t):.8f}")
                print(f"    rung max abs diff    : {max_abs_diff(cuda['cuda_rung_stats'], cpu_r):.8f}")
                print(f"    summary max abs diff : {max_abs_diff(cuda['cuda_summary'], cpu_s):.8f}")
                print("-" * 108)

                result["cuda_timing"] = cuda["timing"]
                result["cuda_summary"] = summary_row(cuda["cuda_summary"], "cuda")
                result["control_summary"] = summary_row(cuda["control_summary"], "independent_bit_shuffle")
                result["validation"] = {
                    "tile_max_abs_diff": max_abs_diff(cuda["cuda_tile_stats"], cpu_t),
                    "rung_max_abs_diff": max_abs_diff(cuda["cuda_rung_stats"], cpu_r),
                    "summary_max_abs_diff": max_abs_diff(cuda["cuda_summary"], cpu_s),
                }

                write_csv(out_dir / "cuda_tile_stats.csv", tile_rows(cuda["cuda_tile_stats"], base),
                          ["tile", "rung", "witness_index", "witness", "base_delay", "offset", "total_delay"] + TILE_METRICS)
                write_csv(out_dir / "cuda_rung_stats.csv", rung_rows(cuda["cuda_rung_stats"]),
                          ["rung"] + RUNG_METRICS + ["pi_phase_deg"])
                write_csv(out_dir / "cuda_summary.csv", [summary_row(cuda["cuda_summary"], "cuda")],
                          ["label"] + SUMMARY_METRICS + ["pi_periodic_mode_label"])
                write_csv(out_dir / "control_summary.csv", [summary_row(cuda["control_summary"], "independent_bit_shuffle")],
                          ["label"] + SUMMARY_METRICS + ["pi_periodic_mode_label"])

            except Exception as e:
                print("  CUDA")
                print(f"    [ERROR] {type(e).__name__}: {e}")
                result["cuda_error"] = f"{type(e).__name__}: {e}"

    # Always write CPU outputs.
    write_csv(out_dir / "cpu_tile_stats.csv", tile_rows(cpu_t, base),
              ["tile", "rung", "witness_index", "witness", "base_delay", "offset", "total_delay"] + TILE_METRICS)
    write_csv(out_dir / "cpu_rung_stats.csv", rung_rows(cpu_r),
              ["rung"] + RUNG_METRICS + ["pi_phase_deg"])
    write_csv(out_dir / "cpu_summary.csv", [summary_row(cpu_s, "cpu")],
              ["label"] + SUMMARY_METRICS + ["pi_periodic_mode_label"])

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2)

    print(f"  [SAVED] {out_dir}")
    if not cuda_ok and not args.cpu_only:
        print("  [note] CPU reference outputs were still saved.")
    print("=" * 108)


if __name__ == "__main__":
    main()
