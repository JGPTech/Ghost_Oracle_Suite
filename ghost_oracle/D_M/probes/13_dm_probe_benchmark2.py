#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — D_M FINAL BENCHMARK
==============================================================================

D_M = Dimensional Entanglement Projection Operator

This benchmark follows the loose capstone pattern of G_M:

    1. VERIFY STORY
       qproj / gproj / geo condition projection:
           null
           base_only
           offset_on

       The verify stage checks that D_M separates null from active
       YZ-primary / ZY-reciprocal manifolds across substrates and that
       destructive pair controls collapse the active record path.

    2. CLASSICAL TASK STORY
       D_M dimensional entanglement retrieval:

           retrieve the correct directional paired manifold from a candidate
           bank containing scalar-equivalent decoys.

       The decoys preserve energy-like scalar structure but damage the
       load-bearing D_M dimensions:

           - YZ/return swap
           - reciprocal break
           - delay-order permutation
           - pi-phase scramble
           - comparison-channel decoy

       This is the D_M analogue of the G_M attack / retrieval test:
       the task is constructed so scalar similarity can be fooled while the
       directional D_M manifold should remain recoverable.

    3. CONTROLS ARE FIRST CLASS
       The benchmark reports:

           - independent-bit shuffle collapse for qproj/gproj record bases
           - DER hard-negative recall
           - DER control collapses when D_M-specific structure is scrambled

Current D_M witness orientation
-------------------------------
    YZ = primary witness dimension
    ZY = reciprocal / inverted witness dimension
    XY / YX = comparison dimensions

Projected rung coordinates
--------------------------
    Y  = connected(YZ)
    Z  = connected(ZY)
    R  = -Z
    E  = sqrt(Y^2 + R^2)
    S  = E - sqrt(XY^2 + YX^2)
    phi = atan2(R, Y) mod pi
    piW = E_mean * pi_fit_score

Non-claims
----------
This benchmark does NOT claim:

    - D_M reconstructs density matrices.
    - D_M certifies device-independent Bell nonlocality.
    - D_M proves prepared Bell states.
    - D_M is a QPU speedup or quantum advantage claim.
    - GPROJ is an IBM hardware simulator.
    - The classical DER task is a real-world application by itself.

Bounded claim
-------------
D_M projects a YZ-primary / ZY-reciprocal dimensional manifold and applies that
operator to a classical retrieval task where directional paired structure, not
scalar energy alone, is load-bearing.

Usage
-----
Full run with explicit bases:

    python ghost_oracle/D_M/d_m_benchmark.py ^
      --qproj-null   ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_<NULL>.npz ^
      --qproj-base   ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_<BASE>.npz ^
      --qproj-offset ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_<OFFSET>.npz ^
      --gproj-null   ghost_oracle/D_M/data/dm_gpu_data_null_4096shots_seed<SEED>.npz ^
      --gproj-base   ghost_oracle/D_M/data/dm_gpu_data_base_delay_4096shots_seed<SEED>.npz ^
      --gproj-offset ghost_oracle/D_M/data/dm_gpu_data_offset_deformed_4096shots_seed<SEED>.npz ^
      --classical

Classical DER only:

    python ghost_oracle/D_M/d_m_benchmark.py --skip-verify --classical --probe

Verify only:

    python ghost_oracle/D_M/d_m_benchmark.py --skip-classical

Outputs
-------
    ghost_oracle/D_M/analysis/d_m_final_<timestamp>/
        result.json
        verify_projection_summary.csv
        verify_rung_projection.csv
        verify_condition_separation.csv
        verify_substrate_agreement.csv
        verify_control_collapse.csv
        der_summary.csv
        der_controls.csv
        artifacts.npz

==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import secrets
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp
    _HAVE_CUPY = True
    _CUPY_IMPORT_ERROR = None
except Exception as e:
    cp = None
    _HAVE_CUPY = False
    _CUPY_IMPORT_ERROR = repr(e)

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    plt = None
    _HAVE_MPL = False


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

HERE = Path(__file__).resolve().parent
D_M_DIR = HERE.parent
DATA_DIR = D_M_DIR / "data"
ANALYSIS_DIR = HERE / "analyze"
KERNEL_PATH = D_M_DIR / "kernels" / "dm_projector_kernel.cu"

DEFAULT_BASE_DELAYS = [0, 256, 1024, 4096, 16384]
DEFAULT_NULL_DELAYS = [0, 0, 0, 0, 0]
DEFAULT_OFFSET_DT = 128
DEFAULT_QPROJ_NULL   = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fm4ihvjngc73aq3ccg.npz"
DEFAULT_QPROJ_BASE   = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8flk2jo3njc73f0g560.npz"
DEFAULT_QPROJ_OFFSET = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fl82bo3njc73f0fgd0.npz"
DEFAULT_GPROJ_NULL   = DATA_DIR / "dm_gpu_data_null_4096shots_seed9031229662612491082.npz"
DEFAULT_GPROJ_BASE   = DATA_DIR / "dm_gpu_data_base_delay_4096shots_seed2669559634056472362.npz"
DEFAULT_GPROJ_OFFSET = DATA_DIR / "dm_gpu_data_offset_deformed_4096shots_seed6727069190982977623.npz"

WITNESS_LABELS = ["XY", "YZ", "ZY", "YX"]
WITNESS_PAIRS = [(0, 1), (1, 2), (2, 1), (1, 0)]

TILE_METRICS = [
    "n_shots", "mean_q0", "mean_q1", "corr", "connected",
    "p00", "p01", "p10", "p11",
    "q0_one_rate", "q1_one_rate", "abs_connected",
]
N_TILE_METRICS = len(TILE_METRICS)

RUNG_METRICS = [
    "XY", "YZ", "ZY", "YX",
    "YZ_primary", "ZY_return",
    "YZ_ZY_energy", "comparison_energy",
    "directional_specificity", "directional_gap", "inversion",
    "pi_phase", "pi_cos2", "pi_sin2",
    "base_delay", "offset", "total_delay",
    "count_all", "count_yzzy",
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

SUMMARY_COMPARE_FIELDS = [
    "yz_mean",
    "yz_pos_frac",
    "zy_mean",
    "zy_inverted_frac",
    "yzzy_energy_mean",
    "specificity_mean",
    "pi_periodic_score",
    "pi_witness_strength",
    "energy_tracking_r",
    "specificity_tracking_r",
    "phase_velocity_r",
    "phase_span_pi_units",
    "projection_score",
]

DER_TOP_K = [1, 5, 10]


# =============================================================================
# UTILS
# =============================================================================

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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2)


def write_csv(path: Path, rows: List[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
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


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a = a[:n]
    b = b[:n]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    if n <= 0:
        return float("nan")
    return float(np.linalg.norm(a[:n] - b[:n]))


def normalize_x(x: np.ndarray, mode: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    if mode == 1:
        x = np.log1p(np.maximum(0.0, x))
    span = float(np.max(x) - np.min(x)) if x.size else 0.0
    if abs(span) <= 1e-12:
        return np.zeros_like(x)
    return (x - np.min(x)) / span


def wrap_pi(x: float) -> float:
    y = math.fmod(float(x), math.pi)
    if y < 0:
        y += math.pi
    return y


def wrap_pi_delta(d: float) -> float:
    y = math.fmod(float(d) + 0.5 * math.pi, math.pi)
    if y < 0:
        y += math.pi
    return y - 0.5 * math.pi


def pi_periodic_score(x_raw: np.ndarray, phase: np.ndarray, mode: int) -> float:
    if len(x_raw) < 3:
        return 0.0
    x = normalize_x(x_raw, mode)
    c2 = np.cos(2.0 * phase)
    s2 = np.sin(2.0 * phase)
    rc = safe_corr(x, c2)
    rs = safe_corr(x, s2)
    return float(min(1.0, math.sqrt(rc * rc + rs * rs)))


def section(title: str, width: int = 112) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# =============================================================================
# VERIFY DATA MODEL
# =============================================================================

@dataclass
class DMProjection:
    substrate: str
    condition: str
    condition_label: str
    source: str
    backend: str
    job_id: str
    tiles: int
    shots: int
    n_rungs: int
    tile_stats: Optional[np.ndarray]
    rung_stats: np.ndarray
    summary: np.ndarray
    control_summary: Optional[np.ndarray]
    timing: Dict[str, Any]
    notes: str = ""


@dataclass
class DMCudaContext:
    module: Any
    k_tile: Any
    k_shuffle: Any
    k_rung: Any
    k_summary: Any
    k_geo: Optional[Any]
    k_geo_sweep: Optional[Any]
    k_der: Optional[Any]
    path: Path


def compile_dm_cuda(args: argparse.Namespace) -> Optional[DMCudaContext]:
    if args.no_cuda:
        return None
    if not _HAVE_CUPY:
        if not args.quiet:
            print(f"[CUDA][skip] CuPy unavailable: {_CUPY_IMPORT_ERROR}")
        return None
    if not KERNEL_PATH.exists():
        if not args.quiet:
            print(f"[CUDA][skip] kernel missing: {KERNEL_PATH}")
        return None

    try:
        src = KERNEL_PATH.read_text(encoding="utf-8")
        mod = cp.RawModule(
            code=src,
            options=("--std=c++11",),
            name_expressions=[
                "dm_tile_correlator_kernel_u8",
                "dm_independent_bit_shuffle_tile_kernel_u8",
                "dm_rung_projection_kernel_f32",
                "dm_projection_summary_kernel_f32",
                "dm_geo_rung_projection_kernel_f32",
                "dm_geo_sweep_summary_kernel_f32",
                "dm_der_topk_kernel_f32",
            ],
        )
        return DMCudaContext(
            module=mod,
            k_tile=mod.get_function("dm_tile_correlator_kernel_u8"),
            k_shuffle=mod.get_function("dm_independent_bit_shuffle_tile_kernel_u8"),
            k_rung=mod.get_function("dm_rung_projection_kernel_f32"),
            k_summary=mod.get_function("dm_projection_summary_kernel_f32"),
            k_geo=mod.get_function("dm_geo_rung_projection_kernel_f32"),
            k_geo_sweep=mod.get_function("dm_geo_sweep_summary_kernel_f32"),
            k_der=mod.get_function("dm_der_topk_kernel_f32"),
            path=KERNEL_PATH,
        )
    except Exception as e:
        if not args.quiet:
            print(f"[CUDA][warn] Could not compile full D_M kernel: {type(e).__name__}: {e}")
            print("[CUDA][warn] Falling back to CPU verify projection.")
        return None


# =============================================================================
# BASE LOADING / REPAIR
# =============================================================================

def infer_condition_from_delays(base: np.ndarray, off: np.ndarray, total: np.ndarray) -> Tuple[str, str]:
    if np.max(base) == 0 and np.max(off) == 0 and np.max(total) == 0:
        return "null", "no_delay_no_offset"
    if np.max(base) > 0 and np.max(off) == 0:
        return "base_only", "base_delay_only"
    if np.max(base) > 0 and np.max(off) > 0:
        return "offset_on", "base_delay_plus_offset"
    return "unknown", "unknown"


def build_metadata_from_condition(condition: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if condition == "null":
        base_delays = DEFAULT_NULL_DELAYS
        offset_dt = 0
    elif condition == "base_only":
        base_delays = DEFAULT_BASE_DELAYS
        offset_dt = 0
    elif condition == "offset_on":
        base_delays = DEFAULT_BASE_DELAYS
        offset_dt = DEFAULT_OFFSET_DT
    else:
        raise ValueError(condition)

    tile_rung = []
    tile_wi = []
    tile_base = []
    tile_off = []
    tile_total = []
    t = 0

    for r, base in enumerate(base_delays):
        for wi in range(4):
            off = t * offset_dt
            tile_rung.append(r)
            tile_wi.append(wi)
            tile_base.append(base)
            tile_off.append(off)
            tile_total.append(base + off)
            t += 1

    return (
        np.asarray(tile_rung, dtype=np.int32),
        np.asarray(tile_wi, dtype=np.int32),
        np.asarray(tile_base, dtype=np.int32),
        np.asarray(tile_off, dtype=np.int32),
        np.asarray(tile_total, dtype=np.int32),
    )


def load_dm_base(path: Path, substrate: str, condition_hint: Optional[str], repair: bool) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)

    if "pair" in z.files:
        pair = np.asarray(z["pair"], dtype=np.uint8)
    else:
        keys = sorted(
            [k for k in z.files if k.startswith("pair_tile")],
            key=lambda k: int(k.replace("pair_tile", "")),
        )
        if not keys:
            raise KeyError(f"{path} has no pair or pair_tile* arrays.")
        pair = np.stack([np.asarray(z[k], dtype=np.uint8) for k in keys], axis=0)

    if pair.ndim != 3 or pair.shape[2] != 2:
        raise ValueError(f"{path} pair must have shape (tiles, shots, 2), got {pair.shape}")

    tiles, shots, _ = pair.shape

    def arr(name: str, dtype) -> Optional[np.ndarray]:
        if name in z.files:
            a = np.asarray(z[name], dtype=dtype)
            if a.shape[0] == tiles:
                return a
        return None

    tile_rung = arr("tile_rung_index", np.int32)
    tile_wi = arr("tile_witness_index", np.int32)
    tile_base = arr("tile_base_delay_dt", np.int32)
    tile_off = arr("tile_offset_dt", np.int32)
    tile_total = arr("tile_total_delay_dt", np.int32)

    if tile_wi is None and "tile_witness_label" in z.files:
        labels = decode_str_array(z["tile_witness_label"])
        lookup = {lab: i for i, lab in enumerate(WITNESS_LABELS)}
        tile_wi = np.asarray([lookup.get(x, -1) for x in labels[:tiles]], dtype=np.int32)

    missing = (
        tile_rung is None or tile_wi is None or tile_base is None or tile_off is None or tile_total is None
        or np.any(tile_wi < 0) or np.any(tile_base < 0) or np.any(tile_total < 0)
    )

    repaired = False
    if missing:
        if not repair:
            raise KeyError(f"{path} is missing usable D_M metadata. Use --repair-metadata.")
        if condition_hint is None:
            raise KeyError("Cannot repair metadata without condition hint.")
        tile_rung, tile_wi, tile_base, tile_off, tile_total = build_metadata_from_condition(condition_hint)
        if len(tile_rung) != tiles:
            raise ValueError(f"repair metadata length {len(tile_rung)} != tiles {tiles}")
        repaired = True

    if tile_rung is None or np.any(tile_rung < 0):
        tile_rung = (np.arange(tiles) // 4).astype(np.int32)

    condition, label = infer_condition_from_delays(tile_base, tile_off, tile_total)
    if condition_hint is not None:
        condition = condition_hint
        label = {
            "null": "no_delay_no_offset",
            "base_only": "base_delay_only",
            "offset_on": "base_delay_plus_offset",
        }.get(condition_hint, label)

    backend = scalar_str(z["backend"]) if "backend" in z.files else ""
    job_id = scalar_str(z["job_id"]) if "job_id" in z.files else ""

    return {
        "path": str(path),
        "substrate": substrate,
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
        "n_rungs": int(np.max(tile_rung)) + 1 if tiles else 0,
        "condition": condition,
        "condition_label": label,
        "repaired": repaired,
    }


# =============================================================================
# CPU VERIFY PROJECTION
# =============================================================================

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


def rung_stats_from_tile_stats(base: Dict[str, Any], tile_stats: np.ndarray) -> np.ndarray:
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

        vals = [
            xy, yz, zy, yx,
            yz, ret,
            energy, comp,
            spec, gap, inversion,
            phase, math.cos(2.0 * phase), math.sin(2.0 * phase),
            float(np.mean(bdelay[rows])) if len(rows) else 0.0,
            float(np.mean(odelay[rows])) if len(rows) else 0.0,
            float(np.mean(tdelay[rows])) if len(rows) else 0.0,
            float(len(rows)),
            1.0 if seen[1] and seen[2] else 0.0,
        ]
        out[r, :] = np.asarray(vals, dtype=np.float32)

    return out


def summary_from_rung_stats(rung_stats: np.ndarray) -> np.ndarray:
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

    # Non-normalized physical projection score.
    pi_witness_strength = max(0.0, energy_mean) * pi_score
    projection = (
        0.35 * max(0.0, energy_mean) +
        0.25 * max(0.0, spec_mean) +
        0.15 * max(0.0, yz_mean) +
        0.15 * pi_witness_strength +
        0.10 * (0.5 * (abs(energy_r) + abs(spec_r)))
    )

    vals = [
        n, yz_mean, yz_pos_frac, zy_mean, zy_inv_frac,
        energy_mean, energy_max, spec_mean, spec_max,
        pi_score, pi_mode,
        energy_r, spec_r, phase_velocity_r, phase_span,
        projection,
    ]
    out[:] = np.asarray(vals, dtype=np.float32)
    return out


# =============================================================================
# CUDA VERIFY PROJECTION
# =============================================================================

def project_record_cuda(base: Dict[str, Any], ctx: DMCudaContext, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    pair = np.asarray(base["pair"], dtype=np.uint8)
    tiles = np.int32(base["tiles"])
    shots = np.int32(base["shots"])
    n_rungs = np.int32(base["n_rungs"])
    block_size = int(args.cuda_threads)

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

    ctx.k_tile((int(tiles),), (block_size,), (d_pair, tiles, shots, d_tile_stats))
    ctx.k_rung((int(n_rungs),), (1,), (
        d_tile_stats, d_tile_rung, d_tile_wi, d_base, d_off, d_total,
        tiles, n_rungs, d_rung_stats,
    ))
    ctx.k_summary((1,), (1,), (d_rung_stats, n_rungs, d_summary))
    cp.cuda.Stream.null.synchronize()

    reps = max(1, int(args.reps))
    ev0 = cp.cuda.Event()
    ev1 = cp.cuda.Event()
    ev0.record()
    for _ in range(reps):
        ctx.k_tile((int(tiles),), (block_size,), (d_pair, tiles, shots, d_tile_stats))
        ctx.k_rung((int(n_rungs),), (1,), (
            d_tile_stats, d_tile_rung, d_tile_wi, d_base, d_off, d_total,
            tiles, n_rungs, d_rung_stats,
        ))
        ctx.k_summary((1,), (1,), (d_rung_stats, n_rungs, d_summary))
    ev1.record()
    ev1.synchronize()
    elapsed_ms = cp.cuda.get_elapsed_time(ev0, ev1)
    per_rep_ms = elapsed_ms / reps

    c0 = cp.cuda.Event()
    c1 = cp.cuda.Event()
    c0.record()
    for _ in range(reps):
        ctx.k_shuffle((int(tiles),), (block_size,), (
            d_pair, tiles, shots, np.int32(args.control_seed), d_control_tile_stats,
        ))
        ctx.k_rung((int(n_rungs),), (1,), (
            d_control_tile_stats, d_tile_rung, d_tile_wi, d_base, d_off, d_total,
            tiles, n_rungs, d_control_rung_stats,
        ))
        ctx.k_summary((1,), (1,), (d_control_rung_stats, n_rungs, d_control_summary))
    c1.record()
    c1.synchronize()
    control_ms = cp.cuda.get_elapsed_time(c0, c1)

    records = int(base["tiles"] * base["shots"] * 2)
    timing = {
        "mode": "cuda",
        "reps": reps,
        "projection_total_ms": float(elapsed_ms),
        "projection_per_rep_ms": float(per_rep_ms),
        "control_total_ms": float(control_ms),
        "control_per_rep_ms": float(control_ms / reps),
        "records_per_rep": records,
        "records_per_sec_projection": float(records / (per_rep_ms / 1000.0)) if per_rep_ms > 0 else None,
        "records_semantics": "pair_bit_records_per_second",
    }

    return (
        cp.asnumpy(d_tile_stats),
        cp.asnumpy(d_rung_stats),
        cp.asnumpy(d_summary),
        cp.asnumpy(d_control_summary),
        timing,
    )


def project_record_cpu(base: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    t0 = time.perf_counter()
    tile = cpu_tile_stats(base["pair"])
    rung = rung_stats_from_tile_stats(base, tile)
    summary = summary_from_rung_stats(rung)
    elapsed = time.perf_counter() - t0
    timing = {
        "mode": "cpu",
        "projection_per_rep_ms": float(elapsed * 1000.0),
        "records_per_rep": int(base["tiles"] * base["shots"] * 2),
        "records_per_sec_projection": float((base["tiles"] * base["shots"] * 2) / elapsed) if elapsed > 0 else None,
        "records_semantics": "pair_bit_records_per_second",
    }
    return tile, rung, summary, None, timing


# =============================================================================
# GEO VERIFY PATH
# =============================================================================

def geo_x_values(base_delays: Sequence[int], offset_dt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = []
    off = []
    for r, bd in enumerate(base_delays):
        quartet_offsets = [(4 * r + wi) * int(offset_dt) for wi in range(4)]
        off_mean = float(np.mean(quartet_offsets))
        off.append(off_mean)
        total.append(float(bd) + off_mean)
    return np.asarray(base_delays, dtype=np.float64), np.asarray(off, dtype=np.float64), np.asarray(total, dtype=np.float64)


def geo_condition_projection(condition: str, args: argparse.Namespace) -> DMProjection:
    if condition == "null":
        base_delays = DEFAULT_NULL_DELAYS
        offset_dt = 0
        label = "no_delay_no_offset"
    elif condition == "base_only":
        base_delays = DEFAULT_BASE_DELAYS
        offset_dt = 0
        label = "base_delay_only"
    elif condition == "offset_on":
        base_delays = DEFAULT_BASE_DELAYS
        offset_dt = DEFAULT_OFFSET_DT
        label = "base_delay_plus_offset"
    else:
        raise ValueError(condition)

    base, off, total = geo_x_values(base_delays, offset_dt)
    n = len(base_delays)
    x_log = normalize_x(total, 1)
    x_lin = normalize_x(total, 0)
    rung = np.zeros((n, N_RUNG_METRICS), dtype=np.float32)

    for r in range(n):
        if condition == "null":
            yz = 0.010 * math.sin(1.7 * r + 0.2)
            zy = 0.010 * math.cos(1.3 * r + 0.5)
            xy = 0.010 * math.cos(0.9 * r)
            yx = 0.010 * math.sin(0.8 * r + 0.4)
        else:
            x = float(x_log[r])
            xl = float(x_lin[r])
            amp = args.geo_base_energy + args.geo_energy_gain * (0.20 + 0.80 * (x ** 1.10))
            phase = math.pi * (0.05 + args.geo_phase_scale * x)

            if condition == "offset_on":
                phase += args.geo_offset_deform * math.sin(2.0 * math.pi * xl + 0.35)
                amp *= 0.90 + 0.18 * math.cos(2.0 * math.pi * xl + 0.17)

            phase = phase % math.pi
            yz = amp * math.cos(phase)
            ret = amp * math.sin(phase)
            zy = -ret

            if yz < 0:
                yz = -0.35 * yz

            xy = args.geo_comparison_scale * math.cos(1.2 * r + 0.3)
            yx = args.geo_comparison_scale * math.sin(1.1 * r + 0.9)

        ret = -zy
        energy = math.sqrt(yz * yz + ret * ret)
        comp = math.sqrt(xy * xy + yx * yx)
        spec = energy - comp
        gap = yz - zy
        inversion = -yz * zy
        phase2 = wrap_pi(math.atan2(ret, yz))

        vals = [
            xy, yz, zy, yx,
            yz, ret,
            energy, comp,
            spec, gap, inversion,
            phase2, math.cos(2.0 * phase2), math.sin(2.0 * phase2),
            float(base[r]), float(off[r]), float(total[r]),
            4.0, 1.0,
        ]
        rung[r, :] = np.asarray(vals, dtype=np.float32)

    summary = summary_from_rung_stats(rung)

    return DMProjection(
        substrate="geo",
        condition=condition,
        condition_label=label,
        source="analytic_geo",
        backend="analytic_classical",
        job_id=f"geo_{condition}",
        tiles=n * 4,
        shots=0,
        n_rungs=n,
        tile_stats=None,
        rung_stats=rung,
        summary=summary,
        control_summary=None,
        timing={"mode": "geo", "projection_per_rep_ms": 0.0, "records_per_sec_projection": None},
        notes="Analytic D_M manifold. No shot records.",
    )


def run_geo_sweep_cuda(ctx: DMCudaContext, args: argparse.Namespace) -> Dict[str, Any]:
    if ctx.k_geo_sweep is None or args.skip_geo_sweep:
        return {"enabled": False, "reason": "disabled_or_missing"}

    n_candidates = max(1, int(args.geo_sweep_candidates))
    reps = max(1, int(args.geo_sweep_reps))
    n_rungs = len(DEFAULT_BASE_DELAYS)
    n_conditions = 3
    points_per_rep = n_candidates * n_conditions * n_rungs * 4

    idx = cp.arange(n_candidates, dtype=cp.float32)
    denom = cp.float32(max(1, n_candidates - 1))
    u = idx / denom

    base_energy = cp.asarray(args.geo_base_energy + 0.010 * cp.sin(17.0 * u + 0.10), dtype=cp.float32)
    energy_gain = cp.asarray(args.geo_energy_gain + 0.060 * cp.sin(23.0 * u + 0.70), dtype=cp.float32)
    comparison = cp.asarray(args.geo_comparison_scale + 0.004 * cp.sin(31.0 * u + 1.20), dtype=cp.float32)
    offset_def = cp.asarray(args.geo_offset_deform + 0.080 * cp.sin(19.0 * u + 0.30), dtype=cp.float32)
    phase_scale = cp.asarray(args.geo_phase_scale + 0.090 * cp.sin(29.0 * u + 0.50), dtype=cp.float32)
    phase_shift = cp.asarray(0.18 * cp.sin(37.0 * u + 0.90), dtype=cp.float32)

    d_base = cp.asarray(np.asarray(DEFAULT_BASE_DELAYS, dtype=np.int32))
    d_out = cp.empty((n_candidates, n_conditions, N_SUMMARY_METRICS), dtype=cp.float32)

    args_tuple = (
        d_base,
        np.int32(n_rungs),
        np.int32(DEFAULT_OFFSET_DT),
        np.int32(n_candidates),
        base_energy,
        energy_gain,
        comparison,
        offset_def,
        phase_scale,
        phase_shift,
        d_out,
    )

    ctx.k_geo_sweep((n_candidates, n_conditions), (1,), args_tuple)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(reps):
        ctx.k_geo_sweep((n_candidates, n_conditions), (1,), args_tuple)
    end.record()
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    per_rep_ms = elapsed_ms / float(reps)
    points_per_sec = float(points_per_rep / (per_rep_ms / 1000.0)) if per_rep_ms > 0 else None
    candidates_per_sec = float(n_candidates * n_conditions / (per_rep_ms / 1000.0)) if per_rep_ms > 0 else None

    sanity = cp.asnumpy(d_out[0, :, :]).astype(np.float32)

    return {
        "enabled": True,
        "mode": "geo_sweep_cuda",
        "candidates": int(n_candidates),
        "conditions": int(n_conditions),
        "rungs": int(n_rungs),
        "points_per_rep": int(points_per_rep),
        "reps": int(reps),
        "total_ms": float(elapsed_ms),
        "per_rep_ms": float(per_rep_ms),
        "analytic_points_per_second": points_per_sec,
        "candidate_conditions_per_second": candidates_per_sec,
        "sanity_summary_candidate0": sanity,
    }


# =============================================================================
# VERIFY REPORT ROWS
# =============================================================================

def summary_dict(summary: np.ndarray) -> Dict[str, float]:
    return {SUMMARY_METRICS[i]: float(summary[i]) for i in range(N_SUMMARY_METRICS)}


def add_pi_witness(d: Dict[str, float]) -> Dict[str, float]:
    d["pi_witness_strength"] = float(d["yzzy_energy_mean"] * d["pi_periodic_score"])
    return d


def projection_summary_row(p: DMProjection) -> Dict[str, Any]:
    d = add_pi_witness(summary_dict(p.summary))
    return {
        "substrate": p.substrate,
        "condition": p.condition,
        "condition_label": p.condition_label,
        "source": p.source,
        "backend": p.backend,
        "job_id": p.job_id,
        "tiles": p.tiles,
        "shots": p.shots,
        "n_rungs": p.n_rungs,
        "timing_mode": p.timing.get("mode"),
        "projection_per_rep_ms": p.timing.get("projection_per_rep_ms", ""),
        "records_per_sec_projection": p.timing.get("records_per_sec_projection", ""),
        "records_semantics": p.timing.get("records_semantics", ""),
        "pi_periodic_mode_label": "log1p" if int(round(d["pi_periodic_mode"])) == 1 else "linear",
        **d,
    }


def summary_vector(p: DMProjection) -> np.ndarray:
    d = add_pi_witness(summary_dict(p.summary))
    return np.asarray([d[k] for k in SUMMARY_COMPARE_FIELDS], dtype=np.float32)


def control_row(p: DMProjection) -> Optional[Dict[str, Any]]:
    if p.control_summary is None:
        return None
    real = add_pi_witness(summary_dict(p.summary))
    ctrl = add_pi_witness(summary_dict(p.control_summary))
    return {
        "substrate": p.substrate,
        "condition": p.condition,
        "control": "independent_bit_shuffle",
        "projection_real": real["projection_score"],
        "projection_control": ctrl["projection_score"],
        "projection_delta": real["projection_score"] - ctrl["projection_score"],
        "projection_drop_fraction": (
            (real["projection_score"] - ctrl["projection_score"]) / real["projection_score"]
            if abs(real["projection_score"]) > 1e-12 else float("nan")
        ),
        "energy_real": real["yzzy_energy_mean"],
        "energy_control": ctrl["yzzy_energy_mean"],
        "piW_real": real["pi_witness_strength"],
        "piW_control": ctrl["pi_witness_strength"],
        "specificity_real": real["specificity_mean"],
        "specificity_control": ctrl["specificity_mean"],
    }


def rung_rows(p: DMProjection) -> List[Dict[str, Any]]:
    rows = []
    for r in range(p.rung_stats.shape[0]):
        item = {
            "substrate": p.substrate,
            "condition": p.condition,
            "rung": r,
            "source": p.source,
        }
        item.update({RUNG_METRICS[i]: float(p.rung_stats[r, i]) for i in range(N_RUNG_METRICS)})
        item["pi_phase_deg"] = float(p.rung_stats[r, 11] * 180.0 / math.pi)
        rows.append(item)
    return rows


def condition_separation(projs: List[DMProjection]) -> List[Dict[str, Any]]:
    rows = []
    by_sub: Dict[str, Dict[str, DMProjection]] = {}
    for p in projs:
        by_sub.setdefault(p.substrate, {})[p.condition] = p

    pairs = [("null", "base_only"), ("null", "offset_on"), ("base_only", "offset_on")]
    for sub, conds in sorted(by_sub.items()):
        for a, b in pairs:
            if a not in conds or b not in conds:
                continue
            va = add_pi_witness(summary_dict(conds[a].summary))
            vb = add_pi_witness(summary_dict(conds[b].summary))
            fa = summary_vector(conds[a])
            fb = summary_vector(conds[b])
            rows.append({
                "substrate": sub,
                "condition_a": a,
                "condition_b": b,
                "delta_projection": vb["projection_score"] - va["projection_score"],
                "delta_energy": vb["yzzy_energy_mean"] - va["yzzy_energy_mean"],
                "delta_specificity": vb["specificity_mean"] - va["specificity_mean"],
                "delta_pi_score": vb["pi_periodic_score"] - va["pi_periodic_score"],
                "delta_piW": vb["pi_witness_strength"] - va["pi_witness_strength"],
                "summary_l2": l2(fa, fb),
                "summary_corr": safe_corr(fa, fb),
            })
    return rows


def substrate_agreement(projs: List[DMProjection]) -> List[Dict[str, Any]]:
    rows = []
    by_cond: Dict[str, Dict[str, DMProjection]] = {}
    for p in projs:
        by_cond.setdefault(p.condition, {})[p.substrate] = p

    for cond, subs in sorted(by_cond.items()):
        names = sorted(subs.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pa, pb = subs[a], subs[b]
                va = summary_vector(pa)
                vb = summary_vector(pb)
                n = min(pa.rung_stats.shape[0], pb.rung_stats.shape[0])
                rows.append({
                    "condition": cond,
                    "substrate_a": a,
                    "substrate_b": b,
                    "summary_corr": safe_corr(va, vb),
                    "summary_l2": l2(va, vb),
                    "projection_a": float(pa.summary[15]),
                    "projection_b": float(pb.summary[15]),
                    "projection_delta_abs": abs(float(pa.summary[15]) - float(pb.summary[15])),
                    "rung_energy_corr": safe_corr(pa.rung_stats[:n, 6], pb.rung_stats[:n, 6]),
                    "rung_specificity_corr": safe_corr(pa.rung_stats[:n, 8], pb.rung_stats[:n, 8]),
                    "rung_yz_corr": safe_corr(pa.rung_stats[:n, 1], pb.rung_stats[:n, 1]),
                    "rung_pi_cos2_corr": safe_corr(np.cos(2.0 * pa.rung_stats[:n, 11]), np.cos(2.0 * pb.rung_stats[:n, 11])),
                })
    return rows


def load_and_project(path: Path, substrate: str, condition: str, ctx: Optional[DMCudaContext], args: argparse.Namespace) -> DMProjection:
    base = load_dm_base(path, substrate, condition_hint=condition, repair=args.repair_metadata)

    if ctx is not None:
        tile, rung, summary, control_summary, timing = project_record_cuda(base, ctx, args)
    else:
        tile, rung, summary, control_summary, timing = project_record_cpu(base)

    return DMProjection(
        substrate=substrate,
        condition=condition,
        condition_label=base["condition_label"],
        source=str(path),
        backend=base["backend"],
        job_id=base["job_id"],
        tiles=base["tiles"],
        shots=base["shots"],
        n_rungs=base["n_rungs"],
        tile_stats=tile,
        rung_stats=rung,
        summary=summary,
        control_summary=control_summary,
        timing=timing,
        notes="record_projection",
    )


# =============================================================================
# CLASSICAL DER ENVIRONMENT
# =============================================================================

@dataclass
class DEREnvironment:
    query: Dict[str, np.ndarray]
    keys: Dict[str, np.ndarray]
    truth: np.ndarray
    candidate_kind: np.ndarray
    group_id: np.ndarray
    config: Dict[str, Any]


def make_true_manifold_arrays(
    n: int,
    rungs: int,
    rng: np.random.Generator,
    base_energy: float,
    energy_gain: float,
    comparison_scale: float,
    offset_deform: float,
    noise: float = 0.0,
) -> Dict[str, np.ndarray]:
    delay = normalize_x(np.asarray(DEFAULT_BASE_DELAYS[:rungs], dtype=np.float64), 1).astype(np.float32)
    lin_delay = normalize_x(np.asarray(DEFAULT_BASE_DELAYS[:rungs], dtype=np.float64), 0).astype(np.float32)

    amp0 = rng.uniform(base_energy, base_energy + energy_gain, size=(n, 1)).astype(np.float32)
    amp_slope = rng.uniform(0.60, 1.15, size=(n, 1)).astype(np.float32)
    phase_scale = rng.uniform(0.22, 0.52, size=(n, 1)).astype(np.float32)
    phase_shift = rng.uniform(-0.15, 0.15, size=(n, 1)).astype(np.float32)
    deform = rng.uniform(-offset_deform, offset_deform, size=(n, 1)).astype(np.float32)

    x = delay.reshape(1, rungs)
    xl = lin_delay.reshape(1, rungs)

    amp = amp0 * (0.45 + amp_slope * (0.20 + 0.80 * (x ** 1.10)))
    phase = np.pi * (0.05 + phase_scale * x) + phase_shift + deform * np.sin(2.0 * np.pi * xl + 0.35)
    phase = np.mod(phase, np.pi).astype(np.float32)

    yz = amp * np.cos(phase)
    ret = amp * np.sin(phase)
    yz = np.where(yz < 0.0, -0.35 * yz, yz)

    xy = comparison_scale * np.cos(1.2 * np.arange(rungs, dtype=np.float32)[None, :] + 0.3)
    yx = comparison_scale * np.sin(1.1 * np.arange(rungs, dtype=np.float32)[None, :] + 0.9)
    xy = np.tile(xy, (n, 1)).astype(np.float32)
    yx = np.tile(yx, (n, 1)).astype(np.float32)

    if noise > 0.0:
        yz = yz + rng.normal(0.0, noise, size=yz.shape).astype(np.float32)
        ret = ret + rng.normal(0.0, noise, size=ret.shape).astype(np.float32)
        xy = xy + rng.normal(0.0, noise * 0.5, size=xy.shape).astype(np.float32)
        yx = yx + rng.normal(0.0, noise * 0.5, size=yx.shape).astype(np.float32)

    energy = np.sqrt(yz * yz + ret * ret).astype(np.float32)
    comp = np.sqrt(xy * xy + yx * yx).astype(np.float32)
    spec = (energy - comp).astype(np.float32)
    phase2 = np.mod(np.arctan2(ret, yz), np.pi).astype(np.float32)
    c2 = np.cos(2.0 * phase2).astype(np.float32)
    s2 = np.sin(2.0 * phase2).astype(np.float32)

    return {
        "yz": yz.astype(np.float32),
        "ret": ret.astype(np.float32),
        "xy": xy.astype(np.float32),
        "yx": yx.astype(np.float32),
        "energy": energy,
        "spec": spec,
        "cos2": c2,
        "sin2": s2,
        "delay": np.tile(delay.reshape(1, rungs), (n, 1)).astype(np.float32),
    }


def clone_features(feat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in feat.items()}


def recompute_derived(feat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    yz = feat["yz"]
    ret = feat["ret"]
    xy = feat.get("xy", np.zeros_like(yz))
    yx = feat.get("yx", np.zeros_like(yz))
    energy = np.sqrt(yz * yz + ret * ret).astype(np.float32)
    comp = np.sqrt(xy * xy + yx * yx).astype(np.float32)
    phase = np.mod(np.arctan2(ret, yz), np.pi).astype(np.float32)
    feat["energy"] = energy
    feat["spec"] = (energy - comp).astype(np.float32)
    feat["cos2"] = np.cos(2.0 * phase).astype(np.float32)
    feat["sin2"] = np.sin(2.0 * phase).astype(np.float32)
    return feat


def make_decoy(feat: Dict[str, np.ndarray], kind: str, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    d = clone_features(feat)

    if kind == "true":
        return d

    if kind == "yz_ret_swap":
        yz = d["yz"].copy()
        ret = d["ret"].copy()
        d["yz"] = ret
        d["ret"] = yz
        return recompute_derived(d)

    if kind == "reciprocal_break":
        # Preserve energy but break the signed reciprocal orientation.
        e = d["energy"].copy()
        phase = np.mod(np.arctan2(d["ret"], d["yz"]) + 0.70, np.pi)
        d["yz"] = e * np.cos(phase)
        d["ret"] = -e * np.sin(phase)
        return recompute_derived(d)

    if kind == "delay_permute":
        perm = np.arange(d["yz"].shape[1])[::-1]
        for k in ["yz", "ret", "xy", "yx", "energy", "spec", "cos2", "sin2"]:
            d[k] = d[k][:, perm]
        return d

    if kind == "phase_scramble":
        e = d["energy"].copy()
        rungs = e.shape[1]
        ph = rng.uniform(0.0, np.pi, size=e.shape).astype(np.float32)
        d["yz"] = e * np.cos(ph)
        d["ret"] = e * np.sin(ph)
        return recompute_derived(d)

    if kind == "comparison_decoy":
        # Move the same scalar energy into comparison channels and attenuate YZ/ret.
        e = d["energy"].copy()
        d["xy"] = e * 0.75
        d["yx"] = e * 0.50
        d["yz"] = d["yz"] * 0.10
        d["ret"] = d["ret"] * 0.10
        return recompute_derived(d)

    raise ValueError(kind)


def stack_feature_rows(rows: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = rows[0].keys()
    return {k: np.concatenate([r[k] for r in rows], axis=0).astype(np.float32) for k in keys}


def generate_der_environment(
    groups: int,
    n_queries: int,
    rungs: int,
    seed: int,
    query_noise: float,
    base_energy: float,
    energy_gain: float,
    comparison_scale: float,
    offset_deform: float,
) -> DEREnvironment:
    rng = np.random.default_rng(seed)
    true = make_true_manifold_arrays(
        groups,
        rungs,
        rng,
        base_energy=base_energy,
        energy_gain=energy_gain,
        comparison_scale=comparison_scale,
        offset_deform=offset_deform,
        noise=0.0,
    )

    decoy_kinds = ["true", "yz_ret_swap", "reciprocal_break", "delay_permute", "phase_scramble", "comparison_decoy"]

    key_rows: List[Dict[str, np.ndarray]] = []
    kind_labels: List[str] = []
    group_labels: List[int] = []

    for g in range(groups):
        one = {k: v[g:g + 1] for k, v in true.items()}
        for kind in decoy_kinds:
            row = make_decoy(one, kind, rng)
            key_rows.append(row)
            kind_labels.append(kind)
            group_labels.append(g)

    keys = stack_feature_rows(key_rows)

    q_groups = rng.choice(groups, size=n_queries, replace=True)
    query = {k: v[q_groups].copy() for k, v in true.items()}

    # Measurement/query noise.
    if query_noise > 0:
        query["yz"] += rng.normal(0.0, query_noise, size=query["yz"].shape).astype(np.float32)
        query["ret"] += rng.normal(0.0, query_noise, size=query["ret"].shape).astype(np.float32)
        query["xy"] += rng.normal(0.0, query_noise * 0.5, size=query["xy"].shape).astype(np.float32)
        query["yx"] += rng.normal(0.0, query_noise * 0.5, size=query["yx"].shape).astype(np.float32)
        query = recompute_derived(query)

    truth = (q_groups * len(decoy_kinds)).astype(np.int64)

    return DEREnvironment(
        query=query,
        keys=keys,
        truth=truth,
        candidate_kind=np.asarray(kind_labels, dtype=object),
        group_id=np.asarray(group_labels, dtype=np.int64),
        config={
            "groups": groups,
            "n_queries": n_queries,
            "rungs": rungs,
            "seed": seed,
            "query_noise": query_noise,
            "decoy_kinds": decoy_kinds,
        },
    )


def flatten_features(feat: Dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.concatenate([feat[n] for n in names], axis=1).astype(np.float32)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return (X / n).astype(np.float32)


def topk_from_scores_gpu(scores: Any, topk: int) -> Any:
    idx = cp.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
    part = cp.take_along_axis(scores, idx, axis=1)
    order = cp.argsort(-part, axis=1)
    return cp.take_along_axis(idx, order, axis=1).astype(cp.int32)


def topk_from_scores_np(scores: np.ndarray, topk: int) -> np.ndarray:
    idx = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int64)


def recall_at_k(topk: Any, truth: np.ndarray, k: int) -> float:
    arr = topk.get() if hasattr(topk, "get") else np.asarray(topk)
    return float(np.mean([int(truth[i] in arr[i, :k]) for i in range(arr.shape[0])]))


def mean_reciprocal_rank(topk: Any, truth: np.ndarray) -> float:
    arr = topk.get() if hasattr(topk, "get") else np.asarray(topk)
    rr = []
    for row, gt in zip(arr, truth):
        val = 0.0
        for rank, idx in enumerate(row, start=1):
            if int(idx) == int(gt):
                val = 1.0 / rank
                break
        rr.append(val)
    return float(np.mean(rr))


def rank_summary(topk: Any, truth: np.ndarray) -> Tuple[float, float]:
    """
    Mean/median 1-indexed rank inside the provided top-k list.

    If the true item is not present in the provided list, it is assigned
    len(row)+1. For local DER, the list contains the full matched candidate set,
    so this is the exact local rank.
    """
    arr = topk.get() if hasattr(topk, "get") else np.asarray(topk)
    ranks = []
    for row, gt in zip(arr, truth):
        r = len(row) + 1
        for rank, idx in enumerate(row, start=1):
            if int(idx) == int(gt):
                r = rank
                break
        ranks.append(float(r))
    return float(np.mean(ranks)), float(np.median(ranks))


def der_backend_kind(backend: str) -> int:
    if backend == "energy":
        return 0
    if backend == "pi_fit":
        return 1
    if backend == "flat_cosine":
        return 2
    if backend == "dm":
        return 3
    raise ValueError(backend)


def score_der_kernel_backend(
    env: DEREnvironment,
    backend: str,
    topk: int,
    ctx: DMCudaContext,
    args: argparse.Namespace,
    is_local: bool,
) -> Tuple[Any, float]:
    """
    Custom D_M DER megakernel path.

    This replaces generic CuPy GEMM/argpartition for DER with one fused kernel:
        query x candidate scoring + top-k

    backend_kind:
        0 energy profile distance
        1 pi-fit profile distance
        2 flat full-feature cosine
        3 D_M paired manifold distance
    """
    if ctx.k_der is None:
        raise RuntimeError("D_M DER kernel is not available in CUDA context.")

    q = env.query
    k = env.keys
    n_queries = int(q["yz"].shape[0])
    n_keys = int(k["yz"].shape[0])
    rungs = int(q["yz"].shape[1])
    local_width = int(len(env.config.get("decoy_kinds", []))) if is_local else 0

    topk = int(max(1, min(topk, 16)))
    if is_local:
        topk = min(topk, max(1, local_width))

    def cp_f32(x: np.ndarray) -> Any:
        return cp.asarray(np.ascontiguousarray(x.astype(np.float32)))

    d_q_yz = cp_f32(q["yz"])
    d_q_ret = cp_f32(q["ret"])
    d_q_energy = cp_f32(q["energy"])
    d_q_spec = cp_f32(q["spec"])
    d_q_cos2 = cp_f32(q["cos2"])
    d_q_sin2 = cp_f32(q["sin2"])

    d_k_yz = cp_f32(k["yz"])
    d_k_ret = cp_f32(k["ret"])
    d_k_energy = cp_f32(k["energy"])
    d_k_spec = cp_f32(k["spec"])
    d_k_cos2 = cp_f32(k["cos2"])
    d_k_sin2 = cp_f32(k["sin2"])

    d_truth = cp.asarray(np.ascontiguousarray(env.truth.astype(np.int64)))
    d_out_idx = cp.empty((n_queries, topk), dtype=cp.int32)
    d_out_score = cp.empty((n_queries, topk), dtype=cp.float32)

    threads = int(max(32, min(256, getattr(args, "der_kernel_threads", 256))))

    kernel_args = (
        d_q_yz,
        d_q_ret,
        d_q_energy,
        d_q_spec,
        d_q_cos2,
        d_q_sin2,
        d_k_yz,
        d_k_ret,
        d_k_energy,
        d_k_spec,
        d_k_cos2,
        d_k_sin2,
        d_truth,
        np.int32(n_queries),
        np.int32(n_keys),
        np.int32(rungs),
        np.int32(der_backend_kind(backend)),
        np.int32(1 if is_local else 0),
        np.int32(local_width),
        np.int32(topk),
        d_out_idx,
        d_out_score,
    )

    # Warmup.
    ctx.k_der((n_queries,), (threads,), kernel_args)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    ctx.k_der((n_queries,), (threads,), kernel_args)
    end.record()
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    return d_out_idx, float(elapsed_ms / 1000.0)



def score_der_backend(
    env: DEREnvironment,
    backend: str,
    topk: int,
    use_cuda: bool,
    chunk: int,
    ctx: Optional[DMCudaContext] = None,
    args: Optional[argparse.Namespace] = None,
) -> Tuple[Any, float]:
    """
    Return top-k indices and elapsed seconds.

    Important:
        DER is a matching/retrieval task, not a raw projection-amplitude task.

    The first repo-ready draft used raw dot products for the D_M backend. That
    let high-amplitude/global-similar candidates dominate the bank and produced
    near-zero recall. This fixed version scores by manifold closeness:

        energy      : negative energy-profile squared distance
        pi_fit      : negative pi-shape squared distance
        flat_cosine : cosine over flattened features
        dm          : negative D_M manifold distance

    The D_M distance is directional and paired:

        yz/ret directional mismatch
        pi-phase mismatch
        energy mismatch
        specificity mismatch

    This is the intended classical DER test: scalar-equivalent decoys should
    fool scalar energy, while D_M should retrieve the candidate preserving the
    YZ-primary / ZY-reciprocal paired structure.
    """
    if use_cuda and ctx is not None and ctx.k_der is not None and args is not None:
        return score_der_kernel_backend(env, backend, topk, ctx, args, is_local=False)

    q = env.query
    k = env.keys
    n = q["yz"].shape[0]
    m = k["yz"].shape[0]
    topk = min(topk, m)

    t0 = time.perf_counter()

    def gpu_neg_sqdist(Q_np: np.ndarray, K_np: np.ndarray) -> Any:
        Q = cp.asarray(Q_np.astype(np.float32))
        K = cp.asarray(K_np.astype(np.float32))
        q2 = cp.sum(Q * Q, axis=1, keepdims=True)
        k2 = cp.sum(K * K, axis=1, keepdims=True).T
        return -(q2 + k2 - 2.0 * (Q @ K.T))

    if use_cuda and _HAVE_CUPY:
        out = cp.empty((n, topk), dtype=cp.int32)

        if backend == "energy":
            Q = flatten_features(q, ["energy"])
            K = flatten_features(k, ["energy"])
            for i in range(0, n, chunk):
                S = gpu_neg_sqdist(Q[i:i + chunk], K)
                out[i:i + chunk] = topk_from_scores_gpu(S, topk)
            cp.cuda.Stream.null.synchronize()

        elif backend == "pi_fit":
            Q = flatten_features(q, ["cos2", "sin2"])
            K = flatten_features(k, ["cos2", "sin2"])
            for i in range(0, n, chunk):
                S = gpu_neg_sqdist(Q[i:i + chunk], K)
                out[i:i + chunk] = topk_from_scores_gpu(S, topk)
            cp.cuda.Stream.null.synchronize()

        elif backend == "flat_cosine":
            Q = cp.asarray(normalize_rows(flatten_features(q, ["yz", "ret", "energy", "spec", "cos2", "sin2"])))
            K = cp.asarray(normalize_rows(flatten_features(k, ["yz", "ret", "energy", "spec", "cos2", "sin2"])))
            for i in range(0, n, chunk):
                S = Q[i:i + chunk] @ K.T
                out[i:i + chunk] = topk_from_scores_gpu(S, topk)
            cp.cuda.Stream.null.synchronize()

        elif backend == "dm":
            Q_dir = flatten_features(q, ["yz", "ret"])
            K_dir = flatten_features(k, ["yz", "ret"])

            Q_pi = flatten_features(q, ["cos2", "sin2"])
            K_pi = flatten_features(k, ["cos2", "sin2"])

            Q_e = flatten_features(q, ["energy"])
            K_e = flatten_features(k, ["energy"])

            Q_s = flatten_features(q, ["spec"])
            K_s = flatten_features(k, ["spec"])

            # Natural-scale D_M manifold distance. The phase term is already
            # bounded by sin/cos coordinates; energy/spec are connected-scale.
            for i in range(0, n, chunk):
                S_dir = gpu_neg_sqdist(Q_dir[i:i + chunk], K_dir)
                S_pi = gpu_neg_sqdist(Q_pi[i:i + chunk], K_pi)
                S_e = gpu_neg_sqdist(Q_e[i:i + chunk], K_e)
                S_s = gpu_neg_sqdist(Q_s[i:i + chunk], K_s)

                S = (
                    0.46 * S_dir +
                    0.24 * S_pi +
                    0.18 * S_e +
                    0.12 * S_s
                )
                out[i:i + chunk] = topk_from_scores_gpu(S, topk)
            cp.cuda.Stream.null.synchronize()

        else:
            raise ValueError(backend)

        elapsed = time.perf_counter() - t0
        return out, elapsed

    # CPU fallback.
    def neg_sqdist(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        Q = Q.astype(np.float32)
        K = K.astype(np.float32)
        q2 = np.sum(Q * Q, axis=1, keepdims=True)
        k2 = np.sum(K * K, axis=1, keepdims=True).T
        return -(q2 + k2 - 2.0 * (Q @ K.T))

    if backend == "energy":
        scores = neg_sqdist(flatten_features(q, ["energy"]), flatten_features(k, ["energy"]))
    elif backend == "pi_fit":
        scores = neg_sqdist(flatten_features(q, ["cos2", "sin2"]), flatten_features(k, ["cos2", "sin2"]))
    elif backend == "flat_cosine":
        Q = normalize_rows(flatten_features(q, ["yz", "ret", "energy", "spec", "cos2", "sin2"]))
        K = normalize_rows(flatten_features(k, ["yz", "ret", "energy", "spec", "cos2", "sin2"]))
        scores = Q @ K.T
    elif backend == "dm":
        scores = (
            0.46 * neg_sqdist(flatten_features(q, ["yz", "ret"]), flatten_features(k, ["yz", "ret"])) +
            0.24 * neg_sqdist(flatten_features(q, ["cos2", "sin2"]), flatten_features(k, ["cos2", "sin2"])) +
            0.18 * neg_sqdist(flatten_features(q, ["energy"]), flatten_features(k, ["energy"])) +
            0.12 * neg_sqdist(flatten_features(q, ["spec"]), flatten_features(k, ["spec"]))
        )
    else:
        raise ValueError(backend)

    top = topk_from_scores_np(scores, topk)
    elapsed = time.perf_counter() - t0
    return top, elapsed


def mutate_der_env(env: DEREnvironment, mode: str, seed: int) -> DEREnvironment:
    rng = np.random.default_rng(seed)

    q = clone_features(env.query)
    k = clone_features(env.keys)

    if mode == "real":
        pass

    elif mode == "query_yz_ret_swap":
        yz = q["yz"].copy()
        ret = q["ret"].copy()
        q["yz"] = ret
        q["ret"] = yz
        q = recompute_derived(q)

    elif mode == "key_delay_permute":
        perm = np.arange(k["yz"].shape[1])[::-1]
        for name in ["yz", "ret", "xy", "yx", "energy", "spec", "cos2", "sin2"]:
            k[name] = k[name][:, perm]

    elif mode == "key_phase_scramble":
        e = k["energy"].copy()
        ph = rng.uniform(0.0, np.pi, size=e.shape).astype(np.float32)
        k["yz"] = e * np.cos(ph)
        k["ret"] = e * np.sin(ph)
        k = recompute_derived(k)

    elif mode in ("residual_scalar_direction_uniform", "residual_scalar_direction_uniform"):
        # Residual scalar/phase control:
        # This destroys directional diversity, but intentionally leaves enough
        # energy/spec/phase residue that it is not treated as a required full
        # collapse control.
        e = k["energy"].copy()
        k["yz"] = e * 0.50
        k["ret"] = e * 0.50
        k = recompute_derived(k)

    else:
        raise ValueError(mode)

    return DEREnvironment(
        query=q,
        keys=k,
        truth=env.truth.copy(),
        candidate_kind=env.candidate_kind.copy(),
        group_id=env.group_id.copy(),
        config={**env.config, "control_mode": mode},
    )



def score_der_local_backend(
    env: DEREnvironment,
    backend: str,
    ctx: Optional[DMCudaContext] = None,
    args: Optional[argparse.Namespace] = None,
) -> Tuple[Any, float]:
    """
    Local hard-negative DER.

    Each query is scored only against its matched candidate set:

        true
        yz_ret_swap
        reciprocal_break
        delay_permute
        phase_scramble
        comparison_decoy

    This is the clean G_M-style operator test. The candidate set preserves
    scalar-equivalent decoys, so the backend must use the correct D_M structure
    rather than global nearest-neighbor luck.
    """
    if ctx is not None and ctx.k_der is not None and args is not None and not getattr(args, "no_cuda", False):
        local_width = int(len(env.config.get("decoy_kinds", [])))
        return score_der_kernel_backend(env, backend, local_width, ctx, args, is_local=True)

    q = env.query
    k = env.keys
    truth = env.truth.astype(np.int64)
    n = int(truth.shape[0])
    decoy_kinds = list(env.config.get("decoy_kinds", []))
    local_width = len(decoy_kinds)
    if local_width <= 0:
        raise ValueError("DER environment missing decoy_kinds config.")

    local_idx = truth[:, None] + np.arange(local_width, dtype=np.int64)[None, :]

    t0 = time.perf_counter()

    def local_arrays(name: str) -> Tuple[np.ndarray, np.ndarray]:
        qv = q[name].astype(np.float32)[:, None, :]
        kv = k[name].astype(np.float32)[local_idx]
        return qv, kv

    def neg_sqdist(names: Sequence[str], weights: Optional[Sequence[float]] = None) -> np.ndarray:
        if weights is None:
            weights = [1.0] * len(names)
        score = np.zeros((n, local_width), dtype=np.float32)
        for name, w in zip(names, weights):
            qv, kv = local_arrays(name)
            diff = qv - kv
            score += np.float32(w) * (-np.sum(diff * diff, axis=2))
        return score

    if backend == "energy":
        scores = neg_sqdist(["energy"])

    elif backend == "pi_fit":
        scores = neg_sqdist(["cos2", "sin2"])

    elif backend == "flat_cosine":
        qf = flatten_features(q, ["yz", "ret", "energy", "spec", "cos2", "sin2"]).astype(np.float32)
        kf_all = flatten_features(k, ["yz", "ret", "energy", "spec", "cos2", "sin2"]).astype(np.float32)
        kf = kf_all[local_idx]
        qn = np.linalg.norm(qf, axis=1, keepdims=True)
        kn = np.linalg.norm(kf, axis=2)
        qn[qn < 1e-12] = 1.0
        kn[kn < 1e-12] = 1.0
        scores = np.einsum("nd,nkd->nk", qf / qn, kf / kn[:, :, None]).astype(np.float32)

    elif backend == "dm":
        scores = (
            0.46 * neg_sqdist(["yz", "ret"]) +
            0.24 * neg_sqdist(["cos2", "sin2"]) +
            0.18 * neg_sqdist(["energy"]) +
            0.12 * neg_sqdist(["spec"])
        )

    else:
        raise ValueError(backend)

    order = np.argsort(-scores, axis=1)
    top_global = np.take_along_axis(local_idx, order, axis=1).astype(np.int64)

    elapsed = time.perf_counter() - t0
    return top_global, elapsed


def format_der_table(title: str, rows: List[dict], local: bool = False) -> None:
    print()
    print(f"  {title}")

    if local:
        print(f"  {'backend/control':<36} {'R@1':>8} {'MRR':>8} {'mean_rank':>10} {'median_rank':>12} {'q/s':>12}")
        print("  " + "-" * 92)
    else:
        print(f"  {'backend/control':<24} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'q/s':>12}")
        print("  " + "-" * 78)

    for r in rows:
        label = str(r.get("backend", r.get("control", "")))
        if r.get("control", "real") != "real" and r.get("task", "").endswith("_control"):
            label = str(r.get("control"))

        if local:
            print(
                f"  {label:<36} "
                f"{r['R@1']:>7.2%} {r['MRR']:>8.3f} "
                f"{float(r.get('mean_rank', float('nan'))):>10.3f} "
                f"{float(r.get('median_rank', float('nan'))):>12.3f} "
                f"{r['queries_per_second']:>12,.0f}"
            )
        else:
            print(
                f"  {label:<24} "
                f"{r['R@1']:>7.2%} {r['R@5']:>7.2%} {r['R@10']:>7.2%} "
                f"{r['MRR']:>8.3f} {r['queries_per_second']:>12,.0f}"
            )



def run_der(args: argparse.Namespace, out_dir: Path, ctx: Optional[DMCudaContext] = None) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    section("D_M CLASSICAL TASK — DIMENSIONAL ENTANGLEMENT RETRIEVAL")

    groups = int(args.der_groups)
    n_queries = int(args.der_queries)
    seed = int(args.seed)

    env = generate_der_environment(
        groups=groups,
        n_queries=n_queries,
        rungs=5,
        seed=seed,
        query_noise=float(args.der_noise),
        base_energy=float(args.der_base_energy),
        energy_gain=float(args.der_energy_gain),
        comparison_scale=float(args.der_comparison_scale),
        offset_deform=float(args.der_offset_deform),
    )

    use_cuda = (not args.no_cuda) and _HAVE_CUPY
    use_der_kernel = use_cuda and ctx is not None and ctx.k_der is not None
    topk = max(DER_TOP_K)
    backends = ["energy", "pi_fit", "flat_cosine", "dm"]

    print(f"  groups       : {groups:,}")
    print(f"  candidates   : {env.keys['yz'].shape[0]:,}")
    print(f"  queries      : {n_queries:,}")
    print(f"  query noise  : {args.der_noise}")
    print(f"  CUDA         : {'yes' if use_cuda else 'no'}")
    print(f"  DER kernel   : {'yes' if use_der_kernel else 'no'}")
    print("  hard decoys  : yz_ret_swap, reciprocal_break, delay_permute, phase_scramble, comparison_decoy")
    print()
    print("  DER tasks:")
    print("    Task A local_hard_negative : true candidate vs matched scalar-equivalent decoys")
    print("    Task B global_bank_stress  : retrieve true candidate from the full candidate bank")
    print()
    print("  DER scoring:")
    print("    energy/pi_fit use profile distance")
    print("    flat_cosine uses normalized full-feature cosine")
    print("    dm uses paired directional manifold distance")

    rows: List[dict] = []

    # -------------------------------------------------------------------------
    # Task A: local matched hard-negative retrieval.
    # -------------------------------------------------------------------------
    local_rows: List[dict] = []
    for backend in backends:
        top, elapsed = score_der_local_backend(env, backend, ctx, args)
        mean_rank, median_rank = rank_summary(top, env.truth)
        row = {
            "task": "DER_local_hard_negative",
            "backend": backend,
            "control": "real",
            "groups": groups,
            "candidates": int(len(env.config["decoy_kinds"])),
            "queries": n_queries,
            "seconds": float(elapsed),
            "queries_per_second": float(n_queries / elapsed) if elapsed > 0 else None,
            "R@1": recall_at_k(top, env.truth, 1),
            "R@5": recall_at_k(top, env.truth, min(5, top.shape[1])),
            "R@10": recall_at_k(top, env.truth, min(10, top.shape[1])),
            "MRR": mean_reciprocal_rank(top, env.truth),
            "mean_rank": mean_rank,
            "median_rank": median_rank,
        }
        local_rows.append(row)
        rows.append(row)

    format_der_table("TASK A - LOCAL HARD-NEGATIVE DER", local_rows, local=True)

    # -------------------------------------------------------------------------
    # Task B: global bank retrieval stress test.
    # -------------------------------------------------------------------------
    global_rows: List[dict] = []
    for backend in backends:
        top, elapsed = score_der_backend(env, backend, topk, use_cuda, int(args.der_chunk), ctx, args)
        row = {
            "task": "DER_global_bank_stress",
            "backend": backend,
            "control": "real",
            "groups": groups,
            "candidates": int(env.keys["yz"].shape[0]),
            "queries": n_queries,
            "seconds": float(elapsed),
            "queries_per_second": float(n_queries / elapsed) if elapsed > 0 else None,
            "R@1": recall_at_k(top, env.truth, 1),
            "R@5": recall_at_k(top, env.truth, 5),
            "R@10": recall_at_k(top, env.truth, 10),
            "MRR": mean_reciprocal_rank(top, env.truth),
        }
        global_rows.append(row)
        rows.append(row)

    format_der_table("TASK B — GLOBAL BANK DER STRESS", global_rows)

    # -------------------------------------------------------------------------
    # Probe controls: local task only by default.
    # This is the clean load-bearing control read.
    # -------------------------------------------------------------------------
    control_rows: List[dict] = []

    if args.probe:
        section("D_M DER CONTROLS — IS D_M STRUCTURE LOAD-BEARING?")
        controls = ["real", "query_yz_ret_swap", "key_delay_permute", "key_phase_scramble", "residual_scalar_direction_uniform"]

        for mode in controls:
            cenv = mutate_der_env(env, mode, seed + 1009)
            top, elapsed = score_der_local_backend(cenv, "dm", ctx, args)
            mean_rank, median_rank = rank_summary(top, cenv.truth)
            row = {
                "task": "DER_local_control",
                "backend": "dm",
                "control": mode,
                "groups": groups,
                "candidates": int(len(cenv.config["decoy_kinds"])),
                "queries": n_queries,
                "seconds": float(elapsed),
                "queries_per_second": float(n_queries / elapsed) if elapsed > 0 else None,
                "R@1": recall_at_k(top, cenv.truth, 1),
                "R@5": recall_at_k(top, cenv.truth, min(5, top.shape[1])),
                "R@10": recall_at_k(top, cenv.truth, min(10, top.shape[1])),
                "MRR": mean_reciprocal_rank(top, cenv.truth),
                "mean_rank": mean_rank,
                "median_rank": median_rank,
            }
            control_rows.append(row)

        print()
        print("  note: residual_scalar_direction_uniform is a residual scalar/phase control,")
        print("        not a required full-collapse control.")
        format_der_table("LOCAL D_M CONTROL / RESIDUAL TESTS", control_rows, local=True)

    artifacts = {
        "der_truth": env.truth.astype(np.int64),
        "der_candidate_kind": np.asarray([str(x) for x in env.candidate_kind]),
        "der_group_id": env.group_id.astype(np.int64),
    }
    for name in ["yz", "ret", "energy", "spec", "cos2", "sin2"]:
        artifacts[f"der_query_{name}"] = env.query[name].astype(np.float32)
        artifacts[f"der_keys_{name}"] = env.keys[name].astype(np.float32)

    return rows, control_rows, artifacts


# =============================================================================
# PLOTS
# =============================================================================

def plot_verify_metric(summary_rows: List[dict], metric: str, out_path: Path) -> None:
    if not _HAVE_MPL:
        return
    order = {"null": 0, "base_only": 1, "offset_on": 2}
    subs = sorted(set(r["substrate"] for r in summary_rows))
    conds = sorted(set(r["condition"] for r in summary_rows), key=lambda c: order.get(c, 99))
    x = np.arange(len(conds), dtype=np.float64)
    width = 0.8 / max(1, len(subs))

    fig = plt.figure(figsize=(9, 5), dpi=150)
    ax = fig.add_subplot(111)

    for si, sub in enumerate(subs):
        vals = []
        for c in conds:
            found = [r for r in summary_rows if r["substrate"] == sub and r["condition"] == c]
            vals.append(float(found[0].get(metric, 0.0)) if found else 0.0)
        ax.bar(x + (si - (len(subs) - 1) / 2) * width, vals, width=width, label=sub)

    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel(metric)
    ax.set_title(f"D_M {metric} by condition/substrate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_der(rows: List[dict], out_path: Path) -> None:
    if not _HAVE_MPL or not rows:
        return
    rows = [r for r in rows if r.get("task") == "DER_local_hard_negative"] or rows
    labels = [r["backend"] for r in rows]
    vals = [float(r["R@1"]) for r in rows]
    fig = plt.figure(figsize=(8, 5), dpi=150)
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(vals)), vals)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Recall@1")
    ax.set_title("D_M DER — backend comparison")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =============================================================================
# MAIN VERIFY RUN
# =============================================================================

def run_verify(args: argparse.Namespace, ctx: Optional[DMCudaContext], out_dir: Path) -> Tuple[List[DMProjection], Dict[str, Any], Dict[str, np.ndarray]]:
    section("D_M VERIFY — QPROJ / GPROJ / GEO CONDITION PROJECTION")

    projs: List[DMProjection] = []
    base_meta: Dict[str, Any] = {}

    qproj_args = {
        "null": args.qproj_null,
        "base_only": args.qproj_base,
        "offset_on": args.qproj_offset,
    }
    gproj_args = {
        "null": args.gproj_null,
        "base_only": args.gproj_base,
        "offset_on": args.gproj_offset,
    }

    if not args.skip_qproj:
        for cond, value in qproj_args.items():
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            print(f"[QPROJ] {cond}: {path}")
            p = load_and_project(path, "qproj", cond, ctx, args)
            projs.append(p)
            base_meta[f"qproj_{cond}"] = {"path": str(path), "backend": p.backend, "job_id": p.job_id}

    if not args.skip_gproj:
        for cond, value in gproj_args.items():
            if not value and args.auto_gproj:
                value = str(auto_gproj_path(cond) or "")
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            print(f"[GPROJ] {cond}: {path}")
            p = load_and_project(path, "gproj", cond, ctx, args)
            projs.append(p)
            base_meta[f"gproj_{cond}"] = {"path": str(path), "backend": p.backend, "job_id": p.job_id}

    if not args.skip_geo:
        for cond in ["null", "base_only", "offset_on"]:
            print(f"[GEO] {cond}: analytic manifold")
            p = geo_condition_projection(cond, args)
            projs.append(p)
            base_meta[f"geo_{cond}"] = {"path": "analytic_geo", "backend": p.backend, "job_id": p.job_id}

    geo_sweep = {"enabled": False, "reason": "no_cuda_or_disabled"}
    if ctx is not None and not args.skip_geo and not args.skip_geo_sweep:
        try:
            print(f"[GEO SWEEP] candidates={args.geo_sweep_candidates} reps={args.geo_sweep_reps}")
            geo_sweep = run_geo_sweep_cuda(ctx, args)
            if geo_sweep.get("enabled") and geo_sweep.get("analytic_points_per_second"):
                for p in projs:
                    if p.substrate == "geo":
                        p.timing["mode"] = "geo_sweep_cuda"
                        p.timing["records_per_sec_projection"] = geo_sweep["analytic_points_per_second"]
                        p.timing["records_semantics"] = "batched_analytic_points_per_second"
                        p.timing["geo_sweep_candidates"] = geo_sweep["candidates"]
                        p.timing["geo_sweep_per_rep_ms"] = geo_sweep["per_rep_ms"]
        except Exception as e:
            geo_sweep = {"enabled": False, "error": f"{type(e).__name__}: {e}"}
            print(f"[GEO SWEEP][warn] {geo_sweep['error']}")

    summary_rows = [projection_summary_row(p) for p in projs]
    rung_all: List[Dict[str, Any]] = []
    for p in projs:
        rung_all.extend(rung_rows(p))

    control_rows = [r for r in (control_row(p) for p in projs) if r is not None]
    sep_rows = condition_separation(projs)
    agree_rows = substrate_agreement(projs)

    print()
    print("=" * 112)
    print("  D_M VERIFY SUMMARY")
    print("=" * 112)
    print(
        f"  {'substrate':<8} | {'condition':<10} | {'projection':>10} | "
        f"{'energy':>10} | {'spec':>10} | {'pi fit':>8} | {'piW':>8} | {'YZ+':>5} | {'rate/s':>14}"
    )
    print("  " + "-" * 110)
    print("  note: piW = yzzy_energy_mean * pi_periodic_score")
    print("  " + "-" * 110)

    for row in sorted(summary_rows, key=lambda r: (r["substrate"], r["condition"])):
        rps = row.get("records_per_sec_projection", "")
        rps_s = f"{float(rps):,.0f}" if isinstance(rps, (int, float)) and rps else ""
        print(
            f"  {row['substrate']:<8} | {row['condition']:<10} | "
            f"{float(row['projection_score']):>10.6f} | "
            f"{float(row['yzzy_energy_mean']):>10.6f} | "
            f"{float(row['specificity_mean']):>+10.6f} | "
            f"{float(row['pi_periodic_score']):>8.4f} | "
            f"{float(row['pi_witness_strength']):>8.5f} | "
            f"{float(row['yz_pos_frac']):>5.2f} | {rps_s:>14}"
        )

    if geo_sweep.get("enabled"):
        print()
        print("  GEO SWEEP")
        print("  " + "-" * 110)
        aps = geo_sweep.get("analytic_points_per_second")
        cps = geo_sweep.get("candidate_conditions_per_second")
        print(
            f"  candidates={geo_sweep.get('candidates'):,} "
            f"conditions={geo_sweep.get('conditions')} "
            f"points/rep={geo_sweep.get('points_per_rep'):,} "
            f"per_rep={geo_sweep.get('per_rep_ms'):.6f} ms"
        )
        print(f"  analytic points/sec={aps:,.0f} candidate-conditions/sec={cps:,.0f}")

    if control_rows:
        print()
        print("  CONTROL COLLAPSE")
        print("  " + "-" * 110)
        for r in control_rows:
            print(
                f"  {r['substrate']:<8} {r['condition']:<10} "
                f"projection {r['projection_real']:.6f} -> {r['projection_control']:.6f} "
                f"drop={r['projection_drop_fraction']:.2%}"
            )

    summary_fields = [
        "substrate", "condition", "condition_label", "source", "backend", "job_id",
        "tiles", "shots", "n_rungs", "timing_mode", "projection_per_rep_ms",
        "records_per_sec_projection", "records_semantics", "pi_periodic_mode_label",
    ] + SUMMARY_METRICS + ["pi_witness_strength"]

    rung_fields = ["substrate", "condition", "rung", "source"] + RUNG_METRICS + ["pi_phase_deg"]
    sep_fields = [
        "substrate", "condition_a", "condition_b",
        "delta_projection", "delta_energy", "delta_specificity", "delta_pi_score", "delta_piW",
        "summary_l2", "summary_corr",
    ]
    agree_fields = [
        "condition", "substrate_a", "substrate_b",
        "summary_corr", "summary_l2",
        "projection_a", "projection_b", "projection_delta_abs",
        "rung_energy_corr", "rung_specificity_corr", "rung_yz_corr", "rung_pi_cos2_corr",
    ]
    control_fields = [
        "substrate", "condition", "control",
        "projection_real", "projection_control", "projection_delta", "projection_drop_fraction",
        "energy_real", "energy_control", "piW_real", "piW_control",
        "specificity_real", "specificity_control",
    ]

    write_csv(out_dir / "verify_projection_summary.csv", summary_rows, summary_fields)
    write_csv(out_dir / "verify_rung_projection.csv", rung_all, rung_fields)
    write_csv(out_dir / "verify_condition_separation.csv", sep_rows, sep_fields)
    write_csv(out_dir / "verify_substrate_agreement.csv", agree_rows, agree_fields)
    write_csv(out_dir / "verify_control_collapse.csv", control_rows, control_fields)

    artifacts: Dict[str, np.ndarray] = {}
    for p in projs:
        key = f"{p.substrate}_{p.condition}"
        artifacts[f"{key}_summary"] = p.summary.astype(np.float32)
        artifacts[f"{key}_rung_stats"] = p.rung_stats.astype(np.float32)
        if p.tile_stats is not None:
            artifacts[f"{key}_tile_stats"] = p.tile_stats.astype(np.float32)
        if p.control_summary is not None:
            artifacts[f"{key}_control_summary"] = p.control_summary.astype(np.float32)

    plot_verify_metric(summary_rows, "projection_score", out_dir / "verify_projection_score_by_condition.png")
    plot_verify_metric(summary_rows, "pi_witness_strength", out_dir / "verify_piW_by_condition.png")

    meta = {
        "base_meta": base_meta,
        "geo_sweep": geo_sweep,
        "projection_summary": summary_rows,
        "condition_separation": sep_rows,
        "substrate_agreement": agree_rows,
        "control_collapse": control_rows,
    }
    return projs, meta, artifacts


# =============================================================================
# AUTO-DISCOVERY
# =============================================================================

def newest_glob(pattern: str) -> Optional[Path]:
    paths = list(DATA_DIR.glob(pattern))
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def auto_gproj_path(condition: str) -> Optional[Path]:
    if condition == "null":
        return newest_glob("dm_gpu_data_null_*.npz")
    if condition == "base_only":
        return newest_glob("dm_gpu_data_base_delay_*.npz")
    if condition == "offset_on":
        return newest_glob("dm_gpu_data_offset_deformed_*.npz")
    return None


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="D_M final benchmark: substrate verification + dimensional entanglement retrieval."
    )

    # Verify bases.
    p.add_argument("--qproj-null", default=str(DEFAULT_QPROJ_NULL))
    p.add_argument("--qproj-base", default=str(DEFAULT_QPROJ_BASE))
    p.add_argument("--qproj-offset", default=str(DEFAULT_QPROJ_OFFSET))

    p.add_argument("--gproj-null", default=str(DEFAULT_GPROJ_NULL))
    p.add_argument("--gproj-base", default=str(DEFAULT_GPROJ_BASE))
    p.add_argument("--gproj-offset", default=str(DEFAULT_GPROJ_OFFSET))

    p.add_argument("--auto-gproj", action="store_true")
    p.add_argument("--repair-metadata", action="store_true")

    p.add_argument("--skip-qproj", action="store_true")
    p.add_argument("--skip-gproj", action="store_true")
    p.add_argument("--skip-geo", action="store_true")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--skip-classical", action="store_true")
    p.add_argument("--classical", action="store_true", help="Run DER classical retrieval. Default runs unless --skip-classical.")
    p.add_argument("--probe", action="store_true", default=True, help="Run DER destructive controls.")

    # Timing / CUDA.
    p.add_argument("--reps", type=int, default=200, help="CUDA timing reps for qproj/gproj record projection.")
    p.add_argument("--cuda-threads", type=int, default=256)
    p.add_argument("--control-seed", type=int, default=12345)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--quiet", action="store_true")

    # GEO.
    p.add_argument("--geo-sweep-candidates", type=int, default=262144)
    p.add_argument("--geo-sweep-reps", type=int, default=200)
    p.add_argument("--skip-geo-sweep", action="store_true")
    p.add_argument("--geo-base-energy", type=float, default=0.030)
    p.add_argument("--geo-energy-gain", type=float, default=0.285)
    p.add_argument("--geo-comparison-scale", type=float, default=0.010)
    p.add_argument("--geo-offset-deform", type=float, default=0.18)
    p.add_argument("--geo-phase-scale", type=float, default=0.37)

    # Classical DER.
    p.add_argument("--der-groups", type=int, default=50000, help="Number of true groups; candidates = groups * 6.")
    p.add_argument("--der-queries", type=int, default=4096)
    p.add_argument("--der-noise", type=float, default=0.010)
    p.add_argument("--der-chunk", type=int, default=128)
    p.add_argument("--der-kernel-threads", type=int, default=256, help="Threads per query block for fused DER megakernel.")
    p.add_argument("--der-base-energy", type=float, default=0.040)
    p.add_argument("--der-energy-gain", type=float, default=0.290)
    p.add_argument("--der-comparison-scale", type=float, default=0.012)
    p.add_argument("--der-offset-deform", type=float, default=0.20)

    # Shared.
    p.add_argument("--seed", type=int, default=20260603)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-plots", action="store_true")

    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    t_all = time.time()

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"dm_probe_13_benchmark_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = compile_dm_cuda(args)

    section("D_M FINAL BENCHMARK")
    print(f"  Data dir     : {DATA_DIR}")
    print(f"  Analysis dir : {out_dir}")
    print(f"  CUDA kernel  : {'yes' if ctx is not None else 'no'}")
    if ctx is not None:
        print(f"  Kernel path  : {ctx.path}")
    print(f"  Verify       : {'no' if args.skip_verify else 'yes'}")
    print(f"  Classical DER: {'no' if args.skip_classical else 'yes'}")
    print(f"  Probe        : {'yes' if args.probe else 'no'}")

    verify_meta: Dict[str, Any] = {}
    artifacts: Dict[str, np.ndarray] = {}

    if not args.skip_verify:
        _projs, verify_meta, verify_artifacts = run_verify(args, ctx, out_dir)
        artifacts.update(verify_artifacts)

    der_rows: List[dict] = []
    der_control_rows: List[dict] = []
    der_meta: Dict[str, Any] = {}

    if not args.skip_classical:
        der_rows, der_control_rows, der_artifacts = run_der(args, out_dir, ctx)
        artifacts.update(der_artifacts)

        der_fields = [
            "task", "backend", "control", "groups", "candidates", "queries",
            "seconds", "queries_per_second", "R@1", "R@5", "R@10", "MRR", "mean_rank", "median_rank",
        ]
        write_csv(out_dir / "der_summary.csv", der_rows, der_fields)
        write_csv(out_dir / "der_controls.csv", der_control_rows, der_fields)
        plot_der(der_rows, out_dir / "der_backend_recall.png")

        der_meta = {
            "summary": der_rows,
            "controls": der_control_rows,
        }

    if artifacts:
        np.savez_compressed(out_dir / "artifacts.npz", **artifacts)

    result = {
        "schema": "ghost_oracle.dm.final_benchmark.v1",
        "created": now_tag(),
        "seconds": time.time() - t_all,
        "config": {
            "cuda_enabled": ctx is not None,
            "kernel_path": str(ctx.path) if ctx is not None else None,
            "seed": args.seed,
            "verify": not args.skip_verify,
            "classical": not args.skip_classical,
            "probe": args.probe,
            "analysis_dir": str(out_dir),
            "der_groups": args.der_groups,
            "der_queries": args.der_queries,
            "der_noise": args.der_noise,
            "der_kernel_threads": args.der_kernel_threads,
        },
        "verify": verify_meta,
        "der": der_meta,
        "bounded_claim": (
            "D_M projects a YZ-primary / ZY-reciprocal dimensional manifold and "
            "applies that operator to a classical retrieval task where directional "
            "paired structure, not scalar energy alone, is load-bearing."
        ),
        "non_claims": [
            "D_M does not reconstruct density matrices.",
            "D_M does not certify device-independent Bell nonlocality.",
            "D_M does not prove prepared Bell states.",
            "D_M is not a QPU speedup or quantum advantage claim.",
            "GPROJ is not an IBM hardware simulator.",
            "The DER task is a controlled classical stress test, not a real-world deployment claim.",
        ],
    }
    write_json(out_dir / "result.json", result)

    section("DONE")
    print(f"  result              : {out_dir / 'result.json'}")
    if not args.skip_verify:
        print(f"  verify summary      : {out_dir / 'verify_projection_summary.csv'}")
        print(f"  verify controls     : {out_dir / 'verify_control_collapse.csv'}")
    if not args.skip_classical:
        print(f"  DER summary         : {out_dir / 'der_summary.csv'}")
        print(f"  DER controls        : {out_dir / 'der_controls.csv'}")
    print(f"  artifacts           : {out_dir / 'artifacts.npz'}")
    print()
    print("Done. Break it, fix it, document what happened.")


if __name__ == "__main__":
    main()
