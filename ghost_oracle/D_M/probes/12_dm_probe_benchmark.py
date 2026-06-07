#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
D_M BENCHMARK — FINAL DIMENSIONAL ENTANGLEMENT PROJECTION OPERATOR
==============================================================================

Purpose
-------
Canonical benchmark runner for the D_M operator.

D_M is the Dimensional Entanglement Projection channel: a Bell-witness manifold
operator over basis, chip geometry, delay, offset deformation, and π-phase.

Current discovered D_M witness orientation:

    YZ = primary witness dimension
    ZY = reciprocal / inverted witness dimension
    XY / YX = comparison dimensions

The projected rung coordinates are:

    Y  = connected(YZ)
    Z  = connected(ZY)
    R  = -Z
    E  = sqrt(Y^2 + R^2)
    S  = E - sqrt(XY^2 + YX^2)
    φ  = atan2(R, Y) mod π

where connected(PQ) means:

    connected = <P0 P1> - <P0><P1>

Substrate paths
---------------
For each condition/base, this benchmark compares:

    qproj:
        real QPU D_M listener records from IBM Runtime.

    gproj:
        GPU-generated controlled D_M listener records from d_m_gpu_generate.py.

    geo:
        analytic classical D_M manifold path. No shots required. It computes
        the same D_M rung/summary projection directly from condition metadata.

Current benchmark conditions
----------------------------
The first D_M benchmark uses three discovered conditions:

    null:
        base_delays_dt = [0,0,0,0,0]
        offset_dt      = 0
        expected       = weak/no structured D_M manifold

    base_only:
        base_delays_dt = [0,256,1024,4096,16384]
        offset_dt      = 0
        expected       = base-delay π-phase dimensional entanglement manifold

    offset_on:
        base_delays_dt = [0,256,1024,4096,16384]
        offset_dt      = 128
        expected       = offset-deformed D_M manifold

What this benchmark tests
-------------------------
Task A:
    condition separation:
        null vs base_only vs offset_on

Task B:
    active-manifold projection:
        null -> active witness manifold separation

Task C:
    substrate agreement:
        qproj/gproj/geo preserve the same qualitative condition ordering and
        D_M projection vector structure

Task D:
    qproj/gproj control collapse where record bases exist:
        independent_bit_shuffle breaks q0/q1 same-shot pairing and should
        reduce active D_M projection.

Controls
--------
The CUDA qproj/gproj path can run:

    independent_bit_shuffle

which preserves q0/q1 marginal distributions but breaks the same-shot pair
needed for the connected D_M witness.

The GEO path has no shot-pair control because it is the analytic reference path.
Later GEO controls can directly scramble witness labels or delay order.

Non-claims
----------
This benchmark does NOT claim:

    D_M reconstructs density matrices.
    D_M certifies full Bell nonlocality.
    D_M proves prepared Bell states.
    D_M is a QPU speedup or quantum advantage claim.
    GPROJ is an IBM hardware simulator.

The bounded claim is:

    D_M projects a YZ-primary / ZY-reciprocal dimensional entanglement manifold
    from qproj/gproj/geo substrates and separates null, base-delay, and
    offset-deformed witness conditions under shared projection metrics.

Usage
-----
Explicit qproj/gproj bases:

    python ghost_oracle/D_M/d_m_benchmark.py ^
      --qproj-null   ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_<NULL_JOB>.npz ^
      --qproj-base   ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_<BASE_JOB>.npz ^
      --qproj-offset ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_<OFFSET_JOB>.npz ^
      --gproj-null   ghost_oracle/D_M/data/dm_gpu_data_null_4096shots_seed<SEED>.npz ^
      --gproj-base   ghost_oracle/D_M/data/dm_gpu_data_base_delay_4096shots_seed<SEED>.npz ^
      --gproj-offset ghost_oracle/D_M/data/dm_gpu_data_offset_deformed_4096shots_seed<SEED>.npz

Run only analytic GEO:

    python ghost_oracle/D_M/d_m_benchmark.py --skip-qproj --skip-gproj

Disable CUDA:

    python ghost_oracle/D_M/d_m_benchmark.py --no-cuda

Outputs
-------
    ghost_oracle/D_M/analysis/d_m_probe_12_benchmark<timestamp>/
        result.json
        projection_summary.csv
        rung_projection.csv
        condition_separation.csv
        substrate_agreement.csv
        control_collapse.csv
        artifacts.npz
        projection_score_by_condition.png
        energy_by_condition.png
        pi_score_by_condition.png

==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    plt = None
    _HAVE_MPL = False

try:
    import cupy as cp
    _HAVE_CUPY = True
    _CUPY_IMPORT_ERROR = None
except Exception as e:
    cp = None
    _HAVE_CUPY = False
    _CUPY_IMPORT_ERROR = repr(e)


# =============================================================================
# PATHS / DEFAULTS
# =============================================================================

HERE = Path(__file__).resolve().parent
D_M_DIR = HERE.parent
DATA_DIR = D_M_DIR / "data"
ANALYSIS_DIR = HERE / "analyze"
KERNEL_PATH = D_M_DIR / "kernels" / "dm_projector_kernel.cu"

DEFAULT_BASE_DELAYS = [0, 256, 1024, 4096, 16384]
DEFAULT_NULL_DELAYS = [0, 0, 0, 0, 0]
DEFAULT_OFFSET_DT = 128
DEFAULT_SHOTS = 4096
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


# =============================================================================
# UTILITIES
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


# =============================================================================
# DATA OBJECTS
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


# =============================================================================
# D_M BASE LOADING
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


def repair_metadata(num_tiles: int, condition_hint: Optional[str]) -> Dict[str, np.ndarray]:
    if condition_hint is None:
        raise KeyError("Cannot repair metadata without --*-condition hint.")
    rung, wi, base, off, total = build_metadata_from_condition(condition_hint)
    if len(rung) != num_tiles:
        raise ValueError(f"repair metadata length {len(rung)} != tiles {num_tiles}")
    return {
        "tile_rung_index": rung,
        "tile_witness_index": wi,
        "tile_base_delay_dt": base,
        "tile_offset_dt": off,
        "tile_total_delay_dt": total,
    }


def load_dm_base(path: Path, substrate: str, condition_hint: Optional[str] = None, repair: bool = False) -> Dict[str, Any]:
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
            raise KeyError(f"{path} is missing usable D_M metadata. Use --repair-metadata with condition hints.")
        fixed = repair_metadata(tiles, condition_hint)
        tile_rung = fixed["tile_rung_index"]
        tile_wi = fixed["tile_witness_index"]
        tile_base = fixed["tile_base_delay_dt"]
        tile_off = fixed["tile_offset_dt"]
        tile_total = fixed["tile_total_delay_dt"]
        repaired = True

    if tile_rung is None or np.any(tile_rung < 0):
        tile_rung = (np.arange(tiles) // 4).astype(np.int32)

    condition, label = infer_condition_from_delays(tile_base, tile_off, tile_total)
    if condition_hint is not None:
        # User-provided role wins for reporting, but delay metadata remains the source of truth.
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
# CPU PROJECTION
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

        denom = max(1, len(rows))
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

    # Same optimized qproj score as dm_projector_kernel.cu.
    # Non-normalized physical projection score.
    #
    # This intentionally removes empirical calibration divisors such as /0.35
    # and /0.25. The score lives on the natural connected-correlator scale:
    #
    #     connected correlator C in [-1, 1]
    #     YZ/ZY energy sqrt(YZ^2 + (-ZY)^2)
    #     specificity = YZ/ZY energy - comparison energy
    #
    # The π score remains a shape/fit diagnostic, but the load-bearing π term is
    # physical π witness strength:
    #
    #     pi_witness_strength = energy_mean * pi_score
    #
    # That keeps π influence naturally bounded by the measured YZ/ZY witness
    # energy instead of letting an idealized phase fit dominate low-energy residue.
    energy_term = max(0.0, energy_mean)
    spec_term = max(0.0, spec_mean)
    yz_term = max(0.0, yz_mean)
    pi_witness_strength = max(0.0, energy_mean) * pi_score
    pi_term = pi_witness_strength
    tracking_term = 0.5 * (abs(energy_r) + abs(spec_r))

    projection = (
        0.35 * energy_term +
        0.25 * spec_term +
        0.15 * yz_term +
        0.15 * pi_term +
        0.10 * tracking_term
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
# CUDA PROJECTION
# =============================================================================

@dataclass
class DMCudaContext:
    module: Any
    k_tile: Any
    k_shuffle: Any
    k_rung: Any
    k_summary: Any
    k_geo: Any
    k_geo_sweep: Any
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
        path=KERNEL_PATH,
    )


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

    # Warmup.
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
    control_per_rep_ms = control_ms / reps

    records = int(base["tiles"] * base["shots"] * 2)
    timing = {
        "mode": "cuda",
        "reps": reps,
        "projection_total_ms": float(elapsed_ms),
        "projection_per_rep_ms": float(per_rep_ms),
        "control_total_ms": float(control_ms),
        "control_per_rep_ms": float(control_per_rep_ms),
        "records_per_rep": records,
        "records_per_sec_projection": float(records / (per_rep_ms / 1000.0)) if per_rep_ms > 0 else None,
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
    }
    return tile, rung, summary, None, timing


# =============================================================================
# GEO ANALYTIC MANIFOLD
# =============================================================================

def geo_x_values(base_delays: Sequence[int], offset_dt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = []
    off = []
    for r, bd in enumerate(base_delays):
        # Average tile offset over witness quartet.
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
            # Weak/no manifold. Keep tiny deterministic residual so curves exist.
            yz = 0.010 * math.sin(1.7 * r + 0.2)
            zy = 0.010 * math.cos(1.3 * r + 0.5)
            xy = 0.010 * math.cos(0.9 * r)
            yx = 0.010 * math.sin(0.8 * r + 0.4)
        else:
            x = float(x_log[r])
            xl = float(x_lin[r])
            # Analytic D_M manifold. Base delay creates the π-phase curve;
            # offset deforms phase/amplitude without changing the operator axis.
            amp = args.geo_base_energy + args.geo_energy_gain * (0.20 + 0.80 * (x ** 1.10))
            phase = math.pi * (0.05 + 0.37 * x)

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



def condition_kind(condition: str) -> int:
    if condition == "null":
        return 0
    if condition == "base_only":
        return 1
    if condition == "offset_on":
        return 2
    raise ValueError(condition)


def geo_metadata_for_condition(condition: str) -> Tuple[np.ndarray, int, str]:
    if condition == "null":
        return np.asarray(DEFAULT_NULL_DELAYS, dtype=np.int32), 0, "no_delay_no_offset"
    if condition == "base_only":
        return np.asarray(DEFAULT_BASE_DELAYS, dtype=np.int32), 0, "base_delay_only"
    if condition == "offset_on":
        return np.asarray(DEFAULT_BASE_DELAYS, dtype=np.int32), DEFAULT_OFFSET_DT, "base_delay_plus_offset"
    raise ValueError(condition)


def timed_geo_condition_projection_cuda(condition: str, ctx: DMCudaContext, args: argparse.Namespace) -> DMProjection:
    """
    CUDA analytic GEO path.

    Uses dm_geo_rung_projection_kernel_f32 to compute rung_stats directly on GPU,
    then dm_projection_summary_kernel_f32 for the same projection vector as
    qproj/gproj. GEO rate is analytic D_M points/sec, not shot records/sec.
    """
    reps = max(1, int(getattr(args, "geo_reps", 1)))
    base_delays, offset_dt, label = geo_metadata_for_condition(condition)
    n_rungs = int(base_delays.shape[0])
    points_per_rep = int(n_rungs * 4)

    d_base = cp.asarray(base_delays.astype(np.int32))
    d_rung = cp.zeros((n_rungs, N_RUNG_METRICS), dtype=cp.float32)
    d_summary = cp.zeros((N_SUMMARY_METRICS,), dtype=cp.float32)

    ck = np.int32(condition_kind(condition))
    nr = np.int32(n_rungs)
    odt = np.int32(offset_dt)

    args_tuple = (
        ck,
        d_base,
        nr,
        odt,
        np.float32(args.geo_base_energy),
        np.float32(args.geo_energy_gain),
        np.float32(args.geo_comparison_scale),
        np.float32(args.geo_offset_deform),
        d_rung,
    )

    # Warmup.
    ctx.k_geo((n_rungs,), (1,), args_tuple)
    ctx.k_summary((1,), (1,), (d_rung, nr, d_summary))
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(reps):
        ctx.k_geo((n_rungs,), (1,), args_tuple)
        ctx.k_summary((1,), (1,), (d_rung, nr, d_summary))
    end.record()
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    per_rep_ms = elapsed_ms / float(reps)
    points_per_sec = float(points_per_rep / (per_rep_ms / 1000.0)) if per_rep_ms > 0 else None

    rung = cp.asnumpy(d_rung)
    summary = cp.asnumpy(d_summary)

    return DMProjection(
        substrate="geo",
        condition=condition,
        condition_label=label,
        source="analytic_geo_cuda",
        backend="cuda_analytic",
        job_id=f"geo_cuda_{condition}",
        tiles=n_rungs * 4,
        shots=0,
        n_rungs=n_rungs,
        tile_stats=None,
        rung_stats=rung,
        summary=summary,
        control_summary=None,
        timing={
            "mode": "geo_cuda",
            "reps": reps,
            "projection_total_ms": float(elapsed_ms),
            "projection_per_rep_ms": float(per_rep_ms),
            "records_per_rep": points_per_rep,
            "records_per_sec_projection": points_per_sec,
            "records_semantics": "analytic_points_per_second",
        },
        notes="CUDA analytic D_M manifold. No shot records.",
    )



def timed_geo_condition_projection(condition: str, args: argparse.Namespace, ctx: Optional[DMCudaContext] = None) -> DMProjection:
    """
    Time the analytic GEO path.

    GEO has no shots, so records/sec is reported as analytic D_M points/sec:
        points_per_rep = n_rungs * 4 witness dimensions

    The returned projection object is computed once for output, while a tight
    loop repeats the same analytic projection to estimate per-rep runtime.
    """
    if ctx is not None:
        try:
            return timed_geo_condition_projection_cuda(condition, ctx, args)
        except Exception as e:
            if not getattr(args, "quiet", False):
                print(f"[GEO][warn] CUDA analytic path failed for {condition}: {e}; falling back to Python GEO.")

    reps = max(1, int(getattr(args, "geo_reps", 1)))

    out = geo_condition_projection(condition, args)

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = geo_condition_projection(condition, args)
    elapsed = time.perf_counter() - t0

    per_rep_ms = (elapsed * 1000.0) / float(reps)
    points_per_rep = int(out.n_rungs * 4)
    points_per_sec = float(points_per_rep / (per_rep_ms / 1000.0)) if per_rep_ms > 0 else None

    out.timing = {
        "mode": "geo",
        "reps": reps,
        "projection_total_ms": float(elapsed * 1000.0),
        "projection_per_rep_ms": float(per_rep_ms),
        "records_per_rep": points_per_rep,
        "records_per_sec_projection": points_per_sec,
        "records_semantics": "analytic_points_per_second",
    }
    return out



def run_geo_sweep_cuda(ctx: DMCudaContext, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Batched analytic GEO sweep speed path.

    This is the true minimized geometric workload: many candidate manifolds are
    evaluated in parallel, each across null/base_only/offset_on conditions.
    """
    if getattr(args, "skip_geo_sweep", False):
        return {"enabled": False, "reason": "disabled"}

    n_candidates = max(1, int(args.geo_sweep_candidates))
    reps = max(1, int(args.geo_sweep_reps))
    n_rungs = len(DEFAULT_BASE_DELAYS)
    n_conditions = 3
    points_per_rep = n_candidates * n_conditions * n_rungs * 4

    # Deterministic parameter cloud around canonical GEO values.
    idx = cp.arange(n_candidates, dtype=cp.float32)
    denom = cp.float32(max(1, n_candidates - 1))
    u = idx / denom

    base_energy = cp.asarray(args.geo_base_energy + 0.010 * cp.sin(17.0 * u + 0.10), dtype=cp.float32)
    energy_gain = cp.asarray(args.geo_energy_gain + 0.060 * cp.sin(23.0 * u + 0.70), dtype=cp.float32)
    comparison = cp.asarray(args.geo_comparison_scale + 0.004 * cp.sin(31.0 * u + 1.20), dtype=cp.float32)
    offset_def = cp.asarray(args.geo_offset_deform + 0.080 * cp.sin(19.0 * u + 0.30), dtype=cp.float32)
    phase_scale = cp.asarray(0.37 + 0.090 * cp.sin(29.0 * u + 0.50), dtype=cp.float32)
    phase_shift = cp.asarray(0.18 * cp.sin(37.0 * u + 0.90), dtype=cp.float32)

    d_base = cp.asarray(np.asarray(DEFAULT_BASE_DELAYS, dtype=np.int32))
    d_out = cp.empty((n_candidates, n_conditions, N_SUMMARY_METRICS), dtype=cp.float32)

    # Warmup.
    ctx.k_geo_sweep(
        (n_candidates, n_conditions),
        (1,),
        (
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
        ),
    )
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(reps):
        ctx.k_geo_sweep(
            (n_candidates, n_conditions),
            (1,),
            (
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
            ),
        )
    end.record()
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    per_rep_ms = elapsed_ms / float(reps)
    points_per_sec = float(points_per_rep / (per_rep_ms / 1000.0)) if per_rep_ms > 0 else None
    candidates_per_sec = float(n_candidates * n_conditions / (per_rep_ms / 1000.0)) if per_rep_ms > 0 else None

    # Pull a tiny sanity slice only.
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
# REPORT ROWS
# =============================================================================

def summary_dict(summary: np.ndarray) -> Dict[str, float]:
    return {SUMMARY_METRICS[i]: float(summary[i]) for i in range(N_SUMMARY_METRICS)}


def projection_summary_row(p: DMProjection) -> Dict[str, Any]:
    d = summary_dict(p.summary)
    d["pi_witness_strength"] = float(d["yzzy_energy_mean"] * d["pi_periodic_score"])
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
        "pi_periodic_mode_label": "log1p" if int(round(d["pi_periodic_mode"])) == 1 else "linear",
        **d,
    }


def control_row(p: DMProjection) -> Optional[Dict[str, Any]]:
    if p.control_summary is None:
        return None
    real = summary_dict(p.summary)
    ctrl = summary_dict(p.control_summary)
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
            va = summary_dict(conds[a].summary)
            vb = summary_dict(conds[b].summary)
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
                "summary_l2": l2(fa, fb),
                "summary_corr": safe_corr(fa, fb),
            })
    return rows


def summary_vector(p: DMProjection) -> np.ndarray:
    d = summary_dict(p.summary)
    d["pi_witness_strength"] = float(d["yzzy_energy_mean"] * d["pi_periodic_score"])
    return np.asarray([d[k] for k in SUMMARY_COMPARE_FIELDS], dtype=np.float32)


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

                # Rung-level same-condition shape agreement.
                n = min(pa.rung_stats.shape[0], pb.rung_stats.shape[0])
                energy_corr = safe_corr(pa.rung_stats[:n, 6], pb.rung_stats[:n, 6])
                spec_corr = safe_corr(pa.rung_stats[:n, 8], pb.rung_stats[:n, 8])
                yz_corr = safe_corr(pa.rung_stats[:n, 1], pb.rung_stats[:n, 1])
                phase_corr = safe_corr(np.cos(2.0 * pa.rung_stats[:n, 11]), np.cos(2.0 * pb.rung_stats[:n, 11]))

                rows.append({
                    "condition": cond,
                    "substrate_a": a,
                    "substrate_b": b,
                    "summary_corr": safe_corr(va, vb),
                    "summary_l2": l2(va, vb),
                    "projection_a": float(pa.summary[15]),
                    "projection_b": float(pb.summary[15]),
                    "projection_delta_abs": abs(float(pa.summary[15]) - float(pb.summary[15])),
                    "rung_energy_corr": energy_corr,
                    "rung_specificity_corr": spec_corr,
                    "rung_yz_corr": yz_corr,
                    "rung_pi_cos2_corr": phase_corr,
                })
    return rows


# =============================================================================
# PLOTTING
# =============================================================================

def plot_metric(projs: List[DMProjection], metric: str, out_path: Path) -> None:
    if not _HAVE_MPL:
        return
    idx = SUMMARY_METRICS.index(metric)
    order = {"null": 0, "base_only": 1, "offset_on": 2}
    subs = sorted(set(p.substrate for p in projs))
    conds = sorted(set(p.condition for p in projs), key=lambda c: order.get(c, 99))
    x = np.arange(len(conds), dtype=np.float64)
    width = 0.8 / max(1, len(subs))

    fig = plt.figure(figsize=(9, 5), dpi=150)
    ax = fig.add_subplot(111)

    for si, sub in enumerate(subs):
        vals = []
        for c in conds:
            found = [p for p in projs if p.substrate == sub and p.condition == c]
            vals.append(float(found[0].summary[idx]) if found else 0.0)
        ax.bar(x + (si - (len(subs) - 1) / 2) * width, vals, width=width, label=sub)

    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel(metric)
    ax.set_title(f"D_M {metric} by condition/substrate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =============================================================================
# AUTO DISCOVERY
# =============================================================================

def newest_glob(pattern: str) -> Optional[Path]:
    paths = list(DATA_DIR.glob(pattern))
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def auto_path(substrate: str, condition: str) -> Optional[Path]:
    if substrate == "gproj":
        if condition == "null":
            return newest_glob("dm_gpu_data_null_*.npz")
        if condition == "base_only":
            return newest_glob("dm_gpu_data_base_delay_*.npz")
        if condition == "offset_on":
            return newest_glob("dm_gpu_data_offset_deformed_*.npz")
    # QPROJ auto-discovery is intentionally conservative; names do not encode
    # condition reliably, so explicit qproj paths are preferred.
    return None


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D_M final benchmark — dimensional entanglement projection operator. Non-normalized projection score with physical pi witness strength.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--qproj-null", default=str(DEFAULT_QPROJ_NULL))
    p.add_argument("--qproj-base", default=str(DEFAULT_QPROJ_BASE))
    p.add_argument("--qproj-offset", default=str(DEFAULT_QPROJ_OFFSET))

    p.add_argument("--gproj-null", default=str(DEFAULT_GPROJ_NULL))
    p.add_argument("--gproj-base", default=str(DEFAULT_GPROJ_BASE))
    p.add_argument("--gproj-offset", default=str(DEFAULT_GPROJ_OFFSET))

    p.add_argument("--skip-qproj", action="store_true")
    p.add_argument("--skip-gproj", action="store_true")
    p.add_argument("--skip-geo", action="store_true")

    p.add_argument("--auto-gproj", action="store_true", help="Auto-load newest gproj files by condition.")
    p.add_argument("--repair-metadata", action="store_true", help="Repair legacy D_M files with condition hints.")

    p.add_argument("--out-dir", default=None)
    p.add_argument("--reps", type=int, default=200, help="CUDA timing reps for qproj/gproj record bases.")
    p.add_argument("--geo-reps", type=int, default=20000, help="Timing reps for single analytic GEO condition projection.")
    p.add_argument("--geo-sweep-candidates", type=int, default=262144, help="Candidates for batched GEO sweep speed test.")
    p.add_argument("--geo-sweep-reps", type=int, default=200, help="Timing reps for batched GEO sweep kernel.")
    p.add_argument("--skip-geo-sweep", action="store_true", help="Disable batched GEO sweep timing.")
    p.add_argument("--cuda-threads", type=int, default=256)
    p.add_argument("--control-seed", type=int, default=12345)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")

    # GEO analytic parameters.
    p.add_argument("--geo-base-energy", type=float, default=0.030)
    p.add_argument("--geo-energy-gain", type=float, default=0.285)
    p.add_argument("--geo-comparison-scale", type=float, default=0.010)
    p.add_argument("--geo-offset-deform", type=float, default=0.18)

    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

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


def main() -> None:
    args = parse_args()
    t0 = time.time()

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"dm_probe_12_benchmark_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = compile_dm_cuda(args)

    if not args.quiet:
        print("\n" + "=" * 112)
        print("  D_M BENCHMARK — FINAL DIMENSIONAL ENTANGLEMENT PROJECTION OPERATOR")
        print("=" * 112)
        print(f"  Data dir     : {DATA_DIR}")
        print(f"  Analysis dir : {out_dir}")
        print(f"  CUDA kernel  : {'yes' if ctx is not None else 'no'}")
        if ctx is not None:
            print(f"  Kernel path  : {ctx.path}")
        print(f"  GEO path     : {'yes' if not args.skip_geo else 'no'}")
        print("-" * 112)

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
            if not args.quiet:
                print(f"[QPROJ] {cond}: {path}")
            p = load_and_project(path, "qproj", cond, ctx, args)
            projs.append(p)
            base_meta[f"qproj_{cond}"] = {"path": str(path), "backend": p.backend, "job_id": p.job_id}

    if not args.skip_gproj:
        for cond, value in gproj_args.items():
            if not value and args.auto_gproj:
                found = auto_path("gproj", cond)
                value = str(found) if found else None
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            if not args.quiet:
                print(f"[GPROJ] {cond}: {path}")
            p = load_and_project(path, "gproj", cond, ctx, args)
            projs.append(p)
            base_meta[f"gproj_{cond}"] = {"path": str(path), "backend": p.backend, "job_id": p.job_id}

    if not args.skip_geo:
        for cond in ["null", "base_only", "offset_on"]:
            if not args.quiet:
                print(f"[GEO] {cond}: analytic manifold")
            p = timed_geo_condition_projection(cond, args, ctx)
            projs.append(p)
            base_meta[f"geo_{cond}"] = {"path": "analytic_geo", "backend": p.backend, "job_id": p.job_id}

    if not projs:
        raise RuntimeError("No D_M substrates available. Provide qproj/gproj files or enable GEO.")

    geo_sweep = {"enabled": False, "reason": "no cuda context"}
    if ctx is not None and not args.skip_geo and not args.skip_geo_sweep:
        if not args.quiet:
            print(f"[GEO SWEEP] candidates={args.geo_sweep_candidates} reps={args.geo_sweep_reps}")
        try:
            geo_sweep = run_geo_sweep_cuda(ctx, args)
            # Report batched sweep rate on each GEO projection row so the table
            # reflects optimized minimized GEO throughput rather than tiny
            # single-manifold launch overhead.
            if geo_sweep.get("enabled") and geo_sweep.get("analytic_points_per_second"):
                for p in projs:
                    if p.substrate == "geo":
                        p.timing["records_per_sec_projection"] = geo_sweep["analytic_points_per_second"]
                        p.timing["records_semantics"] = "batched_analytic_points_per_second"
                        p.timing["geo_sweep_candidates"] = geo_sweep["candidates"]
                        p.timing["geo_sweep_per_rep_ms"] = geo_sweep["per_rep_ms"]
        except Exception as e:
            geo_sweep = {"enabled": False, "error": f"{type(e).__name__}: {e}"}
            if not args.quiet:
                print(f"[GEO SWEEP][warn] {geo_sweep['error']}")

    summary_rows = [projection_summary_row(p) for p in projs]
    rung_all: List[Dict[str, Any]] = []
    for p in projs:
        rung_all.extend(rung_rows(p))

    control_rows = [r for r in (control_row(p) for p in projs) if r is not None]
    sep_rows = condition_separation(projs)
    agree_rows = substrate_agreement(projs)

    summary_fields = [
        "substrate", "condition", "condition_label", "source", "backend", "job_id",
        "tiles", "shots", "n_rungs", "timing_mode", "projection_per_rep_ms",
        "records_per_sec_projection", "pi_periodic_mode_label",
    ] + SUMMARY_METRICS + ["pi_witness_strength"]

    rung_fields = ["substrate", "condition", "rung", "source"] + RUNG_METRICS + ["pi_phase_deg"]

    sep_fields = [
        "substrate", "condition_a", "condition_b",
        "delta_projection", "delta_energy", "delta_specificity", "delta_pi_score",
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
        "energy_real", "energy_control", "specificity_real", "specificity_control",
    ]

    write_csv(out_dir / "projection_summary.csv", summary_rows, summary_fields)
    write_csv(out_dir / "rung_projection.csv", rung_all, rung_fields)
    write_csv(out_dir / "condition_separation.csv", sep_rows, sep_fields)
    write_csv(out_dir / "substrate_agreement.csv", agree_rows, agree_fields)
    write_csv(out_dir / "control_collapse.csv", control_rows, control_fields)

    artifacts: Dict[str, np.ndarray] = {}
    for p in projs:
        key = f"{p.substrate}_{p.condition}"
        artifacts[f"{key}_summary"] = p.summary.astype(np.float32)
        artifacts[f"{key}_rung_stats"] = p.rung_stats.astype(np.float32)
        if p.tile_stats is not None:
            artifacts[f"{key}_tile_stats"] = p.tile_stats.astype(np.float32)
        if p.control_summary is not None:
            artifacts[f"{key}_control_summary"] = p.control_summary.astype(np.float32)
    np.savez_compressed(out_dir / "artifacts.npz", **artifacts)

    result = {
        "schema": "ghost_oracle.dm.benchmark_result.v1",
        "created": now_tag(),
        "seconds": time.time() - t0,
        "config": {
            "cuda_enabled": ctx is not None,
            "kernel_path": str(ctx.path) if ctx is not None else None,
            "reps": args.reps,
            "geo_reps": args.geo_reps,
            "geo_sweep_candidates": args.geo_sweep_candidates,
            "geo_sweep_reps": args.geo_sweep_reps,
            "cuda_threads": args.cuda_threads,
            "skip_qproj": args.skip_qproj,
            "skip_gproj": args.skip_gproj,
            "skip_geo": args.skip_geo,
            "analysis_dir": str(out_dir),
        },
        "base_meta": base_meta,
        "projection_summary": summary_rows,
        "condition_separation": sep_rows,
        "substrate_agreement": agree_rows,
        "control_collapse": control_rows,
        "geo_sweep": geo_sweep,
        "bounded_claim": (
            "D_M projects a YZ-primary / ZY-reciprocal dimensional entanglement "
            "manifold from qproj/gproj/geo substrates and separates null, base-delay, "
            "and offset-deformed witness conditions under shared projection metrics."
        ),
        "non_claims": [
            "D_M does not reconstruct density matrices.",
            "D_M does not certify device-independent Bell nonlocality.",
            "D_M does not prove prepared Bell states.",
            "D_M is not a QPU speedup or quantum advantage claim.",
            "GPROJ is not an IBM hardware simulator.",
        ],
    }
    write_json(out_dir / "result.json", result)

    if not args.no_plots:
        plot_metric(projs, "projection_score", out_dir / "projection_score_by_condition.png")
        plot_metric(projs, "yzzy_energy_mean", out_dir / "energy_by_condition.png")
        plot_metric(projs, "pi_periodic_score", out_dir / "pi_score_by_condition.png")

    if not args.quiet:
        print("\n" + "=" * 112)
        print("  D_M BENCHMARK SUMMARY")
        print("=" * 112)
        print(
            f"  {'substrate':<8} | {'condition':<10} | {'projection':>10} | "
            f"{'energy':>10} | {'spec':>10} | {'π score':>8} | {'YZ+':>5} | {'rate/s':>14}"
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
                f"{float(row.get('pi_witness_strength', float(row['yzzy_energy_mean']) * float(row['pi_periodic_score']))):>8.5f} | "
                f"{float(row['yz_pos_frac']):>5.2f} | {rps_s:>14}"
            )

        if geo_sweep.get("enabled"):
            print("\n  GEO SWEEP")
            print("  " + "-" * 110)
            aps = geo_sweep.get("analytic_points_per_second")
            cps = geo_sweep.get("candidate_conditions_per_second")
            print(
                f"  candidates={geo_sweep.get('candidates'):,} "
                f"conditions={geo_sweep.get('conditions')} "
                f"points/rep={geo_sweep.get('points_per_rep'):,} "
                f"per_rep={geo_sweep.get('per_rep_ms'):.6f} ms"
            )
            print(
                f"  analytic points/sec={aps:,.0f} "
                f"candidate-conditions/sec={cps:,.0f}"
            )

        if control_rows:
            print("\n  CONTROL COLLAPSE")
            print("  " + "-" * 110)
            for r in control_rows:
                print(
                    f"  {r['substrate']:<8} {r['condition']:<10} "
                    f"projection {r['projection_real']:.6f} -> {r['projection_control']:.6f} "
                    f"drop={r['projection_drop_fraction']:.2%}"
                )

        print("\n[SAVED]")
        print(f"  result              : {out_dir / 'result.json'}")
        print(f"  projection summary  : {out_dir / 'projection_summary.csv'}")
        print(f"  rung projection     : {out_dir / 'rung_projection.csv'}")
        print(f"  condition separation: {out_dir / 'condition_separation.csv'}")
        print(f"  substrate agreement : {out_dir / 'substrate_agreement.csv'}")
        print(f"  control collapse    : {out_dir / 'control_collapse.csv'}")
        print(f"  artifacts           : {out_dir / 'artifacts.npz'}")
        print("\nDone. Break it, fix it, document what happened.\n")


if __name__ == "__main__":
    main()
