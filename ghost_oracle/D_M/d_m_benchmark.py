#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — D_M FINAL CAPSTONE BENCHMARK
==============================================================================

D_M = Dimensional Entanglement Projection Operator

This is the repo-facing capstone benchmark for the final D_M claim envelope.
It updates the older benchmark to the Probe 22 / 23 / 24 / 25 standard:

    Probe 22  -> final qproj/gproj CUDA record path
    Probe 23  -> allowed-channel invariance and forbidden single-fault checks
    Probe 24  -> compound corruption / collapse-boundary checks
    Probe 25  -> exact closed-form GEO classical reference path

Default run
-----------
    VERIFY:
        qproj / gproj / exact GEO condition projection
        null / base_only / offset_on

    CONTROLS:
        qproj/gproj independent-bit-shuffle collapse
        allowed channel re-description retention
        forbidden single-fault weakening
        compound corruption boundary

    GEO VALIDATION:
        exact zero null manifold
        active positive projection
        active pi score
        CPU/GPU agreement for the exact GEO rule

Optional appendix
-----------------
The old synthetic DER path is intentionally not a default final claim. Probe 19/22
retrieval saturated under the small task, so utility claims should not lean on it.
This file focuses the default capstone on the stronger, control-clearing evidence.

Bounded claim
-------------
D_M projects a YZ-primary / ZY-reciprocal dimensional witness manifold across
qproj, gproj, and exact GEO substrates; active base-delay / offset manifolds
separate from null; same-shot pairing, reciprocal structure, and delay order are
load-bearing; compound corruptions cross a measurable collapse boundary.

Non-claims
----------
    - D_M does not reconstruct density matrices.
    - D_M does not certify device-independent Bell nonlocality.
    - D_M does not prove prepared Bell states.
    - D_M is not a QPU speedup or quantum advantage claim.
    - GPROJ is not an IBM hardware simulator.
    - GEO is a closed-form classical reference, not a hardware simulator.
    - GPT-2 is not a D_M input in the final claim.

Usage
-----
    python ghost_oracle/D_M/d_m_benchmark.py \
      --qproj-null   ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_d8fm4ihvjngc73aq3ccg.npz \
      --qproj-base   ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_d8flk2jo3njc73f0g560.npz \
      --qproj-offset ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_d8fl82bo3njc73f0fgd0.npz \
      --gproj-null   ghost_oracle/D_M/data/dm_gpu_data_null_4096shots_seed9031229662612491082.npz \
      --gproj-base   ghost_oracle/D_M/data/dm_gpu_data_base_delay_4096shots_seed2669559634056472362.npz \
      --gproj-offset ghost_oracle/D_M/data/dm_gpu_data_offset_deformed_4096shots_seed6727069190982977623.npz

==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "[FATAL] CuPy is required for the final GPU D_M benchmark and could not "
        f"be imported: {e!r}\nInstall a CUDA-matched CuPy build and rerun."
    )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    plt = None
    HAVE_MPL = False


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

HERE = Path(__file__).resolve().parent
D_M_DIR = HERE
DATA_DIR = D_M_DIR / "data"
ANALYSIS_DIR = D_M_DIR / "analysis"
KERNEL_PATH = D_M_DIR / "kernels" / "dm_projector_kernel.cu"

DEFAULT_BASE_DELAYS = [0, 256, 1024, 4096, 16384]
DEFAULT_NULL_DELAYS = [0, 0, 0, 0, 0]
DEFAULT_OFFSET_DT = 128

DEFAULT_QPROJ_NULL = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fm4ihvjngc73aq3ccg.npz"
DEFAULT_QPROJ_BASE = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8flk2jo3njc73f0g560.npz"
DEFAULT_QPROJ_OFFSET = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fl82bo3njc73f0fgd0.npz"
DEFAULT_GPROJ_NULL = DATA_DIR / "dm_gpu_data_null_4096shots_seed9031229662612491082.npz"
DEFAULT_GPROJ_BASE = DATA_DIR / "dm_gpu_data_base_delay_4096shots_seed2669559634056472362.npz"
DEFAULT_GPROJ_OFFSET = DATA_DIR / "dm_gpu_data_offset_deformed_4096shots_seed6727069190982977623.npz"

WITNESS_LABELS = ["XY", "YZ", "ZY", "YX"]
N_TILE_METRICS = 12
N_RUNG_METRICS = 19
N_SUMMARY_METRICS = 16
SIGN_EPS = 1.0e-6

RUNG_METRICS = [
    "XY", "YZ", "ZY", "YX",
    "YZ_primary", "ZY_return", "YZ_ZY_energy", "comparison_energy",
    "directional_specificity", "directional_gap", "inversion",
    "pi_phase", "pi_cos2", "pi_sin2",
    "base_delay", "offset", "total_delay",
    "count_all", "count_yzzy",
]
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
SUMMARY_PROJECTION_IDX = 15
CONDITION_TO_KIND = {"null": 0, "base_only": 1, "offset_on": 2}
CONDITION_LABEL = {
    "null": "no_delay_no_offset",
    "base_only": "base_delay_only",
    "offset_on": "base_delay_plus_offset",
}
CONDITION_ORDER = {"null": 0, "base_only": 1, "offset_on": 2}
SUBSTRATE_ORDER = {"geo": 0, "gproj": 1, "qproj": 2}

CORE_SUMMARY_FIELDS = [
    "n_rungs",
    "yz_mean",
    "zy_mean",
    "yzzy_energy_mean",
    "yzzy_energy_max",
    "specificity_mean",
    "specificity_max",
    "pi_periodic_score",
    "energy_tracking_r",
    "specificity_tracking_r",
    "phase_span_pi_units",
    "projection_score",
]
AUX_SUMMARY_FIELDS = [
    "yz_pos_frac",
    "zy_inverted_frac",
    "pi_periodic_mode",
    "phase_velocity_r",
]

ORIENTATIONS = [
    ("YZ<-ZY canonical", 1, 2, 0, 3),
    ("ZY<-YZ reciprocal", 2, 1, 3, 0),
    ("XY<-YX pair", 0, 3, 1, 2),
    ("YX<-XY reciprocal", 3, 0, 2, 1),
]
ALLOWED_TRANSFORMS = {
    "identity": (0, 1, 2, 3),
    "equiv_pair_swap": (1, 0, 3, 2),
    "equiv_reciprocal_swap": (3, 2, 1, 0),
    "equiv_cyclic_rotation": (1, 2, 3, 0),
}
FAULTS = [
    "reciprocal_break",
    "cross_rung_delay_scramble",
    "same_label_wrong_delay",
    "non_equivalence_channel_corruption",
    "independent_bit_shuffle_model",
]


# =============================================================================
# IO / SMALL MATH
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
    path.write_text(json.dumps(json_safe(obj), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def scalar_str(obj: Any) -> str:
    arr = np.asarray(obj)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(obj)


def decode_str_array(arr: Any) -> List[str]:
    a = np.asarray(arr)
    out: List[str] = []
    for x in a.reshape(-1):
        out.append(x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x))
    return out


def section(title: str, width: int = 112) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n < 3:
        return 0.0
    x = x[:n]
    y = y[:n]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def l2(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n <= 0:
        return float("nan")
    return float(np.linalg.norm(x[:n] - y[:n]))

def finite_array(x: Sequence[float]) -> np.ndarray:
    """Return only finite values from a numeric sequence; used to avoid NumPy all-NaN warnings."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def safe_nanmean(x: Sequence[float], default: float = float("nan")) -> float:
    arr = finite_array(x)
    return float(np.mean(arr)) if arr.size else float(default)


def safe_nanmedian(x: Sequence[float], default: float = float("nan")) -> float:
    arr = finite_array(x)
    return float(np.median(arr)) if arr.size else float(default)


def safe_nanpercentile(x: Sequence[float], q: float, default: float = float("nan")) -> float:
    arr = finite_array(x)
    return float(np.percentile(arr, q)) if arr.size else float(default)


def safe_fraction(mask_values: Sequence[float], default: float = float("nan")) -> float:
    arr = finite_array(mask_values)
    return float(np.mean(arr)) if arr.size else float(default)



def wrap_pi(x: np.ndarray | float) -> np.ndarray | float:
    return np.mod(x, math.pi)


def wrap_pi_delta(d: float) -> float:
    y = math.fmod(float(d) + 0.5 * math.pi, math.pi)
    if y < 0:
        y += math.pi
    return y - 0.5 * math.pi


def norm_x(x_raw: np.ndarray, mode: int) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64).copy()
    if mode == 1:
        x = np.log1p(np.maximum(0.0, x))
    span = float(np.max(x) - np.min(x)) if x.size else 0.0
    if abs(span) <= 1e-12:
        return np.zeros_like(x)
    return (x - np.min(x)) / span


def pi_periodic_score(x_raw: np.ndarray, phase: np.ndarray, mode: int) -> float:
    if len(x_raw) < 3:
        return 0.0
    x = norm_x(np.asarray(x_raw, dtype=np.float64), mode)
    c2 = np.cos(2.0 * phase)
    s2 = np.sin(2.0 * phase)
    rc = safe_corr(x, c2)
    rs = safe_corr(x, s2)
    return float(min(1.0, math.sqrt(rc * rc + rs * rs)))


# =============================================================================
# CUDA WRAPPER
# =============================================================================

@dataclass
class DMCudaContext:
    module: Any
    k_make_pair: Any
    k_tile: Any
    k_shuffle: Any
    k_rung: Any
    k_summary: Any
    k_geo_legacy: Any
    k_geo_sweep_legacy: Any
    k_geo_exact: Any
    k_geo_exact_sweep: Any
    k_der: Any
    path: Path


def as_cp_i32(x: Any) -> "cp.ndarray":
    return cp.ascontiguousarray(cp.asarray(np.asarray(x, dtype=np.int32)))


def compile_dm_cuda(kernel_path: Path) -> DMCudaContext:
    if not kernel_path.exists():
        raise SystemExit(f"[FATAL] D_M kernel not found: {kernel_path}")
    src = kernel_path.read_text(encoding="utf-8")
    names = [
        "dm_make_projected_pair_from_bits_kernel_u8",
        "dm_tile_correlator_kernel_u8",
        "dm_independent_bit_shuffle_tile_kernel_u8",
        "dm_rung_projection_kernel_f32",
        "dm_projection_summary_kernel_f32",
        "dm_geo_rung_projection_kernel_f32",
        "dm_geo_sweep_summary_kernel_f32",
        "dm_geo_exact_rung_projection_kernel_f32",
        "dm_geo_exact_sweep_summary_kernel_f32",
        "dm_der_topk_kernel_f32",
    ]
    try:
        mod = cp.RawModule(code=src, options=("--std=c++11",), name_expressions=names)
    except Exception as e:
        raise SystemExit(f"[FATAL] Could not compile {kernel_path}: {type(e).__name__}: {e}")
    return DMCudaContext(
        module=mod,
        k_make_pair=mod.get_function("dm_make_projected_pair_from_bits_kernel_u8"),
        k_tile=mod.get_function("dm_tile_correlator_kernel_u8"),
        k_shuffle=mod.get_function("dm_independent_bit_shuffle_tile_kernel_u8"),
        k_rung=mod.get_function("dm_rung_projection_kernel_f32"),
        k_summary=mod.get_function("dm_projection_summary_kernel_f32"),
        k_geo_legacy=mod.get_function("dm_geo_rung_projection_kernel_f32"),
        k_geo_sweep_legacy=mod.get_function("dm_geo_sweep_summary_kernel_f32"),
        k_geo_exact=mod.get_function("dm_geo_exact_rung_projection_kernel_f32"),
        k_geo_exact_sweep=mod.get_function("dm_geo_exact_sweep_summary_kernel_f32"),
        k_der=mod.get_function("dm_der_topk_kernel_f32"),
        path=kernel_path,
    )


# =============================================================================
# DATA MODEL / LOADING
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
    rung_stats: np.ndarray
    summary: np.ndarray
    control_summary: Optional[np.ndarray]
    timing: Dict[str, Any]
    notes: str = ""


def build_metadata_from_condition(condition: str) -> Tuple[np.ndarray, ...]:
    if condition == "null":
        base_delays, offset_dt = DEFAULT_NULL_DELAYS, 0
    elif condition == "base_only":
        base_delays, offset_dt = DEFAULT_BASE_DELAYS, 0
    elif condition == "offset_on":
        base_delays, offset_dt = DEFAULT_BASE_DELAYS, DEFAULT_OFFSET_DT
    else:
        raise ValueError(condition)
    tile_rung, tile_wi, tile_base, tile_off, tile_total = [], [], [], [], []
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
    return tuple(np.asarray(x, dtype=np.int32) for x in (tile_rung, tile_wi, tile_base, tile_off, tile_total))


def load_dm_base(path: Path, substrate: str, condition_hint: str, repair: bool) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    if "pair" in z.files:
        pair = np.asarray(z["pair"], dtype=np.uint8)
    else:
        keys = sorted([k for k in z.files if k.startswith("pair_tile")], key=lambda k: int(k.replace("pair_tile", "")))
        if not keys:
            raise KeyError(f"{path} has no pair or pair_tile* arrays")
        pair = np.stack([np.asarray(z[k], dtype=np.uint8) for k in keys], axis=0)
    if pair.ndim != 3 or pair.shape[2] != 2:
        raise ValueError(f"{path} pair must have shape (tiles, shots, 2), got {pair.shape}")
    tiles, shots, _ = pair.shape

    def arr(name: str, dtype: Any) -> Optional[np.ndarray]:
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
            raise KeyError(f"{path} missing usable D_M metadata. Use --repair-metadata.")
        tile_rung, tile_wi, tile_base, tile_off, tile_total = build_metadata_from_condition(condition_hint)
        if len(tile_rung) != tiles:
            raise ValueError(f"repair metadata length {len(tile_rung)} != tiles {tiles}")
        repaired = True

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
        "condition": condition_hint,
        "condition_label": CONDITION_LABEL[condition_hint],
        "repaired": repaired,
    }


# =============================================================================
# CPU REFERENCE SUMMARY / GEO
# =============================================================================

def cpu_summary_from_rung(rung_stats: np.ndarray) -> np.ndarray:
    rs = np.asarray(rung_stats, dtype=np.float64)
    valid = rs[:, 18] > 0.0
    if not np.any(valid):
        return np.zeros((N_SUMMARY_METRICS,), dtype=np.float64)
    v = rs[valid]
    yz = v[:, 4]
    zy = v[:, 2]
    energy = v[:, 6]
    spec = v[:, 8]
    phase = v[:, 11]
    x_total = v[:, 16]
    n = len(v)
    yz_mean = float(np.mean(yz))
    zy_mean = float(np.mean(zy))
    e_mean = float(np.mean(energy))
    s_mean = float(np.mean(spec))
    yz_pos_frac = float(np.mean(yz > SIGN_EPS))
    zy_inv_frac = float(np.mean(yz * zy < -SIGN_EPS))
    e_max = float(np.max(energy))
    s_max = float(np.max(spec))
    x_lin = norm_x(x_total, 0)
    x_log = norm_x(x_total, 1)
    e_r_lin = safe_corr(x_lin, energy)
    e_r_log = safe_corr(x_log, energy)
    s_r_lin = safe_corr(x_lin, spec)
    s_r_log = safe_corr(x_log, spec)
    e_r = e_r_log if abs(e_r_log) > abs(e_r_lin) else e_r_lin
    s_r = s_r_log if abs(s_r_log) > abs(s_r_lin) else s_r_lin
    pi_lin = pi_periodic_score(x_total, phase, 0)
    pi_log = pi_periodic_score(x_total, phase, 1)
    pi_score = pi_log if pi_log > pi_lin else pi_lin
    pi_mode = 1.0 if pi_log > pi_lin else 0.0
    phase_vel_r = 0.0
    phase_span = 0.0
    if n >= 3:
        acc = float(phase[0])
        pmin = pmax = acc
        mid_x, vel = [], []
        for i in range(1, n):
            dx = float(x_total[i] - x_total[i - 1])
            dp = wrap_pi_delta(float(phase[i] - phase[i - 1]))
            acc += dp
            pmin = min(pmin, acc)
            pmax = max(pmax, acc)
            if abs(dx) > 1e-12:
                mid_x.append(0.5 * float(x_total[i] + x_total[i - 1]))
                vel.append(dp / dx)
        phase_span = abs(pmax - pmin) / math.pi
        if len(vel) >= 3:
            phase_vel_r = safe_corr(norm_x(np.asarray(mid_x), 1), np.asarray(vel))
    pi_witness_strength = max(0.0, e_mean) * pi_score
    projection = (
        0.35 * max(0.0, e_mean)
        + 0.25 * max(0.0, s_mean)
        + 0.15 * max(0.0, yz_mean)
        + 0.15 * pi_witness_strength
        + 0.10 * (0.5 * (abs(e_r) + abs(s_r)))
    )
    return np.asarray([
        float(n), yz_mean, yz_pos_frac, zy_mean, zy_inv_frac,
        e_mean, e_max, s_mean, s_max,
        pi_score, pi_mode, e_r, s_r, phase_vel_r, phase_span, projection,
    ], dtype=np.float64)


def write_rung_from_witnesses(xy: float, yz: float, zy: float, yx: float, base: float, off: float, total: float, count_yzzy: float = 1.0) -> np.ndarray:
    ret = -zy
    energy = math.sqrt(yz * yz + ret * ret)
    comp = math.sqrt(xy * xy + yx * yx)
    spec = energy - comp
    phase = float(wrap_pi(math.atan2(ret, yz)))
    return np.asarray([
        xy, yz, zy, yx,
        yz, ret, energy, comp, spec, yz - zy, -yz * zy,
        phase, math.cos(2.0 * phase), math.sin(2.0 * phase),
        base, off, total, 4.0, count_yzzy,
    ], dtype=np.float64)


def cpu_geo_exact(condition: str, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if condition == "null":
        base_delays, offset_dt, kind = DEFAULT_NULL_DELAYS, 0, 0
    elif condition == "base_only":
        base_delays, offset_dt, kind = DEFAULT_BASE_DELAYS, 0, 1
    elif condition == "offset_on":
        base_delays, offset_dt, kind = DEFAULT_BASE_DELAYS, DEFAULT_OFFSET_DT, 2
    else:
        raise ValueError(condition)
    rows = []
    if kind == 0:
        for r, base in enumerate(base_delays):
            rows.append(write_rung_from_witnesses(0.0, 0.0, 0.0, 0.0, float(base), 0.0, float(base), 0.0))
        rung = np.vstack(rows)
        return rung, cpu_summary_from_rung(rung)

    base_arr = np.asarray(base_delays, dtype=np.float64)
    off_arr = np.asarray([((4 * r) + 1.5) * offset_dt for r in range(len(base_delays))], dtype=np.float64)
    x_space = norm_x(np.log1p(np.maximum(0.0, base_arr)), 0)
    x_time = norm_x(np.log1p(np.maximum(0.0, base_arr + off_arr)), 0)
    ws = max(0.0, float(args.geo_weight_space))
    wt = max(0.0, float(args.geo_weight_time))
    denom = max(ws + wt, 1e-12)
    gamma = max(float(args.geo_energy_gamma), 1e-9)
    for r, base in enumerate(base_delays):
        x_dm = math.sqrt((ws * x_space[r] * x_space[r] + wt * x_time[r] * x_time[r]) / denom)
        energy = max(0.0, float(args.geo_energy_floor) + float(args.geo_energy_scale) * (max(0.0, x_dm) ** gamma))
        cos2phi = min(1.0, max(-1.0, 2.0 * x_time[r] - 1.0))
        phi = 0.5 * math.acos(cos2phi)
        yz = energy * math.cos(phi)
        zy = -energy * math.sin(phi)
        off = float(off_arr[r])
        rows.append(write_rung_from_witnesses(0.0, yz, zy, 0.0, float(base), off, float(base) + off, 1.0))
    rung = np.vstack(rows)
    return rung, cpu_summary_from_rung(rung)


# =============================================================================
# GPU PROJECTION PATHS
# =============================================================================

def project_record(base: Dict[str, Any], ctx: DMCudaContext, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    pair = np.asarray(base["pair"], dtype=np.uint8)
    tiles = np.int32(base["tiles"])
    shots = np.int32(base["shots"])
    n_rungs = np.int32(base["n_rungs"])
    block = int(args.cuda_threads)

    d_pair = cp.asarray(pair)
    d_tile_rung = as_cp_i32(base["tile_rung_index"])
    d_tile_wi = as_cp_i32(base["tile_witness_index"])
    d_base = as_cp_i32(base["tile_base_delay_dt"])
    d_off = as_cp_i32(base["tile_offset_dt"])
    d_total = as_cp_i32(base["tile_total_delay_dt"])

    d_tile = cp.zeros((int(tiles), N_TILE_METRICS), dtype=cp.float32)
    d_rung = cp.zeros((int(n_rungs), N_RUNG_METRICS), dtype=cp.float32)
    d_sum = cp.zeros((N_SUMMARY_METRICS,), dtype=cp.float32)
    d_ctrl_tile = cp.zeros_like(d_tile)
    d_ctrl_rung = cp.zeros_like(d_rung)
    d_ctrl_sum = cp.zeros_like(d_sum)

    def project_once(tile_dst: Any, rung_dst: Any, sum_dst: Any, shuffle: bool = False) -> None:
        if shuffle:
            ctx.k_shuffle((int(tiles),), (block,), (d_pair, tiles, shots, np.int32(args.control_seed), tile_dst))
        else:
            ctx.k_tile((int(tiles),), (block,), (d_pair, tiles, shots, tile_dst))
        ctx.k_rung((int(n_rungs),), (1,), (tile_dst, d_tile_rung, d_tile_wi, d_base, d_off, d_total, tiles, n_rungs, rung_dst))
        ctx.k_summary((1,), (1,), (rung_dst, n_rungs, sum_dst))

    project_once(d_tile, d_rung, d_sum, False)
    project_once(d_ctrl_tile, d_ctrl_rung, d_ctrl_sum, True)
    cp.cuda.Stream.null.synchronize()

    reps = max(1, int(args.reps))
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(reps):
        project_once(d_tile, d_rung, d_sum, False)
    end.record(); end.synchronize()
    proj_ms = float(cp.cuda.get_elapsed_time(start, end))

    c0, c1 = cp.cuda.Event(), cp.cuda.Event()
    c0.record()
    for _ in range(reps):
        project_once(d_ctrl_tile, d_ctrl_rung, d_ctrl_sum, True)
    c1.record(); c1.synchronize()
    ctrl_ms = float(cp.cuda.get_elapsed_time(c0, c1))

    records = int(base["tiles"] * base["shots"] * 2)
    per_ms = proj_ms / reps
    timing = {
        "mode": "cuda_record",
        "reps": reps,
        "projection_total_ms": proj_ms,
        "projection_per_rep_ms": per_ms,
        "control_total_ms": ctrl_ms,
        "control_per_rep_ms": ctrl_ms / reps,
        "records_per_rep": records,
        "records_per_sec_projection": float(records / (per_ms / 1000.0)) if per_ms > 0 else None,
        "records_semantics": "pair_bit_records_per_second",
    }
    return cp.asnumpy(d_rung), cp.asnumpy(d_sum), cp.asnumpy(d_ctrl_sum), timing


def load_and_project(path: Path, substrate: str, condition: str, ctx: DMCudaContext, args: argparse.Namespace) -> DMProjection:
    base = load_dm_base(path, substrate, condition, repair=args.repair_metadata)
    rung, summary, ctrl, timing = project_record(base, ctx, args)
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
        rung_stats=rung,
        summary=summary,
        control_summary=ctrl,
        timing=timing,
        notes="qproj/gproj record path via final kernel",
    )


def project_geo_exact_gpu(condition: str, ctx: DMCudaContext, args: argparse.Namespace) -> DMProjection:
    if condition == "null":
        base_delays, offset_dt = DEFAULT_NULL_DELAYS, 0
    elif condition == "base_only":
        base_delays, offset_dt = DEFAULT_BASE_DELAYS, 0
    elif condition == "offset_on":
        base_delays, offset_dt = DEFAULT_BASE_DELAYS, DEFAULT_OFFSET_DT
    else:
        raise ValueError(condition)
    n_rungs = len(base_delays)
    d_base = as_cp_i32(np.asarray(base_delays, dtype=np.int32))
    d_rung = cp.zeros((n_rungs, N_RUNG_METRICS), dtype=cp.float32)
    d_sum = cp.zeros((N_SUMMARY_METRICS,), dtype=cp.float32)

    kargs = (
        np.int32(CONDITION_TO_KIND[condition]), d_base, np.int32(n_rungs), np.int32(offset_dt),
        np.float32(args.geo_energy_floor), np.float32(args.geo_energy_scale), np.float32(args.geo_energy_gamma),
        np.float32(args.geo_weight_space), np.float32(args.geo_weight_time), d_rung,
    )
    ctx.k_geo_exact((n_rungs,), (1,), kargs)
    ctx.k_summary((1,), (1,), (d_rung, np.int32(n_rungs), d_sum))
    cp.cuda.Stream.null.synchronize()

    reps = max(1, int(args.geo_reps))
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(reps):
        ctx.k_geo_exact((n_rungs,), (1,), kargs)
        ctx.k_summary((1,), (1,), (d_rung, np.int32(n_rungs), d_sum))
    end.record(); end.synchronize()
    total_ms = float(cp.cuda.get_elapsed_time(start, end))
    return DMProjection(
        substrate="geo",
        condition=condition,
        condition_label=CONDITION_LABEL[condition],
        source="exact_geo_probe25",
        backend="closed_form_classical_cuda",
        job_id=f"geo_exact_{condition}",
        tiles=n_rungs * 4,
        shots=0,
        n_rungs=n_rungs,
        rung_stats=cp.asnumpy(d_rung),
        summary=cp.asnumpy(d_sum),
        control_summary=None,
        timing={
            "mode": "cuda_exact_geo",
            "reps": reps,
            "projection_total_ms": total_ms,
            "projection_per_rep_ms": total_ms / reps,
            "records_per_sec_projection": None,
            "records_semantics": "closed_form_rungs_per_second",
        },
        notes="Probe 25 exact closed-form GEO reference path",
    )


# =============================================================================
# REPORT ROWS
# =============================================================================

def summary_dict(summary: np.ndarray) -> Dict[str, float]:
    return {SUMMARY_METRICS[i]: float(summary[i]) for i in range(N_SUMMARY_METRICS)}


def add_pi_witness(d: Dict[str, float]) -> Dict[str, float]:
    d = dict(d)
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
        "timing_mode": p.timing.get("mode", ""),
        "projection_per_rep_ms": p.timing.get("projection_per_rep_ms", ""),
        "records_per_sec_projection": p.timing.get("records_per_sec_projection", ""),
        "records_semantics": p.timing.get("records_semantics", ""),
        **d,
    }


def rung_rows(p: DMProjection) -> List[Dict[str, Any]]:
    rows = []
    for r in range(p.rung_stats.shape[0]):
        row: Dict[str, Any] = {"substrate": p.substrate, "condition": p.condition, "rung": r, "source": p.source}
        row.update({RUNG_METRICS[i]: float(p.rung_stats[r, i]) for i in range(N_RUNG_METRICS)})
        row["pi_phase_degrees_mod180"] = float(p.rung_stats[r, 11] * 180.0 / math.pi)
        rows.append(row)
    return rows


def control_row(p: DMProjection) -> Optional[Dict[str, Any]]:
    if p.control_summary is None:
        return None
    real = add_pi_witness(summary_dict(p.summary))
    ctrl = add_pi_witness(summary_dict(p.control_summary))
    denom = real["projection_score"] if abs(real["projection_score"]) > 1e-12 else float("nan")
    return {
        "substrate": p.substrate,
        "condition": p.condition,
        "control": "independent_bit_shuffle",
        "projection_real": real["projection_score"],
        "projection_control": ctrl["projection_score"],
        "projection_delta": real["projection_score"] - ctrl["projection_score"],
        "projection_drop_fraction": (real["projection_score"] - ctrl["projection_score"]) / denom if math.isfinite(denom) else float("nan"),
        "energy_real": real["yzzy_energy_mean"],
        "energy_control": ctrl["yzzy_energy_mean"],
        "specificity_real": real["specificity_mean"],
        "specificity_control": ctrl["specificity_mean"],
        "piW_real": real["pi_witness_strength"],
        "piW_control": ctrl["pi_witness_strength"],
    }


def condition_separation(projs: List[DMProjection]) -> List[Dict[str, Any]]:
    rows = []
    by_sub: Dict[str, Dict[str, DMProjection]] = {}
    for p in projs:
        by_sub.setdefault(p.substrate, {})[p.condition] = p
    for sub, conds in sorted(by_sub.items()):
        for a, b in [("null", "base_only"), ("null", "offset_on"), ("base_only", "offset_on")]:
            if a not in conds or b not in conds:
                continue
            sa = add_pi_witness(summary_dict(conds[a].summary))
            sb = add_pi_witness(summary_dict(conds[b].summary))
            rows.append({
                "substrate": sub,
                "condition_a": a,
                "condition_b": b,
                "delta_projection": sb["projection_score"] - sa["projection_score"],
                "delta_energy": sb["yzzy_energy_mean"] - sa["yzzy_energy_mean"],
                "delta_specificity": sb["specificity_mean"] - sa["specificity_mean"],
                "delta_pi_score": sb["pi_periodic_score"] - sa["pi_periodic_score"],
                "delta_piW": sb["pi_witness_strength"] - sa["pi_witness_strength"],
                "summary_l2": l2(conds[a].summary, conds[b].summary),
                "summary_corr": safe_corr(conds[a].summary, conds[b].summary),
            })
    return rows


def substrate_agreement(projs: List[DMProjection]) -> List[Dict[str, Any]]:
    rows = []
    by_cond: Dict[str, Dict[str, DMProjection]] = {}
    for p in projs:
        by_cond.setdefault(p.condition, {})[p.substrate] = p
    for cond, subs in sorted(by_cond.items(), key=lambda kv: CONDITION_ORDER.get(kv[0], 99)):
        names = sorted(subs.keys(), key=lambda x: SUBSTRATE_ORDER.get(x, 99))
        for a, b in itertools.combinations(names, 2):
            pa, pb = subs[a], subs[b]
            n = min(pa.rung_stats.shape[0], pb.rung_stats.shape[0])
            rows.append({
                "condition": cond,
                "substrate_a": a,
                "substrate_b": b,
                "summary_corr": safe_corr(pa.summary, pb.summary),
                "summary_l2": l2(pa.summary, pb.summary),
                "projection_a": float(pa.summary[SUMMARY_PROJECTION_IDX]),
                "projection_b": float(pb.summary[SUMMARY_PROJECTION_IDX]),
                "projection_delta_abs": abs(float(pa.summary[SUMMARY_PROJECTION_IDX]) - float(pb.summary[SUMMARY_PROJECTION_IDX])),
                "rung_energy_corr": safe_corr(pa.rung_stats[:n, 6], pb.rung_stats[:n, 6]),
                "rung_specificity_corr": safe_corr(pa.rung_stats[:n, 8], pb.rung_stats[:n, 8]),
                "rung_yz_corr": safe_corr(pa.rung_stats[:n, 1], pb.rung_stats[:n, 1]),
                "rung_pi_cos2_corr": safe_corr(pa.rung_stats[:n, 12], pb.rung_stats[:n, 12]),
            })
    return rows


# =============================================================================
# DIMENSIONAL INVARIANCE / CORRUPTION CONTROLS
# =============================================================================

def W_from_rung(rung: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rs = np.asarray(rung, dtype=np.float64)
    W = rs[:, :4].copy()
    base = rs[:, 14].copy()
    off = rs[:, 15].copy()
    total = rs[:, 16].copy()
    return W, base, off, total


def rung_from_W(W: np.ndarray, base: np.ndarray, off: np.ndarray, total: np.ndarray, active: bool = True) -> np.ndarray:
    rows = []
    count = 1.0 if active else 0.0
    for i in range(W.shape[0]):
        rows.append(write_rung_from_witnesses(float(W[i, 0]), float(W[i, 1]), float(W[i, 2]), float(W[i, 3]), float(base[i]), float(off[i]), float(total[i]), count))
    return np.vstack(rows)


def orientation_score(W: np.ndarray, base: np.ndarray, off: np.ndarray, total: np.ndarray, orientation: Tuple[str, int, int, int, int]) -> Tuple[float, np.ndarray]:
    _name, p, ret_idx, ca, cb = orientation
    C = np.zeros_like(W)
    C[:, 0] = W[:, ca]
    C[:, 1] = W[:, p]
    C[:, 2] = W[:, ret_idx]
    C[:, 3] = W[:, cb]
    rung = rung_from_W(C, base, off, total, active=True)
    summary = cpu_summary_from_rung(rung)
    return float(summary[SUMMARY_PROJECTION_IDX]), summary


def best_dimensional_score(W: np.ndarray, base: np.ndarray, off: np.ndarray, total: np.ndarray) -> Tuple[float, str, np.ndarray]:
    best_score = -1e30
    best_name = ""
    best_summary = np.zeros((N_SUMMARY_METRICS,), dtype=np.float64)
    for orient in ORIENTATIONS:
        score, summary = orientation_score(W, base, off, total, orient)
        if score > best_score:
            best_score = score
            best_name = orient[0]
            best_summary = summary
    return float(best_score), best_name, best_summary


def apply_fault(W: np.ndarray, fault: str, rng: np.random.Generator) -> np.ndarray:
    X = np.asarray(W, dtype=np.float64).copy()
    n = X.shape[0]
    if n <= 0:
        return X
    if fault == "reciprocal_break":
        X[:, 2] = np.roll(X[:, 2], 1)
    elif fault == "cross_rung_delay_scramble":
        perm = rng.permutation(n)
        X = X[perm, :]
    elif fault == "same_label_wrong_delay":
        for c in range(4):
            X[:, c] = np.roll(X[:, c], c + 1)
    elif fault == "non_equivalence_channel_corruption":
        for r in range(n):
            perm = rng.permutation(4)
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=4)
            X[r, :] = X[r, perm] * signs
    elif fault == "independent_bit_shuffle_model":
        X *= rng.uniform(0.02, 0.20)
        X += rng.normal(0.0, 0.002, size=X.shape)
    else:
        raise ValueError(fault)
    return X


def run_invariance_controls(projs: List[DMProjection], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    allowed_rows: List[Dict[str, Any]] = []
    forbidden_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed) + 2300)
    for p in projs:
        W, base, off, total = W_from_rung(p.rung_stats)
        obs_score, obs_desc, _ = best_dimensional_score(W, base, off, total)
        canonical = float(p.summary[SUMMARY_PROJECTION_IDX])
        for name, perm in ALLOWED_TRANSFORMS.items():
            W2 = W[:, perm]
            score, desc, _ = best_dimensional_score(W2, base, off, total)
            allowed_rows.append({
                "substrate": p.substrate,
                "condition": p.condition,
                "transform": name,
                "observed_dim_score": obs_score,
                "observed_best_description": obs_desc,
                "transformed_dim_score": score,
                "transformed_best_description": desc,
                "retention_vs_observed": score / obs_score if abs(obs_score) > 1e-12 else float("nan"),
                "canonical_projection_score": canonical,
            })
        for fault in FAULTS:
            W2 = apply_fault(W, fault, rng)
            score, desc, _ = best_dimensional_score(W2, base, off, total)
            forbidden_rows.append({
                "substrate": p.substrate,
                "condition": p.condition,
                "fault": fault,
                "observed_dim_score": obs_score,
                "faulted_dim_score": score,
                "faulted_best_description": desc,
                "retention_vs_observed": score / obs_score if abs(obs_score) > 1e-12 else float("nan"),
                "expectation": "active should weaken/collapse; null may not move meaningfully",
            })
    return allowed_rows, forbidden_rows


def run_corruption_boundary(projs: List[DMProjection], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(int(args.seed) + 2400)
    trial_rows: List[Dict[str, Any]] = []
    depth_rows: List[Dict[str, Any]] = []
    trials = int(args.boundary_trials)
    max_depth = int(args.boundary_max_depth)
    for p in projs:
        W, base, off, total = W_from_rung(p.rung_stats)
        obs_score, obs_desc, _ = best_dimensional_score(W, base, off, total)
        for depth in range(max_depth + 1):
            vals = []
            if depth == 0:
                vals = [1.0]
                trial_rows.append({
                    "substrate": p.substrate, "condition": p.condition, "depth": 0, "trial": 0,
                    "faults": "clean", "observed_score": obs_score, "faulted_score": obs_score,
                    "retention": 1.0, "best_description": obs_desc,
                })
            else:
                for t in range(trials):
                    X = W.copy()
                    chosen = list(rng.choice(np.asarray(FAULTS, dtype=object), size=depth, replace=False if depth <= len(FAULTS) else True))
                    for fault in chosen:
                        X = apply_fault(X, str(fault), rng)
                    score, desc, _ = best_dimensional_score(X, base, off, total)
                    retention = score / obs_score if abs(obs_score) > 1e-12 else float("nan")
                    vals.append(retention)
                    if args.save_boundary_trials:
                        trial_rows.append({
                            "substrate": p.substrate, "condition": p.condition, "depth": depth, "trial": t,
                            "faults": "+".join(map(str, chosen)), "observed_score": obs_score,
                            "faulted_score": score, "retention": retention, "best_description": desc,
                        })
            arr = np.asarray(vals, dtype=np.float64)
            depth_rows.append({
                "substrate": p.substrate,
                "condition": p.condition,
                "depth": depth,
                "trials": len(vals),
                "observed_score": obs_score,
                "retention_mean": safe_nanmean(arr),
                "retention_median": safe_nanmedian(arr),
                "retention_p10": safe_nanpercentile(arr, 10),
                "retention_p90": safe_nanpercentile(arr, 90),
                "collapse_fraction": safe_fraction(np.where(np.isfinite(arr), arr < float(args.boundary_collapse_threshold), np.nan)),
                "preserve_fraction": safe_fraction(np.where(np.isfinite(arr), arr >= float(args.boundary_preserve_threshold), np.nan)),
                "collapse_threshold": float(args.boundary_collapse_threshold),
                "preserve_threshold": float(args.boundary_preserve_threshold),
            })
    return depth_rows, trial_rows


# =============================================================================
# GEO VALIDATION
# =============================================================================

def run_geo_validation(projs: List[DMProjection], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, np.ndarray]]:
    validation_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    artifacts: Dict[str, np.ndarray] = {}
    geo_by_cond = {p.condition: p for p in projs if p.substrate == "geo"}
    core_idx = [SUMMARY_METRICS.index(k) for k in CORE_SUMMARY_FIELDS]
    aux_idx = [SUMMARY_METRICS.index(k) for k in AUX_SUMMARY_FIELDS]

    for cond in ["null", "base_only", "offset_on"]:
        if cond not in geo_by_cond:
            continue
        gpu = geo_by_cond[cond]
        cpu_rung, cpu_sum = cpu_geo_exact(cond, args)
        gpu_sum = np.asarray(gpu.summary, dtype=np.float64)
        artifacts[f"geo_cpu_{cond}_summary"] = cpu_sum.astype(np.float64)
        artifacts[f"geo_cpu_{cond}_rung"] = cpu_rung.astype(np.float64)
        core_delta = float(np.max(np.abs(cpu_sum[core_idx] - gpu_sum[core_idx]))) if core_idx else 0.0
        aux_delta = float(np.max(np.abs(cpu_sum[aux_idx] - gpu_sum[aux_idx]))) if aux_idx else 0.0
        projection_delta = float(gpu_sum[SUMMARY_PROJECTION_IDX] - cpu_sum[SUMMARY_PROJECTION_IDX])
        pi_delta = float(gpu_sum[9] - cpu_sum[9])
        for i, name in enumerate(SUMMARY_METRICS):
            delta_rows.append({
                "condition": cond,
                "metric": name,
                "cpu": float(cpu_sum[i]),
                "gpu": float(gpu_sum[i]),
                "delta": float(gpu_sum[i] - cpu_sum[i]),
                "abs_delta": float(abs(gpu_sum[i] - cpu_sum[i])),
                "class": "core" if name in CORE_SUMMARY_FIELDS else "aux",
            })
        validation_rows.extend([
            {"check": f"{cond}_cpu_gpu_core_summary_agreement", "observed": core_delta, "threshold": args.validation_tol, "passed": int(core_delta <= args.validation_tol), "backend": "cpu_vs_cuda"},
            {"check": f"{cond}_cpu_gpu_aux_summary_diagnostic", "observed": aux_delta, "threshold": args.aux_validation_tol, "passed": int(aux_delta <= args.aux_validation_tol), "backend": "cpu_vs_cuda"},
            {"check": f"{cond}_projection_delta", "observed": abs(projection_delta), "threshold": args.validation_tol, "passed": int(abs(projection_delta) <= args.validation_tol), "backend": "cpu_vs_cuda"},
            {"check": f"{cond}_pi_score_delta", "observed": abs(pi_delta), "threshold": args.validation_tol, "passed": int(abs(pi_delta) <= args.validation_tol), "backend": "cpu_vs_cuda"},
        ])
        if cond == "null":
            validation_rows.append({"check": "geo_null_is_zero_manifold", "observed": float(np.max(np.abs(cpu_sum))), "threshold": args.validation_tol, "passed": int(float(np.max(np.abs(cpu_sum))) <= args.validation_tol), "backend": "cpu_float64"})
        else:
            validation_rows.append({"check": f"geo_{cond}_projection_positive", "observed": float(cpu_sum[SUMMARY_PROJECTION_IDX]), "threshold": args.active_projection_threshold, "passed": int(float(cpu_sum[SUMMARY_PROJECTION_IDX]) > args.active_projection_threshold), "backend": "cpu_float64"})
            validation_rows.append({"check": f"geo_{cond}_pi_score_exact", "observed": float(cpu_sum[9]), "threshold": args.pi_score_threshold, "passed": int(float(cpu_sum[9]) >= args.pi_score_threshold), "backend": "cpu_float64"})
    return validation_rows, delta_rows, artifacts


# =============================================================================
# CLAIM REPORT
# =============================================================================

def make_claim_report(out_dir: Path, summary_rows: List[Dict[str, Any]], sep_rows: List[Dict[str, Any]], control_rows: List[Dict[str, Any]], allowed_rows: List[Dict[str, Any]], forbidden_rows: List[Dict[str, Any]], boundary_rows: List[Dict[str, Any]], validation_rows: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    failed_validation = [r for r in validation_rows if int(r.get("passed", 0)) != 1 and "aux" not in str(r.get("check", ""))]
    active_sep = [r for r in sep_rows if r["condition_a"] == "null" and r["condition_b"] in ("base_only", "offset_on")]
    active_sep_pass = all(float(r["delta_projection"]) > float(args.active_projection_threshold) for r in active_sep) if active_sep else False
    record_ctrl = [r for r in control_rows if r["condition"] != "null"]
    ctrl_pass = any(float(r.get("projection_drop_fraction", 0.0)) >= float(args.control_drop_threshold) for r in record_ctrl) if record_ctrl else False
    allowed_active = [r for r in allowed_rows if r["condition"] != "null" and r["transform"] != "identity"]
    allowed_pass = bool(allowed_active) and safe_nanmedian([float(r["retention_vs_observed"]) for r in allowed_active]) >= float(args.allowed_retention_threshold)
    forbidden_active = [r for r in forbidden_rows if r["condition"] != "null"]
    forbidden_weaken = bool(forbidden_active) and safe_nanmedian([float(r["retention_vs_observed"]) for r in forbidden_active]) < float(args.forbidden_retention_threshold)
    boundary_active = [r for r in boundary_rows if r["condition"] != "null" and int(r["depth"]) >= 2]
    boundary_pass = bool(boundary_active) and any(float(r["retention_median"]) < float(args.boundary_collapse_threshold) for r in boundary_active)

    lines = []
    lines.append("# D_M Final Capstone Benchmark Report")
    lines.append("")
    lines.append("## Bounded claim")
    lines.append("")
    lines.append("D_M projects a YZ-primary / ZY-reciprocal dimensional witness manifold across qproj, gproj, and exact GEO substrates; active base-delay / offset manifolds separate from null; same-shot pairing, reciprocal structure, and delay order are load-bearing; compound corruptions cross a measurable collapse boundary.")
    lines.append("")
    lines.append("## Claim checks")
    lines.append("")
    lines.append("| Check | Passed | Reading |")
    lines.append("|---|---:|---|")
    lines.append(f"| GEO exact validation | {int(not failed_validation)} | {len(failed_validation)} non-aux validation failure(s) |")
    lines.append(f"| Active-vs-null separation | {int(active_sep_pass)} | projection delta over threshold for available substrates |")
    lines.append(f"| Record control collapse | {int(ctrl_pass)} | independent-bit shuffle drops active qproj/gproj projection |")
    lines.append(f"| Allowed channel re-description retention | {int(allowed_pass)} | median allowed active retention >= {args.allowed_retention_threshold} |")
    lines.append(f"| Forbidden single-fault weakening | {int(forbidden_weaken)} | median forbidden active retention < {args.forbidden_retention_threshold} |")
    lines.append(f"| Compound corruption boundary | {int(boundary_pass)} | median retention crosses below {args.boundary_collapse_threshold} at depth >= 2 |")
    lines.append("")
    lines.append("## Projection summary")
    lines.append("")
    lines.append("| Substrate | Condition | Projection | E_mean | S_mean | π score | YZ+ | ms/rep |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(summary_rows, key=lambda x: (SUBSTRATE_ORDER.get(x["substrate"], 99), CONDITION_ORDER.get(x["condition"], 99))):
        ms = r.get("projection_per_rep_ms", "")
        ms_s = f"{float(ms):.6f}" if isinstance(ms, (int, float)) or str(ms).replace('.', '', 1).isdigit() else ""
        lines.append(f"| `{r['substrate']}` | `{r['condition']}` | {float(r['projection_score']):.9f} | {float(r['yzzy_energy_mean']):.9f} | {float(r['specificity_mean']):.9f} | {float(r['pi_periodic_score']):.9f} | {float(r['yz_pos_frac']):.3f} | {ms_s} |")
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    for item in [
        "D_M does not reconstruct density matrices.",
        "D_M does not certify device-independent Bell nonlocality.",
        "D_M does not prove prepared Bell states.",
        "D_M is not a QPU speedup or quantum advantage claim.",
        "GPROJ is not an IBM hardware simulator.",
        "GEO is a closed-form classical reference, not a hardware simulator.",
        "GPT-2 is not a D_M input in the final claim.",
    ]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for name in [
        "verify_projection_summary.csv",
        "verify_rung_projection.csv",
        "verify_condition_separation.csv",
        "verify_substrate_agreement.csv",
        "verify_control_collapse.csv",
        "dimensional_invariance_allowed.csv",
        "dimensional_invariance_forbidden.csv",
        "corruption_boundary_summary.csv",
        "geo_exact_validation.csv",
        "geo_exact_summary_delta_by_metric.csv",
        "result.json",
        "artifacts.npz",
    ]:
        lines.append(f"- `{name}`")
    report = "\n".join(lines) + "\n"
    (out_dir / "final_claim_report.md").write_text(report, encoding="utf-8")
    return report


# =============================================================================
# VERIFY RUN
# =============================================================================

def path_from_arg(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def maybe_project_record(label: str, path_value: Optional[str], substrate: str, condition: str, ctx: DMCudaContext, args: argparse.Namespace) -> Optional[DMProjection]:
    p = path_from_arg(path_value)
    if p is None or not p.exists():
        if args.require_all_bases:
            raise FileNotFoundError(f"Missing required {label}: {p}")
        print(f"[{substrate.upper()}][skip] {condition}: {p if p else '(none)'}")
        return None
    print(f"[{substrate.upper()}] {condition}: {p}")
    return load_and_project(p, substrate, condition, ctx, args)


def run_verify(args: argparse.Namespace, ctx: DMCudaContext, out_dir: Path) -> Tuple[List[DMProjection], Dict[str, Any], Dict[str, np.ndarray]]:
    section("D_M VERIFY — QPROJ / GPROJ / EXACT GEO")
    projs: List[DMProjection] = []
    inputs = {
        "qproj": {"null": args.qproj_null, "base_only": args.qproj_base, "offset_on": args.qproj_offset},
        "gproj": {"null": args.gproj_null, "base_only": args.gproj_base, "offset_on": args.gproj_offset},
    }
    if not args.skip_qproj:
        for cond, path in inputs["qproj"].items():
            p = maybe_project_record(f"qproj_{cond}", path, "qproj", cond, ctx, args)
            if p is not None:
                projs.append(p)
    if not args.skip_gproj:
        for cond, path in inputs["gproj"].items():
            p = maybe_project_record(f"gproj_{cond}", path, "gproj", cond, ctx, args)
            if p is not None:
                projs.append(p)
    if not args.skip_geo:
        for cond in ["null", "base_only", "offset_on"]:
            print(f"[GEO] {cond}: exact closed-form reference")
            projs.append(project_geo_exact_gpu(cond, ctx, args))

    summary_rows = [projection_summary_row(p) for p in projs]
    rung_all = [row for p in projs for row in rung_rows(p)]
    control_rows = [r for r in (control_row(p) for p in projs) if r is not None]
    sep_rows = condition_separation(projs)
    agree_rows = substrate_agreement(projs)
    allowed_rows, forbidden_rows = run_invariance_controls(projs, args)
    boundary_rows, boundary_trial_rows = run_corruption_boundary(projs, args)
    geo_validation_rows, geo_delta_rows, geo_artifacts = run_geo_validation(projs, args)

    print()
    print("=" * 112)
    print("  D_M FINAL VERIFY SUMMARY")
    print("=" * 112)
    print(f"  {'substrate':<8} {'condition':<10} {'projection':>11} {'E_mean':>11} {'S_mean':>11} {'pi':>8} {'YZ+':>5} {'ms/rep':>10}")
    print("  " + "-" * 108)
    for row in sorted(summary_rows, key=lambda r: (SUBSTRATE_ORDER.get(r["substrate"], 99), CONDITION_ORDER.get(r["condition"], 99))):
        ms = row.get("projection_per_rep_ms", "")
        ms_s = f"{float(ms):.4f}" if isinstance(ms, (int, float)) else ""
        print(f"  {row['substrate']:<8} {row['condition']:<10} {float(row['projection_score']):>+11.6f} {float(row['yzzy_energy_mean']):>11.6f} {float(row['specificity_mean']):>+11.6f} {float(row['pi_periodic_score']):>8.4f} {float(row['yz_pos_frac']):>5.2f} {ms_s:>10}")

    if control_rows:
        print()
        print("  RECORD CONTROL COLLAPSE")
        print("  " + "-" * 108)
        for r in control_rows:
            print(f"  {r['substrate']:<8} {r['condition']:<10} {r['projection_real']:.6f} -> {r['projection_control']:.6f} drop={r['projection_drop_fraction']:.2%}")

    summary_fields = [
        "substrate", "condition", "condition_label", "source", "backend", "job_id",
        "tiles", "shots", "n_rungs", "timing_mode", "projection_per_rep_ms",
        "records_per_sec_projection", "records_semantics",
    ] + SUMMARY_METRICS + ["pi_witness_strength"]
    write_csv(out_dir / "verify_projection_summary.csv", summary_rows, summary_fields)
    write_csv(out_dir / "verify_rung_projection.csv", rung_all, ["substrate", "condition", "rung", "source"] + RUNG_METRICS + ["pi_phase_degrees_mod180"])
    write_csv(out_dir / "verify_condition_separation.csv", sep_rows, ["substrate", "condition_a", "condition_b", "delta_projection", "delta_energy", "delta_specificity", "delta_pi_score", "delta_piW", "summary_l2", "summary_corr"])
    write_csv(out_dir / "verify_substrate_agreement.csv", agree_rows, ["condition", "substrate_a", "substrate_b", "summary_corr", "summary_l2", "projection_a", "projection_b", "projection_delta_abs", "rung_energy_corr", "rung_specificity_corr", "rung_yz_corr", "rung_pi_cos2_corr"])
    write_csv(out_dir / "verify_control_collapse.csv", control_rows, ["substrate", "condition", "control", "projection_real", "projection_control", "projection_delta", "projection_drop_fraction", "energy_real", "energy_control", "specificity_real", "specificity_control", "piW_real", "piW_control"])
    write_csv(out_dir / "dimensional_invariance_allowed.csv", allowed_rows, ["substrate", "condition", "transform", "observed_dim_score", "observed_best_description", "transformed_dim_score", "transformed_best_description", "retention_vs_observed", "canonical_projection_score"])
    write_csv(out_dir / "dimensional_invariance_forbidden.csv", forbidden_rows, ["substrate", "condition", "fault", "observed_dim_score", "faulted_dim_score", "faulted_best_description", "retention_vs_observed", "expectation"])
    write_csv(out_dir / "corruption_boundary_summary.csv", boundary_rows, ["substrate", "condition", "depth", "trials", "observed_score", "retention_mean", "retention_median", "retention_p10", "retention_p90", "collapse_fraction", "preserve_fraction", "collapse_threshold", "preserve_threshold"])
    if boundary_trial_rows:
        write_csv(out_dir / "corruption_boundary_trials.csv", boundary_trial_rows, ["substrate", "condition", "depth", "trial", "faults", "observed_score", "faulted_score", "retention", "best_description"])
    write_csv(out_dir / "geo_exact_validation.csv", geo_validation_rows, ["check", "observed", "threshold", "passed", "backend"])
    write_csv(out_dir / "geo_exact_summary_delta_by_metric.csv", geo_delta_rows, ["condition", "metric", "cpu", "gpu", "delta", "abs_delta", "class"])

    artifacts: Dict[str, np.ndarray] = dict(geo_artifacts)
    for p in projs:
        key = f"{p.substrate}_{p.condition}"
        artifacts[f"{key}_summary"] = np.asarray(p.summary, dtype=np.float32)
        artifacts[f"{key}_rung_stats"] = np.asarray(p.rung_stats, dtype=np.float32)
        if p.control_summary is not None:
            artifacts[f"{key}_control_summary"] = np.asarray(p.control_summary, dtype=np.float32)

    if not args.no_plots and HAVE_MPL and summary_rows:
        plot_projection_summary(summary_rows, out_dir / "verify_projection_score_by_condition.png")

    meta = {
        "projection_summary": summary_rows,
        "condition_separation": sep_rows,
        "substrate_agreement": agree_rows,
        "control_collapse": control_rows,
        "allowed_invariance": allowed_rows,
        "forbidden_faults": forbidden_rows,
        "corruption_boundary": boundary_rows,
        "geo_validation": geo_validation_rows,
    }
    make_claim_report(out_dir, summary_rows, sep_rows, control_rows, allowed_rows, forbidden_rows, boundary_rows, geo_validation_rows, args)
    return projs, meta, artifacts


def plot_projection_summary(summary_rows: List[Dict[str, Any]], out_path: Path) -> None:
    subs = sorted({r["substrate"] for r in summary_rows}, key=lambda x: SUBSTRATE_ORDER.get(x, 99))
    conds = sorted({r["condition"] for r in summary_rows}, key=lambda x: CONDITION_ORDER.get(x, 99))
    x = np.arange(len(conds), dtype=np.float64)
    width = 0.8 / max(1, len(subs))
    fig = plt.figure(figsize=(9, 5), dpi=150)
    ax = fig.add_subplot(111)
    for si, sub in enumerate(subs):
        vals = []
        for cond in conds:
            m = [r for r in summary_rows if r["substrate"] == sub and r["condition"] == cond]
            vals.append(float(m[0]["projection_score"]) if m else 0.0)
        ax.bar(x + (si - (len(subs) - 1) / 2) * width, vals, width=width, label=sub)
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel("projection_score")
    ax.set_title("D_M final projection score by condition/substrate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--qproj-null", default=str(DEFAULT_QPROJ_NULL))
    p.add_argument("--qproj-base", default=str(DEFAULT_QPROJ_BASE))
    p.add_argument("--qproj-offset", default=str(DEFAULT_QPROJ_OFFSET))
    p.add_argument("--gproj-null", default=str(DEFAULT_GPROJ_NULL))
    p.add_argument("--gproj-base", default=str(DEFAULT_GPROJ_BASE))
    p.add_argument("--gproj-offset", default=str(DEFAULT_GPROJ_OFFSET))
    p.add_argument("--skip-qproj", action="store_true")
    p.add_argument("--skip-gproj", action="store_true")
    p.add_argument("--skip-geo", action="store_true")
    p.add_argument("--require-all-bases", action="store_true")
    p.add_argument("--repair-metadata", action="store_true")
    p.add_argument("--kernel", default=str(KERNEL_PATH))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=20260607)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--geo-reps", type=int, default=500)
    p.add_argument("--cuda-threads", type=int, default=256)
    p.add_argument("--control-seed", type=int, default=12345)
    p.add_argument("--geo-energy-floor", type=float, default=0.125)
    p.add_argument("--geo-energy-scale", type=float, default=0.875)
    p.add_argument("--geo-energy-gamma", type=float, default=1.0)
    p.add_argument("--geo-weight-space", type=float, default=1.0)
    p.add_argument("--geo-weight-time", type=float, default=1.0)
    p.add_argument("--boundary-trials", type=int, default=500)
    p.add_argument("--boundary-max-depth", type=int, default=5)
    p.add_argument("--boundary-collapse-threshold", type=float, default=0.50)
    p.add_argument("--boundary-preserve-threshold", type=float, default=0.75)
    p.add_argument("--save-boundary-trials", action="store_true")
    p.add_argument("--validation-tol", type=float, default=5e-5)
    p.add_argument("--aux-validation-tol", type=float, default=0.25)
    p.add_argument("--active-projection-threshold", type=float, default=5e-5)
    p.add_argument("--pi-score-threshold", type=float, default=0.99995)
    p.add_argument("--control-drop-threshold", type=float, default=0.25)
    p.add_argument("--allowed-retention-threshold", type=float, default=0.75)
    p.add_argument("--forbidden-retention-threshold", type=float, default=0.95)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"d_m_final_capstone_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = compile_dm_cuda(Path(args.kernel))
    try:
        dev_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        dev_name = "unknown-cuda-gpu"

    section("D_M FINAL CAPSTONE BENCHMARK")
    print(f"  Analysis dir : {out_dir}")
    print(f"  Kernel       : {ctx.path}")
    print(f"  Device       : {dev_name}")
    print("  Claim        : YZ-primary / ZY-reciprocal dimensional witness manifold")
    print("  GEO          : Probe-25 exact closed-form reference")
    print("  Controls     : bit-shuffle, invariance, forbidden faults, corruption boundary")

    projs, verify_meta, artifacts = run_verify(args, ctx, out_dir)
    if artifacts:
        np.savez_compressed(out_dir / "artifacts.npz", **artifacts)

    result = {
        "schema": "ghost_oracle.dm.final_capstone_benchmark.v1",
        "created": now_tag(),
        "seconds": time.time() - t0,
        "device": dev_name,
        "kernel_path": str(ctx.path),
        "analysis_dir": str(out_dir),
        "config": vars(args),
        "verify": verify_meta,
        "bounded_claim": "D_M projects a YZ-primary / ZY-reciprocal dimensional witness manifold across qproj, gproj, and exact GEO substrates; active base-delay / offset manifolds separate from null; same-shot pairing, reciprocal structure, and delay order are load-bearing; compound corruptions cross a measurable collapse boundary.",
        "non_claims": [
            "D_M does not reconstruct density matrices.",
            "D_M does not certify device-independent Bell nonlocality.",
            "D_M does not prove prepared Bell states.",
            "D_M is not a QPU speedup or quantum advantage claim.",
            "GPROJ is not an IBM hardware simulator.",
            "GEO is a closed-form classical reference, not a hardware simulator.",
            "GPT-2 is not a D_M input in the final claim.",
        ],
    }
    write_json(out_dir / "result.json", result)

    section("DONE")
    print(f"  final report : {out_dir / 'final_claim_report.md'}")
    print(f"  result       : {out_dir / 'result.json'}")
    print(f"  artifacts    : {out_dir / 'artifacts.npz'}")
    print("  Done. Break it, fix it, document what happened.")


if __name__ == "__main__":
    main()
