#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
D_M NATIVE STAGGER PROBE
==============================================================================

Drop this file in:

    ghost_oracle/D_M/probes/d_m_native_stagger_probe.py

Purpose
-------
One focused sidequest probe:

    Does the qproj NULL base contain an ordered D_M-like phase/stagger trace
    even though the explicit delay ladder is disabled?

This probe does NOT use GPT-2.
This probe does NOT use softmax.
This probe does NOT project external data.
This probe only interrogates the D_M base files themselves.

Main hypothesis
---------------
The qproj null condition may not be a true structural null. It may be an
explicit-delay null while still preserving native QPU ordering from hardware
layout, scheduling, readout grouping, calibration heterogeneity, pulse timing,
or physical tile order.

What it tests
-------------
For a D_M base with pair[tile, shot, 2] records and tile metadata:

1. Witness structure
   - XY, YZ, ZY, YX connected correlators per rung.
   - YZ/ZY energy vs XY/YX comparison energy.
   - Specificity = YZ/ZY energy - comparison energy.

2. Phase ordering
   - D_M phase = atan2(-ZY, YZ) mod pi.
   - Unwrap phase over rung index.
   - Fit phase ~ rung and report slope, correlation, R^2.

3. Controls
   - Independent bit-shuffle within each tile.
   - Rung-permutation preserving witness labels.
   - Tile-order permutation preserving only global counts.

4. Hardware/order correlations
   - Correlate per-tile connected signal with tile index, rung index,
     witness index, physical qubit ids, qubit id gap, and calibration fields if
     available in the npz.

Outputs
-------
Creates:

    ghost_oracle/D_M/analysis/d_m_native_stagger_probe_<timestamp>/

with:

    native_stagger_summary.json
    rung_metrics.csv
    tile_metrics.csv
    control_stats.csv
    calibration_correlations.csv
    native_stagger_report.txt
    optional PNG plots if matplotlib is available

Usage
-----
From repo root:

    python ghost_oracle/D_M/probes/d_m_native_stagger_probe.py

With explicit qproj null path:

    python ghost_oracle/D_M/probes/d_m_native_stagger_probe.py ^
      --qproj-null ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_d8fm4ihvjngc73aq3ccg.npz

Compare extra bases if present:

    python ghost_oracle/D_M/probes/d_m_native_stagger_probe.py --compare-defaults

==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    plt = None
    HAVE_MPL = False


# =============================================================================
# PATHS / DEFAULTS
# =============================================================================

PROBE_DIR = Path(__file__).resolve().parent
D_M_DIR = PROBE_DIR.parent
DATA_DIR = D_M_DIR / "data"
ANALYSIS_DIR = PROBE_DIR / "analyze"

DEFAULT_QPROJ_NULL = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fm4ihvjngc73aq3ccg.npz"
DEFAULT_QPROJ_BASE = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8flk2jo3njc73f0g560.npz"
DEFAULT_QPROJ_OFFSET = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fl82bo3njc73f0fgd0.npz"

DEFAULT_GPROJ_NULL = DATA_DIR / "dm_gpu_data_null_4096shots_seed9031229662612491082.npz"
DEFAULT_GPROJ_BASE = DATA_DIR / "dm_gpu_data_base_delay_4096shots_seed2669559634056472362.npz"
DEFAULT_GPROJ_OFFSET = DATA_DIR / "dm_gpu_data_offset_deformed_4096shots_seed6727069190982977623.npz"

WITNESS_LABELS = ["XY", "YZ", "ZY", "YX"]
WITNESS_TO_INDEX = {x: i for i, x in enumerate(WITNESS_LABELS)}
EPS = 1e-12


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BaseData:
    name: str
    path: Path
    pair: np.ndarray
    tile_index: np.ndarray
    tile_rung: np.ndarray
    tile_witness: np.ndarray
    tile_base_delay: np.ndarray
    tile_offset: np.ndarray
    tile_total_delay: np.ndarray
    tile_q0: np.ndarray
    tile_q1: np.ndarray
    labels: List[str]
    backend: str
    job_id: str
    raw_keys: List[str]
    calibration: Dict[str, Any]


@dataclass
class Metrics:
    tile_rows: List[Dict[str, Any]]
    rung_rows: List[Dict[str, Any]]
    summary: Dict[str, Any]


# =============================================================================
# IO HELPERS
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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def scalar_str(x: Any) -> str:
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(x)


def decode_str_array(arr: Any) -> List[str]:
    a = np.asarray(arr)
    out: List[str] = []
    for x in a.reshape(-1):
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


def try_parse_json_value(x: Any) -> Any:
    try:
        if isinstance(x, np.ndarray):
            if x.shape == ():
                x = x.item()
            elif x.size == 1:
                x = x.reshape(-1)[0]
            else:
                return None
        if isinstance(x, bytes):
            x = x.decode("utf-8", errors="replace")
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return None
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                return json.loads(s)
    except Exception:
        return None
    return None


# =============================================================================
# MATH HELPERS
# =============================================================================

def safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n < 3:
        return 0.0
    x = x[:n]
    y = y[:n]
    if float(np.std(x)) < EPS or float(np.std(y)) < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def linear_fit_stats(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(xx.size, yy.size)
    if n < 2:
        return {"slope": 0.0, "intercept": 0.0, "r": 0.0, "r2": 0.0, "rmse": 0.0}
    xx = xx[:n]
    yy = yy[:n]
    A = np.vstack([xx, np.ones_like(xx)]).T
    slope, intercept = np.linalg.lstsq(A, yy, rcond=None)[0]
    pred = slope * xx + intercept
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else 0.0
    rmse = math.sqrt(ss_res / max(1, n))
    r = safe_corr(xx, yy)
    return {"slope": float(slope), "intercept": float(intercept), "r": float(r), "r2": float(r2), "rmse": float(rmse)}


def unwrap_mod_pi(phases: Sequence[float]) -> np.ndarray:
    """
    Unwrap phases that live on [0, pi).

    D_M phase is pi-periodic because R=-ZY / YZ direction is projective. To
    unwrap, double the phase, unwrap with 2pi periodicity, then divide by 2.
    """
    p = np.asarray(phases, dtype=np.float64)
    if p.size == 0:
        return p
    return np.unwrap(2.0 * p) / 2.0


def phase_distance_pi(a: float, b: float) -> float:
    d = abs(float(a) - float(b))
    d = min(d, math.pi - d)
    return d / math.pi


def bits_to_spins(bits: np.ndarray) -> np.ndarray:
    return 2.0 * bits.astype(np.float64) - 1.0


def connected_from_pair(pair_tile: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    pair_tile shape: (shots, 2), bits 0/1.

    Returns:
        connected, raw_corr, mean0, mean1, spin_agreement
    """
    p = np.asarray(pair_tile, dtype=np.uint8)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"Expected pair tile shape (shots, 2), got {p.shape}")
    s0 = bits_to_spins(p[:, 0])
    s1 = bits_to_spins(p[:, 1])
    raw = float(np.mean(s0 * s1))
    m0 = float(np.mean(s0))
    m1 = float(np.mean(s1))
    conn = float(raw - m0 * m1)
    agree = float(np.mean(p[:, 0] == p[:, 1]))
    return conn, raw, m0, m1, agree


def manifold_from_witnesses(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Witness columns: XY, YZ, ZY, YX.

    D_M orientation:
        YZ = primary Bell-witness channel.
        ZY = reciprocal/inverted side, represented as R=-ZY.
    """
    arr = np.asarray(w, dtype=np.float64)
    xy = arr[:, 0]
    yz = arr[:, 1]
    zy = arr[:, 2]
    yx = arr[:, 3]
    reciprocal = -zy
    yzzy_energy = np.sqrt(yz * yz + reciprocal * reciprocal)
    comparison_energy = np.sqrt(xy * xy + yx * yx)
    specificity = yzzy_energy - comparison_energy
    phase = np.mod(np.arctan2(reciprocal, yz), math.pi)
    return yzzy_energy, comparison_energy, specificity, phase


# =============================================================================
# BASE LOADING
# =============================================================================

def load_base(path: Path, name: str) -> BaseData:
    if not path.exists():
        raise FileNotFoundError(f"Missing base file: {path}")
    z = np.load(path, allow_pickle=True)
    files = list(z.files)

    if "pair" in files:
        pair = np.asarray(z["pair"], dtype=np.uint8)
    else:
        keys = sorted(
            [k for k in files if k.startswith("pair_tile")],
            key=lambda k: int(k.replace("pair_tile", "")),
        )
        if not keys:
            raise KeyError(f"{path} has no pair or pair_tile* arrays")
        pair = np.stack([np.asarray(z[k], dtype=np.uint8) for k in keys], axis=0)

    if pair.ndim != 3 or pair.shape[2] != 2:
        raise ValueError(f"pair must have shape (tiles, shots, 2), got {pair.shape}")

    tiles = int(pair.shape[0])

    def arr_i32(key: str, default: Optional[np.ndarray] = None) -> np.ndarray:
        if key in files:
            a = np.asarray(z[key], dtype=np.int32).reshape(-1)
            if a.size == tiles:
                return a
        if default is not None:
            return np.asarray(default, dtype=np.int32).reshape(-1)
        return np.zeros((tiles,), dtype=np.int32)

    tile_index = arr_i32("tile_indices", np.arange(tiles, dtype=np.int32))
    tile_rung = arr_i32("tile_rung_index", np.arange(tiles, dtype=np.int32) // 4)

    if "tile_witness_index" in files:
        tile_witness = arr_i32("tile_witness_index")
        labels = [WITNESS_LABELS[int(x)] if 0 <= int(x) < 4 else "?" for x in tile_witness]
    elif "tile_witness_label" in files:
        labels = decode_str_array(z["tile_witness_label"])
        labels = labels[:tiles]
        tile_witness = np.asarray([WITNESS_TO_INDEX.get(x, i % 4) for i, x in enumerate(labels)], dtype=np.int32)
    else:
        tile_witness = np.arange(tiles, dtype=np.int32) % 4
        labels = [WITNESS_LABELS[int(x)] for x in tile_witness]

    tile_base_delay = arr_i32("tile_base_delay_dt")
    tile_offset = arr_i32("tile_offset_dt")
    tile_total_delay = arr_i32("tile_total_delay_dt")
    tile_q0 = arr_i32("tile_physical_q0", arr_i32("physical_q0", np.full(tiles, -1, dtype=np.int32)))
    tile_q1 = arr_i32("tile_physical_q1", arr_i32("physical_q1", np.full(tiles, -1, dtype=np.int32)))

    # Some files use q0/q1-ish names or only tile_meta_json.
    if np.all(tile_q0 < 0) or np.all(tile_q1 < 0):
        for meta_key in ("tile_meta_json", "metadata_json", "meta_json"):
            if meta_key in files:
                parsed = try_parse_json_value(z[meta_key])
                if isinstance(parsed, list):
                    q0s, q1s = [], []
                    for m in parsed[:tiles]:
                        if isinstance(m, dict):
                            q0s.append(int(m.get("physical_q0", m.get("q0", -1))))
                            q1s.append(int(m.get("physical_q1", m.get("q1", -1))))
                    if len(q0s) == tiles and len(q1s) == tiles:
                        tile_q0 = np.asarray(q0s, dtype=np.int32)
                        tile_q1 = np.asarray(q1s, dtype=np.int32)
                        break

    backend = scalar_str(z["backend"]) if "backend" in files else ""
    job_id = scalar_str(z["job_id"]) if "job_id" in files else ""

    calibration: Dict[str, Any] = {}
    for key in files:
        low = key.lower()
        if "cal" in low or "backend" in low or "target" in low:
            parsed = try_parse_json_value(z[key])
            if isinstance(parsed, dict):
                calibration[key] = parsed

    return BaseData(
        name=name,
        path=path,
        pair=pair,
        tile_index=tile_index,
        tile_rung=tile_rung,
        tile_witness=tile_witness,
        tile_base_delay=tile_base_delay,
        tile_offset=tile_offset,
        tile_total_delay=tile_total_delay,
        tile_q0=tile_q0,
        tile_q1=tile_q1,
        labels=labels,
        backend=backend,
        job_id=job_id,
        raw_keys=files,
        calibration=calibration,
    )


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_metrics(base: BaseData, pair_override: Optional[np.ndarray] = None, rung_override: Optional[np.ndarray] = None) -> Metrics:
    pair = base.pair if pair_override is None else np.asarray(pair_override, dtype=np.uint8)
    tile_rung = base.tile_rung if rung_override is None else np.asarray(rung_override, dtype=np.int32)
    tiles = int(pair.shape[0])
    shots = int(pair.shape[1])
    n_rungs = int(np.max(tile_rung)) + 1 if tiles else 0

    tile_rows: List[Dict[str, Any]] = []
    witnesses = np.zeros((n_rungs, 4), dtype=np.float64)
    counts = np.zeros((n_rungs, 4), dtype=np.int32)

    for t in range(tiles):
        conn, raw, m0, m1, agree = connected_from_pair(pair[t])
        r = int(tile_rung[t])
        wi = int(base.tile_witness[t])
        label = WITNESS_LABELS[wi] if 0 <= wi < 4 else "?"

        if 0 <= r < n_rungs and 0 <= wi < 4:
            witnesses[r, wi] += conn
            counts[r, wi] += 1

        tile_rows.append({
            "base": base.name,
            "tile": int(base.tile_index[t]) if t < base.tile_index.size else t,
            "tile_array_index": t,
            "rung": r,
            "witness_index": wi,
            "witness": label,
            "physical_q0": int(base.tile_q0[t]) if t < base.tile_q0.size else -1,
            "physical_q1": int(base.tile_q1[t]) if t < base.tile_q1.size else -1,
            "physical_q_gap": int(abs(int(base.tile_q1[t]) - int(base.tile_q0[t]))) if t < base.tile_q0.size and t < base.tile_q1.size and base.tile_q0[t] >= 0 and base.tile_q1[t] >= 0 else None,
            "base_delay_dt": int(base.tile_base_delay[t]) if t < base.tile_base_delay.size else 0,
            "offset_dt": int(base.tile_offset[t]) if t < base.tile_offset.size else 0,
            "total_delay_dt": int(base.tile_total_delay[t]) if t < base.tile_total_delay.size else 0,
            "connected": conn,
            "raw_corr": raw,
            "mean_q0": m0,
            "mean_q1": m1,
            "spin_agreement": agree,
            "abs_connected": abs(conn),
            "shots": shots,
        })

    # Average if more than one tile contributed to a witness/rung cell.
    for r in range(n_rungs):
        for wi in range(4):
            if counts[r, wi] > 0:
                witnesses[r, wi] /= float(counts[r, wi])

    yzzy_energy, comp_energy, specificity, phase = manifold_from_witnesses(witnesses)
    phase_unwrapped = unwrap_mod_pi(phase)
    rung_x = np.arange(n_rungs, dtype=np.float64)
    phase_fit = linear_fit_stats(rung_x, phase_unwrapped / math.pi)
    energy_fit = linear_fit_stats(rung_x, yzzy_energy)
    spec_fit = linear_fit_stats(rung_x, specificity)

    rung_rows: List[Dict[str, Any]] = []
    for r in range(n_rungs):
        row = {
            "base": base.name,
            "rung": r,
            "XY": witnesses[r, 0],
            "YZ": witnesses[r, 1],
            "ZY": witnesses[r, 2],
            "YX": witnesses[r, 3],
            "YZ_primary": witnesses[r, 1],
            "ZY_reciprocal_R_neg_ZY": -witnesses[r, 2],
            "yzzy_energy": yzzy_energy[r],
            "comparison_energy": comp_energy[r],
            "specificity": specificity[r],
            "phase_rad": phase[r],
            "phase_pi": phase[r] / math.pi,
            "phase_unwrapped_pi": phase_unwrapped[r] / math.pi,
            "count_XY": int(counts[r, 0]),
            "count_YZ": int(counts[r, 1]),
            "count_ZY": int(counts[r, 2]),
            "count_YX": int(counts[r, 3]),
        }
        rung_rows.append(row)

    yzzy_mean = float(np.mean(yzzy_energy)) if n_rungs else 0.0
    comp_mean = float(np.mean(comp_energy)) if n_rungs else 0.0
    spec_mean = float(np.mean(specificity)) if n_rungs else 0.0
    dominance = float(yzzy_mean / max(comp_mean, EPS))

    summary = {
        "base": base.name,
        "path": str(base.path),
        "backend": base.backend,
        "job_id": base.job_id,
        "tiles": tiles,
        "shots": shots,
        "n_rungs": n_rungs,
        "max_base_delay_dt": int(np.max(base.tile_base_delay)) if base.tile_base_delay.size else 0,
        "max_offset_dt": int(np.max(base.tile_offset)) if base.tile_offset.size else 0,
        "max_total_delay_dt": int(np.max(base.tile_total_delay)) if base.tile_total_delay.size else 0,
        "yzzy_energy_mean": yzzy_mean,
        "comparison_energy_mean": comp_mean,
        "specificity_mean": spec_mean,
        "yzzy_to_comparison_ratio": dominance,
        "phase_unwrapped_slope_pi_per_rung": phase_fit["slope"],
        "phase_unwrapped_r": phase_fit["r"],
        "phase_unwrapped_r2": phase_fit["r2"],
        "phase_unwrapped_rmse_pi": phase_fit["rmse"],
        "energy_r": energy_fit["r"],
        "energy_r2": energy_fit["r2"],
        "specificity_r": spec_fit["r"],
        "specificity_r2": spec_fit["r2"],
        "rung_witnesses": witnesses,
        "phase_pi": phase / math.pi,
        "phase_unwrapped_pi": phase_unwrapped / math.pi,
        "counts": counts,
    }

    return Metrics(tile_rows=tile_rows, rung_rows=rung_rows, summary=summary)


# =============================================================================
# CONTROLS
# =============================================================================

def independent_bit_shuffle(pair: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Destroy same-shot pairing inside each tile while preserving both marginal
    bit streams for each tile.
    """
    p = np.asarray(pair, dtype=np.uint8).copy()
    for t in range(p.shape[0]):
        idx = rng.permutation(p.shape[1])
        p[t, :, 1] = p[t, idx, 1]
    return p


def rung_permutation(base: BaseData, rng: np.random.Generator, preserve_witness: bool = True) -> np.ndarray:
    """
    Return a shuffled rung assignment.

    preserve_witness=True keeps each witness label's tile count intact while
    scrambling which rung each same-witness tile belongs to. This is the most
    direct test of whether rung/order is doing work.
    """
    shuffled = np.asarray(base.tile_rung, dtype=np.int32).copy()
    if preserve_witness:
        for wi in range(4):
            idx = np.where(base.tile_witness == wi)[0]
            vals = shuffled[idx].copy()
            rng.shuffle(vals)
            shuffled[idx] = vals
    else:
        vals = shuffled.copy()
        rng.shuffle(vals)
        shuffled = vals
    return shuffled


def run_controls(base: BaseData, observed: Metrics, n_perm: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    obs_phase_r = float(observed.summary["phase_unwrapped_r"])
    obs_phase_abs = abs(obs_phase_r)
    obs_phase_r2 = float(observed.summary["phase_unwrapped_r2"])
    obs_spec = float(observed.summary["specificity_mean"])
    obs_dom = float(observed.summary["yzzy_to_comparison_ratio"])

    bit_phase_abs: List[float] = []
    bit_r2: List[float] = []
    bit_spec: List[float] = []
    bit_dom: List[float] = []

    rung_phase_abs: List[float] = []
    rung_r2: List[float] = []
    rung_spec: List[float] = []
    rung_dom: List[float] = []

    global_phase_abs: List[float] = []
    global_r2: List[float] = []
    global_spec: List[float] = []
    global_dom: List[float] = []

    for i in range(int(n_perm)):
        # Independent bit shuffle.
        bp = independent_bit_shuffle(base.pair, rng)
        bm = compute_metrics(base, pair_override=bp)
        bit_phase_abs.append(abs(float(bm.summary["phase_unwrapped_r"])))
        bit_r2.append(float(bm.summary["phase_unwrapped_r2"]))
        bit_spec.append(float(bm.summary["specificity_mean"]))
        bit_dom.append(float(bm.summary["yzzy_to_comparison_ratio"]))

        rows.append({
            "control": "independent_bit_shuffle",
            "iter": i,
            "phase_abs_r": bit_phase_abs[-1],
            "phase_r2": bit_r2[-1],
            "specificity_mean": bit_spec[-1],
            "yzzy_to_comparison_ratio": bit_dom[-1],
        })

        # Rung permutation preserving witness labels.
        rr = rung_permutation(base, rng, preserve_witness=True)
        rm = compute_metrics(base, rung_override=rr)
        rung_phase_abs.append(abs(float(rm.summary["phase_unwrapped_r"])))
        rung_r2.append(float(rm.summary["phase_unwrapped_r2"]))
        rung_spec.append(float(rm.summary["specificity_mean"]))
        rung_dom.append(float(rm.summary["yzzy_to_comparison_ratio"]))

        rows.append({
            "control": "rung_permutation_preserve_witness",
            "iter": i,
            "phase_abs_r": rung_phase_abs[-1],
            "phase_r2": rung_r2[-1],
            "specificity_mean": rung_spec[-1],
            "yzzy_to_comparison_ratio": rung_dom[-1],
        })

        # Full tile-order/rung permutation.
        gr = rung_permutation(base, rng, preserve_witness=False)
        gm = compute_metrics(base, rung_override=gr)
        global_phase_abs.append(abs(float(gm.summary["phase_unwrapped_r"])))
        global_r2.append(float(gm.summary["phase_unwrapped_r2"]))
        global_spec.append(float(gm.summary["specificity_mean"]))
        global_dom.append(float(gm.summary["yzzy_to_comparison_ratio"]))

        rows.append({
            "control": "rung_permutation_global",
            "iter": i,
            "phase_abs_r": global_phase_abs[-1],
            "phase_r2": global_r2[-1],
            "specificity_mean": global_spec[-1],
            "yzzy_to_comparison_ratio": global_dom[-1],
        })

    def summarize_control(name: str, vals_phase: List[float], vals_r2: List[float], vals_spec: List[float], vals_dom: List[float]) -> Dict[str, Any]:
        def arr(v: List[float]) -> np.ndarray:
            return np.asarray(v, dtype=np.float64)
        a_phase = arr(vals_phase)
        a_r2 = arr(vals_r2)
        a_spec = arr(vals_spec)
        a_dom = arr(vals_dom)
        return {
            "control": name,
            "n": int(len(vals_phase)),
            "phase_abs_r_mean": float(np.mean(a_phase)) if a_phase.size else 0.0,
            "phase_abs_r_median": float(np.median(a_phase)) if a_phase.size else 0.0,
            "phase_abs_r_p95": float(np.quantile(a_phase, 0.95)) if a_phase.size else 0.0,
            "phase_abs_r_p_value_upper": float((1 + np.sum(a_phase >= obs_phase_abs)) / (1 + max(1, a_phase.size))) if a_phase.size else 1.0,
            "phase_r2_mean": float(np.mean(a_r2)) if a_r2.size else 0.0,
            "phase_r2_p95": float(np.quantile(a_r2, 0.95)) if a_r2.size else 0.0,
            "phase_r2_p_value_upper": float((1 + np.sum(a_r2 >= obs_phase_r2)) / (1 + max(1, a_r2.size))) if a_r2.size else 1.0,
            "specificity_mean_control": float(np.mean(a_spec)) if a_spec.size else 0.0,
            "specificity_p95": float(np.quantile(a_spec, 0.95)) if a_spec.size else 0.0,
            "specificity_p_value_upper": float((1 + np.sum(a_spec >= obs_spec)) / (1 + max(1, a_spec.size))) if a_spec.size else 1.0,
            "dominance_mean_control": float(np.mean(a_dom)) if a_dom.size else 0.0,
            "dominance_p95": float(np.quantile(a_dom, 0.95)) if a_dom.size else 0.0,
            "dominance_p_value_upper": float((1 + np.sum(a_dom >= obs_dom)) / (1 + max(1, a_dom.size))) if a_dom.size else 1.0,
        }

    summary = {
        "observed": {
            "phase_abs_r": obs_phase_abs,
            "phase_r2": obs_phase_r2,
            "specificity_mean": obs_spec,
            "yzzy_to_comparison_ratio": obs_dom,
        },
        "controls": [
            summarize_control("independent_bit_shuffle", bit_phase_abs, bit_r2, bit_spec, bit_dom),
            summarize_control("rung_permutation_preserve_witness", rung_phase_abs, rung_r2, rung_spec, rung_dom),
            summarize_control("rung_permutation_global", global_phase_abs, global_r2, global_spec, global_dom),
        ],
    }

    return rows, summary


# =============================================================================
# ORDER / CALIBRATION CORRELATIONS
# =============================================================================

def collect_order_correlations(tile_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    y_fields = ["connected", "abs_connected", "raw_corr", "spin_agreement"]
    x_fields = [
        "tile_array_index", "tile", "rung", "witness_index", "physical_q0",
        "physical_q1", "physical_q_gap", "base_delay_dt", "offset_dt", "total_delay_dt",
        "mean_q0", "mean_q1",
    ]
    for yf in y_fields:
        y = []
        valid_rows = []
        for r in tile_rows:
            val = r.get(yf)
            if val is None:
                continue
            try:
                y.append(float(val))
                valid_rows.append(r)
            except Exception:
                pass
        if len(y) < 3:
            continue
        for xf in x_fields:
            x = []
            yy = []
            for rr, yv in zip(valid_rows, y):
                xv = rr.get(xf)
                if xv is None or xv == "":
                    continue
                try:
                    x.append(float(xv))
                    yy.append(float(yv))
                except Exception:
                    pass
            if len(x) >= 3:
                rows.append({
                    "y": yf,
                    "x": xf,
                    "n": len(x),
                    "r": safe_corr(x, yy),
                })

    rows.sort(key=lambda r: abs(float(r["r"])), reverse=True)
    return rows


def extract_calibration_features(base: BaseData, tile_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Best-effort calibration feature correlation extraction.

    QPU dumps vary by version, so this function is deliberately forgiving. If it
    cannot find useful calibration dictionaries, it returns an empty list.
    """
    if not base.calibration:
        return []

    # Merge nested calibration-like dicts into a searchable object.
    cal_root: Dict[str, Any] = {}
    for _, v in base.calibration.items():
        if isinstance(v, dict):
            cal_root.update(v)

    # Common structure from the D_M QPU generator:
    # cal["readout"][str(q)], cal["single_qubit"][str(q)], cal["idling"][str(q)] = {t1,t2}
    readout = cal_root.get("readout", {}) if isinstance(cal_root.get("readout", {}), dict) else {}
    single = cal_root.get("single_qubit", {}) if isinstance(cal_root.get("single_qubit", {}), dict) else {}
    idling = cal_root.get("idling", {}) if isinstance(cal_root.get("idling", {}), dict) else {}

    enriched: List[Dict[str, Any]] = []
    for r in tile_rows:
        q0 = r.get("physical_q0")
        q1 = r.get("physical_q1")
        if q0 is None or q1 is None or int(q0) < 0 or int(q1) < 0:
            continue
        q0s, q1s = str(int(q0)), str(int(q1))

        def fnum(x: Any) -> Optional[float]:
            try:
                if x is None:
                    return None
                return float(x)
            except Exception:
                return None

        ro0 = fnum(readout.get(q0s))
        ro1 = fnum(readout.get(q1s))
        sg0 = fnum(single.get(q0s))
        sg1 = fnum(single.get(q1s))

        t10 = t20 = t11 = t21 = None
        if isinstance(idling.get(q0s), dict):
            t10 = fnum(idling[q0s].get("t1"))
            t20 = fnum(idling[q0s].get("t2"))
        if isinstance(idling.get(q1s), dict):
            t11 = fnum(idling[q1s].get("t1"))
            t21 = fnum(idling[q1s].get("t2"))

        rr = dict(r)
        rr.update({
            "readout_q0": ro0,
            "readout_q1": ro1,
            "readout_mean": np.nanmean([x for x in [ro0, ro1] if x is not None]) if any(x is not None for x in [ro0, ro1]) else None,
            "readout_abs_diff": abs(ro0 - ro1) if ro0 is not None and ro1 is not None else None,
            "single_q0": sg0,
            "single_q1": sg1,
            "single_mean": np.nanmean([x for x in [sg0, sg1] if x is not None]) if any(x is not None for x in [sg0, sg1]) else None,
            "single_abs_diff": abs(sg0 - sg1) if sg0 is not None and sg1 is not None else None,
            "t1_q0": t10,
            "t1_q1": t11,
            "t1_mean": np.nanmean([x for x in [t10, t11] if x is not None]) if any(x is not None for x in [t10, t11]) else None,
            "t1_abs_diff": abs(t10 - t11) if t10 is not None and t11 is not None else None,
            "t2_q0": t20,
            "t2_q1": t21,
            "t2_mean": np.nanmean([x for x in [t20, t21] if x is not None]) if any(x is not None for x in [t20, t21]) else None,
            "t2_abs_diff": abs(t20 - t21) if t20 is not None and t21 is not None else None,
        })
        enriched.append(rr)

    if not enriched:
        return []

    y_fields = ["connected", "abs_connected", "raw_corr", "spin_agreement"]
    x_fields = [
        "readout_mean", "readout_abs_diff", "single_mean", "single_abs_diff",
        "t1_mean", "t1_abs_diff", "t2_mean", "t2_abs_diff",
    ]
    out: List[Dict[str, Any]] = []
    for yf in y_fields:
        for xf in x_fields:
            x, y = [], []
            for r in enriched:
                xv = r.get(xf)
                yv = r.get(yf)
                if xv is None or yv is None:
                    continue
                try:
                    if math.isnan(float(xv)):
                        continue
                    x.append(float(xv))
                    y.append(float(yv))
                except Exception:
                    continue
            if len(x) >= 3:
                out.append({"y": yf, "x": xf, "n": len(x), "r": safe_corr(x, y)})
    out.sort(key=lambda r: abs(float(r["r"])), reverse=True)
    return out


# =============================================================================
# PLOTS / REPORT
# =============================================================================

def maybe_plot(out_dir: Path, metrics: Metrics, control_rows: List[Dict[str, Any]]) -> None:
    if not HAVE_MPL:
        return

    rung = np.asarray([r["rung"] for r in metrics.rung_rows], dtype=np.float64)
    phase = np.asarray([r["phase_unwrapped_pi"] for r in metrics.rung_rows], dtype=np.float64)
    energy = np.asarray([r["yzzy_energy"] for r in metrics.rung_rows], dtype=np.float64)
    comp = np.asarray([r["comparison_energy"] for r in metrics.rung_rows], dtype=np.float64)
    spec = np.asarray([r["specificity"] for r in metrics.rung_rows], dtype=np.float64)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(rung, phase, marker="o")
    plt.xlabel("rung index")
    plt.ylabel("unwrapped phase / pi")
    plt.title("D_M qproj null phase trace")
    plt.tight_layout()
    fig.savefig(out_dir / "phase_trace.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(rung, energy, marker="o", label="YZ/ZY energy")
    plt.plot(rung, comp, marker="o", label="XY/YX comparison")
    plt.plot(rung, spec, marker="o", label="specificity")
    plt.xlabel("rung index")
    plt.ylabel("connected energy")
    plt.title("D_M qproj null energy/specificity")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "energy_specificity_trace.png", dpi=160)
    plt.close(fig)

    # Histograms for phase abs correlation controls.
    for cname in ["independent_bit_shuffle", "rung_permutation_preserve_witness", "rung_permutation_global"]:
        vals = [float(r["phase_abs_r"]) for r in control_rows if r.get("control") == cname]
        if not vals:
            continue
        fig = plt.figure(figsize=(8, 5))
        plt.hist(vals, bins=30)
        plt.xlabel("abs phase-order correlation")
        plt.ylabel("count")
        plt.title(cname)
        plt.tight_layout()
        fig.savefig(out_dir / f"control_{cname}_phase_abs_r.png", dpi=160)
        plt.close(fig)


def write_report(path: Path, base: BaseData, metrics: Metrics, controls: Dict[str, Any], order_corrs: List[Dict[str, Any]], cal_corrs: List[Dict[str, Any]]) -> None:
    s = metrics.summary
    lines: List[str] = []
    lines.append("D_M NATIVE STAGGER PROBE REPORT")
    lines.append("=" * 80)
    lines.append(f"base          : {base.name}")
    lines.append(f"path          : {base.path}")
    lines.append(f"backend       : {base.backend}")
    lines.append(f"job_id        : {base.job_id}")
    lines.append(f"tiles/shots   : {s['tiles']} / {s['shots']}")
    lines.append(f"rungs         : {s['n_rungs']}")
    lines.append(f"max delays    : base={s['max_base_delay_dt']} offset={s['max_offset_dt']} total={s['max_total_delay_dt']}")
    lines.append("")
    lines.append("CORE OBSERVED METRICS")
    lines.append("-" * 80)
    lines.append(f"YZ/ZY energy mean       : {s['yzzy_energy_mean']:+.8f}")
    lines.append(f"XY/YX comparison mean   : {s['comparison_energy_mean']:+.8f}")
    lines.append(f"specificity mean        : {s['specificity_mean']:+.8f}")
    lines.append(f"YZ/ZY : comparison      : {s['yzzy_to_comparison_ratio']:+.4f}x")
    lines.append(f"phase slope             : {s['phase_unwrapped_slope_pi_per_rung']:+.6f} pi/rung")
    lines.append(f"phase order r           : {s['phase_unwrapped_r']:+.6f}")
    lines.append(f"phase order R^2         : {s['phase_unwrapped_r2']:+.6f}")
    lines.append("")
    lines.append("RUNG TRACE")
    lines.append("-" * 80)
    lines.append("rung | XY        YZ        ZY        YX        E_yzzy    E_comp    spec      phase_pi  unwrap_pi")
    for r in metrics.rung_rows:
        lines.append(
            f"{int(r['rung']):4d} | "
            f"{r['XY']:+.6f} {r['YZ']:+.6f} {r['ZY']:+.6f} {r['YX']:+.6f} "
            f"{r['yzzy_energy']:+.6f} {r['comparison_energy']:+.6f} {r['specificity']:+.6f} "
            f"{r['phase_pi']:+.3f} {r['phase_unwrapped_pi']:+.3f}"
        )
    lines.append("")
    lines.append("CONTROL SUMMARY")
    lines.append("-" * 80)
    for c in controls.get("controls", []):
        lines.append(
            f"{c['control']}: "
            f"phase_abs_r mean={c['phase_abs_r_mean']:.4f}, p95={c['phase_abs_r_p95']:.4f}, "
            f"p_upper={c['phase_abs_r_p_value_upper']:.4f}; "
            f"R2 mean={c['phase_r2_mean']:.4f}, p_upper={c['phase_r2_p_value_upper']:.4f}; "
            f"spec mean={c['specificity_mean_control']:+.6f}, p_upper={c['specificity_p_value_upper']:.4f}"
        )
    lines.append("")
    lines.append("TOP ORDER CORRELATIONS")
    lines.append("-" * 80)
    if order_corrs:
        for r in order_corrs[:12]:
            lines.append(f"{r['y']:>16s} vs {r['x']:<20s} n={r['n']:2d} r={r['r']:+.4f}")
    else:
        lines.append("none")
    lines.append("")
    lines.append("TOP CALIBRATION CORRELATIONS")
    lines.append("-" * 80)
    if cal_corrs:
        for r in cal_corrs[:12]:
            lines.append(f"{r['y']:>16s} vs {r['x']:<20s} n={r['n']:2d} r={r['r']:+.4f}")
    else:
        lines.append("none found in this base file")
    lines.append("")
    lines.append("INTERPRETATION TEMPLATE")
    lines.append("-" * 80)
    lines.append("Strong native-stagger evidence would be:")
    lines.append("  - nonzero YZ/ZY specificity in explicit-delay null")
    lines.append("  - coherent phase progression over rung/tile order")
    lines.append("  - collapse or weakening under independent bit-shuffle")
    lines.append("  - collapse or weakening under rung/tile permutation")
    lines.append("  - stronger YZ/ZY structure than XY/YX comparison")
    lines.append("  - plausible physical/order/calibration correlations")
    lines.append("")
    lines.append("Careful wording:")
    lines.append("  The null condition is an explicit-delay null, not necessarily a physical-structure null.")
    lines.append("  If ordered YZ/ZY phase remains, it may indicate a native QPU stagger/background manifold.")
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# COMPARE BASES
# =============================================================================

def compare_bases(paths: Sequence[Tuple[str, Path]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, path in paths:
        if not path.exists():
            continue
        try:
            b = load_base(path, name)
            m = compute_metrics(b)
            s = m.summary
            rows.append({
                "base": name,
                "path": str(path),
                "backend": s.get("backend", b.backend),
                "tiles": s["tiles"],
                "shots": s["shots"],
                "max_base_delay_dt": s["max_base_delay_dt"],
                "max_offset_dt": s["max_offset_dt"],
                "yzzy_energy_mean": s["yzzy_energy_mean"],
                "comparison_energy_mean": s["comparison_energy_mean"],
                "specificity_mean": s["specificity_mean"],
                "yzzy_to_comparison_ratio": s["yzzy_to_comparison_ratio"],
                "phase_unwrapped_slope_pi_per_rung": s["phase_unwrapped_slope_pi_per_rung"],
                "phase_unwrapped_r": s["phase_unwrapped_r"],
                "phase_unwrapped_r2": s["phase_unwrapped_r2"],
            })
        except Exception as e:
            rows.append({"base": name, "path": str(path), "error": f"{type(e).__name__}: {e}"})
    return rows


# =============================================================================
# CLI / MAIN
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe qproj null for implicit/native D_M stagger structure.")
    p.add_argument("--qproj-null", type=Path, default=DEFAULT_QPROJ_NULL)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--permutations", type=int, default=5000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--compare-defaults", action="store_true", default=True, help="Also summarize qproj/gproj null/base/offset defaults if found.")
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    tag = now_tag()
    out_dir = args.out_dir or (ANALYSIS_DIR / f"dm_probe_15_native_stagger_probe_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 112)
    print("  D_M NATIVE STAGGER PROBE")
    print("=" * 112)
    print(f"  D_M dir      : {D_M_DIR}")
    print(f"  Output dir   : {out_dir}")
    print(f"  qproj null   : {args.qproj_null}")
    print(f"  permutations : {args.permutations}")
    print()

    base = load_base(args.qproj_null, "qproj_null")
    metrics = compute_metrics(base)

    print("[OBSERVED]")
    s = metrics.summary
    print(f"  backend/job       : {base.backend} / {base.job_id}")
    print(f"  tiles/shots/rungs : {s['tiles']} / {s['shots']} / {s['n_rungs']}")
    print(f"  explicit delays   : max base={s['max_base_delay_dt']} max offset={s['max_offset_dt']} max total={s['max_total_delay_dt']}")
    print(f"  YZ/ZY energy mean : {s['yzzy_energy_mean']:+.8f}")
    print(f"  XY/YX comp mean   : {s['comparison_energy_mean']:+.8f}")
    print(f"  specificity mean  : {s['specificity_mean']:+.8f}")
    print(f"  YZ/ZY:comparison  : {s['yzzy_to_comparison_ratio']:+.4f}x")
    print(f"  phase slope       : {s['phase_unwrapped_slope_pi_per_rung']:+.6f} pi/rung")
    print(f"  phase r / R^2     : {s['phase_unwrapped_r']:+.6f} / {s['phase_unwrapped_r2']:+.6f}")

    print()
    print("[CONTROLS]")
    control_rows, control_summary = run_controls(base, metrics, n_perm=args.permutations, seed=args.seed)
    for c in control_summary["controls"]:
        print(
            f"  {c['control']:<34s} "
            f"phase|r| mean={c['phase_abs_r_mean']:.4f} p95={c['phase_abs_r_p95']:.4f} "
            f"p_upper={c['phase_abs_r_p_value_upper']:.4f} | "
            f"spec mean={c['specificity_mean_control']:+.6f} p_upper={c['specificity_p_value_upper']:.4f}"
        )

    order_corrs = collect_order_correlations(metrics.tile_rows)
    cal_corrs = extract_calibration_features(base, metrics.tile_rows)

    print()
    print("[TOP ORDER CORRELATIONS]")
    for r in order_corrs[:8]:
        print(f"  {r['y']:>16s} vs {r['x']:<20s} n={r['n']:2d} r={r['r']:+.4f}")
    if not order_corrs:
        print("  none")

    print()
    print("[TOP CALIBRATION CORRELATIONS]")
    for r in cal_corrs[:8]:
        print(f"  {r['y']:>16s} vs {r['x']:<20s} n={r['n']:2d} r={r['r']:+.4f}")
    if not cal_corrs:
        print("  none found")

    # Write artifacts.
    rung_csv = out_dir / "rung_metrics.csv"
    tile_csv = out_dir / "tile_metrics.csv"
    control_csv = out_dir / "control_stats.csv"
    order_csv = out_dir / "order_correlations.csv"
    cal_csv = out_dir / "calibration_correlations.csv"
    summary_json = out_dir / "native_stagger_summary.json"
    report_txt = out_dir / "native_stagger_report.txt"

    write_csv(rung_csv, metrics.rung_rows, [
        "base", "rung", "XY", "YZ", "ZY", "YX", "YZ_primary", "ZY_reciprocal_R_neg_ZY",
        "yzzy_energy", "comparison_energy", "specificity", "phase_rad", "phase_pi", "phase_unwrapped_pi",
        "count_XY", "count_YZ", "count_ZY", "count_YX",
    ])
    write_csv(tile_csv, metrics.tile_rows, [
        "base", "tile", "tile_array_index", "rung", "witness_index", "witness",
        "physical_q0", "physical_q1", "physical_q_gap",
        "base_delay_dt", "offset_dt", "total_delay_dt",
        "connected", "raw_corr", "mean_q0", "mean_q1", "spin_agreement", "abs_connected", "shots",
    ])
    write_csv(control_csv, control_rows, [
        "control", "iter", "phase_abs_r", "phase_r2", "specificity_mean", "yzzy_to_comparison_ratio",
    ])
    write_csv(order_csv, order_corrs, ["y", "x", "n", "r"])
    write_csv(cal_csv, cal_corrs, ["y", "x", "n", "r"])

    compare_rows: List[Dict[str, Any]] = []
    if args.compare_defaults:
        compare_paths = [
            ("qproj_null", DEFAULT_QPROJ_NULL),
            ("qproj_base_only", DEFAULT_QPROJ_BASE),
            ("qproj_offset_on", DEFAULT_QPROJ_OFFSET),
            ("gproj_null", DEFAULT_GPROJ_NULL),
            ("gproj_base_only", DEFAULT_GPROJ_BASE),
            ("gproj_offset_on", DEFAULT_GPROJ_OFFSET),
        ]
        compare_rows = compare_bases(compare_paths)
        write_csv(out_dir / "base_comparison.csv", compare_rows, [
            "base", "path", "error", "backend", "tiles", "shots", "max_base_delay_dt", "max_offset_dt",
            "yzzy_energy_mean", "comparison_energy_mean", "specificity_mean", "yzzy_to_comparison_ratio",
            "phase_unwrapped_slope_pi_per_rung", "phase_unwrapped_r", "phase_unwrapped_r2",
        ])

    summary_obj = {
        "created": tag,
        "probe": "d_m_native_stagger_probe",
        "hypothesis": "qproj null may contain implicit/native staggered ordering despite zero explicit delays",
        "base_summary": {k: v for k, v in metrics.summary.items() if k not in {"rung_witnesses", "phase_pi", "phase_unwrapped_pi", "counts"}},
        "rung_witnesses": metrics.summary["rung_witnesses"],
        "phase_pi": metrics.summary["phase_pi"],
        "phase_unwrapped_pi": metrics.summary["phase_unwrapped_pi"],
        "control_summary": control_summary,
        "top_order_correlations": order_corrs[:20],
        "top_calibration_correlations": cal_corrs[:20],
        "compare_defaults": compare_rows,
        "raw_npz_keys": base.raw_keys,
    }
    write_json(summary_json, summary_obj)
    write_report(report_txt, base, metrics, control_summary, order_corrs, cal_corrs)

    if not args.no_plots:
        maybe_plot(out_dir, metrics, control_rows)

    print()
    print("[DONE]")
    print(f"  report      : {report_txt}")
    print(f"  summary     : {summary_json}")
    print(f"  rung csv    : {rung_csv}")
    print(f"  tile csv    : {tile_csv}")
    print(f"  controls csv: {control_csv}")
    if args.compare_defaults:
        print(f"  comparison  : {out_dir / 'base_comparison.csv'}")
    print()


if __name__ == "__main__":
    main()
