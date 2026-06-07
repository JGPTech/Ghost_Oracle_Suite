#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — D_M PROBE 10: QPROJ DIMENSIONAL ENTANGLEMENT PROJECTION
==============================================================================

Purpose
-------
Probe 10 is the QPROJ task-lock probe for D_M.

Earlier probes found the D_M shape:

    YZ = primary Bell-witness dimension
    ZY = reciprocal / inverted witness dimension

and suggested that D_M is one dimensional witness manifold over:

    basis × chip-geometry × base-delay × offset-deformation × π-phase

Probe 10 takes the three real QPU/QPROJ conditions and turns them into the
first benchmark task:

    0. null / no temporal structure
       base_delays_dt = [0,0,0,0,0]
       offset_dt      = 0

    1. base-delay π-phase witness
       base_delays_dt = [0,256,1024,4096,16384]
       offset_dt      = 0

    2. offset-deformed witness manifold
       base_delays_dt = [0,256,1024,4096,16384]
       offset_dt      = 128 or nonzero

The script projects each QPROJ base into a D_M dimensional-entanglement vector:

    YZ primary amplitude
    ZY reciprocal / inverted amplitude
    YZ/ZY energy
    directional specificity against XY/YX
    π-phase trajectory score
    phase velocity / delay-lock measures
    null-distance and condition-separation summaries

Interpretation discipline
-------------------------
This is a Bell-witness / dimensional-entanglement projection benchmark.
It does NOT certify Bell nonlocality, reconstruct a density matrix, or prove a
prepared Bell state. It asks whether a structured Bell-witness manifold is
projectable from the qproj base and whether the structure survives controls.

Inputs
------
Explicit run:

    python ghost_oracle/D_M/probes/d_m_probe10_qproj_dimensional_entanglement.py ^
      --null       ghost_oracle/D_M/data/dm_data_<NULL_JOB>.npz ^
      --base-only  ghost_oracle/D_M/data/dm_data_<BASE_ONLY_JOB>.npz ^
      --offset-on  ghost_oracle/D_M/data/dm_data_<OFFSET_ON_JOB>.npz

Auto-discovery run from repo root, if D_M/data contains all three files:

    python ghost_oracle/D_M/probes/d_m_probe10_qproj_dimensional_entanglement.py --auto

Outputs
-------
    analysis/dm_probe_10_qproj_dimensional_entanglement_<timestamp>/
        result.json
        projection_vectors.csv
        rung_projection.csv
        condition_separation.csv
        classification_summary.csv
        control_summary.csv
        metadata_used.csv
        optional PNG plots if matplotlib is available
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# DEFAULT D_M LISTENER PLAN
# =============================================================================

WITNESS_PAIRS: List[Tuple[int, int]] = [(0, 1), (1, 2), (2, 1), (1, 0)]
WITNESS_ORDER = ["XY", "YZ", "ZY", "YX"]
BASIS_LABELS = ["X", "Y", "Z"]
DEFAULT_BASE_DELAYS_DT = [0, 256, 1024, 4096, 16384]
DEFAULT_OFFSET_DT = 128
CONDITION_ORDER = ["null", "base_only", "offset_on"]
CONDITION_LABELS = {
    "null": "no_delay_no_offset",
    "base_only": "base_delay_only",
    "offset_on": "base_delay_plus_offset",
}


# =============================================================================
# PATHS / IO
# =============================================================================

HERE = Path(__file__).resolve().parent
DM_ROOT = HERE.parent if HERE.name == "probes" else HERE
REPO_ROOT = next(
    (p for p in [DM_ROOT, *DM_ROOT.parents] if (p / ".git").exists() or (p / "requirements.txt").exists()),
    DM_ROOT,
)
DEFAULT_DATA_DIR = DM_ROOT / "data"
DEFAULT_ANALYSIS_DIR = DM_ROOT / "probes" / "analyze"
DEFAULT_QPROJ_NULL   = DEFAULT_DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fm4ihvjngc73aq3ccg.npz"
DEFAULT_QPROJ_BASE   = DEFAULT_DATA_DIR / "dm_data_bell_listener_cavity_offset_d8flk2jo3njc73f0g560.npz"
DEFAULT_QPROJ_OFFSET = DEFAULT_DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fl82bo3njc73f0fgd0.npz"
DEFAULT_GPROJ_NULL   = DEFAULT_DATA_DIR / "dm_gpu_data_null_4096shots_seed9031229662612491082.npz"
DEFAULT_GPROJ_BASE   = DEFAULT_DATA_DIR / "dm_gpu_data_base_delay_4096shots_seed2669559634056472362.npz"
DEFAULT_GPROJ_OFFSET = DEFAULT_DATA_DIR / "dm_gpu_data_offset_deformed_4096shots_seed6727069190982977623.npz"


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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def scalar_str(obj: Any) -> str:
    try:
        arr = np.asarray(obj)
        if arr.shape == ():
            return str(arr.item())
        if arr.size == 1:
            return str(arr.ravel()[0].item())
        return str(arr)
    except Exception:
        return str(obj)


def scalar_int(obj: Any, default: int = -1) -> int:
    try:
        return int(np.asarray(obj).item())
    except Exception:
        return int(default)


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class TileMeta:
    tile: int
    base_delay_dt: int
    offset_dt: int
    total_delay_dt: int
    basis_q0: int
    basis_q1: int
    witness: str
    repaired: bool


@dataclass
class TileCorr:
    condition: str
    tile: int
    rung_index: int
    witness_index: int
    witness: str
    base_delay_dt: int
    offset_dt: int
    total_delay_dt: int
    shots: int
    mean_q0: float
    mean_q1: float
    corr_raw: float
    corr_connected: float


# =============================================================================
# LOAD / REPAIR METADATA
# =============================================================================


def load_pair_stack(npz: Any) -> np.ndarray:
    if "pair" in npz.files:
        pair = np.asarray(npz["pair"], dtype=np.uint8)
        if pair.ndim != 3 or pair.shape[2] != 2:
            raise ValueError(f"pair must have shape (tiles, shots, 2), got {pair.shape}")
        return pair

    tiles = []
    t = 0
    while f"pair_tile{t}" in npz.files:
        arr = np.asarray(npz[f"pair_tile{t}"], dtype=np.uint8)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"pair_tile{t} must have shape (shots, 2), got {arr.shape}")
        tiles.append(arr)
        t += 1
    if not tiles:
        raise ValueError("No pair array or pair_tile* arrays found in D_M qproj base")
    return np.stack(tiles, axis=0)


def read_optional(npz: Any, key: str) -> Optional[np.ndarray]:
    if key not in npz.files:
        return None
    return np.asarray(npz[key])


def valid_int_array(a: Optional[np.ndarray], n: int, min_value: int = 0) -> bool:
    if a is None or a.shape[0] < n:
        return False
    return bool(np.all(np.asarray(a[:n]) >= min_value))


def witness_label(b0: int, b1: int) -> str:
    if 0 <= b0 < len(BASIS_LABELS) and 0 <= b1 < len(BASIS_LABELS):
        return BASIS_LABELS[b0] + BASIS_LABELS[b1]
    return "??"


def build_repaired_plan(
    num_tiles: int,
    existing_base: Optional[np.ndarray],
    existing_offset: Optional[np.ndarray],
    existing_total: Optional[np.ndarray],
    existing_b0: Optional[np.ndarray],
    existing_b1: Optional[np.ndarray],
    base_delays_dt: Sequence[int],
    offset_step_dt: int,
    witness_pairs: Sequence[Tuple[int, int]],
    force_repair: bool = False,
) -> List[TileMeta]:
    have_existing = (
        not force_repair
        and valid_int_array(existing_base, num_tiles, 0)
        and valid_int_array(existing_offset, num_tiles, 0)
        and valid_int_array(existing_total, num_tiles, 0)
        and valid_int_array(existing_b0, num_tiles, 0)
        and valid_int_array(existing_b1, num_tiles, 0)
    )

    plan: List[TileMeta] = []
    for t in range(num_tiles):
        if have_existing:
            base = int(existing_base[t])
            off = int(existing_offset[t])
            total = int(existing_total[t])
            b0 = int(existing_b0[t])
            b1 = int(existing_b1[t])
            repaired = False
        else:
            rung = t // len(witness_pairs)
            wi = t % len(witness_pairs)
            base = int(base_delays_dt[min(rung, len(base_delays_dt) - 1)])
            off = int(t * offset_step_dt)
            total = int(base + off)
            b0, b1 = witness_pairs[wi]
            repaired = True
        plan.append(TileMeta(
            tile=int(t),
            base_delay_dt=base,
            offset_dt=off,
            total_delay_dt=total,
            basis_q0=int(b0),
            basis_q1=int(b1),
            witness=witness_label(int(b0), int(b1)),
            repaired=repaired,
        ))
    return plan


def infer_condition_from_metadata(path: Path) -> Optional[str]:
    try:
        obj = np.load(path, allow_pickle=True)
        pair = load_pair_stack(obj)
        n = pair.shape[0]
        base = read_optional(obj, "tile_base_delay_dt")
        off = read_optional(obj, "tile_offset_dt")
        total = read_optional(obj, "tile_total_delay_dt")
        if base is None or off is None or total is None or base.shape[0] < n or off.shape[0] < n:
            return None
        base_v = np.asarray(base[:n], dtype=np.int64)
        off_v = np.asarray(off[:n], dtype=np.int64)
        has_base = bool(np.max(base_v) > 0)
        has_offset = bool(np.max(off_v) > 0)
        if not has_base and not has_offset:
            return "null"
        if has_base and not has_offset:
            return "base_only"
        if has_base and has_offset:
            return "offset_on"
        return None
    except Exception:
        return None


def auto_discover_paths(data_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Tuple[float, Path]] = {}
    for path in sorted(data_dir.glob("dm_data_*.npz")):
        cond = infer_condition_from_metadata(path)
        if cond is None:
            continue
        mtime = path.stat().st_mtime
        if cond not in found or mtime > found[cond][0]:
            found[cond] = (mtime, path)
    return {k: v[1] for k, v in found.items()}


def load_base(path: Path, condition: str, args: argparse.Namespace) -> Tuple[np.ndarray, List[TileMeta], Dict[str, Any]]:
    obj = np.load(path, allow_pickle=True)
    pair = load_pair_stack(obj)
    n_tiles, shots, width = pair.shape
    if width != 2:
        raise ValueError(f"expected 2-qubit pair width, got {width}")

    offset_step_dt = int(args.offset_dt)
    if "offset_dt" in obj.files and not args.force_offset:
        try:
            stored = int(np.asarray(obj["offset_dt"]).item())
            if stored >= 0:
                offset_step_dt = stored
        except Exception:
            pass

    plan = build_repaired_plan(
        num_tiles=n_tiles,
        existing_base=read_optional(obj, "tile_base_delay_dt"),
        existing_offset=read_optional(obj, "tile_offset_dt"),
        existing_total=read_optional(obj, "tile_total_delay_dt"),
        existing_b0=read_optional(obj, "tile_basis_q0"),
        existing_b1=read_optional(obj, "tile_basis_q1"),
        base_delays_dt=args.base_delays_dt,
        offset_step_dt=offset_step_dt,
        witness_pairs=WITNESS_PAIRS,
        force_repair=args.force_repair,
    )

    meta = {
        "condition": condition,
        "condition_label": CONDITION_LABELS.get(condition, condition),
        "path": str(path),
        "schema": scalar_str(obj["schema"]) if "schema" in obj.files else "unknown",
        "operator": scalar_str(obj["operator"]) if "operator" in obj.files else "D_M",
        "substrate": scalar_str(obj["substrate"]) if "substrate" in obj.files else "qproj",
        "job_id": scalar_str(obj["job_id"]) if "job_id" in obj.files else path.stem,
        "backend": scalar_str(obj["backend"]) if "backend" in obj.files else "unknown",
        "num_tiles": int(n_tiles),
        "shots": int(shots),
        "metadata_repaired": bool(any(m.repaired for m in plan)),
        "base_delays_dt_used": list(map(int, args.base_delays_dt)),
        "offset_step_dt_used": int(offset_step_dt),
        "witness_order_used": [witness_label(a, b) for a, b in WITNESS_PAIRS],
    }
    return pair, plan, meta


# =============================================================================
# CORRELATORS / PROJECTION
# =============================================================================


def pair_to_signs(bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    b = np.asarray(bits, dtype=np.uint8)
    s0 = 1.0 - 2.0 * b[:, 0].astype(np.float64)
    s1 = 1.0 - 2.0 * b[:, 1].astype(np.float64)
    return s0, s1


def corr_from_bits(bits: np.ndarray) -> Tuple[float, float, float, float]:
    s0, s1 = pair_to_signs(bits)
    m0 = float(np.mean(s0))
    m1 = float(np.mean(s1))
    raw = float(np.mean(s0 * s1))
    conn = float(raw - m0 * m1)
    return raw, conn, m0, m1


def compute_tile_corrs(condition: str, pair: np.ndarray, plan: Sequence[TileMeta]) -> List[TileCorr]:
    rows: List[TileCorr] = []
    for t, meta in enumerate(plan):
        raw, conn, m0, m1 = corr_from_bits(pair[t])
        rows.append(TileCorr(
            condition=condition,
            tile=int(t),
            rung_index=int(t // len(WITNESS_ORDER)),
            witness_index=int(t % len(WITNESS_ORDER)),
            witness=meta.witness,
            base_delay_dt=int(meta.base_delay_dt),
            offset_dt=int(meta.offset_dt),
            total_delay_dt=int(meta.total_delay_dt),
            shots=int(pair[t].shape[0]),
            mean_q0=m0,
            mean_q1=m1,
            corr_raw=raw,
            corr_connected=conn,
        ))
    return rows


def safe_corrcoef(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def permutation_p_corr(x: Sequence[float], y: Sequence[float], obs: float, n_perm: int, seed: int, two_sided: bool = True) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 1.0
    vals = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        vals[i] = safe_corrcoef(x, rng.permutation(y))
    if two_sided:
        return float((np.count_nonzero(np.abs(vals) >= abs(obs)) + 1) / (n_perm + 1))
    return float((np.count_nonzero(vals >= obs) + 1) / (n_perm + 1))


def pi_phase_from_yz_zy(yz: float, zy: float) -> Tuple[float, float, float, float, float]:
    """
    YZ is primary coordinate. ZY is reciprocal/inverted, so R=-ZY.
    Phase is modulo π because witness orientation is axial, not a full directed vector.
    """
    y = float(yz)
    r = float(-zy)
    energy = float(math.sqrt(y * y + r * r))
    energy_rms = float(energy / math.sqrt(2.0))
    phase = float(math.atan2(r, y) % math.pi)
    phase_unit = float(phase / math.pi)
    inversion = float(-y * zy)
    return energy, energy_rms, phase, phase_unit, inversion


def phase_rung_rows(condition: str, tile_rows: Sequence[TileCorr]) -> List[Dict[str, Any]]:
    groups: Dict[int, List[TileCorr]] = {}
    for row in tile_rows:
        groups.setdefault(row.rung_index, []).append(row)

    out: List[Dict[str, Any]] = []
    for rung, rows in sorted(groups.items()):
        by_w = {r.witness: r for r in rows}
        if "YZ" not in by_w or "ZY" not in by_w:
            continue
        xy = float(by_w["XY"].corr_connected) if "XY" in by_w else 0.0
        yz = float(by_w["YZ"].corr_connected)
        zy = float(by_w["ZY"].corr_connected)
        yx = float(by_w["YX"].corr_connected) if "YX" in by_w else 0.0
        energy, energy_rms, phase, phase_unit, inversion = pi_phase_from_yz_zy(yz, zy)
        comp_energy = float(math.sqrt((xy * xy + yx * yx) / 2.0))
        specificity = float(energy_rms - comp_energy)
        total_delay = float(np.mean([r.total_delay_dt for r in rows]))
        base_delay = float(np.mean([r.base_delay_dt for r in rows]))
        offset = float(np.mean([r.offset_dt for r in rows]))
        raw_rms = float(math.sqrt(np.mean([r.corr_raw * r.corr_raw for r in rows])))
        conn_rms = float(math.sqrt(np.mean([r.corr_connected * r.corr_connected for r in rows])))
        out.append({
            "condition": condition,
            "condition_label": CONDITION_LABELS.get(condition, condition),
            "rung_index": int(rung),
            "base_delay_dt_mean": base_delay,
            "offset_dt_mean": offset,
            "total_delay_dt_mean": total_delay,
            "XY_connected": xy,
            "YZ_connected": yz,
            "ZY_connected": zy,
            "YX_connected": yx,
            "return_coord_minus_ZY": float(-zy),
            "yzzy_energy_euclidean": energy,
            "yzzy_energy_rms": energy_rms,
            "xyyx_energy_rms": comp_energy,
            "directional_specificity": specificity,
            "yzzy_inversion_score": inversion,
            "pi_phase_rad": phase,
            "pi_phase_unit": phase_unit,
            "pi_phase_degrees_mod180": float(phase * 180.0 / math.pi),
            "zy_inverted_relative_to_yz": bool(yz * zy < 0.0),
            "raw_bell_rms": raw_rms,
            "connected_bell_rms": conn_rms,
        })
    return out


# =============================================================================
# π-PERIODIC FIT / PROJECTION VECTOR
# =============================================================================


def normalize_x(x: Sequence[float], mode: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if mode == "log":
        arr = np.log1p(np.maximum(arr, 0.0))
    lo = float(np.min(arr)) if arr.size else 0.0
    hi = float(np.max(arr)) if arr.size else 1.0
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def axial_coherence_pi(phases: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    p = np.asarray(phases, dtype=np.float64)
    if p.size == 0:
        return 0.0
    if weights is None:
        w = np.ones_like(p)
    else:
        w = np.asarray(weights, dtype=np.float64)
    if np.sum(np.abs(w)) < 1e-15:
        return 0.0
    # phase is modulo π, so use doubled-angle coherence.
    z = np.sum(w * np.exp(2j * p)) / np.sum(np.abs(w))
    return float(abs(z))


def unwrap_pi_phase(phases: Sequence[float]) -> np.ndarray:
    return 0.5 * np.unwrap(2.0 * np.asarray(phases, dtype=np.float64))


def pi_periodic_fit_score(
    delays: Sequence[float],
    phases: Sequence[float],
    weights: Optional[Sequence[float]],
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Fit a simple axial phase model phase ≈ ω*x + φ0 mod π.

    The score is the weighted doubled-angle coherence of residuals after the
    best ω fit. This is a compact phase-trajectory score, not a theorem.
    """
    d = np.asarray(delays, dtype=np.float64)
    ph = np.asarray(phases, dtype=np.float64)
    wt = np.asarray(weights, dtype=np.float64) if weights is not None else None

    if d.size < 3 or np.std(d) < 1e-12 or np.std(ph) < 1e-12:
        return {
            "pi_periodic_score": 0.0,
            "pi_periodic_p": 1.0,
            "pi_periodic_mode": "none",
            "pi_periodic_omega": 0.0,
            "phase_velocity_r": 0.0,
            "phase_velocity_p": 1.0,
            "phase_span_pi_units": 0.0,
        }

    best = {"score": -1.0, "mode": "linear", "omega": 0.0}
    omegas = np.linspace(-2.5 * math.pi, 2.5 * math.pi, 801)
    for mode in ("linear", "log"):
        x = normalize_x(d, mode)
        for omega in omegas:
            residual = (ph - omega * x) % math.pi
            score = axial_coherence_pi(residual, wt)
            if score > best["score"]:
                best = {"score": float(score), "mode": mode, "omega": float(omega)}

    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_perm, dtype=np.float64)
    x_best = normalize_x(d, best["mode"])
    for i in range(n_perm):
        perm = rng.permutation(ph)
        # Keep ω fixed to avoid overfitting the null too hard; this tests the discovered trajectory.
        residual = (perm - best["omega"] * x_best) % math.pi
        null_scores[i] = axial_coherence_pi(residual, wt)
    p = float((np.count_nonzero(null_scores >= best["score"]) + 1) / (n_perm + 1))

    unwrapped = unwrap_pi_phase(ph)
    phase_r = safe_corrcoef(normalize_x(d, best["mode"]), unwrapped)
    phase_p = permutation_p_corr(normalize_x(d, best["mode"]), unwrapped, phase_r, n_perm, seed + 313)
    span = float((np.max(unwrapped) - np.min(unwrapped)) / math.pi) if unwrapped.size else 0.0

    return {
        "pi_periodic_score": float(best["score"]),
        "pi_periodic_p": p,
        "pi_periodic_mode": best["mode"],
        "pi_periodic_omega": float(best["omega"]),
        "phase_velocity_r": float(phase_r),
        "phase_velocity_p": float(phase_p),
        "phase_span_pi_units": span,
    }


def projection_vector(condition: str, rung_rows: Sequence[Dict[str, Any]], n_perm: int, seed: int) -> Dict[str, Any]:
    rows = list(rung_rows)
    if not rows:
        return {"condition": condition, "condition_label": CONDITION_LABELS.get(condition, condition)}

    yz = np.asarray([r["YZ_connected"] for r in rows], dtype=np.float64)
    zy = np.asarray([r["ZY_connected"] for r in rows], dtype=np.float64)
    ret = -zy
    energy = np.asarray([r["yzzy_energy_euclidean"] for r in rows], dtype=np.float64)
    energy_rms = np.asarray([r["yzzy_energy_rms"] for r in rows], dtype=np.float64)
    spec = np.asarray([r["directional_specificity"] for r in rows], dtype=np.float64)
    inv = np.asarray([r["yzzy_inversion_score"] for r in rows], dtype=np.float64)
    phase = np.asarray([r["pi_phase_rad"] for r in rows], dtype=np.float64)
    total = np.asarray([r["total_delay_dt_mean"] for r in rows], dtype=np.float64)
    base = np.asarray([r["base_delay_dt_mean"] for r in rows], dtype=np.float64)
    offset = np.asarray([r["offset_dt_mean"] for r in rows], dtype=np.float64)
    raw_rms = np.asarray([r["raw_bell_rms"] for r in rows], dtype=np.float64)
    conn_rms = np.asarray([r["connected_bell_rms"] for r in rows], dtype=np.float64)

    energy_r = safe_corrcoef(total, energy)
    spec_r = safe_corrcoef(total, spec)
    yz_r = safe_corrcoef(total, yz)
    conn_rms_r = safe_corrcoef(total, conn_rms)

    phase_fit = pi_periodic_fit_score(total, phase, energy, n_perm=n_perm, seed=seed + 17)

    # Compact score used only for ranking/benchmarking, not scientific certification.
    strength = float(np.mean(energy_rms) + max(0.0, float(np.mean(spec))))
    delay_lock = float(np.mean([
        max(0.0, abs(energy_r)),
        max(0.0, abs(spec_r)),
        max(0.0, abs(yz_r)),
    ]))
    phase_strength = float(phase_fit["pi_periodic_score"] * (1.0 - min(1.0, phase_fit["pi_periodic_p"])))
    projection_score = float(strength * (1.0 + 0.5 * delay_lock + 0.5 * phase_strength))

    return {
        "condition": condition,
        "condition_label": CONDITION_LABELS.get(condition, condition),
        "complete_rungs": int(len(rows)),
        "total_delay_min": float(np.min(total)),
        "total_delay_max": float(np.max(total)),
        "base_delay_max": float(np.max(base)),
        "offset_delay_max": float(np.max(offset)),
        "raw_bell_rms_mean": float(np.mean(raw_rms)),
        "connected_bell_rms_mean": float(np.mean(conn_rms)),
        "YZ_primary_mean": float(np.mean(yz)),
        "YZ_primary_abs_mean": float(np.mean(np.abs(yz))),
        "YZ_positive_fraction": float(np.mean(yz > 0.0)),
        "ZY_mean": float(np.mean(zy)),
        "return_minus_ZY_mean": float(np.mean(ret)),
        "ZY_inverted_fraction_vs_YZ": float(np.mean(yz * zy < 0.0)),
        "YZ_ZY_energy_mean": float(np.mean(energy)),
        "YZ_ZY_energy_max": float(np.max(energy)),
        "YZ_ZY_energy_rms_mean": float(np.mean(energy_rms)),
        "directional_specificity_mean": float(np.mean(spec)),
        "directional_specificity_max": float(np.max(spec)),
        "inversion_score_mean": float(np.mean(inv)),
        "late_rung_YZ": float(yz[-1]),
        "late_rung_ZY": float(zy[-1]),
        "late_rung_energy": float(energy[-1]),
        "late_rung_specificity": float(spec[-1]),
        "total_delay_YZ_tracking_r": float(yz_r),
        "total_delay_YZ_tracking_p": permutation_p_corr(total, yz, yz_r, n_perm, seed + 101),
        "total_delay_energy_tracking_r": float(energy_r),
        "total_delay_energy_tracking_p": permutation_p_corr(total, energy, energy_r, n_perm, seed + 102),
        "total_delay_specificity_tracking_r": float(spec_r),
        "total_delay_specificity_tracking_p": permutation_p_corr(total, spec, spec_r, n_perm, seed + 103),
        "total_delay_connected_rms_tracking_r": float(conn_rms_r),
        "total_delay_connected_rms_tracking_p": permutation_p_corr(total, conn_rms, conn_rms_r, n_perm, seed + 104),
        "delay_lock_score": float(delay_lock),
        "dimensional_entanglement_strength": float(strength),
        "projection_score": projection_score,
        **phase_fit,
    }


# =============================================================================
# CONTROLS
# =============================================================================


def independent_bit_shuffle_pair(pair: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(pair, copy=True)
    for t in range(out.shape[0]):
        out[t, :, 1] = rng.permutation(out[t, :, 1])
    return out


def projection_from_pair(condition: str, pair: np.ndarray, plan: Sequence[TileMeta], n_perm: int, seed: int) -> Tuple[List[TileCorr], List[Dict[str, Any]], Dict[str, Any]]:
    tiles = compute_tile_corrs(condition, pair, plan)
    rungs = phase_rung_rows(condition, tiles)
    vec = projection_vector(condition, rungs, n_perm=n_perm, seed=seed)
    return tiles, rungs, vec


def witness_label_shuffle_rungs(condition: str, tile_rows: Sequence[TileCorr], rng: np.random.Generator) -> List[Dict[str, Any]]:
    groups: Dict[int, List[TileCorr]] = {}
    for row in tile_rows:
        groups.setdefault(row.rung_index, []).append(row)

    shuffled_rows: List[TileCorr] = []
    for rung, rows in sorted(groups.items()):
        rows = list(rows)
        labels = [r.witness for r in rows]
        labels_perm = list(rng.permutation(labels))
        for r, new_label in zip(rows, labels_perm):
            shuffled_rows.append(TileCorr(
                condition=condition,
                tile=r.tile,
                rung_index=r.rung_index,
                witness_index=r.witness_index,
                witness=str(new_label),
                base_delay_dt=r.base_delay_dt,
                offset_dt=r.offset_dt,
                total_delay_dt=r.total_delay_dt,
                shots=r.shots,
                mean_q0=r.mean_q0,
                mean_q1=r.mean_q1,
                corr_raw=r.corr_raw,
                corr_connected=r.corr_connected,
            ))
    return phase_rung_rows(condition, shuffled_rows)


def summarize_control_distribution(obs: float, vals: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=np.float64)
    mu = float(np.mean(vals)) if vals.size else 0.0
    sd = float(np.std(vals)) if vals.size else 0.0
    z = float((obs - mu) / sd) if sd > 1e-12 else 0.0
    p_up = float((np.count_nonzero(vals >= obs) + 1) / (vals.size + 1)) if vals.size else 1.0
    p_two = float((np.count_nonzero(np.abs(vals - mu) >= abs(obs - mu)) + 1) / (vals.size + 1)) if vals.size else 1.0
    return {"null_mean": mu, "null_std": sd, "z": z, "p_upper": p_up, "p_two_sided": p_two}


def run_controls(
    condition: str,
    pair: np.ndarray,
    plan: Sequence[TileMeta],
    tile_rows: Sequence[TileCorr],
    obs_vec: Dict[str, Any],
    n_perm: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    metrics = [
        "projection_score",
        "dimensional_entanglement_strength",
        "YZ_ZY_energy_mean",
        "directional_specificity_mean",
        "pi_periodic_score",
        "delay_lock_score",
    ]
    controls: List[Dict[str, Any]] = []

    # Independent bit shuffle: preserves single-qubit marginals but breaks q0/q1 shot pairing.
    dist: Dict[str, List[float]] = {m: [] for m in metrics}
    for i in range(n_perm):
        p2 = independent_bit_shuffle_pair(pair, rng)
        _, rungs, vec = projection_from_pair(condition, p2, plan, n_perm=64, seed=seed + 10000 + i)
        for m in metrics:
            dist[m].append(float(vec.get(m, 0.0)))
    for m in metrics:
        vals = np.asarray(dist[m], dtype=np.float64)
        obs = float(obs_vec.get(m, 0.0))
        s = summarize_control_distribution(obs, vals)
        controls.append({
            "condition": condition,
            "control": "independent_bit_shuffle",
            "metric": m,
            "observed": obs,
            **s,
        })

    # Witness label shuffle: asks whether the YZ/ZY assignment is load-bearing.
    dist = {m: [] for m in metrics}
    for i in range(n_perm):
        shuffled_rungs = witness_label_shuffle_rungs(condition, tile_rows, rng)
        vec = projection_vector(condition, shuffled_rungs, n_perm=64, seed=seed + 20000 + i)
        for m in metrics:
            dist[m].append(float(vec.get(m, 0.0)))
    for m in metrics:
        vals = np.asarray(dist[m], dtype=np.float64)
        obs = float(obs_vec.get(m, 0.0))
        s = summarize_control_distribution(obs, vals)
        controls.append({
            "condition": condition,
            "control": "witness_label_shuffle",
            "metric": m,
            "observed": obs,
            **s,
        })

    # Delay-order controls are already embedded as permutation p-values in the projection vector.
    for m in [
        "total_delay_YZ_tracking_r",
        "total_delay_energy_tracking_r",
        "total_delay_specificity_tracking_r",
        "total_delay_connected_rms_tracking_r",
        "phase_velocity_r",
    ]:
        p_key = m.replace("_r", "_p")
        controls.append({
            "condition": condition,
            "control": "delay_order_permutation",
            "metric": m,
            "observed": float(obs_vec.get(m, 0.0)),
            "null_mean": "",
            "null_std": "",
            "z": "",
            "p_upper": "",
            "p_two_sided": float(obs_vec.get(p_key, 1.0)),
        })

    return controls


# =============================================================================
# CONDITION SEPARATION / CLASSIFICATION
# =============================================================================


def vector_feature_names(include_order: bool = False) -> List[str]:
    names = [
        "YZ_connected",
        "return_coord_minus_ZY",
        "yzzy_energy_euclidean",
        "directional_specificity",
        "yzzy_inversion_score",
        "phase_cos2",
        "phase_sin2",
    ]
    if include_order:
        names += ["total_delay_norm"]
    return names


def rung_feature_matrix(rung_rows: Sequence[Dict[str, Any]], include_order: bool = False) -> Tuple[np.ndarray, List[str], List[str]]:
    rows = list(rung_rows)
    delays = np.asarray([r["total_delay_dt_mean"] for r in rows], dtype=np.float64)
    if np.max(delays) - np.min(delays) < 1e-12:
        dnorm = np.zeros_like(delays)
    else:
        dnorm = (delays - np.min(delays)) / (np.max(delays) - np.min(delays))

    X: List[List[float]] = []
    y: List[str] = []
    for i, r in enumerate(rows):
        phase = float(r["pi_phase_rad"])
        vals = [
            float(r["YZ_connected"]),
            float(r["return_coord_minus_ZY"]),
            float(r["yzzy_energy_euclidean"]),
            float(r["directional_specificity"]),
            float(r["yzzy_inversion_score"]),
            float(math.cos(2.0 * phase)),
            float(math.sin(2.0 * phase)),
        ]
        if include_order:
            vals.append(float(dnorm[i]))
        X.append(vals)
        y.append(str(r["condition"]))

    return np.asarray(X, dtype=np.float64), y, vector_feature_names(include_order)


def standardize_train_apply(X: np.ndarray, train_idx: np.ndarray, test_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    mu = X[train_idx].mean(axis=0, keepdims=True)
    sd = X[train_idx].std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (X[train_idx] - mu) / sd, (X[[test_idx]] - mu) / sd


def leave_one_out_nearest_centroid(X: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=object)
    classes = sorted(set(labels))
    if X.shape[0] < len(classes) + 1 or len(classes) < 2:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "n": int(X.shape[0]), "predictions": []}

    preds: List[str] = []
    truths: List[str] = []
    for i in range(X.shape[0]):
        train_idx = np.asarray([j for j in range(X.shape[0]) if j != i], dtype=np.int64)
        Xtr, Xte = standardize_train_apply(X, train_idx, i)
        ytr = y[train_idx]
        centroids = {}
        for c in classes:
            mask = ytr == c
            if np.any(mask):
                centroids[c] = Xtr[mask].mean(axis=0)
        dists = {c: float(np.linalg.norm(Xte[0] - mu)) for c, mu in centroids.items()}
        pred = min(dists, key=dists.get)
        preds.append(pred)
        truths.append(str(y[i]))

    acc = float(np.mean([p == t for p, t in zip(preds, truths)]))
    recalls = []
    for c in classes:
        idx = [i for i, t in enumerate(truths) if t == c]
        if idx:
            recalls.append(float(np.mean([preds[i] == c for i in idx])))
    bal = float(np.mean(recalls)) if recalls else 0.0
    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "n": int(X.shape[0]),
        "classes": classes,
        "predictions": [{"truth": t, "pred": p} for t, p in zip(truths, preds)],
    }


def classification_with_permutation(rung_rows: Sequence[Dict[str, Any]], include_order: bool, n_perm: int, seed: int) -> Dict[str, Any]:
    X, labels, names = rung_feature_matrix(rung_rows, include_order=include_order)
    obs = leave_one_out_nearest_centroid(X, labels)
    rng = np.random.default_rng(seed)
    null_bal = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm_labels = list(rng.permutation(labels))
        null_bal[i] = float(leave_one_out_nearest_centroid(X, perm_labels).get("balanced_accuracy", 0.0))
    p = float((np.count_nonzero(null_bal >= obs["balanced_accuracy"]) + 1) / (n_perm + 1))
    return {
        "classifier": "leave_one_out_nearest_centroid",
        "feature_set": "manifold_plus_delay_order" if include_order else "manifold_only_no_delay_metadata",
        "features": names,
        "accuracy": float(obs["accuracy"]),
        "balanced_accuracy": float(obs["balanced_accuracy"]),
        "n_samples": int(obs["n"]),
        "condition_label_shuffle_p": p,
        "null_balanced_accuracy_mean": float(np.mean(null_bal)),
        "null_balanced_accuracy_std": float(np.std(null_bal)),
    }


def base_vector_for_distance(vec: Dict[str, Any]) -> np.ndarray:
    keys = [
        "YZ_primary_mean",
        "return_minus_ZY_mean",
        "YZ_ZY_energy_mean",
        "directional_specificity_mean",
        "inversion_score_mean",
        "pi_periodic_score",
        "phase_velocity_r",
        "delay_lock_score",
        "projection_score",
    ]
    return np.asarray([float(vec.get(k, 0.0)) for k in keys], dtype=np.float64)


def condition_separation_rows(vectors: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keys = [k for k in CONDITION_ORDER if k in vectors]
    raw = {k: base_vector_for_distance(vectors[k]) for k in keys}
    if not raw:
        return rows
    mat = np.vstack([raw[k] for k in keys])
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    zvec = {k: (raw[k] - mu[0]) / sd[0] for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            rows.append({
                "condition_a": a,
                "condition_b": b,
                "label_a": CONDITION_LABELS.get(a, a),
                "label_b": CONDITION_LABELS.get(b, b),
                "euclidean_raw": float(np.linalg.norm(raw[a] - raw[b])),
                "euclidean_standardized": float(np.linalg.norm(zvec[a] - zvec[b])),
                "projection_score_a": float(vectors[a].get("projection_score", 0.0)),
                "projection_score_b": float(vectors[b].get("projection_score", 0.0)),
                "projection_score_delta_b_minus_a": float(vectors[b].get("projection_score", 0.0) - vectors[a].get("projection_score", 0.0)),
                "energy_mean_a": float(vectors[a].get("YZ_ZY_energy_mean", 0.0)),
                "energy_mean_b": float(vectors[b].get("YZ_ZY_energy_mean", 0.0)),
                "energy_delta_b_minus_a": float(vectors[b].get("YZ_ZY_energy_mean", 0.0) - vectors[a].get("YZ_ZY_energy_mean", 0.0)),
                "pi_score_a": float(vectors[a].get("pi_periodic_score", 0.0)),
                "pi_score_b": float(vectors[b].get("pi_periodic_score", 0.0)),
                "pi_score_delta_b_minus_a": float(vectors[b].get("pi_periodic_score", 0.0) - vectors[a].get("pi_periodic_score", 0.0)),
            })
    return rows


# =============================================================================
# PLOTS
# =============================================================================


def maybe_make_plots(out_dir: Path, rung_rows: Sequence[Dict[str, Any]], vectors: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    rows = list(rung_rows)
    if rows:
        for metric, fname, ylabel in [
            ("yzzy_energy_euclidean", "energy_vs_total_delay.png", "YZ/ZY energy"),
            ("directional_specificity", "specificity_vs_total_delay.png", "Directional specificity"),
            ("pi_phase_degrees_mod180", "pi_phase_vs_total_delay.png", "π phase (degrees mod 180)"),
        ]:
            plt.figure(figsize=(8, 5))
            for cond in CONDITION_ORDER:
                cr = [r for r in rows if r["condition"] == cond]
                if not cr:
                    continue
                x = [r["total_delay_dt_mean"] for r in cr]
                y = [r[metric] for r in cr]
                plt.plot(x, y, marker="o", label=CONDITION_LABELS.get(cond, cond))
            plt.xlabel("total delay dt")
            plt.ylabel(ylabel)
            plt.title(f"D_M Probe 10 — {ylabel}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / fname, dpi=160)
            plt.close()

    if vectors:
        names = [CONDITION_LABELS.get(v["condition"], v["condition"]) for v in vectors]
        x = np.arange(len(names))
        metrics = [
            ("projection_score", "projection_score_summary.png", "Projection score"),
            ("YZ_ZY_energy_mean", "energy_summary.png", "Mean YZ/ZY energy"),
            ("pi_periodic_score", "pi_phase_score_summary.png", "π-periodic score"),
        ]
        for metric, fname, ylabel in metrics:
            plt.figure(figsize=(8, 5))
            vals = [float(v.get(metric, 0.0)) for v in vectors]
            plt.bar(x, vals)
            plt.xticks(x, names, rotation=20, ha="right")
            plt.ylabel(ylabel)
            plt.title(f"D_M Probe 10 — {ylabel}")
            plt.tight_layout()
            plt.savefig(out_dir / fname, dpi=160)
            plt.close()


# =============================================================================
# CLI / MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D_M Probe 10 — QPROJ Dimensional Entanglement Projection Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--null", default=str(DEFAULT_QPROJ_NULL), help="No-delay/no-offset qproj .npz base")
    p.add_argument("--base-only", default=str(DEFAULT_QPROJ_BASE), help="Base-delay-only qproj .npz base")
    p.add_argument("--offset-on", default=str(DEFAULT_QPROJ_OFFSET), help="Base-delay + offset qproj .npz base")
    p.add_argument("--auto", action="store_true", help="Auto-discover latest null/base_only/offset_on bases in D_M/data")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--base-delays-dt", type=int, nargs="+", default=DEFAULT_BASE_DELAYS_DT)
    p.add_argument("--offset-dt", type=int, default=DEFAULT_OFFSET_DT)
    p.add_argument("--force-repair", action="store_true", help="Ignore stored metadata and reconstruct canonical plan")
    p.add_argument("--force-offset", action="store_true", help="Use CLI --offset-dt rather than stored file offset_dt if repairing")
    return p.parse_args()


def collect_paths(args: argparse.Namespace) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    if args.null:
        paths["null"] = Path(args.null)
    if args.base_only:
        paths["base_only"] = Path(args.base_only)
    if args.offset_on:
        paths["offset_on"] = Path(args.offset_on)

    if args.auto or not paths:
        auto = auto_discover_paths(Path(args.data_dir))
        for k, v in auto.items():
            paths.setdefault(k, v)

    missing = [k for k in CONDITION_ORDER if k not in paths]
    if missing:
        raise FileNotFoundError(
            "Missing required D_M qproj condition file(s): " + ", ".join(missing) + "\n"
            "Pass --null, --base-only, --offset-on explicitly, or use --auto after generating all three."
        )
    for k, v in paths.items():
        if not v.exists():
            raise FileNotFoundError(f"{k} path does not exist: {v}")
    return paths


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    paths = collect_paths(args)

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_ANALYSIS_DIR / f"dm_probe_10_qproj_dimensional_entanglement_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("  GHOST ORACLE SUITE — D_M PROBE 10: QPROJ DIMENSIONAL ENTANGLEMENT PROJECTION")
    print("=" * 108)
    print(f"  Out dir : {out_dir}")
    print(f"  n_perm  : {args.n_perm}")
    print("-" * 108)

    all_tile_rows: List[TileCorr] = []
    all_rung_rows: List[Dict[str, Any]] = []
    projection_rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    loaded: Dict[str, Dict[str, Any]] = {}

    for idx, condition in enumerate(CONDITION_ORDER):
        path = paths[condition]
        pair, plan, meta = load_base(path, condition, args)
        tiles, rungs, vec = projection_from_pair(condition, pair, plan, n_perm=args.n_perm, seed=args.seed + idx * 1000)
        all_tile_rows.extend(tiles)
        all_rung_rows.extend(rungs)
        projection_rows.append(vec)
        metadata_rows.append(meta)
        loaded[condition] = {"pair": pair, "plan": plan, "meta": meta, "tiles": tiles, "rungs": rungs, "vector": vec}

        print(f"  CONDITION: {condition:<10} {CONDITION_LABELS[condition]}")
        print(f"    input      : {path}")
        print(f"    backend/job: {meta['backend']} / {meta['job_id']}")
        print(f"    tiles/shots: {meta['num_tiles']} / {meta['shots']}  repaired={meta['metadata_repaired']}")
        print(f"    delay max  : base={vec.get('base_delay_max', 0):.0f} off={vec.get('offset_delay_max', 0):.0f} total={vec.get('total_delay_max', 0):.0f}")
        print(f"    YZ mean    : {vec.get('YZ_primary_mean', 0):+.6f}  pos_frac={vec.get('YZ_positive_fraction', 0):.3f}")
        print(f"    energy     : mean={vec.get('YZ_ZY_energy_mean', 0):.6f} max={vec.get('YZ_ZY_energy_max', 0):.6f}")
        print(f"    specificity: mean={vec.get('directional_specificity_mean', 0):+.6f} max={vec.get('directional_specificity_max', 0):+.6f}")
        print(f"    π score    : {vec.get('pi_periodic_score', 0):.4f} p={vec.get('pi_periodic_p', 1):.4f} mode={vec.get('pi_periodic_mode', 'none')}")
        print(f"    projection : {vec.get('projection_score', 0):.6f}")

        control_rows.extend(run_controls(
            condition=condition,
            pair=pair,
            plan=plan,
            tile_rows=tiles,
            obs_vec=vec,
            n_perm=args.n_perm,
            seed=args.seed + 50000 + idx * 1000,
        ))
        print("-" * 108)

    vector_map = {r["condition"]: r for r in projection_rows}
    sep_rows = condition_separation_rows(vector_map)

    # Null-distance is computed after all vectors exist.
    if "null" in vector_map:
        null_vec = base_vector_for_distance(vector_map["null"])
        for row in projection_rows:
            cond = row["condition"]
            row["null_distance_raw"] = float(np.linalg.norm(base_vector_for_distance(row) - null_vec))

    class_rows = [
        classification_with_permutation(all_rung_rows, include_order=False, n_perm=args.n_perm, seed=args.seed + 70000),
        classification_with_permutation(all_rung_rows, include_order=True, n_perm=args.n_perm, seed=args.seed + 71000),
    ]

    print("  CONDITION SEPARATION")
    for r in sep_rows:
        print(
            f"    {r['condition_a']} -> {r['condition_b']}: "
            f"Δprojection={r['projection_score_delta_b_minus_a']:+.6f} "
            f"Δenergy={r['energy_delta_b_minus_a']:+.6f} "
            f"std_dist={r['euclidean_standardized']:.3f}"
        )
    print("-" * 108)
    print("  RUNG-LEVEL CONDITION CLASSIFICATION")
    for r in class_rows:
        print(
            f"    {r['feature_set']}: bal_acc={r['balanced_accuracy']:.3f} "
            f"acc={r['accuracy']:.3f} p={r['condition_label_shuffle_p']:.4f}"
        )
    print("-" * 108)

    # Flatten tile rows for CSV.
    tile_dicts = [asdict(r) for r in all_tile_rows]

    projection_fields = [
        "condition", "condition_label", "complete_rungs",
        "total_delay_min", "total_delay_max", "base_delay_max", "offset_delay_max",
        "raw_bell_rms_mean", "connected_bell_rms_mean",
        "YZ_primary_mean", "YZ_primary_abs_mean", "YZ_positive_fraction",
        "ZY_mean", "return_minus_ZY_mean", "ZY_inverted_fraction_vs_YZ",
        "YZ_ZY_energy_mean", "YZ_ZY_energy_max", "YZ_ZY_energy_rms_mean",
        "directional_specificity_mean", "directional_specificity_max", "inversion_score_mean",
        "late_rung_YZ", "late_rung_ZY", "late_rung_energy", "late_rung_specificity",
        "total_delay_YZ_tracking_r", "total_delay_YZ_tracking_p",
        "total_delay_energy_tracking_r", "total_delay_energy_tracking_p",
        "total_delay_specificity_tracking_r", "total_delay_specificity_tracking_p",
        "total_delay_connected_rms_tracking_r", "total_delay_connected_rms_tracking_p",
        "delay_lock_score", "dimensional_entanglement_strength", "projection_score", "null_distance_raw",
        "pi_periodic_score", "pi_periodic_p", "pi_periodic_mode", "pi_periodic_omega",
        "phase_velocity_r", "phase_velocity_p", "phase_span_pi_units",
    ]
    rung_fields = [
        "condition", "condition_label", "rung_index",
        "base_delay_dt_mean", "offset_dt_mean", "total_delay_dt_mean",
        "XY_connected", "YZ_connected", "ZY_connected", "YX_connected",
        "return_coord_minus_ZY", "yzzy_energy_euclidean", "yzzy_energy_rms", "xyyx_energy_rms",
        "directional_specificity", "yzzy_inversion_score",
        "pi_phase_rad", "pi_phase_unit", "pi_phase_degrees_mod180", "zy_inverted_relative_to_yz",
        "raw_bell_rms", "connected_bell_rms",
    ]
    tile_fields = [
        "condition", "tile", "rung_index", "witness_index", "witness",
        "base_delay_dt", "offset_dt", "total_delay_dt", "shots",
        "mean_q0", "mean_q1", "corr_raw", "corr_connected",
    ]
    sep_fields = [
        "condition_a", "condition_b", "label_a", "label_b",
        "euclidean_raw", "euclidean_standardized",
        "projection_score_a", "projection_score_b", "projection_score_delta_b_minus_a",
        "energy_mean_a", "energy_mean_b", "energy_delta_b_minus_a",
        "pi_score_a", "pi_score_b", "pi_score_delta_b_minus_a",
    ]
    class_fields = [
        "classifier", "feature_set", "features", "accuracy", "balanced_accuracy", "n_samples",
        "condition_label_shuffle_p", "null_balanced_accuracy_mean", "null_balanced_accuracy_std",
    ]
    control_fields = [
        "condition", "control", "metric", "observed", "null_mean", "null_std", "z", "p_upper", "p_two_sided",
    ]
    metadata_fields = [
        "condition", "condition_label", "path", "schema", "operator", "substrate", "job_id", "backend",
        "num_tiles", "shots", "metadata_repaired", "base_delays_dt_used", "offset_step_dt_used", "witness_order_used",
    ]

    write_csv(out_dir / "projection_vectors.csv", projection_rows, projection_fields)
    write_csv(out_dir / "rung_projection.csv", all_rung_rows, rung_fields)
    write_csv(out_dir / "tile_correlators.csv", tile_dicts, tile_fields)
    write_csv(out_dir / "condition_separation.csv", sep_rows, sep_fields)
    write_csv(out_dir / "classification_summary.csv", class_rows, class_fields)
    write_csv(out_dir / "control_summary.csv", control_rows, control_fields)
    write_csv(out_dir / "metadata_used.csv", metadata_rows, metadata_fields)

    maybe_make_plots(out_dir, all_rung_rows, projection_rows)

    result = {
        "schema": "ghost_oracle.dm.probe10.qproj_dimensional_entanglement.v1",
        "operator": "D_M",
        "probe": "Probe 10 — QPROJ Dimensional Entanglement Projection Benchmark",
        "framing": "Dimensional entanglement projection from YZ-primary / ZY-reciprocal Bell-witness qproj records",
        "bounded_claim": (
            "Reports structured Bell-witness manifold projection; does not certify Bell nonlocality, "
            "does not reconstruct a density matrix, and does not prove a prepared Bell state."
        ),
        "config": vars(args),
        "paths": {k: str(v) for k, v in paths.items()},
        "metadata": metadata_rows,
        "projection_vectors": projection_rows,
        "condition_separation": sep_rows,
        "classification_summary": class_rows,
        "controls": control_rows,
        "elapsed_sec": time.perf_counter() - t0,
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2)

    print(f"  [SAVED] {out_dir}")
    print("=" * 108)


if __name__ == "__main__":
    main()
