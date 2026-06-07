#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — D_M PROBE 09: π-PHASE / π-ADIC WITNESS
==============================================================================

Purpose
-------
Third D_M probe for the Bell-listener / cavity-offset qproj base.

Probe 01 asked:

    Do the four Bell-witness correlators light up, and do they track delay/offset?

Probe 02 narrowed the structure:

    YZ is the primary Bell-witness dimension.
    ZY is the reciprocal / inverted side of the witness.

Probe 09 treats that YZ/ZY pair as a phase-space coordinate of one D_M
witness manifold:

    Y  = connected(YZ)
    R  = -connected(ZY)       # reciprocal return coordinate

    energy = sqrt(Y^2 + R^2)
    phase  = atan2(R, Y) mod π

The π-periodic witness is the serious part: it asks whether the YZ/ZY witness
forms a coherent π-periodic phase trajectory over delay/offset.

The π-adic witness is intentionally experimental / toy / playful. π is not a
prime and this is NOT a formal p-adic construction. It is a made-up folded-
residue diagnostic that asks whether the phase coordinate aligns with nested
irrational π-folds of the delay coordinate. If it finds something interesting,
that means "worth probing", not "mathematical theorem achieved."

Interpretation discipline
-------------------------
This script reports Bell-witness correlation structure. It does NOT certify
entanglement, reconstruct a density matrix, or claim a literal Bell state.

Current D_M framing:

    D_M is one dimensional witness manifold, not independent channels.

Useful axes/components:

    basis / witness orientation  : YZ primary, ZY reciprocal/inverted
    distance / geometry axis     : witness can persist without explicit offset
    temporal / offset axis       : offset can lock witness to delay order
    π-phase axis                 : YZ/ZY witness may rotate through phase

Inputs
------
One clean D_M qproj .npz base, or two bases for offset-on/off comparison.

Expected schema from fixed D_M generator:

    pair                      uint8, shape (tiles, shots, 2)
    basis                     int8,  shape (tiles, 2)
    tile_base_delay_dt        int32, shape (tiles,)
    tile_offset_dt            int32, shape (tiles,)
    tile_total_delay_dt       int32, shape (tiles,)
    tile_basis_q0             int8,  shape (tiles,)
    tile_basis_q1             int8,  shape (tiles,)

Usage
-----
Recommended comparator run from repo root:

    python ghost_oracle/D_M/probes/d_m_probe09_pi_phase_witness.py ^
      --offset-on  ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_<ON_JOB>.npz ^
      --offset-off ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_<OFF_JOB>.npz

Single-base mode:

    python ghost_oracle/D_M/probes/d_m_probe09_pi_phase_witness.py ^
      --qpu-base ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_<JOB>.npz

Outputs
-------
    analysis/dm_probe_09_pi_phase_witness_<timestamp>/
        result.json
        phase_rungs.csv
        phase_summary.csv
        comparator_summary.csv
        pi_adic_levels.csv
        optional PNG plots if matplotlib is available
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# DEFAULT D_M LISTENER PLAN
# =============================================================================

WITNESS_PAIRS: List[Tuple[int, int]] = [(0, 1), (1, 2), (2, 1), (1, 0)]
WITNESS_ORDER = ["XY", "YZ", "ZY", "YX"]
BASIS_LABELS = ["X", "Y", "Z"]
DEFAULT_BASE_DELAYS_DT = [0, 256, 1024, 4096, 16384]
DEFAULT_OFFSET_DT = 128


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
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def scalar_str(obj: Any) -> str:
    try:
        arr = np.asarray(obj)
        if arr.shape == ():
            return str(arr.item())
        return str(arr)
    except Exception:
        return str(obj)


def resolve_latest_qpu_base(data_dir: Path) -> Optional[Path]:
    for name in ("latest_dm_qpu_data.json", "latest_dm_data.json"):
        p = data_dir / name
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        raw = obj.get("path", obj.get("npz", ""))
        q = Path(raw)
        if q.exists():
            return q
        fallback = data_dir / q.name
        if fallback.exists():
            return fallback
    return None


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
# CORRELATORS / RUNG METRICS
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


def circular_resultant(angles: np.ndarray, weights: Optional[np.ndarray] = None, period: float = 2.0 * math.pi) -> complex:
    a = np.asarray(angles, dtype=np.float64)
    if weights is None:
        w = np.ones_like(a, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    if a.size == 0 or np.sum(np.abs(w)) < 1e-15:
        return 0.0 + 0.0j
    theta = (2.0 * math.pi / period) * a
    return complex(np.sum(w * np.exp(1j * theta)) / np.sum(np.abs(w)))


def axial_coherence_pi(phase_pi: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Coherence for angles defined modulo π using doubled-angle trick."""
    z = circular_resultant(2.0 * np.asarray(phase_pi), weights=weights, period=2.0 * math.pi)
    return float(abs(z))


def unwrap_pi_phase(phase_pi: np.ndarray) -> np.ndarray:
    """Unwrap phase values with π-periodic equivalence."""
    doubled = np.unwrap(2.0 * np.asarray(phase_pi, dtype=np.float64))
    return 0.5 * doubled


def pi_phase_from_yz_zy(yz: float, zy: float) -> Tuple[float, float, float, float]:
    """
    YZ is primary coordinate. ZY is reciprocal/inverted, so return coordinate R=-ZY.
    Phase is modulo π because witness orientation is axial, not a full directed vector.
    """
    y = float(yz)
    r = float(-zy)
    energy = float(math.sqrt(y * y + r * r))
    phase = float(math.atan2(r, y) % math.pi)
    phase_unit = float(phase / math.pi)
    inversion = float(-y * zy)
    return energy, phase, phase_unit, inversion


def phase_rung_rows(condition: str, tile_rows: Sequence[TileCorr]) -> List[Dict[str, Any]]:
    groups: Dict[int, List[TileCorr]] = {}
    for row in tile_rows:
        groups.setdefault(row.rung_index, []).append(row)

    out: List[Dict[str, Any]] = []
    for rung, rows in sorted(groups.items()):
        by_w = {r.witness: r for r in rows}
        if "YZ" not in by_w or "ZY" not in by_w:
            continue
        xy = float(by_w.get("XY", by_w["YZ"]).corr_connected) if "XY" in by_w else 0.0
        yz = float(by_w["YZ"].corr_connected)
        zy = float(by_w["ZY"].corr_connected)
        yx = float(by_w.get("YX", by_w["YZ"]).corr_connected) if "YX" in by_w else 0.0
        energy, phase, phase_unit, inversion = pi_phase_from_yz_zy(yz, zy)
        comp_energy = float(math.sqrt((xy * xy + yx * yx) / 2.0))
        specificity = float(energy / math.sqrt(2.0) - comp_energy)
        total_delay = float(np.mean([r.total_delay_dt for r in rows]))
        base_delay = float(np.mean([r.base_delay_dt for r in rows]))
        offset = float(np.mean([r.offset_dt for r in rows]))
        out.append({
            "condition": condition,
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
            "yzzy_energy_rms": float(energy / math.sqrt(2.0)),
            "xyyx_energy_rms": comp_energy,
            "directional_specificity": specificity,
            "yzzy_inversion_score": inversion,
            "pi_phase_rad": phase,
            "pi_phase_unit": phase_unit,
            "pi_phase_degrees_mod180": float(phase * 180.0 / math.pi),
            "zy_inverted_relative_to_yz": bool(yz * zy < 0.0),
        })
    return out


# =============================================================================
# π-PERIODIC WITNESS
# =============================================================================


def normalize_x(x: Sequence[float], use_log: bool = False) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if use_log:
        arr = np.log1p(np.maximum(arr, 0.0))
    lo = float(np.min(arr)) if arr.size else 0.0
    hi = float(np.max(arr)) if arr.size else 1.0
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def fit_pi_periodic_phase(
    x_delay: Sequence[float],
    phase_pi: Sequence[float],
    energy: Sequence[float],
    omega_grid: int = 2001,
    max_cycles: float = 3.0,
    use_log_delay: bool = False,
) -> Dict[str, Any]:
    """
    Fit θ ≈ φ + π * cycles * x, modulo π.

    Because θ is π-periodic, residual coherence is measured with exp(2i residual).
    The intercept φ is optimized analytically for each cycles value by the resultant.
    """
    theta = np.asarray(phase_pi, dtype=np.float64)
    x = normalize_x(x_delay, use_log=use_log_delay)
    w = np.asarray(energy, dtype=np.float64)
    if theta.size < 3 or np.sum(w) < 1e-15:
        return {
            "pi_periodic_score": 0.0,
            "pi_periodic_cycles": 0.0,
            "pi_periodic_phase0_rad": 0.0,
            "pi_periodic_mode": "log" if use_log_delay else "linear",
            "pi_periodic_circular_variance": 1.0,
        }

    cycles_values = np.linspace(-max_cycles, max_cycles, int(omega_grid), dtype=np.float64)
    best_score = -1.0
    best_cycles = 0.0
    best_phase0 = 0.0

    for cycles in cycles_values:
        model = math.pi * cycles * x
        residual = theta - model
        # Optimize phase0: resultant of exp(2i residual). Half angle gives modulo-π intercept.
        z = np.sum(w * np.exp(2j * residual)) / max(float(np.sum(w)), 1e-15)
        score = float(abs(z))
        if score > best_score:
            best_score = score
            best_cycles = float(cycles)
            best_phase0 = float((0.5 * math.atan2(z.imag, z.real)) % math.pi)

    pred = (best_phase0 + math.pi * best_cycles * x) % math.pi
    residual = ((theta - pred + math.pi / 2.0) % math.pi) - math.pi / 2.0
    rms_resid = float(math.sqrt(np.average(residual * residual, weights=w))) if np.sum(w) > 0 else 0.0

    return {
        "pi_periodic_score": float(best_score),
        "pi_periodic_cycles": best_cycles,
        "pi_periodic_phase0_rad": best_phase0,
        "pi_periodic_phase0_deg_mod180": float(best_phase0 * 180.0 / math.pi),
        "pi_periodic_mode": "log" if use_log_delay else "linear",
        "pi_periodic_circular_variance": float(1.0 - best_score),
        "pi_periodic_weighted_rms_residual_rad": rms_resid,
        "pi_periodic_weighted_rms_residual_deg": float(rms_resid * 180.0 / math.pi),
    }


def permutation_p_phase_fit(
    x_delay: Sequence[float],
    phase_pi: Sequence[float],
    energy: Sequence[float],
    obs_score: float,
    n_perm: int,
    seed: int,
    omega_grid: int,
    max_cycles: float,
    use_log_delay: bool,
) -> float:
    rng = np.random.default_rng(seed)
    phase = np.asarray(phase_pi, dtype=np.float64)
    if phase.size < 4:
        return 1.0
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm_phase = rng.permutation(phase)
        fit = fit_pi_periodic_phase(
            x_delay=x_delay,
            phase_pi=perm_phase,
            energy=energy,
            omega_grid=omega_grid,
            max_cycles=max_cycles,
            use_log_delay=use_log_delay,
        )
        null[i] = float(fit["pi_periodic_score"])
    return float((np.count_nonzero(null >= obs_score) + 1) / (n_perm + 1))


# =============================================================================
# π-ADIC TOY WITNESS
# =============================================================================


def frac(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)


def pi_adic_toy_levels(
    x_delay: Sequence[float],
    phase_unit: Sequence[float],
    energy: Sequence[float],
    levels: int = 7,
    use_log_delay: bool = False,
) -> List[Dict[str, Any]]:
    """
    Toy π-adic / π-fold residue diagnostic.

    This is deliberately NOT formal p-adic math. It takes normalized delay x∈[0,1],
    folds it by irrational powers π^k, and checks whether the witness phase unit
    aligns with the folded residue.

        residue_k = frac(x * π^k)
        score_k   = |Σ energy * exp(2πi * (phase_unit - residue_k))| / Σ energy

    If this is high, it means the phase coordinate is aligning with a nested π-fold
    delay residue. Treat as playful telemetry unless controls keep validating it.
    """
    x = normalize_x(x_delay, use_log=use_log_delay)
    ph = np.asarray(phase_unit, dtype=np.float64) % 1.0
    w = np.asarray(energy, dtype=np.float64)
    rows: List[Dict[str, Any]] = []
    if ph.size == 0 or np.sum(w) < 1e-15:
        return rows

    for k in range(1, int(levels) + 1):
        residue = frac(x * (math.pi ** k))
        delta = ph - residue
        z = np.sum(w * np.exp(2j * math.pi * delta)) / max(float(np.sum(w)), 1e-15)
        score = float(abs(z))
        mean_resid = float((math.atan2(z.imag, z.real) / (2.0 * math.pi)) % 1.0)
        abs_fold_error = np.abs(((delta + 0.5) % 1.0) - 0.5)
        weighted_fold_error = float(np.average(abs_fold_error, weights=w)) if np.sum(w) > 0 else 0.0
        rows.append({
            "pi_adic_level": int(k),
            "pi_adic_mode": "log" if use_log_delay else "linear",
            "pi_power": float(math.pi ** k),
            "pi_adic_score": score,
            "pi_adic_circular_variance": float(1.0 - score),
            "pi_adic_mean_residual_unit": mean_resid,
            "pi_adic_weighted_abs_fold_error_unit": weighted_fold_error,
        })
    return rows


def permutation_p_pi_adic_best(
    x_delay: Sequence[float],
    phase_unit: Sequence[float],
    energy: Sequence[float],
    obs_best: float,
    levels: int,
    n_perm: int,
    seed: int,
    use_log_delay: bool,
) -> float:
    rng = np.random.default_rng(seed)
    ph = np.asarray(phase_unit, dtype=np.float64)
    if ph.size < 4:
        return 1.0
    vals = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        rows = pi_adic_toy_levels(
            x_delay=x_delay,
            phase_unit=rng.permutation(ph),
            energy=energy,
            levels=levels,
            use_log_delay=use_log_delay,
        )
        vals[i] = max((float(r["pi_adic_score"]) for r in rows), default=0.0)
    return float((np.count_nonzero(vals >= obs_best) + 1) / (n_perm + 1))


# =============================================================================
# CONDITION SUMMARY
# =============================================================================


def summarize_condition(condition: str, phase_rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    delays = np.asarray([r["total_delay_dt_mean"] for r in phase_rows], dtype=np.float64)
    base_delays = np.asarray([r["base_delay_dt_mean"] for r in phase_rows], dtype=np.float64)
    energies = np.asarray([r["yzzy_energy_euclidean"] for r in phase_rows], dtype=np.float64)
    rms_energies = np.asarray([r["yzzy_energy_rms"] for r in phase_rows], dtype=np.float64)
    specificity = np.asarray([r["directional_specificity"] for r in phase_rows], dtype=np.float64)
    yz = np.asarray([r["YZ_connected"] for r in phase_rows], dtype=np.float64)
    zy = np.asarray([r["ZY_connected"] for r in phase_rows], dtype=np.float64)
    ret = np.asarray([r["return_coord_minus_ZY"] for r in phase_rows], dtype=np.float64)
    phases = np.asarray([r["pi_phase_rad"] for r in phase_rows], dtype=np.float64)
    phase_unit = np.asarray([r["pi_phase_unit"] for r in phase_rows], dtype=np.float64)
    unwrapped = unwrap_pi_phase(phases)

    phase_span = float(np.max(unwrapped) - np.min(unwrapped)) if unwrapped.size else 0.0
    phase_span_pi_units = float(phase_span / math.pi)
    phase_velocity_r = safe_corrcoef(delays, unwrapped)
    energy_r = safe_corrcoef(delays, energies)
    specificity_r = safe_corrcoef(delays, specificity)
    yz_r = safe_corrcoef(delays, yz)
    ret_r = safe_corrcoef(delays, ret)
    phase_energy_r = safe_corrcoef(energies, unwrapped)

    p_energy = permutation_p_corr(delays, energies, energy_r, args.n_perm, args.seed + 11)
    p_spec = permutation_p_corr(delays, specificity, specificity_r, args.n_perm, args.seed + 12)
    p_phase = permutation_p_corr(delays, unwrapped, phase_velocity_r, args.n_perm, args.seed + 13)

    fit_linear = fit_pi_periodic_phase(
        delays, phases, energies,
        omega_grid=args.omega_grid,
        max_cycles=args.max_cycles,
        use_log_delay=False,
    )
    fit_linear_p = permutation_p_phase_fit(
        delays, phases, energies, float(fit_linear["pi_periodic_score"]),
        n_perm=args.n_perm,
        seed=args.seed + 21,
        omega_grid=max(101, args.omega_grid // 5),
        max_cycles=args.max_cycles,
        use_log_delay=False,
    )
    fit_log = fit_pi_periodic_phase(
        delays, phases, energies,
        omega_grid=args.omega_grid,
        max_cycles=args.max_cycles,
        use_log_delay=True,
    )
    fit_log_p = permutation_p_phase_fit(
        delays, phases, energies, float(fit_log["pi_periodic_score"]),
        n_perm=args.n_perm,
        seed=args.seed + 22,
        omega_grid=max(101, args.omega_grid // 5),
        max_cycles=args.max_cycles,
        use_log_delay=True,
    )

    if fit_log["pi_periodic_score"] > fit_linear["pi_periodic_score"]:
        best_fit = dict(fit_log)
        best_fit["pi_periodic_perm_p"] = fit_log_p
    else:
        best_fit = dict(fit_linear)
        best_fit["pi_periodic_perm_p"] = fit_linear_p

    pi_adic_linear = pi_adic_toy_levels(delays, phase_unit, energies, levels=args.pi_adic_levels, use_log_delay=False)
    pi_adic_log = pi_adic_toy_levels(delays, phase_unit, energies, levels=args.pi_adic_levels, use_log_delay=True)
    for row in pi_adic_linear + pi_adic_log:
        row["condition"] = condition

    all_pi_adic = pi_adic_linear + pi_adic_log
    best_pi = max(all_pi_adic, key=lambda r: float(r["pi_adic_score"])) if all_pi_adic else {}
    best_pi_score = float(best_pi.get("pi_adic_score", 0.0))
    best_pi_p = permutation_p_pi_adic_best(
        delays, phase_unit, energies, best_pi_score,
        levels=args.pi_adic_levels,
        n_perm=args.n_perm,
        seed=args.seed + 31,
        use_log_delay=(best_pi.get("pi_adic_mode", "linear") == "log"),
    )

    summary = {
        "condition": condition,
        "n_rungs": int(len(phase_rows)),
        "delay_min": float(np.min(delays)) if delays.size else 0.0,
        "delay_max": float(np.max(delays)) if delays.size else 0.0,
        "YZ_positive_fraction": float(np.mean(yz > 0.0)) if yz.size else 0.0,
        "ZY_inverted_fraction_vs_YZ": float(np.mean(yz * zy < 0.0)) if yz.size else 0.0,
        "YZ_primary_mean": float(np.mean(yz)) if yz.size else 0.0,
        "ZY_return_mean": float(np.mean(zy)) if zy.size else 0.0,
        "return_coord_mean_minus_ZY": float(np.mean(ret)) if ret.size else 0.0,
        "energy_mean_euclidean": float(np.mean(energies)) if energies.size else 0.0,
        "energy_max_euclidean": float(np.max(energies)) if energies.size else 0.0,
        "energy_mean_rms": float(np.mean(rms_energies)) if rms_energies.size else 0.0,
        "directional_specificity_mean": float(np.mean(specificity)) if specificity.size else 0.0,
        "directional_specificity_max": float(np.max(specificity)) if specificity.size else 0.0,
        "total_delay_energy_r": energy_r,
        "total_delay_energy_perm_p": p_energy,
        "total_delay_specificity_r": specificity_r,
        "total_delay_specificity_perm_p": p_spec,
        "total_delay_YZ_primary_r": yz_r,
        "total_delay_return_coord_r": ret_r,
        "phase_unwrapped_span_rad": phase_span,
        "phase_unwrapped_span_pi_units": phase_span_pi_units,
        "total_delay_phase_velocity_r": phase_velocity_r,
        "total_delay_phase_velocity_perm_p": p_phase,
        "energy_phase_r": phase_energy_r,
        "axial_phase_coherence": axial_coherence_pi(phases, weights=energies),
        "late_rung_phase_deg_mod180": float(phase_rows[-1]["pi_phase_degrees_mod180"]) if phase_rows else 0.0,
        "late_rung_energy_euclidean": float(phase_rows[-1]["yzzy_energy_euclidean"]) if phase_rows else 0.0,
        "late_rung_YZ": float(phase_rows[-1]["YZ_connected"]) if phase_rows else 0.0,
        "late_rung_ZY": float(phase_rows[-1]["ZY_connected"]) if phase_rows else 0.0,
        **best_fit,
        "pi_adic_best_score": best_pi_score,
        "pi_adic_best_level": int(best_pi.get("pi_adic_level", 0)) if best_pi else 0,
        "pi_adic_best_mode": str(best_pi.get("pi_adic_mode", "")) if best_pi else "",
        "pi_adic_best_perm_p": best_pi_p,
        "pi_adic_best_weighted_abs_fold_error_unit": float(best_pi.get("pi_adic_weighted_abs_fold_error_unit", 0.0)) if best_pi else 0.0,
    }
    return summary, all_pi_adic


def comparator_summary(summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by = {str(s["condition"]): s for s in summaries}
    if "offset_on" not in by or "offset_off" not in by:
        return {}
    on = by["offset_on"]
    off = by["offset_off"]

    def delta(key: str) -> float:
        return float(on.get(key, 0.0)) - float(off.get(key, 0.0))

    def ratio(key: str) -> float:
        denom = abs(float(off.get(key, 0.0)))
        return float(on.get(key, 0.0)) / denom if denom > 1e-12 else float("inf")

    return {
        "has_comparator": True,
        "energy_tracking_gain_delta": delta("total_delay_energy_r"),
        "specificity_tracking_gain_delta": delta("total_delay_specificity_r"),
        "phase_velocity_tracking_gain_delta": delta("total_delay_phase_velocity_r"),
        "pi_periodic_score_delta": delta("pi_periodic_score"),
        "pi_adic_best_score_delta": delta("pi_adic_best_score"),
        "phase_span_pi_units_delta": delta("phase_unwrapped_span_pi_units"),
        "energy_tracking_gain_ratio": ratio("total_delay_energy_r"),
        "specificity_tracking_gain_ratio": ratio("total_delay_specificity_r"),
        "pi_periodic_score_ratio": ratio("pi_periodic_score"),
        "interpretation_hint": (
            "positive tracking deltas support offset-on phase/order locking; "
            "similar energy but weaker tracking suggests distance witness persists while phase/order lock changes"
        ),
    }


# =============================================================================
# PLOTTING
# =============================================================================


def maybe_make_plots(out_dir: Path, phase_rows: List[Dict[str, Any]], summaries: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    conditions = sorted({str(r["condition"]) for r in phase_rows})

    # Plot phase vs delay.
    plt.figure(figsize=(9, 5))
    for cond in conditions:
        rows = [r for r in phase_rows if r["condition"] == cond]
        x = [r["total_delay_dt_mean"] for r in rows]
        y = [r["pi_phase_degrees_mod180"] for r in rows]
        plt.plot(x, y, marker="o", label=cond)
    plt.xlabel("total delay dt")
    plt.ylabel("π-phase (degrees mod 180)")
    plt.title("D_M π-phase trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pi_phase_vs_total_delay.png", dpi=160)
    plt.close()

    # Plot energy and specificity.
    plt.figure(figsize=(9, 5))
    for cond in conditions:
        rows = [r for r in phase_rows if r["condition"] == cond]
        x = [r["total_delay_dt_mean"] for r in rows]
        y = [r["yzzy_energy_euclidean"] for r in rows]
        plt.plot(x, y, marker="o", label=f"{cond} energy")
    plt.xlabel("total delay dt")
    plt.ylabel("YZ/ZY Euclidean energy")
    plt.title("D_M YZ/ZY phase energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "yzzy_energy_vs_total_delay.png", dpi=160)
    plt.close()

    # Bar summary.
    if summaries:
        names = [s["condition"] for s in summaries]
        vals = [s.get("pi_periodic_score", 0.0) for s in summaries]
        plt.figure(figsize=(7, 4))
        plt.bar(names, vals)
        plt.ylim(0, 1.05)
        plt.ylabel("π-periodic fit score")
        plt.title("D_M π-periodic witness score")
        plt.tight_layout()
        plt.savefig(out_dir / "pi_periodic_score_summary.png", dpi=160)
        plt.close()


# =============================================================================
# CLI / MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D_M Probe 09 — π-phase and toy π-adic witness comparator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qpu-base", default=str(DEFAULT_QPROJ_OFFSET), help="Single D_M qproj .npz base")
    p.add_argument("--offset-on", default=str(DEFAULT_QPROJ_OFFSET), help="Offset-on D_M qproj .npz base")
    p.add_argument("--offset-off", default=str(DEFAULT_QPROJ_BASE), help="Offset-off D_M qproj .npz base")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--base-delays-dt", type=int, nargs="+", default=DEFAULT_BASE_DELAYS_DT)
    p.add_argument("--offset-dt", type=int, default=DEFAULT_OFFSET_DT, help="Fallback offset step if metadata needs repair")
    p.add_argument("--force-repair", action="store_true", help="Ignore stored metadata and rebuild default tile plan")
    p.add_argument("--force-offset", action="store_true", help="Use CLI --offset-dt even if file has stored offset_dt")
    p.add_argument("--n-perm", type=int, default=5000, help="Permutation count for p-values")
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--omega-grid", type=int, default=2001)
    p.add_argument("--max-cycles", type=float, default=3.0)
    p.add_argument("--pi-adic-levels", type=int, default=7)
    return p.parse_args()


def input_conditions(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    conditions: List[Tuple[str, Path]] = []
    if args.offset_on:
        conditions.append(("offset_on", Path(args.offset_on)))
    if args.offset_off:
        conditions.append(("offset_off", Path(args.offset_off)))
    if args.qpu_base:
        label = "single" if conditions else "qproj"
        conditions.append((label, Path(args.qpu_base)))

    if not conditions:
        latest = resolve_latest_qpu_base(Path(args.data_dir))
        if latest is None:
            raise FileNotFoundError("No input provided and no latest D_M qproj pointer found")
        conditions.append(("qproj", latest))

    for _, p in conditions:
        if not p.exists():
            raise FileNotFoundError(f"Input base does not exist: {p}")
    return conditions


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    conditions = input_conditions(args)

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_ANALYSIS_DIR / f"dm_probe_09_pi_phase_witness_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tile_rows: List[Dict[str, Any]] = []
    all_phase_rows: List[Dict[str, Any]] = []
    all_summaries: List[Dict[str, Any]] = []
    all_pi_adic_rows: List[Dict[str, Any]] = []
    metas: Dict[str, Any] = {}

    print("=" * 104)
    print("  GHOST ORACLE SUITE — D_M PROBE 09: π-PHASE / π-ADIC WITNESS")
    print("=" * 104)
    print(f"  Out dir     : {out_dir}")
    print(f"  n_perm      : {args.n_perm}")
    print(f"  π-adic toy  : levels=1..{args.pi_adic_levels} (experimental / made-up)")
    print("-" * 104)

    for cond, path in conditions:
        pair, plan, meta = load_base(path, cond, args)
        metas[cond] = meta
        tile_corrs = compute_tile_corrs(cond, pair, plan)
        phase_rows = phase_rung_rows(cond, tile_corrs)
        summary, pi_rows = summarize_condition(cond, phase_rows, args)

        all_tile_rows.extend([asdict(r) for r in tile_corrs])
        all_phase_rows.extend(phase_rows)
        all_summaries.append(summary)
        all_pi_adic_rows.extend(pi_rows)

        print(f"  CONDITION: {cond}")
        print(f"    input      : {path}")
        print(f"    backend    : {meta.get('backend')}  job={meta.get('job_id')}")
        print(f"    tiles/shots: {meta.get('num_tiles')} / {meta.get('shots')}  repaired={meta.get('metadata_repaired')}")
        print(f"    offset_dt  : {meta.get('offset_step_dt_used')}")
        print("    rung phase rows:")
        for r in phase_rows:
            print(
                f"      rung {int(r['rung_index']):02d} total={r['total_delay_dt_mean']:8.1f} "
                f"YZ={r['YZ_connected']:+.5f} ZY={r['ZY_connected']:+.5f} "
                f"R=-ZY={r['return_coord_minus_ZY']:+.5f} "
                f"E={r['yzzy_energy_euclidean']:.5f} "
                f"phase={r['pi_phase_degrees_mod180']:7.2f}° "
                f"spec={r['directional_specificity']:+.5f}"
            )
        print("    summary:")
        print(f"      energy tracking r,p       : {summary['total_delay_energy_r']:+.4f}, {summary['total_delay_energy_perm_p']:.4f}")
        print(f"      specificity tracking r,p  : {summary['total_delay_specificity_r']:+.4f}, {summary['total_delay_specificity_perm_p']:.4f}")
        print(f"      phase velocity r,p        : {summary['total_delay_phase_velocity_r']:+.4f}, {summary['total_delay_phase_velocity_perm_p']:.4f}")
        print(f"      π-periodic score,p        : {summary['pi_periodic_score']:.4f}, {summary['pi_periodic_perm_p']:.4f} ({summary['pi_periodic_mode']})")
        print(f"      π-adic toy best score,p   : {summary['pi_adic_best_score']:.4f}, {summary['pi_adic_best_perm_p']:.4f} level={summary['pi_adic_best_level']} mode={summary['pi_adic_best_mode']}")
        print("-" * 104)

    comp = comparator_summary(all_summaries)
    if comp:
        print("  COMPARATOR: offset_on - offset_off")
        for k, v in comp.items():
            if k in ("has_comparator", "interpretation_hint"):
                continue
            if isinstance(v, float):
                print(f"    {k:<42}: {v:+.6f}")
        print(f"    hint: {comp['interpretation_hint']}")
        print("-" * 104)

    tile_fields = list(all_tile_rows[0].keys()) if all_tile_rows else []
    phase_fields = list(all_phase_rows[0].keys()) if all_phase_rows else []
    summary_fields = list(all_summaries[0].keys()) if all_summaries else []
    pi_fields = list(all_pi_adic_rows[0].keys()) if all_pi_adic_rows else []

    if tile_fields:
        write_csv(out_dir / "tile_correlators.csv", all_tile_rows, tile_fields)
    if phase_fields:
        write_csv(out_dir / "phase_rungs.csv", all_phase_rows, phase_fields)
    if summary_fields:
        write_csv(out_dir / "phase_summary.csv", all_summaries, summary_fields)
    if pi_fields:
        write_csv(out_dir / "pi_adic_levels.csv", all_pi_adic_rows, pi_fields)
    if comp:
        write_csv(out_dir / "comparator_summary.csv", [comp], list(comp.keys()))

    maybe_make_plots(out_dir, all_phase_rows, all_summaries)

    result = {
        "operator": "D_M",
        "probe": "probe09_pi_phase_witness",
        "framing": "one D_M witness manifold with basis, distance, temporal-offset, and π-phase axes",
        "pi_adic_warning": "π-adic witness is a playful toy diagnostic, not formal p-adic mathematics.",
        "config": vars(args),
        "conditions": metas,
        "phase_summary": all_summaries,
        "comparator_summary": comp,
        "phase_rungs": all_phase_rows,
        "pi_adic_levels": all_pi_adic_rows,
        "elapsed_sec": time.perf_counter() - t0,
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2)

    print(f"  [SAVED] {out_dir}")
    print("=" * 104)


if __name__ == "__main__":
    main()
