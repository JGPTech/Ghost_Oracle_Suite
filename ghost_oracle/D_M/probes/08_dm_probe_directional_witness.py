#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — D_M PROBE 08: DIRECTIONAL YZ / ZY WITNESS LOCK
==============================================================================

Purpose
-------
Second D_M probe for the Bell-listener / cavity-offset qproj base.

Probe 01 asked the broad listener question:

    Do the four Bell-witness correlators light up, and do they track delay/offset?

Probe 08 narrows the discovered structure:

    YZ is treated as the primary Bell-witness channel.
    ZY is treated as the inverted / reciprocal side of that witness.

This script does NOT treat XY, YZ, ZY, YX as four symmetric peers. It tests a
more specific D_M hypothesis:

    The ghost channel is directional. The primary response is YZ, and ZY is the
    reciprocal/return channel that may align, amplify, or invert as cavity delay
    grows.

Interpretation discipline
-------------------------
This script reports Bell-witness correlation structure. It does NOT certify
entanglement or reconstruct a Bell state. The output is evidence for or against
a directional Bell-witness signature in the shared-chip ghost channel.

Inputs
------
A clean D_M qproj .npz base, preferably produced by the fixed D_M generator:

    pair                      uint8, shape (tiles, shots, 2)
    basis                     int8,  shape (tiles, 2)
    tile_base_delay_dt        int32, shape (tiles,)
    tile_offset_dt            int32, shape (tiles,)
    tile_total_delay_dt       int32, shape (tiles,)
    tile_basis_q0             int8,  shape (tiles,)
    tile_basis_q1             int8,  shape (tiles,)

The script can still repair metadata for old 20-tile runs, but that should be
used only for rescue analysis, not canonical fresh data.

What it computes
----------------
Per tile:
    raw correlator            <P0 P1>
    connected correlator      <P0 P1> - <P0><P1>
    independent q1-shuffle null for connected correlation

Per complete rung:
    YZ primary connected response
    ZY reciprocal connected response
    YZ/ZY pair energy
    YZ/ZY signed product
    YZ/ZY inversion score       = -YZ_connected * ZY_connected
    YZ minus ZY directional gap
    XY/YX comparison energy
    directional specificity     = YZ/ZY energy - XY/YX energy

Controls:
    1. independent_bit_shuffle:
       destroys shot-paired two-qubit relation while preserving q0/q1 marginals.

    2. witness_label_shuffle:
       shuffles XY/YZ/ZY/YX labels within each rung. Tests whether the specific
       YZ-primary / ZY-return interpretation is stronger than arbitrary labels.

    3. delay_order_permutation:
       permutes delay/order labels across rungs. Tests whether primary/return
       metrics track the cavity-delay ordering better than random rung order.

Outputs
-------
    analysis/dm_probe_08_directional_witness_<timestamp>/
        result.json
        tile_directional_stats.csv
        rung_directional_scores.csv
        control_summary.csv
        optional PNG plots if matplotlib is available

Usage
-----
From ghost_oracle/D_M:

    python probes/d_m_probe08_directional_witness.py \
      --qpu-base data/dm_data_bell_listener_cavity_offset_<JOB_ID>.npz

If latest_dm_qpu_data.json exists:

    python probes/d_m_probe08_directional_witness.py

For old broken bases only:

    python probes/d_m_probe08_directional_witness.py \
      --qpu-base data/dm_job_<JOB_ID>.npz \
      --force-repair
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


def resolve_latest_qpu_base(data_dir: Path) -> Optional[Path]:
    candidates = [data_dir / "latest_dm_qpu_data.json", data_dir / "latest_dm_data.json"]
    for latest in candidates:
        if not latest.exists():
            continue
        with open(latest, "r", encoding="utf-8") as f:
            obj = json.load(f)
        raw_path = obj.get("path", obj.get("npz", ""))
        p = Path(raw_path)
        if p.exists():
            return p
        fallback = data_dir / p.name
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
class TileStats:
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
    conn_null_mean: float
    conn_null_std: float
    conn_null_z: float
    conn_null_p_two_sided: float


# =============================================================================
# LOAD / REPAIR METADATA
# =============================================================================


def scalar_str(obj: Any) -> str:
    try:
        arr = np.asarray(obj)
        if arr.shape == ():
            return str(arr.item())
        return str(arr)
    except Exception:
        return str(obj)


def load_pair_stack(npz: Any) -> np.ndarray:
    """Load stacked pair array with shape (tiles, shots, 2)."""
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


def valid_int_array(a: np.ndarray, n: int, min_value: int = 0) -> bool:
    if a.shape[0] < n:
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
        and existing_base is not None
        and existing_offset is not None
        and existing_total is not None
        and existing_b0 is not None
        and existing_b1 is not None
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
            tile=t,
            base_delay_dt=base,
            offset_dt=off,
            total_delay_dt=total,
            basis_q0=int(b0),
            basis_q1=int(b1),
            witness=witness_label(int(b0), int(b1)),
            repaired=repaired,
        ))
    return plan


def load_base(path: Path, args: argparse.Namespace) -> Tuple[np.ndarray, List[TileMeta], Dict[str, Any]]:
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
# CORRELATORS
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


def conn_from_signs(s0: np.ndarray, s1: np.ndarray) -> float:
    raw = float(np.mean(s0 * s1))
    return float(raw - float(np.mean(s0)) * float(np.mean(s1)))


def independent_conn_null(bits: np.ndarray, n_null: int, rng: np.random.Generator) -> np.ndarray:
    s0, s1 = pair_to_signs(bits)
    vals = np.empty(n_null, dtype=np.float64)
    for i in range(n_null):
        vals[i] = conn_from_signs(s0, s1[rng.permutation(s1.shape[0])])
    return vals


def empirical_two_sided_p(obs: float, null: np.ndarray, center: Optional[float] = None) -> float:
    if null.size == 0:
        return 1.0
    mu = float(np.mean(null)) if center is None else float(center)
    return float((np.count_nonzero(np.abs(null - mu) >= abs(obs - mu)) + 1) / (null.size + 1))


def summarize_null(obs: float, null: np.ndarray) -> Tuple[float, float, float, float]:
    mu = float(np.mean(null)) if null.size else 0.0
    sd = float(np.std(null, ddof=1)) if null.size > 1 else 0.0
    z = float((obs - mu) / sd) if sd > 1e-12 else 0.0
    p = empirical_two_sided_p(obs, null, center=mu)
    return mu, sd, z, p


def compute_tile_stats(pair: np.ndarray, plan: Sequence[TileMeta], n_null: int, seed: int) -> List[TileStats]:
    rng = np.random.default_rng(seed)
    rows: List[TileStats] = []
    n_w = len(WITNESS_ORDER)
    for t, meta in enumerate(plan):
        bits = pair[t]
        raw, conn, m0, m1 = corr_from_bits(bits)
        null = independent_conn_null(bits, n_null=n_null, rng=rng)
        mu, sd, z, p = summarize_null(conn, null)
        rows.append(TileStats(
            tile=int(t),
            rung_index=int(t // n_w),
            witness_index=int(t % n_w),
            witness=meta.witness,
            base_delay_dt=int(meta.base_delay_dt),
            offset_dt=int(meta.offset_dt),
            total_delay_dt=int(meta.total_delay_dt),
            shots=int(bits.shape[0]),
            mean_q0=m0,
            mean_q1=m1,
            corr_raw=raw,
            corr_connected=conn,
            conn_null_mean=mu,
            conn_null_std=sd,
            conn_null_z=z,
            conn_null_p_two_sided=p,
        ))
    return rows


def safe_corrcoef(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rms(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    return float(math.sqrt(np.mean(arr * arr))) if arr.size else 0.0


# =============================================================================
# DIRECTIONAL RUNG METRICS
# =============================================================================


def rung_groups(tile_rows: Sequence[TileStats]) -> Dict[int, List[TileStats]]:
    groups: Dict[int, List[TileStats]] = {}
    for r in tile_rows:
        groups.setdefault(r.rung_index, []).append(r)
    return {k: sorted(v, key=lambda x: x.witness_index) for k, v in groups.items()}


def directional_metrics_from_conn(conn_by_w: Dict[str, float]) -> Dict[str, float]:
    xy = float(conn_by_w.get("XY", 0.0))
    yz = float(conn_by_w.get("YZ", 0.0))
    zy = float(conn_by_w.get("ZY", 0.0))
    yx = float(conn_by_w.get("YX", 0.0))

    yzzy_energy = rms([yz, zy])
    xyyx_energy = rms([xy, yx])
    product = float(yz * zy)
    inversion = float(-product)
    gap = float(yz - zy)
    sum_pair = float(yz + zy)
    specificity = float(yzzy_energy - xyyx_energy)
    dominance = float(abs(yz) - max(abs(xy), abs(yx)))

    return {
        "XY_connected": xy,
        "YZ_connected": yz,
        "ZY_connected": zy,
        "YX_connected": yx,
        "yz_primary": yz,
        "zy_return": zy,
        "yzzy_energy": yzzy_energy,
        "xyyx_energy": xyyx_energy,
        "yzzy_signed_product": product,
        "yzzy_inversion_score": inversion,
        "yz_minus_zy_gap": gap,
        "yz_plus_zy_sum": sum_pair,
        "directional_specificity": specificity,
        "yz_primary_dominance_vs_xy_yx": dominance,
        "zy_is_inverted_relative_to_yz": bool((yz * zy) < 0.0),
    }


def compute_rung_scores(tile_rows: Sequence[TileStats]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rung, tiles in rung_groups(tile_rows).items():
        conn_by_w = {tr.witness: tr.corr_connected for tr in tiles}
        raw_by_w = {tr.witness: tr.corr_raw for tr in tiles}
        complete = all(w in conn_by_w for w in WITNESS_ORDER)
        metrics = directional_metrics_from_conn(conn_by_w)
        row = {
            "rung_index": int(rung),
            "complete": bool(complete),
            "base_delay_dt": int(tiles[0].base_delay_dt) if tiles else -1,
            "mean_offset_dt": float(np.mean([tr.offset_dt for tr in tiles])) if tiles else -1.0,
            "mean_total_delay_dt": float(np.mean([tr.total_delay_dt for tr in tiles])) if tiles else -1.0,
            "XY_raw": float(raw_by_w.get("XY", 0.0)),
            "YZ_raw": float(raw_by_w.get("YZ", 0.0)),
            "ZY_raw": float(raw_by_w.get("ZY", 0.0)),
            "YX_raw": float(raw_by_w.get("YX", 0.0)),
            **metrics,
        }
        rows.append(row)
    return rows


def perm_tracking_p(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> Tuple[float, float]:
    obs = safe_corrcoef(x, y)
    if x.size < 3:
        return obs, 1.0
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        null[i] = safe_corrcoef(rng.permutation(x), y)
    p = float((np.count_nonzero(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, p


def global_directional_summary(rung_rows: Sequence[Dict[str, Any]], n_perm: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 404)
    complete = [r for r in rung_rows if r.get("complete")]
    if not complete:
        return {"complete_rungs": 0}

    x_base = np.asarray([math.log1p(float(r["base_delay_dt"])) for r in complete], dtype=np.float64)
    x_total = np.asarray([math.log1p(float(r["mean_total_delay_dt"])) for r in complete], dtype=np.float64)

    metric_names = [
        "yz_primary",
        "zy_return",
        "yzzy_energy",
        "yzzy_inversion_score",
        "yz_minus_zy_gap",
        "directional_specificity",
    ]

    summary: Dict[str, Any] = {
        "complete_rungs": int(len(complete)),
        "yz_positive_fraction": float(np.mean([float(r["yz_primary"]) > 0.0 for r in complete])),
        "zy_inverted_fraction": float(np.mean([bool(r["zy_is_inverted_relative_to_yz"]) for r in complete])),
        "yz_primary_mean": float(np.mean([float(r["yz_primary"]) for r in complete])),
        "zy_return_mean": float(np.mean([float(r["zy_return"]) for r in complete])),
        "yzzy_energy_mean": float(np.mean([float(r["yzzy_energy"]) for r in complete])),
        "yzzy_energy_max": float(np.max([float(r["yzzy_energy"]) for r in complete])),
        "directional_specificity_mean": float(np.mean([float(r["directional_specificity"]) for r in complete])),
        "directional_specificity_max": float(np.max([float(r["directional_specificity"]) for r in complete])),
        "late_rung_index": int(complete[-1]["rung_index"]),
        "late_yz_primary": float(complete[-1]["yz_primary"]),
        "late_zy_return": float(complete[-1]["zy_return"]),
        "late_yzzy_inversion_score": float(complete[-1]["yzzy_inversion_score"]),
    }

    for metric in metric_names:
        y = np.asarray([float(r[metric]) for r in complete], dtype=np.float64)
        rb, pb = perm_tracking_p(x_base, y, n_perm=n_perm, rng=rng)
        rt, pt = perm_tracking_p(x_total, y, n_perm=n_perm, rng=rng)
        summary[f"base_delay_{metric}_r"] = rb
        summary[f"base_delay_{metric}_p"] = pb
        summary[f"total_delay_{metric}_r"] = rt
        summary[f"total_delay_{metric}_p"] = pt

    return summary


# =============================================================================
# DESTRUCTIVE CONTROLS
# =============================================================================


def compute_rung_scores_from_conn_rows(conn_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_rung: Dict[int, List[Dict[str, Any]]] = {}
    for row in conn_rows:
        by_rung.setdefault(int(row["rung_index"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for rung, items in sorted(by_rung.items()):
        conn_by_w = {str(it["witness"]): float(it["corr_connected"]) for it in items}
        base = int(items[0]["base_delay_dt"])
        mean_off = float(np.mean([float(it["offset_dt"]) for it in items]))
        mean_total = float(np.mean([float(it["total_delay_dt"]) for it in items]))
        complete = all(w in conn_by_w for w in WITNESS_ORDER)
        out.append({
            "rung_index": rung,
            "complete": complete,
            "base_delay_dt": base,
            "mean_offset_dt": mean_off,
            "mean_total_delay_dt": mean_total,
            **directional_metrics_from_conn(conn_by_w),
        })
    return out


def independent_shuffle_global_null(
    pair: np.ndarray,
    tile_rows: Sequence[TileStats],
    n_null: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Destroy q0/q1 shot pairing in every tile, preserving marginals."""
    rng = np.random.default_rng(seed + 505)
    metrics = {
        "yzzy_energy_mean": np.empty(n_null, dtype=np.float64),
        "directional_specificity_mean": np.empty(n_null, dtype=np.float64),
        "total_delay_yzzy_energy_r": np.empty(n_null, dtype=np.float64),
        "total_delay_directional_specificity_r": np.empty(n_null, dtype=np.float64),
    }

    for i in range(n_null):
        conn_rows: List[Dict[str, Any]] = []
        for tr in tile_rows:
            bits = pair[tr.tile]
            s0, s1 = pair_to_signs(bits)
            conn = conn_from_signs(s0, s1[rng.permutation(s1.shape[0])])
            conn_rows.append({
                "rung_index": tr.rung_index,
                "witness": tr.witness,
                "base_delay_dt": tr.base_delay_dt,
                "offset_dt": tr.offset_dt,
                "total_delay_dt": tr.total_delay_dt,
                "corr_connected": conn,
            })
        rr = [r for r in compute_rung_scores_from_conn_rows(conn_rows) if r.get("complete")]
        if not rr:
            for k in metrics:
                metrics[k][i] = 0.0
            continue
        metrics["yzzy_energy_mean"][i] = float(np.mean([float(r["yzzy_energy"]) for r in rr]))
        metrics["directional_specificity_mean"][i] = float(np.mean([float(r["directional_specificity"]) for r in rr]))
        x_total = np.asarray([math.log1p(float(r["mean_total_delay_dt"])) for r in rr], dtype=np.float64)
        metrics["total_delay_yzzy_energy_r"][i] = safe_corrcoef(x_total, [float(r["yzzy_energy"]) for r in rr])
        metrics["total_delay_directional_specificity_r"][i] = safe_corrcoef(x_total, [float(r["directional_specificity"]) for r in rr])
    return metrics


def witness_label_shuffle_null(
    rung_rows: Sequence[Dict[str, Any]],
    n_null: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Shuffle labels within each rung while preserving four rung values."""
    rng = np.random.default_rng(seed + 606)
    complete = [r for r in rung_rows if r.get("complete")]
    metrics = {
        "yzzy_energy_mean": np.empty(n_null, dtype=np.float64),
        "directional_specificity_mean": np.empty(n_null, dtype=np.float64),
        "total_delay_yzzy_energy_r": np.empty(n_null, dtype=np.float64),
        "total_delay_directional_specificity_r": np.empty(n_null, dtype=np.float64),
    }
    if not complete:
        for k in metrics:
            metrics[k].fill(0.0)
        return metrics

    for i in range(n_null):
        rr: List[Dict[str, Any]] = []
        for r in complete:
            vals = np.asarray([
                float(r["XY_connected"]),
                float(r["YZ_connected"]),
                float(r["ZY_connected"]),
                float(r["YX_connected"]),
            ], dtype=np.float64)
            perm = rng.permutation(vals)
            conn_by_w = dict(zip(WITNESS_ORDER, perm))
            rr.append({
                "rung_index": int(r["rung_index"]),
                "complete": True,
                "base_delay_dt": int(r["base_delay_dt"]),
                "mean_total_delay_dt": float(r["mean_total_delay_dt"]),
                **directional_metrics_from_conn(conn_by_w),
            })
        metrics["yzzy_energy_mean"][i] = float(np.mean([float(r["yzzy_energy"]) for r in rr]))
        metrics["directional_specificity_mean"][i] = float(np.mean([float(r["directional_specificity"]) for r in rr]))
        x_total = np.asarray([math.log1p(float(r["mean_total_delay_dt"])) for r in rr], dtype=np.float64)
        metrics["total_delay_yzzy_energy_r"][i] = safe_corrcoef(x_total, [float(r["yzzy_energy"]) for r in rr])
        metrics["total_delay_directional_specificity_r"][i] = safe_corrcoef(x_total, [float(r["directional_specificity"]) for r in rr])
    return metrics


def summarize_control(control_name: str, observed: Dict[str, float], nulls: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, obs in observed.items():
        null = nulls.get(metric, np.asarray([], dtype=np.float64))
        mu, sd, z, _ = summarize_null(float(obs), null)
        # For positive evidence metrics, use upper-tail as well as two-sided.
        upper_p = float((np.count_nonzero(null >= float(obs)) + 1) / (null.size + 1)) if null.size else 1.0
        two_p = empirical_two_sided_p(float(obs), null, center=mu)
        rows.append({
            "control": control_name,
            "metric": metric,
            "observed": float(obs),
            "null_mean": mu,
            "null_std": sd,
            "z": z,
            "p_upper": upper_p,
            "p_two_sided": two_p,
        })
    return rows


def run_controls(
    pair: np.ndarray,
    tile_rows: Sequence[TileStats],
    rung_rows: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    n_null: int,
    seed: int,
) -> List[Dict[str, Any]]:
    observed = {
        "yzzy_energy_mean": float(summary.get("yzzy_energy_mean", 0.0)),
        "directional_specificity_mean": float(summary.get("directional_specificity_mean", 0.0)),
        "total_delay_yzzy_energy_r": float(summary.get("total_delay_yzzy_energy_r", 0.0)),
        "total_delay_directional_specificity_r": float(summary.get("total_delay_directional_specificity_r", 0.0)),
    }
    rows: List[Dict[str, Any]] = []

    bit_nulls = independent_shuffle_global_null(pair, tile_rows, n_null=n_null, seed=seed)
    rows.extend(summarize_control("independent_bit_shuffle", observed, bit_nulls))

    label_nulls = witness_label_shuffle_null(rung_rows, n_null=n_null, seed=seed)
    rows.extend(summarize_control("witness_label_shuffle", observed, label_nulls))

    # Delay/order permutation controls are already computed in summary; surface them.
    rows.extend([
        {
            "control": "delay_order_permutation",
            "metric": "total_delay_yz_primary_r",
            "observed": float(summary.get("total_delay_yz_primary_r", 0.0)),
            "null_mean": "",
            "null_std": "",
            "z": "",
            "p_upper": "",
            "p_two_sided": float(summary.get("total_delay_yz_primary_p", 1.0)),
        },
        {
            "control": "delay_order_permutation",
            "metric": "total_delay_zy_return_r",
            "observed": float(summary.get("total_delay_zy_return_r", 0.0)),
            "null_mean": "",
            "null_std": "",
            "z": "",
            "p_upper": "",
            "p_two_sided": float(summary.get("total_delay_zy_return_p", 1.0)),
        },
        {
            "control": "delay_order_permutation",
            "metric": "total_delay_yzzy_energy_r",
            "observed": float(summary.get("total_delay_yzzy_energy_r", 0.0)),
            "null_mean": "",
            "null_std": "",
            "z": "",
            "p_upper": "",
            "p_two_sided": float(summary.get("total_delay_yzzy_energy_p", 1.0)),
        },
        {
            "control": "delay_order_permutation",
            "metric": "total_delay_directional_specificity_r",
            "observed": float(summary.get("total_delay_directional_specificity_r", 0.0)),
            "null_mean": "",
            "null_std": "",
            "z": "",
            "p_upper": "",
            "p_two_sided": float(summary.get("total_delay_directional_specificity_p", 1.0)),
        },
    ])
    return rows


# =============================================================================
# OPTIONAL PLOTS
# =============================================================================


def maybe_write_plots(out_dir: Path, rung_rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    complete = [r for r in rung_rows if r.get("complete")]
    if not complete:
        return

    x = [float(r["mean_total_delay_dt"]) for r in complete]
    yz = [float(r["yz_primary"]) for r in complete]
    zy = [float(r["zy_return"]) for r in complete]
    energy = [float(r["yzzy_energy"]) for r in complete]
    spec = [float(r["directional_specificity"]) for r in complete]

    plt.figure()
    plt.axhline(0.0, linewidth=1)
    plt.plot(x, yz, marker="o", label="YZ primary")
    plt.plot(x, zy, marker="o", label="ZY return")
    plt.xscale("symlog")
    plt.xlabel("mean_total_delay_dt")
    plt.ylabel("connected correlator")
    plt.title("D_M Probe 08 — YZ primary / ZY return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "yz_zy_connected_vs_total_delay.png", dpi=160)
    plt.close()

    plt.figure()
    plt.axhline(0.0, linewidth=1)
    plt.plot(x, energy, marker="o", label="YZ/ZY energy")
    plt.plot(x, spec, marker="o", label="directional specificity")
    plt.xscale("symlog")
    plt.xlabel("mean_total_delay_dt")
    plt.ylabel("score")
    plt.title("D_M Probe 08 — directional witness scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "directional_scores_vs_total_delay.png", dpi=160)
    plt.close()


# =============================================================================
# CLI / MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D_M Probe 08 — directional YZ-primary / ZY-inverted witness analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qpu-base", default=str(DEFAULT_QPROJ_OFFSET), help="Path to D_M qproj .npz. Defaults to latest_dm_qpu_data.json/latest_dm_data.json.")
    p.add_argument("--base-delays-dt", type=int, nargs="+", default=DEFAULT_BASE_DELAYS_DT,
                   help="Base-delay rungs used for old metadata repair.")
    p.add_argument("--offset-dt", type=int, default=DEFAULT_OFFSET_DT,
                   help="Offset step in dt used for old metadata repair.")
    p.add_argument("--force-offset", action="store_true",
                   help="Use --offset-dt even if the npz has an offset_dt field.")
    p.add_argument("--force-repair", action="store_true",
                   help="Ignore stored tile metadata and reconstruct the intended D_M tile plan.")
    p.add_argument("--n-null", type=int, default=5000,
                   help="Number of destructive-control null shuffles / permutations.")
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    qpu_base = Path(args.qpu_base) if args.qpu_base else resolve_latest_qpu_base(DEFAULT_DATA_DIR)
    if qpu_base is None:
        raise FileNotFoundError(
            "No --qpu-base provided and latest D_M data pointer could not be resolved. "
            "Pass --qpu-base data/dm_data_bell_listener_cavity_offset_<JOB_ID>.npz"
        )
    qpu_base = qpu_base.expanduser().resolve()
    if not qpu_base.exists():
        raise FileNotFoundError(qpu_base)

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_ANALYSIS_DIR / f"dm_probe_08_directional_witness_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair, plan, base_meta = load_base(qpu_base, args)
    tile_rows = compute_tile_stats(pair, plan, n_null=args.n_null, seed=args.seed)
    rung_rows = compute_rung_scores(tile_rows)
    summary = global_directional_summary(rung_rows, n_perm=args.n_null, seed=args.seed)
    control_rows = run_controls(pair, tile_rows, rung_rows, summary, n_null=args.n_null, seed=args.seed)

    tile_csv_rows = [asdict(r) for r in tile_rows]
    meta_csv_rows = [asdict(m) for m in plan]

    write_csv(out_dir / "tile_directional_stats.csv", tile_csv_rows, [
        "tile", "rung_index", "witness_index", "witness", "base_delay_dt", "offset_dt", "total_delay_dt", "shots",
        "mean_q0", "mean_q1", "corr_raw", "corr_connected",
        "conn_null_mean", "conn_null_std", "conn_null_z", "conn_null_p_two_sided",
    ])
    write_csv(out_dir / "rung_directional_scores.csv", rung_rows, [
        "rung_index", "complete", "base_delay_dt", "mean_offset_dt", "mean_total_delay_dt",
        "XY_raw", "YZ_raw", "ZY_raw", "YX_raw",
        "XY_connected", "YZ_connected", "ZY_connected", "YX_connected",
        "yz_primary", "zy_return", "yzzy_energy", "xyyx_energy",
        "yzzy_signed_product", "yzzy_inversion_score", "yz_minus_zy_gap", "yz_plus_zy_sum",
        "directional_specificity", "yz_primary_dominance_vs_xy_yx", "zy_is_inverted_relative_to_yz",
    ])
    write_csv(out_dir / "control_summary.csv", control_rows, [
        "control", "metric", "observed", "null_mean", "null_std", "z", "p_upper", "p_two_sided",
    ])
    write_csv(out_dir / "metadata_used.csv", meta_csv_rows, [
        "tile", "base_delay_dt", "offset_dt", "total_delay_dt", "basis_q0", "basis_q1", "witness", "repaired",
    ])

    if not args.no_plots:
        maybe_write_plots(out_dir, rung_rows)

    complete = [r for r in rung_rows if r.get("complete")]
    top_tiles = sorted(tile_csv_rows, key=lambda r: abs(float(r["corr_connected"])), reverse=True)[:10]
    top_rungs = sorted(rung_rows, key=lambda r: float(r.get("yzzy_energy", 0.0)), reverse=True)[:10]

    result = {
        "operator": "D_M",
        "probe": "D_M Probe 08 — Directional YZ / ZY Witness Lock",
        "framing": (
            "YZ is treated as the primary Bell-witness channel; ZY is treated as "
            "the inverted/reciprocal witness side. This is not a symmetric four-channel rank scan."
        ),
        "input": base_meta,
        "config": vars(args),
        "global_summary": summary,
        "rung_directional_scores": rung_rows,
        "controls": control_rows,
        "top_tiles_by_abs_connected": top_tiles,
        "top_rungs_by_yzzy_energy": top_rungs,
        "elapsed_sec": time.perf_counter() - t0,
        "outputs": {
            "out_dir": str(out_dir),
            "tile_directional_stats_csv": str(out_dir / "tile_directional_stats.csv"),
            "rung_directional_scores_csv": str(out_dir / "rung_directional_scores.csv"),
            "control_summary_csv": str(out_dir / "control_summary.csv"),
            "metadata_used_csv": str(out_dir / "metadata_used.csv"),
        },
        "bounded_claim_note": (
            "Probe 08 can support a directional Bell-witness signature in the ghost channel. "
            "It does not by itself certify entanglement. A fresh offset-off QPU control is still needed "
            "to lock the cavity-offset mechanism experimentally."
        ),
    }

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2)

    print("=" * 100)
    print("  GHOST ORACLE SUITE — D_M PROBE 08: DIRECTIONAL YZ / ZY WITNESS LOCK")
    print("=" * 100)
    print(f"  Input     : {qpu_base}")
    print(f"  Backend   : {base_meta.get('backend')}")
    print(f"  Job ID    : {base_meta.get('job_id')}")
    print(f"  Tiles     : {base_meta.get('num_tiles')}  Shots: {base_meta.get('shots')}")
    print(f"  Repaired  : {base_meta.get('metadata_repaired')}  offset_step_dt={base_meta.get('offset_step_dt_used')}")
    print(f"  Out dir   : {out_dir}")
    print("-" * 100)
    print("  RUNG DIRECTIONAL SCORES")
    for r in rung_rows:
        print(
            f"  rung {int(r['rung_index']):02d} base={int(r['base_delay_dt']):6d} "
            f"total={float(r['mean_total_delay_dt']):8.1f} "
            f"YZ={float(r['yz_primary']):+.5f} ZY={float(r['zy_return']):+.5f} "
            f"energy={float(r['yzzy_energy']):.5f} inv={float(r['yzzy_inversion_score']):+.5f} "
            f"gap={float(r['yz_minus_zy_gap']):+.5f} spec={float(r['directional_specificity']):+.5f} "
            f"inverted={bool(r['zy_is_inverted_relative_to_yz'])}"
        )
    print("-" * 100)
    print("  GLOBAL DIRECTIONAL SUMMARY")
    print(f"  complete_rungs                       : {summary.get('complete_rungs', 0)}")
    print(f"  YZ positive fraction                 : {summary.get('yz_positive_fraction', 0.0):.4f}")
    print(f"  ZY inverted fraction vs YZ           : {summary.get('zy_inverted_fraction', 0.0):.4f}")
    print(f"  YZ primary mean                      : {summary.get('yz_primary_mean', 0.0):+.6f}")
    print(f"  ZY return mean                       : {summary.get('zy_return_mean', 0.0):+.6f}")
    print(f"  YZ/ZY energy mean/max                : {summary.get('yzzy_energy_mean', 0.0):.6f} / {summary.get('yzzy_energy_max', 0.0):.6f}")
    print(f"  directional specificity mean/max     : {summary.get('directional_specificity_mean', 0.0):+.6f} / {summary.get('directional_specificity_max', 0.0):+.6f}")
    print(f"  total_delay YZ-primary tracking r,p  : {summary.get('total_delay_yz_primary_r', 0.0):+.4f}, {summary.get('total_delay_yz_primary_p', 1.0):.4f}")
    print(f"  total_delay ZY-return tracking r,p   : {summary.get('total_delay_zy_return_r', 0.0):+.4f}, {summary.get('total_delay_zy_return_p', 1.0):.4f}")
    print(f"  total_delay YZ/ZY energy r,p         : {summary.get('total_delay_yzzy_energy_r', 0.0):+.4f}, {summary.get('total_delay_yzzy_energy_p', 1.0):.4f}")
    print(f"  total_delay specificity r,p          : {summary.get('total_delay_directional_specificity_r', 0.0):+.4f}, {summary.get('total_delay_directional_specificity_p', 1.0):.4f}")
    print(f"  late rung YZ / ZY / inversion        : {summary.get('late_yz_primary', 0.0):+.6f} / {summary.get('late_zy_return', 0.0):+.6f} / {summary.get('late_yzzy_inversion_score', 0.0):+.6f}")
    print("-" * 100)
    print("  CONTROLS")
    for c in control_rows:
        print(
            f"  {str(c['control']):<28} {str(c['metric']):<42} "
            f"obs={c.get('observed')} z={c.get('z')} p2={c.get('p_two_sided')} pup={c.get('p_upper')}"
        )
    print("-" * 100)
    print(f"  [SAVED] {out_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
