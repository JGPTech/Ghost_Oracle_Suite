#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — D_M PROBE 07: BELL LISTENER / CAVITY OFFSET
==============================================================================

Purpose
-------
Proper first probe for the D_M Bell-listener qproj base.

D_M is NOT treated here as effective rank, reduced density reconstruction, or
an intrinsic-dimension scan. This probe treats D_M as:

    a bare shared-chip Bell-witness listener

The QPU base contains 2-qubit tiles. Each tile measures one Pauli-pair witness:

    XY, YZ, ZY, YX

The experiment listens for whether the ghost channel produces two-qubit
correlation in those four Bell-witness correlators, and whether the correlation
tracks the deliberate cavity-delay offset / base-delay ladder.

Important interpretation discipline
-----------------------------------
This probe reports Bell-witness correlation. It does NOT claim certified Bell
entanglement by itself. Certification would require a stricter witness/CHSH or
tomographic validation circuit. This script is for discovery and controls.

What it does
------------
1. Loads a D_M qproj .npz base.
2. Repairs missing submit metadata if needed.
   For the current 20-tile run, the intended layout is assumed to be:

       5 base-delay rungs × 4 witness bases = 20 tiles
       witness order per rung = XY, YZ, ZY, YX
       total_delay_dt = base_delay_dt + tile_index * offset_dt

3. Computes per-tile two-qubit Pauli correlators:

       <P0 ⊗ P1> = mean((+1/-1 from q0) * (+1/-1 from q1))

4. Also computes connected correlators:

       C_conn = <P0P1> - <P0><P1>

   This helps separate true pair correlation from simple readout/marginal bias.

5. Groups the four witnesses into complete rungs and scores each rung:

       bell_rms_raw       = sqrt(mean(corr^2))
       bell_mean_abs_raw  = mean(abs(corr))
       bell_rms_connected = sqrt(mean(conn_corr^2))
       reciprocal products XY*YX and YZ*ZY

6. Runs destructive controls:

       independent_bit_shuffle:
           shuffle q1 shots inside each tile, preserving q0/q1 marginals while
           destroying shot-paired two-qubit correlation.

       rung_label_shuffle:
           shuffle witness labels inside complete rungs. This does not destroy
           tile-level correlation, but checks whether the 4-witness structure is
           being interpreted as a coherent Bell-witness block.

       offset_tracking_permutation:
           tests whether rung Bell score tracks log1p(base_delay_dt) more than
           random ordering would suggest.

Outputs
-------
    analysis/dm_probe_07_bell_listener_<timestamp>/
        result.json
        tile_correlators.csv
        rung_bell_scores.csv
        control_summary.csv
        repaired_metadata.csv
        optional PNG plots if matplotlib is available

Usage
-----
From ghost_oracle/D_M:

    python probes/d_m_probe07_bell_listener.py \
      --qpu-base data/dm_job_d8fl0787jphs739mehdg.npz

If latest_dm_qpu_data.json exists, you can usually run:

    python probes/d_m_probe07_bell_listener.py

If the submit metadata was missing, use the intended run parameters:

    python probes/d_m_probe07_bell_listener.py \
      --qpu-base data/dm_job_d8fl0787jphs739mehdg.npz \
      --base-delays-dt 0 256 1024 4096 16384 \
      --offset-dt 128
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
    latest = data_dir / "latest_dm_qpu_data.json"
    if not latest.exists():
        return None
    with open(latest, "r", encoding="utf-8") as f:
        obj = json.load(f)
    p = Path(obj.get("path", ""))
    if p.exists():
        return p
    # If pointer was written on another machine / relative path changed.
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
    p00: float
    p01: float
    p10: float
    p11: float
    mean_q0: float
    mean_q1: float
    corr_raw: float
    corr_connected: float
    corr_se: float
    corr_abs: float
    null_mean: float
    null_std: float
    null_z: float
    null_p_two_sided: float


# =============================================================================
# LOAD / REPAIR METADATA
# =============================================================================


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
    """
    Repair missing metadata. The current 20-tile dump had no local submit meta,
    so tile basis/delays were saved as -1. This reconstructs the intended
    submit ordering used by d_m_qpu_generate.py:

        for base_delay in base_delays:
            for witness in [XY, YZ, ZY, YX]:
                tile t has total_delay = base_delay + t * offset_step_dt
    """
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
            if rung < len(base_delays_dt):
                base = int(base_delays_dt[rung])
            else:
                # Continue the pattern if there are more tiles than expected.
                base = int(base_delays_dt[-1])
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


def read_optional(npz: Any, key: str) -> Optional[np.ndarray]:
    if key not in npz.files:
        return None
    return np.asarray(npz[key])


def load_base(path: Path, args: argparse.Namespace) -> Tuple[np.ndarray, List[TileMeta], Dict[str, Any]]:
    obj = np.load(path, allow_pickle=True)
    pair = load_pair_stack(obj)
    n_tiles, shots, width = pair.shape
    assert width == 2

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
        "schema": str(obj["schema"]) if "schema" in obj.files else "unknown",
        "operator": str(obj["operator"]) if "operator" in obj.files else "D_M",
        "substrate": str(obj["substrate"]) if "substrate" in obj.files else "qproj",
        "job_id": str(obj["job_id"]) if "job_id" in obj.files else path.stem,
        "backend": str(obj["backend"]) if "backend" in obj.files else "unknown",
        "num_tiles": int(n_tiles),
        "shots": int(shots),
        "metadata_repaired": bool(any(m.repaired for m in plan)),
        "base_delays_dt_used": list(map(int, args.base_delays_dt)),
        "offset_step_dt_used": int(offset_step_dt),
        "witness_order_used": [witness_label(a, b) for a, b in WITNESS_PAIRS],
    }
    return pair, plan, meta


# =============================================================================
# CORRELATORS / CONTROLS
# =============================================================================


def pair_to_signs(bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map measurement bits {0,1} to Pauli eigenvalue signs {+1,-1}."""
    b = np.asarray(bits, dtype=np.uint8)
    s0 = 1.0 - 2.0 * b[:, 0].astype(np.float64)
    s1 = 1.0 - 2.0 * b[:, 1].astype(np.float64)
    return s0, s1


def joint_probs(bits: np.ndarray) -> Tuple[float, float, float, float]:
    b0 = bits[:, 0]
    b1 = bits[:, 1]
    n = max(1, bits.shape[0])
    p00 = float(np.count_nonzero((b0 == 0) & (b1 == 0)) / n)
    p01 = float(np.count_nonzero((b0 == 0) & (b1 == 1)) / n)
    p10 = float(np.count_nonzero((b0 == 1) & (b1 == 0)) / n)
    p11 = float(np.count_nonzero((b0 == 1) & (b1 == 1)) / n)
    return p00, p01, p10, p11


def corr_from_bits(bits: np.ndarray) -> Tuple[float, float, float, float]:
    s0, s1 = pair_to_signs(bits)
    m0 = float(np.mean(s0))
    m1 = float(np.mean(s1))
    raw = float(np.mean(s0 * s1))
    conn = float(raw - m0 * m1)
    return raw, conn, m0, m1


def independent_shuffle_null(bits: np.ndarray, n_null: int, rng: np.random.Generator) -> np.ndarray:
    """
    Null: preserve q0/q1 marginal distributions, destroy shot-paired relation.
    """
    s0, s1 = pair_to_signs(bits)
    vals = np.empty(n_null, dtype=np.float64)
    for i in range(n_null):
        perm = rng.permutation(s1.shape[0])
        vals[i] = float(np.mean(s0 * s1[perm]))
    return vals


def summarize_null(obs: float, null: np.ndarray) -> Tuple[float, float, float, float]:
    mu = float(np.mean(null)) if null.size else 0.0
    sd = float(np.std(null, ddof=1)) if null.size > 1 else 0.0
    z = float((obs - mu) / sd) if sd > 1e-12 else 0.0
    p = float((np.count_nonzero(np.abs(null - mu) >= abs(obs - mu)) + 1) / (null.size + 1)) if null.size else 1.0
    return mu, sd, z, p


def compute_tile_stats(pair: np.ndarray, plan: Sequence[TileMeta], n_null: int, seed: int) -> List[TileStats]:
    rng = np.random.default_rng(seed)
    rows: List[TileStats] = []
    n_w = len(WITNESS_PAIRS)

    for t, meta in enumerate(plan):
        bits = pair[t]
        raw, conn, m0, m1 = corr_from_bits(bits)
        p00, p01, p10, p11 = joint_probs(bits)
        se = math.sqrt(max(0.0, 1.0 - raw * raw) / max(1, bits.shape[0] - 1))
        null = independent_shuffle_null(bits, n_null=n_null, rng=rng)
        null_mu, null_sd, z, p = summarize_null(raw, null)
        rows.append(TileStats(
            tile=int(t),
            rung_index=int(t // n_w),
            witness_index=int(t % n_w),
            witness=meta.witness,
            base_delay_dt=int(meta.base_delay_dt),
            offset_dt=int(meta.offset_dt),
            total_delay_dt=int(meta.total_delay_dt),
            shots=int(bits.shape[0]),
            p00=p00,
            p01=p01,
            p10=p10,
            p11=p11,
            mean_q0=m0,
            mean_q1=m1,
            corr_raw=raw,
            corr_connected=conn,
            corr_se=se,
            corr_abs=abs(raw),
            null_mean=null_mu,
            null_std=null_sd,
            null_z=z,
            null_p_two_sided=p,
        ))
    return rows


def rung_groups(tile_rows: Sequence[TileStats]) -> Dict[int, List[TileStats]]:
    groups: Dict[int, List[TileStats]] = {}
    for r in tile_rows:
        groups.setdefault(r.rung_index, []).append(r)
    return {k: sorted(v, key=lambda x: x.witness_index) for k, v in groups.items()}


def bell_rms(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    return float(math.sqrt(np.mean(arr * arr))) if arr.size else 0.0


def mean_abs(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(np.abs(arr))) if arr.size else 0.0


def safe_corrcoef(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rung_null_score(pair: np.ndarray, tiles: Sequence[TileStats], n_null: int, rng: np.random.Generator) -> np.ndarray:
    vals = np.empty(n_null, dtype=np.float64)
    for i in range(n_null):
        cs = []
        for tr in tiles:
            bits = pair[tr.tile]
            s0, s1 = pair_to_signs(bits)
            cs.append(float(np.mean(s0 * s1[rng.permutation(s1.shape[0])])))
        vals[i] = bell_rms(cs)
    return vals


def compute_rung_scores(
    pair: np.ndarray,
    tile_rows: Sequence[TileStats],
    n_null: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(seed + 101)
    rows: List[Dict[str, Any]] = []

    for rung, tiles in rung_groups(tile_rows).items():
        if len(tiles) != len(WITNESS_PAIRS):
            complete = False
        else:
            complete = True

        raw_by_w = {tr.witness: tr.corr_raw for tr in tiles}
        conn_by_w = {tr.witness: tr.corr_connected for tr in tiles}
        raw_vec = [raw_by_w.get(label, 0.0) for label in ["XY", "YZ", "ZY", "YX"]]
        conn_vec = [conn_by_w.get(label, 0.0) for label in ["XY", "YZ", "ZY", "YX"]]

        rms_raw = bell_rms(raw_vec)
        rms_conn = bell_rms(conn_vec)
        null = rung_null_score(pair, tiles, n_null=n_null, rng=rng) if complete else np.zeros(n_null)
        null_mu, null_sd, null_z, null_p = summarize_null(rms_raw, null)

        xy = raw_by_w.get("XY", 0.0)
        yx = raw_by_w.get("YX", 0.0)
        yz = raw_by_w.get("YZ", 0.0)
        zy = raw_by_w.get("ZY", 0.0)

        row = {
            "rung_index": rung,
            "complete": bool(complete),
            "base_delay_dt": int(tiles[0].base_delay_dt) if tiles else -1,
            "mean_offset_dt": float(np.mean([tr.offset_dt for tr in tiles])) if tiles else -1,
            "mean_total_delay_dt": float(np.mean([tr.total_delay_dt for tr in tiles])) if tiles else -1,
            "XY": xy,
            "YZ": yz,
            "ZY": zy,
            "YX": yx,
            "XY_connected": conn_by_w.get("XY", 0.0),
            "YZ_connected": conn_by_w.get("YZ", 0.0),
            "ZY_connected": conn_by_w.get("ZY", 0.0),
            "YX_connected": conn_by_w.get("YX", 0.0),
            "bell_rms_raw": rms_raw,
            "bell_mean_abs_raw": mean_abs(raw_vec),
            "bell_rms_connected": rms_conn,
            "bell_mean_abs_connected": mean_abs(conn_vec),
            "reciprocal_XY_YX_product": float(xy * yx),
            "reciprocal_YZ_ZY_product": float(yz * zy),
            "reciprocal_product_mean": float(0.5 * (xy * yx + yz * zy)),
            "null_mean": null_mu,
            "null_std": null_sd,
            "null_z": null_z,
            "null_p_two_sided": null_p,
        }
        rows.append(row)

    # Delay/offset tracking: does rung Bell score track delay ordering?
    complete_rows = [r for r in rows if r["complete"]]
    x_base = np.asarray([math.log1p(r["base_delay_dt"]) for r in complete_rows], dtype=np.float64)
    x_total = np.asarray([math.log1p(r["mean_total_delay_dt"]) for r in complete_rows], dtype=np.float64)
    y_raw = np.asarray([r["bell_rms_raw"] for r in complete_rows], dtype=np.float64)
    y_conn = np.asarray([r["bell_rms_connected"] for r in complete_rows], dtype=np.float64)

    def perm_p(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> Tuple[float, float]:
        obs = safe_corrcoef(x, y)
        if x.size < 3:
            return obs, 1.0
        null = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            null[i] = safe_corrcoef(rng.permutation(x), y)
        p = float((np.count_nonzero(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
        return obs, p

    tracking = {}
    for name, x in [("base_delay", x_base), ("total_delay", x_total)]:
        r_raw, p_raw = perm_p(x, y_raw, n_null, rng)
        r_conn, p_conn = perm_p(x, y_conn, n_null, rng)
        tracking[f"{name}_corr_raw_r"] = r_raw
        tracking[f"{name}_corr_raw_perm_p"] = p_raw
        tracking[f"{name}_corr_connected_r"] = r_conn
        tracking[f"{name}_corr_connected_perm_p"] = p_conn

    global_summary = {
        "complete_rungs": int(len(complete_rows)),
        "global_bell_rms_raw_mean": float(np.mean(y_raw)) if y_raw.size else 0.0,
        "global_bell_rms_raw_max": float(np.max(y_raw)) if y_raw.size else 0.0,
        "global_bell_rms_connected_mean": float(np.mean(y_conn)) if y_conn.size else 0.0,
        "global_bell_rms_connected_max": float(np.max(y_conn)) if y_conn.size else 0.0,
        **tracking,
    }

    return rows, global_summary


def witness_label_shuffle_control(rung_rows: Sequence[Dict[str, Any]], n_null: int, seed: int) -> Dict[str, Any]:
    """
    Label shuffle control: tests whether reciprocal/block structure depends on
    the witness arrangement, not just four arbitrary tile magnitudes.
    """
    rng = np.random.default_rng(seed + 202)
    complete = [r for r in rung_rows if r.get("complete")]
    if not complete:
        return {"control": "rung_label_shuffle", "observed": 0.0, "null_mean": 0.0, "null_std": 0.0, "z": 0.0, "p_two_sided": 1.0}

    def reciprocal_score(v: Sequence[float]) -> float:
        xy, yz, zy, yx = v
        return float(0.5 * (xy * yx + yz * zy))

    observed_vals = []
    all_vecs = []
    for r in complete:
        v = np.asarray([r["XY"], r["YZ"], r["ZY"], r["YX"]], dtype=np.float64)
        all_vecs.append(v)
        observed_vals.append(reciprocal_score(v))
    observed = float(np.mean(observed_vals))

    null = np.empty(n_null, dtype=np.float64)
    for i in range(n_null):
        vals = []
        for v in all_vecs:
            vals.append(reciprocal_score(rng.permutation(v)))
        null[i] = float(np.mean(vals))

    mu, sd, z, p = summarize_null(observed, null)
    return {
        "control": "rung_label_shuffle",
        "observed": observed,
        "null_mean": mu,
        "null_std": sd,
        "z": z,
        "p_two_sided": p,
    }


# =============================================================================
# OPTIONAL PLOTS
# =============================================================================


def maybe_write_plots(out_dir: Path, tile_rows: Sequence[TileStats], rung_rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    complete = [r for r in rung_rows if r.get("complete")]
    if complete:
        x = [r["base_delay_dt"] for r in complete]
        y = [r["bell_rms_raw"] for r in complete]
        yc = [r["bell_rms_connected"] for r in complete]
        plt.figure()
        plt.plot(x, y, marker="o", label="raw")
        plt.plot(x, yc, marker="o", label="connected")
        plt.xscale("symlog")
        plt.xlabel("base_delay_dt")
        plt.ylabel("Bell witness RMS")
        plt.title("D_M Bell-listener score vs base delay")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "bell_score_vs_base_delay.png", dpi=160)
        plt.close()

    if tile_rows:
        labels = [r.witness for r in tile_rows]
        vals = [r.corr_raw for r in tile_rows]
        plt.figure(figsize=(max(8, len(vals) * 0.35), 4))
        plt.axhline(0.0, linewidth=1)
        plt.plot(range(len(vals)), vals, marker="o")
        plt.xticks(range(len(vals)), labels, rotation=60)
        plt.xlabel("tile / witness order")
        plt.ylabel("<P0⊗P1>")
        plt.title("D_M per-tile Bell-witness correlators")
        plt.tight_layout()
        plt.savefig(out_dir / "tile_correlators.png", dpi=160)
        plt.close()


# =============================================================================
# CLI / MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D_M Probe 07 — Bell-listener / cavity-offset qproj analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qpu-base", default=str(DEFAULT_QPROJ_OFFSET), help="Path to data/dm_job_<JOB_ID>.npz. Defaults to latest_dm_qpu_data.json.")
    p.add_argument("--base-delays-dt", type=int, nargs="+", default=DEFAULT_BASE_DELAYS_DT,
                   help="Base-delay rungs used for metadata repair.")
    p.add_argument("--offset-dt", type=int, default=DEFAULT_OFFSET_DT,
                   help="Offset step in dt used for metadata repair.")
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
            "No --qpu-base provided and latest_dm_qpu_data.json could not be resolved. "
            "Pass --qpu-base data/dm_job_<JOB_ID>.npz"
        )
    qpu_base = qpu_base.expanduser().resolve()
    if not qpu_base.exists():
        raise FileNotFoundError(qpu_base)

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_ANALYSIS_DIR / f"dm_probe_07_bell_listener_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair, plan, base_meta = load_base(qpu_base, args)
    tile_rows = compute_tile_stats(pair, plan, n_null=args.n_null, seed=args.seed)
    rung_rows, global_summary = compute_rung_scores(pair, tile_rows, n_null=args.n_null, seed=args.seed)
    label_control = witness_label_shuffle_control(rung_rows, n_null=args.n_null, seed=args.seed)

    repaired_rows = [asdict(m) for m in plan]
    tile_csv_rows = [asdict(r) for r in tile_rows]
    control_rows = [label_control]
    for k, v in global_summary.items():
        if k.endswith("_perm_p") or k.endswith("_r"):
            continue
    control_rows.append({
        "control": "offset_tracking_base_delay_raw",
        "observed": global_summary.get("base_delay_corr_raw_r", 0.0),
        "null_mean": "",
        "null_std": "",
        "z": "",
        "p_two_sided": global_summary.get("base_delay_corr_raw_perm_p", 1.0),
    })
    control_rows.append({
        "control": "offset_tracking_base_delay_connected",
        "observed": global_summary.get("base_delay_corr_connected_r", 0.0),
        "null_mean": "",
        "null_std": "",
        "z": "",
        "p_two_sided": global_summary.get("base_delay_corr_connected_perm_p", 1.0),
    })
    control_rows.append({
        "control": "offset_tracking_total_delay_raw",
        "observed": global_summary.get("total_delay_corr_raw_r", 0.0),
        "null_mean": "",
        "null_std": "",
        "z": "",
        "p_two_sided": global_summary.get("total_delay_corr_raw_perm_p", 1.0),
    })
    control_rows.append({
        "control": "offset_tracking_total_delay_connected",
        "observed": global_summary.get("total_delay_corr_connected_r", 0.0),
        "null_mean": "",
        "null_std": "",
        "z": "",
        "p_two_sided": global_summary.get("total_delay_corr_connected_perm_p", 1.0),
    })

    write_csv(out_dir / "repaired_metadata.csv", repaired_rows, [
        "tile", "base_delay_dt", "offset_dt", "total_delay_dt", "basis_q0", "basis_q1", "witness", "repaired",
    ])
    write_csv(out_dir / "tile_correlators.csv", tile_csv_rows, [
        "tile", "rung_index", "witness_index", "witness", "base_delay_dt", "offset_dt", "total_delay_dt", "shots",
        "p00", "p01", "p10", "p11", "mean_q0", "mean_q1", "corr_raw", "corr_connected", "corr_se", "corr_abs",
        "null_mean", "null_std", "null_z", "null_p_two_sided",
    ])
    write_csv(out_dir / "rung_bell_scores.csv", rung_rows, [
        "rung_index", "complete", "base_delay_dt", "mean_offset_dt", "mean_total_delay_dt",
        "XY", "YZ", "ZY", "YX", "XY_connected", "YZ_connected", "ZY_connected", "YX_connected",
        "bell_rms_raw", "bell_mean_abs_raw", "bell_rms_connected", "bell_mean_abs_connected",
        "reciprocal_XY_YX_product", "reciprocal_YZ_ZY_product", "reciprocal_product_mean",
        "null_mean", "null_std", "null_z", "null_p_two_sided",
    ])
    write_csv(out_dir / "control_summary.csv", control_rows, [
        "control", "observed", "null_mean", "null_std", "z", "p_two_sided",
    ])

    if not args.no_plots:
        maybe_write_plots(out_dir, tile_rows, rung_rows)

    result = {
        "operator": "D_M",
        "probe": "D_M Probe 07 — Bell Listener / Cavity Offset",
        "framing": "Bell-witness correlation in the shared-chip ghost channel; not density reconstruction or effective rank.",
        "input": base_meta,
        "config": vars(args),
        "global_summary": global_summary,
        "controls": control_rows,
        "top_tiles_by_abs_corr": sorted(tile_csv_rows, key=lambda r: abs(float(r["corr_raw"])), reverse=True)[:10],
        "top_rungs_by_bell_rms_raw": sorted(rung_rows, key=lambda r: float(r["bell_rms_raw"]), reverse=True)[:10],
        "elapsed_sec": time.perf_counter() - t0,
        "outputs": {
            "out_dir": str(out_dir),
            "repaired_metadata_csv": str(out_dir / "repaired_metadata.csv"),
            "tile_correlators_csv": str(out_dir / "tile_correlators.csv"),
            "rung_bell_scores_csv": str(out_dir / "rung_bell_scores.csv"),
            "control_summary_csv": str(out_dir / "control_summary.csv"),
        },
        "bounded_claim_note": (
            "This probe can identify Bell-witness correlation and its offset/delay dependence. "
            "It does not by itself certify entanglement or reconstruct a Bell state."
        ),
    }

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2)

    # -------------------------------------------------------------------------
    # Terminal report
    # -------------------------------------------------------------------------
    print("=" * 96)
    print("  GHOST ORACLE SUITE — D_M PROBE 07: BELL LISTENER / CAVITY OFFSET")
    print("=" * 96)
    print(f"  Input     : {qpu_base}")
    print(f"  Backend   : {base_meta.get('backend')}")
    print(f"  Job ID    : {base_meta.get('job_id')}")
    print(f"  Tiles     : {base_meta.get('num_tiles')}  Shots: {base_meta.get('shots')}")
    print(f"  Repaired  : {base_meta.get('metadata_repaired')}  offset_step_dt={base_meta.get('offset_step_dt_used')}")
    print(f"  Out dir   : {out_dir}")
    print("-" * 96)
    print("  PER-TILE CORRELATORS")
    for r in tile_rows:
        print(
            f"  tile {r.tile:02d} rung={r.rung_index:02d} {r.witness:<2} "
            f"base={r.base_delay_dt:6d} off={r.offset_dt:6d} total={r.total_delay_dt:6d} "
            f"corr={r.corr_raw:+.5f} conn={r.corr_connected:+.5f} "
            f"z={r.null_z:+7.2f} p={r.null_p_two_sided:.4f}"
        )
    print("-" * 96)
    print("  RUNG BELL-WITNESS BLOCK SCORES")
    for r in rung_rows:
        print(
            f"  rung {r['rung_index']:02d} base={int(r['base_delay_dt']):6d} "
            f"rms={float(r['bell_rms_raw']):.5f} conn_rms={float(r['bell_rms_connected']):.5f} "
            f"XY={float(r['XY']):+.5f} YZ={float(r['YZ']):+.5f} "
            f"ZY={float(r['ZY']):+.5f} YX={float(r['YX']):+.5f} "
            f"z={float(r['null_z']):+.2f} p={float(r['null_p_two_sided']):.4f}"
        )
    print("-" * 96)
    print("  GLOBAL")
    print(f"  complete_rungs                 : {global_summary['complete_rungs']}")
    print(f"  global_bell_rms_raw_mean       : {global_summary['global_bell_rms_raw_mean']:.6f}")
    print(f"  global_bell_rms_raw_max        : {global_summary['global_bell_rms_raw_max']:.6f}")
    print(f"  global_bell_rms_connected_mean : {global_summary['global_bell_rms_connected_mean']:.6f}")
    print(f"  global_bell_rms_connected_max  : {global_summary['global_bell_rms_connected_max']:.6f}")
    print(f"  base_delay raw tracking r,p    : {global_summary['base_delay_corr_raw_r']:+.4f}, {global_summary['base_delay_corr_raw_perm_p']:.4f}")
    print(f"  base_delay conn tracking r,p   : {global_summary['base_delay_corr_connected_r']:+.4f}, {global_summary['base_delay_corr_connected_perm_p']:.4f}")
    print(f"  total_delay raw tracking r,p   : {global_summary['total_delay_corr_raw_r']:+.4f}, {global_summary['total_delay_corr_raw_perm_p']:.4f}")
    print(f"  total_delay conn tracking r,p  : {global_summary['total_delay_corr_connected_r']:+.4f}, {global_summary['total_delay_corr_connected_perm_p']:.4f}")
    print("-" * 96)
    print("  CONTROLS")
    for c in control_rows:
        print(f"  {c['control']:<40} obs={c.get('observed')} p={c.get('p_two_sided')}")
    print("-" * 96)
    print(f"  [SAVED] {out_dir}")
    print("=" * 96)


if __name__ == "__main__":
    main()
