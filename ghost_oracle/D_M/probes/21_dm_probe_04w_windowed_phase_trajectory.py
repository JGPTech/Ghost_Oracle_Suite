#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — D_M PROBE 04W: WINDOWED-RESOLUTION QPROJ PHASE TRAJECTORY
==============================================================================

What this is
------------
The higher-resolution version of Probe 04. It changes NOTHING about the D_M
projection math. It keeps:

    pi_phase_from_yz_zy      axial phase, mod pi (witness orientation is axial)
    axial_coherence_pi       doubled-angle coherence (exp(2j*phase))
    unwrap_pi_phase          0.5 * unwrap(2*phase)
    normalize_x              linear AND log delay normalization
    pi_periodic_fit_score    omega sweep + permutation p-value
    safe_corrcoef / permutation_p_corr
    independent_bit_shuffle  control: breaks q0/q1 pairing, keeps marginals
    witness_label_shuffle    control: is the YZ/ZY assignment load-bearing?

The ONLY change vs Probe 04: instead of one connected correlator per tile over
all shots, the shots are windowed, so each window is a Probe-04-style snapshot
of the witness manifold. The pi-phase trajectory then has (n_windows * n_rungs)
points instead of n_rungs, feeding the identical pi_periodic_fit_score. Same
question -- "is a structured axial phase trajectory projectable and does it
survive controls" -- asked at window resolution.

Interpretation discipline (unchanged from Probe 04)
--------------------------------------------------
Reports structured Bell-witness phase-trajectory projection. Does NOT certify
Bell nonlocality, reconstruct a density matrix, or prove a prepared Bell state.
A high windowed pi-periodic score that ALSO clears the bit-shuffle and
label-shuffle controls means the axial phase trajectory is real structure; if
the controls match it, it is not.

Usage
-----
    python d_m_probe04w_windowed_phase_trajectory.py --auto --window 256
    python d_m_probe04w_windowed_phase_trajectory.py ^
      --null      ghost_oracle/D_M/data/dm_data_<NULL>.npz ^
      --base-only ghost_oracle/D_M/data/dm_data_<BASE>.npz ^
      --offset-on ghost_oracle/D_M/data/dm_data_<OFFSET>.npz ^
      --window 256
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# DEFAULTS (identical to Probe 04)
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


# =============================================================================
# LOAD / REPAIR (identical to Probe 04)
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
        tiles.append(arr)
        t += 1
    if not tiles:
        raise ValueError("No pair array or pair_tile* arrays found")
    return np.stack(tiles, axis=0)


def read_optional(npz: Any, key: str) -> Optional[np.ndarray]:
    return np.asarray(npz[key]) if key in npz.files else None


def valid_int_array(a: Optional[np.ndarray], n: int, min_value: int = 0) -> bool:
    if a is None or a.shape[0] < n:
        return False
    return bool(np.all(np.asarray(a[:n]) >= min_value))


def witness_label(b0: int, b1: int) -> str:
    if 0 <= b0 < len(BASIS_LABELS) and 0 <= b1 < len(BASIS_LABELS):
        return BASIS_LABELS[b0] + BASIS_LABELS[b1]
    return "??"


def build_repaired_plan(num_tiles, eb, eo, et, e0, e1, base_delays_dt, offset_step_dt, witness_pairs, force_repair=False):
    have = (not force_repair
            and valid_int_array(eb, num_tiles) and valid_int_array(eo, num_tiles)
            and valid_int_array(et, num_tiles) and valid_int_array(e0, num_tiles)
            and valid_int_array(e1, num_tiles))
    plan = []
    for t in range(num_tiles):
        if have:
            base, off, total = int(eb[t]), int(eo[t]), int(et[t])
            b0, b1, repaired = int(e0[t]), int(e1[t]), False
        else:
            rung, wi = t // len(witness_pairs), t % len(witness_pairs)
            base = int(base_delays_dt[min(rung, len(base_delays_dt) - 1)])
            off = int(t * offset_step_dt)
            total = int(base + off)
            b0, b1 = witness_pairs[wi]
            repaired = True
        plan.append(TileMeta(t, base, off, total, b0, b1, witness_label(b0, b1), repaired))
    return plan


def infer_condition_from_metadata(path: Path) -> Optional[str]:
    try:
        obj = np.load(path, allow_pickle=True)
        pair = load_pair_stack(obj)
        n = pair.shape[0]
        base = read_optional(obj, "tile_base_delay_dt")
        off = read_optional(obj, "tile_offset_dt")
        if base is None or off is None or base.shape[0] < n or off.shape[0] < n:
            return None
        hb = bool(np.max(base[:n]) > 0)
        ho = bool(np.max(off[:n]) > 0)
        if not hb and not ho:
            return "null"
        if hb and not ho:
            return "base_only"
        if hb and ho:
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
        mt = path.stat().st_mtime
        if cond not in found or mt > found[cond][0]:
            found[cond] = (mt, path)
    return {k: v[1] for k, v in found.items()}


def load_base(path: Path, condition: str, args) -> Tuple[np.ndarray, List[TileMeta], Dict[str, Any]]:
    obj = np.load(path, allow_pickle=True)
    pair = load_pair_stack(obj)
    n_tiles, shots, width = pair.shape
    offset_step = int(args.offset_dt)
    if "offset_dt" in obj.files and not args.force_offset:
        try:
            stored = int(np.asarray(obj["offset_dt"]).item())
            if stored >= 0:
                offset_step = stored
        except Exception:
            pass
    plan = build_repaired_plan(
        n_tiles, read_optional(obj, "tile_base_delay_dt"), read_optional(obj, "tile_offset_dt"),
        read_optional(obj, "tile_total_delay_dt"), read_optional(obj, "tile_basis_q0"),
        read_optional(obj, "tile_basis_q1"), args.base_delays_dt, offset_step, WITNESS_PAIRS, args.force_repair)
    meta = {
        "condition": condition, "condition_label": CONDITION_LABELS.get(condition, condition),
        "path": str(path),
        "backend": scalar_str(obj["backend"]) if "backend" in obj.files else "unknown",
        "job_id": scalar_str(obj["job_id"]) if "job_id" in obj.files else path.stem,
        "num_tiles": int(n_tiles), "shots": int(shots),
        "metadata_repaired": bool(any(m.repaired for m in plan)),
        "offset_step_dt_used": int(offset_step),
    }
    return pair, plan, meta


# =============================================================================
# PROBE-04 PHASE MACHINERY  (verbatim)
# =============================================================================

def pair_to_signs(bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    b = np.asarray(bits, dtype=np.uint8)
    return 1.0 - 2.0 * b[:, 0].astype(np.float64), 1.0 - 2.0 * b[:, 1].astype(np.float64)


def corr_connected_from_bits(bits: np.ndarray) -> float:
    s0, s1 = pair_to_signs(bits)
    if s0.size < 2:
        return 0.0
    return float(np.mean(s0 * s1) - np.mean(s0) * np.mean(s1))


def pi_phase_from_yz_zy(yz: float, zy: float) -> Tuple[float, float, float, float, float]:
    y = float(yz); r = float(-zy)
    energy = float(math.sqrt(y * y + r * r))
    energy_rms = float(energy / math.sqrt(2.0))
    phase = float(math.atan2(r, y) % math.pi)
    phase_unit = float(phase / math.pi)
    inversion = float(-y * zy)
    return energy, energy_rms, phase, phase_unit, inversion


def safe_corrcoef(x, y) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def permutation_p_corr(x, y, obs, n_perm, seed, two_sided=True) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 1.0
    vals = np.array([safe_corrcoef(x, rng.permutation(y)) for _ in range(n_perm)])
    if two_sided:
        return float((np.count_nonzero(np.abs(vals) >= abs(obs)) + 1) / (n_perm + 1))
    return float((np.count_nonzero(vals >= obs) + 1) / (n_perm + 1))


def normalize_x(x, mode: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if mode == "log":
        arr = np.log1p(np.maximum(arr, 0.0))
    lo, hi = float(np.min(arr)) if arr.size else 0.0, float(np.max(arr)) if arr.size else 1.0
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def axial_coherence_pi(phases, weights=None) -> float:
    p = np.asarray(phases, dtype=np.float64)
    if p.size == 0:
        return 0.0
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=np.float64)
    if np.sum(np.abs(w)) < 1e-15:
        return 0.0
    z = np.sum(w * np.exp(2j * p)) / np.sum(np.abs(w))
    return float(abs(z))


def unwrap_pi_phase(phases) -> np.ndarray:
    return 0.5 * np.unwrap(2.0 * np.asarray(phases, dtype=np.float64))


def pi_periodic_fit_score(delays, phases, weights, n_perm, seed) -> Dict[str, Any]:
    d = np.asarray(delays, dtype=np.float64)
    ph = np.asarray(phases, dtype=np.float64)
    wt = np.asarray(weights, dtype=np.float64) if weights is not None else None
    if d.size < 3 or np.std(d) < 1e-12 or np.std(ph) < 1e-12:
        return {"pi_periodic_score": 0.0, "pi_periodic_p": 1.0, "pi_periodic_mode": "none",
                "pi_periodic_omega": 0.0, "phase_velocity_r": 0.0, "phase_velocity_p": 1.0,
                "phase_span_pi_units": 0.0}
    best = {"score": -1.0, "mode": "linear", "omega": 0.0}
    omegas = np.linspace(-2.5 * math.pi, 2.5 * math.pi, 801)
    for mode in ("linear", "log"):
        x = normalize_x(d, mode)
        for omega in omegas:
            score = axial_coherence_pi((ph - omega * x) % math.pi, wt)
            if score > best["score"]:
                best = {"score": float(score), "mode": mode, "omega": float(omega)}
    rng = np.random.default_rng(seed)
    x_best = normalize_x(d, best["mode"])
    null_scores = np.array([axial_coherence_pi((rng.permutation(ph) - best["omega"] * x_best) % math.pi, wt)
                            for _ in range(n_perm)])
    p = float((np.count_nonzero(null_scores >= best["score"]) + 1) / (n_perm + 1))
    unwrapped = unwrap_pi_phase(ph)
    phase_r = safe_corrcoef(normalize_x(d, best["mode"]), unwrapped)
    phase_p = permutation_p_corr(normalize_x(d, best["mode"]), unwrapped, phase_r, n_perm, seed + 313)
    span = float((np.max(unwrapped) - np.min(unwrapped)) / math.pi) if unwrapped.size else 0.0
    return {"pi_periodic_score": float(best["score"]), "pi_periodic_p": p, "pi_periodic_mode": best["mode"],
            "pi_periodic_omega": float(best["omega"]), "phase_velocity_r": float(phase_r),
            "phase_velocity_p": float(phase_p), "phase_span_pi_units": span}


# =============================================================================
# WINDOWED SNAPSHOT EXTRACTION  (the only new thing)
# =============================================================================

def windowed_rung_points(condition: str, pair: np.ndarray, plan: Sequence[TileMeta], window: int) -> List[Dict[str, Any]]:
    """
    Same projection as Probe 04's phase_rung_rows, but per WINDOW of shots.
    For each window and each rung, build the axial phase point from that window's
    connected correlators on the rung's witness tiles. Returns one row per
    (window, rung) with the Probe-04 fields needed for the phase fit.
    """
    tiles, shots, _ = pair.shape
    n_windows = max(1, shots // window)

    # map rung -> {witness: tile_index}
    rung_tiles: Dict[int, Dict[str, int]] = {}
    for t, meta in enumerate(plan):
        rung_tiles.setdefault(t // len(WITNESS_ORDER), {})[meta.witness] = t

    out: List[Dict[str, Any]] = []
    for w in range(n_windows):
        lo, hi = w * window, w * window + window
        for rung, by_w in sorted(rung_tiles.items()):
            if "YZ" not in by_w or "ZY" not in by_w:
                continue
            yz = corr_connected_from_bits(pair[by_w["YZ"], lo:hi, :])
            zy = corr_connected_from_bits(pair[by_w["ZY"], lo:hi, :])
            xy = corr_connected_from_bits(pair[by_w["XY"], lo:hi, :]) if "XY" in by_w else 0.0
            yx = corr_connected_from_bits(pair[by_w["YX"], lo:hi, :]) if "YX" in by_w else 0.0
            energy, energy_rms, phase, phase_unit, inversion = pi_phase_from_yz_zy(yz, zy)
            comp_energy = float(math.sqrt((xy * xy + yx * yx) / 2.0))
            meta_tiles = [plan[by_w[wn]] for wn in by_w]
            total_delay = float(np.mean([m.total_delay_dt for m in meta_tiles]))
            out.append({
                "condition": condition, "window": int(w), "rung_index": int(rung),
                "total_delay_dt_mean": total_delay,
                "XY_connected": xy, "YZ_connected": yz, "ZY_connected": zy, "YX_connected": yx,
                "yzzy_energy_euclidean": energy, "yzzy_energy_rms": energy_rms,
                "directional_specificity": float(energy_rms - comp_energy),
                "yzzy_inversion_score": inversion,
                "pi_phase_rad": phase, "pi_phase_unit": phase_unit,
            })
    return out


def fit_windowed(condition: str, rows: Sequence[Dict[str, Any]], n_perm: int, seed: int) -> Dict[str, Any]:
    """Feed the windowed (window x rung) phase trajectory into Probe 04's exact fit."""
    if not rows:
        return {"condition": condition}
    total = np.asarray([r["total_delay_dt_mean"] for r in rows], dtype=np.float64)
    phase = np.asarray([r["pi_phase_rad"] for r in rows], dtype=np.float64)
    energy = np.asarray([r["yzzy_energy_euclidean"] for r in rows], dtype=np.float64)
    energy_rms = np.asarray([r["yzzy_energy_rms"] for r in rows], dtype=np.float64)
    spec = np.asarray([r["directional_specificity"] for r in rows], dtype=np.float64)
    yz = np.asarray([r["YZ_connected"] for r in rows], dtype=np.float64)

    fit = pi_periodic_fit_score(total, phase, energy, n_perm=n_perm, seed=seed + 17)
    energy_r = safe_corrcoef(total, energy)
    spec_r = safe_corrcoef(total, spec)
    yz_r = safe_corrcoef(total, yz)
    return {
        "condition": condition, "condition_label": CONDITION_LABELS.get(condition, condition),
        "n_points": int(len(rows)),
        "YZ_ZY_energy_mean": float(np.mean(energy)),
        "YZ_ZY_energy_rms_mean": float(np.mean(energy_rms)),
        "directional_specificity_mean": float(np.mean(spec)),
        "total_delay_energy_tracking_r": float(energy_r),
        "total_delay_energy_tracking_p": permutation_p_corr(total, energy, energy_r, n_perm, seed + 102),
        "total_delay_specificity_tracking_r": float(spec_r),
        "total_delay_YZ_tracking_r": float(yz_r),
        **fit,
    }


# =============================================================================
# CONTROLS  (Probe 04, adapted to the windowed extraction)
# =============================================================================

def independent_bit_shuffle_pair(pair: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(pair, copy=True)
    for t in range(out.shape[0]):
        out[t, :, 1] = rng.permutation(out[t, :, 1])
    return out


def witness_label_shuffle_plan(plan: Sequence[TileMeta], rng: np.random.Generator) -> List[TileMeta]:
    """Shuffle the YZ/ZY/XY/YX label assignment within each rung; geometry untouched."""
    groups: Dict[int, List[int]] = {}
    for t, m in enumerate(plan):
        groups.setdefault(t // len(WITNESS_ORDER), []).append(t)
    new_plan = [TileMeta(**asdict(m)) for m in plan]
    for rung, idxs in groups.items():
        labels = [plan[i].witness for i in idxs]
        perm = list(rng.permutation(labels))
        for i, lab in zip(idxs, perm):
            new_plan[i].witness = str(lab)
    return new_plan


def run_controls(condition, pair, plan, window, obs_fit, n_perm, seed):
    rng = np.random.default_rng(seed)
    metrics = ["pi_periodic_score", "YZ_ZY_energy_mean", "directional_specificity_mean",
               "total_delay_energy_tracking_r"]
    n_ctrl = max(32, n_perm // 8)  # control resamples; fit inside uses small n_perm
    controls = []

    # independent bit shuffle: breaks q0/q1 pairing
    dist = {m: [] for m in metrics}
    for i in range(n_ctrl):
        p2 = independent_bit_shuffle_pair(pair, rng)
        rows = windowed_rung_points(condition, p2, plan, window)
        f = fit_windowed(condition, rows, n_perm=48, seed=seed + 10000 + i)
        for m in metrics:
            dist[m].append(float(f.get(m, 0.0)))
    for m in metrics:
        vals = np.asarray(dist[m]); obs = float(obs_fit.get(m, 0.0))
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        controls.append({"condition": condition, "control": "independent_bit_shuffle", "metric": m,
                         "observed": obs, "null_mean": mu, "null_std": sd,
                         "z": float((obs - mu) / sd) if sd > 1e-12 else 0.0,
                         "p_upper": float((np.count_nonzero(vals >= obs) + 1) / (vals.size + 1))})

    # witness label shuffle: is YZ/ZY assignment load-bearing
    dist = {m: [] for m in metrics}
    for i in range(n_ctrl):
        sp = witness_label_shuffle_plan(plan, rng)
        rows = windowed_rung_points(condition, pair, sp, window)
        f = fit_windowed(condition, rows, n_perm=48, seed=seed + 20000 + i)
        for m in metrics:
            dist[m].append(float(f.get(m, 0.0)))
    for m in metrics:
        vals = np.asarray(dist[m]); obs = float(obs_fit.get(m, 0.0))
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        controls.append({"condition": condition, "control": "witness_label_shuffle", "metric": m,
                         "observed": obs, "null_mean": mu, "null_std": sd,
                         "z": float((obs - mu) / sd) if sd > 1e-12 else 0.0,
                         "p_upper": float((np.count_nonzero(vals >= obs) + 1) / (vals.size + 1))})
    return controls


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="D_M Probe 04W — Windowed-resolution QPROJ phase trajectory",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--null", default=str(DEFAULT_QPROJ_NULL))
    p.add_argument("--base-only", default=str(DEFAULT_QPROJ_BASE))
    p.add_argument("--offset-on", default=str(DEFAULT_QPROJ_OFFSET))
    p.add_argument("--auto", action="store_true")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--window", type=int, default=512, help="shots per windowed snapshot")
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--base-delays-dt", type=int, nargs="+", default=DEFAULT_BASE_DELAYS_DT)
    p.add_argument("--offset-dt", type=int, default=DEFAULT_OFFSET_DT)
    p.add_argument("--force-repair", action="store_true")
    p.add_argument("--force-offset", action="store_true")
    return p.parse_args()


def collect_paths(args) -> Dict[str, Path]:
    paths = {}
    if args.null: paths["null"] = Path(args.null)
    if args.base_only: paths["base_only"] = Path(args.base_only)
    if args.offset_on: paths["offset_on"] = Path(args.offset_on)
    if args.auto or not paths:
        for k, v in auto_discover_paths(Path(args.data_dir)).items():
            paths.setdefault(k, v)
    missing = [k for k in CONDITION_ORDER if k not in paths]
    if missing:
        raise FileNotFoundError("Missing condition file(s): " + ", ".join(missing) +
                                "\nPass --null/--base-only/--offset-on or --auto.")
    for k, v in paths.items():
        if not v.exists():
            raise FileNotFoundError(f"{k} path does not exist: {v}")
    return paths


def main():
    args = parse_args()
    t0 = time.perf_counter()
    paths = collect_paths(args)
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_ANALYSIS_DIR / f"dm_probe_21_windowed_phase_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("  D_M PROBE 04W — WINDOWED-RESOLUTION QPROJ PHASE TRAJECTORY")
    print("=" * 100)
    print(f"  Out dir : {out_dir}")
    print(f"  window  : {args.window} shots/snapshot     n_perm : {args.n_perm}")
    print("-" * 100)

    all_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []

    for idx, condition in enumerate(CONDITION_ORDER):
        pair, plan, meta = load_base(paths[condition], condition, args)
        rows = windowed_rung_points(condition, pair, plan, args.window)
        fit = fit_windowed(condition, rows, n_perm=args.n_perm, seed=args.seed + idx * 1000)
        all_rows.extend(rows)
        fit_rows.append(fit)

        n_win = max(1, meta["shots"] // args.window)
        print(f"  {condition:<10} {CONDITION_LABELS[condition]}")
        print(f"    tiles/shots : {meta['num_tiles']}/{meta['shots']}  windows={n_win}  points={fit['n_points']}  repaired={meta['metadata_repaired']}")
        print(f"    energy mean : {fit.get('YZ_ZY_energy_mean', 0):.6f}  specificity={fit.get('directional_specificity_mean', 0):+.6f}")
        print(f"    pi score    : {fit.get('pi_periodic_score', 0):.4f}  p={fit.get('pi_periodic_p', 1):.4f}  mode={fit.get('pi_periodic_mode','none')}  omega={fit.get('pi_periodic_omega',0):+.3f}")
        print(f"    phase_vel_r : {fit.get('phase_velocity_r', 0):+.4f}  p={fit.get('phase_velocity_p',1):.4f}  span_pi={fit.get('phase_span_pi_units',0):.3f}")
        print(f"    energy_track: r={fit.get('total_delay_energy_tracking_r',0):+.4f} p={fit.get('total_delay_energy_tracking_p',1):.4f}")

        control_rows.extend(run_controls(condition, pair, plan, args.window, fit,
                                         n_perm=args.n_perm, seed=args.seed + 50000 + idx * 1000))
        print("-" * 100)

    print("  CONTROL CHECK (does observed clear the structural nulls?)")
    for cond in CONDITION_ORDER:
        for ctrl in ("independent_bit_shuffle", "witness_label_shuffle"):
            rs = [r for r in control_rows if r["condition"] == cond and r["control"] == ctrl
                  and r["metric"] == "pi_periodic_score"]
            if rs:
                r = rs[0]
                print(f"    {cond:<10} {ctrl:<24} pi_score obs={r['observed']:.4f} "
                      f"null={r['null_mean']:.4f}±{r['null_std']:.4f} z={r['z']:+.2f} p={r['p_upper']:.4f}")
    print("-" * 100)

    write_csv(out_dir / "windowed_points.csv", all_rows,
              ["condition", "window", "rung_index", "total_delay_dt_mean",
               "XY_connected", "YZ_connected", "ZY_connected", "YX_connected",
               "yzzy_energy_euclidean", "yzzy_energy_rms", "directional_specificity",
               "yzzy_inversion_score", "pi_phase_rad", "pi_phase_unit"])
    write_csv(out_dir / "windowed_fit.csv", fit_rows,
              ["condition", "condition_label", "n_points", "YZ_ZY_energy_mean", "YZ_ZY_energy_rms_mean",
               "directional_specificity_mean", "total_delay_energy_tracking_r", "total_delay_energy_tracking_p",
               "total_delay_specificity_tracking_r", "total_delay_YZ_tracking_r",
               "pi_periodic_score", "pi_periodic_p", "pi_periodic_mode", "pi_periodic_omega",
               "phase_velocity_r", "phase_velocity_p", "phase_span_pi_units"])
    write_csv(out_dir / "control_summary.csv", control_rows,
              ["condition", "control", "metric", "observed", "null_mean", "null_std", "z", "p_upper"])

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({
            "schema": "ghost_oracle.dm.probe04w.windowed_phase_trajectory.v1",
            "operator": "D_M",
            "bounded_claim": ("Windowed-resolution structured Bell-witness phase-trajectory projection. "
                              "Does not certify Bell nonlocality, reconstruct a density matrix, or prove a prepared Bell state."),
            "window": int(args.window), "config": vars(args),
            "paths": {k: str(v) for k, v in paths.items()},
            "fit": fit_rows, "controls": control_rows,
            "elapsed_sec": time.perf_counter() - t0,
        }), f, indent=2)

    print(f"  [SAVED] {out_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
