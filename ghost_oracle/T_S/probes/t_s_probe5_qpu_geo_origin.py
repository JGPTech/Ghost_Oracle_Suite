#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
T_S PROBE 05 — QPU GEO ORIGIN / STRUCTURE ABLATION
====================================================================================================
Proper QPU-side Probe 05 for T_S — Temporal Stress Metric.

This probe stays inside the QPU project.

It does NOT:
    - build GPU bases
    - test GPU projection
    - use G_M bases
    - use S_M bases
    - treat geo as the projector

It DOES:
    - load the existing QPU-generated T_S field
    - use the existing T_S QPU generator schema
    - ask which QPU-side structures cause the raw geo path to emerge
    - ablate/mutate delay, round, edge, mode/site, and physical-layout views
    - compare raw geo route stability under those QPU-structure disruptions

Context
-------
The T_S QPU generator defines the QPU object as:

    F[mode, delay_site, delay_value, shot, round, edge]

where the field is measured from repeated parity-probe responses between
neighboring channel qubits under structured delay/coupling/perturbation.

Probe 01 found:
    The QPU field is a structured delay-channel object.

Probe 02 found:
    The field defines navigable geometry.

Probe 03 found:
    Raw-first JumpGeo mostly stays raw and only jumps to pow_exp_2 for stressed
    delay-only paths.

Probe 04 found:
    The distilled raw geo arithmetic path can run fast as a standalone CUDA
    speed model.

Probe 05 asks:
    How did the QPU produce that geo path naturally?

Method
------
For each mode × delay_site block, this probe computes local stress and raw geo
routes under real and ablated controls.

Controls include:
    real
    shot_shuffle
    delay_shuffle
    round_shuffle
    edge_shuffle
    delay_reverse
    round_reverse
    edge_reverse
    uniform_by_cell
    all_uniform
    mode_pool_same_site
    site_pool_same_mode

Metrics include:
    full_diag route cost
    delay_only route cost
    edge_only route cost
    stress trace
    stress avoidance proxy
    route deltas from real
    origin damage score

The main result is an origin table:

    Which structural ablation most damages the raw geo path?

Expected location
-----------------
    ghost_oracle/T_S/probes/t_s_probe5_qpu_geo_origin.py

Path convention
---------------
    HERE = Path(__file__).resolve().parent
    DATA_DIR = HERE.parent / "data"
    ANALYSIS_DIR = HERE / "analysis"

Usage
-----
Latest T_S QPU dump:

    python ghost_oracle/T_S/probes/t_s_probe5_qpu_geo_origin.py

Specific dump:

    python ghost_oracle/T_S/probes/t_s_probe5_qpu_geo_origin.py --npz ghost_oracle/T_S/data/ts_data_<JOB_ID>.npz

Include metadata/calibration summary:

    python ghost_oracle/T_S/probes/t_s_probe5_qpu_geo_origin.py --meta ghost_oracle/T_S/data/ts_job_<JOB_ID>.json

====================================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    plt = None
    _HAVE_MPL = False


# --------------------------------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
ANALYSIS_DIR = HERE / "analysis"


# --------------------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------------------

EPS = 1e-12


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def bits(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x).astype(np.int64) & 1).astype(np.uint8)


def finite_clean(x: np.ndarray, clip_abs: float = 1e9) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=clip_abs, neginf=0.0)
    y = np.clip(y, -clip_abs, clip_abs)
    return y


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def resolve_inputs(npz_arg: Optional[str], meta_arg: Optional[str]) -> Tuple[Path, Optional[Path], str]:
    if npz_arg is None:
        latest = DATA_DIR / "latest_ts_data.json"
        if not latest.exists():
            raise FileNotFoundError(f"No --npz provided and latest T_S data not found: {latest}")
        obj = load_json(latest)
        npz = Path(obj["npz"])
        meta = Path(obj["meta"]) if obj.get("meta") else None
        job_id = str(obj.get("job_id", npz.stem))
        if meta_arg:
            meta = Path(meta_arg)
        return npz, meta, job_id

    npz = Path(npz_arg)
    stem = npz.stem
    job_id = stem.split("_")[-1] if "_" in stem else stem
    meta = Path(meta_arg) if meta_arg else None
    return npz, meta, job_id


def load_ts(npz_path: Path) -> Dict[str, Any]:
    z = np.load(npz_path, allow_pickle=False)
    if "field" not in z.files:
        raise KeyError("T_S Probe 05 expects a `field` array in the dumped T_S npz.")

    field = bits(z["field"])
    if field.ndim != 6:
        raise ValueError(
            "Expected field shape (modes, delay_sites, delays, shots, rounds, edges), "
            f"got {field.shape}"
        )

    def str_array(name: str) -> List[str]:
        if name not in z.files:
            return []
        return [str(x) for x in z[name].tolist()]

    obj = {
        "field": field,
        "final": bits(z["final"]) if "final" in z.files else None,
        "modes": str_array("modes"),
        "delay_sites": str_array("delay_sites"),
        "delays": z["delays"].astype(int).tolist() if "delays" in z.files else list(range(field.shape[2])),
        "delay_unit": str(z["delay_unit"].item()) if "delay_unit" in z.files and z["delay_unit"].shape == () else "",
        "job_id": str(z["job_id"].item()) if "job_id" in z.files and z["job_id"].shape == () else npz_path.stem,
        "rounds": int(z["rounds"].item()) if "rounds" in z.files else int(field.shape[4]),
        "channels": int(z["channels"].item()) if "channels" in z.files else int(field.shape[5] + 1),
        "edges": int(z["edges"].item()) if "edges" in z.files else int(field.shape[5]),
    }

    if not obj["modes"]:
        obj["modes"] = [f"mode_{i}" for i in range(field.shape[0])]
    if not obj["delay_sites"]:
        obj["delay_sites"] = [f"site_{i}" for i in range(field.shape[1])]

    return obj


def calibration_summary(meta: Optional[dict]) -> Dict[str, Any]:
    """
    Summarize calibration metadata if present.

    This probe does not need calibration to run, but including a summary helps
    identify whether geo-origin effects are plausibly layout/readout/gate-linked.
    """
    if not meta or not isinstance(meta, dict):
        return {"available": False}

    cal = meta.get("calibration")
    if not cal:
        return {"available": False}

    out: Dict[str, Any] = {"available": True}
    for key in ("single_qubit", "readout", "two_qubit"):
        vals = []
        obj = cal.get(key, {})
        if isinstance(obj, dict):
            for v in obj.values():
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    pass
        if vals:
            arr = np.asarray(vals, dtype=float)
            out[key] = {
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        else:
            out[key] = {"n": 0}

    idle_vals_t1 = []
    idle_vals_t2 = []
    idle = cal.get("idling", {})
    if isinstance(idle, dict):
        for v in idle.values():
            if isinstance(v, dict):
                if v.get("t1") is not None:
                    idle_vals_t1.append(float(v["t1"]))
                if v.get("t2") is not None:
                    idle_vals_t2.append(float(v["t2"]))
    for name, vals in (("t1", idle_vals_t1), ("t2", idle_vals_t2)):
        if vals:
            arr = np.asarray(vals, dtype=float)
            out[name] = {
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        else:
            out[name] = {"n": 0}

    out["meta"] = cal.get("meta", {})
    return out


# --------------------------------------------------------------------------------------------------
# QPU STRUCTURE MUTATIONS / ABLATIONS
# --------------------------------------------------------------------------------------------------

CONTROL_MODES = [
    "real",
    "shot_shuffle",
    "delay_shuffle",
    "round_shuffle",
    "edge_shuffle",
    "delay_reverse",
    "round_reverse",
    "edge_reverse",
    "uniform_by_cell",
    "all_uniform",
    "mode_pool_same_site",
    "site_pool_same_mode",
]


def mutate_block(
    field: np.ndarray,
    mi: int,
    si: int,
    control: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return one mutated block with shape:

        delay, shot, round, edge

    field shape:
        mode, site, delay, shot, round, edge

    mode_pool_same_site:
        Replace this mode/site block with another mode at same site.

    site_pool_same_mode:
        Replace this mode/site block with another site at same mode.
    """
    block = field[mi, si].copy()

    if control == "real":
        return block

    if control == "shot_shuffle":
        return block[:, rng.permutation(block.shape[1]), :, :]

    if control == "delay_shuffle":
        return block[rng.permutation(block.shape[0]), :, :, :]

    if control == "round_shuffle":
        for d in range(block.shape[0]):
            for sh in range(block.shape[1]):
                for e in range(block.shape[3]):
                    block[d, sh, :, e] = block[d, sh, rng.permutation(block.shape[2]), e]
        return block

    if control == "edge_shuffle":
        for d in range(block.shape[0]):
            for sh in range(block.shape[1]):
                for r in range(block.shape[2]):
                    block[d, sh, r, :] = block[d, sh, r, rng.permutation(block.shape[3])]
        return block

    if control == "delay_reverse":
        return block[::-1, :, :, :].copy()

    if control == "round_reverse":
        return block[:, :, ::-1, :].copy()

    if control == "edge_reverse":
        return block[:, :, :, ::-1].copy()

    if control == "uniform_by_cell":
        # Preserve each delay/round/edge marginal rate across shots.
        p = block.mean(axis=1, keepdims=True)
        return (rng.random(block.shape) < p).astype(np.uint8)

    if control == "all_uniform":
        p = float(block.mean())
        return (rng.random(block.shape) < p).astype(np.uint8)

    if control == "mode_pool_same_site":
        n_modes = field.shape[0]
        if n_modes <= 1:
            return block
        choices = [j for j in range(n_modes) if j != mi]
        mj = int(rng.choice(choices))
        return field[mj, si].copy()

    if control == "site_pool_same_mode":
        n_sites = field.shape[1]
        if n_sites <= 1:
            return block
        choices = [j for j in range(n_sites) if j != si]
        sj = int(rng.choice(choices))
        return field[mi, sj].copy()

    raise ValueError(f"unknown control: {control}")


# --------------------------------------------------------------------------------------------------
# STRESS + RAW GEO ROUTES
# --------------------------------------------------------------------------------------------------

def gradients_delay_round_edge(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    block shape:
        delay, shot, round, edge

    Return aligned gradients:
        delay-1, shot, round-1, edge-1
    """
    if block.shape[0] < 2 or block.shape[2] < 2 or block.shape[3] < 2:
        empty = np.zeros((0, block.shape[1], 0, 0), dtype=np.float64)
        return empty, empty, empty

    d_tau = np.bitwise_xor(block[1:, :, :, :], block[:-1, :, :, :]).astype(np.float64)
    d_round = np.bitwise_xor(block[:, :, 1:, :], block[:, :, :-1, :]).astype(np.float64)
    d_edge = np.bitwise_xor(block[:, :, :, 1:], block[:, :, :, :-1]).astype(np.float64)

    d_tau = d_tau[:, :, :-1, :-1]
    d_round = d_round[:-1, :, :, :-1]
    d_edge = d_edge[:-1, :, :-1, :]

    return d_tau, d_round, d_edge


def local_stress(block: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Average over shots to produce local stress cells:

        delay_cell, round_cell, edge_cell
    """
    d_tau, d_round, d_edge = gradients_delay_round_edge(block)
    if d_tau.size == 0:
        z = np.zeros((0, 0, 0), dtype=np.float64)
        return {
            "tau_tau": z, "rr": z, "xx": z,
            "tau_r": z, "tau_x": z, "r_x": z,
            "trace": z,
        }

    tau_tau = np.mean(d_tau * d_tau, axis=1)
    rr = np.mean(d_round * d_round, axis=1)
    xx = np.mean(d_edge * d_edge, axis=1)
    tau_r = np.mean(d_tau * d_round, axis=1)
    tau_x = np.mean(d_tau * d_edge, axis=1)
    r_x = np.mean(d_round * d_edge, axis=1)
    trace = tau_tau + rr + xx

    return {
        "tau_tau": finite_clean(tau_tau),
        "rr": finite_clean(rr),
        "xx": finite_clean(xx),
        "tau_r": finite_clean(tau_r),
        "tau_x": finite_clean(tau_x),
        "r_x": finite_clean(r_x),
        "trace": finite_clean(trace),
    }


def movement_costs(
    st: Dict[str, np.ndarray],
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    transform: str = "raw",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Movement costs for raw or pow_exp_2 geometry.

    transform:
        raw
        pow_exp_2
    """
    comps = {k: np.asarray(v, dtype=np.float64) for k, v in st.items() if k != "trace"}

    if transform == "pow_exp_2":
        comps = {k: np.power(np.maximum(v, 0.0), 2.0) for k, v in comps.items()}
    elif transform == "raw":
        pass
    else:
        raise ValueError(f"unsupported transform: {transform}")

    tau = comps["tau_tau"] + coupling_weight * (comps["tau_r"] + comps["tau_x"])
    rnd = comps["rr"] + coupling_weight * (comps["tau_r"] + comps["r_x"])
    edge = comps["xx"] + coupling_weight * (comps["tau_x"] + comps["r_x"])

    tau = np.clip(finite_clean(tau), min_cost, max_cost)
    rnd = np.clip(finite_clean(rnd), min_cost, max_cost)
    edge = np.clip(finite_clean(edge), min_cost, max_cost)

    info = {
        "transform": transform,
        "tau_mean": float(np.mean(tau)) if tau.size else float("nan"),
        "round_mean": float(np.mean(rnd)) if rnd.size else float("nan"),
        "edge_mean": float(np.mean(edge)) if edge.size else float("nan"),
    }
    return tau, rnd, edge, info


def monotonic_dp_routes(
    st: Dict[str, np.ndarray],
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    transform: str = "raw",
) -> Dict[str, Any]:
    tau, rnd, edge, cost_info = movement_costs(
        st,
        coupling_weight=coupling_weight,
        min_cost=min_cost,
        max_cost=max_cost,
        transform=transform,
    )

    if tau.size == 0:
        return {
            "status": "degenerate",
            "full_cost": float("nan"),
            "delay_cost": float("nan"),
            "edge_cost": float("nan"),
            "stress_avoidance": float("nan"),
            "path_trace_mean_proxy": float("nan"),
            "global_trace_mean": float("nan"),
            **cost_info,
        }

    A, R, E = tau.shape
    dp = np.empty((A, R, E), dtype=np.float64)

    for a in range(A):
        for r in range(R):
            for e in range(E):
                if a == 0 and r == 0 and e == 0:
                    dp[a, r, e] = 0.0
                    continue
                best = float("inf")
                if a > 0:
                    best = min(best, dp[a - 1, r, e] + tau[a, r, e])
                if r > 0:
                    best = min(best, dp[a, r - 1, e] + rnd[a, r, e])
                if e > 0:
                    best = min(best, dp[a, r, e - 1] + edge[a, r, e])
                dp[a, r, e] = best

    cr = R // 2
    ce = E // 2
    ca = A // 2

    full_cost = float(dp[-1, -1, -1])
    delay_cost = float(np.sum(tau[1:, cr, ce])) if A > 1 else 0.0
    edge_cost = float(np.sum(edge[ca, cr, 1:])) if E > 1 else 0.0

    trace = np.asarray(st["trace"], dtype=np.float64)
    global_trace_mean = float(np.mean(trace)) if trace.size else float("nan")

    # Lightweight path trace proxy: diagonal-ish cells plus forced delay/edge paths.
    diag_vals = []
    steps = max(A, R, E)
    for k in range(steps):
        a = min(A - 1, round(k * (A - 1) / max(steps - 1, 1)))
        r = min(R - 1, round(k * (R - 1) / max(steps - 1, 1)))
        e = min(E - 1, round(k * (E - 1) / max(steps - 1, 1)))
        diag_vals.append(trace[a, r, e])
    path_trace_mean = float(np.mean(diag_vals)) if diag_vals else float("nan")
    stress_avoidance = float(global_trace_mean - path_trace_mean) if math.isfinite(global_trace_mean) else float("nan")

    return {
        "status": "ok",
        "full_cost": full_cost,
        "delay_cost": delay_cost,
        "edge_cost": edge_cost,
        "stress_avoidance": stress_avoidance,
        "path_trace_mean_proxy": path_trace_mean,
        "global_trace_mean": global_trace_mean,
        **cost_info,
    }


def stress_summary(st: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {}
    for k in ("tau_tau", "rr", "xx", "tau_r", "tau_x", "r_x", "trace"):
        x = np.asarray(st[k], dtype=np.float64)
        if x.size:
            out[f"{k}_mean"] = float(np.mean(x))
            out[f"{k}_std"] = float(np.std(x))
            out[f"{k}_min"] = float(np.min(x))
            out[f"{k}_max"] = float(np.max(x))
        else:
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_std"] = float("nan")
            out[f"{k}_min"] = float("nan")
            out[f"{k}_max"] = float("nan")
    return out


def jumpgeo_transform_for_block(real_routes: Dict[str, Any], *, weak_delay_threshold: float) -> str:
    """
    Mirror Probe 03's raw-first lesson at the block level.

    The only jump we allow here is pow_exp_2 when the delay path is clearly
    stressed relative to the full route.

    This is not projector logic. It is a QPU-origin diagnostic:
        Did the QPU's delay-axis structure require nonlinear sharpening?
    """
    full = safe_float(real_routes.get("full_cost"))
    delay = safe_float(real_routes.get("delay_cost"))
    if not math.isfinite(full) or not math.isfinite(delay) or full <= 0:
        return "raw"

    ratio = delay / (full + EPS)
    if ratio > weak_delay_threshold:
        return "pow_exp_2"
    return "raw"


# --------------------------------------------------------------------------------------------------
# ORIGIN ANALYSIS
# --------------------------------------------------------------------------------------------------

def route_delta(real: Dict[str, Any], ctrl: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k in ("full_cost", "delay_cost", "edge_cost", "stress_avoidance", "global_trace_mean"):
        rv = safe_float(real.get(k))
        cv = safe_float(ctrl.get(k))
        out[f"{k}_delta"] = cv - rv if math.isfinite(rv) and math.isfinite(cv) else float("nan")
        out[f"{k}_rel_delta"] = (cv - rv) / (abs(rv) + EPS) if math.isfinite(rv) and math.isfinite(cv) else float("nan")
    return out


def origin_damage_score(real: Dict[str, Any], ctrl: Dict[str, Any]) -> float:
    """
    Positive score means the control disrupts/damages the real raw geo path.

    Uses relative route changes plus stress-avoidance loss.
    """
    deltas = route_delta(real, ctrl)

    full = abs(safe_float(deltas.get("full_cost_rel_delta"), 0.0))
    delay = abs(safe_float(deltas.get("delay_cost_rel_delta"), 0.0))
    edge = abs(safe_float(deltas.get("edge_cost_rel_delta"), 0.0))

    real_avoid = safe_float(real.get("stress_avoidance"), 0.0)
    ctrl_avoid = safe_float(ctrl.get("stress_avoidance"), 0.0)
    avoid_loss = max(0.0, real_avoid - ctrl_avoid)

    return float(full + 0.75 * delay + 0.50 * edge + 0.50 * avoid_loss)


def run_probe(
    ts: Dict[str, Any],
    *,
    meta: Optional[dict],
    controls: List[str],
    seed: int,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    weak_delay_threshold: float,
) -> Dict[str, Any]:
    field = ts["field"]
    rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    for mi, mode_name in enumerate(ts["modes"]):
        for si, site_name in enumerate(ts["delay_sites"]):
            rng_base = seed + 1009 * mi + 7919 * si

            real_block = mutate_block(field, mi, si, "real", np.random.default_rng(rng_base))
            real_stress = local_stress(real_block)
            real_stress_summary = stress_summary(real_stress)
            real_routes_raw = monotonic_dp_routes(
                real_stress,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
                transform="raw",
            )
            selected_transform = jumpgeo_transform_for_block(
                real_routes_raw,
                weak_delay_threshold=weak_delay_threshold,
            )
            real_routes_selected = monotonic_dp_routes(
                real_stress,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
                transform=selected_transform,
            )

            block_rows = []

            for ci, control in enumerate(controls):
                rng = np.random.default_rng(rng_base + 31337 * ci)
                ctrl_block = mutate_block(field, mi, si, control, rng)
                ctrl_stress = local_stress(ctrl_block)
                ctrl_stress_summary = stress_summary(ctrl_stress)
                ctrl_routes_raw = monotonic_dp_routes(
                    ctrl_stress,
                    coupling_weight=coupling_weight,
                    min_cost=min_cost,
                    max_cost=max_cost,
                    transform="raw",
                )
                ctrl_routes_selected = monotonic_dp_routes(
                    ctrl_stress,
                    coupling_weight=coupling_weight,
                    min_cost=min_cost,
                    max_cost=max_cost,
                    transform=selected_transform,
                )

                deltas_raw = route_delta(real_routes_raw, ctrl_routes_raw)
                damage_raw = origin_damage_score(real_routes_raw, ctrl_routes_raw)

                deltas_selected = route_delta(real_routes_selected, ctrl_routes_selected)
                damage_selected = origin_damage_score(real_routes_selected, ctrl_routes_selected)

                row = {
                    "mode": mode_name,
                    "delay_site": site_name,
                    "control": control,
                    "selected_transform": selected_transform,
                    "damage_raw": damage_raw,
                    "damage_selected": damage_selected,
                    "real_full_cost_raw": real_routes_raw["full_cost"],
                    "control_full_cost_raw": ctrl_routes_raw["full_cost"],
                    "real_delay_cost_raw": real_routes_raw["delay_cost"],
                    "control_delay_cost_raw": ctrl_routes_raw["delay_cost"],
                    "real_edge_cost_raw": real_routes_raw["edge_cost"],
                    "control_edge_cost_raw": ctrl_routes_raw["edge_cost"],
                    "real_stress_avoidance_raw": real_routes_raw["stress_avoidance"],
                    "control_stress_avoidance_raw": ctrl_routes_raw["stress_avoidance"],
                    "real_trace_mean": real_stress_summary["trace_mean"],
                    "control_trace_mean": ctrl_stress_summary["trace_mean"],
                    **{f"raw_{k}": v for k, v in deltas_raw.items()},
                    **{f"selected_{k}": v for k, v in deltas_selected.items()},
                }
                rows.append(row)
                block_rows.append(row)

            # Most damaging ablation excluding real.
            ablations = [r for r in block_rows if r["control"] != "real"]
            best_damage = max(ablations, key=lambda r: safe_float(r["damage_raw"])) if ablations else None

            summary.append({
                "mode": mode_name,
                "delay_site": site_name,
                "selected_transform": selected_transform,
                "real_trace_mean": real_stress_summary["trace_mean"],
                "real_full_cost_raw": real_routes_raw["full_cost"],
                "real_delay_cost_raw": real_routes_raw["delay_cost"],
                "real_edge_cost_raw": real_routes_raw["edge_cost"],
                "real_delay_to_full_ratio": real_routes_raw["delay_cost"] / (real_routes_raw["full_cost"] + EPS),
                "real_stress_avoidance_raw": real_routes_raw["stress_avoidance"],
                "most_damaging_control": best_damage["control"] if best_damage else None,
                "most_damaging_score": best_damage["damage_raw"] if best_damage else float("nan"),
                "most_damaging_full_rel_delta": best_damage.get("raw_full_cost_rel_delta", float("nan")) if best_damage else float("nan"),
                "most_damaging_delay_rel_delta": best_damage.get("raw_delay_cost_rel_delta", float("nan")) if best_damage else float("nan"),
                "most_damaging_edge_rel_delta": best_damage.get("raw_edge_cost_rel_delta", float("nan")) if best_damage else float("nan"),
            })

    # Aggregate which ablation type matters most.
    aggregate = []
    for control in controls:
        if control == "real":
            continue
        vals = [safe_float(r["damage_raw"]) for r in rows if r["control"] == control]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            aggregate.append({
                "control": control,
                "mean_damage_raw": float(np.mean(vals)),
                "std_damage_raw": float(np.std(vals)),
                "max_damage_raw": float(np.max(vals)),
                "n": int(len(vals)),
            })

    aggregate.sort(key=lambda r: r["mean_damage_raw"], reverse=True)

    return {
        "rows": rows,
        "summary": summary,
        "aggregate_damage": aggregate,
        "calibration_summary": calibration_summary(meta),
    }


# --------------------------------------------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------------------------------------------

def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_damage(aggregate: List[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAVE_MPL or not aggregate:
        return

    labels = [r["control"] for r in aggregate]
    vals = [r["mean_damage_raw"] for r in aggregate]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    ax.bar(np.arange(len(labels)), vals)
    ax.set_title("T_S Probe 05 — QPU geo-origin damage by ablation")
    ax.set_ylabel("mean raw geo damage score")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out_dir / "qpu_geo_origin_damage.png", bbox_inches="tight")
    plt.close(fig)


def plot_block_summary(summary: List[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAVE_MPL or not summary:
        return

    labels = [f"{r['mode']}\n{r['delay_site']}" for r in summary]
    ratio = [safe_float(r["real_delay_to_full_ratio"]) for r in summary]
    trace = [safe_float(r["real_trace_mean"]) for r in summary]

    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(11, len(summary) * 0.75), 6), dpi=160)
    ax.plot(x, ratio, marker="o", label="delay/full route ratio")
    ax.plot(x, trace, marker="o", label="trace mean")
    ax.set_title("T_S Probe 05 — real QPU block geo indicators")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "qpu_geo_origin_block_summary.png", bbox_inches="tight")
    plt.close(fig)


def print_summary(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 132)
    print("  T_S PROBE 05 — QPU GEO ORIGIN SUMMARY")
    print("=" * 132)
    print(
        f"  {'mode':>13} | {'site':>14} | {'xform':>10} | {'trace':>8} | "
        f"{'full':>8} | {'delay':>8} | {'d/full':>8} | {'top damage':>18} | {'score':>9}"
    )
    print("  " + "-" * 130)

    for r in report["summary"]:
        print(
            f"  {r['mode']:>13} | {r['delay_site']:>14} | {r['selected_transform']:>10} | "
            f"{safe_float(r['real_trace_mean']):>8.5f} | "
            f"{safe_float(r['real_full_cost_raw']):>8.4f} | "
            f"{safe_float(r['real_delay_cost_raw']):>8.4f} | "
            f"{safe_float(r['real_delay_to_full_ratio']):>8.4f} | "
            f"{str(r['most_damaging_control']):>18} | "
            f"{safe_float(r['most_damaging_score']):>9.5f}"
        )

    print("\n" + "=" * 132)
    print("  AGGREGATE GEO-ORIGIN DAMAGE")
    print("=" * 132)
    print(f"  {'control':>22} | {'mean damage':>12} | {'std':>10} | {'max':>10} | {'n':>4}")
    print("  " + "-" * 70)
    for r in report["aggregate_damage"]:
        print(
            f"  {r['control']:>22} | {r['mean_damage_raw']:>12.6f} | "
            f"{r['std_damage_raw']:>10.6f} | {r['max_damage_raw']:>10.6f} | {r['n']:>4}"
        )


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T_S Probe 05 — QPU geo-origin / structure ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz", default=None, help="Dumped T_S QPU .npz. Defaults to latest.")
    p.add_argument("--meta", default=None, help="Optional T_S job metadata JSON. Defaults to latest metadata if available.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--controls", nargs="+", default=CONTROL_MODES, choices=CONTROL_MODES)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--coupling-weight", type=float, default=0.50)
    p.add_argument("--min-cost", type=float, default=1e-9)
    p.add_argument("--max-cost", type=float, default=1e9)
    p.add_argument("--weak-delay-threshold", type=float, default=0.62,
                   help="If delay_cost/full_cost exceeds this, record pow_exp_2 as the raw-first jump diagnostic.")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    npz_path, meta_path, job_id = resolve_inputs(args.npz, args.meta)
    ts = load_ts(npz_path)
    meta = load_json(meta_path) if meta_path and meta_path.exists() else None

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"ts_qpu_geo_origin_{job_id}_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 132)
    print("  T_S PROBE 05 — QPU GEO ORIGIN / STRUCTURE ABLATION")
    print("=" * 132)
    print(f"  NPZ        : {npz_path}")
    print(f"  Metadata   : {meta_path if meta_path else '(not provided)'}")
    print(f"  Out dir    : {out_dir}")
    print(f"  Field shape: {ts['field'].shape}")
    print(f"  Modes      : {ts['modes']}")
    print(f"  Sites      : {ts['delay_sites']}")
    print(f"  Delays     : {ts['delays']} {ts['delay_unit']}")
    print(f"  Controls   : {args.controls}")

    report_core = run_probe(
        ts,
        meta=meta,
        controls=list(args.controls),
        seed=int(args.seed),
        coupling_weight=float(args.coupling_weight),
        min_cost=float(args.min_cost),
        max_cost=float(args.max_cost),
        weak_delay_threshold=float(args.weak_delay_threshold),
    )

    report = {
        "schema": "ts_probe5_qpu_geo_origin_ablation",
        "description": (
            "QPU-side geo-origin probe. Tests which QPU field structures damage or preserve "
            "the raw geo path discovered from T_S analysis."
        ),
        "job_id": job_id,
        "npz": str(npz_path),
        "meta": str(meta_path) if meta_path else None,
        "settings": {
            "controls": list(args.controls),
            "seed": int(args.seed),
            "coupling_weight": float(args.coupling_weight),
            "min_cost": float(args.min_cost),
            "max_cost": float(args.max_cost),
            "weak_delay_threshold": float(args.weak_delay_threshold),
        },
        "source_shapes": {
            "field": list(ts["field"].shape),
            "modes": ts["modes"],
            "delay_sites": ts["delay_sites"],
            "delays": ts["delays"],
            "delay_unit": ts["delay_unit"],
            "rounds": ts["rounds"],
            "channels": ts["channels"],
            "edges": ts["edges"],
        },
        **report_core,
    }

    with open(out_dir / "qpu_geo_origin_report.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2)

    write_csv(report["rows"], out_dir / "qpu_geo_origin_rows.csv")
    write_csv(report["summary"], out_dir / "qpu_geo_origin_summary.csv")
    write_csv(report["aggregate_damage"], out_dir / "qpu_geo_origin_aggregate_damage.csv")

    if not args.no_plots:
        plot_damage(report["aggregate_damage"], out_dir)
        plot_block_summary(report["summary"], out_dir)

    print_summary(report)

    print(f"\n[SAVED] {out_dir}")
    print("=" * 132 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"[FATAL] {type(e).__name__}: {e}")
