#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
T_S PROBE 06 — QPU PROJECTION FINALIZER: RAW DAMAGE SIGNATURE
====================================================================================================
Corrected Probe 06 for T_S — Temporal Stress Metric.

This is the QPU-side projection finalizer.

NO normalization.
NO cosine similarity.
NO classifier framing.
NO GPU base generation.
NO G_M/S_M bases.
NO projector benchmark.

This probe stays inside the QPU project and uses the existing T_S QPU generator
object:

    F[mode, delay_site, delay_value, shot, round, edge]

The purpose is to finalize the QPU-side projection signature by measuring direct
raw damage to the raw geo path under edge / round / round-edge ablations.

Why this exists
---------------
Probe 05 showed that raw geo origin is mostly carried by:

    edge/channel adjacency
    round interaction order
    mode perturbation structure
    cellwise field structure

and only weakly by exact delay order.

The previous Probe 06 attempt incorrectly normalized signature vectors and used
classification-style retrieval. That is removed here.

This corrected Probe 06 does only raw-damage measurements.

Core damage metric
------------------
For a real block and an ablated block:

    damage =
        |full_cost_ablated  - full_cost_real|
      + |delay_cost_ablated - delay_cost_real|
      + |edge_cost_ablated  - edge_cost_real|
      + max(0, real_stress_avoidance - ablated_stress_avoidance)

Optional relative components are also recorded, but the primary signature is raw.

Main outputs
------------
qpu_projection_signature.npz contains:

    field_edge_profile[mode, site, edge]
    field_round_profile[mode, site, round]
    field_round_edge_profile[mode, site, round, edge]
    field_delay_edge_profile[mode, site, delay, edge]

    stress_edge_profile[mode, site, edge_cell]
    stress_round_profile[mode, site, round_cell]
    stress_round_edge_profile[mode, site, round_cell, edge_cell]
    stress_delay_edge_profile[mode, site, delay_cell, edge_cell]

    raw_geo_routes[mode, site, route_component]
    edge_damage[mode, site, edge]
    round_damage[mode, site, round]
    round_edge_damage[mode, site, round, edge]

Expected location
-----------------
    ghost_oracle/T_S/probes/t_s_probe6_qpu_projection_raw_damage.py

Path convention
---------------
    HERE = Path(__file__).resolve().parent
    DATA_DIR = HERE.parent / "data"
    ANALYSIS_DIR = HERE / "analysis"

Usage
-----
Latest T_S QPU dump:

    python ghost_oracle/T_S/probes/t_s_probe6_qpu_projection_raw_damage.py

Explicit files:

    python ghost_oracle/T_S/probes/t_s_probe6_qpu_projection_raw_damage.py ^
        --npz ghost_oracle/T_S/data/ts_data_<JOB_ID>.npz ^
        --meta ghost_oracle/T_S/data/ts_job_<JOB_ID>.json

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


def finite_clean32(x: np.ndarray, clip_abs: float = 1e9) -> np.ndarray:
    return finite_clean(x, clip_abs=clip_abs).astype(np.float32)


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
        raise KeyError("Probe 06 expects a `field` array in the dumped T_S npz.")

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
    if not meta or not isinstance(meta, dict):
        return {"available": False}

    cal = meta.get("calibration")
    if not cal:
        return {"available": False}

    out: Dict[str, Any] = {"available": True, "meta": cal.get("meta", {})}

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
            a = np.asarray(vals, dtype=np.float64)
            out[key] = {
                "n": int(a.size),
                "mean": float(a.mean()),
                "std": float(a.std()),
                "min": float(a.min()),
                "max": float(a.max()),
            }
        else:
            out[key] = {"n": 0}

    idle_t1 = []
    idle_t2 = []
    idle = cal.get("idling", {})
    if isinstance(idle, dict):
        for v in idle.values():
            if isinstance(v, dict):
                if v.get("t1") is not None:
                    idle_t1.append(float(v["t1"]))
                if v.get("t2") is not None:
                    idle_t2.append(float(v["t2"]))

    for name, vals in (("t1", idle_t1), ("t2", idle_t2)):
        if vals:
            a = np.asarray(vals, dtype=np.float64)
            out[name] = {
                "n": int(a.size),
                "mean": float(a.mean()),
                "std": float(a.std()),
                "min": float(a.min()),
                "max": float(a.max()),
            }
        else:
            out[name] = {"n": 0}

    return out


# --------------------------------------------------------------------------------------------------
# FIELD / STRESS / RAW GEO
# --------------------------------------------------------------------------------------------------

def gradients_delay_round_edge(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    block shape:
        delay, shot, round, edge

    returns:
        d_tau, d_round, d_edge with shape:
        delay_cell, shot, round_cell, edge_cell
    """
    if block.shape[0] < 2 or block.shape[2] < 2 or block.shape[3] < 2:
        empty = np.zeros((0, block.shape[1], 0, 0), dtype=np.float64)
        return empty, empty, empty

    d_tau = np.bitwise_xor(block[1:, :, :, :], block[:-1, :, :, :]).astype(np.float64)
    d_round = np.bitwise_xor(block[:, :, 1:, :], block[:, :, :-1, :]).astype(np.float64)
    d_edge = np.bitwise_xor(block[:, :, :, 1:], block[:, :, :, :-1]).astype(np.float64)

    # Common lattice.
    d_tau = d_tau[:, :, :-1, :-1]
    d_round = d_round[:-1, :, :, :-1]
    d_edge = d_edge[:-1, :, :-1, :]

    return d_tau, d_round, d_edge


def local_stress(block: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Shot-averaged local T_S stress field.

    block shape:
        delay, shot, round, edge

    output components:
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


def raw_routes_from_stress(
    st: Dict[str, np.ndarray],
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> Dict[str, float]:
    """
    Raw monotonic geo route cost. No normalization.
    """
    if st["trace"].size == 0:
        return {
            "status": "degenerate",
            "full_cost": float("nan"),
            "delay_cost": float("nan"),
            "edge_cost": float("nan"),
            "delay_to_full": float("nan"),
            "edge_to_full": float("nan"),
            "trace_mean": float("nan"),
            "stress_avoidance": float("nan"),
            "path_trace_mean_proxy": float("nan"),
        }

    tau = np.clip(
        st["tau_tau"] + coupling_weight * (st["tau_r"] + st["tau_x"]),
        min_cost,
        max_cost,
    )
    rnd = np.clip(
        st["rr"] + coupling_weight * (st["tau_r"] + st["r_x"]),
        min_cost,
        max_cost,
    )
    edge = np.clip(
        st["xx"] + coupling_weight * (st["tau_x"] + st["r_x"]),
        min_cost,
        max_cost,
    )

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

    full = float(dp[-1, -1, -1])
    delay = float(np.sum(tau[1:, cr, ce])) if A > 1 else 0.0
    edge_cost = float(np.sum(edge[ca, cr, 1:])) if E > 1 else 0.0
    trace = st["trace"]
    trace_mean = float(np.mean(trace))

    # A simple route trace proxy; direct DP predecessor reconstruction is not
    # needed for the signature, and this remains raw/no-normalization.
    diag_vals = []
    steps = max(A, R, E)
    for k in range(steps):
        a = min(A - 1, round(k * (A - 1) / max(steps - 1, 1)))
        r = min(R - 1, round(k * (R - 1) / max(steps - 1, 1)))
        e = min(E - 1, round(k * (E - 1) / max(steps - 1, 1)))
        diag_vals.append(trace[a, r, e])
    path_trace_mean = float(np.mean(diag_vals)) if diag_vals else float("nan")
    stress_avoidance = trace_mean - path_trace_mean if math.isfinite(trace_mean) else float("nan")

    return {
        "status": "ok",
        "full_cost": full,
        "delay_cost": delay,
        "edge_cost": edge_cost,
        "delay_to_full": delay / (full + EPS),
        "edge_to_full": edge_cost / (full + EPS),
        "trace_mean": trace_mean,
        "stress_avoidance": float(stress_avoidance),
        "path_trace_mean_proxy": path_trace_mean,
    }


def raw_damage(real: Dict[str, float], ablated: Dict[str, float]) -> Dict[str, float]:
    """
    Direct raw damage. No normalization.

    Primary damage:
        abs route deltas + avoidance loss
    """
    full_delta = safe_float(ablated.get("full_cost")) - safe_float(real.get("full_cost"))
    delay_delta = safe_float(ablated.get("delay_cost")) - safe_float(real.get("delay_cost"))
    edge_delta = safe_float(ablated.get("edge_cost")) - safe_float(real.get("edge_cost"))

    if not math.isfinite(full_delta):
        full_delta = float("nan")
    if not math.isfinite(delay_delta):
        delay_delta = float("nan")
    if not math.isfinite(edge_delta):
        edge_delta = float("nan")

    real_avoid = safe_float(real.get("stress_avoidance"), 0.0)
    ab_avoid = safe_float(ablated.get("stress_avoidance"), 0.0)
    avoidance_loss = max(0.0, real_avoid - ab_avoid)

    damage = (
        abs(full_delta if math.isfinite(full_delta) else 0.0)
        + abs(delay_delta if math.isfinite(delay_delta) else 0.0)
        + abs(edge_delta if math.isfinite(edge_delta) else 0.0)
        + avoidance_loss
    )

    return {
        "damage": float(damage),
        "full_delta": float(full_delta),
        "delay_delta": float(delay_delta),
        "edge_delta": float(edge_delta),
        "avoidance_loss": float(avoidance_loss),
        "full_abs_delta": abs(float(full_delta)) if math.isfinite(full_delta) else float("nan"),
        "delay_abs_delta": abs(float(delay_delta)) if math.isfinite(delay_delta) else float("nan"),
        "edge_abs_delta": abs(float(edge_delta)) if math.isfinite(edge_delta) else float("nan"),
        "full_rel_delta": float(full_delta / (abs(safe_float(real.get("full_cost"))) + EPS)) if math.isfinite(full_delta) else float("nan"),
        "delay_rel_delta": float(delay_delta / (abs(safe_float(real.get("delay_cost"))) + EPS)) if math.isfinite(delay_delta) else float("nan"),
        "edge_rel_delta": float(edge_delta / (abs(safe_float(real.get("edge_cost"))) + EPS)) if math.isfinite(edge_delta) else float("nan"),
    }


# --------------------------------------------------------------------------------------------------
# ABLATIONS — RAW ONLY
# --------------------------------------------------------------------------------------------------

def replace_edge_with_cell_marginal(block: np.ndarray, edge_index: int, rng: np.random.Generator) -> np.ndarray:
    """
    Remove one edge by replacing its values with delay/round cell marginals.

    This preserves rough per-cell intensity while removing that edge's shot-level
    participation in the QPU scaffold.
    """
    f = block.copy()
    if edge_index < 0 or edge_index >= f.shape[3]:
        return f

    # p shape: delay, 1, round
    p = f[:, :, :, edge_index].mean(axis=1, keepdims=True)
    f[:, :, :, edge_index] = (rng.random(f[:, :, :, edge_index].shape) < p).astype(np.uint8)
    return f


def replace_round_with_cell_marginal(block: np.ndarray, round_index: int, rng: np.random.Generator) -> np.ndarray:
    """
    Remove one round by replacing it with delay/edge cell marginals.
    """
    f = block.copy()
    if round_index < 0 or round_index >= f.shape[2]:
        return f

    # p shape: delay, 1, edge
    p = f[:, :, round_index, :].mean(axis=1, keepdims=True)
    f[:, :, round_index, :] = (rng.random(f[:, :, round_index, :].shape) < p).astype(np.uint8)
    return f


def replace_round_edge_with_cell_marginal(
    block: np.ndarray,
    round_index: int,
    edge_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Remove one round-edge track across delays by replacing it with delay-specific
    shot marginals.

    This targets the QPU projection scaffold at the round×edge level.
    """
    f = block.copy()
    if round_index < 0 or round_index >= f.shape[2]:
        return f
    if edge_index < 0 or edge_index >= f.shape[3]:
        return f

    # p shape: delay, 1
    p = f[:, :, round_index, edge_index].mean(axis=1, keepdims=True)
    f[:, :, round_index, edge_index] = (
        rng.random(f[:, :, round_index, edge_index].shape) < p
    ).astype(np.uint8)
    return f


def shuffle_edges(block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = block.copy()
    for d in range(f.shape[0]):
        for sh in range(f.shape[1]):
            for r in range(f.shape[2]):
                f[d, sh, r, :] = f[d, sh, r, rng.permutation(f.shape[3])]
    return f


def shuffle_rounds(block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = block.copy()
    for d in range(f.shape[0]):
        for sh in range(f.shape[1]):
            for e in range(f.shape[3]):
                f[d, sh, :, e] = f[d, sh, rng.permutation(f.shape[2]), e]
    return f


def uniform_by_cell(block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = block.mean(axis=1, keepdims=True)
    return (rng.random(block.shape) < p).astype(np.uint8)


def all_uniform(block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = float(block.mean())
    return (rng.random(block.shape) < p).astype(np.uint8)


# --------------------------------------------------------------------------------------------------
# SIGNATURE EXTRACTION
# --------------------------------------------------------------------------------------------------

def extract_raw_profiles(block: np.ndarray, st: Dict[str, np.ndarray], routes: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Raw QPU projection signature components.

    These are saved as qproj artifacts, not used as normalized embeddings.
    """
    f = block.astype(np.float64)  # delay, shot, round, edge

    field_edge = f.mean(axis=(0, 1, 2))          # edge
    field_round = f.mean(axis=(0, 1, 3))         # round
    field_delay = f.mean(axis=(1, 2, 3))         # delay
    field_round_edge = f.mean(axis=(0, 1))       # round, edge
    field_delay_edge = f.mean(axis=(1, 2))       # delay, edge
    field_delay_round = f.mean(axis=(1, 3))      # delay, round

    trace = st["trace"]
    if trace.size:
        stress_edge = trace.mean(axis=(0, 1))        # edge_cell
        stress_round = trace.mean(axis=(0, 2))       # round_cell
        stress_delay = trace.mean(axis=(1, 2))       # delay_cell
        stress_round_edge = trace.mean(axis=0)       # round_cell, edge_cell
        stress_delay_edge = trace.mean(axis=1)       # delay_cell, edge_cell
        stress_delay_round = trace.mean(axis=2)      # delay_cell, round_cell
    else:
        stress_edge = np.zeros((0,), dtype=np.float64)
        stress_round = np.zeros((0,), dtype=np.float64)
        stress_delay = np.zeros((0,), dtype=np.float64)
        stress_round_edge = np.zeros((0, 0), dtype=np.float64)
        stress_delay_edge = np.zeros((0, 0), dtype=np.float64)
        stress_delay_round = np.zeros((0, 0), dtype=np.float64)

    raw_geo_routes = np.asarray([
        routes["full_cost"],
        routes["delay_cost"],
        routes["edge_cost"],
        routes["delay_to_full"],
        routes["edge_to_full"],
        routes["trace_mean"],
        routes["stress_avoidance"],
        routes["path_trace_mean_proxy"],
    ], dtype=np.float64)

    return {
        "field_edge_profile": finite_clean32(field_edge),
        "field_round_profile": finite_clean32(field_round),
        "field_delay_profile": finite_clean32(field_delay),
        "field_round_edge_profile": finite_clean32(field_round_edge),
        "field_delay_edge_profile": finite_clean32(field_delay_edge),
        "field_delay_round_profile": finite_clean32(field_delay_round),
        "stress_edge_profile": finite_clean32(stress_edge),
        "stress_round_profile": finite_clean32(stress_round),
        "stress_delay_profile": finite_clean32(stress_delay),
        "stress_round_edge_profile": finite_clean32(stress_round_edge),
        "stress_delay_edge_profile": finite_clean32(stress_delay_edge),
        "stress_delay_round_profile": finite_clean32(stress_delay_round),
        "raw_geo_routes": finite_clean32(raw_geo_routes),
    }


# --------------------------------------------------------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------------------------------------------------------

def analyze_block(
    block: np.ndarray,
    *,
    mode_index: int,
    site_index: int,
    mode_name: str,
    site_name: str,
    seed: int,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> Dict[str, Any]:
    real_st = local_stress(block)
    real_routes = raw_routes_from_stress(
        real_st,
        coupling_weight=coupling_weight,
        min_cost=min_cost,
        max_cost=max_cost,
    )
    profiles = extract_raw_profiles(block, real_st, real_routes)

    edge_count = block.shape[3]
    round_count = block.shape[2]

    edge_damage = np.zeros((edge_count,), dtype=np.float64)
    round_damage = np.zeros((round_count,), dtype=np.float64)
    round_edge_damage = np.zeros((round_count, edge_count), dtype=np.float64)

    edge_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []
    round_edge_rows: List[Dict[str, Any]] = []

    # Edge ablations.
    for e in range(edge_count):
        rng = np.random.default_rng(seed + 100003 * mode_index + 1009 * site_index + 17 * e)
        ab = replace_edge_with_cell_marginal(block, e, rng)
        ab_st = local_stress(ab)
        ab_routes = raw_routes_from_stress(
            ab_st,
            coupling_weight=coupling_weight,
            min_cost=min_cost,
            max_cost=max_cost,
        )
        dmg = raw_damage(real_routes, ab_routes)
        edge_damage[e] = dmg["damage"]
        edge_rows.append({
            "mode": mode_name,
            "site": site_name,
            "mode_index": mode_index,
            "site_index": site_index,
            "edge_index": e,
            **dmg,
        })

    # Round ablations.
    for r in range(round_count):
        rng = np.random.default_rng(seed + 200003 * mode_index + 2009 * site_index + 19 * r)
        ab = replace_round_with_cell_marginal(block, r, rng)
        ab_st = local_stress(ab)
        ab_routes = raw_routes_from_stress(
            ab_st,
            coupling_weight=coupling_weight,
            min_cost=min_cost,
            max_cost=max_cost,
        )
        dmg = raw_damage(real_routes, ab_routes)
        round_damage[r] = dmg["damage"]
        round_rows.append({
            "mode": mode_name,
            "site": site_name,
            "mode_index": mode_index,
            "site_index": site_index,
            "round_index": r,
            **dmg,
        })

    # Round-edge ablations.
    for r in range(round_count):
        for e in range(edge_count):
            rng = np.random.default_rng(seed + 300003 * mode_index + 3001 * site_index + 101 * r + e)
            ab = replace_round_edge_with_cell_marginal(block, r, e, rng)
            ab_st = local_stress(ab)
            ab_routes = raw_routes_from_stress(
                ab_st,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
            )
            dmg = raw_damage(real_routes, ab_routes)
            round_edge_damage[r, e] = dmg["damage"]
            round_edge_rows.append({
                "mode": mode_name,
                "site": site_name,
                "mode_index": mode_index,
                "site_index": site_index,
                "round_index": r,
                "edge_index": e,
                **dmg,
            })

    # Coarse controls, raw damage only.
    coarse_rows: List[Dict[str, Any]] = []
    coarse_controls = {
        "edge_shuffle": shuffle_edges,
        "round_shuffle": shuffle_rounds,
        "round_reverse": lambda b, rng: b[:, :, ::-1, :].copy(),
        "edge_reverse": lambda b, rng: b[:, :, :, ::-1].copy(),
        "delay_shuffle": lambda b, rng: b[rng.permutation(b.shape[0]), :, :, :].copy(),
        "delay_reverse": lambda b, rng: b[::-1, :, :, :].copy(),
        "uniform_by_cell": uniform_by_cell,
        "all_uniform": all_uniform,
    }

    for i, (name, fn) in enumerate(coarse_controls.items()):
        rng = np.random.default_rng(seed + 400003 * mode_index + 4001 * site_index + i)
        ab = fn(block, rng)
        ab_st = local_stress(ab)
        ab_routes = raw_routes_from_stress(
            ab_st,
            coupling_weight=coupling_weight,
            min_cost=min_cost,
            max_cost=max_cost,
        )
        dmg = raw_damage(real_routes, ab_routes)
        coarse_rows.append({
            "mode": mode_name,
            "site": site_name,
            "mode_index": mode_index,
            "site_index": site_index,
            "control": name,
            **dmg,
        })

    summary = {
        "mode": mode_name,
        "site": site_name,
        "mode_index": mode_index,
        "site_index": site_index,
        "full_cost": real_routes["full_cost"],
        "delay_cost": real_routes["delay_cost"],
        "edge_cost": real_routes["edge_cost"],
        "delay_to_full": real_routes["delay_to_full"],
        "edge_to_full": real_routes["edge_to_full"],
        "trace_mean": real_routes["trace_mean"],
        "stress_avoidance": real_routes["stress_avoidance"],
        "edge_damage_mean": float(np.mean(edge_damage)),
        "edge_damage_max": float(np.max(edge_damage)),
        "edge_damage_argmax": int(np.argmax(edge_damage)),
        "round_damage_mean": float(np.mean(round_damage)),
        "round_damage_max": float(np.max(round_damage)),
        "round_damage_argmax": int(np.argmax(round_damage)),
        "round_edge_damage_mean": float(np.mean(round_edge_damage)),
        "round_edge_damage_max": float(np.max(round_edge_damage)),
        "round_edge_damage_argmax_round": int(np.unravel_index(np.argmax(round_edge_damage), round_edge_damage.shape)[0]),
        "round_edge_damage_argmax_edge": int(np.unravel_index(np.argmax(round_edge_damage), round_edge_damage.shape)[1]),
    }

    return {
        "summary": summary,
        "profiles": profiles,
        "edge_damage": finite_clean32(edge_damage),
        "round_damage": finite_clean32(round_damage),
        "round_edge_damage": finite_clean32(round_edge_damage),
        "edge_rows": edge_rows,
        "round_rows": round_rows,
        "round_edge_rows": round_edge_rows,
        "coarse_rows": coarse_rows,
    }


def run_probe(
    ts: Dict[str, Any],
    *,
    seed: int,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> Dict[str, Any]:
    field = ts["field"]
    modes = ts["modes"]
    sites = ts["delay_sites"]

    M, S = field.shape[0], field.shape[1]
    edge_count = field.shape[5]
    round_count = field.shape[4]
    delay_count = field.shape[2]

    # Stress lattice shapes are one less in delay/round/edge.
    stress_delay_cells = max(delay_count - 1, 0)
    stress_round_cells = max(round_count - 1, 0)
    stress_edge_cells = max(edge_count - 1, 0)

    summaries: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []
    round_edge_rows: List[Dict[str, Any]] = []
    coarse_rows: List[Dict[str, Any]] = []

    # Signature arrays.
    field_edge_profile = np.zeros((M, S, edge_count), dtype=np.float32)
    field_round_profile = np.zeros((M, S, round_count), dtype=np.float32)
    field_delay_profile = np.zeros((M, S, delay_count), dtype=np.float32)
    field_round_edge_profile = np.zeros((M, S, round_count, edge_count), dtype=np.float32)
    field_delay_edge_profile = np.zeros((M, S, delay_count, edge_count), dtype=np.float32)
    field_delay_round_profile = np.zeros((M, S, delay_count, round_count), dtype=np.float32)

    stress_edge_profile = np.zeros((M, S, stress_edge_cells), dtype=np.float32)
    stress_round_profile = np.zeros((M, S, stress_round_cells), dtype=np.float32)
    stress_delay_profile = np.zeros((M, S, stress_delay_cells), dtype=np.float32)
    stress_round_edge_profile = np.zeros((M, S, stress_round_cells, stress_edge_cells), dtype=np.float32)
    stress_delay_edge_profile = np.zeros((M, S, stress_delay_cells, stress_edge_cells), dtype=np.float32)
    stress_delay_round_profile = np.zeros((M, S, stress_delay_cells, stress_round_cells), dtype=np.float32)

    raw_geo_routes = np.zeros((M, S, 8), dtype=np.float32)
    edge_damage = np.zeros((M, S, edge_count), dtype=np.float32)
    round_damage = np.zeros((M, S, round_count), dtype=np.float32)
    round_edge_damage = np.zeros((M, S, round_count, edge_count), dtype=np.float32)

    for mi, mode_name in enumerate(modes):
        for si, site_name in enumerate(sites):
            block = field[mi, si]
            result = analyze_block(
                block,
                mode_index=mi,
                site_index=si,
                mode_name=mode_name,
                site_name=site_name,
                seed=seed,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
            )

            summaries.append(result["summary"])
            edge_rows.extend(result["edge_rows"])
            round_rows.extend(result["round_rows"])
            round_edge_rows.extend(result["round_edge_rows"])
            coarse_rows.extend(result["coarse_rows"])

            p = result["profiles"]
            field_edge_profile[mi, si, :] = p["field_edge_profile"]
            field_round_profile[mi, si, :] = p["field_round_profile"]
            field_delay_profile[mi, si, :] = p["field_delay_profile"]
            field_round_edge_profile[mi, si, :, :] = p["field_round_edge_profile"]
            field_delay_edge_profile[mi, si, :, :] = p["field_delay_edge_profile"]
            field_delay_round_profile[mi, si, :, :] = p["field_delay_round_profile"]

            if stress_edge_cells:
                stress_edge_profile[mi, si, :] = p["stress_edge_profile"]
            if stress_round_cells:
                stress_round_profile[mi, si, :] = p["stress_round_profile"]
            if stress_delay_cells:
                stress_delay_profile[mi, si, :] = p["stress_delay_profile"]
            if stress_round_cells and stress_edge_cells:
                stress_round_edge_profile[mi, si, :, :] = p["stress_round_edge_profile"]
            if stress_delay_cells and stress_edge_cells:
                stress_delay_edge_profile[mi, si, :, :] = p["stress_delay_edge_profile"]
            if stress_delay_cells and stress_round_cells:
                stress_delay_round_profile[mi, si, :, :] = p["stress_delay_round_profile"]

            raw_geo_routes[mi, si, :] = p["raw_geo_routes"]
            edge_damage[mi, si, :] = result["edge_damage"]
            round_damage[mi, si, :] = result["round_damage"]
            round_edge_damage[mi, si, :, :] = result["round_edge_damage"]

    edge_aggregate = aggregate_by_index(edge_rows, "edge_index")
    round_aggregate = aggregate_by_index(round_rows, "round_index")
    round_edge_aggregate = aggregate_by_pair(round_edge_rows, "round_index", "edge_index")
    coarse_aggregate = aggregate_by_control(coarse_rows)

    signature = {
        "field_edge_profile": field_edge_profile,
        "field_round_profile": field_round_profile,
        "field_delay_profile": field_delay_profile,
        "field_round_edge_profile": field_round_edge_profile,
        "field_delay_edge_profile": field_delay_edge_profile,
        "field_delay_round_profile": field_delay_round_profile,
        "stress_edge_profile": stress_edge_profile,
        "stress_round_profile": stress_round_profile,
        "stress_delay_profile": stress_delay_profile,
        "stress_round_edge_profile": stress_round_edge_profile,
        "stress_delay_edge_profile": stress_delay_edge_profile,
        "stress_delay_round_profile": stress_delay_round_profile,
        "raw_geo_routes": raw_geo_routes,
        "edge_damage": edge_damage,
        "round_damage": round_damage,
        "round_edge_damage": round_edge_damage,
    }

    return {
        "summary": summaries,
        "edge_rows": edge_rows,
        "round_rows": round_rows,
        "round_edge_rows": round_edge_rows,
        "coarse_rows": coarse_rows,
        "edge_aggregate": edge_aggregate,
        "round_aggregate": round_aggregate,
        "round_edge_aggregate": round_edge_aggregate,
        "coarse_aggregate": coarse_aggregate,
        "signature": signature,
    }


# --------------------------------------------------------------------------------------------------
# AGGREGATION
# --------------------------------------------------------------------------------------------------

def aggregate_by_index(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(int(r[key]), []).append(r)

    out = []
    for idx, vals in sorted(groups.items()):
        dmg = np.asarray([safe_float(v["damage"]) for v in vals], dtype=np.float64)
        full = np.asarray([safe_float(v["full_abs_delta"]) for v in vals], dtype=np.float64)
        delay = np.asarray([safe_float(v["delay_abs_delta"]) for v in vals], dtype=np.float64)
        edge = np.asarray([safe_float(v["edge_abs_delta"]) for v in vals], dtype=np.float64)
        out.append({
            key: idx,
            "mean_damage": float(np.mean(dmg)),
            "std_damage": float(np.std(dmg)),
            "max_damage": float(np.max(dmg)),
            "mean_full_abs_delta": float(np.mean(full)),
            "mean_delay_abs_delta": float(np.mean(delay)),
            "mean_edge_abs_delta": float(np.mean(edge)),
            "n": int(len(vals)),
        })
    return out


def aggregate_by_pair(rows: List[Dict[str, Any]], key_a: str, key_b: str) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((int(r[key_a]), int(r[key_b])), []).append(r)

    out = []
    for (a, b), vals in sorted(groups.items()):
        dmg = np.asarray([safe_float(v["damage"]) for v in vals], dtype=np.float64)
        out.append({
            key_a: a,
            key_b: b,
            "mean_damage": float(np.mean(dmg)),
            "std_damage": float(np.std(dmg)),
            "max_damage": float(np.max(dmg)),
            "n": int(len(vals)),
        })
    return out


def aggregate_by_control(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r["control"]), []).append(r)

    out = []
    for control, vals in sorted(groups.items()):
        dmg = np.asarray([safe_float(v["damage"]) for v in vals], dtype=np.float64)
        out.append({
            "control": control,
            "mean_damage": float(np.mean(dmg)),
            "std_damage": float(np.std(dmg)),
            "max_damage": float(np.max(dmg)),
            "n": int(len(vals)),
        })
    out.sort(key=lambda r: r["mean_damage"], reverse=True)
    return out


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


def plot_edge_round_damage(edge_agg: List[Dict[str, Any]], round_agg: List[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAVE_MPL:
        return

    if edge_agg:
        labels = [str(r["edge_index"]) for r in edge_agg]
        vals = [r["mean_damage"] for r in edge_agg]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        ax.bar(np.arange(len(labels)), vals)
        ax.set_title("T_S Probe 06 — raw edge damage")
        ax.set_xlabel("edge index")
        ax.set_ylabel("mean raw damage")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(out_dir / "edge_damage.png", bbox_inches="tight")
        plt.close(fig)

    if round_agg:
        labels = [str(r["round_index"]) for r in round_agg]
        vals = [r["mean_damage"] for r in round_agg]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        ax.bar(np.arange(len(labels)), vals)
        ax.set_title("T_S Probe 06 — raw round damage")
        ax.set_xlabel("round index")
        ax.set_ylabel("mean raw damage")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(out_dir / "round_damage.png", bbox_inches="tight")
        plt.close(fig)


def plot_round_edge_heatmap(round_edge_agg: List[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAVE_MPL or not round_edge_agg:
        return

    max_r = max(int(r["round_index"]) for r in round_edge_agg)
    max_e = max(int(r["edge_index"]) for r in round_edge_agg)
    heat = np.zeros((max_r + 1, max_e + 1), dtype=np.float64)

    for r in round_edge_agg:
        heat[int(r["round_index"]), int(r["edge_index"])] = float(r["mean_damage"])

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=160)
    im = ax.imshow(heat, aspect="auto")
    ax.set_title("T_S Probe 06 — raw round×edge damage")
    ax.set_xlabel("edge index")
    ax.set_ylabel("round index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean raw damage")
    fig.savefig(out_dir / "round_edge_damage_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_coarse_damage(coarse_agg: List[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAVE_MPL or not coarse_agg:
        return

    labels = [r["control"] for r in coarse_agg]
    vals = [r["mean_damage"] for r in coarse_agg]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax.bar(np.arange(len(labels)), vals)
    ax.set_title("T_S Probe 06 — coarse raw structure damage")
    ax.set_ylabel("mean raw damage")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out_dir / "coarse_raw_damage.png", bbox_inches="tight")
    plt.close(fig)


def save_signature_npz(
    signature: Dict[str, np.ndarray],
    ts: Dict[str, Any],
    job_id: str,
    out_path: Path,
) -> None:
    payload = {
        "schema": np.array("ts_qpu_projection_signature_raw_damage"),
        "job_id": np.array(job_id),
        "modes": np.asarray(ts["modes"]),
        "delay_sites": np.asarray(ts["delay_sites"]),
        "delays": np.asarray(ts["delays"], dtype=np.int64),
        "delay_unit": np.array(ts["delay_unit"]),
        "rounds": np.array(ts["rounds"], dtype=np.int64),
        "channels": np.array(ts["channels"], dtype=np.int64),
        "edges": np.array(ts["edges"], dtype=np.int64),
    }
    payload.update(signature)
    np.savez_compressed(out_path, **payload)


def print_summary(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 132)
    print("  T_S PROBE 06 — QPU RAW-DAMAGE PROJECTION SIGNATURE SUMMARY")
    print("=" * 132)
    print(
        f"  {'mode':>13} | {'site':>14} | {'full':>8} | {'delay':>8} | {'edge':>8} | "
        f"{'trace':>8} | {'edge max':>9} | {'round max':>9} | {'r×e max':>9}"
    )
    print("  " + "-" * 130)

    for r in report["summary"]:
        print(
            f"  {r['mode']:>13} | {r['site']:>14} | "
            f"{safe_float(r['full_cost']):>8.4f} | "
            f"{safe_float(r['delay_cost']):>8.4f} | "
            f"{safe_float(r['edge_cost']):>8.4f} | "
            f"{safe_float(r['trace_mean']):>8.5f} | "
            f"{safe_float(r['edge_damage_max']):>9.5f} | "
            f"{safe_float(r['round_damage_max']):>9.5f} | "
            f"{safe_float(r['round_edge_damage_max']):>9.5f}"
        )

    print("\n" + "=" * 132)
    print("  COARSE RAW STRUCTURE DAMAGE")
    print("=" * 132)
    print(f"  {'control':>18} | {'mean damage':>12} | {'std':>10} | {'max':>10} | {'n':>4}")
    print("  " + "-" * 68)
    for r in report["coarse_aggregate"]:
        print(
            f"  {r['control']:>18} | {r['mean_damage']:>12.6f} | "
            f"{r['std_damage']:>10.6f} | {r['max_damage']:>10.6f} | {r['n']:>4}"
        )

    print("\nTop edge damages:")
    for r in sorted(report["edge_aggregate"], key=lambda x: x["mean_damage"], reverse=True)[:5]:
        print(f"  edge {r['edge_index']}: mean={r['mean_damage']:.6f}, max={r['max_damage']:.6f}")

    print("\nTop round damages:")
    for r in sorted(report["round_aggregate"], key=lambda x: x["mean_damage"], reverse=True)[:5]:
        print(f"  round {r['round_index']}: mean={r['mean_damage']:.6f}, max={r['max_damage']:.6f}")

    print("\nTop round×edge damages:")
    for r in sorted(report["round_edge_aggregate"], key=lambda x: x["mean_damage"], reverse=True)[:8]:
        print(
            f"  round {r['round_index']}, edge {r['edge_index']}: "
            f"mean={r['mean_damage']:.6f}, max={r['max_damage']:.6f}"
        )


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T_S Probe 06 — QPU projection finalizer using raw damage only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz", default=None, help="Dumped T_S QPU .npz. Defaults to latest.")
    p.add_argument("--meta", default=None, help="Optional T_S job metadata JSON.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--coupling-weight", type=float, default=0.50)
    p.add_argument("--min-cost", type=float, default=1e-9)
    p.add_argument("--max-cost", type=float, default=1e9)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    npz_path, meta_path, job_id = resolve_inputs(args.npz, args.meta)
    ts = load_ts(npz_path)
    meta = load_json(meta_path) if meta_path and meta_path.exists() else None

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"ts_qpu_projection_raw_damage_{job_id}_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 132)
    print("  T_S PROBE 06 — QPU PROJECTION FINALIZER: RAW DAMAGE SIGNATURE")
    print("=" * 132)
    print(f"  NPZ        : {npz_path}")
    print(f"  Metadata   : {meta_path if meta_path else '(not provided)'}")
    print(f"  Out dir    : {out_dir}")
    print(f"  Field shape: {ts['field'].shape}")
    print(f"  Modes      : {ts['modes']}")
    print(f"  Sites      : {ts['delay_sites']}")
    print(f"  Delays     : {ts['delays']} {ts['delay_unit']}")
    print("  Method     : raw route/stress damage only; no normalization, no cosine, no classifier")

    core = run_probe(
        ts,
        seed=int(args.seed),
        coupling_weight=float(args.coupling_weight),
        min_cost=float(args.min_cost),
        max_cost=float(args.max_cost),
    )

    signature_npz = out_dir / "qpu_projection_signature_raw_damage.npz"
    save_signature_npz(core["signature"], ts, job_id, signature_npz)

    report = {
        "schema": "ts_probe6_qpu_projection_raw_damage",
        "description": (
            "Corrected QPU-side projection finalizer. Uses direct raw geo route damage under "
            "edge, round, and round-edge ablations. No normalization, cosine, or classifier "
            "framing is used."
        ),
        "job_id": job_id,
        "npz": str(npz_path),
        "meta": str(meta_path) if meta_path else None,
        "signature_file": str(signature_npz),
        "settings": {
            "seed": int(args.seed),
            "coupling_weight": float(args.coupling_weight),
            "min_cost": float(args.min_cost),
            "max_cost": float(args.max_cost),
            "normalization": "none",
            "comparison": "direct_raw_damage",
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
        "calibration_summary": calibration_summary(meta),
        "summary": core["summary"],
        "edge_rows": core["edge_rows"],
        "round_rows": core["round_rows"],
        "round_edge_rows": core["round_edge_rows"],
        "coarse_rows": core["coarse_rows"],
        "edge_aggregate": core["edge_aggregate"],
        "round_aggregate": core["round_aggregate"],
        "round_edge_aggregate": core["round_edge_aggregate"],
        "coarse_aggregate": core["coarse_aggregate"],
    }

    with open(out_dir / "qpu_projection_raw_damage_report.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2)

    write_csv(report["summary"], out_dir / "qpu_projection_raw_damage_summary.csv")
    write_csv(report["edge_rows"], out_dir / "edge_damage_rows.csv")
    write_csv(report["round_rows"], out_dir / "round_damage_rows.csv")
    write_csv(report["round_edge_rows"], out_dir / "round_edge_damage_rows.csv")
    write_csv(report["coarse_rows"], out_dir / "coarse_raw_damage_rows.csv")
    write_csv(report["edge_aggregate"], out_dir / "edge_damage_aggregate.csv")
    write_csv(report["round_aggregate"], out_dir / "round_damage_aggregate.csv")
    write_csv(report["round_edge_aggregate"], out_dir / "round_edge_damage_aggregate.csv")
    write_csv(report["coarse_aggregate"], out_dir / "coarse_raw_damage_aggregate.csv")

    if not args.no_plots:
        plot_edge_round_damage(report["edge_aggregate"], report["round_aggregate"], out_dir)
        plot_round_edge_heatmap(report["round_edge_aggregate"], out_dir)
        plot_coarse_damage(report["coarse_aggregate"], out_dir)

    print_summary(report)

    print(f"\n[SAVED] {out_dir}")
    print(f"[SIGNATURE] {signature_npz}")
    print("=" * 132 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"[FATAL] {type(e).__name__}: {e}")
