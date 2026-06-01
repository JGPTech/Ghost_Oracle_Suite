#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
T_S PROBE 03 — RAW-FIRST JUMPGEO ROUTING
====================================================================================================
Adaptive geometry probe for T_S — Temporal Stress Metric.

Probe 01:
    QPU-generated delay/round/edge field exists and is structured.

Probe 02:
    That field defines a geodesic-like geometry.  Raw works.  Trace normalization
    usually improves global/path geometry.  Low-order power expansion helps
    sharpen delay-only structure under stress.

Probe 03:
    Convert that discovery into a fast-path routing policy:

        Use raw geometry by default.
        Jump to trace/fro normalization or low-order power expansion only when
        the raw route is stressed, weak, or path-specific evidence says the
        extra operation is worth it.

The production motivation is speed.  Raw geo is the target GPU path:
no global reductions, no normalization pass, no exponentiation.  Transform
jumps are adaptive fallbacks, not the default.

Expected location
-----------------
This probe is intended to live in:

    ghost_oracle/T_S/probes/t_s_probe3_raw_first_jumpgeo.py

with the probe path convention:

    HERE = Path(__file__).resolve().parent
    DATA_DIR = HERE.parent / "data"
    ANALYSIS_DIR = HERE / "analysis"

Inputs
------
Preferred input:
    Probe 02 output JSON:
        ghost_oracle/T_S/probes/analysis/ts_geo_<JOB_ID>/geo_reconstruction_report.json

Fallback input:
    The dumped T_S npz.  If --geo-report is not provided and the report cannot
    be found, this script can recompute the small transform set needed for
    Raw-First JumpGeo.

Raw-first policy
----------------
For each mode × delay_site × path:

    1. Evaluate raw.
    2. If raw is healthy, keep raw.
    3. If raw is weak/stressed:
        - delay_only  -> try pow_exp_2, pow_exp_4, raw
        - edge_only   -> try trace_norm, fro_norm, raw
        - full/delay-round -> try trace_norm, raw, pow_exp_2
    4. Select by penalized score:

        score = geo_gap_z
                + avoidance_weight * max(stress_avoidance, 0)
                - negative_avoidance_penalty * max(-stress_avoidance, 0)
                - op_penalty(transform)

The algorithm is deliberately biased toward raw.  A transform has to beat raw
after the operation penalty to win.

Outputs
-------
    raw_first_jumpgeo_report.json
    raw_first_jumpgeo_rows.csv
    raw_first_jumpgeo_summary.csv
    raw_usage_summary.png
    jump_gain_summary.png

Usage
-----
Use existing Probe 02 report automatically:

    python ghost_oracle/T_S/probes/t_s_probe3_raw_first_jumpgeo.py

Specific report:

    python ghost_oracle/T_S/probes/t_s_probe3_raw_first_jumpgeo.py ^
        --geo-report ghost_oracle/T_S/probes/analysis/ts_geo_<JOB_ID>/geo_reconstruction_report.json

Fallback recompute from npz:

    python ghost_oracle/T_S/probes/t_s_probe3_raw_first_jumpgeo.py ^
        --npz ghost_oracle/T_S/data/ts_data_<JOB_ID>.npz

====================================================================================================
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    plt = None
    _HAVE_MPL = False


# --------------------------------------------------------------------------------------------------
# PATHS — probe convention
# --------------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
ANALYSIS_DIR = HERE / "analysis"


# --------------------------------------------------------------------------------------------------
# CONSTANTS / BASIC HELPERS
# --------------------------------------------------------------------------------------------------

EPS = 1e-12

DEFAULT_NEEDED_TRANSFORMS = [
    "raw",
    "trace_norm",
    "fro_norm",
    "pow_exp_2",
    "pow_exp_4",
]

CONTROL_MODES = [
    "real",
    "shot_shuffle",
    "delay_shuffle",
    "round_shuffle",
    "edge_shuffle",
    "uniform_field",
    "all_uniform",
    "delay_reverse",
    "round_reverse",
    "edge_reverse",
]


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


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def finite_clean(
    x: np.ndarray,
    *,
    nan: float = 0.0,
    posinf: Optional[float] = None,
    neginf: float = 0.0,
    clip_abs: Optional[float] = None,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if posinf is None:
        posinf = clip_abs if clip_abs is not None else 1e12
    arr = np.nan_to_num(arr, nan=nan, posinf=posinf, neginf=neginf)
    if clip_abs is not None:
        arr = np.clip(arr, -float(clip_abs), float(clip_abs))
    return arr


def bits(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x).astype(np.int64) & 1).astype(np.uint8)


def resolve_npz(npz_arg: Optional[str]) -> tuple[Optional[Path], Optional[str]]:
    if npz_arg:
        p = Path(npz_arg)
        stem = p.stem
        job_id = stem.split("_")[-1] if "_" in stem else stem
        return p, job_id

    latest = DATA_DIR / "latest_ts_data.json"
    if latest.exists():
        obj = load_json(latest)
        p = Path(obj["npz"])
        return p, str(obj.get("job_id", p.stem))

    return None, None


def find_latest_geo_report(job_id: Optional[str]) -> Optional[Path]:
    """
    Find an existing Probe 02 report.

    Preference:
        analysis/ts_geo_<job_id>/geo_reconstruction_report.json
    Fallback:
        newest analysis/ts_geo_*/geo_reconstruction_report.json by modified time.
    """
    if job_id:
        p = ANALYSIS_DIR / f"ts_geo_{job_id}" / "geo_reconstruction_report.json"
        if p.exists():
            return p

    candidates = list(ANALYSIS_DIR.glob("ts_geo_*/geo_reconstruction_report.json"))
    if candidates:
        return max(candidates, key=lambda x: x.stat().st_mtime)

    return None


def get_job_id_from_report(report: dict, report_path: Path) -> str:
    if report.get("job_id"):
        return str(report["job_id"])
    parent = report_path.parent.name
    if parent.startswith("ts_geo_"):
        return parent[len("ts_geo_"):]
    return report_path.stem


# --------------------------------------------------------------------------------------------------
# REPORT ROW LOADING
# --------------------------------------------------------------------------------------------------

def rows_from_geo_report(report: dict) -> List[Dict[str, Any]]:
    rows = list(report.get("rows", []))
    if not rows:
        raise ValueError("geo report contains no rows")
    return rows


def filter_candidate_rows(
    rows: List[Dict[str, Any]],
    needed_transforms: List[str],
) -> List[Dict[str, Any]]:
    out = []
    needed = set(needed_transforms)
    for r in rows:
        if r.get("transform") in needed and r.get("status") == "ok":
            out.append(r)
    return out


# --------------------------------------------------------------------------------------------------
# FALLBACK GEO RECOMPUTE
# This is a compact copy of the Probe 02 machinery for only the transforms needed by Raw-First
# JumpGeo.  If Probe 02 report exists, these functions are not used.
# --------------------------------------------------------------------------------------------------

@dataclass
class StressLocal:
    tau_tau: np.ndarray
    rr: np.ndarray
    xx: np.ndarray
    tau_r: np.ndarray
    tau_x: np.ndarray
    r_x: np.ndarray
    trace: np.ndarray

    def component_dict(self) -> Dict[str, np.ndarray]:
        return {
            "tau_tau": self.tau_tau,
            "rr": self.rr,
            "xx": self.xx,
            "tau_r": self.tau_r,
            "tau_x": self.tau_x,
            "r_x": self.r_x,
            "trace": self.trace,
        }


@dataclass
class TransformSpec:
    name: str
    kind: str
    value: Optional[float] = None


@dataclass
class GeoResult:
    status: str
    geodesic_cost: float
    euclidean_steps: int
    path_length: int
    path_entropy: float
    path_concentration: float
    mean_trace_on_path: float
    mean_trace_global: float
    stress_avoidance: float
    axis_tau_fraction: float
    axis_round_fraction: float
    axis_edge_fraction: float
    visited_nodes: int
    message: str = ""


def needed_transform_specs() -> List[TransformSpec]:
    return [
        TransformSpec("raw", "raw"),
        TransformSpec("trace_norm", "trace_norm"),
        TransformSpec("fro_norm", "fro_norm"),
        TransformSpec("pow_exp_2", "pow_exp", 2.0),
        TransformSpec("pow_exp_4", "pow_exp", 4.0),
    ]


def load_ts(npz_path: Path) -> Dict[str, Any]:
    z = np.load(npz_path, allow_pickle=False)
    if "field" not in z.files:
        raise KeyError("Raw-First JumpGeo fallback recompute expects a `field` array.")

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
        "modes": str_array("modes"),
        "delay_sites": str_array("delay_sites"),
        "delays": z["delays"].astype(int).tolist() if "delays" in z.files else list(range(field.shape[2])),
        "delay_unit": str(z["delay_unit"].item()) if "delay_unit" in z.files and z["delay_unit"].shape == () else "",
        "job_id": str(z["job_id"].item()) if "job_id" in z.files and z["job_id"].shape == () else npz_path.stem,
    }

    if not obj["modes"]:
        obj["modes"] = [f"mode_{i}" for i in range(field.shape[0])]
    if not obj["delay_sites"]:
        obj["delay_sites"] = [f"site_{i}" for i in range(field.shape[1])]

    return obj


def mutate_field(field: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    f = field.copy()

    if mode == "real":
        return f

    if mode == "shot_shuffle":
        return f[:, rng.permutation(f.shape[1]), :, :]

    if mode == "delay_shuffle":
        return f[rng.permutation(f.shape[0]), :, :, :]

    if mode == "round_shuffle":
        for d in range(f.shape[0]):
            for sh in range(f.shape[1]):
                for e in range(f.shape[3]):
                    f[d, sh, :, e] = f[d, sh, rng.permutation(f.shape[2]), e]
        return f

    if mode == "edge_shuffle":
        for d in range(f.shape[0]):
            for sh in range(f.shape[1]):
                for r in range(f.shape[2]):
                    f[d, sh, r, :] = f[d, sh, r, rng.permutation(f.shape[3])]
        return f

    if mode == "uniform_field":
        p = f.mean(axis=1, keepdims=True)
        return (rng.random(f.shape) < p).astype(np.uint8)

    if mode == "all_uniform":
        p = float(f.mean())
        return (rng.random(f.shape) < p).astype(np.uint8)

    if mode == "delay_reverse":
        return f[::-1, :, :, :].copy()

    if mode == "round_reverse":
        return f[:, :, ::-1, :].copy()

    if mode == "edge_reverse":
        return f[:, :, :, ::-1].copy()

    raise ValueError(mode)


def gradients_delay_round_edge(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if f.shape[0] < 2 or f.shape[2] < 2 or f.shape[3] < 2:
        empty = np.zeros((0, f.shape[1], 0, 0), dtype=np.float64)
        return empty, empty, empty

    d_tau = np.bitwise_xor(f[1:, :, :, :], f[:-1, :, :, :]).astype(np.float64)
    d_round = np.bitwise_xor(f[:, :, 1:, :], f[:, :, :-1, :]).astype(np.float64)
    d_edge = np.bitwise_xor(f[:, :, :, 1:], f[:, :, :, :-1]).astype(np.float64)

    d_tau = d_tau[:, :, :-1, :-1]
    d_round = d_round[:-1, :, :, :-1]
    d_edge = d_edge[:-1, :, :-1, :]

    return d_tau, d_round, d_edge


def local_stress_from_field(f: np.ndarray) -> StressLocal:
    d_tau, d_round, d_edge = gradients_delay_round_edge(f)
    if d_tau.size == 0:
        z = np.zeros((0, 0, 0), dtype=np.float64)
        return StressLocal(z, z, z, z, z, z, z)

    tau_tau = np.mean(d_tau * d_tau, axis=1)
    rr = np.mean(d_round * d_round, axis=1)
    xx = np.mean(d_edge * d_edge, axis=1)
    tau_r = np.mean(d_tau * d_round, axis=1)
    tau_x = np.mean(d_tau * d_edge, axis=1)
    r_x = np.mean(d_round * d_edge, axis=1)
    trace = tau_tau + rr + xx

    return StressLocal(
        tau_tau=finite_clean(tau_tau),
        rr=finite_clean(rr),
        xx=finite_clean(xx),
        tau_r=finite_clean(tau_r),
        tau_x=finite_clean(tau_x),
        r_x=finite_clean(r_x),
        trace=finite_clean(trace),
    )


def stress_scale(stress: StressLocal, mode: str) -> float:
    vals = np.concatenate([np.ravel(v) for v in stress.component_dict().values() if v.size])
    vals = finite_clean(vals)
    if vals.size == 0:
        return 1.0

    if mode == "trace_norm":
        s = float(np.nanmean(stress.trace))
    elif mode == "fro_norm":
        s = float(np.sqrt(np.nanmean(vals * vals)))
    else:
        s = 1.0

    if not math.isfinite(s) or abs(s) < EPS:
        return 1.0
    return s


def apply_transform(
    stress: StressLocal,
    spec: TransformSpec,
    *,
    clip_abs: float,
    min_cost: float,
) -> tuple[Optional[StressLocal], Optional[str]]:
    try:
        comps = stress.component_dict()
        out: Dict[str, np.ndarray] = {}

        if spec.kind == "raw":
            for k, v in comps.items():
                out[k] = finite_clean(v, clip_abs=clip_abs)

        elif spec.kind in ("trace_norm", "fro_norm"):
            s = stress_scale(stress, spec.kind)
            for k, v in comps.items():
                out[k] = finite_clean(v / s, clip_abs=clip_abs)

        elif spec.kind == "pow_exp":
            p = float(spec.value or 1.0)
            for k, v in comps.items():
                vv = finite_clean(np.maximum(v, 0.0), clip_abs=clip_abs)
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    out[k] = finite_clean(np.power(vv, p), clip_abs=clip_abs)

        else:
            return None, f"unknown transform kind: {spec.kind}"

        tau_tau = np.maximum(out["tau_tau"], min_cost)
        rr = np.maximum(out["rr"], min_cost)
        xx = np.maximum(out["xx"], min_cost)
        tau_r = finite_clean(np.maximum(out["tau_r"], 0.0), clip_abs=clip_abs)
        tau_x = finite_clean(np.maximum(out["tau_x"], 0.0), clip_abs=clip_abs)
        r_x = finite_clean(np.maximum(out["r_x"], 0.0), clip_abs=clip_abs)
        trace = finite_clean(tau_tau + rr + xx, clip_abs=clip_abs)

        return StressLocal(tau_tau, rr, xx, tau_r, tau_x, r_x, trace), None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def neighbors_3d(node: Tuple[int, int, int], shape: Tuple[int, int, int]) -> Iterable[Tuple[Tuple[int, int, int], int]]:
    a, b, c = node
    A, B, C = shape

    if a > 0:
        yield (a - 1, b, c), 0
    if a + 1 < A:
        yield (a + 1, b, c), 0

    if b > 0:
        yield (a, b - 1, c), 1
    if b + 1 < B:
        yield (a, b + 1, c), 1

    if c > 0:
        yield (a, b, c - 1), 2
    if c + 1 < C:
        yield (a, b, c + 1), 2


def edge_cost(
    stress: StressLocal,
    u: Tuple[int, int, int],
    v: Tuple[int, int, int],
    axis: int,
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> float:
    if axis == 0:
        base = 0.5 * (stress.tau_tau[u] + stress.tau_tau[v])
        coupling = 0.5 * (stress.tau_r[u] + stress.tau_r[v] + stress.tau_x[u] + stress.tau_x[v])
    elif axis == 1:
        base = 0.5 * (stress.rr[u] + stress.rr[v])
        coupling = 0.5 * (stress.tau_r[u] + stress.tau_r[v] + stress.r_x[u] + stress.r_x[v])
    else:
        base = 0.5 * (stress.xx[u] + stress.xx[v])
        coupling = 0.5 * (stress.tau_x[u] + stress.tau_x[v] + stress.r_x[u] + stress.r_x[v])

    cost = float(base + coupling_weight * coupling)
    if not math.isfinite(cost):
        return max_cost
    return float(np.clip(cost, min_cost, max_cost))


def reconstruct_path(
    prev: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], int]],
    target: Tuple[int, int, int],
) -> tuple[List[Tuple[int, int, int]], List[int]]:
    path = [target]
    axes: List[int] = []
    cur = target
    while cur in prev:
        parent, axis = prev[cur]
        axes.append(axis)
        cur = parent
        path.append(cur)
    path.reverse()
    axes.reverse()
    return path, axes


def path_entropy_axes(axes: List[int]) -> tuple[float, float, float, float, float]:
    if not axes:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    counts = np.bincount(np.asarray(axes, dtype=int), minlength=3).astype(float)
    p = counts / max(float(counts.sum()), EPS)
    entropy = float(-np.sum([pi * math.log(pi + EPS) for pi in p]))
    concentration = float(np.max(p))
    return entropy, concentration, float(p[0]), float(p[1]), float(p[2])


def dijkstra_geodesic(
    stress: StressLocal,
    *,
    source: Tuple[int, int, int],
    target: Tuple[int, int, int],
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    max_visits: int,
) -> GeoResult:
    shape = stress.trace.shape
    if len(shape) != 3 or min(shape) <= 0:
        return GeoResult("degenerate", float("nan"), 0, 0, float("nan"), float("nan"),
                         float("nan"), float("nan"), float("nan"), float("nan"),
                         float("nan"), float("nan"), 0, f"shape={shape}")

    dist = {source: 0.0}
    prev: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], int]] = {}
    heap = [(0.0, source)]
    visited = set()

    while heap:
        cur_dist, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        if len(visited) > max_visits:
            return GeoResult("visit_cap", float("nan"), sum(abs(s - t) for s, t in zip(source, target)),
                             0, float("nan"), float("nan"), float("nan"), float(np.mean(stress.trace)),
                             float("nan"), float("nan"), float("nan"), float("nan"), len(visited),
                             f"exceeded {max_visits}")

        if u == target:
            path, axes = reconstruct_path(prev, target)
            trace_vals = finite_clean(np.asarray([stress.trace[p] for p in path], dtype=np.float64))
            mean_path = float(np.mean(trace_vals)) if trace_vals.size else float("nan")
            mean_global = float(np.mean(finite_clean(stress.trace))) if stress.trace.size else float("nan")
            entropy, concentration, p_tau, p_round, p_edge = path_entropy_axes(axes)
            return GeoResult("ok", float(cur_dist), sum(abs(s - t) for s, t in zip(source, target)),
                             len(path), entropy, concentration, mean_path, mean_global,
                             float(mean_global - mean_path), p_tau, p_round, p_edge, len(visited), "")

        if cur_dist > dist.get(u, float("inf")):
            continue

        for v, axis in neighbors_3d(u, shape):
            c = edge_cost(stress, u, v, axis,
                          coupling_weight=coupling_weight,
                          min_cost=min_cost,
                          max_cost=max_cost)
            nd = cur_dist + c
            if not math.isfinite(nd):
                continue
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = (u, axis)
                heapq.heappush(heap, (nd, v))

    return GeoResult("unreachable", float("nan"), sum(abs(s - t) for s, t in zip(source, target)),
                     0, float("nan"), float("nan"), float("nan"), float(np.mean(stress.trace)),
                     float("nan"), float("nan"), float("nan"), float("nan"), len(visited), "target not reached")


def default_endpoints(shape: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]]:
    A, B, C = shape
    if min(shape) <= 0:
        return []
    ca, cb, cc = A // 2, B // 2, C // 2
    return [
        ((0, 0, 0), (A - 1, B - 1, C - 1), "full_diag"),
        ((0, 0, cc), (A - 1, B - 1, cc), "delay_round"),
        ((0, cb, cc), (A - 1, cb, cc), "delay_only"),
        ((ca, cb, 0), (ca, cb, C - 1), "edge_only"),
    ]


def recompute_needed_geo_rows(
    npz_path: Path,
    *,
    seed: int,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    clip_abs: float,
    max_visits: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ts = load_ts(npz_path)
    field = ts["field"]
    specs = needed_transform_specs()

    rows: List[Dict[str, Any]] = []

    for mi, mode_name in enumerate(ts["modes"]):
        for si, site_name in enumerate(ts["delay_sites"]):
            block = field[mi, si]
            rng_base = seed + 10007 * mi + 1009 * si

            stress_by_control: Dict[str, StressLocal] = {}
            for ci, control in enumerate(CONTROL_MODES):
                rng = np.random.default_rng(rng_base + 7919 * ci)
                stress_by_control[control] = local_stress_from_field(mutate_field(block, control, rng))

            shape = stress_by_control["real"].trace.shape
            endpoints = default_endpoints(shape)

            for spec in specs:
                transformed: Dict[str, StressLocal] = {}
                for control, st in stress_by_control.items():
                    tr, err = apply_transform(st, spec, clip_abs=clip_abs, min_cost=min_cost)
                    if tr is not None:
                        transformed[control] = tr

                if "real" not in transformed:
                    continue

                real_trace = finite_clean(transformed["real"].trace)
                trace_mean = float(np.mean(real_trace)) if real_trace.size else float("nan")
                trace_std = float(np.std(real_trace)) if real_trace.size else float("nan")
                trace_cv = float(trace_std / (abs(trace_mean) + EPS)) if math.isfinite(trace_std) else float("nan")

                for source, target, path_name in endpoints:
                    real_geo = dijkstra_geodesic(
                        transformed["real"],
                        source=source,
                        target=target,
                        coupling_weight=coupling_weight,
                        min_cost=min_cost,
                        max_cost=max_cost,
                        max_visits=max_visits,
                    )

                    control_costs = {}
                    control_statuses = {}
                    for control, st in transformed.items():
                        if control == "real":
                            continue
                        g = dijkstra_geodesic(
                            st,
                            source=source,
                            target=target,
                            coupling_weight=coupling_weight,
                            min_cost=min_cost,
                            max_cost=max_cost,
                            max_visits=max_visits,
                        )
                        control_costs[control] = g.geodesic_cost
                        control_statuses[control] = g.status

                    finite_control_costs = [
                        safe_float(c) for c in control_costs.values()
                        if math.isfinite(safe_float(c))
                    ]

                    if finite_control_costs and math.isfinite(real_geo.geodesic_cost):
                        control_mean = float(np.mean(finite_control_costs))
                        geo_gap_mean = float(control_mean - real_geo.geodesic_cost)
                        geo_gap_z = float(geo_gap_mean / (float(np.std(finite_control_costs)) + EPS))
                    else:
                        control_mean = float("nan")
                        geo_gap_mean = float("nan")
                        geo_gap_z = float("nan")

                    rows.append({
                        "mode": mode_name,
                        "delay_site": site_name,
                        "transform": spec.name,
                        "transform_kind": spec.kind,
                        "transform_value": spec.value,
                        "path_name": path_name,
                        "source": source,
                        "target": target,
                        "shape_delay_cells": int(shape[0]),
                        "shape_round_cells": int(shape[1]),
                        "shape_edge_cells": int(shape[2]),
                        "status": real_geo.status,
                        "geodesic_cost": real_geo.geodesic_cost,
                        "euclidean_steps": real_geo.euclidean_steps,
                        "path_length": real_geo.path_length,
                        "path_entropy": real_geo.path_entropy,
                        "path_concentration": real_geo.path_concentration,
                        "axis_tau_fraction": real_geo.axis_tau_fraction,
                        "axis_round_fraction": real_geo.axis_round_fraction,
                        "axis_edge_fraction": real_geo.axis_edge_fraction,
                        "mean_trace_on_path": real_geo.mean_trace_on_path,
                        "mean_trace_global": real_geo.mean_trace_global,
                        "stress_avoidance": real_geo.stress_avoidance,
                        "visited_nodes": real_geo.visited_nodes,
                        "trace_mean": trace_mean,
                        "trace_std": trace_std,
                        "trace_cv": trace_cv,
                        "control_cost_mean": control_mean,
                        "geo_gap_mean_control_minus_real": geo_gap_mean,
                        "geo_gap_z": geo_gap_z,
                        "control_costs": control_costs,
                        "control_statuses": control_statuses,
                        "message": real_geo.message,
                    })

    meta = {
        "source": "fallback_recompute",
        "npz": str(npz_path),
        "job_id": ts.get("job_id", npz_path.stem),
        "modes": ts["modes"],
        "delay_sites": ts["delay_sites"],
        "delays": ts["delays"],
        "delay_unit": ts["delay_unit"],
    }
    return rows, meta


# --------------------------------------------------------------------------------------------------
# RAW-FIRST POLICY
# --------------------------------------------------------------------------------------------------

def op_penalty(transform: str, penalties: Dict[str, float]) -> float:
    return float(penalties.get(transform, penalties.get("default", 0.25)))


def candidate_transforms_for_path(path_name: str) -> List[str]:
    """
    Candidate order is intentionally raw-biased.
    """
    if path_name == "delay_only":
        return ["raw", "pow_exp_2", "pow_exp_4", "trace_norm"]
    if path_name == "edge_only":
        return ["raw", "trace_norm", "fro_norm"]
    return ["raw", "trace_norm", "pow_exp_2"]


def raw_health_class(
    raw: Dict[str, Any],
    *,
    raw_good_z: float,
    raw_min_avoidance: float,
    weak_z: float,
) -> tuple[str, List[str]]:
    z = safe_float(raw.get("geo_gap_z"))
    avoid = safe_float(raw.get("stress_avoidance"))
    trace_cv = safe_float(raw.get("trace_cv"))

    reasons: List[str] = []

    if not math.isfinite(z):
        reasons.append("raw_gap_z_nonfinite")
        return "bad", reasons
    if z < weak_z:
        reasons.append(f"raw_gap_z<{weak_z}")
    if math.isfinite(avoid) and avoid < raw_min_avoidance:
        reasons.append(f"raw_avoidance<{raw_min_avoidance}")
    if not math.isfinite(trace_cv):
        reasons.append("trace_cv_nonfinite")

    if z >= raw_good_z and (not math.isfinite(avoid) or avoid >= raw_min_avoidance):
        return "healthy", ["raw_fast_path_ok"]

    if reasons:
        return "stressed", reasons

    return "watch", ["raw_ok_but_below_good_threshold"]


def disruption_level(raw: Dict[str, Any]) -> str:
    """
    Lightweight disruption classifier from raw geometry.

    Uses raw cost/trace/cv/avoidance rather than external labels so it can be
    used in future projector paths.
    """
    z = safe_float(raw.get("geo_gap_z"))
    avoid = safe_float(raw.get("stress_avoidance"))
    cost = safe_float(raw.get("geodesic_cost"))
    trace_mean = safe_float(raw.get("trace_mean"))
    trace_cv = safe_float(raw.get("trace_cv"))

    score = 0.0
    if math.isfinite(trace_mean):
        score += trace_mean
    if math.isfinite(trace_cv):
        score += 0.25 * trace_cv
    if math.isfinite(cost):
        score += 0.05 * cost
    if math.isfinite(avoid) and avoid < 0:
        score += 0.25
    if math.isfinite(z) and z < 0.5:
        score += 0.25

    if score < 0.90:
        return "low"
    if score < 1.25:
        return "moderate"
    return "high"


def adaptive_score(
    row: Dict[str, Any],
    *,
    penalties: Dict[str, float],
    avoidance_weight: float,
    negative_avoidance_penalty: float,
    raw_bias_bonus: float,
) -> float:
    z = safe_float(row.get("geo_gap_z"), -1e9)
    avoid = safe_float(row.get("stress_avoidance"), 0.0)
    transform = str(row.get("transform", ""))

    score = z
    score += avoidance_weight * max(avoid, 0.0)
    score -= negative_avoidance_penalty * max(-avoid, 0.0)
    score -= op_penalty(transform, penalties)

    if transform == "raw":
        score += raw_bias_bonus

    return float(score)


def choose_raw_first(
    group_rows: List[Dict[str, Any]],
    *,
    raw_good_z: float,
    weak_z: float,
    raw_min_avoidance: float,
    min_jump_gain: float,
    penalties: Dict[str, float],
    avoidance_weight: float,
    negative_avoidance_penalty: float,
    raw_bias_bonus: float,
    force_raw_if_healthy: bool,
) -> Dict[str, Any]:
    """
    Select a transform for one mode/site/path group.

    Raw is selected immediately if healthy and force_raw_if_healthy is true.
    Otherwise transforms compete with operation penalties and a raw bias.
    """
    by_transform = {str(r.get("transform")): r for r in group_rows if r.get("status") == "ok"}

    raw = by_transform.get("raw")
    if raw is None:
        return {
            "status": "missing_raw",
            "selected_transform": None,
            "selection_reason": "raw row not found",
        }

    health, health_reasons = raw_health_class(
        raw,
        raw_good_z=raw_good_z,
        raw_min_avoidance=raw_min_avoidance,
        weak_z=weak_z,
    )

    path_name = str(raw.get("path_name", ""))
    candidates = [t for t in candidate_transforms_for_path(path_name) if t in by_transform]
    if "raw" not in candidates:
        candidates.insert(0, "raw")

    raw_z = safe_float(raw.get("geo_gap_z"))
    raw_score = adaptive_score(
        raw,
        penalties=penalties,
        avoidance_weight=avoidance_weight,
        negative_avoidance_penalty=negative_avoidance_penalty,
        raw_bias_bonus=raw_bias_bonus,
    )

    if force_raw_if_healthy and health == "healthy":
        selected = raw
        selected_score = raw_score
        reason = "raw_fast_path_healthy"
        jump_used = False
    else:
        scored = []
        for t in candidates:
            r = by_transform.get(t)
            if r is None:
                continue
            s = adaptive_score(
                r,
                penalties=penalties,
                avoidance_weight=avoidance_weight,
                negative_avoidance_penalty=negative_avoidance_penalty,
                raw_bias_bonus=raw_bias_bonus,
            )
            scored.append((s, t, r))

        if not scored:
            selected = raw
            selected_score = raw_score
            reason = "no_valid_candidates_fallback_raw"
            jump_used = False
        else:
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_t, best_r = scored[0]

            best_gain_over_raw = safe_float(best_r.get("geo_gap_z")) - raw_z
            best_score_gain_over_raw = best_score - raw_score

            if best_t != "raw" and best_gain_over_raw >= min_jump_gain and best_score_gain_over_raw > 0:
                selected = best_r
                selected_score = best_score
                reason = f"jump_gain_met:{best_t}"
                jump_used = True
            else:
                selected = raw
                selected_score = raw_score
                reason = "raw_kept_after_penalty"
                jump_used = False

    # Best static among candidate rows by unpenalized gap_z, used for regret.
    finite_static = [
        r for r in group_rows
        if r.get("status") == "ok" and math.isfinite(safe_float(r.get("geo_gap_z")))
    ]
    best_static = max(finite_static, key=lambda r: safe_float(r.get("geo_gap_z"))) if finite_static else raw

    selected_z = safe_float(selected.get("geo_gap_z"))
    best_static_z = safe_float(best_static.get("geo_gap_z"))
    selected_transform = str(selected.get("transform"))

    return {
        "status": "ok",
        "mode": raw.get("mode"),
        "delay_site": raw.get("delay_site"),
        "path_name": raw.get("path_name"),
        "disruption_level": disruption_level(raw),
        "raw_health": health,
        "raw_health_reasons": health_reasons,
        "candidate_transforms": candidates,
        "selected_transform": selected_transform,
        "selected_score": selected_score,
        "selected_gap_z": selected_z,
        "selected_geodesic_cost": safe_float(selected.get("geodesic_cost")),
        "selected_stress_avoidance": safe_float(selected.get("stress_avoidance")),
        "selection_reason": reason,
        "jump_used": bool(jump_used),
        "raw_gap_z": raw_z,
        "raw_score": raw_score,
        "raw_geodesic_cost": safe_float(raw.get("geodesic_cost")),
        "raw_stress_avoidance": safe_float(raw.get("stress_avoidance")),
        "raw_trace_cv": safe_float(raw.get("trace_cv")),
        "raw_trace_mean": safe_float(raw.get("trace_mean")),
        "gain_over_raw": selected_z - raw_z if math.isfinite(selected_z) and math.isfinite(raw_z) else float("nan"),
        "score_gain_over_raw": selected_score - raw_score if math.isfinite(selected_score) and math.isfinite(raw_score) else float("nan"),
        "best_static_transform": best_static.get("transform"),
        "best_static_gap_z": best_static_z,
        "adaptive_regret_vs_best_static": best_static_z - selected_z if math.isfinite(best_static_z) and math.isfinite(selected_z) else float("nan"),
        "op_penalty_paid": op_penalty(selected_transform, penalties),
        "extra_op_paid": 0.0 if selected_transform == "raw" else op_penalty(selected_transform, penalties),
    }


def group_rows_for_policy(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (str(r.get("mode")), str(r.get("delay_site")), str(r.get("path_name")))
        groups.setdefault(key, []).append(r)
    return groups


def run_raw_first_policy(
    rows: List[Dict[str, Any]],
    *,
    raw_good_z: float,
    weak_z: float,
    raw_min_avoidance: float,
    min_jump_gain: float,
    penalties: Dict[str, float],
    avoidance_weight: float,
    negative_avoidance_penalty: float,
    raw_bias_bonus: float,
    force_raw_if_healthy: bool,
) -> List[Dict[str, Any]]:
    candidate_rows = filter_candidate_rows(rows, DEFAULT_NEEDED_TRANSFORMS)
    groups = group_rows_for_policy(candidate_rows)

    out = []
    for key in sorted(groups.keys()):
        out.append(
            choose_raw_first(
                groups[key],
                raw_good_z=raw_good_z,
                weak_z=weak_z,
                raw_min_avoidance=raw_min_avoidance,
                min_jump_gain=min_jump_gain,
                penalties=penalties,
                avoidance_weight=avoidance_weight,
                negative_avoidance_penalty=negative_avoidance_penalty,
                raw_bias_bonus=raw_bias_bonus,
                force_raw_if_healthy=force_raw_if_healthy,
            )
        )

    return out


# --------------------------------------------------------------------------------------------------
# SUMMARY / CSV / PLOTS
# --------------------------------------------------------------------------------------------------

def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    flat_rows = []
    for r in rows:
        rr = dict(r)
        for key, value in list(rr.items()):
            if isinstance(value, (dict, list, tuple)):
                rr[key] = json.dumps(json_safe(value), sort_keys=True)
        flat_rows.append(rr)

    fields = list(flat_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)


def summarize_policy(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {"n": 0}

    n = len(ok)
    raw_count = sum(1 for r in ok if r.get("selected_transform") == "raw")
    jump_count = sum(1 for r in ok if r.get("jump_used"))

    by_transform: Dict[str, int] = {}
    by_path: Dict[str, Dict[str, Any]] = {}
    by_disruption: Dict[str, Dict[str, Any]] = {}

    for r in ok:
        t = str(r.get("selected_transform"))
        by_transform[t] = by_transform.get(t, 0) + 1

        path = str(r.get("path_name"))
        by_path.setdefault(path, {"n": 0, "raw": 0, "jump": 0, "mean_gain_over_raw": []})
        by_path[path]["n"] += 1
        by_path[path]["raw"] += int(t == "raw")
        by_path[path]["jump"] += int(bool(r.get("jump_used")))
        if math.isfinite(safe_float(r.get("gain_over_raw"))):
            by_path[path]["mean_gain_over_raw"].append(safe_float(r.get("gain_over_raw")))

        d = str(r.get("disruption_level"))
        by_disruption.setdefault(d, {"n": 0, "raw": 0, "jump": 0, "mean_gain_over_raw": []})
        by_disruption[d]["n"] += 1
        by_disruption[d]["raw"] += int(t == "raw")
        by_disruption[d]["jump"] += int(bool(r.get("jump_used")))
        if math.isfinite(safe_float(r.get("gain_over_raw"))):
            by_disruption[d]["mean_gain_over_raw"].append(safe_float(r.get("gain_over_raw")))

    for bucket in (by_path, by_disruption):
        for k, v in bucket.items():
            vals = v["mean_gain_over_raw"]
            v["mean_gain_over_raw"] = float(np.mean(vals)) if vals else float("nan")
            v["raw_fraction"] = float(v["raw"] / max(v["n"], 1))
            v["jump_fraction"] = float(v["jump"] / max(v["n"], 1))

    gains = [safe_float(r.get("gain_over_raw")) for r in ok if math.isfinite(safe_float(r.get("gain_over_raw")))]
    regrets = [safe_float(r.get("adaptive_regret_vs_best_static")) for r in ok if math.isfinite(safe_float(r.get("adaptive_regret_vs_best_static")))]
    extra_ops = [safe_float(r.get("extra_op_paid")) for r in ok if math.isfinite(safe_float(r.get("extra_op_paid")))]

    return {
        "n": n,
        "raw_count": raw_count,
        "jump_count": jump_count,
        "raw_fraction": float(raw_count / n),
        "jump_fraction": float(jump_count / n),
        "selected_by_transform": by_transform,
        "by_path": by_path,
        "by_disruption": by_disruption,
        "mean_gain_over_raw": float(np.mean(gains)) if gains else float("nan"),
        "mean_regret_vs_best_static": float(np.mean(regrets)) if regrets else float("nan"),
        "mean_extra_op_paid": float(np.mean(extra_ops)) if extra_ops else float("nan"),
    }


def print_policy_summary(rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 140)
    print("  T_S PROBE 03 — RAW-FIRST JUMPGEO SUMMARY")
    print("=" * 140)
    print(
        f"  {'mode':>13} | {'site':>14} | {'path':>11} | {'disrupt':>8} | {'raw health':>10} | "
        f"{'selected':>11} | {'jump':>5} | {'raw z':>8} | {'sel z':>8} | {'gain':>8} | {'regret':>8} | reason"
    )
    print("  " + "-" * 138)

    for r in rows:
        if r.get("status") != "ok":
            continue
        print(
            f"  {str(r.get('mode')):>13} | {str(r.get('delay_site')):>14} | {str(r.get('path_name')):>11} | "
            f"{str(r.get('disruption_level')):>8} | {str(r.get('raw_health')):>10} | "
            f"{str(r.get('selected_transform')):>11} | {str(bool(r.get('jump_used'))):>5} | "
            f"{safe_float(r.get('raw_gap_z')):>8.4f} | {safe_float(r.get('selected_gap_z')):>8.4f} | "
            f"{safe_float(r.get('gain_over_raw')):>8.4f} | {safe_float(r.get('adaptive_regret_vs_best_static')):>8.4f} | "
            f"{str(r.get('selection_reason'))}"
        )

    print("\n" + "=" * 140)
    print("  FAST-PATH ACCOUNTING")
    print("=" * 140)
    print(f"  total routes          : {summary.get('n', 0)}")
    print(f"  raw selected          : {summary.get('raw_count', 0)} ({summary.get('raw_fraction', float('nan')):.2%})")
    print(f"  transform jumps       : {summary.get('jump_count', 0)} ({summary.get('jump_fraction', float('nan')):.2%})")
    print(f"  mean gain over raw    : {summary.get('mean_gain_over_raw', float('nan')):.5f}")
    print(f"  mean regret vs static : {summary.get('mean_regret_vs_best_static', float('nan')):.5f}")
    print(f"  mean extra op paid    : {summary.get('mean_extra_op_paid', float('nan')):.5f}")
    print(f"  selected transforms   : {summary.get('selected_by_transform', {})}")


def plot_raw_usage(summary: Dict[str, Any], out_dir: Path) -> None:
    if not _HAVE_MPL or not summary or summary.get("n", 0) == 0:
        return

    by_path = summary.get("by_path", {})
    labels = list(by_path.keys())
    raw_fracs = [by_path[k]["raw_fraction"] for k in labels]
    jump_fracs = [by_path[k]["jump_fraction"] for k in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=160)
    ax.plot(x, raw_fracs, marker="o", label="raw selected")
    ax.plot(x, jump_fracs, marker="o", label="jump used")
    ax.set_title("T_S Probe 03 — Raw-first usage by path")
    ax.set_ylabel("fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "raw_usage_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_jump_gain(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAVE_MPL:
        return

    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return

    labels = [f"{r['mode']}\n{r['delay_site']}\n{r['path_name']}" for r in ok]
    gains = [safe_float(r.get("gain_over_raw")) for r in ok]
    regrets = [safe_float(r.get("adaptive_regret_vs_best_static")) for r in ok]
    x = np.arange(len(ok))

    fig, ax = plt.subplots(figsize=(max(12, len(ok) * 0.45), 6.0), dpi=160)
    ax.plot(x, gains, marker="o", label="gain over raw")
    ax.plot(x, regrets, marker="o", label="regret vs best static")
    ax.axhline(0.0, linewidth=1)
    ax.set_title("T_S Probe 03 — Adaptive gain/regret")
    ax.set_ylabel("gap z-score delta")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "jump_gain_summary.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

def parse_penalties(arg: Optional[str]) -> Dict[str, float]:
    penalties = {
        "raw": 0.00,
        "trace_norm": 0.05,
        "fro_norm": 0.07,
        "pow_exp_2": 0.08,
        "pow_exp_4": 0.12,
        "default": 0.25,
    }
    if not arg:
        return penalties

    # Format: raw=0,trace_norm=0.05,pow_exp_2=0.08
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad penalty entry: {part}")
        k, v = part.split("=", 1)
        penalties[k.strip()] = float(v.strip())
    return penalties


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T_S Probe 03 — raw-first adaptive JumpGeo routing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--geo-report", default=None, help="Probe 02 geo_reconstruction_report.json. Defaults to latest matching report.")
    p.add_argument("--npz", default=None, help="Fallback T_S dump if Probe 02 report is unavailable.")
    p.add_argument("--out-dir", default=None, help="Output directory. Defaults to probes/analysis/ts_jumpgeo_<JOB_ID>.")
    p.add_argument("--seed", type=int, default=20260601)

    p.add_argument("--raw-good-z", type=float, default=0.80,
                   help="If raw gap z is at least this and avoidance is healthy, raw stays selected.")
    p.add_argument("--weak-z", type=float, default=0.50,
                   help="Raw below this is considered weak/stressed.")
    p.add_argument("--raw-min-avoidance", type=float, default=0.0,
                   help="Minimum healthy raw stress avoidance.")
    p.add_argument("--min-jump-gain", type=float, default=0.08,
                   help="Minimum unpenalized gap-z improvement required to jump off raw.")
    p.add_argument("--penalties", default=None,
                   help="Optional comma-separated penalties, e.g. trace_norm=0.05,pow_exp_2=0.08")
    p.add_argument("--avoidance-weight", type=float, default=0.10)
    p.add_argument("--negative-avoidance-penalty", type=float, default=0.25)
    p.add_argument("--raw-bias-bonus", type=float, default=0.05)
    p.add_argument("--allow-healthy-jump", action="store_true",
                   help="Allow transforms to compete even when raw is healthy.")

    # Fallback recompute settings.
    p.add_argument("--coupling-weight", type=float, default=0.50)
    p.add_argument("--min-cost", type=float, default=1e-9)
    p.add_argument("--max-cost", type=float, default=1e9)
    p.add_argument("--clip-abs", type=float, default=1e9)
    p.add_argument("--max-visits", type=int, default=2_000_000)

    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    penalties = parse_penalties(args.penalties)

    npz_path, npz_job_id = resolve_npz(args.npz)

    source_meta: Dict[str, Any] = {}
    if args.geo_report:
        geo_report_path = Path(args.geo_report)
    else:
        geo_report_path = find_latest_geo_report(npz_job_id)

    if geo_report_path and geo_report_path.exists():
        report = load_json(geo_report_path)
        rows = rows_from_geo_report(report)
        job_id = get_job_id_from_report(report, geo_report_path)
        source_meta = {
            "source": "probe2_geo_report",
            "geo_report": str(geo_report_path),
            "npz": report.get("npz"),
            "job_id": job_id,
        }
    else:
        if npz_path is None:
            raise FileNotFoundError(
                "No Probe 02 geo report found and no T_S npz available. "
                "Run Probe 02 first or pass --npz."
            )
        rows, source_meta = recompute_needed_geo_rows(
            npz_path,
            seed=int(args.seed),
            coupling_weight=float(args.coupling_weight),
            min_cost=float(args.min_cost),
            max_cost=float(args.max_cost),
            clip_abs=float(args.clip_abs),
            max_visits=int(args.max_visits),
        )
        job_id = str(source_meta.get("job_id", npz_job_id or npz_path.stem))

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"ts_jumpgeo_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 140}")
    print("  T_S PROBE 03 — RAW-FIRST JUMPGEO ROUTING")
    print(f"{'=' * 140}")
    print(f"  Source     : {source_meta}")
    print(f"  Out dir    : {out_dir}")
    print(f"  Raw policy : raw_good_z={args.raw_good_z}, weak_z={args.weak_z}, min_jump_gain={args.min_jump_gain}")
    print(f"  Penalties  : {penalties}")

    selected_rows = run_raw_first_policy(
        rows,
        raw_good_z=float(args.raw_good_z),
        weak_z=float(args.weak_z),
        raw_min_avoidance=float(args.raw_min_avoidance),
        min_jump_gain=float(args.min_jump_gain),
        penalties=penalties,
        avoidance_weight=float(args.avoidance_weight),
        negative_avoidance_penalty=float(args.negative_avoidance_penalty),
        raw_bias_bonus=float(args.raw_bias_bonus),
        force_raw_if_healthy=not bool(args.allow_healthy_jump),
    )

    summary = summarize_policy(selected_rows)

    full_report = {
        "schema": "ts_probe3_raw_first_jumpgeo",
        "description": (
            "Raw-first adaptive JumpGeo policy. Raw geodesic geometry is selected by default; "
            "trace/fro normalization and low-order power expansion are only used when raw route "
            "health and penalized gain justify the extra operation."
        ),
        "job_id": job_id,
        "source": source_meta,
        "settings": {
            "raw_good_z": float(args.raw_good_z),
            "weak_z": float(args.weak_z),
            "raw_min_avoidance": float(args.raw_min_avoidance),
            "min_jump_gain": float(args.min_jump_gain),
            "penalties": penalties,
            "avoidance_weight": float(args.avoidance_weight),
            "negative_avoidance_penalty": float(args.negative_avoidance_penalty),
            "raw_bias_bonus": float(args.raw_bias_bonus),
            "force_raw_if_healthy": not bool(args.allow_healthy_jump),
        },
        "summary": summary,
        "rows": selected_rows,
    }

    with open(out_dir / "raw_first_jumpgeo_report.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(full_report), f, indent=2)

    write_csv(selected_rows, out_dir / "raw_first_jumpgeo_rows.csv")
    write_csv([summary], out_dir / "raw_first_jumpgeo_summary.csv")

    if not args.no_plots:
        plot_raw_usage(summary, out_dir)
        plot_jump_gain(selected_rows, out_dir)

    print_policy_summary(selected_rows, summary)

    print(f"\n[SAVED] {out_dir}")
    print(f"{'=' * 140}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"[FATAL] {type(e).__name__}: {e}")
