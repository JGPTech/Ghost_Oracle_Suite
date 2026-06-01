#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
T_S PROBE 02 — GEO RECONSTRUCTION SWEEP
====================================================================================================
Geometric reconstruction probe for T_S — Temporal Stress Metric.

This probe assumes the QPU generator is frozen.  It does not modify the QPU
channel.  It asks a downstream question:

    Given the QPU-generated delay/round/edge field from T_S Probe 01,
    does the resulting stress tensor define a coherent geometry?

The input object is the same dumped T_S data:

    field[mode, delay_site, delay_value, shot, round, edge]

Probe 02 computes local delay-aware stress:

    DτF = F[τ+1,r,x] XOR F[τ,r,x]
    DrF = F[τ,r+1,x] XOR F[τ,r,x]
    DxF = F[τ,r,x+1] XOR F[τ,r,x]

    T_ab = <D_aF D_bF>,    a,b ∈ {τ,r,x}

Then it treats every local cell:

    node = (delay_cell, round_cell, edge_cell)

as a point in a discrete geometry.  Neighbor moves along delay/round/edge are
assigned costs from the local stress components.  Shortest paths become
geodesic-like paths through the preserved channel.

Important framing
-----------------
This is not claiming physical GR stress-energy.  This is a benchmark geometry:

    Does the Temporal Stress Metric define navigable structure that survives
    perturbation and separates from controls?

Normalization sweep
-------------------
The probe intentionally sweeps multiple geometry transforms:

    raw                       no normalization; trust QPU-bounded field
    trace_norm                divide by local/global trace scale
    fro_norm                  divide by global Frobenius-like stress scale
    maxabs_norm               divide by maximum absolute component
    minmax_norm               normalize scalar costs into [0,1]
    log1p                     compress large costs with log1p
    sqrt                      mild compression
    pow_exp_2 ... pow_exp_256 power-of-two exponent expansion
    scale_pow2_2 ... 256      multiply by power-of-two scale

Why both pow_exp and scale_pow2?
--------------------------------
Earlier G_M-style blow-ups used power-of-two amplification logic.  Depending on
the exact intended operation, this can mean either exponentiating by a power of
two or scaling by a power of two.  This probe includes both families and records
which transforms survive without crashing or degenerating.

Crash guards
------------
Every transform and graph calculation is guarded:

    - NaN/Inf cleanup
    - clipping to configurable cap
    - negative-cost prevention
    - degenerate-grid handling
    - Dijkstra iteration cap
    - failed transform records instead of hard crashes

Default paths
-------------
This probe is intended to live in:

    ghost_oracle/T_S/probes/t_s_probe2_geo.py

with shared data in:

    ghost_oracle/T_S/data/

Therefore:

    HERE = Path(__file__).resolve().parent
    DATA_DIR = HERE.parent / "data"
    ANALYSIS_DIR = HERE / "analysis"

Usage
-----
Latest T_S dump:

    python ghost_oracle/T_S/probes/t_s_probe2_geo.py

Specific dump:

    python ghost_oracle/T_S/probes/t_s_probe2_geo.py --npz ghost_oracle/T_S/data/ts_data_<JOB_ID>.npz

Fast/smaller transform sweep:

    python ghost_oracle/T_S/probes/t_s_probe2_geo.py --quick

Skip plots:

    python ghost_oracle/T_S/probes/t_s_probe2_geo.py --no-plots
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
# PATHS — probe convention requested by repo owner
# --------------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
ANALYSIS_DIR = HERE / "analysis"


# --------------------------------------------------------------------------------------------------
# GENERAL HELPERS
# --------------------------------------------------------------------------------------------------

EPS = 1e-12


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


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
        return default
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


def l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = finite_clean(np.asarray(a, dtype=np.float64))
    bb = finite_clean(np.asarray(b, dtype=np.float64))
    return float(np.linalg.norm(aa - bb))


def resolve_inputs(npz_arg: Optional[str], meta_arg: Optional[str]) -> tuple[Path, Optional[Path], str]:
    if npz_arg is None:
        latest = DATA_DIR / "latest_ts_data.json"
        if not latest.exists():
            raise FileNotFoundError(
                "No --npz provided and latest T_S data file was not found at "
                f"{latest}"
            )
        obj = load_json(latest)
        npz = Path(obj["npz"])
        meta = Path(obj["meta"]) if obj.get("meta") else None
        job_id = str(obj.get("job_id", npz.stem))
        return npz, meta, job_id

    npz = Path(npz_arg)
    meta = Path(meta_arg) if meta_arg else None
    stem = npz.stem
    job_id = stem.split("_")[-1] if "_" in stem else stem
    return npz, meta, job_id


def load_ts(npz_path: Path) -> Dict[str, Any]:
    z = np.load(npz_path, allow_pickle=False)
    if "field" not in z.files:
        raise KeyError("T_S Probe 02 expects a `field` array in the dumped npz.")

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


# --------------------------------------------------------------------------------------------------
# CONTROLS
# --------------------------------------------------------------------------------------------------

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


def mutate_field(field: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """
    Mutate one field block with shape:

        delay, shot, round, edge

    These controls mirror the T_S Probe 01 analysis controls.
    """
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
        # Preserve per-delay/per-round/per-edge marginal rates across shots.
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


# --------------------------------------------------------------------------------------------------
# LOCAL TEMPORAL STRESS FIELD
# --------------------------------------------------------------------------------------------------

@dataclass
class StressLocal:
    """
    Local stress field on a common cell lattice:

        delay_cell × round_cell × edge_cell

    Components:
        tau_tau : delay sensitivity
        rr      : round/layer instability
        xx      : edge/channel roughness
        tau_r   : delay-round coupling
        tau_x   : delay-edge coupling
        r_x     : round-edge coupling
        trace   : tau_tau + rr + xx
    """
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


def gradients_delay_round_edge(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute aligned binary gradients for one field block:

        f shape = delay, shot, round, edge

    Returned arrays have common shape:

        delay-1, shot, round-1, edge-1
    """
    if f.shape[0] < 2 or f.shape[2] < 2 or f.shape[3] < 2:
        empty = np.zeros((0, f.shape[1], 0, 0), dtype=np.float64)
        return empty, empty, empty

    d_tau = np.bitwise_xor(f[1:, :, :, :], f[:-1, :, :, :]).astype(np.float64)
    d_round = np.bitwise_xor(f[:, :, 1:, :], f[:, :, :-1, :]).astype(np.float64)
    d_edge = np.bitwise_xor(f[:, :, :, 1:], f[:, :, :, :-1]).astype(np.float64)

    # Align to common delay/round/edge cell lattice.
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


# --------------------------------------------------------------------------------------------------
# GEOMETRY TRANSFORMS
# --------------------------------------------------------------------------------------------------

@dataclass
class TransformSpec:
    name: str
    kind: str
    value: Optional[float] = None


def build_transform_sweep(quick: bool = False) -> List[TransformSpec]:
    base = [
        TransformSpec("raw", "raw"),
        TransformSpec("trace_norm", "trace_norm"),
        TransformSpec("fro_norm", "fro_norm"),
        TransformSpec("maxabs_norm", "maxabs_norm"),
        TransformSpec("minmax_norm", "minmax_norm"),
        TransformSpec("log1p", "log1p"),
        TransformSpec("sqrt", "sqrt"),
    ]

    if quick:
        powers = [2, 4, 16, 64, 256]
    else:
        powers = [2, 4, 8, 16, 32, 64, 128, 256]

    for p in powers:
        base.append(TransformSpec(f"pow_exp_{p}", "pow_exp", float(p)))

    for p in powers:
        base.append(TransformSpec(f"scale_pow2_{p}", "scale_pow2", float(p)))

    return base


def stress_scale(stress: StressLocal, mode: str) -> float:
    comps = stress.component_dict()
    vals = np.concatenate([np.ravel(v) for v in comps.values() if v.size])
    vals = finite_clean(vals)
    if vals.size == 0:
        return 1.0

    if mode == "trace_norm":
        s = float(np.nanmean(stress.trace))
    elif mode == "fro_norm":
        s = float(np.sqrt(np.nanmean(vals * vals)))
    elif mode == "maxabs_norm":
        s = float(np.nanmax(np.abs(vals)))
    else:
        s = 1.0

    if not math.isfinite(s) or abs(s) < EPS:
        return 1.0
    return s


def apply_transform_to_components(
    stress: StressLocal,
    spec: TransformSpec,
    *,
    clip_abs: float,
    min_cost: float,
) -> tuple[Optional[StressLocal], Optional[str]]:
    """
    Apply a geometry transform to local stress components.

    Returns (transformed_stress, error_message).  Errors are recorded rather
    than thrown so the sweep can continue.
    """
    try:
        comps = stress.component_dict()
        out: Dict[str, np.ndarray] = {}

        if stress.trace.size == 0:
            z = np.zeros((0, 0, 0), dtype=np.float64)
            return StressLocal(z, z, z, z, z, z, z), None

        if spec.kind in ("trace_norm", "fro_norm", "maxabs_norm"):
            s = stress_scale(stress, spec.kind)
            for k, v in comps.items():
                out[k] = finite_clean(v / s, clip_abs=clip_abs)

        elif spec.kind == "raw":
            for k, v in comps.items():
                out[k] = finite_clean(v, clip_abs=clip_abs)

        elif spec.kind == "minmax_norm":
            # Apply min/max to the trace-derived scalar field first, then use
            # the same affine transform for components to preserve relative
            # directionality as much as possible.
            vals = finite_clean(stress.trace)
            lo = float(np.nanmin(vals))
            hi = float(np.nanmax(vals))
            denom = max(hi - lo, EPS)
            for k, v in comps.items():
                out[k] = finite_clean((v - lo) / denom, clip_abs=clip_abs)

        elif spec.kind == "log1p":
            for k, v in comps.items():
                out[k] = finite_clean(np.log1p(np.maximum(v, 0.0)), clip_abs=clip_abs)

        elif spec.kind == "sqrt":
            for k, v in comps.items():
                out[k] = finite_clean(np.sqrt(np.maximum(v, 0.0)), clip_abs=clip_abs)

        elif spec.kind == "pow_exp":
            # True exponentiation by powers of two.  Because QPU-derived
            # components are usually bounded in [0,1], this can underflow into
            # a degenerate geometry.  That is not a crash; it is a result.
            p = float(spec.value or 1.0)
            for k, v in comps.items():
                vv = finite_clean(np.maximum(v, 0.0), clip_abs=clip_abs)
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    out[k] = finite_clean(np.power(vv, p), clip_abs=clip_abs)

        elif spec.kind == "scale_pow2":
            # Multiplicative power-of-two amplification.
            p = float(spec.value or 1.0)
            for k, v in comps.items():
                with np.errstate(over="ignore", invalid="ignore"):
                    out[k] = finite_clean(v * p, clip_abs=clip_abs)

        else:
            return None, f"unknown transform kind: {spec.kind}"

        # Enforce nonnegative diagonal/cost fields.  Mixed components may be
        # nonnegative in the binary product construction, but keep the guard.
        tau_tau = np.maximum(out["tau_tau"], min_cost)
        rr = np.maximum(out["rr"], min_cost)
        xx = np.maximum(out["xx"], min_cost)
        tau_r = finite_clean(np.maximum(out["tau_r"], 0.0), clip_abs=clip_abs)
        tau_x = finite_clean(np.maximum(out["tau_x"], 0.0), clip_abs=clip_abs)
        r_x = finite_clean(np.maximum(out["r_x"], 0.0), clip_abs=clip_abs)
        trace = finite_clean(tau_tau + rr + xx, clip_abs=clip_abs)

        # Degeneracy guard: if everything collapses to min_cost, we still allow
        # the geometry but mark it later via low dynamic range.
        return StressLocal(tau_tau, rr, xx, tau_r, tau_x, r_x, trace), None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------------------------------
# DISCRETE GEODESIC GRAPH
# --------------------------------------------------------------------------------------------------

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


def neighbors_3d(node: Tuple[int, int, int], shape: Tuple[int, int, int]) -> Iterable[Tuple[Tuple[int, int, int], int]]:
    """
    Yield 6-neighbors and axis index:
        0 = delay/tau move
        1 = round move
        2 = edge/x move
    """
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
    """
    Cost for moving between adjacent cells.

    Diagonal component provides the axis cost.  Mixed components add coupling
    pressure.  We average endpoints for symmetry.
    """
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


def path_entropy_axes(axes: List[int]) -> tuple[float, float, float, float]:
    if not axes:
        return 0.0, 0.0, 0.0, 0.0
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
        return GeoResult(
            status="degenerate",
            geodesic_cost=float("nan"),
            euclidean_steps=0,
            path_length=0,
            path_entropy=float("nan"),
            path_concentration=float("nan"),
            mean_trace_on_path=float("nan"),
            mean_trace_global=float("nan"),
            stress_avoidance=float("nan"),
            axis_tau_fraction=float("nan"),
            axis_round_fraction=float("nan"),
            axis_edge_fraction=float("nan"),
            visited_nodes=0,
            message=f"degenerate stress grid shape={shape}",
        )

    for idx, limit in zip(source + target, shape + shape):
        if idx < 0 or idx >= limit:
            return GeoResult(
                status="bad_endpoint",
                geodesic_cost=float("nan"),
                euclidean_steps=0,
                path_length=0,
                path_entropy=float("nan"),
                path_concentration=float("nan"),
                mean_trace_on_path=float("nan"),
                mean_trace_global=float("nan"),
                stress_avoidance=float("nan"),
                axis_tau_fraction=float("nan"),
                axis_round_fraction=float("nan"),
                axis_edge_fraction=float("nan"),
                visited_nodes=0,
                message=f"source={source}, target={target}, shape={shape}",
            )

    dist: Dict[Tuple[int, int, int], float] = {source: 0.0}
    prev: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], int]] = {}
    heap: List[Tuple[float, Tuple[int, int, int]]] = [(0.0, source)]
    visited = set()

    while heap:
        cur_dist, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        if len(visited) > max_visits:
            return GeoResult(
                status="visit_cap",
                geodesic_cost=float("nan"),
                euclidean_steps=sum(abs(s - t) for s, t in zip(source, target)),
                path_length=0,
                path_entropy=float("nan"),
                path_concentration=float("nan"),
                mean_trace_on_path=float("nan"),
                mean_trace_global=float(np.mean(stress.trace)),
                stress_avoidance=float("nan"),
                axis_tau_fraction=float("nan"),
                axis_round_fraction=float("nan"),
                axis_edge_fraction=float("nan"),
                visited_nodes=len(visited),
                message=f"exceeded max_visits={max_visits}",
            )

        if u == target:
            path, axes = reconstruct_path(prev, target)
            trace_vals = np.asarray([stress.trace[p] for p in path], dtype=np.float64)
            trace_vals = finite_clean(trace_vals)
            mean_path = float(np.mean(trace_vals)) if trace_vals.size else float("nan")
            mean_global = float(np.mean(finite_clean(stress.trace))) if stress.trace.size else float("nan")
            entropy, concentration, p_tau, p_round, p_edge = path_entropy_axes(axes)
            return GeoResult(
                status="ok",
                geodesic_cost=float(cur_dist),
                euclidean_steps=sum(abs(s - t) for s, t in zip(source, target)),
                path_length=len(path),
                path_entropy=entropy,
                path_concentration=concentration,
                mean_trace_on_path=mean_path,
                mean_trace_global=mean_global,
                stress_avoidance=float(mean_global - mean_path),
                axis_tau_fraction=p_tau,
                axis_round_fraction=p_round,
                axis_edge_fraction=p_edge,
                visited_nodes=len(visited),
                message="",
            )

        if cur_dist > dist.get(u, float("inf")):
            continue

        for v, axis in neighbors_3d(u, shape):
            c = edge_cost(
                stress,
                u,
                v,
                axis,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
            )
            nd = cur_dist + c
            if not math.isfinite(nd):
                continue
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = (u, axis)
                heapq.heappush(heap, (nd, v))

    return GeoResult(
        status="unreachable",
        geodesic_cost=float("nan"),
        euclidean_steps=sum(abs(s - t) for s, t in zip(source, target)),
        path_length=0,
        path_entropy=float("nan"),
        path_concentration=float("nan"),
        mean_trace_on_path=float("nan"),
        mean_trace_global=float(np.mean(stress.trace)),
        stress_avoidance=float("nan"),
        axis_tau_fraction=float("nan"),
        axis_round_fraction=float("nan"),
        axis_edge_fraction=float("nan"),
        visited_nodes=len(visited),
        message="target not reached",
    )


def default_endpoints(shape: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]]:
    """
    Build a compact set of source/target geodesic questions.

    We include:
        full_diag    : early-delay/round/edge to late-delay/round/edge
        delay_round  : early delay+round through center edge
        delay_only   : early to late delay at center round/edge
        edge_only    : left to right channel at center delay/round
    """
    A, B, C = shape
    if min(shape) <= 0:
        return []

    ca = A // 2
    cb = B // 2
    cc = C // 2

    endpoints = [
        ((0, 0, 0), (A - 1, B - 1, C - 1), "full_diag"),
        ((0, 0, cc), (A - 1, B - 1, cc), "delay_round"),
        ((0, cb, cc), (A - 1, cb, cc), "delay_only"),
        ((ca, cb, 0), (ca, cb, C - 1), "edge_only"),
    ]

    # Remove duplicates for tiny grids.
    unique = []
    seen = set()
    for s, t, name in endpoints:
        key = (s, t, name)
        if key not in seen:
            seen.add(key)
            unique.append((s, t, name))
    return unique


# --------------------------------------------------------------------------------------------------
# GEO SWEEP
# --------------------------------------------------------------------------------------------------

def run_geo_sweep(
    ts: Dict[str, Any],
    *,
    transforms: List[TransformSpec],
    seed: int,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    clip_abs: float,
    max_visits: int,
    controls: List[str],
) -> Dict[str, Any]:
    field = ts["field"]
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for mi, mode_name in enumerate(ts["modes"]):
        for si, site_name in enumerate(ts["delay_sites"]):
            block = field[mi, si]  # delays, shots, rounds, edges
            rng_base = seed + 10007 * mi + 1009 * si

            # Precompute real/control local stress fields.
            stress_by_control: Dict[str, StressLocal] = {}
            for ci, control in enumerate(controls):
                rng = np.random.default_rng(rng_base + 7919 * ci)
                mutated = mutate_field(block, control, rng)
                stress_by_control[control] = local_stress_from_field(mutated)

            real_raw = stress_by_control["real"]
            shape = real_raw.trace.shape
            endpoints = default_endpoints(shape)

            if not endpoints:
                failures.append({
                    "mode": mode_name,
                    "delay_site": site_name,
                    "reason": f"degenerate real stress shape={shape}",
                })
                continue

            for spec in transforms:
                transformed: Dict[str, StressLocal] = {}
                transform_errors: Dict[str, str] = {}

                for control, stress in stress_by_control.items():
                    st, err = apply_transform_to_components(
                        stress,
                        spec,
                        clip_abs=clip_abs,
                        min_cost=min_cost,
                    )
                    if st is None:
                        transform_errors[control] = err or "unknown transform failure"
                    else:
                        transformed[control] = st

                if "real" not in transformed:
                    failures.append({
                        "mode": mode_name,
                        "delay_site": site_name,
                        "transform": spec.name,
                        "reason": transform_errors.get("real", "real transform failed"),
                    })
                    continue

                real_st = transformed["real"]

                # Transform diagnostics.
                trace = finite_clean(real_st.trace)
                trace_mean = float(np.mean(trace)) if trace.size else float("nan")
                trace_std = float(np.std(trace)) if trace.size else float("nan")
                trace_min = float(np.min(trace)) if trace.size else float("nan")
                trace_max = float(np.max(trace)) if trace.size else float("nan")
                trace_dynamic_range = float(trace_max - trace_min) if math.isfinite(trace_max - trace_min) else float("nan")
                trace_cv = float(trace_std / (abs(trace_mean) + EPS)) if math.isfinite(trace_std) else float("nan")

                for source, target, path_name in endpoints:
                    real_geo = dijkstra_geodesic(
                        real_st,
                        source=source,
                        target=target,
                        coupling_weight=coupling_weight,
                        min_cost=min_cost,
                        max_cost=max_cost,
                        max_visits=max_visits,
                    )

                    control_costs = {}
                    control_statuses = {}
                    control_avoidance = {}
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
                        control_avoidance[control] = g.stress_avoidance

                    finite_control_costs = [
                        c for c in control_costs.values()
                        if isinstance(c, (float, int)) and math.isfinite(float(c))
                    ]

                    if finite_control_costs and math.isfinite(real_geo.geodesic_cost):
                        control_mean = float(np.mean(finite_control_costs))
                        control_min = float(np.min(finite_control_costs))
                        control_max = float(np.max(finite_control_costs))
                        geo_gap_mean = float(control_mean - real_geo.geodesic_cost)
                        geo_gap_min = float(control_min - real_geo.geodesic_cost)
                        geo_gap_z = float(
                            geo_gap_mean / (float(np.std(finite_control_costs)) + EPS)
                        )
                    else:
                        control_mean = float("nan")
                        control_min = float("nan")
                        control_max = float("nan")
                        geo_gap_mean = float("nan")
                        geo_gap_min = float("nan")
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
                        "trace_min": trace_min,
                        "trace_max": trace_max,
                        "trace_dynamic_range": trace_dynamic_range,
                        "trace_cv": trace_cv,
                        "control_cost_mean": control_mean,
                        "control_cost_min": control_min,
                        "control_cost_max": control_max,
                        "geo_gap_mean_control_minus_real": geo_gap_mean,
                        "geo_gap_min_control_minus_real": geo_gap_min,
                        "geo_gap_z": geo_gap_z,
                        "control_costs": control_costs,
                        "control_statuses": control_statuses,
                        "control_avoidance": control_avoidance,
                        "message": real_geo.message,
                    })

    return {
        "rows": rows,
        "failures": failures,
        "controls": controls,
        "transforms": [spec.__dict__ for spec in transforms],
    }


# --------------------------------------------------------------------------------------------------
# REPORTING / PLOTS
# --------------------------------------------------------------------------------------------------

def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # Flatten dict-ish fields for CSV.
    flat_rows = []
    for r in rows:
        rr = dict(r)
        for key in ("source", "target", "control_costs", "control_statuses", "control_avoidance"):
            if key in rr:
                rr[key] = json.dumps(json_safe(rr[key]), sort_keys=True)
        flat_rows.append(rr)

    fields = list(flat_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)


def summarize_best(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    One compact summary per mode/site/path:
        - best real-vs-control separation by geo_gap_z
        - best raw/no-normalization result retained separately
    """
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["mode"], r["delay_site"], r["path_name"])
        groups.setdefault(key, []).append(r)

    out = []
    for key, vals in groups.items():
        vals_finite = [v for v in vals if math.isfinite(safe_float(v.get("geo_gap_z")))]
        if vals_finite:
            best = max(vals_finite, key=lambda x: x["geo_gap_z"])
        else:
            best = min(vals, key=lambda x: safe_float(x.get("geodesic_cost"), float("inf")))

        raw_vals = [v for v in vals if v.get("transform") == "raw"]
        raw = raw_vals[0] if raw_vals else None

        row = {
            "mode": key[0],
            "delay_site": key[1],
            "path_name": key[2],
            "best_transform": best.get("transform"),
            "best_geo_gap_z": best.get("geo_gap_z"),
            "best_geo_gap_mean": best.get("geo_gap_mean_control_minus_real"),
            "best_geodesic_cost": best.get("geodesic_cost"),
            "best_stress_avoidance": best.get("stress_avoidance"),
            "best_trace_cv": best.get("trace_cv"),
        }
        if raw is not None:
            row.update({
                "raw_geodesic_cost": raw.get("geodesic_cost"),
                "raw_geo_gap_z": raw.get("geo_gap_z"),
                "raw_geo_gap_mean": raw.get("geo_gap_mean_control_minus_real"),
                "raw_stress_avoidance": raw.get("stress_avoidance"),
                "raw_trace_cv": raw.get("trace_cv"),
            })
        out.append(row)

    return out


def plot_best_summary(summary: List[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAVE_MPL or not summary:
        return

    # Plot best gap Z per mode/site for full_diag only if available.
    rows = [r for r in summary if r["path_name"] == "full_diag"]
    if not rows:
        rows = summary

    labels = [f"{r['mode']}\n{r['delay_site']}" for r in rows]
    vals = [safe_float(r.get("best_geo_gap_z")) for r in rows]
    raw_vals = [safe_float(r.get("raw_geo_gap_z")) for r in rows]

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(11, len(rows) * 0.85), 6.5), dpi=160)
    ax.plot(x, vals, marker="o", label="best transform")
    ax.plot(x, raw_vals, marker="o", label="raw / no normalization")
    ax.axhline(0.0, linewidth=1)
    ax.set_title("T_S Probe 02 — Geo separation from controls")
    ax.set_ylabel("geo gap z-score  (control mean - real) / control std")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "geo_gap_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_transform_sweep(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAVE_MPL or not rows:
        return

    # One plot per path name, averaging over mode/site.
    path_names = sorted({r["path_name"] for r in rows})
    for path_name in path_names:
        sub = [r for r in rows if r["path_name"] == path_name and r.get("status") == "ok"]
        if not sub:
            continue

        transforms = []
        for r in sub:
            if r["transform"] not in transforms:
                transforms.append(r["transform"])

        y = []
        for t in transforms:
            vals = [safe_float(r.get("geo_gap_z")) for r in sub if r["transform"] == t]
            vals = [v for v in vals if math.isfinite(v)]
            y.append(float(np.mean(vals)) if vals else float("nan"))

        fig, ax = plt.subplots(figsize=(max(12, len(transforms) * 0.45), 6.0), dpi=160)
        ax.plot(np.arange(len(transforms)), y, marker="o")
        ax.axhline(0.0, linewidth=1)
        ax.set_title(f"T_S Probe 02 — transform sweep, path={path_name}")
        ax.set_ylabel("mean geo gap z-score")
        ax.set_xticks(np.arange(len(transforms)))
        ax.set_xticklabels(transforms, rotation=70, ha="right")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / f"transform_sweep_{path_name}.png", bbox_inches="tight")
        plt.close(fig)


def print_summary(summary: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 126)
    print("  T_S PROBE 02 — GEO RECONSTRUCTION SUMMARY")
    print("=" * 126)
    print(
        f"  {'mode':>13} | {'site':>14} | {'path':>11} | {'best xform':>16} | "
        f"{'best gap z':>10} | {'raw gap z':>10} | {'raw cost':>10} | {'avoid raw':>10}"
    )
    print("  " + "-" * 124)

    for r in summary:
        print(
            f"  {r['mode']:>13} | {r['delay_site']:>14} | {r['path_name']:>11} | "
            f"{str(r.get('best_transform')):>16} | "
            f"{safe_float(r.get('best_geo_gap_z')):>10.4f} | "
            f"{safe_float(r.get('raw_geo_gap_z')):>10.4f} | "
            f"{safe_float(r.get('raw_geodesic_cost')):>10.4f} | "
            f"{safe_float(r.get('raw_stress_avoidance')):>10.4f}"
        )

    if failures:
        print("\n" + "=" * 126)
        print("  FAILURES / GUARDED SKIPS")
        print("=" * 126)
        for f in failures[:20]:
            print("  - " + json.dumps(json_safe(f), sort_keys=True))
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more guarded skips")


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T_S Probe 02 — geo reconstruction sweep over normalization and power-of-two transforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz", default=None, help="Dumped T_S .npz. Defaults to latest dumped dataset.")
    p.add_argument("--meta", default=None, help="Optional metadata path.")
    p.add_argument("--out-dir", default=None, help="Output directory. Defaults to probes/analysis/ts_geo_<JOB_ID>.")
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--quick", action="store_true", help="Use smaller transform sweep.")
    p.add_argument("--controls", nargs="+", default=CONTROL_MODES,
                   choices=CONTROL_MODES, help="Controls to compare. Must include real.")
    p.add_argument("--coupling-weight", type=float, default=0.50,
                   help="How much mixed stress components contribute to movement cost.")
    p.add_argument("--min-cost", type=float, default=1e-9,
                   help="Minimum positive graph edge cost.")
    p.add_argument("--max-cost", type=float, default=1e9,
                   help="Maximum graph edge cost after guarding.")
    p.add_argument("--clip-abs", type=float, default=1e9,
                   help="Absolute clipping guard for transformed stress components.")
    p.add_argument("--max-visits", type=int, default=2_000_000,
                   help="Dijkstra visit cap.")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if "real" not in args.controls:
        raise ValueError("--controls must include `real`.")

    npz_path, meta_path, job_id = resolve_inputs(args.npz, args.meta)
    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"ts_geo_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 126}")
    print("  T_S PROBE 02 — GEO RECONSTRUCTION SWEEP")
    print(f"{'=' * 126}")
    print(f"  NPZ        : {npz_path}")
    print(f"  Metadata   : {meta_path if meta_path else '(not provided)'}")
    print(f"  Out dir    : {out_dir}")
    print(f"  Controls   : {args.controls}")
    print(f"  Quick sweep: {args.quick}")
    print(f"  Guard caps : min_cost={args.min_cost}, max_cost={args.max_cost}, clip_abs={args.clip_abs}")

    ts = load_ts(npz_path)
    transforms = build_transform_sweep(quick=args.quick)

    report = run_geo_sweep(
        ts,
        transforms=transforms,
        seed=int(args.seed),
        coupling_weight=float(args.coupling_weight),
        min_cost=float(args.min_cost),
        max_cost=float(args.max_cost),
        clip_abs=float(args.clip_abs),
        max_visits=int(args.max_visits),
        controls=list(args.controls),
    )

    rows = report["rows"]
    summary = summarize_best(rows)

    full_report = {
        "job_id": job_id,
        "npz": str(npz_path),
        "meta": str(meta_path) if meta_path else None,
        "schema": "ts_probe2_geo_reconstruction",
        "description": (
            "Geodesic reconstruction sweep over raw, normalized, exponentiated, "
            "and power-of-two scaled T_S local stress fields."
        ),
        "settings": {
            "seed": int(args.seed),
            "quick": bool(args.quick),
            "controls": list(args.controls),
            "coupling_weight": float(args.coupling_weight),
            "min_cost": float(args.min_cost),
            "max_cost": float(args.max_cost),
            "clip_abs": float(args.clip_abs),
            "max_visits": int(args.max_visits),
        },
        "source_shapes": {
            "field": list(ts["field"].shape),
            "modes": ts["modes"],
            "delay_sites": ts["delay_sites"],
            "delays": ts["delays"],
            "delay_unit": ts["delay_unit"],
        },
        "transforms": report["transforms"],
        "summary": summary,
        "rows": rows,
        "failures": report["failures"],
    }

    with open(out_dir / "geo_reconstruction_report.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(full_report), f, indent=2)

    write_csv(rows, out_dir / "geo_reconstruction_rows.csv")
    write_csv(summary, out_dir / "geo_reconstruction_summary.csv")

    if not args.no_plots:
        plot_best_summary(summary, out_dir)
        plot_transform_sweep(rows, out_dir)

    print_summary(summary, report["failures"])

    print(f"\n[SAVED] {out_dir}")
    print(f"{'=' * 126}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"[FATAL] {type(e).__name__}: {e}")
