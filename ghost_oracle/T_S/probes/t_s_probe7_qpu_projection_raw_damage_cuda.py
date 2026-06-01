#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
T_S PROBE 07 — OPTIMIZED QPU RAW-DAMAGE PROJECTION SIGNATURE
====================================================================================================
CUDA-optimized version of Probe 06.

This probe finalizes the optimized QPU-side projection signature path using the
enhanced single-file T_S geo kernel:

    ghost_oracle/T_S/kernels/ts_geo_kernel.cu

Required kernel entry point:
    ts_raw_geo_route_vector_kernel

Optional future entry points available in the enhanced kernel:
    ts_raw_geo_damage_kernel
    ts_raw_geo_packed_damage_kernel
    ts_raw_geo_damage_mean_by_k_kernel

Methodology
-----------
RAW ONLY.
NO normalization.
NO cosine similarity.
NO classifier framing.
NO GPU base generation.
NO G_M/S_M bases.
NO projector benchmark.

Probe 06 established the correct raw-damage signature:

    F[mode, delay_site, delay_value, shot, round, edge]
      -> raw stress
      -> raw geo routes
      -> edge / round / round-edge ablation damage
      -> qpu_projection_signature_raw_damage.npz

Probe 07 keeps the same signature and damage definition, but accelerates route
evaluation through the enhanced CUDA kernel.

Damage
------
For each real block and ablated block:

    damage =
        abs(full_cost_ablated  - full_cost_real)
      + abs(delay_cost_ablated - delay_cost_real)
      + abs(edge_cost_ablated  - edge_cost_real)
      + max(0, real_avoidance - ablated_avoidance)

Expected location
-----------------
    ghost_oracle/T_S/probes/t_s_probe7_qpu_projection_raw_damage_cuda.py

Path convention
---------------
    HERE = Path(__file__).resolve().parent
    DATA_DIR = HERE.parent / "data"
    ANALYSIS_DIR = HERE / "analysis"
    KERNEL_DIR = HERE.parent / "kernels"

Usage
-----
Latest T_S QPU dump:

    python ghost_oracle/T_S/probes/t_s_probe7_qpu_projection_raw_damage_cuda.py

Explicit files:

    python ghost_oracle/T_S/probes/t_s_probe7_qpu_projection_raw_damage_cuda.py ^
        --npz ghost_oracle/T_S/data/ts_data_<JOB_ID>.npz ^
        --meta ghost_oracle/T_S/data/ts_job_<JOB_ID>.json

CPU fallback / validation mode:

    python ghost_oracle/T_S/probes/t_s_probe7_qpu_projection_raw_damage_cuda.py --cpu-only

Validation subset:

    python ghost_oracle/T_S/probes/t_s_probe7_qpu_projection_raw_damage_cuda.py --validate-cpu

Outputs
-------
    qpu_projection_raw_damage_cuda_report.json
    qpu_projection_raw_damage_cuda_summary.csv
    qpu_projection_signature_raw_damage_cuda.npz
    edge_damage_aggregate.csv
    round_damage_aggregate.csv
    round_edge_damage_aggregate.csv
    coarse_raw_damage_aggregate.csv
    speed_summary.csv
    plots

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
    import cupy as cp
    _HAVE_CUPY = True
except Exception:
    cp = None
    _HAVE_CUPY = False

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
KERNEL_DIR = HERE.parent / "kernels"
KERNEL_PATH = KERNEL_DIR / "ts_geo_kernel.cu"


# --------------------------------------------------------------------------------------------------
# CONSTANTS / HELPERS
# --------------------------------------------------------------------------------------------------

EPS = 1e-12
ROUTE_STRIDE = 8
DAMAGE_STRIDE = 8


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if _HAVE_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x).tolist()
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
        raise KeyError("Probe 07 expects a `field` array in the dumped T_S npz.")

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


def gpu_info() -> Dict[str, Any]:
    info = {"cupy_available": bool(_HAVE_CUPY)}
    if not _HAVE_CUPY:
        return info
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        info.update({
            "device_id": int(dev.id),
            "name": str(name),
            "compute_capability": f"{props.get('major', '?')}.{props.get('minor', '?')}",
            "total_global_mem_gb": float(props.get("totalGlobalMem", 0)) / 1e9,
            "multi_processor_count": int(props.get("multiProcessorCount", -1)),
            "cupy_version": getattr(cp, "__version__", "unknown"),
        })
    except Exception as e:
        info["error"] = str(e)
    return info


def calibration_summary(meta: Optional[dict]) -> Dict[str, Any]:
    if not meta or not isinstance(meta, dict):
        return {"available": False}
    cal = meta.get("calibration")
    if not cal:
        return {"available": False}
    return {"available": True, "meta": cal.get("meta", {})}


# --------------------------------------------------------------------------------------------------
# STRESS FIELD CONSTRUCTION
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

    d_tau = d_tau[:, :, :-1, :-1]
    d_round = d_round[:-1, :, :, :-1]
    d_edge = d_edge[:-1, :, :-1, :]

    return d_tau, d_round, d_edge


def local_stress(block: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Shot-averaged stress field for one block.

    output shape:
        A, R, E = delay_cell, round_cell, edge_cell
    """
    d_tau, d_round, d_edge = gradients_delay_round_edge(block)
    if d_tau.size == 0:
        z = np.zeros((0, 0, 0), dtype=np.float32)
        return {
            "tau_tau": z, "rr": z, "xx": z,
            "tau_r": z, "tau_x": z, "r_x": z,
        }

    tau_tau = np.mean(d_tau * d_tau, axis=1)
    rr = np.mean(d_round * d_round, axis=1)
    xx = np.mean(d_edge * d_edge, axis=1)
    tau_r = np.mean(d_tau * d_round, axis=1)
    tau_x = np.mean(d_tau * d_edge, axis=1)
    r_x = np.mean(d_round * d_edge, axis=1)

    return {
        "tau_tau": finite_clean32(tau_tau),
        "rr": finite_clean32(rr),
        "xx": finite_clean32(xx),
        "tau_r": finite_clean32(tau_r),
        "tau_x": finite_clean32(tau_x),
        "r_x": finite_clean32(r_x),
    }


def trace_from_stress(st: Dict[str, np.ndarray]) -> np.ndarray:
    return finite_clean32(st["tau_tau"] + st["rr"] + st["xx"])


# --------------------------------------------------------------------------------------------------
# CPU ROUTE FALLBACK / VALIDATION
# --------------------------------------------------------------------------------------------------

def cpu_raw_routes_from_stress(
    st: Dict[str, np.ndarray],
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> np.ndarray:
    tau_tau = st["tau_tau"].astype(np.float64)
    rr = st["rr"].astype(np.float64)
    xx = st["xx"].astype(np.float64)
    tau_r = st["tau_r"].astype(np.float64)
    tau_x = st["tau_x"].astype(np.float64)
    r_x = st["r_x"].astype(np.float64)

    if tau_tau.size == 0:
        return np.full((ROUTE_STRIDE,), np.nan, dtype=np.float32)

    tau = np.clip(tau_tau + coupling_weight * (tau_r + tau_x), min_cost, max_cost)
    rnd = np.clip(rr + coupling_weight * (tau_r + r_x), min_cost, max_cost)
    edge = np.clip(xx + coupling_weight * (tau_x + r_x), min_cost, max_cost)

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

    trace = tau_tau + rr + xx
    trace_mean = float(np.mean(trace))

    steps = max(A, R, E)
    diag_vals = []
    for k in range(steps):
        a = min(A - 1, round(k * (A - 1) / max(steps - 1, 1)))
        r = min(R - 1, round(k * (R - 1) / max(steps - 1, 1)))
        e = min(E - 1, round(k * (E - 1) / max(steps - 1, 1)))
        diag_vals.append(trace[a, r, e])
    path_trace_mean = float(np.mean(diag_vals)) if diag_vals else float("nan")
    avoidance = trace_mean - path_trace_mean

    return np.asarray([
        full,
        delay,
        edge_cost,
        delay / (full + EPS),
        edge_cost / (full + EPS),
        trace_mean,
        avoidance,
        path_trace_mean,
    ], dtype=np.float32)


def raw_damage(real_route: np.ndarray, ab_route: np.ndarray) -> np.ndarray:
    full_delta = ab_route[0] - real_route[0]
    delay_delta = ab_route[1] - real_route[1]
    edge_delta = ab_route[2] - real_route[2]
    avoidance_loss = max(0.0, float(real_route[6] - ab_route[6]))

    abs_full = abs(float(full_delta))
    abs_delay = abs(float(delay_delta))
    abs_edge = abs(float(edge_delta))
    total = abs_full + abs_delay + abs_edge + avoidance_loss

    return np.asarray([
        total,
        full_delta,
        delay_delta,
        edge_delta,
        avoidance_loss,
        abs_full,
        abs_delay,
        abs_edge,
    ], dtype=np.float32)


# --------------------------------------------------------------------------------------------------
# CUDA ROUTE KERNEL WRAPPER
# --------------------------------------------------------------------------------------------------

_KERNEL_CACHE = None


def load_route_kernel() -> Any:
    global _KERNEL_CACHE

    if _KERNEL_CACHE is not None:
        return _KERNEL_CACHE

    if not _HAVE_CUPY:
        raise RuntimeError("CuPy is not available.")
    if not KERNEL_PATH.exists():
        raise FileNotFoundError(f"Kernel not found: {KERNEL_PATH}")

    code = KERNEL_PATH.read_text(encoding="utf-8")
    module = cp.RawModule(
        code=code,
        options=("--std=c++11",),
        name_expressions=("ts_raw_geo_route_vector_kernel",),
    )
    _KERNEL_CACHE = module.get_function("ts_raw_geo_route_vector_kernel")
    return _KERNEL_CACHE


def cuda_route_vectors(
    stress_batch: Dict[str, np.ndarray],
    *,
    A: int,
    R: int,
    E: int,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    block_size: int,
    cpu_only: bool,
) -> np.ndarray:
    """
    Compute route vectors [B,8] for a batch of stress fields.

    stress_batch components:
        [B,A,R,E]
    """
    B = int(stress_batch["tau_tau"].shape[0])

    if cpu_only or not _HAVE_CUPY:
        out = np.zeros((B, ROUTE_STRIDE), dtype=np.float32)
        for i in range(B):
            st = {k: v[i] for k, v in stress_batch.items()}
            out[i] = cpu_raw_routes_from_stress(
                st,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
            )
        return out

    kernel = load_route_kernel()
    N = A * R * E

    comps = {
        k: cp.asarray(np.ascontiguousarray(v.reshape(B, N).astype(np.float32)))
        for k, v in stress_batch.items()
    }

    dp = cp.empty((B, N), dtype=cp.float32)
    routes = cp.empty((B, ROUTE_STRIDE), dtype=cp.float32)

    grid = ((B + block_size - 1) // block_size,)
    block = (block_size,)

    kernel(
        grid,
        block,
        (
            comps["tau_tau"],
            comps["rr"],
            comps["xx"],
            comps["tau_r"],
            comps["tau_x"],
            comps["r_x"],
            dp,
            routes,
            np.int32(B),
            np.int32(A),
            np.int32(R),
            np.int32(E),
            np.float32(coupling_weight),
            np.float32(min_cost),
            np.float32(max_cost),
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(routes).astype(np.float32)


# --------------------------------------------------------------------------------------------------
# ABLATIONS
# --------------------------------------------------------------------------------------------------

def replace_edge_with_cell_marginal(block: np.ndarray, edge_index: int, rng: np.random.Generator) -> np.ndarray:
    f = block.copy()
    if edge_index < 0 or edge_index >= f.shape[3]:
        return f
    p = f[:, :, :, edge_index].mean(axis=1, keepdims=True)
    f[:, :, :, edge_index] = (rng.random(f[:, :, :, edge_index].shape) < p).astype(np.uint8)
    return f


def replace_round_with_cell_marginal(block: np.ndarray, round_index: int, rng: np.random.Generator) -> np.ndarray:
    f = block.copy()
    if round_index < 0 or round_index >= f.shape[2]:
        return f
    p = f[:, :, round_index, :].mean(axis=1, keepdims=True)
    f[:, :, round_index, :] = (rng.random(f[:, :, round_index, :].shape) < p).astype(np.uint8)
    return f


def replace_round_edge_with_cell_marginal(
    block: np.ndarray,
    round_index: int,
    edge_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    f = block.copy()
    if round_index < 0 or round_index >= f.shape[2]:
        return f
    if edge_index < 0 or edge_index >= f.shape[3]:
        return f
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
                perm = rng.permutation(f.shape[2])
                f[d, sh, :, e] = f[d, sh, perm, e]
    return f


def uniform_by_cell(block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = block.mean(axis=1, keepdims=True)
    return (rng.random(block.shape) < p).astype(np.uint8)


def all_uniform(block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = float(block.mean())
    return (rng.random(block.shape) < p).astype(np.uint8)


COARSE_CONTROL_FNS = {
    "edge_shuffle": shuffle_edges,
    "round_shuffle": shuffle_rounds,
    "round_reverse": lambda b, rng: b[:, :, ::-1, :].copy(),
    "edge_reverse": lambda b, rng: b[:, :, :, ::-1].copy(),
    "delay_shuffle": lambda b, rng: b[rng.permutation(b.shape[0]), :, :, :].copy(),
    "delay_reverse": lambda b, rng: b[::-1, :, :, :].copy(),
    "uniform_by_cell": uniform_by_cell,
    "all_uniform": all_uniform,
}


# --------------------------------------------------------------------------------------------------
# PROFILE EXTRACTION
# --------------------------------------------------------------------------------------------------

def extract_raw_profiles(block: np.ndarray, st: Dict[str, np.ndarray], route: np.ndarray) -> Dict[str, np.ndarray]:
    f = block.astype(np.float64)

    field_edge = f.mean(axis=(0, 1, 2))
    field_round = f.mean(axis=(0, 1, 3))
    field_delay = f.mean(axis=(1, 2, 3))
    field_round_edge = f.mean(axis=(0, 1))
    field_delay_edge = f.mean(axis=(1, 2))
    field_delay_round = f.mean(axis=(1, 3))

    trace = trace_from_stress(st)
    if trace.size:
        stress_edge = trace.mean(axis=(0, 1))
        stress_round = trace.mean(axis=(0, 2))
        stress_delay = trace.mean(axis=(1, 2))
        stress_round_edge = trace.mean(axis=0)
        stress_delay_edge = trace.mean(axis=1)
        stress_delay_round = trace.mean(axis=2)
    else:
        stress_edge = np.zeros((0,), dtype=np.float32)
        stress_round = np.zeros((0,), dtype=np.float32)
        stress_delay = np.zeros((0,), dtype=np.float32)
        stress_round_edge = np.zeros((0, 0), dtype=np.float32)
        stress_delay_edge = np.zeros((0, 0), dtype=np.float32)
        stress_delay_round = np.zeros((0, 0), dtype=np.float32)

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
        "raw_geo_routes": finite_clean32(route),
    }


# --------------------------------------------------------------------------------------------------
# BATCHED ANALYSIS
# --------------------------------------------------------------------------------------------------

def stress_batch_from_blocks(blocks: List[np.ndarray]) -> Dict[str, np.ndarray]:
    stresses = [local_stress(b) for b in blocks]
    keys = ("tau_tau", "rr", "xx", "tau_r", "tau_x", "r_x")
    return {k: finite_clean32(np.stack([s[k] for s in stresses], axis=0)) for k in keys}


def analyze_block_optimized(
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
    block_size: int,
    cpu_only: bool,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    real_st = local_stress(block)
    A, R, E = real_st["tau_tau"].shape

    real_route = cuda_route_vectors(
        {k: real_st[k][None, :, :, :] for k in ("tau_tau", "rr", "xx", "tau_r", "tau_x", "r_x")},
        A=A, R=R, E=E,
        coupling_weight=coupling_weight,
        min_cost=min_cost,
        max_cost=max_cost,
        block_size=block_size,
        cpu_only=cpu_only,
    )[0]

    profiles = extract_raw_profiles(block, real_st, real_route)

    edge_count = block.shape[3]
    round_count = block.shape[2]

    # Build all ablations once, then run route kernel in one batch.
    ablation_meta: List[Dict[str, Any]] = []
    ablation_blocks: List[np.ndarray] = []

    for e in range(edge_count):
        rng = np.random.default_rng(seed + 100003 * mode_index + 1009 * site_index + 17 * e)
        ablation_meta.append({"kind": "edge", "edge_index": e})
        ablation_blocks.append(replace_edge_with_cell_marginal(block, e, rng))

    for r in range(round_count):
        rng = np.random.default_rng(seed + 200003 * mode_index + 2009 * site_index + 19 * r)
        ablation_meta.append({"kind": "round", "round_index": r})
        ablation_blocks.append(replace_round_with_cell_marginal(block, r, rng))

    for r in range(round_count):
        for e in range(edge_count):
            rng = np.random.default_rng(seed + 300003 * mode_index + 3001 * site_index + 101 * r + e)
            ablation_meta.append({"kind": "round_edge", "round_index": r, "edge_index": e})
            ablation_blocks.append(replace_round_edge_with_cell_marginal(block, r, e, rng))

    for i, (control, fn) in enumerate(COARSE_CONTROL_FNS.items()):
        rng = np.random.default_rng(seed + 400003 * mode_index + 4001 * site_index + i)
        ablation_meta.append({"kind": "coarse", "control": control})
        ablation_blocks.append(fn(block, rng))

    t_ab_build = time.perf_counter()

    ab_stress_batch = stress_batch_from_blocks(ablation_blocks)
    t_stress = time.perf_counter()

    ab_routes = cuda_route_vectors(
        ab_stress_batch,
        A=A, R=R, E=E,
        coupling_weight=coupling_weight,
        min_cost=min_cost,
        max_cost=max_cost,
        block_size=block_size,
        cpu_only=cpu_only,
    )
    t_routes = time.perf_counter()

    edge_damage = np.zeros((edge_count,), dtype=np.float32)
    round_damage = np.zeros((round_count,), dtype=np.float32)
    round_edge_damage = np.zeros((round_count, edge_count), dtype=np.float32)

    edge_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []
    round_edge_rows: List[Dict[str, Any]] = []
    coarse_rows: List[Dict[str, Any]] = []

    for meta, route in zip(ablation_meta, ab_routes):
        dmg = raw_damage(real_route, route)
        row_base = {
            "mode": mode_name,
            "site": site_name,
            "mode_index": mode_index,
            "site_index": site_index,
            "damage": float(dmg[0]),
            "full_delta": float(dmg[1]),
            "delay_delta": float(dmg[2]),
            "edge_delta": float(dmg[3]),
            "avoidance_loss": float(dmg[4]),
            "full_abs_delta": float(dmg[5]),
            "delay_abs_delta": float(dmg[6]),
            "edge_abs_delta": float(dmg[7]),
        }

        if meta["kind"] == "edge":
            e = int(meta["edge_index"])
            edge_damage[e] = dmg[0]
            edge_rows.append({**row_base, "edge_index": e})

        elif meta["kind"] == "round":
            r = int(meta["round_index"])
            round_damage[r] = dmg[0]
            round_rows.append({**row_base, "round_index": r})

        elif meta["kind"] == "round_edge":
            r = int(meta["round_index"])
            e = int(meta["edge_index"])
            round_edge_damage[r, e] = dmg[0]
            round_edge_rows.append({**row_base, "round_index": r, "edge_index": e})

        elif meta["kind"] == "coarse":
            coarse_rows.append({**row_base, "control": str(meta["control"])})

    t_done = time.perf_counter()

    summary = {
        "mode": mode_name,
        "site": site_name,
        "mode_index": mode_index,
        "site_index": site_index,
        "full_cost": float(real_route[0]),
        "delay_cost": float(real_route[1]),
        "edge_cost": float(real_route[2]),
        "delay_to_full": float(real_route[3]),
        "edge_to_full": float(real_route[4]),
        "trace_mean": float(real_route[5]),
        "stress_avoidance": float(real_route[6]),
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

    speed = {
        "mode": mode_name,
        "site": site_name,
        "n_ablations": int(len(ablation_blocks)),
        "seconds_total": float(t_done - t0),
        "seconds_build_ablations": float(t_ab_build - t0),
        "seconds_stress_batch": float(t_stress - t_ab_build),
        "seconds_route_batch": float(t_routes - t_stress),
        "ablations_per_second_total": float(len(ablation_blocks) / max(t_done - t0, EPS)),
        "ablations_per_second_route": float(len(ablation_blocks) / max(t_routes - t_stress, EPS)),
    }

    return {
        "summary": summary,
        "profiles": profiles,
        "edge_damage": edge_damage,
        "round_damage": round_damage,
        "round_edge_damage": round_edge_damage,
        "edge_rows": edge_rows,
        "round_rows": round_rows,
        "round_edge_rows": round_edge_rows,
        "coarse_rows": coarse_rows,
        "speed": speed,
        "real_route": real_route,
        "real_stress": real_st,
    }


def run_probe(
    ts: Dict[str, Any],
    *,
    seed: int,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    block_size: int,
    cpu_only: bool,
) -> Dict[str, Any]:
    field = ts["field"]
    modes = ts["modes"]
    sites = ts["delay_sites"]

    M, S = field.shape[0], field.shape[1]
    edge_count = field.shape[5]
    round_count = field.shape[4]
    delay_count = field.shape[2]

    stress_delay_cells = max(delay_count - 1, 0)
    stress_round_cells = max(round_count - 1, 0)
    stress_edge_cells = max(edge_count - 1, 0)

    summaries: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []
    round_edge_rows: List[Dict[str, Any]] = []
    coarse_rows: List[Dict[str, Any]] = []
    speed_rows: List[Dict[str, Any]] = []

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

    raw_geo_routes = np.zeros((M, S, ROUTE_STRIDE), dtype=np.float32)
    edge_damage = np.zeros((M, S, edge_count), dtype=np.float32)
    round_damage = np.zeros((M, S, round_count), dtype=np.float32)
    round_edge_damage = np.zeros((M, S, round_count, edge_count), dtype=np.float32)

    for mi, mode_name in enumerate(modes):
        for si, site_name in enumerate(sites):
            block = field[mi, si]

            result = analyze_block_optimized(
                block,
                mode_index=mi,
                site_index=si,
                mode_name=mode_name,
                site_name=site_name,
                seed=seed,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
                block_size=block_size,
                cpu_only=cpu_only,
            )

            summaries.append(result["summary"])
            edge_rows.extend(result["edge_rows"])
            round_rows.extend(result["round_rows"])
            round_edge_rows.extend(result["round_edge_rows"])
            coarse_rows.extend(result["coarse_rows"])
            speed_rows.append(result["speed"])

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
        "speed_rows": speed_rows,
        "edge_aggregate": aggregate_by_index(edge_rows, "edge_index"),
        "round_aggregate": aggregate_by_index(round_rows, "round_index"),
        "round_edge_aggregate": aggregate_by_pair(round_edge_rows, "round_index", "edge_index"),
        "coarse_aggregate": aggregate_by_control(coarse_rows),
        "signature": signature,
    }


# --------------------------------------------------------------------------------------------------
# VALIDATION
# --------------------------------------------------------------------------------------------------

def validate_cpu_subset(
    ts: Dict[str, Any],
    *,
    seed: int,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    block_size: int,
    max_blocks: int,
) -> Dict[str, Any]:
    """
    Compares CUDA route vector against CPU route vector on real blocks.
    """
    if not _HAVE_CUPY:
        return {"ran": False, "reason": "CuPy not available."}

    rows = []
    count = 0

    for mi, mode_name in enumerate(ts["modes"]):
        for si, site_name in enumerate(ts["delay_sites"]):
            if count >= max_blocks:
                break

            block = ts["field"][mi, si]
            st = local_stress(block)
            A, R, E = st["tau_tau"].shape

            cpu_route = cpu_raw_routes_from_stress(
                st,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
            )

            cuda_route = cuda_route_vectors(
                {k: st[k][None, :, :, :] for k in ("tau_tau", "rr", "xx", "tau_r", "tau_x", "r_x")},
                A=A, R=R, E=E,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
                block_size=block_size,
                cpu_only=False,
            )[0]

            rows.append({
                "mode": mode_name,
                "site": site_name,
                "max_abs_diff": float(np.max(np.abs(cpu_route.astype(np.float64) - cuda_route.astype(np.float64)))),
                "cpu_route": cpu_route,
                "cuda_route": cuda_route,
            })

            count += 1

        if count >= max_blocks:
            break

    return {
        "ran": True,
        "rows": rows,
        "max_abs_diff": float(max(r["max_abs_diff"] for r in rows)) if rows else float("nan"),
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


def save_signature_npz(signature: Dict[str, np.ndarray], ts: Dict[str, Any], job_id: str, out_path: Path) -> None:
    payload = {
        "schema": np.array("ts_qpu_projection_signature_raw_damage_cuda"),
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


def plot_edge_round_damage(edge_agg: List[Dict[str, Any]], round_agg: List[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAVE_MPL:
        return

    if edge_agg:
        labels = [str(r["edge_index"]) for r in edge_agg]
        vals = [r["mean_damage"] for r in edge_agg]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        ax.bar(np.arange(len(labels)), vals)
        ax.set_title("T_S Probe 07 — CUDA raw edge damage")
        ax.set_xlabel("edge index")
        ax.set_ylabel("mean raw damage")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(out_dir / "edge_damage_cuda.png", bbox_inches="tight")
        plt.close(fig)

    if round_agg:
        labels = [str(r["round_index"]) for r in round_agg]
        vals = [r["mean_damage"] for r in round_agg]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        ax.bar(np.arange(len(labels)), vals)
        ax.set_title("T_S Probe 07 — CUDA raw round damage")
        ax.set_xlabel("round index")
        ax.set_ylabel("mean raw damage")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(out_dir / "round_damage_cuda.png", bbox_inches="tight")
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
    ax.set_title("T_S Probe 07 — CUDA raw round×edge damage")
    ax.set_xlabel("edge index")
    ax.set_ylabel("round index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean raw damage")
    fig.savefig(out_dir / "round_edge_damage_heatmap_cuda.png", bbox_inches="tight")
    plt.close(fig)


def plot_coarse_damage(coarse_agg: List[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAVE_MPL or not coarse_agg:
        return
    labels = [r["control"] for r in coarse_agg]
    vals = [r["mean_damage"] for r in coarse_agg]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax.bar(np.arange(len(labels)), vals)
    ax.set_title("T_S Probe 07 — CUDA coarse raw structure damage")
    ax.set_ylabel("mean raw damage")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out_dir / "coarse_raw_damage_cuda.png", bbox_inches="tight")
    plt.close(fig)


def print_summary(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 132)
    print("  T_S PROBE 07 — OPTIMIZED QPU RAW-DAMAGE SIGNATURE SUMMARY")
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

    print("\nSpeed summary:")
    total_ablations = sum(r["n_ablations"] for r in report["speed_rows"])
    total_seconds = sum(r["seconds_total"] for r in report["speed_rows"])
    route_seconds = sum(r["seconds_route_batch"] for r in report["speed_rows"])
    print(f"  total ablations       : {total_ablations}")
    print(f"  total seconds         : {total_seconds:.6f}")
    print(f"  route seconds         : {route_seconds:.6f}")
    print(f"  total ablations/sec   : {total_ablations / max(total_seconds, EPS):,.2f}")
    print(f"  route ablations/sec   : {total_ablations / max(route_seconds, EPS):,.2f}")

    val = report.get("validation", {})
    if val.get("ran"):
        print(f"\nValidation max_abs_diff CUDA vs CPU routes: {val.get('max_abs_diff'):.8g}")


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T_S Probe 07 — optimized CUDA raw-damage QPU projection signature.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz", default=None, help="Dumped T_S QPU .npz. Defaults to latest.")
    p.add_argument("--meta", default=None, help="Optional T_S job metadata JSON.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--coupling-weight", type=float, default=0.50)
    p.add_argument("--min-cost", type=float, default=1e-9)
    p.add_argument("--max-cost", type=float, default=1e9)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--cpu-only", action="store_true", help="Disable CUDA and run CPU fallback.")
    p.add_argument("--validate-cpu", action="store_true", help="Validate CUDA routes against CPU on a small subset.")
    p.add_argument("--validate-blocks", type=int, default=3)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    npz_path, meta_path, job_id = resolve_inputs(args.npz, args.meta)
    ts = load_ts(npz_path)
    meta = load_json(meta_path) if meta_path and meta_path.exists() else None

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"ts_qpu_projection_raw_damage_cuda_{job_id}_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 132)
    print("  T_S PROBE 07 — OPTIMIZED QPU PROJECTION RAW-DAMAGE SIGNATURE")
    print("=" * 132)
    print(f"  NPZ        : {npz_path}")
    print(f"  Metadata   : {meta_path if meta_path else '(not provided)'}")
    print(f"  Out dir    : {out_dir}")
    print(f"  Kernel     : {KERNEL_PATH}")
    print(f"  GPU        : {gpu_info()}")
    print(f"  Field shape: {ts['field'].shape}")
    print(f"  Modes      : {ts['modes']}")
    print(f"  Sites      : {ts['delay_sites']}")
    print(f"  Delays     : {ts['delays']} {ts['delay_unit']}")
    print("  Method     : optimized raw route/stress damage only; no normalization, no cosine, no classifier")

    if not args.cpu_only and not _HAVE_CUPY:
        raise RuntimeError("CuPy is not available. Use --cpu-only or install CuPy.")
    if not args.cpu_only and not KERNEL_PATH.exists():
        raise FileNotFoundError(f"Enhanced kernel not found: {KERNEL_PATH}")

    validation = {"ran": False}
    if args.validate_cpu and not args.cpu_only:
        validation = validate_cpu_subset(
            ts,
            seed=int(args.seed),
            coupling_weight=float(args.coupling_weight),
            min_cost=float(args.min_cost),
            max_cost=float(args.max_cost),
            block_size=int(args.block_size),
            max_blocks=int(args.validate_blocks),
        )

    core = run_probe(
        ts,
        seed=int(args.seed),
        coupling_weight=float(args.coupling_weight),
        min_cost=float(args.min_cost),
        max_cost=float(args.max_cost),
        block_size=int(args.block_size),
        cpu_only=bool(args.cpu_only),
    )

    signature_npz = out_dir / "qpu_projection_signature_raw_damage_cuda.npz"
    save_signature_npz(core["signature"], ts, job_id, signature_npz)

    report = {
        "schema": "ts_probe7_qpu_projection_raw_damage_cuda",
        "description": (
            "Optimized CUDA version of corrected Probe 06. Uses enhanced ts_geo_kernel.cu "
            "route-vector kernel for raw QPU projection damage. No normalization, cosine, "
            "classifier, GPU base generation, or projector benchmark."
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
            "block_size": int(args.block_size),
            "cpu_only": bool(args.cpu_only),
            "normalization": "none",
            "comparison": "direct_raw_damage",
            "kernel": str(KERNEL_PATH),
            "kernel_entry": "ts_raw_geo_route_vector_kernel",
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
        "gpu": gpu_info(),
        "calibration_summary": calibration_summary(meta),
        "validation": validation,
        "summary": core["summary"],
        "edge_rows": core["edge_rows"],
        "round_rows": core["round_rows"],
        "round_edge_rows": core["round_edge_rows"],
        "coarse_rows": core["coarse_rows"],
        "speed_rows": core["speed_rows"],
        "edge_aggregate": core["edge_aggregate"],
        "round_aggregate": core["round_aggregate"],
        "round_edge_aggregate": core["round_edge_aggregate"],
        "coarse_aggregate": core["coarse_aggregate"],
    }

    with open(out_dir / "qpu_projection_raw_damage_cuda_report.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2)

    write_csv(report["summary"], out_dir / "qpu_projection_raw_damage_cuda_summary.csv")
    write_csv(report["edge_rows"], out_dir / "edge_damage_rows_cuda.csv")
    write_csv(report["round_rows"], out_dir / "round_damage_rows_cuda.csv")
    write_csv(report["round_edge_rows"], out_dir / "round_edge_damage_rows_cuda.csv")
    write_csv(report["coarse_rows"], out_dir / "coarse_raw_damage_rows_cuda.csv")
    write_csv(report["speed_rows"], out_dir / "speed_summary.csv")
    write_csv(report["edge_aggregate"], out_dir / "edge_damage_aggregate_cuda.csv")
    write_csv(report["round_aggregate"], out_dir / "round_damage_aggregate_cuda.csv")
    write_csv(report["round_edge_aggregate"], out_dir / "round_edge_damage_aggregate_cuda.csv")
    write_csv(report["coarse_aggregate"], out_dir / "coarse_raw_damage_aggregate_cuda.csv")

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
