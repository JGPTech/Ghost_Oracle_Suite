#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
T_S PROBE 04 — STANDALONE RAW GEO SPEED MODEL
====================================================================================================
Final standalone geo probe before projector work.

Purpose
-------
This probe does NOT depend on QPU bases, T_S dumps, G_M bases, or S_M bases.

It creates bounded synthetic Temporal Stress Metric fields and benchmarks a
standalone raw-geo route engine against adjacent baseline approaches:

    1. T_S custom CUDA kernel
       Specialized structured-grid monotonic route engine.
       Intended future GPU fast path.

    2. CuPy vectorized dynamic-programming baseline
       GPU array baseline using Python-level wavefront/triple-loop control.

    3. NumPy CPU dynamic-programming baseline
       Same structured-grid recurrence on CPU.

    4. SciPy sparse csgraph Dijkstra baseline, optional
       General positive weighted graph shortest path on a sparse grid graph.

    5. NetworkX Dijkstra baseline, optional/tiny only
       General Python graph baseline for sanity, not large-scale speed.

The current best-practice reference points are intentionally adjacent rather
than identical:
    - RAPIDS cuGraph provides CUDA graph analytics and weighted SSSP for general
      graphs.
    - SciPy exposes sparse graph shortest-path and Dijkstra implementations.
    - NetworkX exposes Dijkstra shortest-path APIs for Python graph workflows.

T_S geo is different by design:
    - structured 3D lattice
    - bounded local stress components
    - monotonic path objective
    - future projector fusion

Expected repo layout
--------------------
Place this file at:

    ghost_oracle/T_S/probes/t_s_probe4_standalone_geo.py

Place the kernel at:

    ghost_oracle/T_S/Kernels/ts_geo_kernel.cu

Path convention:
    HERE = Path(__file__).resolve().parent
    DATA_DIR = HERE.parent / "data"
    ANALYSIS_DIR = HERE / "analysis"
    KERNEL_DIR = HERE.parent / "Kernels"

Usage
-----
Fast default:

    python ghost_oracle/T_S/probes/t_s_probe4_standalone_geo.py

3090-oriented larger run:

    python ghost_oracle/T_S/probes/t_s_probe4_standalone_geo.py --batch 4096 --dims 24 24 24

Skip slow baselines:

    python ghost_oracle/T_S/probes/t_s_probe4_standalone_geo.py --no-scipy --no-networkx

Outputs
-------
    ghost_oracle/T_S/probes/analysis/ts_geo_standalone_<timestamp>/
        probe4_standalone_geo_report.json
        probe4_standalone_geo_summary.csv
        speed_summary.png

====================================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
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
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
    _HAVE_SCIPY = True
except Exception:
    sp = None
    scipy_dijkstra = None
    _HAVE_SCIPY = False

try:
    import networkx as nx
    _HAVE_NETWORKX = True
except Exception:
    nx = None
    _HAVE_NETWORKX = False

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
KERNEL_DIR = HERE.parent / "Kernels"
KERNEL_PATH = KERNEL_DIR / "ts_geo_kernel.cu"


# --------------------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------------------

EPS = 1e-12


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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


def finite_clean_np(x: np.ndarray, clip_abs: float = 1e6) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=clip_abs, neginf=0.0)
    y = np.clip(y, 0.0, clip_abs)
    return y.astype(np.float32, copy=False)


def quantiles_np(x: np.ndarray) -> Dict[str, float]:
    y = np.asarray(x, dtype=np.float64).ravel()
    if y.size == 0:
        return {}
    return {
        "min": float(np.min(y)),
        "p50": float(np.percentile(y, 50)),
        "p90": float(np.percentile(y, 90)),
        "p99": float(np.percentile(y, 99)),
        "max": float(np.max(y)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
    }


def time_cpu(fn, repeats: int = 3) -> Tuple[float, Any]:
    best = float("inf")
    best_result = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
            best_result = result
    return best, best_result


def time_gpu(fn, repeats: int = 5) -> Tuple[float, Any]:
    if not _HAVE_CUPY:
        raise RuntimeError("CuPy is not available.")
    best = float("inf")
    best_result = None
    for _ in range(max(1, repeats)):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        result = fn()
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
            best_result = result
    return best, best_result


def gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cupy_available": bool(_HAVE_CUPY),
    }
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


# --------------------------------------------------------------------------------------------------
# SYNTHETIC BOUNDED T_S STRESS FIELD GENERATOR
# --------------------------------------------------------------------------------------------------

def make_synthetic_stress(
    batch: int,
    A: int,
    R: int,
    E: int,
    *,
    mode: str,
    seed: int,
    dtype=np.float32,
) -> Dict[str, np.ndarray]:
    """
    Create bounded local stress components.

    This intentionally mimics the Probe 01/02 structure without depending on
    bases or QPU outputs.

    Components:
        tau_tau, rr, xx, tau_r, tau_x, r_x

    Shapes:
        [B, A, R, E]
    """
    rng = np.random.default_rng(seed)
    shape = (batch, A, R, E)

    # Bounded base stress.
    tau_tau = rng.beta(2.0, 6.0, size=shape).astype(dtype) * 0.45
    rr = rng.beta(2.0, 7.0, size=shape).astype(dtype) * 0.35
    xx = rng.beta(2.0, 7.0, size=shape).astype(dtype) * 0.35

    a = np.linspace(0, 1, A, dtype=dtype)[None, :, None, None]
    r = np.linspace(0, 1, R, dtype=dtype)[None, None, :, None]
    e = np.linspace(0, 1, E, dtype=dtype)[None, None, None, :]

    # Smooth channel geometry scaffold.
    tau_tau += (0.025 + 0.020 * a).astype(dtype)
    rr += (0.020 + 0.015 * r).astype(dtype)
    xx += (0.020 + 0.015 * e).astype(dtype)

    if mode == "clean":
        pass

    elif mode == "phase_shear":
        shear = (0.08 * a * e + 0.04 * r * e).astype(dtype)
        tau_tau += 0.04 * shear
        rr += 0.06 * shear
        xx += 0.08 * shear

    elif mode == "local_shock":
        ca = 0.60
        cr = 0.55
        ce = 0.50
        sigma = 0.12
        shock = np.exp(
            -((a - ca) ** 2 + (r - cr) ** 2 + (e - ce) ** 2) / (2 * sigma * sigma)
        ).astype(dtype)
        tau_tau += 0.22 * shock
        rr += 0.18 * shock
        xx += 0.18 * shock

    elif mode == "corridor":
        # Low-cost diagonal-ish corridor in a bounded field.
        line = np.abs(a - r) + np.abs(r - e)
        corridor = np.exp(-(line ** 2) / (2 * 0.10 * 0.10)).astype(dtype)
        tau_tau += 0.12 * (1.0 - corridor)
        rr += 0.10 * (1.0 - corridor)
        xx += 0.10 * (1.0 - corridor)

    else:
        raise ValueError(f"unknown synthetic mode: {mode}")

    # Coupling components are bounded products plus small mode signal.
    tau_r = 0.35 * np.minimum(tau_tau, rr)
    tau_x = 0.35 * np.minimum(tau_tau, xx)
    r_x = 0.35 * np.minimum(rr, xx)

    if mode == "phase_shear":
        tau_x += (0.04 * a * e).astype(dtype)
        r_x += (0.03 * r * e).astype(dtype)
    elif mode == "local_shock":
        tau_r += 0.04 * np.minimum(tau_tau, rr)
        tau_x += 0.04 * np.minimum(tau_tau, xx)

    out = {
        "tau_tau": finite_clean_np(tau_tau),
        "rr": finite_clean_np(rr),
        "xx": finite_clean_np(xx),
        "tau_r": finite_clean_np(tau_r),
        "tau_x": finite_clean_np(tau_x),
        "r_x": finite_clean_np(r_x),
    }
    return out


def flatten_components(stress: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.ascontiguousarray(v.reshape(v.shape[0], -1).astype(np.float32)) for k, v in stress.items()}


# --------------------------------------------------------------------------------------------------
# RAW GEO RECURRENCE BASELINES
# --------------------------------------------------------------------------------------------------

def movement_costs_np(
    stress: Dict[str, np.ndarray],
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = stress["tau_tau"] + coupling_weight * (stress["tau_r"] + stress["tau_x"])
    rnd = stress["rr"] + coupling_weight * (stress["tau_r"] + stress["r_x"])
    edge = stress["xx"] + coupling_weight * (stress["tau_x"] + stress["r_x"])
    tau = np.clip(tau, min_cost, max_cost).astype(np.float32)
    rnd = np.clip(rnd, min_cost, max_cost).astype(np.float32)
    edge = np.clip(edge, min_cost, max_cost).astype(np.float32)
    return tau, rnd, edge


def raw_geo_numpy_dp(
    stress: Dict[str, np.ndarray],
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CPU structured-grid monotonic DP.
    """
    tau, rnd, edge = movement_costs_np(stress, coupling_weight, min_cost, max_cost)
    B, A, R, E = tau.shape
    dp = np.empty((B, A, R, E), dtype=np.float32)
    inf = np.float32(np.inf)

    for b in range(B):
        for a in range(A):
            for r in range(R):
                for e in range(E):
                    if a == 0 and r == 0 and e == 0:
                        dp[b, a, r, e] = 0.0
                        continue
                    best = inf
                    if a > 0:
                        best = min(best, dp[b, a - 1, r, e] + tau[b, a, r, e])
                    if r > 0:
                        best = min(best, dp[b, a, r - 1, e] + rnd[b, a, r, e])
                    if e > 0:
                        best = min(best, dp[b, a, r, e - 1] + edge[b, a, r, e])
                    dp[b, a, r, e] = best

    full = dp[:, -1, -1, -1].copy()
    cr = R // 2
    ce = E // 2
    ca = A // 2
    delay = tau[:, 1:, cr, ce].sum(axis=1)
    edge_only = edge[:, ca, cr, 1:].sum(axis=1)
    return full, delay.astype(np.float32), edge_only.astype(np.float32)


def raw_geo_cupy_vectorized_dp(
    stress_cp: Dict[str, Any],
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> Tuple[Any, Any, Any]:
    """
    CuPy baseline: GPU arrays, but Python-level DP loop control.

    This is useful because it is a common "best practical first try" for GPU
    acceleration before writing a custom kernel.
    """
    tau = cp.clip(stress_cp["tau_tau"] + coupling_weight * (stress_cp["tau_r"] + stress_cp["tau_x"]), min_cost, max_cost)
    rnd = cp.clip(stress_cp["rr"] + coupling_weight * (stress_cp["tau_r"] + stress_cp["r_x"]), min_cost, max_cost)
    edge = cp.clip(stress_cp["xx"] + coupling_weight * (stress_cp["tau_x"] + stress_cp["r_x"]), min_cost, max_cost)

    B, A, R, E = tau.shape
    dp = cp.empty((B, A, R, E), dtype=cp.float32)
    inf = cp.float32(3.402823466e38)

    for a in range(A):
        for r in range(R):
            for e in range(E):
                if a == 0 and r == 0 and e == 0:
                    dp[:, a, r, e] = 0.0
                    continue
                best = cp.full((B,), inf, dtype=cp.float32)
                if a > 0:
                    best = cp.minimum(best, dp[:, a - 1, r, e] + tau[:, a, r, e])
                if r > 0:
                    best = cp.minimum(best, dp[:, a, r - 1, e] + rnd[:, a, r, e])
                if e > 0:
                    best = cp.minimum(best, dp[:, a, r, e - 1] + edge[:, a, r, e])
                dp[:, a, r, e] = best

    full = dp[:, -1, -1, -1]
    cr = R // 2
    ce = E // 2
    ca = A // 2
    delay = tau[:, 1:, cr, ce].sum(axis=1)
    edge_only = edge[:, ca, cr, 1:].sum(axis=1)
    return full, delay, edge_only


# --------------------------------------------------------------------------------------------------
# CUSTOM CUDA KERNEL
# --------------------------------------------------------------------------------------------------

def load_cuda_kernel() -> Any:
    if not _HAVE_CUPY:
        raise RuntimeError("CuPy is required for the custom CUDA kernel.")
    if not KERNEL_PATH.exists():
        raise FileNotFoundError(f"Kernel not found: {KERNEL_PATH}")

    code = KERNEL_PATH.read_text(encoding="utf-8")
    module = cp.RawModule(code=code, options=("--std=c++11",), name_expressions=("ts_raw_geo_monotonic_kernel",))
    return module.get_function("ts_raw_geo_monotonic_kernel")


def raw_geo_cuda_kernel(
    stress_cp_flat: Dict[str, Any],
    *,
    B: int,
    A: int,
    R: int,
    E: int,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    block_size: int,
) -> Tuple[Any, Any, Any]:
    kernel = load_cuda_kernel()
    N = A * R * E

    dp = cp.empty((B, N), dtype=cp.float32)
    out_full = cp.empty((B,), dtype=cp.float32)
    out_delay = cp.empty((B,), dtype=cp.float32)
    out_edge = cp.empty((B,), dtype=cp.float32)

    grid = ((B + block_size - 1) // block_size,)
    block = (block_size,)

    kernel(
        grid,
        block,
        (
            stress_cp_flat["tau_tau"],
            stress_cp_flat["rr"],
            stress_cp_flat["xx"],
            stress_cp_flat["tau_r"],
            stress_cp_flat["tau_x"],
            stress_cp_flat["r_x"],
            dp,
            out_full,
            out_delay,
            out_edge,
            np.int32(B),
            np.int32(A),
            np.int32(R),
            np.int32(E),
            np.float32(coupling_weight),
            np.float32(min_cost),
            np.float32(max_cost),
        ),
    )

    return out_full, out_delay, out_edge


# --------------------------------------------------------------------------------------------------
# GENERAL GRAPH BASELINES
# --------------------------------------------------------------------------------------------------

def node_id(a: int, r: int, e: int, R: int, E: int) -> int:
    return (a * R + r) * E + e


def scipy_graph_for_instance(
    stress_one: Dict[str, np.ndarray],
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> Any:
    """
    Build sparse directed graph for monotonic forward moves only.
    """
    tau, rnd, edge = movement_costs_np({k: v[None, ...] for k, v in stress_one.items()}, coupling_weight, min_cost, max_cost)
    tau = tau[0]
    rnd = rnd[0]
    edge = edge[0]

    A, R, E = tau.shape
    rows = []
    cols = []
    data = []

    for a in range(A):
        for r in range(R):
            for e in range(E):
                u = node_id(a, r, e, R, E)
                if a + 1 < A:
                    v = node_id(a + 1, r, e, R, E)
                    rows.append(u); cols.append(v); data.append(float(tau[a + 1, r, e]))
                if r + 1 < R:
                    v = node_id(a, r + 1, e, R, E)
                    rows.append(u); cols.append(v); data.append(float(rnd[a, r + 1, e]))
                if e + 1 < E:
                    v = node_id(a, r, e + 1, R, E)
                    rows.append(u); cols.append(v); data.append(float(edge[a, r, e + 1]))

    N = A * R * E
    return sp.csr_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(N, N))


def scipy_dijkstra_single(
    stress_one: Dict[str, np.ndarray],
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> float:
    graph = scipy_graph_for_instance(stress_one, coupling_weight=coupling_weight, min_cost=min_cost, max_cost=max_cost)
    A, R, E = stress_one["tau_tau"].shape
    source = 0
    target = node_id(A - 1, R - 1, E - 1, R, E)
    dist = scipy_dijkstra(graph, directed=True, indices=source, return_predecessors=False)
    return float(dist[target])


def networkx_dijkstra_single(
    stress_one: Dict[str, np.ndarray],
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> float:
    tau, rnd, edge = movement_costs_np({k: v[None, ...] for k, v in stress_one.items()}, coupling_weight, min_cost, max_cost)
    tau = tau[0]
    rnd = rnd[0]
    edge = edge[0]
    A, R, E = tau.shape

    G = nx.DiGraph()
    for a in range(A):
        for r in range(R):
            for e in range(E):
                u = (a, r, e)
                if a + 1 < A:
                    G.add_edge(u, (a + 1, r, e), weight=float(tau[a + 1, r, e]))
                if r + 1 < R:
                    G.add_edge(u, (a, r + 1, e), weight=float(rnd[a, r + 1, e]))
                if e + 1 < E:
                    G.add_edge(u, (a, r, e + 1), weight=float(edge[a, r, e + 1]))

    return float(nx.dijkstra_path_length(G, (0, 0, 0), (A - 1, R - 1, E - 1), weight="weight"))


# --------------------------------------------------------------------------------------------------
# BENCHMARK
# --------------------------------------------------------------------------------------------------

def max_abs_diff(a: Any, b: Any) -> float:
    aa = cp.asnumpy(a) if _HAVE_CUPY and isinstance(a, cp.ndarray) else np.asarray(a)
    bb = cp.asnumpy(b) if _HAVE_CUPY and isinstance(b, cp.ndarray) else np.asarray(b)
    return float(np.max(np.abs(aa.astype(np.float64) - bb.astype(np.float64))))


def throughput_routes_per_sec(batch: int, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return float(batch / seconds)


def benchmark_mode(
    *,
    mode: str,
    batch: int,
    A: int,
    R: int,
    E: int,
    seed: int,
    repeats_cpu: int,
    repeats_gpu: int,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    block_size: int,
    run_scipy: bool,
    scipy_instances: int,
    run_networkx: bool,
    networkx_instances: int,
) -> Dict[str, Any]:
    stress = make_synthetic_stress(batch, A, R, E, mode=mode, seed=seed)
    flat = flatten_components(stress)

    result: Dict[str, Any] = {
        "mode": mode,
        "batch": batch,
        "dims": [A, R, E],
        "nodes_per_instance": A * R * E,
        "stress_quantiles": {k: quantiles_np(v) for k, v in stress.items()},
        "benchmarks": {},
        "validation": {},
    }

    # CPU structured DP baseline.  For large B/dims this can be slow, so allow small CPU batch.
    cpu_batch = min(batch, max(1, int(256 if A * R * E <= 8192 else 64)))
    stress_cpu = {k: v[:cpu_batch].copy() for k, v in stress.items()}

    cpu_t, cpu_out = time_cpu(
        lambda: raw_geo_numpy_dp(
            stress_cpu,
            coupling_weight=coupling_weight,
            min_cost=min_cost,
            max_cost=max_cost,
        ),
        repeats=repeats_cpu,
    )
    result["benchmarks"]["numpy_structured_dp"] = {
        "seconds": cpu_t,
        "batch": cpu_batch,
        "routes_per_sec": throughput_routes_per_sec(cpu_batch, cpu_t),
        "note": "CPU structured-grid DP baseline, batch may be capped for runtime.",
    }

    cpu_full, cpu_delay, cpu_edge = cpu_out

    if _HAVE_CUPY:
        stress_cp = {k: cp.asarray(v) for k, v in stress.items()}
        stress_cp_flat = {k: cp.asarray(v) for k, v in flat.items()}

        # Custom kernel.
        cuda_t, cuda_out = time_gpu(
            lambda: raw_geo_cuda_kernel(
                stress_cp_flat,
                B=batch,
                A=A,
                R=R,
                E=E,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
                block_size=block_size,
            ),
            repeats=repeats_gpu,
        )
        cuda_full, cuda_delay, cuda_edge = cuda_out

        result["benchmarks"]["ts_custom_cuda_raw_geo"] = {
            "seconds": cuda_t,
            "batch": batch,
            "routes_per_sec": throughput_routes_per_sec(batch, cuda_t),
            "nodes_per_sec": float(batch * A * R * E / cuda_t) if cuda_t > 0 else float("inf"),
            "block_size": block_size,
            "note": "Specialized raw T_S structured-grid CUDA kernel.",
        }

        result["validation"]["cuda_vs_numpy_full_max_abs"] = max_abs_diff(cuda_full[:cpu_batch], cpu_full)
        result["validation"]["cuda_vs_numpy_delay_max_abs"] = max_abs_diff(cuda_delay[:cpu_batch], cpu_delay)
        result["validation"]["cuda_vs_numpy_edge_max_abs"] = max_abs_diff(cuda_edge[:cpu_batch], cpu_edge)

        # CuPy vectorized DP baseline.
        cupy_t, cupy_out = time_gpu(
            lambda: raw_geo_cupy_vectorized_dp(
                stress_cp,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
            ),
            repeats=max(1, min(repeats_gpu, 3)),
        )
        cupy_full, cupy_delay, cupy_edge = cupy_out

        result["benchmarks"]["cupy_vectorized_dp"] = {
            "seconds": cupy_t,
            "batch": batch,
            "routes_per_sec": throughput_routes_per_sec(batch, cupy_t),
            "nodes_per_sec": float(batch * A * R * E / cupy_t) if cupy_t > 0 else float("inf"),
            "note": "GPU array baseline with Python-level DP loop control.",
        }

        result["validation"]["cupy_vs_cuda_full_max_abs"] = max_abs_diff(cupy_full, cuda_full)
        result["validation"]["cupy_vs_cuda_delay_max_abs"] = max_abs_diff(cupy_delay, cuda_delay)
        result["validation"]["cupy_vs_cuda_edge_max_abs"] = max_abs_diff(cupy_edge, cuda_edge)

    else:
        result["benchmarks"]["ts_custom_cuda_raw_geo"] = {
            "skipped": True,
            "reason": "CuPy not available.",
        }
        result["benchmarks"]["cupy_vectorized_dp"] = {
            "skipped": True,
            "reason": "CuPy not available.",
        }

    # SciPy general graph Dijkstra baseline.
    if run_scipy and _HAVE_SCIPY:
        n = min(batch, scipy_instances)
        def scipy_run():
            vals = []
            for i in range(n):
                one = {k: v[i] for k, v in stress.items()}
                vals.append(
                    scipy_dijkstra_single(
                        one,
                        coupling_weight=coupling_weight,
                        min_cost=min_cost,
                        max_cost=max_cost,
                    )
                )
            return np.asarray(vals, dtype=np.float32)

        scipy_t, scipy_vals = time_cpu(scipy_run, repeats=1)
        result["benchmarks"]["scipy_sparse_dijkstra"] = {
            "seconds": scipy_t,
            "batch": n,
            "routes_per_sec": throughput_routes_per_sec(n, scipy_t),
            "note": "General sparse positive weighted shortest-path baseline.",
        }
        result["validation"]["scipy_vs_numpy_full_max_abs"] = max_abs_diff(scipy_vals, cpu_full[:n])
    else:
        result["benchmarks"]["scipy_sparse_dijkstra"] = {
            "skipped": True,
            "reason": "disabled or SciPy unavailable.",
        }

    # NetworkX baseline, tiny only.
    if run_networkx and _HAVE_NETWORKX:
        n = min(batch, networkx_instances)
        def nx_run():
            vals = []
            for i in range(n):
                one = {k: v[i] for k, v in stress.items()}
                vals.append(
                    networkx_dijkstra_single(
                        one,
                        coupling_weight=coupling_weight,
                        min_cost=min_cost,
                        max_cost=max_cost,
                    )
                )
            return np.asarray(vals, dtype=np.float32)

        nx_t, nx_vals = time_cpu(nx_run, repeats=1)
        result["benchmarks"]["networkx_dijkstra"] = {
            "seconds": nx_t,
            "batch": n,
            "routes_per_sec": throughput_routes_per_sec(n, nx_t),
            "note": "General Python graph baseline; intended tiny sanity check only.",
        }
        result["validation"]["networkx_vs_numpy_full_max_abs"] = max_abs_diff(nx_vals, cpu_full[:n])
    else:
        result["benchmarks"]["networkx_dijkstra"] = {
            "skipped": True,
            "reason": "disabled or NetworkX unavailable.",
        }

    # Speedup summaries against available baselines.
    cuda = result["benchmarks"].get("ts_custom_cuda_raw_geo", {})
    if "routes_per_sec" in cuda:
        c_rps = cuda["routes_per_sec"]
        for key, b in list(result["benchmarks"].items()):
            if key == "ts_custom_cuda_raw_geo":
                continue
            if isinstance(b, dict) and "routes_per_sec" in b and b["routes_per_sec"] > 0:
                result["benchmarks"][key]["custom_cuda_speedup_x"] = float(c_rps / b["routes_per_sec"])

    return result


def make_summary_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for mode_result in report["mode_results"]:
        mode = mode_result["mode"]
        for name, bench in mode_result["benchmarks"].items():
            row = {
                "mode": mode,
                "benchmark": name,
                "batch": bench.get("batch"),
                "seconds": bench.get("seconds"),
                "routes_per_sec": bench.get("routes_per_sec"),
                "nodes_per_sec": bench.get("nodes_per_sec"),
                "custom_cuda_speedup_x": bench.get("custom_cuda_speedup_x"),
                "skipped": bench.get("skipped", False),
                "reason": bench.get("reason", ""),
            }
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_speed(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAVE_MPL:
        return

    usable = [r for r in rows if not r.get("skipped") and r.get("routes_per_sec") not in (None, "")]
    if not usable:
        return

    labels = [f"{r['mode']}\n{r['benchmark']}" for r in usable]
    values = [float(r["routes_per_sec"]) for r in usable]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.55), 6.5), dpi=160)
    ax.bar(np.arange(len(labels)), values)
    ax.set_yscale("log")
    ax.set_ylabel("routes/sec (log scale)")
    ax.set_title("T_S Probe 04 — Standalone raw geo speed")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out_dir / "speed_summary.png", bbox_inches="tight")
    plt.close(fig)


def print_summary(report: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 132)
    print("  T_S PROBE 04 — STANDALONE RAW GEO SPEED SUMMARY")
    print("=" * 132)
    print(
        f"  {'mode':>13} | {'benchmark':>26} | {'batch':>8} | {'seconds':>10} | "
        f"{'routes/s':>14} | {'nodes/s':>14} | {'cuda speedup':>13}"
    )
    print("  " + "-" * 130)

    for r in rows:
        if r.get("skipped"):
            print(f"  {r['mode']:>13} | {r['benchmark']:>26} | {'SKIPPED':>8} | {str(r.get('reason'))}")
            continue
        seconds = float(r["seconds"])
        rps = float(r["routes_per_sec"])
        nps = r.get("nodes_per_sec")
        sx = r.get("custom_cuda_speedup_x")
        print(
            f"  {r['mode']:>13} | {r['benchmark']:>26} | {int(r['batch']):>8} | "
            f"{seconds:>10.6f} | {rps:>14,.2f} | "
            f"{float(nps):>14,.2f} | " if nps is not None else
            f"  {r['mode']:>13} | {r['benchmark']:>26} | {int(r['batch']):>8} | "
            f"{seconds:>10.6f} | {rps:>14,.2f} | {'n/a':>14} | ",
            end=""
        )
        if sx is None:
            print(f"{'n/a':>13}")
        else:
            print(f"{float(sx):>13.2f}x")

    print("\nValidation max absolute diffs:")
    for mr in report["mode_results"]:
        print(f"  {mr['mode']}: {mr.get('validation', {})}")


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T_S Probe 04 — standalone raw geo speed model with CUDA kernel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--dims", type=int, nargs=3, default=[16, 16, 16], metavar=("DELAY", "ROUND", "EDGE"))
    p.add_argument("--modes", nargs="+", default=["clean", "phase_shear", "local_shock", "corridor"],
                   choices=["clean", "phase_shear", "local_shock", "corridor"])
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--repeats-cpu", type=int, default=2)
    p.add_argument("--repeats-gpu", type=int, default=5)
    p.add_argument("--coupling-weight", type=float, default=0.50)
    p.add_argument("--min-cost", type=float, default=1e-9)
    p.add_argument("--max-cost", type=float, default=1e9)
    p.add_argument("--block-size", type=int, default=128)

    p.add_argument("--no-scipy", action="store_true")
    p.add_argument("--scipy-instances", type=int, default=8)
    p.add_argument("--no-networkx", action="store_true")
    p.add_argument("--networkx-instances", type=int, default=1)

    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    A, R, E = [int(x) for x in args.dims]
    if min(A, R, E) < 2:
        raise ValueError("--dims must all be >= 2")

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"ts_geo_standalone_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 132)
    print("  T_S PROBE 04 — STANDALONE RAW GEO SPEED MODEL")
    print("=" * 132)
    print(f"  Out dir     : {out_dir}")
    print(f"  Kernel      : {KERNEL_PATH}")
    print(f"  Batch       : {args.batch}")
    print(f"  Dims        : {A} x {R} x {E}  ({A*R*E:,} nodes/instance)")
    print(f"  Modes       : {args.modes}")
    print(f"  GPU info    : {gpu_info()}")
    print(f"  SciPy       : {_HAVE_SCIPY and not args.no_scipy}")
    print(f"  NetworkX    : {_HAVE_NETWORKX and not args.no_networkx}")

    if not KERNEL_PATH.exists():
        raise FileNotFoundError(
            f"Kernel not found at {KERNEL_PATH}. "
            "Place ts_geo_kernel.cu in ghost_oracle/T_S/Kernels/."
        )

    mode_results = []
    for i, mode in enumerate(args.modes):
        print(f"\n[RUN] mode={mode}")
        res = benchmark_mode(
            mode=mode,
            batch=int(args.batch),
            A=A,
            R=R,
            E=E,
            seed=int(args.seed + 1009 * i),
            repeats_cpu=int(args.repeats_cpu),
            repeats_gpu=int(args.repeats_gpu),
            coupling_weight=float(args.coupling_weight),
            min_cost=float(args.min_cost),
            max_cost=float(args.max_cost),
            block_size=int(args.block_size),
            run_scipy=(not args.no_scipy),
            scipy_instances=int(args.scipy_instances),
            run_networkx=(not args.no_networkx),
            networkx_instances=int(args.networkx_instances),
        )
        mode_results.append(res)

    report = {
        "schema": "ts_probe4_standalone_raw_geo_speed",
        "description": (
            "Standalone raw T_S geo speed model. Generates bounded synthetic stress fields, "
            "runs a custom CUDA structured-grid geo kernel, and compares against adjacent "
            "general/array baselines."
        ),
        "settings": {
            "batch": int(args.batch),
            "dims": [A, R, E],
            "modes": list(args.modes),
            "seed": int(args.seed),
            "repeats_cpu": int(args.repeats_cpu),
            "repeats_gpu": int(args.repeats_gpu),
            "coupling_weight": float(args.coupling_weight),
            "min_cost": float(args.min_cost),
            "max_cost": float(args.max_cost),
            "block_size": int(args.block_size),
            "scipy_instances": int(args.scipy_instances),
            "networkx_instances": int(args.networkx_instances),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "gpu": gpu_info(),
            "have_cupy": bool(_HAVE_CUPY),
            "have_scipy": bool(_HAVE_SCIPY),
            "have_networkx": bool(_HAVE_NETWORKX),
        },
        "baseline_context": {
            "cugraph": "General CUDA graph analytics / SSSP adjacent approach.",
            "scipy_csgraph_dijkstra": "General sparse graph shortest-path/Dijkstra baseline.",
            "networkx_dijkstra": "General Python graph Dijkstra baseline.",
            "cupy_vectorized_dp": "GPU array baseline before custom kernel fusion.",
            "ts_custom_cuda_raw_geo": "Specialized structured-grid raw geo path intended for future projector fusion.",
        },
        "mode_results": mode_results,
    }

    rows = make_summary_rows(report)

    with open(out_dir / "probe4_standalone_geo_report.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2)

    write_csv(rows, out_dir / "probe4_standalone_geo_summary.csv")

    if not args.no_plots:
        plot_speed(rows, out_dir)

    print_summary(report, rows)

    print(f"\n[SAVED] {out_dir}")
    print("=" * 132 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"[FATAL] {type(e).__name__}: {e}")
