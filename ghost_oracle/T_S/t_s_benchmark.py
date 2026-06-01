#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
T_S FINAL BENCHMARK — GEO / QPROJ / GPROJ
====================================================================================================
Final benchmark for T_S — Temporal Stress Metric.

Purpose
-------
Compare T_S substrate paths on one common task:

    route-preserving edge/round scaffold recovery

The benchmark uses the same analysis-facing T_S schema for both QPU and GPU
files:

    field[mode, delay_site, delay_value, shot, round, edge]
    final[mode, delay_site, delay_value, shot, channel]

Substrate paths
---------------
geo:
    Raw arithmetic route path derived from the T_S field.

qproj:
    QPU-generated field -> raw-damage projection signature.

gproj:
    GPU-generated field -> raw-damage projection signature.

Common task
-----------
For each T_S file, identify the scaffold components whose removal damages the
raw geo route most:

    edge_damage[edge]
    round_damage[round]
    round_edge_damage[round, edge]
    coarse_damage[control]

Then compare:
    - QPU vs GPU scaffold alignment,
    - CUDA route timing,
    - generic profile/scalar/stress baselines against the T_S raw-damage target.

Methodology
-----------
RAW ONLY for T_S damage.
NO normalization.
NO cosine.
NO classifier framing.

The T_S route evaluation uses the enhanced single-file kernel:

    ghost_oracle/T_S/kernels/ts_geo_kernel.cu
    ghost_oracle/T_S/Kernels/ts_geo_kernel.cu   # fallback path

Required kernel entry point:

    ts_raw_geo_route_vector_kernel

Benchmark baselines
-------------------
Baselines operate on the same ablation task, but use simpler non-route scoring:

    scalar_rate
        damage = abs(mean(ablated_field) - mean(real_field))

    field_profile_l1
        L1 damage on raw field delay/round/edge profiles.

    stress_profile_l1
        L1 damage on stress trace delay/round/edge profiles.

    route_cpu
        Optional CPU DP route baseline, for validation/timing only.

These are not claimed to be equivalent to T_S. They are adjacent generic methods
for the same scaffold-recovery task.

Usage
-----
Auto-load latest QPU and GPU pointers:

    python ghost_oracle/T_S/t_s_benchmark.py

Explicit files:

    python ghost_oracle/T_S/t_s_benchmark.py ^
      --qpu ghost_oracle/T_S/data/ts_data_<JOB_ID>.npz ^
      --qpu-meta ghost_oracle/T_S/data/ts_job_<JOB_ID>.json ^
      --gpu ghost_oracle/T_S/data/ts_gpu_data_<TAG>.npz ^
      --gpu-meta ghost_oracle/T_S/data/ts_gpu_job_<TAG>.json

PowerShell example:

    python ghost_oracle/T_S/t_s_benchmark.py `
      --qpu ghost_oracle/T_S/data/ts_data_d8e9ab3o3njc73eue47g.npz `
      --gpu ghost_oracle/T_S/data/ts_gpu_data_4096shots_seed4204302182560473160.npz

Optional CPU route timing:

    python ghost_oracle/T_S/t_s_benchmark.py --include-cpu-route

Outputs
-------
    ghost_oracle/T_S/analysis/ts_benchmark_<timestamp>/
      ts_benchmark_report.json
      ts_benchmark_summary.csv
      ts_benchmark_rows.csv
      qpu_gpu_alignment.csv
      baseline_comparison.csv
      benchmark_speed.csv
      edge_damage_compare.png
      round_damage_compare.png
      round_edge_damage_compare.png
      coarse_damage_compare.png

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
    _HAVE_SCIPY = True
except Exception:
    csr_matrix = None
    scipy_dijkstra = None
    _HAVE_SCIPY = False

try:
    import networkx as nx
    _HAVE_NETWORKX = True
except Exception:
    nx = None
    _HAVE_NETWORKX = False


# --------------------------------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
ANALYSIS_DIR = HERE / "analysis"

KERNEL_PATHS = [
    HERE / "kernels" / "ts_geo_kernel.cu",
    HERE / "Kernels" / "ts_geo_kernel.cu",
]


# --------------------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------------------

EPS = 1e-12
ROUTE_STRIDE = 8

COARSE_CONTROL_NAMES = [
    "edge_shuffle",
    "round_shuffle",
    "round_reverse",
    "edge_reverse",
    "delay_shuffle",
    "delay_reverse",
    "uniform_by_cell",
    "all_uniform",
]


# --------------------------------------------------------------------------------------------------
# BASIC HELPERS
# --------------------------------------------------------------------------------------------------

def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if _HAVE_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x).tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
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


def find_kernel_path() -> Path:
    for p in KERNEL_PATHS:
        if p.exists():
            return p
    return KERNEL_PATHS[0]


def latest_pointer(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        return read_json(path)
    return None


def resolve_source(npz_arg: Optional[str], meta_arg: Optional[str], latest_names: Sequence[str]) -> Tuple[Optional[Path], Optional[Path], str]:
    if npz_arg:
        npz = Path(npz_arg)
        meta = Path(meta_arg) if meta_arg else None
        job_id = npz.stem
        return npz, meta, job_id

    for name in latest_names:
        ptr = latest_pointer(DATA_DIR / name)
        if ptr and ptr.get("npz"):
            npz = Path(ptr["npz"])
            meta = Path(ptr["meta"]) if ptr.get("meta") else None
            job_id = str(ptr.get("job_id", npz.stem))
            if meta_arg:
                meta = Path(meta_arg)
            return npz, meta, job_id

    return None, None, ""


def load_ts(npz_path: Path, source_label: str) -> Dict[str, Any]:
    z = np.load(npz_path, allow_pickle=False)
    if "field" not in z.files:
        raise KeyError(f"{source_label}: expected `field` in {npz_path}")

    field = bits(z["field"])
    if field.ndim != 6:
        raise ValueError(
            f"{source_label}: expected field shape "
            "(modes, delay_sites, delays, shots, rounds, edges), got "
            f"{field.shape}"
        )

    def str_array(name: str) -> List[str]:
        if name not in z.files:
            return []
        return [str(x) for x in z[name].tolist()]

    modes = str_array("modes") or [f"mode_{i}" for i in range(field.shape[0])]
    sites = str_array("delay_sites") or [f"site_{i}" for i in range(field.shape[1])]

    obj = {
        "source_label": source_label,
        "npz": str(npz_path),
        "field": field,
        "final": bits(z["final"]) if "final" in z.files else None,
        "modes": modes,
        "delay_sites": sites,
        "delays": z["delays"].astype(int).tolist() if "delays" in z.files else list(range(field.shape[2])),
        "delay_unit": str(z["delay_unit"].item()) if "delay_unit" in z.files and z["delay_unit"].shape == () else "",
        "job_id": str(z["job_id"].item()) if "job_id" in z.files and z["job_id"].shape == () else npz_path.stem,
        "rounds": int(z["rounds"].item()) if "rounds" in z.files else int(field.shape[4]),
        "channels": int(z["channels"].item()) if "channels" in z.files else int(field.shape[5] + 1),
        "edges": int(z["edges"].item()) if "edges" in z.files else int(field.shape[5]),
    }
    return obj



def is_valid_ts_npz(path: Path) -> bool:
    """
    A valid benchmark input is any .npz in T_S/data that contains a 6D `field`
    array matching:

        field[mode, delay_site, delay, shot, round, edge]
    """
    try:
        z = np.load(path, allow_pickle=False)
        if "field" not in z.files:
            return False
        f = z["field"]
        return bool(f.ndim == 6)
    except Exception:
        return False


def classify_npz_source(path: Path) -> str:
    """
    Repo-native source label.

    Files produced by t_s_gpu_generate.py are gproj.
    QPU dumps are qproj.
    Unknown valid files are included as data_<stem>.
    """
    name = path.name.lower()
    try:
        z = np.load(path, allow_pickle=False)
        if "source" in z.files:
            source = str(z["source"].item()).lower()
            if source == "gpu":
                return "gproj"
            if source == "qpu":
                return "qproj"
        if "backend" in z.files:
            backend = str(z["backend"].item()).lower()
            if "gpu" in backend or "cupy" in backend:
                return "gproj"
    except Exception:
        pass

    if "gpu" in name:
        return "gproj"
    if name.startswith("ts_data_") or "qpu" in name:
        return "qproj"
    return f"data_{path.stem}"


def find_meta_for_npz(npz_path: Path) -> Optional[Path]:
    """
    Best-effort metadata match for either QPU or GPU generated files.
    """
    data_dir = npz_path.parent
    stem = npz_path.stem

    candidates = []

    # GPU file:
    #   ts_gpu_data_4096shots_seedX.npz
    #   ts_gpu_job_ts_gpu_4096shots_seedX.json
    if stem.startswith("ts_gpu_data_"):
        tag = stem.replace("ts_gpu_data_", "ts_gpu_", 1)
        candidates.append(data_dir / f"ts_gpu_job_{tag}.json")

    # QPU file:
    #   ts_data_JOBID.npz
    #   ts_job_JOBID.json
    if stem.startswith("ts_data_"):
        tag = stem.replace("ts_data_", "", 1)
        candidates.append(data_dir / f"ts_job_{tag}.json")

    # Generic fallbacks.
    candidates.extend([
        data_dir / f"{stem}.json",
        data_dir / f"{stem.replace('data', 'job', 1)}.json",
    ])

    for p in candidates:
        if p.exists():
            return p
    return None


def collect_input_files(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Default behavior:
        run every valid T_S .npz in ghost_oracle/T_S/data/

    CLI overrides:
        --files file1.npz file2.npz ...
        --qpu / --gpu legacy explicit pair mode
        --qpu-only / --gpu-only source filters
    """
    items: List[Dict[str, Any]] = []

    explicit_files = getattr(args, "files", None)
    if explicit_files:
        paths = [Path(p) for p in explicit_files]
    elif args.qpu or args.gpu:
        paths = []
        if args.qpu:
            paths.append(Path(args.qpu))
        if args.gpu:
            paths.append(Path(args.gpu))
    else:
        paths = sorted(DATA_DIR.glob("*.npz"))

    for path in paths:
        if not path.exists():
            print(f"[WARN] skipping missing file: {path}")
            continue
        if not is_valid_ts_npz(path):
            print(f"[WARN] skipping non-T_S npz: {path}")
            continue

        source = classify_npz_source(path)
        if args.qpu_only and source != "qproj":
            continue
        if args.gpu_only and source != "gproj":
            continue

        meta = None
        if args.qpu and Path(args.qpu) == path and args.qpu_meta:
            meta = Path(args.qpu_meta)
        elif args.gpu and Path(args.gpu) == path and args.gpu_meta:
            meta = Path(args.gpu_meta)
        else:
            meta = find_meta_for_npz(path)

        # Make labels unique if multiple GPU/QPU files are benchmarked.
        stem_tag = path.stem
        if source in ("qproj", "gproj"):
            label = f"{source}:{stem_tag}"
        else:
            label = source

        items.append({
            "npz": path,
            "meta": meta,
            "source": source,
            "label": label,
            "job_id": stem_tag,
        })

    return items


# --------------------------------------------------------------------------------------------------
# STRESS + ROUTES
# --------------------------------------------------------------------------------------------------

def gradients_delay_round_edge(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    block:
        delay, shot, round, edge

    output:
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
    d_tau, d_round, d_edge = gradients_delay_round_edge(block)
    if d_tau.size == 0:
        z = np.zeros((0, 0, 0), dtype=np.float32)
        return {"tau_tau": z, "rr": z, "xx": z, "tau_r": z, "tau_x": z, "r_x": z}

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
    diag = []
    for k in range(steps):
        a = min(A - 1, round(k * (A - 1) / max(steps - 1, 1)))
        r = min(R - 1, round(k * (R - 1) / max(steps - 1, 1)))
        e = min(E - 1, round(k * (E - 1) / max(steps - 1, 1)))
        diag.append(trace[a, r, e])
    path_trace_mean = float(np.mean(diag)) if diag else float("nan")
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


_KERNEL_CACHE = None


def load_route_kernel(kernel_path: Path):
    global _KERNEL_CACHE
    if _KERNEL_CACHE is not None:
        return _KERNEL_CACHE

    if not _HAVE_CUPY:
        raise RuntimeError("CuPy is not available.")
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")

    code = kernel_path.read_text(encoding="utf-8")
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
    kernel_path: Path,
    cpu_only: bool,
) -> np.ndarray:
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

    kernel = load_route_kernel(kernel_path)
    N = A * R * E

    comps = {
        k: cp.asarray(np.ascontiguousarray(v.reshape(B, N).astype(np.float32)))
        for k, v in stress_batch.items()
    }
    dp = cp.empty((B, N), dtype=cp.float32)
    routes = cp.empty((B, ROUTE_STRIDE), dtype=cp.float32)

    grid = ((B + block_size - 1) // block_size,)
    block = (block_size,)

    t0 = time.perf_counter()
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
    _ = time.perf_counter() - t0

    return cp.asnumpy(routes).astype(np.float32)


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
# ABLATIONS
# --------------------------------------------------------------------------------------------------

def replace_edge_with_cell_marginal(block: np.ndarray, edge_index: int, rng: np.random.Generator) -> np.ndarray:
    f = block.copy()
    p = f[:, :, :, edge_index].mean(axis=1, keepdims=True)
    f[:, :, :, edge_index] = (rng.random(f[:, :, :, edge_index].shape) < p).astype(np.uint8)
    return f


def replace_round_with_cell_marginal(block: np.ndarray, round_index: int, rng: np.random.Generator) -> np.ndarray:
    f = block.copy()
    p = f[:, :, round_index, :].mean(axis=1, keepdims=True)
    f[:, :, round_index, :] = (rng.random(f[:, :, round_index, :].shape) < p).astype(np.uint8)
    return f


def replace_round_edge_with_cell_marginal(block: np.ndarray, round_index: int, edge_index: int, rng: np.random.Generator) -> np.ndarray:
    f = block.copy()
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


COARSE_FNS = {
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
# CLASSICAL ROUTE BASELINES
# --------------------------------------------------------------------------------------------------

def movement_cost_arrays(
    st: Dict[str, np.ndarray],
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = np.clip(
        st["tau_tau"].astype(np.float64) + coupling_weight * (st["tau_r"].astype(np.float64) + st["tau_x"].astype(np.float64)),
        min_cost,
        max_cost,
    )
    rnd = np.clip(
        st["rr"].astype(np.float64) + coupling_weight * (st["tau_r"].astype(np.float64) + st["r_x"].astype(np.float64)),
        min_cost,
        max_cost,
    )
    edge = np.clip(
        st["xx"].astype(np.float64) + coupling_weight * (st["tau_x"].astype(np.float64) + st["r_x"].astype(np.float64)),
        min_cost,
        max_cost,
    )
    return tau, rnd, edge


def route_vector_from_full_cost(
    st: Dict[str, np.ndarray],
    full_cost: float,
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> np.ndarray:
    tau, rnd, edge = movement_cost_arrays(
        st,
        coupling_weight=coupling_weight,
        min_cost=min_cost,
        max_cost=max_cost,
    )
    A, R, E = tau.shape
    cr = R // 2
    ce = E // 2
    ca = A // 2
    delay = float(np.sum(tau[1:, cr, ce])) if A > 1 else 0.0
    edge_cost = float(np.sum(edge[ca, cr, 1:])) if E > 1 else 0.0

    trace = st["tau_tau"].astype(np.float64) + st["rr"].astype(np.float64) + st["xx"].astype(np.float64)
    trace_mean = float(np.mean(trace))

    steps = max(A, R, E)
    diag = []
    for k in range(steps):
        a = min(A - 1, round(k * (A - 1) / max(steps - 1, 1)))
        r = min(R - 1, round(k * (R - 1) / max(steps - 1, 1)))
        e = min(E - 1, round(k * (E - 1) / max(steps - 1, 1)))
        diag.append(trace[a, r, e])
    path_trace_mean = float(np.mean(diag)) if diag else float("nan")
    avoidance = trace_mean - path_trace_mean

    return np.asarray([
        float(full_cost),
        delay,
        edge_cost,
        delay / (float(full_cost) + EPS),
        edge_cost / (float(full_cost) + EPS),
        trace_mean,
        avoidance,
        path_trace_mean,
    ], dtype=np.float32)


def scipy_route_from_stress(
    st: Dict[str, np.ndarray],
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> np.ndarray:
    """
    Generic sparse-graph Dijkstra baseline over the same monotonic T_S grid.

    This is intentionally more general than T_S geo_cuda and should be slower.
    It is included as an adjacent best-practice graph baseline.
    """
    if not _HAVE_SCIPY:
        return np.full((ROUTE_STRIDE,), np.nan, dtype=np.float32)

    tau, rnd, edge = movement_cost_arrays(
        st,
        coupling_weight=coupling_weight,
        min_cost=min_cost,
        max_cost=max_cost,
    )
    A, R, E = tau.shape
    N = A * R * E

    rows = []
    cols = []
    data = []

    def idx(a: int, r: int, e: int) -> int:
        return (a * R + r) * E + e

    for a in range(A):
        for r in range(R):
            for e in range(E):
                u = idx(a, r, e)
                if a + 1 < A:
                    v = idx(a + 1, r, e)
                    rows.append(u); cols.append(v); data.append(float(tau[a + 1, r, e]))
                if r + 1 < R:
                    v = idx(a, r + 1, e)
                    rows.append(u); cols.append(v); data.append(float(rnd[a, r + 1, e]))
                if e + 1 < E:
                    v = idx(a, r, e + 1)
                    rows.append(u); cols.append(v); data.append(float(edge[a, r, e + 1]))

    graph = csr_matrix((np.asarray(data, dtype=np.float64), (rows, cols)), shape=(N, N))
    dist = scipy_dijkstra(graph, directed=True, indices=0, unweighted=False)
    full = float(dist[N - 1])
    return route_vector_from_full_cost(
        st,
        full,
        coupling_weight=coupling_weight,
        min_cost=min_cost,
        max_cost=max_cost,
    )


def networkx_route_from_stress(
    st: Dict[str, np.ndarray],
    *,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
) -> np.ndarray:
    """
    NetworkX Dijkstra sanity baseline over the same monotonic T_S grid.

    Disabled by default because it is intentionally slow.
    """
    if not _HAVE_NETWORKX:
        return np.full((ROUTE_STRIDE,), np.nan, dtype=np.float32)

    tau, rnd, edge = movement_cost_arrays(
        st,
        coupling_weight=coupling_weight,
        min_cost=min_cost,
        max_cost=max_cost,
    )
    A, R, E = tau.shape

    def idx(a: int, r: int, e: int) -> int:
        return (a * R + r) * E + e

    G = nx.DiGraph()
    for a in range(A):
        for r in range(R):
            for e in range(E):
                u = idx(a, r, e)
                if a + 1 < A:
                    G.add_edge(u, idx(a + 1, r, e), weight=float(tau[a + 1, r, e]))
                if r + 1 < R:
                    G.add_edge(u, idx(a, r + 1, e), weight=float(rnd[a, r + 1, e]))
                if e + 1 < E:
                    G.add_edge(u, idx(a, r, e + 1), weight=float(edge[a, r, e + 1]))

    full = float(nx.shortest_path_length(G, 0, idx(A - 1, R - 1, E - 1), weight="weight"))
    return route_vector_from_full_cost(
        st,
        full,
        coupling_weight=coupling_weight,
        min_cost=min_cost,
        max_cost=max_cost,
    )


def route_damage_series(
    real_st: Dict[str, np.ndarray],
    ab_stress_list: List[Dict[str, np.ndarray]],
    *,
    method: str,
    real_route_cuda: Optional[np.ndarray],
    ab_routes_cuda: Optional[np.ndarray],
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    include_networkx: bool,
) -> Tuple[np.ndarray, float]:
    """
    Return damage vector [n_ablations] for one route method and total timing.

    method:
        geo_cuda
        geo_cpu_dp
        scipy_dijkstra
        networkx_dijkstra
    """
    t0 = time.perf_counter()

    if method == "geo_cuda":
        if real_route_cuda is None or ab_routes_cuda is None:
            return np.full((len(ab_stress_list),), np.nan, dtype=np.float32), 0.0
        out = np.asarray([raw_damage(real_route_cuda, r)[0] for r in ab_routes_cuda], dtype=np.float32)
        return out, float(time.perf_counter() - t0)

    if method == "geo_cpu_dp":
        real_route = cpu_raw_routes_from_stress(
            real_st,
            coupling_weight=coupling_weight,
            min_cost=min_cost,
            max_cost=max_cost,
        )
        vals = []
        for st in ab_stress_list:
            route = cpu_raw_routes_from_stress(
                st,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
            )
            vals.append(float(raw_damage(real_route, route)[0]))
        return np.asarray(vals, dtype=np.float32), float(time.perf_counter() - t0)

    if method == "scipy_dijkstra":
        if not _HAVE_SCIPY:
            return np.full((len(ab_stress_list),), np.nan, dtype=np.float32), 0.0
        real_route = scipy_route_from_stress(
            real_st,
            coupling_weight=coupling_weight,
            min_cost=min_cost,
            max_cost=max_cost,
        )
        vals = []
        for st in ab_stress_list:
            route = scipy_route_from_stress(
                st,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
            )
            vals.append(float(raw_damage(real_route, route)[0]))
        return np.asarray(vals, dtype=np.float32), float(time.perf_counter() - t0)

    if method == "networkx_dijkstra":
        if not include_networkx or not _HAVE_NETWORKX:
            return np.full((len(ab_stress_list),), np.nan, dtype=np.float32), 0.0
        real_route = networkx_route_from_stress(
            real_st,
            coupling_weight=coupling_weight,
            min_cost=min_cost,
            max_cost=max_cost,
        )
        vals = []
        for st in ab_stress_list:
            route = networkx_route_from_stress(
                st,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
            )
            vals.append(float(raw_damage(real_route, route)[0]))
        return np.asarray(vals, dtype=np.float32), float(time.perf_counter() - t0)

    raise ValueError(f"unknown route method: {method}")


def split_damage_vector_by_meta(vals: np.ndarray, ab_meta: List[Dict[str, Any]], edge_count: int, round_count: int) -> Dict[str, Any]:
    edge = np.zeros((edge_count,), dtype=np.float32)
    rnd = np.zeros((round_count,), dtype=np.float32)
    re = np.zeros((round_count, edge_count), dtype=np.float32)
    coarse: Dict[str, float] = {}

    for v, meta in zip(vals, ab_meta):
        if meta["kind"] == "edge":
            edge[int(meta["edge_index"])] = float(v)
        elif meta["kind"] == "round":
            rnd[int(meta["round_index"])] = float(v)
        elif meta["kind"] == "round_edge":
            re[int(meta["round_index"]), int(meta["edge_index"])] = float(v)
        elif meta["kind"] == "coarse":
            coarse[str(meta["control"])] = float(v)

    return {"edge": edge, "round": rnd, "round_edge": re, "coarse": coarse}


# --------------------------------------------------------------------------------------------------
# BASELINE DAMAGE SCORING
# --------------------------------------------------------------------------------------------------

def field_profiles(block: np.ndarray) -> Dict[str, np.ndarray]:
    f = block.astype(np.float64)
    return {
        "scalar": np.asarray([f.mean()], dtype=np.float64),
        "edge": f.mean(axis=(0, 1, 2)),
        "round": f.mean(axis=(0, 1, 3)),
        "delay": f.mean(axis=(1, 2, 3)),
        "round_edge": f.mean(axis=(0, 1)).ravel(),
    }


def stress_profiles(block: np.ndarray) -> Dict[str, np.ndarray]:
    st = local_stress(block)
    trace = trace_from_stress(st).astype(np.float64)
    if trace.size == 0:
        return {"scalar": np.asarray([np.nan]), "edge": np.array([]), "round": np.array([]), "round_edge": np.array([])}
    return {
        "scalar": np.asarray([trace.mean()], dtype=np.float64),
        "edge": trace.mean(axis=(0, 1)),
        "round": trace.mean(axis=(0, 2)),
        "delay": trace.mean(axis=(1, 2)),
        "round_edge": trace.mean(axis=0).ravel(),
    }


def l1_delta(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    n = min(aa.size, bb.size)
    if n == 0:
        return float("nan")
    return float(np.sum(np.abs(bb[:n] - aa[:n])))


def baseline_damage(real_block: np.ndarray, ab_block: np.ndarray, family: str) -> float:
    if family == "scalar_rate":
        return abs(float(ab_block.mean()) - float(real_block.mean()))

    if family == "field_profile_l1":
        rp = field_profiles(real_block)
        ap = field_profiles(ab_block)
        return (
            l1_delta(rp["edge"], ap["edge"])
            + l1_delta(rp["round"], ap["round"])
            + l1_delta(rp["delay"], ap["delay"])
            + l1_delta(rp["round_edge"], ap["round_edge"])
        )

    if family == "stress_profile_l1":
        rp = stress_profiles(real_block)
        ap = stress_profiles(ab_block)
        return (
            l1_delta(rp["edge"], ap["edge"])
            + l1_delta(rp["round"], ap["round"])
            + l1_delta(rp["round_edge"], ap["round_edge"])
        )

    raise ValueError(f"unknown baseline family: {family}")


# --------------------------------------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------------------------------------

def rankdata_desc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    order = np.argsort(-x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    n = min(aa.size, bb.size)
    if n < 2:
        return float("nan")
    ra = rankdata_desc(aa[:n])
    rb = rankdata_desc(bb[:n])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float(np.sum(ra * ra) * np.sum(rb * rb))) + EPS
    return float(np.sum(ra * rb) / den)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    n = min(aa.size, bb.size)
    if n == 0:
        return float("nan")
    kk = min(k, n)
    sa = set(np.argsort(-aa[:n])[:kk].tolist())
    sb = set(np.argsort(-bb[:n])[:kk].tolist())
    return float(len(sa & sb) / kk)


def top1_match(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    n = min(aa.size, bb.size)
    if n == 0:
        return float("nan")
    return float(int(np.argmax(aa[:n]) == np.argmax(bb[:n])))


def coarse_rank_dict(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {str(r["control"]): float(r["mean_damage"]) for r in rows}


def align_named_rank_corr(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]]) -> float:
    da = coarse_rank_dict(a_rows)
    db = coarse_rank_dict(b_rows)
    names = sorted(set(da) & set(db))
    if len(names) < 2:
        return float("nan")
    return spearman_corr(np.asarray([da[n] for n in names]), np.asarray([db[n] for n in names]))


# --------------------------------------------------------------------------------------------------
# BENCHMARK CORE
# --------------------------------------------------------------------------------------------------

def stress_batch_from_blocks(blocks: List[np.ndarray]) -> Dict[str, np.ndarray]:
    stresses = [local_stress(b) for b in blocks]
    keys = ("tau_tau", "rr", "xx", "tau_r", "tau_x", "r_x")
    return {k: finite_clean32(np.stack([s[k] for s in stresses], axis=0)) for k in keys}


def analyze_block(
    block: np.ndarray,
    *,
    source_label: str,
    mode_name: str,
    site_name: str,
    mode_index: int,
    site_index: int,
    seed: int,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    block_size: int,
    kernel_path: Path,
    cpu_only: bool,
    include_cpu_route: bool,
    include_networkx: bool,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    real_st = local_stress(block)
    A, R, E = real_st["tau_tau"].shape

    t_route0 = time.perf_counter()
    real_route = cuda_route_vectors(
        {k: real_st[k][None, :, :, :] for k in ("tau_tau", "rr", "xx", "tau_r", "tau_x", "r_x")},
        A=A, R=R, E=E,
        coupling_weight=coupling_weight,
        min_cost=min_cost,
        max_cost=max_cost,
        block_size=block_size,
        kernel_path=kernel_path,
        cpu_only=cpu_only,
    )[0]
    t_route1 = time.perf_counter()

    cpu_route_diff = float("nan")
    cpu_route_seconds = float("nan")
    if include_cpu_route:
        tc0 = time.perf_counter()
        cpu_route = cpu_raw_routes_from_stress(
            real_st,
            coupling_weight=coupling_weight,
            min_cost=min_cost,
            max_cost=max_cost,
        )
        tc1 = time.perf_counter()
        cpu_route_diff = float(np.max(np.abs(cpu_route.astype(np.float64) - real_route.astype(np.float64))))
        cpu_route_seconds = float(tc1 - tc0)

    edge_count = block.shape[3]
    round_count = block.shape[2]

    ab_meta: List[Dict[str, Any]] = []
    ab_blocks: List[np.ndarray] = []

    # T_S target ablations.
    for e in range(edge_count):
        rng = np.random.default_rng(seed + 100003 * mode_index + 1009 * site_index + 17 * e)
        ab_meta.append({"kind": "edge", "edge_index": e})
        ab_blocks.append(replace_edge_with_cell_marginal(block, e, rng))

    for r in range(round_count):
        rng = np.random.default_rng(seed + 200003 * mode_index + 2009 * site_index + 19 * r)
        ab_meta.append({"kind": "round", "round_index": r})
        ab_blocks.append(replace_round_with_cell_marginal(block, r, rng))

    for r in range(round_count):
        for e in range(edge_count):
            rng = np.random.default_rng(seed + 300003 * mode_index + 3001 * site_index + 101 * r + e)
            ab_meta.append({"kind": "round_edge", "round_index": r, "edge_index": e})
            ab_blocks.append(replace_round_edge_with_cell_marginal(block, r, e, rng))

    for i, (name, fn) in enumerate(COARSE_FNS.items()):
        rng = np.random.default_rng(seed + 400003 * mode_index + 4001 * site_index + i)
        ab_meta.append({"kind": "coarse", "control": name})
        ab_blocks.append(fn(block, rng))

    t_ab0 = time.perf_counter()
    ab_stress_batch = stress_batch_from_blocks(ab_blocks)
    t_ab1 = time.perf_counter()

    t_routes0 = time.perf_counter()
    ab_routes = cuda_route_vectors(
        ab_stress_batch,
        A=A, R=R, E=E,
        coupling_weight=coupling_weight,
        min_cost=min_cost,
        max_cost=max_cost,
        block_size=block_size,
        kernel_path=kernel_path,
        cpu_only=cpu_only,
    )
    t_routes1 = time.perf_counter()

    edge_damage = np.zeros((edge_count,), dtype=np.float32)
    round_damage = np.zeros((round_count,), dtype=np.float32)
    round_edge_damage = np.zeros((round_count, edge_count), dtype=np.float32)

    baseline_families = ["scalar_rate", "field_profile_l1", "stress_profile_l1"]
    baseline_edge = {b: np.zeros((edge_count,), dtype=np.float32) for b in baseline_families}
    baseline_round = {b: np.zeros((round_count,), dtype=np.float32) for b in baseline_families}
    baseline_round_edge = {b: np.zeros((round_count, edge_count), dtype=np.float32) for b in baseline_families}
    baseline_seconds = {b: 0.0 for b in baseline_families}

    edge_rows = []
    round_rows = []
    round_edge_rows = []
    coarse_rows = []
    baseline_rows = []

    # Keep stress list so classical route baselines can run on the exact same ablations.
    ab_stress_list = [{k: ab_stress_batch[k][i] for k in ("tau_tau", "rr", "xx", "tau_r", "tau_x", "r_x")}
                      for i in range(len(ab_blocks))]

    for meta, ab_block, route in zip(ab_meta, ab_blocks, ab_routes):
        dmg = raw_damage(real_route, route)
        base = {
            "source": source_label,
            "method": "geo_cuda",
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

        for fam in baseline_families:
            tb0 = time.perf_counter()
            bd = baseline_damage(block, ab_block, fam)
            baseline_seconds[fam] += time.perf_counter() - tb0
            brow = {
                "source": source_label,
                "method": fam,
                "mode": mode_name,
                "site": site_name,
                "mode_index": mode_index,
                "site_index": site_index,
                "ablation_kind": meta["kind"],
                "damage": float(bd),
            }
            if meta["kind"] == "edge":
                brow["edge_index"] = int(meta["edge_index"])
                baseline_edge[fam][int(meta["edge_index"])] = bd
            elif meta["kind"] == "round":
                brow["round_index"] = int(meta["round_index"])
                baseline_round[fam][int(meta["round_index"])] = bd
            elif meta["kind"] == "round_edge":
                brow["round_index"] = int(meta["round_index"])
                brow["edge_index"] = int(meta["edge_index"])
                baseline_round_edge[fam][int(meta["round_index"]), int(meta["edge_index"])] = bd
            elif meta["kind"] == "coarse":
                brow["control"] = str(meta["control"])
            baseline_rows.append(brow)

        if meta["kind"] == "edge":
            e = int(meta["edge_index"])
            edge_damage[e] = dmg[0]
            edge_rows.append({**base, "ablation_kind": "edge", "edge_index": e})
        elif meta["kind"] == "round":
            r = int(meta["round_index"])
            round_damage[r] = dmg[0]
            round_rows.append({**base, "ablation_kind": "round", "round_index": r})
        elif meta["kind"] == "round_edge":
            r = int(meta["round_index"])
            e = int(meta["edge_index"])
            round_edge_damage[r, e] = dmg[0]
            round_edge_rows.append({**base, "ablation_kind": "round_edge", "round_index": r, "edge_index": e})
        elif meta["kind"] == "coarse":
            coarse_rows.append({**base, "ablation_kind": "coarse", "control": str(meta["control"])})

    # Route baselines: explicit geo path and classical graph baselines.
    route_methods = ["geo_cuda", "geo_cpu_dp", "scipy_dijkstra"]
    if include_networkx:
        route_methods.append("networkx_dijkstra")

    route_method_summary = []
    route_method_rows = []

    geo_target = {
        "edge": edge_damage,
        "round": round_damage,
        "round_edge": round_edge_damage,
        "coarse": {r["control"]: r["damage"] for r in coarse_rows},
    }

    for method in route_methods:
        vals, seconds = route_damage_series(
            real_st,
            ab_stress_list,
            method=method,
            real_route_cuda=real_route,
            ab_routes_cuda=ab_routes,
            coupling_weight=coupling_weight,
            min_cost=min_cost,
            max_cost=max_cost,
            include_networkx=include_networkx,
        )
        split = split_damage_vector_by_meta(vals, ab_meta, edge_count, round_count)

        # For geo_cuda, count the actual CUDA ablation route kernel timing from above.
        # route_damage_series("geo_cuda") itself only computes damage from already-computed routes.
        method_seconds = float(t_routes1 - t_routes0) if method == "geo_cuda" else float(seconds)

        route_method_summary.append({
            "source": source_label,
            "mode": mode_name,
            "site": site_name,
            "method": method,
            "family": "route",
            "seconds": method_seconds,
            "ablations_per_second": float(len(ab_meta) / max(method_seconds, EPS)) if method_seconds > 0 else float("nan"),
            "edge_top1_match": top1_match(geo_target["edge"], split["edge"]),
            "round_top1_match": top1_match(geo_target["round"], split["round"]),
            "round_edge_top1_match": top1_match(geo_target["round_edge"], split["round_edge"]),
            "edge_spearman": spearman_corr(geo_target["edge"], split["edge"]),
            "round_spearman": spearman_corr(geo_target["round"], split["round"]),
            "round_edge_spearman": spearman_corr(geo_target["round_edge"], split["round_edge"]),
            "edge_top3_overlap": topk_overlap(geo_target["edge"], split["edge"], 3),
            "round_top3_overlap": topk_overlap(geo_target["round"], split["round"], 3),
            "round_edge_top5_overlap": topk_overlap(geo_target["round_edge"], split["round_edge"], 5),
        })

        for val, meta in zip(vals, ab_meta):
            row = {
                "source": source_label,
                "method": method,
                "family": "route",
                "mode": mode_name,
                "site": site_name,
                "mode_index": mode_index,
                "site_index": site_index,
                "ablation_kind": meta["kind"],
                "damage": float(val),
            }
            if meta["kind"] == "edge":
                row["edge_index"] = int(meta["edge_index"])
            elif meta["kind"] == "round":
                row["round_index"] = int(meta["round_index"])
            elif meta["kind"] == "round_edge":
                row["round_index"] = int(meta["round_index"])
                row["edge_index"] = int(meta["edge_index"])
            elif meta["kind"] == "coarse":
                row["control"] = str(meta["control"])
            route_method_rows.append(row)

    baseline_summary = []
    for fam in baseline_families:
        baseline_summary.append({
            "source": source_label,
            "mode": mode_name,
            "site": site_name,
            "method": fam,
            "family": "generic_profile",
            "seconds": float(baseline_seconds[fam]),
            "ablations_per_second": float(len(ab_meta) / max(baseline_seconds[fam], EPS)),
            "edge_top1_match": top1_match(edge_damage, baseline_edge[fam]),
            "round_top1_match": top1_match(round_damage, baseline_round[fam]),
            "round_edge_top1_match": top1_match(round_edge_damage, baseline_round_edge[fam]),
            "edge_spearman": spearman_corr(edge_damage, baseline_edge[fam]),
            "round_spearman": spearman_corr(round_damage, baseline_round[fam]),
            "round_edge_spearman": spearman_corr(round_edge_damage, baseline_round_edge[fam]),
            "edge_top3_overlap": topk_overlap(edge_damage, baseline_edge[fam], 3),
            "round_top3_overlap": topk_overlap(round_damage, baseline_round[fam], 3),
            "round_edge_top5_overlap": topk_overlap(round_edge_damage, baseline_round_edge[fam], 5),
        })

    baseline_summary.extend(route_method_summary)
    baseline_rows.extend(route_method_rows)

    summary = {
        "source": source_label,
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
        "source": source_label,
        "mode": mode_name,
        "site": site_name,
        "n_ablations": int(len(ab_blocks)),
        "real_route_seconds": float(t_route1 - t_route0),
        "stress_batch_seconds": float(t_ab1 - t_ab0),
        "ablation_route_seconds": float(t_routes1 - t_routes0),
        "total_block_seconds": float(time.perf_counter() - t0),
        "route_ablations_per_second": float(len(ab_blocks) / max(t_routes1 - t_routes0, EPS)),
        "total_ablations_per_second": float(len(ab_blocks) / max(time.perf_counter() - t0, EPS)),
        "cpu_route_seconds": cpu_route_seconds,
        "cpu_route_max_abs_diff": cpu_route_diff,
    }

    return {
        "summary": summary,
        "speed": speed,
        "edge_rows": edge_rows,
        "round_rows": round_rows,
        "round_edge_rows": round_edge_rows,
        "coarse_rows": coarse_rows,
        "baseline_rows": baseline_rows,
        "baseline_summary": baseline_summary,
        "edge_damage": edge_damage,
        "round_damage": round_damage,
        "round_edge_damage": round_edge_damage,
    }


def aggregate_by_index(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(int(r[key]), []).append(r)

    out = []
    for idx, vals in sorted(groups.items()):
        dmg = np.asarray([safe_float(v["damage"]) for v in vals], dtype=np.float64)
        out.append({
            key: idx,
            "mean_damage": float(np.mean(dmg)),
            "std_damage": float(np.std(dmg)),
            "max_damage": float(np.max(dmg)),
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


def analyze_source(
    ts: Dict[str, Any],
    *,
    source_label: str,
    seed: int,
    coupling_weight: float,
    min_cost: float,
    max_cost: float,
    block_size: int,
    kernel_path: Path,
    cpu_only: bool,
    include_cpu_route: bool,
    include_networkx: bool,
) -> Dict[str, Any]:
    summaries = []
    speeds = []
    edge_rows = []
    round_rows = []
    round_edge_rows = []
    coarse_rows = []
    baseline_rows = []
    baseline_summary = []

    edge_damage_blocks = []
    round_damage_blocks = []
    round_edge_damage_blocks = []

    for mi, mode_name in enumerate(ts["modes"]):
        for si, site_name in enumerate(ts["delay_sites"]):
            block = ts["field"][mi, si]
            res = analyze_block(
                block,
                source_label=source_label,
                mode_name=mode_name,
                site_name=site_name,
                mode_index=mi,
                site_index=si,
                seed=seed,
                coupling_weight=coupling_weight,
                min_cost=min_cost,
                max_cost=max_cost,
                block_size=block_size,
                kernel_path=kernel_path,
                cpu_only=cpu_only,
                include_cpu_route=include_cpu_route,
                include_networkx=include_networkx,
            )

            summaries.append(res["summary"])
            speeds.append(res["speed"])
            edge_rows.extend(res["edge_rows"])
            round_rows.extend(res["round_rows"])
            round_edge_rows.extend(res["round_edge_rows"])
            coarse_rows.extend(res["coarse_rows"])
            baseline_rows.extend(res["baseline_rows"])
            baseline_summary.extend(res["baseline_summary"])
            edge_damage_blocks.append(res["edge_damage"])
            round_damage_blocks.append(res["round_damage"])
            round_edge_damage_blocks.append(res["round_edge_damage"])

    edge_agg = aggregate_by_index(edge_rows, "edge_index")
    round_agg = aggregate_by_index(round_rows, "round_index")
    round_edge_agg = aggregate_by_pair(round_edge_rows, "round_index", "edge_index")
    coarse_agg = aggregate_by_control(coarse_rows)

    return {
        "source_label": source_label,
        "summary": summaries,
        "speed": speeds,
        "edge_rows": edge_rows,
        "round_rows": round_rows,
        "round_edge_rows": round_edge_rows,
        "coarse_rows": coarse_rows,
        "baseline_rows": baseline_rows,
        "baseline_summary": baseline_summary,
        "edge_aggregate": edge_agg,
        "round_aggregate": round_agg,
        "round_edge_aggregate": round_edge_agg,
        "coarse_aggregate": coarse_agg,
        "edge_damage_blocks": np.stack(edge_damage_blocks, axis=0),
        "round_damage_blocks": np.stack(round_damage_blocks, axis=0),
        "round_edge_damage_blocks": np.stack(round_edge_damage_blocks, axis=0),
    }


def qpu_gpu_alignment(qpu: Dict[str, Any], gpu: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []

    q_edge = np.asarray([r["mean_damage"] for r in qpu["edge_aggregate"]], dtype=np.float64)
    g_edge = np.asarray([r["mean_damage"] for r in gpu["edge_aggregate"]], dtype=np.float64)
    q_round = np.asarray([r["mean_damage"] for r in qpu["round_aggregate"]], dtype=np.float64)
    g_round = np.asarray([r["mean_damage"] for r in gpu["round_aggregate"]], dtype=np.float64)

    q_re = np.asarray([r["mean_damage"] for r in qpu["round_edge_aggregate"]], dtype=np.float64)
    g_re = np.asarray([r["mean_damage"] for r in gpu["round_edge_aggregate"]], dtype=np.float64)

    rows.append({
        "comparison": "qproj_vs_gproj",
        "component": "edge",
        "top1_match": top1_match(q_edge, g_edge),
        "spearman": spearman_corr(q_edge, g_edge),
        "top3_overlap": topk_overlap(q_edge, g_edge, 3),
        "top5_overlap": topk_overlap(q_edge, g_edge, 5),
    })
    rows.append({
        "comparison": "qproj_vs_gproj",
        "component": "round",
        "top1_match": top1_match(q_round, g_round),
        "spearman": spearman_corr(q_round, g_round),
        "top3_overlap": topk_overlap(q_round, g_round, 3),
        "top5_overlap": topk_overlap(q_round, g_round, 5),
    })
    rows.append({
        "comparison": "qproj_vs_gproj",
        "component": "round_edge",
        "top1_match": top1_match(q_re, g_re),
        "spearman": spearman_corr(q_re, g_re),
        "top3_overlap": topk_overlap(q_re, g_re, 3),
        "top5_overlap": topk_overlap(q_re, g_re, 5),
    })
    rows.append({
        "comparison": "qproj_vs_gproj",
        "component": "coarse",
        "top1_match": float("nan"),
        "spearman": align_named_rank_corr(qpu["coarse_aggregate"], gpu["coarse_aggregate"]),
        "top3_overlap": float("nan"),
        "top5_overlap": float("nan"),
    })

    return rows


# --------------------------------------------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------------------------------------------

def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """
    Write heterogeneous row dictionaries safely.

    Some benchmark rows are edge rows, some are round rows, some are round-edge
    rows, and some are coarse-control rows. Their key sets are intentionally not
    identical, so the CSV field list must be the union of all keys rather than
    the first row's keys.
    """
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "source", "source_label", "job_id", "npz", "method", "component",
        "mode", "site", "mode_index", "site_index",
        "ablation_kind", "control", "edge_index", "round_index",
        "damage", "mean_damage", "std_damage", "max_damage", "n",
        "full_delta", "delay_delta", "edge_delta", "avoidance_loss",
        "full_abs_delta", "delay_abs_delta", "edge_abs_delta",
        "full_cost", "delay_cost", "edge_cost", "trace_mean",
    ]
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    fields = [k for k in preferred if k in all_keys]
    fields.extend(sorted(k for k in all_keys if k not in fields))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def flatten_source_rows(src: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for r in src["edge_aggregate"]:
        rows.append({"source": src["source_label"], "component": "edge", **r})
    for r in src["round_aggregate"]:
        rows.append({"source": src["source_label"], "component": "round", **r})
    for r in src["round_edge_aggregate"]:
        rows.append({"source": src["source_label"], "component": "round_edge", **r})
    for r in src["coarse_aggregate"]:
        rows.append({"source": src["source_label"], "component": "coarse", **r})
    return rows


def speed_summary(source_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for src in source_results:
        speeds = src["speed"]
        total_ablations = sum(int(s["n_ablations"]) for s in speeds)
        total_seconds = sum(float(s["total_block_seconds"]) for s in speeds)
        route_seconds = sum(float(s["ablation_route_seconds"]) for s in speeds)
        stress_seconds = sum(float(s["stress_batch_seconds"]) for s in speeds)
        cpu_route_seconds = [safe_float(s["cpu_route_seconds"]) for s in speeds]
        cpu_route_seconds = [x for x in cpu_route_seconds if math.isfinite(x)]
        cpu_diffs = [safe_float(s["cpu_route_max_abs_diff"]) for s in speeds]
        cpu_diffs = [x for x in cpu_diffs if math.isfinite(x)]

        out.append({
            "source": src["source_label"],
            "total_ablations": total_ablations,
            "total_seconds": total_seconds,
            "stress_batch_seconds": stress_seconds,
            "route_seconds": route_seconds,
            "total_ablations_per_second": total_ablations / max(total_seconds, EPS),
            "route_ablations_per_second": total_ablations / max(route_seconds, EPS),
            "mean_cpu_route_seconds": float(np.mean(cpu_route_seconds)) if cpu_route_seconds else float("nan"),
            "max_cpu_route_diff": float(np.max(cpu_diffs)) if cpu_diffs else float("nan"),
        })
    return out


def plot_compare(qpu: Optional[Dict[str, Any]], gpu: Optional[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAVE_MPL:
        return
    if not qpu or not gpu:
        return

    def bar_pair(component: str, key: str, q_rows: List[Dict[str, Any]], g_rows: List[Dict[str, Any]], fname: str, title: str):
        q_vals = np.asarray([r["mean_damage"] for r in q_rows], dtype=np.float64)
        g_vals = np.asarray([r["mean_damage"] for r in g_rows], dtype=np.float64)
        n = min(q_vals.size, g_vals.size)
        labels = [str(i) for i in range(n)]
        x = np.arange(n)
        width = 0.38

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
        ax.bar(x - width / 2, q_vals[:n], width, label="qproj")
        ax.bar(x + width / 2, g_vals[:n], width, label="gproj")
        ax.set_title(title)
        ax.set_ylabel("mean raw damage")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)

    bar_pair("edge", "edge_index", qpu["edge_aggregate"], gpu["edge_aggregate"], "edge_damage_compare.png", "T_S edge damage: qproj vs gproj")
    bar_pair("round", "round_index", qpu["round_aggregate"], gpu["round_aggregate"], "round_damage_compare.png", "T_S round damage: qproj vs gproj")

    q_re = np.asarray([r["mean_damage"] for r in qpu["round_edge_aggregate"]], dtype=np.float64)
    g_re = np.asarray([r["mean_damage"] for r in gpu["round_edge_aggregate"]], dtype=np.float64)
    n = min(q_re.size, g_re.size)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=160)
    ax.scatter(q_re[:n], g_re[:n])
    ax.set_title("T_S round×edge damage agreement")
    ax.set_xlabel("qproj mean damage")
    ax.set_ylabel("gproj mean damage")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "round_edge_damage_compare.png", bbox_inches="tight")
    plt.close(fig)

    q_coarse = coarse_rank_dict(qpu["coarse_aggregate"])
    g_coarse = coarse_rank_dict(gpu["coarse_aggregate"])
    names = sorted(set(q_coarse) & set(g_coarse))
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax.bar(x - width / 2, [q_coarse[n] for n in names], width, label="qproj")
    ax.bar(x + width / 2, [g_coarse[n] for n in names], width, label="gproj")
    ax.set_title("T_S coarse damage: qproj vs gproj")
    ax.set_ylabel("mean raw damage")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out_dir / "coarse_damage_compare.png", bbox_inches="tight")
    plt.close(fig)


def print_report(
    qpu: Optional[Dict[str, Any]],
    gpu: Optional[Dict[str, Any]],
    alignment: List[Dict[str, Any]],
    speeds: List[Dict[str, Any]],
) -> None:
    print("\n" + "=" * 132)
    print("  T_S FINAL BENCHMARK SUMMARY")
    print("=" * 132)

    for src in [x for x in (qpu, gpu) if x is not None]:
        print(f"\n[{src['source_label'].upper()}]")
        print("  Top edge damages:")
        for r in sorted(src["edge_aggregate"], key=lambda x: x["mean_damage"], reverse=True)[:5]:
            print(f"    edge {r['edge_index']}: mean={r['mean_damage']:.6f}, max={r['max_damage']:.6f}")
        print("  Top round damages:")
        for r in sorted(src["round_aggregate"], key=lambda x: x["mean_damage"], reverse=True)[:5]:
            print(f"    round {r['round_index']}: mean={r['mean_damage']:.6f}, max={r['max_damage']:.6f}")
        print("  Top round×edge damages:")
        for r in sorted(src["round_edge_aggregate"], key=lambda x: x["mean_damage"], reverse=True)[:5]:
            print(
                f"    round {r['round_index']}, edge {r['edge_index']}: "
                f"mean={r['mean_damage']:.6f}, max={r['max_damage']:.6f}"
            )

    if alignment:
        print("\n[QPROJ/GPROJ ALIGNMENT]")
        for r in alignment:
            print(
                f"  {r['component']:>10}: top1={safe_float(r['top1_match']):.3f}, "
                f"spearman={safe_float(r['spearman']):.4f}, "
                f"top3={safe_float(r['top3_overlap']):.3f}, "
                f"top5={safe_float(r['top5_overlap']):.3f}"
            )

    print("\n[SPEED]")
    for s in speeds:
        print(
            f"  {s['source']:>5}: total={s['total_seconds']:.4f}s, "
            f"route={s['route_seconds']:.6f}s, "
            f"route ablations/s={s['route_ablations_per_second']:,.2f}, "
            f"total ablations/s={s['total_ablations_per_second']:,.2f}"
        )


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T_S final benchmark — geo / qproj / gproj scaffold recovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--files", nargs="+", default=None, help="Explicit set of T_S .npz files to benchmark. Overrides folder scan.")
    p.add_argument("--qpu", default=None, help="Legacy explicit QPU T_S .npz. If omitted, default scans every valid .npz in T_S/data/.")
    p.add_argument("--qpu-meta", default=None, help="QPU metadata JSON.")
    p.add_argument("--gpu", default=None, help="GPU T_S .npz. Defaults to latest_ts_gpu_data.json.")
    p.add_argument("--gpu-meta", default=None, help="GPU metadata JSON.")
    p.add_argument("--out-dir", default=None)

    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--coupling-weight", type=float, default=0.50)
    p.add_argument("--min-cost", type=float, default=1e-9)
    p.add_argument("--max-cost", type=float, default=1e9)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--include-cpu-route", action="store_true")
    p.add_argument("--include-networkx", action="store_true", help="Include slow NetworkX Dijkstra classical baseline.")
    p.add_argument("--qpu-only", action="store_true")
    p.add_argument("--gpu-only", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.qpu_only and args.gpu_only:
        sys.exit("[FATAL] Choose at most one of --qpu-only / --gpu-only.")

    kernel_path = find_kernel_path()

    if not args.cpu_only and not _HAVE_CUPY:
        sys.exit("[FATAL] CuPy/CUDA unavailable. Use --cpu-only or install CuPy.")
    if not args.cpu_only and not kernel_path.exists():
        sys.exit(f"[FATAL] Kernel not found: {kernel_path}")

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"ts_benchmark_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = collect_input_files(args)

    print("\n" + "=" * 132)
    print("  T_S FINAL BENCHMARK — GEO / QPROJ / GPROJ")
    print("=" * 132)
    print(f"  Out dir       : {out_dir}")
    print(f"  Kernel        : {kernel_path}")
    print(f"  GPU           : {gpu_info()}")
    print(f"  Data dir      : {DATA_DIR}")
    print(f"  Files found   : {len(inputs)}")
    print("  Method        : raw route/stress damage; no normalization, no cosine, no classifier")
    print(f"  Classical     : scipy_dijkstra={_HAVE_SCIPY}, networkx_dijkstra={_HAVE_NETWORKX and bool(args.include_networkx)}")
    method_line = "geo_cuda, geo_cpu_dp, scipy_dijkstra, scalar_rate, field_profile_l1, stress_profile_l1"
    if bool(args.include_networkx):
        method_line = "geo_cuda, geo_cpu_dp, scipy_dijkstra, networkx_dijkstra, scalar_rate, field_profile_l1, stress_profile_l1"
    print(f"  Methods       : {method_line}")
    for item in inputs:
        print(f"    - {item['label']}: {item['npz']}")

    if not inputs:
        sys.exit("[FATAL] No valid T_S .npz files found. Use --files <file1> <file2> or generate data first.")

    source_results = []

    for item in inputs:
        ts = load_ts(item["npz"], item["label"])
        res = analyze_source(
            ts,
            source_label=item["label"],
            seed=int(args.seed),
            coupling_weight=float(args.coupling_weight),
            min_cost=float(args.min_cost),
            max_cost=float(args.max_cost),
            block_size=int(args.block_size),
            kernel_path=kernel_path,
            cpu_only=bool(args.cpu_only),
            include_cpu_route=bool(args.include_cpu_route),
            include_networkx=bool(args.include_networkx),
        )
        res["source_kind"] = item["source"]
        res["npz"] = str(item["npz"])
        res["meta"] = str(item["meta"]) if item["meta"] else None
        res["job_id"] = item["job_id"]
        source_results.append(res)

    # Pairwise qproj/gproj alignment for every valid QPU/GPU pair.
    alignment: List[Dict[str, Any]] = []
    qprojs = [r for r in source_results if r.get("source_kind") == "qproj"]
    gprojs = [r for r in source_results if r.get("source_kind") == "gproj"]

    for q in qprojs:
        for g in gprojs:
            rows = qpu_gpu_alignment(q, g)
            for row in rows:
                row["left_source"] = q["source_label"]
                row["right_source"] = g["source_label"]
                row["comparison"] = "qproj_vs_gproj"
            alignment.extend(rows)

    speeds = speed_summary(source_results)

    benchmark_rows = []
    baseline_rows = []
    summary_rows = []
    speed_rows = []
    all_edge_rows = []
    all_round_rows = []
    all_round_edge_rows = []
    all_coarse_rows = []

    for src in source_results:
        benchmark_rows.extend(flatten_source_rows(src))
        baseline_rows.extend(src["baseline_summary"])
        summary_rows.extend(src["summary"])
        speed_rows.extend(src["speed"])
        all_edge_rows.extend(src["edge_rows"])
        all_round_rows.extend(src["round_rows"])
        all_round_edge_rows.extend(src["round_edge_rows"])
        all_coarse_rows.extend(src["coarse_rows"])

    report = {
        "schema": "ts_final_benchmark",
        "description": (
            "Final T_S benchmark comparing all valid T_S .npz files in data/ on "
            "route-preserving edge/round scaffold recovery using raw-damage metrics "
            "and enhanced CUDA kernel timing."
        ),
        "settings": {
            "seed": int(args.seed),
            "coupling_weight": float(args.coupling_weight),
            "min_cost": float(args.min_cost),
            "max_cost": float(args.max_cost),
            "block_size": int(args.block_size),
            "cpu_only": bool(args.cpu_only),
            "include_cpu_route": bool(args.include_cpu_route),
            "include_networkx": bool(args.include_networkx),
            "scipy_available": bool(_HAVE_SCIPY),
            "networkx_available": bool(_HAVE_NETWORKX),
            "normalization": "none",
            "kernel": str(kernel_path),
            "kernel_entry": "ts_raw_geo_route_vector_kernel",
            "input_mode": "explicit_files" if args.files else ("legacy_qpu_gpu" if (args.qpu or args.gpu) else "scan_all_valid_npz"),
        },
        "inputs": [
            {
                "label": item["label"],
                "source": item["source"],
                "npz": str(item["npz"]),
                "meta": str(item["meta"]) if item["meta"] else None,
            }
            for item in inputs
        ],
        "gpu": gpu_info(),
        "alignment": alignment,
        "speed_summary": speeds,
        "source_summaries": {src["source_label"]: src["summary"] for src in source_results},
        "edge_aggregate": {src["source_label"]: src["edge_aggregate"] for src in source_results},
        "round_aggregate": {src["source_label"]: src["round_aggregate"] for src in source_results},
        "round_edge_aggregate": {src["source_label"]: src["round_edge_aggregate"] for src in source_results},
        "coarse_aggregate": {src["source_label"]: src["coarse_aggregate"] for src in source_results},
        "baseline_summary": baseline_rows,
    }

    with open(out_dir / "ts_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2)

    write_csv(summary_rows, out_dir / "ts_benchmark_summary.csv")
    write_csv(benchmark_rows, out_dir / "ts_benchmark_rows.csv")
    write_csv(alignment, out_dir / "qpu_gpu_alignment.csv")
    write_csv(baseline_rows, out_dir / "baseline_comparison.csv")
    write_csv(speeds, out_dir / "benchmark_speed.csv")
    write_csv(speed_rows, out_dir / "benchmark_speed_blocks.csv")
    write_csv(all_edge_rows, out_dir / "edge_damage_rows.csv")
    write_csv(all_round_rows, out_dir / "round_damage_rows.csv")
    write_csv(all_round_edge_rows, out_dir / "round_edge_damage_rows.csv")
    write_csv(all_coarse_rows, out_dir / "coarse_damage_rows.csv")

    # Plot only the first QPU/GPU pair, if present, so output stays readable.
    if not args.no_plots and qprojs and gprojs:
        plot_compare(qprojs[0], gprojs[0], out_dir)

    # Reuse old printer for first QPU/GPU pair if present, otherwise print all source summaries.
    if qprojs or gprojs:
        print_report(qprojs[0] if qprojs else None, gprojs[0] if gprojs else None, alignment, speeds)

    print("\n[METHOD BASELINES]")
    method_order = [
        "geo_cuda",
        "geo_cpu_dp",
        "scipy_dijkstra",
        "networkx_dijkstra",
        "stress_profile_l1",
        "field_profile_l1",
        "scalar_rate",
    ]
    methods_present = set(r["method"] for r in baseline_rows)
    ordered_methods = [m for m in method_order if m in methods_present]
    ordered_methods.extend(sorted(m for m in methods_present if m not in ordered_methods))

    scipy_secs = [
        safe_float(r.get("seconds"))
        for r in baseline_rows
        if r["method"] == "scipy_dijkstra"
    ]
    scipy_secs = [v for v in scipy_secs if math.isfinite(v) and v > 0]
    scipy_mean = float(np.mean(scipy_secs)) if scipy_secs else float("nan")

    for method in ordered_methods:
        vals = [safe_float(r.get("round_edge_top1_match")) for r in baseline_rows if r["method"] == method]
        vals = [v for v in vals if math.isfinite(v)]
        spears = [safe_float(r.get("round_edge_spearman")) for r in baseline_rows if r["method"] == method]
        spears = [v for v in spears if math.isfinite(v)]
        secs = [safe_float(r.get("seconds")) for r in baseline_rows if r["method"] == method]
        secs = [v for v in secs if math.isfinite(v) and v >= 0]
        sec_mean = float(np.mean(secs)) if secs else float("nan")
        speedup = scipy_mean / sec_mean if math.isfinite(scipy_mean) and math.isfinite(sec_mean) and sec_mean > 0 else float("nan")

        print(
            f"  {method:>18}: "
            f"round_edge_top1_mean={np.mean(vals) if vals else float('nan'):.3f}, "
            f"round_edge_spearman_mean={np.mean(spears) if spears else float('nan'):.4f}, "
            f"seconds_mean={sec_mean:.6f}, "
            f"vs_scipy={speedup:.2f}x"
        )

    print("\n[ALL SOURCES]")
    for src in source_results:
        print(f"  {src['source_label']}:")
        top_edge = sorted(src["edge_aggregate"], key=lambda x: x["mean_damage"], reverse=True)[:3]
        top_round = sorted(src["round_aggregate"], key=lambda x: x["mean_damage"], reverse=True)[:3]
        print("    top edges : " + ", ".join(f"{r['edge_index']}={r['mean_damage']:.4f}" for r in top_edge))
        print("    top rounds: " + ", ".join(f"{r['round_index']}={r['mean_damage']:.4f}" for r in top_round))

    print(f"\n[SAVED] {out_dir}")
    print("=" * 132 + "\n")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("[CANCELLED]")
    except Exception as e:
        sys.exit(f"[FATAL] {type(e).__name__}: {e}")
