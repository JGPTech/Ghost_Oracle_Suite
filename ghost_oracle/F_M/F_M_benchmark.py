#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — F_M FINAL BENCHMARK
==============================================================================

Final capstone benchmark for the F_M operator.

This benchmark assumes probes are locked. It does not discover new operator
logic. It only runs the finalized F_M CUDA projector paths and compares them
against adjacent best-practice classical signal-analysis baselines.

Substrates
----------
QPROJ:
    QPU hardware record:
        g[tile, shot, bit]
        em[tile, shot, bit]

GPROJ:
    GPU synthetic/emulated record:
        g[tile, shot, bit]
        em[tile, shot, bit]

GEO:
    Optimized classical/math path:
        tile_delay_dt
        tile_scale_level
        tile_theta
        mode_id
    -> fm_geo_curve_kernel_f32
    -> fm_wave_metric_kernel_f32

Classical adjacent baselines
----------------------------
All baselines run on the same delay-ordered response curve, usually:

    xor_delta / bit_diff / delay

Baselines:
    FFT peak score      : cupy.fft.rfft
    DCT-style energy    : GPU even-extension FFT-derived DCT-II-ish score
    Autocorrelation     : direct GPU lag correlation
    SinFit grid         : GPU-ish vectorized CuPy sinusoid least-squares scan

Core F_M target
---------------
Primary locked signature:

    xor_delta / bit_diff / delay

Reference qproj target:

    score = 0.6571
    peak  = 0.769
    r2    = 0.819
    freq  = 1.30
    amp   = 0.05800

Usage
-----
    python ghost_oracle/F_M/F_M_final_benchmark.py

    python ghost_oracle/F_M/F_M_final_benchmark.py ^
      --qproj ghost_oracle/F_M/data/fm_job_d8eu8bjo3njc73evdd8g.npz ^
      --gproj ghost_oracle/F_M/data/fm_gpu_data_4096shots_seed142985762.npz

    python ghost_oracle/F_M/F_M_final_benchmark.py --skip-sweep

    python ghost_oracle/F_M/F_M_final_benchmark.py --geo-profile wide --max-candidates 1000000

Outputs
-------
analysis/fm_final_benchmark_<timestamp>/
    result.json
    substrate_signature.csv
    classical_baselines.csv
    speed_summary.csv
    geo_comparison.csv
    curve_values.csv

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

try:
    import cupy as cp
    HAVE_CUPY = True
except Exception:
    cp = None
    HAVE_CUPY = False


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
KERNEL_PATH = HERE / "kernels" / "fm_projector_kernel.cu"
ANALYSIS_DIR = HERE / "analysis"


# =============================================================================
# CONSTANTS
# =============================================================================

FIELD_KINDS = {
    "delta": 0,
    "xor_delta": 1,
    "g": 2,
    "em": 3,
}

RESPONSE_KINDS = {
    "mean": 0,
    "energy": 1,
    "transition": 2,
    "imbalance": 3,
    "bit0_mean": 4,
    "bit1_mean": 5,
    "bit_diff": 6,
}

CURVE_NAMES = [
    ("xor_delta", "bit_diff"),
    ("xor_delta", "bit1_mean"),
    ("xor_delta", "transition"),
    ("xor_delta", "energy"),
    ("delta", "bit_diff"),
    ("delta", "bit1_mean"),
    ("delta", "transition"),
    ("delta", "energy"),
]

METRIC_NAMES = [
    "wave_score",
    "peak_ratio",
    "spectral_entropy",
    "best_r2",
    "best_freq",
    "best_amp",
    "best_phase",
    "low_high_ratio",
]

PRIMARY_TARGET = {
    "field": "xor_delta",
    "response": "bit_diff",
    "order": "delay",
    "target_score": 0.6571,
    "target_peak": 0.769,
    "target_r2": 0.819,
    "target_freq": 1.30,
    "target_amp": 0.05800,
}

TARGET_ROWS = [
    {
        "field": "xor_delta",
        "response": "bit_diff",
        "order": "delay",
        "target_score": 0.6571,
        "target_peak": 0.769,
        "target_r2": 0.819,
        "target_freq": 1.30,
        "target_amp": 0.05800,
        "weight": 1.00,
    },
    {
        "field": "xor_delta",
        "response": "bit1_mean",
        "order": "delay",
        "target_score": 0.6466,
        "target_peak": 0.703,
        "target_r2": 0.985,
        "target_freq": 1.30,
        "target_amp": 0.04231,
        "weight": 0.85,
    },
    {
        "field": "xor_delta",
        "response": "transition",
        "order": "delay",
        "target_score": 0.5773,
        "target_peak": 0.868,
        "target_r2": 0.445,
        "target_freq": 2.50,
        "target_amp": 0.02188,
        "weight": 0.50,
    },
    {
        "field": "delta",
        "response": "transition",
        "order": "delay",
        "target_score": 0.5745,
        "target_peak": 0.872,
        "target_r2": 0.420,
        "target_freq": 2.50,
        "target_amp": 0.02175,
        "weight": 0.50,
    },
]


# =============================================================================
# IO HELPERS
# =============================================================================

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def section(title: str, width: int = 108) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2)


def write_csv(path: Path, rows: List[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def find_latest(substrate: str) -> Optional[Path]:
    if substrate == "qproj":
        ptr = DATA_DIR / "latest_fm_qpu_data.json"
        pattern = "fm_job_*.npz"
    elif substrate == "gproj":
        ptr = DATA_DIR / "latest_fm_gpu_data.json"
        pattern = "fm_gpu_data_*.npz"
    else:
        raise ValueError(substrate)

    if ptr.exists():
        try:
            with open(ptr, "r", encoding="utf-8") as f:
                j = json.load(f)
            p = Path(j["path"])
            if p.exists():
                return p
        except Exception:
            pass

    files = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def npz_value(z: Any, key: str) -> Any:
    v = z[key]
    try:
        return v.item() if v.shape == () else np.asarray(v).tolist()
    except Exception:
        return str(v)


def load_record_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=True) as z:
        if "g" not in z or "em" not in z:
            raise RuntimeError(f"Expected stacked g/em arrays in {path}. Keys={list(z.keys())}")

        g = np.asarray(z["g"], dtype=np.uint8)
        em = np.asarray(z["em"], dtype=np.uint8)

        meta: Dict[str, Any] = {}
        for k in [
            "schema",
            "operator",
            "substrate",
            "job_id",
            "backend",
            "shots",
            "num_tiles",
            "tile_indices",
            "tile_delay_dt",
            "tile_scale_level",
            "tile_mode",
            "tile_theta",
            "circuit_family",
        ]:
            if k in z:
                meta[k] = npz_value(z, k)

        meta["shape"] = tuple(g.shape)

    return g, em, meta


def load_metadata(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None

    with np.load(path, allow_pickle=True) as z:
        meta: Dict[str, Any] = {}
        for k in [
            "schema",
            "operator",
            "substrate",
            "job_id",
            "backend",
            "shots",
            "num_tiles",
            "tile_indices",
            "tile_delay_dt",
            "tile_scale_level",
            "tile_mode",
            "tile_theta",
            "circuit_family",
        ]:
            if k in z:
                meta[k] = npz_value(z, k)

        if "g" in z:
            meta["shape"] = tuple(np.asarray(z["g"]).shape)

    return meta


def meta_array(meta: Dict[str, Any], key: str, n: int, default: Any) -> List[Any]:
    value = meta.get(key, None)

    if value is None:
        if isinstance(default, list):
            if len(default) >= n:
                return default[:n]
            return default + [default[-1] if default else 0 for _ in range(n - len(default))]
        return [default for _ in range(n)]

    if not isinstance(value, list):
        return [value for _ in range(n)]

    if len(value) < n:
        return value + [default for _ in range(n - len(value))]

    return value[:n]


def mode_to_id(mode: str) -> int:
    if mode == "phase_shear":
        return 1
    if mode == "local_shock":
        return 2
    return 0


def build_geo_metadata(
    qproj_meta: Optional[Dict[str, Any]],
    gproj_meta: Optional[Dict[str, Any]],
    tiles: Optional[int],
) -> Dict[str, Any]:
    source = qproj_meta or gproj_meta or {}

    if "shape" in source:
        n_tiles = int(source["shape"][0])
    elif "num_tiles" in source:
        n_tiles = int(source["num_tiles"])
    else:
        n_tiles = int(tiles or 7)

    tile_indices = np.asarray(
        meta_array(source, "tile_indices", n_tiles, list(range(n_tiles))),
        dtype=np.int32,
    )
    delays = np.asarray(
        meta_array(source, "tile_delay_dt", n_tiles, [0, 1, 2, 4, 8, 16, 0]),
        dtype=np.float32,
    )
    scales = np.asarray(
        meta_array(source, "tile_scale_level", n_tiles, 1),
        dtype=np.float32,
    )
    theta = np.asarray(
        meta_array(source, "tile_theta", n_tiles, 0.5),
        dtype=np.float32,
    )
    modes = [str(v) for v in meta_array(source, "tile_mode", n_tiles, "clean")]
    mode_id = np.asarray([mode_to_id(m) for m in modes], dtype=np.int32)

    return {
        "substrate": "geo",
        "num_tiles": int(n_tiles),
        "tile_indices": tile_indices,
        "tile_delay_dt": delays,
        "tile_scale_level": scales,
        "tile_theta": theta,
        "tile_mode": modes,
        "mode_id": mode_id,
        "source_job_id": source.get("job_id", "none"),
        "source_backend": source.get("backend", "none"),
    }


# =============================================================================
# TIMING HELPERS
# =============================================================================

def cuda_event_time(fn, warmup: int = 3, reps: int = 20) -> Tuple[Any, float]:
    """
    Time GPU work using CUDA events.

    Returns:
        last_output, average_ms
    """
    for _ in range(warmup):
        out = fn()
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    out = None
    start.record()
    for _ in range(reps):
        out = fn()
    end.record()
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end) / float(reps)
    return out, float(elapsed_ms)


def cpu_time(fn, warmup: int = 2, reps: int = 10) -> Tuple[Any, float]:
    for _ in range(warmup):
        out = fn()

    t0 = time.perf_counter()
    out = None
    for _ in range(reps):
        out = fn()
    elapsed = (time.perf_counter() - t0) * 1000.0 / float(reps)
    return out, float(elapsed)


# =============================================================================
# CUDA WRAPPER
# =============================================================================

@dataclass
class GeoParams:
    wave_freq: float
    phase0: float
    bitdiff_amp: float
    bit1_amp: float
    transition_amp: float
    energy_amp: float
    scale_phase: float
    theta_phase: float
    base_xor: float
    base_delta: float


class FMCUDA:
    def __init__(self, kernel_path: Path):
        if not HAVE_CUPY:
            raise RuntimeError("CuPy required.")
        if not kernel_path.exists():
            raise FileNotFoundError(kernel_path)

        code = kernel_path.read_text(encoding="utf-8")
        self.mod = cp.RawModule(
            code=code,
            options=("--std=c++11", "--use_fast_math"),
            name_expressions=[
                "fm_response_kernel_u8",
                "fm_path_pair_break_response_kernel_u8",
                "fm_geo_curve_kernel_f32",
                "fm_geo_sweep_kernel_f32",
                "fm_wave_metric_kernel_f32",
            ],
        )

        self.response_kernel = self.mod.get_function("fm_response_kernel_u8")
        self.path_break_kernel = self.mod.get_function("fm_path_pair_break_response_kernel_u8")
        self.geo_curve_kernel = self.mod.get_function("fm_geo_curve_kernel_f32")
        self.geo_sweep_kernel = self.mod.get_function("fm_geo_sweep_kernel_f32")
        self.wave_metric_kernel = self.mod.get_function("fm_wave_metric_kernel_f32")

    def record_responses(
        self,
        g: np.ndarray,
        em: np.ndarray,
        fields: Sequence[str],
        responses: Sequence[str],
    ) -> np.ndarray:
        g_gpu = cp.asarray(np.ascontiguousarray(g, dtype=np.uint8))
        em_gpu = cp.asarray(np.ascontiguousarray(em, dtype=np.uint8))

        tiles, shots, bits = g.shape

        field_ids = cp.asarray(np.asarray([FIELD_KINDS[f] for f in fields], dtype=np.int32))
        response_ids = cp.asarray(np.asarray([RESPONSE_KINDS[r] for r in responses], dtype=np.int32))

        out = cp.zeros((len(fields), len(responses), tiles), dtype=cp.float32)

        self.response_kernel(
            (tiles, len(responses), len(fields)),
            (1,),
            (
                g_gpu,
                em_gpu,
                np.int32(tiles),
                np.int32(shots),
                np.int32(bits),
                field_ids,
                np.int32(len(fields)),
                response_ids,
                np.int32(len(responses)),
                out,
            ),
        )
        return cp.asnumpy(out)

    def record_path_break_responses(
        self,
        g: np.ndarray,
        em: np.ndarray,
        fields: Sequence[str],
        responses: Sequence[str],
        seed: int,
    ) -> np.ndarray:
        g_gpu = cp.asarray(np.ascontiguousarray(g, dtype=np.uint8))
        em_gpu = cp.asarray(np.ascontiguousarray(em, dtype=np.uint8))

        tiles, shots, bits = g.shape

        field_ids = cp.asarray(np.asarray([FIELD_KINDS[f] for f in fields], dtype=np.int32))
        response_ids = cp.asarray(np.asarray([RESPONSE_KINDS[r] for r in responses], dtype=np.int32))

        out = cp.zeros((len(fields), len(responses), tiles), dtype=cp.float32)

        self.path_break_kernel(
            (tiles, len(responses), len(fields)),
            (1,),
            (
                g_gpu,
                em_gpu,
                np.int32(tiles),
                np.int32(shots),
                np.int32(bits),
                field_ids,
                np.int32(len(fields)),
                response_ids,
                np.int32(len(responses)),
                np.int32(seed),
                out,
            ),
        )
        return cp.asnumpy(out)

    def wave_metrics(self, curves: np.ndarray, xs: np.ndarray) -> np.ndarray:
        curves_gpu = cp.asarray(np.ascontiguousarray(curves, dtype=np.float32))
        xs_gpu = cp.asarray(np.ascontiguousarray(xs, dtype=np.float32))
        out = cp.zeros((curves.shape[0], len(METRIC_NAMES)), dtype=cp.float32)

        self.wave_metric_kernel(
            (curves.shape[0],),
            (1,),
            (
                curves_gpu,
                xs_gpu,
                np.int32(curves.shape[0]),
                np.int32(curves.shape[1]),
                out,
            ),
        )
        return cp.asnumpy(out)

    def geo_curves(
        self,
        meta: Dict[str, Any],
        order_idx: np.ndarray,
        params: GeoParams,
    ) -> np.ndarray:
        delays = cp.asarray(np.asarray(meta["tile_delay_dt"], dtype=np.float32))
        scales = cp.asarray(np.asarray(meta["tile_scale_level"], dtype=np.float32))
        theta = cp.asarray(np.asarray(meta["tile_theta"], dtype=np.float32))
        mode_id = cp.asarray(np.asarray(meta["mode_id"], dtype=np.int32))
        order_gpu = cp.asarray(order_idx.astype(np.int32))

        max_delay = float(np.max(np.abs(np.asarray(meta["tile_delay_dt"], dtype=np.float32))))
        if max_delay <= 0:
            max_delay = 1.0

        n_points = int(meta["num_tiles"])
        curves = cp.zeros((len(CURVE_NAMES), n_points), dtype=cp.float32)

        self.geo_curve_kernel(
            (len(CURVE_NAMES),),
            (128,),
            (
                delays,
                scales,
                theta,
                mode_id,
                order_gpu,
                np.int32(n_points),
                np.float32(max_delay),
                np.float32(params.wave_freq),
                np.float32(params.phase0),
                np.float32(params.bitdiff_amp),
                np.float32(params.bit1_amp),
                np.float32(params.transition_amp),
                np.float32(params.energy_amp),
                np.float32(params.scale_phase),
                np.float32(params.theta_phase),
                np.float32(params.base_xor),
                np.float32(params.base_delta),
                curves,
            ),
        )
        return cp.asnumpy(curves)

    def geo_curves_and_metrics(
        self,
        meta: Dict[str, Any],
        order_idx: np.ndarray,
        xs: np.ndarray,
        params: GeoParams,
    ) -> Tuple[np.ndarray, np.ndarray]:
        delays = cp.asarray(np.asarray(meta["tile_delay_dt"], dtype=np.float32))
        scales = cp.asarray(np.asarray(meta["tile_scale_level"], dtype=np.float32))
        theta = cp.asarray(np.asarray(meta["tile_theta"], dtype=np.float32))
        mode_id = cp.asarray(np.asarray(meta["mode_id"], dtype=np.int32))
        order_gpu = cp.asarray(order_idx.astype(np.int32))
        xs_gpu = cp.asarray(xs.astype(np.float32))

        max_delay = float(np.max(np.abs(np.asarray(meta["tile_delay_dt"], dtype=np.float32))))
        if max_delay <= 0:
            max_delay = 1.0

        n_points = int(meta["num_tiles"])
        curves = cp.zeros((len(CURVE_NAMES), n_points), dtype=cp.float32)
        metrics = cp.zeros((len(CURVE_NAMES), len(METRIC_NAMES)), dtype=cp.float32)

        self.geo_curve_kernel(
            (len(CURVE_NAMES),),
            (128,),
            (
                delays,
                scales,
                theta,
                mode_id,
                order_gpu,
                np.int32(n_points),
                np.float32(max_delay),
                np.float32(params.wave_freq),
                np.float32(params.phase0),
                np.float32(params.bitdiff_amp),
                np.float32(params.bit1_amp),
                np.float32(params.transition_amp),
                np.float32(params.energy_amp),
                np.float32(params.scale_phase),
                np.float32(params.theta_phase),
                np.float32(params.base_xor),
                np.float32(params.base_delta),
                curves,
            ),
        )

        self.wave_metric_kernel(
            (len(CURVE_NAMES),),
            (1,),
            (
                curves,
                xs_gpu,
                np.int32(len(CURVE_NAMES)),
                np.int32(n_points),
                metrics,
            ),
        )

        return cp.asnumpy(curves), cp.asnumpy(metrics)

    def geo_sweep(
        self,
        meta: Dict[str, Any],
        order_idx: np.ndarray,
        xs: np.ndarray,
        candidates: np.ndarray,
        batch_size: int,
    ) -> Tuple[np.ndarray, float]:
        delays = cp.asarray(np.asarray(meta["tile_delay_dt"], dtype=np.float32))
        scales = cp.asarray(np.asarray(meta["tile_scale_level"], dtype=np.float32))
        theta = cp.asarray(np.asarray(meta["tile_theta"], dtype=np.float32))
        mode_id = cp.asarray(np.asarray(meta["mode_id"], dtype=np.int32))
        order_gpu = cp.asarray(order_idx.astype(np.int32))
        xs_gpu = cp.asarray(xs.astype(np.float32))

        max_delay = float(np.max(np.abs(np.asarray(meta["tile_delay_dt"], dtype=np.float32))))
        if max_delay <= 0:
            max_delay = 1.0

        n_points = int(meta["num_tiles"])
        n_total = int(candidates.shape[0])
        n_curves = len(CURVE_NAMES)
        n_metrics = len(METRIC_NAMES)

        all_metrics = np.empty((n_total, n_curves, n_metrics), dtype=np.float32)

        start_t = time.perf_counter()
        for offset in range(0, n_total, batch_size):
            batch = np.ascontiguousarray(candidates[offset: offset + batch_size], dtype=np.float32)
            n_batch = int(batch.shape[0])
            cand_gpu = cp.asarray(batch)

            wave_freqs = cp.ascontiguousarray(cand_gpu[:, 0])
            phase0s = cp.ascontiguousarray(cand_gpu[:, 1])
            bitdiff_amps = cp.ascontiguousarray(cand_gpu[:, 2])
            bit1_amps = cp.ascontiguousarray(cand_gpu[:, 3])
            transition_amps = cp.ascontiguousarray(cand_gpu[:, 4])
            energy_amps = cp.ascontiguousarray(cand_gpu[:, 5])
            scale_phases = cp.ascontiguousarray(cand_gpu[:, 6])
            theta_phases = cp.ascontiguousarray(cand_gpu[:, 7])
            base_xors = cp.ascontiguousarray(cand_gpu[:, 8])
            base_deltas = cp.ascontiguousarray(cand_gpu[:, 9])

            out = cp.zeros((n_batch, n_curves, n_metrics), dtype=cp.float32)

            self.geo_sweep_kernel(
                (n_batch,),
                (1,),
                (
                    delays,
                    scales,
                    theta,
                    mode_id,
                    order_gpu,
                    xs_gpu,
                    np.int32(n_points),
                    np.float32(max_delay),
                    wave_freqs,
                    phase0s,
                    bitdiff_amps,
                    bit1_amps,
                    transition_amps,
                    energy_amps,
                    scale_phases,
                    theta_phases,
                    base_xors,
                    base_deltas,
                    np.int32(n_batch),
                    out,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            all_metrics[offset: offset + n_batch] = cp.asnumpy(out)

        elapsed = time.perf_counter() - start_t
        return all_metrics, elapsed


# =============================================================================
# ORDER / CURVE HELPERS
# =============================================================================

def build_order(meta: Dict[str, Any], order: str) -> Tuple[np.ndarray, np.ndarray]:
    tile_indices = np.asarray(meta["tile_indices"], dtype=np.int32)
    delays = np.asarray(meta["tile_delay_dt"], dtype=np.float32)
    scales = np.asarray(meta["tile_scale_level"], dtype=np.float32)
    n = int(meta["num_tiles"])

    if order == "delay":
        idx = np.lexsort((tile_indices, delays)).astype(np.int32)
        xs = delays[idx].astype(np.float32)
    elif order == "tile":
        idx = np.argsort(tile_indices).astype(np.int32)
        xs = tile_indices[idx].astype(np.float32)
    elif order == "scale_delay":
        idx = np.lexsort((tile_indices, delays, scales)).astype(np.int32)
        xs = np.arange(n, dtype=np.float32)
    else:
        raise ValueError(order)

    return idx, xs


def build_curves_from_record_responses(
    response_arr: np.ndarray,
    fields: Sequence[str],
    responses: Sequence[str],
    order_idx: np.ndarray,
) -> Tuple[np.ndarray, List[dict]]:
    curves = []
    meta = []

    for fi, field in enumerate(fields):
        for ri, response in enumerate(responses):
            curves.append(response_arr[fi, ri, order_idx])
            meta.append({"field": field, "response": response})

    return np.vstack(curves).astype(np.float32), meta


def metric_to_rows(
    substrate: str,
    order: str,
    curve_meta: List[dict],
    metrics: np.ndarray,
) -> List[dict]:
    rows = []
    for i, cm in enumerate(curve_meta):
        row = {
            "substrate": substrate,
            "field": cm["field"],
            "response": cm["response"],
            "order": order,
        }
        for mi, name in enumerate(METRIC_NAMES):
            row[name] = float(metrics[i, mi])
        rows.append(row)
    return rows


def geo_metric_to_rows(
    substrate: str,
    order: str,
    metrics: np.ndarray,
) -> List[dict]:
    rows = []
    for i, (field, response) in enumerate(CURVE_NAMES):
        row = {
            "substrate": substrate,
            "field": field,
            "response": response,
            "order": order,
        }
        for mi, name in enumerate(METRIC_NAMES):
            row[name] = float(metrics[i, mi])
        rows.append(row)
    return rows


def primary_row(rows: List[dict], substrate: str) -> Optional[dict]:
    for r in rows:
        if (
            r.get("substrate") == substrate
            and r.get("field") == PRIMARY_TARGET["field"]
            and r.get("response") == PRIMARY_TARGET["response"]
            and r.get("order") == PRIMARY_TARGET["order"]
        ):
            return r
    return None


# =============================================================================
# GEO PARAMETER SWEEP
# =============================================================================

def make_candidate_grid(profile: str, max_candidates: Optional[int]) -> Tuple[np.ndarray, List[GeoParams]]:
    if profile == "small":
        wave_freqs = np.linspace(1.10, 1.45, 8)
        phase0s = np.linspace(1.80, 2.25, 10)
        bitdiff_amps = np.linspace(0.045, 0.075, 7)
        bit1_amps = np.linspace(0.032, 0.060, 6)
        transition_amps = np.linspace(0.016, 0.032, 5)
        scale_phases = np.linspace(0.00, 0.18, 5)
        theta_phases = np.linspace(0.00, 0.12, 4)

    elif profile == "wide":
        wave_freqs = np.linspace(0.70, 1.60, 19)
        phase0s = np.linspace(1.20, 2.80, 25)
        bitdiff_amps = np.linspace(0.025, 0.090, 14)
        bit1_amps = np.linspace(0.020, 0.080, 13)
        transition_amps = np.linspace(0.010, 0.040, 10)
        scale_phases = np.linspace(0.00, 0.25, 8)
        theta_phases = np.linspace(0.00, 0.18, 7)

    else:
        wave_freqs = np.linspace(0.95, 1.55, 13)
        phase0s = np.linspace(1.55, 2.50, 20)
        bitdiff_amps = np.linspace(0.035, 0.080, 10)
        bit1_amps = np.linspace(0.026, 0.068, 8)
        transition_amps = np.linspace(0.014, 0.035, 7)
        scale_phases = np.linspace(0.00, 0.22, 7)
        theta_phases = np.linspace(0.00, 0.15, 6)

    energy_amps = np.asarray([0.010, 0.014, 0.018], dtype=np.float32)

    rows: List[GeoParams] = []
    for wf in wave_freqs:
        for ph in phase0s:
            for bda in bitdiff_amps:
                for b1a in bit1_amps:
                    for tra in transition_amps:
                        for ena in energy_amps:
                            for scp in scale_phases:
                                for thp in theta_phases:
                                    rows.append(
                                        GeoParams(
                                            wave_freq=float(wf),
                                            phase0=float(ph),
                                            bitdiff_amp=float(bda),
                                            bit1_amp=float(b1a),
                                            transition_amp=float(tra),
                                            energy_amp=float(ena),
                                            scale_phase=float(scp),
                                            theta_phase=float(thp),
                                            base_xor=0.0,
                                            base_delta=0.0,
                                        )
                                    )

    if max_candidates is not None and max_candidates > 0 and len(rows) > max_candidates:
        idx = np.linspace(0, len(rows) - 1, max_candidates).round().astype(np.int64)
        rows = [rows[int(i)] for i in idx]

    arr = np.asarray(
        [
            [
                p.wave_freq,
                p.phase0,
                p.bitdiff_amp,
                p.bit1_amp,
                p.transition_amp,
                p.energy_amp,
                p.scale_phase,
                p.theta_phase,
                p.base_xor,
                p.base_delta,
            ]
            for p in rows
        ],
        dtype=np.float32,
    )

    return arr, rows


def loss_vector(metrics: np.ndarray) -> np.ndarray:
    """
    metrics shape:
        (n_candidates, 8 curves, 8 metrics)
    """
    loss = np.zeros(metrics.shape[0], dtype=np.float32)

    curve_lookup = {
        (field, response, "delay"): i
        for i, (field, response) in enumerate(CURVE_NAMES)
    }

    for t in TARGET_ROWS:
        idx = curve_lookup[(t["field"], t["response"], t["order"])]

        score = metrics[:, idx, 0]
        peak = metrics[:, idx, 1]
        r2 = metrics[:, idx, 3]
        freq = metrics[:, idx, 4]
        amp = metrics[:, idx, 5]
        w = np.float32(t["weight"])

        loss += w * (
            np.float32(2.00) * (score - np.float32(t["target_score"])) ** 2
            + np.float32(1.00) * (peak - np.float32(t["target_peak"])) ** 2
            + np.float32(0.60) * (r2 - np.float32(t["target_r2"])) ** 2
            + np.float32(0.15) * (freq - np.float32(t["target_freq"])) ** 2
            + np.float32(0.50) * (amp - np.float32(t["target_amp"])) ** 2
        )

    return loss.astype(np.float32)


def target_comparison_rows(rows: List[dict], substrate: str) -> List[dict]:
    row_map = {
        (r["field"], r["response"], r["order"]): r
        for r in rows
        if r["substrate"] == substrate
    }

    out = []
    for t in TARGET_ROWS:
        key = (t["field"], t["response"], t["order"])
        r = row_map.get(key)
        if r is None:
            continue

        out.append({
            "substrate": substrate,
            "field": t["field"],
            "response": t["response"],
            "order": t["order"],
            "score": r["wave_score"],
            "target_score": t["target_score"],
            "score_error": r["wave_score"] - t["target_score"],
            "peak": r["peak_ratio"],
            "target_peak": t["target_peak"],
            "peak_error": r["peak_ratio"] - t["target_peak"],
            "r2": r["best_r2"],
            "target_r2": t["target_r2"],
            "r2_error": r["best_r2"] - t["target_r2"],
            "freq": r["best_freq"],
            "target_freq": t["target_freq"],
            "freq_error": r["best_freq"] - t["target_freq"],
            "amp": r["best_amp"],
            "target_amp": t["target_amp"],
            "amp_error": r["best_amp"] - t["target_amp"],
            "weight": t["weight"],
        })

    return out


# =============================================================================
# CLASSICAL BASELINES
# =============================================================================

def baseline_fft_gpu(curve: np.ndarray) -> Dict[str, float]:
    y = cp.asarray(np.asarray(curve, dtype=np.float32))
    y = y - cp.mean(y)

    spec = cp.fft.rfft(y)
    power = cp.abs(spec) ** 2

    if power.size <= 1:
        return {"baseline_score": 0.0, "peak_ratio": 0.0, "peak_index": 0.0}

    p = power[1:]
    total = cp.sum(p) + cp.float32(1e-12)
    peak = cp.max(p)
    idx = cp.argmax(p) + 1
    ratio = peak / total

    return {
        "baseline_score": float(ratio.get()),
        "peak_ratio": float(ratio.get()),
        "peak_index": float(idx.get()),
    }


def baseline_dct_gpu(curve: np.ndarray) -> Dict[str, float]:
    """
    DCT-II-ish GPU baseline via even extension + FFT.

    This is intentionally GPU-resident and dependency-light.
    """
    y = cp.asarray(np.asarray(curve, dtype=np.float32))
    y = y - cp.mean(y)
    n = int(y.size)

    if n <= 1:
        return {"baseline_score": 0.0, "dct_peak_ratio": 0.0, "dct_peak_index": 0.0}

    ext = cp.concatenate([y, y[::-1]])
    fft = cp.fft.fft(ext)
    k = cp.arange(n, dtype=cp.float32)
    phase = cp.exp(-1j * cp.pi * k / (2.0 * n))
    coeff = cp.real(fft[:n] * phase)

    energy = coeff[1:] ** 2 if n > 1 else coeff ** 2
    total = cp.sum(energy) + cp.float32(1e-12)
    peak = cp.max(energy)
    idx = cp.argmax(energy) + 1
    ratio = peak / total

    return {
        "baseline_score": float(ratio.get()),
        "dct_peak_ratio": float(ratio.get()),
        "dct_peak_index": float(idx.get()),
    }


def baseline_autocorr_gpu(curve: np.ndarray) -> Dict[str, float]:
    y = cp.asarray(np.asarray(curve, dtype=np.float32))
    y = y - cp.mean(y)
    denom = cp.sum(y * y) + cp.float32(1e-12)

    vals = []
    for lag in range(1, int(y.size)):
        v = cp.sum(y[:-lag] * y[lag:]) / denom
        vals.append(v)

    if not vals:
        return {"baseline_score": 0.0, "max_abs_autocorr": 0.0, "best_lag": 0.0}

    arr = cp.stack(vals)
    idx = cp.argmax(cp.abs(arr))
    score = cp.abs(arr[idx])

    return {
        "baseline_score": float(score.get()),
        "max_abs_autocorr": float(score.get()),
        "best_lag": float((idx + 1).get()),
    }


def baseline_sinfit_gpu(curve: np.ndarray, xs: np.ndarray) -> Dict[str, float]:
    """
    Vectorized GPU sinusoid fit over fixed frequency grid.

    Fits:
        y ≈ a sin(2*pi*f*x_norm) + b cos(2*pi*f*x_norm) + c

    Uses cupy.linalg.lstsq per frequency. Small-N, but still GPU-resident.
    """
    y = cp.asarray(np.asarray(curve, dtype=np.float32))
    x = cp.asarray(np.asarray(xs, dtype=np.float32))

    if y.size < 4:
        return {"baseline_score": 0.0, "best_r2": 0.0, "best_freq": 0.0, "best_amp": 0.0}

    xmin = cp.min(x)
    xmax = cp.max(x)
    xnorm = cp.where(xmax > xmin, (x - xmin) / (xmax - xmin), cp.arange(y.size, dtype=cp.float32) / max(1, y.size - 1))

    ymean = cp.mean(y)
    ss_tot = cp.sum((y - ymean) ** 2) + cp.float32(1e-12)

    best_r2 = cp.asarray(-1e20, dtype=cp.float32)
    best_freq = cp.asarray(0.0, dtype=cp.float32)
    best_amp = cp.asarray(0.0, dtype=cp.float32)

    freqs = cp.linspace(0.5, 3.0, 26, dtype=cp.float32)

    for f in freqs:
        w = cp.float32(2.0 * math.pi) * f
        A = cp.stack([
            cp.sin(w * xnorm),
            cp.cos(w * xnorm),
            cp.ones_like(xnorm),
        ], axis=1)

        coef, *_ = cp.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = cp.sum((y - pred) ** 2)
        r2 = cp.float32(1.0) - ss_res / ss_tot
        amp = cp.sqrt(coef[0] * coef[0] + coef[1] * coef[1])

        valid = cp.isfinite(r2) & cp.isfinite(amp) & (amp <= 10.0)
        if bool(valid.get()) and float(r2.get()) > float(best_r2.get()):
            best_r2 = r2
            best_freq = f
            best_amp = amp

    if float(best_r2.get()) < 0.0:
        best_r2 = cp.asarray(0.0, dtype=cp.float32)

    return {
        "baseline_score": float(best_r2.get()),
        "best_r2": float(best_r2.get()),
        "best_freq": float(best_freq.get()),
        "best_amp": float(best_amp.get()),
    }


def run_classical_baselines(curve: np.ndarray, xs: np.ndarray, reps: int) -> Tuple[List[dict], List[dict]]:
    baselines = [
        ("FFT_GPU", lambda: baseline_fft_gpu(curve)),
        ("DCT_GPU", lambda: baseline_dct_gpu(curve)),
        ("AUTOCORR_GPU", lambda: baseline_autocorr_gpu(curve)),
        ("SINFIT_GPU", lambda: baseline_sinfit_gpu(curve, xs)),
    ]

    rows = []
    speed_rows = []

    for name, fn in baselines:
        out, ms = cuda_event_time(fn, warmup=3, reps=reps)
        row = {
            "backend": name,
            "curve": "xor_delta/bit_diff/delay",
            **out,
        }
        rows.append(row)

        speed_rows.append({
            "path": name,
            "operation": "classical_baseline",
            "time_ms": ms,
            "reps": reps,
        })

    return rows, speed_rows


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_record_substrate(
    cuda: FMCUDA,
    substrate: str,
    path: Path,
    order_name: str,
    fields: Sequence[str],
    responses: Sequence[str],
    reps: int,
) -> Tuple[List[dict], List[dict], List[dict], np.ndarray, np.ndarray]:
    g, em, meta = load_record_npz(path)
    order_idx, xs = build_order_for_record(meta, g.shape[0], order_name)

    def response_fn():
        return cuda.record_responses(g, em, fields, responses)

    response_arr, response_ms = cuda_event_time(response_fn, warmup=3, reps=reps)

    curves, curve_meta = build_curves_from_record_responses(response_arr, fields, responses, order_idx)

    def metric_fn():
        return cuda.wave_metrics(curves, xs)

    metrics, metric_ms = cuda_event_time(metric_fn, warmup=3, reps=reps)

    rows = metric_to_rows(substrate, order_name, curve_meta, metrics)

    speed_rows = [
        {
            "path": substrate,
            "operation": "record_response_kernel",
            "time_ms": response_ms,
            "reps": reps,
        },
        {
            "path": substrate,
            "operation": "wave_metric_kernel",
            "time_ms": metric_ms,
            "reps": reps,
        },
        {
            "path": substrate,
            "operation": "record_response_plus_metric",
            "time_ms": response_ms + metric_ms,
            "reps": reps,
        },
    ]

    curve_rows = []
    for ci, cm in enumerate(curve_meta):
        for p in range(curves.shape[1]):
            curve_rows.append({
                "substrate": substrate,
                "field": cm["field"],
                "response": cm["response"],
                "order": order_name,
                "point": int(p),
                "x": float(xs[p]),
                "y": float(curves[ci, p]),
            })

    return rows, speed_rows, curve_rows, curves, xs


def build_order_for_record(meta: Dict[str, Any], n_tiles: int, order: str) -> Tuple[np.ndarray, np.ndarray]:
    tile_indices = np.asarray(meta_array(meta, "tile_indices", n_tiles, list(range(n_tiles))), dtype=np.int32)
    delays = np.asarray(meta_array(meta, "tile_delay_dt", n_tiles, [0, 1, 2, 4, 8, 16, 0]), dtype=np.float32)
    scales = np.asarray(meta_array(meta, "tile_scale_level", n_tiles, 1), dtype=np.float32)

    if order == "delay":
        idx = np.lexsort((tile_indices, delays)).astype(np.int32)
        xs = delays[idx].astype(np.float32)
    elif order == "tile":
        idx = np.argsort(tile_indices).astype(np.int32)
        xs = tile_indices[idx].astype(np.float32)
    elif order == "scale_delay":
        idx = np.lexsort((tile_indices, delays, scales)).astype(np.int32)
        xs = np.arange(n_tiles, dtype=np.float32)
    else:
        raise ValueError(order)

    return idx, xs


def run_geo_substrate(
    cuda: FMCUDA,
    meta: Dict[str, Any],
    params: GeoParams,
    order_name: str,
    reps: int,
) -> Tuple[List[dict], List[dict], List[dict], np.ndarray, np.ndarray]:
    order_idx, xs = build_order(meta, order_name)

    def geo_fn():
        return cuda.geo_curves_and_metrics(meta, order_idx, xs, params)

    (curves, metrics), geo_ms = cuda_event_time(geo_fn, warmup=3, reps=reps)

    rows = geo_metric_to_rows("GEO", order_name, metrics)

    speed_rows = [
        {
            "path": "GEO",
            "operation": "geo_curve_plus_metric",
            "time_ms": geo_ms,
            "reps": reps,
        },
    ]

    curve_rows = []
    for ci, (field, response) in enumerate(CURVE_NAMES):
        for p in range(curves.shape[1]):
            curve_rows.append({
                "substrate": "GEO",
                "field": field,
                "response": response,
                "order": order_name,
                "point": int(p),
                "x": float(xs[p]),
                "y": float(curves[ci, p]),
            })

    return rows, speed_rows, curve_rows, curves, xs


def run_geo_sweep(
    cuda: FMCUDA,
    meta: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[GeoParams, List[dict], List[dict], float]:
    if args.skip_sweep:
        params = GeoParams(
            wave_freq=args.wave_freq,
            phase0=args.phase0,
            bitdiff_amp=args.bitdiff_amp,
            bit1_amp=args.bit1_amp,
            transition_amp=args.transition_amp,
            energy_amp=args.energy_amp,
            scale_phase=args.scale_phase,
            theta_phase=args.theta_phase,
            base_xor=0.0,
            base_delta=0.0,
        )
        return params, [], [], 0.0

    order_idx, xs = build_order(meta, "delay")

    candidates, param_rows = make_candidate_grid(args.geo_profile, args.max_candidates)

    metrics, sweep_seconds = cuda.geo_sweep(
        meta=meta,
        order_idx=order_idx,
        xs=xs,
        candidates=candidates,
        batch_size=args.batch_size,
    )

    losses = loss_vector(metrics)
    order = np.argsort(losses)
    best_i = int(order[0])
    best_params = param_rows[best_i]

    sweep_top = []
    for rank, idx in enumerate(order[: min(100, len(order))]):
        p = param_rows[int(idx)]
        sweep_top.append({
            "rank": int(rank),
            **asdict(p),
            "loss": float(losses[int(idx)]),
        })

    speed_rows = [
        {
            "path": "GEO",
            "operation": "geo_cuda_sweep",
            "time_ms": float(sweep_seconds * 1000.0),
            "seconds": float(sweep_seconds),
            "candidates": int(candidates.shape[0]),
            "candidates_per_second": float(candidates.shape[0] / max(sweep_seconds, 1e-12)),
            "profile": args.geo_profile,
        }
    ]

    return best_params, sweep_top, speed_rows, sweep_seconds


def print_signature_table(rows: List[dict], title: str, limit: int = 20) -> None:
    section(title)
    sorted_rows = sorted(rows, key=lambda r: r["wave_score"], reverse=True)
    for r in sorted_rows[:limit]:
        print(
            f"  {r['substrate']:<7s} "
            f"{r['field']:10s} {r['response']:12s} order={r['order']:11s} "
            f"score={r['wave_score']:7.4f} peak={r['peak_ratio']:6.3f} "
            f"r2={r['best_r2']:6.3f} freq={r['best_freq']:5.2f} "
            f"amp={r['best_amp']:8.5f}"
        )


def print_speed_table(speed_rows: List[dict]) -> None:
    section("SPEED SUMMARY")
    print(f"  {'path':<16} {'operation':<32} {'time_ms':>12} {'extra':>20}")
    print("  " + "-" * 84)
    for r in speed_rows:
        extra = ""
        if "candidates_per_second" in r:
            extra = f"{r['candidates_per_second'] / 1e6:.2f}M cand/s"
        elif "reps" in r:
            extra = f"reps={r['reps']}"
        print(f"  {r['path']:<16} {r['operation']:<32} {r['time_ms']:>12.6f} {extra:>20}")


def print_classical_table(rows: List[dict]) -> None:
    section("CLASSICAL ADJACENT BASELINES")
    print(f"  {'backend':<16} {'score':>10} {'details':>42}")
    print("  " + "-" * 72)
    for r in rows:
        details = ", ".join(
            f"{k}={v:.4f}" for k, v in r.items()
            if k not in ("backend", "curve", "baseline_score") and isinstance(v, (float, int))
        )
        print(f"  {r['backend']:<16} {r['baseline_score']:>10.4f} {details:>42}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F_M final benchmark: qproj/gproj/geo projector signature + speed comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--qproj", default=None, help="F_M qproj .npz. Defaults to latest.")
    p.add_argument("--gproj", default=None, help="F_M gproj .npz. Defaults to latest.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tiles", type=int, default=None)

    p.add_argument("--kernel", default=str(KERNEL_PATH))
    p.add_argument("--reps", type=int, default=50)

    p.add_argument("--skip-qproj", action="store_true")
    p.add_argument("--skip-gproj", action="store_true")
    p.add_argument("--skip-geo", action="store_true")
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--skip-classical", action="store_true")

    p.add_argument("--geo-profile", default="default", choices=["small", "default", "wide"])
    p.add_argument("--max-candidates", type=int, default=250000)
    p.add_argument("--batch-size", type=int, default=65536)

    p.add_argument("--wave-freq", type=float, default=1.30)
    p.add_argument("--phase0", type=float, default=2.02)
    p.add_argument("--bitdiff-amp", type=float, default=0.058)
    p.add_argument("--bit1-amp", type=float, default=0.042)
    p.add_argument("--transition-amp", type=float, default=0.022)
    p.add_argument("--energy-amp", type=float, default=0.014)
    p.add_argument("--scale-phase", type=float, default=0.13)
    p.add_argument("--theta-phase", type=float, default=0.05)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not HAVE_CUPY:
        raise RuntimeError("CuPy is required for F_M final benchmark.")

    qproj_path = Path(args.qproj) if args.qproj else find_latest("qproj")
    gproj_path = Path(args.gproj) if args.gproj else find_latest("gproj")

    if not args.skip_qproj and (qproj_path is None or not qproj_path.exists()):
        raise FileNotFoundError("No qproj file found. Use --skip-qproj or pass --qproj.")
    if not args.skip_gproj and (gproj_path is None or not gproj_path.exists()):
        raise FileNotFoundError("No gproj file found. Use --skip-gproj or pass --gproj.")

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"fm_final_benchmark_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        gpu_name = "unknown-gpu"

    section("GHOST ORACLE SUITE — F_M FINAL BENCHMARK")
    print(f"  qproj          : {qproj_path}")
    print(f"  gproj          : {gproj_path}")
    print(f"  kernel         : {args.kernel}")
    print(f"  out_dir        : {out_dir}")
    print(f"  GPU            : {gpu_name}")
    print(f"  reps           : {args.reps}")
    print(f"  geo_profile    : {args.geo_profile}")
    print(f"  max_candidates : {args.max_candidates}")
    print(f"  batch_size     : {args.batch_size}")

    cuda = FMCUDA(Path(args.kernel))

    fields = ["xor_delta", "delta"]
    responses = ["bit_diff", "bit1_mean", "transition", "energy"]
    order_name = "delay"

    all_signature_rows: List[dict] = []
    all_speed_rows: List[dict] = []
    all_curve_rows: List[dict] = []
    classical_rows: List[dict] = []
    comparison_rows_all: List[dict] = []
    sweep_top: List[dict] = []
    best_geo_params: Optional[GeoParams] = None

    primary_curve = None
    primary_xs = None

    if not args.skip_qproj:
        rows, speed, curves_out, curves, xs = run_record_substrate(
            cuda, "QPROJ", qproj_path, order_name, fields, responses, args.reps
        )
        all_signature_rows.extend(rows)
        all_speed_rows.extend(speed)
        all_curve_rows.extend(curves_out)

        # Primary baseline curve from QPROJ if available.
        primary_curve = curves[0]
        primary_xs = xs

    if not args.skip_gproj:
        rows, speed, curves_out, curves, xs = run_record_substrate(
            cuda, "GPROJ", gproj_path, order_name, fields, responses, args.reps
        )
        all_signature_rows.extend(rows)
        all_speed_rows.extend(speed)
        all_curve_rows.extend(curves_out)

        if primary_curve is None:
            primary_curve = curves[0]
            primary_xs = xs

    qproj_meta = load_metadata(qproj_path) if qproj_path and qproj_path.exists() else None
    gproj_meta = load_metadata(gproj_path) if gproj_path and gproj_path.exists() else None
    geo_meta = build_geo_metadata(qproj_meta, gproj_meta, args.tiles)

    if not args.skip_geo:
        best_geo_params, sweep_top, sweep_speed_rows, _ = run_geo_sweep(cuda, geo_meta, args)
        all_speed_rows.extend(sweep_speed_rows)

        rows, speed, curves_out, geo_curves, geo_xs = run_geo_substrate(
            cuda, geo_meta, best_geo_params, order_name, args.reps
        )
        all_signature_rows.extend(rows)
        all_speed_rows.extend(speed)
        all_curve_rows.extend(curves_out)

        comparison_rows_all.extend(target_comparison_rows(all_signature_rows, "GEO"))

        if primary_curve is None:
            primary_curve = geo_curves[0]
            primary_xs = geo_xs

    # Comparison rows for qproj/gproj too.
    for sub in ["QPROJ", "GPROJ"]:
        comparison_rows_all.extend(target_comparison_rows(all_signature_rows, sub))

    if not args.skip_classical and primary_curve is not None and primary_xs is not None:
        classical_rows, classical_speed = run_classical_baselines(primary_curve, primary_xs, args.reps)
        all_speed_rows.extend(classical_speed)

    # Sort signature rows for display/save.
    all_signature_rows_sorted = sorted(all_signature_rows, key=lambda r: r["wave_score"], reverse=True)

    print_signature_table(all_signature_rows_sorted, "FINAL PROJECTOR SIGNATURE")
    print_classical_table(classical_rows)
    print_speed_table(all_speed_rows)

    section("PRIMARY TARGET SUMMARY")
    for substrate in ["QPROJ", "GPROJ", "GEO"]:
        r = primary_row(all_signature_rows, substrate)
        if r is None:
            continue
        print(
            f"  {substrate:<7s} "
            f"score={r['wave_score']:.4f} "
            f"target={PRIMARY_TARGET['target_score']:.4f} "
            f"err={r['wave_score'] - PRIMARY_TARGET['target_score']:+.4f} "
            f"peak={r['peak_ratio']:.3f} "
            f"r2={r['best_r2']:.3f} "
            f"freq={r['best_freq']:.2f} "
            f"amp={r['best_amp']:.5f}"
        )

    # Save outputs.
    write_csv(
        out_dir / "substrate_signature.csv",
        all_signature_rows_sorted,
        [
            "substrate",
            "field",
            "response",
            "order",
            "wave_score",
            "peak_ratio",
            "spectral_entropy",
            "best_r2",
            "best_freq",
            "best_amp",
            "best_phase",
            "low_high_ratio",
        ],
    )

    write_csv(
        out_dir / "classical_baselines.csv",
        classical_rows,
        sorted(set().union(*(r.keys() for r in classical_rows))) if classical_rows else ["backend"],
    )

    write_csv(
        out_dir / "speed_summary.csv",
        all_speed_rows,
        sorted(set().union(*(r.keys() for r in all_speed_rows))) if all_speed_rows else ["path"],
    )

    write_csv(
        out_dir / "geo_comparison.csv",
        comparison_rows_all,
        [
            "substrate",
            "field",
            "response",
            "order",
            "score",
            "target_score",
            "score_error",
            "peak",
            "target_peak",
            "peak_error",
            "r2",
            "target_r2",
            "r2_error",
            "freq",
            "target_freq",
            "freq_error",
            "amp",
            "target_amp",
            "amp_error",
            "weight",
        ],
    )

    write_csv(
        out_dir / "curve_values.csv",
        all_curve_rows,
        [
            "substrate",
            "field",
            "response",
            "order",
            "point",
            "x",
            "y",
        ],
    )

    write_csv(
        out_dir / "geo_sweep_top.csv",
        sweep_top,
        [
            "rank",
            "wave_freq",
            "phase0",
            "bitdiff_amp",
            "bit1_amp",
            "transition_amp",
            "energy_amp",
            "scale_phase",
            "theta_phase",
            "base_xor",
            "base_delta",
            "loss",
        ],
    )

    result = {
        "benchmark": "F_M Final Benchmark",
        "qproj_path": str(qproj_path) if qproj_path else None,
        "gproj_path": str(gproj_path) if gproj_path else None,
        "kernel": str(args.kernel),
        "out_dir": str(out_dir),
        "gpu": gpu_name,
        "config": vars(args),
        "best_geo_params": asdict(best_geo_params) if best_geo_params else None,
        "signature": all_signature_rows_sorted,
        "classical_baselines": classical_rows,
        "speed_summary": all_speed_rows,
        "target_comparison": comparison_rows_all,
        "geo_sweep_top": sweep_top,
    }
    write_json(out_dir / "result.json", result)

    section("DONE")
    print(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()