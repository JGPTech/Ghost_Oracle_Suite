#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
F_M PROBE 06 — GEO CUDA FINALIZER
==============================================================================

Purpose
-------
Final optimized CUDA geo path for F_M.

This probe performs the full GPU-side classical/geo workflow:

    1. Load qproj/gproj metadata.
    2. Build ordered tile metadata arrays.
    3. Generate a large parameter sweep on CPU as compact candidate arrays.
    4. Run fm_geo_sweep_kernel_f32 on GPU.
    5. Score candidates against the discovered qproj/gproj target signature.
    6. Select best params.
    7. Run fm_geo_curve_kernel_f32 for final geo curves.
    8. Run fm_wave_metric_kernel_f32 for final wave metrics.
    9. Save final optimized geo signature.

This is the optimized classical math path, not a sampled shot path.

Known target signature from qproj/gproj
---------------------------------------
Primary:
    xor_delta / bit_diff / delay

Runners-up:
    xor_delta / bit1_mean / delay
    xor_delta / transition / delay
    delta     / transition / delay

Expected output:
    geo_signature.csv
    geo_comparison.csv
    geo_curve_values.csv
    geo_sweep_top.csv
    result.json

Usage
-----
    python ghost_oracle/F_M/probes/f_m_probe06_geo_cuda_finalizer.py

or explicit:

    python ghost_oracle/F_M/probes/f_m_probe06_geo_cuda_finalizer.py ^
      --qproj ghost_oracle/F_M/data/fm_job_d8eu8bjo3njc73evdd8g.npz ^
      --gproj ghost_oracle/F_M/data/fm_gpu_data_4096shots_seed142985762.npz

Notes
-----
The sweep is GPU-accelerated but still uses one block per candidate and small-N
serial math inside each block. That is fine here because each candidate only
evaluates 8 curves over ~7 points. The win is eliminating Python-loop scoring.

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
FM_DIR = HERE.parent
DATA_DIR = FM_DIR / "data"
KERNEL_PATH = FM_DIR / "kernels" / "fm_projector_kernel.cu"
ANALYSIS_DIR = HERE / "analysis"


# =============================================================================
# GEO CURVE / METRIC CONSTANTS
# =============================================================================

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

# Target from current qproj discovery.
# These are intentionally qproj-anchored, not overfit to gproj.
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
# SMALL HELPERS
# =============================================================================

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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2)


def write_csv(path: Path, rows: List[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


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


def load_metadata(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None

    with np.load(path, allow_pickle=True) as z:
        meta: Dict[str, Any] = {}
        for key in [
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
            if key in z:
                meta[key] = npz_value(z, key)

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


# =============================================================================
# PARAMETER GRID
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


def make_candidate_grid(
    profile: str,
    max_candidates: Optional[int] = None,
) -> Tuple[np.ndarray, List[GeoParams]]:
    """
    Create candidate matrix shape (n_candidates, 10).

    The default profile is centered around the successful NumPy default:
        wave_freq=1.3, phase0=2.02, bitdiff_amp=0.058, ...
    """
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
        # balanced/default
        wave_freqs = np.linspace(0.95, 1.55, 13)
        phase0s = np.linspace(1.55, 2.50, 20)
        bitdiff_amps = np.linspace(0.035, 0.080, 10)
        bit1_amps = np.linspace(0.026, 0.068, 8)
        transition_amps = np.linspace(0.014, 0.035, 7)
        scale_phases = np.linspace(0.00, 0.22, 7)
        theta_phases = np.linspace(0.00, 0.15, 6)

    # Keep energy amplitude small; it was already close in NumPy default.
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
        # Deterministic downsample across the full grid.
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


# =============================================================================
# CUDA WRAPPER
# =============================================================================

class FMGeoCUDA:
    def __init__(self, kernel_path: Path):
        if not HAVE_CUPY:
            raise RuntimeError("CuPy required. Install cupy-cuda12x or matching CuPy build.")
        if not kernel_path.exists():
            raise FileNotFoundError(kernel_path)

        code = kernel_path.read_text(encoding="utf-8")
        self.mod = cp.RawModule(
            code=code,
            options=("--std=c++11", "--use_fast_math"),
            name_expressions=[
                "fm_geo_curve_kernel_f32",
                "fm_geo_sweep_kernel_f32",
                "fm_wave_metric_kernel_f32",
            ],
        )

        self.geo_curve_kernel = self.mod.get_function("fm_geo_curve_kernel_f32")
        self.geo_sweep_kernel = self.mod.get_function("fm_geo_sweep_kernel_f32")
        self.wave_metric_kernel = self.mod.get_function("fm_wave_metric_kernel_f32")

    def sweep(
        self,
        meta: Dict[str, Any],
        order: str,
        candidates: np.ndarray,
        batch_size: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Run GPU geo sweep.

        Returns:
            metrics shape (n_candidates, 8 curves, 8 metrics)
            elapsed_seconds
        """
        delays = cp.asarray(np.asarray(meta["tile_delay_dt"], dtype=np.float32))
        scales = cp.asarray(np.asarray(meta["tile_scale_level"], dtype=np.float32))
        theta = cp.asarray(np.asarray(meta["tile_theta"], dtype=np.float32))
        mode_id = cp.asarray(np.asarray(meta["mode_id"], dtype=np.int32))

        order_idx, xs = build_order(meta, order)
        order_gpu = cp.asarray(order_idx.astype(np.int32))
        xs_gpu = cp.asarray(xs.astype(np.float32))

        max_delay = float(np.max(np.abs(np.asarray(meta["tile_delay_dt"], dtype=np.float32))))
        if max_delay <= 0:
            max_delay = 1.0

        n_points = int(meta["num_tiles"])
        n_total = int(candidates.shape[0])
        n_curves = 8
        n_metrics = 8

        all_metrics = np.empty((n_total, n_curves, n_metrics), dtype=np.float32)

        start = time.perf_counter()

        for offset in range(0, n_total, batch_size):
            batch = np.ascontiguousarray(candidates[offset: offset + batch_size], dtype=np.float32)
            n_batch = batch.shape[0]

            cand_gpu = cp.asarray(batch)

            # Column views copied as contiguous arrays because kernels expect one array per param.
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

        elapsed = time.perf_counter() - start
        return all_metrics, elapsed

    def compute_geo_curves_and_metrics(
        self,
        meta: Dict[str, Any],
        order: str,
        params: GeoParams,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict], float]:
        delays = cp.asarray(np.asarray(meta["tile_delay_dt"], dtype=np.float32))
        scales = cp.asarray(np.asarray(meta["tile_scale_level"], dtype=np.float32))
        theta = cp.asarray(np.asarray(meta["tile_theta"], dtype=np.float32))
        mode_id = cp.asarray(np.asarray(meta["mode_id"], dtype=np.int32))

        order_idx, xs = build_order(meta, order)
        order_gpu = cp.asarray(order_idx.astype(np.int32))
        xs_gpu = cp.asarray(xs.astype(np.float32))

        max_delay = float(np.max(np.abs(np.asarray(meta["tile_delay_dt"], dtype=np.float32))))
        if max_delay <= 0:
            max_delay = 1.0

        n_points = int(meta["num_tiles"])
        n_curves = 8
        n_metrics = 8

        curves_gpu = cp.zeros((n_curves, n_points), dtype=cp.float32)
        metrics_gpu = cp.zeros((n_curves, n_metrics), dtype=cp.float32)

        start = time.perf_counter()

        self.geo_curve_kernel(
            (n_curves,),
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
                curves_gpu,
            ),
        )

        self.wave_metric_kernel(
            (n_curves,),
            (1,),
            (
                curves_gpu,
                xs_gpu,
                np.int32(n_curves),
                np.int32(n_points),
                metrics_gpu,
            ),
        )

        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start

        curves = cp.asnumpy(curves_gpu)
        metrics = cp.asnumpy(metrics_gpu)

        point_meta = []
        tile_indices = np.asarray(meta["tile_indices"], dtype=np.int32)
        delays_np = np.asarray(meta["tile_delay_dt"], dtype=np.float32)
        scales_np = np.asarray(meta["tile_scale_level"], dtype=np.float32)
        modes = list(meta["tile_mode"])

        for p, tile in enumerate(order_idx):
            point_meta.append({
                "point": int(p),
                "tile_index": int(tile_indices[tile]),
                "delay_dt": float(delays_np[tile]),
                "scale_level": float(scales_np[tile]),
                "mode": str(modes[tile]),
                "x": float(xs[p]),
            })

        return curves, metrics, xs, point_meta, elapsed


# =============================================================================
# SCORING
# =============================================================================

def curve_key(curve_index: int, order: str) -> Tuple[str, str, str]:
    field, response = CURVE_NAMES[curve_index]
    return field, response, order


def metric_row(curve_index: int, order: str, metrics: np.ndarray) -> dict:
    field, response = CURVE_NAMES[curve_index]
    row = {
        "field": field,
        "response": response,
        "order": order,
    }
    for i, name in enumerate(METRIC_NAMES):
        row[name] = float(metrics[curve_index, i])
    return row


def target_loss(metrics: np.ndarray, order: str) -> float:
    if order != "delay":
        # Targets are currently defined on delay order only.
        return 0.0

    row_map = {
        curve_key(i, order): metrics[i]
        for i in range(len(CURVE_NAMES))
    }

    loss = 0.0

    for t in TARGET_ROWS:
        key = (t["field"], t["response"], t["order"])
        if key not in row_map:
            loss += 999.0
            continue

        m = row_map[key]
        score = float(m[0])
        peak = float(m[1])
        r2 = float(m[3])
        freq = float(m[4])
        amp = float(m[5])
        w = float(t["weight"])

        loss += w * (
            2.00 * (score - t["target_score"]) ** 2
            + 1.00 * (peak - t["target_peak"]) ** 2
            + 0.60 * (r2 - t["target_r2"]) ** 2
            + 0.15 * (freq - t["target_freq"]) ** 2
            + 0.50 * (amp - t["target_amp"]) ** 2
        )

    return float(loss)


def loss_vector(metrics: np.ndarray, order: str) -> np.ndarray:
    """
    metrics shape:
        (n_candidates, 8, 8)
    """
    if order != "delay":
        return np.zeros(metrics.shape[0], dtype=np.float32)

    loss = np.zeros(metrics.shape[0], dtype=np.float32)

    curve_lookup = {
        (field, response, order): i
        for i, (field, response) in enumerate(CURVE_NAMES)
    }

    for t in TARGET_ROWS:
        key = (t["field"], t["response"], t["order"])
        idx = curve_lookup[key]

        score = metrics[:, idx, 0]
        peak = metrics[:, idx, 1]
        r2 = metrics[:, idx, 3]
        freq = metrics[:, idx, 4]
        amp = metrics[:, idx, 5]

        w = float(t["weight"])

        loss += np.float32(w) * (
            np.float32(2.00) * (score - np.float32(t["target_score"])) ** 2
            + np.float32(1.00) * (peak - np.float32(t["target_peak"])) ** 2
            + np.float32(0.60) * (r2 - np.float32(t["target_r2"])) ** 2
            + np.float32(0.15) * (freq - np.float32(t["target_freq"])) ** 2
            + np.float32(0.50) * (amp - np.float32(t["target_amp"])) ** 2
        )

    return loss.astype(np.float32)


def comparison_rows(metrics: np.ndarray, order: str) -> List[dict]:
    rows = []
    curve_lookup = {
        (field, response, order): i
        for i, (field, response) in enumerate(CURVE_NAMES)
    }

    for t in TARGET_ROWS:
        key = (t["field"], t["response"], t["order"])
        if key not in curve_lookup:
            continue

        idx = curve_lookup[key]
        m = metrics[idx]

        rows.append({
            "field": t["field"],
            "response": t["response"],
            "order": t["order"],
            "geo_score": float(m[0]),
            "target_score": float(t["target_score"]),
            "score_error": float(m[0] - t["target_score"]),
            "geo_peak": float(m[1]),
            "target_peak": float(t["target_peak"]),
            "peak_error": float(m[1] - t["target_peak"]),
            "geo_r2": float(m[3]),
            "target_r2": float(t["target_r2"]),
            "r2_error": float(m[3] - t["target_r2"]),
            "geo_freq": float(m[4]),
            "target_freq": float(t["target_freq"]),
            "freq_error": float(m[4] - t["target_freq"]),
            "geo_amp": float(m[5]),
            "target_amp": float(t["target_amp"]),
            "amp_error": float(m[5] - t["target_amp"]),
            "weight": float(t["weight"]),
        })

    return rows


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F_M Probe 06: CUDA geo finalizer and optimized sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--qproj", default=None, help="F_M qproj .npz. Defaults to latest.")
    p.add_argument("--gproj", default=None, help="F_M gproj .npz. Defaults to latest if available.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tiles", type=int, default=None)

    p.add_argument("--profile", default="default", choices=["small", "default", "wide"])
    p.add_argument("--max-candidates", type=int, default=250000)
    p.add_argument("--batch-size", type=int, default=65536)

    p.add_argument("--no-sweep", action="store_true", help="Use default params without GPU sweep.")

    # Defaults from NumPy probe.
    p.add_argument("--wave-freq", type=float, default=1.30)
    p.add_argument("--phase0", type=float, default=2.02)
    p.add_argument("--bitdiff-amp", type=float, default=0.058)
    p.add_argument("--bit1-amp", type=float, default=0.042)
    p.add_argument("--transition-amp", type=float, default=0.022)
    p.add_argument("--energy-amp", type=float, default=0.014)
    p.add_argument("--scale-phase", type=float, default=0.13)
    p.add_argument("--theta-phase", type=float, default=0.05)

    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    if not HAVE_CUPY:
        raise RuntimeError("CuPy is required.")

    qproj_path = Path(args.qproj) if args.qproj else find_latest("qproj")
    gproj_path = Path(args.gproj) if args.gproj else find_latest("gproj")

    qproj_meta = load_metadata(qproj_path)
    gproj_meta = load_metadata(gproj_path)

    meta = build_geo_metadata(qproj_meta, gproj_meta, args.tiles)

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"fm_probe06_geo_cuda_finalizer_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        gpu_name = "unknown-gpu"

    print("=" * 108)
    print("  F_M PROBE 06 — GEO CUDA FINALIZER")
    print("=" * 108)
    print(f"  qproj          : {qproj_path}")
    print(f"  gproj          : {gproj_path}")
    print(f"  out_dir        : {out_dir}")
    print(f"  GPU            : {gpu_name}")
    print(f"  tiles          : {meta['num_tiles']}")
    print(f"  delays         : {np.asarray(meta['tile_delay_dt']).tolist()}")
    print(f"  scales         : {np.asarray(meta['tile_scale_level']).tolist()}")
    print(f"  profile        : {args.profile}")
    print(f"  max_candidates : {args.max_candidates}")
    print(f"  batch_size     : {args.batch_size}")
    print("-" * 108)

    cuda = FMGeoCUDA(KERNEL_PATH)

    if args.no_sweep:
        best_params = GeoParams(
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
        sweep_top_rows = [dict(**asdict(best_params), loss=np.nan, rank=0)]
        sweep_seconds = 0.0
        n_candidates = 1
        print("[GEO] Using provided/default params, no sweep.")
    else:
        print("[GEO] Building candidate grid...")
        candidates, param_rows = make_candidate_grid(args.profile, max_candidates=args.max_candidates)
        n_candidates = int(candidates.shape[0])
        print(f"[GEO] Candidate count: {n_candidates:,}")

        print("[GEO] Running CUDA geo sweep...")
        metrics, sweep_seconds = cuda.sweep(
            meta=meta,
            order="delay",
            candidates=candidates,
            batch_size=args.batch_size,
        )

        losses = loss_vector(metrics, order="delay")
        order = np.argsort(losses)
        best_i = int(order[0])
        best_params = param_rows[best_i]

        sweep_top_rows = []
        for rank, idx in enumerate(order[: min(100, len(order))]):
            p = param_rows[int(idx)]
            sweep_top_rows.append({
                "rank": int(rank),
                **asdict(p),
                "loss": float(losses[int(idx)]),
            })

        print(f"[GEO] Sweep seconds: {sweep_seconds:.6f}")
        print(f"[GEO] Best loss    : {float(losses[best_i]):.6f}")

    # Final curves and metrics for all orders.
    all_signature_rows: List[dict] = []
    all_curve_rows: List[dict] = []
    all_comparison: List[dict] = []
    final_timing: Dict[str, float] = {}

    for order_name in ["delay", "tile", "scale_delay"]:
        curves, metrics, xs, point_meta, elapsed = cuda.compute_geo_curves_and_metrics(
            meta=meta,
            order=order_name,
            params=best_params,
        )
        final_timing[f"{order_name}_seconds"] = float(elapsed)

        for curve_idx in range(len(CURVE_NAMES)):
            row = metric_row(curve_idx, order_name, metrics)
            all_signature_rows.append(row)

            field, response = CURVE_NAMES[curve_idx]
            for p in range(curves.shape[1]):
                all_curve_rows.append({
                    "field": field,
                    "response": response,
                    "order": order_name,
                    "curve_index": int(curve_idx),
                    "point": int(p),
                    "x": float(xs[p]),
                    "y": float(curves[curve_idx, p]),
                    **point_meta[p],
                })

        if order_name == "delay":
            all_comparison.extend(comparison_rows(metrics, order_name))

    all_signature_rows.sort(key=lambda r: r["wave_score"], reverse=True)

    # Save files.
    write_csv(
        out_dir / "geo_signature.csv",
        all_signature_rows,
        [
            "field", "response", "order",
            "wave_score", "peak_ratio", "spectral_entropy",
            "best_r2", "best_freq", "best_amp", "best_phase", "low_high_ratio",
        ],
    )

    write_csv(
        out_dir / "geo_comparison.csv",
        all_comparison,
        [
            "field", "response", "order",
            "geo_score", "target_score", "score_error",
            "geo_peak", "target_peak", "peak_error",
            "geo_r2", "target_r2", "r2_error",
            "geo_freq", "target_freq", "freq_error",
            "geo_amp", "target_amp", "amp_error",
            "weight",
        ],
    )

    write_csv(
        out_dir / "geo_curve_values.csv",
        all_curve_rows,
        [
            "field", "response", "order", "curve_index",
            "point", "x", "y", "tile_index", "delay_dt", "scale_level", "mode",
        ],
    )

    write_csv(
        out_dir / "geo_sweep_top.csv",
        sweep_top_rows,
        [
            "rank",
            "wave_freq", "phase0", "bitdiff_amp", "bit1_amp",
            "transition_amp", "energy_amp", "scale_phase",
            "theta_phase", "base_xor", "base_delta", "loss",
        ],
    )

    result = {
        "probe": "F_M Probe 06 — Geo CUDA Finalizer",
        "qproj_path": str(qproj_path) if qproj_path else None,
        "gproj_path": str(gproj_path) if gproj_path else None,
        "out_dir": str(out_dir),
        "gpu": gpu_name,
        "metadata": json_safe(meta),
        "config": {
            "profile": args.profile,
            "max_candidates": int(args.max_candidates),
            "batch_size": int(args.batch_size),
            "no_sweep": bool(args.no_sweep),
        },
        "best_params": asdict(best_params),
        "sweep_seconds": float(sweep_seconds),
        "n_candidates": int(n_candidates),
        "final_timing": final_timing,
        "geo_signature": all_signature_rows,
        "geo_comparison": all_comparison,
        "sweep_top": sweep_top_rows,
    }
    write_json(out_dir / "result.json", result)

    print("\n" + "=" * 108)
    print("  BEST GEO CUDA PARAMS")
    print("=" * 108)
    for k, v in asdict(best_params).items():
        print(f"  {k:16s}: {v}")

    print("\n" + "=" * 108)
    print("  TOP GEO CUDA SIGNATURE")
    print("=" * 108)
    for r in all_signature_rows[:16]:
        print(
            f"  {r['field']:10s} {r['response']:12s} order={r['order']:11s} "
            f"score={r['wave_score']:7.4f} peak={r['peak_ratio']:6.3f} "
            f"r2={r['best_r2']:6.3f} freq={r['best_freq']:5.2f} "
            f"amp={r['best_amp']:8.5f}"
        )

    print("\n" + "=" * 108)
    print("  TARGET COMPARISON")
    print("=" * 108)
    for r in all_comparison:
        print(
            f"  {r['field']:10s} {r['response']:12s} order={r['order']:11s} "
            f"geo={r['geo_score']:7.4f} target={r['target_score']:7.4f} "
            f"err={r['score_error']:+8.4f} "
            f"freq={r['geo_freq']:5.2f}/{r['target_freq']:5.2f} "
            f"amp={r['geo_amp']:8.5f}/{r['target_amp']:8.5f}"
        )

    print("\n" + "=" * 108)
    print("  TIMING")
    print("=" * 108)
    print(f"  candidates     : {n_candidates:,}")
    print(f"  sweep seconds  : {sweep_seconds:.6f}")
    for k, v in final_timing.items():
        print(f"  {k:16s}: {v:.6f}")
    print(f"  output dir     : {out_dir}")
    print("=" * 108)


if __name__ == "__main__":
    main()