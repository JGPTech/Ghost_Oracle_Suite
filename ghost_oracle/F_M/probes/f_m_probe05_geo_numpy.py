#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
F_M PROBE 05 — GEO NUMPY PATH DISCOVERY
==============================================================================

Purpose
-------
Build the first pure classical/geo path for F_M.

qproj:
    Hardware record from the QPU.

gproj:
    GPU-generated paired-path record that preserves the discovered qproj
    projector signature.

geo:
    Minimal classical math path. No hardware shots required. No synthetic
    g/em sampling required unless explicitly requested later.

This probe constructs analytic response curves directly from tile metadata:

    delay_dt
    scale_level
    theta
    mode

and asks whether a compact wave formula can reproduce the observed F_M
projector signature:

    primary:
        xor_delta / bit_diff / delay

    runners-up:
        xor_delta / bit1_mean / delay
        delta     / transition / delay
        delta     / bit_diff / delay

This is a NumPy discovery probe. Once the formula is good, the next step is
to add the optimized geo path to fm_projector_kernel.cu.

Usage
-----
    python ghost_oracle/F_M/probes/f_m_probe05_geo_numpy.py

or:

    python ghost_oracle/F_M/probes/f_m_probe05_geo_numpy.py ^
      --qproj ghost_oracle/F_M/data/fm_job_d8eu8bjo3njc73evdd8g.npz ^
      --gproj ghost_oracle/F_M/data/fm_gpu_data_4096shots_seed142985762.npz

Outputs
-------
analysis/fm_probe05_geo_numpy_<timestamp>/
    result.json
    geo_signature.csv
    comparison.csv
    curve_values.csv
    sweep.csv

Interpretation
--------------
A good first geo path should reproduce the ordering and approximate magnitude
of the discovered projector signature:

    xor_delta / bit_diff / delay should be near the top.
    delay_shuffle should weaken the wave score.
    geo should be closer to qproj/gproj than naive flat/no-wave baselines.

==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
FM_DIR = HERE.parent
DATA_DIR = FM_DIR / "data"
ANALYSIS_DIR = HERE / "analysis"


# =============================================================================
# TARGET SIGNATURE FROM CURRENT QPROJ/GPROJ DISCOVERY
# =============================================================================

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


# =============================================================================
# METADATA LOADING
# =============================================================================

def _read_npz_scalar_or_array(z: Any, key: str) -> Any:
    v = z[key]
    try:
        return v.item() if v.shape == () else np.asarray(v).tolist()
    except Exception:
        return str(v)


def load_metadata(path: Path) -> Dict[str, Any]:
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
                meta[k] = _read_npz_scalar_or_array(z, k)

        if "g" in z:
            meta["shape"] = tuple(np.asarray(z["g"]).shape)

    return meta


def meta_array(meta: Dict[str, Any], key: str, n: int, default: Any) -> List[Any]:
    v = meta.get(key, None)
    if v is None:
        return [default for _ in range(n)]
    if not isinstance(v, list):
        return [v for _ in range(n)]
    if len(v) < n:
        return v + [default for _ in range(n - len(v))]
    return v[:n]


def build_geo_metadata(
    qproj_meta: Optional[Dict[str, Any]],
    gproj_meta: Optional[Dict[str, Any]],
    tiles: Optional[int],
) -> Dict[str, Any]:
    """
    Prefer qproj metadata, then gproj metadata, then defaults.
    """
    source = qproj_meta or gproj_meta or {}

    if "shape" in source:
        n_tiles = int(source["shape"][0])
    elif "num_tiles" in source:
        n_tiles = int(source["num_tiles"])
    else:
        n_tiles = int(tiles or 7)

    tile_indices = np.asarray(meta_array(source, "tile_indices", n_tiles, list(range(n_tiles))), dtype=np.int32)
    delays = np.asarray(meta_array(source, "tile_delay_dt", n_tiles, [0, 1, 2, 4, 8, 16, 0]), dtype=np.float64)
    scales = np.asarray(meta_array(source, "tile_scale_level", n_tiles, 1), dtype=np.float64)
    theta = np.asarray(meta_array(source, "tile_theta", n_tiles, 0.5), dtype=np.float64)
    modes = [str(v) for v in meta_array(source, "tile_mode", n_tiles, "clean")]

    return {
        "substrate": "geo",
        "num_tiles": n_tiles,
        "tile_indices": tile_indices,
        "tile_delay_dt": delays,
        "tile_scale_level": scales,
        "tile_theta": theta,
        "tile_mode": modes,
        "source_job_id": source.get("job_id", "none"),
        "source_backend": source.get("backend", "none"),
    }


# =============================================================================
# WAVE METRICS — MATCH PROJECTOR KERNEL LOGIC
# =============================================================================

def demean(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return y - float(np.mean(y))


def spectral_metrics(y: np.ndarray) -> Dict[str, float]:
    yy = demean(y)
    n = int(yy.size)

    if n < 3 or float(np.std(yy)) < 1e-12:
        return {
            "n": n,
            "peak_index": 0,
            "peak_freq": 0.0,
            "peak_power": 0.0,
            "total_power": 0.0,
            "peak_ratio": 0.0,
            "spectral_entropy": 0.0,
            "dominant_phase": 0.0,
            "low_high_ratio": 0.0,
        }

    fft = np.fft.rfft(yy)
    power = np.abs(fft) ** 2

    p_non = power[1:] if power.size > 1 else power
    offset = 1 if power.size > 1 else 0

    total = float(np.sum(p_non) + 1e-12)
    peak_local = int(np.argmax(p_non))
    peak_idx = int(peak_local + offset)
    peak_power = float(p_non[peak_local])
    peak_ratio = float(peak_power / total)

    probs = p_non / total
    probs = probs[probs > 1e-12]
    sent = float(-np.sum(probs * np.log(probs)) / math.log(max(2, p_non.size)))

    dom_phase = float(np.angle(fft[peak_idx]))
    peak_freq = float(peak_idx / max(1, n))

    half = max(1, p_non.size // 2)
    low = float(np.sum(p_non[:half]))
    high = float(np.sum(p_non[half:]))
    low_high = float(low / (high + 1e-12))

    return {
        "n": n,
        "peak_index": peak_idx,
        "peak_freq": peak_freq,
        "peak_power": peak_power,
        "total_power": total,
        "peak_ratio": peak_ratio,
        "spectral_entropy": sent,
        "dominant_phase": dom_phase,
        "low_high_ratio": low_high,
    }


def sinusoid_fit_metrics(xs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(xs, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)

    if yy.size < 4 or np.std(yy) < 1e-12:
        return {
            "best_r2": 0.0,
            "best_freq": 0.0,
            "best_amp": 0.0,
            "best_phase": 0.0,
        }

    xx = x.copy()
    if np.max(xx) > np.min(xx):
        xx = (xx - np.min(xx)) / (np.max(xx) - np.min(xx))
    else:
        xx = np.arange(yy.size, dtype=np.float64) / max(1, yy.size - 1)

    ymean = float(np.mean(yy))
    ss_tot = float(np.sum((yy - ymean) ** 2) + 1e-12)

    best = {
        "best_r2": -np.inf,
        "best_freq": 0.0,
        "best_amp": 0.0,
        "best_phase": 0.0,
    }

    for freq in np.linspace(0.5, 3.0, 26):
        w = 2.0 * np.pi * freq
        A = np.column_stack([
            np.sin(w * xx),
            np.cos(w * xx),
            np.ones_like(xx),
        ])

        try:
            coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        except Exception:
            continue

        pred = A @ coef
        ss_res = float(np.sum((yy - pred) ** 2))
        r2 = float(1.0 - ss_res / ss_tot)
        amp = float(math.sqrt(coef[0] ** 2 + coef[1] ** 2))
        phase = float(math.atan2(coef[1], coef[0]))

        if not math.isfinite(amp) or amp > 10.0 or not math.isfinite(r2):
            continue

        if r2 > best["best_r2"]:
            best = {
                "best_r2": r2,
                "best_freq": float(freq),
                "best_amp": amp,
                "best_phase": phase,
            }

    if not math.isfinite(best["best_r2"]) or best["best_r2"] < 0.0:
        best = {
            "best_r2": 0.0,
            "best_freq": 0.0,
            "best_amp": 0.0,
            "best_phase": 0.0,
        }

    return best


def wave_score(xs: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
    spec = spectral_metrics(y)
    fit = sinusoid_fit_metrics(xs, y)

    score = (
        0.40 * spec["peak_ratio"]
        + 0.25 * max(0.0, fit["best_r2"])
        + 0.20 * (1.0 - spec["spectral_entropy"])
        + 0.15 * min(1.0, abs(spec["low_high_ratio"]) / 10.0)
    )

    out = {}
    out.update(spec)
    out.update(fit)
    out["wave_score"] = float(score)
    return float(score), out


# =============================================================================
# GEO FORMULA
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


def geo_curves(meta: Dict[str, Any], params: GeoParams) -> Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Return analytic curves keyed by:

        (field, response, order)

    Minimal F_M geo idea:

        delay_norm = delay / max(delay)

        xor_delta.bit_diff(delay)
            = base + A * sin(2*pi*f*delay_norm + phase + scale/theta terms)

        xor_delta.bit1_mean(delay)
            = base + A * sin(2*pi*f*delay_norm + phase + offset)

        delta.transition(delay)
            = base + A * sin(2*pi*f2*delay_norm + phase2)

    This is not a sampled record. It is the clean math path.
    """
    delays = np.asarray(meta["tile_delay_dt"], dtype=np.float64)
    scales = np.asarray(meta["tile_scale_level"], dtype=np.float64)
    theta = np.asarray(meta["tile_theta"], dtype=np.float64)
    tile_indices = np.asarray(meta["tile_indices"], dtype=np.float64)

    n = delays.size
    if n == 0:
        raise ValueError("no tiles")

    max_delay = float(np.max(np.abs(delays)))
    delay_norm = delays / (max_delay if max_delay > 0 else 1.0)

    scale_term = params.scale_phase * np.log2(np.maximum(1.0, scales) + 1.0)
    theta_term = params.theta_phase * theta

    # Core phase field.
    phi = 2.0 * np.pi * params.wave_freq * delay_norm + params.phase0 + scale_term + theta_term

    # Secondary phase for transition-like response.
    phi2 = 2.0 * np.pi * (params.wave_freq + 1.1) * delay_norm + params.phase0 * 0.5 + scale_term

    curves_raw: Dict[Tuple[str, str], np.ndarray] = {
        ("xor_delta", "bit_diff"): (
            params.base_xor
            + params.bitdiff_amp * np.sin(phi)
        ),
        ("xor_delta", "bit1_mean"): (
            0.11
            + params.bit1_amp * np.sin(phi + 0.19)
        ),
        ("xor_delta", "transition"): (
            0.50
            + params.transition_amp * np.sin(phi2)
        ),
        ("xor_delta", "energy"): (
            0.12
            + params.energy_amp * np.sin(phi2 + 0.33)
        ),
        ("delta", "bit_diff"): (
            params.base_delta
            + 0.78 * params.bitdiff_amp * np.sin(phi + 0.34)
        ),
        ("delta", "bit1_mean"): (
            0.02
            + 0.70 * params.bit1_amp * np.sin(phi + 0.52)
        ),
        ("delta", "transition"): (
            0.50
            + 1.00 * params.transition_amp * np.sin(phi2 + 0.06)
        ),
        ("delta", "energy"): (
            0.12
            + params.energy_amp * np.sin(phi2 + 0.33)
        ),
    }

    out: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]] = {}

    for order in ["delay", "tile", "scale_delay"]:
        if order == "delay":
            idx = np.lexsort((tile_indices, delays))
            xs = delays[idx]
        elif order == "tile":
            idx = np.argsort(tile_indices)
            xs = tile_indices[idx]
        elif order == "scale_delay":
            idx = np.lexsort((tile_indices, delays, scales))
            xs = np.arange(n, dtype=np.float64)
        else:
            raise ValueError(order)

        for key, y in curves_raw.items():
            out[(key[0], key[1], order)] = (xs.astype(np.float64), y[idx].astype(np.float64))

    return out


# =============================================================================
# SCORING
# =============================================================================

def score_geo(meta: Dict[str, Any], params: GeoParams) -> Tuple[List[dict], float]:
    curves = geo_curves(meta, params)

    rows: List[dict] = []
    loss = 0.0

    for (field, response, order), (xs, y) in curves.items():
        score, metrics = wave_score(xs, y)

        row = {
            "field": field,
            "response": response,
            "order": order,
            "wave_score": float(score),
            "peak_ratio": float(metrics["peak_ratio"]),
            "spectral_entropy": float(metrics["spectral_entropy"]),
            "best_r2": float(metrics["best_r2"]),
            "best_freq": float(metrics["best_freq"]),
            "best_amp": float(metrics["best_amp"]),
            "best_phase": float(metrics["best_phase"]),
            "low_high_ratio": float(metrics["low_high_ratio"]),
        }
        rows.append(row)

    # Loss against qproj/gproj discovered targets.
    row_map = {(r["field"], r["response"], r["order"]): r for r in rows}

    for target in TARGET_ROWS:
        key = (target["field"], target["response"], target["order"])
        if key not in row_map:
            loss += 999.0
            continue

        r = row_map[key]
        w = float(target["weight"])

        # Weighted, forgiving loss. We care more about ordering/shape than exact.
        loss += w * (
            2.00 * (r["wave_score"] - target["target_score"]) ** 2
            + 1.00 * (r["peak_ratio"] - target["target_peak"]) ** 2
            + 0.60 * (r["best_r2"] - target["target_r2"]) ** 2
            + 0.15 * (r["best_freq"] - target["target_freq"]) ** 2
            + 0.50 * (r["best_amp"] - target["target_amp"]) ** 2
        )

    return rows, float(loss)


def target_comparison(rows: List[dict]) -> List[dict]:
    row_map = {(r["field"], r["response"], r["order"]): r for r in rows}
    out = []

    for target in TARGET_ROWS:
        key = (target["field"], target["response"], target["order"])
        r = row_map.get(key)
        if r is None:
            continue

        out.append({
            "field": target["field"],
            "response": target["response"],
            "order": target["order"],
            "geo_score": r["wave_score"],
            "target_score": target["target_score"],
            "score_error": r["wave_score"] - target["target_score"],
            "geo_peak": r["peak_ratio"],
            "target_peak": target["target_peak"],
            "peak_error": r["peak_ratio"] - target["target_peak"],
            "geo_r2": r["best_r2"],
            "target_r2": target["target_r2"],
            "r2_error": r["best_r2"] - target["target_r2"],
            "geo_freq": r["best_freq"],
            "target_freq": target["target_freq"],
            "freq_error": r["best_freq"] - target["target_freq"],
            "geo_amp": r["best_amp"],
            "target_amp": target["target_amp"],
            "amp_error": r["best_amp"] - target["target_amp"],
            "weight": target["weight"],
        })

    return out


def curve_values(meta: Dict[str, Any], params: GeoParams) -> List[dict]:
    curves = geo_curves(meta, params)
    rows = []

    for (field, response, order), (xs, y) in curves.items():
        for i in range(y.size):
            rows.append({
                "field": field,
                "response": response,
                "order": order,
                "point": int(i),
                "x": float(xs[i]),
                "y": float(y[i]),
            })

    return rows


# =============================================================================
# PARAMETER SWEEP
# =============================================================================

def sweep_params(meta: Dict[str, Any], max_rows: int = 2500) -> Tuple[GeoParams, List[dict], List[dict], float]:
    """
    Small deterministic grid sweep. This is intentionally cheap.

    Once we like the family, the CUDA geo path will expose these params directly.
    """
    sweep_rows: List[dict] = []

    best_params: Optional[GeoParams] = None
    best_rows: List[dict] = []
    best_loss = float("inf")

    freqs = np.linspace(0.70, 1.50, 9)
    phases = np.linspace(1.60, 2.45, 10)
    bitdiff_amps = [0.035, 0.045, 0.055, 0.065]
    bit1_amps = [0.030, 0.040, 0.050, 0.060]
    transition_amps = [0.015, 0.022, 0.030]
    scale_phases = [0.00, 0.05, 0.10, 0.13]
    theta_phases = [0.00, 0.05, 0.10]

    count = 0

    for freq in freqs:
        for phase in phases:
            for bamp in bitdiff_amps:
                for bit1 in bit1_amps:
                    for tamp in transition_amps:
                        for sp in scale_phases:
                            for tp in theta_phases:
                                params = GeoParams(
                                    wave_freq=float(freq),
                                    phase0=float(phase),
                                    bitdiff_amp=float(bamp),
                                    bit1_amp=float(bit1),
                                    transition_amp=float(tamp),
                                    energy_amp=float(0.014),
                                    scale_phase=float(sp),
                                    theta_phase=float(tp),
                                    base_xor=float(0.00),
                                    base_delta=float(0.00),
                                )

                                rows, loss = score_geo(meta, params)

                                count += 1
                                if len(sweep_rows) < max_rows:
                                    sweep_rows.append({
                                        **asdict(params),
                                        "loss": float(loss),
                                    })

                                if loss < best_loss:
                                    best_loss = float(loss)
                                    best_params = params
                                    best_rows = rows

    if best_params is None:
        raise RuntimeError("sweep failed")

    sweep_rows.sort(key=lambda r: r["loss"])
    return best_params, best_rows, sweep_rows, best_loss


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F_M Probe 05: NumPy geo path discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--qproj", default=None, help="F_M qproj .npz. Defaults to latest.")
    p.add_argument("--gproj", default=None, help="F_M gproj .npz. Defaults to latest if available.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tiles", type=int, default=None)
    p.add_argument("--no-sweep", action="store_true", help="Use default params instead of grid sweep.")

    p.add_argument("--wave-freq", type=float, default=1.30)
    p.add_argument("--phase0", type=float, default=2.02)
    p.add_argument("--bitdiff-amp", type=float, default=0.058)
    p.add_argument("--bit1-amp", type=float, default=0.042)
    p.add_argument("--transition-amp", type=float, default=0.022)
    p.add_argument("--scale-phase", type=float, default=0.13)
    p.add_argument("--theta-phase", type=float, default=0.05)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    qproj_path = Path(args.qproj) if args.qproj else find_latest("qproj")
    gproj_path = Path(args.gproj) if args.gproj else find_latest("gproj")

    qproj_meta = load_metadata(qproj_path) if qproj_path and qproj_path.exists() else None
    gproj_meta = load_metadata(gproj_path) if gproj_path and gproj_path.exists() else None

    meta = build_geo_metadata(qproj_meta, gproj_meta, tiles=args.tiles)

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"fm_probe05_geo_numpy_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("  F_M PROBE 05 — GEO NUMPY PATH DISCOVERY")
    print("=" * 108)
    print(f"  qproj     : {qproj_path}")
    print(f"  gproj     : {gproj_path}")
    print(f"  out_dir   : {out_dir}")
    print(f"  tiles     : {meta['num_tiles']}")
    print(f"  delays    : {np.asarray(meta['tile_delay_dt']).tolist()}")
    print(f"  scales    : {np.asarray(meta['tile_scale_level']).tolist()}")
    print("-" * 108)

    if args.no_sweep:
        params = GeoParams(
            wave_freq=args.wave_freq,
            phase0=args.phase0,
            bitdiff_amp=args.bitdiff_amp,
            bit1_amp=args.bit1_amp,
            transition_amp=args.transition_amp,
            energy_amp=0.014,
            scale_phase=args.scale_phase,
            theta_phase=args.theta_phase,
            base_xor=0.0,
            base_delta=0.0,
        )
        geo_rows, loss = score_geo(meta, params)
        sweep_rows = [dict(**asdict(params), loss=float(loss))]
        print("[GEO] Used provided/default params.")
    else:
        print("[GEO] Sweeping compact classical parameter family...")
        params, geo_rows, sweep_rows, loss = sweep_params(meta)
        print("[GEO] Sweep complete.")

    geo_rows.sort(key=lambda r: r["wave_score"], reverse=True)
    comparison = target_comparison(geo_rows)
    curves = curve_values(meta, params)

    write_csv(
        out_dir / "geo_signature.csv",
        geo_rows,
        [
            "field", "response", "order",
            "wave_score", "peak_ratio", "spectral_entropy",
            "best_r2", "best_freq", "best_amp", "best_phase",
            "low_high_ratio",
        ],
    )

    write_csv(
        out_dir / "comparison.csv",
        comparison,
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
        out_dir / "curve_values.csv",
        curves,
        ["field", "response", "order", "point", "x", "y"],
    )

    write_csv(
        out_dir / "sweep.csv",
        sweep_rows,
        [
            "wave_freq", "phase0", "bitdiff_amp", "bit1_amp",
            "transition_amp", "energy_amp", "scale_phase",
            "theta_phase", "base_xor", "base_delta", "loss",
        ],
    )

    result = {
        "probe": "F_M Probe 05 — Geo NumPy Path Discovery",
        "qproj_path": str(qproj_path) if qproj_path else None,
        "gproj_path": str(gproj_path) if gproj_path else None,
        "out_dir": str(out_dir),
        "metadata": json_safe(meta),
        "best_params": asdict(params),
        "loss": float(loss),
        "geo_signature": geo_rows,
        "comparison": comparison,
    }
    write_json(out_dir / "result.json", result)

    print("\n" + "=" * 108)
    print("  BEST GEO PARAMS")
    print("=" * 108)
    for k, v in asdict(params).items():
        print(f"  {k:16s}: {v}")
    print(f"  loss            : {loss:.6f}")

    print("\n" + "=" * 108)
    print("  TOP GEO SIGNATURE")
    print("=" * 108)
    for r in geo_rows[:16]:
        print(
            f"  {r['field']:10s} {r['response']:12s} order={r['order']:11s} "
            f"score={r['wave_score']:7.4f} peak={r['peak_ratio']:6.3f} "
            f"r2={r['best_r2']:6.3f} freq={r['best_freq']:5.2f} "
            f"amp={r['best_amp']:8.5f}"
        )

    print("\n" + "=" * 108)
    print("  TARGET COMPARISON")
    print("=" * 108)
    for r in comparison:
        print(
            f"  {r['field']:10s} {r['response']:12s} order={r['order']:11s} "
            f"geo={r['geo_score']:7.4f} target={r['target_score']:7.4f} "
            f"err={r['score_error']:+8.4f} "
            f"freq={r['geo_freq']:5.2f}/{r['target_freq']:5.2f} "
            f"amp={r['geo_amp']:8.5f}/{r['target_amp']:8.5f}"
        )

    print("\n" + "=" * 108)
    print("  SAVED")
    print("=" * 108)
    print(f"  {out_dir}")
    print("=" * 108)


if __name__ == "__main__":
    main()