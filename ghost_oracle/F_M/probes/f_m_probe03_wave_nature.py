#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
F_M PROBE 03 — WAVE NATURE TEST / DELAY-PHASE COHERENCE
==============================================================================

Purpose
-------
Probe 01 found that F_M signal lives mainly in:

    delta      = em - g
    xor_delta  = em XOR g

Probe 02 localized strong addresses:

    xor_delta tile6
    xor_delta tile3 / tile3.bit0
    delta     tile3 / tile3.bit0

This probe asks a sharper question:

    Does the F_M differential field behave like an ordered wave-like response
    over delay/scale/tile address, or is it just static bias / independent noise?

This is NOT a broad feature scanner. It tests wave-specific evidence:

    1. Delay-order coherence
    2. Spectral peak concentration
    3. Phase coherence across addresses
    4. Cross-field coherence between delta and xor_delta
    5. Ordered metadata curve strength
    6. Collapse under delay/phase/path-pair destruction controls

Main records
------------
The probe builds one scalar response per tile/address/field:

    mean
    variance
    transition rate
    signed imbalance
    bit0/bit1 split where available

Then it orders those responses by tile metadata:

    delay_dt
    scale_level
    tile_index

and measures wave-like structure.

Controls
--------
    delay_shuffle
        preserves values, destroys delay order

    delay_reverse
        reverses delay ordering

    phase_scramble
        preserves Fourier magnitudes, randomizes phase

    circular_shift
        preserves wave shape but shifts phase

    path_pair_break
        recomputes delta/xor_delta after independently shuffling g/em shots

    tile_shuffle
        destroys tile/address order

    iid_gaussian
        same mean/std as response curve

Output
------
analysis/fm_probe03_wave_nature_<timestamp>/
    result.json
    wave_rows.csv
    control_rows.csv
    address_curves.csv

Interpretation
--------------
A wave-like F_M signature should show:

    high spectral peak ratio
    stable phase/coherence across related addresses
    strong real-vs-delay_shuffle separation
    survival under circular_shift better than phase_scramble
    collapse under path_pair_break
    ordered delay curve stronger than iid/no-order controls

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


def find_latest_fm_file() -> Optional[Path]:
    ptr = DATA_DIR / "latest_fm_qpu_data.json"
    if ptr.exists():
        try:
            with open(ptr, "r", encoding="utf-8") as f:
                j = json.load(f)
            p = Path(j["path"])
            if p.exists():
                return p
        except Exception:
            pass

    files = sorted(DATA_DIR.glob("fm_job_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def stable_seed(*parts: Any, base: int = 20260601) -> int:
    s = "|".join(str(p) for p in parts)
    h = base & 0xFFFFFFFF
    for ch in s:
        h = ((h * 131) + ord(ch)) & 0xFFFFFFFF
    return int(h % 2_000_000_000)


# =============================================================================
# DATA LOADING
# =============================================================================

def _read_npz_scalar_or_array(z: Any, key: str) -> Any:
    v = z[key]
    try:
        return v.item() if v.shape == () else np.asarray(v).tolist()
    except Exception:
        return str(v)


def load_fm_npz(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    with np.load(path, allow_pickle=True) as z:
        keys = list(z.keys())

        if "g" not in keys or "em" not in keys:
            raise RuntimeError(f"F_M file requires stacked g/em arrays. Keys: {keys}")

        g = np.asarray(z["g"]).astype(np.float64)
        em = np.asarray(z["em"]).astype(np.float64)

        fields: Dict[str, np.ndarray] = {
            "g": g,
            "em": em,
            "delta": np.asarray(z["delta"]).astype(np.float64) if "delta" in z else em - g,
            "xor_delta": np.asarray(z["xor_delta"]).astype(np.float64)
            if "xor_delta" in z else np.bitwise_xor(em.astype(np.uint8), g.astype(np.uint8)).astype(np.float64),
        }

        for optional in ["ctrl", "scale", "branch"]:
            if optional in z:
                fields[optional] = np.asarray(z[optional]).astype(np.float64)

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
            "tile_theta",
            "tile_delay_dt",
            "tile_scale_level",
            "tile_mode",
            "tile_role",
            "delays_dt",
            "scale_levels",
            "circuit_family",
        ]:
            if k in z:
                meta[k] = _read_npz_scalar_or_array(z, k)

        return fields, meta


def meta_array(meta: Dict[str, Any], key: str, n: int, default: Any) -> List[Any]:
    v = meta.get(key, None)
    if v is None:
        return [default for _ in range(n)]
    if not isinstance(v, list):
        return [v for _ in range(n)]
    if len(v) < n:
        return v + [default for _ in range(n - len(v))]
    return v[:n]


# =============================================================================
# RESPONSE CURVES
# =============================================================================

@dataclass
class AddressCurve:
    field: str
    address: str
    response_kind: str
    bit_index: int
    x_order: str
    xs: np.ndarray
    ys: np.ndarray
    tile_indices: np.ndarray
    delays: np.ndarray
    scales: np.ndarray
    modes: List[str]


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def transition_rate(arr: np.ndarray) -> float:
    y = np.asarray(arr).reshape(-1)
    if y.size <= 1:
        return 0.0
    return float(np.mean(y[1:] != y[:-1]))


def response_scalar(block: np.ndarray, response_kind: str) -> float:
    """
    Convert one tile/address block into a scalar response.

    block can be shape:
        (shots,)
        (shots, bits)
    """
    x = np.asarray(block, dtype=np.float64)

    if response_kind == "mean":
        return float(np.mean(x))

    if response_kind == "std":
        return float(np.std(x))

    if response_kind == "energy":
        return float(np.mean(x * x))

    if response_kind == "transition":
        return transition_rate(x)

    if response_kind == "imbalance":
        # Useful for binary/xor fields and signed delta.
        return float(np.mean(x > 0) - np.mean(x < 0))

    if response_kind == "bit0_mean":
        if x.ndim >= 2 and x.shape[-1] >= 1:
            return float(np.mean(x[..., 0]))
        return float(np.mean(x))

    if response_kind == "bit1_mean":
        if x.ndim >= 2 and x.shape[-1] >= 2:
            return float(np.mean(x[..., 1]))
        return float(np.mean(x))

    if response_kind == "bit_diff":
        if x.ndim >= 2 and x.shape[-1] >= 2:
            return float(np.mean(x[..., 1]) - np.mean(x[..., 0]))
        return 0.0

    raise ValueError(f"unknown response_kind: {response_kind}")


def build_curves(
    fields: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    selected_fields: Sequence[str],
    response_kinds: Sequence[str],
    x_order: str,
) -> List[AddressCurve]:
    """
    Build ordered response curves across tiles.

    x_order:
        delay       -> sort by delay_dt, then tile
        tile        -> sort by tile index
        scale_delay -> sort by scale_level, then delay_dt, then tile
    """
    if not selected_fields:
        return []

    first = fields[selected_fields[0]]
    n_tiles = int(first.shape[0])

    tile_indices = np.asarray(meta_array(meta, "tile_indices", n_tiles, default=list(range(n_tiles))), dtype=object)
    delays = np.asarray([safe_float(v, -1) for v in meta_array(meta, "tile_delay_dt", n_tiles, default=-1)], dtype=np.float64)
    scales = np.asarray([safe_float(v, -1) for v in meta_array(meta, "tile_scale_level", n_tiles, default=-1)], dtype=np.float64)
    modes = [str(v) for v in meta_array(meta, "tile_mode", n_tiles, default="unknown")]

    # Convert tile ids to ints where possible.
    tile_int = []
    for i, v in enumerate(tile_indices):
        try:
            tile_int.append(int(v))
        except Exception:
            tile_int.append(i)
    tile_int = np.asarray(tile_int, dtype=np.int64)

    if x_order == "delay":
        order = np.lexsort((tile_int, delays))
        xs = delays[order]
    elif x_order == "tile":
        order = np.argsort(tile_int)
        xs = tile_int[order].astype(np.float64)
    elif x_order == "scale_delay":
        order = np.lexsort((tile_int, delays, scales))
        xs = np.arange(n_tiles, dtype=np.float64)
    else:
        raise ValueError(f"unknown x_order: {x_order}")

    curves: List[AddressCurve] = []

    for field_name in selected_fields:
        field = np.asarray(fields[field_name])

        for response_kind in response_kinds:
            ys = np.array(
                [response_scalar(field[t], response_kind) for t in range(n_tiles)],
                dtype=np.float64,
            )

            curves.append(
                AddressCurve(
                    field=field_name,
                    address="all_tiles",
                    response_kind=response_kind,
                    bit_index=-1,
                    x_order=x_order,
                    xs=xs.astype(np.float64),
                    ys=ys[order].astype(np.float64),
                    tile_indices=tile_int[order],
                    delays=delays[order],
                    scales=scales[order],
                    modes=[modes[i] for i in order],
                )
            )

            # Bit-specific curves for fields with final bit axis.
            if field.ndim >= 3:
                for b in range(field.shape[-1]):
                    ys_b = np.array(
                        [response_scalar(field[t, ..., b], "mean") for t in range(n_tiles)],
                        dtype=np.float64,
                    )

                    curves.append(
                        AddressCurve(
                            field=field_name,
                            address=f"bit{b}",
                            response_kind="bit_mean",
                            bit_index=b,
                            x_order=x_order,
                            xs=xs.astype(np.float64),
                            ys=ys_b[order].astype(np.float64),
                            tile_indices=tile_int[order],
                            delays=delays[order],
                            scales=scales[order],
                            modes=[modes[i] for i in order],
                        )
                    )

    return curves


# =============================================================================
# WAVE METRICS
# =============================================================================

def demean(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return np.array([0.0])
    return y - np.mean(y)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = demean(a)
    y = demean(b)
    sx = float(np.linalg.norm(x))
    sy = float(np.linalg.norm(y))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.dot(x, y) / (sx * sy))


def spectral_metrics(y: np.ndarray) -> Dict[str, float]:
    """
    Wave-like spectral metrics for a short ordered curve.

    With only 7 tiles in the current qproj base, do not overread frequency.
    Treat this as coarse evidence: concentration, peak index, and phase.
    """
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

    # Ignore DC.
    if power.size <= 1:
        p_non = power
        offset = 0
    else:
        p_non = power[1:]
        offset = 1

    total = float(np.sum(p_non) + 1e-12)
    peak_local = int(np.argmax(p_non))
    peak_idx = peak_local + offset
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
    """
    Fit simple sin/cos models over possible discrete frequencies and report best R2.

    y ≈ a sin(w x) + b cos(w x) + c

    For short qproj curves, this is only a coarse wave-shape score.
    """
    x = np.asarray(xs, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)

    if yy.size < 4 or np.std(yy) < 1e-12:
        return {
            "best_r2": 0.0,
            "best_freq": 0.0,
            "best_amp": 0.0,
            "best_phase": 0.0,
        }

    # Normalize x to [0, 1] for stable frequency sweep.
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

    # Coarse frequency sweep: half to three cycles over the observed span.
    for freq in np.linspace(0.5, 3.0, 26):
        w = 2.0 * np.pi * freq
        A = np.column_stack([
            np.sin(w * xx),
            np.cos(w * xx),
            np.ones_like(xx),
        ])
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((yy - pred) ** 2))
        r2 = float(1.0 - ss_res / ss_tot)
        amp = float(math.sqrt(coef[0] ** 2 + coef[1] ** 2))
        phase = float(math.atan2(coef[1], coef[0]))

        if r2 > best["best_r2"]:
            best = {
                "best_r2": r2,
                "best_freq": float(freq),
                "best_amp": amp,
                "best_phase": phase,
            }

    if not math.isfinite(best["best_r2"]):
        best["best_r2"] = 0.0

    return best


def wave_score(xs: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
    spec = spectral_metrics(y)
    fit = sinusoid_fit_metrics(xs, y)

    # Composite score. Keep transparent and simple.
    score = (
        0.40 * spec["peak_ratio"]
        + 0.25 * max(0.0, fit["best_r2"])
        + 0.20 * (1.0 - spec["spectral_entropy"])
        + 0.15 * min(1.0, abs(spec["low_high_ratio"]) / 10.0)
    )

    metrics = {}
    metrics.update(spec)
    metrics.update(fit)
    metrics["wave_score"] = float(score)
    return float(score), metrics


# =============================================================================
# CONTROLS
# =============================================================================

def phase_scramble_curve(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    yy = demean(y)
    n = yy.size
    if n < 4 or np.std(yy) < 1e-12:
        return rng.permutation(y)

    fft = np.fft.rfft(yy)
    mag = np.abs(fft)

    phases = rng.uniform(-np.pi, np.pi, size=fft.shape)
    phases[0] = 0.0

    # Preserve Nyquist realness for even n.
    if n % 2 == 0 and phases.size > 1:
        phases[-1] = 0.0

    new_fft = mag * np.exp(1j * phases)
    out = np.fft.irfft(new_fft, n=n)
    return out + np.mean(y)


def control_curve(
    curve: AddressCurve,
    control: str,
    fields: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.asarray(curve.ys, dtype=np.float64)

    if control == "delay_shuffle":
        out = y.copy()
        rng.shuffle(out)
        return out

    if control == "delay_reverse":
        return y[::-1].copy()

    if control == "phase_scramble":
        return phase_scramble_curve(y, rng)

    if control == "circular_shift":
        if y.size <= 1:
            return y.copy()
        shift = int(rng.integers(1, y.size))
        return np.roll(y, shift)

    if control == "tile_shuffle":
        out = y.copy()
        rng.shuffle(out)
        return out

    if control == "iid_gaussian":
        return rng.normal(float(np.mean(y)), float(np.std(y) + 1e-12), size=y.shape)

    if control == "path_pair_break":
        # Recompute a new whole-curve response after independently shuffling
        # g/em per tile, then rebuild the same scalar response type.
        if curve.field not in ("delta", "xor_delta"):
            out = y.copy()
            rng.shuffle(out)
            return out

        g = np.asarray(fields["g"]).copy()
        em = np.asarray(fields["em"]).copy()

        n_tiles = g.shape[0]
        recomputed = []

        for t in range(n_tiles):
            gb = g[t].copy()
            eb = em[t].copy()

            pg = rng.permutation(gb.shape[0])
            pe = rng.permutation(eb.shape[0])
            gb = gb[pg]
            eb = eb[pe]

            if curve.field == "delta":
                block = eb.astype(np.float64) - gb.astype(np.float64)
            else:
                block = np.bitwise_xor(eb.astype(np.uint8), gb.astype(np.uint8)).astype(np.float64)

            if curve.bit_index >= 0 and block.ndim >= 2:
                block = block[..., curve.bit_index]

            recomputed.append(response_scalar(block, curve.response_kind if curve.response_kind != "bit_mean" else "mean"))

        recomputed = np.asarray(recomputed, dtype=np.float64)

        # Reapply the same ordering. curve.tile_indices are original tile ids in order;
        # for first base tile ids usually equal positions. Use current curve order length.
        if recomputed.size == y.size:
            # Need to match curve order, which already sorted tiles.
            # Use tile_indices as positions where possible.
            order_vals = []
            for tid in curve.tile_indices:
                idx = int(tid)
                if idx < 0 or idx >= recomputed.size:
                    idx = 0
                order_vals.append(recomputed[idx])
            return np.asarray(order_vals, dtype=np.float64)

        out = recomputed[: y.size]
        if out.size < y.size:
            out = np.pad(out, (0, y.size - out.size), mode="edge")
        return out

    raise ValueError(f"unknown control: {control}")


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class WaveRow:
    field: str
    address: str
    response_kind: str
    bit_index: int
    x_order: str
    n: int
    real_score: float
    peak_ratio: float
    spectral_entropy: float
    peak_index: int
    peak_freq: float
    dominant_phase: float
    best_r2: float
    best_freq: float
    best_amp: float
    best_phase: float
    low_high_ratio: float


@dataclass
class ControlRow:
    field: str
    address: str
    response_kind: str
    bit_index: int
    x_order: str
    control: str
    n_null: int
    real_score: float
    null_mean: float
    null_std: float
    effect: float
    separation_z: float
    auc_rank: float


def score_controls(
    curve: AddressCurve,
    fields: Dict[str, np.ndarray],
    controls: Sequence[str],
    n_null: int,
    seed: int,
) -> List[ControlRow]:
    real_score, _ = wave_score(curve.xs, curve.ys)
    rows: List[ControlRow] = []

    for control in controls:
        rng = np.random.default_rng(stable_seed(seed, curve.field, curve.address, curve.response_kind, curve.bit_index, curve.x_order, control))

        scores = []
        for _ in range(n_null):
            cy = control_curve(curve, control, fields, rng)
            s, _ = wave_score(curve.xs, cy)
            scores.append(s)

        ns = np.asarray(scores, dtype=np.float64)
        nmean = float(np.mean(ns))
        nstd = float(np.std(ns) + 1e-9)
        effect = float(real_score - nmean)
        z = float(effect / nstd)
        auc = float(np.mean(real_score > ns) + 0.5 * np.mean(real_score == ns))

        rows.append(
            ControlRow(
                field=curve.field,
                address=curve.address,
                response_kind=curve.response_kind,
                bit_index=curve.bit_index,
                x_order=curve.x_order,
                control=control,
                n_null=int(n_null),
                real_score=float(real_score),
                null_mean=nmean,
                null_std=nstd,
                effect=effect,
                separation_z=z,
                auc_rank=auc,
            )
        )

    return rows


def pairwise_coherence(curves: List[AddressCurve]) -> List[dict]:
    """
    Coherence/correlation between related curves.

    This helps answer whether delta and xor_delta share phase/shape.
    """
    rows: List[dict] = []

    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            a = curves[i]
            b = curves[j]

            if a.x_order != b.x_order:
                continue
            if a.ys.size != b.ys.size:
                continue
            if a.response_kind != b.response_kind:
                continue

            corr = safe_corr(a.ys, b.ys)

            _, ma = wave_score(a.xs, a.ys)
            _, mb = wave_score(b.xs, b.ys)

            phase_diff = float(np.angle(np.exp(1j * (ma["dominant_phase"] - mb["dominant_phase"]))))

            rows.append({
                "field_a": a.field,
                "address_a": a.address,
                "field_b": b.field,
                "address_b": b.address,
                "response_kind": a.response_kind,
                "x_order": a.x_order,
                "corr": corr,
                "phase_diff": phase_diff,
                "peak_index_a": int(ma["peak_index"]),
                "peak_index_b": int(mb["peak_index"]),
                "same_peak": int(ma["peak_index"] == mb["peak_index"]),
                "score_a": float(ma["wave_score"]),
                "score_b": float(mb["wave_score"]),
            })

    rows.sort(key=lambda r: (abs(r["corr"]), r["same_peak"]), reverse=True)
    return rows


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F_M Probe 03: wave-nature test over delay/tile ordered delta/xor_delta curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", default=None, help="F_M qproj .npz. Defaults to latest pointer.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-null", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260601)

    p.add_argument(
        "--fields",
        nargs="+",
        default=["delta", "xor_delta"],
        choices=["delta", "xor_delta"],
    )
    p.add_argument(
        "--response-kinds",
        nargs="+",
        default=["mean", "energy", "transition", "imbalance", "bit0_mean", "bit1_mean", "bit_diff"],
        help="Scalar response curves to build per tile.",
    )
    p.add_argument(
        "--x-orders",
        nargs="+",
        default=["delay", "tile", "scale_delay"],
        choices=["delay", "tile", "scale_delay"],
    )
    p.add_argument(
        "--controls",
        nargs="+",
        default=[
            "delay_shuffle",
            "delay_reverse",
            "phase_scramble",
            "circular_shift",
            "path_pair_break",
            "tile_shuffle",
            "iid_gaussian",
        ],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.file is None:
        p = find_latest_fm_file()
        if p is None:
            raise FileNotFoundError(f"No latest F_M qproj file found in {DATA_DIR}")
        in_path = p
    else:
        in_path = Path(args.file)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"fm_probe03_wave_nature_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fields, meta = load_fm_npz(in_path)

    print("=" * 108)
    print("  F_M PROBE 03 — WAVE NATURE TEST / DELAY-PHASE COHERENCE")
    print("=" * 108)
    print(f"  file      : {in_path}")
    print(f"  out_dir   : {out_dir}")
    print(f"  backend   : {meta.get('backend', 'unknown')}")
    print(f"  job_id    : {meta.get('job_id', 'unknown')}")
    print(f"  fields    : {args.fields}")
    print(f"  responses : {args.response_kinds}")
    print(f"  x_orders  : {args.x_orders}")
    print(f"  controls  : {args.controls}")
    print(f"  n_null    : {args.n_null}")
    print("-" * 108)

    curves: List[AddressCurve] = []
    for x_order in args.x_orders:
        curves.extend(
            build_curves(
                fields=fields,
                meta=meta,
                selected_fields=args.fields,
                response_kinds=args.response_kinds,
                x_order=x_order,
            )
        )

    wave_rows: List[dict] = []
    control_rows: List[dict] = []
    curve_rows: List[dict] = []

    for curve in curves:
        score, metrics = wave_score(curve.xs, curve.ys)

        wave_row = WaveRow(
            field=curve.field,
            address=curve.address,
            response_kind=curve.response_kind,
            bit_index=curve.bit_index,
            x_order=curve.x_order,
            n=int(curve.ys.size),
            real_score=float(score),
            peak_ratio=float(metrics["peak_ratio"]),
            spectral_entropy=float(metrics["spectral_entropy"]),
            peak_index=int(metrics["peak_index"]),
            peak_freq=float(metrics["peak_freq"]),
            dominant_phase=float(metrics["dominant_phase"]),
            best_r2=float(metrics["best_r2"]),
            best_freq=float(metrics["best_freq"]),
            best_amp=float(metrics["best_amp"]),
            best_phase=float(metrics["best_phase"]),
            low_high_ratio=float(metrics["low_high_ratio"]),
        )
        wave_rows.append(asdict(wave_row))

        for idx in range(curve.ys.size):
            curve_rows.append({
                "field": curve.field,
                "address": curve.address,
                "response_kind": curve.response_kind,
                "bit_index": curve.bit_index,
                "x_order": curve.x_order,
                "point_index": int(idx),
                "x": float(curve.xs[idx]),
                "y": float(curve.ys[idx]),
                "tile_index": int(curve.tile_indices[idx]),
                "delay_dt": float(curve.delays[idx]),
                "scale_level": float(curve.scales[idx]),
                "mode": str(curve.modes[idx]),
            })

        crows = score_controls(
            curve=curve,
            fields=fields,
            controls=args.controls,
            n_null=args.n_null,
            seed=args.seed,
        )
        control_rows.extend([asdict(r) for r in crows])

    coherence_rows = pairwise_coherence(curves)

    wave_rows.sort(key=lambda r: r["real_score"], reverse=True)
    control_rows.sort(key=lambda r: (r["effect"], r["auc_rank"], r["real_score"]), reverse=True)

    write_csv(
        out_dir / "wave_rows.csv",
        wave_rows,
        [
            "field", "address", "response_kind", "bit_index", "x_order", "n",
            "real_score", "peak_ratio", "spectral_entropy",
            "peak_index", "peak_freq", "dominant_phase",
            "best_r2", "best_freq", "best_amp", "best_phase",
            "low_high_ratio",
        ],
    )

    write_csv(
        out_dir / "control_rows.csv",
        control_rows,
        [
            "field", "address", "response_kind", "bit_index", "x_order",
            "control", "n_null",
            "real_score", "null_mean", "null_std",
            "effect", "separation_z", "auc_rank",
        ],
    )

    write_csv(
        out_dir / "address_curves.csv",
        curve_rows,
        [
            "field", "address", "response_kind", "bit_index", "x_order",
            "point_index", "x", "y", "tile_index", "delay_dt", "scale_level", "mode",
        ],
    )

    write_csv(
        out_dir / "coherence_rows.csv",
        coherence_rows,
        [
            "field_a", "address_a", "field_b", "address_b",
            "response_kind", "x_order",
            "corr", "phase_diff", "peak_index_a", "peak_index_b",
            "same_peak", "score_a", "score_b",
        ],
    )

    result = {
        "probe": "F_M Probe 03 — Wave Nature Test / Delay-Phase Coherence",
        "input_file": str(in_path),
        "out_dir": str(out_dir),
        "metadata": meta,
        "config": {
            "n_null": args.n_null,
            "seed": args.seed,
            "fields": args.fields,
            "response_kinds": args.response_kinds,
            "x_orders": args.x_orders,
            "controls": args.controls,
        },
        "wave_rows": wave_rows,
        "control_rows": control_rows,
        "coherence_rows": coherence_rows,
    }
    write_json(out_dir / "result.json", result)

    print("\n" + "=" * 108)
    print("  TOP WAVE SCORES")
    print("=" * 108)
    for r in wave_rows[:16]:
        print(
            f"  {r['field']:10s} {r['address']:10s} "
            f"{r['response_kind']:12s} order={r['x_order']:11s} "
            f"score={r['real_score']:7.4f} "
            f"peak={r['peak_ratio']:6.3f} "
            f"r2={r['best_r2']:6.3f} "
            f"freq={r['best_freq']:5.2f} "
            f"phase={r['dominant_phase']:7.3f}"
        )

    print("\n" + "=" * 108)
    print("  TOP CONTROL COLLAPSES / SEPARATIONS")
    print("=" * 108)
    for r in control_rows[:20]:
        print(
            f"  {r['field']:10s} {r['address']:10s} "
            f"{r['response_kind']:12s} order={r['x_order']:11s} "
            f"vs {r['control']:15s} "
            f"effect={r['effect']:8.4f} "
            f"auc={r['auc_rank']:6.3f} "
            f"z={r['separation_z']:8.2f}"
        )

    print("\n" + "=" * 108)
    print("  TOP COHERENCE PAIRS")
    print("=" * 108)
    for r in coherence_rows[:16]:
        print(
            f"  {r['field_a']:10s}:{r['address_a']:8s} "
            f"<-> {r['field_b']:10s}:{r['address_b']:8s} "
            f"{r['response_kind']:12s} order={r['x_order']:11s} "
            f"corr={r['corr']:7.3f} "
            f"phase_diff={r['phase_diff']:7.3f} "
            f"same_peak={r['same_peak']}"
        )

    print("\n" + "=" * 108)
    print("  SAVED")
    print("=" * 108)
    print(f"  {out_dir}")
    print("=" * 108)


if __name__ == "__main__":
    main()