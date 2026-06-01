#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
F_M PROBE 01 — DELTA SPACE / BASELINE FAMILY SCAN
==============================================================================

Purpose
-------
First qproj probe for F_M.

This probe loads an F_M qproj .npz base and asks:

    Which field is most structured?

        g
        em
        delta       = em - g
        xor_delta   = em XOR g
        ctrl
        scale
        branch

    Which baseline family sees that structure best?

        benford_multiscale
        padic_residue
        spectral_dct
        autocorr_runs
        simple_stats

    Which destructive controls collapse it?

        shot_shuffle
        independent_path_shuffle
        path_swap
        tile_shuffle
        bit_shuffle
        uniform_by_tile

This is not the final F_M benchmark. It is a qproj-space discovery probe.

Expected F_M base schema
------------------------
Created by:

    ghost_oracle/F_M/f_m_qpu_generate.py dump <JOB_ID>

Useful arrays:

    g            uint8, shape (tiles, shots, 2)
    em           uint8, shape (tiles, shots, 2)
    delta        int8,  shape (tiles, shots, 2)
    xor_delta    uint8, shape (tiles, shots, 2)
    ctrl         uint8, shape (tiles, shots)
    scale        uint8, shape (tiles, shots)
    branch       uint8, shape (tiles, shots, 2)

Usage
-----
    python ghost_oracle/F_M/probes/f_m_probe01_delta_space.py

or:

    python ghost_oracle/F_M/probes/f_m_probe01_delta_space.py \
      --file ghost_oracle/F_M/data/fm_job_<JOB_ID>.npz

Outputs
-------
    ghost_oracle/F_M/probes/analysis/fm_probe01_delta_space_<timestamp>/
        result.json
        summary.csv
        family_summary.csv
        control_summary.csv
        feature_scores.csv

Interpretation rule
-------------------
The first useful F_M signal is not "Benford exists" or "p-adic exists."

The first useful F_M signal is:

    delta/xor_delta shows stronger real-vs-control separation than g/em alone,
    and the strongest controls are path-pair destroying controls.

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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


# =============================================================================
# NUMERIC HELPERS
# =============================================================================

def safe_float(x: Any) -> float:
    try:
        y = float(x)
        if not math.isfinite(y):
            return 0.0
        return y
    except Exception:
        return 0.0


def flatten(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.array([0.0], dtype=np.float64)
    return y


def zscore_distance(real: np.ndarray, null: np.ndarray) -> Tuple[float, float, float]:
    """
    Distance of real feature vector from null mean in null-standardized space.

    Returns:
        real_distance, null_mean_distance, z
    """
    real = np.asarray(real, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)

    mu = null.mean(axis=0)
    sd = null.std(axis=0) + 1e-9

    rz = (real - mu) / sd
    nz = (null - mu) / sd

    rdist = float(np.linalg.norm(rz) / math.sqrt(max(1, real.size)))
    ndist = np.linalg.norm(nz, axis=1) / math.sqrt(max(1, real.size))

    nmean = float(ndist.mean())
    nstd = float(ndist.std() + 1e-9)
    z = float((rdist - nmean) / nstd)
    return rdist, nmean, z


def rank_auc_single_positive(real_score: float, null_scores: Sequence[float]) -> float:
    """
    Single-real AUC-like rank score:
        fraction of null scores below the real score.
    """
    n = np.asarray(null_scores, dtype=np.float64)
    return float(np.mean(real_score > n) + 0.5 * np.mean(real_score == n))


# =============================================================================
# FEATURE FAMILIES
# =============================================================================

def benford_probs(base: int) -> np.ndarray:
    d = np.arange(1, base, dtype=np.float64)
    return np.log1p(1.0 / d) / np.log(base)


def first_digit_hist(x: np.ndarray, base: int, eps: float = 1e-12) -> np.ndarray:
    v = np.abs(flatten(x))
    v = v[v > eps]
    if v.size == 0:
        return np.ones(base - 1, dtype=np.float64) / (base - 1)

    logs = np.log(v) / np.log(base)
    mant = logs - np.floor(logs)
    digits = np.floor(np.power(base, mant) + 1e-12).astype(np.int64)
    digits = np.clip(digits, 1, base - 1)

    hist = np.bincount(digits, minlength=base)[1:base].astype(np.float64)
    s = hist.sum()
    if s <= 0:
        return np.ones(base - 1, dtype=np.float64) / (base - 1)
    return hist / s


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def chi2(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sum((p - q) ** 2 / (q + eps)))


def entropy_norm(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > eps]
    if p.size <= 1:
        return 0.0
    h = -np.sum(p * np.log(p))
    return float(h / math.log(p.size))


def coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    y = flatten(x)
    scale = int(scale)
    if scale <= 1:
        return y
    n = (y.size // scale) * scale
    if n < scale:
        return y
    return y[:n].reshape(-1, scale).mean(axis=1)


def features_benford_multiscale(
    x: np.ndarray,
    bases: Sequence[int] = (3, 5, 7, 10),
    scales: Sequence[int] = (1, 2, 4, 8, 16, 32),
) -> Tuple[np.ndarray, List[str]]:
    feats: List[float] = []
    names: List[str] = []

    y0 = flatten(x)
    records = {
        "value": np.abs(y0) + 1e-9,
        "delta1": np.abs(np.diff(y0)) + 1e-9 if y0.size > 1 else np.array([0.0]),
    }

    for rec_name, rec in records.items():
        for base in bases:
            vals_by_scale: List[float] = []

            for scale in scales:
                y = coarse_grain(rec, scale)
                h = first_digit_hist(y, base)
                b = benford_probs(base)

                btvd = tvd(h, b)
                vals_by_scale.append(btvd)

                feats.extend([
                    btvd,
                    chi2(h, b),
                    entropy_norm(h),
                    float(np.mean(y)),
                    float(np.std(y)),
                ])
                names.extend([
                    f"benford.{rec_name}.base{base}.scale{scale}.tvd",
                    f"benford.{rec_name}.base{base}.scale{scale}.chi2",
                    f"benford.{rec_name}.base{base}.scale{scale}.entropy",
                    f"benford.{rec_name}.base{base}.scale{scale}.mean",
                    f"benford.{rec_name}.base{base}.scale{scale}.std",
                ])

            log_s = np.log(np.asarray(scales, dtype=np.float64))
            vals = np.asarray(vals_by_scale, dtype=np.float64)
            slope = float(np.polyfit(log_s, vals, 1)[0]) if vals.size >= 2 else 0.0
            feats.append(slope)
            names.append(f"benford.{rec_name}.base{base}.scale_slope")

    return np.asarray(feats, dtype=np.float64), names


def padic_valuation_array(ints: np.ndarray, p: int) -> np.ndarray:
    y = np.maximum(1, np.asarray(ints, dtype=np.int64))
    vals = np.zeros(y.shape[0], dtype=np.int32)

    active = (y % p) == 0
    guard = 0
    while np.any(active) and guard < 64:
        vals[active] += 1
        y[active] //= p
        active = (y % p) == 0
        guard += 1

    return vals.astype(np.float64)


def features_padic_residue(
    x: np.ndarray,
    primes: Sequence[int] = (2, 3, 5, 7, 11),
    moduli: Sequence[int] = (2, 3, 4, 5, 7, 8, 16, 32),
    scale_to_int: int = 4096,
) -> Tuple[np.ndarray, List[str]]:
    y = flatten(x)
    # Preserve sign partly by shifting before scaling.
    y_shift = y - np.min(y)
    if np.max(y_shift) > 0:
        y_shift = y_shift / (np.max(y_shift) + 1e-12)
    ints = np.rint(y_shift * scale_to_int).astype(np.int64)

    feats: List[float] = []
    names: List[str] = []

    for p in primes:
        vals = padic_valuation_array(ints + 1, int(p))
        feats.extend([
            float(vals.mean()),
            float(vals.std()),
            float(np.mean(vals > 0)),
            float(np.mean(vals >= 2)),
            float(np.max(vals)),
        ])
        names.extend([
            f"padic.p{p}.valuation_mean",
            f"padic.p{p}.valuation_std",
            f"padic.p{p}.frac_gt0",
            f"padic.p{p}.frac_ge2",
            f"padic.p{p}.valuation_max",
        ])

    for m in moduli:
        r = np.mod(ints, int(m))
        hist = np.bincount(r, minlength=int(m)).astype(np.float64)
        hist /= max(1.0, hist.sum())
        uniform = np.ones(int(m), dtype=np.float64) / int(m)

        feats.extend([
            tvd(hist, uniform),
            chi2(hist, uniform),
            entropy_norm(hist),
            float(hist.max()),
        ])
        names.extend([
            f"residue.mod{m}.tvd_uniform",
            f"residue.mod{m}.chi2_uniform",
            f"residue.mod{m}.entropy",
            f"residue.mod{m}.max_bin",
        ])

    return np.asarray(feats, dtype=np.float64), names


def dct2_numpy(x: np.ndarray) -> np.ndarray:
    """
    Tiny dependency-free DCT-II approximation via FFT-even extension.

    Good enough for probe ranking. If scipy is installed later, final benchmark
    can use scipy.fft.dct with explicit normalization.
    """
    y = flatten(x)
    n = y.size
    if n <= 1:
        return y.copy()

    ext = np.concatenate([y, y[::-1]])
    fft = np.fft.fft(ext)
    k = np.arange(n)
    phase = np.exp(-1j * np.pi * k / (2 * n))
    c = np.real(fft[:n] * phase)
    return c


def features_spectral_dct(
    x: np.ndarray,
    bands: Sequence[Tuple[float, float]] = (
        (0.00, 0.02),
        (0.02, 0.05),
        (0.05, 0.10),
        (0.10, 0.25),
        (0.25, 0.50),
        (0.50, 1.00),
    ),
) -> Tuple[np.ndarray, List[str]]:
    y = flatten(x)
    if y.size < 4:
        y = np.pad(y, (0, 4 - y.size), mode="constant")

    y = y - np.mean(y)
    c = dct2_numpy(y)
    e = c ** 2
    total = float(e.sum() + 1e-12)

    feats: List[float] = []
    names: List[str] = []

    n = e.size
    for lo, hi in bands:
        a = int(max(0, min(n - 1, math.floor(lo * n))))
        b = int(max(a + 1, min(n, math.ceil(hi * n))))
        frac = float(e[a:b].sum() / total)
        feats.append(frac)
        names.append(f"spectral.band_{lo:.2f}_{hi:.2f}.energy_frac")

    # Spectral concentration and slope.
    probs = e / total
    feats.extend([
        entropy_norm(probs),
        float(np.max(probs)),
        float(np.argmax(probs) / max(1, n - 1)),
    ])
    names.extend([
        "spectral.energy_entropy",
        "spectral.max_energy_frac",
        "spectral.argmax_norm",
    ])

    # Log-energy slope across nonzero bins.
    idx = np.arange(1, n, dtype=np.float64)
    le = np.log(e[1:] + 1e-12)
    lx = np.log(idx + 1.0)
    slope = float(np.polyfit(lx, le, 1)[0]) if idx.size >= 2 else 0.0
    feats.append(slope)
    names.append("spectral.log_energy_slope")

    return np.asarray(feats, dtype=np.float64), names


def features_autocorr_runs(
    x: np.ndarray,
    lags: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
) -> Tuple[np.ndarray, List[str]]:
    y = flatten(x)
    y = y - np.mean(y)
    sd = float(np.std(y) + 1e-12)

    feats: List[float] = []
    names: List[str] = []

    for lag in lags:
        lag = int(lag)
        if y.size <= lag:
            val = 0.0
        else:
            val = float(np.mean(y[:-lag] * y[lag:]) / (sd ** 2))
        feats.append(val)
        names.append(f"autocorr.lag{lag}")

    # Run features on thresholded signal.
    raw = flatten(x)
    thr = float(np.median(raw))
    bits = (raw > thr).astype(np.uint8)

    if bits.size <= 1:
        run_lengths = np.array([1.0])
    else:
        changes = np.flatnonzero(bits[1:] != bits[:-1]) + 1
        cuts = np.concatenate([[0], changes, [bits.size]])
        run_lengths = np.diff(cuts).astype(np.float64)

    feats.extend([
        float(np.mean(run_lengths)),
        float(np.std(run_lengths)),
        float(np.max(run_lengths)),
        float(len(run_lengths) / max(1, bits.size)),
    ])
    names.extend([
        "runs.mean",
        "runs.std",
        "runs.max",
        "runs.count_per_sample",
    ])

    return np.asarray(feats, dtype=np.float64), names


def features_simple_stats(x: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    y = flatten(x)
    q = np.quantile(y, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

    feats = [
        float(np.mean(y)),
        float(np.std(y)),
        float(np.min(y)),
        float(np.max(y)),
        float(np.mean(np.abs(y - np.mean(y)))),
        float(np.mean(y == 0)),
        float(np.mean(y > 0)),
        float(np.mean(y < 0)),
        *[float(v) for v in q],
    ]
    names = [
        "stats.mean",
        "stats.std",
        "stats.min",
        "stats.max",
        "stats.mad_mean",
        "stats.frac_zero",
        "stats.frac_pos",
        "stats.frac_neg",
        "stats.q01",
        "stats.q05",
        "stats.q25",
        "stats.q50",
        "stats.q75",
        "stats.q95",
        "stats.q99",
    ]
    return np.asarray(feats, dtype=np.float64), names


FEATURE_FAMILIES = {
    "benford_multiscale": features_benford_multiscale,
    "padic_residue": features_padic_residue,
    "spectral_dct": features_spectral_dct,
    "autocorr_runs": features_autocorr_runs,
    "simple_stats": features_simple_stats,
}


# =============================================================================
# CONTROLS
# =============================================================================

def rng_shuffle_flat_same_shape(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(x).copy()
    flat = y.reshape(-1)
    rng.shuffle(flat)
    return y


def control_for_field(
    field_name: str,
    field: np.ndarray,
    all_fields: Dict[str, np.ndarray],
    control: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build destructive controls.

    For delta/xor_delta, some controls are path-aware and recompute from g/em.
    For all other fields, controls degrade to shape-preserving shuffles.
    """
    x = np.asarray(field)

    if control == "real":
        return x.copy()

    if control == "shot_shuffle":
        # Shuffle shots within each tile, preserving tile and bit axes.
        y = x.copy()
        if y.ndim >= 2:
            for t in range(y.shape[0]):
                perm = rng.permutation(y.shape[1])
                y[t] = y[t, perm, ...]
            return y
        return rng_shuffle_flat_same_shape(y, rng)

    if control == "tile_shuffle":
        y = x.copy()
        if y.ndim >= 1:
            perm = rng.permutation(y.shape[0])
            return y[perm]
        return y

    if control == "bit_shuffle":
        return rng_shuffle_flat_same_shape(x, rng)

    if control == "uniform_by_tile":
        y = x.copy().astype(np.float64)
        if y.ndim >= 2:
            out = np.empty_like(y, dtype=np.float64)
            for t in range(y.shape[0]):
                lo = float(np.min(y[t]))
                hi = float(np.max(y[t]))
                if abs(hi - lo) < 1e-12:
                    hi = lo + 1.0
                out[t] = rng.uniform(lo, hi, size=y[t].shape)
            return out
        lo = float(np.min(y))
        hi = float(np.max(y))
        if abs(hi - lo) < 1e-12:
            hi = lo + 1.0
        return rng.uniform(lo, hi, size=y.shape)

    if control == "path_swap":
        if field_name == "delta":
            return -np.asarray(all_fields["delta"])
        if field_name == "xor_delta":
            return np.asarray(all_fields["xor_delta"]).copy()
        if field_name == "g":
            return np.asarray(all_fields["em"]).copy()
        if field_name == "em":
            return np.asarray(all_fields["g"]).copy()
        return control_for_field(field_name, field, all_fields, "shot_shuffle", rng)

    if control == "independent_path_shuffle":
        if field_name in ("delta", "xor_delta"):
            g = np.asarray(all_fields["g"]).copy()
            em = np.asarray(all_fields["em"]).copy()

            # Destroy shotwise pairing independently per tile/path.
            for t in range(g.shape[0]):
                pg = rng.permutation(g.shape[1])
                pe = rng.permutation(em.shape[1])
                g[t] = g[t, pg, :]
                em[t] = em[t, pe, :]

            if field_name == "delta":
                return em.astype(np.int16) - g.astype(np.int16)
            return np.bitwise_xor(em.astype(np.uint8), g.astype(np.uint8))

        return control_for_field(field_name, field, all_fields, "shot_shuffle", rng)

    raise ValueError(f"unknown control: {control}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_fm_npz(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    with np.load(path, allow_pickle=True) as z:
        keys = list(z.keys())

        required = ["g", "em"]
        missing = [k for k in required if k not in keys]
        if missing:
            raise RuntimeError(
                f"F_M file missing required stacked arrays {missing}. "
                f"Available keys: {keys}"
            )

        g = np.asarray(z["g"])
        em = np.asarray(z["em"])

        fields: Dict[str, np.ndarray] = {
            "g": g.astype(np.float64),
            "em": em.astype(np.float64),
        }

        if "delta" in z:
            fields["delta"] = np.asarray(z["delta"]).astype(np.float64)
        else:
            fields["delta"] = em.astype(np.float64) - g.astype(np.float64)

        if "xor_delta" in z:
            fields["xor_delta"] = np.asarray(z["xor_delta"]).astype(np.float64)
        else:
            fields["xor_delta"] = np.bitwise_xor(em.astype(np.uint8), g.astype(np.uint8)).astype(np.float64)

        for optional in ["ctrl", "scale", "branch"]:
            if optional in z:
                fields[optional] = np.asarray(z[optional]).astype(np.float64)

        meta: Dict[str, Any] = {}
        for k in [
            "schema",
            "suite",
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
                v = z[k]
                try:
                    if v.shape == ():
                        meta[k] = v.item()
                    else:
                        meta[k] = np.asarray(v).tolist()
                except Exception:
                    meta[k] = str(v)

        return fields, meta


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class ScoreRow:
    field: str
    family: str
    control: str
    n_null: int
    real_dist: float
    null_dist_mean: float
    separation_z: float
    auc_rank: float
    n_features: int


def eval_field_family_control(
    field_name: str,
    field: np.ndarray,
    all_fields: Dict[str, np.ndarray],
    family_name: str,
    feature_fn,
    control: str,
    n_null: int,
    seed: int,
) -> Tuple[ScoreRow, np.ndarray, List[str], np.ndarray]:
    rng = np.random.default_rng(seed)

    real_features, names = feature_fn(field)

    null_features: List[np.ndarray] = []
    for _ in range(n_null):
        y = control_for_field(field_name, field, all_fields, control, rng)
        f, _ = feature_fn(y)
        null_features.append(f)

    null = np.vstack(null_features)
    real_dist, null_mean, sep_z = zscore_distance(real_features, null)

    # Per-null distances for rank score.
    mu = null.mean(axis=0)
    sd = null.std(axis=0) + 1e-9
    null_z = (null - mu) / sd
    null_dists = np.linalg.norm(null_z, axis=1) / math.sqrt(max(1, real_features.size))
    auc = rank_auc_single_positive(real_dist, null_dists)

    row = ScoreRow(
        field=field_name,
        family=family_name,
        control=control,
        n_null=int(n_null),
        real_dist=float(real_dist),
        null_dist_mean=float(null_mean),
        separation_z=float(sep_z),
        auc_rank=float(auc),
        n_features=int(real_features.size),
    )

    return row, real_features, names, null


def summarize(rows: List[dict], group_keys: Sequence[str]) -> List[dict]:
    groups: Dict[Tuple[Any, ...], List[dict]] = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(r)

    out: List[dict] = []
    for key, vals in groups.items():
        zs = np.asarray([safe_float(v["separation_z"]) for v in vals], dtype=np.float64)
        aucs = np.asarray([safe_float(v["auc_rank"]) for v in vals], dtype=np.float64)

        row = {k: key[i] for i, k in enumerate(group_keys)}
        row.update({
            "mean_z": float(np.mean(zs)),
            "max_z": float(np.max(zs)),
            "median_z": float(np.median(zs)),
            "mean_auc": float(np.mean(aucs)),
            "n": len(vals),
        })
        out.append(row)

    out.sort(key=lambda r: (r["mean_z"], r["max_z"]), reverse=True)
    return out


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F_M Probe 01: delta-space baseline family scan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", default=None, help="F_M qproj .npz file. Defaults to latest_fm_qpu_data.json.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-null", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument(
        "--fields",
        nargs="+",
        default=["g", "em", "delta", "xor_delta", "ctrl", "scale", "branch"],
        help="Fields to evaluate.",
    )
    p.add_argument(
        "--families",
        nargs="+",
        default=["benford_multiscale", "padic_residue", "spectral_dct", "autocorr_runs", "simple_stats"],
        choices=sorted(FEATURE_FAMILIES.keys()),
    )
    p.add_argument(
        "--controls",
        nargs="+",
        default=[
            "shot_shuffle",
            "independent_path_shuffle",
            "path_swap",
            "tile_shuffle",
            "bit_shuffle",
            "uniform_by_tile",
        ],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.file is None:
        p = find_latest_fm_file()
        if p is None:
            raise FileNotFoundError(
                f"No latest F_M qproj file found. Expected pointer or fm_job_*.npz in {DATA_DIR}"
            )
        in_path = p
    else:
        in_path = Path(args.file)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"fm_probe01_delta_space_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fields, meta = load_fm_npz(in_path)

    print("=" * 104)
    print("  F_M PROBE 01 — DELTA SPACE / BASELINE FAMILY SCAN")
    print("=" * 104)
    print(f"  file      : {in_path}")
    print(f"  out_dir   : {out_dir}")
    print(f"  backend   : {meta.get('backend', 'unknown')}")
    print(f"  job_id    : {meta.get('job_id', 'unknown')}")
    print(f"  fields    : {args.fields}")
    print(f"  families  : {args.families}")
    print(f"  controls  : {args.controls}")
    print(f"  n_null    : {args.n_null}")
    print("-" * 104)

    available_fields = [f for f in args.fields if f in fields]
    missing_fields = [f for f in args.fields if f not in fields]
    if missing_fields:
        print(f"[warn] missing fields skipped: {missing_fields}")

    rows: List[dict] = []
    feature_rows: List[dict] = []

    for field_name in available_fields:
        field = fields[field_name]
        print(f"\n[FIELD] {field_name:10s} shape={field.shape} mean={np.mean(field):.6f} std={np.std(field):.6f}")

        for family_name in args.families:
            feature_fn = FEATURE_FAMILIES[family_name]

            for control in args.controls:
                row, real_feat, feat_names, null = eval_field_family_control(
                    field_name=field_name,
                    field=field,
                    all_fields=fields,
                    family_name=family_name,
                    feature_fn=feature_fn,
                    control=control,
                    n_null=args.n_null,
                    seed=args.seed + hash((field_name, family_name, control)) % 1_000_000,
                )
                d = asdict(row)
                rows.append(d)

                print(
                    f"  {family_name:20s} vs {control:25s} "
                    f"z={row.separation_z:8.3f} "
                    f"auc={row.auc_rank:6.3f} "
                    f"dist={row.real_dist:7.3f} "
                    f"null={row.null_dist_mean:7.3f}"
                )

                # Feature-level diagnostic: which dimensions moved most?
                mu = null.mean(axis=0)
                sd = null.std(axis=0) + 1e-9
                zf = (real_feat - mu) / sd
                order = np.argsort(np.abs(zf))[::-1][:12]

                for idx in order:
                    feature_rows.append({
                        "field": field_name,
                        "family": family_name,
                        "control": control,
                        "feature": feat_names[idx],
                        "feature_z": float(zf[idx]),
                        "real_value": float(real_feat[idx]),
                        "null_mean": float(mu[idx]),
                        "null_std": float(sd[idx]),
                    })

    summary_by_field_family = summarize(rows, ["field", "family"])
    summary_by_family = summarize(rows, ["family"])
    summary_by_field = summarize(rows, ["field"])
    summary_by_control = summarize(rows, ["control"])
    summary_by_field_control = summarize(rows, ["field", "control"])

    fields_csv = [
        "field", "family", "control", "n_null",
        "real_dist", "null_dist_mean", "separation_z", "auc_rank", "n_features",
    ]
    write_csv(out_dir / "summary.csv", rows, fields_csv)

    group_fields = ["field", "family", "mean_z", "max_z", "median_z", "mean_auc", "n"]
    write_csv(out_dir / "family_summary.csv", summary_by_field_family, group_fields)

    write_csv(
        out_dir / "control_summary.csv",
        summary_by_control,
        ["control", "mean_z", "max_z", "median_z", "mean_auc", "n"],
    )

    write_csv(
        out_dir / "field_summary.csv",
        summary_by_field,
        ["field", "mean_z", "max_z", "median_z", "mean_auc", "n"],
    )

    write_csv(
        out_dir / "field_control_summary.csv",
        summary_by_field_control,
        ["field", "control", "mean_z", "max_z", "median_z", "mean_auc", "n"],
    )

    write_csv(
        out_dir / "feature_scores.csv",
        feature_rows,
        ["field", "family", "control", "feature", "feature_z", "real_value", "null_mean", "null_std"],
    )

    result = {
        "probe": "F_M Probe 01 — Delta Space / Baseline Family Scan",
        "input_file": str(in_path),
        "out_dir": str(out_dir),
        "metadata": meta,
        "config": {
            "n_null": args.n_null,
            "seed": args.seed,
            "fields": available_fields,
            "families": args.families,
            "controls": args.controls,
        },
        "rows": rows,
        "summary_by_field_family": summary_by_field_family,
        "summary_by_family": summary_by_family,
        "summary_by_field": summary_by_field,
        "summary_by_control": summary_by_control,
        "summary_by_field_control": summary_by_field_control,
    }
    write_json(out_dir / "result.json", result)

    print("\n" + "=" * 104)
    print("  TOP FIELD × FAMILY")
    print("=" * 104)
    for r in summary_by_field_family[:12]:
        print(
            f"  {r['field']:10s} {r['family']:20s} "
            f"mean_z={r['mean_z']:8.3f} max_z={r['max_z']:8.3f} "
            f"auc={r['mean_auc']:6.3f}"
        )

    print("\n" + "=" * 104)
    print("  TOP FIELDS")
    print("=" * 104)
    for r in summary_by_field[:8]:
        print(
            f"  {r['field']:10s} mean_z={r['mean_z']:8.3f} "
            f"max_z={r['max_z']:8.3f} auc={r['mean_auc']:6.3f}"
        )

    print("\n" + "=" * 104)
    print("  TOP CONTROLS")
    print("=" * 104)
    for r in summary_by_control[:8]:
        print(
            f"  {r['control']:25s} mean_z={r['mean_z']:8.3f} "
            f"max_z={r['max_z']:8.3f} auc={r['mean_auc']:6.3f}"
        )

    print("\n" + "=" * 104)
    print("  SAVED")
    print("=" * 104)
    print(f"  {out_dir}")
    print("=" * 104)


if __name__ == "__main__":
    main()