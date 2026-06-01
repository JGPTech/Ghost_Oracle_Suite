#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
F_M PROBE 02 — DELTA ADDRESS / METADATA-RESOLVED SIGNAL LOCALIZATION
==============================================================================

Purpose
-------
Probe 01 found that the useful F_M qproj signal lives mainly in:

    delta      = em - g
    xor_delta  = em XOR g

and that the most important non-catastrophic destructive control is:

    independent_path_shuffle

Probe 02 asks the next question:

    WHERE does the delta/xor_delta signal live?

It resolves the signal by:

    tile
    delay_dt
    scale_level
    mode
    theta
    bit index

Main controls
-------------
Probe 02 keeps only subtle / meaningful controls in the main score:

    shot_shuffle
    independent_path_shuffle
    tile_shuffle
    bit_shuffle_soft

Nuke controls like uniform_by_tile are excluded from the headline tables.

Feature families
----------------
The families kept from Probe 01:

    benford_multiscale
    spectral_dct
    autocorr_runs
    padic_residue

Output
------
analysis/fm_probe02_delta_address_<timestamp>/
    result.json
    tile_summary.csv
    metadata_summary.csv
    address_rows.csv
    feature_scores.csv

Interpretation
--------------
The strongest F_M address is the tile/metadata/field/family combination where:

    real signal separates from independent_path_shuffle
    shot_shuffle does not fully erase it
    tile_shuffle or metadata grouping changes it
    delta/xor_delta beat g/em-style raw channels

This probe prepares Probe 03, which should test wave-like behavior directly.
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
# IO
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

def stable_seed(*parts: Any, base: int = 20260601) -> int:
    """
    Deterministic small hash. Avoids Python's salted hash().
    """
    s = "|".join(str(p) for p in parts)
    h = base & 0xFFFFFFFF
    for ch in s:
        h = ((h * 131) + ord(ch)) & 0xFFFFFFFF
    return int(h % 2_000_000_000)


def flatten(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.array([0.0], dtype=np.float64)
    return y


def zscore_distance(real: np.ndarray, null: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
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

    return rdist, nmean, z, rz


def auc_rank(real_score: float, null_scores: np.ndarray) -> float:
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
    return hist / s if s > 0 else np.ones(base - 1, dtype=np.float64) / (base - 1)


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def chi2(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sum((p - q) ** 2 / (q + eps)))


def entropy_norm(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > eps]
    if p.size <= 1:
        return 0.0
    return float((-np.sum(p * np.log(p))) / math.log(p.size))


def coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    y = flatten(x)
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
                y = coarse_grain(rec, int(scale))
                h = first_digit_hist(y, int(base))
                b = benford_probs(int(base))

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

    # Keep the transform deterministic and comparable.
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
    y = flatten(x)
    n = y.size
    if n <= 1:
        return y.copy()

    ext = np.concatenate([y, y[::-1]])
    fft = np.fft.fft(ext)
    k = np.arange(n)
    phase = np.exp(-1j * np.pi * k / (2 * n))
    return np.real(fft[:n] * phase)


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
        feats.append(float(e[a:b].sum() / total))
        names.append(f"spectral.band_{lo:.2f}_{hi:.2f}.energy_frac")

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
        if y.size <= lag:
            val = 0.0
        else:
            val = float(np.mean(y[:-lag] * y[lag:]) / (sd ** 2))
        feats.append(val)
        names.append(f"autocorr.lag{lag}")

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


FEATURE_FAMILIES = {
    "benford_multiscale": features_benford_multiscale,
    "padic_residue": features_padic_residue,
    "spectral_dct": features_spectral_dct,
    "autocorr_runs": features_autocorr_runs,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_fm_npz(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    with np.load(path, allow_pickle=True) as z:
        keys = list(z.keys())

        if "g" not in z or "em" not in z:
            raise RuntimeError(f"F_M file requires stacked g/em arrays. Keys: {keys}")

        g = np.asarray(z["g"]).astype(np.float64)
        em = np.asarray(z["em"]).astype(np.float64)

        fields = {
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
                v = z[k]
                try:
                    meta[k] = v.item() if v.shape == () else np.asarray(v).tolist()
                except Exception:
                    meta[k] = str(v)

        return fields, meta


# =============================================================================
# ADDRESS SLICING
# =============================================================================

@dataclass
class Address:
    address_kind: str
    address_value: str
    tile_index: int
    delay_dt: int
    scale_level: int
    mode: str
    theta: float
    bit_index: int


def meta_array(meta: Dict[str, Any], key: str, n: int, default: Any) -> List[Any]:
    v = meta.get(key, None)
    if v is None:
        return [default for _ in range(n)]
    if not isinstance(v, list):
        return [v for _ in range(n)]
    if len(v) < n:
        return v + [default for _ in range(n - len(v))]
    return v[:n]


def build_address_slices(
    field: np.ndarray,
    meta: Dict[str, Any],
    max_tiles: Optional[int] = None,
) -> List[Tuple[Address, np.ndarray]]:
    """
    Return address-resolved slices.

    For a field shape:
        (tiles, shots)       -> bit_index=-1
        (tiles, shots, bits) -> one slice per bit plus flattened per tile
    """
    x = np.asarray(field)
    if x.ndim < 2:
        x = x.reshape(1, -1)

    n_tiles = int(x.shape[0])
    if max_tiles is not None:
        n_tiles = min(n_tiles, int(max_tiles))

    tile_indices = meta_array(meta, "tile_indices", n_tiles, default=-1)
    delays = meta_array(meta, "tile_delay_dt", n_tiles, default=-1)
    scales = meta_array(meta, "tile_scale_level", n_tiles, default=-1)
    modes = meta_array(meta, "tile_mode", n_tiles, default="unknown")
    thetas = meta_array(meta, "tile_theta", n_tiles, default=np.nan)

    slices: List[Tuple[Address, np.ndarray]] = []

    # Per-tile slices.
    for t in range(n_tiles):
        tile_id = int(tile_indices[t]) if str(tile_indices[t]).lstrip("-").isdigit() else t
        delay = int(delays[t]) if str(delays[t]).lstrip("-").isdigit() else -1
        scale = int(scales[t]) if str(scales[t]).lstrip("-").isdigit() else -1
        mode = str(modes[t])
        theta = float(thetas[t]) if str(thetas[t]) != "nan" else float("nan")

        tile_block = x[t]

        slices.append((
            Address(
                address_kind="tile_flat",
                address_value=f"tile{tile_id}",
                tile_index=tile_id,
                delay_dt=delay,
                scale_level=scale,
                mode=mode,
                theta=theta,
                bit_index=-1,
            ),
            tile_block,
        ))

        if tile_block.ndim >= 2:
            for b in range(tile_block.shape[-1]):
                slices.append((
                    Address(
                        address_kind="tile_bit",
                        address_value=f"tile{tile_id}.bit{b}",
                        tile_index=tile_id,
                        delay_dt=delay,
                        scale_level=scale,
                        mode=mode,
                        theta=theta,
                        bit_index=b,
                    ),
                    tile_block[..., b],
                ))

    return slices


def grouped_slices(field: np.ndarray, meta: Dict[str, Any], key: str) -> List[Tuple[str, np.ndarray]]:
    x = np.asarray(field)
    if x.ndim < 2:
        x = x.reshape(1, -1)

    n_tiles = x.shape[0]
    vals = meta_array(meta, key, n_tiles, default="unknown")

    groups: Dict[str, List[np.ndarray]] = {}
    for t, val in enumerate(vals[:n_tiles]):
        label = str(val)
        groups.setdefault(label, []).append(x[t])

    return [(label, np.concatenate([flatten(v) for v in blocks])) for label, blocks in groups.items()]


# =============================================================================
# CONTROLS
# =============================================================================

def control_slice(
    slice_data: np.ndarray,
    full_field: np.ndarray,
    field_name: str,
    fields: Dict[str, np.ndarray],
    address: Address,
    control: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Address-local controls.

    For independent_path_shuffle on delta/xor_delta, recompute from independently
    shuffled g/em at the same tile and bit index. This is the main meaningful
    path-pair destruction control.
    """
    y = np.asarray(slice_data).copy()

    if control == "shot_shuffle":
        flat = y.reshape(-1)
        rng.shuffle(flat)
        return y

    if control == "tile_shuffle":
        # For a single tile address, compare against another random tile.
        x = np.asarray(full_field)
        if x.ndim < 2 or x.shape[0] <= 1:
            flat = y.reshape(-1)
            rng.shuffle(flat)
            return y

        candidates = [i for i in range(x.shape[0]) if i != address.tile_index]
        if not candidates:
            candidates = list(range(x.shape[0]))
        pick = int(rng.choice(candidates))
        block = x[pick]
        if address.bit_index >= 0 and block.ndim >= 2:
            block = block[..., address.bit_index]
        return np.asarray(block).copy()

    if control == "bit_shuffle_soft":
        flat = y.reshape(-1)
        rng.shuffle(flat)
        return y

    if control == "independent_path_shuffle":
        if field_name not in ("delta", "xor_delta"):
            flat = y.reshape(-1)
            rng.shuffle(flat)
            return y

        g = np.asarray(fields["g"])
        em = np.asarray(fields["em"])

        # The stored tile index usually equals position for first generated base.
        # Clamp to position if metadata tile IDs are not positional.
        t = int(address.tile_index)
        if t < 0 or t >= g.shape[0]:
            t = 0

        gb = g[t].copy()
        eb = em[t].copy()

        pg = rng.permutation(gb.shape[0])
        pe = rng.permutation(eb.shape[0])
        gb = gb[pg]
        eb = eb[pe]

        if address.bit_index >= 0:
            gb = gb[..., address.bit_index]
            eb = eb[..., address.bit_index]

        if field_name == "delta":
            return eb.astype(np.float64) - gb.astype(np.float64)
        return np.bitwise_xor(eb.astype(np.uint8), gb.astype(np.uint8)).astype(np.float64)

    raise ValueError(f"unknown control: {control}")


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class AddressRow:
    field: str
    family: str
    control: str
    address_kind: str
    address_value: str
    tile_index: int
    delay_dt: int
    scale_level: int
    mode: str
    theta: float
    bit_index: int
    n_values: int
    n_features: int
    real_dist: float
    null_dist_mean: float
    separation_z: float
    auc_rank: float


def eval_address(
    field_name: str,
    full_field: np.ndarray,
    fields: Dict[str, np.ndarray],
    address: Address,
    data: np.ndarray,
    family_name: str,
    control: str,
    n_null: int,
    seed: int,
) -> Tuple[AddressRow, np.ndarray, List[str], np.ndarray]:
    feature_fn = FEATURE_FAMILIES[family_name]
    rng = np.random.default_rng(seed)

    real_feat, feat_names = feature_fn(data)

    null_feats: List[np.ndarray] = []
    for _ in range(n_null):
        c = control_slice(
            slice_data=data,
            full_field=full_field,
            field_name=field_name,
            fields=fields,
            address=address,
            control=control,
            rng=rng,
        )
        f, _ = feature_fn(c)
        null_feats.append(f)

    null = np.vstack(null_feats)

    rdist, nmean, sep_z, rz = zscore_distance(real_feat, null)

    mu = null.mean(axis=0)
    sd = null.std(axis=0) + 1e-9
    nz = (null - mu) / sd
    ndists = np.linalg.norm(nz, axis=1) / math.sqrt(max(1, real_feat.size))
    auc = auc_rank(rdist, ndists)

    row = AddressRow(
        field=field_name,
        family=family_name,
        control=control,
        address_kind=address.address_kind,
        address_value=address.address_value,
        tile_index=address.tile_index,
        delay_dt=address.delay_dt,
        scale_level=address.scale_level,
        mode=address.mode,
        theta=address.theta,
        bit_index=address.bit_index,
        n_values=int(flatten(data).size),
        n_features=int(real_feat.size),
        real_dist=float(rdist),
        null_dist_mean=float(nmean),
        separation_z=float(sep_z),
        auc_rank=float(auc),
    )

    return row, rz, feat_names, real_feat


def summarize(rows: List[dict], group_keys: Sequence[str]) -> List[dict]:
    groups: Dict[Tuple[Any, ...], List[dict]] = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(r)

    out: List[dict] = []
    for key, vals in groups.items():
        zs = np.asarray([float(v["separation_z"]) for v in vals], dtype=np.float64)
        aucs = np.asarray([float(v["auc_rank"]) for v in vals], dtype=np.float64)

        row = {k: key[i] for i, k in enumerate(group_keys)}
        row.update({
            "mean_z": float(np.mean(zs)),
            "median_z": float(np.median(zs)),
            "max_z": float(np.max(zs)),
            "mean_auc": float(np.mean(aucs)),
            "n": int(len(vals)),
        })
        out.append(row)

    out.sort(key=lambda r: (r["mean_z"], r["max_z"], r["mean_auc"]), reverse=True)
    return out


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F_M Probe 02: metadata-resolved delta/xor_delta signal localization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", default=None, help="F_M qproj .npz. Defaults to latest pointer.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-null", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument(
        "--fields",
        nargs="+",
        default=["delta", "xor_delta"],
        choices=["delta", "xor_delta"],
    )
    p.add_argument(
        "--families",
        nargs="+",
        default=["benford_multiscale", "spectral_dct", "autocorr_runs", "padic_residue"],
        choices=sorted(FEATURE_FAMILIES.keys()),
    )
    p.add_argument(
        "--controls",
        nargs="+",
        default=["shot_shuffle", "independent_path_shuffle", "tile_shuffle", "bit_shuffle_soft"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.file is None:
        p = find_latest_fm_file()
        if p is None:
            raise FileNotFoundError(f"No F_M qproj file found in {DATA_DIR}")
        in_path = p
    else:
        in_path = Path(args.file)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"fm_probe02_delta_address_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fields, meta = load_fm_npz(in_path)

    print("=" * 104)
    print("  F_M PROBE 02 — DELTA ADDRESS / METADATA-RESOLVED SIGNAL LOCALIZATION")
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

    rows: List[dict] = []
    feature_rows: List[dict] = []

    for field_name in args.fields:
        full_field = fields[field_name]
        slices = build_address_slices(full_field, meta, max_tiles=args.max_tiles)

        print(f"\n[FIELD] {field_name} shape={full_field.shape} slices={len(slices)}")

        for address, data in slices:
            if flatten(data).size < 16:
                continue

            for family_name in args.families:
                for control in args.controls:
                    seed = stable_seed(args.seed, field_name, address.address_value, family_name, control)
                    row, rz, feat_names, real_feat = eval_address(
                        field_name=field_name,
                        full_field=full_field,
                        fields=fields,
                        address=address,
                        data=data,
                        family_name=family_name,
                        control=control,
                        n_null=args.n_null,
                        seed=seed,
                    )
                    d = asdict(row)
                    rows.append(d)

                    # Top feature movement for debug.
                    order = np.argsort(np.abs(rz))[::-1][:8]
                    for idx in order:
                        feature_rows.append({
                            "field": field_name,
                            "family": family_name,
                            "control": control,
                            "address_kind": address.address_kind,
                            "address_value": address.address_value,
                            "tile_index": address.tile_index,
                            "delay_dt": address.delay_dt,
                            "scale_level": address.scale_level,
                            "mode": address.mode,
                            "theta": address.theta,
                            "bit_index": address.bit_index,
                            "feature": feat_names[idx],
                            "feature_z": float(rz[idx]),
                            "real_value": float(real_feat[idx]),
                        })

        # Small field-level print after collecting this field.
        field_rows = [r for r in rows if r["field"] == field_name]
        top = sorted(field_rows, key=lambda r: r["separation_z"], reverse=True)[:8]
        print(f"  top {field_name}:")
        for r in top:
            print(
                f"    {r['address_value']:12s} {r['family']:20s} "
                f"vs {r['control']:25s} "
                f"z={r['separation_z']:8.3f} "
                f"auc={r['auc_rank']:6.3f} "
                f"delay={r['delay_dt']} scale={r['scale_level']} mode={r['mode']}"
            )

    tile_summary = summarize(rows, ["field", "tile_index", "address_kind", "bit_index"])
    metadata_summary = summarize(rows, ["field", "delay_dt", "scale_level", "mode"])
    family_summary = summarize(rows, ["field", "family"])
    control_summary = summarize(rows, ["field", "control"])
    address_summary = summarize(rows, ["field", "address_value"])

    address_fields = [
        "field", "family", "control",
        "address_kind", "address_value", "tile_index",
        "delay_dt", "scale_level", "mode", "theta", "bit_index",
        "n_values", "n_features",
        "real_dist", "null_dist_mean", "separation_z", "auc_rank",
    ]
    write_csv(out_dir / "address_rows.csv", rows, address_fields)

    write_csv(
        out_dir / "tile_summary.csv",
        tile_summary,
        ["field", "tile_index", "address_kind", "bit_index", "mean_z", "median_z", "max_z", "mean_auc", "n"],
    )

    write_csv(
        out_dir / "metadata_summary.csv",
        metadata_summary,
        ["field", "delay_dt", "scale_level", "mode", "mean_z", "median_z", "max_z", "mean_auc", "n"],
    )

    write_csv(
        out_dir / "family_summary.csv",
        family_summary,
        ["field", "family", "mean_z", "median_z", "max_z", "mean_auc", "n"],
    )

    write_csv(
        out_dir / "control_summary.csv",
        control_summary,
        ["field", "control", "mean_z", "median_z", "max_z", "mean_auc", "n"],
    )

    write_csv(
        out_dir / "address_summary.csv",
        address_summary,
        ["field", "address_value", "mean_z", "median_z", "max_z", "mean_auc", "n"],
    )

    write_csv(
        out_dir / "feature_scores.csv",
        feature_rows,
        [
            "field", "family", "control", "address_kind", "address_value",
            "tile_index", "delay_dt", "scale_level", "mode", "theta", "bit_index",
            "feature", "feature_z", "real_value",
        ],
    )

    result = {
        "probe": "F_M Probe 02 — Delta Address / Metadata-Resolved Signal Localization",
        "input_file": str(in_path),
        "out_dir": str(out_dir),
        "metadata": meta,
        "config": {
            "n_null": args.n_null,
            "seed": args.seed,
            "fields": args.fields,
            "families": args.families,
            "controls": args.controls,
            "max_tiles": args.max_tiles,
        },
        "rows": rows,
        "tile_summary": tile_summary,
        "metadata_summary": metadata_summary,
        "family_summary": family_summary,
        "control_summary": control_summary,
        "address_summary": address_summary,
    }
    write_json(out_dir / "result.json", result)

    print("\n" + "=" * 104)
    print("  TOP ADDRESSES")
    print("=" * 104)
    for r in address_summary[:16]:
        print(
            f"  {r['field']:10s} {r['address_value']:14s} "
            f"mean_z={r['mean_z']:8.3f} max_z={r['max_z']:8.3f} auc={r['mean_auc']:6.3f}"
        )

    print("\n" + "=" * 104)
    print("  TOP METADATA GROUPS")
    print("=" * 104)
    for r in metadata_summary[:16]:
        print(
            f"  {r['field']:10s} delay={r['delay_dt']:>4} "
            f"scale={r['scale_level']:>4} mode={r['mode']:14s} "
            f"mean_z={r['mean_z']:8.3f} max_z={r['max_z']:8.3f} auc={r['mean_auc']:6.3f}"
        )

    print("\n" + "=" * 104)
    print("  TOP FIELD × FAMILY")
    print("=" * 104)
    for r in family_summary[:12]:
        print(
            f"  {r['field']:10s} {r['family']:20s} "
            f"mean_z={r['mean_z']:8.3f} max_z={r['max_z']:8.3f} auc={r['mean_auc']:6.3f}"
        )

    print("\n" + "=" * 104)
    print("  SAVED")
    print("=" * 104)
    print(f"  {out_dir}")
    print("=" * 104)


if __name__ == "__main__":
    main()