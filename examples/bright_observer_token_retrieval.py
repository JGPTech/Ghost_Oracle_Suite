#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
BRIGHT OBSERVER TOKEN RETRIEVAL PROJECTOR EXAMPLE
===============================================================================

Purpose
-------
Example token-retrieval benchmark runner for the Ghost Oracle Suite with a
BrightDate-compatible observer clock.

This example does not modify the retrieval operator. It attaches an external
observer-time coordinate to the benchmark so run duration, calibration age, and
cross-run provenance can be analyzed without feeding calendar time into the
projection score.

This script compares classical token retrieval against bounded projected
retrieval under one shared candidate/evaluation harness. It is intentionally
separate from the PyTorch dataset builder:

    build_torch_token_dataset.py
        creates real transformer hidden-state retrieval datasets

    token_retrieval_projector.py
        scores those datasets with classical and projected backends

The script uses the same discipline as the S_M/TSP projector work:

    same problem
    same query/key/candidate set
    multiple scoring coordinates
    shared metrics
    rankΔ to measure ordering deformation

This is the token-retrieval analogue of:

    delta_batch       -> raw classical control
    sm_improve_batch  -> bounded projector spine
    sm_field_batch    -> field deformation channel

Here the coordinates are:

1. cosine
   Classical control baseline.

2. geo_projected
   Analytical bounded projection coordinate:

       c_q[j] = tanh(q[j] / scale[j])
       c_k[j] = tanh(k[j] / scale[j])

       G_dim(q,k,j) = sqrt((1 + c_q[j] * c_k[j]) / 2)
       G(q,k)       = mean_j G_dim(q,k,j)

3. gpu_projected / qpu_projected
   Same coordinate shape, but the per-dimension response is read from a
   projection table instead of the closed-form analytical expression.

   By default this script generates synthetic GPU/QPU-like projection tables:

       GPU table : analytical table + small shot-like noise
       QPU table : attenuated/noisier table

   Later, pass real tables with:

       --gpu-base data/gpu_projection_base.npz
       --qpu-base data/qpu_projection_base.npz

   Recognized table array names inside .npz:
       projection_table, score_table, table, g_table,
       gpu_table, qpu_table, scores

   The expected table shape is (B, B), mapping binned c_q and c_k values in
   [-1, 1] to a projected score in [0, 1].

   This script also accepts raw S_M dump files from sm_dump.py:

       sm_data_plus_<JOB_ID>.npz

   If a raw S_M dump is passed as --qpu-base or --gpu-base, the script derives a
   calibration-style projection table from the syndrome/data field statistics.
   That table is not a claim that the S_M dump directly contains token scores.
   It is the clean bridge used in this repo:

       raw QPU S_M syndrome-spacetime field
       -> calibration-derived bounded projection response surface
       -> same token retrieval scoring harness

   This keeps the scientific claim honest: the QPU/S_M file supplies a
   measured projection calibration surface, while the token task itself is
   built and evaluated classically/reproducibly.

4. field_<backend>
   Field deformation channel:

       sort candidates by classical cosine rank
       rough_i = |S_i - S_{i-1}| + |S_{i+1} - S_i|
       score_i = S_i + λ*zscore(rough_i)

   This mirrors the TSP roughness deformation, but swaps tour-edge position for
   retrieval-rank position.

Outputs
-------
    analysis/token_retrieval_projector_<timestamp>/
        result.json
        summary.csv
        per_query.csv
        projection_tables.npz

Default run
-----------
    python examples/token_retrieval_projector.py

Useful runs
-----------
    python examples/token_retrieval_projector.py --n-queries 2000 --n-keys 20000 --dim 128
    python examples/token_retrieval_projector.py --field-weights 0.001 0.005 0.01 0.05
    python examples/token_retrieval_projector.py --qpu-base data/qpu_projection_base.npz
    python examples/token_retrieval_projector.py --write-per-query
    python examples/token_retrieval_projector.py --sweep --qpu-base data/sm_data_plus_<JOB_ID>.npz

Framing
-------
Do not read this as "projection beats cosine everywhere." The claim this file
tests is cleaner:

    Given one token retrieval task and one candidate set, compare classical,
    analytical-projected, noiseless-projected, and QPU-projected scoring under a
    shared evaluation harness. Report both retrieval metrics and rankΔ so that
    helpful deformation and over-steering are visible.
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# PATHS / IO
# =============================================================================

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(
    (p for p in [HERE, *HERE.parents] if (p / ".git").exists() or (p / "requirements.txt").exists()),
    HERE.parent.parent,
)
DATA_DIR = REPO_ROOT / "data"
ANALYSIS_DIR = REPO_ROOT / "analysis"



# =============================================================================
# BRIGHT OBSERVER
# =============================================================================
#
# This is intentionally a lightweight Python-side observer, not a dependency on
# the TypeScript/Rust BrightDate packages. It uses the same public anchor used by
# BrightDate labels:
#
#     BD 0 == 2000-01-01T11:58:55.816Z
#
# and records a scalar day count relative to that label. Elapsed benchmark time
# is measured with time.monotonic_ns() so duration is stable even if the wall
# clock is adjusted during a run.
#
# The observer is provenance only. Do not feed bright_value into any score.
# =============================================================================

BRIGHT_J2000_UTC_UNIX_NS = 946_727_935_816_000_000
BRIGHT_DAY_NS = 86_400_000_000_000


def bright_value_from_unix_ns(unix_ns: int) -> float:
    """Return BrightDate-style elapsed SI days from the J2000 UTC label."""
    return (int(unix_ns) - BRIGHT_J2000_UTC_UNIX_NS) / float(BRIGHT_DAY_NS)


def format_bright_label(value: float, decimals: int = 9) -> str:
    """
    Human-facing BD/PBD label.

    Internally the scalar is signed. PBD is only a display convention for values
    before the J2000 anchor.
    """
    if abs(float(value)) < 0.5 * 10 ** (-decimals):
        return f"BD {0:.{decimals}f}"
    if value < 0:
        return f"PBD {abs(float(value)):.{decimals}f}"
    return f"BD {float(value):.{decimals}f}"


def bright_observer_snapshot(event: str) -> Dict[str, Any]:
    unix_ns = time.time_ns()
    mono_ns = time.monotonic_ns()
    bd = bright_value_from_unix_ns(unix_ns)
    return {
        "event": event,
        "unix_ns": int(unix_ns),
        "monotonic_ns": int(mono_ns),
        "bright_value": float(bd),
        "bright_label": format_bright_label(bd),
    }


def bright_file_snapshot(path_like: Optional[str]) -> Optional[Dict[str, Any]]:
    """Best-effort file timestamp snapshot for calibration/projection bases."""
    if not path_like:
        return None
    p = Path(path_like)
    if not p.exists():
        return {
            "path": str(p),
            "exists": False,
        }

    st = p.stat()
    # Python exposes mtime_ns portably. Birth/creation time is platform-specific,
    # so mtime is the least surprising reproducibility coordinate.
    mtime_ns = int(st.st_mtime_ns)
    bd = bright_value_from_unix_ns(mtime_ns)
    return {
        "path": str(p),
        "exists": True,
        "modified_unix_ns": mtime_ns,
        "modified_bright_value": float(bd),
        "modified_bright_label": format_bright_label(bd),
        "size_bytes": int(st.st_size),
    }


def build_bright_observer_record(
    *,
    start: Dict[str, Any],
    end: Dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, Any]:
    elapsed_ns = int(end["monotonic_ns"]) - int(start["monotonic_ns"])
    elapsed_seconds = elapsed_ns / 1_000_000_000.0
    elapsed_days = elapsed_seconds / 86_400.0

    gpu_file = bright_file_snapshot(args.gpu_base)
    qpu_file = bright_file_snapshot(args.qpu_base)

    def age_days(file_meta: Optional[Dict[str, Any]]) -> Optional[float]:
        if not file_meta or not file_meta.get("exists"):
            return None
        return float(start["bright_value"]) - float(file_meta["modified_bright_value"])

    return {
        "observer": "brightdate_compatible_external_observer",
        "role": "provenance_only_not_a_scoring_feature",
        "anchor": {
            "label": "BD 0",
            "utc_label": "2000-01-01T11:58:55.816Z",
            "j2000_utc_unix_ns": BRIGHT_J2000_UTC_UNIX_NS,
            "day_ns": BRIGHT_DAY_NS,
        },
        "start": start,
        "end": end,
        "elapsed": {
            "monotonic_ns": elapsed_ns,
            "seconds": elapsed_seconds,
            "days": elapsed_days,
        },
        "output_dir": str(out_dir),
        "projection_bases": {
            "gpu_base": gpu_file,
            "qpu_base": qpu_file,
            "gpu_calibration_age_days_at_start": age_days(gpu_file),
            "qpu_calibration_age_days_at_start": age_days(qpu_file),
        },
        "boundary": (
            "Bright observer values are written to metadata only. Retrieval "
            "scores, projection tables, rank deltas, and backend comparisons "
            "are computed without observer-time input."
        ),
    }


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
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


def write_csv(path: Path, rows: List[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# =============================================================================
# DATASET
# =============================================================================

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < eps] = 1.0
    return (X / n).astype(np.float32)


def build_synthetic_retrieval(
    n_queries: int,
    n_keys: int,
    dim: int,
    candidate_k: int,
    jitter: float,
    seed: int,
    normalize: bool,
    attack_fraction: float,
    attack_magnitude: float,
    attack_dim: int,
    query_attack_magnitude: float,
) -> Dict[str, np.ndarray]:
    """
    Builds a controlled retrieval task.

    Each query has a true key:
        q_i = key[true_i] + jitter * noise

    Candidate set contains the true key plus random distractors.

    Optional coherent attack:
        a fraction of non-true keys receive a large same-dim spike.
        queries can also receive a smaller same-dim spike, making dot/cosine
        susceptible depending on normalization and dimensionality.
    """
    rng = np.random.default_rng(seed)
    if candidate_k < 2:
        raise ValueError("candidate_k must be >= 2")
    if candidate_k > n_keys:
        raise ValueError("candidate_k cannot exceed n_keys")

    keys = rng.normal(0.0, 1.0, size=(n_keys, dim)).astype(np.float32)
    true_ids = rng.integers(0, n_keys, size=n_queries, dtype=np.int64)
    queries = keys[true_ids] + float(jitter) * rng.normal(0.0, 1.0, size=(n_queries, dim)).astype(np.float32)

    attack_dim = int(attack_dim) % dim
    attacked_keys = np.zeros(n_keys, dtype=bool)

    if attack_fraction > 0.0 and attack_magnitude != 0.0:
        n_attack = int(round(float(attack_fraction) * n_keys))
        n_attack = max(0, min(n_attack, n_keys))
        if n_attack:
            attacked = rng.choice(np.arange(n_keys), size=n_attack, replace=False)
            attacked_keys[attacked] = True
            keys[attacked, attack_dim] += np.float32(attack_magnitude)

    if query_attack_magnitude != 0.0:
        queries[:, attack_dim] += np.float32(query_attack_magnitude)

    if normalize:
        keys = l2_normalize(keys)
        queries = l2_normalize(queries)

    candidates = np.empty((n_queries, candidate_k), dtype=np.int64)
    true_pos = np.empty(n_queries, dtype=np.int64)

    all_ids = np.arange(n_keys, dtype=np.int64)

    for i in range(n_queries):
        tid = int(true_ids[i])
        # Draw distractors without the true id.
        pool = all_ids[all_ids != tid]
        distractors = rng.choice(pool, size=candidate_k - 1, replace=False)
        row = np.concatenate([[tid], distractors]).astype(np.int64)
        rng.shuffle(row)
        candidates[i] = row
        true_pos[i] = int(np.where(row == tid)[0][0])

    return {
        "queries": queries.astype(np.float32),
        "keys": keys.astype(np.float32),
        "true_ids": true_ids.astype(np.int64),
        "candidates": candidates.astype(np.int64),
        "true_pos": true_pos.astype(np.int64),
        "attacked_keys": attacked_keys.astype(bool),
        "attack_dim": np.array(attack_dim, dtype=np.int64),
    }


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    required = ["queries", "keys", "true_ids", "candidates"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"dataset missing arrays {missing}; required={required}")

    queries = np.asarray(z["queries"], dtype=np.float32)
    keys = np.asarray(z["keys"], dtype=np.float32)
    true_ids = np.asarray(z["true_ids"], dtype=np.int64)
    candidates = np.asarray(z["candidates"], dtype=np.int64)

    true_pos = np.empty(len(queries), dtype=np.int64)
    for i, row in enumerate(candidates):
        hit = np.where(row == true_ids[i])[0]
        if hit.size == 0:
            raise ValueError(f"candidate row {i} does not contain true id {true_ids[i]}")
        true_pos[i] = int(hit[0])

    attacked_keys = np.asarray(z["attacked_keys"], dtype=bool) if "attacked_keys" in z.files else np.zeros(len(keys), dtype=bool)

    return {
        "queries": queries,
        "keys": keys,
        "true_ids": true_ids,
        "candidates": candidates,
        "true_pos": true_pos,
        "attacked_keys": attacked_keys,
        "attack_dim": np.asarray(z["attack_dim"], dtype=np.int64) if "attack_dim" in z.files else np.array(-1, dtype=np.int64),
    }


# =============================================================================
# PROJECTION TABLES
# =============================================================================

TABLE_KEYS = [
    "projection_table",
    "score_table",
    "table",
    "g_table",
    "gpu_table",
    "qpu_table",
    "scores",
]


def analytical_table(bins: int) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.linspace(-1.0, 1.0, bins, dtype=np.float32)
    Cq, Ck = np.meshgrid(centers, centers, indexing="ij")
    table = np.sqrt(np.maximum(0.0, (1.0 + Cq * Ck) * 0.5)).astype(np.float32)
    return centers, table


def normalize_table(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32)
    T = np.nan_to_num(T, nan=0.5, posinf=1.0, neginf=0.0)
    mn = float(np.min(T))
    mx = float(np.max(T))
    if mx - mn < 1e-12:
        return np.full_like(T, 0.5, dtype=np.float32)
    # If already mostly in [0,1], clip. Otherwise min-max normalize.
    if mn >= -1e-4 and mx <= 1.0001:
        return np.clip(T, 0.0, 1.0).astype(np.float32)
    return ((T - mn) / (mx - mn)).astype(np.float32)



def _bits(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x).astype(np.int64) & 1).astype(np.uint8)


def _terminal_edge_parity(data: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(data[:, :-1], data[:, 1:]).astype(np.uint8)


def _agreement_field(data: np.ndarray, synd: np.ndarray) -> np.ndarray:
    edges = _terminal_edge_parity(data)[:, None, :]
    return (1.0 - np.bitwise_xor(edges, synd).astype(np.float32)).astype(np.float32)


def _detection_events(synd: np.ndarray) -> np.ndarray:
    if synd.shape[1] < 2:
        return np.zeros((synd.shape[0], 0, synd.shape[2]), dtype=np.float32)
    return np.bitwise_xor(synd[:, 1:, :], synd[:, :-1, :]).astype(np.float32)


def _sm_gradients(synd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if synd.shape[1] < 2 or synd.shape[2] < 2:
        empty = np.zeros((synd.shape[0], 0, 0), dtype=np.float32)
        return empty, empty
    dt_full = np.bitwise_xor(synd[:, 1:, :], synd[:, :-1, :]).astype(np.float32)
    dx_full = np.bitwise_xor(synd[:, :, 1:], synd[:, :, :-1]).astype(np.float32)
    return dt_full[:, :, :-1], dx_full[:, :-1, :]


def _has_sm_dump_schema(z: Any) -> bool:
    files = set(z.files)
    if "distances" not in files:
        return False
    for d in np.asarray(z["distances"]).astype(int).tolist():
        if f"data_d{d}" in files and f"synd_d{d}" in files:
            return True
    return False


def sm_dump_to_projection_table(
    z: Any,
    bins: int,
    seed: int = 20260529,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a bounded projection response table from a raw S_M dump.

    Important:
        raw S_M dumps do not contain token query/key responses. They contain
        data/syndrome spacetime records. This function converts the S_M field
        into a calibration-shaped response surface by using measurable field
        quantities:
            - terminal-edge / syndrome agreement
            - detection-event rate
            - temporal/spatial stress tensor anisotropy
            - local edge/time profile roughness

    The result is a deterministic BxB table in [0,1] suitable for the shared
    token retrieval harness. It is a bridge/calibration table, not a claim that
    the dump directly measured the token task.
    """
    centers, base = analytical_table(bins)

    distances = [int(d) for d in np.asarray(z["distances"]).astype(int).tolist()]
    stats: List[Dict[str, float]] = []
    profile_chunks: List[np.ndarray] = []

    for d in distances:
        dk = f"data_d{d}"
        sk = f"synd_d{d}"
        if dk not in z.files or sk not in z.files:
            continue

        data = _bits(z[dk])
        synd = _bits(z[sk])

        A = _agreement_field(data, synd)
        X = _detection_events(synd)
        dt, dx = _sm_gradients(synd)

        agreement = float(A.mean())
        det_rate = float(X.mean()) if X.size else 0.0

        if dt.size:
            Ttt = float(np.mean(dt * dt))
            Txx = float(np.mean(dx * dx))
            Ttx = float(np.mean(dt * dx))
            anis = float((Ttt - Txx) / (Ttt + Txx + 1e-12))
            coupling = float(Ttx / math.sqrt(max(Ttt * Txx, 1e-12)))
            trace = float(Ttt + Txx)
        else:
            Ttt = Txx = Ttx = anis = coupling = trace = 0.0

        edge_prof = A.mean(axis=(0, 1)).astype(np.float32)
        time_prof = A.mean(axis=(0, 2)).astype(np.float32)
        det_prof = X.mean(axis=(0, 1)).astype(np.float32) if X.size else np.zeros_like(edge_prof)

        prof = np.concatenate([
            edge_prof.reshape(-1),
            time_prof.reshape(-1),
            det_prof.reshape(-1),
            np.asarray([agreement, det_rate, Ttt, Txx, Ttx, trace, anis, coupling], dtype=np.float32),
        ]).astype(np.float32)
        profile_chunks.append(prof)

        stats.append({
            "distance": d,
            "agreement": agreement,
            "detection_rate": det_rate,
            "Ttt": Ttt,
            "Txx": Txx,
            "Ttx": Ttx,
            "trace": trace,
            "anisotropy": anis,
            "coupling": coupling,
        })

    if not profile_chunks:
        raise KeyError("S_M dump schema was detected, but no data_d*/synd_d* arrays could be used.")

    flat = np.concatenate(profile_chunks).astype(np.float32)
    mean_agreement = float(np.mean([s["agreement"] for s in stats]))
    mean_det = float(np.mean([s["detection_rate"] for s in stats]))
    mean_trace = float(np.mean([s["trace"] for s in stats]))
    mean_anis = float(np.mean([s["anisotropy"] for s in stats]))
    mean_coupling = float(np.mean([s["coupling"] for s in stats]))

    # Convert the measured field into a smooth perturbation profile.
    # Keep magnitudes conservative so the QPU/S_M table stays a calibration
    # deformation, not a made-up replacement for the analytical operator.
    q_axis = centers[:, None]
    k_axis = centers[None, :]

    # Symmetric profile responds to same-sign / opposite-sign phase relation.
    same_sign = q_axis * k_axis
    edge_wave = np.sin(np.pi * (q_axis + 1.0)) * np.sin(np.pi * (k_axis + 1.0))
    cross_wave = np.sin(0.5 * np.pi * (q_axis - k_axis))

    # Data-derived coefficients.
    attenuation = float(np.clip(0.70 + 0.30 * mean_agreement, 0.70, 1.00))
    det_amp = float(np.clip(mean_det, 0.0, 0.5))
    trace_amp = float(np.clip(mean_trace, 0.0, 0.5))
    anis_amp = float(np.clip(mean_anis, -0.5, 0.5))
    coupling_amp = float(np.clip(mean_coupling, -1.0, 1.0))

    table = 0.5 + attenuation * (base - 0.5)
    table += 0.035 * anis_amp * cross_wave
    table += 0.025 * coupling_amp * edge_wave
    table -= 0.030 * det_amp * (1.0 - same_sign)
    table += 0.020 * trace_amp * same_sign

    # Deterministic tiny texture from the actual field profile.
    # This makes different jobs/calibrations produce different tables without
    # adding nondeterministic RNG noise.
    profile = flat - float(flat.mean())
    denom = float(profile.std()) + 1e-12
    profile = profile / denom
    need = bins * bins
    reps = int(math.ceil(need / max(1, profile.size)))
    texture = np.tile(profile, reps)[:need].reshape(bins, bins)
    # Smooth by mixing neighboring directions.
    texture = (
        texture
        + np.roll(texture, 1, axis=0)
        + np.roll(texture, -1, axis=0)
        + np.roll(texture, 1, axis=1)
        + np.roll(texture, -1, axis=1)
    ) / 5.0
    table += 0.005 * texture.astype(np.float32)

    meta = {
        "source_schema": "sm_data",
        "distances": distances,
        "mean_agreement": mean_agreement,
        "mean_detection_rate": mean_det,
        "mean_trace": mean_trace,
        "mean_anisotropy": mean_anis,
        "mean_coupling": mean_coupling,
        "attenuation": attenuation,
        "distance_stats": stats,
        "note": (
            "Derived from raw S_M syndrome/data dump as a calibration response surface. "
            "This is not a direct token-measurement table."
        ),
    }

    return normalize_table(table.astype(np.float32)), meta



def load_projection_table(path: Path, fallback_bins: int) -> Tuple[np.ndarray, str]:
    z = np.load(path, allow_pickle=False)
    for key in TABLE_KEYS:
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim == 2:
                return normalize_table(arr), key
            if arr.ndim == 1:
                root = int(round(math.sqrt(arr.size)))
                if root * root == arr.size:
                    return normalize_table(arr.reshape(root, root)), key
            if arr.ndim > 2:
                # Average leading axes until a 2D response surface remains.
                while arr.ndim > 2:
                    arr = arr.mean(axis=0)
                return normalize_table(arr), key

    if _has_sm_dump_schema(z):
        table, meta = sm_dump_to_projection_table(z, fallback_bins)
        # Store metadata on the function for main() to optionally report/save.
        load_projection_table.last_sm_meta = meta
        return table, "derived_from_sm_data"

    raise KeyError(
        f"No projection table found in {path}. "
        f"Looked for one of {TABLE_KEYS}. Available arrays: {list(z.files)}. "
        "Expected either a 2D projection table or a raw S_M dump with "
        "distances + data_d*/synd_d* arrays."
    )


load_projection_table.last_sm_meta = None


def synthetic_projection_tables(
    bins: int,
    seed: int,
    gpu_noise: float,
    qpu_noise: float,
    qpu_attenuation: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers, base = analytical_table(bins)
    rng = np.random.default_rng(seed)

    gpu = base + float(gpu_noise) * rng.normal(0.0, 1.0, size=base.shape).astype(np.float32)

    qpu = 0.5 + float(qpu_attenuation) * (base - 0.5)
    qpu = qpu + float(qpu_noise) * rng.normal(0.0, 1.0, size=base.shape).astype(np.float32)

    return centers, normalize_table(gpu), normalize_table(qpu)


# =============================================================================
# SCORING
# =============================================================================

def candidate_key_block(keys: np.ndarray, candidates: np.ndarray, q0: int, q1: int) -> np.ndarray:
    # shape: (batch, candidate_k, dim)
    return keys[candidates[q0:q1]]


def dot_candidate_scores(queries: np.ndarray, keys: np.ndarray, candidates: np.ndarray, batch: int) -> np.ndarray:
    n_queries, candidate_k = candidates.shape
    out = np.empty((n_queries, candidate_k), dtype=np.float32)
    for q0 in range(0, n_queries, batch):
        q1 = min(n_queries, q0 + batch)
        Q = queries[q0:q1]
        K = candidate_key_block(keys, candidates, q0, q1)
        out[q0:q1] = np.einsum("bd,bkd->bk", Q, K, optimize=True).astype(np.float32)
    return out


def cosine_candidate_scores(queries: np.ndarray, keys: np.ndarray, candidates: np.ndarray, batch: int) -> np.ndarray:
    Qn = l2_normalize(queries)
    Kn = l2_normalize(keys)
    return dot_candidate_scores(Qn, Kn, candidates, batch)


def phase_scale(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
    X = np.vstack([queries, keys]).astype(np.float32)
    s = np.std(X, axis=0).astype(np.float32)
    s[s < 1e-6] = 1.0
    return s


def phase_lift(X: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # This returns cos(angle)-like values in [-1,1].
    return np.tanh(X / scale.reshape(1, -1)).astype(np.float32)


def geo_projected_scores(
    queries: np.ndarray,
    keys: np.ndarray,
    candidates: np.ndarray,
    scale: np.ndarray,
    batch: int,
) -> np.ndarray:
    n_queries, candidate_k = candidates.shape
    out = np.empty((n_queries, candidate_k), dtype=np.float32)
    Cq_all = phase_lift(queries, scale)
    Ck_all = phase_lift(keys, scale)

    for q0 in range(0, n_queries, batch):
        q1 = min(n_queries, q0 + batch)
        Cq = Cq_all[q0:q1]
        Ck = Ck_all[candidates[q0:q1]]
        G = np.sqrt(np.maximum(0.0, (1.0 + Cq[:, None, :] * Ck) * 0.5))
        out[q0:q1] = G.mean(axis=2).astype(np.float32)
    return out


def bin_indices(C: np.ndarray, bins: int) -> np.ndarray:
    idx = np.rint((np.clip(C, -1.0, 1.0) + 1.0) * 0.5 * (bins - 1)).astype(np.int32)
    return np.clip(idx, 0, bins - 1)


def table_projected_scores(
    queries: np.ndarray,
    keys: np.ndarray,
    candidates: np.ndarray,
    table: np.ndarray,
    scale: np.ndarray,
    batch: int,
) -> np.ndarray:
    table = normalize_table(table)
    bins = int(table.shape[0])
    if table.shape[0] != table.shape[1]:
        raise ValueError(f"projection table must be square, got shape={table.shape}")

    n_queries, candidate_k = candidates.shape
    out = np.empty((n_queries, candidate_k), dtype=np.float32)

    Cq_all = phase_lift(queries, scale)
    Ck_all = phase_lift(keys, scale)
    Iq_all = bin_indices(Cq_all, bins)
    Ik_all = bin_indices(Ck_all, bins)

    for q0 in range(0, n_queries, batch):
        q1 = min(n_queries, q0 + batch)
        Iq = Iq_all[q0:q1]                       # (B,d)
        Ik = Ik_all[candidates[q0:q1]]           # (B,K,d)
        vals = table[Iq[:, None, :], Ik]         # (B,K,d)
        out[q0:q1] = vals.mean(axis=2).astype(np.float32)

    return out


def field_deform_scores(
    base_scores: np.ndarray,
    reference_scores: np.ndarray,
    field_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deform candidate scores along classical retrieval-rank position.

    Returns:
        deformed scores
        per-query rankΔ/top20 vs reference order
    """
    n, k = base_scores.shape
    out = base_scores.copy().astype(np.float32)
    rank_delta = np.zeros(n, dtype=np.float32)

    top = min(20, k)

    for i in range(n):
        ref_order = np.argsort(-reference_scores[i])
        S_sorted = base_scores[i, ref_order]

        if k <= 1 or float(np.std(S_sorted)) < 1e-12:
            continue

        left = np.empty_like(S_sorted)
        right = np.empty_like(S_sorted)
        left[0] = S_sorted[0]
        left[1:] = S_sorted[:-1]
        right[-1] = S_sorted[-1]
        right[:-1] = S_sorted[1:]

        rough = np.abs(S_sorted - left) + np.abs(right - S_sorted)
        rz = (rough - rough.mean()) / (rough.std() + 1e-12)

        deformed_sorted = S_sorted + float(field_weight) * rz
        out[i, ref_order] = deformed_sorted.astype(np.float32)

        base_top = set(np.argsort(-base_scores[i])[:top].tolist())
        def_top = set(np.argsort(-out[i])[:top].tolist())
        union = max(1, len(base_top | def_top))
        rank_delta[i] = 1.0 - len(base_top & def_top) / union

    return out, rank_delta


# =============================================================================
# METRICS
# =============================================================================

def ranks_from_scores(scores: np.ndarray, true_pos: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(scores.shape[0], dtype=np.int32)
    for i in range(scores.shape[0]):
        ranks[i] = int(np.where(order[i] == true_pos[i])[0][0]) + 1
    return ranks


def rank_delta_vs_reference(scores: np.ndarray, reference: np.ndarray, top: int = 20) -> float:
    n, k = scores.shape
    top = min(top, k)
    vals = []
    for i in range(n):
        a = set(np.argsort(-reference[i])[:top].tolist())
        b = set(np.argsort(-scores[i])[:top].tolist())
        vals.append(1.0 - len(a & b) / max(1, len(a | b)))
    return float(np.mean(vals))


def evaluate_backend(
    name: str,
    scores: np.ndarray,
    true_pos: np.ndarray,
    candidates: np.ndarray,
    attacked_keys: np.ndarray,
    reference_scores: Optional[np.ndarray],
    seconds: float,
    field_weight: float = 0.0,
    field_rank_delta: Optional[np.ndarray] = None,
) -> Tuple[dict, List[dict]]:
    ranks = ranks_from_scores(scores, true_pos)
    n, k = scores.shape
    top5_k = min(5, k)

    true_scores = scores[np.arange(n), true_pos]
    masked = scores.copy()
    masked[np.arange(n), true_pos] = -np.inf
    best_false = np.max(masked, axis=1)
    margin = true_scores - best_false

    top1_idx = np.argmax(scores, axis=1)
    pred_key = candidates[np.arange(n), top1_idx]
    pred_attacked = attacked_keys[pred_key] if attacked_keys.size else np.zeros(n, dtype=bool)

    if attacked_keys.size:
        attacked_candidate_mask = attacked_keys[candidates]
        attacked_mass = float(np.mean(attacked_candidate_mask))
        attacked_top1 = float(np.mean(pred_attacked))
    else:
        attacked_mass = 0.0
        attacked_top1 = 0.0

    summary = {
        "backend": name,
        "field_weight": float(field_weight),
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= top5_k)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_true_score": float(np.mean(true_scores)),
        "mean_best_false_score": float(np.mean(best_false)),
        "mean_margin": float(np.mean(margin)),
        "median_margin": float(np.median(margin)),
        "attacked_candidate_fraction": attacked_mass,
        "attacked_top1_fraction": attacked_top1,
        "rank_delta_vs_cosine_top20": (
            rank_delta_vs_reference(scores, reference_scores, top=20)
            if reference_scores is not None else 0.0
        ),
        "field_rank_delta_top20": (
            float(np.mean(field_rank_delta)) if field_rank_delta is not None else 0.0
        ),
        "seconds": float(seconds),
    }

    per_query = []
    for i in range(n):
        per_query.append({
            "query": int(i),
            "backend": name,
            "rank": int(ranks[i]),
            "true_pos": int(true_pos[i]),
            "pred_pos": int(top1_idx[i]),
            "pred_key": int(pred_key[i]),
            "pred_attacked": int(bool(pred_attacked[i])),
            "true_score": float(true_scores[i]),
            "best_false_score": float(best_false[i]),
            "margin": float(margin[i]),
        })

    return summary, per_query


def print_summary(rows: List[dict]) -> None:
    rows = sorted(rows, key=lambda r: (-r["top1"], -r["mrr"], r["seconds"]))

    print("\n" + "=" * 128)
    print("  TOKEN RETRIEVAL PROJECTOR SUMMARY")
    print("=" * 128)
    print(
        f"  {'rank':>4} | {'backend':<24} | {'fw':>7} | {'top1':>7} | {'top5':>7} | "
        f"{'MRR':>7} | {'meanR':>7} | {'margin':>9} | {'atk@1':>7} | "
        f"{'rankΔ':>7} | {'sec':>8}"
    )
    print("  " + "-" * 126)
    for i, r in enumerate(rows, 1):
        print(
            f"  {i:>4} | {r['backend']:<24} | {r['field_weight']:>7.4f} | "
            f"{r['top1']:>7.3f} | {r['top5']:>7.3f} | {r['mrr']:>7.3f} | "
            f"{r['mean_rank']:>7.2f} | {r['mean_margin']:>9.5f} | "
            f"{r['attacked_top1_fraction']:>7.3f} | "
            f"{r['rank_delta_vs_cosine_top20']:>7.3f} | {r['seconds']:>8.3f}"
        )


# =============================================================================
# RUNNER
# =============================================================================

SUMMARY_FIELDS = [
    "backend",
    "field_weight",
    "top1",
    "top5",
    "mrr",
    "mean_rank",
    "median_rank",
    "mean_true_score",
    "mean_best_false_score",
    "mean_margin",
    "median_margin",
    "attacked_candidate_fraction",
    "attacked_top1_fraction",
    "rank_delta_vs_cosine_top20",
    "field_rank_delta_top20",
    "seconds",
]

PER_QUERY_FIELDS = [
    "query",
    "backend",
    "rank",
    "true_pos",
    "pred_pos",
    "pred_key",
    "pred_attacked",
    "true_score",
    "best_false_score",
    "margin",
]


def run_backend(
    name: str,
    score_fn,
    dataset: Dict[str, np.ndarray],
    cosine_ref: Optional[np.ndarray],
    field_weights: Sequence[float],
    include_field: bool,
) -> Tuple[List[dict], List[dict], Dict[str, np.ndarray]]:
    queries = dataset["queries"]
    keys = dataset["keys"]
    candidates = dataset["candidates"]
    true_pos = dataset["true_pos"]
    attacked_keys = dataset["attacked_keys"]

    summaries: List[dict] = []
    per_query_rows: List[dict] = []
    score_store: Dict[str, np.ndarray] = {}

    t0 = time.time()
    scores = score_fn()
    seconds = time.time() - t0

    summ, pq = evaluate_backend(
        name=name,
        scores=scores,
        true_pos=true_pos,
        candidates=candidates,
        attacked_keys=attacked_keys,
        reference_scores=cosine_ref,
        seconds=seconds,
    )
    summaries.append(summ)
    per_query_rows.extend(pq)
    score_store[name] = scores

    if include_field:
        if cosine_ref is None:
            raise ValueError("field deformation requires cosine reference scores")
        for fw in field_weights:
            t1 = time.time()
            fscores, rd = field_deform_scores(scores, cosine_ref, float(fw))
            fseconds = time.time() - t1
            fname = f"field_{name}"
            summ, pq = evaluate_backend(
                name=fname,
                scores=fscores,
                true_pos=true_pos,
                candidates=candidates,
                attacked_keys=attacked_keys,
                reference_scores=cosine_ref,
                seconds=seconds + fseconds,
                field_weight=float(fw),
                field_rank_delta=rd,
            )
            summaries.append(summ)
            per_query_rows.extend(pq)
            score_store[f"{fname}_fw{fw:g}"] = fscores

    return summaries, per_query_rows, score_store



def _parse_num_list(values: Sequence[str], cast) -> List[Any]:
    out: List[Any] = []
    for v in values:
        for part in str(v).replace(",", " ").split():
            if part.strip():
                out.append(cast(part.strip()))
    return out


def _short_backend(rows: List[dict], name: str, fw: float = 0.0) -> Optional[dict]:
    for r in rows:
        if r.get("backend") == name and abs(float(r.get("field_weight", 0.0)) - float(fw)) < 1e-12:
            return r
    return None



def run_single_experiment(
    args: argparse.Namespace,
    *,
    n_queries: int,
    n_keys: int,
    dim: int,
    candidate_k: int,
    jitter: float,
    attack_fraction: float,
    attack_magnitude: float,
    query_attack_magnitude: float,
    seed: int,
    gpu_table: np.ndarray,
    qpu_table: np.ndarray,
    write_per_query: bool = False,
) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    dataset = build_synthetic_retrieval(
        n_queries=n_queries,
        n_keys=n_keys,
        dim=dim,
        candidate_k=candidate_k,
        jitter=jitter,
        seed=seed,
        normalize=args.normalize,
        attack_fraction=attack_fraction,
        attack_magnitude=attack_magnitude,
        attack_dim=args.attack_dim,
        query_attack_magnitude=query_attack_magnitude,
    )

    queries = dataset["queries"]
    keys = dataset["keys"]
    candidates = dataset["candidates"]
    scale = phase_scale(queries, keys)

    summaries: List[dict] = []
    per_query_rows: List[dict] = []

    t0 = time.time()
    cosine_scores = cosine_candidate_scores(queries, keys, candidates, args.batch)
    cosine_seconds = time.time() - t0

    summ, pq = evaluate_backend(
        name="cosine",
        scores=cosine_scores,
        true_pos=dataset["true_pos"],
        candidates=candidates,
        attacked_keys=dataset["attacked_keys"],
        reference_scores=None,
        seconds=cosine_seconds,
    )
    summaries.append(summ)
    if write_per_query:
        per_query_rows.extend(pq)

    # Dot baseline.
    s, pq, _ = run_backend(
        name="dot",
        score_fn=lambda: dot_candidate_scores(queries, keys, candidates, args.batch),
        dataset=dataset,
        cosine_ref=cosine_scores,
        field_weights=[],
        include_field=False,
    )
    summaries.extend(s)
    if write_per_query:
        per_query_rows.extend(pq)

    include_field = not args.no_field

    # Analytical projected.
    s, pq, _ = run_backend(
        name="geo_projected",
        score_fn=lambda: geo_projected_scores(queries, keys, candidates, scale, args.batch),
        dataset=dataset,
        cosine_ref=cosine_scores,
        field_weights=args.field_weights,
        include_field=include_field,
    )
    summaries.extend(s)
    if write_per_query:
        per_query_rows.extend(pq)

    # GPU/noiseless table projected.
    s, pq, _ = run_backend(
        name="gpu_projected",
        score_fn=lambda: table_projected_scores(queries, keys, candidates, gpu_table, scale, args.batch),
        dataset=dataset,
        cosine_ref=cosine_scores,
        field_weights=args.field_weights,
        include_field=include_field,
    )
    summaries.extend(s)
    if write_per_query:
        per_query_rows.extend(pq)

    # QPU/S_M-derived table projected.
    s, pq, _ = run_backend(
        name="qpu_projected",
        score_fn=lambda: table_projected_scores(queries, keys, candidates, qpu_table, scale, args.batch),
        dataset=dataset,
        cosine_ref=cosine_scores,
        field_weights=args.field_weights,
        include_field=include_field,
    )
    summaries.extend(s)
    if write_per_query:
        per_query_rows.extend(pq)

    meta = {
        "n_queries": n_queries,
        "n_keys": n_keys,
        "dim": dim,
        "candidate_k": candidate_k,
        "jitter": jitter,
        "attack_fraction": attack_fraction,
        "attack_magnitude": attack_magnitude,
        "query_attack_magnitude": query_attack_magnitude,
        "seed": seed,
        "attacked_keys": int(np.sum(dataset["attacked_keys"])),
    }
    return summaries, per_query_rows, meta


def run_sweep(
    args: argparse.Namespace,
    out_dir: Path,
    gpu_table: np.ndarray,
    qpu_table: np.ndarray,
    gpu_sm_meta: Optional[dict],
    qpu_sm_meta: Optional[dict],
    observer_start: Optional[Dict[str, Any]] = None,
) -> None:
    dims = _parse_num_list(args.sweep_dims, int)
    jitters = _parse_num_list(args.sweep_jitters, float)
    candidate_ks = _parse_num_list(args.sweep_candidate_ks, int)
    attack_mags = _parse_num_list(args.sweep_attack_magnitudes, float)
    query_attack_mags = _parse_num_list(args.sweep_query_attack_magnitudes, float)

    grid = list(itertools.product(dims, jitters, candidate_ks, attack_mags, query_attack_mags))
    if args.sweep_limit > 0:
        grid = grid[:args.sweep_limit]

    print("\n" + "=" * 128)
    print("  TOKEN RETRIEVAL PROJECTOR SWEEP")
    print("=" * 128)
    print(f"  Grid points    : {len(grid)}")
    print(f"  n_queries      : {args.sweep_n_queries}")
    print(f"  n_keys         : {args.sweep_n_keys}")
    print(f"  dims           : {dims}")
    print(f"  jitters        : {jitters}")
    print(f"  candidate_ks   : {candidate_ks}")
    print(f"  attack mags    : {attack_mags}")
    print(f"  query atk mags : {query_attack_mags}")
    print("  " + "-" * 126)

    sweep_rows: List[dict] = []
    full_rows: List[dict] = []

    t_all = time.time()

    for gi, (dim, jitter, candidate_k, attack_mag, query_attack_mag) in enumerate(grid, 1):
        if candidate_k > args.sweep_n_keys:
            print(f"  [{gi:>3}/{len(grid)}] skip candidate_k={candidate_k} > n_keys={args.sweep_n_keys}")
            continue

        seed = int(args.seed + 1000003 * gi + 9176 * dim + int(jitter * 1000))
        t0 = time.time()
        summaries, _, meta = run_single_experiment(
            args,
            n_queries=args.sweep_n_queries,
            n_keys=args.sweep_n_keys,
            dim=int(dim),
            candidate_k=int(candidate_k),
            jitter=float(jitter),
            attack_fraction=float(args.attack_fraction),
            attack_magnitude=float(attack_mag),
            query_attack_magnitude=float(query_attack_mag),
            seed=seed,
            gpu_table=gpu_table,
            qpu_table=qpu_table,
            write_per_query=False,
        )
        elapsed = time.time() - t0

        cosine = _short_backend(summaries, "cosine")
        dot = _short_backend(summaries, "dot")
        geo = _short_backend(summaries, "geo_projected")
        gpu = _short_backend(summaries, "gpu_projected")
        qpu = _short_backend(summaries, "qpu_projected")

        if cosine is None or qpu is None or geo is None or gpu is None:
            continue

        # Best field-QPU row, if fields are enabled.
        field_qpu = [r for r in summaries if r["backend"] == "field_qpu_projected"]
        best_field_qpu = max(field_qpu, key=lambda r: (r["top1"], r["mrr"])) if field_qpu else None

        row = {
            **meta,
            "grid_index": gi,
            "seconds": elapsed,

            "cosine_top1": cosine["top1"],
            "cosine_top5": cosine["top5"],
            "cosine_mrr": cosine["mrr"],
            "cosine_atk1": cosine["attacked_top1_fraction"],
            "cosine_margin": cosine["mean_margin"],

            "dot_top1": dot["top1"] if dot else None,
            "dot_atk1": dot["attacked_top1_fraction"] if dot else None,

            "geo_top1": geo["top1"],
            "geo_top5": geo["top5"],
            "geo_mrr": geo["mrr"],
            "geo_atk1": geo["attacked_top1_fraction"],
            "geo_rank_delta": geo["rank_delta_vs_cosine_top20"],

            "gpu_top1": gpu["top1"],
            "gpu_top5": gpu["top5"],
            "gpu_mrr": gpu["mrr"],
            "gpu_atk1": gpu["attacked_top1_fraction"],
            "gpu_rank_delta": gpu["rank_delta_vs_cosine_top20"],

            "qpu_top1": qpu["top1"],
            "qpu_top5": qpu["top5"],
            "qpu_mrr": qpu["mrr"],
            "qpu_atk1": qpu["attacked_top1_fraction"],
            "qpu_rank_delta": qpu["rank_delta_vs_cosine_top20"],

            "qpu_adv_top1": qpu["top1"] - cosine["top1"],
            "qpu_adv_mrr": qpu["mrr"] - cosine["mrr"],
            "qpu_attack_reduction": cosine["attacked_top1_fraction"] - qpu["attacked_top1_fraction"],

            "geo_adv_top1": geo["top1"] - cosine["top1"],
            "gpu_adv_top1": gpu["top1"] - cosine["top1"],

            "best_field_qpu_top1": best_field_qpu["top1"] if best_field_qpu else None,
            "best_field_qpu_fw": best_field_qpu["field_weight"] if best_field_qpu else None,
            "best_field_qpu_adv_top1": (best_field_qpu["top1"] - cosine["top1"]) if best_field_qpu else None,
            "best_field_qpu_rank_delta": best_field_qpu["rank_delta_vs_cosine_top20"] if best_field_qpu else None,
        }

        sweep_rows.append(row)

        for s in summaries:
            full = {**meta, "grid_index": gi, **s}
            full_rows.append(full)

        print(
            f"  [{gi:>3}/{len(grid)}] "
            f"d={dim:<3} jit={jitter:<4g} k={candidate_k:<4} "
            f"atk={attack_mag:<4g} qAtk={query_attack_mag:<4g} | "
            f"cos={cosine['top1']:.3f} qpu={qpu['top1']:.3f} "
            f"adv={row['qpu_adv_top1']:+.3f} "
            f"atk↓={row['qpu_attack_reduction']:+.3f} "
            f"rankΔ={qpu['rank_delta_vs_cosine_top20']:.3f} "
            f"{elapsed:.2f}s"
        )

    # Sort by useful regimes:
    # 1. QPU improves top1
    # 2. QPU reduces attacked top1 selection
    # 3. cosine is not trivially perfect
    sweep_ranked = sorted(
        sweep_rows,
        key=lambda r: (
            r["qpu_adv_top1"],
            r["qpu_attack_reduction"],
            -abs(r["cosine_top1"] - 0.55),  # prefer nontrivial not-total-collapse bands
            r["qpu_mrr"],
        ),
        reverse=True,
    )

    fields = [
        "grid_index", "dim", "jitter", "candidate_k", "attack_magnitude", "query_attack_magnitude",
        "cosine_top1", "geo_top1", "gpu_top1", "qpu_top1", "dot_top1",
        "qpu_adv_top1", "geo_adv_top1", "gpu_adv_top1",
        "cosine_atk1", "qpu_atk1", "qpu_attack_reduction",
        "qpu_rank_delta", "geo_rank_delta", "gpu_rank_delta",
        "cosine_mrr", "qpu_mrr", "qpu_adv_mrr",
        "best_field_qpu_top1", "best_field_qpu_fw", "best_field_qpu_adv_top1", "best_field_qpu_rank_delta",
        "seconds", "seed",
    ]

    write_csv(out_dir / "sweep_summary.csv", sweep_rows, fields)
    write_csv(out_dir / "sweep_ranked.csv", sweep_ranked, fields)

    full_fields = [
        "grid_index", "dim", "jitter", "candidate_k", "attack_magnitude", "query_attack_magnitude",
        *SUMMARY_FIELDS,
    ]
    write_csv(out_dir / "sweep_all_backends.csv", full_rows, full_fields)

    observer_record = None
    if not getattr(args, "no_bright_observer", False) and observer_start is not None:
        observer_end = bright_observer_snapshot("sweep_end")
        observer_record = build_bright_observer_record(
            start=observer_start,
            end=observer_end,
            args=args,
            out_dir=out_dir,
        )
        with open(out_dir / "bright_observer.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(observer_record), f, indent=2)

    payload = {
        "args": vars(args),
        "observer": observer_record,
        "grid_count": len(grid),
        "rows": sweep_rows,
        "ranked": sweep_ranked,
        "tables": {
            "gpu_shape": list(gpu_table.shape),
            "qpu_shape": list(qpu_table.shape),
            "gpu_base": args.gpu_base,
            "qpu_base": args.qpu_base,
            "gpu_sm_meta": gpu_sm_meta,
            "qpu_sm_meta": qpu_sm_meta,
        },
        "total_seconds": time.time() - t_all,
    }
    with open(out_dir / "sweep_result.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)

    print("\n" + "=" * 128)
    print("  SWEEP TOP REGIMES — ranked by qpu_top1 - cosine_top1")
    print("=" * 128)
    print(
        f"  {'rank':>4} | {'dim':>4} | {'jit':>5} | {'k':>5} | {'atk':>5} | {'qAtk':>5} | "
        f"{'cos':>6} | {'qpu':>6} | {'adv':>7} | {'atk↓':>7} | {'rankΔ':>7}"
    )
    print("  " + "-" * 126)
    for i, r in enumerate(sweep_ranked[:20], 1):
        print(
            f"  {i:>4} | {r['dim']:>4} | {r['jitter']:>5.2f} | {r['candidate_k']:>5} | "
            f"{r['attack_magnitude']:>5.1f} | {r['query_attack_magnitude']:>5.1f} | "
            f"{r['cosine_top1']:>6.3f} | {r['qpu_top1']:>6.3f} | "
            f"{r['qpu_adv_top1']:>+7.3f} | {r['qpu_attack_reduction']:>+7.3f} | "
            f"{r['qpu_rank_delta']:>7.3f}"
        )

    print(f"\n[SAVED] {out_dir}")
    print(f"{'=' * 128}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Token retrieval projector example: classical vs analytical/GPU/QPU projected scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--dataset", default=None, help="Optional .npz dataset with queries, keys, true_ids, candidates.")
    p.add_argument("--save-dataset", action="store_true", help="Save generated synthetic dataset into the output dir.")

    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--n-keys", type=int, default=8192)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--candidate-k", type=int, default=256)
    p.add_argument("--jitter", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=20260529)
    p.add_argument("--normalize", action="store_true", help="L2-normalize generated query/key vectors.")

    p.add_argument("--attack-fraction", type=float, default=0.05)
    p.add_argument("--attack-magnitude", type=float, default=8.0)
    p.add_argument("--query-attack-magnitude", type=float, default=0.0)
    p.add_argument("--attack-dim", type=int, default=0)

    p.add_argument("--bins", type=int, default=256)
    p.add_argument("--gpu-base", default=None, help="Optional .npz projection table for noiseless/GPU base.")
    p.add_argument("--qpu-base", default=None, help="Optional .npz projection table for QPU base.")
    p.add_argument("--gpu-noise", type=float, default=0.002)
    p.add_argument("--qpu-noise", type=float, default=0.025)
    p.add_argument("--qpu-attenuation", type=float, default=0.82)

    p.add_argument("--field-weights", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.05])
    p.add_argument("--no-field", action="store_true")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--write-per-query", action="store_true")

    # Sweep mode: maps operating regimes automatically.
    p.add_argument("--sweep", action="store_true", help="Run an operating-regime sweep instead of one benchmark.")
    p.add_argument("--sweep-n-queries", type=int, default=600, help="Queries per sweep point.")
    p.add_argument("--sweep-n-keys", type=int, default=8192, help="Keys per sweep point.")
    p.add_argument("--sweep-dims", nargs="+", default=["8", "16", "32", "64"])
    p.add_argument("--sweep-jitters", nargs="+", default=["0.5", "0.75", "1.0", "1.25", "1.5"])
    p.add_argument("--sweep-candidate-ks", nargs="+", default=["256", "512", "1024"])
    p.add_argument("--sweep-attack-magnitudes", nargs="+", default=["8", "16", "24"])
    p.add_argument("--sweep-query-attack-magnitudes", nargs="+", default=["0", "2", "4", "8"])
    p.add_argument("--sweep-limit", type=int, default=0, help="For quick tests, cap number of grid points. 0 = no cap.")

    p.add_argument("--out-dir", default=None)

    p.add_argument(
        "--no-bright-observer",
        action="store_true",
        help="Disable BrightDate-compatible observer metadata.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_all = time.time()
    observer_start = None if args.no_bright_observer else bright_observer_snapshot("benchmark_start")

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"bright_observer_token_retrieval_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 128}")
    print("  BRIGHT OBSERVER TOKEN RETRIEVAL EXAMPLE — CLASSICAL vs PROJECTED")
    print(f"{'=' * 128}")
    print(f"  Out dir       : {out_dir}")
    if observer_start:
        print(f"  Observer      : {observer_start['bright_label']} ({observer_start['bright_value']:.9f} days)")
    else:
        print("  Observer      : disabled")
    print(f"  Dataset       : {args.dataset or 'synthetic'}")
    if args.dataset:
        print("  requested N   : ignored for --dataset; loaded array shapes are printed below")
    else:
        print(f"  n_queries     : {args.n_queries}")
        print(f"  n_keys        : {args.n_keys}")
        print(f"  dim           : {args.dim}")
        print(f"  candidate_k   : {args.candidate_k}")
    print(f"  attack frac   : {args.attack_fraction}")
    print(f"  field weights : {args.field_weights if not args.no_field else 'disabled'}")

    if args.dataset:
        dataset = load_dataset(Path(args.dataset))
    else:
        dataset = build_synthetic_retrieval(
            n_queries=args.n_queries,
            n_keys=args.n_keys,
            dim=args.dim,
            candidate_k=args.candidate_k,
            jitter=args.jitter,
            seed=args.seed,
            normalize=args.normalize,
            attack_fraction=args.attack_fraction,
            attack_magnitude=args.attack_magnitude,
            attack_dim=args.attack_dim,
            query_attack_magnitude=args.query_attack_magnitude,
        )

    queries = dataset["queries"]
    keys = dataset["keys"]
    candidates = dataset["candidates"]

    print("\n[DATA]")
    print(f"  queries       : {queries.shape}")
    print(f"  keys          : {keys.shape}")
    print(f"  candidates    : {candidates.shape}")
    print(f"  attacked keys : {int(np.sum(dataset['attacked_keys']))} / {len(dataset['attacked_keys'])}")
    if args.sweep:
        print("  [sweep note]  : this initial dataset is only a smoke-load; sweep points build their own datasets")

    if args.save_dataset:
        np.savez_compressed(out_dir / "dataset.npz", **dataset)

    print("\n[TABLES]")
    gpu_sm_meta = None
    qpu_sm_meta = None

    if args.gpu_base:
        load_projection_table.last_sm_meta = None
        gpu_table, gpu_key = load_projection_table(Path(args.gpu_base), args.bins)
        gpu_sm_meta = load_projection_table.last_sm_meta
        print(f"  GPU table     : {args.gpu_base} [{gpu_key}] shape={gpu_table.shape}")
        if gpu_sm_meta:
            print(
                "                 "
                f"S_M agreement={gpu_sm_meta['mean_agreement']:.4f}, "
                f"det={gpu_sm_meta['mean_detection_rate']:.4f}, "
                f"trace={gpu_sm_meta['mean_trace']:.4f}"
            )
    else:
        _, gpu_table, _tmp_qpu = synthetic_projection_tables(
            bins=args.bins,
            seed=args.seed + 101,
            gpu_noise=args.gpu_noise,
            qpu_noise=args.qpu_noise,
            qpu_attenuation=args.qpu_attenuation,
        )
        print(f"  GPU table     : synthetic shape={gpu_table.shape}")

    if args.qpu_base:
        load_projection_table.last_sm_meta = None
        qpu_table, qpu_key = load_projection_table(Path(args.qpu_base), args.bins)
        qpu_sm_meta = load_projection_table.last_sm_meta
        print(f"  QPU table     : {args.qpu_base} [{qpu_key}] shape={qpu_table.shape}")
        if qpu_sm_meta:
            print(
                "                 "
                f"S_M agreement={qpu_sm_meta['mean_agreement']:.4f}, "
                f"det={qpu_sm_meta['mean_detection_rate']:.4f}, "
                f"trace={qpu_sm_meta['mean_trace']:.4f}"
            )
    else:
        _, _tmp_gpu, qpu_table = synthetic_projection_tables(
            bins=args.bins,
            seed=args.seed + 202,
            gpu_noise=args.gpu_noise,
            qpu_noise=args.qpu_noise,
            qpu_attenuation=args.qpu_attenuation,
        )
        print(f"  QPU table     : synthetic shape={qpu_table.shape}")

    np.savez_compressed(
        out_dir / "projection_tables.npz",
        gpu_table=gpu_table,
        qpu_table=qpu_table,
    )

    if gpu_sm_meta or qpu_sm_meta:
        with open(out_dir / "projection_base_metadata.json", "w", encoding="utf-8") as f:
            json.dump(json_safe({"gpu_sm_meta": gpu_sm_meta, "qpu_sm_meta": qpu_sm_meta}), f, indent=2)

    if args.sweep:
        run_sweep(
            args=args,
            out_dir=out_dir,
            gpu_table=gpu_table,
            qpu_table=qpu_table,
            gpu_sm_meta=gpu_sm_meta,
            qpu_sm_meta=qpu_sm_meta,
            observer_start=observer_start,
        )
        return

    scale = phase_scale(queries, keys)
    include_field = not args.no_field

    summaries: List[dict] = []
    per_query_rows: List[dict] = []
    all_scores: Dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------------
    # Scoring backends
    # ---------------------------------------------------------------------
    # Every backend receives the same:
    #     queries, keys, true_ids, candidates
    #
    # This is the central honesty constraint. A backend may score candidates
    # differently, but it may not change the retrieval problem.
    # ---------------------------------------------------------------------
    print("\n[SCORING] cosine baseline")
    t0 = time.time()
    cosine_scores = cosine_candidate_scores(queries, keys, candidates, args.batch)
    cosine_seconds = time.time() - t0

    summ, pq = evaluate_backend(
        name="cosine",
        scores=cosine_scores,
        true_pos=dataset["true_pos"],
        candidates=candidates,
        attacked_keys=dataset["attacked_keys"],
        reference_scores=None,
        seconds=cosine_seconds,
    )
    summaries.append(summ)
    per_query_rows.extend(pq)
    all_scores["cosine"] = cosine_scores

    print("[SCORING] dot baseline")
    s, pq, sc = run_backend(
        name="dot",
        score_fn=lambda: dot_candidate_scores(queries, keys, candidates, args.batch),
        dataset=dataset,
        cosine_ref=cosine_scores,
        field_weights=[],
        include_field=False,
    )
    summaries.extend(s)
    per_query_rows.extend(pq)
    all_scores.update(sc)

    print("[SCORING] geo_projected")
    s, pq, sc = run_backend(
        name="geo_projected",
        score_fn=lambda: geo_projected_scores(queries, keys, candidates, scale, args.batch),
        dataset=dataset,
        cosine_ref=cosine_scores,
        field_weights=args.field_weights,
        include_field=include_field,
    )
    summaries.extend(s)
    per_query_rows.extend(pq)
    all_scores.update(sc)

    print("[SCORING] gpu_projected")
    s, pq, sc = run_backend(
        name="gpu_projected",
        score_fn=lambda: table_projected_scores(queries, keys, candidates, gpu_table, scale, args.batch),
        dataset=dataset,
        cosine_ref=cosine_scores,
        field_weights=args.field_weights,
        include_field=include_field,
    )
    summaries.extend(s)
    per_query_rows.extend(pq)
    all_scores.update(sc)

    print("[SCORING] qpu_projected")
    s, pq, sc = run_backend(
        name="qpu_projected",
        score_fn=lambda: table_projected_scores(queries, keys, candidates, qpu_table, scale, args.batch),
        dataset=dataset,
        cosine_ref=cosine_scores,
        field_weights=args.field_weights,
        include_field=include_field,
    )
    summaries.extend(s)
    per_query_rows.extend(pq)
    all_scores.update(sc)

    print_summary(summaries)

    write_csv(out_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    if args.write_per_query:
        write_csv(out_dir / "per_query.csv", per_query_rows, PER_QUERY_FIELDS)

    # Save top-level score summaries, not the full dense score matrices by default.
    observer_record = None
    if observer_start is not None:
        observer_end = bright_observer_snapshot("benchmark_end")
        observer_record = build_bright_observer_record(
            start=observer_start,
            end=observer_end,
            args=args,
            out_dir=out_dir,
        )
        with open(out_dir / "bright_observer.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(observer_record), f, indent=2)

    payload = {
        "claim": {
            "cosine": "classical retrieval control",
            "geo_projected": "analytical bounded projection coordinate",
            "gpu_projected": "projection-table score using noiseless/GPU-style base",
            "qpu_projected": "projection-table score using QPU-style base",
            "field_backends": "retrieval-rank field deformation; rankΔ reports ordering change",
        },
        "args": vars(args),
        "observer": observer_record,
        "dataset": {
            "queries": list(queries.shape),
            "keys": list(keys.shape),
            "candidates": list(candidates.shape),
            "attacked_keys": int(np.sum(dataset["attacked_keys"])),
        },
        "tables": {
            "gpu_shape": list(gpu_table.shape),
            "qpu_shape": list(qpu_table.shape),
            "gpu_base": args.gpu_base,
            "qpu_base": args.qpu_base,
            "gpu_sm_meta": gpu_sm_meta,
            "qpu_sm_meta": qpu_sm_meta,
        },
        "summary": summaries,
        "total_seconds": time.time() - t_all,
    }

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)

    print(f"\n[SAVED] {out_dir}")
    print(f"{'=' * 128}\n")


if __name__ == "__main__":
    main()
