#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GHOST ORACLE SUITE — D_M PROBE 00: PRUNE FAILED ASSUMPTIONS
================================================================================

Purpose
-------
Load one frozen D_M QPU scene and prune failed assumptions from the model.

This probe does NOT define D_M.
This probe does NOT claim the dimensional operator has been found.
This probe asks which assumptions survive the first frozen QPU record.

The one-shot D_M QPU generator intentionally embedded multiple assumptions:

    smooth_walk       : nearest-neighbor dimensional transport
    boundary_reflect  : edge/boundary pressure or reflection
    nonlocal_jump     : long-range dimensional hop
    collapse_gate     : controlled dimensional collapse
    phase_shear       : delay/phase changes the map
    scramble_order    : scrambled order should weaken/destroy ordered structure
    mirror_parity     : mirrored parity symmetry should survive
    rank_spread       : effective spread is more stable than raw bit patterns

Probe logic
-----------
For each tile:

    1. Load raw registers:
        ctrl, dim[4], edge, aux, meta

    2. Build basic observable fields:
        dim_state       : integer 0..15
        state histogram : probability over 16 states
        popcount        : active dim-bit count
        parity          : XOR over dim bits
        boundary        : d0 XOR d3
        interior        : d1 XOR d2

    3. Score assumptions using effect-size style tests:
        - state structure vs uniform
        - effective dimensionality
        - collapse strength
        - boundary/interior split
        - parity bias
        - adjacent vs non-adjacent bit coupling
        - ctrl/edge/aux association with dim_state and popcount
        - repeat consistency for modes that appear more than once

    4. Classify each tile:
        SURVIVE, WEAK, FAIL, or MUTATED

    5. Summarize by mode and prune:
        - keep modes with clear surviving or mutated signal
        - prune modes that are indistinguishable/noisy/failed
        - flag assumptions where hardware response contradicts expectation

Important
---------
The thresholds are intentionally simple and transparent. They are not final
statistics. They are a first-pass pruning blade. The output is designed to guide
the next probe, not to make the final claim.

Usage
-----
    python dm_probe_00_prune.py

    python dm_probe_00_prune.py --input data/dm_job_d8fb033o3njc73f01170.npz

    python dm_probe_00_prune.py --input data/dm_job_d8fb033o3njc73f01170.npz --out analysis/my_prune

================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
ANALYSIS_DIR = HERE / "analyze"


# =============================================================================
# THRESHOLDS
# =============================================================================

# Chi-square z-score against uniform 16-state distribution.
# df = 15. z ~= (chi2 - df) / sqrt(2df)
STRUCTURED_Z_WEAK = 2.0
STRUCTURED_Z_STRONG = 4.0

# Effective dimension thresholds over 16 possible states.
# These are deliberately loose for first prune.
EFF_DIM_COLLAPSED = 3.0
EFF_DIM_LOW = 5.0
EFF_DIM_MED = 8.0
EFF_DIM_HIGH = 10.0

# Bias thresholds for Bernoulli observables like parity/boundary/ctrl.
BIAS_WEAK = 0.08
BIAS_STRONG = 0.16

# Correlation thresholds.
CORR_WEAK = 0.05
CORR_STRONG = 0.12

# Repeat-consistency threshold. Lower distance means two repeated modes agree.
REPEAT_L1_GOOD = 0.35
REPEAT_L1_WEAK = 0.60


# =============================================================================
# HELPERS
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
    return x


def read_latest_pointer() -> Path:
    ptr = DATA_DIR / "latest_dm_qpu_data.json"
    if not ptr.exists():
        raise FileNotFoundError(
            f"No latest pointer found: {ptr}\n"
            "Pass --input path/to/dm_job_<JOB_ID>.npz explicitly."
        )

    with open(ptr, "r", encoding="utf-8") as f:
        meta = json.load(f)

    path = Path(meta["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Latest pointer exists but target file is missing:\n{path}"
        )

    return path


def load_npz_scalar(x: Any) -> Any:
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    return x


def safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def safe_std(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.std(x))


def bernoulli_bias(p: float) -> float:
    return abs(float(p) - 0.5)


def binary_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation for binary vectors.
    Returns 0 if either side is constant.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size:
        raise ValueError("binary_corr arrays must have same size")
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    return float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / (sa * sb))


def eta_squared_categorical_binary_group(values: np.ndarray, group: np.ndarray) -> float:
    """
    Association strength between numeric values and binary group.
    Equivalent to eta^2 for two groups.
    Returns 0 if degenerate.
    """
    values = np.asarray(values, dtype=np.float64)
    group = np.asarray(group, dtype=np.uint8)

    if values.size == 0:
        return 0.0

    overall = float(np.mean(values))
    total_ss = float(np.sum((values - overall) ** 2))
    if total_ss <= 1e-12:
        return 0.0

    between = 0.0
    for g in [0, 1]:
        mask = group == g
        if np.any(mask):
            group_mean = float(np.mean(values[mask]))
            between += float(np.sum(mask)) * (group_mean - overall) ** 2

    return float(between / total_ss)


def entropy_base2(prob: np.ndarray) -> float:
    p = np.asarray(prob, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def effective_dimension(prob: np.ndarray) -> float:
    p = np.asarray(prob, dtype=np.float64)
    denom = float(np.sum(p * p))
    if denom <= 1e-15:
        return 0.0
    return float(1.0 / denom)


def chi_square_uniform_z(hist: np.ndarray) -> Tuple[float, float]:
    """
    Return chi-square statistic and rough normal z-score against uniform.
    Avoid scipy dependency.

    For k=16 states:
        df = 15
        z ~= (chi2 - df) / sqrt(2df)
    """
    h = np.asarray(hist, dtype=np.float64)
    n = float(np.sum(h))
    k = int(h.size)
    if n <= 0 or k <= 1:
        return 0.0, 0.0

    expected = n / k
    chi2 = float(np.sum((h - expected) ** 2 / max(expected, 1e-12)))
    df = k - 1
    z = (chi2 - df) / math.sqrt(max(2.0 * df, 1e-12))
    return chi2, float(z)


def classify_strength(value: float, weak: float, strong: float) -> str:
    av = abs(float(value))
    if av >= strong:
        return "strong"
    if av >= weak:
        return "weak"
    return "none"


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64))))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TileMetrics:
    tile: int
    mode: str
    assumption: str
    role: str
    theta: float
    delay_dt: int
    scale_level: int
    shots: int

    mean_state: float
    mean_popcount: float
    mean_ctrl: float
    mean_edge: float
    mean_aux: float
    mean_meta: float
    mean_parity: float
    mean_boundary: float
    mean_interior: float

    state_entropy_bits: float
    effective_dim: float
    max_state_prob: float
    active_states_1pct: int
    chi2_uniform: float
    chi2_uniform_z: float

    bit0_mean: float
    bit1_mean: float
    bit2_mean: float
    bit3_mean: float

    adjacent_corr_mean_abs: float
    nonadjacent_corr_mean_abs: float
    adjacent_minus_nonadjacent: float

    boundary_minus_interior: float
    parity_bias: float
    boundary_bias: float
    ctrl_bias: float
    edge_bias: float
    aux_bias: float

    ctrl_eta_state: float
    ctrl_eta_popcount: float
    edge_eta_state: float
    edge_eta_popcount: float
    aux_eta_state: float
    aux_eta_popcount: float

    raw_class: str
    prune_vote: str
    reason: str


# =============================================================================
# LOAD TILE ARRAYS
# =============================================================================

def get_tile_arrays(npz: Any, tile: int) -> Dict[str, np.ndarray]:
    """
    Load tile arrays with fallback to stacked arrays when per-tile keys are absent.
    """
    out: Dict[str, np.ndarray] = {}

    def get(name: str) -> np.ndarray:
        per_key = f"{name}_tile{tile}"
        if per_key in npz.files:
            return np.asarray(npz[per_key])
        if name in npz.files:
            return np.asarray(npz[name][tile])
        raise KeyError(f"missing {name} for tile {tile}")

    has_ctrl = f"ctrl_tile{tile}" in npz.files or "ctrl" in npz.files
    if has_ctrl:
        out["ctrl"] = get("ctrl").astype(np.uint8)
        out["dim"] = get("dim").astype(np.uint8)
        out["edge"] = get("edge").astype(np.uint8)
        out["aux"] = get("aux").astype(np.uint8)
        out["meta"] = get("meta").astype(np.uint8)
    else:
        pair_key = f"pair_tile{tile}"
        pair = np.asarray(npz[pair_key] if pair_key in npz.files else npz["pair"][tile]).astype(np.uint8)
        q0, q1 = pair[:, 0], pair[:, 1]
        p00 = ((q0 == 0) & (q1 == 0)).astype(np.uint8)
        p01 = ((q0 == 0) & (q1 == 1)).astype(np.uint8)
        p10 = ((q0 == 1) & (q1 == 0)).astype(np.uint8)
        p11 = ((q0 == 1) & (q1 == 1)).astype(np.uint8)
        out["ctrl"] = q0
        out["dim"] = np.column_stack([p00, p01, p10, p11])
        out["edge"] = np.bitwise_xor(q0, q1)
        out["aux"] = q1
        out["meta"] = q0

    if f"dim_state_tile{tile}" in npz.files:
        out["dim_state"] = np.asarray(npz[f"dim_state_tile{tile}"]).astype(np.uint8)
    elif "dim_state" in npz.files:
        out["dim_state"] = np.asarray(npz["dim_state"][tile]).astype(np.uint8)
    else:
        weights = np.asarray([1, 2, 4, 8], dtype=np.uint8)
        out["dim_state"] = (out["dim"] * weights[None, :]).sum(axis=1).astype(np.uint8)

    out["popcount"] = out["dim"].sum(axis=1).astype(np.uint8)
    out["parity"] = np.bitwise_xor.reduce(out["dim"], axis=1).astype(np.uint8)
    out["boundary"] = np.bitwise_xor(out["dim"][:, 0], out["dim"][:, 3]).astype(np.uint8)
    out["interior"] = np.bitwise_xor(out["dim"][:, 1], out["dim"][:, 2]).astype(np.uint8)

    return out


def get_metadata(npz: Any, tile: int) -> Dict[str, Any]:
    def arr_value(name: str, default: Any) -> Any:
        if name not in npz.files:
            return default
        arr = np.asarray(npz[name])
        if arr.ndim == 0:
            return arr.item()
        if tile < len(arr):
            v = arr[tile]
            if isinstance(v, np.generic):
                return v.item()
            return v
        return default

    return {
        "mode": str(arr_value("tile_mode", "unknown")),
        "assumption": str(arr_value("tile_assumption", "unknown")),
        "role": str(arr_value("tile_role", "unknown")),
        "theta": float(arr_value("tile_theta", float("nan"))),
        "delay_dt": int(arr_value("tile_delay_dt", -1)),
        "scale_level": int(arr_value("tile_scale_level", -1)),
    }


# =============================================================================
# TILE SCORING
# =============================================================================

def score_tile(npz: Any, tile: int) -> Tuple[TileMetrics, np.ndarray]:
    arrays = get_tile_arrays(npz, tile)
    meta = get_metadata(npz, tile)

    ctrl = arrays["ctrl"]
    dim = arrays["dim"]
    edge = arrays["edge"]
    aux = arrays["aux"]
    meta_bit = arrays["meta"]
    state = arrays["dim_state"]
    pop = arrays["popcount"]
    parity = arrays["parity"]
    boundary = arrays["boundary"]
    interior = arrays["interior"]

    shots = int(state.shape[0])

    hist = np.bincount(state.astype(np.int64), minlength=16).astype(np.float64)
    prob = hist / max(1.0, float(np.sum(hist)))

    ent = entropy_base2(prob)
    eff = effective_dimension(prob)
    maxp = float(np.max(prob))
    active_1pct = int(np.sum(prob >= 0.01))
    chi2, z = chi_square_uniform_z(hist)

    bit_means = [float(np.mean(dim[:, i])) for i in range(4)]

    adjacent_corrs = [
        abs(binary_corr(dim[:, 0], dim[:, 1])),
        abs(binary_corr(dim[:, 1], dim[:, 2])),
        abs(binary_corr(dim[:, 2], dim[:, 3])),
    ]
    nonadjacent_corrs = [
        abs(binary_corr(dim[:, 0], dim[:, 2])),
        abs(binary_corr(dim[:, 0], dim[:, 3])),
        abs(binary_corr(dim[:, 1], dim[:, 3])),
    ]

    adj_mean = float(np.mean(adjacent_corrs))
    nonadj_mean = float(np.mean(nonadjacent_corrs))
    adj_minus = adj_mean - nonadj_mean

    mean_ctrl = safe_mean(ctrl)
    mean_edge = safe_mean(edge)
    mean_aux = safe_mean(aux)
    mean_meta = safe_mean(meta_bit)
    mean_parity = safe_mean(parity)
    mean_boundary = safe_mean(boundary)
    mean_interior = safe_mean(interior)

    boundary_minus_interior = mean_boundary - mean_interior

    ctrl_eta_state = eta_squared_categorical_binary_group(state, ctrl)
    ctrl_eta_pop = eta_squared_categorical_binary_group(pop, ctrl)
    edge_eta_state = eta_squared_categorical_binary_group(state, edge)
    edge_eta_pop = eta_squared_categorical_binary_group(pop, edge)
    aux_eta_state = eta_squared_categorical_binary_group(state, aux)
    aux_eta_pop = eta_squared_categorical_binary_group(pop, aux)

    raw_class, prune_vote, reason = classify_tile(
        mode=meta["mode"],
        effective_dim_value=eff,
        chi2_z=z,
        max_state_prob=maxp,
        adjacent_minus_nonadjacent=adj_minus,
        boundary_minus_interior=boundary_minus_interior,
        parity_bias_value=bernoulli_bias(mean_parity),
        boundary_bias_value=bernoulli_bias(mean_boundary),
        ctrl_eta_state=ctrl_eta_state,
        ctrl_eta_pop=ctrl_eta_pop,
        edge_eta_state=edge_eta_state,
        edge_eta_pop=edge_eta_pop,
        aux_eta_state=aux_eta_state,
        aux_eta_pop=aux_eta_pop,
    )

    metrics = TileMetrics(
        tile=tile,
        mode=meta["mode"],
        assumption=meta["assumption"],
        role=meta["role"],
        theta=meta["theta"],
        delay_dt=meta["delay_dt"],
        scale_level=meta["scale_level"],
        shots=shots,

        mean_state=safe_mean(state),
        mean_popcount=safe_mean(pop),
        mean_ctrl=mean_ctrl,
        mean_edge=mean_edge,
        mean_aux=mean_aux,
        mean_meta=mean_meta,
        mean_parity=mean_parity,
        mean_boundary=mean_boundary,
        mean_interior=mean_interior,

        state_entropy_bits=ent,
        effective_dim=eff,
        max_state_prob=maxp,
        active_states_1pct=active_1pct,
        chi2_uniform=chi2,
        chi2_uniform_z=z,

        bit0_mean=bit_means[0],
        bit1_mean=bit_means[1],
        bit2_mean=bit_means[2],
        bit3_mean=bit_means[3],

        adjacent_corr_mean_abs=adj_mean,
        nonadjacent_corr_mean_abs=nonadj_mean,
        adjacent_minus_nonadjacent=adj_minus,

        boundary_minus_interior=boundary_minus_interior,
        parity_bias=bernoulli_bias(mean_parity),
        boundary_bias=bernoulli_bias(mean_boundary),
        ctrl_bias=bernoulli_bias(mean_ctrl),
        edge_bias=bernoulli_bias(mean_edge),
        aux_bias=bernoulli_bias(mean_aux),

        ctrl_eta_state=ctrl_eta_state,
        ctrl_eta_popcount=ctrl_eta_pop,
        edge_eta_state=edge_eta_state,
        edge_eta_popcount=edge_eta_pop,
        aux_eta_state=aux_eta_state,
        aux_eta_popcount=aux_eta_pop,

        raw_class=raw_class,
        prune_vote=prune_vote,
        reason=reason,
    )

    return metrics, prob


def classify_tile(
    mode: str,
    effective_dim_value: float,
    chi2_z: float,
    max_state_prob: float,
    adjacent_minus_nonadjacent: float,
    boundary_minus_interior: float,
    parity_bias_value: float,
    boundary_bias_value: float,
    ctrl_eta_state: float,
    ctrl_eta_pop: float,
    edge_eta_state: float,
    edge_eta_pop: float,
    aux_eta_state: float,
    aux_eta_pop: float,
) -> Tuple[str, str, str]:
    """
    First-pass mode-specific pruning classifier.

    raw_class describes the observed behavior.
    prune_vote says whether the original assumption survives.
    """
    structured = chi2_z >= STRUCTURED_Z_WEAK
    strongly_structured = chi2_z >= STRUCTURED_Z_STRONG
    collapsed = effective_dim_value <= EFF_DIM_COLLAPSED or max_state_prob >= 0.55
    low_dim = effective_dim_value <= EFF_DIM_LOW
    high_spread = effective_dim_value >= EFF_DIM_MED

    ctrl_assoc = max(ctrl_eta_state, ctrl_eta_pop)
    edge_assoc = max(edge_eta_state, edge_eta_pop)
    aux_assoc = max(aux_eta_state, aux_eta_pop)

    association_max = max(ctrl_assoc, edge_assoc, aux_assoc)
    boundary_signal = abs(boundary_minus_interior) >= BIAS_WEAK or boundary_bias_value >= BIAS_WEAK
    parity_signal = parity_bias_value >= BIAS_WEAK
    adjacent_signal = adjacent_minus_nonadjacent >= CORR_WEAK
    nonlocal_signal = aux_assoc >= CORR_WEAK or abs(adjacent_minus_nonadjacent) >= CORR_WEAK

    if not structured and association_max < CORR_WEAK and not boundary_signal and not parity_signal:
        return (
            "NO_CLEAR_STRUCTURE",
            "FAIL",
            "distribution near uniform/no clear control, boundary, parity, or association signal",
        )

    if mode == "smooth_walk":
        if adjacent_signal and structured:
            return (
                "ORDERED_LOCAL_STRUCTURE",
                "SURVIVE",
                "adjacent coupling exceeds non-adjacent coupling and state distribution is structured",
            )
        if structured and not adjacent_signal:
            return (
                "STRUCTURED_BUT_NOT_LOCAL_WALK",
                "MUTATED",
                "state structure survives but not as a clean nearest-neighbor walk",
            )
        return (
            "WEAK_LOCAL_WALK",
            "WEAK",
            "some structure present but nearest-neighbor assumption is not clean",
        )

    if mode == "boundary_reflect":
        if boundary_signal and structured:
            return (
                "BOUNDARY_ACTIVE",
                "SURVIVE",
                "boundary/interior or boundary bias signal present in structured distribution",
            )
        if structured and not boundary_signal:
            return (
                "STRUCTURED_WITHOUT_BOUNDARY",
                "MUTATED",
                "distribution structured but boundary-reflection assumption is not primary",
            )
        return (
            "WEAK_BOUNDARY",
            "WEAK",
            "boundary behavior not strong enough for first-pass survival",
        )

    if mode == "nonlocal_jump":
        if nonlocal_signal and structured:
            return (
                "NONLOCAL_OR_AUX_LINKED_STRUCTURE",
                "SURVIVE",
                "aux/nonlocal association or non-adjacent structure appears in record",
            )
        if structured and not nonlocal_signal:
            return (
                "STRUCTURED_NOT_NONLOCAL",
                "MUTATED",
                "state structure survives but not through obvious nonlocal/aux channel",
            )
        return (
            "WEAK_NONLOCAL",
            "WEAK",
            "nonlocal assumption not strongly supported",
        )

    if mode == "collapse_gate":
        if collapsed or low_dim:
            return (
                "COLLAPSED_LOW_DIMENSION",
                "SURVIVE",
                "effective dimension is low or max state probability indicates collapse",
            )
        if structured and not low_dim:
            return (
                "STRUCTURED_NOT_COLLAPSED",
                "MUTATED",
                "controlled structure appears but not as dimensional collapse",
            )
        return (
            "WEAK_COLLAPSE",
            "WEAK",
            "collapse-gate assumption not cleanly supported",
        )

    if mode == "phase_shear":
        if structured and (association_max >= CORR_WEAK or low_dim or boundary_signal):
            return (
                "PHASE_SHEAR_ALTERS_STRUCTURE",
                "SURVIVE",
                "structured low/spread-shifted response with control/edge/aux association",
            )
        if structured:
            return (
                "STRUCTURED_PHASE_RESPONSE_UNKNOWN",
                "MUTATED",
                "phase-shear tile structured but mechanism unclear",
            )
        return (
            "WEAK_PHASE_SHEAR",
            "WEAK",
            "phase-shear assumption not cleanly supported",
        )

    if mode == "scramble_order":
        if structured and not high_spread:
            return (
                "SCRAMBLE_DID_NOT_DESTROY_STRUCTURE",
                "MUTATED",
                "scrambled path still produced structured non-uniform record",
            )
        if not structured:
            return (
                "SCRAMBLE_DESTROYED_STRUCTURE",
                "SURVIVE",
                "scramble behaved like destructive control",
            )
        return (
            "SCRAMBLE_AMBIGUOUS",
            "WEAK",
            "scramble effect is neither clearly destructive nor clearly structured",
        )

    if mode == "mirror_parity":
        if parity_signal and structured:
            return (
                "MIRROR_PARITY_ACTIVE",
                "SURVIVE",
                "parity bias survives in structured record",
            )
        if structured and not parity_signal:
            return (
                "STRUCTURED_NOT_PARITY",
                "MUTATED",
                "mirror tile structured but parity symmetry is not the dominant signal",
            )
        return (
            "WEAK_MIRROR_PARITY",
            "WEAK",
            "mirror/parity assumption not cleanly supported",
        )

    if mode == "rank_spread":
        if high_spread and structured:
            return (
                "HIGH_EFFECTIVE_SPREAD",
                "SURVIVE",
                "effective dimension/spread remains high in structured distribution",
            )
        if structured and not high_spread:
            return (
                "STRUCTURED_BUT_NOT_HIGH_SPREAD",
                "MUTATED",
                "rank/spread assumption changed into lower-dimensional structure",
            )
        return (
            "WEAK_RANK_SPREAD",
            "WEAK",
            "rank/spread assumption not cleanly supported",
        )

    if strongly_structured:
        return (
            "UNKNOWN_STRONG_STRUCTURE",
            "MUTATED",
            "unknown mode produced strong structure; keep for investigation",
        )

    return (
        "UNKNOWN_WEAK",
        "WEAK",
        "unknown mode with weak/ambiguous structure",
    )


# =============================================================================
# MODE SUMMARY / PRUNING
# =============================================================================

def summarize_modes(metrics: List[TileMetrics], probs_by_tile: Dict[int, np.ndarray]) -> List[Dict[str, Any]]:
    by_mode: Dict[str, List[TileMetrics]] = {}
    for m in metrics:
        by_mode.setdefault(m.mode, []).append(m)

    rows: List[Dict[str, Any]] = []

    for mode, items in sorted(by_mode.items()):
        votes = [m.prune_vote for m in items]
        classes = [m.raw_class for m in items]

        survive_n = votes.count("SURVIVE")
        mutate_n = votes.count("MUTATED")
        weak_n = votes.count("WEAK")
        fail_n = votes.count("FAIL")

        eff_mean = float(np.mean([m.effective_dim for m in items]))
        z_mean = float(np.mean([m.chi2_uniform_z for m in items]))
        boundary_mean = float(np.mean([m.mean_boundary for m in items]))
        parity_mean = float(np.mean([m.mean_parity for m in items]))
        pop_mean = float(np.mean([m.mean_popcount for m in items]))

        repeat_l1 = float("nan")
        repeat_status = "single"
        if len(items) >= 2:
            dists: List[float] = []
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    p = probs_by_tile[items[i].tile]
                    q = probs_by_tile[items[j].tile]
                    dists.append(l1_distance(p, q))
            repeat_l1 = float(np.mean(dists))
            if repeat_l1 <= REPEAT_L1_GOOD:
                repeat_status = "consistent"
            elif repeat_l1 <= REPEAT_L1_WEAK:
                repeat_status = "weakly_consistent"
            else:
                repeat_status = "inconsistent"

        # Mode-level decision.
        if survive_n > 0:
            decision = "KEEP"
            decision_reason = "at least one tile directly supports the embedded assumption"
        elif mutate_n > 0:
            decision = "KEEP_AS_MUTATION"
            decision_reason = "assumption failed, but structured mutation survived"
        elif weak_n > 0 and fail_n == 0:
            decision = "HOLD"
            decision_reason = "weak/ambiguous; do not build on it yet"
        else:
            decision = "PRUNE"
            decision_reason = "no usable signal under first-pass criteria"

        rows.append({
            "mode": mode,
            "n_tiles": len(items),
            "survive_n": survive_n,
            "mutate_n": mutate_n,
            "weak_n": weak_n,
            "fail_n": fail_n,
            "decision": decision,
            "decision_reason": decision_reason,
            "repeat_l1": repeat_l1,
            "repeat_status": repeat_status,
            "effective_dim_mean": eff_mean,
            "chi2_uniform_z_mean": z_mean,
            "boundary_mean": boundary_mean,
            "parity_mean": parity_mean,
            "popcount_mean": pop_mean,
            "classes": ";".join(classes),
            "votes": ";".join(votes),
        })

    return rows


def build_pruned_json(
    input_path: Path,
    npz: Any,
    metrics: List[TileMetrics],
    mode_summary: List[Dict[str, Any]],
) -> Dict[str, Any]:
    job_id = str(load_npz_scalar(npz["job_id"])) if "job_id" in npz.files else "unknown"
    backend = str(load_npz_scalar(npz["backend"])) if "backend" in npz.files else "unknown"

    keep = [r["mode"] for r in mode_summary if r["decision"] == "KEEP"]
    keep_mutation = [r["mode"] for r in mode_summary if r["decision"] == "KEEP_AS_MUTATION"]
    hold = [r["mode"] for r in mode_summary if r["decision"] == "HOLD"]
    prune = [r["mode"] for r in mode_summary if r["decision"] == "PRUNE"]

    tile_votes = [
        {
            "tile": m.tile,
            "mode": m.mode,
            "assumption": m.assumption,
            "vote": m.prune_vote,
            "class": m.raw_class,
            "reason": m.reason,
            "effective_dim": m.effective_dim,
            "chi2_uniform_z": m.chi2_uniform_z,
            "mean_popcount": m.mean_popcount,
            "mean_boundary": m.mean_boundary,
            "mean_parity": m.mean_parity,
        }
        for m in metrics
    ]

    return {
        "schema": "ghost_oracle.dm.probe_00_prune.v1",
        "operator": "D_M",
        "probe": "dm_probe_00_prune",
        "input": str(input_path),
        "job_id": job_id,
        "backend": backend,
        "created": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "keep": keep,
            "keep_as_mutation": keep_mutation,
            "hold": hold,
            "prune": prune,
        },
        "mode_summary": mode_summary,
        "tile_votes": tile_votes,
        "note": (
            "This is a first-pass pruning report. KEEP means the original assumption "
            "has support. KEEP_AS_MUTATION means the assumption failed but produced "
            "structured behavior worth following. PRUNE means no usable first-pass "
            "signal. HOLD means ambiguous."
        ),
    }


# =============================================================================
# OUTPUT WRITERS
# =============================================================================

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_report(
    path: Path,
    input_path: Path,
    npz: Any,
    metrics: List[TileMetrics],
    mode_summary: List[Dict[str, Any]],
    pruned: Dict[str, Any],
) -> None:
    job_id = pruned["job_id"]
    backend = pruned["backend"]

    lines: List[str] = []
    lines.append("# D_M Probe 00 — Prune Failed Assumptions")
    lines.append("")
    lines.append("## Scene")
    lines.append("")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Job ID: `{job_id}`")
    lines.append(f"- Backend: `{backend}`")
    lines.append(f"- Tiles: `{len(metrics)}`")
    lines.append("")
    lines.append("## Probe Rule")
    lines.append("")
    lines.append(
        "This probe does not define `D_M`. It prunes the assumptions embedded in "
        "the frozen QPU scene. Raw records remain canonical; derived fields are "
        "only used for first-pass analysis."
    )
    lines.append("")
    lines.append("## Mode Decisions")
    lines.append("")
    lines.append("| Mode | Decision | Tiles | Repeat | EffDim mean | z(uniform) mean | Reason |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in mode_summary:
        repeat = r["repeat_status"]
        eff = f"{r['effective_dim_mean']:.3f}"
        z = f"{r['chi2_uniform_z_mean']:.3f}"
        lines.append(
            f"| `{r['mode']}` | **{r['decision']}** | {r['n_tiles']} | "
            f"{repeat} | {eff} | {z} | {r['decision_reason']} |"
        )

    lines.append("")
    lines.append("## Pruned Model")
    lines.append("")
    lines.append(f"- KEEP: `{', '.join(pruned['summary']['keep']) or 'none'}`")
    lines.append(f"- KEEP_AS_MUTATION: `{', '.join(pruned['summary']['keep_as_mutation']) or 'none'}`")
    lines.append(f"- HOLD: `{', '.join(pruned['summary']['hold']) or 'none'}`")
    lines.append(f"- PRUNE: `{', '.join(pruned['summary']['prune']) or 'none'}`")
    lines.append("")
    lines.append("## Tile-Level Votes")
    lines.append("")
    lines.append("| Tile | Mode | Vote | Class | EffDim | Pop | Boundary | Parity | Reason |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---|")
    for m in metrics:
        lines.append(
            f"| {m.tile} | `{m.mode}` | **{m.prune_vote}** | `{m.raw_class}` | "
            f"{m.effective_dim:.3f} | {m.mean_popcount:.3f} | "
            f"{m.mean_boundary:.3f} | {m.mean_parity:.3f} | {m.reason} |"
        )

    lines.append("")
    lines.append("## Next Probe Direction")
    lines.append("")
    lines.append(
        "Follow `KEEP` modes directly. Follow `KEEP_AS_MUTATION` modes as hardware "
        "disagreement signals. Do not build future logic on `PRUNE` modes unless a "
        "later control proves this first-pass classifier was too aggressive."
    )
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def maybe_make_plots(
    out_dir: Path,
    metrics: List[TileMetrics],
    probs_by_tile: Dict[int, np.ndarray],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable; skipping plots: {e}")
        return

    tiles = [m.tile for m in metrics]
    modes = [m.mode for m in metrics]

    # Heatmap: state probability by tile.
    mat = np.stack([probs_by_tile[m.tile] for m in metrics], axis=0)
    plt.figure(figsize=(12, max(5, 0.45 * len(metrics))))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="P(state)")
    plt.xlabel("dim_state 0..15")
    plt.ylabel("tile")
    plt.yticks(range(len(metrics)), [f"{m.tile}:{m.mode}" for m in metrics], fontsize=8)
    plt.title("D_M Probe 00 — Dimensional State Probability Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "dm_state_prob_heatmap.png", dpi=160)
    plt.close()

    # Effective dimension.
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(metrics)), [m.effective_dim for m in metrics])
    plt.xticks(range(len(metrics)), [f"{m.tile}\n{m.mode}" for m in metrics], rotation=45, ha="right", fontsize=8)
    plt.ylabel("Effective dimension")
    plt.title("D_M Probe 00 — Effective Dimension by Tile")
    plt.tight_layout()
    plt.savefig(out_dir / "dm_effective_dimension_by_tile.png", dpi=160)
    plt.close()

    # Boundary/parity.
    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, [m.mean_boundary for m in metrics], width, label="boundary")
    plt.bar(x + width / 2, [m.mean_parity for m in metrics], width, label="parity")
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xticks(range(len(metrics)), [f"{m.tile}\n{m.mode}" for m in metrics], rotation=45, ha="right", fontsize=8)
    plt.ylabel("Mean")
    plt.title("D_M Probe 00 — Boundary and Parity by Tile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dm_boundary_parity_by_tile.png", dpi=160)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — D_M Probe 00 prune failed assumptions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default=None,
        help="Input dm_job_<JOB_ID>.npz. Defaults to data/latest_dm_qpu_data.json target.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output analysis directory. Defaults to analysis/dm_probe_00_prune_<job_id>_<timestamp>/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input) if args.input else read_latest_pointer()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    npz = np.load(input_path, allow_pickle=True)

    job_id = str(load_npz_scalar(npz["job_id"])) if "job_id" in npz.files else input_path.stem
    backend = str(load_npz_scalar(npz["backend"])) if "backend" in npz.files else "unknown"
    num_tiles = int(load_npz_scalar(npz["num_tiles"])) if "num_tiles" in npz.files else int(np.asarray(npz["dim"]).shape[0])

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = ANALYSIS_DIR / f"dm_probe_00_prune_{job_id}_{now_tag()}"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 88}")
    print("  GHOST ORACLE SUITE — D_M PROBE 00: PRUNE FAILED ASSUMPTIONS")
    print(f"{'=' * 88}")
    print(f"  Input   : {input_path}")
    print(f"  Job ID  : {job_id}")
    print(f"  Backend : {backend}")
    print(f"  Tiles   : {num_tiles}")
    print(f"  Out dir : {out_dir}")

    metrics: List[TileMetrics] = []
    probs_by_tile: Dict[int, np.ndarray] = {}

    print("\n[SCORE] Tile-level assumption scoring...")
    for tile in range(num_tiles):
        m, prob = score_tile(npz, tile)
        metrics.append(m)
        probs_by_tile[tile] = prob

        print(
            f"  tile {tile:02d} "
            f"mode={m.mode:<16} "
            f"vote={m.prune_vote:<8} "
            f"class={m.raw_class:<34} "
            f"eff={m.effective_dim:6.3f} "
            f"z={m.chi2_uniform_z:7.2f} "
            f"pop={m.mean_popcount:5.3f} "
            f"b={m.mean_boundary:5.3f} "
            f"p={m.mean_parity:5.3f}"
        )

    mode_summary = summarize_modes(metrics, probs_by_tile)
    pruned = build_pruned_json(input_path, npz, metrics, mode_summary)

    tile_rows = [asdict(m) for m in metrics]

    write_csv(out_dir / "dm_probe_00_tile_metrics.csv", tile_rows)
    write_csv(out_dir / "dm_probe_00_mode_summary.csv", mode_summary)

    with open(out_dir / "dm_probe_00_pruned_assumptions.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(pruned), f, indent=2)

    write_report(
        path=out_dir / "dm_probe_00_report.md",
        input_path=input_path,
        npz=npz,
        metrics=metrics,
        mode_summary=mode_summary,
        pruned=pruned,
    )

    maybe_make_plots(out_dir, metrics, probs_by_tile)

    print("\n[PRUNE] Mode decisions:")
    for row in mode_summary:
        print(
            f"  {row['mode']:<16} -> {row['decision']:<16} "
            f"repeat={row['repeat_status']:<18} "
            f"eff_mean={row['effective_dim_mean']:.3f} "
            f"z_mean={row['chi2_uniform_z_mean']:.2f}"
        )

    print(f"\n{'=' * 88}")
    print("  D_M PROBE 00 COMPLETE")
    print(f"{'=' * 88}")
    print(f"  Report      : {out_dir / 'dm_probe_00_report.md'}")
    print(f"  Tile CSV    : {out_dir / 'dm_probe_00_tile_metrics.csv'}")
    print(f"  Mode CSV    : {out_dir / 'dm_probe_00_mode_summary.csv'}")
    print(f"  Pruned JSON : {out_dir / 'dm_probe_00_pruned_assumptions.json'}")
    print(f"{'=' * 88}\n")


if __name__ == "__main__":
    main()