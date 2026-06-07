#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GHOST ORACLE SUITE — D_M PROBE 01: CHANNEL EXTRACTION
================================================================================

Purpose
-------
Probe 00 pruned failed assumptions.

Probe 01 extracts self-contained D_M candidate channels from the frozen QPU scene.

This probe does NOT benchmark D_M yet.
This probe does NOT compare against PCA / UMAP / Faiss / autoencoders yet.
This probe does NOT use G_M, S_M, T_S, F_M, or any other operator.

It only asks:

    Given one frozen D_M QPU base, and the Probe 00 prune/mutation report,
    what candidate dimensional channels survive strongly enough to carry forward?

Output channels
---------------
The probe builds candidate channels from tile modes:

    local_order_channel:
        smooth_walk

    collapse_channel:
        collapse_gate + phase_shear

    mutation_channel:
        scramble_order + nonlocal_jump

    symmetry_boundary_channel:
        mirror_parity + boundary_reflect

    rank_spread_channel:
        rank_spread

    composite_dm_channel:
        weighted aggregate of all KEEP and KEEP_AS_MUTATION channels

Each channel exports:

    - 16-state probability weights
    - 4-bit dimension weights
    - scalar feature summary
    - candidate projection recipe metadata

Later benchmark direction
-------------------------
Probe 02+ can use these extracted channels to guide self-contained dimensional
projection/compression tasks:

    high-dimensional data -> lower-dimensional representation -> nearest-neighbor preservation

This script does not perform that task yet. It only freezes the channel candidates.

Usage
-----
    python ghost_oracle/D_M/probes/dm_probe_01_channel_extract.py

    python ghost_oracle/D_M/probes/dm_probe_01_channel_extract.py ^
        --input ghost_oracle/D_M/data/dm_job_d8fb033o3njc73f01170.npz ^
        --prune-json ghost_oracle/D_M/probes/analysis/<probe00>/dm_probe_00_pruned_assumptions.json

================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
DM_DIR = HERE.parent
DATA_DIR = DM_DIR / "data"
ANALYSIS_DIR = HERE / "analyze"


# =============================================================================
# CHANNEL DEFINITIONS
# =============================================================================

CHANNEL_MODE_MAP: Dict[str, List[str]] = {
    "local_order_channel": [
        "smooth_walk",
    ],
    "collapse_channel": [
        "collapse_gate",
        "phase_shear",
    ],
    "mutation_channel": [
        "scramble_order",
        "nonlocal_jump",
    ],
    "symmetry_boundary_channel": [
        "mirror_parity",
        "boundary_reflect",
    ],
    "rank_spread_channel": [
        "rank_spread",
    ],
}

VOTE_WEIGHT = {
    "SURVIVE": 1.00,
    "MUTATED": 0.85,
    "WEAK": 0.35,
    "FAIL": 0.00,
}

DECISION_WEIGHT = {
    "KEEP": 1.00,
    "KEEP_AS_MUTATION": 0.85,
    "HOLD": 0.35,
    "PRUNE": 0.00,
}


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


def load_npz_scalar(x: Any) -> Any:
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    return x


def read_latest_input() -> Path:
    ptr = DATA_DIR / "latest_dm_qpu_data.json"
    if not ptr.exists():
        raise FileNotFoundError(
            f"No latest D_M pointer found at {ptr}. Pass --input explicitly."
        )

    with open(ptr, "r", encoding="utf-8") as f:
        meta = json.load(f)

    path = Path(meta["path"])
    if not path.exists():
        raise FileNotFoundError(f"Latest D_M input missing: {path}")

    return path


def find_latest_probe00_json() -> Optional[Path]:
    """
    Find latest Probe 00 prune JSON from probes/analysis.
    """
    if not ANALYSIS_DIR.exists():
        return None

    matches = sorted(
        ANALYSIS_DIR.glob("dm_probe_00_prune_*/dm_probe_00_pruned_assumptions.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


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


def normalize_prob(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    s = float(np.sum(x))
    if s <= 1e-15:
        return np.ones_like(x, dtype=np.float64) / max(1, x.size)
    return x / s


def state_prob_from_dim_state(dim_state: np.ndarray) -> np.ndarray:
    hist = np.bincount(dim_state.astype(np.int64), minlength=16).astype(np.float64)
    return normalize_prob(hist)


def bit_weights_from_state_prob(prob16: np.ndarray) -> np.ndarray:
    """
    Convert 16-state weights into 4 bit-position weights.

    State bits are interpreted as:
        state = d0 + 2*d1 + 4*d2 + 8*d3
    """
    p = normalize_prob(prob16)
    weights = np.zeros(4, dtype=np.float64)

    for state in range(16):
        for bit in range(4):
            if (state >> bit) & 1:
                weights[bit] += p[state]

    return weights


def centered_weights(x: np.ndarray) -> np.ndarray:
    """
    Convert positive probability-like weights into centered signed weights.

    This is useful later for projection rules because it gives a contrast pattern
    rather than only an occupancy pattern.
    """
    x = np.asarray(x, dtype=np.float64)
    return x - float(np.mean(x))


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64))))


# =============================================================================
# LOAD TILE DATA
# =============================================================================

def get_tile_metadata(npz: Any, tile: int) -> Dict[str, Any]:
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


def get_tile_arrays(npz: Any, tile: int) -> Dict[str, np.ndarray]:
    def get(name: str) -> np.ndarray:
        key = f"{name}_tile{tile}"
        if key in npz.files:
            return np.asarray(npz[key])
        if name in npz.files:
            return np.asarray(npz[name][tile])
        raise KeyError(f"missing {name} for tile {tile}")

    out: Dict[str, np.ndarray] = {}
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


# =============================================================================
# PRUNE JSON HANDLING
# =============================================================================

def load_prune_report(path: Optional[Path]) -> Dict[str, Any]:
    """
    Load Probe 00 prune JSON if available.

    If unavailable, every tile starts as HOLD-ish evidence rather than failing.
    """
    if path is None:
        latest = find_latest_probe00_json()
        path = latest

    if path is None:
        return {
            "available": False,
            "tile_votes": [],
            "mode_summary": [],
            "summary": {},
            "path": None,
        }

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["available"] = True
    data["path"] = str(path)
    return data


def build_vote_lookup(prune: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    tile_lookup: Dict[int, Dict[str, Any]] = {}
    mode_lookup: Dict[str, Dict[str, Any]] = {}

    for row in prune.get("tile_votes", []):
        tile_lookup[int(row["tile"])] = row

    for row in prune.get("mode_summary", []):
        mode_lookup[str(row["mode"])] = row

    return tile_lookup, mode_lookup


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TileChannelInput:
    tile: int
    mode: str
    assumption: str
    role: str
    theta: float
    delay_dt: int
    scale_level: int
    vote: str
    decision: str
    evidence_weight: float

    state_prob: List[float]
    bit_weight: List[float]
    bit_centered: List[float]

    entropy_bits: float
    effective_dim: float
    mean_state: float
    mean_popcount: float
    mean_ctrl: float
    mean_edge: float
    mean_aux: float
    mean_parity: float
    mean_boundary: float
    mean_interior: float


@dataclass
class ChannelSummary:
    channel: str
    modes: str
    tiles: str
    n_tiles: int
    total_weight: float

    state_entropy_bits: float
    effective_dim: float
    max_state_prob: float
    active_states_1pct: int

    bit0_weight: float
    bit1_weight: float
    bit2_weight: float
    bit3_weight: float

    bit0_centered: float
    bit1_centered: float
    bit2_centered: float
    bit3_centered: float

    mean_popcount: float
    mean_ctrl: float
    mean_edge: float
    mean_aux: float
    mean_parity: float
    mean_boundary: float
    mean_interior: float

    channel_class: str
    carry_forward: bool
    reason: str


# =============================================================================
# EXTRACTION LOGIC
# =============================================================================

def build_tile_inputs(npz: Any, prune: Dict[str, Any]) -> Tuple[List[TileChannelInput], Dict[int, np.ndarray]]:
    tile_lookup, mode_lookup = build_vote_lookup(prune)

    num_tiles = int(load_npz_scalar(npz["num_tiles"])) if "num_tiles" in npz.files else int(np.asarray(npz["dim"]).shape[0])

    tile_inputs: List[TileChannelInput] = []
    prob_by_tile: Dict[int, np.ndarray] = {}

    for tile in range(num_tiles):
        meta = get_tile_metadata(npz, tile)
        arrays = get_tile_arrays(npz, tile)

        mode = meta["mode"]
        vote_row = tile_lookup.get(tile, {})
        mode_row = mode_lookup.get(mode, {})

        vote = str(vote_row.get("vote", "HOLD"))
        decision = str(mode_row.get("decision", "HOLD"))

        evidence_weight = VOTE_WEIGHT.get(vote, 0.35) * DECISION_WEIGHT.get(decision, 0.35)

        state_prob = state_prob_from_dim_state(arrays["dim_state"])
        bit_weight = bit_weights_from_state_prob(state_prob)
        bit_ctr = centered_weights(bit_weight)

        prob_by_tile[tile] = state_prob

        item = TileChannelInput(
            tile=tile,
            mode=mode,
            assumption=meta["assumption"],
            role=meta["role"],
            theta=meta["theta"],
            delay_dt=meta["delay_dt"],
            scale_level=meta["scale_level"],
            vote=vote,
            decision=decision,
            evidence_weight=float(evidence_weight),

            state_prob=state_prob.tolist(),
            bit_weight=bit_weight.tolist(),
            bit_centered=bit_ctr.tolist(),

            entropy_bits=entropy_base2(state_prob),
            effective_dim=effective_dimension(state_prob),
            mean_state=float(np.mean(arrays["dim_state"])),
            mean_popcount=float(np.mean(arrays["popcount"])),
            mean_ctrl=float(np.mean(arrays["ctrl"])),
            mean_edge=float(np.mean(arrays["edge"])),
            mean_aux=float(np.mean(arrays["aux"])),
            mean_parity=float(np.mean(arrays["parity"])),
            mean_boundary=float(np.mean(arrays["boundary"])),
            mean_interior=float(np.mean(arrays["interior"])),
        )

        tile_inputs.append(item)

    return tile_inputs, prob_by_tile


def weighted_channel_state_prob(items: List[TileChannelInput]) -> np.ndarray:
    if not items:
        return np.ones(16, dtype=np.float64) / 16.0

    acc = np.zeros(16, dtype=np.float64)
    total = 0.0

    for item in items:
        w = max(0.0, float(item.evidence_weight))
        acc += w * np.asarray(item.state_prob, dtype=np.float64)
        total += w

    if total <= 1e-15:
        # If everything is ambiguous, use an unweighted average rather than zero.
        for item in items:
            acc += np.asarray(item.state_prob, dtype=np.float64)
        total = float(len(items))

    return normalize_prob(acc / max(total, 1e-15))


def summarize_channel(channel: str, items: List[TileChannelInput]) -> Tuple[ChannelSummary, np.ndarray]:
    state_prob = weighted_channel_state_prob(items)
    bit_weight = bit_weights_from_state_prob(state_prob)
    bit_ctr = centered_weights(bit_weight)

    total_weight = float(np.sum([max(0.0, item.evidence_weight) for item in items]))

    def wmean(attr: str) -> float:
        if not items:
            return float("nan")
        weights = np.asarray([max(0.0, item.evidence_weight) for item in items], dtype=np.float64)
        vals = np.asarray([float(getattr(item, attr)) for item in items], dtype=np.float64)
        if float(np.sum(weights)) <= 1e-15:
            return float(np.mean(vals))
        return float(np.sum(weights * vals) / np.sum(weights))

    eff = effective_dimension(state_prob)
    ent = entropy_base2(state_prob)
    maxp = float(np.max(state_prob))
    active = int(np.sum(state_prob >= 0.01))

    modes = sorted(set(item.mode for item in items))
    tiles = [item.tile for item in items]

    channel_class, carry_forward, reason = classify_channel(
        channel=channel,
        items=items,
        effective_dim_value=eff,
        max_state_prob=maxp,
        total_weight=total_weight,
        bit_centered=bit_ctr,
    )

    summary = ChannelSummary(
        channel=channel,
        modes=";".join(modes),
        tiles=";".join(str(t) for t in tiles),
        n_tiles=len(items),
        total_weight=total_weight,

        state_entropy_bits=ent,
        effective_dim=eff,
        max_state_prob=maxp,
        active_states_1pct=active,

        bit0_weight=float(bit_weight[0]),
        bit1_weight=float(bit_weight[1]),
        bit2_weight=float(bit_weight[2]),
        bit3_weight=float(bit_weight[3]),

        bit0_centered=float(bit_ctr[0]),
        bit1_centered=float(bit_ctr[1]),
        bit2_centered=float(bit_ctr[2]),
        bit3_centered=float(bit_ctr[3]),

        mean_popcount=wmean("mean_popcount"),
        mean_ctrl=wmean("mean_ctrl"),
        mean_edge=wmean("mean_edge"),
        mean_aux=wmean("mean_aux"),
        mean_parity=wmean("mean_parity"),
        mean_boundary=wmean("mean_boundary"),
        mean_interior=wmean("mean_interior"),

        channel_class=channel_class,
        carry_forward=carry_forward,
        reason=reason,
    )

    return summary, state_prob


def classify_channel(
    channel: str,
    items: List[TileChannelInput],
    effective_dim_value: float,
    max_state_prob: float,
    total_weight: float,
    bit_centered: np.ndarray,
) -> Tuple[str, bool, str]:
    if not items:
        return "EMPTY", False, "no tiles assigned to channel"

    if total_weight <= 0.05:
        return "NO_EVIDENCE", False, "channel has no surviving/mutated evidence weight"

    contrast = float(np.max(np.abs(bit_centered)))

    if channel == "local_order_channel":
        if effective_dim_value >= 4.0 and contrast >= 0.03:
            return "LOCAL_ORDER_CANDIDATE", True, "smooth/local channel has structured dimensional spread"
        return "WEAK_LOCAL_ORDER", True, "local channel exists but dimensional contrast is weak"

    if channel == "collapse_channel":
        if effective_dim_value <= 4.5 or max_state_prob >= 0.30:
            return "COLLAPSE_CANDIDATE", True, "collapse/phase channel concentrates dimensional state mass"
        return "WEAK_COLLAPSE", True, "collapse channel exists but does not strongly concentrate"

    if channel == "mutation_channel":
        return "MUTATION_CANDIDATE", True, "mutated assumptions produced structured hardware response"

    if channel == "symmetry_boundary_channel":
        if contrast >= 0.04:
            return "SYMMETRY_BOUNDARY_CANDIDATE", True, "symmetry/boundary channel has bit-level contrast"
        return "WEAK_SYMMETRY_BOUNDARY", True, "symmetry/boundary channel exists but contrast is weak"

    if channel == "rank_spread_channel":
        if effective_dim_value >= 10.0:
            return "HIGH_SPREAD_CANDIDATE", True, "rank/spread channel preserves high effective dimension"
        return "WEAK_RANK_SPREAD", True, "rank/spread channel exists but spread is not high"

    if channel == "composite_dm_channel":
        return "COMPOSITE_CANDIDATE", True, "aggregate channel for later benchmark initialization"

    return "UNKNOWN_CHANNEL", True, "unknown channel carried forward for inspection"


def build_channels(tile_inputs: List[TileChannelInput]) -> Tuple[List[ChannelSummary], Dict[str, np.ndarray]]:
    by_mode: Dict[str, List[TileChannelInput]] = {}
    for item in tile_inputs:
        by_mode.setdefault(item.mode, []).append(item)

    summaries: List[ChannelSummary] = []
    channel_probs: Dict[str, np.ndarray] = {}

    for channel, modes in CHANNEL_MODE_MAP.items():
        items: List[TileChannelInput] = []
        for mode in modes:
            items.extend(by_mode.get(mode, []))

        summary, prob = summarize_channel(channel, items)
        summaries.append(summary)
        channel_probs[channel] = prob

    # Composite channel: all KEEP and KEEP_AS_MUTATION evidence, excluding FAIL/zero.
    composite_items = [
        item for item in tile_inputs
        if item.evidence_weight > 0.0 and item.vote in {"SURVIVE", "MUTATED", "WEAK", "HOLD"}
    ]
    composite_summary, composite_prob = summarize_channel("composite_dm_channel", composite_items)
    summaries.append(composite_summary)
    channel_probs["composite_dm_channel"] = composite_prob

    return summaries, channel_probs


# =============================================================================
# OUTPUT
# =============================================================================

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_channel_npz(
    path: Path,
    input_path: Path,
    prune_path: Optional[str],
    job_id: str,
    backend: str,
    tile_inputs: List[TileChannelInput],
    summaries: List[ChannelSummary],
    channel_probs: Dict[str, np.ndarray],
) -> None:
    data: Dict[str, Any] = {
        "schema": "ghost_oracle.dm.probe_01_channels.v1",
        "operator": "D_M",
        "probe": "dm_probe_01_channel_extract",
        "input_path": str(input_path),
        "prune_path": str(prune_path) if prune_path else "",
        "job_id": str(job_id),
        "backend": str(backend),
        "created": datetime.now().isoformat(timespec="seconds"),

        "channel_names": np.asarray([s.channel for s in summaries]),
        "channel_classes": np.asarray([s.channel_class for s in summaries]),
        "channel_carry_forward": np.asarray([s.carry_forward for s in summaries], dtype=np.uint8),
        "channel_effective_dim": np.asarray([s.effective_dim for s in summaries], dtype=np.float64),
        "channel_entropy_bits": np.asarray([s.state_entropy_bits for s in summaries], dtype=np.float64),
        "channel_total_weight": np.asarray([s.total_weight for s in summaries], dtype=np.float64),

        "tile_indices": np.asarray([t.tile for t in tile_inputs], dtype=np.int32),
        "tile_modes": np.asarray([t.mode for t in tile_inputs]),
        "tile_votes": np.asarray([t.vote for t in tile_inputs]),
        "tile_decisions": np.asarray([t.decision for t in tile_inputs]),
        "tile_evidence_weight": np.asarray([t.evidence_weight for t in tile_inputs], dtype=np.float64),
    }

    for summary in summaries:
        name = summary.channel
        prob = channel_probs[name]
        bits = bit_weights_from_state_prob(prob)
        ctr = centered_weights(bits)

        data[f"{name}_state_prob"] = prob.astype(np.float64)
        data[f"{name}_bit_weight"] = bits.astype(np.float64)
        data[f"{name}_bit_centered"] = ctr.astype(np.float64)

    np.savez_compressed(path, **data)


def write_json_bundle(
    path: Path,
    input_path: Path,
    prune: Dict[str, Any],
    job_id: str,
    backend: str,
    tile_inputs: List[TileChannelInput],
    summaries: List[ChannelSummary],
    channel_probs: Dict[str, np.ndarray],
) -> None:
    bundle = {
        "schema": "ghost_oracle.dm.probe_01_channels.v1",
        "operator": "D_M",
        "probe": "dm_probe_01_channel_extract",
        "input_path": str(input_path),
        "prune_path": prune.get("path"),
        "job_id": job_id,
        "backend": backend,
        "created": datetime.now().isoformat(timespec="seconds"),
        "scope": {
            "self_consistent_operator_only": True,
            "uses_other_operators": False,
            "benchmark_performed": False,
            "intended_future_task": (
                "structure-preserving dimensional compression for nearest-neighbor retrieval"
            ),
        },
        "channel_mode_map": CHANNEL_MODE_MAP,
        "tile_inputs": [asdict(t) for t in tile_inputs],
        "channel_summaries": [asdict(s) for s in summaries],
        "channel_state_prob": {
            name: prob.tolist()
            for name, prob in channel_probs.items()
        },
        "channel_bit_weight": {
            name: bit_weights_from_state_prob(prob).tolist()
            for name, prob in channel_probs.items()
        },
        "channel_bit_centered": {
            name: centered_weights(bit_weights_from_state_prob(prob)).tolist()
            for name, prob in channel_probs.items()
        },
        "notes": [
            "This file freezes candidate D_M channels for later probes.",
            "No benchmark comparison is performed here.",
            "No cross-operator interaction is used here.",
            "KEEP_AS_MUTATION channels are carried forward as hardware-disagreement candidates.",
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(bundle), f, indent=2)


def write_report(
    path: Path,
    input_path: Path,
    prune: Dict[str, Any],
    job_id: str,
    backend: str,
    tile_inputs: List[TileChannelInput],
    summaries: List[ChannelSummary],
) -> None:
    lines: List[str] = []

    lines.append("# D_M Probe 01 — Channel Extraction")
    lines.append("")
    lines.append("## Scene")
    lines.append("")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Probe 00 prune JSON: `{prune.get('path') or 'not provided'}`")
    lines.append(f"- Job ID: `{job_id}`")
    lines.append(f"- Backend: `{backend}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Self-contained `D_M` only.")
    lines.append("- No `G_M`, `S_M`, `T_S`, `F_M`, or node-network interactions.")
    lines.append("- No benchmark comparison yet.")
    lines.append("- Output is a frozen candidate-channel set for later probes.")
    lines.append("")
    lines.append("## Candidate Channels")
    lines.append("")
    lines.append("| Channel | Class | Carry | Modes | Tiles | EffDim | Entropy | Max P(state) | Reason |")
    lines.append("|---|---|---:|---|---|---:|---:|---:|---|")

    for s in summaries:
        lines.append(
            f"| `{s.channel}` | `{s.channel_class}` | {int(s.carry_forward)} | "
            f"`{s.modes}` | `{s.tiles}` | {s.effective_dim:.3f} | "
            f"{s.state_entropy_bits:.3f} | {s.max_state_prob:.3f} | {s.reason} |"
        )

    lines.append("")
    lines.append("## Bit-Level Channel Weights")
    lines.append("")
    lines.append("| Channel | b0 | b1 | b2 | b3 | centered b0 | centered b1 | centered b2 | centered b3 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for s in summaries:
        lines.append(
            f"| `{s.channel}` | "
            f"{s.bit0_weight:.4f} | {s.bit1_weight:.4f} | {s.bit2_weight:.4f} | {s.bit3_weight:.4f} | "
            f"{s.bit0_centered:.4f} | {s.bit1_centered:.4f} | {s.bit2_centered:.4f} | "
            f"{s.bit3_centered:.4f} |"
        )

    lines.append("")
    lines.append("## Tile Evidence")
    lines.append("")
    lines.append("| Tile | Mode | Vote | Decision | Weight | EffDim | Pop | Boundary | Parity |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|")

    for t in tile_inputs:
        lines.append(
            f"| {t.tile} | `{t.mode}` | `{t.vote}` | `{t.decision}` | "
            f"{t.evidence_weight:.3f} | {t.effective_dim:.3f} | "
            f"{t.mean_popcount:.3f} | {t.mean_boundary:.3f} | {t.mean_parity:.3f} |"
        )

    lines.append("")
    lines.append("## Locked Future Benchmark Direction")
    lines.append("")
    lines.append(
        "The extracted channels are intended for later self-contained `D_M` tests on "
        "structure-preserving dimensional compression for nearest-neighbor retrieval. "
        "The first challenger set should include raw exact cosine/L2, random projection, "
        "PCA, TruncatedSVD, UMAP, and exact vector search."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_make_plots(
    out_dir: Path,
    summaries: List[ChannelSummary],
    channel_probs: Dict[str, np.ndarray],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable; skipping plots: {e}")
        return

    names = [s.channel for s in summaries]
    mat = np.stack([channel_probs[name] for name in names], axis=0)

    plt.figure(figsize=(12, max(4, 0.5 * len(names))))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="P(state)")
    plt.xlabel("dim_state 0..15")
    plt.ylabel("channel")
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.title("D_M Probe 01 — Candidate Channel State Weights")
    plt.tight_layout()
    plt.savefig(out_dir / "dm_probe_01_channel_state_heatmap.png", dpi=160)
    plt.close()

    bit_mat = np.stack([bit_weights_from_state_prob(channel_probs[name]) for name in names], axis=0)

    plt.figure(figsize=(9, max(4, 0.5 * len(names))))
    plt.imshow(bit_mat, aspect="auto")
    plt.colorbar(label="bit weight")
    plt.xlabel("dimension bit")
    plt.ylabel("channel")
    plt.xticks(range(4), ["d0", "d1", "d2", "d3"])
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.title("D_M Probe 01 — Candidate Channel Bit Weights")
    plt.tight_layout()
    plt.savefig(out_dir / "dm_probe_01_channel_bit_heatmap.png", dpi=160)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — D_M Probe 01 channel extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default=None,
        help="Input dm_job_<JOB_ID>.npz. Defaults to D_M/data/latest_dm_qpu_data.json target.",
    )
    p.add_argument(
        "--prune-json",
        default=None,
        help="Probe 00 dm_probe_00_pruned_assumptions.json. Defaults to latest in probes/analysis.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output analysis directory. Defaults to probes/analysis/dm_probe_01_channel_extract_<job_id>_<timestamp>/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input) if args.input else read_latest_input()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    prune_path = Path(args.prune_json) if args.prune_json else None
    prune = load_prune_report(prune_path)

    npz = np.load(input_path, allow_pickle=True)

    job_id = str(load_npz_scalar(npz["job_id"])) if "job_id" in npz.files else input_path.stem
    backend = str(load_npz_scalar(npz["backend"])) if "backend" in npz.files else "unknown"

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = ANALYSIS_DIR / f"dm_probe_01_channel_extract_{job_id}_{now_tag()}"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 88}")
    print("  GHOST ORACLE SUITE — D_M PROBE 01: CHANNEL EXTRACTION")
    print(f"{'=' * 88}")
    print(f"  Input      : {input_path}")
    print(f"  Prune JSON : {prune.get('path') or 'not provided'}")
    print(f"  Job ID     : {job_id}")
    print(f"  Backend    : {backend}")
    print(f"  Out dir    : {out_dir}")
    print("\n[SCOPE]")
    print("  Self-contained D_M only.")
    print("  No cross-operator interaction.")
    print("  No benchmark comparison yet.")

    tile_inputs, prob_by_tile = build_tile_inputs(npz, prune)
    summaries, channel_probs = build_channels(tile_inputs)

    tile_rows = [asdict(t) for t in tile_inputs]
    summary_rows = [asdict(s) for s in summaries]

    write_csv(out_dir / "dm_probe_01_tile_channel_inputs.csv", tile_rows)
    write_csv(out_dir / "dm_probe_01_channel_summary.csv", summary_rows)

    write_json_bundle(
        path=out_dir / "dm_probe_01_channels.json",
        input_path=input_path,
        prune=prune,
        job_id=job_id,
        backend=backend,
        tile_inputs=tile_inputs,
        summaries=summaries,
        channel_probs=channel_probs,
    )

    write_channel_npz(
        path=out_dir / "dm_probe_01_channels.npz",
        input_path=input_path,
        prune_path=prune.get("path"),
        job_id=job_id,
        backend=backend,
        tile_inputs=tile_inputs,
        summaries=summaries,
        channel_probs=channel_probs,
    )

    write_report(
        path=out_dir / "dm_probe_01_report.md",
        input_path=input_path,
        prune=prune,
        job_id=job_id,
        backend=backend,
        tile_inputs=tile_inputs,
        summaries=summaries,
    )

    maybe_make_plots(out_dir, summaries, channel_probs)

    print("\n[CHANNELS]")
    for s in summaries:
        print(
            f"  {s.channel:<28} "
            f"class={s.channel_class:<32} "
            f"carry={int(s.carry_forward)} "
            f"tiles={s.tiles:<8} "
            f"eff={s.effective_dim:6.3f} "
            f"entropy={s.state_entropy_bits:6.3f} "
            f"maxp={s.max_state_prob:6.3f}"
        )

    print(f"\n{'=' * 88}")
    print("  D_M PROBE 01 COMPLETE")
    print(f"{'=' * 88}")
    print(f"  Report       : {out_dir / 'dm_probe_01_report.md'}")
    print(f"  Channel JSON : {out_dir / 'dm_probe_01_channels.json'}")
    print(f"  Channel NPZ  : {out_dir / 'dm_probe_01_channels.npz'}")
    print(f"  Summary CSV  : {out_dir / 'dm_probe_01_channel_summary.csv'}")
    print(f"  Tile CSV     : {out_dir / 'dm_probe_01_tile_channel_inputs.csv'}")
    print(f"{'=' * 88}\n")


if __name__ == "__main__":
    main()