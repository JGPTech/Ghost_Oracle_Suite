#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
T_S — GPU / SYNTHETIC BASE GENERATOR
==============================================================================
Constructs a local GPU-generated T_S base with the same analysis-facing schema as
raw QPU dumps produced by t_s_qpu_generate.py.

This file follows the repo-native generator pattern used by S_M:

    S_M/s_m_gpu_generate.py -> controlled syndrome-spacetime field base
    T_S/t_s_gpu_generate.py -> controlled temporal-stress field base

Purpose
-------
T_S is not a scalar benchmark object. It is a temporal stress field:

    F[mode, delay_site, delay_value, shot, round, edge]

The QPU path creates this object by submitting delay/coupling/perturbation
circuits and dumping shot-order parity-probe classical registers.

The GPU path creates a controlled local base with the same downstream `.npz`
arrays:

    field  uint8, shape (modes, delay_sites, delays, shots, rounds, edges)
    final  uint8, shape (modes, delay_sites, delays, shots, channels)

The generated base is intended for:

    - development without waiting for QPU jobs,
    - control/fixture generation,
    - benchmark plumbing tests,
    - T_S documentation examples,
    - comparing real QPU fields against controlled synthetic fields,
    - running the same probes against GPU-generated data.

It is NOT a QPU simulator claim. It is a controlled T_S field generator.

Interchangeability target
-------------------------
The generated `.npz` intentionally matches the QPU-facing schema so existing T_S
probes can consume it like a QPU job dump.

Default QPU-like shape:

    modes       = clean, phase_shear, local_shock
    delay_sites = pre_coupling, post_coupling, post_perturb
    delays      = 0, 1, 2, 4, 8, 16
    shots       = 4096
    rounds      = 6
    channels    = 8
    edges       = 7

The important field object is:

    field[mode, delay_site, delay_index, shot, round, edge]

Default output
--------------
If this file lives in:

    ghost_oracle/T_S/t_s_gpu_generate.py

then generated files are written under:

    ghost_oracle/T_S/data/

Example:

    ghost_oracle/T_S/data/ts_gpu_data_4096shots_seed1234.npz

Metadata and latest pointers are also written:

    ghost_oracle/T_S/data/ts_gpu_job_<TAG>.json
    ghost_oracle/T_S/data/latest_ts_gpu_data.json
    ghost_oracle/T_S/data/latest_ts_data.json

The final pointer is intentionally named `latest_ts_data.json` so analysis tools
can consume the GPU-generated base the same way they consume dumped QPU data.

Usage
-----
Default controlled T_S base:

    python ghost_oracle/T_S/t_s_gpu_generate.py

Verify diagnostics:

    python ghost_oracle/T_S/t_s_gpu_generate.py --verify

Match a QPU run shape:

    python ghost_oracle/T_S/t_s_gpu_generate.py --shots 4096 --rounds 6 --channels 8

Force CPU fallback when CuPy is unavailable:

    python ghost_oracle/T_S/t_s_gpu_generate.py --allow-cpu

Avoid updating latest pointer:

    python ghost_oracle/T_S/t_s_gpu_generate.py --no-latest

Notes
-----
- CuPy is used when available.
- CPU fallback is available only when explicitly allowed with --allow-cpu.
- The saved arrays are NumPy arrays regardless of generation backend.
- The schema is intentionally aligned with T_S qpu dump output.
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp
    _HAVE_CUPY = True
except Exception:
    cp = None
    _HAVE_CUPY = False


# =============================================================================
# DEFAULTS — match current T_S QPU defaults unless intentionally overridden
# =============================================================================

DEFAULT_SHOTS = 4096
DEFAULT_ROUNDS = 6
DEFAULT_CHANNELS = 8
DEFAULT_DELAYS = [0, 1, 2, 4, 8, 16]
DEFAULT_DELAY_UNIT = "dt"
DEFAULT_MODES = ["clean", "phase_shear", "local_shock"]
DEFAULT_DELAY_SITES = ["pre_coupling", "post_coupling", "post_perturb"]

# Field/noise defaults chosen to produce T_S-like nontrivial structure without
# becoming fully random. They are controls, not QPU-simulation claims.
DEFAULT_P_BASE = 0.165
DEFAULT_P_READOUT = 0.018
DEFAULT_EDGE_MEMORY = 0.62
DEFAULT_ROUND_MEMORY = 0.70
DEFAULT_DELAY_GAIN = 0.020
DEFAULT_SITE_GAIN = 0.010
DEFAULT_PHASE_SHEAR = 0.045
DEFAULT_LOCAL_SHOCK = 0.115
DEFAULT_EDGE_SCAFFOLD = 0.095
DEFAULT_ROUND_SCAFFOLD = 0.060
DEFAULT_BURST_PROB = 0.000
DEFAULT_BURST_WIDTH = 2


# =============================================================================
# PATHS / IO
# =============================================================================

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def to_numpy(x: Any) -> np.ndarray:
    if _HAVE_CUPY and cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def gpu_name() -> str:
    if not _HAVE_CUPY:
        return "cupy-unavailable"
    try:
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return "unknown-cuda-gpu"


def as_backend_array(x: Any, xp: Any, dtype: Any = None) -> Any:
    if dtype is None:
        return xp.asarray(x)
    return xp.asarray(x, dtype=dtype)


# =============================================================================
# SMALL BIT / RATE HELPERS
# =============================================================================


def bits(x: Any, xp: Any) -> Any:
    return (x.astype(xp.uint8) & xp.uint8(1)).astype(xp.uint8)


def xor(a: Any, b: Any, xp: Any) -> Any:
    return xp.bitwise_xor(a.astype(xp.uint8), b.astype(xp.uint8)).astype(xp.uint8)


def clip01(x: Any, xp: Any) -> Any:
    return xp.clip(x, 0.0, 1.0)


def sigmoid_like(x: Any, xp: Any) -> Any:
    """
    Cheap bounded squasher. Avoids extreme rates while preserving ordering.
    """
    return 0.5 + 0.5 * xp.tanh(x)


def field_rate(field: np.ndarray) -> float:
    return float(np.asarray(field, dtype=np.float64).mean())


def delay_cv(field: np.ndarray) -> float:
    # field: delay, shots, rounds, edges
    prof = np.asarray(field, dtype=np.float64).mean(axis=(1, 2, 3))
    mu = abs(float(prof.mean())) + 1e-12
    return float(prof.std() / mu)


def round_cv(field: np.ndarray) -> float:
    prof = np.asarray(field, dtype=np.float64).mean(axis=(0, 1, 3))
    mu = abs(float(prof.mean())) + 1e-12
    return float(prof.std() / mu)


def edge_cv(field: np.ndarray) -> float:
    prof = np.asarray(field, dtype=np.float64).mean(axis=(0, 1, 2))
    mu = abs(float(prof.mean())) + 1e-12
    return float(prof.std() / mu)


def terminal_edge_parity(final: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(final[..., :-1], final[..., 1:]).astype(np.uint8)


def final_field_mismatch(final_block: np.ndarray, field_block: np.ndarray) -> float:
    """
    Diagnostic mismatch between final adjacent parity and final measured edge
    field round. This is a loose consistency diagnostic, not a correctness claim.
    """
    if final_block.shape[-1] - 1 != field_block.shape[-1]:
        return float("nan")
    edge = terminal_edge_parity(final_block)
    last = field_block[:, -1, :]
    return float(np.bitwise_xor(edge, last).mean())


# =============================================================================
# T_S GPU GENERATION MODEL
# =============================================================================


def parse_csv_names(raw: Optional[str], default: Sequence[str]) -> List[str]:
    if raw is None:
        return list(default)
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty name list")
    return vals


def parse_int_list(raw: Optional[Sequence[int]], default: Sequence[int]) -> List[int]:
    if raw is None:
        return [int(x) for x in default]
    return [int(x) for x in raw]


def site_offset(site_index: int, n_sites: int) -> float:
    if n_sites <= 1:
        return 0.0
    center = (n_sites - 1) / 2.0
    return (site_index - center) / max(center, 1.0)


def mode_strength(mode_name: str, *, phase_shear: float, local_shock: float) -> float:
    if mode_name == "phase_shear":
        return float(phase_shear)
    if mode_name == "local_shock":
        return float(local_shock)
    return 0.0


def build_scaffold(
    *,
    delays: Sequence[int],
    rounds: int,
    edges: int,
    mode_name: str,
    mode_index: int,
    site_index: int,
    n_sites: int,
    p_base: float,
    delay_gain: float,
    site_gain: float,
    phase_shear: float,
    local_shock: float,
    edge_scaffold: float,
    round_scaffold: float,
    xp: Any,
) -> Any:
    """
    Build bounded probability scaffold:

        p[delay, round, edge]

    This scaffold is deliberately T_S-shaped. It encodes:
        - weak delay dependence,
        - stronger edge/round scaffold dependence,
        - phase shear mode,
        - local shock mode,
        - delay-site offset.
    """
    D = len(delays)
    R = int(rounds)
    E = int(edges)

    delay_arr = as_backend_array(np.asarray(delays, dtype=np.float32), xp, xp.float32)
    if D > 1:
        dnorm = delay_arr / (float(max(delays)) + 1e-6)
    else:
        dnorm = xp.zeros((D,), dtype=xp.float32)

    r = xp.arange(R, dtype=xp.float32)
    e = xp.arange(E, dtype=xp.float32)

    if R > 1:
        rnorm = r / float(R - 1)
    else:
        rnorm = xp.zeros((R,), dtype=xp.float32)

    if E > 1:
        enorm = e / float(E - 1)
    else:
        enorm = xp.zeros((E,), dtype=xp.float32)

    # Shapes: D, R, E.
    Dv = dnorm.reshape(D, 1, 1)
    Rv = rnorm.reshape(1, R, 1)
    Ev = enorm.reshape(1, 1, E)

    # Edge/round scaffold intentionally mirrors Probe 06/07 result:
    # edge- and round-localized structure matters more than exact delay order.
    edge_wave = xp.sin((Ev + 0.15) * 2.0 * math.pi)
    edge_peak_hi = xp.exp(-((Ev - 0.86) ** 2) / (2.0 * 0.12 * 0.12))
    edge_peak_mid = xp.exp(-((Ev - 0.42) ** 2) / (2.0 * 0.13 * 0.13))
    round_peak = xp.exp(-((Rv - 0.40) ** 2) / (2.0 * 0.16 * 0.16))
    round_wave = xp.cos((Rv + 0.10) * 2.0 * math.pi)

    scaffold = xp.full((D, R, E), float(p_base), dtype=xp.float32)
    scaffold = scaffold + float(delay_gain) * Dv
    scaffold = scaffold + float(site_gain) * site_offset(site_index, n_sites)

    scaffold = scaffold + float(edge_scaffold) * (0.35 * edge_wave + 0.80 * edge_peak_hi + 0.55 * edge_peak_mid)
    scaffold = scaffold + float(round_scaffold) * (0.65 * round_peak + 0.25 * round_wave)

    if mode_name == "phase_shear":
        # Round-edge shear plus delay modulation.
        shear = (Rv * Ev) + 0.20 * Dv * Ev
        scaffold = scaffold + float(phase_shear) * shear

    elif mode_name == "local_shock":
        # Localized shock around the same kind of round/edge scaffold found in
        # Probe 06/07, with delay increasing the shock visibility.
        shock = xp.exp(
            -(
                ((Rv - 0.42) ** 2) / (2.0 * 0.13 * 0.13)
                + ((Ev - 0.86) ** 2) / (2.0 * 0.14 * 0.14)
            )
        )
        scaffold = scaffold + float(local_shock) * shock * (0.70 + 0.30 * Dv)

    # Low-amplitude mode-index phase offset, useful when custom mode names are used.
    scaffold = scaffold + 0.005 * float(mode_index) * xp.sin((Rv + Ev) * math.pi)

    return clip01(scaffold, xp)


def apply_round_edge_memory(
    raw_bits: Any,
    *,
    edge_memory: float,
    round_memory: float,
    rng: Any,
    xp: Any,
) -> Any:
    """
    Add persistence across rounds and edges without changing the analysis schema.

    raw_bits shape:
        shots, delay, round, edge
    """
    f = raw_bits.astype(xp.uint8)
    shots, D, R, E = f.shape

    # Round persistence: if active, copy previous round bit into current round.
    if round_memory > 0.0 and R > 1:
        keep_round = rng.random((shots, D, R - 1, E)) < float(round_memory)
        prev = f[:, :, :-1, :]
        cur = f[:, :, 1:, :]
        mixed = xp.where(keep_round, prev, cur).astype(xp.uint8)
        f[:, :, 1:, :] = mixed

    # Edge persistence: if active, copy previous edge bit into current edge.
    if edge_memory > 0.0 and E > 1:
        keep_edge = rng.random((shots, D, R, E - 1)) < float(edge_memory)
        prev = f[:, :, :, :-1]
        cur = f[:, :, :, 1:]
        mixed = xp.where(keep_edge, prev, cur).astype(xp.uint8)
        f[:, :, :, 1:] = mixed

    return f.astype(xp.uint8)


def apply_bursts(
    f: Any,
    *,
    p_burst: float,
    burst_width: int,
    rng: Any,
    xp: Any,
) -> Any:
    """
    Optional local round-edge bursts, off by default.
    """
    if p_burst <= 0.0:
        return f

    shots, D, R, E = f.shape
    active = (rng.random((shots, D, 1, 1)) < float(p_burst)).astype(xp.uint8)
    r0 = rng.integers(0, R, size=(shots, D, 1, 1)) if hasattr(rng, "integers") else rng.randint(0, R, size=(shots, D, 1, 1))
    e0 = rng.integers(0, E, size=(shots, D, 1, 1)) if hasattr(rng, "integers") else rng.randint(0, E, size=(shots, D, 1, 1))

    rr = xp.arange(R, dtype=xp.int32).reshape(1, 1, R, 1)
    ee = xp.arange(E, dtype=xp.int32).reshape(1, 1, 1, E)

    width = max(1, int(burst_width))
    patch = (((rr - r0) % R) < width) & (((ee - e0) % E) < width)
    mask = (patch.astype(xp.uint8) * active).astype(xp.uint8)
    return xor(f, mask, xp)


def generate_block(
    *,
    shots: int,
    delays: Sequence[int],
    rounds: int,
    channels: int,
    mode_name: str,
    mode_index: int,
    site_index: int,
    n_sites: int,
    p_base: float,
    p_readout: float,
    edge_memory: float,
    round_memory: float,
    delay_gain: float,
    site_gain: float,
    phase_shear: float,
    local_shock: float,
    edge_scaffold: float,
    round_scaffold: float,
    p_burst: float,
    burst_width: int,
    rng: Any,
    xp: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one mode/site block.

    Returns
    -------
    field_block : uint8, shape (delays, shots, rounds, edges)
    final_block : uint8, shape (delays, shots, channels)
    """
    if channels < 2:
        raise ValueError("channels must be >= 2")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    D = len(delays)
    E = channels - 1

    scaffold = build_scaffold(
        delays=delays,
        rounds=rounds,
        edges=E,
        mode_name=mode_name,
        mode_index=mode_index,
        site_index=site_index,
        n_sites=n_sites,
        p_base=p_base,
        delay_gain=delay_gain,
        site_gain=site_gain,
        phase_shear=phase_shear,
        local_shock=local_shock,
        edge_scaffold=edge_scaffold,
        round_scaffold=round_scaffold,
        xp=xp,
    )  # D,R,E

    # Generate edge parity/probe field in shot-first layout for easy memory logic.
    p = scaffold.reshape(1, D, rounds, E)
    raw = (rng.random((shots, D, rounds, E)) < p).astype(xp.uint8)
    raw = apply_round_edge_memory(
        raw,
        edge_memory=edge_memory,
        round_memory=round_memory,
        rng=rng,
        xp=xp,
    )
    raw = apply_bursts(
        raw,
        p_burst=p_burst,
        burst_width=burst_width,
        rng=rng,
        xp=xp,
    )

    # Convert to required field layout:
    # D, shots, rounds, edges
    field_block_xp = xp.transpose(raw, (1, 0, 2, 3)).astype(xp.uint8)

    # Construct final channel bits whose adjacent parity is loosely tied to the
    # last round field. This preserves useful terminal structure without claiming
    # a physical QPU simulation.
    final = xp.zeros((D, shots, channels), dtype=xp.uint8)
    branch = (rng.random((D, shots, 1)) < 0.5).astype(xp.uint8)
    final[:, :, 0:1] = branch

    last_edge = field_block_xp[:, :, -1, :]  # D, shots, E
    for c in range(1, channels):
        final[:, :, c] = xor(final[:, :, c - 1], last_edge[:, :, c - 1], xp)

    if p_readout > 0.0:
        readout = (rng.random((D, shots, channels)) < float(p_readout)).astype(xp.uint8)
        final = xor(final, readout, xp)

    return (
        to_numpy(field_block_xp).astype(np.uint8),
        to_numpy(final).astype(np.uint8),
    )


def build_ts_gpu_base(
    *,
    shots: int,
    rounds: int,
    channels: int,
    delays: Sequence[int],
    delay_unit: str,
    modes: Sequence[str],
    delay_sites: Sequence[str],
    p_base: float,
    p_readout: float,
    edge_memory: float,
    round_memory: float,
    delay_gain: float,
    site_gain: float,
    phase_shear: float,
    local_shock: float,
    edge_scaffold: float,
    round_scaffold: float,
    p_burst: float,
    burst_width: int,
    seed: Optional[int],
    use_gpu: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the complete T_S GPU base and metadata."""
    if seed is None:
        seed = secrets.randbits(63)

    if use_gpu:
        xp = cp
        rng = cp.random.default_rng(seed)
        backend = "cupy_gpu"
        device = gpu_name()
    else:
        xp = np
        rng = np.random.default_rng(seed)
        backend = "numpy_cpu_fallback"
        device = "cpu"

    modes = list(modes)
    delay_sites = list(delay_sites)
    delays = [int(d) for d in delays]
    edges = int(channels) - 1

    tag = f"ts_gpu_{shots}shots_seed{seed}"

    field = np.zeros(
        (len(modes), len(delay_sites), len(delays), int(shots), int(rounds), edges),
        dtype=np.uint8,
    )
    final = np.zeros(
        (len(modes), len(delay_sites), len(delays), int(shots), int(channels)),
        dtype=np.uint8,
    )

    diagnostics: List[Dict[str, Any]] = []

    for mi, mode_name in enumerate(modes):
        for si, site_name in enumerate(delay_sites):
            f_block, final_block = generate_block(
                shots=int(shots),
                delays=delays,
                rounds=int(rounds),
                channels=int(channels),
                mode_name=mode_name,
                mode_index=mi,
                site_index=si,
                n_sites=len(delay_sites),
                p_base=float(p_base),
                p_readout=float(p_readout),
                edge_memory=float(edge_memory),
                round_memory=float(round_memory),
                delay_gain=float(delay_gain),
                site_gain=float(site_gain),
                phase_shear=float(phase_shear),
                local_shock=float(local_shock),
                edge_scaffold=float(edge_scaffold),
                round_scaffold=float(round_scaffold),
                p_burst=float(p_burst),
                burst_width=int(burst_width),
                rng=rng,
                xp=xp,
            )

            field[mi, si] = f_block
            final[mi, si] = final_block

            diagnostics.append({
                "mode": mode_name,
                "delay_site": site_name,
                "field_rate": field_rate(f_block),
                "delay_cv": delay_cv(f_block),
                "round_cv": round_cv(f_block),
                "edge_cv": edge_cv(f_block),
                "final_mean": float(final_block.mean()),
                "terminal_mismatch": final_field_mismatch(final_block.reshape(-1, shots, channels)[-1], f_block.reshape(-1, shots, rounds, edges)[-1]),
            })

    saved: Dict[str, Any] = {
        "schema": np.array("ts_temporal_stress_metric"),
        "job_id": np.array(tag),
        "backend": np.array(backend),
        "source": np.array("gpu"),
        "shots": np.array(int(shots), dtype=np.int64),
        "rounds": np.array(int(rounds), dtype=np.int64),
        "channels": np.array(int(channels), dtype=np.int64),
        "edges": np.array(int(edges), dtype=np.int64),
        "modes": np.asarray(modes),
        "delay_sites": np.asarray(delay_sites),
        "delays": np.asarray(delays, dtype=np.int64),
        "delay_unit": np.array(str(delay_unit)),
        "field": field,
        "final": final,
    }

    meta = {
        "schema": "ts_gpu_synthetic_base",
        "qpu_schema_compat": "ts_temporal_stress_metric",
        "job_id": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "t_s_gpu_generate.py",
        "generation_backend": backend,
        "device": device,
        "source": "gpu",
        "shots": int(shots),
        "rounds": int(rounds),
        "channels": int(channels),
        "edges": int(edges),
        "modes": modes,
        "delay_sites": delay_sites,
        "delays": delays,
        "delay_unit": delay_unit,
        "field_shape": list(field.shape),
        "final_shape": list(final.shape),
        "seed": int(seed),
        "model": {
            "p_base": float(p_base),
            "p_readout": float(p_readout),
            "edge_memory": float(edge_memory),
            "round_memory": float(round_memory),
            "delay_gain": float(delay_gain),
            "site_gain": float(site_gain),
            "phase_shear": float(phase_shear),
            "local_shock": float(local_shock),
            "edge_scaffold": float(edge_scaffold),
            "round_scaffold": float(round_scaffold),
            "p_burst": float(p_burst),
            "burst_width": int(burst_width),
        },
        "diagnostics": diagnostics,
        "notes": (
            "Controlled T_S temporal-stress generator. This is not a QPU simulation claim. "
            "It writes the same analysis-facing field/final arrays as a T_S QPU dump so probes "
            "can consume GPU and QPU files interchangeably."
        ),
    }

    return saved, meta


# =============================================================================
# REPORTING
# =============================================================================


def print_diagnostics(meta: Dict[str, Any]) -> None:
    print("\n" + "=" * 118)
    print("  LIGHT DIAGNOSTICS")
    print("=" * 118)
    print(
        f"  {'mode':>13} | {'site':>14} | {'rate':>8} | {'delay CV':>8} | "
        f"{'round CV':>8} | {'edge CV':>8} | {'final':>8} | {'term mis':>8}"
    )
    print("  " + "-" * 116)
    for row in meta["diagnostics"]:
        print(
            f"  {row['mode']:>13} | {row['delay_site']:>14} | "
            f"{row['field_rate']:>8.5f} | {row['delay_cv']:>8.4f} | "
            f"{row['round_cv']:>8.4f} | {row['edge_cv']:>8.4f} | "
            f"{row['final_mean']:>8.4f} | {row['terminal_mismatch']:>8.4f}"
        )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T_S — GPU / synthetic temporal-stress base generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help="Shots per mode/site/delay block.")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="Temporal rounds.")
    p.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help="Channel qubits. Edges = channels - 1.")
    p.add_argument("--delays", type=int, nargs="+", default=DEFAULT_DELAYS, help="Delay values.")
    p.add_argument("--delay-unit", default=DEFAULT_DELAY_UNIT, help="Delay unit label.")

    p.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated mode names.")
    p.add_argument("--delay-sites", default=",".join(DEFAULT_DELAY_SITES), help="Comma-separated delay-site names.")

    p.add_argument("--p-base", type=float, default=DEFAULT_P_BASE, help="Base parity/probe rate.")
    p.add_argument("--p-readout", type=float, default=DEFAULT_P_READOUT, help="Final channel readout noise.")
    p.add_argument("--edge-memory", type=float, default=DEFAULT_EDGE_MEMORY, help="Edge persistence probability.")
    p.add_argument("--round-memory", type=float, default=DEFAULT_ROUND_MEMORY, help="Round persistence probability.")
    p.add_argument("--delay-gain", type=float, default=DEFAULT_DELAY_GAIN, help="Weak delay-rate gain.")
    p.add_argument("--site-gain", type=float, default=DEFAULT_SITE_GAIN, help="Delay-site offset gain.")
    p.add_argument("--phase-shear", type=float, default=DEFAULT_PHASE_SHEAR, help="Phase-shear mode strength.")
    p.add_argument("--local-shock", type=float, default=DEFAULT_LOCAL_SHOCK, help="Local-shock mode strength.")
    p.add_argument("--edge-scaffold", type=float, default=DEFAULT_EDGE_SCAFFOLD, help="Edge scaffold strength.")
    p.add_argument("--round-scaffold", type=float, default=DEFAULT_ROUND_SCAFFOLD, help="Round scaffold strength.")
    p.add_argument("--p-burst", type=float, default=DEFAULT_BURST_PROB, help="Optional local burst probability.")
    p.add_argument("--burst-width", type=int, default=DEFAULT_BURST_WIDTH, help="Optional round-edge burst width.")

    p.add_argument("--seed", type=int, default=None, help="RNG seed. Defaults to cryptographic random.")
    p.add_argument("--out", default=None, help="Output .npz path. Defaults to T_S/data/.")
    p.add_argument("--meta-out", default=None, help="Metadata JSON path. Defaults to T_S/data/.")
    p.add_argument("--allow-cpu", action="store_true", help="Allow NumPy CPU fallback if CuPy/CUDA is unavailable.")
    p.add_argument("--verify", action="store_true", help="Print light diagnostics after generation.")
    p.add_argument("--no-latest", action="store_true", help="Do not update latest_ts_gpu_data.json/latest_ts_data.json.")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()

    if int(args.channels) < 2:
        sys.exit("[FATAL] --channels must be >= 2")

    modes = parse_csv_names(args.modes, DEFAULT_MODES)
    delay_sites = parse_csv_names(args.delay_sites, DEFAULT_DELAY_SITES)
    delays = parse_int_list(args.delays, DEFAULT_DELAYS)

    use_gpu = _HAVE_CUPY
    if not use_gpu and not args.allow_cpu:
        sys.exit("[FATAL] CuPy/CUDA unavailable. Install cupy-cuda12x or rerun with --allow-cpu.")

    print(f"\n{'=' * 104}")
    print("  T_S — GPU / SYNTHETIC BASE GENERATOR")
    print(f"{'=' * 104}")
    print(f"  Backend      : {'cupy_gpu' if use_gpu else 'numpy_cpu_fallback'}")
    print(f"  Device       : {gpu_name() if use_gpu else 'cpu'}")
    print(f"  Modes        : {modes}")
    print(f"  Delay sites  : {delay_sites}")
    print(f"  Delays       : {delays} {args.delay_unit}")
    print(f"  Rounds       : {args.rounds}")
    print(f"  Channels     : {args.channels}")
    print(f"  Edges        : {int(args.channels) - 1}")
    print(f"  Shots        : {args.shots}")
    print(f"  Data dir     : {DATA_DIR}")
    print("\n[MODEL]")
    print(f"  p_base        : {args.p_base}")
    print(f"  p_readout     : {args.p_readout}")
    print(f"  edge_memory   : {args.edge_memory}")
    print(f"  round_memory  : {args.round_memory}")
    print(f"  delay_gain    : {args.delay_gain}")
    print(f"  site_gain     : {args.site_gain}")
    print(f"  phase_shear   : {args.phase_shear}")
    print(f"  local_shock   : {args.local_shock}")
    print(f"  edge_scaffold : {args.edge_scaffold}")
    print(f"  round_scaffold: {args.round_scaffold}")
    print(f"  p_burst       : {args.p_burst}")

    saved, meta = build_ts_gpu_base(
        shots=int(args.shots),
        rounds=int(args.rounds),
        channels=int(args.channels),
        delays=delays,
        delay_unit=str(args.delay_unit),
        modes=modes,
        delay_sites=delay_sites,
        p_base=float(args.p_base),
        p_readout=float(args.p_readout),
        edge_memory=float(args.edge_memory),
        round_memory=float(args.round_memory),
        delay_gain=float(args.delay_gain),
        site_gain=float(args.site_gain),
        phase_shear=float(args.phase_shear),
        local_shock=float(args.local_shock),
        edge_scaffold=float(args.edge_scaffold),
        round_scaffold=float(args.round_scaffold),
        p_burst=float(args.p_burst),
        burst_width=int(args.burst_width),
        seed=args.seed,
        use_gpu=use_gpu,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = str(meta["job_id"])

    out_path = Path(args.out) if args.out else DATA_DIR / f"ts_gpu_data_{args.shots}shots_seed{meta['seed']}.npz"
    meta_path = Path(args.meta_out) if args.meta_out else DATA_DIR / f"ts_gpu_job_{tag}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_path, **saved)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(meta), f, indent=2)

    if not args.no_latest:
        latest_gpu = DATA_DIR / "latest_ts_gpu_data.json"
        latest_any = DATA_DIR / "latest_ts_data.json"
        latest_obj = {"job_id": tag, "npz": str(out_path), "meta": str(meta_path), "source": "gpu"}
        with open(latest_gpu, "w", encoding="utf-8") as f:
            json.dump(latest_obj, f, indent=2)
        with open(latest_any, "w", encoding="utf-8") as f:
            json.dump(latest_obj, f, indent=2)

    if args.verify:
        print_diagnostics(meta)

    print(f"\n{'=' * 104}")
    print("  T_S GPU BASE COMPLETE")
    print(f"{'=' * 104}")
    print(f"  Output    : {out_path}")
    print(f"  Metadata  : {meta_path}")
    print(f"  Latest GPU: {DATA_DIR / 'latest_ts_gpu_data.json'}")
    print(f"  Analysis  : {DATA_DIR / 'latest_ts_data.json'}")
    print(f"  Seed      : {meta['seed']}")
    print("\n  Next:")
    print("    python ghost_oracle/T_S/t_s_gpu_generate.py --verify")
    print("    python ghost_oracle/T_S/probes/t_s_probe1.py --npz <OUTPUT_PATH>")
    print(f"{'=' * 104}\n")


if __name__ == "__main__":
    main()
