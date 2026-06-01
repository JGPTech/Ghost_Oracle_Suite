#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — F_M QPU BASE (SUBMIT + DUMP)
==============================================================================
Unified F_M base script. Two modes, one file, following the Ghost Oracle Suite
QPU-base convention:

    submit : build the tiled paired-delay-cavity circuit and submit it to an
             IBM Quantum backend. Prints the job ID. Does NOT wait.

    dump   : fetch a completed job by ID and FREEZE the raw measurement record
             into data/ as a canonical .npz qproj base consumed by F_M probes,
             GPU emulation, geo extraction, and final benchmarks.

F_M working target
------------------
F_M is the Fractal Expansion / paired-delay-cavity discovery operator.

This script does NOT claim the circuit proves a physical model. It constructs a
circuit from the current paired-cavity / timestep-delay model, freezes the QPU
record, and preserves enough structure for later probes to determine what the
hardware actually computed.

The intended discovery path is:

    1. Build QPU circuit from paired-delay-cavity model.
    2. Dump qproj base.
    3. Probe qproj to see what survives controls.
    4. Build gproj GPU emulation with the same schema.
    5. Derive geo from qproj/gproj-aligned behavior.
    6. Project geo through the same probes.
    7. Build final F_M benchmark against classical equivalents.

Per-tile circuit
----------------
Each F_M tile uses 9 qubits:

    [seed, g_path, g_mem, em_path, em_mem, ctrl, scale, branch, aux]

Conceptually:

    seed      : common source state
    g_path    : first matched cavity/path record
    g_mem     : memory/readout qubit for g_path
    em_path   : second matched cavity/path record
    em_mem    : memory/readout qubit for em_path
    ctrl      : interference/control readout
    scale     : recursive scale flag
    branch    : branch/ancestry flag
    aux       : extra mixing / residue qubit

The circuit intentionally preserves separate readouts for the two paths.
The differential field is derived during dump/probe, not asserted in-circuit.

Register names per tile:

    fm_ctrl_tile{t}    : 1 bit
    fm_g_tile{t}       : 2 bits  [g_path, g_mem]
    fm_em_tile{t}      : 2 bits  [em_path, em_mem]
    fm_scale_tile{t}   : 1 bit
    fm_branch_tile{t}  : 2 bits  [branch, aux]

Dumped arrays per tile:

    ctrl_tile{t}       uint8, shape (shots,)
    g_tile{t}          uint8, shape (shots, 2)
    em_tile{t}         uint8, shape (shots, 2)
    scale_tile{t}      uint8, shape (shots,)
    branch_tile{t}     uint8, shape (shots, 2)
    delta_tile{t}      int8,  shape (shots, 2), em_tile - g_tile
    xor_delta_tile{t}  uint8, shape (shots, 2), em_tile XOR g_tile

Shared base schema
------------------
The .npz is designed so later network/Converger scripts can load operator bases
interchangeably across G_M/S_M/T_S/F_M:

    schema              str
    suite               str
    operator            str
    substrate           str, "qproj"
    job_id              str
    backend             str
    shots               int
    num_tiles           int
    tile_indices        int32 array
    qubits_per_tile     int
    circuit_family      str
    delays_dt           int32 array
    scale_levels        int32 array
    tile_theta          float64 array
    tile_delay_dt       int32 array
    tile_scale_level    int32 array
    tile_mode           unicode array
    tile_role           unicode array
    tile_meta_json      unicode JSON string

Usage
-----
    export IBM_QUANTUM_TOKEN=<your_token>

    python f_m_qpu_generate.py submit
    python f_m_qpu_generate.py submit --backend ibm_marrakesh --shots 4096
    python f_m_qpu_generate.py submit --delays-dt 0 1 2 4 8 16
    python f_m_qpu_generate.py submit --theta-values 0.25 0.50 0.75 1.00

    # ... wait for the job to finish ...

    python f_m_qpu_generate.py dump <JOB_ID>
    python f_m_qpu_generate.py dump <JOB_ID> --num-tiles 12
    python f_m_qpu_generate.py dump <JOB_ID> --out data/fm_job_<JOB_ID>.npz

Notes
-----
This is intentionally an early operator-discovery circuit. The goal is not to
make the final F_M claim here. The goal is to freeze a rich qproj record whose
scale/path/delay structure survives long enough for probes to interrogate.
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import XGate, YGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
    BasisTranslator,
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

warnings.filterwarnings("ignore", category=DeprecationWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BACKEND_NAME = "ibm_marrakesh"
DEFAULT_SHOTS = 4096

# Each F_M tile uses 9 physical qubits:
# [seed, g_path, g_mem, em_path, em_mem, ctrl, scale, branch, aux]
QUBITS_PER_TILE = 9

# Delay ladder in backend dt units. This is the F_M timestep-delay scaffold.
DEFAULT_DELAYS_DT = [0, 1, 2, 4, 8, 16]

# Scale levels are recorded as metadata and used to vary recursive depth.
DEFAULT_SCALE_LEVELS = [1, 2, 4, 8, 16, 32]

# Seed angle family. These are not final "data"; they are circuit diversity.
DEFAULT_THETA_VALUES = [0.25, 0.50, 0.75, 1.00]

# Small phase coefficient used to create an EM-path phase response per delay.
# This is intentionally a model knob, not a claim.
DEFAULT_PHASE_PER_DT = 0.015

# Per-tile stagger to reduce simultaneous pulse density.
STAGGER_DT = 72

# Backend-specific known-bad/high-error exclusions inherited from G_M style.
# Override with --excluded-qubits for a new backend/calibration campaign.
EXCLUDED_QUBITS = [15, 19, 29, 38, 49, 51, 67, 70, 114, 116, 118, 120, 122, 147]


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


# =============================================================================
# SMALL HELPERS
# =============================================================================

def json_safe(x: Any) -> Any:
    """Convert numpy/path objects into JSON-safe objects."""
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


def get_token(required: bool = True) -> Optional[str]:
    """
    Load IBM Quantum token from environment.

    submit mode requires the token unless the user has already saved an account.
    dump mode can often use a saved default account, so required=False is used.
    """
    token = os.environ.get("IBM_QUANTUM_TOKEN", "").strip()
    if token:
        return token

    if required:
        sys.stderr.write(
            "\n[ERROR] IBM_QUANTUM_TOKEN environment variable not set.\n"
            "        export IBM_QUANTUM_TOKEN=<your_token>\n"
            "        or save a default account via QiskitRuntimeService.save_account.\n\n"
        )
        sys.exit(1)

    return None


def parse_int_list(values: Optional[Sequence[str]]) -> Optional[List[int]]:
    if values is None:
        return None
    return [int(v) for v in values]


def public_attrs(obj: Any) -> List[str]:
    return [a for a in dir(obj) if not a.startswith("_")]


def extract_bitstrings(register_obj: Any) -> List[str]:
    """
    Extract bitstrings from a Qiskit Runtime register object.

    This matches the working SamplerV2 DataBin extraction path used in the
    existing Ghost Oracle QPU scripts.
    """
    if hasattr(register_obj, "get_bitstrings"):
        return register_obj.get_bitstrings()

    raise RuntimeError(
        "Register object does not expose get_bitstrings(). "
        f"type={type(register_obj)} attrs={public_attrs(register_obj)}"
    )


def bits_to_array(bitstrings: Sequence[str], width: int, reverse: bool = True) -> np.ndarray:
    """
    Convert bitstrings to uint8 array shape (shots, width).

    Qiskit classical bitstrings often display MSB-first relative to register
    order. For consistency with the G_M dump style, reverse=True maps display
    strings back into register-index order.
    """
    rows: List[List[int]] = []
    for bs in bitstrings:
        s = bs[::-1] if reverse else bs
        if len(s) != width:
            raise ValueError(f"expected bitstring width {width}, got {len(s)} from {bs!r}")
        rows.append([int(ch) for ch in s])
    return np.asarray(rows, dtype=np.uint8)


def single_bit_to_array(bitstrings: Sequence[str]) -> np.ndarray:
    return np.asarray([int(bs[-1]) for bs in bitstrings], dtype=np.uint8)


# =============================================================================
# TILE METADATA
# =============================================================================

@dataclass
class FMTileMeta:
    tile: int
    theta: float
    delay_dt: int
    scale_level: int
    mode: str
    role: str


def build_tile_plan(
    num_tiles: int,
    theta_values: Sequence[float],
    delays_dt: Sequence[int],
    scale_levels: Sequence[int],
    modes: Sequence[str],
) -> List[FMTileMeta]:
    """
    Build deterministic per-tile metadata.

    The plan intentionally cycles through theta, delay, scale, and mode so the
    first qproj base contains enough diversity for early structure probes.
    """
    if not theta_values:
        raise ValueError("theta_values must not be empty")
    if not delays_dt:
        raise ValueError("delays_dt must not be empty")
    if not scale_levels:
        raise ValueError("scale_levels must not be empty")
    if not modes:
        raise ValueError("modes must not be empty")

    plan: List[FMTileMeta] = []
    for t in range(num_tiles):
        theta = float(theta_values[t % len(theta_values)])
        delay_dt = int(delays_dt[t % len(delays_dt)])
        scale_level = int(scale_levels[(t // len(delays_dt)) % len(scale_levels)])
        mode = str(modes[(t // max(1, len(delays_dt) * len(scale_levels))) % len(modes)])
        role = f"theta{t % len(theta_values)}_delay{delay_dt}_scale{scale_level}_{mode}"

        plan.append(
            FMTileMeta(
                tile=t,
                theta=theta,
                delay_dt=delay_dt,
                scale_level=scale_level,
                mode=mode,
                role=role,
            )
        )
    return plan


# =============================================================================
# QUBIT CLUSTER SELECTION
# =============================================================================

def get_readout_error_from_target(target: Any, qubit: int, fallback: float = 0.5) -> float:
    """
    Best-effort readout error lookup.

    Qiskit Target measure keys are often tuple-keyed, e.g. (q,). This helper
    avoids the older int-vs-tuple membership bug while still falling back safely.
    """
    try:
        measure = target["measure"]
        for key in ((qubit,), qubit):
            try:
                inst = measure[key]
                err = getattr(inst, "error", None)
                if err is not None:
                    return float(err)
            except Exception:
                pass
    except Exception:
        pass
    return fallback


def find_qubit_clusters(
    backend: Any,
    cluster_size: int = QUBITS_PER_TILE,
    max_tiles: int = 16,
    excluded_qubits: Optional[Sequence[int]] = None,
    reserve_neighbors: bool = True,
) -> List[List[int]]:
    """
    Greedy connected-cluster selection.

    This is intentionally simple and reproducible:
      1. Build undirected coupling adjacency.
      2. Sort candidate seeds by readout error where available.
      3. BFS-grow connected clusters.
      4. Reserve each cluster and optionally its one-hop neighborhood.

    The F_M discovery script should preserve the selected layout in metadata so
    later probes can decide whether topology mattered.
    """
    excluded = set(excluded_qubits or [])
    target = backend.target

    adjacency: Dict[int, set] = {i: set() for i in range(backend.num_qubits)}
    for u, v in backend.coupling_map.get_edges():
        adjacency[u].add(v)
        adjacency[v].add(u)

    readout_error = {
        i: get_readout_error_from_target(target, i, fallback=0.5)
        for i in range(backend.num_qubits)
    }

    candidates = sorted(
        [q for q in range(backend.num_qubits) if q not in excluded],
        key=lambda q: (readout_error.get(q, 1.0), q),
    )

    clusters: List[List[int]] = []
    used = set(excluded)

    for seed in candidates:
        if seed in used:
            continue

        cluster = [seed]
        frontier = [seed]

        while frontier and len(cluster) < cluster_size:
            current = frontier.pop(0)
            neighbors = sorted(
                [n for n in adjacency[current] if n not in used and n not in cluster],
                key=lambda q: (readout_error.get(q, 1.0), q),
            )
            for n in neighbors:
                if len(cluster) >= cluster_size:
                    break
                cluster.append(n)
                frontier.append(n)

        if len(cluster) == cluster_size:
            reserved = set(cluster)
            if reserve_neighbors:
                for q in cluster:
                    reserved.update(adjacency[q])
            used.update(reserved)
            clusters.append(cluster)

        if len(clusters) >= max_tiles:
            break

    return clusters


# =============================================================================
# F_M CIRCUIT CONSTRUCTION
# =============================================================================

def apply_mode_path(
    qc: QuantumCircuit,
    mode: str,
    g_path: Any,
    g_mem: Any,
    em_path: Any,
    em_mem: Any,
    ctrl: Any,
    scale: Any,
    branch: Any,
    aux: Any,
    delay_dt: int,
    scale_level: int,
    phase_per_dt: float,
) -> None:
    """
    Apply one F_M paired-cavity model step.

    This is intentionally model-building code, not final physics. Modes are
    circuit variants that give early probes different ways to interrogate the
    hardware response.

    clean:
        matched delay paths with minimal additional phase structure.

    phase_shear:
        EM path receives delay-proportional phase rotations and memory coupling.

    local_shock:
        branch/aux path injects a local perturbation before recombination.
    """
    delay_dt = int(delay_dt)
    scale_level = max(1, int(scale_level))
    phase = float(phase_per_dt) * float(delay_dt + 1) * np.log2(scale_level + 1)

    # Matched idle delay on both path records.
    if delay_dt > 0:
        qc.delay(delay_dt, g_path, unit="dt")
        qc.delay(delay_dt, em_path, unit="dt")
        qc.delay(delay_dt, g_mem, unit="dt")
        qc.delay(delay_dt, em_mem, unit="dt")

    # Shared memory imprint.
    qc.cx(g_path, g_mem)
    qc.cx(em_path, em_mem)

    if mode == "clean":
        # Minimal matched evolution.
        qc.rz(0.5 * phase, g_path)
        qc.rz(0.5 * phase, em_path)

    elif mode == "phase_shear":
        # EM-side phase shear plus memory echo.
        qc.rz(0.25 * phase, g_path)
        qc.rz(phase, em_path)
        qc.cx(em_path, em_mem)
        qc.rz(0.5 * phase, em_mem)
        qc.cx(em_path, em_mem)

    elif mode == "local_shock":
        # Localized perturbation carried through branch/aux.
        qc.h(aux)
        qc.cx(branch, aux)
        qc.rz(phase, aux)
        qc.cx(aux, em_path)
        qc.cx(aux, g_mem)
        qc.h(aux)

    else:
        raise ValueError(f"unknown F_M mode: {mode}")

    # Recombination / interference scaffold.
    qc.cx(g_mem, ctrl)
    qc.cx(em_mem, ctrl)
    qc.cz(scale, ctrl)
    qc.cx(branch, ctrl)


def build_circuit(
    tile_plan: Sequence[FMTileMeta],
    phase_per_dt: float = DEFAULT_PHASE_PER_DT,
    stagger_dt: int = STAGGER_DT,
) -> QuantumCircuit:
    """
    Build the tiled F_M paired-delay-cavity circuit.

    Each tile records two matched path/cavity branches separately. The later
    dump/probe layer derives delta/xor-delta fields from those raw records.
    """
    num_tiles = len(tile_plan)
    qr = QuantumRegister(num_tiles * QUBITS_PER_TILE, name="q")
    qc = QuantumCircuit(qr)

    tile_cregs: List[Tuple[ClassicalRegister, ClassicalRegister, ClassicalRegister, ClassicalRegister, ClassicalRegister]] = []

    for meta in tile_plan:
        t = meta.tile

        ctrl_cr = ClassicalRegister(1, name=f"fm_ctrl_tile{t}")
        g_cr = ClassicalRegister(2, name=f"fm_g_tile{t}")
        em_cr = ClassicalRegister(2, name=f"fm_em_tile{t}")
        scale_cr = ClassicalRegister(1, name=f"fm_scale_tile{t}")
        branch_cr = ClassicalRegister(2, name=f"fm_branch_tile{t}")

        qc.add_register(ctrl_cr)
        qc.add_register(g_cr)
        qc.add_register(em_cr)
        qc.add_register(scale_cr)
        qc.add_register(branch_cr)

        tile_cregs.append((ctrl_cr, g_cr, em_cr, scale_cr, branch_cr))

        base = t * QUBITS_PER_TILE
        seed = qr[base + 0]
        g_path = qr[base + 1]
        g_mem = qr[base + 2]
        em_path = qr[base + 3]
        em_mem = qr[base + 4]
        ctrl = qr[base + 5]
        scale = qr[base + 6]
        branch = qr[base + 7]
        aux = qr[base + 8]

        # Per-tile stagger, inherited from the G_M "spread pulse density" idea.
        tile_stagger = int(t * stagger_dt)
        if tile_stagger > 0:
            for q in [seed, g_path, g_mem, em_path, em_mem, ctrl, scale, branch, aux]:
                qc.delay(tile_stagger, q, unit="dt")

        qc.barrier([seed, g_path, g_mem, em_path, em_mem, ctrl, scale, branch, aux])

        # Common source state. Theta is intentionally stored in metadata.
        qc.ry(float(meta.theta), seed)

        # Scale and branch flags encode recursive-scale ancestry.
        # These are not labels only: they participate in the circuit.
        if meta.scale_level > 1:
            qc.ry(np.pi / min(8.0, float(meta.scale_level)), scale)
        else:
            qc.h(scale)

        qc.h(branch)

        # Split source into two matched paths.
        qc.cx(seed, g_path)
        qc.cx(seed, em_path)

        # Prepare ctrl as an interference readout.
        qc.h(ctrl)

        # Recursive-depth loop. Keep it shallow enough for current hardware.
        # scale_depth grows slowly with scale level.
        scale_depth = int(max(1, min(5, round(np.log2(max(1, meta.scale_level)) + 1))))

        for depth in range(scale_depth):
            # Depth-dependent delay: base delay plus a small recursive increment.
            step_delay = int(meta.delay_dt * (depth + 1))
            apply_mode_path(
                qc=qc,
                mode=meta.mode,
                g_path=g_path,
                g_mem=g_mem,
                em_path=em_path,
                em_mem=em_mem,
                ctrl=ctrl,
                scale=scale,
                branch=branch,
                aux=aux,
                delay_dt=step_delay,
                scale_level=meta.scale_level,
                phase_per_dt=phase_per_dt,
            )

            # Small echo structure to keep both paths comparable but not identical.
            if depth % 2 == 0:
                qc.x(g_path)
                qc.x(em_path)
                qc.x(g_path)
                qc.x(em_path)

        # Final interference readout.
        qc.h(ctrl)

        qc.barrier([seed, g_path, g_mem, em_path, em_mem, ctrl, scale, branch, aux])

        # Measurements. Registers intentionally preserve structure.
        qc.measure(ctrl, ctrl_cr[0])
        qc.measure([g_path, g_mem], g_cr)
        qc.measure([em_path, em_mem], em_cr)
        qc.measure(scale, scale_cr[0])
        qc.measure([branch, aux], branch_cr)

    return qc


# =============================================================================
# REGISTER INFERENCE FOR DUMP MODE
# =============================================================================

def infer_tile_indices(databin: Any) -> List[int]:
    """
    Infer tile indices from DataBin attributes.

    Expected register names:

        fm_ctrl_tile{t}
        fm_g_tile{t}
        fm_em_tile{t}
        fm_scale_tile{t}
        fm_branch_tile{t}
    """
    attrs = public_attrs(databin)

    groups = {
        "ctrl": set(),
        "g": set(),
        "em": set(),
        "scale": set(),
        "branch": set(),
    }

    patterns = {
        "ctrl": r"fm_ctrl_tile(\d+)",
        "g": r"fm_g_tile(\d+)",
        "em": r"fm_em_tile(\d+)",
        "scale": r"fm_scale_tile(\d+)",
        "branch": r"fm_branch_tile(\d+)",
    }

    for name in attrs:
        for key, pat in patterns.items():
            m = re.fullmatch(pat, name)
            if m:
                groups[key].add(int(m.group(1)))

    both = sorted(set.intersection(*groups.values()))
    if not both:
        raise RuntimeError(
            "No complete F_M tile register set found.\n"
            f"Available DataBin attrs: {attrs}"
        )

    for key, vals in groups.items():
        missing = sorted(set(both) - vals)
        if missing:
            print(f"[warn] missing {key} registers for tiles: {missing}")

    extras = {
        key: sorted(vals - set(both))
        for key, vals in groups.items()
        if sorted(vals - set(both))
    }
    if extras:
        print(f"[warn] unmatched extra tile registers: {extras}")

    return both


# =============================================================================
# SUBMIT MODE
# =============================================================================

def run_submit(args: argparse.Namespace) -> None:
    token = get_token(required=False)

    delays_dt = args.delays_dt if args.delays_dt is not None else DEFAULT_DELAYS_DT
    scale_levels = args.scale_levels if args.scale_levels is not None else DEFAULT_SCALE_LEVELS
    theta_values = args.theta_values if args.theta_values is not None else DEFAULT_THETA_VALUES
    modes = args.modes if args.modes is not None else ["clean", "phase_shear", "local_shock"]
    excluded = args.excluded_qubits if args.excluded_qubits is not None else EXCLUDED_QUBITS

    print(f"\n{'=' * 86}")
    print("  GHOST ORACLE SUITE — F_M QPU BASE SUBMITTER")
    print(f"{'=' * 86}")
    print(f"  Backend        : {args.backend}")
    print(f"  Shots          : {args.shots}")
    print(f"  Qubits/tile    : {QUBITS_PER_TILE}")
    print(f"  Delays dt      : {delays_dt}")
    print(f"  Scale levels   : {scale_levels}")
    print(f"  Theta values   : {theta_values}")
    print(f"  Modes          : {modes}")
    print(f"  Phase/dt       : {args.phase_per_dt}")
    print(f"  Stagger dt     : {args.stagger_dt}")

    if token:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")

    backend = service.backend(args.backend)

    print("\n[SETUP] Selecting connected 9-qubit F_M clusters...")
    clusters = find_qubit_clusters(
        backend=backend,
        cluster_size=QUBITS_PER_TILE,
        max_tiles=args.max_tiles,
        excluded_qubits=excluded,
        reserve_neighbors=not args.no_neighbor_reserve,
    )

    if not clusters:
        sys.stderr.write("[ERROR] No suitable 9-qubit clusters found on this backend.\n")
        sys.exit(1)

    num_tiles = len(clusters)
    tile_plan = build_tile_plan(
        num_tiles=num_tiles,
        theta_values=theta_values,
        delays_dt=delays_dt,
        scale_levels=scale_levels,
        modes=modes,
    )

    print(f"        Found {num_tiles} clusters.")
    for i, cluster in enumerate(clusters[: min(8, len(clusters))]):
        print(f"        tile {i:02d}: qubits={cluster} meta={asdict(tile_plan[i])}")
    if len(clusters) > 8:
        print(f"        ... {len(clusters) - 8} more tiles")

    print(f"\n[BUILD] Constructing F_M tiled circuit ({num_tiles * QUBITS_PER_TILE} qubits total)...")
    qc = build_circuit(
        tile_plan=tile_plan,
        phase_per_dt=args.phase_per_dt,
        stagger_dt=args.stagger_dt,
    )

    global_layout = [q for cluster in clusters for q in cluster]

    print("[BUILD] Transpiling with fixed initial layout...")
    pm = generate_preset_pass_manager(
        optimization_level=args.optimization_level,
        backend=backend,
        initial_layout=global_layout,
    )
    transpiled = pm.run(qc)

    isa_circuit = transpiled

    if not args.no_dd:
        print("[BUILD] Injecting XY4 dynamical decoupling sequence...")
        durations = InstructionDurations.from_backend(backend)

        # Make Y duration available when X exists.
        for i in range(backend.num_qubits):
            try:
                durations.update([("y", (i,), durations.get("x", i))])
            except Exception:
                continue

        xy4_sequence = [XGate(), YGate(), XGate(), YGate()]
        dd_pm = PassManager([
            ALAPScheduleAnalysis(durations),
            PadDynamicalDecoupling(durations, xy4_sequence),
        ])
        isa_circuit = dd_pm.run(isa_circuit)

        print("[BUILD] Translating back to backend basis...")
        translation_pm = PassManager([BasisTranslator(sel, backend.operation_names)])
        isa_circuit = translation_pm.run(isa_circuit)

    print(f"\n[CIRCUIT]")
    print(f"  depth          : {isa_circuit.depth()}")
    print(f"  size           : {isa_circuit.size()}")
    print(f"  width          : {isa_circuit.width()}")
    print(f"  classical bits : {isa_circuit.num_clbits}")

    print(f"\n[SUBMIT] Sending F_M job to {args.backend}...")
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=args.shots)
    job_id = job.job_id()

    # Save local submit metadata for later dump/probe convenience.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    submit_meta = {
        "schema": "ghost_oracle.fm.submit_meta.v1",
        "suite": "Ghost Oracle Suite",
        "operator": "F_M",
        "substrate": "qproj",
        "job_id": job_id,
        "backend": args.backend,
        "shots": int(args.shots),
        "qubits_per_tile": QUBITS_PER_TILE,
        "num_tiles": int(num_tiles),
        "clusters": clusters,
        "global_layout": global_layout,
        "delays_dt": delays_dt,
        "scale_levels": scale_levels,
        "theta_values": theta_values,
        "modes": modes,
        "phase_per_dt": float(args.phase_per_dt),
        "stagger_dt": int(args.stagger_dt),
        "optimization_level": int(args.optimization_level),
        "dd_enabled": not args.no_dd,
        "tile_plan": [asdict(m) for m in tile_plan],
        "circuit_depth": int(isa_circuit.depth()),
        "circuit_size": int(isa_circuit.size()),
        "circuit_width": int(isa_circuit.width()),
        "circuit_num_clbits": int(isa_circuit.num_clbits),
    }

    meta_path = DATA_DIR / f"fm_job_{job_id}_submit_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(submit_meta), f, indent=2)

    print(f"\n{'=' * 86}")
    print("  F_M JOB SUBMITTED")
    print(f"{'=' * 86}")
    print(f"  Job ID      : {job_id}")
    print(f"  Backend     : {args.backend}")
    print(f"  Tiles       : {num_tiles}")
    print(f"  Shots       : {args.shots}")
    print(f"  Submit meta : {meta_path}")
    print("\n  Next step:")
    print(f"      python f_m_qpu_generate.py dump {job_id}")
    print(f"{'=' * 86}\n")


# =============================================================================
# DUMP MODE
# =============================================================================

def load_submit_meta(job_id: str) -> Optional[Dict[str, Any]]:
    path = DATA_DIR / f"fm_job_{job_id}_submit_meta.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_dump(args: argparse.Namespace) -> None:
    job_id = args.job_id

    print(f"\n{'=' * 86}")
    print("  GHOST ORACLE SUITE — F_M JOB DUMP")
    print(f"{'=' * 86}")
    print(f"  Job ID   : {job_id}")
    print(f"  Data dir : {DATA_DIR}")

    print("\n[LOAD] Connecting to IBM Runtime...")
    token = get_token(required=False)
    if token:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")

    job = service.job(job_id)
    result = job.result()[0]
    databin = result.data

    try:
        backend_name = job.backend().name
    except Exception:
        backend_name = "unknown"

    submit_meta = load_submit_meta(job_id)
    if submit_meta:
        print(f"        Found local submit metadata.")
    else:
        print(f"        No local submit metadata found; dump will infer registers only.")

    print(f"        Backend: {backend_name}")

    if args.num_tiles is None:
        tile_indices = infer_tile_indices(databin)
        print(f"  Tiles    : auto-detected {len(tile_indices)} -> {tile_indices}")
    else:
        tile_indices = list(range(args.num_tiles))
        print(f"  Tiles    : requested {args.num_tiles} -> {tile_indices}")

    # Build metadata arrays. Prefer submit metadata when available.
    tile_meta_list: List[Dict[str, Any]] = []
    if submit_meta and "tile_plan" in submit_meta:
        tile_plan_by_tile = {int(m["tile"]): m for m in submit_meta["tile_plan"]}
        for out_t, original_t in enumerate(tile_indices):
            m = dict(tile_plan_by_tile.get(int(original_t), {}))
            if not m:
                m = {
                    "tile": int(out_t),
                    "theta": np.nan,
                    "delay_dt": -1,
                    "scale_level": -1,
                    "mode": "unknown",
                    "role": "unknown",
                }
            # Store output tile index too, since original tile indices can differ.
            m["stored_tile"] = int(out_t)
            m["original_tile"] = int(original_t)
            tile_meta_list.append(m)
    else:
        for out_t, original_t in enumerate(tile_indices):
            tile_meta_list.append({
                "tile": int(out_t),
                "stored_tile": int(out_t),
                "original_tile": int(original_t),
                "theta": np.nan,
                "delay_dt": -1,
                "scale_level": -1,
                "mode": "unknown",
                "role": "unknown",
            })

    tile_theta = np.asarray([m.get("theta", np.nan) for m in tile_meta_list], dtype=np.float64)
    tile_delay_dt = np.asarray([m.get("delay_dt", -1) for m in tile_meta_list], dtype=np.int32)
    tile_scale_level = np.asarray([m.get("scale_level", -1) for m in tile_meta_list], dtype=np.int32)
    tile_mode = np.asarray([str(m.get("mode", "unknown")) for m in tile_meta_list])
    tile_role = np.asarray([str(m.get("role", "unknown")) for m in tile_meta_list])

    delays_dt = np.asarray(
        submit_meta.get("delays_dt", DEFAULT_DELAYS_DT) if submit_meta else DEFAULT_DELAYS_DT,
        dtype=np.int32,
    )
    scale_levels = np.asarray(
        submit_meta.get("scale_levels", DEFAULT_SCALE_LEVELS) if submit_meta else DEFAULT_SCALE_LEVELS,
        dtype=np.int32,
    )

    data: Dict[str, Any] = {
        "schema": "ghost_oracle.fm.qproj.v1",
        "suite": "Ghost Oracle Suite",
        "operator": "F_M",
        "substrate": "qproj",
        "circuit_family": "paired_delay_cavity_discovery_v1",
        "job_id": str(job_id),
        "backend": str(backend_name),
        "num_tiles": np.int32(len(tile_indices)),
        "tile_indices": np.asarray(tile_indices, dtype=np.int32),
        "qubits_per_tile": np.int32(QUBITS_PER_TILE),
        "delays_dt": delays_dt,
        "scale_levels": scale_levels,
        "tile_theta": tile_theta,
        "tile_delay_dt": tile_delay_dt,
        "tile_scale_level": tile_scale_level,
        "tile_mode": tile_mode,
        "tile_role": tile_role,
        "tile_meta_json": json.dumps(json_safe(tile_meta_list)),
    }

    if submit_meta:
        data["submit_meta_json"] = json.dumps(json_safe(submit_meta))

    print("\n[EXTRACT] Per-tile F_M bitstrings...")
    observed_shots: Optional[int] = None

    for out_t, original_t in enumerate(tile_indices):
        names = {
            "ctrl": f"fm_ctrl_tile{original_t}",
            "g": f"fm_g_tile{original_t}",
            "em": f"fm_em_tile{original_t}",
            "scale": f"fm_scale_tile{original_t}",
            "branch": f"fm_branch_tile{original_t}",
        }

        for key, name in names.items():
            if not hasattr(databin, name):
                raise AttributeError(
                    f"DataBin has no register {name!r}. "
                    f"Available F_M tiles: {infer_tile_indices(databin)}"
                )

        ctrl_bits = extract_bitstrings(getattr(databin, names["ctrl"]))
        g_bits = extract_bitstrings(getattr(databin, names["g"]))
        em_bits = extract_bitstrings(getattr(databin, names["em"]))
        scale_bits = extract_bitstrings(getattr(databin, names["scale"]))
        branch_bits = extract_bitstrings(getattr(databin, names["branch"]))

        ctrl_arr = single_bit_to_array(ctrl_bits)
        g_arr = bits_to_array(g_bits, width=2, reverse=True)
        em_arr = bits_to_array(em_bits, width=2, reverse=True)
        scale_arr = single_bit_to_array(scale_bits)
        branch_arr = bits_to_array(branch_bits, width=2, reverse=True)

        if observed_shots is None:
            observed_shots = int(ctrl_arr.shape[0])
        else:
            if int(ctrl_arr.shape[0]) != observed_shots:
                raise RuntimeError(
                    f"shot count mismatch on tile {original_t}: "
                    f"{ctrl_arr.shape[0]} vs expected {observed_shots}"
                )

        # Derived convenience fields. These are not replacements for raw records.
        # Keep raw g/em arrays as the canonical record.
        delta_arr = em_arr.astype(np.int8) - g_arr.astype(np.int8)
        xor_delta_arr = np.bitwise_xor(em_arr, g_arr).astype(np.uint8)

        data[f"ctrl_tile{out_t}"] = ctrl_arr
        data[f"g_tile{out_t}"] = g_arr
        data[f"em_tile{out_t}"] = em_arr
        data[f"scale_tile{out_t}"] = scale_arr
        data[f"branch_tile{out_t}"] = branch_arr
        data[f"delta_tile{out_t}"] = delta_arr
        data[f"xor_delta_tile{out_t}"] = xor_delta_arr

        print(
            f"  tile{original_t} -> stored tile{out_t}: "
            f"ctrl {ctrl_arr.shape}, "
            f"g {g_arr.shape}, "
            f"em {em_arr.shape}, "
            f"scale {scale_arr.shape}, "
            f"branch {branch_arr.shape}, "
            f"p_ctrl1={ctrl_arr.mean():.4f}, "
            f"xor_delta_mean={xor_delta_arr.mean():.4f}"
        )

    data["shots"] = np.int32(observed_shots if observed_shots is not None else -1)

    # Optional stacked arrays for convenience. Per-tile arrays above remain the
    # canonical compatibility layer.
    try:
        data["ctrl"] = np.stack([data[f"ctrl_tile{t}"] for t in range(len(tile_indices))], axis=0)
        data["g"] = np.stack([data[f"g_tile{t}"] for t in range(len(tile_indices))], axis=0)
        data["em"] = np.stack([data[f"em_tile{t}"] for t in range(len(tile_indices))], axis=0)
        data["scale"] = np.stack([data[f"scale_tile{t}"] for t in range(len(tile_indices))], axis=0)
        data["branch"] = np.stack([data[f"branch_tile{t}"] for t in range(len(tile_indices))], axis=0)
        data["delta"] = np.stack([data[f"delta_tile{t}"] for t in range(len(tile_indices))], axis=0)
        data["xor_delta"] = np.stack([data[f"xor_delta_tile{t}"] for t in range(len(tile_indices))], axis=0)
    except Exception as e:
        print(f"[warn] could not create stacked convenience arrays: {e}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else DATA_DIR / f"fm_job_{job_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_path, **data)

    # Also write a tiny pointer for scripts that want "latest".
    latest_path = DATA_DIR / "latest_fm_qpu_data.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema": "ghost_oracle.fm.latest_pointer.v1",
                "operator": "F_M",
                "substrate": "qproj",
                "job_id": str(job_id),
                "backend": str(backend_name),
                "path": str(out_path),
                "shots": int(data["shots"]),
                "num_tiles": int(data["num_tiles"]),
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 86}")
    print("  F_M DUMP COMPLETE")
    print(f"{'=' * 86}")
    print(f"  Output     : {out_path}")
    print(f"  Latest ptr : {latest_path}")
    print(f"  Backend    : {backend_name}")
    print(f"  Tiles      : {len(tile_indices)}")
    print(f"  Shots      : {data['shots']}")
    print(f"{'=' * 86}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — F_M QPU base submit + dump",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # -------------------------------------------------------------------------
    # submit
    # -------------------------------------------------------------------------
    ps = sub.add_parser(
        "submit",
        help="Build and submit the tiled F_M paired-delay-cavity circuit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ps.add_argument("--backend", default=DEFAULT_BACKEND_NAME)
    ps.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    ps.add_argument("--max-tiles", type=int, default=16)
    ps.add_argument("--optimization-level", type=int, default=0, choices=[0, 1, 2, 3])

    ps.add_argument(
        "--delays-dt",
        type=int,
        nargs="+",
        default=None,
        help="Delay ladder in backend dt units.",
    )
    ps.add_argument(
        "--scale-levels",
        type=int,
        nargs="+",
        default=None,
        help="Recursive scale levels to encode in tile metadata/circuit.",
    )
    ps.add_argument(
        "--theta-values",
        type=float,
        nargs="+",
        default=None,
        help="Seed Ry(theta) values.",
    )
    ps.add_argument(
        "--modes",
        nargs="+",
        default=None,
        choices=["clean", "phase_shear", "local_shock"],
        help="Circuit path modes to cycle across tiles.",
    )
    ps.add_argument(
        "--phase-per-dt",
        type=float,
        default=DEFAULT_PHASE_PER_DT,
        help="Model coefficient for delay-proportional phase response.",
    )
    ps.add_argument(
        "--stagger-dt",
        type=int,
        default=STAGGER_DT,
        help="Per-tile stagger delay in dt.",
    )
    ps.add_argument(
        "--excluded-qubits",
        type=int,
        nargs="+",
        default=None,
        help="Override excluded qubit list.",
    )
    ps.add_argument(
        "--no-neighbor-reserve",
        action="store_true",
        help="Do not reserve one-hop neighbors around selected clusters.",
    )
    ps.add_argument(
        "--no-dd",
        action="store_true",
        help="Disable XY4 dynamical decoupling injection.",
    )

    # -------------------------------------------------------------------------
    # dump
    # -------------------------------------------------------------------------
    pd = sub.add_parser(
        "dump",
        help="Fetch completed F_M job by ID and freeze to data/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pd.add_argument("job_id", help="IBM Runtime job ID.")
    pd.add_argument(
        "--num-tiles",
        type=int,
        default=None,
        help="Number of tiles. If omitted, infer from DataBin register names.",
    )
    pd.add_argument(
        "--out",
        default=None,
        help="Optional output .npz path. Default: <F_M>/data/fm_job_<JOB_ID>.npz",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "submit":
        run_submit(args)
    elif args.mode == "dump":
        run_dump(args)
    else:
        raise RuntimeError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()