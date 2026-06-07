#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
D_M QPU TOOL — SUBMIT + DUMP
==============================================================================
Unified QPU workflow for the D_M Bell-listener / cavity-offset ghost-channel
operator.

This follows the same storage discipline as the Ghost Oracle Suite QPU tools:

    python ghost_oracle/D_M/d_m_qpu_generate.py submit
    python ghost_oracle/D_M/d_m_qpu_generate.py dump <JOB_ID>

The submit command writes metadata FIRST:

    data/dm_job_<JOB_ID>.json
    data/latest_dm_job.json

The dump command requires that metadata, fetches a completed SamplerV2 job, and
freezes a canonical qproj base:

    data/dm_data_bell_listener_<JOB_ID>.npz
    data/latest_dm_data.json
    data/latest_dm_qpu_data.json       # compatibility alias

D_M working target
------------------
D_M listens for Bell-witness correlation in the ghost channel.

It does NOT prepare a Bell state, reconstruct a density matrix, use an ancilla,
apply dynamical decoupling, or reserve neighboring qubits by default. It places
bare coherent two-qubit listener tiles across the chip, lets neighboring tiles
share the silicon, sweeps cavity-delay offsets, and reads the four witness
correlators:

    XY, YZ, ZY, YX

Each tile is:

    H(q0), H(q1)       # coherence present, no forced entanglement
    delay(base + t*offset_dt) on both qubits
    rotate into assigned witness basis
    measure q0/q1

The qproj dump stores raw pair bits AND all tile metadata in a stable schema so
one downstream parser can consume D_M alongside S_M, F_M, G_M, T_S, etc.
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    _HAVE_QISKIT = True
except Exception:
    ClassicalRegister = None
    QuantumCircuit = None
    QuantumRegister = None
    generate_preset_pass_manager = None
    _HAVE_QISKIT = False

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    _HAVE_RUNTIME = True
except Exception:
    QiskitRuntimeService = None
    Sampler = None
    _HAVE_RUNTIME = False

warnings.filterwarnings("ignore", category=DeprecationWarning)


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_BACKEND = "ibm_marrakesh"
DEFAULT_SHOTS = 4096
DEFAULT_OPT_LEVEL = 0

QUBITS_PER_TILE = 2
DEFAULT_BASE_DELAYS_DT = [0, 256, 1024, 4096, 16384]
DEFAULT_OFFSET_DT = 128

# Basis ids: 0=X, 1=Y, 2=Z
WITNESS_PAIRS = [(0, 1), (1, 2), (2, 1), (1, 0)]
BASIS_LABELS = ["X", "Y", "Z"]
WITNESS_LABELS = [BASIS_LABELS[a] + BASIS_LABELS[b] for a, b in WITNESS_PAIRS]

# Inherited suite-style default exclusions for Marrakesh-ish layouts.
DEFAULT_EXCLUDED_QUBITS = [15, 19, 29, 38, 49, 51, 67, 70, 114, 116, 118, 120, 122, 147]


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


# =============================================================================
# COMMON HELPERS
# =============================================================================

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


def public_attrs(obj: Any) -> List[str]:
    return [a for a in dir(obj) if not a.startswith("_")]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_submit_deps() -> None:
    if not _HAVE_QISKIT or not _HAVE_RUNTIME:
        raise RuntimeError(
            "Qiskit submit dependencies are not available. Install qiskit and "
            "qiskit-ibm-runtime, then retry the submit command."
        )


def require_runtime() -> None:
    if not _HAVE_RUNTIME:
        raise RuntimeError(
            "qiskit-ibm-runtime is not installed. Install it, then retry the dump command."
        )


def build_service(channel: str = "ibm_quantum_platform", instance: Optional[str] = None) -> Any:
    kwargs: Dict[str, Any] = {"channel": channel}
    token = os.environ.get("IBM_QUANTUM_TOKEN", "").strip()
    if token:
        kwargs["token"] = token
    if instance:
        kwargs["instance"] = instance
    return QiskitRuntimeService(**kwargs)


def get_backend_name(job: Any) -> str:
    try:
        b = job.backend()
        name = getattr(b, "name", None)
        return str(name() if callable(name) else name)
    except Exception:
        return "unknown"


def result_len(result: Any) -> int:
    try:
        return len(result)
    except Exception:
        return 1


def get_databin_for_pub(result: Any, pub_index: int) -> Any:
    try:
        pub = result[pub_index]
    except Exception as e:
        raise RuntimeError(
            f"Could not access result[{pub_index}]. "
            f"Result type={type(result)} attrs={public_attrs(result)}"
        ) from e
    if not hasattr(pub, "data"):
        raise RuntimeError(f"result[{pub_index}] has no .data")
    return pub.data


def bitarray_to_columns(bitarray: Any, nbits: int, reverse_bits: bool = False) -> np.ndarray:
    """
    Convert Qiskit SamplerV2 BitArray-like register to shape (shots, nbits).

    This mirrors the S_M dump path: prefer to_bool_array(), fall back to packed
    raw bytes, and only reverse bits if explicitly requested.
    """
    arr: Optional[np.ndarray] = None

    if hasattr(bitarray, "to_bool_array"):
        try:
            b = np.asarray(bitarray.to_bool_array()).astype(np.uint8)
            if b.ndim == 1:
                b = b.reshape(1, -1)
            elif b.ndim > 2:
                b = b.reshape((-1, b.shape[-1]))
            if b.shape[1] >= nbits:
                arr = b[:, :nbits].astype(np.uint8)
        except Exception:
            arr = None

    if arr is None and hasattr(bitarray, "array"):
        try:
            raw = np.asarray(bitarray.array, dtype=np.uint8)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            elif raw.ndim > 2:
                raw = raw.reshape((-1, raw.shape[-1]))
            bits = np.unpackbits(raw, axis=1, bitorder="little")
            if bits.shape[1] >= nbits:
                arr = bits[:, :nbits].astype(np.uint8)
        except Exception:
            arr = None

    # Last-resort compatibility for older result objects.
    if arr is None and hasattr(bitarray, "get_bitstrings"):
        try:
            rows: List[List[int]] = []
            for bs in bitarray.get_bitstrings():
                s = bs[::-1] if reverse_bits else bs
                if len(s) < nbits:
                    raise ValueError(f"expected at least {nbits} bits, got {len(s)} from {bs!r}")
                rows.append([int(ch) for ch in s[:nbits]])
            arr = np.asarray(rows, dtype=np.uint8)
            reverse_bits = False  # already handled above if requested
        except Exception:
            arr = None

    if arr is None:
        raise RuntimeError(
            f"Could not convert BitArray-like object. "
            f"type={type(bitarray)} attrs={public_attrs(bitarray)}"
        )

    arr = arr[:, :nbits].astype(np.uint8)
    if reverse_bits:
        arr = arr[:, ::-1].copy()
    return arr


def extract_register(databin: Any, name: str, nbits: int, reverse_bits: bool = False) -> np.ndarray:
    if not hasattr(databin, name):
        raise KeyError(f"register {name!r} not found. Available: {public_attrs(databin)}")
    return bitarray_to_columns(getattr(databin, name), nbits, reverse_bits=reverse_bits)


# =============================================================================
# CALIBRATION SNAPSHOT
# =============================================================================

def snapshot_calibration(backend: Any, phys_qubits: Sequence[int]) -> Dict[str, Any]:
    target = backend.target
    cal: Dict[str, Any] = {
        "single_qubit": {},
        "readout": {},
        "idling": {},
        "two_qubit": {},
        "meta": {"captured_from": str(getattr(backend, "name", "unknown"))},
    }

    for q in phys_qubits:
        sq_err = None
        for gate in ("sx", "x", "rz"):
            try:
                props = target[gate].get((q,))
                if props is not None and getattr(props, "error", None) is not None:
                    sq_err = float(props.error)
                    break
            except Exception:
                pass
        cal["single_qubit"][str(q)] = sq_err

    for q in phys_qubits:
        ro = None
        try:
            props = target["measure"].get((q,))
            if props is not None and getattr(props, "error", None) is not None:
                ro = float(props.error)
        except Exception:
            pass
        cal["readout"][str(q)] = ro

    for q in phys_qubits:
        try:
            qp = target.qubit_properties[q]
            t1 = getattr(qp, "t1", None)
            t2 = getattr(qp, "t2", None)
            cal["idling"][str(q)] = {
                "t1": float(t1) if t1 is not None else None,
                "t2": float(t2) if t2 is not None else None,
            }
        except Exception:
            cal["idling"][str(q)] = None

    pset = set(int(q) for q in phys_qubits)
    try:
        edges = backend.coupling_map.get_edges()
    except Exception:
        edges = []
    for u, v in edges:
        if u in pset and v in pset:
            err = None
            for gate in ("ecr", "cz", "cx"):
                try:
                    props = target[gate].get((u, v))
                    if props is not None and getattr(props, "error", None) is not None:
                        err = float(props.error)
                        break
                except Exception:
                    pass
            cal["two_qubit"][f"{u}_{v}"] = err

    return cal


# =============================================================================
# TILE METADATA
# =============================================================================

@dataclass
class DMTileMeta:
    tile: int
    rung_index: int
    witness_index: int
    base_delay_dt: int
    offset_dt: int
    total_delay_dt: int
    basis_q0: int
    basis_q1: int
    witness_label: str
    physical_q0: int
    physical_q1: int
    role: str


def required_tiles(base_delays_dt: Sequence[int], witness_pairs: Sequence[Tuple[int, int]]) -> int:
    return int(len(base_delays_dt) * len(witness_pairs))


def build_tile_plan(
    clusters: Sequence[Sequence[int]],
    base_delays_dt: Sequence[int],
    offset_dt: int,
    witness_pairs: Sequence[Tuple[int, int]],
) -> List[DMTileMeta]:
    if not base_delays_dt:
        raise ValueError("base_delays_dt must not be empty")
    if not witness_pairs:
        raise ValueError("witness_pairs must not be empty")

    plan: List[DMTileMeta] = []
    tile = 0
    for rung_index, base_delay in enumerate(base_delays_dt):
        for witness_index, (b0, b1) in enumerate(witness_pairs):
            if tile >= len(clusters):
                return plan
            q0, q1 = [int(x) for x in clusters[tile]]
            off = tile * int(offset_dt)
            total = int(base_delay) + off
            label = BASIS_LABELS[int(b0)] + BASIS_LABELS[int(b1)]
            role = f"rung{rung_index}_base{int(base_delay)}_off{off}_{label}"
            plan.append(
                DMTileMeta(
                    tile=int(tile),
                    rung_index=int(rung_index),
                    witness_index=int(witness_index),
                    base_delay_dt=int(base_delay),
                    offset_dt=int(off),
                    total_delay_dt=int(total),
                    basis_q0=int(b0),
                    basis_q1=int(b1),
                    witness_label=str(label),
                    physical_q0=int(q0),
                    physical_q1=int(q1),
                    role=str(role),
                )
            )
            tile += 1
    return plan


# =============================================================================
# QUBIT CLUSTER SELECTION
# =============================================================================

def get_readout_error_from_target(target: Any, qubit: int, fallback: float = 0.5) -> float:
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


def build_adjacency(backend: Any) -> Dict[int, set]:
    adj: Dict[int, set] = {i: set() for i in range(backend.num_qubits)}
    for u, v in backend.coupling_map.get_edges():
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    return adj


def find_qubit_clusters(
    backend: Any,
    cluster_size: int = QUBITS_PER_TILE,
    max_tiles: int = 64,
    excluded_qubits: Optional[Sequence[int]] = None,
    reserve_neighbors: bool = False,
) -> List[List[int]]:
    """
    Greedy connected-cluster selection.

    D_M defaults to reserve_neighbors=False because shared chip space is the
    experiment. Turn --reserve-neighbors on only as a destructive control / sanity
    check, not for the main listener base.
    """
    excluded = set(int(q) for q in (excluded_qubits or []))
    target = backend.target
    adjacency = build_adjacency(backend)

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
            clusters.append([int(x) for x in cluster])

        if len(clusters) >= max_tiles:
            break

    return clusters


# =============================================================================
# CIRCUIT
# =============================================================================

def apply_basis_rotation(qc: Any, qubit: Any, basis: int) -> None:
    if basis == 0:          # X
        qc.h(qubit)
    elif basis == 1:        # Y
        qc.sdg(qubit)
        qc.h(qubit)
    elif basis == 2:        # Z
        pass
    else:
        raise ValueError(f"unknown basis id: {basis}")


def build_circuit(tile_plan: Sequence[DMTileMeta]) -> Tuple[Any, Dict[str, Any]]:
    """Build one packed D_M listener circuit with one 2-bit register per tile."""
    num_tiles = len(tile_plan)
    qr = QuantumRegister(num_tiles * QUBITS_PER_TILE, name="q")
    qc = QuantumCircuit(qr)

    creg_names: List[str] = []

    for meta in tile_plan:
        t = int(meta.tile)
        pair_cr = ClassicalRegister(2, name=f"dm_pair_tile{t}")
        qc.add_register(pair_cr)
        creg_names.append(pair_cr.name)

        q0 = qr[t * QUBITS_PER_TILE + 0]
        q1 = qr[t * QUBITS_PER_TILE + 1]

        qc.barrier([q0, q1])
        qc.h(q0)
        qc.h(q1)

        if int(meta.total_delay_dt) > 0:
            qc.delay(int(meta.total_delay_dt), q0, unit="dt")
            qc.delay(int(meta.total_delay_dt), q1, unit="dt")

        qc.barrier([q0, q1])
        apply_basis_rotation(qc, q0, int(meta.basis_q0))
        apply_basis_rotation(qc, q1, int(meta.basis_q1))
        qc.measure([q0, q1], pair_cr)

    return qc, {"pair_registers": creg_names}


# =============================================================================
# SUBMIT / METADATA RESOLUTION
# =============================================================================

def add_submit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", default=DEFAULT_BACKEND)
    p.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--optimization-level", type=int, choices=[0, 1, 2, 3], default=DEFAULT_OPT_LEVEL)
    p.add_argument("--base-delays-dt", type=int, nargs="+", default=DEFAULT_BASE_DELAYS_DT)
    p.add_argument("--offset-dt", type=int, default=DEFAULT_OFFSET_DT)
    p.add_argument("--exclude", "--excluded-qubits", type=int, nargs="*", default=DEFAULT_EXCLUDED_QUBITS)
    p.add_argument("--reserve-neighbors", action="store_true")
    p.add_argument("--channel", default="ibm_quantum_platform")
    p.add_argument("--instance", default=None)
    p.add_argument("--dry-run", action="store_true")


def add_dump_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("job_id", nargs="?", default=None, help="IBM Runtime job ID. Defaults to latest submitted D_M job.")
    p.add_argument("--meta", default=None, help="Optional metadata path. Usually not needed for jobs from this submit command.")
    p.add_argument("--save", "--out", dest="save", default=None, help="Output .npz path.")
    p.add_argument("--channel", default="ibm_quantum_platform")
    p.add_argument("--instance", default=None)
    p.add_argument("--reverse-bits", action="store_true")
    p.add_argument("--list-registers", action="store_true")

    # Legacy rescue only. Fresh jobs should never need this.
    p.add_argument("--repair-missing-meta", action="store_true",
                   help="Repair old one-circuit D_M jobs that were dumped/submitted without metadata.")
    p.add_argument("--num-tiles", type=int, default=None,
                   help="Only used with --repair-missing-meta if registers cannot be inferred.")
    p.add_argument("--base-delays-dt", type=int, nargs="+", default=DEFAULT_BASE_DELAYS_DT,
                   help="Only used with --repair-missing-meta.")
    p.add_argument("--offset-dt", type=int, default=DEFAULT_OFFSET_DT,
                   help="Only used with --repair-missing-meta.")


def resolve_job_and_meta(job_id_arg: Optional[str], meta_arg: Optional[str]) -> Tuple[str, Optional[Path]]:
    if job_id_arg is None:
        latest = DATA_DIR / "latest_dm_job.json"
        if not latest.exists():
            raise FileNotFoundError(
                "No JOB_ID provided and data/latest_dm_job.json does not exist. "
                "Run the submit command first or pass JOB_ID explicitly."
            )
        obj = load_json(latest)
        return str(obj["job_id"]), Path(obj["meta"])

    job_id = str(job_id_arg)
    if meta_arg:
        return job_id, Path(meta_arg)

    candidates = [
        DATA_DIR / f"dm_job_{job_id}.json",                 # fixed current schema
        DATA_DIR / f"dm_job_{job_id}_submit_meta.json",     # legacy current D_M schema
        Path(f"dm_job_{job_id}.json"),
        Path(f"dm_job_{job_id}_submit_meta.json"),
    ]
    for p in candidates:
        if p.exists():
            return job_id, p

    return job_id, None


def infer_tile_indices(databin: Any) -> List[int]:
    attrs = public_attrs(databin)
    tiles = set()
    for name in attrs:
        m = re.fullmatch(r"dm_pair_tile(\d+)", name)
        if m:
            tiles.add(int(m.group(1)))
    return sorted(tiles)


def repair_metadata_from_registers(
    job_id: str,
    backend_name: str,
    databin: Any,
    num_tiles: Optional[int],
    base_delays_dt: Sequence[int],
    offset_dt: int,
) -> Dict[str, Any]:
    tile_indices = infer_tile_indices(databin)
    if not tile_indices and num_tiles is not None:
        tile_indices = list(range(int(num_tiles)))
    if not tile_indices:
        raise RuntimeError("Could not infer dm_pair_tile* registers for repair mode.")

    # Unknown physical clusters in repair mode. Preserve the logical tile plan.
    clusters = [[-1, -1] for _ in tile_indices]
    plan = build_tile_plan(clusters, base_delays_dt, offset_dt, WITNESS_PAIRS)
    plan = plan[:len(tile_indices)]

    return {
        "schema": "ghost_oracle.dm.submit_meta.v2.repaired",
        "suite": "Ghost Oracle Suite",
        "operator": "D_M",
        "substrate": "qproj",
        "circuit_family": "bell_listener_cavity_offset_v1",
        "job_id": str(job_id),
        "backend": str(backend_name),
        "shots": None,
        "qubits_per_tile": QUBITS_PER_TILE,
        "num_tiles": int(len(plan)),
        "clusters": clusters[:len(plan)],
        "global_layout": [],
        "base_delays_dt": [int(x) for x in base_delays_dt],
        "offset_dt": int(offset_dt),
        "witness_pairs": [list(wp) for wp in WITNESS_PAIRS],
        "witness_labels": WITNESS_LABELS,
        "basis_labels": BASIS_LABELS,
        "dd_enabled": False,
        "reserve_neighbors": False,
        "optimization_level": None,
        "tile_plan": [asdict(m) for m in plan],
        "notes": "Repaired metadata from register names only. Physical qubit layout unavailable.",
    }


def command_submit(args: argparse.Namespace) -> None:
    require_submit_deps()

    base_delays_dt = [int(x) for x in args.base_delays_dt]
    offset_dt = int(args.offset_dt)
    need = required_tiles(base_delays_dt, WITNESS_PAIRS)
    max_tiles = int(args.max_tiles) if args.max_tiles is not None else need

    print(f"\n{'=' * 90}")
    print("  D_M SUBMIT — BELL-LISTENER CAVITY-OFFSET JOB")
    print(f"{'=' * 90}")
    print(f"  Backend          : {args.backend}")
    print(f"  Shots            : {args.shots}")
    print(f"  Qubits/tile      : {QUBITS_PER_TILE}  (no ancilla)")
    print(f"  Base delays dt   : {base_delays_dt}")
    print(f"  Offset step dt   : {offset_dt}")
    print(f"  Witness pairs    : {WITNESS_LABELS}")
    print(f"  Tiles for full   : {need} = {len(base_delays_dt)} rungs x {len(WITNESS_PAIRS)} witnesses")
    print(f"  Neighbor reserve : {'on' if args.reserve_neighbors else 'off (shared space is the experiment)'}")
    print(f"  Data dir         : {DATA_DIR}")

    service = build_service(channel=args.channel, instance=args.instance)
    backend = service.backend(args.backend)

    print("\n[SETUP] Selecting connected two-qubit listener tiles...")
    clusters = find_qubit_clusters(
        backend=backend,
        cluster_size=QUBITS_PER_TILE,
        max_tiles=max_tiles,
        excluded_qubits=args.exclude,
        reserve_neighbors=bool(args.reserve_neighbors),
    )
    if not clusters:
        raise RuntimeError("No suitable two-qubit clusters found on this backend.")

    tile_plan = build_tile_plan(clusters, base_delays_dt, offset_dt, WITNESS_PAIRS)
    num_tiles = len(tile_plan)
    if num_tiles < need:
        print(f"  [note] selected {num_tiles} tiles < {need}; trailing rungs are dropped.")

    full_rungs = []
    partial_rungs = []
    for base_delay in base_delays_dt:
        c = sum(1 for m in tile_plan if m.base_delay_dt == int(base_delay))
        if c == len(WITNESS_PAIRS):
            full_rungs.append(int(base_delay))
        elif c > 0:
            partial_rungs.append(int(base_delay))
    print(f"  Found tiles                 : {num_tiles}")
    print(f"  Witness-complete rungs      : {full_rungs}")
    if partial_rungs:
        print(f"  Partial rungs               : {partial_rungs}")

    for i, meta in enumerate(tile_plan[:8]):
        print(f"  tile {i:02d}: cluster={clusters[i]} meta={asdict(meta)}")
    if num_tiles > 8:
        print(f"  ... {num_tiles - 8} more tiles")

    print(f"\n[BUILD] Constructing listener circuit ({num_tiles * QUBITS_PER_TILE} qubits)...")
    qc, creg_names = build_circuit(tile_plan)
    global_layout = [int(q) for cluster in clusters[:num_tiles] for q in cluster]

    print("[BUILD] Transpiling with fixed initial layout...")
    pm = generate_preset_pass_manager(
        optimization_level=int(args.optimization_level),
        backend=backend,
        initial_layout=global_layout,
    )
    isa = pm.run(qc)

    print("\n[CIRCUIT]")
    print(f"  depth          : {isa.depth()}")
    print(f"  size           : {isa.size()}")
    print(f"  width          : {isa.width()}")
    print(f"  classical bits : {isa.num_clbits}")

    if args.dry_run:
        print("\n[DRY RUN] Circuit built and transpiled. Not submitting.")
        return

    print(f"\n[SUBMIT] Sending D_M listener job to {args.backend}...")
    sampler = Sampler(mode=backend)
    job = sampler.run([isa], shots=int(args.shots))
    job_id = str(job.job_id())

    all_used = sorted({int(q) for q in global_layout})
    calibration = None
    try:
        calibration = snapshot_calibration(backend, all_used)
        print(f"  [cal] snapshotted calibration for {len(all_used)} qubits")
    except Exception as e:
        print(f"  [cal][warn] calibration snapshot failed: {e}")

    meta = {
        "schema": "ghost_oracle.dm.submit_meta.v2",
        "suite": "Ghost Oracle Suite",
        "operator": "D_M",
        "substrate": "qproj",
        "circuit_family": "bell_listener_cavity_offset_v1",
        "job_id": job_id,
        "backend": str(args.backend),
        "shots": int(args.shots),
        "qubits_per_tile": QUBITS_PER_TILE,
        "num_tiles": int(num_tiles),
        "clusters": [[int(x) for x in c] for c in clusters[:num_tiles]],
        "global_layout": global_layout,
        "base_delays_dt": base_delays_dt,
        "offset_dt": int(offset_dt),
        "witness_pairs": [list(map(int, wp)) for wp in WITNESS_PAIRS],
        "witness_labels": WITNESS_LABELS,
        "basis_labels": BASIS_LABELS,
        "dd_enabled": False,
        "reserve_neighbors": bool(args.reserve_neighbors),
        "optimization_level": int(args.optimization_level),
        "circuits": [{
            "pub_index": 0,
            "num_tiles": int(num_tiles),
            "creg_names": creg_names,
            "tile_plan": [asdict(m) for m in tile_plan],
            "depth_pre_transpile": int(qc.depth()),
            "depth_isa": int(isa.depth()),
            "size_isa": int(isa.size()),
            "width_isa": int(isa.width()),
            "num_clbits_isa": int(isa.num_clbits),
        }],
        "tile_plan": [asdict(m) for m in tile_plan],
        "calibration": calibration,
        "protocol": "D_M Bell-listener cavity-offset ghost-channel probe",
        "notes": (
            "No ancilla, no forced Bell prep, no entangling gate, no DD. "
            "Default neighbor reservation is off because shared chip space is the signal path."
        ),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = DATA_DIR / f"dm_job_{job_id}.json"
    latest_path = DATA_DIR / "latest_dm_job.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(meta), f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump({"job_id": job_id, "meta": str(meta_path)}, f, indent=2)

    print(f"\n{'=' * 90}")
    print("  D_M JOB SUBMITTED")
    print(f"{'=' * 90}")
    print(f"  Job ID   : {job_id}")
    print(f"  Metadata : {meta_path}")
    print(f"  Latest   : {latest_path}")
    print("\n  Dump command:")
    print(f"    python ghost_oracle/D_M/d_m_qpu_generate.py dump {job_id}")
    print(f"{'=' * 90}\n")


# =============================================================================
# DUMP
# =============================================================================

def default_output(job_id: str, meta: Dict[str, Any]) -> Path:
    family = str(meta.get("circuit_family", "bell_listener")).replace("_v1", "")
    return DATA_DIR / f"dm_data_{family}_{job_id}.npz"


def compute_pair_stats(pair: np.ndarray) -> Dict[str, np.ndarray]:
    # pair shape: (tiles, shots, 2), bits 0/1. Map 0->+1 and 1->-1.
    s0 = 1.0 - 2.0 * pair[:, :, 0].astype(np.float64)
    s1 = 1.0 - 2.0 * pair[:, :, 1].astype(np.float64)
    corr = np.mean(s0 * s1, axis=1)
    mean0 = np.mean(s0, axis=1)
    mean1 = np.mean(s1, axis=1)
    connected = corr - mean0 * mean1

    b0 = pair[:, :, 0]
    b1 = pair[:, :, 1]
    p00 = np.mean((b0 == 0) & (b1 == 0), axis=1)
    p01 = np.mean((b0 == 0) & (b1 == 1), axis=1)
    p10 = np.mean((b0 == 1) & (b1 == 0), axis=1)
    p11 = np.mean((b0 == 1) & (b1 == 1), axis=1)

    return {
        "tile_corr": corr.astype(np.float64),
        "tile_connected_corr": connected.astype(np.float64),
        "tile_mean_q0": mean0.astype(np.float64),
        "tile_mean_q1": mean1.astype(np.float64),
        "tile_p00": p00.astype(np.float64),
        "tile_p01": p01.astype(np.float64),
        "tile_p10": p10.astype(np.float64),
        "tile_p11": p11.astype(np.float64),
    }


def command_dump(args: argparse.Namespace) -> None:
    require_runtime()

    job_id, meta_path = resolve_job_and_meta(args.job_id, args.meta)

    print(f"\n{'=' * 96}")
    print("  D_M DUMP — QISKIT RUNTIME DATA")
    print(f"{'=' * 96}")
    print(f"  Job ID   : {job_id}")

    print("\n[FETCH] Connecting to Qiskit Runtime...")
    service = build_service(channel=args.channel, instance=args.instance)
    job = service.job(job_id)
    result = job.result()
    backend_name = get_backend_name(job)

    databin = get_databin_for_pub(result, 0)
    if args.list_registers:
        for i in range(result_len(result)):
            db = get_databin_for_pub(result, i)
            print(f"  PUB {i} registers: {public_attrs(db)}")

    if meta_path is None:
        if not args.repair_missing_meta:
            raise FileNotFoundError(
                f"No metadata found for D_M job {job_id}.\n"
                "This generator now refuses to create a qproj base with unknown witness/delay metadata.\n"
                "For old jobs only, rerun with --repair-missing-meta to reconstruct the logical "
                "XY/YZ/ZY/YX tile plan from register order."
            )
        print("  [repair] No metadata found; repairing logical tile metadata from register names.")
        meta = repair_metadata_from_registers(
            job_id=job_id,
            backend_name=backend_name,
            databin=databin,
            num_tiles=args.num_tiles,
            base_delays_dt=args.base_delays_dt,
            offset_dt=int(args.offset_dt),
        )
        meta_path_display = "<repaired from registers>"
    else:
        meta = load_json(meta_path)
        meta_path_display = str(meta_path)

    print(f"  Metadata : {meta_path_display}")
    print(f"  Backend  : {backend_name}")

    tile_plan_raw = meta.get("tile_plan")
    if not tile_plan_raw and meta.get("circuits"):
        tile_plan_raw = meta["circuits"][0].get("tile_plan")
    if not tile_plan_raw:
        raise KeyError("metadata has no tile_plan")

    tile_plan = [dict(m) for m in tile_plan_raw]
    num_tiles = int(len(tile_plan))

    print(f"\n[EXTRACT] Reading {num_tiles} D_M pair registers...")
    pair_tiles: List[np.ndarray] = []
    basis_tiles: List[np.ndarray] = []

    for out_t, m in enumerate(tile_plan):
        tile = int(m.get("tile", out_t))
        reg_name = f"dm_pair_tile{tile}"
        pair_arr = extract_register(databin, reg_name, 2, reverse_bits=bool(args.reverse_bits))
        pair_tiles.append(pair_arr.astype(np.uint8))
        basis = np.asarray([int(m["basis_q0"]), int(m["basis_q1"])], dtype=np.int8)
        basis_tiles.append(basis)

        s0 = 1.0 - 2.0 * pair_arr[:, 0].astype(float)
        s1 = 1.0 - 2.0 * pair_arr[:, 1].astype(float)
        corr = float(np.mean(s0 * s1))
        if out_t < 8 or out_t % 8 == 0:
            print(
                f"  tile{tile:02d}: pair={pair_arr.shape}, "
                f"base={int(m['base_delay_dt'])}, off={int(m['offset_dt'])}, "
                f"total={int(m['total_delay_dt'])}, witness={m.get('witness_label')}, "
                f"<P0xP1>={corr:+.4f}"
            )

    shots = int(pair_tiles[0].shape[0]) if pair_tiles else 0
    for i, arr in enumerate(pair_tiles):
        if int(arr.shape[0]) != shots:
            raise RuntimeError(f"shot mismatch: tile {i} has {arr.shape[0]} shots, expected {shots}")

    pair = np.stack(pair_tiles, axis=0).astype(np.uint8)
    basis = np.stack(basis_tiles, axis=0).astype(np.int8)
    stats = compute_pair_stats(pair)

    tile_cluster = np.asarray(
        [[int(m.get("physical_q0", -1)), int(m.get("physical_q1", -1))] for m in tile_plan],
        dtype=np.int32,
    )

    saved: Dict[str, Any] = {
        "schema": np.array("dm_bell_listener_qproj_v2"),
        "suite": np.array("Ghost Oracle Suite"),
        "operator": np.array("D_M"),
        "substrate": np.array("qproj"),
        "circuit_family": np.array(str(meta.get("circuit_family", "bell_listener_cavity_offset_v1"))),
        "job_id": np.array(str(job_id)),
        "backend": np.array(str(backend_name)),
        "shots": np.array(shots, dtype=np.int64),
        "num_tiles": np.array(num_tiles, dtype=np.int64),
        "qubits_per_tile": np.array(QUBITS_PER_TILE, dtype=np.int64),

        "base_delays_dt": np.asarray(meta.get("base_delays_dt", DEFAULT_BASE_DELAYS_DT), dtype=np.int32),
        "offset_dt": np.array(int(meta.get("offset_dt", DEFAULT_OFFSET_DT)), dtype=np.int32),
        "witness_pairs": np.asarray(meta.get("witness_pairs", WITNESS_PAIRS), dtype=np.int8),
        "basis_labels": np.asarray(meta.get("basis_labels", BASIS_LABELS)),
        "witness_labels": np.asarray(meta.get("witness_labels", WITNESS_LABELS)),

        "tile_indices": np.asarray([int(m.get("tile", i)) for i, m in enumerate(tile_plan)], dtype=np.int32),
        "tile_rung_index": np.asarray([int(m.get("rung_index", i // 4)) for i, m in enumerate(tile_plan)], dtype=np.int32),
        "tile_witness_index": np.asarray([int(m.get("witness_index", i % 4)) for i, m in enumerate(tile_plan)], dtype=np.int32),
        "tile_base_delay_dt": np.asarray([int(m.get("base_delay_dt", -1)) for m in tile_plan], dtype=np.int32),
        "tile_offset_dt": np.asarray([int(m.get("offset_dt", -1)) for m in tile_plan], dtype=np.int32),
        "tile_total_delay_dt": np.asarray([int(m.get("total_delay_dt", -1)) for m in tile_plan], dtype=np.int32),
        "tile_basis_q0": np.asarray([int(m.get("basis_q0", -1)) for m in tile_plan], dtype=np.int8),
        "tile_basis_q1": np.asarray([int(m.get("basis_q1", -1)) for m in tile_plan], dtype=np.int8),
        "tile_witness_label": np.asarray([str(m.get("witness_label", "??")) for m in tile_plan]),
        "tile_role": np.asarray([str(m.get("role", "unknown")) for m in tile_plan]),
        "tile_cluster": tile_cluster,
        "tile_physical_q0": tile_cluster[:, 0].astype(np.int32),
        "tile_physical_q1": tile_cluster[:, 1].astype(np.int32),

        "pair": pair,
        "basis": basis,

        "tile_meta_json": np.array(json.dumps(json_safe(tile_plan))),
        "submit_meta_json": np.array(json.dumps(json_safe(meta))),
    }
    saved.update(stats)

    for t in range(num_tiles):
        saved[f"pair_tile{t}"] = pair[t]
        saved[f"basis_tile{t}"] = basis[t]

    out_path = Path(args.save) if args.save else default_output(job_id, meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **saved)

    latest = {
        "job_id": str(job_id),
        "npz": str(out_path),
        "meta": str(meta_path) if meta_path is not None else None,
        "operator": "D_M",
        "substrate": "qproj",
        "schema": "dm_bell_listener_qproj_v2",
    }

    latest_data = DATA_DIR / "latest_dm_data.json"
    latest_qpu_data = DATA_DIR / "latest_dm_qpu_data.json"
    with open(latest_data, "w", encoding="utf-8") as f:
        json.dump(latest, f, indent=2)
    with open(latest_qpu_data, "w", encoding="utf-8") as f:
        json.dump({
            **latest,
            "path": str(out_path),
            "shots": int(shots),
            "num_tiles": int(num_tiles),
        }, f, indent=2)

    print(f"\n{'=' * 96}")
    print("  D_M DUMP COMPLETE")
    print(f"{'=' * 96}")
    print(f"  Output  : {out_path}")
    print(f"  Latest  : {latest_data}")
    print(f"  Alias   : {latest_qpu_data}")
    print(f"  Backend : {backend_name}")
    print(f"  Tiles   : {num_tiles}")
    print(f"  Shots   : {shots}")
    print(f"{'=' * 96}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D_M QPU tool — submit Bell-listener QPU jobs and dump Runtime results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Submit the D_M Bell-listener cavity-offset experiment.")
    add_submit_args(p_submit)

    p_dump = sub.add_parser("dump", help="Dump a completed D_M QPU job to canonical qproj .npz.")
    add_dump_args(p_dump)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "submit":
            command_submit(args)
        elif args.command == "dump":
            command_dump(args)
        else:
            raise ValueError(f"unknown command: {args.command}")
    except Exception as e:
        sys.exit(f"[FATAL] {e}")


if __name__ == "__main__":
    main()
