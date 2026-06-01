#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
T_S QPU TOOL — TEMPORAL STRESS METRIC SUBMIT + DUMP
==============================================================================
Probe 01 for the Ghost Oracle Suite T_S package.

T_S means:

    Temporal Stress Metric

This file follows the S_M convention:

    python ghost_oracle/T_S/t_s_qpu_generate.py submit
    python ghost_oracle/T_S/t_s_qpu_generate.py dump <JOB_ID>

The submit command builds a QPU-native delay/channel field experiment.  It does
not treat time as a passive round index.  It explicitly sweeps structured delay
values and delay insertion sites, then records repeated parity-probe field
measurements.

The useful T_S object is:

    F[mode, delay_site, delay_value, shot, round, edge]

where each field bit is a measured parity/probe response between neighboring
channel qubits.

Downstream analysis computes a delay-aware stress tensor:

    DτF = F[τ+1, r, x] XOR F[τ,   r, x]
    DrF = F[τ,   r+1, x] XOR F[τ, r, x]
    DxF = F[τ,   r, x+1] XOR F[τ, r, x]

    T_ab = <D_a F D_b F>,    a,b ∈ {τ, r, x}

This is intentionally a T_S probe, not an S_M property.

Outputs
-------
submit:
    ghost_oracle/T_S/data/ts_job_<JOB_ID>.json
    ghost_oracle/T_S/data/latest_ts_job.json

dump:
    ghost_oracle/T_S/data/ts_data_<JOB_ID>.npz
    ghost_oracle/T_S/data/latest_ts_data.json

Default probe
-------------
backend       : ibm_marrakesh
channels      : 8
rounds        : 6
shots         : 4096
delay values  : 0 1 2 4 8 16
delay unit    : dt
delay sites   : pre_coupling post_coupling post_perturb
modes         : clean phase_shear local_shock

Notes
-----
- Measurements are taken on reusable probe qubits, not directly on the channel
  qubits each round.
- The final channel measurement is diagnostic only.
- The field/order/delay structure is the object.
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
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
DEFAULT_CHANNELS = 8
DEFAULT_ROUNDS = 6
DEFAULT_DELAYS = [0, 1, 2, 4, 8, 16]
DEFAULT_DELAY_UNIT = "dt"
DEFAULT_DELAY_SITES = ["pre_coupling", "post_coupling", "post_perturb"]
DEFAULT_MODES = ["clean", "phase_shear", "local_shock"]
DEFAULT_OPT_LEVEL = 1
DEFAULT_SEED = 20260531
DEFAULT_EXCLUDED_QUBITS: List[int] = []


# =============================================================================
# PATHS
# =============================================================================

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "requirements.txt").exists() or (p / ".git").exists():
            return p
    parents = cur.parents
    return parents[2] if len(parents) >= 3 else cur


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


# =============================================================================
# SERVICE / CALIBRATION
# =============================================================================

def build_service() -> QiskitRuntimeService:
    token = os.environ.get("IBM_QUANTUM_TOKEN", "").strip()
    if token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    return QiskitRuntimeService(channel="ibm_quantum_platform")


def snapshot_calibration(backend, phys_qubits: Sequence[int]) -> Dict[str, Any]:
    """Best-effort calibration snapshot for reproducibility and controls."""
    target = backend.target
    cal: Dict[str, Any] = {
        "single_qubit": {},
        "readout": {},
        "idling": {},
        "two_qubit": {},
        "meta": {
            "captured_from": getattr(backend, "name", "unknown"),
            "dt": getattr(backend, "dt", None),
        },
    }

    for q in phys_qubits:
        sq_err = None
        for gate in ("sx", "x"):
            try:
                props = target[gate].get((q,))
                if props is not None and props.error is not None:
                    sq_err = float(props.error)
                    break
            except Exception:
                pass
        cal["single_qubit"][str(q)] = sq_err

    for q in phys_qubits:
        ro = None
        try:
            props = target["measure"].get((q,))
            if props is not None and props.error is not None:
                ro = float(props.error)
        except Exception:
            pass
        cal["readout"][str(q)] = ro

    for q in phys_qubits:
        try:
            qp = target.qubit_properties[q]
            cal["idling"][str(q)] = {
                "t1": float(qp.t1) if getattr(qp, "t1", None) is not None else None,
                "t2": float(qp.t2) if getattr(qp, "t2", None) is not None else None,
            }
        except Exception:
            cal["idling"][str(q)] = None

    pset = set(phys_qubits)
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
                    if props is not None and props.error is not None:
                        err = float(props.error)
                        break
                except Exception:
                    pass
            cal["two_qubit"][f"{u}_{v}"] = err

    return cal


# =============================================================================
# LAYOUT DISCOVERY
# =============================================================================

def build_adjacency(backend) -> Dict[int, set]:
    adj = {i: set() for i in range(backend.num_qubits)}
    for u, v in backend.coupling_map.get_edges():
        adj[u].add(v)
        adj[v].add(u)
    return adj


def chain_length_for(channels: int) -> int:
    # Alternating chain:
    #   channel_0, probe_0, channel_1, probe_1, ..., probe_n-2, channel_n-1
    return 2 * int(channels) - 1


def find_path(adj: Dict[int, set], length: int, used: set, n: int) -> Optional[List[int]]:
    def dfs(node: int, path: List[int], seen: set) -> Optional[List[int]]:
        if len(path) == length:
            return list(path)
        for nb in sorted(adj[node]):
            if nb in seen or nb in used:
                continue
            seen.add(nb)
            path.append(nb)
            found = dfs(nb, path, seen)
            if found:
                return found
            path.pop()
            seen.discard(nb)
        return None

    for start in range(n):
        if start in used:
            continue
        found = dfs(start, [start], {start})
        if found:
            return found
    return None


def assign_roles(path: List[int], channels: int) -> Dict[str, Any]:
    channel_q = [int(path[2 * i]) for i in range(channels)]
    probe_q = [int(path[2 * i + 1]) for i in range(channels - 1)]
    return {
        "channels": int(channels),
        "channel_q": channel_q,
        "probe_q": probe_q,
        "path": [int(x) for x in path],
    }


def discover_layout(
    backend,
    channels: int,
    excluded_qubits: Sequence[int],
    reserve_neighbors: bool = True,
) -> Dict[str, Any]:
    adj = build_adjacency(backend)
    used = set(excluded_qubits)
    length = chain_length_for(channels)

    path = find_path(adj, length, used, backend.num_qubits)
    if path is None:
        raise RuntimeError(
            f"no length-{length} path found for channels={channels}. "
            "Try fewer channels, different backend, or --no-reserve-neighbors."
        )

    if reserve_neighbors:
        used.update(path)
        for q in path:
            used.update(adj[q])

    return assign_roles(path, channels)


# =============================================================================
# CIRCUIT HELPERS
# =============================================================================

def angle_for(channel: int, mode: str, seed: int, kind: str) -> float:
    """
    Deterministic small angle generator.

    The angles are deliberately simple: they make the probe reproducible while
    creating non-identical channel coordinates.  In later T_S token-retrieval
    probes, these angles can be replaced by embedding-derived token angles.
    """
    rng = np.random.default_rng(seed + 7919 * channel + 104729 * (0 if kind == "theta" else 1))
    base = float(rng.uniform(-0.35, 0.35))
    ramp = (channel + 1) * 0.03125
    if mode == "phase_shear" and kind == "phi":
        return base + 0.22 * channel
    if mode == "local_shock" and channel == 0 and kind == "theta":
        return base + 0.65
    return base + ramp


def apply_structured_delay(
    qc: QuantumCircuit,
    qr: QuantumRegister,
    virtual_indices: Sequence[int],
    delay_value: int,
    unit: str,
) -> None:
    """
    Insert intentional delay on selected virtual qubits.

    Qiskit's delay instruction is the cleanest representation of structured
    waiting time.  If delay_value is zero, this is a no-op.  If a backend later
    rejects a unit/duration choice, adjust --delay-unit or --delays rather than
    removing the delay axis.
    """
    if int(delay_value) <= 0:
        return
    for vi in virtual_indices:
        qc.delay(int(delay_value), qr[int(vi)], unit=unit)


def apply_channel_coupling(qc: QuantumCircuit, qr: QuantumRegister, channel_v: List[int]) -> None:
    """
    Lightweight nearest-neighbor coupling layer.

    We use an even/odd CX ladder so channel information can propagate while
    keeping depth bounded.  This is not intended as standard attention.  It is a
    physical channel-coupling primitive whose stress response is measured.
    """
    n = len(channel_v)
    for start in (0, 1):
        for i in range(start, n - 1, 2):
            qc.cx(qr[channel_v[i]], qr[channel_v[i + 1]])


def apply_perturbation(
    qc: QuantumCircuit,
    qr: QuantumRegister,
    channel_v: List[int],
    mode: str,
    round_index: int,
) -> None:
    """Apply mode-specific channel perturbation."""
    n = len(channel_v)
    if mode == "clean":
        return

    if mode == "phase_shear":
        for i, vi in enumerate(channel_v):
            qc.rz(0.035 * (round_index + 1) * i, qr[vi])
        return

    if mode == "local_shock":
        center = n // 2
        qc.rx(0.18 * (round_index + 1), qr[channel_v[center]])
        if center - 1 >= 0:
            qc.rz(0.07 * (round_index + 1), qr[channel_v[center - 1]])
        if center + 1 < n:
            qc.rz(-0.07 * (round_index + 1), qr[channel_v[center + 1]])
        return

    if mode == "edge_shock":
        qc.rx(0.18 * (round_index + 1), qr[channel_v[0]])
        return

    raise ValueError(f"unknown T_S mode: {mode}")


def build_ts_circuit(
    roles: Dict[str, Any],
    rounds: int,
    delay_value: int,
    delay_unit: str,
    delay_site: str,
    mode: str,
    seed: int,
) -> Tuple[QuantumCircuit, List[int], Dict[str, Any]]:
    """
    Build one T_S circuit for a single mode × delay_site × delay_value.

    Physical-role convention:
        channel_q: persistent channel state
        probe_q  : measured/reset parity-probe field sites

    Registers:
        field_r<r>  : one bit per edge/probe per round
        channel_final: final channel readout, diagnostic only
    """
    channel_q = list(map(int, roles["channel_q"]))
    probe_q = list(map(int, roles["probe_q"]))
    all_phys = sorted(set(channel_q) | set(probe_q))
    v = {p: i for i, p in enumerate(all_phys)}

    qr = QuantumRegister(len(all_phys), name="q")
    qc = QuantumCircuit(qr)

    field_cregs = []
    for r in range(rounds):
        cr = ClassicalRegister(len(probe_q), name=f"field_r{r}")
        qc.add_register(cr)
        field_cregs.append(cr)

    final_cr = ClassicalRegister(len(channel_q), name="channel_final")
    qc.add_register(final_cr)

    channel_v = [v[p] for p in channel_q]
    probe_v = [v[p] for p in probe_q]
    all_v = channel_v + probe_v

    # Channel state preparation.
    for i, vi in enumerate(channel_v):
        theta = angle_for(i, mode, seed, "theta")
        phi = angle_for(i, mode, seed, "phi")
        qc.ry(theta, qr[vi])
        qc.rz(phi, qr[vi])

    # A small initial entangling pass makes the object a channel field rather
    # than independent single-qubit traces.
    apply_channel_coupling(qc, qr, channel_v)
    qc.barrier()

    # Repeated delay/coupling/perturbation/probe rounds.
    for r in range(rounds):
        if delay_site == "pre_coupling":
            apply_structured_delay(qc, qr, all_v, delay_value, delay_unit)

        apply_channel_coupling(qc, qr, channel_v)

        if delay_site == "post_coupling":
            apply_structured_delay(qc, qr, all_v, delay_value, delay_unit)

        apply_perturbation(qc, qr, channel_v, mode, r)

        if delay_site == "post_perturb":
            apply_structured_delay(qc, qr, all_v, delay_value, delay_unit)

        # Probe parity-like response on every neighboring channel edge:
        #   probe_i receives C_i XOR C_{i+1}, then is measured and reset.
        for i, pvi in enumerate(probe_v):
            qc.cx(qr[channel_v[i]], qr[pvi])
            qc.cx(qr[channel_v[i + 1]], qr[pvi])

        qc.barrier()

        for i, pvi in enumerate(probe_v):
            qc.measure(qr[pvi], field_cregs[r][i])
            qc.reset(qr[pvi])

        qc.barrier()

    for i, vi in enumerate(channel_v):
        qc.measure(qr[vi], final_cr[i])

    creg_names = {
        "field_rounds": [f"field_r{r}" for r in range(rounds)],
        "final": "channel_final",
    }

    return qc, all_phys, creg_names


# =============================================================================
# RUNTIME RESULT EXTRACTION
# =============================================================================

def public_attrs(obj: Any) -> List[str]:
    return [a for a in dir(obj) if not a.startswith("_")]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_job_and_meta(job_id_arg: Optional[str], meta_arg: Optional[str]) -> tuple[str, Path]:
    if job_id_arg is None:
        latest = DATA_DIR / "latest_ts_job.json"
        if not latest.exists():
            raise FileNotFoundError(
                "No JOB_ID provided and ghost_oracle/T_S/data/latest_ts_job.json does not exist."
            )
        obj = load_json(latest)
        return str(obj["job_id"]), Path(obj["meta"])

    if meta_arg:
        return str(job_id_arg), Path(meta_arg)

    candidates = [
        DATA_DIR / f"ts_job_{job_id_arg}.json",
        Path(f"ts_job_{job_id_arg}.json"),
    ]
    for p in candidates:
        if p.exists():
            return str(job_id_arg), p

    raise FileNotFoundError(
        f"Could not find metadata for job {job_id_arg}. Pass --meta explicitly."
    )


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


def result_len(result: Any) -> int:
    try:
        return len(result)
    except Exception:
        return 1


def bitarray_to_columns(bitarray: Any, nbits: int, reverse_bits: bool = False) -> np.ndarray:
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
        raise KeyError(f"register '{name}' not found. Available: {public_attrs(databin)}")
    return bitarray_to_columns(getattr(databin, name), nbits, reverse_bits=reverse_bits)


def default_output(job_id: str) -> Path:
    return DATA_DIR / f"ts_data_{job_id}.npz"


def dump_ts_result(result: Any, meta: Dict[str, Any], job_id: str, reverse_bits: bool) -> Dict[str, Any]:
    modes = list(meta["modes"])
    delay_sites = list(meta["delay_sites"])
    delays = [int(x) for x in meta["delays"]]
    circuits = list(meta["circuits"])

    mode_to_i = {m: i for i, m in enumerate(modes)}
    site_to_i = {s: i for i, s in enumerate(delay_sites)}
    delay_to_i = {int(d): i for i, d in enumerate(delays)}

    rounds = int(meta["rounds"])
    channels = int(meta["channels"])
    edges = channels - 1

    field = None
    final = None

    print("\n" + "=" * 104)
    print("  T_S LIGHT DIAGNOSTICS")
    print("=" * 104)
    print(f"  {'pub':>4} | {'mode':>13} | {'site':>14} | {'delay':>8} | {'shots':>6} | {'field 1s':>9} | {'final 1s':>9}")
    print("  " + "-" * 102)

    for cm in circuits:
        pub_index = int(cm["pub_index"])
        databin = get_databin_for_pub(result, pub_index)
        creg = cm["creg_names"]
        mode = cm["mode"]
        site = cm["delay_site"]
        delay = int(cm["delay_value"])

        round_arrays = []
        for r, reg_name in enumerate(creg["field_rounds"]):
            round_arrays.append(extract_register(databin, reg_name, edges, reverse_bits))
        field_this = np.stack(round_arrays, axis=1).astype(np.uint8)  # shots, rounds, edges
        final_this = extract_register(databin, creg["final"], channels, reverse_bits).astype(np.uint8)

        if field is None:
            shots = field_this.shape[0]
            field = np.zeros((len(modes), len(delay_sites), len(delays), shots, rounds, edges), dtype=np.uint8)
            final = np.zeros((len(modes), len(delay_sites), len(delays), shots, channels), dtype=np.uint8)

        mi = mode_to_i[mode]
        si = site_to_i[site]
        di = delay_to_i[delay]
        field[mi, si, di, :, :, :] = field_this
        final[mi, si, di, :, :] = final_this

        print(
            f"  {pub_index:>4} | {mode:>13} | {site:>14} | {delay:>8} | "
            f"{field_this.shape[0]:>6} | {float(field_this.mean()):>9.5f} | {float(final_this.mean()):>9.5f}"
        )

    if field is None or final is None:
        raise RuntimeError("No circuits were dumped from the result.")

    saved: Dict[str, Any] = {
        "schema": np.array("ts_temporal_stress_metric"),
        "job_id": np.array(job_id),
        "backend": np.array(str(meta.get("backend", ""))),
        "shots": np.array(int(meta.get("shots", 0)), dtype=np.int64),
        "rounds": np.array(rounds, dtype=np.int64),
        "channels": np.array(channels, dtype=np.int64),
        "edges": np.array(edges, dtype=np.int64),
        "delay_unit": np.array(str(meta.get("delay_unit", ""))),
        "delays": np.asarray(delays, dtype=np.int64),
        "modes": np.asarray(modes),
        "delay_sites": np.asarray(delay_sites),
        "field": field,
        "final": final,
    }

    return saved


# =============================================================================
# COMMANDS
# =============================================================================

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


def add_submit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", default=DEFAULT_BACKEND)
    p.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    p.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--delays", type=int, nargs="+", default=DEFAULT_DELAYS)
    p.add_argument("--delay-unit", default=DEFAULT_DELAY_UNIT, choices=["dt", "ns", "us", "ms", "s"])
    p.add_argument("--delay-sites", nargs="+", default=DEFAULT_DELAY_SITES,
                   choices=["pre_coupling", "post_coupling", "post_perturb"])
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                   choices=["clean", "phase_shear", "local_shock", "edge_shock"])
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--optimization-level", type=int, choices=[0, 1, 2, 3], default=DEFAULT_OPT_LEVEL)
    p.add_argument("--exclude", type=int, nargs="*", default=DEFAULT_EXCLUDED_QUBITS)
    p.add_argument("--no-reserve-neighbors", action="store_true")


def add_dump_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("job_id", nargs="?", default=None, help="IBM Quantum Runtime job ID. Defaults to latest submitted T_S job.")
    p.add_argument("--meta", default=None, help="Optional metadata path.")
    p.add_argument("--save", default=None, help="Output .npz path. Defaults to ghost_oracle/T_S/data/ts_data_<JOB_ID>.npz.")
    p.add_argument("--channel", default="ibm_quantum_platform")
    p.add_argument("--instance", default=None)
    p.add_argument("--reverse-bits", action="store_true")
    p.add_argument("--list-registers", action="store_true")


def command_submit(args: argparse.Namespace) -> None:
    require_submit_deps()

    print(f"\n{'=' * 96}")
    print("  T_S SUBMIT — TEMPORAL STRESS METRIC QPU PROBE")
    print(f"{'=' * 96}")
    print(f"  Backend      : {args.backend}")
    print(f"  Channels     : {args.channels}")
    print(f"  Rounds       : {args.rounds}")
    print(f"  Shots        : {args.shots}")
    print(f"  Modes        : {args.modes}")
    print(f"  Delay sites  : {args.delay_sites}")
    print(f"  Delays       : {args.delays} {args.delay_unit}")
    print(f"  Data dir     : {DATA_DIR}")

    if args.channels < 3:
        raise ValueError("--channels must be at least 3 for T_S spatial gradients.")

    service = build_service()
    backend = service.backend(args.backend)

    print("\n[SETUP] Discovering alternating channel/probe layout.")
    roles = discover_layout(
        backend,
        channels=args.channels,
        excluded_qubits=args.exclude,
        reserve_neighbors=not args.no_reserve_neighbors,
    )

    print(f"  path     : {roles['path']}")
    print(f"  channels : {roles['channel_q']}")
    print(f"  probes   : {roles['probe_q']}")

    print("\n[BUILD] Constructing and transpiling circuits.")
    isa_circuits = []
    circ_meta = []
    pub_index = 0

    for mode in args.modes:
        for delay_site in args.delay_sites:
            for delay_value in args.delays:
                qc, all_phys, creg_names = build_ts_circuit(
                    roles=roles,
                    rounds=args.rounds,
                    delay_value=int(delay_value),
                    delay_unit=args.delay_unit,
                    delay_site=delay_site,
                    mode=mode,
                    seed=int(args.seed),
                )

                pm = generate_preset_pass_manager(
                    optimization_level=args.optimization_level,
                    backend=backend,
                    initial_layout=all_phys,
                )
                isa = pm.run(qc)
                isa_circuits.append(isa)

                circ_meta.append({
                    "pub_index": int(pub_index),
                    "mode": mode,
                    "delay_site": delay_site,
                    "delay_value": int(delay_value),
                    "channels": int(args.channels),
                    "rounds": int(args.rounds),
                    "channel_q": [int(x) for x in roles["channel_q"]],
                    "probe_q": [int(x) for x in roles["probe_q"]],
                    "path": [int(x) for x in roles["path"]],
                    "phys": [int(x) for x in all_phys],
                    "creg_names": creg_names,
                    "depth_pre_transpile": int(qc.depth()),
                    "depth_isa": int(isa.depth()) if hasattr(isa, "depth") else None,
                })
                pub_index += 1

    print(f"\n[SUBMIT] Sending {len(isa_circuits)} T_S circuits to {args.backend}.")
    sampler = Sampler(mode=backend)
    job = sampler.run(isa_circuits, shots=args.shots)
    job_id = job.job_id()

    all_used = sorted({q for cm in circ_meta for q in cm["phys"]})

    meta = {
        "schema": "ts_temporal_stress_metric_job",
        "job_id": job_id,
        "backend": args.backend,
        "shots": int(args.shots),
        "channels": int(args.channels),
        "edges": int(args.channels - 1),
        "rounds": int(args.rounds),
        "modes": list(args.modes),
        "delay_sites": list(args.delay_sites),
        "delays": [int(x) for x in args.delays],
        "delay_unit": str(args.delay_unit),
        "seed": int(args.seed),
        "layout": roles,
        "circuits": circ_meta,
        "protocol": "T_S Probe 01 — structured delay channel stress",
        "notes": (
            "The object is field[mode, delay_site, delay_value, shot, round, edge]. "
            "Final channel bits are diagnostic only.  Delay order/placement is part "
            "of the operator, not incidental runtime."
        ),
    }

    try:
        meta["calibration"] = snapshot_calibration(backend, all_used)
        print(f"  [cal] snapshotted calibration for {len(all_used)} qubits")
    except Exception as e:
        meta["calibration"] = None
        print(f"  [cal][warn] calibration snapshot failed: {e}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = DATA_DIR / f"ts_job_{job_id}.json"
    latest_path = DATA_DIR / "latest_ts_job.json"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump({"job_id": job_id, "meta": str(meta_path)}, f, indent=2)

    dump_cmd = f"python ghost_oracle/T_S/t_s_qpu_generate.py dump {job_id}"

    print(f"\n{'=' * 96}")
    print("  T_S JOB SUBMITTED")
    print(f"{'=' * 96}")
    print(f"  Job ID   : {job_id}")
    print(f"  Metadata : {meta_path}")
    print(f"  Latest   : {latest_path}")
    print("\n  Dump command:")
    print(f"    {dump_cmd}")
    print("\n  Then analyze:")
    print("    python ghost_oracle/T_S/t_s_analyze.py")
    print(f"{'=' * 96}\n")


def command_dump(args: argparse.Namespace) -> None:
    require_runtime()

    job_id, meta_path = resolve_job_and_meta(args.job_id, args.meta)
    meta = load_json(meta_path)

    print(f"\n{'=' * 96}")
    print("  T_S DUMP — QISKIT RUNTIME DATA")
    print(f"{'=' * 96}")
    print(f"  Job ID   : {job_id}")
    print(f"  Metadata : {meta_path}")

    kwargs = {"channel": args.channel}
    if args.instance:
        kwargs["instance"] = args.instance

    print("\n[FETCH] Connecting to Qiskit Runtime.")
    service = QiskitRuntimeService(**kwargs)
    job = service.job(job_id)
    result = job.result()

    if args.list_registers:
        for i in range(result_len(result)):
            db = get_databin_for_pub(result, i)
            print(f"  PUB {i} registers: {public_attrs(db)}")

    saved = dump_ts_result(result, meta, job_id, args.reverse_bits)

    out_path = Path(args.save) if args.save else default_output(job_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **saved)

    latest_data = DATA_DIR / "latest_ts_data.json"
    with open(latest_data, "w", encoding="utf-8") as f:
        json.dump({"job_id": job_id, "npz": str(out_path), "meta": str(meta_path)}, f, indent=2)

    print(f"\n[SAVED] {out_path}")
    print(f"[LATEST] {latest_data}")
    print("\n  Next:")
    print("    python ghost_oracle/T_S/t_s_analyze.py")
    print(f"{'=' * 96}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T_S QPU tool — submit structured-delay QPU jobs and dump completed Runtime results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Submit T_S Probe 01 structured-delay QPU experiment.")
    add_submit_args(p_submit)

    p_dump = sub.add_parser("dump", help="Dump a completed T_S QPU job to .npz.")
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
