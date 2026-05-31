#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
S_M QPU TOOL — SUBMIT + DUMP
==============================================================================
Unified QPU workflow for the S_M pipeline.

This replaces the separate sm_submit.py and sm_dump.py scripts with one
subcommand-based CLI:

    python ghost_oracle/S_M/s_m_qpu_generate.py submit
    python ghost_oracle/S_M/s_m_qpu_generate.py dump <JOB_ID>

The submit command writes:

    data/sm_job_<JOB_ID>.json
    data/latest_sm_job.json

and prints the exact dump command for the submitted job.

The dump command fetches a completed SamplerV2 job, extracts shot-order
classical registers, and writes:

    data/sm_data_<init_state>_<JOB_ID>.npz
    data/latest_sm_data.json

Default submit run:

    backend      : ibm_marrakesh
    flag level   : f=0
    distances    : 3 5 7 9
    rounds       : 10
    shots        : 4096
    init state   : plus
    basis        : z

The useful S_M object is final edge parity plus syndrome spacetime:

    E[i] = D[i] XOR D[i+1]
    S[t, i]

For plus/minus logical-cat states, majority-vote logical error is diagnostic
only. Downstream S_M analysis should use final edge parity and syndrome
spacetime records.
==============================================================================
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_ROUNDS = 10
DEFAULT_DISTANCES = [3, 5, 7, 9]
DEFAULT_FLAG_LEVEL = 0
DEFAULT_INIT_STATE = "plus"
DEFAULT_BASIS = "z"
DEFAULT_OPT_LEVEL = 1
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


def snapshot_calibration(backend, phys_qubits: Sequence[int]) -> Dict:
    """Best-effort calibration snapshot for later reproducibility."""
    target = backend.target
    cal = {
        "single_qubit": {},
        "readout": {},
        "idling": {},
        "two_qubit": {},
        "meta": {"captured_from": getattr(backend, "name", "unknown")},
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
            t1 = getattr(qp, "t1", None)
            t2 = getattr(qp, "t2", None)
            cal["idling"][str(q)] = {
                "t1": float(t1) if t1 is not None else None,
                "t2": float(t2) if t2 is not None else None,
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


def chain_length_for(distance: int, flag_level: int) -> int:
    bond_internal = {0: 1, 1: 3, 2: 5}[flag_level]
    return distance + (distance - 1) * bond_internal


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


def assign_roles(path: List[int], distance: int, flag_level: int) -> Dict:
    data_q: List[int] = []
    synd_q: List[int] = []
    flag_q: List[List[int]] = []

    idx = 0
    data_q.append(path[idx])
    idx += 1

    for _ in range(distance - 1):
        if flag_level == 0:
            synd_q.append(path[idx])
            idx += 1
            flag_q.append([])
        elif flag_level == 1:
            fL = path[idx]
            s = path[idx + 1]
            fR = path[idx + 2]
            idx += 3
            synd_q.append(s)
            flag_q.append([fL, fR])
        elif flag_level == 2:
            outerL = path[idx]
            innerL = path[idx + 1]
            s = path[idx + 2]
            innerR = path[idx + 3]
            outerR = path[idx + 4]
            idx += 5
            synd_q.append(s)
            flag_q.append([outerL, innerL, outerR, innerR])
        else:
            raise ValueError(f"unsupported flag level: {flag_level}")

        data_q.append(path[idx])
        idx += 1

    return {"data_q": data_q, "synd_q": synd_q, "flag_q": flag_q, "path": path}


def discover_layouts(
    backend,
    distances: Sequence[int],
    flag_level: int,
    excluded_qubits: Sequence[int],
    reserve_neighbors: bool = True,
) -> List[Dict]:
    adj = build_adjacency(backend)
    n = backend.num_qubits
    used = set(excluded_qubits)
    layouts: List[Dict] = []

    for d in distances:
        L = chain_length_for(int(d), flag_level)
        path = find_path(adj, L, used, n)
        if path is None:
            raise RuntimeError(
                f"no length-{L} path found for d={d}, flag={flag_level}. "
                "Try fewer distances, lower flag level, different backend, or --no-reserve-neighbors."
            )

        roles = assign_roles(path, int(d), flag_level)
        layouts.append(roles)

        used.update(path)
        if reserve_neighbors:
            for q in path:
                used.update(adj[q])

    return layouts


# =============================================================================
# CIRCUIT
# =============================================================================

def build_circuit(
    roles: Dict,
    distance: int,
    rounds: int,
    flag_level: int,
    logical_init: int,
    basis: str,
    init_state: str,
) -> Tuple[QuantumCircuit, List[int], Dict]:
    data_q = list(map(int, roles["data_q"]))
    synd_q = list(map(int, roles["synd_q"]))
    flag_q = [[int(x) for x in fs] for fs in roles["flag_q"]]

    all_phys = sorted(set(data_q) | set(synd_q) | {f for fs in flag_q for f in fs})
    v = {p: i for i, p in enumerate(all_phys)}

    qr = QuantumRegister(len(all_phys), name="q")
    qc = QuantumCircuit(qr)

    synd_cregs = []
    for r in range(rounds):
        scr = ClassicalRegister(distance - 1, name=f"synd_r{r}")
        qc.add_register(scr)
        synd_cregs.append(scr)

    n_flag_per_round = sum(len(fs) for fs in flag_q)
    flag_cregs = []
    if n_flag_per_round > 0:
        for r in range(rounds):
            fcr = ClassicalRegister(n_flag_per_round, name=f"flag_r{r}")
            qc.add_register(fcr)
            flag_cregs.append(fcr)

    dcr = ClassicalRegister(distance, name="data_final")
    qc.add_register(dcr)

    def D(i: int):
        return qr[v[data_q[i]]]

    def S(i: int):
        return qr[v[synd_q[i]]]

    def Q(p: int):
        return qr[v[p]]

    z_run = basis == "z"

    # Data preparation.
    if init_state in ("zero", "one"):
        init_bit = 1 if init_state == "one" else int(logical_init)
        if init_bit == 1:
            for i in range(distance):
                qc.x(D(i))
        if not z_run:
            for i in range(distance):
                qc.h(D(i))

    elif init_state in ("plus", "minus"):
        if basis != "z":
            print("[warn] plus/minus S_M run is intended for --basis z.")
        qc.h(D(0))
        if init_state == "minus":
            qc.z(D(0))
        for i in range(1, distance):
            qc.cx(D(0), D(i))
    else:
        raise ValueError(f"unknown init_state: {init_state}")

    qc.barrier()

    # Syndrome rounds.
    for r in range(rounds):
        for i in range(distance - 1):
            fs = flag_q[i]

            if flag_level == 0:
                if z_run:
                    qc.cx(D(i), S(i))
                    qc.cx(D(i + 1), S(i))
                else:
                    qc.h(S(i))
                    qc.cx(S(i), D(i))
                    qc.cx(S(i), D(i + 1))
                    qc.h(S(i))

            elif flag_level == 1:
                fL, fR = fs[0], fs[1]
                qc.h(S(i))
                qc.cx(S(i), Q(fL))
                qc.cx(S(i), Q(fR))
                if z_run:
                    qc.cz(Q(fL), D(i))
                    qc.cz(Q(fR), D(i + 1))
                else:
                    qc.cx(Q(fL), D(i))
                    qc.cx(Q(fR), D(i + 1))
                qc.cx(S(i), Q(fR))
                qc.cx(S(i), Q(fL))
                qc.h(S(i))

            elif flag_level == 2:
                fL1, fL2, fR1, fR2 = fs[0], fs[1], fs[2], fs[3]
                qc.h(S(i))
                qc.cx(S(i), Q(fL2))
                qc.cx(Q(fL2), Q(fL1))
                qc.cx(S(i), Q(fR2))
                qc.cx(Q(fR2), Q(fR1))
                if z_run:
                    qc.cz(Q(fL1), D(i))
                    qc.cz(Q(fR1), D(i + 1))
                else:
                    qc.cx(Q(fL1), D(i))
                    qc.cx(Q(fR1), D(i + 1))
                qc.cx(Q(fR2), Q(fR1))
                qc.cx(S(i), Q(fR2))
                qc.cx(Q(fL2), Q(fL1))
                qc.cx(S(i), Q(fL2))
                qc.h(S(i))

        qc.barrier()

        for i in range(distance - 1):
            qc.measure(S(i), synd_cregs[r][i])
            qc.reset(S(i))

        if n_flag_per_round:
            fpos = 0
            for i in range(distance - 1):
                for fq in flag_q[i]:
                    qc.measure(Q(fq), flag_cregs[r][fpos])
                    qc.reset(Q(fq))
                    fpos += 1

        qc.barrier()

    if not z_run and init_state in ("zero", "one"):
        for i in range(distance):
            qc.h(D(i))

    for i in range(distance):
        qc.measure(D(i), dcr[i])

    creg_names = {
        "rounds_synd": [f"synd_r{r}" for r in range(rounds)],
        "rounds_flag": [f"flag_r{r}" for r in range(rounds)] if n_flag_per_round else [],
        "final": "data_final",
        "flag_layout": [len(fs) for fs in flag_q],
    }

    return qc, all_phys, creg_names




def public_attrs(obj: Any) -> List[str]:
    return [a for a in dir(obj) if not a.startswith("_")]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_job_and_meta(job_id_arg: Optional[str], meta_arg: Optional[str]) -> tuple[str, Path]:
    if job_id_arg is None:
        latest = DATA_DIR / "latest_sm_job.json"
        if not latest.exists():
            raise FileNotFoundError(
                "No JOB_ID provided and data/latest_sm_job.json does not exist. "
                "Run sm_submit.py first or pass JOB_ID explicitly."
            )
        obj = load_json(latest)
        job_id = str(obj["job_id"])
        meta_path = Path(obj["meta"])
        return job_id, meta_path

    job_id = job_id_arg
    if meta_arg:
        return job_id, Path(meta_arg)

    candidates = [
        DATA_DIR / f"sm_job_{job_id}.json",
        DATA_DIR / f"repcode_flag_superposition_job_{job_id}.json",
        DATA_DIR / f"repcode_flag_job_{job_id}.json",
        DATA_DIR / f"repcode_distance3_job_{job_id}.json",
        Path(f"sm_job_{job_id}.json"),
        Path(f"repcode_flag_superposition_job_{job_id}.json"),
        Path(f"repcode_flag_job_{job_id}.json"),
        Path(f"repcode_distance3_job_{job_id}.json"),
    ]
    for p in candidates:
        if p.exists():
            return job_id, p

    raise FileNotFoundError(
        f"Could not find metadata for job {job_id}. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\nPass --meta explicitly if the file is elsewhere."
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


def majority_vote(data: np.ndarray) -> np.ndarray:
    return (data.sum(axis=1) > (data.shape[1] / 2)).astype(np.uint8)


def terminal_edge_parity(data: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(data[:, :-1], data[:, 1:]).astype(np.uint8)


def last_syndrome_mismatch(data: np.ndarray, synd: np.ndarray) -> float:
    if data.shape[1] - 1 != synd.shape[2]:
        return float("nan")
    return float(np.bitwise_xor(terminal_edge_parity(data), synd[:, -1, :]).mean())


def syndrome_instability(synd: np.ndarray) -> float:
    if synd.shape[1] < 2:
        return 0.0
    return float(np.bitwise_xor(synd[:, 1:, :], synd[:, :-1, :]).mean())


def print_header() -> None:
    print("\n" + "=" * 96)
    print("  LIGHT DIAGNOSTICS")
    print("=" * 96)
    print(
        f"  {'item':>10} | {'shots':>6} | {'bits':>5} | {'maj LER':>8} | "
        f"{'data 1s':>8} | {'synd 1s':>8} | {'flag 1s':>8} | "
        f"{'lastmis':>8} | {'instab':>8}"
    )
    print("  " + "-" * 94)


def print_row(label: str, data: np.ndarray, synd: np.ndarray, flags: Optional[np.ndarray], logical: int) -> None:
    flag_rate = "n/a" if flags is None else f"{float(flags.mean()):.4f}"
    ler = float(np.mean(majority_vote(data) != np.uint8(logical)))
    print(
        f"  {label:>10} | {data.shape[0]:>6} | {data.shape[1]:>5} | "
        f"{ler:>8.2%} | {float(data.mean()):>8.4f} | {float(synd.mean()):>8.4f} | "
        f"{flag_rate:>8} | {last_syndrome_mismatch(data, synd):>8.4f} | "
        f"{syndrome_instability(synd):>8.4f}"
    )


def dump_flag_like(result: Any, meta: Dict[str, Any], job_id: str, reverse_bits: bool) -> Dict[str, Any]:
    rounds = int(meta["rounds"])
    logical = int(meta.get("logical_init", 0))
    init_state = str(meta.get("init_state", ""))
    circuits = list(meta["circuits"])
    distances = [int(c["distance"]) for c in circuits]

    saved: Dict[str, Any] = {
        "schema": np.array("sm_data"),
        "job_id": np.array(job_id),
        "backend": np.array(str(meta.get("backend", ""))),
        "shots": np.array(int(meta.get("shots", 0)), dtype=np.int64),
        "rounds": np.array(rounds, dtype=np.int64),
        "flag_level": np.array(int(meta.get("flag_level", -1)), dtype=np.int64),
        "logical_init": np.array(logical, dtype=np.int64),
        "basis": np.array(str(meta.get("basis", ""))),
        "init_state": np.array(init_state),
        "distances": np.asarray(distances, dtype=np.int64),
    }

    print_header()

    for pub_index, cm in enumerate(circuits):
        d = int(cm["distance"])
        databin = get_databin_for_pub(result, pub_index)
        creg = cm.get("creg_names", {})

        data = extract_register(databin, creg.get("final", "data_final"), d, reverse_bits)
        synd_names = creg.get("rounds_synd", [f"synd_r{r}" for r in range(rounds)])

        synd = np.zeros((data.shape[0], rounds, d - 1), dtype=np.uint8)
        for r, reg_name in enumerate(synd_names):
            synd[:, r, :] = extract_register(databin, reg_name, d - 1, reverse_bits)

        flags = None
        flag_names = creg.get("rounds_flag", [])
        flag_layout = creg.get("flag_layout", [])
        n_flags = int(sum(flag_layout)) if flag_layout else 0

        if flag_names:
            flags = np.zeros((data.shape[0], rounds, n_flags), dtype=np.uint8)
            for r, reg_name in enumerate(flag_names):
                flags[:, r, :] = extract_register(databin, reg_name, n_flags, reverse_bits)

        saved[f"data_d{d}"] = data
        saved[f"synd_d{d}"] = synd
        if flags is not None:
            saved[f"flag_d{d}"] = flags

        print_row(f"d{d}", data, synd, flags, logical)

    if init_state in ("plus", "minus"):
        print("\n  [note] plus/minus cat state: majority LER is diagnostic only.")

    return saved


def dump_plain_distance3(result: Any, meta: Dict[str, Any], job_id: str, reverse_bits: bool) -> Dict[str, Any]:
    rounds = int(meta["rounds"])
    logical = int(meta.get("logical_init", 0))
    num_blocks = int(meta["num_blocks"])
    databin = get_databin_for_pub(result, 0)

    saved: Dict[str, Any] = {
        "schema": np.array("repcode_distance3"),
        "job_id": np.array(job_id),
        "backend": np.array(str(meta.get("backend", ""))),
        "shots": np.array(int(meta.get("shots", 0)), dtype=np.int64),
        "rounds": np.array(rounds, dtype=np.int64),
        "logical_init": np.array(logical, dtype=np.int64),
        "inject_qubit": np.array(int(meta.get("inject_qubit", -1)), dtype=np.int64),
        "num_blocks": np.array(num_blocks, dtype=np.int64),
        "block_layout": np.asarray(meta.get("block_layout", []), dtype=np.int64),
    }

    print_header()

    for blk in range(num_blocks):
        data = extract_register(databin, f"data_b{blk}", 3, reverse_bits)
        synd = np.zeros((data.shape[0], rounds, 2), dtype=np.uint8)
        for r in range(rounds):
            synd[:, r, :] = extract_register(databin, f"syndrome_b{blk}_r{r}", 2, reverse_bits)

        saved[f"data_b{blk}"] = data
        saved[f"syndrome_b{blk}"] = synd
        print_row(f"b{blk}", data, synd, None, logical)

    return saved


def default_output(job_id: str, meta: Dict[str, Any]) -> Path:
    init_state = str(meta.get("init_state", "")).strip()
    if "circuits" in meta and "distances" in meta:
        tag = init_state if init_state else "job"
        return DATA_DIR / f"sm_data_{tag}_{job_id}.npz"
    return DATA_DIR / f"repcode_distance3_job_{job_id}.npz"




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
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--distances", type=int, nargs="+", default=DEFAULT_DISTANCES)
    p.add_argument("--flag", type=int, choices=[0, 1, 2], default=DEFAULT_FLAG_LEVEL)
    p.add_argument("--basis", choices=["z", "x"], default=DEFAULT_BASIS)
    p.add_argument("--init-state", choices=["zero", "one", "plus", "minus"], default=DEFAULT_INIT_STATE)
    p.add_argument("--logical", type=int, choices=[0, 1], default=0)
    p.add_argument("--optimization-level", type=int, choices=[0, 1, 2, 3], default=DEFAULT_OPT_LEVEL)
    p.add_argument("--exclude", type=int, nargs="*", default=DEFAULT_EXCLUDED_QUBITS)
    p.add_argument("--no-reserve-neighbors", action="store_true")


def add_dump_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("job_id", nargs="?", default=None, help="IBM Quantum Runtime job ID. Defaults to latest submitted S_M job.")
    p.add_argument("--meta", default=None, help="Optional metadata path. Usually not needed for jobs from this submit command.")
    p.add_argument("--save", default=None, help="Output .npz path. Defaults to data/sm_data_<state>_<JOB_ID>.npz.")
    p.add_argument("--channel", default="ibm_quantum_platform")
    p.add_argument("--instance", default=None)
    p.add_argument("--reverse-bits", action="store_true")
    p.add_argument("--list-registers", action="store_true")


def command_submit(args: argparse.Namespace) -> None:
    require_submit_deps()

    print(f"\n{'=' * 90}")
    print("  S_M SUBMIT — QPU SUPERPOSITION JOB")
    print(f"{'=' * 90}")
    print(f"  Backend      : {args.backend}")
    print(f"  Flag level   : f={args.flag}")
    print(f"  Distances    : {args.distances}")
    print(f"  Rounds       : {args.rounds}")
    print(f"  Shots        : {args.shots}")
    print(f"  Basis        : {args.basis.upper()}")
    print(f"  Init state   : {args.init_state}")
    print(f"  Data dir     : {DATA_DIR}")

    service = build_service()
    backend = service.backend(args.backend)

    print("\n[SETUP] Discovering layouts...")
    layouts = discover_layouts(
        backend,
        distances=args.distances,
        flag_level=args.flag,
        excluded_qubits=args.exclude,
        reserve_neighbors=not args.no_reserve_neighbors,
    )

    for d, roles in zip(args.distances, layouts):
        print(f"  d={d}: path={roles['path']}")
        print(f"       data={roles['data_q']}")
        print(f"       synd={roles['synd_q']}")
        if args.flag:
            print(f"       flag={roles['flag_q']}")

    print("\n[BUILD] Constructing and transpiling circuits...")
    isa_circuits = []
    circ_meta = []

    for d, roles in zip(args.distances, layouts):
        qc, all_phys, creg_names = build_circuit(
            roles=roles,
            distance=int(d),
            rounds=args.rounds,
            flag_level=args.flag,
            logical_init=args.logical,
            basis=args.basis,
            init_state=args.init_state,
        )

        pm = generate_preset_pass_manager(
            optimization_level=args.optimization_level,
            backend=backend,
            initial_layout=all_phys,
        )
        isa = pm.run(qc)
        isa_circuits.append(isa)

        circ_meta.append({
            "distance": int(d),
            "data_q": [int(x) for x in roles["data_q"]],
            "synd_q": [int(x) for x in roles["synd_q"]],
            "flag_q": [[int(y) for y in fs] for fs in roles["flag_q"]],
            "path": [int(x) for x in roles["path"]],
            "phys": [int(x) for x in all_phys],
            "creg_names": creg_names,
            "depth_pre_transpile": int(qc.depth()),
            "depth_isa": int(isa.depth()) if hasattr(isa, "depth") else None,
        })

    print(f"\n[SUBMIT] Sending {len(isa_circuits)} circuits to {args.backend}...")
    sampler = Sampler(mode=backend)
    job = sampler.run(isa_circuits, shots=args.shots)
    job_id = job.job_id()

    all_used = sorted({q for cm in circ_meta for q in cm["phys"]})

    meta = {
        "schema": "sm_superposition",
        "job_id": job_id,
        "backend": args.backend,
        "shots": int(args.shots),
        "rounds": int(args.rounds),
        "flag_level": int(args.flag),
        "logical_init": int(args.logical),
        "basis": args.basis,
        "init_state": args.init_state,
        "state_family": "logical_cat" if args.init_state in ("plus", "minus") else "logical_product",
        "distances": [int(d) for d in args.distances],
        "circuits": circ_meta,
        "protocol": "S_M logical-cat syndrome-spacetime probe",
        "notes": (
            "For plus/minus cat states, final majority-vs-logical_init is diagnostic only. "
            "Use final edge parity and syndrome spacetime analyses."
        ),
    }

    try:
        meta["calibration"] = snapshot_calibration(backend, all_used)
        print(f"  [cal] snapshotted calibration for {len(all_used)} qubits")
    except Exception as e:
        meta["calibration"] = None
        print(f"  [cal][warn] calibration snapshot failed: {e}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = DATA_DIR / f"sm_job_{job_id}.json"
    latest_path = DATA_DIR / "latest_sm_job.json"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump({"job_id": job_id, "meta": str(meta_path)}, f, indent=2)

    dump_cmd = f"python ghost_oracle/S_M/s_m_qpu_generate.py dump {job_id}"

    print(f"\n{'=' * 90}")
    print("  JOB SUBMITTED")
    print(f"{'=' * 90}")
    print(f"  Job ID   : {job_id}")
    print(f"  Metadata : {meta_path}")
    print(f"  Latest   : {latest_path}")
    print("\n  Dump command:")
    print(f"    {dump_cmd}")
    print("\n  Then analyze:")
    print("    python ghost_oracle/S_M/s_m_benchmark.py")
    print(f"{'=' * 90}\n")


def command_dump(args: argparse.Namespace) -> None:
    require_runtime()

    job_id, meta_path = resolve_job_and_meta(args.job_id, args.meta)
    meta = load_json(meta_path)

    print(f"\n{'=' * 96}")
    print("  S_M DUMP — QISKIT RUNTIME DATA")
    print(f"{'=' * 96}")
    print(f"  Job ID   : {job_id}")
    print(f"  Metadata : {meta_path}")

    kwargs = {"channel": args.channel}
    if args.instance:
        kwargs["instance"] = args.instance

    print("\n[FETCH] Connecting to Qiskit Runtime...")
    service = QiskitRuntimeService(**kwargs)
    job = service.job(job_id)
    result = job.result()

    if args.list_registers:
        for i in range(result_len(result)):
            db = get_databin_for_pub(result, i)
            print(f"  PUB {i} registers: {public_attrs(db)}")

    if "circuits" in meta and "distances" in meta:
        saved = dump_flag_like(result, meta, job_id, args.reverse_bits)
    elif "num_blocks" in meta and "block_layout" in meta:
        saved = dump_plain_distance3(result, meta, job_id, args.reverse_bits)
    else:
        raise KeyError("metadata schema not recognized")

    out_path = Path(args.save) if args.save else default_output(job_id, meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **saved)

    latest_data = DATA_DIR / "latest_sm_data.json"
    with open(latest_data, "w", encoding="utf-8") as f:
        json.dump({"job_id": job_id, "npz": str(out_path), "meta": str(meta_path)}, f, indent=2)

    print(f"\n[SAVED] {out_path}")
    print(f"[LATEST] {latest_data}")
    print("\n  Next:")
    print("    python ghost_oracle/S_M/s_m_benchmark.py")
    print(f"{'=' * 96}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S_M QPU tool — submit QPU jobs and dump completed Runtime results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Submit the default/current S_M QPU experiment.")
    add_submit_args(p_submit)

    p_dump = sub.add_parser("dump", help="Dump a completed S_M QPU job to .npz.")
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
