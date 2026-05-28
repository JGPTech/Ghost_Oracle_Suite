#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
S_M STEP 1 — SUBMIT QPU SUPERPOSITION JOB
==============================================================================
One-button QPU submitter for the S_M pipeline.

Default behavior
----------------
Runs the current recommended S_M experiment:

    backend      : ibm_marrakesh
    flag level   : f=0
    distances    : 3 5 7 9
    rounds       : 10
    shots        : 4096
    init state   : plus
    basis        : z

That prepares a logical-cat state inside the Z-repetition-code space:

    |+_L> = (|000...0> + |111...1>) / sqrt(2)

then runs repeated syndrome extraction and final Z-basis data readout.

Why this is the default
-----------------------
The S_M result is not based on final majority-vote logical error. For plus/minus
cat states, Z readout gives a broad physical record. The useful object is final
edge parity plus syndrome spacetime:

    E_i = D_i XOR D_{i+1}
    S[t, i]

The downstream analysis script tests whether this record forms a bounded
syndrome-spacetime field and stress tensor.

Outputs
-------
Writes metadata to:

    data/sm_job_<JOB_ID>.json
    data/latest_sm_job.json

The dumper can then be run without manually passing --meta:

    python ghost_oracle/S_M/sm_dump.py <JOB_ID>

Usage
-----
One-button default:

    python ghost_oracle/S_M/sm_submit.py

Override options:

    python ghost_oracle/S_M/sm_submit.py --backend ibm_fez --shots 8192
    python ghost_oracle/S_M/sm_submit.py --init-state minus
    python ghost_oracle/S_M/sm_submit.py --flag 1 --distances 3 5
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

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
REPO_ROOT = find_repo_root(HERE)
DATA_DIR = REPO_ROOT / "data"


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


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S_M step 1 — submit the default logical-cat QPU job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    print(f"\n{'=' * 90}")
    print("  S_M STEP 1 — SUBMIT QPU SUPERPOSITION JOB")
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

    print(f"\n{'=' * 90}")
    print("  JOB SUBMITTED")
    print(f"{'=' * 90}")
    print(f"  Job ID   : {job_id}")
    print(f"  Metadata : {meta_path}")
    print(f"  Latest   : {latest_path}")
    print("\n  Next:")
    print(f"    python ghost_oracle/S_M/sm_dump.py {job_id}")
    print(f"    python ghost_oracle/S_M/sm_analyze.py")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
