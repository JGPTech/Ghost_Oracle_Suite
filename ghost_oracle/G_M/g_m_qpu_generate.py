#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — G_M QPU BASE (SUBMIT + DUMP)
==============================================================================
Unified G_M base script. Two modes, one file, because the QPU is asynchronous:
you submit a job, it sits in the queue for an unpredictable stretch, and you
pull the shots later. Submitting and dumping cannot be a single linear run.

    submit : build the tiled Hadamard-test circuit (paper Section 2) and send
             it to an IBM Quantum backend. Prints the job ID. Does NOT wait.

    dump   : fetch a completed job by ID and FREEZE it into data/ as the
             canonical .npz base consumed by the projection script and probes.
             This is the freeze step — the only place a base is written.

Per-tile circuit (qubits: a1, v1, a2, ctrl, b1, v2, b2):
    1. Ry(theta_a) on v1; Ry(theta_b) on v2
    2. CNOT(v1 -> a1); CNOT(v1 -> a2); CNOT(v2 -> b1); CNOT(v2 -> b2)
    3. H on ctrl
    4. CSWAP(ctrl; v1, v2)
    5. H on ctrl
    6. Measure {ctrl, a1, a2, b1, b2}

Each tile covers one (r, c) entry of the parameterized matrix family. The
output registers are named matmul_tile{t} (control bit) and ghost_tile{t}
(ancilla bits) to remain bit-compatible with the sample dumps shipped in data/.

Layout note: cluster selection, ANGLE_SCALE, EXCLUDED_QUBITS, the readout-error
sort fallback, and the bit ordering are preserved exactly as calibrated. They
are intentional. See find_qubit_clusters() for the readout-error detail.

Usage:
    export IBM_QUANTUM_TOKEN=<your_token>
    python qpu.py submit                          # use defaults
    python qpu.py submit --backend ibm_marrakesh  # override backend
    python qpu.py submit --shots 8192             # override shot count
    # ... wait for the job to finish ...
    python qpu.py dump <JOB_ID>
    python qpu.py dump <JOB_ID> --num-tiles 12
==============================================================================
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit.circuit.library import XGate, YGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
    BasisTranslator,
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# CONFIGURATION (defaults; overridable via CLI)
# =============================================================================
DEFAULT_BACKEND_NAME = "ibm_marrakesh"
DEFAULT_SHOTS        = 4096
STAGGER_DT           = 72              # per-tile delay step in backend dt units
ANGLE_SCALE          = 1.05            # suite-wide calibration; see docs/math.md

# Qubits to exclude from cluster selection (high-error or known-bad).
# Specific to ibm_marrakesh; override if using a different backend.
EXCLUDED_QUBITS = [15, 19, 29, 38, 49, 51, 67, 70, 147, 114, 116, 118, 120, 122]

# Parameterized matrix family (default 4x4 sweep).
DEFAULT_MATRIX_A = np.array([0.25, 0.50, 0.75, 1.00])
DEFAULT_MATRIX_B = np.array([1.00, 0.80, 0.40, 0.10])


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent          # repo/
DATA_DIR = HERE / "data"


def data_to_angles(data, scale=ANGLE_SCALE):
    """Scale a real-valued vector into rotation angles in [0, pi/2 * scale]."""
    max_val = np.max(np.abs(data))
    return (data / max_val) * (np.pi / 2) * scale


# =============================================================================
# TOKEN HANDLING
# =============================================================================
def get_token():
    token = os.environ.get("IBM_QUANTUM_TOKEN", "").strip()
    if not token:
        sys.stderr.write(
            "\n[ERROR] IBM_QUANTUM_TOKEN environment variable not set.\n"
            "        export IBM_QUANTUM_TOKEN=<your_token>\n"
            "        (or save a default account via QiskitRuntimeService.save_account)\n\n"
        )
        sys.exit(1)
    return token


# =============================================================================
# QUBIT CLUSTER SELECTION
# Greedy low-error 7-qubit connected tiles, with neighbor exclusion to keep
# tiles spatially separated.
# =============================================================================
def find_qubit_clusters(backend, cluster_size=7, max_tiles=16):
    """
    Greedy 7-qubit connected-cluster selection.

    NOTE on the readout-error sort: the lookup below uses Qiskit's Target with
    `i in target['measure']`, which compares an int against a tuple-keyed dict
    and is always False on current Qiskit. Every qubit therefore falls back to
    the 0.5 sentinel and the "sort by readout error" reduces to candidates in
    raw ascending qubit-index order, with greedy BFS from qubit 0 outward.

    This is the layout under which the entire Ghost Oracle probe series was
    calibrated (ALPHA_NORM = 0.9127, ANGLE_SCALE = 1.05, etc.). Changing it
    invalidates those constants and breaks reproducibility against historical
    QPU dumps. The behavior is preserved as-is intentionally.
    """
    target = backend.target
    adjacency = {i: set() for i in range(backend.num_qubits)}
    for u, v in backend.coupling_map.get_edges():
        adjacency[u].add(v)
        adjacency[v].add(u)

    readout_error = {
        i: target['measure'][i,].error if i in target['measure'] else 0.5
        for i in range(backend.num_qubits)
    }

    candidates = sorted(
        [q for q in range(backend.num_qubits) if q not in EXCLUDED_QUBITS],
        key=lambda q: readout_error.get(q, 1.0),
    )

    clusters = []
    used = set(EXCLUDED_QUBITS)

    for seed_qubit in candidates:
        if seed_qubit in used:
            continue

        cluster = [seed_qubit]
        frontier = [seed_qubit]
        while frontier and len(cluster) < cluster_size:
            current = frontier.pop(0)
            neighbors = sorted(
                [n for n in adjacency[current] if n not in cluster and n not in used],
                key=lambda n: readout_error.get(n, 1.0),
            )
            for n in neighbors:
                if len(cluster) >= cluster_size:
                    break
                cluster.append(n)
                frontier.append(n)

        if len(cluster) == cluster_size:
            # Reserve cluster and one-hop neighborhood so tiles don't share boundaries.
            reserved = set(cluster)
            for q in cluster:
                reserved.update(adjacency[q])
            used.update(reserved)
            clusters.append(cluster)

        if len(clusters) >= max_tiles:
            break

    return clusters


# =============================================================================
# CIRCUIT CONSTRUCTION
# =============================================================================
def build_circuit(num_tiles, pairs, angles_a, angles_b):
    """
    Build the tiled circuit. Each tile uses 7 qubits with layout
    [a1, v1, a2, ctrl, b1, v2, b2] and emits two classical registers:
        matmul_tile{t} : 1 bit  (Hadamard-test control measurement)
        ghost_tile{t}  : 4 bits (ancilla measurements, MSB-first in bitstring)
    """
    qr = QuantumRegister(num_tiles * 7, name='q')
    qc = QuantumCircuit(qr)

    # Hold register handles per-tile so later measurement loop doesn't depend
    # on positional order of qc.cregs.
    tile_cregs = []

    for t in range(num_tiles):
        r, c = pairs[t]
        matmul_cr = ClassicalRegister(1, name=f'matmul_tile{t}')
        ghost_cr  = ClassicalRegister(4, name=f'ghost_tile{t}')
        qc.add_register(matmul_cr)
        qc.add_register(ghost_cr)
        tile_cregs.append((matmul_cr, ghost_cr))

        base = t * 7
        a1, v1, a2, ctrl, b1, v2, b2 = [qr[base + i] for i in range(7)]

        # Per-tile delay stagger to spread microwave pulse density across the chip.
        delay_dt = t * STAGGER_DT
        if delay_dt > 0:
            for q in [a1, v1, a2, ctrl, b1, v2, b2]:
                qc.delay(delay_dt, q, unit='dt')

        # State preparation.
        qc.barrier([v1, v2])
        qc.ry(angles_a[r], v1)
        qc.ry(angles_b[c], v2)
        qc.barrier([v1, v2])

        # Ghost CNOTs — entangle v with two ancillas each (the GHZ structure
        # that gives the T3 mixed-state target).
        qc.cx(v1, a1)
        qc.cx(v1, a2)
        qc.cx(v2, b1)
        qc.cx(v2, b2)

    qc.barrier()

    # Hadamard test on each tile's (v1, v2) via the ctrl ancilla.
    for t in range(num_tiles):
        base = t * 7
        v1, ctrl, v2 = qr[base + 1], qr[base + 3], qr[base + 5]
        if t % 2 == 0:
            qc.delay(36, ctrl, unit='dt')
        qc.h(ctrl)
        qc.cswap(ctrl, v1, v2)
        qc.h(ctrl)

    qc.barrier()

    # Measurements (using saved register handles instead of positional cregs index).
    for t, (matmul_cr, ghost_cr) in enumerate(tile_cregs):
        base = t * 7
        qc.measure(qr[base + 3], matmul_cr[0])
        qc.measure(
            [qr[base + 0], qr[base + 2], qr[base + 4], qr[base + 6]],
            ghost_cr,
        )
    return qc


# =============================================================================
# DUMP HELPERS
# =============================================================================

def public_attrs(obj):
    return [a for a in dir(obj) if not a.startswith("_")]


def infer_tile_indices(databin):
    """
    Infer tile indices from DataBin attributes.

    Expected register names:
        matmul_tile{t}
        ghost_tile{t}
    """
    attrs = public_attrs(databin)

    matmul = set()
    ghost = set()

    for name in attrs:
        m = re.fullmatch(r"matmul_tile(\d+)", name)
        if m:
            matmul.add(int(m.group(1)))

        g = re.fullmatch(r"ghost_tile(\d+)", name)
        if g:
            ghost.add(int(g.group(1)))

    both = sorted(matmul & ghost)
    if not both:
        raise RuntimeError(
            "No matching matmul_tile*/ghost_tile* register pairs found.\n"
            f"Available DataBin attrs: {attrs}"
        )

    missing_matmul = sorted(ghost - matmul)
    missing_ghost = sorted(matmul - ghost)

    if missing_matmul or missing_ghost:
        print("[warn] unmatched tile registers detected:")
        if missing_matmul:
            print(f"       ghost exists but matmul missing for tiles: {missing_matmul}")
        if missing_ghost:
            print(f"       matmul exists but ghost missing for tiles: {missing_ghost}")

    return both


def extract_bitstrings(register_obj):
    """
    Extract bitstrings from a Qiskit Runtime register object.

    Current code path uses get_bitstrings(), matching your working extraction.
    """
    if hasattr(register_obj, "get_bitstrings"):
        return register_obj.get_bitstrings()

    raise RuntimeError(
        f"Register object does not expose get_bitstrings(). "
        f"type={type(register_obj)} attrs={public_attrs(register_obj)}"
    )


# =============================================================================
# SUBMIT MODE
# =============================================================================
def run_submit(args):
    token = get_token()

    matrix_a = np.array(args.matrix_a) if args.matrix_a else DEFAULT_MATRIX_A
    matrix_b = np.array(args.matrix_b) if args.matrix_b else DEFAULT_MATRIX_B
    angles_a = data_to_angles(matrix_a)
    angles_b = data_to_angles(matrix_b)

    print(f"\n{'='*78}\n  GHOST ORACLE SUITE — QPU BASE SUBMITTER\n{'='*78}")
    print(f"  Backend       : {args.backend}")
    print(f"  Shots         : {args.shots}")
    print(f"  Matrix A      : {matrix_a}")
    print(f"  Matrix B      : {matrix_b}")
    print(f"  Angle scale   : {ANGLE_SCALE}")

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(args.backend)

    print(f"\n[SETUP] Selecting low-error 7-qubit clusters...")
    clusters = find_qubit_clusters(backend)
    num_tiles = len(clusters)
    if num_tiles == 0:
        sys.stderr.write("[ERROR] No 7-qubit clusters found on this backend.\n")
        sys.exit(1)

    rows, cols = len(matrix_a), len(matrix_b)
    pairs = [(r, c) for r in range(rows) for c in range(cols)][:num_tiles]
    print(f"        Found {num_tiles} tiles covering pairs {pairs}")

    print(f"\n[BUILD] Constructing tiled circuit ({num_tiles * 7} qubits total)...")
    qc = build_circuit(num_tiles, pairs, angles_a, angles_b)
    global_layout = [q for cluster in clusters for q in cluster]

    pm = generate_preset_pass_manager(
        optimization_level=0,
        backend=backend,
        initial_layout=global_layout,
    )
    transpiled = pm.run(qc)

    print(f"[BUILD] Injecting XY4 dynamical decoupling sequence...")
    durations = InstructionDurations.from_backend(backend)
    for i in range(backend.num_qubits):
        try:
            durations.update([('y', (i,), durations.get('x', i))])
        except (KeyError, TypeError) as e:
            # Qubit lacks an 'x' duration entry; XY4 won't be padded there.
            continue

    xy4_sequence = [XGate(), YGate(), XGate(), YGate()]
    dd_pm = PassManager([
        ALAPScheduleAnalysis(durations),
        PadDynamicalDecoupling(durations, xy4_sequence),
    ])
    transpiled_dd = dd_pm.run(transpiled)

    translation_pm = PassManager([BasisTranslator(sel, backend.operation_names)])
    isa_circuit = translation_pm.run(transpiled_dd)

    print(f"\n[SUBMIT] Sending job to {args.backend}...")
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=args.shots)
    job_id = job.job_id()

    print(f"\n{'='*78}")
    print(f"  JOB SUBMITTED")
    print(f"{'='*78}")
    print(f"  Job ID    : {job_id}")
    print(f"  Backend   : {args.backend}")
    print(f"  Tiles     : {num_tiles}")
    print(f"  Shots     : {args.shots}")
    print(f"\n  Next step:")
    print(f"      python qpu.py dump {job_id}")
    print(f"{'='*78}\n")


# =============================================================================
# DUMP MODE
# =============================================================================
def run_dump(args):
    job_id = args.job_id

    print(f"\n{'=' * 78}\n  GHOST ORACLE SUITE — JOB DUMP\n{'=' * 78}")
    print(f"  Job ID    : {job_id}")
    print(f"  Data dir  : {DATA_DIR}")

    print("\n[LOAD] Connecting to IBM Runtime...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    job = service.job(job_id)
    result = job.result()[0]
    databin = result.data

    backend_name = job.backend().name if hasattr(job, "backend") else "unknown"
    print(f"        Backend: {backend_name}")

    if args.num_tiles is None:
        tile_indices = infer_tile_indices(databin)
        print(f"  Tiles     : auto-detected {len(tile_indices)} -> {tile_indices}")
    else:
        tile_indices = list(range(args.num_tiles))
        print(f"  Tiles     : requested {args.num_tiles} -> {tile_indices}")

    data = {
        "job_id": job_id,
        "backend": backend_name,
        "num_tiles": len(tile_indices),
        "tile_indices": np.array(tile_indices, dtype=np.int32),
    }

    print("\n[EXTRACT] Per-tile bitstrings...")
    for out_t, t in enumerate(tile_indices):
        matmul_name = f"matmul_tile{t}"
        ghost_name = f"ghost_tile{t}"

        if not hasattr(databin, matmul_name):
            raise AttributeError(
                f"DataBin has no register '{matmul_name}'. "
                f"Available tile registers: {infer_tile_indices(databin)}"
            )
        if not hasattr(databin, ghost_name):
            raise AttributeError(
                f"DataBin has no register '{ghost_name}'. "
                f"Available tile registers: {infer_tile_indices(databin)}"
            )

        c_bits = extract_bitstrings(getattr(databin, matmul_name))
        g_bits = extract_bitstrings(getattr(databin, ghost_name))

        # Control: single bit per shot.
        ctrl_arr = np.array([int(b) for b in c_bits], dtype=np.uint8)

        # Ghost: 4 bits per shot, reversed to [a1, a2, b1, b2] ordering.
        ghost_arr = np.array(
            [[int(x) for x in bs[::-1]] for bs in g_bits],
            dtype=np.uint8,
        )

        # Store contiguously as ctrl_tile0..N-1 even if original tile IDs differ.
        data[f"ctrl_tile{out_t}"] = ctrl_arr
        data[f"ghost_tile{out_t}"] = ghost_arr

        print(
            f"  tile{t} -> stored tile{out_t}: "
            f"ctrl shape {ctrl_arr.shape}, "
            f"ghost shape {ghost_arr.shape}, "
            f"p(0)={(ctrl_arr == 0).mean():.4f}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else DATA_DIR / f"job_{job_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_path, **data)

    print(f"\n{'=' * 78}")
    print("  DUMP COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Output    : {out_path}")
    print(f"  Backend   : {backend_name}")
    print(f"  Tiles     : {len(tile_indices)}")
    print(f"{'=' * 78}\n")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — G_M QPU base submit + dump",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ---- submit ----
    ps = sub.add_parser(
        "submit",
        help="Build and submit the tiled Hadamard-test circuit. Prints the job ID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ps.add_argument("--backend", default=DEFAULT_BACKEND_NAME,
                    help="IBM Quantum backend name.")
    ps.add_argument("--shots", type=int, default=DEFAULT_SHOTS,
                    help="Number of shots.")
    ps.add_argument("--matrix-a", type=float, nargs="+", default=None,
                    help="Override row angles (space-separated floats).")
    ps.add_argument("--matrix-b", type=float, nargs="+", default=None,
                    help="Override column angles (space-separated floats).")

    # ---- dump ----
    pd = sub.add_parser(
        "dump",
        help="Fetch a completed job by ID and freeze it to data/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pd.add_argument("job_id", help="IBM Runtime job ID.")
    pd.add_argument(
        "--num-tiles",
        type=int,
        default=None,
        help=(
            "Number of tiles in the job. If omitted, infer from DataBin "
            "register names like matmul_tile0, ghost_tile0, ..."
        ),
    )
    pd.add_argument(
        "--out",
        default=None,
        help="Optional output .npz path. Default: <repo>/data/job_<JOB_ID>.npz",
    )

    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "submit":
        run_submit(args)
    elif args.mode == "dump":
        run_dump(args)


if __name__ == "__main__":
    main()