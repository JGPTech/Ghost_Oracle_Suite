#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — QPU BASE SUBMITTER
==============================================================================
Builds the tiled Hadamard-test circuit described in the Ghost Oracle paper
(Section 2), submits it to an IBM Quantum backend, and prints the job ID.

This script does NOT wait for results and does NOT perform analysis.
After submission, run:

    python dump.py <JOB_ID>

to fetch the shot data once the job completes. The dumped .npz is the
canonical QPU base consumed by the projection script and probes.

Per-tile circuit (qubits: a1, v1, a2, ctrl, b1, v2, b2):
    1. Ry(theta_a) on v1; Ry(theta_b) on v2
    2. CNOT(v1 -> a1); CNOT(v1 -> a2); CNOT(v2 -> b1); CNOT(v2 -> b2)
    3. H on ctrl
    4. CSWAP(ctrl; v1, v2)
    5. H on ctrl
    6. Measure {ctrl, a1, a2, b1, b2}

Each tile covers one (r, c) entry of the parameterized matrix family. The
output registers are named matmul_tile{t} (control bit) and ghost_tile{t}
(ancilla bits) to remain bit-compatible with the sample dumps shipped in `data/`.

Usage:
    export IBM_QUANTUM_TOKEN=<your_token>
    python qpu.py                              # use defaults
    python qpu.py --backend ibm_marrakesh      # override backend
    python qpu.py --shots 8192                 # override shot count
==============================================================================
"""

import argparse
import os
import sys
import warnings

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
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — QPU base submitter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backend", default=DEFAULT_BACKEND_NAME,
                   help="IBM Quantum backend name.")
    p.add_argument("--shots", type=int, default=DEFAULT_SHOTS,
                   help="Number of shots.")
    p.add_argument("--matrix-a", type=float, nargs="+", default=None,
                   help="Override row angles (space-separated floats).")
    p.add_argument("--matrix-b", type=float, nargs="+", default=None,
                   help="Override column angles (space-separated floats).")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
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
    print(f"      python dump.py {job_id}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()