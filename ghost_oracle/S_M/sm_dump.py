#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
S_M STEP 2 — DUMP QISKIT DATA
==============================================================================
One-button Runtime result dumper for S_M jobs.

Default behavior
----------------
Fetches a completed SamplerV2 job, extracts shot-order classical registers, and
saves a stable `.npz` file for analysis.

This script can run with only a JOB_ID if the job was submitted by sm_submit.py:

    python ghost_oracle/S_M/sm_dump.py <JOB_ID>

It automatically looks for:

    data/sm_job_<JOB_ID>.json

If JOB_ID is omitted, it uses:

    data/latest_sm_job.json

Output
------
For S_M flag/superposition jobs:

    data/sm_data_<init_state>_<JOB_ID>.npz

Arrays:
    data_d{d}      uint8, shape (shots, d)
    synd_d{d}      uint8, shape (shots, rounds, d-1)
    flag_d{d}      optional

Usage
-----
Dump latest submitted job:

    python ghost_oracle/S_M/sm_dump.py

Dump a specific job:

    python ghost_oracle/S_M/sm_dump.py <JOB_ID>

Debug registers:

    python ghost_oracle/S_M/sm_dump.py <JOB_ID> --list-registers
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    _HAVE_RUNTIME = True
except Exception:
    QiskitRuntimeService = None
    _HAVE_RUNTIME = False


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S_M step 2 — dump a Qiskit Runtime job to an analysis .npz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("job_id", nargs="?", default=None, help="IBM Quantum Runtime job ID. Defaults to latest submitted S_M job.")
    p.add_argument("--meta", default=None, help="Optional metadata path. Usually not needed for jobs from sm_submit.py.")
    p.add_argument("--save", default=None, help="Output .npz path. Defaults to data/sm_data_<state>_<JOB_ID>.npz.")
    p.add_argument("--channel", default="ibm_quantum_platform")
    p.add_argument("--instance", default=None)
    p.add_argument("--reverse-bits", action="store_true")
    p.add_argument("--list-registers", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not _HAVE_RUNTIME:
        sys.exit("[FATAL] qiskit-ibm-runtime is not installed.")

    job_id, meta_path = resolve_job_and_meta(args.job_id, args.meta)
    meta = load_json(meta_path)

    print(f"\n{'=' * 96}")
    print("  S_M STEP 2 — DUMP QISKIT DATA")
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
    print(f"{'=' * 96}\n")


if __name__ == "__main__":
    main()
