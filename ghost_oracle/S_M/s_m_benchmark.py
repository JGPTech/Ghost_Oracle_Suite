#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
S_M BENCHMARK — FINAL SYNDROME METRIC OPERATOR BENCHMARK
==============================================================================

Purpose
-------
Canonical benchmark runner for the S_M operator.

S_M is the Syndrome Metric channel: a syndrome-spacetime field operator. It
measures whether final data edge parity and repeated syndrome records form a
load-bearing field structure.

This file intentionally keeps the S_M claim narrow:

    S_M = {S[t,i], E[i], A[t,i]}

where:

    D[i]   = final data bit at code position i
    E[i]   = D[i] XOR D[i+1]
    S[t,i] = measured syndrome bit at round t and edge i
    A[t,i] = 1 - (S[t,i] XOR E[i])

The stress tensor is NOT the headline S_M claim here. Stress-derived features
belong to T_S. This benchmark may optionally report very small continuity
summaries, but the default feature set is S_M-only.

What this benchmark tests
-------------------------
For each substrate/base:

    geo      synthetic analytical/reference S_M field
    gproj    GPU-generated S_M field base from s_m_gpu_generate.py
    qproj    QPU-dumped S_M field base from sm_qpu.py dump

it runs three field-level tasks:

    Task A: real-vs-control classification
    Task B: control-source classification
    Task C: code-distance prediction

using windowed S_M features:

    raw_rates
    detection_rates
    agreement_profiles
    sm_field
    sm_all

Controls deliberately destroy different parts of the channel:

    real
    shot_shuffle_synd
    time_shuffle_synd
    edge_shuffle_synd
    uniform_synd
    final_shuffle
    all_uniform
    time_reverse_synd
    edge_reverse_synd

The key read is the same forensic discipline as G_M:

    freeze the record
    build matched controls
    scramble the channel
    compare substrates
    measure what survives

Valid S_M signatures
--------------------
A useful S_M run should show some combination of:

    agreement/field features outperform raw scalar rates
    real records separate from destructive controls
    final_shuffle hurts agreement features
    time/edge shuffles hurt field features
    windowed aggregation improves stability over single shots
    distance prediction is above chance when field structure scales with d
    qproj/gproj/geo show comparable profile structure when calibrated

Non-claims
----------
This benchmark does NOT claim:

    S_M is a logical-error-rate benchmark
    S_M is the T_S stress tensor
    S_M is a token retrieval benchmark
    S_M is a universal hardware advantage claim

Usage
-----
Default auto-discovery run:

    python ghost_oracle/S_M/s_m_benchmark.py

Use explicit bases:

    python ghost_oracle/S_M/s_m_benchmark.py \
      --qpu-base ghost_oracle/S_M/data/sm_data_plus_<JOB_ID>.npz \
      --gpu-base ghost_oracle/S_M/data/sm_gpu_data_plus_<TAG>.npz

Probe mode:

    python ghost_oracle/S_M/s_m_benchmark.py --probe

Sweep mode:

    python ghost_oracle/S_M/s_m_benchmark.py --sweep ALL

Useful options:

    python ghost_oracle/S_M/s_m_benchmark.py --windows 8 16 32 64
    python ghost_oracle/S_M/s_m_benchmark.py --skip-geo
    python ghost_oracle/S_M/s_m_benchmark.py --skip-gpu
    python ghost_oracle/S_M/s_m_benchmark.py --skip-qpu
    python ghost_oracle/S_M/s_m_benchmark.py --no-plots
    python ghost_oracle/S_M/s_m_benchmark.py --write-windows

Outputs
-------
    S_M/analysis/s_m_<timestamp>/
        result.json
        summary.csv
        per_feature.csv
        control_collapse.csv
        substrate_agreement.csv
        window_rows.csv                  optional with --write-windows
        artifacts.npz
        task_A_accuracy.png              if matplotlib available
        task_B_accuracy.png              if matplotlib available
        task_C_accuracy.png              if matplotlib available
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    plt = None
    _HAVE_MPL = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    _HAVE_SKLEARN = True
except Exception:
    LogisticRegression = None
    RandomForestClassifier = None
    roc_auc_score = None
    _HAVE_SKLEARN = False

try:
    import cupy as cp
    _HAVE_CUPY = True
    _CUPY_IMPORT_ERROR = None
except Exception as e:
    cp = None
    _HAVE_CUPY = False
    _CUPY_IMPORT_ERROR = repr(e)

# =============================================================================
# PATHS / DEFAULTS
# =============================================================================

HERE = Path(__file__).resolve().parent


S_M_DIR = HERE

DATA_DIR = S_M_DIR / "data"
ANALYSIS_DIR = S_M_DIR / "analysis"
KERNEL_PATH = S_M_DIR / "kernels" / "sm_kernel.cu"

def print_cuda_debug(args: argparse.Namespace) -> None:
    print("\n[CUDA DEBUG]")
    print(f"  _HAVE_CUPY       : {_HAVE_CUPY}")
    print(f"  CuPy import error: {_CUPY_IMPORT_ERROR}")
    print(f"  --no-cuda        : {getattr(args, 'no_cuda', False)}")
    print(f"  HERE             : {HERE}")
    print(f"  REPO_ROOT        : {REPO_ROOT}")
    print(f"  S_M_DIR          : {S_M_DIR}")
    print(f"  DATA_DIR         : {DATA_DIR}")
    print(f"  ANALYSIS_DIR     : {ANALYSIS_DIR}")
    print(f"  KERNEL_PATH      : {KERNEL_PATH}")
    print(f"  Kernel exists    : {KERNEL_PATH.exists()}")

    if KERNEL_PATH.exists():
        try:
            print(f"  Kernel size      : {KERNEL_PATH.stat().st_size} bytes")
        except Exception as e:
            print(f"  Kernel stat error: {repr(e)}")

    if _HAVE_CUPY:
        try:
            print(f"  CuPy version     : {cp.__version__}")
        except Exception as e:
            print(f"  CuPy version err : {repr(e)}")

        try:
            ndev = cp.cuda.runtime.getDeviceCount()
            print(f"  CUDA devices     : {ndev}")
            if ndev > 0:
                props = cp.cuda.runtime.getDeviceProperties(0)
                name = props.get("name", b"unknown")
                if isinstance(name, bytes):
                    name = name.decode(errors="ignore")
                print(f"  CUDA device 0    : {name}")
        except Exception as e:
            print(f"  CUDA device error: {repr(e)}")

    print("[/CUDA DEBUG]\n")

DEFAULT_DISTANCES = [3, 5, 7, 9]
DEFAULT_ROUNDS = 10
DEFAULT_SHOTS = 4096
DEFAULT_WINDOWS = [8, 16, 32, 64]
DEFAULT_CONTROL_MODES = [
    "real",
    "shot_shuffle_synd",
    "time_shuffle_synd",
    "edge_shuffle_synd",
    "uniform_synd",
    "final_shuffle",
    "all_uniform",
    "time_reverse_synd",
    "edge_reverse_synd",
]
FEATURE_FAMILIES = [
    "raw_rates",
    "detection_rates",
    "agreement_profiles",
    "sm_field",
    "sm_all",
]


# =============================================================================
# UTILITIES
# =============================================================================

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2)


def write_csv(path: Path, rows: List[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def bits(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x).astype(np.int64) & 1).astype(np.uint8)


def pad_to_width(X: np.ndarray, width: int) -> np.ndarray:
    if X.shape[1] == width:
        return X.astype(np.float32)
    out = np.zeros((X.shape[0], width), dtype=np.float32)
    out[:, : X.shape[1]] = X.astype(np.float32)
    return out


def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True)
    sig[sig < 1e-6] = 1.0
    return mu.astype(np.float32), sig.astype(np.float32)


def zscore_apply(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return ((X - mu) / sig).astype(np.float32)


def label_encode(y_raw: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    labels = sorted(str(x) for x in set(y_raw.tolist()))
    idx = {lab: i for i, lab in enumerate(labels)}
    y = np.asarray([idx[str(x)] for x in y_raw.tolist()], dtype=np.int64)
    return y, labels


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for lab in np.unique(y_true):
        m = y_true == lab
        if np.any(m):
            vals.append(float(np.mean(y_pred[m] == lab)))
    return float(np.mean(vals)) if vals else float("nan")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int) -> np.ndarray:
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def stratified_split(y: np.ndarray, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr: List[int] = []
    te: List[int] = []
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        if len(idx) <= 1:
            tr.extend(idx.tolist())
            continue
        n_test = max(1, int(round(len(idx) * float(test_frac))))
        n_test = min(n_test, len(idx) - 1)
        te.extend(idx[:n_test].tolist())
        tr.extend(idx[n_test:].tolist())
    tr_a = np.asarray(tr, dtype=np.int64)
    te_a = np.asarray(te, dtype=np.int64)
    rng.shuffle(tr_a)
    rng.shuffle(te_a)
    return tr_a, te_a


def knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int, metric: str) -> np.ndarray:
    k = int(max(1, min(k, len(y_train))))
    if metric == "cosine":
        trn = np.linalg.norm(X_train, axis=1, keepdims=True)
        tst = np.linalg.norm(X_test, axis=1, keepdims=True)
        trn[trn < 1e-12] = 1.0
        tst[tst < 1e-12] = 1.0
        Xtr = X_train / trn
        Xte = X_test / tst
        score = Xte @ Xtr.T
        nn = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
    elif metric == "euclidean":
        q2 = np.sum(X_test * X_test, axis=1, keepdims=True)
        t2 = np.sum(X_train * X_train, axis=1, keepdims=True).T
        dist = q2 + t2 - 2.0 * (X_test @ X_train.T)
        nn = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    else:
        raise ValueError(metric)

    pred = []
    for row in nn:
        vals, counts = np.unique(y_train[row], return_counts=True)
        pred.append(vals[np.argmax(counts)])
    return np.asarray(pred, dtype=np.int64)


# =============================================================================
# S_M RECORDS
# =============================================================================

@dataclass
class SMRecord:
    substrate: str
    source: str
    distance: int
    data: np.ndarray      # uint8, shape (shots, d)
    synd: np.ndarray      # uint8, shape (shots, rounds, d-1)
    flags: Optional[np.ndarray] = None


def terminal_edge_parity(data: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(data[:, :-1], data[:, 1:]).astype(np.uint8)


def agreement_field(data: np.ndarray, synd: np.ndarray) -> np.ndarray:
    edges = terminal_edge_parity(data)[:, None, :]
    return (1.0 - np.bitwise_xor(edges, synd).astype(np.float32)).astype(np.float32)


def detection_events(synd: np.ndarray) -> np.ndarray:
    if synd.shape[1] < 2:
        return np.zeros((synd.shape[0], 0, synd.shape[2]), dtype=np.float32)
    return np.bitwise_xor(synd[:, 1:, :], synd[:, :-1, :]).astype(np.float32)


def load_sm_npz(path: Path, substrate: str) -> Tuple[List[SMRecord], Dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    files = set(z.files)
    if "distances" not in files:
        raise KeyError(f"{path} is missing 'distances'; expected S_M dump/base schema.")

    meta = {}
    for k in ["schema", "job_id", "backend", "shots", "rounds", "flag_level", "logical_init", "basis", "init_state"]:
        if k in files:
            v = z[k]
            meta[k] = v.item() if getattr(v, "shape", ()) == () else v

    records: List[SMRecord] = []
    for d0 in np.asarray(z["distances"]).astype(int).tolist():
        d = int(d0)
        dk = f"data_d{d}"
        sk = f"synd_d{d}"
        fk = f"flag_d{d}"
        if dk not in files or sk not in files:
            continue
        records.append(
            SMRecord(
                substrate=substrate,
                source=str(path),
                distance=d,
                data=bits(z[dk]),
                synd=bits(z[sk]),
                flags=bits(z[fk]) if fk in files else None,
            )
        )
    if not records:
        raise KeyError(f"{path} had distances but no usable data_d*/synd_d* arrays.")
    return records, meta


def latest_file(kind: str) -> Optional[Path]:
    """Find latest known S_M data file for qpu or gpu."""
    candidates = []
    if kind == "qpu":
        latest = DATA_DIR / "latest_sm_data.json"
        if latest.exists():
            try:
                obj = load_json(latest)
                p = Path(obj.get("npz", ""))
                if p.exists():
                    return p
            except Exception:
                pass
        candidates.extend(DATA_DIR.glob("sm_data_*.npz"))
    elif kind == "gpu":
        latest = DATA_DIR / "latest_sm_gpu_data.json"
        if latest.exists():
            try:
                obj = load_json(latest)
                p = Path(obj.get("npz", ""))
                if p.exists():
                    return p
            except Exception:
                pass
        candidates.extend(DATA_DIR.glob("sm_gpu_data_*.npz"))
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)



# =============================================================================
# CUDA FEATURE EXTRACTION
# =============================================================================

@dataclass
class SMCudaContext:
    module: Any
    kernel: Any
    path: Path
    enabled: bool = True


_SM_CUDA_CONTEXT: Optional[SMCudaContext] = None
_SM_CUDA_FAILED = False


def compile_sm_cuda_kernel(args: argparse.Namespace) -> Optional[SMCudaContext]:
    """Compile S_M/kernels/sm_kernel.cu with CuPy RawModule if available."""
    global _SM_CUDA_CONTEXT, _SM_CUDA_FAILED

    if getattr(args, "no_cuda", False):
        return None
    if _SM_CUDA_CONTEXT is not None:
        return _SM_CUDA_CONTEXT
    if _SM_CUDA_FAILED:
        return None
    if not _HAVE_CUPY:
        _SM_CUDA_FAILED = True
        return None
    if not KERNEL_PATH.exists():
        _SM_CUDA_FAILED = True
        return None

    try:
        src = KERNEL_PATH.read_text(encoding="utf-8")
        mod = cp.RawModule(code=src, options=("-use_fast_math",))
        kernel = mod.get_function("sm_window_features_kernel")
        _SM_CUDA_CONTEXT = SMCudaContext(module=mod, kernel=kernel, path=KERNEL_PATH)
        return _SM_CUDA_CONTEXT
    except Exception as e:
        _SM_CUDA_FAILED = True
        if not getattr(args, "quiet", False):
            print(f"[CUDA][warn] Could not compile {KERNEL_PATH}: {e}")
        return None


def window_starts(n: int, window: int, max_windows: int, rng: np.random.Generator) -> List[int]:
    starts = list(range(0, n - window + 1, window))
    if not starts:
        return []
    if max_windows > 0 and len(starts) > max_windows:
        starts = sorted(rng.choice(starts, size=max_windows, replace=False).tolist())
    return [int(s) for s in starts]


def sm_scalar_descriptors_for_windows(
    data: np.ndarray,
    synd: np.ndarray,
    window: int,
    selected_window_ids: np.ndarray,
    agreement_profiles_rows: np.ndarray,
) -> np.ndarray:
    """Exact legacy scalar descriptors appended to sm_field.

    The CUDA kernel computes the core field reductions. These eight scalar
    descriptors are kept here to preserve compatibility with the original
    benchmark feature definitions.
    """
    out = np.zeros((len(selected_window_ids), 8), dtype=np.float32)
    edges = data.shape[1] - 1
    rounds = synd.shape[1]

    for row_i, wid in enumerate(selected_window_ids.astype(int).tolist()):
        start = int(wid) * int(window)
        data_w = data[start:start + window]
        synd_w = synd[start:start + window]

        A = agreement_field(data_w, synd_w)
        X = detection_events(synd_w)

        edge_profile = agreement_profiles_rows[row_i, :edges]
        time_profile = agreement_profiles_rows[row_i, edges:edges + rounds]

        A_flat = A.reshape(A.shape[0], -1)
        if X.size:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = np.zeros((A.shape[0], 0), dtype=np.float32)

        out[row_i] = np.asarray([
            float(A.mean()),
            float(A.std()),
            float(edge_profile.std()),
            float(time_profile.std()),
            float(X.mean()) if X.size else 0.0,
            float(X.std()) if X.size else 0.0,
            float(A_flat.mean(axis=1).std()),
            float(X_flat.mean(axis=1).std()) if X.size else 0.0,
        ], dtype=np.float32)

    return out


def window_features_cuda_batch(
    data: np.ndarray,
    synd: np.ndarray,
    window: int,
    selected_starts: Sequence[int],
    cuda_ctx: SMCudaContext,
    threads: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Return batched feature arrays for selected non-overlapping windows.

    The CUDA kernel computes all non-overlapping windows, then the host selects
    the same subset the reference benchmark would have used.
    """
    if not selected_starts:
        return None

    try:
        data = np.ascontiguousarray(bits(data), dtype=np.uint8)
        synd = np.ascontiguousarray(bits(synd), dtype=np.uint8)

        shots, d = data.shape
        if synd.ndim != 3:
            return None
        rounds = int(synd.shape[1])
        edges = int(synd.shape[2])
        if edges != d - 1 or rounds <= 0:
            return None

        n_all = int(shots // int(window))
        if n_all <= 0:
            return None

        selected_window_ids = np.asarray([int(s) // int(window) for s in selected_starts], dtype=np.int64)
        selected_window_ids = selected_window_ids[(selected_window_ids >= 0) & (selected_window_ids < n_all)]
        if selected_window_ids.size == 0:
            return None

        det_rounds = max(0, rounds - 1)
        raw_width = d + rounds * edges
        det_width = det_rounds * edges
        agree_width = edges + rounds
        field_width_core = rounds * edges + det_width

        d_data = cp.asarray(data)
        d_synd = cp.asarray(synd)
        raw_out = cp.empty((n_all, raw_width), dtype=cp.float32)
        det_out = cp.empty((n_all, det_width), dtype=cp.float32)
        agree_out = cp.empty((n_all, agree_width), dtype=cp.float32)
        field_out = cp.empty((n_all, field_width_core), dtype=cp.float32)

        tpb = int(max(32, min(256, threads)))
        cuda_ctx.kernel(
            (n_all,),
            (tpb,),
            (
                d_data,
                d_synd,
                np.int32(shots),
                np.int32(d),
                np.int32(rounds),
                np.int32(window),
                raw_out,
                det_out,
                agree_out,
                field_out,
            ),
        )
        cp.cuda.Stream.null.synchronize()

        ids_cp = cp.asarray(selected_window_ids, dtype=cp.int64)
        raw = cp.asnumpy(raw_out[ids_cp]).astype(np.float32)
        det = cp.asnumpy(det_out[ids_cp]).astype(np.float32)
        agree = cp.asnumpy(agree_out[ids_cp]).astype(np.float32)
        field_core = cp.asnumpy(field_out[ids_cp]).astype(np.float32)

        scalars = sm_scalar_descriptors_for_windows(data, synd, window, selected_window_ids, agree)
        sm_field = np.concatenate([field_core, scalars], axis=1).astype(np.float32)
        sm_all = np.concatenate([raw, det, agree, sm_field], axis=1).astype(np.float32)

        return {
            "raw_rates": raw,
            "detection_rates": det,
            "agreement_profiles": agree,
            "sm_field": sm_field,
            "sm_all": sm_all,
        }
    except Exception:
        return None

# =============================================================================
# GEO / SYNTHETIC FIELD MODEL
# =============================================================================

def synthetic_geo_records(
    distances: Sequence[int],
    shots: int,
    rounds: int,
    seed: int,
    init_state: str = "plus",
    source: str = "synthetic_geo",
) -> List[SMRecord]:
    """
    Build a classical reference S_M field.

    This is not meant to be a hardware simulator. It is a controlled field model
    that creates final edge parity E and a syndrome field S[t,i] with tunable
    agreement, temporal persistence, and edge-dependent structure. Controls then
    test whether the benchmark detects destruction of that structure.
    """
    rng = np.random.default_rng(seed)
    records: List[SMRecord] = []

    for d in [int(x) for x in distances]:
        # Cat-like final readout: each shot chooses logical branch, then small
        # per-bit flips. For plus/minus in Z basis this gives broad records,
        # which is exactly why edge parity matters more than majority LER.
        branch = rng.integers(0, 2, size=(shots, 1), dtype=np.uint8)
        flip_p = min(0.015 + 0.0025 * d, 0.08)
        flips = (rng.random((shots, d)) < flip_p).astype(np.uint8)
        data = np.bitwise_xor(branch, flips).astype(np.uint8)
        E = terminal_edge_parity(data)

        synd = np.zeros((shots, rounds, d - 1), dtype=np.uint8)
        # Edge-dependent and time-dependent agreement probabilities. Larger d
        # has slightly richer field variation so Task C has a real signal.
        edge_axis = np.linspace(-1.0, 1.0, d - 1, dtype=np.float32)
        time_axis = np.linspace(0.0, 1.0, rounds, dtype=np.float32)
        base_agree = 0.86 - 0.015 * (d - 3)
        base_agree = float(np.clip(base_agree, 0.66, 0.92))

        edge_mod = 0.035 * np.cos(np.pi * edge_axis)
        time_mod = 0.025 * np.sin(2.0 * np.pi * time_axis)
        P_agree = base_agree + time_mod[:, None] + edge_mod[None, :]
        P_agree = np.clip(P_agree, 0.55, 0.96).astype(np.float32)

        for t in range(rounds):
            agree = rng.random((shots, d - 1)) < P_agree[t][None, :]
            noise = (1 - agree).astype(np.uint8)
            synd[:, t, :] = np.bitwise_xor(E, noise).astype(np.uint8)

        records.append(SMRecord("geo", source, d, data, synd, None))

    return records


# =============================================================================
# CONTROLS AND FEATURES
# =============================================================================

def mutate_record(data: np.ndarray, synd: np.ndarray, mode: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    d = data.copy()
    s = synd.copy()

    if mode == "real":
        return d, s

    if mode == "shot_shuffle_synd":
        return d, s[rng.permutation(s.shape[0])]

    if mode == "time_shuffle_synd":
        for sh in range(s.shape[0]):
            for e in range(s.shape[2]):
                s[sh, :, e] = s[sh, rng.permutation(s.shape[1]), e]
        return d, s

    if mode == "edge_shuffle_synd":
        for sh in range(s.shape[0]):
            for t in range(s.shape[1]):
                s[sh, t, :] = s[sh, t, rng.permutation(s.shape[2])]
        return d, s

    if mode == "uniform_synd":
        p = s.mean(axis=(0, 1), keepdims=True)
        return d, (rng.random(s.shape) < p).astype(np.uint8)

    if mode == "final_shuffle":
        return d[rng.permutation(d.shape[0])], s

    if mode == "all_uniform":
        pd = d.mean(axis=0, keepdims=True)
        ps = s.mean(axis=(0, 1), keepdims=True)
        d2 = (rng.random(d.shape) < pd).astype(np.uint8)
        s2 = (rng.random(s.shape) < ps).astype(np.uint8)
        return d2, s2

    if mode == "time_reverse_synd":
        return d, s[:, ::-1, :].copy()

    if mode == "edge_reverse_synd":
        return d, s[:, :, ::-1].copy()

    raise ValueError(f"unknown control mode: {mode}")


def window_indices(n: int, window: int, max_windows: int, rng: np.random.Generator) -> List[np.ndarray]:
    return [np.arange(s, s + window, dtype=np.int64) for s in window_starts(n, window, max_windows, rng)]


def window_features(data_w: np.ndarray, synd_w: np.ndarray) -> Dict[str, np.ndarray]:
    A = agreement_field(data_w, synd_w)
    X = detection_events(synd_w)

    raw_rates = np.concatenate([
        data_w.mean(axis=0).reshape(-1),
        synd_w.mean(axis=0).reshape(-1),
    ]).astype(np.float32)

    detection_rates = X.mean(axis=0).reshape(-1).astype(np.float32) if X.size else np.zeros(0, dtype=np.float32)

    edge_profile = A.mean(axis=(0, 1)).reshape(-1).astype(np.float32)
    time_profile = A.mean(axis=(0, 2)).reshape(-1).astype(np.float32)
    agreement_profiles = np.concatenate([edge_profile, time_profile]).astype(np.float32)

    # S_M field signature: keep agreement + detection shape, no stress tensor.
    agreement_field_mean = A.mean(axis=0).reshape(-1).astype(np.float32)
    detection_field_mean = X.mean(axis=0).reshape(-1).astype(np.float32) if X.size else np.zeros(0, dtype=np.float32)

    # Small scalar descriptors that are still S_M field descriptors, not T_S.
    A_flat = A.reshape(A.shape[0], -1)
    X_flat = X.reshape(X.shape[0], -1) if X.size else np.zeros((A.shape[0], 0), dtype=np.float32)
    sm_scalars = np.asarray([
        float(A.mean()),
        float(A.std()),
        float(edge_profile.std()),
        float(time_profile.std()),
        float(X.mean()) if X.size else 0.0,
        float(X.std()) if X.size else 0.0,
        float(A_flat.mean(axis=1).std()),
        float(X_flat.mean(axis=1).std()) if X.size else 0.0,
    ], dtype=np.float32)

    sm_field = np.concatenate([
        agreement_field_mean,
        detection_field_mean,
        sm_scalars,
    ]).astype(np.float32)

    sm_all = np.concatenate([
        raw_rates,
        detection_rates,
        agreement_profiles,
        sm_field,
    ]).astype(np.float32)

    return {
        "raw_rates": raw_rates,
        "detection_rates": detection_rates,
        "agreement_profiles": agreement_profiles,
        "sm_field": sm_field,
        "sm_all": sm_all,
    }


def build_windowed_dataset(
    records: List[SMRecord],
    modes: Sequence[str],
    window: int,
    max_windows: int,
    seed: int,
    cuda_ctx: Optional[SMCudaContext] = None,
    cuda_threads: int = 256,
) -> Dict[str, Dict[str, Any]]:
    blocks: Dict[str, Dict[str, List[Any]]] = {}

    for rec_idx, rec in enumerate(records):
        for mi, mode in enumerate(modes):
            rng = np.random.default_rng(seed + 1009 * rec.distance + 9176 * rec_idx + 37 * mi + window)
            d_m, s_m = mutate_record(rec.data, rec.synd, mode, rng)
            starts = window_starts(d_m.shape[0], int(window), int(max_windows), rng)
            if not starts:
                continue

            feats_batch: Optional[Dict[str, np.ndarray]] = None
            if cuda_ctx is not None:
                feats_batch = window_features_cuda_batch(
                    d_m, s_m, int(window), starts, cuda_ctx, int(cuda_threads)
                )

            # If CUDA is unavailable or a kernel path fails validation, preserve
            # the original NumPy reference behavior exactly.
            if feats_batch is None:
                feats_rows = []
                for s0 in starts:
                    idx = np.arange(s0, s0 + int(window), dtype=np.int64)
                    feats_rows.append(window_features(d_m[idx], s_m[idx]))
                for row_i, feats in enumerate(feats_rows):
                    for name, vec in feats.items():
                        if name not in blocks:
                            blocks[name] = {
                                "X_blocks": [],
                                "substrate": [],
                                "source": [],
                                "control": [],
                                "binary": [],
                                "distance": [],
                                "window": [],
                            }
                        blocks[name]["X_blocks"].append(vec.reshape(1, -1))
                        blocks[name]["substrate"].append(rec.substrate)
                        blocks[name]["source"].append(rec.source)
                        blocks[name]["control"].append(mode)
                        blocks[name]["binary"].append("real" if mode == "real" else "control")
                        blocks[name]["distance"].append(rec.distance)
                        blocks[name]["window"].append(window)
                continue

            n_rows = next(iter(feats_batch.values())).shape[0]
            for row_i in range(n_rows):
                for name, arr in feats_batch.items():
                    if name not in blocks:
                        blocks[name] = {
                            "X_blocks": [],
                            "substrate": [],
                            "source": [],
                            "control": [],
                            "binary": [],
                            "distance": [],
                            "window": [],
                        }
                    blocks[name]["X_blocks"].append(arr[row_i].reshape(1, -1))
                    blocks[name]["substrate"].append(rec.substrate)
                    blocks[name]["source"].append(rec.source)
                    blocks[name]["control"].append(mode)
                    blocks[name]["binary"].append("real" if mode == "real" else "control")
                    blocks[name]["distance"].append(rec.distance)
                    blocks[name]["window"].append(window)

    out: Dict[str, Dict[str, Any]] = {}
    for name, obj in blocks.items():
        width = max(x.shape[1] for x in obj["X_blocks"])
        X = np.vstack([pad_to_width(x, width) for x in obj["X_blocks"]]).astype(np.float32)
        out[name] = {
            "X": X,
            "substrate": np.asarray(obj["substrate"], dtype=object),
            "source": np.asarray(obj["source"], dtype=object),
            "control": np.asarray(obj["control"], dtype=object),
            "binary": np.asarray(obj["binary"], dtype=object),
            "distance": np.asarray(obj["distance"], dtype=np.int64),
            "window": np.asarray(obj["window"], dtype=np.int64),
        }
    return out


# =============================================================================
# MODELING
# =============================================================================

def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else float("nan"),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, len(labels)),
    }


def run_models_for_task(
    X: np.ndarray,
    y_raw: np.ndarray,
    task: str,
    feature: str,
    substrate: str,
    window: int,
    args: argparse.Namespace,
    seed: int,
) -> List[Dict[str, Any]]:
    y, labels = label_encode(y_raw)
    if len(labels) < 2:
        return []

    tr, te = stratified_split(y, args.test_frac, seed)
    if len(tr) < 2 or len(te) < 1:
        return []

    Xtr_raw, Xte_raw = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    mu, sig = zscore_fit(Xtr_raw)
    Xtr = zscore_apply(Xtr_raw, mu, sig)
    Xte = zscore_apply(Xte_raw, mu, sig)

    rows: List[Dict[str, Any]] = []

    for metric in ("euclidean", "cosine"):
        t0 = time.time()
        yp = knn_predict(Xtr, ytr, Xte, args.knn_k, metric)
        seconds = time.time() - t0
        rows.append({
            "task": task,
            "feature": feature,
            "substrate": substrate,
            "window": int(window),
            "model": f"kNN-{metric}",
            "n_features": int(X.shape[1]),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "seconds": float(seconds),
            **evaluate_prediction(yte, yp, labels),
        })

    if _HAVE_SKLEARN and not args.no_sklearn:
        try:
            t0 = time.time()
            lr = LogisticRegression(max_iter=10000, solver="lbfgs")
            lr.fit(Xtr, ytr)
            yp = lr.predict(Xte)
            seconds = time.time() - t0
            item = {
                "task": task,
                "feature": feature,
                "substrate": substrate,
                "window": int(window),
                "model": "logistic",
                "n_features": int(X.shape[1]),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "seconds": float(seconds),
                **evaluate_prediction(yte, yp, labels),
            }
            if task == "A_real_vs_control" and len(labels) == 2 and roc_auc_score is not None:
                try:
                    proba = lr.predict_proba(Xte)[:, 1]
                    item["auc"] = float(roc_auc_score(yte, proba))
                except Exception:
                    item["auc"] = float("nan")
            rows.append(item)
        except Exception as e:
            rows.append({
                "task": task,
                "feature": feature,
                "substrate": substrate,
                "window": int(window),
                "model": "logistic",
                "error": str(e),
                "balanced_accuracy": float("nan"),
                "accuracy": float("nan"),
                "labels": labels,
                "confusion_matrix": np.zeros((len(labels), len(labels)), dtype=np.int64),
            })

        try:
            t0 = time.time()
            rf = RandomForestClassifier(n_estimators=args.rf_trees, random_state=seed, n_jobs=-1)
            rf.fit(Xtr_raw, ytr)
            yp = rf.predict(Xte_raw)
            seconds = time.time() - t0
            rows.append({
                "task": task,
                "feature": feature,
                "substrate": substrate,
                "window": int(window),
                "model": "random_forest",
                "n_features": int(X.shape[1]),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "seconds": float(seconds),
                **evaluate_prediction(yte, yp, labels),
            })
        except Exception as e:
            rows.append({
                "task": task,
                "feature": feature,
                "substrate": substrate,
                "window": int(window),
                "model": "random_forest",
                "error": str(e),
                "balanced_accuracy": float("nan"),
                "accuracy": float("nan"),
                "labels": labels,
                "confusion_matrix": np.zeros((len(labels), len(labels)), dtype=np.int64),
            })

    return rows


def run_tasks_for_dataset(
    ds_by_feature: Dict[str, Dict[str, Any]],
    substrate: str,
    window: int,
    args: argparse.Namespace,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for feat_name, ds in ds_by_feature.items():
        X = ds["X"]
        # Task A: real vs all controls.
        rows.extend(run_models_for_task(
            X, ds["binary"], "A_real_vs_control", feat_name, substrate, window, args, seed + 11
        ))

        # Task B: identify which control/source. Includes real as a class.
        rows.extend(run_models_for_task(
            X, ds["control"], "B_control_source", feat_name, substrate, window, args, seed + 23
        ))

        # Task C: distance prediction only on real windows, so it tests the real
        # field shape rather than just recognizing the artificial controls.
        m_real = ds["control"] == "real"
        if np.sum(m_real) >= max(4, len(np.unique(ds["distance"][m_real]))):
            rows.extend(run_models_for_task(
                X[m_real], ds["distance"][m_real].astype(str),
                "C_distance_prediction", feat_name, substrate, window, args, seed + 37
            ))

    return rows


# =============================================================================
# AGREEMENT / COLLAPSE REPORTS
# =============================================================================

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a = a[:n]
    b = b[:n]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def real_profiles(records: List[SMRecord]) -> Dict[str, np.ndarray]:
    """Aggregate substrate-level real profile vectors for agreement reporting."""
    chunks: Dict[str, List[np.ndarray]] = {
        "agreement_edge": [],
        "agreement_time": [],
        "agreement_field": [],
        "detection_edge": [],
        "detection_time": [],
        "detection_field": [],
    }
    for rec in records:
        A = agreement_field(rec.data, rec.synd)
        X = detection_events(rec.synd)
        chunks["agreement_edge"].append(A.mean(axis=(0, 1)).reshape(-1).astype(np.float32))
        chunks["agreement_time"].append(A.mean(axis=(0, 2)).reshape(-1).astype(np.float32))
        chunks["agreement_field"].append(A.mean(axis=0).reshape(-1).astype(np.float32))
        if X.size:
            chunks["detection_edge"].append(X.mean(axis=(0, 1)).reshape(-1).astype(np.float32))
            chunks["detection_time"].append(X.mean(axis=(0, 2)).reshape(-1).astype(np.float32))
            chunks["detection_field"].append(X.mean(axis=0).reshape(-1).astype(np.float32))
        else:
            chunks["detection_edge"].append(np.zeros(rec.distance - 1, dtype=np.float32))
            chunks["detection_time"].append(np.zeros(0, dtype=np.float32))
            chunks["detection_field"].append(np.zeros(0, dtype=np.float32))

    out = {}
    for k, vals in chunks.items():
        out[k] = np.concatenate([v.reshape(-1) for v in vals]).astype(np.float32) if vals else np.zeros(0, dtype=np.float32)
    return out


def substrate_agreement(substrate_records: Dict[str, List[SMRecord]]) -> List[dict]:
    profiles = {k: real_profiles(v) for k, v in substrate_records.items() if v}
    subs = sorted(profiles.keys())
    rows: List[dict] = []
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            a, b = subs[i], subs[j]
            row = {"substrate_a": a, "substrate_b": b}
            for key in ["agreement_edge", "agreement_time", "agreement_field", "detection_edge", "detection_time", "detection_field"]:
                va, vb = profiles[a][key], profiles[b][key]
                n = min(len(va), len(vb))
                row[f"{key}_corr"] = safe_corr(va[:n], vb[:n]) if n else float("nan")
                row[f"{key}_l2"] = l2(va[:n], vb[:n]) if n else float("nan")
            rows.append(row)
    return rows


def best_rows(rows: List[dict]) -> List[dict]:
    keyed: Dict[Tuple[str, str, str], dict] = {}
    for r in rows:
        val = float(r.get("balanced_accuracy", float("nan")))
        if math.isnan(val):
            continue
        key = (str(r.get("substrate")), str(r.get("task")), str(r.get("feature")))
        old = keyed.get(key)
        if old is None or val > float(old.get("balanced_accuracy", -1)):
            keyed[key] = r
    return list(keyed.values())


def control_collapse_rows(rows: List[dict]) -> List[dict]:
    """Summarize Task B confusion-style collapse using best Task B rows.

    This is a lightweight benchmark-table helper. The full confusion matrices are
    preserved in result.json. Here we report the best balanced accuracy by
    substrate/feature/window/model for control-source classification.
    """
    out = []
    for r in rows:
        if r.get("task") != "B_control_source":
            continue
        out.append({
            "substrate": r.get("substrate"),
            "feature": r.get("feature"),
            "window": r.get("window"),
            "model": r.get("model"),
            "balanced_accuracy": r.get("balanced_accuracy"),
            "accuracy": r.get("accuracy"),
            "labels": ";".join(map(str, r.get("labels", []))),
            "n_test": r.get("n_test"),
        })
    out.sort(key=lambda x: (-float(x.get("balanced_accuracy") or -1), str(x.get("substrate"))))
    return out


# =============================================================================
# PLOTTING / REPORTING
# =============================================================================

def print_summary(rows: List[dict]) -> None:
    subset = [r for r in rows if not math.isnan(float(r.get("balanced_accuracy", float("nan"))))]
    subset.sort(key=lambda r: (-float(r["balanced_accuracy"]), str(r.get("substrate")), str(r.get("task"))))
    print("\n" + "=" * 132)
    print("  S_M BENCHMARK SUMMARY — BEST ROWS")
    print("=" * 132)
    print(
        f"  {'rank':>4} | {'substrate':<8} | {'task':<24} | {'feature':<20} | "
        f"{'window':>6} | {'model':<15} | {'bal_acc':>8} | {'acc':>8} | {'n_test':>7}"
    )
    print("  " + "-" * 130)
    for i, r in enumerate(subset[:40], 1):
        print(
            f"  {i:>4} | {str(r.get('substrate')):<8} | {str(r.get('task')):<24} | "
            f"{str(r.get('feature')):<20} | {int(r.get('window', 0)):>6} | "
            f"{str(r.get('model')):<15} | {float(r.get('balanced_accuracy')):>8.3f} | "
            f"{float(r.get('accuracy')):>8.3f} | {int(r.get('n_test', 0)):>7}"
        )
    print("=" * 132 + "\n")


def plot_task_bars(rows: List[dict], out_dir: Path) -> None:
    if not _HAVE_MPL:
        return
    for task in ["A_real_vs_control", "B_control_source", "C_distance_prediction"]:
        sub = [r for r in rows if r.get("task") == task and str(r.get("feature")) == "sm_all"]
        if not sub:
            continue
        # Best per substrate/window.
        best: Dict[Tuple[str, int], float] = {}
        for r in sub:
            val = float(r.get("balanced_accuracy", float("nan")))
            if math.isnan(val):
                continue
            key = (str(r.get("substrate")), int(r.get("window", 0)))
            best[key] = max(best.get(key, -1.0), val)
        if not best:
            continue
        labels = [f"{s}\nw{w}" for (s, w) in sorted(best.keys())]
        vals = [best[k] for k in sorted(best.keys())]
        fig = plt.figure(figsize=(max(8, 0.6 * len(labels)), 5), dpi=150)
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(vals)), vals)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(f"S_M {task} — best sm_all model per substrate/window")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        safe = task.replace("/", "_")
        fig.savefig(out_dir / f"{safe}_accuracy.png")
        plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_num_list(values: Optional[Sequence[str]], cast) -> Optional[List[Any]]:
    if values is None:
        return None
    out: List[Any] = []
    for v in values:
        for part in str(v).replace(",", " ").split():
            if part.strip():
                out.append(cast(part.strip()))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S_M final benchmark — syndrome-spacetime field operator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--qpu-base", default=None, help="QPU S_M .npz from sm_qpu.py dump. Defaults to latest if found.")
    p.add_argument("--gpu-base", default=None, help="GPU S_M .npz from s_m_gpu_generate.py. Defaults to latest if found.")
    p.add_argument("--out-dir", default=None, help="Output analysis directory. Defaults to S_M/analysis/s_m_<timestamp>.")

    p.add_argument("--skip-geo", action="store_true")
    p.add_argument("--skip-gpu", action="store_true")
    p.add_argument("--skip-qpu", action="store_true")

    p.add_argument("--windows", nargs="*", default=None, help="Window sizes. Example: --windows 8 16 32 64")
    p.add_argument("--max-windows", type=int, default=256, help="Max windows per record/control/window. 0 = all.")
    p.add_argument("--modes", nargs="*", default=None, help="Control modes. Defaults to all canonical controls.")

    p.add_argument("--sweep", default=None, choices=[None, "SMALL", "MEDIUM", "LARGE", "ALL"], help="Preconfigured window/geo-size sweep.")
    p.add_argument("--probe", action="store_true", help="Run probe-style settings: more windows and all destructive controls.")

    p.add_argument("--geo-shots", type=int, default=DEFAULT_SHOTS)
    p.add_argument("--geo-rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--geo-distances", nargs="*", default=None)
    p.add_argument("--init-state", default="plus")

    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--test-frac", type=float, default=0.30)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--rf-trees", type=int, default=180)
    p.add_argument("--no-sklearn", action="store_true", help="Disable sklearn models even if installed.")
    p.add_argument("--no-cuda", action="store_true", help="Disable S_M/kernels/sm_kernel.cu acceleration and use NumPy features only.")
    p.add_argument("--cuda-threads", type=int, default=256, help="CUDA threads per window block for sm_window_features_kernel.")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--write-windows", action="store_true", help="Write window feature arrays into artifacts/window_rows outputs.")
    p.add_argument("--quiet", action="store_true")

    return p.parse_args()


def sweep_windows(args: argparse.Namespace) -> List[int]:
    explicit = parse_num_list(args.windows, int)
    if explicit:
        return sorted(set(int(x) for x in explicit if int(x) > 0))
    if args.probe:
        return [1, 2, 4, 8, 16, 32, 64, 128]
    if args.sweep == "SMALL":
        return [4, 8, 16, 32]
    if args.sweep == "MEDIUM":
        return [8, 16, 32, 64]
    if args.sweep == "LARGE":
        return [16, 32, 64, 128]
    if args.sweep == "ALL":
        return [4, 8, 16, 32, 64, 128]
    return DEFAULT_WINDOWS


def sweep_geo_config(args: argparse.Namespace) -> Tuple[List[int], int, int]:
    d_arg = parse_num_list(args.geo_distances, int)
    distances = d_arg if d_arg else list(DEFAULT_DISTANCES)
    shots = int(args.geo_shots)
    rounds = int(args.geo_rounds)
    if args.sweep == "SMALL":
        distances = [3, 5]
        shots = min(shots, 1024)
        rounds = min(rounds, 5)
    elif args.sweep == "LARGE":
        shots = max(shots, 8192)
        rounds = max(rounds, 10)
    return distances, shots, rounds


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    t_all = time.time()

    windows = sweep_windows(args)
    modes = parse_num_list(args.modes, str) if args.modes else list(DEFAULT_CONTROL_MODES)
    distances, geo_shots, geo_rounds = sweep_geo_config(args)

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"s_m_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cuda_ctx = compile_sm_cuda_kernel(args)

    if not args.quiet:
        print("\n" + "=" * 100)
        print("  S_M BENCHMARK — FINAL SYNDROME METRIC OPERATOR")
        print("=" * 100)
        print(f"  Data dir     : {DATA_DIR}")
        print(f"  Analysis dir : {out_dir}")
        print(f"  Windows      : {windows}")
        print(f"  Modes        : {modes}")
        print(f"  sklearn      : {'yes' if _HAVE_SKLEARN and not args.no_sklearn else 'no'}")
        print(f"  matplotlib   : {'yes' if _HAVE_MPL and not args.no_plots else 'no'}")
        print(f"  CUDA kernel  : {'yes' if cuda_ctx is not None else 'no'}")
        if cuda_ctx is not None:
            print(f"  Kernel path  : {cuda_ctx.path}")

    substrate_records: Dict[str, List[SMRecord]] = {}
    base_meta: Dict[str, Any] = {}

    if not args.skip_geo:
        if not args.quiet:
            print("\n[GEO] Building synthetic reference S_M field...")
        substrate_records["geo"] = synthetic_geo_records(
            distances=distances,
            shots=geo_shots,
            rounds=geo_rounds,
            seed=args.seed,
            init_state=args.init_state,
        )
        base_meta["geo"] = {
            "source": "synthetic_geo",
            "distances": distances,
            "shots": geo_shots,
            "rounds": geo_rounds,
            "init_state": args.init_state,
        }

    if not args.skip_gpu:
        gpu_path = Path(args.gpu_base) if args.gpu_base else latest_file("gpu")
        if gpu_path and Path(gpu_path).exists():
            if not args.quiet:
                print(f"\n[GPROJ] Loading GPU S_M base: {gpu_path}")
            recs, meta = load_sm_npz(Path(gpu_path), "gproj")
            substrate_records["gproj"] = recs
            base_meta["gproj"] = {"path": str(gpu_path), **meta}
        else:
            if not args.quiet:
                print("\n[GPROJ][skip] No GPU S_M base found. Use --gpu-base or run s_m_gpu_generate.py.")

    if not args.skip_qpu:
        qpu_path = Path(args.qpu_base) if args.qpu_base else latest_file("qpu")
        if qpu_path and Path(qpu_path).exists():
            if not args.quiet:
                print(f"\n[QPROJ] Loading QPU S_M base: {qpu_path}")
            recs, meta = load_sm_npz(Path(qpu_path), "qproj")
            substrate_records["qproj"] = recs
            base_meta["qproj"] = {"path": str(qpu_path), **meta}
        else:
            if not args.quiet:
                print("\n[QPROJ][skip] No QPU S_M base found. Use --qpu-base or run sm_qpu.py dump.")

    if not substrate_records:
        raise RuntimeError("No substrates available. Enable geo or provide GPU/QPU base files.")

    all_rows: List[dict] = []
    per_feature_rows: List[dict] = []
    artifact_arrays: Dict[str, np.ndarray] = {}
    window_row_csv: List[dict] = []

    for substrate, records in substrate_records.items():
        if not args.quiet:
            ds_desc = ", ".join(f"d{r.distance}:{r.data.shape[0]}x{r.synd.shape[1]}" for r in records)
            print(f"\n[{substrate.upper()}] Records: {ds_desc}")

        for window in windows:
            if not args.quiet:
                print(f"  [window={window}] building features...")
            ds_by_feature = build_windowed_dataset(
                records, modes, window, args.max_windows, args.seed,
                cuda_ctx=cuda_ctx, cuda_threads=args.cuda_threads,
            )

            if args.write_windows:
                for feat, ds in ds_by_feature.items():
                    key_prefix = f"{substrate}_w{window}_{feat}"
                    artifact_arrays[f"{key_prefix}_X"] = ds["X"].astype(np.float32)
                    # Keep compact CSV metadata for window rows.
                    for i in range(len(ds["binary"])):
                        window_row_csv.append({
                            "substrate": substrate,
                            "window": int(window),
                            "feature": feat,
                            "row": int(i),
                            "control": str(ds["control"][i]),
                            "binary": str(ds["binary"][i]),
                            "distance": int(ds["distance"][i]),
                            "source": str(ds["source"][i]),
                        })

            rows = run_tasks_for_dataset(ds_by_feature, substrate, window, args, args.seed + 100 * int(window))
            all_rows.extend(rows)

            for r in rows:
                per_feature_rows.append({
                    "substrate": r.get("substrate"),
                    "task": r.get("task"),
                    "feature": r.get("feature"),
                    "window": r.get("window"),
                    "model": r.get("model"),
                    "balanced_accuracy": r.get("balanced_accuracy"),
                    "accuracy": r.get("accuracy"),
                    "auc": r.get("auc", ""),
                    "n_features": r.get("n_features", ""),
                    "n_train": r.get("n_train", ""),
                    "n_test": r.get("n_test", ""),
                    "seconds": r.get("seconds", ""),
                    "error": r.get("error", ""),
                })

    agreement_rows = substrate_agreement(substrate_records)
    collapse = control_collapse_rows(all_rows)
    best = best_rows(all_rows)

    summary_fields = [
        "substrate", "task", "feature", "window", "model", "balanced_accuracy", "accuracy",
        "auc", "n_features", "n_train", "n_test", "seconds", "error",
    ]
    agreement_fields = [
        "substrate_a", "substrate_b",
        "agreement_edge_corr", "agreement_edge_l2",
        "agreement_time_corr", "agreement_time_l2",
        "agreement_field_corr", "agreement_field_l2",
        "detection_edge_corr", "detection_edge_l2",
        "detection_time_corr", "detection_time_l2",
        "detection_field_corr", "detection_field_l2",
    ]
    collapse_fields = [
        "substrate", "feature", "window", "model", "balanced_accuracy", "accuracy", "labels", "n_test",
    ]

    write_csv(out_dir / "summary.csv", best, summary_fields)
    write_csv(out_dir / "per_feature.csv", per_feature_rows, summary_fields)
    write_csv(out_dir / "control_collapse.csv", collapse, collapse_fields)
    write_csv(out_dir / "substrate_agreement.csv", agreement_rows, agreement_fields)

    if args.write_windows:
        write_csv(out_dir / "window_rows.csv", window_row_csv, [
            "substrate", "window", "feature", "row", "control", "binary", "distance", "source"
        ])

    # Always write at least compact artifacts: real substrate profiles.
    for sub, recs in substrate_records.items():
        prof = real_profiles(recs)
        for k, v in prof.items():
            artifact_arrays[f"{sub}_{k}"] = v.astype(np.float32)
    if artifact_arrays:
        np.savez_compressed(out_dir / "artifacts.npz", **artifact_arrays)

    result = {
        "schema": "s_m_benchmark_result",
        "created": now_tag(),
        "seconds": time.time() - t_all,
        "config": {
            "windows": windows,
            "modes": modes,
            "seed": args.seed,
            "test_frac": args.test_frac,
            "knn_k": args.knn_k,
            "max_windows": args.max_windows,
            "cuda_enabled": cuda_ctx is not None,
            "cuda_kernel_path": str(cuda_ctx.path) if cuda_ctx is not None else None,
            "cuda_threads": args.cuda_threads,
            "sweep": args.sweep,
            "probe": args.probe,
            "data_dir": str(DATA_DIR),
            "analysis_dir": str(out_dir),
        },
        "base_meta": base_meta,
        "records": {
            sub: [
                {
                    "source": r.source,
                    "distance": r.distance,
                    "data_shape": list(r.data.shape),
                    "synd_shape": list(r.synd.shape),
                    "flags_shape": list(r.flags.shape) if r.flags is not None else None,
                }
                for r in recs
            ]
            for sub, recs in substrate_records.items()
        },
        "best_rows": best,
        "all_rows": all_rows,
        "substrate_agreement": agreement_rows,
        "control_collapse": collapse,
        "bounded_claim": (
            "S_M is a syndrome-spacetime field operator. This benchmark tests whether "
            "final edge parity and repeated syndrome records form a load-bearing field "
            "structure that survives matched controls and can be compared across geo, "
            "gproj, and qproj substrates."
        ),
        "non_claims": [
            "S_M is not a logical-error-rate benchmark.",
            "S_M is not the T_S stress tensor.",
            "S_M is not a token retrieval benchmark.",
            "S_M is not a universal hardware advantage claim.",
        ],
    }
    write_json(out_dir / "result.json", result)

    if not args.no_plots:
        plot_task_bars(all_rows, out_dir)

    if not args.quiet:
        print_summary(best)
        print("[SAVED]")
        print(f"  result              : {out_dir / 'result.json'}")
        print(f"  summary             : {out_dir / 'summary.csv'}")
        print(f"  per-feature         : {out_dir / 'per_feature.csv'}")
        print(f"  control collapse    : {out_dir / 'control_collapse.csv'}")
        print(f"  substrate agreement : {out_dir / 'substrate_agreement.csv'}")
        print(f"  artifacts           : {out_dir / 'artifacts.npz'}")
        print("\nDone. Break it, fix it, document what happened.\n")


if __name__ == "__main__":
    main()
