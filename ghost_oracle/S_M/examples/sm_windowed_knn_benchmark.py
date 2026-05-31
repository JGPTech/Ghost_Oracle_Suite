#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
S_M WINDOWED KNN BENCHMARK — FIELD-LEVEL FEATURE PROBE
==============================================================================
Purpose
-------
The first kNN benchmark classified individual shots. That is useful, but S_M is
not really a single-shot object. It is a syndrome-spacetime field/statistical
operator.

This probe aggregates shots into windows, computes S_M features per window, and
then runs the same style of classification tests.

Main question
-------------
Do S_M features become stronger when treated at the field/window level?

Tasks
-----
TASK A: real-vs-control classification
    label = real or control

TASK B: control-source classification
    label = real, shot_shuffle_synd, time_shuffle_synd, edge_shuffle_synd, ...

TASK C: distance prediction
    label = d in {3, 5, 7, 9}

Feature families
----------------
raw_rates
    Mean syndrome rates and final data rates over the window.

detection_rates
    Mean detection-event rates over the window.

agreement_profiles
    Edge profile + time profile of terminal edge/syndrome agreement.

sm_tensor
    Window-level stress tensor:
        Ttt, Txx, Ttx, trace, anisotropy, coupling

sm_local_trace
    Flattened local trace field:
        <ΔtS^2 + ΔxS^2> per spacetime cell

sm_all
    Combined S_M feature set.

Why windowed?
-------------
The S_M stress tensor uses expectations over syndrome gradients. A single shot
is only one sample from the field. A window of 16/32/64 shots gives the operator
room to express the structure we actually care about.

Usage
-----
Default latest data:

    python ghost_oracle/S_M/sm_windowed_knn_benchmark.py

Specific data:

    python ghost_oracle/S_M/sm_windowed_knn_benchmark.py --npz data/sm_data_plus_<JOB_ID>.npz

Try several window sizes:

    python ghost_oracle/S_M/sm_windowed_knn_benchmark.py --windows 8 16 32 64 128

Fast run:

    python ghost_oracle/S_M/sm_windowed_knn_benchmark.py --windows 32 --max-windows 128

Outputs
-------
    analysis/sm_windowed_<JOB_ID>/
        sm_windowed_results.json
        sm_windowed_results.csv
        windowed_task_A_accuracy.png
        windowed_task_B_accuracy.png
        windowed_task_C_accuracy.png
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    _HAVE_SKLEARN = True
except Exception:
    LogisticRegression = None
    RandomForestClassifier = None
    _HAVE_SKLEARN = False


CONTROL_MODES = [
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


# =============================================================================
# PATHS / IO
# =============================================================================

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "requirements.txt").exists() or (p / ".git").exists():
            return p
    parents = cur.parents
    return parents[2] if len(parents) >= 3 else cur


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
ANALYSIS_DIR = HERE / "analysis"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def bits(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x).astype(np.int64) & 1).astype(np.uint8)


def resolve_npz(npz_arg: Optional[str]) -> Tuple[Path, str]:
    if npz_arg:
        p = Path(npz_arg)
        job_id = p.stem.split("_")[-1] if "_" in p.stem else p.stem
        return p, job_id

    latest = DATA_DIR / "latest_sm_data.json"
    if not latest.exists():
        raise FileNotFoundError(
            "No --npz provided and data/latest_sm_data.json does not exist. "
            "Run sm_dump.py first or pass --npz explicitly."
        )
    obj = load_json(latest)
    return Path(obj["npz"]), str(obj.get("job_id", Path(obj["npz"]).stem))


def load_records(npz_path: Path) -> Tuple[List[dict], dict]:
    z = np.load(npz_path, allow_pickle=False)
    if "distances" not in z.files:
        raise KeyError("Expected S_M .npz with distances and data_d*/synd_d* arrays.")

    meta = {
        k: z[k].item() if z[k].shape == () else z[k]
        for k in z.files
        if k in ("job_id", "backend", "rounds", "init_state", "basis", "flag_level", "shots")
    }

    records = []
    for d in [int(x) for x in z["distances"]]:
        records.append({
            "d": d,
            "data": bits(z[f"data_d{d}"]),
            "synd": bits(z[f"synd_d{d}"]),
        })
    return records, meta


# =============================================================================
# S_M FIELD OPS
# =============================================================================

def terminal_edge_parity(data: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(data[:, :-1], data[:, 1:]).astype(np.uint8)


def detection_events(synd: np.ndarray) -> np.ndarray:
    if synd.shape[1] < 2:
        return np.zeros((synd.shape[0], 0, synd.shape[2]), dtype=np.float32)
    return np.bitwise_xor(synd[:, 1:, :], synd[:, :-1, :]).astype(np.float32)


def agreement_field(data: np.ndarray, synd: np.ndarray) -> np.ndarray:
    edges = terminal_edge_parity(data)[:, None, :]
    return (1.0 - np.bitwise_xor(edges, synd).astype(np.float32)).astype(np.float32)


def gradients(synd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if synd.shape[1] < 2 or synd.shape[2] < 2:
        empty = np.zeros((synd.shape[0], 0, 0), dtype=np.float32)
        return empty, empty
    dt_full = np.bitwise_xor(synd[:, 1:, :], synd[:, :-1, :]).astype(np.float32)
    dx_full = np.bitwise_xor(synd[:, :, 1:], synd[:, :, :-1]).astype(np.float32)
    return dt_full[:, :, :-1], dx_full[:, :-1, :]


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
        d = (rng.random(d.shape) < pd).astype(np.uint8)
        s = (rng.random(s.shape) < ps).astype(np.uint8)
        return d, s
    if mode == "time_reverse_synd":
        return d, s[:, ::-1, :].copy()
    if mode == "edge_reverse_synd":
        return d, s[:, :, ::-1].copy()
    raise ValueError(mode)


def window_indices(n: int, window: int, max_windows: int, rng: np.random.Generator) -> List[np.ndarray]:
    starts = list(range(0, n - window + 1, window))
    if not starts:
        return []

    if max_windows > 0 and len(starts) > max_windows:
        starts = sorted(rng.choice(starts, size=max_windows, replace=False).tolist())

    return [np.arange(s, s + window) for s in starts]


def pad_to_width(X: np.ndarray, width: int) -> np.ndarray:
    if X.shape[1] == width:
        return X
    out = np.zeros((X.shape[0], width), dtype=np.float32)
    out[:, :X.shape[1]] = X
    return out


def tensor_from_window(synd_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dt, dx = gradients(synd_w)
    if dt.size == 0:
        return np.zeros(6, dtype=np.float32), np.zeros(0, dtype=np.float32)

    local_Ttt = np.mean(dt * dt, axis=0)
    local_Txx = np.mean(dx * dx, axis=0)
    local_Ttx = np.mean(dt * dx, axis=0)

    Ttt = float(local_Ttt.mean())
    Txx = float(local_Txx.mean())
    Ttx = float(local_Ttx.mean())
    trace = Ttt + Txx
    anis = (Ttt - Txx) / (Ttt + Txx + 1e-12)
    coupling = Ttx / math.sqrt(max(Ttt * Txx, 1e-12))

    tensor = np.asarray([Ttt, Txx, Ttx, trace, anis, coupling], dtype=np.float32)
    local_trace = (local_Ttt + local_Txx).reshape(-1).astype(np.float32)
    return tensor, local_trace


def window_features(data_w: np.ndarray, synd_w: np.ndarray) -> Dict[str, np.ndarray]:
    A = agreement_field(data_w, synd_w)
    X = detection_events(synd_w)
    tensor, local_trace = tensor_from_window(synd_w)

    raw_rates = np.concatenate([
        data_w.mean(axis=0).reshape(-1),
        synd_w.mean(axis=0).reshape(-1),
    ]).astype(np.float32)

    det_rates = X.mean(axis=0).reshape(-1).astype(np.float32) if X.size else np.zeros(0, dtype=np.float32)

    edge_profile = A.mean(axis=(0, 1)).reshape(-1).astype(np.float32)
    time_profile = A.mean(axis=(0, 2)).reshape(-1).astype(np.float32)
    agreement_profiles = np.concatenate([edge_profile, time_profile]).astype(np.float32)

    sm_all = np.concatenate([agreement_profiles, det_rates, tensor, local_trace]).astype(np.float32)

    return {
        "raw_rates": raw_rates,
        "detection_rates": det_rates,
        "agreement_profiles": agreement_profiles,
        "sm_tensor": tensor,
        "sm_local_trace": local_trace,
        "sm_all": sm_all,
    }


# =============================================================================
# DATASET
# =============================================================================

def build_windowed_dataset(
    records: List[dict],
    modes: Sequence[str],
    window: int,
    max_windows: int,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    blocks: Dict[str, Dict[str, List[Any]]] = {}

    for rec in records:
        dist = int(rec["d"])
        for mi, mode in enumerate(modes):
            rng = np.random.default_rng(seed + 1000 * dist + 19 * mi + window)
            data_m, synd_m = mutate_record(rec["data"], rec["synd"], mode, rng)

            idxs = window_indices(data_m.shape[0], window, max_windows, rng)
            for idx in idxs:
                feats = window_features(data_m[idx], synd_m[idx])
                for name, vec in feats.items():
                    if name not in blocks:
                        blocks[name] = {"X_blocks": [], "source": [], "binary": [], "distance": []}
                    blocks[name]["X_blocks"].append(vec.reshape(1, -1))
                    blocks[name]["source"].append(mode)
                    blocks[name]["binary"].append("real" if mode == "real" else "control")
                    blocks[name]["distance"].append(dist)

    out: Dict[str, Dict[str, Any]] = {}
    for name, obj in blocks.items():
        width = max(x.shape[1] for x in obj["X_blocks"])
        X = np.vstack([pad_to_width(x.astype(np.float32), width) for x in obj["X_blocks"]])
        out[name] = {
            "X": X,
            "source": np.asarray(obj["source"], dtype=object),
            "binary": np.asarray(obj["binary"], dtype=object),
            "distance": np.asarray(obj["distance"], dtype=np.int64),
        }

    return out


# =============================================================================
# MODELING
# =============================================================================

def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    labels = sorted(str(v) for v in set(y.tolist()))
    index = {lab: i for i, lab in enumerate(labels)}
    return np.asarray([index[str(v)] for v in y], dtype=np.int64), labels


def stratified_split(y: np.ndarray, test_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = []
    test_idx = []
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac))) if len(idx) > 1 else 0
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True)
    sig[sig < 1e-6] = 1.0
    return mu.astype(np.float32), sig.astype(np.float32)


def standardize_apply(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return ((X - mu) / sig).astype(np.float32)


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


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for lab in np.unique(y_true):
        mask = y_true == lab
        vals.append(float(np.mean(y_pred[mask] == lab)))
    return float(np.mean(vals)) if vals else float("nan")


def eval_pred(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "labels": labels,
        "confusion_matrix": cm,
    }


def run_models(X: np.ndarray, y_raw: np.ndarray, task: str, feature: str, args: argparse.Namespace, seed: int) -> List[Dict[str, Any]]:
    y, labels = encode_labels(y_raw)
    rng = np.random.default_rng(seed)
    tr, te = stratified_split(y, args.test_frac, rng)

    Xtr_raw = X[tr]
    Xte_raw = X[te]
    ytr = y[tr]
    yte = y[te]

    mu, sig = standardize_fit(Xtr_raw)
    Xtr = standardize_apply(Xtr_raw, mu, sig)
    Xte = standardize_apply(Xte_raw, mu, sig)

    rows = []
    for metric in ("euclidean", "cosine"):
        yp = knn_predict(Xtr, ytr, Xte, args.knn_k, metric)
        rows.append({
            "task": task,
            "feature": feature,
            "model": f"kNN-{metric}",
            "n_features": int(X.shape[1]),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            **eval_pred(yte, yp, labels),
        })

    if _HAVE_SKLEARN and not args.no_sklearn:
        try:
            lr = LogisticRegression(max_iter=500, solver="lbfgs")
            lr.fit(Xtr, ytr)
            yp = lr.predict(Xte)
            rows.append({
                "task": task,
                "feature": feature,
                "model": "logistic",
                "n_features": int(X.shape[1]),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                **eval_pred(yte, yp, labels),
            })
        except Exception as e:
            rows.append({
                "task": task,
                "feature": feature,
                "model": "logistic",
                "error": str(e),
                "n_features": int(X.shape[1]),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
                "labels": labels,
                "confusion_matrix": np.zeros((len(labels), len(labels)), dtype=np.int64),
            })

        try:
            rf = RandomForestClassifier(n_estimators=160, random_state=seed, n_jobs=-1)
            rf.fit(Xtr_raw, ytr)
            yp = rf.predict(Xte_raw)
            rows.append({
                "task": task,
                "feature": feature,
                "model": "random_forest",
                "n_features": int(X.shape[1]),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                **eval_pred(yte, yp, labels),
            })
        except Exception as e:
            rows.append({
                "task": task,
                "feature": feature,
                "model": "random_forest",
                "error": str(e),
                "n_features": int(X.shape[1]),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
                "labels": labels,
                "confusion_matrix": np.zeros((len(labels), len(labels)), dtype=np.int64),
            })

    return rows


# =============================================================================
# REPORTING
# =============================================================================

def print_table(rows: List[Dict[str, Any]], task: str, window: int) -> None:
    subset = [
        r for r in rows
        if r["task"] == task and int(r["window"]) == int(window)
        and not math.isnan(float(r.get("balanced_accuracy", float("nan"))))
    ]
    subset.sort(key=lambda r: (float(r["balanced_accuracy"]), float(r["accuracy"])), reverse=True)

    print("\n" + "=" * 116)
    print(f"  window={window}  {task}")
    print("=" * 116)
    print(f"  {'rank':>4} | {'feature':<20} | {'model':<14} | {'acc':>8} | {'bal acc':>8} | {'nfeat':>7} | {'train':>6} | {'test':>6}")
    print("  " + "-" * 114)
    for i, r in enumerate(subset[:10], 1):
        print(
            f"  {i:>4} | {r['feature']:<20} | {r['model']:<14} | "
            f"{float(r['accuracy']):>8.3f} | {float(r['balanced_accuracy']):>8.3f} | "
            f"{int(r['n_features']):>7} | {int(r['n_train']):>6} | {int(r['n_test']):>6}"
        )


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = ["window", "task", "feature", "model", "accuracy", "balanced_accuracy", "n_features", "n_train", "n_test"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def plot_by_window(rows: List[Dict[str, Any]], task: str, out: Path) -> None:
    if not _HAVE_MPL:
        return

    subset = [r for r in rows if r["task"] == task and r["model"].startswith("kNN")]
    if not subset:
        return

    windows = sorted(set(int(r["window"]) for r in subset))
    features = ["raw_rates", "detection_rates", "agreement_profiles", "sm_tensor", "sm_local_trace", "sm_all"]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    for feat in features:
        vals = []
        xs = []
        for w in windows:
            cand = [r for r in subset if int(r["window"]) == w and r["feature"] == feat]
            if not cand:
                continue
            best = max(cand, key=lambda r: r["balanced_accuracy"])
            xs.append(w)
            vals.append(best["balanced_accuracy"])
        if xs:
            ax.plot(xs, vals, marker="o", label=feat)

    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("shot window size")
    ax.set_ylabel("best kNN balanced accuracy")
    ax.set_title(task)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S_M windowed field-level benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz", default=None, help="S_M dumped .npz. Defaults to latest_sm_data.json.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--windows", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--max-windows", type=int, default=160, help="Max windows per distance/control pair. <=0 uses all.")
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--controls", nargs="+", default=CONTROL_MODES)
    p.add_argument("--no-sklearn", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path, job_id = resolve_npz(args.npz)
    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"sm_windowed_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = list(dict.fromkeys(args.controls))
    if "real" not in modes:
        modes.insert(0, "real")

    bad = [m for m in modes if m not in CONTROL_MODES]
    if bad:
        raise ValueError(f"unknown controls: {bad}; valid={CONTROL_MODES}")

    print(f"\n{'=' * 116}")
    print("  S_M WINDOWED KNN BENCHMARK — FIELD-LEVEL FEATURE PROBE")
    print(f"{'=' * 116}")
    print(f"  NPZ         : {npz_path}")
    print(f"  Out dir     : {out_dir}")
    print(f"  Windows     : {args.windows}")
    print(f"  Max windows : {args.max_windows}")
    print(f"  Controls    : {modes}")
    print(f"  sklearn     : {'yes' if (_HAVE_SKLEARN and not args.no_sklearn) else 'no'}")

    records, meta = load_records(npz_path)
    all_rows: List[Dict[str, Any]] = []

    for window in args.windows:
        print(f"\n[BUILD] window={window}")
        datasets = build_windowed_dataset(records, modes, window, args.max_windows, args.seed)
        for name, obj in datasets.items():
            print(f"  {name:<20} X={obj['X'].shape}")

        # A/B use all modes.
        for feat, obj in datasets.items():
            for r in run_models(obj["X"], obj["binary"], "A_real_vs_control", feat, args, args.seed + window + 11):
                r["window"] = int(window)
                all_rows.append(r)
            for r in run_models(obj["X"], obj["source"], "B_source_multiclass", feat, args, args.seed + window + 22):
                r["window"] = int(window)
                all_rows.append(r)

        # C uses only real windows.
        real_modes = ["real"]
        real_datasets = build_windowed_dataset(records, real_modes, window, args.max_windows, args.seed + 303)
        for feat, obj in real_datasets.items():
            for r in run_models(obj["X"], obj["distance"].astype(str), "C_distance_prediction", feat, args, args.seed + window + 33):
                r["window"] = int(window)
                all_rows.append(r)

        print_table(all_rows, "A_real_vs_control", window)
        print_table(all_rows, "B_source_multiclass", window)
        print_table(all_rows, "C_distance_prediction", window)

    payload = {
        "npz": str(npz_path),
        "job_id": job_id,
        "windows": args.windows,
        "max_windows": args.max_windows,
        "controls": modes,
        "results": all_rows,
        "meta": json_safe(meta),
    }

    with open(out_dir / "sm_windowed_results.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)
    write_csv(all_rows, out_dir / "sm_windowed_results.csv")

    if not args.no_plots:
        plot_by_window(all_rows, "A_real_vs_control", out_dir / "windowed_task_A_accuracy.png")
        plot_by_window(all_rows, "B_source_multiclass", out_dir / "windowed_task_B_accuracy.png")
        plot_by_window(all_rows, "C_distance_prediction", out_dir / "windowed_task_C_accuracy.png")

    print(f"\n[SAVED] {out_dir}")
    print(f"{'=' * 116}\n")


if __name__ == "__main__":
    main()
