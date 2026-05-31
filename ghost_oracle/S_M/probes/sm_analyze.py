#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
S_M STEP 3 — UNIFIED ANALYSIS PIPELINE
==============================================================================
One-button analysis for dumped S_M `.npz` data.

Default behavior
----------------
Uses the latest dumped dataset recorded by:

    data/latest_sm_data.json

and runs the whole S_M analysis stack:

1. shape probe
       Tests whether the object is scalar, edge-vector, time-vector, or
       spacetime field.

2. stress tensor probe
       Builds the syndrome-spacetime stress tensor:

           ΔtS[t,i] = S[t+1,i] XOR S[t,i]
           ΔxS[t,i] = S[t,i+1] XOR S[t,i]

           T_S = [[<ΔtS ΔtS>, <ΔtS ΔxS>],
                  [<ΔxS ΔtS>, <ΔxS ΔxS>]]

3. teaser plot
       Creates a compact edge-stress figure from the shape report.

Outputs
-------
By default writes into:

    analysis/sm_<JOB_ID>/

Files:
    operator_shape_report.json
    stress_tensor_report.json
    sister_operator_teaser.png
    stress_tensor_summary.png
    stress_tensor_anisotropy.png
    stress_tensor_local_*.png

Usage
-----
One-button latest analysis:

    python ghost_oracle/S_M/sm_analyze.py

Specific file:

    python ghost_oracle/S_M/sm_analyze.py --npz data/sm_data_plus_<JOB_ID>.npz

Skip plots:

    python ghost_oracle/S_M/sm_analyze.py --no-plots
==============================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    plt = None
    _HAVE_MPL = False


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
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    return x


def bits(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x).astype(np.int64) & 1).astype(np.uint8)


def resolve_inputs(npz_arg: Optional[str], meta_arg: Optional[str]) -> tuple[Path, Optional[Path], str]:
    if npz_arg is None:
        latest = DATA_DIR / "latest_sm_data.json"
        if not latest.exists():
            raise FileNotFoundError("No --npz provided and data/latest_sm_data.json does not exist.")
        obj = load_json(latest)
        npz = Path(obj["npz"])
        meta = Path(obj["meta"]) if obj.get("meta") else None
        job_id = str(obj.get("job_id", npz.stem))
        return npz, meta, job_id

    npz = Path(npz_arg)
    meta = Path(meta_arg) if meta_arg else None
    stem = npz.stem
    job_id = stem.split("_")[-1] if "_" in stem else stem
    return npz, meta, job_id


def load_records(npz_path: Path) -> tuple[List[dict], dict]:
    z = np.load(npz_path, allow_pickle=False)
    meta = {k: z[k].item() if z[k].shape == () else z[k] for k in z.files if k in ("job_id", "rounds", "init_state", "basis", "flag_level")}
    if "distances" not in z.files:
        raise KeyError("S_M unified analysis expects flag/superposition arrays with distances and data_d*/synd_d*.")
    distances = [int(d) for d in z["distances"]]
    records = []
    for d in distances:
        records.append({
            "d": d,
            "data": bits(z[f"data_d{d}"]),
            "synd": bits(z[f"synd_d{d}"]),
            "flags": bits(z[f"flag_d{d}"]) if f"flag_d{d}" in z.files else None,
        })
    return records, meta


# =============================================================================
# SHAPE PROBE
# =============================================================================

CONTROL_MODES = ["real", "shot_shuffle_synd", "time_shuffle_synd", "edge_shuffle_synd", "uniform_synd", "final_shuffle", "all_uniform"]


def terminal_edge_parity(data: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(data[:, :-1], data[:, 1:]).astype(np.uint8)


def agreement_field(data: np.ndarray, synd: np.ndarray) -> np.ndarray:
    edges = terminal_edge_parity(data)[:, None, :]
    return (1.0 - np.bitwise_xor(edges, synd).astype(np.float32)).astype(np.float32)


def detection_field(synd: np.ndarray) -> np.ndarray:
    if synd.shape[1] < 2:
        return np.zeros((synd.shape[0], 0, synd.shape[2]), dtype=np.float32)
    return np.bitwise_xor(synd[:, 1:, :], synd[:, :-1, :]).astype(np.float32)


def mutate(data: np.ndarray, synd: np.ndarray, mode: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    d = data.copy()
    s = synd.copy()
    if mode == "real":
        pass
    elif mode == "shot_shuffle_synd":
        s = s[rng.permutation(s.shape[0])]
    elif mode == "time_shuffle_synd":
        for sh in range(s.shape[0]):
            for e in range(s.shape[2]):
                s[sh, :, e] = s[sh, rng.permutation(s.shape[1]), e]
    elif mode == "edge_shuffle_synd":
        for sh in range(s.shape[0]):
            for t in range(s.shape[1]):
                s[sh, t, :] = s[sh, t, rng.permutation(s.shape[2])]
    elif mode == "uniform_synd":
        p = s.mean(axis=(0, 1), keepdims=True)
        s = (rng.random(s.shape) < p).astype(np.uint8)
    elif mode == "final_shuffle":
        d = d[rng.permutation(d.shape[0])]
    elif mode == "all_uniform":
        pd = d.mean(axis=0, keepdims=True)
        ps = s.mean(axis=(0, 1), keepdims=True)
        d = (rng.random(d.shape) < pd).astype(np.uint8)
        s = (rng.random(s.shape) < ps).astype(np.uint8)
    else:
        raise ValueError(mode)
    return d, s


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def summarize_repr(A: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
    return {
        "scalar": float(A.mean()),
        "edge_vec": A.mean(axis=(0, 1)),
        "time_vec": A.mean(axis=(0, 2)),
        "field": A.mean(axis=0),
        "det_scalar": float(X.mean()) if X.size else 0.0,
        "det_edge_vec": X.mean(axis=(0, 1)) if X.size else np.zeros(A.shape[2]),
        "det_time_vec": X.mean(axis=(0, 2)) if X.size else np.zeros(max(A.shape[1] - 1, 0)),
        "det_field": X.mean(axis=0) if X.size else np.zeros((max(A.shape[1] - 1, 0), A.shape[2])),
    }


def cv(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.std(x) / (abs(np.mean(x)) + 1e-12))


def classify(row: Dict[str, float]) -> str:
    scalar = row["best_scalar_gap"]
    edge = row["best_edge_l2"]
    time = row["best_time_l2"]
    field = row["best_field_l2"]
    edge_cv = row["edge_cv"]
    time_cv = row["time_cv"]
    if field >= 4.0 * max(scalar, 1e-12):
        if edge_cv > 0.010:
            return "field / edge-anisotropic"
        if time_cv > 0.005:
            return "field / time-anisotropic"
        return "field / smooth-distributed"
    if edge >= 3.0 * max(scalar, 1e-12) and edge >= time:
        return "edge-vector"
    if time >= 3.0 * max(scalar, 1e-12) and time > edge:
        return "time-vector"
    if scalar > 0 and field < 2.0 * scalar:
        return "scalar-like"
    return "mixed/undecided"


def run_shape(records: List[dict], seed: int) -> dict:
    rows = []
    edge_profiles = {}
    for rec in records:
        rng = np.random.default_rng(seed + rec["d"])
        reps = {}
        for mode in CONTROL_MODES:
            d, s = mutate(rec["data"], rec["synd"], mode, rng)
            reps[mode] = summarize_repr(agreement_field(d, s), detection_field(s))

        real = reps["real"]
        gaps = {}
        for mode, rep in reps.items():
            if mode == "real":
                continue
            gaps[mode] = {
                "scalar_gap": abs(real["scalar"] - rep["scalar"]),
                "edge_l2": l2(real["edge_vec"], rep["edge_vec"]),
                "time_l2": l2(real["time_vec"], rep["time_vec"]),
                "field_l2": l2(real["field"], rep["field"]),
                "det_scalar_gap": abs(real["det_scalar"] - rep["det_scalar"]),
                "det_edge_l2": l2(real["det_edge_vec"], rep["det_edge_vec"]),
                "det_time_l2": l2(real["det_time_vec"], rep["det_time_vec"]),
                "det_field_l2": l2(real["det_field"], rep["det_field"]),
            }

        edge = np.asarray(real["edge_vec"])
        time = np.asarray(real["time_vec"])
        field = np.asarray(real["field"])
        row = {
            "d": rec["d"],
            "best_scalar_gap": max(g["scalar_gap"] for g in gaps.values()),
            "best_edge_l2": max(g["edge_l2"] for g in gaps.values()),
            "best_time_l2": max(g["time_l2"] for g in gaps.values()),
            "best_field_l2": max(g["field_l2"] for g in gaps.values()),
            "best_det_scalar_gap": max(g["det_scalar_gap"] for g in gaps.values()),
            "best_det_edge_l2": max(g["det_edge_l2"] for g in gaps.values()),
            "best_det_time_l2": max(g["det_time_l2"] for g in gaps.values()),
            "best_det_field_l2": max(g["det_field_l2"] for g in gaps.values()),
            "edge_cv": cv(edge),
            "time_cv": cv(time),
            "field_cv": cv(field.ravel()),
            "edge_range": float(edge.max() - edge.min()),
            "time_range": float(time.max() - time.min()),
            "field_range": float(field.max() - field.min()),
        }
        row["shape_guess"] = classify(row)
        rows.append(row)
        edge_profiles[str(rec["d"])] = {
            "real_edge_vec": real["edge_vec"].tolist(),
            "real_time_vec": real["time_vec"].tolist(),
            "real_field": real["field"].tolist(),
            "control_gaps": gaps,
        }
    return {"rows": rows, "edge_profiles": edge_profiles}


# =============================================================================
# STRESS TENSOR
# =============================================================================

STRESS_CONTROLS = ["real", "shot_shuffle_synd", "time_shuffle_synd", "edge_shuffle_synd", "uniform_synd", "all_uniform", "time_reverse_synd", "edge_reverse_synd"]


def mutate_synd(synd: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    s = synd.copy()
    if mode == "real":
        return s
    if mode == "shot_shuffle_synd":
        return s[rng.permutation(s.shape[0])]
    if mode == "time_shuffle_synd":
        for sh in range(s.shape[0]):
            for e in range(s.shape[2]):
                s[sh, :, e] = s[sh, rng.permutation(s.shape[1]), e]
        return s
    if mode == "edge_shuffle_synd":
        for sh in range(s.shape[0]):
            for t in range(s.shape[1]):
                s[sh, t, :] = s[sh, t, rng.permutation(s.shape[2])]
        return s
    if mode == "uniform_synd":
        p = s.mean(axis=(0, 1), keepdims=True)
        return (rng.random(s.shape) < p).astype(np.uint8)
    if mode == "all_uniform":
        p = float(s.mean())
        return (rng.random(s.shape) < p).astype(np.uint8)
    if mode == "time_reverse_synd":
        return s[:, ::-1, :].copy()
    if mode == "edge_reverse_synd":
        return s[:, :, ::-1].copy()
    raise ValueError(mode)


def gradients(synd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if synd.shape[1] < 2 or synd.shape[2] < 2:
        empty = np.zeros((synd.shape[0], 0, 0), dtype=np.float32)
        return empty, empty
    dt_full = np.bitwise_xor(synd[:, 1:, :], synd[:, :-1, :]).astype(np.float32)
    dx_full = np.bitwise_xor(synd[:, :, 1:], synd[:, :, :-1]).astype(np.float32)
    return dt_full[:, :, :-1], dx_full[:, :-1, :]


def stress_from(dt: np.ndarray, dx: np.ndarray) -> Dict[str, Any]:
    if dt.size == 0:
        return {"T": np.zeros((2, 2)), "Ttt": 0.0, "Txx": 0.0, "Ttx": 0.0, "trace": 0.0, "anisotropy": 0.0, "coupling_ratio": 0.0, "eigvals": [0.0, 0.0], "local_trace": np.zeros((0, 0)), "local_anisotropy": np.zeros((0, 0)), "local_Ttx": np.zeros((0, 0))}
    local_Ttt = np.mean(dt * dt, axis=0)
    local_Txx = np.mean(dx * dx, axis=0)
    local_Ttx = np.mean(dt * dx, axis=0)
    Ttt = float(local_Ttt.mean())
    Txx = float(local_Txx.mean())
    Ttx = float(local_Ttx.mean())
    T = np.array([[Ttt, Ttx], [Ttx, Txx]], dtype=float)
    trace = float(np.trace(T))
    anis = float((Ttt - Txx) / (Ttt + Txx + 1e-12))
    coupling = float(Ttx / np.sqrt(max(Ttt * Txx, 1e-24)))
    return {
        "T": T,
        "Ttt": Ttt,
        "Txx": Txx,
        "Ttx": Ttx,
        "trace": trace,
        "anisotropy": anis,
        "coupling_ratio": coupling,
        "eigvals": np.linalg.eigvalsh(T).tolist(),
        "local_trace": (local_Ttt + local_Txx),
        "local_anisotropy": (local_Ttt - local_Txx) / (local_Ttt + local_Txx + 1e-12),
        "local_Ttx": local_Ttx,
    }


def tensor_gap(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    return {
        "fro_gap": float(np.linalg.norm(np.asarray(a["T"]) - np.asarray(b["T"]))),
        "trace_gap": abs(float(a["trace"]) - float(b["trace"])),
        "anisotropy_gap": abs(float(a["anisotropy"]) - float(b["anisotropy"])),
        "coupling_gap": abs(float(a["coupling_ratio"]) - float(b["coupling_ratio"])),
        "local_trace_l2": l2(a["local_trace"], b["local_trace"]),
    }


def run_stress(records: List[dict], seed: int) -> dict:
    results = []
    for rec in records:
        rng = np.random.default_rng(seed + rec["d"] * 1009)
        tensors = {}
        for mode in STRESS_CONTROLS:
            s = mutate_synd(rec["synd"], mode, rng)
            dt, dx = gradients(s)
            tensors[mode] = stress_from(dt, dx)
        real = tensors["real"]
        gaps = {m: tensor_gap(real, t) for m, t in tensors.items() if m != "real"}
        best = {
            "best_fro_gap": max(g["fro_gap"] for g in gaps.values()),
            "best_trace_gap": max(g["trace_gap"] for g in gaps.values()),
            "best_anisotropy_gap": max(g["anisotropy_gap"] for g in gaps.values()),
            "best_coupling_gap": max(g["coupling_gap"] for g in gaps.values()),
            "best_local_trace_l2": max(g["local_trace_l2"] for g in gaps.values()),
        }
        results.append({
            "distance": rec["d"],
            "shots": int(rec["synd"].shape[0]),
            "rounds": int(rec["synd"].shape[1]),
            "edges": int(rec["synd"].shape[2]),
            "real": real,
            "gaps": gaps,
            "best": best,
        })
    return {"results": results}


# =============================================================================
# PLOTS
# =============================================================================

def plot_teaser(shape_report: dict, out_path: Path) -> None:
    if not _HAVE_MPL:
        return
    distances = sorted(int(d) for d in shape_report["edge_profiles"].keys())
    rows = {int(r["d"]): r for r in shape_report["rows"]}
    max_edges = max(len(shape_report["edge_profiles"][str(d)]["real_edge_vec"]) for d in distances)
    agreement = np.full((len(distances), max_edges), np.nan)
    for r, d in enumerate(distances):
        vals = shape_report["edge_profiles"][str(d)]["real_edge_vec"]
        agreement[r, :len(vals)] = vals
    stress = 1.0 - agreement
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=160)
    im = ax.imshow(stress, aspect="auto")
    ax.set_title("S_M syndrome-consistency field: real QPU edge profiles", fontsize=16, pad=16)
    ax.set_xlabel("code edge / stabilizer index")
    ax.set_ylabel("repetition-code distance")
    ax.set_xticks(np.arange(max_edges))
    ax.set_xticklabels([f"e{i}" for i in range(max_edges)])
    ax.set_yticks(np.arange(len(distances)))
    ax.set_yticklabels([f"d={d}" for d in distances])
    for r, d in enumerate(distances):
        vals = shape_report["edge_profiles"][str(d)]["real_edge_vec"]
        for c, val in enumerate(vals):
            ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=9)
        row = rows[d]
        ax.text(max_edges + 0.25, r, f"{row['shape_guess']}\nfield L2={row['best_field_l2']:.4f}  det-field L2={row['best_det_field_l2']:.4f}", va="center", fontsize=9)
    ax.set_xlim(-0.5, max_edges + 4.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("edge stress = 1 − agreement")
    fig.text(0.5, 0.035, "Real QPU syndrome/data pairing separates from shuffled controls.", ha="center", fontsize=10)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_stress(stress_report: dict, out_dir: Path) -> None:
    if not _HAVE_MPL:
        return
    results = stress_report["results"]
    d = np.array([r["distance"] for r in results])
    Ttt = np.array([r["real"]["Ttt"] for r in results])
    Txx = np.array([r["real"]["Txx"] for r in results])
    Ttx = np.array([r["real"]["Ttx"] for r in results])
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.plot(d, Ttt, marker="o", label="Ttt temporal")
    ax.plot(d, Txx, marker="o", label="Txx spatial")
    ax.plot(d, Ttx, marker="o", label="Ttx coupling")
    ax.set_title("S_M syndrome-spacetime stress tensor")
    ax.set_xlabel("distance")
    ax.set_ylabel("tensor component")
    ax.set_xticks(d)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "stress_tensor_summary.png", bbox_inches="tight")
    plt.close(fig)


def print_summary(shape_report: dict, stress_report: dict) -> None:
    print("\n" + "=" * 112)
    print("  S_M SHAPE SUMMARY")
    print("=" * 112)
    print(f"  {'d':>3} | {'field L2':>9} | {'det-field L2':>12} | {'edge CV':>8} | {'time CV':>8} | shape")
    print("  " + "-" * 110)
    for r in shape_report["rows"]:
        print(f"  {r['d']:>3} | {r['best_field_l2']:>9.4f} | {r['best_det_field_l2']:>12.4f} | {r['edge_cv']:>8.4f} | {r['time_cv']:>8.4f} | {r['shape_guess']}")
    print("\n" + "=" * 112)
    print("  S_M STRESS TENSOR SUMMARY")
    print("=" * 112)
    print(f"  {'d':>3} | {'Ttt':>9} | {'Txx':>9} | {'Ttx':>9} | {'trace':>9} | {'anis':>9} | {'coupling':>9} | {'best local':>10}")
    print("  " + "-" * 110)
    for r in stress_report["results"]:
        real = r["real"]
        best = r["best"]
        print(f"  {r['distance']:>3} | {real['Ttt']:>9.5f} | {real['Txx']:>9.5f} | {real['Ttx']:>9.5f} | {real['trace']:>9.5f} | {real['anisotropy']:>9.4f} | {real['coupling_ratio']:>9.4f} | {best['best_local_trace_l2']:>10.5f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S_M step 3 — run unified analysis on dumped data.")
    p.add_argument("--npz", default=None, help="Dumped S_M .npz. Defaults to latest dumped dataset.")
    p.add_argument("--meta", default=None, help="Optional metadata path.")
    p.add_argument("--out-dir", default=None, help="Output directory. Defaults to analysis/sm_<JOB_ID>.")
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path, meta_path, job_id = resolve_inputs(args.npz, args.meta)
    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"sm_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 112}")
    print("  S_M STEP 3 — UNIFIED ANALYSIS PIPELINE")
    print(f"{'=' * 112}")
    print(f"  NPZ     : {npz_path}")
    print(f"  Metadata: {meta_path if meta_path else '(not provided)'}")
    print(f"  Out dir : {out_dir}")

    records, npz_meta = load_records(npz_path)
    shape_report = run_shape(records, args.seed)
    stress_report = run_stress(records, args.seed)

    with open(out_dir / "operator_shape_report.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(shape_report), f, indent=2)
    with open(out_dir / "stress_tensor_report.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(stress_report), f, indent=2)

    if not args.no_plots:
        plot_teaser(shape_report, out_dir / "sister_operator_teaser.png")
        plot_stress(stress_report, out_dir)

    print_summary(shape_report, stress_report)

    print(f"\n[SAVED] {out_dir}")
    print(f"{'=' * 112}\n")


if __name__ == "__main__":
    main()
