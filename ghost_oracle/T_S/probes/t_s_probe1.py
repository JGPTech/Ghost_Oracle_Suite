#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
T_S STEP 2 — TEMPORAL STRESS METRIC ANALYSIS
==============================================================================
One-button analysis for dumped T_S `.npz` data.

Default behavior
----------------
Uses the latest dumped dataset recorded by:

    ghost_oracle/T_S/data/latest_ts_data.json

and runs the T_S analysis stack:

1. field shape probe
       Tests whether the measured channel object behaves like scalar noise,
       delay-vector, round-vector, edge-vector, or full delay-round-edge field.

2. delay-aware stress tensor probe
       Builds the Temporal Stress Metric tensor:

           DτF = F[τ+1,r,x] XOR F[τ,r,x]
           DrF = F[τ,r+1,x] XOR F[τ,r,x]
           DxF = F[τ,r,x+1] XOR F[τ,r,x]

           T_ab = <D_aF D_bF>,    a,b ∈ {τ,r,x}

3. matched controls
       real
       shot shuffle
       delay shuffle
       round shuffle
       edge shuffle
       uniform per delay/round/edge
       all-uniform
       delay reverse
       round reverse
       edge reverse

4. summary plots
       stress component summary
       delay survival curves
       local trace maps

Outputs
-------
By default writes into:

    ghost_oracle/T_S/analysis/ts_<JOB_ID>/

Files:
    temporal_shape_report.json
    temporal_stress_report.json
    temporal_stress_summary.png
    delay_survival_summary.png
    local_trace_*.png

Usage
-----
One-button latest analysis:

    python ghost_oracle/T_S/t_s_analyze.py

Specific file:

    python ghost_oracle/T_S/t_s_analyze.py --npz ghost_oracle/T_S/data/ts_data_<JOB_ID>.npz

Skip plots:

    python ghost_oracle/T_S/t_s_analyze.py --no-plots
==============================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    plt = None
    _HAVE_MPL = False


# =============================================================================
# PATHS / IO
# =============================================================================

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
        latest = DATA_DIR / "latest_ts_data.json"
        if not latest.exists():
            raise FileNotFoundError("No --npz provided and ghost_oracle/T_S/data/latest_ts_data.json does not exist.")
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


def load_ts(npz_path: Path) -> Dict[str, Any]:
    z = np.load(npz_path, allow_pickle=False)
    if "field" not in z.files:
        raise KeyError("T_S analysis expects a `field` array.")

    field = bits(z["field"])
    if field.ndim != 6:
        raise ValueError(
            "Expected field shape (modes, delay_sites, delays, shots, rounds, edges), "
            f"got {field.shape}"
        )

    def str_array(name: str) -> List[str]:
        return [str(x) for x in z[name].tolist()] if name in z.files else []

    obj = {
        "field": field,
        "final": bits(z["final"]) if "final" in z.files else None,
        "modes": str_array("modes"),
        "delay_sites": str_array("delay_sites"),
        "delays": z["delays"].astype(int).tolist() if "delays" in z.files else list(range(field.shape[2])),
        "delay_unit": str(z["delay_unit"].item()) if "delay_unit" in z.files and z["delay_unit"].shape == () else "",
        "job_id": str(z["job_id"].item()) if "job_id" in z.files and z["job_id"].shape == () else npz_path.stem,
        "rounds": int(z["rounds"].item()) if "rounds" in z.files else int(field.shape[4]),
        "channels": int(z["channels"].item()) if "channels" in z.files else int(field.shape[5] + 1),
        "edges": int(z["edges"].item()) if "edges" in z.files else int(field.shape[5]),
    }

    if not obj["modes"]:
        obj["modes"] = [f"mode_{i}" for i in range(field.shape[0])]
    if not obj["delay_sites"]:
        obj["delay_sites"] = [f"site_{i}" for i in range(field.shape[1])]

    return obj


# =============================================================================
# CONTROLS
# =============================================================================

CONTROL_MODES = [
    "real",
    "shot_shuffle",
    "delay_shuffle",
    "round_shuffle",
    "edge_shuffle",
    "uniform_field",
    "all_uniform",
    "delay_reverse",
    "round_reverse",
    "edge_reverse",
]


def mutate_field(field: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """
    Mutate one field block with shape:

        delay, shot, round, edge

    Controls preserve increasingly weaker marginal structure.
    """
    f = field.copy()

    if mode == "real":
        return f

    if mode == "shot_shuffle":
        return f[:, rng.permutation(f.shape[1]), :, :]

    if mode == "delay_shuffle":
        return f[rng.permutation(f.shape[0]), :, :, :]

    if mode == "round_shuffle":
        for d in range(f.shape[0]):
            for sh in range(f.shape[1]):
                for e in range(f.shape[3]):
                    f[d, sh, :, e] = f[d, sh, rng.permutation(f.shape[2]), e]
        return f

    if mode == "edge_shuffle":
        for d in range(f.shape[0]):
            for sh in range(f.shape[1]):
                for r in range(f.shape[2]):
                    f[d, sh, r, :] = f[d, sh, r, rng.permutation(f.shape[3])]
        return f

    if mode == "uniform_field":
        # Preserves per-delay/per-round/per-edge marginal rates.
        p = f.mean(axis=1, keepdims=True)
        return (rng.random(f.shape) < p).astype(np.uint8)

    if mode == "all_uniform":
        p = float(f.mean())
        return (rng.random(f.shape) < p).astype(np.uint8)

    if mode == "delay_reverse":
        return f[::-1, :, :, :].copy()

    if mode == "round_reverse":
        return f[:, :, ::-1, :].copy()

    if mode == "edge_reverse":
        return f[:, :, :, ::-1].copy()

    raise ValueError(mode)


# =============================================================================
# SHAPE PROBE
# =============================================================================

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def cv(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.std(x) / (abs(np.mean(x)) + 1e-12))


def summarize_repr(f: np.ndarray) -> Dict[str, Any]:
    """
    Summarize a block with shape delay, shot, round, edge.
    """
    return {
        "scalar": float(f.mean()),
        "delay_vec": f.mean(axis=(1, 2, 3)),
        "round_vec": f.mean(axis=(0, 1, 3)),
        "edge_vec": f.mean(axis=(0, 1, 2)),
        "delay_round": f.mean(axis=(1, 3)),
        "delay_edge": f.mean(axis=(1, 2)),
        "round_edge": f.mean(axis=(0, 1)),
        "field": f.mean(axis=1),  # delay, round, edge
    }


def classify_shape(row: Dict[str, float]) -> str:
    scalar = row["best_scalar_gap"]
    full = row["best_field_l2"]
    delay = row["best_delay_l2"]
    round_ = row["best_round_l2"]
    edge = row["best_edge_l2"]

    if full >= 4.0 * max(scalar, 1e-12):
        if row["delay_cv"] > 0.010 and row["edge_cv"] > 0.010:
            return "delay-channel field"
        if row["delay_cv"] > 0.010:
            return "delay-anisotropic field"
        if row["round_cv"] > 0.010:
            return "round-anisotropic field"
        if row["edge_cv"] > 0.010:
            return "edge-anisotropic field"
        return "smooth delay-round-edge field"

    if delay >= 3.0 * max(scalar, 1e-12):
        return "delay-vector"
    if round_ >= 3.0 * max(scalar, 1e-12):
        return "round-vector"
    if edge >= 3.0 * max(scalar, 1e-12):
        return "edge-vector"
    if scalar > 0 and full < 2.0 * scalar:
        return "scalar-like"
    return "mixed/undecided"


def run_shape(ts: Dict[str, Any], seed: int) -> Dict[str, Any]:
    field = ts["field"]
    rows = []
    profiles = {}

    for mi, mode_name in enumerate(ts["modes"]):
        for si, site_name in enumerate(ts["delay_sites"]):
            block = field[mi, si]  # delays, shots, rounds, edges
            rng = np.random.default_rng(seed + 1009 * mi + 9176 * si)

            reps = {}
            for cmode in CONTROL_MODES:
                reps[cmode] = summarize_repr(mutate_field(block, cmode, rng))

            real = reps["real"]
            gaps = {}
            for cmode, rep in reps.items():
                if cmode == "real":
                    continue
                gaps[cmode] = {
                    "scalar_gap": abs(real["scalar"] - rep["scalar"]),
                    "delay_l2": l2(real["delay_vec"], rep["delay_vec"]),
                    "round_l2": l2(real["round_vec"], rep["round_vec"]),
                    "edge_l2": l2(real["edge_vec"], rep["edge_vec"]),
                    "delay_round_l2": l2(real["delay_round"], rep["delay_round"]),
                    "delay_edge_l2": l2(real["delay_edge"], rep["delay_edge"]),
                    "round_edge_l2": l2(real["round_edge"], rep["round_edge"]),
                    "field_l2": l2(real["field"], rep["field"]),
                }

            row = {
                "mode": mode_name,
                "delay_site": site_name,
                "best_scalar_gap": max(g["scalar_gap"] for g in gaps.values()),
                "best_delay_l2": max(g["delay_l2"] for g in gaps.values()),
                "best_round_l2": max(g["round_l2"] for g in gaps.values()),
                "best_edge_l2": max(g["edge_l2"] for g in gaps.values()),
                "best_delay_round_l2": max(g["delay_round_l2"] for g in gaps.values()),
                "best_delay_edge_l2": max(g["delay_edge_l2"] for g in gaps.values()),
                "best_round_edge_l2": max(g["round_edge_l2"] for g in gaps.values()),
                "best_field_l2": max(g["field_l2"] for g in gaps.values()),
                "delay_cv": cv(real["delay_vec"]),
                "round_cv": cv(real["round_vec"]),
                "edge_cv": cv(real["edge_vec"]),
                "field_cv": cv(real["field"].ravel()),
                "field_rate": float(real["scalar"]),
            }
            row["shape_guess"] = classify_shape(row)
            rows.append(row)

            key = f"{mode_name}::{site_name}"
            profiles[key] = {
                "delay_vec": real["delay_vec"],
                "round_vec": real["round_vec"],
                "edge_vec": real["edge_vec"],
                "field": real["field"],
                "control_gaps": gaps,
            }

    return {"rows": rows, "profiles": profiles}


# =============================================================================
# TEMPORAL STRESS TENSOR
# =============================================================================

def gradients_delay_round_edge(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute aligned binary gradients for one field block:

        f shape = delay, shot, round, edge

    Returned arrays have common shape:

        delay-1, shot, round-1, edge-1
    """
    if f.shape[0] < 2 or f.shape[2] < 2 or f.shape[3] < 2:
        empty = np.zeros((0, f.shape[1], 0, 0), dtype=np.float32)
        return empty, empty, empty

    d_tau = np.bitwise_xor(f[1:, :, :, :], f[:-1, :, :, :]).astype(np.float32)
    d_round = np.bitwise_xor(f[:, :, 1:, :], f[:, :, :-1, :]).astype(np.float32)
    d_edge = np.bitwise_xor(f[:, :, :, 1:], f[:, :, :, :-1]).astype(np.float32)

    # Align to common delay/round/edge cell lattice.
    d_tau = d_tau[:, :, :-1, :-1]
    d_round = d_round[:-1, :, :, :-1]
    d_edge = d_edge[:-1, :, :-1, :]

    return d_tau, d_round, d_edge


def stress_from(d_tau: np.ndarray, d_round: np.ndarray, d_edge: np.ndarray) -> Dict[str, Any]:
    if d_tau.size == 0:
        zero_local = np.zeros((0, 0, 0), dtype=float)
        return {
            "T": np.zeros((3, 3), dtype=float),
            "Ttau_tau": 0.0,
            "Trr": 0.0,
            "Txx": 0.0,
            "Ttau_r": 0.0,
            "Ttau_x": 0.0,
            "Trx": 0.0,
            "trace": 0.0,
            "delay_fraction": 0.0,
            "spatial_fraction": 0.0,
            "coupling_norm": 0.0,
            "eigvals": [0.0, 0.0, 0.0],
            "local_trace": zero_local,
            "local_delay": zero_local,
            "local_round": zero_local,
            "local_edge": zero_local,
            "local_coupling": zero_local,
        }

    comps = [d_tau, d_round, d_edge]
    local = [[np.mean(a * b, axis=1) for b in comps] for a in comps]  # each delay, round, edge
    T = np.array([[float(x.mean()) for x in row] for row in local], dtype=float)

    local_trace = local[0][0] + local[1][1] + local[2][2]
    local_coupling = np.sqrt(local[0][1] ** 2 + local[0][2] ** 2 + local[1][2] ** 2)

    trace = float(np.trace(T))
    diag_sum = float(T[0, 0] + T[1, 1] + T[2, 2])
    delay_fraction = float(T[0, 0] / (diag_sum + 1e-12))
    spatial_fraction = float(T[2, 2] / (diag_sum + 1e-12))
    coupling_norm = float(
        np.sqrt(T[0, 1] ** 2 + T[0, 2] ** 2 + T[1, 2] ** 2)
        / (diag_sum + 1e-12)
    )

    return {
        "T": T,
        "Ttau_tau": float(T[0, 0]),
        "Trr": float(T[1, 1]),
        "Txx": float(T[2, 2]),
        "Ttau_r": float(T[0, 1]),
        "Ttau_x": float(T[0, 2]),
        "Trx": float(T[1, 2]),
        "trace": trace,
        "delay_fraction": delay_fraction,
        "spatial_fraction": spatial_fraction,
        "coupling_norm": coupling_norm,
        "eigvals": np.linalg.eigvalsh(T).tolist(),
        "local_trace": local_trace,
        "local_delay": local[0][0],
        "local_round": local[1][1],
        "local_edge": local[2][2],
        "local_coupling": local_coupling,
    }


def tensor_gap(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    return {
        "fro_gap": float(np.linalg.norm(np.asarray(a["T"]) - np.asarray(b["T"]))),
        "trace_gap": abs(float(a["trace"]) - float(b["trace"])),
        "delay_fraction_gap": abs(float(a["delay_fraction"]) - float(b["delay_fraction"])),
        "spatial_fraction_gap": abs(float(a["spatial_fraction"]) - float(b["spatial_fraction"])),
        "coupling_norm_gap": abs(float(a["coupling_norm"]) - float(b["coupling_norm"])),
        "local_trace_l2": l2(a["local_trace"], b["local_trace"]),
    }


def delay_survival_curve(f: np.ndarray) -> np.ndarray:
    """
    Simple delay survival diagnostic.

    For each delay index, compute field agreement against delay zero across
    shots/rounds/edges.  This is not a claim of coherence by itself; it is a
    useful ordering-sensitive diagnostic that delay shuffle/reverse controls
    should disturb if the delay structure matters.
    """
    base = f[0:1]
    return (1.0 - np.bitwise_xor(base, f).astype(np.float32)).mean(axis=(1, 2, 3))


def run_stress(ts: Dict[str, Any], seed: int) -> Dict[str, Any]:
    field = ts["field"]
    results = []

    for mi, mode_name in enumerate(ts["modes"]):
        for si, site_name in enumerate(ts["delay_sites"]):
            block = field[mi, si]  # delays, shots, rounds, edges
            rng = np.random.default_rng(seed + 5519 * mi + 31337 * si)

            tensors = {}
            survival = {}
            for cmode in CONTROL_MODES:
                mutated = mutate_field(block, cmode, rng)
                d_tau, d_round, d_edge = gradients_delay_round_edge(mutated)
                tensors[cmode] = stress_from(d_tau, d_round, d_edge)
                survival[cmode] = delay_survival_curve(mutated)

            real = tensors["real"]
            gaps = {m: tensor_gap(real, t) for m, t in tensors.items() if m != "real"}

            delay_shuffle_gap = tensor_gap(real, tensors["delay_shuffle"])
            best = {
                "best_fro_gap": max(g["fro_gap"] for g in gaps.values()),
                "best_trace_gap": max(g["trace_gap"] for g in gaps.values()),
                "best_delay_fraction_gap": max(g["delay_fraction_gap"] for g in gaps.values()),
                "best_spatial_fraction_gap": max(g["spatial_fraction_gap"] for g in gaps.values()),
                "best_coupling_norm_gap": max(g["coupling_norm_gap"] for g in gaps.values()),
                "best_local_trace_l2": max(g["local_trace_l2"] for g in gaps.values()),
                "delay_survival_gap": float(np.linalg.norm(survival["real"] - survival["delay_shuffle"])),
                "delay_reverse_survival_gap": float(np.linalg.norm(survival["real"] - survival["delay_reverse"])),
                "delay_shuffle_fro_gap": delay_shuffle_gap["fro_gap"],
            }

            results.append({
                "mode": mode_name,
                "delay_site": site_name,
                "delays": ts["delays"],
                "delay_unit": ts["delay_unit"],
                "shots": int(block.shape[1]),
                "rounds": int(block.shape[2]),
                "edges": int(block.shape[3]),
                "real": real,
                "gaps": gaps,
                "survival": survival,
                "best": best,
            })

    return {"results": results}


# =============================================================================
# PLOTS
# =============================================================================

def plot_stress_summary(stress_report: Dict[str, Any], out_dir: Path) -> None:
    if not _HAVE_MPL:
        return

    rows = stress_report["results"]
    labels = [f"{r['mode']}\n{r['delay_site']}" for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(11, len(rows) * 0.85), 6.5), dpi=160)
    ax.plot(x, [r["real"]["Ttau_tau"] for r in rows], marker="o", label="Tττ delay")
    ax.plot(x, [r["real"]["Trr"] for r in rows], marker="o", label="Trr round")
    ax.plot(x, [r["real"]["Txx"] for r in rows], marker="o", label="Txx edge")
    ax.plot(x, [r["real"]["Ttau_x"] for r in rows], marker="o", label="Tτx delay-edge")
    ax.plot(x, [r["real"]["Trx"] for r in rows], marker="o", label="Trx round-edge")
    ax.set_title("T_S Temporal Stress Metric components")
    ax.set_ylabel("component value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "temporal_stress_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_delay_survival(stress_report: Dict[str, Any], out_dir: Path) -> None:
    if not _HAVE_MPL:
        return

    for r in stress_report["results"]:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
        delays = np.asarray(r["delays"], dtype=float)
        ax.plot(delays, r["survival"]["real"], marker="o", label="real")
        ax.plot(delays, r["survival"]["delay_shuffle"], marker="o", label="delay shuffle")
        ax.plot(delays, r["survival"]["delay_reverse"], marker="o", label="delay reverse")
        ax.plot(delays, r["survival"]["all_uniform"], marker="o", label="all uniform")
        ax.set_title(f"T_S delay survival — {r['mode']} / {r['delay_site']}")
        ax.set_xlabel(f"delay ({r['delay_unit']})")
        ax.set_ylabel("agreement with delay[0]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        safe = f"{r['mode']}_{r['delay_site']}".replace("/", "_")
        fig.savefig(out_dir / f"delay_survival_{safe}.png", bbox_inches="tight")
        plt.close(fig)


def plot_local_trace(stress_report: Dict[str, Any], out_dir: Path) -> None:
    if not _HAVE_MPL:
        return

    for r in stress_report["results"]:
        local = np.asarray(r["real"]["local_trace"], dtype=float)
        if local.size == 0:
            continue

        # Average over edge to show delay × round trace.
        heat = local.mean(axis=2)
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)
        im = ax.imshow(heat, aspect="auto")
        ax.set_title(f"T_S local trace — {r['mode']} / {r['delay_site']}")
        ax.set_xlabel("round cell")
        ax.set_ylabel("delay cell")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="local trace")
        safe = f"{r['mode']}_{r['delay_site']}".replace("/", "_")
        fig.savefig(out_dir / f"local_trace_{safe}.png", bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(shape_report: Dict[str, Any], stress_report: Dict[str, Any]) -> None:
    print("\n" + "=" * 118)
    print("  T_S SHAPE SUMMARY")
    print("=" * 118)
    print(
        f"  {'mode':>13} | {'site':>14} | {'rate':>8} | {'field L2':>9} | "
        f"{'delay CV':>8} | {'round CV':>8} | {'edge CV':>8} | shape"
    )
    print("  " + "-" * 116)
    for r in shape_report["rows"]:
        print(
            f"  {r['mode']:>13} | {r['delay_site']:>14} | {r['field_rate']:>8.5f} | "
            f"{r['best_field_l2']:>9.4f} | {r['delay_cv']:>8.4f} | "
            f"{r['round_cv']:>8.4f} | {r['edge_cv']:>8.4f} | {r['shape_guess']}"
        )

    print("\n" + "=" * 118)
    print("  T_S TEMPORAL STRESS SUMMARY")
    print("=" * 118)
    print(
        f"  {'mode':>13} | {'site':>14} | {'Tττ':>8} | {'Trr':>8} | {'Txx':>8} | "
        f"{'Tτx':>8} | {'Trx':>8} | {'trace':>8} | {'delay gap':>9} | {'surv gap':>9}"
    )
    print("  " + "-" * 116)
    for r in stress_report["results"]:
        real = r["real"]
        best = r["best"]
        print(
            f"  {r['mode']:>13} | {r['delay_site']:>14} | "
            f"{real['Ttau_tau']:>8.5f} | {real['Trr']:>8.5f} | {real['Txx']:>8.5f} | "
            f"{real['Ttau_x']:>8.5f} | {real['Trx']:>8.5f} | {real['trace']:>8.5f} | "
            f"{best['delay_shuffle_fro_gap']:>9.5f} | {best['delay_survival_gap']:>9.5f}"
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T_S step 2 — analyze Temporal Stress Metric QPU data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz", default=None, help="Dumped T_S .npz. Defaults to latest dumped dataset.")
    p.add_argument("--meta", default=None, help="Optional metadata path.")
    p.add_argument("--out-dir", default=None, help="Output directory. Defaults to ghost_oracle/T_S/analysis/ts_<JOB_ID>.")
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path, meta_path, job_id = resolve_inputs(args.npz, args.meta)
    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"ts_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 118}")
    print("  T_S STEP 2 — TEMPORAL STRESS METRIC ANALYSIS")
    print(f"{'=' * 118}")
    print(f"  NPZ     : {npz_path}")
    print(f"  Metadata: {meta_path if meta_path else '(not provided)'}")
    print(f"  Out dir : {out_dir}")

    ts = load_ts(npz_path)
    shape_report = run_shape(ts, args.seed)
    stress_report = run_stress(ts, args.seed)

    with open(out_dir / "temporal_shape_report.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(shape_report), f, indent=2)
    with open(out_dir / "temporal_stress_report.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(stress_report), f, indent=2)

    if not args.no_plots:
        plot_stress_summary(stress_report, out_dir)
        plot_delay_survival(stress_report, out_dir)
        plot_local_trace(stress_report, out_dir)

    print_summary(shape_report, stress_report)

    print(f"\n[SAVED] {out_dir}")
    print(f"{'=' * 118}\n")


if __name__ == "__main__":
    main()
