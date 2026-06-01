#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
F_M PROBE 04 — CUDA PROJECTOR SIGNATURE FINALIZER
==============================================================================

Purpose
-------
Finalize the first F_M qproj signature using the CUDA projector kernel.

This probe is intentionally narrower than Probes 01-03.

Known from prior probes:
    primary fields:
        delta
        xor_delta

    primary responses:
        bit_diff
        bit1_mean
        transition

    primary order:
        delay

    strongest control:
        path_pair_break

Probe 04 uses the CUDA kernel to compute response curves and wave metrics,
then confirms the qproj signature using GPU-side projector outputs.

This is the first optimized F_M projector path

    operator base -> CUDA projector -> F_M projector signature

Outputs
-------
analysis/fm_probe04_qproj_kernel_finalizer_<timestamp>/
    result.json
    qproj_signature.csv
    curve_values.csv
    control_scores.csv
    gpu_metrics.csv

Usage
-----
    python ghost_oracle/F_M/probes/f_m_probe04_qproj_kernel_finalizer.py

or:

    python ghost_oracle/F_M/probes/f_m_probe04_qproj_kernel_finalizer.py ^
      --file ghost_oracle/F_M/data/fm_job_<JOB_ID>.npz

Requirements
------------
    cupy

==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp
    HAVE_CUPY = True
except Exception:
    cp = None
    HAVE_CUPY = False


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
FM_DIR = HERE.parent
DATA_DIR = FM_DIR / "data"
KERNEL_PATH = FM_DIR / "kernels" / "fm_projector_kernel.cu"
ANALYSIS_DIR = HERE / "analysis"


# =============================================================================
# CONSTANTS
# =============================================================================

FIELD_KINDS = {
    "delta": 0,
    "xor_delta": 1,
    "g": 2,
    "em": 3,
}

RESPONSE_KINDS = {
    "mean": 0,
    "energy": 1,
    "transition": 2,
    "imbalance": 3,
    "bit0_mean": 4,
    "bit1_mean": 5,
    "bit_diff": 6,
}

METRIC_NAMES = [
    "wave_score",
    "peak_ratio",
    "spectral_entropy",
    "best_r2",
    "best_freq",
    "best_amp",
    "best_phase",
    "low_high_ratio",
]


# =============================================================================
# IO HELPERS
# =============================================================================

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2)


def write_csv(path: Path, rows: List[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def find_latest_fm_file() -> Optional[Path]:
    ptr = DATA_DIR / "latest_fm_qpu_data.json"
    if ptr.exists():
        try:
            with open(ptr, "r", encoding="utf-8") as f:
                j = json.load(f)
            p = Path(j["path"])
            if p.exists():
                return p
        except Exception:
            pass

    files = sorted(DATA_DIR.glob("fm_job_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_fm_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=True) as z:
        if "g" not in z or "em" not in z:
            raise RuntimeError(f"F_M qproj file requires stacked g/em arrays. Keys={list(z.keys())}")

        g = np.asarray(z["g"], dtype=np.uint8)
        em = np.asarray(z["em"], dtype=np.uint8)

        meta: Dict[str, Any] = {}
        for k in [
            "schema", "operator", "substrate", "job_id", "backend",
            "shots", "num_tiles", "tile_indices",
            "tile_delay_dt", "tile_scale_level", "tile_mode",
            "tile_theta", "circuit_family",
        ]:
            if k in z:
                v = z[k]
                try:
                    meta[k] = v.item() if v.shape == () else np.asarray(v).tolist()
                except Exception:
                    meta[k] = str(v)

    if g.shape != em.shape:
        raise RuntimeError(f"g/em shape mismatch: {g.shape} vs {em.shape}")
    if g.ndim != 3:
        raise RuntimeError(f"expected g/em shape (tiles, shots, bits), got {g.shape}")

    return g, em, meta


def meta_array(meta: Dict[str, Any], key: str, n: int, default: Any) -> List[Any]:
    v = meta.get(key, None)
    if v is None:
        return [default for _ in range(n)]
    if not isinstance(v, list):
        return [v for _ in range(n)]
    if len(v) < n:
        return v + [default for _ in range(n - len(v))]
    return v[:n]


# =============================================================================
# CUDA WRAPPER
# =============================================================================

class FMProjectorCUDA:
    def __init__(self, kernel_path: Path):
        if not HAVE_CUPY:
            raise RuntimeError("CuPy is required for F_M CUDA projector probe.")
        if not kernel_path.exists():
            raise FileNotFoundError(kernel_path)

        code = kernel_path.read_text(encoding="utf-8")
        self.mod = cp.RawModule(
            code=code,
            options=("--std=c++11",),
            name_expressions=[
                "fm_response_kernel_u8",
                "fm_path_pair_break_response_kernel_u8",
                "fm_wave_metric_kernel_f32",
            ],
        )
        self.response_kernel = self.mod.get_function("fm_response_kernel_u8")
        self.path_break_kernel = self.mod.get_function("fm_path_pair_break_response_kernel_u8")
        self.wave_kernel = self.mod.get_function("fm_wave_metric_kernel_f32")

    def compute_responses(
        self,
        g: np.ndarray,
        em: np.ndarray,
        fields: Sequence[str],
        responses: Sequence[str],
        path_pair_break_seed: Optional[int] = None,
    ) -> np.ndarray:
        g_gpu = cp.asarray(np.ascontiguousarray(g, dtype=np.uint8))
        em_gpu = cp.asarray(np.ascontiguousarray(em, dtype=np.uint8))

        tiles, shots, bits = g.shape

        field_ids = np.asarray([FIELD_KINDS[f] for f in fields], dtype=np.int32)
        response_ids = np.asarray([RESPONSE_KINDS[r] for r in responses], dtype=np.int32)

        field_gpu = cp.asarray(field_ids)
        response_gpu = cp.asarray(response_ids)

        out = cp.zeros((len(fields), len(responses), tiles), dtype=cp.float32)

        grid = (tiles, len(responses), len(fields))
        block = (1,)

        if path_pair_break_seed is None:
            self.response_kernel(
                grid,
                block,
                (
                    g_gpu,
                    em_gpu,
                    np.int32(tiles),
                    np.int32(shots),
                    np.int32(bits),
                    field_gpu,
                    np.int32(len(fields)),
                    response_gpu,
                    np.int32(len(responses)),
                    out,
                ),
            )
        else:
            self.path_break_kernel(
                grid,
                block,
                (
                    g_gpu,
                    em_gpu,
                    np.int32(tiles),
                    np.int32(shots),
                    np.int32(bits),
                    field_gpu,
                    np.int32(len(fields)),
                    response_gpu,
                    np.int32(len(responses)),
                    np.int32(path_pair_break_seed),
                    out,
                ),
            )

        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(out)

    def compute_wave_metrics(self, curves: np.ndarray, xs: np.ndarray) -> np.ndarray:
        curves = np.ascontiguousarray(curves, dtype=np.float32)
        xs = np.ascontiguousarray(xs, dtype=np.float32)

        curves_gpu = cp.asarray(curves)
        xs_gpu = cp.asarray(xs)
        out = cp.zeros((curves.shape[0], len(METRIC_NAMES)), dtype=cp.float32)

        self.wave_kernel(
            (curves.shape[0],),
            (1,),
            (
                curves_gpu,
                xs_gpu,
                np.int32(curves.shape[0]),
                np.int32(curves.shape[1]),
                out,
            ),
        )

        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(out)


# =============================================================================
# CURVE BUILDING / CONTROLS
# =============================================================================

def order_indices(meta: Dict[str, Any], tiles: int, order: str) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    tile_indices = np.asarray(meta_array(meta, "tile_indices", tiles, default=list(range(tiles))), dtype=object)

    tile_int = []
    for i, v in enumerate(tile_indices):
        try:
            tile_int.append(int(v))
        except Exception:
            tile_int.append(i)
    tile_int = np.asarray(tile_int, dtype=np.int32)

    delays = np.asarray([float(v) for v in meta_array(meta, "tile_delay_dt", tiles, default=-1)], dtype=np.float32)
    scales = np.asarray([float(v) for v in meta_array(meta, "tile_scale_level", tiles, default=-1)], dtype=np.float32)
    modes = [str(v) for v in meta_array(meta, "tile_mode", tiles, default="unknown")]

    if order == "delay":
        idx = np.lexsort((tile_int, delays))
        xs = delays[idx]
    elif order == "tile":
        idx = np.argsort(tile_int)
        xs = tile_int[idx].astype(np.float32)
    elif order == "scale_delay":
        idx = np.lexsort((tile_int, delays, scales))
        xs = np.arange(tiles, dtype=np.float32)
    else:
        raise ValueError(order)

    point_meta = []
    for j in idx:
        point_meta.append({
            "tile_index": int(tile_int[j]),
            "delay_dt": float(delays[j]),
            "scale_level": float(scales[j]),
            "mode": modes[j],
        })

    return idx.astype(np.int32), xs.astype(np.float32), point_meta


def build_curves_from_responses(
    responses_arr: np.ndarray,
    fields: Sequence[str],
    responses: Sequence[str],
    meta: Dict[str, Any],
    order: str,
) -> Tuple[np.ndarray, List[dict], np.ndarray, List[dict]]:
    """
    responses_arr shape:
        (n_fields, n_responses, tiles)
    """
    tiles = responses_arr.shape[2]
    idx, xs, point_meta = order_indices(meta, tiles, order)

    curves = []
    curve_meta = []

    for fi, field in enumerate(fields):
        for ri, response in enumerate(responses):
            y = responses_arr[fi, ri, idx]
            curves.append(y.astype(np.float32))
            curve_meta.append({
                "field": field,
                "response": response,
                "order": order,
            })

    return np.vstack(curves).astype(np.float32), curve_meta, xs, point_meta


def phase_scramble_cpu(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    yy = y.astype(np.float64) - float(np.mean(y))
    n = yy.size
    if n < 4 or float(np.std(yy)) < 1e-12:
        return rng.permutation(y).astype(np.float32)

    fft = np.fft.rfft(yy)
    mag = np.abs(fft)
    phases = rng.uniform(-np.pi, np.pi, size=fft.shape)
    phases[0] = 0.0
    if n % 2 == 0 and phases.size > 1:
        phases[-1] = 0.0

    new_fft = mag * np.exp(1j * phases)
    out = np.fft.irfft(new_fft, n=n)
    return (out + np.mean(y)).astype(np.float32)


def make_control_curves(
    base_curves: np.ndarray,
    control: str,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.empty_like(base_curves)

    for i in range(base_curves.shape[0]):
        y = base_curves[i].copy()

        if control == "delay_shuffle":
            rng.shuffle(y)
            out[i] = y

        elif control == "delay_reverse":
            out[i] = y[::-1]

        elif control == "circular_shift":
            shift = int(rng.integers(1, max(2, y.size)))
            out[i] = np.roll(y, shift)

        elif control == "phase_scramble":
            out[i] = phase_scramble_cpu(y, rng)

        elif control == "iid_gaussian":
            out[i] = rng.normal(float(np.mean(y)), float(np.std(y) + 1e-12), size=y.shape).astype(np.float32)

        else:
            raise ValueError(control)

    return out


# =============================================================================
# MAIN
# =============================================================================

@dataclass
class SignatureRow:
    field: str
    response: str
    order: str
    wave_score: float
    peak_ratio: float
    spectral_entropy: float
    best_r2: float
    best_freq: float
    best_amp: float
    best_phase: float
    low_high_ratio: float


@dataclass
class ControlRow:
    field: str
    response: str
    order: str
    control: str
    real_score: float
    null_mean: float
    null_std: float
    effect: float
    separation_z: float
    auc_rank: float
    n_null: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F_M Probe 04: CUDA qproj projector finalizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", default=None, help="F_M qproj .npz. Defaults to latest pointer.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-null", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20260601)

    p.add_argument(
        "--fields",
        nargs="+",
        default=["xor_delta", "delta"],
        choices=list(FIELD_KINDS.keys()),
    )
    p.add_argument(
        "--responses",
        nargs="+",
        default=["bit_diff", "bit1_mean", "transition", "energy"],
        choices=list(RESPONSE_KINDS.keys()),
    )
    p.add_argument(
        "--orders",
        nargs="+",
        default=["delay", "tile", "scale_delay"],
        choices=["delay", "tile", "scale_delay"],
    )
    p.add_argument(
        "--controls",
        nargs="+",
        default=["path_pair_break", "delay_shuffle", "phase_scramble", "circular_shift", "iid_gaussian"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not HAVE_CUPY:
        raise RuntimeError("CuPy is required. Install cupy-cuda12x or the matching CuPy build for your CUDA version.")

    in_path = Path(args.file) if args.file else find_latest_fm_file()
    if in_path is None or not in_path.exists():
        raise FileNotFoundError("No F_M qproj input file found.")

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"fm_probe04_qproj_kernel_finalizer_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    g, em, meta = load_fm_npz(in_path)
    tiles, shots, bits = g.shape

    projector = FMProjectorCUDA(KERNEL_PATH)

    print("=" * 108)
    print("  F_M PROBE 04 — QPROJ CUDA PROJECTOR FINALIZER")
    print("=" * 108)
    print(f"  file      : {in_path}")
    print(f"  out_dir   : {out_dir}")
    print(f"  backend   : {meta.get('backend', 'unknown')}")
    print(f"  job_id    : {meta.get('job_id', 'unknown')}")
    print(f"  substrate : {meta.get('substrate', 'unknown')}")
    print(f"  shape     : tiles={tiles}, shots={shots}, bits={bits}")
    print(f"  fields    : {args.fields}")
    print(f"  responses : {args.responses}")
    print(f"  orders    : {args.orders}")
    print(f"  controls  : {args.controls}")
    print(f"  n_null    : {args.n_null}")
    print("-" * 108)

    t0 = time.perf_counter()
    response_arr = projector.compute_responses(
        g=g,
        em=em,
        fields=args.fields,
        responses=args.responses,
    )
    t_response = time.perf_counter() - t0

    signature_rows: List[dict] = []
    curve_rows: List[dict] = []
    control_rows: List[dict] = []
    gpu_metric_rows: List[dict] = []

    for order in args.orders:
        curves, curve_meta, xs, point_meta = build_curves_from_responses(
            response_arr,
            fields=args.fields,
            responses=args.responses,
            meta=meta,
            order=order,
        )

        t1 = time.perf_counter()
        metrics = projector.compute_wave_metrics(curves, xs)
        t_wave = time.perf_counter() - t1

        for ci, cm in enumerate(curve_meta):
            md = {name: float(metrics[ci, mi]) for mi, name in enumerate(METRIC_NAMES)}

            row = SignatureRow(
                field=cm["field"],
                response=cm["response"],
                order=cm["order"],
                wave_score=md["wave_score"],
                peak_ratio=md["peak_ratio"],
                spectral_entropy=md["spectral_entropy"],
                best_r2=md["best_r2"],
                best_freq=md["best_freq"],
                best_amp=md["best_amp"],
                best_phase=md["best_phase"],
                low_high_ratio=md["low_high_ratio"],
            )
            signature_rows.append(asdict(row))

            gpu_metric_rows.append({
                **asdict(row),
                "response_seconds": t_response,
                "wave_seconds": t_wave,
                "curves_per_second": float(curves.shape[0] / max(t_wave, 1e-12)),
            })

            for pi, y in enumerate(curves[ci]):
                pm = point_meta[pi]
                curve_rows.append({
                    "field": cm["field"],
                    "response": cm["response"],
                    "order": cm["order"],
                    "point": int(pi),
                    "x": float(xs[pi]),
                    "y": float(y),
                    **pm,
                })

        # Controls.
        for control in args.controls:
            if control == "path_pair_break":
                null_scores_all = [[] for _ in range(curves.shape[0])]

                for k in range(args.n_null):
                    broken_response = projector.compute_responses(
                        g=g,
                        em=em,
                        fields=args.fields,
                        responses=args.responses,
                        path_pair_break_seed=args.seed + k + 1,
                    )
                    broken_curves, _, _, _ = build_curves_from_responses(
                        broken_response,
                        fields=args.fields,
                        responses=args.responses,
                        meta=meta,
                        order=order,
                    )
                    broken_metrics = projector.compute_wave_metrics(broken_curves, xs)

                    for ci in range(curves.shape[0]):
                        null_scores_all[ci].append(float(broken_metrics[ci, 0]))

            else:
                null_scores_all = [[] for _ in range(curves.shape[0])]
                rng = np.random.default_rng(args.seed + len(control) * 917)

                for _ in range(args.n_null):
                    ccurves = make_control_curves(curves, control, rng)
                    cmetrics = projector.compute_wave_metrics(ccurves, xs)

                    for ci in range(curves.shape[0]):
                        null_scores_all[ci].append(float(cmetrics[ci, 0]))

            for ci, cm in enumerate(curve_meta):
                real_score = float(metrics[ci, 0])
                null_scores = np.asarray(null_scores_all[ci], dtype=np.float64)
                nmean = float(np.mean(null_scores))
                nstd = float(np.std(null_scores) + 1e-9)
                effect = float(real_score - nmean)
                z = float(effect / nstd)
                auc = float(np.mean(real_score > null_scores) + 0.5 * np.mean(real_score == null_scores))

                control_rows.append(asdict(ControlRow(
                    field=cm["field"],
                    response=cm["response"],
                    order=cm["order"],
                    control=control,
                    real_score=real_score,
                    null_mean=nmean,
                    null_std=nstd,
                    effect=effect,
                    separation_z=z,
                    auc_rank=auc,
                    n_null=int(args.n_null),
                )))

    signature_rows.sort(key=lambda r: r["wave_score"], reverse=True)
    control_rows.sort(key=lambda r: (r["effect"], r["auc_rank"], r["real_score"]), reverse=True)

    write_csv(
        out_dir / "projector_signature.csv",
        signature_rows,
        [
            "field", "response", "order",
            "wave_score", "peak_ratio", "spectral_entropy",
            "best_r2", "best_freq", "best_amp", "best_phase",
            "low_high_ratio",
        ],
    )

    write_csv(
        out_dir / "curve_values.csv",
        curve_rows,
        [
            "field", "response", "order", "point", "x", "y",
            "tile_index", "delay_dt", "scale_level", "mode",
        ],
    )

    write_csv(
        out_dir / "control_scores.csv",
        control_rows,
        [
            "field", "response", "order", "control",
            "real_score", "null_mean", "null_std",
            "effect", "separation_z", "auc_rank", "n_null",
        ],
    )

    write_csv(
        out_dir / "gpu_metrics.csv",
        gpu_metric_rows,
        [
            "field", "response", "order",
            "wave_score", "peak_ratio", "spectral_entropy",
            "best_r2", "best_freq", "best_amp", "best_phase",
            "low_high_ratio",
            "response_seconds", "wave_seconds", "curves_per_second",
        ],
    )

    result = {
        "probe": "F_M Probe 04 — CUDA Projector Signature Finalizer",
        "input_file": str(in_path),
        "out_dir": str(out_dir),
        "metadata": meta,
        "shape": {
            "tiles": int(tiles),
            "shots": int(shots),
            "bits": int(bits),
        },
        "config": {
            "fields": args.fields,
            "responses": args.responses,
            "orders": args.orders,
            "controls": args.controls,
            "n_null": args.n_null,
            "seed": args.seed,
        },
        "timing": {
            "response_seconds": float(t_response),
        },
        "projector_signature": signature_rows,
        "control_scores": control_rows,
    }
    write_json(out_dir / "result.json", result)

    print("\n" + "=" * 108)
    print("  TOP CUDA PROJECTOR SIGNATURE")
    print("=" * 108)
    for r in signature_rows[:16]:
        print(
            f"  {r['field']:10s} {r['response']:12s} order={r['order']:11s} "
            f"score={r['wave_score']:7.4f} peak={r['peak_ratio']:6.3f} "
            f"r2={r['best_r2']:6.3f} freq={r['best_freq']:5.2f} "
            f"amp={r['best_amp']:8.5f}"
        )

    print("\n" + "=" * 108)
    print("  TOP CONTROL SEPARATIONS")
    print("=" * 108)
    for r in control_rows[:20]:
        print(
            f"  {r['field']:10s} {r['response']:12s} order={r['order']:11s} "
            f"vs {r['control']:15s} "
            f"effect={r['effect']:8.4f} auc={r['auc_rank']:6.3f} "
            f"z={r['separation_z']:8.2f}"
        )

    print("\n" + "=" * 108)
    print("  TIMING")
    print("=" * 108)
    print(f"  response kernel seconds : {t_response:.6f}")
    print(f"  output dir              : {out_dir}")
    print("=" * 108)


if __name__ == "__main__":
    main()