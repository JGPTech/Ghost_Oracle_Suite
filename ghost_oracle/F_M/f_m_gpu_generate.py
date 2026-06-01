#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — F_M GPU BASE BUILDER
==============================================================================
Constructs an F_M GPU / gproj base compatible with the F_M QPU dump schema.

This is the GPU-side companion to:

    ghost_oracle/F_M/f_m_qpu_generate.py

The goal is not to simulate the QPU at the full pulse/noise level. The goal is
to generate a GPU base with the SAME analysis-facing schema and a controllable
paired-path differential wave signature matching the qproj target discovered
by Probes 01-04.

Current qproj signature to emulate
----------------------------------
Probe 04 CUDA qproj finalizer identified the strongest F_M signature as:

    field    : xor_delta
    response : bit_diff
    order    : delay

with runner-up:

    field    : xor_delta
    response : bit1_mean
    order    : delay

and meaningful collapse under:

    path_pair_break

Therefore this GPU builder generates paired g/em records such that:

    xor_delta / bit_diff / delay
    xor_delta / bit1_mean / delay

form an ordered wave-like response curve across tile delays.

Output schema
-------------
The .npz is intentionally compatible with F_M QPU dumps:

    schema              str
    suite               str
    operator            str
    substrate           str, "gproj"
    job_id              str
    backend             str
    shots               int
    num_tiles           int
    tile_indices        int32 array
    qubits_per_tile     int
    circuit_family      str
    delays_dt           int32 array
    scale_levels        int32 array
    tile_theta          float64 array
    tile_delay_dt       int32 array
    tile_scale_level    int32 array
    tile_mode           unicode array
    tile_role           unicode array
    tile_meta_json      unicode JSON string

    ctrl_tile{t}        uint8, shape (shots,)
    g_tile{t}           uint8, shape (shots, bits)
    em_tile{t}          uint8, shape (shots, bits)
    scale_tile{t}       uint8, shape (shots,)
    branch_tile{t}      uint8, shape (shots, 2)
    delta_tile{t}       int8,  shape (shots, bits)
    xor_delta_tile{t}   uint8, shape (shots, bits)

Stacked convenience arrays:

    ctrl                uint8, shape (tiles, shots)
    g                   uint8, shape (tiles, shots, bits)
    em                  uint8, shape (tiles, shots, bits)
    scale               uint8, shape (tiles, shots)
    branch              uint8, shape (tiles, shots, 2)
    delta               int8,  shape (tiles, shots, bits)
    xor_delta           uint8, shape (tiles, shots, bits)

Usage
-----
    python f_m_gpu_generate.py

    python f_m_gpu_generate.py --shots 4096 --seed 42

    python f_m_gpu_generate.py ^
      --match-qpu data/fm_job_d8eu8bjo3njc73evdd8g.npz ^
      --verify

    python f_m_gpu_generate.py --out data/fm_gpu_data_seed42.npz

Verification
------------
If the F_M CUDA projector kernel exists, this script runs the same projector
signature used in Probe 04:

    xor_delta / bit_diff / delay
    xor_delta / bit1_mean / delay
    delta     / transition / delay

This gives a direct gproj sanity check before running probes.

==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import secrets
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp
    _HAVE_CUPY = True
except Exception:
    cp = None
    _HAVE_CUPY = False


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
KERNEL_DIR = HERE / "kernels"
FM_PROJECTOR_KERNEL = KERNEL_DIR / "fm_projector_kernel.cu"


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_SHOTS = 4096
DEFAULT_BITS = 2

DEFAULT_DELAYS_DT = [0, 1, 2, 4, 8, 16]
DEFAULT_SCALE_LEVELS = [1, 2, 4, 8, 16, 32]
DEFAULT_THETA_VALUES = [0.25, 0.50, 0.75, 1.00]
DEFAULT_MODES = ["clean"]

# These were the strongest qproj signatures from Probe 04.
DEFAULT_TARGET_FIELD_RESPONSES = [
    ("xor_delta", "bit_diff"),
    ("xor_delta", "bit1_mean"),
    ("delta", "transition"),
]

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


def parse_int_list(values: Optional[Sequence[int]], default: Sequence[int]) -> List[int]:
    if values is None:
        return [int(v) for v in default]
    return [int(v) for v in values]


def parse_float_list(values: Optional[Sequence[float]], default: Sequence[float]) -> List[float]:
    if values is None:
        return [float(v) for v in default]
    return [float(v) for v in values]


# =============================================================================
# QPU MATCHING
# =============================================================================

@dataclass
class TileMeta:
    tile: int
    theta: float
    delay_dt: int
    scale_level: int
    mode: str
    role: str


def _read_npz_scalar_or_array(z: Any, key: str) -> Any:
    v = z[key]
    try:
        return v.item() if v.shape == () else np.asarray(v).tolist()
    except Exception:
        return str(v)


def load_qpu_metadata(path: Path) -> Dict[str, Any]:
    """
    Load metadata from an F_M qproj .npz so gproj can use identical tile layout.

    This lets the later probes treat qproj and gproj as interchangeable bases.
    """
    with np.load(path, allow_pickle=True) as z:
        meta: Dict[str, Any] = {}
        for k in [
            "shots",
            "num_tiles",
            "tile_indices",
            "tile_theta",
            "tile_delay_dt",
            "tile_scale_level",
            "tile_mode",
            "tile_role",
            "delays_dt",
            "scale_levels",
            "circuit_family",
        ]:
            if k in z:
                meta[k] = _read_npz_scalar_or_array(z, k)

        if "g" in z:
            meta["shape"] = tuple(np.asarray(z["g"]).shape)

    return meta


def meta_list(meta: Dict[str, Any], key: str, n: int, default: Any) -> List[Any]:
    v = meta.get(key, None)
    if v is None:
        return [default for _ in range(n)]
    if not isinstance(v, list):
        return [v for _ in range(n)]
    if len(v) < n:
        return v + [default for _ in range(n - len(v))]
    return v[:n]


def build_tile_plan(
    num_tiles: int,
    theta_values: Sequence[float],
    delays_dt: Sequence[int],
    scale_levels: Sequence[int],
    modes: Sequence[str],
    qpu_meta: Optional[Dict[str, Any]] = None,
) -> List[TileMeta]:
    """
    Build deterministic tile metadata.

    If qpu_meta is supplied, preserve QPU tile delay/scale/mode/theta exactly.
    """
    if qpu_meta is not None:
        thetas = meta_list(qpu_meta, "tile_theta", num_tiles, np.nan)
        delays = meta_list(qpu_meta, "tile_delay_dt", num_tiles, -1)
        scales = meta_list(qpu_meta, "tile_scale_level", num_tiles, -1)
        q_modes = meta_list(qpu_meta, "tile_mode", num_tiles, "clean")
        roles = meta_list(qpu_meta, "tile_role", num_tiles, "matched_qpu_tile")

        plan: List[TileMeta] = []
        for t in range(num_tiles):
            theta = float(thetas[t]) if str(thetas[t]) != "nan" else float(theta_values[t % len(theta_values)])
            delay = int(delays[t]) if int(delays[t]) >= 0 else int(delays_dt[t % len(delays_dt)])
            scale = int(scales[t]) if int(scales[t]) >= 0 else int(scale_levels[t % len(scale_levels)])
            mode = str(q_modes[t])
            role = str(roles[t])
            plan.append(TileMeta(t, theta, delay, scale, mode, role))
        return plan

    plan = []
    for t in range(num_tiles):
        theta = float(theta_values[t % len(theta_values)])
        delay = int(delays_dt[t % len(delays_dt)])
        scale = int(scale_levels[(t // max(1, len(delays_dt))) % len(scale_levels)])
        mode = str(modes[(t // max(1, len(delays_dt) * len(scale_levels))) % len(modes)])
        role = f"gpu_theta{t % len(theta_values)}_delay{delay}_scale{scale}_{mode}"
        plan.append(TileMeta(t, theta, delay, scale, mode, role))
    return plan


# =============================================================================
# CUDA GENERATOR KERNEL
# =============================================================================

_FM_GPU_GENERATE_KERNEL = r"""
extern "C" {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ __forceinline__ unsigned int fm_hash_u32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float fm_rand01(unsigned int seed, unsigned int tile, unsigned int shot, unsigned int lane) {
    unsigned int x = seed;
    x ^= tile * 0x9e3779b9U;
    x ^= shot * 0x85ebca6bU;
    x ^= lane * 0xc2b2ae35U;
    x = fm_hash_u32(x);
    return ((float)(x & 0x00FFFFFFU) + 0.5f) / 16777216.0f;
}

__device__ __forceinline__ int fm_idx3(int tile, int shot, int bit, int shots, int bits) {
    return (tile * shots + shot) * bits + bit;
}

/*
------------------------------------------------------------------------------
fm_generate_base_kernel
------------------------------------------------------------------------------
Generates paired g/em bit records with a delay-ordered differential wave in
xor_delta.

Design:
    g bits are sampled from tile/bit-specific base probabilities.
    xor_delta bits are sampled from delay-wave probabilities.
    em = g XOR xor_delta.

This makes the paired-path differential field controllable while preserving
raw g/em bit streams.

Arguments:
    delays[tile]        tile delay metadata
    scales[tile]        tile scale metadata
    theta[tile]         tile theta metadata
    mode_id[tile]       mode id; currently used as small phase offset
------------------------------------------------------------------------------
*/
__global__ void fm_generate_base_kernel(
    const int* delays,
    const int* scales,
    const float* theta,
    const int* mode_id,
    const int tiles,
    const int shots,
    const int bits,
    const unsigned int seed,
    const float base_g,
    const float g_wave_amp,
    const float xor_base,
    const float xor_wave_amp,
    const float xor_bit_skew,
    const float wave_freq,
    const float phase0,
    unsigned char* g,
    unsigned char* em,
    unsigned char* ctrl,
    unsigned char* scale_out,
    unsigned char* branch
) {
    int tile = blockIdx.x;
    int shot = blockIdx.x * blockDim.x + threadIdx.x;

    if (tile >= tiles) return;

    // This kernel is launched as grid=(tiles,), block=(threads,)
    // so shot should come from thread/block-stride within tile.
    for (int s = threadIdx.x; s < shots; s += blockDim.x) {
        int delay = delays[tile];
        int scale = scales[tile];
        float th = theta[tile];
        int mode = mode_id[tile];

        float delay_norm = 0.0f;
        // Fixed normalization chosen for default delay ladder 0..16.
        delay_norm = ((float)delay) / 16.0f;

        float scale_phase = log2f((float)max(1, scale) + 1.0f) * 0.13f;
        float mode_phase = (float)mode * 0.37f;

        // Slow shot phase creates local variation without destroying tile mean.
        float shot_phase = 2.0f * (float)M_PI * ((float)(s & 255)) / 256.0f;

        for (int b = 0; b < bits; ++b) {
            float bit_phase = (float)b * 0.61f;

            // g marginal. Keep this modest and stable.
            float pg =
                base_g
                + g_wave_amp * sinf(2.0f * (float)M_PI * 0.5f * delay_norm + bit_phase + 0.1f * shot_phase)
                + 0.03f * sinf(th + bit_phase);

            pg = fminf(0.95f, fmaxf(0.02f, pg));

            // xor differential wave. This is the gproj target channel.
            float px =
                xor_base
                + xor_wave_amp * sinf(2.0f * (float)M_PI * wave_freq * delay_norm + phase0 + bit_phase + scale_phase + mode_phase)
                + xor_bit_skew * ((b == 1) ? 1.0f : -1.0f)
                + 0.015f * sinf(shot_phase + bit_phase);

            px = fminf(0.95f, fmaxf(0.01f, px));

            float rg = fm_rand01(seed, tile, s, 17U + (unsigned int)b);
            float rx = fm_rand01(seed, tile, s, 71U + (unsigned int)b);

            unsigned char gv = (rg < pg) ? 1U : 0U;
            unsigned char xv = (rx < px) ? 1U : 0U;
            unsigned char ev = (unsigned char)(gv ^ xv);

            int idx = fm_idx3(tile, s, b, shots, bits);
            g[idx] = gv;
            em[idx] = ev;
        }

        // ctrl is a compact readout influenced by path differential and delay.
        float pc =
            0.45f
            + 0.08f * sinf(2.0f * (float)M_PI * wave_freq * delay_norm + phase0 + scale_phase)
            + 0.02f * sinf(th);

        pc = fminf(0.95f, fmaxf(0.05f, pc));
        ctrl[tile * shots + s] = (fm_rand01(seed, tile, s, 211U) < pc) ? 1U : 0U;

        // scale register: not just metadata, but a stable generated channel.
        float ps = 0.50f + 0.10f * sinf(scale_phase + 0.25f * shot_phase);
        ps = fminf(0.95f, fmaxf(0.05f, ps));
        scale_out[tile * shots + s] = (fm_rand01(seed, tile, s, 313U) < ps) ? 1U : 0U;

        // branch two-bit record.
        float pb0 = 0.25f + 0.08f * sinf(th + phase0 + 0.1f * shot_phase);
        float pb1 = 0.25f + 0.08f * cosf(th + scale_phase + 0.1f * shot_phase);
        pb0 = fminf(0.95f, fmaxf(0.02f, pb0));
        pb1 = fminf(0.95f, fmaxf(0.02f, pb1));

        int bidx = (tile * shots + s) * 2;
        branch[bidx + 0] = (fm_rand01(seed, tile, s, 401U) < pb0) ? 1U : 0U;
        branch[bidx + 1] = (fm_rand01(seed, tile, s, 409U) < pb1) ? 1U : 0U;
    }
}

} // extern "C"
"""


def compile_generator_kernel():
    if not _HAVE_CUPY:
        sys.exit("[FATAL] CuPy not available — install cupy-cuda12x or matching CuPy build.")
    try:
        mod = cp.RawModule(
            code=_FM_GPU_GENERATE_KERNEL,
            options=("--std=c++11", "--use_fast_math"),
            name_expressions=["fm_generate_base_kernel"],
        )
        return mod.get_function("fm_generate_base_kernel")
    except Exception as e:
        sys.exit(f"[FATAL] F_M generator kernel compile failed: {e}")


# =============================================================================
# PROJECTOR VERIFY WRAPPER
# =============================================================================

class FMProjectorCUDA:
    def __init__(self, kernel_path: Path):
        if not _HAVE_CUPY:
            raise RuntimeError("CuPy required.")
        if not kernel_path.exists():
            raise FileNotFoundError(kernel_path)

        code = kernel_path.read_text(encoding="utf-8")
        mod = cp.RawModule(
            code=code,
            options=("--std=c++11",),
            name_expressions=[
                "fm_response_kernel_u8",
                "fm_wave_metric_kernel_f32",
            ],
        )
        self.response_kernel = mod.get_function("fm_response_kernel_u8")
        self.wave_kernel = mod.get_function("fm_wave_metric_kernel_f32")

    def compute_responses(
        self,
        g: np.ndarray,
        em: np.ndarray,
        fields: Sequence[str],
        responses: Sequence[str],
    ) -> np.ndarray:
        g_gpu = cp.asarray(np.ascontiguousarray(g, dtype=np.uint8))
        em_gpu = cp.asarray(np.ascontiguousarray(em, dtype=np.uint8))

        tiles, shots, bits = g.shape

        field_ids = cp.asarray(np.asarray([FIELD_KINDS[f] for f in fields], dtype=np.int32))
        response_ids = cp.asarray(np.asarray([RESPONSE_KINDS[r] for r in responses], dtype=np.int32))

        out = cp.zeros((len(fields), len(responses), tiles), dtype=cp.float32)

        self.response_kernel(
            (tiles, len(responses), len(fields)),
            (1,),
            (
                g_gpu,
                em_gpu,
                np.int32(tiles),
                np.int32(shots),
                np.int32(bits),
                field_ids,
                np.int32(len(fields)),
                response_ids,
                np.int32(len(responses)),
                out,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(out)

    def compute_wave_metrics(self, curves: np.ndarray, xs: np.ndarray) -> np.ndarray:
        curves_gpu = cp.asarray(np.ascontiguousarray(curves, dtype=np.float32))
        xs_gpu = cp.asarray(np.ascontiguousarray(xs, dtype=np.float32))
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


def order_by_delay(tile_delay_dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tile_idx = np.arange(tile_delay_dt.size, dtype=np.int32)
    order = np.lexsort((tile_idx, tile_delay_dt))
    xs = tile_delay_dt[order].astype(np.float32)
    return order.astype(np.int32), xs


def verify_projector_signature(
    g: np.ndarray,
    em: np.ndarray,
    tile_delay_dt: np.ndarray,
    verbose: bool = True,
) -> List[dict]:
    if not FM_PROJECTOR_KERNEL.exists():
        if verbose:
            print(f"[warn] F_M projector kernel not found, skipping verify: {FM_PROJECTOR_KERNEL}")
        return []

    projector = FMProjectorCUDA(FM_PROJECTOR_KERNEL)

    fields = ["xor_delta", "delta"]
    responses = ["bit_diff", "bit1_mean", "transition", "energy"]

    response_arr = projector.compute_responses(g, em, fields, responses)

    order, xs = order_by_delay(tile_delay_dt)

    curves = []
    curve_meta = []
    for fi, field in enumerate(fields):
        for ri, response in enumerate(responses):
            curves.append(response_arr[fi, ri, order])
            curve_meta.append({"field": field, "response": response, "order": "delay"})

    curves_np = np.vstack(curves).astype(np.float32)
    metrics = projector.compute_wave_metrics(curves_np, xs)

    rows: List[dict] = []
    for i, cm in enumerate(curve_meta):
        row = dict(cm)
        for mi, name in enumerate(METRIC_NAMES):
            row[name] = float(metrics[i, mi])
        rows.append(row)

    rows.sort(key=lambda r: r["wave_score"], reverse=True)

    if verbose:
        print("\n  Projector signature sanity:")
        for r in rows[:8]:
            print(
                f"    {r['field']:10s} {r['response']:12s} "
                f"score={r['wave_score']:.4f} "
                f"peak={r['peak_ratio']:.3f} "
                f"r2={r['best_r2']:.3f} "
                f"freq={r['best_freq']:.2f} "
                f"amp={r['best_amp']:.5f}"
            )

    return rows


# =============================================================================
# BASE BUILDER
# =============================================================================

def mode_to_id(mode: str) -> int:
    if mode == "clean":
        return 0
    if mode == "phase_shear":
        return 1
    if mode == "local_shock":
        return 2
    return 0


def build_base(
    n_tiles: int,
    n_shots: int,
    bits: int,
    seed: Optional[int],
    tile_plan: Sequence[TileMeta],
    base_g: float,
    g_wave_amp: float,
    xor_base: float,
    xor_wave_amp: float,
    xor_bit_skew: float,
    wave_freq: float,
    phase0: float,
    kernel=None,
) -> Dict[str, Any]:
    if kernel is None:
        kernel = compile_generator_kernel()

    if seed is None:
        seed = secrets.randbits(31)

    delays = np.asarray([m.delay_dt for m in tile_plan], dtype=np.int32)
    scales = np.asarray([m.scale_level for m in tile_plan], dtype=np.int32)
    theta = np.asarray([m.theta for m in tile_plan], dtype=np.float32)
    mode_id = np.asarray([mode_to_id(m.mode) for m in tile_plan], dtype=np.int32)

    d_delays = cp.asarray(delays)
    d_scales = cp.asarray(scales)
    d_theta = cp.asarray(theta)
    d_mode = cp.asarray(mode_id)

    d_g = cp.empty((n_tiles, n_shots, bits), dtype=cp.uint8)
    d_em = cp.empty((n_tiles, n_shots, bits), dtype=cp.uint8)
    d_ctrl = cp.empty((n_tiles, n_shots), dtype=cp.uint8)
    d_scale = cp.empty((n_tiles, n_shots), dtype=cp.uint8)
    d_branch = cp.empty((n_tiles, n_shots, 2), dtype=cp.uint8)

    threads = 256

    kernel(
        (n_tiles,),
        (threads,),
        (
            d_delays,
            d_scales,
            d_theta,
            d_mode,
            np.int32(n_tiles),
            np.int32(n_shots),
            np.int32(bits),
            np.uint32(seed),
            np.float32(base_g),
            np.float32(g_wave_amp),
            np.float32(xor_base),
            np.float32(xor_wave_amp),
            np.float32(xor_bit_skew),
            np.float32(wave_freq),
            np.float32(phase0),
            d_g,
            d_em,
            d_ctrl,
            d_scale,
            d_branch,
        ),
    )
    cp.cuda.Stream.null.synchronize()

    g = cp.asnumpy(d_g)
    em = cp.asnumpy(d_em)
    ctrl = cp.asnumpy(d_ctrl)
    scale = cp.asnumpy(d_scale)
    branch = cp.asnumpy(d_branch)

    delta = em.astype(np.int8) - g.astype(np.int8)
    xor_delta = np.bitwise_xor(em, g).astype(np.uint8)

    tile_theta = np.asarray([m.theta for m in tile_plan], dtype=np.float64)
    tile_delay_dt = np.asarray([m.delay_dt for m in tile_plan], dtype=np.int32)
    tile_scale_level = np.asarray([m.scale_level for m in tile_plan], dtype=np.int32)
    tile_mode = np.asarray([m.mode for m in tile_plan])
    tile_role = np.asarray([m.role for m in tile_plan])
    tile_indices = np.arange(n_tiles, dtype=np.int32)

    tile_meta_list = [asdict(m) for m in tile_plan]

    job_id = f"fm_gpu_seed{seed}"

    data: Dict[str, Any] = {
        "schema": "ghost_oracle.fm.gproj.v1",
        "suite": "Ghost Oracle Suite",
        "operator": "F_M",
        "substrate": "gproj",
        "circuit_family": "paired_delay_cavity_gpu_emulation_v1",
        "job_id": job_id,
        "backend": "gpu",
        "shots": np.int32(n_shots),
        "num_tiles": np.int32(n_tiles),
        "tile_indices": tile_indices,
        "qubits_per_tile": np.int32(9),
        "delays_dt": np.unique(tile_delay_dt).astype(np.int32),
        "scale_levels": np.unique(tile_scale_level).astype(np.int32),
        "tile_theta": tile_theta,
        "tile_delay_dt": tile_delay_dt,
        "tile_scale_level": tile_scale_level,
        "tile_mode": tile_mode,
        "tile_role": tile_role,
        "tile_meta_json": json.dumps(json_safe(tile_meta_list)),
        "generator_meta_json": json.dumps(json_safe({
            "seed": int(seed),
            "bits": int(bits),
            "base_g": float(base_g),
            "g_wave_amp": float(g_wave_amp),
            "xor_base": float(xor_base),
            "xor_wave_amp": float(xor_wave_amp),
            "xor_bit_skew": float(xor_bit_skew),
            "wave_freq": float(wave_freq),
            "phase0": float(phase0),
            "target_signature": DEFAULT_TARGET_FIELD_RESPONSES,
        })),
    }

    for t in range(n_tiles):
        data[f"ctrl_tile{t}"] = ctrl[t]
        data[f"g_tile{t}"] = g[t]
        data[f"em_tile{t}"] = em[t]
        data[f"scale_tile{t}"] = scale[t]
        data[f"branch_tile{t}"] = branch[t]
        data[f"delta_tile{t}"] = delta[t]
        data[f"xor_delta_tile{t}"] = xor_delta[t]

    data["ctrl"] = ctrl
    data["g"] = g
    data["em"] = em
    data["scale"] = scale
    data["branch"] = branch
    data["delta"] = delta
    data["xor_delta"] = xor_delta

    data["_seed"] = int(seed)

    return data


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ghost Oracle Suite — F_M GPU/gproj base builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    p.add_argument("--tiles", type=int, default=None, help="Number of tiles. Defaults to matched QPU tile count or len(delays).")
    p.add_argument("--bits", type=int, default=DEFAULT_BITS)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--match-qpu", default=None, help="Optional F_M qproj .npz to copy tile metadata/layout from.")

    p.add_argument("--delays-dt", type=int, nargs="+", default=None)
    p.add_argument("--scale-levels", type=int, nargs="+", default=None)
    p.add_argument("--theta-values", type=float, nargs="+", default=None)
    p.add_argument("--modes", nargs="+", default=None, choices=["clean", "phase_shear", "local_shock"])

    p.add_argument("--base-g", type=float, default=0.13)
    p.add_argument("--g-wave-amp", type=float, default=0.018)
    p.add_argument("--xor-base", type=float, default=0.11)
    p.add_argument("--xor-wave-amp", type=float, default=0.055)
    p.add_argument("--xor-bit-skew", type=float, default=0.018)
    p.add_argument("--wave-freq", type=float, default=1.30)
    p.add_argument("--phase0", type=float, default=2.02)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not _HAVE_CUPY:
        sys.exit("[FATAL] CuPy required.")

    try:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        gpu_name = "unknown-gpu"

    qpu_meta = None
    if args.match_qpu:
        qpu_meta = load_qpu_metadata(Path(args.match_qpu))

    delays_dt = parse_int_list(args.delays_dt, qpu_meta.get("delays_dt", DEFAULT_DELAYS_DT) if qpu_meta else DEFAULT_DELAYS_DT)
    scale_levels = parse_int_list(args.scale_levels, qpu_meta.get("scale_levels", DEFAULT_SCALE_LEVELS) if qpu_meta else DEFAULT_SCALE_LEVELS)
    theta_values = parse_float_list(args.theta_values, DEFAULT_THETA_VALUES)
    modes = args.modes if args.modes is not None else DEFAULT_MODES

    if qpu_meta and "shape" in qpu_meta:
        q_tiles, q_shots, q_bits = qpu_meta["shape"]
        n_tiles = int(args.tiles if args.tiles is not None else q_tiles)
        n_shots = int(args.shots if args.shots is not None else q_shots)
        bits = int(args.bits if args.bits is not None else q_bits)
    else:
        n_tiles = int(args.tiles if args.tiles is not None else len(delays_dt))
        n_shots = int(args.shots)
        bits = int(args.bits)

    tile_plan = build_tile_plan(
        num_tiles=n_tiles,
        theta_values=theta_values,
        delays_dt=delays_dt,
        scale_levels=scale_levels,
        modes=modes,
        qpu_meta=qpu_meta,
    )

    print(f"\n{'=' * 86}")
    print("  GHOST ORACLE SUITE — F_M GPU / GPROJ BASE BUILDER")
    print(f"{'=' * 86}")
    print(f"  GPU           : {gpu_name}")
    print(f"  Tiles         : {n_tiles}")
    print(f"  Shots         : {n_shots}")
    print(f"  Bits          : {bits}")
    print(f"  Match QPU     : {args.match_qpu or 'no'}")
    print(f"  Delays dt     : {[m.delay_dt for m in tile_plan]}")
    print(f"  Scale levels  : {[m.scale_level for m in tile_plan]}")
    print(f"  Modes         : {[m.mode for m in tile_plan]}")
    print(f"  Wave freq     : {args.wave_freq}")
    print(f"  Phase0        : {args.phase0}")
    print(f"  XOR base/amp  : {args.xor_base} / {args.xor_wave_amp}")
    print("-" * 86)

    print("[BUILD] Compiling F_M GPU generator kernel...")
    kernel = compile_generator_kernel()

    print("[BUILD] Launching F_M GPU generator...")
    t0 = time.perf_counter()
    data = build_base(
        n_tiles=n_tiles,
        n_shots=n_shots,
        bits=bits,
        seed=args.seed,
        tile_plan=tile_plan,
        base_g=args.base_g,
        g_wave_amp=args.g_wave_amp,
        xor_base=args.xor_base,
        xor_wave_amp=args.xor_wave_amp,
        xor_bit_skew=args.xor_bit_skew,
        wave_freq=args.wave_freq,
        phase0=args.phase0,
        kernel=kernel,
    )
    build_seconds = time.perf_counter() - t0

    seed = int(data.pop("_seed"))

    # Projector sanity before writing.
    verify_rows: List[dict] = []
    if args.verify:
        verify_rows = verify_projector_signature(
            g=np.asarray(data["g"], dtype=np.uint8),
            em=np.asarray(data["em"], dtype=np.uint8),
            tile_delay_dt=np.asarray(data["tile_delay_dt"], dtype=np.int32),
            verbose=True,
        )

    if args.out:
        out_path = Path(args.out)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DATA_DIR / f"fm_gpu_data_{n_shots}shots_seed{seed}.npz"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if verify_rows:
        data["verify_projector_json"] = json.dumps(json_safe(verify_rows))

    np.savez_compressed(out_path, **data)

    latest_path = DATA_DIR / "latest_fm_gpu_data.json"
    write_json(
        latest_path,
        {
            "schema": "ghost_oracle.fm.latest_pointer.v1",
            "operator": "F_M",
            "substrate": "gproj",
            "job_id": data["job_id"],
            "backend": "gpu",
            "path": str(out_path),
            "shots": int(n_shots),
            "num_tiles": int(n_tiles),
            "seed": int(seed),
        },
    )

    print("\n  Empirical sanity:")
    print(f"    mean(g)          : {float(np.mean(data['g'])):.6f}")
    print(f"    mean(em)         : {float(np.mean(data['em'])):.6f}")
    print(f"    mean(delta)      : {float(np.mean(data['delta'])):.6f}")
    print(f"    mean(xor_delta)  : {float(np.mean(data['xor_delta'])):.6f}")
    print(f"    build seconds    : {build_seconds:.6f}")

    print(f"\n{'=' * 86}")
    print("  F_M GPU BASE COMPLETE")
    print(f"{'=' * 86}")
    print(f"  Output      : {out_path}")
    print(f"  Latest ptr  : {latest_path}")
    print(f"  Tiles       : {n_tiles}")
    print(f"  Shots       : {n_shots}")
    print(f"  Bits        : {bits}")
    print(f"  Seed        : {seed}")
    print(f"{'=' * 86}\n")


if __name__ == "__main__":
    main()