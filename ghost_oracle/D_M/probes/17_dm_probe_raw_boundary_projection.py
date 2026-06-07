#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
D_M RAW-BOUNDARY PROJECTION PROBE  (GPU-only, kernel-driven)
==============================================================================

This is the kernel-driven rewrite of the D_M / GPT-2 compatibility probe.
There is no CPU fallback. CUDA (CuPy) is required for the D_M side, and
torch + transformers (on CUDA) is required for the GPT-2 reference path.

What changed vs the NumPy version
---------------------------------
Every piece of D_M physics now runs in a CUDA kernel from dm_probe_kernels.cu:

    record path : raw text bits + base -> projected pair (kernel)
                  projected pair -> per-tile connected correlator (kernel)
                  per-tile connected -> per-rung witnesses + E/phase/spec (kernel)

    geo path    : raw text bits -> per-(rung,witness) connected (kernel)
                  data witnesses * analytic geo aperture -> manifold (kernel)

The GPT-2 path keeps torch for the model forward and the raw pre-softmax QK
product (you cannot use flash-attention here: it fuses the softmax and hides
the score matrix the probe needs). The witness reduction that used to be a
triple Python loop over (batch, seq, seq) is now a single vectorized masked
gather on the GPU, handed to CuPy zero-copy via DLPack. No CPU loops.

The only host-side compute that remains is:
  - bit-bank construction (SHA-256 chaining + unpackbits): one-time input
    encoding, inherently serial crypto, ~hundreds of KB, not the timed path.
  - reporting math (cosine / phase distance / score aggregation over a handful
    of per-rung vectors): negligible, not D_M physics.

Architecture (unchanged in intent)
----------------------------------
The same raw text is sent to four independent paths. GPT-2 QK is never used as
D_M input; it is only a free-running comparison product. No softmax.

Float note: kernels are float32, the old reference was float64. Expect ~1e-5
relative agreement, not bit-exact. To validate the kernels, run the old NumPy
probe and this one on identical --text-file / --data-stride / --delay-scale
(the math is deterministic) and diff d_m_projected_manifolds.csv.

Usage
-----
    python d_m_raw_boundary_projection_probe.py \
      --text-file d_m_probe_corpus.txt --split-lines \
      --max-texts 256 --max-length 128 --gpt2-batch-size 16

D_M-only (still GPU, no torch needed):
    python d_m_raw_boundary_projection_probe.py --no-gpt2
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# HARD GPU REQUIREMENT (no CPU fallback)
# =============================================================================

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This probe is GPU-only and requires CuPy.\n"
        "Install the build matching your CUDA toolkit, e.g. `pip install cupy-cuda12x`.\n"
        f"Import error: {exc}"
    )

if cp.cuda.runtime.getDeviceCount() < 1:
    raise SystemExit("No CUDA device visible to CuPy. This probe has no CPU fallback.")


# =============================================================================
# PATHS
# =============================================================================

PROBE_DIR = Path(__file__).resolve().parent
D_M_DIR = PROBE_DIR.parent
DATA_DIR = D_M_DIR / "data"
ANALYSIS_DIR = PROBE_DIR / "analyze"
DEFAULT_KERNEL_FILE = PROBE_DIR.parent / "kernels"/ "dm_projector_kernel.cu"


# =============================================================================
# DEFAULT BASE FILES
# =============================================================================

DEFAULT_QPROJ_NULL = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fm4ihvjngc73aq3ccg.npz"
DEFAULT_QPROJ_BASE = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8flk2jo3njc73f0g560.npz"
DEFAULT_QPROJ_OFFSET = DATA_DIR / "dm_data_bell_listener_cavity_offset_d8fl82bo3njc73f0fgd0.npz"

DEFAULT_GPROJ_NULL = DATA_DIR / "dm_gpu_data_null_4096shots_seed9031229662612491082.npz"
DEFAULT_GPROJ_BASE = DATA_DIR / "dm_gpu_data_base_delay_4096shots_seed2669559634056472362.npz"
DEFAULT_GPROJ_OFFSET = DATA_DIR / "dm_gpu_data_offset_deformed_4096shots_seed6727069190982977623.npz"


# =============================================================================
# CONSTANTS
# =============================================================================

WITNESS_LABELS = ["XY", "YZ", "ZY", "YX"]
WITNESS_TO_INDEX = {x: i for i, x in enumerate(WITNESS_LABELS)}

DEFAULT_BASE_DELAYS = [0, 256, 1024, 4096, 16384]
DEFAULT_NULL_DELAYS = [0, 0, 0, 0, 0]
DEFAULT_OFFSET_DT = 128

GEO_SHOTS = 4096
REDUCE_THREADS = 256  # block size for shot-reduction kernels (power of two)

EPS = 1.0e-12


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DMRecordBase:
    substrate: str
    condition: str
    source: str
    pair: np.ndarray            # (tiles, shots, 2), uint8
    tile_rung: np.ndarray          # (tiles,) int32
    tile_witness: np.ndarray       # (tiles,) int32
    tile_base_delay: np.ndarray    # (tiles,) int32
    tile_offset_delay: np.ndarray  # (tiles,) int32
    tile_total_delay: np.ndarray   # (tiles,) int32
    tiles: int
    shots: int
    n_rungs: int
    max_abs_delay: int
    meta: Dict[str, Any]


@dataclass
class DMProjectedManifold:
    substrate: str
    condition: str
    source: str
    projection_kind: str
    n_rungs: int
    witnesses: np.ndarray
    energy: np.ndarray
    phase: np.ndarray
    specificity: np.ndarray
    meta: Dict[str, Any]


@dataclass
class GPT2FreeManifold:
    model_name: str
    layer: int
    head: int
    center_mode: str
    n_rungs: int
    witnesses: np.ndarray
    energy: np.ndarray
    phase: np.ndarray
    specificity: np.ndarray
    meta: Dict[str, Any]


# =============================================================================
# GENERAL HELPERS (host-side I/O and reporting only)
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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def decode_str_array(arr: Any) -> List[str]:
    a = np.asarray(arr)
    out: List[str] = []
    for x in a.reshape(-1):
        out.append(x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x))
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n <= 0:
        return 0.0
    x = x[:n]
    y = y[:n]
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < EPS or ny < EPS:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def phase_distance_pi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    d = np.abs(aa - bb)
    d = np.minimum(d, math.pi - d)
    return d / math.pi


# =============================================================================
# CUDA KERNEL MODULE
# =============================================================================

class DMKernels:
    """Adapter for ghost_oracle/D_M/kernels/dm_projector_kernel.cu."""

    # Keep these in sync with dm_projector_kernel.cu enums.
    DM_TILE_N_METRICS = 12
    DM_RUNG_XY = 0
    DM_RUNG_YZ = 1
    DM_RUNG_ZY = 2
    DM_RUNG_YX = 3
    DM_RUNG_YZZY_ENERGY = 6
    DM_RUNG_DIRECTIONAL_SPECIFICITY = 8
    DM_RUNG_PI_PHASE = 11
    DM_RUNG_N_METRICS = 19

    def __init__(self, cu_path: Path) -> None:
        if not Path(cu_path).exists():
            raise FileNotFoundError(f"kernel source not found: {cu_path}")
        src = Path(cu_path).read_text(encoding="utf-8")
        self.module = cp.RawModule(code=src, options=("--std=c++11",))

        # dm_projector_kernel.cu exports the general projector API, not the newer
        # dm_probe_kernels.cu API. The Python probe adapts to that layout here:
        #   bits + base pair -> projected pair
        #   projected pair   -> tile_stats
        #   tile_stats       -> rung_stats/manifold
        self._build_pair = self.module.get_function("dm_make_projected_pair_from_bits_kernel_u8")
        self._tile_corr = self.module.get_function("dm_tile_correlator_kernel_u8")
        self._rung_proj = self.module.get_function("dm_rung_projection_kernel_f32")
        self._geo_rung = self.module.get_function("dm_geo_rung_projection_kernel_f32")

    # -- record path ----------------------------------------------------------

    def build_projected_pair(
        self,
        bits_gpu: "cp.ndarray",
        nbits: int,
        base_pair_gpu: "cp.ndarray",
        tile_rung_gpu: "cp.ndarray",
        tile_witness_gpu: "cp.ndarray",
        tile_total_delay_gpu: "cp.ndarray",
        tiles: int,
        shots: int,
        stride: int,
        dscale: int,
    ) -> "cp.ndarray":
        out_pair = cp.empty_like(base_pair_gpu)
        total = int(tiles) * int(shots)
        threads = REDUCE_THREADS
        blocks = (total + threads - 1) // threads
        self._build_pair(
            (blocks,), (threads,),
            (
                bits_gpu, np.int32(nbits), base_pair_gpu,
                tile_rung_gpu, tile_witness_gpu, tile_total_delay_gpu,
                np.int32(tiles), np.int32(shots),
                np.int32(stride), np.int32(dscale),
                out_pair,
            ),
        )
        return out_pair

    def tile_stats(self, pair_gpu: "cp.ndarray", tiles: int, shots: int) -> "cp.ndarray":
        out = cp.empty((tiles, self.DM_TILE_N_METRICS), dtype=cp.float32)
        self._tile_corr(
            (tiles,), (REDUCE_THREADS,),
            (pair_gpu, np.int32(tiles), np.int32(shots), out),
        )
        return out

    def _extract_rung_manifold(
        self,
        rung_stats_gpu: "cp.ndarray",
        n_rungs: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rs = cp.asnumpy(rung_stats_gpu).reshape(n_rungs, self.DM_RUNG_N_METRICS)
        witnesses = rs[:, [self.DM_RUNG_XY, self.DM_RUNG_YZ, self.DM_RUNG_ZY, self.DM_RUNG_YX]].astype(np.float32)
        energy = rs[:, self.DM_RUNG_YZZY_ENERGY].astype(np.float32)
        phase = rs[:, self.DM_RUNG_PI_PHASE].astype(np.float32)
        specificity = rs[:, self.DM_RUNG_DIRECTIONAL_SPECIFICITY].astype(np.float32)
        return witnesses, energy, phase, specificity

    def rung_manifold(
        self,
        tile_stats_gpu: "cp.ndarray",
        tile_rung_gpu: "cp.ndarray",
        tile_witness_gpu: "cp.ndarray",
        tile_base_delay_gpu: "cp.ndarray",
        tile_offset_delay_gpu: "cp.ndarray",
        tile_total_delay_gpu: "cp.ndarray",
        tiles: int,
        n_rungs: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rung_stats = cp.empty((n_rungs, self.DM_RUNG_N_METRICS), dtype=cp.float32)
        self._rung_proj(
            (n_rungs,), (1,),
            (
                tile_stats_gpu,
                tile_rung_gpu, tile_witness_gpu,
                tile_base_delay_gpu, tile_offset_delay_gpu, tile_total_delay_gpu,
                np.int32(tiles), np.int32(n_rungs),
                rung_stats,
            ),
        )
        return self._extract_rung_manifold(rung_stats, n_rungs)

    # -- geo path -------------------------------------------------------------

    def geo_project(
        self,
        n_rungs: int,
        condition_kind: int,
        base_delays_gpu: "cp.ndarray",
        offset_dt: int,
        geo_base_energy: float,
        geo_energy_gain: float,
        geo_comparison_scale: float,
        geo_offset_deform: float,
        geo_phase_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rung_stats = cp.empty((n_rungs, self.DM_RUNG_N_METRICS), dtype=cp.float32)
        self._geo_rung(
            (n_rungs,), (1,),
            (
                np.int32(condition_kind), base_delays_gpu,
                np.int32(n_rungs), np.int32(offset_dt),
                np.float32(geo_base_energy), np.float32(geo_energy_gain),
                np.float32(geo_comparison_scale), np.float32(geo_offset_deform),
                np.float32(geo_phase_scale),
                rung_stats,
            ),
        )
        return self._extract_rung_manifold(rung_stats, n_rungs)

    # -- generic manifold (GPT-2 reference) -----------------------------------

    def witness_manifold(self, witnesses_gpu: "cp.ndarray", n_rungs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # dm_projector_kernel.cu does not expose dm_witness_manifold_f32. This is
        # only for the GPT-2 comparison reference, not for the D_M record path, so
        # keep the tiny final manifold reduction host-side.
        w = cp.asnumpy(witnesses_gpu).reshape(n_rungs, 4).astype(np.float32)
        xy = w[:, 0]
        yz = w[:, 1]
        zy = w[:, 2]
        yx = w[:, 3]
        ret = -zy
        energy = np.sqrt(yz * yz + ret * ret).astype(np.float32)
        comp = np.sqrt(xy * xy + yx * yx).astype(np.float32)
        specificity = (energy - comp).astype(np.float32)
        phase = np.mod(np.arctan2(ret, yz), math.pi).astype(np.float32)
        return energy, phase, specificity

# =============================================================================
# RAW TEXT -> BIT BANK  (host-side input encoding, one-time)
# =============================================================================

def load_texts(args: argparse.Namespace) -> List[str]:
    if args.text_file:
        p = Path(args.text_file)
        if not p.exists():
            raise FileNotFoundError(f"text file not found: {p}")
        raw = p.read_text(encoding="utf-8", errors="replace")
        if args.split_lines:
            texts = [line.strip() for line in raw.splitlines() if line.strip()]
        else:
            chunks = [x.strip() for x in raw.split("\n\n") if x.strip()]
            texts = chunks if chunks else [raw.strip()]
    elif args.text:
        texts = [args.text]
    else:
        texts = [
            "D_M projects raw data through bounded qproj, gproj, and geo operator bases.",
            "GPT-2 is a separate free-running reference path and is not used as D_M input.",
            "The same input data can produce different products under different projection boundaries.",
        ]

    if args.max_texts and args.max_texts > 0:
        texts = texts[: int(args.max_texts)]

    if not texts:
        raise ValueError("no input texts loaded")
    return texts


def make_text_bit_bank(texts: Sequence[str], min_bits: int, salt: str = "D_M_RAW_BOUNDARY") -> np.ndarray:
    """Deterministic bit bank: raw UTF-8 bits, extended by SHA-256 chunks if needed."""
    text = "\n".join(texts)
    raw_bytes = text.encode("utf-8", errors="replace") or b"empty"

    raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bits = np.unpackbits(raw_arr, bitorder="little").astype(np.uint8)

    chunks = [bits]
    total = int(bits.size)
    counter = 0
    seed = hashlib.sha256(salt.encode("utf-8") + b"\0" + raw_bytes).digest()

    while total < min_bits:
        h = hashlib.sha256(seed + counter.to_bytes(8, "little") + raw_bytes).digest()
        h_bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8), bitorder="little").astype(np.uint8)
        chunks.append(h_bits)
        total += int(h_bits.size)
        counter += 1

    return np.concatenate(chunks)[: max(min_bits, total)].astype(np.uint8)


# =============================================================================
# LOAD QPROJ/GPROJ RECORD BASES  (host-side npz I/O)
# =============================================================================

def optional_int_array(z: Any, name: str, tiles: int, default: int = 0) -> np.ndarray:
    if name in z.files:
        arr = np.asarray(z[name], dtype=np.int32)
        if arr.shape[0] == tiles:
            return arr
    return np.full((tiles,), default, dtype=np.int32)


def infer_condition_from_metadata(base: np.ndarray, off: np.ndarray, total: np.ndarray) -> str:
    if base.size == 0 or off.size == 0 or total.size == 0:
        return "unknown"
    if int(np.max(base)) == 0 and int(np.max(off)) == 0 and int(np.max(total)) == 0:
        return "null"
    if int(np.max(base)) > 0 and int(np.max(off)) == 0:
        return "base_only"
    if int(np.max(base)) > 0 and int(np.max(off)) > 0:
        return "offset_on"
    return "unknown"


def load_record_base(path: Path, substrate: str, condition_hint: Optional[str]) -> DMRecordBase:
    if not path.exists():
        raise FileNotFoundError(f"Missing D_M base file: {path}")

    z = np.load(path, allow_pickle=True)

    if "pair" in z.files:
        pair = np.asarray(z["pair"], dtype=np.uint8)
    else:
        pair_keys = sorted(
            [k for k in z.files if k.startswith("pair_tile")],
            key=lambda k: int(k.replace("pair_tile", "")),
        )
        if not pair_keys:
            raise KeyError(f"{path} has no pair or pair_tile* arrays.")
        pair = np.stack([np.asarray(z[k], dtype=np.uint8) for k in pair_keys], axis=0)

    if pair.ndim != 3 or pair.shape[2] != 2:
        raise ValueError(f"{path} pair must have shape (tiles, shots, 2), got {pair.shape}")

    pair = np.ascontiguousarray(pair, dtype=np.uint8)
    tiles = int(pair.shape[0])
    shots = int(pair.shape[1])

    if "tile_rung_index" in z.files:
        tile_rung = np.asarray(z["tile_rung_index"], dtype=np.int32)
    else:
        tile_rung = (np.arange(tiles) // 4).astype(np.int32)

    if "tile_witness_index" in z.files:
        tile_witness = np.asarray(z["tile_witness_index"], dtype=np.int32)
    elif "tile_witness_label" in z.files:
        labels = decode_str_array(z["tile_witness_label"])
        tile_witness = np.asarray(
            [WITNESS_TO_INDEX.get(x, i % 4) for i, x in enumerate(labels[:tiles])],
            dtype=np.int32,
        )
    else:
        tile_witness = (np.arange(tiles) % 4).astype(np.int32)

    tile_base = optional_int_array(z, "tile_base_delay_dt", tiles, 0)
    tile_off = optional_int_array(z, "tile_offset_dt", tiles, 0)
    tile_total = optional_int_array(z, "tile_total_delay_dt", tiles, 0)

    condition = condition_hint or infer_condition_from_metadata(tile_base, tile_off, tile_total)
    n_rungs = int(np.max(tile_rung)) + 1 if tiles else 0
    max_abs_delay = int(np.max(np.abs(tile_total))) if tiles else 0

    meta = {
        "path": str(path),
        "substrate": substrate,
        "condition": condition,
        "tiles": tiles,
        "shots": shots,
        "n_rungs": n_rungs,
        "tile_base_delay_dt": tile_base.tolist(),
        "tile_offset_dt": tile_off.tolist(),
        "tile_total_delay_dt": tile_total.tolist(),
    }

    return DMRecordBase(
        substrate=substrate,
        condition=condition,
        source=str(path),
        pair=pair,
        tile_rung=tile_rung,
        tile_witness=tile_witness,
        tile_base_delay=tile_base,
        tile_offset_delay=tile_off,
        tile_total_delay=tile_total,
        tiles=tiles,
        shots=shots,
        n_rungs=n_rungs,
        max_abs_delay=max_abs_delay,
        meta=meta,
    )


# =============================================================================
# PROJECTION (all GPU)
# =============================================================================

def project_record_base_gpu(
    K: DMKernels,
    bits_gpu: "cp.ndarray",
    nbits: int,
    base: DMRecordBase,
    args: argparse.Namespace,
) -> DMProjectedManifold:
    pair_gpu = cp.asarray(base.pair, dtype=cp.uint8)
    tr = cp.asarray(base.tile_rung, dtype=cp.int32)
    tw = cp.asarray(base.tile_witness, dtype=cp.int32)
    tb = cp.asarray(base.tile_base_delay, dtype=cp.int32)
    toff = cp.asarray(base.tile_offset_delay, dtype=cp.int32)
    td = cp.asarray(base.tile_total_delay, dtype=cp.int32)

    projected = K.build_projected_pair(
        bits_gpu, nbits, pair_gpu, tr, tw, td,
        base.tiles, base.shots, args.data_stride, args.delay_scale,
    )
    tile_stats = K.tile_stats(projected, base.tiles, base.shots)
    w, e, ph, sp = K.rung_manifold(tile_stats, tr, tw, tb, toff, td, base.tiles, base.n_rungs)

    return DMProjectedManifold(
        substrate=base.substrate,
        condition=base.condition,
        source=base.source,
        projection_kind="raw_text_xor_record_base_aperture",
        n_rungs=base.n_rungs,
        witnesses=w,
        energy=e,
        phase=ph,
        specificity=sp,
        meta={
            "base_meta": base.meta,
            "data_stride": args.data_stride,
            "delay_scale": args.delay_scale,
            "note": "Raw text projected independently through qproj/gproj base. GPT-2 not used.",
        },
    )


def project_geo_gpu(
    K: DMKernels,
    bits_gpu: "cp.ndarray",
    nbits: int,
    condition: str,
    args: argparse.Namespace,
) -> DMProjectedManifold:
    if condition == "null":
        base_delays = list(DEFAULT_NULL_DELAYS)
        offset_dt = 0
        condition_kind = 0
    elif condition == "base_only":
        base_delays = list(DEFAULT_BASE_DELAYS)
        offset_dt = 0
        condition_kind = 1
    elif condition == "offset_on":
        base_delays = list(DEFAULT_BASE_DELAYS)
        offset_dt = DEFAULT_OFFSET_DT
        condition_kind = 2
    else:
        raise ValueError(f"unknown geo condition: {condition}")

    n_rungs = len(base_delays)
    bd = cp.asarray(np.asarray(base_delays, dtype=np.int32))

    w, e, ph, sp = K.geo_project(
        n_rungs=n_rungs,
        condition_kind=condition_kind,
        base_delays_gpu=bd,
        offset_dt=offset_dt,
        geo_base_energy=args.geo_base_energy,
        geo_energy_gain=args.geo_energy_gain,
        geo_comparison_scale=args.geo_comparison_scale,
        geo_offset_deform=args.geo_offset_deform,
        geo_phase_scale=args.geo_phase_scale,
    )

    return DMProjectedManifold(
        substrate="geo",
        condition=condition,
        source=f"analytic_geo_{condition}",
        projection_kind="analytic_geo_projector_kernel_aperture",
        n_rungs=n_rungs,
        witnesses=w,
        energy=e,
        phase=ph,
        specificity=sp,
        meta={
            "condition": condition,
            "base_delays": base_delays,
            "offset_dt": offset_dt,
            "geo_base_energy": args.geo_base_energy,
            "geo_energy_gain": args.geo_energy_gain,
            "geo_comparison_scale": args.geo_comparison_scale,
            "geo_offset_deform": args.geo_offset_deform,
            "geo_phase_scale": args.geo_phase_scale,
        },
    )


# =============================================================================
# GPT-2 FREE-RUNNING REFERENCE PATH (torch on GPU, vectorized; no Python loops)
# =============================================================================

def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or str(s).strip() == "":
        return None
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(out))


def _tri_index_cache(seq_len: int, device, cache: Dict[int, Any]) -> Tuple[Any, Any, Any, Any]:
    """Strictly-lower-triangle indices (i>j) plus the probe's comparison shifts."""
    import torch

    if seq_len in cache:
        return cache[seq_len]

    ij = torch.tril_indices(seq_len, seq_len, offset=-1, device=device)
    i = ij[0]
    j = ij[1]
    jcmp = torch.remainder(j + 1, seq_len)
    jcmp = torch.where(jcmp == i, torch.remainder(j + 2, seq_len), jcmp)
    icmp = torch.remainder(i - 1, seq_len)
    icmp = torch.where(icmp == j, torch.remainder(i - 2, seq_len), icmp)

    cache[seq_len] = (i, j, jcmp, icmp)
    return cache[seq_len]


def _center_logits_torch(logits, mode: str):
    """Per-(batch,head) centering over the (S,S) score matrix. logits: (B,H,S,S)."""
    import torch

    if mode == "none":
        return logits
    if mode == "row":
        return logits - logits.mean(dim=-1, keepdim=True)
    if mode == "global":
        return logits - logits.mean(dim=(-2, -1), keepdim=True)
    if mode == "zscore":
        mu = logits.mean(dim=(-2, -1), keepdim=True)
        sd = logits.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return torch.where(sd < EPS, logits - mu, (logits - mu) / sd)
    raise ValueError(f"unknown center mode: {mode}")


def load_gpt2_free_manifolds(
    K: DMKernels,
    texts: Sequence[str],
    model_name: str,
    layers: Optional[List[int]],
    heads: Optional[List[int]],
    center_modes: Sequence[str],
    n_rungs: int,
    max_length: int,
    gpt2_batch_size: int,
) -> List[GPT2FreeManifold]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError("GPT-2 path requires torch + transformers. Use --no-gpt2 to skip.") from e

    if not torch.cuda.is_available():
        raise RuntimeError("GPT-2 path is GPU-only and torch.cuda is not available. Use --no-gpt2.")

    device = "cuda"
    # Keep fp32 matmul for parity with the float64 reference shape. Flip these
    # to True (and/or cast the model to fp16) for more speed at a precision cost.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    gpt2_batch_size = max(1, int(gpt2_batch_size))

    print(f"  loading tokenizer/model: {model_name}")
    print(f"  device               : {device}")
    print(f"  gpt2 batch size      : {gpt2_batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(device)

    n_layer = len(model.transformer.h)
    n_head = int(model.config.n_head)
    n_embd = int(model.config.n_embd)
    head_dim = n_embd // n_head
    inv_sqrt_d = 1.0 / math.sqrt(float(head_dim))

    layer_ids = list(range(n_layer)) if layers is None else [x for x in layers if 0 <= x < n_layer]
    head_ids = list(range(n_head)) if heads is None else [x for x in heads if 0 <= x < n_head]
    head_sel = torch.tensor(head_ids, dtype=torch.long, device=device)

    # accumulators on GPU: per (layer, center_mode) -> sum over pairs of [xy,yz,zy,yx], (Hsel,4)
    sum4: Dict[Tuple[int, str], Any] = {}
    pairs_total: Dict[Tuple[int, str], int] = {}
    for layer in layer_ids:
        for cm in center_modes:
            sum4[(layer, cm)] = torch.zeros((len(head_ids), 4), dtype=torch.float64, device=device)
            pairs_total[(layer, cm)] = 0

    tri_cache: Dict[int, Any] = {}
    total = len(texts)
    n_batches = int(math.ceil(total / gpt2_batch_size))

    for bi in range(n_batches):
        lo = bi * gpt2_batch_size
        hi = min(total, lo + gpt2_batch_size)
        batch_texts = list(texts[lo:hi])
        print(f"  GPT-2 batch {bi + 1}/{n_batches}: texts {lo}:{hi}")

        enc = tokenizer(
            batch_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            out = model.transformer(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False,
            )
        hidden_states = out.hidden_states
        B = int(input_ids.shape[0])
        S = int(input_ids.shape[1])

        if S < 2:
            del out, hidden_states
            continue

        i, j, jcmp, icmp = _tri_index_cache(S, device, tri_cache)
        P = int(i.numel())  # pairs per (b, head)
        pairs_here = B * P

        for layer in layer_ids:
            block = model.transformer.h[layer]
            x = hidden_states[layer]  # NOTE: matches the original probe (no ln_1)

            with torch.no_grad():
                qkv = block.attn.c_attn(x)
                q, k, _v = qkv.split(n_embd, dim=2)
                q = q.view(B, S, n_head, head_dim).permute(0, 2, 1, 3)
                k = k.view(B, S, n_head, head_dim).permute(0, 2, 1, 3)
                logits = torch.matmul(q, k.transpose(-1, -2)) * inv_sqrt_d  # (B,H,S,S), raw pre-softmax

                for cm in center_modes:
                    Lc = _center_logits_torch(logits, cm)
                    # gather the four channels: (B,H,P)
                    yz = Lc[..., i, j]
                    zy = Lc[..., j, i]
                    xy = Lc[..., i, jcmp]
                    yx = Lc[..., j, icmp]
                    # sum over batch and pairs -> (H,)
                    xy_s = xy.sum(dim=(0, 2)).double()
                    yz_s = yz.sum(dim=(0, 2)).double()
                    zy_s = zy.sum(dim=(0, 2)).double()
                    yx_s = yx.sum(dim=(0, 2)).double()
                    vec = torch.stack([xy_s, yz_s, zy_s, yx_s], dim=1)  # (H,4)
                    sum4[(layer, cm)] += vec.index_select(0, head_sel)
                    pairs_total[(layer, cm)] += pairs_here

            del qkv, q, k, _v, logits

        del out, hidden_states, input_ids
        if attention_mask is not None:
            del attention_mask
        torch.cuda.empty_cache()

    # finalize -> witnesses -> manifold via kernel
    manifolds: List[GPT2FreeManifold] = []
    for layer in layer_ids:
        for cm in center_modes:
            denom = max(1, pairs_total[(layer, cm)])
            mean4 = (sum4[(layer, cm)] / float(denom)).detach().cpu().numpy()  # (Hsel,4)
            for hpos, head in enumerate(head_ids):
                vec = mean4[hpos].astype(np.float32)  # [xy,yz,zy,yx]
                tiled = np.tile(vec.reshape(1, 4), (n_rungs, 1)).astype(np.float32)
                wg = cp.asarray(np.ascontiguousarray(tiled).ravel())
                e, ph, sp = K.witness_manifold(wg, n_rungs)
                manifolds.append(
                    GPT2FreeManifold(
                        model_name=model_name,
                        layer=int(layer),
                        head=int(head),
                        center_mode=str(cm),
                        n_rungs=int(n_rungs),
                        witnesses=tiled,
                        energy=e,
                        phase=ph,
                        specificity=sp,
                        meta={
                            "center_mode": cm,
                            "pairs_total": int(pairs_total[(layer, cm)]),
                            "note": "Free-running GPT-2 raw QK reference. Not used as D_M input.",
                        },
                    )
                )
    return manifolds


# =============================================================================
# SCORING + ROW BUILDERS (host-side reporting)
# =============================================================================

def summary_score(m: DMProjectedManifold) -> float:
    energy_mean = float(np.mean(m.energy)) if m.energy.size else 0.0
    spec_mean = float(np.mean(m.specificity)) if m.specificity.size else 0.0
    phase_span = float((np.max(m.phase) - np.min(m.phase)) / math.pi) if m.phase.size else 0.0
    return float(energy_mean + max(0.0, spec_mean) + 0.1 * phase_span)


def active_margin_rows(manifolds: Sequence[DMProjectedManifold]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_sub: Dict[str, Dict[str, DMProjectedManifold]] = {}
    for m in manifolds:
        by_sub.setdefault(m.substrate, {})[m.condition] = m

    for substrate, conds in by_sub.items():
        if "null" not in conds:
            continue
        active = [conds[c] for c in ("base_only", "offset_on") if c in conds]
        if not active:
            continue
        null = conds["null"]
        best = max(active, key=summary_score)

        rows.append({
            "substrate": substrate,
            "best_active_condition": best.condition,
            "active_summary_score": summary_score(best),
            "null_summary_score": summary_score(null),
            "active_score_margin": summary_score(best) - summary_score(null),
            "active_energy_mean": float(np.mean(best.energy)),
            "null_energy_mean": float(np.mean(null.energy)),
            "active_energy_margin": float(np.mean(best.energy) - np.mean(null.energy)),
            "active_specificity_mean": float(np.mean(best.specificity)),
            "null_specificity_mean": float(np.mean(null.specificity)),
            "active_specificity_margin": float(np.mean(best.specificity) - np.mean(null.specificity)),
            "active_phase_span_pi": float((np.max(best.phase) - np.min(best.phase)) / math.pi),
            "null_phase_span_pi": float((np.max(null.phase) - np.min(null.phase)) / math.pi),
        })

    return sorted(rows, key=lambda r: r["active_score_margin"], reverse=True)


def compatibility_rows(gpt2: Sequence[GPT2FreeManifold], dm: Sequence[DMProjectedManifold]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for h in gpt2:
        for m in dm:
            r = min(h.n_rungs, m.n_rungs)
            if r <= 0:
                continue
            w_cos = cosine(h.witnesses[:r], m.witnesses[:r])
            e_cos = cosine(h.energy[:r], m.energy[:r])
            p_mae = float(np.mean(phase_distance_pi(h.phase[:r], m.phase[:r])))
            spec_cos = cosine(h.specificity[:r], m.specificity[:r])
            compat = (
                0.45 * (0.5 * (w_cos + 1.0))
                + 0.25 * (0.5 * (e_cos + 1.0))
                + 0.20 * max(0.0, 1.0 - p_mae)
                + 0.10 * (0.5 * (spec_cos + 1.0))
            )
            rows.append({
                "model_name": h.model_name,
                "layer": h.layer,
                "head": h.head,
                "center_mode": h.center_mode,
                "dm_substrate": m.substrate,
                "dm_condition": m.condition,
                "dm_projection_kind": m.projection_kind,
                "witness_cosine": w_cos,
                "energy_cosine": e_cos,
                "specificity_cosine": spec_cos,
                "phase_mae_pi": p_mae,
                "compatibility_score": float(compat),
                "note": "GPT-2 QK compared after independent D_M raw-data projection; GPT-2 was not D_M input.",
            })
    return sorted(rows, key=lambda r: r["compatibility_score"], reverse=True)


def dm_rows(manifolds: Sequence[DMProjectedManifold]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in manifolds:
        for r in range(m.n_rungs):
            rows.append({
                "substrate": m.substrate, "condition": m.condition, "source": m.source,
                "projection_kind": m.projection_kind, "rung": r,
                "XY": m.witnesses[r, 0], "YZ": m.witnesses[r, 1],
                "ZY": m.witnesses[r, 2], "YX": m.witnesses[r, 3],
                "energy": m.energy[r], "phase": m.phase[r],
                "phase_pi": m.phase[r] / math.pi, "specificity": m.specificity[r],
            })
    return rows


def gpt2_rows(manifolds: Sequence[GPT2FreeManifold]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for h in manifolds:
        for r in range(h.n_rungs):
            rows.append({
                "model_name": h.model_name, "layer": h.layer, "head": h.head,
                "center_mode": h.center_mode, "rung": r,
                "XY": h.witnesses[r, 0], "YZ": h.witnesses[r, 1],
                "ZY": h.witnesses[r, 2], "YX": h.witnesses[r, 3],
                "energy": h.energy[r], "phase": h.phase[r],
                "phase_pi": h.phase[r] / math.pi, "specificity": h.specificity[r],
                "note": "Free GPT-2 raw QK product reference. Not D_M input.",
            })
    return rows


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPU-only kernel-driven D_M raw-boundary projection probe.")

    p.add_argument("--qproj-null", type=Path, default=DEFAULT_QPROJ_NULL)
    p.add_argument("--qproj-base", type=Path, default=DEFAULT_QPROJ_BASE)
    p.add_argument("--qproj-offset", type=Path, default=DEFAULT_QPROJ_OFFSET)
    p.add_argument("--gproj-null", type=Path, default=DEFAULT_GPROJ_NULL)
    p.add_argument("--gproj-base", type=Path, default=DEFAULT_GPROJ_BASE)
    p.add_argument("--gproj-offset", type=Path, default=DEFAULT_GPROJ_OFFSET)

    p.add_argument("--kernel-file", type=Path, default=DEFAULT_KERNEL_FILE)

    p.add_argument("--no-geo", action="store_true")
    p.add_argument("--geo-base-energy", type=float, default=0.04)
    p.add_argument("--geo-energy-gain", type=float, default=0.18)
    p.add_argument("--geo-comparison-scale", type=float, default=0.012)
    p.add_argument("--geo-offset-deform", type=float, default=0.35)
    p.add_argument("--geo-phase-scale", type=float, default=0.37)
    p.add_argument("--no-gpt2", action="store_true")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--gpt2-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--layers", default="", help="Comma-separated GPT-2 layer ids. Default: all.")
    p.add_argument("--heads", default="", help="Comma-separated GPT-2 head ids. Default: all.")
    p.add_argument("--center-modes", default="row", help="Comma-separated centering modes: none,row,global,zscore.")

    p.add_argument("--text", default="")
    p.add_argument("--text-file", default="")
    p.add_argument("--split-lines", action="store_true")
    p.add_argument("--max-texts", type=int, default=0)

    p.add_argument("--data-stride", type=int, default=17)
    p.add_argument("--delay-scale", type=int, default=1)
    p.add_argument("--min-bit-multiplier", type=int, default=8)

    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--top", type=int, default=20)

    return p


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = build_arg_parser().parse_args()
    tag = now_tag()
    out_dir = args.out_dir or (ANALYSIS_DIR / f"dm_probe_17_raw_boundary_probe_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = load_texts(args)
    center_modes = [x.strip() for x in args.center_modes.split(",") if x.strip()] or ["row"]
    layers = parse_int_list(args.layers)
    heads = parse_int_list(args.heads)

    dev = cp.cuda.Device()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode("utf-8", "replace")

    print()
    print("=" * 112)
    print("  D_M RAW-BOUNDARY PROJECTION PROBE  (GPU-only, kernel-driven)")
    print("=" * 112)
    print(f"  CUDA device   : {dev_name}")
    print(f"  Kernel source : {args.kernel_file}")
    print(f"  D_M dir       : {D_M_DIR}")
    print(f"  Output dir    : {out_dir}")
    print(f"  Text sequences: {len(texts)}")
    print(f"  GPT-2 path    : {'disabled' if args.no_gpt2 else args.model}")
    print(f"  Rule          : GPT-2 QK is comparison only; D_M gets raw text bits")
    print(f"  Projection    : data_pair XOR qproj/gproj base; analytic geo via projector kernel")
    print()

    K = DMKernels(args.kernel_file)

    print("[LOAD] qproj/gproj record bases")
    base_specs = [
        ("qproj", "null", args.qproj_null),
        ("qproj", "base_only", args.qproj_base),
        ("qproj", "offset_on", args.qproj_offset),
        ("gproj", "null", args.gproj_null),
        ("gproj", "base_only", args.gproj_base),
        ("gproj", "offset_on", args.gproj_offset),
    ]
    record_bases: List[DMRecordBase] = []
    for substrate, condition, path in base_specs:
        b = load_record_base(path, substrate, condition)
        record_bases.append(b)
        print(f"  {substrate:5s} {condition:9s} rungs={b.n_rungs} tiles={b.tiles} shots={b.shots}  {path}")

    n_rungs = min(int(b.n_rungs) for b in record_bases)
    max_records = max(int(b.tiles * b.shots) for b in record_bases)
    max_delay = max(int(b.max_abs_delay) for b in record_bases)
    min_bits = max(8192, args.min_bit_multiplier * max_records + max_delay + 1024)

    print()
    print("[BUILD] raw text bit bank (host-side input encoding)")
    bit_bank = make_text_bit_bank(texts, min_bits=min_bits)
    bits_gpu = cp.asarray(np.ascontiguousarray(bit_bank, dtype=np.uint8))
    nbits = int(bits_gpu.size)
    print(f"  bits          : {nbits:,}")
    print(f"  data stride   : {args.data_stride}")
    print(f"  delay scale   : {args.delay_scale}")

    # ---- D_M record projection (timed) --------------------------------------
    print()
    print("[PROJECT] raw text through qproj/gproj bases (kernel)")
    dm_manifolds: List[DMProjectedManifold] = []
    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    for b in record_bases:
        m = project_record_base_gpu(K, bits_gpu, nbits, b, args)
        dm_manifolds.append(m)
    cp.cuda.Stream.null.synchronize()
    record_secs = time.time() - t0
    for m in dm_manifolds:
        print(f"  {m.substrate:5s} {m.condition:9s} E_mean={np.mean(m.energy):+.6f} "
              f"S_mean={np.mean(m.specificity):+.6f} summary={summary_score(m):+.6f}")

    total_shots = sum(b.tiles * b.shots for b in record_bases)
    print(f"  record path   : {record_secs:.4f}s  "
          f"({total_shots / max(record_secs, 1e-9):,.0f} shot-pairs/s over {len(record_bases)} bases)")

    # ---- geo projection (timed) ---------------------------------------------
    if not args.no_geo:
        print("[PROJECT] geo analytic apertures (projector kernel)")
        cp.cuda.Stream.null.synchronize()
        t0 = time.time()
        geo_manifolds = [project_geo_gpu(K, bits_gpu, nbits, c, args) for c in ("null", "base_only", "offset_on")]
        cp.cuda.Stream.null.synchronize()
        geo_secs = time.time() - t0
        for m in geo_manifolds:
            print(f"  {'geo':5s} {m.condition:9s} E_mean={np.mean(m.energy):+.6f} "
                  f"S_mean={np.mean(m.specificity):+.6f} summary={summary_score(m):+.6f}")
        print(f"  geo path      : {geo_secs:.4f}s")
        dm_manifolds.extend(geo_manifolds)

    # ---- active vs null margins ---------------------------------------------
    print()
    print("[MARGIN] active-vs-null D_M separation")
    margin_rows = active_margin_rows(dm_manifolds)
    for row in margin_rows:
        print(f"  {row['substrate']:5s} active={row['best_active_condition']:9s} "
              f"score_margin={row['active_score_margin']:+.6f} "
              f"E_margin={row['active_energy_margin']:+.6f} "
              f"S_margin={row['active_specificity_margin']:+.6f}")

    # ---- GPT-2 free reference (timed) ---------------------------------------
    print()
    print("[GPT-2] free-running reference path")
    gpt2_manifolds: List[GPT2FreeManifold] = []
    if args.no_gpt2:
        print("  skipped")
    else:
        t0 = time.time()
        gpt2_manifolds = load_gpt2_free_manifolds(
            K=K, texts=texts, model_name=args.model,
            layers=layers, heads=heads, center_modes=center_modes,
            n_rungs=n_rungs, max_length=args.max_length, gpt2_batch_size=args.gpt2_batch_size,
        )
        print(f"  built GPT-2 free manifolds: {len(gpt2_manifolds)} in {time.time() - t0:.2f}s")

    compat = compatibility_rows(gpt2_manifolds, dm_manifolds) if gpt2_manifolds else []

    # ---- write outputs ------------------------------------------------------
    dm_csv = out_dir / "d_m_projected_manifolds.csv"
    margin_csv = out_dir / "d_m_active_margin_scores.csv"
    gpt2_csv = out_dir / "gpt2_free_qk_manifolds.csv"
    compat_csv = out_dir / "gpt2_vs_dm_compatibility.csv"
    config_json = out_dir / "probe_config.json"

    write_csv(dm_csv, dm_rows(dm_manifolds),
              fields=["substrate", "condition", "source", "projection_kind", "rung",
                      "XY", "YZ", "ZY", "YX", "energy", "phase", "phase_pi", "specificity"])
    write_csv(margin_csv, margin_rows,
              fields=["substrate", "best_active_condition",
                      "active_summary_score", "null_summary_score", "active_score_margin",
                      "active_energy_mean", "null_energy_mean", "active_energy_margin",
                      "active_specificity_mean", "null_specificity_mean", "active_specificity_margin",
                      "active_phase_span_pi", "null_phase_span_pi"])

    if gpt2_manifolds:
        write_csv(gpt2_csv, gpt2_rows(gpt2_manifolds),
                  fields=["model_name", "layer", "head", "center_mode", "rung",
                          "XY", "YZ", "ZY", "YX", "energy", "phase", "phase_pi", "specificity", "note"])
        write_csv(compat_csv, compat,
                  fields=["model_name", "layer", "head", "center_mode",
                          "dm_substrate", "dm_condition", "dm_projection_kind",
                          "witness_cosine", "energy_cosine", "specificity_cosine",
                          "phase_mae_pi", "compatibility_score", "note"])

    write_json(config_json, {
        "created": tag,
        "probe": "d_m_raw_boundary_projection_probe_gpu",
        "device": dev_name,
        "rule": "GPU-only, kernel-driven. GPT-2 QK is comparison only; D_M receives raw text bits independently.",
        "no_softmax": True,
        "d_m_projection": "qproj/gproj projected_pair = raw_data_pair XOR base_pair; geo uses dm_geo_rung_projection_kernel_f32 analytic manifold",
        "normalize_base": False,
        "texts_count": len(texts),
        "n_rungs": n_rungs,
        "bit_bank_size": nbits,
        "data_stride": args.data_stride,
        "delay_scale": args.delay_scale,
        "center_modes": center_modes,
        "record_path_seconds": record_secs,
        "base_paths": {
            "qproj_null": args.qproj_null, "qproj_base": args.qproj_base, "qproj_offset": args.qproj_offset,
            "gproj_null": args.gproj_null, "gproj_base": args.gproj_base, "gproj_offset": args.gproj_offset,
        },
        "outputs": {
            "dm_csv": dm_csv, "margin_csv": margin_csv,
            "gpt2_csv": gpt2_csv if gpt2_manifolds else None,
            "compat_csv": compat_csv if compat else None,
        },
    })

    if compat:
        print()
        print("=" * 112)
        print("  TOP GPT-2 FREE-QK VS INDEPENDENT D_M RAW-BOUNDARY COMPATIBILITY")
        print("=" * 112)
        print("  layer head center  substrate condition   compat  wcos   ecos   phaseMAE")
        print("  " + "-" * 108)
        for row in compat[: max(1, int(args.top))]:
            print(f"  {int(row['layer']):5d} {int(row['head']):4d} {str(row['center_mode']):7s} "
                  f"{str(row['dm_substrate']):8s} {str(row['dm_condition']):9s} "
                  f"{float(row['compatibility_score']):7.3f} "
                  f"{float(row['witness_cosine']):+6.3f} {float(row['energy_cosine']):+6.3f} "
                  f"{float(row['phase_mae_pi']):8.3f}")

    print()
    print("[DONE]")
    print(f"  D_M projected manifolds : {dm_csv}")
    print(f"  D_M active margins      : {margin_csv}")
    if gpt2_manifolds:
        print(f"  GPT-2 free QK manifolds : {gpt2_csv}")
        print(f"  GPT-2/D_M compatibility : {compat_csv}")
    print(f"  Config                  : {config_json}")
    print()


if __name__ == "__main__":
    main()