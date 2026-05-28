#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
AUTO-ORACLE: STREAMLINED CALIBRATION + MIXED RETRIEVAL
==============================================================================
Performs a super-fast in-memory calibration using the C++ megakernel to 
find the top-K (tile, mask) combinations, applies softmax weighting, and 
immediately runs the full retrieval sweep without touching the disk.

--probe runs two negative-control probes that reuse the CALIBRATED winning
component (same tile/mask/angles the sweep uses) so the only thing varied is
the physical shot counts. This answers "is the base load-bearing?" and "is
the task too easy?" using the real proj/geo kernels — no stand-in operator.
==============================================================================
"""

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

try:
    import cupy as cp
    _HAVE_CUPY = True
except Exception:
    cp = None
    _HAVE_CUPY = False

from tqdm import tqdm
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
ANGLE_SCALE = 1.05
ALPHA_NORM = 0.9127
MATRIX_A_ORIG = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B_ORIG = np.array([1.00, 0.80, 0.40, 0.10])
SEED = 42
MAX_VRAM_BYTES = 512 * 1024**2
TOP_K = [1, 5, 10]
MK_TILE = 32

MASK_CONFIGS = [
    ("M1", "Baseline (all 9)",               []),
    ("M2", "CUDA accident (drop 4-8)",       [4, 5, 6, 7, 8]),
    ("M3", "Anti-pillars (0,2)(2,0)",        [2, 6]),
    ("M4", "Drop (0,1)(1,0)",                [1, 3]),
    ("M5", "Drop (1,2)(2,1)",                [5, 7]),
    ("M6", "Drop pillars (0,0)(2,2)",        [0, 8]),
    ("M7", "Pure core (0,0|0,1|1,0|1,1)",    [2, 5, 6, 7, 8]),
    ("M8", "Mirror core (1,1|1,2|2,1|2,2)",  [0, 1, 2, 3, 6]),
]

SWEEPS = {
    "SMALL":  {"M":   50_000, "N": 1024, "NOISE": 0.08, "OUTLIER_FRAC": 0.01, "OUTLIER_MAG":  40.0},
    "MEDIUM": {"M":  250_000, "N": 1024, "NOISE": 0.12, "OUTLIER_FRAC": 0.03, "OUTLIER_MAG":  60.0},
    "LARGE":  {"M": 1_000_000, "N": 1024, "NOISE": 0.18, "OUTLIER_FRAC": 0.05, "OUTLIER_MAG": 100.0},
}

HERE = Path(__file__).resolve().parent

# =============================================================================
# UTILITIES
# =============================================================================
def data_to_angles(data, scale=ANGLE_SCALE):
    return (data / np.max(np.abs(data))) * (math.pi / 2) * scale

ORIG_A = data_to_angles(MATRIX_A_ORIG)
ORIG_B = data_to_angles(MATRIX_B_ORIG)

def get_bitmask(dropped):
    bits = 0b111111111
    for b in dropped:
        bits &= ~(1 << b)
    return bits

def zero_buckets_counts(counts18, dropped):
    arr = counts18.copy()
    for b in dropped:
        arr[b * 2] = 0; arr[b * 2 + 1] = 0
    return arr

def load_base(path):
    d = np.load(path)
    n = int(d["num_tiles"])
    return {t: d[f"ctrl_tile{t}"] for t in range(n)}, {t: d[f"ghost_tile{t}"] for t in range(n)}, n

def build_bucket_counts(ctrl_dict, ghost_dict, n):
    counts = np.zeros((n, 3, 3, 2), dtype=np.int32)
    for t in range(n):
        g = ghost_dict[t]
        a_sum = g[:, 0].astype(np.int32) + g[:, 1].astype(np.int32)
        b_sum = g[:, 2].astype(np.int32) + g[:, 3].astype(np.int32)
        c = ctrl_dict[t]
        for a_b in range(3):
            for b_b in range(3):
                m = (a_sum == a_b) & (b_sum == b_b)
                counts[t, a_b, b_b, 0] = int(((c == 0) & m).sum())
                counts[t, a_b, b_b, 1] = int(((c == 1) & m).sum())
    return counts

def generate_semantic_environment(M, N, dim, num_clusters, noise, outlier_frac, outlier_mag, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(num_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    cluster_ids = rng.integers(0, num_clusters, size=M)
    X_K = centers[cluster_ids]
    X_K += 0.25 * rng.normal(size=(M, dim)).astype(np.float32)
    query_indices = rng.choice(M, size=N, replace=False)
    X_Q = X_K[query_indices].copy()
    X_Q += noise * rng.normal(size=(N, dim)).astype(np.float32)
    X_Q /= np.linalg.norm(X_Q, axis=1, keepdims=True)
    n_outliers = max(1, int(M * outlier_frac))
    outlier_idx = rng.choice(M, size=n_outliers, replace=False)
    bad_dims = rng.integers(0, dim, size=n_outliers)
    X_K /= np.linalg.norm(X_K, axis=1, keepdims=True)
    for i in range(n_outliers):
        X_K[outlier_idx[i], bad_dims[i]] = outlier_mag
    return X_Q.astype(np.float32), X_K.astype(np.float32), query_indices.astype(np.int64)

# =============================================================================
# CORE KERNELS (Cosine, Geo, QProj, Mixed)
# =============================================================================
# (Functions: cosine_retrieval, geo_retrieval_topk, proj_retrieval_topk_single, 
# proj_retrieval_topk_mixed, recall_at_k, mean_reciprocal_rank are identical to your v2)

def cosine_retrieval(X_Q, X_K, topk, max_vram_bytes):
    N, d = X_Q.shape; M = X_K.shape[0]
    XQ = X_Q / (np.linalg.norm(X_Q, axis=1, keepdims=True) + 1e-8)
    XK = X_K / (np.linalg.norm(X_K, axis=1, keepdims=True) + 1e-8)
    d_K = cp.asarray(XK)
    chunk = max(1, max_vram_bytes // (M * 4))
    out = cp.empty((N, topk), dtype=cp.int32)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        d_q = cp.asarray(XQ[i:end])
        s = d_q @ d_K.T
        idx = cp.argpartition(-s, kth=topk - 1, axis=1)[:, :topk]
        partial = cp.take_along_axis(s, idx, axis=1)
        order = cp.argsort(-partial, axis=1)
        out[i:end] = cp.take_along_axis(idx, order, axis=1).astype(cp.int32)
    cp.cuda.Device().synchronize()
    return out, (chunk * M * 4) / (1024**2)

def geo_retrieval_topk(k_geo, X_Q, X_K, topk, max_vram_bytes):
    N, d = X_Q.shape; M = X_K.shape[0]
    d_K = cp.asarray(X_K.reshape(-1))
    chunk = max(1, max_vram_bytes // (M * 4))
    out = cp.empty((N, topk), dtype=cp.int32)
    for i in range(0, N, chunk):
        end = min(i + chunk, N); cur = end - i
        d_Q = cp.asarray(X_Q[i:end].reshape(-1))
        scores = cp.empty((cur, M), dtype=cp.float32)
        k_geo(((M+31)//32, (cur+31)//32, 1), (32, 32, 1), 
              (d_Q, d_K, scores, np.int32(cur), np.int32(M), np.int32(d), np.float32(ALPHA_NORM)))
        idx = cp.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
        out[i:end] = cp.take_along_axis(idx, cp.argsort(-cp.take_along_axis(scores, idx, axis=1), axis=1), axis=1).astype(cp.int32)
    cp.cuda.Device().synchronize()
    return out, (chunk * M * 4) / (1024**2)

def proj_retrieval_topk_single(k_proj, X_Q, X_K, counts18, orig_a, orig_b, mask_bits, topk, max_vram_bytes, silent=False):
    N, d = X_Q.shape; M = X_K.shape[0]
    d_K = cp.asarray(X_K.reshape(-1)); d_counts = cp.asarray(counts18)
    chunk = max(1, max_vram_bytes // (M * 4))
    out = cp.empty((N, topk), dtype=cp.int32)
    iterable = range(0, N, chunk) if silent else tqdm(range(0, N, chunk), desc="    qproj", leave=False)
    for i in iterable:
        end = min(i + chunk, N); cur = end - i
        d_Q = cp.asarray(X_Q[i:end].reshape(-1))
        scores = cp.empty((cur, M), dtype=cp.float32)
        k_proj(((M+31)//32, (cur+31)//32, 1), (32, 32, 1), 
               (d_counts, np.float32(orig_a), np.float32(orig_b), d_Q, d_K, scores, np.int32(cur), np.int32(M), np.int32(d), np.float32(ALPHA_NORM), np.int32(mask_bits)))
        idx = cp.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
        out[i:end] = cp.take_along_axis(idx, cp.argsort(-cp.take_along_axis(scores, idx, axis=1), axis=1), axis=1).astype(cp.int32)
    cp.cuda.Device().synchronize()
    return out, (chunk * M * 4) / (1024**2)

def proj_retrieval_topk_mixed(k_proj, X_Q, X_K, components, topk, max_vram_bytes):
    N, d = X_Q.shape; M = X_K.shape[0]
    d_K = cp.asarray(X_K.reshape(-1))
    comp_state = [{"d_counts": cp.asarray(c["counts18"]), "orig_a": c["orig_a"], "orig_b": c["orig_b"], "mask_bits": c["mask_bits"], "weight": c["weight"]} for c in components]
    chunk = max(1, max_vram_bytes // (M * 4 * 2))
    out = cp.empty((N, topk), dtype=cp.int32)
    for i in tqdm(range(0, N, chunk), desc="    qproj-mixed", leave=False):
        end = min(i + chunk, N); cur = end - i
        d_Q = cp.asarray(X_Q[i:end].reshape(-1))
        combined = cp.zeros((cur, M), dtype=cp.float32)
        temp = cp.empty((cur, M), dtype=cp.float32)
        for cs in comp_state:
            k_proj(((M+31)//32, (cur+31)//32, 1), (32, 32, 1), 
                   (cs["d_counts"], np.float32(cs["orig_a"]), np.float32(cs["orig_b"]), d_Q, d_K, temp, np.int32(cur), np.int32(M), np.int32(d), np.float32(ALPHA_NORM), np.int32(cs["mask_bits"])))
            combined += cs["weight"] * temp
        idx = cp.argpartition(-combined, kth=topk - 1, axis=1)[:, :topk]
        out[i:end] = cp.take_along_axis(idx, cp.argsort(-cp.take_along_axis(combined, idx, axis=1), axis=1), axis=1).astype(cp.int32)
    cp.cuda.Device().synchronize()
    return out, (chunk * M * 8) / (1024**2)

def recall_at_k(topk_indices, truth_indices, k):
    top = topk_indices[:, :k].get() if hasattr(topk_indices, 'get') else topk_indices[:, :k]
    return float(np.mean([int(truth_indices[i] in top[i]) for i in range(top.shape[0])]))

def mean_reciprocal_rank(topk_indices, truth_indices):
    top = topk_indices.get() if hasattr(topk_indices, 'get') else topk_indices
    rr = []
    for row, gt in zip(top, truth_indices):
        found = False
        for rank, idx in enumerate(row, start=1):
            if idx == gt:
                rr.append(1.0 / rank); found = True; break
        if not found: rr.append(0.0)
    return float(np.mean(rr))

def clear_gpu():
    try: cp.get_default_memory_pool().free_all_blocks()
    except Exception: pass

# =============================================================================
# SUPER-FAST IN-MEMORY CALIBRATION
# =============================================================================
def fast_calibrate_mixed(bases_data, k_proj, args):
    """Uses a tiny C++ Megakernel batch to score all masks in RAM instantly."""
    print("\n" + "="*84 + "\n  AUTO-CALIBRATION: IN-MEMORY MIXED STRATEGY\n" + "="*84)
    
    # Generate a tiny, fast evaluation environment
    X_Q_cal, X_K_cal, gt_cal = generate_semantic_environment(
        M=4096, N=128, dim=args.d, num_clusters=32, noise=0.1, outlier_frac=0.05, outlier_mag=50.0)
    
    base_payloads = {}
    for fname, counts, n_tiles in bases_data:
        print(f"  Calibrating {fname}...")
        results = []
        pairs = [(r, c) for r in range(4) for c in range(4)][:n_tiles]
        
        for t in range(n_tiles):
            for tag, name, drops in MASK_CONFIGS:
                mask_bits = get_bitmask(drops)
                counts18 = zero_buckets_counts(counts[t].reshape(-1), drops)
                oa, ob = float(ORIG_A[pairs[t][0]]), float(ORIG_B[pairs[t][1]])
                
                # Fast Megakernel score
                topk_idx, _ = proj_retrieval_topk_single(
                    k_proj, X_Q_cal, X_K_cal, counts18, oa, ob, mask_bits, 1, MAX_VRAM_BYTES, silent=True)
                
                results.append({
                    "tile": t, "mask": tag, "r1": recall_at_k(topk_idx, gt_cal, 1),
                    "oa": oa, "ob": ob, "bits": mask_bits, "counts18": counts18
                })
                
        # Top-K Softmax Logic
        ranked = sorted(results, key=lambda x: -x["r1"])[:args.mixed_k]
        scores = np.array([x["r1"] for x in ranked])
        logits = args.mixed_temp * scores
        weights = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
        
        components = []
        print(f"    Selected: ", end="")
        for p, w in zip(ranked, weights):
            print(f"t{p['tile']}/{p['mask']}={w:.2f}  ", end="")
            components.append({
                "tile_idx": p["tile"], "mask_tag": p["mask"], "mask_bits": p["bits"],
                "orig_a": p["oa"], "orig_b": p["ob"], "weight": float(w), "counts18": p["counts18"]
            })
        print()
        base_payloads[fname] = {"components": components}
    return base_payloads

# =============================================================================
# PROBES (negative controls) — reuse the CALIBRATED winning component.
# The only thing varied across columns is the physical shot counts, so any
# difference is attributable to the base, not to a different tile/mask/angle.
# =============================================================================
def scramble_counts18(counts18, mode, seed=7):
    """Two negative controls that preserve the total shot count but destroy
    the structure the base carries:
        'perm'    — redistribute shots uniformly at random across 18 buckets
        'uniform' — every bucket exactly equal (strongest control)
    A masked bucket (zeroed by the calibrated mask) stays zero so the mask
    geometry is held fixed; only the surviving buckets are scrambled."""
    rng = np.random.default_rng(seed)
    arr = counts18.astype(np.int64).copy()
    live = np.where(arr > 0)[0]
    if live.size == 0:
        return counts18
    total = int(arr.sum())
    out = np.zeros_like(arr)
    if mode == "uniform":
        share = max(total // live.size, 1)
        out[live] = share
    elif mode == "perm":
        draw = rng.multinomial(total, np.full(live.size, 1.0 / live.size))
        out[live] = draw
    else:
        return counts18
    return out.astype(counts18.dtype)


def probe_base_load_bearing(k_proj, base_to_payload, args):
    print("\n" + "="*84 + "\n  PROBE A: IS THE BASE LOAD-BEARING? (negative control)\n" + "="*84)
    print("  Reusing the CALIBRATED winning component per base; only shot counts vary.\n")
    X_Q, X_K, gt = generate_semantic_environment(
        250_000, 1024, args.d, 64, 0.12, 0.03, 60.0, SEED)

    print(f"  {'base':>28} | {'tile/mask':>10} | {'real R@1':>9} | {'perm R@1':>9} | {'uniform R@1':>11}")
    print("  " + "-"*78)
    for fname, payload in base_to_payload.items():
        comp = payload["components"][0]            # the calibration winner
        oa, ob, bits = comp["orig_a"], comp["orig_b"], comp["mask_bits"]
        c_real = comp["counts18"]
        tag = f"t{comp['tile_idx']}/{comp['mask_tag']}"

        def run(cnts):
            idx, _ = proj_retrieval_topk_single(
                k_proj, X_Q, X_K, cnts, oa, ob, bits, 1, MAX_VRAM_BYTES, silent=True)
            return recall_at_k(idx, gt, 1)

        r_real = run(c_real); clear_gpu()
        r_perm = run(scramble_counts18(c_real, "perm")); clear_gpu()
        r_unif = run(scramble_counts18(c_real, "uniform")); clear_gpu()
        print(f"  {fname[:28]:>28} | {tag:>10} | {r_real:>8.2%} | {r_perm:>8.2%} | {r_unif:>10.2%}")
    print("""
  Read: real >> perm and real >> uniform means the physical shot counts are
  doing the work — proj is not silently reducing to geometry. All numbers come
  from proj_megakernel_2d run on the calibrated component; the only difference
  between columns is the counts18 array fed in.
""")


def probe_separation(k_proj, k_geo, base_to_payload, args):
    print("\n" + "="*84 + "\n  PROBE B: SEPARATION SPECTRUM (is the task too easy?)\n" + "="*84)
    fname = next(iter(base_to_payload))
    comp = base_to_payload[fname]["components"][0]   # calibrated winner
    oa, ob, bits, c_real = comp["orig_a"], comp["orig_b"], comp["mask_bits"], comp["counts18"]
    print(f"  base {fname[:40]}  component t{comp['tile_idx']}/{comp['mask_tag']}  d={args.d}\n")
    print(f"  {'outlier_mag':>12} | {'noise':>6} | {'cosine':>8} | {'geo':>8} | {'proj':>8}")
    print("  " + "-"*56)
    for mag, noise in [(0.0, 0.30), (5.0, 0.25), (20.0, 0.20),
                       (40.0, 0.15), (60.0, 0.12), (100.0, 0.10)]:
        X_Q, X_K, gt = generate_semantic_environment(
            250_000, 1024, args.d, 64, noise, 0.05, mag, SEED)
        cos_idx, _ = cosine_retrieval(X_Q, X_K, 1, MAX_VRAM_BYTES); clear_gpu()
        geo_idx, _ = geo_retrieval_topk(k_geo, X_Q, X_K, 1, MAX_VRAM_BYTES); clear_gpu()
        prj_idx, _ = proj_retrieval_topk_single(
            k_proj, X_Q, X_K, c_real, oa, ob, bits, 1, MAX_VRAM_BYTES, silent=True); clear_gpu()
        print(f"  {mag:>12.1f} | {noise:>6.2f} | "
              f"{recall_at_k(cos_idx, gt, 1):>7.2%} | "
              f"{recall_at_k(geo_idx, gt, 1):>7.2%} | "
              f"{recall_at_k(prj_idx, gt, 1):>7.2%}")
    print("""
  Read: at outlier_mag=0 cosine should be competitive — the check that geo/proj
  aren't winning only on the attack. As mag rises cosine falls and the gap
  opens. proj uses the calibrated component, matching the headline sweep.
""")

# =============================================================================
# MAIN RUNNER
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Auto-Oracle: Fast Calibrate & Sweep")
    p.add_argument("--sweep", choices=["SMALL", "MEDIUM", "LARGE", "ALL"], default="MEDIUM")
    p.add_argument("--d", type=int, default=1024)
    p.add_argument("--mixed-k", type=int, default=1)
    p.add_argument("--mixed-temp", type=float, default=10.0)
    p.add_argument("--data", default="data")
    p.add_argument("--kernel", default="ghost_oracle\G_M\kernels\ghost_kernel.cu")
    p.add_argument("--mk-kernel", default="ghost_oracle\G_M\kernels\megakernels_2d.cu")
    p.add_argument("--probe", action="store_true",
                   help="Run load-bearing + separation probes (after calibration) and exit.")
    args = p.parse_args()

    if not _HAVE_CUPY: sys.exit("[FATAL] cupy not available.")

    # 1. Compile Kernels
    try:
        src = Path(args.kernel).read_text() + "\n\n" + Path(args.mk_kernel).read_text()
        mod = cp.RawModule(code=src, options=("-use_fast_math", "-std=c++17"))
        k_geo = mod.get_function("geo_megakernel_2d")
        k_proj = mod.get_function("proj_megakernel_2d")
    except Exception as e:
        sys.exit(f"[FATAL] kernel compile failed: {e}")

    # 2. Load Bases
    data_dir = Path(args.data)
    qpu_files = sorted(data_dir.glob("job_*.npz"))
    if not qpu_files: sys.exit(f"[FATAL] No QPU bases found in {data_dir}")
    
    bases_data = []
    for f in qpu_files:
        ctrl, ghost, n = load_base(str(f))
        counts = build_bucket_counts(ctrl, ghost, n)
        bases_data.append((f.name, counts, n))

    # 3. FAST IN-MEMORY CALIBRATION
    clear_gpu()
    base_to_payload = fast_calibrate_mixed(bases_data, k_proj, args)
    clear_gpu()

    # 3b. PROBES (optional) — use the calibrated component, then exit.
    if args.probe:
        probe_base_load_bearing(k_proj, base_to_payload, args)
        probe_separation(k_proj, k_geo, base_to_payload, args)
        return

    # 4. FULL RETRIEVAL SWEEP
    sweep_keys = list(SWEEPS.keys()) if args.sweep == "ALL" else [args.sweep]
    for name in sweep_keys:
        cfg = SWEEPS[name]
        print("\n" + "="*84 + f"\n  PROBE: {name}  [strategy=mixed]\n" + "="*84)
        
        X_Q, X_K, gt = generate_semantic_environment(
            cfg['M'], cfg['N'], args.d, 64, cfg['NOISE'], cfg['OUTLIER_FRAC'], cfg['OUTLIER_MAG'], SEED)
        
        # Baselines
        print("  cosine baseline")
        t0 = time.perf_counter(); cos_topk, _ = cosine_retrieval(X_Q, X_K, max(TOP_K), MAX_VRAM_BYTES)
        cos_time = time.perf_counter() - t0; cos_r1 = recall_at_k(cos_topk, gt, 1)
        
        print("  geo_megakernel_2d")
        clear_gpu(); t0 = time.perf_counter(); geo_topk, _ = geo_retrieval_topk(k_geo, X_Q, X_K, max(TOP_K), MAX_VRAM_BYTES)
        geo_time = time.perf_counter() - t0; geo_r1 = recall_at_k(geo_topk, gt, 1)

        # Mixed Strategy QProj
        proj_results = []
        for fname, payload in base_to_payload.items():
            clear_gpu()
            t0 = time.perf_counter()
            p_topk, _ = proj_retrieval_topk_mixed(k_proj, X_Q, X_K, payload["components"], max(TOP_K), MAX_VRAM_BYTES)
            proj_results.append((fname, time.perf_counter() - t0, recall_at_k(p_topk, gt, 1)))

        # Output Table
        def short_base_name(name, width=18):
            stem = Path(name).stem
            if len(stem) <= width:
                return stem
            return stem[:width - 1] + "…"

        label_w = 14
        col_w = 18

        headers = ["cosine", "geo"] + [short_base_name(r[0], col_w) for r in proj_results]

        print("\n  " + "-" * (label_w + col_w * len(headers)))
        print(f"  {'metric':<{label_w}}", end="")
        for h in headers:
            print(f"{h:>{col_w}}", end="")
        print()

        print("  " + "-" * (label_w + col_w * len(headers)))

        print(f"  {'Recall@1':<{label_w}}", end="")
        print(f"{cos_r1:>{col_w}.2%}", end="")
        print(f"{geo_r1:>{col_w}.2%}", end="")
        for _, _, r1 in proj_results:
            print(f"{r1:>{col_w}.2%}", end="")
        print()

        print(f"  {'Time (s)':<{label_w}}", end="")
        print(f"{cos_time:>{col_w}.3f}", end="")
        print(f"{geo_time:>{col_w}.3f}", end="")
        for _, t, _ in proj_results:
            print(f"{t:>{col_w}.3f}", end="")
        print()

        print("  " + "-" * (label_w + col_w * len(headers)) + "\n")

if __name__ == "__main__":
    main()
