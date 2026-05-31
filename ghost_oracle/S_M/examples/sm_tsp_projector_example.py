#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
S_M PROJECTOR TSP EXAMPLE — CUDA FIELD-GUIDED 2-OPT TESTBED
===============================================================================

Mini-paper / README-in-a-file
-----------------------------
This example demonstrates the current S_M -> TSP projector path on a large
TSPLIB instance.

The script compares three coordinates:

1. delta_batch
   Classical control coordinate:
       score(i) = -ΔL(i)

2. sm_improve_batch
   Bounded projector spine:
       S_I(i) = 0.5 + 0.5*tanh(-ΔL(i)/scale)

   This preserves the local improvement ordering because tanh is monotonic and
   scale is positive, but it maps the raw improvement into a stable [0,1]
   coordinate suitable for projection, coin weighting, or amplitude encoding.

3. sm_field_batch
   S_M deformation channel:
       rough(i) = |S_I(i)-S_I(i-1)| + |S_I(i+1)-S_I(i)|
       score(i) = S_I(i) + λ*zscore(rough(i))

   This is the actual field perturbation. A small λ can improve the trajectory;
   too much λ over-steers. The diagnostic column rankΔ reports how much the
   field changes the top move ordering relative to raw ΔL.

Default "press play" run
------------------------
Place the TSPLIB file here:

    data/pla85900.tsp

Then run from the repo root:

    python examples/sm_tsp_projector_example.py

Default settings reproduce the current projector comparison style:

    candidate_k   = 128
    passes        = 500
    max_batch     = 32
    policies      = delta_batch, sm_improve_batch, sm_field_batch
    field weights = 0.0001, 0.001, 0.005, 0.01, 0.05

Outputs
-------
    analysis/sm_tsp_projector_<timestamp>/
        result.json
        summary.csv
        routes.csv
        tour_<policy>_fw<weight>.txt

Requirements
------------
    pip install numpy scipy tqdm cupy-cuda12x

Use the CuPy wheel that matches your CUDA installation.

This is not presented as a final exact TSP solver. It is a reproducible
projector testbed: delta is the baseline, sm_improve is the bounded projector
coordinate, and sm_field is the tunable field deformation.
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
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

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    cKDTree = None
    HAVE_SCIPY = False

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:
    tqdm = None
    HAVE_TQDM = False


HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE / "analysis"
DATA_DIR = HERE.parent / "data"


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
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def write_csv(path: Path, rows: List[dict], fields: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_tour(path: Path, tour: Sequence[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for x in tour:
            f.write(f"{int(x)}\n")


def parse_tsplib(path: str | Path) -> np.ndarray:
    coords = []
    on = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("NODE_COORD_SECTION"):
                on = True
                continue
            if on:
                if s.startswith("EOF"):
                    break
                parts = s.split()
                if len(parts) >= 3:
                    coords.append([float(parts[1]), float(parts[2])])
    if not coords:
        raise ValueError(f"No NODE_COORD_SECTION found in {path}")
    return np.asarray(coords, dtype=np.float32)


def euclidean_instance(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((n, 2), dtype=np.float32)


def tour_length(points: np.ndarray, tour: Sequence[int]) -> float:
    t = np.asarray(tour, dtype=np.int64)
    if len(t) >= 2 and t[0] == t[-1]:
        t = t[:-1]
    diff = points[t] - points[np.roll(t, -1)]
    return float(np.sqrt(np.sum(diff * diff, axis=1)).sum())


def closed_tour(tour: Sequence[int]) -> List[int]:
    t = [int(x) for x in tour]
    if t and t[0] != t[-1]:
        t.append(t[0])
    return t


def validate_tour(tour: Sequence[int], n: int, closed: bool = False) -> Tuple[bool, str]:
    t = list(map(int, tour))
    if closed:
        if len(t) != n + 1:
            return False, f"closed length {len(t)} != {n+1}"
        if t[0] != t[-1]:
            return False, "closed tour does not return to start"
        t = t[:-1]
    else:
        if len(t) >= 2 and t[0] == t[-1]:
            t = t[:-1]
        if len(t) != n:
            return False, f"length {len(t)} != n={n}"
    if len(set(t)) != n:
        return False, "duplicate/missing city"
    if min(t) < 0 or max(t) >= n:
        return False, "city out of range"
    return True, "ok"


def canonical_anchor(tour: Sequence[int]) -> List[int]:
    t = list(map(int, tour))
    if len(t) >= 2 and t[0] == t[-1]:
        t = t[:-1]
    i0 = t.index(0)
    return t[i0:] + t[:i0]


def reverse_tour(tour: Sequence[int]) -> List[int]:
    return [0] + list(reversed(list(tour)[1:]))


def is_exact_hit(tour: Sequence[int], opt: Sequence[int]) -> bool:
    a = canonical_anchor(tour)
    b = canonical_anchor(opt)
    return a == b or a == reverse_tour(b)


def held_karp_exact(points: np.ndarray) -> Tuple[float, List[int]]:
    n = len(points)
    D = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))
    size = 1 << (n - 1)
    INF = 1e100
    dp = [[INF] * n for _ in range(size)]
    prev = [[-1] * n for _ in range(size)]

    for j in range(1, n):
        m = 1 << (j - 1)
        dp[m][j] = float(D[0, j])
        prev[m][j] = 0

    for mask in range(size):
        for j in range(1, n):
            if not (mask & (1 << (j - 1))):
                continue
            pm = mask ^ (1 << (j - 1))
            if pm == 0:
                continue
            sub = pm
            while sub:
                bit = sub & -sub
                k = bit.bit_length()
                c = dp[pm][k] + float(D[k, j])
                if c < dp[mask][j]:
                    dp[mask][j] = c
                    prev[mask][j] = k
                sub ^= bit

    full = size - 1
    best = INF
    end = -1
    for j in range(1, n):
        c = dp[full][j] + float(D[j, 0])
        if c < best:
            best = c
            end = j

    stack = []
    mask = full
    j = end
    while j != 0:
        stack.append(j)
        pj = prev[mask][j]
        mask ^= 1 << (j - 1)
        j = pj

    return best, [0] + list(reversed(stack))


def brute_nearest_unvisited(points: np.ndarray, current: int, visited: np.ndarray) -> int:
    ids = np.where(~visited)[0]
    ids = ids[ids != current]
    if ids.size == 0:
        return -1
    d = points[ids] - points[current]
    return int(ids[np.argmin(np.sum(d * d, axis=1))])


def kd_nearest_unvisited(points: np.ndarray, tree: Any, current: int, visited: np.ndarray, k0: int) -> int:
    n = len(points)
    k = min(max(2, int(k0)), n)
    while True:
        _, idx = tree.query(points[current], k=k)
        for j in np.atleast_1d(idx):
            jj = int(j)
            if jj != current and not visited[jj]:
                return jj
        if k >= n:
            break
        k = min(n, k * 2)
    return brute_nearest_unvisited(points, current, visited)


def nearest_construct(points: np.ndarray, start: int, candidate_k: int, progress: bool) -> List[int]:
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    tour = [int(start)]
    visited[int(start)] = True
    current = int(start)
    tree = cKDTree(points) if HAVE_SCIPY and n > 200 else None

    it = range(n - 1)
    if progress and HAVE_TQDM:
        it = tqdm(it, desc="construct", unit="city")

    for _ in it:
        nxt = kd_nearest_unvisited(points, tree, current, visited, candidate_k) if tree is not None else brute_nearest_unvisited(points, current, visited)
        if nxt < 0:
            break
        tour.append(nxt)
        visited[nxt] = True
        current = nxt

    ok, msg = validate_tour(tour, n)
    if not ok:
        raise RuntimeError(f"construction invalid: {msg}")
    return tour


def choose_starts(points: np.ndarray, starts: int, seed: int) -> List[int]:
    n = len(points)
    out = [0]
    if starts <= 1:
        return out
    xs, ys = points[:, 0], points[:, 1]
    for c in [int(np.argmin(xs)), int(np.argmax(xs)), int(np.argmin(ys)), int(np.argmax(ys))]:
        if c not in out:
            out.append(c)
        if len(out) >= starts:
            return out
    rng = np.random.default_rng(seed)
    while len(out) < starts:
        c = int(rng.integers(0, n))
        if c not in out:
            out.append(c)
    return out


def multi_start_construct(points: np.ndarray, starts: int, candidate_k: int, seed: int, progress: bool) -> Tuple[List[int], List[dict]]:
    best_tour = None
    best_len = float("inf")
    rows = []
    for s in choose_starts(points, starts, seed):
        t0 = time.time()
        tour = nearest_construct(points, s, candidate_k, progress)
        L = tour_length(points, tour)
        rows.append({"start": s, "length": L, "seconds": time.time() - t0})
        if L < best_len:
            best_len = L
            best_tour = tour
    assert best_tour is not None
    return best_tour, rows


def precompute_neighbors(points: np.ndarray, candidate_k: int) -> np.ndarray:
    n = len(points)
    k = min(max(2, int(candidate_k) + 1), n)
    if HAVE_SCIPY:
        tree = cKDTree(points)
        _, idx = tree.query(points, k=k)
        rows = [[int(x) for x in np.atleast_1d(idx[i]) if int(x) != i][:candidate_k] for i in range(n)]
        width = max(len(r) for r in rows)
        out = np.full((n, width), -1, dtype=np.int32)
        for i, r in enumerate(rows):
            out[i, :len(r)] = r
        return out
    D2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)
    return np.argsort(D2, axis=1)[:, 1:candidate_k + 1].astype(np.int32)


def reverse_segment_inplace(tour: np.ndarray, pos: np.ndarray, lo: int, hi: int) -> None:
    tour[lo:hi + 1] = tour[lo:hi + 1][::-1]
    for p in range(lo, hi + 1):
        pos[int(tour[p])] = p


CUDA_SRC = r'''
extern "C" __global__
void best_2opt_moves(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const int* __restrict__ tour,
    const int* __restrict__ pos,
    const int* __restrict__ neigh,
    const int n,
    const int kmax,
    float* __restrict__ best_delta,
    int* __restrict__ best_lo,
    int* __restrict__ best_hi
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int a = tour[i];
    int b = tour[(i + 1) % n];

    float ax = x[a], ay = y[a];
    float bx = x[b], by = y[b];

    float abx = ax - bx;
    float aby = ay - by;
    float ab = sqrtf(abx * abx + aby * aby);

    float bd = 0.0f;
    int blo = -1;
    int bhi = -1;

    for (int kk = 0; kk < kmax; ++kk) {
        int c = neigh[a * kmax + kk];
        if (c < 0) continue;

        int j = pos[c];
        if (j < 0 || j >= n) continue;

        if (i == j) continue;
        if (((i + 1) % n) == j) continue;
        if (((j + 1) % n) == i) continue;

        int lo = i + 1;
        int hi = j;

        if (lo > hi) continue;

        int d = tour[(j + 1) % n];

        float cx = x[c], cy = y[c];
        float dx = x[d], dy = y[d];

        float cd_x = cx - dx;
        float cd_y = cy - dy;
        float cd = sqrtf(cd_x * cd_x + cd_y * cd_y);

        float ac_x = ax - cx;
        float ac_y = ay - cy;
        float ac = sqrtf(ac_x * ac_x + ac_y * ac_y);

        float bd_x = bx - dx;
        float bd_y = by - dy;
        float bd2 = sqrtf(bd_x * bd_x + bd_y * bd_y);

        float delta = ac + bd2 - ab - cd;

        if (delta < bd) {
            bd = delta;
            blo = lo;
            bhi = hi;
        }
    }

    best_delta[i] = bd;
    best_lo[i] = blo;
    best_hi[i] = bhi;
}
'''


class CudaBestMoves:
    def __init__(self, points: np.ndarray, neighbors: np.ndarray):
        if not HAVE_CUPY:
            raise RuntimeError("CuPy not installed. Install cupy-cuda12x or the wheel matching your CUDA version.")
        points = np.asarray(points, dtype=np.float32)
        neighbors = np.asarray(neighbors, dtype=np.int32)
        self.n = int(points.shape[0])
        self.kmax = int(neighbors.shape[1])

        self.x_gpu = cp.asarray(points[:, 0], dtype=cp.float32)
        self.y_gpu = cp.asarray(points[:, 1], dtype=cp.float32)
        self.neigh_gpu = cp.asarray(neighbors.reshape(-1), dtype=cp.int32)

        self.best_delta_gpu = cp.empty(self.n, dtype=cp.float32)
        self.best_lo_gpu = cp.empty(self.n, dtype=cp.int32)
        self.best_hi_gpu = cp.empty(self.n, dtype=cp.int32)
        self.kernel = cp.RawKernel(CUDA_SRC, "best_2opt_moves")

    def eval(self, tour: np.ndarray, pos: np.ndarray, threads: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tour_gpu = cp.asarray(tour.astype(np.int32), dtype=cp.int32)
        pos_gpu = cp.asarray(pos.astype(np.int32), dtype=cp.int32)
        blocks = (self.n + threads - 1) // threads
        self.kernel(
            (blocks,),
            (threads,),
            (
                self.x_gpu,
                self.y_gpu,
                tour_gpu,
                pos_gpu,
                self.neigh_gpu,
                np.int32(self.n),
                np.int32(self.kmax),
                self.best_delta_gpu,
                self.best_lo_gpu,
                self.best_hi_gpu,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(self.best_delta_gpu), cp.asnumpy(self.best_lo_gpu), cp.asnumpy(self.best_hi_gpu)


def compute_scores(policy: str, delta: np.ndarray, field_weight: float) -> Tuple[np.ndarray, Dict[str, float]]:
    improving = delta < -1e-12
    if not np.any(improving):
        return np.zeros_like(delta), {"rank_diff_top20": 0.0, "rough_mean": 0.0, "candidate_count": 0}

    scale = float(np.median(np.abs(delta[improving])) + np.std(delta[improving]) + 1e-12)
    sm = 0.5 + 0.5 * np.tanh(-delta / scale)

    S = np.full_like(sm, float(np.median(sm[improving])))
    S[improving] = sm[improving]
    rough = np.abs(S - np.roll(S, 1)) + np.abs(np.roll(S, -1) - S)
    rough_z = (rough - rough.mean()) / (rough.std() + 1e-12)

    if policy == "delta_batch":
        score = -delta
    elif policy == "sm_improve_batch":
        score = sm
    elif policy == "sm_field_batch":
        score = sm + float(field_weight) * rough_z
    else:
        raise ValueError(policy)

    score = score.copy()
    score[~improving] = -1e30

    delta_order = np.argsort(delta)
    score_order = np.argsort(-score)
    top = min(20, len(delta))
    dtop = set(int(x) for x in delta_order[:top] if improving[x])
    stop = set(int(x) for x in score_order[:top] if improving[x])
    rank_diff = 1.0 - len(dtop & stop) / max(1, len(dtop | stop)) if (dtop or stop) else 0.0

    return score, {
        "rank_diff_top20": float(rank_diff),
        "rough_mean": float(np.mean(rough[improving])),
        "candidate_count": int(np.sum(improving)),
    }


def select_batch(delta: np.ndarray, lo: np.ndarray, hi: np.ndarray, score: np.ndarray, max_batch: int) -> List[Tuple[int, int, int, float, float]]:
    order = np.argsort(-score)
    n = len(delta)
    occupied = np.zeros(n, dtype=bool)
    selected: List[Tuple[int, int, int, float, float]] = []

    for idx in order:
        idx = int(idx)
        if len(selected) >= max_batch:
            break
        if score[idx] < -1e20:
            break
        if delta[idx] >= -1e-12:
            continue
        l = int(lo[idx])
        h = int(hi[idx])
        if l < 0 or h < l:
            continue
        if np.any(occupied[l:h + 1]):
            continue
        selected.append((idx, l, h, float(delta[idx]), float(score[idx])))
        occupied[l:h + 1] = True

    return selected


def apply_selected(tour: np.ndarray, selected: List[Tuple[int, int, int, float, float]]) -> None:
    pos = np.empty(len(tour), dtype=np.int32)
    for p, c in enumerate(tour):
        pos[int(c)] = p
    for _, lo, hi, _, _ in sorted(selected, key=lambda x: x[1], reverse=True):
        reverse_segment_inplace(tour, pos, lo, hi)


def rollout_cuda_policy(
    points: np.ndarray,
    init_tour: Sequence[int],
    engine: CudaBestMoves,
    policy: str,
    field_weight: float,
    passes: int,
    max_batch: int,
    validate_every: int,
    progress: bool,
) -> Tuple[List[int], dict]:
    n = len(points)
    tour = np.asarray(init_tour, dtype=np.int32).copy()
    ok, msg = validate_tour(tour.tolist(), n)
    if not ok:
        raise ValueError(f"bad init tour: {msg}")

    total_gain = 0.0
    total_selected = 0
    rank_diffs = []
    pass_rows = []

    it = range(int(passes))
    if progress and HAVE_TQDM:
        it = tqdm(it, desc=f"{policy} fw={field_weight:g}", unit="pass")

    for p in it:
        pos = np.empty(n, dtype=np.int32)
        for i, c in enumerate(tour):
            pos[int(c)] = int(i)

        delta, lo, hi = engine.eval(tour, pos)
        score, meta = compute_scores(policy, delta, field_weight)
        selected = select_batch(delta, lo, hi, score, max_batch=max_batch)

        if not selected:
            pass_rows.append({"pass": p + 1, "selected": 0, "gain": 0.0, **meta})
            break

        gain = -float(sum(x[3] for x in selected))
        apply_selected(tour, selected)

        total_gain += gain
        total_selected += len(selected)
        rank_diffs.append(meta["rank_diff_top20"])
        pass_rows.append({
            "pass": p + 1,
            "selected": int(len(selected)),
            "gain": gain,
            "mean_delta": float(np.mean([x[3] for x in selected])),
            "mean_score": float(np.mean([x[4] for x in selected])),
            **meta,
        })

        if validate_every > 0 and (p + 1) % validate_every == 0:
            ok, msg = validate_tour(tour.tolist(), n)
            if not ok:
                raise RuntimeError(f"invalid tour after pass {p+1}: {msg}")

    ok, msg = validate_tour(tour.tolist(), n)
    if not ok:
        raise RuntimeError(f"final tour invalid: {msg}")

    return tour.tolist(), {
        "passes_run": len(pass_rows),
        "total_gain": float(total_gain),
        "total_selected": int(total_selected),
        "mean_selected_per_pass": float(total_selected / max(1, len(pass_rows))),
        "mean_rank_diff_top20": float(np.mean(rank_diffs)) if rank_diffs else 0.0,
        "pass_rows": pass_rows,
    }


ROUTE_FIELDS = [
    "route", "policy", "field_weight", "initial_length", "final_length",
    "improvement", "improvement_pct", "passes_run", "total_selected",
    "mean_selected_per_pass", "mean_rank_diff_top20", "seconds",
    "known_opt", "gap_pct", "hit",
]
SUMMARY_FIELDS = [
    "policy", "field_weight", "runs", "mean_final_length", "median_final_length",
    "mean_improvement_pct", "mean_selected_per_pass", "mean_rank_diff_top20",
    "mean_seconds", "mean_gap_pct", "median_gap_pct", "best_gap_pct", "hit_rate",
]


def summarize(rows: List[dict]) -> List[dict]:
    keys = sorted(set((r["policy"], r["field_weight"]) for r in rows))
    out = []
    for policy, fw in keys:
        rr = [r for r in rows if r["policy"] == policy and r["field_weight"] == fw]
        item = {
            "policy": policy,
            "field_weight": fw,
            "runs": len(rr),
            "mean_final_length": float(np.mean([r["final_length"] for r in rr])),
            "median_final_length": float(np.median([r["final_length"] for r in rr])),
            "mean_improvement_pct": float(np.mean([r["improvement_pct"] for r in rr])),
            "mean_selected_per_pass": float(np.mean([r["mean_selected_per_pass"] for r in rr])),
            "mean_rank_diff_top20": float(np.mean([r["mean_rank_diff_top20"] for r in rr])),
            "mean_seconds": float(np.mean([r["seconds"] for r in rr])),
        }
        if "gap_pct" in rr[0]:
            item.update({
                "mean_gap_pct": float(np.mean([r["gap_pct"] for r in rr])),
                "median_gap_pct": float(np.median([r["gap_pct"] for r in rr])),
                "best_gap_pct": float(np.min([r["gap_pct"] for r in rr])),
            })
            # Large TSPLIB runs usually have a known optimum length but not an
            # exact reference tour. In that case gap_pct exists but hit does not.
            hits = [r["hit"] for r in rr if "hit" in r]
            if hits:
                item["hit_rate"] = float(np.mean(hits))
        out.append(item)
    if out and "mean_gap_pct" in out[0]:
        out.sort(key=lambda x: (x["mean_gap_pct"], -x.get("hit_rate", 0.0)))
    else:
        out.sort(key=lambda x: x["mean_final_length"])
    return out


def print_summary(summary: List[dict]) -> None:
    print("\n" + "=" * 118)
    print("  SM_FIELD_TSP_CUDA SUMMARY")
    print("=" * 118)
    has_gap = bool(summary) and "mean_gap_pct" in summary[0]
    if has_gap:
        has_hit = "hit_rate" in summary[0]
        if has_hit:
            print(f"  {'rank':>4} | {'policy':<18} | {'fw':>6} | {'mean gap%':>10} | {'median%':>9} | {'hit':>7} | {'imp%':>8} | {'sel/pass':>8} | {'rankΔ':>7}")
        else:
            print(f"  {'rank':>4} | {'policy':<18} | {'fw':>6} | {'mean gap%':>10} | {'median%':>9} | {'final length':>14} | {'imp%':>8} | {'sel/pass':>8} | {'rankΔ':>7}")
    else:
        print(f"  {'rank':>4} | {'policy':<18} | {'fw':>6} | {'final length':>14} | {'imp%':>8} | {'sel/pass':>8} | {'rankΔ':>7} | {'sec':>8}")
    print("  " + "-" * 116)
    for i, s in enumerate(summary, 1):
        if has_gap:
            if "hit_rate" in s:
                print(
                    f"  {i:>4} | {s['policy']:<18} | {s['field_weight']:>6.3f} | "
                    f"{s['mean_gap_pct']:>10.4f} | {s['median_gap_pct']:>9.4f} | "
                    f"{s['hit_rate']:>7.3f} | {s['mean_improvement_pct']:>8.3f} | "
                    f"{s['mean_selected_per_pass']:>8.2f} | {s['mean_rank_diff_top20']:>7.3f}"
                )
            else:
                print(
                    f"  {i:>4} | {s['policy']:<18} | {s['field_weight']:>6.3f} | "
                    f"{s['mean_gap_pct']:>10.4f} | {s['median_gap_pct']:>9.4f} | "
                    f"{s['mean_final_length']:>14.3f} | {s['mean_improvement_pct']:>8.3f} | "
                    f"{s['mean_selected_per_pass']:>8.2f} | {s['mean_rank_diff_top20']:>7.3f}"
                )
        else:
            print(
                f"  {i:>4} | {s['policy']:<18} | {s['field_weight']:>6.3f} | "
                f"{s['mean_final_length']:>14.3f} | {s['mean_improvement_pct']:>8.3f} | "
                f"{s['mean_selected_per_pass']:>8.2f} | {s['mean_rank_diff_top20']:>7.3f} | "
                f"{s['mean_seconds']:>8.2f}"
            )


def run_policy_set(
    points: np.ndarray,
    init: List[int],
    neighbors: np.ndarray,
    policies: Sequence[str],
    field_weights: Sequence[float],
    args: argparse.Namespace,
    opt_len: Optional[float] = None,
    opt_tour: Optional[List[int]] = None,
) -> Tuple[List[dict], Dict[str, List[int]], Dict[str, dict]]:
    rows: List[dict] = []
    tours: Dict[str, List[int]] = {}
    stats_by_key: Dict[str, dict] = {}

    engine = CudaBestMoves(points, neighbors)
    init_len = tour_length(points, init)

    for policy in policies:
        weights = field_weights if policy == "sm_field_batch" else [0.0]
        for fw in weights:
            key = f"{policy}_fw{fw:g}"
            t0 = time.time()
            tour, stats = rollout_cuda_policy(
                points,
                init,
                engine,
                policy=policy,
                field_weight=float(fw),
                passes=args.passes,
                max_batch=args.max_batch,
                validate_every=args.validate_every,
                progress=not args.no_progress,
            )
            seconds = time.time() - t0
            L = tour_length(points, tour)
            row = {
                "route": 0,
                "policy": policy,
                "field_weight": float(fw),
                "initial_length": init_len,
                "final_length": L,
                "improvement": init_len - L,
                "improvement_pct": (init_len - L) / init_len * 100.0,
                "passes_run": stats["passes_run"],
                "total_selected": stats["total_selected"],
                "mean_selected_per_pass": stats["mean_selected_per_pass"],
                "mean_rank_diff_top20": stats["mean_rank_diff_top20"],
                "seconds": seconds,
            }
            if opt_len is not None:
                row["opt_len"] = opt_len
                row["gap_pct"] = (L - opt_len) / opt_len * 100.0
                row["hit"] = int(is_exact_hit(tour, opt_tour or []))
            rows.append(row)
            tours[key] = tour
            stats_by_key[key] = stats
    return rows, tours, stats_by_key


def validate_small(args: argparse.Namespace, out_dir: Path) -> None:
    rows = []
    for r in range(args.routes):
        points = euclidean_instance(args.N, args.seed + r)
        opt_len, opt_tour = held_karp_exact(points)
        init, _ = multi_start_construct(points, 1, args.candidate_k, args.seed + r, progress=False)
        neighbors = precompute_neighbors(points, args.candidate_k)
        rr, _, _ = run_policy_set(
            points,
            init,
            neighbors,
            policies=["delta_batch", "sm_improve_batch", "sm_field_batch"],
            field_weights=args.field_weights,
            args=args,
            opt_len=opt_len,
            opt_tour=opt_tour,
        )
        for row in rr:
            row["route"] = r
        rows.extend(rr)
        if r % max(1, args.routes // 10) == 0 or r == args.routes - 1:
            best = min(rr, key=lambda x: x["gap_pct"])
            print(f"  route {r:>4}/{args.routes-1}: best={best['policy']} fw={best['field_weight']} gap={best['gap_pct']:.4f}%")
    summary = summarize(rows)
    print_summary(summary)
    write_csv(out_dir / "validation.csv", rows, ROUTE_FIELDS)
    write_csv(out_dir / "summary.csv", summary, SUMMARY_FIELDS)
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({"args": vars(args), "rows": rows, "summary": summary}), f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S_M projector TSP CUDA example. Defaults to data/pla85900.tsp.")
    p.add_argument("--tsp-file", default=str(DATA_DIR / "pla85900.tsp"), help="TSPLIB .tsp file. Default expects data/pla85900.tsp.")
    p.add_argument("--known-opt", type=float, default=142382641.0, help="Known/reference optimum. Set 0 to disable gap reporting.")
    p.add_argument("--validate-small", action="store_true")
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--routes", type=int, default=100)
    p.add_argument("--seed", type=int, default=8008135)
    p.add_argument("--starts", type=int, default=1)
    p.add_argument("--candidate-k", type=int, default=128)
    p.add_argument("--passes", type=int, default=500)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--policy", choices=["delta_batch", "sm_improve_batch", "sm_field_batch"], default="sm_field_batch")
    p.add_argument("--field-weight", type=float, default=0.001)
    p.add_argument("--field-weights", type=float, nargs="+", default=[0.0001, 0.001, 0.005, 0.01, 0.05])
    p.add_argument("--compare", action="store_true", default=True, help="Run delta, sm_improve, and sm_field sweep. Enabled by default.")
    p.add_argument("--single", action="store_true", help="Run only --policy/--field-weight instead of the default comparison.")
    p.add_argument("--validate-every", type=int, default=25)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not HAVE_CUPY:
        raise RuntimeError("CuPy is required. Install cupy-cuda12x or matching wheel.")

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / f"sm_tsp_projector_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 118}")
    print("  S_M PROJECTOR TSP EXAMPLE — CUDA FIELD-GUIDED 2-OPT")
    print(f"{'=' * 118}")
    print(f"  Out dir      : {out_dir}")
    print(f"  scipy KDTree : {'yes' if HAVE_SCIPY else 'no'}")
    print(f"  cupy CUDA    : {'yes' if HAVE_CUPY else 'no'}")
    print(f"  candidate-k  : {args.candidate_k}")
    print(f"  passes       : {args.passes}")
    print(f"  max batch    : {args.max_batch}")
    print(f"  compare      : {not args.single}")

    if args.validate_small:
        if args.N > 11:
            raise ValueError("Small validation uses Held-Karp; keep N <= 11.")
        validate_small(args, out_dir)
        print(f"\n[SAVED] {out_dir}")
        print(f"{'=' * 118}\n")
        return

    if not args.tsp_file:
        raise ValueError("Provide --tsp-file or use --validate-small.")
    tsp_path = Path(args.tsp_file)
    if not tsp_path.exists():
        raise FileNotFoundError(f"Missing TSP file: {tsp_path}\nPlace pla85900.tsp in {DATA_DIR}, or pass --tsp-file <path>.")

    t_all = time.time()
    points = parse_tsplib(tsp_path)
    n = len(points)
    print(f"  TSP file     : {tsp_path}")
    print(f"  Loaded       : {n} cities")
    if args.known_opt and args.known_opt > 0:
        print(f"  Known opt    : {args.known_opt:.3f}")

    print("\n[CONSTRUCT]")
    init, start_rows = multi_start_construct(points, args.starts, args.candidate_k, args.seed, progress=not args.no_progress)
    init_len = tour_length(points, init)
    print(f"  initial length : {init_len:.6f}")

    print("\n[NEIGHBORS]")
    t0 = time.time()
    neighbors = precompute_neighbors(points, args.candidate_k)
    print(f"  built in       : {time.time() - t0:.3f} s")
    print(f"  shape          : {neighbors.shape}")

    if args.single:
        policies = [args.policy]
        weights = [args.field_weight]
    else:
        policies = ["delta_batch", "sm_improve_batch", "sm_field_batch"]
        weights = args.field_weights

    rows, tours, stats = run_policy_set(points, init, neighbors, policies, weights, args)
    if args.known_opt and args.known_opt > 0:
        for row in rows:
            row["known_opt"] = float(args.known_opt)
            row["gap_pct"] = (float(row["final_length"]) - float(args.known_opt)) / float(args.known_opt) * 100.0
    summary = summarize(rows)
    print_summary(summary)

    for key, tour in tours.items():
        write_tour(out_dir / f"tour_{key}.txt", closed_tour(tour))

    write_csv(out_dir / "summary.csv", summary, SUMMARY_FIELDS)
    write_csv(out_dir / "routes.csv", rows, ROUTE_FIELDS)

    result = {
        "mini_paper_claim": {
            "delta_batch": "raw 2-opt improvement baseline",
            "sm_improve_batch": "bounded monotonic projector spine",
            "sm_field_batch": "tunable S_M field deformation channel"
        },
        "args": vars(args),
        "n": n,
        "initial_length": init_len,
        "starts": start_rows,
        "rows": rows,
        "summary": summary,
        "total_seconds": time.time() - t_all,
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2)

    print(f"\n[SAVED] {out_dir}")
    print(f"{'=' * 118}\n")


if __name__ == "__main__":
    main()
