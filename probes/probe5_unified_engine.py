#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — PROBE 5 — UNIFIED ENGINE (T3 RERUN)
==============================================================================
Consolidates the diagnostics of Probes 1, 2, and 3 into a single engine and
reruns them against the corrected T3 target discovered in Probe 4. Designed
to be pointed at either a physical QPU base (from dump.py) or a noiseless
GPU base (from gpu.py or Probe 4), and report:

    [PROBE 1] IDENTITY BRIDGE  — manifold MAE/RMSE against T3 ideal_p0
    [PROBE 2] GEOMETRY SCRAMBLE — intended vs permuted-geometry separation,
                                  Benford and recursive z-scores
    [PROBE 3] CHANNEL ANALYSIS  — uniform vs anchor-conditioned WLS channel
                                  fits, anchor-conditioned manifold metrics

The T3 target (Probe 4 derivation):
    P(ctrl=0) = 0.5 * (1 + cos^2(a/2) cos^2(b/2) + sin^2(a/2) sin^2(b/2))

This replaces the textbook product-state formula |cos((a-b)/2)|^2 that
Probes 1-3 originally tested against. The ghost CNOTs entangle v1 with
(a1, a2) and v2 with (b1, b2) before the Hadamard test, breaking the
product-state assumption.

HISTORICAL CONTEXT:
    Probe 5 was the validation that the Probe 4 pivot held up. After
    Probe 4 derived T3 from the statevector and showed analytical-vs-
    empirical agreement to shot noise, this probe reran the entire
    Probes 1-3 battery against the corrected target.

    The original validation result on the noiseless GPU base:
    Identity Bridge MAE dropped from 0.19 (Probe 1 against T2 =
    |cos((a-b)/2)|) to 9.8e-3 against T3 — i.e. shot noise. The
    Benford z-scores stayed low, confirming the "geometry-coupled
    holographic structure" Probe 2 chased was a sampling artifact.
    The anchor-conditioned channel fit coefficients collapsed near
    zero because there's no longer a structured residual to fit on
    the noiseless base.

    When run against a physical QPU base, the picture is different:
    Identity Bridge MAE sits in the 1e-1 range (hardware error
    against T3, not against a wrong target), and the channel fit
    coefficients pick up real structure — which is the residual
    that Probes 8.0 through 8.4 spent their span characterizing.

    Running this script against the sample bases shipped in data/
    will produce different specific numbers (different jobs and
    seeds), but the qualitative findings hold: noiseless GPU base
    matches T3 at shot noise, QPU base matches T3 within hardware
    error, and the Benford structure is null on both. This is the
    probe that closed the validation loop on the Probe 4 pivot and
    set up Probe 6's explicit three-way convergence framing.

    Probe 9 later simplified T3 to the G_M operator that drives the
    rest of the suite. See PROCESS_RECORD.md for the full arc.

USAGE:
    python probe5_unified_engine.py
    python probe5_unified_engine.py --mode stress
    python probe5_unified_engine.py --qpu data/job_xyz.npz --gpu data/ghost_oracle_gpu_xyz.npz
==============================================================================
"""

import argparse
import math
import secrets
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
# Probe 5 originally ran against 12-tile bases. The rest of the suite uses
# 16 tiles (4x4); pass --num-tiles 16 to compare against a current-generation
# QPU base. The findings reported in the docstring and PROCESS_RECORD are
# from the historical 12-tile run.
NUM_TILES        = 12
PAIRS            = [(r, c) for r in range(4) for c in range(4)][:NUM_TILES]
ANGLE_SCALE      = 1.05
ALPHA_NORM       = 0.9127
RECURSIVE_MEMORY = 0.35     # probe-local hyperparameter; not suite-wide
BENFORD_LOG      = np.array([math.log10(1 + 1 / d) for d in range(1, 10)])

MATRIX_A = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B = np.array([1.00, 0.80, 0.40, 0.10])

# Repo-root data/ directory: this file lives at <repo>/probes/probe5_*.py.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def data_to_angles(data, scale=ANGLE_SCALE):
    """Scale a real-valued vector into rotation angles in [0, pi/2 * scale]."""
    max_val = np.max(np.abs(data))
    return (data / max_val) * (np.pi / 2) * scale


ORIG_A = data_to_angles(MATRIX_A)
ORIG_B = data_to_angles(MATRIX_B)


def auto_find_base(kind):
    """Find the first base .npz of the given kind in <repo>/data/, or None.
        kind == "qpu" -> files starting with 'job_'            (from dump.py)
        kind == "gpu" -> files starting with 'noiseless_base_' (from Probe 4)
                         or 'ghost_oracle_gpu_'                (from gpu.py)
    """
    if kind == "qpu":
        patterns = ["job_*.npz"]
    elif kind == "gpu":
        patterns = ["noiseless_base_*.npz", "ghost_oracle_gpu_*.npz"]
    else:
        return None
    candidates = []
    for pat in patterns:
        candidates.extend(sorted(DATA_DIR.glob(pat)))
    return str(candidates[0]) if candidates else None


# =============================================================================
# BENFORD / RECURSIVE TELEMETRY
# =============================================================================
def leading_digit(x):
    s = f"{abs(x):e}"
    return int(s[0])


def benford_distribution_score(stream, depth=3):
    stream = np.asarray(stream, dtype=np.float64)
    deltas = np.diff(stream)
    digits = [leading_digit(d) for d in deltas if d != 0 and not np.isnan(d)]
    valid = [d for d in digits if 1 <= d <= 9]
    if len(valid) < 5:
        return 0.0
    obs = np.array([valid.count(d) / len(valid) for d in range(1, 10)])
    score = math.exp(-np.sum(((obs - BENFORD_LOG) ** 2) / BENFORD_LOG) * 5.0)
    if depth <= 1:
        return score
    return (score + benford_distribution_score(deltas, depth - 1)) / 2.0


def recursive_manifold_analysis(stream, depth=4):
    stream = np.asarray(stream, dtype=np.float64)
    if len(stream) < 8:
        return 0.0, 0.0
    scores = []
    memory = np.zeros_like(stream)
    current = stream.copy()
    for _ in range(depth):
        current = (1.0 - RECURSIVE_MEMORY) * current + RECURSIVE_MEMORY * memory[:len(current)]
        mu, sigma = np.mean(current), np.std(current)
        if sigma > 1e-12:
            current = (current - mu) / sigma
        scores.append(benford_distribution_score(current, depth=2))
        memory = 0.7 * memory[:len(current)] + 0.3 * current
        current = np.diff(current)
        if len(current) < 8:
            break
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(1.0 - np.std(scores))


def mixed_state_ideal_p0(a, b):
    """T3 target: GHZ-entangled mixed-state expectation derived in Probe 4."""
    return 0.5 * (1.0 + (math.cos(a / 2) ** 2) * (math.cos(b / 2) ** 2) +
                        (math.sin(a / 2) ** 2) * (math.sin(b / 2) ** 2))


# =============================================================================
# DATA LOAD
# =============================================================================
def load_base(path, num_tiles):
    """Load a base .npz (dump.py or Probe 4 / gpu.py schema)."""
    d = np.load(path)
    ctrl  = {t: d[f"ctrl_tile{t}"]  for t in range(num_tiles)}
    ghost = {t: d[f"ghost_tile{t}"] for t in range(num_tiles)}
    job_id = str(d.get("job_id", "unknown_job"))
    return ctrl, ghost, job_id


def extract_observed_p0(ctrl_dict, num_tiles):
    p0 = np.zeros(num_tiles)
    for t in range(num_tiles):
        p0[t] = float((ctrl_dict[t] == 0).mean())
    return p0


def p0_to_normalized_matrix(p0_vector, num_tiles):
    M = np.full((4, 4), np.nan)
    for t in range(num_tiles):
        r, c = PAIRS[t]
        raw_val = math.sqrt(max(0.0, 2 * p0_vector[t] - 1.0))
        M[r, c] = min(1.0, raw_val / ALPHA_NORM)
    return M


# =============================================================================
# PROJECTION
# =============================================================================
def project_shots_importance(ctrl_t, ghost_t, orig_a, orig_b,
                             new_a, new_b, channel_params=None):
    """Importance-reweighted projection, optionally followed by depolarization
    channel inversion (Probe 3 anchor-conditioned mode if channel_params given)."""
    a_fire = ghost_t[:, :2].mean(axis=1)
    b_fire = ghost_t[:, 2:].mean(axis=1)

    exp_a_orig = math.sin(orig_a / 2) ** 2
    exp_b_orig = math.sin(orig_b / 2) ** 2
    exp_a_new  = math.sin(new_a / 2) ** 2
    exp_b_new  = math.sin(new_b / 2) ** 2

    eps = 0.05
    def log_likelihood(obs, p):
        p = np.clip(p, eps, 1 - eps)
        return obs * np.log(p) + (1 - obs) * np.log(1 - p)

    log_w = np.clip(
        (log_likelihood(a_fire, exp_a_new) - log_likelihood(a_fire, exp_a_orig)) +
        (log_likelihood(b_fire, exp_b_new) - log_likelihood(b_fire, exp_b_orig)),
        -3, 3,
    )
    w = np.exp(log_w)
    w_sum = w.sum()
    p0_blind = float(np.sum(w * (ctrl_t == 0))) / w_sum if w_sum > 1e-12 else 0.5

    if channel_params is None:
        return float(np.clip(p0_blind, 0.0, 1.0))

    ideal_new = mixed_state_ideal_p0(new_a, new_b)
    ab_diff_new = abs(new_a - new_b)
    p_est = channel_params[0] + channel_params[1] * ab_diff_new + channel_params[2] * ideal_new
    p_est = np.clip(p_est, 0.0, 0.95)

    if (1.0 - p_est) > 1e-3:
        p0_corrected = (p0_blind - 0.5 * p_est) / (1.0 - p_est)
    else:
        p0_corrected = p0_blind
    return float(np.clip(p0_corrected, 0.0, 1.0))


def generate_manifold_stream(ctrl, ghost, test_matrices, num_tiles,
                             channel_params=None):
    stream = []
    for a_vec, b_vec in test_matrices:
        new_a = data_to_angles(a_vec)
        new_b = data_to_angles(b_vec)
        for t in range(num_tiles):
            r, c = PAIRS[t]
            p0 = project_shots_importance(
                ctrl[t], ghost[t], ORIG_A[r], ORIG_B[c],
                new_a[r], new_b[c], channel_params,
            )
            raw = math.sqrt(max(0.0, 2 * p0 - 1.0))
            stream.append(min(1.0, raw / ALPHA_NORM))
    return np.array(stream)


# =============================================================================
# WLS CHANNEL FIT (Probe 3 core, T3-aware)
# =============================================================================
def fit_channel_wls(p0_obs, weights, num_tiles):
    """Weighted least squares fit: p = alpha + beta*|a-b| + gamma*ideal_p0,
    where ideal_p0 is the T3 target (not the textbook product-state formula)."""
    X, y = [], []
    for t in range(num_tiles):
        r, c = PAIRS[t]
        ideal = mixed_state_ideal_p0(ORIG_A[r], ORIG_B[c])
        p_dep = (ideal - p0_obs[t]) / (ideal - 0.5) if abs(ideal - 0.5) > 1e-6 else 0.0
        X.append([1.0, abs(ORIG_A[r] - ORIG_B[c]), ideal])
        y.append(p_dep)
    X, y, W = np.array(X), np.array(y), np.diag(weights)
    try:
        return np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]


# =============================================================================
# UNIFIED PROBE BATTERY
# =============================================================================
def run_unified_battery(label, path, num_tiles, n_traj, n_null):
    print("\n" + "=" * 110)
    print(f"  DATASET: {label}")
    print(f"  PATH:    {path}")
    print("=" * 110)

    ctrl, ghost, job_id = load_base(path, num_tiles)
    print(f"  Job ID: {job_id}")

    # ---- 1. IDENTITY BRIDGE (Probe 1 against T3) ----
    p0_obs = extract_observed_p0(ctrl, num_tiles)
    M_obs = p0_to_normalized_matrix(p0_obs, num_tiles)

    M_ideal = np.full((4, 4), np.nan)
    for t in range(num_tiles):
        r, c = PAIRS[t]
        p0_id = mixed_state_ideal_p0(ORIG_A[r], ORIG_B[c])
        M_ideal[r, c] = min(1.0, math.sqrt(max(0.0, 2 * p0_id - 1.0)) / ALPHA_NORM)

    mask = ~np.isnan(M_obs)
    diff = M_obs[mask] - M_ideal[mask]
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    print(f"\n  [PROBE 1] IDENTITY BRIDGE (against T3 target):")
    print(f"    Manifold MAE  : {mae:.6e}")
    print(f"    Manifold RMSE : {rmse:.6e}")
    print(f"    Max |diff|    : {np.max(np.abs(diff)):.6e}")

    # ---- 2. GEOMETRY SCRAMBLE (Probe 2 rerun) ----
    sysrand = secrets.SystemRandom()

    def gen_trajectories(n, permute=False):
        out = []
        for _ in range(n):
            a = np.array([0.1 + 0.9 * sysrand.random() for _ in range(4)])
            b = np.array([0.1 + 0.9 * sysrand.random() for _ in range(4)])
            if permute:
                pa, pb = list(range(4)), list(range(4))
                sysrand.shuffle(pa); sysrand.shuffle(pb)
                a, b = a[pa], b[pb]
            out.append((a, b))
        return out

    traj_intended = gen_trajectories(n_traj, permute=False)
    traj_permuted = gen_trajectories(n_traj, permute=True)

    stream_int  = generate_manifold_stream(ctrl, ghost, traj_intended, num_tiles)
    stream_perm = generate_manifold_stream(ctrl, ghost, traj_permuted, num_tiles)

    b_int  = benford_distribution_score(stream_int)
    r_int, _ = recursive_manifold_analysis(stream_int)
    b_perm = benford_distribution_score(stream_perm)
    r_perm, _ = recursive_manifold_analysis(stream_perm)

    # Null ensemble
    null_b, null_r = [], []
    for _ in range(n_null):
        null_traj = gen_trajectories(max(50, n_traj // 4), permute=True)
        ns = generate_manifold_stream(ctrl, ghost, null_traj, num_tiles)
        null_b.append(benford_distribution_score(ns))
        null_r.append(recursive_manifold_analysis(ns)[0])
    null_b, null_r = np.array(null_b), np.array(null_r)

    z_b = (b_int - null_b.mean()) / max(null_b.std(), 1e-9)
    z_r = (r_int - null_r.mean()) / max(null_r.std(), 1e-9)

    print(f"\n  [PROBE 2] GEOMETRY SCRAMBLE:")
    print(f"    Intended Geometry -> Benford: {b_int:.6f} | Recursive: {r_int:.6f}")
    print(f"    Permuted Geometry -> Benford: {b_perm:.6f} | Recursive: {r_perm:.6f}")
    print(f"    Null Ensemble Mu  -> Benford: {null_b.mean():.4f} | Recursive: {null_r.mean():.4f}")
    print(f"    Z-Scores          -> Z_Benford: {z_b:+.2f} | Z_Recursive: {z_r:+.2f}")

    # ---- 3. CHANNEL GEOMETRY (Probe 3 rerun) ----
    w_uniform = np.full(num_tiles, 1.0 / num_tiles)
    w_anchor = np.zeros(num_tiles)
    # Anchors at tiles 3 and 11 (historical 12-tile choice; see Probe 3).
    if num_tiles > 11:
        w_anchor[3] = 0.5
        w_anchor[11] = 0.5
    else:
        # Fall back to uniform if num_tiles too small for the historical anchor pair.
        w_anchor = w_uniform.copy()

    p_uniform = fit_channel_wls(p0_obs, w_uniform, num_tiles)
    p_anchor  = fit_channel_wls(p0_obs, w_anchor,  num_tiles)

    stream_ac_a = generate_manifold_stream(ctrl, ghost, traj_intended, num_tiles,
                                            channel_params=p_anchor)
    b_ac_a = benford_distribution_score(stream_ac_a)
    r_ac_a, _ = recursive_manifold_analysis(stream_ac_a)

    print(f"\n  [PROBE 3] ANCHOR-CONDITIONED CHANNEL FITS (T3-aware):")
    print(f"    Uniform Fit Coeffs : p = {p_uniform[0]:+.4f} + "
          f"({p_uniform[1]:+.4f})*|a-b| + ({p_uniform[2]:+.4f})*ideal_p0")
    print(f"    Anchor Fit Coeffs  : p = {p_anchor[0]:+.4f} + "
          f"({p_anchor[1]:+.4f})*|a-b| + ({p_anchor[2]:+.4f})*ideal_p0")
    print(f"    Anchor-Conditioned Intended -> Benford: {b_ac_a:.6f} | Recursive: {r_ac_a:.6f}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Ghost Oracle Suite — Probe 5: Unified Engine. Reruns the "
                    "Probes 1-3 diagnostic battery against the T3 target "
                    "derived in Probe 4. Compares physical QPU and noiseless "
                    "GPU bases side by side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--qpu", default=None,
                        help="Path to QPU base .npz (auto-finds job_*.npz in data/ if omitted).")
    parser.add_argument("--gpu", default=None,
                        help="Path to noiseless GPU base .npz "
                             "(auto-finds ghost_oracle_gpu_*.npz or ghost_oracle_gpu_*.npz in data/ if omitted).")
    parser.add_argument("--num-tiles", type=int, default=NUM_TILES,
                        help="Number of tiles (must match both bases).")
    parser.add_argument("--mode", choices=["smoke", "stress"], default="smoke",
                        help="smoke = small N for sanity; stress = full N for real result.")
    args = parser.parse_args()

    qpu_path = args.qpu or auto_find_base("qpu")
    gpu_path = args.gpu or auto_find_base("gpu")

    if not qpu_path and not gpu_path:
        sys.exit(f"[FATAL] no bases found. Pass --qpu and/or --gpu, or put "
                 f"job_*.npz / ghost_oracle_gpu_*.npz in {DATA_DIR}/")

    n_traj, n_null = (100, 30) if args.mode == "smoke" else (1000, 500)

    print("\n" + "=" * 110)
    print(f"  GHOST ORACLE SUITE — PROBE 5 — UNIFIED ENGINE (T3 RERUN)  (mode={args.mode})")
    print("=" * 110)
    print(f"  Intended trajectories : {n_traj}")
    print(f"  Null ensemble draws   : {n_null}")
    print(f"  QPU base              : {qpu_path or '(none)'}")
    print(f"  GPU base              : {gpu_path or '(none)'}")

    if qpu_path:
        run_unified_battery("PHYSICAL QPU BASE", qpu_path,
                            args.num_tiles, n_traj, n_null)
    if gpu_path:
        run_unified_battery("NOISELESS GPU BASE", gpu_path,
                            args.num_tiles, n_traj, n_null)

    print()


if __name__ == "__main__":
    main()