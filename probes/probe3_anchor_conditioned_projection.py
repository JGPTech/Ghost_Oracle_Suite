#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — PROBE 3 — ANCHOR-CONDITIONED PROJECTION
==============================================================================
Builds an anchor-conditioned projector: estimates a per-tile depolarization
channel from the data, inverts it to recover ideal_p0, then applies the
half-angle Hadamard normalization.

Tests four channel-fit configurations side by side:

  CFG-A  PRIMARY-INTERIOR
    Tile 3 and Tile 11 at weight 0.5 each (primary budget = 1.0).
    Other 10 tiles weighted by |ideal_p0 - mean(ideal_p0)|, normalized to
    sum to 1.0 (secondary budget = 1.0).
    Reading: "extremes anchor, interior interpolates."

  CFG-B  PRIMARY-DECOHERENCE
    Tile 3 and Tile 11 at weight 0.5 each.
    Other 10 tiles weighted by their p_dep value, normalized to sum to 1.0.
    Reading: "noisier tiles carry more channel information."

  CFG-C  HELD-OUT-ANCHORS
    Tile 3 and Tile 11 NOT in the fit. Channel parameters fit on the
    interior 10 tiles with uniform weights. Anchors used as held-out
    validation: does interior-fit predict the extremes?

  CFG-D  UNIFORM
    All 12 tiles at weight 1/12. The "null" channel fit — what you get
    with no anchor logic at all.

For each configuration we report:
  - Channel fit parameters (alpha, beta, gamma)
  - Leave-one-out R^2 (how smooth is the channel)
  - Anchor-conditioned manifold Benford / Recursive
  - Z-score against the same permuted-geometry null from Probe 2

The wrong-channel control:
  CFG-W  WRONG-CHANNEL
    Anchor-conditioned projection but with the channel model's beta and
    gamma signs flipped. Same complexity, structurally invalid. Tells us
    whether ANY channel correction helps, or only the right one.
HISTORICAL CONTEXT:
    Probe 3 was the second control experiment in the trajectory. After
    Probe 2 showed that intended-geometry projections didn't separate
    from scrambled-geometry projections, this probe tried to rescue the
    holographic-structure claim by inverting a per-tile depolarization
    channel before projecting — the idea being that maybe the structure
    was there but obscured by noise that varied tile-by-tile.

    The original 12-tile run reported no scheme beating blind baseline:
    none of the four anchor-conditioned variants reached statistical
    separation against null (all |z| < 2), and LOO R^2 was negative
    across the board — a smooth 3-parameter depolarization model
    doesn't fit the residual. The channel has per-tile structure that
    a linear |a-b| + ideal_p0 model can't capture. Running this script
    against the sample QPU base shipped in data/ will produce
    different specific z-scores (depending on which job dump is
    auto-found), but the qualitative finding is the same: no scheme
    reaches significance, LOO R^2 stays negative, channel correction
    does not rescue geometry-coupled structure.

    That finding closed the door on the original framing and led
    directly to Probe 4's pivot, where the team discovered the ghost
    CNOTs entangle the swap-test qubits with their ancillas BEFORE the
    Hadamard test, breaking the product-state assumption the channel
    model was built on. The actual target turned out to be T3 (the
    ghost-CNOT mixed-state formula), which Probe 9 simplified to the
    G_M operator that drives the rest of the suite. See
    PROCESS_RECORD.md for the full arc.

USAGE:
    python probe3_anchor_conditioned_projection.py --mode smoke
    python probe3_anchor_conditioned_projection.py --mode stress
    python probe3_anchor_conditioned_projection.py --mode smoke --base data/job_xyz.npz
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

try:
    import cupy as cp
    _HAVE_CUPY = True
except Exception:
    cp = None
    _HAVE_CUPY = False


# =============================================================================
# CONFIG
# =============================================================================
# Probe 3 originally ran against a 12-tile job. The rest of the suite uses
# 16 tiles (4x4); pass --num-tiles 16 to compare against a current-generation
# QPU base. The findings reported in the docstring and PROCESS_RECORD are
# from the historical 12-tile run.
NUM_TILES        = 12
PAIRS            = [(r, c) for r in range(4) for c in range(4)][:NUM_TILES]
ANGLE_SCALE      = 1.05
ALPHA_NORM       = 0.9127
RECURSIVE_MEMORY = 0.35     # probe-local hyperparameter; not suite-wide

MATRIX_A_ORIG = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B_ORIG = np.array([1.00, 0.80, 0.40, 0.10])

# Primary anchor tiles (from Probe 1 extremes).
PRIMARY_ANCHORS = [3, 11]  # tile 3: p_dep=0.80, tile 11: p_dep=0.09

# Repo-root data/ directory: this file lives at <repo>/probes/probe3_*.py.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def data_to_angles(data, scale=ANGLE_SCALE):
    """Scale a real-valued vector into rotation angles in [0, pi/2 * scale]."""
    max_val = np.max(np.abs(data))
    return (data / max_val) * (np.pi / 2) * scale


ORIG_A = data_to_angles(MATRIX_A_ORIG)
ORIG_B = data_to_angles(MATRIX_B_ORIG)


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
BENFORD_LOG = np.array([math.log10(1 + 1 / d) for d in range(1, 10)])


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


# =============================================================================
# CHANNEL MODEL
#   observed_p0 = (1 - p) * ideal_p0 + p * 0.5
#   p(a, b) = alpha + beta * |a - b| + gamma * ideal_p0
# =============================================================================
def compute_per_tile_p_dep(ctrl_data, num_tiles):
    """Returns dict: tile_idx -> (a, b, ideal_p0, observed_p0, p_dep)"""
    info = {}
    for t in range(num_tiles):
        r, c = PAIRS[t]
        a = ORIG_A[r]
        b = ORIG_B[c]
        obs = float((ctrl_data[t] == 0).mean())
        ideal = (1 + math.cos((a - b) / 2.0) ** 2) / 2.0
        if abs(ideal - 0.5) > 1e-6:
            p = (ideal - obs) / (ideal - 0.5)
        else:
            p = 0.0
        info[t] = {"a": a, "b": b, "ideal_p0": ideal, "obs_p0": obs,
                   "p_dep": p, "ab_diff": abs(a - b)}
    return info


def fit_channel_wls(info, weights):
    """Weighted least squares fit: p = alpha + beta*|a-b| + gamma*ideal_p0"""
    tiles = sorted(info.keys())
    X = np.array([[1.0, info[t]["ab_diff"], info[t]["ideal_p0"]] for t in tiles])
    y = np.array([info[t]["p_dep"] for t in tiles])
    w = np.array([weights[t] for t in tiles])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    try:
        params = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
    return params  # alpha, beta, gamma


def predict_p(params, ab_diff, ideal_p0):
    return params[0] + params[1] * ab_diff + params[2] * ideal_p0


def leave_one_out_r2(info, weight_fn):
    """For each tile, fit on the others (using weight_fn) and predict held-out."""
    tiles = sorted(info.keys())
    preds = {}
    for held in tiles:
        sub_info = {t: info[t] for t in tiles if t != held}
        sub_weights = weight_fn(sub_info)
        sub_params = fit_channel_wls(sub_info, sub_weights)
        preds[held] = predict_p(sub_params, info[held]["ab_diff"],
                                info[held]["ideal_p0"])
    actuals = np.array([info[t]["p_dep"] for t in tiles])
    predicts = np.array([preds[t] for t in tiles])
    ss_res = np.sum((actuals - predicts) ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return r2, preds


# =============================================================================
# WEIGHT SCHEMES
# =============================================================================
def weights_primary_interior(info):
    """CFG-A: anchors 0.5 each (primary budget=1.0), interior tiles weighted
    by |ideal_p0 - mean(ideal_p0)| normalized to sum=1.0 (secondary budget=1.0)."""
    tiles = sorted(info.keys())
    interior = [t for t in tiles if t not in PRIMARY_ANCHORS]
    weights = {}
    for t in PRIMARY_ANCHORS:
        if t in info:
            weights[t] = 0.5
    if interior:
        mean_ideal = np.mean([info[t]["ideal_p0"] for t in tiles])
        raw = {t: abs(info[t]["ideal_p0"] - mean_ideal) for t in interior}
        total = sum(raw.values())
        if total > 1e-12:
            for t in interior:
                weights[t] = raw[t] / total
        else:
            for t in interior:
                weights[t] = 1.0 / len(interior)
    return weights


def weights_primary_decoherence(info):
    """CFG-B: anchors 0.5 each, interior tiles weighted by their p_dep."""
    tiles = sorted(info.keys())
    interior = [t for t in tiles if t not in PRIMARY_ANCHORS]
    weights = {}
    for t in PRIMARY_ANCHORS:
        if t in info:
            weights[t] = 0.5
    if interior:
        raw = {t: max(info[t]["p_dep"], 1e-6) for t in interior}
        total = sum(raw.values())
        for t in interior:
            weights[t] = raw[t] / total
    return weights


def weights_held_out(info):
    """CFG-C: anchors get weight 0 (held out), interior gets uniform."""
    tiles = sorted(info.keys())
    interior = [t for t in tiles if t not in PRIMARY_ANCHORS]
    weights = {t: 0.0 for t in tiles}
    if interior:
        for t in interior:
            weights[t] = 1.0 / len(interior)
    return weights


def weights_uniform(info):
    """CFG-D: all tiles equal."""
    tiles = sorted(info.keys())
    return {t: 1.0 / len(tiles) for t in tiles}


SCHEMES = {
    "A_PRIMARY_INTERIOR":    weights_primary_interior,
    "B_PRIMARY_DECOHERENCE": weights_primary_decoherence,
    "C_HELD_OUT_ANCHORS":    weights_held_out,
    "D_UNIFORM":             weights_uniform,
}


# =============================================================================
# PROJECTION
# =============================================================================
def project_shots_blind(ctrl_t, ghost_t, orig_a, orig_b, new_a, new_b):
    """Original blind projection (same as Probe 2)."""
    a_fire = ghost_t[:, :2].mean(axis=1)
    b_fire = ghost_t[:, 2:].mean(axis=1)
    exp_a_orig = math.sin(orig_a / 2) ** 2
    exp_b_orig = math.sin(orig_b / 2) ** 2
    exp_a_new  = math.sin(new_a / 2) ** 2
    exp_b_new  = math.sin(new_b / 2) ** 2
    eps = 0.05
    def ll(obs, p):
        p = np.clip(p, eps, 1 - eps)
        return obs * np.log(p) + (1 - obs) * np.log(1 - p)
    log_w = np.clip(
        (ll(a_fire, exp_a_new) - ll(a_fire, exp_a_orig)) +
        (ll(b_fire, exp_b_new) - ll(b_fire, exp_b_orig)),
        -3, 3,
    )
    w = np.exp(log_w); w /= w.sum()
    return float(np.sum(w * (ctrl_t == 0)))


def anchor_conditioned_projection(ctrl_t, ghost_t, orig_a, orig_b,
                                  new_a, new_b, channel_params):
    """
    Project to new geometry, then invert the depolarization channel using
    the fitted channel model evaluated at (new_a, new_b).

    p0_corrected = (p0_observed - 0.5 * p_estimated) / (1 - p_estimated)
    """
    p0_blind = project_shots_blind(ctrl_t, ghost_t, orig_a, orig_b, new_a, new_b)
    # Ideal_p0 for the NEW geometry
    ideal_new = (1 + math.cos((new_a - new_b) / 2.0) ** 2) / 2.0
    ab_diff_new = abs(new_a - new_b)
    p_est = predict_p(channel_params, ab_diff_new, ideal_new)
    p_est = np.clip(p_est, 0.0, 0.95)  # avoid divide-by-zero
    if (1.0 - p_est) > 1e-3:
        p0_corrected = (p0_blind - 0.5 * p_est) / (1.0 - p_est)
    else:
        p0_corrected = p0_blind
    return float(np.clip(p0_corrected, 0.0, 1.0))


def manifold_stream(ctrl_data, ghost_data, test_matrices, orig_a, orig_b,
                    project_fn, **proj_kwargs):
    stream = []
    for a_vec, b_vec in test_matrices:
        new_a = data_to_angles(a_vec)
        new_b = data_to_angles(b_vec)
        for t_idx, (r, c) in enumerate(PAIRS):
            p0 = project_fn(ctrl_data[t_idx], ghost_data[t_idx],
                            orig_a[r], orig_b[c], new_a[r], new_b[c],
                            **proj_kwargs)
            if 0.45 <= p0 < 0.5:
                p0 = 0.5
            raw = math.sqrt(max(0.0, 2 * p0 - 1.0))
            norm = min(1.0, raw / ALPHA_NORM)
            stream.append(norm)
    return np.array(stream)


# =============================================================================
# GEOMETRY GENERATORS
# =============================================================================
sysrand = secrets.SystemRandom()


def gen_test_matrices_intended(n):
    out = []
    for _ in range(n):
        a = np.array([0.1 + 0.9 * sysrand.random() for _ in range(4)])
        b = np.array([0.1 + 0.9 * sysrand.random() for _ in range(4)])
        out.append((a, b))
    return out


def gen_test_matrices_permuted(n):
    out = []
    for _ in range(n):
        a = np.array([0.1 + 0.9 * sysrand.random() for _ in range(4)])
        b = np.array([0.1 + 0.9 * sysrand.random() for _ in range(4)])
        pa = list(range(4)); sysrand.shuffle(pa)
        pb = list(range(4)); sysrand.shuffle(pb)
        out.append((a[pa], b[pb]))
    return out


# =============================================================================
# DATA LOAD
# =============================================================================
def load_qpu_base(path, num_tiles):
    """Load a QPU base .npz (dump.py schema)."""
    print(f"[LOAD] {path}")
    d = np.load(path)
    ctrl  = {t: d[f"ctrl_tile{t}"]  for t in range(num_tiles)}
    ghost = {t: d[f"ghost_tile{t}"] for t in range(num_tiles)}
    return ctrl, ghost


# =============================================================================
# REPORT HELPERS
# =============================================================================
def print_channel_diagnostics(name, params, weights, info, loo_r2, loo_preds):
    print(f"\n  [{name}]")
    print(f"    Fit:   p = {params[0]:+.4f} + ({params[1]:+.4f})*|a-b| "
          f"+ ({params[2]:+.4f})*ideal_p0")
    print(f"    LOO R^2: {loo_r2:+.4f}")
    print(f"    {'tile':>5} {'(r,c)':>8} {'|a-b|':>8} {'ideal_p0':>10} "
          f"{'p_actual':>10} {'p_pred_LOO':>12} {'resid':>10} {'weight':>10}")
    for t in sorted(info.keys()):
        r, c = PAIRS[t]
        marker = " *" if t in PRIMARY_ANCHORS else "  "
        resid = info[t]["p_dep"] - loo_preds[t]
        print(f"    {t:>5}{marker}({r},{c})  {info[t]['ab_diff']:>7.4f}  "
              f"{info[t]['ideal_p0']:>9.4f}  {info[t]['p_dep']:>9.4f}  "
              f"{loo_preds[t]:>11.4f}  {resid:>+9.4f}  {weights[t]:>9.4f}")


def print_metric_row(label, m):
    print(f"  {label:<36} | len={m['stream_len']:>6} | "
          f"mean={m['mean']:>6.4f} | std={m['std']:>6.4f} | "
          f"benford={m['benford']:>8.6f} | recursive={m['recursive']:>8.6f}")


def evaluate(ctrl, ghost, test_mats, orig_a, orig_b, project_fn, **kw):
    stream = manifold_stream(ctrl, ghost, test_mats, orig_a, orig_b, project_fn, **kw)
    bscore = benford_distribution_score(stream)
    rscore, rstab = recursive_manifold_analysis(stream)
    return {
        "stream_len": len(stream),
        "mean": float(np.mean(stream)), "std": float(np.std(stream)),
        "benford": bscore, "recursive": rscore, "stability": rstab,
        "stream": stream,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Ghost Oracle Suite — Probe 3: Anchor-Conditioned Projection. "
                    "Tests whether inverting a per-tile depolarization channel "
                    "rescues geometry-coupled structure in the QPU manifold stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mode", choices=["smoke", "stress"], default="smoke",
                    help="smoke = small N for sanity; stress = full N for real result.")
    ap.add_argument("--base", default=None,
                    help="Path to QPU base .npz from dump.py. "
                         "Auto-finds in data/ if omitted.")
    ap.add_argument("--num-tiles", type=int, default=NUM_TILES,
                    help="Number of tiles in the base (must match the .npz).")
    args = ap.parse_args()

    base_path = args.base or auto_find_base("qpu")
    if not base_path:
        sys.exit(f"[FATAL] no QPU base found. Pass --base or put a "
                 f"job_*.npz in {DATA_DIR}/")

    if args.mode == "smoke":
        N_INT, N_NULL = 100, 30
    else:
        N_INT, N_NULL = 1000, 500

    print("\n" + "=" * 110)
    print(f"  GHOST ORACLE SUITE — PROBE 3 — ANCHOR-CONDITIONED PROJECTION  (mode={args.mode})")
    print("=" * 110)
    print(f"  Intended trajectories : {N_INT}")
    print(f"  Null ensemble draws   : {N_NULL}")
    print(f"  CuPy available        : {_HAVE_CUPY}")

    ctrl, ghost = load_qpu_base(base_path, args.num_tiles)

    # ---- Build per-tile channel info
    info = compute_per_tile_p_dep(ctrl, args.num_tiles)

    print("\n" + "-" * 110)
    print("  CHANNEL FITS (per scheme)")
    print("-" * 110)

    scheme_params = {}
    scheme_loo = {}
    for name, wfn in SCHEMES.items():
        weights = wfn(info)
        params = fit_channel_wls(info, weights)
        loo_r2, loo_preds = leave_one_out_r2(info, wfn)
        scheme_params[name] = params
        scheme_loo[name] = loo_r2
        print_channel_diagnostics(name, params, weights, info, loo_r2, loo_preds)

    # ---- Generate trajectories for projection sweep
    print("\n" + "-" * 110)
    print("  GENERATING TRAJECTORIES")
    print("-" * 110)
    test_mats_intended = gen_test_matrices_intended(N_INT)

    # ---- Baselines
    print("\n" + "-" * 110)
    print("  BASELINE (blind half-angle Hadamard, no channel correction)")
    print("-" * 110)
    blind = evaluate(ctrl, ghost, test_mats_intended, ORIG_A, ORIG_B,
                     project_shots_blind)
    print_metric_row("BLIND  REAL + INTENDED geom", blind)

    # ---- Anchor-conditioned, all schemes
    print("\n" + "-" * 110)
    print("  ANCHOR-CONDITIONED PROJECTIONS")
    print("-" * 110)
    scheme_results = {}
    for name in SCHEMES:
        m = evaluate(ctrl, ghost, test_mats_intended, ORIG_A, ORIG_B,
                     anchor_conditioned_projection,
                     channel_params=scheme_params[name])
        scheme_results[name] = m
        print_metric_row(f"AC-{name}  REAL + INTENDED geom", m)

    # ---- Wrong-channel controls
    #   W1 CONSTANT  : flat channel at mean(p_dep). Right magnitude, no position info.
    #   W2 SHUFFLED  : channel coefficients shuffled so beta and gamma swap roles.
    #                  Structurally valid (same magnitude scale), geometry scrambled.
    print("\n" + "-" * 110)
    print("  WRONG-CHANNEL CONTROLS")
    print("-" * 110)
    mean_p = float(np.mean([info[t]["p_dep"] for t in info]))
    constant_params = np.array([mean_p, 0.0, 0.0])
    print(f"  W1 CONSTANT: p = {constant_params[0]:+.4f}  (flat, no geometry)")
    w1 = evaluate(ctrl, ghost, test_mats_intended, ORIG_A, ORIG_B,
                  anchor_conditioned_projection,
                  channel_params=constant_params)
    print_metric_row("AC-W1_CONSTANT", w1)

    # Swap the geometry coefficients: |a-b| coeff and ideal_p0 coeff trade places.
    # Magnitude preserved, geometric meaning scrambled.
    src = scheme_params["A_PRIMARY_INTERIOR"]
    shuffled_params = np.array([src[0], src[2], src[1]])
    print(f"  W2 SHUFFLED: p = {shuffled_params[0]:+.4f} + ({shuffled_params[1]:+.4f})*|a-b| "
          f"+ ({shuffled_params[2]:+.4f})*ideal_p0")
    w2 = evaluate(ctrl, ghost, test_mats_intended, ORIG_A, ORIG_B,
                  anchor_conditioned_projection,
                  channel_params=shuffled_params)
    print_metric_row("AC-W2_SHUFFLED", w2)

    # ---- Null distribution: permuted-geometry projections for each scheme
    print("\n" + "-" * 110)
    print("  NULL ENSEMBLE (permuted-geometry, per scheme)")
    print("-" * 110)
    print(f"  Running {N_NULL} permuted-geometry draws for each of "
          f"{len(SCHEMES) + 2} projection variants...")

    nulls = {name: {"benford": [], "recursive": []} for name in
             list(SCHEMES.keys()) + ["BLIND", "W1_CONSTANT", "W2_SHUFFLED"]}

    for i in range(N_NULL):
        test_mats_null = gen_test_matrices_permuted(max(50, N_INT // 4))
        # Blind
        m = evaluate(ctrl, ghost, test_mats_null, ORIG_A, ORIG_B,
                     project_shots_blind)
        nulls["BLIND"]["benford"].append(m["benford"])
        nulls["BLIND"]["recursive"].append(m["recursive"])
        # Each scheme
        for name in SCHEMES:
            m = evaluate(ctrl, ghost, test_mats_null, ORIG_A, ORIG_B,
                         anchor_conditioned_projection,
                         channel_params=scheme_params[name])
            nulls[name]["benford"].append(m["benford"])
            nulls[name]["recursive"].append(m["recursive"])
        # W1 constant
        m = evaluate(ctrl, ghost, test_mats_null, ORIG_A, ORIG_B,
                     anchor_conditioned_projection,
                     channel_params=constant_params)
        nulls["W1_CONSTANT"]["benford"].append(m["benford"])
        nulls["W1_CONSTANT"]["recursive"].append(m["recursive"])
        # W2 shuffled
        m = evaluate(ctrl, ghost, test_mats_null, ORIG_A, ORIG_B,
                     anchor_conditioned_projection,
                     channel_params=shuffled_params)
        nulls["W2_SHUFFLED"]["benford"].append(m["benford"])
        nulls["W2_SHUFFLED"]["recursive"].append(m["recursive"])

        if (i + 1) % max(1, N_NULL // 10) == 0:
            print(f"    null draw {i+1}/{N_NULL}")

    # ---- Final separation table
    print("\n" + "=" * 110)
    print("  SEPARATION TABLE  (intended vs permuted-geometry null)")
    print("=" * 110)
    print(f"  {'variant':<24} | {'B_intended':>10} | {'B_null_mu':>10} | "
          f"{'B_null_sd':>10} | {'Z_B':>6} | "
          f"{'R_intended':>10} | {'R_null_mu':>10} | {'R_null_sd':>10} | {'Z_R':>6} | "
          f"{'LOO_R2':>7}")
    print("  " + "-" * 130)

    def row(label, intended_b, intended_r, null_b, null_r, loo=None):
        nb = np.array(null_b); nr = np.array(null_r)
        zb = (intended_b - nb.mean()) / max(nb.std(), 1e-9)
        zr = (intended_r - nr.mean()) / max(nr.std(), 1e-9)
        loo_str = f"{loo:+.4f}" if loo is not None else "  -- "
        print(f"  {label:<24} | {intended_b:>10.4f} | {nb.mean():>10.4f} | "
              f"{nb.std():>10.4f} | {zb:>+6.2f} | "
              f"{intended_r:>10.4f} | {nr.mean():>10.4f} | {nr.std():>10.4f} | "
              f"{zr:>+6.2f} | {loo_str:>7}")

    row("BLIND (baseline)", blind["benford"], blind["recursive"],
        nulls["BLIND"]["benford"], nulls["BLIND"]["recursive"])
    for name in SCHEMES:
        row(f"AC-{name}", scheme_results[name]["benford"],
            scheme_results[name]["recursive"],
            nulls[name]["benford"], nulls[name]["recursive"],
            loo=scheme_loo[name])
    row("AC-W1_CONSTANT", w1["benford"], w1["recursive"],
        nulls["W1_CONSTANT"]["benford"], nulls["W1_CONSTANT"]["recursive"])
    row("AC-W2_SHUFFLED", w2["benford"], w2["recursive"],
        nulls["W2_SHUFFLED"]["benford"], nulls["W2_SHUFFLED"]["recursive"])

    print("\n" + "=" * 110)
    print("  INTERPRETATION KEY")
    print("=" * 110)
    print("  Z_B/Z_R > BLIND  -> channel correction adds discriminating power.")
    print("                       Favors Interpretation A (geometry-coupled structure).")
    print("  Z_B/Z_R ~ BLIND  -> channel correction is a no-op on structure.")
    print("                       Favors Interpretation B (structure passes through).")
    print("  Z_B/Z_R < BLIND  -> channel correction destroys structure.")
    print("                       Structure was an artifact of the blind projection.")
    print()
    print("  LOO_R^2 high     -> channel is smooth across the manifold.")
    print("  LOO_R^2 low/neg  -> channel has per-tile structure; smooth model insufficient.")
    print("  AC-WRONG separation tells us whether ANY structured projection helps")
    print("  or only the geometrically motivated one.")
    print()


if __name__ == "__main__":
    main()