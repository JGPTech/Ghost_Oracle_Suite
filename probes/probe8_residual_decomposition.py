#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — PROBE 8 — RESIDUAL DECOMPOSITION (FOUR PHASES)
==============================================================================
This file unifies four exploratory scripts from the original investigation
(probe8_residual_decomposition, probe8_1_split_readout, probe8_2_drift_first,
probe8_4) into a single phased probe. The four shared so much code that
keeping them as separate files in the public repo was paying overhead for
no clarity; they are presented here in the order the original investigation
ran them, which is also the order that makes them readable as a single
arc.

GPU base is ground truth (samples T3 to shot noise); QPU base is the
device under test. The 32-bin joint distribution per tile is the full
resolution at which both bases agree, and is where the QPU's structural
deviation actually lives. Probes 1-7 only ever looked at the scalar
P(ctrl=0) per tile; this probe looks at the entire joint.

PHASES:

  PHASE A — INITIAL CHANNEL FIT (originally probe8.0)
      Five stages: distribution metrics with shot-noise floor, residual
      SVD, three-channel mixture fit (depol + ghost-decoherence +
      symmetric 5-qubit readout), likelihood ratio, channel-inverted
      reduction. Finds that the symmetric-readout model is *degenerate*
      with the ghost-decoherence channel — the fitter slides along a
      (lam_g, lam_r) ridge — and that channel inversion makes the MAE
      *worse*. Sets up the diagnosis that drives Phase B.

  PHASE B — SPLIT READOUT (originally probe8.1)
      Replaces the symmetric 5-qubit readout with two independent rates:
      eps_ctrl on the control qubit, eps_ghost on the four ghost qubits.
      Each channel now has a distinct fingerprint on which marginals
      shift, so the fitter degeneracy is broken. Reports cross-tile
      channel correlations and a held-out-channel diagnostic (refit with
      one channel forced to zero, observe how much TVD costs). Finds
      ghost decoherence as the only essential channel; achieves a
      ~+20% MAE reduction. Also discovers a large coherent angle drift
      (~-0.2 rad on a, ~-0.1 rad on b) in the post-fit residual, which
      motivates Phase C.

  PHASE C — DRIFT-FIRST ALTERNATING (originally probe8.2)
      Fits a shared coherent drift (d_a, d_b) across all tiles FIRST,
      then per-tile residual drift with OUT regularization (penalizes
      per-tile deviation from cohort mean), then channels on the
      drift-corrected reference. Alternates until converged. Tests
      the hypothesis that the drift was a single global calibration
      error being scattered into 12 per-tile fits by Phase B. Finds
      the *opposite*: the alternating optimizer settles on a large
      shared drift AND large per-tile residual drift, and the
      MAE reduction is WORSE than Phase B's. The drift is per-tile /
      layout-dependent, not a clean global error. With many
      alternations the optimizer can wind the angles past 2pi (the
      OUT regularizer can't pull per-tile drifts back to consensus
      once the shared drift starts chasing channel residuals); the
      default of one alternation matches the original Probe 8.2 run.
      This negative result is what motivated the Probe 9 G_M
      derivation: the residual structure is not in any of the
      canonical channel or coherent-error families.

  PHASE D — BENFORD / p-ADIC NULL SWEEP (originally probe8.4)
      Forensic test of whether anything detectable as numerical
      structure survives either of the corrections from Phases B and
      C. Four residual matrices (B-raw, B-post, C-raw, C-post) tested
      across five Benford bases ({2,3,5,7,10}), three orderings
      (row-major, column-major, top-SVD-mode), with 200 within-tile
      shuffled-null draws each, plus p-adic valuation chi^2 against
      the analytic geometric reference for p in {2, 3, 5}. The
      finding: a handful of scattered |z|>2 hits, none consistent
      across bases, no v_p chi^2/dof above 3.5. Nothing meaningful
      survives. The QPU residual after channel + drift correction is
      decoherence noise, not hidden structure.

HISTORICAL CONTEXT:
    Probe 8's role in the trajectory was to characterize what's
    LEFT after Probes 1-5 confirmed that T3 is the right target. The
    QPU still has ~10% MAE against T3, and Probe 8 was the systematic
    attempt to attribute that residual to canonical error channels
    (depolarization, decoherence, readout, coherent drift) one at a
    time.

    The four phases trace the investigation faithfully: A finds a
    degeneracy, B fixes it and discovers a drift, C tries to fit the
    drift cleanly and fails, D confirms nothing exotic remains. The
    cumulative finding is that the residual is dominated by ghost
    decoherence (lam_g ~ 0.3 across tiles) and per-tile structure
    that no smooth global model captures. That negative result
    motivated Probe 9's pivot: instead of trying to model the
    residual structure, derive a simpler target operator (G_M) that
    absorbs the bulk of it into a tied-channel form. The G_M
    operator drives the rest of the suite and the headline
    projection_benchmark.py.

    Running this script against the sample bases shipped in data/
    will produce different specific numbers (different jobs, seeds),
    but the qualitative findings hold across phases: A's degeneracy,
    B's split-readout reduction with ghost-decoherence dominance, C's
    failure-to-improve from drift-first, D's null sweep. See
    PROCESS_RECORD.md for the full arc.

USAGE:
    python probe8_residual_decomposition.py
    python probe8_residual_decomposition.py --phase A,B
    python probe8_residual_decomposition.py --phase D --base data/job_xyz.npz
                                                       --gpu  data/noiseless_xyz.npz
==============================================================================
"""

import argparse
import math
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
# Probe 8 originally ran against 12-tile bases. The rest of the suite uses
# 16 tiles (4x4); pass --num-tiles 16 to compare against a current-generation
# QPU base. All four phases share this config.
NUM_TILES        = 12
PAIRS            = [(r, c) for r in range(4) for c in range(4)][:NUM_TILES]
ANGLE_SCALE      = 1.05
ALPHA_NORM       = 0.9127
RECURSIVE_MEMORY = 0.35    # probe-local hyperparameter (used in Phase D)

MATRIX_A = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B = np.array([1.00, 0.80, 0.40, 0.10])

UNIFORM_32 = np.ones(32) / 32.0

# Repo-root data/ directory: this file lives at <repo>/probes/probe8_*.py.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def data_to_angles(data, scale=ANGLE_SCALE):
    """Scale a real-valued vector into rotation angles in [0, pi/2 * scale]."""
    max_val = np.max(np.abs(data))
    return (data / max_val) * (np.pi / 2) * scale


ANGLES_A = data_to_angles(MATRIX_A)
ANGLES_B = data_to_angles(MATRIX_B)


def auto_find_base(kind):
    """Find the first base .npz of the given kind in <repo>/data/, or None.
        kind == "qpu" -> files starting with 'job_'            (from dump.py)
        kind == "gpu" -> files starting with 'ghost_oracle_gpu_' (from Probe 4)
                         or 'ghost_oracle_gpu_'                (from gpu.py)
    """
    if kind == "qpu":
        patterns = ["job_*.npz"]
    elif kind == "gpu":
        patterns = ["ghost_oracle_gpu_*.npz", "ghost_oracle_gpu_*.npz"]
    else:
        return None
    candidates = []
    for pat in patterns:
        candidates.extend(sorted(DATA_DIR.glob(pat)))
    return str(candidates[0]) if candidates else None


# =============================================================================
# 32-BIN JOINT INDEX ENCODING
# Bins are indexed: idx = ctrl*16 + a1*8 + a2*4 + b1*2 + b2
# =============================================================================
def bits_to_index(ctrl, a1, a2, b1, b2):
    return (ctrl << 4) | (a1 << 3) | (a2 << 2) | (b1 << 1) | b2


def shots_to_distribution(ctrl, ghost):
    """(ctrl: (N,), ghost: (N, 4)) -> (p: (32,), counts: (32,))"""
    a1, a2, b1, b2 = ghost[:, 0], ghost[:, 1], ghost[:, 2], ghost[:, 3]
    idx = bits_to_index(ctrl, a1, a2, b1, b2).astype(np.int64)
    counts = np.bincount(idx, minlength=32)
    return counts / counts.sum(), counts


# =============================================================================
# EXACT T3 JOINT DISTRIBUTION (closed form, no simulator dependency)
#
# After the ghost CNOTs the 6-qubit pre-Hadamard-test state is
#   |Psi> = (c_a|000> + s_a|111>)_{v1,a1,a2} (x) (c_b|000> + s_b|111>)_{v2,b1,b2}
# Hadamard test on ctrl, CSWAP, second Hadamard gives, for each (v1,v2):
#   P(ctrl=0|v1,v2) = (1 + |<v1|v2>|^2) / 2 = 1 if v1==v2 else 1/2
# Joint amplitudes for the four GHZ configurations are c_a c_b, c_a s_b,
# s_a c_b, s_a s_b, all in distinct basis states, so probabilities are
# squared amplitudes.
# =============================================================================
def t3_joint(theta_a, theta_b):
    """Exact 32-bin joint P(ctrl, a1, a2, b1, b2) for one tile."""
    c_a = math.cos(theta_a / 2); s_a = math.sin(theta_a / 2)
    c_b = math.cos(theta_b / 2); s_b = math.sin(theta_b / 2)
    p_v = {
        (0, 0): (c_a * c_b) ** 2,
        (0, 1): (c_a * s_b) ** 2,
        (1, 0): (s_a * c_b) ** 2,
        (1, 1): (s_a * s_b) ** 2,
    }

    def p_ctrl(v1, v2, ctrl):
        if v1 == v2:
            return 1.0 if ctrl == 0 else 0.0
        return 0.5

    p = np.zeros(32)
    for (v1, v2), pv in p_v.items():
        a1 = a2 = v1; b1 = b2 = v2
        for ctrl in (0, 1):
            p[bits_to_index(ctrl, a1, a2, b1, b2)] += pv * p_ctrl(v1, v2, ctrl)
    return p


def all_t3_joints(num_tiles):
    return [t3_joint(ANGLES_A[r], ANGLES_B[c]) for (r, c) in PAIRS[:num_tiles]]


# =============================================================================
# CHANNEL BUILDING BLOCKS
# =============================================================================
def p_ghost_decoherence(theta_a, theta_b):
    """Ghost-decoherence: ctrl marginal stays at T3 P0; ghost bits become
    independent of ctrl and of each other, with their T3 marginals."""
    c_a = math.cos(theta_a / 2); s_a = math.sin(theta_a / 2)
    c_b = math.cos(theta_b / 2); s_b = math.sin(theta_b / 2)
    p_a = s_a ** 2
    p_b = s_b ** 2
    p_ctrl0 = 0.5 * (1.0 + c_a**2 * c_b**2 + s_a**2 * s_b**2)
    p_ctrl1 = 1.0 - p_ctrl0
    p = np.zeros(32)
    for ctrl, a1, a2, b1, b2 in product([0, 1], repeat=5):
        pc = p_ctrl0 if ctrl == 0 else p_ctrl1
        pa = (p_a if a1 == 1 else (1 - p_a)) * (p_a if a2 == 1 else (1 - p_a))
        pb = (p_b if b1 == 1 else (1 - p_b)) * (p_b if b2 == 1 else (1 - p_b))
        p[bits_to_index(ctrl, a1, a2, b1, b2)] = pc * pa * pb
    return p


def p_readout_symmetric(p_ref, eps_per_qubit):
    """Symmetric bit-flip readout on all 5 measured qubits (Phase A model)."""
    e = eps_per_qubit
    out = np.zeros(32)
    for new in range(32):
        for old in range(32):
            d = bin(new ^ old).count("1")
            out[new] += p_ref[old] * (e ** d) * ((1 - e) ** (5 - d))
    return out


def p_readout_split(p_ref, eps_ctrl, eps_ghost):
    """Split bit-flip readout: separate rates for ctrl bit vs the four ghost
    bits (Phase B model). The 32x32 transition matrix is built explicitly."""
    out = np.zeros(32)
    for new in range(32):
        for old in range(32):
            diff = new ^ old
            ctrl_flip = (diff >> 4) & 1
            g_flips = bin(diff & 0b01111).count("1")
            g_matches = 4 - g_flips
            ctrl_factor = (eps_ctrl if ctrl_flip else (1 - eps_ctrl))
            g_factor = (eps_ghost ** g_flips) * ((1 - eps_ghost) ** g_matches)
            out[new] += ctrl_factor * g_factor * p_ref[old]
    return out


# =============================================================================
# METRICS
# =============================================================================
def tvd(p, q):
    return 0.5 * np.abs(p - q).sum()


def kl_div(p, q, eps=1e-12):
    """KL(p || q)."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    mask = p > 0
    return float(np.sum(p_safe[mask] * np.log(p_safe[mask] / q_safe[mask])))


def chi2_stat(counts, p_ref, n_shots):
    """Standard chi^2 against expected counts."""
    expected = p_ref * n_shots
    mask = expected >= 5
    if not mask.any():
        return 0.0, 0
    diff = counts[mask] - expected[mask]
    return float(np.sum(diff * diff / expected[mask])), int(mask.sum())


def log_likelihood(counts, p):
    """Multinomial log-likelihood up to the multinomial coefficient."""
    p_safe = np.clip(p, 1e-12, 1.0)
    return float(np.sum(counts * np.log(p_safe)))


def shot_noise_floor(p_ref, n_shots, n_trials=400, seed=0):
    """Monte-Carlo TVD/KL floor between a sample from p_ref and p_ref itself."""
    rng = np.random.default_rng(seed)
    tvds, kls = [], []
    for _ in range(n_trials):
        sample = rng.choice(32, size=n_shots, p=p_ref)
        q = np.bincount(sample, minlength=32) / n_shots
        tvds.append(tvd(p_ref, q))
        kls.append(kl_div(q, p_ref))
    return float(np.mean(tvds)), float(np.std(tvds)), \
           float(np.mean(kls)), float(np.std(kls))


def residual_svd(q_list, p_list):
    """Stack (q-p) for tiles into an (n_tiles, 32) matrix and SVD it."""
    R = np.stack([q - p for q, p in zip(q_list, p_list)], axis=0)
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    return s, U, Vt, R


def matrix_entry_from_p0(p0):
    raw = math.sqrt(max(0.0, 2.0 * p0 - 1.0))
    return min(1.0, raw / ALPHA_NORM)


# =============================================================================
# I/O
# =============================================================================
def load_base(path, num_tiles):
    print(f"[LOAD] {path}")
    d = np.load(path, allow_pickle=True)
    ctrl  = {t: d[f"ctrl_tile{t}"]  for t in range(num_tiles)}
    ghost = {t: d[f"ghost_tile{t}"] for t in range(num_tiles)}
    label = str(d["job_id"]) if "job_id" in d.files else "?"
    return ctrl, ghost, label


def hline(c="-", w=92):
    print(c * w)


def section(title, w=92):
    print()
    hline("=", w)
    print(f"  {title}")
    hline("=", w)


# =============================================================================
# CHANNEL FITS
# =============================================================================
def fit_three_channel(q_obs, theta_a, theta_b, max_iter=200, tol=1e-9):
    """Phase A fit: q ~ (1-s) T3 + lam_d uniform + lam_g ghost_decoh
                       + lam_r p_readout_symmetric(eps).
    Grid search + local random refinement."""
    p_t3 = t3_joint(theta_a, theta_b)
    p_gd = p_ghost_decoherence(theta_a, theta_b)

    def model(lam_d, lam_g, lam_r, eps):
        lam_d = max(0.0, lam_d); lam_g = max(0.0, lam_g); lam_r = max(0.0, lam_r)
        eps = float(np.clip(eps, 0.0, 0.4))
        s = lam_d + lam_g + lam_r
        if s > 1.0:
            lam_d /= s; lam_g /= s; lam_r /= s; s = 1.0
        p_ro = p_readout_symmetric(p_t3, eps)
        return ((1.0 - s) * p_t3 + lam_d * UNIFORM_32 + lam_g * p_gd
                + lam_r * p_ro)

    best = (np.inf, None, None)
    grid = np.linspace(0.0, 1.0, 11)
    for lam_d in grid:
        for lam_g in grid:
            if lam_d + lam_g > 1.0: continue
            for lam_r in grid:
                if lam_d + lam_g + lam_r > 1.0: continue
                for eps in [0.0, 0.02, 0.05, 0.10, 0.20]:
                    q_m = model(lam_d, lam_g, lam_r, eps)
                    L = tvd(q_obs, q_m)
                    if L < best[0]:
                        best = (L, (lam_d, lam_g, lam_r, eps), q_m)

    if best[1] is not None:
        lam_d, lam_g, lam_r, eps = best[1]
        rng = np.random.default_rng(0)
        for _ in range(800):
            cand = (
                max(0.0, lam_d + rng.normal(0, 0.03)),
                max(0.0, lam_g + rng.normal(0, 0.03)),
                max(0.0, lam_r + rng.normal(0, 0.03)),
                float(np.clip(eps + rng.normal(0, 0.01), 0.0, 0.4)),
            )
            if cand[0] + cand[1] + cand[2] > 1.0: continue
            q_m = model(*cand)
            L = tvd(q_obs, q_m)
            if L < best[0]:
                best = (L, cand, q_m)
                lam_d, lam_g, lam_r, eps = cand

    lam_d, lam_g, lam_r, eps = best[1]
    q_model = best[2]
    return {
        "lam_depol": lam_d, "lam_ghost": lam_g, "lam_readout": lam_r,
        "readout_eps": eps, "lam_ideal": 1.0 - lam_d - lam_g - lam_r,
        "tvd_fit": best[0], "tvd_unfit": tvd(q_obs, p_t3),
        "kl_fit": kl_div(q_obs, q_model), "kl_unfit": kl_div(q_obs, p_t3),
        "q_model": q_model,
    }


def _project_simplex(lams):
    """Clip nonneg and ensure sum <= 1."""
    lams = np.maximum(lams, 0.0)
    s = lams.sum()
    if s > 1.0:
        lams = lams / s
    return lams


def fit_split_readout(q_obs, theta_a, theta_b, n_grid=6, n_local=1500, seed=0):
    """Phase B fit (and used by C, D): split-readout 5-channel mixture.
        q ~ (1-s) T3 + lam_d uniform + lam_g ghost_decoh
                    + lam_rc readout(eps_c on ctrl only)
                    + lam_rg readout(0, eps_g on ghosts only)
    Grid + local refinement."""
    p_t3 = t3_joint(theta_a, theta_b)
    p_gd = p_ghost_decoherence(theta_a, theta_b)

    def model(lam_d, lam_g, lam_rc, lam_rg, eps_c, eps_g):
        lams = _project_simplex(np.array([lam_d, lam_g, lam_rc, lam_rg]))
        lam_d, lam_g, lam_rc, lam_rg = lams
        s = lams.sum()
        eps_c = float(np.clip(eps_c, 0.0, 0.4))
        eps_g = float(np.clip(eps_g, 0.0, 0.4))
        p_rc = p_readout_split(p_t3, eps_c, 0.0)
        p_rg = p_readout_split(p_t3, 0.0, eps_g)
        return ((1 - s) * p_t3 + lam_d * UNIFORM_32 + lam_g * p_gd
                + lam_rc * p_rc + lam_rg * p_rg), (lam_d, lam_g, lam_rc, lam_rg, eps_c, eps_g)

    best = (np.inf, None, None)
    lam_grid = np.linspace(0, 0.6, n_grid)
    eps_grid = np.array([0.0, 0.02, 0.05, 0.10, 0.20])
    for lam_d in [0.0, 0.1]:
        for lam_g in lam_grid:
            for lam_rc in lam_grid:
                if lam_d + lam_g + lam_rc > 1.0: continue
                for lam_rg in lam_grid:
                    if lam_d + lam_g + lam_rc + lam_rg > 1.0: continue
                    for eps_c in eps_grid:
                        for eps_g in eps_grid:
                            q_m, params = model(lam_d, lam_g, lam_rc, lam_rg, eps_c, eps_g)
                            L = tvd(q_obs, q_m)
                            if L < best[0]:
                                best = (L, params, q_m)

    rng = np.random.default_rng(seed)
    lam_d, lam_g, lam_rc, lam_rg, eps_c, eps_g = best[1]
    for it in range(n_local):
        scale = 0.05 * (0.97 ** (it / 50))
        cand = (
            max(0.0, lam_d  + rng.normal(0, scale)),
            max(0.0, lam_g  + rng.normal(0, scale)),
            max(0.0, lam_rc + rng.normal(0, scale)),
            max(0.0, lam_rg + rng.normal(0, scale)),
            float(np.clip(eps_c + rng.normal(0, scale * 0.5), 0.0, 0.4)),
            float(np.clip(eps_g + rng.normal(0, scale * 0.5), 0.0, 0.4)),
        )
        if cand[0] + cand[1] + cand[2] + cand[3] > 1.0: continue
        q_m, params = model(*cand)
        L = tvd(q_obs, q_m)
        if L < best[0]:
            best = (L, params, q_m)
            lam_d, lam_g, lam_rc, lam_rg, eps_c, eps_g = params

    lam_d, lam_g, lam_rc, lam_rg, eps_c, eps_g = best[1]
    q_model = best[2]
    return {
        "lam_d": lam_d, "lam_g": lam_g, "lam_rc": lam_rc, "lam_rg": lam_rg,
        "eps_ctrl": eps_c, "eps_ghost": eps_g,
        "lam_ideal": 1.0 - lam_d - lam_g - lam_rc - lam_rg,
        "tvd_fit": best[0], "tvd_unfit": tvd(q_obs, p_t3),
        "kl_fit": kl_div(q_obs, q_model), "kl_unfit": kl_div(q_obs, p_t3),
        "q_model": q_model,
    }


def fit_split_readout_restricted(q_obs, theta_a, theta_b, force_zero, seed=0):
    """Like fit_split_readout but with one or more lam_* forced to zero.
    Used by Phase B's held-out-channel degeneracy diagnostic."""
    p_t3 = t3_joint(theta_a, theta_b)
    p_gd = p_ghost_decoherence(theta_a, theta_b)
    fz = set(force_zero)

    def model(lams_eps):
        lam_d, lam_g, lam_rc, lam_rg, eps_c, eps_g = lams_eps
        if 'lam_d'  in fz: lam_d  = 0
        if 'lam_g'  in fz: lam_g  = 0
        if 'lam_rc' in fz: lam_rc = 0
        if 'lam_rg' in fz: lam_rg = 0
        lams = _project_simplex(np.array([lam_d, lam_g, lam_rc, lam_rg]))
        lam_d, lam_g, lam_rc, lam_rg = lams
        s = lams.sum()
        eps_c = float(np.clip(eps_c, 0.0, 0.4))
        eps_g = float(np.clip(eps_g, 0.0, 0.4))
        p_rc = p_readout_split(p_t3, eps_c, 0.0)
        p_rg = p_readout_split(p_t3, 0.0, eps_g)
        return ((1 - s) * p_t3 + lam_d * UNIFORM_32 + lam_g * p_gd
                + lam_rc * p_rc + lam_rg * p_rg), (lam_d, lam_g, lam_rc, lam_rg, eps_c, eps_g)

    best = (np.inf, None, None)
    lam_grid = np.linspace(0, 0.6, 7)
    eps_grid = np.array([0.0, 0.02, 0.05, 0.10, 0.20])
    for lam_d in [0.0, 0.1]:
        if 'lam_d' in fz: lam_d = 0
        for lam_g in lam_grid:
            if 'lam_g' in fz: lam_g = 0
            for lam_rc in lam_grid:
                if 'lam_rc' in fz: lam_rc = 0
                for lam_rg in lam_grid:
                    if 'lam_rg' in fz: lam_rg = 0
                    if lam_d + lam_g + lam_rc + lam_rg > 1.0: continue
                    for eps_c in eps_grid:
                        for eps_g in eps_grid:
                            q_m, params = model((lam_d, lam_g, lam_rc, lam_rg, eps_c, eps_g))
                            L = tvd(q_obs, q_m)
                            if L < best[0]:
                                best = (L, params, q_m)
                    if 'lam_rg' in fz: break
                if 'lam_rc' in fz: break
            if 'lam_g' in fz: break

    rng = np.random.default_rng(seed)
    params = best[1]
    for it in range(1500):
        scale = 0.04 * (0.97 ** (it / 50))
        cand = list(params)
        for i in range(6):
            cand[i] = cand[i] + rng.normal(0, scale)
        if 'lam_d'  in fz: cand[0] = 0
        if 'lam_g'  in fz: cand[1] = 0
        if 'lam_rc' in fz: cand[2] = 0
        if 'lam_rg' in fz: cand[3] = 0
        cand[:4] = [max(0.0, c) for c in cand[:4]]
        cand[4] = float(np.clip(cand[4], 0, 0.4))
        cand[5] = float(np.clip(cand[5], 0, 0.4))
        if sum(cand[:4]) > 1.0: continue
        q_m, params_new = model(cand)
        L = tvd(q_obs, q_m)
        if L < best[0]:
            best = (L, params_new, q_m)
            params = params_new
    return best[0], best[1]


def apply_channels_forward(p_t3, theta_a, theta_b, fit):
    """Apply Phase B/C channel mixture forward to T3 (for comparison with observed q)."""
    p_gd = p_ghost_decoherence(theta_a, theta_b)
    s = fit['lam_d'] + fit['lam_g'] + fit['lam_rc'] + fit['lam_rg']
    p_rc = p_readout_split(p_t3, fit['eps_ctrl'], 0.0)
    p_rg = p_readout_split(p_t3, 0.0, fit['eps_ghost'])
    return ((1 - s) * p_t3 + fit['lam_d'] * UNIFORM_32 + fit['lam_g'] * p_gd
            + fit['lam_rc'] * p_rc + fit['lam_rg'] * p_rg)


# =============================================================================
# COHERENT-DRIFT FITS (Phase B Stage 6 and Phase C)
# =============================================================================
def fit_coherent_drift_per_tile(q_obs, theta_a, theta_b, search_range=0.4, n=21):
    """Phase B Stage 6: best (d_a, d_b) for a single tile against bare T3."""
    best = (np.inf, 0.0, 0.0)
    deltas = np.linspace(-search_range, search_range, n)
    for d_a in deltas:
        for d_b in deltas:
            p = t3_joint(theta_a + d_a, theta_b + d_b)
            L = tvd(q_obs, p)
            if L < best[0]:
                best = (L, d_a, d_b)
    return best  # (tvd, d_a, d_b)


def fit_shared_drift(q_qpu_list, num_tiles, channel_fits=None,
                     n_grid=41, search_range=0.5, n_local=2000, seed=0):
    """Phase C Stage 2: a single shared (d_a, d_b) across all tiles, optionally
    with a forward channel applied."""
    def total_loss(d_a, d_b):
        L = 0.0
        for t, (r, c) in enumerate(PAIRS[:num_tiles]):
            p_drifted = t3_joint(ANGLES_A[r] + d_a, ANGLES_B[c] + d_b)
            if channel_fits is not None:
                p_drifted = apply_channels_forward(
                    p_drifted, ANGLES_A[r] + d_a, ANGLES_B[c] + d_b,
                    channel_fits[t])
            L += tvd(q_qpu_list[t], p_drifted)
        return L / num_tiles

    deltas = np.linspace(-search_range, search_range, n_grid)
    best = (np.inf, 0.0, 0.0)
    for d_a in deltas:
        for d_b in deltas:
            L = total_loss(d_a, d_b)
            if L < best[0]:
                best = (L, d_a, d_b)

    rng = np.random.default_rng(seed)
    d_a, d_b = best[1], best[2]
    for it in range(n_local):
        scale = 0.02 * (0.97 ** (it / 100))
        cand_a = d_a + rng.normal(0, scale)
        cand_b = d_b + rng.normal(0, scale)
        L = total_loss(cand_a, cand_b)
        if L < best[0]:
            best = (L, cand_a, cand_b)
            d_a, d_b = cand_a, cand_b
    return best[1], best[2], best[0]


def fit_residual_drift(q_qpu_list, num_tiles, d_a_shared, d_b_shared,
                       channel_fits=None, lam_out=0.5, n_iter=1000, seed=0):
    """Phase C Stage 3: per-tile residual drift (eps_a^t, eps_b^t) with
    cross-tile OUT regularization (penalize deviation from cohort mean)."""
    eps_a = np.zeros(num_tiles)
    eps_b = np.zeros(num_tiles)
    rng = np.random.default_rng(seed)

    def loss(eps_a_arr, eps_b_arr):
        L = 0.0
        for t, (r, c) in enumerate(PAIRS[:num_tiles]):
            p = t3_joint(ANGLES_A[r] + d_a_shared + eps_a_arr[t],
                         ANGLES_B[c] + d_b_shared + eps_b_arr[t])
            if channel_fits is not None:
                p = apply_channels_forward(
                    p, ANGLES_A[r] + d_a_shared + eps_a_arr[t],
                    ANGLES_B[c] + d_b_shared + eps_b_arr[t],
                    channel_fits[t])
            L += tvd(q_qpu_list[t], p)
        L /= num_tiles
        ma = eps_a_arr.mean(); mb = eps_b_arr.mean()
        out_pen = (np.mean((eps_a_arr - ma) ** 2)
                   + np.mean((eps_b_arr - mb) ** 2))
        return L + lam_out * out_pen

    current_L = loss(eps_a, eps_b)
    for it in range(n_iter):
        scale = 0.05 * (0.99 ** (it / 50))
        cand_a = eps_a + rng.normal(0, scale, size=num_tiles)
        cand_b = eps_b + rng.normal(0, scale, size=num_tiles)
        L = loss(cand_a, cand_b)
        if L < current_L:
            eps_a, eps_b = cand_a, cand_b
            current_L = L
    return eps_a, eps_b, current_L


# =============================================================================
# REDUCTION (channel inversion)
# =============================================================================
def correct_p0_three_channel(p0_obs, fit, theta_a, theta_b):
    """Invert Phase A's three-channel mixture to recover P0_T3."""
    lam_d = fit["lam_depol"]; lam_g = fit["lam_ghost"]
    lam_r = fit["lam_readout"]; eps = fit["readout_eps"]
    p_t3 = t3_joint(theta_a, theta_b)
    p_t3_ctrl0 = p_t3[:16].sum()
    if eps > 0 and lam_r > 0:
        p_ro = p_readout_symmetric(p_t3, eps)
        p_ro_ctrl0 = p_ro[:16].sum()
    else:
        p_ro_ctrl0 = p_t3_ctrl0
    denom = 1.0 - lam_d - lam_r  # lam_g doesn't shift ctrl marginal
    if denom < 1e-3:
        return p0_obs
    numer = p0_obs - lam_d * 0.5 - lam_r * p_ro_ctrl0
    return float(np.clip(numer / denom, 0.0, 1.0))


def correct_p0_split(p0_obs, fit, theta_a, theta_b):
    """Invert Phase B/C's split-readout mixture for P(ctrl=0)."""
    lam_d  = fit["lam_d"]; lam_g  = fit["lam_g"]
    lam_rc = fit["lam_rc"]; lam_rg = fit["lam_rg"]
    lam_id = fit["lam_ideal"]; eps_c = fit["eps_ctrl"]
    denom = lam_id + lam_g + lam_rg + lam_rc * (1.0 - 2.0 * eps_c)
    if denom < 1e-3:
        return p0_obs
    numer = p0_obs - 0.5 * lam_d - lam_rc * eps_c
    return float(np.clip(numer / denom, 0.0, 1.0))


# =============================================================================
# PHASE D — BENFORD / p-ADIC TELEMETRY (originally probe8.4)
# =============================================================================
BENFORD_BASES = [2, 3, 5, 7, 10]
VALUATION_BASES = [2, 3, 5]


def leading_digit_base(x, base):
    """Leading digit of |x| in the given integer base."""
    x = abs(float(x))
    if x == 0 or not math.isfinite(x):
        return None
    while x >= base:
        x /= base
    while x < 1.0:
        x *= base
    return int(x)


def benford_law_base(base):
    return np.array([math.log(1 + 1 / d, base) for d in range(1, base)])


def benford_score(stream, base, depth=3):
    """Same recursive Benford score used by Probes 2/3/5, parameterized on base."""
    stream = np.asarray(stream, dtype=np.float64)
    deltas = np.diff(stream)
    digits = [leading_digit_base(d, base) for d in deltas
              if d != 0 and not np.isnan(d)]
    valid = [d for d in digits if d is not None and 1 <= d <= base - 1]
    if len(valid) < 5:
        return 0.0
    obs = np.array([valid.count(d) / len(valid) for d in range(1, base)])
    law = benford_law_base(base)
    score = math.exp(-np.sum(((obs - law) ** 2) / np.maximum(law, 1e-9)) * 5.0)
    if depth <= 1:
        return score
    return (score + benford_score(deltas, base, depth - 1)) / 2.0


def benford_chi2(stream, base):
    deltas = np.diff(np.asarray(stream, dtype=np.float64))
    digits = [leading_digit_base(d, base) for d in deltas
              if d != 0 and not np.isnan(d)]
    valid = [d for d in digits if d is not None and 1 <= d <= base - 1]
    if len(valid) < 5:
        return 0.0, 0
    n = len(valid)
    obs = np.array([valid.count(d) for d in range(1, base)])
    expected = benford_law_base(base) * n
    mask = expected >= 5
    if not mask.any():
        return 0.0, 0
    chi2 = float(np.sum((obs[mask] - expected[mask]) ** 2 / expected[mask]))
    return chi2, int(mask.sum() - 1)


def recursive_manifold_analysis(stream, depth=4):
    """Reused verbatim from Probes 2/3/5 for direct comparison."""
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
        scores.append(benford_score(current, base=10, depth=2))
        memory = 0.7 * memory[:len(current)] + 0.3 * current
        current = np.diff(current)
        if len(current) < 8:
            break
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(1.0 - np.std(scores))


def v_p(n, p):
    """p-adic valuation of integer n (the exponent of p in n)."""
    if n == 0:
        return float("inf")
    n = abs(int(n))
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k


def valuation_distribution(R, p, n_shots, max_k=15):
    """Empirical distribution of v_p over integer-scaled residuals."""
    counts = np.zeros(max_k + 1)
    for x in R.flatten():
        n = int(round(x * n_shots))
        if n == 0:
            continue
        k = min(v_p(n, p), max_k)
        counts[k] += 1
    s = counts.sum()
    return counts / s if s > 0 else counts


def geometric_law(p, max_k=15):
    pmf = np.array([(1 - 1.0/p) * (1.0/p) ** k for k in range(max_k + 1)])
    return pmf / pmf.sum()


def valuation_chi2(R, p, n_shots, max_k=10):
    emp = valuation_distribution(R, p, n_shots, max_k=max_k)
    law = geometric_law(p, max_k=max_k)
    total = sum(1 for x in R.flatten() if int(round(x * n_shots)) != 0)
    if total == 0:
        return 0.0, 0
    obs = emp * total
    expected = law * total
    mask = expected >= 5
    if not mask.any():
        return 0.0, 0
    return float(np.sum((obs[mask] - expected[mask]) ** 2 / expected[mask])), int(mask.sum() - 1)


def order_row_major(R):
    return R.flatten()


def order_col_major(R):
    return R.T.flatten()


def order_svd(R, k=3):
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    modes = []
    for i in range(min(k, len(s))):
        modes.append(s[i] * Vt[i])
    return modes, s, Vt


def shuffle_within_tile(R, seed):
    rng = np.random.default_rng(seed)
    out = R.copy()
    for t in range(out.shape[0]):
        rng.shuffle(out[t])
    return out


def null_distribution_benford(R, ordering_fn, base, n_null=200, seed=0):
    out = []
    for i in range(n_null):
        R_sh = shuffle_within_tile(R, seed + i)
        stream = ordering_fn(R_sh)
        out.append(benford_score(stream, base))
    return np.array(out)


def null_distribution_recursive(R, ordering_fn, n_null=200, seed=0):
    out = []
    for i in range(n_null):
        R_sh = shuffle_within_tile(R, seed + i)
        stream = ordering_fn(R_sh)
        out.append(recursive_manifold_analysis(stream)[0])
    return np.array(out)


# =============================================================================
# PHASE A IMPLEMENTATION
# =============================================================================
def run_phase_A(qpu_ctrl, qpu_ghost, gpu_ctrl, gpu_ghost, num_tiles):
    section("PHASE A — INITIAL CHANNEL FIT (3-channel mixture)")
    print(f"  Tiles : {num_tiles}")

    q_qpu_list, q_gpu_list = [], []
    counts_qpu, counts_gpu = [], []
    for t in range(num_tiles):
        q_q, c_q = shots_to_distribution(qpu_ctrl[t], qpu_ghost[t])
        q_g, c_g = shots_to_distribution(gpu_ctrl[t], gpu_ghost[t])
        q_qpu_list.append(q_q); q_gpu_list.append(q_g)
        counts_qpu.append(c_q); counts_gpu.append(c_g)
    p_t3_list = all_t3_joints(num_tiles)
    n_shots = int(counts_qpu[0].sum())
    print(f"  Shots/tile QPU: {n_shots}")
    print(f"  Shots/tile GPU: {int(counts_gpu[0].sum())}")

    # ---- A.1 Per-tile distribution metrics
    section("A.1 — PER-TILE DISTRIBUTION METRICS")
    tvd_floor_mean, tvd_floor_std, kl_floor_mean, kl_floor_std = shot_noise_floor(
        p_t3_list[num_tiles // 2], n_shots, n_trials=400, seed=42)
    print(f"  Shot-noise floor (N={n_shots}): TVD {tvd_floor_mean:.4f} +/- "
          f"{tvd_floor_std:.4f}, KL {kl_floor_mean:.4f} +/- {kl_floor_std:.4f}")
    print()
    print(f"  {'tile':>4} {'(r,c)':>7} | {'TVD(GPU,T3)':>12} {'TVD(QPU,T3)':>12} "
          f"{'KL(QPU||T3)':>12} {'chi2/dof':>10} {'signal':>8}")
    hline()
    tile_metrics = []
    for t in range(num_tiles):
        r, c = PAIRS[t]
        tvd_g = tvd(q_gpu_list[t], p_t3_list[t])
        tvd_q = tvd(q_qpu_list[t], p_t3_list[t])
        kl_q  = kl_div(q_qpu_list[t], p_t3_list[t])
        chi2_q, dof_q = chi2_stat(counts_qpu[t], p_t3_list[t], n_shots)
        chi2_per_dof = chi2_q / dof_q if dof_q > 0 else float("nan")
        sigmas = (tvd_q - tvd_floor_mean) / max(tvd_floor_std, 1e-6)
        tile_metrics.append({"sigmas": sigmas, "tvd_q": tvd_q})
        print(f"  {t:>4} ({r},{c})   {tvd_g:>11.4f} {tvd_q:>11.4f} {kl_q:>11.4f} "
              f"{chi2_per_dof:>10.2f} {sigmas:>+7.1f}s")
    n_above = sum(1 for m in tile_metrics if m["sigmas"] > 3)
    print(f"\n  Tiles with QPU TVD > floor + 3 sigma: {n_above}/{num_tiles}")

    # ---- A.2 Residual SVD
    section("A.2 — RESIDUAL SVD")
    s, U, Vt, R = residual_svd(q_qpu_list, p_t3_list)
    total_energy = float(np.sum(s ** 2))
    print(f"  Frobenius energy of residual matrix : {total_energy:.4e}")
    cum = 0.0
    print(f"  {'k':>3} {'sigma_k':>12} {'cum_energy_frac':>18}")
    for i, sv in enumerate(s[:6]):
        cum += sv ** 2
        print(f"  {i+1:>3} {sv:>12.4e} {cum/total_energy:>18.4f}")
    top_mode = Vt[0]
    mode_ctrl_imbalance = top_mode[:16].sum() - top_mode[16:].sum()
    print(f"\n  Top-1 mode ctrl-imbalance (ctrl=0 sum - ctrl=1 sum): "
          f"{mode_ctrl_imbalance:+.4f}")
    if abs(mode_ctrl_imbalance) > 0.5:
        print("    -> dominant mode is largely a ctrl-bit shift (depolarization-like).")
    else:
        print("    -> dominant mode involves ghost bits, not just ctrl.")

    # ---- A.3 Channel decomposition (3-channel symmetric readout)
    section("A.3 — CHANNEL DECOMPOSITION (3-channel, symmetric readout)")
    print(f"  Fit q ~ (1-s) T3 + lam_d uniform + lam_g ghost_decoh + lam_r readout_sym(eps)")
    print()
    print(f"  {'tile':>4} {'(r,c)':>7} | {'lam_d':>8} {'lam_g':>8} {'lam_r':>8} {'eps':>8} | "
          f"{'TVD_T3':>9} {'TVD_fit':>9}")
    hline()
    fits = []
    for t in range(num_tiles):
        r, c = PAIRS[t]
        fit = fit_three_channel(q_qpu_list[t], ANGLES_A[r], ANGLES_B[c])
        fits.append(fit)
        print(f"  {t:>4} ({r},{c})   {fit['lam_depol']:>8.3f} {fit['lam_ghost']:>8.3f} "
              f"{fit['lam_readout']:>8.3f} {fit['readout_eps']:>8.3f} | "
              f"{fit['tvd_unfit']:>9.4f} {fit['tvd_fit']:>9.4f}")

    mean_lam_d = float(np.mean([f["lam_depol"] for f in fits]))
    mean_lam_g = float(np.mean([f["lam_ghost"] for f in fits]))
    mean_lam_r = float(np.mean([f["lam_readout"] for f in fits]))
    mean_eps   = float(np.mean([f["readout_eps"] for f in fits]))
    print(f"\n  Mean: lam_d={mean_lam_d:.4f}  lam_g={mean_lam_g:.4f}  "
          f"lam_r={mean_lam_r:.4f}  eps={mean_eps:.4f}")
    print(f"  Degeneracy diagnostic: lam_g and lam_r often substitute for each other")
    print(f"    -> Phase B splits the readout channel to break this degeneracy.")

    # ---- A.4 Likelihood ratio
    section("A.4 — LIKELIHOOD-RATIO SUMMARY")
    print(f"  {'tile':>4} {'(r,c)':>7} | {'logL(T3)':>14} {'logL(model)':>14} {'Delta':>14}")
    hline()
    deltas_ll = []
    for t in range(num_tiles):
        r, c = PAIRS[t]
        ll_t3 = log_likelihood(counts_qpu[t], p_t3_list[t])
        ll_m  = log_likelihood(counts_qpu[t], fits[t]["q_model"])
        delta = ll_m - ll_t3
        deltas_ll.append(delta)
        print(f"  {t:>4} ({r},{c})    {ll_t3:>14.2f} {ll_m:>14.2f} {delta:>+14.2f}")
    print(f"\n  Mean Delta per tile : {np.mean(deltas_ll):+.2f} nats")
    print(f"  Mean Delta per shot : {np.mean(deltas_ll)/n_shots:+.4f} nats")

    # ---- A.5 Reduction attempt
    section("A.5 — REDUCTION ATTEMPT (channel inversion)")
    p0_qpu = np.array([(qpu_ctrl[t] == 0).mean() for t in range(num_tiles)])
    p0_gpu = np.array([(gpu_ctrl[t] == 0).mean() for t in range(num_tiles)])
    M_qpu = np.array([matrix_entry_from_p0(p) for p in p0_qpu])
    M_gpu = np.array([matrix_entry_from_p0(p) for p in p0_gpu])
    baseline_mae = float(np.mean(np.abs(M_qpu - M_gpu)))

    p0_corr = []
    for t in range(num_tiles):
        r, c = PAIRS[t]
        p0_corr.append(correct_p0_three_channel(p0_qpu[t], fits[t], ANGLES_A[r], ANGLES_B[c]))
    p0_corr = np.array(p0_corr)
    M_corr = np.array([matrix_entry_from_p0(p) for p in p0_corr])
    corr_mae = float(np.mean(np.abs(M_corr - M_gpu)))

    print(f"  Baseline  MAE(QPU,  GPU) : {baseline_mae:.4e}")
    print(f"  Corrected MAE(QPUc, GPU) : {corr_mae:.4e}")
    rel = (baseline_mae - corr_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
    print(f"  Reduction                : {baseline_mae - corr_mae:+.4e}  ({rel:+.1f}%)")
    if rel > 5:
        print(f"\n  VERDICT: MEANINGFUL REDUCTION — channel model captures structure.")
    elif rel > 0:
        print(f"\n  VERDICT: MARGINAL REDUCTION — channels capture a little, mostly noise.")
    else:
        print(f"\n  VERDICT: NO REDUCTION — channel model does not span the residual.")
    print(f"  (Note: degeneracy between lam_g and lam_r is the suspected cause; see Phase B.)")

    return {"q_qpu_list": q_qpu_list, "q_gpu_list": q_gpu_list,
            "p_t3_list": p_t3_list, "n_shots": n_shots,
            "fits": fits, "baseline_mae": baseline_mae, "corr_mae": corr_mae,
            "mean": (mean_lam_d, mean_lam_g, mean_lam_r, mean_eps)}


# =============================================================================
# PHASE B IMPLEMENTATION
# =============================================================================
def run_phase_B(qpu_ctrl, qpu_ghost, gpu_ctrl, gpu_ghost, num_tiles, skip_drift=False):
    section("PHASE B — SPLIT-READOUT CHANNEL DECOMPOSITION")
    print(f"  Tiles : {num_tiles}")
    print(f"  Channels: depol, ghost-decoh, readout-ctrl, readout-ghost")

    q_qpu_list, q_gpu_list = [], []
    counts_qpu = []
    for t in range(num_tiles):
        q_q, c_q = shots_to_distribution(qpu_ctrl[t], qpu_ghost[t])
        q_g, _   = shots_to_distribution(gpu_ctrl[t], gpu_ghost[t])
        q_qpu_list.append(q_q); q_gpu_list.append(q_g)
        counts_qpu.append(c_q)
    p_t3_list = all_t3_joints(num_tiles)
    n_shots = int(counts_qpu[0].sum())

    # ---- B.1 Distribution metrics + B.2 SVD (compact form)
    section("B.1 — DISTRIBUTION METRICS (vs T3)")
    floor_mean, floor_std, _, _ = shot_noise_floor(p_t3_list[num_tiles // 2], n_shots, seed=42)
    print(f"  Shot-noise TVD floor (N={n_shots}): {floor_mean:.4f} +/- {floor_std:.4f}\n")
    print(f"  {'tile':>4} {'(r,c)':>7} | {'TVD(GPU,T3)':>12} {'TVD(QPU,T3)':>12} "
          f"{'KL(QPU||T3)':>12} {'signal':>8}")
    hline()
    for t in range(num_tiles):
        r, c = PAIRS[t]
        tvd_g = tvd(q_gpu_list[t], p_t3_list[t])
        tvd_q = tvd(q_qpu_list[t], p_t3_list[t])
        kl_q  = kl_div(q_qpu_list[t], p_t3_list[t])
        sig   = (tvd_q - floor_mean) / max(floor_std, 1e-6)
        print(f"  {t:>4} ({r},{c})   {tvd_g:>11.4f} {tvd_q:>11.4f} {kl_q:>11.4f} "
              f"{sig:>+7.1f}s")

    section("B.2 — RESIDUAL SVD")
    s, U, Vt, R = residual_svd(q_qpu_list, p_t3_list)
    total_energy = float(np.sum(s ** 2))
    cum = 0.0
    print(f"  {'k':>3} {'sigma_k':>12} {'cum_energy_frac':>18}")
    for k in range(min(6, len(s))):
        cum += s[k] ** 2
        print(f"  {k+1:>3} {s[k]:>12.4e} {cum/total_energy:>18.4f}")

    # ---- B.3 Split-readout 5-channel fit
    section("B.3 — SPLIT-READOUT MIXTURE FIT")
    print(f"  {'tile':>4} {'(r,c)':>7} | "
          f"{'lam_d':>7} {'lam_g':>7} {'lam_rc':>7} {'lam_rg':>7} "
          f"{'eps_c':>7} {'eps_g':>7} | {'TVD_T3':>8} {'TVD_fit':>8}")
    hline()
    fits = []
    for t in range(num_tiles):
        r, c = PAIRS[t]
        fit = fit_split_readout(q_qpu_list[t], ANGLES_A[r], ANGLES_B[c],
                                n_grid=6, n_local=1500, seed=t)
        fits.append(fit)
        print(f"  {t:>4} ({r},{c})   "
              f"{fit['lam_d']:>7.3f} {fit['lam_g']:>7.3f} "
              f"{fit['lam_rc']:>7.3f} {fit['lam_rg']:>7.3f} "
              f"{fit['eps_ctrl']:>7.3f} {fit['eps_ghost']:>7.3f} | "
              f"{fit['tvd_unfit']:>8.4f} {fit['tvd_fit']:>8.4f}")

    mean = {k: float(np.mean([f[k] for f in fits])) for k in
            ["lam_d", "lam_g", "lam_rc", "lam_rg", "eps_ctrl", "eps_ghost"]}
    print(f"\n  Mean lam_d  : {mean['lam_d']:.4f}")
    print(f"  Mean lam_g  : {mean['lam_g']:.4f}")
    print(f"  Mean lam_rc : {mean['lam_rc']:.4f}   (ctrl-only readout)")
    print(f"  Mean lam_rg : {mean['lam_rg']:.4f}   (ghost-only readout)")
    print(f"  Mean eps_c  : {mean['eps_ctrl']:.4f}")
    print(f"  Mean eps_g  : {mean['eps_ghost']:.4f}")

    lam_g_arr  = np.array([f["lam_g"]  for f in fits])
    lam_rc_arr = np.array([f["lam_rc"] for f in fits])
    lam_rg_arr = np.array([f["lam_rg"] for f in fits])
    print()
    print("  Cross-tile channel correlations (Pearson r):")
    print(f"    corr(lam_g,  lam_rc)  = {np.corrcoef(lam_g_arr,  lam_rc_arr)[0,1]:+.3f}")
    print(f"    corr(lam_g,  lam_rg)  = {np.corrcoef(lam_g_arr,  lam_rg_arr)[0,1]:+.3f}")
    print(f"    corr(lam_rc, lam_rg)  = {np.corrcoef(lam_rc_arr, lam_rg_arr)[0,1]:+.3f}")
    print("  (Strong negative correlations would indicate residual fitter degeneracy.)")

    # ---- B.4 Held-out-channel degeneracy diagnostic
    section("B.4 — DEGENERACY DIAGNOSTIC (held-out channel refits)")
    print("  Refit with one channel forced to 0; cost = TVD_restricted - TVD_full.")
    print("  Cost < 0.005 -> channel is non-essential / degenerate.")
    print("  Cost > 0.05  -> channel is genuinely identified by the data.\n")
    print(f"  {'tile':>4} {'(r,c)':>7} | {'TVD_full':>9} "
          f"{'no lam_g':>10} {'no lam_rc':>11} {'no lam_rg':>11}")
    hline()
    cost_no_g, cost_no_rc, cost_no_rg = [], [], []
    for t in range(num_tiles):
        r, c = PAIRS[t]
        tvd_full = fits[t]["tvd_fit"]
        tvd_no_g,  _ = fit_split_readout_restricted(q_qpu_list[t], ANGLES_A[r], ANGLES_B[c],
                                                    force_zero=['lam_g'], seed=t)
        tvd_no_rc, _ = fit_split_readout_restricted(q_qpu_list[t], ANGLES_A[r], ANGLES_B[c],
                                                    force_zero=['lam_rc'], seed=t)
        tvd_no_rg, _ = fit_split_readout_restricted(q_qpu_list[t], ANGLES_A[r], ANGLES_B[c],
                                                    force_zero=['lam_rg'], seed=t)
        cost_no_g.append(tvd_no_g - tvd_full)
        cost_no_rc.append(tvd_no_rc - tvd_full)
        cost_no_rg.append(tvd_no_rg - tvd_full)
        print(f"  {t:>4} ({r},{c})   {tvd_full:>9.4f}  "
              f"+{tvd_no_g - tvd_full:>8.4f}  "
              f"+{tvd_no_rc - tvd_full:>9.4f}  "
              f"+{tvd_no_rg - tvd_full:>9.4f}")
    print(f"\n  Mean cost of removing lam_g  : {np.mean(cost_no_g):.4f}  (max {np.max(cost_no_g):.4f})")
    print(f"  Mean cost of removing lam_rc : {np.mean(cost_no_rc):.4f}  (max {np.max(cost_no_rc):.4f})")
    print(f"  Mean cost of removing lam_rg : {np.mean(cost_no_rg):.4f}  (max {np.max(cost_no_rg):.4f})")

    # ---- B.5 Reduction
    section("B.5 — CORRECTED REDUCTION (split-readout inversion)")
    p0_qpu = np.array([(qpu_ctrl[t] == 0).mean() for t in range(num_tiles)])
    p0_gpu = np.array([(gpu_ctrl[t] == 0).mean() for t in range(num_tiles)])
    M_qpu = np.array([matrix_entry_from_p0(p) for p in p0_qpu])
    M_gpu = np.array([matrix_entry_from_p0(p) for p in p0_gpu])
    baseline_mae = float(np.mean(np.abs(M_qpu - M_gpu)))

    p0_corr = np.array([correct_p0_split(p0_qpu[t], fits[t],
                                          ANGLES_A[PAIRS[t][0]], ANGLES_B[PAIRS[t][1]])
                        for t in range(num_tiles)])
    M_corr = np.array([matrix_entry_from_p0(p) for p in p0_corr])
    corr_mae = float(np.mean(np.abs(M_corr - M_gpu)))
    rel = (baseline_mae - corr_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
    print(f"  Baseline  MAE : {baseline_mae:.4e}")
    print(f"  Corrected MAE : {corr_mae:.4e}")
    print(f"  Change        : {baseline_mae - corr_mae:+.4e}   ({rel:+.1f}%)")

    # ---- B.6 Coherent-drift scan (per-tile)
    if not skip_drift:
        section("B.6 — COHERENT ANGLE-DRIFT SCAN (per-tile, no channel)")
        print("  After channel fit, try (theta_a + d_a, theta_b + d_b) on bare T3.")
        print("  Persistent nonzero (d_a, d_b) -> coherent rotation error.\n")
        print(f"  {'tile':>4} {'(r,c)':>7} | {'best d_a':>10} {'best d_b':>10} "
              f"{'TVD_drift':>11} {'TVD_T3':>9}")
        hline()
        d_a_list, d_b_list = [], []
        for t in range(num_tiles):
            r, c = PAIRS[t]
            tvd_d, d_a, d_b = fit_coherent_drift_per_tile(
                q_qpu_list[t], ANGLES_A[r], ANGLES_B[c],
                search_range=0.4, n=21)
            d_a_list.append(d_a); d_b_list.append(d_b)
            print(f"  {t:>4} ({r},{c})   {d_a:>+10.4f} {d_b:>+10.4f} "
                  f"{tvd_d:>11.4f} {fits[t]['tvd_unfit']:>9.4f}")
        print(f"\n  Mean d_a = {np.mean(d_a_list):+.4f}   std = {np.std(d_a_list):.4f}")
        print(f"  Mean d_b = {np.mean(d_b_list):+.4f}   std = {np.std(d_b_list):.4f}")
        print(f"  -> If means are large and stds small, global calibration shift.")
        print(f"  -> If stds large, drift is per-tile (layout-dependent). See Phase C.")

    print(f"\n  Summary: lam_d={mean['lam_d']:.3f}  lam_g={mean['lam_g']:.3f}  "
          f"lam_rc={mean['lam_rc']:.3f}  lam_rg={mean['lam_rg']:.3f}")
    print(f"           Reduction {rel:+.1f}%   (Phase A had ~0% or worse.)")

    return {"q_qpu_list": q_qpu_list, "q_gpu_list": q_gpu_list,
            "p_t3_list": p_t3_list, "n_shots": n_shots,
            "fits": fits, "baseline_mae": baseline_mae, "corr_mae": corr_mae,
            "mean": mean}


# =============================================================================
# PHASE C IMPLEMENTATION
# =============================================================================
def run_phase_C(qpu_ctrl, qpu_ghost, gpu_ctrl, gpu_ghost, num_tiles,
                n_alternations=4, lam_out=0.5):
    section("PHASE C — DRIFT-FIRST ALTERNATING OPTIMIZATION")
    print(f"  Tiles : {num_tiles}")
    print(f"  Alternations : {n_alternations}")
    print(f"  OUT regularization lambda : {lam_out}")

    q_qpu_list = []
    counts_qpu = []
    for t in range(num_tiles):
        q_q, c_q = shots_to_distribution(qpu_ctrl[t], qpu_ghost[t])
        q_qpu_list.append(q_q)
        counts_qpu.append(c_q)
    n_shots = int(counts_qpu[0].sum())
    p_t3_list = all_t3_joints(num_tiles)

    # ---- C.1 Initial distribution metrics
    section("C.1 — INITIAL DISTRIBUTION METRICS (vs T3, NO DRIFT)")
    floor_mean, floor_std, _, _ = shot_noise_floor(p_t3_list[num_tiles // 2], n_shots, seed=42)
    print(f"  Shot-noise TVD floor: {floor_mean:.4f} +/- {floor_std:.4f}\n")
    print(f"  {'tile':>4} {'(r,c)':>7} | {'TVD(QPU,T3)':>12} {'sigmas':>8}")
    hline()
    for t in range(num_tiles):
        r, c = PAIRS[t]
        tvd_q = tvd(q_qpu_list[t], p_t3_list[t])
        sig = (tvd_q - floor_mean) / max(floor_std, 1e-6)
        print(f"  {t:>4} ({r},{c})    {tvd_q:>11.4f}  {sig:>+7.1f}s")

    # ---- C.2 Shared drift (no channels)
    section("C.2 — INITIAL SHARED DRIFT FIT (no channels)")
    d_a, d_b, L = fit_shared_drift(q_qpu_list, num_tiles, channel_fits=None,
                                    n_grid=41, search_range=0.5, n_local=2000, seed=0)
    print(f"  Shared (d_a, d_b) = ({d_a:+.4f}, {d_b:+.4f}) rad")
    print(f"  Total TVD with shared drift: {L * num_tiles:.4f}  (mean {L:.4f})")

    # ---- C.3 Initial channels on drift-corrected reference
    section("C.3 — INITIAL CHANNEL FIT (drift applied)")
    channel_fits = []
    print(f"  {'tile':>4} {'(r,c)':>7} | {'lam_d':>7} {'lam_g':>7} "
          f"{'lam_rc':>7} {'lam_rg':>7} {'eps_c':>7} {'eps_g':>7} | {'TVD_fit':>8}")
    hline()
    for t in range(num_tiles):
        r, c = PAIRS[t]
        fit = fit_split_readout(q_qpu_list[t],
                                 ANGLES_A[r] + d_a, ANGLES_B[c] + d_b,
                                 n_grid=6, n_local=1500, seed=t)
        channel_fits.append(fit)
        print(f"  {t:>4} ({r},{c})   "
              f"{fit['lam_d']:>7.3f} {fit['lam_g']:>7.3f} "
              f"{fit['lam_rc']:>7.3f} {fit['lam_rg']:>7.3f} "
              f"{fit['eps_ctrl']:>7.3f} {fit['eps_ghost']:>7.3f} | "
              f"{fit['tvd_fit']:>8.4f}")

    # ---- C.4 Alternating drift <-> channel
    section("C.4 — ALTERNATING DRIFT <-> CHANNEL OPTIMIZATION")
    print(f"  {'iter':>4} | {'d_a':>10} {'d_b':>10} | "
          f"{'mean eps_a':>12} {'mean eps_b':>12} | {'mean TVD':>10}")
    hline()
    eps_a = np.zeros(num_tiles)
    eps_b = np.zeros(num_tiles)
    history = []
    for it in range(n_alternations):
        d_a_new, d_b_new, _ = fit_shared_drift(
            q_qpu_list, num_tiles, channel_fits=channel_fits,
            n_grid=21, search_range=0.2, n_local=800, seed=it)
        d_a += d_a_new
        d_b += d_b_new
        eps_a, eps_b, _ = fit_residual_drift(
            q_qpu_list, num_tiles, d_a, d_b, channel_fits=channel_fits,
            lam_out=lam_out, n_iter=800, seed=it)
        new_fits = []
        for t in range(num_tiles):
            r, c = PAIRS[t]
            theta_a_eff = ANGLES_A[r] + d_a + eps_a[t]
            theta_b_eff = ANGLES_B[c] + d_b + eps_b[t]
            new_fits.append(fit_split_readout(q_qpu_list[t], theta_a_eff, theta_b_eff,
                                              n_grid=5, n_local=800, seed=t + 100 * it))
        channel_fits = new_fits
        mean_tvd = sum(f['tvd_fit'] for f in channel_fits) / num_tiles
        history.append((d_a, d_b, eps_a.mean(), eps_b.mean(), mean_tvd))
        warn = ""
        if it > 0 and mean_tvd > history[-2][4] + 1e-4:
            warn = "  (mean TVD increased — alternation is chasing channel residuals)"
        print(f"  {it+1:>4}   {d_a:>+10.4f} {d_b:>+10.4f}   "
              f"{eps_a.mean():>+12.4f} {eps_b.mean():>+12.4f}   {mean_tvd:>10.4f}{warn}")
        if it > 0 and abs(history[-1][4] - history[-2][4]) < 1e-4:
            print(f"\n  Converged (mean TVD change < 1e-4)")
            break

    # ---- C.5 Where did the residual go?
    section("C.5 — RESIDUAL BREAKDOWN")
    print(f"  {'tile':>4} {'(r,c)':>7} | {'eps_a^t':>10} {'eps_b^t':>10} | "
          f"{'lam_g':>7} {'lam_rg':>7} {'eps_g':>7}")
    hline()
    for t in range(num_tiles):
        r, c = PAIRS[t]
        print(f"  {t:>4} ({r},{c})   "
              f"{eps_a[t]:>+10.4f} {eps_b[t]:>+10.4f}   "
              f"{channel_fits[t]['lam_g']:>7.3f} "
              f"{channel_fits[t]['lam_rg']:>7.3f} "
              f"{channel_fits[t]['eps_ghost']:>7.3f}")
    eps_a_std = eps_a.std(); eps_b_std = eps_b.std()
    print(f"\n  Shared drift converged at (d_a, d_b) = ({d_a:+.4f}, {d_b:+.4f}) rad")
    print(f"                                       ({math.degrees(d_a):+.2f}, {math.degrees(d_b):+.2f}) deg")
    print(f"  Residual per-tile drift std: eps_a={eps_a_std:.4f}, eps_b={eps_b_std:.4f}")
    if eps_a_std < 0.02 and eps_b_std < 0.02:
        print("    -> per-tile drift collapsed to ~zero: shared drift captures it all.")
    elif eps_a_std < 0.05 and eps_b_std < 0.05:
        print("    -> small per-tile residual: shared drift is the dominant story.")
    else:
        print("    -> substantial per-tile drift remains: likely layout-dependent.")

    # ---- C.6 Reduction
    section("C.6 — CORRECTED REDUCTION (drift + channel)")
    p0_qpu = np.array([(qpu_ctrl[t] == 0).mean() for t in range(num_tiles)])
    p0_gpu = np.array([(gpu_ctrl[t] == 0).mean() for t in range(num_tiles)])
    M_qpu = np.array([matrix_entry_from_p0(p) for p in p0_qpu])
    M_gpu = np.array([matrix_entry_from_p0(p) for p in p0_gpu])
    baseline_mae = float(np.mean(np.abs(M_qpu - M_gpu)))

    p0_corr = []
    for t in range(num_tiles):
        r, c = PAIRS[t]
        theta_a_eff = ANGLES_A[r] + d_a + eps_a[t]
        theta_b_eff = ANGLES_B[c] + d_b + eps_b[t]
        p0c = correct_p0_split(p0_qpu[t], channel_fits[t], theta_a_eff, theta_b_eff)
        p0_corr.append(p0c)
    p0_corr = np.array(p0_corr)
    M_corr = np.array([matrix_entry_from_p0(p) for p in p0_corr])
    corr_mae = float(np.mean(np.abs(M_corr - M_gpu)))
    rel = (baseline_mae - corr_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
    print(f"  Baseline  MAE : {baseline_mae:.4e}")
    print(f"  Corrected MAE : {corr_mae:.4e}")
    print(f"  Change        : {baseline_mae - corr_mae:+.4e}   ({rel:+.1f}%)")

    return {"q_qpu_list": q_qpu_list, "p_t3_list": p_t3_list,
            "channel_fits": channel_fits, "d_a": d_a, "d_b": d_b,
            "eps_a": eps_a, "eps_b": eps_b, "n_shots": n_shots,
            "baseline_mae": baseline_mae, "corr_mae": corr_mae}


# =============================================================================
# PHASE D IMPLEMENTATION
# =============================================================================
def run_phase_D(qpu_ctrl, qpu_ghost, gpu_ctrl, gpu_ghost, num_tiles, n_null=200):
    section("PHASE D — BENFORD / p-ADIC NULL SWEEP")
    print(f"  Tiles : {num_tiles}")
    print(f"  Benford bases tested : {BENFORD_BASES}")
    print(f"  Valuation bases tested : {VALUATION_BASES}")
    print(f"  Null shuffles per metric : {n_null}")

    q_qpu_list = []
    counts_qpu = []
    for t in range(num_tiles):
        q_q, c_q = shots_to_distribution(qpu_ctrl[t], qpu_ghost[t])
        q_qpu_list.append(q_q)
        counts_qpu.append(c_q)
    n_shots = int(counts_qpu[0].sum())
    p_t3_list = all_t3_joints(num_tiles)

    # ---- D.1 Recompute the two correction paths
    section("D.1 — RECOMPUTING PATH B (split-readout, no drift)")
    fits_B = []
    for t in range(num_tiles):
        r, c = PAIRS[t]
        fits_B.append(fit_split_readout(q_qpu_list[t], ANGLES_A[r], ANGLES_B[c],
                                         n_local=1500, seed=t))
    print(f"  Mean lam_g (Path B) = {np.mean([f['lam_g'] for f in fits_B]):.3f}")

    section("D.2 — RECOMPUTING PATH C-LITE (shared drift + channel only, no alternation)")
    d_a, d_b, _ = fit_shared_drift(q_qpu_list, num_tiles, n_grid=41,
                                    search_range=0.5, n_local=2000, seed=0)
    print(f"  Shared drift (d_a, d_b) = ({d_a:+.4f}, {d_b:+.4f}) rad")
    fits_C = []
    for t in range(num_tiles):
        r, c = PAIRS[t]
        fits_C.append(fit_split_readout(q_qpu_list[t],
                                         ANGLES_A[r] + d_a, ANGLES_B[c] + d_b,
                                         n_local=1500, seed=t + 1000))
    print(f"  Mean lam_g (Path C) = {np.mean([f['lam_g'] for f in fits_C]):.3f}")

    # ---- D.3 Build four residual matrices
    section("D.3 — BUILDING FOUR RESIDUAL MATRICES")
    R_B_raw = np.stack([q_qpu_list[t] - p_t3_list[t] for t in range(num_tiles)], axis=0)
    R_B_post = np.stack([q_qpu_list[t] - fits_B[t]['q_model'] for t in range(num_tiles)], axis=0)
    p_t3_drifted = [t3_joint(ANGLES_A[r] + d_a, ANGLES_B[c] + d_b)
                    for (r, c) in PAIRS[:num_tiles]]
    R_C_raw = np.stack([q_qpu_list[t] - p_t3_drifted[t] for t in range(num_tiles)], axis=0)
    R_C_post = np.stack([q_qpu_list[t] - fits_C[t]['q_model'] for t in range(num_tiles)], axis=0)
    matrices = {"B_raw": R_B_raw, "B_post": R_B_post,
                "C_raw": R_C_raw, "C_post": R_C_post}
    print(f"  {'name':<10} | {'TVD_mean':>10} {'TVD_max':>10} {'Frobenius':>12}")
    hline()
    for name, R in matrices.items():
        tvd_per_tile = 0.5 * np.abs(R).sum(axis=1)
        fro = np.linalg.norm(R)
        print(f"  {name:<10}   {tvd_per_tile.mean():>10.4f} "
              f"{tvd_per_tile.max():>10.4f} {fro:>12.4f}")

    # ---- D.4 Z-score table
    section("D.4 — Z-SCORES vs SHUFFLED-WITHIN-TILE NULL")
    print("  |z| > 3 (***), |z| > 2 (** ), |z| > 1 (*  )\n")
    header = (f"  {'path':<8} {'residual':<8} {'ordering':<12} "
              f"{'metric':<12} | "
              + " ".join(f"{f'base{b}':>10}" for b in BENFORD_BASES)
              + f" {'recursive':>10}")
    print(header)
    hline()

    def flag(z):
        if abs(z) > 3: return "***"
        if abs(z) > 2: return "** "
        if abs(z) > 1: return "*  "
        return "   "

    orderings = [
        ("row-major", lambda R: order_row_major(R)),
        ("col-major", lambda R: order_col_major(R)),
        ("svd-mode1", lambda R: order_svd(R, k=1)[0][0]),
    ]

    for path_name, R in matrices.items():
        path_label = "B" if path_name.startswith("B") else "C"
        residual_label = path_name.split("_")[1]
        for ord_name, ord_fn in orderings:
            stream = ord_fn(R)
            bz = []
            for base in BENFORD_BASES:
                observed = benford_score(stream, base)
                null = null_distribution_benford(R, ord_fn, base, n_null=n_null, seed=42)
                z = (observed - null.mean()) / max(null.std(), 1e-9)
                bz.append((observed, z))
            obs_rec, _ = recursive_manifold_analysis(stream)
            null_rec = null_distribution_recursive(R, ord_fn, n_null=n_null, seed=42)
            z_rec = (obs_rec - null_rec.mean()) / max(null_rec.std(), 1e-9)
            row = (f"  {path_label:<8} {residual_label:<8} {ord_name:<12} "
                   f"{'Benford_z':<12} | "
                   + " ".join(f"{z:>+7.2f}{flag(z)}" for _, z in bz)
                   + f" {z_rec:>+7.2f}{flag(z_rec)}")
            print(row)
            chi_row = (f"  {'':<8} {'':<8} {'':<12} {'chi2/dof':<12} | ")
            for base in BENFORD_BASES:
                chi2, dof = benford_chi2(stream, base)
                if dof > 0:
                    chi_row += f"{chi2/dof:>9.2f}   "
                else:
                    chi_row += f"{'--':>10}  "
            chi_row += f" {'--':>10}"
            print(chi_row)

    # ---- D.5 p-adic valuation
    section("D.5 — p-ADIC VALUATION STRUCTURE TESTS")
    print(f"  {'path':<8} {'residual':<8} | "
          + " ".join(f"{f'v_{p}_chi2/dof':>14}" for p in VALUATION_BASES))
    hline()
    for path_name, R in matrices.items():
        path_label = "B" if path_name.startswith("B") else "C"
        residual_label = path_name.split("_")[1]
        row = f"  {path_label:<8} {residual_label:<8} | "
        for p in VALUATION_BASES:
            chi2, dof = valuation_chi2(R, p, n_shots)
            if dof > 0:
                row += f"{chi2/dof:>14.2f}"
            else:
                row += f"{'--':>14}"
        print(row)

    print("\n  Interpretation:")
    print("    Benford z-scores: |z|>3 strong, |z|>2 suggestive, |z|<1 no signal.")
    print("    A few scattered |z|>2 hits across many cells are expected by chance;")
    print("    look for cross-base consistency for a real signal.")
    print("    v_p chi2/dof ~ 1 means random-integer (geometric) distribution.")
    print("    v_p chi2/dof > 5 would indicate divisibility / combinatorial structure.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Ghost Oracle Suite — Probe 8: Residual Decomposition. "
                    "Four phases (A: 3-channel; B: split-readout; C: drift-first; "
                    "D: Benford/p-adic null sweep) that characterize the QPU's "
                    "structural deviation from T3. The negative cumulative finding "
                    "motivated the Probe 9 G_M derivation. Run any subset of phases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--phase", default="all",
                    help="Comma-separated subset of {A,B,C,D,all}. Phases run in order.")
    ap.add_argument("--qpu", default=None,
                    help="Path to QPU base .npz (auto-finds job_*.npz in data/ if omitted).")
    ap.add_argument("--gpu", default=None,
                    help="Path to noiseless GPU base .npz "
                         "(auto-finds ghost_oracle_gpu_*.npz or ghost_oracle_gpu_*.npz in data/ if omitted).")
    ap.add_argument("--num-tiles", type=int, default=NUM_TILES,
                    help="Number of tiles in the bases.")
    ap.add_argument("--skip-drift", action="store_true",
                    help="In Phase B, skip the per-tile coherent-drift scan (saves ~10s).")
    ap.add_argument("--n-alternations", type=int, default=1,
                    help="Phase C: number of drift<->channel alternations. "
                         "Default 1 matches the historical run; higher values "
                         "let the optimizer wind the angles past 2pi.")
    ap.add_argument("--lam-out", type=float, default=0.5,
                    help="Phase C: OUT regularization on per-tile residual drift.")
    ap.add_argument("--n-null", type=int, default=200,
                    help="Phase D: number of within-tile shuffles per null distribution.")
    args = ap.parse_args()

    phases = [p.strip().upper() for p in args.phase.split(",")]
    if "ALL" in phases:
        phases = ["A", "B", "C", "D"]
    valid = {"A", "B", "C", "D"}
    bad = [p for p in phases if p not in valid]
    if bad:
        sys.exit(f"[FATAL] unknown phase(s): {bad}. Choices: A, B, C, D, all.")

    qpu_path = args.qpu or auto_find_base("qpu")
    gpu_path = args.gpu or auto_find_base("gpu")
    if not qpu_path or not gpu_path:
        sys.exit(f"[FATAL] Probe 8 needs both a QPU base and a GPU base. "
                 f"Found qpu={qpu_path}, gpu={gpu_path}. Pass --qpu and --gpu "
                 f"or put job_*.npz / ghost_oracle_gpu_*.npz in {DATA_DIR}/")

    print("\n" + "=" * 92)
    print("  GHOST ORACLE SUITE — PROBE 8 — RESIDUAL DECOMPOSITION")
    print("=" * 92)
    print(f"  QPU base : {qpu_path}")
    print(f"  GPU base : {gpu_path}")
    print(f"  Phases   : {','.join(phases)}")
    print(f"  Tiles    : {args.num_tiles}")
    print(f"  Angles A : {np.round(ANGLES_A, 4)}")
    print(f"  Angles B : {np.round(ANGLES_B, 4)}")

    qpu_ctrl, qpu_ghost, qpu_label = load_base(qpu_path, args.num_tiles)
    gpu_ctrl, gpu_ghost, gpu_label = load_base(gpu_path, args.num_tiles)
    print(f"  QPU job_id: {qpu_label}")
    print(f"  GPU job_id: {gpu_label}")

    if "A" in phases:
        run_phase_A(qpu_ctrl, qpu_ghost, gpu_ctrl, gpu_ghost, args.num_tiles)
    if "B" in phases:
        run_phase_B(qpu_ctrl, qpu_ghost, gpu_ctrl, gpu_ghost, args.num_tiles,
                    skip_drift=args.skip_drift)
    if "C" in phases:
        run_phase_C(qpu_ctrl, qpu_ghost, gpu_ctrl, gpu_ghost, args.num_tiles,
                    n_alternations=args.n_alternations, lam_out=args.lam_out)
    if "D" in phases:
        run_phase_D(qpu_ctrl, qpu_ghost, gpu_ctrl, gpu_ghost, args.num_tiles,
                    n_null=args.n_null)

    print()


if __name__ == "__main__":
    main()
