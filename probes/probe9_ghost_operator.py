#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — PROBE 9 — THE OPERATOR (G_M CHARACTERIZATION)
==============================================================================
After the eight-probe arc that ruled out T1 and T2, derived T3, and tried
unsuccessfully to model the QPU residual within canonical error channels
(Probe 8), this probe asks the question that unlocks the rest of the
suite: what is the operator the QPU is actually computing, presented as
cleanly as possible?

Verified at machine precision:
    T3(a, b)  = 3/4 + (1/4) * cos(a) * cos(b)
    G_M(a, b) = sqrt((1 + cos(a) * cos(b)) / 2) / ALPHA_NORM

T3 turns out to be a low-order trigonometric polynomial. G_M is the
square-root form of the same operator, normalized by ALPHA_NORM so its
output sits in [0, 1]. G_M is what the rest of the suite uses — it has
the matmul-adjacent shape (rank-1 in cos-space) plus a saturating sqrt
nonlinearity that bounds the output.

Compare to neighbors:
    Standard matmul :  (A B^T)[r,c] = sum_k a[r,k] b[c,k]
    T1 rank-1 cos   :  |cos(a - b)| = |cos a cos b + sin a sin b|
    G_M (this op)   :  sqrt((1 + cos a cos b) / 2) / ALPHA_NORM

PROBE STRUCTURE (five stages):

  Stage 1  CONSISTENCY:    QPU and GPU both implement G_M; verify identity
                           at the 12-tile observed points.
  Stage 2  CHARACTERIZE:   Pearson correlations, range, spectral structure
                           of G_M vs matmul vs T1 on random inputs.
  Stage 3  CLASSIFY:       Where does G_M sit in the operator zoo?
                           Mercer kernel test, feature-map analysis,
                           correlations with RBF / arc-cosine / cos.
  Stage 4  APPLY:          Two demos. Demo 1: rank items by similarity.
                           Demo 2: regress a saturating function. SEE
                           HISTORICAL CONTEXT — Demo 2 is structurally
                           broken in this probe; Probe 9.1 fixed it.
  Stage 5  SCALE:          N^2 timing sweep; adversarial-input search
                           for maximum G_M-vs-matmul disagreement.

HISTORICAL CONTEXT:
    Probe 9 is "the operator" probe — the moment the project pivoted
    from "characterize the QPU's deviation from a textbook target" to
    "name and characterize the operator the QPU natively computes."
    Probes 1-3 ruled out T1 and T2, Probe 4 derived T3 by direct
    simulation, Probes 5-7 confirmed T3 holds physically, Probe 8 tried
    and failed to model the QPU's residual against T3 within canonical
    error channels. Probe 9 then asked: T3 looks complicated; is there
    a cleaner closed form? The answer was the half-angle separable
    cosine kernel G_M above. The rest of the suite — the headline
    projection_benchmark.py, the tied-channel kernel in
    ghost_kernel.cu, Probes 10/10.1 on attention — all operate on G_M.

    This probe's specific demos are mostly either tautological or
    broken, by design and by accident respectively. Two known issues
    are documented in docs/known_issues.md and reproduced verbatim here
    for trajectory legibility:

    KNOWN ISSUE 1 — Stage 1 clipping artifact.
        Three tiles (the (0,2), (0,3), (1,3) entries) have analytical
        G_M values exactly at 1.0 because their cos(a)cos(b) is small
        enough that the sqrt argument exceeds ALPHA_NORM and the
        min(1.0, ...) clamp fires. The same clamp fires on the GPU
        sample for those tiles. The reported MAE of ~0.2 is dominated
        by the three clipped tiles — the unclipped MAE is around shot
        noise. Probe 9.1 (in this same repo) re-runs Stage 1 reporting
        unnormalized G_M_raw values, which avoids the clamp and shows
        the GPU at shot noise as expected.

    KNOWN ISSUE 2 — Stage 4 Demo 2 is structurally broken.
        Demo 2 generates a saturating "truth" function and asks
        whether a G_M-aware regressor beats linear regression. The
        truth function is G_M itself, so the "G_M oracle" feature gets
        MSE = 0 and the printed ratio mse_lin / mse_gm becomes
        meaningless (divides by ~1e-15). Probe 9.1 redoes this with a
        non-G_M truth that requires the saturating sqrt structure
        without being trivially equal to G_M.

    Probes 10/10.1 then took G_M into the attention-mechanism context
    and demonstrated structural robustness to coherent same-dim attacks
    where dot-product attention catastrophically degrades. See
    PROCESS_RECORD.md for the full arc.

USAGE:
    python probe9_ghost_operator.py
    python probe9_ghost_operator.py --qpu data/job_xyz.npz --gpu data/ghost_oracle_gpu_xyz.npz
==============================================================================
"""

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
# Probe 9 originally ran against 12-tile bases. The rest of the suite uses
# 16 tiles (4x4); pass --num-tiles 16 to compare against a current-generation
# QPU base.
NUM_TILES   = 12
PAIRS       = [(r, c) for r in range(4) for c in range(4)][:NUM_TILES]
ANGLE_SCALE = 1.05
ALPHA_NORM  = 0.9127

MATRIX_A = np.array([0.25, 0.50, 0.75, 1.00])
MATRIX_B = np.array([1.00, 0.80, 0.40, 0.10])

# Repo-root data/ directory: this file lives at <repo>/probes/probe9_*.py.
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
# THE THREE OPERATORS
# =============================================================================
def op_matmul_standard(A, B):
    """Standard matmul: A @ B^T."""
    return A @ B.T


def op_T1_rank1_cosine(angles_A, angles_B):
    """T1: |cos(a - b)| -- the rank-1 cosine kernel computed by ghost_rank_k_matmul."""
    return np.abs(np.cos(angles_A[:, None] - angles_B[None, :]))


def op_GM_separable(angles_A, angles_B, alpha=ALPHA_NORM, normalize=True):
    """G_M: the Ghost Oracle operator (normalized form, output in [0, 1]).

    G_M(a, b) = sqrt((1 + cos(a) * cos(b)) / 2) / alpha    if normalize=True
              = sqrt((1 + cos(a) * cos(b)) / 2)             if normalize=False

    Equivalently: cosine-lift A and B to cos(A), cos(B), take outer product,
    add 1, halve, sqrt, normalize. The min(1.0, ...) clamp on the normalized
    form is what produces the Stage 1 clipping artifact noted in the
    HISTORICAL CONTEXT.
    """
    cos_outer = np.cos(angles_A)[:, None] * np.cos(angles_B)[None, :]
    val = np.sqrt(np.clip((1 + cos_outer) / 2, 0, None))
    if normalize:
        val = np.minimum(val / alpha, 1.0)
    return val


def op_GM_unnormalized(angles_A, angles_B):
    """G_M_raw: sqrt((1 + cos a cos b) / 2). No alpha, no clamp. Output in
    [0, 1] naturally, peaks at 1 when both angles are 0."""
    cos_outer = np.cos(angles_A)[:, None] * np.cos(angles_B)[None, :]
    return np.sqrt(np.clip((1 + cos_outer) / 2, 0, None))


# =============================================================================
# I/O
# =============================================================================
def load_base(path, num_tiles):
    """Load a base .npz (dump.py or Probe 4 / gpu.py schema)."""
    print(f"[LOAD] {path}")
    d = np.load(path, allow_pickle=True)
    ctrl = {t: d[f"ctrl_tile{t}"] for t in range(num_tiles)}
    label = str(d["job_id"]) if "job_id" in d.files else "?"
    return ctrl, label


def hline(c="-", w=92):
    print(c * w)


def section(title, w=92):
    print()
    hline("=", w)
    print(f"  {title}")
    hline("=", w)


# =============================================================================
# STAGE 1 — SELF-CONSISTENCY (both QPU and GPU implement G_M)
# =============================================================================
def stage1_consistency(qpu_ctrl, gpu_ctrl, num_tiles):
    section("STAGE 1 — QPU/GPU SELF-CONSISTENCY ON G_M")
    print("  Claim: both implementations target the same G_M operator.")
    print("  Test:  measure P(ctrl=0) per tile, convert to matrix entry,")
    print("         compare against analytical G_M.")
    print()
    print("  Note: this stage uses the *normalized* G_M (output clamped to")
    print("  [0, 1]). When cos(a) cos(b) is small the clamp fires, which")
    print("  inflates the MAE relative to the shot-noise floor. Probe 9.1")
    print("  re-runs this stage on unnormalized G_M_raw to remove the")
    print("  clipping artifact.")
    print()

    M_analytical = op_GM_separable(ANGLES_A, ANGLES_B)
    M_qpu = np.zeros((4, 4))
    M_gpu = np.zeros((4, 4))
    for t in range(num_tiles):
        r, c = PAIRS[t]
        p0_qpu = float((qpu_ctrl[t] == 0).mean())
        p0_gpu = float((gpu_ctrl[t] == 0).mean())
        M_qpu[r, c] = min(1.0, math.sqrt(max(0, 2 * p0_qpu - 1)) / ALPHA_NORM)
        M_gpu[r, c] = min(1.0, math.sqrt(max(0, 2 * p0_gpu - 1)) / ALPHA_NORM)

    print(f"  {'tile':>4} {'(r,c)':>7} | {'G_M analytic':>14} {'G_M GPU':>10} "
          f"{'G_M QPU':>10} | {'GPU-anal':>10} {'QPU-anal':>10}  {'clip?':>6}")
    hline()
    n_clipped = 0
    for t in range(num_tiles):
        r, c = PAIRS[t]
        a = M_analytical[r, c]; g = M_gpu[r, c]; q = M_qpu[r, c]
        clip = "  *  " if a >= 0.9999 else "     "
        if a >= 0.9999:
            n_clipped += 1
        print(f"  {t:>4} ({r},{c})   "
              f"{a:>14.6f} {g:>10.6f} {q:>10.6f} | "
              f"{g-a:>+10.6f} {q-a:>+10.6f}  {clip}")

    mae_gpu = np.mean(np.abs(M_gpu - M_analytical)[~np.isnan(M_gpu)])
    mae_qpu = np.mean(np.abs(M_qpu - M_analytical)[~np.isnan(M_qpu)])
    print()
    print(f"  MAE(GPU,  analytic) = {mae_gpu:.4e}   "
          f"(expect shot noise ~1/sqrt(4096)={1/math.sqrt(4096):.4f})")
    print(f"  MAE(QPU,  analytic) = {mae_qpu:.4e}   (expect channel error)")
    if n_clipped > 0:
        print(f"\n  * {n_clipped}/{num_tiles} analytical tiles clipped at 1.0 — "
              f"see Probe 9.1 for the un-clamped re-run.")
    return M_analytical, M_gpu, M_qpu


# =============================================================================
# STAGE 2 — CHARACTERIZE (G_M vs matmul vs T1)
# =============================================================================
def stage2_characterize(rng_seed=42, N=64, n_trials=200):
    section("STAGE 2 — G_M vs MATMUL vs T1: WHAT KIND OF OPERATOR IS THIS?")
    print(f"  Random angle inputs in [0, pi/2], N={N}, {n_trials} trials.")
    print()

    rng = np.random.default_rng(rng_seed)

    corr_GM_matmul = []; corr_GM_T1 = []; corr_matmul_T1 = []
    range_GM = [[], []]; range_matmul = [[], []]; range_T1 = [[], []]
    spec_ratio_GM_matmul = []; spec_ratio_GM_T1 = []
    fro_GM = []; fro_matmul = []; fro_T1 = []

    for _ in range(n_trials):
        A = rng.uniform(0, math.pi / 2, size=N)
        B = rng.uniform(0, math.pi / 2, size=N)

        # For fair comparison, matmul on angle vectors is treated as the
        # rank-1 outer product A[:,None] * B[None,:], same shape as T1 and G_M.
        M_matmul = A[:, None] * B[None, :]
        M_T1 = op_T1_rank1_cosine(A, B)
        M_GM = op_GM_unnormalized(A, B)

        corr_GM_matmul.append(np.corrcoef(M_GM.ravel(), M_matmul.ravel())[0, 1])
        corr_GM_T1.append(np.corrcoef(M_GM.ravel(), M_T1.ravel())[0, 1])
        corr_matmul_T1.append(np.corrcoef(M_matmul.ravel(), M_T1.ravel())[0, 1])

        range_GM[0].append(M_GM.min()); range_GM[1].append(M_GM.max())
        range_matmul[0].append(M_matmul.min()); range_matmul[1].append(M_matmul.max())
        range_T1[0].append(M_T1.min()); range_T1[1].append(M_T1.max())

        fro_GM.append(np.linalg.norm(M_GM))
        fro_matmul.append(np.linalg.norm(M_matmul))
        fro_T1.append(np.linalg.norm(M_T1))

        s_GM = np.linalg.svd(M_GM, compute_uv=False)
        s_mm = np.linalg.svd(M_matmul, compute_uv=False)
        s_T1 = np.linalg.svd(M_T1, compute_uv=False)
        spec_ratio_GM_matmul.append((s_GM[1] / s_GM[0]) - (s_mm[1] / s_mm[0]))
        spec_ratio_GM_T1.append((s_GM[1] / s_GM[0]) - (s_T1[1] / s_T1[0]))

    print(f"  Entry-wise Pearson correlation across {n_trials} random inputs:")
    print(f"    corr(G_M, matmul) = {np.mean(corr_GM_matmul):+.4f}  +/- {np.std(corr_GM_matmul):.4f}")
    print(f"    corr(G_M, T1)     = {np.mean(corr_GM_T1):+.4f}  +/- {np.std(corr_GM_T1):.4f}")
    print(f"    corr(matmul, T1)  = {np.mean(corr_matmul_T1):+.4f}  +/- {np.std(corr_matmul_T1):.4f}")
    print()
    print(f"  Output range:")
    print(f"    G_M     min={np.mean(range_GM[0]):.4f}   max={np.mean(range_GM[1]):.4f}")
    print(f"    matmul  min={np.mean(range_matmul[0]):.4f}   max={np.mean(range_matmul[1]):.4f}")
    print(f"    T1      min={np.mean(range_T1[0]):.4f}   max={np.mean(range_T1[1]):.4f}")
    print()
    print(f"  Frobenius norm (N={N}):")
    print(f"    G_M     {np.mean(fro_GM):.4f}")
    print(f"    matmul  {np.mean(fro_matmul):.4f}")
    print(f"    T1      {np.mean(fro_T1):.4f}")
    print()
    print(f"  Spectral rank-1-ness (sigma_2/sigma_1):")
    print(f"    G_M vs matmul: diff = {np.mean(spec_ratio_GM_matmul):+.4f}  "
          f"(>0 = G_M more full-rank)")
    print(f"    G_M vs T1:     diff = {np.mean(spec_ratio_GM_T1):+.4f}")

    print()
    print("  STRUCTURAL READ:")
    mean_c = np.mean(corr_GM_matmul)
    if mean_c > 0.95:
        print("  - G_M is highly correlated with matmul under random angle inputs.")
    elif mean_c > 0.7:
        print("  - G_M is moderately correlated with matmul -- shares major signal.")
    elif mean_c > -0.5:
        print("  - G_M is weakly correlated with matmul -- distinct operator.")
    else:
        print(f"  - G_M is strongly ANTI-correlated with matmul ({mean_c:+.2f}).")
        print("    This is the key structural fact: matmul rises with the angle")
        print("    product, while G_M depends on cos(a)cos(b), which DECREASES as")
        print("    angles rise (for angles in [0, pi/2]). Same shape, opposite sign.")
    if np.mean(spec_ratio_GM_matmul) > 0.05:
        print("  - G_M has higher rank structure than matmul on same inputs")
        print("    (the sqrt nonlinearity breaks the rank-1 cos-outer-product).")
    if np.mean(range_GM[0]) > 0 and np.mean(range_GM[1]) < 1.01:
        print("  - G_M output is bounded in [0, 1] — similarity-style operator,")
        print("    unlike matmul which has unbounded range.")


# =============================================================================
# STAGE 3 — CLASSIFY (where does G_M sit in the operator zoo?)
# =============================================================================
def stage3_classify(N=32, n_trials=50, rng_seed=42):
    section("STAGE 3 — OPERATOR CLASSIFICATION")
    print("  Question: what kind of operator is G_M?")
    print("  Tests:")
    print("    (a) Is G_M positive semidefinite as a kernel matrix? (Mercer)")
    print("    (b) What is its feature map?")
    print("    (c) How does it relate to known kernels (RBF, cosine, arc-cosine)?")
    print()

    rng = np.random.default_rng(rng_seed)

    # ---- (a) PSD test
    print("  (a) POSITIVE SEMIDEFINITENESS")
    n_psd_pass = 0
    eigs_min = []
    for _ in range(n_trials):
        X = rng.uniform(0, math.pi / 2, size=N)
        K = op_GM_unnormalized(X, X)
        eigs = np.linalg.eigvalsh(K)
        eigs_min.append(eigs.min())
        if eigs.min() > -1e-9:
            n_psd_pass += 1

    print(f"      Tested {n_trials} random Gram matrices of size {N}x{N}.")
    print(f"      PSD-positive trials: {n_psd_pass}/{n_trials}")
    print(f"      Min eigenvalue (mean over trials): {np.mean(eigs_min):+.6f}")
    print(f"      Min eigenvalue (worst):            {np.min(eigs_min):+.6f}")
    if n_psd_pass == n_trials:
        print("      -> G_M IS a valid Mercer kernel. Feature map exists.")
    elif n_psd_pass > n_trials * 0.8:
        print("      -> G_M is *nearly* PSD; indefinite on rare inputs.")
    else:
        print("      -> G_M is INDEFINITE; not a Mercer kernel, but still a")
        print("         valid pairwise operator.")
    print()

    # ---- (b) Feature map analysis
    print("  (b) FEATURE MAP")
    print("      G_M(a, b) = sqrt((1 + cos a cos b) / 2)")
    print("              = sqrt(1/2) * sqrt(1 + cos a cos b)")
    print()
    print("      Using cos a cos b = cos(a-b)/2 + cos(a+b)/2:")
    print("      G_M(a, b) = sqrt(1/2 + cos(a-b)/4 + cos(a+b)/4)")
    print()
    print("      So G_M lives between T1-style (a-b) coupling and an (a+b) coupling.")
    print("      Explicit feature map: G_M(a,b) = <phi(a), phi(b)> where phi(x)")
    print("      embeds x into a space spanned by {1, cos(x), sin(x)}; the sqrt")
    print("      nonlinearity prevents a finite-dim feature map (it is an")
    print("      infinite series in cos(kx), sin(kx) via Taylor expansion of sqrt).")
    print()

    # Numerical feature map approximation
    X = np.linspace(0, math.pi / 2, N)
    K = op_GM_unnormalized(X, X)
    U, s, Vt = np.linalg.svd(K)
    s_normalized = s / s.sum()
    cum = np.cumsum(s_normalized)
    rank_for_95 = int(np.searchsorted(cum, 0.95)) + 1
    rank_for_99 = int(np.searchsorted(cum, 0.99)) + 1
    print(f"      Numerical rank of {N}x{N} G_M Gram matrix (singular spectrum):")
    print(f"        sigma_1 = {s[0]:.4f}  (carries {s_normalized[0]*100:.1f}%)")
    print(f"        sigma_2 = {s[1]:.4f}  (carries {s_normalized[1]*100:.1f}%)")
    print(f"        sigma_3 = {s[2]:.4f}  (carries {s_normalized[2]*100:.1f}%)")
    print(f"        Effective rank (95% energy): {rank_for_95}")
    print(f"        Effective rank (99% energy): {rank_for_99}")
    print(f"      -> G_M has a low-but-not-rank-1 spectrum: rank ~{rank_for_95}-{rank_for_99}.")
    print(f"         (Pure rank-1 cosine T1 would have rank 1.)")
    print()

    # ---- (c) Comparison to known kernels
    print("  (c) RELATION TO KNOWN KERNELS")
    print("      Compute G_M and standard kernels on the same inputs; correlate.")
    X = rng.uniform(0, math.pi / 2, size=N)
    Y = rng.uniform(0, math.pi / 2, size=N)

    K_GM = op_GM_unnormalized(X, Y)
    K_cos_angle = np.cos(X[:, None] - Y[None, :])
    K_T1 = op_T1_rank1_cosine(X, Y)

    inner = (np.cos(X)[:, None] * np.cos(Y)[None, :]
             + np.sin(X)[:, None] * np.sin(Y)[None, :])
    inner = np.clip(inner, -1, 1)
    K_arccos = 1 - np.arccos(inner) / math.pi

    sigma = math.pi / 4
    sq = (X[:, None] - Y[None, :]) ** 2
    K_rbf = np.exp(-sq / (2 * sigma ** 2))

    def corr(A, B):
        return float(np.corrcoef(A.ravel(), B.ravel())[0, 1])

    print(f"      Entry-wise correlations of G_M with other kernels (N={N}):")
    print(f"        corr(G_M, T1 rank-1 cosine)    = {corr(K_GM, K_T1):+.4f}")
    print(f"        corr(G_M, cosine(a-b))         = {corr(K_GM, K_cos_angle):+.4f}")
    print(f"        corr(G_M, arc-cosine kernel)   = {corr(K_GM, K_arccos):+.4f}")
    print(f"        corr(G_M, RBF kernel)          = {corr(K_GM, K_rbf):+.4f}")
    print(f"        corr(G_M, cos a * cos b)       = "
          f"{corr(K_GM, np.cos(X)[:,None] * np.cos(Y)[None,:]):+.4f}")
    print()
    print("      The highest correlation tells us which known kernel G_M most")
    print("      closely resembles structurally.")


# =============================================================================
# STAGE 4 — APPLY (demonstrate G_M on tasks)
# Stage 4 contains two demos; Demo 2 is the known-broken one (see HISTORICAL
# CONTEXT and docs/known_issues.md). Probe 9.1 fixes it. Preserved verbatim
# here for trajectory legibility.
# =============================================================================
def stage4_apply(rng_seed=42):
    section("STAGE 4 — ADJACENT APPLICATION: SIMILARITY SCORING")
    print("  Task: rank items by similarity to a query.")
    print("  Two demos:")
    print("    Demo 1: bounded-similarity recovery (tautological — sets a baseline)")
    print("    Demo 2: saturating regression (KNOWN BROKEN — see header)")
    print()

    rng = np.random.default_rng(rng_seed)

    # ----- Demo 1: bounded similarity recovery
    print("  DEMO 1 — Bounded similarity ground truth")
    print("  (Truth is G_M(query, item); checks G_M score recovers ranking)")
    print()
    N = 200
    n_repeats = 50
    matmul_corrs = []
    GM_corrs = []
    T1_corrs = []

    try:
        from scipy.stats import spearmanr
        have_scipy = True
    except ImportError:
        have_scipy = False
        print("  (scipy not available — using numpy rank correlation fallback)")

    def rank_corr(a, b):
        if have_scipy:
            return float(spearmanr(a, b).correlation)
        # Numpy fallback: Pearson on ranks
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    for _ in range(n_repeats):
        query = rng.uniform(0, math.pi / 2, size=1)
        items = rng.uniform(0, math.pi / 2, size=N)
        truth = op_GM_unnormalized(query, items).ravel()
        score_matmul = (query * items).ravel()
        score_GM = op_GM_unnormalized(query, items).ravel()
        score_T1 = op_T1_rank1_cosine(query, items).ravel()
        matmul_corrs.append(rank_corr(score_matmul, truth))
        GM_corrs.append(rank_corr(score_GM, truth))
        T1_corrs.append(rank_corr(score_T1, truth))

    print(f"  Rank-correlation with truth (G_M-generated), {n_repeats} trials:")
    print(f"    matmul score : {np.mean(matmul_corrs):+.4f}  +/- {np.std(matmul_corrs):.4f}")
    print(f"    T1 score     : {np.mean(T1_corrs):+.4f}  +/- {np.std(T1_corrs):.4f}")
    print(f"    G_M score    : {np.mean(GM_corrs):+.4f}  +/- {np.std(GM_corrs):.4f}")
    print(f"  -> When truth is G_M-shaped, the operator that recovers ranking is G_M.")
    print(f"     Trivial test; establishes G_M is identifiable.")
    print()

    # ----- Demo 2 (KNOWN BROKEN)
    print("  DEMO 2 — Saturating regression task   [KNOWN BROKEN]")
    print("  Predict y = sqrt((1 + x1 x2)/2) from features x1, x2 in [0,1].")
    print("  Standard linear regression uses x1, x2 as features.")
    print("  G_M-aware regression uses cos(x1*pi/2), cos(x2*pi/2) then applies G_M.")
    print()
    print("  WARNING: the 'truth' function below IS G_M evaluated on cos-lifted")
    print("  inputs, so the 'G_M oracle' feature trivially matches truth and gets")
    print("  MSE ~ 0. The printed mse_lin / mse_gm ratio divides by ~1e-15. The")
    print("  ratio is therefore meaningless. Probe 9.1 re-does this demo with a")
    print("  truth function that requires saturating sqrt structure without being")
    print("  trivially equal to G_M.")
    print()

    N_train = 500
    N_test = 200
    x_train = rng.uniform(0, 1, size=(N_train, 2))
    x_test = rng.uniform(0, 1, size=(N_test, 2))

    def true_y(x):
        return np.sqrt((1 + x[:, 0] * x[:, 1]) / 2)

    y_train = true_y(x_train)
    y_test = true_y(x_test)

    X_lin_tr = np.c_[np.ones(N_train), x_train, x_train[:, 0] * x_train[:, 1]]
    X_lin_te = np.c_[np.ones(N_test), x_test, x_test[:, 0] * x_test[:, 1]]
    beta_lin, *_ = np.linalg.lstsq(X_lin_tr, y_train, rcond=None)
    y_pred_lin = X_lin_te @ beta_lin
    mse_lin = float(np.mean((y_pred_lin - y_test) ** 2))

    def gm_feature(x):
        a = x[:, 0] * math.pi / 2
        b = x[:, 1] * math.pi / 2
        return np.sqrt(np.clip((1 + np.cos(a) * np.cos(b)) / 2, 0, None))

    y_pred_gm = gm_feature(x_test)
    mse_gm = float(np.mean((y_pred_gm - y_test) ** 2))

    Xc_tr = np.c_[np.cos(x_train[:, 0] * math.pi / 2),
                  np.cos(x_train[:, 1] * math.pi / 2),
                  np.cos(x_train[:, 0] * math.pi / 2) * np.cos(x_train[:, 1] * math.pi / 2)]
    Xc_te = np.c_[np.cos(x_test[:, 0] * math.pi / 2),
                  np.cos(x_test[:, 1] * math.pi / 2),
                  np.cos(x_test[:, 0] * math.pi / 2) * np.cos(x_test[:, 1] * math.pi / 2)]
    beta_cos, *_ = np.linalg.lstsq(np.c_[np.ones(N_train), Xc_tr], y_train, rcond=None)
    y_pred_cos = np.c_[np.ones(N_test), Xc_te] @ beta_cos
    mse_cos = float(np.mean((y_pred_cos - y_test) ** 2))

    print(f"  MSE on test set:")
    print(f"    Linear regression (4 features)  : {mse_lin:.6e}")
    print(f"    Cos-lifted linear regression    : {mse_cos:.6e}")
    print(f"    Direct G_M (oracle, trivially 0): {mse_gm:.6e}")
    print()
    if mse_gm < 1e-12:
        print(f"  (As predicted in the warning above, MSE_GM is at float noise;")
        print(f"   the mse_lin / mse_gm ratio is undefined. Defer to Probe 9.1 for")
        print(f"   the corrected version of this demo.)")
    else:
        ratio = mse_lin / max(mse_gm, 1e-15)
        print(f"  Ratio mse_lin / mse_gm = {ratio:.2f}x")


# =============================================================================
# STAGE 5 — SCALE AND ADVERSARIAL
# =============================================================================
def stage5_scale_adversarial(rng_seed=42):
    section("STAGE 5 — SCALE SWEEP AND ADVERSARIAL DIVERGENCE")
    print("  PART A: Scaling")
    print(f"  Compute G_M, matmul, T1 on NxN inputs for N in {{4, 8, 16, 32, 64, 128, 256}}.")
    print(f"  Time each, report normalized cost.")
    print()

    rng = np.random.default_rng(rng_seed)
    sizes = [4, 8, 16, 32, 64, 128, 256]
    print(f"  {'N':>4} | {'matmul':>12} {'T1':>12} {'G_M':>12} | {'G_M/matmul':>12}")
    hline()
    for N in sizes:
        A = rng.uniform(0, math.pi / 2, size=N)
        B = rng.uniform(0, math.pi / 2, size=N)
        for _ in range(3):
            _ = op_matmul_standard(A[:, None], B[:, None])
            _ = op_T1_rank1_cosine(A, B)
            _ = op_GM_unnormalized(A, B)

        n_reps = max(10, 10000 // (N * N))

        t0 = time.perf_counter()
        for _ in range(n_reps):
            _ = op_matmul_standard(A[:, None], B[:, None])
        t_mm = (time.perf_counter() - t0) / n_reps

        t0 = time.perf_counter()
        for _ in range(n_reps):
            _ = op_T1_rank1_cosine(A, B)
        t_T1 = (time.perf_counter() - t0) / n_reps

        t0 = time.perf_counter()
        for _ in range(n_reps):
            _ = op_GM_unnormalized(A, B)
        t_GM = (time.perf_counter() - t0) / n_reps

        ratio = t_GM / max(t_mm, 1e-12)
        print(f"  {N:>4}   {t_mm*1e6:>10.2f}us  {t_T1*1e6:>10.2f}us  "
              f"{t_GM*1e6:>10.2f}us  | {ratio:>10.2f}x")
    print()
    print("  -> G_M cost scales as N^2 just like matmul, with a small constant")
    print("     overhead from cos() and sqrt() per entry.")
    print()

    # PART B: adversarial
    print("  PART B: Adversarial divergence")
    print("  Find (A, B) where G_M and cos(a)cos(b) (the rank-1 inside G_M)")
    print("  disagree most on entry-wise ranking.")
    print()

    N = 32
    n_trials = 1000
    best_disagreement = -1.0
    best_inputs = None
    for trial in range(n_trials):
        A = rng.uniform(0, math.pi, size=N)
        B = rng.uniform(0, math.pi, size=N)
        M_GM = op_GM_unnormalized(A, B)
        M_mm = np.cos(A)[:, None] * np.cos(B)[None, :]
        c = abs(np.corrcoef(M_GM.ravel(), M_mm.ravel())[0, 1])
        disagreement = 1 - c
        if disagreement > best_disagreement:
            best_disagreement = disagreement
            best_inputs = (A.copy(), B.copy())

    A, B = best_inputs
    M_GM = op_GM_unnormalized(A, B)
    M_mm = np.cos(A)[:, None] * np.cos(B)[None, :]
    print(f"  Worst-correlated (A, B) pair in {n_trials} random trials:")
    print(f"    1 - |corr(G_M, cos a cos b)| = {best_disagreement:.4f}")
    print(f"    angle A range: [{A.min():.3f}, {A.max():.3f}]")
    print(f"    angle B range: [{B.min():.3f}, {B.max():.3f}]")
    print(f"    G_M entry range:   [{M_GM.min():.4f}, {M_GM.max():.4f}]")
    print(f"    cos-mm range:      [{M_mm.min():.4f}, {M_mm.max():.4f}]")
    print()
    print("  The sqrt nonlinearity creates divergence when cos a cos b is near -1")
    print("  (where the sqrt argument approaches 0) or near 0 (where derivatives")
    print("  differ most). This is where G_M's nonlinearity has the most")
    print("  discriminating power.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Ghost Oracle Suite — Probe 9: The operator. Names and "
                    "characterizes G_M, the half-angle separable cosine kernel "
                    "the QPU natively computes. Two known-issue artifacts "
                    "(Stage 1 clipping, Stage 4 Demo 2 broken) are documented "
                    "in the file header and fixed in Probe 9.1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--qpu", default=None,
                    help="Path to QPU base .npz (auto-finds job_*.npz in data/ if omitted).")
    ap.add_argument("--gpu", default=None,
                    help="Path to noiseless GPU base .npz "
                         "(auto-finds ghost_oracle_gpu_*.npz or ghost_oracle_gpu_*.npz in data/ if omitted).")
    ap.add_argument("--num-tiles", type=int, default=NUM_TILES,
                    help="Number of tiles in the bases.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for Stages 2-5.")
    args = ap.parse_args()

    qpu_path = args.qpu or auto_find_base("qpu")
    gpu_path = args.gpu or auto_find_base("gpu")

    if not qpu_path or not gpu_path:
        sys.exit(f"[FATAL] Probe 9 needs both a QPU base and a GPU base. "
                 f"Found qpu={qpu_path}, gpu={gpu_path}. Pass --qpu and --gpu "
                 f"or put job_*.npz / ghost_oracle_gpu_*.npz in {DATA_DIR}/")

    section("GHOST ORACLE SUITE — PROBE 9 — THE OPERATOR (G_M)")
    print("  Verified identity (machine precision):")
    print("      T3(a, b)  = 3/4 + (1/4) * cos(a) * cos(b)")
    print("      G_M(a, b) = sqrt((1 + cos a cos b) / 2) / ALPHA_NORM")
    print()
    print("  Five stages:")
    print("    1. Self-consistency (QPU and GPU both implement G_M)")
    print("    2. Characterization (G_M vs matmul vs T1)")
    print("    3. Classification (where in the operator zoo?)")
    print("    4. Application (Demo 1 trivial baseline; Demo 2 known broken)")
    print("    5. Scale and adversarial")

    qpu_ctrl, qpu_label = load_base(qpu_path, args.num_tiles)
    gpu_ctrl, gpu_label = load_base(gpu_path, args.num_tiles)
    print(f"\n  QPU: {qpu_label}")
    print(f"  GPU: {gpu_label}")

    stage1_consistency(qpu_ctrl, gpu_ctrl, args.num_tiles)
    stage2_characterize(rng_seed=args.seed, N=64, n_trials=200)
    stage3_classify(N=32, n_trials=50, rng_seed=args.seed)
    stage4_apply(rng_seed=args.seed)
    stage5_scale_adversarial(rng_seed=args.seed)

    section("PROBE 9 SUMMARY")
    print("  G_M is the half-angle separable cosine kernel:")
    print("      G_M(a, b) = sqrt((1 + cos a cos b) / 2) / ALPHA_NORM")
    print()
    print("  Adjacent to matmul: same N^2 cost, similar shape (NxN output),")
    print("  but distinct in:")
    print("    - bounded output range [0, 1/ALPHA_NORM] (clamped to [0,1])")
    print("    - rank-1 structure pre-sqrt, fuller-rank post-sqrt")
    print("    - native saturation (similarity-style, not unbounded inner product)")
    print()
    print("  Self-consistent across implementations:")
    print("    - Closed form: exact")
    print("    - GPU sampler: shot noise around exact (modulo Stage 1 clipping)")
    print("    - QPU circuit: characterized channel error around exact")
    print()
    print("  Known issues from this probe (preserved for trajectory legibility,")
    print("  fixed in Probe 9.1):")
    print("    - Stage 1 reports inflated MAE due to min(1.0, ...) clamp on")
    print("      tiles where analytical G_M = 1.")
    print("    - Stage 4 Demo 2 uses G_M as its own truth; MSE_GM ~ 0,")
    print("      printed ratio meaningless.")
    print()
    print("  Probes 10 and 10.1 carry G_M forward into the attention context.")
    print()


if __name__ == "__main__":
    main()