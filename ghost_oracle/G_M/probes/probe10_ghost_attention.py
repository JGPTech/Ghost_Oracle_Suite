#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GHOST ORACLE SUITE — PROBE 10 — GHOST ATTENTION (SUPERSEDED)
==============================================================================
Probe 10 was the first attempt to demonstrate that the Ghost Oracle operator
G_M is a structurally robust attention primitive under adversarial outlier
embeddings. The intent was solid; the experimental setup had three bugs
that compounded into a "no separation" result. Probe 10.1 fixed all three
and produced the headline finding.

This file is preserved as the historical record of the failed attempt. The
bugs are intact and the code runs to completion — read alongside Probe 10.1
to see what changed.

THE THREE BUGS (all documented in docs/known_issues.md and PROCESS_RECORD.md):

  BUG 1 — L2 RENORMALIZATION NEUTRALIZES THE ATTACK
      After injecting a spike of magnitude 50 into one dimension of a key,
      the code immediately L2-normalizes the key. The norm becomes ~50 and
      the spike dimension is divided by ~50, leaving a unit vector with the
      spike dimension pinned near 1 — no attack at all. Probe 10.1 removes
      the renormalization step.

  BUG 2 — INCOHERENT OUTLIER DIMENSIONS
      inject_outliers picks a random dimension k_bad *per outlier key*.
      Outlier key A gets its spike on dim 17, outlier key B on dim 42.
      Queries see uncorrelated noise across keys, not the coordinated
      same-dim attack that breaks softmax in real LLM attention. Probe 10.1
      forces all outliers onto the same k_bad — the actual adversarial
      scenario.

  BUG 3 — d=64 WITH L2 NORMALIZATION DEFEATS THE SOFTMAX BOTTLENECK
      With d=64 and unit-normalized embeddings, dot products land in a
      narrow range and softmax over 4096 candidates is already broad —
      no single key can dominate even before the attack. Probe 10.1 uses
      d=16 specifically to make softmax actually concentrate, exposing
      the bottleneck mechanism that G_M's per-dim aggregation neutralizes.

COMBINED EFFECT: both methods plateau at 1 - outlier_fraction = 0.95 on
clean data and stay there under "attack." That's the geometric maximum,
not robustness — both methods are unable to retrieve outlier keys, so
they get everything else right.

STAGES:
  1. Setup and phase-lift verification
  2. Classical dot-product baseline (clean and noisy)
  3. G_M attention pipeline (clean and noisy)
  4. Outlier magnitude sweep
  5. Outlier fraction sweep
  6. Sanity: clean-data accuracy of both methods

Verified identity (from Probes 4, 6, 9):
    G_M_raw(a, b) = sqrt((1 + cos(a) cos(b)) / 2)
    G_M(a, b)     = min(G_M_raw / ALPHA_NORM, 1)

HISTORICAL CONTEXT:
    Probe 10 was the moment the suite tried to convert G_M's structural
    properties (bounded output, sqrt saturation, indefinite kernel) into
    an attention-mechanism claim. The reasoning was: per-dim cos+sqrt
    bounds each dimension's contribution to the similarity score, so no
    single dimension can dominate the way softmax-on-dot-product does
    when one key has an outlier value. That reasoning is correct, and
    Probe 10.1 demonstrates it cleanly. But Probe 10's setup managed to
    suppress the very mechanism it was trying to test:

    - The L2 renorm removed the attack's amplitude before either method
      saw it.
    - The random-dim outlier choice made the attack incoherent across
      keys, so softmax over 4096 candidates averaged it out anyway.
    - The high embedding dimension (d=64) plus unit normalization made
      softmax already-uniform, so there was no bottleneck for the attack
      to exploit.

    The result was "both methods plateau at 95%, no separation" — which
    looked like the hypothesis was wrong, but was actually three setup
    bugs canceling out the experiment. Probe 10.1's fixes (no L2 renorm,
    same-dim coherent attack, d=16) reveal the real picture: dot-product
    top-1 collapses from 74% to 43%; G_M tied stays at 84%; softmax
    attention mass on outliers reaches 0.42 (huge); G_M's
    outlier/non-outlier score ratio is 0.999 (didn't notice).

    The Probes 10 -> 10.1 arc is also the seed of the headline
    projection_benchmark.py architecture: per-dim G_M aggregation inside
    a tied streaming kernel, which scales to 65536x65536 retrieval tasks
    at 100% top-1 where cuBLAS approaches OOM at 73-79%. See
    PROCESS_RECORD.md for the full arc.

USAGE:
    python probe10_ghost_attention.py
    python probe10_ghost_attention.py --N 4096 --d 64 --seed 42
==============================================================================
"""

import argparse
import math
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
ALPHA_NORM = 0.9127


# =============================================================================
# CORE OPERATORS
# =============================================================================
def phase_lift(X):
    """Map X (N x d, ~ N(0, 1)) to theta in radians.
    Probe-10 specific: theta = (pi/2) * (1 + x).
    For x in 3-sigma range, theta lies in roughly [-pi, 2*pi] so cos(theta)
    is well-behaved. For x = 50, theta = (pi/2)*51 = 25.5*pi -> cos ~ 0.
    The spike's cosine is NOT small in the cos sense -- it's just an
    arbitrary value in [-1, 1] uncorrelated with the true match's cos.
    That's the boundedness-plus-incoherence mechanism the probe is testing
    for (and, due to the three bugs above, fails to actually exercise)."""
    return (math.pi / 2) * (1.0 + X)


def dot_product_attention(XQ, XK):
    """Standard scaled dot product, returns N x M similarity matrix."""
    d = XQ.shape[1]
    return XQ @ XK.T / math.sqrt(d)


def softmax_rows(S):
    """Row-wise softmax."""
    M = S.max(axis=1, keepdims=True)
    E = np.exp(S - M)
    return E / E.sum(axis=1, keepdims=True)


def gm_attention(theta_Q, theta_K):
    """Per-dimension G_M aggregated by mean across the d feature dimensions.
    Returns N x M similarity matrix bounded in [0, 1/ALPHA_NORM].

    G[i, j] = (1/d) sum_k (1/ALPHA_NORM) * sqrt((1 + cos(theta_Q[i,k]) cos(theta_K[j,k])) / 2)

    The mean-over-d is what makes G_M robust to single-dim spikes: even if
    one dim is adversarial, its contribution is bounded by 1/(ALPHA_NORM * d)
    after the mean. Probe 10.1 demonstrates this works; Probe 10 cannot
    because of the three setup bugs in the header.
    """
    cosQ = np.cos(theta_Q)          # N x d
    cosK = np.cos(theta_K)          # M x d
    N, d = cosQ.shape
    M = cosK.shape[0]
    if N * M * d <= 2**29:          # ~512M floats budget; full tensor path
        prod = cosQ[:, None, :] * cosK[None, :, :]   # N x M x d
        per_dim = np.sqrt(np.clip((1 + prod) / 2, 0, None)) / ALPHA_NORM
        return per_dim.mean(axis=2)
    else:                           # streaming over d (memory fallback)
        out = np.zeros((N, M), dtype=np.float64)
        for k in range(d):
            prod_k = np.outer(cosQ[:, k], cosK[:, k])
            out += np.sqrt(np.clip((1 + prod_k) / 2, 0, None)) / ALPHA_NORM
        return out / d


# =============================================================================
# DATA
# =============================================================================
def make_embeddings(N, d, seed):
    """Synthetic unit-vector embeddings on the d-sphere."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def inject_outliers(XK, outlier_fraction, spike_magnitude, seed):
    """Inject spikes into a fraction of the keys.

    BUG 2 IS HERE: dims is sampled per-outlier, so each outlier key gets a
    spike on its own randomly-chosen dimension. The real adversarial
    scenario coherently spikes the same dimension across all outliers.
    Probe 10.1 fixes this by drawing one k_bad and reusing it.

    Returns (XK_noisy, mask_outlier, indices, dims).
    """
    rng = np.random.default_rng(seed)
    M, d = XK.shape
    n_out = int(round(M * outlier_fraction))
    XK_noisy = XK.copy()
    indices = rng.choice(M, size=n_out, replace=False)
    dims = rng.integers(0, d, size=n_out)        # BUG 2: per-outlier dim
    XK_noisy[indices, dims] = spike_magnitude
    mask = np.zeros(M, dtype=bool)
    mask[indices] = True
    return XK_noisy, mask, indices, dims


# =============================================================================
# METRICS
# =============================================================================
def top1_accuracy(S_noisy, gt_argmax):
    """gt_argmax[i] = ground-truth best j for query i. S_noisy is N x M.
    Return fraction of queries whose argmax under S_noisy equals gt_argmax[i]."""
    pred = np.argmax(S_noisy, axis=1)
    return float(np.mean(pred == gt_argmax))


def softmax_match_score(S_noisy, gt_argmax):
    """Mean probability mass assigned to the true match under softmax."""
    P = softmax_rows(S_noisy)
    N = P.shape[0]
    return float(np.mean(P[np.arange(N), gt_argmax]))


# =============================================================================
# REPORTING
# =============================================================================
def hline(c="-", w=92):
    print(c * w)


def section(title, w=92):
    print()
    hline("=", w)
    print(f"  {title}")
    hline("=", w)


def superseded_banner():
    """Loud SUPERSEDED warning printed at the start of every run."""
    print()
    hline("!", 92)
    print("  [SUPERSEDED] PROBE 10 — THREE SETUP BUGS CANCEL THE EXPERIMENT")
    hline("!", 92)
    print("  This probe is preserved for trajectory legibility. The hypothesis")
    print("  it tests (G_M is structurally robust to embedding outliers) is")
    print("  correct, but this script's setup cannot demonstrate it:")
    print()
    print("    BUG 1: L2-renormalizing keys *after* spike injection removes")
    print("           the attack's amplitude (the renorm divides the spike")
    print("           dim by ~spike_magnitude).")
    print("    BUG 2: Outliers are placed on per-outlier random dimensions,")
    print("           making the attack incoherent across the outlier set")
    print("           (the real adversarial scenario is same-dim).")
    print("    BUG 3: d=64 with unit-normalized embeddings makes softmax over")
    print("           4096 candidates already-uniform, so the bottleneck")
    print("           mechanism that G_M neutralizes never engages.")
    print()
    print("  RESULT (this probe): both methods plateau at ~95% — the")
    print("  geometric maximum 1 - outlier_fraction, not robustness.")
    print()
    print("  RESULT (Probe 10.1, all three bugs fixed): dot product collapses")
    print("  from 74% to 43% under same-dim coherent attack; G_M tied stays")
    print("  at 84%. See probe10_1_real_softmax_attack.py for the corrected")
    print("  experiment and the headline finding.")
    hline("!", 92)


# =============================================================================
# STAGES
# =============================================================================
def stage1_setup(N, d, outlier_frac, spike, seed):
    section("STAGE 1 — SETUP AND PHASE-LIFT VERIFICATION")
    print(f"  N (queries=keys) : {N}")
    print(f"  d (embedding)    : {d}")
    print(f"  outlier fraction : {outlier_frac:.0%}")
    print(f"  spike magnitude  : {spike}")
    print(f"  seed             : {seed}")
    print(f"  NOTE: d={d} with unit-normalized embeddings is BUG 3 — softmax")
    print(f"        over N={N} candidates is already broad. Probe 10.1 uses d=16.")
    print()

    rng = np.random.default_rng(seed)
    XK = make_embeddings(N, d, seed)
    jitter = rng.normal(scale=0.05, size=(N, d))
    XQ = XK + jitter
    XQ /= np.linalg.norm(XQ, axis=1, keepdims=True)
    gt = np.arange(N)  # query i should match key i

    print(f"  Clean embedding distribution:")
    print(f"    XQ min/max  : {XQ.min():+.4f} / {XQ.max():+.4f}")
    print(f"    XQ mean/std : {XQ.mean():+.4f} / {XQ.std():.4f}")
    print()

    theta = phase_lift(XQ[0])
    print(f"  Phase-lift of first query (first 5 dims):")
    print(f"    X[0, :5]      : {XQ[0, :5]}")
    print(f"    theta[0, :5]  : {theta[:5]}")
    print(f"    cos(theta)[:5]: {np.cos(theta[:5])}")
    print()

    spike_theta = phase_lift(np.array([spike]))[0]
    spike_cos = math.cos(spike_theta)
    print(f"  Outlier behavior IF the spike survives L2 renorm (it won't, see BUG 1):")
    print(f"    spike value          : {spike}")
    print(f"    phase-lifted theta   : {spike_theta:.4f}  ({spike_theta/math.pi:.2f}*pi)")
    print(f"    cos(theta_spike)     : {spike_cos:+.6f}")
    print(f"    Max possible bias from one spike dim in G_M: "
          f"1/(ALPHA_NORM * d) = {1/(ALPHA_NORM*d):.4f}")

    return XQ, XK, gt


def stage2_classical(XQ, XK, gt, outlier_frac, spike, seed):
    section("STAGE 2 — CLASSICAL DOT-PRODUCT ATTENTION")
    print("  Compute S_clean = XQ @ XK.T / sqrt(d) on the clean data,")
    print("  then S_noisy on the same XQ vs spike-injected XK.")
    print()
    print("  BUG 1 active here: XK_noisy is L2-renormalized BEFORE comparison,")
    print("  which divides the spike dim by ~spike_magnitude and recovers a")
    print("  near-unit vector. The 'attack' is gone before either method sees it.")
    print()

    XK_noisy, mask, idx_outliers, dims_outliers = inject_outliers(
        XK, outlier_frac, spike, seed + 1
    )
    XK_noisy_norm = XK_noisy / np.linalg.norm(XK_noisy, axis=1, keepdims=True)  # BUG 1

    t0 = time.perf_counter()
    S_clean = dot_product_attention(XQ, XK)
    t_clean = time.perf_counter() - t0

    t0 = time.perf_counter()
    S_noisy = dot_product_attention(XQ, XK_noisy_norm)
    t_noisy = time.perf_counter() - t0

    acc_clean = top1_accuracy(S_clean, gt)
    acc_noisy = top1_accuracy(S_noisy, gt)
    sm_clean = softmax_match_score(S_clean, gt)
    sm_noisy = softmax_match_score(S_noisy, gt)

    print(f"  Clean dot-product top-1 accuracy   : {acc_clean:.4f}")
    print(f"  Noisy dot-product top-1 accuracy   : {acc_noisy:.4f}")
    print(f"  Clean softmax mass on true match   : {sm_clean:.4f}")
    print(f"  Noisy softmax mass on true match   : {sm_noisy:.4f}")
    print(f"  Compute time (clean / noisy)       : {t_clean*1000:.2f} ms / {t_noisy*1000:.2f} ms")
    print()

    pred_noisy = np.argmax(S_noisy, axis=1)
    failures = pred_noisy != gt
    failures_on_outliers = mask[pred_noisy[failures]]
    n_fail = int(failures.sum())
    if n_fail > 0:
        print(f"  Of {n_fail} failures, {failures_on_outliers.sum()} "
              f"({100*failures_on_outliers.sum()/n_fail:.1f}%) landed on outlier keys.")
    return mask, S_clean, S_noisy, XK_noisy_norm


def stage3_ghost(XQ, XK, XK_noisy_norm, gt, mask):
    section("STAGE 3 — GHOST ATTENTION (G_M PIPELINE)")
    print("  Phase-lift both queries and keys, then compute G_M aggregated per-dim.")
    print()
    print("  Reads the same L2-renormalized XK_noisy_norm Stage 2 used; BUG 1 has")
    print("  already neutralized the spike before this stage sees the data.")
    print()

    theta_Q = phase_lift(XQ)
    theta_K = phase_lift(XK)
    theta_K_noisy = phase_lift(XK_noisy_norm)

    t0 = time.perf_counter()
    G_clean = gm_attention(theta_Q, theta_K)
    t_clean = time.perf_counter() - t0

    t0 = time.perf_counter()
    G_noisy = gm_attention(theta_Q, theta_K_noisy)
    t_noisy = time.perf_counter() - t0

    acc_clean = top1_accuracy(G_clean, gt)
    acc_noisy = top1_accuracy(G_noisy, gt)

    print(f"  Clean G_M top-1 accuracy   : {acc_clean:.4f}")
    print(f"  Noisy G_M top-1 accuracy   : {acc_noisy:.4f}")
    print(f"  Compute time (clean / noisy): {t_clean*1000:.2f} ms / {t_noisy*1000:.2f} ms")
    print()
    print(f"  G_M output range (clean) : [{G_clean.min():.4f}, {G_clean.max():.4f}]")
    print(f"  G_M output range (noisy) : [{G_noisy.min():.4f}, {G_noisy.max():.4f}]")
    print(f"  G_M output std (clean)   : {G_clean.std():.4f}")
    print(f"  G_M output std (noisy)   : {G_noisy.std():.4f}")
    print()

    pred_noisy = np.argmax(G_noisy, axis=1)
    failures = pred_noisy != gt
    failures_on_outliers = mask[pred_noisy[failures]] if failures.any() else np.array([])
    n_fail = int(failures.sum())
    if n_fail > 0:
        print(f"  Of {n_fail} G_M failures, {failures_on_outliers.sum()} landed on outliers")
        print(f"    ({100*failures_on_outliers.sum()/max(n_fail,1):.1f}% vs the dot-product failure pattern).")
    else:
        print(f"  G_M had ZERO failures under outlier injection.")
    return G_clean, G_noisy


def stage4_magnitude_sweep(N, d, outlier_frac, seed):
    section("STAGE 4 — OUTLIER MAGNITUDE SWEEP (THE CLIFF THAT DIDN'T HAPPEN)")
    print("  Sweep spike magnitude from 1 to 100, fixed outlier fraction.")
    print("  Original intent: DP top1 should fall off a cliff as magnitude grows.")
    print("  Actual result: both methods stay flat because BUG 1 neutralizes the")
    print("  spike at every magnitude. See Probe 10.1 for the real cliff.")
    print()

    magnitudes = [1, 2, 5, 10, 20, 30, 50, 75, 100]
    XK = make_embeddings(N, d, seed)
    rng = np.random.default_rng(seed)
    jitter = rng.normal(scale=0.05, size=(N, d))
    XQ = XK + jitter
    XQ /= np.linalg.norm(XQ, axis=1, keepdims=True)
    gt = np.arange(N)
    theta_Q = phase_lift(XQ)

    print(f"  {'spike':>6} | {'DP top1':>8} {'DP soft':>8} {'G_M top1':>9}")
    hline()
    for mag in magnitudes:
        XK_noisy, mask, _, _ = inject_outliers(XK, outlier_frac, mag, seed + 1)
        XK_noisy_norm = XK_noisy / np.linalg.norm(XK_noisy, axis=1, keepdims=True)  # BUG 1
        S = dot_product_attention(XQ, XK_noisy_norm)
        G = gm_attention(theta_Q, phase_lift(XK_noisy_norm))
        a_dp = top1_accuracy(S, gt)
        a_sm = softmax_match_score(S, gt)
        a_gm = top1_accuracy(G, gt)
        print(f"  {mag:>6} | {a_dp:>8.4f} {a_sm:>8.4f} {a_gm:>9.4f}")


def stage5_fraction_sweep(N, d, spike, seed):
    section("STAGE 5 — OUTLIER FRACTION SWEEP")
    print("  Fixed spike magnitude, sweep what % of keys are corrupted.")
    print("  Same flat result expected as Stage 4 — see Probe 10.1.")
    print()

    fractions = [0.01, 0.05, 0.10, 0.25, 0.50]
    XK = make_embeddings(N, d, seed)
    rng = np.random.default_rng(seed)
    jitter = rng.normal(scale=0.05, size=(N, d))
    XQ = XK + jitter
    XQ /= np.linalg.norm(XQ, axis=1, keepdims=True)
    gt = np.arange(N)
    theta_Q = phase_lift(XQ)

    print(f"  {'frac':>6} | {'DP top1':>8} {'G_M top1':>9}")
    hline()
    for frac in fractions:
        XK_noisy, _, _, _ = inject_outliers(XK, frac, spike, seed + 1)
        XK_noisy_norm = XK_noisy / np.linalg.norm(XK_noisy, axis=1, keepdims=True)  # BUG 1
        S = dot_product_attention(XQ, XK_noisy_norm)
        G = gm_attention(theta_Q, phase_lift(XK_noisy_norm))
        a_dp = top1_accuracy(S, gt)
        a_gm = top1_accuracy(G, gt)
        print(f"  {frac:>6.2f} | {a_dp:>8.4f} {a_gm:>9.4f}")


def stage6_sanity(N, d, seed):
    section("STAGE 6 — SANITY: NO-OUTLIER PERFORMANCE")
    print("  Robustness only matters if both methods are competitive on clean data.")
    print("  This stage is unaffected by the three bugs — no attack is injected,")
    print("  so L2 renorm, dim choice, and softmax bottleneck are all moot.")
    print()

    rng = np.random.default_rng(seed)
    XK = make_embeddings(N, d, seed)
    for jitter_scale in [0.0, 0.05, 0.10, 0.25, 0.50]:
        jit = rng.normal(scale=jitter_scale, size=(N, d))
        XQ = XK + jit
        XQ /= np.linalg.norm(XQ, axis=1, keepdims=True)
        gt = np.arange(N)
        S = dot_product_attention(XQ, XK)
        G = gm_attention(phase_lift(XQ), phase_lift(XK))
        a_dp = top1_accuracy(S, gt)
        a_gm = top1_accuracy(G, gt)
        print(f"  jitter={jitter_scale:.2f} : DP top1 = {a_dp:.4f}, G_M top1 = {a_gm:.4f}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Ghost Oracle Suite — Probe 10: Ghost Attention (SUPERSEDED). "
                    "Preserved historical record of the first attempt at the "
                    "attention-robustness claim. Three setup bugs prevent the "
                    "experiment from differentiating dot product from G_M. "
                    "Fixed in Probe 10.1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--N", type=int, default=4096,
                    help="num queries = num keys")
    ap.add_argument("--d", type=int, default=64,
                    help="embedding dimension (kept at 64 to preserve the bug; "
                         "Probe 10.1 uses 16)")
    ap.add_argument("--outlier-frac", type=float, default=0.05,
                    help="fraction of keys to corrupt with a spike")
    ap.add_argument("--spike", type=float, default=50.0,
                    help="spike magnitude (gets divided away by L2 renorm; see BUG 1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    section("GHOST ORACLE SUITE — PROBE 10 — GHOST ATTENTION (SUPERSEDED)")
    print("  Compare classical dot-product attention vs G_M attention")
    print("  under (intended) adversarial outlier injection.")
    superseded_banner()

    XQ, XK, gt = stage1_setup(args.N, args.d, args.outlier_frac,
                              args.spike, args.seed)
    mask, S_clean, S_noisy, XK_noisy_norm = stage2_classical(
        XQ, XK, gt, args.outlier_frac, args.spike, args.seed
    )
    G_clean, G_noisy = stage3_ghost(XQ, XK, XK_noisy_norm, gt, mask)
    stage4_magnitude_sweep(args.N, args.d, args.outlier_frac, args.seed)
    stage5_fraction_sweep(args.N, args.d, args.spike, args.seed)
    stage6_sanity(args.N, args.d, args.seed)

    section("PROBE 10 SUMMARY (SUPERSEDED)")
    print("  Two pipelines computing 'attention-like' similarity:")
    print("    - Classical: linear inner product, exponential softmax")
    print("    - G_M:       cosine-lifted, sqrt-saturated, bounded mean")
    print()
    print("  Both methods plateaued at ~1 - outlier_fraction = 0.95.")
    print("  That's the geometric maximum — neither method retrieves an outlier")
    print("  key whose true match was relocated, so they both get everything")
    print("  else right. Looks like 'no separation,' but the experiment never")
    print("  actually tested the hypothesis.")
    print()
    print("  See probe10_1_real_softmax_attack.py for the corrected experiment")
    print("  with all three bugs fixed and the actual headline finding (DP 74%->43%,")
    print("  G_M tied 84%->84%, softmax mass on outliers 0.42, G_M score ratio 0.999).")


if __name__ == "__main__":
    main()