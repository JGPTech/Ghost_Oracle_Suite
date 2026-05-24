# Ghost Oracle Suite — Full Process Record

This document records the entire research and engineering trajectory of the Ghost Oracle Suite from initial probes through the final projector benchmark. It exists so that any future contributor — human or AI agent — can pick up the work with full context.

It is chronological. It includes the wrong turns. It includes the bugs we caught and the bugs we missed at first. It is not a polished narrative. It is a working record.

---

## Part 1 — Origin and the wrong target

### What started the project

A tiled Hadamard-test circuit submitted to IBM Quantum. Each tile has seven qubits arranged so a control qubit, after two Hadamards and a CSWAP, encodes a similarity measure between two angle-parameterized states. 12 tiles cover entries of a 4×4 matrix, 4096 shots each.

Supporting code: `ghost_oracle_qpu.py` (job submission), `dump_job.py` (result-to-npz), `ghost_kernel.cu` (T1 rank-K CUDA kernel), `ghost_projection.cu` (G_M bucketed projection kernel), `ghost_oracle_gpu.py` (noiseless GPU sampler), `ghost_oracle_projection.py` (legacy projection benchmark).

### The assumed target — T1

The original code was built assuming the QPU was computing the textbook Hadamard-test result:

```
T1(a, b) = |cos(a - b)|
```

This is the rank-1 cosine kernel — what cuBLAS computes via the lifted representation `[cos a, sin a] · [cos b, sin b]^T`. The custom `ghost_kernel.cu` also computes T1, hand-rolled for tiny matrices where cuBLAS dispatch is wasteful.

The first set of telemetry — Benford-statistical structure scores on the projected manifold stream — claimed to find "holographic structure" in the QPU output that aligned with T1 via an `ALPHA_NORM = 0.9127` correction.

---

## Part 2 — Probes 1 through 7: dismantling the wrong target, building the right one

### Probe 1: Identity Bridge

Tested whether the QPU's normalized output matches `|cos((a-b)/2)|` (the textbook half-angle Hadamard formula).

**Result:** It does not. GPU vs analytical: 1e-7 (fp32 noise). QPU vs analytical: MAE 0.19, max diff 0.50, with -0.18 bias. The "GPU rank-1 = QPU" identity claim fails. ALPHA looks like a global gain, but something structural is missing.

### Probe 2: Projection-Scrambled Control

Tested whether the Benford "structure" was coupled to the intended geometry or just sitting in the bitstream.

**Result:** No separation. Intended geometry scored z = -0.36 (Benford) and -0.99 (recursive) against a scrambled null. The holographic geometry-coupling claim collapses. Structure is in the bitstream, not the projection.

### Probe 3: Anchor-Conditioned Projection

Four channel-fit schemes plus wrong-channel controls, attempting to recover the projection via various conditioning approaches.

**Result:** No scheme beats blind baseline. All anchor-conditioned variants score at or below z = 0 against null. LOO R² is negative across the board — a smooth depolarization channel model doesn't fit. The channel has per-tile structure a 3-parameter linear model can't capture.

### Probe 4: Noiseless GPU Base — THE PIVOT

This is where the project changed direction. The ghost CNOTs (`v1→a1, a2` and `v2→b1, b2`) entangle the swap-test qubits with their ancillas *before* the Hadamard test, breaking the product-state assumption the original code relied on. The actual expectation is:

```
P(0) = ½(1 + cos²(a/2) cos²(b/2) + sin²(a/2) sin²(b/2))
```

Not the textbook `cos²((a-b)/2)`. Built a statevector sampler reproducing this exactly. Analytical and empirical marginals match to shot noise.

**Result:** T3 is the real target. T1 was never what this circuit computes.

### Probe 5: Unified Engine

Reran Probes 1-3 against the corrected T3 target.

**Result:** Identity Bridge MAE = 9.8e-3 (vs original 0.19). Z-scores low — the Benford "signal" was a sampling artifact, not geometry-coupled structure. The corrected target is the right one.

### Probe 6: Explicit 3-Way Convergence

Formalized three distinct targets:
- **T1**: `|cos(a-b)|` — what cuBLAS and rank-K kernel compute
- **T2**: `|cos((a-b)/2)|` — the original code's assumed target
- **T3**: the mixed-state formula from Probe 4 — what the circuit actually produces

Each backend scored against its own correct target. cuBLAS = TQN to fp32 noise. Noiseless GPU base hits T3 at shot noise. QPU's deviation from T3 is the residual signal characterized in Probes 1-5.

### Probe 7: Ghost Entanglement Parity Test

Direct physical confirmation of the Probe-4 model. If ancillas were independent, `P(a1 ≠ a2)` and `P(b1 ≠ b2)` would equal independent marginals. Under GHZ correlation from ghost CNOTs, they should be ≈ 0.

**Result on QPU:** mean `P(a1 ≠ a2) = 0.15` (null = 0.30); `P(b1 ≠ b2) = 0.17` (null = 0.36). Most tiles flagged with strong entanglement evidence. The GHZ correlation is physically present in the hardware.

---

## Part 3 — Probe 8: Characterizing the QPU residual

### Probe 8.0 — Residual Decomposition

Five-stage analysis of the QPU's deviation from T3 at full 32-bin joint distribution resolution (not just scalar P0).

**Result:**
- Stage 1: 12/12 tiles show QPU TVD > shot-noise floor + 3σ
- Stage 2: Top singular value of residual carries 58.9% — not rank-1, ghost-coupled
- Stage 3: Channel mixture fit shows fitter degeneracy (lam_ghost and lam_readout substitute for each other)
- Stage 4: Mixture likelihood beats T3 by +34k nats per tile — but per-shot only +8.4 nats, dominated by support expansion not structure capture
- Stage 5: Reduction attempt: MAE got WORSE by 32.5%. Channel model doesn't span the residual.

The 8.0 channel fit was diagnosing model misspecification, not real channel structure.

### Probe 8.1 — Split-Readout Decomposition

Split the readout into ctrl-qubit and ghost-qubit components to break the fitter degeneracy.

**Result:**
- Channel decomposition cleaner (corr(lam_g, lam_rc) = -0.290 vs -0.757 before)
- Stage 5 reduction WORKED: MAE 0.099 → 0.078 (+20.9%)
- But Stage 4 degeneracy diagnostic shows only ghost decoherence is essential; readout components are mostly degenerate

The split-readout inversion math was the actual fix from 8.0. Channel identification still imperfect, but the inversion direction was now algebraically correct.

### Probe 8.2 — Drift-First Alternating

Attempted joint fit of coherent drift (shared d_a, d_b) and incoherent channels via alternating optimization with OUT regularization on per-tile residual drifts.

**Result:** OPTIMIZER BROKEN. Three bugs compounding:
1. `d += d_new` accumulating instead of `d = d_new`
2. OUT penalty on variance(eps) not magnitude — allowed mean(eps) to walk
3. No hard bound on |d|

At n=4 iterations, d_b ran away to -3.66 rad (-210°), totally unphysical. Even at n=1, MAE corrected only +10.9% vs 8.1's +20.9%.

**Status:** Known broken. Stage 2+3 (single shared-drift fit + per-tile channel) usable; alternation loop is not. Documented in `docs/known_issues.md`.

### Probe 8.4 — Multi-Base Benford + p-Adic Telemetry

Forensic telemetry on the 8.1 and 8.2-S3 residuals: multi-base Benford (bases 2, 3, 5, 7, 10) and p-adic valuation tests on integer-scaled residual entries.

**Result:**
- Base-2 column flat (P(d=1)=1 for base-2 Benford, my implementation bug — not informative)
- Cross-base structure detected in 8.1_raw col-major residuals (genuine signal, not artifact)
- 8.2_post residuals essentially clean — drift+channel captures structure 8.1 alone missed
- p-adic ν_p tests inconclusive due to 4096-shot artifact (shot count is 2^12, leaks into divisibility tests)

The telemetry confirmed 8.2's clean path (Stage 2+3) captures more structure than 8.1 alone, even though 8.1's MAE was lower. Different metrics, different conclusions, both honest.

---

## Part 4 — Probe 9: The operator

### The reframe

After 8 stages of trying to fix the QPU residual, the question shifted: **what is the operator the QPU is actually computing?**

### Probe 9: Ghost Oracle Operator Characterization

Verified at machine precision that T3 has a much cleaner closed form:

```
T3(a, b) = ¾ + ¼ cos(a) cos(b)
G_M(a, b) = sqrt((1 + cos(a) cos(b)) / 2) / α
```

Five-stage analysis:

1. **Self-consistency** — closed form, GPU sampler at shot noise, QPU at characterized channel error
2. **Characterization** — G_M vs matmul vs T1: corr(G_M, matmul) = -0.75 on random inputs (anti-correlated — important structural fact)
3. **Classification** — G_M is NOT positive semidefinite (0/50 PSD tests passed). Not a Mercer kernel. Indefinite pairwise operator. Correlation with `cos a · cos b` outer product: 0.9992. G_M is structurally cos-outer-product plus saturating sqrt.
4. **Application** — Demo 2 was BROKEN (truth was G_M itself, gave G_M oracle MSE=0, ratio reporting divided by zero, claimed wrong winner)
5. **Scaling** — 3-5× constant overhead vs matmul, N² scaling

**Result:** G_M is a real operator with three consistent implementations and clear structural properties. But Probe 9's specific demos were largely either tautological or broken. The Stage 1 consistency claim looked terrible due to clipping artifact (3 tiles at G_M=1 by saturation).

### Probe 9.1 — Fixed Demos

- Stage 1: now reports unnormalized G_M_raw, clipping artifact resolved (GPU MAE = 0.01, below shot noise floor)
- Stage 2: saturation-regime regression — polynomial-degree-3 needs 10 parameters to hit 6e-7 MSE on a function G_M's structural form captures in zero
- Stage 3: indefinite-kernel SVM — G_M loses cleanly to RBF (35% vs 98%). Indefinite-kernel angle dead.
- Stage 4: G_M attention with scalar phase-lift — loses on representation tasks. Attention angle dead unless data is already angle-encoded.

**Result:** The structural-form advantage is real for data with saturating cosine-similarity structure. Indefinite-kernel angle dead. Attention angle requires per-dim aggregation (probe 10).

---

## Part 5 — Probe 10: Attention as the killer app

### Probe 10 — Ghost Attention Benchmark

First attempt to demonstrate G_M as a native attention primitive under outlier attack.

**Result:** Didn't differentiate from cuBLAS. Three setup bugs:
1. L2 renormalization after spike injection neutralized the attack
2. Outliers on different dims per key — incoherent attack, not the real LLM scenario
3. d=64 with renormalized embeddings made softmax already-uniform — bottleneck mechanism never engaged

Both methods plateaued at 1 - outlier_fraction = 0.95. Not robustness, just the geometric maximum.

### Probe 10.1 — Real Softmax Bottleneck

Fixed all three:
1. No L2 renormalization
2. Same-dim coherent attack (all outliers spike on same k_bad)
3. Smaller d (16) where softmax actually concentrates
4. Per-dim G_M aggregation
5. tanh-based phase-lift for unbounded inputs

**Result:** DP top-1 collapses from 74% to 43% under same-dim attack. G_M tied: 84% to 84%. cuBLAS attention mass on outliers: 0.42 (huge). G_M outlier/non-outlier score ratio: 0.999 (didn't notice).

The headline finding. Per-dim G_M aggregation under same-dim coherent attack: structurally robust where dot-product attention catastrophically collapses.

---

## Part 6 — The benchmark trajectory

Five iterations before the headline:

### `final_benchmark.py`

Initial attempt. Four backends (cuBLAS T1, ghost_kernel T1, GPU projection, QPU projection). Default 4×4 mode, three sweep regimes (small/attention/extreme). Operator-only timing, no attention pipeline.

Problem: still framed projection as estimating T3 separately from "G_M backend." Conceptually muddled.

### `final_benchmark_combined.py`

Three honesty fixes:
1. Score projection outputs directly (not via parallel cuBLAS T1 call)
2. Real OOM handling (no hardcoded 12GB threshold)
3. MAE verification at small sizes against analytical G_M

Result: MAE = 0.375 (bad). The single-tile-at-scale estimator throws away too much info. Top-1 dropped from 94% to whatever the lone-projection signal supported.

This is where you flagged that we'd "thrown away the good stuff" — the parallel cosine evaluation in the earlier code was actually a **ghost channel** providing the accuracy signal, not a bug to remove.

### `final_benchmark_tied.py`

The tied-channel design. Two channels in one streaming kernel:
- **Projection channel** — physical bucket reweighting (certifies operator)
- **Geometry channel** — analytical closed form (provides sharp argmax)
- Agreement = mean |proj - geom| (the ghost-channel certificate)

Result on 1024² correctness check:
- GPU agreement: 0.0088 (below shot noise floor)
- QPU agreement: 0.1993 (within probe 7-8 characterized range)
- Geometry kernel vs numpy: 3e-8 (fp32 precision)

The architecture worked. But the scalar phase-lift `mean(X) * sqrt(d_k)` collapsed embeddings to a single number, destroying retrieval information. Top-1 at 0% on extreme sweep.

### `final_benchmark_tied_perdim.py`

Restored per-dim aggregation (Probe 10.1 architecture) inside the tied streaming kernel. Each output entry is mean over d of per-dim G_M from both channels.

Result: 100% top-1 across all shapes. Suspiciously perfect. Diagnosis: `X_K = X_Q.copy()` made it a self-match detection task, not a real retrieval task. 100% was the right answer for a trivial problem.

### `final_benchmark_tied_perdim_v2.py` — THE HEADLINE

Added:
1. **Jitter** — query = key + 0.3·gaussian (Probe 10.1's actual task)
2. **Ops budget** — three-axis comparison: speed, accuracy, ops-per-correct-retrieval

Result on extreme sweep (full_info mode):

| Shape | cuBLAS top-1 | G_M tied top-1 | cuBLAS VRAM | G_M VRAM | cuBLAS ops/correct | G_M ops/correct |
|---|---|---|---|---|---|---|
| 4096² | 79.03% | 100.00% | 0.06 GB | 0.002 GB | 6.6e5 | 1.3e7 |
| 16384² | 76.28% | 100.00% | 1.01 GB | 0.008 GB | 2.7e6 | 5.2e7 |
| 65536² | 73.92% | 100.00% | 16.03 GB | 0.032 GB | 1.1e7 | 2.1e8 |

Geometry kernel correctness: 2.3e-5 MAE vs numpy reference. GPU agreement: 0.0136 across all shapes (constant, as expected — agreement is per-(a,b) noise). QPU agreement: 0.1037 (probe 7-8 territory).

The tradeoff is honest: G_M tied is ~20× more ops-per-correct-retrieval, but it gets 100% under attack vs cuBLAS's 75-79%, uses 500× less VRAM at extreme, and scales to regimes where cuBLAS approaches OOM.

---

## Part 7 — What the benchmark actually establishes

Four cleanly defensible claims, each backed by code and data:

**1. G_M is a well-defined operator with three consistent implementations.**

`G_M(a, b) = sqrt((1 + cos a cos b)/2)/α`, verified analytically (Probe 9), implemented by the noiseless GPU sampler at shot noise (~0.01 MAE), and by the physical QPU circuit at characterized channel error (~0.10 MAE, in the range probes 7-8 measured). The tied-channel agreement metric reproduces this at scale.

**2. Per-dim aggregated G_M is structurally robust to coherent same-dim outlier attacks.**

100% retrieval at jitter 0.3 with 5% same-dim attack at magnitude 50, across shapes from 4096² to 65536². cuBLAS dot-product attention on the same task degrades from 79% to 74% as N grows. This holds because per-dim averaging neutralizes single-dim spikes that softmax catastrophically amplifies.

**3. Streaming with fused argmax gives O(N) memory scaling.**

500× less VRAM than cuBLAS at 65536². cuBLAS approaches OOM at 131072² where its score matrix would be 64 GB; the streaming kernel uses only the embedding inputs plus N output cells.

**4. The compute tradeoff is real and reported honestly.**

~20× more ops-per-correct-retrieval. The robustness and memory savings come at a real compute cost. Whether that's worth it depends on the application — for LLM attention where one bad head cascades, probably yes; for low-stakes ranking where 80% is fine, probably no.

---

## Part 8 — Repository structure

Final layout for the CC0 community release:

```
ghost-oracle-suite/
├── README.md
├── LICENSE                            # CC0 1.0 Universal
├── CONTRIBUTING.md                    # break-it-fix-it philosophy
├── requirements.txt
├── .gitignore
│
├── ghost_oracle/                      # the library
│   ├── __init__.py
│   ├── projector_benchmark.py         # headline tied-perdim-v2 benchmark
│   ├── qpu.py                         # QPU job submission
│   ├── gpu.py                         # noiseless GPU sampler
│   ├── dump.py                        # QPU result -> npz
│   └── kernels/
│       └── ghost_kernel.cu            # CONSOLIDATED: all CUDA kernels in one file
│                                       # (projection_eval, geometry_channel,
│                                       # tied_streaming_perdim, tied_materialize_perdim,
│                                       # projection_4x4_legacy, ghost_t1_batch)
│
├── probes/                            # forensic trajectory
│   ├── README.md                      # narrative arc
│   ├── probe1_identity_bridge.txt
│   ├── probe2_projection_scrambled_control.txt
│   ├── probe3_anchor_conditioned_projection.txt
│   ├── probe4_build_base.txt
│   ├── probe5_unified_engine.txt
│   ├── probe6_3way_convergence.txt
│   ├── probe7_ghost_parity.txt
│   ├── probe8_residual_decomposition.py
│   ├── probe8_1_split_readout.py
│   ├── probe8_2_drift_alternating.py     # marked KNOWN ISSUE
│   ├── probe8_4_padic_benford.py
│   ├── probe9_ghost_operator.py
│   ├── probe9_1_indef_kernel_attn.py
│   ├── probe10_ghost_attention.py        # marked SUPERSEDED
│   ├── probe10_1_real_softmax_attack.py
│   └── benchmark_evolution/
│       ├── final_benchmark.py
│       ├── final_benchmark_combined.py
│       ├── final_benchmark_tied.py
│       └── final_benchmark_tied_perdim.py
│
├── data/
│   ├── README.md
│   └── (sample npz files)
│
├── docs/
│   ├── math.md                        # T1, T3, G_M derivations
│   ├── architecture.md                # tied-channel design
│   └── known_issues.md                # running list
│
└── examples/
    └── minimal_usage.py
```

---

## Part 9 — Known issues, carried forward

### Probe 8.2 alternation loop is broken

Three compounding bugs in the alternating drift-channel optimizer:
1. `d_a += d_new` accumulates instead of replacing
2. OUT regularization penalizes variance not magnitude
3. No bound on |d_a|, |d_b|

Stage 2+3 of probe 8.2 is usable (single shared-drift fit + per-tile channel). The alternation loop should not be invoked. Marked in the probe file header. Fix would require joint optimization with hard bounds and L2 anchors instead of alternation.

### Probe 9 Demo 2 broken

Truth function was G_M itself, so G_M oracle got MSE=0, ratios divided by zero, reporting claimed G_M won when it didn't. Fixed in Probe 9.1 by using saturation-regime inputs where polynomial models need more parameters to capture both curvature and ceiling.

### Probe 10 superseded by 10.1

Three setup bugs (L2 renorm, random-dim attack, too-large d) made the test fail to differentiate. Kept in repo for trajectory legibility, marked SUPERSEDED in header.

### Probe 8.4 base-2 Benford column is non-informative

Base-2 Benford has only one valid leading digit (d=1, P=1), so the test always returns the same value. Not a bug in the implementation, but a mathematical degenracy of leading-digit analysis in base 2. The base-3, 5, 7, 10 columns and ν_p valuation tests are informative.

### Probe 8.4 ν_p valuation tests confounded by shot count

4096 = 2^12 means any integer-scaled residual inherits 2-adic structure regardless of physics. ν_2 chi² results are not interpretable without controlling for shot count. ν_3 and ν_5 tests are more reliable but still affected by ALPHA_NORM = 0.9127 leaking through fp32 rounding.

### Phase-lift design is informal

`θ = (π/2)(1 + tanh(x/3))` was chosen as "simplest saturating map." Other choices would change the operator's effective input distribution. The benchmark currently uses this fixed map; a proper analysis of how phase-lift choice affects retrieval accuracy is open work.

---

## Part 10 — Open questions for the next session

These are the things I would work on next, ordered roughly by impact:

1. **Sweep jitter scale.** At jitter=0.3 we hit 100% — the benchmark isn't actually hard enough to differentiate at d=64. Try 0.5, 0.8, 1.0. Find where G_M starts to fail. If it holds at 95%+ up to jitter=1.0, that's the strongest possible demonstration. If it cracks at 0.5, that's also informative.

2. **Clean-data baseline.** Run no-attack version on every shape. If cuBLAS gets 99% clean and 75% attacked, while G_M gets 95% clean and 100% attacked, that's a more nuanced (and accurate) picture than just "G_M is robust."

3. **Real LLM embeddings.** Pull keys and queries from an open pretrained transformer (any layer's attention K/Q after the QKV projection). Run the benchmark on actual learned representations. If the synthetic result holds on real data, the attention claim is fully grounded.

4. **Probe 8.2 joint-fit replacement.** Write a properly bounded joint optimizer for drift + channels instead of the broken alternation. This would close the loop on probes 8.0-8.2 and give the cleanest possible channel decomposition of the QPU residual.

5. **Probe 11 candidate — value aggregation.** We've shown G_M as a similarity function under attack. The next step is the full `softmax(S) V` pipeline replaced by `normalize(G_M) V` and seeing if downstream loss holds end-to-end.

6. **QPU implementability crossover.** At what (N, shot budget, error tolerance) does the QPU projection beat the GPU projection? Currently the GPU sampler is faster *and* more accurate per shot. The QPU's win has to be at scale or specialized hardware — when?

---

## Part 11 — Philosophy and license

The Ghost Oracle Suite is CC0. No attribution required. No restrictions. The intent is: build, break, fix, document, repeat — all in the open.

The "break-it-fix-it" rule: if you find something wrong, you provide the fix alongside the bug report. Not as a gatekeeper, but as a norm — fixes-with-bugs travel through the project faster than bugs alone.

This document is part of that. The bugs in probes 8.2, 9, 10 are documented because the *process* of finding and fixing them is the research. A future contributor (human or agent) who reads this should know exactly where we landed, what we tried, what worked, and what didn't.

---

## Closing

The QPU isn't a noisy matrix multiplier. It's a native implementation of a different operator — G_M — that has built-in bounded saturation, structural outlier resistance, and three consistent implementations across analytical, classical, and quantum hardware. The probes prove the operator exists and characterize its physical implementation. The projector benchmark proves it runs.

Everything in this repo is the working out of that single result.
