# Ghost Oracle Suite — Full Process Record

This document records the entire research and engineering trajectory of the Ghost Oracle Suite from initial probes through the final five-way benchmark. It exists so that any future contributor — human or AI agent — can pick up the work with full context.

It is chronological. It includes the wrong turns. It includes the bugs we caught and the bugs we missed at first. It includes the framings we adopted, used for months, and then had to retract when the data didn't support them. It is not a polished narrative. It is a working record.

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

The headline finding at this point in the trajectory. Per-dim G_M aggregation under same-dim coherent attack: structurally robust where dot-product attention catastrophically collapses.

---

## Part 6 — The benchmark trajectory (pre-revision)

Five iterations before what was then thought to be the headline:

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
- **Projection channel** — physical bucket reweighting (then-described as: certifies operator)
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

### `final_benchmark_tied_perdim_v2.py` — THE ORIGINAL "HEADLINE"

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

The tradeoff was framed as honest: G_M tied was ~20× more ops-per-correct-retrieval, but it got 100% under attack vs cuBLAS's 75-79%, used 500× less VRAM at extreme, and scaled to regimes where cuBLAS approached OOM. This was the result the project sat on for some time before the probes 11–20 trajectory below.

---

## Part 7 — The original four claims (subsequently revised)

For the historical record, what the benchmark trajectory above was thought to establish before the probes 11–20 sequence:

**1. G_M is a well-defined operator with three consistent implementations.** Verified analytically (Probe 9), implemented by the noiseless GPU sampler at shot noise (~0.01 MAE), and by the physical QPU circuit at characterized channel error (~0.10 MAE, in the range probes 7-8 measured). The tied-channel agreement metric reproduces this at scale. **This claim survived the later trajectory in modified form** — the operator is real, the three implementations are real, but the framing of QPU as a privileged substrate did not survive Probe 12's later correction.

**2. Per-dim aggregated G_M is structurally robust to coherent same-dim outlier attacks.** **Survived as stated**, with the caveat from Probe 18 across multiple bases that at certain attack profiles cuBLAS is no longer catastrophically failing — it's just indiscriminate at the attack-fraction rate.

**3. Streaming with fused argmax gives O(N) memory scaling.** **Survived.** Architectural, not a physics claim.

**4. The compute tradeoff is real and reported honestly.** **Survived** but the tradeoff is now better characterized: at d=256 the projection paths land at ~500× cuBLAS time (not ~20× as originally claimed), because the original number was measured without the full Flash-Squelch scoring step and at d=64.

---

## Part 8 — Probes 11 through 11.2: the projection channel reopened

After the V2 benchmark seemed settled, external feedback claimed the projection channel was "broken — only of interest to classical chip designers." The implication: projection only provides the certificate, geometry does all the real retrieval, projection contributes zero argmax signal.

### Probe 11 v1 — measure the projection-channel cost

Built two kernels in the same `cupy.RawModule`: the production `tied_streaming_perdim` and a `geom_only_streaming_perdim` variant with projection stripped. Three sweep sizes, same data, same attack.

**Result:**
- Δacc (tied − geom) = +0.00% on every row, every size, both bases.
- 5.3× speedup from removing projection from the hot loop.
- Agreement metric (when present in the dual kernel) reproduced Probes 7-8 numbers exactly: 0.104 QPU, 0.057 GPU.

The geometry channel was doing 100% of the retrieval work. Projection in the streaming kernel was paying 5.3× in compute for an output (the agreement value) that the runtime printed and discarded — no threshold, no assert, no gate.

**Implication framing at the time:** the projection channel is an "observer with no consumer" in the hot loop. Speedup is free if you move projection to a one-shot certify pass.

**Subsequent correction:** the user pointed out that the projection channel's primary value is *historical and derivational*, not runtime-active — it's how G_M was derived from physical hardware. The derivation role was complete by Probe 9; the runtime question is separable. This reframe was correct.

### Probe 11 v2 — `cp.async` pipelined tiling (bug discovered)

Tried to upgrade the three kernels (dual, geom-only, proj-only) to use `cuda::pipeline<thread_scope_block, 2>` double-buffered async loads, in preparation for a future group-GEMM rewrite.

**Result:** all three kernels at floor-of-random accuracy (~0.1-0.2%). Agreement metric was correct (because it sums over all keys and doesn't care about per-key attribution), which is what made the bug diagnosable.

**Two bugs in operand staging:**
1. K-side `memcpy_async` gated on `tid == 0` — 255 of 256 threads in each block read stale shared memory.
2. Q-side wrote per-thread query data into a single per-block shared buffer — last writer wins, 255 query rows clobbered.

**Lesson:** the v2 results looked like "all three kernels are aligned" because all three were getting identical wrong inputs. Not a finding about projection, just a bug.

### Probe 11 v3 — known-good memory pattern with proj-only

Reverted to the production memory pattern (Q in registers, K cooperatively loaded by all threads with thread-strided indexing) and added a `proj_only_streaming_perdim` kernel that drives the argmax off projection alone instead of geometry.

**Result:**
- tied / geom: 100% top-1 (matched, as expected)
- proj-only: 0.00% top-1 across all sizes, both bases

This was the surprise. The projection channel produced agreement of ~0.10 / ~0.06 (correct values) on a per-pair basis, but driving the argmax off it gave floor-of-random retrieval.

### Probe 11.1 — diagnosing why proj-only fails

CPU diagnostic that dumped projection per-pair distributions on query 0 against all keys.

**Result:**
- Per-pair geometry range on QPU: [0.345, 1.000], total range 0.655
- Per-pair projection range on QPU: [0.662, 0.714], total range **0.052** — **12× compression**
- Per-key aggregated projection std: 0.0005 (vs geometry's 0.0087 — **17× compression**)
- 78 keys (QPU) and 287 keys (GPU) score within 0.001 of the projection-argmax

**Mechanism:** importance reweighting compresses the per-pair dynamic range by ~10×. After averaging over d=64 dimensions, the per-key spread is below the float32 noise floor on a sum of 64 values. The signal is there; the per-key argmax has no resolution to pick it up.

GPU base showed the same compression and additionally placed the true match at rank 1023 out of 1024 — anti-correlated, not just noisy. Different `(orig_a, orig_b)` regimes produced different orderings.

### Probe 11.2 — minmax-rescale geometry into projection's range

Tested whether geometry's argmax survives min-max rescaling into projection's compressed window. Three variants: linear, quantile-match, per-row.

**Result:** all monotone rescalings preserved 100% top-1. The math is monotone-stable in float32 even with compressed values; projection's failure is **ordering distortion, not range compression alone**. Projection at certain `(orig_a, orig_b)` regimes doesn't track true similarity ordering, no matter how you stretch its output.

**Standing conclusion at this point in the trajectory:** projection serves the certificate role only. Geometry drives retrieval. Two different jobs.

---

## Part 9 — Probes 13 through 18: the bucket-mask exploration

The Probe 11.x conclusion seemed settled. Then a different question emerged: what if specific bucket pairs in the 18-int histogram are corrupting the projection signal, and dropping them recovers the retrieval capability?

### Probe 13 — bucket ablation

Systematic ablation of the nine `(f_a, f_b)` bucket pairs on the representative tile.

**Result:**
- Buckets (0,0) and (2,2) are pillars — dropping them kills the signal
- Buckets (0,2) and (2,0) are anti-pillars — dropping them sharpens the signal

### Probe 14 / 15 — the CUDA accident and the Golden Mask

A CUDA accident in v7 of an earlier mask-sweep iteration accidentally left `BLOCK_SIZE = 9` for the counts18 shared-mem load, dropping the upper 9 of the 18 ints. Signal recovery jumped from 0.4% to 99.1%. By accident.

Probe 15 then systematically swept all candidate masks at every threshold from 0 to 0.9, found that the principled mask (drop only the anti-pillars (0,2) and (2,0)) hit 99.2% — a cleaner result than the accidental block-9 mask.

**Framing at the time:** the anti-pillars carry "asymmetric quantum crosstalk." Dropping them removes the toxic component while keeping the denominator intact. This was named the "Golden Mask."

### Probe 16 — GhostFlow V4 production kernel

Bundled three findings into one CUDA kernel:
1. 7-bucket Golden Mask (drop (0,2) and (2,0)) — ~22% inner-loop reduction
2. Threshold + power (P=4096) Flash-Squelch scoring instead of mean-pool argmax
3. Online running-max accumulator (FlashAttention-style) so P=4096 doesn't overflow

Result on QPU representative tile: V3 (9-bucket) 8.7% signal, V4 (7-bucket Golden Mask) 87.4% signal under attack. With a dynamic threshold sweep.

**At this point we thought:** the production V5 kernel ships V4 with the Golden Mask hardcoded and a constant threshold, runs at 5.3× speedup over tied with measurable signal recovery, end of project.

### Probe 17 — threshold robustness sweep

12 QPU tiles × 10 seeds × 35 thresholds × 2 kernels = 8400 launches. Question: does V4 win on all 12 tiles, or only the representative one?

**Result:**
- V4 mean signal across all tiles at the recommended fixed threshold: **2.3%**
- V4 dynamic-optimum mean: 10.6%
- **Tile 11 alone hit 85.8% dynamic, 27.8% fixed. Ten of twelve tiles produced 0% under V4 regardless of threshold.**
- 90/120 V4 optima landed at the 0.50 threshold floor — the sweep range had been too narrow to find true optima on most tiles.

**Implication:** Probe 16 had been measured on the representative tile only (`representative_tile()` happens to select tile 11 for that base). The Golden Mask works on tile 11. It does not generalize.

### Probe 18 — cross-tile mask sweep on a single base

Eight masks × 12 tiles × 3 seeds × 200 thresholds. CPU implementation projected at 6 hours per base, then rewritten as a two-stage GPU kernel (precompute score matrix once per (tile, mask, seed); sweep all thresholds in one launch) that ran in 6 seconds per base.

**Result on base 1:** no universal mask. M3 (anti-pillars) wins on 0 of 12 tiles at ≥80% lock. M1 (baseline, no mask) wins on 6 of 12. Different tiles prefer different masks.

### Probe 18 fleet — across three independent QPU jobs

Ran the same 8-mask × 12-tile × 3-seed sweep on three QPU job files from the same algorithm. Three job-base configurations × 12 tiles each = 36 (tile, job) pairs.

**Result:** the optimal mask varies *both* across tiles within a job *and* across jobs for the same tile. Examples:
- Tile 10 (2,2): Job 1 wins M1 at threshold 0.0; Job 2 wins M5 at 0.0; Job 3 wins M6 at 0.5
- Tile 11 (2,3): Job 1 dead (0.3%), Job 2 dead (0.3%), Job 3 wins M5 at 92.3%

**Conclusion:** the Golden Mask was tile-11-of-one-specific-job-specific. The "anti-pillars carry toxic quantum crosstalk" physics framing from Probes 13-15 was a story about one calibration of one tile, not a universal property. Mask selection is calibration-dependent rather than physics-dependent.

This was a hard correction. The "GhostFlow V4 with Golden Mask" framing from Probe 16 — which was the production kernel for several iterations — was retracted.

---

## Part 10 — Probe 19: the Dynamic Mask Router

The cross-job result raised the question: can we predict the optimal mask from `(orig_a, orig_b)` alone, so the kernel picks per-tile masks at load time without seeing the bucket counts?

### Hypothesis (from a Gemini-articulated reframe of the cross-job data)

The QPU's error channel rotates with `(orig_a, orig_b)` through the bucket space. As the prepared angles sweep through the Bloch sphere, the "toxic" buckets migrate. Different `(a, b)` zones need different masks. A static lookup `(orig_a, orig_b) → mask` would be enough.

### Probe 19 — leave-one-job-out CV on the three Probe 18 dumps

Built a router that uses only `(orig_a, orig_b)` (no bucket counts) to predict the optimal mask, trained on two jobs and evaluated on the third, cycled through all three holdouts.

**Result:** router accuracy was poor. The same `(orig_a, orig_b)` produces different optimal masks across different calibrations of the same algorithm. The "rotating quantum noise" framing didn't survive the data.

This eliminated the simplest version of the production path. The router needed to see bucket counts, not just angles.

---

## Part 11 — Probe 20: the auto-calibrating production kernel

### V5 design

Per-base pre-flight calibration: for each base file, sweep all 8 candidate masks at 50 candidate thresholds against a small calibration set (CALIB_N×CALIB_N attention problem), with multi-seed median over CALIB_SEEDS seeds, subject to a spike-fraction constraint (`spike <= 0.05`). Pick the best (tile, mask, threshold) triple per base, write to a JSON manifest, lock for inference.

### First run (default settings: d=64, power=4096)

Found a `gpu.py` bug worth recording: bimodal behavior on identical inputs. The same algorithm on the same files would press-play to 100% lock or 0% lock, never in between, with no changes to anything. This had been hiding the whole time.

The user's diagnosis and fix to `gpu.py`: the original classical sampler was randomizing the control bit when `a_bit == b_bit`, but the GHZ-block circuit produces a deterministic `ctrl = 0` outcome in that case. The fix constrained `ctrl = 0` when `a_bit == b_bit` and a fair coin when they differ.

The fix made the bimodal behavior go away. The classical sampler now produces what the noiseless circuit actually outputs.

### Production run (d=256, power=256)

After increasing dimensionality:

```
type  file                                          tile  mask     thr    Signal     Status
QPU   job_d83putvtjchs73bpg5o0.npz                     0   M1   0.129   100.0%     LOCKED
QPU   job_d83q037oha1c73bn14p0.npz                     0   M1   0.055   100.0%     LOCKED
QPU   job_d83q0ivoha1c73bn15d0.npz                     0   M1   0.827   100.0%     LOCKED
GPU   ghost_oracle_gpu_seed_*.npz                      0   M1   0.092   100.0%     LOCKED
GPU   ghost_oracle_gpu_seed_*.npz                      0   M1   0.073   100.0%     LOCKED
GPU   ghost_oracle_gpu_seed_*.npz                      0   M1   0.073   100.0%     LOCKED
```

At d=256, M1 (no mask) wins on every base, at thresholds that vary per base but produce 100% lock everywhere. The Golden Mask story, the Dynamic Mask Router story, the rotating-noise story — none of it was needed. The √d SNR gain from d=64 → d=256 lifts the projection signal cleanly above the float32 noise floor without any mask architecture.

The probes 13–19 bucket-mask exploration was solving an SNR problem that increasing d dissolves. The mask infrastructure was preserved in the production V5 kernel as a defensive measure for harder operating points (narrower d, fewer shots, harsher attacks) but does nothing at the production point.

---

## Part 12 — Probe 12 redux: the substrate-equivalence correction

After the GPU sampler fix in Probe 20 and the d=256 production result, Probe 12 (the original "quantum advantage" claim) needed re-running with the corrected classical sampler.

### The original Probe 12 finding (now retracted)

"GPU has true match below noise floor (gap = −0.003), QPU has it above (gap = +0.007). Topologically impossible to run on a GPU. Quantum advantage from physical preservation."

This had been the project's strongest claim for many iterations.

### The corrected Probe 12 run

Same FP64 numpy reference verifier, d=64, against the corrected `gpu.py`:

```
QPU base:
  GEOM gap (True - Bg):     0.07014
  PROJ gap (True - Bg):     0.06161

GPU base:
  GEOM gap (True - Bg):     0.07014    (identical to QPU)
  PROJ gap (True - Bg):     0.12119    (almost 2× the QPU gap)
```

**The classical noiseless sampler produces a *larger* projection gap than the QPU does.** Both substrates retrieve cleanly. The QPU's projection signal is *attenuated* by hardware noise relative to the noiseless reference; it isn't a privileged substrate.

The original Probe 12 result had been measuring the `gpu.py` bug, not a physics property. The buggy classical sampler was producing bucket counts so noisy that projection couldn't extract a signal. The QPU was less broken than the buggy classical baseline. Both signals were degraded; the QPU's was degraded less. The interpretation "topological preservation gives quantum advantage" was wrong — the right interpretation was "the QPU is less broken than the buggy classical sampler."

**This invalidates the "quantum advantage" framing the project had been operating under for the entire benchmark trajectory before this correction.**

The user, to their credit, had been consistently framing the project as "same physics, three platforms" rather than "quantum advantage" throughout. The "quantum advantage" framing was something I (the AI assistant collaborating on these probes) kept reaching for and being corrected on. Eventually the data forced the issue.

---

## Part 13 — The final five-way benchmark

### `final_benchmark_5way.py`

Same retrieval problem, five score backends, all measured on the same 4096×4096 attention matrix under the Probe 10.1 attack profile, with per-base calibration via Probe 20's pre-flight pass:

1. **CUBLAS** — standard dot-product attention via cuBLAS gemm. Classical transformer baseline.
2. **TIED** — production dual-channel kernel. Argmax = geometry; agreement metric computed and reported.
3. **GEO** — geometry-only argmax. Same top-1 as TIED, differs only in that TIED computes agreement.
4. **QPROJ** — projection driven by QPU bucket counts, calibrated per-base.
5. **GPROJ** — projection driven by classical noiseless bucket counts, calibrated the same way.

### Result

```
CUBLAS              top1=100.0%  sig=100.0%  spk=0.0499  t=1.21 ms
TIED (per QPU base) top1=100.0%  sig=100.0%  spk=0.0498  t≈500 ms
GEO  (per base)     top1=100.0%  sig=100.0%  spk=0.0498  t≈500 ms
QPROJ (mean 3 QPU)  top1=100.0%  sig=100.0%  spk=0.0498
GPROJ (mean 3 GPU)  top1=100.0%  sig=100.0%  spk=0.0498

Agreement metric per base:
  QPU base 0:  0.0818       GPU base 3:  0.0163
  QPU base 1:  0.0727       GPU base 4:  0.0289
  QPU base 2:  0.1307       GPU base 5:  0.0103
```

### What this verifies

Same algorithm, three substrates (mathematical, classical noiseless, quantum hardware), one classical control. All retrieve at 100% top-1 at d=256 with calibrated per-base squelch.

The platform-specific story is in the agreement column: GPU agreement clusters at ~0.02, QPU agreement at ~0.10 — a 5× ratio. This is the **quantitative hardware-noise readout** the substrate-comparison architecture was built to produce. Real QPU shots reproduce the analytical operator with ~5× more divergence than noiseless classical shots of the same circuit. Same algorithm, measurable hardware-noise attenuation, no exotic advantage.

The cuBLAS spike fraction of 0.0499 is exactly the attack fraction (5%) — cuBLAS isn't catastrophically broken under this attack profile at d=256, it's just *indiscriminate*, giving attack keys proportional weight. The projection paths don't win on attack robustness at this operating point either; they all sit at the same spike rate.

The original Probe 10.1 attack-robustness finding still holds in principle (per-dim G_M is bounded so single dims can't dominate), but at d=256 the cuBLAS attention is dispersed enough that the attack doesn't catastrophically concentrate. The robustness comparison would need a harder attack profile (larger magnitude, higher fraction, smaller d) to show the differentiation cleanly.

### What the benchmark establishes

The project's actual final claim, stripped of the framings the trajectory disproved:

**Same physics, three platforms.** The projection-channel attention operator is defined mathematically. It admits three faithful substrate implementations (analytical, classical noiseless, real QPU shots). All three retrieve cleanly at d=256 with per-base calibration. The agreement metric quantifies hardware-noise attenuation when run on physical QPU shots (~5× more divergence than noiseless classical of the same circuit). The architecture is substrate-agnostic by construction.

The cuBLAS comparison is honest: at this operating point cuBLAS does the same retrieval ~500× faster on tensor cores. The projection-channel kernel earns its place on (a) the substrate-comparison framework — running the same algorithm faithfully on different physical implementations — and (b) the agreement metric as a continuous integrity check, not on raw throughput.

---

## Part 14 — Updated open questions

These supersede the open questions from the original Part 10 of this record.

1. **Harder attack profiles.** The d=256 result locks at 100% on cuBLAS too. The original Probe 10.1 attack differentiation (DP top-1 collapses from 74% to 43%) was at d=16-64. Sweep magnitudes / fractions / dimensions to find the regime where cuBLAS catastrophically fails and the projection paths still hold. That's the legitimate version of the attack-robustness claim.

2. **Real LLM embeddings.** Still open from the original record. Pull K/Q from a real transformer and run the five-way benchmark. The synthetic result needs grounding on actual learned representations.

3. **Production kernel performance gap.** cuBLAS runs at ~1ms; the projection kernels at ~500ms. The gap is not all transcendental ops — the projection kernel does d=256 fp32 work per pair without tensor cores. Investigate whether a tensor-core-friendly G_M formulation exists that closes the gap. If yes, the substrate-comparison story becomes shippable production code, not just a research demonstration.

4. **Why does QPU agreement vary across calibrations?** Across the three QPU bases the suite ships with, agreement values are 0.0818, 0.0727, 0.1307 — meaningfully different. Some calibration-dependent factor (gate calibration drift, queue timing, backend rotation) drives a ~80% variance in projection-channel agreement. Characterizing this would let the agreement metric serve as a per-job hardware-quality readout for users selecting which jobs to use downstream.

5. **The mask infrastructure's actual purpose.** At d=256 it's defensive. At what operating point does it become load-bearing? If the answer is "never under realistic configurations," it could be removed. If it's "narrower d for memory-constrained inference," it's worth keeping. Run the auto-calibrating V5 across a d sweep and find out.

6. **The geometry-only production path.** Probe 11 v1 showed the geometry kernel alone gives identical retrieval at ~5.3× the speed. The reason this isn't the production path is the per-query agreement certificate. But for production environments that don't need the certificate (e.g. inference at scale where the kernel has been validated), the geometry-only kernel is the right shipping target. Document the use-case split.

---

## Part 15 — Known issues, carried forward and updated

### Probe 8.2 alternation loop is broken

Unchanged from the original record. Stage 2+3 usable, alternation loop is not. Marked in the probe file header.

### Probe 9 Demo 2 broken

Unchanged. Fixed in Probe 9.1.

### Probe 10 superseded by 10.1

Unchanged. Kept for trajectory legibility.

### Probe 8.4 base-2 Benford / ν_p shot-count artifacts

Unchanged.

### `gpu.py` GHZ-block correlation

Fixed during Probe 20 trajectory. The original `gpu.py` randomized the control bit when `a_bit == b_bit`, producing classical samples that didn't faithfully implement the noiseless circuit. The current version constrains `ctrl = 0` when `a_bit == b_bit` per the deterministic GHZ-block circuit output. **All bench numbers prior to this fix that compared QPU vs GPU bases were measuring the bug, not the physics.** The corrected Probe 12 result in Part 12 supersedes any prior "GPU below noise floor" / "quantum advantage" framings.

### Probe 16 "Golden Mask" framing retracted

The Probe 16 V4 production kernel's central claim — that dropping anti-pillar buckets (0,2) and (2,0) is a universal property of the projection circuit — was tile-and-calibration-specific. The Probe 18 cross-job sweep proved no universal mask exists. The auto-calibrating V5 kernel in Probe 20 handles this correctly; the historical V4 framing in the codebase is retracted in favor of "per-base calibration picks the right mask for each calibration."

### Probe 12 original framing retracted

The original Probe 12 claim ("topologically impossible to run on a GPU; quantum advantage from physical preservation") was measuring the `gpu.py` bug. Corrected result in Part 12 shows substrate equivalence with measurable QPU hardware-noise attenuation. The "quantum advantage" framing is retracted; the "same physics, three platforms" framing is what the data actually supports.

### Phase-lift design is informal

Unchanged from the original record. Still open work.

---

## Part 16 — Philosophy and license

Unchanged from the original record. CC0. Build, break, fix, document, repeat. All in the open.

The "break-it-fix-it" rule: if you find something wrong, you provide the fix alongside the bug report. Not as a gatekeeper, but as a norm — fixes-with-bugs travel through the project faster than bugs alone.

This document is part of that. The bugs in probes 8.2, 9, 10 are documented because the *process* of finding and fixing them is the research. The retracted framings in probes 12 and 16 are documented because the process of overclaiming and then being corrected by the data is also part of the research, and arguably the most important part. A future contributor (human or agent) who reads this should know exactly where we landed, what we tried, what worked, what didn't, and what we got wrong and had to fix.

The lesson from the Probes 11–20 trajectory is worth pinning down explicitly: **a result framed as "quantum advantage" needs an unimpeachable classical control to back it up.** Our classical control was buggy for the entire pre-Probe-12-correction trajectory, and the bug happened to make the quantum result look better than the classical. Once the classical control was fixed, the framing changed from "the QPU does something a GPU can't" to "the QPU faithfully implements the same algorithm as the classical noiseless reference, with measurable hardware noise on top." That second framing is weaker-sounding but stronger — it's the framing the data actually supports, and it's the one that survives review.

---

## Closing

The QPU isn't a noisy matrix multiplier. It is also not a privileged substrate that performs computations classical hardware cannot. It is a faithful physical implementation of the same projection-channel attention algorithm that the classical noiseless sampler and the mathematical reference also implement. Hardware noise on the QPU attenuates the projection-channel signal relative to the noiseless reference by a measurable factor (~5× in the agreement metric), but does not break the algorithm at the production operating point.

The probes establish that the operator exists, that it has a clean closed form, that all three substrates implement it faithfully, and that per-base calibration handles the variance that arises from physical noise. The five-way benchmark establishes that all three substrates plus the cuBLAS classical control retrieve at 100% top-1 on a 4096×4096 attention matrix under coherent attack at d=256.

The project's actual contribution is the substrate-equivalence demonstration plus the auto-calibrating production kernel that runs faithfully on all three. Same physics, three platforms.

Everything in this repo is the working out of that single result, including the parts where we got it wrong first.
