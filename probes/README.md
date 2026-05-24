# The Probes

The numbered probes are the research log. They're chronological, not curated. The wrong target is in here. The bugs are in here. The negative results are in here. So are the corrections.

This is not the writeup — `PROCESS_RECORD.md` at the repo root is. This is the table of contents.

If you want to *use* G_M, go to `ghost_oracle/projection_benchmark.py` and `docs/math.md`. If you want to *understand how the project got there*, read in order.

---

## The arc in one paragraph

A 12-tile Hadamard-test circuit on IBM Quantum was assumed to be computing `T1(a, b) = |cos(a-b)|`, the textbook rank-1 cosine. Probes 1, 2, 3 dismantled that assumption — the QPU isn't computing T1, the "holographic geometry-coupled structure" earlier code thought it found is a sampling artifact, and no smooth channel correction rescues either claim. Probe 4 is the pivot: simulate the actual circuit and see what it computes. The answer is `T3 = 3/4 + (1/4) cos(a) cos(b)`, a mixed-state form coming from ghost CNOTs that entangle the swap-test qubits with their ancillas before the Hadamard test. Probes 5, 6, 7 validate T3 against the GPU sampler, the analytical reference, and the physical QPU data. Probe 8 spends four phases trying to attribute the residual QPU error to canonical channels and concludes the residual is dominated by ghost decoherence with per-tile structure no smooth model captures. Probe 9 reframes the question and discovers T3 has a much cleaner closed form, `G_M(a, b) = sqrt((1 + cos a cos b) / 2) / α` — the operator that drives the rest of the suite. Probe 9.1 fixes Probe 9's two known bugs and characterizes G_M as a real but indefinite kernel. Probes 10 and 10.1 carry G_M into the attention context; 10.1 is what closed the arc with the headline result.

---

## Reading order

| # | File | One-line role |
|---|---|---|
| 1 | `probe1_identity_bridge.py` | Tests `T2 = \|cos((a-b)/2)\|`. Fails. |
| 2 | `probe2_projection_scrambled_control.py` | Tests whether Benford structure is geometry-coupled. It isn't. |
| 3 | `probe3_anchor_conditioned_projection.py` | Tries to rescue Probe 2 with channel inversion. Doesn't work. |
| 4 | `probe4_build_base.py` | **THE PIVOT.** Simulates the circuit, derives T3, builds the noiseless reference base. |
| 5 | `probe5_unified_engine.py` | Reruns Probes 1–3 against T3 instead of T2. Everything snaps into place. |
| 6 | `probe6_3way_convergence.py` | Formalizes the three-target framing (T1 / T2 / T3) for reviewers. |
| 7 | `probe7_ghost_parity.py` | Confirms GHZ correlation is physically present in QPU shots. |
| 8 | `probe8_residual_decomposition.py` | Four-phase residual characterization (A/B/C/D). |
| 9 | `probe9_ghost_operator.py` | Derives G_M from T3. Names the operator. Two known bugs. |
| 9.1 | `probe9_1_indef_kernel_attn.py` | Fixes Probe 9's bugs. Closes the indefinite-kernel and scalar-attention angles. |
| 10 | `probe10_ghost_attention.py` | First attention attempt. Three setup bugs suppress the mechanism. **SUPERSEDED.** |
| 10.1 | `probe10_1_real_softmax_attack.py` | The headline result. Per-dim G_M aggregation under coherent same-dim attack. |

Plus `benchmark_evolution/` — five iterations of the projection benchmark before the production version landed in `ghost_oracle/projection_benchmark.py`. Useful if you want to see why the headline benchmark looks the way it does.

---

## Probes 1–3 — Dismantling the wrong target

The original framing was: the QPU is a Hadamard-test circuit, the Hadamard test computes `|cos((a-b)/2)|`, ergo the QPU's normalized output should match `|cos((a-b)/2)|` up to a global gain `ALPHA_NORM = 0.9127`. Three probes systematically took that apart.

**Probe 1 — Identity Bridge.** Direct comparison of QPU output, GPU rank-1 kernel, and analytical `|cos((a-b)/2)|`. Result on the original 12-tile job: QPU vs analytical MAE 0.19, max diff 0.50, with a -0.18 bias. The GPU kernel matches analytical to fp32 noise (1e-7). The QPU does not match the textbook formula, and `ALPHA_NORM` is behaving like a global gain rather than a geometry correction. Something structural is missing.

**Probe 2 — Projection-Scrambled Control.** Earlier code claimed to find "Benford holographic structure" in the QPU manifold stream that aligned with the intended projection geometry. This probe tests whether the alignment is real by running the same Benford and recursive-manifold telemetry through (1) the intended geometry, (2) a structurally-identical but scrambled-angle projection, and (3) a null ensemble of M scrambled projections. Result: z = -0.36 (Benford) and z = -0.99 (recursive) — no separation. The structure is in the bitstream, not coupled to the projection.

**Probe 3 — Anchor-Conditioned Projection.** Tries to rescue Probe 2 by inverting a per-tile depolarization channel before projecting — four anchor schemes (interior-weighted, decoherence-weighted, held-out anchors, uniform) plus a wrong-channel control. Result: none beat blind baseline. All four anchor variants score `|z| < 2` against null, and leave-one-out R² is negative across the board — a smooth `|a-b|` + `ideal_p0` model doesn't fit the residual. The channel has per-tile structure no linear model can capture.

End of Probes 1–3: the original framing is dead. ALPHA_NORM isn't a geometry correction, the "holographic structure" isn't geometry-coupled, and no smooth channel inversion rescues either. Something else is going on.

---

## Probe 4 — The pivot

Instead of trying to fix the QPU's deviation from the textbook formula, **simulate the actual circuit and see what it computes**.

The circuit has seven qubits per tile: `[a1, v1, a2, ctrl, b1, v2, b2]`. Two `Ry(θ)` rotations prepare `v1` and `v2`. Four CNOTs entangle each `v` with its ancillas (`v1→a1, v1→a2, v2→b1, v2→b2`) — these are the "ghost CNOTs." Then the standard Hadamard test: H on ctrl, CSWAP(ctrl; v1, v2), H on ctrl, measure.

The ghost CNOTs are the thing the textbook analysis misses. After them, `{v1, a1, a2}` and `{v2, b1, b2}` are each in a GHZ block, so the swap test isn't acting on product states — it's acting on parts of a larger entangled state. The probe derives the resulting 32-bin joint distribution in closed form, then samples it on the GPU and verifies that the analytical and empirical marginals agree to shot noise on all 12 tiles.

The result is `T3`:

```
T3(a, b) = ¾ + ¼ cos(a) cos(b)
```

Probe 4 also builds the **noiseless GPU base** — a `.npz` byte-compatible with QPU dumps but generated from the closed form. The whole rest of the suite consumes either physical QPU shots or this noiseless reference, swapping freely.

---

## Probes 5–7 — Validating T3

**Probe 5 — Unified Engine.** Reruns the Probes 1, 2, 3 battery against T3 instead of T2. Identity Bridge MAE drops from 0.19 (against T2) to 9.8e-3 (against T3 on the noiseless GPU base) — shot noise. The Benford "signal" Probe 2 chased now lands at low z-scores even on the noiseless base, confirming it was a sampling artifact. The anchor-conditioned channel fit coefficients collapse near zero on the noiseless base; on the QPU base they pick up real structure — that structure is the residual Probe 8 spent four phases on.

**Probe 6 — Three-Way Convergence.** Formalizes the framing the rest of the suite inherits: cuBLAS hits T1 to fp32 noise, the custom CUDA kernel hits T1 to the same precision via a different code path, the noiseless GPU base hits T3 to shot noise, and the QPU hits T3 within hardware error. The throughput benchmark in this probe is the seed of `ghost_oracle/projection_benchmark.py`.

**Probe 7 — Ghost Parity.** Direct physical confirmation. If the ghost CNOTs are working on the QPU, then `a1` and `a2` should be perfectly correlated (and similarly `b1`, `b2`), since each pair sits inside a GHZ block. The probe measures `P(a1 ≠ a2)` and `P(b1 ≠ b2)` and compares to the independence null. Result on QPU: mean `P(a1 ≠ a2) = 0.15` (null = 0.30), `P(b1 ≠ b2) = 0.17` (null = 0.36). Strong entanglement evidence on most tiles. The gap from zero is decoherence — what Probe 8 went on to characterize.

End of Probes 5–7: T3 is the right target. The pivot held.

---

## Probe 8 — Residual decomposition (four phases)

The QPU still has ~10% MAE against T3 after the pivot. Probe 8 is the systematic attempt to attribute that residual to canonical error channels. It unifies what were originally four separate scripts (8.0, 8.1, 8.2, 8.4) into a single phased file because they share so much code.

**Phase A — Initial channel fit (was probe 8.0).** Three-channel mixture (depolarization + ghost-decoherence + symmetric 5-qubit readout). Finds the symmetric-readout model is *degenerate* with ghost decoherence — the fitter slides along a `(lam_g, lam_r)` ridge — and channel inversion actually makes MAE *worse* by 32.5%. The channel fit is diagnosing model misspecification, not real structure.

**Phase B — Split readout (was probe 8.1).** Replaces the symmetric readout with two independent rates: `eps_ctrl` on the control qubit, `eps_ghost` on the four ghost qubits. The degeneracy breaks, MAE reduction works (+20.9%), and ghost decoherence emerges as the only essential channel. Also discovers a large coherent angle drift (~-0.2 rad on `a`, ~-0.1 rad on `b`) in the post-fit residual.

**Phase C — Drift-first alternating (was probe 8.2). KNOWN ISSUE.** Tries to fit a shared coherent drift first, then per-tile residual drift with regularization, then channels. The alternation optimizer has three compounding bugs (`d += d_new` accumulating, OUT penalty on variance not magnitude, no hard bound on |d|) that let `d_b` run to -210° at n=4. Even at n=1 the result is worse than Phase B. Documented in `docs/known_issues.md`. **Phases C.2 and C.3 (single shared-drift + per-tile channel without alternation) are usable; the alternation loop is not.**

**Phase D — Benford / p-adic null sweep (was probe 8.4).** Forensic telemetry on the Phase B and Phase C residuals across five Benford bases and three p-adic valuations, with shuffled-within-tile nulls. Result: a handful of scattered `|z| > 2` hits, none consistent across bases. Nothing meaningful survives. The residual is decoherence, not hidden structure.

Cumulative finding: the QPU residual is dominated by ghost decoherence (`lam_g ~ 0.3` across tiles) plus per-tile structure no smooth global model captures. This negative result is what motivated Probe 9.

---

## Probe 9 — The operator

After eight probes of trying to fix the residual against T3, the question shifted: **what is the operator the QPU is actually computing, presented as cleanly as possible?**

T3 is `¾ + ¼ cos(a) cos(b)`, a low-order trigonometric polynomial. Take its square-root form and normalize:

```
G_M(a, b) = sqrt((1 + cos(a) cos(b)) / 2) / α
```

Verified at machine precision against T3. Three consistent implementations: the closed form, the GPU sampler at shot noise, the QPU at characterized channel error. Probe 9 runs five stages — consistency, characterization (Pearson, range, spectral), classification (Mercer test, kernel-zoo comparison), application (two demos), scaling (timing sweep).

The structural findings are real and survive into the rest of the suite:

- G_M is NOT positive semidefinite (0/50 PSD tests passed). Indefinite pairwise operator, not a Mercer kernel.
- corr(G_M, cos(a)·cos(b) outer product) = 0.9992 — G_M is structurally cos-outer-product with a saturating sqrt nonlinearity.
- corr(G_M, standard matmul) = -0.75 on random inputs — anti-correlated. Important structural fact.
- N² scaling, 3-5× constant overhead vs matmul.

Two known issues, both fixed in Probe 9.1:

- **Stage 1 clipping artifact.** Three tiles have analytical G_M = 1 due to the `min(1.0, ...)` clamp firing when `cos(a) cos(b)` is small. The reported MAE looks bad but is dominated by the clipped tiles.
- **Stage 4 Demo 2 broken.** The "saturating regression" demo uses inputs where the truth function is nearly linear over the range, and the G_M-oracle feature was literally equal to the truth function, so its MSE was machine-epsilon and the ratio reporting divided by zero.

Both kept verbatim in the repo for trajectory legibility.

---

## Probe 9.1 — Fixed demos plus three new

Re-runs the broken pieces and adds three demos that close out the question of where G_M actually carries its weight.

**Stage 1 fixed.** Reports unnormalized `G_M_raw`, removing the clipping artifact entirely. GPU MAE drops to 0.01, below shot noise.

**Stage 2 fixed.** Moves the regression inputs to angle space (where the sqrt curvature is visible) and adds a 0.85 ceiling so the truth has both curvature and saturation. G_M's structural form captures the function in zero parameters; degree-3 polynomial needs 10 parameters to hit 6e-7 MSE. **Clean positive result for G_M's structural advantage.**

**Stage 3 new — Indefinite-kernel SVM.** Tests whether G_M's indefiniteness is informative on PSD-friendly classification tasks. Result: G_M loses cleanly to RBF (35% vs 98%). **Indefinite-kernel angle dead.**

**Stage 4 new — Scalar phase-lift attention.** First attempt at G_M-as-attention by mapping each (Q, K) embedding to a single angle. Loses to dot-product on representation tasks because the scalar lift throws away most of the embedding. **Motivates Probe 10's per-dim aggregation.**

**Stage 5 new — Scaling.** Classical G_M vs cos-outer on the same input sizes; 2-4× overhead from sqrt + add.

Two doors closed, one door opened.

---

## Probes 10, 10.1 — Attention

**Probe 10 — first attempt. SUPERSEDED.** Tries to demonstrate G_M as a native attention primitive under outlier attack. Per-dim aggregation instead of the scalar lift from Probe 9.1 Stage 4. Three setup bugs cancel the experiment:

1. L2 renormalization after spike injection neutralized the attack
2. Outliers on different dims per key — incoherent attack, not the real LLM-attention failure mode
3. d=64 with renormalized embeddings made softmax already-uniform — bottleneck mechanism never engaged

Both methods plateaued at 95% = 1 - outlier_fraction. Looked like the hypothesis was wrong; was actually three bugs canceling out the test.

**Probe 10.1 — the headline.** Fixes all three:

1. No L2 renormalization, so a spike survives as a real attack
2. Same-dim coherent attack: all corrupted keys spike on a single shared dimension — the actual LLM failure mode
3. d=16 so softmax over N=1024 has room to concentrate

Plus a tanh-based phase-lift for unbounded inputs.

**Result at N=1024, d=16, 5% attack at magnitude 50:**

- Dot-product top-1: 74% clean → **43% under attack**
- G_M tied per-dim top-1: 84% clean → **84% under attack**
- Softmax attention mass on outliers: **0.42** (catastrophic concentration)
- G_M outlier/non-outlier score ratio: **0.999** (the attack is invisible to G_M)

The architectural justification for everything in `ghost_oracle/projection_benchmark.py`. The benchmark extends the same mechanism to N=65536 inside a tied streaming kernel where cuBLAS approaches OOM and G_M holds at 100% retrieval with 500× less VRAM.

---

## Known issues, summarized

The probe headers and `docs/known_issues.md` carry the full text. Quick index:

| Probe | Issue | Status |
|---|---|---|
| 8.2 (Phase C alternation) | Three compounding bugs in alternating optimizer | Documented, alternation disabled; Phase B usable |
| 8.4 (Phase D, base-2 Benford) | Implementation bug makes base-2 column flat | Documented as non-informative; other bases informative |
| 8.4 (Phase D, p-adic) | ν_p tests confounded by shot count = 2^12 | Documented; results inconclusive at 4096 shots |
| 9 Stage 1 | Clipping artifact inflates MAE on 3 tiles | Fixed in 9.1 Stage 1; original kept verbatim |
| 9 Stage 4 Demo 2 | Truth function equals the oracle feature | Fixed in 9.1 Stage 2; original kept verbatim |
| 10 | Three setup bugs neutralize the attack | Marked SUPERSEDED; 10.1 is the corrected experiment |

The norm: bugs stay in the repo with header markers, the corrections live in numbered follow-ups, and `docs/known_issues.md` is the running list with full diagnoses. The point of keeping the broken versions is that the *process* of finding and fixing them is part of the research record.

---

## Running probes

Every probe is self-contained and runnable:

```bash
# Most probes auto-find their bases in data/
python probes/probe1_identity_bridge.py
python probes/probe4_build_base.py            # generates a noiseless base
python probes/probe9_ghost_operator.py        # needs both QPU and GPU bases
python probes/probe10_1_real_softmax_attack.py  # synthetic, no base needed

# Or point at specific bases
python probes/probe5_unified_engine.py \
    --qpu data/job_xyz.npz \
    --gpu data/noiseless_base_xyz.npz
```

Probes 2, 3, 5, 8 take `--mode smoke|stress`. Probe 8 also takes `--phase A,B,C,D,all`. Most probes accept `--num-tiles` to match the base they're being run against (the historical runs used 12 tiles; current QPU jobs use 16).

The numbers won't match the historical record byte-for-byte — different jobs, different seeds, different library versions — but the qualitative findings hold. Each probe's `HISTORICAL CONTEXT:` block in the header docstring explains what the original run reported and what to expect now.

---

## If you want to read just one thing

`PROCESS_RECORD.md` at the repo root. The probes are the working artifacts; PROCESS_RECORD is the narrative with the receipts.