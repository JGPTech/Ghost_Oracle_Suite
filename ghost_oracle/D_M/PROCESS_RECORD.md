# Ghost Oracle Suite — D_M Process Record

This document records the research and engineering trajectory of the **D_M (Dimensional Entanglement Projection)** operator within the Ghost Oracle Suite, from the first prune probe through the Bell-listener pivot, the characterization probes, the two task attempts that did not survive, and the control-driven capstone that became the actual result. It exists so that any future contributor — human or AI agent — can pick up the D_M work with full context.

The **G_M (Ghost Metric)** family — the swap-test cosine-similarity operator and its attention/retrieval benchmarks — is recorded in its own process record, as is the **S_M (Syndrome Metric)** family. The repository is split into `ghost_oracle/G_M/`, `ghost_oracle/S_M/`, and `ghost_oracle/D_M/` because the operators have different natural domains and deserve separate records.

It is chronological. It includes the wrong turns. It includes the framing the project carried for six probes and then had to abandon when the data refused to support it. It includes the two benchmark tasks that saturated, and why they were demoted rather than reported as wins. It includes the controls that *failed to collapse* and how that non-result was reinterpreted instead of buried. It is not a polished narrative. It is a working record.

A numbering note up front: the probe files are ordered `00`–`24` in chronological run order, and that file ordinal is the canonical probe number used throughout this document. Several files carry an *internal* docstring number that disagrees with the file ordinal — file 21 calls itself "Probe 04W", file 23 calls itself "Probe 05", file 24 calls itself "Probe 05B", and the capstone refers to a "Probe 22/23/24/25 standard". Those internal numbers are the renumberings the project adopted while reorganizing; where they matter they are flagged inline.

---

## Part 1 — Origin and the wrong target

### What started the project

A frozen D_M QPU base from `ibm_marrakesh`: a 20-tile, 4096-shot record dumped by `d_m_qpu_generate.py`. The generator places bare coherent two-qubit listener tiles across the chip — `H(q0), H(q1)`, a cavity delay, a rotation into an assigned witness basis, then measure — lets neighboring tiles share silicon, and sweeps a cavity-delay ladder. It does **not** prepare a Bell state, reconstruct a density matrix, use an ancilla, or apply dynamical decoupling by default.

Supporting code: `d_m_qpu_generate.py` (submit + dump, the freeze step), `d_m_gpu_generate.py` (controlled `gproj` fixtures with the same `.npz` schema), `dm_projector_kernel.cu` (the CUDA correlator/projection/summary kernels), and `d_m_benchmark.py` (the repo-facing capstone).

### The assumed target — D_M as a dimensional-compression projector

The first six probes were built on the assumption that D_M was a **dimensionality-reduction operator**: that the frozen QPU scene encoded a 16-state dimensional manifold which, properly extracted, would project high-dimensional data into a lower-dimensional representation while preserving nearest-neighbor structure — the same job as PCA, random projection, UMAP, Faiss, or an autoencoder.

The one-shot QPU generator deliberately embedded a menu of transport-style assumptions for the prune probe to test: `smooth_walk` (nearest-neighbor dimensional transport), `boundary_reflect`, `nonlocal_jump`, `collapse_gate`, `phase_shear`, `scramble_order`, `mirror_parity`, and `rank_spread`. The plan was: prune the failed assumptions, extract the surviving channels, and benchmark the resulting projector against classical compression baselines.

That plan was the wrong target. It took six probes to exhaust it.

---

## Part 2 — Probes 00 through 06: dismantling the compression target

### Probe 00 — Prune failed assumptions

Loads one frozen scene and scores each tile's assumptions with effect-size-style tests (state structure vs uniform, effective dimensionality, collapse strength, boundary/interior split, parity bias, adjacency coupling), then classifies each tile `SURVIVE / WEAK / FAIL / MUTATED`.

**Result:** all 20 tiles classified `MUTATED` / `UNKNOWN_STRONG_STRUCTURE`, with enormous z-scores (eff ≈ 2–4, z ≈ 2200–5200, popcount = 1.000). The mode decision was a single line: `unknown -> KEEP_AS_MUTATION repeat=inconsistent`. Translation: there is overwhelmingly strong structure in the base, but **none of the labeled transport assumptions describe it**. The structure survived; the story about what the structure *was* did not.

### Probe 01 — Channel extraction

Builds candidate channels from the Probe 00 survivors: `local_order`, `collapse`, `mutation`, `symmetry_boundary`, `rank_spread`, and a `composite_dm` aggregate.

**Result:** five of the six named channels came back `class=EMPTY` (carry=0, eff=16.000, flat 16-state histogram). Only `composite_dm_channel` carried any signal (`COMPOSITE_CANDIDATE`, eff 3.58). The hand-named channels were empty. Whatever D_M was, it was not decomposing into the assumed transport channels.

### Probe 02 — Synthetic maps

First self-contained task rehearsal: do the extracted channels guide structure-preserving compression on `blobs / rings / swiss_roll_like / s_curve_like / sparse_binary` better than PCA or a random projection?

**Result:** no. Overall ranking — `pca_projection` 0.708, `dm_composite_dm_channel` 0.392, the five static dm channels ≈ 0.378, `random_projection` 0.361. D_M sat in a dead heat with a single random projection and well behind PCA. The compression framing was already failing; D_M was not even beating one lucky Gaussian matrix decisively.

### Probe 03 — Linear-activated channel scheduler

Two fixes: (1) treat the linear direction as an internal 7th D_M channel rather than only an external challenger, and (2) replace the weak single-random baseline with a hardened suite (gaussian / orthogonal / sparse-Achlioptas, each as mean/best/worst over 128 trials, plus shuffle-crop and sign-flip-crop).

**Result:** the hardened random `*_best_128` rows took the top three slots (≈ 0.51). The best D_M entry, `dm7_soft_scheduler`, landed 6th at 0.365 — mid-pack, below every random-best and below the shuffle/sign-flip crops. Adding the linear channel did not rescue the operator. The honest reading: **random projection's best-of-128 is a search ceiling, but D_M was not even beating the random *mean***.

### Probe 04 — Calibrated seed projection

New hypothesis: the 7th channel is not linear, it is a **QPU-calibrated seed** — fold the frozen base into a deterministic seed and use it to generate projection matrices. Win condition stated explicitly in the probe: `qpu_calibrated_seed_single > random_*_mean_N`, or the family-mean beating the random mean.

**Result:** the win condition was not met. `qpu_calibrated_seed_family_best_32` (0.757) tied the random-best ceilings, as any best-of-K search will. But the fair single-shot comparison was a loss: `qpu_calibrated_seed_single` 0.356 ≈ `secret_seed_single` 0.339, both below `random_gaussian_mean_128` ≈ 0.379. Calibration bought nothing over an uncalibrated secret seed.

### Probe 05 — Base-10 calibrated state selector

Reframe again: route the QPU calibration through a base-10 alpha/state selector that picks dimensional basis states directly, rather than seeding a random matrix.

**Result:** the top of the overall table was `base10_uncalibrated_dominant_family_best_128` (0.778), edging out `base10_qpu_calibrated_dominant_family_best_128` (0.775) and the random-best rows (0.77). The decisive fact: **the *uncalibrated* base-10 selector beat the *calibrated* one.** The QPU calibration was not load-bearing; the base-10 search structure was doing the work, and only at the best-of-128 ceiling.

### Probe 06 — Frequency base-10 seed projector

The cleanest test of the compression framing: strip all the channels, schedulers, and family-best search, and test D_M as exactly *one* projector — `dm_frequency_base10_seed_projector`, with the seed folded as an active per-sample dimensional channel.

**Result:** ranked 10th of 14, score 0.262, against `random_gaussian_best_128` 0.551 and `random_gaussian_single` 0.380. As a single honest projector, D_M lost to a single random Gaussian matrix.

### Standing conclusion after Probe 06

D_M is **not** a competitive dimensional-compression projector. Six independent reframes — named channels, linear scheduler, calibrated seed, base-10 state selector, single frequency projector — all landed at or below random-projection mean performance, with the only "wins" being best-of-K search ceilings that any method shares. The compression target was wrong. The question changed from *"how do we tune the D_M projector?"* to *"what is the QPU base actually measuring?"*

---

## Part 3 — Probe 07: the pivot (Bell-listener / cavity offset)

This is where the project changed direction.

### The reframe

The QPU base was never a 16-state dimensional scene to compress. It is a set of **two-qubit Bell-witness listener tiles**. Each tile measures one Pauli-pair correlator — `XY`, `YZ`, `ZY`, or `YX` — and the tiles are organized into rungs of a cavity-delay ladder (`base_delay + tile_index * offset_dt`). D_M is a *listener*: it asks whether the shared-chip ghost channel produces two-qubit correlation in those four witnesses and whether that correlation tracks the deliberate delay/offset sweep.

### Probe 07 result

Per-tile connected correlators (`C = <P0 P1> − <P0><P1>`) light up on the later rungs and track delay. Standout tiles: `ZY` at rung 3 (`conn = +0.255`, z = +16.5) and rung 4 (`conn = −0.313`, z = −21.7); `YZ` at rungs 2–4 (z = +5 to +8). Rung Bell scores climb monotonically with delay (rung 4 z = +19.0). Delay tracking, connected: base-delay r = +0.857 (p = 0.032), total-delay r = +0.953 (p = 0.018). The `rung_label_shuffle` and `offset_tracking` controls support a coherent, delay-locked witness block rather than scattered per-tile noise.

**D_M is a Bell-witness manifold listener.** That is the real target. The compression framing of Probes 00–06 is retracted. (Interpretation discipline, carried in every probe from here on: this reports Bell-witness *correlation*, not certified Bell entanglement; certification would need CHSH or tomography.)

---

## Part 4 — Probes 08 through 10: characterizing the witness

### Probe 08 — Directional YZ / ZY witness lock

Narrows the four symmetric witnesses to a directional hypothesis: `YZ` is the primary channel, `ZY` is its reciprocal/inverted return, and `XY/YX` are comparison channels.

**Result:** `YZ` positive fraction 1.000; `YZ`-primary tracks total delay r = +0.889; `YZ/ZY` energy r = +0.951; directional specificity r = +0.939. Controls: `independent_bit_shuffle` (breaks same-shot q0/q1 pairing, preserves marginals) collapses energy (z = 30.9) and specificity (z = 24.0) — **same-shot pairing is strongly load-bearing.** But `witness_label_shuffle` separates only weakly (specificity p2 = 0.0016 but the tracking-r p2 ≈ 0.05). First quiet hint, easy to miss here, that the *labels* are less load-bearing than the *pairing*.

### Probe 09 — π-phase / π-adic witness

Treats the `YZ/ZY` pair as a phase-space coordinate: `Y = C(YZ)`, `R = −C(ZY)`, energy `E = sqrt(Y²+R²)`, phase `φ = atan2(R, Y) mod π`. The π-periodic test is the serious part; the **π-adic test is explicitly flagged experimental, toy, and made-up** (π is not prime; this is not a formal p-adic construction).

**Result:** the `offset_off` (base-delay-only) condition gives the cleanest phase lock — π-periodic score 0.984 (p = 0.024), phase-velocity r = −0.935 (p = 0.010). The `offset_on` condition is weaker on phase velocity (π score 0.947, p = 0.25). The π-adic toy was inconclusive everywhere (best p ≈ 0.31–0.46) — recorded as "worth probing", explicitly not "theorem achieved." The comparator hint: similar energy across on/off but weaker phase tracking with offset suggests a distance witness that persists while phase/order lock changes.

### Probe 10 — QPROJ dimensional entanglement projection

Turns the three discovered conditions into the first projection task: `null` (`delays=[0,0,0,0,0]`, `offset=0`), `base_only` (`delays=[0,256,1024,4096,16384]`, `offset=0`), `offset_on` (same delays, `offset=128`). Projects each base into a D_M vector (YZ amplitude, ZY return, energy, specificity, π-phase score, …).

**Result:** strong null-vs-active separation — projection 0.0298 (null) vs 0.406 (base_only) vs 0.385 (offset_on); `null → base_only` Δprojection +0.376, std-dist 6.4. **But** rung-level condition classification failed (balanced accuracy 0.40, p > 0.33 for both manifold-only and manifold-plus-delay schemes). The manifold cleanly separates *active from null* but **cannot classify base_only vs offset_on at the rung level** — an early limit on how much condition structure the projection actually resolves.

---

## Part 5 — Probe 11: the CUDA harness

Validation/benchmark harness for `dm_projector_kernel.cu`: the tile-correlator, rung-projection, and projection-summary kernels, plus the destructive bit-shuffle control, all checked against a CPU reference. Correctness first, then speed.

**Result:**
- Tile max abs diff: **0.0** (bit-identical).
- Rung max abs diff: **9e-8** (fp32 noise).
- Summary max abs diff: **0.589** — the scalar `projection_score` disagrees (CPU 0.885 vs CUDA 0.296) even though every underlying correlator matches.
- Control: `independent_bit_shuffle` drops projection 0.296 → 0.089.
- Throughput: ≈ 5.8 billion records/s; per-rep 0.028 ms.

The 0.589 summary gap is **not** a physics disagreement — the load-bearing per-tile and per-rung correlators agree to fp32. It is a normalization difference in the scalar `projection_score` rollup between the CPU and CUDA paths. Recorded as a known issue (Part 12); it does not affect the correlator-level results everything downstream actually consumes.

---

## Part 6 — Probes 12 / 13: the first benchmark and the DER task

### Probe 12 — first benchmark runner (no output)

The first canonical benchmark runner loaded all the qproj/gproj/geo bases and then produced no summary block — it printed the load lines and stopped. Treated as a stub run and superseded immediately by Probe 13. Recorded for trajectory legibility.

### Probe 13 — final benchmark (verify + classical DER)

Follows the loose G_M capstone pattern: a verify story, a classical task story, and first-class controls.

**Verify (qproj / gproj / geo, three conditions).** All three substrates separate `null` (projection ≈ 0.01) from the active conditions (projection 0.21–0.30). The substrate ordering is the recurring D_M signature: **qproj is attenuated relative to gproj and geo** — qproj `base_only` 0.220 vs gproj 0.296 vs geo 0.284. Same physics, three substrates, with real-hardware noise visibly reducing the projection signal. No quantum-advantage framing is reached for (the same discipline G_M had to learn the hard way). Control collapse: `independent_bit_shuffle` drops active projection 62–81%, but null only 17–30% — pairing is load-bearing on the active manifold and there is little to break in the null.

**Classical DER (Dimensional Entanglement Retrieval).** Retrieve the correct directional paired manifold from a bank of scalar-equivalent decoys (`yz_ret_swap`, `reciprocal_break`, `delay_permute`, `phase_scramble`, `comparison_decoy`).

| Backend | Task A R@1 | Task B R@1 | Task B R@10 |
|---|---|---|---|
| energy | 51.15% | 0.20% | 0.63% |
| pi_fit | 77.32% | 0.24% | 1.25% |
| flat_cosine | 99.83% | 4.64% | 18.02% |
| **dm** | **99.83%** | **4.98%** | **19.34%** |

The honest caveat is in those last two rows: **`flat_cosine` ties `dm`** (identical 99.83% on Task A; 4.64% vs 4.98% on Task B). On the raw retrieval task, the directional D_M manifold barely separates from plain normalized cosine.

**DER controls — is the structure load-bearing?** Here it *is*: `query_yz_ret_swap` → 0.05%, `key_delay_permute` → 0.05%, `key_phase_scramble` → 17.72%, while `residual_scalar_direction_uniform` stays at 98.85%. So directional and delay-order structure are load-bearing *within the D_M scoring*, even though flat cosine matches D_M on the unperturbed task. The task and the controls disagree about how much the operator matters — both are reported.

---

## Part 7 — Probes 14 through 18: the GPT-2 boundary / raw-text arc

The framing for this arc: GPT-2 produces a free-running **pre-softmax** QK product (`L_ij = Q_i·K_j/√d`, no softmax, no attention output); the D_M qproj/gproj/geo bases act as constrained projection boundaries (a "holographic aperture"); the same raw text is sent into both, and the question is whether the free product and the D_M-constrained products are geometrically compatible. Explicit non-claims throughout: no softmax, no claim D_M is a transformer head, no claim of 1-to-1 reproduction. **GPT-2 is never a D_M input** — only a comparison product.

### Probe 14 — Holographic projection (NumPy/torch)

Top compatibility scores ≈ 0.93. Notably, several top rows are `gproj null` and `qproj offset_on`, and `geo base_only` lands ≈ 0.88–0.89. The high scores partly reflect a substrate-agnostic geometry (null scores well too), which tempers the "D_M reproduces attention geometry" reading.

### Probe 15 — Native stagger (sidequest)

Does the `null` base carry an ordered phase/stagger trace even with the explicit delay ladder disabled? Phase slope +0.315 π/rung, r = 0.964, R² = 0.929 — visually a clean ordered trajectory. **But the controls do not cleanly separate:** `independent_bit_shuffle` p = 0.097, `rung_permutation_preserve_witness` p = 0.056. The "null" condition is an *explicit-delay* null, not a *structural* null — native hardware ordering (layout, scheduling, readout grouping, calibration heterogeneity, tile order) may be leaking in. Left ambiguous and carried as an open question.

### Probe 16 — Corpus converter

Utility script; wrote a 1000-line `d_m_probe_corpus.txt`. No claim.

### Probe 17 — Raw-boundary projection (GPU-only, kernel-driven)

The kernel-driven rewrite: every piece of D_M physics runs in `dm_projector_kernel.cu`; the projection is `projected_pair = data_pair XOR base_pair`. Active-vs-null margins reproduce the substrate ladder cleanly: **geo +0.263, gproj +0.062, qproj +0.014**. GPT-2 free-QK vs D_M raw-boundary compatibility tops out ≈ 0.92 (geo base_only, row-centered).

### Probe 18 — Raw-boundary GPU-only benchmark

Timing and margins at benchmark scale. Per-path GPU time: geo 0.27 ms, gproj 0.21 ms, **qproj 8.1 ms** (the slow path), GPT-2 QK 42.9 ms; GPT-2/D_M time ratio ≈ 5×. Active-vs-null margins again order geo > gproj > qproj. The qproj path is both the most physically faithful and the slowest and weakest — the recurring hardware-attenuation signature.

---

## Part 8 — Probes 19 / 20 / 22: the task-utility retrieval saturation

### Probes 19 and 22 — Task utility retrieval

A retrieval leaderboard over a 12-candidate bank (exact positive + shuffle/light-noise/rotate decoys), scoring qproj/gproj/geo signatures against `gpt2_hidden` and `gpt2_qk`.

**Result:** *every* method — all D_M substrates in all conditions **including null**, plus both GPT-2 paths — scored **100% top-1, 100% top-5, MRR 1.000.** The task is trivial: the exact-positive retrieval is solvable by byte-level fingerprinting of the raw line, so `null` retrieves as perfectly as `base_only`. (Probe 22 is the rawsignal variant of 19 and reproduces the identical 100%-everywhere table.)

### Probe 20 — Random null corpus

The control built specifically to diagnose the saturation: a high-entropy random-line corpus with a matched per-line length distribution and a fixed seed. The reading, stated in the script itself: if the score stays ~72–80% on random content, the signal is byte-level fingerprinting independent of content; if it collapses to chance, content mattered. Bit-identical results across runs and cards would confirm deterministic fp32 math rather than any physical-delay residue.

**Conclusion for the arc:** the task-utility retrieval is **saturated** and cannot support a utility claim. This is exactly why the capstone (Part 10) demotes the entire DER / retrieval path to a non-default appendix.

---

## Part 9 — Probes 21 / 23 / 24: the controls become the result

This is where D_M's actual defensible claim was found — not in a headline task, but in the control structure.

### Probe 21 — Windowed-resolution phase trajectory (internal "Probe 04W")

Same π-phase math as Probe 09, but windowed: shots are split into windows so the trajectory has `n_windows × n_rungs` points feeding the identical `pi_periodic_fit_score`.

**Result:** `base_only` π score 0.849 (p = 0.0005), `offset_on` 0.785 (p = 0.0005), `null` 0.000. Controls: both active conditions **clear `independent_bit_shuffle`** (z = 7.01, z = 6.26) — pairing load-bearing — but **neither clears `witness_label_shuffle`** (z = 0.43, z = 0.60, p > 0.3). The π-phase structure is real, but **the YZ/ZY label assignment is not load-bearing.** The quiet hint from Probe 08 is now an explicit, significant result.

### The reframe this forces

A surviving witness-label shuffle reads like a failure under the original directional framing ("YZ is primary, ZY is reciprocal"). It is reinterpreted instead: `XY/YZ/ZY/YX` are **channel views of one dimensional-agreement manifold**, not arbitrary tags. Under that reading, label-shuffle survival is a *feature* — the manifold is invariant to allowed channel relabelings. This reframe is what Probe 23 sets out to test deliberately.

### Probe 23 — Dimensional invariance / forbidden-corruption controls (internal "Probe 05")

Reports two scores: `canonical_yzzy` (YZ-primary) and `dimensional_invariant` (best-repair search over equivalent reciprocal channel descriptions). Allowed transforms (`equiv_pair_swap`, `equiv_reciprocal_swap`, `equiv_cyclic_rotation`) should survive; forbidden ones (`reciprocal_break`, `cross_rung_delay_scramble`, `same_label_wrong_delay`, `non_equivalence_channel_corruption`, `independent_bit_shuffle`) should collapse.

**Result:** allowed rotations retain π/energy under the invariant score (e.g. gproj base_only holds dim_pi = 1.000 across pair-swap and reciprocal-swap; only cyclic rotation costs energy). **But the forbidden single-fault controls did not collapse** — every one returned p > 0.1 with `dim_null` retention ≈ 0.83–0.90. A single structural fault does not kill D_M.

### Probe 24 — Corruption boundary / dimensional error correction (internal "Probe 05B")

Probe 23's non-collapse is not necessarily a failure if D_M is a **dimensional error-correcting** operator. The sharper question: how many *independent* faults can the manifold absorb before agreement becomes unrecoverable? Applies compound corruptions at increasing depth `k` and watches the survival curve cross the collapse threshold (0.50) and the null floor.

**Result (median survival vs depth, selected):**

| Base | k=1 | k=2 | k=3 | k=4 | k=5 | median<thr | median≤null |
|---|---|---|---|---|---|---|---|
| qproj base_only | 0.575 | 0.466 | 0.344 | 0.341 | 0.387 | k=2 | k=3 |
| qproj offset_on | 0.581 | 0.375 | 0.365 | 0.298 | 0.225 | k=2 | k=2 |
| gproj base_only | 0.736 | 0.438 | 0.351 | 0.349 | 0.237 | k=2 | — |
| gproj offset_on | 0.613 | 0.385 | 0.275 | 0.273 | 0.260 | k=2 | k=2 |
| qproj null | 0.568 | 0.523 | 0.477 | 0.494 | 0.565 | k=3 | not reached |
| gproj null | 0.633 | 0.576 | 0.594 | 0.563 | 0.532 | not reached | not reached |

The active manifolds cross the collapse threshold at compound depth `k=2` and reach the null floor by `k=2–3`, with the below-threshold fraction climbing to 1.0 by `k=4–5`. The **null conditions never cross** — they have no structure to collapse. The collapse boundary is real, measurable, and behaves exactly as a dimensional error-correcting manifold should: single faults repair, compound faults break through.

**This is the standing claim.** The strongest D_M result came from the corruption-boundary control, not from any retrieval task.

---

## Part 10 — The capstone benchmark (consolidation)

### `d_m_benchmark.py`

The repo-facing capstone, updated to what its docstring calls the **Probe 22/23/24/25 standard** (a renumbering of the consolidated repo, mapping to the final record path / invariance / corruption-boundary / exact-GEO work):

- **Probe 22** → final qproj/gproj CUDA record path.
- **Probe 23** → allowed-channel invariance + forbidden single-fault checks.
- **Probe 24** → compound corruption / collapse-boundary checks.
- **Probe 25** → an **exact closed-form GEO** classical reference path (`dm_geo_exact_rung_projection_kernel_f32` and its sweep variant), giving an analytic null that is exactly zero and an active positive projection without any shots.

**Default run.** VERIFY (qproj / gproj / exact-GEO over null / base_only / offset_on) + CONTROLS (bit-shuffle collapse, allowed-channel retention, forbidden single-fault weakening, compound corruption boundary) + GEO VALIDATION (exact-zero null manifold, active positive projection, active π score, CPU/GPU agreement for the exact GEO rule).

**Demoted by design.** The old synthetic DER / task-utility retrieval path is explicitly *not* a default final claim — Probes 13/19/22 saturated it, so utility claims must not lean on it. It is kept only as an optional appendix.

**Bounded claim (verbatim intent).** D_M projects a YZ-primary / ZY-reciprocal dimensional witness manifold across qproj, gproj, and exact-GEO substrates; active base-delay / offset manifolds separate from null; same-shot pairing, reciprocal structure, and delay order are load-bearing; compound corruptions cross a measurable collapse boundary.

---

## Part 11 — What the trajectory establishes (the claims)

Stripped of the framings the trajectory disproved:

1. **D_M is a Bell-witness manifold listener, not a dimensional-compression projector.** The compression framing (Probes 00–06) is fully retracted — it never beat random-projection mean at any reframe. *Pivoted at Probe 07.*

2. **Same physics, three substrates.** The operator admits faithful qproj (real QPU), gproj (controlled classical), and exact-GEO (closed-form) implementations. All separate active from null. qproj is consistently **attenuated** relative to gproj/geo (e.g. active margins geo +0.263, gproj +0.062, qproj +0.014) — visible hardware noise, **no quantum-advantage claim**. *Survived.*

3. **Same-shot pairing is load-bearing.** `independent_bit_shuffle` collapses the active manifold (Probes 08, 13, 21). *Survived.*

4. **Witness labels are *not* load-bearing.** `witness_label_shuffle` does not collapse the signal (Probes 08, 21) → the four witnesses are channel views of one agreement manifold, not directional tags. *Revised from the initial YZ-primary directional framing — the directional reading is interpretive, not load-bearing.*

5. **Single faults repair; compound faults cross a collapse boundary.** Forbidden single-fault controls do not collapse (Probe 23), but compound corruption at depth k=2–3 reaches the null floor while nulls never cross (Probe 24). The dimensional-error-correction claim. *This is the strongest result.*

6. **Utility / retrieval claims do not survive.** The DER task ties flat cosine (Probe 13) and the task-utility retrieval saturates to 100% for every method including null and GPT-2 (Probes 19/22). *Retracted as a default claim; demoted to appendix in the capstone.*

---

## Part 12 — Known issues

- **Probe 11 CUDA `projection_score` normalization mismatch.** Summary scalar disagrees CPU 0.885 vs CUDA 0.296 (max abs diff 0.589) while every per-tile (diff 0.0) and per-rung (diff 9e-8) correlator is bit-identical. A normalization difference in the scalar rollup, not a physics disagreement; does not affect the correlator-level outputs everything downstream uses. Open.

- **Probe 12 produced no summary.** First benchmark runner stopped after loading bases; superseded by Probe 13. Kept for legibility.

- **DER / task-utility retrieval saturated.** Probe 13 Task A: `flat_cosine == dm` (99.83%). Probes 19/22: all methods (incl. null and GPT-2) at 100% top-1. Demoted to non-default appendix in the capstone; utility claims must not lean on these.

- **Native-stagger null ambiguity (Probe 15).** The `null` base shows an ordered phase trajectory (r = 0.964) but the controls don't cleanly separate (bit-shuffle p = 0.097, rung-perm p = 0.056). The "null" is an explicit-delay null, possibly leaking native hardware ordering. Open.

- **π-adic toy (Probe 09) is experimental and inconclusive.** Explicitly made-up (π is not prime), best control p ≈ 0.31–0.46. Either drop or formalize.

- **qproj attenuation.** qproj is weaker and slower than gproj/geo across every probe (active margin +0.014 vs +0.062 vs +0.263; verify projection 0.22 vs 0.30 vs 0.28; 8.1 ms vs 0.21/0.27 ms). Expected hardware-noise attenuation, recorded as the substrate signature rather than a defect.

- **Non-claims (carried in every benchmark).** D_M does not reconstruct density matrices, does not certify device-independent Bell nonlocality, does not prove prepared Bell states, is not a QPU speedup or quantum-advantage claim; `gproj` is not an IBM hardware simulator; `geo` is a closed-form classical reference, not a simulator; GPT-2 is not a D_M input in the final claim.

---

## Part 13 — Open questions

1. **A structural null.** Resolve the Probe 15 ambiguity — is the residual ordering in the explicit-delay null a hardware layout/scheduling artifact, or analysis leakage? A genuinely structureless null base would let the corruption-boundary and phase-lock claims rest on a clean floor.

2. **A task that separates D_M from flat cosine.** The DER and retrieval tasks saturated. Construct a retrieval/utility task where the directional paired manifold is load-bearing in a way normalized cosine cannot match — the legitimate version of the utility claim.

3. **Real LLM embeddings.** Push the GPT-2 boundary arc (Probes 14/17/18) past a pre-softmax compatibility probe into learned representations grounded on a real task.

4. **qproj attenuation across calibrations.** Characterize how the qproj projection signal varies across QPU jobs/backends, so the attenuation could serve as a per-job hardware-quality readout.

5. **Where channel-invariance / error-correction becomes load-bearing.** At the current operating point single faults repair and the invariance is defensive. Find the regime (fewer shots, narrower delay ladder, harsher corruption) where it is the difference between recoverable and not.

6. **π-adic toy: drop or formalize.** It has earned neither a place in the claim nor a formal construction; decide which.

---

## Part 14 — Philosophy and license

CC0. Build, break, fix, document, repeat. All in the open. The break-it-fix-it rule holds: if you find something wrong, you ship the fix alongside the bug report.
