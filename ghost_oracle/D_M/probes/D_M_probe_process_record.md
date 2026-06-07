# D_M (Dimensional Entanglement Projection) — Probe Suite Process Record

*Ghost Oracle Suite · D_M operator · reconstructed from probe docstrings, run logs, and the Probe 25 GEO cleanup pass*

This document walks the D_M investigation from start to finish. For each step it states (a) what the file's own docstring says the probe is *for*, and (b) what the run actually *did* and *found* according to `proberesults.txt`. The main run log captures a session dated `20260606` against backend `ibm_marrakesh`; the final cleanup notes add the `20260607` Probe 25 GEO precision-reference pass.

---

## 0. The shape of the whole thing

The suite has two distinct halves, and the run log makes the seam between them very visible.

**Half A — Probes 00→06: the "dimensional compression" rehearsal (a dead end that taught the real question).** These treat D_M as a candidate dimensionality-reduction operator and benchmark it against PCA / random projection on synthetic datasets. They all run against the **null** QPU base (`d8fm4ihvjngc73aq3ccg`: no delay, no offset). D_M channels never decisively beat well-tuned random projection. This half ends without a win.

**Half B — Probes 07→24: the Bell-witness pivot and hardening arc.** Probe 07 reframes D_M entirely: not a compressor, but a *bare two-qubit Bell-witness listener* that reads four Pauli-pair correlators (XY, YZ, ZY, YX) and asks whether they track a cavity-delay ladder. This is where structure actually appears, and the rest of the suite narrows, hardens, benchmarks, and stress-tests that finding.

**Half C — Probe 25: the GEO cleanup/reference path.** Probe 25 isolates GEO from the older text/aperture plumbing and turns it into the thing the final benchmark needs: a closed-form classical reference representation of the D_M manifold. It uses no raw text, no shots, no qproj/gproj records, no GPT-2, and no retrieval task. Its job is to define the exact classical GEO rule that the final official benchmark should import.

The three canonical QPU conditions (used everywhere in Half B) are:

| Condition | Job ID | base delays (dt) | offset (dt) |
|---|---|---|---|
| `null` | `d8fm4ihvjngc73aq3ccg` | `[0,0,0,0,0]` | 0 |
| `base_only` | `d8flk2jo3njc73f0g560` | `[0,256,1024,4096,16384]` | 0 |
| `offset_on` | `d8fl82bo3njc73f0fgd0` | `[0,256,1024,4096,16384]` | 128 |

The discovered orientation that the whole back half rests on:

> **YZ = primary witness dimension · ZY = reciprocal / inverted witness dimension · XY, YX = comparison dimensions.**
> Rung coordinates: `Y=connected(YZ)`, `R=-connected(ZY)`, `E=√(Y²+R²)`, `S=E-√(XY²+YX²)`, `φ=atan2(R,Y) mod π`, where `connected(PQ)=⟨P0P1⟩-⟨P0⟩⟨P1⟩`.

---

## 1. Data generators (not in the run log, but the source of everything)

### `d_m_qpu_generate.py` — QPU submit + dump
**Docstring purpose:** Unified IBM Runtime workflow (`submit` then `dump <JOB_ID>`). Writes metadata first, then freezes a canonical qproj `.npz` base. D_M is explicitly *not* preparing a Bell state, reconstructing a density matrix, or using ancillas/dynamical decoupling — it places bare coherent two-qubit listener tiles (`H(q0),H(q1)` → delay → rotate into witness basis → measure), lets neighbouring tiles share silicon, sweeps cavity-delay offsets, and reads the four witness correlators.
**Cross-reference:** This is the producer of the three `dm_data_bell_listener_cavity_offset_*.npz` files consumed throughout Half B.

### `d_m_gpu_generate.py` — GPU / synthetic base generator (gproj)
**Docstring purpose:** Produces a local GPU-generated base with the *same schema* as the QPU dump, for development without QPU jobs, controlled fixtures, kernel validation, and benchmark plumbing. For each tile it samples two-bit shot records targeting a chosen connected correlator. Explicitly **not** an IBM hardware simulator — a controlled witness-manifold generator. Supports the same three conditions (`null`, `base_delay`, `offset_deformed`).
**Cross-reference:** Source of the `dm_gpu_data_*_4096shots_seed*.npz` files that appear as the `gproj` substrate in probes 12–24.

---

## 2. Half A — the dimensional-compression rehearsal (Probes 00→06)

### Probe 00 — `00_dm_probe_prune.py` · Prune failed assumptions
**Purpose (docstring):** Load one frozen QPU scene and prune the eight embedded assumptions (`smooth_walk`, `boundary_reflect`, `nonlocal_jump`, `collapse_gate`, `phase_shear`, `scramble_order`, `mirror_parity`, `rank_spread`). Score each tile with simple effect-size tests, classify as SURVIVE/WEAK/FAIL/MUTATED, and prune. Explicitly a "first-pass pruning blade," not final statistics.
**Run result:** 20 tiles, all classified `vote=MUTATED · class=UNKNOWN_STRONG_STRUCTURE` with enormous z-scores (z≈2240–5230), eff alternating ≈2.0/≈4.0. Single mode decision: `unknown → KEEP_AS_MUTATION`, `repeat=inconsistent`, `eff_mean=3.023`. So the prune blade kept everything as a mutation candidate rather than cleanly resolving any named assumption — the structure was strong but unidentified.

### Probe 01 — `01_dm_probe_channel_extract.py` · Channel extraction
**Purpose:** Turn the prune report into self-contained candidate channels: `local_order`, `collapse`, `mutation`, `symmetry_boundary`, `rank_spread`, and a `composite_dm`. Each exports 16-state weights, 4-bit weights, scalar summary, projection recipe. No benchmarking yet.
**Run result:** Five named channels came back `class=EMPTY` (eff=16.000, entropy=4.000, maxp=0.062 — i.e. uniform/no carry), and only `composite_dm_channel` came back `COMPOSITE_CANDIDATE` (carry=1, all 20 tiles, eff=3.577, entropy=1.910). Translation: on the null base, only the aggregate channel carried anything.

### Probe 02 — `02_dm_probe_synthetic_maps.py` · Synthetic maps (first task rehearsal)
**Purpose:** Test whether Probe-01 channels guide structure-preserving compression (neighbor overlap, trustworthiness, label purity, rank correlation) on `blobs/rings/swiss_roll_like/s_curve_like/sparse_binary` vs identity/random/PCA baselines.
**Run result — first verdict against baselines:** Overall ranking led by **`pca_projection` (overlap 0.7079)**. Best D_M entry was `dm_composite_dm_channel` at 0.3917 — second place, but well behind PCA and only marginally ahead of plain random projection (0.3607). The docstring of the *next* probe names this outcome explicitly as the failure that exposed a missing dimension.

### Probe 03 — `03_dm_probe_linear_scheduler.py` · Linear-activated channel scheduler
**Purpose:** React to Probe 02 losing to PCA/random by (a) promoting the *linear direction* to internal channel 0 (D_M now has 7 channels), and (b) replacing the single weak random baseline with a hardened suite (gaussian/orthogonal/sparse-achlioptas × mean/best/worst, plus shuffle/sign-flip crops). Uses a secrets-generated 128-bit master seed.
**Run result:** The hardened controls dominate. Overall top three were `random_sparse_achlioptas_best_128 / random_orthogonal_best_128 / random_gaussian_best_128` (≈0.51). Best D_M was `dm7_soft_scheduler` at 6th (0.3649), beating PCA (18th, 0.2274) but losing to every "best_N" random ceiling and to the crop baselines. Lesson recorded in Probe 04's docstring: *random/secret seed projection is not just a baseline, it is a strong channel.*

### Probe 04 — `04_dm_probe_calibrated_seed_projection.py` · Calibrated seed projection
**Purpose:** Test the corrected hypothesis that the 7th channel is a *calibrated seed* channel (QPU-derived deterministic projection matrices), with the honest win condition stated as `qpu_calibrated_seed_single > random_*_mean_N` (the `best_N` rows are explicitly diagnostic ceilings, not fair baselines).
**Run result:** `qpu_calibrated_seed_family_best_32` reached 4th overall (0.7572), but among the *fair single-shot* comparison the calibrated seed did **not** clear the bar: `qpu_calibrated_seed_single` finished 18th (0.6678), below `random_gaussian_mean_128` (0.6724) and the dm_static channels. The honest win condition was not met.

### Probe 05 — `05_dm_probe_base10_state_selector.py` · Base-10 calibrated state selector
**Purpose:** Reframe calibration as a base-10 alpha-ladder / state selector that picks dimensional basis states directly, rather than seeding a random matrix.
**Run result:** Top of the board is `base10_uncalibrated_dominant_family_best_128` (0.7776) — i.e. the **un**calibrated variant edged out the QPU-calibrated one (`base10_qpu_calibrated_dominant_family_best_128`, 0.7753). Calibration provided no clear advantage over its own uncalibrated control. Still a "best_N" ceiling result, not a fair single-shot win.

### Probe 06 — `06_dm_probe_single_calibrated_projector.py` · Frequency base-10 seed projector
**Purpose:** Strip everything down to **one** projector (`dm_frequency_base10_seed_projector`) — no channels, no scheduler, no family-best, no variant search — with the seed folded in as an active per-sample dimensional channel. Fair comparison is vs `random_*_mean_N`, `random_gaussian_single`, and PCA.
**Run result:** The single projector finished **10th of 14** (0.6018), below all random mean/single baselines and only ahead of worst-case ceilings and PCA. This is the clean end of Half A: as a standalone projector on the null base, D_M does not beat random projection.

> **Half A summary:** Across 00→06, on the null QPU scene, D_M framed as a compressor/projector never produced a fair-comparison win over PCA or mean random projection. The recurring runner-up was always the *aggregate/composite* form, and the recurring lesson was that random projection is a genuinely strong channel. The investigation needed a different question.

---

## 3. The pivot — Probe 07 (Bell listener)

### Probe 07 — `07_dm_probe_bell_listener.py` · Bell listener / cavity offset
**Purpose:** Treat D_M not as rank/intrinsic-dimension but as a *bare shared-chip Bell-witness listener*. Each 2-qubit tile measures one Pauli-pair witness (XY/YZ/ZY/YX); the experiment listens for two-qubit correlation and whether it tracks the cavity-delay ladder. Discipline is explicit: this reports Bell-witness *correlation*, not certified entanglement. Runs destructive controls (independent bit shuffle, rung-label shuffle, offset-tracking permutation).
**Run result (offset_on base, 20 tiles / 4096 shots):** Structure appears and it **tracks delay**:
- Per-rung connected RMS climbs monotonically with base delay: rung 0 → 4 conn_rms = 0.0057 → 0.0219 → 0.0592 → 0.1336 → 0.1639.
- `base_delay connected tracking r=+0.857, p=0.0316`; `total_delay connected tracking r=+0.953, p=0.0178`.
- Late rungs are dominated by the YZ/ZY axis (rung 3 ZY=+0.247, rung 4 ZY=-0.334), foreshadowing the directional finding.

This is the first real signal in the whole suite. Everything downstream builds on it.

---

## 4. Narrowing the witness — Probes 08→10

### Probe 08 — `08_dm_probe_directional_witness.py` · Directional YZ/ZY lock
**Purpose:** Stop treating the four witnesses as symmetric peers. Test the specific hypothesis that the channel is *directional*: YZ primary, ZY the reciprocal/return that may invert as delay grows. Controls: independent bit shuffle, witness-label shuffle, delay-order permutation.
**Run result:** Directional structure confirmed and delay-locked:
- `YZ positive fraction = 1.000`; YZ/ZY energy and specificity rise with delay.
- `total_delay YZ/ZY energy r=+0.951 (p=0.018)`, `specificity r=+0.939 (p=0.015)`.
- **Controls behave correctly:** `independent_bit_shuffle` collapses energy/specificity (z≈31 / z≈24, p≈0.0002) — the same-shot pairing is load-bearing. `witness_label_shuffle` is weaker/ambiguous (p2 up to ~0.20), an early hint that the YZ/ZY labels are *views*, not arbitrary tags — a theme Probes 21/23/24 return to.
- ZY inverts vs YZ on 40% of rungs (the reciprocal/return behaviour).

### Probe 09 — `09_dm_probe_pi_phase_witness.py` · π-phase / π-adic witness
**Purpose:** Treat YZ/ZY as a phase-space coordinate (`E=√(Y²+R²)`, `φ=atan2(R,Y) mod π`) and ask whether the witness forms a coherent π-periodic phase trajectory over delay. The π-*adic* part is flagged in the docstring as deliberately experimental/toy (π is not prime; not a formal p-adic construction). Compares offset-on vs offset-off.
**Run result:**
- `offset_off` (= base_only, job `d8flk2j…`): strong π-periodic fit — `π-periodic score 0.9841, p=0.0240 (linear)`, energy tracking r=+0.979, **phase velocity r=-0.935, p=0.010**.
- `offset_on` (job `d8fl82b…`): π-periodic score 0.9470, p=0.2509 (log), but phase velocity flat (r=-0.104, p=0.858).
- Comparator: turning the offset on *raises* phase-velocity tracking delta (+0.831) while slightly lowering π-periodic and energy tracking — the log hint reads: a distance/geometry witness persists while the phase/order lock changes character. The toy π-adic scores never reach significance (best p≈0.31–0.46), consistent with the docstring's "worth probing, not theorem" caveat.

### Probe 10 — `10_dm_probe_qproj_dimensional_entanglement.py` · QPROJ dimensional-entanglement projection (task-lock)
**Purpose:** Turn the three conditions into the first benchmark task — project each base into a D_M entanglement vector and test condition separation (null vs base vs offset) plus controls.
**Run result — clean condition separation:**
| Condition | YZ mean | energy mean | π score (p) | projection |
|---|---|---|---|---|
| null | -0.0119 (pos_frac 0.40) | 0.0224 | 0.000 (1.000) | 0.0298 |
| base_only | +0.0794 (1.00) | 0.1560 | 0.9841 (0.0065) | 0.4056 |
| offset_on | +0.0570 (1.00) | 0.1525 | 0.9470 (0.0145) | 0.3848 |

- `null → base_only` Δprojection +0.376 (std_dist 6.38); `null → offset_on` +0.355 (std_dist 5.00); `base_only ↔ offset_on` differ only slightly (Δ-0.021). So **active-vs-null separation is large and significant; base-vs-offset separation is subtle.**
- Caveat the log surfaces: the rung-level 5-way condition *classification* was at chance (bal_acc 0.400, p≈0.33–0.37) — separating the *manifolds* works far better than classifying individual rungs.

---

## 5. Kernels and benchmarks — Probes 11→13

### Probe 11 — `11_dm_probe_cuda_qproj_harness.py` · CUDA qproj harness
**Purpose:** Validate `dm_projector_kernel.cu` (tile correlator → rung projection → summary, plus a bit-shuffle control) against a CPU reference. Correctness first, then speed.
**Run result (gproj base_delay base, RTX 3090 / cupy):**
- **Tile and rung match to numerical noise:** tile max abs diff `0.0`, rung max abs diff `9e-8`. The physics kernels are correct.
- **Summary differs by 0.589** (CPU projection_score 0.8848 vs CUDA 0.2955) — a score-normalization/aggregation difference at the summary stage, *not* a physics mismatch (the underlying yz_mean, energy, specificity, π-score are identical: 0.1226 / 0.2470 / 0.2171 / 1.000 on both). Worth flagging as a known reconciliation item.
- Control collapses correctly: `independent_bit_shuffle` drops projection 0.2955 → 0.0891. Throughput ≈5.8 billion records/s.

### Probe 12 — `12_dm_probe_benchmark.py` · Canonical benchmark ⚠️ **incomplete in this run**
**Purpose:** Canonical runner comparing the three substrates (`qproj` real QPU, `gproj` GPU, `geo` analytic) across the three conditions, with four tasks (condition separation, active-manifold projection, substrate agreement, control collapse) and explicit non-claims.
**Run result:** The log shows the header, the data-dir/kernel/GEO config, and the six `[QPROJ]/[GPROJ]` loads plus `[GEO] null: analytic manifold` — and then **stops**, jumping straight to the Probe 13 invocation. No summary, no completion banner. **Probe 12 did not finish in this session** (apparent abort/interrupt right after loading bases). Its role was effectively superseded by Probe 13.

### Probe 13 — `13_dm_probe_benchmark2.py` · Final benchmark (verify + classical DER)
**Purpose:** Capstone in the G_M pattern — (1) VERIFY condition projection across qproj/gproj/geo; (2) CLASSICAL TASK: Dimensional Entanglement Retrieval (DER), retrieve the correct directional paired manifold from a bank of scalar-equivalent decoys (yz/return swap, reciprocal break, delay permute, phase scramble, comparison decoy); (3) controls as first-class. Bounded claim only.
**Run result — this is the substantive benchmark of the suite:**

*Verify summary (projection by substrate × condition):* all three substrates separate null from active and agree qualitatively.
- geo: base_only 0.2845 / offset_on 0.2584 / null 0.0036
- gproj: base_only 0.2955 / offset_on 0.2982 / null 0.0103
- qproj: base_only 0.2200 / offset_on 0.2087 / null 0.0128
- π-fit ≈1.0 on active conditions for geo/gproj; qproj weaker (0.93 / 0.79). YZ-positive fraction = 1.00 on every active condition, 0.40–0.80 on null.

*Control collapse (independent_bit_shuffle):* active conditions drop hard — qproj base_only -62%, offset_on -52%; gproj base_only -70%, offset_on -81%; nulls barely move (-17 to -30%). Same-shot pairing is load-bearing.

*Classical DER — Task A (local hard-negative, true vs matched scalar-equivalent decoys):*
| method | R@1 | MRR |
|---|---|---|
| energy | 51.15% | 0.740 |
| pi_fit | 77.32% | 0.886 |
| flat_cosine | 99.83% | 0.999 |
| **dm** | **99.83%** | **0.999** |

*Task B (global bank stress, 300k candidates):* everything is hard; `dm` leads narrowly (R@1 4.98%, R@10 19.34%) just above flat_cosine (4.64% / 18.02%), with energy/pi_fit near zero.

*DER controls — is D_M structure load-bearing?* This is the cleanest evidence in the suite:
| control | R@1 | reading |
|---|---|---|
| dm (intact) | 99.83% | baseline |
| query_yz_ret_swap | 0.05% | swapping primary/return **destroys** it |
| key_delay_permute | 0.05% | breaking delay order **destroys** it |
| key_phase_scramble | 17.72% | phase matters, partially |
| residual_scalar_direction_uniform | 98.85% | scalar direction is **not** required |

So directional pairing + delay order are load-bearing; raw scalar energy/direction is not. The log signs off with the project motto: *"Done. Break it, fix it, document what happened."*

---

## 6. GPT-2 compatibility & retrieval utility — Probes 14, 17, 18, 19, 22

A shared "hard rule" runs through these: **GPT-2 is never D_M input.** The same raw text is sent to independent products; GPT-2 contributes only a free-running *pre-softmax* QK product `L_ij = Q_i·K_j/√d` (no softmax, no masking, no attention output).

Important cleanup note: the `geo` path in Probes 14/17/18/19/22 is the **older raw-text / analytic-aperture GEO plumbing**, not the final reference GEO. Probe 25 exists because that older GEO was useful for architecture tests but not mathematically clean enough to serve as the official classical representation.

### Probe 14 — `14_dm_probe_holographic_projection.py` · Holographic projection
**Purpose:** Ask whether GPT-2's free pre-softmax QK geometry and D_M-constrained (qproj/gproj/geo "aperture") products are geometrically coherent when the same text is mapped through both. Bounded: not claiming D_M is a transformer head.
**Run result:** 6 text sequences → 432 manifolds. Top projection scores reach **0.93** (`layer 2 head 4`, gproj null, align +0.973). Many high scorers sit on `geo base_only` and `qproj offset_on` (align up to +0.997). Strong geometric compatibility exists — though note the top score lands on a *null* boundary, so the score reflects geometric alignment, not active-manifold specificity by itself.

### Probe 17 — `17_dm_probe_raw_boundary_projection.py` · Raw-boundary projection (GPU-only, kernel-driven)
**Purpose:** Kernel-driven rewrite — every piece of D_M physics runs in CUDA (`dm_probe_kernels.cu`); only bit-bank construction and tiny reporting math stay on host. `projected_pair = data_pair XOR base_pair`.
**Run result:** Active-vs-null margins are **ordered by substrate** exactly as the analytic path predicts: `geo` margin +0.263 ≫ `gproj` +0.062 ≫ `qproj` +0.014. Record path ≈164M shot-pairs/s. GPT-2/D_M compatibility tops out at **0.919** (layer 0 head 10, geo base_only, ecos +0.958). The analytic geo aperture shows the cleanest active separation; real qproj is the noisiest, as expected.

### Probe 18 — `18_dm_probe_raw_boundary_gpu_benchmark.py` · Raw-boundary GPU benchmark
**Purpose:** Full GPU-only benchmark of the corrected architecture with timers; no CPU fallback (raises if CUDA/CuPy/Torch-CUDA missing).
**Run result:** Active scores beat null across substrates (e.g. gproj base_only 0.1073 vs null 0.0114; geo base_only 0.1093 vs null 0.0060; qproj base_only 0.0738 vs null 0.0089). Timing: D_M gproj ≈1.19B items/s, geo ≈0.91B, qproj slower (≈30M, dominated by a 7.9 ms first-call on the null base). **GPT-2 QK path is ≈4.99× slower than the full D_M path.** Two `RuntimeWarning`s (invalid value in subtract / scalar divide) are logged — NaN guards worth noting but not fatal.

### Probes 19 & 22 — `*_task_utility_retrieval_gpu.py` / `*_rawsignal.py` · Task-utility retrieval ⚠️ **non-discriminating as run**
**Purpose:** Same GPU-only architecture pointed at a small retrieval task (queries vs candidate bank with shuffle/light-noise/rotate decoys), comparing D_M signatures (dim 62) against GPT-2 hidden (768) and GPT-2 QK (576). *(Probes 19 and 22 are near-identical scripts — 22 is the "rawsignal" variant; both produced the same leaderboard.)*
**Run result:** With only **3 queries / 12 candidates / exact positives**, *every* method scored **top1=100% top5=100% MRR=1.0000** — D_M (all conditions, including null), GPT-2 hidden, and GPT-2 QK alike. This is a saturated, non-discriminating result: the task is too easy to separate methods, and the docstring of Probe 20 spells out exactly why this is a trap. D_M's value here is efficiency (encode ≈1 ms / dim 62) vs GPT-2 (≈255 ms / dim 768), not accuracy. **Treat 19/22 as plumbing validation, not a utility claim.**

### Probe 20 — `20_dm_probe_make_random_null_corpus.py` · Random-text null control (utility)
**Purpose:** Generate a high-entropy random corpus with matched per-line length distribution and a fixed seed, to detect whether retrieval "success" is just byte-level fingerprinting (score stays high on random text ⇒ fingerprinting, not the card; score collapses ⇒ content mattered). Bit-identical reruns ⇒ deterministic fp32 math, the opposite of a physical-delay residue.
**Run result:** Wrote 5000 random lines, length min=max=mean=120. This is the control corpus *built to debunk* the 100%-everywhere result of 19/22 — its existence is the project acknowledging the 19/22 confound. (The log does not show the benchmark re-run against this corpus in this session.)

### Probe 16 — `16_dm_probe_convert_corpus.py` · Corpus converter (utility, no docstring)
**Purpose (from code):** Strip Project-Gutenberg header/footer, normalise whitespace, pack ~3 sentences per line → `d_m_probe_corpus.txt`.
**Run result:** "Wrote 1000 lines." Feeds the text path used by 17/18/19/22.

---

## 7. Hardening the structure — Probes 15, 21, 23, 24

### Probe 15 — `15_dm_probe_native_stagger.py` · Native stagger
**Purpose:** A focused sidequest: does the **null** base contain an ordered D_M-like phase/stagger trace even with the explicit delay ladder disabled (from hardware layout, scheduling, readout grouping, calibration heterogeneity, pulse/tile order)? No GPT-2, no projection — interrogates the base file itself.
**Run result:** On the null base, the explicit delays are all zero, yet a phase order appears: **phase slope +0.315 π/rung, r=+0.964, R²=0.929**, with YZ/ZY:comparison energy ratio 8.37×. **But the controls don't clear it:** independent_bit_shuffle phase|r| p_upper=0.097, rung-permutation p_upper=0.056–0.068 — none below 0.05. No calibration-field correlations found. Verdict: a suggestive native ordering, **not** a control-clearing signal. Honest negative/ambiguous result on the null.

### Probe 21 — `21_dm_probe_04w_windowed_phase_trajectory.py` · Windowed phase trajectory (04W)
**Purpose:** Higher-resolution Probe-04-style phase trajectory — identical projection math, but shots are windowed so the trajectory has (windows × rungs) points feeding the same π-periodic fit + permutation controls (independent_bit_shuffle, witness_label_shuffle).
**Run result (window 512 → 8 windows, 40 points):**
| condition | π score (p) | bit-shuffle control | label-shuffle control |
|---|---|---|---|
| null | 0.0000 (1.000) | z=0.00 p=1.000 | z=0.00 p=1.000 |
| base_only | 0.8487 (0.0005) | **z=+7.01 p=0.0040 ✓ clears** | z=+0.43 p=0.4104 ✗ |
| offset_on | 0.7850 (0.0005) | **z=+6.26 p=0.0040 ✓ clears** | z=+0.60 p=0.3386 ✗ |

Key reading: the active phase trajectory **survives the bit-shuffle control** (real same-shot structure) but the **witness-label shuffle does *not* kill it**. Per the suite's framing that's a *feature* — XY/YZ/ZY/YX are channel views of one manifold, not arbitrary labels — and it sets up the next two probes.

### Probe 23 — `23_dm_probe_dimensional_invariance_controls.py` · Dimensional invariance / forbidden-corruption controls
**Purpose:** Test the claim that *allowed* channel transformations preserve D_M while *forbidden* ones collapse it. Reports two scores: `canonical_yzzy` and `dimensional_invariant` (best over equivalent reciprocal descriptions). Allowed: pair-swap, reciprocal-swap, cyclic rotation. Forbidden: reciprocal break, cross-rung delay scramble, same-label-wrong-delay, non-equivalence corruption, independent bit shuffle.
**Run result:** Allowed transforms largely preserve the score (the `dim_pi` invariant stays at the observed value under pair-swap and reciprocal-swap across all active conditions; cyclic rotation sometimes degrades, e.g. qproj base_only dim_pi 0.9264 → 0.5421). **Forbidden single-fault corruptions mostly do *not* collapse the manifold** — every forbidden control on the active bases returns p > 0.05 (e.g. qproj base_only reciprocal_break p=0.10, independent_bit_shuffle p=0.24). The null bases stay at 0.0000 throughout. This *single-fault* robustness is the explicit motivation for Probe 24.

### Probe 24 — `24_dm_probe_corruption_boundary.py` · Dimensional error-correction / corruption boundary
**Purpose:** Reframe Probe 23's non-collapse: if D_M is a dimensional *error-correcting* operator, single faults *should* often be repairable. So apply **compound** corruptions at increasing depth k and find where dimensional agreement becomes unrecoverable.
**Run result (trials/depth 500, collapse threshold 0.5, preserve 0.75):** A clear depth-dependent collapse on every active base:
- **qproj base_only:** allowed identity 0.995; median crosses below threshold at **k=2**, drops to/below null floor at **k=3** (k=5 mean 0.387).
- **qproj offset_on:** median below threshold at **k=2**, ≤null at **k=2** (k=5 mean 0.225).
- **gproj base_only / offset_on:** median below threshold at **k=2**, k=5 means 0.237 / 0.260.
- **null bases:** never cross (qproj null median<thr never reached; gproj null stays ≈0.53–0.64 throughout) — there is no structure to break.

The worst-case corruption combinations at every depth are dominated by `independent_bit_shuffle`, `non_equivalence_channel_corruption`, `reciprocal_break`, and delay scrambles — exactly the same-shot / reciprocal / delay-order structure that Probe 13's DER controls flagged as load-bearing. The picture is consistent: **single structural faults are absorbed; ~2–3 independent faults cross the collapse boundary on active manifolds, while nulls have nothing to collapse.**

---

---

## 8. GEO cleanup — Probe 25

### Probe 25 — `25_dm_probe_geo_precision_reference.py` · GEO precision reference
**Purpose:** Isolate GEO as a closed-form classical reference for D_M before the final official benchmark. This probe deliberately removes every dependency that made the old GEO path muddy: no raw text, no shot sampling, no qproj/gproj base records, no GPT-2, no retrieval scoring, and no random controls. GEO is now a deterministic mathematical manifold generated directly from condition metadata.

**GEO rule:** For active conditions, Probe 25 computes separable spatial and temporal coordinates and folds them into one D_M coordinate:

```text
x_space = normalize(log1p(base_delay))
x_time  = normalize(log1p(base_delay + mean_offset))
x_dm    = sqrt((w_space*x_space^2 + w_time*x_time^2)/(w_space+w_time))
cos(2φ) = 2*x_time - 1
YZ      = E*cos(φ)
ZY      = -E*sin(φ)
E       = energy_floor + energy_scale*x_dm^energy_gamma
```

The null condition is an exact zero manifold. XY and YX are zero comparison axes, so `S = E - sqrt(XY² + YX²)` reduces to `S = E` for this clean reference.

**Built-in kernel:** Probe 25 carries its own CUDA source for the GEO reference path:
- `dm_geo_exact_rung_kernel_f32`
- `dm_projection_summary_kernel_f32`

The CPU path is float64 and the CUDA path is float32. The probe writes rung rows, summary rows, CPU/GPU agreement, validation checks, and per-metric summary deltas.

**Run result / cleanup result:** The closed-form GEO path produces the intended active/null structure:

| condition | backend | projection | E mean | S mean | π score | phase span π | elapsed ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `null` | CPU | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.167400 |
| `base_only` | CPU | 0.693826892 | 0.675089952 | 0.675089952 | 1.000000000 | 0.500000000 | 0.741100 |
| `offset_on` | CPU | 0.650525495 | 0.634190160 | 0.634190160 | 1.000000000 | 0.500000000 | 0.566800 |
| `null` | CUDA | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | first-call/JIT dominated |
| `base_only` | CUDA | 0.693826973 | 0.675090015 | 0.675090015 | 1.000000000 | 0.499999970 | 0.067584 |
| `offset_on` | CUDA | 0.650525451 | 0.634190142 | 0.634190142 | 1.000000000 | 0.500000000 | 0.047104 |

Core CPU/GPU agreement is at numerical precision:
- `base_only` rung max abs Δ ≈ `8.74e-08`; projection Δ ≈ `8.10e-08`; π score Δ = `0.0`.
- `offset_on` rung max abs Δ ≈ `9.51e-08`; projection Δ ≈ `-4.47e-08`; π score Δ = `0.0`.

The first local validation attempt reported two failures with `summary_max_abs_delta = 0.199999988`. That was not a GEO failure. It was traced to brittle sign/fraction diagnostics at an analytic zero endpoint:
- CPU float64 evaluated rung-0 `YZ` as tiny positive (`≈ +7.65e-18`).
- CUDA float32 evaluated the same analytic zero as tiny negative (`≈ -5.46e-09`).
- One rung out of five changed `yz_pos_frac` and `zy_inverted_frac`, yielding exactly `1/5 = 0.2`.

The included source was patched to separate **core summary agreement** from **auxiliary sign-count diagnostics**, add a sign deadband (`SIGN_EPS = 1e-6`), and write `probe25_summary_delta_by_metric.csv` so future validation points to the exact field instead of hiding it behind one max-delta number.

**Status for final benchmark:** Probe 25 is the GEO path the final official benchmark should import. It replaces the older raw-text/aperture GEO plumbing from Probes 14/17/18/19/22 and gives D_M a fast, exact, classical reference manifold.


## 9. End-to-end narrative (what the run actually demonstrates)

1. **00→06 (null base):** D_M-as-compressor was the wrong frame; it never fairly beat PCA/random projection. The repeated near-miss was always the composite/aggregate channel, and the durable lesson was "random projection is strong."
2. **07:** Reframing D_M as a Bell-witness *listener* immediately surfaced delay-tracking correlation (connected tracking r=+0.95).
3. **08→10:** The signal is *directional* (YZ primary / ZY reciprocal), *delay-locked* (energy/specificity r≈0.94–0.95), forms a *π-periodic phase trajectory* (best π-fit 0.984, p=0.024 on base_only), and *separates conditions* cleanly (null→active std_dist 5–6).
4. **11:** CUDA physics kernels match CPU to ~1e-8 at tile/rung level (a summary-aggregation diff of 0.59 remains to reconcile).
5. **12:** did **not** complete this session.
6. **13:** The real benchmark — three substrates agree, active manifolds separate from null and **collapse under bit-shuffle**, and DER controls show directional pairing + delay order are load-bearing (yz-swap/delay-permute → 0.05% R@1) while scalar direction is not (98.85%).
7. **14/17/18:** GPT-2 free QK geometry is highly *compatible* with the older D_M-constrained aperture path (compat up to 0.93–0.92), with margins ordered old-geo ≫ gproj ≫ qproj, and the full D_M path ≈5× faster than the GPT-2 QK path.
8. **19/22:** Retrieval task was saturated (100% everywhere) and is non-discriminating as configured; **20** is the null-corpus control built to expose that exact fingerprinting confound.
9. **15/21/23/24:** Hardening — native null ordering is suggestive but does **not** clear controls; the active phase trajectory **clears** the bit-shuffle control but is (deliberately) invariant to witness-label shuffle; forbidden *single* faults are absorbed, but *compound* faults (k≈2–3) cross a clean collapse boundary, consistent with an error-correcting dimensional manifold.
10. **25:** GEO was cleaned into an exact classical reference path. The CPU and CUDA rung/projection fields agree to ~1e-7, active conditions have π score 1.0, null is exact zero, and the only observed validation issue was a fixed sign-count endpoint diagnostic. This is the GEO implementation to import into the final official benchmark.

## 10. Items worth flagging for the final benchmark pass

- **Final benchmark should import Probe 25 GEO**, not the older raw-text/aperture GEO from Probes 14/17/18/19/22.
- **Run Probe 25 v4 once after replacement** and archive the clean validation artifacts (`probe25_summary_delta_by_metric.csv`, agreement CSV, validation CSV) beside the final benchmark record.
- **Probe 12 aborted** — retire it or replace its role with the new final benchmark. Probe 13 superseded its scope, but Probe 13 also predates the Probe 25 GEO cleanup.
- **Probe 11 summary diff (0.59)** — tile/rung physics match exactly; reconcile the summary-score normalisation between CPU and CUDA paths before freezing `kernel.cu`.
- **Probes 19/22 are non-discriminating** (100% across all methods incl. null and GPT-2) on 3 queries / 12 candidates — re-run against the `random_null_corpus.txt` from Probe 20, scale up the bank, and report margins rather than top-k if using retrieval utility at all.
- **Probe 18 NaN warnings** (invalid value in subtract / scalar divide) — add guards for empty/degenerate cosine denominators.
- **Witness-label invariance** is intentional per the docstrings, but it means label-shuffle cannot serve as a falsification control; the load-bearing controls are `independent_bit_shuffle`, `reciprocal_break`, and `delay_permute` (Probes 13/21/24 confirm this).
- **base_only vs offset_on** separate only weakly in qproj (Probe 10 Δprojection ≈0.02), while Probe 25's exact GEO separates them by a larger controlled geometry difference. Decide whether the final benchmark claim is active-vs-null, or active-condition discrimination.
- **Final bounded claim should remain conservative:** D_M projects a YZ-primary / ZY-reciprocal dimensional manifold; active manifolds separate from null; same-shot pairing, reciprocal structure, and delay order are load-bearing; compound faults cross a collapse boundary. Do not claim Bell certification, density reconstruction, prepared Bell states, QPU speedup, or quantum advantage.

---

*Numerical values through Probe 24 are transcribed from `proberesults.txt` (session `20260606`, backend `ibm_marrakesh`, device RTX 3090). Probe 25 values are from the `20260607` GEO precision-reference cleanup run and subsequent validation patch. Purposes are paraphrased from each file's module docstring. Bounded-claim discipline is the suite's own: D_M projects a YZ-primary / ZY-reciprocal dimensional-entanglement manifold and separates null / base-delay / offset conditions under shared metrics — it does **not** certify Bell nonlocality, reconstruct density matrices, prove prepared Bell states, or claim quantum advantage.*
