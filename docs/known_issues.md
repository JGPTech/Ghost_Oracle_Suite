# Known Issues

Running list of everything broken, suspect, or under-justified in this codebase. Some are bugs preserved verbatim for trajectory legibility — Probe 9's clipping artifact, Probe 10's three setup bugs, Probe 8.2's alternation optimizer — that have follow-up probes implementing the correct version. Some are genuine limitations of the current production code that no follow-up has addressed yet. Some are caveats on the data, not the code.

The norm for this project is: **bugs stay in the repo with header markers, the corrections live in numbered follow-ups, and this file is the running index with full diagnoses.** The point of keeping the broken versions is that the *process* of finding and fixing them is part of the research record. A user who runs `probe10_ghost_attention.py` and sees a SUPERSEDED banner learns more than one who finds it quietly deleted.

Three categories, in order of severity:

- **Broken** — code paths that produce wrong answers if invoked. Have either been replaced (follow-up probe, fixed in production) or are gated behind a banner.
- **Confounded** — code paths that produce real numbers but those numbers don't mean what they appear to mean.
- **Open** — design choices that are working but unjustified, or analyses that should be done and haven't been.

---

## Broken

### Probe 8.2 — alternation loop diverges

**File:** `probes/probe8_residual_decomposition.py`, Phase C (originally `probe8_2_drift_alternating.py`).

**Symptom:** With `--phase C` and the alternation loop enabled, the optimizer can drive the shared coherent drift `(d_a, d_b)` to physically meaningless values. The original 12-tile run reached `d_b = -3.66 rad` (-210°) at iteration 4. Even at one iteration, MAE reduction is +10.9% versus Phase B's +20.9% — the alternation makes the answer worse, not better.

**Three compounding bugs:**

1. **Drift accumulation.** Update step is `d_a += d_new` instead of `d_a = d_new`. Each iteration adds the *next* estimate to the *current* drift rather than replacing it, so the drift grows roughly linearly in the iteration count regardless of the data.
2. **Wrong regularization target.** The OUT regularizer penalizes the variance of per-tile residual drifts `eps`, not their magnitude. This holds the *spread* of `eps` small while letting `mean(eps)` drift freely, which is the opposite of what the regularizer was meant to do.
3. **No hard bound on |d|.** The drift search range is unbounded, so once bug 1 starts compounding, nothing catches it before angles wind past 2π.

**What works.** Phase C.2 (single shared-drift fit, no alternation) and Phase C.3 (per-tile channels on the drift-corrected reference) are usable. The full alternation loop in Phase C.4 is not.

**Workaround.** Run Phase B (`--phase B`) for the production-quality channel decomposition. Phase C is gated by `--phase C` and emits the relevant warning.

**Fix.** Would require replacing the alternation with a properly bounded joint optimizer (drift + channels in one pass) with hard bounds on `|d|` and an L2 anchor on per-tile `eps` toward zero rather than toward cohort mean. Listed in PROCESS_RECORD Part 10 as future work.

---

### Probe 9 Stage 1 — clipping artifact inflates MAE

**File:** `probes/probe9_ghost_operator.py`, Stage 1.

**Symptom:** Stage 1 reports `MAE(GPU, analytic) = 0.19` and `MAE(QPU, analytic) = 0.26`, both much larger than the expected shot-noise floor of ~0.016. Reads as "QPU and GPU disagree with the closed form by 19-26%."

**Root cause.** Stage 1 evaluates the *normalized* `G_M = min(1.0, sqrt((1 + cos a cos b)/2) / α)`. Three of the 12 tiles in the original run had `cos(a)cos(b)` small enough that the raw `G_M_raw / α` exceeds 1, and the `min(1.0, ...)` clamp fires. The analytical reference is clamped to 1.0 on those three tiles, but the GPU sample isn't (it's at shot-noise distance from `G_M_raw / α ≈ 1.05`). The reported MAE is dominated by the three clipped tiles.

**Verified by.** Probe 9.1 Stage 1 re-runs the same comparison against the unclipped `G_M_raw = sqrt((1 + cos a cos b)/2)`. Result on the same bases: `MAE(GPU, analytic) = 3.6e-3` (below the 1/√4096 = 0.016 shot-noise floor) and `MAE(QPU, analytic) = 9.7e-2` (the actual characterized channel error).

**Workaround.** Read Probe 9.1's Stage 1 instead of Probe 9's Stage 1 for the consistency claim. Production code (`projection_benchmark.py`) clips in software at the end of the pipeline, so the artifact is purely a Probe 9 reporting issue — nothing in the production runtime is affected.

---

### Probe 9 Stage 4 Demo 2 — truth function equals the oracle

**File:** `probes/probe9_ghost_operator.py`, Stage 4 Demo 2.

**Symptom:** Demo 2 prints `mse_lin / mse_gm ≈ 1e15` and claims the G_M-oracle regressor crushes a linear baseline by 15 orders of magnitude. The number is meaningless — it's a divide-by-zero artifact.

**Root cause.** The "truth" function in Demo 2 is `y = sqrt((1 + x1 * x2) / 2)`, which is *literally* `G_M_raw(x1, x2)`. The "G_M oracle" feature is the same expression. So `mse_gm ≈ 1e-15` (machine epsilon, because they're the same function up to fp32 rounding) and the printed ratio divides by it. The linear baseline isn't being beaten meaningfully; the "winner" is just being compared against zero.

Compounding this: the inputs are `x ∈ [0, 1]`, where `sqrt((1 + x1·x2)/2)` is nearly linear over the range anyway. So even if you fixed the tautology, a 4-parameter linear model wins.

**Verified by.** Probe 9.1 Stage 2 redoes the regression demo with two changes:

- Inputs in angle space `(π/4, 3π/8)` so `cos(a)·cos(b)` spans `[0.15, 0.50]` — the regime where sqrt curvature is visible.
- A clipping ceiling at 0.85 on the truth function `y = min(sqrt((1 + cos a cos b)/2), 0.85)` — so the truth has both curvature and saturation, neither of which a simple polynomial captures.

Result with the fix: G_M-structured regressor needs zero parameters to capture the function (it *is* the function up to the ceiling); polynomial degree 3 needs 10 parameters to get to 6e-7 MSE. Clean positive result for G_M's structural form.

**Workaround.** Skip Probe 9 Stage 4 Demo 2. Read Probe 9.1 Stage 2 for the actual saturation-regime regression result.

---

### Probe 10 — three setup bugs cancel the experiment

**File:** `probes/probe10_ghost_attention.py`. Marked `SUPERSEDED` in the header, prints a loud banner at the start of every run.

**Symptom:** Both the dot-product baseline and the G_M attention pipeline plateau at exactly `1 - outlier_fraction = 0.95` on clean data and stay there under "attack." Looks like "no separation, hypothesis falsified." Actually three bugs canceling out the experiment.

**The three bugs:**

1. **L2 renormalization defuses the spike.** After injecting `spike_magnitude = 50` into one dimension of a key, the code calls `XK_noisy /= np.linalg.norm(XK_noisy, axis=1)`. The norm becomes ~50 and the spike dimension is divided by ~50, leaving a unit vector with the spike pinned near 1 — i.e., not an outlier any more.
2. **Incoherent dimensions across the outlier set.** `inject_outliers` picks a random `k_bad` *per outlier key*. Outlier A's spike is on dim 17, outlier B's on dim 42, etc. The attention head sees uncorrelated noise across keys, not the coordinated same-dimension attack that breaks softmax in real LLM scenarios where one feature dimension goes hyperactive across many tokens.
3. **`d = 64` defeats the softmax bottleneck.** With `d = 64` and unit-normalized embeddings, dot products land in a narrow range, and softmax over 4096 candidates is already broad — no single key can dominate even before the attack. The mechanism G_M's per-dim aggregation is supposed to neutralize never engages in the first place.

**Combined effect.** Both methods correctly retrieve every non-outlier key. The spike fraction (5%) caps retrieval at 95% because no method handles the spike — but neither method is actually being challenged. Looks like a tie; was actually a non-experiment.

**Verified by.** Probe 10.1 (`probes/probe10_1_real_softmax_attack.py`) fixes all three:

- No L2 renormalization (the spike survives).
- Same-dim coherent attack: one shared `k_bad` for all outlier keys (the actual LLM failure mode).
- `d = 16` so softmax over N=1024 candidates concentrates and the bottleneck engages.

Result: DP top-1 drops from 74% (clean) to 43% (attacked). G_M tied per-dim holds at 84% clean and 84% attacked. Softmax attention mass on outliers reaches 0.42 (catastrophic). G_M outlier-to-non-outlier score ratio is 0.999 (the attack is invisible to G_M).

**Workaround.** Run Probe 10.1, not Probe 10. The latter is preserved in the repo as historical record and prints a SUPERSEDED banner.

---

## Confounded

### Probe 8.4 — base-2 Benford column is non-informative

**File:** `probes/probe8_residual_decomposition.py`, Phase D, Benford table column for base 2.

**Symptom:** The base-2 Benford column produces a constant value across every residual matrix the probe tests. Reads as "base-2 Benford detects no structure anywhere" — which is true but misleading.

**Root cause.** Benford's law in base $b$ describes the distribution of leading digits in $\{1, \ldots, b-1\}$. In base 2 there's only one possible leading digit ($d = 1$, since every nonzero number written in base 2 starts with 1), so $P(d=1) = 1$ identically. The test compares observed leading-digit frequencies to the Benford-law reference; in base 2 both are the singleton distribution $\{1: 1\}$, and the χ² statistic is identically zero regardless of input.

**Not a bug**, in the sense that the implementation is mathematically correct given the chosen formulation. It's a degeneracy of leading-digit analysis in base 2 — the question itself is empty.

**Workaround.** Read the base-3, 5, 7, 10 columns. Those are informative.

---

### Probe 8.4 — ν_p valuation tests confounded by shot count

**File:** `probes/probe8_residual_decomposition.py`, Phase D, p-adic valuation table.

**Symptom:** The ν_2 χ² statistics in Phase D look high enough to be suspicious of "hidden structure" in the residuals.

**Root cause.** Shot count is $N_{\text{shots}} = 4096 = 2^{12}$. The residuals are computed as differences of fp32-rounded ratios with $N_{\text{shots}}$ in their denominators, so any integer-scaled residual entry inherits 2-adic structure as an artifact of the denominator, regardless of any physical signal in the numerator. ν_2 χ²/dof reflects this artifact, not the residual physics.

ν_3 and ν_5 are less affected (3 and 5 don't divide 4096), but `ALPHA_NORM = 0.9127` is rational-ish and its fp32 representation can leak small amounts of 3- and 5-adic structure through rounding. Less severe than ν_2 but still not pristine.

**Verified by.** The probe header notes the confounder, the table prints `(confounded by shot=2^12)` next to the ν_2 column, and the cross-base Benford analysis is consistent with "the residual is decoherence noise, not hidden structure" — which is also what Probe 8 Phase B finds via direct channel decomposition.

**Workaround.** Read ν_2 as "shot-count confounded, uninterpretable" and use ν_3, ν_5 as soft signals only. The cumulative Phase D verdict — "nothing meaningful survives" — is supported by the Benford columns 3 and 5 alone, so no claim depends on the confounded ν_p numbers.

**Fix.** Would require either resampling shots to a different (non-power-of-2) count or scaling residuals by `N_{shots}` before the valuation test. Listed as future work in PROCESS_RECORD Part 10.

---

### Projection benchmark — agreement metric uses a single representative tile

**File:** `ghost_oracle/projection_benchmark.py`, `representative_tile()`.

**Symptom:** The reported `agreement` value in `projection_benchmark.py` runs is computed against one tile selected from each base — the one with the most balanced `ctrl=0 / ctrl=1` split. Not the mean over all 12 (or 16) tiles in the base.

**Why it's done this way.** The projection channel is `O(18)` per evaluation regardless of tile count — it operates on a single 18-int histogram. Running across all tiles and averaging would mean 12-16× the projection-channel work for what is essentially a noise-floor diagnostic. The benchmark optimizes for one defensible number rather than twelve noisy ones.

**What this means for the certificate.** The agreement metric is the per-row mean `|projection − geometry|` against *one* tile's worth of physical shots. If the QPU's residual were strongly tile-dependent, the representative-tile estimate would underrepresent or overrepresent the true cross-tile residual structure depending on which tile gets picked.

Probes 7 and 8 measure the actual cross-tile spread of the QPU residual. The 0.10-0.20 range the agreement metric falls in is consistent with their findings, but the *exact* number depends on which tile `representative_tile()` selected, which depends on the base. A different base could produce a different selection and a slightly different agreement number — both still correct as certificates, both still in the Probe 7-8 range.

**Workaround.** None needed for the headline benchmark. For a more thorough cross-tile certificate, run Probe 7 and Probe 8 directly — they characterize the residual across every tile.

**Fix.** Could average the projection-channel evaluation over all tiles in a future revision. Tradeoff: 12-16× projection-channel work per evaluation, in exchange for a tighter certificate. Not currently judged worth the cost; the representative-tile estimate is already well inside the Probe 7-8 envelope.

---

## Open

### Phase-lift design is informal

**File:** `ghost_oracle/projection_benchmark.py`, `phase_lift_perdim()`; also `probes/probe10_1_real_softmax_attack.py`, `phase_lift_bounded()`.

**Current choice:** $\theta(x) = \frac{\pi}{2}(1 + \tanh(x / 3))$. Chosen because it's bounded, smooth, near-identity for small $|x|$, and saturating at both extremes. The $1/3$ inside the tanh was picked so $\theta(x) \approx \pi/2 + (\pi x)/6$ near zero, giving reasonable spread over typical embedding ranges $|x| \lesssim 3$.

**What's open.** No systematic comparison of phase-lift alternatives. Other valid choices — `(π/2)(1 + erf(x))`, `(π/2)(1 + x/sqrt(1+x²))`, piecewise linear with different cutoffs — would all produce bounded saturating maps with slightly different effective input distributions. Which one maximizes retrieval accuracy on real embedding statistics is currently unknown.

What we do know: any unbounded affine lift like Probe 10's $\theta = (π/2)(1+x)$ breaks the architecture (spike values wrap the cosine). And anything bounded and saturating will inherit the robustness property (each dim contributes at most $1/(d\alpha)$ to the aggregate), so the qualitative claim holds across the family. The quantitative optimum within the family is open work.

**Fix path.** Sweep over a parameterized family ($\theta(x) = (\pi/2)(1 + \tanh(x/s))$ for $s \in \{1, 2, 3, 5, 10\}$, plus erf and rational-tanh variants), measure top-1 accuracy on the Probe 10.1 same-dim attack benchmark across the family. Whichever survives the sweep without obvious overfitting becomes the recommended default.

---

### Headline benchmark accuracy saturates at jitter 0.3, d=64

**File:** `ghost_oracle/projection_benchmark.py`, default sweep settings.

**Observation.** The headline result reports `G_M tied = 100.00%` at every shape from 4096² through 65536² under jitter 0.3 and 5% same-dim attack at magnitude 50. Cleanly 100%, not 99.97% — the benchmark isn't actually hard enough to differentiate G_M from "perfect" at this operating point.

**What this means.** The architectural claim ("per-dim G_M holds where cuBLAS degrades") is supported. But 100% can also mean the task is easy. A more nuanced demonstration — say, 95% G_M vs 75% cuBLAS at higher jitter — would be a tighter binding of "G_M is robust here, cuBLAS isn't" than 100% vs 75%.

**What to run.** Sweep jitter at fixed N=16384: jitter ∈ {0.3, 0.5, 0.7, 1.0, 1.5, 2.0}. Find the value where G_M tied starts to fail. If it holds at 95%+ up to jitter 1.0, that's the strongest possible demonstration. If it cracks at 0.5, that's also informative — the operating range where G_M wins is bounded.

Same logic applies to attack fraction (currently fixed at 5%; sweep 1%, 5%, 10%, 20%) and embedding dimension (currently d=64 in production but d=16 in Probe 10.1 where the headline mechanism was first measured; sweep d ∈ {8, 16, 32, 64, 128}).

Listed in PROCESS_RECORD Part 10 as the first item of future work.

---

### Real LLM embedding statistics untested

**File:** N/A — synthetic data throughout.

**Observation.** Every benchmark in the suite — Probe 10.1, `projection_benchmark.py`, the agreement metric — uses synthetic Gaussian embeddings. The attention claim ("G_M attention is robust to coherent outlier attacks in LLM contexts") is structurally grounded but hasn't been tested on actual LLM activations.

**What to run.** Pull Q/K projections from any layer of a publicly available transformer (e.g. a Llama or Mistral checkpoint), use them as the embedding inputs to `projection_benchmark.py`, run the same attack. If the synthetic 100%-vs-75% gap persists on real learned representations, the attention claim graduates from "plausible architectural property" to "demonstrated on the actual target". If the gap collapses, the synthetic result was a feature of Gaussian embeddings and the claim narrows accordingly.

This is the single most-impactful next experiment. Listed in PROCESS_RECORD Part 10 as item 3.

---

### Clean-data baseline not reported alongside attacked-data

**File:** `ghost_oracle/projection_benchmark.py`.

**Observation.** The headline table reports cuBLAS and G_M top-1 *under attack* but not on clean data side-by-side at the same shapes. So a reader sees "G_M 100%, cuBLAS 73-79%" without knowing what cuBLAS would score with no attack on the same task.

**Why this matters.** "G_M wins under attack" is the right claim, but it could be made more honest with the clean-data row alongside. A picture like

| Shape | cuBLAS clean | cuBLAS attacked | G_M clean | G_M attacked |
|---|---|---|---|---|
| 4096² | 99% | 79% | 95% | 100% |

would be more informative than the current table. It shows the tradeoff: G_M loses a few points on clean data in exchange for resilience. Hiding the clean-data column makes G_M look strictly better, when actually it has a real cost.

Probe 10.1 Stage 5 reports clean-data competitiveness at d=16 and finds DP slightly ahead. The same comparison at d=64 in the production sweep hasn't been run.

Listed in PROCESS_RECORD Part 10 as item 2.

---

### `ALPHA_NORM` and `ANGLE_SCALE` are calibrated, not derived

**File:** `ghost_oracle/qpu.py` and every consumer.

**Current values.** `ALPHA_NORM = 0.9127`, `ANGLE_SCALE = 1.05`. Both predate the Probe 4 pivot and survived into the production code unchanged.

**What `ALPHA_NORM` does.** Normalizes the `G_M` output peak to approximately 1. The peak value of `sqrt((1 + cos a cos b)/2)` over the suite's angle range $(0, \pi/2 \cdot 1.05)$ is roughly 0.9127, so dividing by it pushes the peak just above 1, where the `min(1.0, ...)` clamp catches it. So `ALPHA_NORM` isn't a free parameter; it's a function of `ANGLE_SCALE` and the matrix-family range.

**What `ANGLE_SCALE` does.** Multiplies the angle range to $[0, \pi/2 \cdot 1.05]$, slightly into the saturation regime where the half-angle Hadamard form starts to flatten. Originally chosen so that the most-aligned matrix entries push into the saturation regime and the dynamic range fills the $G_M$ output range.

**What's open.** Neither value has been derived from first principles. They're "what works at the matrix-family scale." Probe 1 showed that `ALPHA_NORM` originally looked like a global gain rather than a geometry correction, and that intuition was correct under the T2 framing — but under the T3 framing, both constants are absorbed into a single calibration of the operator output range, and the question of whether 1.05 is *optimal* or merely *adequate* has not been re-asked since the Probe 4 pivot.

**Fix path.** Re-derive both constants under the T3 framing. `ALPHA_NORM` should equal `max_{(a,b) ∈ range} sqrt((1 + cos a cos b)/2)`. `ANGLE_SCALE` should be chosen so the peak of the output distribution sits at a target fraction of 1 (probably ~0.95, leaving headroom). Update both with the derivation as comments and verify nothing downstream silently changes — the projection benchmark and every probe should produce numerically the same headline result up to rounding.

---

## What's *not* on this list

A few things might look like issues but aren't, and stating them explicitly saves cycles later:

- **The QPU has ~10% MAE against G_M.** This is characterized hardware error, not a software bug. Probes 7 and 8 measured it; the agreement metric in `projection_benchmark.py` reproduces it at scale; PROCESS_RECORD Part 3 documents the four phases of attempted residual characterization. It's the cost of using physical quantum hardware.
- **`G_M` is not positive semidefinite.** Probe 9 Stage 3 tested 50 random Gram matrices and got 0/50 PSD. This is a *structural property* of `G_M`, not a bug. It's also why Probe 9.1 Stage 3 found `G_M` loses to RBF on PSD-friendly classification tasks. The indefiniteness is informative about where `G_M` does and doesn't apply.
- **The 12-tile vs 16-tile difference between probes and production.** Probe headers default to 12 because the historical runs used that layout. The `--num-tiles 16` override is documented in every probe and PROCESS_RECORD. Not a bug, just a transition.
- **`cuBLAS` is much faster on small inputs.** It's faster by 50-70× at small shapes and 4-5× at extreme. This is the honest compute tradeoff documented in PROCESS_RECORD Part 7 — `G_M` tied wins on accuracy under attack and on VRAM at scale, not on speed.

---

## Pointers

- **`PROCESS_RECORD.md` Part 9** — Originating list of known issues in narrative form.
- **`PROCESS_RECORD.md` Part 10** — Open questions for the next session, of which several appear in the "Open" section above.
- **`docs/architecture.md`** — How the production runtime is wired; explains why the representative-tile choice and the phase-lift design are where they are.
- **`docs/math.md`** — The math that the broken probes were trying to verify and that the production code implements correctly.
- **`probes/README.md`** — Narrative arc with per-probe context.