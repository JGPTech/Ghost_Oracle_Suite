# Architecture

How the suite is wired together. The math is in `math.md`; the trajectory is in `PROCESS_RECORD.md`. This document explains the *design* — why the layout looks the way it does, what each piece is responsible for, and how data flows from physical quantum hardware to a top-1 retrieval index.

The headline claim is operational: a streaming kernel that scores `(query, key)` pairs against the G_M operator, runnable against three substrates (mathematical reference, classical noiseless sampler, physical QPU shots) with the agreement metric giving a continuous quantitative readout of substrate-specific noise. The architecture exists to make that one claim defensible.

---

## The central abstraction: bases

A **base** is a `.npz` file with shot-level measurement data for the per-tile Hadamard-test circuit at a fixed `(a, b)` angle. Every consumer downstream reads bases through one schema.

```
job_id        : str
num_tiles     : int
ctrl_tile{t}  : uint8, shape (n_shots,)         per-shot control-qubit measurement
ghost_tile{t} : uint8, shape (n_shots, 4)       per-shot [a1, a2, b1, b2]
```

The schema is identical whether the base came from a physical quantum processor or a classical noiseless sampler. That symmetry is the whole point: the same projection code that estimates G_M from QPU shots also estimates G_M from classical-substrate shots, with no branching. **Same algorithm, three substrates.**

Three producers exist:

- **`qpu.py`** submits the circuit to IBM Runtime. Produces no file directly; the job ID is the handle.
- **`dump.py`** pulls a completed Runtime job by ID and writes the canonical `data/job_<JOB_ID>.npz`.
- **`gpu.py`** is a faithful classical implementation of the same projection circuit. It samples the noiseless 8-bin joint distribution that the GHZ-block circuit produces by construction: $a_1 = a_2$ drawn from $\mathrm{Bernoulli}(\sin^2(a/2))$, $b_1 = b_2$ drawn from $\mathrm{Bernoulli}(\sin^2(b/2))$, control bit constrained by the swap-test outcome on the matching `(a, b)` basis state. The output writes `data/ghost_oracle_gpu_<shots>shots_seed<seed>.npz` with the same key schema. **This is not an arbitrary classical baseline — it is the noiseless reference implementation of the same circuit the QPU runs.**

Probe 4 has a fourth, older producer (`noiseless_base_*.npz`) that statevector-simulates the circuit in pure Python. It's preserved for trajectory legibility and remains byte-compatible with the schema. The `auto_find_base("gpu")` helper finds both `noiseless_base_*` and `ghost_oracle_gpu_*` files transparently.

The fact that the schema makes no distinction between QPU and classical bases is what enables the substrate-comparison architecture the final benchmark verifies.

---

## The two channels

The headline benchmark and the production projection runtime both evaluate **G_M two different ways at the same `(a, b)` pair, simultaneously, in one kernel pass.**

```
Projection channel        Geometry channel
─────────────────         ────────────────
Reads a base's 18-bin     Evaluates the closed form:
bucket counts, applies      G_M(a, b) = sqrt((1 + cos a cos b)/2) / α
log-likelihood ratio      inline, in fast __sinf/__cosf intrinsics.
reweighting under the
change of angle
(orig_a, orig_b) → (a, b).
Returns G_M from physical
or simulated shots.
```

The projection channel is the **substrate-specific** evaluation: it returns G_M as implemented by whatever shots are in the base — QPU hardware, classical noiseless sampler, or mathematical reference. The geometry channel is the **substrate-agnostic** closed form, used for the argmax. Together they form a per-pair self-consistency check.

For each pair the kernel emits both values, plus their absolute difference. Averaged across the `M` keys, this becomes the **agreement metric**:

```
agreement = (1/M) Σ_j | projection(q, k_j) − geometry(q, k_j) |
```

Empirically, across the bases the suite ships with:

- Classical noiseless base (`gpu.py` output): 0.01–0.03. Pure shot noise at $N_{\text{shots}} = 4096$ per bucket.
- Physical QPU base: 0.07–0.13. Shot noise plus hardware noise — decoherence on the ghost CNOTs, gate-fidelity errors, calibration drift.

The ratio of QPU to classical agreement is roughly **5×**. This is the quantitative hardware-noise readout the substrate-comparison architecture was built to produce. It is *not* a "quantum advantage" measurement — the corrected Probe 12 result shows the classical noiseless projection gap is actually *larger* than the QPU's, because hardware noise attenuates the QPU's projection signal relative to its noiseless reference. Both substrates retrieve cleanly at the production operating point; the agreement metric is the quantitative measurement of how much hardware noise costs you in projection-signal fidelity.

This is what makes the tied design load-bearing. Without it, the benchmark is reporting numbers from a kernel that could be doing anything. With it, every reported retrieval is shadowed by a per-row certificate that the geometry it argmaxed over is the same operator the projection channel reconstructed from substrate shots, within a substrate-dependent bound.

The lesson the trajectory taught and the tied design encodes: `final_benchmark_combined.py` removed the parallel cosine evaluation because it looked redundant; the certificate vanished and the result became unfalsifiable. The parallel evaluation isn't redundant; it's the channel that ties the analytical operator to the substrate's actual output.

---

## The projection channel, in detail

The projection channel implements importance-weighted bucket reweighting. Conceptually: the base was sampled at `(orig_a, orig_b)`; to estimate the operator at a new `(na, nb)`, reweight each shot by the likelihood ratio of having observed its measurement outcome at the new angles vs. the original ones. The math is in `math.md`; the implementation has three details worth pinning down.

**Bucket compression to 18 ints.** The raw per-shot data is `(ctrl, ghost4)` for thousands of shots per tile. The projection channel doesn't need the shot ordering; only the joint counts over `(a_bucket, b_bucket, ctrl)` matter, where `a_bucket = a1 + a2 ∈ {0, 1, 2}` (and likewise for `b`). That's `3 × 3 × 2 = 18` ints per tile. The collapse happens once, in Python, at base-load time. From that point on every consumer reads 18 ints, not 4096 raw shots.

**Log-space reweighting with hard clipping.** The per-shot log-weight can blow up when `p` is near 0 or 1. Two clips defuse this: `p` is clipped to `[ε, 1 − ε]` with `ε = 0.05` before any log, and the resulting log-weight is clipped to `[-C, +C]` with `C = 3.0`. These constants correspond to the `EPS` and `CLIP_LOG_W` `#define`s at the top of Section 1 in `ghost_kernel.cu`, calibrated for fp32 stability across the angle range the suite uses.

**Bucket-mask calibration.** Some bases under some calibrations give a sharper projection signal when specific bucket pairs are zeroed before reweighting. Probes 13 through 18 systematically explored this and found that the right mask is calibration-dependent — different bases (even from the same algorithm) sometimes prefer different masks, and there is no universal "Golden Mask" that wins everywhere. The final production kernel (`final_benchmark_5way.py`) includes a per-base pre-flight that searches over candidate masks and squelch thresholds on a small calibration set, picks the best for that base, and locks it for inference. At the production operating point (d=256, power=256), the baseline mask (no buckets dropped) wins on all observed bases — mask calibration is preserved as defensive infrastructure for harder operating points.

The geometry channel by contrast is a handful of fp32 ops (two `__cosf`, one multiply, one `sqrtf`, one divide) — its cost barely registers per pair.

---

## The streaming kernel

`tied_streaming_perdim` (production) and `fb_compute_all_channels` (final-benchmark variant) in `ghost_kernel.cu` are the operational core. The simplified pseudocode:

```
For each query q_i (i = 0 .. N − 1):
  For each key   k_j (j = 0 .. M − 1):
    For each dim k    (k = 0 .. d − 1):
      g[i, j, k] = geometry_channel  (θ_Q[i, k], θ_K[j, k], α)
      p[i, j, k] = projection_channel(θ_Q[i, k], θ_K[j, k], α; counts18, mask)
  G[i, j] = mean_k g[i, j, k]
  P[i, j] = mean_k p[i, j, k]
  argmax_j G[i, j]   →   out_idx[i]
  out_agreement[i] = mean_j |G[i, j] − P[i, j]|
```

What never materializes: the full `N × M × d` per-dim tensor, the `N × M` score matrix, the `N × M` per-dim difference tensor. Everything is accumulated into per-row registers and reduced into the `N`-long output vectors at the end. That's where the memory advantage comes from — the streaming kernel's working set is the embedding inputs plus four `N`-long output arrays, regardless of the score matrix's notional size.

Three implementation details that took several iterations to land:

**Per-dim aggregation is non-negotiable.** `final_benchmark_tied.py` used a scalar phase-lift (collapse the embedding to a single angle via `mean(X) · sqrt(d_k)`) and got 0% top-1 at extreme scale because all retrieval signal lived in the per-dim structure. The current design phase-lifts each dimension independently, evaluates G_M per dim, and averages. Probe 10.1 is the architectural justification — per-dim averaging is what neutralizes coherent same-dim outlier attacks (no single dim can dominate the aggregate score because each dim is bounded in `[0, 1/α]` before the mean).

**Dimension matters more than mask architecture.** Probes 11.1 and 11.2 showed that at `d = 64`, projection's per-pair output is compressed into a narrow window that, after mean-pooling over `d`, sits near the float32 noise floor. This motivated the bucket-mask exploration (probes 13–18) and the Flash-Squelch threshold-power scoring (probes 16, 20) as workarounds. Probe 20 found that at `d = 256`, the √d SNR gain lifts the projection signal cleanly above the noise floor across all observed bases with no mask and a modest squelch power. The production operating point is `d = 256, power = 256`; the mask infrastructure remains as a defensive measure for narrower regimes.

**Phase-lift saturates.** `θ = (π/2)(1 + tanh(x/3))`. Bounded in `[0, π]`, monotonic, smooth, near-identity for small `|x|`. The earlier affine lift `θ = (π/2)(1 + x)` from Probe 10 wraps the cosine around for spike values, which accidentally suppressed attacks and made experiments uninformative. The tanh map keeps spike cosines well-defined without destroying clean signal.

**FlashAttention-style 2D shared-memory tiling.** Naive per-thread streaming hits a register/spill cliff once `d` grows large because each thread needs `local_q[d]` register space. The kernel tiles both dimensions: `TILE_M = 32` keys at a time, `TILE_D = 64` embedding dimensions at a time, with the key block cooperatively loaded into shared memory and the query slice held in registers. Partial sums accumulate across the inner `d` loop, then `argmax` resolves within each `TILE_M` block. This is what lets the kernel handle `d` up to several hundred without running out of registers or shared memory.

The result: one CUDA launch produces the top-1 retrieval index, top score, projection-channel score, and the per-row agreement metric.

---

## ghost_kernel.cu

Single source of truth for every CUDA kernel in the suite. Loaded once via `cupy.RawModule` by `gpu.py`, `projection_benchmark.py`, and `final_benchmark_5way.py`. The sections, all linked at top:

| § | Symbol(s) | Role |
|---|---|---|
| 1 | `clipped_log_pair`, `projection_channel`, `geometry_channel` | Shared device helpers used by everything below |
| 2 | `ghost_rank_k_matmul`, `ghost_rank_k_matmul_batch` | Custom T1 kernel — matches cuBLAS-on-lifted-representation to fp32 noise, used for the rank-K matmul baseline |
| 3 | `ghost_projection` | N-matrix projection: applies the projection channel to a full `(N, matrix_size, matrix_size)` output. The 4×4 case is `N=1, matrix_size=4` |
| 4 | `tied_streaming_perdim`, `tied_materialize_perdim` | Streaming kernel and its diagnostic sibling (materializes the full `N×M` score matrices for correctness checks) |
| 5 | `ghost_ghz_sample` | GHZ closed-form sampler used by `gpu.py` to produce noiseless classical bases |

Reasons for one file instead of many:

- `clipped_log_pair` is `__device__ inline` so every consumer shares it without indirect calls.
- `projection_channel` and `geometry_channel` are shared device functions used by both the standalone projection (Section 3), the tied kernels (Section 4), and the final-benchmark `fb_compute_all_channels` kernel.
- Section 2's rank-K kernel is in here even though it's not part of the G_M pipeline because Probe 6 (3-way convergence) needs it to demonstrate that cuBLAS and the custom T1 kernel agree on T1 to fp32 noise — that comparison is part of how the suite establishes its bona fides on the trigonometric operator family before claiming anything novel about G_M.

The final-benchmark kernel `fb_compute_all_channels` is appended to this source at runtime by `final_benchmark_5way.py` rather than living in the file directly; it computes both channels plus agreement in one pass and supports the dynamic-mask bitmask interface the per-base calibration produces.

---

## The Python layer

`ghost_oracle/final_benchmark_5way.py` is the headline runtime — the five-way verification: cuBLAS, tied, geometry-only, projection-driven-by-QPU, projection-driven-by-classical. `ghost_oracle/projection_benchmark.py` is the earlier headline benchmark (Probe 10.1 era) and remains in the repo for trajectory legibility and continuous-integration sanity checks.

The final benchmark's responsibilities:

1. **Auto-find or accept bases.** `auto_find_base()` finds both QPU and classical bases by glob pattern.
2. **Compress bases to 18-int per-tile bucket counts.** `build_bucket_counts()` does the `(ctrl, ghost4)` → `(3, 3, 2)` collapse.
3. **Per-base pre-flight calibration.** For each base, sweep candidate masks and Flash-Squelch thresholds on a small calibration set, pick the best (tile, mask, threshold) triple, write to a JSON manifest. Re-runs can skip calibration by passing the manifest.
4. **Phase-lift the input embeddings.** `phase_lift_perdim()` is the production tanh map.
5. **Stage queries and keys for the attack.** `make_attacked_jittered_embeddings()` produces the Probe 10.1 setup: gaussian keys, queries = keys + jitter, fraction of keys spiked on a shared dim at fixed magnitude.
6. **Run all five paths on the same inputs.** cuBLAS uses raw embeddings (it's the classical control showing what standard attention does); the other four run through the projection kernel at the per-base calibrated mask and threshold.
7. **Report.** Per-path top-1 accuracy, Flash-Squelch signal fraction, attack-spike fraction, wall time. Per-base agreement metric. All saved to a JSON dump for downstream analysis.

The CLI is deliberately small. `--N` and `--d` set the operating point; `--master-seed` makes runs reproducible; `--manifest` reuses a saved calibration; `--skip-gpu` / `--skip-qpu` restrict the substrate sweep.

There is one separate correctness check that runs by default in the earlier `projection_benchmark.py`: `run_correctness()` invokes `tied_materialize_perdim` instead of the streaming kernel, materializes the full `N × M` per-dim score matrices, and compares to the analytical numpy reference. That's how the geometry kernel's `2.3e-5 MAE vs numpy` is established at every run.

---

## The probes layer

The probes are not on the production data path. They live in `probes/` and read the same bases, but each one is an end-to-end script that tests a specific claim and prints its own report. Probes 1 through 10.1 are in the repo; probes 11 through 20 are documented in `PROCESS_RECORD.md` but their final results are folded into the production benchmark rather than preserved as standalone scripts.

Three constants are shared suite-wide and live in every probe's CONFIG block plus `qpu.py`:

| Constant | Value | Source |
|---|---|---|
| `ANGLE_SCALE` | 1.05 | Suite-wide; geometric scaling that keeps angles inside the smooth region of the phase-lift while exploiting the saturation near `π/2` |
| `ALPHA_NORM` | 0.9127 | Suite-wide; normalization such that `G_M` peaks at 1 over the expected angle range |
| `NUM_TILES` | 12 (probes), 16 (production) | Historical 12-tile runs vs. current 4×4 = 16-tile QPU jobs |

The 12-vs-16 split is the only nontrivial mismatch. Original probe runs used a 12-tile layout when the IBM hardware available at the time couldn't reliably hold a 4×4 cluster. The current generation handles 16 tiles. Every probe takes `--num-tiles` to override.

---

## Data flow, end to end

The full pipeline from substrate to retrieval index:

```
IBM Runtime backend                                  classical sampler              mathematical reference
    │                                                     │                                  │
    ▼                                                     ▼                                  ▼
 qpu.py submits circuit                              gpu.py invokes                   numpy FP64
    │                                                ghost_ghz_sample                  reference
    ▼                                                     │                                  │
 (job ID)                                                 ▼                                  ▼
    │                                                noiseless base                  reference base
    ▼                                                     │                                  │
 dump.py extracts                                         │                                  │
    │                                                     │                                  │
    ▼                                                     ▼                                  │
 data/job_*.npz   ◄────  shared schema  ────►  data/ghost_oracle_gpu_*.npz  ◄────────────────┘
    │                                                     │
    └─────────────────────────┬───────────────────────────┘
                              ▼
                  final_benchmark_5way.py
                              │
                  per-base pre-flight calibration
                  (pick best tile, mask, threshold)
                              │
                  load_base + build_bucket_counts
                              │
                              ▼
                  per-tile 18-int counts + calibrated mask
                              │
            phase_lift_perdim(query, key embeddings)
                              │
                              ▼
                  fb_compute_all_channels kernel
                  ─────────────────────────────
                  projection channel:  bucket reweighting → substrate readout
                  geometry channel  :  closed form        → argmax + truth
                  per-dim aggregation:  mean over d       → robustness
                  fused argmax       :  no N×M matrix     → O(N) memory
                              │
                              ▼
            out_idx, out_score, out_proj_score, out_agreement
            ──────────────────────────────────────────────────
            top-1 retrieval   substrate-specific certificate
                              │
                              ▼
            five-way scoring: cuBLAS, tied, geo, qproj, gproj
            ──────────────────────────────────────────────────
            top-1 / signal fraction / attack-spike / wall time
            per substrate, with per-base agreement quantifying
            hardware-noise attenuation
```

Every arrow in this diagram is exercised by every default run of `final_benchmark_5way.py`. Removing any one of them breaks a claim:

- Without the shared base schema, the substrate paths fork and the certificate stops being meaningful (different operators).
- Without bucket compression, the projection channel runs in `O(n_shots)` per evaluation and the kernel no longer fits.
- Without log-clipping, the projection channel produces NaNs on angle extremes where bucket counts are near zero.
- Without per-base calibration, narrow-d operating points fail (probes 11.1 / 11.2 / 13–18 trajectory).
- Without per-dim aggregation, single-dim outlier attacks pass through the operator (Probe 10's failure mode).
- Without the tanh phase-lift, spike values wrap the cosine and cancel themselves (also Probe 10's failure mode).
- Without fused argmax, the score matrix materializes and the memory advantage collapses.
- Without the projection channel, the geometry channel is unfalsifiable; the substrate-equivalence claim has no continuous integrity check.

The architecture is the set of decisions that keep all eight arrows alive at once. The probes establish that each individual claim along this path is real; the final benchmark establishes that they hold simultaneously across three substrates under load.

---

## Why not …

A few designs that look tempting at first and aren't.

**Why not just use cos·outer-product?** G_M correlates with `cos a · cos b` at 0.9992. They're almost the same operator on random angles. The 0.0008 difference is the `sqrt((1 + ·)/2)` saturation, and *that's where the structural robustness lives*. Strip the sqrt and the operator is unbounded above; under a coherent same-dim attack, one big dim dominates the mean and per-dim aggregation stops protecting the retrieval. The sqrt is load-bearing.

**Why not cache the geometry result and use only the projection channel?** Because then there's no substrate readout. The geometry kernel doesn't know whether the projection channel agrees; it just computes a closed form. The tied design's value is in the *coupled* evaluation — every reported retrieval is shadowed by a per-row check that the operator the kernel argmaxed against is the operator the physical or classical substrate implements, within a quantitatively bounded gap.

**Why not use the projection channel for the argmax instead of the geometry channel?** Two reasons. First, the projection channel is `O(18)` per evaluation in operations but has roughly 5×–15× the constant factor of the geometry channel in practice; on the inner loop it would dominate the kernel time. Second, the projection channel is shot-noise limited, so its argmax has higher variance than the geometry channel's. Probe 11.1 measured this concretely: at d=64 on certain bases, projection's per-key aggregated values cluster within a window narrower than fp32 noise, so the argmax becomes essentially random. The geometry channel argmax is the sharp signal; the projection channel readout is the substrate truth. Both jobs are necessary, and assigning them to the right channels is what makes the kernel fast.

**Why not split the kernel into two passes — geometry first, projection second only for the top-K?** Because the agreement metric is the *whole certificate*. Computing it only on the argmax leaves the bulk of the score matrix uncertified. The full per-row mean over `M` is what reproduces the probe 7–8 channel-error measurement at scale; restricting it to top-K would let the substrate be wildly off everywhere except where the argmax already pointed.

**Why not aggregate per-dim G_M with something other than the mean?** Tried various aggregations during the benchmark trajectory. The mean is what survived: it's the only aggregator that's `O(1)` to update inside the streaming loop without buffering per-dim scores, and it's the only one whose robustness behavior is straightforward to argue from first principles (each dim is bounded in `[0, 1/α]`; the mean is bounded by the per-dim bound divided by `d`). Median would defend better against asymmetric attacks but at the cost of either materializing the per-dim vector or accepting an approximate streaming median.

**Why not strip the projection channel from the hot loop and run it once per base at startup?** Probe 11 (v1) measured this directly. Stripping projection from the streaming kernel gave a 5.3× speedup with zero accuracy loss because the geometry channel drives the argmax in both cases. The reason this isn't done in production is the agreement metric: a one-shot certify pass at startup would certify the *base*, but it would no longer certify each `(query, key)` evaluation under the operating-point angles the queries actually visit. The tied design pays ~5× in inner-loop cost to keep the per-query certificate live. Worth it depends on the application; the architecture supports both modes.

---

## Pointers

- **The math:** `docs/math.md` — derivations of T1, T2, T3, G_M, the projection-channel identity, and what the agreement metric actually measures across the three substrates.
- **The trajectory:** `PROCESS_RECORD.md` — how the architecture got here, including the bucket-mask exploration (probes 13–18), the substrate-equivalence correction (Probe 12 redux), and the auto-calibrating production kernel (Probe 20).
- **Known issues:** `docs/known_issues.md` — limits of the current implementation.
- **Per-probe context:** `probes/README.md` — what each numbered probe contributes to the arc.
``
