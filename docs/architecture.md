# Architecture

How the suite is wired together. The math is in `math.md`; the trajectory is in `PROCESS_RECORD.md`. This document explains the *design* — why the layout looks the way it does, what each piece is responsible for, and how data flows from physical quantum hardware to a top-1 retrieval index.

The headline claim is operational: a tied-channel streaming kernel that scores `(query, key)` pairs against the G_M operator, certified against physical QPU shots, with fused argmax and `O(N)` memory. The architecture exists to make that one claim defensible.

---

## The central abstraction: bases

A **base** is a `.npz` file with shot-level measurement data for the per-tile Hadamard-test circuit at a fixed `(a, b)` angle. Every consumer downstream reads bases through one schema.

```
job_id        : str
num_tiles     : int
ctrl_tile{t}  : uint8, shape (n_shots,)         per-shot control-qubit measurement
ghost_tile{t} : uint8, shape (n_shots, 4)       per-shot [a1, a2, b1, b2]
```

The schema is identical whether the base came from a physical quantum processor or a noiseless GPU sampler. That symmetry is the whole point: the same projection code that estimates G_M from QPU shots also estimates G_M from GPU shots, with no branching. The QPU base certifies the operator on physical hardware; the GPU base is shot-noise-limited ground truth. A probe that reports MAE against one number against either base is comparable to itself across hardware.

Three producers exist:

- **`qpu.py`** submits the circuit to IBM Runtime. Produces no file directly; the job ID is the handle.
- **`dump.py`** pulls a completed Runtime job by ID and writes the canonical `data/job_<JOB_ID>.npz`.
- **`gpu.py`** invokes the GHZ closed-form sampler in `ghost_kernel.cu` (Section 5) and writes `data/ghost_oracle_gpu_<shots>shots_seed<seed>.npz` with the same key schema.

Probe 4 has a fourth, older producer (`noiseless_base_*.npz`) that statevector-simulates the circuit in pure Python. It's preserved for trajectory legibility and remains byte-compatible with the schema. The `auto_find_base("gpu")` helper in the probe suite finds both `noiseless_base_*` and `ghost_oracle_gpu_*` files transparently.

The fact that the schema makes no distinction between QPU and GPU bases is what enables the whole tied-channel certificate that follows.

---

## The two channels

The headline benchmark and the production projection runtime both evaluate **G_M evaluated two different ways at the same `(a, b)` pair, simultaneously, in one kernel pass.**

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
shots.
```

The projection channel is the certificate: it cannot beat the physical sampling rate, but it *cannot lie about what the QPU produces*. The geometry channel is the engine: sharp, exact, fast, and used for the argmax. Together they form a self-consistency check at every single `(a, b)` evaluation.

For each pair the kernel emits both values, plus their absolute difference. Averaged across the `M` keys, this becomes the **agreement metric**:

```
agreement = (1/M) Σ_j | projection(q, k_j) − geometry(q, k_j) |
```

- GPU base: 0.01–0.06 (shot noise on 4096 shots per bucket bin).
- QPU base: 0.10–0.20 (the channel error characterized by Probes 7–8).

The agreement metric is the *whole point* of the tied design. Without it, the benchmark is reporting numbers from a kernel that could be doing anything. With it, every reported retrieval is shadowed by a per-row certificate that the geometry it argmaxed over is the geometry the physical circuit implements, within a quantitatively bounded gap.

This is what was missing from the early benchmark iterations. `final_benchmark_combined.py` removed the parallel cosine evaluation because it looked redundant; agreement collapsed and MAE went to 0.375 because the certificate had been thrown away. The lesson — "the parallel cosine evaluation isn't redundant, it's the ghost channel providing the certificate" — is hard-baked into the tied design.

---

## The projection channel, in detail

The projection channel implements importance-weighted bucket reweighting. Conceptually: the base was sampled at `(orig_a, orig_b)`; to estimate the operator at a new `(na, nb)`, reweight each shot by the likelihood ratio of having observed its measurement outcome at the new angles vs. the original ones.

Three implementation details earn their keep:

**Bucket compression to 18 ints.** The raw per-shot data is `(ctrl, ghost4)` for thousands of shots per tile. The projection channel doesn't need the shot ordering; only the joint counts over `(a_bucket, b_bucket, ctrl)` matter, where `a_bucket = a1 + a2 ∈ {0, 1, 2}` (and likewise for `b`). That's `3 × 3 × 2 = 18` ints per tile. The collapse happens once, in Python, at base-load time. From that point on every consumer reads 18 ints, not 4096 raw shots.

**Log-space reweighting with hard clipping.** The per-shot log-weight `f · (log p_n − log p_o) + (1 − f) · (log (1 − p_n) − log (1 − p_o))` can blow up when `p` is near 0 or 1. Two clips defuse this: `p` is clipped to `[EPS, 1 − EPS]` with `EPS = 0.05` before any log, and the resulting log-weight is clipped to `[-CLIP_LOG_W, +CLIP_LOG_W]` with `CLIP_LOG_W = 3.0`. These constants are calibrated for fp32 stability across the angle range the suite uses; they live as `#define`s at the top of Section 1 in `ghost_kernel.cu`.

**Per-tile representative selection.** The projection channel is `O(1)` per `(query, key)` pair regardless of how many tiles the base contains — it operates on a single tile's 18-int counts. `representative_tile()` picks the tile with the most balanced `ctrl=0 / ctrl=1` split before kernel launch. The benchmark reports agreement against that single tile, which is the noisiest per-shot but the most informative per bit. Future revisions could average across tiles; the current design favors one defensible number over twelve noisy ones.

The geometry channel by contrast is `4` fp32 ops (one `__cosf`, one `__cosf`, one multiply, one `sqrtf`) — its cost barely registers, which is why running both channels in lockstep is essentially free.

---

## The streaming kernel

`tied_streaming_perdim` in `ghost_kernel.cu` Section 4 is the operational core. It computes:

```
For each query q_i (i = 0 .. N − 1):
  For each key   k_j (j = 0 .. M − 1):
    For each dim k    (k = 0 .. d − 1):
      g[i, j, k] = geometry_channel  (θ_Q[i, k], θ_K[j, k], α)
      p[i, j, k] = projection_channel(θ_Q[i, k], θ_K[j, k], α; counts18)
  G[i, j] = mean_k g[i, j, k]
  P[i, j] = mean_k p[i, j, k]
  argmax_j G[i, j]   →   out_idx[i]
  out_agreement[i] = mean_j |G[i, j] − P[i, j]|
```

What never materializes: the full `N × M × d` per-dim tensor, the `N × M` score matrix, the `N × M` per-dim difference tensor. Everything is accumulated into per-row registers and reduced into the `N`-long output vectors at the end. That's where the 500× VRAM advantage comes from — at `N = M = 65536, d = 64`, the cuBLAS dot-product attention score matrix alone is 16 GB; the streaming kernel's working set is the embedding inputs (`N · d · 4` plus `M · d · 4` bytes) plus four `N`-long output arrays.

Three implementation details that took several iterations to land:

**Per-dim aggregation is non-negotiable.** `final_benchmark_tied.py` used a scalar phase-lift (collapse the embedding to a single angle via `mean(X) · sqrt(d_k)`) and got 0% top-1 at extreme scale because all retrieval signal lived in the per-dim structure. The current design phase-lifts each dimension independently, evaluates G_M per dim, and averages. Probe 10.1 is the architectural justification — Section 4 of its docstring spells out why per-dim averaging is what neutralizes coherent same-dim outlier attacks (no single dim can dominate the aggregate score because each dim is bounded in `[0, 1/α]` before the mean).

**Phase-lift saturates.** `θ = (π/2)(1 + tanh(x/3))`. Bounded in `[0, π]`, monotonic, smooth, near-identity for small `|x|`. The earlier affine lift `θ = (π/2)(1 + x)` from Probe 10 wraps the cosine around for spike values, which accidentally suppressed the attack and made the experiment uninformative. The tanh map keeps spike cosines well-defined without destroying clean signal.

**FlashAttention-style 2D shared-memory tiling.** Naive per-thread streaming hits a register/spill cliff once `d` grows beyond 64 because each thread needs `local_q[d]` register space. The kernel tiles both dimensions: `TILE_M = 32` keys at a time, `TILE_D = 64` embedding dimensions at a time, with the key block cooperatively loaded into 8 KB of shared memory and the query slice held in registers. Partial sums accumulate across the inner `d` loop, then `argmax` resolves within each `TILE_M` block. This is what lets the kernel handle `d` up to several hundred without running out of registers or shared memory.

The result: one CUDA launch produces the top-1 retrieval index, top score, projection-channel score (for ops accounting), and the per-row agreement metric.

---

## ghost_kernel.cu

Single source of truth for every CUDA kernel in the suite. Loaded once via `cupy.RawModule` by `gpu.py` and `projection_benchmark.py`. The five sections, all linked at top:

| § | Symbol(s) | Role |
|---|---|---|
| 1 | `clipped_log_pair`, `projection_channel`, `geometry_channel` | Shared device helpers used by everything below |
| 2 | `ghost_rank_k_matmul`, `ghost_rank_k_matmul_batch` | Custom T1 kernel — matches cuBLAS-on-lifted-representation to fp32 noise, used for the rank-K matmul baseline |
| 3 | `ghost_projection` | N-matrix projection: applies the projection channel to a full `(N, matrix_size, matrix_size)` output. The 4×4 case is `N=1, matrix_size=4` |
| 4 | `tied_streaming_perdim`, `tied_materialize_perdim` | Headline kernel (streaming) and its diagnostic sibling (materializes the full `N×M` score matrices for correctness checks) |
| 5 | `ghost_ghz_sample` | GHZ closed-form sampler used by `gpu.py` to produce noiseless bases; three `curand_uniform` draws per shot |

Reasons for one file instead of five:

- `clipped_log_pair` was duplicated between an earlier `ghost_projection.cu` and the embedded benchmark kernel. Now it appears once and is `__device__ inline` so both Section 3 and Section 4 share it without indirect calls.
- `projection_channel` and `geometry_channel` are shared device functions used by both the standalone projection (Section 3) and the tied kernels (Section 4). One definition, two callers.
- Section 2's rank-K kernel is in here even though it's not part of the G_M pipeline because Probe 6 (3-way convergence) needs it to demonstrate that cuBLAS and the custom T1 kernel agree on T1 to fp32 noise — that comparison is part of how the suite establishes its bona fides on the trigonometric operator family before claiming anything novel about G_M.

The legacy `projection_4x4_legacy` name from earlier benchmark iterations is dropped in favor of the unified `ghost_projection` entry point.

---

## The Python layer

`ghost_oracle/projection_benchmark.py` is the only Python file the user runs to see the headline result. Its responsibilities:

1. **Auto-find or accept bases.** `auto_find_base("qpu")` finds `data/job_*.npz`; `auto_find_base("gpu")` finds either `noiseless_base_*.npz` or `ghost_oracle_gpu_*.npz`.
2. **Compress bases to 18-int per-tile bucket counts.** `build_bucket_counts()` does the `(ctrl, ghost4)` → `(3, 3, 2)` collapse described above.
3. **Pick the representative tile.** `representative_tile()` selects the most balanced one.
4. **Phase-lift the input embeddings.** `phase_lift_perdim()` is the production tanh map; `analytical_G_M_perdim()` is the numpy reference for correctness checks.
5. **Stage queries and keys for the attack.** `make_attacked_jittered_embeddings()` produces the Probe 10.1 setup: gaussian keys, queries = keys + jitter, fraction of keys spiked on a shared dim at fixed magnitude.
6. **Launch the kernel.** `tied_streaming_perdim_run()` handles cupy buffer prep, kernel invocation, synchronization.
7. **Report.** Three-axis comparison vs cuBLAS dot-product: speed (entries/s, GFLOPS), accuracy (top-1 under the attack), efficiency (ops per correct retrieval). Plus the agreement metric — the certificate.

There is one separate correctness check that runs by default: `run_correctness()` invokes `tied_materialize_perdim` instead of the streaming kernel, materializes the full `N × M` per-dim score matrices, and compares to the analytical numpy reference. That's how the geometry kernel's `2.3e-5 MAE vs numpy` is established at every run.

The CLI is deliberately small. `--sweep {None, attention, extreme}` picks the shape regime, `--d` sets the embedding dimension, `--jitter / --attack-fraction / --magnitude` reproduce or modify Probe 10.1's setup. There is no benchmark mode that doesn't also run the correctness check; the certificate is mandatory.

---

## The probes layer

The probes are not on the production data path. They live in `probes/` and read the same bases, but each one is an end-to-end script that tests a specific claim and prints its own report. Probe 4 also produces a base (the noiseless reference). Probe 6 is the only probe that materializes a full scoring matrix as part of its three-way convergence demonstration; the rest stream like the production runtime.

Three constants are shared suite-wide and live in every probe's CONFIG block plus `qpu.py`:

| Constant | Value | Source |
|---|---|---|
| `ANGLE_SCALE` | 1.05 | Suite-wide; geometric scaling that keeps angles inside the smooth region of the phase-lift while exploiting the saturation near `π/2` |
| `ALPHA_NORM` | 0.9127 | Suite-wide; normalization such that `G_M` peaks at 1 over the expected angle range |
| `NUM_TILES` | 12 (probes), 16 (production) | Historical 12-tile runs vs. current 4×4 = 16-tile QPU jobs |

The 12-vs-16 split is the only nontrivial mismatch. Original probe runs used a 12-tile layout (3 rows × 4 cols, last 4 cells empty) when the IBM hardware available at the time couldn't reliably hold a 4×4 cluster. The current generation handles 16 tiles. Every probe takes `--num-tiles` to override, and the historical context block in every probe header notes which configuration the reported numbers came from.

---

## Data flow, end to end

The full pipeline from quantum hardware to retrieval index:

```
IBM Runtime backend                                  GPU sampler
    │                                                     │
    ▼                                                     ▼
 qpu.py submits circuit                              gpu.py invokes
    │                                                ghost_ghz_sample
    ▼                                                     │
 (job ID)                                                 ▼
    │                                                noiseless base
    ▼                                                     │
 dump.py extracts                                         │
    │                                                     │
    ▼                                                     ▼
 data/job_*.npz   ◄────  shared schema  ────►  data/ghost_oracle_gpu_*.npz
    │                                                     │
    └─────────────────────────┬───────────────────────────┘
                              ▼
                  projection_benchmark.py
                              │
                  load_base + build_bucket_counts
                              │
                              ▼
                  per-tile 18-int counts
                              │
            phase_lift_perdim(query, key embeddings)
                              │
                              ▼
                  tied_streaming_perdim kernel
                  ─────────────────────────────
                  projection channel:  bucket reweighting → certificate
                  geometry channel  :  closed form        → argmax
                  per-dim aggregation:  mean over d       → robustness
                  fused argmax       :  no N×M matrix     → O(N) memory
                              │
                              ▼
                  out_idx, out_score, out_proj_score, out_agreement
                  ─────────────────────────────
                  top-1 retrieval   physical certificate
```

Every arrow in this diagram is exercised by every default run of `projection_benchmark.py`. Removing any one of them breaks a claim:

- Without the shared base schema, the QPU and GPU paths fork and the certificate stops being meaningful (different operators).
- Without bucket compression, the projection channel runs in `O(n_shots)` per evaluation and the kernel no longer fits.
- Without log-clipping, the projection channel produces NaNs on the angle extremes where the bucket counts are near zero.
- Without per-dim aggregation, single-dim outlier attacks pass through the operator (Probe 10's failure mode).
- Without the tanh phase-lift, spike values wrap the cosine and accidentally cancel themselves (also Probe 10's failure mode).
- Without fused argmax, the score matrix materializes and 500× VRAM advantage collapses to ~1× and 65536² OOMs.
- Without the projection channel, the geometry channel is unfalsifiable; we'd be claiming the QPU implements G_M without any data backing it.

The architecture is the set of decisions that keep all seven arrows alive at once. The probes establish that each individual claim along this path is real; the benchmark establishes that they hold simultaneously under load, at scale, against a real adversarial task.

---

## Why not …

A few designs that look tempting at first and aren't.

**Why not just use cos·outer-product?** G_M correlates with `cos a · cos b` at 0.9992. They're almost the same operator on random angles. The 0.0008 difference is the `sqrt((1 + ·)/2)` saturation, and *that's where the structural robustness lives*. Strip the sqrt and the operator is unbounded above; under a coherent same-dim attack, one big dim dominates the mean and per-dim aggregation stops protecting the retrieval. The sqrt is load-bearing.

**Why not cache the geometry result and use only the projection channel?** Because then there's no certificate. The geometry kernel doesn't know whether the projection channel agrees; it just computes a closed form. The tied design's value is in the *coupled* evaluation — every reported retrieval is shadowed by a per-row check that the operator the kernel argmaxed against is the operator the physical hardware implements, modulo a quantitatively bounded gap.

**Why not use the projection channel for the argmax instead of the geometry channel?** Two reasons. First, the projection channel is `O(18)` per evaluation in operations but has roughly 10× the constant factor of the geometry channel in practice; on the inner loop it would dominate the kernel time. Second, the projection channel is shot-noise limited, so its argmax has higher variance than the geometry channel's. The geometry channel argmax is the sharp signal; the projection channel certificate is the truth check. Both jobs are necessary, and assigning them to the right channels is what makes the kernel fast.

**Why not split the kernel into two passes — geometry first, projection second only for the top-K?** Because the agreement metric is the *whole certificate*. Computing it only on the argmax leaves the bulk of the score matrix uncertified. The full per-row mean over `M` is what reproduces the probe 7–8 channel error measurement at scale; restricting it to top-K would let the QPU be wildly off everywhere except where the argmax already pointed.

**Why not aggregate per-dim G_M with something other than the mean — e.g., median, trimmed mean?** Tried various aggregations during the benchmark trajectory. The mean is what survived: it's the only aggregator that's `O(1)` to update inside the streaming loop without buffering per-dim scores, and it's the only one whose robustness behavior is straightforward to argue from first principles (each dim is bounded in `[0, 1/α]`; the mean is bounded by the per-dim bound divided by `d`). Median would defend better against asymmetric attacks but at the cost of either materializing the per-dim vector or accepting an approximate streaming median.

---

## Pointers

- **The math:** `docs/math.md` — derivations of T1, T2, T3, G_M, and the equivalence at machine precision.
- **The trajectory:** `PROCESS_RECORD.md` — how the architecture got here, including the four failed benchmark iterations before the tied design landed.
- **Known issues:** `docs/known_issues.md` — limits of the current implementation, including the Probe 8.2 alternation optimizer (broken, marked) and the Probe 8.4 base-2 Benford column (non-informative).
- **Per-probe context:** `probes/README.md` — what each numbered probe contributes to the arc.