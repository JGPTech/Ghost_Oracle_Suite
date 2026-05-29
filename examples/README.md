# `examples/`

Three runnable scripts that exercise the operators on workloads beyond the headline benchmark. They are self-contained: point them at the sample bases in `data/` and they produce reportable numbers without further setup.

```bash
# From the repo root

# G_M examples
python examples/parameter_ablation.py
python examples/ghost_oracle_ai_retrieval_probe_v1.py

# S_M / TSP projector example
python examples/sm_tsp_projector_example.py
```

Treat these as "what the operator does on your actual workload." The probes in `probes/` characterize the operators structurally; the projection benchmark in `ghost_oracle/projection_benchmark.py` demonstrates the headline attention-robustness claim; these examples translate the operators into application contexts where a practitioner can read the numbers and decide whether the operator fits their problem.

The first two examples are `G_M` workloads: parameter sensitivity and semantic retrieval. The third example is an `S_M` projector testbed on a large TSP instance. It is not presented as a finished exact TSP solver; it is a CUDA-backed experiment showing how the bounded `S_M` improvement coordinate and field-deformation channel behave against a classical 2-opt control.

---

## `parameter_ablation.py` — 1D sensitivity sweeps

Locks a target shape (default 1024×1024) and runs four 1D sweeps to characterize where G_M's advantage lives:

1. **Embedding dimension** `d ∈ {8, 16, 32, 64, 128, 256}`
2. **Jitter scale** (query/key noise) `∈ {0.0, 0.1, 0.3, 0.5, 0.8, 1.0}`
3. **Attack magnitude** (same-dim coherent spike) `∈ {0.0, 5.0, 20.0, 50.0, 100.0, 500.0}`
4. **Attack fraction** (sparsity) `∈ {0.01, 0.05, 0.10, 0.20, 0.50}`

Same `tied_streaming_perdim` kernel as production; only the inputs change.

### What it shows

Each sweep produces a small table of cuBLAS DP top-1 vs G_M tied top-1 vs delta. The four findings together close out the open questions in `docs/known_issues.md` about jitter saturation, the missing clean-data baseline, and the d=64 question.

**Dimension sweep — where G_M's advantage actually lives.**

| d | cuBLAS | G_M | Δ |
|---|---:|---:|---:|
| 8 | 12.6% | 27.1% | +14.5 |
| 16 | 42.2% | 83.4% | +41.2 |
| 32 | 62.2% | 99.9% | +37.7 |
| 64 | 79.3% | 100.0% | +20.7 |
| 128 | 97.6% | 100.0% | +2.4 |
| 256 | 100.0% | 100.0% | 0.0 |

The operating range where G_M cleanly wins is roughly `d ∈ [16, 64]`. Above d=128 the attack is too weak relative to the embedding dimension to engage the softmax bottleneck, and cuBLAS catches up. This is a sharper claim than "G_M is robust" — it is "G_M is robust in the regime where the attack actually engages the softmax bottleneck."

**Jitter sweep — robustness compounds with noise.**

The cuBLAS top-1 drops from 81% at jitter 0 to 70% at jitter 1.0 under the fixed attack. G_M stays at 100% until jitter 1.0, where it dips to 99.5%. The gap *widens* with jitter: the two failure modes, jitter and attack, interact constructively against DP, not against G_M.

**Magnitude sweep — the softmax cliff diagram.**

cuBLAS goes 100% → 100% → 97.8% → 79.3% → 65.6% → 52.3% as magnitude climbs 0 → 5 → 20 → 50 → 100 → 500. G_M stays at 100% throughout. The shape of the cuBLAS curve is the softmax bottleneck mechanism made explicit: magnitude 5 does not move it, magnitude 20 barely moves it, magnitude 50 is where it cracks, and magnitude 500 settles into a floor around 52%.

**Fraction sweep — one spike is enough.**

cuBLAS sits at roughly 79–80% across attack fractions from 1% to 50%; G_M is 100% throughout. The attack does not scale with how many keys are corrupted. One same-dim spike is essentially as bad for DP as 500 spikes, because softmax converges fast onto the dominant column. This is the cleanest demonstration of the bottleneck mechanism in the suite.

### How it is wired

The script is a thin wrapper around `projection_benchmark.py`'s machinery: same kernel, same base loading, same `make_attacked_jittered_embeddings`. Only the sweep harness is new. To add a fifth axis, such as representative tile or alpha, copy the pattern of `run_ablation()`.

### Reproducibility note

Each sweep value reuses seed 42, so the underlying random keys and the chosen `k_bad` are held constant across the sweep. That makes the per-row deltas comparable to their neighbors, but the sweeps are not independent runs. They are "the same experiment with one parameter varied." A worst-case search over `k_bad` would be a different, harder experiment.

---

## `ghost_oracle_ai_retrieval_probe_v1.py` — semantic retrieval at scale

Tests whether G_M behaves like a useful bounded-memory retrieval mechanism against cosine similarity on clustered embeddings. Three sweeps:

| name | DB size (M) | queries (N) | noise | outlier frac | outlier mag |
|---|---:|---:|---:|---:|---:|
| SMALL | 50,000 | 1,024 | 0.08 | 1% | 40 |
| MEDIUM | 250,000 | 1,024 | 0.12 | 3% | 60 |
| LARGE | 1,000,000 | 1,024 | 0.18 | 5% | 100 |

Both sides stream under a 512 MB VRAM budget so the comparison is fair: same memory constraint, different operator.

### What it shows

**Recall@1 / Recall@5 / Recall@10, MRR, time, VRAM** for cosine and G_M on identical inputs. Headline results:

| | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| **SMALL 50K** | | | | |
| cosine | 98.6% | 98.6% | 98.6% | 0.986 |
| G_M | 99.9% | 100.0% | 100.0% | 1.000 |
| **MEDIUM 250K** | | | | |
| cosine | 96.2% | 96.2% | 96.2% | 0.962 |
| G_M | 99.6% | 99.8% | 99.8% | 0.997 |
| **LARGE 1M** | | | | |
| cosine | 94.9% | 94.9% | 94.9% | 0.949 |
| G_M | 96.0% | 99.3% | 99.8% | 0.975 |

The most interesting line is the **cosine ceiling**. At every scale, cosine's Recall@5 and Recall@10 are identical to its Recall@1. That means when cosine is wrong, the true match is not anywhere in its top-10; it is deep in the tail. G_M's Recall@10 is consistently above its Recall@1, which means when G_M is wrong on rank-1, the true match is almost always at rank 2 or 3.

That is an architectural finding for two-stage retrieval: RAG, vector databases, and anything with a downstream re-ranker. G_M giving the true match at rank 2 when it loses is much more recoverable than cosine giving the true match at rank 47. The re-ranker can fix rank 2 trivially; rank 47 is usually unreachable.

### Implementation notes

The script uses its own `FUSED_MEGAKERNEL` rather than reusing `tied_streaming_perdim` from `ghost_kernel.cu`. This is intentional. The retrieval workload has different shape constraints: very large M, modest d, and full top-K rather than only argmax. The megakernel materializes the chunked score matrix before extracting top-K via `argpartition`. The production kernel's fused argmax saves memory but does not give top-K. Both are valid for their use cases.

The cosine baseline streams under the same VRAM budget so the comparison is fair. Both methods peak at 512 MB; the difference is purely the operator.

### Critical: do not L2-normalize the inputs

Cosine similarity *requires* L2 normalization; it is part of the metric definition. G_M requires the opposite: its phase-lift

```text
θ = (π/2)(1 + tanh(x/3))
```

is calibrated for inputs at roughly `N(0, 1)` variance. Pre-normalizing to unit L2 norm at d=128 contracts each component's variance to `~1/√d ≈ 0.088`, which squashes `tanh(x/3)` into its near-linear region and pins `cos θ` to a tiny window around 0. The result is that `G_M(Q, K) ≈ √(1/2)/α ≈ 0.7748` for *every* `(Q, K)` pair, regardless of similarity. Argmax over 999,999 near-identical scores becomes essentially random.

This is what an earlier version of this script did, and it produced Recall@1 = 0.59% at M=1M while the kernel itself was correct to fp32 precision. Diagnosis was painful: the kernel passed every numerical check, including per-cell agreement with analytical, per-bin error analysis along M, per-mod block-boundary check, and tail-vs-body. Retrieval still collapsed at scale.

Two operators, two normalization conventions. Cosine wants unit norm; G_M wants unit variance. They are not interchangeable, and using the wrong one fails silently at scale rather than loudly at compile time. The current `generate_semantic_environment()` deliberately does not L2-normalize before passing embeddings into the G_M retrieval. If adapting this script to your own data, do not normalize the inputs to G_M to unit norm; pass them with their natural variance.

### Where this goes next

The script is marked `v1` because two avenues are open:

1. **Real LLM embeddings.** Currently uses synthetic clustered Gaussians as a stand-in for "language-like embedding manifolds." Pulling Q/K projections from an actual open transformer, such as any layer's attention K/Q after the QKV projection, would graduate the retrieval claim from synthetic to demonstrated on learned representations.

2. **Top-K with per-block partial reduction.** The current design materializes the chunked score matrix and partitions on the host with CPU `argpartition`. For database sizes beyond roughly 10M, this stops being viable. The fix is a two-pass kernel: each thread block maintains a register-resident top-K array as it streams keys, the block emits its partial top-K, and a second pass merges block-level top-Ks into the final result.

---

## `sm_tsp_projector_example.py` — S_M projector on TSP

CUDA-backed testbed for the current S_M → TSP projector path. It is designed as a "press play" example, not as a final production TSP solver.

Place the TSPLIB instance in:

```text
data/pla85900.tsp
```

Then run:

```bash
python examples/sm_tsp_projector_example.py
```

Default settings:

| parameter | value |
|---|---:|
| candidate-k | 128 |
| passes | 500 |
| max batch | 32 |
| known optimum/reference | 142,382,641 |
| field weights | 0.0001, 0.001, 0.005, 0.01, 0.05 |

The script compares three related policies:

### `delta_batch` — classical control baseline

Plain candidate 2-opt. The score is raw local improvement:

```text
score(i) = -ΔL(i)
```

This is the baseline that tells us what ordinary local search does with the same candidate list, same CUDA evaluation kernel, same batch size, and same validity constraints.

### `sm_improve_batch` — bounded projector spine

A monotonic bounded transform of local improvement:

```text
S_I(i) = 0.5 + 0.5 * tanh(-ΔL(i) / scale)
```

Because `tanh` is monotonic and `scale > 0`, this preserves local improvement ordering in the single-move case. The reason to keep it is not that it magically changes local ranking; it converts raw unbounded improvement into a stable `[0,1] coordinate that can be used by the projector, coin, amplitude weighting, or later unitary evolution.

Under batched non-overlap selection, this coordinate can still behave differently from raw delta because compression and tie structure interact with batch selection.

### `sm_field_batch` — S_M deformation channel

Build a tour-edge field from `S_I(i)` and add a local roughness/stress term:

```text
rough(i) = |S_I(i) - S_I(i-1)| + |S_I(i+1) - S_I(i)|

score(i) = S_I(i) + λ * zscore(rough(i))
```

This is the actual field-deformation test. The field weight `λ` is the projector dial:

- small `λ`: gentle deformation of the `sm_improve` spine
- medium `λ`: nontrivial field steering
- large `λ`: over-steering, usually worse

The diagnostic to watch is:

```text
mean_rank_diff_top20
```

If this is zero, the field is not changing the top move set. If it is nonzero and final length improves, the field channel is doing useful work. If it is nonzero and final length worsens, the deformation is real but mistuned.

### Representative result

One current pla85900 run with these settings produced the following ordering:

| rank | policy | field weight | final length | improvement | rankΔ |
|---:|---|---:|---:|---:|---:|
| 1 | sm_field_batch | 0.001 | 150,976,816 | 14.668% | 0.340 |
| 2 | sm_field_batch | 0.005 | 151,134,192 | 14.579% | 0.443 |
| 3 | sm_improve_batch | 0.000 | 151,226,464 | 14.527% | 0.012 |
| 4 | sm_field_batch | 0.0001 | 151,897,344 | 14.147% | 0.072 |
| 5 | sm_field_batch | 0.010 | 153,416,080 | 13.289% | 0.595 |
| 6 | sm_field_batch | 0.050 | 153,556,272 | 13.210% | 0.845 |
| 7 | delta_batch | 0.000 | 154,175,328 | 12.860% | 0.000 |

This is the shape the example is meant to expose:

```text
delta baseline
→ bounded projector coordinate improves the batch path
→ small S_M field deformation improves again
→ large field deformation over-steers
```

The important point is not that the example is a state-of-the-art TSP solver. The important point is that the projector ingredients are separable and measurable:

```text
delta_batch       = classical control
sm_improve_batch  = bounded projector spine
sm_field_batch    = tunable geometry/field perturbation
```

### Outputs

The default run writes:

```text
analysis/sm_tsp_projector_<timestamp>/
  result.json
  summary.csv
  routes.csv
  tour_delta_batch_fw0.txt
  tour_sm_improve_batch_fw0.txt
  tour_sm_field_batch_fw*.txt
```

The `result.json` includes the mini-paper framing, run arguments, tour lengths, rank deformation diagnostics, and total runtime.

### Requirements

```bash
pip install numpy scipy tqdm cupy-cuda12x
```

Use the CuPy wheel that matches your CUDA version. For CUDA 11, install the matching CuPy CUDA 11 package instead of `cupy-cuda12x`.

### Small sanity check

To run a small exact Held-Karp sanity check instead of the large TSPLIB instance:

```bash
python examples/sm_tsp_projector_example.py --validate-small --N 8 --routes 50
```

This is useful for verifying the environment and tour-validity mechanics before running pla85900.

---

## Pointers

- **`probes/README.md`** — what the operators do structurally.
- **`ghost_oracle/projection_benchmark.py`** — the headline attention-robustness benchmark these examples sit alongside.
- **`docs/architecture.md`** — design rationale for the tied-channel kernel and projection path.
- **`docs/math.md`** — phase-lift derivation and why calibrated range matters.
- **`docs/known_issues.md`** — the open items several of these sweeps were built to answer.
- **`data/README.md`** — expected data files, including TSP files used by examples.
