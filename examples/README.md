# `examples/`

Two runnable scripts that exercise the operator on workloads beyond the headline benchmark. Both are self-contained — point them at the sample bases in `data/` and they produce reportable numbers without further setup.

```bash
# From the repo root
python examples/parameter_ablation.py
python examples/ghost_oracle_ai_retrieval_probe_v1.py
```

Treat these as "what the operator does on your actual workload." The probes in `probes/` characterize the operator structurally; the projection benchmark in `ghost_oracle/projection_benchmark.py` demonstrates the headline attention-robustness claim; these examples translate G_M into application contexts (parameter sensitivity, semantic retrieval) where a practitioner can read the numbers and decide whether the operator fits their problem.

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
|---|---|---|---|
| 8 | 12.6% | 27.1% | +14.5 |
| 16 | 42.2% | 83.4% | +41.2 |
| 32 | 62.2% | 99.9% | +37.7 |
| 64 | 79.3% | 100.0% | +20.7 |
| 128 | 97.6% | 100.0% | +2.4 |
| 256 | 100.0% | 100.0% | 0.0 |

The operating range where G_M cleanly wins is roughly `d ∈ [16, 64]`. Above d=128 the attack is too weak relative to the embedding dimension to engage the softmax bottleneck, and cuBLAS catches up. This is a sharper claim than "G_M is robust" — it's "G_M is robust in the regime where the attack actually engages the softmax bottleneck."

**Jitter sweep — robustness compounds with noise.**

The cuBLAS top-1 drops from 81% at jitter 0 to 70% at jitter 1.0 under the fixed attack. G_M stays at 100% until jitter 1.0, where it dips to 99.5%. The gap *widens* with jitter — the two failure modes (jitter and attack) interact constructively against DP, not against G_M.

**Magnitude sweep — the softmax cliff diagram.**

cuBLAS goes 100% → 100% → 97.8% → 79.3% → 65.6% → 52.3% as magnitude climbs 0 → 5 → 20 → 50 → 100 → 500. G_M stays at 100% throughout. The shape of the cuBLAS curve is the softmax bottleneck mechanism made explicit: magnitude 5 doesn't move it, magnitude 20 barely (97.8%), magnitude 50 is where it cracks, magnitude 500 settles into a floor around 52%.

**Fraction sweep — one spike is enough.**

cuBLAS sits at ~79-80% across attack fractions from 1% to 50%; G_M is 100% throughout. The attack doesn't scale with how many keys are corrupted — one same-dim spike is essentially as bad for DP as 500 spikes, because softmax converges fast onto the dominant column. The cleanest demonstration of the bottleneck mechanism in the suite.

### How it's wired

The script is a thin wrapper around `projection_benchmark.py`'s machinery — same kernel, same base loading, same `make_attacked_jittered_embeddings`. Only the sweep harness is new. If you want to add a fifth axis (representative-tile, alpha, anything), copy the pattern of `run_ablation()`.

### Reproducibility note

Each sweep value reuses seed 42, so the underlying random keys and the chosen `k_bad` are held constant across the sweep. That makes the per-row deltas comparable to their neighbors but means the sweeps aren't independent runs — they're "the same experiment with one parameter varied." A worst-case search over `k_bad` would be a different (and harder) experiment.

---

## `ghost_oracle_ai_retrieval_probe_v1.py` — semantic retrieval at scale

Tests whether G_M behaves like a useful bounded-memory retrieval mechanism against cosine similarity on clustered embeddings. Three sweeps:

| name | DB size (M) | queries (N) | noise | outlier frac | outlier mag |
|---|---|---|---|---|---|
| SMALL | 50,000 | 1,024 | 0.08 | 1% | 40 |
| MEDIUM | 250,000 | 1,024 | 0.12 | 3% | 60 |
| LARGE | 1,000,000 | 1,024 | 0.18 | 5% | 100 |

Both sides stream under a 512 MB VRAM budget so the comparison is fair: same memory constraint, different operator.

### What it shows

**Recall@1 / Recall@5 / Recall@10, MRR, time, VRAM** for cosine and G_M on identical inputs. Headline results:

| | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| **SMALL 50K** | | | | |
| cosine | 98.6% | 98.6% | 98.6% | 0.986 |
| G_M | 99.9% | 100.0% | 100.0% | 1.000 |
| **MEDIUM 250K** | | | | |
| cosine | 96.2% | 96.2% | 96.2% | 0.962 |
| G_M | 99.6% | 99.8% | 99.8% | 0.997 |
| **LARGE 1M** | | | | |
| cosine | 94.9% | 94.9% | 94.9% | 0.949 |
| G_M | 96.0% | 99.3% | 99.8% | 0.975 |

The most interesting line is the **cosine ceiling**. At every scale, cosine's Recall@5 and Recall@10 are identical to its Recall@1. That means when cosine is wrong, the true match isn't anywhere in its top-10 — it's deep in the tail. G_M's Recall@10 is consistently above its Recall@1, which means when G_M is wrong on rank-1, the true match is almost always at rank 2 or 3.

That's a real architectural finding for two-stage retrieval (RAG, vector databases, anything with a re-ranker downstream). G_M giving you the true match at rank 2 when it loses is vastly more recoverable than cosine giving you the true match at rank 47 when it loses. The re-ranker fixes rank-2 trivially; rank-47 it can't.

### Implementation notes

The script uses its own `FUSED_MEGAKERNEL` rather than reusing `tied_streaming_perdim` from `ghost_kernel.cu`. This is by choice — the retrieval workload has different shape constraints (very large M, modest d, full top-K not just argmax) and the megakernel materializes the chunked score matrix before extracting top-K via `argpartition`. The production kernel's fused argmax saves memory but doesn't give you top-K. Both are valid for their use cases.

The cosine baseline streams under the same VRAM budget so the comparison is fair. Both methods peak at 512 MB; the difference is purely the operator.

### **Critical: don't L2-normalize the inputs**

Cosine similarity *requires* L2 normalization (it's part of the metric definition). G_M *requires the opposite* — its phase-lift `θ = (π/2)(1 + tanh(x/3))` is calibrated for inputs at roughly `N(0, 1)` variance. Pre-normalizing to unit L2 norm at d=128 contracts each component's variance to `~1/√d ≈ 0.088`, which squashes `tanh(x/3)` into its near-linear region and pins `cos θ` to a tiny window around 0. The result is that `G_M(Q, K) ≈ √(1/2)/α ≈ 0.7748` for *every* (Q, K) pair, regardless of similarity. Argmax over 999,999 near-identical scores becomes essentially random.

This is what an earlier version of this script did, and it produced Recall@1 = 0.59% at M=1M while the kernel itself was correct to fp32 precision. Diagnosis was painful — the kernel passed every numerical check (per-cell agreement with analytical, per-bin error analysis along M, per-mod block-boundary check, tail-vs-body) but retrieval collapsed at scale.

Two operators, two normalization conventions. Cosine wants unit norm; G_M wants unit variance. They are not interchangeable, and using the wrong one fails silently at scale rather than loudly at compile time. The current `generate_semantic_environment()` deliberately does not L2-normalize before passing embeddings into the G_M retrieval. If you adapt this script to your own data, **do not normalize the inputs to G_M to unit norm** — pass them with their natural variance.

### Where this goes next

The script is marked `v1` because two avenues are open:

1. **Real LLM embeddings.** Currently uses synthetic clustered Gaussians as a stand-in for "language-like embedding manifolds." Pulling Q/K projections from an actual open transformer (any layer's attention K/Q after the QKV projection) would graduate the retrieval claim from "synthetic" to "demonstrated on learned representations." Listed in PROCESS_RECORD Part 10 as future work, and the architecture/known-issues docs flag it as the highest-impact next experiment.

2. **Top-K with per-block partial reduction.** The current design materializes the chunked score matrix and partitions on the host (CPU `argpartition`). For database sizes beyond ~10M, this stops being viable. The fix is a two-pass kernel: each thread block maintains a register-resident top-K array as it streams keys, the block emits its partial top-K, and a second pass merges block-level top-K's into the final result. Standard pattern in large-scale retrieval; not yet implemented here.

---

## Pointers

- **`probes/README.md`** — what the operator does structurally (10 numbered probes, 1 → 10.1 trajectory).
- **`ghost_oracle/projection_benchmark.py`** — the headline attention-robustness benchmark these examples sit alongside.
- **`docs/architecture.md`** — design rationale for the tied-channel kernel both examples build on.
- **`docs/math.md`** — phase-lift derivation and why its calibrated range matters for the input-variance issue above.
- **`docs/known_issues.md`** — the open items several of these sweeps were built to answer.
