# Architecture

Architecture for the `G_M` operator package.

`G_M` is the **Generalized Metric** channel. It is the first completed operator package in the Ghost Oracle Suite Converger architecture.

This document replaces the older architecture framing that treated `G_M` mainly as a standalone projection benchmark. The new framing is:

```text
G_M is a finished operator package for this version.
G_M is one channel in the larger Converger roadmap.
G_M uses the standard geo / gproj / qproj substrate pattern.
G_M claims are made only through the benchmark runner and controls.
```

The math is in:

```text
docs/math.md
```

The process record is in:

```text
PROCESS_RECORD.md
```

The future Converger roadmap is in:

```text
docs/future_directions_whitepaper.md
```

This architecture document explains how the finished `G_M` operator is wired together, what each file is responsible for, what counts as a valid benchmark claim, and how the package fits into the larger Converger direction without inflating the current result beyond what the benchmark supports.

---

## 1. Architectural status

`G_M` is complete for this version.

That means the package now has:

```text
1. a closed-form geometry path,
2. a noiseless GPU projection-base path,
3. a real QPU shot-count projection-base path,
4. a canonical benchmark runner,
5. semantic scale sweeps,
6. load-bearing controls,
7. separation-spectrum controls,
8. saved benchmark output,
9. documented math,
10. documented known limits.
```

The current package is not a placeholder for future work. It is a finished operator implementation with a bounded claim.

The bounded claim is:

```text
G_M is a bounded, calibrated, substrate-comparable generalized metric.
```

The claim is not:

```text
G_M replaces dot-product attention universally.
G_M proves quantum advantage.
G_M makes QPU shot reconstruction faster than cuBLAS.
G_M is distribution-shift-proof without recalibration.
```

The architecture exists to keep that distinction clear.

---

## 2. Converger framing

The larger Ghost Oracle Suite roadmap frames ghost-channel operators as components of a transformer-adjacent Converger.

A transformer answers:

```text
What is the next useful representation, token, action, or score?
```

A Converger asks:

```text
What hidden structure exists around that score, and does it survive controls?
```

Within that system, `G_M` is the metric projection component.

```text
Converger operator stack
├── G_M        Generalized Metric channel
├── S_M        Syndrome Metric channel
├── T_S        Stress channel
├── I_M.local  Local Interaction channel
├── I_M.field  Field Interaction channel
├── F_M        Fractal Expansion channel
└── D_M        Dimensional Metric channel
```

`G_M` is not the entire Converger.

`G_M` is the first fully benchmarked operator package in that stack.

Its role is:

```text
bounded similarity
retrieval structure
ranking behavior
projection-vs-geometry substrate comparison
```

Transformer-adjacent inputs include:

```text
embeddings
attention scores
retrieval candidates
key/query/value geometry
memory candidate sets
```

The `G_M` package does not replace those transformer components. It measures them through a bounded generalized metric and compares that measurement across substrates.

---

## 3. Standard operator package pattern

The future Converger architecture uses the same package shape for every operator:

```text
operator/
├── operator_gpu_generate.py
├── operator_qpu_generate.py
└── operator_benchmark.py
```

For `G_M`, the current package uses:

```text
G_M/
├── g_m_gpu_generate.py
├── g_m_qpu_generate.py
└── g_m_benchmark.py
```

Supporting directories:

```text
G_M/
├── data/
├── docs/
├── examples/
├── kernels/
└── probes/
```

The current main entry points are:

```bash
python g_m_benchmark.py
python g_m_benchmark.py --sweep ALL
python g_m_benchmark.py --probe
```

Base generation:

```bash
python g_m_gpu_generate.py
python g_m_qpu_generate.py
```

Legacy examples remain in the repo:

```bash
python examples/final_benchmark_5way.py
python examples/auto_oracle.py
python examples/projection_benchmark.py
```

Those examples are retained for continuity and process history. They are not the current source of benchmark claims.

Current benchmark claims should come from:

```text
g_m_benchmark.py
```

and from the saved output:

```text
data/G_M_final_benchmark.json
```

---

## 4. Standard substrate paths

`G_M` follows the standard three-substrate pattern used by the Converger roadmap.

```text
G_M^{geo}
G_M^{gproj}
G_M^{qproj}
```

In GitHub-safe plain text:

```text
G_M_geo    = closed-form geometry path
G_M_gproj  = GPU / noiseless projection-base path
G_M_qproj  = QPU / hardware shot-count projection-base path
```

### 4.1 Geometry path

The geometry path is the analytical reference.

It evaluates the closed-form generalized metric directly:

```text
G_M(a,b) = sqrt((1 + cos(a) cos(b)) / 2) / alpha
```

For vector inputs, the benchmark applies the production phase lift per dimension, evaluates the bounded coordinate metric, and averages across dimensions.

The geometry path is substrate-agnostic.

It is the clean mathematical ceiling.

### 4.2 GPU projection path

The GPU projection path uses a noiseless classical projection base.

```text
G_M_gproj(X; B_g)
```

where `B_g` is a GPU-generated base file.

The GPU base is not an arbitrary baseline. It is a noiseless classical implementation of the same projection-circuit bucket structure used by the QPU path.

Its purpose is to separate:

```text
projection-circuit behavior
```

from:

```text
hardware noise and drift
```

### 4.3 QPU projection path

The QPU projection path uses real shot-count buckets from IBM Runtime jobs.

```text
G_M_qproj(X; B_q)
```

where `B_q` is a dumped QPU base file.

The QPU path is the hardware-calibrated projection substrate.

It is not used to claim raw throughput superiority. It is used to test whether the same projection-style operator survives physical shot-count reconstruction and controls.

---

## 5. Bases

A **base** is a `.npz` file containing measurement data for tiled projection circuits.

The shared base schema is:

```text
job_id        : str
num_tiles     : int
ctrl_tile{t}  : uint8, shape (n_shots,)
ghost_tile{t} : uint8, shape (n_shots, 4)
```

The same schema is used for:

```text
real QPU bases
noiseless GPU bases
older compatible reference bases
```

This shared schema is load-bearing.

It allows the same projection code to consume QPU and GPU bases without changing the operator.

Generated QPU base files use the pattern:

```text
data/job_<JOB_ID>.npz
```

Generated GPU base files use the pattern:

```text
data/ghost_oracle_gpu_<...>.npz
```

The benchmark output file is:

```text
data/G_M_final_benchmark.json
```

---

## 6. Bucket compression

Raw base files contain per-shot measurements.

The projection channel does not need shot order. It only needs bucket counts over:

```text
a_bucket in {0, 1, 2}
b_bucket in {0, 1, 2}
ctrl     in {0, 1}
```

That gives:

```text
3 x 3 x 2 = 18 buckets
```

So each tile compresses to:

```text
counts18
```

This compression happens once at load time.

After compression, the hot projection path reads 18 integers per tile instead of thousands of raw shots.

This is the practical bridge from physical shot records to a reusable projection operator.

---

## 7. Geometry channel and projection channel

The `G_M` benchmark evaluates the generalized metric through two linked channels.

```text
geometry channel    = closed-form reference
projection channel  = calibrated bucket-count reconstruction
```

### 7.1 Geometry channel

The geometry channel evaluates:

```text
G_M(a,b) = sqrt((1 + cos(a) cos(b)) / 2) / alpha
```

This is the analytical reference.

It is fast, deterministic, and substrate-independent.

### 7.2 Projection channel

The projection channel estimates the same operator from calibrated bucket counts.

It uses importance reweighting to estimate what the control probability would have been at a target angle pair, then converts that estimate into the generalized metric.

In plain text:

```text
base sampled at        : (orig_a, orig_b)
target evaluation at   : (a, b)
projection estimate    : reweighted bucket-count estimate
```

The projection channel is substrate-specific because its input buckets come from a specific base:

```text
GPU/noiseless base
QPU/hardware base
```

### 7.3 Why both channels exist

The geometry channel gives the sharp analytical score.

The projection channel gives the substrate-backed readout.

Together they answer:

```text
Does the substrate-backed projection channel agree with the closed-form metric
on the same task?
```

That is the core architecture.

---

## 8. Agreement metric

The tied benchmark reports an agreement value:

```text
agreement = mean |geometry - projection| per query
```

Expanded:

```text
For each query:
    compare geometry score and projection score against all keys
    average the absolute difference over keys

Then:
    average over queries
```

Agreement is not a speed metric.

Agreement is not a quantum-advantage metric.

Agreement is not a dense-attention metric.

Agreement is a substrate-quality readout:

```text
How far does this projection substrate drift from the closed-form geometry
channel on the same task?
```

The latest capstone run reports example agreement ranges:

```text
QPU bases : 0.0049 - 0.0888
GPU bases : 0.0104 - 0.1049
```

Those values depend on selected tile, mask, threshold, bucket structure, and task distribution.

The correct read is:

```text
agreement = projection-vs-geometry substrate comparison
```

---

## 9. Current CUDA architecture

The core CUDA sources are:

```text
kernels/ghost_kernel.cu
kernels/megakernels_2d.cu
```

The current benchmark compiles one CUDA module from both sources.

The production path uses the 2D megakernels:

```text
geo_megakernel_2d
proj_megakernel_2d
tied_megakernel_2d
```

### 9.1 `geo_megakernel_2d`

Role:

```text
closed-form geometry scoring
```

Consumes raw embeddings, applies the production lift/cosine path inside the kernel, and emits geometry scores.

### 9.2 `proj_megakernel_2d`

Role:

```text
projection-channel scoring
```

Consumes raw embeddings, calibrated bucket counts, tile selection, mask selection, and projection parameters.

Emits projection scores.

### 9.3 `tied_megakernel_2d`

Role:

```text
geometry + projection + agreement
```

Runs both channels in one tied pass and reports agreement.

This is the verification path that binds analytical metric behavior to substrate-backed projection behavior.

---

## 10. Raw embeddings and phase lift

The current 2D megakernel path consumes raw embeddings.

The phase lift and cosine are folded into the kernel transfer / evaluation path.

The production phase lift is conceptually:

```text
theta(x) = (pi / 2) * (1 + tanh(x / 3))
```

This map is:

```text
bounded
smooth near zero
saturating at fixed endpoints
```

That matters because `G_M` is intended to bound per-dimension influence.

A large shared coordinate should saturate, not explode.

The bounded per-dimension path is:

```text
raw component
  -> phase lift
  -> cosine endpoint
  -> bounded G_M coordinate
  -> mean over dimensions
```

This is the mechanism behind the same-dimension outlier resistance.

---

## 11. Per-dimension aggregation

`G_M` is evaluated per dimension and averaged.

Plainly:

```text
for each query/key pair:
    for each dimension:
        compute bounded G_M coordinate
    average coordinates over dimension
```

This is not incidental. It is the structural reason coherent single-dimension outliers cannot dominate the metric the way they can dominate dot-product attention.

Dot-product attention can be dominated by:

```text
one large shared coordinate
```

because the dot product is unbounded.

`G_M` bounds each coordinate before aggregation.

The maximum influence of one coordinate shrinks with dimension because it is averaged into the whole score.

This is the core architecture-level difference between `G_M` and ordinary dense dot-product attention.

---

## 12. Calibration

The current benchmark uses two calibration objectives because verification and semantic retrieval are different tasks.

### 12.1 Verification calibration

Function:

```text
calibrate_candidates()
```

Purpose:

```text
select tile / mask / threshold for five-way verification
```

It scores candidate combinations:

```text
(tile, mask, threshold)
```

using the Flash-Squelch certificate.

The selection rule is:

```text
clean := median spike fraction <= SPIKE_TOLERANCE
rank  := clean first, then signal fraction descending
```

This calibration matches the five-way verification metric.

### 12.2 Sweep / probe calibration

Function:

```text
calibrate_recall1()
```

Purpose:

```text
select tile / mask for semantic sweeps and probes
```

It ranks:

```text
(tile, mask)
```

by the same recall@1 argmax retrieval objective used by the sweep.

A component must retrieve to be selected.

If a base has no component clearing the floor:

```text
CALIB_MIN_R1 = 0.50
```

that base is excluded from sweep/probe evaluation.

This prevents the benchmark from including a dead component as if it were a valid projection path.

---

## 13. Bucket masks

The projection estimator can zero selected buckets before reweighting.

This is called a mask.

Masks are not changes to the `G_M` operator.

They are calibration choices for the projection estimator.

Current mask candidates include:

```text
M1: Baseline all buckets
M2: Drop buckets 4-8
M3: Anti-pillars
M4: Drop (0,1)(1,0)
M5: Drop (1,2)(2,1)
M6: Drop pillars
M7: Pure core
M8: Mirror core
```

The key rule is:

```text
mask selection is calibration-dependent
```

There is no universal golden mask.

The current benchmark handles this by calibrating mask choice against the metric it actually reports.

---

## 14. Canonical data flow

End-to-end `G_M` package flow:

```text
QPU hardware path
    g_m_qpu_generate.py
        -> IBM Runtime job
        -> dumped base
        -> data/job_<JOB_ID>.npz

GPU/noiseless path
    g_m_gpu_generate.py
        -> noiseless projection base
        -> data/ghost_oracle_gpu_<...>.npz

Shared benchmark path
    g_m_benchmark.py
        -> load bases
        -> compress to counts18
        -> calibrate components
        -> run geo / gproj / qproj
        -> run controls
        -> save result JSON
```

Inside the benchmark:

```text
raw embeddings
    -> 2D megakernel path
    -> geometry score
    -> projection score
    -> agreement metric
    -> retrieval metrics
    -> controls
```

The current architecture intentionally keeps data movement simple:

```text
base files in data/
CUDA kernels in kernels/
legacy examples in examples/
research probes in probes/
current claims from g_m_benchmark.py
```

---

## 15. Current benchmark stages

The canonical benchmark has three major stages.

```text
1. five-way verification
2. semantic scale sweep
3. negative control probes
```

### 15.1 Five-way verification

The five-way verification compares:

```text
CUBLAS
TIED
GEO
QPROJ
GPROJ
```

Path meanings:

```text
CUBLAS = standard dense dot-product attention control
TIED   = geometry + projection + agreement
GEO    = closed-form G_M geometry
QPROJ  = projection from QPU shot-count buckets
GPROJ  = projection from noiseless GPU shot-count buckets
```

Current verification configuration:

```text
N = 4096
d = 64
jitter = 0.3
attack fraction = 0.05
attack magnitude = 200.0
squelch power = 256.0
base files = 10
```

Current summary:

```text
CUBLAS top-1              : 57.9%
CUBLAS spike fraction     : 0.426528

GEO mean top-1            : 100.0%

QPROJ mean top-1          : 100.0%
QPROJ mean signal         : 100.0%
QPROJ mean spike fraction : 0.0498

GPROJ mean top-1          : 99.7%
GPROJ mean signal         : 99.6%
GPROJ mean spike fraction : 0.0502
```

The read:

```text
Under coherent same-dimension attack, standard dot-product attention
concentrates on the attack dimension, while bounded G_M geometry/projection
preserves retrieval.
```

The non-claim:

```text
This does not say QPU projection is faster than cuBLAS.
```

cuBLAS is the dense GEMM throughput control and is much faster as raw dense attention.

### 15.2 Semantic scale sweep

The sweep evaluates:

```text
Recall@1
Recall@5
Recall@10
MRR
```

Settings:

```text
d = 1024
TOP_K = [1, 5, 10]
strategy = mixed projection components
```

Current sweep summary:

```text
SMALL:
    cosine R@1 = 99.12%
    GEO R@1    = 100.00%
    QPU proj   = 99.90% - 100.00%
    usable GPU = 97.27% - 100.00%

MEDIUM:
    cosine R@1 = 96.88%
    GEO R@1    = 100.00%
    QPU proj   = 99.41% - 100.00%
    usable GPU = 40.33% - 100.00%

LARGE:
    cosine R@1 = 95.31%
    GEO R@1    = 100.00%
    QPU proj   = 91.99% - 100.00%
    usable GPU = 1.46% - 100.00%
```

One GPU base was excluded from sweep/probe evaluation because no component cleared the 50% recall@1 calibration floor.

The read:

```text
GEO is the clean mathematical ceiling.
Projection performance depends on calibrated bucket structure and distribution match.
```

### 15.3 Probe A: load-bearing shot-count control

Probe A asks whether projection is actually using the calibrated shot buckets.

It compares:

```text
real counts
permuted counts
uniformized counts
```

Current result pattern:

```text
real counts       : high retrieval
permuted counts   : 0.00% R@1
uniformized counts: 0.00% R@1
```

The read:

```text
real >> permuted
real >> uniformized
```

Therefore:

```text
the calibrated bucket structure is load-bearing
```

Projection is not silently collapsing to geometry.

### 15.4 Probe B: separation spectrum

Probe B asks whether the task is too easy.

It compares:

```text
cosine
GEO
PROJ
```

across attack and noise axes.

B1 varies outlier magnitude while holding noise fixed.

At magnitude zero:

```text
cosine = 100.00%
GEO    = 100.00%
PROJ   = 100.00%
```

As magnitude rises:

```text
cosine falls to 94.63%
GEO remains 100.00%
PROJ remains about 99.32%
```

B2 varies noise while holding outlier magnitude fixed.

Projection degrades as noise moves away from the calibration distribution:

```text
noise 0.05 -> PROJ 99.51%
noise 0.10 -> PROJ 99.32%
noise 0.15 -> PROJ 98.14%
noise 0.20 -> PROJ 95.02%
noise 0.25 -> PROJ 88.18%
noise 0.30 -> PROJ 74.22%
```

The read:

```text
projection is calibrated, not magic
```

Distribution shift matters.

That is a feature of the benchmark discipline, not a failure of the architecture.

---

## 16. Valid claim boundary

The `G_M` architecture supports the following claims.

### Supported

```text
G_M has a closed-form geometry channel.
G_M has a GPU/noiseless projection-base path.
G_M has a QPU/hardware shot-count projection-base path.
The same benchmark compares geo / gproj / qproj.
The projection path uses load-bearing bucket structure.
Destroying bucket structure destroys projection retrieval.
The geometry path resists coherent same-dimension outlier attack.
Calibrated projection can reproduce that retrieval behavior.
Agreement measures substrate drift from closed-form geometry.
```

### Not supported

```text
G_M is a universal replacement for attention.
G_M proves quantum advantage.
G_M is faster than cuBLAS for dense attention.
One mask works universally.
Projection works out-of-distribution without recalibration.
All future operators are proven because G_M works.
```

This is the claims discipline the Converger roadmap requires.

---

## 17. Why the architecture is finished for this version

`G_M` is finished for this version because every required operator-package element exists.

```text
operator math               : complete
geometry path               : complete
GPU base generator          : complete
QPU base generator          : complete
canonical benchmark runner  : complete
CUDA kernels                : complete
calibration path            : complete
negative controls           : complete
saved benchmark output      : complete
known limits                : documented
```

The next work is not to keep mutating `G_M` indefinitely.

The next work is to use `G_M` as the completed pattern for the remaining Converger operators:

```text
S_M
T_S
I_M.local
I_M.field
F_M
D_M
```

Each future operator should follow the same discipline:

```text
define the operator
define geo / gproj / qproj
generate bases
benchmark under controls
scramble the channel
measure what survives
make only bounded claims
```

---

## 18. Why not...

### Why not call this a transformer replacement?

Because it is not one.

`G_M` is a measurement and scoring channel. It can sit beside transformer embeddings, attention scores, and retrieval candidates. It does not replace the primary model.

The Converger framing is:

```text
transformer predicts
converger measures
benchmark controls
operator survives or fails
```

### Why not claim quantum advantage?

Because the benchmark does not show that.

The QPU path is valuable because it provides a physical shot-count substrate for the same projection operator. The claim is substrate comparability and calibrated hardware readout, not quantum speedup.

### Why not use only geometry?

Because geometry alone gives no substrate certificate.

The projection channel tells whether the calibrated shot-count substrate can reproduce the behavior. The tied architecture keeps that measurement attached to the geometry path.

### Why not use only projection?

Because projection is shot-limited and calibration-dependent.

Geometry is the sharp mathematical reference. Projection is the substrate readout. The architecture needs both.

### Why not trust projection without controls?

Because the controls are what make the claim defensible.

The load-bearing test shows that real bucket structure matters. The separation spectrum shows the task is not merely trivial. Without those controls, the benchmark would be much weaker.

---

## 19. File responsibilities

```text
README.md
    Human-facing summary and current benchmark claims.

docs/math.md
    Derivation of T1, T2, T3, G_M, projection identity, agreement metric.

docs/architecture.md
    This document. System design for the finished G_M operator package.

docs/known_issues.md
    Current limitations and failure modes.

docs/future_directions_whitepaper.md
    Larger Converger roadmap.

g_m_benchmark.py
    Canonical benchmark runner for G_M.

g_m_gpu_generate.py
    Noiseless GPU projection-base generator.

g_m_qpu_generate.py
    QPU projection-base generator / submission path.

kernels/ghost_kernel.cu
    Shared CUDA helpers and legacy kernels.

kernels/megakernels_2d.cu
    Current 2D tiled geometry/projection/tied megakernels.

data/
    Curated benchmark JSON and projection base files.

examples/
    Legacy examples retained for continuity.

probes/
    Research trajectory and forensic record.
```

---

## 20. Summary

`G_M` is the completed Generalized Metric operator package for this version of Ghost Oracle Suite.

It demonstrates the operator-package pattern the future Converger architecture will use:

```text
geo
gproj
qproj
benchmark
controls
bounded claims
```

The current architecture says:

```text
G_M geometry retrieves.
G_M projection retrieves when calibrated.
G_M projection fails when bucket structure is destroyed.
G_M agreement measures substrate drift.
cuBLAS remains the dense throughput control.
```

The larger roadmap says:

```text
Use this completed pattern to build the remaining ghost-channel operators.
```

That is the architectural handoff.

The process is the process:

```text
freeze the record
build controls
scramble the channel
compare substrates
measure what survives
```
