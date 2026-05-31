# G_M — Generalized Metric

`G_M` is the original Ghost Oracle operator family: **Generalized Metric**, formerly **Ghost Metric**.

It began as a failed Hadamard-test interpretation. The QPU circuit was first assumed to compute a textbook overlap target, but the probes showed that assumption was wrong. The useful step was to stop asking why the circuit failed to compute the intended object and instead ask what it was consistently computing.

The resulting closed-form operator is:

```text
G_M(a, b) = sqrt((1 + cos(a) cos(b)) / 2) / α
```

where `α` is the suite’s empirical normalization constant.

Current framing:

```text
G_M = bounded projection-channel / geometry-channel generalized similarity operator
```

`G_M` is implemented across three substrates:

1. analytical closed form,
2. noiseless classical GPU projection bases,
3. real QPU shot data from IBM Runtime.

The core claim is not that QPU projection is faster than classical GPU attention.

The core claim is that the same generalized projection-style operator can be expressed across mathematical, classical-sampler, and physical-shot substrates, with an agreement metric that exposes substrate quality.

---

## Quick path

The current canonical benchmark path is:

```bash
python G_M_final_benchmark.py
python G_M_final_benchmark.py --sweep ALL
python G_M_final_benchmark.py --probe
```

Useful variants:

```bash
python G_M_final_benchmark.py --N 4096
python G_M_final_benchmark.py --skip-gpu
python G_M_final_benchmark.py --skip-qpu
python G_M_final_benchmark.py --skip-verify --sweep ALL --probe
```

The older scripts are still useful as examples and continuity paths:

| Script                             | Status                 | Purpose                                              |
| ---------------------------------- | ---------------------- | ---------------------------------------------------- |
| `G_M_final_benchmark.py`           | current canonical path | Combined verification, semantic sweep, and controls. |
| `examples/final_benchmark_5way.py` | legacy example         | Earlier five-way verification path.                  |
| `examples/auto_oracle.py`          | legacy example         | Earlier calibration + semantic retrieval path.       |
| `examples/projection_benchmark.py` | legacy diagnostic      | Earlier attention benchmark / diagnostic path.       |

The QPU/GPU base-generation tools are still used for producing fresh bases:

| Script    | Purpose                                  |
| --------- | ---------------------------------------- |
| `qpu.py`  | Submit a tiled IBM Runtime QPU base job. |
| `dump.py` | Dump a completed QPU base job to `.npz`. |
| `gpu.py`  | Generate noiseless classical GPU bases.  |

---

## Operator

The simplified operator is:

```text
G_M(a, b) = sqrt((1 + cos(a) cos(b)) / 2) / α
```

The path to this form was:

```text
intended target     : T1 / textbook Hadamard-test overlap
actual QPU behavior : T3-style mixed-state target
clean operator      : G_M
```

Earlier probes showed that the original QPU data did not match the assumed textbook target. The process record documents the shift from the wrong target, to the `T3` mixed-state target, and then to the `G_M` square-root form.

Important structural notes:

```text
G_M is bounded.
G_M is not a normal dot product.
G_M is not a Mercer / PSD kernel in the usual sense.
G_M is useful as a generalized projection-style similarity operator, not as a drop-in cosine clone.
```

For unbounded vector inputs, the production path uses a phase lift before applying `G_M` per dimension.

In the current CUDA megakernel path, the phase lift and cosine are folded into the global-to-shared transfer path. The production benchmark consumes raw embeddings; there is no host-side phase-lift step in the current capstone runner.

---

## Repository structure

```text
GHOST_ORACLE_SUITE/
└── ghost_oracle/
    └── G_M/
        ├── README.md
        ├── g_m_benchmark.py
        ├── g_m_gpu_generate.py
        ├── g_m_qpu_generate.py
        │
        ├── data/
        │   ├── G_M_final_benchmark.json
        │   ├── ghost_oracle_gpu_4096shots_seed68133331...npz
        │   ├── ghost_oracle_gpu_4096shots_seed21517893...npz
        │   ├── ghost_oracle_gpu_4096shots_seed39219492...npz
        │   ├── ghost_oracle_gpu_4096shots_seed54342042...npz
        │   ├── ghost_oracle_gpu_4096shots_seed65686043...npz
        │   ├── job_d8c4q5r8ch0s738uaq30.npz
        │   ├── job_d8c4qjr8ch0s738uaqk0.npz
        │   ├── job_d8c4qmr8amns73bj0b0g.npz
        │   ├── job_d8dod1i4gq0s73aqj3m0.npz
        │   └── job_d8e4fmpvjngc73ansgug.npz
        │
        ├── docs/
        │   ├── architecture.md
        │   ├── known_issues.md
        │   └── math.md
        │
        ├── examples/
        │   ├── auto_oracle.py
        │   ├── bsgs_geometric_engine.py
        │   ├── final_benchmark_5way.py
        │   ├── parameter_ablation.py
        │   └── projection_benchmark.py
        │
        ├── kernels/
        │   ├── ghost_kernel.cu
        │   └── megakernels_2d.cu
        │
        └── probes/
            ├── benchmark_evolution/
            │   ├── final_benchmark_combined.py
            │   ├── final_benchmark_tied_perdim.py
            │   ├── final_benchmark_tied.py
            │   └── final_benchmark.py
            │
            ├── probe1_identity_bridge.py
            ├── probe2_projection_scrambled_control.py
            ├── probe3_anchor_conditioned_projection.py
            ├── probe4_build_base.py
            ├── probe5_unified_engine.py
            ├── probe6_3way_convergence.py
            ├── probe7_ghost_parity.py
            ├── probe8_residual_decomposition.py
            ├── probe9_1_indef_kernel_attn.py
            ├── probe9_ghost_operator.py
            ├── probe10_1_real_softmax_attack.py
            ├── probe10_ghost_attention.py
            └── README.md
```

## Directory map

| Path                          | Role                                                                                           |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| `README.md`                   | Main G_M documentation and current benchmark summary.                                          |
| `g_m_benchmark.py`            | Current canonical benchmark runner for verification, sweeps, and controls.                     |
| `g_m_gpu_generate.py`         | Generates noiseless GPU projection bases.                                                      |
| `g_m_qpu_generate.py`         | Submits/generates QPU projection bases.                                                        |
| `data/`                       | Curated benchmark outputs and projection base files.                                           |
| `docs/`                       | Architecture notes, math notes, known issues, and future direction documents.                  |
| `examples/`                   | Legacy and supporting examples kept for continuity.                                            |
| `examples/analysis/`          | Output / analysis workspace for example scripts.                                               |
| `kernels/`                    | CUDA source for geometry, projection, tied kernels, and 2D megakernels.                        |
| `probes/`                     | Chronological research probes that document the path from failed target to generalized metric. |
| `probes/benchmark_evolution/` | Earlier benchmark variants preserved as part of the process record.                            |

## Current main entry points

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

Legacy examples:

```bash
python examples/final_benchmark_5way.py
python examples/auto_oracle.py
python examples/projection_benchmark.py
```

## Data files

The `data/` folder contains two kinds of files:

```text
G_M_final_benchmark.json
```

Current saved benchmark summary.

```text
job_<JOB_ID>.npz
```

Real QPU projection bases dumped from IBM Runtime jobs.

```text
ghost_oracle_gpu_4096shots_seed<SEED>.npz
```

Noiseless GPU-generated projection bases.

Recommended policy:

```text
Keep small curated fixtures if they are part of the reproducibility story.
Keep large generated bases out of git unless intentionally shipping them.
```

Recommended `.gitignore` patterns:

```gitignore
data/job_*.npz
data/ghost_oracle_gpu_*.npz
probe*_calibration*.json
*_report.json
```

## Probe path

The `probes/` directory is not dead code. It is the research record.

It preserves the chronological path from:

```text
failed textbook Hadamard-test interpretation
```

to:

```text
observed QPU behavior
```

to:

```text
T3-style mixed-state target
```

to:

```text
G_M generalized metric
```

The probes should stay in the repo because they document how the operator was found, what failed, what was corrected, and what controls were added.

---

## Current capstone benchmark

Run:

```bash
python G_M_final_benchmark.py
```

Optional full run:

```bash
python G_M_final_benchmark.py --sweep ALL --probe
```

The current capstone combines the earlier verification and Auto Oracle-style retrieval paths into one optimized benchmark.

It includes:

```text
five-way verification
semantic scale sweeps
load-bearing shot-count controls
separation spectrum controls
```

The benchmark compiles one CUDA module from:

```text
kernels/ghost_kernel.cu
kernels/megakernels_2d.cu
```

The production path uses the 2D megakernels:

| Kernel               | Role                                                  |
| -------------------- | ----------------------------------------------------- |
| `geo_megakernel_2d`  | Geometry-channel scoring.                             |
| `proj_megakernel_2d` | Projection-channel scoring.                           |
| `tied_megakernel_2d` | Geometry + projection + agreement in one tied launch. |

---

## Calibration

The capstone benchmark uses two calibration objectives because the verification task and the semantic retrieval task measure different things.

### Verification calibration

The five-way verification path uses:

```text
calibrate_candidates()
```

This selector scores candidate `(tile, mask, threshold)` combinations using the projection megakernel and the Flash-Squelch certificate:

```text
clean := median spike fraction <= SPIKE_TOLERANCE
rank  := clean first, then signal fraction descending
```

This is used for the coherent same-dimension attack benchmark.

The calibrated threshold is then used in the verification run.

### Sweep / probe calibration

The semantic sweep and probe path uses:

```text
calibrate_recall1()
```

This selector ranks `(tile, mask)` components by the same recall@1 argmax retrieval objective that the sweep and probes actually run.

This matters because a component can be clean under the Flash-Squelch attack certificate while still being a poor semantic retriever. The sweep/probe selector prevents that mismatch by requiring retrieval performance directly.

A base whose best component does not clear the recall@1 floor is excluded from sweep/probe evaluation instead of being silently included as a dead component.

---

## Five-way verification

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

The benchmark compares five paths:

| Path     | Role                                                                |
| -------- | ------------------------------------------------------------------- |
| `CUBLAS` | Standard dot-product attention control on raw embeddings.           |
| `TIED`   | Dual-channel geometry + projection kernel with agreement reporting. |
| `GEO`    | Closed-form `G_M` geometry channel.                                 |
| `QPROJ`  | Projection channel driven by real QPU shot-count buckets.           |
| `GPROJ`  | Projection channel driven by noiseless GPU shot-count buckets.      |

Current five-way summary:

| Path                          |  Top-1 | Signal | Spike fraction |               Time |
| ----------------------------- | -----: | -----: | -------------: | -----------------: |
| `CUBLAS`                      |  57.9% |  57.9% |       0.426528 |            0.74 ms |
| `GEO` mean across bases       | 100.0% |      — |              — |                  — |
| `QPROJ` mean across QPU bases | 100.0% | 100.0% |         0.0498 | ~22–26 ms per base |
| `GPROJ` mean across GPU bases |  99.7% |  99.6% |         0.0502 | ~22–30 ms per base |

Interpretation:

```text
CUBLAS is the dense dot-product control.
GEO is the closed-form geometry channel.
QPROJ is calibrated projection from real QPU shot records.
GPROJ is calibrated projection from noiseless GPU shot records.
agreement measures mean |geometry - projection| per query.
```

This benchmark is not a claim that QPU projection is faster than cuBLAS dense attention. cuBLAS is much faster as a dense GEMM-style primitive.

The claim is narrower:

```text
Under this coherent same-dimension attack, standard dot-product attention
concentrates probability mass on the attack dimension, while the bounded
G_M geometry/projection channels preserve retrieval.
```

In this run, `CUBLAS` drops to 57.9% top-1 with spike fraction `0.426528`, while the `GEO` and `QPROJ` paths reach 100.0% top-1 and keep spike fraction near `0.05`.

---

## Agreement metric

The tied path reports:

```text
agreement = mean |geometry - projection| per query
```

This is the substrate-quality readout.

The geometry channel is the closed-form reference. The projection channel is reconstructed from calibrated shot-count buckets. The agreement number shows how far the projection substrate diverges from the closed-form geometry channel on the same task.

Example agreement values from the current run:

| Base type | Example agreement range |
| --------- | ----------------------: |
| QPU bases |           0.0049–0.0888 |
| GPU bases |           0.0104–0.1049 |

The agreement metric should not be read as a throughput claim. It is a projection-vs-geometry substrate comparison.

---

## Semantic scale sweep

Run:

```bash
python G_M_final_benchmark.py --sweep ALL
```

The semantic sweep uses clustered semantic embeddings with coherent outliers and evaluates:

```text
Recall@1
Recall@5
Recall@10
MRR
```

Current sweep settings:

```text
d = 1024
TOP_K = [1, 5, 10]
strategy = mixed projection component(s)
```

One GPU base was excluded from sweep/probe evaluation because no component cleared the 50% recall@1 calibration floor.

### SMALL sweep

| Backend                     |           R@1 |           R@5 |          R@10 |         MRR |
| --------------------------- | ------------: | ------------: | ------------: | ----------: |
| cosine                      |        99.12% |        99.12% |        99.12% |       0.991 |
| `GEO`                       |       100.00% |       100.00% |       100.00% |       1.000 |
| QPU projection bases        | 99.90–100.00% |       100.00% |       100.00% |      ~1.000 |
| usable GPU projection bases | 97.27–100.00% | 99.61–100.00% | 99.90–100.00% | 0.983–1.000 |

### MEDIUM sweep

| Backend                     |           R@1 |           R@5 |          R@10 |         MRR |
| --------------------------- | ------------: | ------------: | ------------: | ----------: |
| cosine                      |        96.88% |        96.88% |        96.88% |       0.969 |
| `GEO`                       |       100.00% |       100.00% |       100.00% |       1.000 |
| QPU projection bases        | 99.41–100.00% | 99.80–100.00% |       100.00% | 0.995–1.000 |
| usable GPU projection bases | 40.33–100.00% | 58.98–100.00% | 68.16–100.00% | 0.485–1.000 |

### LARGE sweep

| Backend                     |           R@1 |           R@5 |          R@10 |         MRR |
| --------------------------- | ------------: | ------------: | ------------: | ----------: |
| cosine                      |        95.31% |        95.31% |        95.31% |       0.953 |
| `GEO`                       |       100.00% |       100.00% |       100.00% |       1.000 |
| QPU projection bases        | 91.99–100.00% | 95.41–100.00% | 96.78–100.00% | 0.935–1.000 |
| usable GPU projection bases |  1.46–100.00% |  2.73–100.00% |  3.22–100.00% | 0.019–1.000 |

The sweep should be read as a substrate/component quality diagnostic, not as a universal projection guarantee.

The geometry channel is the clean mathematical ceiling. The projection channels depend on calibrated shot-bucket structure, selected tile/mask components, and the distribution being tested.

---

## Projection controls

Run:

```bash
python G_M_final_benchmark.py --probe
```

The probe mode runs two controls:

```text
Probe A: load-bearing shot-count control
Probe B: separation spectrum
```

---

## Probe A — load-bearing shot counts

Probe A reuses each calibrated projection component and changes only the shot-count structure.

| Count condition        | Meaning                                            |
| ---------------------- | -------------------------------------------------- |
| real calibrated counts | The actual calibrated shot bucket structure.       |
| permuted counts        | Same total count mass, bucket structure destroyed. |
| uniformized counts     | Live buckets forced uniform.                       |

Current result:

| Base                           | Component | Real R@1 | Permuted R@1 | Uniform R@1 |
| ------------------------------ | --------: | -------: | -----------: | ----------: |
| `job_d8c4q5r8ch0s738uaq30.npz` |   `t1/M4` |   99.41% |        0.00% |       0.00% |
| `job_d8c4qjr8ch0s738uaqk0.npz` |   `t1/M4` |  100.00% |        0.00% |       0.00% |
| `job_d8c4qmr8amns73bj0b0g.npz` |   `t1/M4` |  100.00% |        0.00% |       0.00% |
| `job_d8dod1i4gq0s73aqj3m0.npz` |   `t0/M1` |  100.00% |        0.00% |       0.00% |
| `job_d8e4fmpvjngc73ansgug.npz` |   `t2/M5` |  100.00% |        0.00% |       0.00% |
| usable GPU base                |   `t6/M6` |  100.00% |        0.00% |       0.00% |
| usable GPU base                |  `t12/M1` |  100.00% |        0.00% |       0.00% |
| usable GPU base                |   `t5/M6` |   40.33% |        0.00% |       0.00% |
| usable GPU base                |   `t0/M1` |   96.78% |        0.00% |       0.00% |

Key read:

```text
real >> permuted
real >> uniformized
```

This means the calibrated bucket structure is load-bearing.

The projection path is not silently collapsing to the geometry channel. Every number is produced by the projection megakernel on the calibrated component; only the `counts18` bucket structure changes.

---

## Probe B — separation spectrum

Probe B tests whether the benchmark is only winning because the task is too easy.

Selected base/component:

```text
base = job_d8c4q5r8ch0s738uaq30.npz
component = t1/M4
d = 1024
```

### B1 — outlier magnitude varies, noise fixed

```text
noise = 0.10
```

| Outlier magnitude | Noise |  Cosine |     GEO |    PROJ |
| ----------------: | ----: | ------: | ------: | ------: |
|               0.0 |  0.10 | 100.00% | 100.00% | 100.00% |
|               5.0 |  0.10 |  99.61% | 100.00% |  99.32% |
|              20.0 |  0.10 |  94.63% | 100.00% |  99.32% |
|              40.0 |  0.10 |  94.63% | 100.00% |  99.32% |
|              60.0 |  0.10 |  94.63% | 100.00% |  99.32% |
|             100.0 |  0.10 |  94.63% | 100.00% |  99.32% |

At outlier magnitude `0.0`, all three methods are competitive. This checks that `GEO` and `PROJ` are not winning only because of the attack.

As the coherent outlier magnitude rises, cosine falls toward its single-shared-dim floor while `GEO` and calibrated projection hold.

Noise is fixed, so this column isolates the magnitude axis.

### B2 — noise varies, outlier magnitude fixed

```text
outlier magnitude = 60.0
```

| Noise | Outlier magnitude | Cosine |     GEO |   PROJ |
| ----: | ----------------: | -----: | ------: | -----: |
|  0.05 |              60.0 | 94.63% | 100.00% | 99.51% |
|  0.10 |              60.0 | 94.63% | 100.00% | 99.32% |
|  0.15 |              60.0 | 94.63% | 100.00% | 98.14% |
|  0.20 |              60.0 | 94.63% | 100.00% | 95.02% |
|  0.25 |              60.0 | 94.63% | 100.00% | 88.18% |
|  0.30 |              60.0 | 94.63% | 100.00% | 74.22% |

Here, query noise rises while the attack stays fixed.

The projection component was recall@1-calibrated near:

```text
noise = 0.10
```

So if projection degrades faster than cosine as noise climbs past that, the read is calibration-distribution mismatch, not a base failure.

The geometry channel remains the noise-robust ceiling because it never touches bucket counts.

---

## QPU base workflow

The default repo may include curated base files, but fresh QPU bases can be generated.

### Step 1 — submit a QPU base job

```bash
python qpu.py
```

Common overrides:

```bash
python qpu.py --backend ibm_marrakesh
python qpu.py --shots 8192
```

The submitter prints a job ID.

### Step 2 — dump the completed QPU job

```bash
python dump.py <JOB_ID>
```

The dumper writes:

```text
data/job_<JOB_ID>.npz
```

Expected QPU base arrays:

```text
ctrl_tile{t}   : uint8, shape (shots,)
ghost_tile{t}  : uint8, shape (shots, 4)
```

A QPU base is consumed by the projection scripts and the final benchmark.

---

## Noiseless GPU bases

Generate noiseless classical bases with:

```bash
python gpu.py
```

Typical output:

```text
data/ghost_oracle_gpu_<...>.npz
```

The GPU sampler is not an arbitrary baseline. It is a noiseless classical implementation of the same projection circuit used by the QPU path.

The point of the GPU base is to separate:

```text
projection-circuit behavior
```

from:

```text
hardware noise and drift
```

---

## CUDA kernels

The core CUDA sources are:

```text
kernels/ghost_kernel.cu
kernels/megakernels_2d.cu
```

Kernel roles include:

```text
projection channel
geometry channel
tied geometry/projection agreement
2D-tiled semantic retrieval
materialized diagnostic variants
GHZ/noiseless sampling support
```

The current benchmark uses one compiled CUDA module so shared device helpers stay consistent across the projection path, geometry path, tied kernels, and capstone benchmark.

---

## Input normalization warning

Do **not** L2-normalize inputs before `G_M`.

Cosine similarity requires L2 normalization, but `G_M` uses a phase lift calibrated for roughly normal component-scale inputs. L2-normalizing high-dimensional vectors collapses component variance and makes the lifted angles nearly indistinguishable.

Bad pattern:

```text
normalize vectors to unit L2
then run G_M
```

Good pattern:

```text
keep component-scale structure
apply the phase lift
run per-dimension G_M aggregation
```

This is one of the easiest ways to accidentally make `G_M` look broken.

The semantic sweep uses its own controlled environment and retrieval setup. Do not generalize the benchmark by swapping preprocessing rules without re-running calibration and controls.

---

## What to look for

A clean `G_M` run should show at least one of these signatures, depending on mode:

```text
geometry path retrieves under coherent same-dimension attack
projection path agrees with geometry within substrate-dependent error
real QPU counts beat permuted/uniform controls
semantic retrieval survives calibrated coherent outlier settings
cosine remains competitive when the attack is removed
projection degrades under distribution shift rather than pretending to be universal
```

For dense attention, expect cuBLAS to be faster.

For the coherent attack and semantic retrieval regimes tested here, the value of `G_M` is bounded retrieval behavior, calibrated projection evidence, and cross-substrate comparison.

---

## Files produced by the pipeline

Common generated files:

```text
data/job_<JOB_ID>.npz
data/ghost_oracle_gpu_<...>.npz
data/G_M_final_benchmark.json
probe20_calibration.json
```

Recommended `.gitignore` patterns:

```gitignore
data/job_*.npz
data/ghost_oracle_gpu_*.npz
probe*_calibration*.json
*_report.json
```

Keep small curated fixtures if they are part of the reproducibility story.

Keep large generated bases out of git unless intentionally shipping them.

---

## Script map

```text
G_M_final_benchmark.py
    Current capstone runner:
    five-way verification + semantic sweep + controls.

qpu.py
    Submit real QPU projection-base jobs.

dump.py
    Fetch completed QPU jobs and save .npz bases.

gpu.py
    Generate noiseless classical projection bases.

examples/final_benchmark_5way.py
    Legacy example:
    earlier five-way substrate verification path.

examples/auto_oracle.py
    Legacy example:
    earlier in-memory calibration plus semantic retrieval path.

examples/projection_benchmark.py
    Legacy diagnostic:
    earlier attention benchmark / diagnostic path kept for continuity.

kernels/ghost_kernel.cu
    Shared CUDA implementation and device helpers.

kernels/megakernels_2d.cu
    Current 2D tiled megakernel path.
```

---

## Current bounded claim

`G_M` is a live research object. The repo keeps the wrong turns because the wrong turns are how the generalized metric was found.

The current bounded claim is:

```text
G_M is a bounded projection-channel / geometry-channel generalized similarity operator
with three-substrate expression:
  1. analytical closed form,
  2. noiseless classical GPU projection base,
  3. real QPU shot-count projection base.
```

The current benchmark evidence is:

```text
1. The closed-form geometry channel retrieves under coherent same-dimension attack.
2. Real QPU projection bases retrieve when calibrated.
3. Noiseless GPU projection bases retrieve when calibrated.
4. Destroying calibrated bucket structure destroys projection retrieval.
5. The projection path is therefore using load-bearing shot-count structure.
6. cuBLAS remains the correct dense GEMM throughput control.
```

The honest framing is:

```text
G_M is not a universal replacement for dot-product attention.
G_M is not claimed to make QPU shot reconstruction faster than cuBLAS.
G_M is useful as a bounded, calibrated, substrate-comparable generalized metric.
```

That is the claim to defend.

---

## Next development steps

Likely next steps:

```text
cleaner multi-job QPU comparison
more backend runs
expanded semantic retrieval sweeps
additional dynamic-mask calibration tests
tighter documentation of the relationship between G_M and S_M
clearer fixture policy for curated reproducibility data
```

The process is the process.

Break it, fix it, document what happened.
