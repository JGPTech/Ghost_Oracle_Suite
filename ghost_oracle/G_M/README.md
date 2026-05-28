# G_M — Ghost Metric

`G_M` is the original Ghost Oracle operator family.

It began as a failed Hadamard-test interpretation. The QPU circuit was first assumed to compute a textbook overlap target, but the probes showed that assumption was wrong. The useful step was to stop asking why the circuit failed to compute the intended object and instead ask what it was consistently computing.

The resulting closed-form operator is:

```text
G_M(a, b) = sqrt((1 + cos(a) cos(b)) / 2) / α
```

where `α` is the suite’s empirical normalization constant.

Current framing:

```text
G_M = bounded projection-channel / geometry-channel similarity operator
```

`G_M` is implemented across three substrates:

1. analytical closed form,
2. noiseless classical GPU sampler,
3. real QPU shot data from IBM Runtime.

The core claim is not that QPU projection is faster than classical GPU attention. The core claim is that the same projection-style operator can be expressed across mathematical, classical-sampler, and physical-shot substrates, with an agreement metric that exposes substrate quality.

---

## Quick path

The folder has two main paths:

```bash
python ghost_oracle/G_M/final_benchmark_5way.py
python ghost_oracle/G_M/auto_oracle.py
python ghost_oracle/G_M/auto_oracle.py --probe
```

What each script does:

| Script | Purpose |
|---|---|
| `final_benchmark_5way.py` | Five-way verification against cuBLAS, tied-channel, geometry-only, QPU projection, and GPU projection. |
| `auto_oracle.py` | In-memory calibration over QPU bases plus semantic retrieval against cosine and `G_M`. |
| `auto_oracle.py --probe` | Negative controls for the projection path. |

The QPU/GPU base-generation tools are available for producing fresh bases:

| Script | Purpose |
|---|---|
| `qpu.py` | Submit a tiled IBM Runtime QPU base job. |
| `dump.py` | Dump a completed QPU base job to `.npz`. |
| `gpu.py` | Generate noiseless classical GPU bases. |

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

Earlier probes showed that the original QPU data did not match the assumed textbook target. The process record documents the shift from the wrong target to the `T3` mixed-state target and then to the `G_M` square-root form.

Important structural notes:

```text
G_M is bounded.
G_M is not a normal dot product.
G_M is not a Mercer / PSD kernel in the usual sense.
G_M is useful as a projection-style similarity operator, not as a drop-in cosine clone.
```

For unbounded vector inputs, the production path uses a phase lift before applying `G_M` per dimension.

---

## Folder layout

```text
ghost_oracle/G_M/
├── README.md
├── auto_oracle.py
├── dump.py
├── final_benchmark_5way.py
├── gpu.py
├── projection_benchmark.py
├── qpu.py
└── kernels/
    └── ghost_kernel.cu
```

Optional examples / adjacent tools may live outside this folder:

```text
examples/
├── parameter_ablation.py
└── bsgs_geometric_engine.py

docs/
├── math.md
├── architecture.md
└── known_issues.md
```

---

## Final five-way benchmark

Run:

```bash
python ghost_oracle/G_M/final_benchmark_5way.py
```

The benchmark compares five paths on the same task:

| Path | Role |
|---|---|
| `CUBLAS` | dot-product attention control |
| `TIED` | dual-channel geometry + projection kernel |
| `GEO` | closed-form `G_M` geometry |
| `QPROJ` | projection driven by real QPU shots |
| `GPROJ` | projection driven by noiseless GPU shots |

The benchmark’s purpose is to verify the operator across substrates and report agreement between the sharp geometry channel and the physical/noiseless projection channels.

Interpretation:

- `GEO` is the closed-form operator.
- `GPROJ` is the same projection circuit sampled noiselessly on GPU.
- `QPROJ` is the same projection circuit sampled from real QPU shot records.
- `agreement` measures how far projection diverges from geometry.
- GPU projection should sit near the shot-noise floor.
- QPU projection should show additional hardware-noise attenuation.

The honest dense-attention framing:

```text
cuBLAS is much faster on dense GEMM-style attention.
G_M projection paths provide substrate-specific physical certification.
The projection-channel value is agreement / hardware readout, not raw dense-attention throughput.
```

---

## Auto Oracle semantic retrieval

Run:

```bash
python ghost_oracle/G_M/auto_oracle.py
```

Then run controls:

```bash
python ghost_oracle/G_M/auto_oracle.py --probe
```

The Auto Oracle path is the current “good stuff” retrieval demo. It calibrates over available QPU bases in memory, selects a winning tile/mask component, and compares semantic retrieval against cosine.

Medium run:

```text
M = 250,000
N = 1024
d = 1024
noise = 0.12
outlier fraction = 0.03
outlier magnitude = 60
```

Example result:

| Path | Recall@1 | Time | Speed vs cosine |
|---|---:|---:|---:|
| cosine baseline | 96.88% | 1.156 s | 1.00× |
| geometry `G_M` megakernel | 100.00% | 0.897 s | **1.29× faster** |
| QPU projection — base 1 | 100.00% | 2.416 s | 0.48× |
| QPU projection — base 2 | 100.00% | 2.404 s | 0.48× |
| QPU projection — base 3 | 100.00% | 2.415 s | 0.48× |

The speed result is specific to this semantic-retrieval operating point. In this run, the closed-form `G_M` geometry megakernel beat the cosine baseline even though cosine uses the tensor-core-friendly GEMM path and the `G_M` geometry kernel does not use tensor cores.

The QPU projection path is slower because it reconstructs scores from calibrated physical shot-count buckets. Its purpose is substrate-backed projection evidence, not raw throughput.

---

## Auto Oracle controls

Run:

```bash
python ghost_oracle/G_M/auto_oracle.py --probe
```

The negative controls test whether the projection path is actually using physical shot structure.

Example result:

| Control | Recall@1 | Interpretation |
|---|---:|---|
| real calibrated counts | 100.00% | physical shot structure retrieves |
| permuted counts | 0.00% | destroying bucket structure destroys retrieval |
| uniformized counts | 0.00% | projection is not silently reducing to geometry |

The key read:

```text
real >> permuted
real >> uniformized
```

means the calibrated physical shot counts are load-bearing.

---

## QPU base workflow

The default repo should already include usable data files, but fresh QPU bases can be generated.

### Step 1 — submit a QPU base job

```bash
python ghost_oracle/G_M/qpu.py
```

Common overrides:

```bash
python ghost_oracle/G_M/qpu.py --backend ibm_marrakesh
python ghost_oracle/G_M/qpu.py --shots 8192
```

The submitter prints a job ID.

### Step 2 — dump the completed QPU job

```bash
python ghost_oracle/G_M/dump.py <JOB_ID>
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
python ghost_oracle/G_M/gpu.py
```

The GPU sampler is not an arbitrary baseline. It is a noiseless classical implementation of the same projection circuit used by the QPU path.

Typical output:

```text
data/ghost_oracle_gpu_<...>.npz
```

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

The core CUDA source is:

```text
ghost_oracle/G_M/kernels/ghost_kernel.cu
```

It is loaded by the Python scripts through CuPy.

Kernel roles include:

```text
projection channel
geometry channel
tied streaming per-dimension retrieval
materialized diagnostic variants
GHZ/noiseless sampling support
```

The design uses one CUDA source so shared device helpers stay consistent across the projection path, geometry path, tied kernels, and final benchmark.

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

---

## What to look for

A clean `G_M` run should show at least one of these signatures, depending on the script:

```text
geometry path retrieves under same-dimension coherent outlier attack
projection path agrees with geometry near expected substrate noise
real QPU counts beat permuted/uniform controls
Auto Oracle G_M beats cosine recall under the configured semantic retrieval attack
```

For dense attention, expect cuBLAS to be faster. For the current semantic retrieval operating point, the `G_M` geometry megakernel can be faster than the cosine baseline.

---

## Files produced by the pipeline

Common generated files:

```text
data/job_<JOB_ID>.npz
data/ghost_oracle_gpu_<...>.npz
probe20_calibration.json
```

Recommended `.gitignore` patterns:

```gitignore
data/job_*.npz
data/ghost_oracle_gpu_*.npz
probe*_calibration*.json
*_report.json
```

Keep small curated fixtures if they are part of the reproducibility story. Keep large generated bases out of git unless intentionally shipping them.

---

## Script map

```text
final_benchmark_5way.py
    Five-way substrate verification.

auto_oracle.py
    In-memory calibration plus semantic retrieval.

qpu.py
    Submit real QPU projection-base jobs.

dump.py
    Fetch completed QPU jobs and save .npz bases.

gpu.py
    Generate noiseless classical projection bases.

projection_benchmark.py
    Earlier attention benchmark / diagnostic path kept for continuity.

kernels/ghost_kernel.cu
    Shared CUDA implementation.
```

---

## Notes

`G_M` is a live research object. The repo keeps the wrong turns because the wrong turns are how the operator was found.

The bounded claim:

```text
observed closed-form operator
observed three-substrate implementation
observed projection-vs-geometry agreement metric
observed retrieval behavior under controlled attacks
```

rather than claiming a universal replacement for dot-product attention.

The next development steps are likely:

- cleaner multi-job QPU comparison,
- more backend runs,
- expanded semantic retrieval sweeps,
- additional dynamic-mask calibration tests,
- tighter documentation of the relationship between `G_M` and `S_M`.
