# S_M — Syndrome Metric

`S_M` is the Ghost Oracle Suite syndrome-field operator family: **Syndrome Metric**.

`S_M` treats repeated syndrome measurements not as a single logical-error statistic, but as a syndrome-spacetime field. The useful object is the relationship between final data edge parity and repeated syndrome records.

Current framing:

```text
S_M = syndrome-spacetime field operator
```

The core field is:

```text
S_M = {S[t,i], E[i], A[t,i]}
```

where:

```text
D[i]   = final data bit at code position i
E[i]   = D[i] XOR D[i+1]
S[t,i] = measured syndrome bit at round/time t and edge i
A[t,i] = 1 - (S[t,i] XOR E[i])
```

`S_M` is implemented across three substrates:

1. `geo` — synthetic/reference syndrome-spacetime field,
2. `gproj` — GPU-generated syndrome-spacetime base,
3. `qproj` — real QPU syndrome-spacetime data from IBM Runtime.

The core claim is not that `S_M` is a logical-error-rate benchmark.

The core claim is that final data edge parity and repeated syndrome records form a load-bearing field structure that can be measured, scrambled, classified, and compared across synthetic, GPU-generated, and physical QPU-derived records.

---

## Quick path

The current canonical benchmark path is:

```bash
python s_m_benchmark.py
python s_m_benchmark.py --sweep ALL
python s_m_benchmark.py --probe
```

Useful variants:

```bash
python s_m_benchmark.py --windows 8 16 32 64
python s_m_benchmark.py --skip-gpu
python s_m_benchmark.py --skip-qpu
python s_m_benchmark.py --skip-geo
python s_m_benchmark.py --no-cuda
python s_m_benchmark.py --cuda-debug
```

Use explicit base files:

```bash
python s_m_benchmark.py ^
  --gpu-base data/sm_gpu_data_plus_<TAG>.npz ^
  --qpu-base data/sm_data_plus_<JOB_ID>.npz
```

The main scripts are:

| Script                                  | Status                      | Purpose                                                                        |
| --------------------------------------- | --------------------------- | ------------------------------------------------------------------------------ |
| `s_m_benchmark.py`                      | current canonical path      | Final S_M benchmark: verification, sweeps, controls, and substrate comparison. |
| `s_m_gpu_generate.py`                   | current base generator      | Generates GPU S_M bases with the same analysis schema as QPU dumps.            |
| `s_m_qpu_generate.py`                   | current QPU path            | Submit and dump IBM Runtime S_M jobs using one CLI.                            |
| `examples/sm_windowed_knn_benchmark.py` | legacy / continuity example | Earlier windowed kNN feature probe.                                            |
| `examples/sm_tsp_projector_example.py`  | downstream example          | S_M-inspired bounded field-deformation example for TSP.                        |
| `probes/sm_analyze.py`                  | legacy / analysis probe     | Earlier operator-shape and field-analysis pipeline.                            |
| `probes/token_retrieval_projector.py`   | downstream probe            | Token-retrieval projector bridge.                                              |

The QPU/GPU base-generation tools are used for producing fresh bases:

| Script                              | Purpose                                 |
| ----------------------------------- | --------------------------------------- |
| `s_m_qpu_generate.py submit`        | Submit a logical-cat S_M QPU job.       |
| `s_m_qpu_generate.py dump <JOB_ID>` | Dump a completed S_M QPU job to `.npz`. |
| `s_m_gpu_generate.py`               | Generate GPU S_M base data locally.     |

---

## Operator

The core S_M objects are:

```text
D[i]   = final data bit
E[i]   = D[i] XOR D[i+1]
S[t,i] = syndrome bit at round t and edge i
A[t,i] = 1 - (S[t,i] XOR E[i])
```

The current benchmark treats `S_M` as a windowed field operator.

For a window of shots, it computes:

```text
raw_rates
detection_rates
agreement_profiles
sm_field
sm_all
```

Feature meanings:

| Feature family       | Meaning                                                                  |
| -------------------- | ------------------------------------------------------------------------ |
| `raw_rates`          | Mean final data rates and syndrome rates.                                |
| `detection_rates`    | Mean syndrome transition / detection-event rates.                        |
| `agreement_profiles` | Edge and time profiles of final-edge/syndrome agreement.                 |
| `sm_field`           | Flattened agreement field, detection field, and compact S_M descriptors. |
| `sm_all`             | Combined S_M feature set.                                                |

Detection events are:

```text
X[t,i] = S[t+1,i] XOR S[t,i]
```

Agreement is:

```text
A[t,i] = 1 - (S[t,i] XOR E[i])
```

Important structural notes:

```text
S_M is a field operator.
S_M is windowed/statistical, not single-shot only.
S_M is not the final majority-vote logical error rate.
S_M is not the T_S stress tensor.
S_M is not a token retrieval operator.
```

The stress tensor derived from syndrome gradients is now treated as a separate future/sibling operator:

```text
T_S = stress tensor channel
```

S_M may feed T_S later, but T_S is not the headline claim in this operator package.

---

## Repository structure

```text
GHOST_ORACLE_SUITE/
└── ghost_oracle/
    └── S_M/
        ├── README.md
        ├── s_m_benchmark.py
        ├── s_m_gpu_generate.py
        ├── s_m_qpu_generate.py
        │
        ├── data/
        │   ├── sm_gpu_data_plus_<TAG>.npz
        │   ├── sm_data_plus_<JOB_ID>.npz
        │   ├── sm_gpu_job_<TAG>.json
        │   ├── sm_job_<JOB_ID>.json
        │   ├── latest_sm_gpu_data.json
        │   ├── latest_sm_job.json
        │   └── latest_sm_data.json
        │
        ├── docs/
        │   ├── architecture.md
        │   └── math.md
        │
        ├── examples/
        │   ├── sm_tsp_projector_example.py
        │   └── sm_windowed_knn_benchmark.py
        │
        ├── kernels/
        │   └── sm_kernel.cu
        │
        └── probes/
            ├── bright_observer_token_retrieval.py
            ├── build_torch_token_dataset.py
            ├── sm_analyze.py
            └── token_retrieval_projector.py
```

## Directory map

| Path                  | Role                                                                                                       |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| `README.md`           | Main S_M documentation and current benchmark summary.                                                      |
| `s_m_benchmark.py`    | Current canonical S_M benchmark runner for field verification, sweeps, controls, and substrate comparison. |
| `s_m_gpu_generate.py` | Generates GPU S_M bases with QPU-compatible analysis schema.                                               |
| `s_m_qpu_generate.py` | Unified QPU submit/dump CLI for S_M jobs.                                                                  |
| `data/`               | S_M base records, metadata, latest-file pointers, and optional curated fixtures.                           |
| `docs/`               | Architecture notes, math notes, known issues, and future direction documents.                              |
| `examples/`           | Legacy and supporting examples kept for continuity.                                                        |
| `kernels/`            | CUDA source for optimized S_M windowed feature extraction.                                                 |
| `probes/`             | Earlier probes and downstream bridge experiments.                                                          |

## Current main entry points

```bash
python s_m_benchmark.py
python s_m_benchmark.py --sweep ALL
python s_m_benchmark.py --probe
```

Base generation:

```bash
python s_m_gpu_generate.py
python s_m_qpu_generate.py submit
python s_m_qpu_generate.py dump <JOB_ID>
```

Legacy / supporting examples:

```bash
python examples/sm_windowed_knn_benchmark.py
python examples/sm_tsp_projector_example.py
```

Probe / downstream paths:

```bash
python probes/sm_analyze.py
python probes/build_torch_token_dataset.py
python probes/token_retrieval_projector.py
python probes/bright_observer_token_retrieval.py
```

---

## Data files

The `data/` folder contains S_M field bases and metadata.

```text
sm_data_plus_<JOB_ID>.npz
```

Real QPU S_M field base dumped from IBM Runtime.

```text
sm_gpu_data_plus_<TAG>.npz
```

GPU-generated S_M field base.

```text
sm_job_<JOB_ID>.json
```

QPU job metadata.

```text
sm_gpu_job_<TAG>.json
```

GPU base metadata.

```text
latest_sm_job.json
latest_sm_data.json
latest_sm_gpu_data.json
```

Convenience pointers used by the benchmark and dump scripts.

Expected S_M `.npz` arrays:

```text
schema        : str
job_id        : str
backend       : str, optional
shots         : int
rounds        : int
flag_level    : int
basis         : str
init_state    : str
distances     : int array

data_d{d}     : uint8, shape (shots, d)
synd_d{d}     : uint8, shape (shots, rounds, d-1)
flag_d{d}     : optional uint8, shape (shots, rounds, n_flags)
```

Recommended policy:

```text
Keep small curated fixtures if they are part of the reproducibility story.
Keep large generated bases out of git unless intentionally shipping them.
```

Recommended `.gitignore` patterns:

```gitignore
data/sm_data_*.npz
data/sm_gpu_data_*.npz
data/sm_job_*.json
data/sm_gpu_job_*.json
analysis/s_m_*/
*_report.json
```

Keep `latest_*.json` only if they are useful for local workflow. Avoid relying on them for published reproducibility unless the pointed files are also included.

---

## Probe path

The `probes/` directory is not the final S_M claim. It is the research and bridge workspace.

It contains earlier analysis paths and downstream projection experiments, including:

```text
sm_analyze.py
build_torch_token_dataset.py
token_retrieval_projector.py
bright_observer_token_retrieval.py
```

These are useful continuity paths, but the final S_M operator claim lives in:

```text
s_m_benchmark.py
```

The current boundary is:

```text
S_M      = syndrome-spacetime field operator
T_S      = stress tensor operator, separate
I_M      = interaction / field deformation, separate
token    = downstream projector probe, not S_M itself
TSP      = downstream field-deformation example, not S_M itself
```

---

## Current capstone benchmark

Run:

```bash
python s_m_benchmark.py
```

Optional full run:

```bash
python s_m_benchmark.py --sweep ALL --probe
```

The current capstone compares three substrates:

| Path    | Role                           |
| ------- | ------------------------------ |
| `GEO`   | Synthetic/reference S_M field. |
| `GPROJ` | GPU-generated S_M field base.  |
| `QPROJ` | Real QPU S_M field base.       |

It runs three tasks:

| Task                    | Meaning                                                       |
| ----------------------- | ------------------------------------------------------------- |
| `A_real_vs_control`     | Classify real records versus destructive controls.            |
| `B_control_source`      | Identify which control transformation produced the record.    |
| `C_distance_prediction` | Predict repetition-code distance from real S_M field windows. |

It evaluates the feature families:

```text
raw_rates
detection_rates
agreement_profiles
sm_field
sm_all
```

and the controls:

```text
real
shot_shuffle_synd
time_shuffle_synd
edge_shuffle_synd
uniform_synd
final_shuffle
all_uniform
time_reverse_synd
edge_reverse_synd
```

The benchmark uses the optimized CUDA path when available:

```text
kernels/sm_kernel.cu
```

If CuPy or the kernel is unavailable, the benchmark falls back to the NumPy reference path.

---

## Current benchmark result

Current run:

```text
Windows      : [8, 16, 32, 64]
Substrates   : GEO, GPROJ, QPROJ
CUDA kernel  : yes
Kernel path  : S_M/kernels/sm_kernel.cu
Distances    : d3, d5, d7, d9
Rounds       : 10
Shots        : 4096 per distance/base
```

### Distance prediction

All three substrates preserve code-distance structure:

| Substrate | Feature examples                                                           | Best balanced accuracy |
| --------- | -------------------------------------------------------------------------- | ---------------------: |
| `GEO`     | `raw_rates`, `detection_rates`, `agreement_profiles`, `sm_field`, `sm_all` |                  1.000 |
| `GPROJ`   | `raw_rates`, `detection_rates`, `agreement_profiles`, `sm_field`, `sm_all` |                  1.000 |
| `QPROJ`   | `raw_rates`, `detection_rates`, `agreement_profiles`, `sm_field`, `sm_all` |                  1.000 |

This result should be read carefully. Distance prediction is useful, but it can be influenced by shape, rate, and distance-dependent field statistics. It is not the main S_M claim by itself.

### Real-vs-control separation

The strongest S_M result is real-vs-control separation on field-aware features.

| Substrate | Feature              | Window | Model         | Balanced accuracy |
| --------- | -------------------- | -----: | ------------- | ----------------: |
| `QPROJ`   | `sm_all`             |     64 | logistic      |             0.999 |
| `QPROJ`   | `sm_field`           |     64 | kNN-euclidean |             0.998 |
| `QPROJ`   | `agreement_profiles` |     64 | kNN-cosine    |             0.990 |
| `GPROJ`   | `sm_field`           |     64 | kNN-euclidean |             0.985 |
| `GPROJ`   | `agreement_profiles` |     64 | kNN-cosine    |             0.982 |
| `GPROJ`   | `sm_all`             |     64 | random forest |             0.980 |

Scalar-like controls stay near chance:

| Substrate | Feature           | Example best balanced accuracy |
| --------- | ----------------- | -----------------------------: |
| `QPROJ`   | `raw_rates`       |                          0.535 |
| `QPROJ`   | `detection_rates` |                          0.509 |
| `GPROJ`   | `raw_rates`       |                          0.502 |
| `GPROJ`   | `detection_rates` |                          0.500 |
| `GEO`     | `raw_rates`       |                          0.503 |
| `GEO`     | `detection_rates` |                          0.502 |

Key read:

```text
raw_rates / detection_rates ≈ chance
agreement_profiles / sm_field / sm_all ≈ near-perfect
```

This is the main S_M operator signature.

It means the benchmark is not merely reading scalar syndrome density. The useful signal is carried by final-edge-parity agreement and syndrome-spacetime field structure.

### Control-source classification

The benchmark can also identify which destructive control produced the field.

| Substrate | Feature              | Window | Model         | Balanced accuracy |
| --------- | -------------------- | -----: | ------------- | ----------------: |
| `QPROJ`   | `sm_field`           |     64 | kNN-cosine    |             0.853 |
| `GPROJ`   | `sm_field`           |     64 | random forest |             0.848 |
| `QPROJ`   | `sm_all`             |     64 | random forest |             0.848 |
| `GPROJ`   | `sm_all`             |     64 | random forest |             0.843 |
| `QPROJ`   | `agreement_profiles` |     64 | random forest |             0.759 |

This is stronger than a simple real/fake test. It shows that different destructive controls leave distinguishable field signatures.

---

## CUDA feature extraction

The current benchmark uses one S_M CUDA kernel file:

```text
kernels/sm_kernel.cu
```

Kernel role:

```text
windowed S_M feature extraction
```

Core kernel:

```text
sm_window_features_kernel
```

It computes per-window:

```text
raw_rates
detection_rates
agreement_profiles
sm_field
```

The benchmark then assembles:

```text
sm_all = raw_rates + detection_rates + agreement_profiles + sm_field
```

The optimized path accelerates the expensive loop:

```text
substrate × distance × control mode × window × shots × rounds × edges
```

The CUDA boundary is intentionally narrow:

```text
included:
  final edge parity
  syndrome field
  agreement field
  detection events
  windowed feature reductions

excluded:
  stress tensor
  token retrieval
  TSP field deformation
```

The benchmark reports whether CUDA is active:

```text
CUDA kernel  : yes
Kernel path  : S_M/kernels/sm_kernel.cu
```

If CUDA fails, use:

```bash
python s_m_benchmark.py --cuda-debug
```

to print CuPy, path, kernel, and device diagnostics.

Force the CPU/reference path:

```bash
python s_m_benchmark.py --no-cuda
```

---

## Field controls

The controls deliberately destroy different parts of the S_M channel.

| Control             | What it destroys                                          | What it preserves                          |
| ------------------- | --------------------------------------------------------- | ------------------------------------------ |
| `real`              | Nothing.                                                  | Full field structure.                      |
| `shot_shuffle_synd` | Shot-level pairing between final data and syndrome field. | Syndrome marginal structure.               |
| `time_shuffle_synd` | Temporal order of the syndrome field.                     | Per-edge syndrome rates.                   |
| `edge_shuffle_synd` | Spatial/edge order of the syndrome field.                 | Per-time syndrome rates.                   |
| `uniform_synd`      | Structured syndrome field.                                | Approximate syndrome probability envelope. |
| `final_shuffle`     | Final data / edge-parity pairing.                         | Syndrome field structure.                  |
| `all_uniform`       | Data and syndrome structure.                              | Broad marginal probability envelope.       |
| `time_reverse_synd` | Forward temporal orientation.                             | Time content.                              |
| `edge_reverse_synd` | Spatial orientation.                                      | Edge content.                              |

The key S_M read is:

```text
real field > destructive controls
```

More specifically:

```text
agreement_profiles should beat raw scalar rates
sm_field should beat raw scalar rates
final_shuffle should damage agreement features
time/edge shuffles should damage field features
uniform/all_uniform should collapse structured field signatures
```

---

## Substrate agreement

The benchmark writes:

```text
substrate_agreement.csv
```

This compares real field profiles across available substrates:

```text
agreement_edge
agreement_time
agreement_field
detection_edge
detection_time
detection_field
```

Metrics include:

```text
correlation
L2 distance
```

This is the S_M analogue of the G_M agreement readout.

For G_M, agreement is score-level:

```text
mean |geometry - projection| per query
```

For S_M, agreement is field-profile-level:

```text
field correlation and field-profile distance across substrates
```

The substrate agreement table is a diagnostic, not a throughput claim.

---

## QPU base workflow

Fresh QPU S_M bases can be generated with the unified QPU CLI.

### Step 1 — submit

```bash
python s_m_qpu_generate.py submit
```

Common overrides:

```bash
python s_m_qpu_generate.py submit --backend ibm_marrakesh
python s_m_qpu_generate.py submit --shots 8192
python s_m_qpu_generate.py submit --init-state minus
python s_m_qpu_generate.py submit --flag 1 --distances 3 5
```

The submitter prints a job ID and the next dump command.

### Step 2 — dump

```bash
python s_m_qpu_generate.py dump <JOB_ID>
```

The dumper writes:

```text
data/sm_data_<init_state>_<JOB_ID>.npz
data/latest_sm_data.json
```

Expected QPU base arrays:

```text
data_d{d}      : uint8, shape (shots, d)
synd_d{d}      : uint8, shape (shots, rounds, d-1)
flag_d{d}      : optional uint8, shape (shots, rounds, n_flags)
```

A QPU base is consumed by:

```bash
python s_m_benchmark.py
```

or explicitly:

```bash
python s_m_benchmark.py --qpu-base data/sm_data_plus_<JOB_ID>.npz
```

---

## GPU bases

Generate GPU S_M bases with:

```bash
python s_m_gpu_generate.py
```

Typical output:

```text
data/sm_gpu_data_plus_<TAG>.npz
data/sm_gpu_job_<TAG>.json
data/latest_sm_gpu_data.json
```

Common options:

```bash
python s_m_gpu_generate.py --shots 4096
python s_m_gpu_generate.py --seed 42
python s_m_gpu_generate.py --verify
python s_m_gpu_generate.py --allow-cpu
```

The GPU generator is not an arbitrary baseline. It exists to create a controlled syndrome-spacetime base with the same downstream schema as the QPU dump.

The point of the GPU base is to separate:

```text
field-operator behavior
```

from:

```text
hardware noise, drift, queue timing, and backend-specific calibration effects
```

---

## CUDA kernels

The core CUDA source is:

```text
kernels/sm_kernel.cu
```

Kernel roles include:

```text
terminal edge parity
agreement field reduction
detection-event reduction
windowed raw rates
windowed agreement profiles
windowed sm_field construction
```

The current benchmark uses the CUDA kernel for feature extraction and falls back to NumPy if the CUDA path is unavailable.

The kernel intentionally does not compute:

```text
stress tensor
gradient tensor
token retrieval scores
TSP move scores
```

Those belong to separate operator or example paths.

---

## What to look for

A clean `S_M` run should show at least one of these signatures:

```text
CUDA kernel loads successfully when available
agreement/field features outperform raw scalar rates
real QPU records separate from destructive controls
final_shuffle weakens agreement-based features
time/edge shuffles alter field features
windowed aggregation improves stability
control-source classification rises above chance
distance prediction is stable across geo, gproj, and qproj
substrate field profiles are comparable but not identical
```

The strongest current signature is:

```text
raw_rates / detection_rates ≈ chance
agreement_profiles / sm_field / sm_all ≈ near-perfect
```

That is the evidence that `S_M` is reading field structure rather than only scalar syndrome density.

Do not overread the result as a hardware-speed claim. The value of `S_M` is field measurement, destructive controls, and substrate comparison.

---

## Files produced by the pipeline

Common generated files:

```text
data/sm_data_plus_<JOB_ID>.npz
data/sm_gpu_data_plus_<TAG>.npz
data/sm_job_<JOB_ID>.json
data/sm_gpu_job_<TAG>.json
data/latest_sm_job.json
data/latest_sm_data.json
data/latest_sm_gpu_data.json
```

Benchmark output:

```text
analysis/s_m_<timestamp>/
    result.json
    summary.csv
    per_feature.csv
    control_collapse.csv
    substrate_agreement.csv
    artifacts.npz
    A_real_vs_control_accuracy.png
    B_control_source_accuracy.png
    C_distance_prediction_accuracy.png
```

Optional output with `--write-windows`:

```text
window_rows.csv
```

Recommended `.gitignore` patterns:

```gitignore
data/sm_data_*.npz
data/sm_gpu_data_*.npz
data/sm_job_*.json
data/sm_gpu_job_*.json
analysis/s_m_*/
*_report.json
```

Keep small curated fixtures if they are part of the reproducibility story.

Keep large generated bases out of git unless intentionally shipping them.

---

## Script map

```text
s_m_benchmark.py
    Current capstone runner:
    S_M field verification + sweeps + controls + substrate comparison.

s_m_gpu_generate.py
    Generate GPU S_M bases with QPU-compatible analysis schema.

s_m_qpu_generate.py
    Unified QPU submit/dump CLI:
    submit IBM Runtime jobs and dump completed jobs into S_M .npz bases.

kernels/sm_kernel.cu
    Optimized CUDA feature extraction for windowed S_M fields.

examples/sm_windowed_knn_benchmark.py
    Legacy example:
    earlier field-level kNN benchmark.

examples/sm_tsp_projector_example.py
    Downstream example:
    bounded S_M-style field deformation applied to TSP.

probes/sm_analyze.py
    Legacy analysis probe:
    operator-shape and field-analysis reports.

probes/build_torch_token_dataset.py
    Downstream token-retrieval dataset builder.

probes/token_retrieval_projector.py
    Downstream token-retrieval projector benchmark.

probes/bright_observer_token_retrieval.py
    Token-retrieval projector with BrightDate-compatible provenance metadata.
```

---

## Current bounded claim

`S_M` is a live research object. The repo keeps earlier probes and downstream examples because they document how the field framing evolved.

The current bounded claim is:

```text
S_M is a syndrome-spacetime field operator with three-substrate expression:
  1. synthetic/reference field,
  2. GPU-generated syndrome-spacetime base,
  3. real QPU syndrome-spacetime base.
```

The current benchmark evidence is:

```text
1. The benchmark loads geo, gproj, and qproj S_M records under one shared task harness.
2. Field-aware features separate real QPU records from destructive controls.
3. Raw scalar-like rates remain near chance for real-vs-control separation.
4. Agreement and full field features approach near-perfect real-vs-control separation.
5. Control-source classification rises well above chance, showing different scrambles leave distinguishable field signatures.
6. Distance prediction is stable across geo, gproj, and qproj records.
7. The CUDA kernel accelerates S_M feature extraction while preserving the same operator boundary.
```

The honest framing is:

```text
S_M is not a logical-error-rate benchmark.
S_M is not the T_S stress tensor.
S_M is not a token retrieval benchmark.
S_M is not a universal hardware advantage claim.
S_M is useful as a field-structured, control-tested, substrate-comparable syndrome metric.
```

That is the claim to defend.

---

## Next development steps

Likely next steps:

```text
cleaner multi-job QPU comparison
more backend runs
expanded qproj/gproj substrate agreement tables
explicit final_shuffle collapse tables
single-shot versus windowed emergence plot
stress-tensor split into a separate T_S package
documentation cleanup for examples versus probes
fixture policy for curated reproducibility data
```

The process is the process.

Break it, fix it, document what happened.
