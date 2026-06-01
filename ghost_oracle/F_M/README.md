# F_M — Fractal / Frequency / Field Metric

`F_M` is the Ghost Oracle operator family for **paired-path differential wave signatures**.

It began as a QPU circuit experiment built around matched path delays. The original working model was that the circuit could create two related hardware readouts — a `g` path and an `em` path — whose difference would expose a delay-ordered structure. The useful step was not to force the circuit into the initial physical interpretation, but to freeze the QPU output and ask what the hardware record actually contained.

The resulting locked signature is:

```text
xor_delta / bit_diff / delay
```

where:

```text
delta     = em - g
xor_delta = em XOR g
bit_diff  = mean(bit1) - mean(bit0)
```

Current framing:

```text
F_M = substrate-linked paired-path differential wave operator
```

`F_M` is implemented across three substrates:

1. real QPU paired-path shot records,
2. GPU-generated paired-path bases,
3. optimized classical GEO metadata path.

The core claim is not that the physical cavity model is proven.

The core claim is that a paired-delay QPU circuit produced a stable differential wave signature, that the signature can be reproduced in a compatible GPU-generated base, and that the useful response can be computed directly by an optimized classical GEO path.

---

## Quick path

The current canonical benchmark path is:

```bash
python F_M_final_benchmark.py
```

Useful variants:

```bash
python F_M_final_benchmark.py --skip-sweep --reps 20
python F_M_final_benchmark.py --geo-profile default --max-candidates 250000
python F_M_final_benchmark.py --geo-profile wide --max-candidates 1000000 --reps 100
python F_M_final_benchmark.py --skip-qproj
python F_M_final_benchmark.py --skip-gproj
python F_M_final_benchmark.py --skip-classical
```

The QPU/GPU base-generation tools are used for producing fresh bases:

| Script                   | Purpose                                             |
| ------------------------ | --------------------------------------------------- |
| `f_m_qpu_generate.py`    | Submit / dump a tiled F_M QPU paired-path base.     |
| `f_m_gpu_generate.py`    | Generate a GPU-compatible paired-path `gproj` base. |
| `F_M_final_benchmark.py` | Current canonical final benchmark.                  |

Research probes are retained for continuity and process history:

| Script                                         | Status                 | Purpose                                           |
| ---------------------------------------------- | ---------------------- | ------------------------------------------------- |
| `probes/f_m_probe01_*.py`                      | historical probe       | Broad field/family/control scan.                  |
| `probes/f_m_probe02_delta_address.py`          | historical probe       | Address localization across tile/delay/scale/bit. |
| `probes/f_m_probe03_wave_nature.py`            | historical probe       | Wave-nature and destructive-control test.         |
| `probes/f_m_probe04_qproj_kernel_finalizer.py` | locked projector probe | CUDA projector signature finalizer.               |
| `probes/f_m_probe05_geo_numpy.py`              | locked geo probe       | NumPy GEO formula discovery.                      |
| `probes/f_m_probe06_geo_cuda_finalizer.py`     | locked geo probe       | CUDA GEO finalizer and optimized sweep.           |

Current benchmark claims should come from:

```text
F_M_final_benchmark.py
```

and from the saved output:

```text
analysis/fm_final_benchmark_<timestamp>/result.json
```

---

## Operator

The core paired-path fields are:

```text
delta     = em - g
xor_delta = em XOR g
```

The primary locked response is:

```text
bit_diff = bit1_mean - bit0_mean
```

The primary ordering is:

```text
delay
```

So the primary operator signature is:

```text
xor_delta / bit_diff / delay
```

In plain form, the qproj/gproj response curve is:

```text
R(t)
=
mean over shots of xor_delta[t, :, 1]
-
mean over shots of xor_delta[t, :, 0]
```

Then the curve is sorted by tile delay and scored by the F_M wave metric.

The first locked QPU base used:

```text
tiles            = 7
shots            = 4096
backend          = ibm_marrakesh
tile_delay_dt    = [0, 1, 2, 4, 8, 16, 0]
tile_scale_level = [1, 1, 1, 1, 1, 1, 2]
```

Important structural notes:

```text
F_M is differential.
F_M is delay-ordered.
F_M is path-pair dependent.
F_M is not raw g or raw em alone.
F_M is not FFT alone.
F_M is not a proof of the physical cavity model.
```

The physical cavity language motivated the circuit. The benchmarked operator is the measured paired-path differential wave signature.

---

## Repository structure

```text
GHOST_ORACLE_SUITE/
└── ghost_oracle/
    └── F_M/
        ├── README.md
        ├── F_M_final_benchmark.py
        ├── f_m_gpu_generate.py
        ├── f_m_qpu_generate.py
        │
        ├── data/
        │   ├── latest_fm_qpu_data.json
        │   ├── latest_fm_gpu_data.json
        │   ├── fm_job_<JOB_ID>.npz
        │   └── fm_gpu_data_<...>.npz
        │
        ├── docs/
        │   ├── architecture.md
        │   ├── known_issues.md
        │   └── math.md
        │
        ├── examples/
        │   └── ...
        │
        ├── kernels/
        │   └── fm_projector_kernel.cu
        │
        └── probes/
            ├── f_m_probe01_*.py
            ├── f_m_probe02_delta_address.py
            ├── f_m_probe03_wave_nature.py
            ├── f_m_probe04_qproj_kernel_finalizer.py
            ├── f_m_probe05_geo_numpy.py
            ├── f_m_probe06_geo_cuda_finalizer.py
            └── analysis/
```

---

## Directory map

| Path                     | Role                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| `README.md`              | Main F_M documentation and current benchmark summary.                                       |
| `F_M_final_benchmark.py` | Current canonical benchmark runner for qproj/gproj/geo and speed baselines.                 |
| `f_m_gpu_generate.py`    | Generates GPU-compatible paired-path `gproj` bases.                                         |
| `f_m_qpu_generate.py`    | Submits / dumps QPU paired-path `qproj` bases.                                              |
| `data/`                  | Frozen qproj/gproj base files and latest pointers.                                          |
| `docs/`                  | Architecture notes, math notes, known issues, and future direction documents.               |
| `examples/`              | Optional examples retained for continuity.                                                  |
| `kernels/`               | CUDA source for response, control, wave metric, GEO curve, and GEO sweep kernels.           |
| `probes/`                | Chronological research probes that document the path from qproj discovery to GEO benchmark. |
| `probes/analysis/`       | Output / analysis workspace for probe scripts.                                              |

---

## Current main entry points

Final benchmark:

```bash
python F_M_final_benchmark.py
```

Fast debugging pass:

```bash
python F_M_final_benchmark.py --skip-sweep --reps 20
```

Bigger capstone run:

```bash
python F_M_final_benchmark.py --geo-profile wide --max-candidates 1000000 --reps 100
```

Base generation:

```bash
python f_m_gpu_generate.py
python f_m_qpu_generate.py
```

Run the GEO CUDA finalizer directly:

```bash
python probes/f_m_probe06_geo_cuda_finalizer.py --profile default --max-candidates 250000
```

Run the projector finalizer directly:

```bash
python probes/f_m_probe04_qproj_kernel_finalizer.py --file data/fm_job_<JOB_ID>.npz
python probes/f_m_probe04_qproj_kernel_finalizer.py --file data/fm_gpu_data_<...>.npz
```

---

## Data files

The `data/` folder contains two kinds of F_M bases:

```text
fm_job_<JOB_ID>.npz
```

Real QPU paired-path bases dumped from IBM Runtime jobs.

```text
fm_gpu_data_<...>.npz
```

GPU-generated paired-path bases.

Latest pointers:

```text
latest_fm_qpu_data.json
latest_fm_gpu_data.json
```

Recommended policy:

```text
Keep small curated fixtures if they are part of the reproducibility story.
Keep large generated bases out of git unless intentionally shipping them.
```

Recommended `.gitignore` patterns:

```gitignore
data/fm_job_*.npz
data/fm_gpu_data_*.npz
probes/analysis/
analysis/
*_report.json
```

---

## Base schema

A frozen F_M base is a `.npz` file with shared qproj/gproj schema.

Core metadata:

```text
schema
suite
operator
substrate
job_id
backend
shots
num_tiles
tile_indices
tile_delay_dt
tile_scale_level
tile_theta
tile_mode
tile_role
```

Core stacked arrays:

```text
ctrl       : uint8, shape (tiles, shots)
g          : uint8, shape (tiles, shots, bits)
em         : uint8, shape (tiles, shots, bits)
scale      : uint8, shape (tiles, shots)
branch     : uint8, shape (tiles, shots, 2)
delta      : int8,  shape (tiles, shots, bits)
xor_delta  : uint8, shape (tiles, shots, bits)
```

Per-tile arrays are also stored for compatibility:

```text
ctrl_tile{t}
g_tile{t}
em_tile{t}
scale_tile{t}
branch_tile{t}
delta_tile{t}
xor_delta_tile{t}
```

The shared schema is load-bearing because it allows the same projector scripts to consume QPU and GPU bases without changing operator logic.

---

## Probe path

The `probes/` directory is not dead code. It is the research record.

It preserves the chronological path from:

```text
rough multi-base Benford / p-adic telemetry
```

to:

```text
delta/xor_delta field localization
```

to:

```text
wave-nature and control-collapse testing
```

to:

```text
CUDA projector finalization
```

to:

```text
GEO NumPy discovery
```

to:

```text
GEO CUDA finalizer
```

The probes should stay in the repo because they document how the operator was found, what controls were used, what was locked, and what the final benchmark is allowed to claim.

---

## Current capstone benchmark

Run:

```bash
python F_M_final_benchmark.py
```

The current capstone includes:

```text
substrate signature comparison
GEO sweep / parameter selection
speed comparison
adjacent classical readers
saved JSON / CSV output
```

The benchmark compiles one CUDA module from:

```text
kernels/fm_projector_kernel.cu
```

The production path uses these CUDA kernels:

| Kernel                                  | Role                                                |
| --------------------------------------- | --------------------------------------------------- |
| `fm_response_kernel_u8`                 | QPROJ/GPROJ response compression from `g` and `em`. |
| `fm_path_pair_break_response_kernel_u8` | Paired-path destruction control.                    |
| `fm_geo_curve_kernel_f32`               | Optimized GEO curve generation from metadata.       |
| `fm_geo_sweep_kernel_f32`               | Fast GEO parameter sweep.                           |
| `fm_wave_metric_kernel_f32`             | Shared wave metric for qproj/gproj/geo curves.      |

---

## Three-substrate comparison

The benchmark compares three F_M substrate paths:

| Path    | Meaning                                                             |
| ------- | ------------------------------------------------------------------- |
| `QPROJ` | Real QPU paired-path record through response + wave metric kernels. |
| `GPROJ` | GPU-generated paired-path record through the same kernels.          |
| `GEO`   | Analytic metadata path through GEO curve + wave metric kernels.     |

Current primary signature:

```text
xor_delta / bit_diff / delay
```

Current final benchmark values:

| Substrate |  Score |  Peak |    R2 | Freq |     Amp |
| --------- | -----: | ----: | ----: | ---: | ------: |
| `QPROJ`   | 0.6571 | 0.769 | 0.819 | 1.30 | 0.05800 |
| `GPROJ`   | 0.6796 | 0.772 | 0.986 | 0.90 | 0.03837 |
| `GEO`     | 0.7356 | 0.812 | 0.988 | 1.10 | 0.04813 |

Interpretation:

```text
QPROJ discovers it.
GPROJ reproduces the signature family.
GEO computes the signature directly.
```

The paths are not expected to be numerically identical. They are expected to preserve the same primary signature family.

---

## Speed comparison

Current final benchmark speed values:

| Path    | Operation            |        Time |
| ------- | -------------------- | ----------: |
| `QPROJ` | response + metric    | 1.045625 ms |
| `GPROJ` | response + metric    | 1.059980 ms |
| `GEO`   | geo curve + metric   | 0.382735 ms |
| `GEO`   | 250k candidate sweep |  290.061 ms |

Approximate speedup:

```text
GEO is about 2.7x faster than QPROJ projector evaluation.
GEO is about 2.8x faster than GPROJ projector evaluation.
```

The read:

```text
QPROJ and GPROJ are record-based substrate paths.
GEO is the optimized runtime path.
```

---

## Adjacent classical baselines

The final benchmark also compares adjacent signal-analysis readers on the primary curve:

```text
xor_delta / bit_diff / delay
```

Current values:

| Baseline       |  Score |      Time |
| -------------- | -----: | --------: |
| `FFT_GPU`      | 0.7688 |  0.459 ms |
| `DCT_GPU`      | 0.6047 |  0.477 ms |
| `AUTOCORR_GPU` | 0.4519 |  1.010 ms |
| `SINFIT_GPU`   | 0.8193 | 44.191 ms |

Interpretation:

```text
FFT and SinFit are strong readers of the already-built wave curve.
F_M GEO constructs the operator-specific curve from metadata and scores it.
```

This distinction matters.

`FFT_GPU` is a strong fast reader. `SINFIT_GPU` scores highest but is much slower in this implementation. Neither is a qproj/gproj/geo substrate path.

---

## QPU base workflow

Fresh QPU bases can be generated through the F_M QPU tool.

Typical workflow:

```bash
python f_m_qpu_generate.py submit
```

Then, after the IBM Runtime job completes:

```bash
python f_m_qpu_generate.py dump <JOB_ID>
```

Expected output:

```text
data/fm_job_<JOB_ID>.npz
data/latest_fm_qpu_data.json
```

The QPU base is consumed by:

```text
F_M_final_benchmark.py
probes/f_m_probe04_qproj_kernel_finalizer.py
probes/f_m_probe05_geo_numpy.py
probes/f_m_probe06_geo_cuda_finalizer.py
```

A QPU base is the frozen hardware record. Do not mutate it after dumping. Generate a new base if the circuit changes.

---

## GPU base workflow

Generate a GPU-compatible paired-path base with:

```bash
python f_m_gpu_generate.py
```

Recommended matched run:

```bash
python f_m_gpu_generate.py --match-qpu data/fm_job_<JOB_ID>.npz --seed 42 --verify
```

Expected output:

```text
data/fm_gpu_data_<...>.npz
data/latest_fm_gpu_data.json
```

The GPU base is not arbitrary noise. It is a controlled paired-path generator designed to preserve the discovered qproj signature family while keeping schema compatibility.

The point of the GPU base is to separate:

```text
paired-path differential operator behavior
```

from:

```text
hardware-specific shot noise and backend drift
```

---

## GEO workflow

The GEO path is the optimized classical path.

It does not sample shots.

It computes analytic curves from metadata:

```text
tile_delay_dt
tile_scale_level
tile_theta
tile_mode
```

Run the GEO finalizer:

```bash
python probes/f_m_probe06_geo_cuda_finalizer.py --profile default --max-candidates 250000
```

Typical output:

```text
probes/analysis/fm_probe06_geo_cuda_finalizer_<timestamp>/
    result.json
    geo_signature.csv
    geo_comparison.csv
    geo_curve_values.csv
    geo_sweep_top.csv
```

The final benchmark calls the same locked GEO path.

---

## CUDA kernels

The core CUDA source is:

```text
kernels/fm_projector_kernel.cu
```

Kernel roles include:

```text
record response compression
path-pair-break controls
wave metric scoring
GEO curve generation
GEO parameter sweep
```

The current benchmark uses one compiled CUDA module so shared device helpers stay consistent across the qproj path, gproj path, geo path, and capstone benchmark.

---

## What to look for

A clean `F_M` run should show:

```text
xor_delta / bit_diff / delay near the top for QPROJ
xor_delta / bit_diff / delay near the top for GPROJ
xor_delta / bit_diff / delay near the top for GEO
GEO faster than record-based QPROJ/GPROJ evaluation
FFT/SinFit acting as strong adjacent readers, not replacing F_M
```

A clean control history should show:

```text
path_pair_break weakens the primary signal
delay_shuffle weakens the primary signal
simple raw marginal statistics are not the whole story
```

If the primary signal does not weaken under path-pair break, it is probably not measuring the same operator.

---

## Files produced by the pipeline

Common generated files:

```text
data/fm_job_<JOB_ID>.npz
data/fm_gpu_data_<...>.npz
data/latest_fm_qpu_data.json
data/latest_fm_gpu_data.json
analysis/fm_final_benchmark_<timestamp>/result.json
analysis/fm_final_benchmark_<timestamp>/substrate_signature.csv
analysis/fm_final_benchmark_<timestamp>/classical_baselines.csv
analysis/fm_final_benchmark_<timestamp>/speed_summary.csv
analysis/fm_final_benchmark_<timestamp>/geo_comparison.csv
analysis/fm_final_benchmark_<timestamp>/curve_values.csv
```

Recommended `.gitignore` patterns:

```gitignore
data/fm_job_*.npz
data/fm_gpu_data_*.npz
analysis/
probes/analysis/
*_report.json
```

Keep small curated fixtures if they are part of the reproducibility story.

Keep large generated bases out of git unless intentionally shipping them.

---

## Script map

```text
F_M_final_benchmark.py
    Current capstone runner:
    qproj/gproj/geo comparison + speed baselines.

f_m_qpu_generate.py
    Submit and dump real QPU paired-path base jobs.

f_m_gpu_generate.py
    Generate GPU-compatible paired-path bases.

probes/f_m_probe01_*.py
    Broad early field/family/control scan.

probes/f_m_probe02_delta_address.py
    Tile/delay/scale/bit address localization.

probes/f_m_probe03_wave_nature.py
    Wave-nature and destructive-control probe.

probes/f_m_probe04_qproj_kernel_finalizer.py
    CUDA projector signature finalizer.

probes/f_m_probe05_geo_numpy.py
    NumPy GEO formula discovery.

probes/f_m_probe06_geo_cuda_finalizer.py
    CUDA GEO finalizer and optimized parameter sweep.

kernels/fm_projector_kernel.cu
    Shared CUDA implementation and device helpers.
```

---

## Current bounded claim

`F_M` is a live research object, but this operator package is complete for this version.

The current bounded claim is:

```text
F_M is a substrate-linked paired-path differential wave operator
with three-substrate expression:
  1. real QPU paired-path qproj base,
  2. GPU-generated paired-path gproj base,
  3. optimized classical geo path.
```

The current benchmark evidence is:

```text
1. The QPU record contains a stable xor_delta / bit_diff / delay signature.
2. The GPU-generated base reproduces the same signature family.
3. The GEO path computes the signature directly from metadata.
4. Path-pair structure is load-bearing.
5. Delay order is load-bearing.
6. GEO is faster than record-based qproj/gproj projector evaluation.
7. FFT and SinFit are useful adjacent readers of the primary curve, not full substrate paths.
```

The honest framing is:

```text
F_M is not a proof of literal hardware gravity/electromagnetic cavities.
F_M is not a universal replacement for FFT or sinusoid fitting.
F_M is not claimed to make QPU shot reconstruction faster than optimized GPU analysis.
F_M is useful as a paired-path differential wave operator with qproj/gproj/geo substrate linkage.
```

That is the claim to defend.

---

## Next development steps

Likely next steps:

```text
repeat QPU runs across multiple IBM backends/calibrations
expand from 7 tiles to richer delay ladders
test larger delay/scale grids for more stable frequency estimation
integrate F_M as a feature channel in the node-point network
compare F_M with wavelet and Lomb-Scargle-style baselines if needed
document the exact gate-level QPU circuit in a dedicated circuit spec
```

The process is the process.

Break it, fix it, document what happened.
