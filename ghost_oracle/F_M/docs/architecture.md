# Architecture

Architecture for the `F_M` operator package.

`F_M` is the **Fractal / Frequency / Field Metric** channel. It is a completed operator package in the Ghost Oracle Suite Converger architecture.

This document replaces any earlier framing that treated `F_M` mainly as a loose Benford/p-adic probe or a rough QPU experiment. The current framing is:

```text
F_M is a finished operator package for this version.
F_M is one channel in the larger Converger roadmap.
F_M uses the standard qproj / gproj / geo substrate pattern.
F_M claims are made only through the benchmark runner and controls.
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

This architecture document explains how the finished `F_M` operator is wired together, what each file is responsible for, what counts as a valid benchmark claim, and how the package fits into the larger Converger direction without inflating the current result beyond what the benchmark supports.

---

## 1. Architectural status

`F_M` is complete for this version.

That means the package now has:

```text
1. a real QPU paired-path base,
2. a GPU-generated paired-path base,
3. an optimized classical geo path,
4. a CUDA projector kernel,
5. a CUDA geo curve kernel,
6. a CUDA geo sweep kernel,
7. a canonical final benchmark runner,
8. adjacent classical speed baselines,
9. saved benchmark output,
10. documented math,
11. documented process,
12. documented known limits.
```

The current package is not a placeholder for future work. It is a finished operator implementation with a bounded claim.

The bounded claim is:

```text
F_M is a substrate-linked paired-path differential wave operator.
```

The claim is not:

```text
F_M proves literal gravitational or electromagnetic cavities in QPU hardware.
F_M replaces FFT, DCT, wavelet, or sinusoid readers universally.
F_M proves quantum advantage.
F_M makes QPU shot reconstruction faster than optimized GPU signal analysis.
F_M is guaranteed to preserve the same frequency/amplitude across future QPU jobs.
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

Within that system, `F_M` is the paired-path differential wave component.

```text
Converger operator stack
├── G_M        Generalized Metric channel
├── S_M        Syndrome Metric channel
├── T_S        Stress channel
├── I_M.local  Local Interaction channel
├── I_M.field  Field Interaction channel
├── F_M        Fractal / Frequency / Field Metric channel
└── D_M        Dimensional Metric channel
```

`F_M` is not the entire Converger.

`F_M` is one completed operator package in that stack.

Its role is:

```text
paired-path differential structure
delay-ordered wave signatures
frequency / phase response
qproj-to-gproj-to-geo substrate comparison
fast metadata-to-signature evaluation
```

Transformer-adjacent inputs may eventually include:

```text
operator node states
delay/frequency features
memory candidate fields
node-point network edges
field interaction traces
temporal response signatures
```

The `F_M` package does not replace transformer components. It supplies a bounded differential-wave measurement channel that can become one feature family in the larger Converger stack.

---

## 3. Standard operator package pattern

The future Converger architecture uses the same package shape for every operator:

```text
operator/
├── operator_gpu_generate.py
├── operator_qpu_generate.py
└── operator_benchmark.py
```

For `F_M`, the current package uses:

```text
F_M/
├── f_m_gpu_generate.py
├── f_m_qpu_generate.py
└── F_M_final_benchmark.py
```

Supporting directories:

```text
F_M/
├── data/
├── docs/
├── examples/
├── kernels/
└── probes/
```

The current main entry point is:

```bash
python F_M_final_benchmark.py
```

Base generation:

```bash
python f_m_gpu_generate.py
python f_m_qpu_generate.py
```

Research probes remain in the repo:

```bash
python probes/f_m_probe01_*.py
python probes/f_m_probe02_delta_address.py
python probes/f_m_probe03_wave_nature.py
python probes/f_m_probe04_qproj_kernel_finalizer.py
python probes/f_m_probe05_geo_numpy.py
python probes/f_m_probe06_geo_cuda_finalizer.py
```

Those probes are retained for continuity and process history. They are not the current source of benchmark claims.

Current benchmark claims should come from:

```text
F_M_final_benchmark.py
```

and from the saved output:

```text
analysis/fm_final_benchmark_<timestamp>/result.json
```

---

## 4. Standard substrate paths

`F_M` follows the standard three-substrate pattern used by the Converger roadmap.

```text
F_M_qproj
F_M_gproj
F_M_geo
```

In plain text:

```text
F_M_qproj  = QPU / hardware paired-path shot record
F_M_gproj  = GPU / generated paired-path record
F_M_geo    = optimized classical analytic path
```

### 4.1 QPU projection path

The QPU projection path uses real shot records from the F_M paired-path circuit.

```text
F_M_qproj(B_q)
```

where `B_q` is a dumped QPU base file.

The QPU base contains paired path arrays:

```text
g[tile, shot, bit]
em[tile, shot, bit]
```

and metadata:

```text
tile_delay_dt
tile_scale_level
tile_theta
tile_mode
```

The QPU path is the hardware-discovered substrate. It is the source of the original locked signature.

It is not used to claim raw throughput superiority. It is used to discover and verify whether a paired-path differential wave signature exists in real hardware records.

### 4.2 GPU projection path

The GPU projection path uses a generated base with the same schema as the QPU dump.

```text
F_M_gproj(B_g)
```

where `B_g` is a GPU-generated base file.

The GPU base is not an arbitrary baseline. It is a controlled paired-path generator designed to reproduce the discovered qproj signature family while preserving the same analysis-facing schema.

Its purpose is to separate:

```text
paired-path differential operator behavior
```

from:

```text
hardware-specific shot noise and calibration drift
```

### 4.3 Geometry path

The geometry path is the optimized classical reference.

It does not generate shots.

It evaluates an analytic delay-wave formula directly from metadata:

```text
tile_delay_dt
tile_scale_level
tile_theta
tile_mode
```

and emits the same projector-facing response curves used by qproj and gproj.

The geometry path is the clean mathematical operator path.

It is the fastest current F_M implementation.

---

## 5. Bases

A **base** is a `.npz` file containing measurement or generated data for tiled F_M paired-path circuits.

The shared F_M base schema is:

```text
schema              : str
suite               : str
operator            : str
substrate           : str
job_id              : str
backend             : str
shots               : int
num_tiles           : int
tile_indices        : int array
tile_delay_dt       : int/float array
tile_scale_level    : int/float array
tile_theta          : float array
tile_mode           : string array
tile_role           : string array

ctrl                : uint8, shape (tiles, shots)
g                   : uint8, shape (tiles, shots, bits)
em                  : uint8, shape (tiles, shots, bits)
scale               : uint8, shape (tiles, shots)
branch              : uint8, shape (tiles, shots, 2)
delta               : int8,  shape (tiles, shots, bits)
xor_delta           : uint8, shape (tiles, shots, bits)
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

Generated QPU base files use the pattern:

```text
data/fm_job_<JOB_ID>.npz
```

Generated GPU base files use the pattern:

```text
data/fm_gpu_data_<...>.npz
```

Latest pointers:

```text
data/latest_fm_qpu_data.json
data/latest_fm_gpu_data.json
```

This shared schema is load-bearing.

It allows the same projector code to consume QPU and GPU bases without changing the operator.

---

## 6. Circuit and record structure

`F_M` begins with a paired-path QPU circuit.

The circuit is designed so each tile produces two matched path records:

```text
g path
em path
```

The model language used during design was:

```text
g  = gravity-like / delay-sensitive path
em = electromagnetic-like / phase-sensitive path
```

That language motivated the circuit. It is not the benchmark claim.

The benchmark claim is:

```text
The paired-delay circuit produced a stable differential wave signature.
```

Each tile has delay and scale metadata. The first locked QPU base used:

```text
tile_delay_dt    = [0, 1, 2, 4, 8, 16, 0]
tile_scale_level = [1, 1, 1, 1, 1, 1, 2]
shots            = 4096
tiles            = 7
backend          = ibm_marrakesh
```

The reason the delay metadata matters is that the primary F_M signature is not just a scalar statistic. It is a delay-ordered wave-like curve.

The reason the paired paths matter is that the primary destructive control is:

```text
path_pair_break
```

If the paired relation between `g` and `em` is destroyed, the primary wave signature weakens.

---

## 7. Differential fields

The two raw paths are not the final operator.

The useful F_M signal lives in the differential fields:

```text
delta     = em - g
xor_delta = em XOR g
```

### 7.1 Signed differential field

```text
delta[t, shot, bit] = em[t, shot, bit] - g[t, shot, bit]
```

This field takes values:

```text
-1, 0, 1
```

### 7.2 Binary differential field

```text
xor_delta[t, shot, bit] = em[t, shot, bit] XOR g[t, shot, bit]
```

This field takes values:

```text
0, 1
```

### 7.3 Primary locked field

The final locked primary field is:

```text
xor_delta
```

The final locked primary response is:

```text
bit_diff
```

The final locked primary order is:

```text
delay
```

So the core F_M signature is:

```text
xor_delta / bit_diff / delay
```

---

## 8. Response compression

Raw base files contain per-shot paired-path measurements.

The projector does not need to keep the full shot record in the final metric stage. It compresses each tile into response curves.

Supported response kinds:

```text
mean
energy
transition
imbalance
bit0_mean
bit1_mean
bit_diff
```

The primary response is:

```text
bit_diff = bit1_mean - bit0_mean
```

For `xor_delta`, this is:

```text
R(t)
=
mean over shots of xor_delta[t, :, 1]
-
mean over shots of xor_delta[t, :, 0]
```

That response is then sorted by tile delay:

```text
delay order
```

and passed into the wave metric kernel.

This compression is the practical bridge from shot records to a reusable F_M operator.

---

## 9. Wave metric

The `F_M` benchmark evaluates ordered response curves through a wave metric.

Input:

```text
curve C[0:N]
x coordinates xs[0:N]
```

Computed quantities:

```text
peak_ratio
spectral_entropy
best_r2
best_freq
best_amp
best_phase
low_high_ratio
wave_score
```

The composite wave score is:

```text
wave_score
=
0.40 * peak_ratio
+
0.25 * max(0, best_r2)
+
0.20 * (1 - spectral_entropy)
+
0.15 * min(1, abs(low_high_ratio) / 10)
```

This score rewards:

```text
spectral concentration
sinusoidal fit
low spectral entropy
coarse low/high frequency structure
```

The primary qproj value was:

```text
xor_delta / bit_diff / delay

score = 0.6571
peak  = 0.769
R2    = 0.819
freq  = 1.30
amp   = 0.05800
```

---

## 10. Geometry channel and projection channels

The `F_M` benchmark evaluates the same operator family through three linked channels.

```text
qproj channel = QPU paired-path record
gproj channel = GPU paired-path record
geo channel   = analytic metadata path
```

### 10.1 QPROJ channel

The QPROJ channel evaluates:

```text
QPU g/em record
  -> delta/xor_delta response curves
  -> delay ordering
  -> wave metrics
```

This is the discovery substrate.

### 10.2 GPROJ channel

The GPROJ channel evaluates:

```text
GPU-generated g/em record
  -> delta/xor_delta response curves
  -> delay ordering
  -> wave metrics
```

This is the controlled generated substrate.

### 10.3 GEO channel

The GEO channel evaluates:

```text
metadata
  -> analytic delay-wave curves
  -> wave metrics
```

This is the optimized classical operator path.

### 10.4 Why all three channels exist

They answer different questions:

```text
QPROJ:
    Did real hardware produce the signature?

GPROJ:
    Can a compatible generated base reproduce the signature family?

GEO:
    Can the useful signature be computed directly and quickly?
```

Together they establish:

```text
QPROJ discovers it.
GPROJ reproduces it.
GEO computes it directly.
```

That is the core architecture.

---

## 11. Current CUDA architecture

The core CUDA source is:

```text
kernels/fm_projector_kernel.cu
```

The current benchmark compiles one CUDA module from this source.

The production path uses these kernels:

```text
fm_response_kernel_u8
fm_path_pair_break_response_kernel_u8
fm_geo_curve_kernel_f32
fm_geo_sweep_kernel_f32
fm_wave_metric_kernel_f32
```

### 11.1 `fm_response_kernel_u8`

Role:

```text
record-based qproj/gproj response compression
```

Inputs:

```text
g
em
field_kinds
response_kinds
```

Outputs:

```text
response[field, response, tile]
```

Supported fields:

```text
delta
xor_delta
g
em
```

Supported responses:

```text
mean
energy
transition
imbalance
bit0_mean
bit1_mean
bit_diff
```

This kernel is used by both QPROJ and GPROJ.

### 11.2 `fm_path_pair_break_response_kernel_u8`

Role:

```text
paired-path destruction control
```

It independently permutes the `g` and `em` shot indices before recomputing differential fields.

This tests whether the F_M signature depends on matched path pairing.

### 11.3 `fm_geo_curve_kernel_f32`

Role:

```text
optimized classical geo curve generation
```

Consumes metadata:

```text
tile_delay_dt
tile_scale_level
tile_theta
mode_id
order_indices
```

and formula parameters:

```text
wave_freq
phase0
bitdiff_amp
bit1_amp
transition_amp
energy_amp
scale_phase
theta_phase
base_xor
base_delta
```

Outputs eight fixed geo curves:

```text
0 = xor_delta / bit_diff
1 = xor_delta / bit1_mean
2 = xor_delta / transition
3 = xor_delta / energy
4 = delta     / bit_diff
5 = delta     / bit1_mean
6 = delta     / transition
7 = delta     / energy
```

### 11.4 `fm_geo_sweep_kernel_f32`

Role:

```text
optimized parameter sweep for the geo path
```

Each block evaluates one candidate parameter set against all eight fixed geo curves and writes wave metrics.

This is what makes the geo finalizer fast.

Current measured speed:

```text
250,000 candidates in about 290 ms
```

### 11.5 `fm_wave_metric_kernel_f32`

Role:

```text
wave metric computation for precomputed curves
```

Inputs:

```text
curves
xs
```

Outputs:

```text
wave_score
peak_ratio
spectral_entropy
best_r2
best_freq
best_amp
best_phase
low_high_ratio
```

This kernel is shared by QPROJ, GPROJ, and GEO.

---

## 12. GEO formula architecture

The GEO path is the optimized analytic approximation of the locked F_M signature family.

It computes curves directly from metadata.

Conceptually:

```text
delay_norm = delay / max_delay

phase =
    2*pi*wave_freq*delay_norm
    + phase0
    + scale_phase*log2(scale + 1)
    + theta_phase*theta
    + mode_phase
```

The primary geo curve is:

```text
xor_delta / bit_diff
=
base_xor
+
bitdiff_amp * sin(phase)
```

The transition phase uses a secondary frequency:

```text
phase2 =
    2*pi*(wave_freq + 1.1)*delay_norm
    + 0.5*phase0
    + scale_phase*log2(scale + 1)
    + 0.5*mode_phase
```

The GEO path is not a shot simulator.

It is the optimized classical form of the discovered operator-facing response.

---

## 13. Calibration and sweep

`F_M` does not use the same kind of calibration as `G_M`.

There are no bucket masks or threshold squelch parameters in the F_M final benchmark.

Instead, the geo path uses a parameter sweep.

The sweep searches:

```text
wave_freq
phase0
bitdiff_amp
bit1_amp
transition_amp
energy_amp
scale_phase
theta_phase
base_xor
base_delta
```

against target rows discovered from qproj:

```text
xor_delta / bit_diff / delay
xor_delta / bit1_mean / delay
xor_delta / transition / delay
delta     / transition / delay
```

The loss compares:

```text
wave_score
peak_ratio
best_r2
best_freq
best_amp
```

against qproj target values.

The canonical sweep is run by:

```text
probes/f_m_probe06_geo_cuda_finalizer.py
```

and included in the final benchmark by:

```text
F_M_final_benchmark.py
```

The architecture rule is:

```text
The sweep may tune GEO parameters,
but it must not change the locked operator definition.
```

---

## 14. Canonical data flow

End-to-end `F_M` package flow:

```text
QPU hardware path
    f_m_qpu_generate.py
        -> IBM Runtime job
        -> dumped base
        -> data/fm_job_<JOB_ID>.npz

GPU/generated path
    f_m_gpu_generate.py
        -> generated paired-path base
        -> data/fm_gpu_data_<...>.npz

Shared benchmark path
    F_M_final_benchmark.py
        -> load qproj base
        -> load gproj base
        -> compile fm_projector_kernel.cu
        -> run QPROJ response + metric
        -> run GPROJ response + metric
        -> run GEO curve + metric
        -> run adjacent classical readers
        -> save result JSON / CSV
```

Inside the benchmark:

```text
qproj/gproj records
    -> response kernel
    -> delay-ordered curves
    -> wave metric kernel

geo metadata
    -> geo curve kernel
    -> wave metric kernel

primary curve
    -> FFT / DCT / autocorr / sinfit baselines
```

The current architecture intentionally keeps data movement simple:

```text
base files in data/
CUDA kernels in kernels/
research probes in probes/
final claims from F_M_final_benchmark.py
```

---

## 15. Current benchmark stages

The canonical final benchmark has three major stages.

```text
1. substrate signature comparison
2. speed comparison
3. adjacent classical baseline comparison
```

### 15.1 Substrate signature comparison

The substrate comparison runs:

```text
QPROJ
GPROJ
GEO
```

Primary locked signature:

```text
xor_delta / bit_diff / delay
```

Current final benchmark values:

```text
QPROJ:
    score = 0.6571
    peak  = 0.769
    R2    = 0.819
    freq  = 1.30
    amp   = 0.05800

GPROJ:
    score = 0.6796
    peak  = 0.772
    R2    = 0.986
    freq  = 0.90
    amp   = 0.03837

GEO:
    score = 0.7356
    peak  = 0.812
    R2    = 0.988
    freq  = 1.10
    amp   = 0.04813
```

The read:

```text
The same signature family survives across qproj, gproj, and geo.
```

### 15.2 Speed comparison

Current final benchmark speed values:

```text
QPROJ response + metric : 1.045625 ms
GPROJ response + metric : 1.059980 ms
GEO curve + metric      : 0.382735 ms
GEO 250k sweep          : 290.061 ms
```

Approximate speedup:

```text
GEO is about 2.7x faster than QPROJ projector evaluation.
GEO is about 2.8x faster than GPROJ projector evaluation.
```

The read:

```text
GEO is the optimized path for runtime evaluation.
QPROJ and GPROJ are record-based substrate paths.
```

### 15.3 Adjacent classical baselines

The classical adjacent baselines are run on the primary curve:

```text
xor_delta / bit_diff / delay
```

They are not full F_M substrate paths.

Current values:

```text
FFT_GPU:
    score = 0.7688
    time  = 0.459 ms

DCT_GPU:
    score = 0.6047
    time  = 0.477 ms

AUTOCORR_GPU:
    score = 0.4519
    time  = 1.010 ms

SINFIT_GPU:
    score = 0.8193
    time  = 44.191 ms
```

The read:

```text
FFT and SinFit are strong readers of the already-built curve.
F_M GEO constructs the operator-specific curve from metadata and scores it.
```

That distinction is load-bearing.

---

## 16. Valid claim boundary

The `F_M` architecture supports the following claims.

### Supported

```text
F_M has a real QPU paired-path qproj base.
F_M has a GPU-generated paired-path gproj base.
F_M has an optimized classical geo path.
The same benchmark compares qproj / gproj / geo.
The useful signal is differential, not raw-path marginal.
The locked signature is xor_delta / bit_diff / delay.
Path-pair breaking weakens the signal.
Delay shuffling weakens the signal.
GEO computes the signature faster than record-based qproj/gproj evaluation.
Adjacent classical readers can score the primary curve, but are not full substrate paths.
```

### Not supported

```text
F_M proves literal gravitational or electromagnetic hardware cavities.
F_M proves quantum advantage.
F_M universally replaces FFT / DCT / wavelet / sinusoid analysis.
F_M qproj, gproj, and geo are numerically identical.
One frequency/amplitude is guaranteed across all future QPU jobs.
The seven-tile base is enough for high-resolution frequency estimation.
```

This is the claims discipline the Converger roadmap requires.

---

## 17. Why the architecture is finished for this version

`F_M` is finished for this version because every required operator-package element exists.

```text
operator math               : complete
QPU base generator          : complete
GPU base generator          : complete
geometry path               : complete
canonical benchmark runner  : complete
CUDA projector kernel       : complete
CUDA geo path               : complete
CUDA geo sweep              : complete
control logic               : complete
speed comparison            : complete
adjacent baselines          : complete
saved benchmark output      : complete
known limits                : documented
```

The next work is not to keep mutating `F_M` indefinitely.

The next work is to use `F_M` as another completed pattern for the remaining Converger operators and the eventual node-point network.

Each future operator should follow the same discipline:

```text
define the operator
define qproj / gproj / geo
generate bases
benchmark under controls
scramble the channel
measure what survives
make only bounded claims
```

---

## 18. Why not...

### Why not call this proof of hardware gravity or electromagnetic cavities?

Because the benchmark does not establish that.

The cavity language motivated the circuit. The measured claim is that a paired-delay circuit produced a stable differential wave signature that survives the qproj/gproj/geo pipeline.

### Why not claim quantum advantage?

Because the benchmark does not show that.

The QPU path is valuable because it provides a physical substrate record. The optimized path is the GEO CUDA path. The claim is substrate linkage and operator extraction, not quantum speedup.

### Why not use only GEO?

Because GEO alone does not explain where the operator came from.

The QPU base discovered the signature. GPROJ showed a compatible generated base could reproduce the signature family. GEO made the useful path fast.

The architecture needs all three for the full story.

### Why not use only QPROJ?

Because QPROJ is record-based and hardware-shot-limited.

QPROJ is the discovery substrate. GEO is the optimized runtime path.

### Why not use only FFT?

Because FFT reads the curve after it exists.

F_M GEO constructs the operator-specific curve from metadata and then scores it. FFT is a strong adjacent reader, but not a substrate path.

### Why not trust the signature without controls?

Because the controls are what make the claim defensible.

The path-pair break and delay-shuffle controls show that the signal depends on paired-path structure and delay order.

Without those controls, the benchmark would be much weaker.

---

## 19. File responsibilities

```text
README.md
    Human-facing summary and current benchmark claims.

docs/math.md
    Mathematical definitions for paired-path fields, response curves,
    wave metrics, qproj/gproj/geo, and final benchmark interpretation.

docs/architecture.md
    This document. System design for the finished F_M operator package.

docs/known_issues.md
    Current limitations and failure modes.

docs/future_directions_whitepaper.md
    Larger Converger roadmap.

F_M_final_benchmark.py
    Canonical final benchmark runner for F_M.

f_m_gpu_generate.py
    GPU-compatible paired-path gproj base generator.

f_m_qpu_generate.py
    QPU paired-path generator / submission / dump path.

kernels/fm_projector_kernel.cu
    CUDA response, path-pair-break, wave metric, geo curve,
    and geo sweep kernels.

data/
    Frozen qproj/gproj base files and latest pointers.

examples/
    Optional examples retained for continuity.

probes/
    Research trajectory and forensic record.
```

---

## 20. Summary

`F_M` is the completed Fractal / Frequency / Field Metric operator package for this version of Ghost Oracle Suite.

It demonstrates the operator-package pattern the future Converger architecture will use:

```text
qproj
gproj
geo
benchmark
controls
bounded claims
```

The current architecture says:

```text
F_M qproj discovers the paired-path differential wave signature.
F_M gproj reproduces the signature family in a compatible GPU base.
F_M geo computes the signature directly as an optimized classical path.
F_M controls show the signature depends on path pairing and delay order.
FFT/SinFit are adjacent readers, not full substrate paths.
```

The larger roadmap says:

```text
Use this completed pattern to build the remaining ghost-channel operators
and eventually the node-point network.
```

That is the architectural handoff.

The process is the process:

```text
freeze the record
build controls
scramble the channel
compare substrates
extract the minimal math
measure what survives
```
