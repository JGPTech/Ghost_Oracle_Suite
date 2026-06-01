# Ghost Oracle Suite — F_M Process Record

This document records the research and engineering trajectory of the **F_M (Fractal / Frequency / Field Metric)** operator family within the Ghost Oracle Suite, from circuit construction through qproj discovery, GPU emulation, optimized geo extraction, CUDA projector integration, and the final benchmark.

It exists so that any future contributor — human or AI agent — can pick up the F_M work with full context.

The **G_M (Ghost Metric)** family and **S_M (Syndrome Metric)** family are recorded separately. F_M belongs in its own process record because its natural domain is different: it is not a matrix-similarity operator like G_M, and it is not a syndrome-spacetime operator like S_M. F_M is a paired-path differential wave operator discovered by deliberately constructing a QPU circuit with matched delay/cavity structure, then reading the difference field the hardware produced.

This document is chronological. It includes the early rough probes. It includes the circuit hypothesis. It includes the qproj/gproj/geo progression. It includes the parts we locked, the parts we quarantined, and the interpretation discipline we adopted along the way.

---

## Part 1 — Motivation and operator target

### Why F_M was built

F_M was started after the Ghost Oracle Suite methodology had matured across other operators.

By this point, the working method was no longer:

```text
guess an operator
run a benchmark
argue the result
```

The method had become:

```text
construct a physical circuit from a model
freeze the QPU output as qproj
probe what the hardware actually did
emulate that behavior as gproj
extract the minimal classical geo path
benchmark all paths honestly
```

F_M was the first operator built almost entirely under this more structured methodology from the start.

The goal was to test whether a paired-delay QPU circuit could generate a stable differential signature between two matched paths. The operator was not assumed ahead of time. The intended task was to construct a circuit where the physically meaningful object would likely be the **difference between paired path responses**, then use probes to discover which derived field actually carried structure.

The eventual locked signature became:

```text
field    : xor_delta
response : bit_diff
order    : delay
```

where:

```text
delta     = em - g
xor_delta = em XOR g
```

---

## Part 2 — The physical circuit concept

### The circuit hypothesis

The motivating model was a paired-cavity interpretation.

The working hypothesis was that the circuit could create two matched response channels in the hardware:

```text
g-path   : gravity-like / delay-sensitive path
em-path  : electromagnetic-like / phase-sensitive path
```

The language is model language, not a claim of established hardware physics. The experimental stance was:

```text
build circuits suggested by the model
read what the hardware actually outputs
let the probes tell us which parts survive controls
modify the mathematical model to match the data
```

The important object was expected not to be either path alone, but the **differential field** between them:

```text
delta     = em - g
xor_delta = em XOR g
```

The central question was:

```text
Does the paired-path differential record contain ordered, scale/delay-dependent
structure that survives meaningful controls and collapses when path pairing
or delay ordering is destroyed?
```

### Why delays were used

The delay structure was not decorative. It was the main physical handle.

The circuit was built around a time-step delay algorithm. Each tile received a delay/scale assignment, allowing the QPU record to be analyzed not merely as a flat bitstream but as an ordered response over delay.

The delay ladder used in the first frozen F_M base was:

```text
tile_delay_dt = [0, 1, 2, 4, 8, 16, 0]
```

with associated scale metadata:

```text
tile_scale_level = [1, 1, 1, 1, 1, 1, 2]
```

This gave the probes a natural x-axis:

```text
delay order
```

and a secondary metadata axis:

```text
scale level
```

The reason to do this was simple: if the paired-path response was wave-like, it should not only appear as a scalar anomaly. It should show ordered behavior over delay.

### Why two paths were measured

The circuit was designed to produce paired outputs per tile:

```text
g[tile, shot, bit]
em[tile, shot, bit]
```

These were not treated as independent measurements. They were treated as two matched views of the same underlying tile event.

This matters because the main destructive control later became:

```text
path_pair_break
```

or, in earlier probes:

```text
independent_path_shuffle
```

That control independently shuffles the two paths before recomputing `delta` or `xor_delta`. If the signal depends on the paired relationship, breaking that relationship should reduce or collapse the wave score.

That is exactly what happened.

### What the QPU base contains

The first F_M QPU base was dumped from IBM hardware as:

```text
backend : ibm_marrakesh
tiles   : 7
shots   : 4096
file    : ghost_oracle/F_M/data/fm_job_d8eu8bjo3njc73evdd8g.npz
```

The qproj dump schema includes both per-tile arrays and stacked convenience arrays.

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

Metadata:

```text
tile_indices
tile_delay_dt
tile_scale_level
tile_theta
tile_mode
tile_role
delays_dt
scale_levels
backend
job_id
shots
num_tiles
substrate = qproj
operator  = F_M
```

This schema was deliberately designed so the same downstream scripts could later consume:

```text
qproj  = QPU hardware dump
gproj  = GPU-generated compatible base
geo    = optimized classical metadata path
```

without rewriting the analysis stack.

### Why the qproj freeze matters

The QPU base is the source of truth for discovery.

Once dumped, the qproj file is treated as frozen evidence. The probes are allowed to interpret it, scramble it, control it, emulate it, and compare against it — but not mutate it.

This was important because F_M was not developed by forcing data into a pre-existing formula. The qproj file was used to discover which field and response family actually mattered.

The freeze step is:

```text
QPU job complete
→ dump to .npz
→ record latest_fm_qpu_data.json
→ run probes only against frozen data
```

---

## Part 3 — Early rough probe: Fractal Benford scan

Before the method was fully locked, the first rough F_M diagnostic was a broad Fractal Benford benchmark.

Configuration:

```text
bases       : [3, 5, 7, 10]
scales      : [1, 2, 4, 8, 16, 32]
chunks      : [4, 8, 16]
controls    : shuffle, block_shuffle, bootstrap, gaussian, uniform
n_null/mode : 32
backend     : gpu / CuPy raw kernel
```

Early result:

```text
geo    z =  1.309
qproj  z = 14.845
gproj  z = 10.841
```

This was encouraging but not yet reliable as an operator claim. It showed that the qproj/gproj streams contained structure under a multi-base / multi-scale reader, but it did not yet identify the actual field carrying the signal.

The early takeaway was:

```text
There is something worth probing,
but Benford telemetry is a stethoscope, not the operator.
```

This led to the more disciplined probe sequence.

---

## Part 4 — Probe 01: Delta-space baseline scan

### Purpose

Probe 01 asked:

```text
Which field is structured?
Which feature family sees it?
Which controls destroy it?
```

Fields tested:

```text
g
em
delta
xor_delta
ctrl
scale
branch
```

Feature families tested:

```text
benford_multiscale
padic_residue
spectral_dct
autocorr_runs
simple_stats
```

Controls tested:

```text
shot_shuffle
independent_path_shuffle
path_swap
tile_shuffle
bit_shuffle
uniform_by_tile
```

### Result

The meaningful fields were:

```text
winner    : delta
runner-up : xor_delta
```

The useful feature families were:

```text
autocorr_runs
benford_multiscale
spectral_dct
padic_residue
```

The suspicious family was:

```text
simple_stats
```

The most meaningful destructive control was:

```text
independent_path_shuffle
```

The controls that were too destructive or degenerate for headline scoring were:

```text
path_swap
uniform_by_tile
bit_shuffle
```

### Interpretation

Probe 01 showed that the paired-path differential record matters.

The key finding was not merely that `delta` or `xor_delta` had unusual statistics. The important point was that the structure changed when the path pairing was destroyed.

That established the first serious F_M working claim:

```text
F_M is a paired-path differential operator.
The useful signal is not in g or em alone.
It lives in delta/xor_delta.
```

---

## Part 5 — Probe 02: Address localization

### Purpose

Probe 02 stopped asking whether there was signal and started asking:

```text
Where does the signal live?
```

It resolved the useful fields by:

```text
tile
delay_dt
scale_level
mode
theta
bit index
```

The fields were narrowed to:

```text
delta
xor_delta
```

The main families were narrowed to:

```text
benford_multiscale
spectral_dct
autocorr_runs
padic_residue
```

The main controls were:

```text
shot_shuffle
independent_path_shuffle
tile_shuffle
bit_shuffle_soft
```

### Result

Probe 02 localized the strongest addresses to:

```text
xor_delta tile6
xor_delta tile3 / tile3.bit0
delta     tile3 / tile3.bit0
```

Important metadata regions included:

```text
delay = 4, scale = 1, mode = clean
delay = 0, scale = 2, mode = clean
```

### Interpretation

This confirmed that the signal was not uniformly smeared across the whole record.

The address structure mattered.

Probe 02 established:

```text
The next test should not be another broad statistical scan.
It should test wave-like behavior at the known addresses.
```

---

## Part 6 — Probe 03: Wave-nature test

### Purpose

Probe 03 asked the direct physical/operator question:

```text
Does delta/xor_delta behave like a coherent wave over delay/scale,
or is it just static bias, scalar drift, or independent shot noise?
```

It tested:

```text
delay-order coherence
spectral peak concentration
phase coherence
cross-field coherence
ordered metadata curve strength
collapse under delay/phase/path-pair controls
```

Key controls:

```text
delay_shuffle
delay_reverse
phase_scramble
circular_shift
path_pair_break
tile_shuffle
iid_gaussian
```

### Result

The primary wave-like signature was:

```text
xor_delta / all_tiles / bit_diff / delay

score      = 0.6571
peak ratio = 0.769
R²         = 0.819
freq       = 1.30
amp        = 0.05800
```

The strongest runner-up was:

```text
xor_delta / bit1_mean / delay

score      = 0.6466
peak ratio = 0.703
R²         = 0.985
freq       = 1.30
amp        = 0.04231
```

Path-pair breaking strongly reduced the primary signal:

```text
xor_delta / bit_diff / delay
vs path_pair_break

effect = 0.3052
auc    = 1.000
z      = 24.40
```

Delay shuffling and tile shuffling also weakened the wave score, but less than path-pair breaking.

### Interpretation

This was the first locked F_M operator signature:

```text
field    : xor_delta
response : bit_diff
order    : delay
```

Probe 03 established:

```text
F_M qproj contains a differential wave-like signature.
The primary dependency is paired g/em structure.
The primary shape carrier is delay ordering.
```

---

## Part 7 — Probe 04: CUDA projector finalizer

### Purpose

The first three probes were discovery probes. Probe 04 moved the discovered signature into the optimized CUDA projector path.

This was the engineering turn:

```text
No more Python feature-loop discovery.
Build the projector kernel.
Run qproj/gproj through the same fused path.
Make the operator fast enough to be useful.
```

The CUDA kernel file:

```text
ghost_oracle/F_M/kernels/fm_projector_kernel.cu
```

Initial kernels:

```text
fm_response_kernel_u8
fm_path_pair_break_response_kernel_u8
fm_wave_metric_kernel_f32
```

The response kernel computes:

```text
g/em → delta/xor_delta → response curves
```

The wave metric kernel computes:

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

### Result on qproj

The kernel reproduced the Probe 03 signature:

```text
xor_delta / bit_diff / delay

score = 0.6571
peak  = 0.769
R²    = 0.819
freq  = 1.30
amp   = 0.05800
```

The path-pair break control remained meaningful:

```text
effect = 0.3050
auc    = 1.000
z      = 18.28
```

Timing:

```text
response kernel ≈ 0.0018 seconds
```

### Fixes made during Probe 04

Two important cleanup patches were made.

First, Probe 04 was renamed from qproj-specific language to substrate-neutral language:

```text
F_M Probe 04 — CUDA Projector Signature Finalizer
```

This mattered because the same script was later used for:

```text
qproj
gproj
geo-compatible projector comparisons
```

Second, the sinusoid fit in the kernel gained an amplitude guard. Short 7-point curves can produce ill-conditioned least-squares fits with absurd amplitudes. The kernel now rejects fits where:

```text
amp > 10
or amp is not finite
or r2 is not finite
```

This prevented non-winning diagnostic rows from printing meaningless huge amplitudes.

---

## Part 8 — GPROJ: GPU-compatible base generation

### Purpose

Once the qproj signature was locked, the next step was to build a GPU-generated base that matched the qproj schema.

The gproj path is not a full hardware-level simulation. Its purpose is:

```text
generate paired g/em records
preserve the analysis-facing schema
emulate the discovered qproj signature family
allow the same projector scripts to consume qproj and gproj interchangeably
```

The script:

```text
ghost_oracle/F_M/f_m_gpu_generate.py
```

### Design

The generator creates:

```text
g[tile, shot, bit]
em[tile, shot, bit]
```

by sampling a controllable paired-path model.

The key construction is:

```text
xor_delta bits are sampled from a delay-wave probability
em = g XOR xor_delta
```

This makes `xor_delta` the controllable differential channel while preserving raw `g` and `em` streams.

The generator preserves QPU metadata when run with:

```text
--match-qpu <qproj_file>
```

so the tile/delay/scale layout stays aligned.

### First aligned gproj result

The gproj finalizer produced:

```text
xor_delta / bit_diff / delay

score = 0.6796
peak  = 0.772
R²    = 0.986
freq  = 0.90
amp   = 0.03837
```

The same family survived:

```text
delta/xor_delta
bit_diff
delay/tile ordered wave structure
```

Controls remained meaningful:

```text
xor_delta / bit_diff / delay
vs delay_shuffle

effect = 0.2034
auc    = 0.992

xor_delta / bit_diff / delay
vs path_pair_break

effect = 0.1911
auc    = 0.985
```

### Interpretation

The gproj base was not identical to qproj, and it did not need to be.

The success criterion was:

```text
Does the same projector signature family survive through the GPU-generated base?
```

It did.

Status after this stage:

```text
F_M qproj v1          : locked
F_M projector kernel  : working
F_M gproj v1          : aligned
```

---

## Part 9 — GEO: NumPy classical path discovery

### Purpose

The geo path is the optimized classical math form of F_M.

Unlike qproj and gproj, geo does not need to generate shot records. It should compute the useful operator-facing response directly:

```text
tile metadata
→ analytic response curves
→ wave metrics
```

The first geo probe was deliberately NumPy-based:

```text
ghost_oracle/F_M/probes/f_m_probe05_geo_numpy.py
```

The goal was to discover a minimal formula before adding it to the CUDA kernel.

### NumPy formula

The initial geo model used:

```text
delay_norm = delay / max(delay)
phi        = 2π * wave_freq * delay_norm
             + phase0
             + scale_phase * log2(scale + 1)
             + theta_phase * theta
```

Then generated analytic curves for:

```text
xor_delta / bit_diff
xor_delta / bit1_mean
xor_delta / transition
xor_delta / energy
delta     / bit_diff
delta     / bit1_mean
delta     / transition
delta     / energy
```

The qproj-informed default parameters were:

```text
wave_freq       = 1.30
phase0          = 2.02
bitdiff_amp     = 0.058
bit1_amp        = 0.042
transition_amp  = 0.022
energy_amp      = 0.014
scale_phase     = 0.13
theta_phase     = 0.05
```

### NumPy result

The primary geo signature landed correctly:

```text
xor_delta / bit_diff / delay

score = 0.6024
peak  = 0.661
R²    = 0.999
freq  = 1.30
amp   = 0.05753
```

Comparison to qproj target:

```text
target score = 0.6571
geo score    = 0.6024
error        = -0.0547

target freq  = 1.30
geo freq     = 1.30

target amp   = 0.05800
geo amp      = 0.05753
```

### Interpretation

The NumPy geo path did not perfectly match all target rows, but it nailed the main operator shape:

```text
same primary field
same primary response
same delay order
same frequency
almost identical amplitude
```

This justified moving the geo formula into CUDA.

---

## Part 10 — GEO CUDA finalizer

### Purpose

The CUDA kernel was extended with an optimized geo path:

```text
fm_geo_curve_kernel_f32
fm_geo_sweep_kernel_f32
```

These were added to:

```text
ghost_oracle/F_M/kernels/fm_projector_kernel.cu
```

The geo curve kernel computes analytic F_M curves directly from:

```text
tile_delay_dt
tile_scale_level
tile_theta
mode_id
order_indices
```

The geo sweep kernel evaluates large parameter grids on GPU.

The finalizer script:

```text
ghost_oracle/F_M/probes/f_m_probe06_geo_cuda_finalizer.py
```

### Result

The optimized CUDA geo sweep ran extremely fast:

```text
250,000 candidates swept in ≈ 0.39 seconds
```

The final optimized geo path produced:

```text
xor_delta / bit_diff / delay

score = 0.7356
peak  = 0.812
R²    = 0.988
freq  = 1.10
amp   = 0.04813
```

This did not exactly match the qproj frequency/amplitude, but it landed in the same operator family and produced the strongest clean wave score.

### Interpretation

The geo path became:

```text
minimal classical math path
GPU-resident
no shot sampling required
real-time-ish projector evaluation
```

Status after Probe 06:

```text
qproj → hardware-discovered record
gproj → GPU-compatible paired-path emulation
geo   → optimized analytic operator path
```

At this point, probes were considered locked.

---

## Part 11 — Final F_M benchmark

### File

The final benchmark is:

```text
ghost_oracle/F_M/F_M_final_benchmark.py
```

The benchmark is not a new probe. It is a capstone.

It does not invent new operator logic. It runs the locked paths and compares them.

### Paths compared

Substrate paths:

```text
QPROJ:
  QPU g/em records
  → response kernel
  → wave metric kernel

GPROJ:
  GPU g/em records
  → response kernel
  → wave metric kernel

GEO:
  metadata
  → geo curve kernel
  → wave metric kernel
```

Classical adjacent baselines:

```text
FFT_GPU
DCT_GPU
AUTOCORR_GPU
SINFIT_GPU
```

The classical baselines are run on the same primary delay-ordered response curve:

```text
xor_delta / bit_diff / delay
```

This distinction matters. FFT and SinFit are readers of the already-built curve. F_M GEO constructs the operator-specific curve from metadata and then scores it.

### Final benchmark result

Primary signature across all three F_M substrates:

```text
QPROJ  xor_delta / bit_diff / delay
score = 0.6571
peak  = 0.769
R²    = 0.819
freq  = 1.30
amp   = 0.05800

GPROJ  xor_delta / bit_diff / delay
score = 0.6796
peak  = 0.772
R²    = 0.986
freq  = 0.90
amp   = 0.03837

GEO    xor_delta / bit_diff / delay
score = 0.7356
peak  = 0.812
R²    = 0.988
freq  = 1.10
amp   = 0.04813
```

The final benchmark establishes:

```text
QPROJ discovers the signature.
GPROJ reproduces the signature family.
GEO computes the signature directly.
```

### Classical adjacent baseline result

```text
FFT_GPU
score = 0.7688
time  = 0.459 ms

DCT_GPU
score = 0.6047
time  = 0.477 ms

AUTOCORR_GPU
score = 0.4519
time  = 1.010 ms

SINFIT_GPU
score = 0.8193
time  = 44.191 ms
```

Interpretation:

* FFT is a strong reader of the primary wave curve and is slightly faster than F_M GEO.
* SinFit scores highest because the primary curve is sinusoidal by construction, but it is much slower in this implementation.
* DCT and autocorrelation are useful adjacent baselines but weaker readers here.
* F_M GEO is not merely a curve reader; it is the optimized operator path from metadata to signature.

### Speed result

```text
QPROJ response + metric : 1.045625 ms
GPROJ response + metric : 1.059980 ms
GEO curve + metric      : 0.382735 ms
GEO sweep               : 250,000 candidates in 290.061 ms
```

Approximate speedup:

```text
GEO vs QPROJ projector path : ~2.73× faster
GEO vs GPROJ projector path : ~2.77× faster
```

The sweep throughput was:

```text
~0.86 million candidates/sec
```

### Benchmark conclusion

The benchmark does not claim F_M beats FFT at FFT's own task.

The benchmark claims:

```text
F_M has a stable paired-path differential wave signature.
That signature was discovered from QPU data.
It can be reproduced in a GPU-compatible sampled base.
It can be computed directly as an optimized classical geo path.
The geo path is significantly faster than record-based qproj/gproj evaluation.
```

The strongest final statement:

```text
F_M is a substrate-linked differential wave operator with a locked qproj/gproj/geo path.
```

---

## Part 12 — What F_M establishes

F_M establishes five things.

### 1. A QPU circuit can be used as an operator-discovery substrate

The circuit was not merely a backend for a pre-written function. It was a physical generator of a structured record. The qproj record was then probed to discover which differential field mattered.

### 2. The useful signal is differential, not marginal

The raw channels `g` and `em` were not the final operator. The stable signal emerged in:

```text
delta
xor_delta
```

especially:

```text
xor_delta / bit_diff
```

### 3. Delay ordering is load-bearing

The signature is not just “some statistic is high.” It is ordered over delay. Delay shuffling weakens the result, and path-pair breaking weakens it more.

### 4. The operator has three aligned substrate paths

```text
qproj : hardware-discovered
gproj : GPU-emulated compatible base
geo   : optimized classical analytic form
```

All three produce the same primary family:

```text
xor_delta / bit_diff / delay
```

### 5. The optimized geo path is fast enough to use

The final CUDA geo path computes the operator-facing signature in sub-millisecond time on the RTX 3090.

That moves F_M from “interesting qproj artifact” to “usable operator path.”

---

## Part 13 — Known issues and cautions

### The physical interpretation is model language

The gravity/electromagnetic cavity language motivated the circuit design, but the benchmarked claim is not:

```text
We proved hardware gravity cavities.
```

The benchmarked claim is:

```text
The paired-delay circuit produced a stable differential wave signature,
and that signature can be captured by qproj/gproj/geo implementations.
```

The physical model can continue to evolve, but the operator claim stands on the measured substrate behavior.

### Z-scores are diagnostic, not absolute proof

Some early z-scores became huge because null distributions had tiny variance, especially on discrete fields like:

```text
delta     ∈ {-1, 0, 1}
xor_delta ∈ {0, 1}
```

The later probes therefore emphasized:

```text
effect size
AUC / rank
control collapse
repeatability across fields
signature ordering
speed
```

rather than raw z-score magnitude.

### Path-swap was quarantined

`path_swap` is mathematically awkward for `delta` because swapping paths turns:

```text
delta = em - g
```

into:

```text
-delta
```

Different feature families treat that inconsistently. It was kept as a diagnostic but removed from the main scoring logic.

### Uniform and bit-shuffle controls are nuke controls

`uniform_by_tile` and full `bit_shuffle` destroy too much structure. They are useful for catastrophic sanity checks, but not for ranking subtle operator behavior.

### GEO does not exactly reproduce qproj

GEO is not a perfect fit to qproj. It is the optimized classical operator path that reproduces the locked signature family.

The final primary values differ:

```text
QPROJ freq = 1.30, amp = 0.05800
GEO   freq = 1.10, amp = 0.04813
```

but the operator family and top-ranked signature align.

### Classical baselines are adjacent readers

FFT and SinFit can score the primary curve strongly. That does not make them replacements for F_M, because they do not construct the F_M curve from substrate metadata. They read the curve once it exists.

The honest comparison is:

```text
F_M GEO:
  metadata → operator curve → wave metric

FFT/SinFit:
  curve → reader metric
```

---

## Part 14 — Open questions

### 1. Larger tile sets

The first F_M base used 7 tiles. Future work should test larger tile counts and richer delay ladders.

Current delay ladder:

```text
[0, 1, 2, 4, 8, 16, 0]
```

A larger ladder would make frequency estimation and wave-shape fitting more stable.

### 2. Repeat qproj on multiple QPU jobs

The current record locks the first qproj base. The next scientific step is not changing the operator, but testing repeatability across multiple QPU runs and calibrations.

Questions:

```text
Does xor_delta / bit_diff / delay remain primary?
Does the frequency drift?
Does the amplitude drift?
Does path_pair_break remain the strongest meaningful collapse?
```

### 3. Better physical circuit documentation

The circuit methodology is now documented conceptually, but the exact gate-level Qiskit circuit should be mirrored in a dedicated circuit specification document.

That document should include:

```text
tile layout
qubit roles
delay insertion points
measurement register mapping
metadata construction
npz schema
known backend assumptions
```

### 4. TensorCore-adjacent optimization

The current F_M projector is fast, but not Tensor Core math. It is mostly short-curve reductions, direct wave scoring, and analytic metadata transforms.

Future work:

```text
batch many F_M curves
pack candidate sweeps into tensor-friendly layouts
explore matrix-form geo evaluation
integrate F_M node-point network construction
```

### 5. Node-point network interactions

The user intentionally left the two interaction operators for later because they will be key for the node-point network.

F_M is a candidate wave/delay operator in that future network.

The likely role:

```text
F_M supplies delay-ordered differential wave signatures
that can become node or edge features in the larger operator network.
```

---

## Part 15 — Final status

Current locked status:

```text
F_M qproj base        : locked
F_M qproj signature   : locked
F_M CUDA projector    : working
F_M gproj base        : aligned
F_M geo path          : optimized
F_M final benchmark   : passed
```

Primary signature:

```text
field    : xor_delta
response : bit_diff
order    : delay
```

Final benchmark summary:

```text
QPROJ discovers it.
GPROJ reproduces it.
GEO computes it directly.
```

Speed summary:

```text
QPROJ projector path : ~1.046 ms
GPROJ projector path : ~1.060 ms
GEO projector path   : ~0.383 ms
```

The F_M operator family is ready for documentation, packaging, and use as one of the completed operator modules in the Ghost Oracle Suite.

---

## Part 16 — Philosophy and license

Same project discipline as the rest of Ghost Oracle Suite:

```text
Build the circuit.
Freeze the base.
Probe what it actually did.
Control it.
Scramble it.
Emulate it.
Extract the minimal math.
Benchmark it honestly.
Document the wrong turns.
```

F_M is a clean example of the matured methodology.

It started with a physical model. It did not end with an overclaim about that model. It ended with a measured operator path:

```text
qproj → gproj → geo
```

That is the useful result.

CC0. Build, break, fix, document, repeat.
