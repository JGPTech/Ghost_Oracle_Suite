# Architecture

Architecture for the `S_M` operator package.

`S_M` is the **Syndrome Metric** channel. It is the second completed operator package in the Ghost Oracle Suite Converger architecture pattern.

This document replaces the older architecture framing that treated `S_M` as a mixed workspace containing syndrome fields, stress tensors, token retrieval bridges, TSP examples, and downstream projector experiments. The new framing is:

```text
S_M is a finished operator package for this version.
S_M is one channel in the larger Converger roadmap.
S_M uses the standard geo / gproj / qproj substrate pattern.
S_M claims are made only through the benchmark runner and controls.
S_M does not include the stress tensor as its headline claim.
```

The math is in:

```text
docs/math.md
```

Known limitations are in:

```text
docs/known_issues.md
```

The examples and probe directories preserve continuity paths, but the current architecture claim comes from:

```text
s_m_benchmark.py
```

This architecture document explains how the finished `S_M` operator is wired together, what each file is responsible for, what counts as a valid benchmark claim, and how the package fits into the larger Converger direction without inflating the current result beyond what the benchmark supports.

---

## 1. Architectural status

`S_M` is complete for this version.

That means the package now has:

```text
1. a synthetic/reference field path,
2. a GPU-generated syndrome-spacetime base path,
3. a real QPU syndrome-spacetime dump path,
4. a canonical benchmark runner,
5. windowed field feature extraction,
6. destructive field controls,
7. control-source classification,
8. distance-prediction checks,
9. substrate agreement reporting,
10. CUDA-accelerated feature extraction,
11. documented math,
12. documented known limits.
```

The current package is not a placeholder for future work. It is a finished operator implementation with a bounded claim.

The bounded claim is:

```text
S_M is a field-structured, control-tested, substrate-comparable syndrome metric.
```

The claim is not:

```text
S_M is a logical-error-rate benchmark.
S_M is the stress tensor operator.
S_M proves quantum advantage.
S_M is a token retrieval benchmark.
S_M proves hardware speedup.
S_M makes every downstream field deformation valid.
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

Within that system, `S_M` is the syndrome-field component.

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

`S_M` is not the entire Converger.

`S_M` is the syndrome-spacetime field operator in that stack.

Its role is:

```text
syndrome field structure
final edge parity coupling
agreement-field measurement
real-vs-control separation
control-source classification
distance-structured field behavior
geo / gproj / qproj substrate comparison
```

Transformer-adjacent analogues include:

```text
token-level correction records
activation anomaly fields
retrieval disagreement traces
local error/correction histories
field-like model diagnostics
```

The `S_M` package does not replace those transformer components. It defines and benchmarks a syndrome-spacetime field operator that can later inform larger Converger components.

---

## 3. Standard operator package pattern

The future Converger architecture uses the same package shape for every operator:

```text
operator/
├── operator_gpu_generate.py
├── operator_qpu_generate.py
└── operator_benchmark.py
```

For `S_M`, the current package uses:

```text
S_M/
├── s_m_gpu_generate.py
├── s_m_qpu_generate.py
└── s_m_benchmark.py
```

Supporting directories:

```text
S_M/
├── data/
├── docs/
├── examples/
├── kernels/
└── probes/
```

The current main entry points are:

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

Legacy and downstream examples remain in the repo:

```bash
python examples/sm_windowed_knn_benchmark.py
python examples/sm_tsp_projector_example.py
python probes/sm_analyze.py
python probes/token_retrieval_projector.py
```

Those examples are retained for continuity and process history. They are not the current source of benchmark claims.

Current benchmark claims should come from:

```text
s_m_benchmark.py
```

and from saved benchmark output under:

```text
analysis/s_m_<timestamp>/
```

---

## 4. Standard substrate paths

`S_M` follows the standard three-substrate pattern used by the Converger roadmap.

```text
S_M^{geo}
S_M^{gproj}
S_M^{qproj}
```

In GitHub-safe plain text:

```text
S_M_geo    = synthetic/reference syndrome-spacetime field
S_M_gproj  = GPU-generated syndrome-spacetime base
S_M_qproj  = QPU / hardware syndrome-spacetime base
```

### 4.1 Geometry / reference path

The `geo` path is the synthetic/reference field model.

It generates controlled syndrome-spacetime records with the same downstream schema as real bases:

```text
data_d{d}
synd_d{d}
```

The `geo` path is not presented as a hardware simulator.

Its purpose is:

```text
provide a clean reference field
exercise the same feature and control pipeline
give the benchmark a substrate-independent comparison point
```

The reference field is the clean operator ceiling, not the hardware claim.

### 4.2 GPU-generated path

The `gproj` path uses a GPU-generated S_M base.

```text
S_M_gproj(X; B_g)
```

where `B_g` is a GPU-generated base file.

The GPU base is not an arbitrary baseline. It is a controlled syndrome-spacetime base with the same analysis schema as the QPU dump.

Its purpose is to separate:

```text
field-operator behavior
```

from:

```text
hardware noise, drift, backend calibration, queue timing, and physical execution effects
```

### 4.3 QPU path

The `qproj` path uses real IBM Runtime syndrome-spacetime records.

```text
S_M_qproj(X; B_q)
```

where `B_q` is a dumped QPU base file.

The QPU path is the hardware-derived syndrome-field substrate.

It is not used to claim raw throughput superiority. It is used to test whether final data edge parity and repeated syndrome records form a load-bearing field structure under destructive controls.

---

## 5. Bases

A **base** is a `.npz` file containing S_M field records.

The shared base schema is:

```text
schema        : str
job_id        : str
backend       : str, optional
shots         : int
rounds        : int
flag_level    : int
logical_init  : int
basis         : str
init_state    : str
distances     : int array

data_d{d}     : uint8, shape (shots, d)
synd_d{d}     : uint8, shape (shots, rounds, d-1)
flag_d{d}     : optional uint8, shape (shots, rounds, n_flags)
```

The same analysis schema is used for:

```text
real QPU S_M bases
GPU-generated S_M bases
synthetic/reference records inside the benchmark
```

This shared schema is load-bearing.

It allows the same benchmark code to consume QPU and GPU bases without changing the operator.

Generated QPU base files use the pattern:

```text
data/sm_data_<init_state>_<JOB_ID>.npz
data/sm_data_plus_<JOB_ID>.npz
```

Generated GPU base files use the pattern:

```text
data/sm_gpu_data_plus_<TAG>.npz
```

Benchmark output is written under:

```text
analysis/s_m_<timestamp>/
```

---

## 6. Syndrome-spacetime field

`S_M` is built from final data bits and repeated syndrome measurements.

The primary objects are:

```text
D[i]   = final data bit at code position i
E[i]   = D[i] XOR D[i+1]
S[t,i] = measured syndrome bit at round/time t and edge i
A[t,i] = 1 - (S[t,i] XOR E[i])
```

where:

```text
i = edge index
t = syndrome round / time index
```

The operator is not just:

```text
mean syndrome rate
```

The field relation matters.

The agreement field:

```text
A[t,i]
```

measures whether the repeated syndrome record agrees with the final data edge parity.

This makes `S_M` a field operator:

```text
final data edge parity
    +
syndrome spacetime
    +
agreement structure
```

The benchmark then asks whether that structure survives controls.

---

## 7. Detection events and agreement profiles

The benchmark uses two core derived S_M quantities.

### 7.1 Detection events

Detection events are temporal syndrome transitions:

```text
X[t,i] = S[t+1,i] XOR S[t,i]
```

They measure whether the syndrome field changes between adjacent rounds.

Detection rates alone are not the main claim, but they provide a useful baseline feature family.

### 7.2 Agreement profiles

Agreement profiles summarize the agreement field over edge and time axes:

```text
edge profile:
    mean A[t,i] over shots and rounds for each edge i

time profile:
    mean A[t,i] over shots and edges for each round t
```

These profiles are central to the current S_M benchmark.

The strongest current result is that agreement and field-aware features separate real QPU records from destructive controls while raw scalar rates remain near chance.

---

## 8. Feature families

The current benchmark computes five feature families.

```text
raw_rates
detection_rates
agreement_profiles
sm_field
sm_all
```

### 8.1 `raw_rates`

Raw scalar-like rates:

```text
mean final data bits
mean syndrome bits
```

This is a baseline.

It intentionally tests whether the benchmark is only reading trivial rate differences.

### 8.2 `detection_rates`

Temporal transition rates:

```text
mean X[t,i]
```

where:

```text
X[t,i] = S[t+1,i] XOR S[t,i]
```

This is another baseline feature family.

### 8.3 `agreement_profiles`

Field-aware edge/time agreement summaries:

```text
agreement edge profile
agreement time profile
```

This is the first core S_M feature family.

### 8.4 `sm_field`

Flattened field representation:

```text
agreement field mean per (round, edge)
detection field mean per (round, edge)
compact S_M field descriptors
```

This is the main field feature.

### 8.5 `sm_all`

Combined S_M feature set:

```text
sm_all = raw_rates + detection_rates + agreement_profiles + sm_field
```

This is the widest current S_M feature family.

---

## 9. Current CUDA architecture

The core CUDA source is:

```text
kernels/sm_kernel.cu
```

The current benchmark uses this kernel when CuPy and CUDA are available.

The primary kernel is:

```text
sm_window_features_kernel
```

Role:

```text
windowed S_M feature extraction
```

It computes:

```text
raw_rates
detection_rates
agreement_profiles
sm_field
```

The benchmark assembles:

```text
sm_all
```

from the CUDA outputs.

### 9.1 `sm_window_features_kernel`

Role:

```text
compute windowed S_M features from data_d{d} and synd_d{d}
```

Inputs:

```text
data  : uint8, shape (shots, d)
synd  : uint8, shape (shots, rounds, d-1)
```

Outputs:

```text
raw_out
det_out
agree_prof_out
sm_field_out
```

The kernel computes:

```text
terminal edge parity
agreement field
detection events
windowed data rates
windowed syndrome rates
windowed detection rates
windowed agreement profiles
windowed field summaries
```

### 9.2 CUDA boundary

The CUDA boundary is intentionally narrow.

Included:

```text
final edge parity
syndrome field
agreement field
detection events
windowed feature reductions
```

Excluded:

```text
stress tensor
gradient tensor
token retrieval
TSP field deformation
```

Those belong to separate operator or example paths.

### 9.3 Fallback path

If CUDA is unavailable, the benchmark falls back to the NumPy reference path.

Useful flags:

```bash
python s_m_benchmark.py --no-cuda
python s_m_benchmark.py --cuda-debug
```

The debug flag reports:

```text
CuPy status
kernel path
kernel existence
kernel compile status
CUDA device information
```

---

## 10. Windowed aggregation

`S_M` is a windowed field operator.

The benchmark does not treat one shot as the full object. It groups shots into windows and computes field features over each window.

Default windows:

```text
8
16
32
64
```

Probe mode can include:

```text
1
2
4
8
16
32
64
128
```

Windowing matters because `S_M` measures statistical field structure:

```text
final data edge parity
syndrome spacetime
agreement field
detection-event field
```

A single shot can contain a data/syndrome record, but the stable operator signature emerges more clearly over windows.

The current strongest real-vs-control results occur at:

```text
window = 64
```

This is not a universal constant. It is the current operating point for the tested bases.

---

## 11. Field controls

The benchmark uses destructive controls to test whether S_M field structure is load-bearing.

Control modes:

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

### 11.1 `real`

The intact record.

```text
data and syndrome field unchanged
```

### 11.2 `shot_shuffle_synd`

Destroys shot-level pairing between final data and syndrome field.

Preserves:

```text
syndrome marginal structure
```

Destroys:

```text
which final data record belongs with which syndrome spacetime record
```

### 11.3 `time_shuffle_synd`

Destroys temporal ordering.

Preserves:

```text
per-edge syndrome values
```

Destroys:

```text
round/time structure
```

### 11.4 `edge_shuffle_synd`

Destroys spatial / edge ordering.

Preserves:

```text
per-time syndrome values
```

Destroys:

```text
edge-local structure
```

### 11.5 `uniform_synd`

Replaces syndrome field with uniformized samples matching broad rates.

Preserves:

```text
approximate syndrome probability envelope
```

Destroys:

```text
structured syndrome spacetime
```

### 11.6 `final_shuffle`

Destroys final data / syndrome pairing.

Preserves:

```text
syndrome field
final data marginal distribution
```

Destroys:

```text
final edge parity alignment with syndrome spacetime
```

### 11.7 `all_uniform`

Uniformizes data and syndrome records.

Preserves:

```text
broad probability envelope
```

Destroys:

```text
data structure
syndrome structure
agreement structure
```

### 11.8 `time_reverse_synd`

Reverses the time axis.

Preserves:

```text
time content
```

Changes:

```text
forward temporal orientation
```

### 11.9 `edge_reverse_synd`

Reverses the edge axis.

Preserves:

```text
edge content
```

Changes:

```text
spatial orientation
```

The key read is:

```text
real field > destructive controls
```

For the current run, the clearest result is:

```text
agreement/field features separate real from controls
raw scalar-like rates stay near chance
```

---

## 12. Substrate agreement

The benchmark writes:

```text
substrate_agreement.csv
```

This compares field profiles across available substrates.

Profile families include:

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

Substrate agreement is a diagnostic.

It is not a throughput claim.

It is not a proof that the synthetic field is identical to the QPU field.

It is a way to compare profile structure across:

```text
geo
gproj
qproj
```

---

## 13. Canonical data flow

End-to-end `S_M` package flow:

```text
QPU hardware path
    s_m_qpu_generate.py submit
        -> IBM Runtime job
        -> metadata JSON

    s_m_qpu_generate.py dump <JOB_ID>
        -> completed job result
        -> data/sm_data_<init_state>_<JOB_ID>.npz

GPU/generated path
    s_m_gpu_generate.py
        -> generated syndrome-spacetime base
        -> data/sm_gpu_data_plus_<TAG>.npz

Shared benchmark path
    s_m_benchmark.py
        -> load bases
        -> build synthetic/reference geo records
        -> construct destructive controls
        -> compute windowed S_M features
        -> run classification tasks
        -> write result JSON / CSV / artifacts
```

Inside the benchmark:

```text
data_d{d}, synd_d{d}
    -> controls
    -> windows
    -> S_M feature extraction
    -> Task A: real-vs-control
    -> Task B: control-source classification
    -> Task C: distance prediction
    -> substrate agreement
    -> saved outputs
```

The current architecture intentionally keeps data movement simple:

```text
base files in data/
CUDA kernels in kernels/
legacy examples in examples/
research probes in probes/
current claims from s_m_benchmark.py
```

---

## 14. Current benchmark stages

The canonical benchmark has three major stages.

```text
1. substrate loading / synthetic reference construction
2. windowed feature extraction under controls
3. classification and agreement reporting
```

### 14.1 Substrate loading

The benchmark loads available substrates:

```text
GEO
GPROJ
QPROJ
```

Current example run:

```text
GEO   : d3, d5, d7, d9
GPROJ : d3, d5, d7, d9
QPROJ : d3, d5, d7, d9
```

Each distance has:

```text
shots  = 4096
rounds = 10
```

### 14.2 Feature extraction

For each substrate, distance, control mode, and window size, the benchmark computes:

```text
raw_rates
detection_rates
agreement_profiles
sm_field
sm_all
```

If CUDA is available:

```text
kernels/sm_kernel.cu
```

does the hot feature-reduction path.

If CUDA is not available, the NumPy reference path is used.

### 14.3 Task A: real-vs-control

Task A asks:

```text
Can the benchmark distinguish intact records from destructive controls?
```

Current strongest results:

```text
QPROJ sm_all             : 0.999 balanced accuracy
QPROJ sm_field           : 0.998 balanced accuracy
QPROJ agreement_profiles : 0.990 balanced accuracy
GPROJ sm_field           : 0.985 balanced accuracy
GPROJ agreement_profiles : 0.982 balanced accuracy
GPROJ sm_all             : 0.980 balanced accuracy
```

Current scalar-like baselines stay near chance:

```text
QPROJ raw_rates          : 0.535
QPROJ detection_rates    : 0.509
GPROJ raw_rates          : 0.502
GPROJ detection_rates    : 0.500
GEO raw_rates            : 0.503
GEO detection_rates      : 0.502
```

The read:

```text
raw_rates / detection_rates stay near chance
agreement_profiles / sm_field / sm_all go near-perfect
```

That is the central S_M operator signature.

### 14.4 Task B: control-source classification

Task B asks:

```text
Can the benchmark identify which destructive control produced the record?
```

Current strongest results:

```text
QPROJ sm_field : 0.853 balanced accuracy
GPROJ sm_field : 0.848 balanced accuracy
QPROJ sm_all   : 0.848 balanced accuracy
GPROJ sm_all   : 0.843 balanced accuracy
```

The read:

```text
the benchmark is not merely distinguishing real from fake
different destructive controls leave distinguishable field signatures
```

### 14.5 Task C: distance prediction

Task C asks:

```text
Can field windows predict repetition-code distance?
```

Current result:

```text
GEO   : 1.000 balanced accuracy
GPROJ : 1.000 balanced accuracy
QPROJ : 1.000 balanced accuracy
```

across multiple feature families.

The read must be careful.

Distance prediction is useful, but it can be influenced by:

```text
array shape
distance-dependent rate structure
field size
edge count
syndrome count
```

Therefore, it is not the main S_M claim by itself.

The stronger claim is the Task A split between scalar-like features and field-aware agreement features.

---

## 15. Current benchmark output

The benchmark writes:

```text
analysis/s_m_<timestamp>/
├── result.json
├── summary.csv
├── per_feature.csv
├── control_collapse.csv
├── substrate_agreement.csv
├── artifacts.npz
├── A_real_vs_control_accuracy.png
├── B_control_source_accuracy.png
└── C_distance_prediction_accuracy.png
```

Optional output with:

```bash
python s_m_benchmark.py --write-windows
```

adds:

```text
window_rows.csv
```

### 15.1 `result.json`

Full JSON record containing:

```text
config
base metadata
record shapes
best rows
all rows
substrate agreement
control collapse
bounded claim
non-claims
```

### 15.2 `summary.csv`

Best row per:

```text
substrate
task
feature
```

### 15.3 `per_feature.csv`

All model/task/feature/window rows.

### 15.4 `control_collapse.csv`

Task B control-source classification rows.

### 15.5 `substrate_agreement.csv`

Profile correlation and L2 distance across substrates.

### 15.6 `artifacts.npz`

Compact feature/profile arrays used for reproducibility and follow-up analysis.

---

## 16. Valid claim boundary

The `S_M` architecture supports the following claims.

### Supported

```text
S_M has a defined syndrome-spacetime field object.
S_M has a synthetic/reference geo path.
S_M has a GPU-generated base path.
S_M has a QPU-derived base path.
The same benchmark compares geo / gproj / qproj.
The benchmark uses destructive controls.
Field-aware features separate real records from controls.
Raw scalar-like rates remain near chance in the current real-vs-control test.
Control-source classification rises above chance.
Distance structure is visible across geo, gproj, and qproj records.
The CUDA kernel accelerates windowed feature extraction while preserving the S_M boundary.
```

### Not supported

```text
S_M is a logical-error-rate benchmark.
S_M proves quantum advantage.
S_M is faster because of QPU hardware.
S_M is the stress tensor.
S_M validates every downstream token/TSP projector.
S_M is distribution-shift-proof.
S_M proves all future Converger operators.
```

This is the claims discipline the Converger roadmap requires.

---

## 17. Why the architecture is finished for this version

`S_M` is finished for this version because every required operator-package element exists.

```text
operator definition          : complete
synthetic/reference path     : complete
GPU base generator           : complete
QPU base generator           : complete
canonical benchmark runner   : complete
CUDA feature kernel          : complete
destructive controls         : complete
substrate comparison         : complete
saved benchmark output       : complete
known limits                 : documented
```

The next work is not to keep turning `S_M` into a catch-all container.

The next work is to use `S_M` as the completed field-operator pattern while splitting downstream ideas into their own packages:

```text
T_S        stress tensor channel
I_M.local  local interaction channel
I_M.field  field interaction channel
F_M        fractal expansion channel
D_M        dimensional metric channel
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

### Why not call this logical error rate?

Because it is not the headline metric.

For logical-cat S_M runs, final majority vote can be diagnostic, but the S_M object is:

```text
final edge parity + syndrome spacetime + agreement field
```

The benchmark asks whether that field structure survives controls.

### Why not include the stress tensor here?

Because the stress tensor is its own operator.

`T_S` should derive from S_M fields, but it should have its own architecture, math, benchmark, controls, and bounded claim.

Keeping S_M clean prevents the operator from becoming a junk drawer.

### Why not trust raw syndrome rates?

Because raw rates can be misleading.

The current result shows raw scalar-like rates near chance for real-vs-control separation, while agreement and full field features approach near-perfect separation.

That is why the field structure matters.

### Why not use only synthetic/reference fields?

Because synthetic fields provide a clean reference, not a hardware substrate.

The QPU path tests real physical data.

The GPU path provides a controlled generated comparison.

The architecture needs all three.

### Why not use only QPU data?

Because one hardware substrate alone does not establish the operator boundary.

The geo and gproj paths help separate:

```text
operator structure
```

from:

```text
hardware-specific noise and drift
```

### Why not claim quantum advantage?

Because the benchmark does not show that.

The QPU path is valuable because it supplies real syndrome-spacetime records. The claim is field structure and substrate comparison, not speedup.

### Why not trust the benchmark without controls?

Because the controls are what make the claim defensible.

The destructive controls test whether field structure is load-bearing. Without them, the benchmark would only show classification performance, not why the performance exists.

---

## 19. File responsibilities

```text
README.md
    Human-facing summary and current benchmark claims.

docs/math.md
    Mathematical definition of D[i], E[i], S[t,i], A[t,i],
    detection events, feature families, controls, and benchmark metrics.

docs/architecture.md
    This document. System design for the finished S_M operator package.

docs/known_issues.md
    Current limitations and failure modes.

s_m_benchmark.py
    Canonical benchmark runner for S_M.

s_m_gpu_generate.py
    GPU-generated syndrome-spacetime base generator.

s_m_qpu_generate.py
    Unified QPU submit/dump path.

kernels/sm_kernel.cu
    CUDA feature extraction for windowed S_M fields.

data/
    S_M base files, metadata, latest-file pointers, and optional curated fixtures.

examples/
    Legacy and supporting examples retained for continuity.

probes/
    Earlier analysis probes and downstream bridge experiments.
```

---

## 20. Summary

`S_M` is the completed Syndrome Metric operator package for this version of Ghost Oracle Suite.

It demonstrates the field-operator version of the package pattern the future Converger architecture will use:

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
S_M defines a syndrome-spacetime field.
S_M uses final data edge parity and repeated syndrome records.
S_M agreement features separate real records from destructive controls.
S_M scalar-like rates remain near chance in the current key test.
S_M control-source classification rises above chance.
S_M compares synthetic/reference, GPU-generated, and QPU-derived records.
S_M uses CUDA only to accelerate feature extraction, not to change the claim.
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
