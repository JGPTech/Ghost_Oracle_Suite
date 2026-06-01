# Architecture

Architecture for the `T_S` operator package.

`T_S` is the **Temporal Stress Metric** channel. It is the third completed operator package in the Ghost Oracle Suite Converger architecture pattern.

This document follows the same operator-package discipline used by `S_M`, but applies it to the temporal stress channel. The new framing is:

```text
T_S is a finished operator package for this version.
T_S is one channel in the larger Converger roadmap.
T_S uses the standard geo / gproj / qproj substrate pattern.
T_S claims are made only through the benchmark runner, probes, and controls.
T_S is not a QPU speedup claim.
T_S is not a universal shortest-path claim.
```

The math is in:

```text
docs/math.md
```

Known limitations are in:

```text
docs/known_issues.md
```

The probes preserve the chronological discovery path, but the current architecture claim comes from:

```text
t_s_benchmark.py
```

This architecture document explains how the finished `T_S` operator is wired together, what each file is responsible for, what counts as a valid benchmark claim, and how the package fits into the larger Converger direction without inflating the current result beyond what the benchmark supports.

---

## 1. Architectural status

`T_S` is complete for this version.

That means the package now has:

```text
1. a real QPU temporal-stress generation/dump path,
2. a GPU-generated temporal-stress base path,
3. a raw geo arithmetic route path,
4. a canonical final benchmark runner,
5. temporal stress tensor extraction,
6. raw-first geo route evaluation,
7. edge / round / round-edge scaffold ablations,
8. destructive coarse controls,
9. qproj / gproj substrate comparison,
10. CUDA-accelerated raw route evaluation,
11. classical route baselines,
12. generic scalar/profile baselines,
13. documented math,
14. documented claim boundary.
```

The current package is not a placeholder for future work. It is a finished operator implementation with a bounded claim.

The bounded claim is:

```text
T_S is a field-structured, stress-derived, route-tested temporal metric.
```

The claim is not:

```text
T_S proves quantum advantage.
T_S proves QPU speedup.
T_S is a universal shortest-path algorithm.
T_S is a QPU simulator.
T_S proves every downstream projector.
T_S replaces S_M, G_M, or future Converger channels.
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

Within that system, `T_S` is the temporal stress component.

```text
Converger operator stack
├── G_M        Generalized Metric channel
├── S_M        Syndrome Metric channel
├── T_S        Temporal Stress Metric channel
├── I_M.local  Local Interaction channel
├── I_M.field  Field Interaction channel
├── F_M        Fractal Expansion channel
└── D_M        Dimensional Metric channel
```

`T_S` is not the entire Converger.

`T_S` is the stress/route/scaffold operator in that stack.

Its role is:

```text
temporal edge-field structure
delay / round / edge gradient extraction
stress tensor construction
raw geo route evaluation
edge / round scaffold damage measurement
qproj / gproj substrate comparison
classical route baseline comparison
```

Transformer-adjacent analogues include:

```text
activation stress fields
attention-route damage maps
local perturbation survival paths
representation-scaffold ablation
temporal memory / channel-stress diagnostics
```

The `T_S` package does not replace those transformer components. It defines and benchmarks a temporal stress operator that can later inform larger Converger components.

---

## 3. Standard operator package pattern

The Converger architecture uses the same package shape for every operator:

```text
operator/
├── operator_gpu_generate.py
├── operator_qpu_generate.py
└── operator_benchmark.py
```

For `T_S`, the current package uses:

```text
T_S/
├── t_s_gpu_generate.py
├── t_s_qpu_generate.py
└── t_s_benchmark.py
```

Supporting directories:

```text
T_S/
├── data/
├── docs/
├── kernels/
├── probes/
└── analysis/
```

The current main entry point is:

```bash
python t_s_benchmark.py
```

Base generation:

```bash
python t_s_gpu_generate.py --verify
python t_s_qpu_generate.py submit
python t_s_qpu_generate.py dump <JOB_ID>
```

Probe paths remain in the repo:

```bash
python probes/t_s_probe1.py
python probes/t_s_probe2_geo.py
python probes/t_s_probe3_raw_first_jumpgeo.py
python probes/t_s_probe5_qpu_geo_origin.py
python probes/t_s_probe6_qpu_projection_raw_damage.py
python probes/t_s_probe7_qpu_projection_raw_damage_cuda.py
```

Those probes are retained for continuity and process history. They document how the operator was discovered, corrected, and finalized.

Current benchmark claims should come from:

```text
t_s_benchmark.py
```

and from saved benchmark output under:

```text
analysis/ts_benchmark_<timestamp>/
```

---

## 4. Standard substrate paths

`T_S` follows the standard three-path pattern used by the Converger roadmap.

```text
T_S^{geo}
T_S^{gproj}
T_S^{qproj}
```

In GitHub-safe plain text:

```text
T_S_geo    = raw arithmetic route path derived from a T_S field
T_S_gproj  = GPU-generated temporal-stress base
T_S_qproj  = real QPU temporal-stress base
```

### 4.1 Geometry / raw route path

The `geo` path is the raw arithmetic route operator.

It takes a T_S field, derives the stress tensor, computes raw movement costs, and evaluates a structured monotonic route through the stress grid.

The current CUDA implementation is:

```text
kernels/ts_geo_kernel.cu
```

Main entry point:

```text
ts_raw_geo_route_vector_kernel
```

Compatibility entry point:

```text
ts_raw_geo_monotonic_kernel
```

The `geo` path is not a substrate by itself in the same way qproj/gproj are. It is the operator path applied to any valid substrate.

Its purpose is:

```text
derive route/stress structure from the field
evaluate scaffold damage
provide a route baseline against generic graph methods
```

### 4.2 GPU-generated path

The `gproj` path uses a GPU-generated T_S base:

```text
data/ts_gpu_data_<TAG>.npz
```

It uses the same analysis schema as QPU dumps.

The GPU base is not an arbitrary baseline. It is a controlled temporal-stress field with the same downstream schema as the QPU dump.

Its purpose is to separate:

```text
operator structure
```

from:

```text
hardware noise, backend drift, calibration state, queue timing, and physical execution effects
```

The GPU generator is not a QPU simulator claim.

It is a controlled generated comparison substrate.

### 4.3 QPU path

The `qproj` path uses real IBM Runtime temporal-stress records:

```text
data/ts_data_<JOB_ID>.npz
```

The QPU path is the hardware-derived temporal field substrate.

It is not used to claim raw throughput superiority. It is used to test whether real hardware produces a temporal edge/round field whose stress-derived route scaffold survives destructive controls.

---

## 5. Bases

A **base** is a `.npz` file containing T_S field records.

The shared base schema is:

```text
schema        : str
job_id        : str
backend       : str, optional
source        : str, optional
shots         : int
rounds        : int
channels      : int
edges         : int
modes         : str array
delay_sites   : str array
delays        : int array
delay_unit    : str

field         : uint8, shape (modes, delay_sites, delays, shots, rounds, edges)
final         : uint8, shape (modes, delay_sites, delays, shots, channels)
```

The same analysis schema is used for:

```text
real QPU T_S bases
GPU-generated T_S bases
```

This shared schema is load-bearing.

It allows the same probes and benchmark code to consume QPU and GPU bases without changing the operator.

Generated QPU base files use the pattern:

```text
data/ts_data_<JOB_ID>.npz
data/ts_job_<JOB_ID>.json
```

Generated GPU base files use the pattern:

```text
data/ts_gpu_data_<TAG>.npz
data/ts_gpu_job_<TAG>.json
```

Latest-file pointers:

```text
data/latest_ts_data.json
data/latest_ts_gpu_data.json
```

Benchmark output is written under:

```text
analysis/ts_benchmark_<timestamp>/
```

---

## 6. Temporal edge field

`T_S` is built from a temporal edge-channel field.

The primary object is:

```text
Phi[m, s, delta, shot, round, edge]
```

where:

```text
m      = mode index
s      = delay-site index
delta  = delay-value index
shot   = shot index
round  = repeated interaction / temporal round
edge   = adjacent channel edge
```

The current default labels are:

```text
modes:
    clean
    phase_shear
    local_shock

delay_sites:
    pre_coupling
    post_coupling
    post_perturb

delays:
    0, 1, 2, 4, 8, 16 dt
```

The operator is not just:

```text
mean field rate
```

The field relation matters.

The important axes are:

```text
delay axis
round axis
edge/channel axis
```

The benchmark asks whether that stress-derived structure survives controls.

---

## 7. Temporal stress tensor

For a fixed mode/site block:

```text
Phi[delay, shot, round, edge]
```

the operator computes binary finite differences.

Delay gradient:

```text
d_tau = Phi[delay+1, shot, round, edge] XOR Phi[delay, shot, round, edge]
```

Round gradient:

```text
d_round = Phi[delay, shot, round+1, edge] XOR Phi[delay, shot, round, edge]
```

Edge gradient:

```text
d_edge = Phi[delay, shot, round, edge+1] XOR Phi[delay, shot, round, edge]
```

The gradients are aligned onto the common lattice:

```text
delay_cell × shot × round_cell × edge_cell
```

The stress components are shot-averaged products:

```text
tau_tau = mean(d_tau   * d_tau)
rr      = mean(d_round * d_round)
xx      = mean(d_edge  * d_edge)

tau_r   = mean(d_tau   * d_round)
tau_x   = mean(d_tau   * d_edge)
r_x     = mean(d_round * d_edge)
```

The trace is:

```text
trace = tau_tau + rr + xx
```

This stress tensor is the central field object.

---

## 8. Raw geo route

T_S converts stress into raw movement costs:

```text
cost_tau   = tau_tau + coupling_weight * (tau_r + tau_x)
cost_round = rr      + coupling_weight * (tau_r + r_x)
cost_edge  = xx      + coupling_weight * (tau_x + r_x)
```

Default:

```text
coupling_weight = 0.50
```

The route is a monotonic path through the stress grid:

```text
source = (0, 0, 0)
target = (delay_cell-1, round_cell-1, edge_cell-1)
```

The dynamic-programming recurrence is:

```text
dp[a,r,e] = min(
    dp[a-1,r,e] + cost_tau[a,r,e],
    dp[a,r-1,e] + cost_round[a,r,e],
    dp[a,r,e-1] + cost_edge[a,r,e]
)
```

with:

```text
dp[0,0,0] = 0
```

The route vector is:

```text
route[0] = full_cost
route[1] = delay_cost
route[2] = edge_cost
route[3] = delay_to_full
route[4] = edge_to_full
route[5] = trace_mean
route[6] = stress_avoidance_proxy
route[7] = path_trace_mean_proxy
```

The final benchmark uses raw routes.

No normalization is used.

No cosine similarity is used.

No classifier framing is used.

---

## 9. Raw-damage projection signature

The T_S projection signature is a raw-damage scaffold signature.

For each intact block and ablated block, compute:

```text
damage =
    abs(full_cost_ablated  - full_cost_real)
  + abs(delay_cost_ablated - delay_cost_real)
  + abs(edge_cost_ablated  - edge_cost_real)
  + max(0, real_avoidance - ablated_avoidance)
```

The benchmark records:

```text
edge_damage[mode, site, edge]
round_damage[mode, site, round]
round_edge_damage[mode, site, round, edge]
```

This is the qproj/gproj projection signature.

The core read is:

```text
which edge, round, and round-edge components are load-bearing for the raw geo route?
```

---

## 10. Field controls

The benchmark uses destructive controls to test whether T_S field structure is load-bearing.

Control modes:

```text
edge_shuffle
round_shuffle
round_reverse
edge_reverse
delay_shuffle
delay_reverse
uniform_by_cell
all_uniform
```

### 10.1 `edge_shuffle`

Destroys edge ordering.

Preserves:

```text
per-delay / per-shot / per-round field values
```

Destroys:

```text
channel adjacency structure
```

### 10.2 `round_shuffle`

Destroys temporal round ordering.

Preserves:

```text
per-delay / per-shot / per-edge field values
```

Destroys:

```text
repeated-interaction order
```

### 10.3 `round_reverse`

Reverses the round axis.

Preserves:

```text
round content
```

Changes:

```text
forward temporal orientation
```

### 10.4 `edge_reverse`

Reverses the edge axis.

Preserves:

```text
edge content
```

Changes:

```text
channel orientation
```

### 10.5 `delay_shuffle`

Shuffles delay ordering.

Preserves:

```text
delay content
```

Destroys:

```text
delay order
```

### 10.6 `delay_reverse`

Reverses the delay axis.

Preserves:

```text
delay content
```

Changes:

```text
delay orientation
```

### 10.7 `uniform_by_cell`

Replaces field samples with Bernoulli draws matching each delay/round/edge cell rate.

Preserves:

```text
cellwise field intensity
```

Destroys:

```text
shot-level field structure
```

### 10.8 `all_uniform`

Replaces the whole block with Bernoulli draws matching the global block rate.

Preserves:

```text
broad scalar density
```

Destroys:

```text
delay structure
round structure
edge structure
shot-level structure
```

The key read is:

```text
edge and round scaffold damage is large
delay-order damage is small
```

for the current qproj/gproj results.

---

## 11. Current CUDA architecture

The core CUDA source is:

```text
kernels/ts_geo_kernel.cu
```

The benchmark and optimized probes use this kernel when CuPy and CUDA are available.

Primary kernel:

```text
ts_raw_geo_route_vector_kernel
```

Compatibility kernel:

```text
ts_raw_geo_monotonic_kernel
```

Supporting device helpers compute:

```text
raw movement costs
structured monotonic DP route
route vector
raw damage
damage aggregation
```

### 11.1 `ts_raw_geo_route_vector_kernel`

Role:

```text
compute raw geo route vectors from stress tensor components
```

Inputs:

```text
tau_tau : float32, shape (B, N)
rr      : float32, shape (B, N)
xx      : float32, shape (B, N)
tau_r   : float32, shape (B, N)
tau_x   : float32, shape (B, N)
r_x     : float32, shape (B, N)
```

where:

```text
N = delay_cell × round_cell × edge_cell
```

Outputs:

```text
routes : float32, shape (B, 8)
```

Route vector layout:

```text
full_cost
delay_cost
edge_cost
delay_to_full
edge_to_full
trace_mean
stress_avoidance_proxy
path_trace_mean_proxy
```

### 11.2 Compatibility kernel

The compatibility kernel is:

```text
ts_raw_geo_monotonic_kernel
```

It is retained so earlier probes and callers do not break.

It reports:

```text
full route cost
delay route cost
edge route cost
```

### 11.3 CUDA boundary

The CUDA boundary is intentionally narrow.

Included:

```text
stress tensor route evaluation
raw movement costs
monotonic structured-grid DP
raw route vector output
raw damage support
```

Excluded:

```text
QPU job submission
GPU base generation
ablation construction
stress tensor construction from raw bits
normalization
classification
qproj/gproj interpretation
```

The CUDA path does not change the operator.

It accelerates route evaluation.

The current benchmark shows route evaluation is no longer the dominant cost.

---

## 12. GPU-generated base architecture

The GPU generator is:

```text
t_s_gpu_generate.py
```

It writes the same analysis-facing schema as QPU dumps:

```text
field[mode, delay_site, delay, shot, round, edge]
final[mode, delay_site, delay, shot, channel]
```

Default shape:

```text
modes       = 3
delay_sites = 3
delays      = 6
shots       = 4096
rounds      = 6
channels    = 8
edges       = 7
```

Generated files:

```text
data/ts_gpu_data_<TAG>.npz
data/ts_gpu_job_<TAG>.json
data/latest_ts_gpu_data.json
data/latest_ts_data.json
```

The generator is CuPy-first, with optional CPU fallback.

The current tuned defaults are:

```text
DEFAULT_P_BASE = 0.165
DEFAULT_P_READOUT = 0.018
DEFAULT_EDGE_MEMORY = 0.62
DEFAULT_ROUND_MEMORY = 0.70
DEFAULT_DELAY_GAIN = 0.020
DEFAULT_SITE_GAIN = 0.010
DEFAULT_PHASE_SHEAR = 0.045
DEFAULT_LOCAL_SHOCK = 0.115
DEFAULT_EDGE_SCAFFOLD = 0.095
DEFAULT_ROUND_SCAFFOLD = 0.060
DEFAULT_BURST_PROB = 0.000
DEFAULT_BURST_WIDTH = 2
```

These defaults preserve the main round/round-edge scaffold while keeping the generated base probe-compatible.

The generator is not a hardware simulator claim.

Its purpose is:

```text
create controlled local T_S bases
exercise the same probes and benchmark
compare generated scaffold behavior to QPU scaffold behavior
```

---

## 13. QPU generation architecture

The QPU generator is:

```text
t_s_qpu_generate.py
```

It creates temporal-stress circuits with structured modes, delay sites, and delay values.

Current conceptual generation axes:

```text
mode:
    clean
    phase_shear
    local_shock

delay_site:
    pre_coupling
    post_coupling
    post_perturb

delay:
    0, 1, 2, 4, 8, 16 dt
```

The dumped QPU file writes:

```text
field
final
modes
delay_sites
delays
rounds
channels
edges
job_id
```

The QPU generator is responsible for producing the hardware substrate.

The benchmark is responsible for making claims from that substrate.

The QPU path is not a speed path.

It is a field-substrate path.

---

## 14. Canonical data flow

End-to-end `T_S` package flow:

```text
QPU hardware path
    t_s_qpu_generate.py submit
        -> IBM Runtime job
        -> metadata JSON

    t_s_qpu_generate.py dump <JOB_ID>
        -> completed job result
        -> data/ts_data_<JOB_ID>.npz

GPU/generated path
    t_s_gpu_generate.py
        -> generated temporal-stress base
        -> data/ts_gpu_data_<TAG>.npz

Shared probe path
    probes/t_s_probe*.py
        -> inspect field shape
        -> derive stress
        -> test raw-first route policy
        -> identify QPU geo origin
        -> finalize qproj raw-damage signature
        -> optimize route evaluation with CUDA

Shared benchmark path
    t_s_benchmark.py
        -> load all valid T_S .npz files in data/
        -> compute stress tensors
        -> construct destructive controls
        -> evaluate geo_cuda routes
        -> compare qproj/gproj scaffold signatures
        -> compare classical baselines
        -> write JSON / CSV / plots
```

Inside the benchmark:

```text
field
    -> local stress tensor
    -> raw geo routes
    -> edge / round / round-edge ablations
    -> qproj/gproj scaffold signatures
    -> geo/classical method comparison
    -> saved outputs
```

The current architecture intentionally keeps data movement simple:

```text
base files in data/
CUDA kernels in kernels/
research probes in probes/
current claims from t_s_benchmark.py
```

---

## 15. Current benchmark stages

The canonical benchmark has three major stages.

```text
1. substrate loading
2. stress/route/ablation evaluation
3. substrate and method comparison
```

### 15.1 Substrate loading

The benchmark scans every valid `.npz` in:

```text
T_S/data/
```

A valid file contains:

```text
field
```

with shape:

```text
modes × delay_sites × delays × shots × rounds × edges
```

Files are classified as:

```text
qproj
gproj
data_<stem>
```

Specific files can be supplied with:

```bash
python t_s_benchmark.py --files file1.npz file2.npz
```

### 15.2 Stress and route evaluation

For each valid source, mode, and delay site, the benchmark computes:

```text
local stress tensor
real route vector
edge ablation routes
round ablation routes
round-edge ablation routes
coarse control routes
```

Route evaluation uses:

```text
geo_cuda
```

when CUDA is available.

CPU fallback is available with:

```bash
python t_s_benchmark.py --cpu-only
```

### 15.3 Substrate comparison

The benchmark compares qproj and gproj signatures.

Metrics include:

```text
top1_match
Spearman rank correlation
top3_overlap
top5_overlap
```

Components:

```text
edge
round
round_edge
coarse
```

Current benchmark read:

```text
round top1 match       = 1.000
round_edge top1 match  = 1.000
coarse Spearman        ≈ 0.976
edge top5 overlap      = 1.000
```

The edge top1 mismatch is informative:

```text
QPU has stronger edge-5 localization than the current GPU generator.
```

### 15.4 Method comparison

The benchmark compares T_S geo against classical and generic baselines.

Methods:

```text
geo_cuda
geo_cpu_dp
scipy_dijkstra
scalar_rate
field_profile_l1
stress_profile_l1
```

Optional:

```text
networkx_dijkstra
```

Current benchmark read:

```text
geo_cuda matches CPU DP and SciPy Dijkstra on the route task
geo_cuda is faster than SciPy Dijkstra on the current structured-grid workload
generic scalar/profile baselines do not recover the same round-edge scaffold
```

---

## 16. Current benchmark output

The benchmark writes:

```text
analysis/ts_benchmark_<timestamp>/
├── ts_benchmark_report.json
├── ts_benchmark_summary.csv
├── ts_benchmark_rows.csv
├── qpu_gpu_alignment.csv
├── baseline_comparison.csv
├── benchmark_speed.csv
├── benchmark_speed_blocks.csv
├── edge_damage_rows.csv
├── round_damage_rows.csv
├── round_edge_damage_rows.csv
├── coarse_damage_rows.csv
├── edge_damage_compare.png
├── round_damage_compare.png
├── round_edge_damage_compare.png
└── coarse_damage_compare.png
```

### 16.1 `ts_benchmark_report.json`

Full JSON record containing:

```text
settings
inputs
gpu info
alignment
speed summary
source summaries
edge aggregates
round aggregates
round-edge aggregates
coarse aggregates
baseline summary
```

### 16.2 `ts_benchmark_summary.csv`

Per-source, per-mode, per-site summary rows.

### 16.3 `ts_benchmark_rows.csv`

Flattened aggregate rows for edge, round, round-edge, and coarse components.

### 16.4 `qpu_gpu_alignment.csv`

Pairwise qproj/gproj scaffold alignment.

Printed as:

```text
QPROJ/GPROJ ALIGNMENT
```

### 16.5 `baseline_comparison.csv`

Method comparison rows for:

```text
geo_cuda
geo_cpu_dp
scipy_dijkstra
networkx_dijkstra, optional
scalar_rate
field_profile_l1
stress_profile_l1
```

### 16.6 `benchmark_speed.csv`

Per-source benchmark timing summary.

### 16.7 Row-level damage CSVs

Detailed rows:

```text
edge_damage_rows.csv
round_damage_rows.csv
round_edge_damage_rows.csv
coarse_damage_rows.csv
```

---

## 17. Current benchmark result

Current representative final benchmark output:

```text
QPROJ top edge:
    edge 5

QPROJ top round:
    round 2

QPROJ top round-edge:
    round 2, edge 3

GPROJ top round:
    round 2

GPROJ top round-edge:
    round 2, edge 3
```

QPROJ/GPROJ alignment:

```text
edge:
    top1 = 0.000
    Spearman ≈ 0.357
    top5 = 1.000

round:
    top1 = 1.000
    Spearman ≈ 0.600

round_edge:
    top1 = 1.000
    top5 = 0.600

coarse:
    Spearman ≈ 0.976
```

Method baselines:

```text
geo_cuda:
    round_edge_top1_mean = 1.000
    round_edge_spearman_mean = 1.0000
    faster than scipy_dijkstra on current workload

geo_cpu_dp:
    round_edge_top1_mean = 1.000
    round_edge_spearman_mean = 1.0000

scipy_dijkstra:
    round_edge_top1_mean = 1.000
    round_edge_spearman_mean = 1.0000

stress_profile_l1:
    partial scaffold signal

field_profile_l1:
    weak scaffold signal

scalar_rate:
    weak scaffold signal
```

The read:

```text
qproj and gproj agree on the main round and round-edge scaffold.
qproj and gproj strongly agree on coarse destructive-control ranking.
geo_cuda matches CPU DP and SciPy Dijkstra on the route task.
geo_cuda is the fastest route method in the current structured-grid benchmark.
generic scalar/profile methods do not recover the same scaffold.
```

---

## 18. Valid claim boundary

The `T_S` architecture supports the following claims.

### Supported

```text
T_S has a defined temporal edge-field object.
T_S has a QPU-derived base path.
T_S has a GPU-generated base path.
T_S has a raw geo route path.
The same benchmark compares qproj and gproj.
The benchmark uses destructive controls.
Stress tensor components are derived from delay/round/edge gradients.
Raw geo routes identify load-bearing edge/round scaffold components.
QPROJ and GPROJ agree on main round and round-edge structure in the current run.
QPROJ and GPROJ agree strongly on coarse destructive-control ranking.
GEO CUDA matches CPU DP and SciPy Dijkstra on the current route task.
GEO CUDA is faster than SciPy Dijkstra on the current structured-grid workload.
Generic scalar/profile baselines do not recover the same round-edge scaffold.
```

### Not supported

```text
T_S proves quantum advantage.
T_S proves QPU speedup.
T_S is a universal shortest-path algorithm.
T_S GPU generation simulates the QPU.
T_S fully matches qproj edge localization with gproj.
T_S is a logical-error-rate benchmark.
T_S validates every downstream token/TSP/projector path.
T_S proves all future Converger operators.
```

This is the claims discipline the Converger roadmap requires.

---

## 19. Why the architecture is finished for this version

`T_S` is finished for this version because every required operator-package element exists.

```text
operator definition        : complete
QPU base generator         : complete
GPU base generator         : complete
raw geo route path         : complete
canonical benchmark runner : complete
CUDA geo kernel            : complete
destructive controls       : complete
qproj/gproj comparison     : complete
classical baselines        : complete
saved benchmark output     : complete
math documentation         : complete
known limits               : ready to document / update
```

The next work is not to keep expanding `T_S` into every future interaction channel.

The next work is to use `T_S` as a completed field/stress/route operator pattern while splitting downstream ideas into their own packages:

```text
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

## 20. Why not...

### Why not call this quantum advantage?

Because the benchmark does not show that.

The QPU path is valuable because it supplies a real temporal edge field. The claim is field structure and substrate comparison, not speedup.

### Why not call this QPU speedup?

Because the route speed comes from CUDA geo evaluation, not QPU hardware.

The QPU path is the hardware field substrate.

The CUDA path is the route evaluator.

Those are different claims.

### Why not call the GPU generator a QPU simulator?

Because it is not one.

It is a controlled generated T_S substrate with the same analysis schema. It helps compare operator behavior without requiring a new QPU job.

### Why not use only scalar rates?

Because scalar rates miss the route scaffold.

The final benchmark shows scalar_rate and simple profile baselines do not recover the same round-edge scaffold as raw geo route damage.

### Why not use only SciPy Dijkstra?

Because SciPy Dijkstra is generic.

It is a good correctness baseline, but T_S has a structured monotonic stress grid. The geo path exploits that structure directly.

### Why not treat edge mismatch as failure?

Because the current GPU generator captures the main round and round-edge scaffold, while QPU edge localization remains sharper.

The edge mismatch identifies a generator-tuning target:

```text
stronger qproj edge-5 localization
```

not a failure of the operator definition.

---

## 21. File responsibilities

```text
README.md
    Human-facing summary and current benchmark claims.

docs/math.md
    Mathematical definition of the temporal edge field, stress tensor,
    raw geo route, damage metric, controls, substrate paths, and benchmark.

docs/architecture.md
    This document. System design for the finished T_S operator package.

docs/known_issues.md
    Current limitations and failure modes.

t_s_benchmark.py
    Canonical final benchmark runner for T_S.

t_s_gpu_generate.py
    GPU-generated temporal-stress base generator.

t_s_qpu_generate.py
    QPU submit/dump path for temporal-stress bases.

kernels/ts_geo_kernel.cu
    CUDA raw geo route evaluation and raw-damage support.

data/
    T_S base files, metadata, latest-file pointers, and optional curated fixtures.

analysis/
    Saved benchmark and probe outputs.

probes/
    Discovery, validation, and optimization probes.
```

Probe responsibilities:

```text
probes/t_s_probe1.py
    Field shape, stress tensor, and temporal summary.

probes/t_s_probe2_geo.py
    Geo reconstruction and transform sweep.

probes/t_s_probe3_raw_first_jumpgeo.py
    Raw-first JumpGeo policy.

probes/t_s_probe5_qpu_geo_origin.py
    QPU geo-origin / structure ablation.

probes/t_s_probe6_qpu_projection_raw_damage.py
    Raw-damage qproj signature finalizer.

probes/t_s_probe7_qpu_projection_raw_damage_cuda.py
    Optimized CUDA raw-damage signature finalizer.
```

---

## 22. Summary

`T_S` is the completed Temporal Stress Metric operator package for this version of Ghost Oracle Suite.

It demonstrates the stress/route version of the package pattern the future Converger architecture will use:

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
T_S defines a temporal edge field.
T_S derives a stress tensor from delay, round, and edge gradients.
T_S evaluates raw geo routes through that stress tensor.
T_S uses edge/round/round-edge ablations to identify load-bearing scaffold structure.
T_S compares QPU-derived and GPU-generated records under the same schema.
T_S compares geo_cuda against CPU DP, SciPy Dijkstra, and generic scalar/profile baselines.
T_S uses CUDA only to accelerate route evaluation, not to change the claim.
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
