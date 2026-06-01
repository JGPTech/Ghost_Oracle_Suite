# T_S — Temporal Stress Metric

`T_S` is the Ghost Oracle Suite temporal-stress operator family: **Temporal Stress Metric**.

`T_S` treats repeated delayed channel measurements not as a scalar error statistic, but as a temporal edge field. The useful object is the relationship between delay structure, repeated interaction rounds, channel-edge structure, and the raw geo route that survives destructive controls.

Current framing:

```text
T_S = temporal-stress field operator
```

The core field is:

```text
T_S = {Phi[delay, shot, round, edge], T_munu, raw_geo_route, raw_damage}
```

where:

```text
Phi[delay, shot, round, edge]
    = temporal edge-channel field

d_tau
    = delay gradient of Phi

d_round
    = round / repeated-interaction gradient of Phi

d_edge
    = channel-edge gradient of Phi

T_munu
    = stress tensor components derived from d_tau, d_round, d_edge

raw_geo_route
    = monotonic route through the stress grid

raw_damage
    = route damage under edge / round / round-edge ablation
```

`T_S` is implemented across three paths:

1. `geo` — raw arithmetic route path derived from a T_S field,
2. `gproj` — GPU-generated temporal-stress base,
3. `qproj` — real QPU temporal-stress data from IBM Runtime.

The core claim is not that `T_S` proves quantum advantage.

The core claim is that QPU-derived and GPU-generated temporal fields can be represented under one schema, converted into stress tensors, routed with the raw geo operator, destructively scrambled, and compared against classical route/profile baselines under one benchmark harness.

---

## Quick path

The current canonical benchmark path is:

```bash
python t_s_benchmark.py
```

By default, the benchmark scans every valid `.npz` in:

```text
data/
```

A valid T_S file contains:

```text
field
```

with shape:

```text
modes × delay_sites × delays × shots × rounds × edges
```

Useful variants:

```bash
python t_s_benchmark.py --files data/ts_data_<JOB_ID>.npz data/ts_gpu_data_<TAG>.npz
python t_s_benchmark.py --qpu-only
python t_s_benchmark.py --gpu-only
python t_s_benchmark.py --cpu-only
python t_s_benchmark.py --include-networkx
```

PowerShell example:

```powershell
python t_s_benchmark.py `
  --files `
  data/ts_data_d8e9ab3o3njc73eue47g.npz `
  data/ts_gpu_data_4096shots_seed4204302182560473160.npz
```

Generate a GPU base:

```bash
python t_s_gpu_generate.py --verify
```

Submit / dump a QPU base:

```bash
python t_s_qpu_generate.py submit
python t_s_qpu_generate.py dump <JOB_ID>
```

The main scripts are:

| Script | Status | Purpose |
| ------ | ------ | ------- |
| `t_s_benchmark.py` | current canonical path | Final T_S benchmark: geo/qproj/gproj scaffold recovery, classical baselines, and timing. |
| `t_s_gpu_generate.py` | current base generator | Generates GPU T_S bases with the same analysis schema as QPU dumps. |
| `t_s_qpu_generate.py` | current QPU path | Submit and dump IBM Runtime T_S jobs using one CLI. |
| `kernels/ts_geo_kernel.cu` | current CUDA route path | Raw geo route evaluation and raw-damage support. |
| `probes/t_s_probe1.py` | probe path | Field shape, stress tensor, and temporal summary. |
| `probes/t_s_probe2_geo.py` | probe path | Geo reconstruction and transform sweep. |
| `probes/t_s_probe3_raw_first_jumpgeo.py` | probe path | Raw-first JumpGeo policy. |
| `probes/t_s_probe5_qpu_geo_origin.py` | probe path | QPU geo-origin / structure ablation. |
| `probes/t_s_probe6_qpu_projection_raw_damage.py` | probe path | Raw-damage qproj signature finalizer. |
| `probes/t_s_probe7_qpu_projection_raw_damage_cuda.py` | probe path | CUDA-optimized raw-damage signature finalizer. |

---

## Operator

The core T_S objects are:

```text
Phi[delay, shot, round, edge]
    temporal edge-channel field

d_tau
    Phi[delay+1] XOR Phi[delay]

d_round
    Phi[round+1] XOR Phi[round]

d_edge
    Phi[edge+1] XOR Phi[edge]
```

From these gradients, T_S derives stress components:

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

The raw geo route then moves through the stress grid using:

```text
cost_tau   = tau_tau + coupling_weight * (tau_r + tau_x)
cost_round = rr      + coupling_weight * (tau_r + r_x)
cost_edge  = xx      + coupling_weight * (tau_x + r_x)
```

Default:

```text
coupling_weight = 0.50
```

The benchmark treats `T_S` as a raw field/stress/route operator.

Important structural notes:

```text
T_S is a field operator.
T_S is stress-derived.
T_S is route-tested.
T_S uses raw damage, not normalized cosine similarity.
T_S is not a QPU speedup claim.
T_S is not a universal graph algorithm.
T_S is not a QPU simulator.
```

---

## Repository structure

```text
GHOST_ORACLE_SUITE/
└── ghost_oracle/
    └── T_S/
        ├── README.md
        ├── t_s_benchmark.py
        ├── t_s_gpu_generate.py
        ├── t_s_qpu_generate.py
        │
        ├── data/
        │   ├── ts_data_<JOB_ID>.npz
        │   ├── ts_job_<JOB_ID>.json
        │   ├── ts_gpu_data_<TAG>.npz
        │   ├── ts_gpu_job_<TAG>.json
        │   ├── latest_ts_data.json
        │   └── latest_ts_gpu_data.json
        │
        ├── docs/
        │   ├── architecture.md
        │   ├── math.md
        │   └── known_issues.md
        │
        ├── kernels/
        │   └── ts_geo_kernel.cu
        │
        ├── probes/
        │   ├── t_s_probe1.py
        │   ├── t_s_probe2_geo.py
        │   ├── t_s_probe3_raw_first_jumpgeo.py
        │   ├── t_s_probe5_qpu_geo_origin.py
        │   ├── t_s_probe6_qpu_projection_raw_damage.py
        │   └── t_s_probe7_qpu_projection_raw_damage_cuda.py
        │
        └── analysis/
            └── ts_benchmark_<timestamp>/
```

## Directory map

| Path | Role |
| ---- | ---- |
| `README.md` | Main T_S documentation and current benchmark summary. |
| `t_s_benchmark.py` | Current canonical T_S benchmark runner for geo/qproj/gproj comparison and baselines. |
| `t_s_gpu_generate.py` | Generates GPU T_S bases with QPU-compatible analysis schema. |
| `t_s_qpu_generate.py` | Unified QPU submit/dump CLI for T_S jobs. |
| `data/` | T_S base records, metadata, latest-file pointers, and optional curated fixtures. |
| `docs/` | Architecture notes, math notes, known issues, and future direction documents. |
| `kernels/` | CUDA source for optimized T_S raw geo route evaluation. |
| `probes/` | Discovery, validation, and optimization probes. |
| `analysis/` | Saved probe and benchmark output. |

---

## Data files

The `data/` folder contains T_S field bases and metadata.

```text
ts_data_<JOB_ID>.npz
```

Real QPU T_S field base dumped from IBM Runtime.

```text
ts_gpu_data_<TAG>.npz
```

GPU-generated T_S field base.

```text
ts_job_<JOB_ID>.json
```

QPU job metadata.

```text
ts_gpu_job_<TAG>.json
```

GPU base metadata.

```text
latest_ts_data.json
latest_ts_gpu_data.json
```

Convenience pointers used by generators, probes, and the benchmark.

Expected T_S `.npz` arrays:

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

Recommended policy:

```text
Keep small curated fixtures if they are part of the reproducibility story.
Keep large generated bases out of git unless intentionally shipping them.
```

Recommended `.gitignore` patterns:

```gitignore
data/ts_data_*.npz
data/ts_gpu_data_*.npz
data/ts_job_*.json
data/ts_gpu_job_*.json
analysis/ts_benchmark_*/
probes/analysis/
*_report.json
```

Keep `latest_*.json` only if they are useful for local workflow. Avoid relying on them for published reproducibility unless the pointed files are also included.

---

## Probe path

The `probes/` directory records the T_S discovery and validation path.

It is not the final benchmark claim by itself, but it explains why the current benchmark is shaped the way it is.

Probe sequence:

| Probe | Purpose |
| ----- | ------- |
| `t_s_probe1.py` | Load a T_S base, summarize field shape, rates, and stress tensor components. |
| `t_s_probe2_geo.py` | Test geo reconstruction and transform options. |
| `t_s_probe3_raw_first_jumpgeo.py` | Establish the raw-first route policy and jump behavior. |
| `t_s_probe5_qpu_geo_origin.py` | Identify the QPU geo-origin scaffold through structure ablations. |
| `t_s_probe6_qpu_projection_raw_damage.py` | Finalize raw-damage qproj projection signature. |
| `t_s_probe7_qpu_projection_raw_damage_cuda.py` | Optimize Probe 06 with the CUDA raw route kernel. |

The current boundary is:

```text
T_S      = temporal stress / raw geo route operator
S_M      = syndrome-spacetime field operator, separate
G_M      = generalized metric channel, separate
I_M      = interaction / field deformation, separate
token    = downstream projector probe, not T_S itself
TSP      = downstream field-deformation example, not T_S itself
```

---

## Current capstone benchmark

Run:

```bash
python t_s_benchmark.py
```

The benchmark scans all valid `.npz` files in:

```text
data/
```

or accepts an explicit set:

```bash
python t_s_benchmark.py --files data/file1.npz data/file2.npz
```

The current capstone compares:

| Path | Role |
| ---- | ---- |
| `geo` | Raw arithmetic route path derived from any valid T_S field. |
| `gproj` | GPU-generated T_S field base. |
| `qproj` | Real QPU T_S field base. |

It runs one shared task:

```text
route-preserving edge/round scaffold recovery
```

The benchmark identifies which components are load-bearing for the raw route:

```text
edge_damage
round_damage
round_edge_damage
coarse_control_damage
```

It also compares T_S geo against adjacent classical baselines:

```text
geo_cuda
geo_cpu_dp
scipy_dijkstra
scalar_rate
field_profile_l1
stress_profile_l1
```

Optional slow baseline:

```text
networkx_dijkstra
```

---

## Current benchmark result

Representative current run:

```text
Substrates:
    QPROJ = ts_data_d8e9ab3o3njc73eue47g
    GPROJ = ts_gpu_data_4096shots_seed4204302182560473160

CUDA:
    yes

GPU:
    NVIDIA GeForce RTX 3090

Kernel:
    T_S/kernels/ts_geo_kernel.cu
```

### QPROJ scaffold

Top QPROJ edge damages:

```text
edge 5
edge 2
edge 3
```

Top QPROJ round damages:

```text
round 2
round 3
round 0
```

Top QPROJ round-edge damages:

```text
round 2, edge 3
round 2, edge 5
round 4, edge 5
```

Interpretation:

```text
The QPU-derived T_S field carries a localized edge/round scaffold.
The strongest round is round 2.
The strongest edge is edge 5.
The strongest round-edge cell is round 2, edge 3.
```

### GPROJ scaffold

Top GPROJ edge damages:

```text
edge 1
edge 3
edge 4
```

Top GPROJ round damages:

```text
round 2
round 3
round 4
```

Top GPROJ round-edge damages:

```text
round 2, edge 3
round 1, edge 0
round 2, edge 4
```

Interpretation:

```text
The GPU-generated T_S base reproduces the main round and round-edge scaffold.
It does not yet fully reproduce the QPU edge-5 localization.
```

### QPROJ/GPROJ alignment

Current representative alignment:

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

Key read:

```text
qproj and gproj agree on the main round and round-edge scaffold.
qproj and gproj strongly agree on coarse destructive-control ranking.
qproj has sharper edge-5 localization than the current gproj generator.
```

### Method baselines

Current representative method table:

| Method | Round-edge top1 mean | Round-edge Spearman mean | Relative to SciPy |
| ------ | -------------------: | ------------------------: | ----------------: |
| `geo_cuda` | 1.000 | 1.0000 | faster than SciPy on current workload |
| `geo_cpu_dp` | 1.000 | 1.0000 | faster than SciPy on current workload |
| `scipy_dijkstra` | 1.000 | 1.0000 | baseline |
| `stress_profile_l1` | partial signal | partial signal | slower / weaker scaffold recovery |
| `field_profile_l1` | weak signal | weak signal | weaker scaffold recovery |
| `scalar_rate` | weak signal | weak signal | weaker scaffold recovery |

Key read:

```text
geo_cuda, geo_cpu_dp, and scipy_dijkstra agree on the route-optimal scaffold.
geo_cuda is the fastest route method in the current structured-grid benchmark.
generic scalar/profile baselines do not recover the same scaffold.
```

This is the main T_S operator signature.

It means the benchmark is not merely reading scalar field density. The useful signal is carried by stress-derived raw route damage.

---

## CUDA route evaluation

The current benchmark uses one T_S CUDA kernel file:

```text
kernels/ts_geo_kernel.cu
```

Primary kernel:

```text
ts_raw_geo_route_vector_kernel
```

Compatibility kernel:

```text
ts_raw_geo_monotonic_kernel
```

The route-vector kernel computes:

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

The optimized path accelerates the expensive route-evaluation loop:

```text
source × mode × site × ablation × stress-grid route
```

The CUDA boundary is intentionally narrow:

```text
included:
  raw movement costs
  monotonic structured-grid route
  route vector output
  raw-damage support

excluded:
  QPU job submission
  GPU field generation
  stress construction from raw bits
  ablation construction
  normalization
  classification
  qproj/gproj interpretation
```

The benchmark reports whether CUDA is active through the GPU info and kernel path.

Force the CPU/reference path:

```bash
python t_s_benchmark.py --cpu-only
```

Include CPU route comparison:

```bash
python t_s_benchmark.py --include-cpu-route
```

Include NetworkX Dijkstra:

```bash
python t_s_benchmark.py --include-networkx
```

---

## Field controls

The controls deliberately destroy different parts of the T_S channel.

| Control | What it destroys | What it preserves |
| ------- | ---------------- | ----------------- |
| `edge_shuffle` | Channel adjacency / edge order. | Delay, shot, and round values. |
| `round_shuffle` | Repeated-interaction order. | Delay, shot, and edge values. |
| `round_reverse` | Forward temporal orientation. | Round content. |
| `edge_reverse` | Channel orientation. | Edge content. |
| `delay_shuffle` | Delay ordering. | Delay content. |
| `delay_reverse` | Delay orientation. | Delay content. |
| `uniform_by_cell` | Shot-level field structure. | Cellwise delay/round/edge rates. |
| `all_uniform` | Full structured field. | Broad global field density. |

The key T_S read is:

```text
edge and round scaffold controls cause large raw route damage
delay-order controls cause small raw route damage
```

More specifically, current runs show:

```text
all_uniform and uniform_by_cell are highly destructive
edge_shuffle and round_shuffle are load-bearing
delay_shuffle and delay_reverse are comparatively weak
```

This is why T_S is framed as an edge/round scaffold metric rather than a delay-order-only metric.

---

## QPU base workflow

Fresh QPU T_S bases can be generated with the unified QPU CLI.

### Step 1 — submit

```bash
python t_s_qpu_generate.py submit
```

Common overrides depend on the current submitter implementation, but the conceptual axes are:

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

The submitter prints a job ID and the next dump command.

### Step 2 — dump

```bash
python t_s_qpu_generate.py dump <JOB_ID>
```

The dumper writes:

```text
data/ts_data_<JOB_ID>.npz
data/ts_job_<JOB_ID>.json
data/latest_ts_data.json
```

Expected QPU base arrays:

```text
field : uint8, shape (modes, delay_sites, delays, shots, rounds, edges)
final : uint8, shape (modes, delay_sites, delays, shots, channels)
```

A QPU base is consumed by:

```bash
python t_s_benchmark.py
```

or explicitly:

```bash
python t_s_benchmark.py --files data/ts_data_<JOB_ID>.npz
```

---

## GPU bases

Generate GPU T_S bases with:

```bash
python t_s_gpu_generate.py --verify
```

Typical output:

```text
data/ts_gpu_data_<TAG>.npz
data/ts_gpu_job_<TAG>.json
data/latest_ts_gpu_data.json
data/latest_ts_data.json
```

Common options:

```bash
python t_s_gpu_generate.py --shots 4096
python t_s_gpu_generate.py --seed 42
python t_s_gpu_generate.py --verify
python t_s_gpu_generate.py --allow-cpu
```

The GPU generator is not an arbitrary baseline. It exists to create a controlled temporal-stress base with the same downstream schema as the QPU dump.

Current tuned defaults:

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

The point of the GPU base is to separate:

```text
field-operator behavior
```

from:

```text
hardware noise, drift, queue timing, and backend-specific calibration effects
```

---

## Files produced by the pipeline

Common generated files:

```text
data/ts_data_<JOB_ID>.npz
data/ts_gpu_data_<TAG>.npz
data/ts_job_<JOB_ID>.json
data/ts_gpu_job_<TAG>.json
data/latest_ts_data.json
data/latest_ts_gpu_data.json
```

Benchmark output:

```text
analysis/ts_benchmark_<timestamp>/
    ts_benchmark_report.json
    ts_benchmark_summary.csv
    ts_benchmark_rows.csv
    qpu_gpu_alignment.csv
    baseline_comparison.csv
    benchmark_speed.csv
    benchmark_speed_blocks.csv
    edge_damage_rows.csv
    round_damage_rows.csv
    round_edge_damage_rows.csv
    coarse_damage_rows.csv
    edge_damage_compare.png
    round_damage_compare.png
    round_edge_damage_compare.png
    coarse_damage_compare.png
```

Probe output is typically written under:

```text
probes/analysis/
```

Recommended `.gitignore` patterns:

```gitignore
data/ts_data_*.npz
data/ts_gpu_data_*.npz
data/ts_job_*.json
data/ts_gpu_job_*.json
analysis/ts_benchmark_*/
probes/analysis/
*_report.json
```

Keep small curated fixtures if they are part of the reproducibility story.

Keep large generated bases out of git unless intentionally shipping them.

---

## Script map

```text
t_s_benchmark.py
    Current capstone runner:
    T_S geo / qproj / gproj benchmark, destructive controls,
    classical baselines, and timing.

t_s_gpu_generate.py
    Generate GPU T_S bases with QPU-compatible analysis schema.

t_s_qpu_generate.py
    Unified QPU submit/dump CLI:
    submit IBM Runtime jobs and dump completed jobs into T_S .npz bases.

kernels/ts_geo_kernel.cu
    Optimized CUDA raw geo route evaluation and raw-damage support.

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
    CUDA-optimized raw-damage signature finalizer.
```

---

## What to look for

A clean `T_S` run should show at least one of these signatures:

```text
CUDA kernel loads successfully when available
qproj and gproj use the same field schema
raw geo route evaluation succeeds
edge / round / round-edge ablations produce nonzero scaffold damage
round 2 remains load-bearing in the current QPU/GPU pair
round 2, edge 3 remains a shared high-damage round-edge cell
coarse destructive controls rank similarly across qproj and gproj
geo_cuda matches CPU DP and SciPy Dijkstra on the route task
generic scalar/profile baselines fail to recover the same scaffold
route evaluation is much faster than full Python-side ablation/stress construction
```

The strongest current signature is:

```text
geo_cuda / geo_cpu_dp / scipy_dijkstra agree on route scaffold
scalar_rate / field_profile_l1 / stress_profile_l1 do not recover the same scaffold
qproj and gproj agree on round and round-edge top structure
```

That is the evidence that `T_S` is reading stress-derived route structure rather than only scalar field density.

Do not overread the result as a hardware-speed claim. The value of `T_S` is field measurement, destructive controls, route testing, and substrate comparison.

---

## Current bounded claim

`T_S` is a live research object. The repo keeps probes because they document how the stress/route framing evolved.

The current bounded claim is:

```text
T_S is a temporal-stress field operator with three-path expression:
  1. raw geo route path,
  2. GPU-generated temporal-stress base,
  3. real QPU temporal-stress base.
```

The current benchmark evidence is:

```text
1. The benchmark loads qproj and gproj T_S records under one shared task harness.
2. Stress tensor components are derived from delay/round/edge gradients.
3. Raw geo routes identify load-bearing edge/round scaffold components.
4. QPROJ and GPROJ agree on the main round and round-edge structure in the current run.
5. QPROJ and GPROJ agree strongly on coarse destructive-control ranking.
6. GEO CUDA matches CPU DP and SciPy Dijkstra on the current route task.
7. GEO CUDA is faster than SciPy Dijkstra on the current structured-grid workload.
8. Generic scalar/profile baselines do not recover the same round-edge scaffold.
```

The honest framing is:

```text
T_S is not a quantum advantage claim.
T_S is not a QPU speedup claim.
T_S is not a universal shortest-path algorithm.
T_S is not a QPU simulator.
T_S does not fully match qproj edge localization with gproj yet.
T_S is useful as a field-structured, stress-derived, route-tested temporal metric.
```

That is the claim to defend.

---

## Next development steps

Likely next steps:

```text
more QPU jobs for multi-job qproj comparison
more backend runs
better gproj edge-localization tuning
qproj/gproj alignment table rename from qpu_gpu_alignment.csv to qproj_gproj_alignment.csv
move more ablation/stress construction into CUDA
curated fixture policy for reproducibility
known_issues.md cleanup
docs synchronization with final benchmark output
```

The process is the process.

Break it, fix it, document what happened.
