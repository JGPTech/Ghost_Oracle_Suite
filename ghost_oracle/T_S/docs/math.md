# Math

Mathematical definition of the `T_S` operator: temporal stress fields, raw geo routes, edge/round scaffold damage, qproj/gproj substrate comparison, CUDA route evaluation, classical baselines, and the benchmark-supported claim boundary.

`T_S` is the **Temporal Stress Metric**.

The architecture document explains how this math gets compiled into the package and CUDA kernel. The probes record how the field framing evolved. This document is the math standing alone.

GitHub renders LaTeX inside `$...$` and `$$...$$` delimiters. Everything here is written as raw Markdown and should be pasted as source, not copied from a rendered preview.

Conventions used throughout:

* $m$ indexes a perturbation mode.
* $s$ indexes a delay insertion site.
* $\delta$ indexes a delay value.
* $q$ indexes a shot.
* $r$ indexes temporal interaction round.
* $e$ indexes an adjacent channel edge.
* $N_{\mathrm{shots}}$ is the number of shots in a base, usually $4096$ by default.
* $R$ is the number of temporal rounds, usually $6$ by default.
* $C$ is the number of channels, usually $8$ by default.
* $E=C-1$ is the number of adjacent channel edges, usually $7$ by default.
* A **record** is one measured field block for a mode, delay site, delay, shot, round, and edge.
* A **base** is a `.npz` file containing a T_S field and final channel data.
* A **substrate** is one implementation/source of the same T_S schema:

  * real QPU temporal-stress base from IBM Runtime,
  * GPU-generated temporal-stress base,
  * raw geo arithmetic route path derived from the same field.

The shared T_S array schema is:

```text
field : uint8, shape (modes, delay_sites, delays, shots, rounds, edges)
final : uint8, shape (modes, delay_sites, delays, shots, channels)
```

Default mode labels are:

```text
clean
phase_shear
local_shock
```

Default delay-site labels are:

```text
pre_coupling
post_coupling
post_perturb
```

Default delay values are:

```text
0, 1, 2, 4, 8, 16
```

---

## The T_S object

The current operator is:

$$
T_S = \{\Phi[m,s,\delta,q,r,e],\Theta[m,s,\delta,q,c],\mathcal{T}_{\mu\nu}[m,s]\}
$$

where:

$$
\Phi[m,s,\delta,q,r,e]\in\{0,1\}
$$

is the measured temporal edge field,

$$
\Theta[m,s,\delta,q,c]\in\{0,1\}
$$

is the final channel readout, and:

$$
\mathcal{T}_{\mu\nu}[m,s]
$$

is the stress tensor field derived from delay, round, and edge gradients of $\Phi$.

In words:

```text
Phi    = temporal edge-channel field
Theta  = final channel readout
T_munu = temporal stress tensor derived from Phi
```

`T_S` is not a single scalar.

It is a structured field relation:

```text
delay axis
+
round axis
+
edge/channel axis
+
mode/site perturbation context
+
raw geo route response under ablation
```

The benchmark then asks whether that structure survives destructive controls and whether qproj/gproj substrates carry the same route-preserving scaffold.

---

## Temporal edge field

For one substrate base, the primary object is:

$$
\Phi[m,s,\delta,q,r,e]
$$

with:

$$
m=0,\ldots,M-1
$$

$$
s=0,\ldots,S-1
$$

$$
\delta=0,\ldots,D-1
$$

$$
q=0,\ldots,N_{\mathrm{shots}}-1
$$

$$
r=0,\ldots,R-1
$$

$$
e=0,\ldots,E-1
$$

The field has shape:

```text
modes × delay_sites × delays × shots × rounds × edges
```

For a fixed mode and delay site, write one block as:

$$
\Phi_{m,s}[\delta,q,r,e]
$$

with shape:

```text
delays × shots × rounds × edges
```

The T_S probes operate primarily on this block.

The field is edge-local. It records adjacent channel-pair behavior over repeated temporal rounds under structured delay placement.

This is the first important T_S step:

```text
channel measurements are converted into edge/channel field structure
the repeated rounds make the field temporal
the delay axis lets disruption be probed
```

---

## Final channel readout

The final channel readout is:

$$
\Theta[m,s,\delta,q,c]
$$

where:

$$
c=0,\ldots,C-1
$$

The final channel readout is saved for schema compatibility and downstream diagnostics.

The adjacent final edge parity is:

$$
P_{\Theta}[m,s,\delta,q,e]
=
\Theta[m,s,\delta,q,e]
\oplus
\Theta[m,s,\delta,q,e+1]
$$

This can be compared to the final temporal edge round:

$$
\Phi[m,s,\delta,q,R-1,e]
$$

as a loose terminal consistency diagnostic.

The current T_S claim does not depend on terminal parity alone.

The central object is the temporal stress field derived from $\Phi$.

---

## Delay, round, and edge gradients

T_S begins by taking binary finite differences of the temporal edge field.

For a fixed block $\Phi_{m,s}$, define the delay gradient:

$$
\nabla_{\tau}\Phi[\delta,q,r,e]
=
\Phi[\delta+1,q,r,e]\oplus\Phi[\delta,q,r,e]
$$

the round gradient:

$$
\nabla_{r}\Phi[\delta,q,r,e]
=
\Phi[\delta,q,r+1,e]\oplus\Phi[\delta,q,r,e]
$$

and the edge gradient:

$$
\nabla_{x}\Phi[\delta,q,r,e]
=
\Phi[\delta,q,r,e+1]\oplus\Phi[\delta,q,r,e]
$$

The gradients are aligned onto the common lattice:

```text
delay_cell × shot × round_cell × edge_cell
```

where:

```text
delay_cell = delays - 1
round_cell = rounds - 1
edge_cell  = edges - 1
```

This common lattice is the T_S stress grid.

---

## Temporal stress tensor

The local shot-averaged stress components are:

$$
T_{\tau\tau}[\delta,r,e]
=
\mathbb{E}_{q}\left[(\nabla_{\tau}\Phi)^2\right]
$$

$$
T_{rr}[\delta,r,e]
=
\mathbb{E}_{q}\left[(\nabla_{r}\Phi)^2\right]
$$

$$
T_{xx}[\delta,r,e]
=
\mathbb{E}_{q}\left[(\nabla_{x}\Phi)^2\right]
$$

and the mixed terms are:

$$
T_{\tau r}[\delta,r,e]
=
\mathbb{E}_{q}\left[\nabla_{\tau}\Phi\cdot\nabla_{r}\Phi\right]
$$

$$
T_{\tau x}[\delta,r,e]
=
\mathbb{E}_{q}\left[\nabla_{\tau}\Phi\cdot\nabla_{x}\Phi\right]
$$

$$
T_{rx}[\delta,r,e]
=
\mathbb{E}_{q}\left[\nabla_{r}\Phi\cdot\nabla_{x}\Phi\right]
$$

The stress trace is:

$$
\mathrm{tr}(T)
=
T_{\tau\tau}+T_{rr}+T_{xx}
$$

In code, the stress tensor is represented by six component arrays:

```text
tau_tau
rr
xx
tau_r
tau_x
r_x
```

and the trace:

```text
trace = tau_tau + rr + xx
```

The stress tensor is the central T_S field object.

---

## Raw geo movement costs

T_S converts stress into raw movement costs over the stress grid.

With coupling weight $\lambda$, the movement costs are:

$$
C_{\tau}
=
T_{\tau\tau}
+
\lambda(T_{\tau r}+T_{\tau x})
$$

$$
C_{r}
=
T_{rr}
+
\lambda(T_{\tau r}+T_{rx})
$$

$$
C_{x}
=
T_{xx}
+
\lambda(T_{\tau x}+T_{rx})
$$

The default benchmark uses:

```text
coupling_weight = 0.50
```

These costs are raw.

There is no normalization in the final T_S route benchmark.

The only clipping is a numerical crash guard:

```text
min_cost = 1e-9
max_cost = 1e9
```

This is not feature normalization. It is a finite-value guard.

---

## Raw geo route

The raw geo path is a monotonic route through the stress grid.

Let the stress grid have shape:

```text
A × R' × E'
```

where:

```text
A  = delays - 1
R' = rounds - 1
E' = edges - 1
```

The source is:

$$
(0,0,0)
$$

and the target is:

$$
(A-1,R'-1,E'-1)
$$

The dynamic-programming recurrence is:

$$
G[a,r,e]
=
\min
\begin{cases}
G[a-1,r,e]+C_{\tau}[a,r,e] \\
G[a,r-1,e]+C_{r}[a,r,e] \\
G[a,r,e-1]+C_{x}[a,r,e]
\end{cases}
$$

with:

$$
G[0,0,0]=0
$$

The full raw geo route cost is:

$$
\mathrm{Geo}_{\mathrm{full}}
=
G[A-1,R'-1,E'-1]
$$

The benchmark also records a delay-only route through the center round/edge:

$$
\mathrm{Geo}_{\mathrm{delay}}
=
\sum_{a=1}^{A-1} C_{\tau}[a,r_c,e_c]
$$

and an edge-only route through the center delay/round:

$$
\mathrm{Geo}_{\mathrm{edge}}
=
\sum_{e=1}^{E'-1} C_x[a_c,r_c,e]
$$

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

---

## Stress avoidance proxy

The benchmark computes:

$$
\bar{T}
=
\frac{1}{A R' E'}
\sum_{\delta,r,e}
\mathrm{tr}(T)[\delta,r,e]
$$

It also computes a lightweight diagonal path trace proxy:

$$
\bar{T}_{\mathrm{path}}
=
\frac{1}{K}
\sum_{k=0}^{K-1}
\mathrm{tr}(T)[a_k,r_k,e_k]
$$

where $(a_k,r_k,e_k)$ samples a diagonal-like route through the stress grid.

The stress avoidance proxy is:

$$
\mathrm{avoidance}
=
\bar{T}-\bar{T}_{\mathrm{path}}
$$

This is not the only possible route-trace statistic.

It is the current lightweight benchmark proxy.

---

## Raw-damage projection signature

T_S finalizes qproj/gproj by measuring direct raw damage under scaffold ablations.

For a real block and an ablated block, with route vectors:

$$
\rho
$$

and:

$$
\rho'
$$

define:

$$
\Delta_{\mathrm{full}}
=
\rho'_0-\rho_0
$$

$$
\Delta_{\mathrm{delay}}
=
\rho'_1-\rho_1
$$

$$
\Delta_{\mathrm{edge}}
=
\rho'_2-\rho_2
$$

and:

$$
L_{\mathrm{avoid}}
=
\max(0,\rho_6-\rho'_6)
$$

The total raw damage is:

$$
D_{\mathrm{raw}}
=
|\Delta_{\mathrm{full}}|
+
|\Delta_{\mathrm{delay}}|
+
|\Delta_{\mathrm{edge}}|
+
L_{\mathrm{avoid}}
$$

The damage vector layout is:

```text
damage[0] = total_damage
damage[1] = full_delta
damage[2] = delay_delta
damage[3] = edge_delta
damage[4] = avoidance_loss
damage[5] = abs_full_delta
damage[6] = abs_delay_delta
damage[7] = abs_edge_delta
```

This is the current T_S projection signature.

No cosine similarity is used.

No vector normalization is used.

No classifier is used.

---

## Edge, round, and round-edge ablations

The final projection signature uses three local ablation families.

### Edge ablation

For each edge $e$, replace that edge track with cellwise marginal samples:

$$
\Phi[\delta,q,r,e]
\mapsto
\mathrm{Bernoulli}
\left(
\mathbb{E}_q[\Phi[\delta,q,r,e]]
\right)
$$

Then recompute stress, raw geo routes, and damage.

This produces:

$$
D_{\mathrm{edge}}[e]
$$

### Round ablation

For each round $r$, replace that round with delay/edge cellwise marginal samples:

$$
\Phi[\delta,q,r,e]
\mapsto
\mathrm{Bernoulli}
\left(
\mathbb{E}_q[\Phi[\delta,q,r,e]]
\right)
$$

Then recompute stress, raw geo routes, and damage.

This produces:

$$
D_{\mathrm{round}}[r]
$$

### Round-edge ablation

For each round-edge track $(r,e)$, replace that track with delay-specific shot marginals:

$$
\Phi[\delta,q,r,e]
\mapsto
\mathrm{Bernoulli}
\left(
\mathbb{E}_q[\Phi[\delta,q,r,e]]
\right)
$$

Then recompute stress, raw geo routes, and damage.

This produces:

$$
D_{\mathrm{round,edge}}[r,e]
$$

These arrays are the core projection signature:

```text
edge_damage[mode, site, edge]
round_damage[mode, site, round]
round_edge_damage[mode, site, round, edge]
```

---

## Coarse destructive controls

The benchmark also uses coarse destructive controls.

The control set is:

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

### edge_shuffle

Shuffle edge order within each delay/shot/round cell.

This attacks adjacent channel ordering.

### round_shuffle

Shuffle temporal round order within each delay/shot/edge track.

This attacks repeated-interaction ordering.

### round_reverse

Reverse the temporal round axis.

This preserves round content but changes forward temporal orientation.

### edge_reverse

Reverse the edge axis.

This preserves edge content but changes channel orientation.

### delay_shuffle

Shuffle the delay axis.

This attacks delay ordering.

### delay_reverse

Reverse the delay axis.

This preserves delay content but changes delay orientation.

### uniform_by_cell

Replace samples with Bernoulli draws matching each delay/round/edge cell rate.

This preserves cellwise intensity but destroys shot-level field structure.

### all_uniform

Replace the whole block with Bernoulli draws matching the global block rate.

This preserves only broad scalar density.

---

## Substrate paths

The benchmark uses three conceptual paths:

$$
T_S^{\mathrm{geo}}
$$

$$
T_S^{\mathrm{qproj}}
$$

$$
T_S^{\mathrm{gproj}}
$$

In plain text:

```text
T_S_geo    = raw arithmetic route path derived from a T_S field
T_S_qproj  = real QPU temporal-stress base
T_S_gproj  = GPU-generated temporal-stress base
```

### T_S_geo

The `geo` path is the raw arithmetic route operator.

It takes any valid T_S field, computes stress, and evaluates raw geo routes and route damage.

It is implemented by:

```text
ts_raw_geo_route_vector_kernel
```

inside:

```text
kernels/ts_geo_kernel.cu
```

The CPU reference is:

```text
geo_cpu_dp
```

The CUDA path is:

```text
geo_cuda
```

### T_S_qproj

The `qproj` path loads a real QPU T_S base:

```text
data/ts_data_<JOB_ID>.npz
```

The QPU path is not a speed claim.

It is a field-substrate claim:

```text
Does real hardware produce a temporal stress field whose route-preserving scaffold survives destructive controls?
```

### T_S_gproj

The `gproj` path loads a GPU-generated T_S base:

```text
data/ts_gpu_data_<TAG>.npz
```

It uses the same schema as QPU dumps.

Its role is to provide a controlled generated comparison substrate.

The GPU generator is not a QPU simulator claim.

It is a local controlled substrate generator.

---

## QPROJ/GPROJ agreement metric

The final benchmark writes:

```text
qpu_gpu_alignment.csv
```

The printed report calls this:

```text
QPROJ/GPROJ ALIGNMENT
```

It compares qproj and gproj scaffold signatures.

For edge damage arrays:

$$
D^{q}_{\mathrm{edge}}
$$

and:

$$
D^{g}_{\mathrm{edge}}
$$

the benchmark reports:

```text
top1_match
Spearman rank correlation
top3_overlap
top5_overlap
```

The same is done for:

```text
round
round_edge
coarse controls
```

For the current benchmark run:

```text
edge top1 match      = 0
round top1 match     = 1
round_edge top1 match = 1
coarse Spearman      ≈ 0.976
```

The interpretation is:

```text
gproj reproduces the main round and round-edge scaffold,
but qproj has stronger edge-5 localization than the current GPU generator.
```

---

## Final benchmark task

The current canonical runner is:

```bash
python t_s_benchmark.py
```

By default it scans every valid `.npz` in:

```text
ghost_oracle/T_S/data/
```

A valid T_S file contains:

```text
field
```

with shape:

```text
modes × delay_sites × delays × shots × rounds × edges
```

Specific files can be selected with:

```bash
python t_s_benchmark.py --files file1.npz file2.npz
```

The final benchmark has two layers.

### Layer 1: substrate comparison

This compares qproj and gproj as substrates.

It asks:

```text
Do QPU and GPU generated fields identify similar load-bearing scaffold structure?
```

Current result:

```text
round top1 match      = 1.000
round_edge top1 match = 1.000
coarse Spearman       ≈ 0.976
edge top5 overlap     = 1.000
```

The edge top1 mismatch is informative, not a failure.

It identifies the current gap between real QPU localization and the GPU generator.

### Layer 2: method comparison

This compares route/scaffold methods on the same ablation task.

Current method families:

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

The current benchmark result is:

| Method | Round-edge top1 mean | Round-edge Spearman mean | Mean seconds | Relative to SciPy |
| ------ | -------------------: | ------------------------: | -----------: | ----------------: |
| `geo_cuda` | 1.000 | 1.0000 | 0.001770 | 9.78x |
| `geo_cpu_dp` | 1.000 | 1.0000 | 0.009017 | 1.92x |
| `scipy_dijkstra` | 1.000 | 1.0000 | 0.017298 | 1.00x |
| `stress_profile_l1` | 0.278 | 0.3196 | 2.467438 | 0.01x |
| `field_profile_l1` | 0.056 | 0.0260 | 0.648592 | 0.03x |
| `scalar_rate` | 0.056 | 0.0256 | 0.073423 | 0.24x |

The read is:

```text
geo_cuda, geo_cpu_dp, and scipy_dijkstra agree on the route-optimal scaffold.
geo_cuda is the fastest route method in the current benchmark.
generic scalar/profile baselines do not recover the same scaffold.
```

This supports the claim that the T_S route operator is not just measuring scalar rate or simple profile distance.

---

## Classical baseline boundary

`scipy_dijkstra` is a generic sparse-graph shortest path method.

It solves the same monotonic route task after the T_S stress grid is converted into a graph.

That makes it a fair adjacent baseline for route correctness.

But it is more general than T_S needs.

The T_S geo path is specialized to the structured monotonic stress grid.

The benchmark-supported claim is therefore:

```text
T_S geo_cuda matches the generic graph-route answer on the T_S route task,
while running faster on the current structured-grid workload.
```

This is not a universal shortest-path claim.

It is a structured-grid route claim.

---

## CUDA route evaluation

The optimized CUDA path computes the same raw route vector as the CPU DP reference.

The core kernel is:

```text
ts_raw_geo_route_vector_kernel
```

Inputs are flattened stress components:

```text
tau_tau
rr
xx
tau_r
tau_x
r_x
```

with shape:

```text
B × (delay_cell × round_cell × edge_cell)
```

For each batch item, the kernel computes:

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

The compatibility kernel:

```text
ts_raw_geo_monotonic_kernel
```

is kept for older probes.

The enhanced kernel also defines raw-damage helpers for future optimized packing.

The CUDA path does not change the operator.

It accelerates route evaluation.

The current timing split shows that route evaluation is no longer the bottleneck:

```text
route seconds ≪ total benchmark seconds
```

Most remaining wall-clock time is Python-side ablation generation and stress-batch construction.

---

## GPU base generation

The GPU generator is:

```text
t_s_gpu_generate.py
```

It writes:

```text
data/ts_gpu_data_<TAG>.npz
data/ts_gpu_job_<TAG>.json
data/latest_ts_gpu_data.json
data/latest_ts_data.json
```

The generated file uses the same analysis-facing schema as the QPU dump:

```text
field[mode, delay_site, delay, shot, round, edge]
final[mode, delay_site, delay, shot, channel]
```

The GPU generator is a controlled substrate generator.

It is not a QPU simulator claim.

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

These defaults were chosen because they preserve the round/round-edge scaffold under the current probe stack while keeping delay order weak and raw geo mostly stable.

---

## What the current math supports

The current mathematical and benchmark-supported claim is that `T_S` is a temporal-stress field operator with:

1. a raw geo arithmetic route path,
2. a GPU-generated temporal-stress base path,
3. a real QPU temporal-stress base path.

The current evidence supports:

* T_S fields can be represented under a common qproj/gproj schema;
* delay, round, and edge gradients define a local stress tensor;
* raw geo routes define a structured route over that stress tensor;
* edge/round/round-edge ablations identify load-bearing scaffold components;
* qproj and gproj agree strongly on round and round-edge top structure;
* qproj and gproj agree strongly on coarse destructive-control ranking;
* qproj has stronger edge-5 localization than the current gproj generator;
* geo_cuda matches CPU DP and SciPy Dijkstra on the route task;
* geo_cuda is faster than SciPy Dijkstra on the current structured-grid workload;
* scalar/profile baselines do not recover the same round-edge scaffold.

The current evidence does **not** support:

* a quantum advantage claim;
* a QPU speedup claim;
* a universal shortest-path claim;
* a claim that the GPU generator simulates the QPU;
* a claim that edge localization is fully matched between qproj and gproj;
* a logical-error-rate claim;
* a claim that S_M, G_M, token retrieval, or TSP projectors are proven by T_S;
* a claim that all future Converger operators are validated.

The correct bounded framing is:

$$
T_S
=
\text{field-structured, stress-derived, route-tested temporal metric}
$$

---

## Pointers

* **`t_s_qpu_generate.py`** — QPU temporal-stress submit/dump path.
* **`t_s_gpu_generate.py`** — GPU-generated temporal-stress base generator.
* **`t_s_benchmark.py`** — Final T_S benchmark runner.
* **`kernels/ts_geo_kernel.cu`** — CUDA raw geo route evaluation and raw-damage support.
* **`probes/t_s_probe1.py`** — Temporal-stress shape and tensor summary.
* **`probes/t_s_probe2_geo.py`** — Geo reconstruction and transform sweep.
* **`probes/t_s_probe3_raw_first_jumpgeo.py`** — Raw-first JumpGeo policy.
* **`probes/t_s_probe5_qpu_geo_origin.py`** — QPU geo-origin structure ablation.
* **`probes/t_s_probe6_qpu_projection_raw_damage.py`** — Raw-damage qproj finalizer.
* **`probes/t_s_probe7_qpu_projection_raw_damage_cuda.py`** — Optimized CUDA qproj/gproj raw-damage finalizer.
* **`docs/architecture.md`** — How the math compiles into package architecture and kernels.
* **`docs/known_issues.md`** — Current limitations and failure modes.

---

## Final read

The process is the process.

T_S started as a stress-tensor idea attached to earlier field probes, separated into its own operator, defined a QPU temporal field, extracted stress tensor components, found a raw-first geo path, identified the QPU edge/round scaffold, created an interchangeable GPU substrate, and now benchmarks geo, qproj, and gproj under one harness.

The math says:

$$
\Phi[m,s,\delta,q,r,e]\text{ is the temporal edge field}
$$

$$
\nabla_{\tau}\Phi,\nabla_r\Phi,\nabla_x\Phi
\text{ define delay, round, and edge gradients}
$$

$$
T_{\tau\tau},T_{rr},T_{xx},T_{\tau r},T_{\tau x},T_{rx}
\text{ define the temporal stress tensor}
$$

$$
C_{\tau},C_r,C_x
\text{ define raw route movement costs}
$$

$$
D_{\mathrm{raw}}
\text{ measures damage under edge/round/round-edge ablation}
$$

The benchmark says:

```text
qproj and gproj agree on the main round/round-edge scaffold
qproj and gproj agree strongly on coarse destructive-control structure
geo_cuda matches CPU DP and SciPy Dijkstra on the route task
geo_cuda is faster on the current structured-grid workload
generic scalar/profile baselines do not recover the same scaffold
```

That is the claim to defend.
