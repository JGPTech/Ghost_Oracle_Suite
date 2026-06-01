# T_S Process Record — Temporal Stress Metric Operator

This document records the research and engineering trajectory of the `T_S` operator package inside Ghost Oracle Suite.

It is split from the larger Ghost Oracle process record so that `T_S` can stand on its own as a finished operator package for this version.

It is chronological. It includes the messy path. It includes the older S_M stress-tensor framing, the first QPU temporal field probe, the corrections around normalization, the separation between geo/qproj/gproj, the GPU generator, the CUDA kernel optimization, the final benchmark, and the documentation cleanup.

It is not a polished victory lap. It is a working record.

Current framing:

```text
T_S = Temporal Stress Metric
```

Current bounded claim:

```text
T_S is a temporal-stress field operator.

It measures whether delayed temporal edge-channel records form a load-bearing
stress/route scaffold across QPU-derived and GPU-generated records.
```

Current non-claims:

```text
T_S is not a quantum advantage claim.
T_S is not a QPU speedup claim.
T_S is not a universal shortest-path algorithm.
T_S is not a QPU simulator.
T_S is not an S_M logical-error-rate benchmark.
```

---

## Part 1 — Stress tensor starts inside the S_M workspace

The `T_S` path began inside the older `S_M` work.

At that stage, repeated syndrome or channel records were being explored not only as field objects, but also as possible stress-tensor objects. The original instinct was:

```text
take changes across time / round / edge
treat those changes as rates
turn the rates into tensor-like stress components
```

This was connected to an older idea from the broader EchoKey / Ghost Oracle work:

```text
rotational rates of Pauli-like components can behave like stress-tensor values
```

The early S_M stress tensor framing used syndrome-spacetime gradients:

```text
Ttt = <Delta_t S Delta_t S>
Txx = <Delta_x S Delta_x S>
Ttx = <Delta_t S Delta_x S>
```

That framing was valuable, but it created a boundary problem.

`S_M` was supposed to be:

```text
S_M = Syndrome Metric
```

where the core object is final edge parity and repeated syndrome records.

The stress tensor was becoming its own operator.

The cleanup decision was:

```text
S_M remains the syndrome-spacetime field operator.
T_S becomes the stress tensor / temporal stress channel.
```

This was the first important architectural split.

---

## Part 2 — Naming the package: T_S — Temporal Stress Metric

The operator was eventually named:

```text
T_S — Temporal Stress Metric
```

The name mattered because it forced the package to stop being treated as a loose S_M property.

The new package direction became:

```text
T_S/
├── t_s_qpu_generate.py
├── t_s_gpu_generate.py
├── t_s_benchmark.py
├── kernels/
├── probes/
├── docs/
├── data/
└── analysis/
```

The guiding idea:

```text
same methodology as G_M and S_M
different substrate
different operator
same discipline
```

The standard Converger package pattern was:

```text
operator_qpu_generate.py
operator_gpu_generate.py
operator_benchmark.py
```

For T_S:

```text
t_s_qpu_generate.py
t_s_gpu_generate.py
t_s_benchmark.py
```

The goal was not to keep extending S_M forever.

The goal was to let T_S become its own finished operator package.

---

## Part 3 — Reimagining the old S_M-like probe

The first T_S work reused an older probe where the stress tensor had been treated as an S_M property.

That was intentionally reimagined.

The important instruction was:

```text
same physics
same math
custom QPU generator for the job
```

The object was no longer:

```text
S_M stress side feature
```

It became:

```text
T_S temporal edge-channel field
```

The core QPU field object became:

```text
F[mode, delay_site, delay_value, shot, round, edge]
```

Later documentation used:

```text
Phi[mode, delay_site, delay, shot, round, edge]
```

but the object is the same.

The key design was to use structured time delays rather than treat delay as a decorative parameter.

The modes were:

```text
clean
phase_shear
local_shock
```

The delay sites were:

```text
pre_coupling
post_coupling
post_perturb
```

The delay values were:

```text
0, 1, 2, 4, 8, 16 dt
```

This gave T_S its field object:

```text
mode × delay_site × delay × shot × round × edge
```

---

## Part 4 — Probe 01: temporal stress field exists

The first working T_S run produced a field shape:

```text
(3, 3, 6, 4096, 6, 7)
```

Interpretation:

```text
3 modes
3 delay sites
6 delay values
4096 shots
6 rounds
7 adjacent channel edges
```

Probe 01 summarized rates, field L2, delay coefficient of variation, round coefficient of variation, and edge coefficient of variation.

Representative field summary:

```text
mode          site            rate     field L2   delay CV   round CV   edge CV
clean         pre_coupling    0.17582  1.7172     0.0185     0.1650     0.3394
phase_shear   post_perturb    0.18818  1.9202     0.0255     0.1547     0.3477
local_shock   pre_coupling    0.23958  2.0878     0.0113     0.1917     0.2845
```

The temporal stress summary computed:

```text
Ttau_tau
Trr
Txx
Ttau_x
Trx
trace
delay gap
survival gap
```

Representative values:

```text
clean / pre_coupling:
  trace = 0.72380

phase_shear / post_perturb:
  trace = 0.75387

local_shock / pre_coupling:
  trace = 0.91572
```

The first important result:

```text
local_shock increases stress trace
phase_shear increases stress relative to clean
delay/channel field structure survives the disruption
```

Probe 01 established:

```text
T_S has a real field object
T_S produces a stress tensor summary
the object is not empty or scalar-only
```

---

## Part 5 — Repo path convention for probes

Once the first probe worked, the T_S probe path was standardized.

The probe file was renamed:

```text
t_s_probe1.py
```

and placed in:

```text
T_S/probes/
```

The path convention was fixed:

```python
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
ANALYSIS_DIR = HERE / "analysis"
```

This was important because the probes should not depend on the current working directory.

It also aligned T_S with the S_M package style.

The QPU generator was considered good enough at this stage.

The direction became:

```text
do not keep modifying the generator
use the generator output
find geo first
then projectors
```

---

## Part 6 — Clarifying geo

The next question was:

```text
what is geo?
```

At first, it was easy to drift and confuse geo with qproj/gproj.

The corrected definition became:

```text
geo = arithmetic path derived by the quantum analysis
```

More specifically:

```text
geo = raw route/stress path over the T_S stress tensor
```

This distinction was load-bearing.

Correct separation:

```text
geo:
  route arithmetic / stress-grid path

qproj:
  QPU-generated temporal field substrate

gproj:
  GPU-generated temporal field substrate
```

Incorrect conflation:

```text
geo is the QPU projection
geo is the GPU projection
qproj/gproj are just geo outputs
```

That drift was corrected early.

The architecture after correction:

```text
qproj and gproj produce comparable field substrates
geo evaluates raw route/stress structure from those fields
```

---

## Part 7 — Probe 02 / Probe 03: raw-first JumpGeo policy

The geo path was tested with several transform options.

The initial temptation was to normalize the field.

But the important correction was:

```text
why are we normalizing?
```

T_S had already inherited the key G_M/S_M lesson:

```text
QPU data is naturally bounded by measurement.
Do not add normalization unless the stress level demands it.
```

The raw-first policy was:

```text
lean on raw whenever possible
only jump to trace norm or power expansion under disruption
```

The transform family became:

```text
raw
trace_norm
pow_exp_2
pow_exp_4
pow_exp_8
...
```

The policy was not:

```text
always normalize
```

It was:

```text
raw first
jump only when stress requires it
```

A later GPU-generated run showed:

```text
raw selected    = 35 / 36 = 97.22%
transform jumps = 1 / 36 = 2.78%
selected        = {'raw': 35, 'pow_exp_4': 1}
```

A cleaner earlier GPU run showed:

```text
raw selected    = 36 / 36 = 100%
transform jumps = 0
```

The lesson:

```text
raw is the primary T_S route path
jumps are exceptional stress-handling events
normalization is not the default operator
```

---

## Part 8 — Probe 04: standalone geo speed path

A standalone geo speed probe was created to isolate the route arithmetic path.

The idea was:

```text
geo should be the optimized arithmetic route path
eventually it should run on GPU
it should not depend on qproj/gproj bases
```

That probe motivated a CUDA kernel for the route path.

The first standalone geo speed test showed that the path was becoming clean and fast.

But later the standalone probe was canceled as a final package element because:

```text
it did not use our T_S data
```

The useful part of Probe 04 survived as:

```text
T_S needs a CUDA geo kernel
```

The standalone non-data benchmark itself was not kept as part of the final claim.

This is important process history:

```text
The arithmetic idea was useful.
The standalone non-data benchmark was not the final T_S evidence.
```

---

## Part 9 — First QPU projection attempt and correction

The first attempt at a QPU projection test drifted into a classification-like projection framing.

It compared feature families and targets such as:

```text
mode
site
mode_site
```

with controls such as:

```text
real
delay_shuffle
round_shuffle
edge_shuffle
all_uniform
```

That produced some numbers, but it was the wrong question.

The output looked like a classifier/projection benchmark.

The correction was direct:

```text
geo is the arithmetic path
qpu and gpu are projections of base files
totally different thing
same methodology, different substrate
```

This reset the direction.

The proper qproj task was not:

```text
classify mode/site labels from features
```

The proper qproj task was:

```text
find where the QPU field naturally carries the route-preserving scaffold
```

That correction prevented T_S from being forced into the wrong benchmark box.

---

## Part 10 — Probe 05: QPU geo origin / structure ablation

The corrected Probe 05 became:

```text
T_S PROBE 05 — QPU GEO ORIGIN / STRUCTURE ABLATION
```

It used the real QPU field:

```text
Field shape: (3, 3, 6, 4096, 6, 7)
```

Controls:

```text
real
shot_shuffle
delay_shuffle
round_shuffle
edge_shuffle
delay_reverse
round_reverse
edge_reverse
uniform_by_cell
all_uniform
mode_pool_same_site
site_pool_same_mode
```

The summary showed that raw was the selected transform for every mode/site row:

```text
xform = raw
```

Representative row:

```text
clean / pre_coupling:
  trace = 0.72380
  full  = 2.7777
  delay = 1.4663
  delay/full = 0.5279
  top damage = all_uniform
  score = 1.32909
```

Aggregate geo-origin damage:

```text
all_uniform          1.298311
uniform_by_cell      0.711276
edge_shuffle         0.701817
mode_pool_same_site  0.625173
round_shuffle        0.410396
round_reverse        0.409894
edge_reverse         0.147851
site_pool_same_mode  0.056932
delay_reverse        0.042132
delay_shuffle        0.032538
shot_shuffle         0.000000
```

The first big T_S structural conclusion:

```text
The QPU geo-origin structure is not primarily delay order.
It is carried by edge/round/channel scaffold structure.
```

Delay shuffling and delay reversal caused small damage.

Edge shuffling, uniformization, and round disruption caused much larger damage.

This was the first clean QPU scaffold result.

---

## Part 11 — Probe 06: raw-damage QPU projection finalizer

Probe 06 finalized the QPU projection as a raw-damage signature.

Important correction:

```text
no normalization
no cosine
no classifier
no GPU base generation
no G_M/S_M bases
no projector benchmark
```

The method became:

```text
raw field
-> raw stress
-> raw geo routes
-> raw route damage
-> QPU projection signature
```

Damage definition:

```text
damage =
    abs(full_cost_ablated  - full_cost_real)
  + abs(delay_cost_ablated - delay_cost_real)
  + abs(edge_cost_ablated  - edge_cost_real)
  + max(0, real_avoidance - ablated_avoidance)
```

Probe 06 output:

```text
T_S PROBE 06 — QPU PROJECTION FINALIZER: RAW DAMAGE SIGNATURE
```

Representative QPU signature rows:

```text
clean / pre_coupling:
  full      = 2.7777
  delay     = 1.4663
  edge      = 1.5328
  trace     = 0.72380
  edge max  = 1.00619
  round max = 1.00583
  r×e max   = 0.48896

local_shock / pre_coupling:
  full      = 3.7567
  delay     = 2.1805
  edge      = 1.8739
  trace     = 0.91572
  edge max  = 1.13565
  round max = 1.38195
  r×e max   = 0.62713
```

Coarse raw structure damage:

```text
all_uniform      3.901344
uniform_by_cell  2.158488
edge_shuffle     2.109326
round_shuffle    1.320216
round_reverse    1.244593
edge_reverse     0.445912
delay_shuffle    0.126770
delay_reverse    0.126615
```

Top edge damages:

```text
edge 5: mean=0.997574
edge 2: mean=0.648359
edge 3: mean=0.648087
edge 1: mean=0.375353
edge 4: mean=0.277190
```

Top round damages:

```text
round 2: mean=1.152342
round 3: mean=0.547163
round 0: mean=0.473645
round 4: mean=0.375267
round 5: mean=0.214352
```

Top round-edge damages:

```text
round 2, edge 3: mean=0.514168
round 2, edge 5: mean=0.418525
round 4, edge 5: mean=0.412462
round 2, edge 2: mean=0.373126
round 3, edge 5: mean=0.348199
```

The signature file:

```text
qpu_projection_signature_raw_damage.npz
```

This was the QPU-side finalizer.

The headline:

```text
QPU-side T_S projection is a raw-damage edge/round scaffold signature.
Strongest load is around edge 5, round 2, and round 2 x edge 3 / edge 5.
```

---

## Part 12 — Enhanced CUDA geo kernel

After Probe 06, the next step was to optimize the raw route path.

The requirement:

```text
one enhanced ts_geo_kernel.cu
do not create a pile of kernel files
preserve old entry point
add route/damage support underneath
```

The kernel file:

```text
kernels/ts_geo_kernel.cu
```

Compatibility entry point:

```text
ts_raw_geo_monotonic_kernel
```

New enhanced entry points:

```text
ts_raw_geo_route_vector_kernel
ts_raw_geo_damage_kernel
ts_raw_geo_packed_damage_kernel
ts_raw_geo_damage_mean_by_k_kernel
```

Methodology locked into the kernel:

```text
raw only
no normalization
no cosine
no classifier logic
no GPU base assumptions
no projector assumptions inside the kernel
```

Route vector layout:

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

Damage vector layout:

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

This made the route evaluator an optimized arithmetic component, not a new scientific claim.

---

## Part 13 — Probe 07: optimized Probe 06

Probe 07 was created as:

```text
T_S PROBE 07 — OPTIMIZED QPU RAW-DAMAGE PROJECTION SIGNATURE
```

It was an optimized version of Probe 06.

It used:

```text
ts_raw_geo_route_vector_kernel
```

and kept the same raw-damage method.

The first run hit a bug:

```text
ValueError: could not broadcast input array from shape (6,7) into shape (6,)
```

Cause:

```text
shuffle_rounds() assigned f[d, sh, rng.permutation(f.shape[2])] into f[d, sh, :, e]
without selecting the same edge e on the right side.
```

Fix:

```python
def shuffle_rounds(block, rng):
    f = block.copy()
    for d in range(f.shape[0]):
        for sh in range(f.shape[1]):
            for e in range(f.shape[3]):
                perm = rng.permutation(f.shape[2])
                f[d, sh, :, e] = f[d, sh, perm, e]
    return f
```

After the fix, Probe 07 matched Probe 06 on the QPU file.

Important speed result:

```text
total ablations       = 567
route seconds         ~= 0.012 to 0.017 seconds
route ablations/sec   ~= 33k to 45k depending run
```

The conclusion:

```text
CUDA route evaluation is correct and fast.
The bottleneck is now Python-side ablation construction and stress-batch construction.
```

Probe 07 finalized the optimized route path.

---

## Part 14 — GPU generator: same schema as QPU dump

Next, the package needed:

```text
t_s_gpu_generate.py
```

The requirement:

```text
The projector should be able to use each file interchangeably.
I can rename it to a QPU job and run it through our probes with the same results,
generated on GPU.
```

The generator followed the S_M GPU generator pattern.

It writes the same analysis-facing schema:

```text
field[mode, delay_site, delay_value, shot, round, edge]
final[mode, delay_site, delay_value, shot, channel]
```

Default shape:

```text
modes       = clean, phase_shear, local_shock
delay_sites = pre_coupling, post_coupling, post_perturb
delays      = 0, 1, 2, 4, 8, 16
shots       = 4096
rounds      = 6
channels    = 8
edges       = 7
```

Generated outputs:

```text
data/ts_gpu_data_<TAG>.npz
data/ts_gpu_job_<TAG>.json
data/latest_ts_gpu_data.json
data/latest_ts_data.json
```

The important schema achievement:

```text
GPU-generated file can run through the same probes as the QPU file.
```

This completed the gproj substrate path.

---

## Part 15 — Canceling the standalone non-data probe

At this stage, the standalone Probe 05 that did not use the T_S base data was canceled.

Reason:

```text
we are not benchmarking a toy object
we are benchmarking the T_S field data
```

The useful benchmark path became:

```text
generate qproj base
generate gproj base
run probes/benchmark on the same schema
```

This cleanup was important.

The final package should not claim evidence from a standalone probe that bypasses the actual field records.

---

## Part 16 — First GPU generator probe-compatible run

The GPU-generated file successfully ran through the existing probe stack:

```text
t_s_probe1.py
t_s_probe2_geo.py
t_s_probe3_raw_first_jumpgeo.py
t_s_probe5_qpu_geo_origin.py
t_s_probe6_qpu_projection_raw_damage.py
t_s_probe7_qpu_projection_raw_damage_cuda.py
```

That confirmed the generator requirement:

```text
GPU and QPU files are analysis-schema compatible.
```

The first GPU file showed strong raw stability:

```text
raw selected    = 36 / 36 = 100%
transform jumps = 0
```

Probe 06 / Probe 07 matched on the GPU file too.

This validated:

```text
optimized CUDA path is faithful on both QPU and GPU generated substrates
```

The first mismatch:

```text
QPU strongest edge = edge 5
GPU strongest edges = edge 1 / edge 3
```

But the main round structure matched:

```text
round 2
```

and the top round-edge cell included:

```text
round 2, edge 3
```

Interpretation:

```text
GPU generator v1 is schema-compatible and structurally aligned,
but edge localization is not yet QPU-matched.
```

---

## Part 17 — GPU generator tuning

A small generator tuning pass was suggested and then baked into defaults.

PowerShell run:

```powershell
python ghost_oracle/T_S/t_s_gpu_generate.py --verify `
  --edge-scaffold 0.095 `
  --round-scaffold 0.060 `
  --edge-memory 0.62 `
  --round-memory 0.70 `
  --local-shock 0.115
```

Defaults were updated from:

```text
DEFAULT_P_BASE = 0.165
DEFAULT_P_READOUT = 0.018
DEFAULT_EDGE_MEMORY = 0.78
DEFAULT_ROUND_MEMORY = 0.68
DEFAULT_DELAY_GAIN = 0.020
DEFAULT_SITE_GAIN = 0.010
DEFAULT_PHASE_SHEAR = 0.045
DEFAULT_LOCAL_SHOCK = 0.105
DEFAULT_EDGE_SCAFFOLD = 0.065
DEFAULT_ROUND_SCAFFOLD = 0.055
DEFAULT_BURST_PROB = 0.000
DEFAULT_BURST_WIDTH = 2
```

to:

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

The tuned generator produced:

```text
raw selected    = 35 / 36 = 97.22%
transform jumps = 1 / 36 = 2.78%
selected        = {'raw': 35, 'pow_exp_4': 1}
```

This was accepted as a good substrate:

```text
still raw-first
not too clean
a little stress behavior
```

The edge mismatch remained a known tuning target.

---

## Part 18 — Final benchmark design

The final benchmark requirement:

```text
call geo, qproj, and gproj
benchmark them against a common task
compare to best-in-class adjacent methods on the same task
```

The chosen common task:

```text
route-preserving edge/round scaffold recovery
```

The benchmark file:

```text
t_s_benchmark.py
```

The benchmark should compare:

```text
qproj = QPU-generated T_S file
gproj = GPU-generated T_S file
geo   = raw route/stress arithmetic path using ts_geo_kernel.cu
```

and should include classical/adjacent baselines:

```text
geo_cuda
geo_cpu_dp
scipy_dijkstra
networkx_dijkstra, optional
scalar_rate
field_profile_l1
stress_profile_l1
```

This separated:

```text
substrates:
  qproj, gproj

methods:
  geo_cuda, geo_cpu_dp, scipy_dijkstra, scalar/profile baselines
```

That separation was not present in the first benchmark draft and had to be corrected.

---

## Part 19 — Benchmark v1: multi-file scan and CSV bug

The first benchmark version defaulted to latest pointers, then was corrected to scan every valid `.npz` in:

```text
T_S/data/
```

This was requested because the final benchmark should not only run the latest file.

New behavior:

```text
scan data/
run every valid T_S .npz
skip non-T_S files safely
```

It also got a CLI override:

```bash
python t_s_benchmark.py --files file1.npz file2.npz
```

The first scan version hit a CSV bug:

```text
ValueError: dict contains fields not in fieldnames: 'round_index'
```

Cause:

```text
CSV rows were heterogeneous.
Edge rows, round rows, round-edge rows, and coarse rows do not share identical keys.
```

Fix:

```text
write_csv now unions keys across all row dictionaries
uses extrasaction="ignore"
```

This belongs in the record because it reflects the nature of the benchmark rows:

```text
the benchmark intentionally mixes multiple ablation row types
```

---

## Part 20 — Benchmark v2/v3: adding geo and classical baselines

The next benchmark run revealed a conceptual problem:

```text
there is no geo path
we are not comparing to the classical benchmarks we talked about
it just ran stuff on the two projections
```

The benchmark was patched to include explicit method rows:

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

The key correction:

```text
qproj/gproj = substrate sources
geo/classical/profile = methods
```

This produced the correct final shape.

---

## Part 21 — Benchmark v4: timing cleanup and final output

The benchmark then received final touchups:

```text
geo_cuda timing reports actual CUDA route-kernel time
scalar/profile baselines report timing instead of nan
QPU/GPU label changed to QPROJ/GPROJ in printed output
method table prints vs_scipy speed ratio
method print order cleaned up
```

Representative final benchmark output:

```text
Files found:
  qproj: ts_data_d8e9ab3o3njc73eue47g
  gproj: ts_gpu_data_4096shots_seed4204302182560473160

Classical:
  scipy_dijkstra=True
  networkx_dijkstra=False

Methods:
  geo_cuda
  geo_cpu_dp
  scipy_dijkstra
  scalar_rate
  field_profile_l1
  stress_profile_l1
```

QPROJ scaffold:

```text
Top edges:
  edge 5
  edge 2
  edge 3

Top rounds:
  round 2
  round 3
  round 0

Top round-edge:
  round 2, edge 3
  round 2, edge 5
  round 4, edge 5
```

GPROJ scaffold:

```text
Top edges:
  edge 1
  edge 3
  edge 4

Top rounds:
  round 2
  round 3
  round 4

Top round-edge:
  round 2, edge 3
  round 1, edge 0
  round 2, edge 4
```

QPROJ/GPROJ alignment:

```text
edge:
  top1     = 0.000
  Spearman = 0.3571
  top3     = 0.333
  top5     = 1.000

round:
  top1     = 1.000
  Spearman = 0.6000
  top3     = 0.667
  top5     = 0.800

round_edge:
  top1     = 1.000
  Spearman = 0.1871
  top3     = 0.333
  top5     = 0.600

coarse:
  Spearman = 0.9762
```

Method baseline result:

```text
geo_cuda:
  round_edge_top1_mean     = 1.000
  round_edge_spearman_mean = 1.0000
  vs_scipy                 = 9.78x

geo_cpu_dp:
  round_edge_top1_mean     = 1.000
  round_edge_spearman_mean = 1.0000
  vs_scipy                 = 1.92x

scipy_dijkstra:
  round_edge_top1_mean     = 1.000
  round_edge_spearman_mean = 1.0000
  vs_scipy                 = 1.00x

stress_profile_l1:
  round_edge_top1_mean     = 0.278
  round_edge_spearman_mean = 0.3196

field_profile_l1:
  round_edge_top1_mean     = 0.056
  round_edge_spearman_mean = 0.0260

scalar_rate:
  round_edge_top1_mean     = 0.056
  round_edge_spearman_mean = 0.0256
```

Final read:

```text
qproj/gproj = substrate comparison
geo_cuda/geo_cpu_dp/scipy_dijkstra/profiles = method comparison
```

This became the current canonical benchmark.

---

## Part 22 — Current final benchmark interpretation

The benchmark now tells a clean story.

Substrate result:

```text
QPROJ and GPROJ agree on the main round scaffold.
QPROJ and GPROJ agree on the top round-edge cell.
QPROJ and GPROJ strongly agree on coarse destructive-control ranking.
QPROJ has sharper edge-5 localization than the current GPROJ generator.
```

Method result:

```text
geo_cuda, geo_cpu_dp, and scipy_dijkstra agree on the route-optimal scaffold.
geo_cuda is fastest on the current structured-grid workload.
generic scalar/profile baselines do not recover the same round-edge scaffold.
```

This supports the bounded claim:

```text
T_S is reading stress-derived route structure,
not merely scalar field density or simple profile distance.
```

It does not support:

```text
quantum advantage
QPU speedup
universal shortest path
QPU simulation by GPU generator
```

The QPU path supplies the hardware field substrate.

The CUDA path supplies fast route evaluation.

Those are separate claims.

---

## Part 23 — Documentation cleanup

After the final benchmark stabilized, documentation was written in the same package style as S_M.

Current docs created:

```text
docs/math.md
docs/architecture.md
README.md
PROCESS_RECORD.md
```

The math document records:

```text
temporal edge field
final readout
delay / round / edge gradients
stress tensor components
raw geo movement costs
route vector
stress avoidance proxy
raw-damage signature
edge / round / round-edge ablations
coarse destructive controls
geo / qproj / gproj paths
benchmark task
classical baseline boundary
CUDA route evaluation
GPU generator defaults
bounded claims
```

The architecture document records:

```text
T_S package status
Converger framing
operator package pattern
substrate paths
base schema
temporal edge field
stress tensor
raw geo route
field controls
CUDA architecture
GPU generator
QPU generator
canonical data flow
benchmark stages
valid claim boundary
file responsibilities
```

The README records:

```text
quick path
operator definition
repo structure
data files
probe path
current capstone benchmark
current benchmark result
CUDA route evaluation
field controls
QPU base workflow
GPU bases
files produced
script map
what to look for
bounded claim
next development steps
```

The current process record documents the messy path that produced those clean docs.

---

## Part 24 — Current T_S repo layout

Current T_S package shape:

```text
T_S/
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
│   ├── known_issues.md
│   └── math.md
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

Current main entry points:

```bash
python t_s_benchmark.py
python t_s_gpu_generate.py --verify
python t_s_qpu_generate.py submit
python t_s_qpu_generate.py dump <JOB_ID>
```

Useful benchmark variants:

```bash
python t_s_benchmark.py --files data/file1.npz data/file2.npz
python t_s_benchmark.py --qpu-only
python t_s_benchmark.py --gpu-only
python t_s_benchmark.py --cpu-only
python t_s_benchmark.py --include-networkx
```

---

## Part 25 — Current known issues

### GPU generator edge localization does not fully match QPU

Current QPU top edge:

```text
edge 5
```

Current GPU top edges:

```text
edge 1
edge 3
edge 4
```

But both match on:

```text
round 2
round 2, edge 3
```

and coarse destructive-control ranking is strongly aligned.

Current interpretation:

```text
GPROJ captures the main round and round-edge scaffold.
QPROJ has sharper edge-5 localization.
```

This is a generator-tuning target, not a failure of the operator.

---

### Route evaluation is no longer the bottleneck

Probe 07 and the final benchmark show:

```text
route seconds are tiny
total seconds are dominated by Python-side ablation and stress construction
```

Future optimization target:

```text
move more ablation/stress construction into CUDA
```

---

### qpu_gpu_alignment.csv filename is legacy

The printed benchmark section now says:

```text
QPROJ/GPROJ ALIGNMENT
```

but the output file is still:

```text
qpu_gpu_alignment.csv
```

Suggested future rename:

```text
qproj_gproj_alignment.csv
```

This is cosmetic but would clean up terminology.

---

### NetworkX baseline is optional for a reason

NetworkX Dijkstra is useful as a sanity check but slow.

Default benchmark includes SciPy Dijkstra as the classical graph-route baseline.

Use:

```bash
python t_s_benchmark.py --include-networkx
```

only when a slower sanity baseline is desired.

---

### Scalar/profile baselines are not route baselines

The scalar/profile baselines are included to test whether T_S is merely measuring density or simple profile distance.

They are not expected to match the route scaffold.

Current result shows they mostly do not.

This is useful evidence, but should not be framed as a universal failure of all profile methods.

---

### The GPU generator is not a QPU simulator

Correct framing:

```text
GPU generator = controlled generated T_S substrate with QPU-compatible schema
```

Incorrect framing:

```text
GPU generator = QPU simulator
```

The current gproj is useful because it can run through the same probes and benchmark.

It does not claim to reproduce every QPU hardware detail.

---

### T_S is not a QPU speed path

The QPU path supplies real field data.

The CUDA path evaluates the route.

The final benchmark speedup is:

```text
geo_cuda faster than scipy_dijkstra on the structured-grid route task
```

not:

```text
QPU faster than classical
```

---

## Part 26 — Open questions

### 1. More QPU jobs

The current clean process uses one primary QPU base:

```text
ts_data_d8e9ab3o3njc73eue47g
```

Next validation:

```text
run multiple QPU jobs
compare qproj edge/round/round-edge scaffold stability
measure job-to-job variation
```

---

### 2. More backend runs

The current QPU result should be tested across more hardware conditions.

Questions:

```text
Does edge 5 remain dominant?
Does round 2 remain dominant?
Does round 2, edge 3 remain top round-edge?
Does coarse ranking stay stable?
```

---

### 3. Better GPU edge localization

The current GPU generator matches round and round-edge structure but not the edge-5 top1.

Future tuning target:

```text
increase qproj-like edge-5 localization
without breaking round 2 and round 2, edge 3
```

---

### 4. CUDA ablation/stress construction

The route kernel is already fast.

Future work:

```text
GPU-side ablation generation
GPU-side gradient/stress construction
packed real-vs-ablated damage kernel usage
```

This would turn the benchmark from Python-heavy into GPU-heavy.

---

### 5. Rename qpu_gpu_alignment.csv

Recommended cleanup:

```text
qpu_gpu_alignment.csv -> qproj_gproj_alignment.csv
```

The old name is understandable, but qproj/gproj is more consistent with the operator package language.

---

### 6. Known issues document

Create or update:

```text
docs/known_issues.md
```

with:

```text
edge localization mismatch
route no longer bottleneck
legacy filename naming
NetworkX optionality
GPU generator non-simulator boundary
QPU speed non-claim
```

---

### 7. Fixture policy

The package needs a clear fixture policy:

```text
small curated fixtures may be committed
large generated .npz bases should stay out of git unless intentionally shipped
latest_*.json should not be relied on for published reproducibility
```

---

## Part 27 — Current working philosophy

The T_S path repeated the larger Ghost Oracle lesson.

The useful result is not:

```text
we found one magic scalar
```

The useful result is:

```text
we separated the field object, stress tensor, route operator, controls,
substrates, and benchmark claim
```

The standard remains:

```text
If the result is raw, keep it raw.
If normalization is not needed, do not add it.
If a QPU field carries structure, locate the structure before projecting it.
If a GPU generator matches the schema, run it through the same probes.
If a standalone probe does not use the real data, do not make it the claim.
If geo is an arithmetic route path, do not confuse it with qproj/gproj.
If scalar/profile baselines fail, say exactly what they fail to recover.
If SciPy Dijkstra agrees, use it as correctness baseline.
If CUDA is faster, claim only the structured-grid speed result.
If the QPU supplies data, do not call it a speedup.
```

Current T_S final read:

```text
Phi[delay, shot, round, edge] is the temporal edge field.
Delay, round, and edge gradients define the stress tensor.
Raw geo routes move through that stress tensor.
Raw damage measures which scaffold components are load-bearing.

QPROJ and GPROJ agree on the main round and round-edge scaffold.
QPROJ and GPROJ strongly agree on coarse destructive-control ranking.
QPROJ has sharper edge localization than GPROJ.
GEO CUDA matches CPU DP and SciPy Dijkstra on the route task.
GEO CUDA is faster on the current structured-grid workload.
Scalar/profile baselines do not recover the same scaffold.
```

The process is the process.

Build, break, fix, document, repeat.
