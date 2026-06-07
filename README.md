# Ghost Oracle Suite

A CC0 community project for **ghost-channel operators**: measurement and projection channels that expose structure not captured by a primary classical score alone.

The current suite is organized around completed operator packages. Each operator follows the same discipline:

```text
define the operator
generate a GPU/reference base
generate or dump a QPU/hardware base
benchmark geo / gproj / qproj under shared controls
make only bounded claims
```

The current completed packages are:

```text
G_M    Generalized Metric
S_M    Syndrome Metric
T_S    Temporal Stress Metric
F_M    Fractal / Frequency / Field Metric
D_M    Dimensional Entanglement Projection
```

The larger roadmap frames these operators as parts of a transformer-adjacent **Converger**: a system that does not replace transformers, humans, or existing tools, but measures structure around them.

The core method is:

```text
freeze the record
build matched controls
scramble the channel
compare substrates
measure what survives
```

A ghost channel is considered load-bearing only when destroying that channel destroys the effect while preserving the surrounding task.

---

## Current status

The repo now has five main completed operator packages:

```text
ghost_oracle/
├── G_M/
├── S_M/
├── T_S/
├── F_M/
└── D_M/
```

All completed packages use the same high-level structure:

```text
operator/
├── data/
├── docs/
├── examples/
├── kernels/
├── probes/
├── README.md
├── operator_benchmark.py
├── operator_gpu_generate.py
└── operator_qpu_generate.py
```

For the current repo:

```text
G_M/
├── g_m_benchmark.py
├── g_m_gpu_generate.py
└── g_m_qpu_generate.py

S_M/
├── s_m_benchmark.py
├── s_m_gpu_generate.py
└── s_m_qpu_generate.py

T_S/
├── t_s_benchmark.py
├── t_s_gpu_generate.py
└── t_s_qpu_generate.py

F_M/
├── F_M_final_benchmark.py
├── f_m_gpu_generate.py
└── f_m_qpu_generate.py

D_M/
├── d_m_benchmark.py
├── d_m_gpu_generate.py
└── d_m_qpu_generate.py
```

The active architecture is no longer a loose collection of probes. It is a repeatable ghost-channel benchmark platform.

---

## Operator stack

The long-term Converger roadmap contains seven ghost-channel operators:

| Operator | Name | Current status |
|---|---|---|
| `G_M` | Generalized Metric | Completed for this version. |
| `S_M` | Syndrome Metric | Completed for this version. |
| `T_S` | Temporal Stress Metric | Completed for this version. |
| `F_M` | Fractal / Frequency / Field Metric | Completed for this version. |
| `D_M` | Dimensional Entanglement Projection | Completed for this version. |
| `I_M.local` | Local Interaction channel | Future operator. |
| `I_M.field` | Field Interaction channel | Future operator. |

The current repo contains finished `G_M`, `S_M`, `T_S`, `F_M`, and `D_M` packages. The remaining channels are roadmap items unless and until they receive the same package structure, benchmark controls, and bounded claim discipline.

---

## Standard substrate paths

Each completed operator exposes the same three substrate paths:

```text
geo      synthetic / analytical / classical reference path
gproj    GPU-generated or noiseless projection/base path
qproj    QPU / hardware-derived base path
```

The rule is:

```text
same operator
same input schema
same controls
same benchmark metrics
three substrate paths
```

The benchmark runner is the only place where comparative claims should be made.

---

## Repository layout

```text
GHOST_ORACLE_SUITE/
├── ghost_oracle/
│   ├── G_M/
│   │   ├── data/
│   │   ├── docs/
│   │   │   ├── architecture.md
│   │   │   ├── known_issues.md
│   │   │   └── math.md
│   │   ├── examples/
│   │   ├── kernels/
│   │   ├── probes/
│   │   ├── README.md
│   │   ├── g_m_benchmark.py
│   │   ├── g_m_gpu_generate.py
│   │   └── g_m_qpu_generate.py
│   │
│   ├── S_M/
│   │   ├── data/
│   │   ├── docs/
│   │   │   ├── architecture.md
│   │   │   ├── known_issues.md
│   │   │   └── math.md
│   │   ├── examples/
│   │   ├── kernels/
│   │   │   └── sm_kernel.cu
│   │   ├── probes/
│   │   ├── README.md
│   │   ├── s_m_benchmark.py
│   │   ├── s_m_gpu_generate.py
│   │   └── s_m_qpu_generate.py
│   │
│   ├── T_S/
│   │   ├── data/
│   │   ├── docs/
│   │   │   ├── architecture.md
│   │   │   ├── known_issues.md
│   │   │   └── math.md
│   │   ├── kernels/
│   │   │   └── ts_geo_kernel.cu
│   │   ├── probes/
│   │   ├── README.md
│   │   ├── t_s_benchmark.py
│   │   ├── t_s_gpu_generate.py
│   │   └── t_s_qpu_generate.py
│   │
│   ├── F_M/
│   │   ├── data/
│   │   │   ├── latest_fm_qpu_data.json
│   │   │   ├── latest_fm_gpu_data.json
│   │   │   ├── fm_job_<JOB_ID>.npz
│   │   │   └── fm_gpu_data_<...>.npz
│   │   ├── docs/
│   │   │   ├── architecture.md
│   │   │   ├── known_issues.md
│   │   │   └── math.md
│   │   ├── examples/
│   │   ├── kernels/
│   │   │   └── fm_projector_kernel.cu
│   │   ├── probes/
│   │   ├── README.md
│   │   ├── F_M_final_benchmark.py
│   │   ├── f_m_gpu_generate.py
│   │   └── f_m_qpu_generate.py
│   │
│   └── D_M/
│       ├── data/
│       ├── docs/
│       │   ├── architecture.md
│       │   └── math.md
│       ├── examples/
│       ├── kernels/
│       │   └── dm_projector_kernel.cu
│       ├── probes/
│       │   ├── analysis/
│       │   ├── 00_dm_probe_prune.py
│       │   ├── ...
│       │   ├── 25_dm_probe_geo_precision_reference.py
│       │   └── D_M_probe_process_record.md
│       ├── README.md
│       ├── PROCESS_RECORD.md
│       ├── d_m_benchmark.py
│       ├── d_m_gpu_generate.py
│       └── d_m_qpu_generate.py
│
├── CONTRIBUTING.md
├── LICENSE
├── PROCESS_RECORD.md
├── README.md
└── requirements.txt
```

---

## Quick start

```bash
git clone <repo-url> Ghost_Oracle_Suite
cd Ghost_Oracle_Suite
pip install -r requirements.txt
```

Run the current `G_M` benchmark:

```bash
python ghost_oracle/G_M/g_m_benchmark.py
```

Run the current `S_M` benchmark:

```bash
python ghost_oracle/S_M/s_m_benchmark.py
```

Run the current `T_S` benchmark:

```bash
python ghost_oracle/T_S/t_s_benchmark.py
```

Run the current `F_M` benchmark:

```bash
python ghost_oracle/F_M/F_M_final_benchmark.py
```

Run the current `D_M` benchmark:

```bash
python ghost_oracle/D_M/d_m_benchmark.py
```

Run full benchmark / probe modes where available:

```bash
python ghost_oracle/G_M/g_m_benchmark.py --sweep ALL --probe
python ghost_oracle/S_M/s_m_benchmark.py --sweep ALL --probe
python ghost_oracle/T_S/t_s_benchmark.py
python ghost_oracle/F_M/F_M_final_benchmark.py --geo-profile wide --max-candidates 1000000 --reps 100
python ghost_oracle/D_M/d_m_benchmark.py --repair-metadata
```

---

## `G_M` — Generalized Metric

`G_M` is the original Ghost Oracle operator family: **Generalized Metric**, formerly **Ghost Metric**.

It began as a failed Hadamard-test interpretation. The QPU circuit was first assumed to compute a textbook overlap target, but the probes showed that assumption was wrong. The useful step was to stop asking why the circuit failed to compute the intended object and instead ask what it was consistently computing.

The resulting closed-form operator is:

```text
G_M(a, b) = sqrt((1 + cos(a) cos(b)) / 2) / alpha
```

Current framing:

```text
G_M = bounded projection-channel / geometry-channel generalized similarity operator
```

`G_M` is implemented across three substrates:

```text
geo      closed-form geometry channel
gproj    GPU/noiseless projection base
qproj    real QPU shot-count projection base
```

The core claim is not that QPU projection is faster than classical GPU attention.

The core claim is that the same generalized projection-style operator can be expressed across mathematical, classical-sampler, and physical-shot substrates, with an agreement metric that exposes substrate quality.

### G_M package

```text
ghost_oracle/G_M/
├── README.md
├── g_m_benchmark.py
├── g_m_gpu_generate.py
├── g_m_qpu_generate.py
├── data/
├── docs/
├── examples/
├── kernels/
└── probes/
```

### G_M quick path

```bash
python ghost_oracle/G_M/g_m_benchmark.py
python ghost_oracle/G_M/g_m_benchmark.py --sweep ALL
python ghost_oracle/G_M/g_m_benchmark.py --probe
```

Base generation:

```bash
python ghost_oracle/G_M/g_m_gpu_generate.py
python ghost_oracle/G_M/g_m_qpu_generate.py
```

### G_M benchmark claim

The current bounded claim is:

```text
G_M is a bounded, calibrated, substrate-comparable generalized metric.
```

The benchmark evidence supports:

```text
1. The closed-form geometry channel retrieves under coherent same-dimension attack.
2. Real QPU projection bases retrieve when calibrated.
3. Noiseless GPU projection bases retrieve when calibrated.
4. Destroying calibrated bucket structure destroys projection retrieval.
5. The projection path is therefore using load-bearing shot-count structure.
6. cuBLAS remains the correct dense GEMM throughput control.
```

The non-claims are:

```text
G_M is not a universal replacement for dot-product attention.
G_M is not claimed to make QPU shot reconstruction faster than cuBLAS.
G_M is not a quantum-advantage claim.
G_M is not distribution-shift-proof without recalibration.
```

See:

```text
ghost_oracle/G_M/README.md
ghost_oracle/G_M/docs/math.md
ghost_oracle/G_M/docs/architecture.md
ghost_oracle/G_M/docs/known_issues.md
```

---

## `S_M` — Syndrome Metric

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

```text
geo      synthetic/reference syndrome-spacetime field
gproj    GPU-generated syndrome-spacetime base
qproj    real QPU syndrome-spacetime data from IBM Runtime
```

The core claim is not that `S_M` is a logical-error-rate benchmark.

The core claim is that final data edge parity and repeated syndrome records form a load-bearing field structure that can be measured, scrambled, classified, and compared across synthetic, GPU-generated, and physical QPU-derived records.

### S_M package

```text
ghost_oracle/S_M/
├── README.md
├── s_m_benchmark.py
├── s_m_gpu_generate.py
├── s_m_qpu_generate.py
├── data/
├── docs/
├── examples/
├── kernels/
│   └── sm_kernel.cu
└── probes/
```

### S_M quick path

```bash
python ghost_oracle/S_M/s_m_benchmark.py
python ghost_oracle/S_M/s_m_benchmark.py --sweep ALL
python ghost_oracle/S_M/s_m_benchmark.py --probe
```

Base generation:

```bash
python ghost_oracle/S_M/s_m_gpu_generate.py
python ghost_oracle/S_M/s_m_qpu_generate.py submit
python ghost_oracle/S_M/s_m_qpu_generate.py dump <JOB_ID>
```

Useful CUDA options:

```bash
python ghost_oracle/S_M/s_m_benchmark.py --cuda-debug
python ghost_oracle/S_M/s_m_benchmark.py --no-cuda
```

### S_M current benchmark result

Current benchmark configuration:

```text
windows      = [8, 16, 32, 64]
distances    = d3, d5, d7, d9
rounds       = 10
shots        = 4096 per distance/base
substates    = GEO, GPROJ, QPROJ
CUDA kernel  = yes
```

Current key result:

```text
QPROJ real-vs-control:
  sm_all               = 0.999 balanced accuracy
  sm_field             = 0.998 balanced accuracy
  agreement_profiles   = 0.990 balanced accuracy

QPROJ control-source:
  sm_field             = 0.853 balanced accuracy
  sm_all               = 0.848 balanced accuracy

Distance prediction:
  GEO / GPROJ / QPROJ  = 1.000 balanced accuracy
```

The important S_M operator signature is:

```text
raw_rates / detection_rates stay near chance
agreement_profiles / sm_field / sm_all go near-perfect
```

That split is the evidence that `S_M` is reading field structure rather than only scalar syndrome density.

### S_M benchmark claim

The current bounded claim is:

```text
S_M is a field-structured, control-tested, substrate-comparable syndrome metric.
```

The benchmark evidence supports:

```text
1. The benchmark loads geo, gproj, and qproj S_M records under one shared task harness.
2. Field-aware features separate real QPU records from destructive controls.
3. Raw scalar-like rates remain near chance for real-vs-control separation.
4. Agreement and full field features approach near-perfect real-vs-control separation.
5. Control-source classification rises well above chance.
6. Distance prediction is stable across geo, gproj, and qproj records.
7. The CUDA kernel accelerates S_M feature extraction while preserving the same operator boundary.
```

The non-claims are:

```text
S_M is not a logical-error-rate benchmark.
S_M is not the T_S stress tensor.
S_M is not a token retrieval benchmark.
S_M is not a universal hardware advantage claim.
```

See:

```text
ghost_oracle/S_M/README.md
ghost_oracle/S_M/docs/math.md
ghost_oracle/S_M/docs/architecture.md
ghost_oracle/S_M/docs/known_issues.md
```

---

## `T_S` — Temporal Stress Metric

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

```text
geo      raw arithmetic route path derived from a T_S field
gproj    GPU-generated temporal-stress base
qproj    real QPU temporal-stress data from IBM Runtime
```

The core claim is not that `T_S` proves quantum advantage.

The core claim is that QPU-derived and GPU-generated temporal fields can be represented under one schema, converted into stress tensors, routed with the raw geo operator, destructively scrambled, and compared against classical route/profile baselines under one benchmark harness.

### T_S package

```text
ghost_oracle/T_S/
├── README.md
├── t_s_benchmark.py
├── t_s_gpu_generate.py
├── t_s_qpu_generate.py
├── data/
├── docs/
├── kernels/
│   └── ts_geo_kernel.cu
└── probes/
```

### T_S quick path

```bash
python ghost_oracle/T_S/t_s_benchmark.py
```

Base generation:

```bash
python ghost_oracle/T_S/t_s_gpu_generate.py --verify
python ghost_oracle/T_S/t_s_qpu_generate.py submit
python ghost_oracle/T_S/t_s_qpu_generate.py dump <JOB_ID>
```

Useful benchmark options:

```bash
python ghost_oracle/T_S/t_s_benchmark.py --files data/file1.npz data/file2.npz
python ghost_oracle/T_S/t_s_benchmark.py --qpu-only
python ghost_oracle/T_S/t_s_benchmark.py --gpu-only
python ghost_oracle/T_S/t_s_benchmark.py --cpu-only
python ghost_oracle/T_S/t_s_benchmark.py --include-networkx
```

### T_S current benchmark result

Current representative benchmark configuration:

```text
substates    = QPROJ, GPROJ
geo methods  = geo_cuda, geo_cpu_dp, scipy_dijkstra
profile refs = scalar_rate, field_profile_l1, stress_profile_l1
shots        = 4096 per mode/site/delay/base
rounds       = 6
channels     = 8
delays       = [0, 1, 2, 4, 8, 16] dt
CUDA kernel  = yes
GPU          = RTX 3090 in current run
```

Current QPROJ scaffold:

```text
top edge        = edge 5
top round       = round 2
top round-edge  = round 2, edge 3
```

Current GPROJ scaffold:

```text
top round       = round 2
top round-edge  = round 2, edge 3
```

Current QPROJ/GPROJ alignment:

```text
edge:
  top1     = 0.000
  top5     = 1.000
  Spearman ≈ 0.357

round:
  top1     = 1.000
  Spearman ≈ 0.600

round_edge:
  top1     = 1.000
  top5     = 0.600

coarse controls:
  Spearman ≈ 0.976
```

Current method comparison:

```text
geo_cuda:
  round_edge_top1_mean     = 1.000
  round_edge_spearman_mean = 1.0000
  faster than scipy_dijkstra on the current structured-grid workload

geo_cpu_dp:
  round_edge_top1_mean     = 1.000
  round_edge_spearman_mean = 1.0000

scipy_dijkstra:
  round_edge_top1_mean     = 1.000
  round_edge_spearman_mean = 1.0000

stress_profile_l1:
  partial scaffold signal

field_profile_l1 / scalar_rate:
  weak scaffold signal
```

The important T_S operator signature is:

```text
geo_cuda / geo_cpu_dp / scipy_dijkstra agree on the route-optimal scaffold
scalar_rate / field_profile_l1 / stress_profile_l1 do not recover the same scaffold
qproj and gproj agree on the main round and round-edge structure
qproj has sharper edge localization than the current gproj generator
```

That split is evidence that `T_S` is reading stress-derived route structure rather than only scalar field density.

### T_S benchmark claim

The current bounded claim is:

```text
T_S is a field-structured, stress-derived, route-tested temporal metric.
```

The benchmark evidence supports:

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

The non-claims are:

```text
T_S is not a quantum advantage claim.
T_S is not a QPU speedup claim.
T_S is not a universal shortest-path algorithm.
T_S is not a QPU simulator.
T_S does not fully match qproj edge localization with gproj yet.
```

See:

```text
ghost_oracle/T_S/README.md
ghost_oracle/T_S/docs/math.md
ghost_oracle/T_S/docs/architecture.md
ghost_oracle/T_S/docs/known_issues.md
```

---

## `F_M` — Fractal / Frequency / Field Metric

`F_M` is the Ghost Oracle Suite paired-path differential wave operator family: **Fractal / Frequency / Field Metric**.

`F_M` treats paired delayed channel measurements not as separate raw path outputs, but as a differential wave field. The useful object is the relationship between the `g` path, the `em` path, their differential fields, and the delay-ordered response curve that survives destructive controls.

Current framing:

```text
F_M = substrate-linked paired-path differential wave operator
```

The core fields are:

```text
delta     = em - g
xor_delta = em XOR g
```

The primary locked signature is:

```text
xor_delta / bit_diff / delay
```

where:

```text
bit_diff = mean(bit1) - mean(bit0)
```

`F_M` is implemented across three substrates:

```text
geo      optimized analytic metadata-to-curve path
gproj    GPU-generated paired-path base
qproj    real QPU paired-path data from IBM Runtime
```

The core claim is not that `F_M` proves literal gravity/electromagnetic cavities in hardware.

The core claim is that a paired-delay QPU circuit produced a stable differential wave signature, that a compatible GPU base can reproduce the signature family, and that an optimized GEO path can compute the signature directly.

### F_M package

```text
ghost_oracle/F_M/
├── README.md
├── F_M_final_benchmark.py
├── f_m_gpu_generate.py
├── f_m_qpu_generate.py
├── data/
├── docs/
├── examples/
├── kernels/
│   └── fm_projector_kernel.cu
└── probes/
```

### F_M quick path

```bash
python ghost_oracle/F_M/F_M_final_benchmark.py
```

Fast debugging pass:

```bash
python ghost_oracle/F_M/F_M_final_benchmark.py --skip-sweep --reps 20
```

Bigger capstone run:

```bash
python ghost_oracle/F_M/F_M_final_benchmark.py --geo-profile wide --max-candidates 1000000 --reps 100
```

Base generation:

```bash
python ghost_oracle/F_M/f_m_gpu_generate.py
python ghost_oracle/F_M/f_m_qpu_generate.py submit
python ghost_oracle/F_M/f_m_qpu_generate.py dump <JOB_ID>
```

Useful probe/finalizer paths:

```bash
python ghost_oracle/F_M/probes/f_m_probe04_qproj_kernel_finalizer.py --file ghost_oracle/F_M/data/fm_job_<JOB_ID>.npz
python ghost_oracle/F_M/probes/f_m_probe06_geo_cuda_finalizer.py --profile default --max-candidates 250000
```

### F_M current benchmark result

Current representative benchmark configuration:

```text
substates     = QPROJ, GPROJ, GEO
primary field = xor_delta
primary resp  = bit_diff
primary order = delay
tiles         = 7
shots         = 4096 for qproj/gproj bases
CUDA kernel   = yes
GPU           = RTX 3090 in current run
```

Current primary signature:

```text
QPROJ:
  xor_delta / bit_diff / delay
  score = 0.6571
  peak  = 0.769
  R2    = 0.819
  freq  = 1.30
  amp   = 0.05800

GPROJ:
  xor_delta / bit_diff / delay
  score = 0.6796
  peak  = 0.772
  R2    = 0.986
  freq  = 0.90
  amp   = 0.03837

GEO:
  xor_delta / bit_diff / delay
  score = 0.7356
  peak  = 0.812
  R2    = 0.988
  freq  = 1.10
  amp   = 0.04813
```

Current speed comparison:

```text
QPROJ response + metric : 1.045625 ms
GPROJ response + metric : 1.059980 ms
GEO curve + metric      : 0.382735 ms
GEO 250k sweep          : 290.061 ms
```

Adjacent classical readers on the primary curve:

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

The important F_M operator signature is:

```text
QPROJ discovers the paired-path differential wave signature.
GPROJ reproduces the signature family in a compatible generated base.
GEO computes the signature directly as an optimized analytic path.
Path-pair breaking weakens the signal.
Delay shuffling weakens the signal.
```

FFT and SinFit are strong readers of the already-built primary curve. They are not qproj/gproj/geo substrate paths.

### F_M benchmark claim

The current bounded claim is:

```text
F_M is a substrate-linked paired-path differential wave operator.
```

The benchmark evidence supports:

```text
1. The QPU record contains a stable xor_delta / bit_diff / delay signature.
2. The GPU-generated base reproduces the same signature family.
3. The GEO path computes the signature directly from metadata.
4. Path-pair structure is load-bearing.
5. Delay order is load-bearing.
6. GEO is faster than record-based qproj/gproj projector evaluation.
7. FFT and SinFit are useful adjacent readers of the primary curve, not full substrate paths.
```

The non-claims are:

```text
F_M is not a proof of literal hardware gravity/electromagnetic cavities.
F_M is not a quantum advantage claim.
F_M is not a universal FFT or sinusoid-fit replacement.
F_M does not claim qproj/gproj/geo are numerically identical.
F_M does not claim a universal frequency/amplitude across all future QPU jobs.
```

See:

```text
ghost_oracle/F_M/README.md
ghost_oracle/F_M/docs/math.md
ghost_oracle/F_M/docs/architecture.md
ghost_oracle/F_M/docs/known_issues.md
```

---

## `D_M` — Dimensional Entanglement Projection

`D_M` is the Ghost Oracle Suite dimensional witness-manifold operator family: **Dimensional Entanglement Projection**.

`D_M` began as a dimensional-compression experiment. That initial framing failed across the first six probes: D_M did not beat PCA or random projection baselines under fair compression tests. The useful pivot was to stop forcing the record into a dimensionality-reduction story and ask what the QPU listener record actually contained.

Current framing:

```text
D_M = dimensional Bell-witness manifold listener
```

The locked coordinate frame is:

```text
YZ-primary / ZY-reciprocal dimensional witness manifold
```

The core rung coordinates are:

```text
Y   = connected(YZ)
R   = -connected(ZY)
E   = sqrt(Y^2 + R^2)
S   = E - sqrt(XY^2 + YX^2)
phi = atan2(R, Y) mod pi
```

where:

```text
connected(PQ) = <P0 P1> - <P0><P1>
```

`D_M` is implemented across three substrates:

```text
geo      exact closed-form classical reference
gproj    GPU-generated controlled Bell-witness base
qproj    real QPU Bell-listener data from IBM Runtime
```

The core claim is not that `D_M` certifies Bell nonlocality.

The core claim is that a bare two-qubit QPU listener record contains a dimensional witness manifold; active delay/offset conditions separate from null; same-shot pairing, reciprocal structure, and delay order are load-bearing; and compound corruptions cross a measurable collapse boundary.

### D_M package

```text
ghost_oracle/D_M/
├── README.md
├── PROCESS_RECORD.md
├── d_m_benchmark.py
├── d_m_gpu_generate.py
├── d_m_qpu_generate.py
├── data/
├── docs/
│   ├── architecture.md
│   └── math.md
├── examples/
├── kernels/
│   └── dm_projector_kernel.cu
└── probes/
    ├── 00_dm_probe_prune.py
    ├── ...
    ├── 25_dm_probe_geo_precision_reference.py
    └── D_M_probe_process_record.md
```

### D_M quick path

```bash
python ghost_oracle/D_M/d_m_benchmark.py
```

Metadata repair run:

```bash
python ghost_oracle/D_M/d_m_benchmark.py --repair-metadata
```

Fast debugging pass:

```bash
python ghost_oracle/D_M/d_m_benchmark.py --reps 20
```

Base generation:

```bash
python ghost_oracle/D_M/d_m_gpu_generate.py
python ghost_oracle/D_M/d_m_qpu_generate.py submit
python ghost_oracle/D_M/d_m_qpu_generate.py dump <JOB_ID>
```

Useful probe paths:

```bash
python ghost_oracle/D_M/probes/23_dm_probe_dimensional_invariance_controls.py
python ghost_oracle/D_M/probes/24_dm_probe_corruption_boundary.py --auto --window 4096 --trials-per-depth 500 --save-trials
python ghost_oracle/D_M/probes/25_dm_probe_geo_precision_reference.py
```

### D_M current benchmark result

Current representative benchmark configuration:

```text
substates    = QPROJ, GPROJ, exact GEO
conditions   = null, base_only, offset_on
shots        = 4096 for qproj/gproj bases
rungs        = 5
CUDA kernel  = yes
GPU          = RTX 3090 in current run
```

Current final capstone projection summary:

```text
QPROJ:
  null       projection = 0.012762
  base_only  projection = 0.220018
  offset_on  projection = 0.208717

GPROJ:
  null       projection = 0.010326
  base_only  projection = 0.295522
  offset_on  projection = 0.298216

GEO:
  null       projection = 0.000000
  base_only  projection = 0.693827
  offset_on  projection = 0.650525
```

Current same-shot bit-shuffle control collapse:

```text
QPROJ base_only:
  0.220018 -> 0.083027
  drop = 62.26%

QPROJ offset_on:
  0.208717 -> 0.100915
  drop = 51.65%

GPROJ base_only:
  0.295522 -> 0.089097
  drop = 69.85%

GPROJ offset_on:
  0.298216 -> 0.055686
  drop = 81.33%
```

The important D_M operator signature is:

```text
QPROJ active conditions separate from QPROJ null.
GPROJ active conditions separate from GPROJ null.
GEO active conditions separate from exact GEO null.
Same-shot bit shuffling collapses active qproj/gproj records.
Allowed channel re-descriptions can preserve the manifold.
Single faults often repair.
Compound corruptions cross a collapse boundary around k=2 to k=3.
```

### D_M benchmark claim

The current bounded claim is:

```text
D_M is a dimensional witness-manifold projection operator.
```

The benchmark evidence supports:

```text
1. Active qproj/gproj conditions separate from null.
2. Exact GEO computes an active manifold and exact zero null reference.
3. Same-shot pairing is load-bearing.
4. Reciprocal structure is load-bearing.
5. Delay order is load-bearing.
6. Allowed channel re-descriptions preserve the dimensional manifold.
7. Single structural faults often repair.
8. Compound faults cross a measurable collapse boundary.
9. Retrieval/utility claims are not part of the default final claim.
```

The non-claims are:

```text
D_M does not certify Bell nonlocality.
D_M does not reconstruct density matrices.
D_M does not prove prepared Bell states.
D_M is not a QPU speedup or quantum-advantage claim.
GPROJ is not an IBM hardware simulator.
GEO is a closed-form reference, not a hardware simulator.
GPT-2 is not a D_M input.
D_M is not a dimensional-compression benchmark.
```

See:

```text
ghost_oracle/D_M/README.md
ghost_oracle/D_M/docs/math.md
ghost_oracle/D_M/docs/architecture.md
ghost_oracle/D_M/PROCESS_RECORD.md
ghost_oracle/D_M/probes/D_M_probe_process_record.md
```

---

## Current completed operators

| Package | Operator | Main benchmark | GPU path | QPU path | Kernel |
|---|---|---|---|---|---|
| `G_M/` | Generalized Metric | `g_m_benchmark.py` | `g_m_gpu_generate.py` | `g_m_qpu_generate.py` | `kernels/` |
| `S_M/` | Syndrome Metric | `s_m_benchmark.py` | `s_m_gpu_generate.py` | `s_m_qpu_generate.py` | `kernels/sm_kernel.cu` |
| `T_S/` | Temporal Stress Metric | `t_s_benchmark.py` | `t_s_gpu_generate.py` | `t_s_qpu_generate.py` | `kernels/ts_geo_kernel.cu` |
| `F_M/` | Fractal / Frequency / Field Metric | `F_M_final_benchmark.py` | `f_m_gpu_generate.py` | `f_m_qpu_generate.py` | `kernels/fm_projector_kernel.cu` |
| `D_M/` | Dimensional Entanglement Projection | `d_m_benchmark.py` | `d_m_gpu_generate.py` | `d_m_qpu_generate.py` | `kernels/dm_projector_kernel.cu` |

---

## Converger framing

The larger Ghost Oracle Suite architecture frames ghost-channel operators as components of a transformer-adjacent **Converger**.

A transformer answers:

```text
What is the next useful representation, token, action, or score?
```

A Converger asks:

```text
What hidden structure exists around that score, and does it survive controls?
```

The Converger is adjacent and independent:

```text
transformer predicts
converger measures
benchmark controls
operator survives or fails
```

Current operator mapping:

| Operator | Converger component | Role |
|---|---|---|
| `G_M` | metric projection component | Bounded similarity, retrieval structure, ranking behavior. |
| `S_M` | syndrome field component | Syndrome-spacetime fields and agreement structure. |
| `T_S` | temporal stress component | Delay/round/edge stress tensors and route scaffold damage. |
| `F_M` | differential wave component | Paired-path differential fields and delay-ordered waves. |
| `D_M` | dimensional witness component | Bell-witness manifolds, reciprocal channels, and corruption boundaries. |
| `I_M.local` | local interaction component | Future pointwise interaction channel. |
| `I_M.field` | field interaction component | Future nonlocal deformation channel. |

The goal is not one-off backend claims. The goal is a repeatable benchmark architecture where every operator gets:

```text
same package pattern
same substrate pattern
same control discipline
same bounded-claim standard
```

---

## Data

Each operator package has its own local `data/` directory.

```text
ghost_oracle/G_M/data/
ghost_oracle/S_M/data/
ghost_oracle/T_S/data/
ghost_oracle/F_M/data/
ghost_oracle/D_M/data/
```

Generated files are usually large and should not be committed unless intentionally shipped as small reproducibility fixtures.

Recommended ignore patterns:

```gitignore
ghost_oracle/G_M/data/job_*.npz
ghost_oracle/G_M/data/ghost_oracle_gpu_*.npz

ghost_oracle/S_M/data/sm_data_*.npz
ghost_oracle/S_M/data/sm_gpu_data_*.npz
ghost_oracle/S_M/data/sm_job_*.json
ghost_oracle/S_M/data/sm_gpu_job_*.json

ghost_oracle/T_S/data/ts_data_*.npz
ghost_oracle/T_S/data/ts_gpu_data_*.npz
ghost_oracle/T_S/data/ts_job_*.json
ghost_oracle/T_S/data/ts_gpu_job_*.json

ghost_oracle/F_M/data/fm_job_*.npz
ghost_oracle/F_M/data/fm_gpu_data_*.npz
ghost_oracle/F_M/data/latest_fm_*.json

ghost_oracle/D_M/data/dm_data_bell_listener_cavity_offset_*.npz
ghost_oracle/D_M/data/dm_gpu_data_*.npz
ghost_oracle/D_M/data/latest_dm_*.json

ghost_oracle/*/analysis/
ghost_oracle/*/probes/analysis/
*_report.json
```

Keep small curated fixtures if they are part of the reproducibility story.

Keep large generated bases out of git unless intentionally shipping them.

---

## Docs and process record

Each operator package has its own docs:

```text
ghost_oracle/G_M/docs/
ghost_oracle/S_M/docs/
ghost_oracle/T_S/docs/
ghost_oracle/F_M/docs/
ghost_oracle/D_M/docs/
```

Typical docs:

```text
architecture.md
math.md
known_issues.md
```

`D_M` additionally carries process records at:

```text
ghost_oracle/D_M/PROCESS_RECORD.md
ghost_oracle/D_M/probes/D_M_probe_process_record.md
```

Root-level project files:

```text
PROCESS_RECORD.md
CONTRIBUTING.md
LICENSE
requirements.txt
```

`PROCESS_RECORD.md` is the chronological research log.

The process record is intentionally honest: wrong turns stay in the repo, fixes live in follow-up probes, and claims get retracted when controls do not support them.

---

## Contributing

This is a CC0 community project.

The norm is:

```text
break it, fix it, document what happened
```

Useful contributions:

```text
new probes with controls
bug fixes
documentation improvements
benchmark sweeps
small reproducibility fixtures
clearer null models
backend comparison runs
CUDA/kernel cleanup
additional qproj/gproj/geo comparisons
```

A valid operator contribution should preserve the package discipline:

```text
same operator
same input schema
same base-file structure
same controls
same benchmark metrics
three substrate paths
bounded claims only
```

See:

```text
CONTRIBUTING.md
```

---

## License

CC0 1.0 Universal. Public domain dedication.

Use it for anything. Attribute if you want to. Do not if you do not.

---

## Citation

Not asking for one.

For all my fellow ghosts, may our silence speak your name.
