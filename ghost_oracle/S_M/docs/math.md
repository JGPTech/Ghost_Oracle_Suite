# Math

Mathematical definition of the `S_M` operator: final data edge parity, syndrome-spacetime fields, agreement fields, detection events, windowed feature families, destructive controls, substrate comparison, and the benchmark-supported claim boundary.

`S_M` is the **Syndrome Metric**.

The architecture document explains how this math gets compiled into the package and CUDA kernel. The probes record how the field framing evolved. This document is the math standing alone.

GitHub renders LaTeX inside `$...$` and `$$...$$` delimiters. Everything here is written as raw Markdown and should be pasted as source, not copied from a rendered preview.

Conventions used throughout:

* $d$ is the repetition-code distance.
* $i$ indexes a data qubit or edge location.
* $t$ indexes syndrome round / time.
* $N_{\mathrm{shots}}$ is the number of shots in a base, usually $4096$ by default.
* $R$ is the number of syndrome rounds, usually $10$ by default.
* A **record** is a pair of final data bits and repeated syndrome bits for one distance.
* A **base** is a `.npz` file containing records for one or more distances.
* A **substrate** is one implementation/source of the same S_M schema:

  * synthetic/reference field,
  * GPU-generated syndrome-spacetime base,
  * real QPU syndrome-spacetime base from IBM Runtime.

The shared S_M array schema is:

```text
data_d{d} : uint8, shape (shots, d)
synd_d{d} : uint8, shape (shots, rounds, d-1)
flag_d{d} : optional uint8, shape (shots, rounds, n_flags)
```

---

## The S_M object

The current operator is:

$$
S_M = {S[t,i], E[i], A[t,i]}
$$

where:

$$
D[i] \in {0,1}
$$

is the final measured data bit at code position $i$,

$$
S[t,i] \in {0,1}
$$

is the measured syndrome bit at round $t$ and edge $i$,

$$
E[i] = D[i] \oplus D[i+1]
$$

is the final data edge parity, and:

$$
A[t,i] = 1 - (S[t,i] \oplus E[i])
$$

is the agreement field.

In words:

```text
D[i]   = final data bit
E[i]   = final edge parity
S[t,i] = syndrome spacetime field
A[t,i] = agreement between syndrome and final edge parity
```

`S_M` is not a single scalar.

It is a field relation:

```text
final data edge parity
+
repeated syndrome spacetime
+
agreement structure
```

The benchmark then asks whether that structure survives controls.

---

## Final data edge parity

For a repetition-code record at distance $d$, the final data readout is:

$$
D = (D[0],D[1],\ldots,D[d-1])
$$

with:

$$
D[i]\in{0,1}
$$

The final edge parity field has length $d-1$:

$$
E = (E[0],E[1],\ldots,E[d-2])
$$

where:

$$
E[i] = D[i]\oplus D[i+1]
$$

The edge parity turns the final data record into the same edge-local coordinate system as the syndrome field.

That is the first important S_M step:

```text
data bits live on vertices
syndrome bits live on edges
edge parity maps final data into the syndrome coordinate system
```

Without this step, the final data record and the syndrome record are not directly aligned.

---

## Syndrome-spacetime field

The syndrome field is:

$$
S[t,i]
$$

with:

$$
t=0,\ldots,R-1
$$

and:

$$
i=0,\ldots,d-2
$$

So the syndrome tensor for one distance has shape:

$$
N_{\mathrm{shots}}\times R\times(d-1)
$$

For one shot $s$, write:

$$
S_s[t,i]
$$

and:

$$
D_s[i]
$$

Then:

$$
E_s[i] = D_s[i]\oplus D_s[i+1]
$$

and:

$$
A_s[t,i] = 1 - (S_s[t,i]\oplus E_s[i])
$$

The shot-indexed S_M object is therefore:

$$
S_{M,s}
=======

{S_s[t,i],E_s[i],A_s[t,i]}
$$

The windowed S_M object aggregates these quantities over a window of shots.

---

## Agreement field

The agreement field is the central S_M quantity.

For one shot:

$$
A_s[t,i] = 1 - (S_s[t,i]\oplus E_s[i])
$$

Equivalently:

$$
A_s[t,i]
========

\begin{cases}
1, & S_s[t,i]=E_s[i] \
0, & S_s[t,i]\ne E_s[i]
\end{cases}
$$

The agreement field asks:

```text
Does the repeated syndrome record agree with the final data edge parity?
```

This is why `S_M` is not just a syndrome-rate statistic.

A raw syndrome rate only asks:

$$
\mathbb{E}[S[t,i]]
$$

The S_M agreement field asks:

$$
\mathbb{E}[1-(S[t,i]\oplus E[i])]
$$

That couples final data and syndrome spacetime.

The current benchmark result supports this distinction: raw scalar-like features stay near chance for real-vs-control classification, while agreement and field-aware features approach near-perfect separation.

---

## Detection events

Detection events are temporal syndrome transitions.

For adjacent syndrome rounds:

$$
X[t,i] = S[t+1,i]\oplus S[t,i]
$$

where:

$$
t=0,\ldots,R-2
$$

The detection-event tensor has shape:

$$
N_{\mathrm{shots}}\times(R-1)\times(d-1)
$$

Detection events measure temporal changes in the syndrome field.

They are part of the S_M feature set, but they are not the central claim by themselves.

The benchmark treats detection rates as a baseline family:

```text
detection_rates = mean X[t,i]
```

The current key result is that detection rates alone remain near chance in the real-vs-control test, while agreement and field features separate strongly.

---

## Windowed aggregation

`S_M` is a windowed field operator.

A single shot contains a record, but the stable benchmark object is a window of shots.

Let a window $W$ contain $n_W$ shots:

$$
W={s_1,s_2,\ldots,s_{n_W}}
$$

Default benchmark windows are:

```text
8
16
32
64
```

Probe mode may use:

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

For each window, the benchmark computes rates and profiles such as:

$$
\bar{D}_W[i]
============

\frac{1}{n_W}
\sum_{s\in W}D_s[i]
$$

$$
\bar{S}_W[t,i]
==============

\frac{1}{n_W}
\sum_{s\in W}S_s[t,i]
$$

$$
\bar{A}_W[t,i]
==============

\frac{1}{n_W}
\sum_{s\in W}A_s[t,i]
$$

and:

$$
\bar{X}_W[t,i]
==============

\frac{1}{n_W}
\sum_{s\in W}X_s[t,i]
$$

Windowing matters because S_M measures field structure statistically.

The current strongest real-vs-control results occur at:

```text
window = 64
```

This should be read as the current empirical operating point, not a universal constant.

---

## Feature family 1: raw_rates

The raw-rate feature family is the scalar baseline.

For a window $W$:

$$
\mathrm{raw_data}_W[i]
======================

\frac{1}{n_W}
\sum_{s\in W}D_s[i]
$$

and:

$$
\mathrm{raw_synd}_W[t,i]
========================

\frac{1}{n_W}
\sum_{s\in W}S_s[t,i]
$$

The feature vector is:

$$
\mathrm{raw_rates}_W
====================

\left[
\mathrm{raw_data}_W,
\mathrm{vec}(\mathrm{raw_synd}_W)
\right]
$$

This feature family intentionally tests whether the benchmark is merely reading trivial marginal rates.

Current benchmark interpretation:

```text
raw_rates near chance in Task A is good evidence that the main result is not only scalar density.
```

---

## Feature family 2: detection_rates

Detection rates are windowed means of the temporal transition field:

$$
\mathrm{detection_rates}_W[t,i]
===============================

\frac{1}{n_W}
\sum_{s\in W}X_s[t,i]
$$

where:

$$
X_s[t,i]=S_s[t+1,i]\oplus S_s[t,i]
$$

The feature vector is:

$$
\mathrm{detection_rates}_W
==========================

\mathrm{vec}(\mathrm{detection_rates}_W[t,i])
$$

This measures temporal instability in the syndrome field.

It is useful as a baseline and as part of the full S_M feature vector, but detection rates alone are not the headline S_M claim.

---

## Feature family 3: agreement_profiles

Agreement profiles summarize the agreement field along edge and time axes.

The edge agreement profile is:

$$
P^{\mathrm{edge}}_W[i]
======================

\frac{1}{n_WR}
\sum_{s\in W}
\sum_{t=0}^{R-1}
A_s[t,i]
$$

The time agreement profile is:

$$
P^{\mathrm{time}}_W[t]
======================

\frac{1}{n_W(d-1)}
\sum_{s\in W}
\sum_{i=0}^{d-2}
A_s[t,i]
$$

The feature vector is:

$$
\mathrm{agreement_profiles}_W
=============================

\left[
P^{\mathrm{edge}}_W,
P^{\mathrm{time}}_W
\right]
$$

This is the first core S_M field feature.

It includes final data edge parity through $A_s[t,i]$.

That makes it more structured than raw syndrome rates or detection rates alone.

---

## Feature family 4: sm_field

The `sm_field` feature family keeps a fuller field representation.

It includes:

$$
\bar{A}_W[t,i]
==============

\frac{1}{n_W}
\sum_{s\in W}A_s[t,i]
$$

and:

$$
\bar{X}_W[t,i]
==============

\frac{1}{n_W}
\sum_{s\in W}X_s[t,i]
$$

flattened over time and edge.

It also includes compact descriptors of field variation, such as:

```text
mean agreement
agreement standard deviation
edge-profile variation
time-profile variation
mean detection rate
detection-rate variation
shot-window agreement variation
shot-window detection variation
```

In plain text:

```text
sm_field = agreement field + detection field + compact field descriptors
```

This is the main S_M field feature family.

---

## Feature family 5: sm_all

The full current feature family is:

$$
\mathrm{sm_all}_W
=================

[
\mathrm{raw_rates}_W,
\mathrm{detection_rates}_W,
\mathrm{agreement_profiles}_W,
\mathrm{sm_field}_W
]
$$

It is the widest current S_M feature vector.

Current benchmark result:

```text
QPROJ sm_all reaches 0.999 balanced accuracy on real-vs-control classification.
```

This is useful, but the most important diagnostic is not only that `sm_all` performs well.

The important diagnostic is the split:

```text
raw_rates / detection_rates stay near chance
agreement_profiles / sm_field / sm_all go near-perfect
```

That split is the S_M operator signature.

---

## Substrate paths

The benchmark uses three substrate paths:

$$
S_M^{\mathrm{geo}}
$$

$$
S_M^{\mathrm{gproj}}
$$

$$
S_M^{\mathrm{qproj}}
$$

In plain text:

```text
S_M_geo    = synthetic/reference field
S_M_gproj  = GPU-generated syndrome-spacetime base
S_M_qproj  = real QPU syndrome-spacetime base
```

### S_M_geo

The `geo` path is a synthetic/reference field model.

It creates records with the same downstream schema as real bases:

```text
data_d{d}
synd_d{d}
```

It is not a hardware simulator.

Its job is to provide a clean reference substrate for the same benchmark pipeline.

### S_M_gproj

The `gproj` path loads a GPU-generated S_M base:

```text
data/sm_gpu_data_plus_<TAG>.npz
```

It uses the same schema as QPU dumps.

Its role is to provide a controlled generated comparison substrate.

### S_M_qproj

The `qproj` path loads a real QPU S_M base:

```text
data/sm_data_plus_<JOB_ID>.npz
```

It is the physical syndrome-spacetime record.

The QPU path is not a speed claim.

It is a field-substrate claim:

```text
Does real hardware produce a syndrome-spacetime field whose structure survives controls?
```

---

## Destructive controls

The benchmark uses destructive controls to test whether the S_M channel is load-bearing.

The control set is:

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

### real

No transformation.

$$
(D,S)\mapsto(D,S)
$$

### shot_shuffle_synd

Shuffle syndrome records across shots:

$$
(D_s,S_s)\mapsto(D_s,S_{\pi(s)})
$$

This preserves syndrome marginal structure but destroys shot-level pairing between final data and syndrome spacetime.

### time_shuffle_synd

Shuffle syndrome time order within each shot/edge:

$$
S_s[t,i]\mapsto S_s[\pi_{s,i}(t),i]
$$

This preserves per-edge syndrome values but destroys temporal order.

### edge_shuffle_synd

Shuffle syndrome edge order within each shot/time:

$$
S_s[t,i]\mapsto S_s[t,\pi_{s,t}(i)]
$$

This preserves per-time syndrome values but destroys spatial/edge order.

### uniform_synd

Replace syndrome samples with uniformized draws matching broad syndrome rates.

In plain text:

```text
preserve approximate syndrome probability envelope
destroy structured syndrome spacetime
```

### final_shuffle

Shuffle final data records across shots:

$$
(D_s,S_s)\mapsto(D_{\pi(s)},S_s)
$$

This preserves final-data and syndrome marginals but destroys the final edge parity / syndrome pairing.

This control is especially important for S_M, because it attacks:

$$
E_s[i] \leftrightarrow S_s[t,i]
$$

directly.

### all_uniform

Uniformize data and syndrome records.

In plain text:

```text
preserve broad probability envelope
destroy data structure
destroy syndrome structure
destroy agreement structure
```

### time_reverse_synd

Reverse the time axis:

$$
S_s[t,i]\mapsto S_s[R-1-t,i]
$$

This preserves time content but changes forward temporal orientation.

### edge_reverse_synd

Reverse the edge axis:

$$
S_s[t,i]\mapsto S_s[t,d-2-i]
$$

This preserves edge content but changes spatial orientation.

---

## Benchmark tasks

The current canonical runner is:

```bash
python s_m_benchmark.py
```

Full benchmark modes:

```bash
python s_m_benchmark.py --sweep ALL
python s_m_benchmark.py --probe
```

The benchmark has three main tasks.

---

## Task A: real-vs-control classification

Task A asks:

```text
Can the benchmark distinguish intact S_M records from destructive controls?
```

The label is binary:

$$
y =
\begin{cases}
\mathrm{real}, & \text{mode}=\mathrm{real} \
\mathrm{control}, & \text{otherwise}
\end{cases}
$$

The current strongest results are:

| Substrate | Feature              | Window | Model         | Balanced accuracy |
| --------- | -------------------- | -----: | ------------- | ----------------: |
| `QPROJ`   | `sm_all`             |     64 | logistic      |             0.999 |
| `QPROJ`   | `sm_field`           |     64 | kNN-euclidean |             0.998 |
| `QPROJ`   | `agreement_profiles` |     64 | kNN-cosine    |             0.990 |
| `GPROJ`   | `sm_field`           |     64 | kNN-euclidean |             0.985 |
| `GPROJ`   | `agreement_profiles` |     64 | kNN-cosine    |             0.982 |
| `GPROJ`   | `sm_all`             |     64 | random forest |             0.980 |

Scalar-like baselines stay near chance:

| Substrate | Feature           | Example balanced accuracy |
| --------- | ----------------- | ------------------------: |
| `QPROJ`   | `raw_rates`       |                     0.535 |
| `QPROJ`   | `detection_rates` |                     0.509 |
| `GPROJ`   | `raw_rates`       |                     0.502 |
| `GPROJ`   | `detection_rates` |                     0.500 |
| `GEO`     | `raw_rates`       |                     0.503 |
| `GEO`     | `detection_rates` |                     0.502 |

The mathematical read is:

$$
\mathrm{agreement/field\ structure}
\gg
\mathrm{raw\ scalar\ rates}
$$

for real-vs-control separation in the current benchmark.

---

## Task B: control-source classification

Task B asks:

```text
Can the benchmark identify which destructive control produced the field?
```

The label is multi-class:

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

Current strongest results:

| Substrate | Feature              | Window | Model         | Balanced accuracy |
| --------- | -------------------- | -----: | ------------- | ----------------: |
| `QPROJ`   | `sm_field`           |     64 | kNN-cosine    |             0.853 |
| `GPROJ`   | `sm_field`           |     64 | random forest |             0.848 |
| `QPROJ`   | `sm_all`             |     64 | random forest |             0.848 |
| `GPROJ`   | `sm_all`             |     64 | random forest |             0.843 |
| `QPROJ`   | `agreement_profiles` |     64 | random forest |             0.759 |

This is stronger than a simple real/fake test.

It means different destructive controls leave distinguishable field signatures.

---

## Task C: distance prediction

Task C asks:

```text
Can field windows predict repetition-code distance?
```

The label is:

$$
y=d
$$

using only real windows.

Current result:

```text
GEO   = 1.000 balanced accuracy
GPROJ = 1.000 balanced accuracy
QPROJ = 1.000 balanced accuracy
```

across multiple feature families.

This result is useful, but it should be read carefully.

Distance prediction can be influenced by:

```text
array shape
edge count
syndrome count
distance-dependent rates
field size
```

Therefore, it is not the main S_M claim by itself.

The stronger result is Task A:

```text
field-aware features separate real from destructive controls
while raw scalar-like rates stay near chance
```

---

## Substrate agreement metric

The benchmark writes:

```text
substrate_agreement.csv
```

It compares profile vectors across substrates.

Profile families:

```text
agreement_edge
agreement_time
agreement_field
detection_edge
detection_time
detection_field
```

For a profile vector $p_a$ from substrate $a$ and $p_b$ from substrate $b$, the benchmark reports correlation:

$$
\rho(a,b)
=========

\mathrm{corr}(p_a,p_b)
$$

and L2 distance:

$$
\Delta_2(a,b)
=============

|p_a-p_b|_2
$$

This is the S_M analogue of the G_M agreement metric.

For G_M, agreement is score-level:

$$
\frac{1}{M}\sum_j|\hat G_M^{\mathrm{proj}}(Q_i,K_j)-G_M^{\mathrm{geom}}(Q_i,K_j)|
$$

For S_M, agreement is profile-level:

$$
\mathrm{corr}(\mathrm{field\ profile}_a,\mathrm{field\ profile}_b)
$$

and:

$$
|\mathrm{field\ profile}_a-\mathrm{field\ profile}_b|_2
$$

This is not a throughput metric.

It is a substrate-comparison metric.

---

## CUDA feature extraction math

The optimized CUDA path computes the same S_M feature reductions as the NumPy reference path.

The core kernel is:

```text
sm_window_features_kernel
```

Inputs:

```text
data  : uint8, shape (shots, d)
synd  : uint8, shape (shots, rounds, d-1)
```

For each window, the kernel computes:

```text
raw_out
det_out
agree_prof_out
sm_field_out
```

### raw_out

Contains:

$$
\bar{D}_W[i]
$$

and:

$$
\bar{S}_W[t,i]
$$

### det_out

Contains:

$$
\bar{X}_W[t,i]
$$

where:

$$
X[t,i]=S[t+1,i]\oplus S[t,i]
$$

### agree_prof_out

Contains:

$$
P^{\mathrm{edge}}_W[i]
$$

and:

$$
P^{\mathrm{time}}_W[t]
$$

### sm_field_out

Contains:

$$
\bar{A}_W[t,i]
$$

and:

$$
\bar{X}_W[t,i]
$$

The CUDA path does not change the operator.

It only accelerates:

```text
substrate × distance × control × window × shots × rounds × edges
```

feature extraction.

The operator boundary remains:

```text
S_M only:
  final edge parity
  syndrome spacetime
  agreement field
  detection events
  windowed feature reductions

not included:
  stress tensor
  token retrieval
  TSP deformation
```

---

## What the current math supports

The current mathematical and benchmark-supported claim is that `S_M` is a syndrome-spacetime field operator with:

1. a synthetic/reference field path,
2. a GPU-generated syndrome-spacetime base path,
3. a real QPU syndrome-spacetime base path.

The current evidence supports:

* final data edge parity maps data readout into syndrome edge coordinates;
* agreement fields couple final data and repeated syndrome measurements;
* field-aware features separate real records from destructive controls;
* raw scalar-like rates remain near chance in the key real-vs-control test;
* control-source classification rises above chance;
* distance structure is visible across geo, gproj, and qproj records;
* CUDA accelerates feature extraction without changing the operator boundary.

The current evidence does **not** support:

* a logical-error-rate claim;
* a quantum advantage claim;
* a QPU speedup claim;
* a stress-tensor claim;
* a claim that token retrieval or TSP projectors are proven by S_M;
* a claim that all future Converger operators are validated.

The correct bounded framing is:

$$
S_M
===

\text{field-structured, control-tested, substrate-comparable syndrome metric}
$$

---

## Pointers

* **`s_m_benchmark.py`** — Current canonical S_M benchmark runner.
* **`s_m_gpu_generate.py`** — GPU-generated syndrome-spacetime base generator.
* **`s_m_qpu_generate.py`** — Unified QPU submit/dump path.
* **`kernels/sm_kernel.cu`** — CUDA feature extraction for windowed S_M fields.
* **`docs/architecture.md`** — How the math compiles into package architecture and kernels.
* **`docs/known_issues.md`** — Current limitations and failure modes.
* **`examples/sm_windowed_knn_benchmark.py`** — Earlier field-level kNN benchmark.
* **`examples/sm_tsp_projector_example.py`** — Downstream bounded field-deformation example.
* **`probes/sm_analyze.py`** — Earlier S_M analysis probe.
* **`probes/token_retrieval_projector.py`** — Downstream token-retrieval projector.
* **`probes/bright_observer_token_retrieval.py`** — Token-retrieval projector with BrightDate-compatible observer metadata.

---

## Final read

The process is the process.

The package started as a mixed field-analysis workspace, separated the syndrome field from downstream stress and interaction ideas, defined the core field object, added GPU and QPU base paths, added destructive controls, added CUDA feature extraction, and now reports bounded claims.

The math says:

$$
D[i] \text{ is final data}
$$

$$
E[i]=D[i]\oplus D[i+1] \text{ is final edge parity}
$$

$$
S[t,i] \text{ is syndrome spacetime}
$$

$$
A[t,i]=1-(S[t,i]\oplus E[i]) \text{ is agreement}
$$

The benchmark says:

$$
\text{field-aware features separate real records from destructive controls}
$$

$$
\text{raw scalar-like rates remain near chance in the key test}
$$

$$
\text{control-source classification rises above chance}
$$

$$
\text{geo, gproj, and qproj can be compared under one harness}
$$

That is the claim to defend.
