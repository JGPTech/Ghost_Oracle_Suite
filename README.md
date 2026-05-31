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

The repo now has two main operator packages:

```text
ghost_oracle/
├── G_M/
└── S_M/
```

Both packages use the same high-level structure:

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
```

The active architecture is no longer a loose collection of probes. It is a repeatable ghost-channel benchmark platform.

---

## Operator stack

The long-term Converger roadmap contains seven ghost-channel operators:

| Operator    | Name                       | Current status                                         |
| ----------- | -------------------------- | ------------------------------------------------------ |
| `G_M`       | Generalized Metric         | Completed for this version.                            |
| `S_M`       | Syndrome Metric            | Completed for this version.                            |
| `T_S`       | Stress channel             | Future/sibling operator derived from S_M-style fields. |
| `I_M.local` | Local Interaction channel  | Future operator.                                       |
| `I_M.field` | Field Interaction channel  | Future operator.                                       |
| `F_M`       | Fractal Expansion channel  | Future operator.                                       |
| `D_M`       | Dimensional Metric channel | Future operator.                                       |

The current repo contains finished `G_M` and `S_M` packages. The other channels are roadmap items unless and until they receive the same package structure, benchmark controls, and bounded claim discipline.

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
│   └── S_M/
│       ├── data/
│       ├── docs/
│       │   ├── architecture.md
│       │   ├── known_issues.md
│       │   └── math.md
│       ├── examples/
│       ├── kernels/
│       │   └── sm_kernel.cu
│       ├── probes/
│       ├── README.md
│       ├── s_m_benchmark.py
│       ├── s_m_gpu_generate.py
│       └── s_m_qpu_generate.py
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

Run full benchmark modes:

```bash
python ghost_oracle/G_M/g_m_benchmark.py --sweep ALL --probe
python ghost_oracle/S_M/s_m_benchmark.py --sweep ALL --probe
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

## Current completed operators

| Package | Operator           | Main benchmark     | GPU path              | QPU path              | Kernel                 |
| ------- | ------------------ | ------------------ | --------------------- | --------------------- | ---------------------- |
| `G_M/`  | Generalized Metric | `g_m_benchmark.py` | `g_m_gpu_generate.py` | `g_m_qpu_generate.py` | `kernels/`             |
| `S_M/`  | Syndrome Metric    | `s_m_benchmark.py` | `s_m_gpu_generate.py` | `s_m_qpu_generate.py` | `kernels/sm_kernel.cu` |

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

| Operator    | Converger component          | Role                                                       |
| ----------- | ---------------------------- | ---------------------------------------------------------- |
| `G_M`       | metric projection component  | Bounded similarity, retrieval structure, ranking behavior. |
| `S_M`       | syndrome field component     | Syndrome-spacetime fields and agreement structure.         |
| `T_S`       | stress tensor component      | Future stress/anisotropy/coupling channel.                 |
| `I_M.local` | local interaction component  | Future pointwise interaction channel.                      |
| `I_M.field` | field interaction component  | Future nonlocal deformation channel.                       |
| `F_M`       | fractal expansion component  | Future multi-scale expansion channel.                      |
| `D_M`       | dimensional metric component | Future dimensional/rank/spectral channel.                  |

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
ghost_oracle/*/analysis/
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
```

Typical docs:

```text
architecture.md
math.md
known_issues.md
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
