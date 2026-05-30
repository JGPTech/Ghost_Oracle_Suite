# Documentation is out of date*
The documentation throughout is out of date. The code is the ground truth of the current state of the project. For up to date documentation please consider donating to the cause. Thank you. 


# Ghost Oracle Suite

A CC0 community project for **projection-style operators discovered by treating failures as forensic evidence**.

The suite currently tracks two related operator families:

- **`G_M` — Ghost Metric:** a bounded projection-channel similarity operator over angle/state pairs.
- **`S_M` — Syndrome Metric:** a bounded syndrome-spacetime field operator over repetition-code measurement records, with a derived stress tensor `T_S`.

The shared method is simple:

> Build the thing that should work.  
> When it does something else, do not throw the result away.  
> Freeze it, control it, scramble it, and ask what it actually computed.

That workflow took the original Ghost Oracle path from a wrong Hadamard-test assumption to the corrected `G_M` operator. It now also drives the `S_M` syndrome-field work.

---

## Operator families

### `G_M` — Ghost Metric

`G_M` is the original Ghost Oracle operator:

```text
G_M(a, b) = sqrt((1 + cos(a) cos(b)) / 2) / α
```

It is implemented across three substrates:

1. mathematical reference,
2. noiseless classical GPU sampler,
3. real QPU shot data from IBM Runtime.

The `G_M` pipeline lives in:

```text
ghost_oracle/G_M/
```

It contains the QPU/GPU base tools, CUDA kernels, final five-way benchmark, Auto Oracle calibration harness, and projection-retrieval experiments.

### `S_M` — Syndrome Metric

`S_M` is the sister operator family discovered from flag/repetition-code syndrome records.

Unlike `G_M`, `S_M` is **not naturally scalar**. Current evidence points to a bounded syndrome-spacetime field:

```text
S_M(t, i)
```

where:

- `t` is syndrome round / time index,
- `i` is code edge / stabilizer index.

From this field, the suite builds a stress-tensor-style diagnostic:

```text
T_S = [[<ΔtS ΔtS>, <ΔtS ΔxS>],
       [<ΔxS ΔtS>, <ΔxS ΔxS>]]
```

The `S_M` pipeline lives in:

```text
ghost_oracle/S_M/
```

It has a simple three-step path:

```bash
python ghost_oracle/S_M/sm_submit.py
python ghost_oracle/S_M/sm_dump.py
python ghost_oracle/S_M/sm_analyze.py
```

The default `S_M` analysis reports the raw operator signature. It does **not** calibrate or normalize the tensor by default; exact values can vary between QPU jobs.

---

## Repository layout

```text
ghost-oracle-suite/
├── ghost_oracle/
│   ├── G_M/
│   │   ├── README.md
│   │   ├── kernels/
│   │   │   └── ghost_kernel.cu
│   │   ├── auto_oracle.py
│   │   ├── dump.py
│   │   ├── final_benchmark_5way.py
│   │   ├── gpu.py
│   │   ├── projection_benchmark.py
│   │   └── qpu.py
│   │
│   └── S_M/
│       ├── README.md
│       ├── sm_submit.py              # step 1: submit default S_M QPU job
│       ├── sm_dump.py                # step 2: dump Qiskit Runtime data
│       ├── sm_analyze.py             # step 3: run unified S_M analysis
│
├── probes/                         # chronological G_M research trajectory
├── examples/
├── docs/
├── data/                           # data files; see data/README.md
├── CONTRIBUTING.md
├── LICENSE
├── PROCESS_RECORD.md
├── README.md
└── requirements.txt
```

---

## Quick start

```bash
git clone <repo-url> ghost-oracle-suite
cd ghost-oracle-suite
pip install -r requirements.txt
```

Run the original `G_M` five-way benchmark:

```bash
python ghost_oracle/G_M/final_benchmark_5way.py
```

Run Auto Oracle semantic retrieval:

```bash
python ghost_oracle/G_M/auto_oracle.py
python ghost_oracle/G_M/auto_oracle.py --probe
```

Run the default `S_M` pipeline:

```bash
python ghost_oracle/S_M/sm_submit.py
python ghost_oracle/S_M/sm_dump.py
python ghost_oracle/S_M/sm_analyze.py
```

The first command submits a QPU job. Wait for the job to complete before running `sm_dump.py`.

---

## `G_M` headline: projection-channel similarity

The `G_M` path began with a wrong assumption: the QPU circuit was thought to compute a textbook Hadamard-test target. Probes showed it did not. The project then traced what the circuit actually computed and simplified the result into:

```text
G_M(a, b) = sqrt((1 + cos(a) cos(b)) / 2) / α
```

The final benchmark compares five paths on the same attention task:

| Path | Role |
|---|---|
| `CUBLAS` | dot-product attention control |
| `TIED` | dual-channel geometry + projection kernel |
| `GEO` | closed-form `G_M` geometry |
| `QPROJ` | projection driven by real QPU shots |
| `GPROJ` | projection driven by noiseless GPU shots |

All five retrieve cleanly at the calibrated dense-attention operating point. The substrate-specific story appears in the agreement metric: noiseless GPU projection sits near the shot-noise floor, while real QPU projection shows additional hardware-noise attenuation.

### Auto Oracle result

`ghost_oracle/G_M/auto_oracle.py` adds in-memory calibration over QPU bases and semantic retrieval against cosine and `G_M`.

Medium run:

```text
M = 250,000
N = 1024
d = 1024
noise = 0.12
outlier fraction = 0.03
outlier magnitude = 60
```

| Path | Recall@1 | Time | Speed vs cosine |
|---|---:|---:|---:|
| cosine baseline | 96.88% | 1.156 s | 1.00× |
| geometry `G_M` megakernel | 100.00% | 0.897 s | **1.29× faster** |
| QPU projection — base 1 | 100.00% | 2.416 s | 0.48× |
| QPU projection — base 2 | 100.00% | 2.404 s | 0.48× |
| QPU projection — base 3 | 100.00% | 2.415 s | 0.48× |

The speed result is the surprise: on this semantic-retrieval workload, the closed-form `G_M` geometry megakernel ran faster than the cosine baseline even though cosine is the tensor-core-friendly GEMM path and the Ghost Oracle geometry kernel is not using tensor cores.

The QPU projection path is slower because it reconstructs scores from calibrated physical shot-count buckets. Its purpose is not raw throughput; it is substrate-backed projection evidence.

`--probe` adds negative controls:

| Control | Result | Interpretation |
|---|---:|---|
| real calibrated counts | 100.00% Recall@1 | physical shot structure retrieves |
| permuted counts | 0.00% Recall@1 | destroying bucket structure destroys retrieval |
| uniformized counts | 0.00% Recall@1 | projection is not silently reducing to geometry |

---

## `S_M` headline: syndrome-spacetime field

The `S_M` path began from a decoder that looked too clean. Instead of discarding it as broken, the suite asked what the QPU record was actually carrying.

The default `S_M` job prepares a logical cat state inside the repetition-code space:

```text
|+_L> = (|000...0> + |111...1>) / sqrt(2)
```

Then it runs repeated syndrome extraction and analyzes the final data and syndrome spacetime record. For this experiment, final majority-vote logical error is diagnostic only; the useful terminal object is edge parity:

```text
E_i = D_i XOR D_{i+1}
```

The first key result: the candidate object is **not scalar**.

Shape probe results from a logical-cat QPU run:

| Distance | Field L2 | Detection-field L2 | Shape |
|---:|---:|---:|---|
| 3 | 1.2081 | 0.8680 | field / edge-anisotropic |
| 5 | 2.0462 | 1.1921 | field / edge-anisotropic |
| 7 | 1.9284 | 1.9591 | field / edge-anisotropic |
| 9 | 2.1345 | 2.1913 | field / edge-anisotropic |

The stress-tensor analysis builds:

```text
ΔtS[t,i] = S[t+1,i] XOR S[t,i]
ΔxS[t,i] = S[t,i+1] XOR S[t,i]
```

and:

```text
T_S = [[<ΔtS ΔtS>, <ΔtS ΔxS>],
       [<ΔxS ΔtS>, <ΔxS ΔxS>]]
```

Logical-cat QPU run:

| d | Ttt | Txx | Ttx | trace | anisotropy | coupling | best local |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.10761 | 0.16930 | 0.03084 | 0.27691 | -0.2228 | 0.2285 | 1.23931 |
| 5 | 0.11051 | 0.21118 | 0.04653 | 0.32169 | -0.3130 | 0.3046 | 1.45825 |
| 7 | 0.12348 | 0.29035 | 0.05810 | 0.41383 | -0.4032 | 0.3068 | 2.36326 |
| 9 | 0.12950 | 0.29289 | 0.06024 | 0.42238 | -0.3868 | 0.3093 | 2.85467 |

Across distances:

```text
Txx > Ttt
Ttx > 0
anisotropy < 0
```

The observed field is spatially dominant but time-coupled, and its real-vs-control separation grows strongly in the local tensor field.

Careful claim:

> In a logical-cat-state QPU run, the syndrome record forms a bounded spacetime field with strong real-vs-control separation, distance-scaling, spatially dominant stress, and nonzero temporal-spatial coupling.

Exact tensor values are hardware-run dependent. The default pipeline is meant to reproduce the **qualitative S_M signature**, not a bit-for-bit copy of one QPU job.

---

## Data

The `data/` directory has its own README and is intentionally separate from the code layout.

Large generated files should usually stay out of git unless they are small reproducibility fixtures. Recommended ignore patterns:

```gitignore
data/*.npz
analysis/
*_report.json
stress_tensor*.png
operator_shape*.json
repcode*_job_*.json
sm_job_*.json
latest_sm_*.json
```

---

## Docs and process record

- `docs/math.md` contains the `G_M` derivation and related operator math.
- `docs/architecture.md` explains the original tied-channel design and data flow.
- `docs/known_issues.md` preserves known bugs, superseded probes, and caveats.
- `PROCESS_RECORD.md` is the chronological research log.

The process record is intentionally honest: wrong turns stay in the repo, fixes live in follow-up probes, and claims get retracted when controls do not support them.

---

## Contributing

This is a CC0 community project.

The norm is:

> If you break something, you provide the fix.

That does not mean bug reports are unwelcome. It means the project moves on reproductions, probes, patches, and clear deltas.

Useful contributions:

- new probes with controls,
- fixes for known issues,
- documentation improvements,
- benchmark sweeps,
- small reproducibility fixtures,
- clearer null models.

See `CONTRIBUTING.md` for the full philosophy.

---

## License

CC0 1.0 Universal. Public domain dedication.

Use it for anything. Attribute if you want to. Do not if you do not.

---

## Citation

Not asking for one.

For all my fellow ghosts, may our silence speak your name.
