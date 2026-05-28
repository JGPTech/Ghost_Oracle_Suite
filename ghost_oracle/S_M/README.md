# S_M — Syndrome Metric

`S_M` is the Ghost Oracle Suite’s syndrome-spacetime operator family.

It was discovered by treating a repetition-code decoder failure as forensic evidence instead of discarding it as broken. The useful object was not a logical error-rate scalar. It was the structure hiding in the syndrome record.

Current working model:

```text
S_M(t, i)
```

where:

- `t` is the syndrome round / time index,
- `i` is the code edge / stabilizer index.

From this field, the analysis builds a stress-tensor-style diagnostic:

```text
ΔtS[t,i] = S[t+1,i] XOR S[t,i]
ΔxS[t,i] = S[t,i+1] XOR S[t,i]

T_S = [[<ΔtS ΔtS>, <ΔtS ΔxS>],
       [<ΔxS ΔtS>, <ΔxS ΔxS>]]
```

The careful claim:

> In a logical-cat-state QPU run, the syndrome record forms a bounded spacetime field with strong real-vs-control separation, distance-scaling, spatially dominant stress, and nonzero temporal-spatial coupling.

Exact tensor values are hardware-run dependent. The pipeline is meant to reproduce the qualitative `S_M` signature, not a bit-for-bit copy of one QPU job.

---

## Quick path

The folder has one default path:

```bash
python ghost_oracle/S_M/sm_submit.py
python ghost_oracle/S_M/sm_dump.py
python ghost_oracle/S_M/sm_analyze.py
```

What each step does:

| Step | Script | Purpose |
|---:|---|---|
| 1 | `sm_submit.py` | Submit the default logical-cat repetition-code QPU job. |
| 2 | `sm_dump.py` | Fetch the completed Qiskit Runtime result and save a stable `.npz`. |
| 3 | `sm_analyze.py` | Run the full `S_M` shape + stress-tensor analysis. |

The first command submits a QPU job. Wait for the job to complete before running the dump step.

---

## Step 1 — submit the QPU job

```bash
python ghost_oracle/S_M/sm_submit.py
```

Default experiment:

```text
backend      = ibm_marrakesh
flag level   = f=0
distances    = 3 5 7 9
rounds       = 10
shots        = 4096
init state   = plus
basis        = z
```

The default initial state is a logical cat state inside the repetition-code space:

```text
|+_L> = (|000...0> + |111...1>) / sqrt(2)
```

This is important: for this experiment, final majority-vote logical error is diagnostic only. The useful terminal object is final edge parity:

```text
E_i = D_i XOR D_{i+1}
```

The submitter writes:

```text
data/sm_job_<JOB_ID>.json
data/latest_sm_job.json
```

Common overrides:

```bash
python ghost_oracle/S_M/sm_submit.py --shots 8192
python ghost_oracle/S_M/sm_submit.py --backend ibm_fez
python ghost_oracle/S_M/sm_submit.py --init-state minus
python ghost_oracle/S_M/sm_submit.py --distances 3 5
```

---

## Step 2 — dump Qiskit Runtime data

After the job finishes:

```bash
python ghost_oracle/S_M/sm_dump.py
```

With no arguments, this reads:

```text
data/latest_sm_job.json
```

and dumps the latest submitted job.

To dump a specific job:

```bash
python ghost_oracle/S_M/sm_dump.py <JOB_ID>
```

The dumper writes:

```text
data/sm_data_plus_<JOB_ID>.npz
data/latest_sm_data.json
```

Expected arrays in the `.npz`:

```text
distances
data_d3, synd_d3
data_d5, synd_d5
data_d7, synd_d7
data_d9, synd_d9
```

Array shapes:

```text
data_d{d}  : uint8, shape (shots, d)
synd_d{d}  : uint8, shape (shots, rounds, d-1)
flag_d{d}  : optional, only for flag levels that measure flags
```

Debug options:

```bash
python ghost_oracle/S_M/sm_dump.py <JOB_ID> --list-registers
python ghost_oracle/S_M/sm_dump.py <JOB_ID> --reverse-bits
```

Use `--reverse-bits` only if the light diagnostics clearly indicate a bit-order inversion.

---

## Step 3 — analyze the dumped data

```bash
python ghost_oracle/S_M/sm_analyze.py
```

With no arguments, this reads:

```text
data/latest_sm_data.json
```

and writes results to:

```text
analysis/sm_<JOB_ID>/
```

Main outputs:

```text
operator_shape_report.json
stress_tensor_report.json
sister_operator_teaser.png
stress_tensor_summary.png
```

Run on a specific dumped file:

```bash
python ghost_oracle/S_M/sm_analyze.py --npz data/sm_data_plus_<JOB_ID>.npz
```

Skip plots:

```bash
python ghost_oracle/S_M/sm_analyze.py --no-plots
```

---

## What the analysis tests

### 1. Scalar vs vector vs field

The shape probe asks whether the candidate object is best explained as:

```text
scalar
edge vector
time vector
full spacetime field
```

The important output is:

```text
S_M SHAPE SUMMARY
```

A strong `S_M` run should show field-like behavior across distances.

Example result:

| d | Field L2 | Detection-field L2 | Shape |
|---:|---:|---:|---|
| 3 | 1.2081 | 0.8680 | field / edge-anisotropic |
| 5 | 2.0462 | 1.1921 | field / edge-anisotropic |
| 7 | 1.9284 | 1.9591 | field / edge-anisotropic |
| 9 | 2.1345 | 2.1913 | field / edge-anisotropic |

### 2. Stress tensor

The stress-tensor analysis asks whether the syndrome spacetime field has stable temporal, spatial, and coupling components.

The important output is:

```text
S_M STRESS TENSOR SUMMARY
```

Example result:

| d | Ttt | Txx | Ttx | trace | anisotropy | coupling | best local |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.10761 | 0.16930 | 0.03084 | 0.27691 | -0.2228 | 0.2285 | 1.23931 |
| 5 | 0.11051 | 0.21118 | 0.04653 | 0.32169 | -0.3130 | 0.3046 | 1.45825 |
| 7 | 0.12348 | 0.29035 | 0.05810 | 0.41383 | -0.4032 | 0.3068 | 2.36326 |
| 9 | 0.12950 | 0.29289 | 0.06024 | 0.42238 | -0.3868 | 0.3093 | 2.85467 |

The signature to look for:

```text
Txx > Ttt
Ttx > 0
anisotropy < 0
best local separation grows with distance
```

Interpretation:

- `Ttt` is temporal syndrome-gradient energy.
- `Txx` is spatial syndrome-gradient energy.
- `Ttx` is temporal-spatial coupling.
- negative anisotropy means spatial stress dominates.
- `best local` measures local real-vs-control separation.

---

## Controls

The analysis uses shuffled and uniform controls to test whether the observed structure survives attacks on the data pairing and spacetime order.

Common controls include:

```text
shot_shuffle_synd
time_shuffle_synd
edge_shuffle_synd
uniform_synd
all_uniform
time_reverse_synd
edge_reverse_synd
final_shuffle
```

A useful `S_M` signal should separate real QPU records from these controls, especially in the field and local tensor diagnostics.

---

## Hardware-run dependence

`S_M` is extracted from real QPU shot records. Exact values can change between jobs because backend calibration, drift, layout behavior, and stochastic noise change.

The expected reproducible target is the qualitative operator signature:

```text
field-like structure
spatially dominant stress
nonzero temporal-spatial coupling
real-vs-control separation
distance-growing local tensor structure
```

Not every run should be expected to match one reference job numerically.

---

## Files produced by the pipeline

After one full run, typical generated files are:

```text
data/sm_job_<JOB_ID>.json
data/latest_sm_job.json
data/sm_data_plus_<JOB_ID>.npz
data/latest_sm_data.json

analysis/sm_<JOB_ID>/operator_shape_report.json
analysis/sm_<JOB_ID>/stress_tensor_report.json
analysis/sm_<JOB_ID>/sister_operator_teaser.png
analysis/sm_<JOB_ID>/stress_tensor_summary.png
```

Recommended `.gitignore` patterns:

```gitignore
data/*.npz
analysis/
sm_job_*.json
latest_sm_*.json
*_report.json
sister_operator_teaser.png
stress_tensor*.png
```

---

## Script map

```text
sm_submit.py
    Submit the default logical-cat QPU experiment.

sm_dump.py
    Fetch Qiskit Runtime results and write stable NumPy arrays.

sm_analyze.py
    Run scalar/vector/field shape analysis, stress-tensor analysis, and plots.
```

No legacy scripts are required for the default path.

---

## Notes

`S_M` is a live research object. The current evidence supports a syndrome-spacetime field interpretation, but the repo intentionally keeps claims bounded:

```text
observed field-like QPU structure
observed stress-tensor signature
observed real-vs-control separation
```

rather than claiming a complete theory of the operator.

The next development steps are likely:

- cleaner multi-job comparison,
- additional backends,
- flag-level sweeps,
- controlled round-depth sweeps,
- tighter math for the relation between `S_M` and the original `G_M`.
