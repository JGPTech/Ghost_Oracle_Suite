# D_M — Dimensional Entanglement Projection

`D_M` is the Ghost Oracle operator family for **dimensional witness-manifold projection**.

It began as a QPU dimensional-compression experiment. The first working model was that the frozen QPU scene encoded a useful dimensional-reduction basis. That framing did not survive the probes. Across Probes 00–06, D_M-as-compressor failed against PCA and random projection baselines.

The useful pivot was not to force the circuit into the original interpretation, but to freeze the QPU output and ask what the hardware record actually contained.

The resulting locked signature is:

```text
YZ-primary / ZY-reciprocal dimensional witness manifold
```

with rung coordinates:

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

Current framing:

```text
D_M = dimensional Bell-witness manifold listener
```

`D_M` is implemented across three substrates:

1. real QPU Bell-listener shot records,
2. GPU-generated controlled Bell-witness bases,
3. exact closed-form classical GEO reference.

The core claim is not that Bell nonlocality has been certified.

The core claim is that a bare two-qubit QPU listener record contains a YZ-primary / ZY-reciprocal dimensional witness manifold; that active delay/offset conditions separate from null; that same-shot pairing, reciprocal structure, and delay order are load-bearing; and that compound corruptions cross a measurable collapse boundary.

---

## Quick path

The current canonical benchmark path is:

```bash
python d_m_benchmark.py
```

Useful variants:

```bash
python d_m_benchmark.py --repair-metadata
python d_m_benchmark.py --skip-qproj
python d_m_benchmark.py --skip-gproj
python d_m_benchmark.py --skip-geo
python d_m_benchmark.py --reps 20
```

The QPU/GPU base-generation tools are used for producing fresh bases:

| Script                | Purpose                                            |
| --------------------- | -------------------------------------------------- |
| `d_m_qpu_generate.py` | Submit / dump a tiled D_M QPU Bell-listener base.  |
| `d_m_gpu_generate.py` | Generate a GPU-compatible controlled `gproj` base. |
| `d_m_benchmark.py`    | Current canonical final capstone benchmark.        |

Current benchmark claims should come from:

```text
d_m_benchmark.py
```

and from the saved output:

```text
analysis/d_m_final_capstone_<timestamp>/result.json
analysis/d_m_final_capstone_<timestamp>/final_claim_report.md
```

The process history should come from:

```text
PROCESS_RECORD.md
probes/D_M_probe_process_record.md
```

---

## Operator

The core witness fields are:

```text
XY = connected(XY)
YZ = connected(YZ)
ZY = connected(ZY)
YX = connected(YX)
```

The current canonical coordinate frame is:

```text
YZ = primary witness dimension
ZY = reciprocal / inverted witness dimension
XY / YX = comparison dimensions
```

The projected rung coordinates are:

```text
Y   = YZ
Z   = ZY
R   = -Z
E   = sqrt(Y^2 + R^2)
C   = sqrt(XY^2 + YX^2)
S   = E - C
phi = atan2(R, Y) mod pi
```

The primary D_M signature is therefore:

```text
YZ / ZY-return / delay / pi-phase
```

Important structural notes:

```text
D_M is a Bell-witness manifold listener.
D_M is delay-ordered.
D_M is same-shot-pair dependent.
D_M has a canonical YZ-primary / ZY-reciprocal coordinate frame.
D_M is invariant to allowed witness-channel re-descriptions.
D_M is not a density-matrix reconstruction.
D_M is not device-independent Bell-nonlocality certification.
D_M is not proof of a prepared Bell state.
D_M is not a QPU speedup or quantum advantage claim.
```

The canonical YZ/ZY frame is the coordinate system used for reporting. Allowed channel re-descriptions are not considered falsifying corruptions of the dimensional manifold.

---

## Repository structure

Actual current `D_M/` layout:

```text
D_M/
├── data/
│
├── docs/
│   ├── architecture.md
│   └── math.md
│
├── examples/
│
├── kernels/
│   └── dm_projector_kernel.cu
│
├── probes/
│   ├── analysis/
│   │
│   ├── 00_dm_probe_prune.py
│   ├── 01_dm_probe_channel_extract.py
│   ├── 02_dm_probe_synthetic_maps.py
│   ├── 03_dm_probe_linear_scheduler.py
│   ├── 04_dm_probe_calibrated_seed_projection.py
│   ├── 05_dm_probe_base10_state_selector.py
│   ├── 06_dm_probe_single_calibrated_projector.py
│   ├── 07_dm_probe_bell_listener.py
│   ├── 08_dm_probe_directional_witness.py
│   ├── 09_dm_probe_pi_phase_witness.py
│   ├── 10_dm_probe_qproj_dimensional_entanglement.py
│   ├── 11_dm_probe_cuda_qproj_harness.py
│   ├── 12_dm_probe_benchmark.py
│   ├── 13_dm_probe_benchmark2.py
│   ├── 14_dm_probe_holographic_projection.py
│   ├── 15_dm_probe_native_stagger.py
│   ├── 16_dm_probe_convert_corpus.py
│   ├── 17_dm_probe_raw_boundary_projection.py
│   ├── 18_dm_probe_raw_boundary_gpu_benchmark.py
│   ├── 19_dm_probe_task_utility_retrieval_gpu.py
│   ├── 20_dm_probe_make_random_null_corpus.py
│   ├── 21_dm_probe_04w_windowed_phase_trajectory.py
│   ├── 22_dm_probe_task_utility_retrieval_rawsignal.py
│   ├── 23_dm_probe_dimensional_invariance_controls.py
│   ├── 24_dm_probe_corruption_boundary.py
│   ├── 25_dm_probe_geo_precision_reference.py
│   └── D_M_probe_process_record.md
│
├── d_m_benchmark.py
├── d_m_gpu_generate.py
├── d_m_qpu_generate.py
├── PROCESS_RECORD.md
└── README.md
```

---

## Directory map

| Path                                 | Role                                                                   |
| ------------------------------------ | ---------------------------------------------------------------------- |
| `README.md`                          | Main D_M documentation and current benchmark summary.                  |
| `PROCESS_RECORD.md`                  | Repo-facing D_M process record.                                        |
| `d_m_benchmark.py`                   | Current canonical capstone benchmark runner.                           |
| `d_m_gpu_generate.py`                | Generates GPU-compatible controlled `gproj` bases.                     |
| `d_m_qpu_generate.py`                | Submits / dumps QPU Bell-listener `qproj` bases.                       |
| `data/`                              | Frozen qproj/gproj base files and latest pointers.                     |
| `docs/architecture.md`               | Architecture notes for the D_M operator.                               |
| `docs/math.md`                       | Mathematical notes for the D_M witness manifold.                       |
| `examples/`                          | Optional examples retained for continuity.                             |
| `kernels/dm_projector_kernel.cu`     | Shared CUDA implementation.                                            |
| `probes/`                            | Chronological research probes documenting the full D_M discovery path. |
| `probes/analysis/`                   | Output / analysis workspace for probe scripts.                         |
| `probes/D_M_probe_process_record.md` | Probe-level process record.                                            |
| `analysis/`                          | Generated capstone output folder, created by `d_m_benchmark.py`.       |

---

## Current main entry points

Final capstone benchmark:

```bash
python d_m_benchmark.py
```

Metadata-repair run:

```bash
python d_m_benchmark.py --repair-metadata
```

Fast debugging pass:

```bash
python d_m_benchmark.py --reps 20
```

Run only selected substrates:

```bash
python d_m_benchmark.py --skip-qproj
python d_m_benchmark.py --skip-gproj
python d_m_benchmark.py --skip-geo
```

Base generation:

```bash
python d_m_gpu_generate.py
python d_m_qpu_generate.py submit
python d_m_qpu_generate.py dump <JOB_ID>
```

Run the exact GEO precision probe directly:

```bash
python probes/25_dm_probe_geo_precision_reference.py
```

Run the corruption-boundary probe directly:

```bash
python probes/24_dm_probe_corruption_boundary.py --auto --window 4096 --trials-per-depth 500 --save-trials
```

---

## Data files

The `data/` folder contains two kinds of D_M bases:

```text
dm_data_bell_listener_cavity_offset_<JOB_ID>.npz
```

Real QPU Bell-listener bases dumped from IBM Runtime jobs.

```text
dm_gpu_data_<condition>_4096shots_seed<SEED>.npz
```

GPU-generated controlled Bell-witness bases.

Latest pointers may include:

```text
latest_dm_qpu_data.json
latest_dm_gpu_data.json
latest_dm_data.json
```

Recommended policy:

```text
Keep small curated fixtures if they are part of the reproducibility story.
Keep large generated bases out of git unless intentionally shipping them.
```

Recommended `.gitignore` patterns:

```gitignore
data/dm_data_bell_listener_cavity_offset_*.npz
data/dm_gpu_data_*.npz
analysis/
probes/analysis/
*_report.json
```

---

## Base schema

A frozen D_M qproj/gproj base is a `.npz` file with shared schema.

Core metadata:

```text
schema
suite
operator
substrate
job_id
backend
shots
num_tiles
tile_rung_index
tile_witness_index
tile_base_delay_dt
tile_offset_dt
tile_total_delay_dt
```

Core stacked array:

```text
pair : uint8, shape (tiles, shots, 2)
```

Compatibility arrays may also be stored:

```text
pair_tile{t} : uint8, shape (shots, 2)
tile_witness_label
tile_basis_q0
tile_basis_q1
basis
```

The shared schema is load-bearing because it allows the same projector kernels and benchmark scripts to consume QPU and GPU bases without changing operator logic.

---

## Canonical conditions

The current D_M benchmark uses three canonical conditions:

| Condition   |                base delays dt | offset dt | Meaning                     |
| ----------- | ----------------------------: | --------: | --------------------------- |
| `null`      |             `[0, 0, 0, 0, 0]` |         0 | no explicit delay / offset  |
| `base_only` | `[0, 256, 1024, 4096, 16384]` |         0 | base-delay ladder only      |
| `offset_on` | `[0, 256, 1024, 4096, 16384]` |       128 | base-delay plus tile offset |

The three QPU records used for the locked capstone are:

| Condition   | Job ID                 |
| ----------- | ---------------------- |
| `null`      | `d8fm4ihvjngc73aq3ccg` |
| `base_only` | `d8flk2jo3njc73f0g560` |
| `offset_on` | `d8fl82bo3njc73f0fgd0` |

---

## Probe path

The `probes/` directory is not dead code. It is the research record.

It preserves the chronological path from:

```text
failed dimensional-compression projector
```

to:

```text
Bell-witness listener pivot
```

to:

```text
YZ-primary / ZY-reciprocal phase-space characterization
```

to:

```text
CUDA qproj/gproj record projection
```

to:

```text
raw-boundary and GPT-2 compatibility experiments
```

to:

```text
dimensional invariance and corruption-boundary controls
```

to:

```text
exact closed-form GEO reference
```

The probes should stay in the repo because they document how the operator was found, what was rejected, what controls were used, what was locked, and what the final benchmark is allowed to claim.

---

## Current capstone benchmark

Run:

```bash
python d_m_benchmark.py
```

The current capstone includes:

```text
qproj / gproj / exact-GEO substrate verification
active-vs-null condition separation
record-path independent-bit-shuffle control collapse
allowed channel-invariance checks
forbidden single-fault checks
compound corruption-boundary checks
exact GEO CPU/GPU validation
saved JSON / CSV / NPZ output
final_claim_report.md
```

The benchmark compiles one CUDA module from:

```text
kernels/dm_projector_kernel.cu
```

The production path uses these CUDA kernels:

| Kernel                                       | Role                                                        |
| -------------------------------------------- | ----------------------------------------------------------- |
| `dm_tile_correlator_kernel_u8`               | Compress qproj/gproj `pair[tile, shot, 2]` into tile stats. |
| `dm_independent_bit_shuffle_tile_kernel_u8`  | Destructive same-shot-pair control.                         |
| `dm_rung_projection_kernel_f32`              | Compress tile stats into rung-level D_M manifold stats.     |
| `dm_projection_summary_kernel_f32`           | Compress rung stats into base-level summary vector.         |
| `dm_make_projected_pair_from_bits_kernel_u8` | Probe-22 raw-signal path: raw bits XOR base pair.           |
| `dm_geo_exact_rung_projection_kernel_f32`    | Probe-25 exact closed-form GEO rung reference.              |
| `dm_geo_exact_sweep_summary_kernel_f32`      | Exact GEO batched sweep path.                               |
| `dm_geo_rung_projection_kernel_f32`          | Legacy analytic/aperture GEO retained for old probes.       |
| `dm_geo_sweep_summary_kernel_f32`            | Legacy GEO sweep retained for old probes.                   |
| `dm_der_topk_kernel_f32`                     | DER top-k retrieval kernel retained for appendix/probes.    |

---

## Three-substrate comparison

The benchmark compares three D_M substrate paths:

| Path    | Meaning                                                                |
| ------- | ---------------------------------------------------------------------- |
| `QPROJ` | Real QPU Bell-listener record through tile/rung/summary kernels.       |
| `GPROJ` | GPU-generated controlled Bell-witness record through the same kernels. |
| `GEO`   | Exact closed-form classical reference through the Probe-25 GEO kernel. |

Current primary signature:

```text
YZ-primary / ZY-reciprocal / delay / pi-phase
```

Current final benchmark values:

| Substrate | Condition   | Projection |   E_mean |    S_mean | pi score |  YZ+ |
| --------- | ----------- | ---------: | -------: | --------: | -------: | ---: |
| `QPROJ`   | `null`      |   0.012762 | 0.022385 |  0.019711 |   0.0000 | 0.40 |
| `QPROJ`   | `base_only` |   0.220018 | 0.155983 |  0.134641 |   0.9264 | 1.00 |
| `QPROJ`   | `offset_on` |   0.208717 | 0.152505 |  0.136947 |   0.7890 | 1.00 |
| `GPROJ`   | `null`      |   0.010326 | 0.023433 | -0.000063 |   0.0000 | 0.80 |
| `GPROJ`   | `base_only` |   0.295522 | 0.247033 |  0.217124 |   1.0000 | 1.00 |
| `GPROJ`   | `offset_on` |   0.298216 | 0.275619 |  0.251356 |   1.0000 | 1.00 |
| `GEO`     | `null`      |   0.000000 | 0.000000 |  0.000000 |   0.0000 | 0.00 |
| `GEO`     | `base_only` |   0.693827 | 0.675090 |  0.675090 |   1.0000 | 0.80 |
| `GEO`     | `offset_on` |   0.650525 | 0.634190 |  0.634190 |   1.0000 | 0.80 |

Interpretation:

```text
QPROJ discovers the witness manifold in real QPU records.
GPROJ reproduces the controlled witness family.
GEO computes the exact closed-form reference.
```

The paths are not expected to be numerically identical. They are expected to preserve the same condition ordering and control behavior.

---

## Control collapse

The primary destructive control is:

```text
independent_bit_shuffle
```

It preserves q0/q1 marginal distributions while breaking same-shot q0/q1 pairing.

Current final benchmark control values:

| Substrate | Condition   | Projection |  Control |   Drop |
| --------- | ----------- | ---------: | -------: | -----: |
| `QPROJ`   | `null`      |   0.012762 | 0.008890 | 30.34% |
| `QPROJ`   | `base_only` |   0.220018 | 0.083027 | 62.26% |
| `QPROJ`   | `offset_on` |   0.208717 | 0.100915 | 51.65% |
| `GPROJ`   | `null`      |   0.010326 | 0.008522 | 17.47% |
| `GPROJ`   | `base_only` |   0.295522 | 0.089097 | 69.85% |
| `GPROJ`   | `offset_on` |   0.298216 | 0.055686 | 81.33% |

Interpretation:

```text
Same-shot pairing is load-bearing in active D_M records.
Null conditions have little active structure to collapse.
```

---

## Corruption boundary

Probe 23 showed that single forbidden faults often do **not** collapse D_M.

Probe 24 reframed this as dimensional error correction:

```text
single faults may repair
compound faults should reveal the collapse boundary
```

The active manifolds cross the collapse threshold at approximately:

```text
k = 2 to 3 independent faults
```

Core reading:

```text
allowed channel re-descriptions survive
single structural faults often repair
compound corruption weakens progressively
active manifolds cross the collapse boundary
null manifolds have no comparable structure to collapse
```

Load-bearing corruption families include:

```text
independent_bit_shuffle
reciprocal_break
delay_permute
non_equivalence_channel_corruption
cross_rung_delay_scramble
```

This corruption-boundary result is the strongest D_M control story.

---

## QPU base workflow

Fresh QPU bases can be generated through the D_M QPU tool.

Typical workflow:

```bash
python d_m_qpu_generate.py submit
```

Then, after the IBM Runtime job completes:

```bash
python d_m_qpu_generate.py dump <JOB_ID>
```

Expected output:

```text
data/dm_data_bell_listener_cavity_offset_<JOB_ID>.npz
data/latest_dm_qpu_data.json
```

The QPU base is consumed by:

```text
d_m_benchmark.py
probes/07_dm_probe_bell_listener.py
probes/08_dm_probe_directional_witness.py
probes/09_dm_probe_pi_phase_witness.py
probes/10_dm_probe_qproj_dimensional_entanglement.py
probes/21_dm_probe_04w_windowed_phase_trajectory.py
probes/23_dm_probe_dimensional_invariance_controls.py
probes/24_dm_probe_corruption_boundary.py
```

A QPU base is the frozen hardware record. Do not mutate it after dumping. Generate a new base if the circuit changes.

---

## GPU base workflow

Generate a GPU-compatible controlled witness base with:

```bash
python d_m_gpu_generate.py
```

Expected output:

```text
data/dm_gpu_data_null_4096shots_seed<SEED>.npz
data/dm_gpu_data_base_delay_4096shots_seed<SEED>.npz
data/dm_gpu_data_offset_deformed_4096shots_seed<SEED>.npz
data/latest_dm_gpu_data.json
```

The GPU base is not an IBM hardware simulator. It is a controlled Bell-witness record generator designed to preserve the discovered D_M schema and condition family while separating:

```text
operator behavior
```

from:

```text
hardware-specific shot noise and backend drift
```

---

## GEO workflow

The GEO path is the exact closed-form classical reference.

It does not sample shots.

It computes the D_M witness manifold directly from condition metadata:

```text
x_space = normalize(log1p(base_delay))
x_time  = normalize(log1p(base_delay + mean_offset))
x_dm    = sqrt((w_space*x_space^2 + w_time*x_time^2)/(w_space+w_time))
cos(2*phi) = 2*x_time - 1
YZ      = E*cos(phi)
ZY      = -E*sin(phi)
E       = energy_floor + energy_scale*x_dm^energy_gamma
```

The null condition is an exact zero manifold.

Run the GEO precision probe:

```bash
python probes/25_dm_probe_geo_precision_reference.py
```

Typical output:

```text
probes/analysis/dm_probe25_geo_precision_<timestamp>/
    result.json
    probe25_summary.csv
    probe25_rung_projection.csv
    probe25_validation.csv
    probe25_cpu_gpu_agreement.csv
```

The final benchmark calls the same locked exact-GEO rule.

---

## CUDA kernels

The core CUDA source is:

```text
kernels/dm_projector_kernel.cu
```

Kernel roles include:

```text
record pair compression
same-shot bit-shuffle control
rung-level witness projection
summary-vector projection
raw-signal pair projection
exact GEO reference
legacy GEO compatibility
DER retrieval appendix support
```

The current benchmark uses one compiled CUDA module so shared device helpers stay consistent across the qproj path, gproj path, geo path, controls, and capstone benchmark.

---

## What to look for

A clean `D_M` run should show:

```text
QPROJ active conditions above QPROJ null
GPROJ active conditions above GPROJ null
GEO active conditions above exact GEO null
active record paths collapsing under independent_bit_shuffle
GEO null exactly zero
GEO active pi score near 1.0
allowed channel re-descriptions preserved
compound corruption boundary around k=2 to k=3
```

A clean control history should show:

```text
same-shot pairing is load-bearing
reciprocal structure is load-bearing
delay order is load-bearing
single faults often repair
compound faults cross a collapse boundary
```

If active qproj/gproj conditions do not weaken under independent-bit shuffle, the run is probably not measuring the same D_M operator.

---

## Files produced by the pipeline

Common generated files:

```text
data/dm_data_bell_listener_cavity_offset_<JOB_ID>.npz
data/dm_gpu_data_<...>.npz
data/latest_dm_qpu_data.json
data/latest_dm_gpu_data.json
analysis/d_m_final_capstone_<timestamp>/result.json
analysis/d_m_final_capstone_<timestamp>/final_claim_report.md
analysis/d_m_final_capstone_<timestamp>/verify_projection_summary.csv
analysis/d_m_final_capstone_<timestamp>/verify_rung_projection.csv
analysis/d_m_final_capstone_<timestamp>/verify_condition_separation.csv
analysis/d_m_final_capstone_<timestamp>/verify_substrate_agreement.csv
analysis/d_m_final_capstone_<timestamp>/verify_control_collapse.csv
analysis/d_m_final_capstone_<timestamp>/dimensional_invariance_summary.csv
analysis/d_m_final_capstone_<timestamp>/forbidden_fault_summary.csv
analysis/d_m_final_capstone_<timestamp>/corruption_boundary_summary.csv
analysis/d_m_final_capstone_<timestamp>/geo_exact_validation.csv
analysis/d_m_final_capstone_<timestamp>/artifacts.npz
```

Recommended `.gitignore` patterns:

```gitignore
data/dm_data_bell_listener_cavity_offset_*.npz
data/dm_gpu_data_*.npz
analysis/
probes/analysis/
*_report.json
```

Keep small curated fixtures if they are part of the reproducibility story.

Keep large generated bases out of git unless intentionally shipping them.

---

## Script map

```text
d_m_benchmark.py
    Current capstone runner:
    qproj/gproj/exact-GEO verification + controls + corruption boundary.

d_m_qpu_generate.py
    Submit and dump real QPU Bell-listener base jobs.

d_m_gpu_generate.py
    Generate GPU-compatible controlled Bell-witness bases.

probes/00_dm_probe_prune.py
    First prune of failed dimensional-transport assumptions.

probes/01_dm_probe_channel_extract.py
    Candidate channel extraction from the frozen QPU scene.

probes/02_dm_probe_synthetic_maps.py
    Failed D_M-as-compression rehearsal.

probes/03_dm_probe_linear_scheduler.py
    Failed linear-channel scheduler rehearsal.

probes/04_dm_probe_calibrated_seed_projection.py
    Failed QPU-calibrated seed projection rehearsal.

probes/05_dm_probe_base10_state_selector.py
    Failed base-10 state-selector rehearsal.

probes/06_dm_probe_single_calibrated_projector.py
    Failed single-projector rehearsal.

probes/07_dm_probe_bell_listener.py
    Bell-listener pivot.

probes/08_dm_probe_directional_witness.py
    Directional YZ/ZY witness characterization.

probes/09_dm_probe_pi_phase_witness.py
    Pi-phase witness probe.

probes/10_dm_probe_qproj_dimensional_entanglement.py
    QPROJ dimensional-entanglement projection task-lock.

probes/11_dm_probe_cuda_qproj_harness.py
    CUDA qproj/gproj projector harness.

probes/12_dm_probe_benchmark.py
    Early benchmark runner, superseded.

probes/13_dm_probe_benchmark2.py
    First substantive benchmark with DER.

probes/14_dm_probe_holographic_projection.py
    GPT-2 pre-softmax QK compatibility probe.

probes/15_dm_probe_native_stagger.py
    Native-stagger null sidequest.

probes/16_dm_probe_convert_corpus.py
    Corpus conversion utility.

probes/17_dm_probe_raw_boundary_projection.py
    Raw-boundary CUDA projection path.

probes/18_dm_probe_raw_boundary_gpu_benchmark.py
    Raw-boundary GPU-only benchmark.

probes/19_dm_probe_task_utility_retrieval_gpu.py
    Saturated task-utility retrieval; appendix only.

probes/20_dm_probe_make_random_null_corpus.py
    Random null corpus control for retrieval saturation.

probes/21_dm_probe_04w_windowed_phase_trajectory.py
    Windowed phase trajectory and bit-shuffle control.

probes/22_dm_probe_task_utility_retrieval_rawsignal.py
    Probe-22 rawsignal path; source of mature qproj/gproj kernel path.

probes/23_dm_probe_dimensional_invariance_controls.py
    Allowed-channel invariance and forbidden single-fault controls.

probes/24_dm_probe_corruption_boundary.py
    Compound corruption / collapse-boundary probe.

probes/25_dm_probe_geo_precision_reference.py
    Exact closed-form GEO reference.

probes/D_M_probe_process_record.md
    Probe-level process record.

kernels/dm_projector_kernel.cu
    Shared CUDA implementation and device helpers.
```

---

## Current bounded claim

`D_M` is a live research object, but this operator package is complete for this version.

The current bounded claim is:

```text
D_M is a dimensional witness-manifold projection operator
with three-substrate expression:
  1. real QPU Bell-listener qproj base,
  2. GPU-generated controlled gproj base,
  3. exact closed-form classical geo reference.
```

The current benchmark evidence is:

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

The honest framing is:

```text
D_M does not certify Bell nonlocality.
D_M does not reconstruct density matrices.
D_M does not prove prepared Bell states.
D_M is not a QPU speedup or quantum-advantage claim.
GPROJ is not an IBM hardware simulator.
GEO is a closed-form reference, not a hardware simulator.
GPT-2 is not a D_M input.
D_M is useful as a controlled dimensional witness-manifold operator with qproj/gproj/geo substrate linkage.
```

That is the claim to defend.

---

## Next development steps

Likely next steps:

```text
repeat QPU runs across multiple IBM backends/calibrations
construct a structural-null base to resolve the native-stagger ambiguity
build a task that separates D_M from flat cosine
rerun the corruption boundary across more seeds and shot windows
characterize qproj attenuation as a hardware-quality readout
formalize or remove the pi-adic toy diagnostics
integrate D_M as a feature channel in the node-point network
document the exact gate-level QPU circuit in a dedicated circuit spec
```

The process is the process.

Break it, fix it, document what happened.
