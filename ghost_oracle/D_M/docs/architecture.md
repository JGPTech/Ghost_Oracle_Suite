# Architecture

Architecture for the `D_M` operator package.

`D_M` is the **Dimensional Entanglement Projection** channel. It is a completed operator package in the Ghost Oracle Suite operator-family pattern.

This document replaces the older architecture framing that treated `D_M` as a dimensional-compression projector. The new framing is:

```text
D_M is a finished operator package for this version.
D_M is one channel in the larger Ghost Oracle Suite roadmap.
D_M uses the standard qproj / gproj / geo substrate pattern.
D_M claims are made only through the benchmark runner and controls.
D_M is not a dimensionality-reduction benchmark.
D_M is not a Bell-nonlocality certification claim.
```

The math is in:

```text
docs/math.md
```

The process history is in:

```text
PROCESS_RECORD.md
probes/D_M_probe_process_record.md
```

The examples and probe directories preserve continuity paths, but the current architecture claim comes from:

```text
d_m_benchmark.py
```

This architecture document explains how the finished `D_M` operator is wired together, what each file is responsible for, what counts as a valid benchmark claim, and how the package fits into the larger Ghost Oracle Suite direction without inflating the current result beyond what the benchmark supports.

---

## 1. Architectural status

`D_M` is complete for this version.

That means the package now has:

```text
1. a real QPU Bell-listener record path,
2. a GPU-generated controlled Bell-witness base path,
3. an exact closed-form GEO reference path,
4. a canonical capstone benchmark runner,
5. CUDA-accelerated qproj/gproj record projection,
6. a same-shot bit-shuffle destructive control,
7. allowed channel-invariance checks,
8. forbidden single-fault checks,
9. compound corruption-boundary checks,
10. exact GEO CPU/GPU validation,
11. documented math,
12. documented process history,
13. bounded claims and explicit non-claims.
```

The current package is not a placeholder for future work. It is a finished operator implementation with a bounded claim.

The bounded claim is:

```text
D_M is a dimensional witness-manifold projection operator.
```

More specifically:

```text
D_M projects a YZ-primary / ZY-reciprocal dimensional witness manifold
across qproj, gproj, and exact-GEO substrates.

Active base-delay / offset manifolds separate from null.

Same-shot pairing, reciprocal structure, and delay order are load-bearing.

Compound corruptions cross a measurable collapse boundary.
```

The claim is not:

```text
D_M certifies device-independent Bell nonlocality.
D_M reconstructs density matrices.
D_M proves prepared Bell states.
D_M proves quantum advantage.
D_M is a QPU speedup claim.
D_M is an IBM hardware simulator.
D_M is a general retrieval utility claim.
D_M is a dimensional-compression benchmark.
```

The architecture exists to keep that distinction clear.

---

## 2. Ghost Oracle framing

The larger Ghost Oracle Suite roadmap frames ghost-channel operators as controlled listeners for hidden structure.

A transformer asks:

```text
What representation, token, action, or score should come next?
```

A Ghost Oracle operator asks:

```text
What hidden structure exists around that score or record,
and does it survive controls?
```

Within that system, `D_M` is the dimensional witness-manifold component.

```text
Ghost Oracle operator stack
├── G_M        Generalized Metric / geometry channel
├── S_M        Syndrome Metric channel
├── T_S        Temporal Stress / stress channel
├── F_M        Paired-path differential wave channel
└── D_M        Dimensional Entanglement Projection channel
```

`D_M` is not the entire Ghost Oracle Suite.

`D_M` is the Bell-witness dimensional-manifold operator in that stack.

Its role is:

```text
bare two-qubit witness listening
YZ-primary / ZY-reciprocal manifold projection
delay-ordered rung measurement
same-shot pairing tests
allowed channel re-description tests
compound corruption-boundary measurement
qproj / gproj / geo substrate comparison
```

Transformer-adjacent analogues include:

```text
attention-pair residue tests
directional manifold probes
hidden-structure listeners
control-collapse diagnostics
delay/order sensitivity tests
```

The `D_M` package does not replace transformer components. It defines and benchmarks a dimensional witness-manifold operator that can later inform larger Ghost Oracle / Converger components.

---

## 3. Standard operator package pattern

The current operator-family architecture uses the same package shape for each completed operator:

```text
operator/
├── operator_gpu_generate.py
├── operator_qpu_generate.py
├── operator_benchmark.py
├── kernels/
├── docs/
├── examples/
├── data/
└── probes/
```

For `D_M`, the current package uses:

```text
D_M/
├── d_m_gpu_generate.py
├── d_m_qpu_generate.py
└── d_m_benchmark.py
```

Supporting directories:

```text
D_M/
├── data/
├── docs/
├── examples/
├── kernels/
└── probes/
```

The current main entry points are:

```bash
python d_m_benchmark.py
python d_m_benchmark.py --repair-metadata
python d_m_benchmark.py --reps 20
```

Base generation:

```bash
python d_m_gpu_generate.py
python d_m_qpu_generate.py submit
python d_m_qpu_generate.py dump <JOB_ID>
```

Research probes remain in the repo:

```bash
python probes/07_dm_probe_bell_listener.py
python probes/08_dm_probe_directional_witness.py
python probes/21_dm_probe_04w_windowed_phase_trajectory.py
python probes/23_dm_probe_dimensional_invariance_controls.py
python probes/24_dm_probe_corruption_boundary.py
python probes/25_dm_probe_geo_precision_reference.py
```

Those probes are retained for continuity and process history. They are not the current source of benchmark claims.

Current benchmark claims should come from:

```text
d_m_benchmark.py
```

and from saved benchmark output under:

```text
analysis/d_m_final_capstone_<timestamp>/
```

---

## 4. Standard substrate paths

`D_M` follows the standard three-substrate pattern used by the Ghost Oracle Suite.

```text
D_M^{qproj}
D_M^{gproj}
D_M^{geo}
```

In GitHub-safe plain text:

```text
D_M_qproj  = real QPU Bell-listener shot record
D_M_gproj  = GPU-generated controlled Bell-witness record
D_M_geo    = exact closed-form classical GEO reference
```

### 4.1 QPU path

The `qproj` path uses real IBM Runtime Bell-listener records.

```text
D_M_qproj(B_q)
```

where `B_q` is a dumped QPU base file.

The QPU generator places bare two-qubit listener tiles across the chip:

```text
H(q0), H(q1)
    -> delay
    -> rotate into witness basis
    -> measure
```

Each tile records one witness channel:

```text
XY
YZ
ZY
YX
```

The qproj path is the hardware-derived witness substrate.

It is not used to claim raw throughput superiority. It is used to test whether a real QPU record contains a delay-ordered, same-shot-pair-dependent witness manifold that survives the correct controls.

### 4.2 GPU-generated path

The `gproj` path uses a GPU-generated D_M base.

```text
D_M_gproj(B_g)
```

where `B_g` is a GPU-generated base file.

The GPU base is not an IBM hardware simulator. It is a controlled Bell-witness record generator with the same analysis schema as the QPU dump.

Its purpose is to separate:

```text
operator behavior
```

from:

```text
hardware noise, drift, backend calibration, queue timing, and physical execution effects
```

### 4.3 Exact GEO path

The `geo` path is the exact closed-form classical reference.

```text
D_M_geo(c)
```

where `c` is one of the canonical conditions:

```text
null
base_only
offset_on
```

The GEO path does not sample shots.

It computes the D_M witness manifold directly from condition metadata.

The exact GEO path is not presented as a hardware simulator.

Its purpose is:

```text
provide a clean analytic reference
exercise the same summary and reporting path
give the benchmark an exact zero null manifold
give the benchmark a closed-form active reference
```

The reference field is the clean operator ceiling, not the hardware claim.

---

## 5. Bases

A **base** is a `.npz` file containing D_M witness records.

The shared base schema is:

```text
schema                  : str
suite                   : str
operator                : str
substrate               : str
job_id                  : str
backend                 : str, optional
shots                   : int
num_tiles               : int

pair                    : uint8, shape (tiles, shots, 2)

tile_rung_index         : int32, shape (tiles,)
tile_witness_index      : int32, shape (tiles,)
tile_base_delay_dt      : int32, shape (tiles,)
tile_offset_dt          : int32, shape (tiles,)
tile_total_delay_dt     : int32, shape (tiles,)
```

Compatibility fields may include:

```text
pair_tile{t}            : uint8, shape (shots, 2)
tile_witness_label      : str array
tile_basis_q0           : str array
tile_basis_q1           : str array
basis                   : str array
```

The same analysis schema is used for:

```text
real QPU D_M bases
GPU-generated D_M bases
benchmark record projection
control projection
```

This shared schema is load-bearing.

It allows the same benchmark code and CUDA kernels to consume QPU and GPU bases without changing the operator.

Generated QPU base files use the pattern:

```text
data/dm_data_bell_listener_cavity_offset_<JOB_ID>.npz
```

Generated GPU base files use patterns such as:

```text
data/dm_gpu_data_null_4096shots_seed<SEED>.npz
data/dm_gpu_data_base_delay_4096shots_seed<SEED>.npz
data/dm_gpu_data_offset_deformed_4096shots_seed<SEED>.npz
```

Benchmark output is written under:

```text
analysis/d_m_final_capstone_<timestamp>/
```

---

## 6. Canonical conditions

The current D_M package uses three canonical conditions.

| Condition | Base delays dt | Offset dt | Meaning |
|---|---:|---:|---|
| `null` | `[0, 0, 0, 0, 0]` | `0` | no explicit delay / offset |
| `base_only` | `[0, 256, 1024, 4096, 16384]` | `0` | base-delay ladder only |
| `offset_on` | `[0, 256, 1024, 4096, 16384]` | `128` | base-delay plus tile offset |

The locked QPU capstone bases are:

| Condition | Job ID |
|---|---|
| `null` | `d8fm4ihvjngc73aq3ccg` |
| `base_only` | `d8flk2jo3njc73f0g560` |
| `offset_on` | `d8fl82bo3njc73f0fgd0` |

The `null` condition is an explicit-delay null.

It is not guaranteed to be a fully structureless hardware null because real devices may preserve native layout, schedule, calibration, or readout-order structure. Probe 15 records this ambiguity.

The exact GEO `null` is different:

```text
D_M_geo(null) = exact zero manifold
```

This gives the capstone an exact classical reference floor.

---

## 7. Witness manifold

`D_M` is built from two-bit shot records grouped by rung and witness channel.

Each tile produces:

```text
pair[tile, shot, 2]
```

The bit-to-spin map is:

```text
0 -> +1
1 -> -1
```

For a Pauli-pair witness channel `PQ`, define the connected correlator:

```text
C(PQ) = <P0 P1> - <P0><P1>
```

The four witness channels are:

```text
XY
YZ
ZY
YX
```

The canonical D_M coordinate frame is:

```text
Y = C(YZ)
Z = C(ZY)
R = -Z
```

The comparison channel is:

```text
C_cmp = sqrt(C(XY)^2 + C(YX)^2)
```

The directional energy is:

```text
E = sqrt(Y^2 + R^2)
```

The directional specificity is:

```text
S = E - C_cmp
```

The phase coordinate is:

```text
phi = atan2(R, Y) mod pi
```

The projected rung vector is:

```text
rung =
[
    XY,
    YZ,
    ZY,
    YX,
    YZ_primary,
    ZY_return,
    YZ_ZY_energy,
    comparison_energy,
    directional_specificity,
    directional_gap,
    inversion,
    pi_phase,
    pi_cos2,
    pi_sin2,
    base_delay,
    offset,
    total_delay,
    count_all,
    count_yzzy
]
```

This makes D_M a one-dimensional witness manifold with separable components:

```text
spatial / distance witness component
temporal / offset-ordering component
reciprocal channel component
```

The operator is not two independent channels. `YZ` and `ZY` are a canonical coordinate pair used to express one dimensional witness manifold.

---

## 8. Canonical coordinate frame and channel invariance

The canonical reporting frame is:

```text
YZ = primary witness dimension
ZY = reciprocal / inverted return dimension
```

This frame is useful for naming, plotting, and reporting.

However, the control history shows that the four witness labels are not arbitrary destructive levers.

Probe 21 showed:

```text
independent_bit_shuffle collapses active phase structure
witness_label_shuffle does not collapse active phase structure
```

Probe 23 then reframed this as dimensional invariance:

```text
allowed channel re-descriptions should preserve D_M
forbidden non-equivalent corruptions should weaken D_M
```

The architecture therefore distinguishes:

```text
canonical coordinate frame
```

from:

```text
allowed equivalent descriptions of the same manifold
```

A valid D_M control cannot treat every witness relabeling as destructive.

Allowed channel re-descriptions are not falsifying corruptions of the dimensional manifold.

---

## 9. Current CUDA architecture

The core CUDA source is:

```text
kernels/dm_projector_kernel.cu
```

The current benchmark uses this kernel through CuPy RawModule.

CUDA is required for the final capstone benchmark. There is no silent CPU fallback in the repo-facing capstone path.

The production kernels are:

| Kernel | Role |
|---|---|
| `dm_tile_correlator_kernel_u8` | Compress `pair[tile, shot, 2]` into tile-level connected correlators. |
| `dm_independent_bit_shuffle_tile_kernel_u8` | Same-shot-pair destructive control. |
| `dm_rung_projection_kernel_f32` | Compress tile stats into rung-level D_M manifold stats. |
| `dm_projection_summary_kernel_f32` | Compress rung stats into a base-level summary vector. |
| `dm_make_projected_pair_from_bits_kernel_u8` | Raw-signal path: raw bit bank XOR base pair. |
| `dm_geo_exact_rung_projection_kernel_f32` | Exact closed-form GEO rung reference. |
| `dm_geo_exact_sweep_summary_kernel_f32` | Exact GEO batched summary sweep. |
| `dm_geo_rung_projection_kernel_f32` | Legacy analytic/aperture GEO retained for old probes. |
| `dm_geo_sweep_summary_kernel_f32` | Legacy GEO sweep retained for old probes. |
| `dm_der_topk_kernel_f32` | DER top-k retrieval kernel retained for appendix/probes. |

### 9.1 Record projection path

The qproj/gproj record path is:

```text
pair[tile, shot, 2]
    -> dm_tile_correlator_kernel_u8
    -> tile_stats[tile, metric]
    -> dm_rung_projection_kernel_f32
    -> rung_stats[rung, metric]
    -> dm_projection_summary_kernel_f32
    -> summary[metric]
```

The destructive bit-shuffle control follows the same path, replacing the tile correlator with:

```text
dm_independent_bit_shuffle_tile_kernel_u8
```

This preserves q0/q1 marginals while breaking same-shot q0/q1 pairing.

### 9.2 Raw-signal path

The Probe-22 raw-signal path adds one fused projection step:

```text
raw bit_bank + base_pair
    -> projected_pair = data_pair XOR base_pair
    -> record projection path
```

Kernel:

```text
dm_make_projected_pair_from_bits_kernel_u8
```

This path is retained because it is the mature qproj/gproj kernel source used by the final capstone implementation.

### 9.3 Exact GEO path

The Probe-25 exact GEO path bypasses shot records:

```text
condition metadata
    -> exact D_M rung manifold
    -> shared summary kernel
```

Kernel:

```text
dm_geo_exact_rung_projection_kernel_f32
```

The exact GEO path uses the same summary vector as qproj/gproj.

This is important: `geo` is not a separate scoring system. It is an exact classical source for the same D_M summary interface.

### 9.4 CUDA boundary

Included:

```text
pair compression
connected correlators
rung projection
summary projection
same-shot bit-shuffle control
raw-signal projected pair construction
exact GEO reference
DER appendix support
```

Excluded:

```text
density-matrix reconstruction
CHSH certification
device-independent Bell tests
GPT-2 attention execution
external LLM inference
```

Those belong to separate tools or future work.

---

## 10. Projection summary

The base-level summary vector contains:

```text
n_rungs
yz_mean
yz_pos_frac
zy_mean
zy_inverted_frac
yzzy_energy_mean
yzzy_energy_max
specificity_mean
specificity_max
pi_periodic_score
pi_periodic_mode
energy_tracking_r
specificity_tracking_r
phase_velocity_r
phase_span_pi_units
projection_score
```

The summary is not a proof of entanglement.

It is a bounded projection score for benchmark ranking and control comparison.

The capstone score uses the CUDA summary path consistently across qproj, gproj, and geo. The legacy Probe 11 CPU/CUDA scalar-summary mismatch is recorded in the process record, but the final benchmark uses one shared CUDA summary path for its claims.

---

## 11. Exact GEO reference

The exact GEO reference is the closed-form classical path discovered and locked in Probe 25.

For active conditions, define normalized axes:

```text
x_space = normalize(log1p(base_delay))
x_time  = normalize(log1p(base_delay + mean_offset))
```

Combine them into a D_M coordinate:

```text
x_dm = sqrt((w_space*x_space^2 + w_time*x_time^2) / (w_space + w_time))
```

Use the time axis to define phase:

```text
cos(2*phi) = 2*x_time - 1
```

Define energy:

```text
E = energy_floor + energy_scale*x_dm^energy_gamma
```

Then write the witness pair:

```text
YZ = E*cos(phi)
ZY = -E*sin(phi)
```

The null condition is exact:

```text
D_M_geo(null) = 0
```

The current capstone GEO values are:

| Condition | Projection | E_mean | S_mean | pi score |
|---|---:|---:|---:|---:|
| `null` | `0.000000` | `0.000000` | `0.000000` | `0.0000` |
| `base_only` | `0.693827` | `0.675090` | `0.675090` | `1.0000` |
| `offset_on` | `0.650525` | `0.634190` | `0.634190` | `1.0000` |

The exact GEO path is the reference path, not the physical-hardware claim.

---

## 12. Field controls and corruption controls

The benchmark uses multiple control families.

### 12.1 Independent bit shuffle

Control:

```text
independent_bit_shuffle
```

Preserves:

```text
q0 marginal distribution
q1 marginal distribution
witness/tile/rung metadata
```

Destroys:

```text
same-shot q0/q1 pairing
```

This is the primary destructive control for qproj/gproj records.

A clean D_M run should show active qproj/gproj projections weaken strongly under this control.

### 12.2 Allowed channel re-descriptions

Allowed transformations include:

```text
equiv_pair_swap
equiv_reciprocal_swap
equiv_cyclic_rotation
```

These test whether the dimensional manifold can be re-expressed without being destroyed.

A clean D_M run should show strong retention under allowed equivalent descriptions.

### 12.3 Forbidden single faults

Forbidden single-fault transformations include:

```text
reciprocal_break
cross_rung_delay_scramble
same_label_wrong_delay
non_equivalence_channel_corruption
independent_bit_shuffle
```

Probe 23 showed that single forbidden faults often do not fully collapse D_M.

This is not hidden.

The architecture interprets it as evidence that D_M has a repairable dimensional-agreement structure.

### 12.4 Compound corruption boundary

Probe 24 applies compound corruptions at increasing depth:

```text
k = number of independent faults
```

A clean D_M boundary shows:

```text
single faults may repair
compound faults progressively weaken the manifold
active bases cross the collapse threshold around k=2 to k=3
null bases have no comparable structure to collapse
```

This is the strongest D_M control story.

---

## 13. Canonical data flow

End-to-end `D_M` package flow:

```text
QPU hardware path
    d_m_qpu_generate.py submit
        -> IBM Runtime job
        -> metadata JSON

    d_m_qpu_generate.py dump <JOB_ID>
        -> completed job result
        -> data/dm_data_bell_listener_cavity_offset_<JOB_ID>.npz

GPU/generated path
    d_m_gpu_generate.py
        -> generated controlled Bell-witness bases
        -> data/dm_gpu_data_<condition>_4096shots_seed<SEED>.npz

Exact GEO path
    condition metadata
        -> dm_geo_exact_rung_projection_kernel_f32
        -> shared D_M summary vector

Shared benchmark path
    d_m_benchmark.py
        -> load qproj/gproj bases
        -> run exact GEO reference
        -> run qproj/gproj record projection
        -> run independent-bit-shuffle controls
        -> run invariance checks
        -> run forbidden-fault checks
        -> run corruption-boundary checks
        -> write result JSON / CSV / final claim report / artifacts
```

Inside the benchmark:

```text
pair[tile, shot, 2]
    -> controls
    -> tile stats
    -> rung stats
    -> summary
    -> condition separation
    -> substrate agreement
    -> control collapse
    -> corruption boundary
    -> saved outputs
```

The current architecture intentionally keeps data movement simple:

```text
base files in data/
CUDA kernels in kernels/
current docs in docs/
legacy examples in examples/
research probes in probes/
current claims from d_m_benchmark.py
```

---

## 14. Current benchmark stages

The canonical benchmark has four major stages.

```text
1. substrate loading
2. qproj/gproj/exact-GEO verification
3. destructive and invariance controls
4. claim report / artifact emission
```

### 14.1 Substrate loading

The benchmark loads available substrates:

```text
QPROJ
GPROJ
GEO
```

Current capstone run:

```text
QPROJ null      : real IBM QPU record
QPROJ base_only : real IBM QPU record
QPROJ offset_on : real IBM QPU record

GPROJ null      : GPU-generated controlled base
GPROJ base_only : GPU-generated controlled base
GPROJ offset_on : GPU-generated controlled base

GEO null        : exact closed-form zero reference
GEO base_only   : exact closed-form active reference
GEO offset_on   : exact closed-form active reference
```

### 14.2 Verification

Verification asks:

```text
Do active conditions separate from null across qproj, gproj, and geo?
```

Current capstone values:

| Substrate | `null` | `base_only` | `offset_on` |
|---|---:|---:|---:|
| `QPROJ` | `0.012762` | `0.220018` | `0.208717` |
| `GPROJ` | `0.010326` | `0.295522` | `0.298216` |
| `GEO` | `0.000000` | `0.693827` | `0.650525` |

The read:

```text
active conditions separate from null on every substrate
qproj is attenuated relative to gproj/geo
geo is the exact classical reference
```

### 14.3 Control collapse

Control collapse asks:

```text
Does breaking same-shot pairing weaken active records?
```

Current capstone values:

| Substrate | Condition | Projection | Control | Drop |
|---|---|---:|---:|---:|
| `QPROJ` | `base_only` | `0.220018` | `0.083027` | `62.26%` |
| `QPROJ` | `offset_on` | `0.208717` | `0.100915` | `51.65%` |
| `GPROJ` | `base_only` | `0.295522` | `0.089097` | `69.85%` |
| `GPROJ` | `offset_on` | `0.298216` | `0.055686` | `81.33%` |

The read:

```text
same-shot pairing is load-bearing in active D_M records
```

### 14.4 Corruption boundary

Corruption boundary asks:

```text
How many independent corruptions can the manifold absorb before collapse?
```

Probe 24 result:

```text
active manifolds cross the collapse threshold around k=2 to k=3
null manifolds do not show the same collapse transition
```

The read:

```text
D_M behaves like a repairable dimensional-agreement manifold
single faults may repair
compound faults expose the collapse boundary
```

---

## 15. Current benchmark output

The benchmark writes:

```text
analysis/d_m_final_capstone_<timestamp>/
├── result.json
├── final_claim_report.md
├── verify_projection_summary.csv
├── verify_rung_projection.csv
├── verify_condition_separation.csv
├── verify_substrate_agreement.csv
├── verify_control_collapse.csv
├── dimensional_invariance_summary.csv
├── forbidden_fault_summary.csv
├── corruption_boundary_summary.csv
├── geo_exact_validation.csv
└── artifacts.npz
```

### 15.1 `result.json`

Full JSON record containing:

```text
config
base metadata
projection summaries
condition separation
substrate agreement
control collapse
invariance checks
forbidden-fault checks
corruption boundary
bounded claim
non-claims
```

### 15.2 `final_claim_report.md`

Human-readable capstone report.

This is the preferred artifact for quickly reviewing the finished D_M claim.

### 15.3 `verify_projection_summary.csv`

Projection summary rows for:

```text
qproj / gproj / geo
null / base_only / offset_on
```

### 15.4 `verify_rung_projection.csv`

Rung-level D_M manifold stats.

### 15.5 `verify_condition_separation.csv`

Active-vs-null separation metrics.

### 15.6 `verify_substrate_agreement.csv`

Substrate agreement diagnostics across qproj/gproj/geo.

### 15.7 `verify_control_collapse.csv`

Independent-bit-shuffle control-collapse rows.

### 15.8 `dimensional_invariance_summary.csv`

Allowed channel re-description retention.

### 15.9 `forbidden_fault_summary.csv`

Forbidden single-fault weakening diagnostics.

### 15.10 `corruption_boundary_summary.csv`

Compound corruption-depth collapse boundary.

### 15.11 `geo_exact_validation.csv`

Exact GEO validation rows.

### 15.12 `artifacts.npz`

Compact projection arrays used for reproducibility and follow-up analysis.

---

## 16. Valid claim boundary

The `D_M` architecture supports the following claims.

### Supported

```text
D_M has a defined dimensional witness-manifold object.
D_M has a real QPU qproj path.
D_M has a GPU-generated controlled gproj path.
D_M has an exact closed-form geo reference path.
The same benchmark compares qproj / gproj / geo.
Active base-delay / offset conditions separate from null.
Same-shot pairing is load-bearing.
Allowed channel re-descriptions can preserve the manifold.
Single forbidden faults may repair.
Compound corruptions cross a collapse boundary.
The CUDA kernel accelerates record projection while preserving the D_M boundary.
```

### Not supported

```text
D_M certifies device-independent Bell nonlocality.
D_M reconstructs density matrices.
D_M proves prepared Bell states.
D_M proves quantum advantage.
D_M is faster because of QPU hardware.
D_M is a universal dimensionality-reduction method.
D_M is a general retrieval utility benchmark.
D_M proves all future Ghost Oracle operators.
```

This is the claims discipline the architecture requires.

---

## 17. Why the architecture is finished for this version

`D_M` is finished for this version because every required operator-package element exists.

```text
operator definition          : complete
real QPU base path           : complete
GPU base generator           : complete
exact GEO reference path     : complete
canonical benchmark runner   : complete
CUDA projection kernel       : complete
destructive controls         : complete
invariance controls          : complete
corruption boundary          : complete
saved benchmark output       : complete
math documentation           : complete
process documentation        : complete
bounded claims               : complete
```

The next work is not to keep turning `D_M` into a catch-all container.

The next work is to use `D_M` as the completed dimensional witness-manifold operator while moving downstream ideas into their own packages or appendices.

Each future operator should follow the same discipline:

```text
define the operator
define qproj / gproj / geo
generate bases
benchmark under controls
scramble the channel
measure what survives
make only bounded claims
```

---

## 18. Why not...

### Why not call this Bell nonlocality?

Because the benchmark does not certify Bell nonlocality.

`D_M` reads Bell-witness correlators and projects a witness manifold. Certification would require a different experimental design and statistical claim boundary.

### Why not call this density-matrix reconstruction?

Because no density matrix is reconstructed.

The benchmark uses connected correlators and D_M manifold summaries.

### Why not call this a prepared Bell state?

Because the circuit is a bare two-qubit listener.

The QPU generator does not claim to prepare Bell states.

### Why not call this quantum advantage?

Because the benchmark does not show that.

The QPU path supplies real hardware records. The GPU and GEO paths do the computation. The claim is structure and control survival, not speedup.

### Why not trust the original compression task?

Because it failed.

Probes 00 through 06 tested D_M as a dimensional-compression projector and it did not beat random/PCA baselines under fair comparisons.

That failure is part of the process record and is why the Bell-listener pivot happened.

### Why not use retrieval as the headline task?

Because the retrieval tasks saturated.

Probe 13 tied flat cosine on the unperturbed DER task, and Probes 19/22 produced 100% top-1 for every method, including null. Retrieval remains an appendix/probe path, not the current final claim.

### Why not use only QPU data?

Because one hardware substrate alone does not establish the operator boundary.

The gproj and geo paths help separate:

```text
operator structure
```

from:

```text
hardware-specific noise and drift
```

### Why not use only GEO?

Because GEO is the exact reference, not the hardware record.

The QPU path is still required to show that the discovered witness structure appears in real QPU data.

### Why not treat every label shuffle as destructive?

Because the probe history says otherwise.

Witness-label reshuffling did not collapse the active manifold in the windowed phase test. The architecture therefore treats allowed channel re-descriptions as symmetry-like transformations, not destructive controls.

---

## 19. File responsibilities

```text
README.md
    Human-facing summary and current benchmark claims.

PROCESS_RECORD.md
    Repo-facing D_M process record.

docs/math.md
    Mathematical definition of connected correlators, witness coordinates,
    exact GEO, controls, corruption boundary, and benchmark metrics.

docs/architecture.md
    This document. System design for the finished D_M operator package.

d_m_benchmark.py
    Canonical benchmark runner for D_M.

d_m_gpu_generate.py
    GPU-generated controlled Bell-witness base generator.

d_m_qpu_generate.py
    Unified QPU submit/dump path.

kernels/dm_projector_kernel.cu
    CUDA projection kernels, control kernels, GEO kernels, and DER appendix kernel.

data/
    D_M base files, metadata, latest-file pointers, and optional curated fixtures.

examples/
    Supporting examples retained for continuity.

probes/
    Earlier analysis probes and development path, including the probe-level process record.
```

---

## 20. Summary

`D_M` is the completed Dimensional Entanglement Projection operator package for this version of Ghost Oracle Suite.

It demonstrates the dimensional witness-manifold version of the package pattern:

```text
qproj
gproj
geo
benchmark
controls
bounded claims
```

The current architecture says:

```text
D_M defines a dimensional witness manifold.
D_M uses bare two-qubit Bell-listener records.
D_M projects a canonical YZ-primary / ZY-reciprocal coordinate frame.
D_M compares QPU, GPU-generated, and exact GEO substrates.
D_M active conditions separate from null.
D_M active records collapse under same-shot bit shuffle.
D_M allowed channel re-descriptions preserve the manifold.
D_M compound corruptions cross a measurable collapse boundary.
D_M retrieval/utility results are not part of the default claim.
D_M uses CUDA to implement projection, not to inflate the claim.
```

The larger roadmap says:

```text
Use this completed pattern to build and compare the remaining ghost-channel operators.
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
