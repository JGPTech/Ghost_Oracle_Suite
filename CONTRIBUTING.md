# Contributing

This is a CC0 project. Anyone can fork, modify, redistribute, or rebuild this work without asking. The intent is the opposite of gatekeeping:

```text
build
break
fix
document
push it forward
```

What follows is how to contribute productively, not how to qualify.

Ghost Oracle Suite is now organized around standard operator packages. The current completed packages are:

```text
ghost_oracle/G_M/
ghost_oracle/S_M/
```

Each completed operator package follows the same pattern:

```text
operator/
├── README.md
├── operator_benchmark.py
├── operator_gpu_generate.py
├── operator_qpu_generate.py
├── data/
├── docs/
├── examples/
├── kernels/
└── probes/
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

The contribution standard is now:

```text
same operator
same input schema
same base-file structure
same controls
same benchmark metrics
three substrate paths
bounded claims only
```

---

## The one rule

**If you break something, you provide the fix.**

Not a hard requirement. Nobody will lock the gate against a bug report without a patch. But this project moves on fixes-with-bugs, not bugs alone.

A PR titled:

```text
X is wrong, here is the reproduction, here is the fix
```

gets reviewed.

An issue titled:

```text
X is wrong
```

sits in the queue until someone has time to reproduce it.

This includes the corollaries:

```text
No claims without code that demonstrates them.
No "this is broken" without a reproduction.
No "this should be different" without a proposed difference.
No benchmark claim without controls.
No hardware claim without the matched classical/GPU path.
```

The honest reason is simple: this is a small CC0 project with no funding. The most expensive resource is attention. A careful reproduction plus a proposed fix is the fastest way to move the work forward.

---

## Current package structure

Each operator package should keep this separation clean:

```text
README.md
  Human-facing summary and current benchmark claim.

operator_benchmark.py
  Canonical benchmark runner. Comparative claims live here.

operator_gpu_generate.py
  GPU/generated/noiseless base path.

operator_qpu_generate.py
  QPU/hardware submit/generate/dump path.

data/
  Generated or curated base files and metadata.

docs/
  math.md, architecture.md, known_issues.md.

examples/
  Supporting examples and continuity demos.

kernels/
  CUDA/C++ kernels used by the current package.

probes/
  Research trajectory, exploratory bridge work, and superseded paths.
```

Do not put a new benchmark claim in `examples/` or `probes/` and present it as the package result.

If the claim is meant to be current, it belongs in the package benchmark runner and the package README.

---

## Valid contribution formats

### 1. Operator package contribution

Use this when adding a new ghost-channel operator or promoting a probe into a finished package.

Expected shape:

```text
ghost_oracle/<OPERATOR>/
├── README.md
├── <operator>_benchmark.py
├── <operator>_gpu_generate.py
├── <operator>_qpu_generate.py
├── data/
├── docs/
│   ├── architecture.md
│   ├── known_issues.md
│   └── math.md
├── examples/
├── kernels/
└── probes/
```

Required docs:

```text
README.md
docs/math.md
docs/architecture.md
docs/known_issues.md
```

Required benchmark paths:

```text
geo
gproj
qproj
```

Required benchmark outputs:

```text
analysis/<operator>_<timestamp>/
├── result.json
├── summary.csv
├── per_feature.csv or per_component.csv
└── artifacts.npz
```

A finished operator package must define:

```text
1. the operator object
2. the input schema
3. the geo path
4. the gproj path
5. the qproj path
6. the controls
7. the benchmark metrics
8. the bounded claim
9. the non-claims
10. the known limits
```

---

### 2. Benchmark contribution

Use this when improving or extending a current package benchmark.

Benchmark PRs should include:

```text
the benchmark code
the exact command used
the saved output path
a summary table
the control condition
the interpretation
the non-claim
```

For `G_M`, current benchmark claims should come from:

```text
ghost_oracle/G_M/g_m_benchmark.py
```

For `S_M`, current benchmark claims should come from:

```text
ghost_oracle/S_M/s_m_benchmark.py
```

A benchmark PR should not only report the best row. It should also report what happened to the controls.

Good pattern:

```text
real signal improves
matched control collapses
destructive control damages the channel
baseline remains visible
failure mode is documented
```

Bad pattern:

```text
one high score, no controls
```

---

### 3. Probe contribution

Use this for exploratory research, reproductions, failed ideas, or bridge experiments.

Put probes in:

```text
ghost_oracle/<OPERATOR>/probes/
```

A probe should be one runnable file when practical.

Probe headers should include:

```text
Purpose
What this tests
Expected inputs
Outputs
Historical context
Current status
```

Use explicit status tags:

```text
CURRENT
EXPLORATORY
SUPERSEDED
KNOWN ISSUE
RETRACTED
LEGACY
```

If a probe supersedes or retracts an earlier framing, say so in the file header and update:

```text
docs/known_issues.md
PROCESS_RECORD.md
```

or the operator-specific process record if one exists.

Probes are allowed to fail. Silent failures are not.

---

### 4. Example contribution

Use this when adding a runnable demonstration that helps users understand a package.

Put examples in:

```text
ghost_oracle/<OPERATOR>/examples/
```

An example should:

```text
run end-to-end
have a clear command
write outputs under analysis/
avoid requiring private/local-only data unless documented
include a short interpretation
state what it does not prove
```

Example scripts are not the source of final claims unless promoted into the benchmark runner.

Good example framing:

```text
This is a projector testbed.
This demonstrates one operating regime.
This is not a state-of-the-art solver claim.
```

Bad example framing:

```text
This solves the whole domain.
```

---

### 5. Documentation contribution

Doc-only PRs are welcome.

Especially useful:

```text
math corrections
broken command fixes
known issue updates
repo tree fixes
README clarity
benchmark interpretation cleanup
claim boundary cleanup
```

Documentation should match the current code layout.

If the docs and code disagree, the code is the ground truth. But the preferred contribution is to update the docs so users do not have to discover that mismatch the hard way.

---

### 6. Kernel contribution

CUDA/C++ kernel changes are welcome, but they are load-bearing.

Kernel PRs should include:

```text
the kernel change
the Python call-site change if needed
a CPU/reference comparison
the benchmark before/after
the hardware used
the failure mode if CUDA is unavailable
```

For Windows/CuPy/NVRTC compatibility:

```text
Keep CUDA source comments ASCII-safe.
Avoid Unicode arrows, em dashes, and decorative symbols in .cu files.
```

This matters because Windows code-page handling can fail before NVRTC compilation.

If adding a new kernel, also add:

```text
--no-cuda fallback
--cuda-debug or equivalent diagnostics
clear kernel path printout
```

The kernel must not silently change the operator definition.

---

## Expected claim format

Every serious result should be written with three parts.

### Supported claim

```text
What the benchmark actually supports.
```

### Evidence

```text
Which command produced the result.
Which data/base files were used.
Which controls were run.
Which metrics changed.
Where the output was saved.
```

### Non-claim

```text
What the result does not prove.
```

Example:

```text
Supported:
S_M field-aware features separate real QPU records from destructive controls.

Evidence:
s_m_benchmark.py, QPROJ base <job>, windows [8,16,32,64],
real-vs-control Task A, sm_field/sm_all near 1.0 balanced accuracy,
raw_rates/detection_rates near chance.

Non-claim:
This is not a logical-error-rate benchmark and not a quantum advantage claim.
```

This style is preferred over hype.

---

## Substrate rules

Completed operators use three substrate paths:

```text
geo
gproj
qproj
```

Definitions:

```text
geo
  analytical, synthetic, or classical reference path

gproj
  GPU-generated, noiseless, or controlled base path

qproj
  QPU, hardware-derived, or hardware-calibrated base path
```

A result comparing substrates must use:

```text
same task
same inputs
same control set
same metric table
same benchmark runner
```

Do not compare a polished `geo` path to a stale `qproj` path and call it a substrate result.

Do not claim QPU behavior unless the QPU base file and dump path are documented.

Do not claim GPU/noiseless behavior unless the GPU generator and schema are documented.

---

## Data and fixture policy

Generated base files can be large.

Default policy:

```text
small curated fixtures may be committed
large generated bases should stay out of git unless intentionally shipped
latest_*.json is for local convenience, not published reproducibility
```

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

If your contribution relies on a base file, either:

```text
include a small fixture
provide the generation command
or document exactly where the file comes from
```

---

## Code conventions

These are not enforced by tooling, but they are the norms.

### Python

Use:

```python
argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
```

Preferred script layout:

```text
docstring
imports
paths/defaults
dataclasses/types
utility functions
load/generate functions
core operator functions
benchmark/evaluation functions
reporting/output functions
parse_args()
main()
if __name__ == "__main__":
    main()
```

Paths should be operator-local.

For example:

```text
S_M/data/
S_M/analysis/
S_M/kernels/
```

Do not make scripts depend on a root-level `data/` directory unless the package explicitly uses one.

### Output

Benchmark outputs should usually include:

```text
result.json
summary.csv
per_feature.csv or per_component.csv
artifacts.npz
```

If the benchmark has controls, include a control-specific table such as:

```text
control_collapse.csv
control_ablation.csv
substrate_agreement.csv
```

### Errors

Prefer clear failure messages over silent fallback.

If a fallback exists, print why:

```text
CUDA disabled by --no-cuda
CuPy import failed
kernel file not found
kernel compile failed
QPU base not found
GPU base not found
```

### Comments and docstrings

Document why the script exists, what claim it supports, and what it does not claim.

The best comments preserve intent, not just mechanics.

---

## Documentation conventions

Each completed operator should have:

```text
README.md
docs/math.md
docs/architecture.md
docs/known_issues.md
```

### README.md

Should include:

```text
operator identity
quick path
operator definition
repo structure
current entry points
data files
current capstone benchmark
controls
what to look for
bounded claim
next steps
```

### docs/math.md

Should include:

```text
definitions
derived quantities
operator equations
feature families
controls
metrics
claim boundary
```

### docs/architecture.md

Should include:

```text
package status
Converger framing
substrate paths
base schema
data flow
CUDA/kernel architecture if applicable
valid claim boundary
file responsibilities
```

### docs/known_issues.md

Should include:

```text
known bugs
superseded probes
retracted framings
current limitations
safe workarounds
open questions
```

---

## What good contributions look like

### Good: Adds a controlled benchmark

```text
Adds a benchmark mode.
Uses the same data schema.
Runs geo/gproj/qproj.
Includes destructive controls.
Writes summary.csv and result.json.
Updates README with bounded interpretation.
```

### Good: Fixes a known issue

```text
Selects an issue from docs/known_issues.md.
Provides reproduction.
Patches the script.
Adds a regression command.
Updates known_issues.md.
Adds process-record note if the issue affected prior claims.
```

### Good: Adds a probe

```text
Adds one self-contained script under probes/.
Explains historical context.
Reports both result and failure mode.
Does not overwrite older probes.
Updates the probes README if needed.
```

### Good: Adds an example

```text
Adds one runnable script under examples/.
Includes a clear command.
Uses local data or generation instructions.
Reports what it demonstrates.
States what it does not prove.
```

### Good: Improves docs

```text
Fixes stale paths.
Fixes repo tree.
Clarifies claim boundary.
Adds current commands.
Removes obsolete framing.
```

---

## What unhelpful contributions look like

```text
Issues without reproduction or patch attempt.
Claims without code.
Hardware claims without matched classical controls.
Quantum advantage claims without unimpeachable GPU/classical baselines.
PRs that add dependencies without justification.
PRs that silently change operator definitions.
PRs that rewrite historical probes and erase the research record.
Sweeping refactors that improve style but destroy trajectory legibility.
README hype that outruns benchmark evidence.
```

The project keeps wrong turns on purpose.

Do not delete the historical path just because it is messy. Mark it, supersede it, document it.

---

## How to claim a piece of work

There is no formal claim system.

If you want to work on something:

```text
1. Open an issue describing what you are going to do.
2. Submit a PR when you have something to show.
3. Include commands and outputs.
```

Draft PRs are welcome.

If two people work on the same thing in parallel, both PRs can be reviewed. The better one merges, or the useful pieces get combined.

This is CC0. The work belongs to nobody.

---

## Review and merging

Small project. One maintainer at the moment.

Reviews are direct and technical:

```text
Does it run?
Does the math match docs/math.md?
Does the benchmark output support the claim?
Were controls included?
Does it preserve package boundaries?
Does it update docs when behavior changes?
```

Likely outcomes:

```text
Merged
  works, docs updated, claim bounded

Comment with question/suggestion
  usually a small fix

Closed with reason
  conflicts with direction, breaks load-bearing behavior, or overclaims
```

If you disagree with a close, open an issue and make the case. If the case is good, it can be reopened.

---

## Licensing

The project is CC0 1.0 Universal — public domain dedication.

By contributing, you confirm your contribution is also CC0 or under a compatible license.

Compatible examples:

```text
CC0
MIT
BSD
Apache 2.0, if compatible with the repo's public-domain intent
```

Do not add copyright headers to new files.

The root `LICENSE` applies to the repo.

If you want attribution, the git log is the canonical record. A process-record note describing what you did and what you found is also welcome.

---

## Citing

The README says it: not asking for one.

If this work helps your work, the best thing you can do is push it forward:

```text
file an issue
send a PR
fork it
run a benchmark
break a claim
fix the break
document what happened
```

The process is the process.
