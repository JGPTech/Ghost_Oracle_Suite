# S_M Process Record — Syndrome Metric Operator

This document records the research and engineering trajectory of the `S_M` operator package inside Ghost Oracle Suite.

It is split from the larger Ghost Oracle process record so that `S_M` can stand on its own as a finished operator package for this version.

It is chronological. It includes the messy path. It includes the older stress-tensor framing, the TSP and token-retrieval bridge work, and the later cleanup that separated `S_M` from downstream operators. It is not a polished victory lap. It is a working record.

Current framing:

```text
S_M = Syndrome Metric
```

Current bounded claim:

```text
S_M is a syndrome-spacetime field operator.

It measures whether final data edge parity and repeated syndrome records form
a load-bearing field structure across synthetic/reference, GPU-generated, and
QPU-derived records.
```

Current non-claims:

```text
S_M is not a logical-error-rate benchmark.
S_M is not the T_S stress tensor.
S_M is not a token retrieval benchmark.
S_M is not a universal hardware advantage claim.
```

---

## Part 1 — From repetition-code dump trouble to syndrome-spacetime operator

The `S_M` path began after the original `G_M` benchmark trajectory.

It did not start as a clean planned operator search. It started the same way the original `G_M` path did:

```text
a thing that should have worked
did not work as expected
and then had to be interrogated instead of discarded
```

In this case, the first problem was practical.

A repetition-code dump script could not read a completed IBM Runtime job cleanly.

The old dumper expected metadata fields such as:

```text
num_blocks
rounds
logical_init
inject_qubit
block_layout
```

but some available metadata came from a different flag/superposition job format and did not contain the expected keys.

The immediate symptom was:

```text
KeyError: 'num_blocks'
```

The conclusion was not that the QPU job was bad.

The conclusion was that the `S_M` folder had become too tightly coupled to legacy job layouts and metadata formats.

The cleanup plan became:

```text
1. one script to submit or generate the S_M QPU job
2. one script to dump raw IBM Runtime data into a self-contained .npz
3. one analysis / benchmark path that consumes the .npz schema directly
```

This eventually evolved into the current package shape:

```text
S_M/
├── s_m_qpu_generate.py
├── s_m_gpu_generate.py
└── s_m_benchmark.py
```

---

## Part 2 — Raw Qiskit Runtime dump and schema stabilization

A robust raw-dump path was introduced to extract classical registers from Qiskit Runtime SamplerV2 results without assuming the old repetition-code layout.

The key Runtime object looked like:

```text
job.result()[0].data
```

which is a `DataBin`.

The robust dumper therefore needed to:

```text
list available classical register names
extract register arrays by name
save raw arrays into .npz
write metadata beside the output
```

This separated the fragile Qiskit Runtime API surface from the actual `S_M` analysis.

Once the `.npz` exists, no IBM Runtime connection is needed for downstream work.

The current shared S_M schema is:

```text
schema        : str
job_id        : str
backend       : str, optional
shots         : int
rounds        : int
flag_level    : int
basis         : str
init_state    : str
distances     : int array

data_d{d}     : uint8, shape (shots, d)
synd_d{d}     : uint8, shape (shots, rounds, d-1)
flag_d{d}     : optional uint8, shape (shots, rounds, n_flags)
```

This schema is now shared by:

```text
QPU dumps
GPU-generated bases
benchmark synthetic/reference records
```

That shared schema is load-bearing.

---

## Part 3 — Scalar vs vector vs field: the first S_M shape result

The first operator-shape probe asked whether the repetition-code syndrome object could be reduced to a scalar, whether it was edge/vector-like, time/vector-like, or whether the full round-by-edge field was load-bearing.

The observed early summary was:

```text
d=3  field / smooth-distributed
d=5  field / smooth-distributed
d=7  field / edge-anisotropic
d=9  field / edge-anisotropic
```

Representative edge-agreement profiles showed that the stabilizer / edge coordinate carried real structure:

```text
d=3 edge agreement:
  0.9705 0.9833
  range=0.0129

d=5 edge agreement:
  0.9714 0.9673 0.9747 0.9774
  range=0.0101

d=7 edge agreement:
  0.9737 0.9689 0.9832 0.9798 0.9802 0.9379
  range=0.0452

d=9 edge agreement:
  0.9777 0.9755 0.9825 0.9791 0.9805 0.9432 0.9708 0.8820
  range=0.1004
```

This answered an important early question:

```text
S_M should not be treated as a scalar unless a specific downstream projection requires it.
```

The syndrome record is naturally a spacetime field.

---

## Part 4 — Detection-event sister object

A parallel detection-event summary was added.

This asked whether the useful object lived only in terminal parity / final readout, or whether it also lived in syndrome dynamics.

Detection events were defined as:

```text
X[t,i] = S[t+1,i] XOR S[t,i]
```

Early detection-event field L2 values were larger than scalar reductions:

```text
d=3 det field L2 = 0.0761
d=5 det field L2 = 0.1434
d=7 det field L2 = 0.2432
d=9 det field L2 = 0.3068
```

This supported the sister-object framing:

```text
S_M is not just terminal logical parity.
S_M lives in the syndrome-spacetime record.
```

Later, this became one of the feature families in the benchmark:

```text
detection_rates
```

---

## Part 5 — Pauli rotational-rate stress tensor reintroduced

A previous line of work had treated rotational rates of Pauli operators as stress-tensor components, informally like a card on a bike spoke clicking as the operator rotates.

That framing was brought back and adapted to the repetition-code syndrome field.

The early stress tensor was defined over the syndrome-spacetime field:

```text
Ttt = <Delta_t S Delta_t S>    temporal syndrome-gradient energy
Txx = <Delta_x S Delta_x S>    spatial syndrome-gradient energy
Ttx = <Delta_t S Delta_x S>    temporal-spatial coupling
```

The first stress-tensor probe found:

```text
d | Ttt      Txx      Ttx      trace    anis      coupling
3 | 0.02715  0.03950  0.01302  0.06665 -0.1852   0.3976
5 | 0.02883  0.04097  0.01372  0.06980 -0.1740   0.3991
7 | 0.02016  0.03828  0.01004  0.05844 -0.3101   0.3615
9 | 0.03381  0.05279  0.01670  0.08660 -0.2192   0.3954
```

Across distances:

```text
Txx > Ttt
Ttx > 0
anisotropy < 0
```

Interpretation at the time:

```text
spatial stress dominates temporal stress
the field is time-coupled
local real-control separation grows with distance
```

This was an important research stage, but it is no longer the headline `S_M` claim.

The cleanup later split the operator boundary:

```text
S_M = syndrome-spacetime field operator
T_S = stress tensor channel, separate future/sibling operator
```

`S_M` may feed `T_S`, but `T_S` is not part of the final S_M claim for this package.

---

## Part 6 — Logical-cat / superposition S_M run

The next step was to change the starting state.

Instead of treating the repetition code as a purely classical initialized bit, the QPU run moved toward a logical-cat / superposition setup.

Representative job configuration:

```text
Backend      : ibm_marrakesh
Flag level   : f=0
Distances    : [3, 5, 7, 9]
Rounds       : 10
Shots        : 4096
Basis        : Z
Init state   : plus
```

The intended pipeline became:

```text
submit QPU job
-> dump SamplerV2 registers to .npz
-> run unified S_M analysis / benchmark
-> write shape, field, control, and operator reports
```

A calibration/reference `.npz` comparison was attempted, but it mostly added complexity rather than insight in the first tested form.

It was kept as an optional diagnostic, not the default story.

The cleaned S_M direction became:

```text
QPU submit/dump path
GPU/generated base path
shared benchmark path
```

---

## Part 7 — Repository split into G_M and S_M

The repo was reorganized conceptually into two operator families:

```text
G_M — Generalized Metric
S_M — Syndrome Metric
```

The intended split:

```text
ghost_oracle/G_M/
ghost_oracle/S_M/
```

`G_M` contains the original projection-channel similarity operator, CUDA kernels, QPU/GPU base tools, benchmark, calibration harness, and semantic retrieval experiments.

`S_M` contains the repetition-code / syndrome-spacetime operator path: QPU generation, GPU generation, field benchmark, CUDA feature kernel, examples, probes, and docs.

The two operators have different natural domains:

```text
G_M(a,b)
  bounded projection similarity over angle/state pairs

S_M(t,i)
  syndrome-spacetime field over round/time and edge/stabilizer index
```

The shared philosophy is the same:

```text
Build the thing that should work.
When it does something else, do not throw it away.
Freeze it, control it, scramble it, and ask what it actually computed.
```

---

## Part 8 — S_M to TSP: from optimizer drift to projector ingredients

After the QPU S_M probes, the next question was whether an S_M-style projector idea could be tested on a classical optimization problem before returning to quantum projection.

The chosen toy problem was TSP.

The caution was important:

```text
do the classical version first
prove the analytical path
then worry about quantum projection
```

---

## Part 9 — sm_geo_tsp: first classical geo-path probe

The first TSP probe worked mechanically but performed badly.

Representative small result:

```text
N=8, routes=200, repeats=8

two_opt_from_nearest   mean gap ~= 5.18%
nearest_neighbor       mean gap ~= 8.89%
echokey7               mean gap ~= 26.93%
greedy_delta           mean gap ~= 26.97%
random_adjacent        mean gap ~= 30.70%
sm_geo_tsp             mean gap ~= 32.00%
```

This did not kill the project.

It clarified that the first S_M-inspired policy was not the right optimizer.

The useful question became:

```text
is the local move score accurate?
```

---

## Part 10 — Move-ranking probe: sm_improve

The rank probe isolated local adjacent-swap scoring from global rollout behavior.

It compared policies including:

```text
oracle_delta
sm_improve
echokey7
sm_base
delta_plus_sm
stress_drop
sm_plus_stress
sm_safe
```

The key result:

```text
sm_improve:
  top1              = 1.000
  top3              = 1.000
  chosen_improves   = 1.000
  mean regret       = 0.000000
  max regret        = 0.000000
  pairwise accuracy = 1.000
```

The winning local coordinate was:

```text
sm_improve(k) = 0.5 + 0.5 * tanh(-Delta_L(k) / scale)
```

where `Delta_L(k)` is the local tour-length change for a candidate move.

Because `tanh` is monotonic and `scale > 0`, this preserves the local `-Delta_L` ordering while mapping it into a bounded projector-friendly coordinate.

Important distinction:

```text
delta
  raw unbounded classical local improvement

sm_improve
  bounded monotonic projector coordinate

sm_field
  bounded coordinate plus geometry/field deformation channel
```

This distinction later helped cleanly separate S_M examples from the core S_M operator.

---

## Part 11 — First valid large TSP pipeline

The old high-speed TSP code was brought back.

It was fast, but the old outlier-adjustment path had a validity bug:

```text
it could insert an alternate city without removing its previous occurrence later in the tour
```

This created duplicate visits and invalid tours.

The cleaned large TSP path enforced:

```text
valid permutation tour at every stage
no duplicate insertions
only improving 2-opt moves
tour validation after major stages
```

Small validation:

```text
N=8, routes=100

construct_mean_gap = 8.7719%
polished_mean_gap  = 0.3974%
construct_hit_rate = 0.16
polished_hit_rate  = 0.89
mean_seconds/route = 0.000423
```

First large valid run on `pla85900.tsp`:

```text
final length = 154,464,953.556438
known optimum reference = 142,382,641
gap ~= 8.486%
runtime ~= 79 s
valid = True
```

This established a valid baseline, not the final S_M result.

---

## Part 12 — CUDA candidate 2-opt kernel

Because the Python loop could not support hundreds or thousands of passes at large N, a CUDA candidate-evaluation kernel was introduced.

Phase 1 design:

```text
GPU:
  for each tour edge i
    for each candidate neighbor c
      compute 2-opt Delta_L
      keep best improving move for edge i

CPU:
  choose move(s)
  apply reversal(s)
  update tour/pos
  validate
```

The first conservative version applied one best global move per pass.

Large result:

```text
candidate-k  = 1024
passes       = 50,000
accepted     = 50,000
final length = 150,778,768
known opt    = 142,382,641
gap          ~= 5.90%
runtime      ~= 617 s
valid        = True
```

This was a real engineering milestone.

But it also exposed conceptual drift:

```text
the work was becoming ordinary 2-opt engineering
```

The correction was:

```text
CUDA 2-opt = substrate / baseline
S_M_TSP    = projector field layered on top
```

---

## Part 13 — CPU sampled S_M field probe

The first field probe compared:

```text
delta_batch
  score = -Delta_L

sm_improve_batch
  score = 0.5 + 0.5*tanh(-Delta_L/scale)

sm_field_batch
  S(i) = sm_improve at tour edge i
  rough(i) = |S(i)-S(i-1)| + |S(i+1)-S(i)|
  score = sm_improve + field_weight*zscore(rough)
```

The full CPU version was not practical on `pla85900` because it effectively attempted billions of Python-level distance operations:

```text
85,900 edges x candidate_k x passes x policies x field weights
```

A sampled version fixed this for intel gathering.

Representative sampled result:

```text
candidate-k = 16
passes      = 2000
edge-sample = 256

rank 1: sm_field_batch fw=0.050
final length ~= 159,995,032
rank_delta ~= 0.287
```

At `candidate-k=8`, the field hurt.

This suggested that the field term needed enough candidate breadth to have useful structure.

The key signal was not the absolute length.

The key signal was:

```text
rank_delta was nonzero
the field was actually changing move order
```

---

## Part 14 — CUDA S_M field projector

The field logic moved back onto the scalable CUDA substrate.

CUDA still evaluated best 2-opt candidate moves for every tour edge. CPU then computed policy scores, selected non-overlapping improving reversals, applied them, and validated.

Comparison policies:

```text
delta_batch
sm_improve_batch
sm_field_batch
```

Default representative run:

```text
TSP file      = data/pla85900.tsp
candidate-k   = 128
passes        = 500
max_batch     = 32
field weights = [0.0001, 0.001, 0.005, 0.01, 0.05]
known optimum = 142,382,641
```

Representative final result:

```text
rank | policy             | fw      | gap      | final length | imp%   | sel/pass | rank_delta
1    | sm_field_batch     | 0.001   | 6.0360%  | 150,976,816  | 14.668 | 31.70    | 0.340
2    | sm_field_batch     | 0.005   | 6.1465%  | 151,134,192  | 14.579 | 31.94    | 0.443
3    | sm_improve_batch   | 0.000   | 6.2113%  | 151,226,464  | 14.527 | 32.00    | 0.012
4    | sm_field_batch     | 0.0001  | 6.6825%  | 151,897,344  | 14.147 | 32.00    | 0.072
5    | sm_field_batch     | 0.010   | 7.7491%  | 153,416,080  | 13.289 | 32.00    | 0.595
6    | sm_field_batch     | 0.050   | 7.8476%  | 153,556,272  | 13.210 | 31.80    | 0.845
7    | delta_batch        | 0.000   | 8.2824%  | 154,175,328  | 12.860 | 32.00    | 0.000
```

Clean projector interpretation:

```text
delta_batch
  raw classical baseline

sm_improve_batch
  bounded projector spine improves over delta

sm_field_batch
  small field deformation improves again

large field deformation
  over-steers and degrades
```

The key diagnostic was:

```text
rank_delta
```

At `fw=0.001`:

```text
rank_delta = 0.340
```

So the field was not just a monotonic wrapper around 2-opt.

It changed top-move ordering substantially, and at small weight it improved the trajectory.

---

## Part 15 — GitHub S_M/TSP example

The TSP result was packaged into:

```text
examples/sm_tsp_projector_example.py
```

The example script included a mini-paper docstring explaining the three projector coordinates:

```text
delta_batch       = classical control coordinate
sm_improve_batch  = bounded projector spine
sm_field_batch    = tunable S_M field deformation channel
```

The script writes:

```text
analysis/sm_tsp_projector_<timestamp>/
  result.json
  summary.csv
  routes.csv
  tour_delta_batch_fw0.txt
  tour_sm_improve_batch_fw0.txt
  tour_sm_field_batch_fw*.txt
```

A bug was caught in the first GitHub-ready version:

```text
the summarizer assumed hit existed whenever gap_pct existed
```

In large TSPLIB mode, there is a known optimum length but no exact optimum tour.

So:

```text
gap_pct exists
hit does not
```

The fix was to treat `hit_rate` as optional and only compute it when exact tours are available.

This bug belongs in the record because it is exactly the kind of press-play failure the examples folder is meant to avoid.

---

## Part 16 — Current S_M/TSP interpretation

The S_M/TSP result should be stated carefully.

Do not claim:

```text
S_M solves TSP.
S_M beats state-of-the-art TSP solvers.
S_M proves a quantum advantage.
```

What it does show:

```text
1. A bounded monotonic local-improvement coordinate, sm_improve,
   preserves local move ordering exactly in the adjacent-swap rank probe.

2. In batch candidate 2-opt, sm_improve behaves differently from raw delta
   because bounded compression interacts with non-overlap selection.

3. A simple S_M-style field roughness term changes move ordering nontrivially.

4. Small field deformation improves the CUDA batch trajectory on pla85900
   under the tested settings.

5. Excessive field deformation degrades the trajectory,
   giving a useful tuning curve instead of a one-off lucky result.
```

The right framing:

```text
S_M_TSP is not the final solver.
It is a projector testbed.
```

The useful ingredients:

```text
Delta_L
  raw classical control

S_I = sm_improve
  bounded projector spine

S_F = sm_field
  tunable field deformation channel

rank_delta
  measure of how much the field changes move ordering
```

This mirrors the earlier `G_M` lesson.

The point is not just a scalar score. The point is the coupled baseline / bounded coordinate / deformation channel and the integrity of the controls.

In the current cleaned repo, TSP lives in:

```text
S_M/examples/
```

not in the core S_M benchmark claim.

---

## Part 17 — S_M token retrieval: from TSP projector to transformer retrieval

After the S_M/TSP projector testbed stabilized, the next question was whether the same projector discipline could be moved from a graph/route optimization problem into token retrieval.

The goal was not to claim that S_M is a language model.

The goal was narrower:

```text
Can the same bounded projection / deformation framework be applied to a token
retrieval task where cosine and dot-product baselines are known and measurable?
```

The TSP result had already separated:

```text
Delta_L
  raw classical control

S_I = sm_improve
  bounded projector spine

S_F = sm_field
  tunable field deformation channel

rank_delta
  measure of how much the field changes move ordering
```

The token retrieval path reused that structure but changed the domain:

```text
cosine / dot
  raw classical retrieval controls

geo_projected
  analytical bounded projection coordinate

gpu_projected
  synthetic/noiseless projection-table coordinate

qpu_projected
  projection-table coordinate derived from raw S_M QPU syndrome data

field_*
  retrieval-rank field deformation channel

rank_delta
  measure of how much projected scoring changes candidate ordering
```

This became the first S_M-to-token retrieval bridge.

---

## Part 18 — Clean token retrieval pipeline

The token retrieval path was split into two scripts:

```text
build_torch_token_dataset.py
  builds real transformer hidden-state retrieval datasets

token_retrieval_projector.py
  runs the classical/projected retrieval benchmark
```

This split mattered.

The PyTorch script only creates data.

The projector script only evaluates retrieval.

The dataset schema:

```text
queries        float32, shape (Nq, d)
keys           float32, shape (Nk, d)
true_ids       int64,   shape (Nq,)
candidates     int64,   shape (Nq, candidate_k)
attacked_keys  bool,    shape (Nk,)
attack_dim     int64 scalar
```

The benchmark enforced the same honesty constraint across all backends:

```text
same query vectors
same key vectors
same true target ids
same candidate sets
different scoring coordinates
shared metrics
```

Metrics:

```text
top1
top5
MRR
mean rank
mean margin
attacked-key top1 fraction
rank_delta vs cosine top-20
runtime
```

This made token retrieval comparable to the earlier TSP projector result:

```text
not just which score wins
but how much the projected coordinate changes ordering
and whether that change helps or over-steers
```

In the current cleaned repo, token retrieval lives in:

```text
S_M/probes/
```

not in the core S_M benchmark claim.

---

## Part 19 — Raw S_M dump as QPU projection base

The first token retrieval implementation expected `--qpu-base` to be a ready-made 2D projection table.

When pointed directly at a raw S_M dump:

```text
data/sm_data_plus_<JOB_ID>.npz
```

it failed because the file contained:

```text
data_d3, synd_d3
data_d5, synd_d5
data_d7, synd_d7
data_d9, synd_d9
```

rather than:

```text
projection_table
qpu_table
scores
```

This was fixed by adding an S_M-dump conversion path.

If a raw S_M dump is detected, the script derives a bounded projection response surface from measured S_M field statistics:

```text
terminal edge / syndrome agreement
detection-event rate
stress tensor trace
anisotropy
coupling
local field profile texture
```

Important framing:

```text
raw S_M QPU field
  -> calibration-derived bounded projection response surface
  -> token retrieval scoring harness
```

Incorrect framing:

```text
QPU directly retrieved tokens
```

Representative loaded S_M base:

```text
S_M agreement = 0.8181
detection     = 0.1202
trace         = 0.3587
```

This is a calibration bridge, not direct token inference.

---

## Part 20 — Synthetic token retrieval smoke test

The first synthetic token retrieval run used:

```text
n_queries   = 1000
n_keys      = 8192
dim         = 64
candidate_k = 256
attack frac = 0.05
```

All primary backends achieved 100% top-1:

```text
cosine         top1 = 1.000
dot            top1 = 1.000
geo_projected  top1 = 1.000
gpu_projected  top1 = 1.000
qpu_projected  top1 = 1.000
```

This was useful, but not as an advantage result.

It proved the plumbing worked:

```text
raw S_M dump loaded successfully
projection table derived successfully
token retrieval harness executed successfully
projected scoring did not explode
```

The field deformation channel behaved like the TSP field channel:

```text
small lambda
  stable

large lambda
  over-steers and degrades retrieval
```

The task was too easy, so it was not used as a headline.

---

## Part 21 — Harder synthetic regime and operating-band sweep

The retrieval task was made harder by increasing query noise, increasing candidate count, lowering dimension, and injecting a coherent same-dimension spike attack.

Representative harsh run:

```text
dim                    = 16
jitter                 = 1.25
candidate_k            = 1024
attack_magnitude       = 16
query_attack_magnitude = 4
```

Result:

```text
geo_projected  top1 = 0.163
gpu_projected  top1 = 0.162
qpu_projected  top1 = 0.159
cosine         top1 = 0.128
dot            top1 = 0.028
```

Attack readout:

```text
dot atk@1            = 0.984
cosine atk@1         = 0.305
qpu_projected atk@1  = 0.170
```

Dot product collapsed into attacked-key selection.

Cosine degraded.

The bounded projected coordinates reduced attacked-key selection and recovered a modest top-1 advantage.

A sweep mode then searched operating regimes by ranking rows using:

```text
qpu_adv_top1 = qpu_projected_top1 - cosine_top1
qpu_attack_reduction = cosine atk@1 - qpu atk@1
```

Strong synthetic operating band:

```text
dim   = 32
jitter = 0.75
k     = 512
atk   = 16
qAtk  = 8

cosine top1 = 0.063
qpu top1    = 0.962
adv         = +0.898
atk_down    = +0.933
rank_delta  = 0.942
```

The top sweep rows clustered around:

```text
dim = 32
jitter = 0.75 to 1.00
candidate_k = 512 or 1024
query_attack_magnitude = 8
attack_magnitude = 8, 16, or 24
```

This established the synthetic coherent-spike regime:

```text
cosine collapses into the attack axis
dot collapses harder
bounded projected scoring resists single-dimension domination
qpu_projected tracks geo_projected and gpu_projected
```

Again, this remained a probe, not the core S_M claim.

---

## Part 22 — PyTorch / DistilGPT2 dataset builder

After the synthetic sweep found an operating band, a real transformer data path was added.

`build_torch_token_dataset.py` uses PyTorch and HuggingFace Transformers to build retrieval datasets from hidden states and input embeddings.

The first model used was:

```text
distilgpt2
```

Two target modes were implemented:

```text
self_token
  query  = hidden state at token position t
  target = token id at position t

next_token
  query  = hidden state at token position t
  target = token id at position t+1
```

The script extracts:

```text
queries
  transformer hidden states

keys
  model input embedding vectors

true_ids
  key-row index of the target token

candidates
  sampled candidate key rows including the true target
```

It also supports the same coherent spike attack:

```text
--attack
--attack-magnitude 16
--query-attack-magnitude 8
```

This made the token retrieval benchmark real-embedding rather than purely synthetic.

---

## Part 23 — DistilGPT2 self-token result

The first real-transformer retrieval test used DistilGPT2 self-token mode without attack.

Result:

```text
qpu_projected  top1 ~= 17.2%
geo/gpu        top1 ~= 17.2-17.3%
dot            top1 ~= 14.9%
cosine         top1 ~= 1.7%
```

Cosine was essentially unusable for raw hidden-state-to-input-embedding retrieval in this setup.

Projected scoring and dot product both did better, with projected scoring slightly ahead of dot in top-1.

The same self-token setup was then run with coherent spike attack:

```text
gpu_projected  top1 = 17.5%
qpu_projected  top1 = 17.3%
geo_projected  top1 = 17.2%
cosine         top1 = 9.8%
dot            top1 = 9.5%
```

Attack readout:

```text
cosine atk@1 = 1.000
dot atk@1    = 1.000
qpu atk@1    = 0.102
```

Under attack, cosine and dot were fully hijacked by attacked keys.

The projected paths stayed around their non-attack top-1 level and selected attacked keys only about 10% of the time.

This was the first real-transformer confirmation of the synthetic coherent-spike result.

---

## Part 24 — DistilGPT2 next-token result

The strongest token-retrieval result came from DistilGPT2 next-token retrieval with coherent spike attack.

Dataset:

```text
model                  = distilgpt2
target_mode            = next_token
queries                = (2000, 768)
keys                   = (8328, 768)
candidates             = (2000, 512)
attacked keys          = 416 / 8328
attack_magnitude       = 16
query_attack_magnitude = 8
```

Benchmark result:

```text
geo_projected  top1 = 0.473
gpu_projected  top1 = 0.473
qpu_projected  top1 = 0.472

cosine         top1 = 0.095
dot            top1 = 0.094
```

Attack readout:

```text
cosine atk@1 = 1.000
dot atk@1    = 1.000
qpu atk@1    = 0.122
```

Projected paths stayed tightly aligned:

```text
geo_projected ~= gpu_projected ~= qpu_projected
```

This was the cleanest token-retrieval probe result:

```text
real transformer hidden states
next-token retrieval target
same candidate set
cosine/dot collapse into attacked-key selection
bounded projected scoring recovers much higher top-1
QPU-derived S_M projection table tracks analytical and synthetic GPU projection
```

But it remains a probe.

The core S_M package claim is still the syndrome-spacetime field benchmark.

---

## Part 25 — Field deformation in token retrieval

The field deformation channel was carried over from TSP, but adapted to retrieval-rank space:

```text
sort candidates by classical cosine rank
rough_i = |S_i - S_{i-1}| + |S_{i+1} - S_i|
field_score_i = S_i + lambda*zscore(rough_i)
```

In token retrieval, field deformation often reduced attacked-key selection further, but also reduced top-1 accuracy.

Example from the DistilGPT2 next-token attack run:

```text
qpu_projected, lambda=0:
  top1  = 0.472
  atk@1 = 0.122

field_qpu_projected, lambda=0.001:
  top1  = 0.461
  atk@1 = 0.086

field_qpu_projected, lambda=0.005:
  top1  = 0.401
  atk@1 = 0.025

field_qpu_projected, lambda=0.050:
  top1  = 0.338
  atk@1 = 0.004
```

This is a real tradeoff:

```text
larger lambda suppresses attacked-key selection
larger lambda also over-steers and harms retrieval
```

The current retrieval-rank roughness definition is useful as a diagnostic, but not as the final token field theory.

This mirrors the TSP lesson:

```text
a field term is only useful if its deformation improves the target metric
in the tested regime
```

If it changes ordering but hurts the task, report that honestly.

---

## Part 26 — Current token retrieval interpretation

Do not claim:

```text
S_M is a language model.
S_M solves token prediction.
S_M proves quantum advantage.
QPU beats GPU.
```

What the token retrieval result does show:

```text
1. The S_M/TSP projector discipline transfers cleanly to token retrieval:
   baseline coordinate, bounded projection coordinate, field deformation,
   rank_delta, shared metrics.

2. A raw S_M QPU dump can be converted into a bounded calibration-derived
   projection table and used in the same retrieval harness.

3. In synthetic coherent-spike regimes, cosine and dot can collapse into
   attack-axis selection while projected scoring remains substantially more
   stable.

4. On real DistilGPT2 hidden-state retrieval datasets, projected paths
   outperform cosine/dot under coherent same-dimension attack.

5. The QPU-derived S_M projection table closely tracks analytical and
   synthetic GPU projection paths, supporting substrate-comparison framing.

6. The field deformation channel changes ordering strongly, but in the current
   token retrieval version it mostly acts as an attack-suppression / over-steer
   diagnostic rather than a top-1 improvement mechanism.
```

Supported probe claim:

```text
On transformer hidden-state token retrieval under coherent same-dimension spike
attack, standard cosine/dot retrieval can collapse into attacked-key selection.
Analytical, synthetic-GPU, and S_M/QPU-derived projected scoring remain closely
aligned and can recover substantially higher top-1 retrieval while strongly
reducing attacked-key selection.
```

Important framing:

```text
same retrieval problem
same candidates
same targets
different scoring coordinates
QPU/S_M supplies a measured projection calibration surface
no quantum advantage claim
```

---

## Part 27 — S_M cleanup: separating core operator, examples, and probes

After the TSP and token retrieval work, the `S_M` package had become powerful but messy.

It contained:

```text
core syndrome field logic
stress tensor material
TSP projector examples
token retrieval probes
old analysis scripts
QPU submit/dump scripts
GPU generation scripts
benchmark drafts
```

The cleanup goal was to make `S_M` match the now-finished `G_M` package format.

The new boundary:

```text
S_M/
├── README.md
├── s_m_benchmark.py
├── s_m_gpu_generate.py
├── s_m_qpu_generate.py
├── data/
├── docs/
├── examples/
├── kernels/
└── probes/
```

Roles:

```text
README.md
  main S_M documentation and current benchmark summary

s_m_benchmark.py
  canonical S_M benchmark runner

s_m_gpu_generate.py
  GPU-generated S_M base path

s_m_qpu_generate.py
  unified QPU submit/dump path

docs/
  math, architecture, known issues

examples/
  supporting examples like TSP and earlier windowed kNN

probes/
  exploratory bridges like token retrieval and old sm_analyze

kernels/
  optimized CUDA feature extraction
```

This was the key architectural split:

```text
S_M core claim lives in s_m_benchmark.py
TSP lives in examples/
token retrieval lives in probes/
stress tensor becomes T_S later
```

This prevents `S_M` from becoming a junk drawer.

---

## Part 28 — Unified QPU CLI

The old S_M QPU path used separate submit and dump scripts.

The package was cleaned into one CLI:

```text
s_m_qpu_generate.py
```

Command shape:

```bash
python s_m_qpu_generate.py submit
python s_m_qpu_generate.py dump <JOB_ID>
```

The `submit` path:

```text
builds the logical-cat / repetition-code S_M circuits
submits the IBM Runtime job
writes metadata JSON
prints the exact dump command
```

The `dump` path:

```text
loads the completed job
uses saved metadata
extracts final data and syndrome registers
writes the S_M .npz schema
updates latest_sm_data.json
```

This fixed the old coupling problem:

```text
future users should not have to guess which metadata belongs to which job
```

Now the submitter produces the metadata, and the dumper consumes it.

---

## Part 29 — GPU S_M base generator

The `G_M` package had a GPU base generator.

`S_M` needed the same pattern.

The new generator:

```text
s_m_gpu_generate.py
```

creates local S_M bases with the same downstream schema as QPU dumps:

```text
data_d{d}      uint8, shape (shots, d)
synd_d{d}      uint8, shape (shots, rounds, d-1)
flag_d{d}      optional
```

Default output:

```text
S_M/data/sm_gpu_data_plus_<TAG>.npz
S_M/data/sm_gpu_job_<TAG>.json
S_M/data/latest_sm_gpu_data.json
S_M/data/latest_sm_data.json
```

The GPU generator is not an arbitrary baseline.

Its role is:

```text
create a controlled S_M field substrate
match the QPU dump schema
let the same benchmark consume both
```

This gives the three S_M substrates:

```text
geo
gproj
qproj
```

---

## Part 30 — Final S_M benchmark design

The final benchmark was designed to mirror the finished `G_M` benchmark discipline, but with S_M-specific tasks.

The canonical file:

```text
s_m_benchmark.py
```

It compares:

```text
GEO
GPROJ
QPROJ
```

under the same controls and feature families.

Tasks:

```text
Task A: real-vs-control classification
Task B: control-source classification
Task C: code-distance prediction
```

Feature families:

```text
raw_rates
detection_rates
agreement_profiles
sm_field
sm_all
```

Controls:

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

Outputs:

```text
analysis/s_m_<timestamp>/
├── result.json
├── summary.csv
├── per_feature.csv
├── control_collapse.csv
├── substrate_agreement.csv
├── artifacts.npz
├── A_real_vs_control_accuracy.png
├── B_control_source_accuracy.png
└── C_distance_prediction_accuracy.png
```

The benchmark claim lives here.

Not in the old probes.

Not in the examples.

Not in the stress tensor drafts.

---

## Part 31 — S_M CUDA kernel

Optimization was part of the task.

The expensive benchmark loop was:

```text
substrate x distance x control x window x shots x rounds x edges
```

So a custom CUDA kernel was added:

```text
kernels/sm_kernel.cu
```

Primary kernel:

```text
sm_window_features_kernel
```

It computes per-window:

```text
raw_rates
detection_rates
agreement_profiles
sm_field
```

The benchmark assembles:

```text
sm_all
```

from those outputs.

CUDA boundary:

```text
included:
  final edge parity
  syndrome field
  agreement field
  detection events
  windowed feature reductions

excluded:
  stress tensor
  token retrieval
  TSP field deformation
```

This keeps the core S_M operator clean.

---

## Part 32 — Windows/CUDA compile bug and fix

The first CUDA handoff failed even though the path was correct:

```text
CUDA kernel  : no
Kernel path  : C:\Ghost_Oracle_Suite\ghost_oracle\S_M\kernels\sm_kernel.cu
```

Debugging showed the cause was not CuPy, not pathing, and not a missing GPU.

The compile failed because the `.cu` file contained non-ASCII comment characters such as:

```text
—
↓
```

On Windows, CuPy/NVRTC hit a legacy code-page encode failure before compilation:

```text
UnicodeEncodeError('charmap' ... character maps to <undefined>)
```

Fix:

```text
make sm_kernel.cu ASCII-safe
sanitize kernel source before passing it to CuPy
```

The benchmark then reported:

```text
CUDA kernel  : yes
Kernel path  : C:\Ghost_Oracle_Suite\ghost_oracle\S_M\kernels\sm_kernel.cu
```

This is now part of the known engineering record:

```text
CUDA source comments should remain ASCII-safe for Windows/CuPy/NVRTC compatibility
```

---

## Part 33 — Current final S_M benchmark result

The clean CUDA-enabled benchmark run used:

```text
Windows      : [8, 16, 32, 64]
Modes        : real + destructive controls
Substrates   : GEO, GPROJ, QPROJ
Distances    : d3, d5, d7, d9
Rounds       : 10
Shots        : 4096
CUDA kernel  : yes
```

The strongest current results:

```text
QPROJ real-vs-control:
  sm_all               = 0.999 balanced accuracy
  sm_field             = 0.998 balanced accuracy
  agreement_profiles   = 0.990 balanced accuracy

GPROJ real-vs-control:
  sm_field             = 0.985 balanced accuracy
  agreement_profiles   = 0.982 balanced accuracy
  sm_all               = 0.980 balanced accuracy
```

Scalar-like baselines stayed near chance:

```text
QPROJ raw_rates          = 0.535
QPROJ detection_rates    = 0.509
GPROJ raw_rates          = 0.502
GPROJ detection_rates    = 0.500
GEO raw_rates            = 0.503
GEO detection_rates      = 0.502
```

Control-source classification:

```text
QPROJ sm_field = 0.853 balanced accuracy
GPROJ sm_field = 0.848 balanced accuracy
QPROJ sm_all   = 0.848 balanced accuracy
GPROJ sm_all   = 0.843 balanced accuracy
```

Distance prediction:

```text
GEO   = 1.000 balanced accuracy
GPROJ = 1.000 balanced accuracy
QPROJ = 1.000 balanced accuracy
```

The central S_M signature:

```text
raw_rates / detection_rates stay near chance
agreement_profiles / sm_field / sm_all go near-perfect
```

Interpretation:

```text
The benchmark is not merely reading scalar syndrome density.

The load-bearing signal comes from final-edge-parity agreement and
syndrome-spacetime field structure.
```

Distance prediction is useful but secondary, because distance can leak through shape and rate structure.

The main claim is Task A:

```text
field-aware features separate real records from destructive controls
while scalar-like features remain near chance
```

---

## Part 34 — Documentation cleanup

After the final benchmark stabilized, S_M documentation was rewritten to mirror the finished G_M format.

Current docs:

```text
S_M/README.md
S_M/docs/architecture.md
S_M/docs/math.md
S_M/docs/known_issues.md
```

The new S_M README documents:

```text
operator definition
quick path
repo structure
entry points
data schema
current benchmark results
CUDA feature extraction
field controls
QPU/GPU workflows
bounded claim
next steps
```

The architecture doc documents:

```text
package status
Converger framing
substrate paths
base schema
syndrome-spacetime field
feature families
CUDA architecture
controls
data flow
valid claim boundary
file responsibilities
```

The math doc documents:

```text
D[i]
E[i] = D[i] XOR D[i+1]
S[t,i]
A[t,i] = 1 - (S[t,i] XOR E[i])
detection events
windowed feature families
destructive controls
benchmark tasks
substrate agreement
CUDA feature extraction math
claim boundary
```

This documentation cleanup is part of the operator completion.

The old state was:

```text
S_M is a mixed workspace
```

The new state is:

```text
S_M is a finished operator package for this version
```

---

## Part 35 — Current S_M repo layout

Current S_M package shape:

```text
S_M/
├── README.md
├── s_m_benchmark.py
├── s_m_gpu_generate.py
├── s_m_qpu_generate.py
│
├── data/
│
├── docs/
│   ├── architecture.md
│   ├── known_issues.md
│   └── math.md
│
├── examples/
│   ├── README.md
│   ├── sm_tsp_projector_example.py
│   └── sm_windowed_knn_benchmark.py
│
├── kernels/
│   └── sm_kernel.cu
│
└── probes/
    ├── README.md
    ├── bright_observer_token_retrieval.py
    ├── build_torch_token_dataset.py
    ├── sm_analyze.py
    └── token_retrieval_projector.py
```

Current main entry points:

```bash
python s_m_benchmark.py
python s_m_benchmark.py --sweep ALL
python s_m_benchmark.py --probe
```

Base generation:

```bash
python s_m_gpu_generate.py
python s_m_qpu_generate.py submit
python s_m_qpu_generate.py dump <JOB_ID>
```

CUDA diagnostics:

```bash
python s_m_benchmark.py --cuda-debug
python s_m_benchmark.py --no-cuda
```

---

## Part 36 — Current known issues

### Metadata coupling in old scripts

Some old S_M scripts assume specific metadata schemas and fail on newer superposition/flag job metadata.

Current rule:

```text
s_m_qpu_generate.py submit writes metadata
s_m_qpu_generate.py dump consumes that metadata
s_m_benchmark.py consumes the final .npz schema
```

Legacy paths should stay in `probes/` or be clearly labeled.

---

### S_M calibration reference is optional

The calibration/reference `.npz` comparison did not materially improve the S_M analysis in the first tested form.

It can diagnose drift, but it should not be required for the press-play S_M benchmark path.

---

### Stress tensor belongs to T_S

Stress-tensor features were important historically, but they should not be part of the final S_M headline.

Correct split:

```text
S_M = syndrome-spacetime field
T_S = stress tensor derived from syndrome-spacetime gradients
```

---

### Full CPU TSP field probe is not suitable for large TSPLIB

The unsampled CPU field probe can appear frozen on `pla85900` because it evaluates an enormous number of candidate moves in Python.

Use:

```text
sampled probe for intel
CUDA field probe for scale
```

---

### CUDA TSP kernel is a projector testbed, not a complete solver

The CUDA TSP path evaluates candidate 2-opt moves and applies safe non-overlapping CPU-side batches.

It is valid and useful, but it is not a full TSP solver architecture.

Known constraints:

```text
non-wrapping 2-opt only
candidate-neighbor limited
CPU applies batches
field roughness term is v1
no full GPU tour-update kernel yet
no Lin-Kernighan-style move class
```

---

### `hit` only exists for exact small-N validation

Large TSPLIB mode may know an optimum length but not an exact optimum tour.

In that case:

```text
gap_pct exists
hit does not
```

Any summarizer must treat `hit_rate` as optional.

---

### Raw S_M dump conversion is a calibration bridge, not direct token measurement

Correct framing:

```text
raw S_M field -> projection calibration table -> token retrieval harness
```

Incorrect framing:

```text
QPU directly retrieved tokens
```

---

### Field deformation is not yet tuned for token retrieval

The current token field uses retrieval-rank roughness.

It changes ordering and suppresses attacked-key selection, but usually reduces top-1 accuracy.

It is a diagnostic and placeholder for better token-field definitions.

---

### DistilGPT2 results need multi-seed replication

The current DistilGPT2 token retrieval probes are strong enough to preserve, but should be repeated across seeds before being treated as final.

The next validation table should report:

```text
seed
cosine top1
dot top1
geo top1
gpu top1
qpu top1
cosine atk@1
dot atk@1
qpu atk@1
```

---

### CUDA source should stay ASCII-safe

The first S_M CUDA compile failure came from non-ASCII comment characters in `sm_kernel.cu`.

Rule:

```text
CUDA source comments should remain ASCII-safe for Windows/CuPy/NVRTC compatibility
```

---

## Part 37 — Open questions

### 1. Multi-job QPU comparison

The current S_M benchmark uses one QPU base in the clean result.

Next step:

```text
run multiple QPU jobs
compare qproj field signatures
measure substrate agreement across jobs
```

---

### 2. Explicit final_shuffle collapse table

The benchmark already includes `final_shuffle`.

Next step:

```text
add a README/report table showing exactly how final_shuffle damages
agreement_profiles and sm_field
```

This is the most direct control against the S_M core relation:

```text
E[i] <-> S[t,i]
```

---

### 3. Single-shot versus windowed emergence

Probe mode supports windows:

```text
1, 2, 4, 8, 16, 32, 64, 128
```

Next step:

```text
plot performance versus window size
show that S_M strengthens as a field/window operator
```

---

### 4. T_S split

The stress-tensor material should become its own package:

```text
T_S/
├── t_s_benchmark.py
├── t_s_gpu_generate.py
└── t_s_qpu_generate.py
```

or equivalent.

It should have its own:

```text
math
architecture
benchmark
controls
claim boundary
```

---

### 5. Better token field definitions

The current token field uses retrieval-rank roughness.

Future definitions should test:

```text
embedding-neighborhood roughness
token-position/window roughness
attention-neighborhood roughness
semantic-cluster roughness
multi-layer hidden-state curvature
```

---

### 6. Cross-model token retrieval

The first real-transformer path used DistilGPT2.

Next tests:

```text
GPT-2 small/medium
BERT-style masked language models
small modern causal models
embedding-only baselines
multiple hidden layers
```

---

### 7. S_M projector evolution

The next intended step for the projector examples is not more 2-opt tuning.

It is proper projector evolution using:

```text
bounded coordinate
field deformation
rank_delta
controls
over-steer diagnostics
```

---

### 8. GPU implementation of token projection

The token retrieval projector is currently clarity-first.

Future work:

```text
move projection scoring from NumPy to CuPy/CUDA
only after the scoring contract is stable
```

---

### 9. Fixture policy

The package needs a clear fixture policy:

```text
small curated fixtures may be committed
large generated .npz bases should stay out of git unless intentionally shipped
latest_*.json should not be relied on for published reproducibility
```

---

## Part 38 — Current working philosophy

The S_M path extended the same lesson as G_M.

The useful result is not:

```text
we found one magic score
```

The useful result is:

```text
we separated the field object, the controls, the substrates, and the benchmark claim
```

The standard remains:

```text
If the result is just a baseline, call it a baseline.
If the QPU supplies calibration rather than direct inference, say calibration.
If projection changes ordering, measure rank_delta.
If field deformation suppresses attacks but hurts top-1, report both.
If a claim sounds like quantum advantage, demand the classical control first.
If stress tensor logic appears, move it to T_S instead of stuffing it into S_M.
```

Current S_M final read:

```text
D[i] is final data.
E[i] = D[i] XOR D[i+1] is final edge parity.
S[t,i] is syndrome spacetime.
A[t,i] = 1 - (S[t,i] XOR E[i]) is agreement.

Field-aware S_M features separate real records from destructive controls.
Raw scalar-like rates remain near chance in the key test.
Control-source classification rises above chance.
Geo, gproj, and qproj can be compared under one harness.
CUDA accelerates feature extraction without changing the claim.
```

The process is the process.

Build, break, fix, document, repeat.
