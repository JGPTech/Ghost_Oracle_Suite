# Ghost Oracle Suite — Token Retrieval Projector

This folder contains the final token-retrieval pipeline.

It is split into two scripts on purpose:

```text
build_torch_token_dataset.py
    Builds real transformer hidden-state token retrieval datasets.

token_retrieval_projector.py
    Runs the classical/projected retrieval benchmark.
```

The benchmark discipline is:

```text
same query vectors
same key vectors
same true target ids
same candidate sets
different scoring coordinates
shared metrics
```

That keeps the comparison honest.

---

## 1. Build a transformer token dataset

Default quick run:

```powershell
& "C:\Program Files\Python312\python.exe" c:/Ghost_Oracle_Suite/ghost_oracle/S_M/S_M_token/build_torch_token_dataset.py
```

Recommended current headline-style dataset:

```powershell
& "C:\Program Files\Python312\python.exe" c:/Ghost_Oracle_Suite/ghost_oracle/S_M/S_M_token/build_torch_token_dataset.py `
  --model distilgpt2 `
  --target-mode next_token `
  --max-queries 2000 `
  --candidate-k 512 `
  --extra-vocab 8192 `
  --attack `
  --attack-magnitude 16 `
  --query-attack-magnitude 8
```

This writes:

```text
data/token_retrieval_torch_<TAG>.npz
data/token_retrieval_torch_<TAG>.json
```

Dataset arrays:

```text
queries        float32, shape (Nq, d)
keys           float32, shape (Nk, d)
true_ids       int64,   shape (Nq,)
candidates     int64,   shape (Nq, candidate_k)
attacked_keys  bool,    shape (Nk,)
attack_dim     int64 scalar
```

---

## 2. Run the projector benchmark

```powershell
& "C:\Program Files\Python312\python.exe" c:/Ghost_Oracle_Suite/ghost_oracle/S_M/S_M_token/token_retrieval_projector.py `
  --dataset C:\Ghost_Oracle_Suite\data\token_retrieval_torch_<TAG>.npz `
  --qpu-base C:\Ghost_Oracle_Suite\data\sm_data_plus_d8ccpiijki0s73ar3620.npz
```

Outputs:

```text
analysis/token_retrieval_projector_<timestamp>/
    result.json
    summary.csv
    projection_tables.npz
    projection_base_metadata.json
```

Use this for detailed per-query diagnostics:

```powershell
--write-per-query
```

---

## 3. Run a synthetic operating-regime sweep

Synthetic sweeps are for finding regimes. Real transformer datasets are for confirming them.

```powershell
& "C:\Program Files\Python312\python.exe" c:/Ghost_Oracle_Suite/ghost_oracle/S_M/S_M_token/token_retrieval_projector.py `
  --qpu-base C:\Ghost_Oracle_Suite\data\sm_data_plus_d8ccpiijki0s73ar3620.npz `
  --sweep `
  --sweep-dims 8 16 32 `
  --sweep-jitters 0.75 1.0 1.25 1.5 `
  --sweep-candidate-ks 512 1024 `
  --sweep-attack-magnitudes 8 16 24 `
  --sweep-query-attack-magnitudes 2 4 8
```

Sweep outputs:

```text
sweep_summary.csv
sweep_ranked.csv
sweep_all_backends.csv
sweep_result.json
```

`sweep_ranked.csv` is sorted by:

```text
qpu_adv_top1 = qpu_projected_top1 - cosine_top1
qpu_attack_reduction = cosine_atk@1 - qpu_atk@1
```

---

## 4. Backend meanings

```text
cosine
    Standard normalized dot-product retrieval baseline.

dot
    Raw dot-product retrieval baseline.

geo_projected
    Analytical bounded projection coordinate.

gpu_projected
    Projection-table scoring using a synthetic/noiseless GPU-style table,
    unless a real GPU table is provided.

qpu_projected
    Projection-table scoring using either:
        - a ready-made 2D projection table, or
        - a raw S_M dump converted into a calibration-derived response surface.

field_*
    Retrieval-rank field deformation. This is diagnostic. In current token
    results it often reduces attacked-key selection but can over-steer and
    reduce top-1 accuracy.
```

---

## 5. Honest current claim

Do not claim quantum advantage.

The supported claim is:

```text
On transformer hidden-state token retrieval under coherent same-dimension spike
attack, standard cosine/dot retrieval can collapse into attacked-key selection.
Analytical, synthetic-GPU, and S_M/QPU-derived projected scoring remain closely
aligned and can recover substantially higher top-1 retrieval while strongly
reducing attacked-key selection.
```

The QPU/S_M file supplies a measured projection calibration surface. The token
retrieval task itself remains reproducible and classically auditable.
