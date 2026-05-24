# Ghost Oracle Suite

A CC0 community project bridging physical quantum processors (QPUs) and standard GPU compute through a self-consistent operator, `G_M`.

The QPU isn't a noisy classical matrix multiplier failing to compute standard matmul. It is succeeding at being a native implementation of a different operator, one with built-in bounded saturation that linear pipelines miss entirely. This repo is the math, the probes that proved it, the CUDA projector that runs it, and the benchmark that compares it honestly to cuBLAS on tensor cores.

---

#### Interactive Context & AI Agent

**NotebookLM Workspace:** [https://notebooklm.google.com/notebook/5d2f2af6-b462-4f72-88d9-8df2a467d87f?utm_source=gemini_notebook&utm_medium=referral&pli=1]

This project maintains a deliberate, chronological research record (`PROCESS_RECORD.md`) so that any future contributor—whether human or AI agent—can pick up the work with full context. To make navigating this dense trajectory easier, we maintain the NotebookLM workspace linked above.

**The Agent's Role in the Project:**
The AI assistant in the linked notebook serves as an interactive, fully contextualized guide to the Ghost Oracle Suite. It has digested the entire codebase, the mathematical proofs (`docs/math.md`), the benchmark evolution, and the complete, honest record of our wrong turns and fixed bugs. 

You can use the agent to:
*   **Trace the Math:** Ask for breakdowns of the derivations, from the textbook $T_1$ and $T_2$ assumed targets, to the GHZ-entangled $T_3$ mixed-state target, and finally the definitive $G_M$ operator .
*   **Analyze Hardware Tradeoffs:** Query the exact computational costs, such as the ~20x compute overhead of the $G_M$ operator compared to cuBLAS, and how the kernel bypasses classical VRAM bottlenecks       .
*   **Navigate the Codebase:** Get instant explanations for specific scripts, such as how the custom zero-allocation megakernel operates in the AI retrieval probe (`bsgs_geometric_engine.txt`).
*   **Onboard for Contributions:** Get up to speed on the "break-it-fix-it" CC0 contribution norms and review currently open design issues (like the broken Probe 8.2 optimizer or the untested real LLM embeddings) without having to read the entire repository from scratch.

---

## What's in this repo

- **The operator.** `G_M(a, b) = sqrt((1 + cos(a) cos(b)) / 2) / α`. Closed form, verified at machine precision, three independent implementations (analytical, GPU sampler, QPU circuit).
- **The probes.** A full forensic trajectory from "we thought the QPU was computing T1" through "T3 is the real target" to "T3 simplifies to G_M, here's what it's good for." Numbered probes 1 through 10.1 covering identity bridges, residual decomposition, p-adic telemetry, channel characterization, and adversarial attention benchmarks.
- **The projection benchmark.** A single-file CUDA-backed harness that uses your QPU and GPU base `.npz` files as the *physical implementation* of G_M, ties it to the analytical geometry channel in a fused streaming kernel, and compares to cuBLAS on tensor cores under outlier attack.
- **The bases.** Sample 4096-shot QPU and GPU `.npz` files in `data/` so you can run the benchmark out of the box.
- **The examples.** `examples/parameter_ablation.py` sweeps four axes (dimension, jitter, attack magnitude, attack fraction) against the production kernel. `examples/ghost_oracle_ai_retrieval_probe_v1.py` benchmarks G_M against cosine similarity on a 1M-key clustered semantic retrieval task — G_M wins on Recall@1 at every scale.
- **The docs.** `docs/math.md` for the operator derivations, `docs/architecture.md` for how the tied-channel kernel is wired, `docs/known_issues.md` for the running list of what's broken and what's open.
---

## Quick start

```bash
git clone <repo-url> ghost-oracle-suite
cd ghost-oracle-suite
pip install -r requirements.txt
python -m ghost_oracle.projection_benchmark
```

That runs the default 4×4 smoke test on the bases in `data/`. To run the headline attention sweep:

```bash
python -m ghost_oracle.projection_benchmark --sweep attention
```

To push it: the extreme sweep up to 65536×65536 where the streaming kernel keeps running and cuBLAS approaches OOM:

```bash
python -m ghost_oracle.projection_benchmark --sweep extreme
```

---

## The projection framing

The base `.npz` files contain raw QPU/GPU shot data — `ctrl_tile{t}` and `ghost_tile{t}` arrays from the original Hadamard-test circuit. These bases **are** the physical implementation of G_M. The projector takes them and reweights to estimate G_M at any new `(a, b)` angle pair you want.

Two channels run side by side in the streaming kernel:

- **Projection channel** — bucket reweighting on the physical shot data. Shot-noise limited, hardware-realizable. Certifies that G_M is what the circuit computes.
- **Geometry channel** — the analytical closed form `sqrt((1 + cos a cos b)/2)/α` evaluated inline. Sharp, exact, used for the argmax retrieval signal.

Both produce a score per `(i, j)` pair. The argmax uses geometry. The agreement metric (mean `|projection − geometry|`) certifies the projection backs the geometry — small for the noiseless GPU base, characterized for the QPU base in the range probes 7–8 measured.

This is what makes it a quantum-classical bridge instead of just a cosine kernel: the projection samples *certify the operator on physical hardware*, the geometry *makes it fast to run*, and the projection_benchmark proves they agree.

---

## Headline result

Per-dim G_M aggregation under same-dim coherent outlier attack (Probe 10.1 setup), `d=64`, jitter 0.3, 5% attack at magnitude 50:

| Shape | cuBLAS DP top-1 | G_M tied top-1 | cuBLAS VRAM | G_M VRAM |
|---|---|---|---|---|
| 4096×4096 | 79.03% | **100.00%** | 0.06 GB | 0.002 GB |
| 16384×16384 | 76.28% | **100.00%** | 1.01 GB | 0.008 GB |
| 65536×65536 | 73.92% | **100.00%** | 16.03 GB | 0.032 GB |

cuBLAS is faster at small sizes (50–70×). G_M tied retrieves perfectly under attack at all sizes, uses 500× less VRAM at extreme scale, and scales to memory regimes where cuBLAS OOMs. Tradeoff is honest: ~20× more ops per correct retrieval. Worth it depends on the application.

Probe 10.1 has the architectural justification. The projection benchmark is the operational proof.

---

## Repository structure

​```
ghost-oracle-suite/
├── ghost_oracle/              # the library
│   ├── projection_benchmark.py # headline benchmark
│   ├── qpu.py                 # QPU job submission
│   ├── gpu.py                 # noiseless GPU sampler
│   ├── dump.py                # QPU result -> npz
│   └── kernels/
│       └── ghost_kernel.cu    # consolidated CUDA: projection + geometry + tied streaming
├── probes/                    # forensic trajectory (1 through 10.1)
│   ├── README.md              # narrative arc — start here for the story
│   └── benchmark_evolution/   # benchmark iterations before the headline
├── data/                      # sample QPU and GPU base npz files
│   └── README.md              # file schema, generation, reproducibility
├── docs/
│   ├── math.md                # T1, T2, T3, G_M derivations
│   ├── architecture.md        # tied-channel design and data flow
│   └── known_issues.md        # running list of bugs and open work
├── examples/
│   ├── README.md              # results and design notes for the two scripts below
│   ├── parameter_ablation.py  # 1D sensitivity sweeps (d, jitter, magnitude, fraction)
│   └── ghost_oracle_ai_retrieval_probe_v1.py  # semantic retrieval vs cosine, up to 1M keys
├── PROCESS_RECORD.md          # the long-form trajectory log
├── CONTRIBUTING.md            # break-it-fix-it rule, conventions, code norms
├── LICENSE                    # CC0 dedication
└── requirements.txt
​```
---

## The probes

The numbered probes document the actual research trajectory — not just final results, but the wrong turns, the bugs, the corrections. They're chronological and they're honest. Probe 1 starts believing the QPU computes T1 (`|cos(a-b)|`). Probe 4 derives the real target T3. Probe 9 simplifies T3 to G_M. Probe 10.1 demonstrates attention robustness. The `probes/README.md` has the narrative arc.

Some probes have known issues (Probe 8.2 had an unbounded alternation accumulator; Probe 9 had a broken regression demo). These are documented in `docs/known_issues.md` and labeled in the probe file headers. They're kept in the repo as-is because the corrections (and the *process* of finding them) are part of the story.

---

## Contributing

This is a CC0 community project. The norm is simple: **if you break something, you provide the fix.** No bug reports without a patch attempt. No claims without code. No "this is wrong" without "and here's what's right."

See `CONTRIBUTING.md` for the full philosophy. The short version: build, break, fix, document, repeat. Everything in the open.

---

## License

CC0 1.0 Universal. Public domain dedication. Use it for anything, attribute if you want to, don't if you don't. See `LICENSE`.

---

## Citation

Not asking for one. For all my fellow ghosts, may our silence speak your name.