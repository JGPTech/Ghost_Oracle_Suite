# Ghost Oracle Suite

A CC0 community project implementing projection-channel attention as a single algorithm across three substrates: mathematical reference, classical GPU simulation, and quantum hardware shots from IBM Runtime.

**Same physics, three platforms.** The projection-channel attention operator is defined mathematically. It admits three faithful implementations — analytical, classical noiseless, and real-QPU — and this repo is the math, the probes that mapped the trajectory, the CUDA kernel that runs it, and the final five-way benchmark that compares all three substrates head-to-head against a cuBLAS classical control.

---

#### Interactive Context & AI Agent

**NotebookLM Workspace:** [https://notebooklm.google.com/notebook/5d2f2af6-b462-4f72-88d9-8df2a467d87f]

This project maintains a deliberate, chronological research record (`PROCESS_RECORD.md`) so any future contributor — human or AI agent — can pick up the work with full context, including the wrong turns and the corrections. The NotebookLM workspace gives you an interactive guide to the math, the probes, the architecture, and the open issues.

**The Agent's Role:**
*   **Trace the Math:** Ask for breakdowns from the assumed $T_1$ / $T_2$ targets through the GHZ-entangled $T_3$ mixed-state target to the definitive $G_M$ operator.
*   **Analyze Tradeoffs:** Query the computational costs of the projection paths vs cuBLAS, the agreement metric as a hardware-quality readout, and the role of dynamic-mask calibration.
*   **Navigate the Codebase:** Get instant explanations for any script, kernel, or probe.
*   **Onboard for Contributions:** Get up to speed on the "break-it-fix-it" CC0 norms and the open work without reading the entire repo from scratch.

---

## What's in this repo

- **The operator.** `G_M(a, b) = sqrt((1 + cos(a) cos(b)) / 2) / α`. Closed form, verified at machine precision, with three faithful substrate implementations (analytical, classical sampler, QPU circuit).
- **The probes.** A forensic trajectory from "we thought the QPU was computing T1" through the discovery of T3 as the real target, the simplification to G_M, the projection-channel range investigation (probes 11–11.2), the bucket-mask ablations (probes 13–18), the cross-job validation (probe 18 fleet), the dynamic-mask router exploration (probe 19), the auto-calibrating production kernel (probe 20), and the final five-way benchmark.
- **The final benchmark.** A single-file harness that compares five attention paths on the same data: cuBLAS, the tied dual-channel kernel, geometry-only, projection-driven-by-QPU-shots, and projection-driven-by-noiseless-classical-shots. All scored on top-1, signal sharpness, and attack-spike concentration with calibrated thresholds per base.
- **The bases.** Sample QPU and GPU `.npz` files in `data/` from the same algorithm — the classical GPU sampler is a faithful noiseless simulation of the projection circuit, not an arbitrary baseline.
- **The docs.** `docs/math.md` for the operator derivations, `docs/architecture.md` for the kernel design, `docs/known_issues.md` for the running list of what's open.

---

## Quick start

```bash
git clone <repo-url> ghost-oracle-suite
cd ghost-oracle-suite
pip install -r requirements.txt
python -m ghost_oracle.final_benchmark_5way
```

That runs the final five-way verification on all base files in `data/`. To run with a saved calibration manifest from Probe 20:

```bash
python -m ghost_oracle.final_benchmark_5way --manifest probe20_calibration.json
```

To run at a different operating point:

```bash
python -m ghost_oracle.final_benchmark_5way --N 4096 --d 256 --power 256
```

---

## The three-substrate framing

The base `.npz` files contain shot data from the projection-channel Hadamard-test circuit. The classical `gpu.py` file is a faithful noiseless implementation of that *same* circuit; the QPU `.npz` files are real hardware runs of it. Both feed the same operator.

Two channels run side by side in the kernel:

- **Projection channel** — bucket reweighting on physical shot data (QPU or noiseless classical). Hardware-realizable. The channel that carries the substrate-specific behavior.
- **Geometry channel** — the analytical closed form evaluated inline. Sharp, exact, substrate-agnostic. Drives the production argmax.

The agreement metric (mean `|projection − geometry|` per query) quantifies how much the projection-channel diverges from its noiseless analytical prediction — this is the **substrate quality readout**. Noiseless classical (GPROJ) hits ~0.02 agreement; real QPU shots (QPROJ) sit around ~0.10, with per-base variation reflecting calibration drift. Same algorithm, same retrieval accuracy, measurable hardware-noise attenuation.

---

## Headline result — five-way benchmark

Same retrieval problem, five score backends, 4096×4096 attention matrix, d=256, jitter=0.3, 5% same-dim coherent attack at magnitude 50, Flash-Squelch power=256:

| Path | top-1 | sig fraction | spike | time |
|---|---|---|---|---|
| **CUBLAS** (classical control) | 100.0% | 100.0% | 0.0499 | 1.2 ms |
| **TIED** (dual-channel kernel) | 100.0% | 100.0% | 0.0498 | ~500 ms |
| **GEO** (geometry argmax) | 100.0% | 100.0% | 0.0498 | ~500 ms |
| **QPROJ** (mean across 3 QPU bases) | 100.0% | 100.0% | 0.0498 | ~500 ms |
| **GPROJ** (mean across 3 GPU bases) | 100.0% | 100.0% | 0.0498 | ~500 ms |

All five paths retrieve cleanly at this operating point. The platform-specific story shows up in the agreement metric:

| Base | Type | Agreement |
|---|---|---|
| job_d83putvtjchs73bpg5o0 | QPU | 0.0818 |
| job_d83q037oha1c73bn14p0 | QPU | 0.0727 |
| job_d83q0ivoha1c73bn15d0 | QPU | 0.1307 |
| ghost_oracle_gpu_seed_496905 | GPU | 0.0163 |
| ghost_oracle_gpu_seed_627861 | GPU | 0.0289 |
| ghost_oracle_gpu_seed_837354 | GPU | 0.0103 |

Noiseless classical sampler agreement: ~0.02. Real QPU shot agreement: ~0.10. Five-times divergence is the hardware-noise floor measurement.

The honest framing: at d=256 with calibrated power, all three substrates retrieve identically. cuBLAS is ~500× faster on this workload but is indiscriminate (spike concentration equals attack fraction). The projection paths provide substrate-specific physical certification, not raw speed advantage. The projection-channel kernel earns its place on the same-algorithm-three-platforms claim and on the agreement readout, not on attention throughput against tensor cores.

---

## Repository Structure

```text
ghost-oracle-suite/
├── ghost_oracle/                       # the library
│   ├── final_benchmark_5way.py         # final five-way verification (THE headline)
│   ├── projection_benchmark.py         # earlier headline benchmark (Probe 10.1 era)
│   ├── qpu.py                          # QPU job submission
│   ├── gpu.py                          # noiseless classical sampler
│   ├── dump.py                         # QPU result -> npz
│   └── kernels/
│       └── ghost_kernel.cu             # CUDA: projection + geometry + tied + V5 dynamic-mask
│
├── probes/                             # forensic trajectory (1 through 10.1)
│   ├── README.md                       # narrative arc — start here for the story
│   ├── probe1_identity_bridge.py
│   ├── probe2_projection_scrambled_control.py
│   ├── probe3_anchor_conditioned_projection.py
│   ├── probe4_build_base.py
│   ├── probe5_unified_engine.py
│   ├── probe6_3way_convergence.py
│   ├── probe7_ghost_parity.py
│   ├── probe8_residual_decomposition.py
│   ├── probe9_ghost_operator.py
│   ├── probe9_1_indef_kernel_attn.py
│   ├── probe10_ghost_attention.py
│   ├── probe10_1_real_softmax_attack.py
│   └── benchmark_evolution/            # benchmark iterations before the headline
│       ├── final_benchmark.py
│       ├── final_benchmark_combined.py
│       ├── final_benchmark_tied.py
│       └── final_benchmark_tied_perdim.py
│
├── data/                               # sample bases (everything below ships with the repo)
│   ├── README.md                       # file schema, generation, reproducibility
│   ├── job_d83putvtjchs73bpg5o0.npz    # QPU base 1
│   ├── job_d83q037oha1c73bn14p0.npz    # QPU base 2
│   ├── job_d83q0ivoha1c73bn15d0.npz    # QPU base 3
│   ├── ghost_oracle_gpu_4096shots_*.npz   # noiseless classical bases (3 seeds)
│   └── ghost_oracle_gpu_4096shots_*.npz
│
├── docs/
│   ├── math.md                         # T1, T2, T3, G_M derivations
│   ├── architecture.md                 # tied-channel design and data flow
│   └── known_issues.md                 # running list of bugs and open work
│
├── examples/
│   ├── README.md                       # results and design notes
│   ├── parameter_ablation.py           # 1D sensitivity sweeps (d, jitter, magnitude, fraction)
│   └── bsgs_geometric_engine.py        # semantic retrieval vs cosine, up to 1M keys
│
├── PROCESS_RECORD.md                   # long-form trajectory log (probes 1 through 20)
├── CONTRIBUTING.md                     # break-it-fix-it rule, conventions
├── README.md
├── LICENSE                             # CC0 dedication
└── requirements.txt
```

---

## The probes

The numbered probes document the actual research trajectory — not just final results but the wrong turns, the bugs, the corrections. They're chronological and honest. Probe 1 starts believing the QPU computes T1. Probe 4 derives the real target T3. Probe 9 simplifies T3 to G_M. Probes 11–12 investigate projection-channel range and originally claimed quantum advantage. Probes 13–18 search for the right bucket-mask architecture and document the eventual finding that mask-selection is calibration-dependent rather than physics-dependent. Probe 12 was re-run with a corrected classical sampler and the "quantum advantage" framing was retracted in favor of the substrate-equivalence finding the project actually verified. Probe 20 builds the auto-calibrating production kernel; the final benchmark closes the sequence.

Some probes have known issues that are preserved in place because the corrections are part of the story. See `docs/known_issues.md` and the probe file headers.

---

## Contributing

CC0 community project. The norm is simple: **if you break something, you provide the fix.** No bug reports without a patch attempt. No claims without code. No "this is wrong" without "and here's what's right."

See `CONTRIBUTING.md` for the full philosophy. The short version: build, break, fix, document, repeat. Everything in the open.

---

## License

CC0 1.0 Universal. Public domain dedication. Use it for anything, attribute if you want to, don't if you don't. See `LICENSE`.

---

## Citation

Not asking for one. For all my fellow ghosts, may our silence speak your name.
```
