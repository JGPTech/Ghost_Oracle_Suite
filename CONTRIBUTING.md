# Contributing

This is a CC0 project. Anyone can fork, modify, redistribute, or rebuild this work without asking. The intent is the opposite of gatekeeping: build, break, fix, document, push it forward.

What follows is how to contribute productively, not how to qualify.

---

## The one rule

**If you break something, you provide the fix.**

Not a hard requirement — nobody will lock the gate against a bug report without a patch. But this project moves on fixes-with-bugs, not bugs alone. A PR titled "X is wrong, here's what's right" gets merged. An issue titled "X is wrong" sits in the queue until someone (probably you, eventually) has time to look.

This includes the corollaries:

- No claims without code that demonstrates them.
- No "this is broken" without a reproduction.
- No "this should be different" without a proposed difference.

The honest reason: this is a small project with no funding. The most expensive resource is attention, and a fix takes about the same amount of attention as a careful bug report — so the asymmetry of "report only" is what we can't afford. A community where contributors do both keeps the project moving.

---

## What good contributions look like

**A PR that adds a probe.** New numbered probe (`probe11_*` or higher), self-contained, with a `HISTORICAL CONTEXT` block in the header explaining what trajectory it advances. Reuse the existing `auto_find_base` helper, the suite-wide `ANGLE_SCALE` and `ALPHA_NORM` constants, and the section-header conventions other probes use. Document the result in the probe header docstring; if it advances the open questions in `docs/known_issues.md` or PROCESS_RECORD Part 10, update those files in the same PR.

**A PR that fixes a known issue.** Pick something from `docs/known_issues.md`. The Broken section has explicit fix paths for most entries. Mark the issue as resolved in `known_issues.md` and add a one-line note in `PROCESS_RECORD.md` pointing at your PR.

**A PR that adds an example.** New script in `examples/`, with results in the existing `examples/README.md` style (tables, one paragraph of interpretation, a "Critical:" callout if there's a non-obvious failure mode like the L2-normalization issue documented there). Aim for the script to run end-to-end against the sample bases in `data/` so a reader can reproduce without setup.

**A PR that improves documentation.** Doc-only PRs are welcome. Especially appreciated: typo and math fixes in `docs/math.md`, missing failure modes in `docs/known_issues.md`, clarifications to `docs/architecture.md`.

**A PR that adds a benchmark or sweep.** New entry in `examples/` (not the main `projection_benchmark.py`, which is the headline result and stays small). Frame it as "G_M on workload X" and report against a real baseline (cuBLAS DP, cosine similarity, an HNSW index, whatever fits). Numbers in the README table; one paragraph of interpretation.

---

## What unhelpful contributions look like

- Issues without a reproduction or a patch attempt.
- "This project is wrong because [philosophical objection]." Take it to a fork.
- PRs that add dependencies without justification. Current dependencies: numpy, cupy (optional), qiskit/qiskit-ibm-runtime (only for `qpu.py` and `dump.py`). Anything new needs to earn its place.
- PRs that touch suite-wide constants (`ANGLE_SCALE`, `ALPHA_NORM`, `NUM_TILES` default) without updating every consumer and verifying the headline benchmark still produces the same numbers.
- Sweeping refactors that improve code style at the cost of trajectory legibility. The buggy versions of probes 8.2, 9, and 10 are kept on purpose — their corrections are part of the research record. Don't delete them, don't rewrite them, don't quietly fix them in place.

---

## Code conventions

These aren't enforced by tooling; they're norms the existing code follows.

- **Constants live in module-level CONFIG blocks at the top of each file.** Match the existing format: `ANGLE_SCALE = 1.05`, `ALPHA_NORM = 0.9127`, `NUM_TILES = 12` (probes) or `16` (production).
- **Section headers are `# ===...===` blocks, 79 characters wide.** CUDA section headers in `ghost_kernel.cu` are `// ===...===` 80 characters wide (one extra for the `//`).
- **`auto_find_base(kind)` with `kind ∈ {"qpu", "gpu"}`** is the canonical helper for finding bases in `data/`. Use it in new scripts rather than reinventing the search.
- **CLI args use argparse with `formatter_class=argparse.ArgumentDefaultsHelpFormatter`.** Every probe and example follows this; help text shows defaults automatically.
- **Probe headers have a `HISTORICAL CONTEXT:` block at the bottom of the docstring.** It explains what the probe contributes to the trajectory, what the original run reported, and how the current run differs (if at all) from the historical numbers. Keep this convention for new probes — it's how readers of `probes/README.md` make sense of the arc.
- **Keep probe scripts self-contained.** Each probe is one runnable file. Shared helpers between probes are duplicated rather than factored out, so a reader can understand any single probe without chasing imports. This is a deliberate trade against DRY in favor of legibility.
- **Known-broken probes get a SUPERSEDED or KNOWN ISSUE banner in the header** plus an entry in `docs/known_issues.md`. See Probe 8.2 (Phase C) and Probe 10 for examples of how these are marked.

---

## How to claim a piece of work

There is no formal claim system. If you want to work on something:

1. Open an issue describing what you're going to do. Doesn't need to be detailed — "I'm going to take a shot at the Probe 8.2 joint-fit replacement" is fine.
2. Submit a PR when you have something to show. Drafts are welcome.

If two people end up working on the same thing in parallel, both PRs get reviewed and the better one merges. No drama; this is CC0, the work belongs to nobody.

---

## Review and merging

Small project. One maintainer at the moment. Reviews are direct and technical: numbers checked against the diff, math checked against `docs/math.md`, the headline benchmark re-run if anything touches the production path.

Likely outcomes for a PR:

- **Merged** — works, tests pass, doc updates included.
- **Comment with a question or suggestion** — usually a small thing. Respond and we'll go again.
- **Closed with a reason** — rare, but happens when a PR conflicts with the project direction or breaks something load-bearing.

If you disagree with a close, open an issue and make the case. If the case is good, it gets reopened.

---

## Licensing

The project is CC0 1.0 Universal — public domain dedication. By contributing, you confirm your contribution is also CC0 or under a compatible license (MIT, BSD, Apache 2.0 with your explicit waiver of patent claims).

You don't need to sign anything. Don't add copyright headers to new files; the LICENSE at the repo root applies to everything.

If you want attribution for your contribution, the git log is the canonical record. Beyond that, there's no `AUTHORS` file and no plans for one — but a contribution to `PROCESS_RECORD.md` describing what you did and what you found is welcome and is the de facto way the project remembers who did what.

---

## Citing

The README says it: not asking for one. If this work helps your work, the best thing you can do is push it forward — file an issue, send a PR, fork it and take it somewhere we didn't think of.