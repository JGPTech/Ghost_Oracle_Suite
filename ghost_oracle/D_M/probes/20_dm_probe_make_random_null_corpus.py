#!/usr/bin/env python3
"""
Random-text null control for the D_M task utility retrieval benchmark.

Generates a corpus of high-entropy random lines with no linguistic structure.
Feed it to the benchmark exactly like the real corpus:

    --text-file random_null_corpus.txt --split-lines

How to read the light_noise result:

  * Score stays ~72-80%  ->  the signal is byte-level fingerprinting of the raw
                             line, independent of content. Not the card.
  * Score collapses to chance  ->  content mattered after all; worth digging.

Notes:
  --match-file mirrors the exact per-line length distribution of your real
  corpus, so line length cannot be the confound that explains a difference.

  --seed is fixed on purpose. Rerun and diff the benchmark CSVs: bit-identical
  results across runs (and across cards) mean deterministic fp32 math, which is
  the opposite of a physical-delay residue.
"""

import argparse
import random
import string
from pathlib import Path

ALPHABET = string.ascii_letters + string.digits + string.punctuation + " "


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a random-text null corpus.")
    p.add_argument("--out", type=Path, default=Path("random_null_corpus.txt"))
    p.add_argument("--lines", type=int, default=5000)
    p.add_argument("--line-length", type=int, default=120)
    p.add_argument("--match-file", type=Path, default=None,
                   help="Mirror the per-line character lengths of this file instead of --line-length.")
    p.add_argument("--seed", type=int, default=1234567)
    args = p.parse_args()

    rng = random.Random(args.seed)

    if args.match_file:
        src = args.match_file.read_text(encoding="utf-8", errors="replace").splitlines()
        lengths = [len(line) for line in src if line.strip()]
        if not lengths:
            raise SystemExit(f"no non-empty lines found in {args.match_file}")
    else:
        lengths = [args.line_length] * args.lines

    lines = ["".join(rng.choice(ALPHABET) for _ in range(n)) for n in lengths]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {len(lines)} random lines -> {args.out}")
    print(f"line length: min={min(lengths)} max={max(lengths)} mean={sum(lengths)/len(lengths):.1f}")


if __name__ == "__main__":
    main()
