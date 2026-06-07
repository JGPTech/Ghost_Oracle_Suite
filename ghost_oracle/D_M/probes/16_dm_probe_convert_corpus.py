from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
src = DATA_DIR / "d_m_probe_corpus_raw.txt"
dst = DATA_DIR / "d_m_probe_corpus.txt"

text = src.read_text(encoding="utf-8", errors="replace")

# Remove Gutenberg header/footer if present.
start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", text, re.I | re.S)
if start:
    text = text[start.end():]
if end:
    text = text[:end.start()]

# Split into sentence-ish chunks and pack 3 sentences per line.
sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
lines = []
buf = []

for s in sentences:
    s = s.strip()
    if len(s) < 40:
        continue
    buf.append(s)
    if len(buf) >= 3:
        line = " ".join(buf)
        if 120 <= len(line) <= 1200:
            lines.append(line)
        buf = []

dst.write_text("\n".join(lines[:1000]), encoding="utf-8")
print(f"Wrote {min(len(lines), 1000)} lines to {dst}")
