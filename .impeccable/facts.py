#!/usr/bin/env python3
"""Gate for the humanizer rewrite. Verifies that only prose changed.

Three checks:
  1. no load-bearing token lost   (numbers, units, math, section refs, code)
  2. no HTML tag/class/id/attribute altered or added
  3. no em-dash left in prose text nodes (label captions exempt)

Usage:  python3 .impeccable/facts.py snapshot > /tmp/before.txt
        python3 .impeccable/facts.py check /tmp/before.txt
"""
import re
import sys
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = sorted((ROOT / "_parts").glob("0*.html")) + [ROOT / "README.md"]

NUM = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:%|[KMGTP]?i?B|[kKmMgGtT]b|GB|MB|KB|TB|GB/s|TFLOP/s|FLOP/s|"
    r"FLOP/byte|byte/s|tokens?|ms|ns|s\b|h\b|×|x\b|bits?|heads?|layers?|dim|pt|ppm)?"
)
MATH = re.compile(r"\$[^$]+\$")
SEC = re.compile(r"§+\d+(?:\.\d+)?|§§\d+")
FIGTAB = re.compile(r"\b(?:Fig\.|Figure|Table|eq\.|Eq\.)\s*\(?\d+")
ARXIV = re.compile(r"arXiv:\d{4}\.\d{4,5}")
# Only true value carriers are immutable. <strong>/<em> mark labels and run-in
# headings here, so they hold prose the rewrite owns; any number or identifier
# inside them is still caught by NUM/MATH/code/mono below.
KEEP = re.compile(
    r"<code>([^<]+)</code>|<span class=\"mono\">([^<]+)</span>"
)
# A real tag is "<name ...>" or "</name>" at a word boundary. The "or" matters:
# LaTeX like "$< 0$" would otherwise parse as a tag and make the census lie.
TAG = re.compile(r"<(?:/?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^<>]*?)?/?|!--[\s\S]*?--)>")
# Caption label ("Table 5 - ...", "Fig. 3 - ...") and proper titles keep their dash:
# both are fixed names, not prose the rewrite owns.
LABEL_CAP = re.compile(
    r"^(?:Table|Fig\.|Figure)\s*\d+\s*—"
    r"|Qwen 3\.8-27B\s*—\s*The Architecture"
    r"|^\d{2}\s*·\s*.*?\s*—\s*\S"
)
# Bibliographic entry: "Author et al. 2024 - "Title." (id)."
BIBLIO = re.compile(r"^[\w.\-]+(?:\s+et\s+al\.|\s+Team)?\.?\s*\d{4}\s*—")


def prose(path: Path):
    raw = path.read_text(encoding="utf-8")
    return re.sub(r"<(script|style)[\s\S]*?</\1>", " ", raw)


def tokens(path: Path):
    raw = prose(path)
    out = set()
    for rx in (NUM, MATH, SEC, FIGTAB, ARXIV):
        for m in rx.findall(raw):
            # "1000," and "1000" are the same number; the comma is the sentence's.
            m = re.sub(r"[\s,]+$", "", re.sub(r"\s+", " ", m).strip())
            if m:
                out.add(m)
    for m in KEEP.findall(raw):
        t = next((g for g in m if g), "")
        if t.strip():
            out.add(t.strip())
    return out


def tag_census(path: Path):
    """Every distinct tag with its attributes, counted. A structural change shows here."""
    raw = prose(path)
    c = Counter()
    for tag in TAG.findall(raw):
        name = re.match(r"</?([a-zA-Z0-9-]+)", tag)
        if not name:
            c[json.dumps(tag)] += 1
            continue
        attrs = sorted(re.findall(r'([a-zA-Z-]+)=(?:"[^"]*"|\'[^\']*\')', tag))
        c[json.dumps([name.group(1), attrs])] += 1
    return c


def em_in_prose(path: Path):
    raw = prose(path)
    hits = []
    for i, line in enumerate(raw.split("\n"), 1):
        if "—" not in line:
            continue
        text = line.replace("&mdash;", "—")
        # strip tags to read the visible text
        visible = TAG.sub("", text).strip()
        # strip a markdown heading marker so the title exemption matches
        visible_nohash = re.sub(r"^#{1,6}\s+", "", visible)
        if not visible:
            continue
        if (LABEL_CAP.match(visible_nohash) or BIBLIO.match(visible_nohash)
                or LABEL_CAP.match(visible) or BIBLIO.match(visible)):
            continue
        # dashes inside a code/table cell or widget readout are not prose
        if re.search(r"<code>[^<]*—", line) or re.search(r'class="mono">[^<]*—', line):
            continue
        if "—" in visible:
            hits.append((i, visible[:120]))
    return hits


def main():
    mode = sys.argv[1]
    if mode == "snapshot":
        # JSON: tokens can contain tabs and quotes, TSV cannot survive them.
        snap = {"tokens": {}, "tags": {}}
        for f in FILES:
            snap["tokens"][f.name] = sorted(tokens(f))
            snap["tags"][f.name] = dict(tag_census(f))
        print(json.dumps(snap, indent=1, ensure_ascii=False))
        return

    snap = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    before_tok = {k: set(v) for k, v in snap["tokens"].items()}
    before_tag = {k: Counter(v) for k, v in snap["tags"].items()}

    bad = 0
    for f in FILES:
        now_tok, now_tag = tokens(f), tag_census(f)
        gone = before_tok.get(f.name, set()) - now_tok
        if gone:
            bad += len(gone)
            print(f"LOST TOKEN in {f.name} ({len(gone)}):")
            for t in sorted(gone)[:25]:
                print(f"   - {t}")
        gdiff = before_tag.get(f.name, Counter()) - now_tag
        gadd = now_tag - before_tag.get(f.name, Counter())
        if gdiff or gadd:
            bad += 1
            print(f"TAG CHANGE in {f.name}:")
            for k, n in sorted(gdiff.items(), key=str):
                print(f"   -{k} x{n}")
            for k, n in sorted(gadd.items(), key=str):
                print(f"   +{k} x{n}")

        for i, txt in em_in_prose(f):
            bad += 1
            print(f"EM-DASH LEFT {f.name}:{i} {txt}")

    print("OK: prose only, all tokens and tags intact" if not bad else f"FAIL: {bad} issues")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()