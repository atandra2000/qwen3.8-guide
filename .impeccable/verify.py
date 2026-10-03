#!/usr/bin/env python3
"""Authoritative post-rewrite check: compares the working tree against git HEAD.

facts.py checks against a snapshot file, which concurrent agents kept
overwriting. This reads the committed version straight out of git, so the
baseline cannot drift mid-run.

  python3 .impeccable/verify.py            # against HEAD
  python3 .impeccable/verify.py <ref>      # against any ref
"""
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import facts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PARTS = sorted((ROOT / "_parts").glob("0*.html"))
ALL = PARTS + [ROOT / "README.md"]
REF = sys.argv[1] if len(sys.argv) > 1 else "HEAD"


def committed(path: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{REF}:{path.relative_to(ROOT)}"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def from_text(raw: str, fn):
    """Run one of facts.py's extractors against a string instead of a path."""
    tmp = ROOT / ".impeccable" / ".verify.tmp"
    tmp.write_text(raw, encoding="utf-8")
    try:
        return fn(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    bad = 0
    for f in ALL:
        before_raw = committed(f)
        before_tok = from_text(before_raw, facts.tokens)
        before_tag = from_text(before_raw, facts.tag_census)
        after_tok, after_tag = facts.tokens(f), facts.tag_census(f)

        gone = before_tok - after_tok
        if gone:
            bad += len(gone)
            print(f"LOST TOKEN in {f.name} ({len(gone)}):")
            for t in sorted(gone)[:20]:
                print(f"   - {t}")

        gdiff, gadd = before_tag - after_tag, after_tag - before_tag
        if gdiff or gadd:
            bad += 1
            print(f"TAG CHANGE in {f.name}:")
            for k, n in sorted(gdiff.items()):
                print(f"   -{k} x{n}")
            for k, n in sorted(gadd.items()):
                print(f"   +{k} x{n}")

        for i, txt in facts.em_in_prose(f):
            bad += 1
            print(f"EM-DASH LEFT {f.name}:{i} {txt}")

    print(f"OK vs {REF}: prose only, all tokens and tags intact"
          if not bad else f"FAIL vs {REF}: {bad} issues")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()