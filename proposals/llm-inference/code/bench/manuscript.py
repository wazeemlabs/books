"""Render a chapter source into the manuscript, or check that it is current.

A chapter source contains three kinds of hole:

  {{name}}                             a number, resolved from bench.values
  <!-- include: tables/x.md -->        a generated table, inserted below it
  <!-- listing: path Class.method -->  source code, extracted from the repo

A listing is never typed into a chapter. It names a file and a symbol,
and the renderer reads the code out of the file, so a listing cannot
survive a refactor unchanged. Where a chapter genuinely needs an
abridged listing -- a few lines lifted from different places -- it is
written literally but must be claimed by an `<!-- abridged: path -->`
marker, and `make audit` checks every one of its lines against that
file.

    python3 -m bench.manuscript          render every chapter
    python3 -m bench.manuscript --check  fail if any rendered file is stale

The check is what CI runs: it makes a measurement and the sentence that
quotes it impossible to separate.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from .listings import extract

from .values import load, resolve_chapter

SRC = Path("../chapters")
OUT = Path("..")
INCLUDE = re.compile(r"^<!-- include: (\S+) -->$", re.M)
LISTING = re.compile(r"^<!-- listing: (\S+) (\S+)(?: (no-docstring))? -->$", re.M)
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
CHAPTER_REF = re.compile(r"\{\{ch:([a-z0-9-]+)\}\}")


def render(text: str, values: dict[str, str]) -> str:
    missing: list[str] = []

    # Symbolic chapter references first, so renumbering the outline
    # cannot leave a stale "Chapter 14" behind in the prose.
    text = CHAPTER_REF.sub(lambda m: f"Chapter {resolve_chapter(m.group(1))}", text)

    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in values:
            missing.append(name)
            return m.group(0)
        return values[name]

    text = PLACEHOLDER.sub(sub, text)
    if missing:
        raise KeyError(f"no value for: {', '.join(sorted(set(missing)))}")

    def insert(m: re.Match) -> str:
        path = Path(m.group(1))
        if not path.exists():
            raise FileNotFoundError(f"{path} - run `make tables` first")
        return m.group(0) + "\n" + path.read_text().rstrip("\n")

    def listing(m: re.Match) -> str:
        code = extract(m.group(1), m.group(2), drop_docstring=bool(m.group(3)))
        return m.group(0) + "\n\n```python\n" + code + "\n```"

    text = LISTING.sub(listing, text)
    return INCLUDE.sub(insert, text)


def output_for(stem: str) -> Path:
    """Where a source renders to.

    Chapters render to CHAPTER-nn-DRAFT.md. Design decision records
    (STANDARDS.md section 5) close a Part rather than belonging to one
    chapter, and render to DDR-n-DRAFT.md.
    """
    if stem.startswith("ddr"):
        return OUT / f"DDR-{stem.removeprefix('ddr')}-DRAFT.md"
    return OUT / f"CHAPTER-{stem.removeprefix('ch')}-DRAFT.md"


def sources() -> list[Path]:
    return sorted(SRC.glob("ch*.md")) + sorted(SRC.glob("ddr*.md"))


def main(argv: list[str]) -> int:
    check = "--check" in argv
    stale: list[str] = []
    for src in sources():
        values = load(src.stem)
        out = output_for(src.stem)
        new = render(src.read_text(), values)
        if check:
            if not out.exists() or out.read_text() != new:
                stale.append(str(out))
            continue
        out.write_text(new)
        print(f"  {out.name}")
    if check:
        if stale:
            print("STALE, rerun `make ch12`:\n  " + "\n  ".join(stale), file=sys.stderr)
            return 1
        print("manuscript is in sync with the measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
