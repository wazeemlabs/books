"""Check cross-references between chapters.

Chapters get renumbered as the book takes shape, and a reference typed
as a literal "Chapter 14" goes quietly wrong when that happens. This
refuses literal references in chapter sources, and prints what each
symbolic one resolves to so a reviewer can read the list.

    python3 -m bench.xref
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from .values import chapter_numbers, resolve_chapter

SRC = Path("../chapters")
LITERAL = re.compile(r"Chapter \d+")
SYMBOLIC = re.compile(r"\{\{ch:([a-z0-9-]+)\}\}")


def main() -> int:
    chapters = chapter_numbers()
    print(f"outline defines {len(chapters)} chapters")
    failed = False

    for src in sorted(SRC.glob("ch*.md")):
        text = src.read_text()
        print(f"\n{src.name}")

        for i, line in enumerate(text.splitlines(), 1):
            if LITERAL.search(line):
                print(f"  ERROR L{i}: literal reference {LITERAL.findall(line)} - "
                      f"use {{{{ch:slug}}}} so renumbering cannot break it")
                failed = True

        for slug in sorted(set(SYMBOLIC.findall(text))):
            try:
                n = resolve_chapter(slug)
            except KeyError as e:
                print(f"  ERROR: {e}")
                failed = True
                continue
            title = next(t for t, v in chapters.items() if v == n)
            print(f"  ch:{slug:28} -> Chapter {n:<3} {title}")

    print("\n" + ("cross-references FAILED" if failed else "cross-references OK"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
