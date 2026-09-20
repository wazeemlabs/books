"""Check that no chapter uses a term before the chapter that explains it.

The book is written for a reader who is assumed to know nothing, which
is a promise that decays silently: a term gets used in Chapter 3,
explained properly in Chapter 8, and nobody notices because the author
knew what it meant all along.

GLOSSARY.md lists every term with the chapter that introduces it. That
chapter claims it with a marker beside the explanation:

    <!-- defines: arithmetic intensity, memory-bound -->

This checks three things: every glossary term is claimed by the chapter
the glossary names, every claim matches the glossary, and no term is
used before it is claimed.

    python3 -m bench.terms
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SRC, GLOSSARY = Path("../chapters"), Path("../GLOSSARY.md")
# Which chapter each design decision record is read after.
PART_ENDS = {"ddr1": 19}
ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|", re.M)
MARKER = re.compile(r"<!--\s*defines:\s*([^>]+?)\s*-->")


def strip_uncounted(text: str) -> str:
    """Remove the places a term may appear without being 'used'.

    Code, markers, links and the Sources list are not prose; a term
    inside them does not owe the reader an explanation.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.S)       # code blocks
    text = re.sub(r"`[^`]*`", " ", text)                     # inline code
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)      # markers
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)               # placeholders
    text = re.sub(r"\]\([^)]*\)", " ", text)                 # link targets
    text = re.sub(r"^## Sources.*?(?=^## )", " ", text, flags=re.S | re.M)
    return text


# Words that are technical here and ordinary English everywhere else.
# "a block of numbers", a table column headed "Value", "the key question":
# no checker can tell those from the technical sense, so these terms are
# listed in the glossary for the reader but their order is not enforced.
# Any term added here must still be defined somewhere.
ORDINARY_ENGLISH = {"value", "block", "key", "head", "weight", "precision",
                    "latency", "median", "matrix", "vector", "concurrency",
                    # "at different exponents", "grows exponentially":
                    # Chapter 4 means the ordinary word, not the field of a
                    # floating-point number that Chapter 22 names.
                    "exponent"}


def glossary() -> dict[str, int]:
    """Map every term (and alias) to the chapter that must introduce it."""
    out: dict[str, int] = {}
    for names, chapter in ROW.findall(GLOSSARY.read_text()):
        if names.lower() == "term":
            continue
        for name in names.split(","):
            name = name.strip().lower()
            if name:
                out[name] = int(chapter)
    return out


def main() -> int:
    terms = glossary()
    chapters = {int(p.stem[2:]): p for p in SRC.glob("ch*.md")}
    # A design decision record closes a Part, so it may use every term
    # the Part explained. It is checked at the number of its last
    # chapter, which is the position it is read from.
    records = {p: PART_ENDS[p.stem] for p in SRC.glob("ddr*.md")}
    text = {n: strip_uncounted(p.read_text()) for n, p in chapters.items()}
    for p, n in records.items():
        text[n] = text.get(n, "") + "\n" + strip_uncounted(p.read_text())
    claims: dict[str, int] = {}
    for n, p in list(chapters.items()) + [(n, p) for p, n in records.items()]:
        for marker in MARKER.findall(p.read_text()):
            for name in marker.split(","):
                claims[name.strip().lower()] = n

    problems, drafted = [], sorted(chapters)
    print(f"{len(terms)} glossary terms, {len(drafted)} chapters drafted")

    for term, want in sorted(terms.items()):
        if want not in chapters:
            continue                       # its chapter is not written yet
        got = claims.get(term)
        if got is None:
            problems.append(f"'{term}' is never claimed; chapter {want} should "
                            f"carry <!-- defines: {term} -->")
            continue
        if got != want:
            problems.append(f"'{term}' claimed by chapter {got}, glossary says {want}")
            continue
        if term in ORDINARY_ENGLISH:
            continue                   # defined, but order is not enforceable
        pattern = re.compile(rf"\b{re.escape(term)}s?\b", re.I)
        for n in drafted:
            if n < want and pattern.search(text[n]):
                m = pattern.search(text[n])
                line = text[n][max(0, m.start() - 60):m.end() + 40]
                problems.append(
                    f"'{term}' used in chapter {n} but explained in {want}: "
                    f"...{' '.join(line.split())}...")
                break

    for term, n in sorted(claims.items()):
        if term not in terms:
            problems.append(f"chapter {n} claims '{term}', which is not in the glossary")

    if problems:
        print(f"\nTERMS FAILED - {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("terms: every term is explained before it is used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
