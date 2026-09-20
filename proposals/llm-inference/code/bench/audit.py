"""Whole-manuscript audit: the checks `make check` does not make.

`make check` proves each chapter matches its own measurements. This
proves the manuscript hangs together: that every figure drawn is
actually used, every figure cited exists, figures are numbered in the
order they appear, every generated table is included somewhere, and
every chapter carries the sections STANDARDS.md requires.

    python3 -m bench.audit
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SRC, FIGS, TABLES = Path("../chapters"), Path("figures"), Path("tables")
RESULTS = Path("results")
DRAFTS = Path("..")

# STANDARDS.md section 5, with the Part I variant recorded in section 5.1.
REQUIRED = ["## Objectives", "## Why it matters", "## Numbers to remember",
            "## Sources", "## Exercises"]
# STANDARDS.md section 5.1: the substantive rule is that a chapter is
# evidence-backed -- it shows something generated from a results file --
# not that it carries a particular heading. Part I and walkthrough
# chapters legitimately distribute their evidence through the narrative.
CAVEAT = ["## Where it breaks", "## Where this is soft",
          "## Where this chapter simplifies", "## Where this estimate is soft"]

IMG = re.compile(r"!\[[^\]]*\]\((code/figures/([a-z0-9-]+)\.svg)\)")
FIGNUM = re.compile(r"^\*\*Figure (\d+)\.(\d+)\*\*", re.M)
INCLUDE = re.compile(r"^<!-- include: (tables/\S+) -->$", re.M)


def cross_chapter_consistency() -> list[str]:
    """The same quantity must not have two answers in two chapters.

    tinyserve/reference.py exists so these agree by construction. This
    checks that they actually do, in the results the chapters were
    rendered from -- including a chapter that quotes a figure another
    chapter computes.
    """
    import json
    found: dict[str, dict[str, float]] = {}

    def note(quantity: str, chapter: str, value) -> None:
        if value is not None:
            found.setdefault(quantity, {})[chapter] = float(value)

    r = {p.stem: json.loads(p.read_text()) for p in RESULTS.glob("ch*.json")}

    if "ch01" in r:
        a = r["ch01"]["assumptions"]
        note("8B parameters", "ch01", a["params"])
        note("KV bytes per token", "ch01", a["kv_bytes_per_token"])
        note("HBM bytes/s", "ch01", a["hbm_bytes_per_s"])
        note("concurrent sequences", "ch01", a["max_concurrent"])
    if "ch02" in r:
        c = r["ch02"]["cost"]["reference_8b"]
        note("8B parameters", "ch02", c["params"])
        note("KV bytes per token", "ch02", c["kv_bytes_per_token"])
        note("HBM bytes/s", "ch02", c["hbm_bytes_per_s"])
    if "ch03" in r:
        h = r["ch03"]["reference_8b"]
        note("8B parameters", "ch03", h["config"]["params"])
        note("HBM bytes/s", "ch03", h["hardware"]["hbm_bytes_per_s"])
    if "ch04" in r:
        a = r["ch04"]["accelerator"]
        note("HBM bytes/s", "ch04", a["hbm_bytes_per_s"])
        note("8B weight bytes", "ch04", a["weight_bytes_8b"])
        note("accelerator break-even", "ch04", a["ridge_flop_per_byte"])
    if "ch03" in r:
        note("accelerator break-even", "ch03",
             r["ch03"]["reference_8b"]["hardware"]["ridge_flop_per_byte"])
    if "ch02" in r:
        note("8B weight bytes", "ch02",
             r["ch02"]["cost"]["reference_8b"]["weight_bytes_read_per_token"])
    if "ch10" in r:
        note("tinyserve parameters", "ch10", r["ch10"]["baseline"]["params"])
    if "ch12" in r:
        note("tinyserve parameters", "ch12", r["ch12"]["model"]["params"])
    if "ch05" in r:
        note("concurrent sequences", "ch05", r["ch05"]["case_study"]["max_concurrent"])
    if "ch13" in r:
        note("KV bytes per token", "ch13", r["ch13"]["hardware"]["kv_bytes_per_token"])
        paged = next((x for x in r["ch13"]["policies"] if x["policy"] == "paged_16"), None)
        note("concurrent sequences", "ch13", paged and paged["concurrent_seqs"])

    problems = []
    for quantity, by_chapter in sorted(found.items()):
        values = set(round(v, 6) for v in by_chapter.values())
        where = ", ".join(f"{c}={v:,.0f}" for c, v in sorted(by_chapter.items()))
        if len(values) > 1:
            problems.append(f"{quantity} disagrees across chapters: {where}")
        else:
            print(f"  {quantity:24} agrees ({where})")
    return problems


def main() -> int:
    problems: list[str] = []
    used_figs: set[str] = set()
    used_tables: set[str] = set()

    chapters = sorted(SRC.glob("ch*.md"))
    print(f"auditing {len(chapters)} chapters\n")

    for src in chapters:
        text = src.read_text()
        num = int(src.stem.removeprefix("ch"))
        issues: list[str] = []

        # Figures: cited must exist; drawn must be cited; numbered in order.
        cited = IMG.findall(text)
        for path, stem in cited:
            used_figs.add(stem)
            if not (FIGS / f"{stem}.svg").exists():
                issues.append(f"cites {path}, which does not exist")
        nums = [(int(a), int(b)) for a, b in FIGNUM.findall(text)]
        if len(nums) != len(cited):
            issues.append(f"{len(cited)} figures embedded but {len(nums)} numbered captions")
        for i, (chap, idx) in enumerate(nums, start=1):
            if chap != num:
                issues.append(f"caption 'Figure {chap}.{idx}' is in chapter {num}")
            if idx != i:
                issues.append(f"caption 'Figure {chap}.{idx}' appears in position {i}")

        # Tables.
        for t in INCLUDE.findall(text):
            used_tables.add(Path(t).name)
            if not Path(t).exists():
                issues.append(f"includes {t}, which does not exist")

        # Required sections.
        for heading in REQUIRED:
            if heading not in text:
                issues.append(f"missing section {heading!r}")
        if not (cited or INCLUDE.findall(text)):
            issues.append("not evidence-backed: no generated figure or table "
                          "(STANDARDS section 5.1)")
        if not any(h in text for h in CAVEAT):
            issues.append(f"no limitations section (one of {CAVEAT})")
        if "**Depends on:**" not in text:
            issues.append("missing a 'Depends on:' line (STANDARDS section 10.3)")

        # The rendered draft must exist.
        draft = DRAFTS / f"CHAPTER-{src.stem.removeprefix('ch')}-DRAFT.md"
        if not draft.exists():
            issues.append(f"no rendered draft at {draft.name}")
        elif "{{" in draft.read_text():
            issues.append("rendered draft still contains unresolved placeholders")

        status = "OK" if not issues else f"{len(issues)} problem(s)"
        print(f"  {src.name:10} {status}")
        problems += [f"{src.name}: {i}" for i in issues]

    print("\ncross-chapter quantities")
    problems += cross_chapter_consistency()

    # Anything generated but never shown to a reader is dead weight.
    for f in sorted(FIGS.glob("*.svg")):
        if f.stem not in used_figs:
            problems.append(f"figures/{f.name} is generated but no chapter shows it")
    for t in sorted(TABLES.glob("*.md")):
        if t.name not in used_tables:
            problems.append(f"tables/{t.name} is generated but no chapter includes it")

    print()
    if problems:
        print(f"AUDIT FAILED - {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("audit passed: figures, tables, numbering and structure all consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
