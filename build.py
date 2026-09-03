#!/usr/bin/env python3
"""Assemble books.wazeem.com, the companion site for every book.

One GitHub Pages repo, one DNS record, one folder per book. Each book
keeps its companion page's source in its own repo; this script gathers
them:

    python3 build.py            # rebuild every book's folder and the index

To add a book, append an entry to BOOKS and give it a `build` callable that
writes the book's folder. Then commit and push this repo.
"""

from __future__ import annotations

import html
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SITE = Path(__file__).resolve().parent
BOOKS_ROOT = SITE.parent
DOMAIN = "books.wazeem.com"


@dataclass(frozen=True)
class Book:
    slug: str
    title: str
    subtitle: str
    blurb: str
    build: Callable[[Path], None]


def build_claude_code(out: Path) -> None:
    """The Claude Code book renders its own pages (changes, errata,
    coverage matrix, reference cards) with the book's print stylesheet."""
    book = BOOKS_ROOT / "claude-code-book"
    subprocess.run([str(book / ".venv" / "bin" / "python"),
                    str(book / "tools" / "build_site.py")],
                   check=True, cwd=book, env={**os.environ, "SITE_OUT": str(out)})


def build_llm(out: Path) -> None:
    """The LLM book's hub page lives in the course repo (its tools rewrite
    it on renumbering). Copy it here with the course's relative links made
    absolute, since the labs and stylesheet stay on llm-course.wazeem.com."""
    src = BOOKS_ROOT / "llm-course" / "llm-course" / "book.html"
    course = "https://llm-course.wazeem.com/"
    page = src.read_text(encoding="utf-8")

    def absolute(m: re.Match) -> str:
        attr, target = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|#)", target):
            return m.group(0)
        if target == "./":
            target = ""
        return f'{attr}="{course}{target}"'

    page = re.sub(r'\b(href|src)="([^"]*)"', absolute, page)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")


BOOKS = [
    Book(
        slug="llm",
        title="Large Language Models from the Ground Up",
        subtitle="Understand How ChatGPT Works, Then Build Your Own, "
                 "Step by Step",
        blurb="Companion page: the runnable notebooks in Colab, the "
              "interactive labs, the instructor materials, and the errata.",
        build=build_llm,
    ),
    Book(
        slug="claude-code",
        title="Claude Code from the Ground Up",
        subtitle="Agentic Coding from Your First Session to Autonomous "
                 "Engineering Teams",
        blurb="What changed in Claude Code since print, the reference "
              "cards as free PDFs, the coverage matrix, the companion "
              "code, and the errata.",
        build=build_claude_code,
    ),
]

INDEX_STYLE = """
:root { --navy: #0B1F3A; --amber: #B7780F; --paper: #F6F4EE;
        --rule: #D9DDE5; --muted: #5B6B82; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--navy);
       font: 17px/1.55 Georgia, "Times New Roman", serif; }
header { background: var(--navy); color: #fff; padding: 2.4rem 1.25rem; }
.wrap { max-width: 42rem; margin: 0 auto; padding: 0 1.25rem; }
header h1 { margin: 0 0 .3rem; font-size: 1.8rem; }
header p { margin: 0; color: #C9D3E3; }
main { padding: 1.5rem 0 3rem; }
.book { border-bottom: 1px solid var(--rule); padding: 1.4rem 0; }
.book h2 { margin: 0 0 .2rem; font-size: 1.3rem; }
.book h2 a { color: var(--navy); text-decoration: none;
             border-bottom: 2px solid var(--amber); }
.book .sub { margin: 0 0 .5rem; color: var(--muted); font-style: italic; }
.book p { margin: 0; }
footer { text-align: center; color: var(--muted); font-size: .85rem;
         padding: 2rem 1rem; }
"""


def index_html() -> str:
    cards = "".join(
        f'<section class="book"><h2><a href="{b.slug}/">'
        f'{html.escape(b.title)}</a></h2>'
        f'<p class="sub">{html.escape(b.subtitle)}</p>'
        f'<p>{html.escape(b.blurb)}</p></section>\n'
        for b in BOOKS)
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "<title>Books by Waseem Khan</title>\n"
        f"<style>{INDEX_STYLE}</style>\n</head><body>\n"
        "<header><div class=\"wrap\"><h1>Books by Waseem Khan</h1>"
        "<p>Companion pages: code, labs, corrections, and what changed "
        "since print.</p></div></header>\n"
        f"<main><div class=\"wrap\">\n{cards}</div></main>\n"
        "<footer>Waseem Khan. Reach me at ceo@wazeem.com.</footer>\n"
        "</body></html>\n")


def main() -> int:
    (SITE / "CNAME").write_text(DOMAIN + "\n", encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    (SITE / "index.html").write_text(index_html(), encoding="utf-8")
    for book in BOOKS:
        print(f"building {book.slug}/ ...")
        book.build(SITE / book.slug)
    print(f"built {DOMAIN}: index plus {len(BOOKS)} books")
    return 0


if __name__ == "__main__":
    sys.exit(main())
