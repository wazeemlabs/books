"""Pull a code listing out of the repository, so none is typed by hand.

STANDARDS.md section 6.2: every listing in the text comes from a file
that runs. A listing that is retyped into the manuscript goes stale the
first time the code is refactored, silently, and the chapter then
describes code that no longer exists -- which happened once in this
book already, to Chapter 12's `append`.

    extract("tinyserve/prefix.py", "PrefixTree.match")

returns exactly the lines that define it, dedented, from the file on
disk at render time.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(".")


def _find(tree: ast.Module, symbol: str) -> ast.AST:
    parts = symbol.split(".")
    node: ast.AST = tree
    for part in parts:
        children = getattr(node, "body", [])
        match = [c for c in children
                 if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 and c.name == part]
        if not match:
            raise KeyError(f"{symbol!r}: no definition of {part!r}")
        node = match[0]
    return node


def extract(path: str, symbol: str, drop_docstring: bool = False) -> str:
    """The source of `symbol` in `path`, exactly as it is on disk."""
    file = ROOT / path
    source = file.read_text()
    node = _find(ast.parse(source), symbol)
    lines = source.splitlines()[node.lineno - 1 : node.end_lineno]

    if drop_docstring:
        doc = ast.get_docstring(node, clean=False)
        if doc is not None:
            first = node.body[0]
            start = first.lineno - node.lineno
            end = first.end_lineno - node.lineno + 1
            del lines[start:end]
            while start < len(lines) and not lines[start].strip():
                del lines[start]

    indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[indent:] if l.strip() else "" for l in lines)


def test_extract_is_exact() -> None:
    code = extract("tinyserve/prefix.py", "block_hash")
    assert code.startswith("def block_hash(parent: int, tokens: tuple[int, ...]) -> int:")
    assert code.rstrip().endswith("return hash((parent, tokens))")
    assert "\n" in extract("tinyserve/prefix.py", "PrefixTree.match")
    assert '"""' not in extract("tinyserve/prefix.py", "PrefixTree.match",
                                drop_docstring=True)
