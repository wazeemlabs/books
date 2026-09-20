"""Catch a figure whose labels collide, before a reader has to.

Every figure in this book is generated, which means nobody looks at it
unless they choose to. A label placed at a data point that later moves
ends up on top of a curve, or on top of another label, and the figure
stays that way through every rebuild: the pipeline has no opinion about
whether a chart is readable, only about whether it is current.

This gives it one. After a figure is drawn and before it is written
out, every piece of text on it is checked against every other piece of
text, and against every line that was plotted. Anything that overlaps
is reported by name, and `make audit` fails on it.

Two things it deliberately does not check: whether a label points at
the right thing, and whether the figure says anything worth saying.
Those still need eyes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.text import Annotation, Text
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

# Points squared. Two boxes that share a hairline are not a collision;
# a label sitting a few points into a curve is.
MIN_TEXT_OVERLAP = 6.0
MIN_LINE_OVERLAP = 2.5          # points of a curve inside a text box
# Tick labels sit shoulder to shoulder by design, and an axis label
# under its own ticks is not a collision either.
STRUCTURAL = {"tick", "axis-label"}


def _shrink(box: Bbox, pad: float = 0.75) -> Bbox:
    """Take a hair off every side, so touching is not overlapping."""
    return Bbox.from_extents(box.x0 + pad, box.y0 + pad,
                             box.x1 - pad, box.y1 - pad)


def _texts(fig, renderer) -> list[tuple[str, str, Bbox, object]]:
    """Every visible piece of text on the figure, with what kind it is."""
    out = []
    for ax in fig.axes:
        kinds: list[tuple[object, str]] = []
        kinds += [(t, "tick") for t in ax.get_xticklabels()]
        kinds += [(t, "tick") for t in ax.get_yticklabels()]
        kinds += [(ax.xaxis.label, "axis-label"), (ax.yaxis.label, "axis-label")]
        kinds += [(ax.title, "title")]
        for child in ax.get_children():
            if isinstance(child, (Text, Annotation)) and child not in (
                    ax.title, ax.xaxis.label, ax.yaxis.label):
                kinds.append((child, "label"))
        legend = ax.get_legend()
        if legend is not None:
            kinds += [(t, "legend") for t in legend.get_texts()]
        for artist, kind in kinds:
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            try:
                box = artist.get_window_extent(renderer)
            except Exception:                       # unplaceable; skip
                continue
            if box.width <= 0 or box.height <= 0:
                continue
            out.append((artist.get_text().strip().replace("\n", " "),
                        kind, box, ax))
    return out


def _curve_points(ax, renderer) -> list[tuple[Line2D, np.ndarray]]:
    """Each plotted line, resampled densely in display coordinates.

    Vertices alone are not enough: a label can sit in the middle of a
    long straight segment without being near either of its endpoints.
    """
    out = []
    for line in ax.get_lines():
        if not line.get_visible():
            continue
        data = line.get_xydata()
        if len(data) < 2:
            continue
        pts = line.get_transform().transform(data)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 2:
            continue
        # Resample each segment at roughly one point per 2 display units.
        dense = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.hypot(*(b - a)) / 2))
            dense.append(np.linspace(a, b, n)[1:])
        out.append((line, np.vstack([np.atleast_2d(d) for d in dense])))
    return out


def check(fig, name: str) -> list[str]:
    """Report every overlap on this figure, in words a person can act on."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems: list[str] = []

    texts = _texts(fig, renderer)
    for i, (t1, k1, b1, ax1) in enumerate(texts):
        for t2, k2, b2, ax2 in texts[i + 1:]:
            if k1 in STRUCTURAL and k2 in STRUCTURAL:
                continue
            hit = Bbox.intersection(_shrink(b1), _shrink(b2))
            if hit is not None and hit.width > 0 and hit.height > 0 \
                    and hit.width * hit.height > MIN_TEXT_OVERLAP:
                problems.append(
                    f"{name}: text {t1!r} ({k1}) overlaps text {t2!r} ({k2})")

    for ax in fig.axes:
        curves = _curve_points(ax, renderer)
        for text, kind, box, owner in texts:
            if owner is not ax or kind in STRUCTURAL:
                continue
            inner = _shrink(box, 1.25)
            if inner.width <= 0 or inner.height <= 0:
                continue
            for line, pts in curves:
                inside = ((pts[:, 0] >= inner.x0) & (pts[:, 0] <= inner.x1)
                          & (pts[:, 1] >= inner.y0) & (pts[:, 1] <= inner.y1))
                # Arrow leaders are meant to touch what they point at.
                if inside.sum() > MIN_LINE_OVERLAP:
                    label = line.get_label()
                    which = (f" ({label})" if label
                             and not label.startswith("_") else "")
                    problems.append(
                        f"{name}: text {text!r} sits on a plotted line{which}")
                    break
    return problems


class Report:
    """Collected across one `python3 -m bench.figures` run."""

    PATH = Path("figures/legibility.json")

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.figures = 0

    def add(self, fig, name: str) -> None:
        self.figures += 1
        self.problems += check(fig, name)

    def write(self) -> None:
        self.PATH.parent.mkdir(exist_ok=True)
        self.PATH.write_text(json.dumps(
            {"figures": self.figures, "problems": self.problems}, indent=1) + "\n")


REPORT = Report()


# --- placing labels so they do not land on things -----------------------

# Tried in order, in points: straight out to the side first, then the
# diagonals, then further out. The first offset that hits nothing wins.
CANDIDATES = [(8, -3), (8, 7), (8, -13), (-8, -3), (-8, 7), (-8, -13),
              (0, 10), (0, -16), (14, 12), (14, -18), (-14, 12), (-14, -18),
              (22, 2), (-22, 2), (0, 20), (0, -26), (30, 14), (-30, 14)]


def _align(dx: float, dy: float) -> tuple[str, str]:
    ha = "left" if dx > 0 else "right" if dx < 0 else "center"
    va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
    return ha, va


def _box_at(anchor, dx, dy, w, h) -> Bbox:
    ha, va = _align(dx, dy)
    x = anchor[0] + dx
    x0 = x if ha == "left" else x - w if ha == "right" else x - w / 2
    y = anchor[1] + dy
    y0 = y if va == "bottom" else y - h if va == "top" else y - h / 2
    return Bbox.from_extents(x0, y0, x0 + w, y0 + h)


def label_points(ax, items, *, fontsize=6.8, color=None, candidates=None,
                 weight=None):
    """Annotate points, each at the first offset that collides with nothing.

    `items` is a sequence of `(x, y, text)` in data coordinates. Every
    label is tried against the curves already drawn on the axes, the
    labels already placed, and the edge of the panel, and takes the
    first offset in `candidates` that is clear of all three. A fixed
    offset is what puts two labels on top of each other the moment two
    points come close, which on a log axis they always eventually do.
    """
    fig = ax.figure
    cands = candidates or CANDIDATES
    # Every label is built with a leader; the ones that end up close to
    # their own point have it hidden again below. Matplotlib can only
    # attach an arrow at construction time.
    notes = [ax.annotate(text, (x, y), textcoords="offset points",
                         xytext=(0, 0), fontsize=fontsize,
                         color=color if color is not None else "black",
                         fontweight=weight or "normal",
                         arrowprops=dict(arrowstyle="-", linewidth=0.6,
                                         color="#8A8A8A", shrinkA=2,
                                         shrinkB=3))
             for x, y, text in items]
    for note in notes:
        note.arrow_patch.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    obstacles = [pts for _, pts in _curve_points(ax, renderer)]
    panel = ax.get_window_extent(renderer)
    # Text already on the panel -- a callout, a legend, a note on a
    # reference line -- is as much an obstacle as a curve is.
    placed: list[Bbox] = []
    mine = {n.get_text().strip().replace("\n", " ") for n in notes}
    for text, kind, box, owner in _texts(fig, renderer):
        if owner is ax and kind not in STRUCTURAL and text not in mine:
            if Bbox.intersection(box, panel) is not None:
                placed.append(box)

    for note, (x, y, _) in zip(notes, items):
        extent = note.get_window_extent(renderer)
        w, h = extent.width, extent.height
        anchor = ax.transData.transform((x, y))
        best, best_cost = cands[0], None
        for dx, dy in cands:
            box = _box_at(anchor, dx, dy, w, h)
            grown = Bbox.from_extents(box.x0 - 1.5, box.y0 - 1.5,
                                      box.x1 + 1.5, box.y1 + 1.5)
            cost = 0.0
            for pts in obstacles:
                cost += 6.0 * ((pts[:, 0] >= grown.x0) & (pts[:, 0] <= grown.x1)
                               & (pts[:, 1] >= grown.y0)
                               & (pts[:, 1] <= grown.y1)).sum()
            for other in placed:
                hit = Bbox.intersection(grown, other)
                if hit is not None and hit.width > 0 and hit.height > 0:
                    cost += hit.width * hit.height
            # Leaving the panel is worse than anything inside it.
            spill = (max(0.0, panel.x0 - box.x0) + max(0.0, box.x1 - panel.x1)
                     + max(0.0, panel.y0 - box.y0) + max(0.0, box.y1 - panel.y1))
            cost += spill * 40
            if cost == 0:
                best, best_cost = (dx, dy), 0.0
                break
            if best_cost is None or cost < best_cost:
                best, best_cost = (dx, dy), cost
        ha, va = _align(*best)
        note.set_position(best)
        note.set_horizontalalignment(ha)
        note.set_verticalalignment(va)
        # Labels to the right of their point, at the default offset, read
        # as attached without help. Anything else -- pushed left, pushed
        # far up or down -- gets a leader, because two crowded points
        # can otherwise appear to have swapped labels.
        box = _box_at(anchor, best[0], best[1], w, h)
        gap = max(box.x0 - anchor[0], anchor[0] - box.x1,
                  box.y0 - anchor[1], anchor[1] - box.y1, 0.0)
        # Long enough to read as a leader rather than as a stray tick.
        if gap > 7 and (best[0] <= 0 or abs(best[1]) > 10):
            note.arrow_patch.set_visible(True)
        placed.append(box)
    return notes
