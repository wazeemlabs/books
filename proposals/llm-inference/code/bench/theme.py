"""The book's figure theme: one validated palette, in one place.

Colours are not chosen by eye. The categorical pair below was checked
with the visualization skill's validator
(`validate_palette.js "#2563EB,#B7780F" --mode light`) and passes all
five checks: lightness band, chroma floor, CVD separation
(dE 31.8 protan / 23.0 tritan), normal-vision separation (dE 35.2), and
contrast against the surface.

The book's brand navy (#0B1F3A) is deliberately NOT a series colour: the
validator fails it as categorical (lightness 0.24, chroma 0.06 - it
reads as near-black grey). It is ink. Text, axes and labels wear ink;
a coloured mark beside them carries identity.

Rules applied throughout, from the same source:
  * one axis, never two y-scales;
  * categorical hues in fixed order, never cycled;
  * sequential = one hue, light to dark, monotonic in lightness;
  * a legend whenever there are two or more series, none for one;
  * series distinguishable without colour (line style, hatch, labels).
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# Ink and surface: text tokens, never used for data marks.
INK = "#0B1F3A"
MUTED = "#5B6B82"
RULE = "#D9DDE5"
SURFACE = "#FCFCFB"

# Categorical, in fixed order. Validated as a pair; never cycled.
CATEGORICAL = ["#2563EB", "#B7780F"]
BLUE, AMBER = CATEGORICAL

# Sequential, one hue, light to dark. Lightness is monotonic; checked by
# test_theme_ramp_is_monotonic() below.
SEQUENTIAL_STEPS = [
    "#F2F6FE", "#DCE7FC", "#BBCFF8", "#93B2F2",
    "#6B92EA", "#4A78E0", "#2563EB", "#1A46A8", "#112E6E",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("book_blues", SEQUENTIAL_STEPS)

# Mark specs.
LINE_WIDTH = 1.8
MARKER_SIZE = 4.0


def srgb_to_oklab_l(hex_colour: str) -> float:
    """Perceptual lightness, so the sequential ramp can be checked rather
    than assumed. Formulae from Bjorn Ottosson's OKLab."""
    h = hex_colour.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    r, g, b = lin
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_


def test_theme_ramp_is_monotonic() -> None:
    """The sequential ramp must fall in lightness at every step, or it
    encodes magnitude ambiguously."""
    ls = [srgb_to_oklab_l(c) for c in SEQUENTIAL_STEPS]
    bad = [(SEQUENTIAL_STEPS[i], round(ls[i], 3), round(ls[i + 1], 3))
           for i in range(len(ls) - 1) if ls[i + 1] >= ls[i]]
    assert not bad, f"sequential ramp is not monotonic in lightness: {bad}"


def style(ax, *, hide_left: bool = False) -> None:
    """Recessive grid and axes; ink for text."""
    ax.spines[["top", "right"]].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_color(RULE)
    ax.grid(True, alpha=0.25, linewidth=0.6, color=MUTED)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
