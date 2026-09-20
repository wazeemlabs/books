"""Regenerate every figure from results/*.json.

No figure is drawn by hand and none is edited after generation: run
`python3 -m bench.figures` and the book's figures are rebuilt from the
measurements. Each figure's caption line, including provenance, is
written next to it as <name>.caption.txt.

Style rules: one message per figure; axes labelled with units; series
distinguishable in grayscale (line style and marker, not colour alone).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
FIGS = Path("figures")

NAIVE = dict(color="#B7780F", linestyle="-", marker="o", markersize=3.5, linewidth=1.8)
CACHED = dict(color="#0B1F3A", linestyle="--", marker="s", markersize=3.5, linewidth=1.8)


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def caption(d: dict) -> str:
    p = d["provenance"]
    m = d["model"]
    return (f"tinyserve {m['params']:,} params "
            f"({m['config']['n_layers']}L/{m['config']['n_heads']}H/"
            f"d={m['config']['d_model']}, fp32) - "
            f"{p['hardware']['cpu']}, {p['hardware']['cores_available']} vCPU, "
            f"NumPy {p['software']['numpy']} on {p['software']['blas']} - "
            f"median of {d['experiment']['runs']} runs after "
            f"{d['experiment']['warmup']} warmup - "
            f"commit {p['commit']}, measured {p['measured_utc']}")


def fig_per_step(d: dict) -> None:
    ps = d["per_step"]
    n = [t * 1e3 for t in ps["naive_s"]]
    c = [t * 1e3 for t in ps["cached_s"]]
    x = range(len(n))

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    thin = dict(marker="", linewidth=1.4)
    ax.plot(x, n, label="Without a cache (Ch 11)", **(NAIVE | thin))
    ax.plot(x, c, label="With a KV cache (Ch 12)", **(CACHED | thin))
    ax.set_yscale("log")
    ax.set_xlabel("Generated token (position in the output)")
    ax.set_ylabel("Time for that token (ms, log scale)")
    ax.set_title("Without a cache, every token costs more than the last", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.annotate("prefill: the same work for both",
                xy=(0, n[0]), xytext=(22, n[0] * 2.4), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    style(ax)
    save(fig, "ch12-per-step", d,
         alt=("Time per generated token against position. Without a cache the "
              f"cost climbs from {n[1]:.0f} ms to {n[-1]:.0f} ms across "
              f"{len(n)-1} tokens; with a cache it stays flat near "
              f"{sorted(c[1:])[len(c)//2]:.2f} ms. Both spend the same time on "
              "the first token, which is prefill."))


def fig_scaling(d: dict) -> None:
    s = d["sweep"]
    x = [r["n_new"] for r in s]

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    ax.plot(x, [r["naive_s"] for r in s], label="Without a cache", **NAIVE)
    ax.plot(x, [r["cached_s"] for r in s], label="With a KV cache", **CACHED)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(v) for v in x])
    ax.set_xlabel("Tokens generated (prompt fixed at "
                  f"{d['experiment']['prompt_len']})")
    ax.set_ylabel("Total time (s, log scale)")
    ax.set_title("The gap widens with every token generated", loc="left")
    for r in s:
        ax.annotate(f"{r['speedup']:.0f}x", (r["n_new"], r["cached_s"]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", va="bottom", fontsize=7.5)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    save(fig, "ch12-scaling", d,
         alt=("Total generation time against the number of tokens generated, "
              "both axes logarithmic. Without a cache the curve is steeper "
              "than linear; with a cache it is linear. The speedup grows from "
              f"{s[0]['speedup']:.0f}x at {s[0]['n_new']} tokens to "
              f"{s[-1]['speedup']:.0f}x at {s[-1]['n_new']}."))


def save(fig, name: str, d: dict, alt: str) -> None:
    FIGS.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    (FIGS / f"{name}.caption.txt").write_text(
        f"ALT: {alt}\nPROVENANCE: {caption(d)}\n")
    print(f"  {name}.svg / .png")


def main() -> None:
    d = json.loads((RESULTS / "ch12.json").read_text())
    print("regenerating figures from results/ch12.json")
    fig_per_step(d)
    fig_scaling(d)


if __name__ == "__main__":
    main()
