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
    if d.get("model_not_measurement"):  # a cost model, not a benchmark
        a = d["assumptions"]
        return (f"MODEL, not a measurement: arithmetic over published specs - "
                f"{a['gpu']}, {a['hbm_bytes_per_s']/1e12:.2f} TB/s, "
                f"${a['gpu_usd_per_hour']}/GPU-hour, {a['params']/1e9:.0f}B params bf16, "
                f"{a['seq_len']:,}-token sequences - facts verified "
                f"{a['facts_verified']} (FACTS.md) - commit {p['commit']}")
    m = d.get("model")
    if m is None:  # an accounting chapter: no model was timed
        e = d["experiment"]
        return (f"{e['n_requests']:,} sampled requests, seed {e['seed']} - "
                f"case-study traffic (STANDARDS.md section 7) - "
                f"commit {p['commit']}, computed {p['measured_utc']}")
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


def fig_cost(d: dict) -> None:
    rows = d["batches"]
    x = [r["batch"] for r in rows]
    y = [r["usd_per_m_tokens"] for r in rows]
    api = d["api_reference"]["output_usd_per_m"]

    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=200)
    ax.plot(x, y, color="#0B1F3A", marker="o", markersize=4, linewidth=1.8,
            label="Modelled cost, one H100, 8B model")
    ax.axhline(api, color="#B7780F", linestyle=":", linewidth=1.6,
               label=f"Published API price for the same model (${api:.2f}/M)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(v) for v in x], fontsize=7.5)
    ax.set_xlabel("Sequences decoded at the same time")
    ax.set_ylabel("USD per million output tokens (log)")
    ax.set_title("The same GPU and the same model, "
                 f"{d['spread']['ratio']:.0f}x apart in cost", loc="left")
    for r in (rows[0], rows[-1]):
        ax.annotate(f"${r['usd_per_m_tokens']:.3f}\n{r['flop_utilization']*100:.1f}% of peak FLOPs",
                    (r["batch"], r["usd_per_m_tokens"]), textcoords="offset points",
                    xytext=(10, 6), fontsize=7.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)
    save(fig, "ch01-cost", d,
         alt=("Modelled cost per million output tokens against how many sequences "
              f"are decoded together, both axes logarithmic. Cost falls from "
              f"${rows[0]['usd_per_m_tokens']:.2f} at one sequence to "
              f"${rows[-1]['usd_per_m_tokens']:.3f} at {rows[-1]['batch']}, a factor of "
              f"{d['spread']['ratio']:.0f}, on identical hardware. The published API "
              f"price for the same model sits below even that, at ${api:.2f}."))


def fig_memory(d: dict) -> None:
    rows = d["policies"]
    labels = {"max_model_len": "Reserve the full context\n(8,192 tokens)",
              "prompt_plus_cap": "Reserve prompt + cap\n(prompt + 1,024)",
              "paged_16": "Pages of 16 tokens\n(Chapter 14)",
              "oracle": "Perfect foresight\n(not achievable)"}
    order = ["max_model_len", "prompt_plus_cap", "paged_16", "oracle"]
    rows = sorted(rows, key=lambda r: order.index(r["policy"]))
    y = range(len(rows))
    live = [r["live_tokens_mean"] for r in rows]
    idle = [r["reserved_tokens_mean"] - r["live_tokens_mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=200)
    ax.barh(y, live, color="#0B1F3A", label="Holding live keys and values")
    ax.barh(y, idle, left=live, color="#D9DDE5", edgecolor="#B7780F",
            hatch="//", linewidth=0.6, label="Held, but idle")
    for i, r in enumerate(rows):
        ax.text(r["reserved_tokens_mean"] + 90, i,
                f"{r['utilization']*100:.0f}% used - {r['concurrent_seqs']} concurrent",
                va="center", fontsize=7.5)
    ax.set_yticks(list(y), [labels[r["policy"]] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(r["reserved_tokens_mean"] for r in rows) * 1.42)
    ax.set_xlabel("KV cache held per sequence (tokens), averaged over its lifetime")
    ax.set_title("Most of what a contiguous allocator holds is idle", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "ch13-memory", d,
         alt=("KV cache held per sequence under four allocation policies, split "
              "into the part holding live data and the part held but idle. "
              f"Reserving the full context uses {rows[0]['utilization']*100:.0f}% of what it "
              f"holds and fits {rows[0]['concurrent_seqs']} sequences; 16-token pages "
              f"use {rows[2]['utilization']*100:.0f}% and fit {rows[2]['concurrent_seqs']}, "
              "within a percent of perfect foresight."))


def save(fig, name: str, d: dict, alt: str) -> None:
    FIGS.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    (FIGS / f"{name}.caption.txt").write_text(
        f"ALT: {alt}\nPROVENANCE: {caption(d)}\n")
    print(f"  {name}.svg / .png")


CHAPTERS = {"ch01": [fig_cost], "ch12": [fig_per_step, fig_scaling], "ch13": [fig_memory]}


def main() -> None:
    for chapter, figs in CHAPTERS.items():
        path = RESULTS / f"{chapter}.json"
        if not path.exists():
            continue
        print(f"regenerating figures from {path}")
        d = json.loads(path.read_text())
        for fn in figs:
            fn(d)


if __name__ == "__main__":
    main()
