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

from . import theme as T

RESULTS = Path("results")
FIGS = Path("figures")

NAIVE = dict(color=T.AMBER, linestyle="-", marker="o", markersize=3.5, linewidth=1.8)
CACHED = dict(color=T.BLUE, linestyle="--", marker="s", markersize=3.5, linewidth=1.8)


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
    if "prompt" in d and "shapes" in d:  # a traced forward pass, not a timing run
        m, v = d["model"], d["vocabulary"]
        return (f"one forward pass of tinyserve, {m['params']:,} params "
                f"({m['config']['n_layers']}L/{m['config']['n_heads']}H/"
                f"d={m['config']['d_model']}), {v['size']}-word vocabulary, "
                f"untrained (seed 0) - prompt {d['prompt']['words']} - "
                f"commit {p['commit']}, traced {p['measured_utc']}")
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
    ax.plot(x, y, color=T.BLUE, marker="o", markersize=4, linewidth=1.8,
            label="Modelled cost, one H100, 8B model")
    ax.axhline(api, color=T.AMBER, linestyle=":", linewidth=1.6,
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
    ax.barh(y, live, color=T.BLUE, label="Holding live keys and values")
    ax.barh(y, idle, left=live, color="#E7ECF5", edgecolor=T.AMBER,
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


# --- Chapter 2: what the model does, drawn from a real forward pass -----

def fig_pipeline(d: dict) -> None:
    """A diagram, not a chart: the path one prompt takes through the model."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    prompt = " ".join(d["prompt"]["words"])
    toks = d["prompt"]["tokens"]
    cfg, shapes = d["model"]["config"], d["shapes"]
    chosen = d["next_token"]["chosen"]

    fig, ax = plt.subplots(figsize=(7.0, 6.6), dpi=200)
    ax.set_xlim(-6, 100); ax.set_ylim(0, 78); ax.axis("off")

    X, W, H = 14, 32, 8

    def box(bottom, title, sub, *, accent=False):
        ax.add_patch(FancyBboxPatch(
            (X, bottom), W, H, boxstyle="round,pad=0.5,rounding_size=1.0",
            linewidth=1.1, edgecolor=T.AMBER if accent else T.RULE,
            facecolor="#FDF6EA" if accent else "#FFFFFF"))
        ax.text(X + W / 2, bottom + H * 0.63, title, ha="center", va="center",
                fontsize=8.2, color=T.INK, fontweight="bold")
        ax.text(X + W / 2, bottom + H * 0.26, sub, ha="center", va="center",
                fontsize=7.0, color=T.MUTED)

    def down(from_bottom):
        ax.add_patch(FancyArrowPatch(
            (X + W / 2, from_bottom - 0.4), (X + W / 2, from_bottom - 2.6),
            arrowstyle="-|>", mutation_scale=9, linewidth=1.1, color=T.MUTED))

    stages = [
        (62, f'"{prompt}"', "what the user typed", False),
        (51, str(toks), f"split into {len(toks)} token numbers", False),
        (40, f"{shapes['embedding'][0]} x {cfg['d_model']} numbers",
         "one vector looked up per token", False),
        (29, f"{cfg['n_layers']} layers", "attention, then feed-forward", True),
        (18, f"{cfg['vocab_size']} scores", "one for every word it knows", False),
        (7, f'"{chosen}"', "the one token it writes", False),
    ]
    for bottom, title, sub, accent in stages:
        box(bottom, title, sub, accent=accent)
        if bottom > 7:
            down(bottom)

    # The loop, down the left-hand side.
    ax.add_patch(FancyArrowPatch(
        (X - 0.6, 11), (X - 0.6, 66), arrowstyle="-|>", mutation_scale=9,
        linewidth=1.2, color=T.AMBER, linestyle="--",
        connectionstyle="arc3,rad=-0.42"))
    ax.text(-3.2, 38, "append it to the prompt,\nthen run the whole thing again",
            ha="center", va="center", fontsize=7.0, color=T.AMBER,
            style="italic", rotation=90)

    # What one layer contains, beside the layer box.
    cx, cy, cw, ch = 52, 25, 46, 16
    ax.add_patch(FancyBboxPatch(
        (cx, cy), cw, ch, boxstyle="round,pad=0.5,rounding_size=1.0",
        linewidth=1.0, edgecolor=T.RULE, facecolor="#FFFFFF"))
    ax.plot([X + W, cx], [33, 33], linewidth=0.8, color=T.RULE, linestyle=":")
    ax.text(cx + 2, cy + ch - 3.0, "inside one layer", fontsize=7.8,
            color=T.INK, fontweight="bold", va="center")
    ax.text(cx + 2, cy + ch - 7.4, "attention", fontsize=7.4,
            color=T.BLUE, va="center", fontweight="bold")
    ax.text(cx + 16.5, cy + ch - 7.4, "each token looks at the", fontsize=7.2,
            color=T.INK, va="center")
    ax.text(cx + 16.5, cy + ch - 10.2, "tokens before it", fontsize=7.2,
            color=T.INK, va="center")
    ax.text(cx + 2, cy + ch - 13.4, "feed-forward", fontsize=7.4,
            color=T.BLUE, va="center", fontweight="bold")
    ax.text(cx + 16.5, cy + ch - 13.4, "then it thinks alone", fontsize=7.2,
            color=T.INK, va="center")

    ax.text(-6, 76.5, "One token in, one token out", fontsize=12,
            color=T.INK, fontweight="bold", va="top")
    ax.text(-6, 72.4, "Every box runs again for every single token the model writes.",
            fontsize=7.8, color=T.MUTED, va="top")

    save(fig, "ch02-pipeline", d,
         alt=("A vertical flow diagram. The typed prompt is split into token "
              "numbers; each token becomes a vector; the vectors pass through "
              f"the model's {cfg['n_layers']} layers; the model produces a score for "
              f"every one of {cfg['vocab_size']} words it knows; one token is written. An "
              "arrow loops from the written token back to the prompt, because "
              "the whole path runs again for the next token. A panel beside "
              "the layer box says that inside a layer, attention lets each "
              "token look at the tokens before it and the feed-forward step "
              "then processes each token alone."))


def fig_attention(d: dict) -> None:
    """The causal triangle, from real attention weights."""
    import numpy as np

    words = d["prompt"]["words"]
    w = np.array(d["layers"][0]["attention_head_1"])
    masked = np.ma.masked_where(np.triu(np.ones_like(w), 1) > 0, w)

    fig, ax = plt.subplots(figsize=(6.0, 4.1), dpi=200)
    cmap = T.SEQUENTIAL.copy(); cmap.set_bad(T.SURFACE)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)

    for i in range(len(words)):
        for j in range(i + 1):
            ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.6,
                    color="#FFFFFF" if w[i, j] > 0.55 else T.INK)
        for j in range(i + 1, len(words)):
            ax.text(j, i, "-", ha="center", va="center",
                    fontsize=8, color=T.RULE)

    labels = [f"{k + 1}. {x}" for k, x in enumerate(words)]
    ax.set_xticks(range(len(words)), labels, fontsize=7.6, rotation=30, ha="right")
    ax.set_yticks(range(len(words)), labels, fontsize=7.6)
    ax.set_xlabel("...pays this much attention to this earlier token")
    ax.set_ylabel("When processing this token...")
    ax.set_title("A token can only look backwards", loc="left", fontsize=11)
    ax.tick_params(colors=T.MUTED, length=0)
    ax.xaxis.label.set_color(T.INK); ax.yaxis.label.set_color(T.INK)
    ax.title.set_color(T.INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    cb.set_label("share of attention", fontsize=7.6, color=T.MUTED)
    cb.ax.tick_params(labelsize=7, colors=T.MUTED)
    cb.outline.set_visible(False)

    save(fig, "ch02-attention", d,
         alt=("A five-by-five grid of attention weights for the prompt "
              f"'{' '.join(words)}'. Each row shows how one token divides its "
              "attention across the tokens at or before it; every row sums to "
              "one. The whole upper triangle is empty, because no token may "
              "look at a token that comes after it. This model is untrained, "
              "so which earlier token is favoured is arbitrary; the empty "
              "triangle is not."))


def fig_scores(d: dict) -> None:
    """One series: what the model thinks comes next."""
    rows = d["next_token"]["top_k"][::-1]
    labels = [r["word"] for r in rows]
    probs = [r["probability"] * 100 for r in rows]
    uniform = 100 / d["vocabulary"]["size"]

    fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=200)
    ax.barh(range(len(rows)), probs, color=T.BLUE, height=0.68)
    ax.axvline(uniform, color=T.AMBER, linestyle=":", linewidth=1.5)
    ax.text(uniform + 0.15, -0.9,
            f"a coin-toss guess would be {uniform:.1f}%",
            fontsize=7, color=T.AMBER, va="center")
    for i, p in enumerate(probs):
        ax.text(p + 0.12, i, f"{p:.1f}%", va="center", fontsize=7.4, color=T.MUTED)
    ax.set_yticks(range(len(rows)), labels, fontsize=8)
    ax.set_xlabel("chance the model gives this word of coming next (%)")
    ax.set_title(f'The ten best of {d["vocabulary"]["size"]} possible next words',
                 loc="left", fontsize=11)
    ax.set_xlim(0, max(probs) * 1.22)
    T.style(ax, hide_left=True)

    save(fig, "ch02-scores", d,
         alt=(f"Horizontal bars showing the ten highest-scoring next words out "
              f"of {d['vocabulary']['size']}, led by '{rows[-1]['word']}' at "
              f"{probs[-1]:.1f}%. A dotted line marks {uniform:.1f}%, the score every "
              "word would get from pure chance. The model is untrained, so the "
              "ranking carries no meaning; the shape of the output does."))


CHAPTERS = {"ch01": [fig_cost], "ch02": [fig_pipeline, fig_attention, fig_scores], "ch12": [fig_per_step, fig_scaling], "ch13": [fig_memory]}


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
