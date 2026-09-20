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
    if "measured" in d and "reference_8b" in d:  # Chapter 3: timings + arithmetic
        m, e = d["model"], d["experiment"]
        return (f"measured on tinyserve {m['params']:,} params "
                f"({m['config']['n_layers']}L/{m['config']['n_heads']}H/"
                f"d={m['config']['d_model']}, fp32) - {p['hardware']['cpu']}, "
                f"{p['hardware']['cores_available']} vCPU, NumPy "
                f"{p['software']['numpy']} - median of {e['runs']} runs after "
                f"{e['warmup']} warmup - 8B figures are arithmetic over "
                f"published specs, not measurements - commit {p['commit']}, "
                f"{p['measured_utc']}")
    if "cliff" in d and "wall" in d:  # Chapter 4: this machine's own limits
        mc = d["machine"]
        return (f"measured single-threaded on {p['hardware']['cpu']} "
                f"({p['hardware']['cores_available']} vCPU), NumPy "
                f"{p['software']['numpy']} on {p['software']['blas']} - "
                f"memory {mc['dram_bytes_per_s'] / 1e9:.1f} GB/s, arithmetic "
                f"{mc['flops'] / 1e9:.0f} GFLOP/s, breaks even at "
                f"{mc['ridge_flop_per_byte']:.0f} FLOP/byte - "
                f"commit {p['commit']}, {p['measured_utc']}")
    if "curve" in d and "budgets" in d:  # Chapter 5: measured tail + modelled curve
        mm = d["measured"]
        return (f"tail measured over {mm['samples']:,} decode steps on tinyserve - "
                f"{p['hardware']['cpu']}, {p['hardware']['cores_available']} vCPU, "
                f"NumPy {p['software']['numpy']} - the throughput curve and "
                f"budgets are arithmetic over published specs, not measurements - "
                f"commit {p['commit']}, {p['measured_utc']}")
    if "scenarios" in d and "api" in d:  # Chapter 6: build or buy
        a = d["assumptions"]
        return (f"arithmetic over the shared serving model, not a measurement - "
                f"batch {a['batch']}, {a['context']:,}-token context, GPU prices "
                f"and the ${d['api']['usd_per_m_output']:.2f}/M API price verified "
                f"September 2026 (FACTS.md) - commit {p['commit']}")
    if "sizes" in d and "scaling" in d:  # Chapter 7: what this machine can do
        dev = d["device"]
        return (f"measured on {dev['name']} ({dev['cores_visible']} cores), "
                f"NumPy {dev['numpy']} on {p['software']['blas']} - median of 5 "
                f"runs - accelerator figures are published specifications, not "
                f"measurements - commit {p['commit']}, {p['measured_utc']}")
    if "points" in d and "machine" in d:  # Chapter 8: the roofline
        mc = d["machine"]
        return (f"measured on {p['hardware']['cpu']}, "
                f"{p['hardware']['cores_available']} vCPU, NumPy "
                f"{p['software']['numpy']} - peak {mc['peak_flops'] / 1e9:,.0f} GFLOP/s, "
                f"bandwidth {mc['bandwidth_bytes_per_s'] / 1e9:.1f} GB/s, ridge "
                f"{mc['ridge_flop_per_byte']:.0f} FLOP/byte - accelerator figures are "
                f"published specifications - commit {p['commit']}, {p['measured_utc']}")
    if "ways" in d and "spread" in d:  # Chapter 9: one measurement, many framings
        e = d["experiment"]
        return (f"one unchanged decode step on tinyserve, {e['samples']} samples - "
                f"{p['hardware']['cpu']}, {p['hardware']['cores_available']} vCPU, "
                f"NumPy {p['software']['numpy']} - every row is the same "
                f"operation reported differently - commit {p['commit']}, "
                f"{p['measured_utc']}")
    if "agreement" in d and "baseline" in d:  # Chapter 10: validation and baseline
        a = d["agreement"]
        return (f"tinyserve checked against PyTorch {a['torch_version']} on the same "
                f"weights - {p['hardware']['cpu']}, {p['hardware']['cores_available']} "
                f"vCPU, NumPy {p['software']['numpy']} - median of 5 runs - "
                f"commit {p['commit']}, {p['measured_utc']}")
    if "waste" in d and "per_step_s" in d:  # Chapter 11: the naive loop
        m, e = d["model"], d["experiment"]
        return (f"tinyserve {m['params']:,} params "
                f"({m['config']['n_layers']}L/{m['config']['n_heads']}H/"
                f"d={m['config']['d_model']}, fp32), {e['prompt']}-token prompt - "
                f"{p['hardware']['cpu']}, {p['hardware']['cores_available']} vCPU - "
                f"arithmetic counted from the shapes, time measured as the median "
                f"of {e['runs']} runs - commit {p['commit']}, {p['measured_utc']}")
    if "unlimited" in d and "prefill" in d:  # Chapter 15: prefix caching
        a, pf, tr = d["assumptions"], d["prefill"], d["trace"]
        return (f"prefill measured on tinyserve, {pf['prompt_tokens']}-token prompts, "
                f"median of {pf['runs']} paired runs after {pf['warmup']} warmup - "
                f"{p['hardware']['cpu']}, {p['hardware']['cores_available']} vCPU, "
                f"NumPy {p['software']['numpy']} - hit rates simulated over "
                f"{tr['requests']} requests from {tr['sessions']} sessions "
                f"(seed {a['seed']}) through the real index and allocator; cache "
                f"sizes converted at the reference model's "
                f"{a['kv_bytes_per_token'] / 1024:.0f} KiB per token - "
                f"commit {p['commit']}, {p['measured_utc']}")
    if "block_sizes" in d and "step_cost" in d:  # Chapter 14: paging
        a = d["assumptions"]
        return (f"step cost measured on tinyserve, {p['hardware']['cpu']}, "
                f"{p['hardware']['cores_available']} vCPU; capacity computed for "
                f"the reference model over {a['requests_sampled']:,} sampled requests "
                f"(seed {a['seed']}) in a {a['pool_gb']:.0f} GB pool - "
                f"commit {p['commit']}, {p['measured_utc']}")
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




# --- Chapter 3: the two phases ------------------------------------------

def fig_timeline(d: dict) -> None:
    """Where a request's time actually goes. A diagram, drawn to scale."""
    r = d["reference_8b"]
    o, pre, dec = r["one_request"], r["prefill"], r["decode"]
    total = o["total_s"]

    fig, ax = plt.subplots(figsize=(7.4, 2.9), dpi=200)
    ax.barh([0], [o["prefill_s"]], color=T.AMBER, height=0.44)
    ax.barh([0], [o["decode_s"]], left=o["prefill_s"], color=T.BLUE, height=0.44)

    # Suggest the one-token-at-a-time texture of decode.
    for i in range(0, o["output_tokens"], 6):
        x = o["prefill_s"] + i * dec["seconds"]
        ax.plot([x, x], [-0.22, 0.22], color="#FFFFFF", linewidth=0.5, alpha=0.55)

    ax.annotate(f"prefill\n{o['prefill_s'] * 1e3:.0f} ms",
                xy=(o["prefill_s"] / 2, 0.30), xytext=(0.02 * total, 0.72),
                fontsize=8, color=T.AMBER, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.8, color=T.AMBER))
    ax.text(o["prefill_s"] + o["decode_s"] / 2, 0,
            f"decode - {o['output_tokens']} tokens, one at a time, "
            f"{dec['seconds'] * 1e3:.2f} ms each",
            ha="center", va="center", fontsize=8.4, color="#FFFFFF",
            fontweight="bold")
    ax.text(total * 0.5, -0.62,
            f"{o['decode_share'] * 100:.0f}% of the time this request takes is "
            "spent writing, one token at a time",
            ha="center", fontsize=8, color=T.MUTED)

    ax.set_xlim(0, total * 1.005); ax.set_ylim(-0.9, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("seconds")
    ax.set_title("Reading the question is the cheap part", loc="left", fontsize=11)
    T.style(ax, hide_left=True)
    ax.grid(False)

    save(fig, "ch03-timeline", d,
         alt=(f"A timeline of one request lasting {total:.2f} seconds. Reading the "
              f"{r['config']['prompt']:,}-token prompt takes {o['prefill_s'] * 1e3:.0f} "
              f"milliseconds, a sliver at the left. Writing {o['output_tokens']} tokens "
              f"takes the remaining {o['decode_s']:.2f} seconds, "
              f"{o['decode_share'] * 100:.0f}% of the total, one token at a time."))


def fig_per_token(d: dict) -> None:
    """Measured: what one token costs in each phase."""
    rows = d["measured"]
    x = [r["prompt"] for r in rows]
    pre = [r["prefill_per_token_s"] * 1e3 for r in rows]
    dec = [r["decode_step_p50_s"] * 1e3 for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=200)
    ax.plot(x, dec, label="writing a token (decode)", color=T.BLUE,
            linestyle="--", marker="s", markersize=4, linewidth=T.LINE_WIDTH)
    ax.plot(x, pre, label="reading a token of prompt (prefill)", color=T.AMBER,
            linestyle="-", marker="o", markersize=4, linewidth=T.LINE_WIDTH)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(x, [f"{v:,}" for v in x])
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("time for one token (ms, log)")
    ax.set_title("A token costs more to write than to read", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    worst = max(rows, key=lambda r: r["decode_vs_prefill_per_token"])
    ax.annotate(f"{worst['decode_vs_prefill_per_token']:.1f}x",
                xy=(worst["prompt"], worst["decode_step_p50_s"] * 1e3),
                textcoords="offset points", xytext=(6, 6), fontsize=8, color=T.MUTED)
    T.style(ax)

    save(fig, "ch03-per-token", d,
         alt=("Time to process one token in each phase against prompt length, "
              "both axes logarithmic. Writing a token costs about three times "
              "more than reading one across every prompt length measured. Both "
              "curves rise with prompt length, because attention has more "
              "earlier tokens to consider."))


def fig_intensity(d: dict) -> None:
    """Why the two phases differ: work done per byte fetched."""
    r = d["reference_8b"]
    ridge = r["hardware"]["ridge_flop_per_byte"]
    dec, pre = r["decode"]["intensity"], r["prefill"]["intensity"]

    fig, ax = plt.subplots(figsize=(7.4, 2.6), dpi=200)
    ax.set_xscale("log")
    lo, hi = 0.2, 4000
    ax.axvspan(lo, ridge, color="#EDF2FD", zorder=0)
    ax.axvspan(ridge, hi, color="#FBF2E4", zorder=0)
    ax.axvline(ridge, color=T.INK, linewidth=1.2, zorder=3)

    ax.plot([dec], [0], marker="s", markersize=9, color=T.BLUE, zorder=4)
    ax.plot([pre], [0], marker="o", markersize=9, color=T.AMBER, zorder=4)
    ax.annotate(f"decode\n{dec:.2f} FLOP/byte", xy=(dec, 0), xytext=(dec, 0.42),
                ha="center", fontsize=8, color=T.BLUE, fontweight="bold")
    ax.annotate(f"prefill\n{pre:,.0f} FLOP/byte", xy=(pre, 0), xytext=(pre, 0.42),
                ha="center", fontsize=8, color=T.AMBER, fontweight="bold")
    ax.text(ridge, -0.55, f"this accelerator breaks even at {ridge:.0f}",
            ha="center", fontsize=7.6, color=T.INK)
    ax.text(lo * 1.4, -0.3, "limited by memory", fontsize=8, color=T.MUTED)
    ax.text(hi * 0.72, -0.3, "limited by arithmetic", fontsize=8,
            color=T.MUTED, ha="right")

    ax.set_xlim(lo, hi); ax.set_ylim(-0.75, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("arithmetic performed per byte fetched (log)")
    ax.set_title("The two phases sit on opposite sides of the machine",
                 loc="left", fontsize=11)
    T.style(ax, hide_left=True)
    ax.grid(False)

    save(fig, "ch03-intensity", d,
         alt=(f"A logarithmic scale of arithmetic performed per byte fetched. "
              f"The accelerator breaks even at {ridge:.0f}. Decode sits far to the "
              f"left at {dec:.2f}, deep in the region limited by memory; prefill sits "
              f"to the right at {pre:,.0f}, in the region limited by arithmetic. They "
              f"are {r['intensity_ratio']:,.0f} times apart."))




# --- Chapter 4: the memory wall, measured on the reader's own machine ---

def fig_cliff(d: dict) -> None:
    """The memory hierarchy, made visible by outgrowing it."""
    x = [c["kib"] for c in d["cliff"]]
    y = [c["bytes_per_s"] / 1e9 for c in d["cliff"]]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    ax.plot(x, y, color=T.BLUE, marker="o", markersize=4.5,
            linewidth=T.LINE_WIDTH)
    ax.set_xscale("log", base=2)

    def human(kib: int) -> str:
        if kib >= 1024 * 1024:
            return f"{kib // (1024 * 1024)} GiB"
        return f"{kib // 1024} MiB" if kib >= 1024 else f"{kib} KiB"
    ax.set_xticks(x, [human(v) for v in x], fontsize=7.5)

    for lvl in d["cache_topology"]:
        if lvl["kib"] < min(x) or lvl["kib"] > max(x):
            continue
        ax.axvline(lvl["kib"], color=T.MUTED, linestyle=":", linewidth=1.0)
        ax.text(lvl["kib"], max(y) * 1.03,
                f"L{lvl['level']} ends\n{lvl['kib']:,} KiB", fontsize=7,
                color=T.MUTED, ha="center", va="bottom")

    fastest, slowest = max(y), y[-1]
    ax.annotate(f"{fastest:.0f} GB/s", xy=(x[y.index(fastest)], fastest),
                textcoords="offset points", xytext=(-6, 8), fontsize=8,
                color=T.INK, ha="right")
    ax.annotate(f"{slowest:.0f} GB/s", xy=(x[-1], slowest),
                textcoords="offset points", xytext=(-8, 10), fontsize=8,
                color=T.INK, ha="right")

    ax.set_ylim(0, max(y) * 1.28)
    ax.set_xlabel("working set (KiB, log)")
    ax.set_ylabel("read bandwidth (GB/s)")
    ax.set_title("The further the data, the slower it arrives", loc="left",
                 fontsize=11)
    T.style(ax)

    save(fig, "ch04-cliff", d,
         alt=(f"Read bandwidth against working-set size, x axis logarithmic. "
              f"Bandwidth peaks near {fastest:.0f} GB/s while the data fits in "
              f"cache and falls to {slowest:.0f} GB/s once the working set is far "
              f"larger, a factor of {fastest / slowest:.1f}. Dotted lines mark where "
              "each cache level ends."))


def fig_wall(d: dict) -> None:
    """Decode time against model size, against what bandwidth alone predicts."""
    w = d["wall"]
    x = [r["weight_mib"] for r in w]
    measured = [r["decode_step_s"] * 1e3 for r in w]
    predicted = [r["predicted_from_bandwidth_s"] * 1e3 for r in w]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    ax.plot(x, measured, label="measured", color=T.BLUE, marker="s",
            markersize=4.5, linewidth=T.LINE_WIDTH)
    ax.plot(x, predicted, label="predicted by bandwidth alone", color=T.AMBER,
            linestyle="--", marker="o", markersize=4.5, linewidth=T.LINE_WIDTH)
    ax.set_xscale("log"); ax.set_yscale("log")

    l3 = next((c for c in d["cache_topology"] if c["level"] == 3), None)
    if l3 and min(x) < l3["kib"] / 1024 < max(x):
        ax.axvline(l3["kib"] / 1024, color=T.MUTED, linestyle=":", linewidth=1.0)
        ax.text(l3["kib"] / 1024, max(measured) * 1.1, " weights outgrow\n last-level cache",
                fontsize=7, color=T.MUTED, va="top")

    first, last = w[0], w[-1]
    ax.annotate(f"{first['measured_over_predicted']:.1f}x above",
                xy=(first["weight_mib"], first["decode_step_s"] * 1e3),
                textcoords="offset points", xytext=(8, -3), fontsize=7.5, color=T.MUTED)
    ax.annotate(f"{last['measured_over_predicted']:.2f}x",
                xy=(last["weight_mib"], last["decode_step_s"] * 1e3),
                textcoords="offset points", xytext=(-4, 10), fontsize=7.5,
                color=T.MUTED, ha="right")

    ax.set_xlabel("model weights (MiB, log)")
    ax.set_ylabel("time to write one token (ms, log)")
    ax.set_title("Grow the model and the time becomes the fetch", loc="left",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    T.style(ax)

    save(fig, "ch04-wall", d,
         alt=("Time to write one token against model size, both axes "
              "logarithmic, with the time that memory bandwidth alone would "
              f"predict. For the smallest model the measurement sits "
              f"{first['measured_over_predicted']:.1f} times above the prediction, "
              "because overhead rather than fetching dominates. As the model "
              f"grows the two converge, reaching {last['measured_over_predicted']:.2f} "
              "times at the largest size: the time to write a token has become "
              "the time to fetch the weights."))




# --- Chapter 5: latency, throughput, and what a budget buys -------------

def fig_tradeoff(d: dict) -> None:
    """Throughput is bought with latency. A budget decides how much."""
    c = d["curve"]
    x = [r["tokens_per_s"] for r in c]
    y = [r["inter_token_ms"] for r in c]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    ax.plot(x, y, color=T.BLUE, marker="o", markersize=4,
            linewidth=T.LINE_WIDTH, zorder=3)
    for r in c:
        if r["batch"] in (1, 16, 64, 174, 325) or r is c[-1]:
            ax.annotate(f"batch {r['batch']}", (r["tokens_per_s"], r["inter_token_ms"]),
                        textcoords="offset points", xytext=(7, -3), fontsize=7.2,
                        color=T.MUTED)

    for b in d["budgets"]:
        ax.axhline(b["itl_budget_ms"], color=T.AMBER, linestyle=":", linewidth=1.1)
        ax.text(min(x) * 1.05, b["itl_budget_ms"],
                f" {b['itl_budget_ms']} ms budget -> batch {b['largest_batch']}",
                fontsize=7.2, color=T.AMBER, va="bottom")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("throughput (tokens per second, log)")
    ax.set_ylabel("what one user waits between words (ms, log)")
    ax.set_title("Throughput is bought with latency", loc="left", fontsize=11)
    last = c[-1]
    ax.annotate("the curve stops here because\nmemory runs out, not latency",
                xy=(last["tokens_per_s"], last["inter_token_ms"]),
                xytext=(last["tokens_per_s"] * 0.34, last["inter_token_ms"] * 2.3),
                fontsize=7.2, color=T.INK,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=T.INK))
    T.style(ax)

    save(fig, "ch05-tradeoff", d,
         alt=("Inter-token latency against throughput as the batch grows, both "
              "axes logarithmic. Serving more sequences at once raises "
              f"throughput from {x[0]:,.0f} to {x[-1]:,.0f} tokens per second and raises "
              f"what each user waits between words from {y[0]:.1f} to {y[-1]:.0f} "
              "milliseconds. Dotted lines mark latency budgets; the loosest "
              "budgets all permit the same batch, because memory, not "
              "latency, is what finally stops the curve."))


def fig_tail(d: dict) -> None:
    """The average is not the experience."""
    m = d["measured"]
    edges, counts = m["histogram_edges"], m["histogram"]
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    width = (edges[1] - edges[0]) * 0.92

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    ax.bar(centres, counts, width=width, color=T.BLUE)
    for label, key, style, h in (("p50", "p50_ms", "-", 0.97),
                                 ("p99", "p99_ms", "--", 0.80),
                                 ("p99.9", "p999_ms", ":", 0.63)):
        ax.axvline(m[key], color=T.AMBER, linestyle=style, linewidth=1.4)
        ax.text(m[key], max(counts) * h, f" {label} {m[key]:.2f} ms",
                fontsize=7.4, color=T.AMBER, va="top")

    ax.set_xlabel("time for one decode step (ms)")
    ax.set_ylabel(f"steps (of {m['samples']:,})")
    ax.set_title("Most steps are quick; the ones users remember are not",
                 loc="left", fontsize=11)
    T.style(ax)

    save(fig, "ch05-tail", d,
         alt=(f"Distribution of {m['samples']:,} decode-step times. The bulk sits near "
              f"the median of {m['p50_ms']:.2f} milliseconds, with a right tail reaching "
              f"{m['max_ms']:.2f}. The 99th percentile is {m['p99_ms']:.2f} milliseconds, "
              f"{m['p99_over_p50']:.1f} times the median, and the 99.9th is "
              f"{m['p999_ms']:.2f}."))




# --- Chapter 6: where self-hosting crosses an API price ----------------

def fig_breakeven(d: dict) -> None:
    api = d["api"]["usd_per_m_output"]
    styles = [("-", "o"), ("--", "s"), ("-.", "^"), (":", "D")]

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=200)
    for i, sc in enumerate(d["scenarios"]):
        x = [c["utilization"] * 100 for c in sc["curve"]]
        y = [c["usd_per_m"] for c in sc["curve"]]
        ls, mk = styles[i % len(styles)]
        ax.plot(x, y, label=f"{sc['precision']}, {sc['pricing']}",
                color=T.CATEGORICAL[i % 2], linestyle=ls, marker=mk,
                markersize=4, linewidth=T.LINE_WIDTH)

    ax.axhline(api, color=T.INK, linewidth=1.6)
    ax.text(99, api * 1.18, f"buy it: ${api:.2f} per million output tokens",
            fontsize=8, color=T.INK, ha="right")

    for sc in d["scenarios"]:
        b = sc["breakeven_utilization"]
        if b <= 1.0:
            ax.plot([b * 100], [api], marker="o", markersize=7,
                    markerfacecolor="#FFFFFF", markeredgecolor=T.INK,
                    markeredgewidth=1.4, zorder=5)
            ax.annotate(f"{b * 100:.0f}%", (b * 100, api),
                        textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=7.6, color=T.INK)

    ax.set_yscale("log")
    ax.set_xlabel("how busy you keep the accelerator (%)")
    ax.set_ylabel("cost per million output tokens (log)")
    ax.set_title("Self-hosting is a bet on utilization", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")
    T.style(ax)

    save(fig, "ch06-breakeven", d,
         alt=("Cost per million output tokens against accelerator utilization, "
              "y axis logarithmic, for four self-hosting configurations "
              "(colour shows the GPU pricing, line style shows the precision), "
              f"against a flat published API price of ${api:.2f}. Each curve falls "
              "as the accelerator is kept busier. Circles mark where a "
              "configuration crosses the API price; the most expensive "
              "configuration never crosses it at all."))




# --- Chapter 7: wide hardware needs wide work ---------------------------

def fig_size(d: dict) -> None:
    rows = d["sizes"]
    x = [r["n"] for r in rows]
    y = [r["share_of_peak"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    ax.plot(x, y, color=T.BLUE, marker="o", markersize=4.5,
            linewidth=T.LINE_WIDTH)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(v) for v in x], fontsize=7.5)
    ax.set_ylim(0, 108)

    small = rows[0]
    ax.annotate(f"a {small['n']}x{small['n']} multiply uses\n"
                f"{small['share_of_peak'] * 100:.0f}% of this machine",
                xy=(small["n"], small["share_of_peak"] * 100),
                xytext=(small["n"] * 1.5, 34), fontsize=7.6, color=T.INK,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=T.INK))
    ax.axhline(100, color=T.MUTED, linestyle=":", linewidth=1.0)
    ax.text(x[-1], 102, "everything this machine has", fontsize=7.4,
            color=T.MUTED, ha="right")

    ax.set_xlabel("size of the matrix multiplied (n x n)")
    ax.set_ylabel("share of this machine's best rate (%)")
    ax.set_title("Small work leaves most of the machine idle", loc="left",
                 fontsize=11)
    T.style(ax)

    save(fig, "ch07-size", d,
         alt=("Share of the machine's best arithmetic rate against the size of "
              f"the matrix multiplied, x axis logarithmic. A {small['n']}-by-"
              f"{small['n']} multiply reaches only {small['share_of_peak'] * 100:.0f}% of "
              "what the machine can do; the rate climbs with size and reaches "
              "100% around 1024. The curve is not perfectly smooth, because "
              "some sizes suit the library's blocking better than others."))


def fig_core_scaling(d: dict) -> None:
    rows = d["scaling"]
    x = [r["threads"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    ax.plot(x, [r["gflops_speedup"] for r in rows], label="arithmetic",
            color=T.BLUE, marker="o", markersize=5, linewidth=T.LINE_WIDTH)
    ax.plot(x, [r["bandwidth_speedup"] for r in rows], label="memory bandwidth",
            color=T.AMBER, linestyle="--", marker="s", markersize=5,
            linewidth=T.LINE_WIDTH)
    ax.axhline(1.0, color=T.MUTED, linestyle=":", linewidth=1.0)

    last = rows[-1]
    ax.annotate(f"{last['gflops_speedup']:.1f}x", (last["threads"], last["gflops_speedup"]),
                textcoords="offset points", xytext=(-4, 8), fontsize=8,
                color=T.BLUE, ha="right")
    ax.annotate(f"{last['bandwidth_speedup']:.2f}x", (last["threads"], last["bandwidth_speedup"]),
                textcoords="offset points", xytext=(-4, -14), fontsize=8,
                color=T.AMBER, ha="right")

    ax.set_xticks(x, [str(v) for v in x])
    ax.set_xlabel("cores allowed to work")
    ax.set_ylabel("speed relative to one core")
    ax.set_title("More cores buy arithmetic, not memory", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    T.style(ax)

    save(fig, "ch07-scaling", d,
         alt=("Speed relative to one core against the number of cores allowed, "
              "for an operation limited by arithmetic and one limited by memory. "
              f"Arithmetic reaches {last['gflops_speedup']:.1f} times faster on "
              f"{last['threads']} cores; memory bandwidth reaches "
              f"{last['bandwidth_speedup']:.2f} times, which is to say it does not "
              "improve at all. Adding processing multiplies arithmetic and "
              "leaves the memory system where it was."))




# --- Chapter 8: the roofline -------------------------------------------

def _roof(ax, peak: float, bw: float, lo: float, hi: float, unit: float,
          label: str) -> None:
    xs = [lo, peak / bw, hi]
    ax.plot(xs, [min(peak, x * bw) / unit for x in xs], color=T.INK,
            linewidth=1.8, zorder=2, label=label)
    ax.axvline(peak / bw, color=T.MUTED, linestyle=":", linewidth=1.0, zorder=1)


def fig_roofline(d: dict) -> None:
    """Measured operations against the bound that should contain them."""
    m, pts = d["machine"], d["points"]
    peak, bw = m["peak_flops"], m["bandwidth_bytes_per_s"]
    lo = min(p["intensity"] for p in pts) / 3
    hi = max(p["intensity"] for p in pts) * 3

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=200)
    _roof(ax, peak, bw, lo, hi, 1e9, "the bound: whichever limit binds first")

    for p in pts:
        colour = T.AMBER if p["kind"] == "model" else T.BLUE
        marker = "s" if p["kind"] == "model" else "o"
        ax.plot([p["intensity"]], [p["achieved_flops"] / 1e9], marker=marker,
                markersize=6, color=colour, zorder=4)
        ax.annotate(p["name"], (p["intensity"], p["achieved_flops"] / 1e9),
                    textcoords="offset points", xytext=(7, -3), fontsize=6.8,
                    color=T.INK)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("arithmetic per byte fetched (log)")
    ax.set_ylabel("rate achieved (GFLOP/s, log)")
    ax.set_title("Every operation sits under the roof", loc="left", fontsize=11)
    ax.text(m["ridge_flop_per_byte"], peak / 1e9 * 1.35,
            f" breaks even at {m['ridge_flop_per_byte']:.0f}", fontsize=7.4,
            color=T.MUTED)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    T.style(ax)

    save(fig, "ch08-roofline", d,
         alt=("Measured rate against arithmetic per byte for eight operations, "
              "both axes logarithmic, under a roofline that rises with "
              "intensity and then flattens at the machine's peak. Every "
              "measured point falls on or below the roof. Operations to the "
              "left are limited by memory, those to the right by arithmetic, "
              f"and the two meet at {m['ridge_flop_per_byte']:.0f} operations per byte."))


def fig_batching_roof(d: dict) -> None:
    """Batching climbs the slope, and still does not reach the top."""
    a = d["accelerator"]
    peak, bw = a["peak_flops"], a["bandwidth_bytes_per_s"]
    rows = a["batching"]
    lo, hi = 0.5, a["ridge_flop_per_byte"] * 4

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=200)
    _roof(ax, peak, bw, lo, hi, 1e12, "the bound")

    ax.plot([r["intensity"] for r in rows], [r["achieved_flops"] / 1e12 for r in rows],
            color=T.BLUE, marker="o", markersize=5, linewidth=T.LINE_WIDTH,
            zorder=4, label="decode, as the batch grows")
    for r in rows:
        if r["batch"] in (1, 32, 325):
            ax.annotate(f"batch {r['batch']}", (r["intensity"], r["achieved_flops"] / 1e12),
                        textcoords="offset points", xytext=(8, -4), fontsize=7.2,
                        color=T.MUTED)

    ceiling = a["intensity_ceiling"]
    ax.axvline(ceiling, color=T.AMBER, linestyle="--", linewidth=1.4)
    ax.text(ceiling * 0.93, peak / 1e12 * 0.5,
            f"however large the batch,\nit stops at {ceiling:.0f}",
            fontsize=7.4, color=T.AMBER, ha="right")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("arithmetic per byte fetched (log)")
    ax.set_ylabel("rate achieved (TFLOP/s, log)")
    ax.set_title("Batching climbs the slope but never tops it", loc="left",
                 fontsize=11)
    ax.text(a["ridge_flop_per_byte"], peak / 1e12 * 1.3,
            f" breaks even at {a['ridge_flop_per_byte']:.0f}", fontsize=7.4,
            color=T.MUTED)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    T.style(ax)

    save(fig, "ch08-batching", d,
         alt=("The accelerator's roofline with decoding placed on it as the "
              f"batch grows from 1 to {rows[-1]['batch']}. Each larger batch moves right "
              "and up along the memory-bound slope, from "
              f"{rows[0]['intensity']:.1f} to {rows[-1]['intensity']:.0f} operations per byte. "
              f"A dashed line marks {ceiling:.0f}, the point beyond which no batch can "
              f"go, which is still short of the break-even point at "
              f"{a['ridge_flop_per_byte']:.0f}: batching alone cannot make this workload "
              "limited by arithmetic."))




# --- Chapter 9: the same number, framed several ways --------------------

def fig_framings(d: dict) -> None:
    rows = sorted(d["ways"], key=lambda w: w["tokens_per_s"])
    honest = d["honest_tokens_per_s"]
    labels = [w["how"] for w in rows]
    values = [w["tokens_per_s"] for w in rows]
    colours = [T.AMBER if abs(v - honest) < 1 else T.BLUE for v in values]

    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=200)
    ax.barh(range(len(rows)), values, color=colours, height=0.66)
    ax.axvline(honest, color=T.INK, linestyle=":", linewidth=1.4)

    for i, w in enumerate(rows):
        ax.text(w["tokens_per_s"] + max(values) * 0.012, i,
                f"{w['tokens_per_s']:,.0f}  ({w['relative_to_honest']:.2f}x)",
                va="center", fontsize=7.4, color=T.MUTED)

    ax.set_yticks(range(len(rows)), labels, fontsize=7.6)
    ax.set_xlim(0, max(values) * 1.32)
    ax.set_xlabel("tokens per second reported")
    ax.set_title("One measurement, reported six defensible ways", loc="left",
                 fontsize=11)
    ax.text(honest, len(rows) - 0.3, " what this book reports", fontsize=7.4,
            color=T.INK)
    T.style(ax, hide_left=True)

    save(fig, "ch09-framings", d,
         alt=("Horizontal bars showing the same unchanged operation reported six "
              f"ways, from {min(values):,.0f} to {max(values):,.0f} tokens per second, "
              f"a spread of {d['spread']['ratio']:.1f} times. The amber bar is the median "
              "at stated conditions, which is what this book reports. Every "
              "other bar is also true."))




# --- Chapter 10: a teaching engine against a real framework -------------

def fig_framework(d: dict) -> None:
    rows = d["speed"]
    x = [r["tokens"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    ax.plot(x, [r["tinyserve_s"] * 1e3 for r in rows], label="tinyserve (NumPy, for reading)",
            color=T.AMBER, marker="o", markersize=4.5, linewidth=T.LINE_WIDTH)
    ax.plot(x, [r["torch_s"] * 1e3 for r in rows], label="PyTorch (for running)",
            color=T.BLUE, linestyle="--", marker="s", markersize=4.5,
            linewidth=T.LINE_WIDTH)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(x, [str(v) for v in x])

    worst = min(rows, key=lambda r: r["torch_over_tinyserve"])
    ax.annotate(f"{1 / worst['torch_over_tinyserve']:.0f}x faster",
                xy=(worst["tokens"], worst["torch_s"] * 1e3),
                textcoords="offset points", xytext=(6, -14), fontsize=7.6,
                color=T.BLUE)

    ax.set_xlabel("prompt processed (tokens)")
    ax.set_ylabel("time (ms, log)")
    ax.set_title("The same arithmetic, several times faster", loc="left",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    T.style(ax)

    save(fig, "ch10-framework", d,
         alt=("Time to process a prompt against its length, both axes "
              "logarithmic, for the book's NumPy engine and a PyTorch "
              "implementation of the same architecture running the same "
              f"weights. PyTorch is up to {1 / worst['torch_over_tinyserve']:.0f} times "
              "faster while computing the same result, because its kernels use "
              "the machine better."))




# --- Chapter 11: work done against work needed --------------------------

def fig_waste(d: dict) -> None:
    """Almost everything the loop computes is thrown away."""
    prompt = d["experiment"]["prompt"]
    steps = range(len(d["per_step_s"]))
    cfg = d["model"]["config"]
    hd = d["model"]["head_dim"]

    def flops(t_new: int, t_total: int) -> int:
        dm = cfg["d_model"]
        per_layer = (2 * t_new * dm * (cfg["n_heads"] * hd)
                     + 2 * 2 * t_new * dm * (cfg["n_kv_heads"] * hd)
                     + 2 * t_new * (cfg["n_heads"] * hd) * dm
                     + 2 * 2 * t_new * dm * cfg["d_ff"]
                     + 2 * 2 * cfg["n_heads"] * t_new * t_total * hd)
        return cfg["n_layers"] * per_layer + 2 * t_new * dm * cfg["vocab_size"]

    total = [flops(prompt + i, prompt + i) / 1e6 for i in steps]
    useful = [flops(1, prompt + i) / 1e6 for i in steps]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    ax.fill_between(list(steps), useful, total, color="#E7ECF5",
                    label="computed, then thrown away")
    ax.plot(list(steps), total, color=T.AMBER, linewidth=T.LINE_WIDTH,
            label="arithmetic performed")
    ax.plot(list(steps), useful, color=T.BLUE, linewidth=T.LINE_WIDTH,
            linestyle="--", label="arithmetic that earned the token")

    last = d["waste"][-1]
    ax.annotate(f"{last['wasted_fraction'] * 100:.1f}% discarded",
                xy=(len(total) - 1, total[-1]), xytext=(len(total) * 0.52, total[-1] * 0.78),
                fontsize=8.5, color=T.INK,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=T.INK))

    ax.set_xlabel("tokens written so far")
    ax.set_ylabel("arithmetic for one token (MFLOP)")
    ax.set_title("Almost all of it is thrown away", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(0, max(total) * 1.1)
    T.style(ax)

    save(fig, "ch11-waste", d,
         alt=("Arithmetic performed for each token written, against the "
              "arithmetic that actually earned it. The work needed stays "
              "almost flat while the work performed rises with every token, "
              f"and the gap between them is discarded. By the last token, "
              f"{last['wasted_fraction'] * 100:.1f}% of the arithmetic is computed and "
              "then dropped."))




# --- Chapter 14: a block table, and what block size costs ---------------

def fig_blocktable(d: dict) -> None:
    """A diagram: contiguous to the sequence, scattered in the pool."""
    from matplotlib.patches import FancyArrowPatch, Rectangle

    bs = d["default_block_size"]
    owned = [5, 1, 6]                  # this sequence's blocks, out of order
    pool_blocks = 8
    length = 40                        # tokens it holds

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 68); ax.axis("off")

    # What the sequence sees: one unbroken run of tokens.
    ax.text(0, 55, "What the sequence sees", fontsize=8.6, color=T.INK,
            fontweight="bold", va="top")
    w = 96 / len(owned) / bs
    for i in range(length):
        ax.add_patch(Rectangle((i * w, 44), w * 0.92, 5.5, facecolor="#DCE7FC",
                               edgecolor=T.BLUE, linewidth=0.4))
    ax.text(0, 42, f"tokens 0 to {length - 1}, contiguous", fontsize=7.2,
            color=T.MUTED, va="top")

    # Its block table.
    ax.text(0, 36, "Its block table", fontsize=8.6, color=T.INK,
            fontweight="bold", va="top")
    for j, b in enumerate(owned):
        x = j * 15
        ax.add_patch(Rectangle((x, 26), 13, 6, facecolor="#FDF6EA",
                               edgecolor=T.AMBER, linewidth=1.1))
        ax.text(x + 6.5, 30.2, f"block {b}", ha="center", va="center",
                fontsize=7.6, color=T.INK)
        ax.text(x + 6.5, 27.4, f"tokens {j * bs}-{min((j + 1) * bs, length) - 1}",
                ha="center", va="center", fontsize=6.4, color=T.MUTED)

    # The pool: physical blocks, this sequence's scattered among others.
    ax.text(0, 18, "The pool, shared by everyone", fontsize=8.6, color=T.INK,
            fontweight="bold", va="top")
    bw = 96 / pool_blocks
    for b in range(pool_blocks):
        mine = b in owned
        ax.add_patch(Rectangle((b * bw, 4), bw * 0.9, 8,
                               facecolor="#FDF6EA" if mine else "#F2F4F8",
                               edgecolor=T.AMBER if mine else T.RULE,
                               linewidth=1.2 if mine else 0.8))
        ax.text(b * bw + bw * 0.45, 8, str(b), ha="center", va="center",
                fontsize=7.4, color=T.INK if mine else T.MUTED)
    ax.text(0, 2, "amber blocks belong to this sequence; "
                  "the rest are other sequences' or free",
            fontsize=7.0, color=T.MUTED, va="top")

    for j, b in enumerate(owned):
        ax.add_patch(FancyArrowPatch((j * 15 + 6.5, 25.4),
                                     (b * bw + bw * 0.45, 12.4),
                                     arrowstyle="-|>", mutation_scale=8,
                                     linewidth=1.0, color=T.AMBER,
                                     connectionstyle="arc3,rad=-0.12"))

    ax.text(0, 67, "Contiguous to the sequence, scattered in memory",
            fontsize=11, color=T.INK, fontweight="bold", va="top")

    save(fig, "ch14-blocktable", d,
         alt=("A diagram in three rows. The top row shows a sequence's 40 "
              "tokens as one unbroken run, which is what the sequence sees. "
              "The middle row is its block table, three entries naming "
              f"physical blocks {owned}. The bottom row is the shared pool of "
              f"{pool_blocks} blocks, with this sequence's three highlighted and "
              "scattered among blocks belonging to others. Arrows link each "
              "table entry to the block it names."))


def fig_blocksize(d: dict) -> None:
    rows = d["block_sizes"]
    x = [r["block_size"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    ax.plot(x, [r["utilization"] * 100 for r in rows], color=T.BLUE,
            marker="o", markersize=4.5, linewidth=T.LINE_WIDTH)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(v) for v in x])

    for r in rows:
        if r["block_size"] in (1, d["default_block_size"], x[-1]):
            ax.annotate(f"{r['admitted_paged']} sequences",
                        (r["block_size"], r["utilization"] * 100),
                        textcoords="offset points", xytext=(0, -16),
                        ha="center", fontsize=7.2, color=T.MUTED)

    default = d["at_default"]
    ax.axvline(default["block_size"], color=T.AMBER, linestyle=":", linewidth=1.2)
    ax.text(default["block_size"] * 1.15, min(r["utilization"] * 100 for r in rows) + 0.4,
            f"{default['block_size']} tokens:\nwhat engines use", fontsize=7.2,
            color=T.AMBER)

    ax.set_xlabel("block size (tokens)")
    ax.set_ylabel("share of held memory in use (%)")
    ax.set_title("Bigger blocks waste more, and there is a lot of room",
                 loc="left", fontsize=11)
    T.style(ax)

    save(fig, "ch14-blocksize", d,
         alt=("Share of held memory actually in use against block size, x axis "
              f"logarithmic. Utilization falls from 100% with one-token blocks to "
              f"{rows[-1]['utilization'] * 100:.0f}% with {rows[-1]['block_size']}-token blocks, and "
              f"the number of sequences admitted falls from {rows[0]['admitted_paged']} to "
              f"{rows[-1]['admitted_paged']}. The dotted line marks {d['default_block_size']} tokens, "
              "the size production engines use."))


# --- Chapter 15: a prefix tree, what a hit saves, and what fits --------

def fig_prefixtree(d: dict) -> None:
    """A diagram: three conversations, one copy of what they share."""
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=200)
    ax.set_xlim(0, 88); ax.set_ylim(-7, 58); ax.axis("off")
    ax.text(0, 57, "One copy of what they share", fontsize=11, color=T.INK,
            fontweight="bold", va="top")

    bw, bh = 11.0, 7.5

    def block(x, y, label, sub, shared):
        ax.add_patch(Rectangle((x, y), bw, bh,
                               facecolor="#DCE7FC" if shared else "#FDF6EA",
                               edgecolor=T.BLUE if shared else T.AMBER,
                               linewidth=1.2))
        ax.text(x + bw / 2, y + 4.7, label, ha="center", va="center",
                fontsize=7.4, color=T.INK)
        ax.text(x + bw / 2, y + 2.0, sub, ha="center", va="center",
                fontsize=6.3, color=T.MUTED)

    # The shared trunk: the system prompt, held by everyone at once.
    trunk_y, trunk = 38, [11, 2, 6]
    ax.text(0, 50.5, "the system prompt every session sends, stored once",
            fontsize=7.6, color=T.MUTED, va="center")
    for i, b in enumerate(trunk):
        block(i * (bw + 1.6), trunk_y, f"block {b}", "3 holders", True)
    trunk_right = 3 * (bw + 1.6) - 1.6

    # Three branches, each with its own physical blocks.
    rows = [("session A", 25, [9, 14]), ("session B", 13.5, [4, 12]),
            ("session C", 2, [7, 15])]
    for name, y, blocks in rows:
        for i, b in enumerate(blocks):
            block(46 + i * (bw + 1.6), y, f"block {b}", "1 holder", False)
        ax.text(46 + 2 * (bw + 1.6) + 1.5, y + bh / 2, name, fontsize=7.6,
                color=T.INK, ha="left", va="center")
        ax.add_patch(FancyArrowPatch((trunk_right + 0.6, trunk_y + bh / 2),
                                     (45, y + bh / 2),
                                     arrowstyle="-|>", mutation_scale=8,
                                     linewidth=1.0, color=T.MUTED,
                                     connectionstyle="arc3,rad=0.18"))

    ax.text(0, -4.5, "blue: one set of blocks, three holders     "
                     "amber: private to one session",
            fontsize=7.0, color=T.MUTED, va="center")

    save(fig, "ch15-prefixtree", d,
         alt=("A diagram of a prefix tree. A trunk of three blue blocks holds "
              "the system prompt every session sends, each labelled three "
              "holders. Arrows lead from the end of the trunk to three "
              "branches, one per session, each of two amber blocks with "
              "different block numbers, each labelled one holder. The shared "
              "blocks are stored once rather than three times."))


def fig_prefill(d: dict) -> None:
    """Measured prefill time against how much of the prompt was cached."""
    pts = d["prefill"]["points"]
    x = [p["shared_frac"] * 100 for p in pts]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=200)

    ax.plot(x, [p["cold_s"]["median"] * 1e3 for p in pts], color=T.AMBER,
            linestyle="-", marker="o", markersize=4, linewidth=T.LINE_WIDTH,
            label="no cache hit")
    ax.plot(x, [p["warm_s"]["median"] * 1e3 for p in pts], color=T.BLUE,
            linestyle="--", marker="s", markersize=4, linewidth=T.LINE_WIDTH,
            label="prefix already cached")
    ax.set_xlabel("share of the prompt already cached (%)")
    ax.set_ylabel("prefill time (ms)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7.6)
    ax.set_title("Prefill time falls with the hit", loc="left", fontsize=10)
    T.style(ax)

    ax2.plot(x, [p["speedup_if_proportional"] for p in pts], color=T.MUTED,
             linestyle=":", marker="", linewidth=1.4,
             label="if time went with tokens computed")
    ax2.plot(x, [p["speedup"] for p in pts], color=T.BLUE, linestyle="--",
             marker="s", markersize=4, linewidth=T.LINE_WIDTH, label="measured")
    last = pts[-1]
    ax2.annotate(f"{last['speedup']:.0f}x, not "
                 f"{last['speedup_if_proportional']:.0f}x",
                 (x[-1], last["speedup"]), textcoords="offset points",
                 xytext=(-10, -1), ha="right", fontsize=7.2, color=T.INK)
    ax2.set_xlabel("share of the prompt already cached (%)")
    ax2.set_ylabel("prefill speedup")
    ax2.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax2.set_title("and falls short of proportional", loc="left", fontsize=10)
    T.style(ax2)

    save(fig, "ch15-prefill", d,
         alt=("Two panels. Left: measured prefill time against the share of "
              f"the prompt already cached, falling from about "
              f"{pts[0]['cold_s']['median'] * 1e3:.0f} milliseconds with no hit to "
              f"{pts[-1]['warm_s']['median'] * 1e3:.0f} milliseconds when "
              f"{pts[-1]['shared_frac'] * 100:.0f}% is cached, while the no-hit line "
              "stays flat. Right: the speedup that produces, measured against "
              "what it would be if time went strictly with the number of "
              f"tokens computed. The two agree up to about 75% cached, then "
              f"the measured curve falls short: {pts[-1]['speedup']:.0f} times "
              f"rather than {pts[-1]['speedup_if_proportional']:.0f}."))


def fig_hitrate(d: dict) -> None:
    """Hit rate against cache size, for three eviction policies."""
    ceiling = d["unlimited"]["hit_rate"] * 100
    styles = {"lru": (T.BLUE, "--", "s", "least recently used leaf"),
              "lfu": (T.AMBER, "-", "o", "least frequently used leaf"),
              "unstructured": (T.MUTED, ":", "^", "least recently used block")}

    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    gb = d["assumptions"]["gb_per_block"]
    for policy, (colour, ls, marker, label) in styles.items():
        rows = [r for r in d["sizes"] if r["policy"] == policy]
        ax.plot([r["pool_blocks"] * gb for r in rows],
                [r["hit_rate"] * 100 for r in rows], color=colour, linestyle=ls,
                marker=marker, markersize=4, linewidth=T.LINE_WIDTH, label=label)

    ax.axhline(ceiling, color=T.INK, linestyle="-", linewidth=0.9, alpha=0.5)
    sizes_gb = sorted({r["pool_blocks"] * gb for r in d["sizes"]})
    ax.annotate(f"everything this traffic can share: {ceiling:.0f}%",
                (sizes_gb[0], ceiling), textcoords="offset points",
                xytext=(2, 5), fontsize=7.2, color=T.INK)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 7, 10, 20, 30],
                  ["1", "2", "3", "5", "7", "10", "20", "30"])
    ax.set_xlabel("cache size (GB of keys and values, at 8B scale)")
    ax.set_ylabel("prompt tokens served from cache (%)")
    ax.set_ylim(40, 100)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    ax.set_title("Most of the hit rate comes from the first few gigabytes",
                 loc="left", fontsize=11)
    T.style(ax)

    lru = [r for r in d["sizes"] if r["policy"] == "lru"]
    save(fig, "ch15-hitrate", d,
         alt=("Share of prompt tokens served from cache against cache size, x "
              "axis logarithmic, for three eviction policies. All three rise "
              f"steeply and flatten: least-recently-used-leaf reaches "
              f"{lru[2]['hit_rate'] * 100:.0f}% at "
              f"{lru[2]['pool_blocks'] * gb:.0f} GB and "
              f"{lru[-1]['hit_rate'] * 100:.0f}% at "
              f"{lru[-1]['pool_blocks'] * gb:.0f} GB, against a ceiling of "
              f"{ceiling:.0f}% with unlimited memory. Least-frequently-used "
              "trails the other two badly at small sizes."))


CHAPTERS = {"ch01": [fig_cost], "ch02": [fig_pipeline, fig_attention, fig_scores],
            "ch03": [fig_timeline, fig_per_token, fig_intensity],
            "ch04": [fig_cliff, fig_wall],
            "ch05": [fig_tradeoff, fig_tail],
            "ch06": [fig_breakeven],
            "ch07": [fig_size, fig_core_scaling],
            "ch08": [fig_roofline, fig_batching_roof],
            "ch09": [fig_framings],
            "ch10": [fig_framework],
            "ch11": [fig_waste], "ch12": [fig_per_step, fig_scaling], "ch13": [fig_memory], "ch14": [fig_blocktable, fig_blocksize],
            "ch15": [fig_prefixtree, fig_prefill, fig_hitrate]}


def _check_no_shared_figure_functions() -> None:
    """A function defined twice silently rebinds, and one chapter then draws
    another's figure. Names are checked rather than trusted."""
    seen: dict[int, str] = {}
    for chapter, figs in CHAPTERS.items():
        for fn in figs:
            if id(fn) in seen:
                raise RuntimeError(
                    f"{fn.__name__} is used by both {seen[id(fn)]} and {chapter}; "
                    "a duplicate definition has rebound it")
            seen[id(fn)] = chapter


def main() -> None:
    _check_no_shared_figure_functions()
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
