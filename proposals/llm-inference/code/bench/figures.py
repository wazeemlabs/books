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

# Make an SVG depend only on the figure, not on when or where it was
# drawn. Matplotlib names its reusable elements with a random salt and
# stamps the file with the current time, so every rebuild rewrote every
# figure and a real change was invisible in the diff. With these two
# lines the same data draws the same bytes on any machine, which is
# what makes "regenerate and see nothing change" a check worth running.
matplotlib.rcParams["svg.hashsalt"] = "llm-inference-from-the-ground-up"
SVG_METADATA = {"Date": None}
import matplotlib.pyplot as plt

from . import legibility as L
from . import theme as T

RESULTS = Path("results")
FIGS = Path("figures")

NAIVE = dict(color=T.AMBER, linestyle="-", marker="o", markersize=3.5, linewidth=1.8)
CACHED = dict(color=T.BLUE, linestyle="--", marker="s", markersize=3.5, linewidth=1.8)


def _log2_ticks(ax, x) -> None:
    """Powers-of-two ticks, rotated and anchored under their own tick.

    Rotating a tick label without setting `ha="right"` leaves it centred
    on its own width, which slides it to the right of the tick it
    belongs to -- far enough, on a crowded axis, to sit under the
    neighbouring one.
    """
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [f"{v:,}" for v in x], rotation=45, fontsize=7)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def caption(d: dict) -> str:
    p = d["provenance"]
    if "splits" in d and "transfer" in d:  # Chapter 19: two pools
        a = d["assumptions"]
        return (f"SIMULATION, not a measurement: {a['n_requests']:,} requests "
                f"arriving as a Poisson process at {a['rate']} a second, "
                f"prompt and output lengths from the case study (means "
                f"{a['prompt_mean']:,} and {a['output_mean']}, seed "
                f"{a['seed']}), through {a['fleet']} identical accelerators - "
                f"what a prefill and a decode step cost is arithmetic over the "
                f"reference model and published hardware specifications, not a "
                f"timing run; link speeds are published specifications "
                f"(FACTS.md) - throughput measured over the arrival window "
                f"less the first {a['warm_fraction'] * 100:.0f}% - "
                f"commit {p['commit']}, {p['measured_utc']}")
    if "budgets" in d and "swap_arithmetic" in d:  # Chapter 18: policies
        a = d["assumptions"]
        return (f"SIMULATION, not a measurement: {a['n_requests']} requests "
                f"arriving as a Poisson process, prompt and output lengths "
                f"from the case study (means {a['prompt_mean']:,} and "
                f"{a['output_mean']}, seed {a['seed']}), driven through the "
                f"schedulers in tinyserve/scheduler.py - what a mixed "
                f"iteration costs is arithmetic over the reference model and "
                f"published hardware specifications, not a timing run - block "
                f"pool {a['pool_bytes'] / 1e9:.0f} GB, {a['blocks']:,} blocks "
                f"of {a['block']} tokens; interconnect speeds from FACTS.md - "
                f"commit {p['commit']}, {p['measured_utc']}")
    if "head_to_head" in d and "pool_sweep" in d:  # Chapter 17: the scheduler
        a = d["assumptions"]
        return (f"SIMULATION, not a measurement: {a['n_requests']} requests "
                f"arriving as a Poisson process, prompt and output lengths "
                f"from the case study (means {a['prompt_mean']:,} and "
                f"{a['output_mean']}, seed {a['seed']}), driven through the "
                f"schedulers in tinyserve/scheduler.py - what a prefill and a "
                f"decode step cost is arithmetic over the reference model and "
                f"published hardware specifications, not a timing run - block "
                f"pool {a['pool_bytes'] / 1e9:.0f} GB, {a['blocks']:,} blocks "
                f"of {a['block']} tokens - commit {p['commit']}, "
                f"{p['measured_utc']}")
    if "tiles" in d and "traffic" in d:  # Chapter 20: bytes, counted and derived
        a = d["assumptions"]
        return (f"COUNTED BYTES, not a timing run: every read and write "
                f"between the two levels of memory is counted as the kernels "
                f"in tinyserve/flash.py make it, on "
                f"{a['measured_heads']} heads of {a['measured_head_dim']} "
                f"dimensions at lengths {a['lengths'][0]}-"
                f"{a['lengths'][-1]}, seed {a['seed']}; figures for the "
                f"reference model apply the byte formulas those counts "
                f"verify exactly, to {d['reference']['heads']} heads "
                f"({d['reference']['kv_heads']} shared) of "
                f"{d['reference']['head_dim']} in bf16, with tiles of "
                f"{a['q_tile']}x{a['kv_tile']} - times are bytes over "
                f"{a['hbm_bytes_per_s'] / 1e12:.2f} TB/s and shared memory "
                f"is {a['sram_bytes_per_sm'] / 1024:.0f} KB per "
                f"multiprocessor, both from FACTS.md - commit {p['commit']}, "
                f"{p['measured_utc']}")
    if "sizing" in d and "theory" in d:  # Chapter 41: capacity planning
        a = d["assumptions"]
        return (f"SIMULATION, not a measurement: {a['n_requests']:,} requests "
                f"arriving as a Poisson process at each of "
                f"{len(a['rates'])} offered loads (means {a['prompt_mean']:,} "
                f"prompt and {a['output_mean']} output tokens, seed "
                f"{a['seed']}), driven through the scheduler of Chapter 18 "
                f"at a {a['token_budget']}-token budget over "
                f"{a['blocks']:,} blocks - what a step costs is arithmetic "
                f"over the reference model, not a timing run - everything is "
                f"measured in a window from {a['window'][0]:.0%} to "
                f"{a['window'][1]:.0%} of the arrival span, past the ramp "
                f"and before the drain - commit {p['commit']}, "
                f"{p['measured_utc']}")
    if "exactness" in d and "speedups" in d:  # Chapter 29: speculation
        a = d["assumptions"]
        return (f"SAMPLING AND ARITHMETIC, not a timing run: the acceptance "
                f"rule is sampled {a['draws']:,} times against a "
                f"deliberately wrong draft (seed {a['seed']}); acceptance "
                f"rates come from quantized copies of a "
                f"{a['model']['layers']}-layer model over "
                f"{a['prompt_tokens']} positions, and that model is "
                f"untrained, so its output is flatter than a real one's and "
                f"its sampled acceptance correspondingly higher - speedups "
                f"are the closed form checked against simulation, over a "
                f"decode step of the reference model priced by Chapter 16's "
                f"cost model - commit {p['commit']}, {p['measured_utc']}")
    if "schemes" in d and "outliers" in d:  # Chapter 24: quantization
        a = d["assumptions"]
        return (f"EXACT ARITHMETIC, not a timing run: weights are quantized "
                f"and recovered by hand in tinyserve/quantize.py, over "
                f"{d['schemes']['matrices']} weight matrices of a "
                f"{a['model']['layers']}-layer model of width "
                f"{a['model']['d_model']}, seed {a['seed']} - the outlier "
                f"sweep is constructed, not observed, and says so - memory "
                f"figures are the reference model's "
                f"{a['reference_params'] / 1e9:.0f}B parameters at each "
                f"scheme's bytes a weight, scales included - commit "
                f"{p['commit']}, {p['measured_utc']}")
    if "formats" in d and "accumulation" in d:  # Chapter 22: number formats
        a = d["assumptions"]
        return (f"EXACT ARITHMETIC, not a timing run: rounding is done by "
                f"hand on the bit fields in tinyserve/precision.py, checked "
                f"against NumPy's float16 over "
                f"{d['verification']['values_compared']:,} values; errors are "
                f"root-mean-square over a {a['accumulation_shape'][0]}x"
                f"{a['accumulation_shape'][1]} product, seed {a['seed']} - "
                f"peak arithmetic is the H100 SXM datasheet's dense figures "
                f"(its tensor-core rows are quoted with sparsity and halved "
                f"here), memory bandwidth "
                f"{d['hardware']['hbm_bytes_per_s'] / 1e12:.2f} TB/s, both "
                f"from FACTS.md - model {a['model']['params'] / 1e9:.0f}B "
                f"parameters, width {a['model']['d_model']:,} - "
                f"commit {p['commit']}, {p['measured_utc']}")
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
    if "matmuls" in d and "waste" in d:  # Chapter 16: batching
        e, mc = d["experiment"], d["machine"]
        return (f"measured on {p['hardware']['cpu']}, "
                f"{p['hardware']['cores_available']} vCPU, NumPy "
                f"{p['software']['numpy']} on {p['software']['blas']}, "
                f"{mc['threads']} threads - peak {mc['flops'] / 1e9:,.0f} GFLOP/s, "
                f"{mc['dram_bytes_per_s'] / 1e9:.1f} GB/s from memory - median of "
                f"{e['runs']} runs after {e['warmup']} warmup, {e['steps']} decode "
                f"steps each - batch waste computed over {e['n_requests']:,} "
                f"sampled requests (seed {e['seed']}) - commit {p['commit']}, "
                f"{p['measured_utc']}")
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
    # The last batch is not a power of two, and on a log axis its label
    # lands on top of the one before it. Tick the powers of two; the
    # last point is called out by its annotation instead.
    ticks = [v for v in x if v & (v - 1) == 0]
    ax.set_xticks(ticks, [str(v) for v in ticks], fontsize=7.5)
    ax.set_xlabel("Sequences decoded at the same time")
    ax.set_ylabel("USD per million output tokens (log)")
    ax.set_title("The same GPU and the same model, "
                 f"{d['spread']['ratio']:.0f}x apart in cost", loc="left")
    # Headroom above the first point for its two-line label, which
    # otherwise runs into the title.
    ax.set_ylim(min(y) / 2.4, max(y) * 2.8)
    # Below and right of the first point, not above it: above it is the
    # title. Below and left of the last, which is at the panel's edge.
    # Above each end point. Below the first is the curve's own descent;
    # below the last is the API price line.
    for r, dx, dy, ha in ((rows[0], 7, 9, "left"),
                          (rows[-1], 2, 34, "right")):
        ax.annotate(f"${r['usd_per_m_tokens']:.3f}, batch {r['batch']}\n"
                    f"{r['flop_utilization'] * 100:.1f}% of peak FLOPs",
                    (r["batch"], r["usd_per_m_tokens"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=7.5, color=T.INK)
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
    # Read every label's position now, while the figure is still a live
    # object: once it is an SVG nobody checks whether it can be read.
    L.REPORT.add(fig, name)
    fig.savefig(FIGS / f"{name}.svg", bbox_inches="tight",
                metadata=SVG_METADATA)
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
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
    # On a white ground: this sits below the bars, where a gridline
    # would otherwise run through the middle of a word.
    ax.text(uniform + 0.15, -0.9,
            f"a coin-toss guess would be {uniform:.1f}%",
            fontsize=7, color=T.AMBER, va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
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
    # To the left of the rule: centred on it, the rule runs up through
    # the gap between two of the words.
    ax.text(ridge * 0.93, -0.55, f"this accelerator breaks even at {ridge:.0f}",
            ha="right", fontsize=7.6, color=T.INK)
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
        # Beside the rule, not centred on it: a centred label has the
        # rule running up through the gap between its two words.
        near_right = lvl["kib"] > (min(x) * max(x)) ** 0.5
        ax.text(lvl["kib"] * (0.88 if near_right else 1.14), max(y) * 1.03,
                f"L{lvl['level']} ends\n{lvl['kib']:,} KiB", fontsize=7,
                color=T.MUTED, va="bottom",
                ha="right" if near_right else "left")

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

    # Headroom above and below, for the callouts at both ends of the
    # curves; both ends sit in the corners of the panel.
    ax.set_ylim(min(predicted) / 2.6, max(measured) * 2.4)

    l3 = next((c for c in d["cache_topology"] if c["level"] == 3), None)
    if l3 and min(x) < l3["kib"] / 1024 < max(x):
        ax.axvline(l3["kib"] / 1024, color=T.MUTED, linestyle=":", linewidth=1.0)
        # To the left of the rule, above where both curves run.
        ax.text(l3["kib"] / 1024 * 0.92, max(measured) * 2.0,
                "weights outgrow\nlast-level cache",
                fontsize=7, color=T.MUTED, va="top", ha="right")

    first, last = w[0], w[-1]
    # Below the amber curve, which is below the blue one at this end.
    ax.annotate(f"{first['measured_over_predicted']:.1f}x above",
                xy=(first["weight_mib"], first["predicted_from_bandwidth_s"] * 1e3),
                textcoords="offset points", xytext=(7, -9), fontsize=7.5,
                color=T.MUTED, va="top")
    # Above the blue curve, which is the upper one at this end.
    ax.annotate(f"{last['measured_over_predicted']:.2f}x",
                xy=(last["weight_mib"], last["decode_step_s"] * 1e3),
                textcoords="offset points", xytext=(-3, 9), fontsize=7.5,
                color=T.MUTED, ha="right", va="bottom")

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
    marked = [r for r in c
              if r["batch"] in (1, 16, 64, 174, 325) or r is c[-1]]

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
                xytext=(last["tokens_per_s"] * 0.34, last["inter_token_ms"] * 2.6),
                fontsize=7.2, color=T.INK,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=T.INK))
    # Room on the right for the last point's label, which otherwise
    # hangs outside the panel.
    ax.set_xlim(min(x) / 1.25, max(x) * 1.55)
    T.style(ax)
    # Placed after the budget lines and the callout, so it knows about
    # them: at the left the curve is nearly flat and a fixed offset puts
    # the label straight through it.
    L.label_points(ax, [(r["tokens_per_s"], r["inter_token_ms"],
                         f"batch {r['batch']}") for r in marked],
                   fontsize=7.2, color=T.MUTED)

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
    # A band above every bar, so no label is set over the distribution
    # it is describing. The median's rule stands at the mode, where the
    # bars are tallest and there is otherwise nowhere to put its label.
    ax.set_ylim(0, max(counts) * 1.22)
    for label, key, style, ha in (("p50", "p50_ms", "-", "left"),
                                  ("p99", "p99_ms", "--", "right"),
                                  ("p99.9", "p999_ms", ":", "left")):
        ax.axvline(m[key], color=T.AMBER, linestyle=style, linewidth=1.4)
        pad = (max(centres) - min(centres)) * 0.008
        ax.text(m[key] + (pad if ha == "left" else -pad), max(counts) * 1.13,
                f"{label} {m[key]:.2f} ms", fontsize=7.4, color=T.AMBER,
                va="center", ha=ha)

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

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(min(p["achieved_flops"] for p in pts) / 1e9 / 2.5,
                peak / 1e9 * 2.2)
    ax.set_xlabel("arithmetic per byte fetched (log)")
    ax.set_ylabel("rate achieved (GFLOP/s, log)")
    ax.set_title("Every operation sits under the roof", loc="left", fontsize=11)
    ax.text(m["ridge_flop_per_byte"], peak / 1e9 * 1.35,
            f" breaks even at {m['ridge_flop_per_byte']:.0f}", fontsize=7.4,
            color=T.MUTED)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    T.style(ax)
    # Placed last, once the roof and the axis limits are final: three of
    # these points sit within a hair of each other on the flat part of
    # the roof, and a fixed offset stacks their labels.
    L.label_points(ax, [(p["intensity"], p["achieved_flops"] / 1e9, p["name"])
                        for p in pts], fontsize=6.8, color=T.INK)

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
    marked = [r for r in rows if r["batch"] in (1, 32, 325)]

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
    # To the left of the rule, not the right: to the right there is only
    # the edge of the panel.
    ax.set_ylim(min(r["achieved_flops"] for r in rows) / 1e12 / 2.2,
                peak / 1e12 * 2.4)
    ax.text(a["ridge_flop_per_byte"] * 0.93, peak / 1e12 * 1.3,
            f"breaks even at {a['ridge_flop_per_byte']:.0f}", fontsize=7.4,
            color=T.MUTED, ha="right")
    # Upper left: the panel below the slope is where the batch labels
    # go, and the rule at the break-even point runs through the lower
    # right corner where a legend would otherwise sit.
    ax.legend(frameon=False, fontsize=7.8, loc="upper left",
              bbox_to_anchor=(0.01, 0.94))
    T.style(ax)
    L.label_points(ax, [(r["intensity"], r["achieved_flops"] / 1e12,
                         f"batch {r['batch']}") for r in marked],
                   fontsize=7.2, color=T.MUTED)

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

    # Inside the bars, not past their ends: the reference line stands at
    # the median, so three of the six bars end within a few points of
    # it and their labels are struck through by it.
    for i, w in enumerate(rows):
        ax.text(w["tokens_per_s"] - max(values) * 0.012, i,
                f"{w['tokens_per_s']:,.0f}  ({w['relative_to_honest']:.2f}x)",
                va="center", ha="right", fontsize=7.4, color="#FFFFFF")

    ax.set_yticks(range(len(rows)), labels, fontsize=7.6)
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xlabel("tokens per second reported")
    ax.set_title("One measurement, reported six defensible ways", loc="left",
                 fontsize=11)
    # Below the top bar rather than above it, where the title is.
    ax.text(honest + max(values) * 0.012, -0.62, "what this book reports",
            fontsize=7.4, color=T.INK, va="center")
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

    marked = [r for r in rows
              if r["block_size"] in (1, d["default_block_size"], x[-1])]

    default = d["at_default"]
    ax.axvline(default["block_size"], color=T.AMBER, linestyle=":", linewidth=1.2)
    ax.text(default["block_size"] * 1.15, min(r["utilization"] * 100 for r in rows) + 0.4,
            f"{default['block_size']} tokens:\nwhat engines use", fontsize=7.2,
            color=T.AMBER)

    ax.set_xlabel("block size (tokens)")
    ax.set_ylabel("share of held memory in use (%)")
    ax.set_title("Bigger blocks waste more, and there is a lot of room",
                 loc="left", fontsize=11)
    # Room on both sides: the first and last points carry labels, and at
    # the edges of the panel there is nowhere for them to go.
    ax.set_xlim(x[0] / 1.9, x[-1] * 2.6)
    lo = min(r["utilization"] * 100 for r in rows)
    hi = max(r["utilization"] * 100 for r in rows)
    ax.set_ylim(lo - (hi - lo) * 0.16, hi + (hi - lo) * 0.14)
    T.style(ax)
    L.label_points(ax, [(r["block_size"], r["utilization"] * 100,
                         f"{r['admitted_paged']} sequences") for r in marked],
                   fontsize=7.2, color=T.MUTED)

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


# --- Chapter 16: what a batch buys, and what a static one wastes -------

def fig_matmul(d: dict) -> None:
    """One weight matrix, many batch sizes: batching with nothing else in it."""
    big = next(m for m in d["matmuls"] if m["name"] == "memory-resident")
    x = [r["batch"] for r in big["rows"]]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.5), dpi=200)

    ax.plot(x, [r["seconds"] * 1e3 for r in big["rows"]], color=T.BLUE,
            linestyle="--", marker="s", markersize=4, linewidth=T.LINE_WIDTH)
    flat_to = big["flat_to"]
    flat = [r for r in big["rows"] if 2 <= r["batch"] <= flat_to]
    ax.axvspan(2, flat_to, color=T.BLUE, alpha=0.08)
    ax.annotate(f"2 to {flat_to} sequences:\nthe same time",
                (2, flat[0]["seconds"] * 1e3), textcoords="offset points",
                xytext=(8, 22), fontsize=7.2, color=T.INK)
    ax.set_xscale("log", base=2); ax.set_xticks(x, [str(v) for v in x])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("sequences in the batch")
    ax.set_ylabel("time for one weight matmul (ms)")
    ax.set_title(f"{big['weight_bytes'] / 1024**2:.0f} MiB of weights, "
                 "fetched for the batch", loc="left", fontsize=10)
    T.style(ax)

    peak = d["machine"]["flops"] / 1e9
    ax2.plot(x, [r["flops"] / 1e9 for r in big["rows"]], color=T.BLUE,
             linestyle="--", marker="s", markersize=4, linewidth=T.LINE_WIDTH)
    ax2.axhline(peak, color=T.INK, linewidth=0.9, alpha=0.5)
    ax2.annotate(f"what this machine can do: {peak:,.0f}", (x[0], peak),
                 textcoords="offset points", xytext=(2, -12), fontsize=7.2,
                 color=T.INK)
    ax2.set_xscale("log", base=2); ax2.set_xticks(x, [str(v) for v in x])
    ax2.set_ylim(0, peak * 1.15)
    ax2.set_xlabel("sequences in the batch")
    ax2.set_ylabel("arithmetic rate (GFLOP/s)")
    ax2.set_title("and the machine gets closer to working", loc="left",
                  fontsize=10)
    T.style(ax2)

    save(fig, "ch16-matmul", d,
         alt=("Two panels. Left: the time for one weight matrix multiply "
              f"against batch size. It is {big['rows'][1]['seconds'] * 1e3:.2f} "
              "milliseconds for two sequences and the same for four and eight "
              "-- four times the work for no extra time -- then rises to "
              f"{big['rows'][-1]['seconds'] * 1e3:.2f} milliseconds at "
              f"{big['rows'][-1]['batch']}. Right: the arithmetic rate that "
              f"implies, rising from {big['rows'][1]['flops'] / 1e9:.0f} to "
              f"{big['rows'][-1]['flops'] / 1e9:.0f} GFLOP/s, against the "
              f"{d['machine']['flops'] / 1e9:,.0f} GFLOP/s this machine reaches "
              "on a large square multiply. Batching closes most of that gap "
              "and does not close all of it."))


def fig_batch_tradeoff(d: dict) -> None:
    """Throughput against batch, and what it costs the user waiting."""
    big = next(m for m in d["models"] if m["name"] == "memory-resident")
    small = next(m for m in d["models"] if m["name"] == "cache-resident")
    x = [r["batch"] for r in big["rows"]]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.5), dpi=200)

    ax.plot(x, [r["tokens_per_s"] for r in big["rows"]], color=T.BLUE,
            linestyle="--", marker="s", markersize=4, linewidth=T.LINE_WIDTH,
            label="weights from memory")
    ax.plot(x, [r["tokens_per_s"] for r in small["rows"]], color=T.AMBER,
            linestyle="-", marker="o", markersize=4, linewidth=T.LINE_WIDTH,
            label="weights already in cache")
    ax.set_xscale("log", base=2); ax.set_xticks(x, [str(v) for v in x])
    ax.set_yscale("log")
    ax.set_xlabel("sequences in the batch")
    ax.set_ylabel("tokens per second, whole server")
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    ax.set_title("Throughput rises with the batch", loc="left", fontsize=10)
    T.style(ax)

    ax2.plot([r["inter_token_ms"] for r in big["rows"]],
             [r["tokens_per_s"] for r in big["rows"]], color=T.BLUE,
             linestyle="--", marker="s", markersize=4, linewidth=T.LINE_WIDTH)
    ax2.set_xlabel("what one user waits between tokens (ms)")
    ax2.set_ylabel("tokens per second, whole server")
    ax2.set_title("and the user pays for it", loc="left", fontsize=10)
    # Room at the top right, where the last point sits in the corner.
    tps = [r["tokens_per_s"] for r in big["rows"]]
    ax2.set_ylim(min(tps) - (max(tps) - min(tps)) * 0.12,
                 max(tps) + (max(tps) - min(tps)) * 0.14)
    T.style(ax2)
    L.label_points(ax2, [(r["inter_token_ms"], r["tokens_per_s"],
                          f"batch {r['batch']}") for r in big["rows"]
                         if r["batch"] in (1, 8, 64)],
                   fontsize=7.2, color=T.MUTED)

    save(fig, "ch16-tradeoff", d,
         alt=("Two panels, for the model whose weights do not fit in cache. "
              f"Left: server throughput against batch size, rising from "
              f"{big['rows'][0]['tokens_per_s']:.0f} tokens per second at batch "
              f"one to {big['rows'][-1]['tokens_per_s']:.0f} at batch "
              f"{big['rows'][-1]['batch']}, with a dip at batch two. Right: the "
              "same throughput against the time one user waits between tokens, "
              f"which grows from {big['rows'][0]['inter_token_ms']:.0f} to "
              f"{big['rows'][-1]['inter_token_ms']:.0f} milliseconds. Throughput "
              "is bought with latency."))


def fig_static_batch(d: dict) -> None:
    """A diagram: what a fixed batch does with its slots."""
    from matplotlib.patches import Patch, Rectangle

    rows = sorted(d["picture"], key=lambda r: r["prompt"] + r["output"],
                  reverse=True)
    padded, steps = rows[0]["padded_prompt"], rows[0]["batch_steps"]
    total = padded + steps

    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
    height, gap = 0.62, 0.38
    for i, r in enumerate(rows):
        y = len(rows) - i - 1
        ax.add_patch(Rectangle((0, y), r["prompt"], height, facecolor="#2563EB",
                               edgecolor="none"))
        ax.add_patch(Rectangle((r["prompt"], y), padded - r["prompt"], height,
                               facecolor="#DCE7FC", edgecolor="none", hatch="///"))
        ax.add_patch(Rectangle((padded, y), r["output"], height,
                               facecolor="#B7780F", edgecolor="none"))
        ax.add_patch(Rectangle((padded + r["output"], y), steps - r["output"],
                               height, facecolor="#FDF6EA", edgecolor="none",
                               hatch="///"))
        ax.text(-total * 0.015, y + height / 2, f"seq {i + 1}", ha="right",
                va="center", fontsize=7.2, color=T.MUTED)

    ax.set_xlim(0, total); ax.set_ylim(-0.4, len(rows))
    ax.axvline(padded, color=T.INK, linewidth=0.9, alpha=0.6)
    ax.text(padded, len(rows) - 0.25, "  decoding starts", fontsize=7.2,
            color=T.INK, va="center")
    ax.set_yticks([])
    ax.set_xlabel("tokens the batch holds a slot for "
                  "(prompt, then one per decode step)")
    ax.set_title("A fixed batch pays for the longest of everything",
                 loc="left", fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(handles=[Patch(facecolor="#2563EB", label="prompt"),
                       Patch(facecolor="#DCE7FC", hatch="///", label="padding"),
                       Patch(facecolor="#B7780F", label="generated"),
                       Patch(facecolor="#FDF6EA", hatch="///",
                             label="finished, slot still held")],
              frameon=False, fontsize=7.4, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, -0.42))

    at8 = next(w for w in d["waste"] if w["batch"] == len(rows))
    save(fig, "ch16-static", d,
         alt=(f"A chart of {len(rows)} sequences sharing one fixed batch, each "
              "drawn as a horizontal bar. Every prompt is padded out to the "
              f"longest, {padded} tokens, and every sequence holds its slot "
              f"until the longest generation finishes after {steps} steps. "
              "Solid colour is real work, hatched is paid for and wasted; "
              f"across the traffic only {at8['total_utilization'] * 100:.0f}% of "
              f"what a batch of {len(rows)} pays for is work."))


# --- Chapter 17: the batch decided again at every step -----------------

def fig_scheduler_timeline(d: dict) -> None:
    """The whole idea in one picture: when each request is being worked on."""
    from matplotlib.patches import Patch

    tl = d["timeline"]
    end = max(r["finish_s"] for name in tl for r in tl[name])

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.2), dpi=200, sharex=True)
    for ax, name, title in ((axes[0], "static", "Static: the batch is chosen once"),
                            (axes[1], "continuous",
                             "Continuous: the batch is chosen again every step")):
        rows = tl[name]
        for i, r in enumerate(rows):
            y = len(rows) - i - 1
            ax.barh(y, r["first_token_s"] - r["arrival_s"], left=r["arrival_s"],
                    height=0.62, color="#DCE7FC", edgecolor="none")
            ax.barh(y, r["finish_s"] - r["first_token_s"], left=r["first_token_s"],
                    height=0.62, color=T.BLUE, edgecolor="none")
            ax.plot([r["arrival_s"]], [y], marker="|", markersize=7,
                    color=T.INK, markeredgewidth=1.2)
        ax.set_ylim(-0.8, len(rows) - 0.2)
        ax.set_yticks([])
        ax.set_xlim(0, end * 1.02)
        ax.set_ylabel("requests, in arrival order", fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xlabel("seconds since the first request arrived")
    axes[1].legend(handles=[Patch(facecolor="#DCE7FC", label="waiting"),
                            Patch(facecolor=T.BLUE, label="generating"),
                            plt.Line2D([], [], color=T.INK, marker="|",
                                       linestyle="none", markersize=7,
                                       label="arrived")],
                   frameon=False, fontsize=7.6, ncol=3, loc="lower center",
                   bbox_to_anchor=(0.5, -0.62))

    st, co = tl["static"], tl["continuous"]
    wait = lambda rows: sum(r["first_token_s"] - r["arrival_s"]
                            for r in rows) / len(rows)
    save(fig, "ch17-timeline", d,
         alt=(f"Two stacked charts of the same {len(st)} requests. In the top "
              "one, static batching, every request waits until the whole batch "
              "is formed, then they all start and all keep their slot until the "
              f"longest finishes; the mean wait for a first token is "
              f"{wait(st):.1f} seconds and the last request finishes at "
              f"{max(r['finish_s'] for r in st):.1f} seconds. In the bottom "
              "one, continuous batching, each bar starts almost at its arrival "
              "mark and ends when that request is done; the mean wait is "
              f"{wait(co):.2f} seconds and the last finishes at "
              f"{max(r['finish_s'] for r in co):.1f} seconds."))


def fig_load(d: dict) -> None:
    """What the two schedulers do as the traffic rises."""
    a = d["assumptions"]
    by = lambda policy: sorted((r for r in d["rows"] if r["policy"] == policy),
                               key=lambda r: r["rate"])
    st, co = by("static"), by("continuous")
    x = [r["rate"] for r in st]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.5), dpi=200)

    ax.plot(x, [r["offered_tokens_per_s"] for r in st], color=T.INK,
            linestyle=":", linewidth=1.1, label="offered")
    ax.plot(x, [r["tokens_per_s"] for r in st], color=T.AMBER, linestyle="-",
            marker="o", markersize=4, linewidth=T.LINE_WIDTH, label="static")
    ax.plot(x, [r["tokens_per_s"] for r in co], color=T.BLUE, linestyle="--",
            marker="s", markersize=4, linewidth=T.LINE_WIDTH, label="continuous")
    ax.set_xlabel("requests arriving a second")
    ax.set_ylabel("output tokens a second")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax.set_title("What the server delivers", loc="left", fontsize=10)
    T.style(ax)

    ax2.plot(x, [r["ttft_p99_ms"] for r in st], color=T.AMBER, linestyle="-",
             marker="o", markersize=4, linewidth=T.LINE_WIDTH, label="static")
    ax2.plot(x, [r["ttft_p99_ms"] for r in co], color=T.BLUE, linestyle="--",
             marker="s", markersize=4, linewidth=T.LINE_WIDTH, label="continuous")
    ax2.axhline(a["ttft_budget_ms"], color=T.INK, linewidth=0.9, alpha=0.5)
    ax2.annotate(f"the budget: {a['ttft_budget_ms']:,} ms",
                 (x[0], a["ttft_budget_ms"]), textcoords="offset points",
                 xytext=(2, 5), fontsize=7.2, color=T.INK)
    ax2.set_yscale("log")
    ax2.set_xlabel("requests arriving a second")
    ax2.set_ylabel("wait for the first token, p99 (ms, log scale)")
    ax2.legend(frameon=False, fontsize=7.6, loc="center right")
    ax2.set_title("and what the user waits to see it start", loc="left",
                  fontsize=10)
    T.style(ax2)

    save(fig, "ch17-load", d,
         alt=("Two panels against arrival rate. Left: output tokens a second. "
              f"The offered load rises to {st[-1]['offered_tokens_per_s']:,.0f} "
              f"tokens a second; static flattens near "
              f"{max(r['tokens_per_s'] for r in st):,.0f} and continuous "
              f"reaches {max(r['tokens_per_s'] for r in co):,.0f}. Right: the "
              "p99 wait for a first token on a log scale. Static is between "
              f"{min(r['ttft_p99_ms'] for r in st) / 1e3:.0f} and "
              f"{max(r['ttft_p99_ms'] for r in st) / 1e3:.0f} seconds "
              "throughout, far above the one-second budget; continuous stays "
              f"under {a['ttft_budget_ms']:,} milliseconds until "
              f"{max(r['rate'] for r in co if r['ttft_p99_ms'] <= a['ttft_budget_ms'])} "
              "requests a second."))


def fig_itl(d: dict) -> None:
    """The bill: your wait between tokens is now somebody else's prefill."""
    a = d["assumptions"]
    by = lambda policy: sorted((r for r in d["rows"] if r["policy"] == policy),
                               key=lambda r: r["rate"])
    st, co = by("static"), by("continuous")
    x = [r["rate"] for r in co]
    pure = d["pure_decode_itl_ms"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    ax.fill_between(x, [r["itl_p50_ms"] for r in co],
                    [r["itl_p99_ms"] for r in co], color=T.BLUE, alpha=0.10)
    ax.plot(x, [r["itl_p99_ms"] for r in co], color=T.BLUE, linestyle="--",
            marker="s", markersize=4, linewidth=T.LINE_WIDTH,
            label="continuous, p99")
    ax.plot(x, [r["itl_p50_ms"] for r in co], color=T.BLUE, linestyle="-",
            marker="s", markersize=3, linewidth=1.1, alpha=0.65,
            label="continuous, p50")
    ax.plot(x, [r["itl_p99_ms"] for r in st], color=T.AMBER, linestyle="-",
            marker="o", markersize=4, linewidth=T.LINE_WIDTH,
            label="static, p99")
    ax.axhline(pure, color=T.INK, linewidth=0.9, alpha=0.55)
    # Below the line: above it, and within a few points of it, runs the
    # static p99 curve.
    ax.annotate(f"the decode step alone: {pure:.1f} ms", (x[-1], pure),
                textcoords="offset points", xytext=(-4, -5), ha="right",
                va="top", fontsize=7.2, color=T.INK)
    ax.axhline(a["itl_budget_ms"], color=T.INK, linewidth=0.9, linestyle=":",
               alpha=0.7)
    ax.annotate(f"the budget: {a['itl_budget_ms']} ms", (x[-1], a["itl_budget_ms"]),
                textcoords="offset points", xytext=(-4, 5), ha="right",
                fontsize=7.2, color=T.INK)
    ax.set_xlabel("requests arriving a second")
    ax.set_ylabel("wait between tokens (ms)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    ax.set_title("Continuous batching puts other people's prefills "
                 "inside your reply", loc="left", fontsize=10.5)
    T.style(ax)

    save(fig, "ch17-itl", d,
         alt=("The wait between tokens against arrival rate. Continuous "
              f"batching's median rises gently from {co[0]['itl_p50_ms']:.1f} "
              f"to {co[-1]['itl_p50_ms']:.1f} milliseconds, close to the "
              f"{pure:.1f} milliseconds a decode step alone would take, but "
              f"its 99th percentile rises from {co[0]['itl_p99_ms']:.0f} to "
              f"{max(r['itl_p99_ms'] for r in co):.0f} milliseconds and "
              f"crosses the {a['itl_budget_ms']} millisecond budget. Static "
              "batching's 99th percentile stays between "
              f"{min(r['itl_p99_ms'] for r in st):.0f} and "
              f"{max(r['itl_p99_ms'] for r in st):.0f} milliseconds: a fixed "
              "batch has nothing to interrupt it."))


# --- Chapter 18: a prompt read a chunk at a time -----------------------

def fig_interference(d: dict) -> None:
    """One reply, and the holes that other people's prompts punch in it."""
    a = d["assumptions"]
    whole, chunked = d["interference"]["whole"], d["interference"]["chunked"]
    budget = a["itl_budget_ms"]

    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=200)
    x = range(1, len(whole["victim_gaps_ms"]) + 1)
    ax.plot(x, whole["victim_gaps_ms"], color=T.AMBER, linestyle="-",
            linewidth=T.LINE_WIDTH, marker="", label="prompt read whole")
    ax.plot(x, chunked["victim_gaps_ms"], color=T.BLUE, linestyle="--",
            linewidth=T.LINE_WIDTH, marker="",
            label=f"prompt read {chunked['budget']:,} tokens at a time")
    ax.axhline(budget, color=T.INK, linewidth=0.9, linestyle=":", alpha=0.7)
    ax.annotate(f"the budget: {budget} ms", (len(list(x)), budget),
                textcoords="offset points", xytext=(-4, 5), ha="right",
                fontsize=7.2, color=T.INK)
    for g, gap in enumerate(whole["victim_gaps_ms"], start=1):
        if gap > budget:
            ax.annotate("", xy=(g, gap), xytext=(g, gap * 1.06),
                        arrowprops=dict(arrowstyle="-", lw=0))
    ax.set_xlabel("token of this user's reply")
    ax.set_ylabel("wait before that token (ms)")
    # Upper left, and with headroom: the spikes are on the right of the
    # panel and the tallest of them reaches into a legend placed there.
    ax.set_ylim(0, max(whole["victim_gaps_ms"]) * 1.30)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"{whole['interlopers']} prompts of "
                 f"{whole['interloper_prompt']:,} tokens land during one reply",
                 loc="left", fontsize=10.5)
    T.style(ax)

    save(fig, "ch18-interference", d,
         alt=("The wait before each token of one user's reply while "
              f"{whole['interlopers']} long prompts arrive. Read whole, each "
              f"prompt stops the reply for about {whole['max_ms']:.0f} "
              "milliseconds, so the line has "
              f"{whole['over_budget']} spikes far above the "
              f"{budget} millisecond budget. Read "
              f"{chunked['budget']:,} tokens at a time, the same work raises "
              f"the wait only to {chunked['max_ms']:.1f} milliseconds and "
              "never crosses the budget: the interruptions are still there, "
              "but each is a fraction of the size."))


def fig_budget(d: dict) -> None:
    """The trade a token budget makes, and where it stops being a trade."""
    a = d["assumptions"]
    rows = [r for r in d["budgets_high"] if r["token_budget"]]
    base = next(r for r in d["budgets_high"] if not r["token_budget"])
    x = [r["token_budget"] for r in rows]

    itl = [r["itl_p99_ms"] for r in rows]
    ttft = [r["ttft_p99_ms"] for r in rows]
    tps = [r["tokens_per_s"] for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.6), dpi=200)

    # Left: two series, labelled where they run rather than in a legend
    # box. A box here would sit over the amber curve's steep descent,
    # which is the part of the picture that carries the argument.
    ax.plot(x, itl, color=T.BLUE, linestyle="--", marker="s", markersize=4,
            linewidth=T.LINE_WIDTH)
    ax.plot(x, ttft, color=T.AMBER, linestyle="-", marker="o", markersize=4,
            linewidth=T.LINE_WIDTH)
    ax.set_yscale("log")
    ax.set_ylim(min(itl) / 2.2, max(ttft) * 2.6)

    ax.axhline(a["ttft_budget_ms"], color=T.AMBER, linewidth=0.8,
               linestyle=":", alpha=0.8)
    # Right-aligned: the amber curve crosses this line on the left of
    # the panel, so a label anchored there is struck through by it.
    ax.annotate(f"first-token budget: {a['ttft_budget_ms']:,} ms",
                (x[-1], a["ttft_budget_ms"]), textcoords="offset points",
                xytext=(0, 5), ha="right", fontsize=6.8, color=T.INK)
    ax.axhline(a["itl_budget_ms"], color=T.BLUE, linewidth=0.8, linestyle=":",
               alpha=0.8)
    # Below the line and on the left, where both curves are far away.
    ax.annotate(f"between-token budget: {a['itl_budget_ms']} ms",
                (x[0], a["itl_budget_ms"]), textcoords="offset points",
                xytext=(0, -5), va="top", fontsize=6.8, color=T.INK)

    ax.annotate("to the first token, p99", (x[-1], ttft[-1]),
                textcoords="offset points", xytext=(0, 13), ha="right",
                fontsize=7.4, color=T.AMBER, fontweight="bold")
    ax.annotate("between tokens, p99", (x[-1], itl[-1]),
                textcoords="offset points", xytext=(0, -13), va="top",
                ha="right", fontsize=7.4, color=T.BLUE, fontweight="bold")

    ax.axvline(d["chosen_budget"], color=T.INK, linewidth=0.9, alpha=0.4)
    # High on the line, above where the amber curve has flattened and
    # clear of both budget labels. The band between the curves, which
    # looks empty, already holds the between-token budget label.
    ax.annotate(f"chosen: {d['chosen_budget']:,}",
                (d["chosen_budget"], a["ttft_budget_ms"] * 3),
                textcoords="offset points", xytext=(5, 0), va="center",
                fontsize=7.2, color=T.INK)

    ax.set_xlabel("token budget for one iteration")
    ax.set_ylabel("p99 latency (ms, log scale)")
    ax.set_title("The two promises pull opposite ways", loc="left", fontsize=10)
    _log2_ticks(ax, x)
    T.style(ax)

    # Right: one series and one reference line, both labelled in place.
    ax2.plot(x, tps, color=T.BLUE, linestyle="--", marker="s", markersize=4,
             linewidth=T.LINE_WIDTH)
    ax2.axhline(base["tokens_per_s"], color=T.AMBER, linewidth=1.1,
                linestyle="-", alpha=0.85)
    ax2.set_ylim(0, max(tps) * 1.28)
    # Well below the line, with a leader up to it: the blue curve sits
    # just above the line for most of the range, so there is no room to
    # label it in the gap between the two.
    ax2.annotate(f"prefill on its own iteration: {base['tokens_per_s']:,.0f}",
                 (x[-2], base["tokens_per_s"]),
                 xytext=(x[-1], base["tokens_per_s"] * 0.55),
                 ha="right", va="center", fontsize=7.2, color=T.INK,
                 arrowprops=dict(arrowstyle="-", linewidth=0.7,
                                 color=T.MUTED, shrinkA=3, shrinkB=1))
    ax2.annotate("stall-free, this budget", (x[-1], tps[-1]),
                 textcoords="offset points", xytext=(0, 12), ha="right",
                 fontsize=7.4, color=T.BLUE, fontweight="bold")
    ax2.axvline(d["chosen_budget"], color=T.INK, linewidth=0.9, alpha=0.4)
    ax2.annotate(f"chosen: {d['chosen_budget']:,}",
                 (d["chosen_budget"], max(tps) * 0.30),
                 textcoords="offset points", xytext=(5, 0), va="center",
                 fontsize=7.2, color=T.INK)

    ax2.set_xlabel("token budget for one iteration")
    ax2.set_ylabel("output tokens a second")
    ax2.set_title("and throughput barely notices", loc="left", fontsize=10)
    _log2_ticks(ax2, x)
    T.style(ax2)

    save(fig, "ch18-budget", d,
         alt=("Two panels against the per-iteration token budget, at "
              f"{a['rate_high']} requests a second. Left, on log axes: the p99 "
              f"wait between tokens rises from {rows[0]['itl_p99_ms']:.1f} to "
              f"{rows[-1]['itl_p99_ms']:.1f} milliseconds as the budget grows, "
              "while the p99 wait for a first token falls from "
              f"{rows[0]['ttft_p99_ms'] / 1e3:.1f} seconds to "
              f"{rows[-1]['ttft_p99_ms']:.0f} milliseconds. They cross a range "
              f"where both promises are kept, and {d['chosen_budget']:,} is "
              "the best point in it. Right: throughput over the same range, "
              f"between {min(r['tokens_per_s'] for r in rows):,.0f} and "
              f"{max(r['tokens_per_s'] for r in rows):,.0f} tokens a second "
              "except at the smallest budget, against the "
              f"{base['tokens_per_s']:,.0f} of the previous chapter's "
              "scheduler."))


def fig_policy(d: dict) -> None:
    """Who goes first, and how much it matters when anybody is waiting."""
    import numpy as np

    rows = d["policies"]
    names = [r["policy"] for r in rows if r["pool"] == "full"]
    pos = np.arange(len(names))
    width = 0.36

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.6), dpi=200,
                                  sharey=True)
    for ax_, pool, title in ((ax, "full", "The whole pool: nobody is waiting"),
                             (ax2, "squeezed",
                              "Squeezed: now there is a queue")):
        sel = [r for r in rows if r["pool"] == pool]
        ax_.bar(pos - width / 2, [r["slowdown_p50"] for r in sel], width,
                color=T.BLUE, label="p50")
        ax_.bar(pos + width / 2, [r["slowdown_p99"] for r in sel], width,
                color=T.AMBER, label="p99")
        for i, r in enumerate(sel):
            worst = r["slowdown_p99"]
            ax_.annotate(f"{worst:.0f}x" if worst >= 10 else f"{worst:.1f}x",
                         (i + width / 2, worst),
                         textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=7, color=T.INK)
        ax_.set_xticks(pos, [n.replace("-", "-\n") for n in names], fontsize=7.6)
        ax_.set_yscale("log")
        ax_.set_title(f"{title} ({sel[0]['pool_gb']:.0f} GB)"
                      if pool == "full" else
                      f"{title} ({sel[0]['pool_gb']:.1f} GB)",
                      loc="left", fontsize=9.5)
        T.style(ax_)
    # Once, after both panels are drawn. Setting only the bottom inside
    # the loop froze the shared axis at the left panel's autoscaled top,
    # and the right panel -- which is the one with the argument in it --
    # was drawn entirely above the visible area.
    #
    # 1x is "no slowdown at all", the floor by definition.
    ax.set_ylim(1, max(r["slowdown_p99"] for r in rows) * 2.2)
    ax.set_ylabel("slowdown against running alone (log scale)")
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")

    sq = {r["policy"]: r for r in rows if r["pool"] == "squeezed"}
    save(fig, "ch18-policy", d,
         alt=("Slowdown -- how much longer a request took than it would have "
              "taken alone -- under three queue orders, on a log scale, with "
              "the block pool whole and then squeezed. With the whole pool "
              "all three are identical near "
              f"{rows[0]['slowdown_p50']:.1f}x, because the server admits "
              "almost everyone on arrival and there is no queue to order. "
              "Squeezed, first-come-first-served reaches "
              f"{sq['fcfs']['slowdown_p99']:.0f}x at the 99th percentile, "
              "shortest-output-first "
              f"{sq['shortest-output']['slowdown_p99']:.0f}x, and "
              "longest-output-first "
              f"{sq['longest-output']['slowdown_p99']:.0f}x."))


# --- Chapter 19: the two phases on different machines ------------------

def fig_transfer(d: dict) -> None:
    """What a cache costs to move, against what it cost to make."""
    tr = d["transfer"]
    rows = tr["rows"]
    x = [r["tokens"] for r in rows]
    links = list(tr["links"])

    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    ax.plot(x, [r["prefill_s"] * 1e3 for r in rows], color=T.AMBER,
            linestyle="-", marker="o", markersize=4.5,
            linewidth=T.LINE_WIDTH + 0.4, label="making it (prefill)")
    shades = [T.SEQUENTIAL_STEPS[i] for i in (8, 6, 5, 3, 2)]
    for name, shade in zip(links, shades):
        ax.plot(x, [r["over"][name] * 1e3 for r in rows], color=shade,
                linestyle="--", marker="s", markersize=3.2, linewidth=1.4,
                label=f"moving it over {name}")
    ax.axhline(tr["decode_step_ms"], color=T.INK, linewidth=0.9,
               linestyle=":", alpha=0.7)
    # In the right margin. Six curves cross this line at six different
    # places; there is no room for its label inside the panel.
    ax.annotate(f"one decode step: {tr['decode_step_ms']:.1f} ms",
                xy=(1.0, tr["decode_step_ms"]),
                xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(4, 0), va="center",
                ha="left", fontsize=7, color=T.INK)
    # Rotated: 1,200 and 1,500 are a hair apart on a log axis and their
    # labels run into one word when set horizontally.
    _log2_ticks(ax, x)
    ax.set_yscale("log")
    ax.set_xlabel("tokens of context")
    ax.set_ylabel("milliseconds (log scale)")
    # Headroom for a two-row legend above every curve, including the
    # slowest link, which ends higher than anything else on the panel.
    lows = [r["over"][links[0]] * 1e3 for r in rows]
    highs = [r["over"][links[-1]] * 1e3 for r in rows]
    ax.set_ylim(min(lows) / 3.0, max(highs) * 9.0)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", ncol=2)
    ax.set_title("A cache is cheap to move only on the links that cost money",
                 loc="left", fontsize=10.5)
    T.style(ax)

    last = rows[-1]
    save(fig, "ch19-transfer", d,
         alt=("Time against context length, on log axes. Computing a "
              f"{last['tokens']:,}-token prefill takes "
              f"{last['prefill_s'] * 1e3:.0f} milliseconds; moving the "
              f"{last['bytes'] / 1e6:,.0f} MB of keys and values it produced "
              f"takes {last['over']['NVLink, same node'] * 1e3:.1f} "
              "milliseconds over NVLink inside a node, "
              f"{last['over']['InfiniBand NDR'] * 1e3:.0f} over InfiniBand, "
              f"and {last['over']['25 GbE'] * 1e3:.0f} over 25 gigabit "
              "Ethernet -- longer than the prefill itself. The Ethernet lines "
              "cross above the prefill line, which is where the move becomes "
              "the bottleneck; NVLink stays below one decode step at every "
              "length."))


def fig_split(d: dict) -> None:
    """One fleet, every division of it, against doing both phases everywhere."""
    a, co = d["assumptions"], d["colocated"]
    rows = d["splits"]
    x = [r["prefill_workers"] for r in rows]
    best = d["best_split"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.6), dpi=200)

    ax.plot(x, [r["tokens_per_s"] for r in rows], color=T.BLUE,
            linestyle="--", marker="s", markersize=4, linewidth=T.LINE_WIDTH,
            label="disaggregated, this split")
    ax.axhline(co["tokens_per_s"], color=T.AMBER, linewidth=1.3)
    # Above the line, not below it: the blue curve's peak comes within a
    # few per cent of this line, which is the whole point of the panel
    # and leaves no room underneath it for a label.
    ax.annotate(f"all {a['fleet']} doing both phases: "
                f"{co['tokens_per_s']:,.0f}",
                (x[0], co["tokens_per_s"]), textcoords="offset points",
                xytext=(2, 6), ha="left", fontsize=7.2, color=T.INK)
    ax.plot([best["prefill_workers"]], [best["tokens_per_s"]], marker="o",
            markersize=9, markerfacecolor="none", markeredgecolor=T.INK,
            linestyle="none")
    ax.set_xticks(x, [str(v) for v in x], fontsize=7.6)
    ax.set_ylim(0, co["tokens_per_s"] * 1.18)
    ax.set_xlabel(f"prefill machines (the other {a['fleet']} minus this decode)")
    ax.set_ylabel("output tokens a second")
    ax.legend(frameon=False, fontsize=7.4, loc="lower center")
    ax.set_title("The split is the design decision", loc="left", fontsize=10)
    T.style(ax)

    ax2.plot(x, [r["ttft_p99_ms"] for r in rows], color=T.AMBER, linestyle="-",
             marker="o", markersize=4, linewidth=T.LINE_WIDTH,
             label="to the first token, p99")
    ax2.plot(x, [r["itl_p99_ms"] for r in rows], color=T.BLUE, linestyle="--",
             marker="s", markersize=4, linewidth=T.LINE_WIDTH,
             label="between tokens, p99")
    ax2.axhline(a["ttft_budget_ms"], color=T.AMBER, linewidth=0.8,
                linestyle=":", alpha=0.8)
    ttft = [r["ttft_p99_ms"] for r in rows]
    itl = [r["itl_p99_ms"] for r in rows]
    ax2.set_yscale("log")
    ax2.set_ylim(min(itl) / 2.0, max(ttft) * 2.4)
    # The amber curve is a U, so it crosses this line on both sides of
    # the panel. Above the line the arms are further apart than below
    # it, which is the only place a label of this width fits between
    # them.
    mid = x[len(x) // 2]
    ax2.annotate(f"first-token budget: {a['ttft_budget_ms']:,} ms",
                 (mid, a["ttft_budget_ms"]), textcoords="offset points",
                 xytext=(0, 5), ha="center", va="bottom", fontsize=6.8,
                 color=T.INK)
    # Stopped below the budget label and the legend. A full-height rule
    # marks the best split where the curves are, and strikes through
    # everything written in the empty space above them.
    ax2.axvline(best["prefill_workers"], color=T.INK, linewidth=0.9,
                alpha=0.4, ymax=0.55)
    ax2.set_xticks(x, [str(v) for v in x], fontsize=7.6)
    ax2.set_xlabel("prefill machines")
    ax2.set_ylabel("p99 latency (ms, log scale)")
    # Inside the arms of the U, which is the one part of this panel with
    # nothing plotted in it.
    ax2.legend(frameon=False, fontsize=7.4, loc="upper center",
               bbox_to_anchor=(0.5, 0.97))
    ax2.set_title("and it moves both promises at once", loc="left", fontsize=10)
    T.style(ax2)

    worst = min(rows, key=lambda r: r["tokens_per_s"])
    save(fig, "ch19-split", d,
         alt=(f"Two panels against how many of the {a['fleet']} accelerators "
              "do prefill. Left: throughput, rising from "
              f"{worst['tokens_per_s']:,.0f} tokens a second at the worst "
              f"split to {best['tokens_per_s']:,.0f} at "
              f"{best['prefill_workers']} prefill machines, then falling "
              "again; the same fleet doing both phases everywhere delivers "
              f"{co['tokens_per_s']:,.0f}, above every split. Right: the p99 "
              "waits on a log scale. Too few prefill machines and the wait "
              "for a first token is tens of seconds; too many and the decode "
              "machines are overloaded, so the wait between tokens climbs "
              "and the first-token wait climbs again behind it."))


def fig_second_token(d: dict) -> None:
    """Where the network bill lands: one token, of one user."""
    import numpy as np

    rows = d["links"]
    co = d["colocated"]
    names = [r["link"] for r in rows] + ["colocated\n(no link)"]
    p50 = [r["second_token_p50_ms"] for r in rows] + [co["second_token_p50_ms"]]
    p99 = [r["second_token_p99_ms"] for r in rows] + [co["second_token_p99_ms"]]
    later = [r["itl_p99_ms"] for r in rows] + [co["itl_p99_ms"]]
    pos = np.arange(len(names))
    width = 0.28

    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=200)
    ax.bar(pos - width, p50, width, color=T.SEQUENTIAL_STEPS[5],
           label="wait for the second token, p50")
    ax.bar(pos, p99, width, color=T.BLUE,
           label="wait for the second token, p99")
    ax.bar(pos + width, later, width, color=T.AMBER,
           label="every later gap, p99")
    ax.axhline(d["assumptions"]["itl_budget_ms"], color=T.INK, linewidth=0.9,
               linestyle=":", alpha=0.8)
    ax.annotate(f"the between-token budget: {d['assumptions']['itl_budget_ms']} ms",
                (pos[-1] + width, d["assumptions"]["itl_budget_ms"]),
                textcoords="offset points", xytext=(4, 3), ha="right",
                fontsize=7, color=T.INK)
    ax.set_xticks(pos, names, fontsize=7.4)
    ax.set_ylabel("milliseconds")
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax.set_title("The whole cost of the network lands on one token",
                 loc="left", fontsize=10.5)
    T.style(ax)

    slow = rows[-1]
    save(fig, "ch19-second-token", d,
         alt=("Bars for each link. The wait for a user's second token is "
              f"{rows[0]['second_token_p50_ms']:.1f} milliseconds over "
              f"{rows[0]['link']} and {slow['second_token_p50_ms']:.0f} over "
              f"{slow['link']} at the median, reaching "
              f"{slow['second_token_p99_ms']:.0f} at the 99th percentile and "
              "crossing the budget. Every gap after the second is an "
              f"ordinary decode step -- between "
              f"{min(r['itl_p99_ms'] for r in rows):.1f} and "
              f"{max(r['itl_p99_ms'] for r in rows):.1f} milliseconds -- "
              "whatever the link. A colocated fleet has no link and no second-"
              "token penalty at all."))


# --- Chapter 20: the matrix the kernel does not build -------------------

def fig_score_matrix(d: dict) -> None:
    """A diagram: what is held in memory, one way and the other."""
    from matplotlib.patches import Rectangle

    ref = next(r for r in d["reference"]["rows"]
               if r["tokens"] == d["assumptions"]["prompt_tokens"])
    counted = d["traffic"][-1]
    n, tile = 12, 4

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.6), dpi=200)
    for ax in axes:
        # Row 0 at the top, so the grid reads like the matrix it is.
        ax.set_xlim(-2.8, n + 0.6); ax.set_ylim(n + 6.4, -0.6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])

    def cells(ax, items, face, edge="#FFFFFF", hatch=None):
        for (r, c) in items:
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=face, hatch=hatch,
                                   edgecolor=edge, linewidth=0.35))

    below = [(r, c) for r in range(n) for c in range(n) if c <= r]
    above = [(r, c) for r in range(n) for c in range(n) if c > r]

    # Left: the whole rectangle is computed, including the half that the
    # causal mask then discards.
    cells(axes[0], below, T.SEQUENTIAL(0.45))
    cells(axes[0], above, T.SEQUENTIAL(0.45), hatch="xxx")
    axes[0].set_title("Built in full, then read back", loc="left", fontsize=10)

    # Right: one block resident, the blocks behind it finished, the
    # blocks above the diagonal never computed at all.
    live = [(r, c) for r in range(tile, 2 * tile) for c in range(tile)]
    done = [(r, c) for (r, c) in below if (r, c) not in live and r < 2 * tile]
    todo = [(r, c) for (r, c) in below if r >= 2 * tile]
    cells(axes[1], done, T.SEQUENTIAL(0.5))
    cells(axes[1], todo, "#FFFFFF", edge=T.RULE)
    cells(axes[1], live, T.AMBER)
    cells(axes[1], above, "#ECEDF1", edge="#FFFFFF")
    for k in range(0, n, tile):
        axes[1].add_patch(Rectangle((0, k), n, tile, facecolor="none",
                                    edgecolor=T.RULE, linewidth=0.9))
    axes[1].set_title("Walked a block at a time", loc="left", fontsize=10)

    notes = [
        (axes[0], f"{ref['score_matrix_bytes'] / 1e6:,.0f} MB per layer at "
                  f"{ref['tokens']:,} tokens",
         "written once, read twice, written once more.",
         "Hatched: computed, then thrown away by the mask."),
        (axes[1], f"one block resident: "
                  f"{counted['tiled_largest_intermediate'] / 1e3:,.0f} KB",
         "Amber is resident now, blue is finished and gone, white is still "
         "to come.",
         f"Grey: never computed at all, "
         f"{counted['skipped_share'] * 100:.0f}% of the blocks."),
    ]
    for ax, headline, line1, line2 in notes:
        ax.text(n / 2, n + 1.0, "keys", ha="center", va="top",
                fontsize=7.6, color=T.MUTED)
        ax.text(n / 2, n + 2.4, headline, ha="center", va="top",
                fontsize=8.4, color=T.INK)
        ax.text(n / 2, n + 3.8, line1, ha="center", va="top",
                fontsize=7.2, color=T.MUTED)
        ax.text(n / 2, n + 5.0, line2, ha="center", va="top",
                fontsize=7.2, color=T.MUTED)
        ax.text(-1.8, n / 2, "queries", rotation=90, ha="center", va="center",
                fontsize=7.6, color=T.MUTED)

    save(fig, "ch20-score-matrix", d,
         alt=("Two grids of query positions against key positions. On the "
              "left the whole rectangle is filled, including the hatched "
              "half above the diagonal that the causal mask discards: that "
              "is the score matrix, "
              f"{ref['score_matrix_bytes'] / 1e6:,.0f} megabytes for one "
              f"layer at {ref['tokens']:,} tokens, written out and read "
              "back. On the right only one block is filled at a time, the "
              "blocks behind it are finished and gone, the blocks below are "
              "still to come, and the blocks above the diagonal are never "
              "computed at all -- "
              f"{counted['skipped_share'] * 100:.0f}% of them at the longest "
              "length measured. The largest thing the tiled kernel holds is "
              f"one block, {counted['tiled_largest_intermediate'] / 1e3:,.0f} "
              "kilobytes."))


def fig_traffic(d: dict) -> None:
    """How much of attention's traffic is the intermediate it need not build.

    Plotted as a ratio rather than as two curves of bytes: on log axes
    two curves that diverge slowly look parallel, and the finding is
    precisely that they diverge.
    """
    rows = d["reference"]["rows"]
    n = [r["tokens"] for r in rows]
    ratio = [r["ratio"] for r in rows]
    case_n = d["assumptions"]["prompt_tokens"]
    case = next(r for r in rows if r["tokens"] == case_n)

    resident = [r["ratio_block_resident"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=200)
    ax.fill_between(n, ratio, resident, color=T.BLUE, alpha=0.10, linewidth=0)
    ax.plot(n, resident, marker="s", markersize=T.MARKER_SIZE,
            linewidth=T.LINE_WIDTH, color=T.AMBER, linestyle="--",
            label="block kept in the scratchpad (a real kernel)")
    ax.plot(n, ratio, marker="o", markersize=T.MARKER_SIZE,
            linewidth=T.LINE_WIDTH, color=T.BLUE,
            label="block charged as traffic (this repository)")
    ax.axhline(1.0, color=T.MUTED, linewidth=0.9, linestyle=":")
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax.axvline(case_n, color=T.RULE, linewidth=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n, [f"{x:,}" for x in n])
    ax.set_ylim(0, max(resident) * 1.30)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("times less memory traffic, per layer")
    ax.set_title("The longer the prompt, the more of attention is an "
                 "intermediate\nnobody needs", loc="left", fontsize=10.5)
    T.style(ax)

    L.label_points(ax, [
        (case_n, case["ratio"],
         f"the case study's prompt\n{case['whole_bytes'] / 1e6:,.0f} MB "
         f"down to {case['tiled_bytes'] / 1e6:,.0f} MB"),
        (rows[-1]["tokens"], rows[-1]["ratio"],
         f"{rows[-1]['whole_bytes'] / 1e6:,.0f} MB down to "
         f"{rows[-1]['tiled_bytes'] / 1e6:,.0f} MB"),
    ], fontsize=7.4, color=T.INK)
    # No label at the short-prompt end. The two curves nearly meet there,
    # so anything written between them lands on one or on the dotted
    # "no saving" line below; and the curve starting low already says
    # it, as does the sentence in the chapter beside this figure.

    save(fig, "ch20-traffic", d,
         alt=("How many times less memory traffic the tiled kernel moves per "
              "layer, against prompt length, for the book's 8B model. The "
              f"ratio rises from {rows[0]['ratio']:.1f} at "
              f"{rows[0]['tokens']} tokens to {case['ratio']:.1f} at the "
              f"case study's {case_n:,} and {rows[-1]['ratio']:.1f} at "
              f"{rows[-1]['tokens']:,} when the blocks of scores are "
              "charged as traffic, and from "
              f"{rows[0]['ratio_block_resident']:.1f} to "
              f"{rows[-1]['ratio_block_resident']:.1f} when they are kept in "
              "the scratchpad as a real kernel keeps them. The saving is not "
              "a constant factor: "
              "it grows with the prompt, because the part being avoided "
              "grows with the square of the length while everything else "
              "grows linearly. In absolute terms the case study's prompt "
              f"goes from {case['whole_bytes'] / 1e6:,.0f} to "
              f"{case['tiled_bytes'] / 1e6:,.0f} megabytes a layer."))


def fig_tiles(d: dict) -> None:
    """The tile size is bounded from below by traffic and above by SRAM."""
    t = d["tiles"]
    rows = t["rows"]
    sizes = [r["tile"] for r in rows]
    limit = t["sram_bytes_per_sm"] / 1024

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), dpi=200, sharex=True)

    axes[0].plot(sizes, [r["bytes"] / 1e6 for r in rows], marker="o",
                 markersize=T.MARKER_SIZE, linewidth=T.LINE_WIDTH, color=T.BLUE)
    axes[0].set_ylim(0, max(r["bytes"] for r in rows) / 1e6 * 1.15)
    axes[0].set_ylabel("memory traffic (MB)")
    axes[0].set_title("Bigger tiles move less", loc="left", fontsize=9.6)

    fits = [r for r in rows if r["fits"]]
    over = [r for r in rows if not r["fits"]]
    # Bars at the same log positions as the left panel, so the two
    # panels read against one x axis rather than two different ones.
    for group, colour, hatch, label in ((fits, T.BLUE, None, "fits"),
                                        (over, T.AMBER, "///", "does not fit")):
        if not group:
            continue
        axes[1].bar([r["tile"] for r in group],
                    [r["sram_bytes_reference_model"] / 1024 for r in group],
                    width=[r["tile"] * 0.5 for r in group],
                    color=colour, hatch=hatch, edgecolor="#FFFFFF",
                    label=label)
    axes[1].axhline(limit, color=T.INK, linewidth=1.0, linestyle=":")
    axes[1].set_ylim(0, max(r["sram_bytes_reference_model"]
                            for r in rows) / 1024 * 1.25)
    axes[1].set_ylabel("fast memory needed (KB)")
    axes[1].set_title("and demand more room", loc="left", fontsize=9.6)
    axes[1].legend(frameon=False, fontsize=7.6, loc="upper left")

    for ax in axes:
        ax.set_xlabel("tile size (positions)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, [str(x) for x in sizes])
        T.style(ax)

    best = max(r["tile"] for r in fits)
    L.label_points(axes[1], [
        (sizes[0], limit, f"one multiprocessor: {limit:,.0f} KB"),
        (best, [r for r in fits if r["tile"] == best][0]
         ["sram_bytes_reference_model"] / 1024,
         f"largest tile that fits: {best}"),
    ], fontsize=7.4, color=T.INK)

    save(fig, "ch20-tiles", d,
         alt=("Two panels against tile size. Traffic falls steadily as the "
              f"tile grows, from {rows[0]['bytes'] / 1e6:,.0f} megabytes at "
              f"{rows[0]['tile']} positions to "
              f"{rows[-1]['bytes'] / 1e6:,.0f} at {rows[-1]['tile']}. The "
              "fast memory one tile needs grows with the square of it, and "
              f"at {over[0]['tile'] if over else 0} positions it passes the "
              f"{limit:,.0f} kilobytes a streaming multiprocessor has. The "
              f"largest tile that fits is {best}, which is where the tile "
              "size comes from: not tuning, but the size of the scratchpad."))


# --- Chapter 22: where the bits go -------------------------------------

def fig_bits(d: dict) -> None:
    """A diagram: the same 32 bits, divided up four ways."""
    from matplotlib.patches import Rectangle

    rows = [r for r in d["formats"]
            if r["name"] in ("float32", "float16", "bfloat16",
                             "float8 e4m3", "float8 e5m2")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    unit, height, gap = 1.0, 0.62, 0.42
    ax.set_xlim(-9.5, 33.5)
    ax.set_ylim(len(rows) * (height + gap), -0.5)
    ax.axis("off")

    kinds = [("sign", 1, "#9AA6B8"), ("exponent", None, T.AMBER),
             ("mantissa", None, T.BLUE)]
    for i, r in enumerate(rows):
        y = i * (height + gap)
        widths = [1, r["exponent_bits"], r["mantissa_bits"]]
        x = 0.0
        for (label, _, colour), w in zip(kinds, widths):
            ax.add_patch(Rectangle((x, y), w * unit, height, facecolor=colour,
                                   edgecolor="#FFFFFF", linewidth=0.8))
            if w >= 3:
                ax.text(x + w * unit / 2, y + height / 2, str(w),
                        ha="center", va="center", fontsize=7.4,
                        color="#FFFFFF", fontweight="bold")
            x += w * unit
        ax.text(-0.7, y + height / 2, r["name"], ha="right", va="center",
                fontsize=8.2, color=T.INK)
        ax.text(x + 0.5, y + height / 2,
                f"max {r['max_value']:.3g}", ha="left", va="center",
                fontsize=7.2, color=T.MUTED)

    for (label, _, colour), x in zip(kinds, (0.5, 5, 18)):
        ax.text(x, -0.9, label, ha="left", va="bottom", fontsize=7.6,
                color=colour, fontweight="bold")

    fp16 = next(r for r in rows if r["name"] == "float16")
    bf16 = next(r for r in rows if r["name"] == "bfloat16")
    save(fig, "ch22-bits", d,
         alt=("Bit layouts drawn to scale. Every format is a sign bit, "
              "then exponent bits, then mantissa bits. float16 and "
              "bfloat16 are both sixteen bits and divide them "
              f"differently: float16 keeps {fp16['exponent_bits']} for "
              f"the exponent and {fp16['mantissa_bits']} for the "
              f"mantissa, bfloat16 keeps {bf16['exponent_bits']} and "
              f"{bf16['mantissa_bits']}. The exponent sets how large a "
              f"number can be -- {fp16['max_value']:,.0f} against "
              f"{bf16['max_value']:.3g} -- and the mantissa sets how "
              "many digits survive. bfloat16 is float32's exponent with "
              "most of the mantissa thrown away."))


def fig_accumulation(d: dict) -> None:
    """What the wide accumulator inside a tensor core is worth."""
    rows = d["accumulation"]["rows"]
    formats = []
    for r in rows:
        if r["format"] not in formats:
            formats.append(r["format"])

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=200)
    styles = {formats[0]: (T.BLUE, "-", "o"), formats[1]: (T.AMBER, "--", "s")}
    ends = []
    for name in formats[:2]:
        rs = [r for r in rows if r["format"] == name]
        ks = [r["k"] for r in rs]
        colour, dash, mark = styles[name]
        ax.plot(ks, [r["narrow_accumulator"] for r in rs], marker=mark,
                markersize=T.MARKER_SIZE, linewidth=T.LINE_WIDTH,
                color=colour, linestyle=dash)
        ax.plot(ks, [r["wide_accumulator"] for r in rs], marker=mark,
                markersize=T.MARKER_SIZE, linewidth=T.LINE_WIDTH,
                color=colour, linestyle=":", alpha=0.75)
        ends.append((ks[-1], rs[-1]["narrow_accumulator"],
                     f"{name} in,\n{name} sum", colour))
        ends.append((ks[-1], rs[-1]["wide_accumulator"],
                     f"{name} in,\nfloat32 sum", colour))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ks = sorted({r["k"] for r in rows})
    ax.set_xticks(ks, [f"{k:,}" for k in ks])
    ax.set_xlabel("length of the dot product")
    ax.set_ylabel("error, relative to the result's own size")
    ax.set_title("What a tensor core's wide accumulator is worth",
                 loc="left", fontsize=10.5)
    lo = min(r["wide_accumulator"] for r in rows if r["format"] in formats[:2])
    hi = max(r["narrow_accumulator"] for r in rows if r["format"] in formats[:2])
    ax.set_ylim(lo / 3, hi * 3)
    T.style(ax)
    # Labelled at the right-hand end rather than in a legend: four
    # series on one panel leaves a legend nowhere to sit that is not on
    # top of one of them.
    ax.set_xlim(ks[0] * 0.85, ks[-1] * 5.5)
    for x, y, text, colour in ends:
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(9, -2),
                    fontsize=7.2, color=colour, va="center", ha="left")

    worst = max(rows, key=lambda r: r["ratio"])
    save(fig, "ch22-accumulation", d,
         alt=("Error against the length of the dot product, on log axes. "
              "The dotted lines, where the running total is kept in "
              "float32 as the hardware keeps it, are flat: the error is "
              "set by rounding the inputs and does not grow. The solid "
              "lines, where the running total is kept in the small "
              "format too, climb with every doubling, reaching "
              f"{worst['ratio']:.0f} times worse at "
              f"{worst['k']:,} terms in {worst['format']}. Keeping the "
              "inputs small is cheap; keeping the sum small is not."))


def fig_range(d: dict) -> None:
    """Where float16 runs out of room and bfloat16 does not."""
    r = d["range"]
    rows = r["rows"]
    scales = [x["input_scale"] for x in rows]
    exact = [abs(x["exact"]) for x in rows]
    biggest = [x["largest_product"] for x in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=200)
    ax.plot(scales, biggest, marker="o", markersize=T.MARKER_SIZE,
            linewidth=T.LINE_WIDTH, color=T.BLUE,
            label="largest single product")
    ax.plot(scales, exact, marker="s", markersize=T.MARKER_SIZE,
            linewidth=T.LINE_WIDTH, color=T.AMBER, linestyle="--",
            label="the answer itself")
    ax.axhline(r["fp16_max"], color=T.INK, linewidth=1.0, linestyle=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(scales, [str(s) for s in scales])
    ax.set_xlabel("scale of the numbers going in")
    ax.set_ylabel("magnitude")
    ax.set_title("float16 does not run out of precision first.\nIt runs "
                 "out of room", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    T.style(ax)

    first = r["first_scale_fp16_fails"]
    L.label_points(ax, [
        (scales[0], r["fp16_max"], f"float16 stops here: {r['fp16_max']:,.0f}"),
    ], fontsize=7.4, color=T.INK)

    save(fig, "ch22-range", d,
         alt=("The size of the numbers inside one dot product across "
              f"{r['k']:,} terms, against the scale of the inputs, on "
              "log axes. A dotted line marks float16's largest "
              f"representable value, {r['fp16_max']:,.0f}. Both the "
              "individual products and the answer pass it once the "
              f"inputs reach a scale of about {first}, and a float16 "
              "accumulator returns infinity from there on. bfloat16, "
              "which has float32's exponent, never overflows anywhere "
              "in this sweep -- it just answers coarsely."))


# --- Chapter 24: weights as integers -----------------------------------

def fig_grid(d: dict) -> None:
    """A diagram: the values a scheme can represent, and where the
    weights actually fall."""
    w = d["worked"]
    values = w["values"]
    scale, qmin, qmax = w["scale"], w["qmin"], w["qmax"]
    grid = [c * scale for c in range(qmin, qmax + 1)]

    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=200)
    span = max(grid) - min(grid)
    ax.set_xlim(min(grid) - span * 0.30, max(grid) + span * 0.06)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")

    ax.axhline(0, color=T.RULE, linewidth=1.0)
    for g in grid:
        ax.plot([g, g], [-0.16, 0.16], color=T.RULE, linewidth=0.9)
    for code, g in zip(range(qmin, qmax + 1), grid):
        if code % 2 == 0:
            ax.text(g, -0.42, str(code), ha="center", va="top",
                    fontsize=6.4, color=T.MUTED)
    ax.text(0, -0.62, "integer code", ha="center", va="top",
            fontsize=7.0, color=T.MUTED, style="italic")
    ax.text(0, -0.95, f"the {qmax - qmin + 1} values {w['bits']} bits can hold, "
            f"a step of {scale:.4f} apart",
            ha="center", va="top", fontsize=7.8, color=T.MUTED)

    for v, code, back in zip(values, w["codes"], w["recovered"]):
        ax.plot([v, v], [0.2, 0.62], color=T.AMBER, linewidth=1.1)
        ax.plot([v], [0.62], marker="o", markersize=3.4, color=T.AMBER)
        ax.plot([back], [0.2], marker="v", markersize=3.6, color=T.BLUE)
    left = min(grid) - span * 0.035
    ax.text(left, 0.62, "weights", fontsize=7.8, color=T.AMBER,
            ha="right", va="center", fontweight="bold")
    ax.text(left, 0.20, "where each lands", fontsize=7.8,
            color=T.BLUE, ha="right", va="center", fontweight="bold")

    save(fig, "ch24-grid", d,
         alt=(f"A number line marked with the {qmax - qmin + 1} values "
              f"{w['bits']}-bit symmetric quantization can represent, "
              f"evenly spaced {scale:.4f} apart. Eight example weights sit "
              "above it and each drops to the nearest mark. The largest "
              f"error is {w['largest_error']:.4f}, which is at most half a "
              f"step, {w['half_a_step']:.4f}. Every weight in a group shares "
              "this one grid, so the grid is set by the largest value among "
              "them."))


def fig_outlier(d: dict) -> None:
    """What one large weight costs everything sharing its scale."""
    rows = d["outliers"]["rows"]
    factors = [r["outlier_factor"] for r in rows]
    keys = [k for k in rows[0] if k.startswith("int4")]
    styles = [(T.AMBER, "-", "o"), (T.BLUE, "--", "s"), (T.BLUE, ":", "^")]

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=200)
    ends = []
    for key, (colour, dash, mark) in zip(keys, styles):
        ys = [r[key] for r in rows]
        ax.plot(factors, ys, marker=mark, markersize=T.MARKER_SIZE,
                linewidth=T.LINE_WIDTH, color=colour, linestyle=dash)
        ends.append((factors[-1], ys[-1], key.split(", ")[-1], colour))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(factors, [f"{f}x" for f in factors])
    ax.set_xlim(factors[0] * 0.8, factors[-1] * 3.2)
    ax.set_xlabel("one weight, this many times the largest of the others")
    ax.set_ylabel("error on all the other weights")
    ax.set_title("A scale is set by the largest value sharing it",
                 loc="left", fontsize=10.5)
    T.style(ax)
    for x, y, text, colour in ends:
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(9, 0),
                    fontsize=7.4, color=colour, va="center", ha="left")

    first, last = rows[0], rows[-1]
    tensor = keys[0]
    save(fig, "ch24-outlier", d,
         alt=("Error on the ordinary weights, against the size of a single "
              "outlier placed among them, on log axes. With one scale for "
              "the whole tensor the error rises from "
              f"{first[tensor]:.2e} to {last[tensor]:.2e} -- the ordinary "
              "weights are destroyed, because the grid has been stretched to "
              "reach a value none of them is near. With a scale per group of "
              f"32 the error barely moves, from {first[keys[2]]:.2e} to "
              f"{last[keys[2]]:.2e}, because the outlier stretches only its "
              "own group's grid. This is the argument for group-wise "
              "quantization, and it is why the group size is the number that "
              "matters."))


def fig_margin(d: dict) -> None:
    """When quantization error changes an answer, and when it does not."""
    a = d["answers"]
    rows = a["rows"]
    names = [r["scheme"].replace("int", "int ").replace(", symmetric", "")
             .replace(", asymmetric", " (asym)") for r in rows]
    shift = [r["median_shift"] for r in rows]
    changed = [r["positions_that_changed"] * 100 for r in rows]
    pos = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
    colours = [T.BLUE if r["bits"] == 8 else T.AMBER
               for r in _with_bits(rows)]
    ax.barh(pos, shift, height=0.62, color=colours)
    ax.axvline(a["margin_median"], color=T.INK, linewidth=1.1, linestyle=":")
    ax.set_yticks(pos, names, fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("how far the logits moved (median over positions);\n"
                  "the percentage is next tokens that changed")
    ax.set_title("Quantization changes an answer only where it moves the\n"
                 "logits further than the model's own margin",
                 loc="left", fontsize=10.5)
    T.style(ax, hide_left=True)
    # Short, because the int8 bars end well left of the margin line and
    # a longer label would have to cross it to be read.
    for y, (s, c) in enumerate(zip(shift, changed)):
        ax.text(s * 1.12, y, f"{c:.0f}% changed",
                va="center", ha="left", fontsize=6.9, color=T.MUTED)
    ax.set_xlim(min(shift) / 2.2, max(shift) * 9)
    L.label_points(ax, [(a["margin_median"], len(rows) - 0.6,
                         f"the model's median margin: {a['margin_median']:.2f}")],
                   fontsize=7.4, color=T.INK)

    best8 = min((r for r in rows if "int8" in r["scheme"]),
                key=lambda r: r["median_shift"])
    worst4 = max(rows, key=lambda r: r["median_shift"])
    save(fig, "ch24-margin", d,
         alt=("How far each scheme moves the logits, against the gap between "
              "the model's first and second choice. Eight-bit schemes move "
              f"them about {best8['median_shift']:.2f}, below the median "
              f"margin of {a['margin_median']:.2f}, and change "
              f"{best8['positions_that_changed']*100:.0f}% of next tokens. "
              f"Four-bit schemes move them {worst4['median_shift']:.1f} and "
              f"change up to {max(r['positions_that_changed'] for r in rows)*100:.0f}%. "
              "The error is not what decides; the error relative to how "
              "confident the model was is."))


def _with_bits(rows: list) -> list:
    """Attach the bit width, which the scheme name carries as text."""
    return [{**r, "bits": 8 if "int8" in r["scheme"] else 4} for r in rows]


# --- Chapter 29: guess, then check -------------------------------------

def fig_rule(d: dict) -> None:
    """A diagram: which part of a guess is kept and which is corrected."""
    e = d["exactness"]
    target, draft = e["target"], e["draft"]
    n = len(target)
    x = list(range(n))
    keep = [min(t, q) for t, q in zip(target, draft)]
    short = [max(t - q, 0.0) for t, q in zip(target, draft)]
    waste = [max(q - t, 0.0) for t, q in zip(target, draft)]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), dpi=200, sharey=True)
    # No separate bar for the draft's own distribution: the two below
    # stack to exactly it, so drawing it underneath would be invisible.
    axes[0].bar(x, keep, width=0.66, color=T.BLUE, label="accepted")
    axes[0].bar(x, waste, width=0.66, bottom=keep, color=T.AMBER,
                hatch="///", edgecolor="#FFFFFF", label="rejected")
    axes[0].set_title("What the draft proposes", loc="left", fontsize=9.8)
    axes[0].legend(frameon=False, fontsize=7.0, loc="upper right")

    axes[1].bar(x, keep, width=0.66, color=T.BLUE, label="kept from the draft")
    axes[1].bar(x, short, width=0.66, bottom=keep, color=T.AMBER,
                label="drawn from the shortfall")
    axes[1].plot(x, target, linestyle="none", marker="_", markersize=13,
                 markeredgewidth=1.6, color=T.INK)
    axes[1].set_title("What comes out, against the target (dashes)",
                      loc="left", fontsize=9.8)
    axes[1].legend(frameon=False, fontsize=7.0, loc="upper right")

    for ax in axes:
        ax.set_xticks(x, [str(i) for i in x], fontsize=7)
        ax.set_xlabel("token")
        T.style(ax)
    axes[0].set_ylabel("probability")

    save(fig, "ch29-rule", d,
         alt=("Two panels over eight tokens. On the left, the draft's own "
              "distribution, drawn as two stacked parts: the part that "
              "overlaps the target -- "
              "which is accepted -- and the part where the draft asks for "
              "more than the target wants, which is rejected. On the right, "
              "what comes out: the accepted part, plus a draw from the "
              "shortfall where the target wants more than the draft offered. "
              "The two stack to exactly the target's distribution, marked by "
              "dashes. Nothing is approximated; the correction puts back "
              "precisely what the draft left out."))


def fig_exactness(d: dict) -> None:
    """The distribution that comes out, sampled."""
    e = d["exactness"]
    n = len(e["target"])
    x = list(range(n))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=200)
    ax.bar([i - width / 2 for i in x], e["target"], width=width,
           color=T.INK, label="what the expensive model wanted")
    ax.bar([i + width / 2 for i in x], e["kept"], width=width,
           color=T.BLUE, label=f"what came out ({e['draws']:,} draws)")
    ax.plot(x, e["draft"], linestyle="none", marker="_", markersize=13,
            markeredgewidth=1.6, color=T.AMBER,
            label="what the draft proposed from")
    ax.set_xticks(x, [str(i) for i in x], fontsize=7.4)
    ax.set_xlabel("token")
    ax.set_ylabel("probability")
    ax.set_title("A wrong draft, corrected exactly", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=7.6, loc="upper right")
    T.style(ax)

    save(fig, "ch29-exactness", d,
         alt=("Three series over eight tokens: what the expensive model "
              "wanted, what speculative decoding actually produced over "
              f"{e['draws']:,} draws, and the deliberately wrong "
              "distribution the draft proposed from. The first two are "
              f"indistinguishable -- the largest gap is "
              f"{e['largest_deviation']:.5f}, which is "
              f"{e['largest_in_standard_errors']:.2f} standard errors of the "
              "sampling itself. The draft's own distribution is nothing like "
              f"either: it differs from the target by "
              f"{e['draft_total_variation']:.2f} in total variation, against "
              f"{e['total_variation']:.5f} for the output. A bad draft costs "
              "speed, never correctness."))


def fig_speculative_speedup(d: dict) -> None:
    """How many guesses are worth making."""
    curves = [c for c in d["speedups"]["curves"] if c["alpha"] == 0.9]
    grid = {(g["draft"], g["alpha"]): g for g in d["speedups"]["grid"]}

    fig, ax = plt.subplots(figsize=(6.9, 4.3), dpi=200)
    shades = [0.45, 0.62, 0.78, 0.95]
    ends = []
    missed = []
    for c, shade in zip(curves, shades):
        colour = T.SEQUENTIAL(shade)
        ax.plot(c["ks"], c["speedups"], linewidth=T.LINE_WIDTH, color=colour)
        g = grid[(c["draft"], 0.9)]
        if g["best_k"] in c["ks"]:
            ax.plot([g["best_k"]], [g["speedup"]], marker="o", markersize=7.5,
                    markerfacecolor="#FFFFFF", markeredgecolor=colour,
                    markeredgewidth=1.8, zorder=5)
        else:
            missed.append((c["draft"], g["best_k"]))
        ends.append((c["ks"][-1], c["speedups"][-1], c["draft"], colour))
    if missed:
        raise RuntimeError(
            "these curves' best k is off the chart, so the figure would "
            f"show an optimum that is not there: {missed}. Widen KS.")
    ax.axhline(1.0, color=T.MUTED, linewidth=0.9, linestyle=":")
    ticks = [k for k in curves[0]["ks"] if k % 4 == 0 or k == 1]
    ax.set_xticks(ticks, [str(k) for k in ticks])
    ax.set_xlim(0, curves[0]["ks"][-1] * 1.42)
    ax.set_xlabel("tokens guessed per round")
    ax.set_ylabel("times faster")
    ax.set_title("What a draft costs decides how much to guess\n"
                 "(each guess accepted 90% of the time)",
                 loc="left", fontsize=10.5)
    T.style(ax)
    for x, y, text, colour in ends:
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(9, 0),
                    fontsize=7.0, color=colour, va="center", ha="left")

    cheap = max(d["speedups"]["grid"],
                key=lambda g: g["speedup"] if g["alpha"] == 0.9 else 0)
    save(fig, "ch29-speedup", d,
         alt=("Speedup against how many tokens are guessed per round, on a "
              "for four drafts of different cost, all with 90% "
              "of guesses accepted. Every curve rises, peaks and falls: each "
              "extra guess is paid for whether or not it survives, while the "
              "chance it survives falls geometrically. The circled point on "
              "each is the best number of guesses, and it is larger the "
              f"cheaper the draft -- {cheap['best_k']} guesses and "
              f"{cheap['speedup']:.1f} times faster for the cheapest here. A "
              "draft costing half a target step is barely worth running."))


# --- Chapter 41: how much hardware, and how hot ------------------------

def fig_little(d: dict) -> None:
    """Little's law on a server it was not derived for."""
    rows = d["theory"]["rows"]
    sweep = {r["rate"]: r for r in d["sweep"]}
    steady = [r for r in rows if r["steady"]]
    broken = [r for r in rows if not r["steady"]]

    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=200)
    top = max(sweep[r["rate"]]["lambda_times_w"] for r in rows) * 1.12
    ax.plot([0, top], [0, top], color=T.MUTED, linewidth=0.9, linestyle=":")
    for group, colour, mark, label in (
            (steady, T.BLUE, "o", "in steady state"),
            (broken, T.AMBER, "s", "still draining when the traffic stopped")):
        if not group:
            continue
        ax.plot([sweep[r["rate"]]["in_system"] for r in group],
                [sweep[r["rate"]]["lambda_times_w"] for r in group],
                linestyle="none", marker=mark, markersize=6,
                color=colour, label=label)
    ax.set_xlabel("requests in the system, counted")
    ax.set_ylabel("arrival rate x time in the system")
    ax.set_title("Little's law holds until the system stops being steady",
                 loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    T.style(ax)

    t = d["theory"]
    save(fig, "ch41-little", d,
         alt=("The number of requests in the system plotted against the "
              "arrival rate times the average time each spends there. "
              "Little's law says these are equal, and the dotted line is "
              "where equal would be. Every point taken while the server was "
              "in steady state sits on it, to within "
              f"{t['largest_little_gap_while_steady'] * 100:.1f}%. The "
              "points that leave the line are the offered loads at which "
              "the server was still working after the traffic stopped -- "
              f"from {t['first_unsteady_rate']} requests a second -- where "
              "the law's own assumption no longer holds. It is a check on "
              "the measurement, not a property of the server."))


def fig_utilization(d: dict) -> None:
    """What a batching server does with load, against what theory says."""
    rows = [r for r in d["theory"]["rows"] if r["classical_slowdown"]]

    fig, ax = plt.subplots(figsize=(6.9, 4.3), dpi=200)
    u = [r["utilization"] * 100 for r in rows]
    ax.plot(u, [r["classical_slowdown"] for r in rows], marker="s",
            markersize=T.MARKER_SIZE, linewidth=T.LINE_WIDTH, color=T.AMBER,
            linestyle="--", label="what a classical queue would do")
    ax.plot(u, [r["measured_slowdown"] for r in rows], marker="o",
            markersize=T.MARKER_SIZE, linewidth=T.LINE_WIDTH, color=T.BLUE,
            label="what this server does")
    ax.set_yscale("log")
    ax.set_xlabel("how full the machine is (% of its capacity)")
    ax.set_ylabel("times slower than a request with the machine to itself")
    ax.set_title("A server that batches does not queue",
                 loc="left", fontsize=10.5)
    T.style(ax)

    ax.set_xlim(0, 104)
    # A legend, not end labels: the classical curve turns vertical at
    # the right and the two converge at the left, so there is nowhere
    # beside either line to write.
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    at90 = min(rows, key=lambda r: abs(r["utilization"] - 0.90))
    save(fig, "ch41-utilization", d,
         alt=("Slowdown against how full the machine is, on a log y axis. "
              "The dashed line is the textbook single-server queue, where "
              "time in the system grows as one over one minus utilization. "
              "The solid line is measured. At "
              f"{at90['utilization'] * 100:.0f}% of capacity the classical "
              f"model predicts {at90['classical_slowdown']:.1f} times the "
              f"service time and the server delivers "
              f"{at90['measured_slowdown']:.2f} -- an over-prediction of "
              f"{at90['over_prediction']:.1f} times. Requests do not wait in "
              "a line for this server; they join the batch and everyone "
              "slows down together, which degrades far more gently than "
              "waiting does."))


def fig_sizing(d: dict) -> None:
    """Three ways to size the same fleet."""
    s_ = d["sizing"]
    labels = {
        "throughput_only": "throughput alone\n(ignores the promise)",
        "classical_rule_of_thumb": "the 70% rule\n(classical queueing)",
        "measured_promise": "measured against\nthe promise",
    }
    keys = list(labels)
    counts = [s_["machines"][k] for k in keys]
    costs = [s_["usd_per_hour"][k] for k in keys]
    colours = [T.RULE, T.AMBER, T.BLUE]

    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=200)
    pos = list(range(len(keys)))
    ax.bar(pos, counts, width=0.58, color=colours)
    ax.set_xticks(pos, [labels[k] for k in keys], fontsize=7.8)
    ax.set_ylabel("machines")
    ax.set_ylim(0, max(counts) * 1.32)
    ax.set_title(f"Sizing for {s_['demand_requests_per_s']} requests a "
                 "second, three ways", loc="left", fontsize=10.5)
    T.style(ax)
    for x, (n, c) in enumerate(zip(counts, costs)):
        ax.text(x, n + max(counts) * 0.04, f"{n}\n${c:,.2f}/hour",
                ha="center", va="bottom", fontsize=8.0, color=T.INK)

    save(fig, "ch41-sizing", d,
         alt=("Three answers to how many machines the case study needs. "
              f"Throughput alone says {counts[0]}, which meets no promise. "
              f"The classical rule of never running above 70% says "
              f"{counts[1]}. Measuring the load at which both promises "
              f"actually still hold -- "
              f"{s_['highest_rate_meeting_both_promises']} requests a "
              f"machine, {s_['utilization_there'] * 100:.0f}% of capacity -- "
              f"says {counts[2]}, at ${costs[2]:,.2f} an hour. The rule of "
              f"thumb buys {counts[1] - counts[2]} machines nobody needs."))


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
            "ch15": [fig_prefixtree, fig_prefill, fig_hitrate],
            "ch16": [fig_matmul, fig_batch_tradeoff, fig_static_batch],
            "ch17": [fig_scheduler_timeline, fig_load, fig_itl],
            "ch18": [fig_interference, fig_budget, fig_policy],
            "ch19": [fig_transfer, fig_split, fig_second_token],
            "ch20": [fig_score_matrix, fig_traffic, fig_tiles],
            "ch22": [fig_bits, fig_accumulation, fig_range],
            "ch24": [fig_grid, fig_outlier, fig_margin],
            "ch29": [fig_rule, fig_exactness, fig_speculative_speedup],
            "ch41": [fig_little, fig_utilization, fig_sizing]}


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
    L.REPORT.write()
    if L.REPORT.problems:
        print(f"\nLEGIBILITY: {len(L.REPORT.problems)} overlap(s) "
              f"across {L.REPORT.figures} figures:")
        for problem in L.REPORT.problems:
            print(f"  - {problem}")
    else:
        print(f"\nlegibility: no overlapping labels in "
              f"{L.REPORT.figures} figures")


if __name__ == "__main__":
    main()
