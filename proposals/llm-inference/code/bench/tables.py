"""Regenerate the chapter's tables from results/*.json into tables/*.md.

Same rule as figures: the manuscript includes these files, so no number
in a table is transcribed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS, TABLES = Path("results"), Path("tables")


def head_to_head(d: dict) -> str:
    h, e = d["head_to_head"], d["experiment"]
    n, c = h["naive"], h["cached"]
    ms = lambda x: f"{x * 1e3:,.2f} ms"
    rows = [
        ("Time to first token (prefill)", ms(n["ttft_s"]["median"]), ms(c["ttft_s"]["median"]),
         f"{n['ttft_s']['median'] / c['ttft_s']['median']:.2f}x"),
        ("Decode step, p50", ms(n["decode_step_s"]["p50"]), ms(c["decode_step_s"]["p50"]),
         f"{h['speedup_decode_p50']:.0f}x"),
        ("Decode step, p99", ms(n["decode_step_s"]["p99"]), ms(c["decode_step_s"]["p99"]),
         f"{n['decode_step_s']['p99'] / c['decode_step_s']['p99']:.0f}x"),
        ("Decode throughput", f"{n['decode_tok_per_s']['median']:,.0f} tok/s",
         f"{c['decode_tok_per_s']['median']:,.0f} tok/s",
         f"{c['decode_tok_per_s']['median'] / n['decode_tok_per_s']['median']:.0f}x"),
        (f"Total, {e['prompt_len']} prompt + {e['n_new']} generated",
         f"{n['total_s']['median']:.3f} s", f"{c['total_s']['median']:.3f} s",
         f"{h['speedup_total']:.0f}x"),
    ]
    out = ["| Measurement | Without a cache | With a KV cache | Ratio |",
           "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c_} | **{e_}** |" for a, b, c_, e_ in rows]
    return "\n".join(out)


def scaling(d: dict) -> str:
    out = ["| Tokens generated | Without a cache | With a cache | Measured speedup | Speedup predicted by FLOPs |",
           "|---|---|---|---|---|"]
    for r in d["sweep"]:
        flag = " \\*" if r["naive_noisy"] or r["cached_noisy"] else ""
        out.append(f"| {r['n_new']} | {r['naive_s']:.3f} s{flag} | {r['cached_s']:.3f} s | "
                   f"**{r['speedup']:.0f}x** | {r['flop_ratio']:.0f}x |")
    out.append("")
    out.append("\\* run-to-run spread exceeded 5%; see the note on the measuring machine.")
    return "\n".join(out)


def memory(d: dict) -> str:
    t, r = d["memory"]["tinyserve"], d["memory"]["reference_8b"]
    return "\n".join([
        "| | `tinyserve` (this chapter) | " + r["name"] + " |",
        "|---|---|---|",
        f"| Layers | {d['model']['config']['n_layers']} | {r['n_layers']} |",
        f"| KV heads | {d['model']['config']['n_kv_heads']} | {r['n_kv_heads']} |",
        f"| Head dimension | {d['model']['head_dim']} | {r['head_dim']} |",
        f"| Bytes per element | 4 (fp32) | {r['bytes_per_element']} (bf16) |",
        f"| **KV cache per token** | **{t['bytes_per_token']:,} B** | "
        f"**{r['kib_per_token']:.0f} KiB** |",
        f"| Per 8,192-token sequence | {t['bytes_per_token'] * 8192 / 1024**2:.0f} MiB | "
        f"{r['gib_per_8k_sequence']:.1f} GiB |",
    ])


def ch01_cost(d: dict) -> str:
    out = ["| Sequences at once | Throughput | Cost per 1M output tokens | Arithmetic used | Limited by |",
           "|---|---|---|---|---|"]
    for r in d["batches"]:
        out.append(f"| {r['batch']} | {r['tokens_per_s']:,.0f} tok/s | "
                   f"**${r['usd_per_m_tokens']:.3f}** | "
                   f"{r['flop_utilization'] * 100:.1f}% of peak | {r['bound_by']} |")
    a, api = d["assumptions"], d["api_reference"]
    out += ["", f"A model, not a benchmark: {a['gpu']}, "
                f"${a['gpu_usd_per_hour']}/GPU-hour, {a['params'] / 1e9:.0f}B parameters "
                f"in bf16, {a['seq_len']:,}-token sequences. For comparison, "
                f"{api['model']} is published at ${api['output_usd_per_m']:.2f} per "
                f"million output tokens."]
    return "\n".join(out)


def ch02_shapes(d: dict) -> str:
    sh, cfg, v = d["shapes"], d["model"]["config"], d["vocabulary"]
    n = sh["tokens"][0]
    rows = [
        ("The prompt, as text", '"' + " ".join(d["prompt"]["words"]) + '"', "-"),
        ("As token numbers", str(d["prompt"]["tokens"]), f"{n}"),
        ("After the lookup table", "a vector per token",
         f"{n} x {cfg['d_model']}"),
        (f"Through each of the {cfg['n_layers']} layers", "same shape in, same shape out",
         f"{n} x {cfg['d_model']}"),
        ("Keys and values kept per layer", "what attention reads later",
         f"{sh['keys_per_layer'][0]} x {n} x {d['model']['head_dim']}"),
        ("Scores for the next token", f"one per word in the vocabulary",
         f"{n} x {v['size']}"),
        ("Used to write a token", "only the last row matters", f"{v['size']}"),
    ]
    out = ["| Stage | What it is | Shape |", "|---|---|---|"]
    out += [f"| {a} | {b} | `{c}` |" for a, b, c in rows]
    return "\n".join(out)


def ch02_cost(d: dict) -> str:
    t, r = d["cost"]["tiny"], d["cost"]["reference_8b"]
    tp, tf = d["model"]["params"], 2 * d["model"]["params"]
    return "\n".join([
        "| Per token written | This chapter's model | A production 8B model |",
        "|---|---|---|",
        f"| Parameters | {tp:,} | {r['params'] / 1e9:.0f} billion |",
        f"| Weights read | {t['weight_bytes_read_per_token'] / 1e6:.1f} MB (fp32) | "
        f"{r['weight_bytes_read_per_token'] / 1e9:.0f} GB (bf16) |",
        f"| Arithmetic | {tf / 1e6:.1f} million operations | "
        f"{r['flops_per_token'] / 1e9:.0f} billion operations |",
        f"| Keys and values stored | {t['kv_bytes_per_token']:,} bytes | "
        f"{r['kv_bytes_per_token'] / 1024:.0f} KiB |",
    ])


def ch03_measured(d: dict) -> str:
    out = ["| Prompt length | Reading the prompt | Writing tokens | A token costs this much more to write |",
           "|---|---|---|---|"]
    for r in d["measured"]:
        flag = " \\*" if r["prefill_noisy"] else ""
        out.append(f"| {r['prompt']:,} | {r['prefill_tokens_per_s']:,.0f} tok/s{flag} | "
                   f"{r['decode_tokens_per_s']:,.0f} tok/s | "
                   f"**{r['decode_vs_prefill_per_token']:.1f}x** |")
    out += ["", "\\* run-to-run spread exceeded 5%."]
    return "\n".join(out)


def ch03_reference(d: dict) -> str:
    r = d["reference_8b"]
    ridge = r["hardware"]["ridge_flop_per_byte"]
    rows = [("Reading the prompt (prefill)", r["prefill"]), ("Writing a token (decode)", r["decode"])]
    out = ["| Phase | Arithmetic | Bytes fetched | Work per byte | Limited by |",
           "|---|---|---|---|---|"]
    for name, x in rows:
        out.append(f"| {name} | {x['flops'] / 1e12:,.1f} TFLOP | "
                   f"{x['bytes'] / 1e9:,.1f} GB | **{x['intensity']:,.2f}** | "
                   f"{x['bound_by']} |")
    out += ["", f"An {r['config']['params'] / 1e9:.0f}B model in bf16 on an accelerator that "
                f"breaks even at **{ridge:.0f}** operations per byte: below that it waits for "
                f"memory, above it it waits for arithmetic. Prefill of a "
                f"{r['config']['prompt']:,}-token prompt; decode at a "
                f"{r['config']['context']:,}-token context. Arithmetic over published "
                "specs, not a measurement."]
    return "\n".join(out)


def ch04_machine(d: dict) -> str:
    m, a = d["machine"], d["accelerator"]
    return "\n".join([
        "| | This machine (measured) | An H100 (published) |", "|---|---|---|",
        f"| Memory bandwidth | {m['dram_bytes_per_s'] / 1e9:.0f} GB/s | "
        f"{a['hbm_bytes_per_s'] / 1e12:.2f} TB/s |",
        f"| Arithmetic | {m['flops'] / 1e9:.0f} GFLOP/s | "
        f"{a['peak_bf16_flops'] / 1e12:.0f} TFLOP/s |",
        f"| **Breaks even at** | **{m['ridge_flop_per_byte']:.0f}** FLOP/byte | "
        f"**{a['ridge_flop_per_byte']:.0f}** FLOP/byte |",
        "",
        f"The accelerator is {a['ridge_ratio_vs_machine']:.0f}x more lopsided: it "
        "carries far more arithmetic per unit of memory bandwidth, so work that "
        "is short of arithmetic is punished far more severely on it. Measured "
        "single-threaded; see the note on the measuring machine.",
    ])


def ch04_wall(d: dict) -> str:
    out = ["| Model weights | Time to write one token | Predicted by bandwidth alone | Measured / predicted |",
           "|---|---|---|---|"]
    for r in d["wall"]:
        out.append(f"| {r['weight_mib']:,.0f} MiB | {r['decode_step_s'] * 1e3:.2f} ms | "
                   f"{r['predicted_from_bandwidth_s'] * 1e3:.2f} ms | "
                   f"**{r['measured_over_predicted']:.2f}x** |")
    return "\n".join(out)


def ch05_budgets(d: dict) -> str:
    out = ["| Latency budget | Largest batch it allows | Throughput | Cost per 1M output tokens | What stops you |",
           "|---|---|---|---|---|"]
    for b in d["budgets"]:
        stop = "memory runs out first" if b["limited_by_memory_not_budget"] else "the budget"
        out.append(f"| {b['itl_budget_ms']} ms | {b['largest_batch']} | "
                   f"{b['tokens_per_s']:,.0f} tok/s | "
                   f"**${b['usd_per_m_tokens']:.3f}** | {stop} |")
    out += ["", "Arithmetic over published specs, not a measurement. The "
                "batch is also capped at "
                f"{d['case_study']['max_concurrent']} sequences by the memory "
                "accounting in Chapter 13."]
    return "\n".join(out)


def ch05_percentiles(d: dict) -> str:
    m = d["measured"]
    rows = [("Mean", m["mean_ms"]), ("p50 (median)", m["p50_ms"]),
            ("p90", m["p90_ms"]), ("p99", m["p99_ms"]),
            ("p99.9", m["p999_ms"]), ("Slowest seen", m["max_ms"])]
    out = ["| | Time for one decode step | Relative to the median |", "|---|---|---|"]
    for name, v in rows:
        out.append(f"| {name} | {v:.2f} ms | {v / m['p50_ms']:.2f}x |")
    out += ["", f"{m['samples']:,} consecutive decode steps on tinyserve, "
                "nothing else running."]
    return "\n".join(out)


def ch06_breakeven(d: dict) -> str:
    out = ["| Precision | GPU pricing | Cost per 1M tokens at full tilt | Utilization needed to beat the API |",
           "|---|---|---|---|"]
    for sc in d["scenarios"]:
        b = sc["breakeven_utilization"]
        verdict = (f"**{b * 100:.0f}%**" if b <= 1
                   else f"**never** — would need {b * 100:.0f}%")
        out.append(f"| {sc['precision']} | {sc['pricing']} | "
                   f"${sc['usd_per_m_at_full']:.3f} | {verdict} |")
    out += ["", f"Against a published API price of "
                f"${d['api']['usd_per_m_output']:.2f} per million output tokens "
                f"({d['api']['note']}). Hardware only: engineering and operations "
                f"typically add another "
                f"{d['assumptions']['engineering_multiple'][0]}-"
                f"{d['assumptions']['engineering_multiple'][1]}x on top."]
    return "\n".join(out)


def ch07_sizes(d: dict) -> str:
    out = ["| Matrix multiplied | Rate achieved | Share of this machine's best |",
           "|---|---|---|"]
    for r in d["sizes"]:
        out.append(f"| {r['n']} x {r['n']} | {r['gflops']:,.0f} GFLOP/s | "
                   f"**{r['share_of_peak'] * 100:.0f}%** |")
    dev = d["device"]
    out += ["", f"Measured on {dev['name']}, {dev['cores_visible']} cores, median "
                "of 5 runs. The curve is not perfectly smooth: some sizes suit the "
                "library's internal blocking better than others."]
    return "\n".join(out)


def ch07_scaling(d: dict) -> str:
    out = ["| Cores allowed | Arithmetic | Memory bandwidth |", "|---|---|---|"]
    for r in d["scaling"]:
        out.append(f"| {r['threads']} | {r['gflops']:,.0f} GFLOP/s "
                   f"(**{r['gflops_speedup']:.2f}x**) | "
                   f"{r['bandwidth_gb_s']:.1f} GB/s "
                   f"(**{r['bandwidth_speedup']:.2f}x**) |")
    return "\n".join(out)


def ch08_points(d: dict) -> str:
    out = ["| Operation | Arithmetic per byte | Rate achieved | Share of the bound | Limited by |",
           "|---|---|---|---|---|"]
    for p in d["points"]:
        out.append(f"| {p['name']} | {p['intensity']:,.2f} | "
                   f"{p['achieved_flops'] / 1e9:,.1f} GFLOP/s | "
                   f"**{p['fraction_of_roof'] * 100:.0f}%** | "
                   f"{p['predicted_bound_by']} |")
    m = d["machine"]
    out += ["", f"Peak {m['peak_flops'] / 1e9:,.0f} GFLOP/s and bandwidth "
                f"{m['bandwidth_bytes_per_s'] / 1e9:.1f} GB/s are the largest values "
                "measured here, so the two operations that define them reach "
                "100% by construction. No operation exceeds the bound."]
    return "\n".join(out)


def ch08_ceiling(d: dict) -> str:
    a = d["accelerator"]
    base, ridge = a["intensity_ceiling"], a["ridge_flop_per_byte"]
    out = ["| Change | Ceiling on arithmetic per byte | Reaches break-even? |",
           "|---|---|---|"]
    for v in a["ceiling_variants"]:
        c = base * v["factor"]
        verdict = (f"**no** — {ridge / c:.1f}x short" if c < ridge
                   else "**yes**, just clears it")
        out.append(f"| {v['what']} | {c:,.0f} | {verdict} |")
    out += ["", f"However large the batch, decode's arithmetic per byte cannot "
                f"pass these values, against a break-even point of {ridge:.0f}."]
    return "\n".join(out)


def ch09_framings(d: dict) -> str:
    out = ["| How it is reported | What is wrong with it | Tokens per second | Relative |",
           "|---|---|---|---|"]
    for w in sorted(d["ways"], key=lambda x: -x["tokens_per_s"]):
        out.append(f"| {w['how']} | {w['sin']} | {w['tokens_per_s']:,.0f} | "
                   f"**{w['relative_to_honest']:.2f}x** |")
    sp = d["spread"]
    out += ["", f"The same operation, on the same machine, unchanged. "
                f"**{sp['ratio']:.1f}x** separates the most flattering framing from the "
                "least, and every row is defensible on its own."]
    return "\n".join(out)


def ch10_agreement(d: dict) -> str:
    out = ["| Architecture checked | Largest difference in any score | Same tokens chosen |",
           "|---|---|---|"]
    for c in d["agreement"]["cases"]:
        cfg = c["config"]
        shape = (f"{cfg['n_layers']}L, d={cfg['d_model']}, "
                 f"{cfg['n_heads']}Q/{cfg['n_kv_heads']}KV heads")
        out.append(f"| {c['case']} ({shape}) | {c['max_absolute_difference']:.1e} | "
                   f"**{'yes' if c['same_greedy_tokens'] else 'NO'}** |")
    a = d["agreement"]
    out += ["", f"Same weights, {a['tokens_compared']} tokens, against PyTorch "
                f"{a['torch_version']}. Differences of this size are float32 "
                "rounding: the two implementations do the same arithmetic in a "
                "different order."]
    return "\n".join(out)


def ch10_baseline(d: dict) -> str:
    b = d["baseline"]
    nc, wc = b["no_cache"], b["with_cache"]
    return "\n".join([
        "| The starting point | Value |", "|---|---|",
        f"| Model | {b['params']:,} parameters, {b['weight_bytes'] / 1e6:.1f} MB |",
        f"| Task | {b['prompt']}-token prompt, {b['generated']} tokens generated |",
        f"| Without a cache | {nc['tokens_per_s']:,.0f} tokens/s |",
        f"| With a cache | {wc['tokens_per_s']:,.0f} tokens/s |",
        f"| Time to first token | {wc['ttft_ms']:.1f} ms |",
        f"| Decode step, p50 | {wc['decode_p50_ms']:.2f} ms |",
        f"| Decode step, p99 | {wc['decode_p99_ms']:.2f} ms |",
        f"| KV cache per token | {wc['kv_bytes_per_token']:,} bytes |",
    ])


def ch11_waste(d: dict) -> str:
    out = ["| Tokens written | Sequence length | Arithmetic performed | Arithmetic that earned the token | Discarded |",
           "|---|---|---|---|---|"]
    for w in d["waste"]:
        out.append(f"| {w['step']} | {w['sequence']} | "
                   f"{w['flops_total'] / 1e6:,.0f} MFLOP | "
                   f"{w['flops_useful'] / 1e6:,.1f} MFLOP | "
                   f"**{w['wasted_fraction'] * 100:.2f}%** |")
    return "\n".join(out)


def ch11_growth(d: dict) -> str:
    out = ["| Tokens written | Total time | Time per token | Arithmetic discarded |",
           "|---|---|---|---|"]
    for s_ in d["sweep"]:
        flag = " \\*" if s_["noisy"] else ""
        out.append(f"| {s_['generated']} | {s_['total_s']:.2f} s{flag} | "
                   f"**{s_['seconds_per_token'] * 1e3:.1f} ms** | "
                   f"{s_['wasted_fraction'] * 100:.1f}% |")
    out += ["", "\\* run-to-run spread exceeded 5%."]
    return "\n".join(out)


def ch14_capacity(d: dict) -> str:
    a, dflt = d["assumptions"], d["at_default"]
    sc = d["step_cost"]
    return "\n".join([
        "| | Reserving the full context | Paged, "
        f"{dflt['block_size']}-token blocks |", "|---|---|---|",
        f"| Sequences that fit in {a['pool_gb']:.0f} GB | "
        f"{dflt['admitted_contiguous']} | **{dflt['admitted_paged']}** |",
        f"| Memory held that is in use | 16% (Chapter 13) | "
        f"**{dflt['utilization'] * 100:.1f}%** |",
        f"| Wasted per sequence | up to the whole context | "
        f"**{dflt['wasted_tokens_per_sequence']:.1f} tokens** |",
        f"| Cost per decode step | 1.00x | "
        f"**{sc['paged_over_contiguous']:.2f}x** |",
        "",
        f"Capacity computed for the reference model over "
        f"{a['requests_sampled']:,} sampled requests; the step cost is measured on "
        "tinyserve. Chapter 13 predicted these capacities from arithmetic "
        "before this allocator existed.",
    ])


def ch14_blocksize(d: dict) -> str:
    out = ["| Block size | Sequences admitted | Memory in use | Wasted per sequence | Worst case |",
           "|---|---|---|---|---|"]
    for r in d["block_sizes"]:
        mark = " ←" if r["block_size"] == d["default_block_size"] else ""
        out.append(f"| {r['block_size']}{mark} | {r['admitted_paged']} | "
                   f"{r['utilization'] * 100:.1f}% | "
                   f"{r['wasted_tokens_per_sequence']:.1f} tokens | "
                   f"{r['worst_case_waste_per_sequence']} tokens |")
    out += ["", "← the size production engines default to. Waste per sequence "
                "averages about half a block and can never exceed one block "
                "less one token."]
    return "\n".join(out)


def ch16_matmul(d: dict) -> str:
    big = next(m for m in d["matmuls"] if m["name"] == "memory-resident")
    out = [f"| Sequences | Time for one {big['shape'][0]}x{big['shape'][1]} matmul | "
           "Weights re-read at | Arithmetic rate | Time per token |",
           "|---|---|---|---|---|"]
    for r in big["rows"]:
        flag = " \\*" if r["noisy"] else ""
        out.append(f"| {r['batch']} | {r['seconds'] * 1e3:.2f} ms{flag} | "
                   f"{r['bytes_per_s'] / 1e9:.0f} GB/s | "
                   f"{r['flops'] / 1e9:.0f} GFLOP/s | "
                   f"**{r['per_token_ms'] * 1e3:.0f} us** |")
    out += ["", f"{big['matrices']} weight matrices of "
                f"{big['weight_bytes'] / 1024**2:.0f} MiB "
                f"({big['working_set_bytes'] / 1024**2:.0f} MiB in total, more "
                "than this machine's cache holds), each multiplied once by a "
                "batch of rows; the time is one pass divided by the number of "
                "matrices. Nothing else is in the measurement: no attention, "
                "no cache, no model.",
            "", "\\* run-to-run spread exceeded 5%."]
    return "\n".join(out)


def ch16_sweep(d: dict) -> str:
    big = next(m for m in d["models"] if m["name"] == "memory-resident")
    small = next(m for m in d["models"] if m["name"] == "cache-resident")
    out = ["| Sequences | Step time | Of which shared | Of which per-sequence | "
           "Tokens/s | Tokens/s, cache-resident model |",
           "|---|---|---|---|---|---|"]
    for r, s in zip(big["rows"], small["rows"]):
        out.append(f"| {r['batch']} | {r['inter_token_ms']:.1f} ms | "
                   f"{r['shared_s'] * 1e3:.1f} ms | {r['private_s'] * 1e3:.1f} ms | "
                   f"**{r['tokens_per_s']:,.0f}** | {s['tokens_per_s']:,.0f} |")
    out += ["", f"The whole decode step for a {big['weight_mib']:.0f} MiB model "
                f"whose weights come from memory at "
                f"{big['bandwidth_bytes_per_s'] / 1e9:.1f} GB/s, beside the "
                f"{small['weight_mib']:.1f} MiB model whose weights are already "
                "in cache. The step time is what one user waits between tokens."]
    return "\n".join(out)


def ch16_waste(d: dict) -> str:
    out = ["| Batch | Prompt tokens that are real | Decode steps that are real | "
           "Together | Steps the batch runs | Steps a sequence needs |",
           "|---|---|---|---|---|---|"]
    for w in d["waste"]:
        out.append(f"| {w['batch']} | {w['prompt_utilization'] * 100:.0f}% | "
                   f"{w['output_utilization'] * 100:.0f}% | "
                   f"**{w['total_utilization'] * 100:.0f}%** | "
                   f"{w['steps_per_batch']:,.0f} | "
                   f"{w['useful_steps_per_sequence']:,.0f} |")
    e = d["experiment"]
    out += ["", f"Batches filled from {e['n_requests']:,} sampled requests "
                f"(seed {e['seed']}), each run until its longest sequence "
                "finishes. Nothing here is an implementation flaw: it all "
                "follows from the batch being fixed for its whole life."]
    return "\n".join(out)


def ch15_numerics(d: dict) -> str:
    n, e = d["numerics"], d["equivalence"]
    yes = lambda b: "**yes**" if b else "**no**"
    out = [
        "| Step of the forward pass | Contracts over | Same answer when the sequence is "
        f"{n['padding_tokens']} tokens longer? |",
        "|---|---|---|",
        f"| A projection (Q, K, V, feed-forward) | the model dimension, which does not change | "
        f"{yes(n['projection_same'])} |",
        f"| The softmax in attention | the sequence, which does | "
        f"{yes(n['softmax_same'])} — differs by {n['softmax_diff']:.1e} |",
        "",
        "And what that does to the keys the cache hands back, layer by layer:",
        "",
        "| Layer | Difference from the same key computed afresh |",
        "|---|---|",
    ]
    for i, diff in enumerate(e["cached_key_diff_by_layer"]):
        out.append(f"| {i} | {'exactly 0' if diff == 0 else f'{diff:.1e}'} |")
    out += ["", "Layer 0's keys come from the embedding and a projection alone, so "
                "they are identical. Every layer after it has been through an "
                "attention softmax."]
    return "\n".join(out)


def ch15_prefill(d: dict) -> str:
    pf = d["prefill"]
    out = ["| Prompt already cached | Tokens still computed | Prefill, no hit | "
           "Prefill, hit | Speedup | If time went with tokens |",
           "|---|---|---|---|---|---|"]
    for r in pf["points"]:
        flag = " \\*" if r["noisy"] else ""
        out.append(f"| {r['shared_tokens']} of {pf['prompt_tokens']} "
                   f"({r['shared_frac'] * 100:.0f}%) | {r['tokens_computed']} | "
                   f"{r['cold_s']['median'] * 1e3:.1f} ms | "
                   f"{r['warm_s']['median'] * 1e3:.1f} ms | "
                   f"**{r['speedup']:.2f}x**{flag} | "
                   f"{r['speedup_if_proportional']:.2f}x |")
    out += ["", f"Median of {pf['runs']} paired runs after {pf['warmup']} warmup, "
                "cold and warm measured one after the other in each run so that "
                "a machine-wide stall moves both. The lookup in the index is "
                "timed with the hit; the miss lookup on the cold path is not "
                "charged, which makes the comparison slightly unkind to the "
                "cache.",
            "",
            "\\* spread of the paired ratio exceeded 5%; this machine is a "
            "shared container."]
    return "\n".join(out)


def ch15_policies(d: dict) -> str:
    gb = d["assumptions"]["gb_per_block"]
    names = {"lru": "Least recently used **leaf**",
             "lfu": "Least frequently used leaf",
             "unstructured": "Least recently used block, leaf or not"}
    sizes = sorted({r["pool_blocks"] for r in d["sizes"]})
    out = ["| Cache size | " + " | ".join(names[p] for p in ("lru", "lfu", "unstructured")) + " |",
           "|---|---|---|---|"]
    for n in sizes:
        cells = []
        for policy in ("lru", "lfu", "unstructured"):
            r = next(x for x in d["sizes"]
                     if x["policy"] == policy and x["pool_blocks"] == n)
            cells.append(f"{r['hit_rate'] * 100:.1f}%")
        out.append(f"| {n * gb:,.1f} GB ({n:,} blocks) | " + " | ".join(cells) + " |")
    out += ["", "Share of prompt tokens served from the cache, over "
                f"{d['trace']['requests']} requests from {d['trace']['sessions']} "
                f"sessions (seed {d['assumptions']['seed']}). The largest size "
                "holds everything this traffic can share. Sizes are the "
                "reference model's bytes; the simulation runs the real index "
                "and the real allocator."]
    return "\n".join(out)


def ch13_policies(d: dict) -> str:
    names = {"max_model_len": "Reserve the full context (8,192)",
             "prompt_plus_cap": "Reserve prompt + cap (prompt + 1,024)",
             "paged_16": "Pages of 16 tokens (Chapter 14)",
             "oracle": "Perfect foresight (not achievable)"}
    order = ["max_model_len", "prompt_plus_cap", "paged_16", "oracle"]
    rows = sorted(d["policies"], key=lambda r: order.index(r["policy"]))
    out = ["| Allocation policy | Held per sequence | In use | Wasted | Concurrent sequences |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {names[r['policy']]} | {r['reserved_tokens_mean']:,.0f} tok "
                   f"({r['bytes_per_seq'] / 1024**2:,.0f} MiB) | "
                   f"{r['utilization'] * 100:.0f}% | **{r['waste'] * 100:.0f}%** | "
                   f"**{r['concurrent_seqs']}** |")
    return "\n".join(out)


def ch13_traffic(d: dict) -> str:
    t, e = d["traffic"], d["experiment"]
    return "\n".join([
        "| | p50 | p99 |", "|---|---|---|",
        f"| Prompt tokens | {t['prompt_p50']:,.0f} | {t['prompt_p99']:,.0f} |",
        f"| Output tokens | {t['output_p50']:,.0f} | {t['output_p99']:,.0f} |",
        f"| Total per sequence | {t['total_p50']:,.0f} | {t['total_p99']:,.0f} |",
        "",
        f"{e['n_requests']:,} sampled requests, seed {e['seed']}; lengths are "
        f"lognormal about means of {e['prompt_mean']:,} and {e['output_mean']} "
        f"tokens. {e['rejected_over_context']:,} samples exceeded the "
        f"{e['max_model_len']:,}-token context and were dropped, as the server "
        "would refuse them.",
    ])


def ch17_head_to_head(d: dict) -> str:
    h, a = d["head_to_head"], d["assumptions"]
    st, co = h["static"], h["continuous"]
    ms = lambda x: f"{x:,.0f} ms"
    rows = [
        ("Output tokens a second",
         f"{st['tokens_per_s']:,.0f}", f"{co['tokens_per_s']:,.0f}",
         f"{co['tokens_per_s'] / st['tokens_per_s']:.1f}x"),
        ("Time to first token, p50",
         ms(st["ttft_p50_ms"]), ms(co["ttft_p50_ms"]),
         f"{st['ttft_p50_ms'] / co['ttft_p50_ms']:,.0f}x"),
        ("Time to first token, p99",
         ms(st["ttft_p99_ms"]), ms(co["ttft_p99_ms"]),
         f"{st['ttft_p99_ms'] / co['ttft_p99_ms']:,.0f}x"),
        ("Between tokens, p50",
         f"{st['itl_p50_ms']:.1f} ms", f"{co['itl_p50_ms']:.1f} ms",
         f"{st['itl_p50_ms'] / co['itl_p50_ms']:.1f}x"),
        ("Between tokens, p99",
         f"{st['itl_p99_ms']:.1f} ms", f"{co['itl_p99_ms']:.1f} ms",
         f"{co['itl_p99_ms'] / st['itl_p99_ms']:.1f}x worse"),
        ("End to end, p50",
         f"{st['total_p50_s']:.1f} s", f"{co['total_p50_s']:.1f} s",
         f"{st['total_p50_s'] / co['total_p50_s']:.0f}x"),
        ("Slots the batch held, mean",
         f"{st['mean_slots']:,.0f}", f"{co['mean_slots']:,.1f}", "--"),
        ("Sequences actually advancing, mean",
         f"{st['mean_batch']:.1f}", f"{co['mean_batch']:.1f}", "--"),
        ("Slots held that held live work",
         f"{st['slot_utilization'] * 100:.0f}%",
         f"{co['slot_utilization'] * 100:.0f}%", "--"),
        ("Peak of the block pool",
         f"{st['peak_pool_share'] * 100:.0f}%",
         f"{co['peak_pool_share'] * 100:.0f}%",
         f"{st['peak_blocks'] / co['peak_blocks']:.0f}x"),
        ("Time to drain, over the arrival window",
         f"{st['drained_over_span']:.1f}x", f"{co['drained_over_span']:.2f}x",
         "--"),
    ]
    out = ["| | Static batching | Continuous batching | Ratio |", "|---|---|---|---|"]
    out += [f"| {a_} | {b} | {c} | **{e}** |" for a_, b, c, e in rows]
    out += ["", f"The same {a['n_requests']} requests, the same arrivals "
                f"({a['head_to_head_rate']} a second, Poisson), the same "
                f"prompt and output lengths, the same cost model for a "
                f"prefill and a decode step. Only the scheduler differs. "
                f"Static forms a batch when {a['max_batch']} requests have "
                "arrived or one second has passed, whichever comes first, "
                "and is given all the memory it asks for. Every ratio is "
                "the better number over the worse one, except where it says "
                "otherwise."]
    return "\n".join(out)


def ch17_load(d: dict) -> str:
    a = d["assumptions"]
    by = {}
    for r in d["rows"]:
        by.setdefault(r["rate"], {})[r["policy"]] = r
    out = ["| Requests a second | Offered | Static: tokens/s | Static: TTFT p99 | "
           "Continuous: tokens/s | Continuous: TTFT p99 | Continuous: between tokens, p99 |",
           "|---|---|---|---|---|---|---|"]
    for rate in sorted(by):
        st, co = by[rate]["static"], by[rate]["continuous"]
        mark = lambda r: "" if r["keeping_up"] else " \\*"
        out.append(f"| {rate} | {st['offered_tokens_per_s']:,.0f} | "
                   f"{st['tokens_per_s']:,.0f}{mark(st)} | "
                   f"{st['ttft_p99_ms'] / 1e3:,.1f} s | "
                   f"**{co['tokens_per_s']:,.0f}**{mark(co)} | "
                   f"{co['ttft_p99_ms']:,.0f} ms | "
                   f"{co['itl_p99_ms']:.0f} ms |")
    out += ["", f"\\* not keeping up: the server took more than 10% longer to "
                f"drain than the requests took to arrive, or missed the "
                f"{a['ttft_budget_ms']:,} ms p99 time-to-first-token budget. "
                f"Offered load is the arrival rate times the mean output "
                f"length ({a['output_mean']} tokens), which is what the "
                "service would have to produce to keep up."]
    return "\n".join(out)


def ch17_pool(d: dict) -> str:
    a = d["assumptions"]
    out = ["| Block pool | Sequences in flight | Tokens/s | TTFT p99 | "
           "Preemptions | Tokens generated twice |",
           "|---|---|---|---|---|---|"]
    for r in d["pool_sweep"]:
        out.append(f"| {r['pool_gb']:.1f} GB ({r['pool_share'] * 100:.0f}%) | "
                   f"{r['mean_batch']:.1f} | **{r['tokens_per_s']:,.0f}** | "
                   f"{r['ttft_p99_ms'] / 1e3:,.1f} s | {r['preemptions']:,} | "
                   f"{r['wasted_token_share'] * 100:.1f}% |")
    out += ["", f"Continuous batching at {a['head_to_head_rate']} requests a "
                f"second through smaller and smaller pools. The full pool is "
                f"{a['pool_bytes'] / 1e9:.0f} GB, which is what is left on one "
                "accelerator after the weights. Nothing else changes."]
    return "\n".join(out)


def ch18_budget(d: dict) -> str:
    a = d["assumptions"]
    out = ["| Token budget | Tokens/s | TTFT p50 | TTFT p99 | Between tokens, p50 | "
           "Between tokens, p99 | Iterations carrying prefill | Keeps |",
           "|---|---|---|---|---|---|---|---|"]
    for r in d["budgets_high"]:
        name = (f"**{r['token_budget']:,}**" if r["token_budget"] == d["chosen_budget"]
                else f"{r['token_budget']:,}" if r["token_budget"]
                else "_prefill alone_")
        met = ("both" if r["meets_ttft"] and r["meets_itl"] else
               "TTFT only" if r["meets_ttft"] else
               "between-token only" if r["meets_itl"] else "**neither**")
        ttft = (f"{r['ttft_p50_ms'] / 1e3:,.1f} s" if r["ttft_p50_ms"] >= 1000
                else f"{r['ttft_p50_ms']:,.0f} ms")
        ttft99 = (f"{r['ttft_p99_ms'] / 1e3:,.1f} s" if r["ttft_p99_ms"] >= 1000
                  else f"{r['ttft_p99_ms']:,.0f} ms")
        out.append(f"| {name} | {r['tokens_per_s']:,.0f} | {ttft} | {ttft99} | "
                   f"{r['itl_p50_ms']:.1f} ms | {r['itl_p99_ms']:.1f} ms | "
                   f"{r['mixed_share'] * 100:.0f}% | {met} |")
    out += ["", f"{a['n_requests']} requests at a rate of {a['rate_high']} a second "
                f"(seed {a['seed']}), the rate at which the previous chapter's "
                f"scheduler stopped keeping its promise. The first row is that "
                f"scheduler: a prefill gets an iteration to itself. Every row "
                f"below mixes prefill into the same iteration as the decodes, "
                f"splitting it when it does not fit in the budget. "
                f"\"Keeps\" is against the case study's p99 promises: "
                f"{a['ttft_budget_ms']:,} ms to the first token and "
                f"{a['itl_budget_ms']} ms between them."]
    return "\n".join(out)


def ch18_policy(d: dict) -> str:
    a = d["assumptions"]
    out = ["| Block pool | Queue order | End to end p50 | End to end p99 | "
           "Slowdown p50 | Slowdown p99 | Worst slowdown | Tokens/s |",
           "|---|---|---|---|---|---|---|---|"]
    for r in d["policies"]:
        pool = (f"{r['pool_gb']:.0f} GB" if r["pool"] == "full"
                else f"{r['pool_gb']:.1f} GB")
        out.append(f"| {pool} ({r['pool']}) | `{r['policy']}` | "
                   f"{r['total_p50_s']:.2f} s | {r['total_p99_s']:.2f} s | "
                   f"**{r['slowdown_p50']:.1f}x** | {r['slowdown_p99']:.0f}x | "
                   f"{r['slowdown_max']:.0f}x | {r['tokens_per_s']:,.0f} |")
    out += ["", f"The same {a['n_requests']} requests at {a['rate_high']} a "
                f"second through three queue orders, twice: once with the whole "
                f"block pool and once with it squeezed to "
                f"{a['swap_pool_share'] * 100:.0f}% of it. Slowdown is how much "
                "longer a request took than it would have taken alone on an "
                "idle server -- the fairness number, and the one that moves. "
                "`shortest-output` and `longest-output` sort by the true reply "
                "length, which a real server does not know; they are the best "
                "and worst a perfect oracle could do."]
    return "\n".join(out)


def ch18_preemption(d: dict) -> str:
    a = d["assumptions"]
    out = ["| Way out of a full pool | Tokens/s | Sequences in flight | "
           "Preemptions | Prompt tokens read | Copied | Time copying | TTFT p99 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in d["preemption"]:
        out.append(f"| {r['mode']} | **{r['tokens_per_s']:,.0f}** | "
                   f"{r['mean_batch']:.1f} | {r['preemptions']:,} | "
                   f"{r['prompt_reread']:.2f}x over | "
                   f"{r['swapped_bytes'] / 1e9:,.0f} GB | "
                   f"{r['swap_s']:.2f} s | {r['ttft_p99_ms'] / 1e3:,.1f} s |")
    out += ["", f"A {d['preemption'][0]['pool_gb']:.1f} GB pool "
                f"({a['swap_pool_share'] * 100:.0f}% of the accelerator's free "
                f"memory) at {a['rate']} requests a second, small enough that "
                "the server has to take sequences back out of the batch. "
                "Recomputing throws the evicted cache away; swapping copies it "
                "to host memory and back across the named link."]
    return "\n".join(out)


def ch18_swap(d: dict) -> str:
    arith = d["swap_arithmetic"]
    links = list(arith["links"])
    out = ["| Context | Recompute it | " + " | ".join(f"Copy it over {n}" for n in links)
           + " | Link speed at which they tie |",
           "|---|---|" + "---|" * (len(links) + 1)]
    for r in arith["rows"]:
        out.append(f"| {r['tokens']:,} tokens | {r['recompute_s'] * 1e3:.1f} ms | "
                   + " | ".join(f"{r['swap_s'][n] * 1e3:.1f} ms" for n in links)
                   + f" | **{r['breakeven_bytes_per_s'] / 1e9:.0f} GB/s** |")
    out += ["", "One preempted sequence, both ways. Recomputing re-reads its "
                "context: arithmetic, growing faster than the length. Copying "
                "moves its keys and values out and back: "
                f"{d['assumptions']['kv_bytes_per_token'] / 1024:.0f} KiB a "
                "token each way, growing linearly. The last column is the link "
                "speed at which the two cost the same."]
    return "\n".join(out)


def ch19_transfer(d: dict) -> str:
    tr, a = d["transfer"], d["assumptions"]
    links = list(tr["links"])
    out = ["| Context | Its cache | Prefill takes | "
           + " | ".join(f"Over {n}" for n in links)
           + " | Link to match the prefill |",
           "|---|---|---|" + "---|" * (len(links) + 1)]
    for r in tr["rows"]:
        out.append(f"| {r['tokens']:,} tokens | {r['bytes'] / 1e6:,.0f} MB | "
                   f"{r['prefill_s'] * 1e3:.1f} ms | "
                   + " | ".join(f"{r['over'][n] * 1e3:.1f} ms" for n in links)
                   + f" | **{r['link_to_match_prefill'] / 1e9:.1f} GB/s** |")
    out += ["", f"The reference model keeps "
                f"{a['kv_bytes_per_token'] / 1024:.0f} KiB of keys and values "
                "per token, so a cache is large and moving it is a bandwidth "
                "problem, not a latency one. The last column is the link speed "
                "at which the move takes exactly as long as the prefill that "
                "produced it -- a link slower than that turns the move into "
                f"the bottleneck. One decode step, for scale, is "
                f"{tr['decode_step_ms']:.1f} ms."]
    return "\n".join(out)


def ch19_split(d: dict) -> str:
    a, co = d["assumptions"], d["colocated"]
    out = ["| Machines | Tokens/s | TTFT p50 | TTFT p99 | Between tokens, p50 | "
           "Between tokens, p99 | Sequences per decode machine |",
           "|---|---|---|---|---|---|---|"]
    for r in d["splits"]:
        name = f"{r['prefill_workers']}P + {r['decode_workers']}D"
        if r is d["best_split"] or (r["prefill_workers"]
                                    == d["best_split"]["prefill_workers"]):
            name = f"**{name}**"
        fmt = lambda x: f"{x / 1e3:,.1f} s" if x >= 1000 else f"{x:,.0f} ms"
        out.append(f"| {name} | {r['tokens_per_s']:,.0f} | "
                   f"{fmt(r['ttft_p50_ms'])} | {fmt(r['ttft_p99_ms'])} | "
                   f"{r['itl_p50_ms']:.1f} ms | {r['itl_p99_ms']:.1f} ms | "
                   f"{r['mean_batch']:.1f} |")
    out.append(f"| _{a['fleet']} colocated_ | **{co['tokens_per_s']:,.0f}** | "
               f"{co['ttft_p50_ms']:,.0f} ms | {co['ttft_p99_ms']:,.0f} ms | "
               f"{co['itl_p50_ms']:.1f} ms | {co['itl_p99_ms']:.1f} ms | "
               f"{co['mean_batch']:.1f} |")
    out += ["", f"{a['n_requests']:,} requests at {a['rate']} a second "
                f"(seed {a['seed']}) through {a['fleet']} accelerators, "
                f"divided every way, over {a['link']}. The last row is the "
                f"same {a['fleet']} accelerators each doing both phases with "
                f"Chapter 18's scheduler at a {a['token_budget']}-token "
                "budget. Throughput is measured over the arrival window with "
                f"the first {a['warm_fraction'] * 100:.0f}% discarded, so a "
                "fleet's drain tail is not counted as slow serving."]
    return "\n".join(out)


def ch19_links(d: dict) -> str:
    a, co = d["assumptions"], d["colocated"]
    out = ["| Link | Speed | Tokens/s | Wait for the second token, p50 | "
           "p99 | Every later gap, p99 |",
           "|---|---|---|---|---|---|"]
    for r in d["links"]:
        out.append(f"| {r['link']} | "
                   f"{d['transfer']['links'][r['link']] / 1e9:,.0f} GB/s | "
                   f"{r['tokens_per_s']:,.0f} | "
                   f"**{r['second_token_p50_ms']:.1f} ms** | "
                   f"{r['second_token_p99_ms']:.1f} ms | "
                   f"{r['itl_p99_ms']:.1f} ms |")
    out.append(f"| _colocated: no link_ | -- | {co['tokens_per_s']:,.0f} | "
               f"**{co['second_token_p50_ms']:.1f} ms** | "
               f"{co['second_token_p99_ms']:.1f} ms | "
               f"{co['itl_p99_ms']:.1f} ms |")
    out += ["", f"The same {d['best_split']['prefill_workers']}P + "
                f"{d['best_split']['decode_workers']}D fleet over each link. "
                "The prefill machine produces the first token before the cache "
                "goes anywhere, so the whole cost of the move lands in one "
                "place: the wait for the *second* token. Every gap after that "
                "is an ordinary decode step, which is why the last column "
                "barely moves."]
    return "\n".join(out)


def ch19_regimes(d: dict) -> str:
    a = d["assumptions"]
    by = {}
    for r in d["prompts"]:
        by.setdefault(r["prompt_mean"], {})[r["design"]] = r
    out = ["| Regime | Best split | Disaggregated | Colocated | Ratio |",
           "|---|---|---|---|---|"]
    for mean in sorted(by):
        dis, co = by[mean]["disaggregated"], by[mean]["colocated"]
        out.append(f"| {mean:,}-token prompts at {a['rate']} req/s | "
                   f"{dis['prefill_workers']}P + {dis['decode_workers']}D | "
                   f"{dis['tokens_per_s']:,.0f} tok/s | "
                   f"{co['tokens_per_s']:,.0f} tok/s | "
                   f"**{dis['tokens_per_s'] / co['tokens_per_s']:.2f}x** |")
    lp = {r["design"]: r for r in d["long_prompts"]}
    out.append(f"| {a['long_prompt']:,}-token prompts at "
               f"{a['long_rate']:.0f} req/s | "
               f"{lp['disaggregated']['prefill_workers']}P + "
               f"{lp['disaggregated']['decode_workers']}D | "
               f"{lp['disaggregated']['tokens_per_s']:,.0f} tok/s | "
               f"{lp['colocated']['tokens_per_s']:,.0f} tok/s | "
               f"**{lp['disaggregated']['tokens_per_s'] / lp['colocated']['tokens_per_s']:.2f}x** |")
    for r in d["fleets"]:
        b, c = r["best"], r["colocated"]
        out.append(f"| {r['fleet']} accelerators at {b['rate']:.0f} req/s | "
                   f"{b['prefill_workers']}P + {b['decode_workers']}D | "
                   f"{b['tokens_per_s']:,.0f} tok/s | "
                   f"{c['tokens_per_s']:,.0f} tok/s | "
                   f"**{b['tokens_per_s'] / c['tokens_per_s']:.2f}x** |")
    out += ["", "Every disaggregated row re-searches the split, so none of "
                "them is a straw man. The colocated arm is one accelerator "
                f"per replica running Chapter 18's scheduler. The "
                f"{a['long_prompt']:,}-token row holds the prompt tokens "
                "arriving per second at the case study's, so the fleet is not "
                "simply saturated; the rows above it do not, which is why "
                f"both designs fall behind at {a['rate']} req/s with long "
                "prompts."]
    return "\n".join(out)


def ddr1_decisions(d: dict) -> str:
    """The record itself: what was decided, against what, on what evidence."""
    out = ["| Decision | Instead of | Decided by | What decided it |",
           "|---|---|---|---|"]
    for dec in d["decisions"]:
        n = dec["chapter"].removeprefix("ch").lstrip("0")
        out.append(f"| **{dec['decision']}** | {dec['instead_of']} | "
                   f"Chapter {n} | {dec['because']} |")
    out += ["", "Every row is read out of the named chapter's results file "
                "when this table is generated. Nothing in this table was "
                "measured for this record, and nothing in it was typed in by "
                "hand: change a chapter's measurement and the row changes "
                "with it."]
    return "\n".join(out)


def ddr1_would_change(d: dict) -> str:
    """The other half of a decision record: when to revisit it."""
    out = ["| Decision | What would change it |", "|---|---|"]
    for dec in d["decisions"]:
        out.append(f"| {dec['decision']} | {dec['would_change']} |")
    out += ["", "A decision without a condition attached to it is a habit. "
                "These are the conditions -- the things that, if they became "
                "true of a service, would make the row above the wrong answer "
                "for it."]
    return "\n".join(out)


def ddr1_fleet(d: dict) -> str:
    """How few machines the case study's traffic needs."""
    a, f = d["assumptions"], d["fleet"]
    out = ["| Machines | Tokens/s | Share of what the traffic asks for | "
           "TTFT p99 | Between tokens, p99 | $/hour | $/M output tokens |",
           "|---|---|---|---|---|---|---|"]
    for r in d["sizing"]:
        n = f"**{r['workers']}**" if r["workers"] == f["chosen"] else str(r["workers"])
        mark = "" if r["keeping_up"] else " (behind)"
        fmt = lambda x: f"{x / 1e3:,.1f} s" if x >= 1000 else f"{x:,.0f} ms"
        out.append(f"| {n} | {r['tokens_per_s']:,.0f} | "
                   f"{r['share_of_offered'] * 100:.0f}%{mark} | "
                   f"{fmt(r['ttft_p99_ms'])} | {r['itl_p99_ms']:.1f} ms | "
                   f"${r['usd_per_hour']:,.2f} | "
                   f"${r['usd_per_m_output_tokens']:.3f} |")
    out += ["", f"{a['sizing_requests']:,} requests at {a['requests_per_s']} a "
                f"second (seed 0), every machine running Chapter 18's "
                f"scheduler at a {a['token_budget']}-token budget over "
                f"{a['blocks_per_worker']:,} blocks. Throughput is measured "
                f"over the arrival window with the first "
                f"{a['warm_fraction'] * 100:.0f}% discarded. A fleet is "
                f"\"behind\" when it delivers less than "
                f"{a['keeping_up_share'] * 100:.0f}% of the tokens the "
                f"traffic asks for, or misses the "
                f"{a['ttft_budget_ms']:,.0f} ms first-token promise. The "
                f"dollar figures are ${f['gpu_usd_per_hour']:.2f} an hour a "
                "machine, on-demand (FACTS.md)."]
    return "\n".join(out)


def ch20_counted(d: dict) -> str:
    """The bytes the running code moved, against the formula's prediction."""
    a = d["assumptions"]
    out = ["| Prompt | Score matrix | Traffic, built in full | Traffic, tiled | "
           "Times less | Largest thing held | Blocks skipped | Formula |",
           "|---|---|---|---|---|---|---|---|"]
    for r in d["traffic"]:
        out.append(f"| {r['tokens']:,} | {r['score_matrix_bytes'] / 1e6:,.1f} MB "
                   f"| {r['whole_bytes'] / 1e6:,.1f} MB "
                   f"| {r['tiled_bytes'] / 1e6:,.1f} MB "
                   f"| **{r['ratio']:.2f}x** "
                   f"| {r['whole_largest_intermediate'] / 1e6:,.1f} MB -> "
                   f"{r['tiled_largest_intermediate'] / 1e3:,.0f} KB "
                   f"| {r['skipped_share'] * 100:.0f}% "
                   f"| {'agrees' if r['formulas_agree'] else 'DISAGREES'} |")
    out += ["", f"One layer of an {a['measured_heads']}-head model with "
                f"{a['measured_head_dim']}-dimensional heads in float32, tiles "
                f"of {a['q_tile']}x{a['kv_tile']}, seed {a['seed']}. Every "
                "byte is counted as the code moves it, not estimated. The "
                "last column checks each count against the closed-form "
                "formula the next table applies to a model too large to run "
                "here; they must agree exactly, and `make tests` fails if "
                "they do not."]
    return "\n".join(out)


def ch20_reference(d: dict) -> str:
    """The same arithmetic on the model the case study actually serves."""
    ref, a = d["reference"], d["assumptions"]
    out = ["| Prompt | Score matrix, one layer | Against its own inputs | "
           "Traffic, built in full | Traffic, tiled | Times less | "
           "Times less, block resident | All layers, built in full | "
           "All layers, tiled |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in ref["rows"]:
        mark = " *" if r["tokens"] == a["prompt_tokens"] else ""
        out.append(f"| {r['tokens']:,}{mark} "
                   f"| {r['score_matrix_bytes'] / 1e6:,.0f} MB "
                   f"| {r['square_over_inputs']:.1f}x "
                   f"| {r['whole_bytes'] / 1e6:,.0f} MB "
                   f"| {r['tiled_bytes'] / 1e6:,.0f} MB "
                   f"| **{r['ratio']:.1f}x** "
                   f"| {r['ratio_block_resident']:.1f}x "
                   f"| {r['whole_ms_all_layers']:,.2f} ms "
                   f"| {r['tiled_ms_all_layers']:,.2f} ms |")
    out += ["", f"\\* the case study's prompt. The book's 8B model: "
                f"{ref['heads']} query heads sharing {ref['kv_heads']} key "
                f"heads of {ref['head_dim']} dimensions, bf16, "
                f"{ref['layers']} layers, tiles of {a['q_tile']}x"
                f"{a['kv_tile']}. \"Against its own inputs\" is the score "
                "matrix divided by the queries, keys and values it is built "
                "from. \"Times less\" charges the blocks of scores as "
                "traffic, which is what this NumPy implementation makes them; "
                "the column after it charges them to the scratchpad instead, "
                "which is what a CUDA kernel's tile size is chosen for. The "
                "real figure is the second; the chapter quotes the first, so "
                "every saving in it is a lower bound. Times are bytes over "
                f"{ref['hbm_bytes_per_s'] / 1e12:.2f} TB/s (FACTS.md): a "
                "floor for the memory, not a prediction of a kernel's "
                "runtime, which also has arithmetic to do."]
    return "\n".join(out)


def ch20_tiles(d: dict) -> str:
    """What bounds the tile size from each side."""
    t = d["tiles"]
    out = ["| Tile | Memory traffic | Blocks computed | Blocks skipped | "
           "Fast memory one tile needs | Share of a multiprocessor | Fits |",
           "|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        name = f"**{r['tile']}**" if r["tile"] == t["largest_that_fits"] else str(r["tile"])
        out.append(f"| {name} | {r['bytes'] / 1e6:,.0f} MB "
                   f"| {r['tiles_computed']:,} | {r['tiles_skipped']:,} "
                   f"| {r['sram_bytes_reference_model'] / 1024:,.0f} KB "
                   f"| {r['sram_share'] * 100:.0f}% "
                   f"| {'yes' if r['fits'] else 'no'} |")
    out += ["", f"At {t['tokens']:,} tokens. Traffic is counted on the "
                "runnable model; the fast memory is what one tile of the "
                f"book's 8B model needs in bf16 -- a query tile, a key tile, "
                "a value tile, the block of scores between them, the running "
                "output and the two running numbers per row -- against the "
                f"{t['sram_bytes_per_sm'] / 1024:,.0f} KB a Hopper streaming "
                "multiprocessor has (FACTS.md). Traffic falls with every "
                "increase in tile size and the memory rises with the square "
                f"of it, so the tile is as large as will fit: "
                f"{t['largest_that_fits']}. A real kernel also wants room to "
                "fetch the next tile while it works on this one, so it has "
                "less to spend than this table allows."]
    return "\n".join(out)


def ch22_formats(d: dict) -> str:
    """Each format's reach and resolution, from its field widths."""
    out = ["| Format | Bits | Sign/exponent/mantissa | Largest | "
           "Smallest with full precision | Gap either side of 1.0 | "
           "Decimal digits | Peak on one H100 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in d["formats"]:
        name = f"**{r['name']}**" if r["name"] in ("bfloat16", "float16") else r["name"]
        note = " *" if r["compute_only"] else ""
        out.append(f"| {name}{note} | {r['bits']} "
                   f"| 1 / {r['exponent_bits']} / {r['mantissa_bits']} "
                   f"| {r['max_value']:.4g} | {r['min_normal']:.3g} "
                   f"| {r['eps']:.3g} | {r['decimal_digits']:.1f} "
                   f"| {r['peak_tflops']:,.0f} TFLOP/s |")
    out += ["", "Every column but the last is arithmetic on the three "
                "field widths, computed in `tinyserve/precision.py`. "
                "\\* tensorfloat32 is a compute format: 19 meaningful bits "
                "held in a 32-bit slot, so it changes how a multiply is "
                "done and not what a weight costs to store. Peak figures "
                "are the H100 SXM datasheet's, halved from the quoted "
                "\"with sparsity\" rows to the dense throughput that dense "
                "inference gets; float32 is the one that never reaches a "
                "tensor core (FACTS.md)."]
    return "\n".join(out)


def ch22_tensors(d: dict) -> str:
    """What rounding does to the numbers the model is made of."""
    t = d["tensors"]
    names = [r["name"] for r in d["formats"][1:]]
    out = ["| The model's numbers | Largest | " + " | ".join(names) + " |",
           "|---|---|" + "---|" * len(names)]
    for r in t["rows"]:
        out.append(f"| {r['tensor']} | {r['largest']:.3g} | "
                   + " | ".join(f"{r[n]:.2e}" for n in names) + " |")
    c = t["config"]
    out += ["", f"Root-mean-square error after a round trip through each "
                f"format, as a fraction of the largest value in the tensor. "
                f"From a {c['layers']}-layer model of width {c['d_model']:,} "
                f"over {c['tokens']} tokens. Nothing here overflows: every "
                "number in this model is small. float16 and tensorfloat32 "
                "agree down the column because they have the same number of "
                "mantissa bits, which is what error at this scale depends "
                "on -- the extra exponent bits buy range, and range is not "
                "what is being tested."]
    return "\n".join(out)


def ch22_hardware(d: dict) -> str:
    """What the choice of format is worth on the accelerator."""
    h = d["hardware"]
    out = ["| Format | Bytes a number | Weights | Time to read them | "
           "KV cache a token | Ridge point | Compute-bound above batch | "
           "Peak |",
           "|---|---|---|---|---|---|---|---|"]
    for r in h["rows"]:
        out.append(f"| {r['name']} | {r['bytes_per_number']:.0f} "
                   f"| {r['weight_gb']:.0f} GB "
                   f"| {r['weight_read_ms']:.2f} ms "
                   f"| {r['kv_bytes_per_token'] / 1024:,.0f} KiB "
                   f"| {r['ridge_flops_per_byte']:,.0f} FLOP/byte "
                   f"| {r['compute_bound_above_batch']:,.0f} "
                   f"| {r['peak_tflops']:,.0f} TFLOP/s |")
    out += ["", f"The book's {h['params'] / 1e9:.0f}B model on one H100, by "
                f"the format its weights and cache are kept in. \"Time to "
                f"read them\" is the weights over "
                f"{h['hbm_bytes_per_s'] / 1e12:.2f} TB/s, which "
                "Chapter 4 showed is the floor under every decode step. The "
                "ridge point is where the accelerator stops being limited by "
                "memory and starts being limited by arithmetic; it depends "
                "on the arithmetic the format reaches and not at all on how "
                "many bytes a number takes. The last column puts the two "
                "together: a decode step at batch B does 2PB arithmetic and "
                "reads P times the format's bytes, so halving the format "
                "doubles its intensity exactly as it doubles the ridge, and "
                "the batch at which the step turns compute-bound does not "
                "move."]
    return "\n".join(out)


def ch24_schemes(d: dict) -> str:
    """What each scheme costs on the model's own weights."""
    sc = d["schemes"]
    out = ["| Scheme | Bytes a weight | Values sharing a scale | "
           "Error, typical | Error, worst | Against bfloat16 |",
           "|---|---|---|---|---|---|"]
    base = sc["bfloat16_rms"]
    out.append(f"| _bfloat16, for comparison_ | "
               f"{sc['bfloat16_bytes_per_weight']:.3f} | 1 (each carries its "
               f"own exponent) | {base:.2e} | -- | 1x |")
    for r in sc["rows"]:
        out.append(f"| {r['scheme']} | {r['bytes_per_weight']:.3f} "
                   f"| {r['values_per_scale']:,} "
                   f"| {r['rms']:.2e} | {r['worst']:.2e} "
                   f"| {r['rms'] / base:,.0f}x worse |")
    out += ["", f"Root-mean-square error after a round trip, averaged over "
                f"all {sc['matrices']} weight matrices of the book's small "
                "model and reported as a fraction of the largest weight in "
                "each. \"Bytes a weight\" includes the scales: a scale is a "
                "float32 and an asymmetric scheme needs a zero point beside "
                "it, so a small group is not as cheap as its bit width "
                "suggests."]
    return "\n".join(out)


def ch24_groups(d: dict) -> str:
    """What a finer scale buys and what it costs."""
    out = ["| Values per scale | Error | Bytes a weight | Over four bits |",
           "|---|---|---|---|"]
    for r in d["groups"]:
        out.append(f"| {r['group']} | {r['rms']:.2e} "
                   f"| {r['bytes_per_weight']:.3f} | {r['overhead_pct']:+.0f}% |")
    out += ["", "Four-bit symmetric quantization at four group sizes. Every "
                "halving of the group buys a little accuracy and costs a "
                "fixed amount of storage, because each group needs its own "
                "float32 scale. The knee is where a reader's own tolerance "
                "puts it; 32 and 128 are the sizes the published methods use."]
    return "\n".join(out)


def ch24_memory(d: dict) -> str:
    """The reference model under each scheme."""
    mem = d["memory"]
    out = ["| Scheme | Weights | Time to read them | Smaller than bfloat16 | "
           "Left on an 80 GB card for cache |",
           "|---|---|---|---|---|"]
    out.append(f"| _bfloat16_ | {mem['bf16_gb']:.0f} GB "
               f"| {mem['bf16_read_ms']:.2f} ms | 1.00x "
               f"| {mem['bf16_free_gb']:.0f} GB |")
    for r in mem["rows"]:
        out.append(f"| {r['scheme']} | {r['weights_gb']:.1f} GB "
                   f"| {r['read_ms']:.2f} ms | {r['over_bf16']:.2f}x "
                   f"| {r['free_for_cache_gb']:.0f} GB |")
    out += ["", f"The book's {mem['params'] / 1e9:.0f}B model on one "
                f"{mem['gpu_gb']:.0f} GB accelerator. The time to read the "
                "weights is the floor under every decode step (Chapter 4), "
                "and what is left over is what Chapter 13 spends on the KV "
                "cache -- so a smaller model is worth more than its own "
                "size, because the memory it frees becomes batch size."]
    return "\n".join(out)


def ch29_acceptance(d: dict) -> str:
    """How often a draft's guess survives."""
    a = d["acceptance"]
    out = ["| Draft | Bytes a weight | Picks the same token | "
           "Guess survives, sampled |", "|---|---|---|---|"]
    for r in a["rows"]:
        b = ("--" if r["bytes_per_weight"] is None
             else f"{r['bytes_per_weight']:.3f}")
        s_ = ("--" if r["sampled_acceptance"] is None
              else f"{r['sampled_acceptance'] * 100:.1f}%")
        out.append(f"| {r['draft']} | {b} | "
                   f"{r['greedy_agreement'] * 100:.1f}% | {s_} |")
    out += ["", f"Over {a['positions']} positions of one prompt. The drafts "
                "are quantized copies of the target from Chapter 24: cheaper "
                "to run, and related to what they are drafting for, which is "
                "the property that matters. An unrelated model of the same "
                f"architecture agrees at {a['chance'] * 100:.2f}%, which is "
                "chance.\n\nThe two columns differ because the rule does "
                "not require the draft to pick the same token -- it accepts "
                "in proportion to how much the two distributions overlap. "
                "Read the last column with care: this model is untrained, so "
                f"its output is nearly flat ({a['target_entropy_nats']:.2f} "
                f"nats against {a['uniform_entropy_nats']:.2f} for a uniform "
                "distribution over the same vocabulary), and two flat "
                "distributions overlap heavily whatever they are. A trained "
                "model is far more confident and its acceptance rates are "
                "correspondingly lower."]
    return "\n".join(out)


def ch29_speedup(d: dict) -> str:
    """The best number of guesses, by draft cost and acceptance rate."""
    grid = d["speedups"]["grid"]
    drafts = []
    for g in grid:
        if g["draft"] not in drafts:
            drafts.append(g["draft"])
    alphas = sorted({g["alpha"] for g in grid})
    out = ["| Draft (cost of one draft step) | "
           + " | ".join(f"{a:.0%} accepted" for a in alphas) + " |",
           "|---|" + "---|" * len(alphas)]
    by = {(g["draft"], g["alpha"]): g for g in grid}
    for name in drafts:
        cells = []
        for a in alphas:
            g = by[(name, a)]
            cell = f"**{g['speedup']:.2f}x** at k={g['best_k']}"
            if not g["helps"]:
                cell = f"{g['speedup']:.2f}x (worse than not bothering)"
            cells.append(cell)
        cost = by[(name, alphas[0])]["draft_cost"]
        out.append(f"| {name} ({cost:.3g} of a target step) | "
                   + " | ".join(cells) + " |")
    out += ["", "Each cell is the best number of guesses per round and what "
                "it is worth. A round costs k draft steps and one target "
                "step and yields (1 - a^(k+1)) / (1 - a) tokens, so the best "
                "k rises with the acceptance rate and with how cheap the "
                "draft is. Nothing here is measured on hardware: the "
                "acceptance rate is a parameter and the costs are ratios of "
                "decode steps, priced by Chapter 16."]
    return "\n".join(out)


def ch29_reference(d: dict) -> str:
    """What it would be worth on the model the case study serves."""
    r = d["reference"]
    out = ["| Draft | Guesses | Tokens a round | Times faster | "
           "Between tokens |", "|---|---|---|---|---|"]
    for row in r["rows"]:
        if row["alpha"] != 0.8:
            continue
        out.append(f"| {row['draft']} | {row['best_k']} | "
                   f"{row['tokens_per_round']:.2f} | "
                   f"**{row['speedup']:.2f}x** | {row['itl_ms']:.2f} ms |")
    out += ["", f"The book's 8B model, whose decode step is "
                f"{r['baseline_itl_ms']:.2f} ms between tokens without any "
                "of this (Chapter 16's cost model, one sequence, "
                f"{r['context']:,} tokens of context). Every row assumes 80% "
                "of guesses are accepted, which is a stated assumption and "
                "not a measurement -- the acceptance rate depends on the "
                "draft, the target and the traffic, and the only honest way "
                "to get it is to measure the pair you actually have."]
    return "\n".join(out)


def ch41_load(d: dict) -> str:
    """What each offered load does to the server."""
    a, cap = d["assumptions"], d["capacity"]
    theory = {r["rate"]: r for r in d["theory"]["rows"]}
    out = ["| Requests a second | Of capacity | In the system | "
           "End to end, mean | p99 | TTFT p99 | Between tokens, p99 | "
           "Tokens/s | Moves by |", "|---|---|---|---|---|---|---|---|---|"]
    for r in d["sweep"]:
        t = theory[r["rate"]]
        meets = (r["settled"]
                 and r["ttft_p99_ms"] <= a["ttft_budget_ms"]
                 and r["itl_p99_ms"] <= a["itl_budget_ms"])
        rate = (f"**{r['rate']}**" if meets
                else f"{r['rate']} *" if r["settled"] else f"{r['rate']} +")
        out.append(f"| {rate} | {t['utilization'] * 100:.0f}% "
                   f"| {r['in_system']:.0f} | {r['mean_time_s']:.2f} s "
                   f"| {r['p99_s']:.2f} s | {r['ttft_p99_ms']:,.0f} ms "
                   f"| {r['itl_p99_ms']:.1f} ms "
                   f"| {r['tokens_per_s']:,.0f} "
                   f"| {r['worst_drift'] * 100:.0f}% |")
    out += ["", f"One machine, seed {a['seed']}. Every rate was run twice, "
                f"at {a['n_requests']:,} requests and at "
                f"{a['n_requests_long']:,}; the columns are the longer run "
                f"and \"moves by\" is how far the furthest of the mean, "
                f"the p99 and the throughput shifted between the two. "
                f"Bold rows keep both of the case study's promises -- "
                f"{a['ttft_budget_ms']:,} ms to a first token and "
                f"{a['itl_budget_ms']} ms between tokens, at the 99th "
                f"percentile. Rows marked * break at least one. Rows marked "
                f"+ never settled: their latencies grew with the length of "
                f"the run, so they are numbers about the benchmark and not "
                f"about the machine. Capacity is "
                f"{cap['tokens_per_s']:,.0f} tokens a second, which at "
                f"{cap['output_mean']} tokens a reply is "
                f"{cap['requests_per_s']:.1f} requests a second, and "
                f"\"of capacity\" is measured against that."]
    return "\n".join(out)


def ch41_theory(d: dict) -> str:
    """Measured slowdown against what classical queueing predicts."""
    t = d["theory"]
    out = ["| Of capacity | What this server does | "
           "What a classical queue would do | Over-predicted by |",
           "|---|---|---|---|"]
    for r in t["rows"]:
        if r["classical_slowdown"] is None:
            out.append(f"| {r['utilization'] * 100:.0f}% "
                       f"| {r['measured_slowdown']:.2f}x "
                       f"| over capacity | -- |")
        else:
            out.append(f"| {r['utilization'] * 100:.0f}% "
                       f"| {r['measured_slowdown']:.2f}x "
                       f"| {r['classical_slowdown']:.2f}x "
                       f"| **{r['over_prediction']:.1f}x** |")
    out += ["", f"Slowdown is time in the system divided by "
                f"{t['alone_seconds']:.2f} s, which is what one request "
                "takes with the machine to itself. The classical column is "
                "1 / (1 - utilization), the standard result for a "
                "single-server queue and the arithmetic behind every rule "
                "of thumb about not running servers hot. It does not "
                "describe this server, and the last column is how much "
                "hardware believing it would buy."]
    return "\n".join(out)


def ch41_sizing(d: dict) -> str:
    """Three ways to size the same fleet."""
    s_ = d["sizing"]
    names = {
        "throughput_only": "Throughput alone, ignoring the promise",
        "classical_rule_of_thumb": "The 70% rule from classical queueing",
        "measured_promise": "The load at which the promise still holds",
    }
    out = ["| How it was sized | Load a machine | Machines | Cost an hour |",
           "|---|---|---|---|"]
    loads = {
        "throughput_only": s_["per_machine_at_capacity"],
        "classical_rule_of_thumb": s_["per_machine_at_capacity"] * 0.70,
        "measured_promise": s_["highest_rate_meeting_both_promises"],
    }
    for key, label in names.items():
        n = s_["machines"][key]
        cost = s_["usd_per_hour"][key]
        row = f"| {label} | {loads[key]:.1f} req/s | {n} | ${cost:,.2f} |"
        if key == "measured_promise":
            row = (f"| **{label}** | **{loads[key]:.1f} req/s** | **{n}** "
                   f"| **${cost:,.2f}** |")
        out.append(row)
    out += ["", f"For {s_['demand_requests_per_s']} requests a second. The "
                f"promise holds up to {s_['highest_rate_meeting_both_promises']} "
                f"requests a machine, which is "
                f"{s_['utilization_there'] * 100:.0f}% of what the machine "
                "can do -- far past where the classical rule would stop. "
                "Sizing on throughput alone meets no promise at all; sizing "
                "on the rule of thumb buys machines the measurement says "
                "are not needed."]
    return "\n".join(out)


def ch42_prices(d: dict) -> str:
    """The same fleet at six published prices."""
    f = d["full_tilt"]
    fleet = d["fleet"]
    out = ["| Where you rent it | A GPU-hour | The fleet, an hour | "
           "A month | Per million output tokens |", "|---|---|---|---|---|"]
    for r in sorted(f["rows"], key=lambda r: r["usd_per_gpu_hour"]):
        name = r["price_name"]
        if r["usd_per_gpu_hour"] == d["assumptions"]["gpu_usd_per_hour"]:
            name = f"**{name}**"
        out.append(f"| {name} | ${r['usd_per_gpu_hour']:.2f} "
                   f"| ${r['usd_per_hour']:,.2f} "
                   f"| ${r['usd_per_month']:,.0f} "
                   f"| ${r['usd_per_m_tokens']:.3f} |")
    out += ["", f"{fleet['machines']} machines serving "
                f"{fleet['peak_tokens_per_s']:,.0f} output tokens a second, "
                "which is the fleet Chapter 41 sized and the peak it was "
                f"sized for. The spread from cheapest to dearest is "
                f"{f['spread']:.1f}x, for identical hardware doing identical "
                "work. Named prices are from each provider's own page on the "
                "date in FACTS.md; the bold row is the cross-provider median "
                "the rest of this book uses."]
    return "\n".join(out)


def ch42_duty(d: dict) -> str:
    """What a diurnal day does to the average."""
    out = ["| Peak to trough | Average load, as a share of peak | "
           "Hours a day above 80% | What that does to the cost a token |",
           "|---|---|---|---|"]
    base = d["api"]["at_full_tilt"]
    for r in d["duty_cycle"]["rows"]:
        mean = r["mean_over_peak"]
        out.append(f"| {r['peak_to_trough']}:1 | {mean * 100:.0f}% "
                   f"| {r['hours_above_80pct']} "
                   f"| ${base / mean:.3f} per million |")
    out += ["", "A day shaped as a sine between its trough and its peak. "
                "The fleet is sized for the peak and paid for every hour, so "
                "the cost of a token is the full-tilt cost divided by the "
                "average load. Even a gentle two-to-one day adds a third to "
                "the cost of every token; a working-hours service with a "
                "quiet night adds more."]
    return "\n".join(out)


def ch42_own_or_rent(d: dict) -> str:
    """Where owning starts to pay."""
    api = d["api"]
    fleet = d["fleet"]
    out = ["| What you count | Owning, an hour | Breaks even at | "
           "Billed tokens a day there |", "|---|---|---|---|"]
    for name, b in api["breakeven"].items():
        own = next(r["own_usd_per_hour"] for r in api["rows"]
                   if r["overhead"] == name)
        if b is None:
            out.append(f"| {name} | ${own:,.2f} | never | -- |")
        else:
            out.append(f"| {name} | ${own:,.2f} "
                       f"| **{b['utilization'] * 100:.0f}%** of peak "
                       f"| {b['billed_tokens_per_day'] / 1e6:,.0f}M |")
    out += ["", f"Against {api['api_model']} at "
                f"${api['api_usd_per_m_output']:.2f} per million output "
                f"tokens and ${api['api_usd_per_m_input']:.2f} per million "
                f"input. Both sides are dollars an hour for the same "
                f"traffic, which is the only comparison that holds when one "
                f"is billed per token and the other per machine-hour. This "
                f"service sends {api['input_over_output']:.0f} times as many "
                f"tokens in as it takes out -- {fleet['prompt_tokens']:,} of "
                f"prompt against {fleet['output_tokens']} of reply -- and a "
                "provider charges for both, which is why comparing on output "
                "alone gets the answer wrong."]
    return "\n".join(out)


def ch31_validity(d: dict) -> str:
    """What comes out, with the mask and without."""
    v = d["validity"]
    out = ["| The model's own grasp of JSON | With the mask: valid | "
           "finished | Without it: valid | finished |",
           "|---|---|---|---|---|"]
    names = {0.0: "none at all", 1.0: "slight", 2.0: "some",
             4.0: "good", 8.0: "strong"}
    for r in v["rows"]:
        u = r["unconstrained_valid_if_finished"]
        out.append(f"| {names.get(r['skill'], r['skill'])} "
                   f"| **{r['constrained_valid_if_finished'] * 100:.1f}%** "
                   f"| {r['constrained_finished'] * 100:.1f}% "
                   f"| {(u * 100 if u is not None else 0):.1f}% "
                   f"| {r['unconstrained_finished'] * 100:.1f}% |")
    out += ["", f"{v['trials']} documents at each level, cut off at "
                f"{v['max_tokens']} tokens. \"Valid\" is the share of the "
                "documents that *finished* which parse; a walk cut off at "
                "the token limit is incomplete, which is a different failure "
                "and is counted in the column beside it. The model here is a "
                "stand-in whose preference for legal tokens can be turned "
                "up, which is the only property of a real model this "
                "measurement depends on."]
    return "\n".join(out)


def ch31_states(d: dict) -> str:
    """What a grammar compiler turns the grammar into."""
    t = d["table"]
    out = ["| Nesting allowed | States | Distinct masks | States per mask |",
           "|---|---|---|---|"]
    for r in t["rows"]:
        out.append(f"| {r['depth']} | {r['states']:,} | **{r['masks']}** "
                   f"| {r['states_per_mask']:.0f} |")
    out += ["", f"Enumerated exhaustively. The states double with every "
                f"level of nesting, because the stack is a sequence of "
                f"objects and arrays. The masks do not, because what may "
                f"come next depends on the innermost open container and "
                f"never on the ones beneath it -- so all {t['masks']} of "
                f"them fit in {t['bytes_if_packed']:.0f} bytes over this "
                f"{t['vocab']}-token vocabulary, and the work per token at "
                f"serving time is a lookup. Each mask leaves "
                f"{t['allowed_min']} to {t['allowed_max']} tokens open, "
                f"{t['allowed_mean_share'] * 100:.0f}% of the vocabulary on "
                "average: most of what the model could say, at any moment, "
                "it may not."]
    return "\n".join(out)


def ch31_cost(d: dict) -> str:
    """What the mask costs against what a token costs."""
    c = d["cost"]
    out = ["| Per token | Amount | As a share of a decode step |",
           "|---|---|---|"]
    out.append(f"| Additions to the logits | {c['adds_per_token']:,} "
               f"| {c['mask_share_of_decode_flops'] * 100:.6f}% of its "
               f"arithmetic |")
    out.append(f"| Mask read from the table | "
               f"{c['mask_bytes_per_token']:,.0f} bytes "
               f"| {c['mask_share_of_decode_bytes'] * 100:.5f}% of its bytes |")
    out.append(f"| A decode step, for comparison | "
               f"{c['decode_bytes_per_token'] / 1e9:.1f} GB read, "
               f"{c['decode_flops_per_token'] / 1e9:.1f} GFLOP | 100% |")
    out += ["", f"At the reference model's {c['vocab_reference']:,}-token "
                "vocabulary, one bit of mask per token packed. Counted, not "
                "timed: the mask is one lookup and one addition per "
                "vocabulary entry, and a decode step reads every weight in "
                "the model, so both sides are exact. A real implementation "
                "can be slower than this arithmetic -- the lookup has to "
                "find the right row, and with a real tokenizer whose pieces "
                "cut across the grammar, building the table is the hard "
                "part -- but the floor is five orders of magnitude below the "
                "step it rides on."]
    return "\n".join(out)


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    specs = {
        "ch01": (("ch01-cost", ch01_cost),),
        "ch02": (("ch02-shapes", ch02_shapes), ("ch02-cost", ch02_cost)),
        "ch03": (("ch03-measured", ch03_measured), ("ch03-reference", ch03_reference)),
        "ch04": (("ch04-machine", ch04_machine), ("ch04-wall", ch04_wall)),
        "ch05": (("ch05-percentiles", ch05_percentiles), ("ch05-budgets", ch05_budgets)),
        "ch06": (("ch06-breakeven", ch06_breakeven),),
        "ch07": (("ch07-sizes", ch07_sizes), ("ch07-scaling", ch07_scaling)),
        "ch08": (("ch08-points", ch08_points), ("ch08-ceiling", ch08_ceiling)),
        "ch09": (("ch09-framings", ch09_framings),),
        "ch10": (("ch10-agreement", ch10_agreement), ("ch10-baseline", ch10_baseline)),
        "ch11": (("ch11-waste", ch11_waste), ("ch11-growth", ch11_growth)),
        "ch14": (("ch14-capacity", ch14_capacity), ("ch14-blocksize", ch14_blocksize)),
        "ch12": (("ch12-head-to-head", head_to_head), ("ch12-scaling", scaling),
                 ("ch12-memory", memory)),
        "ch13": (("ch13-policies", ch13_policies), ("ch13-traffic", ch13_traffic)),
        "ch15": (("ch15-numerics", ch15_numerics), ("ch15-prefill", ch15_prefill),
                 ("ch15-policies", ch15_policies)),
        "ch16": (("ch16-matmul", ch16_matmul), ("ch16-sweep", ch16_sweep),
                 ("ch16-waste", ch16_waste)),
        "ch17": (("ch17-head-to-head", ch17_head_to_head),
                 ("ch17-load", ch17_load), ("ch17-pool", ch17_pool)),
        "ch18": (("ch18-budget", ch18_budget), ("ch18-policy", ch18_policy),
                 ("ch18-preemption", ch18_preemption), ("ch18-swap", ch18_swap)),
        "ch19": (("ch19-transfer", ch19_transfer), ("ch19-split", ch19_split),
                 ("ch19-links", ch19_links), ("ch19-regimes", ch19_regimes)),
        "ch20": (("ch20-counted", ch20_counted),
                 ("ch20-reference", ch20_reference),
                 ("ch20-tiles", ch20_tiles)),
        "ch22": (("ch22-formats", ch22_formats),
                 ("ch22-tensors", ch22_tensors),
                 ("ch22-hardware", ch22_hardware)),
        "ch24": (("ch24-schemes", ch24_schemes),
                 ("ch24-groups", ch24_groups),
                 ("ch24-memory", ch24_memory)),
        "ch29": (("ch29-acceptance", ch29_acceptance),
                 ("ch29-speedup", ch29_speedup),
                 ("ch29-reference", ch29_reference)),
        "ch41": (("ch41-load", ch41_load), ("ch41-theory", ch41_theory),
                 ("ch41-sizing", ch41_sizing)),
        "ch42": (("ch42-prices", ch42_prices), ("ch42-duty", ch42_duty),
                 ("ch42-own-or-rent", ch42_own_or_rent)),
        "ch31": (("ch31-validity", ch31_validity),
                 ("ch31-states", ch31_states), ("ch31-cost", ch31_cost)),
        "ddr1": (("ddr1-decisions", ddr1_decisions),
                 ("ddr1-would-change", ddr1_would_change),
                 ("ddr1-fleet", ddr1_fleet)),
    }
    for chapter, entries in specs.items():
        path = RESULTS / f"{chapter}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        for name, fn in entries:
            (TABLES / f"{name}.md").write_text(fn(d) + "\n")
            print(f"  {name}.md")


if __name__ == "__main__":
    main()
