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
        "ch12": (("ch12-head-to-head", head_to_head), ("ch12-scaling", scaling),
                 ("ch12-memory", memory)),
        "ch13": (("ch13-policies", ch13_policies), ("ch13-traffic", ch13_traffic)),
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
