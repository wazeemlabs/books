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
