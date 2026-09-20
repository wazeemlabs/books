"""Named values for the manuscript, derived from results/*.json.

Prose cannot include a generated table, so prose numbers are written as
`{{name}}` in the chapter source and resolved here. The formatting lives
in code, so a number in a sentence and the same number in a table can
never disagree, and neither can drift from the measurement.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

RESULTS = Path("results")
OUTLINE = Path("../OUTLINE.md")


def chapter_numbers() -> dict[str, int]:
    """Map a slug of each chapter's title to its number, read from the outline.

    Chapters get renumbered as the book takes shape. A reference written
    as a literal "Chapter 14" silently becomes wrong when that happens;
    a reference written as {{ch:paged-attention}} does not. This is the
    lookup behind that.
    """
    text = OUTLINE.read_text()
    out: dict[str, int] = {}
    for m in re.finditer(r"^## (\d+)\. (.+)$", text, re.M):
        slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
        out[slug] = int(m.group(1))
    if not out:
        raise RuntimeError(f"no chapters found in {OUTLINE}")
    return out


def resolve_chapter(ref: str) -> int:
    """Resolve a slug, or any unambiguous fragment of one, to a number."""
    chapters = chapter_numbers()
    if ref in chapters:
        return chapters[ref]
    hits = sorted(n for slug, n in chapters.items() if ref in slug)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError(f"no chapter matches {ref!r}")
    raise KeyError(f"{ref!r} is ambiguous: chapters {hits}")


def growth_exponent(sweep: list[dict], key: str) -> float:
    """Least-squares slope of log(time) against log(tokens): the exponent
    in time ~ n**k over the measured range."""
    x = [math.log(r["n_new"]) for r in sweep]
    y = [math.log(r[key]) for r in sweep]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)


def ch12(d: dict) -> dict[str, str]:
    h, s, e, m = d["head_to_head"], d["sweep"], d["experiment"], d["memory"]
    last, ps = s[-1], d["per_step"]
    prompt, longest = e["prompt_len"], ps["n_new"]
    ref, tiny = m["reference_8b"], m["tinyserve"]

    seq_first, seq_last = prompt + 1, prompt + longest - 1
    step_ratio = ps["naive_s"][-1] / ps["naive_s"][1]
    noisy = sum(1 for r in s for k in ("naive_noisy", "cached_noisy") if r[k])

    # An 8B model in bf16 on an 80 GB accelerator.
    gpu_gb, weights_gb = 80, 16
    free_gb = gpu_gb - weights_gb
    seqs_8k = int(free_gb * 1000**3 / (ref["bytes_per_token"] * 8192))

    return {
        "params": f"{d['model']['params']:,}",
        "runs": str(e["runs"]),
        "warmup": str(e["warmup"]),
        "prompt_len": str(prompt),
        "n_new": str(e["n_new"]),
        "longest": str(longest),
        # headline
        "speedup_total": f"{h['speedup_total']:.0f}x",
        "speedup_total_spread": f"{h['speedup_total_stats']['spread_frac'] * 100:.0f}%",
        "ttft_naive_ms": f"{h['naive']['ttft_s']['median'] * 1e3:.1f} ms",
        "ttft_cached_ms": f"{h['cached']['ttft_s']['median'] * 1e3:.1f} ms",
        # the long run
        "naive_longest_s": f"{last['naive_s']:.1f} seconds",
        "cached_longest_s": f"{last['cached_s']:.2f} seconds",
        "speedup_longest": f"{last['speedup']:.0f}x",
        "flop_ratio_longest": f"{last['flop_ratio']:.0f}x",
        "efficiency_gap": f"{last['flop_ratio'] / last['speedup']:.1f}",
        # shape of the curves
        "step_first_ms": f"{ps['naive_s'][1] * 1e3:.0f} ms",
        "step_last_ms": f"{ps['naive_s'][-1] * 1e3:.0f} ms",
        "step_ratio": f"{step_ratio:.2f}",
        "seq_first": f"{seq_first:,}",
        "seq_last": f"{seq_last:,}",
        "seq_ratio": f"{seq_last / seq_first:.2f}",
        "cached_step_ms": f"{sorted(ps['cached_s'][1:])[len(ps['cached_s']) // 2] * 1e3:.2f} ms",
        "exp_naive": f"{growth_exponent(s, 'naive_s'):.2f}",
        "exp_cached": f"{growth_exponent(s, 'cached_s'):.2f}",
        # memory
        "kv_bytes_tiny": f"{tiny['bytes_per_token']:,} bytes",
        "kv_kib_8b": f"{ref['kib_per_token']:.0f} KiB",
        "kv_gib_8k": f"{ref['gib_per_8k_sequence']:.1f} GiB",
        "gpu_gb": f"{gpu_gb} GB",
        "weights_gb": f"{weights_gb} GB",
        "free_gb": f"{free_gb} GB",
        "seqs_8k": str(seqs_8k),
        "n_layers": str(d["model"]["config"]["n_layers"]),
        "n_kv_heads": str(d["model"]["config"]["n_kv_heads"]),
        "head_dim": str(d["model"]["head_dim"]),
        # honesty about the machine
        "noisy_count": str(noisy),
        "sweep_measurements": str(len(s) * 2),
        "cpu": d["provenance"]["hardware"]["cpu"],
        "cores": str(d["provenance"]["hardware"]["cores_available"]),
    }


def ch01(d: dict) -> dict[str, str]:
    a, api, sp = d["assumptions"], d["api_reference"], d["spread"]
    lo, hi = d["batches"][0], d["batches"][-1]
    # The case study: 200 req/s x 300 output tokens, run for a day.
    tok_per_s = 200 * 300
    tok_per_day_m = tok_per_s * 86400 / 1e6
    return {
        "gpu": a["gpu"],
        "gpu_price": f"${a['gpu_usd_per_hour']:.2f}",
        "bandwidth": f"{a['hbm_bytes_per_s'] / 1e12:.2f} TB/s",
        "params_b": f"{a['params'] / 1e9:.0f}B",
        "weight_gb": f"{a['weight_bytes'] / 1e9:.0f} GB",
        "seq_len": f"{a['seq_len']:,}",
        "max_batch": str(a["max_concurrent"]),
        "worst_cost": f"${sp['worst_usd_per_m']:.2f}",
        "best_cost": f"${sp['best_usd_per_m']:.3f}",
        "spread": f"{sp['ratio']:.0f}x",
        "worst_tps": f"{lo['tokens_per_s']:,.0f}",
        "best_tps": f"{hi['tokens_per_s']:,.0f}",
        "worst_util": f"{lo['flop_utilization'] * 100:.1f}%",
        "best_util": f"{hi['flop_utilization'] * 100:.0f}%",
        "best_itl": f"{hi['step_ms']:.0f} ms",
        "kv_share": f"{(1 - hi['weight_share_of_bytes']) * 100:.0f}%",
        "api_out": f"${api['output_usd_per_m']:.2f}",
        "api_in": f"${api['input_usd_per_m']:.2f}",
        "best_vs_api": f"{sp['best_vs_api']:.1f}x",
        "day_tokens_m": f"{tok_per_day_m:,.0f}",
        "day_tokens": f"{tok_per_s * 86400 / 1e9:.1f} billion",
        "day_worst": f"${tok_per_day_m * sp['worst_usd_per_m']:,.0f}",
        "day_best": f"${tok_per_day_m * sp['best_usd_per_m']:,.0f}",
        "year_gap": f"${(tok_per_day_m * (sp['worst_usd_per_m'] - sp['best_usd_per_m']) * 365) / 1e6:,.1f} million",
    }


def ch02(d: dict) -> dict[str, str]:
    cfg, v, nt = d["model"]["config"], d["vocabulary"], d["next_token"]
    t, r = d["cost"]["tiny"], d["cost"]["reference_8b"]
    words = d["prompt"]["words"]
    last_row = d["layers"][0]["attention_head_1"][-1]
    top = nt["top_k"][0]
    return {
        "prompt": " ".join(words),
        "tokens": str(d["prompt"]["tokens"]),
        "n_tokens": str(len(words)),
        "vocab": str(v["size"]),
        "params": f"{d['model']['params']:,}",
        "d_model": str(cfg["d_model"]),
        "n_layers": str(cfg["n_layers"]),
        "n_heads": str(cfg["n_heads"]),
        "head_dim": str(d["model"]["head_dim"]),
        "chosen": nt["chosen"],
        "chosen_prob": f"{top['probability'] * 100:.1f}%",
        "uniform_prob": f"{100 / v['size']:.1f}%",
        # The worked reading of the last row of the attention figure.
        "row_words": ", ".join(f"{w} {x:.2f}" for w, x in zip(words, last_row)),
        "row_sum": f"{sum(last_row):.2f}",
        "row_biggest_word": words[max(range(len(last_row)), key=lambda i: last_row[i])],
        "row_biggest": f"{max(last_row):.2f}",
        "tiny_mb": f"{t['weight_bytes_read_per_token'] / 1e6:.1f} MB",
        "tiny_kv": f"{t['kv_bytes_per_token']:,} bytes",
        "ref_gb": f"{r['weight_bytes_read_per_token'] / 1e9:.0f} GB",
        "ref_kv": f"{r['kv_bytes_per_token'] / 1024:.0f} KiB",
        "ref_flops": f"{r['flops_per_token'] / 1e9:.0f} billion",
        "ffn_share": f"{d['parameter_split']['feed_forward_share'] * 100:.0f}%",
        "ref_floor_ms": f"{r['decode_floor_ms']:.1f} milliseconds",
    }


def ch03(d: dict) -> dict[str, str]:
    r, rows = d["reference_8b"], d["measured"]
    o, pre, dec = r["one_request"], r["prefill"], r["decode"]
    gaps = [x["decode_vs_prefill_per_token"] for x in rows]
    m = d["model"]
    return {
        "params": f"{m['params']:,}",
        "weight_mb": f"{m['params'] * 4 / 1e6:.1f} MB",
        "prompts": ", ".join(f"{x['prompt']:,}" for x in rows),
        "gap_lo": f"{min(gaps):.1f}x",
        "gap_hi": f"{max(gaps):.1f}x",
        "ref_params": f"{r['config']['params'] / 1e9:.0f}B",
        "ref_prompt": f"{r['config']['prompt']:,}",
        "ref_context": f"{r['config']['context']:,}",
        "prefill_intensity": f"{pre['intensity']:,.0f}",
        "decode_intensity": f"{dec['intensity']:.2f}",
        "intensity_ratio": f"{r['intensity_ratio']:,.0f}x",
        "time_ratio": f"{r['time_per_token_ratio']:,.0f}x",
        "prefill_per_token_us": f"{r['prefill_per_token_s'] * 1e6:.0f}",
        "ridge": f"{r['hardware']['ridge_flop_per_byte']:.0f}",
        "prefill_bound": pre["bound_by"],
        "decode_bound": dec["bound_by"],
        "decode_below_ridge": f"{1 / dec['distance_from_ridge']:,.0f}x",
        "prefill_above_ridge": f"{pre['distance_from_ridge']:.1f}x",
        "prefill_ms": f"{o['prefill_s'] * 1e3:.0f} ms",
        "decode_ms": f"{dec['seconds'] * 1e3:.2f} ms",
        "decode_total_s": f"{o['decode_s']:.2f} seconds",
        "total_s": f"{o['total_s']:.2f} seconds",
        "decode_share": f"{o['decode_share'] * 100:.0f}%",
        "out_tokens": str(o["output_tokens"]),
    }


def ch04(d: dict) -> dict[str, str]:
    m, a, w = d["machine"], d["accelerator"], d["wall"]
    levels = {c["level"]: c for c in d["cache_topology"]}
    last = max(levels) if levels else 3
    return {
        "dram_gb_s": f"{m['dram_bytes_per_s'] / 1e9:.0f} GB/s",
        "fastest_gb_s": f"{m['fastest_bytes_per_s'] / 1e9:.0f} GB/s",
        "cache_advantage": f"{m['cache_advantage']:.1f}x",
        "flops": f"{m['flops'] / 1e9:.0f} GFLOP/s",
        "flops_spread": f"{m['flops_spread_frac'] * 100:.0f}%",
        "ridge": f"{m['ridge_flop_per_byte']:.0f}",
        "acc_ridge": f"{a['ridge_flop_per_byte']:.0f}",
        "ridge_ratio": f"{a['ridge_ratio_vs_machine']:.0f}x",
        "acc_bw": f"{a['hbm_bytes_per_s'] / 1e12:.2f} TB/s",
        "acc_flops": f"{a['peak_bf16_flops'] / 1e12:.0f} TFLOP/s",
        "weight_gb": f"{a['weight_bytes_8b'] / 1e9:.0f} GB",
        "decode_floor": f"{a['decode_floor_ms']:.1f} ms",
        "decode_floor_tps": f"{1000 / a['decode_floor_ms']:.0f}",
        "l1_kib": f"{levels[1]['kib']:,} KiB" if 1 in levels else "UNVERIFIED",
        "l2_kib": f"{levels[2]['kib'] // 1024:,} MiB" if 2 in levels else "UNVERIFIED",
        "l3_mib": f"{levels[last]['kib'] // 1024:,} MiB" if last in levels else "UNVERIFIED",
        "small_mib": f"{w[0]['weight_mib']:.1f} MiB",
        "small_ratio": f"{w[0]['measured_over_predicted']:.1f}x",
        "big_mib": f"{w[-1]['weight_mib']:,.0f} MiB",
        "big_ratio": f"{w[-1]['measured_over_predicted']:.2f}x",
        "big_ms": f"{w[-1]['decode_step_s'] * 1e3:.0f} ms",
        "size_span": f"{w[-1]['weight_mib'] / w[0]['weight_mib']:.0f}x",
    }


def ch05(d: dict) -> dict[str, str]:
    m, cs, c = d["measured"], d["case_study"], d["curve"]
    tight = min(d["budgets"], key=lambda b: b["itl_budget_ms"])
    binding = [b for b in d["budgets"] if not b["limited_by_memory_not_budget"]]
    loosest_binding = max(binding, key=lambda b: b["itl_budget_ms"]) if binding else tight
    return {
        "samples": f"{m['samples']:,}",
        "mean_ms": f"{m['mean_ms']:.2f} ms",
        "p50_ms": f"{m['p50_ms']:.2f} ms",
        "p99_ms": f"{m['p99_ms']:.2f} ms",
        "p999_ms": f"{m['p999_ms']:.2f} ms",
        "max_ms": f"{m['max_ms']:.2f} ms",
        "tail_ratio": f"{m['p99_over_p50']:.1f}x",
        "mean_ratio": f"{m['mean_over_p50']:.2f}x",
        "batch1_tps": f"{c[0]['tokens_per_s']:,.0f}",
        "batch1_itl": f"{c[0]['inter_token_ms']:.1f} ms",
        "batchmax": str(c[-1]["batch"]),
        "batchmax_tps": f"{c[-1]['tokens_per_s']:,.0f}",
        "batchmax_itl": f"{c[-1]['inter_token_ms']:.0f} ms",
        "throughput_gain": f"{c[-1]['tokens_per_s'] / c[0]['tokens_per_s']:.0f}x",
        "latency_cost": f"{c[-1]['inter_token_ms'] / c[0]['inter_token_ms']:.0f}x",
        "tight_budget": f"{tight['itl_budget_ms']} ms",
        "tight_batch": str(tight["largest_batch"]),
        "binding_budget": f"{loosest_binding['itl_budget_ms']} ms",
        "ttft_ms": f"{cs['achieved_ttft_ms']:.0f} ms",
        "ttft_allowed": f"{cs['ttft_ms']:,} ms",
        "ttft_headroom": f"{cs['ttft_headroom']:.0f}x",
        "case_itl": f"{cs['itl_ms']} ms",
        "case_batch": str(cs["operating_batch"]),
        "case_achieved_itl": f"{cs['achieved_itl_ms']:.0f} ms",
        "case_tps": f"{cs['tokens_per_s']:,.0f}",
        "reply_s": f"{cs['reply_seconds']:.1f} seconds",
        "words_per_s": f"{1000 / cs['achieved_itl_ms']:.0f}",
        "output": str(cs["output"]),
        "prompt": f"{cs['prompt']:,}",
    }


def ch06(d: dict) -> dict[str, str]:
    sc = {(s["precision"], s["pricing"]): s for s in d["scenarios"]}
    naive = sc[("bf16 (two bytes)", "on-demand median")]
    best = sc[("fp8 (one byte)", "cheapest marketplace")]
    fp8_od = sc[("fp8 (one byte)", "on-demand median")]
    a, eng = d["assumptions"], d["assumptions"]["engineering_multiple"]
    return {
        "api_price": f"${d['api']['usd_per_m_output']:.2f}",
        "naive_cost": f"${naive['usd_per_m_at_full']:.3f}",
        "naive_needed": f"{naive['breakeven_utilization'] * 100:.0f}%",
        "fp8_od_cost": f"${fp8_od['usd_per_m_at_full']:.3f}",
        "fp8_od_needed": f"{fp8_od['breakeven_utilization'] * 100:.0f}%",
        "best_cost": f"${best['usd_per_m_at_full']:.3f}",
        "best_needed": f"{best['breakeven_utilization'] * 100:.0f}%",
        "n_achievable": str(d["summary"]["n_achievable"]),
        "n_total": str(d["summary"]["n_total"]),
        "eng_low": str(eng[0]), "eng_high": str(eng[1]),
        "batch": str(a["batch"]),
        "on_demand": f"${a['prices_usd_per_hour']['on-demand median']:.2f}",
        "marketplace": f"${a['prices_usd_per_hour']['cheapest marketplace']:.2f}",
        "tokens_day": f"{naive['tokens_per_s'] * 86400 / 1e9:.1f} billion",
        "at_ten_pct": f"${naive['usd_per_m_at_full'] / 0.1:.2f}",
    }


def ch07(d: dict) -> dict[str, str]:
    dev, sizes, sc = d["device"], d["sizes"], d["scaling"]
    small, big = sizes[0], max(sizes, key=lambda r: r["share_of_peak"])
    last = sc[-1]
    return {
        "cpu": dev["name"],
        "cores": str(dev["cores_visible"]),
        "small_n": str(small["n"]),
        "small_share": f"{small['share_of_peak'] * 100:.0f}%",
        "small_gflops": f"{small['gflops']:,.0f}",
        "big_n": str(big["n"]),
        "peak_gflops": f"{d['peak_gflops']:,.0f}",
        "threads": str(last["threads"]),
        "arith_speedup": f"{last['gflops_speedup']:.1f}x",
        "mem_speedup": f"{last['bandwidth_speedup']:.2f}x",
        "d_model": f"{d['decode_shape']['k']:,}",
        "acc_flops": f"{d['accelerator']['peak_bf16_flops'] / 1e12:.0f} TFLOP/s",
        "acc_ridge": f"{d['accelerator']['ridge_flop_per_byte']:.0f}",
        "sms": str(d["accelerator"]["streaming_multiprocessors"]),
        "one_sm_share": f"{d['accelerator']['one_sm_share'] * 100:.1f}%",
    }


def ch08(d: dict) -> dict[str, str]:
    m, a = d["machine"], d["accelerator"]
    by = {p["name"]: p for p in d["points"]}
    pre, dec = by["tinyserve prefill"], by["tinyserve decode"]
    small = by["16x16 multiply"]
    rows = a["batching"]
    return {
        "peak": f"{m['peak_flops'] / 1e9:,.0f} GFLOP/s",
        "bandwidth": f"{m['bandwidth_bytes_per_s'] / 1e9:.1f} GB/s",
        "ridge": f"{m['ridge_flop_per_byte']:.0f}",
        "prefill_intensity": f"{pre['intensity']:,.0f}",
        "prefill_share": f"{pre['fraction_of_roof'] * 100:.0f}%",
        "decode_intensity": f"{dec['intensity']:.2f}",
        "decode_share": f"{dec['fraction_of_roof'] * 100:.0f}%",
        "small_share": f"{small['fraction_of_roof'] * 100:.0f}%",
        "acc_ridge": f"{a['ridge_flop_per_byte']:.0f}",
        "acc_peak": f"{a['peak_flops'] / 1e12:.0f} TFLOP/s",
        "acc_bw": f"{a['bandwidth_bytes_per_s'] / 1e12:.2f} TB/s",
        "if_free": f"{a['tokens_in_flight_if_cache_were_free']:.0f}",
        "ceiling": f"{a['intensity_ceiling']:.0f}",
        "short_by": f"{a['ridge_flop_per_byte'] / a['intensity_ceiling']:.1f}x",
        "batch1_intensity": f"{rows[0]['intensity']:.2f}",
        "batchmax": str(rows[-1]["batch"]),
        "batchmax_intensity": f"{rows[-1]['intensity']:.0f}",
        "intensity_gain": f"{rows[-1]['intensity'] / rows[0]['intensity']:.0f}x",
    }


def ch13(d: dict) -> dict[str, str]:
    e, t, h, m = d["experiment"], d["traffic"], d["hardware"], d["measured_tinyserve"]
    p = {r["policy"]: r for r in d["policies"]}
    v = {
        "n_requests": f"{e['n_requests']:,}",
        "seed": str(e["seed"]),
        "max_model_len": f"{e['max_model_len']:,}",
        "max_new": f"{e['max_new']:,}",
        "block": str(e["block"]),
        "prompt_p50": f"{t['prompt_p50']:,.0f}",
        "output_p50": f"{t['output_p50']:,.0f}",
        "total_p50": f"{t['total_p50']:,.0f}",
        "total_p99": f"{t['total_p99']:,.0f}",
        "free_gb": f"{h['free_bytes'] / 1000**3:.0f} GB",
        "kv_kib": f"{h['kv_bytes_per_token'] / 1024:.0f} KiB",
        "tiny_alloc_mib": f"{m['allocated_bytes'] / 1024**2:.0f} MiB",
        "tiny_used_mib": f"{m['used_bytes'] / 1024**2:.0f} MiB",
        "tiny_util": f"{m['utilization'] * 100:.0f}%",
        "tiny_max_seq": f"{m['max_seq']:,}",
        "gain": f"{p['paged_16']['concurrent_seqs'] / p['max_model_len']['concurrent_seqs']:.1f}x",
        "oracle_gap": f"{p['oracle']['concurrent_seqs'] - p['paged_16']['concurrent_seqs']}",
    }
    for key, short in (("max_model_len", "maxlen"), ("prompt_plus_cap", "cap"),
                       ("paged_16", "paged"), ("oracle", "oracle")):
        r = p[key]
        v[f"util_{short}"] = f"{r['utilization'] * 100:.0f}%"
        v[f"waste_{short}"] = f"{r['waste'] * 100:.0f}%"
        v[f"conc_{short}"] = str(r["concurrent_seqs"])
        v[f"held_{short}"] = f"{r['reserved_tokens_mean']:,.0f}"
        v[f"mib_{short}"] = f"{r['bytes_per_seq'] / 1024**2:,.0f} MiB"
    return v


def load(chapter: str = "ch12") -> dict[str, str]:
    d = json.loads((RESULTS / f"{chapter}.json").read_text())
    return {"ch01": ch01, "ch02": ch02, "ch03": ch03, "ch04": ch04,
            "ch05": ch05, "ch06": ch06, "ch07": ch07, "ch08": ch08,
            "ch12": ch12, "ch13": ch13}[chapter](d)


if __name__ == "__main__":
    for k, v in load().items():
        print(f"{k:24} {v}")
