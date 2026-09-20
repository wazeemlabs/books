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


def ch09(d: dict) -> dict[str, str]:
    e, sp, dist = d["experiment"], d["spread"], d["distribution"]
    by = {w["how"]: w for w in d["ways"]}
    best = by["Best single run"]
    return {
        "samples": f"{e['samples']:,}",
        "short": f"{e['short_context']:,}",
        "long": f"{e['long_context']:,}",
        "honest": f"{d['honest_tokens_per_s']:,.0f}",
        "spread": f"{sp['ratio']:.1f}x",
        "flattering": f"{sp['flattering_tokens_per_s']:,.0f}",
        "damning": f"{sp['damning_tokens_per_s']:,.0f}",
        "best_run_gain": f"{best['relative_to_honest']:.2f}x",
        "cold_penalty": f"{d['cold_penalty']:.2f}x",
        "tail_penalty": f"{d['tail_ratio']:.2f}x",
        "context_penalty": f"{d['context_penalty']:.2f}x",
        "p50_ms": f"{dist['p50']:.2f} ms",
        "p99_ms": f"{dist['p99']:.2f} ms",
        "min_ms": f"{dist['min']:.2f} ms",
    }


def ch10(d: dict) -> dict[str, str]:
    a, b, sp = d["agreement"], d["baseline"], d["speed"]
    wc, nc = b["with_cache"], b["no_cache"]
    fastest = min(sp, key=lambda r: r["torch_over_tinyserve"])
    return {
        "cases": str(len(a["cases"])),
        "worst_diff": f"{a['worst_absolute_difference']:.1e}",
        "all_agree": "yes" if a["all_agree"] else "NO",
        "torch": a["torch_version"],
        "tokens_compared": str(a["tokens_compared"]),
        "torch_gain": f"{1 / fastest['torch_over_tinyserve']:.0f}x",
        "torch_at": f"{fastest['tokens']}",
        "params": f"{b['params']:,}",
        "weight_mb": f"{b['weight_bytes'] / 1e6:.1f} MB",
        "prompt": str(b["prompt"]), "generated": str(b["generated"]),
        "no_cache_tps": f"{nc['tokens_per_s']:,.0f}",
        "cache_tps": f"{wc['tokens_per_s']:,.0f}",
        "ttft_ms": f"{wc['ttft_ms']:.1f} ms",
        "p50_ms": f"{wc['decode_p50_ms']:.2f} ms",
        "p99_ms": f"{wc['decode_p99_ms']:.2f} ms",
        "kv_bytes": f"{wc['kv_bytes_per_token']:,}",
    }


def ch11(d: dict) -> dict[str, str]:
    e, su, w, sw = d["experiment"], d["summary"], d["waste"], d["sweep"]
    return {
        "prompt": str(e["prompt"]),
        "longest": str(e["longest"]),
        "first_waste": f"{w[0]['wasted_fraction'] * 100:.2f}%",
        "last_waste": f"{w[-1]['wasted_fraction'] * 100:.2f}%",
        "useful_share": f"{su['useful_share'] * 100:.2f}%",
        "first_step_ms": f"{su['first_step_ms']:.1f} ms",
        "last_step_ms": f"{su['last_step_ms']:.1f} ms",
        "step_growth": f"{su['step_growth']:.2f}",
        "seq_growth": f"{su['sequence_growth']:.2f}",
        "short_per_token": f"{sw[0]['seconds_per_token'] * 1e3:.1f} ms",
        "long_per_token": f"{sw[-1]['seconds_per_token'] * 1e3:.1f} ms",
        "short_n": str(sw[0]["generated"]),
        "long_n": str(sw[-1]["generated"]),
        "long_total": f"{sw[-1]['total_s']:.1f} seconds",
        "params": f"{d['model']['params']:,}",
    }


def ch14(d: dict) -> dict[str, str]:
    dflt, sc, a = d["at_default"], d["step_cost"], d["assumptions"]
    sizes = d["block_sizes"]
    return {
        "block": str(d["default_block_size"]),
        "same_tokens": "yes" if d["equivalence"]["same_tokens"] else "NO",
        "compared": str(d["equivalence"]["tokens_compared"]),
        "contiguous_fit": str(dflt["admitted_contiguous"]),
        "paged_fit": str(dflt["admitted_paged"]),
        "gain": f"{dflt['gain']:.1f}x",
        "utilization": f"{dflt['utilization'] * 100:.1f}%",
        "waste_per_seq": f"{dflt['wasted_tokens_per_sequence']:.1f}",
        "worst_waste": str(dflt["worst_case_waste_per_sequence"]),
        "overhead": f"{sc['paged_over_contiguous']:.2f}x",
        "contiguous_ms": f"{sc['contiguous']['median'] * 1e3:.2f} ms",
        "paged_ms": f"{sc['paged']['median'] * 1e3:.2f} ms",
        "pool_gb": f"{a['pool_gb']:.0f} GB",
        "returned": str(d["reuse"]["returned"]),
        "big_block": str(sizes[-1]["block_size"]),
        "big_waste": f"{sizes[-1]['wasted_tokens_per_sequence']:.0f}",
        "big_util": f"{sizes[-1]['utilization'] * 100:.0f}%",
        "requests": f"{a['requests_sampled']:,}",
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


def ch15(d: dict) -> dict[str, str]:
    e, pf, tr, u, a = (d["equivalence"], d["prefill"], d["trace"],
                       d["unlimited"], d["assumptions"])
    pts = pf["points"]
    best, half = pts[-1], next(p for p in pts if p["shared_frac"] == 0.5)
    by = {(r["policy"], r["pool_blocks"]): r for r in d["sizes"]}
    lru = sorted((r for r in d["sizes"] if r["policy"] == "lru"),
                 key=lambda r: r["pool_blocks"])
    lfu = sorted((r for r in d["sizes"] if r["policy"] == "lfu"),
                 key=lambda r: r["pool_blocks"])
    un = sorted((r for r in d["sizes"] if r["policy"] == "unstructured"),
                key=lambda r: r["pool_blocks"])
    gb = a["gb_per_block"]
    # The smallest cache that is within one point of the ceiling.
    enough = next(r for r in lru if r["hit_rate"] >= u["hit_rate"] - 0.01)
    return {
        "block": str(a["block"]),
        "same_tokens": "yes" if e["same_tokens"] else "NO",
        "compared": str(e["tokens_compared"]),
        "skipped": str(e["tokens_skipped"]),
        "logit_diff": f"{e['max_logit_diff']:.0e}".replace("e-0", "e-"),
        "key_diff": f"{e['cached_key_diff']:.0e}".replace("e-0", "e-"),
        "softmax_diff": f"{d['numerics']['softmax_diff']:.0e}".replace("e-0", "e-"),
        "prompt_tokens": str(pf["prompt_tokens"]),
        "runs": str(pf["runs"]),
        "warmup": str(pf["warmup"]),
        "half_speedup": f"{half['speedup']:.2f}x",
        "best_shared": f"{best['shared_frac'] * 100:.0f}%",
        "best_computed": str(best["tokens_computed"]),
        "best_speedup": f"{best['speedup']:.0f}x",
        "best_proportional": f"{best['speedup_if_proportional']:.0f}x",
        "best_cold_ms": f"{best['cold_s']['median'] * 1e3:.0f} ms",
        "best_warm_ms": f"{best['warm_s']['median'] * 1e3:.1f} ms",
        "requests": f"{tr['requests']:,}",
        "sessions": str(tr["sessions"]),
        "follow_ups": str(tr["follow_up_turns"]),
        "system_prompts": str(tr["system_prompts"]),
        "trace_prompt_tokens": f"{tr['prompt_tokens']:,}",
        "prompt_p50": f"{tr['prompt_p50']:,.0f}",
        "hit_rate": f"{u['hit_rate'] * 100:.0f}%",
        "ideal_rate": f"{u['ideal_rate'] * 100:.1f}%",
        "computed_tokens": f"{u['computed_tokens']:,}",
        "quant_loss": f"{u['quantization_loss_mean']:.0f}",
        "working_set": f"{u['working_set_blocks']:,}",
        "working_gb": f"{u['working_set_gb']:.0f} GB",
        "pool_gb": f"{a['pool_bytes'] / 1000**3:.0f} GB",
        "enough_share": f"{enough['pool_blocks'] * gb * 1000**3 / a['pool_bytes'] * 100:.0f}%",
        "dedup": f"{u['dedup']:.1f}x",
        "index_us": f"{u['index_us_per_request']:.0f}",
        "index_us_block": f"{u['index_us_per_block']:.1f}",
        "cold_points": str(len(pts)),
        "cold_spread": f"{(max(p['cold_s']['median'] for p in pts) - min(p['cold_s']['median'] for p in pts)) / 2 / sorted(p['cold_s']['median'] for p in pts)[len(pts) // 2] * 100:.0f}%",
        "small_gb": f"{lru[0]['pool_blocks'] * gb:.1f} GB",
        "small_hit": f"{lru[0]['hit_rate'] * 100:.0f}%",
        "enough_gb": f"{enough['pool_blocks'] * gb:.0f} GB",
        "enough_hit": f"{enough['hit_rate'] * 100:.0f}%",
        "enough_frac": f"{enough['pool_blocks'] / u['working_set_blocks'] * 100:.0f}%",
        "lfu_gap": f"{(lru[2]['hit_rate'] - lfu[2]['hit_rate']) * 100:.0f}",
        "lfu_at": f"{lru[2]['pool_blocks'] * gb:.1f} GB",
        "unstructured_gap": f"{(lru[0]['hit_rate'] - un[0]['hit_rate']) * 100:.0f}",
        "stranded": f"{max(r['stranded_blocks'] for r in un):,}",
    }


def ch16(d: dict) -> dict[str, str]:
    e, mc = d["experiment"], d["machine"]
    big = next(m for m in d["models"] if m["name"] == "memory-resident")
    small = next(m for m in d["models"] if m["name"] == "cache-resident")
    mm = next(m for m in d["matmuls"] if m["name"] == "memory-resident")
    mrows = {r["batch"]: r for r in mm["rows"]}
    brows = {r["batch"]: r for r in big["rows"]}
    srows = {r["batch"]: r for r in small["rows"]}
    waste = {w["batch"]: w for w in d["waste"]}
    ref = {r["batch"]: r for r in d["reference_8b"]}
    cs = d["case_study"]
    fleet = {f["batch"]: f for f in cs["fleet"]}
    last = max(brows)

    flat_to = mm["flat_to"]
    return {
        "batches": ", ".join(str(b) for b in e["batches"]),
        "same_tokens": "yes" if d["equivalence"]["same_tokens"] else "NO",
        "compared": str(d["equivalence"]["tokens_compared"]),
        "sequences": str(d["equivalence"]["sequences"]),
        "score_diff": f"{d['equivalence']['agreement']['worst']:.0e}".replace("e-0", "e-"),
        "bandwidth": f"{mc['dram_bytes_per_s'] / 1e9:.0f} GB/s",
        "peak_flops": f"{mc['flops'] / 1e9:,.0f} GFLOP/s",
        "ridge": f"{mc['ridge_flop_per_byte']:.0f}",
        "acc_ridge": f"{mc['accelerator_ridge_flop_per_byte']:.0f}",
        # the isolated matmul
        "mm_shape": f"{mm['shape'][0]}x{mm['shape'][1]}",
        "mm_mib": f"{mm['weight_bytes'] / 1024**2:.0f} MiB",
        "mm_one_ms": f"{mrows[1]['seconds'] * 1e3:.2f} ms",
        "mm_two_ms": f"{mrows[2]['seconds'] * 1e3:.2f} ms",
        "mm_two_over_one": f"{mrows[2]['seconds'] / mrows[1]['seconds']:.1f}x",
        "mm_flat_to": str(flat_to),
        "mm_flat_gain": f"{flat_to // 2}x",
        "mm_flat_ms": f"{mrows[flat_to]['seconds'] * 1e3:.2f} ms",
        "mm_working_set": f"{mm['working_set_bytes'] / 1024**2:.0f} MiB",
        "mm_matrices": str(mm["matrices"]),
        "cache_mib": f"{d['cache_bytes'] / 1024**2:.0f} MiB",
        "pack_gemm_ms": f"{d['packing']['matrix_matrix_s'] * 1e3:.2f} ms",
        "pack_gemv_ms": f"{d['packing']['two_matrix_vector_s'] * 1e3:.2f} ms",
        "pack_gap_ms": f"{d['packing']['unexplained_s'] * 1e3:.2f} ms",
        "pack_copy_ms": f"{d['packing']['half_a_copy_s'] * 1e3:.2f} ms",
        "mm_last_ms": f"{mrows[last]['seconds'] * 1e3:.2f} ms",
        "mm_rate_low": f"{mrows[2]['flops'] / 1e9:.0f}",
        "mm_rate_high": f"{mrows[last]['flops'] / 1e9:.0f}",
        "mm_rate_gain": f"{mrows[last]['flops'] / mrows[2]['flops']:.0f}x",
        "mm_rate_share": f"{mrows[last]['flops'] / mc['flops'] * 100:.0f}%",
        "mm_per_token_gain": f"{mrows[last]['per_token_speedup']:.0f}x",
        # the whole step
        "big_mib": f"{big['weight_mib']:.0f} MiB",
        "big_bw": f"{big['bandwidth_bytes_per_s'] / 1e9:.1f} GB/s",
        "small_mib": f"{small['weight_mib']:.1f} MiB",
        "itl_one": f"{brows[1]['inter_token_ms']:.0f} ms",
        "itl_last": f"{brows[last]['inter_token_ms']:.0f} ms",
        "itl_growth": f"{brows[last]['inter_token_ms'] / brows[1]['inter_token_ms']:.0f}x",
        "tok_one": f"{brows[1]['tokens_per_s']:,.0f}",
        "tok_last": f"{brows[last]['tokens_per_s']:,.0f}",
        "big_gain": f"{brows[last]['throughput_gain']:.1f}x",
        "small_gain": f"{srows[last]['throughput_gain']:.1f}x",
        "shared_share": f"{brows[last]['shared_s'] / brows[last]['step_s']['median'] * 100:.0f}%",
        "private_growth": f"{brows[last]['private_s'] / brows[1]['private_s']:.0f}x",
        "last_batch": str(last),
        "knee": str(big["knee_batch"]),
        # what a static batch wastes
        "waste_8": f"{waste[8]['total_utilization'] * 100:.0f}%",
        "waste_last": f"{waste[last]['total_utilization'] * 100:.0f}%",
        "prompt_waste_last": f"{waste[last]['prompt_utilization'] * 100:.0f}%",
        "output_waste_last": f"{waste[last]['output_utilization'] * 100:.0f}%",
        "steps_run": f"{waste[last]['steps_per_batch']:,.0f}",
        "steps_needed": f"{waste[last]['useful_steps_per_sequence']:,.0f}",
        "requests": f"{e['n_requests']:,}",
        "pic_batch": str(len(d["picture"])),
        "pic_short_prompt": f"{min(r['prompt'] for r in d['picture']):,}",
        "pic_padded": f"{d['picture'][0]['padded_prompt']:,}",
        "pic_short_output": f"{min(r['output'] for r in d['picture']):,}",
        "pic_steps": f"{d['picture'][0]['batch_steps']:,}",
        "pic_idle": f"{d['picture'][0]['batch_steps'] - min(r['output'] for r in d['picture']):,}",
        # the same question at the reference model's scale
        "ref_itl_one": f"{ref[1]['inter_token_ms']:.0f} ms",
        "ref_itl_last": f"{ref[last]['inter_token_ms']:.0f} ms",
        "ref_tok_one": f"{ref[1]['tokens_per_s']:,.0f}",
        "ref_tok_last": f"{ref[last]['tokens_per_s']:,.0f}",
        "ref_gain": f"{ref[last]['tokens_per_s'] / ref[1]['tokens_per_s']:.0f}x",
        "ref_bound": ref[last]["bound_by"],
        "agree_from": str(d["equivalence"]["agreement"]["agree_from"]),
        "needed_tokens": f"{cs['tokens_per_s_needed']:,}",
        "case_rate": str(cs["requests_per_s"]),
        "case_output": str(cs["output_tokens"]),
        "gpus_one": f"{fleet[1]['accelerators']:,.0f}",
        "gpus_eight": f"{fleet[8]['accelerators']:,.0f}",
        "gpus_last": f"{fleet[last]['accelerators']:,.0f}",
    }


def ch17(d: dict) -> dict[str, str]:
    a, h = d["assumptions"], d["head_to_head"]
    st, co = h["static"], h["continuous"]
    rows = d["rows"]
    keeps = lambda policy: [r["rate"] for r in rows
                            if r["policy"] == policy and r["keeping_up"]]
    static_ok, cont_ok = keeps("static"), keeps("continuous")
    top = lambda policy: max(r["tokens_per_s"] for r in rows
                             if r["policy"] == policy)
    st_rows = sorted((r for r in rows if r["policy"] == "static"),
                     key=lambda r: r["rate"])
    co_rows = sorted((r for r in rows if r["policy"] == "continuous"),
                     key=lambda r: r["rate"])
    sweep = d["pool_sweep"]
    full = sweep[0]
    # The smallest pool that costs nothing: within 1% of the full pool's
    # throughput and inside the time-to-first-token budget.
    enough = [r for r in sweep
              if r["tokens_per_s"] >= full["tokens_per_s"] * 0.99
              and r["ttft_p99_ms"] <= a["ttft_budget_ms"]][-1]
    worst = sweep[-1]
    tl = d["timeline"]
    ms = lambda x: f"{x:,.0f} ms"
    s_ = lambda x: f"{x:,.1f} s"
    return {
        "rate": str(a["head_to_head_rate"]),
        "requests": f"{a['n_requests']:,}",
        "max_batch": str(a["max_batch"]),
        "block": str(a["block"]),
        "blocks": f"{a['blocks']:,}",
        "pool_gb": f"{a['pool_bytes'] / 1e9:.0f} GB",
        "ttft_budget": ms(a["ttft_budget_ms"]),
        "itl_budget": f"{a['itl_budget_ms']:,.0f} ms",
        "prompt_mean": f"{a['prompt_mean']:,}",
        "output_mean": str(a["output_mean"]),
        "offered": f"{st['offered_tokens_per_s']:,.0f}",
        # the two policies, same trace, same second
        "static_tok": f"{st['tokens_per_s']:,.0f}",
        "cont_tok": f"{co['tokens_per_s']:,.0f}",
        "tok_gain": f"{co['tokens_per_s'] / st['tokens_per_s']:.1f}x",
        "static_ttft50": ms(st["ttft_p50_ms"]),
        "static_ttft99": ms(st["ttft_p99_ms"]),
        "cont_ttft50": ms(co["ttft_p50_ms"]),
        "cont_ttft99": ms(co["ttft_p99_ms"]),
        "ttft_gain": f"{st['ttft_p50_ms'] / co['ttft_p50_ms']:,.0f}x",
        "static_ttft50_s": s_(st["ttft_p50_ms"] / 1e3),
        "static_itl50": f"{st['itl_p50_ms']:.1f} ms",
        "static_itl99": f"{st['itl_p99_ms']:.1f} ms",
        "cont_itl50": f"{co['itl_p50_ms']:.1f} ms",
        "cont_itl99": f"{co['itl_p99_ms']:.1f} ms",
        "cont_itl_max": f"{co['itl_max_ms']:.0f} ms",
        "static_itl_max": f"{st['itl_max_ms']:.0f} ms",
        "pure_itl": f"{d['pure_decode_itl_ms']:.1f} ms",
        "prefill_ms": f"{d['mean_prefill_ms']:.0f} ms",
        "prefill_over_decode": f"{d['mean_prefill_ms'] / d['pure_decode_itl_ms']:.0f}x",
        "itl_over_pure": f"{co['itl_p99_ms'] / d['pure_decode_itl_ms']:.0f}x",
        "static_total50": s_(st["total_p50_s"]),
        "cont_total50": s_(co["total_p50_s"]),
        "static_total99": s_(st["total_p99_s"]),
        "cont_total99": s_(co["total_p99_s"]),
        "total_gain": f"{st['total_p50_s'] / co['total_p50_s']:.0f}x",
        "static_batches": f"{st['prefill_iterations']:,}",
        "static_formed": f"{st['mean_slots']:,.0f}",
        "static_live": f"{st['mean_batch']:.1f}",
        "cont_batch": f"{co['mean_batch']:.1f}",
        "static_slots": f"{st['slot_utilization'] * 100:.0f}%",
        "cont_slots": f"{co['slot_utilization'] * 100:.0f}%",
        "static_idle": f"{st['idle_share'] * 100:.1f}%",
        "cont_prefill_share": f"{co['prefill_share'] * 100:.0f}%",
        "cont_decode_share": f"{co['decode_share'] * 100:.0f}%",
        "static_span": s_(st["arrival_span_s"]),
        "static_makespan": s_(st["makespan_s"]),
        "cont_makespan": s_(co["makespan_s"]),
        "static_drain": f"{st['drained_over_span']:.1f}x",
        "cont_drain": f"{co['drained_over_span']:.2f}x",
        "static_pool": f"{st['peak_pool_share'] * 100:.0f}%",
        "cont_pool": f"{co['peak_pool_share'] * 100:.0f}%",
        "pool_gain": f"{st['peak_blocks'] / co['peak_blocks']:.0f}x",
        # rising load
        "rate_low": str(min(r["rate"] for r in rows)),
        "rate_high": str(max(r["rate"] for r in rows)),
        "static_last_ok": (f"up to {max(static_ok)}" if static_ok
                           else "none of the rates tested"),
        "cont_last_ok": (f"{max(cont_ok)}" if cont_ok else "none"),
        "static_itl99_low": f"{min(r['itl_p99_ms'] for r in st_rows):.0f} ms",
        "static_itl99_high": f"{max(r['itl_p99_ms'] for r in st_rows):.0f} ms",
        "static_ttft99_low": f"{st_rows[0]['ttft_p99_ms'] / 1e3:.0f} s",
        "static_ttft99_high": f"{st_rows[-1]['ttft_p99_ms'] / 1e3:.0f} s",
        "cont_itl99_low": f"{co_rows[0]['itl_p99_ms']:.0f} ms",
        "cont_itl99_high": f"{max(r['itl_p99_ms'] for r in co_rows):.0f} ms",
        "cont_ttft99_high": f"{co_rows[-1]['ttft_p99_ms']:,.0f} ms",
        "static_top": f"{top('static'):,.0f}",
        "cont_top": f"{top('continuous'):,.0f}",
        "top_gain": f"{top('continuous') / top('static'):.1f}x",
        # the pool
        "sweep_rate": str(a["head_to_head_rate"]),
        "enough_share": f"{enough['pool_share'] * 100:.0f}%",
        "enough_gb": f"{enough['pool_gb']:.1f} GB",
        "enough_tok": f"{enough['tokens_per_s']:,.0f}",
        "enough_preempt": f"{enough['preemptions']:,}",
        "worst_share": f"{worst['pool_share'] * 100:.0f}%",
        "worst_gb": f"{worst['pool_gb']:.1f} GB",
        "worst_tok": f"{worst['tokens_per_s']:,.0f}",
        "worst_preempt": f"{worst['preemptions']:,}",
        "worst_wasted": f"{worst['wasted_token_share'] * 100:.1f}%",
        "worst_ttft99": s_(worst["ttft_p99_ms"] / 1e3),
        "worst_batch": f"{worst['mean_batch']:.1f}",
        "enough_batch": f"{enough['mean_batch']:.1f}",
        "worst_drop": f"{full['tokens_per_s'] / worst['tokens_per_s']:.1f}x",
        # the picture
        "pic_n": str(len(tl["static"])),
        "pic_static_last": s_(max(r["finish_s"] for r in tl["static"])),
        "pic_cont_last": s_(max(r["finish_s"] for r in tl["continuous"])),
    }


def ch18(d: dict) -> dict[str, str]:
    a = d["assumptions"]
    iv, chosen = d["interference"], d["chosen_budget"]
    whole, chunked = iv["whole"], iv["chunked"]
    low = {r["token_budget"]: r for r in d["budgets"]}
    high = {r["token_budget"]: r for r in d["budgets_high"]}
    base_low, base_high = low[0], high[0]          # Chapter 17's scheduler
    pick = high[chosen]
    smallest = high[min(k for k in high if k)]
    largest = high[max(high)]
    pol = {(r["pool"], r["policy"]): r for r in d["policies"]}
    sq = lambda name: pol[("squeezed", name)]
    fu = lambda name: pol[("full", name)]
    pre = {r["mode"]: r for r in d["preemption"]}
    rec = pre["recompute"]
    swap = pre["swap over PCIe 5.0 x16"]
    arith = d["swap_arithmetic"]
    at_context = next(r for r in arith["rows"] if r["tokens"] == a["context"])
    longest = arith["rows"][-1]
    ms = lambda x: f"{x:,.0f} ms"
    s_ = lambda x: f"{x:,.1f} s"
    return {
        "rate": str(a["rate"]),
        "rate_high": str(a["rate_high"]),
        "requests": f"{a['n_requests']:,}",
        "block": str(a["block"]),
        "max_batch": str(a["max_batch"]),
        "pool_gb": f"{a['pool_bytes'] / 1e9:.0f} GB",
        "prompt_mean": f"{a['prompt_mean']:,}",
        "output_mean": str(a["output_mean"]),
        "ttft_budget": ms(a["ttft_budget_ms"]),
        "itl_budget": f"{a['itl_budget_ms']:,.0f} ms",
        "pure_itl": f"{d['pure_decode_itl_ms']:.1f} ms",
        "whole_prefill": f"{d['whole_prefill_ms']:.0f} ms",
        "long_prefill": f"{d['long_prefill_ms']:.0f} ms",
        # one reply, with long prompts landing in it
        "iv_interlopers": str(whole["interlopers"]),
        "iv_prompt": f"{whole['interloper_prompt']:,}",
        "iv_output": str(whole["victim_output"]),
        "iv_budget": f"{chunked['budget']:,}",
        "iv_whole_p50": f"{whole['median_ms']:.1f} ms",
        "iv_whole_p99": f"{whole['p99_ms']:.0f} ms",
        "iv_whole_max": f"{whole['max_ms']:.0f} ms",
        "iv_whole_over": str(whole["over_budget"]),
        "iv_chunked_p50": f"{chunked['median_ms']:.1f} ms",
        "iv_chunked_p99": f"{chunked['p99_ms']:.1f} ms",
        "iv_chunked_max": f"{chunked['max_ms']:.1f} ms",
        "iv_chunked_over": str(chunked["over_budget"]),
        "iv_gain": f"{whole['max_ms'] / chunked['max_ms']:.0f}x",
        "iv_whole_finish": f"{whole['victim_finish_s']:.2f} s",
        "iv_chunked_finish": f"{chunked['victim_finish_s']:.2f} s",
        # the budget, at the rate where it binds
        "chosen": f"{chosen:,}",
        "smallest_budget": f"{min(k for k in high if k):,}",
        "largest_budget": f"{max(high):,}",
        "base_tok": f"{base_high['tokens_per_s']:,.0f}",
        "base_itl99": f"{base_high['itl_p99_ms']:.0f} ms",
        "base_ttft99": ms(base_high["ttft_p99_ms"]),
        "base_low_itl99": f"{base_low['itl_p99_ms']:.1f} ms",
        "base_low_tok": f"{base_low['tokens_per_s']:,.0f}",
        "pick_tok": f"{pick['tokens_per_s']:,.0f}",
        "pick_itl50": f"{pick['itl_p50_ms']:.1f} ms",
        "pick_itl99": f"{pick['itl_p99_ms']:.1f} ms",
        "pick_ttft50": ms(pick["ttft_p50_ms"]),
        "pick_ttft99": ms(pick["ttft_p99_ms"]),
        "pick_mixed": f"{pick['mixed_share'] * 100:.0f}%",
        "pick_tok_gain": f"{pick['tokens_per_s'] / base_high['tokens_per_s']:.2f}x",
        "pick_itl_gain": f"{base_high['itl_p99_ms'] / pick['itl_p99_ms']:.0f}x",
        "small_ttft99": s_(smallest["ttft_p99_ms"] / 1e3),
        "small_tok": f"{smallest['tokens_per_s']:,.0f}",
        "small_itl99": f"{smallest['itl_p99_ms']:.1f} ms",
        "small_mixed": f"{smallest['mixed_share'] * 100:.0f}%",
        "large_itl99": f"{largest['itl_p99_ms']:.1f} ms",
        "large_tok": f"{largest['tokens_per_s']:,.0f}",
        "large_ttft99": ms(largest["ttft_p99_ms"]),
        "low_pick_itl99": f"{low[chosen]['itl_p99_ms']:.1f} ms",
        "low_pick_tok": f"{low[chosen]['tokens_per_s']:,.0f}",
        # queue order
        "pol_pool_gb": f"{sq('fcfs')['pool_gb']:.1f} GB",
        "pol_pool_share": f"{a['swap_pool_share'] * 100:.0f}%",
        "full_fcfs_p50": s_(fu("fcfs")["total_p50_s"]),
        "full_spread": f"{max(fu(p)['total_p50_s'] for p in ('fcfs', 'shortest-output', 'longest-output')) / min(fu(p)['total_p50_s'] for p in ('fcfs', 'shortest-output', 'longest-output')) - 1:.1%}",
        "sq_fcfs_p50": s_(sq("fcfs")["total_p50_s"]),
        "sq_fcfs_p99": s_(sq("fcfs")["total_p99_s"]),
        "sq_sjf_p50": s_(sq("shortest-output")["total_p50_s"]),
        "sq_sjf_p99": s_(sq("shortest-output")["total_p99_s"]),
        "sq_sjf_gain": f"{sq('fcfs')['total_p50_s'] / sq('shortest-output')['total_p50_s']:.0f}x",
        "sq_sjf_p99_cost": f"{sq('shortest-output')['total_p99_s'] / sq('fcfs')['total_p99_s']:.1f}x",
        "sq_fcfs_slow99": f"{sq('fcfs')['slowdown_p99']:.0f}",
        "sq_fcfs_slowmax": f"{sq('fcfs')['slowdown_max']:.0f}",
        "sq_sjf_slow99": f"{sq('shortest-output')['slowdown_p99']:.0f}",
        "sq_sjf_slowmax": f"{sq('shortest-output')['slowdown_max']:.0f}",
        "sq_ljf_p50": s_(sq("longest-output")["total_p50_s"]),
        "sq_ljf_slowmax": f"{sq('longest-output')['slowdown_max']:.0f}",
        "sq_sjf_tok": f"{sq('shortest-output')['tokens_per_s']:,.0f}",
        "sq_fcfs_tok": f"{sq('fcfs')['tokens_per_s']:,.0f}",
        # the two ways out of a full pool
        "rec_tok": f"{rec['tokens_per_s']:,.0f}",
        "rec_preempt": f"{rec['preemptions']:,}",
        "rec_reread": f"{rec['prompt_reread']:.2f}x",
        "rec_ttft99": s_(rec["ttft_p99_ms"] / 1e3),
        "rec_batch": f"{rec['mean_batch']:.1f}",
        "swap_tok": f"{swap['tokens_per_s']:,.0f}",
        "swap_preempt": f"{swap['preemptions']:,}",
        "swap_gb": f"{swap['swapped_bytes'] / 1e9:,.0f} GB",
        "swap_s": f"{swap['swap_s']:.1f} s",
        "swap_ttft99": s_(swap["ttft_p99_ms"] / 1e3),
        "swap_batch": f"{swap['mean_batch']:.1f}",
        "swap_batch_gap": f"{rec['mean_batch'] / swap['mean_batch']:.1f}x",
        "rec_over_swap": f"{rec['tokens_per_s'] / swap['tokens_per_s']:.2f}x",
        "swap_ttft_cost": f"{swap['ttft_p99_ms'] / rec['ttft_p99_ms']:.1f}x",
        # the arithmetic behind the choice
        "arith_tokens": f"{at_context['tokens']:,}",
        "arith_recompute": f"{at_context['recompute_s'] * 1e3:.0f} ms",
        "arith_pcie5": f"{at_context['swap_s']['PCIe 5.0 x16'] * 1e3:.1f} ms",
        "arith_nvlink": f"{at_context['swap_s']['NVLink (H100)'] * 1e3:.1f} ms",
        "arith_breakeven": f"{at_context['breakeven_bytes_per_s'] / 1e9:.0f} GB/s",
        "arith_pcie_speed": f"{arith['links']['PCIe 5.0 x16'] / 1e9:.0f} GB/s",
        "arith_long": f"{longest['tokens']:,}",
        "arith_long_recompute": f"{longest['recompute_s'] * 1e3:.0f} ms",
        "arith_long_pcie5": f"{longest['swap_s']['PCIe 5.0 x16'] * 1e3:.0f} ms",
        "arith_long_breakeven": f"{longest['breakeven_bytes_per_s'] / 1e9:.0f} GB/s",
        # what the arithmetic model's attention convention does to this
        "attn_share": f"{d['attention_share'][str(a['prompt_mean'])]['attention_share'] * 100:.0f}%",
        "attn_drift": f"{abs(d['attention_share'][str(a['prompt_mean'])]['worst_drift']) * 100:.1f}%",
        "attn_share_long": f"{d['attention_share']['8192']['attention_share'] * 100:.0f}%",
        "attn_drift_long": f"{abs(d['attention_share']['8192']['worst_drift']) * 100:.0f}%",
    }


def _delta(splits: list[dict], best: dict, offset: int) -> str:
    """How much a split `offset` machines from the best one costs."""
    want = best["prefill_workers"] + offset
    row = next((r for r in splits if r["prefill_workers"] == want), None)
    if row is None:
        return "off the end of the fleet"
    change = row["tokens_per_s"] / best["tokens_per_s"] - 1
    pct = abs(change) * 100
    return f"{pct:.1f}% less" if pct < 1 else f"{pct:.0f}% less"


def ch19(d: dict) -> dict[str, str]:
    a = d["assumptions"]
    splits, co, best = d["splits"], d["colocated"], d["best_split"]
    worst = min(splits, key=lambda r: r["tokens_per_s"])
    tr = d["transfer"]
    at_prompt = next(r for r in tr["rows"] if r["tokens"] == a["prompt_mean"])
    at_context = next(r for r in tr["rows"] if r["tokens"] == a["context"])
    longest = tr["rows"][-1]
    links = {r["link"]: r for r in d["links"]}
    fast, slow = d["links"][0], d["links"][-1]
    by_prompt = {}
    for r in d["prompts"]:
        by_prompt.setdefault(r["prompt_mean"], {})[r["design"]] = r
    short = by_prompt[min(by_prompt)]
    lp = {r["design"]: r for r in d["long_prompts"]}
    fleets = d["fleets"]
    tok = lambda x: f"{x:,.0f}"
    ms = lambda x: f"{x:,.0f} ms"
    return {
        "fleet": str(a["fleet"]),
        "rate": str(a["rate"]),
        "requests": f"{a['n_requests']:,}",
        "prompt_mean": f"{a['prompt_mean']:,}",
        "output_mean": str(a["output_mean"]),
        "budget": f"{a['token_budget']:,}",
        "pool_gb": f"{a['pool_bytes'] / 1e9:.0f} GB",
        "link": a["link"],
        "link_speed": f"{a['link_bytes_per_s'] / 1e9:.0f} GB/s",
        "ttft_budget": ms(a["ttft_budget_ms"]),
        "itl_budget": f"{a['itl_budget_ms']:,.0f} ms",
        "offered": tok(co["offered_tokens_per_s"]),
        "warm": f"{a['warm_fraction'] * 100:.0f}%",
        # what a cache costs to move
        "kv_per_token": f"{a['kv_bytes_per_token'] / 1024:.0f} KiB",
        "prompt_mb": f"{at_prompt['bytes'] / 1e6:.0f} MB",
        "prompt_prefill": f"{at_prompt['prefill_s'] * 1e3:.0f} ms",
        "prompt_over_link": f"{at_prompt['over'][a['link']] * 1e3:.1f} ms",
        "prompt_over_nvlink": f"{at_prompt['over']['NVLink, same node'] * 1e3:.1f} ms",
        "prompt_over_slow": f"{at_prompt['over']['25 GbE'] * 1e3:.0f} ms",
        "prompt_match_link": f"{at_prompt['link_to_match_prefill'] / 1e9:.1f} GB/s",
        "context_mb": f"{at_context['bytes'] / 1e6:.0f} MB",
        "long_tokens": f"{longest['tokens']:,}",
        "long_mb": f"{longest['bytes'] / 1e6:,.0f} MB",
        "long_over_link": f"{longest['over'][a['link']] * 1e3:.0f} ms",
        "long_match_link": f"{longest['link_to_match_prefill'] / 1e9:.1f} GB/s",
        "decode_step": f"{tr['decode_step_ms']:.1f} ms",
        # the split
        "best_p": str(best["prefill_workers"]),
        "best_d": str(best["decode_workers"]),
        "best_tok": tok(best["tokens_per_s"]),
        "best_ttft50": ms(best["ttft_p50_ms"]),
        "best_ttft99": ms(best["ttft_p99_ms"]),
        "best_itl50": f"{best['itl_p50_ms']:.1f} ms",
        "best_itl99": f"{best['itl_p99_ms']:.1f} ms",
        "best_batch": f"{best['mean_batch']:.1f}",
        # the two ends of the sweep, named rather than inferred
        "low_p": str(splits[0]["prefill_workers"]),
        "low_p_d": str(splits[0]["decode_workers"]),
        "low_p_tok": tok(splits[0]["tokens_per_s"]),
        "low_p_ttft99": f"{splits[0]['ttft_p99_ms'] / 1e3:.0f} s",
        "low_p_batch": f"{splits[0]['mean_batch']:.1f}",
        "high_p": str(splits[-1]["prefill_workers"]),
        "high_p_d": str(splits[-1]["decode_workers"]),
        "high_p_tok": tok(splits[-1]["tokens_per_s"]),
        "high_p_ttft99": f"{splits[-1]['ttft_p99_ms'] / 1e3:.0f} s",
        "high_p_itl99": f"{splits[-1]['itl_p99_ms']:.1f} ms",
        "high_p_batch": f"{splits[-1]['mean_batch']:.0f}",
        # how sharp the optimum is, on each side
        "one_below": _delta(splits, best, -1),
        "one_above": _delta(splits, best, +1),
        "three_below": _delta(splits, best, -3),
        "three_above": _delta(splits, best, +3),
        "worst_p": str(worst["prefill_workers"]),
        "worst_d": str(worst["decode_workers"]),
        "worst_tok": tok(worst["tokens_per_s"]),
        "split_range": f"{best['tokens_per_s'] / worst['tokens_per_s']:.1f}x",
        "split_ratio": f"1:{best['decode_workers'] / best['prefill_workers']:.0f}",
        # the fleet doing both phases everywhere
        "co_tok": tok(co["tokens_per_s"]),
        "co_ttft50": ms(co["ttft_p50_ms"]),
        "co_ttft99": ms(co["ttft_p99_ms"]),
        "co_itl50": f"{co['itl_p50_ms']:.1f} ms",
        "co_itl99": f"{co['itl_p99_ms']:.1f} ms",
        "co_batch": f"{co['mean_batch']:.1f}",
        "co_over_best": f"{co['tokens_per_s'] / best['tokens_per_s']:.2f}x",
        "co_gain_pct": f"{(co['tokens_per_s'] / best['tokens_per_s'] - 1) * 100:.0f}%",
        "offered_sampled": tok(co["offered_sampled_tokens_per_s"]),
        "co_keeps": "keeps up" if co["keeping_up"] else "does not keep up",
        "best_keeps": "keeps up" if best["keeping_up"] else "does not keep up",
        "co_of_offered": f"{co['tokens_per_s'] / co['offered_sampled_tokens_per_s'] * 100:.0f}%",
        "best_of_offered": f"{best['tokens_per_s'] / best['offered_sampled_tokens_per_s'] * 100:.0f}%",
        # where the network bill lands
        "second_co50": f"{co['second_token_p50_ms']:.1f} ms",
        "second_co99": f"{co['second_token_p99_ms']:.1f} ms",
        "second_fast50": f"{fast['second_token_p50_ms']:.1f} ms",
        "second_link50": f"{links[a['link']]['second_token_p50_ms']:.1f} ms",
        "second_link99": f"{links[a['link']]['second_token_p99_ms']:.1f} ms",
        "second_slow50": f"{slow['second_token_p50_ms']:.0f} ms",
        "second_slow99": f"{slow['second_token_p99_ms']:.0f} ms",
        "slow_link": slow["link"],
        "link_span": f"{max(tr['links'].values()) / min(tr['links'].values()):.0f}x",
        "fast_link": fast["link"],
        "link_tok_spread": f"{(max(r['tokens_per_s'] for r in d['links']) / min(r['tokens_per_s'] for r in d['links']) - 1) * 100:.1f}%",
        "link_itl_spread": f"{max(r['itl_p99_ms'] for r in d['links']) - min(r['itl_p99_ms'] for r in d['links']):.1f} ms",
        # regimes
        "short_prompt": f"{min(by_prompt):,}",
        "short_dis_tok": tok(short["disaggregated"]["tokens_per_s"]),
        "short_co_tok": tok(short["colocated"]["tokens_per_s"]),
        "short_p": str(short["disaggregated"]["prefill_workers"]),
        "long_prompt": f"{a['long_prompt']:,}",
        "long_rate": f"{a['long_rate']:.0f}",
        "long_dis_p": str(lp["disaggregated"]["prefill_workers"]),
        "long_dis_d": str(lp["disaggregated"]["decode_workers"]),
        "long_dis_tok": tok(lp["disaggregated"]["tokens_per_s"]),
        "long_co_tok": tok(lp["colocated"]["tokens_per_s"]),
        "long_dis_second": f"{lp['disaggregated']['second_token_p50_ms']:.1f} ms",
        "long_co_second": f"{lp['colocated']['second_token_p50_ms']:.1f} ms",
        "small_fleet": str(min(r["fleet"] for r in fleets)),
        "small_p": str(fleets[0]["best"]["prefill_workers"]),
        "small_dis_tok": tok(fleets[0]["best"]["tokens_per_s"]),
        "small_co_tok": tok(fleets[0]["colocated"]["tokens_per_s"]),
        "small_dis_itl": f"{fleets[0]['best']['itl_p99_ms']:.1f} ms",
        "small_co_itl": f"{fleets[0]['colocated']['itl_p99_ms']:.1f} ms",
        "every_fleet": " and ".join(
            f"{min(v):.2f}" if len(v := sorted(
                r["best"]["tokens_per_s"] / r["colocated"]["tokens_per_s"]
                for r in fleets)) else "" for _ in [0]) + "x to "
            + f"{max(r['best']['tokens_per_s'] / r['colocated']['tokens_per_s'] for r in fleets):.2f}x",
    }


def ddr1(d: dict) -> dict[str, str]:
    """Design decision record I: the values its prose quotes.

    Almost every number here is a number some chapter in Part III
    already published. It is read back out of `ddr1.json`, which read it
    out of that chapter's results file, so a decision quoted in this
    record cannot disagree with the measurement it was taken from.
    """
    a, f, m = d["assumptions"], d["fleet"], d["memory_policies"]
    by_chapter: dict[str, list[dict]] = {}
    for dec in d["decisions"]:
        by_chapter.setdefault(dec["chapter"], []).append(dec)
    sizing = {x["workers"]: x for x in d["sizing"]}
    chosen, below = sizing[f["chosen"]], sizing[f["one_below"]]
    biggest = sizing[max(sizing)]
    cheapest = min(d["sizing"], key=lambda x: x["usd_per_m_output_tokens"])
    tok = lambda x: f"{x:,.0f}"
    pct = lambda x: f"{x * 100:.0f}%"
    ms = lambda x: f"{x:,.0f} ms"
    # Match the table: seconds once a wait has stopped being a latency.
    secs = lambda x: f"{x / 1e3:,.1f} s" if x >= 1000 else f"{x:,.0f} ms"
    return {
        # what the record is made of
        "decisions": str(len(d["decisions"])),
        "chapters": str(len(a["from_chapters"])),
        "first_chapter": a["from_chapters"][0].removeprefix("ch").lstrip("0"),
        "last_chapter": a["from_chapters"][-1].removeprefix("ch").lstrip("0"),
        # the standing assumptions
        "block": str(a["block"]),
        "budget": f"{a['token_budget']:,}",
        "pool_gb": f"{a['pool_bytes'] / 1e9:.0f} GB",
        "blocks": f"{a['blocks_per_worker']:,}",
        "prompt_tokens": f"{a['prompt_tokens']:,}",
        "output_tokens": str(a["output_tokens"]),
        "rate": str(a["requests_per_s"]),
        "ttft_budget": ms(a["ttft_budget_ms"]),
        "itl_budget": f"{a['itl_budget_ms']:,.0f} ms",
        "keeping_up_share": pct(a["keeping_up_share"]),
        "sizing_requests": f"{a['sizing_requests']:,}",
        "warm_fraction": pct(a["warm_fraction"]),
        # the fleet
        "chosen": str(f["chosen"]),
        "one_below": str(f["one_below"]),
        "needed": tok(f["tokens_per_s_needed"]),
        "offered": tok(f["offered_tokens_per_s"]),
        "chosen_tps": tok(f["chosen_tokens_per_s"]),
        "chosen_share": pct(f["chosen_share_of_offered"]),
        "chosen_ttft99": f"{f['chosen_ttft_p99_ms']:,.0f} ms",
        "chosen_itl99": f"{f['chosen_itl_p99_ms']:.1f} ms",
        "below_share": pct(f["one_below_share"]),
        "below_ttft99": f"{below['ttft_p99_ms']:,.0f} ms",
        "measured": str(f["measured_in_ch19"]),
        "measured_share": pct(f["measured_share_of_offered"]),
        "measured_ttft99": f"{f['measured_ttft_p99_ms']:,.0f} ms",
        "biggest": str(biggest["workers"]),
        "biggest_share": pct(biggest["share_of_offered"]),
        "biggest_ttft99": f"{biggest['ttft_p99_ms']:,.0f} ms",
        # against Chapter 16's arithmetic
        "arith_batch": str(f["arithmetic_batch"]),
        "arith_tps": tok(f["arithmetic_tokens_per_s"]),
        "arith_fleet": f"{f['arithmetic_accelerators']:.1f}",
        "per_machine_tps": tok(f["per_machine_tokens_per_s"]),
        "per_machine_share": pct(f["per_machine_share_of_arithmetic"]),
        "per_machine_rate": f"{f['per_machine_requests_per_s']:.1f}",
        # the bill
        "usd_hour": f"${f['usd_per_hour']:,.2f}",
        "usd_month": f"${f['usd_per_month']:,.0f}",
        "usd_m_tokens": f"${f['usd_per_m_output_tokens']:.3f}",
        "usd_m_at_measured": f"${f['usd_per_m_at_measured']:.3f}",
        "usd_gpu_hour": f"${f['gpu_usd_per_hour']:.2f}",
        "cheapest": str(cheapest["workers"]),
        "cheapest_usd_m": f"${cheapest['usd_per_m_output_tokens']:.3f}",
        "cheapest_ttft99": secs(cheapest["ttft_p99_ms"]),
        "usd_over_cheapest":
            f"{f['usd_per_m_output_tokens'] / cheapest['usd_per_m_output_tokens']:.1f}x",
        "biggest_usd_m": f"${biggest['usd_per_m_output_tokens']:.3f}",
        # which memory policies this service actually put under pressure
        "cache_small_frac": pct(m["cache_smallest_frac"]),
        "cache_small_hit": pct(m["cache_smallest_hit"]),
        "cache_full_hit": pct(m["cache_largest_hit"]),
        "full_pool_preemptions": str(m["preemptions_at_full_pool"]),
        "first_preempt_share": pct(m["largest_pool_share_that_preempts"]),
        "first_preempt_count": str(m["preemptions_there"]),
        "squeezed_preemptions": f"{m['order_preemptions_squeezed']:,}",
        "usd_biggest_over_chosen":
            f"{biggest['usd_per_m_output_tokens'] / chosen['usd_per_m_output_tokens']:.1f}x",
        "usd_measured_over_chosen":
            f"{f['usd_per_m_at_measured'] / f['usd_per_m_output_tokens']:.1f}x",
        # a few decisions quoted in the prose
        "block_paged": str(by_chapter["ch14"][0]["evidence"]
                           ["sequences admitted, paged"]),
        "block_reserved": str(by_chapter["ch14"][0]["evidence"]
                              ["sequences admitted, whole context reserved"]),
        "cache_hit": pct(by_chapter["ch15"][0]["evidence"]["hit rate, tree with LRU"]),
        "cache_gb": f"{by_chapter['ch15'][0]['evidence']['cache GB']:.0f} GB",
        "static_gain": f"{by_chapter['ch17'][0]['evidence']['tokens/s, continuous'] / by_chapter['ch17'][0]['evidence']['tokens/s, static']:.1f}x",
        "order_worth": f"{abs(by_chapter['ch18'][1]['evidence']['end-to-end p50 s, shortest first, full pool'] / by_chapter['ch18'][1]['evidence']['end-to-end p50 s, FCFS, full pool'] - 1) * 100:.1f}%",
        "order_squeezed": f"{by_chapter['ch18'][1]['evidence']['end-to-end p50 s, FCFS, squeezed pool'] / by_chapter['ch18'][1]['evidence']['end-to-end p50 s, shortest first, squeezed pool']:.1f}x",
        "squeezed_gb": f"{by_chapter['ch18'][1]['evidence']['squeezed pool, GB']:.1f} GB",
        "colocated_gain": f"{by_chapter['ch19'][0]['evidence']['tokens/s, colocated'] / by_chapter['ch19'][0]['evidence']['tokens/s, best split']:.2f}x",
        "recompute_gain": f"{by_chapter['ch18'][2]['evidence']['tokens/s, recompute'] / by_chapter['ch18'][2]['evidence']['tokens/s, swap over PCIe 5.0']:.2f}x",
        "recompute_ttft_gain": f"{by_chapter['ch18'][2]['evidence']['TTFT p99 ms, swap over PCIe 5.0'] / by_chapter['ch18'][2]['evidence']['TTFT p99 ms, recompute']:.1f}x",
        "recompute_ms": f"{by_chapter['ch18'][2]['evidence']['one sequence, recompute ms']:.0f} ms",
        "swap_ms": f"{by_chapter['ch18'][2]['evidence']['one sequence, PCIe 5.0 ms']:.0f} ms",
        "recompute_over_swap": f"{by_chapter['ch18'][2]['evidence']['one sequence, recompute ms'] / by_chapter['ch18'][2]['evidence']['one sequence, PCIe 5.0 ms']:.1f}x",
        "context_tokens": f"{a['prompt_tokens'] + a['output_tokens']:,}",
        "colocated_gain_pct": f"{(by_chapter['ch19'][0]['evidence']['tokens/s, colocated'] / by_chapter['ch19'][0]['evidence']['tokens/s, best split'] - 1) * 100:.0f}%",
    }


def load(chapter: str = "ch12") -> dict[str, str]:
    d = json.loads((RESULTS / f"{chapter}.json").read_text())
    return {"ch01": ch01, "ch02": ch02, "ch03": ch03, "ch04": ch04,
            "ch05": ch05, "ch06": ch06, "ch07": ch07, "ch08": ch08,
            "ch09": ch09, "ch10": ch10, "ch11": ch11,
            "ch12": ch12, "ch13": ch13, "ch14": ch14,
            "ch15": ch15, "ch16": ch16, "ch17": ch17,
            "ch18": ch18, "ch19": ch19,
            "ddr1": ddr1}[chapter](d)


if __name__ == "__main__":
    for k, v in load().items():
        print(f"{k:24} {v}")
