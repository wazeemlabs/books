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
    return {"ch12": ch12, "ch13": ch13}[chapter](d)


if __name__ == "__main__":
    for k, v in load().items():
        print(f"{k:24} {v}")
