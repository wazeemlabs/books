"""Named values for the manuscript, derived from results/*.json.

Prose cannot include a generated table, so prose numbers are written as
`{{name}}` in the chapter source and resolved here. The formatting lives
in code, so a number in a sentence and the same number in a table can
never disagree, and neither can drift from the measurement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

RESULTS = Path("results")


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


def load(chapter: str = "ch12") -> dict[str, str]:
    d = json.loads((RESULTS / f"{chapter}.json").read_text())
    return {"ch12": ch12}[chapter](d)


if __name__ == "__main__":
    for k, v in load().items():
        print(f"{k:24} {v}")
