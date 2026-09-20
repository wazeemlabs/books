"""Chapter 5: the four numbers a serving team argues about.

Two parts:

1. MEASURED -- the distribution of decode-step times on tinyserve. The
   average is not the experience; the tail is. This is a real run.
2. MODELLED -- the trade between throughput and latency as the batch
   grows, and what a latency budget does to it. Arithmetic over
   published specs (tinyserve/serving.py); Chapters 16 and 17 measure
   the same curve.

    python3 -m bench.run_ch05
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from tinyserve import serving
from tinyserve.cost import flops_forward
from tinyserve.model import Config, KVCache, build, forward
from tinyserve.reference import (CONTEXT_TOKENS, GPU_USD_PER_HOUR,
                                 HBM_BYTES_PER_S, MODEL, OUTPUT_TOKENS,
                                 PEAK_BF16_FLOPS, PROMPT_TOKENS, WEIGHT_BYTES)

from .harness import pct, write

SAMPLES, PROMPT = 1500, 128
BATCHES = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 325]
ITL_BUDGETS_MS = [15, 25, 50, 100]
CASE_STUDY_ITL_MS, CASE_STUDY_TTFT_MS = 50, 1000
MAX_CONCURRENT = 325  # Chapter 13's paged allocator; checked in the audit


def measure_step_times() -> list[float]:
    """Time many decode steps, unchanged, so the spread is the model's own."""
    cfg = Config(max_seq=PROMPT + SAMPLES + 1)
    model = build(cfg)
    cache = KVCache(cfg, max_seq=PROMPT + SAMPLES + 1)
    forward(model, np.arange(PROMPT) % cfg.vocab_size, cache)
    nxt, times = np.array([0]), []
    for _ in range(SAMPLES):
        t0 = perf_counter()
        forward(model, nxt, cache)
        times.append(perf_counter() - t0)
    return times


def time_to_first_token_ms(prompt: int = PROMPT_TOKENS) -> float:
    """Prefill under the same roofline: compute-bound at this length."""
    f = flops_forward(MODEL, prompt, prompt)
    return max(f / PEAK_BF16_FLOPS, WEIGHT_BYTES / HBM_BYTES_PER_S) * 1e3


def main() -> None:
    times = measure_step_times()
    ms = [t * 1e3 for t in times]
    p50, p99, p999 = pct(ms, 50), pct(ms, 99), pct(ms, 99.9)
    mean = sum(ms) / len(ms)

    curve = []
    for b in BATCHES:
        step = serving.decode_step(b, CONTEXT_TOKENS)
        curve.append({
            "batch": b,
            "inter_token_ms": step.inter_token_ms,
            "tokens_per_s": step.tokens_per_s,
            "usd_per_m_tokens": step.usd_per_m_tokens(GPU_USD_PER_HOUR),
            "bound_by": step.bound_by,
        })

    ttft_ms = time_to_first_token_ms()
    budgets = []
    for budget in ITL_BUDGETS_MS:
        b = serving.largest_batch_within(budget, CONTEXT_TOKENS, MAX_CONCURRENT)
        step = serving.decode_step(max(b, 1), CONTEXT_TOKENS)
        budgets.append({
            "itl_budget_ms": budget,
            "largest_batch": b,
            "achieved_itl_ms": step.inter_token_ms,
            "tokens_per_s": step.tokens_per_s,
            "usd_per_m_tokens": step.usd_per_m_tokens(GPU_USD_PER_HOUR),
            "limited_by_memory_not_budget": b >= MAX_CONCURRENT,
        })

    case = next(x for x in budgets if x["itl_budget_ms"] == CASE_STUDY_ITL_MS)
    # One reply, end to end, at the case study's operating point.
    reply_s = ttft_ms / 1e3 + OUTPUT_TOKENS * case["achieved_itl_ms"] / 1e3

    payload = {
        "measured": {
            "samples": len(ms), "prompt": PROMPT,
            "mean_ms": mean, "p50_ms": p50, "p90_ms": pct(ms, 90),
            "p99_ms": p99, "p999_ms": p999, "max_ms": max(ms), "min_ms": min(ms),
            "p99_over_p50": p99 / p50, "mean_over_p50": mean / p50,
            "histogram": np.histogram(ms, bins=40)[0].tolist(),
            "histogram_edges": np.histogram(ms, bins=40)[1].tolist(),
        },
        "model_note": "the curve and budgets below are arithmetic over "
                      "published specs, not measurements",
        "curve": curve,
        "ttft_ms": ttft_ms,
        "budgets": budgets,
        "case_study": {
            "itl_ms": CASE_STUDY_ITL_MS, "ttft_ms": CASE_STUDY_TTFT_MS,
            "prompt": PROMPT_TOKENS, "output": OUTPUT_TOKENS,
            "operating_batch": case["largest_batch"],
            "achieved_itl_ms": case["achieved_itl_ms"],
            "achieved_ttft_ms": ttft_ms,
            "ttft_headroom": CASE_STUDY_TTFT_MS / ttft_ms,
            "tokens_per_s": case["tokens_per_s"],
            "reply_seconds": reply_s,
            "max_concurrent": MAX_CONCURRENT,
        },
    }

    path = write("results/ch05.json", payload)
    print(f"wrote {path}")
    print(f"  measured {len(ms)} steps: mean {mean:.3f} ms, p50 {p50:.3f}, "
          f"p99 {p99:.3f} ({p99 / p50:.1f}x p50), p99.9 {p999:.3f}")
    for b in budgets:
        print(f"  budget {b['itl_budget_ms']:4d} ms -> batch {b['largest_batch']:4d}"
              f"  {b['tokens_per_s']:9,.0f} tok/s  ${b['usd_per_m_tokens']:.3f}/M")
    print(f"  case study: TTFT {ttft_ms:.0f} ms of {CASE_STUDY_TTFT_MS} allowed, "
          f"batch {case['largest_batch']}, reply in {reply_s:.1f} s")


if __name__ == "__main__":
    main()
