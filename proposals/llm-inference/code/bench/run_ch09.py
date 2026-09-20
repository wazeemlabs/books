"""Chapter 9: how to make a system look faster without changing it.

Every chapter of this book quotes numbers. This one measures the ways
a number can be true and still mislead, by taking one unchanged
operation and reporting it several defensible-sounding ways.

Each row is honest in isolation. Together they span a wide range, and
the spread is the point: a benchmark without its conditions is not a
measurement, it is a claim.

    python3 -m bench.run_ch09
"""

from __future__ import annotations

from statistics import mean, median
from time import perf_counter

import numpy as np

from tinyserve.model import Config, KVCache, build, forward

from .harness import pct, write

SAMPLES = 400
SHORT_CONTEXT, LONG_CONTEXT = 128, 2048


def decode_times(context: int, samples: int = SAMPLES) -> tuple[float, list[float]]:
    """Return the very first step's time, and the times of the rest.

    The first call is kept separately because it is the one a careless
    benchmark reports: caches cold, buffers unallocated.
    """
    cfg = Config(max_seq=context + samples + 2)
    model = build(cfg)
    cache = KVCache(cfg, max_seq=cfg.max_seq)
    forward(model, np.arange(context) % cfg.vocab_size, cache)

    nxt = np.array([0])
    t0 = perf_counter()
    forward(model, nxt, cache)
    cold = perf_counter() - t0

    warm = []
    for _ in range(samples):
        t0 = perf_counter()
        forward(model, nxt, cache)
        warm.append(perf_counter() - t0)
    return cold, warm


def main() -> None:
    cold, warm = decode_times(SHORT_CONTEXT)
    _, long_warm = decode_times(LONG_CONTEXT)

    honest = median(warm)          # warm, median, conditions stated
    ways = [
        {"how": "Best single run, short context",
         "sin": "cherry-picked run and cherry-picked conditions",
         "tokens_per_s": 1 / min(warm)},
        {"how": "Best single run", "sin": "reports the luckiest run",
         "tokens_per_s": 1 / min(warm)},
        {"how": "Mean of all runs", "sin": "an average hides the tail",
         "tokens_per_s": 1 / mean(warm)},
        {"how": "Median (what this book reports)", "sin": "none",
         "tokens_per_s": 1 / honest},
        {"how": "p99", "sin": "none - this is what unlucky users get",
         "tokens_per_s": 1 / pct(warm, 99)},
        {"how": "Including the first, cold run",
         "sin": "no warmup", "tokens_per_s": 1 / cold},
        {"how": f"Median at a {LONG_CONTEXT:,}-token context",
         "sin": "the same claim, harder conditions",
         "tokens_per_s": 1 / median(long_warm)},
    ]
    # Drop the duplicate first row, kept above only for readability.
    ways = ways[1:]
    for w in ways:
        w["relative_to_honest"] = w["tokens_per_s"] / (1 / honest)

    flattering = max(ways, key=lambda w: w["tokens_per_s"])
    damning = min(ways, key=lambda w: w["tokens_per_s"])

    payload = {
        "experiment": {"samples": SAMPLES, "short_context": SHORT_CONTEXT,
                       "long_context": LONG_CONTEXT},
        "honest_tokens_per_s": 1 / honest,
        "ways": ways,
        "spread": {
            "flattering": flattering["how"],
            "flattering_tokens_per_s": flattering["tokens_per_s"],
            "damning": damning["how"],
            "damning_tokens_per_s": damning["tokens_per_s"],
            "ratio": flattering["tokens_per_s"] / damning["tokens_per_s"],
        },
        "cold_penalty": cold / honest,
        "tail_ratio": pct(warm, 99) / honest,
        "context_penalty": median(long_warm) / honest,
        "distribution": {"p50": pct(warm, 50) * 1e3, "p99": pct(warm, 99) * 1e3,
                         "min": min(warm) * 1e3, "max": max(warm) * 1e3,
                         "mean": mean(warm) * 1e3},
    }

    path = write("results/ch09.json", payload)
    print(f"wrote {path}")
    for w in ways:
        print(f"  {w['how']:42} {w['tokens_per_s']:8,.0f} tok/s "
              f"({w['relative_to_honest']:.2f}x the honest figure)")
    sp = payload["spread"]
    print(f"  same system, same machine, nothing changed: "
          f"{sp['ratio']:.1f}x between the best and worst honest framing")


if __name__ == "__main__":
    main()
