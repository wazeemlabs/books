"""Chapter 10: put the model on the bench, and check it is a real one.

Two jobs before Part III starts changing things.

1. CORRECTNESS. tinyserve is hand-written NumPy. Load one set of
   weights into it and into a PyTorch implementation of the same
   architecture, and check they agree. If they do not, nothing measured
   later means anything.
2. THE BASELINE. Record what the unoptimized engine does, so that every
   technique in Parts III to VI can be measured against a number rather
   than an impression.

    python3 -m bench.run_ch10
"""

from __future__ import annotations

from statistics import median
from time import perf_counter

import numpy as np
import torch

from tinyserve import torch_reference
from tinyserve.generate import cached, naive
from tinyserve.model import Config, KVCache, build, forward

from .harness import Repeated, pct, write

# Shapes worth checking: square attention, grouped-query attention, and
# a deeper model, since a bug can hide in any one of them.
AGREEMENT_CASES = [
    ("reference shape", Config()),
    ("grouped-query attention", Config(n_heads=8, n_kv_heads=2, d_model=256)),
    ("deeper", Config(n_layers=8, d_model=192, n_heads=6, n_kv_heads=6, d_ff=768)),
    ("single key-value head", Config(n_heads=4, n_kv_heads=1)),
]
AGREEMENT_TOKENS = 24
SPEED_SIZES = [64, 256, 512]
BASELINE_PROMPT, BASELINE_NEW = 128, 128
RUNS = 5


def agreement(name: str, cfg: Config) -> dict:
    model = build(cfg, seed=0)
    tokens = np.arange(AGREEMENT_TOKENS) % cfg.vocab_size
    ours = forward(model, tokens)
    theirs = torch_reference.forward(model, torch.from_numpy(tokens)).numpy()
    return {
        "case": name,
        "config": {k: getattr(cfg, k) for k in
                   ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff")},
        "max_absolute_difference": float(np.abs(ours - theirs).max()),
        "max_relative_difference": float(
            (np.abs(ours - theirs) / (np.abs(theirs) + 1e-6)).max()),
        "same_greedy_tokens": bool((ours.argmax(-1) == theirs.argmax(-1)).all()),
    }


def timed(fn, runs: int = RUNS) -> float:
    fn()
    ts = []
    for _ in range(runs):
        t0 = perf_counter(); fn(); ts.append(perf_counter() - t0)
    return median(ts)


def speed(n_tokens: int) -> dict:
    cfg = Config(max_seq=max(SPEED_SIZES) + 2)
    model = build(cfg)
    tokens = np.arange(n_tokens) % cfg.vocab_size
    t_tokens = torch.from_numpy(tokens)
    ours = timed(lambda: forward(model, tokens, KVCache(cfg, max_seq=n_tokens + 2)))
    theirs = timed(lambda: torch_reference.forward(model, t_tokens))
    return {"tokens": n_tokens, "tinyserve_s": ours, "torch_s": theirs,
            "torch_over_tinyserve": theirs / ours}


def main() -> None:
    checks = [agreement(name, cfg) for name, cfg in AGREEMENT_CASES]
    all_agree = all(c["same_greedy_tokens"] for c in checks)
    worst = max(c["max_absolute_difference"] for c in checks)

    speeds = [speed(n) for n in SPEED_SIZES]

    # The default configuration, so this baseline is the same model the
    # later chapters measure. Chapter 12 must report the same parameter
    # count; the audit checks that it does.
    cfg = Config()
    model = build(cfg)
    prompt = list(range(BASELINE_PROMPT))
    naive_runs = [naive(model, prompt, BASELINE_NEW) for _ in range(3)]
    cached_runs = [cached(model, prompt, BASELINE_NEW) for _ in range(3)]
    steps = [s for r in cached_runs for s in r.step_s]

    baseline = {
        "prompt": BASELINE_PROMPT, "generated": BASELINE_NEW,
        "params": model.n_params,
        "weight_bytes": model.n_params * 4,
        "no_cache": {
            "total_s": Repeated([r.total_s for r in naive_runs]).summary(),
            "tokens_per_s": BASELINE_NEW / median([r.total_s for r in naive_runs]),
        },
        "with_cache": {
            "total_s": Repeated([r.total_s for r in cached_runs]).summary(),
            "ttft_ms": median([r.ttft_s for r in cached_runs]) * 1e3,
            "decode_p50_ms": pct(steps, 50) * 1e3,
            "decode_p99_ms": pct(steps, 99) * 1e3,
            "tokens_per_s": BASELINE_NEW / median([r.total_s for r in cached_runs]),
            "kv_bytes_per_token": KVCache(cfg).bytes_per_token(),
        },
    }

    payload = {
        "agreement": {"cases": checks, "all_agree": all_agree,
                      "worst_absolute_difference": worst,
                      "tokens_compared": AGREEMENT_TOKENS,
                      "torch_version": torch.__version__},
        "speed": speeds,
        "baseline": baseline,
        "note": "the baseline is the 'before' figure every later chapter is "
                "measured against",
    }

    path = write("results/ch10.json", payload)
    print(f"wrote {path}")
    for c in checks:
        print(f"  {c['case']:26} max difference {c['max_absolute_difference']:.2e}  "
              f"same tokens: {c['same_greedy_tokens']}")
    for s in speeds:
        print(f"  prefill {s['tokens']:4d} tokens: tinyserve {s['tinyserve_s'] * 1e3:7.2f} ms, "
              f"torch {s['torch_s'] * 1e3:7.2f} ms ({s['torch_over_tinyserve']:.2f}x)")
    b = baseline
    print(f"  baseline: no cache {b['no_cache']['tokens_per_s']:,.0f} tok/s, "
          f"with cache {b['with_cache']['tokens_per_s']:,.0f} tok/s, "
          f"TTFT {b['with_cache']['ttft_ms']:.1f} ms, "
          f"decode p50 {b['with_cache']['decode_p50_ms']:.2f} ms")


if __name__ == "__main__":
    main()
