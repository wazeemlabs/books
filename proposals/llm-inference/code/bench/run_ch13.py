"""Chapter 13 experiment: how much of the KV cache is actually in use.

Chapter 12 showed the cache is what makes decoding fast and what makes
it expensive. This chapter asks the next question: of the memory a
server holds for KV caches, how much holds live data?

Two parts:

1. A direct measurement on `tinyserve`, whose allocator reserves
   `max_seq` positions per sequence whether or not they are used.
2. An accounting of the same policy at production scale, over the
   book's case-study traffic (STANDARDS.md section 7). This part is
   arithmetic over sampled request lengths, not a timing run: it has no
   run-to-run noise, and the only randomness is the seeded traffic.

    python3 -m bench.run_ch13
"""

from __future__ import annotations

import math

import numpy as np

from tinyserve.generate import cached
from tinyserve.model import Config, KVCache, build

from tinyserve.reference import (GPU_BYTES, KV_BYTES_PER_TOKEN, WEIGHT_BYTES)

from .harness import pct, write

# Case study: a customer-support assistant. See STANDARDS.md section 7.
PROMPT_MEAN, PROMPT_CV = 1200, 0.6
OUTPUT_MEAN, OUTPUT_CV = 300, 0.8
N_REQUESTS = 20_000
SEED = 0

MAX_MODEL_LEN = 8192   # the context the server advertises
MAX_NEW = 1024         # the cap a client may request
BLOCK = 16             # Chapter 14's page size, previewed here

FREE_BYTES = GPU_BYTES - WEIGHT_BYTES


def lognormal(mean: float, cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Positive, right-skewed request lengths with a given mean and spread.

    Real prompt and response lengths have a long right tail; a normal
    distribution would produce negative lengths and understate it.
    """
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = math.log(mean) - sigma**2 / 2
    return np.maximum(1, rng.lognormal(mu, sigma, n).round()).astype(int)


def policies(prompt: np.ndarray, output: np.ndarray) -> dict[str, np.ndarray]:
    """Tokens of KV cache each policy holds for a sequence, for its whole life.

    A contiguous allocator must commit a sequence's slot up front: it
    cannot know how long the answer will be, and cannot hand the unused
    tail to another request while the sequence is alive.
    """
    total = prompt + output
    return {
        "max_model_len": np.full_like(prompt, MAX_MODEL_LEN),
        "prompt_plus_cap": prompt + MAX_NEW,
        "oracle": total,  # unattainable: needs the answer's length in advance
        "paged_16": np.ceil(total / BLOCK).astype(int) * BLOCK,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    prompt = lognormal(PROMPT_MEAN, PROMPT_CV, N_REQUESTS, rng)
    output = lognormal(OUTPUT_MEAN, OUTPUT_CV, N_REQUESTS, rng)
    output = np.minimum(output, MAX_NEW)
    keep = prompt + output <= MAX_MODEL_LEN  # the server would refuse the rest
    prompt, output = prompt[keep], output[keep]

    # A sequence uses prompt+t tokens at decode step t, so averaged over
    # its life it holds prompt + output/2. That average is what another
    # request could have used, had the allocator been able to hand it over.
    live = prompt + output / 2

    rows = []
    for name, reserved in policies(prompt, output).items():
        util = live.sum() / reserved.sum()
        rows.append({
            "policy": name,
            "reserved_tokens_mean": float(reserved.mean()),
            "live_tokens_mean": float(live.mean()),
            "utilization": float(util),
            "waste": float(1 - util),
            "bytes_per_seq": float(reserved.mean() * KV_BYTES_PER_TOKEN),
            "concurrent_seqs": int(FREE_BYTES // (reserved.mean() * KV_BYTES_PER_TOKEN)),
        })

    # The same allocator, measured rather than modelled, on tinyserve.
    cfg = Config()
    model = build(cfg)
    prompt_len, n_new = 128, 128
    cache = KVCache(cfg)  # default: reserve cfg.max_seq positions
    cached(model, list(range(prompt_len)), n_new)
    measured = {
        "max_seq": cache.max_seq,
        "allocated_bytes": cache.nbytes,
        "used_bytes": (prompt_len + n_new) * cache.bytes_per_token(),
        "utilization": (prompt_len + n_new) / cache.max_seq,
    }

    payload = {
        "experiment": {
            "n_requests": int(keep.sum()), "seed": SEED,
            "prompt_mean": PROMPT_MEAN, "prompt_cv": PROMPT_CV,
            "output_mean": OUTPUT_MEAN, "output_cv": OUTPUT_CV,
            "max_model_len": MAX_MODEL_LEN, "max_new": MAX_NEW, "block": BLOCK,
            "rejected_over_context": int((~keep).sum()),
        },
        "traffic": {
            "prompt_p50": pct(prompt, 50), "prompt_p99": pct(prompt, 99),
            "output_p50": pct(output, 50), "output_p99": pct(output, 99),
            "total_p50": pct(prompt + output, 50),
            "total_p99": pct(prompt + output, 99),
        },
        "hardware": {
            "gpu_bytes": GPU_BYTES, "weight_bytes": WEIGHT_BYTES,
            "free_bytes": FREE_BYTES, "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        },
        "policies": rows,
        "measured_tinyserve": measured,
    }

    path = write("results/ch13.json", payload)
    print(f"wrote {path}")
    for r in rows:
        print(f"  {r['policy']:16} holds {r['reserved_tokens_mean']:7.0f} tok/seq"
              f"  utilization {r['utilization']*100:5.1f}%"
              f"  concurrency {r['concurrent_seqs']:4d}")
    print(f"  tinyserve measured: {measured['utilization']*100:.0f}% of its"
          f" {measured['allocated_bytes']/1024**2:.0f} MiB allocation in use")


if __name__ == "__main__":
    main()
