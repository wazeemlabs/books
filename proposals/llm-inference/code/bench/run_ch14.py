"""Chapter 14: paging the cache, built and measured.

Chapter 13 predicted what paging would be worth from arithmetic. This
builds it and measures it:

1. It must not change the answer.
2. What it costs per decode step, since the blocks must be gathered.
3. How many sequences actually fit in a fixed pool, against reserving
   each sequence's whole possible length.
4. How the block size trades waste against overhead.

    python3 -m bench.run_ch14
"""

from __future__ import annotations

import math
from statistics import median
from time import perf_counter

import numpy as np

from tinyserve.model import Config, KVCache, build, forward
from tinyserve.paged import BLOCK_SIZE, BlockPool, PagedKVCache
from tinyserve.reference import GPU_BYTES, KV_BYTES_PER_TOKEN, WEIGHT_BYTES

from .harness import Repeated, write

PROMPT, DECODE_STEPS, RUNS = 128, 24, 5
BLOCK_SIZES = [1, 4, 8, 16, 32, 64, 128]
# The case study's traffic, as in Chapter 13.
PROMPT_MEAN, PROMPT_CV, OUTPUT_MEAN, OUTPUT_CV = 1200, 0.6, 300, 0.8
MAX_MODEL_LEN, N_REQUESTS, SEED = 8192, 4000, 0


def lognormal(mean: float, cv: float, n: int, rng) -> np.ndarray:
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = math.log(mean) - sigma**2 / 2
    return np.maximum(1, rng.lognormal(mu, sigma, n).round()).astype(int)


def identical_output(cfg: Config, model) -> dict:
    prompt = np.arange(PROMPT) % cfg.vocab_size
    outs = []
    for make in (lambda: KVCache(cfg, max_seq=PROMPT + DECODE_STEPS + 1),
                 lambda: PagedKVCache(BlockPool(cfg, n_blocks=64))):
        cache = make()
        logits = forward(model, prompt, cache)
        toks = []
        for _ in range(DECODE_STEPS):
            nxt = int(logits[-1].argmax())
            toks.append(nxt)
            logits = forward(model, np.array([nxt]), cache)
        outs.append(toks)
    return {"same_tokens": outs[0] == outs[1], "tokens_compared": DECODE_STEPS}


def step_cost(cfg: Config, model) -> dict:
    prompt = np.arange(PROMPT) % cfg.vocab_size
    out = {}
    for name, make in (("contiguous", lambda: KVCache(cfg, max_seq=PROMPT + DECODE_STEPS + 2)),
                       ("paged", lambda: PagedKVCache(BlockPool(cfg, n_blocks=64)))):
        medians = []
        for _ in range(RUNS):
            cache = make()
            forward(model, prompt, cache)
            nxt, ts = np.array([0]), []
            for _ in range(DECODE_STEPS):
                t0 = perf_counter(); forward(model, nxt, cache)
                ts.append(perf_counter() - t0)
            medians.append(median(ts))
        out[name] = Repeated(medians).summary()
    out["paged_over_contiguous"] = (out["paged"]["median"]
                                    / out["contiguous"]["median"])
    return out


def capacity(lengths: np.ndarray, kv_bytes_per_token: int,
             pool_bytes: int, block_size: int) -> dict:
    """How many sequences fit, paged against reserving the full context."""
    per_block = block_size * kv_bytes_per_token
    n_blocks = int(pool_bytes // per_block)

    admitted, blocks_used, live, reserved = 0, 0, 0, 0
    for total in lengths:
        need = math.ceil(total / block_size)
        if blocks_used + need > n_blocks:
            break
        blocks_used += need
        admitted += 1
        live += total
        reserved += need * block_size

    contiguous = int(pool_bytes // (MAX_MODEL_LEN * kv_bytes_per_token))
    return {
        "block_size": block_size, "blocks": n_blocks,
        "admitted_paged": admitted, "admitted_contiguous": contiguous,
        "gain": admitted / max(contiguous, 1),
        "utilization": live / reserved if reserved else 0.0,
        "wasted_tokens_per_sequence": (reserved - live) / max(admitted, 1),
        "worst_case_waste_per_sequence": block_size - 1,
    }


def main() -> None:
    cfg = Config()
    model = build(cfg)
    # Capacity is computed at the reference model's scale, so that this
    # chapter's built allocator can be checked against Chapter 13's
    # prediction for the same traffic. The step cost above is tinyserve's.
    kv_per_token = KV_BYTES_PER_TOKEN

    rng = np.random.default_rng(SEED)
    prompts = lognormal(PROMPT_MEAN, PROMPT_CV, N_REQUESTS, rng)
    outputs = np.minimum(lognormal(OUTPUT_MEAN, OUTPUT_CV, N_REQUESTS, rng), 1024)
    totals = prompts + outputs
    totals = totals[totals <= MAX_MODEL_LEN]

    pool_bytes = GPU_BYTES - WEIGHT_BYTES  # what Chapter 13 assumed
    sizes = [capacity(totals, kv_per_token, pool_bytes, b) for b in BLOCK_SIZES]
    at_default = next(s for s in sizes if s["block_size"] == BLOCK_SIZE)

    # A pool full of sequences, then one finishes and its blocks return.
    pool = BlockPool(cfg, n_blocks=32)
    seqs = [PagedKVCache(pool) for _ in range(3)]
    for s in seqs:
        s._grow_to(40)
    before = pool.blocks_in_use
    seqs[1].release()
    after = pool.blocks_in_use

    payload = {
        "equivalence": identical_output(cfg, model),
        "step_cost": step_cost(cfg, model),
        "block_sizes": sizes,
        "default_block_size": BLOCK_SIZE,
        "at_default": at_default,
        "reuse": {"blocks_in_use_before": before, "blocks_in_use_after": after,
                  "returned": before - after},
        "assumptions": {"pool_gb": pool_bytes / 1000**3,
                        "kv_bytes_per_token": kv_per_token,
                        "max_model_len": MAX_MODEL_LEN,
                        "requests_sampled": int(len(totals)), "seed": SEED},
    }

    path = write("results/ch14.json", payload)
    print(f"wrote {path}")
    print(f"  identical tokens: {payload['equivalence']['same_tokens']}")
    sc = payload["step_cost"]
    print(f"  decode step: contiguous {sc['contiguous']['median'] * 1e3:.3f} ms, "
          f"paged {sc['paged']['median'] * 1e3:.3f} ms "
          f"({sc['paged_over_contiguous']:.2f}x)")
    for s in sizes:
        print(f"  block {s['block_size']:4d}: {s['admitted_paged']:5d} sequences fit "
              f"(contiguous {s['admitted_contiguous']}), utilization "
              f"{s['utilization'] * 100:5.1f}%, waste "
              f"{s['wasted_tokens_per_sequence']:5.1f} tok/seq")
    print(f"  releasing one sequence returned {payload['reuse']['returned']} blocks")


if __name__ == "__main__":
    main()
