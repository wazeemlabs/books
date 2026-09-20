"""Chapter 17: continuous batching, simulated against the static baseline.

Chapter 16 left a fixed batch spending most of what it paid for on
padding and dead slots. This replaces the decision that caused it --
choose the batch once -- with Orca's: choose it again every iteration.

Four measurements:

1. The same requests through both schedulers, one rate, side by side.
2. Both schedulers under rising load: where each one stops keeping up.
3. What a user's wait between tokens looks like when somebody else's
   prompt lands in the middle of it.
4. What it costs in memory, and how often the pool runs out.

Nothing here runs a model. The cost of a prefill and of a decode step
come from `tinyserve/serving.py`, which is arithmetic over the
reference model and published hardware specifications. What is
simulated, and what is being measured, is the scheduler.

    python3 -m bench.run_ch17
"""

from __future__ import annotations

import math

import numpy as np

from tinyserve.paged import BLOCK_SIZE
from tinyserve.reference import (CONTEXT_TOKENS, GPU_BYTES, KV_BYTES_PER_TOKEN,
                                 WEIGHT_BYTES)
from tinyserve.scheduler import Request, serve_continuous, serve_static
from tinyserve.serving import decode_step, prefill_step

from .harness import pct, write

# The case study's traffic shape (STANDARDS.md section 7).
PROMPT_MEAN, PROMPT_CV = 1200, 0.6
OUTPUT_MEAN, OUTPUT_CV = 300, 0.8
MAX_OUTPUT = 1024

RATES = [4, 8, 12, 16, 20, 24, 28, 32]      # requests per second
HEAD_TO_HEAD_RATE = 12
N_REQUESTS = 600
MAX_BATCH = 256
SEED = 0

# p99 budgets the case study promises.
TTFT_BUDGET_MS, ITL_BUDGET_MS = 1000, 50


def lognormal(mean: float, cv: float, n: int, rng) -> np.ndarray:
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = math.log(mean) - sigma**2 / 2
    return np.maximum(1, rng.lognormal(mu, sigma, n).round()).astype(int)


def make_requests(n: int, rate: float, rng) -> list[Request]:
    """A Poisson arrival stream with the case study's lengths."""
    gaps = rng.exponential(1 / rate, n)
    arrivals = np.cumsum(gaps)
    prompts = lognormal(PROMPT_MEAN, PROMPT_CV, n, rng)
    outputs = np.minimum(lognormal(OUTPUT_MEAN, OUTPUT_CV, n, rng), MAX_OUTPUT)
    return [Request(id=i, arrival_s=float(a), prompt_tokens=int(p),
                    output_tokens=int(o))
            for i, (a, p, o) in enumerate(zip(arrivals, prompts, outputs))]


def summarize(trace, rate: float, policy: str, blocks: int) -> dict:
    ttft = [(r.first_token_s - r.arrival_s) * 1e3 for r in trace.requests
            if r.first_token_s is not None]
    total = [(r.finish_s - r.arrival_s) for r in trace.requests
             if r.finish_s is not None]
    gaps = trace.gaps_ms
    offered = rate * OUTPUT_MEAN
    span = max(r.arrival_s for r in trace.requests)
    return {
        "policy": policy, "rate": rate,
        "requests": len(trace.requests),
        "makespan_s": trace.makespan_s,
        "tokens_per_s": trace.tokens_per_s,
        "offered_tokens_per_s": offered,
        "arrival_span_s": span,
        "drained_over_span": trace.makespan_s / span,
        "keeping_up": (trace.makespan_s <= span * 1.10
                       and pct(ttft, 99) <= TTFT_BUDGET_MS),
        "ttft_p50_ms": pct(ttft, 50), "ttft_p99_ms": pct(ttft, 99),
        "total_p50_s": pct(total, 50), "total_p99_s": pct(total, 99),
        "itl_p50_ms": pct(gaps, 50), "itl_p99_ms": pct(gaps, 99),
        "itl_max_ms": max(gaps) if gaps else 0.0,
        "mean_batch": trace.mean_batch,
        "mean_slots": trace.slots / max(trace.iterations
                                        - trace.prefill_iterations, 1),
        "batch_fill": trace.mean_batch / MAX_BATCH,
        "slot_utilization": trace.slot_utilization,
        "prefill_share": trace.prefill_s / trace.makespan_s,
        "decode_share": trace.decode_s / trace.makespan_s,
        "idle_share": trace.idle_s / trace.makespan_s,
        "iterations": trace.iterations,
        "prefill_iterations": trace.prefill_iterations,
        "preemptions": trace.preemptions,
        "recomputed_tokens": trace.recomputed_tokens,
        "peak_blocks": trace.peak_blocks,
        "peak_pool_share": trace.peak_blocks / blocks,
        "meets_ttft": pct(ttft, 99) <= TTFT_BUDGET_MS,
        "meets_itl": pct(gaps, 99) <= ITL_BUDGET_MS if gaps else False,
    }


POOL_SHARES = [1.0, 0.25, 0.10, 0.05, 0.03, 0.02]


def pool_sweep(blocks: int) -> list[dict]:
    """The same traffic through smaller and smaller pools.

    Continuous batching keeps admitting until the memory runs out, so
    the pool is what caps the batch -- and when it runs out, the
    scheduler has to take a sequence back out again, throwing away the
    tokens it had already produced.
    """
    rows = []
    for share in POOL_SHARES:
        n = max(1, int(blocks * share))
        reqs = make_requests(N_REQUESTS, HEAD_TO_HEAD_RATE,
                             np.random.default_rng(SEED))
        trace = serve_continuous(reqs, max_batch=MAX_BATCH, blocks=n)
        row = summarize(trace, HEAD_TO_HEAD_RATE, "continuous", n)
        row["pool_share"] = share
        row["blocks"] = n
        row["pool_gb"] = n * BLOCK_SIZE * KV_BYTES_PER_TOKEN / 1e9
        row["wasted_token_share"] = (trace.recomputed_tokens
                                     / max(trace.output_tokens, 1))
        rows.append(row)
    return rows


def timeline(rate: float, n: int, blocks: int, rng) -> dict:
    """A handful of requests under each policy, for the figure."""
    out = {}
    for name, serve in (("static", serve_static), ("continuous", serve_continuous)):
        reqs = make_requests(n, rate, np.random.default_rng(SEED + 99))
        serve(reqs, max_batch=4, blocks=blocks)
        out[name] = [{"id": r.id, "arrival_s": r.arrival_s,
                      "first_token_s": r.first_token_s,
                      "finish_s": r.finish_s,
                      "prompt": r.prompt_tokens, "output": r.output_tokens}
                     for r in reqs]
    return out


def main() -> None:
    pool_bytes = GPU_BYTES - WEIGHT_BYTES
    blocks = int(pool_bytes // (BLOCK_SIZE * KV_BYTES_PER_TOKEN))

    rows = []
    for rate in RATES:
        for name, serve in (("static", serve_static),
                            ("continuous", serve_continuous)):
            reqs = make_requests(N_REQUESTS, rate, np.random.default_rng(SEED))
            trace = serve(reqs, max_batch=MAX_BATCH, blocks=blocks)
            rows.append(summarize(trace, rate, name, blocks))

    head = {r["policy"]: r for r in rows if r["rate"] == HEAD_TO_HEAD_RATE}

    # What the wait between tokens would be with no prefills in the way:
    # the decode step alone, at the batch the scheduler actually ran.
    pure = decode_step(round(head["continuous"]["mean_batch"]),
                       CONTEXT_TOKENS).inter_token_ms

    payload = {
        "rows": rows,
        "head_to_head": head,
        "pure_decode_itl_ms": pure,
        "mean_prefill_ms": prefill_step(PROMPT_MEAN).seconds * 1e3,
        "timeline": timeline(6.0, 12, blocks, np.random.default_rng(SEED)),
        "pool_sweep": pool_sweep(blocks),
        "assumptions": {
            "blocks": blocks, "block": BLOCK_SIZE,
            "pool_bytes": pool_bytes,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "max_batch": MAX_BATCH, "n_requests": N_REQUESTS,
            "context": CONTEXT_TOKENS, "seed": SEED,
            "prompt_mean": PROMPT_MEAN, "output_mean": OUTPUT_MEAN,
            "head_to_head_rate": HEAD_TO_HEAD_RATE,
            "ttft_budget_ms": TTFT_BUDGET_MS, "itl_budget_ms": ITL_BUDGET_MS,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }

    path = write("results/ch17.json", payload)
    print(f"wrote {path}")
    print(f"  pool: {blocks:,} blocks; decode-only wait at the batch it ran: "
          f"{pure:.1f} ms")
    for r in rows:
        print(f"  {r['policy']:11} {r['rate']:3.0f} req/s: "
              f"{r['tokens_per_s']:7,.0f} tok/s of {r['offered_tokens_per_s']:,.0f} "
              f"offered{'  ' if r['keeping_up'] else ' *'}, "
              f"TTFT p50 {r['ttft_p50_ms']:9,.0f} ms p99 {r['ttft_p99_ms']:9,.0f}, "
              f"ITL p50 {r['itl_p50_ms']:6.1f} p99 {r['itl_p99_ms']:7.1f}, "
              f"batch {r['mean_batch']:6.1f}, live slots "
              f"{r['slot_utilization'] * 100:5.1f}%, "
              f"pool {r['peak_pool_share'] * 100:5.1f}%, "
              f"preempted {r['preemptions']}")
    print("  * not keeping up with the offered load")
    print("  pool sweep, continuous, "
          f"{HEAD_TO_HEAD_RATE} req/s:")
    for r in payload["pool_sweep"]:
        print(f"    {r['pool_share'] * 100:5.0f}% of the pool "
              f"({r['pool_gb']:5.1f} GB): {r['tokens_per_s']:6,.0f} tok/s, "
              f"batch {r['mean_batch']:6.1f}, "
              f"TTFT p99 {r['ttft_p99_ms']:8,.0f} ms, "
              f"{r['preemptions']:4} preemptions, "
              f"{r['wasted_token_share'] * 100:4.1f}% of tokens generated twice")


if __name__ == "__main__":
    main()
