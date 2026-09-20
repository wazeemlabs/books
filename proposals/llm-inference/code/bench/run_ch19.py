"""Chapter 19: the two phases on different machines.

Chapter 18 interleaved prefill and decode on one accelerator, a few
hundred prompt tokens at a time, and that removed the stall. This is
the other answer to the same problem: put the phases on different
machines entirely and move the keys and values between them.

Four measurements:

1. What a KV cache costs to move, for the reference model, over each
   link a datacentre actually has.
2. A fixed fleet, split every way between prefill and decode machines:
   where the best split is and how sharp the optimum is.
3. The best split against the same fleet doing both phases everywhere.
4. Where disaggregation stops paying: short prompts, a slow link, and
   a fleet too small to split.

Nothing here runs a model. What a prefill and a decode step cost comes
from `tinyserve/serving.py`; link speeds are published specifications
recorded in FACTS.md. What is simulated is the fleet.

    python3 -m bench.run_ch19
"""

from __future__ import annotations

import math

import numpy as np

from tinyserve.paged import BLOCK_SIZE
from tinyserve.reference import (CONTEXT_TOKENS, GPU_BYTES, KV_BYTES_PER_TOKEN,
                                 PROMPT_TOKENS, REQUESTS_PER_S, WEIGHT_BYTES)
from tinyserve.scheduler import Request, serve_colocated, serve_disaggregated
from tinyserve.serving import decode_step, prefill_step

from .harness import pct, write

# The case study's traffic shape (STANDARDS.md section 7).
PROMPT_MEAN, PROMPT_CV = 1200, 0.6
OUTPUT_MEAN, OUTPUT_CV = 300, 0.8
MAX_OUTPUT = 1024

RATE = REQUESTS_PER_S          # the case study's busy hour: 200 a second
N_REQUESTS = 2000
WARM_FRACTION = 0.1            # of the arrival window, discarded as warm-up
FLEET = 12                     # accelerators, the same number for both designs
MAX_BATCH = 256
TOKEN_BUDGET = 512             # Chapter 18's choice, used by the colocated arm
SEED = 0

TTFT_BUDGET_MS, ITL_BUDGET_MS = 1000, 50

# A long-prompt regime the fleet can actually serve: the prompt tokens
# arriving per second are held at the case study's, so the comparison
# is not confounded by a saturated server.
LONG_PROMPT = 4000
LONG_RATE = RATE * PROMPT_MEAN / LONG_PROMPT

# Links a cache could cross between two machines (FACTS.md, 2026-09).
LINKS = [("NVLink, same node", 900e9),
         ("InfiniBand NDR", 50e9),
         ("PCIe 5.0 x16", 64e9),
         ("100 GbE", 12.5e9),
         ("25 GbE", 3.125e9)]
DEFAULT_LINK = "InfiniBand NDR"


def lognormal(mean: float, cv: float, n: int, rng) -> np.ndarray:
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = math.log(mean) - sigma**2 / 2
    return np.maximum(1, rng.lognormal(mu, sigma, n).round()).astype(int)


def make_requests(n: int, rate: float, rng, prompt_mean: int = PROMPT_MEAN
                  ) -> list[Request]:
    gaps = rng.exponential(1 / rate, n)
    arrivals = np.cumsum(gaps)
    prompts = lognormal(prompt_mean, PROMPT_CV, n, rng)
    outputs = np.minimum(lognormal(OUTPUT_MEAN, OUTPUT_CV, n, rng), MAX_OUTPUT)
    return [Request(id=i, arrival_s=float(a), prompt_tokens=int(p),
                    output_tokens=int(o))
            for i, (a, p, o) in enumerate(zip(arrivals, prompts, outputs))]


def summarize(trace, rate: float, **extra) -> dict:
    ttft = [(r.first_token_s - r.arrival_s) * 1e3 for r in trace.requests
            if r.first_token_s is not None]
    total = [r.finish_s - r.arrival_s for r in trace.requests
             if r.finish_s is not None]
    gaps = trace.gaps_ms
    first = [r.first_gap_ms for r in trace.requests if r.first_gap_ms is not None]
    span = max(r.arrival_s for r in trace.requests)
    out = {
        "rate": rate,
        "requests": len(trace.requests),
        "makespan_s": trace.makespan_s,
        "tokens_per_s": trace.steady_tokens_per_s(span * WARM_FRACTION, span),
        "whole_trace_tokens_per_s": trace.tokens_per_s,
        "window_s": span * (1 - WARM_FRACTION),
        "second_token_p50_ms": pct(first, 50),
        "second_token_p99_ms": pct(first, 99),
        "offered_tokens_per_s": rate * OUTPUT_MEAN,
        "drained_over_span": trace.makespan_s / span,
        "ttft_p50_ms": pct(ttft, 50), "ttft_p99_ms": pct(ttft, 99),
        "itl_p50_ms": pct(gaps, 50), "itl_p99_ms": pct(gaps, 99),
        "itl_max_ms": max(gaps) if gaps else 0.0,
        "total_p50_s": pct(total, 50), "total_p99_s": pct(total, 99),
        "mean_batch": trace.mean_batch,
        "iterations": trace.iterations,
        "prefill_iterations": trace.prefill_iterations,
        "preemptions": trace.preemptions,
        "transfer_s": trace.transfer_s,
        "peak_blocks": trace.peak_blocks,
        "meets_ttft": pct(ttft, 99) <= TTFT_BUDGET_MS,
        "meets_itl": pct(gaps, 99) <= ITL_BUDGET_MS if gaps else False,
        "keeping_up": (trace.makespan_s <= span * 1.10
                       and pct(ttft, 99) <= TTFT_BUDGET_MS),
    }
    out.update(extra)
    return out


def transfer_arithmetic() -> dict:
    """What one sequence's cache costs to move, before any simulation."""
    rows = []
    for tokens in (256, 512, PROMPT_TOKENS, CONTEXT_TOKENS, 4096, 8192):
        nbytes = tokens * KV_BYTES_PER_TOKEN
        rows.append({
            "tokens": tokens,
            "bytes": nbytes,
            "prefill_s": prefill_step(tokens).seconds,
            "over": {name: nbytes / bw for name, bw in LINKS},
            # A link fast enough to move the cache in the time the
            # prefill took: below this, the move is the bottleneck.
            "link_to_match_prefill": nbytes / prefill_step(tokens).seconds,
        })
    return {"rows": rows, "links": dict(LINKS),
            "decode_step_ms": decode_step(64, CONTEXT_TOKENS).inter_token_ms}


def split_sweep(blocks: int, link: float) -> list[dict]:
    """One fleet, every way of dividing it between the two phases."""
    rows = []
    for prefill in range(1, FLEET):
        decode = FLEET - prefill
        reqs = make_requests(N_REQUESTS, RATE, np.random.default_rng(SEED))
        trace = serve_disaggregated(reqs, prefill_workers=prefill,
                                    decode_workers=decode, blocks=blocks,
                                    link_bytes_per_s=link, max_batch=MAX_BATCH)
        rows.append(summarize(trace, RATE, prefill_workers=prefill,
                              decode_workers=decode, design="disaggregated",
                              link=DEFAULT_LINK))
    return rows


def colocated(blocks: int, workers: int = FLEET, rate: float = RATE,
              prompt_mean: int = PROMPT_MEAN) -> dict:
    reqs = make_requests(N_REQUESTS, rate, np.random.default_rng(SEED),
                         prompt_mean)
    trace = serve_colocated(reqs, workers=workers, blocks=blocks,
                            token_budget=TOKEN_BUDGET, max_batch=MAX_BATCH)
    return summarize(trace, rate, workers=workers, design="colocated",
                     prompt_mean=prompt_mean)


def one_split(blocks: int, prefill: int, link: float, rate: float = RATE,
              prompt_mean: int = PROMPT_MEAN, fleet: int = FLEET,
              link_name: str = DEFAULT_LINK) -> dict:
    reqs = make_requests(N_REQUESTS, rate, np.random.default_rng(SEED),
                         prompt_mean)
    trace = serve_disaggregated(reqs, prefill_workers=prefill,
                                decode_workers=fleet - prefill, blocks=blocks,
                                link_bytes_per_s=link, max_batch=MAX_BATCH)
    return summarize(trace, rate, prefill_workers=prefill,
                     decode_workers=fleet - prefill, design="disaggregated",
                     link=link_name, prompt_mean=prompt_mean, workers=fleet)


def link_sweep(blocks: int, prefill: int) -> list[dict]:
    """The same fleet over every link, from NVLink down to slow Ethernet."""
    return [one_split(blocks, prefill, bw, link_name=name)
            for name, bw in LINKS]


def best_split(blocks: int, link: float, rate: float = RATE,
               prompt_mean: int = PROMPT_MEAN, fleet: int = FLEET,
               link_name: str = DEFAULT_LINK) -> dict:
    """The split that delivers most, searched rather than assumed.

    A fleet divided for one prompt distribution is not divided right
    for another: longer prompts need more prefill machines. Comparing
    a tuned colocated fleet against an untuned split would be a
    strawman, so every comparison here re-searches.
    """
    best = None
    for prefill in range(1, fleet):
        r = one_split(blocks, prefill, link, rate=rate,
                      prompt_mean=prompt_mean, fleet=fleet,
                      link_name=link_name)
        if best is None or r["tokens_per_s"] > best["tokens_per_s"]:
            best = r
    return best


def prompt_sweep(blocks: int, link: float) -> list[dict]:
    """Where the prompt is short, there is little prefill to move away."""
    rows = []
    for mean in (100, 300, PROMPT_MEAN, 4000):
        rows.append(best_split(blocks, link, prompt_mean=mean))
        rows.append(colocated(blocks, prompt_mean=mean))
    return rows


def fleet_sweep(blocks: int, link: float) -> list[dict]:
    """A fleet too small to split is a fleet that cannot disaggregate."""
    rows = []
    for fleet in (2, 4, 8, 12):
        rate = RATE * fleet / FLEET
        best = best_split(blocks, link, rate=rate, fleet=fleet)
        rows.append({"fleet": fleet, "best": best,
                     "colocated": colocated(blocks, workers=fleet, rate=rate)})
    return rows


def main() -> None:
    pool_bytes = GPU_BYTES - WEIGHT_BYTES
    blocks = int(pool_bytes // (BLOCK_SIZE * KV_BYTES_PER_TOKEN))
    link = dict(LINKS)[DEFAULT_LINK]

    splits = split_sweep(blocks, link)
    best = max(splits, key=lambda r: r["tokens_per_s"])
    co = colocated(blocks)

    payload = {
        "transfer": transfer_arithmetic(),
        "splits": splits,
        "best_split": best,
        "colocated": co,
        "links": link_sweep(blocks, best["prefill_workers"]),
        "prompts": prompt_sweep(blocks, link),
        "long_prompts": [
            best_split(blocks, link, rate=LONG_RATE, prompt_mean=LONG_PROMPT),
            colocated(blocks, rate=LONG_RATE, prompt_mean=LONG_PROMPT),
        ],
        "fleets": fleet_sweep(blocks, link),
        "assumptions": {
            "fleet": FLEET, "blocks_per_worker": blocks,
            "pool_bytes": pool_bytes, "block": BLOCK_SIZE,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "max_batch": MAX_BATCH, "token_budget": TOKEN_BUDGET,
            "n_requests": N_REQUESTS, "rate": RATE, "seed": SEED,
            "prompt_mean": PROMPT_MEAN, "output_mean": OUTPUT_MEAN,
            "warm_fraction": WARM_FRACTION,
            "context": CONTEXT_TOKENS,
            "ttft_budget_ms": TTFT_BUDGET_MS, "itl_budget_ms": ITL_BUDGET_MS,
            "link": DEFAULT_LINK, "link_bytes_per_s": link,
            "long_prompt": LONG_PROMPT, "long_rate": LONG_RATE,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }

    path = write("results/ch19.json", payload)
    print(f"wrote {path}")

    print("  moving one sequence's cache:")
    for r in payload["transfer"]["rows"]:
        over = "  ".join(f"{n} {v * 1e3:7.1f} ms" for n, v in r["over"].items())
        print(f"    {r['tokens']:5,} tokens = {r['bytes'] / 1e6:6.0f} MB: "
              f"prefill {r['prefill_s'] * 1e3:6.1f} ms | {over}")

    print(f"  a fleet of {FLEET} at {RATE} req/s over {DEFAULT_LINK}:")
    for r in splits:
        mark = " <-" if r is best else ""
        print(f"    {r['prefill_workers']:2}P + {r['decode_workers']:2}D: "
              f"{r['tokens_per_s']:7,.0f} tok/s, "
              f"TTFT p50 {r['ttft_p50_ms']:8,.0f} p99 {r['ttft_p99_ms']:9,.0f} ms, "
              f"ITL p50 {r['itl_p50_ms']:5.1f} p99 {r['itl_p99_ms']:6.1f} ms, "
              f"batch {r['mean_batch']:6.1f}{mark}")
    print(f"    {FLEET} colocated: {co['tokens_per_s']:7,.0f} tok/s, "
          f"TTFT p50 {co['ttft_p50_ms']:8,.0f} p99 {co['ttft_p99_ms']:9,.0f} ms, "
          f"ITL p50 {co['itl_p50_ms']:5.1f} p99 {co['itl_p99_ms']:6.1f} ms, "
          f"batch {co['mean_batch']:6.1f}")

    print("  over each link (the transfer lands in the wait for token 2):")
    for r in payload["links"]:
        print(f"    {r['link']:20}: {r['tokens_per_s']:7,.0f} tok/s, "
              f"token 2 p50 {r['second_token_p50_ms']:6.1f} p99 "
              f"{r['second_token_p99_ms']:7.1f} ms, "
              f"ITL p99 {r['itl_p99_ms']:6.1f} ms")
    c = payload["colocated"]
    print(f"    {'colocated':20}: {c['tokens_per_s']:7,.0f} tok/s, "
          f"token 2 p50 {c['second_token_p50_ms']:6.1f} p99 "
          f"{c['second_token_p99_ms']:7.1f} ms, "
          f"ITL p99 {c['itl_p99_ms']:6.1f} ms")

    print("  by prompt length (split re-searched for each):")
    for r in payload["prompts"]:
        tag = (f"{r['design']} {r.get('prefill_workers', '')}P"
               if r["design"] == "disaggregated" else r["design"])
        print(f"    {r['prompt_mean']:5,}-token prompts, {tag:18}: "
              f"{r['tokens_per_s']:7,.0f} tok/s, "
              f"ITL p99 {r['itl_p99_ms']:7.1f} ms, "
              f"TTFT p99 {r['ttft_p99_ms']:8,.0f} ms")

    print(f"  {LONG_PROMPT:,}-token prompts at {LONG_RATE:.0f} req/s "
          "(the same prompt tokens a second):")
    for r in payload["long_prompts"]:
        tag = (f"disaggregated {r.get('prefill_workers')}P+{r.get('decode_workers')}D"
               if r["design"] == "disaggregated" else "colocated")
        print(f"    {tag:24}: {r['tokens_per_s']:7,.0f} tok/s, "
              f"TTFT p50 {r['ttft_p50_ms']:7,.0f} p99 {r['ttft_p99_ms']:8,.0f} ms, "
              f"ITL p50 {r['itl_p50_ms']:5.1f} p99 {r['itl_p99_ms']:6.1f} ms, "
              f"token 2 p50 {r['second_token_p50_ms']:6.1f} ms")

    print("  by fleet size (load scaled with it):")
    for r in payload["fleets"]:
        b, c = r["best"], r["colocated"]
        print(f"    {r['fleet']:2} accelerators: best split "
              f"{b['prefill_workers']}P+{b['decode_workers']}D "
              f"{b['tokens_per_s']:7,.0f} tok/s (ITL p99 {b['itl_p99_ms']:6.1f}) "
              f"vs colocated {c['tokens_per_s']:7,.0f} "
              f"(ITL p99 {c['itl_p99_ms']:6.1f})")


if __name__ == "__main__":
    main()
