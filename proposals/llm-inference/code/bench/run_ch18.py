"""Chapter 18: chunked prefill, and the policies a scheduler needs.

Chapter 17 left one number going the wrong way. Continuous batching
won everything except the wait *between* tokens, whose 99th percentile
climbed to many times the decode step, because a prompt admitted in
the middle of your reply stops your reply for as long as it takes to
read.

This measures the fix and what it costs.

1. One user decoding while long prompts arrive, step by step: what the
   gap between their tokens looks like whole and chunked.
2. The token budget swept from small to large: the trade the vLLM
   documentation describes, measured.
3. Three queue orders on identical traffic: what shortest-job-first
   buys and who pays for it.
4. The two ways out of a full pool -- throw the cache away, or copy it
   to host memory -- against the interconnect that would carry it.

Nothing here runs a model. What a mixed iteration costs comes from
`tinyserve/serving.py`, which is arithmetic over the reference model
and published hardware specifications. What is simulated, and what is
being measured, is the scheduler.

    python3 -m bench.run_ch18
"""

from __future__ import annotations

import math

import numpy as np

from tinyserve.paged import BLOCK_SIZE
from tinyserve.reference import (CONTEXT_TOKENS, GPU_BYTES, KV_BYTES_PER_TOKEN,
                                 WEIGHT_BYTES)
from tinyserve.scheduler import (POLICIES, Request, serve_chunked,
                                 serve_continuous)
from tinyserve.cost import flops_forward
from tinyserve.reference import MODEL
from tinyserve.serving import decode_step, mixed_step, prefill_step

from .harness import pct, steady_state, write

# The case study's traffic shape (STANDARDS.md section 7).
PROMPT_MEAN, PROMPT_CV = 1200, 0.6
OUTPUT_MEAN, OUTPUT_CV = 300, 0.8
MAX_OUTPUT = 1024

RATE = 12                      # requests a second, Chapter 17's head-to-head
RATE_HIGH = 24                 # where Chapter 17's tail broke the promise
# Long enough for the queue to fill at the harder of the two rates.
# At 600 the tail was still building when the run ended and the wait
# for a first token came out about half what it settles at; `settling`
# below records the check, and Chapter 41 explains why it is needed.
N_REQUESTS = 4_800
SETTLING_KEYS = ["ttft_p99_ms", "itl_p99_ms", "tokens_per_s", "total_p99_s"]
MAX_BATCH = 256
SEED = 0
BUDGETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# p99 budgets the case study promises.
TTFT_BUDGET_MS, ITL_BUDGET_MS = 1000, 50

# The interconnect a swapped cache would cross (FACTS.md, PCI-SIG).
PCIE_BYTES_PER_S = 64e9
SWAP_POOL_SHARE = 0.03         # where Chapter 17 found preemption biting


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


def summarize(trace, blocks: int, **extra) -> dict:
    ttft = [(r.first_token_s - r.arrival_s) * 1e3 for r in trace.requests
            if r.first_token_s is not None]
    total = [r.finish_s - r.arrival_s for r in trace.requests
             if r.finish_s is not None]
    # Slowdown: how much longer a request took than it would have alone.
    # The fairness number -- a policy can look good on the mean while
    # making a few requests wait many times their own service time.
    alone = {r.id: (prefill_step(r.prompt_tokens).seconds
                    + r.output_tokens * decode_step(1, CONTEXT_TOKENS).seconds)
             for r in trace.requests}
    slow = [(r.finish_s - r.arrival_s) / alone[r.id] for r in trace.requests
            if r.finish_s is not None]
    gaps = trace.gaps_ms
    span = max(r.arrival_s for r in trace.requests)
    prompt_tokens = sum(r.prompt_tokens for r in trace.requests)
    out = {
        "requests": len(trace.requests),
        "makespan_s": trace.makespan_s,
        "tokens_per_s": trace.tokens_per_s,
        "offered_tokens_per_s": extra.get("rate", RATE) * OUTPUT_MEAN,
        "drained_over_span": trace.makespan_s / span,
        "ttft_p50_ms": pct(ttft, 50), "ttft_p99_ms": pct(ttft, 99),
        "itl_p50_ms": pct(gaps, 50), "itl_p99_ms": pct(gaps, 99),
        "itl_max_ms": max(gaps) if gaps else 0.0,
        "total_p50_s": pct(total, 50), "total_p99_s": pct(total, 99),
        "slowdown_p50": pct(slow, 50), "slowdown_p99": pct(slow, 99),
        "slowdown_max": max(slow) if slow else 0.0,
        "mean_batch": trace.mean_batch,
        "iterations": trace.iterations,
        "prefill_iterations": trace.prefill_iterations,
        "mixed_share": trace.prefill_iterations / max(trace.iterations, 1),
        "chunk_tokens": trace.chunk_tokens,
        "prompt_tokens": prompt_tokens,
        "prompt_reread": trace.chunk_tokens / prompt_tokens,
        "recomputed_prompt_tokens": trace.recomputed_prompt_tokens,
        "preemptions": trace.preemptions,
        "recomputed_tokens": trace.recomputed_tokens,
        "swapped_bytes": trace.swapped_bytes,
        "swap_s": trace.swap_s,
        "peak_blocks": trace.peak_blocks,
        "peak_pool_share": trace.peak_blocks / blocks,
        "meets_ttft": pct(ttft, 99) <= TTFT_BUDGET_MS,
        "meets_itl": pct(gaps, 99) <= ITL_BUDGET_MS if gaps else False,
        "keeping_up": (trace.makespan_s <= span * 1.10
                       and pct(ttft, 99) <= TTFT_BUDGET_MS),
    }
    out.update(extra)
    return out


def interference(blocks: int) -> dict:
    """One user decoding while long prompts land, whole against chunked.

    The smallest experiment in the chapter and the one that explains
    the rest: a single reply in progress, and the gaps in it.
    """
    out = {}
    for name, budget in (("whole", None), ("chunked", 512)):
        reqs = [Request(id=0, arrival_s=0.0, prompt_tokens=200,
                        output_tokens=120)]
        reqs += [Request(id=n, arrival_s=0.25 * n, prompt_tokens=8192,
                         output_tokens=40) for n in range(1, 4)]
        if budget is None:
            trace = serve_continuous(reqs, max_batch=8, blocks=blocks)
        else:
            trace = serve_chunked(reqs, max_batch=8, blocks=blocks,
                                  token_budget=budget)
        victim = reqs[0]
        out[name] = {
            "victim_gaps_ms": victim.gaps_ms,
            "victim_finish_s": victim.finish_s,
            "victim_ttft_ms": victim.first_token_s * 1e3,
            "interlopers": len(reqs) - 1,
            "interloper_prompt": reqs[1].prompt_tokens,
            "victim_prompt": victim.prompt_tokens,
            "victim_output": victim.output_tokens,
            "p99_ms": pct(victim.gaps_ms, 99),
            "max_ms": max(victim.gaps_ms) if victim.gaps_ms else 0.0,
            "median_ms": pct(victim.gaps_ms, 50),
            "over_budget": sum(1 for g in victim.gaps_ms if g > ITL_BUDGET_MS),
            "budget": budget,
        }
    return out


def budget_sweep(blocks: int, rate: float) -> list[dict]:
    """The trade the vLLM documentation describes, measured.

    The first row is Chapter 17's scheduler, which gives a prefill an
    iteration of its own. Everything below it mixes the prefill into
    the same iteration as the decodes, with a budget on how much of it
    fits.
    """
    reqs = make_requests(N_REQUESTS, rate, np.random.default_rng(SEED))
    base = serve_continuous(reqs, max_batch=MAX_BATCH, blocks=blocks)
    rows = [summarize(base, blocks, token_budget=0, policy="fcfs", rate=rate,
                      scheduler="prefill on its own iteration")]
    for budget in BUDGETS:
        reqs = make_requests(N_REQUESTS, rate, np.random.default_rng(SEED))
        trace = serve_chunked(reqs, max_batch=MAX_BATCH, blocks=blocks,
                              token_budget=budget)
        rows.append(summarize(trace, blocks, token_budget=budget,
                              policy="fcfs", rate=rate, scheduler="stall-free"))
    return rows


def policy_sweep(blocks: int, budget: int, rate: float) -> list[dict]:
    """Identical traffic, three queue orders, two amounts of memory.

    A scheduling policy is a decision about who goes first, so it can
    only matter when somebody is waiting. This runs each order twice:
    once with the whole pool, where the server admits almost everyone
    on arrival, and once with the pool squeezed to the point where it
    cannot, which is where a queue actually forms.
    """
    small = max(1, int(blocks * SWAP_POOL_SHARE))
    rows = []
    for pool_name, n in (("full", blocks), ("squeezed", small)):
        for policy in POLICIES:
            reqs = make_requests(N_REQUESTS, rate, np.random.default_rng(SEED))
            trace = serve_chunked(reqs, max_batch=MAX_BATCH, blocks=n,
                                  token_budget=budget, policy=policy)
            rows.append(summarize(trace, n, policy=policy, rate=rate,
                                  pool=pool_name, pool_blocks=n,
                                  pool_gb=n * BLOCK_SIZE * KV_BYTES_PER_TOKEN / 1e9,
                                  token_budget=budget))
    return rows


# Interconnects a swapped cache could cross (FACTS.md, verified 2026-09).
LINKS = [("PCIe 4.0 x16", 32e9), ("PCIe 5.0 x16", 64e9),
         ("NVLink (H100)", 900e9)]


def attention_share() -> dict:
    """How much of a prefill is attention, and what chunking does to it.

    The FLOP model charges attention as a full t_new x t_total
    rectangle rather than the causal half, so splitting a prompt
    reduces its charged arithmetic slightly. This is how much -- the
    chapter states it rather than hoping nobody checks.
    """
    c = MODEL

    def attn(t_new: int, t_total: int) -> int:
        return c.n_layers * 2 * 2 * c.n_heads * t_new * t_total * c.head_dim

    def drift(prompt: int, budget: int) -> float:
        n = math.ceil(prompt / budget)
        split = sum(mixed_step(0, 0, min(budget, prompt - k * budget),
                               k * budget).flops for k in range(n))
        return split / flops_forward(c, prompt, prompt) - 1

    rows = {}
    for prompt in (PROMPT_MEAN, 8192):
        rows[prompt] = {
            "attention_share": attn(prompt, prompt) / flops_forward(c, prompt, prompt),
            "worst_drift": min(drift(prompt, b) for b in BUDGETS),
        }
    return rows


def swap_arithmetic() -> dict:
    """What each way out of a full pool costs for one sequence.

    Recomputing means reading the sequence's context again: a prefill,
    whose cost is arithmetic and scales with the square of the length.
    Swapping means two trips across the interconnect, whose cost is
    bytes and scales linearly. So there is a crossover, and it is worth
    knowing which side of it a given machine sits on.
    """
    rows = []
    for tokens in (256, 512, 1024, CONTEXT_TOKENS, 2048, 4096, 8192):
        recompute = prefill_step(tokens).seconds
        bytes_moved = 2 * tokens * KV_BYTES_PER_TOKEN
        rows.append({
            "tokens": tokens,
            "recompute_s": recompute,
            "bytes": bytes_moved,
            "swap_s": {name: bytes_moved / bw for name, bw in LINKS},
            # The link speed at which the two cost the same.
            "breakeven_bytes_per_s": bytes_moved / recompute,
        })
    return {"rows": rows, "links": dict(LINKS)}


def preemption(blocks: int, budget: int) -> list[dict]:
    """Recompute against swap, in a pool small enough to force the choice."""
    small = max(1, int(blocks * SWAP_POOL_SHARE))
    rows = []
    modes = [("recompute", None)] + [(f"swap over {n}", bw) for n, bw in LINKS]
    for name, bw in modes:
        reqs = make_requests(N_REQUESTS, RATE, np.random.default_rng(SEED))
        trace = serve_chunked(reqs, max_batch=MAX_BATCH, blocks=small,
                              token_budget=budget, swap_bytes_per_s=bw)
        rows.append(summarize(trace, small, mode=name, pool_blocks=small,
                              pool_gb=small * BLOCK_SIZE * KV_BYTES_PER_TOKEN / 1e9,
                              bytes_per_s=bw or 0.0,
                              token_budget=budget))
    return rows


def settling(blocks: int, budget: int, rate: float) -> dict:
    """Evidence that `N_REQUESTS` is long enough to have an answer.

    A tail latency read off a queue that is still filling is too good,
    and nothing about the run says so. This measures the chapter's own
    configuration twice, at `N_REQUESTS` and at twice that, and records
    how far each number moved.
    """
    def run(n: int) -> dict:
        reqs = make_requests(n, rate, np.random.default_rng(SEED))
        trace = serve_chunked(reqs, max_batch=MAX_BATCH, blocks=blocks,
                              token_budget=budget)
        return summarize(trace, blocks, token_budget=budget, rate=rate)
    return steady_state(run, N_REQUESTS, SETTLING_KEYS)


def main() -> None:
    pool_bytes = GPU_BYTES - WEIGHT_BYTES
    blocks = int(pool_bytes // (BLOCK_SIZE * KV_BYTES_PER_TOKEN))

    low = budget_sweep(blocks, RATE)
    high = budget_sweep(blocks, RATE_HIGH)

    # The budget the case study would choose: of those that keep both
    # promises at the harder rate, the one that delivers most. That is
    # what a tuning session is actually looking for, and it is not the
    # largest budget -- which is the point of the sweep.
    ok = [r for r in high if r["token_budget"] and r["meets_ttft"]
          and r["meets_itl"]]
    chosen = (max(ok, key=lambda r: r["tokens_per_s"])["token_budget"] if ok
              else min(r["token_budget"] for r in high if r["token_budget"]))

    payload = {
        "interference": interference(blocks),
        "budgets": low,
        "budgets_high": high,
        "chosen_budget": chosen,
        "settling": settling(blocks, chosen, RATE_HIGH),
        "policies": policy_sweep(blocks, chosen, RATE_HIGH),
        "preemption": preemption(blocks, chosen),
        "swap_arithmetic": swap_arithmetic(),
        "attention_share": attention_share(),
        "pure_decode_itl_ms": decode_step(
            round(next(r["mean_batch"] for r in high
                       if r["token_budget"] == chosen)),
            CONTEXT_TOKENS).inter_token_ms,
        "whole_prefill_ms": prefill_step(PROMPT_MEAN).seconds * 1e3,
        "long_prefill_ms": prefill_step(8192).seconds * 1e3,
        "assumptions": {
            "blocks": blocks, "block": BLOCK_SIZE, "pool_bytes": pool_bytes,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "max_batch": MAX_BATCH, "n_requests": N_REQUESTS,
            "settling_keys": SETTLING_KEYS,
            "rate": RATE, "rate_high": RATE_HIGH,
            "context": CONTEXT_TOKENS, "seed": SEED,
            "prompt_mean": PROMPT_MEAN, "output_mean": OUTPUT_MEAN,
            "ttft_budget_ms": TTFT_BUDGET_MS, "itl_budget_ms": ITL_BUDGET_MS,
            "pcie_bytes_per_s": PCIE_BYTES_PER_S,
            "swap_pool_share": SWAP_POOL_SHARE,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }

    path = write("results/ch18.json", payload)
    print(f"wrote {path}")

    iv = payload["interference"]
    print(f"  one reply, {iv['whole']['interlopers']} prompts of "
          f"{iv['whole']['interloper_prompt']:,} tokens landing in it:")
    for name in ("whole", "chunked"):
        r = iv[name]
        print(f"    {name:8}: gap between tokens p50 {r['median_ms']:6.1f} ms, "
              f"p99 {r['p99_ms']:7.1f}, worst {r['max_ms']:7.1f}; "
              f"{r['over_budget']} gaps over the budget; "
              f"reply done at {r['victim_finish_s']:.2f} s")

    for rate, rows in ((RATE, low), (RATE_HIGH, high)):
        print(f"  token budget at {rate} req/s (chosen {chosen:,}):")
        for r in rows:
            met = ("both" if r["meets_ttft"] and r["meets_itl"] else
                   "TTFT only" if r["meets_ttft"] else
                   "ITL only" if r["meets_itl"] else "neither")
            name = (f"{r['token_budget']:6,}" if r["token_budget"]
                    else "  Ch17")
            print(f"    {name}: {r['tokens_per_s']:6,.0f} tok/s, "
                  f"TTFT p50 {r['ttft_p50_ms']:6,.0f} p99 {r['ttft_p99_ms']:7,.0f} ms, "
                  f"ITL p50 {r['itl_p50_ms']:5.1f} p99 {r['itl_p99_ms']:6.1f} ms, "
                  f"mixed {r['mixed_share'] * 100:4.1f}%, meets {met}")

    print(f"  queue order at {RATE_HIGH} req/s:")
    for r in payload["policies"]:
        print(f"    {r['pool']:9} pool, {r['policy']:16}: "
              f"end to end p50 {r['total_p50_s']:6.2f} s "
              f"p99 {r['total_p99_s']:7.2f}, slowdown p50 {r['slowdown_p50']:5.2f} "
              f"p99 {r['slowdown_p99']:7.2f} worst {r['slowdown_max']:8.2f}, "
              f"{r['tokens_per_s']:6,.0f} tok/s")

    print("  a full pool, two ways out "
          f"({payload['preemption'][0]['pool_gb']:.1f} GB):")
    for r in payload["preemption"]:
        print(f"    {r['mode']:22}: {r['tokens_per_s']:6,.0f} tok/s, "
              f"{r['preemptions']:4} preemptions, "
              f"prompts read {r['prompt_reread']:.2f}x over, "
              f"{r['swapped_bytes'] / 1e9:7.1f} GB copied "
              f"({r['swap_s']:6.2f} s), "
              f"TTFT p99 {r['ttft_p99_ms'] / 1e3:7.1f} s")

    print("  one preempted sequence, the two costs:")
    for r in payload["swap_arithmetic"]["rows"]:
        links = "  ".join(f"{n} {v * 1e3:6.1f} ms"
                          for n, v in r["swap_s"].items())
        print(f"    {r['tokens']:5,} tokens: recompute "
              f"{r['recompute_s'] * 1e3:6.1f} ms | {links} | break-even "
              f"{r['breakeven_bytes_per_s'] / 1e9:5.1f} GB/s")


if __name__ == "__main__":
    main()
