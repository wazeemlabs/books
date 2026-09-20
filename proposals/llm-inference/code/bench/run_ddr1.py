"""Design decision record I: the case study's scheduler and cache policy.

Part III measured seven things. This decides, from those measurements
and nothing else, how the case study's server is built.

Almost nothing new runs here. Every decision is read out of the results
files Chapters 13 to 19 already wrote, so a decision cannot drift from
the evidence that produced it: change a chapter's measurement, re-run
`make`, and this record changes with it or fails to build.

The one thing that is measured here is the fleet's size, because no
chapter asked that question. Chapter 19 compared two designs on a fixed
twelve machines; nobody asked how few would do. The sweep below uses
Chapter 19's own colocated fleet at the case study's rate and varies
only the machine count.

    python3 -m bench.run_ddr1
"""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import numpy as np

from tinyserve.paged import BLOCK_SIZE
from tinyserve.reference import (CONTEXT_TOKENS, GPU_BYTES, GPU_USD_PER_HOUR,
                                 KV_BYTES_PER_TOKEN, OUTPUT_TOKENS,
                                 PROMPT_TOKENS, REQUESTS_PER_S, WEIGHT_BYTES)
from tinyserve.scheduler import serve_colocated

from .harness import write
from .run_ch19 import (MAX_BATCH, SEED, TOKEN_BUDGET, WARM_FRACTION,
                       make_requests, summarize)

RESULTS = Path("results")
NEEDS = ["ch13", "ch14", "ch15", "ch16", "ch17", "ch18", "ch19"]

# The sizing sweep. Chapter 19's trace holds the request count fixed and
# varies the rate, which shrinks the arrival window as the load rises:
# at a high enough rate the measurement window is shorter than one
# reply and measures the fleet warming up rather than the fleet
# working. Here the *window* is held fixed instead -- at least this many
# seconds of arrivals, and at least this many requests -- so every point
# is measured the same way.
SIZING_SPAN_S = 10.0
SIZING_MIN_REQUESTS = 2000
SIZING_FLEETS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16]
# What "keeps up" means, from Chapter 19: delivering this share of the
# tokens the traffic asks for, inside the first-token promise.
KEEPING_UP_SHARE = 0.95


def read() -> dict[str, dict]:
    missing = [c for c in NEEDS if not (RESULTS / f"{c}.json").exists()]
    if missing:
        raise SystemExit(f"run `make {' '.join(missing)}` first: "
                         f"this record has almost no measurements of its own")
    return {c: json.loads((RESULTS / f"{c}.json").read_text()) for c in NEEDS}


def decisions(r: dict[str, dict]) -> list[dict]:
    """Each decision, its alternatives, and the measurement that decided it."""
    ch13, ch14, ch15 = r["ch13"], r["ch14"], r["ch15"]
    ch17, ch18, ch19 = r["ch17"], r["ch18"], r["ch19"]

    # Chapter 13: what reserving the whole context costs.
    reserve = next(p for p in ch13["policies"] if p["policy"] == "max_model_len")

    # Chapter 14: the block size sweep.
    blocks = sorted(ch14["block_sizes"], key=lambda b: b["block_size"])
    chosen_block = ch14["at_default"]
    largest = blocks[-1]

    # Chapter 15: prefix caching, at the smallest cache that reaches the
    # ceiling, against the same size run with the other two policies.
    lru = sorted((x for x in ch15["sizes"] if x["policy"] == "lru"),
                 key=lambda x: x["pool_blocks"])
    ceiling = ch15["unlimited"]["hit_rate"]
    enough = next(x for x in lru if x["hit_rate"] >= ceiling - 0.01)
    at_size = {x["policy"]: x for x in ch15["sizes"]
               if x["pool_blocks"] == enough["pool_blocks"]}
    smallest = {x["policy"]: x for x in ch15["sizes"]
                if x["pool_blocks"] == lru[0]["pool_blocks"]}
    enough_gb = enough["pool_blocks"] * BLOCK_SIZE * KV_BYTES_PER_TOKEN / 1e9

    # Chapter 17: continuous against static, at the rate it was measured.
    h17 = ch17["head_to_head"]

    # Chapter 18: the token budget, the queue order, the eviction mode.
    high = {x["token_budget"]: x for x in ch18["budgets_high"]}
    budget = ch18["chosen_budget"]
    keeps_both = {b: x for b, x in high.items()
                  if x["meets_itl"] and x["meets_ttft"]}
    pol = {(x["pool"], x["policy"]): x for x in ch18["policies"]}
    pre = {x["mode"]: x for x in ch18["preemption"]}
    swap_row = next(x for x in ch18["swap_arithmetic"]["rows"]
                    if x["tokens"] == CONTEXT_TOKENS)

    # Chapter 19: the fleet shape.
    co19, best19 = ch19["colocated"], ch19["best_split"]

    return [
        {
            "decision": f"Page the KV cache in blocks of {BLOCK_SIZE} tokens",
            "instead_of": "reserving each sequence's whole context up front, "
                          "or paging in a larger block",
            "chapter": "ch14",
            "because": (f"reserving the context admits "
                        f"{reserve['concurrent_seqs']} sequences where paging "
                        f"admits {chosen_block['admitted_paged']}, and at "
                        f"this block size paging wastes "
                        f"{chosen_block['wasted_tokens_per_sequence']:.1f} "
                        f"tokens of the sequence's own memory"),
            "evidence": {
                "sequences admitted, paged": chosen_block["admitted_paged"],
                "sequences admitted, whole context reserved":
                    chosen_block["admitted_contiguous"],
                "pool used, this block size": chosen_block["utilization"],
                "wasted tokens per sequence, this block size":
                    chosen_block["wasted_tokens_per_sequence"],
                "pool used, largest block tried": largest["utilization"],
                "wasted tokens per sequence, largest block tried":
                    largest["wasted_tokens_per_sequence"],
                "largest block tried": largest["block_size"],
            },
            "would_change": ("a kernel that needs longer contiguous runs than "
                             "a block, or replies short enough that a block's "
                             "worth of waste stops being small beside them"),
        },
        {
            "decision": "Prefix caching on, with a prefix tree and "
                        "least-recently-used eviction of leaves",
            "instead_of": "no prefix cache, an unstructured block cache, "
                          "or evicting the least frequently used block",
            "chapter": "ch15",
            "because": (f"{enough['hit_rate'] * 100:.0f}% of prompt tokens are "
                        f"already in the cache at {enough_gb:.0f} GB, against "
                        f"an {ceiling * 100:.1f}% ceiling, and the lookup costs "
                        f"{ch15['unlimited']['index_us_per_request']:.0f} "
                        f"microseconds a request"),
            "evidence": {
                "hit rate, tree with LRU": enough["hit_rate"],
                "hit rate, least frequently used, same size":
                    at_size["lfu"]["hit_rate"],
                "hit rate, unstructured, same size":
                    at_size["unstructured"]["hit_rate"],
                "hit rate, unstructured, smallest cache":
                    smallest["unstructured"]["hit_rate"],
                "hit rate, tree with LRU, smallest cache":
                    smallest["lru"]["hit_rate"],
                "ceiling, unlimited cache": ceiling,
                "cache GB": enough_gb,
                "lookup microseconds per request":
                    ch15["unlimited"]["index_us_per_request"],
            },
            "would_change": ("traffic with no shared prefixes, where the index "
                             "costs something and returns nothing. Nothing "
                             "measured here makes the feature worth turning off "
                             "when prefixes are shared at all"),
        },
        {
            "decision": "Continuous batching: decide the batch every iteration",
            "instead_of": "a static batch formed on a timeout",
            "chapter": "ch17",
            "because": (f"{h17['continuous']['tokens_per_s'] / h17['static']['tokens_per_s']:.1f}x "
                        f"the throughput, and a median first token "
                        f"{h17['static']['ttft_p50_ms'] / h17['continuous']['ttft_p50_ms']:,.0f}x "
                        f"sooner, at the same "
                        f"{ch17['assumptions']['head_to_head_rate']} requests "
                        f"a second on the same memory"),
            "evidence": {
                "tokens/s, static": h17["static"]["tokens_per_s"],
                "tokens/s, continuous": h17["continuous"]["tokens_per_s"],
                "TTFT p50 ms, static": h17["static"]["ttft_p50_ms"],
                "TTFT p50 ms, continuous": h17["continuous"]["ttft_p50_ms"],
                "slots holding live work, static":
                    h17["static"]["slot_utilization"],
                "slots holding live work, continuous":
                    h17["continuous"]["slot_utilization"],
            },
            "would_change": ("nothing in this book. Static batching lost on "
                             "every measure at every rate tried"),
        },
        {
            "decision": f"Chunked prefill, with a {budget:,}-token "
                        f"per-iteration budget",
            "instead_of": "giving a prefill an iteration of its own, or "
                          "choosing a larger budget",
            "chapter": "ch18",
            "because": (f"of the budgets that keep both promises at "
                        f"{ch18['assumptions']['rate_high']} requests a second, "
                        f"this one has the highest throughput "
                        f"({high[budget]['tokens_per_s']:,.0f} tokens a second) "
                        f"and the lowest wait between tokens "
                        f"({high[budget]['itl_p99_ms']:.1f} ms at the 99th "
                        f"percentile)"),
            "evidence": {
                "budgets that keep both promises": sorted(keeps_both),
                "tokens/s, prefill on its own iteration":
                    high[0]["tokens_per_s"],
                "tokens/s, this budget": high[budget]["tokens_per_s"],
                "between-token p99 ms, prefill on its own iteration":
                    high[0]["itl_p99_ms"],
                "between-token p99 ms, this budget": high[budget]["itl_p99_ms"],
                "between-token p99 ms, largest budget tried":
                    high[max(high)]["itl_p99_ms"],
                "TTFT p99 ms, this budget": high[budget]["ttft_p99_ms"],
                "TTFT p99 ms, next budget down":
                    high[budget // 2]["ttft_p99_ms"],
                "between-token promise, ms": ch18["assumptions"]["itl_budget_ms"],
                "first-token promise, ms": ch18["assumptions"]["ttft_budget_ms"],
            },
            "would_change": ("a looser between-token promise, which would buy "
                             "a larger budget and a better first token; or a "
                             "tighter one, which the smaller budgets serve at "
                             "a first token this service could not sell"),
        },
        {
            "decision": "First come, first served, with no priority tiers",
            "instead_of": "shortest job first, or longest job first",
            "chapter": "ch18",
            "because": (f"with the memory this service has, the queue order is "
                        f"worth "
                        f"{abs(pol[('full', 'shortest-output')]['total_p50_s'] / pol[('full', 'fcfs')]['total_p50_s'] - 1) * 100:.1f}% "
                        f"of the median end-to-end time, and the simplest "
                        f"order is the one that needs no estimate of how long "
                        f"a reply will be"),
            "evidence": {
                "end-to-end p50 s, FCFS, full pool":
                    pol[("full", "fcfs")]["total_p50_s"],
                "end-to-end p50 s, shortest first, full pool":
                    pol[("full", "shortest-output")]["total_p50_s"],
                "end-to-end p50 s, FCFS, squeezed pool":
                    pol[("squeezed", "fcfs")]["total_p50_s"],
                "end-to-end p50 s, shortest first, squeezed pool":
                    pol[("squeezed", "shortest-output")]["total_p50_s"],
                "worst slowdown, FCFS, squeezed pool":
                    pol[("squeezed", "fcfs")]["slowdown_max"],
                "worst slowdown, shortest first, squeezed pool":
                    pol[("squeezed", "shortest-output")]["slowdown_max"],
                "squeezed pool, GB": pol[("squeezed", "fcfs")]["pool_gb"],
                "full pool, GB": pol[("full", "fcfs")]["pool_gb"],
            },
            "would_change": (
                f"running the pool near full. Squeezed, shortest first is "
                f"{pol[('squeezed', 'fcfs')]['total_p50_s'] / pol[('squeezed', 'shortest-output')]['total_p50_s']:.1f}x "
                f"better on the median and "
                f"{pol[('squeezed', 'fcfs')]['slowdown_max'] / pol[('squeezed', 'shortest-output')]['slowdown_max']:.1f}x "
                f"better on the worst slowdown, and this decision flips"),
        },
        {
            "decision": "Preempt by recomputing, not by swapping the cache "
                        "out to host memory",
            "instead_of": "copying an evicted cache out over PCIe and back",
            "chapter": "ch18",
            "because": (f"{pre['recompute']['tokens_per_s'] / pre['swap over PCIe 5.0 x16']['tokens_per_s']:.2f}x "
                        f"the throughput, although recomputing loses every "
                        f"individual comparison: one "
                        f"{CONTEXT_TOKENS:,}-token sequence costs "
                        f"{swap_row['recompute_s'] * 1e3:.1f} ms to recompute "
                        f"against {swap_row['swap_s']['PCIe 5.0 x16'] * 1e3:.1f} ms "
                        f"to copy"),
            "evidence": {
                "tokens/s, recompute": pre["recompute"]["tokens_per_s"],
                "tokens/s, swap over PCIe 5.0":
                    pre["swap over PCIe 5.0 x16"]["tokens_per_s"],
                "tokens/s, swap over NVLink":
                    pre["swap over NVLink (H100)"]["tokens_per_s"],
                "sequences in flight, recompute": pre["recompute"]["mean_batch"],
                "sequences in flight, swap":
                    pre["swap over PCIe 5.0 x16"]["mean_batch"],
                "TTFT p99 ms, recompute": pre["recompute"]["ttft_p99_ms"],
                "TTFT p99 ms, swap over PCIe 5.0":
                    pre["swap over PCIe 5.0 x16"]["ttft_p99_ms"],
                "one sequence, recompute ms": swap_row["recompute_s"] * 1e3,
                "one sequence, PCIe 5.0 ms":
                    swap_row["swap_s"]["PCIe 5.0 x16"] * 1e3,
                "link fast enough to break even, GB/s":
                    swap_row["breakeven_bytes_per_s"] / 1e9,
            },
            "would_change": ("a swap-in that restores a cache gradually rather "
                             "than all at once. What lost here was the shape of "
                             "the re-entry, not the cost of the copy: even over "
                             "NVLink, which is fast enough on the arithmetic, "
                             "swapping still lost"),
        },
        {
            "decision": "One fleet where every machine does both phases",
            "instead_of": f"splitting the same fleet into "
                          f"{best19['prefill_workers']} prefill machines and "
                          f"{best19['decode_workers']} decode machines",
            "chapter": "ch19",
            "because": (f"{co19['tokens_per_s'] / best19['tokens_per_s']:.2f}x "
                        f"the throughput on identical hardware, a first token "
                        f"{best19['ttft_p99_ms'] / co19['ttft_p99_ms']:.1f}x "
                        f"faster at the 99th percentile, and a second token "
                        f"that does not wait for a cache to cross a network"),
            "evidence": {
                "tokens/s, colocated": co19["tokens_per_s"],
                "tokens/s, best split": best19["tokens_per_s"],
                "tokens/s the traffic asks for":
                    co19["offered_sampled_tokens_per_s"],
                "keeps up, colocated": co19["keeping_up"],
                "keeps up, best split": best19["keeping_up"],
                "TTFT p99 ms, colocated": co19["ttft_p99_ms"],
                "TTFT p99 ms, best split": best19["ttft_p99_ms"],
                "second token p50 ms, colocated": co19["second_token_p50_ms"],
                "second token p50 ms, best split": best19["second_token_p50_ms"],
                "link": best19["link"],
            },
            "would_change": ("hardware that differs by phase, or a promise "
                             "tight enough that one fleet has to over-provision "
                             "to keep it. Neither is true here, and this is the "
                             "decision in this record most likely to be wrong "
                             "for a service that is not this one"),
        },
    ]


def memory_policies(r: dict[str, dict]) -> dict:
    """Which of the memory decisions were exercised, and at what pool size.

    Four of the seven decisions are policies about memory. A policy
    about memory does nothing while memory is plentiful, and the
    record should say which ones this service actually put under
    pressure rather than implying it pressured all four.
    """
    ch15, ch17, ch18 = r["ch15"], r["ch17"], r["ch18"]
    lru = sorted((x for x in ch15["sizes"] if x["policy"] == "lru"),
                 key=lambda x: x["pool_blocks"])
    sweep = [x for x in ch17["pool_sweep"] if x["policy"] == "continuous"]
    preempting = [x for x in sweep if x["preemptions"] > 0]
    full = [x for x in ch18["policies"] if x["pool"] == "full"]
    squeezed = [x for x in ch18["policies"] if x["pool"] == "squeezed"]
    return {
        # prefix caching: measured across the whole range of cache sizes
        "cache_smallest_frac": lru[0]["pool_frac"],
        "cache_smallest_hit": lru[0]["hit_rate"],
        "cache_largest_frac": lru[-1]["pool_frac"],
        "cache_largest_hit": lru[-1]["hit_rate"],
        # queue order: measured at two pool sizes
        "order_pools_gb": sorted({x["pool_gb"] for x in ch18["policies"]}),
        "order_preemptions_full": max(x["preemptions"] for x in full),
        "order_preemptions_squeezed": max(x["preemptions"] for x in squeezed),
        # preemption mode: unexercised until the pool is small
        "preemptions_at_full_pool": max(x["preemptions"]
                                        for x in ch18["budgets_high"]),
        "largest_pool_share_that_preempts": (max(x["pool_share"]
                                                 for x in preempting)
                                             if preempting else None),
        "preemptions_there": (max(x["preemptions"] for x in preempting
                                  if x["pool_share"]
                                  == max(y["pool_share"] for y in preempting))
                              if preempting else 0),
        # block size: one pool size only
        "block_size_pools": 1,
    }


def sizing(blocks: int) -> list[dict]:
    """How few machines serve the case study's traffic.

    Chapter 19 fixed the fleet at twelve and asked which design used it
    better. Nobody asked how many machines the traffic needs, and that
    is the number the bill is written against.
    """
    rate = float(REQUESTS_PER_S)
    n = max(SIZING_MIN_REQUESTS, int(rate * SIZING_SPAN_S))
    rows = []
    for workers in SIZING_FLEETS:
        reqs = make_requests(n, rate, np.random.default_rng(SEED))
        trace = serve_colocated(reqs, workers=workers, blocks=blocks,
                                token_budget=TOKEN_BUDGET, max_batch=MAX_BATCH)
        row = summarize(trace, rate, workers=workers, design="colocated")
        row["share_of_offered"] = (row["tokens_per_s"]
                                   / row["offered_sampled_tokens_per_s"])
        row["usd_per_hour"] = GPU_USD_PER_HOUR * workers
        row["usd_per_m_output_tokens"] = (GPU_USD_PER_HOUR * workers / 3600
                                          / row["tokens_per_s"] * 1e6)
        rows.append(row)
    return rows


def fleet(r: dict[str, dict], rows: list[dict]) -> dict:
    """The fleet the case study buys, and what it costs to run."""
    ch16, ch19 = r["ch16"], r["ch19"]
    keeps = [x for x in rows if x["keeping_up"]]
    if not keeps:
        raise SystemExit("no fleet size tried keeps up: widen SIZING_FLEETS")
    smallest = min(keeps, key=lambda x: x["workers"])
    measured = next(x for x in rows
                    if x["workers"] == ch19["assumptions"]["fleet"])
    # Chapter 16's arithmetic answer, with no scheduler and no prefill:
    # the largest batch it priced, against the tokens the traffic needs.
    arithmetic = max(ch16["case_study"]["fleet"], key=lambda f: f["batch"])
    hours = 24 * 365 / 12                       # an average month
    return {
        "chosen": smallest["workers"],
        "chosen_tokens_per_s": smallest["tokens_per_s"],
        "chosen_share_of_offered": smallest["share_of_offered"],
        "chosen_ttft_p99_ms": smallest["ttft_p99_ms"],
        "chosen_itl_p99_ms": smallest["itl_p99_ms"],
        "one_below": smallest["workers"] - 1,
        "one_below_share": next(
            (x["share_of_offered"] for x in rows
             if x["workers"] == smallest["workers"] - 1), None),
        "measured_in_ch19": measured["workers"],
        "measured_share_of_offered": measured["share_of_offered"],
        "measured_ttft_p99_ms": measured["ttft_p99_ms"],
        "tokens_per_s_needed": ch16["case_study"]["tokens_per_s_needed"],
        "offered_tokens_per_s": smallest["offered_sampled_tokens_per_s"],
        "arithmetic_batch": arithmetic["batch"],
        "arithmetic_tokens_per_s": arithmetic["tokens_per_s"],
        "arithmetic_accelerators": arithmetic["accelerators"],
        "arithmetic_rounded": ceil(arithmetic["accelerators"]),
        "per_machine_tokens_per_s": smallest["tokens_per_s"] / smallest["workers"],
        "per_machine_share_of_arithmetic": (smallest["tokens_per_s"]
                                            / smallest["workers"]
                                            / arithmetic["tokens_per_s"]),
        "per_machine_requests_per_s": REQUESTS_PER_S / smallest["workers"],
        "usd_per_hour": GPU_USD_PER_HOUR * smallest["workers"],
        "usd_per_month": GPU_USD_PER_HOUR * smallest["workers"] * hours,
        "usd_per_m_output_tokens": smallest["usd_per_m_output_tokens"],
        "usd_per_m_at_measured": measured["usd_per_m_output_tokens"],
        "gpu_usd_per_hour": GPU_USD_PER_HOUR,
        "keeping_up_share": KEEPING_UP_SHARE,
    }


def main() -> None:
    r = read()
    pool_bytes = GPU_BYTES - WEIGHT_BYTES
    blocks = pool_bytes // (BLOCK_SIZE * KV_BYTES_PER_TOKEN)
    rows = sizing(blocks)
    payload = {
        "decisions": decisions(r),
        "memory_policies": memory_policies(r),
        "sizing": rows,
        "fleet": fleet(r, rows),
        "assumptions": {
            "block": BLOCK_SIZE,
            "blocks_per_worker": blocks,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "pool_bytes": pool_bytes,
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "requests_per_s": REQUESTS_PER_S,
            "token_budget": r["ch18"]["chosen_budget"],
            "max_batch": MAX_BATCH,
            "ttft_budget_ms": r["ch18"]["assumptions"]["ttft_budget_ms"],
            "itl_budget_ms": r["ch18"]["assumptions"]["itl_budget_ms"],
            "sizing_requests": max(SIZING_MIN_REQUESTS,
                                   int(REQUESTS_PER_S * SIZING_SPAN_S)),
            "sizing_span_s": SIZING_SPAN_S,
            "warm_fraction": WARM_FRACTION,
            "keeping_up_share": KEEPING_UP_SHARE,
            "gpu_usd_per_hour": GPU_USD_PER_HOUR,
            "from_chapters": NEEDS,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ddr1.json", payload)
    print(f"wrote {path} from {', '.join(NEEDS)}")
    for dec in payload["decisions"]:
        print(f"  [{dec['chapter']}] {dec['decision']}")
        print(f"          not: {dec['instead_of']}")
        print(f"      because: {dec['because']}")
    print("  how few machines serve the case study's traffic:")
    for x in rows:
        print(f"    {x['workers']:3d} machines: {x['tokens_per_s']:9,.0f} tok/s "
              f"({x['share_of_offered'] * 100:5.1f}% of offered), "
              f"TTFT p99 {x['ttft_p99_ms']:7.1f} ms, "
              f"ITL p99 {x['itl_p99_ms']:5.2f} ms, "
              f"${x['usd_per_m_output_tokens']:.3f}/M tokens"
              f"{'' if x['keeping_up'] else '   <- behind'}")
    f = payload["fleet"]
    print(f"  fleet: {f['chosen']} machines "
          f"({f['arithmetic_rounded']} by Chapter 16's arithmetic at batch "
          f"{f['arithmetic_batch']}), {f['chosen_tokens_per_s']:,.0f} of "
          f"{f['offered_tokens_per_s']:,.0f} tokens a second, "
          f"${f['usd_per_hour']:,.2f}/hour, "
          f"${f['usd_per_m_output_tokens']:.3f} per million output tokens")


if __name__ == "__main__":
    main()
