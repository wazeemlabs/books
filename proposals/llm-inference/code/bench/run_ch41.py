"""Chapter 41: how much hardware, and how hot can it run.

Four measurements, all on the scheduler Part III built:

1. Little's law on a server that is nothing like the queue it was
   derived for, and the condition under which it stops holding.
2. The capacity: where throughput stops rising with offered load.
3. Latency against utilization, measured, against what classical
   queueing theory predicts for the same utilization. They disagree by
   a factor that decides how much hardware a service buys.
4. The case study, sized from its promise rather than from a rule of
   thumb.

    python3 -m bench.run_ch41
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tinyserve.paged import BLOCK_SIZE
from tinyserve.reference import (GPU_BYTES, GPU_USD_PER_HOUR, KV_BYTES_PER_TOKEN,
                                 REQUESTS_PER_S, WEIGHT_BYTES)
from tinyserve.scheduler import serve_chunked
from tinyserve.serving import decode_step, prefill_step

from .harness import pct, steady_state, write
from .run_ch18 import (ITL_BUDGET_MS, MAX_BATCH, OUTPUT_MEAN, PROMPT_MEAN,
                       TTFT_BUDGET_MS, make_requests)

SEED = 0
# Long enough that the queue settles at the loads that matter. What
# "long enough" means is not guessed: every rate is run again at twice
# this and the answers are compared, and a rate whose numbers move is
# reported as unsettled rather than quoted. See `harness.steady_state`.
N_REQUESTS = 6000
# The statistics the settling test is applied to. Means and the median
# converge; a 99th percentile of a few thousand samples is too noisy to
# gate on, so it is reported from the longer run and not tested.
SETTLING_KEYS = ["mean_time_s", "p99_s", "tokens_per_s"]
RATES = [2, 4, 8, 12, 16, 20, 24, 26, 28, 30, 32, 36, 40]
# The window the steady state is measured in, as fractions of the
# arrival span: past the ramp, before the drain.
WINDOW = (0.30, 0.90)
SAMPLES = 400            # points at which the population is counted


def token_budget() -> int:
    """Chapter 18's chosen budget, read from the file that chose it.

    Design decision record I settled this for the case study. Copying
    the number here would let the two drift; reading it means a
    different decision there changes the plan here.
    """
    import json
    path = Path("results/ch18.json")
    if not path.exists():
        raise SystemExit("run `make ch18` first: this chapter plans capacity "
                         "for the scheduler that chapter configured")
    return int(json.loads(path.read_text())["chosen_budget"])


def blocks() -> int:
    return (GPU_BYTES - WEIGHT_BYTES) // (BLOCK_SIZE * KV_BYTES_PER_TOKEN)


def alone_seconds() -> float:
    """How long one request takes with the machine to itself.

    The denominator of every slowdown in this chapter. A request that
    waits for nothing and shares with no one still takes this long,
    and no amount of hardware makes it shorter.
    """
    return (prefill_step(PROMPT_MEAN).seconds
            + OUTPUT_MEAN * decode_step(1, PROMPT_MEAN + OUTPUT_MEAN).seconds)


def measure(rate: float, n: int = N_REQUESTS) -> dict:
    """One offered load, driven through the scheduler."""
    reqs = make_requests(n, rate, np.random.default_rng(SEED))
    trace = serve_chunked(reqs, max_batch=MAX_BATCH, blocks=blocks(),
                          token_budget=token_budget())
    done = [r for r in trace.requests if r.finish_s is not None]
    span = max(r.arrival_s for r in trace.requests)
    lo, hi = span * WINDOW[0], span * WINDOW[1]

    # Little's law needs the number in the *system*: arrived, not yet
    # finished, whether it is decoding, prefilling or waiting. Counted
    # by sampling the window rather than by asking the scheduler, so
    # the two sides of L = lambda W are measured independently.
    grid = np.linspace(lo, hi, SAMPLES)
    population = float(np.mean([sum(1 for r in done
                                    if r.arrival_s <= t < r.finish_s)
                                for t in grid]))
    inside = [r for r in done if lo <= r.arrival_s < hi]
    times = [r.finish_s - r.arrival_s for r in inside]
    ttft = [(r.first_token_s - r.arrival_s) * 1e3 for r in inside
            if r.first_token_s is not None]
    gaps = trace.gaps_ms
    return {
        "rate": rate,
        "n_requests": n,
        "requests_in_window": len(inside),
        "in_system": population,
        "mean_time_s": float(np.mean(times)),
        "lambda_times_w": rate * float(np.mean(times)),
        "p50_s": pct(times, 50), "p99_s": pct(times, 99),
        "ttft_p99_ms": pct(ttft, 99),
        "itl_p99_ms": pct(gaps, 99),
        "tokens_per_s": trace.tokens_between(lo, hi) / (hi - lo),
        "drained_over_span": trace.makespan_s / span,
        "mean_batch": trace.mean_batch,
        "preemptions": trace.preemptions,
        "peak_blocks": trace.peak_blocks,
    }


def sweep() -> list[dict]:
    """Every offered load, each measured twice at different lengths.

    The longer run is the one reported. The shorter one is there to
    answer a question a single run cannot: has this settled? Past the
    server's capacity nothing settles -- the backlog grows for as long
    as the benchmark runs -- and a latency quoted from there is a
    number about the benchmark, not about the machine.
    """
    rows = []
    for rate in RATES:
        check = steady_state(lambda n, r=rate: measure(r, n),
                             N_REQUESTS, SETTLING_KEYS)
        row = dict(check["long"])
        row["settled"] = check["settled"]
        row["drift"] = check["drift"]
        row["worst_drift"] = check["worst_drift"]
        row["short_run"] = {k: check["short"][k] for k in SETTLING_KEYS}
        rows.append(row)
    return rows


def capacity(rows: list[dict]) -> dict:
    """Where throughput stops rising, in tokens and in requests.

    Read off the sweep rather than assumed: the highest throughput any
    offered load achieved, and the load at which the server first came
    within a whisker of it.
    """
    peak = max(r["tokens_per_s"] for r in rows)
    at_peak = next(r for r in rows if r["tokens_per_s"] >= peak * 0.995)
    return {
        "tokens_per_s": peak,
        "requests_per_s": peak / OUTPUT_MEAN,
        "first_rate_at_peak": at_peak["rate"],
        "output_mean": OUTPUT_MEAN,
    }


def against_theory(rows: list[dict], cap: dict) -> dict:
    """Measured slowdown against what a classical queue predicts.

    The textbook result for a single-server queue is that time in the
    system grows as 1 / (1 - utilization): at 90% busy, ten times the
    service time. That is the arithmetic behind every "never run a
    server above 70%" rule of thumb.

    A server that batches does not behave that way, and the gap
    between the two columns is the subject of the chapter.
    """
    alone = alone_seconds()
    out = []
    for r in rows:
        rho = r["rate"] / cap["requests_per_s"]
        classical = 1 / (1 - rho) if rho < 1 else None
        out.append({
            "rate": r["rate"], "utilization": rho,
            "measured_slowdown": r["mean_time_s"] / alone,
            "classical_slowdown": classical,
            "over_prediction": (classical / (r["mean_time_s"] / alone)
                                if classical else None),
            "p99_s": r["p99_s"],
            "little_gap": r["lambda_times_w"] / r["in_system"] - 1,
            "steady": r["settled"],
            "worst_drift": r["worst_drift"],
        })
    steady = [x for x in out if x["steady"]]
    return {
        "rows": out,
        "alone_seconds": alone,
        "largest_little_gap_while_steady":
            max(abs(x["little_gap"]) for x in steady),
        "largest_little_gap_overall":
            max(abs(x["little_gap"]) for x in out),
        "first_unsteady_rate": next((x["rate"] for x in out
                                     if not x["steady"]), None),
    }


def sizing(rows: list[dict], cap: dict) -> dict:
    """How many machines the case study needs, from its promise.

    Three answers, because the point of the chapter is that they
    differ: what the arithmetic of throughput alone says, what the
    classical rule of thumb says, and what the promise actually
    requires once latency is measured rather than modelled.
    """
    import math

    demand = REQUESTS_PER_S
    per_machine_at_capacity = cap["requests_per_s"]
    # Which offered loads still keep both promises.
    # A load whose numbers did not settle is not a load the machine can
    # be said to sustain, whatever its latency looked like on the run.
    ok = [r for r in rows
          if r["settled"]
          and r["ttft_p99_ms"] <= TTFT_BUDGET_MS
          and r["itl_p99_ms"] <= ITL_BUDGET_MS]
    highest_ok = max(r["rate"] for r in ok) if ok else 0
    classical_rule = per_machine_at_capacity * 0.70    # the 70% rule
    answers = {
        "throughput_only": math.ceil(demand / per_machine_at_capacity),
        "classical_rule_of_thumb": math.ceil(demand / classical_rule),
        "measured_promise": math.ceil(demand / highest_ok) if highest_ok else None,
    }
    return {
        "demand_requests_per_s": demand,
        "per_machine_at_capacity": per_machine_at_capacity,
        "highest_rate_meeting_both_promises": highest_ok,
        "utilization_there": highest_ok / per_machine_at_capacity,
        "machines": answers,
        "usd_per_hour": {k: (v * GPU_USD_PER_HOUR if v else None)
                         for k, v in answers.items()},
        "ttft_budget_ms": TTFT_BUDGET_MS,
        "itl_budget_ms": ITL_BUDGET_MS,
        "gpu_usd_per_hour": GPU_USD_PER_HOUR,
    }


def main() -> None:
    rows = sweep()
    cap = capacity(rows)
    theory = against_theory(rows, cap)
    plan = sizing(rows, cap)
    payload = {
        "sweep": rows, "capacity": cap, "theory": theory, "sizing": plan,
        "assumptions": {
            "seed": SEED, "n_requests": N_REQUESTS,
            "n_requests_long": 2 * N_REQUESTS,
            "settling_keys": SETTLING_KEYS, "rates": RATES,
            "window": list(WINDOW), "samples": SAMPLES,
            "prompt_mean": PROMPT_MEAN, "output_mean": OUTPUT_MEAN,
            "max_batch": MAX_BATCH, "token_budget": token_budget(),
            "blocks": blocks(),
            "ttft_budget_ms": TTFT_BUDGET_MS, "itl_budget_ms": ITL_BUDGET_MS,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ch41.json", payload)
    print(f"wrote {path}")
    print(f"  one request alone: {theory['alone_seconds']:.3f} s")
    print(f"  capacity: {cap['tokens_per_s']:,.0f} tokens a second, "
          f"{cap['requests_per_s']:.1f} requests a second")
    print(f"  Little's law holds to "
          f"{theory['largest_little_gap_while_steady'] * 100:.1f}% while the "
          f"server is in steady state, "
          f"{theory['largest_little_gap_overall'] * 100:.0f}% once it is not "
          f"(from {theory['first_unsteady_rate']} requests a second)")
    print(f"  {'rate':>6}{'util':>7}{'measured':>10}{'classical':>11}"
          f"{'over by':>9}{'p99 s':>8}")
    for x in theory["rows"]:
        c = ("--" if x["classical_slowdown"] is None
             else f"{x['classical_slowdown']:.2f}")
        o = ("--" if x["over_prediction"] is None
             else f"{x['over_prediction']:.1f}x")
        print(f"  {x['rate']:>6}{x['utilization']:>7.2f}"
              f"{x['measured_slowdown']:>10.2f}{c:>11}{o:>9}{x['p99_s']:>8.2f}")
    s = plan
    print(f"  the case study wants {s['demand_requests_per_s']} requests a "
          f"second, and both promises hold up to "
          f"{s['highest_rate_meeting_both_promises']} a machine "
          f"({s['utilization_there'] * 100:.0f}% of capacity):")
    for name, n in s["machines"].items():
        cost = s["usd_per_hour"][name]
        print(f"    {name:26} {n} machines"
              f"{'' if cost is None else f'   ${cost:,.2f}/hour'}")


if __name__ == "__main__":
    main()
