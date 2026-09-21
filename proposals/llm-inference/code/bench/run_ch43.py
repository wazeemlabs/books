"""Chapter 43: what a dashboard can tell you, and what it cannot.

Four measurements, all on the scheduler Part III built and the metric
shapes a production engine actually exports:

1. The percentile on the dashboard against the percentile in the
   trace. They are not the same number, and the gap is the width of a
   bucket.
2. Which metric moves first when something goes wrong, and how long
   before the promise breaks.
3. What a scrape interval does to an incident shorter than itself.
4. What a label costs, counted in time series.

    python3 -m bench.run_ch43
"""

from __future__ import annotations

import numpy as np

from tinyserve.observe import (ITL_BUCKETS, TTFT_BUCKETS, bucket_containing,
                               bucketize, histogram_quantile, seen, series)
from tinyserve.paged import BLOCK_SIZE
from tinyserve.reference import KV_BYTES_PER_TOKEN
from tinyserve.scheduler import serve_chunked

from .harness import pct, write
from .run_ch18 import (ITL_BUDGET_MS, MAX_BATCH, TTFT_BUDGET_MS, make_requests)
from .run_ch41 import blocks, token_budget

SEED = 0
N_REQUESTS = 6_000
RATES = [12, 20, 24, 26, 28]
OPERATING_RATE = 26
QUANTILES = [0.5, 0.9, 0.99]
SCRAPES = [1, 5, 15, 30, 60]
INCIDENTS = [2, 5, 15, 30, 60, 120]
SAMPLES = 600
# What a fault does: the pool is cut to this share of itself, which is
# what a leak, a co-tenant or a bad rollout looks like from inside.
FAULT_POOL = 0.06
LABELS = {"model": 3, "tenant": 40, "endpoint": 4, "instance": 8}


def run(rate: float, pool: float = 1.0, n: int = N_REQUESTS):
    reqs = make_requests(n, rate, np.random.default_rng(SEED))
    return serve_chunked(reqs, max_batch=MAX_BATCH,
                         blocks=max(64, int(blocks() * pool)),
                         token_budget=token_budget())


def samples(trace) -> dict[str, list[float]]:
    ttft = [(r.first_token_s - r.arrival_s) for r in trace.requests
            if r.first_token_s is not None]
    return {"ttft_s": ttft, "itl_s": [g / 1e3 for g in trace.gaps_ms]}


def dashboards() -> dict:
    """The number in the trace against the number on the screen."""
    rows = []
    for rate in RATES:
        s = samples(run(rate))
        for name, values, buckets, budget in (
                ("time to first token", s["ttft_s"], TTFT_BUCKETS,
                 TTFT_BUDGET_MS / 1e3),
                ("between tokens", s["itl_s"], ITL_BUCKETS,
                 ITL_BUDGET_MS / 1e3)):
            counts = bucketize(values, buckets)
            for q in QUANTILES:
                true = float(np.quantile(values, q))
                shown = histogram_quantile(buckets, counts, q)
                lo, hi = bucket_containing(buckets, true)
                rows.append({
                    "rate": rate, "metric": name, "quantile": q,
                    "true_s": true, "shown_s": shown,
                    "error": (shown - true) / true if true else float("nan"),
                    "bucket_low": lo, "bucket_high": hi,
                    "bucket_width_over_value": (hi - lo) / true if true else None,
                    "budget_s": budget,
                    "true_breaks": true > budget,
                    "shown_breaks": shown > budget,
                    "disagree": (true > budget) != (shown > budget),
                })
    return {"rows": rows, "quantiles": QUANTILES,
            "ttft_buckets": list(TTFT_BUCKETS),
            "itl_buckets": list(ITL_BUCKETS),
            "ttft_budget_s": TTFT_BUDGET_MS / 1e3,
            "itl_budget_s": ITL_BUDGET_MS / 1e3,
            "budget_bucket_ttft": bucket_containing(TTFT_BUCKETS,
                                                    TTFT_BUDGET_MS / 1e3),
            "budget_bucket_itl": bucket_containing(ITL_BUCKETS,
                                                   ITL_BUDGET_MS / 1e3)}


def _timeline(trace, metric: str) -> list[tuple[float, float]]:
    """A gauge over time, as a scrape would find it.

    Reconstructed from the trace rather than instrumented, because what
    matters here is which signal moves and when, not how it is
    exported.
    """
    done = [r for r in trace.requests if r.finish_s is not None]
    span = max(r.arrival_s for r in trace.requests)
    grid = np.linspace(0, span, SAMPLES)
    out = []
    for t in grid:
        if metric == "num_requests_waiting":
            v = sum(1 for r in trace.requests
                    if r.arrival_s <= t and (r.first_token_s is None
                                             or r.first_token_s > t))
        elif metric == "num_requests_running":
            v = sum(1 for r in done if r.first_token_s is not None
                    and r.first_token_s <= t < r.finish_s)
        elif metric == "time_to_first_token_seconds":
            recent = [r.first_token_s - r.arrival_s for r in done
                      if r.first_token_s is not None
                      and t - 10 <= r.first_token_s <= t]
            v = pct(recent, 99) if recent else 0.0
        else:
            raise KeyError(metric)
        out.append((float(t), float(v)))
    return out


def detection() -> dict:
    """Which signal moves first when the pool is cut.

    The fault is memory: the block pool is cut to a fraction of itself
    partway through, which is what a leak, a noisy neighbour or a bad
    rollout looks like from inside the server. What is measured is how
    far each signal moves, not how clever a threshold is.
    """
    healthy, broken = run(OPERATING_RATE), run(OPERATING_RATE, FAULT_POOL)
    rows = []
    for metric in ("num_requests_waiting", "num_requests_running",
                   "time_to_first_token_seconds"):
        before = [v for _, v in _timeline(healthy, metric)]
        after = [v for _, v in _timeline(broken, metric)]
        base = float(np.median(before)) or 1e-9
        rows.append({
            "metric": metric,
            "healthy_median": float(np.median(before)),
            "faulty_median": float(np.median(after)),
            "healthy_p99": pct(before, 99),
            "faulty_p99": pct(after, 99),
            "moved_by": float(np.median(after)) / base,
        })
    ht, bt = samples(healthy), samples(broken)
    return {
        "rows": rows, "pool_share": FAULT_POOL,
        "rate": OPERATING_RATE,
        "preemptions_healthy": healthy.preemptions,
        "preemptions_faulty": broken.preemptions,
        "ttft_p99_healthy": pct(ht["ttft_s"], 99) * 1e3,
        "ttft_p99_faulty": pct(bt["ttft_s"], 99) * 1e3,
        "itl_p99_healthy": pct(ht["itl_s"], 99) * 1e3,
        "itl_p99_faulty": pct(bt["itl_s"], 99) * 1e3,
        "tokens_healthy": healthy.tokens_per_s,
        "tokens_faulty": broken.tokens_per_s,
        "pool_blocks_healthy": blocks(),
        "pool_blocks_faulty": max(64, int(blocks() * FAULT_POOL)),
        "pool_gb_faulty": max(64, int(blocks() * FAULT_POOL))
                          * BLOCK_SIZE * KV_BYTES_PER_TOKEN / 1e9,
    }


def scrapes() -> dict:
    """What a scrape interval does to an incident shorter than itself."""
    rows = []
    for length in INCIDENTS:
        signal = [(t / 10.0, 10.0 if 300.0 <= t / 10.0 < 300.0 + length
                   else 1.0) for t in range(6_000)]
        for interval in SCRAPES:
            rows.append({"incident_s": length, "interval_s": interval,
                         "caught": seen(signal, interval, threshold=5.0)})
    return {"rows": rows, "intervals": SCRAPES, "incidents": INCIDENTS}


def cardinality() -> dict:
    """What a label costs, counted in series."""
    rows = []
    running = dict(LABELS)
    used: dict[str, int] = {}
    for name, n in LABELS.items():
        used[name] = n
        rows.append({
            "labels": list(used),
            "added": name, "values": n,
            "gauge_series": series(used),
            "histogram_series": series(used,
                                       histogram_buckets=len(ITL_BUCKETS)),
        })
    return {"rows": rows, "label_values": LABELS,
            "itl_buckets": len(ITL_BUCKETS),
            "one_histogram_alone": series({}, len(ITL_BUCKETS))}


def main() -> None:
    dash, det, scr, card = dashboards(), detection(), scrapes(), cardinality()
    payload = {
        "dashboards": dash, "detection": det, "scrapes": scr,
        "cardinality": card,
        "assumptions": {
            "seed": SEED, "n_requests": N_REQUESTS, "rates": RATES,
            "operating_rate": OPERATING_RATE, "quantiles": QUANTILES,
            "scrape_intervals": SCRAPES, "incidents": INCIDENTS,
            "fault_pool_share": FAULT_POOL, "labels": LABELS,
            "ttft_budget_ms": TTFT_BUDGET_MS, "itl_budget_ms": ITL_BUDGET_MS,
            "token_budget": token_budget(), "blocks": blocks(),
            "max_batch": MAX_BATCH,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ch43.json", payload)
    print(f"wrote {path}")
    print("  the trace against the dashboard:")
    for r in dash["rows"]:
        if r["quantile"] == 0.99:
            flag = "   <- they disagree about the promise" if r["disagree"] else ""
            print(f"    {r['rate']:>3} req/s  {r['metric']:<20} p99 "
                  f"{r['true_s'] * 1e3:8.1f} ms in the trace, "
                  f"{r['shown_s'] * 1e3:8.1f} ms on the screen "
                  f"({r['error'] * 100:+6.1f}%){flag}")
    print(f"  a pool cut to {det['pool_share']:.0%} "
          f"({det['pool_gb_faulty']:.1f} GB) at {det['rate']} req/s:")
    for r in det["rows"]:
        print(f"    {r['metric']:<32}{r['healthy_median']:9.2f} -> "
              f"{r['faulty_median']:9.2f}  ({r['moved_by']:.1f}x)")
    print(f"    preemptions {det['preemptions_healthy']} -> "
          f"{det['preemptions_faulty']}; first-token p99 "
          f"{det['ttft_p99_healthy']:,.0f} -> {det['ttft_p99_faulty']:,.0f} ms")
    print("  what a scrape catches:")
    for r in scr["rows"]:
        if r["interval_s"] in (15, 60):
            print(f"    a {r['incident_s']:>3} s incident, scraped every "
                  f"{r['interval_s']:>2} s: seen "
                  f"{r['caught'] * 100:5.1f}% of the time")
    print("  what a label costs:")
    for r in card["rows"]:
        print(f"    + {r['added']:<10}({r['values']:>2} values): "
              f"{r['gauge_series']:>8,} gauge series, "
              f"{r['histogram_series']:>9,} for one histogram")


if __name__ == "__main__":
    main()
