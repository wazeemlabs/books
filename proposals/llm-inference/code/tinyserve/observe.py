"""What a dashboard can and cannot tell you.

Every chapter of this book has measured something directly. A running
service cannot: it exports counters and histograms, something scrapes
them every so often, and a query turns what survives into a number on
a screen. Each of those steps loses information, and the losses are
not random -- they are largest exactly where a promise is written.

Three of them are modelled here:

* `histogram_quantile` -- how a percentile is computed from buckets
  rather than from samples, which is how every Prometheus-shaped stack
  does it. The answer is interpolated inside whichever bucket the
  percentile lands in, so its error is the width of that bucket.
* `sampled` -- what a scrape interval does to an event shorter than
  itself.
* `series` -- how many time series a set of labels produces, which is
  what a metrics bill is made of.

The bucket boundaries are vLLM's own defaults, quoted in FACTS.md.
"""

from __future__ import annotations

import math
from typing import Sequence

# vllm/v1/metrics/buckets.py, TIME_TO_FIRST_TOKEN_BUCKETS and
# INTER_TOKEN_LATENCY_BUCKETS (FACTS.md). Seconds.
TTFT_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0,
                640.0, 2560.0)
ITL_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
               1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0)


def bucketize(values: Sequence[float],
              buckets: Sequence[float]) -> list[int]:
    """Cumulative counts, as a Prometheus histogram stores them.

    One count per boundary -- everything at or below it -- and a last
    one for everything. What the samples themselves were is gone.
    """
    counts = [0] * (len(buckets) + 1)
    for v in values:
        for i, edge in enumerate(buckets):
            if v <= edge:
                counts[i] += 1
        counts[-1] += 1
    return counts


def histogram_quantile(buckets: Sequence[float], counts: Sequence[int],
                       q: float) -> float:
    """The percentile a dashboard shows, computed the way it computes it.

    Prometheus finds the bucket the rank falls in and interpolates
    linearly between that bucket's lower and upper bound, assuming the
    samples inside are spread evenly. They are not, and where they are
    not, the answer is wrong by up to the width of the bucket.

    A rank in the last, unbounded bucket has no upper bound to
    interpolate to, so the largest boundary is returned -- which is why
    a latency past the top of the scale reads as exactly the top of the
    scale, however far past it the truth is.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"{q} is not a quantile")
    total = counts[-1]
    if total == 0:
        return float("nan")
    rank = q * total
    for i, edge in enumerate(buckets):
        if counts[i] >= rank:
            lower = buckets[i - 1] if i else 0.0
            below = counts[i - 1] if i else 0
            inside = counts[i] - below
            if inside <= 0:
                return lower
            return lower + (edge - lower) * (rank - below) / inside
    return buckets[-1]


def bucket_containing(buckets: Sequence[float], value: float
                      ) -> tuple[float, float]:
    """The bucket a value falls in, as (lower, upper).

    The width of this is the resolution a dashboard has at that value,
    and it is worth looking up for whatever number your promise is
    written in.
    """
    lower = 0.0
    for edge in buckets:
        if value <= edge:
            return lower, edge
        lower = edge
    return lower, float("inf")


def sampled(series: Sequence[tuple[float, float]], interval: float,
            offset: float = 0.0) -> list[tuple[float, float]]:
    """What a scrape every `interval` seconds sees of a signal.

    A gauge is read, not accumulated: whatever it happened to be at the
    moment of the scrape is the whole of what is recorded. An event
    shorter than the interval may be seen once, or not at all,
    depending on where it falls.
    """
    if interval <= 0:
        raise ValueError("a scrape interval is positive")
    out, when = [], offset
    end = series[-1][0] if series else 0.0
    i = 0
    while when <= end:
        while i + 1 < len(series) and series[i + 1][0] <= when:
            i += 1
        out.append((when, series[i][1]))
        when += interval
    return out


def seen(series: Sequence[tuple[float, float]], interval: float,
         threshold: float, offsets: int = 64) -> float:
    """Share of scrape phases in which a signal is caught above `threshold`.

    Averaged over where the scrape happens to land, because that is not
    something anyone controls. A short spike caught at one offset and
    missed at another is a spike the dashboard will show sometimes.
    """
    hits = 0
    for k in range(offsets):
        phase = interval * k / offsets
        if any(v > threshold for _, v in sampled(series, interval, phase)):
            hits += 1
    return hits / offsets


def series(labels: dict[str, int], histogram_buckets: int = 0) -> int:
    """Time series produced by one metric with these label cardinalities.

    A histogram is not one series: it is one per bucket, plus a sum and
    a count. Multiply that by every combination of label values and the
    number gets away from people quickly, which is why a metrics bill
    is usually a cardinality problem rather than a volume one.
    """
    combinations = 1
    for n in labels.values():
        combinations *= n
    per_metric = (histogram_buckets + 3) if histogram_buckets else 1
    return combinations * per_metric


# --- tests ---------------------------------------------------------------


def test_a_histogram_percentile_lands_inside_its_bucket() -> None:
    """Whatever else it is, the answer is in the right bucket."""
    values = [0.012] * 900 + [0.4] * 100
    counts = bucketize(values, ITL_BUCKETS)
    got = histogram_quantile(ITL_BUCKETS, counts, 0.5)
    lower, upper = bucket_containing(ITL_BUCKETS, 0.012)
    assert lower <= got <= upper, (got, lower, upper)


def test_the_error_is_the_width_of_the_bucket() -> None:
    """Samples piled at one end of a wide bucket are reported from the
    middle of it, because the interpolation assumes they are spread."""
    values = [0.011] * 1000                 # just inside (0.01, 0.025]
    counts = bucketize(values, ITL_BUCKETS)
    got = histogram_quantile(ITL_BUCKETS, counts, 0.99)
    assert got > 0.02, got                  # reported near the top
    assert abs(got - 0.011) / 0.011 > 1.0   # more than double the truth


def test_past_the_top_of_the_scale_reads_as_the_top() -> None:
    values = [5000.0] * 100
    counts = bucketize(values, TTFT_BUCKETS)
    assert histogram_quantile(TTFT_BUCKETS, counts, 0.99) == TTFT_BUCKETS[-1]


def test_a_scrape_can_miss_an_event_shorter_than_itself() -> None:
    """The reason a spike is real and the dashboard is flat."""
    # Two seconds of trouble in a minute of calm.
    signal = [(t / 10, 10.0 if 30.0 <= t / 10 < 32.0 else 1.0)
              for t in range(600)]
    assert seen(signal, interval=1.0, threshold=5.0) == 1.0
    caught = seen(signal, interval=60.0, threshold=5.0)
    assert 0.0 < caught < 0.1, caught
    # And what is caught is roughly how much of the minute it lasted.
    assert abs(caught - 2.0 / 60.0) < 0.02, caught


def test_a_histogram_is_not_one_series() -> None:
    plain = series({"model": 3, "tenant": 50})
    hist = series({"model": 3, "tenant": 50}, histogram_buckets=len(ITL_BUCKETS))
    assert plain == 150
    assert hist == 150 * (len(ITL_BUCKETS) + 3)
    assert series({}, 0) == 1
