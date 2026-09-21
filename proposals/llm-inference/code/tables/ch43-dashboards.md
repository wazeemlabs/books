| Requests a second | Metric | p99 in the trace | p99 on the dashboard | Off by | The bucket it landed in |
|---|---|---|---|---|---|
| 12 | time to first token | 90.9 ms | 93.8 ms | +3.2% | 80-100 ms |
| 12 | between tokens | 8.3 ms | 9.9 ms | +19.7% | 0-10 ms |
| 20 | time to first token | 137.2 ms | 218.4 ms | +59.1% | 100-250 ms |
| 20 | between tokens | 8.7 ms | 9.9 ms | +14.1% | 0-10 ms |
| 24 | time to first token | 252.0 ms | 276.1 ms | +9.6% | 250-500 ms |
| 24 | between tokens | 9.5 ms | 9.9 ms | +3.9% | 0-10 ms |
| 26 | time to first token | 436.3 ms | 468.2 ms | +7.3% | 250-500 ms |
| 26 | between tokens | 9.9 ms | 10.0 ms | +0.6% | 0-10 ms |
| 28 | time to first token | 781.5 ms | 827.6 ms | +5.9% | 750-1,000 ms |
| 28 | between tokens | 10.2 ms | 20.8 ms | +104.5% | 10-25 ms |

The scheduler of Chapter 18 at five offered loads, 6,000 requests each, seed 0. The "trace" column is the percentile of the samples themselves; the "dashboard" column is what Prometheus' `histogram_quantile` returns from the counts, using vLLM's own default bucket boundaries (FACTS.md). The error is the width of whichever bucket the percentile landed in: the between-tokens promise of 50 ms sits in a bucket running 25 to 50 ms, and the first-token promise of 1 s in one running 0.75 to 1.00 s.
