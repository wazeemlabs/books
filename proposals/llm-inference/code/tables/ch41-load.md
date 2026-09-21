| Requests a second | Of capacity | In the system | End to end, mean | p99 | TTFT p99 | Between tokens, p99 | Tokens/s | Moves by |
|---|---|---|---|---|---|---|---|---|
| **2** | 7% | 3 | 1.53 s | 5.24 s | 68 ms | 7.9 ms | 607 | 3% |
| **4** | 13% | 7 | 1.60 s | 5.50 s | 74 ms | 8.1 ms | 1,213 | 3% |
| **8** | 26% | 15 | 1.77 s | 6.10 s | 83 ms | 8.3 ms | 2,428 | 3% |
| **12** | 39% | 24 | 1.94 s | 6.70 s | 95 ms | 8.3 ms | 3,643 | 3% |
| **16** | 52% | 35 | 2.12 s | 7.31 s | 112 ms | 8.4 ms | 4,856 | 3% |
| **20** | 65% | 47 | 2.30 s | 7.90 s | 141 ms | 8.7 ms | 6,069 | 3% |
| **24** | 79% | 62 | 2.51 s | 8.60 s | 256 ms | 9.6 ms | 7,279 | 3% |
| **26** | 85% | 71 | 2.66 s | 9.06 s | 469 ms | 10.0 ms | 7,880 | 3% |
| 28 * | 92% | 84 | 2.92 s | 9.72 s | 1,368 ms | 10.3 ms | 8,480 | 4% |
| 30 + | 98% | 131 | 4.31 s | 11.83 s | 3,441 ms | 10.6 ms | 9,049 | 32% |
| 32 + | 105% | 378 | 12.29 s | 23.21 s | 16,239 ms | 10.6 ms | 9,114 | 91% |
| 36 + | 118% | 1105 | 35.81 s | 56.94 s | 51,868 ms | 10.6 ms | 9,146 | 101% |
| 40 + | 131% | 1705 | 55.21 s | 85.59 s | 81,021 ms | 10.6 ms | 9,164 | 102% |

One machine, seed 0. Every rate was run twice, at 6,000 requests and at 12,000; the columns are the longer run and "moves by" is how far the furthest of mean_time_s, p99_s, tokens_per_s shifted between the two. Bold rows keep both of the case study's promises -- 1,000 ms to a first token and 50 ms between tokens, at the 99th percentile. Rows marked * break at least one. Rows marked + never settled: their latencies grew with the length of the run, so they are numbers about the benchmark and not about the machine. Capacity is 9,164 tokens a second, which at 300 tokens a reply is 30.5 requests a second, and "of capacity" is measured against that.
