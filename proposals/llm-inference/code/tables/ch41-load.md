| Requests a second | Of capacity | In the system | End to end, mean | p99 | TTFT p99 | Between tokens, p99 | Tokens/s |
|---|---|---|---|---|---|---|---|
| **2** | 6% | 3 | 1.54 s | 5.19 s | 60 ms | 7.9 ms | 615 |
| **4** | 13% | 7 | 1.61 s | 5.46 s | 63 ms | 8.0 ms | 1,230 |
| **8** | 26% | 15 | 1.78 s | 6.06 s | 68 ms | 8.3 ms | 2,464 |
| **12** | 39% | 24 | 1.96 s | 6.67 s | 76 ms | 8.3 ms | 3,698 |
| **16** | 51% | 35 | 2.14 s | 7.27 s | 88 ms | 8.4 ms | 4,939 |
| **20** | 64% | 48 | 2.31 s | 7.86 s | 105 ms | 8.5 ms | 6,163 |
| **24** | 77% | 62 | 2.50 s | 8.50 s | 134 ms | 9.1 ms | 7,374 |
| **26** | 84% | 70 | 2.64 s | 8.94 s | 177 ms | 9.4 ms | 7,949 |
| **28** | 90% | 80 | 2.82 s | 9.49 s | 281 ms | 9.8 ms | 8,527 |
| 30 * | 97% | 93 | 3.08 s | 10.11 s | 532 ms | 10.2 ms | 9,057 |
| 32 * | 103% | 120 | 3.89 s | 11.05 s | 1,882 ms | 10.6 ms | 9,260 |
| 36 * | 116% | 178 | 5.72 s | 13.58 s | 5,218 ms | 10.6 ms | 9,323 |
| 40 * | 129% | 240 | 7.70 s | 16.38 s | 8,420 ms | 10.6 ms | 9,309 |

One machine, 1,500 requests at each rate, seed 0. Bold rows keep both of the case study's promises -- 1,000 ms to a first token and 50 ms between tokens, at the 99th percentile -- and finish soon after the traffic stops. Rows marked * break at least one. Capacity is 9,323 tokens a second, which at 300 tokens a reply is 31.1 requests a second, and "of capacity" is measured against that.
