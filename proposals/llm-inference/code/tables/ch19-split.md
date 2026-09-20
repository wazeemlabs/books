| Machines | Tokens/s | TTFT p50 | TTFT p99 | Between tokens, p50 | Between tokens, p99 | Sequences per decode machine |
|---|---|---|---|---|---|---|
| 1P + 11D | 15,281 | 13.6 s | 28.0 s | 5.2 ms | 5.4 ms | 6.6 |
| 2P + 10D | 29,955 | 4.3 s | 9.0 s | 5.7 ms | 6.0 ms | 14.2 |
| 3P + 9D | 42,833 | 1.2 s | 2.7 s | 6.5 ms | 7.2 ms | 22.8 |
| **4P + 8D** | 50,969 | 45 ms | 267 ms | 7.6 ms | 8.2 ms | 31.3 |
| 5P + 7D | 50,446 | 22 ms | 89 ms | 8.2 ms | 9.2 ms | 36.8 |
| 6P + 6D | 49,158 | 18 ms | 72 ms | 9.0 ms | 10.7 ms | 44.9 |
| 7P + 5D | 47,083 | 17 ms | 69 ms | 10.4 ms | 12.7 ms | 56.5 |
| 8P + 4D | 43,819 | 16 ms | 67 ms | 12.4 ms | 15.9 ms | 75.7 |
| 9P + 3D | 38,101 | 23 ms | 1.5 s | 16.3 ms | 19.9 ms | 106.1 |
| 10P + 2D | 27,521 | 2.4 s | 8.1 s | 18.4 ms | 20.1 ms | 141.3 |
| 11P + 1D | 14,425 | 13.0 s | 29.9 s | 19.1 ms | 20.5 ms | 192.3 |
| _12 colocated_ | **52,664** | 23 ms | 77 ms | 6.6 ms | 8.4 ms | 26.9 |

2,000 requests at 200 a second (seed 0) through 12 accelerators, divided every way, over InfiniBand NDR. The last row is the same 12 accelerators each doing both phases with Chapter 18's scheduler at a 512-token budget. Throughput is measured over the arrival window with the first 10% discarded, so a fleet's drain tail is not counted as slow serving.
