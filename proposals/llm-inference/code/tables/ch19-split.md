| Machines | Tokens/s | TTFT p50 | TTFT p99 | Between tokens, p50 | Between tokens, p99 | Sequences per decode machine |
|---|---|---|---|---|---|---|
| 1P + 11D | 15,558 | 56.6 s | 112.7 s | 5.2 ms | 5.4 ms | 7.1 |
| 2P + 10D | 30,547 | 18.1 s | 36.7 s | 5.7 ms | 6.1 ms | 16.5 |
| 3P + 9D | 44,882 | 5.3 s | 11.4 s | 6.6 ms | 7.1 ms | 30.1 |
| 4P + 8D | 58,644 | 75 ms | 553 ms | 8.0 ms | 8.8 ms | 48.3 |
| **5P + 7D** | 59,491 | 22 ms | 83 ms | 8.8 ms | 10.0 ms | 60.0 |
| 6P + 6D | 59,163 | 18 ms | 66 ms | 10.1 ms | 12.1 ms | 77.5 |
| 7P + 5D | 58,383 | 17 ms | 63 ms | 12.7 ms | 15.4 ms | 108.2 |
| 8P + 4D | 54,077 | 39 ms | 2.4 s | 18.5 ms | 19.9 ms | 159.8 |
| 9P + 3D | 41,057 | 6.0 s | 15.6 s | 19.0 ms | 20.1 ms | 189.1 |
| 10P + 2D | 27,301 | 20.1 s | 44.1 s | 19.1 ms | 20.3 ms | 213.0 |
| 11P + 1D | 13,510 | 64.8 s | 132.6 s | 19.3 ms | 20.8 ms | 242.4 |
| _12 colocated_ | **59,853** | 24 ms | 73 ms | 6.8 ms | 8.4 ms | 29.6 |

8,000 requests at 200 a second (seed 0) through 12 accelerators, divided every way, over InfiniBand NDR. The last row is the same 12 accelerators each doing both phases with Chapter 18's scheduler at a 512-token budget. Throughput is measured over the arrival window with the first 50% discarded, so a fleet's drain tail is not counted as slow serving.
