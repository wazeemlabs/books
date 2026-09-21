| Requests a second | Offered | Static: tokens/s | Static: TTFT p99 | Continuous: tokens/s | Continuous: TTFT p99 | Continuous: between tokens, p99 |
|---|---|---|---|---|---|---|
| 4 | 1,200 | 1,145 \* | 16.9 s | **1,156** | 70 ms | 23 ms |
| 8 | 2,400 | 1,739 \* | 202.5 s | **2,301** | 79 ms | 36 ms |
| 12 | 3,600 | 1,788 \* | 376.8 s | **3,433** | 85 ms | 52 ms |
| 16 | 4,800 | 1,793 \* | 475.5 s | **4,552** | 94 ms | 74 ms |
| 20 | 6,000 | 1,805 \* | 530.3 s | **5,646** | 107 ms | 114 ms |
| 24 | 7,200 | 1,766 \* | 588.5 s | **6,665** | 402 ms | 186 ms |
| 28 | 8,400 | 1,780 \* | 600.3 s | **6,924** \* | 20,764 ms | 118 ms |
| 32 | 9,600 | 1,796 \* | 621.8 s | **6,964** \* | 41,164 ms | 117 ms |

\* not keeping up: the server took more than 10% longer to drain than the requests took to arrive, or missed the 1,000 ms p99 time-to-first-token budget. Offered load is the arrival rate times the mean output length (300 tokens), which is what the service would have to produce to keep up.
