| Requests a second | Offered | Static: tokens/s | Static: TTFT p99 | Continuous: tokens/s | Continuous: TTFT p99 | Continuous: between tokens, p99 |
|---|---|---|---|---|---|---|
| 4 | 1,200 | 1,056 \* | 14.8 s | **1,094** | 70 ms | 21 ms |
| 8 | 2,400 | 1,591 \* | 37.2 s | **2,121** | 76 ms | 31 ms |
| 12 | 3,600 | 1,637 \* | 54.0 s | **3,081** | 87 ms | 42 ms |
| 16 | 4,800 | 1,709 \* | 60.2 s | **3,960** \* | 91 ms | 58 ms |
| 20 | 6,000 | 1,799 \* | 61.3 s | **4,734** \* | 95 ms | 81 ms |
| 24 | 7,200 | 1,815 \* | 63.8 s | **5,374** \* | 96 ms | 114 ms |
| 28 | 8,400 | 1,822 \* | 66.9 s | **5,839** \* | 105 ms | 153 ms |
| 32 | 9,600 | 1,756 \* | 73.9 s | **6,077** \* | 1,485 ms | 152 ms |

\* not keeping up: the server took more than 10% longer to drain than the requests took to arrive, or missed the 1,000 ms p99 time-to-first-token budget. Offered load is the arrival rate times the mean output length (300 tokens), which is what the service would have to produce to keep up.
