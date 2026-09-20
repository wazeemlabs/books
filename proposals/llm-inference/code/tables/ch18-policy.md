| Block pool | Queue order | End to end p50 | End to end p99 | Slowdown p50 | Slowdown p99 | Worst slowdown | Tokens/s |
|---|---|---|---|---|---|---|---|
| 64 GB (full) | `fcfs` | 1.88 s | 7.99 s | **1.6x** | 2x | 2x | 5,586 |
| 64 GB (full) | `shortest-output` | 1.88 s | 8.00 s | **1.6x** | 2x | 2x | 5,585 |
| 64 GB (full) | `longest-output` | 1.89 s | 7.98 s | **1.6x** | 2x | 3x | 5,587 |
| 1.9 GB (squeezed) | `fcfs` | 26.87 s | 38.62 s | **18.7x** | 111x | 253x | 2,686 |
| 1.9 GB (squeezed) | `shortest-output` | 2.47 s | 54.65 s | **2.5x** | 18x | 19x | 2,822 |
| 1.9 GB (squeezed) | `longest-output` | 40.80 s | 63.67 s | **35.0x** | 236x | 411x | 2,683 |

The same 600 requests at 24 a second through three queue orders, twice: once with the whole block pool and once with it squeezed to 3% of it. Slowdown is how much longer a request took than it would have taken alone on an idle server -- the fairness number, and the one that moves. `shortest-output` and `longest-output` sort by the true reply length, which a real server does not know; they are the best and worst a perfect oracle could do.
