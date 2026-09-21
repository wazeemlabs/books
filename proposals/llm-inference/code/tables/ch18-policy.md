| Block pool | Queue order | End to end p50 | End to end p99 | Slowdown p50 | Slowdown p99 | Worst slowdown | Tokens/s |
|---|---|---|---|---|---|---|---|
| 64 GB (full) | `fcfs` | 1.94 s | 8.28 s | **1.7x** | 2x | 3x | 6,764 |
| 64 GB (full) | `shortest-output` | 1.93 s | 8.33 s | **1.7x** | 2x | 2x | 6,764 |
| 64 GB (full) | `longest-output` | 1.94 s | 8.27 s | **1.7x** | 2x | 9x | 6,764 |
| 1.9 GB (squeezed) | `fcfs` | 48.80 s | 59.18 s | **38.4x** | 200x | 461x | 5,746 |
| 1.9 GB (squeezed) | `shortest-output` | 2.13 s | 189.23 s | **1.9x** | 53x | 61x | 5,836 |
| 1.9 GB (squeezed) | `longest-output` | 83.62 s | 245.02 s | **80.2x** | 831x | 1750x | 5,249 |

The same 4800 requests at 24 a second through three queue orders, twice: once with the whole block pool and once with it squeezed to 3% of it. Slowdown is how much longer a request took than it would have taken alone on an idle server -- the fairness number, and the one that moves. `shortest-output` and `longest-output` sort by the true reply length, which a real server does not know; they are the best and worst a perfect oracle could do.
