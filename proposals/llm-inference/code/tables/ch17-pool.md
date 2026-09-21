| Block pool | Sequences in flight | Tokens/s | TTFT p99 | Preemptions | Tokens generated twice |
|---|---|---|---|---|---|
| 64.0 GB (100%) | 28.0 | **3,433** | 0.1 s | 0 | 0.0% |
| 16.0 GB (25%) | 28.0 | **3,433** | 0.1 s | 0 | 0.0% |
| 6.4 GB (10%) | 28.1 | **3,426** | 2.7 s | 95 | 0.2% |
| 3.2 GB (5%) | 16.3 | **2,365** | 183.5 s | 487 | 1.1% |
| 1.9 GB (3%) | 9.7 | **1,577** | 479.9 s | 761 | 2.2% |
| 1.3 GB (2%) | 6.3 | **1,086** | 881.4 s | 973 | 3.5% |

Continuous batching at 12 requests a second through smaller and smaller pools. The full pool is 64 GB, which is what is left on one accelerator after the weights. Nothing else changes.
