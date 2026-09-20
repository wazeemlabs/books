| Block pool | Sequences in flight | Tokens/s | TTFT p99 | Preemptions | Tokens generated twice |
|---|---|---|---|---|---|
| 64.0 GB (100%) | 22.7 | **3,081** | 0.1 s | 0 | 0.0% |
| 16.0 GB (25%) | 22.7 | **3,081** | 0.1 s | 0 | 0.0% |
| 6.4 GB (10%) | 22.7 | **3,080** | 0.8 s | 3 | 0.0% |
| 3.2 GB (5%) | 15.9 | **2,339** | 18.8 s | 82 | 1.5% |
| 1.9 GB (3%) | 9.7 | **1,604** | 53.9 s | 98 | 2.1% |
| 1.3 GB (2%) | 6.4 | **1,114** | 102.4 s | 116 | 3.0% |

Continuous batching at 12 requests a second through smaller and smaller pools. The full pool is 64 GB, which is what is left on one accelerator after the weights. Nothing else changes.
