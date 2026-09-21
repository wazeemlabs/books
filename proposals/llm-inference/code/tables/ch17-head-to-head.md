| | Static batching | Continuous batching | Ratio |
|---|---|---|---|
| Output tokens a second | 1,788 | 3,433 | **1.9x** |
| Time to first token, p50 | 193,363 ms | 22 ms | **8,659x** |
| Time to first token, p99 | 376,764 ms | 85 ms | **4,411x** |
| Between tokens, p50 | 18.3 ms | 6.5 ms | **2.8x** |
| Between tokens, p99 | 20.7 ms | 52.0 ms | **2.5x worse** |
| End to end, p50 | 197.4 s | 2.0 s | **99x** |
| Slots the batch held, mean | 225 | 28.0 | **--** |
| Sequences actually advancing, mean | 65.0 | 28.0 | **--** |
| Slots held that held live work | 29% | 100% | **--** |
| Peak of the block pool | 86% | 20% | **4x** |
| Time to drain, over the arrival window | 1.9x | 1.01x | **--** |

The same 4800 requests, the same arrivals (12 a second, Poisson), the same prompt and output lengths, the same cost model for a prefill and a decode step. Only the scheduler differs. Static forms a batch when 256 requests have arrived or one second has passed, whichever comes first, and is given all the memory it asks for. Every ratio is the better number over the worse one, except where it says otherwise.
