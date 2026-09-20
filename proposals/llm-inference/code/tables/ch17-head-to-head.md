| | Static batching | Continuous batching | Ratio |
|---|---|---|---|
| Output tokens a second | 1,637 | 3,081 | **1.9x** |
| Time to first token, p50 | 31,472 ms | 20 ms | **1,568x** |
| Time to first token, p99 | 54,040 ms | 87 ms | **619x** |
| Between tokens, p50 | 12.6 ms | 6.2 ms | **2.0x** |
| Between tokens, p99 | 18.7 ms | 41.5 ms | **2.2x worse** |
| End to end, p50 | 36.6 s | 1.9 s | **19x** |
| Slots the batch held, mean | 127 | 22.7 | **--** |
| Sequences actually advancing, mean | 37.4 | 22.7 | **--** |
| Slots held that held live work | 29% | 100% | **--** |
| Peak of the block pool | 74% | 13% | **6x** |
| Time to drain, over the arrival window | 2.1x | 1.10x | **--** |

The same 600 requests, the same arrivals (12 a second, Poisson), the same prompt and output lengths, the same cost model for a prefill and a decode step. Only the scheduler differs. Static forms a batch when 256 requests have arrived or one second has passed, whichever comes first, and is given all the memory it asks for. Every ratio is the better number over the worse one, except where it says otherwise.
