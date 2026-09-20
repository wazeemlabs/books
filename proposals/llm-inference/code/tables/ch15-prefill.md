| Prompt already cached | Tokens still computed | Prefill, no hit | Prefill, hit | Speedup | If time went with tokens |
|---|---|---|---|---|---|
| 0 of 512 (0%) | 512 | 105.4 ms | 104.9 ms | **1.00x** \* | 1.00x |
| 128 of 512 (25%) | 384 | 94.7 ms | 75.1 ms | **1.31x** \* | 1.33x |
| 256 of 512 (50%) | 256 | 105.0 ms | 52.6 ms | **1.96x** \* | 2.00x |
| 384 of 512 (75%) | 128 | 104.3 ms | 28.2 ms | **3.70x** \* | 4.00x |
| 448 of 512 (88%) | 64 | 103.4 ms | 14.7 ms | **7.03x** \* | 8.00x |
| 496 of 512 (97%) | 16 | 101.1 ms | 5.2 ms | **19.55x** \* | 32.00x |

Median of 15 paired runs after 2 warmup, cold and warm measured one after the other in each run so that a machine-wide stall moves both. The lookup in the index is timed with the hit; the miss lookup on the cold path is not charged, which makes the comparison slightly unkind to the cache.

\* spread of the paired ratio exceeded 5%; this machine is a shared container.
