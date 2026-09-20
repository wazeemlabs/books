| Cache size | Least recently used **leaf** | Least frequently used leaf | Least recently used block, leaf or not |
|---|---|---|---|
| 1.1 GB (512 blocks) | 55.2% | 52.5% | 51.0% |
| 1.7 GB (799 blocks) | 66.9% | 56.9% | 63.6% |
| 3.4 GB (1,599 blocks) | 81.4% | 62.6% | 80.7% |
| 7.0 GB (3,332 blocks) | 85.0% | 73.5% | 85.0% |
| 14.0 GB (6,664 blocks) | 85.3% | 85.4% | 85.3% |
| 21.0 GB (9,996 blocks) | 85.4% | 85.4% | 85.4% |
| 28.0 GB (13,328 blocks) | 85.4% | 85.4% | 85.4% |

Share of prompt tokens served from the cache, over 500 requests from 200 sessions (seed 0). The largest size holds everything this traffic can share. Sizes are the reference model's bytes; the simulation runs the real index and the real allocator.
