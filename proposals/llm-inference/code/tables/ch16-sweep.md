| Sequences | Step time | Of which shared | Of which per-sequence | Tokens/s | Tokens/s, cache-resident model |
|---|---|---|---|---|---|
| 1 | 25.6 ms | 24.4 ms | 1.1 ms | **39** | 2,143 |
| 2 | 73.8 ms | 72.0 ms | 1.7 ms | **27** | 3,193 |
| 4 | 82.6 ms | 79.6 ms | 2.7 ms | **48** | 3,900 |
| 8 | 99.6 ms | 95.0 ms | 4.4 ms | **80** | 3,490 |
| 16 | 138.1 ms | 129.4 ms | 8.3 ms | **116** | 4,091 |
| 32 | 231.9 ms | 212.9 ms | 19.1 ms | **138** | 3,712 |
| 64 | 329.9 ms | 297.5 ms | 32.3 ms | **194** | 4,577 |

The whole decode step for a 753 MiB model whose weights come from memory at 12.5 GB/s, beside the 3.8 MiB model whose weights are already in cache. The step time is what one user waits between tokens.
