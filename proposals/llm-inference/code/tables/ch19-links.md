| Link | Speed | Tokens/s | Wait for the second token, p50 | p99 | Every later gap, p99 |
|---|---|---|---|---|---|
| NVLink, same node | 900 GB/s | 59,482 | **9.0 ms** | 10.4 ms | 10.2 ms |
| InfiniBand NDR | 50 GB/s | 59,491 | **11.5 ms** | 18.5 ms | 10.0 ms |
| PCIe 5.0 x16 | 64 GB/s | 59,484 | **11.0 ms** | 16.4 ms | 10.0 ms |
| 100 GbE | 12 GB/s | 59,473 | **19.6 ms** | 47.0 ms | 10.2 ms |
| 25 GbE | 3 GB/s | 59,478 | **52.2 ms** | 162.9 ms | 10.0 ms |
| _colocated: no link_ | -- | 59,853 | **6.7 ms** | 7.9 ms | 8.4 ms |

The same 5P + 7D fleet over each link. The prefill machine produces the first token before the cache goes anywhere, so the whole cost of the move lands in one place: the wait for the *second* token. Every gap after that is an ordinary decode step, which is why the last column barely moves.
