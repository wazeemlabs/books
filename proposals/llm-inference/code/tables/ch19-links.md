| Link | Speed | Tokens/s | Wait for the second token, p50 | p99 | Every later gap, p99 |
|---|---|---|---|---|---|
| NVLink, same node | 900 GB/s | 56,091 | **7.7 ms** | 8.5 ms | 8.2 ms |
| InfiniBand NDR | 50 GB/s | 56,066 | **10.0 ms** | 17.6 ms | 8.2 ms |
| PCIe 5.0 x16 | 64 GB/s | 56,078 | **9.4 ms** | 15.5 ms | 8.4 ms |
| 100 GbE | 12 GB/s | 56,071 | **17.8 ms** | 47.6 ms | 8.2 ms |
| 25 GbE | 3 GB/s | 56,042 | **49.7 ms** | 169.5 ms | 8.3 ms |
| _colocated: no link_ | -- | 57,861 | **6.5 ms** | 7.9 ms | 8.4 ms |

The same 4P + 8D fleet over each link. The prefill machine produces the first token before the cache goes anywhere, so the whole cost of the move lands in one place: the wait for the *second* token. Every gap after that is an ordinary decode step, which is why the last column barely moves.
