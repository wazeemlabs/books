| Latency budget | Largest batch it allows | Throughput | Cost per 1M output tokens | What stops you |
|---|---|---|---|---|
| 15 ms | 174 | 11,608 tok/s | **$0.078** | the budget |
| 25 ms | 325 | 13,626 tok/s | **$0.066** | memory runs out first |
| 50 ms | 325 | 13,626 tok/s | **$0.066** | memory runs out first |
| 100 ms | 325 | 13,626 tok/s | **$0.066** | memory runs out first |

Arithmetic over published specs, not a measurement. The batch is also capped at 325 sequences by the memory accounting in Chapter 13.
