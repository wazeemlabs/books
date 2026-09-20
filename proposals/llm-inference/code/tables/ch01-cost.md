| Sequences at once | Throughput | Cost per 1M output tokens | Arithmetic used | Limited by |
|---|---|---|---|---|
| 1 | 207 tok/s | **$4.365** | 0.3% of peak | memory |
| 2 | 409 tok/s | **$2.209** | 0.7% of peak | memory |
| 4 | 798 tok/s | **$1.131** | 1.3% of peak | memory |
| 8 | 1,525 tok/s | **$0.592** | 2.5% of peak | memory |
| 16 | 2,800 tok/s | **$0.322** | 4.5% of peak | memory |
| 32 | 4,809 tok/s | **$0.188** | 7.8% of peak | memory |
| 64 | 7,501 tok/s | **$0.120** | 12.1% of peak | memory |
| 128 | 10,416 tok/s | **$0.087** | 16.8% of peak | memory |
| 256 | 12,929 tok/s | **$0.070** | 20.9% of peak | memory |
| 325 | 13,627 tok/s | **$0.066** | 22.0% of peak | memory |

A model, not a benchmark: H100 SXM 80GB, $3.25/GPU-hour, 8B parameters in bf16, 1,500-token sequences. For comparison, hosted Llama-3.1-8B, cheapest tracked provider is published at $0.05 per million output tokens.
