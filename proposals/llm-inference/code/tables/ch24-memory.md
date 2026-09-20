| Scheme | Weights | Time to read them | Smaller than bfloat16 | Left on an 80 GB card for cache |
|---|---|---|---|---|
| _bfloat16_ | 16 GB | 4.78 ms | 1.00x | 64 GB |
| int8, symmetric, per tensor | 8.0 GB | 2.39 ms | 2.00x | 72 GB |
| int8, asymmetric, per tensor | 8.0 GB | 2.39 ms | 2.00x | 72 GB |
| int8, symmetric, per channel | 8.0 GB | 2.39 ms | 2.00x | 72 GB |
| int4, symmetric, per tensor | 4.0 GB | 1.19 ms | 4.00x | 76 GB |
| int4, asymmetric, per tensor | 4.0 GB | 1.19 ms | 4.00x | 76 GB |
| int4, symmetric, per channel | 4.0 GB | 1.20 ms | 3.99x | 76 GB |
| int4, symmetric, per group of 64 | 4.5 GB | 1.34 ms | 3.56x | 75 GB |
| int4, symmetric, per group of 32 | 5.0 GB | 1.49 ms | 3.20x | 75 GB |
| int4, asymmetric, per group of 32 | 6.0 GB | 1.79 ms | 2.67x | 74 GB |

The book's 8B model on one 80 GB accelerator. The time to read the weights is the floor under every decode step (Chapter 4), and what is left over is what Chapter 13 spends on the KV cache -- so a smaller model is worth more than its own size, because the memory it frees becomes batch size.
