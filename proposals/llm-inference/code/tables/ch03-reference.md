| Phase | Arithmetic | Bytes fetched | Work per byte | Limited by |
|---|---|---|---|---|
| Reading the prompt (prefill) | 18.7 TFLOP | 16.0 GB | **1,168.39** | compute |
| Writing a token (decode) | 0.0 TFLOP | 16.2 GB | **0.97** | memory |

An 8B model in bf16 on an accelerator that breaks even at **296** operations per byte: below that it waits for memory, above it it waits for arithmetic. Prefill of a 1,200-token prompt; decode at a 1,500-token context. Arithmetic over published specs, not a measurement.
