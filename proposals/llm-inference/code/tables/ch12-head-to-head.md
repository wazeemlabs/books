| Measurement | Without a cache | With a KV cache | Ratio |
|---|---|---|---|
| Time to first token (prefill) | 22.24 ms | 20.27 ms | **1.10x** |
| Decode step, p50 | 27.83 ms | 0.51 ms | **55x** |
| Decode step, p99 | 45.05 ms | 0.70 ms | **65x** |
| Decode throughput | 36 tok/s | 1,937 tok/s | **54x** |
| Total, 128 prompt + 128 generated | 3.580 s | 0.086 s | **42x** |
