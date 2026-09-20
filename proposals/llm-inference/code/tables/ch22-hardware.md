| Format | Bytes a number | Weights | Time to read them | KV cache a token | Ridge point | Compute-bound above batch | Peak |
|---|---|---|---|---|---|---|---|
| float32 | 4 | 32 GB | 9.55 ms | 256 KiB | 20 FLOP/byte | 40 | 67 TFLOP/s |
| tensorfloat32 | 4 | 32 GB | 9.55 ms | 256 KiB | 148 FLOP/byte | 296 | 495 TFLOP/s |
| float16 | 2 | 16 GB | 4.78 ms | 128 KiB | 296 FLOP/byte | 296 | 990 TFLOP/s |
| bfloat16 | 2 | 16 GB | 4.78 ms | 128 KiB | 296 FLOP/byte | 296 | 990 TFLOP/s |
| float8 e4m3 | 1 | 8 GB | 2.39 ms | 64 KiB | 591 FLOP/byte | 296 | 1,980 TFLOP/s |
| float8 e5m2 | 1 | 8 GB | 2.39 ms | 64 KiB | 591 FLOP/byte | 296 | 1,980 TFLOP/s |

The book's 8B model on one H100, by the format its weights and cache are kept in. "Time to read them" is the weights over 3.35 TB/s, which Chapter 4 showed is the floor under every decode step. The ridge point is where the accelerator stops being limited by memory and starts being limited by arithmetic; it depends on the arithmetic the format reaches and not at all on how many bytes a number takes. The last column puts the two together: a decode step at batch B does 2PB arithmetic and reads P times the format's bytes, so halving the format doubles its intensity exactly as it doubles the ridge, and the batch at which the step turns compute-bound does not move.
