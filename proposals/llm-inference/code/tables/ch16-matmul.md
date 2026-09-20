| Sequences | Time for one 1280x5120 matmul | Weights re-read at | Arithmetic rate | Time per token |
|---|---|---|---|---|
| 1 | 0.54 ms | 49 GB/s | 24 GFLOP/s | **537 us** |
| 2 | 1.99 ms \* | 13 GB/s | 13 GFLOP/s | **997 us** |
| 4 | 1.98 ms | 13 GB/s | 26 GFLOP/s | **496 us** |
| 8 | 2.06 ms \* | 13 GB/s | 51 GFLOP/s | **257 us** |
| 16 | 2.16 ms | 12 GB/s | 97 GFLOP/s | **135 us** |
| 32 | 2.67 ms | 10 GB/s | 157 GFLOP/s | **83 us** |
| 64 | 3.80 ms | 7 GB/s | 220 GFLOP/s | **59 us** |

30 weight matrices of 25 MiB (750 MiB in total, more than this machine's cache holds), each multiplied once by a batch of rows; the time is one pass divided by the number of matrices. Nothing else is in the measurement: no attention, no cache, no model.

\* run-to-run spread exceeded 5%.
