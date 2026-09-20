| | This machine (measured) | An H100 (published) |
|---|---|---|
| Memory bandwidth | 13 GB/s | 3.35 TB/s |
| Arithmetic | 156 GFLOP/s | 990 TFLOP/s |
| **Breaks even at** | **12** FLOP/byte | **296** FLOP/byte |

The accelerator is 26x more lopsided: it carries far more arithmetic per unit of memory bandwidth, so work that is short of arithmetic is punished far more severely on it. Measured single-threaded; see the note on the measuring machine.
