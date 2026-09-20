| Operation | Arithmetic per byte | Rate achieved | Share of the bound | Limited by |
|---|---|---|---|---|
| 16x16 multiply | 2.67 | 7.0 GFLOP/s | **20%** | memory |
| 64x64 multiply | 10.67 | 97.3 GFLOP/s | **70%** | memory |
| 256x256 multiply | 42.67 | 269.3 GFLOP/s | **58%** | compute |
| 1024x1024 multiply | 170.67 | 466.2 GFLOP/s | **100%** | compute |
| 2048x2048 multiply | 341.33 | 461.9 GFLOP/s | **99%** | compute |
| streaming dot product | 0.50 | 6.5 GFLOP/s | **100%** | memory |
| tinyserve prefill | 160.33 | 13.2 GFLOP/s | **3%** | compute |
| tinyserve decode | 0.48 | 4.1 GFLOP/s | **65%** | memory |

Peak 466 GFLOP/s and bandwidth 13.1 GB/s are the largest values measured here, so the two operations that define them reach 100% by construction. No operation exceeds the bound.
