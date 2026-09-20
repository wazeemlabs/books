| Format | Bits | Sign/exponent/mantissa | Largest | Smallest with full precision | Gap either side of 1.0 | Decimal digits | Peak on one H100 |
|---|---|---|---|---|---|---|---|
| float32 | 32 | 1 / 8 / 23 | 3.403e+38 | 1.18e-38 | 1.19e-07 | 7.2 | 67 TFLOP/s |
| tensorfloat32 * | 19 | 1 / 8 / 10 | 3.401e+38 | 1.18e-38 | 0.000977 | 3.3 | 495 TFLOP/s |
| **float16** | 16 | 1 / 5 / 10 | 6.55e+04 | 6.1e-05 | 0.000977 | 3.3 | 990 TFLOP/s |
| **bfloat16** | 16 | 1 / 8 / 7 | 3.39e+38 | 1.18e-38 | 0.00781 | 2.4 | 990 TFLOP/s |
| float8 e4m3 | 8 | 1 / 4 / 3 | 448 | 0.0156 | 0.125 | 1.2 | 1,980 TFLOP/s |
| float8 e5m2 | 8 | 1 / 5 / 2 | 5.734e+04 | 6.1e-05 | 0.25 | 0.9 | 1,980 TFLOP/s |

Every column but the last is arithmetic on the three field widths, computed in `tinyserve/precision.py`. \* tensorfloat32 is a compute format: 19 meaningful bits held in a 32-bit slot, so it changes how a multiply is done and not what a weight costs to store. Peak figures are the H100 SXM datasheet's, halved from the quoted "with sparsity" rows to the dense throughput that dense inference gets; float32 is the one that never reaches a tensor core (FACTS.md).
