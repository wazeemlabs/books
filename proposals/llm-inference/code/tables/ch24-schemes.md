| Scheme | Bytes a weight | Values sharing a scale | Error, typical | Error, worst | Against bfloat16 |
|---|---|---|---|---|---|
| _bfloat16, for comparison_ | 2.000 | 1 (each carries its own exponent) | 3.91e-04 | -- | 1x |
| int8, symmetric, per tensor | 1.000 | 16,384 | 2.27e-03 | 3.94e-03 | 6x worse |
| int8, asymmetric, per tensor | 1.000 | 16,384 | 2.19e-03 | 3.89e-03 | 6x worse |
| int8, symmetric, per channel | 1.031 | 128 | 1.57e-03 | 3.93e-03 | 4x worse |
| int4, symmetric, per tensor | 0.500 | 16,384 | 4.13e-02 | 7.14e-02 | 105x worse |
| int4, asymmetric, per tensor | 0.500 | 16,384 | 3.71e-02 | 6.62e-02 | 95x worse |
| int4, symmetric, per channel | 0.531 | 128 | 2.84e-02 | 7.14e-02 | 73x worse |
| int4, symmetric, per group of 64 | 0.562 | 64 | 2.55e-02 | 7.14e-02 | 65x worse |
| int4, symmetric, per group of 32 | 0.625 | 32 | 2.29e-02 | 7.14e-02 | 59x worse |
| int4, asymmetric, per group of 32 | 0.750 | 32 | 1.91e-02 | 6.01e-02 | 49x worse |

Root-mean-square error after a round trip, averaged over all 24 weight matrices of the book's small model and reported as a fraction of the largest weight in each. "Bytes a weight" includes the scales: a scale is a float32 and an asymmetric scheme needs a zero point beside it, so a small group is not as cheap as its bit width suggests.
