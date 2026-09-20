| Values per scale | Error | Bytes a weight | Over four bits |
|---|---|---|---|
| 128 | 2.78e-02 | 0.531 | +6% |
| 64 | 2.55e-02 | 0.562 | +12% |
| 32 | 2.29e-02 | 0.625 | +25% |
| 16 | 2.02e-02 | 0.750 | +50% |

Four-bit symmetric quantization at four group sizes. Every halving of the group buys a little accuracy and costs a fixed amount of storage, because each group needs its own float32 scale. The knee is where a reader's own tolerance puts it; 32 and 128 are the sizes the published methods use.
