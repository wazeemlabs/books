| The model's numbers | Largest | tensorfloat32 | float16 | bfloat16 | float8 e4m3 | float8 e5m2 |
|---|---|---|---|---|---|---|
| attention weights (wq) | 0.356 | 5.15e-05 | 5.15e-05 | 4.16e-04 | 6.54e-03 | 1.31e-02 |
| feed-forward weights (w1) | 0.39 | 4.69e-05 | 4.69e-05 | 3.76e-04 | 6.02e-03 | 1.21e-02 |
| token embeddings | 0.403 | 4.57e-05 | 4.57e-05 | 3.67e-04 | 5.88e-03 | 1.16e-02 |
| activations, after a layer | 3.29 | 2.33e-05 | 2.33e-05 | 1.88e-04 | 2.99e-03 | 6.10e-03 |
| attention weights, after softmax | 1 | 8.16e-06 | 8.16e-06 | 6.68e-05 | 1.15e-03 | 2.00e-03 |
| logits | 4.07 | 5.20e-05 | 5.20e-05 | 4.19e-04 | 6.65e-03 | 1.33e-02 |

Root-mean-square error after a round trip through each format, as a fraction of the largest value in the tensor. From a 4-layer model of width 128 over 64 tokens. Nothing here overflows: every number in this model is small. float16 and tensorfloat32 agree down the column because they have the same number of mantissa bits, which is what error at this scale depends on -- the extra exponent bits buy range, and range is not what is being tested.
