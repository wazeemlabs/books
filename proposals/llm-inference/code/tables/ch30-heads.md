| Drafter | Parameters | Bytes | Of the model | Every decode step |
|---|---|---|---|---|
| Medusa, 1 heads | 0.54B | 1.08 GB | 6.8% | x1.068 |
| Medusa, 2 heads | 1.08B | 2.17 GB | 13.5% | x1.135 |
| Medusa, 3 heads | 1.63B | 3.25 GB | 20.3% | x1.203 |
| Medusa, 5 heads | 2.71B | 5.42 GB | 33.9% | x1.339 |
| EAGLE draft head | 0.24B | 0.48 GB | 3.0% | x1.030 |
| n-gram from the prompt | 0.00B | 0.00 GB | 0.0% | x1.000 |

Exact arithmetic over the reference model (4,096 hidden, 128,256 vocabulary, 16.0 GB of weights in bf16). A Medusa head is a square residual block plus its own projection to the whole vocabulary, which is where all of the cost is. EAGLE's figure is the one its authors publish for Vicuna-7B, the smallest model in their table, rather than a guess at the architecture. The last column is what the extra weights do to a decode step, which is memory-bound: bigger weights take proportionally longer to read, on every step, including the ones where the draft guesses wrong.
