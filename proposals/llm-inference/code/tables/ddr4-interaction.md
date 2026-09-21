| Cache hit rate | Spend it on machines | Batch | Speculation there | Or keep the fleet | Batch | Speculation there |
|---|---|---|---|---|---|---|
| 0% | 8 machines, $26.00/hr | 60 | x3.13 | 8 machines, $26.00/hr | 60 | x3.13 |
| 10% | 7 machines, $22.75/hr | 60 | x3.13 | 8 machines, $26.00/hr | 52 | x3.15 |
| 20% | 7 machines, $22.75/hr | 60 | x3.13 | 8 machines, $26.00/hr | 44 | x3.17 |
| 30% | 6 machines, $19.50/hr | 60 | x3.13 | 8 machines, $26.00/hr | 37 | x3.18 |
| 40% | 5 machines, $16.25/hr | 60 | x3.13 | 8 machines, $26.00/hr | 30 | x3.23 |
| 50% | 4 machines, $13.00/hr | 60 | x3.13 | 8 machines, $26.00/hr | 24 | x3.36 |

A cache hit can be spent two ways: shrink the fleet, or keep it and let each machine run cooler. Only the second lowers the batch, and a lower batch is where speculative decoding is supposed to come into its own. It does not: across the whole range the speedup moves by 7%. The batch sizes come from Chapter 41's own sweep and the speedups from Chapter 30's, interpolated between the loads and batches each measured and never extrapolated past them.
