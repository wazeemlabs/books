| Batch | Nodes a pass carries free | Best tree | Its speedup | The batch-1 tree instead | Tokens 64 nodes must yield |
|---|---|---|---|---|---|
| 1 | 302 | (4, 2, 2, 1, 1, 1, 1, 2) (124 nodes) | x4.65 | x4.65 | 1.20 |
| 8 | 42 | (2, 2, 1, 1, 1, 1, 1, 1) (30 nodes) | x4.01 | x1.54 | 1.87 |
| 32 | 14 | (1, 1, 1, 1, 1, 1, 1, 1) (8 nodes) | x3.20 | x0.49 | 5.90 |
| 128 | 7 | (1, 1, 1, 1, 1, 1) (6 nodes) | x3.06 | x0.23 | 12.78 |

Arithmetic over the reference model at a 1,500-token context, for a drafter whose top-1 acceptance is 0.70 and whose ranked alternatives are modelled rather than measured. "Nodes a pass carries free" is where the verification pass stops being memory-bound and starts paying for every extra token. The fifth column is the same tree the first row chose, run at that batch. The last is what a round would have to accept, with Medusa's three heads on the model, for 64 nodes to be worth verifying at all.
