| Context | Cache, one sequence | Sequences that fit | Cache share of the step | Between tokens | Prefill | Attention's share of it |
|---|---|---|---|---|---|---|
| 1,024 | 0.13 GB | 476 | 35% | 7.3 ms | 16 ms | 3% |
| 4,096 | 0.54 GB | 119 | 68% | 15.0 ms | 71 ms | 13% |
| 8,192 | 1.07 GB | 59 | 80% | 23.7 ms | 159 ms | 22% |
| 16,384 | 2.15 GB | 29 | 80% | 23.4 ms | 390 ms | 36% |
| 32,768 | 4.29 GB | 14 | 79% | 22.7 ms | 1,064 ms | 53% |
| 65,536 | 8.59 GB | 7 | 79% | 22.7 ms | 3,264 ms | 70% |
| 131,072 | 17.18 GB | 3 | 76% | 20.2 ms | 11,078 ms | 82% |

Arithmetic over the reference model on one accelerator: 16.0 GB of weights leave 64 GB for caches, at 131,072 bytes a token. The batch in the fourth and fifth columns is whatever fits, up to 64. Two crossings worth remembering: one sequence's cache outweighs the entire model at 122,104 tokens, and attention overtakes everything else in a prefill at 28,523.
