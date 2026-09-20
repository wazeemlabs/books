| Batch | Prompt tokens that are real | Decode steps that are real | Together | Steps the batch runs | Steps a sequence needs |
|---|---|---|---|---|---|
| 1 | 100% | 100% | **100%** | 298 | 298 |
| 2 | 76% | 73% | **76%** | 407 | 298 |
| 4 | 61% | 56% | **60%** | 534 | 298 |
| 8 | 50% | 44% | **49%** | 672 | 298 |
| 16 | 42% | 38% | **41%** | 794 | 298 |
| 32 | 36% | 33% | **35%** | 899 | 298 |
| 64 | 31% | 30% | **31%** | 983 | 298 |

Batches filled from 4,000 sampled requests (seed 0), each run until its longest sequence finishes. Nothing here is an implementation flaw: it all follows from the batch being fixed for its whole life.
