# Results: checksum-aided recall (3 seeds, mean ± std)

Hard world: 10,000 people × 4 relations, answers are 2-token strings from 1,024 values. 14,444 distinct facts seen in training, singleton rate N1/M = 0.146, never-seen share of test queries = 0.101. Measured filter false-positive rates: triple checksum 0.00129 (14 bits/fact), key filter 0.0079, entity filter 0.0079.

Accuracy = correct answers / all queries. Hallucination = wrong answers / all queries (any answer to a never-seen fact counts). Weights-only systems are shown at their best confidence threshold under each hallucination budget.

## 1. Main result

| weights | params | weights only: greedy acc / halluc | weights only: acc @ hal ≤ 1% | + IDK training: acc @ hal ≤ 1% | **CAR N=1** acc / halluc | **CAR+prefix N=50** acc / halluc | **CARE** acc / halluc | CARE store (% of facts) |
|---|---|---|---|---|---|---|---|---|
| d=16 | 12,352 | 50.7 ± 2.1 / 49.3 ± 2.1 | 49.7 ± 2.4 | 54.6 ± 0.5 | 50.7 ± 2.1 / 0.06 ± 0.02 | 86.0 ± 0.3 / 0.73 ± 0.06 | 89.9 ± 0.3 / 0.00 ± 0.00 | 94 ± 0 |
| d=32 | 36,992 | 65.8 ± 1.2 / 34.2 ± 1.2 | 64.4 ± 1.4 | 68.8 ± 0.2 | 65.8 ± 1.2 / 0.01 ± 0.01 | 88.2 ± 0.2 / 0.43 ± 0.07 | 89.9 ± 0.3 / 0.00 ± 0.00 | 80 ± 1 |
| d=64 | 123,136 | 82.8 ± 0.3 / 17.2 ± 0.3 | 78.9 ± 0.2 | 79.3 ± 0.4 | 82.8 ± 0.3 / 0.00 ± 0.00 | 89.8 ± 0.4 / 0.04 ± 0.01 | 89.9 ± 0.3 / 0.00 ± 0.00 | 23 ± 0 |
| d=96 | 258,432 | 89.3 ± 0.1 / 10.7 ± 0.1 | 85.0 ± 0.7 | 85.1 ± 0.5 | 89.3 ± 0.1 / 0.00 ± 0.00 | 89.9 ± 0.3 / 0.00 ± 0.00 | 89.9 ± 0.3 / 0.00 ± 0.00 | 1 ± 1 |
| store only (no weights) | 0 | | | | | | 89.9 ± 0.3 / 0.00 ± 0.00 | 100 |

## 2. Memory outside the weights (bits per seen fact)

| system | non-weight bits / seen fact | accuracy | hallucination |
|---|---|---|---|
| store only (exact key + value + count, plus filters) | 46.0 | 89.9 ± 0.3 | 0.00 ± 0.00 |
| no weights: full scan of all 1,024 values, 14-bit checksum | 29.0 | 52.4 ± 0.4 | 37.44 ± 0.25 |
| no weights: full scan, 20-bit checksum | 35.0 | 86.2 ± 0.9 | 3.61 ± 0.55 |
| no weights: full scan, 28-bit checksum | 43.0 | 89.8 ± 0.4 | 0.03 ± 0.01 |
| no weights: prefix checksum, random order, N=50 | 37.0 | 75.7 ± 0.3 | 1.97 ± 0.62 |
| CAR N=1, weights d=64 | 29.0 | 82.8 ± 0.3 | 0.00 ± 0.00 |
| CAR N=20, weights d=64 | 29.0 | 86.5 ± 0.4 | 0.10 ± 0.02 |
| CAR+prefix N=50, weights d=64 | 37.0 | 89.8 ± 0.4 | 0.04 ± 0.01 |
| CARE N=20, weights d=64 | 36.2 | 89.9 ± 0.3 | 0.00 ± 0.00 |

## 3. List size N, and the list-decoding formula (docs/theory.md §5)

| weights | N | accuracy (measured) | accuracy (predicted) | hallucination (measured) | hallucination (predicted) | tip of the tongue |
|---|---|---|---|---|---|---|
| d=16 | 1 | 50.7 ± 2.1 | 50.7 ± 2.1 | 0.06 ± 0.02 | 0.05 ± 0.00 | 39.2 ± 2.2 |
| d=16 | 5 | 52.5 ± 1.9 | 52.5 ± 1.9 | 0.25 ± 0.06 | 0.23 ± 0.02 | 37.2 ± 1.9 |
| d=16 | 20 | 54.6 ± 2.0 | 54.6 ± 2.0 | 0.92 ± 0.02 | 0.87 ± 0.08 | 34.4 ± 2.1 |
| d=32 | 1 | 65.8 ± 1.2 | 65.8 ± 1.2 | 0.01 ± 0.01 | 0.03 ± 0.00 | 24.1 ± 1.6 |
| d=32 | 5 | 67.8 ± 1.0 | 67.8 ± 1.0 | 0.10 ± 0.01 | 0.14 ± 0.00 | 21.9 ± 1.4 |
| d=32 | 20 | 70.5 ± 0.8 | 70.5 ± 0.8 | 0.49 ± 0.04 | 0.50 ± 0.02 | 18.9 ± 1.2 |
| d=64 | 1 | 82.8 ± 0.3 | 82.8 ± 0.3 | 0.00 ± 0.00 | 0.01 ± 0.00 | 7.1 ± 0.1 |
| d=64 | 5 | 84.7 ± 0.4 | 84.7 ± 0.4 | 0.02 ± 0.01 | 0.04 ± 0.00 | 5.2 ± 0.0 |
| d=64 | 20 | 86.5 ± 0.4 | 86.5 ± 0.4 | 0.10 ± 0.02 | 0.11 ± 0.01 | 3.3 ± 0.0 |
| d=96 | 1 | 89.3 ± 0.1 | 89.3 ± 0.1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.5 ± 0.3 |
| d=96 | 5 | 89.6 ± 0.2 | 89.6 ± 0.2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.3 ± 0.2 |
| d=96 | 20 | 89.8 ± 0.3 | 89.8 ± 0.3 | 0.00 ± 0.00 | 0.01 ± 0.00 | 0.1 ± 0.1 |

## 4. Graded 'I don't know' (share of all queries, CARE, largest weights)

- never heard of this person: 4.6 ± 0.2%
- know the person, never read this fact: 5.5 ± 0.2%
- tip of the tongue (read it, can't bring it back): 0.04 ± 0.01%
- ghost people (never existed) answered: 0.00 ± 0.00% (weights only, greedy: 100.0 ± 0.0%; IDK-trained: 36.9 ± 4.4%)

## 5. Noisy fact extraction (mentions filed under the wrong person), weights d=32

| mislinks | weights greedy halluc | store only acc / halluc | CAR N=1 | CAR N=5 | CAR N=20 | CARE | CARE strict |
|---|---|---|---|---|---|---|---|
| 5% | 34.2 | 89.0 / 1.09 | 65.8 / 0.08 | 67.7 / 0.31 | 70.2 / 0.86 | 89.0 / 1.09 | 81.2 / 0.49 |
| 10% | 34.5 | 88.2 / 2.11 | 65.5 / 0.34 | 67.4 / 0.80 | 69.6 / 1.45 | 88.2 / 2.11 | 80.9 / 1.03 |

CARE strict answers a fact read only once and not recalled by the weights with "I read that once, but can't confirm it" instead of asserting it.

## 6. Ablation: weights trained only on facts seen ≥ 2 times (d=32)

| | all facts | facts seen ≥ 2× |
|---|---|---|
| weights greedy accuracy | 65.8 ± 1.2 | 68.4 ± 0.6 |
| CAR N=1 accuracy / halluc | 65.8 ± 1.2 / 0.01 ± 0.01 | 68.4 ± 0.6 / 0.02 ± 0.00 |
| CAR N=5 accuracy / halluc | 67.8 ± 1.0 / 0.10 ± 0.01 | 71.0 ± 0.5 / 0.11 ± 0.01 |
| CAR+prefix N=50 accuracy / halluc | 88.2 ± 0.2 / 0.43 ± 0.07 | 88.0 ± 0.3 / 0.36 ± 0.03 |

