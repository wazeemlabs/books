# Results (1 seeds, mean ± std)

World: 40000 facts, 14444 seen, 8790 singletons, singleton rate N1/M = 0.146, missing mass = 0.149, unseen share of test queries = 0.106.

## 1. Can the weights tell 'seen once' from 'never seen'? (AUROC, 0.5 = coin flip)

| model | params | seen 1x vs unseen | seen 2-3x | seen 4-7x | seen 8x+ |
|---|---|---|---|---|---|
| plain LM d=32 | 42,112 | 0.446 ± 0.000 | 0.577 ± 0.000 | 0.838 ± 0.000 | 0.998 ± 0.000 |
| plain LM d=64 | 133,376 | 0.809 ± 0.000 | 0.942 ± 0.000 | 0.978 ± 0.000 | 0.998 ± 0.000 |
| plain LM d=128 | 463,360 | 0.977 ± 0.000 | 0.987 ± 0.000 | 0.994 ± 0.000 | 1.000 ± 0.000 |
| IDK-trained LM d=32 | 42,112 | 0.561 ± 0.000 | 0.723 ± 0.000 | 0.927 ± 0.000 | 0.999 ± 0.000 |
| IDK-trained LM d=64 | 133,376 | 0.774 ± 0.000 | 0.954 ± 0.000 | 0.988 ± 0.000 | 0.999 ± 0.000 |
| IDK-trained LM d=128 | 463,360 | 0.980 ± 0.000 | 0.990 ± 0.000 | 0.995 ± 0.000 | 1.000 ± 0.000 |
| Bloom filter, 10 bits/key | 0 | ≈ 0.995 (1 − FPR/2) | same | same | same |

## 2. Accuracy, hallucination and memory on the natural query stream

Accuracy = correct answers / all queries. Hallucination = wrong answers / all queries (any answer to a never-seen fact counts as a hallucination). Acc@hal≤1% is the best accuracy the model reaches at some confidence threshold while keeping hallucination at or under 1%.

| system | acc (default) | halluc (default) | acc @ hal≤1% | acc @ hal≤5% | ghost-person halluc | total bits |
|---|---|---|---|---|---|---|
| plain LM d=32 | 75.6 ± 0.0 | 24.4 ± 0.0 | 68.7 ± 0.0 | 73.1 ± 0.0 | 100.0 ± 0.0 | 0.67 M |
| plain LM d=64 | 89.3 ± 0.0 | 10.7 ± 0.0 | 80.1 ± 0.0 | 88.5 ± 0.0 | 100.0 ± 0.0 | 2.13 M |
| plain LM d=128 | 89.4 ± 0.0 | 10.6 ± 0.0 | 89.4 ± 0.0 | 89.4 ± 0.0 | 100.0 ± 0.0 | 7.41 M |
| IDK-trained LM d=32 | 73.6 ± 0.0 | 4.9 ± 0.0 | 71.7 ± 0.0 | 73.6 ± 0.0 | 14.0 ± 0.0 | 0.67 M |
| IDK-trained LM d=64 | 87.5 ± 0.0 | 6.6 ± 0.0 | 79.7 ± 0.0 | 86.9 ± 0.0 | 43.8 ± 0.0 | 2.13 M |
| IDK-trained LM d=128 | 89.4 ± 0.0 | 5.0 ± 0.0 | 89.4 ± 0.0 | 89.4 ± 0.0 | 38.9 ± 0.0 | 7.41 M |
| store_only | 89.4 ± 0.0 | 0.0 ± 0.0 | 89.4 ± 0.0* | 89.4 ± 0.0* | 0.0 ± 0.0 | 0.61 M |
| bloom_gated_d32 | 75.6 ± 0.0 | 13.9 ± 0.0 | 75.6 ± 0.0* | 75.6 ± 0.0* | 0.0 ± 0.0 | 0.89 M |
| bloom_gated_d64 | 89.3 ± 0.0 | 0.1 ± 0.0 | 89.3 ± 0.0* | 89.3 ± 0.0* | 0.0 ± 0.0 | 2.35 M |
| bloom_gated_d128 | 89.4 ± 0.0 | 0.0 ± 0.0 | 89.4 ± 0.0* | 89.4 ± 0.0* | 0.0 ± 0.0 | 7.63 M |
| cmlm_d32_k1 | 89.4 ± 0.0 | 0.0 ± 0.0 | 89.4 ± 0.0* | 89.4 ± 0.0* | 0.0 ± 0.0 | 1.21 M |
| cmlm_d32_k2 | 89.4 ± 0.0 | 0.0 ± 0.0 | 89.4 ± 0.0* | 89.4 ± 0.0* | 0.0 ± 0.0 | 1.15 M |
| cmlm_d32_k3 | 89.4 ± 0.0 | 0.0 ± 0.0 | 89.4 ± 0.0* | 89.4 ± 0.0* | 0.0 ± 0.0 | 1.20 M |
| cmlm_d32_k5 | 89.4 ± 0.0 | 0.0 ± 0.0 | 89.4 ± 0.0* | 89.4 ± 0.0* | 0.0 ± 0.0 | 1.24 M |
| cmlm_d32_k10 | 89.4 ± 0.0 | 0.0 ± 0.0 | 89.4 ± 0.0* | 89.4 ± 0.0* | 0.0 ± 0.0 | 1.26 M |
| cmlm_d64_k1 | 89.4 ± 0.0 | 0.0 ± 0.0 | 89.4 ± 0.0* | 89.4 ± 0.0* | 0.0 ± 0.0 | 2.36 M |
| cmlm_d64_k2 | 89.4 ± 0.0 | 0.0 ± 0.0 | 89.4 ± 0.0* | 89.4 ± 0.0* | 0.0 ± 0.0 | 2.59 M |
| cmlm_d64_k3 | 89.4 ± 0.0 | 0.0 ± 0.0 | 89.4 ± 0.0* | 89.4 ± 0.0* | 0.0 ± 0.0 | 2.66 M |
| cmlm_d64_k5 | 89.4 ± 0.0 | 0.0 ± 0.0 | 89.4 ± 0.0* | 89.4 ± 0.0* | 0.0 ± 0.0 | 2.70 M |
| cmlm_d64_k10 | 89.4 ± 0.0 | 0.0 ± 0.0 | 89.4 ± 0.0* | 89.4 ± 0.0* | 0.0 ± 0.0 | 2.72 M |

\* the complementary systems are run at a single operating point; their hallucination is already under 1%, so their default accuracy is their acc@hal≤1%.

## 3. Consolidation threshold k (facts seen ≥ k times are consolidated into the core)

| system | store entries | answered from store | answered from core | core recall on facts seen ≥10x | consolidation failures | total bits |
|---|---|---|---|---|---|---|
| cmlm_d32_k1 | 11975 | 18.5 ± 0.0 | 70.9 ± 0.0 | 100.0 ± 0.0 | 11975 | 1.21 M |
| cmlm_d32_k2 | 9648 | 11.2 ± 0.0 | 78.3 ± 0.0 | 100.0 ± 0.0 | 858 | 1.15 M |
| cmlm_d32_k3 | 11387 | 14.6 ± 0.0 | 74.8 ± 0.0 | 100.0 ± 0.0 | 0 | 1.20 M |
| cmlm_d32_k5 | 12893 | 20.9 ± 0.0 | 68.5 ± 0.0 | 100.0 ± 0.0 | 0 | 1.24 M |
| cmlm_d32_k10 | 13767 | 28.8 ± 0.0 | 60.6 ± 0.0 | 100.0 ± 0.0 | 0 | 1.26 M |
| cmlm_d64_k1 | 295 | 0.3 ± 0.0 | 89.1 ± 0.0 | 100.0 ± 0.0 | 295 | 2.36 M |
| cmlm_d64_k2 | 8790 | 9.1 ± 0.0 | 80.3 ± 0.0 | 100.0 ± 0.0 | 0 | 2.59 M |
| cmlm_d64_k3 | 11387 | 14.6 ± 0.0 | 74.8 ± 0.0 | 100.0 ± 0.0 | 0 | 2.66 M |
| cmlm_d64_k5 | 12893 | 20.9 ± 0.0 | 68.5 ± 0.0 | 100.0 ± 0.0 | 0 | 2.70 M |
| cmlm_d64_k10 | 13767 | 28.8 ± 0.0 | 60.6 ± 0.0 | 100.0 ± 0.0 | 0 | 2.72 M |

Good-Turing check (theory note, P4). Share of seen-fact queries that still need the store, predicted from the training count histogram alone vs measured:

| system | predicted Σ_{n<k} θ_n / (1 − N1/M) | measured |
|---|---|---|
| cmlm_d32_k1 | 0.0% | 20.7% ± 0.0 |
| cmlm_d32_k2 | 10.1% | 12.5% ± 0.0 |
| cmlm_d32_k3 | 15.8% | 16.4% ± 0.0 |
| cmlm_d32_k5 | 23.1% | 23.4% ± 0.0 |
| cmlm_d32_k10 | 32.2% | 32.2% ± 0.0 |
| cmlm_d64_k1 | 0.0% | 0.3% ± 0.0 |
| cmlm_d64_k2 | 10.1% | 10.2% ± 0.0 |
| cmlm_d64_k3 | 15.8% | 16.4% ± 0.0 |
| cmlm_d64_k5 | 23.1% | 23.4% ± 0.0 |
| cmlm_d64_k10 | 32.2% | 32.2% ± 0.0 |

## 4. Does 'I don't know' training cost the weights recall? (forced answer on seen facts)

| size | plain LM | IDK-trained LM |
|---|---|---|
| d=32 | 84.5 ± 0.0 | 84.0 ± 0.0 |
| d=64 | 99.9 ± 0.0 | 98.2 ± 0.0 |
| d=128 | 100.0 ± 0.0 | 100.0 ± 0.0 |

IDK-trained LM: how often it answers, by exposure count (default threshold)

| size | never seen | seen 1x | 2-3x | 4-7x | 8x+ | ghost person |
|---|---|---|---|---|---|---|
| d=32 | 16.6 ± 0.0 | 22.3 ± 0.0 | 49.1 ± 0.0 | 87.6 ± 0.0 | 100.0 ± 0.0 | 14.0 ± 0.0 |
| d=64 | 52.1 ± 0.0 | 90.5 ± 0.0 | 99.8 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 43.8 ± 0.0 |
| d=128 | 47.5 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 38.9 ± 0.0 |

## 5. Graded abstention (complementary systems)

On never-seen facts, the right kind of 'I don't know' (unknown person vs known person, unknown fact): 99.3 ± 0.0. Ghost people labelled 'never heard of them': 98.9 ± 0.0.

