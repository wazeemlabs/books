| Benchmark | Items | Smallest difference it can find, paired | Unpaired | Pairing is worth |
|---|---|---|---|---|
| MMLU (test) | 14,042 | **0.54 points** | 1.27 points | 2.4x |
| GSM8K (test) | 1,319 | **1.79 points** | 4.28 points | 2.4x |
| HumanEval | 164 | **4.88 points** | -- | -- |
| an internal eval | 200 | **4.41 points** | -- | -- |

Simulated, 4,000 runs per point, seed 0: the smallest true difference each benchmark finds 80% of the time at the 5% level, when the two models disagree on 5% of items. "Paired" is McNemar's test, which looks only at the items the two models answer differently. "Unpaired" is the two accuracy rates compared as though they came from different samples, which is what gets run. A dash means no difference the stated disagreement allows is ever found 80% of the time -- at those sizes the unpaired test cannot settle the question at all. Test-set sizes are each dataset's own published split (FACTS.md).
