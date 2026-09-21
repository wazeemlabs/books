| The reply is | A guess at all | 1 token | 2 | 4 | 8 | Tokens a round | Speedup at 8 nodes |
|---|---|---|---|---|---|---|---|
| an extract from the prompt | 100% | 92% | 86% | 77% | 60% | 7.03 | x7.03 |
| an extract, lightly edited | 88% | 73% | 63% | 49% | 29% | 4.83 | x4.83 |
| prose continuing the prompt | 37% | 7% | 2% | 0% | 0% | 1.11 | x1.11 |
| text unrelated to the prompt | 46% | 13% | 6% | 3% | 1% | 1.35 | x1.35 |

Measured over 1,200 words of real prose per row, with the context indexed as the reply is written and n-grams of 2 to 8 words, seed 0. The columns are the chance of getting at least that many tokens from one round, which is the product of the per-depth rates and not any one of them. The first two rows are input-grounded replies -- an extract from the document, and the same extract with one word in ten changed. The third continues the document without quoting it. The fourth is the control: a reply with nothing to do with the prompt, which still earns a little by copying from what it has already written.
