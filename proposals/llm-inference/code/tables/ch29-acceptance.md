| Draft | Bytes a weight | Picks the same token | Guess survives, sampled |
|---|---|---|---|
| int8, per tensor | 1.000 | 97.5% | 99.0% |
| int4, groups of 32 | 0.625 | 70.0% | 90.3% |
| int4, per tensor | 0.500 | 55.0% | 83.5% |
| int2, per tensor | 0.250 | 0.0% | 48.5% |
| an unrelated small model | -- | 0.0% | -- |

Over 40 positions of one prompt. The drafts are quantized copies of the target from Chapter 24: cheaper to run, and related to what they are drafting for, which is the property that matters. An unrelated model of the same architecture agrees at 0.20%, which is chance.

The two columns differ because the rule does not require the draft to pick the same token -- it accepts in proportion to how much the two distributions overlap. Read the last column with care: this model is untrained, so its output is nearly flat (5.72 nats against 6.24 for a uniform distribution over the same vocabulary), and two flat distributions overlap heavily whatever they are. A trained model is far more confident and its acceptance rates are correspondingly lower.
