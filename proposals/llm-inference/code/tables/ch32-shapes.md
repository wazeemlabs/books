| Traffic | What the key folds | Distinct keys | Hit rate | Served wrong |
|---|---|---|---|---|
| FAQ | as typed | 5,123 | 74.4% | none |
| FAQ | case and punctuation | 4,518 | 77.4% | none |
| FAQ | and stop words dropped | 4,506 | 77.5% | **3.77%** |
| chat | as typed | 10,387 | 34.0% | none |
| chat | case and punctuation | 10,191 | 35.2% | none |
| chat | and stop words dropped | 10,179 | 35.3% | **1.53%** |
| agent | as typed | 16,339 | 9.2% | none |
| agent | case and punctuation | 16,247 | 9.7% | none |
| agent | and stop words dropped | 16,236 | 9.8% | **0.69%** |

One day of traffic over a catalogue of 2,000 distinct questions asked with a Zipf skew of 1.1, seed 0: 20,000 single-turn requests, 6,000 conversations averaging 2.6 turns, and 3,000 agent tasks of 6 steps each. Only 17% of the agent requests are the first step of a task; every later one carries a transcript no other session has produced, so it cannot repeat. Folding case and punctuation is free. Dropping function words is not: it makes "is this covered" and "is this not covered" the same key.
