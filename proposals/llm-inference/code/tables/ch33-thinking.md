| Thinking tokens | Output | Visible | Sequences that fit | Seconds an answer | Dollars a thousand answers | Against no thinking |
|---|---|---|---|---|---|---|
| 0 | 300 | 100.0% | 325 | 2.5 | $0.035 | 1x |
| 1,024 | 1,324 | 22.7% | 193 | 12.5 | $0.177 | 5x |
| 2,048 | 2,348 | 12.8% | 137 | 25.2 | $0.355 | 10x |
| 4,096 | 4,396 | 6.8% | 87 | 58.4 | $0.824 | 24x |
| 8,192 | 8,492 | 3.5% | 50 | 131.1 | $2.366 | 68x |
| 16,384 | 16,684 | 1.8% | 27 | 247.9 | $8.289 | 238x |
| 32,768 | 33,068 | 0.9% | 14 | 479.2 | $30.902 | 888x |

The same 1,200-token prompt and the same 300-token visible answer, with thinking in front of it. Costs rise faster than the token count because the reply's own cache grows as it is written, so fewer sequences fit and each one has less of the machine to share. Anthropic's documentation puts the floor on a thinking budget at 1,024 tokens and advises batch processing above 32,768, where requests "can hit system timeouts and open-connection limits" -- which the last row's 479 seconds explains.
