| Token budget | Tokens/s | TTFT p50 | TTFT p99 | Between tokens, p50 | Between tokens, p99 | Iterations carrying prefill | Keeps |
|---|---|---|---|---|---|---|---|
| _prefill alone_ | 5,374 | 25 ms | 96 ms | 10.5 ms | 114.0 ms | 18% | TTFT only |
| 128 | 3,721 | 7.6 s | 16.5 s | 6.2 ms | 6.6 ms | 87% | between-token only |
| 256 | 5,384 | 211 ms | 1.5 s | 7.2 ms | 8.2 ms | 72% | between-token only |
| **512** | 5,586 | 35 ms | 116 ms | 7.9 ms | 8.4 ms | 39% | both |
| 1,024 | 5,536 | 30 ms | 104 ms | 8.4 ms | 16.5 ms | 26% | both |
| 2,048 | 5,506 | 27 ms | 107 ms | 8.8 ms | 33.1 ms | 19% | both |
| 4,096 | 5,499 | 27 ms | 108 ms | 8.9 ms | 42.5 ms | 17% | both |
| 8,192 | 5,499 | 27 ms | 103 ms | 8.9 ms | 42.5 ms | 17% | both |
| 16,384 | 5,499 | 27 ms | 103 ms | 8.9 ms | 42.5 ms | 17% | both |

600 requests at a rate of 24 a second (seed 0), the rate at which the previous chapter's scheduler stopped keeping its promise. The first row is that scheduler: a prefill gets an iteration to itself. Every row below mixes prefill into the same iteration as the decodes, splitting it when it does not fit in the budget. "Keeps" is against the case study's p99 promises: 1,000 ms to the first token and 50 ms between them.
