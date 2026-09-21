| Token budget | Tokens/s | TTFT p50 | TTFT p99 | Between tokens, p50 | Between tokens, p99 | Iterations carrying prefill | Keeps |
|---|---|---|---|---|---|---|---|
| _prefill alone_ | 6,665 | 33 ms | 402 ms | 17.3 ms | 185.9 ms | 36% | TTFT only |
| 128 | 3,886 | 77.1 s | 153.3 s | 6.2 ms | 6.6 ms | 98% | between-token only |
| 256 | 6,163 | 10.3 s | 20.1 s | 7.4 ms | 8.0 ms | 96% | between-token only |
| **512** | 6,764 | 42 ms | 259 ms | 8.0 ms | 9.4 ms | 57% | both |
| 1,024 | 6,754 | 35 ms | 160 ms | 9.4 ms | 17.0 ms | 42% | both |
| 2,048 | 6,746 | 32 ms | 132 ms | 10.3 ms | 33.1 ms | 32% | both |
| 4,096 | 6,744 | 32 ms | 130 ms | 10.5 ms | 50.2 ms | 30% | TTFT only |
| 8,192 | 6,744 | 32 ms | 128 ms | 10.5 ms | 50.2 ms | 30% | TTFT only |
| 16,384 | 6,744 | 32 ms | 128 ms | 10.5 ms | 50.2 ms | 30% | TTFT only |

4800 requests at a rate of 24 a second (seed 0), the rate at which the previous chapter's scheduler stopped keeping its promise. The first row is that scheduler: a prefill gets an iteration to itself. Every row below mixes prefill into the same iteration as the decodes, splitting it when it does not fit in the budget. "Keeps" is against the case study's p99 promises: 1,000 ms to the first token and 50 ms between them.
