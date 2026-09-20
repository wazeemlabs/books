| Machines | Tokens/s | Share of what the traffic asks for | TTFT p99 | Between tokens, p99 | $/hour | $/M output tokens |
|---|---|---|---|---|---|---|
| 4 | 37,044 | 62% (behind) | 5.2 s | 10.3 ms | $13.00 | $0.097 |
| 5 | 45,713 | 77% (behind) | 2.3 s | 10.3 ms | $16.25 | $0.099 |
| 6 | 53,613 | 90% (behind) | 601 ms | 10.3 ms | $19.50 | $0.101 |
| 7 | 56,087 | 94% (behind) | 253 ms | 10.0 ms | $22.75 | $0.113 |
| **8** | 57,143 | 96% | 105 ms | 8.8 ms | $26.00 | $0.126 |
| 9 | 57,425 | 96% | 84 ms | 8.5 ms | $29.25 | $0.141 |
| 10 | 57,648 | 97% | 82 ms | 8.4 ms | $32.50 | $0.157 |
| 11 | 57,770 | 97% | 79 ms | 8.4 ms | $35.75 | $0.172 |
| 12 | 57,861 | 97% | 77 ms | 8.4 ms | $39.00 | $0.187 |
| 14 | 58,011 | 97% | 72 ms | 8.4 ms | $45.50 | $0.218 |
| 16 | 58,177 | 98% | 70 ms | 8.3 ms | $52.00 | $0.248 |

2,000 requests at 200 a second (seed 0), every machine running Chapter 18's scheduler at a 512-token budget over 30,515 blocks. Throughput is measured over the arrival window with the first 50% discarded. A fleet is "behind" when it delivers less than 95% of the tokens the traffic asks for, or misses the 1,000 ms first-token promise. The dollar figures are $3.25 an hour a machine, on-demand (FACTS.md).
