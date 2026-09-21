| Batch | Experts touched | The step reads | Per token | Between tokens, one machine | Over 17 | Network share |
|---|---|---|---|---|---|---|
| 1 | 8 of 256 | 37B | 37.00B | 22 ms | **10.6 ms** | 0% |
| 4 | 31 of 256 | 95B | 23.65B | 57 ms | **12.7 ms** | 0% |
| 8 | 57 of 256 | 163B | 20.42B | 98 ms | **15.2 ms** | 1% |
| 16 | 102 of 256 | 277B | 17.33B | 166 ms | **19.3 ms** | 1% |
| 32 | 163 of 256 | 434B | 13.56B | 261 ms | **25.1 ms** | 2% |
| 64 | 222 of 256 | 585B | 9.14B | 353 ms | **31.1 ms** | 3% |
| 128 | 252 of 256 | 660B | 5.15B | 401 ms | **34.9 ms** | 6% |
| 256 | 256 of 256 | 671B | 2.62B | 416 ms | **37.7 ms** | 11% |
| 512 | 256 of 256 | 671B | 1.31B | 431 ms | **42.6 ms** | 19% |

DeepSeek-V3: 671B total parameters, 37B activated for each token, 256 routed experts with 8 chosen per token (FACTS.md). Each sequence routes independently, so a step reads the union of what the batch chose, and the union fills up. The fifth column is one accelerator, which is hypothetical -- the weights alone are 1,342 GB and need 17 of them. The sixth is the same step with the experts spread across those 17, which is how it is actually served, over NVLink.
