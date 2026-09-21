| In front of the model | Requests a second a machine | Machines for 200 a second | Dollars an hour | First token p99 at capacity |
|---|---|---|---|---|
| no cache | 26 | 8 | $26.00 | 469 ms |
| prefix cache | 40 | 5 | $16.25 | 172 ms |
| response cache | 36 | 6 | $19.50 | 302 ms |

The machine of Chapter 41, measured the same way: every offered load run at 6,000 requests and again at twice that, and the capacity is the highest load whose numbers settled and which kept both promises. The prefix cache is Chapter 15's measured hit -- 85.4% of prompt tokens already computed -- and saves only the reading of them. The response cache hits 30% of *requests* and saves everything about them. Memory is not what binds here: the bare machine peaked at 6,755 of 30,515 blocks.
