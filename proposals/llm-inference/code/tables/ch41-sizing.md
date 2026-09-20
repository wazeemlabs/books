| How it was sized | Load a machine | Machines | Cost an hour |
|---|---|---|---|
| Throughput alone, ignoring the promise | 31.1 req/s | 7 | $22.75 |
| The 70% rule from classical queueing | 21.8 req/s | 10 | $32.50 |
| **The load at which the promise still holds** | **28.0 req/s** | **8** | **$26.00** |

For 200 requests a second. The promise holds up to 28 requests a machine, which is 90% of what the machine can do -- far past where the classical rule would stop. Sizing on throughput alone meets no promise at all; sizing on the rule of thumb buys machines the measurement says are not needed.
