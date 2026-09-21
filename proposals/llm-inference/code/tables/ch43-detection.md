| Signal | Healthy | With the fault | Moved by |
|---|---|---|---|
| `vllm:num_requests_waiting` | 1.00 | 918.00 | 918.0x |
| `vllm:num_requests_running` | 64.00 | 53.00 | 0.8x |
| `vllm:time_to_first_token_seconds` | 0.21 | 36.24 | 173.9x |
| `vllm:num_preemptions` | 0 | 1,653 | -- |
| first-token p99 (ms) | 436 | 43,128 | 98.8x |
| between-tokens p99 (ms) | 9.9 | 10.2 | 1.0x |

The same traffic at 26 requests a second, once with the block pool Chapter 41 sized (30,515 blocks) and once with it cut to 6% of that (3.8 GB), which is what a leak, a noisy neighbour or a bad rollout looks like from inside the server. Read the second row twice: the number of requests *running* goes **down**. A dashboard showing it looks calmer during the incident than before it.
