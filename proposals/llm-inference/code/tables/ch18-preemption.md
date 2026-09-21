| Way out of a full pool | Tokens/s | Sequences in flight | Preemptions | Prompt tokens read | Copied | Time copying | TTFT p99 |
|---|---|---|---|---|---|---|---|
| recompute | **3,435** | 22.1 | 1,601 | 1.36x over | 0 GB | 0.00 s | 10.4 s |
| swap over PCIe 4.0 x16 | **3,435** | 22.5 | 1,532 | 1.00x over | 245 GB | 15.30 s | 35.4 s |
| swap over PCIe 5.0 x16 | **3,435** | 22.0 | 1,442 | 1.00x over | 230 GB | 7.18 s | 32.0 s |
| swap over NVLink (H100) | **3,435** | 21.5 | 1,361 | 1.00x over | 216 GB | 0.48 s | 28.9 s |

A 1.9 GB pool (3% of the accelerator's free memory) at 12 requests a second, small enough that the server has to take sequences back out of the batch. Recomputing throws the evicted cache away; swapping copies it to host memory and back across the named link.
