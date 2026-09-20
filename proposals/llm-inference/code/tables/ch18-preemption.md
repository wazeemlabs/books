| Way out of a full pool | Tokens/s | Sequences in flight | Preemptions | Prompt tokens read | Copied | Time copying | TTFT p99 |
|---|---|---|---|---|---|---|---|
| recompute | **2,738** | 36.6 | 1,163 | 3.04x over | 0 GB | 0.00 s | 11.7 s |
| swap over PCIe 4.0 x16 | **2,139** | 14.3 | 357 | 1.00x over | 52 GB | 3.28 s | 26.0 s |
| swap over PCIe 5.0 x16 | **2,182** | 14.3 | 357 | 1.00x over | 52 GB | 1.64 s | 24.4 s |
| swap over NVLink (H100) | **2,223** | 14.3 | 357 | 1.00x over | 52 GB | 0.12 s | 22.9 s |

A 1.9 GB pool (3% of the accelerator's free memory) at 12 requests a second, small enough that the server has to take sequences back out of the batch. Recomputing throws the evicted cache away; swapping copies it to host memory and back across the named link.
