| Context | Its cache | Prefill takes | Over NVLink, same node | Over InfiniBand NDR | Over PCIe 5.0 x16 | Over 100 GbE | Over 25 GbE | Link to match the prefill |
|---|---|---|---|---|---|---|---|---|
| 256 tokens | 34 MB | 4.8 ms | 0.0 ms | 0.7 ms | 0.5 ms | 2.7 ms | 10.7 ms | **7.0 GB/s** |
| 512 tokens | 67 MB | 7.9 ms | 0.1 ms | 1.3 ms | 1.0 ms | 5.4 ms | 21.5 ms | **8.5 GB/s** |
| 1,200 tokens | 157 MB | 18.9 ms | 0.2 ms | 3.1 ms | 2.5 ms | 12.6 ms | 50.3 ms | **8.3 GB/s** |
| 1,500 tokens | 197 MB | 23.8 ms | 0.2 ms | 3.9 ms | 3.1 ms | 15.7 ms | 62.9 ms | **8.2 GB/s** |
| 4,096 tokens | 537 MB | 70.8 ms | 0.6 ms | 10.7 ms | 8.4 ms | 42.9 ms | 171.8 ms | **7.6 GB/s** |
| 8,192 tokens | 1,074 MB | 159.3 ms | 1.2 ms | 21.5 ms | 16.8 ms | 85.9 ms | 343.6 ms | **6.7 GB/s** |

The reference model keeps 128 KiB of keys and values per token, so a cache is large and moving it is a bandwidth problem, not a latency one. The last column is the link speed at which the move takes exactly as long as the prefill that produced it -- a link slower than that turns the move into the bottleneck. One decode step, for scale, is 8.5 ms.
