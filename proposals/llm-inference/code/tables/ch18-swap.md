| Context | Recompute it | Copy it over PCIe 4.0 x16 | Copy it over PCIe 5.0 x16 | Copy it over NVLink (H100) | Link speed at which they tie |
|---|---|---|---|---|---|
| 256 tokens | 4.8 ms | 2.1 ms | 1.0 ms | 0.1 ms | **14 GB/s** |
| 512 tokens | 7.9 ms | 4.2 ms | 2.1 ms | 0.1 ms | **17 GB/s** |
| 1,024 tokens | 16.0 ms | 8.4 ms | 4.2 ms | 0.3 ms | **17 GB/s** |
| 1,500 tokens | 23.8 ms | 12.3 ms | 6.1 ms | 0.4 ms | **16 GB/s** |
| 2,048 tokens | 33.2 ms | 16.8 ms | 8.4 ms | 0.6 ms | **16 GB/s** |
| 4,096 tokens | 70.8 ms | 33.6 ms | 16.8 ms | 1.2 ms | **15 GB/s** |
| 8,192 tokens | 159.3 ms | 67.1 ms | 33.6 ms | 2.4 ms | **13 GB/s** |

One preempted sequence, both ways. Recomputing re-reads its context: arithmetic, growing faster than the length. Copying moves its keys and values out and back: 128 KiB a token each way, growing linearly. The last column is the link speed at which the two cost the same.
