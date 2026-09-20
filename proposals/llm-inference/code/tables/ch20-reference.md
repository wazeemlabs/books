| Prompt | Score matrix, one layer | Against its own inputs | Traffic, built in full | Traffic, tiled | Times less | Times less, block resident | All layers, built in full | All layers, tiled |
|---|---|---|---|---|---|---|---|---|
| 128 | 1 MB | 0.7x | 7 MB | 4 MB | **1.9x** | 2.4x | 0.07 ms | 0.04 ms |
| 512 | 17 MB | 2.7x | 78 MB | 27 MB | **2.8x** | 4.4x | 0.74 ms | 0.26 ms |
| 1,200 * | 92 MB | 6.2x | 393 MB | 118 MB | **3.3x** | 5.7x | 3.76 ms | 1.13 ms |
| 2,048 | 268 MB | 10.7x | 1,116 MB | 310 MB | **3.6x** | 6.5x | 10.66 ms | 2.96 ms |
| 4,096 | 1,074 MB | 21.3x | 4,379 MB | 1,158 MB | **3.8x** | 7.2x | 41.83 ms | 11.06 ms |
| 8,192 | 4,295 MB | 42.7x | 17,348 MB | 4,463 MB | **3.9x** | 7.5x | 165.71 ms | 42.63 ms |

\* the case study's prompt. The book's 8B model: 32 query heads sharing 8 key heads of 128 dimensions, bf16, 32 layers, tiles of 64x64. "Against its own inputs" is the score matrix divided by the queries, keys and values it is built from. "Times less" charges the blocks of scores as traffic, which is what this NumPy implementation makes them; the column after it charges them to the scratchpad instead, which is what a CUDA kernel's tile size is chosen for. The real figure is the second; the chapter quotes the first, so every saving in it is a lower bound. Times are bytes over 3.35 TB/s (FACTS.md): a floor for the memory, not a prediction of a kernel's runtime, which also has arithmetic to do.
