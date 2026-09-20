| Prompt | Score matrix | Traffic, built in full | Traffic, tiled | Times less | Largest thing held | Blocks skipped | Formula |
|---|---|---|---|---|---|---|---|
| 128 | 0.5 MB | 3.1 MB | 1.7 MB | **1.85x** | 0.5 MB -> 131 KB | 25% | agrees |
| 256 | 2.1 MB | 10.5 MB | 5.0 MB | **2.11x** | 2.1 MB -> 131 KB | 38% | agrees |
| 512 | 8.4 MB | 37.7 MB | 16.3 MB | **2.32x** | 8.4 MB -> 131 KB | 44% | agrees |
| 1,024 | 33.6 MB | 142.6 MB | 57.7 MB | **2.47x** | 33.6 MB -> 131 KB | 47% | agrees |
| 2,048 | 134.2 MB | 553.6 MB | 216.0 MB | **2.56x** | 134.2 MB -> 131 KB | 48% | agrees |

One layer of an 8-head model with 64-dimensional heads in float32, tiles of 64x64, seed 0. Every byte is counted as the code moves it, not estimated. The last column checks each count against the closed-form formula the next table applies to a model too large to run here; they must agree exactly, and `make tests` fails if they do not.
