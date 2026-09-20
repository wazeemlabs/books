| Tile | Memory traffic | Blocks computed | Blocks skipped | Fast memory one tile needs | Share of a multiprocessor | Fits |
|---|---|---|---|---|---|---|
| 16 | 617 MB | 8,256 | 8,128 | 21 KB | 9% | yes |
| 32 | 349 MB | 2,080 | 2,016 | 42 KB | 19% | yes |
| 64 | 216 MB | 528 | 496 | 88 KB | 39% | yes |
| **128** | 151 MB | 136 | 120 | 193 KB | 85% | yes |
| 256 | 122 MB | 36 | 28 | 450 KB | 197% | no |

At 2,048 tokens. Traffic is counted on the runnable model; the fast memory is what one tile of the book's 8B model needs in bf16 -- a query tile, a key tile, a value tile, the block of scores between them, the running output and the two running numbers per row -- against the 228 KB a Hopper streaming multiprocessor has (FACTS.md). Traffic falls with every increase in tile size and the memory rises with the square of it, so the tile is as large as will fit: 128. A real kernel also wants room to fetch the next tile while it works on this one, so it has less to spend than this table allows.
