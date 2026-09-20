| Block size | Sequences admitted | Memory in use | Wasted per sequence | Worst case |
|---|---|---|---|---|
| 1 | 328 | 100.0% | 0.0 tokens | 0 tokens |
| 4 | 328 | 99.9% | 1.5 tokens | 3 tokens |
| 8 | 328 | 99.8% | 3.5 tokens | 7 tokens |
| 16 ← | 327 | 99.5% | 7.7 tokens | 15 tokens |
| 32 | 325 | 99.0% | 14.9 tokens | 31 tokens |
| 64 | 321 | 98.0% | 30.6 tokens | 63 tokens |
| 128 | 314 | 96.0% | 62.4 tokens | 127 tokens |

← the size production engines default to. Waste per sequence averages about half a block and can never exceed one block less one token.
