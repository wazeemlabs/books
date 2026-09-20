| Tokens generated | Without a cache | With a cache | Measured speedup | Speedup predicted by FLOPs |
|---|---|---|---|---|
| 16 | 0.384 s | 0.032 s | **12x** | 15x |
| 32 | 0.819 s | 0.041 s | **20x** | 29x |
| 64 | 1.745 s \* | 0.057 s | **30x** | 55x |
| 128 | 4.190 s \* | 0.090 s | **46x** | 100x |
| 256 | 11.242 s \* | 0.164 s | **69x** | 181x |

\* run-to-run spread exceeded 5%; see the note on the measuring machine.
