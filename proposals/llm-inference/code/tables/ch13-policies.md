| Allocation policy | Held per sequence | In use | Wasted | Concurrent sequences |
|---|---|---|---|---|
| Reserve the full context (8,192) | 8,192 tok (1,024 MiB) | 16% | **84%** | **59** |
| Reserve prompt + cap (prompt + 1,024) | 2,224 tok (278 MiB) | 61% | **39%** | **219** |
| Pages of 16 tokens (Chapter 14) | 1,502 tok (188 MiB) | 90% | **10%** | **325** |
| Perfect foresight (not achievable) | 1,494 tok (187 MiB) | 90% | **10%** | **326** |
