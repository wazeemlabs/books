| Nesting allowed | States | Distinct masks | States per mask |
|---|---|---|---|
| 1 | 27 | **27** | 1 |
| 2 | 77 | **27** | 3 |
| 3 | 177 | **27** | 7 |
| 4 | 377 | **27** | 14 |
| 5 | 777 | **27** | 29 |
| 6 | 1,577 | **27** | 58 |

Enumerated exhaustively. The states double with every level of nesting, because the stack is a sequence of objects and arrays. The masks do not, because what may come next depends on the innermost open container and never on the ones beneath it -- so all 27 of them fit in 128 bytes over this 38-token vocabulary, and the work per token at serving time is a lookup. Each mask leaves 1 to 26 tokens open, 27% of the vocabulary on average: most of what the model could say, at any moment, it may not.
