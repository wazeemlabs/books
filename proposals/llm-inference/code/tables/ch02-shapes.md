| Stage | What it is | Shape |
|---|---|---|
| The prompt, as text | "the cat sat on the" | `-` |
| As token numbers | [2, 3, 4, 5, 2] | `5` |
| After the lookup table | a vector per token | `5 x 128` |
| Through each of the 4 layers | same shape in, same shape out | `5 x 128` |
| Keys and values kept per layer | what attention reads later | `4 x 5 x 32` |
| Scores for the next token | one per word in the vocabulary | `5 x 48` |
| Used to write a token | only the last row matters | `48` |
