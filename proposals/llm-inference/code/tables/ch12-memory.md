| | `tinyserve` (this chapter) | Llama-3-style 8B |
|---|---|---|
| Layers | 4 | 32 |
| KV heads | 4 | 8 |
| Head dimension | 32 | 128 |
| Bytes per element | 4 (fp32) | 2 (bf16) |
| **KV cache per token** | **4,096 B** | **128 KiB** |
| Per 8,192-token sequence | 32 MiB | 1.0 GiB |
