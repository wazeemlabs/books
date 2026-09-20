"""Keys and values in fixed-size blocks, mapped like virtual memory.

Chapter 13 measured what a contiguous cache costs: a sequence must
reserve its whole possible length up front, and most of what it
reserves is never used. The fix, from vLLM's PagedAttention, is the one
an operating system uses for memory. Storage is cut into fixed-size
**blocks**; a sequence holds a **block table** listing the blocks it
owns; and blocks are handed out only as the sequence actually grows.

A sequence's tokens are then contiguous in its block table and
scattered in physical memory, exactly as a process's pages are.

Reference: Kwon et al., "Efficient Memory Management for Large Language
Model Serving with PagedAttention", SOSP 2023.
"""

from __future__ import annotations

import numpy as np

from .model import Config, DType

BLOCK_SIZE = 16


class OutOfBlocks(RuntimeError):
    """The pool is full. A real server preempts a sequence here; see
    Chapter 18."""


class BlockPool:
    """All the KV storage there is, cut into equal blocks.

    One pool serves every sequence. Nothing is reserved for anybody
    until they ask, and what a finished sequence held goes straight
    back.
    """

    def __init__(self, cfg: Config, n_blocks: int,
                 block_size: int = BLOCK_SIZE) -> None:
        self.cfg, self.n_blocks, self.block_size = cfg, n_blocks, block_size
        shape = (n_blocks, cfg.n_kv_heads, block_size, cfg.head_dim)
        self.k = [np.zeros(shape, dtype=DType) for _ in range(cfg.n_layers)]
        self.v = [np.zeros(shape, dtype=DType) for _ in range(cfg.n_layers)]
        self._free = list(reversed(range(n_blocks)))

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.k) + sum(a.nbytes for a in self.v)

    @property
    def blocks_free(self) -> int:
        return len(self._free)

    @property
    def blocks_in_use(self) -> int:
        return self.n_blocks - len(self._free)

    def take(self) -> int:
        if not self._free:
            raise OutOfBlocks(f"all {self.n_blocks} blocks are in use")
        return self._free.pop()

    def give_back(self, blocks: list[int]) -> None:
        self._free.extend(blocks)


class PagedKVCache:
    """One sequence's view of the pool: a block table and a length.

    Implements the same `append` as the contiguous cache in
    `model.py`, so `forward` cannot tell the difference.
    """

    def __init__(self, pool: BlockPool) -> None:
        self.pool = pool
        self.blocks: list[int] = []
        self.length = 0

    @property
    def reserved_tokens(self) -> int:
        return len(self.blocks) * self.pool.block_size

    @property
    def wasted_tokens(self) -> int:
        """Room held in the last block that this sequence is not using.

        This is the whole of the waste under paging, and it can never
        exceed one block less one token.
        """
        return self.reserved_tokens - self.length

    def _grow_to(self, tokens: int) -> None:
        while self.reserved_tokens < tokens:
            self.blocks.append(self.pool.take())

    def release(self) -> None:
        self.pool.give_back(self.blocks)
        self.blocks, self.length = [], 0

    def append(self, layer: int, k: np.ndarray, v: np.ndarray,
               start: int) -> tuple[np.ndarray, np.ndarray]:
        n_new = k.shape[1]
        total = start + n_new
        if layer == 0:                 # grow once per token, not once per layer
            self._grow_to(total)

        bs = self.pool.block_size
        for i in range(n_new):
            pos = start + i
            block, offset = self.blocks[pos // bs], pos % bs
            self.pool.k[layer][block, :, offset] = k[:, i]
            self.pool.v[layer][block, :, offset] = v[:, i]

        # Gather the sequence's blocks back into logical order. A real
        # implementation fuses this into the attention kernel and never
        # materializes it; Chapter 20 is where that happens.
        used = self.blocks[: (total + bs - 1) // bs]
        kk = self.pool.k[layer][used].transpose(1, 0, 2, 3).reshape(
            self.pool.cfg.n_kv_heads, -1, self.pool.cfg.head_dim)
        vv = self.pool.v[layer][used].transpose(1, 0, 2, 3).reshape(
            self.pool.cfg.n_kv_heads, -1, self.pool.cfg.head_dim)
        return kk[:, :total], vv[:, :total]
