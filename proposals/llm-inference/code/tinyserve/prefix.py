"""Blocks shared between sequences that begin with the same tokens.

Chapter 14 made every block reachable through a table, which means two
sequences can name the same physical block. Nothing in Chapter 14 used
that. This does.

Three pieces, and each one is small:

* `block_hash` names a block by its tokens *and* everything before it,
  so the name means "this prefix", not "these tokens somewhere".
* `SharedBlockPool` counts how many holders each block has, so a block
  returns to the free list when the last of them lets go.
* `PrefixTree` remembers which block holds which prefix, matches an
  arriving prompt against what it already has, and evicts when the pool
  runs short.

Only *full* blocks are ever shared. A partly filled block still belongs
to one sequence, which is what makes copy-on-write unnecessary here.

References: Zheng et al., "SGLang: Efficient Execution of Structured
Language Model Programs", NeurIPS 2024 (RadixAttention); and vLLM's
automatic prefix caching, whose chained block hash this follows.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .model import Config
from .paged import BLOCK_SIZE, BlockPool, PagedKVCache

ROOT_HASH = 0


def block_hash(parent: int, tokens: tuple[int, ...]) -> int:
    """A name for "these tokens, in this position, after exactly this prefix".

    The parent's hash is folded in, so the same tokens appearing after a
    different prefix get a different name. Without that, a block would
    be reusable in a context it was never computed in, and the answer
    would be wrong.

    Production engines hash cryptographically (vLLM defaults to SHA-256
    as of v0.11). A collision here is not a slow lookup, it is one
    request reading another request's context.
    """
    return hash((parent, tokens))


class SharedBlockPool(BlockPool):
    """A pool whose blocks may have more than one holder.

    `take` hands out a block with one holder. `incref` adds one.
    `give_back` -- which is what `PagedKVCache.release` already calls --
    removes one, and the block returns to the free list only when the
    count reaches zero.
    """

    def __init__(self, cfg: Config, n_blocks: int,
                 block_size: int = BLOCK_SIZE) -> None:
        super().__init__(cfg, n_blocks, block_size)
        self.refs = [0] * n_blocks

    def take(self) -> int:
        block = super().take()
        self.refs[block] = 1
        return block

    def incref(self, blocks: list[int]) -> None:
        for b in blocks:
            if self.refs[b] < 1:
                raise RuntimeError(f"block {b} is not held by anyone")
            self.refs[b] += 1

    def give_back(self, blocks: list[int]) -> None:
        for b in blocks:
            self.refs[b] -= 1
            if self.refs[b] == 0:
                self._free.append(b)
            elif self.refs[b] < 0:
                raise RuntimeError(f"block {b} was given back too many times")

    def held_by_one(self, block: int) -> bool:
        return self.refs[block] == 1


class SharedKVCache(PagedKVCache):
    """A sequence that may start life holding someone else's blocks.

    Everything else is Chapter 14's cache unchanged: `forward` still
    cannot tell.
    """

    def __init__(self, pool: SharedBlockPool) -> None:
        super().__init__(pool)
        self.shared_tokens = 0     # how much of this sequence is not ours

    def adopt(self, blocks: list[int], n_tokens: int) -> None:
        """Take the cache's word for the first `n_tokens` tokens.

        No keys or values are copied and nothing is computed. The
        sequence simply starts at position `n_tokens` with a block table
        that already points at the right storage.
        """
        if n_tokens % self.pool.block_size:
            raise ValueError("a shared prefix must end on a block boundary")
        self.pool.incref(blocks)
        self.blocks = list(blocks)
        self.length = self.shared_tokens = n_tokens

    def append(self, layer: int, k: np.ndarray, v: np.ndarray,
               start: int) -> tuple[np.ndarray, np.ndarray]:
        # The invariant that removes copy-on-write, made executable:
        # a sequence never writes into a block it adopted.
        if start < self.shared_tokens:
            raise RuntimeError(
                f"write at position {start} would land in an adopted block "
                f"(the first {self.shared_tokens} tokens are shared)")
        return super().append(layer, k, v, start)


@dataclass
class Node:
    """One cached block, and what comes after it."""

    block: int | None = None                       # None only at the root
    tokens: tuple[int, ...] = ()
    parent: "Node | None" = None
    children: dict[int, "Node"] = field(default_factory=dict)
    last_used: int = 0
    hits: int = 0


class PrefixTree:
    """Which block holds which prefix, and which to throw away first.

    The structure is a tree of blocks: the path from the root to a node
    spells out a prefix, one block per edge, and the node holds the
    block with that prefix's keys and values. A prompt is matched by
    walking down from the root as far as the tree goes.

    SGLang's RadixAttention is this tree with two refinements: chains of
    single-child nodes are collapsed into one edge, and matching is done
    at token rather than block granularity. Neither changes which
    prefixes can be reused; `run_ch15.py` measures what the second one
    costs.

    Three eviction policies, because the difference between them is the
    point of having a tree at all:

    * `lru` -- the least recently used *leaf*. A prefix is never thrown
      away while something cached sits behind it.
    * `lfu` -- the least often used leaf.
    * `unstructured` -- the least recently used block, leaf or not. This
      is what an index that does not know its own shape does, and
      Chapter 15 measures what it costs.
    """

    def __init__(self, pool: SharedBlockPool, policy: str = "lru") -> None:
        if policy not in ("lru", "lfu", "unstructured"):
            raise ValueError(f"unknown eviction policy {policy!r}")
        self.pool, self.policy = pool, policy
        self.leaves_only = policy != "unstructured"
        self.root = Node()
        self.clock = 0
        self.evictions = 0
        self.cached_blocks = 0
        # Eviction candidates, in least-recently-used order. Under the
        # structured policies this holds leaves only, which is why a
        # prefix outlives everything cached behind it.
        self._candidates: "OrderedDict[int, Node]" = OrderedDict()

    # --- matching -------------------------------------------------------

    def _blocks_of(self, tokens: list[int]) -> list[tuple[int, ...]]:
        """The prompt cut into whole blocks. A trailing part-block is not
        cacheable and is dropped: it is still being written to."""
        bs = self.pool.block_size
        return [tuple(tokens[i:i + bs]) for i in range(0, len(tokens) - bs + 1, bs)]

    def _touch(self, node: Node) -> None:
        node.last_used, node.hits = self.clock, node.hits + 1
        if id(node) in self._candidates:
            self._candidates.move_to_end(id(node))

    def match(self, tokens: list[int]) -> list[int]:
        """The longest cached prefix of `tokens`, as blocks to adopt."""
        self.clock += 1
        node, parent_hash, found = self.root, ROOT_HASH, []
        for chunk in self._blocks_of(tokens):
            h = block_hash(parent_hash, chunk)
            child = node.children.get(h)
            if child is None:
                break
            self._touch(child)
            node, parent_hash = child, h
            found.append(child.block)
        return found

    def insert(self, tokens: list[int], blocks: list[int]) -> int:
        """Offer this sequence's full blocks to the cache.

        Blocks already known are left alone -- the sequence's copy is
        redundant, and its own reference is dropped when it finishes.
        New ones gain a reference held by the tree, which is what keeps
        them alive after the sequence is gone.
        """
        self.clock += 1
        node, parent_hash, added = self.root, ROOT_HASH, 0
        for i, chunk in enumerate(self._blocks_of(tokens)):
            if i >= len(blocks):
                break
            h = block_hash(parent_hash, chunk)
            child = node.children.get(h)
            if child is None:
                child = Node(block=blocks[i], tokens=chunk, parent=node,
                             last_used=self.clock)
                self.pool.incref([blocks[i]])
                node.children[h] = child
                self.cached_blocks += 1
                added += 1
                if self.leaves_only and node is not self.root:
                    self._candidates.pop(id(node), None)   # no longer a leaf
                self._candidates[id(child)] = child
            else:
                self._touch(child)
            node, parent_hash = child, h
        return added

    # --- eviction -------------------------------------------------------

    def _victim(self) -> Node | None:
        """The next block to throw away, or None if nothing may be.

        A block a live sequence is still holding is never a candidate,
        however old it is.

        Under the structured policies the candidates are the leaves --
        one per distinct conversation, so a few hundred at most -- and
        the oldest is found by looking at all of them. Under the
        unstructured policy every cached block is a candidate, far too
        many to scan, so they are kept in a least-recently-used queue
        and taken from the front.
        """
        if self.leaves_only:
            free = [n for n in self._candidates.values()
                    if self.pool.held_by_one(n.block)]
            if not free:
                return None
            key = ((lambda n: (n.hits, n.last_used)) if self.policy == "lfu"
                   else (lambda n: n.last_used))
            return min(free, key=key)
        for node in self._candidates.values():          # least recent first
            if self.pool.held_by_one(node.block):
                return node
        return None

    def evict(self, n_blocks: int) -> int:
        """Drop cached blocks until `n_blocks` are free, or until nothing
        is left that may be dropped. Returns how many went."""
        dropped = 0
        while self.pool.blocks_free < n_blocks:
            victim = self._victim()
            if victim is None:
                break
            self._drop(victim)
            dropped += 1
        self.evictions += dropped
        return dropped

    def _drop(self, node: Node) -> None:
        parent = node.parent
        for h, child in parent.children.items():
            if child is node:
                del parent.children[h]
                break
        self._candidates.pop(id(node), None)
        self.cached_blocks -= 1
        self.pool.give_back([node.block])
        if self.leaves_only and not parent.children and parent is not self.root:
            self._candidates[id(parent)] = parent      # a leaf again
        # Under the structured policies `node` was a leaf and nothing is
        # left behind. Under the unstructured one its children are still
        # in the pool and no longer reachable from the root: memory held
        # that can never be matched again. `stranded_blocks` counts them.

    # --- what is in there -----------------------------------------------

    @property
    def reachable_blocks(self) -> int:
        seen, stack = 0, [self.root]
        while stack:
            node = stack.pop()
            seen += 1
            stack.extend(node.children.values())
        return seen - 1                       # the root holds no block

    @property
    def stranded_blocks(self) -> int:
        """Cached blocks no prompt can ever reach, because an ancestor was
        evicted out from under them."""
        return self.cached_blocks - self.reachable_blocks


def test_chained_hash_distinguishes_context() -> None:
    """The same tokens after a different prefix must get a different name."""
    a = block_hash(ROOT_HASH, (7, 8, 9))
    b = block_hash(block_hash(ROOT_HASH, (1, 2, 3)), (7, 8, 9))
    assert a != b, "a block hash that ignores its prefix would be unsafe"


def test_shared_blocks_outlive_their_sequence() -> None:
    """A finished sequence's cached blocks stay allocated; its private
    ones come back."""
    pool = SharedBlockPool(Config(), n_blocks=8)
    seq = SharedKVCache(pool)
    seq._grow_to(4 * pool.block_size)
    tree = PrefixTree(pool)
    tree.insert(list(range(2 * pool.block_size)), seq.blocks[:2])
    seq.release()
    assert pool.blocks_in_use == 2, pool.blocks_in_use
    assert tree.cached_blocks == 2


def test_structured_eviction_never_strands() -> None:
    """Evicting leaves first leaves every surviving block reachable;
    evicting by age alone does not."""
    bs = BLOCK_SIZE
    tokens = list(range(6 * bs))
    stranded = {}
    for policy in ("lru", "unstructured"):
        pool = SharedBlockPool(Config(), n_blocks=16)
        tree = PrefixTree(pool, policy=policy)
        seq = SharedKVCache(pool)
        seq._grow_to(len(tokens))
        tree.insert(tokens, seq.blocks)
        seq.release()
        tree.match(tokens[:2 * bs])        # make the first two blocks recent
        tree.evict(pool.blocks_free + 1)   # force exactly one eviction
        stranded[policy] = tree.stranded_blocks
    assert stranded["lru"] == 0, stranded
    assert stranded["unstructured"] > 0, stranded
