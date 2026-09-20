"""Requests joining and leaving the batch at every step.

Chapter 16 measured what a *fixed* batch wastes: every prompt padded to
the longest, every finished sequence holding its slot, and every new
arrival waiting for the whole batch to drain. All three come from the
same decision -- that a batch is chosen once and kept -- and all three
go away when the batch is chosen again at every iteration.

That is Orca's iteration-level scheduling, and it is what every serving
engine does now. The loop is short enough to read in one sitting:

    while there is anything to do:
        if something is waiting and there is room, prefill it
        otherwise, advance every running sequence by one token
        drop the ones that finished

Everything difficult is in "is there room" and in what the choice costs
the sequences already running.

Nothing here runs a model. The two costs -- a prefill of P tokens and a
decode step over B sequences -- come from `serving.py`, which is
arithmetic over the reference model and published hardware specs. What
is being measured is the *scheduler*: which requests are in flight at
each instant, what they wait for, and what the machine is doing with
its time.

Reference: Yu, Jeong, Kim, Kim and Chun, "Orca: A Distributed Serving
System for Transformer-Based Generative Models", OSDI 2022.

Written for `LLM Inference from the Ground Up`, Chapter 17.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .paged import BLOCK_SIZE
from .serving import decode_step, prefill_step


@dataclass
class Request:
    """One arrival, and what became of it."""

    id: int
    arrival_s: float
    prompt_tokens: int
    output_tokens: int

    generated: int = 0
    first_token_s: float | None = None
    finish_s: float | None = None
    prefills: int = 0            # more than one means it was preempted
    waiting_since: float = 0.0
    gap_s: float = 0.0           # time accumulating since its last token

    @property
    def context(self) -> int:
        return self.prompt_tokens + self.generated

    @property
    def done(self) -> bool:
        return self.generated >= self.output_tokens

    def blocks(self, block_size: int = BLOCK_SIZE) -> int:
        return math.ceil(max(self.context, 1) / block_size)


@dataclass
class Trace:
    """What the server did, in enough detail to argue about."""

    requests: list[Request]
    makespan_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    idle_s: float = 0.0
    iterations: int = 0
    prefill_iterations: int = 0
    preemptions: int = 0
    recomputed_tokens: int = 0   # generated once, thrown away, generated again
    peak_blocks: int = 0
    batch_steps: int = 0          # sum of batch sizes over decode iterations
    slots: int = 0                # sum of slots held over decode iterations
    gaps_ms: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def output_tokens(self) -> int:
        return sum(r.generated for r in self.requests if r.finish_s is not None)

    @property
    def tokens_per_s(self) -> float:
        return self.output_tokens / self.makespan_s if self.makespan_s else 0.0

    @property
    def mean_batch(self) -> float:
        return self.batch_steps / max(self.iterations - self.prefill_iterations, 1)

    @property
    def slot_utilization(self) -> float:
        """Of the slots the server held open, what share held live work."""
        return self.batch_steps / self.slots if self.slots else 0.0


class Pool:
    """Just the block count from Chapter 14: capacity, without the storage."""

    def __init__(self, blocks: int, block_size: int = BLOCK_SIZE) -> None:
        self.total, self.block_size, self.used = blocks, block_size, 0
        # vLLM calls this the watermark. Without it, the blocks an
        # eviction frees are exactly enough to re-admit the sequence
        # that was evicted, and the server evicts and re-admits the
        # same request forever instead of finishing anyone.
        self.watermark = max(1, blocks // 100)

    @property
    def free(self) -> int:
        return self.total - self.used

    def room_for(self, tokens: int) -> bool:
        """Enough for this prompt, plus the margin that stops thrashing."""
        return self.free >= math.ceil(tokens / self.block_size) + self.watermark


def _advance(trace: Trace, running: list[Request], seconds: float) -> None:
    """Time passes. Everyone running waits through it, token or no token."""
    for r in running:
        r.gap_s += seconds


def _emit(trace: Trace, r: Request, clock: float) -> None:
    """One token for this request: its wait since the last one is over."""
    r.generated += 1
    if r.first_token_s is None:
        r.first_token_s = clock
    else:
        trace.gaps_ms.append(r.gap_s * 1e3)
    r.gap_s = 0.0
    if r.done:
        r.finish_s = clock


def serve_continuous(requests: list[Request], max_batch: int, blocks: int,
                     block_size: int = BLOCK_SIZE) -> Trace:
    """Iteration-level scheduling: decide the batch again at every step."""
    waiting = sorted(requests, key=lambda r: r.arrival_s)
    queue: list[Request] = []
    running: list[Request] = []
    pool = Pool(blocks, block_size)
    trace = Trace(requests=requests)
    clock, i = 0.0, 0

    while i < len(waiting) or queue or running:
        while i < len(waiting) and waiting[i].arrival_s <= clock:
            queue.append(waiting[i]); i += 1

        if not queue and not running:                 # nothing to do but wait
            nxt = waiting[i].arrival_s
            trace.idle_s += nxt - clock
            clock = nxt
            continue

        # Admit one waiting request if there is a slot and the memory.
        head = queue[0] if queue else None
        # The watermark is a margin, not a wall: if nothing at all is
        # running there is nobody to wait for, so the only question is
        # whether the pool can hold this one sequence.
        fits = head is not None and (pool.room_for(head.prompt_tokens) or
                                     (not running and pool.free >=
                                      math.ceil(head.prompt_tokens / block_size)))
        if head is not None and fits and len(running) < max_batch:
            queue.pop(0)
            seconds = prefill_step(head.prompt_tokens).seconds
            _advance(trace, running, seconds)
            clock += seconds
            head.prefills += 1
            before = math.ceil(head.prompt_tokens / block_size)
            pool.used += before
            _emit(trace, head, clock)
            pool.used += head.blocks(block_size) - before
            trace.peak_blocks = max(trace.peak_blocks, pool.used)
            running.append(head)
            trace.prefill_s += seconds
            trace.iterations += 1
            trace.prefill_iterations += 1
            continue

        if not running:            # a request is waiting but will not fit
            raise RuntimeError(
                f"the pool holds {pool.total} blocks, and request "
                f"{queue[0].id} needs "
                f"{math.ceil(queue[0].prompt_tokens / block_size)} for its "
                "prompt alone: the server would have to refuse it")

        # Otherwise: one token for everyone running.
        context = sum(r.context for r in running) // len(running)
        seconds = decode_step(len(running), context).seconds
        _advance(trace, running, seconds)
        clock += seconds
        trace.decode_s += seconds
        trace.iterations += 1
        trace.batch_steps += len(running)
        trace.slots += len(running)        # nothing is held by a finished sequence
        trace.batch_sizes.append(len(running))

        needed = 0
        for r in running:
            before = r.blocks(block_size)
            _emit(trace, r, clock)
            needed += r.blocks(block_size) - before
        pool.used += needed
        trace.peak_blocks = max(trace.peak_blocks, pool.used)
        for r in list(running):
            if r.done:
                pool.used -= r.blocks(block_size)
                running.remove(r)
        if pool.free < 0:
            trace.preemptions += _make_room(trace, pool, running, queue, block_size)

    trace.makespan_s = clock
    return trace


def _make_room(trace: Trace, pool: Pool, running: list[Request],
               queue: list[Request], block_size: int) -> int:
    """The pool is full. Evict the newest sequence and start it again later.

    Recomputing is the cheaper of the two ways out -- the other is
    copying its cache to host memory and back -- and which to prefer is
    a scheduling-policy question, taken up with chunked prefill.
    Newest-first is the standard choice: the sequence that has invested
    least loses least.
    """
    evicted = 0
    while pool.free < 0 and running:
        victim = running.pop()
        pool.used -= victim.blocks(block_size)
        trace.recomputed_tokens += victim.generated
        victim.generated = 0
        victim.first_token_s = None
        victim.gap_s = 0.0
        queue.insert(0, victim)
        evicted += 1
    return evicted


def serve_static(requests: list[Request], max_batch: int, blocks: int,
                 block_size: int = BLOCK_SIZE, timeout_s: float = 1.0) -> Trace:
    """The Chapter 16 baseline: fill a batch, run it to the end, repeat.

    The batch is formed when `max_batch` requests have arrived or when
    `timeout_s` has passed since the first of them did, whichever comes
    first -- which is what a static server does, and is the kindest
    version of it. Waiting for a full batch would be worse.

    `blocks` is recorded against rather than enforced: a static batch is
    given all the memory it asks for, and it is given the same paged
    allocator as `serve_continuous`, so the comparison is generous to
    it on both counts. `Trace.peak_blocks` says how much it took.
    """
    waiting = sorted(requests, key=lambda r: r.arrival_s)
    trace = Trace(requests=requests)
    clock, i = 0.0, 0

    while i < len(waiting):
        first = waiting[i]
        deadline = max(first.arrival_s, clock) + timeout_s
        batch = [first]
        j = i + 1
        while (j < len(waiting) and len(batch) < max_batch
               and waiting[j].arrival_s <= deadline):
            batch.append(waiting[j]); j += 1
        i = j
        formed = (batch[-1].arrival_s if len(batch) == max_batch
                  else max(deadline, batch[-1].arrival_s))
        if formed > clock:
            trace.idle_s += formed - clock
            clock = formed

        running = list(batch)
        # One padded prefill for the whole batch: every prompt is
        # processed as though it were as long as the longest, and every
        # first token appears when the last of them is done.
        padded = max(x.prompt_tokens for x in batch)
        seconds = prefill_step(padded).seconds * len(batch)
        _advance(trace, running, seconds)
        clock += seconds
        for r in running:
            r.prefills += 1
            _emit(trace, r, clock)
        trace.prefill_s += seconds
        trace.iterations += 1
        trace.prefill_iterations += 1

        # Every sequence keeps its slot until the longest one finishes.
        steps = max(r.output_tokens for r in batch) - 1
        for _ in range(steps):
            context = sum(r.context for r in running) // len(running)
            seconds = decode_step(len(running), context).seconds
            _advance(trace, running, seconds)
            clock += seconds
            trace.decode_s += seconds
            trace.iterations += 1
            trace.batch_steps += sum(1 for r in running if not r.done)
            trace.slots += len(running)
            trace.batch_sizes.append(sum(1 for r in running if not r.done))
            for r in running:
                if not r.done:
                    _emit(trace, r, clock)

        # Nothing is released until the whole batch ends, so the batch
        # holds its high-water mark at the moment the longest finishes.
        trace.peak_blocks = max(trace.peak_blocks,
                                sum(r.blocks(block_size) for r in batch))

    trace.makespan_s = clock
    return trace


def test_every_request_is_served_exactly_once() -> None:
    """Whatever the policy, every request gets all of its tokens."""
    reqs = [Request(id=n, arrival_s=n * 0.01, prompt_tokens=100 + 10 * n,
                    output_tokens=5 + n) for n in range(12)]
    for serve in (serve_continuous, serve_static):
        rs = [Request(**{k: getattr(r, k) for k in
                         ("id", "arrival_s", "prompt_tokens", "output_tokens")})
              for r in reqs]
        trace = serve(rs, max_batch=4, blocks=100_000)
        assert all(r.finish_s is not None for r in rs), serve.__name__
        assert all(r.generated == r.output_tokens for r in rs), serve.__name__
        assert all(r.first_token_s >= r.arrival_s for r in rs), serve.__name__
        assert trace.makespan_s > 0


def test_continuous_finishes_short_requests_first() -> None:
    """A short request behind a long one should not wait for it."""
    reqs = [Request(id=0, arrival_s=0.0, prompt_tokens=200, output_tokens=400),
            Request(id=1, arrival_s=0.001, prompt_tokens=200, output_tokens=5)]
    serve_continuous(reqs, max_batch=8, blocks=100_000)
    assert reqs[1].finish_s < reqs[0].finish_s


def test_a_small_pool_preempts_and_still_finishes_everyone() -> None:
    """When the pool runs out the server evicts, and nobody is lost.

    Eviction throws away tokens that were already generated, so the
    work is done twice -- but every request still ends with exactly the
    output it asked for, which is the property that matters.
    """
    reqs = [Request(id=n, arrival_s=n * 0.001, prompt_tokens=400,
                    output_tokens=120) for n in range(8)]
    trace = serve_continuous(reqs, max_batch=8, blocks=60)
    assert trace.preemptions > 0, "a 60-block pool should not hold this trace"
    assert trace.recomputed_tokens > 0
    assert all(r.generated == r.output_tokens for r in reqs)
    assert all(r.finish_s is not None for r in reqs)
