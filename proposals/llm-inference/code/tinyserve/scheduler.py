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

BIN_S = 0.1                      # resolution of the emission histogram
from .reference import KV_BYTES_PER_TOKEN
from .serving import decode_step, mixed_step, prefill_step


@dataclass
class Request:
    """One arrival, and what became of it."""

    id: int
    arrival_s: float
    prompt_tokens: int
    output_tokens: int

    generated: int = 0
    prefilled: int = 0           # prompt tokens whose keys and values are held
    first_token_s: float | None = None
    finish_s: float | None = None
    prefills: int = 0            # more than one means it was preempted
    swapped: bool = False        # its cache is in host memory, not the pool
    out_since: float = 0.0       # when it left the batch, if it is swapped out
    waiting_since: float = 0.0
    gap_s: float = 0.0           # time accumulating since its last token
    gaps_ms: list[float] = field(default_factory=list)   # this one's own waits

    @property
    def first_gap_ms(self) -> float | None:
        """The wait between the first token and the second.

        On one machine this is an ordinary decode step. Across two, it
        is where the cache's trip over the network lands, because the
        first token comes out of the prefill worker before the cache
        has gone anywhere.
        """
        return self.gaps_ms[0] if self.gaps_ms else None

    @property
    def context(self) -> int:
        """Tokens this sequence holds keys and values for, right now.

        For a sequence that was admitted in one go this is its whole
        prompt plus what it has generated. For one whose prompt is
        being read a chunk at a time (Chapter 18) it is however much of
        that prompt has been read so far.
        """
        return self.prefilled + self.generated

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
    recomputed_prompt_tokens: int = 0    # and the prompt behind them, re-read
    swapped_bytes: int = 0       # cache copied out to host memory and back
    swap_s: float = 0.0          # time spent doing that copying
    transfer_s: float = 0.0      # time moving caches between machines
    chunk_tokens: int = 0        # prompt tokens read, counting every chunk
    peak_blocks: int = 0
    batch_steps: int = 0          # sum of batch sizes over decode iterations
    slots: int = 0                # sum of slots held over decode iterations
    gaps_ms: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    emitted: dict[int, int] = field(default_factory=dict)   # 0.1 s bins

    def tokens_between(self, start_s: float, end_s: float) -> int:
        """Tokens produced in a window, from the emission bins."""
        a, b = int(start_s / BIN_S), int(end_s / BIN_S)
        return sum(n for k, n in self.emitted.items() if a <= k < b)

    def steady_tokens_per_s(self, start_s: float, end_s: float) -> float:
        """Throughput while the server was actually loaded.

        Measuring over a whole trace divides by its drain tail as well,
        and a tail is long: the last request of a short trace can still
        be generating a thousand tokens after everything else is done.
        A window inside the arrival period has the server at its
        steady state throughout.
        """
        span = end_s - start_s
        return self.tokens_between(start_s, end_s) / span if span > 0 else 0.0

    @property
    def output_tokens(self) -> int:
        return sum(r.generated for r in self.requests if r.finish_s is not None)

    @property
    def tokens_per_s(self) -> float:
        return self.output_tokens / self.makespan_s if self.makespan_s else 0.0

    @property
    def mean_batch(self) -> float:
        """Sequences decoded per iteration that decoded anything.

        The denominator is the number of recorded batch sizes, not
        `iterations - prefill_iterations`. Chapter 17's schedulers give
        prefill an iteration to itself, so the two agree there. Chapter
        18's does not: a chunked iteration prefills *and* decodes, so
        subtracting it removes an iteration whose batch was counted and
        the mean comes out above the largest batch ever run.
        """
        return self.batch_steps / max(len(self.batch_sizes), 1)

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
    k = int(clock / BIN_S)
    trace.emitted[k] = trace.emitted.get(k, 0) + 1
    if r.first_token_s is None:
        r.first_token_s = clock
    else:
        trace.gaps_ms.append(r.gap_s * 1e3)
        r.gaps_ms.append(r.gap_s * 1e3)
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
            head.prefilled = head.prompt_tokens     # read in one iteration
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
        for r in list(running):
            if r.done:
                pool.used -= r.blocks(block_size)
                running.remove(r)
        if pool.free < 0:
            trace.preemptions += _make_room(trace, pool, running, queue, block_size)
        # After eviction, not before: an allocator that overshoots and
        # then evicts never actually held the overshoot.
        trace.peak_blocks = max(trace.peak_blocks, pool.used)

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
        trace.recomputed_prompt_tokens += victim.prefilled
        victim.generated = 0
        victim.prefilled = 0
        victim.first_token_s = None
        victim.gaps_ms.clear()
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
            r.prefilled = r.prompt_tokens
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


# --- Chapter 18: a prompt read a chunk at a time ------------------------

POLICIES = ("fcfs", "shortest-output", "longest-output")


def _order(queue: list[Request], policy: str) -> None:
    """Decide who is at the front. The whole of a scheduling policy."""
    if policy == "fcfs":
        queue.sort(key=lambda r: r.arrival_s)
    elif policy == "shortest-output":            # oracle shortest-job-first
        queue.sort(key=lambda r: (r.output_tokens, r.arrival_s))
    elif policy == "longest-output":             # the same oracle, reversed
        queue.sort(key=lambda r: (-r.output_tokens, r.arrival_s))
    else:
        raise ValueError(f"unknown policy {policy!r}; try one of {POLICIES}")


def _evict(trace: Trace, pool: Pool, decoding: list[Request],
           restart: list[Request], block_size: int,
           swap_bytes_per_s: float | None, clock: float) -> tuple[int, float]:
    """The pool is full: take the newest sequence out of the batch.

    Two ways to do it, and the chapter measures both.

    *Recompute* throws the sequence's cache away. Getting it back means
    reading its prompt again and generating its tokens again, which is
    free of any transfer and costs whatever that work costs.

    *Swap* copies the cache out to host memory and back. It costs no
    arithmetic at all, only two trips across the interconnect -- and
    the interconnect is roughly fifty times slower than the memory the
    cache lives in, which is what makes the choice interesting.
    """
    evicted, seconds = 0, 0.0
    while pool.free < 0 and decoding:
        victim = decoding.pop()
        pool.used -= victim.blocks(block_size)
        if swap_bytes_per_s:
            bytes_moved = victim.context * KV_BYTES_PER_TOKEN
            seconds += bytes_moved / swap_bytes_per_s
            trace.swapped_bytes += bytes_moved
            victim.swapped = True            # its tokens are kept, elsewhere
            victim.out_since = clock         # and its user is still waiting
        else:
            trace.recomputed_tokens += victim.generated
            trace.recomputed_prompt_tokens += victim.prefilled
            victim.generated = 0
            victim.prefilled = 0
            victim.first_token_s = None
            victim.gaps_ms.clear()
        victim.gap_s = 0.0
        restart.insert(0, victim)
        evicted += 1
    return evicted, seconds


def serve_chunked(requests: list[Request], max_batch: int, blocks: int,
                  token_budget: int, block_size: int = BLOCK_SIZE,
                  policy: str = "fcfs",
                  swap_bytes_per_s: float | None = None) -> Trace:
    """Stall-free batching: decodes first, then as much prefill as fits.

    Chapter 17's scheduler ran a whole prompt in one iteration, and
    every sequence already decoding waited through all of it. This one
    gives the iteration a *token budget*. The sequences that are
    decoding take one token each and have first claim on that budget;
    whatever is left is spent reading part of somebody's prompt.

    A prompt too long for the remainder is split and finished over
    several iterations. Nobody's decoding ever stops, which is what
    Sarathi-Serve means by a stall-free schedule, and it is what vLLM
    V1 does by default: it "batches all pending decode requests before
    scheduling any prefill operations", then chunks the prefills that
    do not fit.

    Reference: Agrawal, Kedia, Panwar, Mohan, Kwatra, Gulavani,
    Tumanov and Ramjee, "Taming Throughput-Latency Tradeoff in LLM
    Inference with Sarathi-Serve", OSDI 2024.
    """
    waiting = sorted(requests, key=lambda r: r.arrival_s)
    queue: list[Request] = []       # arrived, never started
    restart: list[Request] = []     # taken out of the batch; go back in first
    prefilling: list[Request] = []  # prompt partly read
    decoding: list[Request] = []    # producing tokens
    pool = Pool(blocks, block_size)
    trace = Trace(requests=requests)
    clock, i = 0.0, 0

    def next_up() -> Request | None:
        if restart:
            return restart[0]
        return queue[0] if queue else None

    while i < len(waiting) or queue or restart or prefilling or decoding:
        arrived = False
        while i < len(waiting) and waiting[i].arrival_s <= clock:
            queue.append(waiting[i]); i += 1; arrived = True
        if arrived:
            _order(queue, policy)

        if not (queue or restart or prefilling or decoding):
            nxt = waiting[i].arrival_s
            trace.idle_s += nxt - clock
            clock = nxt
            continue

        # The decodes have first claim on the iteration.
        budget = max(0, token_budget - len(decoding))

        # Bring in one more prompt if nothing is mid-prompt, there is
        # budget left for it, and the pool can hold what it needs.
        if not prefilling and budget > 0 and len(decoding) < max_batch:
            head = next_up()
            if head is not None:
                need = (head.context if head.swapped
                        else min(budget, head.prompt_tokens))
                if pool.room_for(need) or (not decoding and pool.free >=
                                           math.ceil(need / block_size)):
                    (restart if restart and restart[0] is head else queue).pop(0)
                    if head.swapped:            # its cache comes back whole
                        head.swapped = False
                        # The whole outage counts against this user's
                        # wait for their next token: they were mid-reply.
                        head.gap_s += clock - head.out_since
                        seconds = (head.context * KV_BYTES_PER_TOKEN
                                   / swap_bytes_per_s) if swap_bytes_per_s else 0.0
                        _advance(trace, decoding, seconds)
                        clock += seconds
                        trace.swap_s += seconds
                        pool.used += head.blocks(block_size)
                        trace.peak_blocks = max(trace.peak_blocks, pool.used)
                        decoding.append(head)
                    else:
                        head.prefills += 1
                        prefilling.append(head)

        chunk_req = prefilling[0] if prefilling else None
        chunk = (min(budget, chunk_req.prompt_tokens - chunk_req.prefilled)
                 if chunk_req is not None and budget > 0 else 0)

        if not decoding and chunk == 0:
            raise RuntimeError(
                f"the pool holds {pool.total} blocks and nothing is running: "
                "the server would have to refuse the request at the front")

        context = (sum(r.context for r in decoding) // len(decoding)
                   if decoding else 0)
        cached = chunk_req.prefilled if chunk_req is not None else 0
        seconds = mixed_step(len(decoding), context, chunk, cached).seconds
        _advance(trace, decoding, seconds)
        clock += seconds
        trace.iterations += 1
        trace.decode_s += seconds if chunk == 0 else 0.0
        trace.prefill_s += seconds if chunk else 0.0
        trace.prefill_iterations += 1 if chunk else 0
        trace.chunk_tokens += chunk
        if decoding:
            trace.batch_steps += len(decoding)
            trace.slots += len(decoding) + len(prefilling)
            trace.batch_sizes.append(len(decoding))

        needed = 0
        for r in decoding:
            before = r.blocks(block_size)
            _emit(trace, r, clock)
            needed += r.blocks(block_size) - before
        if chunk:
            before = chunk_req.blocks(block_size)
            chunk_req.prefilled += chunk
            if chunk_req.prefilled >= chunk_req.prompt_tokens:
                _emit(trace, chunk_req, clock)     # the last chunk emits a token
                prefilling.remove(chunk_req)
                decoding.append(chunk_req)
            needed += chunk_req.blocks(block_size) - before
        pool.used += needed

        for r in list(decoding):
            if r.done:
                pool.used -= r.blocks(block_size)
                decoding.remove(r)
        if pool.free < 0:
            n, secs = _evict(trace, pool, decoding, restart, block_size,
                             swap_bytes_per_s, clock)
            trace.preemptions += n
            _advance(trace, decoding, secs)
            clock += secs
            trace.swap_s += secs
        trace.peak_blocks = max(trace.peak_blocks, pool.used)

    trace.makespan_s = clock
    return trace


def test_chunking_reads_every_prompt_token_exactly_once() -> None:
    """Splitting a prompt must not lose or repeat any of it."""
    total = 0
    for budget in (32, 64, 256, 4096):
        reqs = [Request(id=n, arrival_s=n * 0.01, prompt_tokens=300 + 37 * n,
                        output_tokens=6 + n) for n in range(10)]
        trace = serve_chunked(reqs, max_batch=8, blocks=100_000,
                              token_budget=budget)
        total = sum(r.prompt_tokens for r in reqs)
        assert trace.chunk_tokens == total, (budget, trace.chunk_tokens, total)
        assert trace.preemptions == 0
        assert all(r.generated == r.output_tokens for r in reqs)
        assert all(r.prefilled == r.prompt_tokens for r in reqs)


def test_a_smaller_budget_smooths_the_gap_between_tokens() -> None:
    """The chapter's claim, asserted rather than asserted-in-prose.

    A long prompt admitted whole stalls everyone already decoding for
    as long as it takes to read. Split into chunks, it cannot.
    """
    def gaps(budget: int) -> float:
        reqs = [Request(id=0, arrival_s=0.0, prompt_tokens=64, output_tokens=200)]
        reqs += [Request(id=n, arrival_s=0.2 * n, prompt_tokens=8192,
                         output_tokens=20) for n in range(1, 5)]
        trace = serve_chunked(reqs, max_batch=8, blocks=100_000,
                              token_budget=budget)
        return max(trace.gaps_ms)

    assert gaps(512) < gaps(8192), (gaps(512), gaps(8192))


def test_swapping_keeps_the_tokens_that_recomputing_throws_away() -> None:
    """The two ways out of a full pool, told apart by what survives."""
    def run(bw: float | None) -> Trace:
        reqs = [Request(id=n, arrival_s=n * 0.001, prompt_tokens=400,
                        output_tokens=120) for n in range(8)]
        return serve_chunked(reqs, max_batch=8, blocks=60, token_budget=512,
                             swap_bytes_per_s=bw)

    recompute, swap = run(None), run(64e9)      # PCIe 5.0 x16, FACTS.md
    assert recompute.preemptions > 0 and swap.preemptions > 0
    assert recompute.recomputed_tokens > 0 and recompute.swapped_bytes == 0
    assert swap.swapped_bytes > 0 and swap.recomputed_tokens == 0
    assert swap.swap_s > 0 and recompute.swap_s == 0
    assert swap.chunk_tokens < recompute.chunk_tokens, (
        "a swapped sequence should not have to read its prompt again")


# --- Chapter 19: the two phases on different machines ------------------

def _merge(traces: list[Trace], requests: list[Request]) -> Trace:
    """One trace for a fleet, from one trace per worker.

    Wall-clock time is shared, so the fleet finishes when its last
    worker does; everything else adds up.
    """
    out = Trace(requests=requests)
    out.makespan_s = max((t.makespan_s for t in traces), default=0.0)
    for t in traces:
        out.prefill_s += t.prefill_s
        out.decode_s += t.decode_s
        out.idle_s += t.idle_s
        out.transfer_s += t.transfer_s
        out.iterations += t.iterations
        out.prefill_iterations += t.prefill_iterations
        out.preemptions += t.preemptions
        out.recomputed_tokens += t.recomputed_tokens
        out.recomputed_prompt_tokens += t.recomputed_prompt_tokens
        out.swapped_bytes += t.swapped_bytes
        out.swap_s += t.swap_s
        out.chunk_tokens += t.chunk_tokens
        out.batch_steps += t.batch_steps
        out.slots += t.slots
        out.peak_blocks += t.peak_blocks          # summed across workers
        out.gaps_ms += t.gaps_ms
        out.batch_sizes += t.batch_sizes
        for k, n in t.emitted.items():
            out.emitted[k] = out.emitted.get(k, 0) + n
    return out


def serve_colocated(requests: list[Request], workers: int, blocks: int,
                    token_budget: int, max_batch: int,
                    block_size: int = BLOCK_SIZE) -> Trace:
    """A fleet of identical servers, each doing both phases.

    Requests are dealt out to workers in arrival order, which is what a
    round-robin load balancer in front of N replicas does. Each worker
    runs Chapter 18's scheduler over its own share, with its own share
    of the memory. This is the baseline disaggregation has to beat.
    """
    shares: list[list[Request]] = [[] for _ in range(workers)]
    for n, r in enumerate(sorted(requests, key=lambda r: r.arrival_s)):
        shares[n % workers].append(r)
    traces = [serve_chunked(share, max_batch=max_batch, blocks=blocks,
                            token_budget=token_budget, block_size=block_size)
              for share in shares if share]
    return _merge(traces, requests)


def serve_disaggregated(requests: list[Request], prefill_workers: int,
                        decode_workers: int, blocks: int,
                        link_bytes_per_s: float, max_batch: int,
                        block_size: int = BLOCK_SIZE) -> Trace:
    """Prefill on one pool of machines, decode on another.

    The phases want opposite hardware and opposite schedules
    (Chapter 3), and Chapter 18's answer was to interleave them on the
    same machine. This is the other answer: give each phase its own
    machines, and move the keys and values between them.

    The model:

      * A prefill worker takes one prompt at a time. Prefill is already
        compute-bound, so batching buys it little (Chapter 16), and one
        at a time is the clearest thing to count.
      * The prompt's last position produces the first token, which the
        prefill worker returns at once. This is why disaggregation's
        time-to-first-token does not include the transfer.
      * The cache then crosses the link at `link_bytes_per_s`. The
        prefill worker does not wait for it -- NVIDIA's Dynamo
        documentation calls the transfer "non-blocking" -- but the user
        does, and it lands in the gap before their second token.
      * A decode worker runs continuous batching over the sequences it
        has been given and never prefills anything. Its inter-token
        latency is therefore one decode step, always.
      * Each decode worker gets `blocks` of pool. When it runs out it
        preempts, and a preempted sequence goes back to the prefill
        queue, because that is where its cache has to be rebuilt.

    Prefill-side memory is not modelled: a prefill worker holds one
    prompt's cache for as long as the transfer takes, which is small
    beside a decode worker's working set.
    """
    waiting = sorted(requests, key=lambda r: r.arrival_s)
    queue: list[Request] = []
    trace = Trace(requests=requests)

    # Prefill side: when each worker frees up, and what it is holding.
    p_free = [0.0] * prefill_workers
    p_job: list[Request | None] = [None] * prefill_workers
    # In flight across the link: (arrival time at the decode worker, request).
    in_flight: list[tuple[float, Request]] = []
    # Decode side: a pool, a running set and a joining set per worker. A
    # sequence that lands mid-iteration waits for the next one, as it
    # would in any iteration-level scheduler.
    d_free = [0.0] * decode_workers
    d_running: list[list[Request]] = [[] for _ in range(decode_workers)]
    d_joining: list[list[Request]] = [[] for _ in range(decode_workers)]
    d_pool = [Pool(blocks, block_size) for _ in range(decode_workers)]

    clock, i = 0.0, 0
    while True:
        while i < len(waiting) and waiting[i].arrival_s <= clock:
            queue.append(waiting[i]); i += 1

        # Prefill completions: the first token, then the cache sets off.
        for w in range(prefill_workers):
            r = p_job[w]
            if r is not None and p_free[w] <= clock:
                _emit(trace, r, clock)               # the prompt's last position
                seconds = r.context * KV_BYTES_PER_TOKEN / link_bytes_per_s
                trace.transfer_s += seconds
                in_flight.append((clock + seconds, r))
                p_job[w] = None

        # Decode iterations that have finished: a token for everyone in them.
        for w in range(decode_workers):
            run = d_running[w]
            if not run or d_free[w] > clock:
                continue
            needed = 0
            for r in run:
                before = r.blocks(block_size)
                _emit(trace, r, clock)
                needed += r.blocks(block_size) - before
            d_pool[w].used += needed
            for r in list(run):
                if r.done:
                    d_pool[w].used -= r.blocks(block_size)
                    run.remove(r)
            trace.preemptions += _spill(trace, d_pool[w], run, queue,
                                        block_size, 0, max_batch)
            d_free[w] = clock                       # free to start another

        # Arrivals at the decode side, to the emptiest worker.
        for when, r in [x for x in in_flight if x[0] <= clock]:
            in_flight.remove((when, r))
            w = min(range(decode_workers),
                    key=lambda k: (len(d_running[k]) + len(d_joining[k]),
                                   d_pool[k].used))
            need = r.blocks(block_size)
            trace.preemptions += _spill(trace, d_pool[w], d_running[w], queue,
                                        block_size, need, max_batch - 1)
            if r.first_token_s is not None:
                r.gap_s += clock - r.first_token_s   # the link, and any wait
            d_pool[w].used += need
            d_joining[w].append(r)
            trace.peak_blocks = max(trace.peak_blocks,
                                    sum(p.used for p in d_pool))

        # Start whatever can start now.
        for w in range(prefill_workers):
            if p_job[w] is None and queue and p_free[w] <= clock:
                r = queue.pop(0)
                r.prefills += 1
                r.prefilled = r.prompt_tokens
                seconds = prefill_step(r.prompt_tokens).seconds
                p_job[w], p_free[w] = r, clock + seconds
                trace.prefill_s += seconds
                trace.iterations += 1
                trace.prefill_iterations += 1
                trace.chunk_tokens += r.prompt_tokens
        for w in range(decode_workers):
            if d_free[w] <= clock and d_joining[w]:
                d_running[w] += d_joining[w]         # they join at a boundary
                d_joining[w] = []
            run = d_running[w]
            if run and d_free[w] <= clock:
                context = sum(r.context for r in run) // len(run)
                seconds = decode_step(len(run), context).seconds
                for r in run:
                    r.gap_s += seconds
                d_free[w] = clock + seconds
                trace.decode_s += seconds
                trace.iterations += 1
                trace.batch_steps += len(run)
                trace.slots += len(run)
                trace.batch_sizes.append(len(run))

        # When does anything next happen?
        nxt = [t for t in
               ([waiting[i].arrival_s] if i < len(waiting) else [])
               + [p_free[w] for w in range(prefill_workers) if p_job[w]]
               + [t for t, _ in in_flight]
               + [d_free[w] for w in range(decode_workers) if d_running[w]]
               if t > clock]
        if not nxt:
            outstanding = (queue or in_flight or any(p_job)
                           or any(d_running) or any(d_joining))
            if outstanding:
                raise RuntimeError("the fleet is stuck with work outstanding")
            break
        clock = min(nxt)

    trace.makespan_s = clock
    return trace


def _spill(trace: Trace, pool: Pool, running: list[Request],
           queue: list[Request], block_size: int, need: int = 0,
           max_batch: int | None = None) -> int:
    """A decode worker is full. Send the newest sequence back to prefill.

    On one machine a preempted sequence just re-enters the same
    scheduler. Across two pools it has to go all the way back: its
    cache lives on the decode worker that is evicting it, and rebuilding
    it is a prefill, which happens somewhere else entirely.
    """
    evicted = 0
    while running and (pool.free < need
                       or (max_batch is not None and len(running) > max_batch)):
        victim = running.pop()
        pool.used -= victim.blocks(block_size)
        trace.recomputed_tokens += victim.generated
        trace.recomputed_prompt_tokens += victim.prefilled
        victim.generated = victim.prefilled = 0
        victim.first_token_s = None
        victim.gaps_ms.clear()
        victim.gap_s = 0.0
        queue.insert(0, victim)
        evicted += 1
    return evicted


def test_the_mean_batch_is_inside_the_batches_it_averages() -> None:
    """A mean has to lie between the smallest and largest thing averaged.

    It did not, for the schedulers whose iterations prefill and decode
    at the same time: `mean_batch` divided the sum of the batch sizes
    by the iterations that were *not* prefill iterations, and in
    Chapter 18 a prefill iteration is also a decode iteration. The
    denominator lost every chunked iteration whose batch the numerator
    had counted, and the average came out larger than any batch the
    server ever ran. This is the check that would have caught it.
    """
    reqs = [Request(id=n, arrival_s=n * 0.02, prompt_tokens=1200,
                    output_tokens=300) for n in range(120)]

    def fresh() -> list[Request]:
        return [Request(id=r.id, arrival_s=r.arrival_s,
                        prompt_tokens=r.prompt_tokens,
                        output_tokens=r.output_tokens) for r in reqs]

    traces = {
        "continuous": serve_continuous(fresh(), max_batch=64, blocks=20_000),
        "static": serve_static(fresh(), max_batch=64, blocks=20_000),
        "chunked": serve_chunked(fresh(), max_batch=64, blocks=20_000,
                                 token_budget=512),
        "colocated": serve_colocated(fresh(), workers=4, blocks=20_000,
                                     token_budget=512, max_batch=64),
        "disaggregated": serve_disaggregated(fresh(), prefill_workers=2,
                                             decode_workers=2, blocks=20_000,
                                             link_bytes_per_s=50e9,
                                             max_batch=64),
    }
    for name, t in traces.items():
        assert t.batch_sizes, f"{name} recorded no batches"
        assert min(t.batch_sizes) <= t.mean_batch <= max(t.batch_sizes), (
            f"{name}: mean batch {t.mean_batch:.1f} is outside "
            f"[{min(t.batch_sizes)}, {max(t.batch_sizes)}]")
        assert t.mean_batch <= 64, (
            f"{name}: mean batch {t.mean_batch:.1f} exceeds the cap of 64")
