"""Chapter 32: the cache that skips the model, and what it gets wrong.

Six measurements:

1. What an exact-match cache hits on three shapes of traffic, and what
   each step of normalizing the question adds -- including the step
   that adds hits by merging questions with opposite answers.
2. What leaving a field out of the key costs, in wrong answers served.
3. Whether a similarity threshold can tell a paraphrase from a
   near-miss, measured on a question set that carries the label.
4. What a per-pair false-positive rate becomes once the cache has more
   than one entry in it.
5. Lifetime against staleness: the two things a TTL trades.
6. What a hit is worth, against Chapter 15's prefix-cache hit, driven
   through the scheduler of Chapter 18.

    python3 -m bench.run_ch32
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from tinyserve.cache import (ALL_FIELDS, NearestCache, Request, ResponseCache,
                             char_ngrams, false_hit_probability, jaccard,
                             normalize, served_wrong)
from tinyserve.paged import BLOCK_SIZE
from tinyserve.questions import TOPICS, Topic
from tinyserve.reference import GPU_USD_PER_HOUR, REQUESTS_PER_S
from tinyserve.scheduler import serve_chunked

from .harness import pct, steady_state, write
from .run_ch18 import (ITL_BUDGET_MS, MAX_BATCH, PROMPT_MEAN, TTFT_BUDGET_MS,
                       make_requests)
from .run_ch41 import (N_REQUESTS as N_SCHEDULED, SETTLING_KEYS,
                       blocks, token_budget)

SEED = 0
HOURS = 24
SPAN_S = HOURS * 3600
N_FAQ = 20_000                  # a day of a small support endpoint
N_CHAT_SESSIONS = 6_000
N_AGENT_SESSIONS = 3_000
AGENT_STEPS = 6                 # tool calls in one task, including the first
CHAT_TURNS_MEAN = 2.6
# How many distinct things the traffic asks about, and how concentrated
# the asking is. Both are swept, because they -- not the cache -- are
# what decides the hit rate, and a reader's own values will differ.
CATALOGUE = 2_000
CATALOGUES = [100, 300, 1_000, 2_000, 3_000, 10_000]
ZIPF_S = 1.1
ZIPF_EXPONENTS = [0.7, 1.1, 1.5]
TENANTS = ("acme", "globex", "initech")
TENANT_WEIGHTS = (0.6, 0.3, 0.1)
# Families whose answer is a policy, and so differs from customer to
# customer. The rest are facts about the product and do not.
TENANT_SENSITIVE = frozenset({"refund", "warranty", "cancel", "leave",
                              "deposit", "rate"})
# What a follow-up turn looks like: a question that means nothing
# without the turn before it.
FOLLOW_UPS = ("and for the other one?", "are you sure?",
              "can you say that more simply?", "what about the second case?",
              "why?", "and if I do not?")
# The tail of the catalogue, past the questions that carry a label.
# Synthetic, and the chapter says so: its job is to give the
# distribution a tail of a realistic length, not to be realistic
# English. The head -- the labelled topics of tinyserve/questions.py --
# is what every claim about right and wrong answers rests on, and it
# sits at the head on purpose: the questions asked most often are the
# policy questions, and the policy questions are the ones that come in
# confusable pairs.
TAIL_VERBS = ("change", "cancel", "renew", "transfer", "pause", "split")
TAIL_NOUNS = ("booking", "invoice", "shipment", "licence", "contract", "seat")

NORMALIZERS = {
    "as typed": None,
    "case and punctuation": lambda s: normalize(s, case=True, punctuation=True),
    "and stop words dropped": lambda s: normalize(s, case=True, punctuation=True,
                                                  stop_words=True),
}
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EXAMPLE_THRESHOLD = 0.80   # the strictest threshold that still adds hits
CACHE_SIZES = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
PER_PAIR = [1e-6, 1e-5, 1e-4, 1e-3]
TTLS_S = [60, 300, 900, 3600, 6 * 3600, 24 * 3600]
POLICY_CHANGES_PER_DAY = 0.5    # how often one topic's answer moves
HIT_RATES = [0.0, 0.1, 0.2, 0.3, 0.5]
RESPONSE_HIT = 0.30
# The trace length and the settling test come from Chapter 41, so the
# bare machine measured here is the bare machine that chapter sized the
# fleet from; `worth` checks that the two agree. The rates start where
# that chapter's answer is and run well past it, because a cache in
# front of the machine moves the answer up.
RATES = [20, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60]
PREFIX_HIT = "results/ch15.json"


def catalogue(n: int) -> tuple[Topic, ...]:
    """The labelled questions, then as much synthetic tail as asked for."""
    if n < len(TOPICS):
        raise ValueError(f"the catalogue has to hold all {len(TOPICS)} "
                         "labelled topics")
    out = list(TOPICS)
    for i in range(n - len(TOPICS)):
        verb = TAIL_VERBS[i % len(TAIL_VERBS)]
        noun = TAIL_NOUNS[(i // len(TAIL_VERBS)) % len(TAIL_NOUNS)]
        base = f"How do I {verb} the {noun} on account {i}?"
        out.append(Topic(
            id=f"tail-{i}",
            # A synthetic topic has no confusable sibling: inventing one
            # would be inventing the very thing being measured.
            family=(f"tail-policy-{i}" if i % 2 else f"tail-fact-{i}"),
            answer=f"See the guide for {noun} {i}.",
            asked=(base,
                   base.lower().rstrip("?"),
                   f"I need to {verb} the {noun} on account {i}",
                   f"What is the process for {noun} {i}?")))
    return tuple(out)


# Half the synthetic tail is policy, matching the labelled head's mix.
def tenant_sensitive(topic: Topic) -> bool:
    return topic.family in TENANT_SENSITIVE or topic.family.startswith("tail-policy")


def answer_for(topic: Topic, tenant: str) -> str:
    """The reply a correct service would send. Policy answers differ by
    customer; facts about the product do not."""
    if tenant_sensitive(topic):
        return f"{topic.answer} [{tenant}]"
    return topic.answer


def zipf_weights(n: int, s: float = ZIPF_S) -> np.ndarray:
    w = 1 / np.arange(1, n + 1) ** s
    return w / w.sum()


# --- the three shapes of traffic ----------------------------------------


def faq_traffic(rng, topics=None, zipf_s: float = ZIPF_S
                ) -> list[tuple[Request, str]]:
    """One question in, one answer out, and the same questions recur."""
    topics = topics or catalogue(CATALOGUE)
    w = zipf_weights(len(topics), zipf_s)
    times = np.sort(rng.uniform(0, SPAN_S, N_FAQ))
    picks = rng.choice(len(topics), size=N_FAQ, p=w)
    tenants = rng.choice(len(TENANTS), size=N_FAQ, p=TENANT_WEIGHTS)
    phrasings = rng.random(N_FAQ)
    out = []
    for t, pick, ten, ph in zip(times, picks, tenants, phrasings):
        topic = topics[int(pick)]
        tenant = TENANTS[int(ten)]
        q = topic.asked[int(ph * len(topic.asked))]
        out.append((Request(question=q, tenant=tenant, time_s=float(t)),
                    answer_for(topic, tenant)))
    return out


def chat_traffic(rng, topics=None, zipf_s: float = ZIPF_S
                 ) -> list[tuple[Request, str]]:
    """A first turn that repeats, and follow-ups that only mean anything
    after it."""
    topics = topics or catalogue(CATALOGUE)
    w = zipf_weights(len(topics), zipf_s)
    starts = np.sort(rng.uniform(0, SPAN_S, N_CHAT_SESSIONS))
    out = []
    for start in starts:
        topic = topics[int(rng.choice(len(topics), p=w))]
        tenant = str(rng.choice(TENANTS, p=TENANT_WEIGHTS))
        answer = answer_for(topic, tenant)
        first = topic.asked[int(rng.integers(len(topic.asked)))]
        history: tuple[str, ...] = ()
        when = float(start)
        out.append((Request(question=first, tenant=tenant, history=history,
                            time_s=when), answer))
        history = (first,)
        for _ in range(int(rng.geometric(1 / CHAT_TURNS_MEAN)) - 1):
            follow = FOLLOW_UPS[int(rng.integers(len(FOLLOW_UPS)))]
            when += float(rng.exponential(20.0))
            # A follow-up's answer depends on the topic it follows, so
            # two sessions asking "are you sure?" must not share one.
            out.append((Request(question=follow, tenant=tenant,
                                history=history, time_s=when),
                        f"{answer} // {follow}"))
            history = history + (follow,)
    out.sort(key=lambda r: r[0].time_s)
    return out


def agent_traffic(rng, topics=None, zipf_s: float = ZIPF_S
                  ) -> list[tuple[Request, str]]:
    """A task stated once, then a transcript that grows and never repeats."""
    topics = topics or catalogue(CATALOGUE)
    w = zipf_weights(len(topics), zipf_s)
    starts = np.sort(rng.uniform(0, SPAN_S, N_AGENT_SESSIONS))
    out = []
    for session, start in enumerate(starts):
        topic = topics[int(rng.choice(len(topics), p=w))]
        tenant = str(rng.choice(TENANTS, p=TENANT_WEIGHTS))
        answer = answer_for(topic, tenant)
        task = topic.asked[int(rng.integers(len(topic.asked)))]
        when = float(start)
        out.append((Request(question=task, tenant=tenant, time_s=when), answer))
        for step in range(1, AGENT_STEPS):
            when += float(rng.exponential(2.0))
            # Every later step carries the tool output from the step
            # before, which no other session has ever produced.
            transcript = f"{task} | step {step} | tool result {session}-{step}"
            out.append((Request(question=transcript, tenant=tenant, time_s=when),
                        f"{answer} // step {step} of {session}"))
    out.sort(key=lambda r: r[0].time_s)
    return out


SHAPES = {"FAQ": faq_traffic, "chat": chat_traffic, "agent": agent_traffic}


# --- 0. what actually sets the hit rate ---------------------------------


def concentration() -> dict:
    """Hit rate against how many distinct questions there are, and how
    concentrated the asking is.

    This is the measurement the rest of the chapter is calibrated
    against. A cache's hit rate is not a property of the cache; it is a
    property of the traffic, and these are the two numbers it depends
    on. Both are worth measuring on a reader's own logs before any of
    the rest of this chapter is worth reading.
    """
    fold = NORMALIZERS["case and punctuation"]
    rows = []
    for zs in ZIPF_EXPONENTS:
        for n in CATALOGUES:
            topics = catalogue(n)
            trace = faq_traffic(np.random.default_rng(SEED), topics, zs)
            cache = ResponseCache(capacity=10**6, fields=ALL_FIELDS,
                                  normalizer=fold)
            for req, correct in trace:
                if cache.get(req, correct=correct) is None:
                    cache.put(req, correct)
            w = zipf_weights(n, zs)
            rows.append({"zipf_s": zs, "catalogue": n,
                         "hit_rate": cache.stats.hit_rate,
                         "wrong_rate": cache.stats.wrong_rate,
                         "entries": len(cache),
                         "top_100_share": float(w[:100].sum()),
                         "requests": len(trace)})
    return {"rows": rows, "requests": N_FAQ, "phrasings_per_topic": 4}


# --- 1. what an exact-match cache hits ----------------------------------


def exact_match(traces: dict[str, list]) -> list[dict]:
    rows = []
    for shape, trace in traces.items():
        for label, norm in NORMALIZERS.items():
            cache = ResponseCache(capacity=10**6, fields=ALL_FIELDS,
                                  normalizer=norm)
            for req, correct in trace:
                if cache.get(req, correct=correct) is None:
                    cache.put(req, correct)
            distinct = len({req.key(ALL_FIELDS, norm) for req, _ in trace})
            rows.append({"shape": shape, "normalizer": label,
                         "requests": len(trace), "distinct_keys": distinct,
                         **cache.stats.summary()})
    return rows


def agent_shape(trace: list) -> dict:
    """Why the agent shape cannot hit: only the first step of a task is
    ever a repeat of anything."""
    firsts = sum(1 for req, _ in trace if "| step " not in req.question)
    return {"requests": len(trace), "first_steps": firsts,
            "first_step_share": firsts / len(trace),
            "steps_per_task": AGENT_STEPS}


# --- 2. what a missing key field costs ----------------------------------


def key_fields(traces: dict[str, list]) -> list[dict]:
    """Hit rate and wrong answers for keys that leave something out.

    Each row drops exactly one thing the answer depends on. The hit
    rate goes up every time, which is what makes this a tempting bug.
    """
    cases = {
        "everything the answer depends on": ALL_FIELDS,
        "no tenant": tuple(f for f in ALL_FIELDS if f != "tenant"),
        "no conversation": tuple(f for f in ALL_FIELDS if f != "history"),
        "the question alone": ("question",),
    }
    fold = NORMALIZERS["case and punctuation"]
    rows = []
    for shape, trace in traces.items():
        for label, fields in cases.items():
            cache = ResponseCache(capacity=10**6, fields=fields, normalizer=fold)
            for req, correct in trace:
                if cache.get(req, correct=correct) is None:
                    cache.put(req, correct)
            rows.append({"shape": shape, "key": label, "fields": list(fields),
                         **cache.stats.summary()})
    return rows


# --- 3. can a threshold tell the two cases apart ------------------------


def similarity() -> dict:
    """Every pair of questions in the set, scored and labelled.

    The label is the thing no public trace has: whether the two
    questions must get the same answer.
    """
    items = [(q, t, char_ngrams(q)) for t in TOPICS for q in t.asked]
    same, different, confusable = [], [], []
    top: list[tuple[float, str, str]] = []
    for (q1, t1, g1), (q2, t2, g2) in combinations(items, 2):
        s = jaccard(g1, g2)
        if t1.id == t2.id:
            same.append(s)
        else:
            different.append(s)
            if t1.family == t2.family:
                confusable.append(s)
                top.append((s, q1, q2))
    same_a, diff_a = np.array(same), np.array(different)
    rows = []
    for th in THRESHOLDS:
        caught = float((same_a >= th).mean())
        wrong = float((diff_a >= th).mean())
        rows.append({"threshold": th, "paraphrases_caught": caught,
                     "different_answers_caught": wrong,
                     "pairs_wrong_per_pair_right": (wrong * len(diff_a))
                                                   / max(caught * len(same_a), 1e-9)})
    top.sort(reverse=True)
    ceiling = float(diff_a.max())
    # True pairs a threshold above the worst false pair can still
    # reach, split by whether folding case and punctuation would have
    # caught them anyway. The difference is everything a similarity
    # threshold buys that a dictionary does not already give away.
    at_one = float((same_a >= 0.999).mean())
    reachable = float(((same_a >= ceiling) & (same_a < 0.999)).mean())
    return {
        "same_scores": [round(float(x), 4) for x in same_a],
        "different_scores": [round(float(x), 4) for x in diff_a],
        "same_identical_after_folding": at_one,
        "same_reachable_above_ceiling": reachable,
        "questions": len(items), "topics": len(TOPICS),
        "same_answer_pairs": len(same), "different_answer_pairs": len(different),
        "confusable_pairs": len(confusable),
        "same_p50": float(np.median(same_a)), "same_p90": pct(same, 90),
        "different_p50": float(np.median(diff_a)),
        "different_p99": pct(different, 99),
        "different_max": float(diff_a.max()),
        "confusable_p50": float(np.median(confusable)),
        "confusable_max": float(np.max(confusable)),
        "same_below_different_max": float((same_a < diff_a.max()).mean()),
        "rows": rows,
        "closest_wrong_pairs": [{"similarity": s, "a": a, "b": b}
                                for s, a, b in top[:5]],
    }


def semantic(trace: list, exact_rate: float, topics=None) -> dict:
    """The same traffic through a cache that matches on similarity.

    The number that matters is not the hit rate -- most of that is
    exact repeats a dictionary would have caught for free. It is what
    the threshold *adds* over exact matching, set against what it gets
    wrong. A semantic cache is only ever bought for the difference.
    """
    rows, examples = [], []
    for th in THRESHOLDS:
        cache = NearestCache(threshold=th, capacity=10**6)
        for req, correct in trace:
            if cache.get(req, correct=correct) is None:
                cache.put(req, correct)
        st = cache.stats
        extra = st.hit_rate - exact_rate
        rows.append({"threshold": th, **st.summary(),
                     "extra_over_exact": extra,
                     "wrong_per_extra_hit": (st.wrong_rate / extra
                                             if extra > 0 else None)})
        if th == EXAMPLE_THRESHOLD:
            examples = cache.confusions
    return {"rows": rows, **classify(examples, topics),
            "example_threshold": EXAMPLE_THRESHOLD,
            "exact_rate": exact_rate}


def classify(confusions: list[dict], topics=None) -> dict:
    """Sort the wrong answers by *why* the two questions looked alike.

    Two different mistakes hide under one rate. One is a pair that
    differs by an identifier -- an account, an order, a date -- where
    the strings are nearly the same and the answers are unrelated. The
    other is a pair that differs by a decisive word: adult or child,
    over or under, covered or not covered. Both are real and they need
    different fixes, so they are counted apart.
    """
    topics = topics or catalogue(CATALOGUE)
    owner = {q: t for t in topics for q in t.asked}
    counts = {"an identifier in the question": 0,
              "a decisive word": 0,
              "nothing in particular": 0}
    shown: dict[str, list[dict]] = {k: [] for k in counts}
    for c in confusions:
        a, b = owner.get(c["asked"]), owner.get(c["answered_from"])
        if a is None or b is None:
            kind = "nothing in particular"
        elif a.id.startswith("tail-") and b.id.startswith("tail-"):
            kind = "an identifier in the question"
        elif a.family == b.family:
            kind = "a decisive word"
        else:
            kind = "nothing in particular"
        counts[kind] += 1
        if len(shown[kind]) < 3:
            shown[kind].append(c)
    total = max(sum(counts.values()), 1)
    return {"confusions": sum(counts.values()),
            "confusion_kinds": [{"kind": k, "count": v, "share": v / total,
                                 "examples": shown[k]}
                                for k, v in counts.items()]}


# --- 4. what a per-pair rate becomes at scale ---------------------------


def scale() -> dict:
    """A threshold is quoted per pair. A lookup is not one pair."""
    grid = [{"per_pair": f,
             "rows": [{"entries": n, "any_false_hit": false_hit_probability(f, n)}
                      for n in CACHE_SIZES]}
            for f in PER_PAIR]
    # What that does once the base rate is applied. `twin_rate` is read
    # from the FAQ traffic rather than assumed: the share of requests
    # whose topic has been seen before under a different wording.
    return {"grid": grid, "sizes": CACHE_SIZES, "per_pair": PER_PAIR}


def operating_points(twin_rate: float, recall: float = 0.9) -> list[dict]:
    rows = []
    for f in PER_PAIR:
        for n in (1_000, 100_000):
            r = served_wrong(f, n, recall, twin_rate)
            rows.append({"per_pair": f, "entries": n, "recall": recall,
                         "twin_rate": twin_rate, **r})
    return rows


def twin_rate_of(trace: list) -> float:
    """Share of requests that really do have an equivalent already cached.

    The base rate: the ceiling on how often any similarity cache can be
    right, and the number a threshold argument has to be weighed
    against.
    """
    fold = NORMALIZERS["case and punctuation"]
    seen_exact: set = set()
    seen_answer: set = set()
    twins = 0
    for req, correct in trace:
        key = req.key(ALL_FIELDS, fold)
        equivalent = (req.tenant, correct)
        if equivalent in seen_answer and key not in seen_exact:
            twins += 1
        seen_exact.add(key)
        seen_answer.add(equivalent)
    return twins / len(trace)


# --- 5. lifetime against staleness --------------------------------------


def staleness(trace: list, rng) -> list[dict]:
    """A TTL trades hits for freshness. Both sides, measured.

    Each topic's answer is changed at Poisson times; a hit whose entry
    was stored before the last change is a stale answer, counted.
    """
    changes = {}
    rate = POLICY_CHANGES_PER_DAY / 86400
    for t in TOPICS:
        times, clock = [], 0.0
        while True:
            clock += float(rng.exponential(1 / rate))
            if clock > SPAN_S:
                break
            times.append(clock)
        changes[t.answer.split(" [")[0]] = times

    def version(correct: str, when: float) -> str:
        base = correct.split(" [")[0].split(" //")[0]
        n = sum(1 for c in changes.get(base, ()) if c <= when)
        return f"{correct}@v{n}"

    fold = NORMALIZERS["case and punctuation"]
    rows = []
    for ttl in TTLS_S:
        cache = ResponseCache(capacity=10**6, ttl_s=ttl, fields=ALL_FIELDS,
                              normalizer=fold)
        for req, correct in trace:
            want = version(correct, req.time_s)
            if cache.get(req, correct=want) is None:
                cache.put(req, want)
        s = cache.stats
        rows.append({"ttl_s": ttl, "ttl_label": _duration(ttl),
                     "hit_rate": s.hit_rate, "stale_rate": s.wrong_rate,
                     "stale_share_of_hits": s.wrong_share_of_hits,
                     "expired": s.expired})
    return rows


def _duration(seconds: float) -> str:
    for unit, n in (("d", 86400), ("h", 3600), ("min", 60)):
        if seconds >= n:
            return f"{seconds / n:g} {unit}"
    return f"{seconds:g} s"


# --- 6. what a hit is worth ---------------------------------------------


def prefix_hit_rate() -> float:
    """Chapter 15's measured share of prompt tokens already cached."""
    import json
    from pathlib import Path
    path = Path(PREFIX_HIT)
    if not path.exists():
        raise SystemExit("run `make ch15` first: this chapter compares a "
                         "response-cache hit against that chapter's prefix hit")
    return float(json.loads(path.read_text())["unlimited"]["hit_rate"])


def _run(rate: float, n: int = N_SCHEDULED, *, hit_rate: float = 0.0,
         prefix: float = 0.0) -> dict:
    """One offered load, with a cache of one kind or the other in front.

    A response-cache hit never reaches the server, so it is removed
    from the stream: the machine sees `(1 - hit_rate)` of what arrived.
    A prefix-cache hit does reach it, but arrives with part of its
    prompt already computed, so it is admitted with `prefilled` already
    set and only the rest is read.

    One thing this does not model: the adopted blocks still occupy the
    pool, and here they are not charged to it. The pool is not what
    binds -- without a cache the peak is `peak_blocks` of the
    `blocks()` the machine has, reported in the results so it can be
    checked -- and Chapter 15 measured sharing to *lower* the memory a
    stream needs, by {{dedup}}x. So the timing here is the compute
    saving alone, which is the conservative half.
    """
    rng = np.random.default_rng(SEED)
    reqs = make_requests(n, rate, rng)
    arrivals = len(reqs)
    if hit_rate:
        keep = rng.random(len(reqs)) >= hit_rate
        reqs = [r for r, k in zip(reqs, keep) if k]
        for i, r in enumerate(reqs):
            r.id = i
    if prefix:
        for r in reqs:
            cached = int(r.prompt_tokens * prefix) // BLOCK_SIZE * BLOCK_SIZE
            r.prefilled = min(cached, r.prompt_tokens - 1)
    trace = serve_chunked(reqs, max_batch=MAX_BATCH, blocks=blocks(),
                          token_budget=token_budget())
    done = [r for r in trace.requests if r.finish_s is not None]
    span = max(r.arrival_s for r in trace.requests)
    lo, hi = span * 0.30, span * 0.90
    inside = [r for r in done if lo <= r.arrival_s < hi]
    ttft = [(r.first_token_s - r.arrival_s) * 1e3 for r in inside
            if r.first_token_s is not None]
    busy = trace.makespan_s - trace.idle_s
    total = [r.finish_s - r.arrival_s for r in inside]
    return {
        "arriving_per_s": rate,
        "n_requests": n,
        "reached_the_model_per_s": rate * (1 - hit_rate),
        "busy_s": busy,
        "mean_time_s": float(np.mean(total)) if total else float("nan"),
        "tokens_per_s": trace.tokens_between(lo, hi) / (hi - lo),
        "gpu_ms_per_arrival": busy / arrivals * 1e3,
        "ttft_p99_ms": pct(ttft, 99),
        "itl_p99_ms": pct(trace.gaps_ms, 99),
        "p99_s": pct(total, 99),
        "mean_batch": trace.mean_batch,
        "drained_over_span": trace.makespan_s / span,
        # Chapter 41's three conditions, unchanged: both promises kept,
        # and the machine actually drained rather than building a
        # backlog it would never clear.
        "meets_promises": bool(pct(ttft, 99) <= TTFT_BUDGET_MS
                               and pct(trace.gaps_ms, 99) <= ITL_BUDGET_MS),
        "peak_blocks": trace.peak_blocks,
    }


def _capacity(rows: list[dict]) -> float:
    """The highest offered load that kept both promises.

    Both ends are checked. If nothing met them the sweep started too
    high; if the top of the sweep met them it stopped too low, and the
    number would be the edge of the grid rather than a capacity.
    """
    ok = [r["arriving_per_s"] for r in rows
          if r["settled"] and r["meets_promises"]]
    if not ok:
        raise RuntimeError(f"no offered load in {RATES} met the promises")
    if max(ok) == max(RATES):
        raise RuntimeError(
            f"every rate up to {max(RATES)} met the promises; the sweep has "
            "to reach past the capacity for the number to mean anything")
    return max(ok)


def worth() -> dict:
    """A response hit against a prefix hit, on the same scheduler."""
    prefix = prefix_hit_rate()
    h = RESPONSE_HIT
    cases = {"no cache": dict(),
             "prefix cache": dict(prefix=prefix),
             "response cache": dict(hit_rate=h)}
    # Every rate measured twice, as Chapter 41 does it, so a latency
    # that has not settled cannot be mistaken for a capacity.
    sweeps = {}
    for name, kw in cases.items():
        rows = []
        for rate in RATES:
            check = steady_state(lambda n, r=rate, k=kw: _run(r, n, **k),
                                 N_SCHEDULED, SETTLING_KEYS)
            row = dict(check["long"])
            row["settled"] = check["settled"]
            row["worst_drift"] = check["worst_drift"]
            rows.append(row)
        sweeps[name] = rows
    caps = {name: _capacity(rows) for name, rows in sweeps.items()}

    # The bare machine here must be the bare machine Chapter 41 sized
    # the fleet from. Same scheduler, same promises, same arrival
    # model -- so if the two disagree, one of them has drifted, and the
    # fleet in this chapter would not be the fleet in that one.
    import json
    from pathlib import Path
    ch41 = json.loads(Path("results/ch41.json").read_text())["sizing"]
    if caps["no cache"] != ch41["highest_rate_meeting_both_promises"]:
        raise RuntimeError(
            f"this chapter measures the bare machine at {caps['no cache']} "
            f"requests a second; Chapter 41 measured "
            f"{ch41['highest_rate_meeting_both_promises']}")
    fleets = {name: math.ceil(REQUESTS_PER_S / c) for name, c in caps.items()}

    # A request that never reaches the model cannot cost anything, so a
    # response cache's capacity ought to be the bare machine's divided
    # by what is left. Predicted before it was measured, and the sweep
    # is fine enough to tell the two apart.
    predicted = caps["no cache"] / (1 - h)
    nearest = min(RATES, key=lambda r: abs(r - predicted))
    fleet_by_hit = [
        {"hit_rate": hr,
         "machines": math.ceil(REQUESTS_PER_S * (1 - hr) / caps["no cache"]),
         "usd_per_hour": math.ceil(REQUESTS_PER_S * (1 - hr) / caps["no cache"])
                         * GPU_USD_PER_HOUR}
        for hr in HIT_RATES]
    baseline = {r["arriving_per_s"]: r for r in sweeps["no cache"]}
    at_cap = baseline[caps["no cache"]]
    # What the same fleet buys in tail latency instead of in machines.
    eased = min((r for r in sweeps["no cache"]
                 if r["arriving_per_s"] >= caps["no cache"] * (1 - h)),
                key=lambda r: r["arriving_per_s"])
    return {
        "prefix_hit_rate": prefix, "response_hit_rate": h,
        "sweeps": sweeps, "capacity": caps, "fleet": fleets,
        "usd_per_hour": {k: v * GPU_USD_PER_HOUR for k, v in fleets.items()},
        "fleet_by_hit_rate": fleet_by_hit,
        "predicted_response_capacity": predicted,
        "prediction_holds": abs(caps["response cache"] - predicted) <= max(
            abs(nearest - predicted), 1e-9) + 1e-9,
        "prefix_gain": caps["prefix cache"] / caps["no cache"] - 1,
        # What share of *requests* a response cache would have to catch
        # to be worth what the prefix cache is worth here. The two are
        # measured in different units -- prompt tokens against whole
        # requests -- and this is the exchange rate between them.
        "response_hit_matching_prefix":
            1 - caps["no cache"] / caps["prefix cache"],
        "response_gain": caps["response cache"] / caps["no cache"] - 1,
        "ttft_p99_at_capacity_ms": at_cap["ttft_p99_ms"],
        "ttft_p99_eased_ms": eased["ttft_p99_ms"],
        "eased_rate": eased["arriving_per_s"],
        "peak_blocks_no_cache": at_cap["peak_blocks"],
        "blocks_available": blocks(),
        "agrees_with_ch41": True,
        "promises": {"ttft_p99_ms": TTFT_BUDGET_MS, "itl_p99_ms": ITL_BUDGET_MS},
        "prompt_mean": PROMPT_MEAN,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    topics = catalogue(CATALOGUE)
    traces = {name: fn(np.random.default_rng(SEED), topics)
              for name, fn in SHAPES.items()}
    conc = concentration()
    exact = exact_match(traces)
    keys = key_fields(traces)
    sim = similarity()
    faq_exact = next(r["hit_rate"] for r in exact
                     if r["shape"] == "FAQ"
                     and r["normalizer"] == "case and punctuation")
    sem = semantic(traces["FAQ"], faq_exact, topics)

    # The same measurement on traffic that asks only the labelled
    # questions. There every confusable pair differs by a decisive
    # word rather than by an account number, which is the other way a
    # similarity cache is wrong and the one the tail drowns out.
    head_trace = faq_traffic(np.random.default_rng(SEED), TOPICS)
    head_exact = ResponseCache(capacity=10**6, fields=ALL_FIELDS,
                               normalizer=NORMALIZERS["case and punctuation"])
    for req, correct in head_trace:
        if head_exact.get(req, correct=correct) is None:
            head_exact.put(req, correct)
    sem_head = semantic(head_trace, head_exact.stats.hit_rate, TOPICS)
    twin = twin_rate_of(traces["FAQ"])
    sc = scale()
    sc["operating_points"] = operating_points(twin)
    sc["twin_rate"] = twin
    stale = staleness(traces["FAQ"], rng)
    w = worth()

    payload = {
        "concentration": conc,
        "exact": exact,
        "agent_shape": agent_shape(traces["agent"]),
        "keys": keys,
        "similarity": sim,
        "semantic": sem,
        "semantic_policy_only": sem_head,
        "scale": sc,
        "staleness": stale,
        "worth": w,
        "assumptions": {
            "seed": SEED, "hours": HOURS, "n_faq": N_FAQ,
            "chat_sessions": N_CHAT_SESSIONS, "agent_sessions": N_AGENT_SESSIONS,
            "agent_steps": AGENT_STEPS, "chat_turns_mean": CHAT_TURNS_MEAN,
            "zipf_s": ZIPF_S, "zipf_exponents": ZIPF_EXPONENTS,
            "catalogue": CATALOGUE, "catalogues": CATALOGUES,
            "labelled_topics": len(TOPICS),
            "questions": sim["questions"], "tenants": list(TENANTS),
            "tenant_weights": list(TENANT_WEIGHTS),
            "tenant_sensitive": sorted(TENANT_SENSITIVE),
            "thresholds": THRESHOLDS, "cache_sizes": CACHE_SIZES,
            "per_pair": PER_PAIR, "ttls_s": TTLS_S,
            "policy_changes_per_day": POLICY_CHANGES_PER_DAY,
            "rates": list(RATES), "prompt_mean": PROMPT_MEAN,
            "max_batch": MAX_BATCH, "token_budget": token_budget(),
            "blocks": blocks(), "n_scheduled": N_SCHEDULED,
            "requests_per_s": REQUESTS_PER_S,
            "gpu_usd_per_hour": GPU_USD_PER_HOUR,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ch32.json", payload)
    print(f"wrote {path}")

    print("  the hit rate is a property of the traffic:")
    for r in conc["rows"]:
        if r["zipf_s"] == ZIPF_S:
            print(f"    {r['catalogue']:>6,} distinct questions, skew "
                  f"{r['zipf_s']}: {r['hit_rate'] * 100:5.1f}% hit "
                  f"(top 100 carry {r['top_100_share'] * 100:.0f}% of asking)")
    print(f"  exact-match hit rate at {CATALOGUE:,} questions, by shape:")
    for r in exact:
        flag = "  <- merges opposite questions" if r["wrong_hits"] else ""
        print(f"    {r['shape']:<6}{r['normalizer']:<24}"
              f"{r['hit_rate'] * 100:6.1f}% hit"
              f"{r['wrong_rate'] * 100:8.2f}% served wrong{flag}")
    a = payload["agent_shape"]
    print(f"    agent traffic: only {a['first_step_share'] * 100:.0f}% of "
          f"requests are a first step, so the rest cannot repeat at all")
    print("  what a shorter key buys, and costs:")
    for r in keys:
        print(f"    {r['shape']:<6}{r['key']:<34}"
              f"{r['hit_rate'] * 100:6.1f}% hit"
              f"{r['wrong_rate'] * 100:8.2f}% served wrong")
    print(f"  similarity over {sim['questions']} labelled questions: "
          f"same-answer median {sim['same_p50']:.3f}, "
          f"different-answer worst {sim['different_max']:.3f} "
          f"({sim['same_below_different_max'] * 100:.0f}% of true pairs score "
          f"below the worst false one)")
    print("  the same threshold, on the FAQ stream:")
    for r in sem["rows"]:
        wpe = r["wrong_per_extra_hit"]
        print(f"    {r['threshold']:.2f}  {r['hit_rate'] * 100:6.1f}% hit"
              f"{r['extra_over_exact'] * 100:+7.2f}% over exact"
              f"{r['wrong_rate'] * 100:8.2f}% served wrong"
              + (f"  ({wpe:,.0f} wrong per extra hit)" if wpe and wpe > 0 else ""))
    print(f"  what the {sem['example_threshold']:.2f} threshold confused, "
          f"by what made the two questions look alike:")
    for k in sem["confusion_kinds"]:
        print(f"    {k['kind']:<32}{k['count']:>7,}  {k['share'] * 100:5.1f}%")
        for e in k["examples"][:1]:
            print(f"        {e['similarity']:.2f}  {e['asked']!r}")
            print(f"        {'':4}  answered from {e['answered_from']!r}")
    hrow = next(r for r in sem_head["rows"]
                if r["threshold"] == sem_head["example_threshold"])
    print(f"  policy questions only, at {hrow['threshold']:.2f}: "
          f"{hrow['hit_rate'] * 100:.1f}% hit, "
          f"{hrow['wrong_rate'] * 100:.2f}% served wrong")
    for k in sem_head["confusion_kinds"]:
        if k["count"]:
            print(f"    {k['kind']:<32}{k['count']:>7,}  {k['share'] * 100:5.1f}%")
            for e in k["examples"][:1]:
                print(f"        {e['similarity']:.2f}  {e['asked']!r}")
                print(f"        {'':4}  answered from {e['answered_from']!r}")
    print(f"  base rate: {sc['twin_rate'] * 100:.1f}% of requests have an "
          f"equivalent already cached")
    print("  a per-pair rate, once the cache has entries in it:")
    for g in sc["grid"]:
        row = {r["entries"]: r["any_false_hit"] for r in g["rows"]}
        print(f"    {g['per_pair']:.0e} per pair -> "
              f"{row[1_000] * 100:6.2f}% at 1k entries, "
              f"{row[100_000] * 100:6.2f}% at 100k")
    print("  lifetime against staleness:")
    for r in stale:
        print(f"    {r['ttl_label']:>6}  {r['hit_rate'] * 100:6.1f}% hit"
              f"{r['stale_rate'] * 100:8.2f}% stale")
    print("  what a hit is worth, through the scheduler:")
    for name, cap in w["capacity"].items():
        print(f"    {name:<16}{cap:5.0f} requests a second a machine, "
              f"fleet {w['fleet'][name]}, ${w['usd_per_hour'][name]:.2f}/hour")
    print(f"    the same fleet at {w['eased_rate']} instead of "
          f"{w['capacity']['no cache']} a second: TTFT p99 "
          f"{w['ttft_p99_at_capacity_ms']:.0f} ms -> "
          f"{w['ttft_p99_eased_ms']:.0f} ms")


if __name__ == "__main__":
    main()
