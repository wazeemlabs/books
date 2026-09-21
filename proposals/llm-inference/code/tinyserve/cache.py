"""A cache in front of the model, and the ways it answers wrongly.

Every other saving in this book makes the model cheaper to run. This
one skips it. A request whose answer is already known costs a dictionary
lookup: no prefill, no decode, no GPU. Nothing else in the book is
close.

The price is that this is the only layer that can change the answer.
Chapter 15's prefix cache re-used arithmetic and was checked to the last
bit; this re-uses a *conclusion*, and a conclusion is only re-usable if
nothing the conclusion depended on has changed. Three pieces, and the
last two are where the danger is:

* `ResponseCache` -- exact match, least-recently-used, with a lifetime.
  What it matches on is a parameter, because the field somebody leaves
  out of the key is the bug.
* `NearestCache` -- match on similarity instead of equality, which
  turns a lookup into a classifier and gives it a false-positive rate.
* `false_hit_probability` -- what that rate does as the cache fills,
  which is the arithmetic most semantic-cache designs are missing.

Both caches record not just hits and misses but *wrong* hits: the
experiment knows each request's correct answer and tells the cache,
so serving the wrong one is measured rather than argued about.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Sequence

# Everything an answer can depend on. A cache key is a subset of these,
# and `KEY_FIELDS` is the subset that is actually safe: drop one and
# `test_an_omitted_field_serves_the_wrong_tenants_answer` fails.
ALL_FIELDS = ("question", "tenant", "system", "model", "history")
KEY_FIELDS = ALL_FIELDS


@dataclass(frozen=True)
class Request:
    """One arriving request, and the context its answer depends on.

    `question` is what the user typed. Everything else is context the
    user never sees and the cache key usually forgets: which customer's
    data the answer is about, which system prompt and tool set was in
    force, which model version produced it, and what was said earlier
    in the conversation.
    """

    question: str
    tenant: str = "acme"
    system: str = "v1"
    model: str = "8b-instruct"
    history: tuple[str, ...] = ()
    time_s: float = 0.0

    def key(self, fields: Sequence[str] = KEY_FIELDS,
            normalize: "Callable[[str], str] | None" = None) -> tuple:
        """This request reduced to what the cache matches on."""
        out = []
        for f in fields:
            if f not in ALL_FIELDS:
                raise ValueError(f"{f!r} is not part of a request")
            v = getattr(self, f)
            out.append(normalize(v) if normalize and f == "question" else v)
        return tuple(out)


@dataclass
class Entry:
    answer: str
    stored_at: float
    question: str = ""
    grams: frozenset[str] = frozenset()


@dataclass
class Stats:
    """What a cache did, counted so the wrong hits cannot hide in the hits."""

    lookups: int = 0
    hits: int = 0
    wrong_hits: int = 0
    expired: int = 0
    evictions: int = 0

    @property
    def misses(self) -> int:
        return self.lookups - self.hits

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def wrong_rate(self) -> float:
        """Wrong answers as a share of everything served, not of hits.

        The share of *hits* that are wrong flatters a cache that
        rarely hits. What a caller experiences is this.
        """
        return self.wrong_hits / self.lookups if self.lookups else 0.0

    @property
    def wrong_share_of_hits(self) -> float:
        return self.wrong_hits / self.hits if self.hits else 0.0

    def summary(self) -> dict:
        return {"lookups": self.lookups, "hits": self.hits,
                "misses": self.misses, "hit_rate": self.hit_rate,
                "wrong_hits": self.wrong_hits, "wrong_rate": self.wrong_rate,
                "wrong_share_of_hits": self.wrong_share_of_hits,
                "expired": self.expired, "evictions": self.evictions}


# --- normalizing the question -------------------------------------------

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")

# Function words of the kind a text-search pipeline drops before
# indexing. The list is short and stated here on purpose: what matters
# is not which words are on it but that several of them -- not, no,
# without, under, over, before, after -- are the whole difference
# between two questions with different answers.
STOP_WORDS = frozenset("""
a an and are as at be by can do does for from had has have how i if in is it
my of on or our that the their there they this to was we what when where which
who will with you your
no not nor without under over before after against
""".split())


def normalize(text: str, *, case: bool = False, punctuation: bool = False,
              stop_words: bool = False) -> str:
    """Fold a question, one optional step at a time.

    Each step raises the hit rate by merging questions that were
    written differently. The last one also merges questions that *mean*
    differently, which is why it is separate from the other two and why
    `run_ch32.py` measures what it costs.
    """
    out = text.strip()
    if case:
        out = out.lower()
    if punctuation:
        out = _PUNCT.sub(" ", out)
    out = _SPACE.sub(" ", out).strip()
    if stop_words:
        folded = out.lower()
        keep = [w for w in folded.split() if w not in STOP_WORDS]
        out = " ".join(keep)
    return out


def char_ngrams(text: str, n: int = 5) -> frozenset[str]:
    """Character n-grams of the folded question.

    A similarity measure the book can carry in thirty lines and a
    reader can reproduce exactly. It is lexical, not semantic, and
    Chapter 32 says so where it uses it: a sentence embedding would
    score genuine paraphrases higher than this does. What it would not
    change is the other half of the picture, because two questions that
    differ in one decisive word are close under any measure that is
    continuous in its input.
    """
    s = normalize(text, case=True, punctuation=True)
    if len(s) < n:
        return frozenset([s])
    return frozenset(s[i:i + n] for i in range(len(s) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


# --- the caches ----------------------------------------------------------


class ResponseCache:
    """Exact match on a key, least-recently-used, with a lifetime.

    `fields` is what the key is made of. Serving every request from
    `ALL_FIELDS` is correct and is what the default does; the
    experiment shortens the list to measure what each omission costs.
    """

    def __init__(self, capacity: int = 10_000, ttl_s: float | None = None,
                 fields: Sequence[str] = KEY_FIELDS,
                 normalizer: "Callable[[str], str] | None" = None) -> None:
        if capacity < 1:
            raise ValueError("a cache with no room is not a cache")
        self.capacity, self.ttl_s = capacity, ttl_s
        self.fields, self.normalizer = tuple(fields), normalizer
        self.entries: "OrderedDict[tuple, Entry]" = OrderedDict()
        self.stats = Stats()

    def _key(self, req: Request) -> tuple:
        return req.key(self.fields, self.normalizer)

    def get(self, req: Request, correct: str | None = None) -> str | None:
        """Look `req` up. `correct` is the answer the model would have
        given; passing it is what lets a wrong hit be counted."""
        self.stats.lookups += 1
        key = self._key(req)
        entry = self.entries.get(key)
        if entry is None:
            return None
        if self.ttl_s is not None and req.time_s - entry.stored_at > self.ttl_s:
            del self.entries[key]
            self.stats.expired += 1
            return None
        self.entries.move_to_end(key)
        self.stats.hits += 1
        if correct is not None and entry.answer != correct:
            self.stats.wrong_hits += 1
        return entry.answer

    def put(self, req: Request, answer: str) -> None:
        key = self._key(req)
        self.entries[key] = Entry(answer=answer, stored_at=req.time_s,
                                  question=req.question)
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
            self.stats.evictions += 1

    def __len__(self) -> int:
        return len(self.entries)


class NearestCache:
    """Match on similarity rather than equality.

    The same cache with `==` replaced by `>= threshold`, which sounds
    like a small change and is not: equality has no false positives and
    a threshold does. Every entry in the cache is one more chance to be
    wrong, so the error rate is a property of the cache's *size* as
    much as of the threshold.

    The scan here is linear and exact. Production vector stores are
    neither -- they use an approximate index, which means the entry
    they return may not even be the nearest one. That can only add
    error to the numbers Chapter 32 reports.
    """

    def __init__(self, threshold: float, capacity: int = 10_000,
                 fields: Sequence[str] = ("tenant", "system", "model")) -> None:
        self.threshold, self.capacity = threshold, capacity
        self.fields = tuple(f for f in fields if f != "question")
        # Every pair of questions this cache confused, kept so the
        # failures can be read and sorted rather than only counted. A
        # rate says how often; these say what.
        self.confusions: list[dict] = []
        # Bucketed by the exact part of the key, so similarity is only
        # ever compared within one tenant's, one system prompt's
        # entries. A cache that does not do this is measured too.
        self.buckets: dict[tuple, "OrderedDict[str, Entry]"] = {}
        self.stats = Stats()
        self.comparisons = 0

    def _bucket(self, req: Request) -> "OrderedDict[str, Entry]":
        return self.buckets.setdefault(req.key(self.fields), OrderedDict())

    def _size(self) -> int:
        return sum(len(b) for b in self.buckets.values())

    def nearest(self, req: Request) -> tuple[Entry | None, float]:
        """The most similar entry in this request's bucket, and how similar."""
        grams = char_ngrams(req.question)
        best, score = None, 0.0
        bucket = self._bucket(req)
        self.comparisons += len(bucket)
        for entry in bucket.values():
            s = jaccard(grams, entry.grams)
            if s > score:
                best, score = entry, s
        return best, score

    def get(self, req: Request, correct: str | None = None) -> str | None:
        self.stats.lookups += 1
        entry, score = self.nearest(req)
        if entry is None or score < self.threshold:
            return None
        self.stats.hits += 1
        if correct is not None and entry.answer != correct:
            self.stats.wrong_hits += 1
            self.confusions.append({"asked": req.question,
                                    "answered_from": entry.question,
                                    "similarity": score})
        bucket = self._bucket(req)
        bucket.move_to_end(entry.question)
        return entry.answer

    def put(self, req: Request, answer: str) -> None:
        bucket = self._bucket(req)
        bucket[req.question] = Entry(answer=answer, stored_at=req.time_s,
                                     question=req.question,
                                     grams=char_ngrams(req.question))
        bucket.move_to_end(req.question)
        while self._size() > self.capacity:
            biggest = max(self.buckets.values(), key=len)
            biggest.popitem(last=False)
            self.stats.evictions += 1


# --- what a false-positive rate does as the cache fills ------------------


def false_hit_probability(per_pair: float, n_entries: int) -> float:
    """Chance that at least one of `n_entries` clears the threshold by accident.

    A similarity threshold is quoted as a per-pair error rate: how often
    two unrelated questions score above it. A lookup does not compare
    one pair, it compares `n_entries` of them and takes the best, so the
    rate that matters is this one. It rises with the size of the cache,
    which is the same thing as saying a semantic cache gets less safe
    the more useful it becomes.
    """
    if not 0.0 <= per_pair <= 1.0:
        raise ValueError(f"{per_pair} is not a probability")
    if n_entries < 0:
        raise ValueError("a cache cannot hold fewer than no entries")
    return 1.0 - (1.0 - per_pair) ** n_entries


def served_wrong(per_pair: float, n_entries: int, recall: float,
                 twin_rate: float) -> dict[str, float]:
    """Per lookup: the chance of a real hit, and of a wrong one.

    `twin_rate` is the share of arriving questions that really do have
    an equivalent already cached -- the base rate, and the number most
    arguments about thresholds forget. `recall` is how often the
    embedder scores that equivalent above the threshold.

    A question with a twin can also be matched to the wrong entry
    first; the model here is the pessimistic one for the *rest* of the
    traffic, which is where the volume is, and it is stated rather than
    buried: a question with no twin is served wrongly whenever any of
    the other entries clears the threshold by accident.
    """
    false = false_hit_probability(per_pair, n_entries)
    good = twin_rate * recall
    bad = (1.0 - twin_rate) * false
    return {"hit_rate": good + bad, "right_rate": good, "wrong_rate": bad,
            "false_hit_probability": false,
            "wrong_share_of_hits": bad / (good + bad) if good + bad else 0.0}


# --- tests ---------------------------------------------------------------


def test_a_stored_answer_comes_back() -> None:
    cache = ResponseCache()
    req = Request("what is the refund window?")
    assert cache.get(req) is None
    cache.put(req, "30 days")
    assert cache.get(Request("what is the refund window?")) == "30 days"
    assert cache.stats.hits == 1 and cache.stats.misses == 1


def test_an_omitted_field_serves_the_wrong_tenants_answer() -> None:
    """The failure this chapter is about, made executable.

    Two customers ask the same words and have different answers. With
    the tenant in the key both are served correctly. Leave it out and
    the second is served the first one's answer -- not occasionally,
    not probabilistically, but every time.
    """
    q = "how many days of leave do i get?"
    a, b = Request(q, tenant="acme"), Request(q, tenant="globex")
    answers = {"acme": "20 days", "globex": "25 days"}

    safe = ResponseCache(fields=ALL_FIELDS)
    safe.put(a, answers["acme"])
    assert safe.get(b, correct=answers["globex"]) is None
    assert safe.stats.wrong_hits == 0

    starved = ResponseCache(fields=("question",))
    starved.put(a, answers["acme"])
    assert starved.get(b, correct=answers["globex"]) == "20 days"
    assert starved.stats.wrong_hits == 1
    assert starved.stats.wrong_share_of_hits == 1.0


def test_an_entry_past_its_lifetime_is_a_miss() -> None:
    cache = ResponseCache(ttl_s=60)
    cache.put(Request("q", time_s=0.0), "old")
    assert cache.get(Request("q", time_s=59.0)) == "old"
    assert cache.get(Request("q", time_s=120.0)) is None
    assert cache.stats.expired == 1
    # And the expired entry is gone, not merely skipped.
    assert len(cache) == 0


def test_eviction_takes_the_least_recently_used() -> None:
    """Reading an entry has to count as using it, or a cache throws away
    exactly the entries that are earning their keep."""
    cache = ResponseCache(capacity=2)
    for q in ("a", "b"):
        cache.put(Request(q), q.upper())
    cache.get(Request("a"))               # "a" is now the recent one
    cache.put(Request("c"), "C")
    assert cache.get(Request("a")) == "A"
    assert cache.get(Request("b")) is None
    assert cache.stats.evictions == 1


def test_similarity_is_a_similarity() -> None:
    a = char_ngrams("how do i cancel my subscription?")
    b = char_ngrams("How do I cancel my subscription")
    assert jaccard(a, a) == 1.0
    assert jaccard(a, b) == jaccard(b, a)
    assert jaccard(a, b) == 1.0, "case and punctuation must fold away"
    assert 0.0 <= jaccard(a, char_ngrams("what is the weather")) < 0.3


def test_one_decisive_word_scores_higher_than_a_paraphrase() -> None:
    """Why a threshold cannot separate the two cases.

    The pair that must *not* share an answer differs by one word. The
    pair that must share one is written from scratch. Lexically the
    dangerous pair is the closer of the two, and no choice of threshold
    puts them on opposite sides.
    """
    near_miss = jaccard(char_ngrams("is this dose safe for an adult?"),
                        char_ngrams("is this dose safe for a child?"))
    paraphrase = jaccard(char_ngrams("how do i cancel my subscription?"),
                         char_ngrams("i want to stop being billed every month"))
    assert near_miss > paraphrase, (near_miss, paraphrase)


def test_stop_word_folding_merges_opposite_questions() -> None:
    """Normalization that drops function words is not normalization.

    Two questions with opposite answers become the same string, which
    an exact-match cache then answers with total confidence.
    """
    yes = "is this covered by the warranty?"
    no = "is this not covered by the warranty?"
    assert normalize(yes, case=True, punctuation=True) != \
           normalize(no, case=True, punctuation=True)
    assert normalize(yes, case=True, punctuation=True, stop_words=True) == \
           normalize(no, case=True, punctuation=True, stop_words=True)


def test_a_bigger_cache_is_a_likelier_accident() -> None:
    small = false_hit_probability(1e-4, 100)
    large = false_hit_probability(1e-4, 100_000)
    assert small < 0.02 < 0.99 < large, (small, large)
    assert false_hit_probability(1e-4, 0) == 0.0
    assert false_hit_probability(0.0, 10**6) == 0.0
