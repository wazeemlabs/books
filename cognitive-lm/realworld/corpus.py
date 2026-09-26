"""What did the model read? Counts over OLMo-2's own pretraining corpus.

OLMo-2 ships with its training data (OLMo-mix-1124), and infini-gram serves
exact n-gram and co-occurrence counts over it
(https://infini-gram.readthedocs.io/en/latest/api.html). That lets us build
the checksum from exactly the text the model was trained on:

    entity filter   count(subject) > 0
    pair checksum   count(subject AND value) > 0, the two within
                    max_diff_tokens of each other

Every response is cached on disk (one JSON line per query), so a rerun or a
crash never repeats a request.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://api.infini-gram.io/"
INDEX = "v4_olmo-mix-1124_llama"
# above this many matches a clause is subsampled and a conjunction can come back
# as a false zero; 500,000 is the API's maximum
MAX_CLAUSE_FREQ = 500_000
# The public API answers 403 when hit too fast (seen at 4 parallel clients), so
# every request waits for a shared slot, and a refusal backs off for minutes
MIN_INTERVAL_S = 0.5
RATE_LIMIT_STATUS = (403, 429)


class Corpus:
    def __init__(self, cache_path, index=INDEX, max_diff_tokens=100, workers=2, retries=8):
        self.index, self.max_diff, self.workers, self.retries = index, max_diff_tokens, workers, retries
        self.slot_lock, self.next_slot = threading.Lock(), 0.0
        self.cache_path = cache_path
        self.cache = {}
        self.lock = threading.Lock()
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    row = json.loads(line)
                    self.cache[row["key"]] = row["response"]

    def _key(self, query):
        return json.dumps([self.index, query, self.max_diff, MAX_CLAUSE_FREQ])

    def _count_payload(self, query):
        payload = {"index": self.index, "query_type": "count", "query": query}
        if " AND " in query:
            payload.update(max_diff_tokens=self.max_diff, max_clause_freq=MAX_CLAUSE_FREQ)
        return payload

    def _request(self, query):
        return self._post(self._count_payload(query), query)

    def _post(self, payload, label):
        body = json.dumps(payload).encode()
        for attempt in range(self.retries):
            self._wait_for_slot()
            try:
                req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    resp = json.loads(r.read())
                if "error" in resp:
                    raise RuntimeError(f"infini-gram error for {label!r}: {resp['error']}")
                return resp
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                # the API documents transient failures and asks clients to retry
                if attempt == self.retries - 1:
                    raise RuntimeError(f"infini-gram unreachable for {label!r} after {self.retries} tries") from e
                limited = isinstance(e, urllib.error.HTTPError) and e.code in RATE_LIMIT_STATUS
                pause = min(300, 30 * 2 ** attempt) if limited else 2 ** attempt
                print(f"infini-gram {'rate limit' if limited else 'error'} ({e}); retrying in {pause}s", flush=True)
                self._hold(pause)

    def _wait_for_slot(self):
        with self.slot_lock:
            now = time.monotonic()
            start = max(now, self.next_slot)
            self.next_slot = start + MIN_INTERVAL_S
        time.sleep(start - now)

    def _hold(self, seconds):
        # a refusal pauses every worker, not just the one that saw it
        with self.slot_lock:
            self.next_slot = max(self.next_slot, time.monotonic() + seconds)

    def count(self, query):
        key = self._key(query)
        with self.lock:
            if key in self.cache:
                return self.cache[key]["count"]
        resp = self._request(query)
        with self.lock:
            self.cache[key] = resp
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "response": resp}) + "\n")
        return resp["count"]

    def query(self, payload, keep=None):
        """Any API call, cached. keep: response fields to store (documents carry
        kilobytes of crawl metadata that nothing here reads)."""
        payload = {"index": self.index, **payload}
        key = "Q" + json.dumps(payload, sort_keys=True)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        resp = self._post(payload, payload.get("query"))
        if keep is not None:
            resp = {k: resp[k] for k in keep if k in resp}
        with self.lock:
            self.cache[key] = resp
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "response": resp}) + "\n")
        return resp

    def queries(self, payloads, keep=None):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda p: self.query(p, keep), payloads))

    def counts(self, queries):
        """Counts for many queries, fetched in parallel, in input order."""
        todo = sorted({q for q in queries if self._key(q) not in self.cache})
        if todo:
            with ThreadPoolExecutor(self.workers) as ex:
                list(ex.map(self.count, todo))
        return [self.cache[self._key(q)]["count"] for q in queries]


def pair_query(subject, value):
    return f"{clean(subject)} AND {clean(value)}"


def clean(s):
    # AND and OR are operators in the query language; a name containing them
    # would change the query's meaning
    return " ".join(w for w in s.split() if w not in ("AND", "OR")).strip()
