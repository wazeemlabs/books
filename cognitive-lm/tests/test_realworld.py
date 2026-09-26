"""Contracts of the real-model pipeline that do not need the model or the network.

    python -m unittest discover -s tests -t .
"""

import json
import os
import tempfile
import unittest
from collections import Counter

import numpy as np

from experiments.real_popqa import candidates, passes_fp
from realworld import popqa
from realworld.corpus import Corpus, clean, pair_query


class Normalize(unittest.TestCase):
    def test_drops_case_punctuation_articles_and_extra_spaces(self):
        self.assertEqual(popqa.normalize("  The  Beatles, Inc. "), "beatles inc")

    def test_unicode_compatibility_forms_fold(self):
        self.assertEqual(popqa.normalize("ﬁlm"), "film")


class IsCorrect(unittest.TestCase):
    def test_alias_inside_a_longer_answer_counts(self):
        self.assertTrue(popqa.is_correct("He was a politician.", ["politician"]))

    def test_alias_must_match_whole_words(self):
        self.assertFalse(popqa.is_correct("Parisian", ["Paris"]))

    def test_empty_answer_or_alias_never_matches(self):
        self.assertFalse(popqa.is_correct("", ["x"]))
        self.assertFalse(popqa.is_correct("anything", ["", "..."]))


class Sample(unittest.TestCase):
    def test_takes_equal_numbers_from_each_popularity_bin(self):
        qs = [popqa.Question(str(i), f"s{i}", "r", "q", ["a"], int(10 ** (i / 200)), []) for i in range(1000)]
        picked, edges = popqa.sample(qs, 100, seed=0)
        self.assertEqual(len(picked), 100)
        bins = np.clip(np.searchsorted(edges, [np.log10(q.s_pop + 1) for q in picked], side="right") - 1, 0, 4)
        self.assertEqual(Counter(bins.tolist()), {b: 20 for b in range(5)})
        self.assertEqual(len({q.id for q in picked}), 100)

    def test_a_smaller_sample_is_a_subset_of_a_larger_one(self):
        qs = [popqa.Question(str(i), f"s{i}", "r", "q", ["a"], int(10 ** (i / 200)), []) for i in range(1000)]
        small = {q.id for q in popqa.sample(qs, 50, seed=3)[0]}
        large = {q.id for q in popqa.sample(qs, 500, seed=3)[0]}
        self.assertLessEqual(small, large)


class Candidates(unittest.TestCase):
    def test_greedy_first_then_beams_deduplicated_by_normalised_text(self):
        gen = {"greedy": "Paris", "beams": [["paris.", -1.0], ["Lyon", -2.0], ["", -3.0], ["Nice", -4.0]]}
        self.assertEqual(candidates(gen, 10), ["Paris", "Lyon", "Nice"])
        self.assertEqual(candidates(gen, 2), ["Paris", "Lyon"])


class BloomEmulation(unittest.TestCase):
    def test_false_positive_rate_matches_eps(self):
        eps = 0.01
        rate = np.mean([passes_fp(f"k{i}", eps, 0) for i in range(100_000)])
        self.assertAlmostEqual(rate, eps, delta=0.0015)

    def test_is_deterministic_and_salted(self):
        keys = [f"k{i}" for i in range(2000)]
        a = [passes_fp(k, 0.5, 0) for k in keys]
        self.assertEqual(a, [passes_fp(k, 0.5, 0) for k in keys])
        self.assertNotEqual(a, [passes_fp(k, 0.5, 1) for k in keys])


class CorpusCache(unittest.TestCase):
    def test_each_query_hits_the_api_once_and_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cache.jsonl")
            calls = []

            def fake(self, q):
                calls.append(q)
                return {"count": len(q)}

            orig = Corpus._request
            Corpus._request = fake
            try:
                c = Corpus(path)
                self.assertEqual(c.counts(["ab", "abc", "ab"]), [2, 3, 2])
                self.assertEqual(sorted(calls), ["ab", "abc"])
                c2 = Corpus(path)
                self.assertEqual(c2.counts(["abc"]), [3])
                self.assertEqual(len(calls), 2)
            finally:
                Corpus._request = orig
            with open(path) as f:
                self.assertEqual(len([json.loads(l) for l in f]), 2)

    def test_requests_are_spaced_and_a_refusal_holds_every_worker(self):
        import time
        from realworld import corpus as cm
        with tempfile.TemporaryDirectory() as d:
            c = Corpus(os.path.join(d, "c.jsonl"))
            orig = cm.MIN_INTERVAL_S
            cm.MIN_INTERVAL_S = 0.05
            try:
                t = time.monotonic()
                for _ in range(4):
                    c._wait_for_slot()
                self.assertGreaterEqual(time.monotonic() - t, 0.15)
                c._hold(0.2)
                t = time.monotonic()
                c._wait_for_slot()
                self.assertGreaterEqual(time.monotonic() - t, 0.18)
            finally:
                cm.MIN_INTERVAL_S = orig

    def test_query_operators_inside_names_are_removed(self):
        self.assertEqual(clean("Tom AND Jerry OR Spike"), "Tom Jerry Spike")
        self.assertEqual(pair_query("Tom and Jerry", "cartoon"), "Tom and Jerry AND cartoon")


if __name__ == "__main__":
    unittest.main()
