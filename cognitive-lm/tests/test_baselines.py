"""Contracts of the scoring helpers and the no-weights baselines.

    python -m unittest discover -s tests
"""

import unittest

import numpy as np

from cogllm.baselines import ExactDict, KeyFilterStatic
from cogllm.checksum import TIP_OF_TONGUE, UNKNOWN_ENTITY, UNKNOWN_FACT, CheckedRecall
from cogllm.evaluate import score, wilson_upper
from cogllm.world import WorldConfig, build_world


def small_world(seed=0):
    return build_world(WorldConfig(n_first=40, n_last=40, n_entities=500, n_values=48, n_mentions=3000, seed=seed))


class WilsonUpper(unittest.TestCase):
    def test_zero_events_gives_about_three_over_n(self):
        # the "rule of three": no events in n trials bounds the rate near 3/n
        z2 = 1.96 ** 2
        self.assertAlmostEqual(wilson_upper(0, 2000), z2 / (2000 + z2), places=9)

    def test_bound_is_above_the_observed_rate(self):
        for k, n in [(1, 10), (5, 100), (50, 100), (99, 100)]:
            self.assertGreater(wilson_upper(k, n), k / n)

    def test_no_trials_bounds_nothing(self):
        self.assertEqual(wilson_upper(0, 0), 1.0)

    def test_all_events_is_bounded_by_one(self):
        self.assertLessEqual(wilson_upper(100, 100), 1.0)


class ScoreUnseenRates(unittest.TestCase):
    def test_conditional_rate_counts_only_never_seen_queries(self):
        truth = np.array([1, 2, 3, 4])
        counts = np.array([1, 1, 0, 0])
        pred = np.array([1, 9, 3, -1])  # one wrong on seen, one answer to an unseen fact
        r = score(pred, truth, counts)
        self.assertEqual(r["unseen_n"], 2)
        self.assertEqual(r["halluc_on_unseen"], 0.5)
        self.assertEqual(r["halluc_on_seen"], 0.5)
        self.assertGreater(r["halluc_on_unseen_ub95"], 0.5)

    def test_ghost_bound_is_reported_with_the_rate(self):
        r = score(np.array([-1]), np.array([0]), np.array([1]), ghost=np.array([-1, -1, 3, -1]))
        self.assertEqual(r["ghost_hallucination"], 0.25)
        self.assertGreater(r["ghost_hallucination_ub95"], 0.25)


class ExactDictContract(unittest.TestCase):
    def setUp(self):
        self.w = small_world()
        self.vals = self.w.values[self.w.stream[:, 0], self.w.stream[:, 1]]

    def test_answers_every_seen_fact_exactly_and_abstains_otherwise(self):
        d = ExactDict(self.w, self.w.stream, self.vals)
        seen = np.argwhere(self.w.counts > 0)
        unseen = np.argwhere(self.w.counts == 0)
        np.testing.assert_array_equal(d.answer(self.w.names[seen[:, 0]], seen[:, 1]), self.w.values[seen[:, 0], seen[:, 1]])
        self.assertTrue((d.answer(self.w.names[unseen[:, 0]], unseen[:, 1]) == -1).all())

    def test_min_count_abstains_on_facts_read_fewer_times(self):
        d = ExactDict(self.w, self.w.stream, self.vals, min_count=2)
        once = np.argwhere(self.w.counts == 1)
        twice = np.argwhere(self.w.counts >= 2)
        self.assertTrue((d.answer(self.w.names[once[:, 0]], once[:, 1]) == -1).all())
        self.assertTrue((d.answer(self.w.names[twice[:, 0]], twice[:, 1]) >= 0).all())

    def test_majority_value_wins_under_conflicting_mentions(self):
        ment = np.array([[0, 0], [0, 0], [0, 0]])
        d = ExactDict(self.w, ment, np.array([5, 7, 5]))
        self.assertEqual(d.answer(self.w.names[[0]], np.array([0]))[0], 5)

    def test_bits_are_key_plus_value_per_fact(self):
        d = ExactDict(self.w, self.w.stream, self.vals)
        n_facts = int((self.w.counts > 0).sum())
        self.assertEqual(d.bits()["dict"], n_facts * (11 + 2 + 6))  # 1,600 names, 4 relations, 48 values


class KeyFilterStaticContract(unittest.TestCase):
    def test_seen_facts_are_exact_and_unseen_ones_leak_at_the_filter_rate(self):
        w = small_world(1)
        vals = w.values[w.stream[:, 0], w.stream[:, 1]]
        k = KeyFilterStatic(w, w.stream, vals, key_bits_per_item=10.0, seed=1)
        seen = np.argwhere(w.counts > 0)
        unseen = np.argwhere(w.counts == 0)
        np.testing.assert_array_equal(k.answer(w.names[seen[:, 0]], seen[:, 1]), w.values[seen[:, 0], seen[:, 1]])
        leak = (k.answer(w.names[unseen[:, 0]], unseen[:, 1]) >= 0).mean()
        self.assertLess(leak, 0.03)  # 10 bits per key: ~0.8% false positives

    def test_bits_charge_log2_v_per_fact_plus_the_filter(self):
        w = small_world(2)
        k = KeyFilterStatic(w, w.stream, w.values[w.stream[:, 0], w.stream[:, 1]])
        n_facts = int((w.counts > 0).sum())
        self.assertAlmostEqual(k.bits()["static_function_bound"], n_facts * np.log2(48))
        self.assertGreaterEqual(k.bits()["key_filter"], n_facts * 10)


class VerifyFlag(unittest.TestCase):
    def test_without_verification_the_top_candidate_is_answered_whenever_the_key_is_familiar(self):
        w = small_world(3)
        cr = CheckedRecall(w, w.stream, triple_bits=14, seed=3)
        seen = np.argwhere(w.counts > 0)[:200]
        names, rels = w.names[seen[:, 0]], seen[:, 1]
        probs = np.random.default_rng(0).random((len(seen), w.cfg.n_values)).astype(np.float32)
        top = probs.argmax(1)
        unverified = cr.answer(None, names, rels, N=1, probs=probs, verify=False)
        verified = cr.answer(None, names, rels, N=1, probs=probs)
        np.testing.assert_array_equal(unverified, top)
        # verification only ever replaces an answer with "tip of the tongue"
        changed = verified != unverified
        self.assertTrue((verified[changed] == TIP_OF_TONGUE).all())
        self.assertTrue(changed.any())

    def test_unknown_people_and_facts_abstain_either_way(self):
        w = small_world(4)
        cr = CheckedRecall(w, w.stream, seed=4)
        unseen = np.argwhere(w.counts == 0)[:100]
        probs = np.ones((len(unseen), w.cfg.n_values), dtype=np.float32)
        pred = cr.answer(None, w.names[unseen[:, 0]], unseen[:, 1], N=1, probs=probs, verify=False)
        self.assertTrue(np.isin(pred, [UNKNOWN_ENTITY, UNKNOWN_FACT]).mean() > 0.95)


if __name__ == "__main__":
    unittest.main()
