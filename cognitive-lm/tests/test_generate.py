"""Generation with the real model (skipped when the weights are not downloaded).

    python -m unittest tests.test_generate
"""

import os
import unittest

MODEL_DIR = "data/models/OLMo-2-0425-1B"


@unittest.skipUnless(os.path.exists(os.path.join(MODEL_DIR, "config.json")), "OLMo-2 1B weights not downloaded")
class PaddedBatch(unittest.TestCase):
    def test_a_left_padded_prompt_answers_like_it_does_alone(self):
        from realworld.generate import Generator
        g = Generator(MODEL_DIR)
        short, long = "Q: What is the capital of France?\nA:", "Q: In what city was the physicist Albert Einstein born?\nA:"
        alone = g.run([short], n_beams=3)[0]
        batched = g.run([short, long], n_beams=3)[0]  # short is left-padded here
        self.assertEqual(batched["greedy"], alone["greedy"])
        self.assertAlmostEqual(batched["greedy_logprob"], alone["greedy_logprob"], delta=0.05)
        self.assertEqual(batched["greedy"], "Paris")


if __name__ == "__main__":
    unittest.main()
