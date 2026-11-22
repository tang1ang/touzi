import unittest

from fedp0.probability import most_likely, three_class_probabilities


class ProbabilityTest(unittest.TestCase):
    def test_hold_when_no_change(self):
        prob = three_class_probabilities(3.88, 3.88)
        self.assertAlmostEqual(prob["hold"], 1.0)
        self.assertAlmostEqual(prob["hike"], 0.0)
        self.assertAlmostEqual(prob["cut"], 0.0)

    def test_full_hike(self):
        prob = three_class_probabilities(4.13, 3.88)
        self.assertAlmostEqual(prob["hike"], 1.0)

    def test_partial_hike(self):
        prob = three_class_probabilities(4.105454545, 3.88)
        self.assertAlmostEqual(prob["hike"], 0.90181818, places=6)
        self.assertAlmostEqual(prob["hold"], 0.09818182, places=6)
        self.assertAlmostEqual(prob["cut"], 0.0)

    def test_partial_cut(self):
        prob = three_class_probabilities(3.78, 3.88)
        self.assertAlmostEqual(prob["cut"], 0.4)
        self.assertAlmostEqual(prob["hold"], 0.6)

    def test_beyond_one_step(self):
        prob = three_class_probabilities(4.28, 3.88)
        self.assertAlmostEqual(prob["hike"], 1.0)
        self.assertTrue(prob["beyond_one_step"])

    def test_probabilities_sum_to_one(self):
        for delta in (-0.4, -0.1, 0.0, 0.1375, 0.25, 0.53):
            prob = three_class_probabilities(3.88 + delta, 3.88)
            self.assertAlmostEqual(
                prob["cut"] + prob["hold"] + prob["hike"], 1.0, places=9
            )

    def test_most_likely(self):
        self.assertEqual(
            most_likely({"cut": 0.1, "hold": 0.2, "hike": 0.7}), "hike"
        )


if __name__ == "__main__":
    unittest.main()
