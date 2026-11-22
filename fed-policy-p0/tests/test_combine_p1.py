import unittest

from fedp1.combine import (
    build_combination,
    decompose,
    pool,
    validate_weights,
    weights_for,
)

MARKET = {"cut": 0.0, "hold": 0.10, "hike": 0.90}
MACRO = {"cut": 0.0, "hold": 0.25, "hike": 0.75}


class WeightTest(unittest.TestCase):
    def test_validate_weights(self):
        self.assertEqual(validate_weights((0.6, 0.25, 0.15)), (0.6, 0.25, 0.15))
        with self.assertRaises(ValueError):
            validate_weights((0.4, 0.4, 0.2))
        with self.assertRaises(ValueError):
            validate_weights((0.6, 0.3, 0.2))

    def test_weights_for(self):
        for index in range(1, 5):
            weights = weights_for(index, (0.6, 0.25, 0.15))
            self.assertAlmostEqual(sum(weights), 1.0, places=9)
            self.assertGreaterEqual(weights[0], 0.5)
            self.assertGreater(weights[2], 0.0)


class PoolTest(unittest.TestCase):
    def test_market_only_reproduces_market(self):
        result = pool(MARKET, MACRO, 0.0, (1.0, 0.0, 0.0))["probability"]
        for name in ("cut", "hold", "hike"):
            self.assertAlmostEqual(result[name], MARKET[name], places=6)

    def test_macro_pulls_down_hike(self):
        result = pool(MARKET, MACRO, 0.0, (0.6, 0.4, 0.0))["probability"]
        self.assertLess(result["hike"], MARKET["hike"])
        self.assertGreater(result["hold"], MARKET["hold"])

    def test_positive_text_pulls_up_hike(self):
        base = pool(MARKET, MACRO, 0.0, (0.6, 0.25, 0.15))["probability"]
        hawkish = pool(MARKET, MACRO, 0.8, (0.6, 0.25, 0.15))["probability"]
        self.assertGreater(hawkish["hike"], base["hike"])

    def test_decomposition_adds_up(self):
        result = decompose(MARKET, MACRO, 0.45, (0.6, 0.25, 0.15))
        for name in ("cut", "hold", "hike"):
            total = (
                result["contributions_pp"]["macro"][name]
                + result["contributions_pp"]["text"][name]
            )
            self.assertAlmostEqual(
                total, result["contributions_pp"]["total"][name], delta=0.011
            )
            self.assertAlmostEqual(
                result["combined"][name] - result["market"][name],
                result["contributions_pp"]["total"][name] / 100.0,
                delta=2e-4,
            )


class CombinationTest(unittest.TestCase):
    def test_path(self):
        market_path = [
            {
                "index": i,
                "meeting": {"decision_date": f"2027-0{i}-15"},
                "pre_rate": 3.88 + 0.25 * (i - 1),
                "post_rate": 3.88 + 0.25 * i,
                "change_bp": 25.0,
                "cumulative_change_bp": 25.0 * i,
                "probability": MARKET,
            }
            for i in range(1, 5)
        ]
        macro_path = [
            {
                "index": i,
                "meeting": {"decision_date": f"2027-0{i}-15"},
                "pre_rate": 3.88 + 0.20 * (i - 1),
                "post_rate": 3.88 + 0.20 * i,
                "change_bp": 20.0,
                "cumulative_bp": 20.0 * i,
                "probability": MACRO,
            }
            for i in range(1, 5)
        ]
        result = build_combination(market_path, macro_path, 0.3, (0.6, 0.25, 0.15))
        self.assertEqual(len(result["path"]), 4)
        for row in result["path"]:
            probs = row["combined"]["probability"]
            self.assertAlmostEqual(probs["cut"] + probs["hold"] + probs["hike"], 1.0, places=6)
            self.assertEqual(set(row["divergence_pp"]), {"cut", "hold", "hike"})
        self.assertFalse(result["summary"]["review_flag"])


if __name__ == "__main__":
    unittest.main()
